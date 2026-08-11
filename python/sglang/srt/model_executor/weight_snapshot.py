from __future__ import annotations

import threading
import time
from typing import Callable
from uuid import uuid4

import msgspec

from sglang.srt.model_executor.weight_inventory_contracts import (
    WeightInventoryError,
    validate_remote_instance_weight_transfer_lease_timeout,
)


class _SnapshotLease(msgspec.Struct, kw_only=True):
    generation: int
    weight_generation: int
    deadline: float | None
    expired: bool = False


class WeightSnapshotLeaseStatus(msgspec.Struct, frozen=True, kw_only=True):
    lease_id: str
    generation: int
    weight_generation: int
    deadline: float | None
    expired: bool


class WeightSnapshotCoordinator:
    """Serialize in-place updates with address-bearing inventory leases."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        completion_fence: Callable[[], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._completion_fence = completion_fence or (lambda: None)
        self._generation = 1
        self._weight_generation = 1
        self._healthy = True
        self._poisoned = False
        self._needs_weight_generation_commit = False
        self._storage_unavailable = False
        self._last_update_success = True
        self._update_token: str | None = None
        self._update_kind: str | None = None
        self._update_full_restore = False
        self._update_fence_pending = False
        self._pending_full_restore_commit = False
        self._leases: dict[str, _SnapshotLease] = {}

    def _refresh_expired_leases_locked(self) -> None:
        now = self._clock()
        for lease in self._leases.values():
            if lease.deadline is not None and lease.deadline <= now:
                lease.expired = True

    @property
    def generation(self) -> int:
        """Return the ephemeral runtime binding fence generation."""

        with self._lock:
            return self._generation

    @property
    def weight_generation(self) -> int:
        """Return the committed logical weight-content generation."""

        with self._lock:
            return self._weight_generation

    def begin_update(self, *, full_restore: bool = False) -> str:
        if not isinstance(full_restore, bool):
            raise TypeError("full_restore must be a boolean")
        return self._begin_transition(kind="content", full_restore=full_restore)

    def begin_storage_transition(self) -> str:
        """Reserve a content-preserving storage relocation."""

        return self._begin_transition(kind="storage", full_restore=False)

    def _begin_transition(self, *, kind: str, full_restore: bool) -> str:
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None:
                raise WeightInventoryError("a weight update is already in progress")
            if kind == "content":
                if self._storage_unavailable:
                    raise WeightInventoryError("weight storage is unavailable")
                if self._poisoned and not full_restore:
                    raise WeightInventoryError(
                        "the last weight update failed; "
                        "a full successful weight restore is required"
                    )
                if self._needs_weight_generation_commit and not self._poisoned:
                    raise WeightInventoryError(
                        "the previous weight update requires a weight generation commit"
                    )
            elif kind == "storage":
                if self._poisoned or self._needs_weight_generation_commit:
                    raise WeightInventoryError(
                        "content-preserving relocation requires healthy weight content"
                    )
            else:
                raise ValueError(f"unsupported weight transition kind: {kind}")
            if self._leases:
                raise WeightInventoryError("a weight inventory lease is active")
            token = uuid4().hex
            expected_generation = self._generation
            self._update_token = token
            self._update_kind = kind
            self._update_full_restore = full_restore
            self._update_fence_pending = True

        try:
            self._completion_fence()
        except BaseException:
            with self._lock:
                if token == self._update_token:
                    self._update_token = None
                    self._update_kind = None
                    self._update_full_restore = False
                    self._update_fence_pending = False
            raise

        with self._lock:
            if (
                token != self._update_token
                or expected_generation != self._generation
                or not self._update_fence_pending
            ):
                raise WeightInventoryError(
                    "weight update reservation changed during completion fence"
                )
            self._update_fence_pending = False
            return token

    def finish_update(self, token: str, *, success: bool) -> int:
        return self._finish_transition(
            token,
            success=success,
            expected_kind="content",
            storage_available=None,
        )

    def finish_storage_transition(
        self,
        token: str,
        *,
        success: bool,
        storage_available: bool,
    ) -> int:
        """Publish new addresses without changing logical weight content."""

        if not isinstance(storage_available, bool):
            raise TypeError("storage_available must be a boolean")
        return self._finish_transition(
            token,
            success=success,
            expected_kind="storage",
            storage_available=storage_available,
        )

    def _finish_transition(
        self,
        token: str,
        *,
        success: bool,
        expected_kind: str,
        storage_available: bool | None,
    ) -> int:
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        with self._lock:
            if not token or token != self._update_token:
                raise WeightInventoryError("weight update token does not match")
            if self._update_kind != expected_kind:
                raise WeightInventoryError(
                    "weight update transition kind does not match"
                )
            if self._update_fence_pending:
                raise WeightInventoryError(
                    "weight update completion fence is in progress"
                )
            self._update_fence_pending = True

        try:
            self._completion_fence()
        except BaseException:
            with self._lock:
                self._publish_failed_transition_locked()
            raise

        with self._lock:
            if not success:
                self._publish_failed_transition_locked()
            elif expected_kind == "content":
                self._publish_content_update_locked()
            else:
                assert storage_available is not None
                self._publish_storage_transition_locked(
                    storage_available=storage_available
                )
            return self._generation

    def _finish_transition_state_locked(self) -> None:
        self._update_token = None
        self._update_kind = None
        self._update_full_restore = False
        self._update_fence_pending = False

    def _publish_content_update_locked(self) -> None:
        self._generation += 1
        self._healthy = False
        self._storage_unavailable = False
        self._needs_weight_generation_commit = True
        self._last_update_success = True
        self._pending_full_restore_commit = self._update_full_restore
        self._finish_transition_state_locked()

    def _publish_storage_transition_locked(self, *, storage_available: bool) -> None:
        self._generation += 1
        self._healthy = storage_available
        self._storage_unavailable = not storage_available
        self._last_update_success = True
        self._finish_transition_state_locked()

    def _publish_failed_transition_locked(self) -> None:
        self._generation += 1
        self._healthy = False
        self._poisoned = True
        self._storage_unavailable = False
        self._needs_weight_generation_commit = True
        self._last_update_success = False
        self._pending_full_restore_commit = False
        self._finish_transition_state_locked()

    def cancel_update(self, token: str) -> None:
        """Cancel a reservation before any local weight mutation starts."""
        with self._lock:
            if not token or token != self._update_token:
                raise WeightInventoryError("weight update token does not match")
            if self._update_fence_pending:
                raise WeightInventoryError(
                    "weight update completion fence is in progress"
                )
            self._update_token = None
            self._update_kind = None
            self._update_full_restore = False

    def pending_weight_generation_commit(self) -> int | None:
        with self._lock:
            if not self._needs_weight_generation_commit:
                return None
            return self._generation

    def poison_global_update_failure(self, *, expected_generation: int) -> None:
        """Fail closed after an upper-layer cross-rank update transaction fails."""
        if isinstance(expected_generation, bool) or not isinstance(
            expected_generation, int
        ):
            raise TypeError("expected_generation must be an integer")
        if expected_generation <= 0:
            raise ValueError("expected_generation must be positive")
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None:
                raise WeightInventoryError("a weight update is in progress")
            if expected_generation != self._generation:
                raise WeightInventoryError("weight update generation does not match")
            if self._leases:
                raise WeightInventoryError("a weight inventory lease is active")
            self._healthy = False
            self._poisoned = True
            self._storage_unavailable = False
            self._needs_weight_generation_commit = True
            self._last_update_success = False
            self._pending_full_restore_commit = False

    def commit_weight_generation(
        self, *, expected_generation: int | None = None
    ) -> int:
        if expected_generation is not None:
            if isinstance(expected_generation, bool) or not isinstance(
                expected_generation, int
            ):
                raise TypeError("expected_generation must be an integer")
            if expected_generation <= 0:
                raise ValueError("expected_generation must be positive")
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None:
                raise WeightInventoryError("a weight update is in progress")
            if (
                expected_generation is not None
                and expected_generation != self._generation
            ):
                raise WeightInventoryError("weight update generation does not match")
            if self._leases:
                raise WeightInventoryError("a weight inventory lease is active")
            if self._storage_unavailable:
                raise WeightInventoryError("weight storage is unavailable")
            if not self._needs_weight_generation_commit:
                return self._generation
            if not self._last_update_success:
                raise WeightInventoryError(
                    "the last weight update failed; "
                    "a full successful weight restore is required"
                )
            if self._poisoned and not self._pending_full_restore_commit:
                raise WeightInventoryError(
                    "the last weight update failed; "
                    "a full successful weight restore is required"
                )
            if self._pending_full_restore_commit:
                self._poisoned = False
            self._weight_generation += 1
            self._healthy = True
            self._needs_weight_generation_commit = False
            self._pending_full_restore_commit = False
            return self._generation

    def adopt_weight_generation(self, weight_generation: int) -> int:
        """Activate already-transferred target content after terminal completion."""

        if type(weight_generation) is not int or weight_generation <= 0:
            raise ValueError("weight_generation must be a positive integer")
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None or self._leases:
                raise WeightInventoryError(
                    "cannot activate target content during an update or active lease"
                )
            if (
                self._poisoned
                or self._needs_weight_generation_commit
                or not self._healthy
            ):
                raise WeightInventoryError(
                    "cannot activate target content on an unhealthy coordinator"
                )
            self._generation += 1
            self._weight_generation = weight_generation
            self._storage_unavailable = False
            return self._generation

    def adopt_weight_generation_from_snapshot(
        self,
        lease_id: str,
        weight_generation: int,
    ) -> int:
        """Atomically release a target binding lease and activate its content."""

        if type(lease_id) is not str or not lease_id:
            raise ValueError("lease_id must not be empty")
        if type(weight_generation) is not int or weight_generation <= 0:
            raise ValueError("weight_generation must be a positive integer")
        with self._lock:
            self._refresh_expired_leases_locked()
            lease = self._leases.get(lease_id)
            if lease is None:
                raise WeightInventoryError("weight inventory lease does not exist")
            if lease.expired:
                raise WeightInventoryError("target weight inventory lease expired")
            if len(self._leases) != 1:
                raise WeightInventoryError(
                    "cannot activate target content while other active leases exist"
                )
            if self._update_token is not None:
                raise WeightInventoryError(
                    "cannot activate target content during a weight update"
                )
            if (
                self._poisoned
                or self._needs_weight_generation_commit
                or not self._healthy
            ):
                raise WeightInventoryError(
                    "cannot activate target content on an unhealthy coordinator"
                )
            del self._leases[lease_id]
            self._generation += 1
            self._weight_generation = weight_generation
            self._storage_unavailable = False
            return self._generation

    def acquire_snapshot(
        self, *, lease_timeout_sec: int | None = None
    ) -> tuple[str, int]:
        if lease_timeout_sec is not None:
            validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None:
                raise WeightInventoryError("a weight update is in progress")
            if self._storage_unavailable:
                raise WeightInventoryError("weight storage is unavailable")
            if self._poisoned:
                if self._pending_full_restore_commit and self._last_update_success:
                    raise WeightInventoryError(
                        "successful full weight restore requires "
                        "an explicit weight generation commit"
                    )
                raise WeightInventoryError(
                    "the last weight update failed; "
                    "a full successful weight restore is required"
                )
            if not self._healthy:
                if self._needs_weight_generation_commit and self._last_update_success:
                    raise WeightInventoryError(
                        "updated weights require an explicit weight generation commit"
                    )
                raise WeightInventoryError(
                    "the last weight update failed; "
                    "a full successful weight restore is required"
                )
            lease_id = uuid4().hex
            deadline = (
                None if lease_timeout_sec is None else self._clock() + lease_timeout_sec
            )
            self._leases[lease_id] = _SnapshotLease(
                generation=self._generation,
                weight_generation=self._weight_generation,
                deadline=deadline,
            )
            return lease_id, self._generation

    def renew_snapshot(self, lease_id: str, *, lease_timeout_sec: int) -> None:
        validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        with self._lock:
            self._refresh_expired_leases_locked()
            lease = self._leases.get(lease_id)
            if lease is None:
                raise WeightInventoryError("weight inventory lease does not exist")
            if lease.expired:
                raise WeightInventoryError(
                    "weight inventory lease expired and requires explicit release"
                )
            lease.deadline = self._clock() + lease_timeout_sec

    def has_snapshot(self, lease_id: str) -> bool:
        with self._lock:
            self._refresh_expired_leases_locked()
            return lease_id in self._leases

    def snapshot_is_active(self, lease_id: str) -> bool:
        """Return whether a retained mutation fence is still usable by a transfer."""

        with self._lock:
            self._refresh_expired_leases_locked()
            lease = self._leases.get(lease_id)
            return lease is not None and not lease.expired

    def list_snapshot_leases(self) -> tuple[WeightSnapshotLeaseStatus, ...]:
        with self._lock:
            self._refresh_expired_leases_locked()
            return tuple(
                WeightSnapshotLeaseStatus(
                    lease_id=lease_id,
                    generation=lease.generation,
                    weight_generation=lease.weight_generation,
                    deadline=lease.deadline,
                    expired=lease.expired,
                )
                for lease_id, lease in sorted(self._leases.items())
            )

    def release_snapshot(self, lease_id: str) -> None:
        with self._lock:
            self._refresh_expired_leases_locked()
            if lease_id not in self._leases:
                raise WeightInventoryError("weight inventory lease does not exist")
            del self._leases[lease_id]

    def invalidate(self) -> None:
        token = self.begin_update()
        generation = self.finish_update(token, success=True)
        self.commit_weight_generation(expected_generation=generation)

    def poison_uncoordinated_mutation(self, lease_id: str) -> None:
        with self._lock:
            self._refresh_expired_leases_locked()
            if lease_id not in self._leases:
                raise WeightInventoryError("weight inventory lease does not exist")
            del self._leases[lease_id]
            self._generation += 1
            self._healthy = False
            self._poisoned = True
            self._storage_unavailable = False
            self._needs_weight_generation_commit = True
            self._last_update_success = False
            self._pending_full_restore_commit = False


__all__ = [
    "WeightSnapshotCoordinator",
    "WeightSnapshotLeaseStatus",
]
