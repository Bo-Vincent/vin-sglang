from __future__ import annotations

import hashlib
import logging
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    BeginRemoteInstanceWeightTransferReqInput,
    BeginRemoteInstanceWeightTransferReqOutput,
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ReleaseRemoteInstanceWeightTransferReqInput,
    ReleaseRemoteInstanceWeightTransferReqOutput,
    RenewRemoteInstanceWeightTransferReqInput,
    RenewRemoteInstanceWeightTransferReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.model_executor.model_runner_components.weight_update_coordination import (
    use_pre_reserved_weight_updates,
)
from sglang.srt.model_executor.weight_inventory_contracts import (
    WeightInventoryError,
    WeightPlacementBindingInventories,
    WeightPlacementInventory,
    WeightRuntimeBindingInventory,
    validate_remote_weight_lineage,
)

logger = logging.getLogger(__name__)

_REMOTE_WEIGHT_TRANSFER_TOMBSTONE_TTL_SEC = 300.0
_REMOTE_WEIGHT_TRANSFER_TOMBSTONE_LIMIT = 4096


class _RemoteWeightTransferSessionError(RuntimeError):
    def __init__(self, message: str, *, session_state: str) -> None:
        super().__init__(message)
        self.session_state = session_state


def _get_draft_model_runner(draft_worker):
    # DFlash / FrozenKVMTP workers expose draft_model_runner directly
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    # EAGLEWorkerV2: _draft_worker.draft_runner
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None:
        runner = getattr(inner, "draft_runner", None)
        if runner is not None:
            return runner
    return None


def _merge_checksum_payloads(target: Dict, draft: Dict) -> Dict:
    merged_checksums = dict(target["checksums"])
    for name, chk in draft["checksums"].items():
        merged_checksums[f"draft.{name}"] = chk
    h = hashlib.sha256()
    for name in sorted(merged_checksums):
        h.update(name.encode())
        h.update(merged_checksums[name].encode())
    target["checksums"] = merged_checksums
    target["per_gpu_checksum"] = h.hexdigest()
    return target


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    world_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    remote_weight_transfer_cpu_group: Any = None
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    remote_weight_transfer_leases: Dict[str, str] = field(default_factory=dict)
    remote_weight_transfer_deadlines: Dict[str, float] = field(default_factory=dict)
    remote_weight_transfer_generations: Dict[str, int] = field(default_factory=dict)
    remote_weight_transfer_expired: set[str] = field(default_factory=set)
    remote_weight_transfer_sessions: Dict[
        str,
        Tuple[Tuple[Any, ...], BeginRemoteInstanceWeightTransferReqOutput],
    ] = field(default_factory=dict)
    remote_weight_transfer_tombstones: Dict[
        str,
        Tuple[Optional[Tuple[Any, ...]], float],
    ] = field(default_factory=dict)
    remote_weight_transfer_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    remote_weight_transfer_executor: Optional[ThreadPoolExecutor] = field(
        default=None, init=False, repr=False
    )
    remote_weight_transfer_pending: List[Tuple[Future, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    @contextmanager
    def _coordinate_weight_memory_transition(
        self,
        *,
        enabled: bool,
        storage_available_after: bool,
    ) -> Iterator[None]:
        coordinator = getattr(
            self.tp_worker.model_runner,
            "weight_snapshot_coordinator",
            None,
        )
        if not enabled or coordinator is None:
            yield
            return

        token = None
        local_error = None
        try:
            token = coordinator.begin_storage_transition()
        except Exception as error:
            local_error = str(error)

        try:
            world_size = torch.distributed.get_world_size(group=self.world_cpu_group)
            gathered_errors = [None] * world_size
            torch.distributed.all_gather_object(
                gathered_errors,
                local_error,
                group=self.world_cpu_group,
            )
        except Exception as error:
            if token is not None:
                coordinator.cancel_update(token)
            raise WeightInventoryError(
                f"failed to coordinate weight memory transition: {error}"
            ) from error

        failures = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(gathered_errors)
            if error is not None
        ]
        if failures:
            if token is not None:
                coordinator.cancel_update(token)
            raise WeightInventoryError(
                "weight memory transition reservation failed: " + " | ".join(failures)
            )

        assert token is not None
        local_exception = None
        local_traceback = None
        try:
            yield
        except Exception as error:
            local_exception = error
            local_traceback = error.__traceback__

        local_error = None if local_exception is None else str(local_exception)
        try:
            gathered_errors = [None] * world_size
            torch.distributed.all_gather_object(
                gathered_errors,
                local_error,
                group=self.world_cpu_group,
            )
        except Exception as error:
            coordinator.finish_storage_transition(
                token,
                success=False,
                storage_available=False,
            )
            raise WeightInventoryError(
                f"failed to publish weight memory transition outcome: {error}"
            ) from error

        failures = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(gathered_errors)
            if error is not None
        ]
        success = not failures
        coordinator.finish_storage_transition(
            token,
            success=success,
            storage_available=success and storage_available_after,
        )
        if local_exception is not None:
            raise local_exception.with_traceback(local_traceback)
        if failures:
            raise WeightInventoryError(
                "weight memory transition failed: " + " | ".join(failures)
            )

    def _prune_remote_weight_transfer_bookkeeping(self) -> None:
        now = time.monotonic()
        with self.remote_weight_transfer_lock:
            expired = [
                transfer_id
                for transfer_id, deadline in self.remote_weight_transfer_deadlines.items()
                if deadline <= now
            ]
            for transfer_id in expired:
                self.remote_weight_transfer_expired.add(transfer_id)
            expired_tombstones = [
                transfer_id
                for transfer_id, (_, deadline) in (
                    self.remote_weight_transfer_tombstones.items()
                )
                if deadline <= now
            ]
            for transfer_id in expired_tombstones:
                self.remote_weight_transfer_tombstones.pop(transfer_id, None)

    def _get_remote_weight_transfer_lease(self, transfer_id: str) -> str | None:
        self._prune_remote_weight_transfer_bookkeeping()
        with self.remote_weight_transfer_lock:
            return self.remote_weight_transfer_leases.get(transfer_id)

    def list_remote_instance_weight_transfer_sessions(self) -> List[Dict[str, Any]]:
        self._prune_remote_weight_transfer_bookkeeping()
        with self.remote_weight_transfer_lock:
            return [
                {
                    "transfer_id": transfer_id,
                    "lease_id": lease_id,
                    "generation": self.remote_weight_transfer_generations.get(
                        transfer_id
                    ),
                    "deadline_monotonic_sec": (
                        self.remote_weight_transfer_deadlines.get(transfer_id)
                    ),
                    "expired": transfer_id in self.remote_weight_transfer_expired,
                    "session_state": (
                        "expired"
                        if transfer_id in self.remote_weight_transfer_expired
                        else (
                            "active"
                            if transfer_id in self.remote_weight_transfer_sessions
                            else "cleanup_pending"
                        )
                    ),
                }
                for transfer_id, lease_id in sorted(
                    self.remote_weight_transfer_leases.items()
                )
            ]

    @staticmethod
    def _remote_weight_transfer_request_identity(
        request: BeginRemoteInstanceWeightTransferReqInput,
    ) -> Tuple[Any, ...]:
        return (
            request.model_id,
            request.revision,
            request.lease_timeout_sec,
        )

    def _cached_remote_weight_transfer_session(
        self,
        request: BeginRemoteInstanceWeightTransferReqInput,
    ) -> BeginRemoteInstanceWeightTransferReqOutput | None:
        self._prune_remote_weight_transfer_bookkeeping()
        with self.remote_weight_transfer_lock:
            cached = self.remote_weight_transfer_sessions.get(request.transfer_id)
            expired = request.transfer_id in self.remote_weight_transfer_expired
            tombstone = self.remote_weight_transfer_tombstones.get(request.transfer_id)
        request_identity = self._remote_weight_transfer_request_identity(request)
        if tombstone is not None:
            terminal_identity, _ = tombstone
            if terminal_identity is not None and terminal_identity != request_identity:
                raise _RemoteWeightTransferSessionError(
                    "remote weight transfer ID was reused with different parameters",
                    session_state="conflict",
                )
            raise _RemoteWeightTransferSessionError(
                "remote weight transfer was already released",
                session_state="released",
            )
        if cached is None:
            return None
        identity, output = cached
        if identity != request_identity:
            raise _RemoteWeightTransferSessionError(
                "remote weight transfer ID was reused with different parameters",
                session_state="conflict",
            )
        if expired:
            raise _RemoteWeightTransferSessionError(
                "remote weight transfer expired and requires explicit release",
                session_state="expired",
            )
        return output

    def _record_remote_weight_transfer_session(
        self,
        request: BeginRemoteInstanceWeightTransferReqInput,
        lease_id: str,
        output: BeginRemoteInstanceWeightTransferReqOutput,
        generation: int | None = None,
    ) -> None:
        if generation is None:
            generation = self._remote_transfer_output_generation(output)
        with self.remote_weight_transfer_lock:
            self.remote_weight_transfer_leases[request.transfer_id] = lease_id
            self.remote_weight_transfer_deadlines.setdefault(
                request.transfer_id,
                time.monotonic() + request.lease_timeout_sec,
            )
            if generation is not None:
                self.remote_weight_transfer_generations[request.transfer_id] = (
                    generation
                )
            self.remote_weight_transfer_expired.discard(request.transfer_id)
            self.remote_weight_transfer_sessions[request.transfer_id] = (
                self._remote_weight_transfer_request_identity(request),
                output,
            )
            self.remote_weight_transfer_tombstones.pop(request.transfer_id, None)

    def _record_remote_weight_transfer_lease(
        self,
        transfer_id: str,
        lease_id: str,
        lease_timeout_sec: int,
        *,
        generation: int | None = None,
    ) -> None:
        with self.remote_weight_transfer_lock:
            self.remote_weight_transfer_leases[transfer_id] = lease_id
            self.remote_weight_transfer_deadlines[transfer_id] = (
                time.monotonic() + lease_timeout_sec
            )
            if generation is not None:
                self.remote_weight_transfer_generations[transfer_id] = generation
            self.remote_weight_transfer_expired.discard(transfer_id)

    def _complete_remote_weight_transfer_session(self, transfer_id: str) -> None:
        now = time.monotonic()
        with self.remote_weight_transfer_lock:
            self.remote_weight_transfer_leases.pop(transfer_id, None)
            self.remote_weight_transfer_deadlines.pop(transfer_id, None)
            self.remote_weight_transfer_generations.pop(transfer_id, None)
            self.remote_weight_transfer_expired.discard(transfer_id)
            session = self.remote_weight_transfer_sessions.pop(transfer_id, None)
            if (
                session is None
                and transfer_id in self.remote_weight_transfer_tombstones
            ):
                return
            identity = session[0] if session is not None else None
            self.remote_weight_transfer_tombstones[transfer_id] = (
                identity,
                now + _REMOTE_WEIGHT_TRANSFER_TOMBSTONE_TTL_SEC,
            )
            while (
                len(self.remote_weight_transfer_tombstones)
                > _REMOTE_WEIGHT_TRANSFER_TOMBSTONE_LIMIT
            ):
                oldest_transfer_id = next(iter(self.remote_weight_transfer_tombstones))
                self.remote_weight_transfer_tombstones.pop(oldest_transfer_id)

    def _discard_remote_weight_transfer_lease(self, transfer_id: str) -> None:
        with self.remote_weight_transfer_lock:
            self.remote_weight_transfer_leases.pop(transfer_id, None)
            self.remote_weight_transfer_deadlines.pop(transfer_id, None)
            self.remote_weight_transfer_generations.pop(transfer_id, None)
            self.remote_weight_transfer_expired.discard(transfer_id)
            self.remote_weight_transfer_sessions.pop(transfer_id, None)

    def _rollback_remote_weight_transfer_snapshot(
        self,
        transfer_id: str,
        lease_id: str,
    ) -> str | None:
        try:
            self.tp_worker.model_runner.release_weight_inventory(lease_id)
        except Exception as error:
            return str(error)
        self._discard_remote_weight_transfer_lease(transfer_id)
        return None

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    @staticmethod
    def _weight_snapshot_coordinators_for_worker(worker) -> tuple:
        if worker is None:
            return ()

        runners = []
        direct_runner = getattr(worker, "model_runner", None)
        if direct_runner is not None:
            runners.append(direct_runner)

        draft_runner = _get_draft_model_runner(worker)
        if draft_runner is not None:
            runners.append(draft_runner)

        target_worker = getattr(worker, "target_worker", None)
        target_runner = getattr(target_worker, "model_runner", None)
        if target_runner is not None:
            runners.append(target_runner)

        coordinators = []
        seen = set()
        for runner in runners:
            coordinator = getattr(runner, "weight_snapshot_coordinator", None)
            if coordinator is None or id(coordinator) in seen:
                continue
            coordinators.append(coordinator)
            seen.add(id(coordinator))
        return tuple(coordinators)

    def _weight_update_coordinators(self, workers) -> tuple:
        coordinators = []
        seen = set()
        for worker in workers:
            for coordinator in self._weight_snapshot_coordinators_for_worker(worker):
                if id(coordinator) in seen:
                    continue
                coordinators.append(coordinator)
                seen.add(id(coordinator))
        return tuple(coordinators)

    def _capture_weight_update_generations(self, workers) -> tuple:
        return tuple(
            (coordinator, coordinator.generation)
            for coordinator in self._weight_update_coordinators(workers)
        )

    def _weight_update_transaction_required(self, workers) -> bool:
        reshard_enabled = bool(
            getattr(
                getattr(self.scheduler, "server_args", None),
                "enable_weight_reshard",
                False,
            )
        )
        return reshard_enabled or bool(self._weight_update_coordinators(workers))

    @staticmethod
    def _validate_weight_update_generations(
        generation_mapping,
        *,
        require_pending_weight_generation: bool,
    ) -> None:
        failures = []
        for coordinator, expected_generation in generation_mapping:
            current_generation = coordinator.generation
            if current_generation != expected_generation:
                failures.append(
                    "weight update generation changed from "
                    f"{expected_generation} to {current_generation}"
                )
                continue
            if require_pending_weight_generation:
                pending_generation = coordinator.pending_weight_generation_commit()
                if pending_generation != expected_generation:
                    failures.append(
                        "weight update generation "
                        f"{expected_generation} is not pending a weight generation commit"
                    )
        if failures:
            raise WeightInventoryError(" | ".join(failures))

    @staticmethod
    def _finalize_weight_update(generation_mapping, *, commit: bool) -> None:
        failures = []
        for coordinator, expected_generation in generation_mapping:
            try:
                if commit:
                    coordinator.commit_weight_generation(
                        expected_generation=expected_generation
                    )
                else:
                    coordinator.poison_global_update_failure(
                        expected_generation=expected_generation
                    )
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
        if failures:
            raise WeightInventoryError(" | ".join(failures))

    @staticmethod
    def _poison_weight_update_best_effort(generation_mapping) -> None:
        for coordinator, expected_generation in generation_mapping:
            try:
                coordinator.poison_global_update_failure(
                    expected_generation=expected_generation
                )
            except Exception:
                logger.exception("Failed to poison a partial weight update")

    def _gather_weight_update_outcome(
        self,
        *,
        success: bool,
        message: str,
        phase: str,
    ) -> Tuple[bool, str]:
        local_outcome = {"success": bool(success), "message": str(message)}
        world_size = torch.distributed.get_world_size(group=self.world_cpu_group)
        gathered = [None] * world_size
        torch.distributed.all_gather_object(
            gathered,
            local_outcome,
            group=self.world_cpu_group,
        )

        failures = []
        for rank, outcome in enumerate(gathered):
            if (
                not isinstance(outcome, dict)
                or not isinstance(outcome.get("success"), bool)
                or not isinstance(outcome.get("message"), str)
            ):
                failures.append(f"rank {rank}: invalid {phase} outcome")
            elif not outcome["success"]:
                failures.append(f"rank {rank}: {outcome['message']}")
        if failures:
            return False, " | ".join(failures)
        return True, "Success."

    def _gather_weight_mutation_outcome(
        self,
        *,
        success: bool,
        message: str,
        mutated: bool,
        phase: str,
    ) -> Tuple[bool, str, bool]:
        local_outcome = {
            "success": bool(success),
            "message": str(message),
            "mutated": bool(mutated),
        }
        world_size = torch.distributed.get_world_size(group=self.world_cpu_group)
        gathered = [None] * world_size
        torch.distributed.all_gather_object(
            gathered,
            local_outcome,
            group=self.world_cpu_group,
        )

        failures = []
        any_mutated = False
        for rank, outcome in enumerate(gathered):
            if (
                not isinstance(outcome, dict)
                or not isinstance(outcome.get("success"), bool)
                or not isinstance(outcome.get("message"), str)
                or not isinstance(outcome.get("mutated"), bool)
            ):
                failures.append(f"rank {rank}: invalid {phase} outcome")
                continue
            any_mutated = any_mutated or outcome["mutated"]
            if not outcome["success"]:
                failures.append(f"rank {rank}: {outcome['message']}")
        if failures:
            return False, " | ".join(failures), any_mutated
        return True, "Success.", any_mutated

    def _run_weight_update_transaction(
        self,
        *,
        operation: str,
        mutate: Callable[[], Tuple[bool, str]],
        workers,
        recv_req,
    ) -> Tuple[bool, str]:
        coordinators = self._weight_update_coordinators(workers)
        reservations = []
        pre_capture_succeeded = False
        before_generation_mapping = ()
        coordinate_snapshots = self._weight_update_transaction_required(workers)
        if not coordinate_snapshots:
            raise AssertionError(
                "ordinary weight updates must bypass the coordinated transaction"
            )
        reservation_error = None
        if coordinate_snapshots:
            try:
                before_generation_mapping = self._capture_weight_update_generations(
                    workers
                )
            except Exception as error:
                reservation_error = (
                    "failed to capture pre-mutation weight generation: "
                    f"{type(error).__name__}: {error}"
                )
            else:
                pre_capture_succeeded = True
                if not coordinators:
                    reservation_error = (
                        "weight reshard is enabled but no snapshot coordinator "
                        "is installed"
                    )

            if reservation_error is None:
                try:
                    for coordinator in coordinators:
                        reservations.append((coordinator, coordinator.begin_update()))
                except Exception as error:
                    reservation_error = f"{type(error).__name__}: {error}"

            try:
                reservation_success, reservation_message = (
                    self._gather_weight_update_outcome(
                        success=reservation_error is None,
                        message=reservation_error or "Success.",
                        phase=f"{operation} reservation",
                    )
                )
            except Exception as error:
                self._cancel_weight_update_reservations(reservations)
                return (
                    False,
                    f"Failed to gather {operation} reservation outcomes: "
                    f"{type(error).__name__}: {error}",
                )

            if not reservation_success:
                self._cancel_weight_update_reservations(reservations)
                return False, reservation_message

        local_success = None
        local_message = ""

        with use_pre_reserved_weight_updates(
            {id(coordinator): token for coordinator, token in reservations}
        ):
            try:
                if local_success is None:
                    local_success, local_message = mutate()
                    if not isinstance(local_success, bool):
                        raise TypeError(
                            "weight update success outcome must be a boolean"
                        )
                    local_message = str(local_message)
            except Exception as error:
                local_success = False
                local_message = f"{type(error).__name__}: {error}"

        finish_errors = []
        for coordinator, token in reservations:
            try:
                coordinator.finish_update(token, success=local_success is True)
            except Exception as error:
                finish_errors.append(f"{type(error).__name__}: {error}")
        if finish_errors:
            local_success = False
            local_message = (
                f"{local_message} | failed to finish local reservation: "
                + " | ".join(finish_errors)
            )

        try:
            generation_mapping = self._capture_weight_update_generations(workers)
        except Exception as error:
            generation_mapping = ()
            local_success = False
            local_message = (
                f"{local_message} | failed to capture weight update generation: "
                f"{type(error).__name__}: {error}"
            )

        before_by_coordinator = {
            id(coordinator): generation
            for coordinator, generation in before_generation_mapping
        }
        local_mutated = pre_capture_succeeded and (
            any(
                before_by_coordinator.get(id(coordinator)) != generation
                for coordinator, generation in generation_mapping
            )
            or len(before_generation_mapping) != len(generation_mapping)
        )

        if local_success:
            try:
                self.flush_cache_after_weight_update(recv_req)
            except Exception as error:
                local_success = False
                local_message = f"{type(error).__name__}: {error}"

        try:
            mutation_success, mutation_message, any_mutated = (
                self._gather_weight_mutation_outcome(
                    success=local_success,
                    message=local_message,
                    mutated=local_mutated,
                    phase=f"{operation} mutation",
                )
            )
        except Exception as error:
            self._poison_weight_update_best_effort(generation_mapping)
            message = (
                f"Failed to gather {operation} mutation outcomes: "
                f"{type(error).__name__}: {error}"
            )
            self._raise_if_unrecoverable_mutation(
                generation_mapping,
                message=message,
            )
            return False, message

        try:
            self._validate_weight_update_generations(
                generation_mapping,
                require_pending_weight_generation=mutation_success,
            )
            local_ready_success = True
            local_ready_message = "Success."
        except Exception as error:
            local_ready_success = False
            local_ready_message = f"{type(error).__name__}: {error}"

        try:
            ready_success, ready_message = self._gather_weight_update_outcome(
                success=local_ready_success,
                message=local_ready_message,
                phase=f"{operation} finalize readiness",
            )
        except Exception as error:
            self._poison_weight_update_best_effort(generation_mapping)
            message = (
                f"Failed to gather {operation} finalize readiness outcomes: "
                f"{type(error).__name__}: {error}"
            )
            self._raise_if_unrecoverable_mutation(
                generation_mapping,
                message=message,
            )
            return False, message

        try:
            if any_mutated:
                self._finalize_weight_update(
                    generation_mapping,
                    commit=mutation_success and ready_success,
                )
            local_finalize_success = True
            local_finalize_message = "Success."
        except Exception as error:
            local_finalize_success = False
            local_finalize_message = f"{type(error).__name__}: {error}"

        try:
            finalize_success, finalize_message = self._gather_weight_update_outcome(
                success=local_finalize_success,
                message=local_finalize_message,
                phase=f"{operation} finalize",
            )
        except Exception as error:
            self._poison_weight_update_best_effort(generation_mapping)
            message = (
                f"Failed to gather {operation} finalize outcomes: "
                f"{type(error).__name__}: {error}"
            )
            self._raise_if_unrecoverable_mutation(
                generation_mapping,
                message=message,
            )
            return False, message

        if not finalize_success:
            self._poison_weight_update_best_effort(generation_mapping)
        if any_mutated and not (
            mutation_success and ready_success and finalize_success
        ):
            self._raise_if_unrecoverable_mutation(
                generation_mapping,
                message=" | ".join(
                    item
                    for item in (
                        mutation_message if not mutation_success else "",
                        ready_message if not ready_success else "",
                        finalize_message if not finalize_success else "",
                    )
                    if item
                ),
            )
        if not mutation_success:
            return False, mutation_message
        if not ready_success:
            return False, ready_message
        if not finalize_success:
            return False, finalize_message
        return True, local_message

    @staticmethod
    def _cancel_weight_update_reservations(reservations) -> None:
        for coordinator, token in reversed(reservations):
            try:
                coordinator.cancel_update(token)
            except Exception:
                logger.exception("Failed to cancel a weight update reservation")

    def _raise_if_unrecoverable_mutation(
        self,
        generation_mapping,
        *,
        message: str,
    ) -> None:
        if not generation_mapping:
            return
        raise WeightInventoryError(
            "A weight update may have left model content unverified; "
            f"the model world must restart before serving. {message}"
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            workers = [self.tp_worker]
            if self.draft_worker is not None:
                workers.append(self.draft_worker)

            coordinated = self._weight_update_transaction_required(workers)
            if self._weight_update_coordinators(workers):
                try:
                    validate_remote_weight_lineage(
                        model_id=recv_req.model_path,
                        revision=recv_req.revision,
                    )
                except ValueError as error:
                    return UpdateWeightFromDiskReqOutput(
                        success=False,
                        message=str(error),
                        num_paused_requests=0,
                    )

            if not coordinated:
                success, message = self.tp_worker.update_weights_from_disk(recv_req)
                tp_success = success
                if success and self.draft_worker is not None:
                    success, message = self.draft_worker.update_weights_from_disk(
                        recv_req
                    )
                if tp_success:
                    self.flush_cache_after_weight_update(recv_req)
                if not success:
                    logger.error(message)
                return UpdateWeightFromDiskReqOutput(
                    success=success,
                    message=message,
                    num_paused_requests=0,
                )

            def mutate():
                success, message = self.tp_worker.update_weights_from_disk(recv_req)
                if success and self.draft_worker is not None:
                    success, message = self.draft_worker.update_weights_from_disk(
                        recv_req
                    )
                return success, message

            success, message = self._run_weight_update_transaction(
                operation="disk weight update",
                mutate=mutate,
                workers=workers,
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
            )

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success=success, message=message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success=success, message=message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        with self._observe_weight_load("distributed"):
            workers = (self.tp_worker,)
            if not self._weight_update_transaction_required(workers):
                success, message = self.tp_worker.update_weights_from_distributed(
                    recv_req
                )
                if success:
                    self.flush_cache_after_weight_update(recv_req)
                else:
                    logger.error(message)
                return UpdateWeightsFromDistributedReqOutput(
                    success=success,
                    message=message,
                )

            success, message = self._run_weight_update_transaction(
                operation="distributed weight update",
                mutate=lambda: self.tp_worker.update_weights_from_distributed(recv_req),
                workers=workers,
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        with self._observe_weight_load("tensor"):
            if recv_req.disable_draft_model:
                worker = self.tp_worker
            else:
                worker = self.draft_worker or self.tp_worker
            workers = (worker,)
            if not self._weight_update_transaction_required(workers):
                success, message = worker.update_weights_from_tensor(recv_req)
                if success:
                    self.flush_cache_after_weight_update(recv_req)
                else:
                    logger.error(message)
                torch.distributed.barrier(group=self.tp_cpu_group)
                return UpdateWeightsFromTensorReqOutput(
                    success=success,
                    message=message,
                )

            success, message = self._run_weight_update_transaction(
                operation="tensor weight update",
                mutate=lambda: worker.update_weights_from_tensor(recv_req),
                workers=workers,
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            workers = [self.tp_worker]
            if self.draft_worker is not None:
                workers.append(self.draft_worker)

            if not self._weight_update_transaction_required(workers):
                success, message = self.tp_worker.update_weights_from_ipc(recv_req)
                tp_success = success
                if success and self.draft_worker is not None:
                    success, message = self.draft_worker.update_weights_from_ipc(
                        recv_req
                    )
                if tp_success:
                    self.flush_cache_after_weight_update(recv_req)
                if not success:
                    logger.error(message)
                torch.distributed.barrier(group=self.tp_cpu_group)
                return UpdateWeightsFromIPCReqOutput(
                    success=success,
                    message=message,
                )

            def mutate():
                success, message = self.tp_worker.update_weights_from_ipc(recv_req)
                if success and self.draft_worker is not None:
                    success, message = self.draft_worker.update_weights_from_ipc(
                        recv_req
                    )
                return success, message

            success, message = self._run_weight_update_transaction(
                operation="IPC weight update",
                mutate=mutate,
                workers=workers,
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def _defer_remote_instance_weight_transfer(self, operation, recv_req) -> None:
        if self.remote_weight_transfer_executor is None:
            self.remote_weight_transfer_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sglang-weight-transfer",
            )
        future = self.remote_weight_transfer_executor.submit(operation, recv_req)
        self.remote_weight_transfer_pending.append((future, recv_req))

    def defer_begin_remote_instance_weight_transfer(
        self, recv_req: BeginRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.begin_remote_instance_weight_transfer, recv_req
        )

    def defer_release_remote_instance_weight_transfer(
        self, recv_req: ReleaseRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.release_remote_instance_weight_transfer, recv_req
        )

    def defer_renew_remote_instance_weight_transfer(
        self, recv_req: RenewRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.renew_remote_instance_weight_transfer, recv_req
        )

    @staticmethod
    def _remote_instance_weight_transfer_failure(recv_req, error: Exception):
        kwargs = {
            "transfer_id": recv_req.transfer_id,
            "success": False,
            "message": str(error),
        }
        if isinstance(recv_req, BeginRemoteInstanceWeightTransferReqInput):
            return BeginRemoteInstanceWeightTransferReqOutput(**kwargs)
        if isinstance(recv_req, ReleaseRemoteInstanceWeightTransferReqInput):
            return ReleaseRemoteInstanceWeightTransferReqOutput(**kwargs)
        return RenewRemoteInstanceWeightTransferReqOutput(**kwargs)

    def check_pending_remote_instance_weight_transfers(self):
        self._prune_remote_weight_transfer_bookkeeping()
        completed = []
        remaining = []
        for future, recv_req in self.remote_weight_transfer_pending:
            if not future.done():
                remaining.append((future, recv_req))
                continue
            try:
                output = future.result()
            except Exception as error:
                logger.exception("Remote instance weight transfer control failed")
                output = self._remote_instance_weight_transfer_failure(recv_req, error)
            completed.append((output, recv_req))
        self.remote_weight_transfer_pending = remaining
        return completed

    def close_remote_instance_weight_transfer_executor(self) -> None:
        if self.remote_weight_transfer_executor is None:
            return
        self.remote_weight_transfer_executor.shutdown(wait=True)
        self.remote_weight_transfer_executor = None

    def begin_remote_instance_weight_transfer(
        self, recv_req: BeginRemoteInstanceWeightTransferReqInput
    ) -> BeginRemoteInstanceWeightTransferReqOutput:
        """Acquire one address-stable snapshot on every model rank."""
        collective_group = self.remote_weight_transfer_cpu_group or self.world_cpu_group
        local_inventories = None
        local_lease_id = None
        local_generation = None
        cached = None
        try:
            validate_remote_weight_lineage(
                model_id=recv_req.model_id,
                revision=recv_req.revision,
            )
            cached = self._cached_remote_weight_transfer_session(recv_req)
            if cached is not None:
                local_result = {
                    "success": True,
                    "message": "Success.",
                    "session_state": "reused",
                }
            elif (
                self._get_remote_weight_transfer_lease(recv_req.transfer_id) is not None
            ):
                raise _RemoteWeightTransferSessionError(
                    f"remote weight transfer already exists: {recv_req.transfer_id}",
                    session_state="cleanup_pending",
                )
            else:
                local_inventories = (
                    self.tp_worker.model_runner.get_remote_instance_weight_inventories(
                        model_id=recv_req.model_id,
                        revision=recv_req.revision,
                        transfer_id=recv_req.transfer_id,
                        lease_timeout_sec=recv_req.lease_timeout_sec,
                    )
                )
                local_lease_id = self._remote_transfer_inventory_lease_id(
                    local_inventories
                )
                self._record_remote_weight_transfer_lease(
                    recv_req.transfer_id,
                    local_lease_id,
                    recv_req.lease_timeout_sec,
                )
                local_generation = self._remote_transfer_inventory_generation(
                    local_inventories
                )
                with self.remote_weight_transfer_lock:
                    self.remote_weight_transfer_generations[recv_req.transfer_id] = (
                        local_generation
                    )
                placement = (
                    local_inventories["placement"]
                    if isinstance(local_inventories, dict)
                    else local_inventories.placement
                )
                binding = (
                    local_inventories["binding"]
                    if isinstance(local_inventories, dict)
                    else local_inventories.binding
                )
                placement_payload = (
                    placement
                    if isinstance(placement, dict)
                    else msgspec.to_builtins(placement)
                )
                binding_payload = (
                    binding
                    if isinstance(binding, dict)
                    else msgspec.to_builtins(binding)
                )
                local_result = {
                    "success": True,
                    "message": "Success.",
                    "session_state": "created",
                    "placement_inventory": placement_payload,
                    "binding_inventory": binding_payload,
                }
        except Exception as error:
            local_result = {
                "success": False,
                "message": str(error),
                "session_state": getattr(error, "session_state", "failed"),
            }

        try:
            world_size = torch.distributed.get_world_size(group=collective_group)
            gathered = [None] * world_size
            torch.distributed.all_gather_object(
                gathered, local_result, group=collective_group
            )
        except Exception as error:
            cleanup_error = None
            if local_lease_id is not None:
                cleanup_error = self._rollback_remote_weight_transfer_snapshot(
                    recv_req.transfer_id,
                    local_lease_id,
                )
            message = f"Failed to gather source weight inventories: {error}"
            session_state = "failed"
            if cleanup_error is not None:
                message += f"; snapshot cleanup remains pending: {cleanup_error}"
                session_state = "cleanup_pending"
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=message,
                session_state=session_state,
            )

        failures = [item["message"] for item in gathered if not item["success"]]
        session_states = {
            item.get("session_state", "created") for item in gathered if item["success"]
        }
        if not failures and len(session_states) != 1:
            failures.append(
                "source ranks have inconsistent session state for remote weight transfer"
            )
        if not failures and session_states == {"reused"}:
            if cached is None:
                failures.append(
                    "source ranks have inconsistent cached session state for "
                    "remote weight transfer"
                )
            else:
                return BeginRemoteInstanceWeightTransferReqOutput(
                    transfer_id=cached.transfer_id,
                    success=cached.success,
                    message=cached.message,
                    session_state="reused",
                    placement_inventories=cached.placement_inventories,
                    binding_inventories=cached.binding_inventories,
                )

        placement_inventories = None
        binding_inventories = None
        if not failures:
            try:
                placement_inventories = [
                    item["placement_inventory"] for item in gathered
                ]
                binding_inventories = [item["binding_inventory"] for item in gathered]
                self._validate_remote_transfer_inventories(
                    placement_inventories,
                    binding_inventories,
                    world_size,
                    model_id=recv_req.model_id,
                    revision=recv_req.revision,
                )
            except Exception as error:
                failures.append(str(error))

        if failures:
            cleanup_error = None
            if local_lease_id is not None:
                cleanup_error = self._rollback_remote_weight_transfer_snapshot(
                    recv_req.transfer_id,
                    local_lease_id,
                )
            all_session_states = {
                item.get(
                    "session_state",
                    "created" if item["success"] else "failed",
                )
                for item in gathered
            }
            failure_states = {
                item.get("session_state", "failed")
                for item in gathered
                if not item["success"]
            }
            if "conflict" in all_session_states:
                session_state = "conflict"
            elif "expired" in all_session_states:
                session_state = "expired"
            elif all_session_states & {"created", "reused", "cleanup_pending"}:
                session_state = "cleanup_pending"
            elif "released" in all_session_states:
                session_state = "released"
            elif len(failure_states) == 1:
                session_state = next(iter(failure_states))
            else:
                session_state = "failed"
            if cleanup_error is not None:
                failures.append(f"snapshot cleanup remains pending: {cleanup_error}")
                if session_state not in {"conflict", "expired"}:
                    session_state = "cleanup_pending"
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=" | ".join(failures),
                session_state=session_state,
            )

        assert local_lease_id is not None
        output = BeginRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=True,
            message="Success.",
            session_state="created",
            placement_inventories=placement_inventories,
            binding_inventories=binding_inventories,
        )
        self._record_remote_weight_transfer_session(
            recv_req,
            local_lease_id,
            output,
            generation=local_generation,
        )
        return output

    @staticmethod
    def _remote_transfer_inventory_lease_id(inventories) -> str:
        binding = (
            inventories["binding"]
            if isinstance(inventories, dict)
            else inventories.binding
        )
        return binding["lease_id"] if isinstance(binding, dict) else binding.lease_id

    @staticmethod
    def _remote_transfer_inventory_generation(inventories) -> int:
        binding = (
            inventories["binding"]
            if isinstance(inventories, dict)
            else inventories.binding
        )
        return (
            binding["generation"] if isinstance(binding, dict) else binding.generation
        )

    @staticmethod
    def _remote_transfer_output_generation(
        output: BeginRemoteInstanceWeightTransferReqOutput,
    ) -> int | None:
        records = output.binding_inventories or ()
        generations = {
            record.get("generation") for record in records if isinstance(record, dict)
        }
        generations.discard(None)
        return next(iter(generations)) if len(generations) == 1 else None

    def _gather_remote_weight_transfer_status(
        self, *, success: bool, message: str, operation: str
    ) -> Tuple[bool, str]:
        local_result = {"success": success, "message": message}
        collective_group = self.remote_weight_transfer_cpu_group or self.world_cpu_group
        try:
            world_size = torch.distributed.get_world_size(group=collective_group)
            gathered = [None] * world_size
            torch.distributed.all_gather_object(
                gathered, local_result, group=collective_group
            )
        except Exception as error:
            return False, f"Failed to gather source {operation} results: {error}"

        failures = [item["message"] for item in gathered if not item["success"]]
        if failures:
            return False, " | ".join(failures)
        return True, "Success."

    def renew_remote_instance_weight_transfer(
        self, recv_req: RenewRemoteInstanceWeightTransferReqInput
    ) -> RenewRemoteInstanceWeightTransferReqOutput:
        self._prune_remote_weight_transfer_bookkeeping()
        with self.remote_weight_transfer_lock:
            expired = recv_req.transfer_id in self.remote_weight_transfer_expired
        lease_id = self._get_remote_weight_transfer_lease(recv_req.transfer_id)
        if expired:
            local_success = False
            local_message = (
                "Remote weight transfer expired and requires explicit release."
            )
        elif lease_id is None:
            local_success = False
            local_message = "Remote weight transfer does not exist or has expired."
        else:
            try:
                self.tp_worker.model_runner.renew_weight_inventory(
                    lease_id,
                    lease_timeout_sec=recv_req.lease_timeout_sec,
                )
                self._record_remote_weight_transfer_lease(
                    recv_req.transfer_id,
                    lease_id,
                    recv_req.lease_timeout_sec,
                )
                local_success = True
                local_message = "Success."
            except Exception as error:
                local_success = False
                local_message = str(error)

        success, message = self._gather_remote_weight_transfer_status(
            success=local_success,
            message=local_message,
            operation="renewal",
        )
        return RenewRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=success,
            message=message,
        )

    @staticmethod
    def _validate_remote_transfer_inventories(
        placements,
        bindings,
        world_size: int,
        *,
        model_id: str,
        revision: str,
    ) -> None:
        if len(placements) != world_size or len(bindings) != world_size:
            raise RuntimeError(
                "source placement and binding counts must match the model world"
            )
        if any(not placement.get("fragments") for placement in placements):
            raise RuntimeError("every source rank must publish placement fragments")
        if any(not binding.get("fragments") for binding in bindings):
            raise RuntimeError("every source rank must publish at least one binding")

        typed_pairs = []
        try:
            for placement, binding in zip(placements, bindings):
                typed_pairs.append(
                    WeightPlacementBindingInventories(
                        placement=msgspec.convert(
                            placement,
                            type=WeightPlacementInventory,
                            strict=True,
                        ),
                        binding=msgspec.convert(
                            binding,
                            type=WeightRuntimeBindingInventory,
                            strict=True,
                        ),
                    )
                )
        except Exception as error:
            raise RuntimeError(
                "source placement/binding inventories are not self-consistent"
            ) from error

        participant_ids = [pair.binding.participant_id for pair in typed_pairs]
        if len(set(participant_ids)) != world_size:
            raise RuntimeError("source inventory participant IDs are not unique")

        identities = {
            (
                pair.placement.model_id,
                pair.placement.revision,
                pair.placement.weight_generation,
            )
            for pair in typed_pairs
        }
        if len(identities) != 1:
            raise RuntimeError(
                "source placements do not describe one logical weight generation"
            )
        identity = next(iter(identities))
        if identity[:2] != (model_id, revision):
            raise RuntimeError(
                "source placements do not match the requested model identity"
            )

    def release_remote_instance_weight_transfer(
        self, recv_req: ReleaseRemoteInstanceWeightTransferReqInput
    ) -> ReleaseRemoteInstanceWeightTransferReqOutput:
        lease_id = self._get_remote_weight_transfer_lease(recv_req.transfer_id)
        with self.remote_weight_transfer_lock:
            generation = self.remote_weight_transfer_generations.get(
                recv_req.transfer_id
            )
        if lease_id is None:
            self._complete_remote_weight_transfer_session(recv_req.transfer_id)
            local_success = True
            local_message = "Remote weight transfer was already released."
        else:
            try:
                self.tp_worker.model_runner.release_weight_inventory(lease_id)
                self._complete_remote_weight_transfer_session(recv_req.transfer_id)
                local_success = True
                local_message = "Success."
            except Exception as error:
                local_success = False
                local_message = str(error)
                logger.warning(
                    "Explicit remote weight transfer release failed: "
                    "transfer_id=%s lease_id=%s generation=%s error=%s",
                    recv_req.transfer_id,
                    lease_id,
                    generation,
                    error,
                )

        success, message = self._gather_remote_weight_transfer_status(
            success=local_success,
            message=local_message,
            operation="release",
        )
        return ReleaseRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=success,
            message=message,
        )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        with self._coordinate_weight_memory_transition(
            enabled=GPU_MEMORY_TYPE_WEIGHTS in tags,
            storage_available_after=False,
        ):
            for tag in tags:
                self.offload_tags.add(tag)

            if GPU_MEMORY_TYPE_KV_CACHE in tags:
                scheduler = self.scheduler
                if scheduler is not None:
                    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                        for queue_name in (
                            "disagg_decode_transfer_queue",
                            "disagg_decode_prealloc_queue",
                        ):
                            queue = getattr(scheduler, queue_name, None)
                            if queue is not None:
                                queue.release_memory_occupation()
                    elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                        queue = getattr(
                            scheduler, "disagg_prefill_bootstrap_queue", None
                        )
                        if queue is not None:
                            queue.release_memory_occupation()
                self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
                self.flush_cache()

            if GPU_MEMORY_TYPE_WEIGHTS in tags:
                self.stashed_model_static_state = _export_static_state(
                    self.tp_worker.model_runner.model
                )
                torch.distributed.barrier(self.tp_cpu_group)
                self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

            if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
                self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

            torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        with self._coordinate_weight_memory_transition(
            enabled=GPU_MEMORY_TYPE_WEIGHTS in tags,
            storage_available_after=True,
        ):
            for tag in tags:
                self.offload_tags.remove(tag)

            if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
                self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

            if GPU_MEMORY_TYPE_WEIGHTS in tags:
                self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
                torch.distributed.barrier(self.tp_cpu_group)
                _import_static_state(
                    self.tp_worker.model_runner.model,
                    self.stashed_model_static_state,
                )
                del self.stashed_model_static_state

            if GPU_MEMORY_TYPE_KV_CACHE in tags:
                self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
                scheduler = self.scheduler
                if scheduler is not None:
                    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                        for queue_name in (
                            "disagg_decode_transfer_queue",
                            "disagg_decode_prealloc_queue",
                        ):
                            queue = getattr(scheduler, queue_name, None)
                            if queue is not None:
                                queue.resume_memory_occupation()
                    elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                        queue = getattr(
                            scheduler, "disagg_prefill_bootstrap_queue", None
                        )
                        if queue is not None:
                            queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(
                action=recv_req.action, allow_quant_error=recv_req.allow_quant_error
            )

            if self.draft_worker is not None:
                draft_runner = _get_draft_model_runner(self.draft_worker)
                if draft_runner is not None:
                    draft_payload = draft_runner.check_weights(
                        action=recv_req.action,
                        allow_quant_error=recv_req.allow_quant_error,
                    )
                    if payload is not None and draft_payload is not None:
                        payload = _merge_checksum_payloads(payload, draft_payload)

            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            if payload is not None:
                # Normalize to one ChecksumInfo per rank so the wire shape is a
                # uniform List[ChecksumInfo] (tp==1 becomes a single-element list).
                per_rank = payload if isinstance(payload, list) else [payload]
                payload = [msgspec.convert(p, ChecksumInfo) for p in per_rank]
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.weight_exporter.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.weight_exporter.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        self.tp_worker.model_runner.weight_exporter.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
