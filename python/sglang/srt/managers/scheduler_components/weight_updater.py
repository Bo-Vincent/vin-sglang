from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

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
    CommitWeightMaterializationReqInput,
    CommitWeightMaterializationReqOutput,
    PrepareWeightMaterializationReqInput,
    PrepareWeightMaterializationReqOutput,
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
from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightManifestError,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from sglang.srt.weight_transfer.api import materialize_weight_snapshot
from sglang.srt.weight_transfer.binding import (
    bind_weight_source,
    project_source_bindings,
)
from sglang.srt.weight_transfer.distributed import (
    TorchDistributedWeightStoreCoordinator,
)
from sglang.srt.weight_transfer.planner import select_weight_storage_placements
from sglang.srt.weight_transfer.provider import (
    WeightPayloadIdentity,
    WeightTransferCompletionUnknownError,
)
from sglang.srt.weight_transfer.runtime import (
    RuntimeWeightSnapshotSource,
    materialize_distributed_runtime_weight_snapshot,
)
from sglang.srt.weight_transfer.store_runtime import (
    WeightSnapshotWriteSpec,
    open_weight_snapshot_write_backend,
)

logger = logging.getLogger(__name__)

_REMOTE_WEIGHT_TRANSFER_TOMBSTONE_TTL_SEC = 300.0
_REMOTE_WEIGHT_TRANSFER_TOMBSTONE_LIMIT = 4096
_WEIGHT_MATERIALIZATION_TERMINAL_TTL_SEC = 300.0
_WEIGHT_MATERIALIZATION_TERMINAL_LIMIT = 4096
_WEIGHT_STORAGE_OWNER_LIMIT = 4096


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


@dataclass(slots=True)
class _WeightStorageBackendOwner:
    stack: ExitStack = field(default_factory=ExitStack)
    closed: bool = False
    terminal_error: str | None = None

    def enter_context(self, context_manager):
        if self.closed or self.terminal_error is not None:
            raise RuntimeError("weight storage backend owner is not open")
        return self.stack.enter_context(context_manager)

    def close(self) -> None:
        if self.terminal_error is not None:
            raise RuntimeError(self.terminal_error)
        if self.closed:
            return
        try:
            self.stack.close()
        except Exception as error:
            self.terminal_error = str(error)
            raise
        self.closed = True


@dataclass(slots=True)
class _WeightMaterializationSession:
    request_identity: tuple[str, str]
    source: RuntimeWeightSnapshotSource | None
    selected_placements: tuple[WeightPlacementManifest, ...]
    selected_bindings: tuple[WeightRuntimeBindingManifest, ...]
    selected_payload_identity: WeightPayloadIdentity | None
    local_selected_placement_ids: tuple[str, ...]
    prepare_output: PrepareWeightMaterializationReqOutput
    state: str
    commit_identity: tuple[int | None, str | None] | None = None
    commit_output: CommitWeightMaterializationReqOutput | None = None
    backend_owner: _WeightStorageBackendOwner | None = None
    terminal_at: float | None = None


@dataclass(frozen=True, slots=True)
class _NoLocalRuntimeSourceAttestor:
    placement_fragment_ids: frozenset[str]
    lease_id: str
    worker_ids: frozenset[str]

    @classmethod
    def from_source(
        cls,
        source: RuntimeWeightSnapshotSource,
    ) -> _NoLocalRuntimeSourceAttestor:
        return cls(
            placement_fragment_ids=frozenset(
                tensor.placement_fragment_id for tensor in source.placement.tensors
            ),
            lease_id=source.binding.lease_id,
            worker_ids=frozenset(
                fragment.worker_id for fragment in source.binding.fragments
            ),
        )

    def attest(self, request: Any) -> None:
        request_fragment_ids = {
            tensor.placement_fragment_id
            for placement in request.source_placements
            for tensor in placement.tensors
        }
        if request_fragment_ids & self.placement_fragment_ids:
            raise RuntimeError(
                "materialization request unexpectedly includes the local source"
            )
        for binding in request.source_bindings:
            if not isinstance(binding, WeightRuntimeBindingManifest):
                raise RuntimeError(
                    "runtime materialization requires runtime source bindings"
                )
            if binding.lease_id == self.lease_id or any(
                fragment.worker_id in self.worker_ids for fragment in binding.fragments
            ):
                raise RuntimeError(
                    "materialization request unexpectedly binds the local source"
                )


def _logical_payload_digest(
    placements: tuple[WeightPlacementManifest, ...],
    payload_identity: WeightPayloadIdentity,
) -> str:
    checksum_by_id = {
        fragment.placement_fragment_id: fragment.checksum
        for fragment in payload_identity.fragments
    }
    records = []
    for placement in placements:
        for tensor in placement.tensors:
            checksum = checksum_by_id.get(tensor.placement_fragment_id)
            if checksum is None:
                raise ValueError(
                    "payload identity does not cover selected placement fragments"
                )
            records.append(
                {
                    "tensor_id": tensor.tensor_id,
                    "global_offset": tuple(tensor.global_offset),
                    "local_shape": tuple(tensor.local_shape),
                    "dtype": tensor.dtype,
                    "itemsize": tensor.itemsize,
                    "layout_fingerprint": tensor.layout_fingerprint,
                    "checksum": checksum,
                }
            )
    payload = json.dumps(
        {
            "format": "sglang-logical-weight-payload-v1",
            "tensors": sorted(
                records,
                key=lambda item: (
                    item["tensor_id"],
                    item["global_offset"],
                    item["local_shape"],
                    item["dtype"],
                    item["itemsize"],
                    item["layout_fingerprint"],
                    item["checksum"],
                ),
            ),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


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
    weight_materialization_cpu_group: Any = None
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
    weight_materialization_executor: Optional[ThreadPoolExecutor] = field(
        default=None, init=False, repr=False
    )
    weight_materialization_pending: List[Tuple[Future, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    weight_materialization_sessions: Dict[
        str,
        _WeightMaterializationSession,
    ] = field(default_factory=dict)
    weight_storage_owners: Dict[
        tuple[str, str, str, str, str, str, str],
        tuple[dict[str, Any], _WeightStorageBackendOwner],
    ] = field(default_factory=dict, init=False, repr=False)
    weight_materialization_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @contextmanager
    def _coordinate_weight_memory_transition(
        self,
        *,
        enabled: bool,
        commit_revision: bool,
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
            token = coordinator.begin_update()
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
            raise WeightManifestError(
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
            raise WeightManifestError(
                "weight memory transition reservation failed: " + " | ".join(failures)
            )

        assert token is not None
        success = False
        try:
            yield
            success = True
        finally:
            coordinator.finish_update(token, success=success)
            if success and commit_revision:
                coordinator.commit_revision()

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
            request.manifest_format,
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
            self.tp_worker.model_runner.release_weight_runtime_manifest(lease_id)
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
    def _commit_weight_runtime_revision(worker) -> None:
        runner = getattr(worker, "model_runner", None)
        if getattr(runner, "weight_snapshot_coordinator", None) is None:
            return
        runner.commit_weight_runtime_revision()

    @staticmethod
    def _weight_snapshot_coordinator(worker):
        if worker is None:
            return None
        runner = getattr(worker, "model_runner", None)
        if runner is None:
            runner = _get_draft_model_runner(worker)
        return getattr(runner, "weight_snapshot_coordinator", None)

    def _weight_update_coordinators(self, workers) -> tuple:
        coordinators = []
        seen = set()
        for worker in workers:
            coordinator = self._weight_snapshot_coordinator(worker)
            if coordinator is None or id(coordinator) in seen:
                continue
            coordinators.append(coordinator)
            seen.add(id(coordinator))
        return tuple(coordinators)

    def _capture_weight_update_generations(self, workers) -> tuple:
        return tuple(
            (coordinator, coordinator.generation)
            for coordinator in self._weight_update_coordinators(workers)
        )

    @staticmethod
    def _validate_weight_update_generations(
        generation_mapping,
        *,
        require_pending_revision: bool,
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
            if require_pending_revision:
                pending_generation = coordinator.pending_revision_generation()
                if pending_generation != expected_generation:
                    failures.append(
                        "weight update generation "
                        f"{expected_generation} is not pending revision commit"
                    )
        if failures:
            raise WeightManifestError(" | ".join(failures))

    @staticmethod
    def _finalize_weight_update(generation_mapping, *, commit: bool) -> None:
        failures = []
        for coordinator, expected_generation in generation_mapping:
            try:
                if commit:
                    coordinator.commit_revision(expected_generation=expected_generation)
                else:
                    coordinator.poison_global_update_failure(
                        expected_generation=expected_generation
                    )
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
        if failures:
            raise WeightManifestError(" | ".join(failures))

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

    def _run_weight_update_transaction(
        self,
        *,
        operation: str,
        mutate: Callable[[], Tuple[bool, str]],
        workers,
        recv_req,
    ) -> Tuple[bool, str]:
        try:
            local_success, local_message = mutate()
            if not isinstance(local_success, bool):
                raise TypeError("weight update success outcome must be a boolean")
            local_message = str(local_message)
        except Exception as error:
            local_success = False
            local_message = f"{type(error).__name__}: {error}"

        try:
            generation_mapping = self._capture_weight_update_generations(workers)
        except Exception as error:
            generation_mapping = ()
            local_success = False
            local_message = (
                f"{local_message} | failed to capture weight update generation: "
                f"{type(error).__name__}: {error}"
            )

        if local_success:
            try:
                self.flush_cache_after_weight_update(recv_req)
            except Exception as error:
                local_success = False
                local_message = f"{type(error).__name__}: {error}"

        try:
            mutation_success, mutation_message = self._gather_weight_update_outcome(
                success=local_success,
                message=local_message,
                phase=f"{operation} mutation",
            )
        except Exception as error:
            self._poison_weight_update_best_effort(generation_mapping)
            return (
                False,
                f"Failed to gather {operation} mutation outcomes: "
                f"{type(error).__name__}: {error}",
            )

        try:
            self._validate_weight_update_generations(
                generation_mapping,
                require_pending_revision=mutation_success,
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
            return (
                False,
                f"Failed to gather {operation} finalize readiness outcomes: "
                f"{type(error).__name__}: {error}",
            )

        try:
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
            return (
                False,
                f"Failed to gather {operation} finalize outcomes: "
                f"{type(error).__name__}: {error}",
            )

        if not finalize_success:
            self._poison_weight_update_best_effort(generation_mapping)
        if not mutation_success:
            return False, mutation_message
        if not ready_success:
            return False, ready_message
        if not finalize_success:
            return False, finalize_message
        return True, local_message

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            workers = [self.tp_worker]
            if self.draft_worker is not None:
                workers.append(self.draft_worker)

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
            success, message = self._run_weight_update_transaction(
                operation="distributed weight update",
                mutate=lambda: self.tp_worker.update_weights_from_distributed(recv_req),
                workers=(self.tp_worker,),
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
            success, message = self._run_weight_update_transaction(
                operation="tensor weight update",
                mutate=lambda: worker.update_weights_from_tensor(recv_req),
                workers=(worker,),
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

    def _assert_weight_cache_inactive(self, op: str) -> None:
        """Reject freeing/restoring model weights while the CUDA IPC weight
        cache is active: the weights are shared with the daemon via CUDA IPC, so
        freeing them would leave the daemon and every peer pointing at released
        memory.
        """
        mode = self.tp_worker.model_runner.server_args.weight_cache_mode
        if mode != "off":
            raise RuntimeError(
                f"[weight_cache] {op} of model weights is not supported while the "
                f"weight cache is active (--weight-cache-mode {mode}): the weights "
                f"are shared with the daemon via CUDA IPC, so freeing them would "
                f"corrupt the daemon's master copy and every co-attached engine. "
                f"Restart with --weight-cache-mode off to use this operation."
            )

    def _external_dp_rank(self) -> int:
        rank = getattr(getattr(self.scheduler, "ps", None), "dp_rank", 0)
        if rank is None:
            return 0
        if type(rank) is not int or rank < 0:
            raise RuntimeError("external DP rank must be a non-negative integer")
        return rank

    def _prepare_materialization_failure(
        self,
        recv_req: PrepareWeightMaterializationReqInput,
        message: str,
        *,
        session_state: str = "failed",
    ) -> PrepareWeightMaterializationReqOutput:
        return PrepareWeightMaterializationReqOutput(
            materialization_id=recv_req.materialization_id,
            success=False,
            message=message,
            external_dp_rank=self._external_dp_rank(),
            session_state=session_state,
        )

    def _commit_materialization_failure(
        self,
        recv_req: CommitWeightMaterializationReqInput,
        message: str,
        *,
        session_state: str = "failed",
        completion_unknown: bool = False,
        completion_ticket: str | None = None,
    ) -> CommitWeightMaterializationReqOutput:
        external_dp_rank = self._external_dp_rank()
        return CommitWeightMaterializationReqOutput(
            materialization_id=recv_req.materialization_id,
            success=False,
            message=message,
            external_dp_rank=external_dp_rank,
            selected=(
                recv_req.selected_external_dp_rank is not None
                and recv_req.selected_external_dp_rank == external_dp_rank
            ),
            completion_unknown=completion_unknown,
            completion_ticket=completion_ticket,
            session_state=session_state,
        )

    def _retain_materialization_cleanup_session(
        self,
        recv_req: PrepareWeightMaterializationReqInput,
        source: RuntimeWeightSnapshotSource | None,
        output: PrepareWeightMaterializationReqOutput,
    ) -> None:
        session = _WeightMaterializationSession(
            request_identity=(recv_req.model_id, recv_req.revision),
            source=source,
            selected_placements=(),
            selected_bindings=(),
            selected_payload_identity=None,
            local_selected_placement_ids=(),
            prepare_output=output,
            state="cleanup_pending",
        )
        with self.weight_materialization_lock:
            self.weight_materialization_sessions.setdefault(
                recv_req.materialization_id,
                session,
            )

    def _prepare_failure_after_world_cleanup(
        self,
        recv_req: PrepareWeightMaterializationReqInput,
        source: RuntimeWeightSnapshotSource | None,
        failures: list[str],
        *,
        operation: str,
    ) -> PrepareWeightMaterializationReqOutput:
        provisional = self._prepare_materialization_failure(
            recv_req,
            " | ".join(failures),
            session_state="cleanup_pending",
        )
        session = _WeightMaterializationSession(
            request_identity=(recv_req.model_id, recv_req.revision),
            source=source,
            selected_placements=(),
            selected_bindings=(),
            selected_payload_identity=None,
            local_selected_placement_ids=(),
            prepare_output=provisional,
            state="cleanup_pending",
        )
        cleanup_errors, _completion_unknown = (
            self._release_materialization_source_world(
                session,
                operation=operation,
            )
        )
        if not cleanup_errors:
            return self._prepare_materialization_failure(
                recv_req,
                " | ".join(failures),
                session_state="failed",
            )

        failures.extend(
            f"source cleanup remains pending: {error}" for error in cleanup_errors
        )
        output = self._prepare_materialization_failure(
            recv_req,
            " | ".join(failures),
            session_state="cleanup_pending",
        )
        self._retain_materialization_cleanup_session(
            recv_req,
            session.source,
            output,
        )
        return output

    def _prune_weight_materialization_sessions_locked(self) -> None:
        now = time.monotonic()
        terminal = [
            (materialization_id, session)
            for materialization_id, session in self.weight_materialization_sessions.items()
            if session.terminal_at is not None
        ]
        for materialization_id, session in terminal:
            assert session.terminal_at is not None
            if now - session.terminal_at >= _WEIGHT_MATERIALIZATION_TERMINAL_TTL_SEC:
                self.weight_materialization_sessions.pop(materialization_id, None)
        terminal = sorted(
            (
                (materialization_id, session)
                for materialization_id, session in (
                    self.weight_materialization_sessions.items()
                )
                if session.terminal_at is not None
            ),
            key=lambda item: item[1].terminal_at,
        )
        for materialization_id, _session in terminal[
            : max(0, len(terminal) - _WEIGHT_MATERIALIZATION_TERMINAL_LIMIT)
        ]:
            self.weight_materialization_sessions.pop(materialization_id, None)

    def _gather_weight_materialization_objects(
        self,
        value: Any,
        *,
        operation: str,
    ) -> list[Any]:
        collective_group = self._weight_materialization_collective_group()
        try:
            world_size = torch.distributed.get_world_size(group=collective_group)
            gathered = [None] * world_size
            torch.distributed.all_gather_object(
                gathered,
                value,
                group=collective_group,
            )
            return gathered
        except Exception as error:
            raise RuntimeError(
                f"failed to gather weight materialization {operation}: {error}"
            ) from error

    def _weight_materialization_collective_group(self) -> Any:
        group = self.weight_materialization_cpu_group
        if group is not None:
            return group
        world_size = torch.distributed.get_world_size(group=self.world_cpu_group)
        if world_size == 1:
            return self.world_cpu_group
        raise RuntimeError(
            "multi-rank weight materialization requires an isolated CPU process group"
        )

    @staticmethod
    def _release_materialization_source(source: Any) -> str | None:
        if source is None:
            return None
        if getattr(source, "released", False):
            return None
        if getattr(source, "quarantined", False):
            return "completion-unknown runtime source remains quarantined"
        try:
            source.release()
        except Exception as error:
            return str(error)
        return None

    def _release_materialization_session_source(
        self,
        session: _WeightMaterializationSession,
    ) -> str | None:
        error = self._release_materialization_source(session.source)
        if error is None:
            session.source = None
        return error

    def _release_materialization_source_world(
        self,
        session: _WeightMaterializationSession,
        *,
        operation: str,
    ) -> tuple[list[str], bool]:
        completion_unknown = bool(
            session.source is not None and getattr(session.source, "quarantined", False)
        )
        local_error = self._release_materialization_session_source(session)
        try:
            statuses = self._gather_weight_materialization_objects(
                {
                    "error": local_error,
                    "completion_unknown": completion_unknown,
                },
                operation=operation,
            )
        except Exception as error:
            return [str(error)], completion_unknown

        errors = []
        any_completion_unknown = False
        for rank, status in enumerate(statuses):
            if (
                not isinstance(status, dict)
                or "error" not in status
                or type(status.get("completion_unknown")) is not bool
                or (
                    status.get("error") is not None
                    and type(status.get("error")) is not str
                )
            ):
                errors.append(f"rank {rank}: invalid source release status")
                continue
            any_completion_unknown |= status["completion_unknown"]
            if status["error"] is not None:
                errors.append(f"rank {rank}: {status['error']}")
        return errors, any_completion_unknown

    @staticmethod
    def _close_materialization_backend(
        session: _WeightMaterializationSession,
    ) -> str | None:
        owner = session.backend_owner
        if owner is None:
            return None
        try:
            owner.close()
        except Exception as error:
            return str(error)
        session.backend_owner = None
        return None

    def _close_materialization_backend_world(
        self,
        session: _WeightMaterializationSession,
        *,
        operation: str,
    ) -> list[str]:
        try:
            ownership = self._gather_weight_materialization_objects(
                {"present": session.backend_owner is not None},
                operation=f"{operation} ownership",
            )
        except Exception as error:
            return [str(error)]
        if any(
            not isinstance(status, dict)
            or "present" not in status
            or type(status.get("present")) is not bool
            for status in ownership
        ):
            return ["model ranks returned invalid Store backend ownership"]
        if len({status["present"] for status in ownership}) != 1:
            return ["model ranks disagree on Store backend ownership"]

        local_error = self._close_materialization_backend(session)
        try:
            statuses = self._gather_weight_materialization_objects(
                {"error": local_error},
                operation=f"{operation} status",
            )
        except Exception as error:
            return [str(error)]
        errors = []
        for rank, status in enumerate(statuses):
            if (
                not isinstance(status, dict)
                or "error" not in status
                or (
                    status.get("error") is not None
                    and type(status.get("error")) is not str
                )
            ):
                errors.append(f"rank {rank}: invalid Store backend close status")
            elif status.get("error") is not None:
                errors.append(f"rank {rank}: {status['error']}")
        return errors

    @staticmethod
    def _weight_storage_owner_key(
        model_id: str,
        revision: str,
        storage_identity: str,
        ref: Mapping[str, Any],
    ) -> tuple[str, str, str, str, str, str, str]:
        return (
            model_id,
            revision,
            storage_identity,
            ref["provider"],
            ref["storage_id"],
            ref["manifest_key"],
            ref["manifest_digest"],
        )

    def _retain_weight_storage_owner(
        self,
        *,
        model_id: str,
        revision: str,
        storage_identity: str,
        ref: dict[str, Any],
        owner: _WeightStorageBackendOwner,
    ) -> str | None:
        key = self._weight_storage_owner_key(
            model_id,
            revision,
            storage_identity,
            ref,
        )
        with self.weight_materialization_lock:
            existing = self.weight_storage_owners.get(key)
        try:
            decisions = self._gather_weight_materialization_objects(
                {
                    "key": key,
                    "already_retained": existing is not None,
                },
                operation="retained Store owner decision",
            )
        except Exception as error:
            return str(error)
        if any(
            not isinstance(decision, dict)
            or tuple(decision.get("key", ())) != key
            or type(decision.get("already_retained")) is not bool
            for decision in decisions
        ):
            return "model ranks returned inconsistent Store owner decisions"
        retained = {decision["already_retained"] for decision in decisions}
        if len(retained) != 1:
            return "model ranks disagree on retained Store owner state"
        if not retained.pop():
            with self.weight_materialization_lock:
                if key in self.weight_storage_owners:
                    return "retained Store owner changed during registration"
                self.weight_storage_owners[key] = (dict(ref), owner)
            return None

        local_close_error = None
        try:
            owner.close()
        except Exception as error:
            local_close_error = str(error)
        try:
            close_statuses = self._gather_weight_materialization_objects(
                {"error": local_close_error},
                operation="duplicate Store owner close",
            )
        except Exception as error:
            return str(error)
        errors = []
        for rank, status in enumerate(close_statuses):
            if (
                not isinstance(status, dict)
                or "error" not in status
                or (
                    status.get("error") is not None
                    and type(status.get("error")) is not str
                )
            ):
                errors.append(f"rank {rank}: invalid Store owner close status")
            elif status["error"] is not None:
                errors.append(f"rank {rank}: {status['error']}")
        return " | ".join(errors) if errors else None

    def _weight_storage_owner_capacity_error(
        self,
        *,
        current_materialization_id: str,
    ) -> str | None:
        with self.weight_materialization_lock:
            session_owned = sum(
                session.backend_owner is not None
                or (
                    session.commit_output is not None
                    and session.commit_output.completion_unknown
                )
                for materialization_id, session in (
                    self.weight_materialization_sessions.items()
                )
                if materialization_id != current_materialization_id
                and session.terminal_at is None
            )
            local_count = len(self.weight_storage_owners) + session_owned
        try:
            statuses = self._gather_weight_materialization_objects(
                {
                    "count": local_count,
                    "limit": _WEIGHT_STORAGE_OWNER_LIMIT,
                },
                operation="retained Store owner capacity",
            )
        except Exception as error:
            return str(error)
        if any(
            not isinstance(status, dict)
            or type(status.get("count")) is not int
            or type(status.get("limit")) is not int
            for status in statuses
        ):
            return "model ranks returned invalid Store owner capacity"
        counts = {status["count"] for status in statuses}
        limits = {status["limit"] for status in statuses}
        if len(counts) != 1 or limits != {_WEIGHT_STORAGE_OWNER_LIMIT}:
            return "model ranks disagree on Store owner capacity"
        if counts.pop() >= _WEIGHT_STORAGE_OWNER_LIMIT:
            return (
                "retained weight storage owner limit reached; release a snapshot "
                "or restart the source before publishing another ref"
            )
        return None

    @staticmethod
    def _commit_request_identity(
        recv_req: CommitWeightMaterializationReqInput,
    ) -> tuple[int | None, str | None]:
        selected_rank = recv_req.selected_external_dp_rank
        if selected_rank is not None and (
            type(selected_rank) is not int or selected_rank < 0
        ):
            raise ValueError(
                "selected_external_dp_rank must be a non-negative integer or None"
            )
        if selected_rank is None:
            return (None, None)
        if not isinstance(recv_req.storage_options, Mapping):
            raise ValueError("storage_options must be a mapping")
        try:
            payload = json.dumps(
                recv_req.storage_options,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("storage_options must be JSON-compatible") from error
        return (
            selected_rank,
            f"sha256:{hashlib.sha256(payload).hexdigest()}",
        )

    @staticmethod
    def _merge_materialization_sources(
        gathered: list[Any],
        *,
        model_id: str,
        revision: str,
        local_source: RuntimeWeightSnapshotSource,
    ) -> tuple[
        tuple[WeightPlacementManifest, ...],
        tuple[WeightRuntimeBindingManifest, ...],
        WeightPayloadIdentity,
        tuple[str, ...],
        int,
        str,
    ]:
        placements = tuple(item["placement"] for item in gathered)
        bindings = tuple(item["binding"] for item in gathered)
        identities = tuple(item["payload_identity"] for item in gathered)
        if not placements or any(
            not isinstance(item, WeightPlacementManifest) for item in placements
        ):
            raise ValueError("model ranks returned invalid weight placements")
        if any(not isinstance(item, WeightRuntimeBindingManifest) for item in bindings):
            raise ValueError("model ranks returned invalid runtime bindings")
        if any(not isinstance(item, WeightPayloadIdentity) for item in identities):
            raise ValueError("model ranks returned invalid payload identities")
        model_revisions = {
            (placement.model_id, placement.revision) for placement in placements
        }
        if model_revisions != {(model_id, revision)}:
            raise ValueError(
                "model ranks do not describe the requested model and revision"
            )
        generations = {binding.generation for binding in bindings}
        if len(generations) != 1:
            raise ValueError("model ranks do not describe one weight generation")

        bind_weight_source(placements, bindings)
        checksums: dict[str, str] = {}
        for placement, identity in zip(placements, identities):
            if identity.select((placement,)) != identity:
                raise ValueError(
                    "rank payload identity differs from its weight placement"
                )
            for fragment in identity.fragments:
                if fragment.placement_fragment_id in checksums:
                    raise ValueError("duplicate payload fragment identity")
                checksums[fragment.placement_fragment_id] = fragment.checksum
        global_identity = WeightPayloadIdentity.create(placements, checksums)

        selected_placements = select_weight_storage_placements(placements)
        projected_bindings = project_source_bindings(
            selected_placements,
            bindings,
        )
        if any(
            not isinstance(binding, WeightRuntimeBindingManifest)
            for binding in projected_bindings
        ):
            raise ValueError("selected Store sources must use runtime bindings")
        selected_bindings = tuple(projected_bindings)
        bind_weight_source(selected_placements, selected_bindings)
        selected_identity = global_identity.select(selected_placements)

        local_fragment_ids = {
            tensor.placement_fragment_id for tensor in local_source.placement.tensors
        }
        local_selected = tuple(
            placement
            for placement in selected_placements
            if any(
                tensor.placement_fragment_id in local_fragment_ids
                for tensor in placement.tensors
            )
        )
        if len(local_selected) > 1:
            raise ValueError("Store selection split one local runtime source")
        if local_selected:
            projected_local_bindings = project_source_bindings(
                local_selected,
                (local_source.binding,),
            )
            local_placement_ids = {
                placement.placement_id for placement in local_selected
            }
            selected_local_bindings = tuple(
                binding
                for binding in selected_bindings
                if binding.placement_id in local_placement_ids
            )
            if selected_local_bindings != projected_local_bindings:
                raise ValueError(
                    "Store selection binding differs from the local runtime projection"
                )
        local_placement_ids = tuple(
            placement.placement_id for placement in local_selected
        )
        return (
            tuple(selected_placements),
            selected_bindings,
            selected_identity,
            local_placement_ids,
            next(iter(generations)),
            _logical_payload_digest(
                tuple(selected_placements),
                selected_identity,
            ),
        )

    def prepare_weight_materialization(
        self,
        recv_req: PrepareWeightMaterializationReqInput,
    ) -> PrepareWeightMaterializationReqOutput:
        """Capture and validate one Store materialization source on every rank."""

        request_identity = (recv_req.model_id, recv_req.revision)
        with self.weight_materialization_lock:
            self._prune_weight_materialization_sessions_locked()
            existing = self.weight_materialization_sessions.get(
                recv_req.materialization_id
            )
        if existing is not None:
            if existing.request_identity != request_identity:
                return self._prepare_materialization_failure(
                    recv_req,
                    "materialization ID is already bound to another model revision",
                    session_state="conflict",
                )
            return existing.prepare_output

        local_source = None
        try:
            local_source = (
                self.tp_worker.model_runner.capture_runtime_weight_snapshot_source(
                    materialization_id=recv_req.materialization_id,
                    model_id=recv_req.model_id,
                    revision=recv_req.revision,
                )
            )
            local_result = {
                "success": True,
                "message": "Success.",
                "placement": local_source.placement,
                "binding": local_source.binding,
                "payload_identity": local_source.payload_identity,
            }
        except Exception as error:
            local_result = {
                "success": False,
                "message": str(error),
                "placement": None,
                "binding": None,
                "payload_identity": None,
            }

        try:
            gathered = self._gather_weight_materialization_objects(
                local_result,
                operation="prepare status",
            )
        except Exception as error:
            cleanup_error = (
                None
                if local_source is None
                else self._release_materialization_source(local_source)
            )
            message = str(error)
            state = "failed"
            if cleanup_error is not None:
                message += f"; source cleanup remains pending: {cleanup_error}"
                state = "cleanup_pending"
            output = self._prepare_materialization_failure(
                recv_req,
                message,
                session_state=state,
            )
            if cleanup_error is not None and local_source is not None:
                self._retain_materialization_cleanup_session(
                    recv_req,
                    local_source,
                    output,
                )
            return output

        failures = []
        for rank, item in enumerate(gathered):
            if not isinstance(item, dict):
                failures.append(f"rank {rank}: invalid capture status")
            elif not item.get("success", False):
                failures.append(f"rank {rank}: {item.get('message', 'capture failed')}")
        if failures:
            return self._prepare_failure_after_world_cleanup(
                recv_req,
                local_source,
                failures,
                operation="failed prepare source release",
            )

        assert local_source is not None
        merged_sources = None
        merge_error = None
        try:
            merged_sources = self._merge_materialization_sources(
                gathered,
                model_id=recv_req.model_id,
                revision=recv_req.revision,
                local_source=local_source,
            )
        except Exception as error:
            merge_error = str(error)

        try:
            merge_statuses = self._gather_weight_materialization_objects(
                {
                    "success": merge_error is None,
                    "message": merge_error or "Success.",
                },
                operation="prepare merge status",
            )
        except Exception as error:
            cleanup_error = self._release_materialization_source(local_source)
            message = str(error)
            if cleanup_error is not None:
                message += f"; source cleanup remains pending: {cleanup_error}"
            output = self._prepare_materialization_failure(
                recv_req,
                message,
                session_state=(
                    "cleanup_pending" if cleanup_error is not None else "failed"
                ),
            )
            if cleanup_error is not None:
                self._retain_materialization_cleanup_session(
                    recv_req,
                    local_source,
                    output,
                )
            return output

        merge_failures = []
        for rank, item in enumerate(merge_statuses):
            if not isinstance(item, dict):
                merge_failures.append(f"rank {rank}: invalid merge status")
            elif not item.get("success", False):
                merge_failures.append(
                    f"rank {rank}: {item.get('message', 'merge failed')}"
                )
        if merge_failures:
            return self._prepare_failure_after_world_cleanup(
                recv_req,
                local_source,
                merge_failures,
                operation="failed prepare merge source release",
            )

        assert merged_sources is not None
        (
            selected_placements,
            selected_bindings,
            selected_identity,
            local_placement_ids,
            generation,
            logical_digest,
        ) = merged_sources
        output = PrepareWeightMaterializationReqOutput(
            materialization_id=recv_req.materialization_id,
            success=True,
            message="Success.",
            external_dp_rank=self._external_dp_rank(),
            generation=generation,
            logical_payload_digest=logical_digest,
            total_bytes=sum(
                tensor.nbytes
                for placement in selected_placements
                for tensor in placement.tensors
            ),
            session_state="prepared",
        )
        session = _WeightMaterializationSession(
            request_identity=request_identity,
            source=local_source,
            selected_placements=selected_placements,
            selected_bindings=selected_bindings,
            selected_payload_identity=selected_identity,
            local_selected_placement_ids=local_placement_ids,
            prepare_output=output,
            state="prepared",
        )
        with self.weight_materialization_lock:
            existing = self.weight_materialization_sessions.setdefault(
                recv_req.materialization_id,
                session,
            )
        if existing is not session:
            cleanup_error = self._release_materialization_source(local_source)
            if existing.request_identity == request_identity and cleanup_error is None:
                return existing.prepare_output
            message = "materialization ID raced with another prepare request"
            if cleanup_error is not None:
                message += f"; source cleanup remains pending: {cleanup_error}"
            return self._prepare_materialization_failure(
                recv_req,
                message,
                session_state="conflict",
            )
        return output

    @staticmethod
    def _weight_storage_ref_builtins(ref: Any) -> dict[str, Any]:
        return {
            "provider": ref.provider,
            "storage_id": ref.storage_id,
            "manifest_key": ref.manifest_key,
            "manifest_digest": ref.manifest_digest,
        }

    def _record_materialization_commit(
        self,
        session: _WeightMaterializationSession,
        output: CommitWeightMaterializationReqOutput,
    ) -> CommitWeightMaterializationReqOutput:
        with self.weight_materialization_lock:
            session.state = output.session_state
            session.commit_output = output
            if (
                session.source is None
                and session.backend_owner is None
                and output.session_state
                not in {"cleanup_pending", "completion_unknown"}
            ):
                session.terminal_at = time.monotonic()
            self._prune_weight_materialization_sessions_locked()
        return output

    def _cleanup_weight_materialization_session(
        self,
        recv_req: CommitWeightMaterializationReqInput,
        session: _WeightMaterializationSession,
    ) -> CommitWeightMaterializationReqOutput:
        previous = session.commit_output
        if previous is not None and previous.completion_unknown:
            return previous
        source_errors, completion_unknown = self._release_materialization_source_world(
            session,
            operation="cleanup source release",
        )
        if source_errors:
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(source_errors),
                    session_state=(
                        "completion_unknown"
                        if completion_unknown
                        else "cleanup_pending"
                    ),
                    completion_unknown=completion_unknown,
                ),
            )
        backend_errors = self._close_materialization_backend_world(
            session,
            operation="cleanup Store backend close",
        )
        output = CommitWeightMaterializationReqOutput(
            materialization_id=recv_req.materialization_id,
            success=not backend_errors,
            message=("Success." if not backend_errors else " | ".join(backend_errors)),
            external_dp_rank=self._external_dp_rank(),
            selected=False,
            session_state="released" if not backend_errors else "cleanup_pending",
        )
        return self._record_materialization_commit(session, output)

    def commit_weight_materialization(
        self,
        recv_req: CommitWeightMaterializationReqInput,
    ) -> CommitWeightMaterializationReqOutput:
        """Materialize the selected external DP replica into Mooncake Store."""

        try:
            commit_identity = self._commit_request_identity(recv_req)
        except Exception as error:
            return self._commit_materialization_failure(recv_req, str(error))
        with self.weight_materialization_lock:
            self._prune_weight_materialization_sessions_locked()
            session = self.weight_materialization_sessions.get(
                recv_req.materialization_id
            )
            if session is None:
                return self._commit_materialization_failure(
                    recv_req,
                    "weight materialization session was not prepared",
                    session_state="not_found",
                )
            if session.commit_output is not None:
                if (
                    session.commit_identity == commit_identity
                    or recv_req.selected_external_dp_rank is None
                ):
                    if recv_req.selected_external_dp_rank is None:
                        pass
                    else:
                        return session.commit_output
                else:
                    return self._commit_materialization_failure(
                        recv_req,
                        "materialization ID is already bound to another destination",
                        session_state="conflict",
                    )
            elif (
                session.commit_identity is not None
                and session.commit_identity != commit_identity
                and recv_req.selected_external_dp_rank is not None
            ):
                return self._commit_materialization_failure(
                    recv_req,
                    "materialization ID is already bound to another destination",
                    session_state="conflict",
                )
            session.commit_identity = commit_identity
            session.state = "committing"

        if recv_req.selected_external_dp_rank is None:
            return self._cleanup_weight_materialization_session(recv_req, session)

        external_dp_rank = self._external_dp_rank()
        if external_dp_rank != recv_req.selected_external_dp_rank:
            source_error = self._release_materialization_session_source(session)
            output = CommitWeightMaterializationReqOutput(
                materialization_id=recv_req.materialization_id,
                success=source_error is None,
                message="Success." if source_error is None else source_error,
                external_dp_rank=external_dp_rank,
                selected=False,
                session_state=(
                    "skipped" if source_error is None else "cleanup_pending"
                ),
            )
            return self._record_materialization_commit(session, output)

        if (
            not session.selected_placements
            or not session.selected_bindings
            or session.selected_payload_identity is None
        ):
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    "weight materialization session is not fully prepared",
                ),
            )

        capacity_error = self._weight_storage_owner_capacity_error(
            current_materialization_id=recv_req.materialization_id,
        )
        if capacity_error is not None:
            cleanup_errors, completion_unknown = (
                self._release_materialization_source_world(
                    session,
                    operation="capacity rejection source release",
                )
            )
            errors = [capacity_error, *cleanup_errors]
            cleanup_pending = bool(cleanup_errors) or session.source is not None
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(errors),
                    session_state=(
                        "completion_unknown"
                        if completion_unknown
                        else ("cleanup_pending" if cleanup_pending else "failed")
                    ),
                    completion_unknown=completion_unknown,
                ),
            )

        owner = _WeightStorageBackendOwner()
        backend = None
        setup_error = None
        try:
            if session.source is None:
                raise RuntimeError("weight materialization source was already released")
            coordinator = TorchDistributedWeightStoreCoordinator(
                self._weight_materialization_collective_group()
            )
            spec = WeightSnapshotWriteSpec.from_mapping(recv_req.storage_options)
            backend = owner.enter_context(
                open_weight_snapshot_write_backend(
                    spec,
                    local_placement_ids=session.local_selected_placement_ids,
                    payload_checksum_verifier=session.source.payload_checksum,
                    coordinator=coordinator,
                )
            )
        except Exception as error:
            setup_error = str(error)

        try:
            setup_results = self._gather_weight_materialization_objects(
                {
                    "success": setup_error is None,
                    "message": setup_error or "Success.",
                },
                operation="Store backend setup",
            )
        except Exception as error:
            setup_results = [{"success": False, "message": str(error)}]
        setup_failures = []
        for rank, item in enumerate(setup_results):
            if not isinstance(item, dict):
                setup_failures.append(
                    f"rank {rank}: invalid Store backend setup status"
                )
            elif not item.get("success", False):
                setup_failures.append(
                    f"rank {rank}: {item.get('message', 'Store backend setup failed')}"
                )
        if setup_failures:
            session.backend_owner = owner
            cleanup_errors, completion_unknown = (
                self._release_materialization_source_world(
                    session,
                    operation="failed Store setup source release",
                )
            )
            setup_failures.extend(cleanup_errors)
            if not cleanup_errors:
                setup_failures.extend(
                    self._close_materialization_backend_world(
                        session,
                        operation="failed Store setup backend close",
                    )
                )
            cleanup_pending = (
                session.source is not None or session.backend_owner is not None
            )
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(setup_failures),
                    session_state=(
                        "completion_unknown"
                        if completion_unknown
                        else ("cleanup_pending" if cleanup_pending else "failed")
                    ),
                    completion_unknown=completion_unknown,
                ),
            )

        assert backend is not None
        publication = None
        materialization_error = None
        completion_unknown_error = None
        try:
            if session.local_selected_placement_ids:
                assert session.source is not None
                publication = materialize_distributed_runtime_weight_snapshot(
                    session.source,
                    global_placements=session.selected_placements,
                    global_bindings=session.selected_bindings,
                    payload_identity=session.selected_payload_identity,
                    destination=spec.destination,
                    provider=backend.provider,
                    catalog=backend.catalog,
                    publication_id=recv_req.materialization_id,
                    release_source=False,
                )
            else:
                assert session.source is not None
                attestor = _NoLocalRuntimeSourceAttestor.from_source(session.source)
                publication = materialize_weight_snapshot(
                    source_placements=session.selected_placements,
                    source_bindings=session.selected_bindings,
                    destination=spec.destination,
                    provider=backend.provider,
                    catalog=backend.catalog,
                    payload_identity=session.selected_payload_identity,
                    publication_id=recv_req.materialization_id,
                    attestor=attestor,
                )
        except WeightTransferCompletionUnknownError as error:
            completion_unknown_error = error
            materialization_error = str(error)
        except Exception as error:
            materialization_error = str(error)

        local_ref = (
            None
            if publication is None
            else self._weight_storage_ref_builtins(publication.snapshot.ref)
        )
        try:
            materialization_statuses = self._gather_weight_materialization_objects(
                {
                    "success": publication is not None,
                    "message": materialization_error or "Success.",
                    "completion_unknown": completion_unknown_error is not None,
                    "completion_ticket": (
                        None
                        if completion_unknown_error is None
                        else completion_unknown_error.completion_ticket
                    ),
                    "ref": local_ref,
                },
                operation="Store materialization outcome",
            )
        except Exception as error:
            session.backend_owner = owner
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    str(error),
                    session_state="completion_unknown",
                    completion_unknown=True,
                ),
            )

        status_errors = []
        success_refs = []
        completion_tickets = []
        unknown_outcome = False
        success_count = 0
        for rank, status in enumerate(materialization_statuses):
            if (
                not isinstance(status, dict)
                or type(status.get("success")) is not bool
                or type(status.get("completion_unknown")) is not bool
                or type(status.get("message")) is not str
                or (
                    status.get("completion_ticket") is not None
                    and type(status.get("completion_ticket")) is not str
                )
                or (
                    status.get("ref") is not None
                    and not isinstance(status.get("ref"), dict)
                )
            ):
                status_errors.append(
                    f"rank {rank}: invalid Store materialization status"
                )
                unknown_outcome = True
                continue
            if status["completion_unknown"]:
                unknown_outcome = True
                if status["completion_ticket"] is not None:
                    completion_tickets.append(status["completion_ticket"])
            if status["success"]:
                success_count += 1
                if status["ref"] is None:
                    status_errors.append(
                        f"rank {rank}: successful materialization has no ref"
                    )
                    unknown_outcome = True
                else:
                    success_refs.append(status["ref"])
            else:
                status_errors.append(f"rank {rank}: {status['message']}")

        if success_count and success_count != len(materialization_statuses):
            unknown_outcome = True
            status_errors.append(
                "model ranks disagree on Store materialization completion"
            )
        ref = success_refs[0] if success_refs else None
        if ref is not None and any(item != ref for item in success_refs):
            unknown_outcome = True
            status_errors.append("model ranks published different weight storage refs")

        if unknown_outcome:
            session.backend_owner = owner
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(status_errors)
                    or "Store materialization completion is unknown",
                    session_state="completion_unknown",
                    completion_unknown=True,
                    completion_ticket=(
                        completion_tickets[0] if completion_tickets else None
                    ),
                ),
            )

        if status_errors or ref is None:
            session.backend_owner = owner
            cleanup_errors, release_completion_unknown = (
                self._release_materialization_source_world(
                    session,
                    operation="failed materialization source release",
                )
            )
            status_errors.extend(cleanup_errors)
            if not cleanup_errors:
                status_errors.extend(
                    self._close_materialization_backend_world(
                        session,
                        operation="failed materialization backend close",
                    )
                )
            cleanup_pending = (
                session.source is not None or session.backend_owner is not None
            )
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(status_errors),
                    session_state=(
                        "completion_unknown"
                        if release_completion_unknown
                        else ("cleanup_pending" if cleanup_pending else "failed")
                    ),
                    completion_unknown=release_completion_unknown,
                ),
            )

        placement = session.selected_placements[0]
        storage_identity = commit_identity[1]
        assert isinstance(storage_identity, str)
        retain_error = self._retain_weight_storage_owner(
            model_id=placement.model_id,
            revision=placement.revision,
            storage_identity=storage_identity,
            ref=ref,
            owner=owner,
        )
        if retain_error is not None:
            session.backend_owner = owner
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    retain_error,
                    session_state="completion_unknown",
                    completion_unknown=True,
                ),
            )

        cleanup_errors, completion_unknown = self._release_materialization_source_world(
            session,
            operation="post-publication source release",
        )
        if cleanup_errors:
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(cleanup_errors),
                    session_state=(
                        "completion_unknown"
                        if completion_unknown
                        else "cleanup_pending"
                    ),
                    completion_unknown=completion_unknown,
                ),
            )
        return self._record_materialization_commit(
            session,
            CommitWeightMaterializationReqOutput(
                materialization_id=recv_req.materialization_id,
                success=True,
                message="Success.",
                external_dp_rank=external_dp_rank,
                selected=True,
                ref=ref,
                session_state="published",
            ),
        )

    def _defer_remote_instance_weight_transfer(self, operation, recv_req) -> None:
        if self.remote_weight_transfer_executor is None:
            self.remote_weight_transfer_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sglang-weight-transfer",
            )
        future = self.remote_weight_transfer_executor.submit(operation, recv_req)
        self.remote_weight_transfer_pending.append((future, recv_req))

    def _defer_weight_materialization(self, operation, recv_req) -> None:
        if self.weight_materialization_executor is None:
            self.weight_materialization_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sglang-weight-materialization",
            )
        future = self.weight_materialization_executor.submit(
            self._run_weight_materialization,
            operation,
            recv_req,
        )
        self.weight_materialization_pending.append((future, recv_req))

    def _run_weight_materialization(self, operation, recv_req):
        torch.distributed.barrier(group=self._weight_materialization_collective_group())
        return operation(recv_req)

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

    def defer_prepare_weight_materialization(
        self,
        recv_req: PrepareWeightMaterializationReqInput,
    ) -> None:
        self._defer_weight_materialization(
            self.prepare_weight_materialization,
            recv_req,
        )

    def defer_commit_weight_materialization(
        self,
        recv_req: CommitWeightMaterializationReqInput,
    ) -> None:
        self._defer_weight_materialization(
            self.commit_weight_materialization,
            recv_req,
        )

    def _remote_instance_weight_transfer_failure(self, recv_req, error: Exception):
        if isinstance(recv_req, PrepareWeightMaterializationReqInput):
            return self._prepare_materialization_failure(recv_req, str(error))
        if isinstance(recv_req, CommitWeightMaterializationReqInput):
            return self._commit_materialization_failure(recv_req, str(error))
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
        for pending_name in (
            "remote_weight_transfer_pending",
            "weight_materialization_pending",
        ):
            remaining = []
            for future, recv_req in getattr(self, pending_name):
                if not future.done():
                    remaining.append((future, recv_req))
                    continue
                try:
                    output = future.result()
                except Exception as error:
                    logger.exception("Remote instance weight transfer control failed")
                    output = self._remote_instance_weight_transfer_failure(
                        recv_req, error
                    )
                if isinstance(
                    recv_req,
                    (
                        PrepareWeightMaterializationReqInput,
                        CommitWeightMaterializationReqInput,
                    ),
                ):
                    try:
                        model_rank = torch.distributed.get_rank(
                            group=self._weight_materialization_collective_group()
                        )
                    except Exception:
                        model_rank = "unknown"
                    log = logger.info if output.success else logger.warning
                    log(
                        "Weight materialization phase completed on model rank %s: "
                        "id=%s state=%s success=%s message=%s",
                        model_rank,
                        recv_req.materialization_id,
                        output.session_state,
                        output.success,
                        output.message,
                    )
                completed.append((output, recv_req))
            setattr(self, pending_name, remaining)
        return completed

    def close_remote_instance_weight_transfer_executor(self) -> None:
        if self.remote_weight_transfer_executor is not None:
            self.remote_weight_transfer_executor.shutdown(wait=True)
            self.remote_weight_transfer_executor = None
        if self.weight_materialization_executor is not None:
            self.weight_materialization_executor.shutdown(wait=True)
            self.weight_materialization_executor = None

        with self.weight_materialization_lock:
            sessions = tuple(self.weight_materialization_sessions.values())
        for session in sessions:
            source_error = self._release_materialization_session_source(session)
            backend_error = self._close_materialization_backend(session)
            if source_error is not None:
                logger.warning(
                    "Weight materialization source cleanup failed during shutdown: %s",
                    source_error,
                )
            if backend_error is not None:
                logger.warning(
                    "Weight materialization backend cleanup failed during shutdown: %s",
                    backend_error,
                )
        with self.weight_materialization_lock:
            retained_owners = tuple(self.weight_storage_owners.values())
            self.weight_storage_owners.clear()
        for _ref, owner in retained_owners:
            try:
                owner.close()
            except Exception as error:
                logger.warning(
                    "Retained weight storage backend cleanup failed during "
                    "shutdown: %s",
                    error,
                )

    def begin_remote_instance_weight_transfer(
        self, recv_req: BeginRemoteInstanceWeightTransferReqInput
    ) -> BeginRemoteInstanceWeightTransferReqOutput:
        """Acquire one address-stable snapshot on every model rank."""
        collective_group = self.remote_weight_transfer_cpu_group or self.world_cpu_group
        local_snapshot = None
        local_lease_id = None
        local_generation = None
        cached = None
        split_manifest = recv_req.manifest_format == "placement_binding_v1"
        try:
            if recv_req.manifest_format not in ("runtime_v1", "placement_binding_v1"):
                raise RuntimeError(
                    f"unsupported source manifest format: {recv_req.manifest_format}"
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
                if split_manifest:
                    local_snapshot = self.tp_worker.model_runner.get_remote_instance_weight_runtime_manifest_parts(
                        model_id=recv_req.model_id,
                        revision=recv_req.revision,
                        transfer_id=recv_req.transfer_id,
                        lease_timeout_sec=recv_req.lease_timeout_sec,
                    )
                else:
                    local_snapshot = self.tp_worker.model_runner.get_remote_instance_weight_runtime_manifest(
                        model_id=recv_req.model_id,
                        revision=recv_req.revision,
                        transfer_id=recv_req.transfer_id,
                        lease_timeout_sec=recv_req.lease_timeout_sec,
                    )
                local_lease_id = self._remote_transfer_snapshot_lease_id(
                    local_snapshot,
                    split_manifest=split_manifest,
                )
                self._record_remote_weight_transfer_lease(
                    recv_req.transfer_id,
                    local_lease_id,
                    recv_req.lease_timeout_sec,
                )
                local_generation = self._remote_transfer_snapshot_generation(
                    local_snapshot,
                    split_manifest=split_manifest,
                )
                with self.remote_weight_transfer_lock:
                    self.remote_weight_transfer_generations[recv_req.transfer_id] = (
                        local_generation
                    )
                if split_manifest:
                    placement = (
                        local_snapshot["placement"]
                        if isinstance(local_snapshot, dict)
                        else local_snapshot.placement
                    )
                    binding = (
                        local_snapshot["binding"]
                        if isinstance(local_snapshot, dict)
                        else local_snapshot.binding
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
                        "placement": placement_payload,
                        "binding": binding_payload,
                    }
                else:
                    local_payload = (
                        local_snapshot
                        if isinstance(local_snapshot, dict)
                        else msgspec.to_builtins(local_snapshot)
                    )
                    local_result = {
                        "success": True,
                        "message": "Success.",
                        "session_state": "created",
                        "manifest": local_payload,
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
            message = f"Failed to gather source runtime manifests: {error}"
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
                    manifests=cached.manifests,
                    placements=cached.placements,
                    bindings=cached.bindings,
                )

        manifests = None
        placements = None
        bindings = None
        if not failures:
            try:
                if split_manifest:
                    placements = [item["placement"] for item in gathered]
                    bindings = [item["binding"] for item in gathered]
                    self._validate_remote_transfer_parts(
                        placements, bindings, world_size
                    )
                else:
                    manifests = [item["manifest"] for item in gathered]
                    self._validate_remote_transfer_manifests(manifests, world_size)
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
            manifests=manifests,
            placements=placements,
            bindings=bindings,
        )
        self._record_remote_weight_transfer_session(
            recv_req,
            local_lease_id,
            output,
            generation=local_generation,
        )
        return output

    @staticmethod
    def _remote_transfer_snapshot_lease_id(snapshot, *, split_manifest: bool) -> str:
        if split_manifest:
            binding = (
                snapshot["binding"] if isinstance(snapshot, dict) else snapshot.binding
            )
            return (
                binding["lease_id"] if isinstance(binding, dict) else binding.lease_id
            )
        return snapshot["lease_id"] if isinstance(snapshot, dict) else snapshot.lease_id

    @staticmethod
    def _remote_transfer_snapshot_generation(snapshot, *, split_manifest: bool) -> int:
        if split_manifest:
            binding = (
                snapshot["binding"] if isinstance(snapshot, dict) else snapshot.binding
            )
            return (
                binding["generation"]
                if isinstance(binding, dict)
                else binding.generation
            )
        return (
            snapshot["generation"]
            if isinstance(snapshot, dict)
            else snapshot.generation
        )

    @staticmethod
    def _remote_transfer_output_generation(
        output: BeginRemoteInstanceWeightTransferReqOutput,
    ) -> int | None:
        records = output.bindings or output.manifests or ()
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
                self.tp_worker.model_runner.renew_weight_runtime_manifest(
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
    def _validate_remote_transfer_manifests(manifests, world_size: int) -> None:
        if len(manifests) != world_size:
            raise RuntimeError(
                f"expected {world_size} source manifests, got {len(manifests)}"
            )
        if any(not manifest.get("tensors") for manifest in manifests):
            raise RuntimeError("every source rank must publish at least one tensor")

        worker_ids = {
            tensor["worker_id"]
            for manifest in manifests
            for tensor in manifest["tensors"][:1]
        }
        if len(worker_ids) != world_size:
            raise RuntimeError("source runtime manifest worker IDs are not unique")

        identities = {
            (
                manifest.get("model_id"),
                manifest.get("revision"),
                manifest.get("generation"),
            )
            for manifest in manifests
        }
        if len(identities) != 1:
            raise RuntimeError(
                "source runtime manifests do not describe one model generation"
            )

    @staticmethod
    def _validate_remote_transfer_parts(placements, bindings, world_size: int) -> None:
        if len(placements) != world_size or len(bindings) != world_size:
            raise RuntimeError(
                "source placement and binding counts must match the model world"
            )
        if any(not placement.get("tensors") for placement in placements):
            raise RuntimeError("every source rank must publish at least one tensor")
        if any(not binding.get("fragments") for binding in bindings):
            raise RuntimeError("every source rank must publish at least one binding")

        for placement, binding in zip(placements, bindings):
            placement_identity = (
                placement.get("model_id"),
                placement.get("revision"),
                placement.get("placement_id"),
            )
            binding_identity = (
                binding.get("model_id"),
                binding.get("revision"),
                binding.get("placement_id"),
            )
            if placement_identity != binding_identity:
                raise RuntimeError(
                    "source runtime binding does not match its placement"
                )

        worker_ids = []
        for binding in bindings:
            fragment_worker_ids = {
                fragment.get("worker_id") for fragment in binding["fragments"]
            }
            if None in fragment_worker_ids or len(fragment_worker_ids) != 1:
                raise RuntimeError(
                    "each source runtime binding must describe exactly one worker"
                )
            worker_ids.append(next(iter(fragment_worker_ids)))
        if len(set(worker_ids)) != world_size:
            raise RuntimeError("source runtime binding worker IDs are not unique")

        identities = {
            (
                placement.get("model_id"),
                placement.get("revision"),
                binding.get("generation"),
            )
            for placement, binding in zip(placements, bindings)
        }
        if len(identities) != 1:
            raise RuntimeError(
                "source placement and bindings do not describe one model generation"
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
                self.tp_worker.model_runner.release_weight_runtime_manifest(lease_id)
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
            commit_revision=False,
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
                self._assert_weight_cache_inactive("release_memory_occupation")
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
            commit_revision=True,
        ):
            for tag in tags:
                self.offload_tags.remove(tag)

            if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
                self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

            if GPU_MEMORY_TYPE_WEIGHTS in tags:
                self._assert_weight_cache_inactive("resume_memory_occupation")
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
