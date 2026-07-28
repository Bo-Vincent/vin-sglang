from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from secrets import token_urlsafe
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
    CommitWeightMaterializationReqInput,
    CommitWeightMaterializationReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetRemoteInstanceWeightTransferSessionReqInput,
    GetRemoteInstanceWeightTransferSessionReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
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
    WeightMaterializationSessionState,
    WeightSnapshotActivationReqInput,
    WeightSnapshotActivationReqOutput,
)
from sglang.srt.managers.weight_materialization import (
    blocks_unresolved_materialization_capacity,
    is_published_materialization_state,
    is_retryable_materialization_state,
    is_terminal_materialization_state,
)
from sglang.srt.model_executor.model_runner_components.weight_update_coordination import (
    observe_weight_update_mutation,
)
from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightManifestError,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from sglang.srt.weight_transfer.api import (
    materialize_weight_snapshot_candidate,
    preflight_weight_transfer,
    prepare_weight_materialization,
    publish_weight_snapshot,
)
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
    WeightTransferError,
    WeightTransferExecutionContext,
    WeightTransferReleaseError,
    WeightTransferTerminalProof,
    WeightTransferTerminalStatus,
)
from sglang.srt.weight_transfer.remote_protocol import (
    HF_REVISION_V1,
    PLACEMENT_BINDING_V1,
    validate_manifest_revision_semantics,
)
from sglang.srt.weight_transfer.runtime import (
    RuntimeWeightSnapshotSource,
)
from sglang.srt.weight_transfer.store_runtime import (
    WeightSnapshotBackendStatus,
    WeightSnapshotWriteSpec,
    open_weight_snapshot_write_backend,
)

logger = logging.getLogger(__name__)

_REMOTE_WEIGHT_TRANSFER_TOMBSTONE_TTL_SEC = 300.0
_REMOTE_WEIGHT_TRANSFER_TOMBSTONE_LIMIT = 4096
_REMOTE_WEIGHT_TRANSFER_BEGIN_FENCE_PREFIX = "begin-v1:"
_REMOTE_WEIGHT_TRANSFER_LEASE_FENCE_PREFIX = "lease-v1:"
_WEIGHT_MATERIALIZATION_TERMINAL_LIMIT = 4096
_WEIGHT_MATERIALIZATION_UNRESOLVED_LIMIT = 64
_WEIGHT_MATERIALIZATION_COLLECTIVE_GRACE_SEC = 1.0
_WEIGHT_MATERIALIZATION_CONTROL_TIMEOUT_SEC = 30.0
_WEIGHT_MATERIALIZATION_SHUTDOWN_TIMEOUT_SEC = 1.0
_WEIGHT_MATERIALIZATION_MAX_RANK_SOURCE_RECORDS = 1_000_000
_WEIGHT_MATERIALIZATION_MAX_WORLD_SOURCE_RECORDS = 10_000_000
_REMOTE_WEIGHT_TRANSFER_CONTROL_TIMEOUT_SEC = 30.0


@dataclass(frozen=True, slots=True)
class _WeightMutationResult:
    success: bool
    message: str
    mutated_any: bool


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
            if not bool(getattr(error, "completion_unknown", False)):
                self.terminal_error = str(error)
            raise
        self.closed = True


@dataclass(slots=True)
class _WeightMaterializationSession:
    request_identity: tuple[str, str]
    deadline_unix_sec: float
    source: RuntimeWeightSnapshotSource | None
    selected_placements: tuple[WeightPlacementManifest, ...]
    selected_bindings: tuple[WeightRuntimeBindingManifest, ...]
    selected_payload_identity: WeightPayloadIdentity | None
    local_selected_placement_ids: tuple[str, ...]
    prepare_output: PrepareWeightMaterializationReqOutput
    state: WeightMaterializationSessionState
    commit_identity: tuple[int | None, str | None] | None = None
    commit_output: CommitWeightMaterializationReqOutput | None = None
    cleanup_output: CommitWeightMaterializationReqOutput | None = None
    publication_ref: dict[str, Any] | None = None
    provider_finalize_pending: bool = False
    backend_owner: _WeightStorageBackendOwner | None = None
    backend: Any | None = None
    backend_close_succeeded: bool = False
    backend_completion_unknown: bool = False
    backend_owner_close_pending: bool = False
    terminal_at: float | None = None


@dataclass(frozen=True, slots=True)
class _MaterializationSourceSummary:
    model_id: str
    revision: str
    generation: int
    logical_payload_digest: str
    total_bytes: int
    placement_count: int
    fragment_count: int


@dataclass(frozen=True, slots=True)
class _RankMaterializationSourceSelection:
    placements: tuple[WeightPlacementManifest, ...]
    bindings: tuple[WeightRuntimeBindingManifest, ...]
    payload_identity: WeightPayloadIdentity
    local_placement_ids: tuple[str, ...]
    summary: _MaterializationSourceSummary


@dataclass(frozen=True, slots=True)
class _MaterializationSourceDistribution:
    rank_selections: tuple[_RankMaterializationSourceSelection, ...]


@dataclass(frozen=True, slots=True)
class _MaterializationSourceProjection:
    placements: tuple[WeightPlacementManifest, ...]
    bindings: tuple[WeightRuntimeBindingManifest, ...]
    selected_placements: tuple[WeightPlacementManifest, ...]
    selected_bindings: tuple[WeightRuntimeBindingManifest, ...]
    local_placement_ids: tuple[tuple[str, ...], ...]
    generation: int


@dataclass(frozen=True, slots=True)
class _MaterializationBackendContext:
    owner: _WeightStorageBackendOwner
    backend: Any
    spec: WeightSnapshotWriteSpec


@dataclass(frozen=True, slots=True)
class _MaterializationPreflightContext:
    owner: _WeightStorageBackendOwner
    backend: Any
    request: Any
    preflight: Any


@dataclass(frozen=True, slots=True)
class _LocalMaterializationOutcome:
    candidate: Any | None
    message: str
    finalize_pending: bool = False
    completion_unknown: bool = False
    completion_ticket: str | None = None


@dataclass(frozen=True, slots=True)
class _ReducedMaterializationOutcome:
    errors: tuple[str, ...]
    ref: dict[str, Any] | None
    completion_ticket: str | None
    completion_unknown: bool
    finalize_pending_messages: tuple[str, ...]
    all_finalize_pending: bool


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


@dataclass(frozen=True, slots=True)
class _SelectedRuntimeSourceAttestor:
    source: RuntimeWeightSnapshotSource
    request_binding: WeightRuntimeBindingManifest

    def attest(self, request: Any) -> None:
        self.source.attest(
            request,
            request_binding=self.request_binding,
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
    remote_weight_transfer_control_cpu_group: Any = None
    weight_materialization_cpu_group: Any = None
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    remote_weight_transfer_leases: Dict[str, str] = field(default_factory=dict)
    remote_weight_transfer_deadlines: Dict[str, float] = field(default_factory=dict)
    remote_weight_transfer_generations: Dict[str, int] = field(default_factory=dict)
    remote_weight_transfer_fences: Dict[str, str] = field(default_factory=dict)
    remote_weight_transfer_begin_fences: Dict[str, str] = field(default_factory=dict)
    remote_weight_transfer_consumed_begin_fences: Dict[str, Tuple[str, float]] = field(
        default_factory=dict
    )
    remote_weight_transfer_expired: set[str] = field(default_factory=set)
    remote_weight_transfer_snapshot_poisoned: str | None = None
    remote_weight_transfer_inflight_begins: Dict[str, int] = field(default_factory=dict)
    remote_weight_transfer_sessions: Dict[
        str,
        Tuple[
            Tuple[Any, ...],
            BeginRemoteInstanceWeightTransferReqOutput | None,
        ],
    ] = field(default_factory=dict)
    remote_weight_transfer_tombstones: Dict[
        str,
        Tuple[Optional[Tuple[Any, ...]], float],
    ] = field(default_factory=dict)
    remote_weight_transfer_tombstone_fences: Dict[str, str] = field(
        default_factory=dict
    )
    remote_weight_transfer_tombstone_generations: Dict[str, int] = field(
        default_factory=dict
    )
    remote_weight_transfer_legacy_reuse_blocked: set[str] = field(default_factory=set)
    remote_weight_transfer_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    remote_weight_transfer_executor: Optional[ThreadPoolExecutor] = field(
        default=None, init=False, repr=False
    )
    remote_weight_transfer_control_executor: Optional[ThreadPoolExecutor] = field(
        default=None, init=False, repr=False
    )
    remote_weight_transfer_snapshot_coordinator: Any = field(
        default=None, init=False, repr=False
    )
    remote_weight_transfer_control_coordinator: Any = field(
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
    weight_materialization_accepting: bool = field(default=True, init=False, repr=False)
    weight_materialization_poisoned: str | None = field(
        default=None, init=False, repr=False
    )
    weight_materialization_coordinator: Any | None = field(
        default=None, init=False, repr=False
    )
    weight_materialization_execution_context: WeightTransferExecutionContext | None = (
        field(default=None, init=False, repr=False)
    )
    weight_materialization_cancel_signal: threading.Event | None = field(
        default=None, init=False, repr=False
    )
    weight_materialization_cancel_signals: set[threading.Event] = field(
        default_factory=set, init=False, repr=False
    )
    weight_materialization_active_id: str | None = field(
        default=None, init=False, repr=False
    )
    weight_materialization_sessions: Dict[
        str,
        _WeightMaterializationSession,
    ] = field(default_factory=dict)
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
        body_error = None
        try:
            yield
        except BaseException as error:
            body_error = error

        local_success = body_error is None
        local_message = "Success." if local_success else str(body_error)
        try:
            generation = coordinator.finish_update(token, success=local_success)
        except BaseException as error:
            generation = coordinator.generation
            local_success = False
            local_message = f"{type(error).__name__}: {error}"

        try:
            world_success, world_message, outcomes = self._gather_weight_update_outcome(
                success=local_success,
                message=local_message,
                phase="weight memory transition mutation",
                mutated_any=True,
                generations=(generation,),
            )
        except Exception as error:
            self._poison_weight_update_best_effort(((coordinator, generation),))
            if body_error is not None:
                raise body_error from error
            raise WeightManifestError(
                f"failed to gather weight memory transition outcomes: {error}"
            ) from error

        generations_match = self._outcome_generations_match(outcomes)
        should_commit = world_success and generations_match and commit_revision
        try:
            if should_commit:
                coordinator.commit_revision(expected_generation=generation)
            elif not world_success or not generations_match:
                coordinator.poison_global_update_failure(expected_generation=generation)
            local_finalize_success = True
            local_finalize_message = "Success."
        except Exception as error:
            local_finalize_success = False
            local_finalize_message = f"{type(error).__name__}: {error}"

        try:
            finalize_success, finalize_message, _ = self._gather_weight_update_outcome(
                success=local_finalize_success,
                message=local_finalize_message,
                phase="weight memory transition finalize",
            )
        except Exception as error:
            self._poison_weight_update_best_effort(((coordinator, generation),))
            if body_error is not None:
                raise body_error from error
            raise WeightManifestError(
                f"failed to gather weight memory transition finalize outcomes: {error}"
            ) from error

        if not finalize_success:
            self._poison_weight_update_best_effort(((coordinator, generation),))
        if body_error is not None:
            raise body_error
        if not world_success:
            raise WeightManifestError(world_message)
        if not generations_match:
            raise WeightManifestError(
                "weight memory transition generations differ across ranks"
            )
        if not finalize_success:
            raise WeightManifestError(finalize_message)

    def _prune_remote_weight_transfer_bookkeeping(self) -> None:
        now = time.monotonic()
        now_unix_sec = time.time()
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
                self.remote_weight_transfer_tombstone_fences.pop(transfer_id, None)
                self.remote_weight_transfer_tombstone_generations.pop(transfer_id, None)
            expired_begin_fences = [
                lease_fence
                for lease_fence, (_, deadline_unix_sec) in (
                    self.remote_weight_transfer_consumed_begin_fences.items()
                )
                if deadline_unix_sec <= now_unix_sec
            ]
            for lease_fence in expired_begin_fences:
                self.remote_weight_transfer_consumed_begin_fences.pop(lease_fence, None)

    def _consume_remote_weight_transfer_begin_fence(
        self,
        *,
        transfer_id: str,
        lease_fence: str | None,
        deadline_unix_sec: float,
        active: bool,
    ) -> bool:
        if lease_fence is None:
            return True
        with self.remote_weight_transfer_lock:
            consumed = self.remote_weight_transfer_consumed_begin_fences.get(
                lease_fence
            )
            if consumed is not None:
                owner_transfer_id, _ = consumed
                return (
                    active
                    and owner_transfer_id == transfer_id
                    and self.remote_weight_transfer_begin_fences.get(transfer_id)
                    == lease_fence
                )
            self.remote_weight_transfer_consumed_begin_fences[lease_fence] = (
                transfer_id,
                deadline_unix_sec,
            )
            return True

    def _poison_remote_weight_transfer_snapshot_lane(self, message: str) -> None:
        with self.remote_weight_transfer_lock:
            if self.remote_weight_transfer_snapshot_poisoned is None:
                self.remote_weight_transfer_snapshot_poisoned = message

    def _remote_weight_transfer_snapshot_poison(self) -> str | None:
        with self.remote_weight_transfer_lock:
            return self.remote_weight_transfer_snapshot_poisoned

    def _get_remote_weight_transfer_lease(self, transfer_id: str) -> str | None:
        self._prune_remote_weight_transfer_bookkeeping()
        with self.remote_weight_transfer_lock:
            return self.remote_weight_transfer_leases.get(transfer_id)

    def list_remote_instance_weight_transfer_sessions(self) -> List[Dict[str, Any]]:
        self._prune_remote_weight_transfer_bookkeeping()
        with self.remote_weight_transfer_lock:
            sessions = []
            for transfer_id, lease_id in sorted(
                self.remote_weight_transfer_leases.items()
            ):
                session = {
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
                lease_fence = self.remote_weight_transfer_fences.get(transfer_id)
                if lease_fence is not None:
                    session["lease_fence"] = lease_fence
                sessions.append(session)
            return sessions

    @staticmethod
    def _remote_weight_transfer_request_identity(
        request: BeginRemoteInstanceWeightTransferReqInput,
    ) -> Tuple[Any, ...]:
        return (
            request.model_id,
            request.revision,
            request.lease_timeout_sec,
            request.manifest_format,
            request.manifest_revision_semantics,
        )

    def _cached_remote_weight_transfer_session(
        self,
        request: BeginRemoteInstanceWeightTransferReqInput,
    ) -> (
        Tuple[
            Tuple[Any, ...],
            BeginRemoteInstanceWeightTransferReqOutput | None,
        ]
        | None
    ):
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
        identity, _ = cached
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
        return cached

    def _record_remote_weight_transfer_session(
        self,
        request: BeginRemoteInstanceWeightTransferReqInput,
        lease_id: str,
        output: BeginRemoteInstanceWeightTransferReqOutput | None,
        generation: int | None = None,
    ) -> None:
        if generation is None and output is not None:
            generation = self._remote_transfer_output_generation(output)
        with self.remote_weight_transfer_lock:
            lease_fence = (
                output.lease_fence
                if output is not None and output.lease_fence is not None
                else self.remote_weight_transfer_fences.get(request.transfer_id)
            )
            self.remote_weight_transfer_leases[request.transfer_id] = lease_id
            self.remote_weight_transfer_deadlines.setdefault(
                request.transfer_id,
                time.monotonic() + request.lease_timeout_sec,
            )
            if generation is not None:
                self.remote_weight_transfer_generations[request.transfer_id] = (
                    generation
                )
            if lease_fence is not None:
                self.remote_weight_transfer_fences[request.transfer_id] = lease_fence
            self.remote_weight_transfer_expired.discard(request.transfer_id)
            self.remote_weight_transfer_sessions[request.transfer_id] = (
                self._remote_weight_transfer_request_identity(request),
                output,
            )
            self.remote_weight_transfer_tombstones.pop(request.transfer_id, None)
            self.remote_weight_transfer_tombstone_fences.pop(request.transfer_id, None)
            self.remote_weight_transfer_tombstone_generations.pop(
                request.transfer_id, None
            )

    def _record_remote_weight_transfer_lease(
        self,
        transfer_id: str,
        lease_id: str,
        lease_timeout_sec: int,
        *,
        generation: int | None = None,
        lease_fence: str | None = None,
        begin_fence: str | None = None,
    ) -> None:
        with self.remote_weight_transfer_lock:
            self.remote_weight_transfer_leases[transfer_id] = lease_id
            self.remote_weight_transfer_deadlines[transfer_id] = (
                time.monotonic() + lease_timeout_sec
            )
            if generation is not None:
                self.remote_weight_transfer_generations[transfer_id] = generation
            if lease_fence is not None:
                self.remote_weight_transfer_fences[transfer_id] = lease_fence
            if begin_fence is not None:
                self.remote_weight_transfer_begin_fences[transfer_id] = begin_fence
            self.remote_weight_transfer_expired.discard(transfer_id)

    def _complete_remote_weight_transfer_session(self, transfer_id: str) -> None:
        now = time.monotonic()
        with self.remote_weight_transfer_lock:
            lease_fence = self.remote_weight_transfer_fences.pop(transfer_id, None)
            self.remote_weight_transfer_begin_fences.pop(transfer_id, None)
            generation = self.remote_weight_transfer_generations.pop(transfer_id, None)
            self.remote_weight_transfer_leases.pop(transfer_id, None)
            self.remote_weight_transfer_deadlines.pop(transfer_id, None)
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
            self.remote_weight_transfer_legacy_reuse_blocked.add(transfer_id)
            if lease_fence is not None:
                self.remote_weight_transfer_tombstone_fences[transfer_id] = lease_fence
            if generation is not None:
                self.remote_weight_transfer_tombstone_generations[transfer_id] = (
                    generation
                )
            while (
                len(self.remote_weight_transfer_tombstones)
                > _REMOTE_WEIGHT_TRANSFER_TOMBSTONE_LIMIT
            ):
                oldest_transfer_id = next(iter(self.remote_weight_transfer_tombstones))
                self.remote_weight_transfer_tombstones.pop(oldest_transfer_id)
                self.remote_weight_transfer_tombstone_fences.pop(
                    oldest_transfer_id, None
                )
                self.remote_weight_transfer_tombstone_generations.pop(
                    oldest_transfer_id, None
                )

    def _mark_remote_weight_transfer_begin(self, transfer_id: str) -> None:
        with self.remote_weight_transfer_lock:
            self.remote_weight_transfer_inflight_begins[transfer_id] = (
                self.remote_weight_transfer_inflight_begins.get(transfer_id, 0) + 1
            )

    def _finish_remote_weight_transfer_begin(self, transfer_id: str) -> None:
        with self.remote_weight_transfer_lock:
            remaining = self.remote_weight_transfer_inflight_begins.get(transfer_id, 0)
            if remaining <= 1:
                self.remote_weight_transfer_inflight_begins.pop(transfer_id, None)
            else:
                self.remote_weight_transfer_inflight_begins[transfer_id] = remaining - 1

    def _remote_weight_transfer_begin_is_inflight(self, transfer_id: str) -> bool:
        with self.remote_weight_transfer_lock:
            return self.remote_weight_transfer_inflight_begins.get(transfer_id, 0) > 0

    def _discard_remote_weight_transfer_lease(self, transfer_id: str) -> None:
        with self.remote_weight_transfer_lock:
            self.remote_weight_transfer_leases.pop(transfer_id, None)
            self.remote_weight_transfer_deadlines.pop(transfer_id, None)
            self.remote_weight_transfer_generations.pop(transfer_id, None)
            self.remote_weight_transfer_fences.pop(transfer_id, None)
            self.remote_weight_transfer_begin_fences.pop(transfer_id, None)
            self.remote_weight_transfer_expired.discard(transfer_id)
            self.remote_weight_transfer_sessions.pop(transfer_id, None)

    def _remote_weight_transfer_control_identity_error(
        self,
        *,
        transfer_id: str,
        lease_fence: str | None,
        generation: int | None,
        allow_begin_fence: bool = False,
    ) -> str | None:
        with self.remote_weight_transfer_lock:
            actual_fence = self.remote_weight_transfer_fences.get(transfer_id)
            actual_begin_fence = self.remote_weight_transfer_begin_fences.get(
                transfer_id
            )
            begin_fence_record = (
                None
                if actual_begin_fence is None
                else self.remote_weight_transfer_consumed_begin_fences.get(
                    actual_begin_fence
                )
            )
            actual_generation = self.remote_weight_transfer_generations.get(transfer_id)
            legacy_reuse_blocked = (
                transfer_id in self.remote_weight_transfer_legacy_reuse_blocked
            )
            if transfer_id not in self.remote_weight_transfer_leases:
                actual_fence = self.remote_weight_transfer_tombstone_fences.get(
                    transfer_id
                )
                actual_begin_fence = None
                begin_fence_record = None
                actual_generation = (
                    self.remote_weight_transfer_tombstone_generations.get(transfer_id)
                )
        if (
            legacy_reuse_blocked
            and actual_fence is None
            and actual_generation is None
            and lease_fence is None
            and generation is None
            and transfer_id in self.remote_weight_transfer_leases
        ):
            return "Remote weight transfer lease fence or generation is required after ID reuse."
        if actual_fence is None:
            if lease_fence is not None:
                return "Remote weight transfer lease fence does not match."
            if (
                generation is not None
                and actual_generation is not None
                and generation != actual_generation
            ):
                return "Remote weight transfer generation does not match."
            return None
        if lease_fence != actual_fence and not (
            allow_begin_fence
            and lease_fence == actual_begin_fence
            and begin_fence_record is not None
            and begin_fence_record[0] == transfer_id
            and begin_fence_record[1] > time.time()
        ):
            return "Remote weight transfer lease fence does not match."
        if generation != actual_generation:
            return "Remote weight transfer generation does not match."
        return None

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
    def _weight_snapshot_coordinators(worker) -> tuple:
        if worker is None:
            return ()
        inner = getattr(worker, "_draft_worker", None)
        runners = [
            getattr(worker, "model_runner", None),
            _get_draft_model_runner(worker),
            getattr(getattr(worker, "target_worker", None), "model_runner", None),
            getattr(getattr(worker, "draft_worker", None), "model_runner", None),
            getattr(getattr(worker, "draft_worker", None), "draft_runner", None),
            getattr(getattr(inner, "target_worker", None), "model_runner", None),
            getattr(getattr(inner, "draft_worker", None), "model_runner", None),
            getattr(inner, "draft_runner", None),
        ]
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
            for coordinator in self._weight_snapshot_coordinators(worker):
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
                current_generation = coordinator.generation
                if current_generation == expected_generation:
                    logger.exception("Failed to poison a partial weight update")
                    continue
                try:
                    coordinator.poison_global_update_failure(
                        expected_generation=current_generation
                    )
                except Exception:
                    logger.exception("Failed to poison a partial weight update")

    def _gather_weight_update_outcome(
        self,
        *,
        success: bool,
        message: str,
        phase: str,
        mutated_any: bool = False,
        generations: tuple[int, ...] = (),
    ) -> Tuple[bool, str, list]:
        local_outcome = {"success": bool(success), "message": str(message)}
        if phase.endswith("mutation"):
            local_outcome["mutated_any"] = bool(mutated_any)
            local_outcome["generations"] = list(generations)
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
            return False, " | ".join(failures), gathered
        return True, "Success.", gathered

    @staticmethod
    def _outcome_generations_match(outcomes) -> bool:
        generation_vectors = []
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                return False
            generations = outcome.get("generations")
            if not isinstance(generations, list) or any(
                isinstance(generation, bool)
                or not isinstance(generation, int)
                or generation <= 0
                for generation in generations
            ):
                return False
            generation_vectors.append(tuple(generations))
        return len(set(generation_vectors)) <= 1

    def _invoke_weight_mutation(self, worker, mutate) -> _WeightMutationResult:
        coordinators = self._weight_snapshot_coordinators(worker)
        initial_generations = tuple(
            coordinator.generation for coordinator in coordinators
        )
        with observe_weight_update_mutation() as observation:
            try:
                result = mutate()
                success, message = result
                if not isinstance(success, bool):
                    raise TypeError("weight update success outcome must be a boolean")
                message = str(message)
            except Exception as error:
                success = False
                message = f"{type(error).__name__}: {error}"

        if not coordinators:
            mutated_any = observation.mutation_started
        else:
            advanced = tuple(
                coordinator.generation != initial_generation
                for coordinator, initial_generation in zip(
                    coordinators,
                    initial_generations,
                    strict=True,
                )
            )
            mutated_any = any(advanced)
            if success and not all(advanced):
                success = False
                message = "weight update did not advance every runtime generation"
        return _WeightMutationResult(
            success=success,
            message=message,
            mutated_any=mutated_any,
        )

    def _run_weight_update_transaction(
        self,
        *,
        operation: str,
        mutate: Callable[[], _WeightMutationResult],
        workers,
        recv_req,
    ) -> Tuple[bool, str, bool]:
        try:
            initial_generation_mapping = self._capture_weight_update_generations(
                workers
            )
            local_reservation_success = True
            local_reservation_message = "Success."
        except Exception as error:
            initial_generation_mapping = ()
            local_reservation_success = False
            local_reservation_message = f"{type(error).__name__}: {error}"

        try:
            reservation_success, reservation_message, _ = (
                self._gather_weight_update_outcome(
                    success=local_reservation_success,
                    message=local_reservation_message,
                    phase=f"{operation} reservation",
                )
            )
        except Exception as error:
            return (
                False,
                f"Failed to gather {operation} reservation outcomes: "
                f"{type(error).__name__}: {error}",
                False,
            )
        if not reservation_success:
            return False, reservation_message, False

        try:
            local_result = mutate()
            if not isinstance(local_result, _WeightMutationResult):
                raise TypeError("weight mutation must return _WeightMutationResult")
        except Exception as error:
            local_result = _WeightMutationResult(
                success=False,
                message=f"{type(error).__name__}: {error}",
                mutated_any=True,
            )

        try:
            generation_mapping = self._capture_weight_update_generations(workers)
            initial_generations = {
                id(coordinator): generation
                for coordinator, generation in initial_generation_mapping
            }
            generation_changed = any(
                initial_generations.get(id(coordinator)) != generation
                for coordinator, generation in generation_mapping
            )
            local_result = _WeightMutationResult(
                success=local_result.success,
                message=local_result.message,
                mutated_any=local_result.mutated_any or generation_changed,
            )
        except Exception as error:
            generation_mapping = ()
            local_result = _WeightMutationResult(
                success=False,
                message=(
                    f"{local_result.message} | failed to capture weight update "
                    f"generation: {type(error).__name__}: {error}"
                ),
                mutated_any=True,
            )

        if local_result.mutated_any:
            try:
                self.flush_cache_after_weight_update(recv_req)
            except Exception as error:
                local_result = _WeightMutationResult(
                    success=False,
                    message=f"{type(error).__name__}: {error}",
                    mutated_any=True,
                )

        try:
            mutation_success, mutation_message, mutation_outcomes = (
                self._gather_weight_update_outcome(
                    success=local_result.success,
                    message=local_result.message,
                    phase=f"{operation} mutation",
                    mutated_any=local_result.mutated_any,
                    generations=tuple(
                        generation for _, generation in generation_mapping
                    ),
                )
            )
        except Exception as error:
            self._poison_weight_update_best_effort(generation_mapping)
            return (
                False,
                f"Failed to gather {operation} mutation outcomes: "
                f"{type(error).__name__}: {error}",
                True,
            )

        global_mutated_any = any(
            not isinstance(outcome, dict) or outcome.get("mutated_any") is not False
            for outcome in mutation_outcomes
        )
        generations_match = self._outcome_generations_match(mutation_outcomes)
        if mutation_success and not generations_match:
            local_ready_success = False
            local_ready_message = "weight update generations differ across ranks"
        else:
            try:
                if mutation_success:
                    self._validate_weight_update_generations(
                        generation_mapping,
                        require_pending_revision=True,
                    )
                local_ready_success = True
                local_ready_message = "Success."
            except Exception as error:
                local_ready_success = False
                local_ready_message = f"{type(error).__name__}: {error}"

        try:
            ready_success, ready_message, _ = self._gather_weight_update_outcome(
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
                global_mutated_any,
            )

        try:
            if mutation_success and ready_success:
                self._finalize_weight_update(generation_mapping, commit=True)
            elif global_mutated_any:
                self._finalize_weight_update(generation_mapping, commit=False)
            local_finalize_success = True
            local_finalize_message = "Success."
        except Exception as error:
            local_finalize_success = False
            local_finalize_message = f"{type(error).__name__}: {error}"

        try:
            finalize_success, finalize_message, _ = self._gather_weight_update_outcome(
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
                global_mutated_any,
            )

        if not finalize_success:
            self._poison_weight_update_best_effort(generation_mapping)
        if global_mutated_any and (not mutation_success or not ready_success):
            self._poison_weight_update_best_effort(generation_mapping)
        if not mutation_success:
            return False, mutation_message, global_mutated_any
        if not ready_success:
            return False, ready_message, global_mutated_any
        if not finalize_success:
            return False, finalize_message, global_mutated_any
        return True, local_result.message, False

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            workers = [self.tp_worker]
            if self.draft_worker is not None:
                workers.append(self.draft_worker)

            def mutate():
                target = self._invoke_weight_mutation(
                    self.tp_worker,
                    lambda: self.tp_worker.update_weights_from_disk(recv_req),
                )
                if not target.success or self.draft_worker is None:
                    return target
                draft = self._invoke_weight_mutation(
                    self.draft_worker,
                    lambda: self.draft_worker.update_weights_from_disk(recv_req),
                )
                return _WeightMutationResult(
                    success=draft.success,
                    message=draft.message,
                    mutated_any=target.mutated_any or draft.mutated_any,
                )

            success, message, fail_closed = self._run_weight_update_transaction(
                operation="disk weight update",
                mutate=mutate,
                workers=workers,
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success,
                message=message,
                num_paused_requests=0,
                fail_closed=fail_closed,
                request_id=recv_req.request_id,
                external_dp_rank=self._external_dp_rank(),
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
            success, message, fail_closed = self._run_weight_update_transaction(
                operation="distributed weight update",
                mutate=lambda: self._invoke_weight_mutation(
                    self.tp_worker,
                    lambda: self.tp_worker.update_weights_from_distributed(recv_req),
                ),
                workers=(self.tp_worker,),
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(
                success=success,
                message=message,
                fail_closed=fail_closed,
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        with self._observe_weight_load("tensor"):
            if recv_req.disable_draft_model:
                worker = self.tp_worker
            else:
                worker = self.draft_worker or self.tp_worker
            success, message, fail_closed = self._run_weight_update_transaction(
                operation="tensor weight update",
                mutate=lambda: self._invoke_weight_mutation(
                    worker,
                    lambda: worker.update_weights_from_tensor(recv_req),
                ),
                workers=(worker,),
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightsFromTensorReqOutput(
                success=success,
                message=message,
                fail_closed=fail_closed,
            )

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            workers = [self.tp_worker]
            if self.draft_worker is not None:
                workers.append(self.draft_worker)

            def mutate():
                target = self._invoke_weight_mutation(
                    self.tp_worker,
                    lambda: self.tp_worker.update_weights_from_ipc(recv_req),
                )
                if not target.success or self.draft_worker is None:
                    return target
                draft = self._invoke_weight_mutation(
                    self.draft_worker,
                    lambda: self.draft_worker.update_weights_from_ipc(recv_req),
                )
                return _WeightMutationResult(
                    success=draft.success,
                    message=draft.message,
                    mutated_any=target.mutated_any or draft.mutated_any,
                )

            success, message, fail_closed = self._run_weight_update_transaction(
                operation="IPC weight update",
                mutate=mutate,
                workers=workers,
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightsFromIPCReqOutput(
                success=success,
                message=message,
                fail_closed=fail_closed,
            )

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

    @staticmethod
    def _validate_materialization_deadline(value: object) -> float:
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                "weight materialization deadline must be a positive finite number"
            )
        return float(value)

    def _prepare_materialization_deadline(
        self,
        recv_req: PrepareWeightMaterializationReqInput,
    ) -> float:
        value = getattr(recv_req, "deadline_unix_sec", None)
        if value is None:
            value = time.time() + recv_req.lease_timeout_sec
        return self._validate_materialization_deadline(value)

    def _commit_materialization_deadline(
        self,
        recv_req: CommitWeightMaterializationReqInput,
        session: _WeightMaterializationSession,
    ) -> float:
        value = getattr(recv_req, "deadline_unix_sec", None)
        if value is None:
            value = session.deadline_unix_sec
        return self._validate_materialization_deadline(value)

    def _prepare_materialization_failure(
        self,
        recv_req: PrepareWeightMaterializationReqInput,
        message: str,
        *,
        session_state: WeightMaterializationSessionState | str = (
            WeightMaterializationSessionState.FAILED
        ),
    ) -> PrepareWeightMaterializationReqOutput:
        return PrepareWeightMaterializationReqOutput(
            materialization_id=recv_req.materialization_id,
            request_id=recv_req.request_id,
            success=False,
            message=message,
            external_dp_rank=self._external_dp_rank(),
            session_state=WeightMaterializationSessionState(session_state),
        )

    def _commit_materialization_failure(
        self,
        recv_req: CommitWeightMaterializationReqInput,
        message: str,
        *,
        session_state: WeightMaterializationSessionState | str = (
            WeightMaterializationSessionState.FAILED
        ),
        completion_unknown: bool = False,
        completion_ticket: str | None = None,
    ) -> CommitWeightMaterializationReqOutput:
        external_dp_rank = self._external_dp_rank()
        return CommitWeightMaterializationReqOutput(
            materialization_id=recv_req.materialization_id,
            request_id=recv_req.request_id,
            success=False,
            message=message,
            external_dp_rank=external_dp_rank,
            selected=(
                recv_req.selected_external_dp_rank is not None
                and recv_req.selected_external_dp_rank == external_dp_rank
            ),
            completion_unknown=completion_unknown,
            completion_ticket=completion_ticket,
            session_state=WeightMaterializationSessionState(session_state),
            phase=recv_req.phase,
        )

    def _retain_materialization_cleanup_session(
        self,
        recv_req: PrepareWeightMaterializationReqInput,
        source: RuntimeWeightSnapshotSource | None,
        output: PrepareWeightMaterializationReqOutput,
    ) -> None:
        session = _WeightMaterializationSession(
            request_identity=(recv_req.model_id, recv_req.revision),
            deadline_unix_sec=self._prepare_materialization_deadline(recv_req),
            source=source,
            selected_placements=(),
            selected_bindings=(),
            selected_payload_identity=None,
            local_selected_placement_ids=(),
            prepare_output=output,
            state=WeightMaterializationSessionState.CLEANUP_PENDING,
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
            session_state=WeightMaterializationSessionState.CLEANUP_PENDING,
        )
        session = _WeightMaterializationSession(
            request_identity=(recv_req.model_id, recv_req.revision),
            deadline_unix_sec=self._prepare_materialization_deadline(recv_req),
            source=source,
            selected_placements=(),
            selected_bindings=(),
            selected_payload_identity=None,
            local_selected_placement_ids=(),
            prepare_output=provisional,
            state=WeightMaterializationSessionState.CLEANUP_PENDING,
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
                session_state=WeightMaterializationSessionState.FAILED,
            )

        failures.extend(
            f"source cleanup remains pending: {error}" for error in cleanup_errors
        )
        output = self._prepare_materialization_failure(
            recv_req,
            " | ".join(failures),
            session_state=WeightMaterializationSessionState.CLEANUP_PENDING,
        )
        self._retain_materialization_cleanup_session(
            recv_req,
            session.source,
            output,
        )
        return output

    def _prune_weight_materialization_sessions_locked(self) -> None:
        terminal_ids = sorted(
            materialization_id
            for materialization_id, session in (
                self.weight_materialization_sessions.items()
            )
            if session.terminal_at is not None
        )
        for materialization_id in terminal_ids[
            : max(0, len(terminal_ids) - _WEIGHT_MATERIALIZATION_TERMINAL_LIMIT)
        ]:
            self.weight_materialization_sessions.pop(materialization_id, None)

    def _gather_weight_materialization_objects(
        self,
        value: Any,
        *,
        operation: str,
    ) -> list[Any]:
        try:
            collective_group = self._weight_materialization_collective_group()
            world_size = torch.distributed.get_world_size(group=collective_group)
            execution_context = self.weight_materialization_execution_context
            if world_size == 1:
                if execution_context is not None and execution_context.expired():
                    raise RuntimeError(
                        f"weight materialization {operation} exceeded its deadline"
                    )
                return [value]
            if execution_context is None:
                raise RuntimeError(
                    "weight materialization execution context is required "
                    f"for multi-rank {operation}"
                )
            coordinator = self._weight_materialization_store_coordinator()
            return coordinator.all_gather_object(
                value,
                phase=operation,
                execution_context=execution_context,
            )
        except Exception as error:
            message = f"failed to gather weight materialization {operation}: {error}"
            coordinator = self.weight_materialization_coordinator
            if bool(getattr(error, "completion_unknown", False)) or bool(
                getattr(coordinator, "poisoned", False)
            ):
                with self.weight_materialization_lock:
                    self.weight_materialization_poisoned = (
                        f"{message}; scheduler restart is required"
                    )
            raise RuntimeError(message) from error

    def _gather_weight_materialization_sources_to_root(
        self,
        value: Any,
        *,
        operation: str,
    ) -> tuple[Any, ...] | None:
        try:
            collective_group = self._weight_materialization_collective_group()
            world_size = torch.distributed.get_world_size(group=collective_group)
            execution_context = self.weight_materialization_execution_context
            if world_size > 1 and execution_context is None:
                raise RuntimeError(
                    "weight materialization execution context is required "
                    f"for multi-rank {operation}"
                )
            coordinator = self._weight_materialization_store_coordinator()
            return coordinator.gather_object_to_root(
                value,
                phase=operation,
                execution_context=execution_context,
            )
        except Exception as error:
            message = (
                f"failed to gather weight materialization {operation} to root: {error}"
            )
            coordinator = self.weight_materialization_coordinator
            if bool(getattr(error, "completion_unknown", False)) or bool(
                getattr(coordinator, "poisoned", False)
            ):
                with self.weight_materialization_lock:
                    self.weight_materialization_poisoned = (
                        f"{message}; scheduler restart is required"
                    )
            raise RuntimeError(message) from error

    def _scatter_weight_materialization_sources_from_root(
        self,
        values: tuple[Any, ...] | None,
        *,
        operation: str,
    ) -> Any:
        try:
            collective_group = self._weight_materialization_collective_group()
            world_size = torch.distributed.get_world_size(group=collective_group)
            execution_context = self.weight_materialization_execution_context
            if world_size > 1 and execution_context is None:
                raise RuntimeError(
                    "weight materialization execution context is required "
                    f"for multi-rank {operation}"
                )
            coordinator = self._weight_materialization_store_coordinator()
            return coordinator.scatter_object_from_root(
                values,
                phase=operation,
                execution_context=execution_context,
            )
        except Exception as error:
            message = (
                f"failed to scatter weight materialization {operation} from root: "
                f"{error}"
            )
            coordinator = self.weight_materialization_coordinator
            if bool(getattr(error, "completion_unknown", False)) or bool(
                getattr(coordinator, "poisoned", False)
            ):
                with self.weight_materialization_lock:
                    self.weight_materialization_poisoned = (
                        f"{message}; scheduler restart is required"
                    )
            raise RuntimeError(message) from error

    def _weight_materialization_store_coordinator(
        self,
    ) -> TorchDistributedWeightStoreCoordinator:
        coordinator = self.weight_materialization_coordinator
        if coordinator is None:
            coordinator = TorchDistributedWeightStoreCoordinator(
                self._weight_materialization_collective_group()
            )
            self.weight_materialization_coordinator = coordinator
        return coordinator

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
    def _prepare_materialization_backend_close(
        session: _WeightMaterializationSession,
        *,
        timeout_sec: float,
    ) -> str | None:
        retry_close = session.backend_owner_close_pending
        owner = session.backend_owner
        if owner is None:
            if session.backend is not None:
                session.backend_completion_unknown = True
                return "weight storage backend has no lifecycle owner"
            session.backend_completion_unknown = False
            return None

        backend = session.backend
        if backend is None:
            session.backend_completion_unknown = False
            return None
        seal = getattr(backend, "seal", None)
        quiesce = getattr(backend, "quiesce", None)
        if not callable(seal) or not callable(quiesce):
            session.backend_completion_unknown = True
            return "weight storage backend does not expose lifecycle control"
        bounded_timeout = (
            timeout_sec
            if isinstance(timeout_sec, (int, float))
            and not isinstance(timeout_sec, bool)
            and math.isfinite(timeout_sec)
            else 0.0
        )
        timeout_ms = max(0, int(max(0.0, bounded_timeout) * 1000))
        try:
            pending_at_seal = tuple(seal())
            status = quiesce(timeout_ms=timeout_ms)
        except Exception as error:
            session.backend_completion_unknown = True
            return f"failed to quiesce weight storage backend: {error}"
        if not isinstance(status, WeightSnapshotBackendStatus):
            session.backend_completion_unknown = True
            return "weight storage backend returned an invalid lifecycle status"
        if not status.terminal:
            session.backend_completion_unknown = True
            pending = status.pending_tickets or pending_at_seal
            detail = ", ".join(pending) if pending else "unknown"
            return f"weight storage backend still has pending calls: {detail}"
        if retry_close:
            close = getattr(backend, "close", None)
            if not callable(close):
                session.backend_completion_unknown = True
                return "weight storage backend does not expose close"
            try:
                status = close(timeout_ms=timeout_ms)
            except Exception as error:
                session.backend_completion_unknown = True
                return f"failed to close weight storage backend: {error}"
            if not isinstance(status, WeightSnapshotBackendStatus):
                session.backend_completion_unknown = True
                return "weight storage backend returned an invalid close status"
            if not status.closed:
                session.backend_completion_unknown = True
                pending = ", ".join(status.pending_tickets) or "unknown"
                return f"weight storage backend close remains pending: {pending}"
            session.backend_owner_close_pending = False

        session.backend_completion_unknown = False
        return None

    @staticmethod
    def _close_materialization_backend_owner(
        session: _WeightMaterializationSession,
    ) -> str | None:
        owner = session.backend_owner
        if owner is None:
            if session.backend is not None:
                session.backend_completion_unknown = True
                return "weight storage backend has no lifecycle owner"
            return None
        try:
            owner.close()
        except Exception as error:
            session.backend_completion_unknown = bool(
                getattr(error, "completion_unknown", False)
            )
            session.backend_owner_close_pending = session.backend_completion_unknown
            return str(error)
        session.backend_owner = None
        session.backend = None
        session.backend_close_succeeded = True
        session.backend_completion_unknown = False
        session.backend_owner_close_pending = False
        return None

    @classmethod
    def _close_materialization_backend(
        cls,
        session: _WeightMaterializationSession,
        *,
        timeout_sec: float = 0.0,
    ) -> str | None:
        prepare_error = cls._prepare_materialization_backend_close(
            session,
            timeout_sec=timeout_sec,
        )
        if prepare_error is not None:
            return prepare_error
        return cls._close_materialization_backend_owner(session)

    def _prepare_materialization_backend_close_world(
        self,
        session: _WeightMaterializationSession,
        *,
        operation: str,
        deadline_unix_sec: float,
    ) -> tuple[list[str], bool]:
        try:
            ownership = self._gather_weight_materialization_objects(
                {"present": session.backend_owner is not None},
                operation=f"{operation} ownership",
            )
        except Exception as error:
            return [str(error)], True
        if any(
            not isinstance(status, dict)
            or "present" not in status
            or type(status.get("present")) is not bool
            for status in ownership
        ):
            return ["model ranks returned invalid Store backend ownership"], True
        if len({status["present"] for status in ownership}) != 1:
            return ["model ranks disagree on Store backend ownership"], True

        prepare_error = self._prepare_materialization_backend_close(
            session,
            timeout_sec=max(0.0, deadline_unix_sec - time.time()),
        )
        try:
            readiness = self._gather_weight_materialization_objects(
                {
                    "error": prepare_error,
                    "completion_unknown": session.backend_completion_unknown,
                },
                operation=f"{operation} readiness",
            )
        except Exception as error:
            return [str(error)], True
        readiness_errors = []
        completion_unknown = False
        for rank, status in enumerate(readiness):
            if (
                not isinstance(status, dict)
                or "error" not in status
                or type(status.get("completion_unknown")) is not bool
                or (
                    status.get("error") is not None
                    and type(status.get("error")) is not str
                )
            ):
                readiness_errors.append(
                    f"rank {rank}: invalid Store backend close readiness"
                )
                completion_unknown = True
                continue
            completion_unknown |= status["completion_unknown"]
            if status["error"] is not None:
                readiness_errors.append(f"rank {rank}: {status['error']}")
        return readiness_errors, completion_unknown

    def _close_materialization_backend_world(
        self,
        session: _WeightMaterializationSession,
        *,
        operation: str,
        deadline_unix_sec: float,
        backend_prepared: bool = False,
    ) -> tuple[list[str], bool]:
        if not backend_prepared:
            readiness_errors, completion_unknown = (
                self._prepare_materialization_backend_close_world(
                    session,
                    operation=operation,
                    deadline_unix_sec=deadline_unix_sec,
                )
            )
            if readiness_errors:
                return readiness_errors, completion_unknown
        local_error = self._close_materialization_backend_owner(session)
        try:
            statuses = self._gather_weight_materialization_objects(
                {
                    "error": local_error,
                    "completion_unknown": session.backend_completion_unknown,
                },
                operation=f"{operation} status",
            )
        except Exception as error:
            return [str(error)], True
        errors = []
        completion_unknown = False
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
                errors.append(f"rank {rank}: invalid Store backend close status")
                completion_unknown = True
                continue
            completion_unknown |= status["completion_unknown"]
            if status.get("error") is not None:
                errors.append(f"rank {rank}: {status['error']}")
        return errors, completion_unknown

    def _cleanup_materialization_session_for_shutdown(
        self,
        materialization_id: str,
        session: _WeightMaterializationSession,
        *,
        timeout_sec: float,
    ) -> None:
        if session.provider_finalize_pending:
            logger.warning(
                "Preserving weight materialization %s for provider finalize",
                materialization_id,
            )
            return
        backend_error = self._prepare_materialization_backend_close(
            session,
            timeout_sec=timeout_sec,
        )
        if backend_error is not None:
            logger.warning(
                "Weight materialization backend cleanup remains pending during "
                "shutdown: %s",
                backend_error,
            )
            return
        source_error = self._release_materialization_session_source(session)
        if source_error is None:
            backend_error = self._close_materialization_backend_owner(session)
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

    def _validate_materialization_commit_world(
        self,
        recv_req: CommitWeightMaterializationReqInput,
        session: _WeightMaterializationSession | None,
        commit_identity: tuple[int | None, str | None],
        deadline_unix_sec: float | None,
        *,
        cleanup_requested: bool,
    ) -> tuple[str, WeightMaterializationSessionState] | None:
        request_state = (
            recv_req.materialization_id,
            recv_req.request_id,
            recv_req.selected_external_dp_rank,
            recv_req.phase,
            commit_identity,
            deadline_unix_sec,
        )
        local_state = {
            "request": request_state,
            "present": session is not None,
            "request_identity": None if session is None else session.request_identity,
            "commit_identity": None if session is None else session.commit_identity,
            "state": (
                None
                if session is None
                else WeightMaterializationSessionState(session.state).value
            ),
            "has_commit_output": (
                False if session is None else session.commit_output is not None
            ),
            "publication_ref": (None if session is None else session.publication_ref),
        }
        try:
            states = self._gather_weight_materialization_objects(
                local_state,
                operation="commit request state",
            )
        except Exception as error:
            return (
                str(error),
                WeightMaterializationSessionState.COMPLETION_UNKNOWN,
            )

        if any(not isinstance(state, dict) for state in states):
            return (
                "model ranks returned invalid materialization commit state",
                WeightMaterializationSessionState.CONFLICT,
            )
        if any(
            type(state.get("present")) is not bool
            or type(state.get("has_commit_output")) is not bool
            or (
                state.get("request_identity") is not None
                and (
                    not isinstance(state.get("request_identity"), tuple)
                    or len(state["request_identity"]) != 2
                    or any(type(item) is not str for item in state["request_identity"])
                )
            )
            or (
                state.get("commit_identity") is not None
                and (
                    not isinstance(state.get("commit_identity"), tuple)
                    or len(state["commit_identity"]) != 2
                )
            )
            or (state.get("state") is not None and type(state.get("state")) is not str)
            or (
                state.get("publication_ref") is not None
                and not isinstance(state.get("publication_ref"), dict)
            )
            for state in states
        ):
            return (
                "model ranks returned invalid materialization commit state",
                WeightMaterializationSessionState.CONFLICT,
            )
        if any(state.get("request") != request_state for state in states):
            return (
                "model ranks received different materialization commit requests",
                WeightMaterializationSessionState.CONFLICT,
            )

        present = {state.get("present") for state in states}
        if present == {False}:
            return (
                "weight materialization session was not prepared",
                WeightMaterializationSessionState.NOT_FOUND,
            )
        if present != {True}:
            return (
                "model ranks disagree on materialization session ownership",
                WeightMaterializationSessionState.COMPLETION_UNKNOWN,
            )

        request_identities = {state.get("request_identity") for state in states}
        if len(request_identities) != 1:
            return (
                "model ranks disagree on materialization model identity",
                WeightMaterializationSessionState.CONFLICT,
            )
        bound_identities = {state.get("commit_identity") for state in states}
        if len(bound_identities) != 1:
            return (
                "model ranks disagree on materialization destination identity",
                WeightMaterializationSessionState.COMPLETION_UNKNOWN,
            )
        bound_identity = next(iter(bound_identities))
        if (
            not cleanup_requested
            and bound_identity is not None
            and bound_identity != commit_identity
        ):
            return (
                "materialization ID is already bound to another destination",
                WeightMaterializationSessionState.CONFLICT,
            )

        refs = {
            json.dumps(state.get("publication_ref"), sort_keys=True) for state in states
        }
        if len(refs) != 1:
            return (
                "model ranks disagree on the published weight storage ref",
                WeightMaterializationSessionState.COMPLETION_UNKNOWN,
            )
        if cleanup_requested:
            return None

        session_states = {state.get("state") for state in states}
        commit_outputs = {state.get("has_commit_output") for state in states}
        if len(session_states) != 1 or len(commit_outputs) != 1:
            return (
                "model ranks disagree on materialization recovery state",
                WeightMaterializationSessionState.COMPLETION_UNKNOWN,
            )
        return None

    def _materialization_deadline_expired_world(
        self,
        deadline_unix_sec: float,
    ) -> tuple[bool, str | None]:
        previous_context = self.weight_materialization_execution_context
        deadline_context = WeightTransferExecutionContext(
            deadline_unix_sec=(
                time.time() + _WEIGHT_MATERIALIZATION_CONTROL_TIMEOUT_SEC
            ),
        )
        with self.weight_materialization_lock:
            self.weight_materialization_execution_context = deadline_context
        try:
            votes = self._gather_weight_materialization_objects(
                {"expired": time.time() >= deadline_unix_sec},
                operation="commit deadline vote",
            )
        except Exception as error:
            return False, str(error)
        finally:
            with self.weight_materialization_lock:
                if self.weight_materialization_execution_context is deadline_context:
                    self.weight_materialization_execution_context = previous_context
        if any(
            not isinstance(vote, dict) or type(vote.get("expired")) is not bool
            for vote in votes
        ):
            return False, "model ranks returned an invalid deadline vote"
        return any(vote["expired"] for vote in votes), None

    def _unresolved_materialization_capacity_error(
        self,
        *,
        current_materialization_id: str,
    ) -> str | None:
        with self.weight_materialization_lock:
            local_count = sum(
                session.terminal_at is None
                and (
                    blocks_unresolved_materialization_capacity(session.state)
                    or (
                        session.commit_output is not None
                        and session.commit_output.completion_unknown
                    )
                )
                for materialization_id, session in (
                    self.weight_materialization_sessions.items()
                )
                if materialization_id != current_materialization_id
            )
        try:
            statuses = self._gather_weight_materialization_objects(
                {
                    "count": local_count,
                    "limit": _WEIGHT_MATERIALIZATION_UNRESOLVED_LIMIT,
                },
                operation="unresolved materialization capacity",
            )
        except Exception as error:
            return str(error)
        if any(
            not isinstance(status, dict)
            or type(status.get("count")) is not int
            or type(status.get("limit")) is not int
            for status in statuses
        ):
            return "model ranks returned invalid unresolved session capacity"
        counts = {status["count"] for status in statuses}
        limits = {status["limit"] for status in statuses}
        if len(counts) != 1 or limits != {_WEIGHT_MATERIALIZATION_UNRESOLVED_LIMIT}:
            return "model ranks disagree on unresolved session capacity"
        if counts.pop() >= _WEIGHT_MATERIALIZATION_UNRESOLVED_LIMIT:
            return (
                "unresolved weight materialization limit reached; recover or "
                "clean up an existing materialization before starting another"
            )
        return None

    @staticmethod
    def _materialization_source_record_count(
        placement: WeightPlacementManifest,
        binding: WeightRuntimeBindingManifest,
        payload_identity: WeightPayloadIdentity | None,
    ) -> int:
        return (
            2
            + len(placement.tensors)
            + len(binding.fragments)
            + (0 if payload_identity is None else len(payload_identity.fragments))
        )

    @classmethod
    def _select_materialization_source_metadata(
        cls,
        gathered: list[Any],
        *,
        model_id: str,
        revision: str,
    ) -> _MaterializationSourceProjection:
        if (
            not gathered
            or len(gathered) > _WEIGHT_MATERIALIZATION_MAX_WORLD_SOURCE_RECORDS
        ):
            raise ValueError("model source world exceeds the configured record limit")
        aggregate_records = 0
        for item in gathered:
            if not isinstance(item, dict):
                raise ValueError("model ranks returned invalid capture status")
            placement = item.get("placement")
            binding = item.get("binding")
            identity = item.get("payload_identity")
            if (
                not isinstance(placement, WeightPlacementManifest)
                or not isinstance(binding, WeightRuntimeBindingManifest)
                or (
                    identity is not None
                    and not isinstance(identity, WeightPayloadIdentity)
                )
            ):
                raise ValueError("model ranks returned invalid materialization sources")
            record_count = cls._materialization_source_record_count(
                placement,
                binding,
                identity,
            )
            if record_count > _WEIGHT_MATERIALIZATION_MAX_RANK_SOURCE_RECORDS:
                raise ValueError(
                    "rank materialization source exceeds the configured record limit"
                )
            aggregate_records += record_count
            if aggregate_records > _WEIGHT_MATERIALIZATION_MAX_WORLD_SOURCE_RECORDS:
                raise ValueError(
                    "model source world exceeds the configured record limit"
                )

        placements = tuple(item["placement"] for item in gathered)
        bindings = tuple(item["binding"] for item in gathered)
        if not placements or any(
            not isinstance(item, WeightPlacementManifest) for item in placements
        ):
            raise ValueError("model ranks returned invalid weight placements")
        if any(not isinstance(item, WeightRuntimeBindingManifest) for item in bindings):
            raise ValueError("model ranks returned invalid runtime bindings")
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
        selected_binding_by_id = {
            binding.placement_id: binding for binding in selected_bindings
        }
        local_placement_ids_by_rank = []
        for item in gathered:
            local_placement = item["placement"]
            local_binding = item["binding"]
            local_fragment_ids = {
                tensor.placement_fragment_id for tensor in local_placement.tensors
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
            current_local_placement_ids = tuple(
                placement.placement_id for placement in local_selected
            )
            if local_selected:
                projected_local_bindings = project_source_bindings(
                    local_selected,
                    (local_binding,),
                )
                selected_local_bindings = tuple(
                    selected_binding_by_id[placement_id]
                    for placement_id in current_local_placement_ids
                )
                if selected_local_bindings != projected_local_bindings:
                    raise ValueError(
                        "Store selection binding differs from the local runtime "
                        "projection"
                    )
            local_placement_ids_by_rank.append(current_local_placement_ids)
        return _MaterializationSourceProjection(
            placements=placements,
            bindings=bindings,
            selected_placements=tuple(selected_placements),
            selected_bindings=selected_bindings,
            local_placement_ids=tuple(local_placement_ids_by_rank),
            generation=next(iter(generations)),
        )

    @classmethod
    def _merge_materialization_sources(
        cls,
        gathered: list[Any],
        *,
        model_id: str,
        revision: str,
    ) -> _MaterializationSourceDistribution:
        projection = cls._select_materialization_source_metadata(
            gathered,
            model_id=model_id,
            revision=revision,
        )
        selected_placement_by_id = {
            placement.placement_id: placement
            for placement in projection.selected_placements
        }
        checksums: dict[str, str] = {}
        for rank, item in enumerate(gathered):
            identity = item.get("payload_identity")
            local_placement = projection.placements[rank]
            if identity is not None:
                if not isinstance(identity, WeightPayloadIdentity):
                    raise ValueError("model ranks returned invalid payload identities")
                if identity.select((local_placement,)) != identity:
                    raise ValueError(
                        "rank payload identity differs from its weight placement"
                    )
            local_selected = tuple(
                selected_placement_by_id[placement_id]
                for placement_id in projection.local_placement_ids[rank]
            )
            if not local_selected:
                continue
            if identity is None:
                raise ValueError("selected Store source has no payload identity")
            selected_local_identity = identity.select(local_selected)
            for fragment in selected_local_identity.fragments:
                if fragment.placement_fragment_id in checksums:
                    raise ValueError("duplicate payload fragment identity")
                checksums[fragment.placement_fragment_id] = fragment.checksum
        selected_identity = WeightPayloadIdentity.create(
            projection.selected_placements,
            checksums,
        )
        logical_digest = _logical_payload_digest(
            projection.selected_placements,
            selected_identity,
        )
        summary = _MaterializationSourceSummary(
            model_id=model_id,
            revision=revision,
            generation=projection.generation,
            logical_payload_digest=logical_digest,
            total_bytes=sum(
                tensor.nbytes
                for placement in projection.selected_placements
                for tensor in placement.tensors
            ),
            placement_count=len(projection.selected_placements),
            fragment_count=sum(
                len(placement.tensors) for placement in projection.selected_placements
            ),
        )
        selected_binding_by_id = {
            binding.placement_id: binding for binding in projection.selected_bindings
        }
        rank_selections = []
        for rank, local_placement_ids in enumerate(projection.local_placement_ids):
            local_selected = tuple(
                selected_placement_by_id[placement_id]
                for placement_id in local_placement_ids
            )
            if rank == 0:
                request_placements = projection.selected_placements
                request_bindings = projection.selected_bindings
                request_identity = selected_identity
            else:
                request_placements = (
                    local_selected
                    if local_selected
                    else (projection.selected_placements[0],)
                )
                request_bindings = tuple(
                    selected_binding_by_id[placement.placement_id]
                    for placement in request_placements
                )
                request_identity = selected_identity.select(request_placements)
            rank_selections.append(
                _RankMaterializationSourceSelection(
                    placements=tuple(request_placements),
                    bindings=request_bindings,
                    payload_identity=request_identity,
                    local_placement_ids=tuple(local_placement_ids),
                    summary=summary,
                )
            )
        return _MaterializationSourceDistribution(
            rank_selections=tuple(rank_selections),
        )

    def _prepare_materialization_replay_vote(
        self,
        recv_req: PrepareWeightMaterializationReqInput,
        existing: _WeightMaterializationSession | None,
        request_identity: tuple[str, str],
    ) -> PrepareWeightMaterializationReqOutput | None:
        if self.weight_materialization_execution_context is None:
            if existing is None:
                return None
            if existing.request_identity != request_identity:
                return self._prepare_materialization_failure(
                    recv_req,
                    "materialization ID is already bound to another model revision",
                    session_state=WeightMaterializationSessionState.CONFLICT,
                )
            return msgspec.structs.replace(
                existing.prepare_output,
                request_id=recv_req.request_id,
            )

        replay_output = (
            None
            if existing is None
            else msgspec.structs.replace(
                existing.prepare_output,
                request_id="",
            )
        )
        try:
            votes = self._gather_weight_materialization_objects(
                {
                    "present": existing is not None,
                    "request_identity": (
                        None if existing is None else existing.request_identity
                    ),
                    "state": (
                        None
                        if existing is None
                        else WeightMaterializationSessionState(existing.state).value
                    ),
                    "prepare_output": replay_output,
                },
                operation="prepare session vote",
            )
        except Exception as error:
            return self._prepare_materialization_failure(recv_req, str(error))
        if any(
            not isinstance(vote, dict) or type(vote.get("present")) is not bool
            for vote in votes
        ):
            return self._prepare_materialization_failure(
                recv_req,
                "model ranks returned invalid materialization session state",
                session_state=WeightMaterializationSessionState.CONFLICT,
            )
        present = tuple(vote["present"] for vote in votes)
        if any(present) and not all(present):
            message = (
                "model ranks disagree on the materialization session; "
                "scheduler restart is required"
            )
            with self.weight_materialization_lock:
                self.weight_materialization_poisoned = message
            return self._prepare_materialization_failure(
                recv_req,
                message,
                session_state=WeightMaterializationSessionState.CONFLICT,
            )
        if not all(present):
            return None

        first = votes[0]
        if any(
            vote.get("request_identity") != first.get("request_identity")
            or vote.get("state") != first.get("state")
            or vote.get("prepare_output") != first.get("prepare_output")
            for vote in votes[1:]
        ):
            message = (
                "model ranks disagree on the materialization replay state; "
                "scheduler restart is required"
            )
            with self.weight_materialization_lock:
                self.weight_materialization_poisoned = message
            return self._prepare_materialization_failure(
                recv_req,
                message,
                session_state=WeightMaterializationSessionState.CONFLICT,
            )
        if first.get("request_identity") != request_identity:
            return self._prepare_materialization_failure(
                recv_req,
                "materialization ID is already bound to another model revision",
                session_state=WeightMaterializationSessionState.CONFLICT,
            )
        assert existing is not None
        return msgspec.structs.replace(
            existing.prepare_output,
            request_id=recv_req.request_id,
        )

    def prepare_weight_materialization(
        self,
        recv_req: PrepareWeightMaterializationReqInput,
    ) -> PrepareWeightMaterializationReqOutput:
        """Capture and validate one Store materialization source on every rank."""

        try:
            self._prepare_materialization_deadline(recv_req)
        except ValueError as error:
            return self._prepare_materialization_failure(recv_req, str(error))
        request_identity = (recv_req.model_id, recv_req.revision)
        with self.weight_materialization_lock:
            self._prune_weight_materialization_sessions_locked()
            existing = self.weight_materialization_sessions.get(
                recv_req.materialization_id
            )
        replay = self._prepare_materialization_replay_vote(
            recv_req,
            existing,
            request_identity,
        )
        if replay is not None:
            return replay

        local_source = None
        try:
            local_source = (
                self.tp_worker.model_runner.capture_runtime_weight_snapshot_source(
                    materialization_id=recv_req.materialization_id,
                    model_id=recv_req.model_id,
                    revision=recv_req.revision,
                    lease_timeout_sec=recv_req.lease_timeout_sec,
                    execution_context=self.weight_materialization_execution_context,
                    defer_payload_identity=True,
                )
            )
            local_result = {
                "success": True,
                "message": "Success.",
                "placement": local_source.placement,
                "binding": local_source.binding,
                "payload_identity": None,
            }
        except Exception as error:
            local_result = {
                "success": False,
                "message": str(error),
                "placement": None,
                "binding": None,
                "payload_identity": None,
            }

        def fail_after_source_capture(error: Exception):
            cleanup_error = (
                None
                if local_source is None
                else self._release_materialization_source(local_source)
            )
            message = str(error)
            state = WeightMaterializationSessionState.FAILED
            if cleanup_error is not None:
                message += f"; source cleanup remains pending: {cleanup_error}"
                state = WeightMaterializationSessionState.CLEANUP_PENDING
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

        metadata_control_complete = False
        try:
            metadata_gathered = self._gather_weight_materialization_sources_to_root(
                local_result,
                operation="prepare source metadata gather",
            )
            hash_packets = None
            if metadata_gathered is not None:
                failures = []
                for rank, item in enumerate(metadata_gathered):
                    if not isinstance(item, dict):
                        failures.append(f"rank {rank}: invalid capture status")
                    elif not item.get("success", False):
                        failures.append(
                            f"rank {rank}: {item.get('message', 'capture failed')}"
                        )
                projection = None
                if not failures:
                    try:
                        projection = self._select_materialization_source_metadata(
                            list(metadata_gathered),
                            model_id=recv_req.model_id,
                            revision=recv_req.revision,
                        )
                    except Exception as error:
                        failures.append(str(error))
                if projection is None:
                    message = " | ".join(failures) or "source merge failed"
                    hash_packets = tuple(
                        {
                            "success": False,
                            "message": message,
                            "hash_required": False,
                        }
                        for _ in range(
                            self._weight_materialization_store_coordinator().world_size
                        )
                    )
                else:
                    hash_packets = tuple(
                        {
                            "success": True,
                            "message": "Success.",
                            "hash_required": bool(local_placement_ids),
                        }
                        for local_placement_ids in projection.local_placement_ids
                    )
            local_hash_packet = self._scatter_weight_materialization_sources_from_root(
                hash_packets,
                operation="prepare source hash selection",
            )
            metadata_control_complete = True
            if (
                not isinstance(local_hash_packet, dict)
                or type(local_hash_packet.get("success")) is not bool
                or type(local_hash_packet.get("message")) is not str
                or type(local_hash_packet.get("hash_required")) is not bool
            ):
                raise ValueError("root returned an invalid source hash packet")
            if not local_hash_packet["success"]:
                raise ValueError(local_hash_packet["message"])
            if (
                local_hash_packet["hash_required"]
                and local_source is not None
                and local_source.payload_identity is None
            ):
                capture_payload_identity = getattr(
                    local_source,
                    "capture_payload_identity",
                    None,
                )
                if not callable(capture_payload_identity):
                    raise ValueError(
                        "selected Store source cannot capture payload identity"
                    )
                identity = capture_payload_identity(
                    execution_context=self.weight_materialization_execution_context,
                )
                if not isinstance(identity, WeightPayloadIdentity):
                    raise ValueError(
                        "selected Store source returned an invalid payload identity"
                    )
            local_result["payload_identity"] = (
                None if local_source is None else local_source.payload_identity
            )
        except Exception as error:
            if not metadata_control_complete:
                return fail_after_source_capture(error)
            local_result = {
                "success": False,
                "message": str(error),
                "placement": (None if local_source is None else local_source.placement),
                "binding": None if local_source is None else local_source.binding,
                "payload_identity": (
                    None if local_source is None else local_source.payload_identity
                ),
            }

        try:
            payload_gathered = self._gather_weight_materialization_sources_to_root(
                {
                    "success": local_result["success"],
                    "message": local_result["message"],
                    "payload_identity": local_result["payload_identity"],
                },
                operation="prepare source payload gather",
            )
            packets = None
            if payload_gathered is not None:
                assert metadata_gathered is not None
                if len(payload_gathered) != len(metadata_gathered):
                    raise ValueError(
                        "source metadata and payload identity worlds differ"
                    )
                gathered = tuple(
                    (
                        {
                            **metadata,
                            "success": payload.get("success"),
                            "message": payload.get("message"),
                            "payload_identity": payload.get("payload_identity"),
                        }
                        if isinstance(metadata, dict) and isinstance(payload, dict)
                        else payload
                    )
                    for metadata, payload in zip(
                        metadata_gathered,
                        payload_gathered,
                        strict=True,
                    )
                )
                failures = []
                for rank, item in enumerate(gathered):
                    if not isinstance(item, dict):
                        failures.append(f"rank {rank}: invalid capture status")
                    elif not item.get("success", False):
                        failures.append(
                            f"rank {rank}: "
                            f"{item.get('message', 'payload capture failed')}"
                        )
                distribution = None
                if not failures:
                    try:
                        distribution = self._merge_materialization_sources(
                            list(gathered),
                            model_id=recv_req.model_id,
                            revision=recv_req.revision,
                        )
                    except Exception as error:
                        failures.append(str(error))
                if distribution is None:
                    message = " | ".join(failures) or "source merge failed"
                    packets = tuple(
                        {
                            "success": False,
                            "message": message,
                            "selection": None,
                        }
                        for _ in range(
                            self._weight_materialization_store_coordinator().world_size
                        )
                    )
                else:
                    packets = tuple(
                        {
                            "success": True,
                            "message": "Success.",
                            "selection": selection,
                        }
                        for selection in distribution.rank_selections
                    )
            local_packet = self._scatter_weight_materialization_sources_from_root(
                packets,
                operation="prepare source selection",
            )
        except Exception as error:
            return fail_after_source_capture(error)

        merge_error = None
        selection = None
        try:
            if (
                not isinstance(local_packet, dict)
                or type(local_packet.get("success")) is not bool
                or type(local_packet.get("message")) is not str
            ):
                raise ValueError("root returned an invalid source selection packet")
            if not local_packet["success"]:
                raise ValueError(local_packet["message"])
            selection = local_packet.get("selection")
            if not isinstance(selection, _RankMaterializationSourceSelection):
                raise ValueError("root returned an invalid source selection")
            if (
                not selection.placements
                or not selection.bindings
                or selection.payload_identity.select(selection.placements)
                != selection.payload_identity
            ):
                raise ValueError("root returned an incomplete source selection")
            bind_weight_source(selection.placements, selection.bindings)
            if (
                selection.summary.model_id != recv_req.model_id
                or selection.summary.revision != recv_req.revision
                or selection.summary.generation <= 0
                or selection.summary.total_bytes <= 0
                or selection.summary.placement_count <= 0
                or selection.summary.fragment_count <= 0
                or not selection.summary.logical_payload_digest.startswith("sha256:")
            ):
                raise ValueError("root returned an invalid source summary")
            local_record_count = sum(
                self._materialization_source_record_count(
                    placement,
                    binding,
                    selection.payload_identity.select((placement,)),
                )
                for placement, binding in zip(
                    selection.placements,
                    selection.bindings,
                    strict=True,
                )
            )
            if local_record_count > _WEIGHT_MATERIALIZATION_MAX_WORLD_SOURCE_RECORDS:
                raise ValueError(
                    "selected materialization source exceeds the configured "
                    "record limit"
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
                    WeightMaterializationSessionState.CLEANUP_PENDING
                    if cleanup_error is not None
                    else WeightMaterializationSessionState.FAILED
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

        assert local_source is not None
        assert selection is not None
        summary = selection.summary
        output = PrepareWeightMaterializationReqOutput(
            materialization_id=recv_req.materialization_id,
            request_id=recv_req.request_id,
            success=True,
            message="Success.",
            external_dp_rank=self._external_dp_rank(),
            generation=summary.generation,
            logical_payload_digest=summary.logical_payload_digest,
            total_bytes=summary.total_bytes,
            session_state=WeightMaterializationSessionState.PREPARED,
        )
        session = _WeightMaterializationSession(
            request_identity=request_identity,
            deadline_unix_sec=self._prepare_materialization_deadline(recv_req),
            source=local_source,
            selected_placements=selection.placements,
            selected_bindings=selection.bindings,
            selected_payload_identity=selection.payload_identity,
            local_selected_placement_ids=selection.local_placement_ids,
            prepare_output=output,
            state=WeightMaterializationSessionState.PREPARED,
        )
        with self.weight_materialization_lock:
            existing = self.weight_materialization_sessions.setdefault(
                recv_req.materialization_id,
                session,
            )
        if existing is not session:
            cleanup_error = self._release_materialization_source(local_source)
            if existing.request_identity == request_identity and cleanup_error is None:
                return msgspec.structs.replace(
                    existing.prepare_output,
                    request_id=recv_req.request_id,
                )
            message = "materialization ID raced with another prepare request"
            if cleanup_error is not None:
                message += f"; source cleanup remains pending: {cleanup_error}"
            return self._prepare_materialization_failure(
                recv_req,
                message,
                session_state=WeightMaterializationSessionState.CONFLICT,
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

    @staticmethod
    def _selected_materialization_binding(
        session: _WeightMaterializationSession,
    ) -> WeightRuntimeBindingManifest | None:
        if not session.local_selected_placement_ids:
            return None
        if len(session.local_selected_placement_ids) != 1:
            raise RuntimeError("local Store source must have one placement")
        placement_id = session.local_selected_placement_ids[0]
        matches = tuple(
            binding
            for binding in session.selected_bindings
            if binding.placement_id == placement_id
        )
        if len(matches) != 1:
            raise RuntimeError("local Store source binding is missing")
        return matches[0]

    @classmethod
    def _materialization_attestor(
        cls,
        session: _WeightMaterializationSession,
    ) -> _SelectedRuntimeSourceAttestor | _NoLocalRuntimeSourceAttestor:
        source = session.source
        if source is None:
            raise RuntimeError("weight materialization source was already released")
        request_binding = cls._selected_materialization_binding(session)
        if request_binding is None:
            return _NoLocalRuntimeSourceAttestor.from_source(source)
        return _SelectedRuntimeSourceAttestor(
            source=source,
            request_binding=request_binding,
        )

    @staticmethod
    def _resolve_materialization_source(
        session: _WeightMaterializationSession,
        *,
        operation_id: str,
        provider_name: str,
    ) -> str | None:
        source = session.source
        if source is None:
            return None
        try:
            if source.quarantined:
                ticket = source.completion_ticket
                if not ticket:
                    raise RuntimeError(
                        "completion-unknown source has no recovery ticket"
                    )
                source.resolve_quarantine(
                    WeightTransferTerminalProof(
                        operation_id=operation_id,
                        provider=provider_name,
                        completion_ticket=ticket,
                        status=WeightTransferTerminalStatus.COMPLETED,
                    )
                )
            else:
                source.release()
        except Exception as error:
            return str(error)
        session.source = None
        return None

    def _resolve_materialization_source_world(
        self,
        session: _WeightMaterializationSession,
        *,
        operation_id: str,
        provider_name: str,
    ) -> list[str]:
        local_error = self._resolve_materialization_source(
            session,
            operation_id=operation_id,
            provider_name=provider_name,
        )
        try:
            statuses = self._gather_weight_materialization_objects(
                {"error": local_error},
                operation="post-publication source resolution",
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
                errors.append(f"rank {rank}: invalid source resolution status")
            elif status["error"] is not None:
                errors.append(f"rank {rank}: {status['error']}")
        return errors

    def _record_materialization_commit(
        self,
        session: _WeightMaterializationSession,
        output: CommitWeightMaterializationReqOutput,
    ) -> CommitWeightMaterializationReqOutput:
        state = WeightMaterializationSessionState(output.session_state)
        if output.session_state is not state:
            output = msgspec.structs.replace(output, session_state=state)
        with self.weight_materialization_lock:
            if output.ref is not None:
                if (
                    session.publication_ref is not None
                    and output.ref != session.publication_ref
                ):
                    raise RuntimeError(
                        "materialization session published different storage refs"
                    )
                session.publication_ref = dict(output.ref)
            session.state = state
            if output.phase == "cleanup":
                session.cleanup_output = output
                if (
                    session.publication_ref is not None
                    and session.commit_output is not None
                    and not output.completion_unknown
                    and is_published_materialization_state(state)
                    and not is_retryable_materialization_state(state)
                ):
                    session.commit_output = msgspec.structs.replace(
                        session.commit_output,
                        success=True,
                        message=(
                            "Success."
                            if output.success
                            else "Published; cleanup failed: " + output.message
                        ),
                        ref=dict(session.publication_ref),
                        session_state=output.session_state,
                        completion_unknown=False,
                        completion_ticket=None,
                    )
            else:
                session.commit_output = output
            if (
                session.source is None
                and session.backend_owner is None
                and session.backend is None
                and not output.completion_unknown
                and is_terminal_materialization_state(state)
            ):
                session.terminal_at = time.monotonic()
            else:
                session.terminal_at = None
            self._prune_weight_materialization_sessions_locked()
        return output

    def _cleanup_weight_materialization_session(
        self,
        recv_req: CommitWeightMaterializationReqInput,
        session: _WeightMaterializationSession,
    ) -> CommitWeightMaterializationReqOutput:
        previous = session.commit_output
        if (
            previous is not None
            and previous.completion_unknown
            and not is_retryable_materialization_state(previous.session_state)
        ):
            return msgspec.structs.replace(
                previous,
                request_id=recv_req.request_id,
                phase=recv_req.phase,
            )
        selected = bool(previous is not None and previous.selected)
        ref = (
            dict(session.publication_ref)
            if selected and session.publication_ref is not None
            else None
        )
        if session.provider_finalize_pending:
            return self._record_materialization_commit(
                session,
                CommitWeightMaterializationReqOutput(
                    materialization_id=recv_req.materialization_id,
                    request_id=recv_req.request_id,
                    success=False,
                    message=(
                        "Provider finalize remains pending; retry the "
                        "selected materialization with the same materialization ID."
                    ),
                    external_dp_rank=self._external_dp_rank(),
                    selected=selected,
                    ref=None,
                    session_state=WeightMaterializationSessionState.FINALIZE_PENDING,
                    phase=recv_req.phase,
                ),
            )
        previous_cleanup_failed = WeightMaterializationSessionState(
            session.state
        ) is WeightMaterializationSessionState.PUBLISHED_CLEANUP_FAILED or (
            previous is not None
            and WeightMaterializationSessionState(previous.session_state)
            is WeightMaterializationSessionState.PUBLISHED_CLEANUP_FAILED
        )
        close_deadline = self._commit_materialization_deadline(
            recv_req,
            session,
        )
        backend_readiness_errors, backend_completion_unknown = (
            self._prepare_materialization_backend_close_world(
                session,
                operation="cleanup Store backend close",
                deadline_unix_sec=close_deadline,
            )
        )
        if backend_readiness_errors:
            return self._record_materialization_commit(
                session,
                CommitWeightMaterializationReqOutput(
                    materialization_id=recv_req.materialization_id,
                    request_id=recv_req.request_id,
                    success=False,
                    message=" | ".join(backend_readiness_errors),
                    external_dp_rank=self._external_dp_rank(),
                    selected=selected,
                    ref=ref,
                    completion_unknown=backend_completion_unknown,
                    completion_ticket=(
                        None if previous is None else previous.completion_ticket
                    ),
                    session_state=(
                        WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING
                        if session.publication_ref is not None
                        else (
                            WeightMaterializationSessionState.COMPLETION_UNKNOWN
                            if backend_completion_unknown
                            else WeightMaterializationSessionState.CLEANUP_PENDING
                        )
                    ),
                    phase=recv_req.phase,
                ),
            )
        source_errors, completion_unknown = self._release_materialization_source_world(
            session,
            operation="cleanup source release",
        )
        if source_errors:
            return self._record_materialization_commit(
                session,
                CommitWeightMaterializationReqOutput(
                    materialization_id=recv_req.materialization_id,
                    request_id=recv_req.request_id,
                    success=False,
                    message=" | ".join(source_errors),
                    external_dp_rank=self._external_dp_rank(),
                    selected=selected,
                    ref=ref,
                    session_state=(
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN
                        if completion_unknown
                        else (
                            WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING
                            if session.publication_ref is not None
                            else WeightMaterializationSessionState.CLEANUP_PENDING
                        )
                    ),
                    completion_unknown=completion_unknown,
                    completion_ticket=(
                        None if previous is None else previous.completion_ticket
                    ),
                    phase=recv_req.phase,
                ),
            )
        backend_errors, backend_completion_unknown = (
            self._close_materialization_backend_world(
                session,
                operation="cleanup Store backend close",
                deadline_unix_sec=close_deadline,
                backend_prepared=True,
            )
        )
        if (
            not backend_errors
            and previous_cleanup_failed
            and not session.backend_close_succeeded
        ):
            prior_message = (
                previous.message
                if previous is not None and previous.message
                else "Store backend cleanup previously failed"
            )
            backend_errors = [
                f"{prior_message}; no terminal backend-close evidence is available"
            ]
        output = CommitWeightMaterializationReqOutput(
            materialization_id=recv_req.materialization_id,
            request_id=recv_req.request_id,
            success=not backend_errors,
            message=("Success." if not backend_errors else " | ".join(backend_errors)),
            external_dp_rank=self._external_dp_rank(),
            selected=selected,
            ref=ref,
            completion_unknown=backend_completion_unknown,
            completion_ticket=(
                None if previous is None else previous.completion_ticket
            ),
            session_state=(
                (
                    WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING
                    if backend_completion_unknown
                    else (
                        WeightMaterializationSessionState.PUBLISHED
                        if not backend_errors
                        else WeightMaterializationSessionState.PUBLISHED_CLEANUP_FAILED
                    )
                )
                if session.publication_ref is not None
                else (
                    WeightMaterializationSessionState.COMPLETION_UNKNOWN
                    if backend_completion_unknown
                    else (
                        WeightMaterializationSessionState.RELEASED
                        if not backend_errors
                        else WeightMaterializationSessionState.CLEANUP_PENDING
                    )
                )
            ),
            phase=recv_req.phase,
        )
        return self._record_materialization_commit(session, output)

    def _setup_materialization_backend_world(
        self,
        recv_req: CommitWeightMaterializationReqInput,
        session: _WeightMaterializationSession,
        *,
        execution_context: WeightTransferExecutionContext,
        deadline_unix_sec: float,
    ) -> _MaterializationBackendContext | CommitWeightMaterializationReqOutput:
        owner = session.backend_owner or _WeightStorageBackendOwner()
        backend = session.backend
        setup_error = None
        spec = None
        try:
            if session.source is None:
                raise RuntimeError("weight materialization source was already released")
            spec = WeightSnapshotWriteSpec.from_mapping(recv_req.storage_options)
            if backend is None:
                coordinator = self._weight_materialization_store_coordinator()
                backend = owner.enter_context(
                    open_weight_snapshot_write_backend(
                        spec,
                        local_placement_ids=session.local_selected_placement_ids,
                        payload_checksum_verifier=session.source.payload_checksum,
                        coordinator=coordinator,
                        execution_context=execution_context,
                    )
                )
                session.backend_close_succeeded = False
                session.backend_completion_unknown = False
            session.backend_owner = owner
            session.backend = backend
        except Exception as error:
            setup_error = str(error)
            session.backend_owner = owner
            session.backend = backend

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
                backend_errors, backend_completion_unknown = (
                    self._close_materialization_backend_world(
                        session,
                        operation="failed Store setup backend close",
                        deadline_unix_sec=deadline_unix_sec,
                    )
                )
                setup_failures.extend(backend_errors)
                completion_unknown |= backend_completion_unknown
            cleanup_pending = (
                session.source is not None or session.backend_owner is not None
            )
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(setup_failures),
                    session_state=(
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN
                        if completion_unknown
                        else (
                            WeightMaterializationSessionState.CLEANUP_PENDING
                            if cleanup_pending
                            else WeightMaterializationSessionState.FAILED
                        )
                    ),
                    completion_unknown=completion_unknown,
                ),
            )

        assert backend is not None
        assert spec is not None
        return _MaterializationBackendContext(
            owner=owner,
            backend=backend,
            spec=spec,
        )

    def _preflight_materialization_world(
        self,
        recv_req: CommitWeightMaterializationReqInput,
        session: _WeightMaterializationSession,
        setup: _MaterializationBackendContext,
        *,
        deadline_unix_sec: float,
    ) -> _MaterializationPreflightContext | CommitWeightMaterializationReqOutput:
        request = None
        preflight = None
        preflight_error = None
        try:
            attestor = self._materialization_attestor(session)
            store_coordinator = self._weight_materialization_store_coordinator()
            store_rank = getattr(store_coordinator, "rank", 0)
            request = prepare_weight_materialization(
                source_placements=session.selected_placements,
                source_bindings=session.selected_bindings,
                destination=setup.spec.destination,
                payload_identity=session.selected_payload_identity,
                operation_id=recv_req.materialization_id,
                source_placements_are_selected=(store_rank != 0),
            )
            preflight = preflight_weight_transfer(
                setup.backend.provider,
                request,
                attestor=attestor,
            )
        except Exception as error:
            preflight_error = str(error)

        try:
            preflight_statuses = self._gather_weight_materialization_objects(
                {
                    "success": preflight_error is None,
                    "message": preflight_error or "Success.",
                },
                operation="Store materialization preflight",
            )
        except Exception as error:
            preflight_statuses = [{"success": False, "message": str(error)}]
        preflight_failures = []
        for rank, status in enumerate(preflight_statuses):
            if (
                not isinstance(status, dict)
                or type(status.get("success")) is not bool
                or type(status.get("message")) is not str
            ):
                preflight_failures.append(
                    f"rank {rank}: invalid materialization preflight status"
                )
            elif not status["success"]:
                preflight_failures.append(f"rank {rank}: {status['message']}")
        if preflight_failures:
            cleanup_errors, completion_unknown = (
                self._release_materialization_source_world(
                    session,
                    operation="failed preflight source release",
                )
            )
            preflight_failures.extend(cleanup_errors)
            backend_errors, backend_completion_unknown = (
                self._close_materialization_backend_world(
                    session,
                    operation="failed preflight backend close",
                    deadline_unix_sec=deadline_unix_sec,
                )
            )
            preflight_failures.extend(backend_errors)
            completion_unknown |= backend_completion_unknown
            cleanup_pending = (
                session.source is not None
                or session.backend_owner is not None
                or session.backend is not None
            )
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(preflight_failures),
                    session_state=(
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN
                        if completion_unknown
                        else (
                            WeightMaterializationSessionState.CLEANUP_PENDING
                            if cleanup_pending
                            else WeightMaterializationSessionState.FAILED
                        )
                    ),
                    completion_unknown=completion_unknown,
                ),
            )

        assert request is not None
        assert preflight is not None
        return _MaterializationPreflightContext(
            owner=setup.owner,
            backend=setup.backend,
            request=request,
            preflight=preflight,
        )

    @staticmethod
    def _materialize_candidate(
        session: _WeightMaterializationSession,
        context: _MaterializationPreflightContext,
        *,
        execution_context: WeightTransferExecutionContext,
    ) -> _LocalMaterializationOutcome:
        try:
            prime_request = getattr(
                context.backend.catalog,
                "prime_materialization_request",
                None,
            )
            if callable(prime_request):
                prime_request(context.request)
            candidate = materialize_weight_snapshot_candidate(
                context.request,
                provider=context.backend.provider,
                catalog=context.backend.catalog,
                preflight=context.preflight,
                execution_context=execution_context,
            )
            return _LocalMaterializationOutcome(
                candidate=candidate,
                message="Success.",
            )
        except WeightTransferReleaseError as error:
            return _LocalMaterializationOutcome(
                candidate=None,
                message=str(error),
                finalize_pending=True,
                completion_ticket=getattr(
                    getattr(error, "receipt", None),
                    "completion_ticket",
                    None,
                ),
            )
        except WeightTransferError as error:
            if error.completion_known:
                return _LocalMaterializationOutcome(
                    candidate=None,
                    message=str(error),
                )
            completion_error = (
                error
                if isinstance(error, WeightTransferCompletionUnknownError)
                else WeightTransferCompletionUnknownError(
                    str(error),
                    provider=error.provider,
                    phase=error.phase,
                    operation_id=error.operation_id,
                    completion_ticket=getattr(error, "completion_ticket", None),
                )
            )
            assert session.source is not None
            session.source.quarantine(completion_error)
            return _LocalMaterializationOutcome(
                candidate=None,
                message=str(completion_error),
                completion_unknown=True,
                completion_ticket=completion_error.completion_ticket,
            )
        except Exception as error:
            return _LocalMaterializationOutcome(
                candidate=None,
                message=str(error),
            )

    def _materialization_outcome_payload(
        self,
        outcome: _LocalMaterializationOutcome,
        request: Any,
        provider: Any,
    ) -> dict[str, Any]:
        candidate = outcome.candidate
        if candidate is not None:
            assert candidate.snapshot is not None
        terminal_ref = None
        terminal_ref_for = getattr(
            provider,
            "materialization_terminal_ref",
            None,
        )
        if callable(terminal_ref_for):
            terminal_ref = terminal_ref_for(request.operation_id)
        if terminal_ref is None and candidate is not None:
            terminal_ref = candidate.snapshot.ref
        return {
            "success": candidate is not None,
            "message": outcome.message,
            "finalize_pending": outcome.finalize_pending,
            "completion_unknown": outcome.completion_unknown,
            "completion_ticket": outcome.completion_ticket,
            "ref": (
                None
                if terminal_ref is None
                else self._weight_storage_ref_builtins(terminal_ref)
            ),
            "snapshot_digest": (
                None if terminal_ref is None else terminal_ref.manifest_digest
            ),
            "model_identity": (
                None
                if candidate is None
                else {
                    "model_id": request.source_placements[0].model_id,
                    "revision": request.source_placements[0].revision,
                    "payload_digest": (
                        None
                        if request.payload_identity is None
                        else request.payload_identity.payload_digest
                    ),
                }
            ),
        }

    @staticmethod
    def _reduce_materialization_outcomes(
        materialization_statuses: list[Any],
    ) -> _ReducedMaterializationOutcome:
        status_errors = []
        success_refs = []
        success_snapshot_digests = []
        success_model_identities = []
        completion_tickets = []
        finalize_pending_messages = []
        completion_unknown = False
        success_count = 0
        for rank, status in enumerate(materialization_statuses):
            if (
                not isinstance(status, dict)
                or type(status.get("success")) is not bool
                or type(status.get("finalize_pending")) is not bool
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
                or (
                    status.get("snapshot_digest") is not None
                    and type(status.get("snapshot_digest")) is not str
                )
                or (
                    status.get("model_identity") is not None
                    and not isinstance(status.get("model_identity"), dict)
                )
            ):
                status_errors.append(
                    f"rank {rank}: invalid Store materialization status"
                )
                completion_unknown = True
                continue
            if status["completion_unknown"]:
                completion_unknown = True
            if status["completion_ticket"] is not None:
                completion_tickets.append(status["completion_ticket"])
            if status["success"]:
                success_count += 1
                if status["ref"] is None:
                    status_errors.append(
                        f"rank {rank}: successful materialization has no ref"
                    )
                    completion_unknown = True
                else:
                    success_refs.append(status["ref"])
                if not status["snapshot_digest"]:
                    status_errors.append(
                        f"rank {rank}: successful materialization has no "
                        "snapshot digest"
                    )
                    completion_unknown = True
                else:
                    success_snapshot_digests.append(status["snapshot_digest"])
                if status["model_identity"] is None:
                    status_errors.append(
                        f"rank {rank}: successful materialization has no model identity"
                    )
                    completion_unknown = True
                else:
                    success_model_identities.append(status["model_identity"])
            elif status["finalize_pending"]:
                finalize_pending_messages.append(f"rank {rank}: {status['message']}")
            else:
                status_errors.append(f"rank {rank}: {status['message']}")

        finalize_pending_count = len(finalize_pending_messages)
        if (success_count or finalize_pending_count) and (
            success_count + finalize_pending_count != len(materialization_statuses)
            or (success_count and finalize_pending_count)
        ):
            completion_unknown = True
            status_errors.append(
                "model ranks disagree on Store materialization completion"
            )
        ref = success_refs[0] if success_refs else None
        if ref is not None and any(item != ref for item in success_refs):
            completion_unknown = True
            status_errors.append(
                "model ranks materialized different weight storage refs"
            )
        if success_snapshot_digests and len(set(success_snapshot_digests)) != 1:
            completion_unknown = True
            status_errors.append("model ranks materialized different snapshot digests")
        if success_model_identities and any(
            item != success_model_identities[0] for item in success_model_identities[1:]
        ):
            completion_unknown = True
            status_errors.append("model ranks materialized different weight identities")
        unique_completion_tickets = set(completion_tickets)
        if len(unique_completion_tickets) > 1:
            completion_unknown = True
            status_errors.append(
                "model ranks reported different materialization recovery tickets"
            )
        completion_ticket = (
            completion_tickets[0] if len(unique_completion_tickets) == 1 else None
        )
        return _ReducedMaterializationOutcome(
            errors=tuple(status_errors),
            ref=ref,
            completion_ticket=completion_ticket,
            completion_unknown=completion_unknown,
            finalize_pending_messages=tuple(finalize_pending_messages),
            all_finalize_pending=(
                finalize_pending_count == len(materialization_statuses)
            ),
        )

    def _publish_materialization_candidate(
        self,
        recv_req: CommitWeightMaterializationReqInput,
        session: _WeightMaterializationSession,
        context: _MaterializationPreflightContext,
        candidate: Any,
        ref: dict[str, Any],
        *,
        execution_context: WeightTransferExecutionContext,
        deadline_unix_sec: float,
        external_dp_rank: int,
    ) -> CommitWeightMaterializationReqOutput:
        publication_error = None
        publication_completion_unknown = False
        try:
            publication = publish_weight_snapshot(
                candidate,
                catalog=context.backend.catalog,
                execution_context=execution_context,
            )
            terminal_ref_for = getattr(
                context.backend.provider,
                "materialization_terminal_ref",
                None,
            )
            terminal_ref = (
                terminal_ref_for(recv_req.materialization_id)
                if callable(terminal_ref_for)
                else None
            )
            published_ref = self._weight_storage_ref_builtins(
                publication.snapshot.ref if terminal_ref is None else terminal_ref
            )
            if published_ref != ref:
                raise ValueError(
                    "published weight storage ref differs from the "
                    "materialized candidate"
                )
        except Exception as error:
            publication_error = str(error)
            publication_completion_unknown = bool(
                getattr(error, "completion_unknown", False)
            ) or bool(
                getattr(self.weight_materialization_coordinator, "poisoned", False)
            )

        if publication_error is not None:
            if publication_completion_unknown:
                session.backend_owner = context.owner
                return self._record_materialization_commit(
                    session,
                    self._commit_materialization_failure(
                        recv_req,
                        publication_error,
                        session_state=(
                            WeightMaterializationSessionState.COMPLETION_UNKNOWN
                        ),
                        completion_unknown=True,
                    ),
                )
            cleanup_errors, release_completion_unknown = (
                self._release_materialization_source_world(
                    session,
                    operation="failed publication source release",
                )
            )
            backend_errors, backend_completion_unknown = (
                self._close_materialization_backend_world(
                    session,
                    operation="failed publication backend close",
                    deadline_unix_sec=deadline_unix_sec,
                )
            )
            errors = [
                publication_error,
                *cleanup_errors,
                *backend_errors,
            ]
            completion_unknown = (
                release_completion_unknown or backend_completion_unknown
            )
            cleanup_pending = (
                session.source is not None
                or session.backend_owner is not None
                or session.backend is not None
            )
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(errors),
                    session_state=(
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN
                        if completion_unknown
                        else (
                            WeightMaterializationSessionState.CLEANUP_PENDING
                            if cleanup_pending
                            else WeightMaterializationSessionState.FAILED
                        )
                    ),
                    completion_unknown=completion_unknown,
                ),
            )

        with self.weight_materialization_lock:
            if session.publication_ref is not None and session.publication_ref != ref:
                return self._commit_materialization_failure(
                    recv_req,
                    "materialization session published different storage refs",
                    session_state=WeightMaterializationSessionState.CONFLICT,
                )
            session.publication_ref = dict(ref)
            session.provider_finalize_pending = False

        backend_readiness_errors, backend_completion_unknown = (
            self._prepare_materialization_backend_close_world(
                session,
                operation="post-publication backend close",
                deadline_unix_sec=deadline_unix_sec,
            )
        )
        if backend_readiness_errors:
            return self._record_materialization_commit(
                session,
                CommitWeightMaterializationReqOutput(
                    materialization_id=recv_req.materialization_id,
                    request_id=recv_req.request_id,
                    success=True,
                    message=(
                        "Published; Store backend cleanup remains pending: "
                        + " | ".join(backend_readiness_errors)
                    ),
                    external_dp_rank=external_dp_rank,
                    selected=True,
                    ref=ref,
                    completion_unknown=backend_completion_unknown,
                    session_state=(
                        WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING
                    ),
                    phase=recv_req.phase,
                ),
            )
        source_errors = self._resolve_materialization_source_world(
            session,
            operation_id=recv_req.materialization_id,
            provider_name=context.backend.provider.name,
        )
        backend_errors, backend_completion_unknown = (
            self._close_materialization_backend_world(
                session,
                operation="post-publication backend close",
                deadline_unix_sec=deadline_unix_sec,
                backend_prepared=True,
            )
        )
        if source_errors:
            errors = [*source_errors, *backend_errors]
            return self._record_materialization_commit(
                session,
                CommitWeightMaterializationReqOutput(
                    materialization_id=recv_req.materialization_id,
                    request_id=recv_req.request_id,
                    success=True,
                    message="Published; cleanup remains pending: " + " | ".join(errors),
                    external_dp_rank=external_dp_rank,
                    selected=True,
                    ref=ref,
                    completion_unknown=backend_completion_unknown,
                    session_state=(
                        WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING
                    ),
                    phase=recv_req.phase,
                ),
            )

        message = "Success."
        state = WeightMaterializationSessionState.PUBLISHED
        if backend_errors:
            if backend_completion_unknown:
                message = (
                    "Published; Store backend cleanup remains pending: "
                    + " | ".join(backend_errors)
                )
                state = WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING
            else:
                message = "Published; Store backend close failed: " + " | ".join(
                    backend_errors
                )
                state = WeightMaterializationSessionState.PUBLISHED_CLEANUP_FAILED
            logger.warning(message)
        return self._record_materialization_commit(
            session,
            CommitWeightMaterializationReqOutput(
                materialization_id=recv_req.materialization_id,
                request_id=recv_req.request_id,
                success=True,
                message=message,
                external_dp_rank=external_dp_rank,
                selected=True,
                ref=ref,
                completion_unknown=backend_completion_unknown,
                session_state=state,
                phase=recv_req.phase,
            ),
        )

    def commit_weight_materialization(
        self,
        recv_req: CommitWeightMaterializationReqInput,
    ) -> CommitWeightMaterializationReqOutput:
        """Materialize the selected external DP replica into Mooncake Store."""

        try:
            commit_identity = self._commit_request_identity(recv_req)
        except Exception as error:
            return self._commit_materialization_failure(recv_req, str(error))
        cleanup_requested = recv_req.selected_external_dp_rank is None
        if cleanup_requested and recv_req.phase != "cleanup":
            return self._commit_materialization_failure(
                recv_req,
                "a cleanup request must use phase='cleanup'",
            )
        if not cleanup_requested and recv_req.phase == "cleanup":
            return self._commit_materialization_failure(
                recv_req,
                "a selected materialization cannot use phase='cleanup'",
            )

        retrying = False
        resume_published_cleanup = False
        with self.weight_materialization_lock:
            self._prune_weight_materialization_sessions_locked()
            session = self.weight_materialization_sessions.get(
                recv_req.materialization_id
            )
        deadline_unix_sec = None
        try:
            if session is not None:
                deadline_unix_sec = self._commit_materialization_deadline(
                    recv_req,
                    session,
                )
            elif getattr(recv_req, "deadline_unix_sec", None) is not None:
                deadline_unix_sec = self._validate_materialization_deadline(
                    recv_req.deadline_unix_sec
                )
        except ValueError as error:
            return self._commit_materialization_failure(recv_req, str(error))
        world_error = self._validate_materialization_commit_world(
            recv_req,
            session,
            commit_identity,
            deadline_unix_sec,
            cleanup_requested=cleanup_requested,
        )
        if world_error is not None:
            message, state = world_error
            return self._commit_materialization_failure(
                recv_req,
                message,
                session_state=state,
                completion_unknown=(
                    state is WeightMaterializationSessionState.COMPLETION_UNKNOWN
                ),
            )
        assert session is not None
        assert deadline_unix_sec is not None
        execution_context = self.weight_materialization_execution_context
        if execution_context is None:
            execution_context = WeightTransferExecutionContext(
                deadline_unix_sec=deadline_unix_sec,
                cancel_signal=self.weight_materialization_cancel_signal,
            )
        deadline_expired, deadline_error = self._materialization_deadline_expired_world(
            deadline_unix_sec
        )
        if deadline_error is not None:
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    deadline_error,
                    session_state=(
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN
                    ),
                    completion_unknown=True,
                    completion_ticket=(
                        None
                        if session.commit_output is None
                        else session.commit_output.completion_ticket
                    ),
                ),
            )
        if deadline_expired:
            prior_state = WeightMaterializationSessionState(session.state)
            completion_unknown = (
                session.commit_output is not None
                and session.commit_output.completion_unknown
            ) or prior_state is WeightMaterializationSessionState.COMPLETION_UNKNOWN
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    "weight materialization deadline expired",
                    session_state=(
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN
                        if completion_unknown
                        else WeightMaterializationSessionState.CLEANUP_PENDING
                    ),
                    completion_unknown=completion_unknown,
                    completion_ticket=(
                        None
                        if session.commit_output is None
                        else session.commit_output.completion_ticket
                    ),
                ),
            )

        with self.weight_materialization_lock:
            if cleanup_requested:
                session.state = WeightMaterializationSessionState.CLEANING
            else:
                previous = session.commit_output
                if (
                    session.commit_identity is not None
                    and session.commit_identity != commit_identity
                ):
                    return self._commit_materialization_failure(
                        recv_req,
                        "materialization ID is already bound to another destination",
                        session_state=WeightMaterializationSessionState.CONFLICT,
                    )
                previous_state = WeightMaterializationSessionState(session.state)
                if previous is not None:
                    previous_state = WeightMaterializationSessionState(
                        previous.session_state
                    )
                    retrying = (
                        previous.completion_unknown
                        or is_retryable_materialization_state(previous_state)
                    )
                    resume_published_cleanup = (
                        previous_state
                        is WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING
                        and session.publication_ref is not None
                    )
                    if not retrying:
                        return msgspec.structs.replace(
                            previous,
                            request_id=recv_req.request_id,
                            phase=recv_req.phase,
                        )
                session.commit_identity = commit_identity
                session.state = (
                    previous_state
                    if resume_published_cleanup
                    else (
                        WeightMaterializationSessionState.RECOVERING
                        if retrying
                        else WeightMaterializationSessionState.COMMITTING
                    )
                )

        if cleanup_requested or resume_published_cleanup:
            return self._cleanup_weight_materialization_session(recv_req, session)

        external_dp_rank = self._external_dp_rank()
        if external_dp_rank != recv_req.selected_external_dp_rank:
            source_errors, completion_unknown = (
                self._release_materialization_source_world(
                    session,
                    operation="unselected external DP source release",
                )
            )
            output = CommitWeightMaterializationReqOutput(
                materialization_id=recv_req.materialization_id,
                request_id=recv_req.request_id,
                success=not source_errors,
                message="Success." if not source_errors else " | ".join(source_errors),
                external_dp_rank=external_dp_rank,
                selected=False,
                session_state=(
                    WeightMaterializationSessionState.SKIPPED
                    if not source_errors
                    else (
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN
                        if completion_unknown
                        else WeightMaterializationSessionState.CLEANUP_PENDING
                    )
                ),
                completion_unknown=completion_unknown,
                phase=recv_req.phase,
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

        if not retrying:
            capacity_error = self._unresolved_materialization_capacity_error(
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
                            WeightMaterializationSessionState.COMPLETION_UNKNOWN
                            if completion_unknown
                            else (
                                WeightMaterializationSessionState.CLEANUP_PENDING
                                if cleanup_pending
                                else WeightMaterializationSessionState.FAILED
                            )
                        ),
                        completion_unknown=completion_unknown,
                    ),
                )

        setup = self._setup_materialization_backend_world(
            recv_req,
            session,
            execution_context=execution_context,
            deadline_unix_sec=deadline_unix_sec,
        )
        if not isinstance(setup, _MaterializationBackendContext):
            return setup
        context = self._preflight_materialization_world(
            recv_req,
            session,
            setup,
            deadline_unix_sec=deadline_unix_sec,
        )
        if not isinstance(context, _MaterializationPreflightContext):
            return context

        local_outcome = self._materialize_candidate(
            session,
            context,
            execution_context=execution_context,
        )
        try:
            materialization_statuses = self._gather_weight_materialization_objects(
                self._materialization_outcome_payload(
                    local_outcome,
                    context.request,
                    context.backend.provider,
                ),
                operation="Store materialization outcome",
            )
        except Exception as error:
            session.backend_owner = context.owner
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    str(error),
                    session_state=(
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN
                    ),
                    completion_unknown=True,
                ),
            )

        outcome = self._reduce_materialization_outcomes(materialization_statuses)
        if outcome.completion_unknown:
            session.backend_owner = context.owner
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(outcome.errors)
                    or "Store materialization completion is unknown",
                    session_state=(
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN
                    ),
                    completion_unknown=True,
                    completion_ticket=outcome.completion_ticket,
                ),
            )

        if outcome.all_finalize_pending:
            with self.weight_materialization_lock:
                session.provider_finalize_pending = True
                session.backend_owner = context.owner
            return self._record_materialization_commit(
                session,
                CommitWeightMaterializationReqOutput(
                    materialization_id=recv_req.materialization_id,
                    request_id=recv_req.request_id,
                    success=False,
                    message=(
                        "Provider finalize remains pending: "
                        + " | ".join(outcome.finalize_pending_messages)
                    ),
                    external_dp_rank=external_dp_rank,
                    selected=True,
                    ref=None,
                    completion_ticket=outcome.completion_ticket,
                    session_state=WeightMaterializationSessionState.FINALIZE_PENDING,
                    phase=recv_req.phase,
                ),
            )

        if outcome.errors or outcome.ref is None:
            status_errors = list(outcome.errors)
            cleanup_errors, release_completion_unknown = (
                self._release_materialization_source_world(
                    session,
                    operation="failed materialization source release",
                )
            )
            status_errors.extend(cleanup_errors)
            backend_errors, backend_completion_unknown = (
                self._close_materialization_backend_world(
                    session,
                    operation="failed materialization backend close",
                    deadline_unix_sec=deadline_unix_sec,
                )
            )
            status_errors.extend(backend_errors)
            release_completion_unknown |= backend_completion_unknown
            cleanup_pending = (
                session.source is not None
                or session.backend_owner is not None
                or session.backend is not None
            )
            return self._record_materialization_commit(
                session,
                self._commit_materialization_failure(
                    recv_req,
                    " | ".join(status_errors),
                    session_state=(
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN
                        if release_completion_unknown
                        else (
                            WeightMaterializationSessionState.CLEANUP_PENDING
                            if cleanup_pending
                            else WeightMaterializationSessionState.FAILED
                        )
                    ),
                    completion_unknown=release_completion_unknown,
                ),
            )

        assert local_outcome.candidate is not None
        return self._publish_materialization_candidate(
            recv_req,
            session,
            context,
            local_outcome.candidate,
            outcome.ref,
            execution_context=execution_context,
            deadline_unix_sec=deadline_unix_sec,
            external_dp_rank=external_dp_rank,
        )

    def _defer_remote_instance_weight_transfer(
        self,
        operation,
        recv_req,
        *,
        control: bool = False,
    ) -> Future:
        executor_name = "remote_weight_transfer_executor"
        thread_name_prefix = "sglang-weight-transfer"
        if control and self._can_run_remote_weight_transfer_control_independently():
            executor_name = "remote_weight_transfer_control_executor"
            thread_name_prefix = "sglang-weight-transfer-control"

        executor = getattr(self, executor_name)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=thread_name_prefix,
            )
            setattr(self, executor_name, executor)
        future = executor.submit(operation, recv_req)
        self.remote_weight_transfer_pending.append((future, recv_req))
        return future

    def _can_run_remote_weight_transfer_control_independently(self) -> bool:
        control_group = self.remote_weight_transfer_control_cpu_group
        snapshot_group = self.remote_weight_transfer_cpu_group or self.world_cpu_group
        if control_group is not None and control_group is not snapshot_group:
            return True
        try:
            return torch.distributed.get_world_size(group=snapshot_group) == 1
        except Exception:
            return False

    @staticmethod
    def _is_weight_materialization_cleanup_request(recv_req: Any) -> bool:
        return (
            isinstance(recv_req, CommitWeightMaterializationReqInput)
            and recv_req.selected_external_dp_rank is None
            and recv_req.phase == "cleanup"
        )

    def _cleanup_weight_materialization_rank_local(
        self,
        recv_req: CommitWeightMaterializationReqInput,
        *,
        poison_message: str,
    ) -> CommitWeightMaterializationReqOutput:
        with self.weight_materialization_lock:
            session = self.weight_materialization_sessions.get(
                recv_req.materialization_id
            )
        if session is None:
            return self._commit_materialization_failure(
                recv_req,
                f"rank-local cleanup could not find a session; {poison_message}",
                session_state=WeightMaterializationSessionState.COMPLETION_UNKNOWN,
                completion_unknown=True,
            )

        previous = session.commit_output
        selected = bool(previous is not None and previous.selected)
        ref = (
            dict(session.publication_ref)
            if selected and session.publication_ref is not None
            else None
        )
        errors = [poison_message]
        if session.provider_finalize_pending:
            errors.append("provider finalize remains pending")
        else:
            request_deadline = getattr(recv_req, "deadline_unix_sec", None)
            timeout_sec = (
                max(0.0, request_deadline - time.time())
                if isinstance(request_deadline, (int, float))
                and not isinstance(request_deadline, bool)
                else 0.0
            )
            backend_error = self._prepare_materialization_backend_close(
                session,
                timeout_sec=timeout_sec,
            )
            if backend_error is not None:
                errors.append(backend_error)
            else:
                source_error = self._release_materialization_session_source(session)
                if source_error is not None:
                    errors.append(source_error)
                else:
                    backend_error = self._close_materialization_backend_owner(session)
                    if backend_error is not None:
                        errors.append(backend_error)

        return self._record_materialization_commit(
            session,
            CommitWeightMaterializationReqOutput(
                materialization_id=recv_req.materialization_id,
                request_id=recv_req.request_id,
                success=False,
                message="rank-local cleanup only; " + " | ".join(errors),
                external_dp_rank=self._external_dp_rank(),
                selected=selected,
                ref=ref,
                completion_unknown=True,
                completion_ticket=(
                    None if previous is None else previous.completion_ticket
                ),
                session_state=(
                    WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING
                    if session.publication_ref is not None
                    else (
                        WeightMaterializationSessionState.FINALIZE_PENDING
                        if session.provider_finalize_pending
                        else WeightMaterializationSessionState.CLEANUP_PENDING
                    )
                ),
                phase=recv_req.phase,
            ),
        )

    def _defer_weight_materialization(self, operation, recv_req) -> None:
        with self.weight_materialization_lock:
            if not self.weight_materialization_accepting:
                raise RuntimeError("weight materialization is shutting down")
            if (
                self._is_weight_materialization_cleanup_request(recv_req)
                and self.weight_materialization_active_id == recv_req.materialization_id
                and self.weight_materialization_cancel_signal is not None
            ):
                self.weight_materialization_cancel_signal.set()
            if self.weight_materialization_executor is None:
                self.weight_materialization_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="sglang-weight-materialization",
                )
            cancel_signal = threading.Event()
            self.weight_materialization_cancel_signals.add(cancel_signal)
            future = self.weight_materialization_executor.submit(
                self._run_weight_materialization,
                operation,
                recv_req,
                cancel_signal,
            )
        self.weight_materialization_pending.append((future, recv_req))

    def _run_weight_materialization(self, operation, recv_req, cancel_signal):
        execution_context = None
        with self.weight_materialization_lock:
            self.weight_materialization_cancel_signal = cancel_signal
            self.weight_materialization_active_id = recv_req.materialization_id
            poison_message = self.weight_materialization_poisoned
        try:
            deadline_unix_sec = None
            deadline_expired = False
            preflight_error = None
            raw_deadline = getattr(recv_req, "deadline_unix_sec", None)
            if raw_deadline is None:
                preflight_error = "weight materialization deadline is required"
            else:
                try:
                    deadline_unix_sec = self._validate_materialization_deadline(
                        raw_deadline
                    )
                except ValueError as error:
                    preflight_error = str(error)
                else:
                    deadline_expired = (
                        deadline_unix_sec - time.time()
                        <= _WEIGHT_MATERIALIZATION_COLLECTIVE_GRACE_SEC
                    )
                    if deadline_expired:
                        preflight_error = (
                            "weight materialization deadline does not leave enough "
                            "time for admission"
                        )
            admission_context = WeightTransferExecutionContext(
                deadline_unix_sec=(
                    time.time() + _WEIGHT_MATERIALIZATION_CONTROL_TIMEOUT_SEC
                ),
            )
            execution_context = admission_context
            with self.weight_materialization_lock:
                self.weight_materialization_execution_context = admission_context
            try:
                votes = self._gather_weight_materialization_objects(
                    {
                        "cancelled": cancel_signal.is_set(),
                        "deadline_expired": deadline_expired,
                        "error": preflight_error,
                        "poisoned": poison_message,
                    },
                    operation="materialization admission vote",
                )
            except Exception:
                with self.weight_materialization_lock:
                    poison_message = self.weight_materialization_poisoned
                if (
                    poison_message is not None
                    and self._is_weight_materialization_cleanup_request(recv_req)
                ):
                    return self._cleanup_weight_materialization_rank_local(
                        recv_req,
                        poison_message=poison_message,
                    )
                raise
            if any(
                not isinstance(vote, dict)
                or type(vote.get("cancelled")) is not bool
                or type(vote.get("deadline_expired")) is not bool
                or (
                    vote.get("error") is not None and type(vote.get("error")) is not str
                )
                or (
                    vote.get("poisoned") is not None
                    and type(vote.get("poisoned")) is not str
                )
                for vote in votes
            ):
                message = (
                    "weight materialization admission vote is invalid; "
                    "scheduler restart is required"
                )
                with self.weight_materialization_lock:
                    self.weight_materialization_poisoned = message
                raise RuntimeError(message)
            poisoned_votes = [
                f"rank {rank}: {vote['poisoned']}"
                for rank, vote in enumerate(votes)
                if vote["poisoned"] is not None
            ]
            if poisoned_votes:
                poison_message = (
                    "weight materialization admission found a poisoned model rank: "
                    + " | ".join(poisoned_votes)
                )
                with self.weight_materialization_lock:
                    self.weight_materialization_poisoned = poison_message
                if self._is_weight_materialization_cleanup_request(recv_req):
                    return self._cleanup_weight_materialization_rank_local(
                        recv_req,
                        poison_message=poison_message,
                    )
                raise RuntimeError(poison_message)
            preflight_errors = [
                f"rank {rank}: {vote['error']}"
                for rank, vote in enumerate(votes)
                if vote["error"] is not None
            ]
            if preflight_errors:
                raise RuntimeError(" | ".join(preflight_errors))
            if any(vote["cancelled"] for vote in votes):
                raise RuntimeError("weight materialization was cancelled")
            if any(vote["deadline_expired"] for vote in votes):
                raise RuntimeError("weight materialization deadline expired")
            assert deadline_unix_sec is not None
            execution_context = WeightTransferExecutionContext(
                deadline_unix_sec=deadline_unix_sec,
                cancel_signal=cancel_signal,
            )
            with self.weight_materialization_lock:
                self.weight_materialization_execution_context = execution_context
            remaining = execution_context.remaining_seconds()
            try:
                torch.distributed.monitored_barrier(
                    group=self._weight_materialization_collective_group(),
                    timeout=timedelta(seconds=max(remaining, 0.001)),
                    wait_all_ranks=True,
                )
            except BaseException as error:
                message = (
                    "weight materialization process group is unusable; "
                    f"scheduler restart is required: {error}"
                )
                with self.weight_materialization_lock:
                    self.weight_materialization_poisoned = message
                raise RuntimeError(message) from error
            return operation(recv_req)
        finally:
            with self.weight_materialization_lock:
                shutdown_session = (
                    self.weight_materialization_sessions.get(
                        recv_req.materialization_id
                    )
                    if not self.weight_materialization_accepting
                    else None
                )
            if shutdown_session is not None:
                self._cleanup_materialization_session_for_shutdown(
                    recv_req.materialization_id,
                    shutdown_session,
                    timeout_sec=0.0,
                )
            with self.weight_materialization_lock:
                if self.weight_materialization_cancel_signal is cancel_signal:
                    self.weight_materialization_cancel_signal = None
                if self.weight_materialization_active_id == recv_req.materialization_id:
                    self.weight_materialization_active_id = None
                if self.weight_materialization_execution_context is execution_context:
                    self.weight_materialization_execution_context = None
                self.weight_materialization_cancel_signals.discard(cancel_signal)

    def defer_begin_remote_instance_weight_transfer(
        self, recv_req: BeginRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._mark_remote_weight_transfer_begin(recv_req.transfer_id)
        try:
            future = self._defer_remote_instance_weight_transfer(
                self.begin_remote_instance_weight_transfer,
                recv_req,
            )
        except BaseException:
            self._finish_remote_weight_transfer_begin(recv_req.transfer_id)
            raise
        future.add_done_callback(
            lambda _future: self._finish_remote_weight_transfer_begin(
                recv_req.transfer_id
            )
        )

    def defer_release_remote_instance_weight_transfer(
        self, recv_req: ReleaseRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.release_remote_instance_weight_transfer,
            recv_req,
            control=True,
        )

    def defer_get_remote_instance_weight_transfer_session(
        self, recv_req: GetRemoteInstanceWeightTransferSessionReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.get_remote_instance_weight_transfer_session,
            recv_req,
            control=True,
        )

    def defer_renew_remote_instance_weight_transfer(
        self, recv_req: RenewRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.renew_remote_instance_weight_transfer,
            recv_req,
            control=True,
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
            kwargs["request_id"] = recv_req.request_id
            kwargs["external_dp_rank"] = self._external_dp_rank()
            return BeginRemoteInstanceWeightTransferReqOutput(**kwargs)
        if isinstance(recv_req, ReleaseRemoteInstanceWeightTransferReqInput):
            kwargs["request_id"] = getattr(recv_req, "request_id", None)
            kwargs["external_dp_rank"] = self._external_dp_rank()
            return ReleaseRemoteInstanceWeightTransferReqOutput(**kwargs)
        if isinstance(recv_req, GetRemoteInstanceWeightTransferSessionReqInput):
            kwargs["request_id"] = getattr(recv_req, "request_id", None)
            kwargs["external_dp_rank"] = self._external_dp_rank()
            kwargs["session_state"] = "failed"
            return GetRemoteInstanceWeightTransferSessionReqOutput(**kwargs)
        if isinstance(recv_req, RenewRemoteInstanceWeightTransferReqInput):
            kwargs["request_id"] = getattr(recv_req, "request_id", None)
            kwargs["external_dp_rank"] = self._external_dp_rank()
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
        shutdown_deadline = (
            time.monotonic() + _WEIGHT_MATERIALIZATION_SHUTDOWN_TIMEOUT_SEC
        )
        with self.weight_materialization_lock:
            self.weight_materialization_accepting = False
            cancel_signals = tuple(self.weight_materialization_cancel_signals)
            executor = self.weight_materialization_executor
            pending_futures = tuple(
                future for future, _request in self.weight_materialization_pending
            )
            remote_pending_futures = tuple(
                future for future, _request in self.remote_weight_transfer_pending
            )
        for cancel_signal in cancel_signals:
            cancel_signal.set()
        remote_executor = self.remote_weight_transfer_executor
        if remote_executor is not None:
            remote_executor.shutdown(wait=False, cancel_futures=True)
            self.remote_weight_transfer_executor = None
        control_executor = self.remote_weight_transfer_control_executor
        if control_executor is not None:
            control_executor.shutdown(wait=False, cancel_futures=True)
            self.remote_weight_transfer_control_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        shutdown_futures = (*remote_pending_futures, *pending_futures)
        if shutdown_futures:
            wait_futures(
                shutdown_futures,
                timeout=max(0.0, shutdown_deadline - time.monotonic()),
            )
        with self.weight_materialization_lock:
            if self.weight_materialization_executor is executor:
                self.weight_materialization_executor = None
            active_id = self.weight_materialization_active_id
            active_signal = self.weight_materialization_cancel_signal
            if active_id is None:
                self.weight_materialization_cancel_signal = None
                self.weight_materialization_cancel_signals.clear()
            elif active_signal is not None:
                self.weight_materialization_cancel_signals.intersection_update(
                    {active_signal}
                )

        with self.weight_materialization_lock:
            sessions = tuple(self.weight_materialization_sessions.items())
        for materialization_id, session in sessions:
            if materialization_id == active_id:
                logger.warning(
                    "Skipping weight materialization shutdown cleanup while the "
                    "serial lane still owns session %s",
                    materialization_id,
                )
                continue
            self._cleanup_materialization_session_for_shutdown(
                materialization_id,
                session,
                timeout_sec=max(0.0, shutdown_deadline - time.monotonic()),
            )

    def begin_remote_instance_weight_transfer(
        self, recv_req: BeginRemoteInstanceWeightTransferReqInput
    ) -> BeginRemoteInstanceWeightTransferReqOutput:
        """Acquire one address-stable snapshot on every model rank."""
        response_identity = {
            "request_id": recv_req.request_id,
            "external_dp_rank": self._external_dp_rank(),
        }
        deadline_unix_sec = recv_req.deadline_unix_sec
        poisoned = self._remote_weight_transfer_snapshot_poison()
        collective_group = None
        control_group = None
        cached_session = None
        cached_output = None
        local_lease_id = None
        local_generation = None
        local_lease_fence = None
        preflight_success = True
        preflight_message = "Success."
        preflight_state = "created"
        try:
            if deadline_unix_sec is None:
                raise _RemoteWeightTransferSessionError(
                    "Remote weight transfer deadline is required.",
                    session_state="failed",
                )
            if (
                isinstance(deadline_unix_sec, bool)
                or not isinstance(deadline_unix_sec, (int, float))
                or not math.isfinite(deadline_unix_sec)
                or deadline_unix_sec <= time.time()
            ):
                raise _RemoteWeightTransferSessionError(
                    "Remote weight transfer deadline is invalid or expired.",
                    session_state="failed",
                )
            deadline_unix_sec = float(deadline_unix_sec)
            if poisoned is not None:
                raise _RemoteWeightTransferSessionError(
                    "Remote weight transfer snapshot lane is unavailable; "
                    f"scheduler restart is required: {poisoned}",
                    session_state=(
                        "cleanup_pending"
                        if self._get_remote_weight_transfer_lease(recv_req.transfer_id)
                        is not None
                        else "failed"
                    ),
                )
            validate_manifest_revision_semantics(
                recv_req.manifest_format,
                recv_req.manifest_revision_semantics,
            )
            if recv_req.lease_fence is not None and (
                type(recv_req.lease_fence) is not str
                or not recv_req.lease_fence.startswith(
                    _REMOTE_WEIGHT_TRANSFER_BEGIN_FENCE_PREFIX
                )
            ):
                raise _RemoteWeightTransferSessionError(
                    "Remote weight transfer begin fence is invalid.",
                    session_state="failed",
                )
            cached_session = self._cached_remote_weight_transfer_session(recv_req)
            with self.remote_weight_transfer_lock:
                legacy_reuse_blocked = (
                    recv_req.transfer_id
                    in self.remote_weight_transfer_legacy_reuse_blocked
                )
            if (
                legacy_reuse_blocked
                and recv_req.lease_fence is None
                and cached_session is None
            ):
                raise _RemoteWeightTransferSessionError(
                    "Remote weight transfer ID reuse requires a lease fence.",
                    session_state="conflict",
                )
            if cached_session is not None:
                _, cached_output = cached_session
                local_lease_id = self._get_remote_weight_transfer_lease(
                    recv_req.transfer_id
                )
                with self.remote_weight_transfer_lock:
                    local_generation = self.remote_weight_transfer_generations.get(
                        recv_req.transfer_id
                    )
                    local_lease_fence = self.remote_weight_transfer_fences.get(
                        recv_req.transfer_id
                    )
                if (
                    type(local_lease_id) is not str
                    or not local_lease_id
                    or type(local_generation) is not int
                    or local_generation <= 0
                ):
                    raise _RemoteWeightTransferSessionError(
                        "cached remote weight transfer bookkeeping is incomplete",
                        session_state="cleanup_pending",
                    )
                preflight_state = "reused"
            elif (
                self._get_remote_weight_transfer_lease(recv_req.transfer_id) is not None
            ):
                raise _RemoteWeightTransferSessionError(
                    f"remote weight transfer already exists: {recv_req.transfer_id}",
                    session_state="cleanup_pending",
                )
            if not self._consume_remote_weight_transfer_begin_fence(
                transfer_id=recv_req.transfer_id,
                lease_fence=recv_req.lease_fence,
                deadline_unix_sec=deadline_unix_sec,
                active=cached_session is not None,
            ):
                raise _RemoteWeightTransferSessionError(
                    "Remote weight transfer begin fence was already consumed.",
                    session_state="conflict",
                )
            collective_group = self._remote_weight_transfer_collective(
                self.remote_weight_transfer_cpu_group or self.world_cpu_group,
                control=False,
            )
            control_group = self._remote_weight_transfer_control_group()
        except Exception as error:
            preflight_success = False
            preflight_message = str(error)
            preflight_state = getattr(error, "session_state", "failed")

        proposed_lease_fence = None
        if preflight_success and recv_req.lease_fence is not None:
            if preflight_state == "reused":
                proposed_lease_fence = local_lease_fence
            if (
                proposed_lease_fence is None
                and self._remote_transfer_group_rank(control_group) == 0
            ):
                proposed_lease_fence = (
                    f"{_REMOTE_WEIGHT_TRANSFER_LEASE_FENCE_PREFIX}{token_urlsafe(32)}"
                )
        local_vote = {
            "success": preflight_success,
            "message": preflight_message,
            "session_state": preflight_state,
            "poisoned": poisoned is not None,
            "lease_fence": proposed_lease_fence,
        }
        control_deadline_unix_sec = (
            time.time() + _REMOTE_WEIGHT_TRANSFER_CONTROL_TIMEOUT_SEC
        )
        if (
            not isinstance(deadline_unix_sec, bool)
            and isinstance(deadline_unix_sec, (int, float))
            and math.isfinite(deadline_unix_sec)
            and deadline_unix_sec > 0
        ):
            control_deadline_unix_sec = min(
                float(deadline_unix_sec),
                control_deadline_unix_sec,
            )
        control_context = WeightTransferExecutionContext(
            deadline_unix_sec=control_deadline_unix_sec,
        )
        try:
            begin_votes = self._all_gather_remote_transfer_object(
                local_vote,
                control_group or self._remote_weight_transfer_control_group(),
                phase="remote_instance.source.begin_vote",
                execution_context=control_context,
            )
        except Exception as error:
            message = f"Remote weight transfer begin vote failed: {error}"
            self._poison_remote_weight_transfer_snapshot_lane(message)
            has_local_lease = (
                self._get_remote_weight_transfer_lease(recv_req.transfer_id) is not None
            )
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=(
                    "Remote weight transfer snapshot lane is unavailable; "
                    f"scheduler restart is required: {message}"
                ),
                session_state=("cleanup_pending" if has_local_lease else "failed"),
                **response_identity,
            )

        if any(
            not isinstance(vote, dict)
            or type(vote.get("success")) is not bool
            or type(vote.get("message")) is not str
            or type(vote.get("session_state")) is not str
            or type(vote.get("poisoned")) is not bool
            or (
                vote.get("lease_fence") is not None
                and type(vote.get("lease_fence")) is not str
            )
            for vote in begin_votes
        ):
            message = "Remote weight transfer begin vote is invalid"
            self._poison_remote_weight_transfer_snapshot_lane(message)
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=(
                    "Remote weight transfer snapshot lane is unavailable; "
                    f"scheduler restart is required: {message}"
                ),
                session_state=(
                    "cleanup_pending"
                    if self._get_remote_weight_transfer_lease(recv_req.transfer_id)
                    is not None
                    else "failed"
                ),
                **response_identity,
            )
        poisoned_votes = [
            f"rank {rank}: {vote['message']}"
            for rank, vote in enumerate(begin_votes)
            if vote["poisoned"]
        ]
        if poisoned_votes:
            message = " | ".join(poisoned_votes)
            self._poison_remote_weight_transfer_snapshot_lane(message)
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=message,
                session_state=self._remote_transfer_begin_vote_failure_state(
                    begin_votes
                ),
                **response_identity,
            )
        failures = [
            f"rank {rank}: {vote['message']}"
            for rank, vote in enumerate(begin_votes)
            if not vote["success"]
        ]
        if failures:
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=" | ".join(failures),
                session_state=self._remote_transfer_begin_vote_failure_state(
                    begin_votes
                ),
                **response_identity,
            )
        begin_states = {vote["session_state"] for vote in begin_votes}
        if len(begin_states) != 1:
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=(
                    "model ranks have inconsistent session state before "
                    "snapshot capture"
                ),
                session_state=(
                    "cleanup_pending" if "reused" in begin_states else "conflict"
                ),
                **response_identity,
            )
        begin_state = next(iter(begin_states))
        vote_fences = [
            vote["lease_fence"]
            for vote in begin_votes
            if vote["lease_fence"] is not None
        ]
        if begin_state == "reused":
            unique_fences = set(vote_fences)
            valid_lease_fence = len(unique_fences) <= 1 and (
                not unique_fences
                or next(iter(unique_fences)).startswith(
                    _REMOTE_WEIGHT_TRANSFER_LEASE_FENCE_PREFIX
                )
            )
        elif recv_req.lease_fence is None:
            valid_lease_fence = not vote_fences
        else:
            valid_lease_fence = len(vote_fences) == 1 and vote_fences[0].startswith(
                _REMOTE_WEIGHT_TRANSFER_LEASE_FENCE_PREFIX
            )
        if not valid_lease_fence:
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message="model ranks returned an invalid lease incarnation",
                session_state=(
                    "cleanup_pending" if begin_state == "reused" else "failed"
                ),
                **response_identity,
            )
        authoritative_lease_fence = vote_fences[0] if vote_fences else None
        if deadline_unix_sec <= time.time():
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=(
                    "Remote weight transfer deadline expired before "
                    "snapshot acquisition."
                ),
                session_state=(
                    "cleanup_pending" if begin_state == "reused" else "failed"
                ),
                **response_identity,
            )
        if begin_state == "reused" and recv_req.lease_fence is not None:
            with self.remote_weight_transfer_lock:
                self.remote_weight_transfer_begin_fences[recv_req.transfer_id] = (
                    recv_req.lease_fence
                )
                if local_lease_fence is None and authoritative_lease_fence is not None:
                    self.remote_weight_transfer_fences[recv_req.transfer_id] = (
                        authoritative_lease_fence
                    )
        if (
            begin_state == "reused"
            and local_lease_fence is None
            and authoritative_lease_fence is not None
        ):
            local_lease_fence = authoritative_lease_fence

        assert collective_group is not None
        assert deadline_unix_sec is not None
        collective_context = WeightTransferExecutionContext(
            deadline_unix_sec=deadline_unix_sec,
        )
        group_rank = self._remote_transfer_group_rank(collective_group)
        is_response_root = group_rank == 0
        acquired_lease_id = None
        split_manifest = recv_req.manifest_format == PLACEMENT_BINDING_V1
        try:
            if cached_session is not None:
                local_result = {
                    "success": True,
                    "message": "Success.",
                    "session_state": "reused",
                    "manifest_revision_semantics": (
                        recv_req.manifest_revision_semantics
                    ),
                    "model_id": recv_req.model_id,
                    "revision": recv_req.revision,
                    "lease_id": local_lease_id,
                    "generation": local_generation,
                    "lease_fence": local_lease_fence,
                }
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
                acquired_lease_id = self._remote_transfer_snapshot_lease_id(
                    local_snapshot,
                    split_manifest=split_manifest,
                )
                self._record_remote_weight_transfer_lease(
                    recv_req.transfer_id,
                    acquired_lease_id,
                    recv_req.lease_timeout_sec,
                    lease_fence=authoritative_lease_fence,
                    begin_fence=recv_req.lease_fence,
                )
                local_lease_fence = authoritative_lease_fence
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
                        "manifest_revision_semantics": (
                            recv_req.manifest_revision_semantics
                        ),
                        "model_id": placement_payload.get("model_id"),
                        "revision": placement_payload.get("revision"),
                        "lease_id": acquired_lease_id,
                        "generation": local_generation,
                        "lease_fence": local_lease_fence,
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
                        "manifest_revision_semantics": (
                            recv_req.manifest_revision_semantics
                        ),
                        "model_id": local_payload.get("model_id"),
                        "revision": local_payload.get("revision"),
                        "lease_id": acquired_lease_id,
                        "generation": local_generation,
                        "lease_fence": local_lease_fence,
                    }
        except Exception as error:
            local_result = {
                "success": False,
                "message": str(error),
                "session_state": getattr(error, "session_state", "failed"),
            }

        try:
            world_size = self._remote_transfer_group_size(collective_group)
            gathered = self._gather_remote_transfer_object(
                local_result,
                collective_group,
                phase="remote_instance.source.manifest_gather",
                execution_context=collective_context,
            )
        except Exception as error:
            if getattr(error, "completion_unknown", None) is not False:
                self._poison_remote_weight_transfer_snapshot_lane(
                    f"manifest gather failed: {error}"
                )
            cleanup_error = None
            if acquired_lease_id is not None:
                cleanup_error = self._rollback_remote_weight_transfer_snapshot(
                    recv_req.transfer_id,
                    acquired_lease_id,
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
                **response_identity,
            )

        root_output = None
        decision = None
        if is_response_root:
            try:
                assert gathered is not None
                decision, root_output = self._reduce_remote_transfer_begin(
                    recv_req,
                    gathered,
                    world_size=world_size,
                    cached_output=cached_output,
                    response_identity=response_identity,
                )
            except Exception as error:
                decision = {
                    "success": False,
                    "message": str(error),
                    "session_state": "failed",
                    "manifest_revision_semantics": None,
                }

        try:
            decision = self._broadcast_remote_transfer_object(
                decision,
                collective_group,
                phase="remote_instance.source.decision_scatter",
                execution_context=collective_context,
            )
            decision = self._validate_remote_transfer_decision(decision)
        except Exception as error:
            self._poison_remote_weight_transfer_snapshot_lane(
                f"manifest decision broadcast failed: {error}"
            )
            cleanup_error = None
            if acquired_lease_id is not None:
                cleanup_error = self._rollback_remote_weight_transfer_snapshot(
                    recv_req.transfer_id,
                    acquired_lease_id,
                )
            message = f"Failed to broadcast source manifest decision: {error}"
            if cleanup_error is not None:
                message += f"; snapshot cleanup remains pending: {cleanup_error}"
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=message,
                session_state=(
                    "cleanup_pending" if cleanup_error is not None else "failed"
                ),
                **response_identity,
            )

        if not decision["success"]:
            cleanup_error = None
            if acquired_lease_id is not None:
                cleanup_error = self._rollback_remote_weight_transfer_snapshot(
                    recv_req.transfer_id,
                    acquired_lease_id,
                )
            cleanup_result = {
                "success": cleanup_error is None,
                "message": cleanup_error or "Success.",
            }
            try:
                cleanup_results = self._gather_remote_transfer_object(
                    cleanup_result,
                    collective_group,
                    phase="remote_instance.source.cleanup_gather",
                    execution_context=collective_context,
                )
                final_decision = None
                if is_response_root:
                    assert cleanup_results is not None
                    final_decision = self._finalize_remote_transfer_failure(
                        decision,
                        cleanup_results,
                    )
                decision = self._broadcast_remote_transfer_object(
                    final_decision,
                    collective_group,
                    phase="remote_instance.source.cleanup_scatter",
                    execution_context=collective_context,
                )
                decision = self._validate_remote_transfer_decision(decision)
            except Exception as error:
                self._poison_remote_weight_transfer_snapshot_lane(
                    f"snapshot cleanup coordination failed: {error}"
                )
                message = (
                    f"{decision['message']} | failed to coordinate snapshot "
                    f"cleanup: {error}"
                )
                if cleanup_error is not None:
                    message += f" | snapshot cleanup remains pending: {cleanup_error}"
                return BeginRemoteInstanceWeightTransferReqOutput(
                    transfer_id=recv_req.transfer_id,
                    success=False,
                    message=message,
                    session_state="cleanup_pending",
                    **response_identity,
                )
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=decision["message"],
                session_state=decision["session_state"],
                manifest_revision_semantics=(
                    decision["manifest_revision_semantics"] or HF_REVISION_V1
                ),
                **response_identity,
            )

        session_state = decision["session_state"]
        consensus_semantics = decision["manifest_revision_semantics"]
        assert consensus_semantics is not None
        if session_state == "reused":
            if is_response_root:
                assert root_output is not None
                return root_output
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=True,
                message="Success.",
                session_state="reused",
                manifest_revision_semantics=consensus_semantics,
                lease_fence=decision.get("lease_fence"),
                generation=decision.get("generation"),
                **response_identity,
            )

        if acquired_lease_id is None or local_generation is None:
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message="Source snapshot decision does not match local ownership.",
                session_state="cleanup_pending",
                **response_identity,
            )
        self._record_remote_weight_transfer_session(
            recv_req,
            acquired_lease_id,
            root_output if is_response_root else None,
            generation=local_generation,
        )
        if is_response_root:
            assert root_output is not None
            return root_output
        return BeginRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=True,
            message="Success.",
            session_state="created",
            manifest_revision_semantics=consensus_semantics,
            lease_fence=decision.get("lease_fence"),
            generation=decision.get("generation"),
            **response_identity,
        )

    @staticmethod
    def _remote_transfer_group_rank(group: Any) -> int:
        rank = getattr(group, "rank_in_group", None)
        if rank is None:
            rank = getattr(group, "rank", None)
        if type(rank) is int:
            return rank
        return torch.distributed.get_rank(group=group)

    @staticmethod
    def _remote_transfer_group_size(group: Any) -> int:
        world_size = getattr(group, "world_size", None)
        if type(world_size) is int:
            return world_size
        return torch.distributed.get_world_size(group=group)

    @staticmethod
    def _remote_transfer_root_global_rank(group: Any) -> int:
        ranks = getattr(group, "ranks", None)
        if ranks:
            return ranks[0]
        get_global_rank = getattr(torch.distributed, "get_global_rank", None)
        if callable(get_global_rank):
            return get_global_rank(group, 0)
        return 0

    def _gather_remote_transfer_object(
        self,
        value: Any,
        group: Any,
        *,
        phase: str | None = None,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> list | None:
        gather_to_root = getattr(group, "gather_object_to_root", None)
        if callable(gather_to_root):
            return gather_to_root(
                value,
                phase=phase or "remote_instance.source.gather",
                execution_context=execution_context,
            )
        gather = getattr(group, "gather_object", None)
        if callable(gather):
            if execution_context is not None and getattr(group, "cpu_group", None):
                return gather(
                    value,
                    dst=0,
                    phase=phase,
                    execution_context=execution_context,
                )
            return gather(value, dst=0)
        gathered = (
            [None] * self._remote_transfer_group_size(group)
            if self._remote_transfer_group_rank(group) == 0
            else None
        )
        torch.distributed.gather_object(
            value,
            gathered,
            dst=self._remote_transfer_root_global_rank(group),
            group=group,
        )
        return gathered

    def _broadcast_remote_transfer_object(
        self,
        value: Any,
        group: Any,
        *,
        phase: str | None = None,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any:
        scatter_from_root = getattr(group, "scatter_object_from_root", None)
        if callable(scatter_from_root):
            values = (
                [value] * self._remote_transfer_group_size(group)
                if self._remote_transfer_group_rank(group) == 0
                else None
            )
            return scatter_from_root(
                values,
                phase=phase or "remote_instance.source.scatter",
                execution_context=execution_context,
            )
        scatter = getattr(group, "scatter_object", None)
        if callable(scatter):
            values = (
                [value] * self._remote_transfer_group_size(group)
                if self._remote_transfer_group_rank(group) == 0
                else None
            )
            if execution_context is not None and getattr(group, "cpu_group", None):
                return scatter(
                    values,
                    src=0,
                    phase=phase,
                    execution_context=execution_context,
                )
            return scatter(values, src=0)
        broadcast = getattr(group, "broadcast_object", None)
        if callable(broadcast):
            return broadcast(value, src=0)
        payload = [value if self._remote_transfer_group_rank(group) == 0 else None]
        torch.distributed.broadcast_object_list(
            payload,
            src=self._remote_transfer_root_global_rank(group),
            group=group,
        )
        return payload[0]

    def _all_gather_remote_transfer_object(
        self,
        value: Any,
        group: Any,
        *,
        phase: str | None = None,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> list:
        all_gather = getattr(group, "all_gather_object", None)
        if callable(all_gather):
            if execution_context is not None and (
                getattr(group, "cpu_group", None)
                or callable(getattr(group, "gather_object_to_root", None))
            ):
                return all_gather(
                    value,
                    phase=phase,
                    execution_context=execution_context,
                )
            return all_gather(value)
        gathered = [None] * self._remote_transfer_group_size(group)
        torch.distributed.all_gather_object(
            gathered,
            value,
            group=group,
        )
        return gathered

    def _reduce_remote_transfer_begin(
        self,
        request: BeginRemoteInstanceWeightTransferReqInput,
        gathered: list,
        *,
        world_size: int,
        cached_output: BeginRemoteInstanceWeightTransferReqOutput | None,
        response_identity: dict[str, Any],
    ) -> tuple[dict[str, Any], BeginRemoteInstanceWeightTransferReqOutput | None]:
        normalized = []
        failures = []
        rank_semantics = set()
        for rank, item in enumerate(gathered):
            if (
                not isinstance(item, dict)
                or type(item.get("success")) is not bool
                or type(item.get("message")) is not str
            ):
                item = {
                    "success": False,
                    "message": f"source rank {rank} returned an invalid result",
                    "session_state": "failed",
                }
            normalized.append(item)
            if not item["success"]:
                failures.append(item["message"])
                continue
            semantics = item.get("manifest_revision_semantics", HF_REVISION_V1)
            try:
                validate_manifest_revision_semantics(
                    request.manifest_format,
                    semantics,
                )
            except Exception as error:
                failures.append(f"source rank {rank}: {error}")
                continue
            model_id = item.get("model_id")
            revision = item.get("revision")
            if not model_id or not revision:
                record = item.get("placement") or item.get("manifest")
                try:
                    model_id, revision = self._remote_transfer_record_identity(record)
                except Exception as error:
                    failures.append(f"source rank {rank}: {error}")
                    continue
            if (
                model_id != request.model_id
                or revision != request.revision
                or semantics != request.manifest_revision_semantics
            ):
                failures.append(
                    f"source rank {rank} returned incompatible weight identity"
                )
                continue
            rank_semantics.add(semantics)

        if not failures and len(rank_semantics) != 1:
            failures.append(
                "source ranks returned inconsistent manifest revision semantics"
            )
        consensus_semantics = (
            next(iter(rank_semantics)) if len(rank_semantics) == 1 else None
        )
        session_states = {
            item.get("session_state", "created")
            for item in normalized
            if item["success"]
        }
        if not failures and len(session_states) != 1:
            failures.append(
                "source ranks have inconsistent session state for remote weight transfer"
            )
        if not failures:
            for rank, item in enumerate(normalized):
                try:
                    self._validate_remote_transfer_local_bookkeeping(
                        item,
                        split_manifest=(
                            request.manifest_format == PLACEMENT_BINDING_V1
                        ),
                    )
                except Exception as error:
                    failures.append(f"source rank {rank}: {error}")

        lease_fences = {
            item.get("lease_fence") for item in normalized if item["success"]
        }
        generations = {item.get("generation") for item in normalized if item["success"]}
        if not failures and len(lease_fences) != 1:
            failures.append("source ranks returned inconsistent lease fences")
        if not failures and len(generations) != 1:
            failures.append("source ranks returned inconsistent snapshot generation")
        authoritative_lease_fence = (
            next(iter(lease_fences)) if len(lease_fences) == 1 else None
        )
        authoritative_generation = (
            next(iter(generations)) if len(generations) == 1 else None
        )
        manifests = None
        placements = None
        bindings = None
        session_state = next(iter(session_states)) if len(session_states) == 1 else None
        if not failures and session_state == "reused":
            try:
                self._validate_remote_transfer_reuse(
                    normalized,
                    cached_output,
                    world_size=world_size,
                    split_manifest=(request.manifest_format == PLACEMENT_BINDING_V1),
                )
            except Exception as error:
                failures.append(str(error))
        elif not failures and session_state == "created":
            try:
                if request.manifest_format == PLACEMENT_BINDING_V1:
                    placements = [item["placement"] for item in normalized]
                    bindings = [item["binding"] for item in normalized]
                    self._validate_remote_transfer_parts(
                        placements,
                        bindings,
                        world_size,
                    )
                else:
                    manifests = [item["manifest"] for item in normalized]
                    self._validate_remote_transfer_manifests(manifests, world_size)
            except Exception as error:
                failures.append(str(error))
        elif not failures:
            failures.append("source ranks returned an invalid session state")

        if failures:
            return (
                {
                    "success": False,
                    "message": " | ".join(failures),
                    "session_state": self._remote_transfer_failure_state(normalized),
                    "manifest_revision_semantics": consensus_semantics,
                },
                None,
            )

        assert consensus_semantics is not None
        if session_state == "reused":
            assert cached_output is not None
            output = BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=cached_output.transfer_id,
                success=True,
                message="Success.",
                session_state="reused",
                manifests=cached_output.manifests,
                placements=cached_output.placements,
                bindings=cached_output.bindings,
                manifest_revision_semantics=consensus_semantics,
                lease_fence=authoritative_lease_fence,
                generation=authoritative_generation,
                **response_identity,
            )
        else:
            output = BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=request.transfer_id,
                success=True,
                message="Success.",
                session_state="created",
                manifests=manifests,
                placements=placements,
                bindings=bindings,
                manifest_revision_semantics=consensus_semantics,
                lease_fence=authoritative_lease_fence,
                generation=authoritative_generation,
                **response_identity,
            )
        return (
            {
                "success": True,
                "message": "Success.",
                "session_state": session_state,
                "manifest_revision_semantics": consensus_semantics,
                "lease_fence": authoritative_lease_fence,
                "generation": authoritative_generation,
            },
            output,
        )

    @staticmethod
    def _validate_remote_transfer_local_bookkeeping(
        item: dict[str, Any],
        *,
        split_manifest: bool,
    ) -> None:
        lease_id = item.get("lease_id")
        generation = item.get("generation")
        if type(lease_id) is not str or not lease_id:
            raise ValueError("source snapshot lease ID is invalid")
        if type(generation) is not int or generation <= 0:
            raise ValueError("source snapshot generation is invalid")
        if item.get("session_state") == "reused":
            return
        record = item.get("binding" if split_manifest else "manifest")
        if not isinstance(record, dict):
            raise ValueError("source snapshot payload is invalid")
        if record.get("lease_id") != lease_id:
            raise ValueError("source snapshot lease bookkeeping does not match payload")
        if record.get("generation") != generation:
            raise ValueError(
                "source snapshot generation bookkeeping does not match payload"
            )

    @staticmethod
    def _validate_remote_transfer_reuse(
        gathered: list[dict[str, Any]],
        cached_output: BeginRemoteInstanceWeightTransferReqOutput | None,
        *,
        world_size: int,
        split_manifest: bool,
    ) -> None:
        if cached_output is None:
            raise RuntimeError("response root does not own the cached source manifest")
        records = cached_output.bindings if split_manifest else cached_output.manifests
        if records is None or len(records) != world_size:
            raise RuntimeError("cached source manifest does not match the model world")
        for rank, (item, record) in enumerate(zip(gathered, records)):
            if not isinstance(record, dict):
                raise RuntimeError(f"cached source rank {rank} record is invalid")
            if item.get("lease_id") != record.get("lease_id"):
                raise RuntimeError(
                    f"source rank {rank} cached lease does not match root manifest"
                )
            if item.get("generation") != record.get("generation"):
                raise RuntimeError(
                    f"source rank {rank} cached generation does not match root manifest"
                )
            if (
                cached_output.lease_fence is not None
                and item.get("lease_fence") != cached_output.lease_fence
            ):
                raise RuntimeError(
                    f"source rank {rank} cached lease fence does not match root"
                )

    @staticmethod
    def _remote_transfer_failure_state(gathered: list[dict[str, Any]]) -> str:
        all_states = {
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
        if "conflict" in all_states:
            return "conflict"
        if "expired" in all_states:
            return "expired"
        if all_states & {"created", "reused", "cleanup_pending"}:
            return "cleanup_pending"
        if "released" in all_states:
            return "released"
        if len(failure_states) == 1:
            return next(iter(failure_states))
        return "failed"

    @staticmethod
    def _remote_transfer_begin_vote_failure_state(
        votes: list[dict[str, Any]],
    ) -> str:
        failure_states = {
            vote["session_state"] for vote in votes if not vote["success"]
        }
        all_states = {vote["session_state"] for vote in votes}
        if "conflict" in failure_states:
            return "conflict"
        if "expired" in failure_states:
            return "expired"
        if all_states & {"reused", "cleanup_pending"}:
            return "cleanup_pending"
        if "released" in failure_states:
            return "released"
        if len(failure_states) == 1:
            return next(iter(failure_states))
        return "failed"

    @staticmethod
    def _validate_remote_transfer_decision(value: Any) -> dict[str, Any]:
        if (
            not isinstance(value, dict)
            or type(value.get("success")) is not bool
            or type(value.get("message")) is not str
            or type(value.get("session_state")) is not str
        ):
            raise RuntimeError("source manifest decision is invalid")
        semantics = value.get("manifest_revision_semantics")
        if value["success"] and (
            value["session_state"] not in {"created", "reused"}
            or type(semantics) is not str
        ):
            raise RuntimeError("source manifest success decision is invalid")
        if semantics is not None and type(semantics) is not str:
            raise RuntimeError("source manifest decision semantics are invalid")
        return value

    @staticmethod
    def _finalize_remote_transfer_failure(
        decision: dict[str, Any],
        cleanup_results: list,
    ) -> dict[str, Any]:
        failures = []
        for rank, item in enumerate(cleanup_results):
            if (
                not isinstance(item, dict)
                or type(item.get("success")) is not bool
                or type(item.get("message")) is not str
            ):
                failures.append(f"rank {rank}: invalid snapshot cleanup result")
            elif not item["success"]:
                failures.append(
                    f"rank {rank}: snapshot cleanup remains pending: {item['message']}"
                )
        final = dict(decision)
        if failures:
            final["message"] = f"{decision['message']} | {' | '.join(failures)}"
            if final["session_state"] not in {"conflict", "expired"}:
                final["session_state"] = "cleanup_pending"
        return final

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

    @staticmethod
    def _remote_transfer_record_identity(record: Any) -> tuple[str, str]:
        if not isinstance(record, dict):
            raise ValueError("source weight manifest record is invalid")
        model_id = record.get("model_id")
        revision = record.get("revision")
        if type(model_id) is not str or not model_id:
            raise ValueError("source weight manifest model ID is invalid")
        if type(revision) is not str or not revision:
            raise ValueError("source weight manifest revision is invalid")
        return model_id, revision

    def _gather_remote_weight_transfer_status(
        self,
        *,
        success: bool,
        message: str,
        operation: str,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Tuple[bool, str]:
        local_result = {"success": success, "message": message}
        collective_group = self._remote_weight_transfer_control_group()
        if execution_context is None:
            execution_context = WeightTransferExecutionContext(
                deadline_unix_sec=(
                    time.time() + _REMOTE_WEIGHT_TRANSFER_CONTROL_TIMEOUT_SEC
                ),
            )
        try:
            gathered = self._all_gather_remote_transfer_object(
                local_result,
                collective_group,
                phase=f"remote_instance.source.{operation}_gather",
                execution_context=execution_context,
            )
        except Exception as error:
            if getattr(error, "completion_unknown", None) is not False:
                self._poison_remote_weight_transfer_snapshot_lane(
                    f"{operation} result gather failed: {error}"
                )
            return False, f"Failed to gather source {operation} results: {error}"

        failures = [item["message"] for item in gathered if not item["success"]]
        if failures:
            return False, " | ".join(failures)
        return True, "Success."

    def _gather_remote_weight_transfer_renewal_status(
        self,
        *,
        success: bool,
        message: str,
        deadline_unix_sec: float | None,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> tuple[bool, str, float | None]:
        local_result = {
            "success": success,
            "message": message,
            "deadline_unix_sec": deadline_unix_sec,
        }
        collective_group = self._remote_weight_transfer_control_group()
        if execution_context is None:
            execution_context = WeightTransferExecutionContext(
                deadline_unix_sec=(
                    time.time() + _REMOTE_WEIGHT_TRANSFER_CONTROL_TIMEOUT_SEC
                ),
            )
        try:
            gathered = self._all_gather_remote_transfer_object(
                local_result,
                collective_group,
                phase="remote_instance.source.renewal_gather",
                execution_context=execution_context,
            )
        except Exception as error:
            if getattr(error, "completion_unknown", None) is not False:
                self._poison_remote_weight_transfer_snapshot_lane(
                    f"renewal result gather failed: {error}"
                )
            return False, f"Failed to gather source renewal results: {error}", None

        failures = []
        deadlines = []
        for rank, item in enumerate(gathered):
            if (
                not isinstance(item, dict)
                or type(item.get("success")) is not bool
                or type(item.get("message")) is not str
            ):
                failures.append(f"rank {rank}: invalid source renewal status")
                continue
            if not item["success"]:
                failures.append(f"rank {rank}: {item['message']}")
                continue
            deadline = item.get("deadline_unix_sec")
            if (
                isinstance(deadline, bool)
                or not isinstance(deadline, (int, float))
                or not math.isfinite(deadline)
                or deadline <= 0
            ):
                failures.append(f"rank {rank}: invalid source renewal lease deadline")
                continue
            deadlines.append(float(deadline))
        if failures:
            return False, " | ".join(failures), None
        return True, "Success.", min(deadlines)

    def _remote_weight_transfer_control_group(self) -> Any:
        return self._remote_weight_transfer_collective(
            (
                self.remote_weight_transfer_control_cpu_group
                or self.remote_weight_transfer_cpu_group
                or self.world_cpu_group
            ),
            control=True,
        )

    def _remote_weight_transfer_collective(
        self,
        group: Any,
        *,
        control: bool,
    ) -> Any:
        if callable(getattr(group, "gather_object", None)) or callable(
            getattr(group, "gather_object_to_root", None)
        ):
            return group
        if not isinstance(group, torch.distributed.ProcessGroup):
            return group
        attribute = (
            "remote_weight_transfer_control_coordinator"
            if control
            else "remote_weight_transfer_snapshot_coordinator"
        )
        coordinator = getattr(self, attribute)
        if coordinator is None:
            coordinator = TorchDistributedWeightStoreCoordinator(group=group)
            setattr(self, attribute, coordinator)
        return coordinator

    def get_remote_instance_weight_transfer_session(
        self, recv_req: GetRemoteInstanceWeightTransferSessionReqInput
    ) -> GetRemoteInstanceWeightTransferSessionReqOutput:
        self._prune_remote_weight_transfer_bookkeeping()
        execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=(
                recv_req.deadline_unix_sec
                if recv_req.deadline_unix_sec is not None
                else time.time() + _REMOTE_WEIGHT_TRANSFER_CONTROL_TIMEOUT_SEC
            )
        )
        identity_error = self._remote_weight_transfer_control_identity_error(
            transfer_id=recv_req.transfer_id,
            lease_fence=recv_req.lease_fence,
            generation=recv_req.generation,
        )
        with self.remote_weight_transfer_lock:
            active = recv_req.transfer_id in self.remote_weight_transfer_leases
            tombstoned = recv_req.transfer_id in self.remote_weight_transfer_tombstones
            expired = recv_req.transfer_id in self.remote_weight_transfer_expired
            if active:
                lease_fence = self.remote_weight_transfer_fences.get(
                    recv_req.transfer_id
                )
                generation = self.remote_weight_transfer_generations.get(
                    recv_req.transfer_id
                )
                deadline_monotonic_sec = self.remote_weight_transfer_deadlines.get(
                    recv_req.transfer_id
                )
            else:
                lease_fence = self.remote_weight_transfer_tombstone_fences.get(
                    recv_req.transfer_id
                )
                generation = self.remote_weight_transfer_tombstone_generations.get(
                    recv_req.transfer_id
                )
                deadline_monotonic_sec = None

        if execution_context.expired():
            success = False
            message = "Remote weight transfer status deadline expired."
            session_state = "failed"
        elif identity_error is not None:
            success = False
            message = identity_error
            session_state = "conflict"
        elif active:
            success = True
            message = "Success."
            session_state = "expired" if expired else "active"
        elif tombstoned:
            success = True
            message = "Remote weight transfer was already released."
            session_state = "released"
        else:
            success = False
            message = "Remote weight transfer does not exist."
            session_state = "unknown"

        deadline_unix_sec = None
        if deadline_monotonic_sec is not None:
            deadline_unix_sec = time.time() + max(
                0.0,
                deadline_monotonic_sec - time.monotonic(),
            )
        return GetRemoteInstanceWeightTransferSessionReqOutput(
            transfer_id=recv_req.transfer_id,
            success=success,
            message=message,
            session_state=session_state,
            lease_fence=lease_fence,
            generation=generation,
            deadline_unix_sec=deadline_unix_sec,
            request_id=recv_req.request_id,
            external_dp_rank=self._external_dp_rank(),
        )

    def renew_remote_instance_weight_transfer(
        self, recv_req: RenewRemoteInstanceWeightTransferReqInput
    ) -> RenewRemoteInstanceWeightTransferReqOutput:
        self._prune_remote_weight_transfer_bookkeeping()
        execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=(
                recv_req.deadline_unix_sec
                if recv_req.deadline_unix_sec is not None
                else time.time() + _REMOTE_WEIGHT_TRANSFER_CONTROL_TIMEOUT_SEC
            )
        )
        begin_inflight = self._remote_weight_transfer_begin_is_inflight(
            recv_req.transfer_id
        )
        with self.remote_weight_transfer_lock:
            expired = recv_req.transfer_id in self.remote_weight_transfer_expired
        lease_id = self._get_remote_weight_transfer_lease(recv_req.transfer_id)
        identity_error = self._remote_weight_transfer_control_identity_error(
            transfer_id=recv_req.transfer_id,
            lease_fence=recv_req.lease_fence,
            generation=recv_req.generation,
        )
        if execution_context.expired():
            local_success = False
            local_message = "Remote weight transfer renewal deadline expired."
            local_deadline_unix_sec = None
        elif identity_error is not None:
            local_success = False
            local_message = identity_error
            local_deadline_unix_sec = None
        elif begin_inflight:
            local_success = False
            local_message = "Remote weight transfer begin is still in progress."
            local_deadline_unix_sec = None
        elif expired:
            local_success = False
            local_message = (
                "Remote weight transfer expired and requires explicit release."
            )
        elif lease_id is None:
            local_success = False
            local_message = "Remote weight transfer does not exist or has expired."
            local_deadline_unix_sec = None
        else:
            try:
                if execution_context.expired():
                    raise TimeoutError(
                        "Remote weight transfer renewal deadline expired."
                    )
                local_deadline_unix_sec = time.time() + recv_req.lease_timeout_sec
                self.tp_worker.model_runner.renew_weight_runtime_manifest(
                    lease_id,
                    lease_timeout_sec=recv_req.lease_timeout_sec,
                )
                self._record_remote_weight_transfer_lease(
                    recv_req.transfer_id,
                    lease_id,
                    recv_req.lease_timeout_sec,
                    generation=recv_req.generation,
                    lease_fence=recv_req.lease_fence,
                )
                local_success = True
                local_message = "Success."
            except Exception as error:
                local_success = False
                local_message = str(error)
                local_deadline_unix_sec = None

        if expired:
            local_deadline_unix_sec = None
        success, message, deadline_unix_sec = (
            self._gather_remote_weight_transfer_renewal_status(
                success=local_success,
                message=local_message,
                deadline_unix_sec=local_deadline_unix_sec,
                execution_context=execution_context,
            )
        )
        return RenewRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=success,
            message=message,
            request_id=recv_req.request_id,
            external_dp_rank=self._external_dp_rank(),
            deadline_unix_sec=deadline_unix_sec,
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
        execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=(
                recv_req.deadline_unix_sec
                if recv_req.deadline_unix_sec is not None
                else time.time() + _REMOTE_WEIGHT_TRANSFER_CONTROL_TIMEOUT_SEC
            )
        )
        begin_inflight = self._remote_weight_transfer_begin_is_inflight(
            recv_req.transfer_id
        )
        lease_id = self._get_remote_weight_transfer_lease(recv_req.transfer_id)
        with self.remote_weight_transfer_lock:
            generation = self.remote_weight_transfer_generations.get(
                recv_req.transfer_id
            )
            tombstoned = recv_req.transfer_id in self.remote_weight_transfer_tombstones
        identity_error = self._remote_weight_transfer_control_identity_error(
            transfer_id=recv_req.transfer_id,
            lease_fence=recv_req.lease_fence,
            generation=recv_req.generation,
            allow_begin_fence=True,
        )
        if execution_context.expired():
            local_success = False
            local_message = "Remote weight transfer release deadline expired."
        elif identity_error is not None:
            local_success = False
            local_message = identity_error
        elif begin_inflight:
            local_success = False
            local_message = "Remote weight transfer begin is still in progress."
        elif lease_id is None and tombstoned:
            local_success = True
            local_message = "Remote weight transfer was already released."
        elif lease_id is None:
            local_success = False
            local_message = "Remote weight transfer does not exist."
        else:
            try:
                if execution_context.expired():
                    raise TimeoutError(
                        "Remote weight transfer release deadline expired."
                    )
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
            execution_context=execution_context,
        )
        return ReleaseRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=success,
            message=message,
            request_id=recv_req.request_id,
            external_dp_rank=self._external_dp_rank(),
        )

    def update_weight_snapshot_activation(
        self,
        recv_req: WeightSnapshotActivationReqInput,
    ) -> WeightSnapshotActivationReqOutput:
        try:
            action = recv_req.action
            structured = (
                recv_req.request_id is not None
                or recv_req.transaction_id is not None
                or recv_req.deadline_unix_sec is not None
                or recv_req.phase != "commit"
            )
            if structured and recv_req.request_id is None:
                raise RuntimeError(
                    "structured weight snapshot activation request_id is missing"
                )
            callback_name = (
                "activate_pending_weight_snapshot"
                if structured
                else {
                    "activate": "activate_pending_weight_snapshot",
                    "close": "close_pending_weight_snapshot_activation",
                }[action]
            )
            callback = getattr(self.tp_worker.model_runner, callback_name, None)
            if not callable(callback):
                raise RuntimeError(
                    f"weight snapshot activation action is unavailable: {action}"
                )
            callback()
            return WeightSnapshotActivationReqOutput(
                action=action,
                success=True,
                message="Success.",
            )
        except Exception as error:
            return WeightSnapshotActivationReqOutput(
                action=recv_req.action,
                success=False,
                message=str(error),
            )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert self.is_fully_idle(), (
            "release_memory_occupation should be called only when server is idle."
        )

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
            assert draft_url is not None, (
                "draft_url must be provided when draft model is enabled"
            )
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
