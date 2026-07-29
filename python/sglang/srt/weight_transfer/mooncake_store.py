from __future__ import annotations

import base64
import hashlib
import importlib
import inspect
import json
import threading
import time
import zlib
from collections import Counter
from dataclasses import dataclass, field, is_dataclass, replace
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from sglang.srt.weight_transfer._threaded_call import _ThreadedCall
from sglang.srt.weight_transfer.binding import bind_weight_source
from sglang.srt.weight_transfer.contracts import (
    RuntimeWeightLocation,
    StorageWeightLocation,
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
)
from sglang.srt.weight_transfer.distributed import (
    LocalWeightStoreDistributedCoordinator,
    WeightStoreDistributedCoordinator,
    WeightStoreDistributedError,
    WeightStorePreflightOutcome,
    WeightStoreUploadOutcome,
)
from sglang.srt.weight_transfer.provider import (
    WeightLoadReceipt,
    WeightLoadRequest,
    WeightMaterializeReceipt,
    WeightMaterializeRequest,
    WeightProviderCapabilities,
    WeightProviderReceipt,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
    WeightTransferExecutionContext,
    WeightTransferReleaseError,
)
from sglang.srt.weight_transfer.storage import (
    StoredWeightSnapshot,
    WeightStorageRef,
    weight_placement_set_digest,
    weight_source_snapshot_digest,
)

_RECOVERY_TICKET_PREFIX = "sglang-mooncake-weight-upload-v1:"
_RECOVERY_RECORD_FORMAT = "sglang-mooncake-weight-upload-recovery"
_RECOVERY_JOURNAL_FORMAT = "sglang-mooncake-weight-upload-journal"
_MAX_RECOVERY_TICKET_BYTES = 64 * 1024 * 1024
_MAX_RECOVERY_RECORD_BYTES = 256 * 1024 * 1024
_RECOVERY_JOURNAL_CHUNK_OPERATIONS = 256
_MAX_RECOVERY_JOURNAL_CHUNK_BYTES = 1024 * 1024
_RECOVERY_MANIFEST_CHUNK_CHARS = _MAX_RECOVERY_JOURNAL_CHUNK_BYTES // 4
_MOONCAKE_OBJECT_NOT_FOUND = -704
_STORE_UPLOAD_DESCRIPTOR_VERSION = 1
_STORE_COMMIT_DESCRIPTOR_VERSION = 1
_MAX_STORE_COMPACT_DESCRIPTOR_BYTES = 64 * 1024
_MAX_DISTRIBUTED_UPLOAD_RECORDS = 10_000_000
_STORE_ROOT_RANK = 0
_STORE_TERMINAL_CONTROL_TIMEOUT_SEC = 5.0


def _store_terminal_execution_context(
    execution_context: WeightTransferExecutionContext | None,
    *,
    include_business_window: bool = False,
) -> WeightTransferExecutionContext | None:
    if execution_context is None:
        return None
    deadline_unix_sec = time.time() + _STORE_TERMINAL_CONTROL_TIMEOUT_SEC
    if include_business_window:
        deadline_unix_sec = (
            execution_context.deadline_unix_sec + _STORE_TERMINAL_CONTROL_TIMEOUT_SEC
        )
    return WeightTransferExecutionContext(
        deadline_unix_sec=deadline_unix_sec,
    )


@dataclass(frozen=True)
class _PreparedStoreLoad:
    request: WeightLoadRequest
    load_plan: Any
    target_manifest: Any


@dataclass(frozen=True)
class _PreparedStoreMaterialize:
    request: WeightMaterializeRequest
    upload_plan: Any
    runtime_manifests: tuple[tuple[str, Any], ...]
    recovery_ticket: str | None = None
    upload_descriptor: _StoreUploadDescriptor | None = None


@dataclass(frozen=True)
class _RankUploadPreparation:
    rank: int
    runtime_manifests: tuple[tuple[str, Any], ...]
    local_placement_ids: tuple[str, ...]


@dataclass(frozen=True)
class _StoreUploadDescriptor:
    version: int
    operation_id: str
    model_id: str
    revision: str
    storage_id: str
    manifest_key: str
    manifest_digest: str
    payload_digest: str
    total_bytes: int
    fragment_count: int
    operation_count: int
    recovery_ticket: str


@dataclass(frozen=True)
class _RankPreparedUpload:
    upload_plan: Any | None
    upload_descriptor: _StoreUploadDescriptor | None
    recovery_ticket: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class _StoreCommitDescriptor:
    version: int
    operation_id: str
    model_id: str
    revision: str
    storage_id: str
    manifest_key: str
    manifest_digest: str
    snapshot_digest: str
    payload_digest: str
    total_bytes: int
    fragment_count: int
    operation_count: int
    recovery_ticket: str


@dataclass(frozen=True)
class _StoreCommitNotStarted:
    detail: str


@dataclass(frozen=True)
class _StoreCommitUncertain:
    detail: str
    observe_manifest: bool


@dataclass(frozen=True)
class _RankRecoveryProjection:
    rank: int
    operation_id: str
    placement_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RankRecoveryResult:
    receipt: WeightMaterializeReceipt
    terminal_ref: WeightStorageRef


@dataclass(frozen=True)
class _ChecksummedUploadReceipt:
    fragment_id: str
    object_key: str
    worker_id: str
    checksum: str


class _StoreTerminalState(str, Enum):
    MANIFEST_MATCH = "manifest_match"
    MANIFEST_ABSENT = "manifest_absent"
    MANIFEST_CONFLICT = "manifest_conflict"
    PAYLOAD_COMPLETE = "payload_complete"
    PAYLOAD_INCOMPLETE = "payload_incomplete"
    OBSERVATION_FAILED = "observation_failed"


@dataclass(frozen=True)
class _StoreTerminalDecision:
    state: _StoreTerminalState
    manifest: Any | None = None
    payload_keys: tuple[str, ...] = ()
    detail: str | None = None


@dataclass(frozen=True)
class _StoreRecoveryResolution:
    prepared: _PreparedStoreMaterialize
    receipts: tuple[Any, ...]
    manifest: Any | None = None


@dataclass
class _StoreSubmission:
    prepared: _PreparedStoreLoad | _PreparedStoreMaterialize
    receipts: list[Any] = field(default_factory=list)
    committed: bool = False
    aborted: bool = False
    local_upload_call: _BoundedStoreCall | None = None
    local_commit_call: _BoundedStoreCall | None = None


class _StoreCallInterrupted(RuntimeError):
    def __init__(self, phase: str, *, started: bool) -> None:
        super().__init__(f"{phase} did not finish before the transfer deadline")
        self.phase = phase
        self.started = started
        self.completion_unknown = started


class _RecoveryJournalReadError(RuntimeError):
    completion_unknown = True


class WeightStoreNativeCallState(str, Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class WeightStoreNativeCallStatus:
    operation_id: str
    phase: str
    state: WeightStoreNativeCallState
    error: str | None = None


@dataclass
class _BoundedStoreCall:
    operation_id: str
    phase: str
    owner: Any = field(repr=False)
    _call: _ThreadedCall = field(default_factory=_ThreadedCall, repr=False)

    @property
    def done(self) -> threading.Event:
        return self._call.done

    @property
    def error(self) -> BaseException | None:
        return self._call.error

    @property
    def thread(self) -> threading.Thread | None:
        return self._call.thread

    def start(self, factory: Callable[[], Any]) -> None:
        self._call.start(
            factory,
            thread_name=f"sglang-weight-store-{self.phase}",
        )

    def result_before(
        self,
        execution_context: WeightTransferExecutionContext,
    ) -> Any:
        return self._call.result_before(
            execution_context,
            interrupted=lambda: _StoreCallInterrupted(
                self.phase,
                started=True,
            ),
        )


class MooncakeWeightStoreProvider:
    """Optional adapter from SGLang plans to Mooncake WeightStore."""

    name = "mooncake-store"
    requires_runtime_attestation = True

    def __init__(
        self,
        weight_store: Any,
        *,
        namespace: str = "default",
        local_placement_ids: Sequence[str] | None = None,
        receipt_exchange: Callable[[Any, tuple[Any, ...]], Sequence[Any]] | None = None,
        coordinator: WeightStoreDistributedCoordinator | None = None,
        payload_checksum_verifier: Callable[[RuntimeWeightLocation], str] | None = None,
        source_pre_registered: bool = False,
        target_pre_registered: bool = False,
        prepare_upload_is_local: bool = False,
        max_total_operations: int = 10_000_000,
    ) -> None:
        if not namespace:
            raise ValueError("Mooncake weight namespace must not be empty")
        if type(max_total_operations) is not int or max_total_operations <= 0:
            raise ValueError("max_total_operations must be a positive integer")
        if type(prepare_upload_is_local) is not bool:
            raise ValueError("prepare_upload_is_local must be a boolean")
        self.weight_store = weight_store
        self.namespace = namespace
        self.local_placement_ids = (
            None if local_placement_ids is None else frozenset(local_placement_ids)
        )
        if receipt_exchange is not None and coordinator is not None:
            raise ValueError(
                "receipt_exchange and distributed coordinator are mutually exclusive"
            )
        if payload_checksum_verifier is not None and not callable(
            payload_checksum_verifier
        ):
            raise ValueError("payload checksum verifier must be callable")
        self.receipt_exchange = receipt_exchange
        self.coordinator = (
            None
            if receipt_exchange is not None
            else coordinator or LocalWeightStoreDistributedCoordinator()
        )
        if (
            self.coordinator is not None
            and self.coordinator.world_size > 1
            and self.local_placement_ids is None
        ):
            raise ValueError("multi-rank Mooncake uploads require local placement IDs")
        self.payload_checksum_verifier = payload_checksum_verifier
        self.source_pre_registered = source_pre_registered
        self.target_pre_registered = target_pre_registered
        self.prepare_upload_is_local = prepare_upload_is_local
        self.max_total_operations = max_total_operations
        self._pending_materializations: dict[str, _StoreSubmission] = {}
        self._materialization_terminal_refs: dict[str, WeightStorageRef] = {}
        self._finalize_calls: dict[str, _BoundedStoreCall] = {}
        self._native_calls: dict[tuple[str, str], _BoundedStoreCall] = {}
        self._native_calls_lock = threading.Lock()
        self._native_calls_sealed = False
        self._execution_context: WeightTransferExecutionContext | None = None
        self._deferred_recovery_cleanup_lock = threading.Lock()
        self._deferred_recovery_cleanups: set[str] = set()
        self._deferred_recovery_cleanup_events: dict[str, threading.Event] = {}

    def current_execution_context(
        self,
    ) -> WeightTransferExecutionContext | None:
        return self._execution_context

    def _remember_execution_context(
        self,
        execution_context: WeightTransferExecutionContext | None,
    ) -> None:
        if execution_context is not None:
            self._execution_context = execution_context

    def _coordinate(
        self,
        method_name: str,
        *args: Any,
        execution_context: WeightTransferExecutionContext | None,
    ) -> Any:
        if self.coordinator is None:
            raise RuntimeError("distributed coordinator is unavailable")
        method = getattr(self.coordinator, method_name)
        if execution_context is None:
            return method(*args)
        return method(*args, execution_context=execution_context)

    @staticmethod
    def _load_backend() -> Any:
        try:
            backend = importlib.import_module("mooncake.weight_transfer")
        except Exception as error:
            raise WeightTransferError(
                "Mooncake WeightStore support is unavailable",
                code="UNAVAILABLE_PROVIDER",
                provider="mooncake-store",
                phase="probe",
                operation_id="unbound",
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            ) from error
        required = (
            "LogicalTransferPlan",
            "PipelineRouteGroup",
            "RuntimeBindingManifest",
            "SourcePlacementManifest",
            "StoredFragment",
            "TargetPlacementManifest",
            "TransferRegion",
            "WeightLoadPlan",
            "WeightStoreError",
            "bind_logical_transfer_plan",
            "bind_runtime_manifest",
        )
        missing = [name for name in required if not hasattr(backend, name)]
        if missing:
            raise WeightTransferError(
                "Mooncake WeightStore provider is missing APIs: " + ", ".join(missing),
                code="UNAVAILABLE_PROVIDER",
                provider="mooncake-store",
                phase="probe",
                operation_id="unbound",
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            )
        return backend

    def probe(
        self,
        request: WeightLoadRequest | WeightMaterializeRequest,
    ) -> WeightProviderCapabilities:
        self._load_backend()
        max_segments = getattr(
            self.weight_store,
            "max_region_segments",
            1_000_000,
        )
        return WeightProviderCapabilities(
            provider=self.name,
            load_profiles=frozenset({"storage_to_runtime"}),
            materialize_profiles=frozenset({"runtime_to_storage"}),
            supports_nd_regions=True,
            supports_strided_regions=True,
            supports_safe_cancel=False,
            supports_completion_ticket=True,
            supports_transactional_publish=True,
            supports_bounded_execution=(
                isinstance(request, WeightLoadRequest)
                or (
                    isinstance(request, WeightMaterializeRequest)
                    and self.prepare_upload_is_local
                )
            ),
            max_regions=1_000_000,
            max_segments_per_region=max_segments,
            max_total_operations=self.max_total_operations,
            max_batch_operations=getattr(
                self.weight_store,
                "max_ranges_per_request",
                1024,
            ),
        )

    def _native_call_status(
        self,
        call: _BoundedStoreCall,
    ) -> WeightStoreNativeCallStatus:
        if not call.done.is_set():
            state = WeightStoreNativeCallState.PENDING
            detail = None
        elif call.error is None:
            state = WeightStoreNativeCallState.SUCCEEDED
            detail = None
        else:
            state = WeightStoreNativeCallState.FAILED
            detail = self._error_detail(call.error)
        return WeightStoreNativeCallStatus(
            operation_id=call.operation_id,
            phase=call.phase,
            state=state,
            error=detail,
        )

    def pending_native_calls(self) -> tuple[WeightStoreNativeCallStatus, ...]:
        with self._native_calls_lock:
            calls = tuple(self._native_calls.values())
        statuses = list(
            self._native_call_status(call) for call in calls if not call.done.is_set()
        )
        with self._deferred_recovery_cleanup_lock:
            statuses.extend(
                WeightStoreNativeCallStatus(
                    operation_id=operation_id,
                    phase="deferred-recovery-cleanup",
                    state=WeightStoreNativeCallState.PENDING,
                )
                for operation_id in self._deferred_recovery_cleanups
            )
        return tuple(statuses)

    def drain_pending_calls(
        self,
        *,
        timeout_ms: int,
    ) -> tuple[WeightStoreNativeCallStatus, ...]:
        if type(timeout_ms) is not int or timeout_ms < 0:
            raise ValueError("Store drain timeout_ms must be a non-negative integer")
        deadline = time.monotonic() + timeout_ms / 1000
        with self._native_calls_lock:
            calls = tuple(self._native_calls.items())
        for _, call in calls:
            remaining = max(0.0, deadline - time.monotonic())
            call.done.wait(timeout=remaining)
        with self._deferred_recovery_cleanup_lock:
            deferred_events = tuple(self._deferred_recovery_cleanup_events.values())
        for event in deferred_events:
            remaining = max(0.0, deadline - time.monotonic())
            event.wait(timeout=remaining)

        statuses = tuple(self._native_call_status(call) for _, call in calls)
        with self._native_calls_lock:
            for key, call in calls:
                if call.done.is_set() and self._native_calls.get(key) is call:
                    self._native_calls.pop(key, None)
                    call.owner = None
        with self._deferred_recovery_cleanup_lock:
            deferred_statuses = tuple(
                WeightStoreNativeCallStatus(
                    operation_id=operation_id,
                    phase="deferred-recovery-cleanup",
                    state=WeightStoreNativeCallState.PENDING,
                )
                for operation_id in self._deferred_recovery_cleanups
            )
        return (*statuses, *deferred_statuses)

    def _defer_recovery_journal_cleanup(
        self,
        operation_id: str,
        keys: Sequence[str],
    ) -> None:
        keys = tuple(dict.fromkeys(keys))
        if not keys:
            return
        with self._deferred_recovery_cleanup_lock:
            if operation_id in self._deferred_recovery_cleanups:
                return
            self._deferred_recovery_cleanups.add(operation_id)
            cleanup_done = threading.Event()
            self._deferred_recovery_cleanup_events[operation_id] = cleanup_done

        def cleanup() -> None:
            while True:
                with self._native_calls_lock:
                    pending = tuple(
                        call
                        for call in self._native_calls.values()
                        if call.operation_id == operation_id and not call.done.is_set()
                    )
                    if not pending:
                        completed = tuple(
                            (key, call)
                            for key, call in self._native_calls.items()
                            if call.operation_id == operation_id
                        )
                        for key, call in completed:
                            if self._native_calls.get(key) is call:
                                self._native_calls.pop(key, None)
                                call.owner = None
                if not pending:
                    try:
                        for index, key in enumerate(keys):
                            self._delete_recovery_journal_chunk(
                                operation_id,
                                key,
                                phase=f"journal.deferred-rollback.{index}",
                                execution_context=None,
                            )
                    except BaseException:
                        time.sleep(0.5)
                        continue
                    with self._deferred_recovery_cleanup_lock:
                        self._deferred_recovery_cleanups.discard(operation_id)
                        self._deferred_recovery_cleanup_events.pop(operation_id, None)
                    cleanup_done.set()
                    return
                for call in pending:
                    call.done.wait(timeout=0.5)

        threading.Thread(
            target=cleanup,
            name="sglang-weight-store-recovery-cleanup",
            daemon=True,
        ).start()

    def seal_native_calls_for_close(
        self,
    ) -> tuple[WeightStoreNativeCallStatus, ...]:
        with self._native_calls_lock:
            self._native_calls_sealed = True
            pending = tuple(
                call for call in self._native_calls.values() if not call.done.is_set()
            )
        statuses = [self._native_call_status(call) for call in pending]
        with self._deferred_recovery_cleanup_lock:
            statuses.extend(
                WeightStoreNativeCallStatus(
                    operation_id=operation_id,
                    phase="deferred-recovery-cleanup",
                    state=WeightStoreNativeCallState.PENDING,
                )
                for operation_id in self._deferred_recovery_cleanups
            )
        return tuple(statuses)

    def _get_or_start_native_call(
        self,
        operation_id: str,
        phase: str,
        factory: Callable[[], Any],
    ) -> _BoundedStoreCall:
        key = (operation_id, phase)
        with self._native_calls_lock:
            with self._deferred_recovery_cleanup_lock:
                terminal_cleanup = phase.startswith(
                    (
                        "preflight.abort",
                        "preflight.finalize",
                        "preflight.discard_recovery",
                        "journal.delete",
                        "journal.rollback",
                    )
                )
                if (
                    operation_id in self._deferred_recovery_cleanups
                    and not terminal_cleanup
                ):
                    raise RuntimeError(
                        "Mooncake Store recovery cleanup is pending for operation"
                    )
                if self._native_calls_sealed:
                    raise RuntimeError(
                        "Mooncake Store provider is closed to native calls"
                    )
                call = self._native_calls.get(key)
                if call is None:
                    call = _BoundedStoreCall(
                        operation_id=operation_id,
                        phase=phase,
                        owner=self,
                    )
                    self._native_calls[key] = call
                    start = True
                else:
                    start = False
        if start:
            call.start(factory)
        return call

    def _await_native_call(
        self,
        call: _BoundedStoreCall,
        execution_context: WeightTransferExecutionContext,
    ) -> Any:
        try:
            result = call.result_before(execution_context)
        except _StoreCallInterrupted:
            raise
        except BaseException:
            self._forget_native_call(call)
            raise
        self._forget_native_call(call)
        return result

    def _run_native_call(
        self,
        operation_id: str,
        phase: str,
        factory: Callable[[], Any],
        execution_context: WeightTransferExecutionContext | None,
    ) -> Any:
        if execution_context is None:
            return factory()
        if execution_context.expired():
            raise _StoreCallInterrupted(phase, started=False)
        call = self._get_or_start_native_call(
            operation_id,
            phase,
            factory,
        )
        return self._await_native_call(call, execution_context)

    def _forget_native_call(self, call: _BoundedStoreCall) -> None:
        key = (call.operation_id, call.phase)
        with self._native_calls_lock:
            if self._native_calls.get(key) is call:
                self._native_calls.pop(key, None)
                call.owner = None

    @staticmethod
    def _collect_descriptors(placements: Sequence[Any]) -> tuple[Any, ...]:
        descriptors = {}
        for placement in placements:
            for descriptor in placement.tensors:
                previous = descriptors.setdefault(
                    descriptor.tensor_id,
                    descriptor,
                )
                if previous != descriptor:
                    raise ValueError(
                        "Mooncake placement descriptor mismatch: "
                        f"{descriptor.tensor_id}"
                    )
        return tuple(descriptors[tensor_id] for tensor_id in sorted(descriptors))

    @staticmethod
    def _replace_record(value: Any, **changes: Any) -> Any:
        if is_dataclass(value):
            return replace(value, **changes)
        values = dict(vars(value))
        values.update(changes)
        return type(value)(**values)

    @staticmethod
    def _route_groups(backend: Any, request: WeightLoadRequest) -> tuple[Any, ...]:
        route_indices: dict[tuple[int, int], list[int]] = {}
        for index, region in enumerate(request.plan.regions):
            route_indices.setdefault(
                (region.source.rank.pp, region.target.rank.pp),
                [],
            ).append(index)
        return tuple(
            backend.PipelineRouteGroup(
                source_pp=source_pp,
                target_pp=target_pp,
                operation_indices=tuple(indices),
            )
            for (source_pp, target_pp), indices in sorted(route_indices.items())
        )

    @staticmethod
    def _runtime_manifests(
        backend: Any,
        placements: Sequence[WeightPlacementManifest],
        bindings: Sequence[WeightRuntimeBindingManifest],
        placement_type: Any,
    ) -> tuple[tuple[str, Any], ...]:
        binding_by_id = {binding.placement_id: binding for binding in bindings}
        result = []
        for placement in sorted(
            placements,
            key=lambda item: item.placement_id,
        ):
            binding = binding_by_id.get(placement.placement_id)
            if binding is None:
                raise ValueError("Mooncake runtime placement has no binding")
            backend_placement = placement_type.from_runtime_inventory(placement)
            backend_binding = backend.RuntimeBindingManifest.from_runtime_inventory(
                binding
            )
            result.append(
                (
                    placement.placement_id,
                    backend.bind_runtime_manifest(
                        backend_placement,
                        backend_binding,
                    ),
                )
            )
        return tuple(result)

    @staticmethod
    def _stored_fragment_signature(fragment: Any) -> tuple[Any, ...]:
        try:
            return (
                fragment.fragment_id,
                fragment.tensor_id,
                tuple(fragment.global_offset),
                tuple(fragment.local_shape),
                fragment.object_key,
                fragment.object_offset,
                fragment.nbytes,
                fragment.checksum,
            )
        except (AttributeError, TypeError) as error:
            raise ValueError("Mooncake stored fragment is incomplete") from error

    @staticmethod
    def _canonical_json(value: Any) -> bytes:
        return json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @staticmethod
    def _text_sha256(value: str) -> str:
        digest = hashlib.sha256()
        for start in range(
            0,
            len(value),
            _RECOVERY_MANIFEST_CHUNK_CHARS,
        ):
            digest.update(
                value[start : start + _RECOVERY_MANIFEST_CHUNK_CHARS].encode("utf-8")
            )
        return digest.hexdigest()

    def _validate_operation_count(
        self,
        request: WeightMaterializeRequest,
        operation_count: int,
        *,
        phase: str,
    ) -> None:
        if type(operation_count) is not int or operation_count < 0:
            raise ValueError("Mooncake operation count is invalid")
        if operation_count > self.max_total_operations:
            raise WeightTransferError(
                "Mooncake WeightStore materialization exceeds the operation limit",
                code="UNSUPPORTED_CAPABILITY",
                provider=self.name,
                phase=phase,
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            )

    def _recovery_request_fields(
        self,
        request: WeightMaterializeRequest,
    ) -> dict[str, Any]:
        if request.payload_identity is None:
            raise ValueError("Mooncake recovery requires payload identity")
        return {
            "operation_id": request.operation_id,
            "placement_digest": weight_placement_set_digest(request.source_placements),
            "source_snapshot_digest": weight_source_snapshot_digest(
                request.source_placements,
                request.source_bindings,
            ),
            "payload_digest": request.payload_identity.payload_digest,
            "destination": {
                "provider": request.destination.provider,
                "storage_id": request.destination.storage_id,
                "object_prefix": request.destination.object_prefix,
            },
        }

    def _recovery_request_digest(
        self,
        request: WeightMaterializeRequest,
    ) -> str:
        payload = self._canonical_json(self._recovery_request_fields(request))
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    @staticmethod
    def _recovery_generation(request: WeightMaterializeRequest) -> int:
        generations = tuple(
            getattr(binding, "generation", None) for binding in request.source_bindings
        )
        if not generations or any(
            type(generation) is not int or generation <= 0 for generation in generations
        ):
            raise ValueError("Mooncake recovery source generation is invalid")
        return max(generations)

    @staticmethod
    def _recovery_operation_record(operation: Any) -> dict[str, Any]:
        source = operation.source
        rank = source.rank
        return {
            "source": {
                "fragment_id": source.fragment_id,
                "tensor_id": source.tensor_id,
                "global_offset": list(source.global_offset),
                "local_shape": list(source.local_shape),
                "nbytes": source.nbytes,
                "worker_id": source.worker_id,
                "endpoint": source.endpoint,
                "rank": [rank.dp, rank.tp, rank.pp, rank.ep],
                "lease_generation": source.lease_generation,
                "aliases": list(source.aliases),
                "placement_fragment_id": source.placement_fragment_id,
            },
            "target_fragment_id": operation.target.fragment_id,
            "source_runtime_lease_id": getattr(
                operation,
                "source_runtime_lease_id",
                None,
            ),
        }

    @staticmethod
    def _recovery_receipt_record(receipt: Any) -> dict[str, Any]:
        return {
            "fragment_id": receipt.fragment_id,
            "object_key": receipt.object_key,
            "worker_id": receipt.worker_id,
            "checksum": receipt.checksum,
        }

    @staticmethod
    def _journal_chunk_key(
        journal_key: str,
        kind: str,
        index: int | None = None,
    ) -> str:
        if index is None:
            return f"{journal_key}/{kind}"
        return f"{journal_key}/{kind}/{index:08d}"

    def _journal_store_method(
        self,
        direct_name: str,
        store_names: Sequence[str],
    ) -> Callable[..., Any]:
        direct = getattr(self.weight_store, direct_name, None)
        if callable(direct):
            return direct
        store = getattr(self.weight_store, "store", None)
        for name in store_names:
            method = getattr(store, name, None)
            if callable(method):
                return method
        raise ValueError("Mooncake Store does not expose recovery journal I/O")

    def _put_recovery_journal_chunk(
        self,
        operation_id: str,
        key: str,
        payload: bytes,
        *,
        phase: str,
        execution_context: WeightTransferExecutionContext | None,
    ) -> None:
        if not payload or len(payload) > _MAX_RECOVERY_JOURNAL_CHUNK_BYTES:
            raise ValueError("Mooncake recovery journal chunk exceeds the size limit")
        put = self._journal_store_method(
            "put_recovery_journal_chunk",
            ("put",),
        )

        def write() -> None:
            try:
                parameters = inspect.signature(put).parameters
            except (TypeError, ValueError):
                parameters = {}
            if "config" in parameters:
                config_factory = getattr(
                    self.weight_store,
                    "config_factory",
                    getattr(self.weight_store, "_config_factory", None),
                )
                if not callable(config_factory):
                    raise ValueError(
                        "Mooncake Store recovery journal config is unavailable"
                    )
                config = config_factory([key], "metadata")
                result = put(key, payload, config)
            else:
                result = put(key, payload)
            if result is False or (type(result) is int and result not in (0, 1)):
                raise ValueError(f"Mooncake recovery journal write failed: {key}")

        self._run_native_call(
            operation_id,
            phase,
            write,
            execution_context,
        )

    def _get_recovery_journal_chunk(
        self,
        operation_id: str,
        key: str,
        *,
        phase: str,
        execution_context: WeightTransferExecutionContext | None,
    ) -> bytes | None:
        get = self._journal_store_method(
            "get_recovery_journal_chunk",
            ("get",),
        )

        def read() -> Any:
            try:
                return get(key)
            except KeyError:
                return None

        result = self._run_native_call(
            operation_id,
            phase,
            read,
            execution_context,
        )
        if result is None:
            return None
        if isinstance(result, tuple) and len(result) == 2 and type(result[0]) is int:
            status, result = result
            if status == _MOONCAKE_OBJECT_NOT_FOUND:
                return None
            if status != 0:
                raise _RecoveryJournalReadError(
                    f"Mooncake recovery journal read failed: {key}: status {status}"
                )
        if isinstance(result, memoryview):
            result = result.tobytes()
        if isinstance(result, bytearray):
            result = bytes(result)
        if not isinstance(result, bytes):
            raise ValueError(f"Mooncake recovery journal read failed: {key}")
        if not result or len(result) > _MAX_RECOVERY_JOURNAL_CHUNK_BYTES:
            raise ValueError("Mooncake recovery journal chunk exceeds the size limit")
        return result

    def _delete_recovery_journal_chunk(
        self,
        operation_id: str,
        key: str,
        *,
        phase: str,
        execution_context: WeightTransferExecutionContext | None,
    ) -> None:
        delete = self._journal_store_method(
            "delete_recovery_journal_chunk",
            ("remove", "delete"),
        )

        def remove() -> None:
            try:
                parameters = inspect.signature(delete).parameters
            except (TypeError, ValueError):
                parameters = {}
            result = delete(key, force=True) if "force" in parameters else delete(key)
            if result is False or (
                type(result) is int and result not in (0, 1, _MOONCAKE_OBJECT_NOT_FOUND)
            ):
                raise ValueError(f"Mooncake recovery journal delete failed: {key}")

        self._run_native_call(
            operation_id,
            phase,
            remove,
            execution_context,
        )

    @staticmethod
    def _update_recovery_journal_digest(
        digest: Any,
        kind: str,
        index: int,
        payload: bytes,
    ) -> None:
        digest.update(f"{kind}:{index}:{len(payload)}\n".encode("ascii"))
        digest.update(payload)

    def _recovery_record_metadata(
        self,
        prepared: _PreparedStoreMaterialize,
        manifest_digest: str,
    ) -> dict[str, Any]:
        plan = prepared.upload_plan
        request = prepared.request
        return {
            "format": _RECOVERY_RECORD_FORMAT,
            "version": 1,
            "provider": self.name,
            **self._recovery_request_fields(request),
            "manifest_digest": manifest_digest,
            "session_group_id": plan.session_group_id,
            "control_key": plan.control_key,
        }

    def _persist_recovery_journal(
        self,
        prepared: _PreparedStoreMaterialize,
        receipts: Sequence[Any] | None,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> dict[str, Any]:
        plan = prepared.upload_plan
        request = prepared.request
        operation_count = len(plan.operations)
        receipt_count = operation_count if receipts is None else len(receipts)
        self._validate_operation_count(
            request,
            operation_count,
            phase="prepare",
        )
        manifest_json = plan.manifest.to_json()
        if type(manifest_json) is not str or not manifest_json:
            raise ValueError("Mooncake upload manifest is not serializable")
        manifest_digest = self._text_sha256(manifest_json)
        request_digest = self._recovery_request_digest(request)
        generation = self._recovery_generation(request)
        journal_token = hashlib.sha256(
            f"{self.namespace}\0{request.operation_id}\0{request_digest}".encode(
                "utf-8"
            )
        ).hexdigest()
        journal_key = (
            f"{request.destination.object_prefix.rstrip('/')}"
            f"/_sglang/recovery/{journal_token}"
        )
        digest = hashlib.sha256()
        written_keys = []

        def write(kind: str, index: int, payload: bytes) -> None:
            key = self._journal_chunk_key(
                journal_key,
                kind,
                None if index < 0 else index,
            )
            # Record the deterministic key before the native call. A Store
            # response can be lost after the object has been committed.
            written_keys.append(key)
            self._put_recovery_journal_chunk(
                request.operation_id,
                key,
                payload,
                phase=f"journal.put.{kind}.{index}",
                execution_context=execution_context,
            )
            self._update_recovery_journal_digest(
                digest,
                kind,
                index,
                payload,
            )

        try:
            metadata_payload = self._canonical_json(
                self._recovery_record_metadata(
                    prepared,
                    manifest_digest,
                )
            )
            write("metadata", -1, metadata_payload)

            manifest_chunk_count = 0
            for start in range(
                0,
                len(manifest_json),
                _RECOVERY_MANIFEST_CHUNK_CHARS,
            ):
                payload = manifest_json[
                    start : start + _RECOVERY_MANIFEST_CHUNK_CHARS
                ].encode("utf-8")
                write("manifest", manifest_chunk_count, payload)
                manifest_chunk_count += 1

            entry_count = max(operation_count, receipt_count)
            record_chunk_count = 0
            start = 0
            while start < entry_count:
                stop = min(
                    entry_count,
                    start + _RECOVERY_JOURNAL_CHUNK_OPERATIONS,
                )
                payload = b""
                while stop > start:
                    operation_records = [
                        self._recovery_operation_record(plan.operations[index])
                        for index in range(start, min(stop, operation_count))
                    ]
                    if receipts is None:
                        receipt_records = [
                            {
                                "fragment_id": plan.operations[
                                    index
                                ].target.fragment_id,
                                "object_key": plan.operations[index].target.object_key,
                                "worker_id": plan.operations[index].source.worker_id,
                                "checksum": plan.operations[index].target.checksum,
                            }
                            for index in range(start, min(stop, receipt_count))
                        ]
                    else:
                        receipt_records = [
                            self._recovery_receipt_record(receipts[index])
                            for index in range(start, min(stop, receipt_count))
                        ]
                    payload = self._canonical_json(
                        {
                            "operations": operation_records,
                            "receipts": receipt_records,
                        }
                    )
                    if len(payload) <= _MAX_RECOVERY_JOURNAL_CHUNK_BYTES:
                        break
                    stop -= 1
                if stop == start:
                    raise ValueError(
                        "Mooncake recovery journal entry exceeds the size limit"
                    )
                write("records", record_chunk_count, payload)
                record_chunk_count += 1
                start = stop

            journal_digest = f"sha256:{digest.hexdigest()}"
            index = {
                "format": _RECOVERY_JOURNAL_FORMAT,
                "version": 1,
                "generation": generation,
                "journal_digest": journal_digest,
                "terminal_state": "prepared",
                "manifest_chars": len(manifest_json),
                "manifest_chunks": manifest_chunk_count,
                "record_chunks": record_chunk_count,
                "operation_count": operation_count,
                "receipt_count": receipt_count,
            }
            index_key = self._journal_chunk_key(journal_key, "index")
            written_keys.append(index_key)
            self._put_recovery_journal_chunk(
                request.operation_id,
                index_key,
                self._canonical_json(index),
                phase="journal.put.index",
                execution_context=execution_context,
            )
        except BaseException:
            rollback_context = execution_context
            pending_calls = tuple(
                status
                for status in self.pending_native_calls()
                if status.operation_id == request.operation_id
            )
            if execution_context is not None and pending_calls:
                drained = self.drain_pending_calls(
                    timeout_ms=int(_STORE_TERMINAL_CONTROL_TIMEOUT_SEC * 1000)
                )
                if any(
                    status.state is WeightStoreNativeCallState.PENDING
                    for status in drained
                ):
                    self._defer_recovery_journal_cleanup(
                        request.operation_id,
                        written_keys,
                    )
                    raise RuntimeError(
                        "Mooncake recovery journal rollback deferred until "
                        "pending native calls finish"
                    )
                rollback_context = _store_terminal_execution_context(
                    execution_context,
                )
            for index, key in enumerate(reversed(written_keys)):
                try:
                    self._delete_recovery_journal_chunk(
                        request.operation_id,
                        key,
                        phase=f"journal.rollback.{index}",
                        execution_context=rollback_context,
                    )
                except _StoreCallInterrupted:
                    break
                except BaseException:
                    pass
            raise

        return {
            "format": _RECOVERY_RECORD_FORMAT,
            "version": 2,
            "provider": self.name,
            "operation_id": request.operation_id,
            "journal_key": journal_key,
            "manifest_key": plan.manifest.manifest_key,
            "generation": generation,
            "request_digest": request_digest,
            "manifest_digest": manifest_digest,
            "journal_digest": journal_digest,
        }

    def _encode_recovery_ticket(self, record: dict[str, Any]) -> str:
        record_payload = self._canonical_json(record)
        envelope = {
            "record": record,
            "sha256": hashlib.sha256(record_payload).hexdigest(),
        }
        compressed = zlib.compress(self._canonical_json(envelope), level=6)
        if len(compressed) > _MAX_RECOVERY_TICKET_BYTES:
            raise ValueError("Mooncake recovery ticket exceeds the size limit")
        encoded = base64.urlsafe_b64encode(compressed).decode("ascii")
        return f"{_RECOVERY_TICKET_PREFIX}{encoded}"

    def _build_recovery_ticket(
        self,
        prepared: _PreparedStoreMaterialize,
        receipts: Sequence[Any] | None = None,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> str:
        reference = self._persist_recovery_journal(
            prepared,
            receipts,
            execution_context=execution_context,
        )
        try:
            return self._encode_recovery_ticket(reference)
        except BaseException:
            try:
                self._delete_recovery_journal(
                    prepared.request,
                    reference,
                    execution_context=execution_context,
                )
            except BaseException:
                pass
            raise

    def _decode_recovery_ticket(self, completion_ticket: str) -> dict[str, Any]:
        if type(completion_ticket) is not str or not completion_ticket.startswith(
            _RECOVERY_TICKET_PREFIX
        ):
            raise ValueError("invalid Mooncake recovery ticket")
        encoded = completion_ticket.removeprefix(_RECOVERY_TICKET_PREFIX)
        try:
            compressed = base64.b64decode(
                encoded.encode("ascii"),
                altchars=b"-_",
                validate=True,
            )
        except (UnicodeEncodeError, ValueError) as error:
            raise ValueError("invalid Mooncake recovery ticket encoding") from error
        if not compressed or len(compressed) > _MAX_RECOVERY_TICKET_BYTES:
            raise ValueError("Mooncake recovery ticket exceeds the size limit")
        decompressor = zlib.decompressobj()
        try:
            payload = decompressor.decompress(
                compressed,
                _MAX_RECOVERY_RECORD_BYTES + 1,
            )
        except zlib.error as error:
            raise ValueError("invalid Mooncake recovery ticket payload") from error
        if (
            len(payload) > _MAX_RECOVERY_RECORD_BYTES
            or not decompressor.eof
            or decompressor.unused_data
        ):
            raise ValueError("invalid Mooncake recovery ticket payload")
        try:
            envelope = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid Mooncake recovery ticket JSON") from error
        if not isinstance(envelope, dict) or set(envelope) != {"record", "sha256"}:
            raise ValueError("invalid Mooncake recovery ticket envelope")
        record = envelope["record"]
        digest = envelope["sha256"]
        if (
            not isinstance(record, dict)
            or type(digest) is not str
            or hashlib.sha256(self._canonical_json(record)).hexdigest() != digest
        ):
            raise ValueError("Mooncake recovery ticket digest mismatch")
        if (
            record.get("format") != _RECOVERY_RECORD_FORMAT
            or record.get("version") not in (1, 2)
            or record.get("provider") != self.name
        ):
            raise ValueError("unsupported Mooncake recovery ticket")
        if record["version"] == 2:
            if set(record) != {
                "format",
                "version",
                "provider",
                "operation_id",
                "journal_key",
                "manifest_key",
                "generation",
                "request_digest",
                "manifest_digest",
                "journal_digest",
            }:
                raise ValueError("invalid Mooncake recovery ticket reference")
            for name in (
                "operation_id",
                "journal_key",
                "manifest_key",
            ):
                if type(record[name]) is not str or not record[name]:
                    raise ValueError("invalid Mooncake recovery ticket reference")
            if type(record["generation"]) is not int or record["generation"] <= 0:
                raise ValueError("invalid Mooncake recovery ticket generation")
            self._require_canonical_sha256(
                record["request_digest"],
                "Mooncake recovery request digest",
            )
            self._require_canonical_sha256(
                f"sha256:{record['manifest_digest']}",
                "Mooncake recovery manifest digest",
            )
            self._require_canonical_sha256(
                record["journal_digest"],
                "Mooncake recovery journal digest",
            )
        return record

    def _validate_recovery_ticket_reference(
        self,
        request: WeightMaterializeRequest,
        reference: dict[str, Any],
    ) -> None:
        expected_manifest_key = (
            f"{request.destination.object_prefix.rstrip('/')}/manifest"
        )
        if (
            reference.get("operation_id") != request.operation_id
            or reference.get("manifest_key") != expected_manifest_key
            or reference.get("generation") != self._recovery_generation(request)
            or reference.get("request_digest") != self._recovery_request_digest(request)
        ):
            raise ValueError("Mooncake recovery ticket differs from the request")

    @staticmethod
    def _decode_recovery_journal_json(
        payload: bytes,
        name: str,
    ) -> dict[str, Any]:
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Mooncake recovery journal {name}") from error
        if not isinstance(value, dict):
            raise ValueError(f"invalid Mooncake recovery journal {name}")
        return value

    def _read_recovery_journal_index(
        self,
        request: WeightMaterializeRequest,
        reference: dict[str, Any],
        *,
        execution_context: WeightTransferExecutionContext | None,
    ) -> dict[str, Any] | None:
        index_key = self._journal_chunk_key(reference["journal_key"], "index")
        payload = self._get_recovery_journal_chunk(
            request.operation_id,
            index_key,
            phase="journal.get.index",
            execution_context=execution_context,
        )
        if payload is None:
            return None
        index = self._decode_recovery_journal_json(payload, "index")
        if set(index) != {
            "format",
            "version",
            "generation",
            "journal_digest",
            "terminal_state",
            "manifest_chars",
            "manifest_chunks",
            "record_chunks",
            "operation_count",
            "receipt_count",
        }:
            raise ValueError("invalid Mooncake recovery journal index")
        if (
            index["format"] != _RECOVERY_JOURNAL_FORMAT
            or index["version"] != 1
            or index["generation"] != reference["generation"]
            or index["journal_digest"] != reference["journal_digest"]
            or index["terminal_state"] != "prepared"
        ):
            raise ValueError("Mooncake recovery journal index mismatch")
        for name in (
            "manifest_chars",
            "manifest_chunks",
            "record_chunks",
            "operation_count",
            "receipt_count",
        ):
            if type(index[name]) is not int or index[name] < 0:
                raise ValueError("invalid Mooncake recovery journal index")
        if index["manifest_chars"] <= 0 or index["manifest_chunks"] <= 0:
            raise ValueError("invalid Mooncake recovery journal manifest")
        expected_manifest_chunks = (
            index["manifest_chars"] + _RECOVERY_MANIFEST_CHUNK_CHARS - 1
        ) // _RECOVERY_MANIFEST_CHUNK_CHARS
        entry_count = max(index["operation_count"], index["receipt_count"])
        valid_record_chunks = (
            0 < index["record_chunks"] <= entry_count
            if entry_count
            else index["record_chunks"] == 0
        )
        if (
            index["manifest_chunks"] != expected_manifest_chunks
            or not valid_record_chunks
        ):
            raise ValueError("invalid Mooncake recovery journal chunk counts")
        self._validate_operation_count(
            request,
            index["operation_count"],
            phase="recover",
        )
        if index["receipt_count"] > self.max_total_operations:
            raise ValueError("Mooncake recovery journal receipt count is invalid")
        return index

    def _load_recovery_journal(
        self,
        request: WeightMaterializeRequest,
        reference: dict[str, Any],
        *,
        execution_context: WeightTransferExecutionContext | None,
    ) -> dict[str, Any]:
        index = self._read_recovery_journal_index(
            request,
            reference,
            execution_context=execution_context,
        )
        if index is None:
            raise ValueError("Mooncake recovery journal is missing")
        digest = hashlib.sha256()
        journal_key = reference["journal_key"]

        metadata_payload = self._get_recovery_journal_chunk(
            request.operation_id,
            self._journal_chunk_key(journal_key, "metadata"),
            phase="journal.get.metadata",
            execution_context=execution_context,
        )
        if metadata_payload is None:
            raise ValueError("Mooncake recovery journal metadata is missing")
        self._update_recovery_journal_digest(
            digest,
            "metadata",
            -1,
            metadata_payload,
        )
        record = self._decode_recovery_journal_json(
            metadata_payload,
            "metadata",
        )

        manifest_parts = []
        for chunk_index in range(index["manifest_chunks"]):
            payload = self._get_recovery_journal_chunk(
                request.operation_id,
                self._journal_chunk_key(
                    journal_key,
                    "manifest",
                    chunk_index,
                ),
                phase=f"journal.get.manifest.{chunk_index}",
                execution_context=execution_context,
            )
            if payload is None:
                raise ValueError("Mooncake recovery journal manifest is incomplete")
            self._update_recovery_journal_digest(
                digest,
                "manifest",
                chunk_index,
                payload,
            )
            try:
                manifest_parts.append(payload.decode("utf-8"))
            except UnicodeDecodeError as error:
                raise ValueError(
                    "Mooncake recovery journal manifest is invalid"
                ) from error

        operations = []
        receipts = []
        for chunk_index in range(index["record_chunks"]):
            payload = self._get_recovery_journal_chunk(
                request.operation_id,
                self._journal_chunk_key(
                    journal_key,
                    "records",
                    chunk_index,
                ),
                phase=f"journal.get.records.{chunk_index}",
                execution_context=execution_context,
            )
            if payload is None:
                raise ValueError("Mooncake recovery journal records are incomplete")
            self._update_recovery_journal_digest(
                digest,
                "records",
                chunk_index,
                payload,
            )
            chunk = self._decode_recovery_journal_json(payload, "records")
            if (
                set(chunk) != {"operations", "receipts"}
                or not isinstance(
                    chunk["operations"],
                    list,
                )
                or not isinstance(chunk["receipts"], list)
            ):
                raise ValueError("Mooncake recovery journal records are invalid")
            if (
                len(chunk["operations"]) > _RECOVERY_JOURNAL_CHUNK_OPERATIONS
                or len(chunk["receipts"]) > _RECOVERY_JOURNAL_CHUNK_OPERATIONS
            ):
                raise ValueError("Mooncake recovery journal records are oversized")
            operations.extend(chunk["operations"])
            receipts.extend(chunk["receipts"])

        if f"sha256:{digest.hexdigest()}" != reference["journal_digest"]:
            raise ValueError("Mooncake recovery journal digest mismatch")
        if (
            len(operations) != index["operation_count"]
            or len(receipts) != index["receipt_count"]
        ):
            raise ValueError("Mooncake recovery journal record counts mismatch")
        manifest_json = "".join(manifest_parts)
        if (
            len(manifest_json) != index["manifest_chars"]
            or self._text_sha256(manifest_json) != reference["manifest_digest"]
            or record.get("manifest_digest") != reference["manifest_digest"]
            or len(operations) != index["operation_count"]
            or len(receipts) != index["receipt_count"]
        ):
            raise ValueError("Mooncake recovery journal content mismatch")
        record["manifest_json"] = manifest_json
        record["operations"] = operations
        record["receipts"] = receipts
        return record

    def _delete_recovery_journal(
        self,
        request: WeightMaterializeRequest,
        reference: dict[str, Any],
        *,
        execution_context: WeightTransferExecutionContext | None,
    ) -> None:
        if reference.get("version") != 2:
            return
        index = self._read_recovery_journal_index(
            request,
            reference,
            execution_context=execution_context,
        )
        if index is None:
            return
        journal_key = reference["journal_key"]

        delete_index = 0
        for kind, chunk_count in (
            ("manifest", index["manifest_chunks"]),
            ("records", index["record_chunks"]),
        ):
            for chunk_index in range(chunk_count):
                self._delete_recovery_journal_chunk(
                    request.operation_id,
                    self._journal_chunk_key(
                        journal_key,
                        kind,
                        chunk_index,
                    ),
                    phase=f"journal.delete.{delete_index}",
                    execution_context=execution_context,
                )
                delete_index += 1
        for kind in ("metadata", "index"):
            self._delete_recovery_journal_chunk(
                request.operation_id,
                self._journal_chunk_key(journal_key, kind),
                phase=f"journal.delete.{delete_index}",
                execution_context=execution_context,
            )
            delete_index += 1

    def _cleanup_recovery_ticket(
        self,
        request: WeightMaterializeRequest,
        completion_ticket: str | None,
        *,
        execution_context: WeightTransferExecutionContext | None,
    ) -> None:
        if completion_ticket is None:
            return
        reference = self._decode_recovery_ticket(completion_ticket)
        if reference.get("version") != 2:
            return
        self._validate_recovery_ticket_reference(request, reference)
        self._delete_recovery_journal(
            request,
            reference,
            execution_context=execution_context,
        )

    def _cleanup_recovery_ticket_on_root(
        self,
        request: WeightMaterializeRequest,
        completion_ticket: str | None,
        *,
        phase: str,
        execution_context: WeightTransferExecutionContext | None,
    ) -> None:
        def cleanup() -> None:
            self._cleanup_recovery_ticket(
                request,
                completion_ticket,
                execution_context=execution_context,
            )

        if self.coordinator is None or self.coordinator.world_size == 1:
            cleanup()
            return
        if execution_context is None:
            self.coordinator.run_root(
                phase,
                cleanup,
                discard_result=True,
            )
        else:
            self.coordinator.run_root(
                phase,
                cleanup,
                discard_result=True,
                execution_context=execution_context,
            )

    def _validate_recovery_record(
        self,
        request: WeightMaterializeRequest,
        record: dict[str, Any],
    ) -> str:
        destination = record.get("destination")
        payload_identity = request.payload_identity
        raw_operations = record.get("operations")
        if (
            record.get("operation_id") != request.operation_id
            or record.get("placement_digest")
            != weight_placement_set_digest(request.source_placements)
            or record.get("source_snapshot_digest")
            != weight_source_snapshot_digest(
                request.source_placements,
                request.source_bindings,
            )
            or not isinstance(destination, dict)
            or destination
            != {
                "provider": request.destination.provider,
                "storage_id": request.destination.storage_id,
                "object_prefix": request.destination.object_prefix,
            }
            or payload_identity is None
            or record.get("payload_digest") != payload_identity.payload_digest
        ):
            raise ValueError("Mooncake recovery ticket differs from the request")
        if not isinstance(raw_operations, list):
            raise ValueError("Mooncake recovery operations are invalid")
        self._validate_operation_count(
            request,
            len(raw_operations),
            phase="recover",
        )
        manifest_json = record.get("manifest_json")
        manifest_digest = record.get("manifest_digest")
        if (
            type(manifest_json) is not str
            or type(manifest_digest) is not str
            or self._text_sha256(manifest_json) != manifest_digest
        ):
            raise ValueError("Mooncake recovery manifest digest mismatch")
        return manifest_json

    def _reconstruct_recovery_plan(
        self,
        backend: Any,
        request: WeightMaterializeRequest,
        record: dict[str, Any],
    ) -> tuple[Any, tuple[Any, ...]]:
        required = (
            "ParallelRank",
            "RuntimeFragment",
            "UploadOperation",
            "WeightManifest",
            "WeightUploadPlan",
        )
        missing = [name for name in required if not hasattr(backend, name)]
        if missing:
            raise ValueError(
                "Mooncake recovery backend is missing APIs: " + ", ".join(missing)
            )
        manifest_json = self._validate_recovery_record(request, record)
        payload_identity = request.payload_identity
        assert payload_identity is not None
        manifest = backend.WeightManifest.from_json(manifest_json)
        expected_group = request.destination.storage_id.rstrip("/")
        expected_manifest_key = (
            f"{request.destination.object_prefix.rstrip('/')}/manifest"
        )
        if (
            manifest.model_id != request.source_placements[0].model_id
            or manifest.revision != request.source_placements[0].revision
            or manifest.group_id != expected_group
            or manifest.manifest_key != expected_manifest_key
        ):
            raise ValueError("Mooncake recovery manifest differs from the request")
        targets = {fragment.fragment_id: fragment for fragment in manifest.fragments}
        if len(targets) != len(manifest.fragments):
            raise ValueError("Mooncake recovery manifest has duplicate fragments")
        raw_operations = record.get("operations")
        if not isinstance(raw_operations, list):
            raise ValueError("Mooncake recovery operations are invalid")
        self._validate_operation_count(
            request,
            len(raw_operations),
            phase="recover",
        )
        operations = []
        used_targets = set()
        checksum_by_placement_fragment = {
            fragment.placement_fragment_id: fragment.checksum
            for fragment in payload_identity.fragments
        }
        for raw_operation in raw_operations:
            if not isinstance(raw_operation, dict):
                raise ValueError("Mooncake recovery operation is invalid")
            raw_source = raw_operation.get("source")
            target_fragment_id = raw_operation.get("target_fragment_id")
            if not isinstance(raw_source, dict) or type(target_fragment_id) is not str:
                raise ValueError("Mooncake recovery operation is invalid")
            target = targets.get(target_fragment_id)
            raw_rank = raw_source.get("rank")
            if (
                target is None
                or target_fragment_id in used_targets
                or not isinstance(raw_rank, list)
                or len(raw_rank) != 4
                or any(type(value) is not int or value < 0 for value in raw_rank)
            ):
                raise ValueError("Mooncake recovery operation target is invalid")
            source = backend.RuntimeFragment(
                fragment_id=raw_source["fragment_id"],
                tensor_id=raw_source["tensor_id"],
                global_offset=tuple(raw_source["global_offset"]),
                local_shape=tuple(raw_source["local_shape"]),
                address=1,
                nbytes=raw_source["nbytes"],
                worker_id=raw_source["worker_id"],
                endpoint=raw_source["endpoint"],
                rank=backend.ParallelRank(
                    dp=raw_rank[0],
                    tp=raw_rank[1],
                    pp=raw_rank[2],
                    ep=raw_rank[3],
                ),
                lease_generation=raw_source["lease_generation"],
                aliases=tuple(raw_source["aliases"]),
                placement_fragment_id=raw_source["placement_fragment_id"],
            )
            if (
                source.tensor_id != target.tensor_id
                or tuple(source.global_offset) != tuple(target.global_offset)
                or tuple(source.local_shape) != tuple(target.local_shape)
                or source.nbytes != target.nbytes
                or target.checksum
                != checksum_by_placement_fragment.get(source.placement_fragment_id)
            ):
                raise ValueError("Mooncake recovery source and target differ")
            operation_arguments = {
                "source": source,
                "target": target,
                "source_runtime_lease_id": raw_operation.get("source_runtime_lease_id"),
            }
            try:
                operation = backend.UploadOperation(**operation_arguments)
            except TypeError as error:
                if "source_runtime_lease_id" not in str(error):
                    raise
                operation = backend.UploadOperation(
                    source=source,
                    target=target,
                )
            operations.append(operation)
            used_targets.add(target_fragment_id)
        if used_targets != set(targets):
            raise ValueError("Mooncake recovery operations are incomplete")
        plan = backend.WeightUploadPlan(
            manifest=manifest,
            session_group_id=record["session_group_id"],
            control_key=record["control_key"],
            operations=tuple(operations),
        )
        raw_receipts = record.get("receipts")
        if not isinstance(raw_receipts, list):
            raise ValueError("Mooncake recovery receipts are invalid")
        receipts = []
        for raw in raw_receipts:
            if not isinstance(raw, dict):
                raise ValueError("Mooncake recovery receipt is invalid")
            try:
                receipt = _ChecksummedUploadReceipt(
                    fragment_id=raw["fragment_id"],
                    object_key=raw["object_key"],
                    worker_id=raw["worker_id"],
                    checksum=raw["checksum"],
                )
            except (KeyError, TypeError) as error:
                raise ValueError("Mooncake recovery receipt is invalid") from error
            self._receipt_identity(receipt)
            receipts.append(receipt)
        expected_receipts = Counter(
            (
                operation.target.fragment_id,
                operation.target.object_key,
                operation.source.worker_id,
                operation.target.checksum,
            )
            for operation in operations
        )
        if Counter(self._receipt_identity(receipt) for receipt in receipts) != (
            expected_receipts
        ):
            raise ValueError("Mooncake recovery receipts differ from the upload plan")
        return plan, tuple(receipts)

    def _validate_upload_plan(
        self,
        request: WeightMaterializeRequest,
        upload_plan: Any,
        runtime_manifests: Sequence[tuple[str, Any]],
    ) -> None:
        manifest = upload_plan.manifest
        expected_group = request.destination.storage_id.rstrip("/")
        expected_manifest_key = (
            f"{request.destination.object_prefix.rstrip('/')}/manifest"
        )
        runtime_values = tuple(manifest for _, manifest in runtime_manifests)
        is_root = self.coordinator is None or getattr(self.coordinator, "rank", 0) == 0
        mismatches = []
        if manifest.model_id != request.source_placements[0].model_id:
            mismatches.append("model_id")
        if manifest.revision != request.source_placements[0].revision:
            mismatches.append("revision")
        if manifest.group_id != expected_group:
            mismatches.append("group_id")
        if manifest.manifest_key != expected_manifest_key:
            mismatches.append("manifest_key")
        if request.destination.object_prefix.rstrip("/") != expected_group:
            mismatches.append("object_prefix")
        expected_descriptors = self._collect_descriptors(runtime_values)
        manifest_descriptors = {
            descriptor.tensor_id: descriptor for descriptor in manifest.tensors
        }
        if is_root:
            if tuple(manifest.tensors) != expected_descriptors:
                mismatches.append("tensors")
        elif any(
            manifest_descriptors.get(descriptor.tensor_id) != descriptor
            for descriptor in expected_descriptors
        ):
            mismatches.append("tensors")
        if mismatches:
            raise ValueError(
                "Mooncake upload plan manifest differs from the materialization "
                f"request: {', '.join(mismatches)}"
            )

        location_by_id = {
            location.fragment_id: location
            for location in request.source_locations
            if isinstance(location, RuntimeWeightLocation)
        }
        payload_identity = request.payload_identity
        checksum_by_placement_fragment = (
            {}
            if payload_identity is None
            else {
                fragment.placement_fragment_id: fragment.checksum
                for fragment in payload_identity.fragments
            }
        )
        if len(location_by_id) != len(request.source_locations):
            raise ValueError("duplicate Mooncake upload source fragment")
        operations = tuple(upload_plan.operations)
        operation_ids = tuple(operation.source.fragment_id for operation in operations)
        if len(operation_ids) != len(set(operation_ids)) or set(operation_ids) != set(
            location_by_id
        ):
            raise ValueError("Mooncake upload plan changed the source fragments")

        planned_fragments = []
        for operation in operations:
            location = location_by_id[operation.source.fragment_id]
            if (
                operation.source.placement_fragment_id != location.placement_fragment_id
                or operation.source.tensor_id != location.tensor_id
                or tuple(operation.source.global_offset) != location.global_offset
                or tuple(operation.source.local_shape) != location.local_shape
                or operation.source.address != location.address
                or operation.source.nbytes != location.nbytes
                or operation.source.worker_id != location.worker_id
                or operation.source.endpoint != location.endpoint
                or operation.source.lease_generation != location.generation
            ):
                raise ValueError("Mooncake upload plan changed the source snapshot")

            target = operation.target
            if (
                type(target.fragment_id) is not str
                or not target.fragment_id
                or target.tensor_id != location.tensor_id
                or tuple(target.global_offset) != location.global_offset
                or tuple(target.local_shape) != location.local_shape
                or type(target.object_key) is not str
                or not target.object_key
                or type(target.object_offset) is not int
                or target.object_offset < 0
                or type(target.nbytes) is not int
                or target.nbytes != location.nbytes
                or target.object_offset > (1 << 64) - 1 - target.nbytes
                or (
                    target.checksum is not None
                    and (type(target.checksum) is not str or not target.checksum)
                )
                or target.checksum
                != checksum_by_placement_fragment.get(location.placement_fragment_id)
            ):
                raise ValueError(
                    "Mooncake upload plan changed the destination fragment"
                )
            planned_fragments.append(target)

        manifest_fragments = Counter(
            self._stored_fragment_signature(fragment) for fragment in manifest.fragments
        )
        planned_fragment_counts = Counter(
            self._stored_fragment_signature(fragment) for fragment in planned_fragments
        )
        if any(
            manifest_fragments[signature] < count
            for signature, count in planned_fragment_counts.items()
        ) or (is_root and manifest_fragments != planned_fragment_counts):
            raise ValueError(
                "Mooncake upload plan manifest differs from its operations"
            )

    def _attach_payload_identity(
        self,
        request: WeightMaterializeRequest,
        upload_plan: Any,
    ) -> Any:
        payload_identity = request.payload_identity
        if payload_identity is None:
            raise WeightTransferError(
                "Mooncake WeightStore materialization requires payload checksums",
                code="PAYLOAD_IDENTITY_REQUIRED",
                provider=self.name,
                phase="prepare",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            )
        checksum_by_placement_fragment = {
            fragment.placement_fragment_id: fragment.checksum
            for fragment in payload_identity.fragments
        }
        operations = []
        target_by_id = {}
        for operation in upload_plan.operations:
            placement_fragment_id = operation.source.placement_fragment_id
            checksum = checksum_by_placement_fragment.get(placement_fragment_id)
            if checksum is None:
                raise ValueError(
                    "payload identity does not cover the Mooncake upload plan"
                )
            target = self._replace_record(
                operation.target,
                checksum=checksum,
            )
            operations.append(
                self._replace_record(
                    operation,
                    target=target,
                )
            )
            if target.fragment_id in target_by_id:
                raise ValueError("Mooncake upload plan has duplicate target fragments")
            target_by_id[target.fragment_id] = target
        fragments = tuple(
            target_by_id[fragment.fragment_id]
            for fragment in upload_plan.manifest.fragments
        )
        manifest = self._replace_record(
            upload_plan.manifest,
            fragments=fragments,
        )
        return self._replace_record(
            upload_plan,
            manifest=manifest,
            operations=tuple(operations),
        )

    @staticmethod
    def _runtime_fragment_geometry(fragment: Any) -> tuple[Any, ...]:
        return (
            fragment.tensor_id,
            tuple(fragment.global_offset),
            tuple(fragment.local_shape),
        )

    def _rank_upload_plan(
        self,
        request: WeightMaterializeRequest,
        full_upload_plan: Any,
        runtime_manifests: Sequence[tuple[str, Any]],
    ) -> Any:
        runtime_fragment_count = sum(
            len(manifest.fragments) for _, manifest in runtime_manifests
        )
        if runtime_fragment_count > self.max_total_operations:
            raise ValueError(
                "Mooncake rank upload preparation exceeds the operation limit"
            )
        full_operations = tuple(full_upload_plan.operations)
        self._validate_operation_count(
            request,
            len(full_operations),
            phase="prepare",
        )
        operation_by_geometry = {}
        for operation in full_operations:
            geometry = self._runtime_fragment_geometry(operation.source)
            if geometry in operation_by_geometry:
                raise ValueError("Mooncake upload plan has duplicate source geometry")
            operation_by_geometry[geometry] = operation

        local_operations = []
        for _, runtime_manifest in runtime_manifests:
            for source in runtime_manifest.fragments:
                operation = operation_by_geometry.get(
                    self._runtime_fragment_geometry(source)
                )
                if operation is None:
                    raise ValueError(
                        "Mooncake rank upload source is absent from the root plan"
                    )
                local_operations.append(
                    self._replace_record(
                        operation,
                        source=source,
                    )
                )
        if len(local_operations) != runtime_fragment_count:
            raise ValueError("Mooncake rank upload plan is incomplete")

        # Keep the complete manifest: WeightManifest validates full tensor
        # coverage even when the executor plan contains only local operations.
        return self._replace_record(
            full_upload_plan,
            operations=tuple(local_operations),
        )

    def _validate_compact_descriptor(
        self,
        descriptor: _StoreUploadDescriptor | _StoreCommitDescriptor,
    ) -> None:
        payload = self._canonical_json(vars(descriptor))
        if len(payload) > _MAX_STORE_COMPACT_DESCRIPTOR_BYTES:
            raise ValueError("Mooncake Store compact descriptor exceeds the size limit")
        self._require_canonical_sha256(
            descriptor.manifest_digest,
            "Mooncake upload manifest digest",
        )
        self._require_canonical_sha256(
            descriptor.payload_digest,
            "Mooncake upload payload digest",
        )
        if isinstance(descriptor, _StoreCommitDescriptor):
            self._require_canonical_sha256(
                descriptor.snapshot_digest,
                "Mooncake committed snapshot digest",
            )
        for name in (
            "operation_id",
            "model_id",
            "revision",
            "storage_id",
            "manifest_key",
            "recovery_ticket",
        ):
            value = getattr(descriptor, name)
            if type(value) is not str or not value:
                raise ValueError(f"Mooncake Store descriptor {name} is invalid")
        for name in (
            "total_bytes",
            "fragment_count",
            "operation_count",
        ):
            value = getattr(descriptor, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Mooncake Store descriptor {name} is invalid")

    def _build_upload_descriptor(
        self,
        prepared: _PreparedStoreMaterialize,
    ) -> _StoreUploadDescriptor:
        payload_identity = prepared.request.payload_identity
        if payload_identity is None or prepared.recovery_ticket is None:
            raise ValueError("Mooncake upload descriptor identity is incomplete")
        manifest_json = prepared.upload_plan.manifest.to_json()
        if type(manifest_json) is not str or not manifest_json:
            raise ValueError("Mooncake upload manifest is not serializable")
        descriptor = _StoreUploadDescriptor(
            version=_STORE_UPLOAD_DESCRIPTOR_VERSION,
            operation_id=prepared.request.operation_id,
            model_id=prepared.upload_plan.manifest.model_id,
            revision=prepared.upload_plan.manifest.revision,
            storage_id=prepared.upload_plan.manifest.group_id,
            manifest_key=prepared.upload_plan.manifest.manifest_key,
            manifest_digest=f"sha256:{self._text_sha256(manifest_json)}",
            payload_digest=payload_identity.payload_digest,
            total_bytes=prepared.request.total_bytes,
            fragment_count=len(prepared.request.source_locations),
            operation_count=len(prepared.upload_plan.operations),
            recovery_ticket=prepared.recovery_ticket,
        )
        self._validate_compact_descriptor(descriptor)
        return descriptor

    def _validate_local_upload_descriptor(
        self,
        request: WeightMaterializeRequest,
        descriptor: _StoreUploadDescriptor,
    ) -> None:
        self._validate_compact_descriptor(descriptor)
        payload_identity = request.payload_identity
        if (
            descriptor.version != _STORE_UPLOAD_DESCRIPTOR_VERSION
            or descriptor.operation_id != request.operation_id
            or descriptor.model_id != request.source_placements[0].model_id
            or descriptor.revision != request.source_placements[0].revision
            or descriptor.storage_id != request.destination.storage_id.rstrip("/")
            or descriptor.manifest_key
            != f"{request.destination.object_prefix.rstrip('/')}/manifest"
            or payload_identity is None
            or (
                (self.coordinator is None or self.coordinator.rank == _STORE_ROOT_RANK)
                and descriptor.payload_digest != payload_identity.payload_digest
            )
        ):
            raise ValueError(
                "Mooncake upload descriptor differs from the local request"
            )

    def _verify_runtime_payload(
        self,
        request: WeightMaterializeRequest,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        payload_identity = request.payload_identity
        assert payload_identity is not None
        verifier = self.payload_checksum_verifier
        if verifier is None:
            raise WeightTransferError(
                "Mooncake WeightStore materialization requires a payload "
                "checksum verifier",
                code="PAYLOAD_CHECKSUM_VERIFIER_REQUIRED",
                provider=self.name,
                phase="prepare",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            )

        self._raise_if_payload_verification_interrupted(
            request,
            execution_context,
        )
        if self.local_placement_ids is None:
            local_placements = request.source_placements
        else:
            local_placements = tuple(
                placement
                for placement in request.source_placements
                if placement.placement_id in self.local_placement_ids
            )
            if {
                placement.placement_id for placement in local_placements
            } != self.local_placement_ids:
                raise WeightTransferError(
                    "Mooncake local placement ownership differs from the "
                    "materialization request",
                    code="PAYLOAD_CHECKSUM_VERIFICATION_FAILED",
                    provider=self.name,
                    phase="prepare",
                    operation_id=request.operation_id,
                    retryable=False,
                    completion_known=True,
                    cleanup_required=False,
                )
        local_placement_ids = {placement.placement_id for placement in local_placements}
        if not local_placement_ids:
            return

        verifier_owner = getattr(verifier, "__self__", None)
        identity_attestor = getattr(
            verifier_owner,
            "attest_payload_identity",
            None,
        )
        if callable(identity_attestor):
            try:
                identity_attestor(
                    request,
                    execution_context=execution_context,
                )
            except Exception as error:
                self._raise_if_payload_verification_interrupted(
                    request,
                    execution_context,
                )
                raise WeightTransferError(
                    f"Mooncake captured payload identity attestation failed: {error}",
                    code="PAYLOAD_CHECKSUM_VERIFICATION_FAILED",
                    provider=self.name,
                    phase="prepare",
                    operation_id=request.operation_id,
                    retryable=False,
                    completion_known=True,
                    cleanup_required=False,
                ) from error
            return

        verifier_accepts_context = self._accepts_execution_context(verifier)
        observed_checksums = {}
        for location in request.source_locations:
            assert isinstance(location, RuntimeWeightLocation)
            if location.placement_id not in local_placement_ids:
                continue
            self._raise_if_payload_verification_interrupted(
                request,
                execution_context,
            )
            if location.placement_fragment_id in observed_checksums:
                raise WeightTransferError(
                    "Mooncake payload checksum verifier received duplicate "
                    "placement fragments",
                    code="PAYLOAD_CHECKSUM_VERIFICATION_FAILED",
                    provider=self.name,
                    phase="prepare",
                    operation_id=request.operation_id,
                    retryable=False,
                    completion_known=True,
                    cleanup_required=False,
                )
            try:
                if execution_context is not None and verifier_accepts_context:
                    checksum = verifier(
                        location,
                        execution_context=execution_context,
                    )
                else:
                    checksum = verifier(location)
                observed_checksums[location.placement_fragment_id] = checksum
            except Exception as error:
                self._raise_if_payload_verification_interrupted(
                    request,
                    execution_context,
                )
                raise WeightTransferError(
                    f"Mooncake payload checksum verifier failed: {error}",
                    code="PAYLOAD_CHECKSUM_VERIFICATION_FAILED",
                    provider=self.name,
                    phase="prepare",
                    operation_id=request.operation_id,
                    retryable=False,
                    completion_known=True,
                    cleanup_required=False,
                ) from error
            self._raise_if_payload_verification_interrupted(
                request,
                execution_context,
            )

        try:
            expected_identity = payload_identity.select(local_placements)
            observed_identity = payload_identity.create(
                local_placements,
                observed_checksums,
            )
        except Exception as error:
            raise WeightTransferError(
                f"Mooncake payload checksum verifier returned invalid checksums: "
                f"{error}",
                code="PAYLOAD_CHECKSUM_VERIFICATION_FAILED",
                provider=self.name,
                phase="prepare",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            ) from error
        if observed_identity != expected_identity:
            raise WeightTransferError(
                "Mooncake runtime payload checksums differ from payload identity",
                code="PAYLOAD_CHECKSUM_MISMATCH",
                provider=self.name,
                phase="prepare",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            )

    @staticmethod
    def _accepts_execution_context(verifier: Callable[..., Any]) -> bool:
        try:
            parameters = inspect.signature(verifier).parameters.values()
        except (TypeError, ValueError):
            return False
        return any(
            parameter.name == "execution_context"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )

    def _raise_if_payload_verification_interrupted(
        self,
        request: WeightMaterializeRequest,
        execution_context: WeightTransferExecutionContext | None,
    ) -> None:
        if execution_context is None:
            return
        cancelled = execution_context.cancelled()
        if not cancelled and execution_context.remaining_seconds() > 0:
            return
        reason = "cancelled" if cancelled else "deadline exceeded"
        raise WeightTransferError(
            f"Mooncake runtime payload verification {reason}",
            code="CANCELLED" if cancelled else "DEADLINE_EXCEEDED",
            provider=self.name,
            phase="prepare",
            operation_id=request.operation_id,
            retryable=False,
            completion_known=True,
            cleanup_required=False,
        )

    def _cleanup_failed_preflight(
        self,
        request: WeightMaterializeRequest,
        upload_plan: Any,
        preflight_errors: Sequence[str],
        *,
        completion_ticket: str | None,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        cleanup_errors = []
        cleanup_context = _store_terminal_execution_context(execution_context)

        def abort_upload() -> None:
            self.weight_store.abort_upload(upload_plan, ())

        try:
            if self.coordinator is None:
                self._run_native_call(
                    request.operation_id,
                    "preflight.abort",
                    abort_upload,
                    cleanup_context,
                )
            else:
                self._coordinate(
                    "abort_upload",
                    lambda: self._run_native_call(
                        request.operation_id,
                        "preflight.abort",
                        abort_upload,
                        cleanup_context,
                    ),
                    execution_context=cleanup_context,
                )
        except BaseException as error:
            cleanup_errors.append(self._error_detail(error))

        if self.coordinator is None:
            try:
                finalize = getattr(
                    self.weight_store,
                    "finalize_upload_session",
                    None,
                )
                if callable(finalize):
                    self._run_native_call(
                        request.operation_id,
                        "preflight.finalize",
                        lambda: finalize(upload_plan),
                        cleanup_context,
                    )
            except BaseException as error:
                cleanup_errors.append(self._error_detail(error))
        else:

            def finalize_upload() -> None:
                finalize = getattr(
                    self.weight_store,
                    "finalize_upload_session",
                    None,
                )
                if callable(finalize):
                    finalize(upload_plan)

            try:
                self._coordinate(
                    "finalize_upload",
                    lambda: self._run_native_call(
                        request.operation_id,
                        "preflight.finalize",
                        finalize_upload,
                        cleanup_context,
                    ),
                    execution_context=cleanup_context,
                )
            except BaseException as error:
                cleanup_errors.append(self._error_detail(error))

        if not cleanup_errors and completion_ticket is not None:
            try:
                self._cleanup_recovery_ticket_on_root(
                    request,
                    completion_ticket,
                    phase="preflight.discard_recovery",
                    execution_context=cleanup_context,
                )
            except BaseException as error:
                cleanup_errors.append(self._error_detail(error))

        if cleanup_errors:
            detail = "; ".join((*preflight_errors, *cleanup_errors))
            if completion_ticket is None:
                raise WeightTransferError(
                    detail,
                    code="PREFLIGHT_CLEANUP_FAILED",
                    provider=self.name,
                    phase="preflight",
                    operation_id=request.operation_id,
                    retryable=False,
                    completion_known=True,
                    cleanup_required=True,
                )
            raise WeightTransferCompletionUnknownError(
                detail,
                provider=self.name,
                phase="preflight",
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            )

    def _prepare_materialize(
        self,
        backend: Any,
        request: WeightMaterializeRequest,
        execution_context: WeightTransferExecutionContext | None,
    ) -> _PreparedStoreMaterialize:
        if request.payload_identity is None:
            raise WeightTransferError(
                "Mooncake WeightStore materialization requires payload checksums",
                code="PAYLOAD_IDENTITY_REQUIRED",
                provider=self.name,
                phase="prepare",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            )
        if any(
            not isinstance(location, RuntimeWeightLocation)
            for location in request.source_locations
        ):
            raise ValueError(
                "Mooncake WeightStore materialization requires runtime sources"
            )
        self._verify_runtime_payload(
            request,
            execution_context,
        )
        runtime_bindings = tuple(
            binding
            for binding in request.source_bindings
            if isinstance(binding, WeightRuntimeBindingManifest)
        )
        if len(runtime_bindings) != len(request.source_bindings):
            raise ValueError(
                "Mooncake WeightStore materialization requires runtime bindings"
            )
        runtime_manifests = self._runtime_manifests(
            backend,
            request.source_placements,
            runtime_bindings,
            backend.SourcePlacementManifest,
        )

        def prepare_upload() -> Any:
            return self.weight_store.prepare_upload(
                tuple(manifest for _, manifest in runtime_manifests),
                namespace=self.namespace,
            )

        def await_prepare() -> Any:
            if execution_context is None:
                return prepare_upload()
            return self._run_native_call(
                request.operation_id,
                "prepare",
                prepare_upload,
                execution_context,
            )

        if self.coordinator is None or self.coordinator.world_size == 1:
            cleanup_plan = None
            recovery_ticket = None
            try:
                raw_upload_plan = await_prepare()
                cleanup_plan = raw_upload_plan
                self._validate_operation_count(
                    request,
                    len(raw_upload_plan.operations),
                    phase="prepare",
                )
                upload_plan = self._attach_payload_identity(
                    request,
                    raw_upload_plan,
                )
                cleanup_plan = upload_plan
                prepared = _PreparedStoreMaterialize(
                    request=request,
                    upload_plan=upload_plan,
                    runtime_manifests=runtime_manifests,
                )
                recovery_ticket = self._build_recovery_ticket(
                    prepared,
                    execution_context=execution_context,
                )
                if type(recovery_ticket) is not str or not recovery_ticket:
                    raise ValueError("Mooncake recovery ticket is invalid")
                prepared = replace(
                    prepared,
                    recovery_ticket=recovery_ticket,
                )
                upload_descriptor = self._build_upload_descriptor(prepared)
                prepared = replace(
                    prepared,
                    upload_descriptor=upload_descriptor,
                )
                self._validate_upload_plan(
                    request,
                    upload_plan,
                    runtime_manifests,
                )
                return prepared
            except BaseException as error:
                if cleanup_plan is not None:
                    self._cleanup_failed_preflight(
                        request,
                        cleanup_plan,
                        (self._error_detail(error),),
                        completion_ticket=recovery_ticket,
                        execution_context=execution_context,
                    )
                raise

        local_preparation = _RankUploadPreparation(
            rank=self.coordinator.rank,
            runtime_manifests=runtime_manifests,
            local_placement_ids=tuple(
                sorted(
                    (placement_id for placement_id, _ in runtime_manifests)
                    if self.local_placement_ids is None
                    else self.local_placement_ids
                )
            ),
        )
        preparations = (
            self.coordinator.gather_object_to_root(
                local_preparation,
                phase="prepare_upload.gather",
            )
            if execution_context is None
            else self.coordinator.gather_object_to_root(
                local_preparation,
                phase="prepare_upload.gather",
                execution_context=execution_context,
            )
        )
        root_packets: tuple[_RankPreparedUpload, ...] | None = None
        root_cleanup_plan = None
        root_recovery_ticket = None

        def prepare_upload_on_root_unsafe() -> None:
            nonlocal root_cleanup_plan, root_packets, root_recovery_ticket
            if preparations is None:
                raise ValueError("root upload preparations are unavailable")
            if (
                not isinstance(preparations, (tuple, list))
                or len(preparations) != self.coordinator.world_size
            ):
                raise ValueError("distributed upload preparations are incomplete")
            for rank, preparation in enumerate(preparations):
                if (
                    not isinstance(preparation, _RankUploadPreparation)
                    or preparation.rank != rank
                    or len(preparation.runtime_manifests)
                    > _MAX_DISTRIBUTED_UPLOAD_RECORDS
                ):
                    raise ValueError("distributed upload preparation is invalid")
                fragment_count = sum(
                    len(manifest.fragments)
                    for _, manifest in preparation.runtime_manifests
                )
                if fragment_count > _MAX_DISTRIBUTED_UPLOAD_RECORDS:
                    raise ValueError("rank upload preparation exceeds the record limit")

            root_preparation = preparations[0]
            expected_placements = {
                placement_id for placement_id, _ in root_preparation.runtime_manifests
            }
            owned_placements = tuple(
                placement_id
                for preparation in preparations
                for placement_id in preparation.local_placement_ids
            )
            if (
                len(owned_placements) != len(set(owned_placements))
                or set(owned_placements) != expected_placements
            ):
                raise ValueError("distributed upload ownership is incomplete")

            raw_upload_plan = await_prepare()
            root_cleanup_plan = raw_upload_plan
            self._validate_operation_count(
                request,
                len(raw_upload_plan.operations),
                phase="prepare",
            )
            upload_plan = self._attach_payload_identity(
                request,
                raw_upload_plan,
            )
            root_cleanup_plan = upload_plan
            root_prepared = _PreparedStoreMaterialize(
                request=request,
                upload_plan=upload_plan,
                runtime_manifests=runtime_manifests,
            )
            recovery_ticket = self._build_recovery_ticket(
                root_prepared,
                execution_context=execution_context,
            )
            if type(recovery_ticket) is not str or not recovery_ticket:
                raise ValueError("Mooncake recovery ticket is invalid")
            root_recovery_ticket = recovery_ticket
            root_prepared = replace(
                root_prepared,
                recovery_ticket=recovery_ticket,
            )
            upload_descriptor = self._build_upload_descriptor(root_prepared)
            root_prepared = replace(
                root_prepared,
                upload_descriptor=upload_descriptor,
            )
            self._validate_upload_plan(
                request,
                upload_plan,
                runtime_manifests,
            )

            scatter_records = len(upload_plan.operations)
            for preparation in preparations[1:]:
                scatter_records += sum(
                    len(manifest.fragments)
                    for _, manifest in preparation.runtime_manifests
                )
                if scatter_records > _MAX_DISTRIBUTED_UPLOAD_RECORDS:
                    raise ValueError(
                        "distributed upload scatter exceeds the record limit"
                    )
            packets = [
                _RankPreparedUpload(
                    upload_plan=upload_plan,
                    upload_descriptor=upload_descriptor,
                    recovery_ticket=recovery_ticket,
                )
            ]
            for preparation in preparations[1:]:
                packets.append(
                    _RankPreparedUpload(
                        upload_plan=self._rank_upload_plan(
                            request,
                            upload_plan,
                            preparation.runtime_manifests,
                        ),
                        upload_descriptor=upload_descriptor,
                        recovery_ticket=recovery_ticket,
                    )
                )
            root_packets = tuple(packets)

        def prepare_upload_on_root() -> None:
            nonlocal root_packets
            try:
                prepare_upload_on_root_unsafe()
            except BaseException as error:
                detail = self._error_detail(error)
                root_packets = tuple(
                    _RankPreparedUpload(
                        upload_plan=(
                            root_cleanup_plan if rank == _STORE_ROOT_RANK else None
                        ),
                        upload_descriptor=None,
                        recovery_ticket=root_recovery_ticket,
                        error=detail,
                    )
                    for rank in range(self.coordinator.world_size)
                )

        if execution_context is None:
            self.coordinator.run_root(
                "prepare_upload",
                prepare_upload_on_root,
                discard_result=True,
            )
        else:
            self.coordinator.run_root(
                "prepare_upload",
                prepare_upload_on_root,
                discard_result=True,
                execution_context=execution_context,
            )
        packet = (
            self.coordinator.scatter_object_from_root(
                root_packets,
                phase="prepare_upload.scatter",
            )
            if execution_context is None
            else self.coordinator.scatter_object_from_root(
                root_packets,
                phase="prepare_upload.scatter",
                execution_context=execution_context,
            )
        )
        if not isinstance(packet, _RankPreparedUpload):
            raise ValueError("distributed upload scatter returned an invalid packet")
        prepared = None
        local_error = None
        if packet.error is not None:
            local_error = ValueError(packet.error)
        elif packet.upload_plan is None or packet.upload_descriptor is None:
            local_error = ValueError(
                "distributed upload scatter omitted the local plan"
            )
        else:
            prepared = _PreparedStoreMaterialize(
                request=request,
                upload_plan=packet.upload_plan,
                runtime_manifests=runtime_manifests,
                recovery_ticket=packet.recovery_ticket,
                upload_descriptor=packet.upload_descriptor,
            )
        try:
            if local_error is not None:
                raise local_error
            assert prepared is not None
            assert packet.upload_plan is not None
            assert packet.upload_descriptor is not None
            self._validate_local_upload_descriptor(
                request,
                packet.upload_descriptor,
            )
            self._validate_operation_count(
                request,
                len(packet.upload_plan.operations),
                phase="prepare",
            )
            self._validate_upload_plan(
                request,
                packet.upload_plan,
                runtime_manifests,
            )
        except BaseException as error:
            local_error = error

        outcome = WeightStorePreflightOutcome(
            rank=self.coordinator.rank,
            error=(None if local_error is None else self._error_detail(local_error)),
        )
        try:
            outcomes = self._coordinate(
                "exchange_preflight_outcome",
                outcome,
                execution_context=execution_context,
            )
        except WeightStoreDistributedError as error:
            if error.completion_unknown:
                raise
            self._cleanup_failed_preflight(
                request,
                packet.upload_plan,
                (self._error_detail(error),),
                completion_ticket=packet.recovery_ticket,
                execution_context=execution_context,
            )
            raise
        preflight_errors = tuple(
            outcome.error for outcome in outcomes if outcome.error is not None
        )
        if preflight_errors:
            self._cleanup_failed_preflight(
                request,
                packet.upload_plan,
                preflight_errors,
                completion_ticket=packet.recovery_ticket,
                execution_context=execution_context,
            )
            if local_error is not None and (
                self.coordinator is None or self.coordinator.world_size == 1
            ):
                raise local_error
            raise ValueError("; ".join(preflight_errors)) from local_error

        assert prepared is not None
        assert prepared.recovery_ticket is not None
        assert prepared.upload_descriptor is not None
        return prepared

    def _load_manifest_if_present(
        self,
        backend: Any,
        manifest_key: str,
    ) -> Any | None:
        store = getattr(self.weight_store, "store", None)
        is_exist = getattr(store, "is_exist", None)
        if callable(is_exist):
            result = is_exist(manifest_key)
            if result == 0:
                return None
            if result != 1:
                raise backend.WeightStoreError(
                    f"manifest existence check failed: {manifest_key}: {result}"
                )
            return self.weight_store.load_manifest(manifest_key)

        manifest_exists = getattr(
            self.weight_store,
            "manifest_exists",
            None,
        )
        if callable(manifest_exists):
            if not manifest_exists(manifest_key):
                return None
            return self.weight_store.load_manifest(manifest_key)

        return self.weight_store.load_manifest(manifest_key)

    def _existing_payload_keys(
        self,
        backend: Any,
        upload_plan: Any,
    ) -> frozenset[str]:
        object_keys = tuple(
            operation.target.object_key for operation in upload_plan.operations
        )
        if len(object_keys) != len(set(object_keys)):
            raise ValueError("Mooncake upload plan has duplicate payload keys")
        store = getattr(self.weight_store, "store", None)
        batch_is_exist = getattr(store, "batch_is_exist", None)
        if callable(batch_is_exist):
            results = batch_is_exist(list(object_keys))
            if (
                isinstance(results, (str, bytes, bytearray))
                or not isinstance(results, Sequence)
                or len(results) != len(object_keys)
                or any(result not in (0, 1) for result in results)
            ):
                raise backend.WeightStoreError(
                    f"payload existence check failed: {results}"
                )
            return frozenset(
                key
                for key, result in zip(object_keys, results, strict=True)
                if result == 1
            )
        is_exist = getattr(store, "is_exist", None)
        if not callable(is_exist):
            raise backend.WeightStoreError(
                "Mooncake Store does not expose payload existence checks"
            )
        existing = set()
        for key in object_keys:
            result = is_exist(key)
            if result not in (0, 1):
                raise backend.WeightStoreError(
                    f"payload existence check failed: {key}: {result}"
                )
            if result == 1:
                existing.add(key)
        return frozenset(existing)

    @staticmethod
    def _validate_terminal_decision(
        decision: Any,
        *,
        phase: str,
    ) -> _StoreTerminalDecision:
        if (
            type(decision) is not _StoreTerminalDecision
            or type(decision.state) is not _StoreTerminalState
            or not isinstance(decision.payload_keys, tuple)
            or any(type(key) is not str or not key for key in decision.payload_keys)
            or len(decision.payload_keys) != len(set(decision.payload_keys))
            or (
                decision.detail is not None
                and (type(decision.detail) is not str or not decision.detail)
            )
        ):
            raise WeightStoreDistributedError(
                phase,
                "invalid Store terminal decision",
            )
        manifest_state = decision.state in {
            _StoreTerminalState.MANIFEST_MATCH,
            _StoreTerminalState.MANIFEST_CONFLICT,
        }
        payload_state = decision.state in {
            _StoreTerminalState.PAYLOAD_COMPLETE,
            _StoreTerminalState.PAYLOAD_INCOMPLETE,
        }
        if (
            (manifest_state and decision.manifest is None)
            or (not manifest_state and decision.manifest is not None)
            or (not payload_state and decision.payload_keys)
            or (
                decision.state is _StoreTerminalState.OBSERVATION_FAILED
                and decision.detail is None
            )
        ):
            raise WeightStoreDistributedError(
                phase,
                "invalid Store terminal decision",
            )
        return decision

    def _run_terminal_observation(
        self,
        operation_id: str,
        phase: str,
        factory: Callable[[], _StoreTerminalDecision],
        execution_context: WeightTransferExecutionContext | None,
    ) -> _StoreTerminalDecision:
        def run_local() -> _StoreTerminalDecision:
            return self._run_native_call(
                operation_id,
                phase,
                factory,
                execution_context,
            )

        decision = (
            run_local()
            if self.coordinator is None
            else self._coordinate(
                "run_root",
                phase,
                run_local,
                execution_context=execution_context,
            )
        )
        return self._validate_terminal_decision(decision, phase=phase)

    def _observe_manifest(
        self,
        backend: Any,
        manifest_key: str,
        validator: Callable[[Any], None],
        *,
        operation_id: str,
        phase: str,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> _StoreTerminalDecision:
        def observe() -> _StoreTerminalDecision:
            try:
                manifest = self._load_manifest_if_present(
                    backend,
                    manifest_key,
                )
            except BaseException as error:
                return _StoreTerminalDecision(
                    state=_StoreTerminalState.OBSERVATION_FAILED,
                    detail=self._error_detail(error),
                )
            if manifest is None:
                return _StoreTerminalDecision(
                    state=_StoreTerminalState.MANIFEST_ABSENT,
                )
            try:
                validator(manifest)
            except BaseException as error:
                return _StoreTerminalDecision(
                    state=_StoreTerminalState.MANIFEST_CONFLICT,
                    manifest=manifest,
                    detail=self._error_detail(error),
                )
            return _StoreTerminalDecision(
                state=_StoreTerminalState.MANIFEST_MATCH,
                manifest=manifest,
            )

        decision = self._run_terminal_observation(
            operation_id,
            phase,
            observe,
            execution_context,
        )
        if decision.state not in {
            _StoreTerminalState.MANIFEST_MATCH,
            _StoreTerminalState.MANIFEST_ABSENT,
            _StoreTerminalState.MANIFEST_CONFLICT,
            _StoreTerminalState.OBSERVATION_FAILED,
        }:
            raise WeightStoreDistributedError(
                phase,
                "invalid manifest terminal decision",
            )
        return decision

    def _observe_payloads(
        self,
        backend: Any,
        upload_plan: Any,
        *,
        operation_id: str,
        phase: str,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> _StoreTerminalDecision:
        expected_keys = frozenset(
            operation.target.object_key for operation in upload_plan.operations
        )

        def observe() -> _StoreTerminalDecision:
            try:
                existing_keys = self._existing_payload_keys(
                    backend,
                    upload_plan,
                )
            except BaseException as error:
                return _StoreTerminalDecision(
                    state=_StoreTerminalState.OBSERVATION_FAILED,
                    detail=self._error_detail(error),
                )
            return _StoreTerminalDecision(
                state=(
                    _StoreTerminalState.PAYLOAD_COMPLETE
                    if existing_keys == expected_keys
                    else _StoreTerminalState.PAYLOAD_INCOMPLETE
                ),
                payload_keys=tuple(sorted(existing_keys)),
            )

        decision = self._run_terminal_observation(
            operation_id,
            phase,
            observe,
            execution_context,
        )
        if decision.state not in {
            _StoreTerminalState.PAYLOAD_COMPLETE,
            _StoreTerminalState.PAYLOAD_INCOMPLETE,
            _StoreTerminalState.OBSERVATION_FAILED,
        }:
            raise WeightStoreDistributedError(
                phase,
                "invalid payload terminal decision",
            )
        return decision

    def _abort_incomplete_recovery(
        self,
        prepared: _PreparedStoreMaterialize,
        receipts: Sequence[Any],
        existing_keys: frozenset[str],
        *,
        completion_ticket: str,
        execution_context: WeightTransferExecutionContext | None,
    ) -> None:
        present_receipts = tuple(
            receipt for receipt in receipts if receipt.object_key in existing_keys
        )

        def abort_upload() -> None:
            self.weight_store.abort_upload(
                prepared.upload_plan,
                present_receipts,
            )

        try:
            if self.coordinator is None:
                self._run_native_call(
                    prepared.request.operation_id,
                    "recover.abort",
                    abort_upload,
                    execution_context,
                )
            else:
                self._coordinate(
                    "abort_upload",
                    lambda: self._run_native_call(
                        prepared.request.operation_id,
                        "recover.abort",
                        abort_upload,
                        execution_context,
                    ),
                    execution_context=execution_context,
                )
        except Exception as error:
            raise WeightTransferCompletionUnknownError(
                f"incomplete recovered payload could not be aborted: {error}",
                provider=self.name,
                phase="recover",
                operation_id=prepared.request.operation_id,
                completion_ticket=completion_ticket,
            ) from error

        release_error = None
        try:
            self.release(
                prepared,
                None,
                execution_context=execution_context,
            )
        except Exception as error:
            release_error = error
        detail = "recovered materialization has incomplete payload objects"
        if release_error is not None:
            detail += f"; terminal cleanup failed: {release_error}"
        raise WeightTransferError(
            detail,
            code="RECOVERY_INCOMPLETE_PAYLOAD",
            provider=self.name,
            phase="recover",
            operation_id=prepared.request.operation_id,
            retryable=False,
            completion_known=True,
            cleanup_required=release_error is not None,
        )

    def _raise_recovery_conflict(
        self,
        prepared: _PreparedStoreMaterialize,
        receipts: Sequence[Any],
        *,
        completion_ticket: str,
        cause: BaseException | None = None,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        cleanup_errors = []

        def abort_upload() -> None:
            self.weight_store.abort_upload(
                prepared.upload_plan,
                tuple(receipts),
            )

        try:
            if self.coordinator is None:
                self._run_native_call(
                    prepared.request.operation_id,
                    "recover.abort_conflict",
                    abort_upload,
                    execution_context,
                )
            else:
                self._coordinate(
                    "abort_upload",
                    lambda: self._run_native_call(
                        prepared.request.operation_id,
                        "recover.abort_conflict",
                        abort_upload,
                        execution_context,
                    ),
                    execution_context=execution_context,
                )
        except Exception as error:
            cleanup_errors.append(error)

        try:
            self.release(
                prepared,
                None,
                execution_context=execution_context,
            )
        except Exception as error:
            cleanup_errors.append(error)

        if cleanup_errors:
            details = "; ".join(self._error_detail(error) for error in cleanup_errors)
            raise WeightTransferCompletionUnknownError(
                "conflicting weight revision cleanup could not be confirmed: "
                f"{details}",
                provider=self.name,
                phase="recover",
                operation_id=prepared.request.operation_id,
                completion_ticket=completion_ticket,
            ) from (cause or cleanup_errors[0])

        pending = self._pending_materializations.pop(
            prepared.request.operation_id,
            None,
        )
        if pending is not None:
            pending.aborted = True
        conflict = WeightTransferError(
            "conflicting weight revision during Mooncake recovery",
            code="STORAGE_CONFLICT",
            provider=self.name,
            phase="recover",
            operation_id=prepared.request.operation_id,
            retryable=False,
            completion_known=True,
            cleanup_required=False,
        )
        if cause is None:
            raise conflict
        raise conflict from cause

    def _complete_recovered_materialization(
        self,
        prepared: _PreparedStoreMaterialize,
        manifest: Any,
        *,
        completion_ticket: str,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> WeightMaterializeReceipt:
        self._validate_committed_manifest(prepared, manifest)
        pending = self._pending_materializations.pop(
            prepared.request.operation_id,
            None,
        )
        if pending is not None:
            pending.committed = True
        receipt = self._materialize_receipt(
            prepared,
            manifest,
            completion_ticket=completion_ticket,
        )
        try:
            self.release(
                prepared,
                receipt,
                execution_context=execution_context,
            )
        except Exception as error:
            raise WeightTransferReleaseError(
                str(error),
                receipt=receipt,
            ) from error
        return receipt

    def _reconcile_pending_recovery(
        self,
        request: WeightMaterializeRequest,
        pending: _StoreSubmission | None,
        *,
        completion_ticket: str,
        execution_context: WeightTransferExecutionContext | None,
    ) -> WeightMaterializeReceipt | None:
        if execution_context is not None:
            local_error = None
            local_completion_unknown = False
            if pending is not None:
                for call in (
                    pending.local_upload_call,
                    pending.local_commit_call,
                ):
                    if call is None or call.done.is_set():
                        continue
                    try:
                        self._await_native_call(
                            call,
                            execution_context,
                        )
                    except _StoreCallInterrupted as error:
                        local_error = str(error)
                        local_completion_unknown = True
                        break
                    except BaseException:
                        break
            elif execution_context.expired():
                local_error = "recovery deadline expired"
                local_completion_unknown = True

            outcome = WeightStorePreflightOutcome(
                rank=0 if self.coordinator is None else self.coordinator.rank,
                error=local_error,
                completion_unknown=local_completion_unknown,
            )
            outcomes = (
                (outcome,)
                if self.coordinator is None
                else self._coordinate(
                    "exchange_preflight_outcome",
                    outcome,
                    execution_context=execution_context,
                )
            )
            unknown = tuple(item for item in outcomes if item.completion_unknown)
            if unknown:
                raise WeightTransferCompletionUnknownError(
                    "; ".join(f"rank {item.rank}: {item.error}" for item in unknown),
                    provider=self.name,
                    phase="recover",
                    operation_id=request.operation_id,
                    completion_ticket=completion_ticket,
                )

        if (
            pending is None
            or (self.coordinator is not None and self.coordinator.world_size != 1)
            or pending.committed
            or pending.aborted
        ):
            return None

        receipt = self.wait(
            pending,
            execution_context=execution_context,
        )
        if not isinstance(receipt, WeightMaterializeReceipt):
            raise WeightTransferError(
                "pending materialization returned an invalid receipt",
                code="INVALID_RECEIPT",
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=False,
                cleanup_required=True,
            )
        self.synchronize(receipt)
        try:
            self.release(
                pending.prepared,
                receipt,
                execution_context=execution_context,
            )
        except BaseException as error:
            raise WeightTransferReleaseError(
                str(error),
                receipt=receipt,
                release_error=error,
            ) from error
        if self._pending_materializations.get(request.operation_id) is pending:
            self._pending_materializations.pop(request.operation_id)
        return receipt

    def _resolve_recovery_payloads(
        self,
        backend: Any,
        request: WeightMaterializeRequest,
        record: dict[str, Any],
        manifest_json: str,
        *,
        completion_ticket: str,
        execution_context: WeightTransferExecutionContext | None,
        manifest_decision: _StoreTerminalDecision | None = None,
    ) -> _StoreRecoveryResolution:
        manifest_key = f"{request.destination.object_prefix.rstrip('/')}/manifest"

        def validate_recovery_manifest(manifest: Any) -> None:
            if manifest.to_json() != manifest_json:
                raise ValueError("persisted manifest differs from the recovery ticket")

        if manifest_decision is None:
            try:
                manifest_decision = self._observe_manifest(
                    backend,
                    manifest_key,
                    validate_recovery_manifest,
                    operation_id=request.operation_id,
                    phase="materialization.recover.observe_manifest",
                    execution_context=execution_context,
                )
            except Exception as error:
                raise WeightTransferCompletionUnknownError(
                    f"manifest decision failed during recovery: {error}",
                    provider=self.name,
                    phase="recover",
                    operation_id=request.operation_id,
                    completion_ticket=completion_ticket,
                ) from error
        if manifest_decision.state is _StoreTerminalState.OBSERVATION_FAILED:
            raise WeightTransferCompletionUnknownError(
                "manifest observation failed during recovery: "
                f"{manifest_decision.detail}",
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            )
        if manifest_decision.state is _StoreTerminalState.MANIFEST_MATCH:
            manifest = manifest_decision.manifest
            return _StoreRecoveryResolution(
                prepared=_PreparedStoreMaterialize(
                    request=request,
                    upload_plan=SimpleNamespace(
                        manifest=manifest,
                        session_group_id=record["session_group_id"],
                        control_key=record["control_key"],
                        operations=(),
                    ),
                    runtime_manifests=(),
                    recovery_ticket=completion_ticket,
                ),
                receipts=(),
                manifest=manifest,
            )

        try:
            upload_plan, receipts = self._reconstruct_recovery_plan(
                backend,
                request,
                record,
            )
        except Exception as error:
            if manifest_decision.state is _StoreTerminalState.MANIFEST_CONFLICT:
                raise WeightTransferCompletionUnknownError(
                    "conflicting weight revision loser plan could not be "
                    f"reconstructed: {error}",
                    provider=self.name,
                    phase="recover",
                    operation_id=request.operation_id,
                    completion_ticket=completion_ticket,
                ) from error
            raise WeightTransferError(
                str(error),
                code="INVALID_COMPLETION_TICKET",
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            ) from error
        prepared = _PreparedStoreMaterialize(
            request=request,
            upload_plan=upload_plan,
            runtime_manifests=(),
            recovery_ticket=completion_ticket,
        )
        if manifest_decision.state is _StoreTerminalState.MANIFEST_CONFLICT:
            self._raise_recovery_conflict(
                prepared,
                receipts,
                completion_ticket=completion_ticket,
                execution_context=execution_context,
            )

        try:
            payload_decision = self._observe_payloads(
                backend,
                upload_plan,
                operation_id=request.operation_id,
                phase="materialization.recover.observe_payloads",
                execution_context=execution_context,
            )
        except Exception as error:
            raise WeightTransferCompletionUnknownError(
                f"payload decision failed during recovery: {error}",
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            ) from error
        if payload_decision.state is _StoreTerminalState.OBSERVATION_FAILED:
            raise WeightTransferCompletionUnknownError(
                "payload observation failed during recovery: "
                f"{payload_decision.detail}",
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            )
        if payload_decision.state is _StoreTerminalState.PAYLOAD_INCOMPLETE:
            self._abort_incomplete_recovery(
                prepared,
                receipts,
                frozenset(payload_decision.payload_keys),
                completion_ticket=completion_ticket,
                execution_context=execution_context,
            )
        return _StoreRecoveryResolution(
            prepared=prepared,
            receipts=tuple(receipts),
        )

    def _commit_recovered_materialization(
        self,
        backend: Any,
        resolution: _StoreRecoveryResolution,
        *,
        completion_ticket: str,
        execution_context: WeightTransferExecutionContext | None,
    ) -> Any:
        prepared = resolution.prepared
        request = prepared.request
        upload_plan = prepared.upload_plan

        def commit_upload() -> Any:
            return self._run_native_call(
                request.operation_id,
                "recover.commit",
                lambda: self.weight_store.commit(
                    upload_plan,
                    resolution.receipts,
                ),
                execution_context,
            )

        try:
            return (
                commit_upload()
                if self.coordinator is None
                else self._coordinate(
                    "commit_upload",
                    commit_upload,
                    execution_context=execution_context,
                )
            )
        except Exception as error:
            try:
                manifest_decision = self._observe_manifest(
                    backend,
                    upload_plan.manifest.manifest_key,
                    lambda manifest: self._validate_committed_manifest(
                        prepared,
                        manifest,
                    ),
                    operation_id=request.operation_id,
                    phase="materialization.recover.reconcile_manifest",
                    execution_context=execution_context,
                )
            except Exception as observation_error:
                raise WeightTransferCompletionUnknownError(
                    f"{error}; manifest decision failed: {observation_error}",
                    provider=self.name,
                    phase="recover",
                    operation_id=request.operation_id,
                    completion_ticket=completion_ticket,
                ) from error
            if manifest_decision.state is _StoreTerminalState.MANIFEST_MATCH:
                return manifest_decision.manifest
            if manifest_decision.state is _StoreTerminalState.MANIFEST_CONFLICT:
                self._raise_recovery_conflict(
                    prepared,
                    resolution.receipts,
                    completion_ticket=completion_ticket,
                    cause=error,
                    execution_context=execution_context,
                )
            detail = str(error)
            if manifest_decision.detail is not None:
                detail += f"; manifest observation failed: {manifest_decision.detail}"
            raise WeightTransferCompletionUnknownError(
                detail,
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            ) from error

    @staticmethod
    def _project_materialize_receipt(
        receipt: WeightMaterializeReceipt,
        placement_ids: tuple[str, ...],
    ) -> WeightMaterializeReceipt:
        placement_id_set = set(placement_ids)
        placements = tuple(
            placement
            for placement in receipt.stored_placements
            if placement.placement_id in placement_id_set
        )
        bindings = tuple(
            binding
            for binding in receipt.storage_bindings
            if binding.placement_id in placement_id_set
        )
        if len(placements) != len(placement_id_set) or len(bindings) != len(
            placement_id_set
        ):
            raise ValueError("distributed recovery receipt projection is incomplete")
        return replace(
            receipt,
            stored_placements=placements,
            storage_bindings=bindings,
            total_bytes=sum(
                tensor.nbytes
                for placement in placements
                for tensor in placement.tensors
            ),
            fragment_count=sum(len(placement.tensors) for placement in placements),
        )

    def _recover_distributed_materialization(
        self,
        request: WeightMaterializeRequest,
        *,
        completion_ticket: str,
        execution_context: WeightTransferExecutionContext | None,
    ) -> WeightMaterializeReceipt:
        assert self.coordinator is not None
        projection = _RankRecoveryProjection(
            rank=self.coordinator.rank,
            operation_id=request.operation_id,
            placement_ids=tuple(
                placement.placement_id for placement in request.source_placements
            ),
        )
        gathered = (
            self.coordinator.gather_object_to_root(
                projection,
                phase="recover_materialization.gather",
            )
            if execution_context is None
            else self.coordinator.gather_object_to_root(
                projection,
                phase="recover_materialization.gather",
                execution_context=execution_context,
            )
        )
        packets = None

        def recover_on_root() -> None:
            nonlocal packets
            if (
                not isinstance(gathered, (tuple, list))
                or len(gathered) != self.coordinator.world_size
            ):
                raise ValueError("distributed recovery projections are incomplete")
            for rank, item in enumerate(gathered):
                if (
                    not isinstance(item, _RankRecoveryProjection)
                    or item.rank != rank
                    or item.operation_id != request.operation_id
                    or not item.placement_ids
                ):
                    raise ValueError("distributed recovery projection is invalid")
            coordinator = self.coordinator
            self.coordinator = None
            try:
                receipt = self.recover_materialization(
                    request,
                    completion_ticket=completion_ticket,
                    execution_context=execution_context,
                )
            finally:
                self.coordinator = coordinator
            if not isinstance(receipt, WeightMaterializeReceipt):
                raise ValueError(
                    "root recovery did not return a materialization receipt"
                )
            snapshot = StoredWeightSnapshot.create(
                provider=self.name,
                storage_id=request.destination.storage_id,
                manifest_key=receipt.manifest_key,
                placements=receipt.stored_placements,
                storage_bindings=receipt.storage_bindings,
            )
            packets = tuple(
                _RankRecoveryResult(
                    receipt=self._project_materialize_receipt(
                        receipt,
                        item.placement_ids,
                    ),
                    terminal_ref=snapshot.ref,
                )
                for item in gathered
            )

        try:
            if execution_context is None:
                self.coordinator.run_root(
                    "recover_materialization",
                    recover_on_root,
                    discard_result=True,
                )
            else:
                self.coordinator.run_root(
                    "recover_materialization",
                    recover_on_root,
                    discard_result=True,
                    execution_context=execution_context,
                )
            result = (
                self.coordinator.scatter_object_from_root(
                    packets,
                    phase="recover_materialization.scatter",
                )
                if execution_context is None
                else self.coordinator.scatter_object_from_root(
                    packets,
                    phase="recover_materialization.scatter",
                    execution_context=execution_context,
                )
            )
        except WeightStoreDistributedError as error:
            if error.completion_unknown:
                raise WeightTransferCompletionUnknownError(
                    str(error),
                    provider=self.name,
                    phase="recover",
                    operation_id=request.operation_id,
                    completion_ticket=completion_ticket,
                ) from error
            raise WeightTransferError(
                str(error),
                code="DISTRIBUTED_FAILURE",
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            ) from error
        if not isinstance(result, _RankRecoveryResult):
            raise WeightTransferCompletionUnknownError(
                "distributed recovery returned an invalid rank-local result",
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            )
        pending = self._pending_materializations.pop(request.operation_id, None)
        if pending is not None:
            pending.committed = True
        self._materialization_terminal_refs[request.operation_id] = result.terminal_ref
        return result.receipt

    def recover_materialization(
        self,
        request: WeightMaterializeRequest,
        *,
        completion_ticket: str | None = None,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> WeightMaterializeReceipt | None:
        self._remember_execution_context(execution_context)
        backend = self._load_backend()
        if request.profile != "runtime_to_storage":
            raise ValueError("Mooncake recovery requires a runtime-to-storage request")
        if completion_ticket is None:
            return None
        if self.coordinator is not None and self.coordinator.world_size > 1:
            return self._recover_distributed_materialization(
                request,
                completion_ticket=completion_ticket,
                execution_context=execution_context,
            )
        try:
            ticket_record = self._decode_recovery_ticket(completion_ticket)
            if ticket_record["version"] == 2:
                self._validate_recovery_ticket_reference(
                    request,
                    ticket_record,
                )
                record = self._load_recovery_journal(
                    request,
                    ticket_record,
                    execution_context=execution_context,
                )
                manifest_json = self._validate_recovery_record(request, record)
            else:
                record = ticket_record
                manifest_json = self._validate_recovery_record(request, record)
        except (_RecoveryJournalReadError, _StoreCallInterrupted) as error:
            raise WeightTransferCompletionUnknownError(
                str(error),
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            ) from error
        except Exception as error:
            raise WeightTransferError(
                str(error),
                code="INVALID_COMPLETION_TICKET",
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            ) from error

        pending_receipt = self._reconcile_pending_recovery(
            request,
            self._pending_materializations.get(request.operation_id),
            completion_ticket=completion_ticket,
            execution_context=execution_context,
        )
        if pending_receipt is not None:
            return pending_receipt

        assert record is not None
        assert manifest_json is not None
        resolution = self._resolve_recovery_payloads(
            backend,
            request,
            record,
            manifest_json,
            completion_ticket=completion_ticket,
            execution_context=execution_context,
        )
        manifest = resolution.manifest
        if manifest is None:
            manifest = self._commit_recovered_materialization(
                backend,
                resolution,
                completion_ticket=completion_ticket,
                execution_context=execution_context,
            )
        return self._complete_recovered_materialization(
            resolution.prepared,
            manifest,
            completion_ticket=completion_ticket,
            execution_context=execution_context,
        )

    def _prepare_load(
        self,
        backend: Any,
        request: WeightLoadRequest,
        execution_context: WeightTransferExecutionContext | None,
    ) -> _PreparedStoreLoad:
        if any(
            not isinstance(region.source, StorageWeightLocation)
            for region in request.plan.regions
        ):
            raise ValueError("Mooncake WeightStore loading requires storage sources")
        max_range_bytes = getattr(
            self.weight_store,
            "max_range_bytes",
            None,
        )
        if max_range_bytes is not None:
            if type(max_range_bytes) is not int or max_range_bytes <= 0:
                raise ValueError(
                    "Mooncake WeightStore max_range_bytes must be positive"
                )
            lowered_operations = sum(
                region.segment_count
                * ((region.inner_bytes + max_range_bytes - 1) // max_range_bytes)
                for region in request.plan.regions
            )
            if lowered_operations > self.max_total_operations:
                raise WeightTransferError(
                    "Mooncake WeightStore lowering exceeds the total operation limit",
                    code="UNSUPPORTED_CAPABILITY",
                    provider=self.name,
                    phase="prepare",
                    operation_id=request.operation_id,
                    retryable=False,
                    completion_known=True,
                    cleanup_required=False,
                )
        source_locations = bind_weight_source(
            request.plan.logical_plan.source_placements,
            request.plan.source_bindings,
        )
        if any(
            not isinstance(location, StorageWeightLocation)
            for location in source_locations
        ):
            raise ValueError("Mooncake WeightStore loading requires storage bindings")
        storage_ids = {
            location.storage_id
            for location in source_locations
            if isinstance(location, StorageWeightLocation)
        }
        providers = {
            location.provider
            for location in source_locations
            if isinstance(location, StorageWeightLocation)
        }
        if len(storage_ids) != 1 or providers != {self.name}:
            raise ValueError(
                "Mooncake WeightStore source must use one storage revision"
            )
        storage_id = next(iter(storage_ids)).rstrip("/")
        manifest = self._run_native_call(
            request.operation_id,
            "load.prepare_manifest",
            lambda: self.weight_store.load_manifest(f"{storage_id}/manifest"),
            execution_context,
        )
        logical = request.plan.logical_plan
        if (
            manifest.model_id != logical.model_id
            or manifest.revision != logical.revision
            or manifest.group_id != storage_id
        ):
            raise ValueError("Mooncake stored manifest identity mismatch")

        source_placements = tuple(
            backend.SourcePlacementManifest.from_runtime_inventory(placement)
            for placement in logical.source_placements
        )
        expected_descriptors = self._collect_descriptors(source_placements)
        if tuple(manifest.tensors) != expected_descriptors:
            raise ValueError("Mooncake stored manifest tensor descriptors differ")
        persisted_by_geometry = {
            (
                fragment.tensor_id,
                tuple(fragment.global_offset),
                tuple(fragment.local_shape),
            ): fragment
            for fragment in manifest.fragments
        }
        location_by_geometry = {
            (
                location.tensor_id,
                location.global_offset,
                location.local_shape,
            ): location
            for location in source_locations
            if isinstance(location, StorageWeightLocation)
        }
        if len(persisted_by_geometry) != len(manifest.fragments) or set(
            persisted_by_geometry
        ) != set(location_by_geometry):
            raise ValueError("Mooncake stored manifest fragment geometry differs")
        for geometry, location in location_by_geometry.items():
            fragment = persisted_by_geometry[geometry]
            if (
                fragment.fragment_id != location.fragment_id
                or fragment.object_key != location.object_key
                or fragment.object_offset != location.object_offset
                or fragment.nbytes != location.nbytes
                or fragment.checksum != location.checksum
            ):
                raise ValueError("Mooncake stored manifest fragment binding differs")

        target_placements = tuple(
            backend.TargetPlacementManifest.from_runtime_inventory(placement)
            for placement in logical.target_placements
        )
        target_fragments = {
            fragment.placement_fragment_id: fragment
            for placement in target_placements
            for fragment in placement.fragments
        }
        operations = []
        for region in request.plan.regions:
            source_geometry = (
                region.source.tensor_id,
                region.source.global_offset,
                region.source.local_shape,
            )
            operations.append(
                backend.TransferRegion(
                    tensor_id=region.tensor_id,
                    source=persisted_by_geometry[source_geometry],
                    target=target_fragments[
                        region.logical_region.target.placement_fragment_id
                    ],
                    overlap_offset=region.logical_region.overlap_offset,
                    overlap_shape=region.logical_region.overlap_shape,
                    source_base_offset=region.source_base_offset,
                    target_base_offset=region.target_base_offset,
                    inner_bytes=region.inner_bytes,
                    outer_loop_counts=region.outer_loop_counts,
                    source_strides=region.source_strides,
                    target_strides=region.target_strides,
                )
            )
        target_bindings = tuple(
            backend.RuntimeBindingManifest.from_runtime_inventory(binding)
            for binding in request.plan.target_bindings
        )
        backend_logical = backend.LogicalTransferPlan(
            model_id=logical.model_id,
            revision=logical.revision,
            source_placements=(),
            target_placements=target_placements,
            source_tensors=tuple(manifest.tensors),
            target_tensors=self._collect_descriptors(target_placements),
            operations=tuple(operations),
            pipeline_routes=self._route_groups(backend, request),
        )
        executable = backend.bind_logical_transfer_plan(
            backend_logical,
            target_bindings,
        )
        target_manifests = tuple(
            backend.bind_runtime_manifest(placement, binding)
            for placement, binding in zip(
                target_placements,
                target_bindings,
                strict=True,
            )
        )
        if len(target_manifests) != 1:
            raise ValueError("Mooncake WeightStore provider requires one local target")
        return _PreparedStoreLoad(
            request=request,
            load_plan=backend.WeightLoadPlan(
                manifest=manifest,
                transfer=executable,
            ),
            target_manifest=target_manifests[0],
        )

    def prepare(
        self,
        request: WeightLoadRequest | WeightMaterializeRequest,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> _PreparedStoreLoad | _PreparedStoreMaterialize:
        self._remember_execution_context(execution_context)
        backend = self._load_backend()
        if isinstance(request, WeightLoadRequest):
            try:
                return self._prepare_load(
                    backend,
                    request,
                    execution_context,
                )
            except _StoreCallInterrupted as error:
                if error.started:
                    raise WeightTransferCompletionUnknownError(
                        str(error),
                        provider=self.name,
                        phase=error.phase,
                        operation_id=request.operation_id,
                    ) from error
                raise WeightTransferError(
                    str(error),
                    code="DEADLINE_EXCEEDED",
                    provider=self.name,
                    phase=error.phase,
                    operation_id=request.operation_id,
                    retryable=False,
                    completion_known=True,
                    cleanup_required=False,
                ) from error
        try:
            return self._prepare_materialize(
                backend,
                request,
                execution_context,
            )
        except WeightStoreDistributedError as error:
            if error.completion_unknown:
                raise WeightTransferCompletionUnknownError(
                    str(error),
                    provider=self.name,
                    phase=error.phase,
                    operation_id=request.operation_id,
                ) from error
            raise WeightTransferError(
                str(error),
                code="DISTRIBUTED_FAILURE",
                provider=self.name,
                phase=error.phase,
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            ) from error
        except _StoreCallInterrupted as error:
            if error.started:
                raise WeightTransferCompletionUnknownError(
                    str(error),
                    provider=self.name,
                    phase=error.phase,
                    operation_id=request.operation_id,
                ) from error
            raise WeightTransferError(
                str(error),
                code=(
                    "CANCELLED"
                    if execution_context is not None and execution_context.cancelled()
                    else "DEADLINE_EXCEEDED"
                ),
                provider=self.name,
                phase=error.phase,
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            ) from error

    def submit(
        self,
        prepared: _PreparedStoreLoad | _PreparedStoreMaterialize,
    ) -> _StoreSubmission:
        return _StoreSubmission(prepared=prepared)

    def materialization_recovery_ticket(
        self,
        prepared: _PreparedStoreLoad | _PreparedStoreMaterialize,
    ) -> str | None:
        if not isinstance(prepared, _PreparedStoreMaterialize):
            return None
        return prepared.recovery_ticket

    def materialization_terminal_ref(
        self,
        operation_id: str,
    ) -> WeightStorageRef | None:
        return self._materialization_terminal_refs.get(operation_id)

    def _materialize_receipt(
        self,
        prepared: _PreparedStoreMaterialize,
        manifest: Any,
        *,
        completion_ticket: str | None = None,
        allow_manifest_superset: bool = False,
    ) -> WeightMaterializeReceipt:
        location_by_geometry = {
            (
                location.tensor_id,
                location.global_offset,
                location.local_shape,
            ): location
            for location in prepared.request.source_locations
        }
        fragments_by_placement: dict[
            str,
            list[WeightStorageFragmentBinding],
        ] = {}
        seen = set()
        for fragment in manifest.fragments:
            geometry = (
                fragment.tensor_id,
                tuple(fragment.global_offset),
                tuple(fragment.local_shape),
            )
            location = location_by_geometry.get(geometry)
            if location is None:
                if allow_manifest_superset:
                    continue
                raise ValueError(
                    "Mooncake committed manifest changed fragment geometry"
                )
            if geometry in seen:
                raise ValueError(
                    "Mooncake committed manifest changed fragment geometry"
                )
            seen.add(geometry)
            fragments_by_placement.setdefault(
                location.placement_id,
                [],
            ).append(
                WeightStorageFragmentBinding(
                    placement_fragment_id=location.placement_fragment_id,
                    fragment_id=fragment.fragment_id,
                    object_key=fragment.object_key,
                    object_offset=fragment.object_offset,
                    nbytes=fragment.nbytes,
                    checksum=fragment.checksum,
                )
            )
        if seen != set(location_by_geometry):
            raise ValueError("Mooncake committed manifest is missing source fragments")
        placement_by_id = {
            placement.placement_id: placement
            for placement in prepared.request.source_placements
        }
        storage_bindings = tuple(
            WeightStorageBindingManifest(
                model_id=placement_by_id[placement_id].model_id,
                revision=placement_by_id[placement_id].revision,
                placement_id=placement_id,
                storage_id=manifest.group_id,
                provider=prepared.request.destination.provider,
                fragments=tuple(
                    sorted(
                        fragments,
                        key=lambda item: item.placement_fragment_id,
                    )
                ),
            )
            for placement_id, fragments in sorted(fragments_by_placement.items())
        )
        return WeightMaterializeReceipt(
            operation_id=prepared.request.operation_id,
            provider=self.name,
            manifest_key=manifest.manifest_key,
            stored_placements=prepared.request.source_placements,
            storage_bindings=storage_bindings,
            total_bytes=prepared.request.total_bytes,
            fragment_count=len(prepared.request.source_locations),
            completion_ticket=completion_ticket,
        )

    def _validate_committed_manifest(
        self,
        prepared: _PreparedStoreMaterialize,
        manifest: Any,
    ) -> None:
        request = prepared.request
        planned = prepared.upload_plan.manifest
        expected_group = request.destination.storage_id.rstrip("/")
        expected_manifest_key = (
            f"{request.destination.object_prefix.rstrip('/')}/manifest"
        )
        if (
            manifest.model_id != request.source_placements[0].model_id
            or manifest.model_id != planned.model_id
            or manifest.revision != request.source_placements[0].revision
            or manifest.revision != planned.revision
            or manifest.group_id != expected_group
            or manifest.group_id != planned.group_id
            or manifest.manifest_key != expected_manifest_key
            or manifest.manifest_key != planned.manifest_key
            or tuple(manifest.tensors) != tuple(planned.tensors)
            or Counter(
                self._stored_fragment_signature(fragment)
                for fragment in manifest.fragments
            )
            != Counter(
                self._stored_fragment_signature(fragment)
                for fragment in planned.fragments
            )
        ):
            raise ValueError(
                "Mooncake committed manifest differs from the request or upload plan"
            )

    @staticmethod
    def _error_detail(error: BaseException) -> str:
        detail = str(error)
        if not detail:
            return type(error).__name__
        return f"{type(error).__name__}: {detail}"

    @staticmethod
    def _require_canonical_sha256(value: Any, name: str) -> str:
        if type(value) is not str or not value.startswith("sha256:"):
            raise ValueError(f"{name} must be a canonical sha256 checksum")
        digest = value.removeprefix("sha256:")
        if (
            len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} must be a canonical sha256 checksum")
        return value

    @staticmethod
    def _receipt_core_identity(receipt: Any) -> tuple[str, str, str]:
        try:
            identity = (
                receipt.fragment_id,
                receipt.object_key,
                receipt.worker_id,
            )
        except AttributeError as error:
            raise ValueError(
                "Mooncake upload receipt is missing identity fields"
            ) from error
        if any(type(value) is not str or not value for value in identity):
            raise ValueError(
                "Mooncake upload receipt identity fields must not be empty"
            )
        return identity

    @classmethod
    def _receipt_identity(
        cls,
        receipt: Any,
    ) -> tuple[str, str, str, str]:
        identity = cls._receipt_core_identity(receipt)
        try:
            checksum = receipt.checksum
        except AttributeError as error:
            raise ValueError("Mooncake upload receipt is missing checksum") from error
        return (
            *identity,
            cls._require_canonical_sha256(
                checksum,
                "Mooncake upload receipt checksum",
            ),
        )

    @classmethod
    def _operation_receipt_identity(
        cls,
        operation: Any,
    ) -> tuple[str, str, str, str]:
        return (
            operation.target.fragment_id,
            operation.target.object_key,
            operation.source.worker_id,
            cls._require_canonical_sha256(
                operation.target.checksum,
                "Mooncake upload plan checksum",
            ),
        )

    def _attach_local_receipt_checksums(
        self,
        prepared: _PreparedStoreMaterialize,
        receipts: Sequence[Any],
    ) -> tuple[_ChecksummedUploadReceipt, ...]:
        expected = {}
        for operation in prepared.upload_plan.operations:
            identity = self._operation_receipt_identity(operation)
            core_identity = identity[:3]
            if core_identity in expected:
                raise ValueError(
                    "Mooncake upload plan has duplicate receipt identities"
                )
            expected[core_identity] = identity[3]

        checksummed = []
        for receipt in receipts:
            core_identity = self._receipt_core_identity(receipt)
            expected_checksum = expected.get(core_identity)
            if expected_checksum is None:
                raise ValueError(
                    "Mooncake local upload receipt does not match the upload plan"
                )
            checksum = getattr(receipt, "checksum", None)
            checksummed.append(
                _ChecksummedUploadReceipt(
                    fragment_id=core_identity[0],
                    object_key=core_identity[1],
                    worker_id=core_identity[2],
                    checksum=(expected_checksum if checksum is None else checksum),
                )
            )
        return tuple(checksummed)

    def _validate_materialization_receipts(
        self,
        prepared: _PreparedStoreMaterialize,
        receipts: Sequence[Any],
    ) -> tuple[Any, ...]:
        expected = Counter(
            self._operation_receipt_identity(operation)
            for operation in prepared.upload_plan.operations
        )
        actual = Counter(self._receipt_identity(receipt) for receipt in receipts)
        if actual != expected:
            raise ValueError("Mooncake upload receipts do not match the upload plan")
        return tuple(receipts)

    def _validate_upload_outcomes(
        self,
        prepared: _PreparedStoreMaterialize,
        outcomes: Sequence[WeightStoreUploadOutcome],
    ) -> tuple[Any, ...]:
        assert self.coordinator is not None
        expected_placements = {
            placement_id for placement_id, _ in prepared.runtime_manifests
        }
        actual_placements = tuple(
            placement_id
            for outcome in outcomes
            for placement_id in outcome.placement_ids
        )
        if (
            len(actual_placements) != len(set(actual_placements))
            or set(actual_placements) != expected_placements
        ):
            raise ValueError("Mooncake distributed upload ownership is incomplete")
        ranks = tuple(outcome.rank for outcome in outcomes)
        if len(ranks) != self.coordinator.world_size or set(ranks) != set(
            range(self.coordinator.world_size)
        ):
            raise ValueError("Mooncake distributed upload ranks are incomplete")
        errors = tuple(
            outcome.error for outcome in outcomes if outcome.error is not None
        )
        if errors:
            raise RuntimeError("; ".join(errors))

        placement_by_source_fragment = {}
        for placement_id, runtime_manifest in prepared.runtime_manifests:
            for fragment in runtime_manifest.fragments:
                previous = placement_by_source_fragment.setdefault(
                    fragment.fragment_id,
                    placement_id,
                )
                if previous != placement_id:
                    raise ValueError(
                        "Mooncake upload source fragment has multiple placements"
                    )
        expected_by_placement = {
            placement_id: set() for placement_id in expected_placements
        }
        for operation in prepared.upload_plan.operations:
            placement_id = placement_by_source_fragment.get(
                operation.source.fragment_id
            )
            if placement_id is None:
                raise ValueError("Mooncake upload operation has no declared placement")
            expected_by_placement[placement_id].add(
                self._operation_receipt_identity(operation)
            )

        receipts = []
        for outcome in outcomes:
            expected_receipts = set().union(
                *(
                    expected_by_placement[placement_id]
                    for placement_id in outcome.placement_ids
                )
            )
            receipt_identities = tuple(
                self._receipt_identity(receipt) for receipt in outcome.receipts
            )
            if (
                len(receipt_identities) != len(set(receipt_identities))
                or set(receipt_identities) != expected_receipts
            ):
                raise ValueError(
                    "Mooncake distributed upload receipts do not match "
                    "declared placements"
                )
            receipts.extend(outcome.receipts)
        return tuple(receipts)

    def _abort_materialization(
        self,
        submission: _StoreSubmission,
        receipts: Sequence[Any],
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        if submission.aborted or submission.committed:
            return
        prepared = submission.prepared
        if not isinstance(prepared, _PreparedStoreMaterialize):
            return

        def abort_upload() -> None:
            self.weight_store.abort_upload(
                prepared.upload_plan,
                tuple(receipts),
            )

        if self.coordinator is None:
            self._run_native_call(
                prepared.request.operation_id,
                "abort",
                abort_upload,
                execution_context,
            )
        else:
            self._coordinate(
                "abort_upload",
                lambda: self._run_native_call(
                    prepared.request.operation_id,
                    "abort",
                    abort_upload,
                    execution_context,
                ),
                execution_context=execution_context,
            )
        submission.aborted = True

    def _resolve_materialization_commit_error(
        self,
        submission: _StoreSubmission,
        prepared: _PreparedStoreMaterialize,
        error: BaseException,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> WeightMaterializeReceipt:
        backend = self._load_backend()
        completion_ticket = prepared.recovery_ticket
        if completion_ticket is None:
            raise WeightTransferCompletionUnknownError(
                str(error),
                provider=self.name,
                phase="commit",
                operation_id=prepared.request.operation_id,
            ) from error
        try:
            decision = self._observe_manifest(
                backend,
                prepared.upload_plan.manifest.manifest_key,
                lambda manifest: self._validate_committed_manifest(
                    prepared,
                    manifest,
                ),
                operation_id=prepared.request.operation_id,
                phase="materialization.commit.observe_manifest",
                execution_context=execution_context,
            )
        except Exception as observation_error:
            self._pending_materializations[prepared.request.operation_id] = submission
            raise WeightTransferCompletionUnknownError(
                f"{error}; manifest decision failed: {observation_error}",
                provider=self.name,
                phase="commit",
                operation_id=prepared.request.operation_id,
                completion_ticket=completion_ticket,
            ) from error
        if decision.state is _StoreTerminalState.MANIFEST_MATCH:
            submission.committed = True
            return self._materialize_receipt(
                prepared,
                decision.manifest,
                completion_ticket=completion_ticket,
                allow_manifest_superset=(
                    self.coordinator is not None and self.coordinator.rank != 0
                ),
            )
        if decision.state is _StoreTerminalState.MANIFEST_CONFLICT:
            submission.aborted = True
            raise WeightTransferError(
                str(error),
                code="STORAGE_CONFLICT",
                provider=self.name,
                phase="commit",
                operation_id=prepared.request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            ) from error
        self._pending_materializations[prepared.request.operation_id] = submission
        detail = str(error)
        if decision.detail is not None:
            detail += f"; manifest observation: {decision.detail}"
        raise WeightTransferCompletionUnknownError(
            detail,
            provider=self.name,
            phase="commit",
            operation_id=prepared.request.operation_id,
            completion_ticket=completion_ticket,
        ) from error

    def _complete_committed_materialization(
        self,
        submission: _StoreSubmission,
        prepared: _PreparedStoreMaterialize,
        manifest: Any,
    ) -> WeightMaterializeReceipt:
        try:
            self._validate_committed_manifest(prepared, manifest)
        except BaseException as error:
            self._pending_materializations[prepared.request.operation_id] = submission
            raise WeightTransferCompletionUnknownError(
                str(error),
                provider=self.name,
                phase="commit",
                operation_id=prepared.request.operation_id,
                completion_ticket=prepared.recovery_ticket,
            ) from error
        submission.committed = True
        receipt = self._materialize_receipt(
            prepared,
            manifest,
            completion_ticket=prepared.recovery_ticket,
        )
        snapshot = StoredWeightSnapshot.create(
            provider=self.name,
            storage_id=prepared.request.destination.storage_id,
            manifest_key=receipt.manifest_key,
            placements=receipt.stored_placements,
            storage_bindings=receipt.storage_bindings,
        )
        self._materialization_terminal_refs[prepared.request.operation_id] = (
            snapshot.ref
        )
        return receipt

    def _build_commit_descriptor(
        self,
        prepared: _PreparedStoreMaterialize,
        manifest: Any,
    ) -> _StoreCommitDescriptor:
        upload_descriptor = prepared.upload_descriptor
        if upload_descriptor is None:
            raise ValueError("Mooncake upload descriptor is unavailable")
        self._validate_committed_manifest(prepared, manifest)
        receipt = self._materialize_receipt(
            prepared,
            manifest,
            completion_ticket=prepared.recovery_ticket,
        )
        snapshot = StoredWeightSnapshot.create(
            provider=self.name,
            storage_id=prepared.request.destination.storage_id,
            manifest_key=receipt.manifest_key,
            placements=receipt.stored_placements,
            storage_bindings=receipt.storage_bindings,
        )
        descriptor = _StoreCommitDescriptor(
            version=_STORE_COMMIT_DESCRIPTOR_VERSION,
            operation_id=upload_descriptor.operation_id,
            model_id=upload_descriptor.model_id,
            revision=upload_descriptor.revision,
            storage_id=upload_descriptor.storage_id,
            manifest_key=upload_descriptor.manifest_key,
            manifest_digest=upload_descriptor.manifest_digest,
            snapshot_digest=snapshot.digest,
            payload_digest=upload_descriptor.payload_digest,
            total_bytes=upload_descriptor.total_bytes,
            fragment_count=upload_descriptor.fragment_count,
            operation_count=upload_descriptor.operation_count,
            recovery_ticket=upload_descriptor.recovery_ticket,
        )
        self._validate_compact_descriptor(descriptor)
        return descriptor

    def _validate_commit_descriptor(
        self,
        prepared: _PreparedStoreMaterialize,
        descriptor: _StoreCommitDescriptor,
    ) -> None:
        self._validate_compact_descriptor(descriptor)
        upload_descriptor = prepared.upload_descriptor
        if upload_descriptor is None:
            raise ValueError("Mooncake upload descriptor is unavailable")
        if (
            descriptor.version != _STORE_COMMIT_DESCRIPTOR_VERSION
            or descriptor.operation_id != upload_descriptor.operation_id
            or descriptor.model_id != upload_descriptor.model_id
            or descriptor.revision != upload_descriptor.revision
            or descriptor.storage_id != upload_descriptor.storage_id
            or descriptor.manifest_key != upload_descriptor.manifest_key
            or descriptor.manifest_digest != upload_descriptor.manifest_digest
            or descriptor.payload_digest != upload_descriptor.payload_digest
            or descriptor.total_bytes != upload_descriptor.total_bytes
            or descriptor.fragment_count != upload_descriptor.fragment_count
            or descriptor.operation_count != upload_descriptor.operation_count
            or descriptor.recovery_ticket != upload_descriptor.recovery_ticket
        ):
            raise ValueError(
                "Mooncake commit descriptor differs from the upload descriptor"
            )

    def _complete_compact_materialization(
        self,
        submission: _StoreSubmission,
        prepared: _PreparedStoreMaterialize,
        descriptor: _StoreCommitDescriptor,
    ) -> WeightMaterializeReceipt:
        try:
            self._validate_commit_descriptor(prepared, descriptor)
            receipt = self._materialize_receipt(
                prepared,
                prepared.upload_plan.manifest,
                completion_ticket=prepared.recovery_ticket,
                allow_manifest_superset=True,
            )
        except BaseException as error:
            self._pending_materializations[prepared.request.operation_id] = submission
            raise WeightTransferCompletionUnknownError(
                str(error),
                provider=self.name,
                phase="commit",
                operation_id=prepared.request.operation_id,
                completion_ticket=prepared.recovery_ticket,
            ) from error
        submission.committed = True
        self._materialization_terminal_refs[prepared.request.operation_id] = (
            WeightStorageRef(
                provider=self.name,
                storage_id=descriptor.storage_id,
                manifest_key=descriptor.manifest_key,
                manifest_digest=descriptor.snapshot_digest,
            )
        )
        return receipt

    def _run_local_upload(
        self,
        submission: _StoreSubmission,
        prepared: _PreparedStoreMaterialize,
        local_manifests: Sequence[tuple[str, Any]],
        execution_context: WeightTransferExecutionContext | None,
    ) -> tuple[Any, ...]:
        def upload() -> tuple[Any, ...]:
            receipts = []
            for _, runtime_manifest in local_manifests:
                receipts.extend(
                    self._attach_local_receipt_checksums(
                        prepared,
                        self.weight_store.upload(
                            prepared.upload_plan,
                            runtime_manifest,
                            pre_registered=self.source_pre_registered,
                        ),
                    )
                )
            return tuple(receipts)

        if execution_context is None:
            return upload()
        if submission.local_upload_call is None:
            if execution_context.expired():
                raise _StoreCallInterrupted("upload", started=False)
            submission.local_upload_call = self._get_or_start_native_call(
                prepared.request.operation_id,
                "upload",
                upload,
            )
        return tuple(
            self._await_native_call(
                submission.local_upload_call,
                execution_context,
            )
        )

    def _run_local_commit(
        self,
        submission: _StoreSubmission,
        prepared: _PreparedStoreMaterialize,
        receipts: tuple[Any, ...],
        execution_context: WeightTransferExecutionContext | None,
    ) -> Any:
        def commit() -> Any:
            return self.weight_store.commit(
                prepared.upload_plan,
                receipts,
            )

        if execution_context is None:
            return commit()
        if submission.local_commit_call is None:
            if execution_context.expired():
                raise _StoreCallInterrupted("commit", started=False)
            submission.local_commit_call = self._get_or_start_native_call(
                prepared.request.operation_id,
                "commit",
                commit,
            )
        return self._await_native_call(
            submission.local_commit_call,
            execution_context,
        )

    def _wait_legacy_materialization(
        self,
        submission: _StoreSubmission,
        prepared: _PreparedStoreMaterialize,
        execution_context: WeightTransferExecutionContext | None,
    ) -> WeightMaterializeReceipt:
        local_manifests = tuple(
            (placement_id, runtime_manifest)
            for placement_id, runtime_manifest in prepared.runtime_manifests
            if self.local_placement_ids is None
            or placement_id in self.local_placement_ids
        )
        try:
            submission.receipts[:] = self._run_local_upload(
                submission,
                prepared,
                local_manifests,
                execution_context,
            )
        except _StoreCallInterrupted as error:
            if error.started:
                self._pending_materializations[prepared.request.operation_id] = (
                    submission
                )
                raise WeightTransferCompletionUnknownError(
                    str(error),
                    provider=self.name,
                    phase=error.phase,
                    operation_id=prepared.request.operation_id,
                    completion_ticket=prepared.recovery_ticket,
                ) from error
            raise WeightTransferError(
                str(error),
                code="DEADLINE_EXCEEDED",
                provider=self.name,
                phase=error.phase,
                operation_id=prepared.request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            ) from error
        except BaseException as error:
            try:
                self._abort_materialization(
                    submission,
                    submission.receipts,
                    execution_context,
                )
            except BaseException as abort_error:
                self._pending_materializations[prepared.request.operation_id] = (
                    submission
                )
                raise WeightTransferCompletionUnknownError(
                    f"{error}; abort failed: {abort_error}",
                    provider=self.name,
                    phase="abort",
                    operation_id=prepared.request.operation_id,
                    completion_ticket=prepared.recovery_ticket,
                ) from abort_error
            raise WeightTransferError(
                str(error),
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="upload",
                operation_id=prepared.request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            ) from error
        receipts = tuple(submission.receipts)
        if self.receipt_exchange is not None:
            receipts = tuple(self.receipt_exchange(prepared.upload_plan, receipts))
        receipts = self._validate_materialization_receipts(
            prepared,
            receipts,
        )
        try:
            persisted = self._run_local_commit(
                submission,
                prepared,
                receipts,
                execution_context,
            )
        except _StoreCallInterrupted as error:
            if error.started:
                self._pending_materializations[prepared.request.operation_id] = (
                    submission
                )
                raise WeightTransferCompletionUnknownError(
                    str(error),
                    provider=self.name,
                    phase=error.phase,
                    operation_id=prepared.request.operation_id,
                    completion_ticket=prepared.recovery_ticket,
                ) from error
            terminal_context = _store_terminal_execution_context(
                execution_context,
                include_business_window=True,
            )
            try:
                self._abort_materialization(
                    submission,
                    receipts,
                    terminal_context,
                )
            except BaseException as abort_error:
                self._pending_materializations[prepared.request.operation_id] = (
                    submission
                )
                raise WeightTransferCompletionUnknownError(
                    f"{error}; abort failed: {abort_error}",
                    provider=self.name,
                    phase="abort",
                    operation_id=prepared.request.operation_id,
                    completion_ticket=prepared.recovery_ticket,
                ) from abort_error
            raise WeightTransferError(
                str(error),
                code="DEADLINE_EXCEEDED",
                provider=self.name,
                phase=error.phase,
                operation_id=prepared.request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            ) from error
        except Exception as error:
            return self._resolve_materialization_commit_error(
                submission,
                prepared,
                error,
                execution_context,
            )
        submission.receipts[:] = receipts
        return self._complete_committed_materialization(
            submission,
            prepared,
            persisted,
        )

    def _wait_distributed_materialization(
        self,
        submission: _StoreSubmission,
        prepared: _PreparedStoreMaterialize,
        execution_context: WeightTransferExecutionContext | None,
    ) -> WeightMaterializeReceipt:
        assert self.coordinator is not None
        local_manifests = tuple(
            (placement_id, runtime_manifest)
            for placement_id, runtime_manifest in prepared.runtime_manifests
            if self.local_placement_ids is None
            or placement_id in self.local_placement_ids
        )
        local_error = None
        local_completion_unknown = False
        try:
            submission.receipts[:] = self._run_local_upload(
                submission,
                prepared,
                local_manifests,
                execution_context,
            )
        except _StoreCallInterrupted as error:
            local_error = error
            local_completion_unknown = error.started
        except Exception as error:
            local_error = error

        upload_terminal_context = _store_terminal_execution_context(
            execution_context,
            include_business_window=True,
        )
        outcome = WeightStoreUploadOutcome(
            rank=self.coordinator.rank,
            placement_ids=tuple(placement_id for placement_id, _ in local_manifests),
            receipts=tuple(submission.receipts),
            error=(None if local_error is None else self._error_detail(local_error)),
            completion_unknown=local_completion_unknown,
        )
        try:
            outcomes = self._coordinate(
                "exchange_upload_outcome",
                outcome,
                execution_context=upload_terminal_context,
            )
        except WeightStoreDistributedError as error:
            operation_id = prepared.request.operation_id
            if error.completion_unknown:
                self._pending_materializations[operation_id] = submission
                raise WeightTransferCompletionUnknownError(
                    str(error),
                    provider=self.name,
                    phase="exchange",
                    operation_id=operation_id,
                    completion_ticket=prepared.recovery_ticket,
                ) from error
            try:
                self._abort_materialization(
                    submission,
                    (),
                    upload_terminal_context,
                )
            except Exception as abort_error:
                self._pending_materializations[operation_id] = submission
                cause = abort_error.__cause__ or abort_error
                raise WeightTransferCompletionUnknownError(
                    str(abort_error),
                    provider=self.name,
                    phase="abort",
                    operation_id=operation_id,
                    completion_ticket=prepared.recovery_ticket,
                ) from cause
            raise WeightTransferError(
                str(error),
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="exchange",
                operation_id=operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            ) from error

        root_receipts: tuple[Any, ...] = ()

        def validate_upload() -> None:
            nonlocal root_receipts
            if outcomes is None:
                raise RuntimeError("root upload outcomes are unavailable")
            unknown_outcomes = tuple(
                item for item in outcomes if item.completion_unknown
            )
            if unknown_outcomes:
                error = RuntimeError(
                    "; ".join(
                        f"rank {item.rank}: {item.error}" for item in unknown_outcomes
                    )
                )
                error.completion_unknown = True
                raise error
            root_receipts = self._validate_upload_outcomes(prepared, outcomes)

        try:
            self._coordinate(
                "run_root",
                "validate_upload",
                validate_upload,
                execution_context=upload_terminal_context,
            )
        except WeightStoreDistributedError as error:
            operation_id = prepared.request.operation_id
            if error.completion_unknown:
                self._pending_materializations[operation_id] = submission
                raise WeightTransferCompletionUnknownError(
                    str(error),
                    provider=self.name,
                    phase="upload",
                    operation_id=operation_id,
                    completion_ticket=prepared.recovery_ticket,
                ) from error
            try:
                self._abort_materialization(
                    submission,
                    root_receipts,
                    upload_terminal_context,
                )
            except Exception as abort_error:
                self._pending_materializations[operation_id] = submission
                cause = abort_error.__cause__ or abort_error
                raise WeightTransferCompletionUnknownError(
                    str(abort_error),
                    provider=self.name,
                    phase="abort",
                    operation_id=operation_id,
                    completion_ticket=prepared.recovery_ticket,
                ) from cause
            raise WeightTransferError(
                str(error),
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="upload",
                operation_id=operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            ) from (local_error or error)

        if self.coordinator.rank == 0:
            submission.receipts[:] = root_receipts

        root_persisted_manifest = None
        root_commit_error = None

        def commit_upload() -> (
            _StoreCommitDescriptor | _StoreCommitNotStarted | _StoreCommitUncertain
        ):
            nonlocal root_commit_error, root_persisted_manifest
            try:
                root_persisted_manifest = self._run_local_commit(
                    submission,
                    prepared,
                    root_receipts,
                    execution_context,
                )
                return self._build_commit_descriptor(
                    prepared,
                    root_persisted_manifest,
                )
            except _StoreCallInterrupted as error:
                root_commit_error = error
                if not error.started:
                    return _StoreCommitNotStarted(detail=str(error))
                return _StoreCommitUncertain(
                    detail=self._error_detail(error),
                    observe_manifest=True,
                )
            except BaseException as error:
                root_commit_error = error
                return _StoreCommitUncertain(
                    detail=self._error_detail(error),
                    observe_manifest=root_persisted_manifest is None,
                )

        commit_terminal_context = _store_terminal_execution_context(
            execution_context,
            include_business_window=True,
        )
        try:
            commit_descriptor = self._coordinate(
                "commit_upload",
                commit_upload,
                execution_context=commit_terminal_context,
            )
        except WeightStoreDistributedError as error:
            cause = error.__cause__ or error
            return self._resolve_materialization_commit_error(
                submission,
                prepared,
                cause,
                commit_terminal_context,
            )
        if isinstance(commit_descriptor, _StoreCommitUncertain):
            error = root_commit_error or RuntimeError(commit_descriptor.detail)
            if commit_descriptor.observe_manifest:
                return self._resolve_materialization_commit_error(
                    submission,
                    prepared,
                    error,
                    commit_terminal_context,
                )
            self._pending_materializations[prepared.request.operation_id] = submission
            raise WeightTransferCompletionUnknownError(
                commit_descriptor.detail,
                provider=self.name,
                phase="commit",
                operation_id=prepared.request.operation_id,
                completion_ticket=prepared.recovery_ticket,
            ) from error
        if isinstance(commit_descriptor, _StoreCommitNotStarted):
            try:
                self._abort_materialization(
                    submission,
                    root_receipts,
                    commit_terminal_context,
                )
            except BaseException as abort_error:
                self._pending_materializations[prepared.request.operation_id] = (
                    submission
                )
                raise WeightTransferCompletionUnknownError(
                    f"{commit_descriptor.detail}; abort failed: {abort_error}",
                    provider=self.name,
                    phase="abort",
                    operation_id=prepared.request.operation_id,
                    completion_ticket=prepared.recovery_ticket,
                ) from abort_error
            raise WeightTransferError(
                commit_descriptor.detail,
                code=(
                    "CANCELLED"
                    if execution_context is not None and execution_context.cancelled()
                    else "DEADLINE_EXCEEDED"
                ),
                provider=self.name,
                phase="commit",
                operation_id=prepared.request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            )
        if not isinstance(commit_descriptor, _StoreCommitDescriptor):
            self._pending_materializations[prepared.request.operation_id] = submission
            raise WeightTransferCompletionUnknownError(
                "distributed commit returned an invalid terminal descriptor",
                provider=self.name,
                phase="commit",
                operation_id=prepared.request.operation_id,
                completion_ticket=prepared.recovery_ticket,
            )
        if self.coordinator.rank == 0:
            if root_persisted_manifest is None:
                self._pending_materializations[prepared.request.operation_id] = (
                    submission
                )
                raise WeightTransferCompletionUnknownError(
                    "root commit completed without a persisted manifest",
                    provider=self.name,
                    phase="commit",
                    operation_id=prepared.request.operation_id,
                    completion_ticket=prepared.recovery_ticket,
                )
            self._validate_commit_descriptor(prepared, commit_descriptor)
            return self._complete_committed_materialization(
                submission,
                prepared,
                root_persisted_manifest,
            )
        return self._complete_compact_materialization(
            submission,
            prepared,
            commit_descriptor,
        )

    def wait(
        self,
        submission: _StoreSubmission,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> WeightProviderReceipt:
        self._remember_execution_context(execution_context)
        prepared = submission.prepared
        if isinstance(prepared, _PreparedStoreLoad):
            try:
                self._run_native_call(
                    prepared.request.operation_id,
                    "load",
                    lambda: self.weight_store.load(
                        prepared.load_plan,
                        prepared.target_manifest,
                        pre_registered=self.target_pre_registered,
                    ),
                    execution_context,
                )
            except _StoreCallInterrupted as error:
                if error.started:
                    raise WeightTransferCompletionUnknownError(
                        str(error),
                        provider=self.name,
                        phase=error.phase,
                        operation_id=prepared.request.operation_id,
                    ) from error
                raise WeightTransferError(
                    str(error),
                    code="DEADLINE_EXCEEDED",
                    provider=self.name,
                    phase=error.phase,
                    operation_id=prepared.request.operation_id,
                    retryable=False,
                    completion_known=True,
                    cleanup_required=False,
                ) from error
            except Exception as error:
                raise WeightTransferError(
                    str(error),
                    code="BACKEND_FAILURE",
                    provider=self.name,
                    phase="load",
                    operation_id=prepared.request.operation_id,
                    retryable=False,
                    completion_known=True,
                    cleanup_required=True,
                ) from error
            return WeightLoadReceipt(
                operation_id=prepared.request.operation_id,
                provider=self.name,
                plan_digest=prepared.request.plan.digest,
                total_bytes=prepared.request.plan.total_bytes,
                region_count=len(prepared.request.plan.regions),
            )

        if self.coordinator is None:
            return self._wait_legacy_materialization(
                submission,
                prepared,
                execution_context,
            )
        return self._wait_distributed_materialization(
            submission,
            prepared,
            execution_context,
        )

    def cancel(self, submission: _StoreSubmission) -> None:
        if (
            isinstance(
                submission.prepared,
                _PreparedStoreMaterialize,
            )
            and not submission.committed
            and not submission.aborted
        ):
            self._abort_materialization(
                submission,
                submission.receipts,
            )

    def synchronize(
        self,
        receipt: WeightProviderReceipt,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        self._remember_execution_context(execution_context)
        del receipt

    def release(
        self,
        prepared: _PreparedStoreLoad | _PreparedStoreMaterialize,
        receipt: WeightProviderReceipt | None,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        self._remember_execution_context(execution_context)
        del receipt
        if isinstance(prepared, _PreparedStoreMaterialize):
            finalize = getattr(
                self.weight_store,
                "finalize_upload_session",
                None,
            )
            if callable(finalize):
                operation_id = prepared.request.operation_id

                def finalize_upload() -> None:
                    call = self._finalize_calls.get(operation_id)
                    if (
                        call is not None
                        and call.done.is_set()
                        and call.error is not None
                    ):
                        self._forget_native_call(call)
                        if self._finalize_calls.get(operation_id) is call:
                            self._finalize_calls.pop(operation_id, None)
                        call = None
                    if execution_context is None:
                        if call is None:
                            finalize(prepared.upload_plan)
                        else:
                            call.done.wait()
                            if call.error is not None:
                                raise call.error
                            self._forget_native_call(call)
                            self._finalize_calls.pop(operation_id, None)
                        return
                    if call is None:
                        if execution_context.expired():
                            raise _StoreCallInterrupted(
                                "finalize",
                                started=False,
                            )
                        call = self._get_or_start_native_call(
                            operation_id,
                            "finalize",
                            lambda: finalize(prepared.upload_plan),
                        )
                        self._finalize_calls[operation_id] = call
                    self._await_native_call(
                        call,
                        execution_context,
                    )
                    self._finalize_calls.pop(operation_id, None)

                if self.coordinator is None:
                    finalize_upload()
                else:
                    self._coordinate(
                        "finalize_upload",
                        finalize_upload,
                        execution_context=execution_context,
                    )

    def discard_materialization_recovery(
        self,
        request: WeightMaterializeRequest,
        *,
        completion_ticket: str,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        self._remember_execution_context(execution_context)
        self._cleanup_recovery_ticket_on_root(
            request,
            completion_ticket,
            phase="discard_materialization_recovery",
            execution_context=execution_context,
        )
