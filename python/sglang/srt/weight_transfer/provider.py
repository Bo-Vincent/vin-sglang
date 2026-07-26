from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from sglang.srt.weight_transfer.contracts import (
    BoundWeightTransferPlan,
    RuntimeWeightLocation,
    SourceBindingManifest,
    StorageWeightLocation,
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
)
from sglang.srt.weight_transfer.lowering import (
    WeightLoweringLimits,
    iter_bounded_transfer_batches,
)


@dataclass(frozen=True)
class WeightProviderCapabilities:
    provider: str
    load_profiles: frozenset[str]
    materialize_profiles: frozenset[str]
    supports_nd_regions: bool
    supports_strided_regions: bool
    supports_safe_cancel: bool
    supports_completion_ticket: bool
    supports_transactional_publish: bool
    max_regions: int | None = None
    max_segments_per_region: int | None = None
    max_total_operations: int | None = None
    max_batch_operations: int | None = None
    max_batch_bytes: int | None = None
    max_total_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider name must not be empty")
        for name in (
            "max_regions",
            "max_segments_per_region",
            "max_total_operations",
            "max_batch_operations",
            "max_batch_bytes",
            "max_total_bytes",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")


class WeightTransferError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        provider: str,
        phase: str,
        operation_id: str,
        retryable: bool,
        completion_known: bool,
        cleanup_required: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.provider = provider
        self.phase = phase
        self.operation_id = operation_id
        self.retryable = retryable
        self.completion_known = completion_known
        self.cleanup_required = cleanup_required


class WeightTransferCompletionUnknownError(WeightTransferError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        phase: str,
        operation_id: str,
        completion_ticket: str | None = None,
    ) -> None:
        super().__init__(
            message,
            code="COMPLETION_UNKNOWN",
            provider=provider,
            phase=phase,
            operation_id=operation_id,
            retryable=False,
            completion_known=False,
            cleanup_required=True,
        )
        if completion_ticket is not None and (
            type(completion_ticket) is not str or not completion_ticket
        ):
            raise ValueError("completion ticket must be a non-empty string")
        self.completion_ticket = completion_ticket


class WeightTransferReleaseError(WeightTransferError):
    def __init__(
        self,
        message: str,
        *,
        receipt: WeightProviderReceipt,
    ) -> None:
        super().__init__(
            message,
            code="RELEASE_FAILED",
            provider=receipt.provider,
            phase="release",
            operation_id=receipt.operation_id,
            retryable=True,
            completion_known=True,
            cleanup_required=True,
        )
        self.receipt = receipt
        self.publication = None


@dataclass(frozen=True)
class WeightStorageDestination:
    provider: str
    storage_id: str
    object_prefix: str

    def __post_init__(self) -> None:
        if not self.provider or not self.storage_id or not self.object_prefix:
            raise ValueError("storage destination identifiers must not be empty")


@dataclass(frozen=True)
class WeightLoadRequest:
    operation_id: str
    plan: BoundWeightTransferPlan
    profile: str


def _require_sha256_digest(value: object, name: str) -> str:
    if type(value) is not str or not value.startswith("sha256:"):
        raise ValueError(f"{name} must be a canonical sha256 digest")
    payload = value.removeprefix("sha256:")
    if (
        len(payload) != 64
        or payload != payload.lower()
        or any(character not in "0123456789abcdef" for character in payload)
    ):
        raise ValueError(f"{name} must be a canonical sha256 digest")
    return value


@dataclass(frozen=True)
class WeightPayloadFragmentIdentity:
    placement_fragment_id: str
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    nbytes: int
    checksum: str

    def __post_init__(self) -> None:
        if not self.placement_fragment_id or not self.tensor_id:
            raise ValueError("payload fragment identifiers must not be empty")
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("payload fragment nbytes must be positive")
        if (
            not self.global_offset
            or len(self.global_offset) != len(self.local_shape)
            or any(type(value) is not int or value < 0 for value in self.global_offset)
            or any(type(value) is not int or value <= 0 for value in self.local_shape)
        ):
            raise ValueError("payload fragment geometry is invalid")
        _require_sha256_digest(self.checksum, "payload fragment checksum")


@dataclass(frozen=True)
class WeightPayloadIdentity:
    fragments: tuple[WeightPayloadFragmentIdentity, ...]
    payload_digest: str

    def __post_init__(self) -> None:
        fragments = tuple(self.fragments)
        fragment_ids = [fragment.placement_fragment_id for fragment in fragments]
        if (
            not fragments
            or not all(
                isinstance(fragment, WeightPayloadFragmentIdentity)
                for fragment in fragments
            )
            or len(fragment_ids) != len(set(fragment_ids))
        ):
            raise ValueError("payload identity fragments are invalid")
        object.__setattr__(
            self,
            "fragments",
            tuple(sorted(fragments, key=lambda item: item.placement_fragment_id)),
        )
        _require_sha256_digest(self.payload_digest, "payload_digest")
        if self.payload_digest != self._compute_digest(self.fragments):
            raise ValueError("payload_digest is not canonical")

    @staticmethod
    def _compute_digest(
        fragments: Sequence[WeightPayloadFragmentIdentity],
    ) -> str:
        payload = json.dumps(
            {
                "format": "sglang-weight-payload-v1",
                "fragments": [
                    {
                        "placement_fragment_id": fragment.placement_fragment_id,
                        "tensor_id": fragment.tensor_id,
                        "global_offset": fragment.global_offset,
                        "local_shape": fragment.local_shape,
                        "nbytes": fragment.nbytes,
                        "checksum": fragment.checksum,
                    }
                    for fragment in sorted(
                        fragments,
                        key=lambda item: item.placement_fragment_id,
                    )
                ],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    @classmethod
    def create(
        cls,
        placements: Sequence[WeightPlacementManifest],
        checksums: Mapping[str, str],
    ) -> WeightPayloadIdentity:
        tensors = [tensor for placement in placements for tensor in placement.tensors]
        fragment_ids = {tensor.placement_fragment_id for tensor in tensors}
        if fragment_ids != set(checksums):
            raise ValueError("payload checksums differ from placement fragments")
        fragments = tuple(
            WeightPayloadFragmentIdentity(
                placement_fragment_id=tensor.placement_fragment_id,
                tensor_id=tensor.tensor_id,
                global_offset=tuple(tensor.global_offset),
                local_shape=tuple(tensor.local_shape),
                nbytes=tensor.nbytes,
                checksum=checksums[tensor.placement_fragment_id],
            )
            for tensor in tensors
        )
        return cls(
            fragments=fragments,
            payload_digest=cls._compute_digest(fragments),
        )

    def select(
        self,
        placements: Sequence[WeightPlacementManifest],
    ) -> WeightPayloadIdentity:
        checksum_by_id = {
            fragment.placement_fragment_id: fragment.checksum
            for fragment in self.fragments
        }
        selected_ids = {
            tensor.placement_fragment_id
            for placement in placements
            for tensor in placement.tensors
        }
        if not selected_ids.issubset(checksum_by_id):
            raise ValueError("payload identity does not cover selected placements")
        return self.create(
            placements,
            {fragment_id: checksum_by_id[fragment_id] for fragment_id in selected_ids},
        )


@dataclass(frozen=True)
class WeightMaterializeRequest:
    operation_id: str
    source_placements: tuple[WeightPlacementManifest, ...]
    source_bindings: tuple[SourceBindingManifest, ...]
    source_locations: tuple[RuntimeWeightLocation | StorageWeightLocation, ...]
    destination: WeightStorageDestination
    profile: str
    payload_identity: WeightPayloadIdentity | None = None

    def __post_init__(self) -> None:
        if self.payload_identity is not None:
            if not isinstance(self.payload_identity, WeightPayloadIdentity):
                raise ValueError("payload_identity is invalid")
            if self.payload_identity.select(self.source_placements) != (
                self.payload_identity
            ):
                raise ValueError(
                    "payload identity differs from materialization placements"
                )

    @property
    def total_bytes(self) -> int:
        return sum(location.nbytes for location in self.source_locations)


WeightProviderRequest = WeightLoadRequest | WeightMaterializeRequest


class WeightTransferAttestor(Protocol):
    def attest(self, request: WeightProviderRequest) -> None: ...


@dataclass(frozen=True)
class WeightLoadReceipt:
    operation_id: str
    provider: str
    plan_digest: str
    total_bytes: int
    region_count: int
    backend_receipts: tuple[Any, ...] = ()
    provider_phase_seconds: tuple[tuple[str, float], ...] = ()


class WeightTargetLoadState(str, Enum):
    CREATED = "created"
    PREPARING = "preparing"
    LOADING = "loading"
    TRANSFERRED = "transferred"
    READY = "ready"
    ACTIVE = "active"
    ABORTED = "aborted"
    POISONED = "poisoned"
    QUARANTINED = "quarantined"


class WeightTargetLoadMode(str, Enum):
    COLD_START = "cold_start"
    LIVE_UPDATE = "live_update"


class WeightTransferTerminalStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class WeightTransferTerminalProof:
    operation_id: str
    provider: str
    completion_ticket: str
    status: WeightTransferTerminalStatus

    def __post_init__(self) -> None:
        if not self.operation_id or not self.provider or not self.completion_ticket:
            raise ValueError("terminal proof identifiers must not be empty")
        if not isinstance(self.status, WeightTransferTerminalStatus):
            raise ValueError("terminal proof status is invalid")


class WeightTargetActivationController(Protocol):
    @property
    def generation(self) -> int: ...

    def begin_target_update(
        self,
        binding: WeightRuntimeBindingManifest,
        *,
        full_restore: bool,
    ) -> str: ...

    def cancel_update(self, token: str) -> None: ...

    def finish_update(self, token: str, *, success: bool) -> int: ...

    def poison_global_update_failure(
        self,
        *,
        expected_generation: int,
    ) -> None: ...

    def commit_revision(self, *, expected_generation: int | None = None) -> int: ...


@dataclass
class WeightTargetLoadSession:
    """Own target buffers until a load is proven ready or terminally failed."""

    target_bindings: tuple[WeightRuntimeBindingManifest, ...]
    owners: tuple[Any, ...]
    coordinator: WeightTargetActivationController
    state: WeightTargetLoadState = WeightTargetLoadState.CREATED
    operation_id: str | None = None
    receipt: WeightLoadReceipt | None = None
    failure: BaseException | None = None
    completion_ticket: str | None = None
    update_token: str | None = None
    pending_generation: int | None = None
    full_restore: bool = False
    request: WeightLoadRequest | None = None
    provider_name: str | None = None

    def __init__(
        self,
        *,
        target_bindings: Sequence[WeightRuntimeBindingManifest],
        owners: Sequence[Any],
        coordinator: WeightTargetActivationController,
        full_restore: bool = False,
    ) -> None:
        bindings = tuple(target_bindings)
        retained_owners = tuple(owners)
        if not bindings or not all(
            isinstance(binding, WeightRuntimeBindingManifest) for binding in bindings
        ):
            raise ValueError("target load session bindings are invalid")
        if not retained_owners or any(owner is None for owner in retained_owners):
            raise ValueError("target load session owners must not be empty")
        if len(bindings) != 1:
            raise ValueError("target load session requires one local target binding")
        if not isinstance(full_restore, bool):
            raise TypeError("full_restore must be a boolean")
        required_controller_methods = (
            "begin_target_update",
            "cancel_update",
            "finish_update",
            "poison_global_update_failure",
            "commit_revision",
        )
        if coordinator is None or any(
            not callable(getattr(coordinator, name, None))
            for name in required_controller_methods
        ):
            raise ValueError("target load session coordinator is invalid")
        self.target_bindings = bindings
        self.owners = retained_owners
        self.coordinator = coordinator
        self.state = WeightTargetLoadState.CREATED
        self.operation_id = None
        self.receipt = None
        self.failure = None
        self.completion_ticket = None
        self.update_token = None
        self.pending_generation = None
        self.full_restore = full_restore
        self.request = None
        self.provider_name = None

    def begin(self, request: WeightLoadRequest, *, provider_name: str) -> None:
        if self.state is not WeightTargetLoadState.CREATED:
            raise ValueError("target load session has already started")
        if type(provider_name) is not str or not provider_name:
            raise ValueError("target load session provider must not be empty")
        if tuple(request.plan.target_bindings) != self.target_bindings:
            raise ValueError("target load session bindings differ from the request")
        self.update_token = self.coordinator.begin_target_update(
            self.target_bindings[0],
            full_restore=self.full_restore,
        )
        self.operation_id = request.operation_id
        self.request = request
        self.provider_name = provider_name
        self.state = WeightTargetLoadState.PREPARING

    def mark_mutating(self) -> None:
        if (
            self.state is not WeightTargetLoadState.PREPARING
            or self.update_token is None
        ):
            raise ValueError("target load session cannot start mutation")
        self.state = WeightTargetLoadState.LOADING

    def complete_transfer(self, receipt: WeightLoadReceipt) -> None:
        request = self.request
        if (
            self.state is not WeightTargetLoadState.LOADING
            or request is None
            or receipt.operation_id != self.operation_id
            or receipt.provider != self.provider_name
            or receipt.plan_digest != request.plan.digest
            or receipt.total_bytes != request.plan.total_bytes
            or receipt.region_count != len(request.plan.regions)
        ):
            raise ValueError("target load session completion is invalid")
        self.receipt = receipt
        self.state = WeightTargetLoadState.TRANSFERRED

    def mark_ready(self) -> int:
        if (
            self.state is not WeightTargetLoadState.TRANSFERRED
            or self.update_token is None
        ):
            raise ValueError("target load session is not transfer-complete")
        try:
            self.pending_generation = self.coordinator.finish_update(
                self.update_token,
                success=True,
            )
        except BaseException as error:
            self.failure = error
            self.update_token = None
            self.pending_generation = self.coordinator.generation
            self.state = WeightTargetLoadState.POISONED
            raise
        self.update_token = None
        self.state = WeightTargetLoadState.READY
        return self.pending_generation

    def activate(self) -> int:
        if (
            self.state is not WeightTargetLoadState.READY
            or self.pending_generation is None
        ):
            raise ValueError("target load session is not ready for activation")
        generation = self.coordinator.commit_revision(
            expected_generation=self.pending_generation,
        )
        self.state = WeightTargetLoadState.ACTIVE
        return generation

    def fail(self, error: BaseException) -> None:
        if self.state is WeightTargetLoadState.POISONED:
            if self.failure is None:
                self.failure = error
            return
        if self.state is WeightTargetLoadState.PREPARING:
            if self.update_token is None:
                raise ValueError("target load session update token is missing")
            self.coordinator.cancel_update(self.update_token)
            self.update_token = None
            self.failure = error
            self.state = WeightTargetLoadState.ABORTED
            return
        if self.state is WeightTargetLoadState.ACTIVE:
            self.failure = error
            if self.pending_generation is None:
                raise ValueError("active target generation is missing")
            self.coordinator.poison_global_update_failure(
                expected_generation=self.pending_generation,
            )
            self.state = WeightTargetLoadState.POISONED
            return
        if self.state not in (
            WeightTargetLoadState.LOADING,
            WeightTargetLoadState.TRANSFERRED,
            WeightTargetLoadState.READY,
        ):
            raise ValueError("target load session failure is invalid")
        self.failure = error
        completion_known = not (
            isinstance(error, WeightTransferError) and not error.completion_known
        )
        self.completion_ticket = getattr(error, "completion_ticket", None)
        if not completion_known:
            if self.state is WeightTargetLoadState.READY:
                raise ValueError("ready target cannot have unknown transfer completion")
            self.state = WeightTargetLoadState.QUARANTINED
            return
        if self.state is WeightTargetLoadState.READY:
            assert self.pending_generation is not None
            self.coordinator.poison_global_update_failure(
                expected_generation=self.pending_generation,
            )
        else:
            if self.update_token is None:
                raise ValueError("target load session update token is missing")
            self.pending_generation = self.coordinator.finish_update(
                self.update_token,
                success=False,
            )
            self.update_token = None
        self.state = WeightTargetLoadState.POISONED

    def resolve_quarantine(self, proof: WeightTransferTerminalProof) -> int:
        if self.state is not WeightTargetLoadState.QUARANTINED:
            raise ValueError("target load session is not quarantined")
        if not isinstance(proof, WeightTransferTerminalProof):
            raise ValueError("quarantine requires a terminal proof")
        if (
            proof.operation_id != self.operation_id
            or proof.provider != self.provider_name
            or proof.completion_ticket != self.completion_ticket
        ):
            raise ValueError("terminal proof differs from the quarantined transfer")
        if self.update_token is None:
            raise ValueError("target load session update token is missing")
        self.pending_generation = self.coordinator.finish_update(
            self.update_token,
            success=False,
        )
        self.update_token = None
        self.state = WeightTargetLoadState.POISONED
        return self.pending_generation

    def require_ready(self) -> WeightLoadReceipt:
        if self.state is not WeightTargetLoadState.ACTIVE or self.receipt is None:
            raise RuntimeError(f"target load session is not ready: {self.state.value}")
        return self.receipt


@dataclass(frozen=True)
class WeightMaterializeReceipt:
    operation_id: str
    provider: str
    manifest_key: str
    stored_placements: tuple[WeightPlacementManifest, ...]
    storage_bindings: tuple[WeightStorageBindingManifest, ...]
    total_bytes: int
    fragment_count: int
    completion_ticket: str | None = None
    provider_phase_seconds: tuple[tuple[str, float], ...] = ()


WeightProviderReceipt = WeightLoadReceipt | WeightMaterializeReceipt


class WeightTransferProvider(Protocol):
    name: str
    requires_runtime_attestation: bool

    def probe(self, request: WeightProviderRequest) -> WeightProviderCapabilities: ...

    def prepare(self, request: WeightProviderRequest) -> Any: ...

    def submit(self, prepared: Any) -> Any: ...

    def wait(self, submission: Any) -> WeightProviderReceipt: ...

    def cancel(self, submission: Any) -> None: ...

    def synchronize(self, receipt: WeightProviderReceipt) -> None: ...

    def release(
        self,
        prepared: Any,
        receipt: WeightProviderReceipt | None,
    ) -> None: ...


class LocalWeightBufferRegistry:
    """Reference address/object registry used by deterministic CPU tests."""

    def __init__(self) -> None:
        self.runtime_buffers: dict[int, bytearray] = {}
        self.storage_objects: dict[str, bytearray] = {}

    def register_runtime(
        self,
        address: int,
        buffer: bytearray | bytes | memoryview,
    ) -> None:
        if type(address) is not int or address <= 0:
            raise ValueError("runtime address must be positive")
        payload = buffer if isinstance(buffer, bytearray) else bytearray(buffer)
        end = address + len(payload)
        for base, current in self.runtime_buffers.items():
            if address < base + len(current) and base < end:
                raise ValueError("runtime buffer registration overlaps")
        self.runtime_buffers[address] = payload

    def _runtime_slice(
        self,
        address: int,
        nbytes: int,
    ) -> tuple[bytearray, int]:
        for base, buffer in self.runtime_buffers.items():
            offset = address - base
            if offset >= 0 and offset + nbytes <= len(buffer):
                return buffer, offset
        raise ValueError("runtime range is not registered")

    def read_runtime(self, address: int, nbytes: int) -> bytes:
        buffer, offset = self._runtime_slice(address, nbytes)
        return bytes(buffer[offset : offset + nbytes])

    def write_runtime(self, address: int, payload: bytes) -> None:
        buffer, offset = self._runtime_slice(address, len(payload))
        buffer[offset : offset + len(payload)] = payload

    def read_storage(self, object_key: str, offset: int, nbytes: int) -> bytes:
        payload = self.storage_objects.get(object_key)
        if payload is None or offset < 0 or offset + nbytes > len(payload):
            raise ValueError("storage range is unavailable")
        return bytes(payload[offset : offset + nbytes])

    def publish_storage(self, objects: dict[str, bytearray]) -> None:
        if set(objects).intersection(self.storage_objects):
            raise ValueError("storage object already exists")
        self.storage_objects.update(objects)


@dataclass
class _LocalSubmission:
    request: WeightProviderRequest
    receipt: WeightProviderReceipt | None = None


class LocalWeightTransferProvider:
    """Backend-neutral reference executor, not a production transport."""

    name = "local"
    requires_runtime_attestation = False

    def __init__(
        self,
        registry: LocalWeightBufferRegistry,
        *,
        lowering_limits: WeightLoweringLimits | None = None,
    ) -> None:
        self.registry = registry
        self.lowering_limits = lowering_limits or WeightLoweringLimits(
            max_total_operations=10_000_000,
            max_batch_operations=8192,
            max_batch_bytes=256 * 1024 * 1024,
        )

    def probe(self, request: WeightProviderRequest) -> WeightProviderCapabilities:
        del request
        return WeightProviderCapabilities(
            provider=self.name,
            load_profiles=frozenset({"runtime_to_runtime", "storage_to_runtime"}),
            materialize_profiles=frozenset(
                {"runtime_to_storage", "storage_to_storage"}
            ),
            supports_nd_regions=True,
            supports_strided_regions=True,
            supports_safe_cancel=True,
            supports_completion_ticket=False,
            supports_transactional_publish=True,
            max_regions=1_000_000,
            max_segments_per_region=10_000_000,
            max_total_operations=self.lowering_limits.max_total_operations,
            max_batch_operations=self.lowering_limits.max_batch_operations,
            max_batch_bytes=self.lowering_limits.max_batch_bytes,
        )

    def prepare(self, request: WeightProviderRequest) -> WeightProviderRequest:
        return request

    def submit(self, prepared: WeightProviderRequest) -> _LocalSubmission:
        submission = _LocalSubmission(request=prepared)
        if isinstance(prepared, WeightLoadRequest):
            submission.receipt = self._execute_load(prepared)
        else:
            submission.receipt = self._execute_materialize(prepared)
        return submission

    def wait(self, submission: _LocalSubmission) -> WeightProviderReceipt:
        if submission.receipt is None:
            raise WeightTransferCompletionUnknownError(
                "local submission has no receipt",
                provider=self.name,
                phase="wait",
                operation_id=submission.request.operation_id,
            )
        return submission.receipt

    def cancel(self, submission: _LocalSubmission) -> None:
        del submission

    def synchronize(self, receipt: WeightProviderReceipt) -> None:
        del receipt

    def release(
        self,
        prepared: WeightProviderRequest,
        receipt: WeightProviderReceipt | None,
    ) -> None:
        del prepared, receipt

    def _read_source(
        self,
        source: RuntimeWeightLocation | StorageWeightLocation,
        offset: int,
        nbytes: int,
    ) -> bytes:
        if isinstance(source, RuntimeWeightLocation):
            return self.registry.read_runtime(source.address + offset, nbytes)
        return self.registry.read_storage(
            source.object_key,
            source.object_offset + offset,
            nbytes,
        )

    def _execute_load(self, request: WeightLoadRequest) -> WeightLoadReceipt:
        for batch in iter_bounded_transfer_batches(
            request.plan,
            self.lowering_limits,
        ):
            for operation in batch.operations:
                payload = self._read_source(
                    operation.source,
                    operation.source_offset,
                    operation.nbytes,
                )
                self.registry.write_runtime(
                    operation.target.address + operation.target_offset,
                    payload,
                )
        return WeightLoadReceipt(
            operation_id=request.operation_id,
            provider=self.name,
            plan_digest=request.plan.digest,
            total_bytes=request.plan.total_bytes,
            region_count=len(request.plan.regions),
        )

    def _execute_materialize(
        self,
        request: WeightMaterializeRequest,
    ) -> WeightMaterializeReceipt:
        staged_objects, receipt = self._stage_materialization(request)
        self.registry.publish_storage(staged_objects)
        return receipt

    def recover_materialization(
        self,
        request: WeightMaterializeRequest,
        *,
        completion_ticket: str | None = None,
    ) -> WeightMaterializeReceipt | None:
        del completion_ticket
        staged_objects, receipt = self._stage_materialization(request)
        existing_keys = set(staged_objects).intersection(self.registry.storage_objects)
        if not existing_keys:
            return None
        if existing_keys != set(staged_objects):
            raise ValueError(
                "local materialization recovery found partial storage objects"
            )
        for key, expected in staged_objects.items():
            if self.registry.storage_objects[key] != expected:
                raise ValueError(
                    "local materialization recovery found conflicting payload"
                )
        return receipt

    def _stage_materialization(
        self,
        request: WeightMaterializeRequest,
    ) -> tuple[dict[str, bytearray], WeightMaterializeReceipt]:
        staged_objects: dict[str, bytearray] = {}
        fragments_by_placement: dict[
            str,
            list[WeightStorageFragmentBinding],
        ] = {}
        for location in request.source_locations:
            object_name = hashlib.sha256(
                location.placement_fragment_id.encode()
            ).hexdigest()[:24]
            object_key = (
                f"{request.destination.object_prefix.rstrip('/')}/{object_name}"
            )
            payload = self._read_source(location, 0, location.nbytes)
            staged_objects[object_key] = bytearray(payload)
            fragments_by_placement.setdefault(
                location.placement_id,
                [],
            ).append(
                WeightStorageFragmentBinding(
                    placement_fragment_id=location.placement_fragment_id,
                    fragment_id=f"stored:{location.fragment_id}",
                    object_key=object_key,
                    object_offset=0,
                    nbytes=location.nbytes,
                    checksum=f"sha256:{hashlib.sha256(payload).hexdigest()}",
                )
            )

        placement_by_id = {
            placement.placement_id: placement for placement in request.source_placements
        }
        storage_bindings = tuple(
            WeightStorageBindingManifest(
                model_id=placement_by_id[placement_id].model_id,
                revision=placement_by_id[placement_id].revision,
                placement_id=placement_id,
                storage_id=request.destination.storage_id,
                provider=request.destination.provider,
                fragments=tuple(
                    sorted(
                        fragments,
                        key=lambda item: item.placement_fragment_id,
                    )
                ),
            )
            for placement_id, fragments in sorted(fragments_by_placement.items())
        )
        return (
            staged_objects,
            WeightMaterializeReceipt(
                operation_id=request.operation_id,
                provider=self.name,
                manifest_key=(
                    f"{request.destination.object_prefix.rstrip('/')}/manifest"
                ),
                stored_placements=request.source_placements,
                storage_bindings=storage_bindings,
                total_bytes=request.total_bytes,
                fragment_count=len(request.source_locations),
            ),
        )


def new_operation_id() -> str:
    return uuid4().hex
