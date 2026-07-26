from __future__ import annotations

import base64
import hashlib
import importlib
import json
import zlib
from collections import Counter
from dataclasses import dataclass, field, is_dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
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
    WeightTransferReleaseError,
)
from sglang.srt.weight_transfer.storage import (
    weight_placement_set_digest,
    weight_source_snapshot_digest,
)

_RECOVERY_TICKET_PREFIX = "sglang-mooncake-weight-upload-v1:"
_RECOVERY_RECORD_FORMAT = "sglang-mooncake-weight-upload-recovery"
_MAX_RECOVERY_TICKET_BYTES = 64 * 1024 * 1024
_MAX_RECOVERY_RECORD_BYTES = 256 * 1024 * 1024


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


@dataclass(frozen=True)
class _ChecksummedUploadReceipt:
    fragment_id: str
    object_key: str
    worker_id: str
    checksum: str


@dataclass
class _StoreSubmission:
    prepared: _PreparedStoreLoad | _PreparedStoreMaterialize
    receipts: list[Any] = field(default_factory=list)
    committed: bool = False
    aborted: bool = False


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
        max_total_operations: int = 10_000_000,
    ) -> None:
        if not namespace:
            raise ValueError("Mooncake weight namespace must not be empty")
        if type(max_total_operations) is not int or max_total_operations <= 0:
            raise ValueError("max_total_operations must be a positive integer")
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
        self.max_total_operations = max_total_operations
        self._pending_materializations: dict[str, _StoreSubmission] = {}

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
            supports_safe_cancel=isinstance(
                request,
                WeightMaterializeRequest,
            ),
            supports_completion_ticket=True,
            supports_transactional_publish=True,
            max_regions=1_000_000,
            max_segments_per_region=max_segments,
            max_total_operations=self.max_total_operations,
            max_batch_operations=getattr(
                self.weight_store,
                "max_ranges_per_request",
                1024,
            ),
        )

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

    def _build_recovery_ticket(
        self,
        prepared: _PreparedStoreMaterialize,
        receipts: Sequence[Any] | None = None,
    ) -> str:
        plan = prepared.upload_plan
        manifest_json = plan.manifest.to_json()
        if type(manifest_json) is not str or not manifest_json:
            raise ValueError("Mooncake upload manifest is not serializable")
        operations = []
        for operation in plan.operations:
            source = operation.source
            rank = source.rank
            operations.append(
                {
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
            )
        request = prepared.request
        if request.payload_identity is None:
            raise ValueError("Mooncake recovery requires payload identity")
        record = {
            "format": _RECOVERY_RECORD_FORMAT,
            "version": 1,
            "provider": self.name,
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
            "manifest_json": manifest_json,
            "manifest_digest": hashlib.sha256(
                manifest_json.encode("utf-8")
            ).hexdigest(),
            "session_group_id": plan.session_group_id,
            "control_key": plan.control_key,
            "operations": operations,
            "receipts": (
                [
                    {
                        "fragment_id": operation.target.fragment_id,
                        "object_key": operation.target.object_key,
                        "worker_id": operation.source.worker_id,
                        "checksum": operation.target.checksum,
                    }
                    for operation in plan.operations
                ]
                if receipts is None
                else [
                    {
                        "fragment_id": receipt.fragment_id,
                        "object_key": receipt.object_key,
                        "worker_id": receipt.worker_id,
                        "checksum": receipt.checksum,
                    }
                    for receipt in receipts
                ]
            ),
        }
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
            or record.get("version") != 1
            or record.get("provider") != self.name
        ):
            raise ValueError("unsupported Mooncake recovery ticket")
        return record

    def _validate_recovery_record(
        self,
        request: WeightMaterializeRequest,
        record: dict[str, Any],
    ) -> str:
        destination = record.get("destination")
        payload_identity = request.payload_identity
        if (
            record.get("operation_id") != request.operation_id
            or record.get("placement_digest")
            != weight_placement_set_digest(request.source_placements)
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
        manifest_json = record.get("manifest_json")
        manifest_digest = record.get("manifest_digest")
        if (
            type(manifest_json) is not str
            or type(manifest_digest) is not str
            or hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
            != manifest_digest
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
        if (
            manifest.model_id != request.source_placements[0].model_id
            or manifest.revision != request.source_placements[0].revision
            or manifest.group_id != expected_group
            or manifest.manifest_key != expected_manifest_key
            or request.destination.object_prefix.rstrip("/") != expected_group
            or tuple(manifest.tensors) != self._collect_descriptors(runtime_values)
        ):
            raise ValueError(
                "Mooncake upload plan manifest differs from the materialization request"
            )

        location_by_id = {
            location.fragment_id: location
            for location in request.source_locations
            if isinstance(location, RuntimeWeightLocation)
        }
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
            ):
                raise ValueError(
                    "Mooncake upload plan changed the destination fragment"
                )
            planned_fragments.append(target)

        if Counter(
            self._stored_fragment_signature(fragment) for fragment in manifest.fragments
        ) != Counter(
            self._stored_fragment_signature(fragment) for fragment in planned_fragments
        ):
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

    def _verify_runtime_payload(
        self,
        request: WeightMaterializeRequest,
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
        observed_checksums = {}
        for location in request.source_locations:
            assert isinstance(location, RuntimeWeightLocation)
            if location.placement_id not in local_placement_ids:
                continue
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
                observed_checksums[location.placement_fragment_id] = verifier(location)
            except Exception as error:
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

    def _cleanup_failed_preflight(
        self,
        request: WeightMaterializeRequest,
        upload_plan: Any,
        preflight_errors: Sequence[str],
        *,
        completion_ticket: str,
    ) -> None:
        cleanup_errors = []

        def abort_upload() -> None:
            self.weight_store.abort_upload(upload_plan, ())

        try:
            if self.coordinator is None:
                abort_upload()
            else:
                self.coordinator.abort_upload(abort_upload)
        except Exception as error:
            cleanup_errors.append(self._error_detail(error))

        finalize = getattr(
            self.weight_store,
            "finalize_upload_session",
            None,
        )
        if callable(finalize):
            try:
                if self.coordinator is None:
                    finalize(upload_plan)
                else:
                    self.coordinator.finalize_upload(lambda: finalize(upload_plan))
            except Exception as error:
                cleanup_errors.append(self._error_detail(error))

        if cleanup_errors:
            raise WeightTransferCompletionUnknownError(
                "; ".join((*preflight_errors, *cleanup_errors)),
                provider=self.name,
                phase="preflight",
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            )

    def _prepare_materialize(
        self,
        backend: Any,
        request: WeightMaterializeRequest,
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
        self._verify_runtime_payload(request)
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
            return self._attach_payload_identity(
                request,
                self.weight_store.prepare_upload(
                    tuple(manifest for _, manifest in runtime_manifests),
                    namespace=self.namespace,
                ),
            )

        upload_plan = (
            prepare_upload()
            if self.coordinator is None
            else self.coordinator.prepare_upload(prepare_upload)
        )
        prepared = _PreparedStoreMaterialize(
            request=request,
            upload_plan=upload_plan,
            runtime_manifests=runtime_manifests,
        )
        prepared = replace(
            prepared,
            recovery_ticket=self._build_recovery_ticket(prepared),
        )
        assert prepared.recovery_ticket is not None
        local_error = None
        try:
            self._validate_upload_plan(
                request,
                upload_plan,
                runtime_manifests,
            )
        except Exception as error:
            local_error = error

        if self.coordinator is None:
            preflight_errors = (
                () if local_error is None else (self._error_detail(local_error),)
            )
        else:
            outcome = WeightStorePreflightOutcome(
                rank=self.coordinator.rank,
                error=(
                    None if local_error is None else self._error_detail(local_error)
                ),
            )
            try:
                outcomes = self.coordinator.exchange_preflight_outcome(outcome)
            except WeightStoreDistributedError as error:
                self._cleanup_failed_preflight(
                    request,
                    upload_plan,
                    (self._error_detail(error),),
                    completion_ticket=prepared.recovery_ticket,
                )
                raise ValueError(str(error)) from error
            preflight_errors = tuple(
                outcome.error for outcome in outcomes if outcome.error is not None
            )
        if preflight_errors:
            self._cleanup_failed_preflight(
                request,
                upload_plan,
                preflight_errors,
                completion_ticket=prepared.recovery_ticket,
            )
            raise ValueError("; ".join(preflight_errors)) from local_error

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

    def _abort_incomplete_recovery(
        self,
        prepared: _PreparedStoreMaterialize,
        receipts: Sequence[Any],
        existing_keys: frozenset[str],
        *,
        completion_ticket: str,
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
                abort_upload()
            else:
                self.coordinator.abort_upload(abort_upload)
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
            self.release(prepared, None)
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
    ) -> None:
        cleanup_errors = []

        def abort_upload() -> None:
            self.weight_store.abort_upload(
                prepared.upload_plan,
                tuple(receipts),
            )

        try:
            if self.coordinator is None:
                abort_upload()
            else:
                self.coordinator.abort_upload(abort_upload)
        except Exception as error:
            cleanup_errors.append(error)

        try:
            self.release(prepared, None)
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
            self.release(prepared, receipt)
        except Exception as error:
            raise WeightTransferReleaseError(
                str(error),
                receipt=receipt,
            ) from error
        return receipt

    def recover_materialization(
        self,
        request: WeightMaterializeRequest,
        *,
        completion_ticket: str | None = None,
    ) -> WeightMaterializeReceipt | None:
        backend = self._load_backend()
        if request.profile != "runtime_to_storage":
            raise ValueError("Mooncake recovery requires a runtime-to-storage request")
        if completion_ticket is None:
            return None
        try:
            record = self._decode_recovery_ticket(completion_ticket)
            manifest_json = self._validate_recovery_record(request, record)
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

        manifest_key = f"{request.destination.object_prefix.rstrip('/')}/manifest"
        try:
            manifest = self._load_manifest_if_present(
                backend,
                manifest_key,
            )
        except Exception as error:
            raise WeightTransferCompletionUnknownError(
                f"manifest observation failed during recovery: {error}",
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            ) from error
        if manifest is not None:
            persisted_json = manifest.to_json()
            if persisted_json != manifest_json:
                try:
                    upload_plan, receipts = self._reconstruct_recovery_plan(
                        backend,
                        request,
                        record,
                    )
                except Exception as error:
                    raise WeightTransferCompletionUnknownError(
                        "conflicting weight revision loser plan could not be "
                        f"reconstructed: {error}",
                        provider=self.name,
                        phase="recover",
                        operation_id=request.operation_id,
                        completion_ticket=completion_ticket,
                    ) from error
                prepared = _PreparedStoreMaterialize(
                    request=request,
                    upload_plan=upload_plan,
                    runtime_manifests=(),
                    recovery_ticket=completion_ticket,
                )
                self._raise_recovery_conflict(
                    prepared,
                    receipts,
                    completion_ticket=completion_ticket,
                )
            prepared = _PreparedStoreMaterialize(
                request=request,
                upload_plan=SimpleNamespace(
                    manifest=manifest,
                    session_group_id=record["session_group_id"],
                    control_key=record["control_key"],
                    operations=(),
                ),
                runtime_manifests=(),
                recovery_ticket=completion_ticket,
            )
            return self._complete_recovered_materialization(
                prepared,
                manifest,
                completion_ticket=completion_ticket,
            )

        try:
            upload_plan, receipts = self._reconstruct_recovery_plan(
                backend,
                request,
                record,
            )
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
        prepared = _PreparedStoreMaterialize(
            request=request,
            upload_plan=upload_plan,
            runtime_manifests=(),
            recovery_ticket=completion_ticket,
        )
        if manifest is None:
            try:
                existing_keys = self._existing_payload_keys(
                    backend,
                    upload_plan,
                )
            except Exception as error:
                raise WeightTransferCompletionUnknownError(
                    f"payload observation failed during recovery: {error}",
                    provider=self.name,
                    phase="recover",
                    operation_id=request.operation_id,
                    completion_ticket=completion_ticket,
                ) from error
            expected_keys = frozenset(
                operation.target.object_key for operation in upload_plan.operations
            )
            if existing_keys != expected_keys:
                self._abort_incomplete_recovery(
                    prepared,
                    receipts,
                    existing_keys,
                    completion_ticket=completion_ticket,
                )

            def commit_upload() -> Any:
                return self.weight_store.commit(upload_plan, receipts)

            try:
                manifest = (
                    commit_upload()
                    if self.coordinator is None
                    else self.coordinator.commit_upload(commit_upload)
                )
            except Exception as error:
                try:
                    persisted = self._load_manifest_if_present(
                        backend,
                        upload_plan.manifest.manifest_key,
                    )
                except Exception as observation_error:
                    raise WeightTransferCompletionUnknownError(
                        f"{error}; manifest observation failed: {observation_error}",
                        provider=self.name,
                        phase="recover",
                        operation_id=request.operation_id,
                        completion_ticket=completion_ticket,
                    ) from error
                if persisted is not None and persisted != upload_plan.manifest:
                    self._raise_recovery_conflict(
                        prepared,
                        receipts,
                        completion_ticket=completion_ticket,
                        cause=error,
                    )
                if persisted == upload_plan.manifest:
                    manifest = persisted
                else:
                    raise WeightTransferCompletionUnknownError(
                        str(error),
                        provider=self.name,
                        phase="recover",
                        operation_id=request.operation_id,
                        completion_ticket=completion_ticket,
                    ) from error
        return self._complete_recovered_materialization(
            prepared,
            manifest,
            completion_ticket=completion_ticket,
        )

    def _prepare_load(
        self,
        backend: Any,
        request: WeightLoadRequest,
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
        manifest = self.weight_store.load_manifest(f"{storage_id}/manifest")
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
    ) -> _PreparedStoreLoad | _PreparedStoreMaterialize:
        backend = self._load_backend()
        if isinstance(request, WeightLoadRequest):
            return self._prepare_load(backend, request)
        return self._prepare_materialize(backend, request)

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

    def _materialize_receipt(
        self,
        prepared: _PreparedStoreMaterialize,
        manifest: Any,
        *,
        completion_ticket: str | None = None,
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
            if location is None or geometry in seen:
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
            abort_upload()
        else:
            self.coordinator.abort_upload(abort_upload)
        submission.aborted = True

    def _resolve_materialization_commit_error(
        self,
        submission: _StoreSubmission,
        prepared: _PreparedStoreMaterialize,
        error: BaseException,
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
            persisted = self._load_manifest_if_present(
                backend,
                prepared.upload_plan.manifest.manifest_key,
            )
        except Exception as observation_error:
            self._pending_materializations[prepared.request.operation_id] = submission
            raise WeightTransferCompletionUnknownError(
                f"{error}; manifest observation failed: {observation_error}",
                provider=self.name,
                phase="commit",
                operation_id=prepared.request.operation_id,
                completion_ticket=completion_ticket,
            ) from error
        if persisted == prepared.upload_plan.manifest:
            submission.committed = True
            self._validate_committed_manifest(prepared, persisted)
            return self._materialize_receipt(
                prepared,
                persisted,
                completion_ticket=completion_ticket,
            )
        if persisted is not None:
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
        raise WeightTransferCompletionUnknownError(
            str(error),
            provider=self.name,
            phase="commit",
            operation_id=prepared.request.operation_id,
            completion_ticket=completion_ticket,
        ) from error

    def _wait_legacy_materialization(
        self,
        submission: _StoreSubmission,
        prepared: _PreparedStoreMaterialize,
    ) -> WeightMaterializeReceipt:
        for placement_id, runtime_manifest in prepared.runtime_manifests:
            if (
                self.local_placement_ids is not None
                and placement_id not in self.local_placement_ids
            ):
                continue
            submission.receipts.extend(
                self._attach_local_receipt_checksums(
                    prepared,
                    self.weight_store.upload(
                        prepared.upload_plan,
                        runtime_manifest,
                        pre_registered=self.source_pre_registered,
                    ),
                )
            )
        receipts = tuple(submission.receipts)
        if self.receipt_exchange is not None:
            receipts = tuple(self.receipt_exchange(prepared.upload_plan, receipts))
        receipts = self._validate_materialization_receipts(
            prepared,
            receipts,
        )
        try:
            persisted = self.weight_store.commit(
                prepared.upload_plan,
                receipts,
            )
        except Exception as error:
            return self._resolve_materialization_commit_error(
                submission,
                prepared,
                error,
            )
        submission.receipts[:] = receipts
        submission.committed = True
        self._validate_committed_manifest(prepared, persisted)
        return self._materialize_receipt(
            prepared,
            persisted,
            completion_ticket=prepared.recovery_ticket,
        )

    def _wait_distributed_materialization(
        self,
        submission: _StoreSubmission,
        prepared: _PreparedStoreMaterialize,
    ) -> WeightMaterializeReceipt:
        assert self.coordinator is not None
        local_manifests = tuple(
            (placement_id, runtime_manifest)
            for placement_id, runtime_manifest in prepared.runtime_manifests
            if self.local_placement_ids is None
            or placement_id in self.local_placement_ids
        )
        local_error = None
        try:
            for _, runtime_manifest in local_manifests:
                submission.receipts.extend(
                    self._attach_local_receipt_checksums(
                        prepared,
                        self.weight_store.upload(
                            prepared.upload_plan,
                            runtime_manifest,
                            pre_registered=self.source_pre_registered,
                        ),
                    )
                )
        except Exception as error:
            local_error = error

        outcome = WeightStoreUploadOutcome(
            rank=self.coordinator.rank,
            placement_ids=tuple(placement_id for placement_id, _ in local_manifests),
            receipts=tuple(submission.receipts),
            error=(None if local_error is None else self._error_detail(local_error)),
        )
        try:
            outcomes = self.coordinator.exchange_upload_outcome(outcome)
        except WeightStoreDistributedError as error:
            operation_id = prepared.request.operation_id
            self._pending_materializations[operation_id] = submission
            raise WeightTransferCompletionUnknownError(
                str(error),
                provider=self.name,
                phase="exchange",
                operation_id=operation_id,
                completion_ticket=prepared.recovery_ticket,
            ) from error

        validation_error = None
        receipts: tuple[Any, ...] = ()
        try:
            receipts = self._validate_upload_outcomes(
                prepared,
                outcomes,
            )
        except Exception as error:
            validation_error = error
        if validation_error is not None:
            submission.receipts[:] = receipts or [
                receipt for gathered in outcomes for receipt in gathered.receipts
            ]
            try:
                self._abort_materialization(
                    submission,
                    submission.receipts,
                )
            except Exception as abort_error:
                operation_id = prepared.request.operation_id
                self._pending_materializations[operation_id] = submission
                cause = abort_error.__cause__ or abort_error
                raise WeightTransferCompletionUnknownError(
                    str(abort_error),
                    provider=self.name,
                    phase="abort",
                    operation_id=operation_id,
                    completion_ticket=prepared.recovery_ticket,
                ) from cause
            cause = local_error or validation_error
            raise WeightTransferError(
                str(validation_error),
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="wait",
                operation_id=prepared.request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            ) from cause

        submission.receipts[:] = receipts

        def commit_upload() -> Any:
            return self.weight_store.commit(
                prepared.upload_plan,
                receipts,
            )

        try:
            persisted = self.coordinator.commit_upload(commit_upload)
        except WeightStoreDistributedError as error:
            cause = error.__cause__ or error
            return self._resolve_materialization_commit_error(
                submission,
                prepared,
                cause,
            )
        submission.committed = True
        self._validate_committed_manifest(prepared, persisted)
        return self._materialize_receipt(
            prepared,
            persisted,
            completion_ticket=prepared.recovery_ticket,
        )

    def wait(self, submission: _StoreSubmission) -> WeightProviderReceipt:
        prepared = submission.prepared
        if isinstance(prepared, _PreparedStoreLoad):
            try:
                self.weight_store.load(
                    prepared.load_plan,
                    prepared.target_manifest,
                    pre_registered=self.target_pre_registered,
                )
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
            )
        return self._wait_distributed_materialization(
            submission,
            prepared,
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

    def synchronize(self, receipt: WeightProviderReceipt) -> None:
        del receipt

    def release(
        self,
        prepared: _PreparedStoreLoad | _PreparedStoreMaterialize,
        receipt: WeightProviderReceipt | None,
    ) -> None:
        del receipt
        if isinstance(prepared, _PreparedStoreMaterialize):
            finalize = getattr(
                self.weight_store,
                "finalize_upload_session",
                None,
            )
            if callable(finalize):
                if self.coordinator is None:
                    finalize(prepared.upload_plan)
                else:
                    self.coordinator.finalize_upload(
                        lambda: finalize(prepared.upload_plan)
                    )
