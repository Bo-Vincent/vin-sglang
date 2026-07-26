from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Sequence

from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightRuntimeBindingManifest,
)
from sglang.srt.weight_transfer.contracts import RuntimeWeightLocation
from sglang.srt.weight_transfer.provider import (
    WeightLoadReceipt,
    WeightLoadRequest,
    WeightProviderCapabilities,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
)


class _MooncakeBackendView:
    def __init__(self, module: Any, runtime_lease_snapshot: Any) -> None:
        self._module = module
        self.RuntimeLeaseSnapshot = runtime_lease_snapshot

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)


class MooncakeWeightTransferCompletionUnknownError(
    WeightTransferCompletionUnknownError
):
    def __init__(
        self,
        message: str,
        *,
        operation_id: str,
        pending_transfer_id: str,
    ) -> None:
        super().__init__(
            message,
            provider="mooncake-te",
            phase="wait",
            operation_id=operation_id,
            completion_ticket=pending_transfer_id,
        )
        self.pending_transfer_id = pending_transfer_id


@dataclass(frozen=True)
class _PreparedMooncakeLoad:
    request: WeightLoadRequest
    executable_plan: Any
    source_manifests: tuple[Any, ...]
    target_manifest: Any
    source_registrations: tuple[Any, ...] | None
    target_registrations: tuple[Any, ...] | None


class MooncakeWeightTransferProvider:
    """Optional adapter from SGLang bound regions to Mooncake TE reads."""

    name = "mooncake-te"
    requires_runtime_attestation = True

    def __init__(
        self,
        transfer_engine: Any,
        *,
        source_pre_registered: bool = True,
        source_registrations: Sequence[Any] | None = None,
        target_pre_registered: bool = True,
        target_registrations: Sequence[Any] | None = None,
        max_batch_operations: int = 8192,
        max_region_segments: int = 1_000_000,
        max_total_operations: int = 10_000_000,
    ) -> None:
        if (
            max_batch_operations <= 0
            or max_region_segments <= 0
            or max_total_operations <= 0
        ):
            raise ValueError("Mooncake lowering limits must be positive")
        self.transfer_engine = transfer_engine
        self.source_pre_registered = source_pre_registered
        self.source_registrations = (
            tuple(source_registrations) if source_registrations is not None else None
        )
        self.target_pre_registered = target_pre_registered
        self.target_registrations = (
            tuple(target_registrations) if target_registrations is not None else None
        )
        self.max_batch_operations = max_batch_operations
        self.max_region_segments = max_region_segments
        self.max_total_operations = max_total_operations
        self._reader = None

    @staticmethod
    def _load_backend() -> Any:
        try:
            weight_transfer = importlib.import_module("mooncake.weight_transfer")
        except Exception as error:
            raise WeightTransferError(
                "Mooncake weight-transfer support is unavailable",
                code="UNAVAILABLE_PROVIDER",
                provider="mooncake-te",
                phase="probe",
                operation_id="unbound",
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            ) from error
        required = (
            "ExecutorTransferPlan",
            "ParallelRank",
            "PipelineRouteGroup",
            "RuntimeFragment",
            "RuntimeManifest",
            "TensorDescriptor",
            "TransferPlan",
            "TransferRegion",
            "MooncakeTransferEngineReader",
            "MemoryRegistrationLease",
            "TransferCompletionUnknownError",
            "TransferEngineError",
        )
        missing = [name for name in required if not hasattr(weight_transfer, name)]
        if missing:
            raise WeightTransferError(
                "Mooncake provider is missing APIs: " + ", ".join(missing),
                code="UNAVAILABLE_PROVIDER",
                provider="mooncake-te",
                phase="probe",
                operation_id="unbound",
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            )
        runtime_lease_snapshot = getattr(
            weight_transfer,
            "RuntimeLeaseSnapshot",
            None,
        )
        if runtime_lease_snapshot is None:
            try:
                planner = importlib.import_module("mooncake.weight_transfer.planner")
                runtime_lease_snapshot = planner.RuntimeLeaseSnapshot
            except Exception as error:
                raise WeightTransferError(
                    "Mooncake provider is missing API: RuntimeLeaseSnapshot",
                    code="UNAVAILABLE_PROVIDER",
                    provider="mooncake-te",
                    phase="probe",
                    operation_id="unbound",
                    retryable=False,
                    completion_known=True,
                    cleanup_required=False,
                ) from error
        return _MooncakeBackendView(
            weight_transfer,
            runtime_lease_snapshot,
        )

    def probe(self, request: WeightLoadRequest) -> WeightProviderCapabilities:
        self._load_backend()
        return WeightProviderCapabilities(
            provider=self.name,
            load_profiles=frozenset({"runtime_to_runtime"}),
            materialize_profiles=frozenset(),
            supports_nd_regions=True,
            supports_strided_regions=True,
            supports_safe_cancel=False,
            supports_completion_ticket=True,
            supports_transactional_publish=False,
            max_regions=1_000_000,
            max_segments_per_region=self.max_region_segments,
            max_total_operations=self.max_total_operations,
            max_batch_operations=self.max_batch_operations,
        )

    @staticmethod
    def _parallel_rank(backend: Any, rank: Any) -> Any:
        return backend.ParallelRank(
            dp=rank.dp,
            tp=rank.tp,
            pp=rank.pp,
            ep=rank.ep,
        )

    @classmethod
    def _runtime_manifest(
        cls,
        backend: Any,
        placement: Any,
        binding: WeightRuntimeBindingManifest,
    ) -> Any:
        if (
            placement.model_id != binding.model_id
            or placement.revision != binding.revision
            or placement.placement_id != binding.placement_id
        ):
            raise ValueError("Mooncake runtime manifest identity mismatch")
        binding_fragments = {
            fragment.placement_fragment_id: fragment for fragment in binding.fragments
        }
        if len(binding_fragments) != len(binding.fragments):
            raise ValueError("Mooncake runtime binding has duplicate fragments")
        placement_fragments = {
            tensor.placement_fragment_id: tensor for tensor in placement.tensors
        }
        if set(binding_fragments) != set(placement_fragments):
            raise ValueError("Mooncake runtime binding fragments differ")

        descriptors = {}
        fragments = []
        for placement_fragment_id, tensor in placement_fragments.items():
            shard_dims = tuple(tensor.shard_dims)
            if not shard_dims and tensor.partition_dim is not None:
                shard_dims = (tensor.partition_dim,)
            descriptor = backend.TensorDescriptor(
                tensor_id=tensor.tensor_id,
                global_shape=tuple(tensor.global_shape),
                dtype=tensor.dtype,
                itemsize=tensor.itemsize,
                partition_dim=tensor.partition_dim,
                layer_id=tensor.layer_id,
                expert_id=tensor.expert_id,
                layout_fingerprint=tensor.layout_fingerprint,
                shard_dims=shard_dims,
            )
            previous = descriptors.setdefault(tensor.tensor_id, descriptor)
            if previous != descriptor:
                raise ValueError(
                    f"Mooncake runtime tensor descriptor mismatch: {tensor.tensor_id}"
                )
            bound = binding_fragments[placement_fragment_id]
            fragments.append(
                backend.RuntimeFragment(
                    fragment_id=bound.fragment_id,
                    tensor_id=tensor.tensor_id,
                    global_offset=tuple(tensor.global_offset),
                    local_shape=tuple(tensor.local_shape),
                    address=bound.address,
                    nbytes=bound.nbytes,
                    worker_id=bound.worker_id,
                    endpoint=bound.endpoint,
                    rank=cls._parallel_rank(backend, tensor.rank),
                    lease_generation=binding.generation,
                    aliases=tuple(tensor.aliases),
                    placement_fragment_id=placement_fragment_id,
                )
            )
        return backend.RuntimeManifest(
            model_id=placement.model_id,
            revision=placement.revision,
            instance_id=binding.instance_id,
            tensors=tuple(descriptors[tensor_id] for tensor_id in sorted(descriptors)),
            fragments=tuple(
                sorted(fragments, key=lambda fragment: fragment.fragment_id)
            ),
            lease_id=binding.lease_id,
            format_version=2,
            placement_id=placement.placement_id,
        )

    @classmethod
    def _executor_plans(
        cls,
        backend: Any,
        groups: Sequence[Any],
        manifests: Sequence[Any],
        logical_to_bound: dict[int, int],
    ) -> tuple[Any, ...]:
        manifests_by_placement = {
            manifest.placement_id: manifest for manifest in manifests
        }
        result = []
        for group in groups:
            operation_indices = tuple(
                logical_to_bound[index]
                for index in group.region_indices
                if index in logical_to_bound
            )
            if not operation_indices:
                continue
            manifest = manifests_by_placement.get(group.placement_id)
            if manifest is None:
                raise ValueError("Mooncake executor placement is missing")
            fragments_by_placement = {
                fragment.placement_fragment_id: fragment
                for fragment in manifest.fragments
            }
            try:
                referenced_fragments = tuple(
                    fragments_by_placement[placement_fragment_id]
                    for placement_fragment_id in group.placement_fragment_ids
                )
            except KeyError as error:
                raise ValueError("Mooncake executor fragment is missing") from error
            rank = cls._parallel_rank(backend, group.rank)
            if any(fragment.rank != rank for fragment in referenced_fragments):
                raise ValueError("Mooncake executor fragment rank differs")
            fragments = tuple(
                sorted(
                    manifest.fragments,
                    key=lambda fragment: fragment.fragment_id,
                )
            )
            if any(fragment.rank != rank for fragment in fragments):
                raise ValueError("Mooncake runtime manifest mixes executor ranks")
            worker_ids = {fragment.worker_id for fragment in fragments}
            if len(worker_ids) != 1:
                raise ValueError("Mooncake executor spans multiple workers")
            result.append(
                backend.ExecutorTransferPlan(
                    instance_id=manifest.instance_id,
                    runtime_lease_id=manifest.lease_id,
                    worker_id=next(iter(worker_ids)),
                    rank=cls._parallel_rank(backend, group.rank),
                    fragment_ids=tuple(fragment.fragment_id for fragment in fragments),
                    fragment_leases=tuple(
                        backend.RuntimeLeaseSnapshot.from_fragment(fragment)
                        for fragment in fragments
                    ),
                    operation_indices=operation_indices,
                )
            )
        return tuple(result)

    @staticmethod
    def _bound_region_index_map(request: WeightLoadRequest) -> dict[int, int]:
        logical_regions = request.plan.logical_plan.regions
        logical_indices = {
            region: index for index, region in enumerate(logical_regions)
        }
        if len(logical_indices) != len(logical_regions):
            raise ValueError("Mooncake logical plan has duplicate regions")
        try:
            result = {
                logical_indices[region.logical_region]: bound_index
                for bound_index, region in enumerate(request.plan.regions)
            }
        except KeyError as error:
            raise ValueError(
                "Mooncake bound region is absent from the logical plan"
            ) from error
        if len(result) != len(request.plan.regions):
            raise ValueError("Mooncake bound region projection is ambiguous")
        return result

    @staticmethod
    def _pipeline_routes(
        backend: Any,
        routes: Sequence[Any],
        logical_to_bound: dict[int, int],
    ) -> tuple[Any, ...]:
        result = []
        for route in routes:
            operation_indices = tuple(
                logical_to_bound[index]
                for index in route.region_indices
                if index in logical_to_bound
            )
            if not operation_indices:
                continue
            result.append(
                backend.PipelineRouteGroup(
                    source_pp=route.source_pp,
                    target_pp=route.target_pp,
                    operation_indices=operation_indices,
                )
            )
        return tuple(result)

    @staticmethod
    def _registration_leases(
        backend: Any,
        bindings: Sequence[WeightRuntimeBindingManifest],
    ) -> tuple[Any, ...]:
        registrations = []
        fragment_ids = set()
        for binding in bindings:
            for fragment in binding.fragments:
                if fragment.fragment_id in fragment_ids:
                    raise ValueError(
                        "Mooncake runtime binding has duplicate fragment IDs"
                    )
                fragment_ids.add(fragment.fragment_id)
                registrations.append(
                    backend.MemoryRegistrationLease(
                        fragment_id=fragment.fragment_id,
                        worker_id=fragment.worker_id,
                        address=fragment.address,
                        nbytes=fragment.nbytes,
                        lease_generation=binding.generation,
                        runtime_lease_id=binding.lease_id,
                    )
                )
        return tuple(
            sorted(
                registrations,
                key=lambda item: (item.worker_id, item.fragment_id),
            )
        )

    def prepare(self, request: WeightLoadRequest) -> _PreparedMooncakeLoad:
        if request.profile != "runtime_to_runtime" or any(
            not isinstance(region.source, RuntimeWeightLocation)
            for region in request.plan.regions
        ):
            raise WeightTransferError(
                "Mooncake TE provider requires runtime sources",
                code="UNSUPPORTED_CAPABILITY",
                provider=self.name,
                phase="prepare",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            )
        backend = self._load_backend()
        logical = request.plan.logical_plan
        native_source_bindings = tuple(
            binding
            for binding in request.plan.source_bindings
            if isinstance(binding, WeightRuntimeBindingManifest)
        )
        if len(native_source_bindings) != len(request.plan.source_bindings):
            raise ValueError("Mooncake TE provider requires runtime source bindings")
        native_target_bindings = tuple(request.plan.target_bindings)

        source_bindings_by_id = {
            binding.placement_id: binding for binding in native_source_bindings
        }
        target_bindings_by_id = {
            binding.placement_id: binding for binding in native_target_bindings
        }
        source_manifests = tuple(
            self._runtime_manifest(
                backend,
                placement,
                source_bindings_by_id[placement.placement_id],
            )
            for placement in logical.source_placements
        )
        target_manifests = tuple(
            self._runtime_manifest(
                backend,
                placement,
                target_bindings_by_id[placement.placement_id],
            )
            for placement in logical.target_placements
        )
        logical_to_bound = self._bound_region_index_map(request)
        source_executors = self._executor_plans(
            backend,
            logical.source_executors,
            source_manifests,
            logical_to_bound,
        )
        target_executors = self._executor_plans(
            backend,
            logical.target_executors,
            target_manifests,
            logical_to_bound,
        )
        used_source_instances = {executor.instance_id for executor in source_executors}
        source_manifests = tuple(
            manifest
            for manifest in source_manifests
            if manifest.instance_id in used_source_instances
        )
        used_source_placements = {
            manifest.placement_id for manifest in source_manifests
        }
        source_registrations = self.source_registrations
        if self.source_pre_registered and source_registrations is None:
            source_registrations = self._registration_leases(
                backend,
                tuple(
                    binding
                    for binding in native_source_bindings
                    if binding.placement_id in used_source_placements
                ),
            )
        target_registrations = self.target_registrations
        if self.target_pre_registered and target_registrations is None:
            target_registrations = self._registration_leases(
                backend,
                native_target_bindings,
            )
        source_fragments = {
            (manifest.placement_id, fragment.placement_fragment_id): fragment
            for manifest in source_manifests
            for fragment in manifest.fragments
        }
        target_fragments = {
            (manifest.placement_id, fragment.placement_fragment_id): fragment
            for manifest in target_manifests
            for fragment in manifest.fragments
        }
        operations = tuple(
            backend.TransferRegion(
                tensor_id=region.tensor_id,
                source=source_fragments[
                    (
                        region.source.placement_id,
                        region.source.placement_fragment_id,
                    )
                ],
                target=target_fragments[
                    (
                        region.target.placement_id,
                        region.target.placement_fragment_id,
                    )
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
            for region in request.plan.regions
        )
        routes = self._pipeline_routes(
            backend,
            logical.pipeline_routes,
            logical_to_bound,
        )
        executable_plan = backend.TransferPlan(
            model_id=logical.model_id,
            revision=logical.revision,
            operations=operations,
            source_executors=source_executors,
            target_executors=target_executors,
            pipeline_routes=routes,
        )
        if len(target_manifests) != 1:
            raise WeightTransferError(
                "Mooncake reader requires one local target manifest",
                code="UNSUPPORTED_CAPABILITY",
                provider=self.name,
                phase="prepare",
                operation_id=request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=False,
            )
        self._reader = backend.MooncakeTransferEngineReader(
            self.transfer_engine,
            max_batch_operations=self.max_batch_operations,
            max_region_segments=self.max_region_segments,
        )
        return _PreparedMooncakeLoad(
            request=request,
            executable_plan=executable_plan,
            source_manifests=source_manifests,
            target_manifest=target_manifests[0],
            source_registrations=source_registrations,
            target_registrations=target_registrations,
        )

    def submit(self, prepared: _PreparedMooncakeLoad) -> _PreparedMooncakeLoad:
        return prepared

    def wait(self, submission: _PreparedMooncakeLoad) -> WeightLoadReceipt:
        backend = self._load_backend()
        assert self._reader is not None
        try:
            receipts = self._reader.execute(
                submission.executable_plan,
                submission.source_manifests,
                submission.target_manifest,
                source_pre_registered=self.source_pre_registered,
                source_registrations=submission.source_registrations,
                target_pre_registered=self.target_pre_registered,
                target_registrations=submission.target_registrations,
            )
        except backend.TransferCompletionUnknownError as error:
            raise MooncakeWeightTransferCompletionUnknownError(
                str(error),
                operation_id=submission.request.operation_id,
                pending_transfer_id=error.pending_transfer_id,
            ) from error
        except backend.TransferEngineError as error:
            raise WeightTransferError(
                str(error),
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="wait",
                operation_id=submission.request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            ) from error
        transferred_bytes = sum(receipt.nbytes for receipt in receipts)
        if transferred_bytes != submission.request.plan.total_bytes:
            raise WeightTransferError(
                "Mooncake receipt byte count does not match the bound plan",
                code="RECEIPT_MISMATCH",
                provider=self.name,
                phase="wait",
                operation_id=submission.request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            )
        return WeightLoadReceipt(
            operation_id=submission.request.operation_id,
            provider=self.name,
            plan_digest=submission.request.plan.digest,
            total_bytes=transferred_bytes,
            region_count=len(submission.request.plan.regions),
            backend_receipts=tuple(receipts),
        )

    def cancel(self, submission: _PreparedMooncakeLoad) -> None:
        del submission

    def synchronize(self, receipt: WeightLoadReceipt) -> None:
        del receipt

    def release(
        self,
        prepared: _PreparedMooncakeLoad,
        receipt: WeightLoadReceipt | None,
    ) -> None:
        del prepared, receipt

    def drain_pending_transfer(
        self,
        pending_transfer_id: str,
        *,
        timeout_ms: int,
    ) -> str:
        if self._reader is None:
            raise RuntimeError("Mooncake reader has not been prepared")
        return self._reader.drain_pending_transfer(
            pending_transfer_id,
            timeout_ms=timeout_ms,
        )

    def drain_completion(
        self,
        completion_ticket: str,
        *,
        timeout_ms: int,
    ) -> str:
        return self.drain_pending_transfer(
            completion_ticket,
            timeout_ms=timeout_ms,
        )
