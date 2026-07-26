from __future__ import annotations

import hashlib
from typing import Sequence

from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightTensor,
    RuntimeWeightBinding,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    WeightRuntimeManifest,
    WeightRuntimeManifestParts,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.contracts import (
    BoundWeightTransferPlan,
    BoundWeightTransferRegion,
    LogicalWeightTransferPlan,
    RuntimeWeightLocation,
    SourceBindingManifest,
    StorageWeightLocation,
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
    _storage_payload_slice_identity,
)


def _legacy_placement_fragment_id(tensor: RuntimeWeightTensor) -> str:
    shard_dims = tuple(tensor.shard_dims)
    if not shard_dims and tensor.partition_dim is not None:
        shard_dims = (tensor.partition_dim,)
    identity = (
        "weight-placement-v1",
        tensor.tensor_id,
        tuple(tensor.global_shape),
        tuple(tensor.global_offset),
        tuple(tensor.local_shape),
        tensor.partition_dim,
        shard_dims,
        tensor.byte_offset,
        tensor.layer_id,
        tensor.expert_id,
        tensor.layout_fingerprint,
        tuple(tensor.aliases),
        tensor.dtype,
        tensor.itemsize,
        tensor.rank.dp,
        tensor.rank.tp,
        tensor.rank.pp,
        tensor.rank.ep,
    )
    return hashlib.sha256(repr(identity).encode()).hexdigest()[:24]


def runtime_manifest_to_parts(
    manifest: WeightRuntimeManifest,
) -> WeightRuntimeManifestParts:
    """Adapt the combined runtime manifest to placement plus binding."""

    placement_fragment_ids = tuple(
        _legacy_placement_fragment_id(tensor) for tensor in manifest.tensors
    )
    if len(placement_fragment_ids) != len(set(placement_fragment_ids)):
        raise ValueError("legacy manifest has duplicate placement semantics")
    placement_tensors = tuple(
        WeightPlacementTensor(
            placement_fragment_id=placement_fragment_id,
            tensor_id=tensor.tensor_id,
            runtime_name=tensor.runtime_name,
            aliases=tuple(tensor.aliases),
            global_shape=tuple(tensor.global_shape),
            global_offset=tuple(tensor.global_offset),
            local_shape=tuple(tensor.local_shape),
            dtype=tensor.dtype,
            itemsize=tensor.itemsize,
            partition_dim=tensor.partition_dim,
            shard_dims=(
                tuple(tensor.shard_dims)
                if tensor.shard_dims
                else (
                    (tensor.partition_dim,) if tensor.partition_dim is not None else ()
                )
            ),
            layer_id=tensor.layer_id,
            expert_id=tensor.expert_id,
            layout_fingerprint=tensor.layout_fingerprint,
            nbytes=tensor.nbytes,
            byte_offset=tensor.byte_offset,
            rank=tensor.rank,
        )
        for tensor, placement_fragment_id in zip(
            manifest.tensors,
            placement_fragment_ids,
            strict=True,
        )
    )
    placement_id = compute_weight_placement_id(placement_tensors)
    placement = WeightPlacementManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=placement_id,
        tensors=placement_tensors,
    )
    binding = WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=placement_id,
        instance_id=manifest.instance_id,
        generation=manifest.generation,
        lease_id=manifest.lease_id,
        fragments=tuple(
            RuntimeWeightBinding(
                placement_fragment_id=placement_fragment_id,
                fragment_id=tensor.fragment_id,
                address=tensor.address,
                nbytes=tensor.nbytes,
                storage_offset=tensor.storage_offset,
                device=tensor.device,
                is_contiguous=tensor.is_contiguous,
                worker_id=tensor.worker_id,
                endpoint=tensor.endpoint,
            )
            for tensor, placement_fragment_id in zip(
                manifest.tensors,
                placement_fragment_ids,
                strict=True,
            )
        ),
    )
    return WeightRuntimeManifestParts(placement=placement, binding=binding)


def _placements_by_id(
    placements: Sequence[WeightPlacementManifest],
    label: str,
) -> dict[str, WeightPlacementManifest]:
    result = {}
    for placement in placements:
        if placement.placement_id in result:
            raise ValueError(f"duplicate {label} placement ID")
        result[placement.placement_id] = placement
    return result


def _validate_binding_identity(
    placement: WeightPlacementManifest,
    binding: SourceBindingManifest,
    label: str,
) -> None:
    if placement.model_id != binding.model_id or placement.revision != binding.revision:
        raise ValueError(f"{label} binding model identity differs")
    if placement.placement_id != binding.placement_id:
        raise ValueError(f"{label} placement IDs differ")


def _runtime_locations(
    placements: Sequence[WeightPlacementManifest],
    bindings: Sequence[WeightRuntimeBindingManifest],
    label: str,
) -> dict[str, RuntimeWeightLocation]:
    placement_by_id = _placements_by_id(placements, label)
    binding_by_id = {}
    for binding in bindings:
        if binding.placement_id in binding_by_id:
            raise ValueError(f"duplicate {label} runtime binding")
        binding_by_id[binding.placement_id] = binding
    if set(placement_by_id) != set(binding_by_id):
        raise ValueError(f"logical plan and {label} placement IDs differ")

    result = {}
    for placement_id, placement in placement_by_id.items():
        binding = binding_by_id[placement_id]
        _validate_binding_identity(placement, binding, label)
        placement_fragments = {
            tensor.placement_fragment_id: tensor for tensor in placement.tensors
        }
        binding_fragments = {
            fragment.placement_fragment_id: fragment for fragment in binding.fragments
        }
        if len(binding_fragments) != len(binding.fragments) or set(
            placement_fragments
        ) != set(binding_fragments):
            raise ValueError(f"{label} binding fragment IDs differ")
        for fragment_id, tensor in placement_fragments.items():
            fragment = binding_fragments[fragment_id]
            if fragment.nbytes != tensor.nbytes:
                raise ValueError(f"{label} binding byte size differs")
            if (
                fragment.address <= 0
                or fragment.storage_offset < 0
                or not fragment.is_contiguous
                or not fragment.worker_id
                or not fragment.endpoint
                or not fragment.device
            ):
                raise ValueError(f"{label} runtime binding is invalid")
            if fragment_id in result:
                raise ValueError(f"duplicate bound {label} fragment")
            result[fragment_id] = RuntimeWeightLocation(
                placement_id=placement_id,
                placement_fragment_id=fragment_id,
                fragment_id=fragment.fragment_id,
                tensor_id=tensor.tensor_id,
                address=fragment.address,
                nbytes=fragment.nbytes,
                storage_offset=fragment.storage_offset,
                device=fragment.device,
                worker_id=fragment.worker_id,
                endpoint=fragment.endpoint,
                generation=binding.generation,
                lease_id=binding.lease_id,
                rank=tensor.rank,
                global_offset=tuple(tensor.global_offset),
                local_shape=tuple(tensor.local_shape),
                aliases=tuple(tensor.aliases),
            )
    return result


def _storage_locations(
    placements: Sequence[WeightPlacementManifest],
    bindings: Sequence[WeightStorageBindingManifest],
    label: str,
) -> dict[str, StorageWeightLocation]:
    placement_by_id = _placements_by_id(placements, label)
    binding_by_id = {}
    for binding in bindings:
        if binding.placement_id in binding_by_id:
            raise ValueError(f"duplicate {label} storage binding")
        binding_by_id[binding.placement_id] = binding
    if set(placement_by_id) != set(binding_by_id):
        raise ValueError(f"logical plan and {label} placement IDs differ")

    result = {}
    for placement_id, placement in placement_by_id.items():
        binding = binding_by_id[placement_id]
        _validate_binding_identity(placement, binding, label)
        placement_fragments = {
            tensor.placement_fragment_id: tensor for tensor in placement.tensors
        }
        binding_fragments = {
            fragment.placement_fragment_id: fragment for fragment in binding.fragments
        }
        if len(binding_fragments) != len(binding.fragments) or set(
            placement_fragments
        ) != set(binding_fragments):
            raise ValueError(f"{label} binding fragment IDs differ")
        for fragment_id, tensor in placement_fragments.items():
            fragment = binding_fragments[fragment_id]
            if fragment.nbytes != tensor.nbytes:
                raise ValueError(f"{label} binding byte size differs")
            if fragment_id in result:
                raise ValueError(f"duplicate bound {label} fragment")
            result[fragment_id] = StorageWeightLocation(
                placement_id=placement_id,
                placement_fragment_id=fragment_id,
                fragment_id=fragment.fragment_id,
                tensor_id=tensor.tensor_id,
                provider=binding.provider,
                storage_id=binding.storage_id,
                object_key=fragment.object_key,
                object_offset=fragment.object_offset,
                nbytes=fragment.nbytes,
                checksum=fragment.checksum,
                rank=tensor.rank,
                global_offset=tuple(tensor.global_offset),
                local_shape=tuple(tensor.local_shape),
                aliases=tuple(tensor.aliases),
            )
    return result


def _runtime_address_space(location: RuntimeWeightLocation) -> tuple[str, str, str]:
    return (
        location.worker_id,
        location.endpoint,
        location.device,
    )


def _validate_target_allocations(
    locations: Sequence[RuntimeWeightLocation],
) -> None:
    by_address_space: dict[
        tuple[str, str, str],
        list[RuntimeWeightLocation],
    ] = {}
    for location in locations:
        by_address_space.setdefault(_runtime_address_space(location), []).append(
            location
        )
    for scoped in by_address_space.values():
        ordered = sorted(scoped, key=lambda item: (item.address, item.nbytes))
        for index, left in enumerate(ordered):
            left_end = left.address + left.nbytes
            for right in ordered[index + 1 :]:
                if right.address >= left_end:
                    break
                same_alias = (
                    left.address == right.address
                    and left.nbytes == right.nbytes
                    and len(left.aliases) > 1
                    and left.aliases == right.aliases
                    and left.global_offset == right.global_offset
                    and left.local_shape == right.local_shape
                )
                if not same_alias:
                    raise ValueError("overlapping target physical allocations")


def _validate_runtime_source_target_allocations(
    source_locations: Sequence[RuntimeWeightLocation],
    target_locations: Sequence[RuntimeWeightLocation],
) -> None:
    sources_by_address_space: dict[
        tuple[str, str, str],
        list[RuntimeWeightLocation],
    ] = {}
    targets_by_address_space: dict[
        tuple[str, str, str],
        list[RuntimeWeightLocation],
    ] = {}
    for location in source_locations:
        sources_by_address_space.setdefault(
            _runtime_address_space(location),
            [],
        ).append(location)
    for location in target_locations:
        targets_by_address_space.setdefault(
            _runtime_address_space(location),
            [],
        ).append(location)

    for address_space in sources_by_address_space.keys() & targets_by_address_space:
        sources = sorted(
            sources_by_address_space[address_space],
            key=lambda item: (item.address, item.nbytes),
        )
        targets = sorted(
            targets_by_address_space[address_space],
            key=lambda item: (item.address, item.nbytes),
        )
        source_index = 0
        target_index = 0
        while source_index < len(sources) and target_index < len(targets):
            source = sources[source_index]
            target = targets[target_index]
            source_end = source.address + source.nbytes
            target_end = target.address + target.nbytes
            if source.address < target_end and target.address < source_end:
                raise ValueError(
                    "overlapping runtime source and target physical allocations"
                )
            if source_end <= target_end:
                source_index += 1
            else:
                target_index += 1


def _source_identity(region: BoundWeightTransferRegion) -> tuple:
    source = region.source
    if isinstance(source, RuntimeWeightLocation):
        location = (
            "runtime",
            source.worker_id,
            source.endpoint,
            source.generation,
            source.address + region.source_base_offset,
        )
    else:
        location = (
            "storage",
            source.provider,
            source.object_key,
            source.object_offset + region.source_base_offset,
        )
    return (
        location,
        region.inner_bytes,
        region.outer_loop_counts,
        region.source_strides,
    )


def _storage_payload_identity(
    region: BoundWeightTransferRegion,
) -> tuple | None:
    source = region.source
    if not isinstance(source, StorageWeightLocation):
        return None
    return _storage_payload_slice_identity(
        checksum=source.checksum,
        nbytes=source.nbytes,
        source_base_offset=region.source_base_offset,
        inner_bytes=region.inner_bytes,
        outer_loop_counts=region.outer_loop_counts,
        source_strides=region.source_strides,
    )


def _sources_have_identical_payload(
    left: BoundWeightTransferRegion,
    right: BoundWeightTransferRegion,
) -> bool:
    if _source_identity(left) == _source_identity(right):
        return True
    left_storage = _storage_payload_identity(left)
    return left_storage is not None and left_storage == _storage_payload_identity(right)


def _target_identity(region: BoundWeightTransferRegion) -> tuple:
    return (
        region.target.worker_id,
        region.target.endpoint,
        region.target.address + region.target_base_offset,
        region.inner_bytes,
        region.outer_loop_counts,
        region.target_strides,
    )


def _deduplicate_aliases(
    regions: Sequence[BoundWeightTransferRegion],
) -> tuple[BoundWeightTransferRegion, ...]:
    result = []
    by_target: dict[tuple, BoundWeightTransferRegion] = {}
    for region in regions:
        target_key = _target_identity(region)
        previous = by_target.get(target_key)
        if previous is None:
            by_target[target_key] = region
            result.append(region)
            continue
        exact_alias = (
            len(previous.target.aliases) > 1
            and previous.target.aliases == region.target.aliases
            and previous.source.aliases == region.source.aliases
            and _sources_have_identical_payload(previous, region)
            and previous.logical_region.overlap_offset
            == region.logical_region.overlap_offset
            and previous.logical_region.overlap_shape
            == region.logical_region.overlap_shape
        )
        if not exact_alias:
            raise ValueError(
                "overlapping target writes: "
                f"tensor={region.tensor_id!r}, "
                f"target_address={region.target.address + region.target_base_offset}, "
                f"source_fragments=({previous.source.placement_fragment_id!r}, "
                f"{region.source.placement_fragment_id!r}), "
                f"logical_regions=("
                f"{previous.logical_region.overlap_offset}/"
                f"{previous.logical_region.overlap_shape}, "
                f"{region.logical_region.overlap_offset}/"
                f"{region.logical_region.overlap_shape})"
            )
    return tuple(result)


def bind_weight_transfer_plan(
    logical_plan: LogicalWeightTransferPlan,
    *,
    target_bindings: Sequence[WeightRuntimeBindingManifest],
    source_bindings: Sequence[SourceBindingManifest],
) -> BoundWeightTransferPlan:
    """Bind an address-free plan to one current physical snapshot."""

    if not source_bindings:
        raise ValueError("source bindings must not be empty")
    source_runtime = all(
        isinstance(item, WeightRuntimeBindingManifest) for item in source_bindings
    )
    source_storage = all(
        isinstance(item, WeightStorageBindingManifest) for item in source_bindings
    )
    if source_runtime == source_storage:
        raise ValueError("source bindings must use one physical location kind")

    if source_runtime:
        runtime_bindings = tuple(
            item
            for item in source_bindings
            if isinstance(item, WeightRuntimeBindingManifest)
        )
        source_generations = {item.generation for item in runtime_bindings}
        if len(source_generations) != 1:
            raise ValueError("source generations differ")
        source_locations = _runtime_locations(
            logical_plan.source_placements,
            runtime_bindings,
            "source",
        )
    else:
        storage_bindings = tuple(
            item
            for item in source_bindings
            if isinstance(item, WeightStorageBindingManifest)
        )
        source_locations = _storage_locations(
            logical_plan.source_placements,
            storage_bindings,
            "source",
        )

    target_bindings = tuple(target_bindings)
    target_locations = _runtime_locations(
        logical_plan.target_placements,
        target_bindings,
        "target",
    )
    _validate_target_allocations(tuple(target_locations.values()))
    if source_runtime:
        _validate_runtime_source_target_allocations(
            tuple(source_locations.values()),
            tuple(target_locations.values()),
        )

    regions = []
    for logical_region in logical_plan.regions:
        source = source_locations.get(logical_region.source.placement_fragment_id)
        target = target_locations.get(logical_region.target.placement_fragment_id)
        if source is None or target is None:
            raise ValueError("logical region is missing a physical binding")
        regions.append(
            BoundWeightTransferRegion(
                logical_region=logical_region,
                source=source,
                target=target,
            )
        )
    bound_regions = _deduplicate_aliases(regions)
    return BoundWeightTransferPlan(
        logical_plan=logical_plan,
        regions=bound_regions,
        source_bindings=tuple(source_bindings),
        target_bindings=target_bindings,
    )


def bind_weight_source(
    placements: Sequence[WeightPlacementManifest],
    bindings: Sequence[SourceBindingManifest],
) -> tuple[RuntimeWeightLocation | StorageWeightLocation, ...]:
    """Validate and bind a source snapshot without creating a load plan."""

    if not bindings:
        raise ValueError("source bindings must not be empty")
    source_runtime = all(
        isinstance(item, WeightRuntimeBindingManifest) for item in bindings
    )
    source_storage = all(
        isinstance(item, WeightStorageBindingManifest) for item in bindings
    )
    if source_runtime == source_storage:
        raise ValueError("source bindings must use one physical location kind")
    if source_runtime:
        runtime_bindings = tuple(
            item for item in bindings if isinstance(item, WeightRuntimeBindingManifest)
        )
        if len({item.generation for item in runtime_bindings}) != 1:
            raise ValueError("source generations differ")
        locations = _runtime_locations(placements, runtime_bindings, "source")
    else:
        storage_bindings = tuple(
            item for item in bindings if isinstance(item, WeightStorageBindingManifest)
        )
        locations = _storage_locations(placements, storage_bindings, "source")
    return tuple(
        sorted(
            locations.values(),
            key=lambda item: (
                item.placement_id,
                item.placement_fragment_id,
            ),
        )
    )


def project_source_bindings(
    placements: Sequence[WeightPlacementManifest],
    bindings: Sequence[SourceBindingManifest],
) -> tuple[SourceBindingManifest, ...]:
    """Project validated bindings onto an address-free placement subset."""

    if not placements or not bindings:
        raise ValueError("placements and bindings must not be empty")
    runtime = all(
        isinstance(binding, WeightRuntimeBindingManifest) for binding in bindings
    )
    storage = all(
        isinstance(binding, WeightStorageBindingManifest) for binding in bindings
    )
    if runtime == storage:
        raise ValueError("source bindings must use one physical location kind")
    binding_by_id = {}
    binding_by_fragment_id = {}
    for binding in bindings:
        if binding.placement_id in binding_by_id:
            raise ValueError("duplicate source binding")
        binding_by_id[binding.placement_id] = binding
        for fragment in binding.fragments:
            if fragment.placement_fragment_id in binding_by_fragment_id:
                raise ValueError("duplicate source binding fragment")
            binding_by_fragment_id[fragment.placement_fragment_id] = binding

    projected = []
    for placement in sorted(placements, key=lambda item: item.placement_id):
        selected_ids = {tensor.placement_fragment_id for tensor in placement.tensors}
        original_placement_ids = {
            binding_by_fragment_id[fragment_id].placement_id
            for fragment_id in selected_ids
            if fragment_id in binding_by_fragment_id
        }
        if len(original_placement_ids) != 1 or any(
            fragment_id not in binding_by_fragment_id for fragment_id in selected_ids
        ):
            raise ValueError(
                "selected placement fragments do not share one source binding"
            )
        binding = binding_by_id[next(iter(original_placement_ids))]
        if (
            placement.model_id != binding.model_id
            or placement.revision != binding.revision
        ):
            raise ValueError("source binding model identity differs")
        if isinstance(binding, WeightRuntimeBindingManifest):
            fragments = tuple(
                fragment
                for fragment in binding.fragments
                if fragment.placement_fragment_id in selected_ids
            )
            if {
                fragment.placement_fragment_id for fragment in fragments
            } != selected_ids:
                raise ValueError("selected runtime binding fragment IDs differ")
            projected.append(
                WeightRuntimeBindingManifest(
                    model_id=binding.model_id,
                    revision=binding.revision,
                    placement_id=placement.placement_id,
                    instance_id=binding.instance_id,
                    generation=binding.generation,
                    lease_id=binding.lease_id,
                    fragments=fragments,
                    format_version=binding.format_version,
                )
            )
        else:
            fragments = tuple(
                fragment
                for fragment in binding.fragments
                if fragment.placement_fragment_id in selected_ids
            )
            if (
                not all(
                    isinstance(fragment, WeightStorageFragmentBinding)
                    for fragment in fragments
                )
                or {fragment.placement_fragment_id for fragment in fragments}
                != selected_ids
            ):
                raise ValueError("selected storage binding fragment IDs differ")
            projected.append(
                WeightStorageBindingManifest(
                    model_id=binding.model_id,
                    revision=binding.revision,
                    placement_id=placement.placement_id,
                    storage_id=binding.storage_id,
                    provider=binding.provider,
                    fragments=fragments,
                    format_version=binding.format_version,
                )
            )
    return tuple(projected)
