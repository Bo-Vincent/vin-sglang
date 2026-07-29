from __future__ import annotations

from dataclasses import replace
from math import prod

import msgspec
import pytest
from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightManifestError,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.binding import (
    bind_weight_transfer_plan,
)
from sglang.srt.weight_transfer.contracts import (
    BoundWeightTransferPlan,
    BoundWeightTransferRegion,
    LogicalPlacementFragment,
    LogicalWeightTransferPlan,
    PipelineRouteGroup,
    StorageWeightLocation,
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
    build_region,
)
from sglang.srt.weight_transfer.planner import plan_weight_transfer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def placement(
    side: str,
    *,
    tensor_id: str = "weight",
    shape: tuple[int, ...] = (8,),
    offset: tuple[int, ...] | None = None,
    global_shape: tuple[int, ...] = (8,),
    shard_dims: tuple[int, ...] = (),
    dtype: str = "bfloat16",
    itemsize: int = 2,
    layer_id: int | None = 0,
    expert_id: int | None = None,
    layout_fingerprint: str = "layout:v1",
    nbytes: int | None = None,
    rank: WeightParallelRank = WeightParallelRank(),
    aliases: tuple[str, ...] = ("weight",),
) -> WeightPlacementManifest:
    fragment_id = f"{side}:{tensor_id}:{rank.dp}:{rank.tp}:{rank.pp}:{rank.ep}"
    tensors = (
        WeightPlacementTensor(
            placement_fragment_id=fragment_id,
            tensor_id=tensor_id,
            runtime_name=tensor_id,
            aliases=aliases,
            global_shape=global_shape,
            global_offset=(0,) * len(shape) if offset is None else offset,
            local_shape=shape,
            dtype=dtype,
            itemsize=itemsize,
            partition_dim=(shard_dims[0] if len(shard_dims) == 1 else None),
            shard_dims=shard_dims,
            layer_id=layer_id,
            expert_id=expert_id,
            layout_fingerprint=layout_fingerprint,
            nbytes=prod(shape) * itemsize if nbytes is None else nbytes,
            byte_offset=0,
            rank=rank,
        ),
    )
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id(tuple(tensors)),
        tensors=tensors,
    )


def logical_fragment(
    manifest: WeightPlacementManifest,
) -> LogicalPlacementFragment:
    tensor = manifest.tensors[0]
    return LogicalPlacementFragment(
        placement_id=manifest.placement_id,
        placement_fragment_id=tensor.placement_fragment_id,
        tensor_id=tensor.tensor_id,
        global_offset=tuple(tensor.global_offset),
        local_shape=tuple(tensor.local_shape),
        nbytes=tensor.nbytes,
        rank=tensor.rank,
        aliases=tuple(tensor.aliases),
    )


def test_legacy_partition_dim_cannot_describe_multi_dimensional_sharding() -> None:
    manifest = placement(
        "source",
        shape=(2, 4),
        global_shape=(4, 4),
        shard_dims=(0,),
    )
    tensor = msgspec.structs.replace(
        manifest.tensors[0],
        shard_dims=(0, 1),
    )

    with pytest.raises(WeightManifestError, match="shard dimensions"):
        WeightPlacementManifest(
            model_id=manifest.model_id,
            revision=manifest.revision,
            placement_id=compute_weight_placement_id((tensor,)),
            tensors=(tensor,),
        )


def runtime_binding(
    manifest: WeightPlacementManifest,
    *,
    address: int,
    generation: int = 1,
) -> WeightRuntimeBindingManifest:
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id=f"instance:{manifest.placement_id}",
        generation=generation,
        lease_id=f"lease:{manifest.placement_id}",
        fragments=tuple(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=address + tensor.byte_offset,
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id=f"worker:{manifest.placement_id}",
                endpoint=f"endpoint:{manifest.placement_id}",
            )
            for tensor in manifest.tensors
        ),
    )


def alias_runtime_binding(
    binding: WeightRuntimeBindingManifest,
    reference: WeightRuntimeBindingManifest,
    *,
    generation: int | None = None,
    lease_id: str | None = None,
) -> WeightRuntimeBindingManifest:
    return msgspec.structs.replace(
        binding,
        generation=reference.generation if generation is None else generation,
        lease_id=reference.lease_id if lease_id is None else lease_id,
        fragments=tuple(
            msgspec.structs.replace(
                fragment,
                address=reference_fragment.address,
                nbytes=reference_fragment.nbytes,
                device=reference_fragment.device,
                worker_id=reference_fragment.worker_id,
                endpoint=reference_fragment.endpoint,
            )
            for fragment, reference_fragment in zip(
                binding.fragments,
                reference.fragments,
                strict=True,
            )
        ),
    )


def bound_plan() -> BoundWeightTransferPlan:
    source = placement("source")
    targets = (
        placement(
            "target-0",
            shape=(4,),
            offset=(0,),
            global_shape=(8,),
            shard_dims=(0,),
        ),
        placement(
            "target-1",
            shape=(4,),
            offset=(4,),
            global_shape=(8,),
            shard_dims=(0,),
            rank=WeightParallelRank(tp=1),
        ),
    )
    logical = plan_weight_transfer((source,), targets)
    return bind_weight_transfer_plan(
        logical,
        source_bindings=(runtime_binding(source, address=0x10000),),
        target_bindings=tuple(
            runtime_binding(target, address=0x20000 + index * 0x1000)
            for index, target in enumerate(targets)
        ),
    )


def unexpected_bound_region(
    plan: BoundWeightTransferPlan,
) -> BoundWeightTransferRegion:
    reference = plan.regions[0]
    logical = reference.logical_region
    unexpected_logical = build_region(
        tensor_id=logical.tensor_id,
        source=logical.source,
        target=logical.target,
        overlap_offset=logical.overlap_offset,
        overlap_shape=(logical.overlap_shape[0] // 2,),
    )
    return BoundWeightTransferRegion(
        logical_region=unexpected_logical,
        source=reference.source,
        target=reference.target,
    )


def test_bound_plan_rejects_empty_regions() -> None:
    valid = bound_plan()

    with pytest.raises(ValueError, match="must not be empty"):
        replace(valid, regions=())


def test_bound_plan_rejects_missing_region() -> None:
    valid = bound_plan()

    with pytest.raises(ValueError, match="exactly match"):
        replace(valid, regions=valid.regions[:-1])


def test_bound_plan_rejects_logical_region_identity_mismatch() -> None:
    valid = bound_plan()

    with pytest.raises(ValueError, match="exactly match"):
        replace(
            valid,
            regions=(unexpected_bound_region(valid), valid.regions[1]),
        )


@pytest.mark.parametrize("side", ("source", "target"))
def test_bound_plan_rejects_physical_location_outside_binding(side) -> None:
    valid = bound_plan()
    region = valid.regions[0]
    location = getattr(region, side)
    changed = replace(location, address=location.address + 0x1000)
    invalid_region = replace(region, **{side: changed})

    with pytest.raises(ValueError, match=f"{side} bindings"):
        replace(valid, regions=(invalid_region, valid.regions[1]))


def test_bound_plan_rejects_duplicate_region() -> None:
    valid = bound_plan()

    with pytest.raises(ValueError, match="exactly match"):
        replace(valid, regions=(valid.regions[0], valid.regions[0]))


def test_bound_plan_rejects_mixed_runtime_source_generations() -> None:
    sources = (
        placement(
            "source-0",
            shape=(4,),
            offset=(0,),
            global_shape=(8,),
            shard_dims=(0,),
            rank=WeightParallelRank(tp=0),
        ),
        placement(
            "source-1",
            shape=(4,),
            offset=(4,),
            global_shape=(8,),
            shard_dims=(0,),
            rank=WeightParallelRank(tp=1),
        ),
    )
    target = placement("target")
    valid = bind_weight_transfer_plan(
        plan_weight_transfer(sources, (target,)),
        source_bindings=(
            runtime_binding(sources[0], address=0x10000),
            runtime_binding(sources[1], address=0x11000),
        ),
        target_bindings=(runtime_binding(target, address=0x20000),),
    )
    source_bindings = (
        valid.source_bindings[0],
        msgspec.structs.replace(valid.source_bindings[1], generation=2),
    )
    regions = tuple(
        (
            replace(
                region,
                source=replace(region.source, generation=2),
            )
            if region.source.placement_id == sources[1].placement_id
            else region
        )
        for region in valid.regions
    )

    with pytest.raises(ValueError, match="source generations differ"):
        replace(
            valid,
            regions=regions,
            source_bindings=source_bindings,
        )


@pytest.mark.parametrize("side", ("source", "target"))
def test_bound_plan_rejects_unreferenced_binding_fragment(side: str) -> None:
    valid = bound_plan()
    attribute = f"{side}_bindings"
    bindings = getattr(valid, attribute)
    binding = bindings[0]
    fragment = binding.fragments[0]
    extra_fragment = msgspec.structs.replace(
        fragment,
        placement_fragment_id=f"unused:{side}",
        fragment_id=f"unused:{side}",
        address=fragment.address + 0x1000,
    )
    changed = msgspec.structs.replace(
        binding,
        fragments=(*binding.fragments, extra_fragment),
    )

    with pytest.raises(
        ValueError,
        match=f"{side} binding fragments must exactly match logical regions",
    ):
        replace(valid, **{attribute: (changed, *bindings[1:])})


@pytest.mark.parametrize("side", ("source", "target"))
def test_bound_plan_rejects_missing_binding_fragment(side: str) -> None:
    valid = bound_plan()
    attribute = f"{side}_bindings"
    bindings = getattr(valid, attribute)

    with pytest.raises(
        ValueError,
        match=f"{side} binding fragments must exactly match logical regions",
    ):
        replace(valid, **{attribute: bindings[:-1]})


def test_bound_plan_allows_storage_source_without_generation() -> None:
    source = placement("source")
    target = placement("target")
    logical = plan_weight_transfer((source,), (target,))
    source_tensor = source.tensors[0]
    storage_binding = WeightStorageBindingManifest(
        model_id=source.model_id,
        revision=source.revision,
        placement_id=source.placement_id,
        storage_id="weight-store",
        provider="store",
        fragments=(
            WeightStorageFragmentBinding(
                placement_fragment_id=source_tensor.placement_fragment_id,
                fragment_id="stored:weight",
                object_key="weights/weight",
                object_offset=0,
                nbytes=source_tensor.nbytes,
            ),
        ),
    )

    bound = bind_weight_transfer_plan(
        logical,
        source_bindings=(storage_binding,),
        target_bindings=(runtime_binding(target, address=0x20000),),
    )

    assert isinstance(bound.regions[0].source, StorageWeightLocation)


def test_bound_plan_rejects_extra_region() -> None:
    valid = bound_plan()

    with pytest.raises(ValueError, match="exactly match"):
        replace(valid, regions=(*valid.regions, unexpected_bound_region(valid)))


def test_bound_plan_contract_allows_exact_target_alias() -> None:
    aliases = ("shared.weight", "shared.weight.alias")
    source = placement("source", aliases=aliases)
    targets = (
        placement(
            "target-0",
            rank=WeightParallelRank(dp=0),
            aliases=aliases,
        ),
        placement(
            "target-1",
            rank=WeightParallelRank(dp=1),
            aliases=aliases,
        ),
    )
    logical = plan_weight_transfer((source,), targets)
    valid = bind_weight_transfer_plan(
        logical,
        source_bindings=(runtime_binding(source, address=0x10000),),
        target_bindings=(
            runtime_binding(targets[0], address=0x20000),
            runtime_binding(targets[1], address=0x30000),
        ),
    )
    aliased_target = alias_runtime_binding(
        valid.target_bindings[1],
        valid.target_bindings[0],
    )

    rebound = bind_weight_transfer_plan(
        logical,
        source_bindings=valid.source_bindings,
        target_bindings=(valid.target_bindings[0], aliased_target),
    )

    assert len(rebound.regions) == 1


def test_bound_plan_contract_rejects_partial_target_alias_overlap() -> None:
    aliases = ("shared.weight", "shared.weight.alias")
    source_full = placement(
        "source-full",
        tensor_id="full",
        shape=(2, 4),
        global_shape=(2, 4),
        aliases=aliases,
    )
    source_rows = (
        placement(
            "source-row-0",
            tensor_id="rows",
            shape=(1, 4),
            offset=(0, 0),
            global_shape=(2, 4),
            shard_dims=(0,),
            rank=WeightParallelRank(tp=0),
            aliases=aliases,
        ),
        placement(
            "source-row-1",
            tensor_id="rows",
            shape=(1, 4),
            offset=(1, 0),
            global_shape=(2, 4),
            shard_dims=(0,),
            rank=WeightParallelRank(tp=1),
            aliases=aliases,
        ),
    )
    targets = (
        placement(
            "target-full",
            tensor_id="full",
            shape=(2, 4),
            global_shape=(2, 4),
            aliases=aliases,
        ),
        placement(
            "target-rows",
            tensor_id="rows",
            shape=(2, 4),
            global_shape=(2, 4),
            rank=WeightParallelRank(pp=1),
            aliases=aliases,
        ),
    )
    logical = plan_weight_transfer(
        (source_full, *source_rows),
        targets,
    )
    valid = bind_weight_transfer_plan(
        logical,
        source_bindings=(
            runtime_binding(source_full, address=0x10000),
            runtime_binding(source_rows[0], address=0x11000),
            runtime_binding(source_rows[1], address=0x12000),
        ),
        target_bindings=(
            runtime_binding(targets[0], address=0x20000),
            runtime_binding(targets[1], address=0x30000),
        ),
    )
    aliased_target = alias_runtime_binding(
        valid.target_bindings[1],
        valid.target_bindings[0],
    )

    with pytest.raises(
        ValueError,
        match="overlapping target writes: logical regions partially overlap",
    ):
        replace(
            valid,
            target_bindings=(valid.target_bindings[0], aliased_target),
        )


def test_bound_plan_contract_rejects_target_alias_snapshot_mismatch() -> None:
    aliases = ("shared.weight", "shared.weight.alias")
    source = placement("source", aliases=aliases)
    targets = (
        placement(
            "target-0",
            rank=WeightParallelRank(dp=0),
            aliases=aliases,
        ),
        placement(
            "target-1",
            rank=WeightParallelRank(dp=1),
            aliases=aliases,
        ),
    )
    logical = plan_weight_transfer((source,), targets)
    valid = bind_weight_transfer_plan(
        logical,
        source_bindings=(runtime_binding(source, address=0x10000),),
        target_bindings=(
            runtime_binding(targets[0], address=0x20000),
            runtime_binding(targets[1], address=0x30000),
        ),
    )
    aliased_target = alias_runtime_binding(
        valid.target_bindings[1],
        valid.target_bindings[0],
        generation=valid.target_bindings[0].generation + 1,
        lease_id="other-lease",
    )

    with pytest.raises(
        ValueError,
        match="target alias snapshot identity differs",
    ):
        replace(
            valid,
            target_bindings=(valid.target_bindings[0], aliased_target),
        )


def test_bound_plan_contract_rejects_different_source_payloads() -> None:
    aliases = ("shared.weight", "shared.weight.alias")
    sources = (
        placement(
            "source-0",
            rank=WeightParallelRank(dp=0),
            aliases=aliases,
        ),
        placement(
            "source-1",
            rank=WeightParallelRank(dp=1),
            aliases=aliases,
        ),
    )
    targets = (
        placement(
            "target-0",
            rank=WeightParallelRank(dp=0),
            aliases=aliases,
        ),
        placement(
            "target-1",
            rank=WeightParallelRank(dp=1),
            aliases=aliases,
        ),
    )
    logical = plan_weight_transfer(sources, targets)
    valid = bind_weight_transfer_plan(
        logical,
        source_bindings=(
            runtime_binding(sources[0], address=0x10000),
            runtime_binding(sources[1], address=0x11000),
        ),
        target_bindings=(
            runtime_binding(targets[0], address=0x20000),
            runtime_binding(targets[1], address=0x30000),
        ),
    )
    aliased_target = alias_runtime_binding(
        valid.target_bindings[1],
        valid.target_bindings[0],
    )

    with pytest.raises(
        ValueError,
        match="overlapping target writes: source payload differs",
    ):
        replace(
            valid,
            target_bindings=(valid.target_bindings[0], aliased_target),
        )


@pytest.mark.parametrize(
    "boxes",
    [
        (((0,), (3,)), ((4,), (4,))),
        (((0,), (6,)), ((4,), (4,))),
    ],
    ids=("gap", "partial-overlap"),
)
def test_logical_plan_rejects_incomplete_target_coverage(boxes) -> None:
    source = placement("source")
    target = placement("target")
    valid = plan_weight_transfer((source,), (target,))
    source_fragment = valid.regions[0].source
    target_fragment = valid.regions[0].target
    regions = tuple(
        build_region(
            tensor_id="weight",
            source=source_fragment,
            target=target_fragment,
            overlap_offset=offset,
            overlap_shape=shape,
        )
        for offset, shape in boxes
    )

    with pytest.raises(ValueError, match="exactly cover target fragment"):
        LogicalWeightTransferPlan(
            model_id=valid.model_id,
            revision=valid.revision,
            source_placements=valid.source_placements,
            target_placements=valid.target_placements,
            regions=regions,
            pipeline_routes=(
                PipelineRouteGroup(
                    source_pp=0,
                    target_pp=0,
                    region_indices=tuple(range(len(regions))),
                ),
            ),
        )


@pytest.mark.parametrize(
    "target_overrides",
    (
        {"global_shape": (16,), "shard_dims": (0,)},
        {"dtype": "float16"},
        {"layer_id": 1},
        {"expert_id": 1},
        {"layout_fingerprint": "layout:v2"},
    ),
    ids=(
        "global-shape",
        "dtype",
        "layer",
        "expert",
        "layout",
    ),
)
def test_logical_plan_rejects_forged_tensor_semantics(target_overrides) -> None:
    source = placement("source")
    target = placement("target", **target_overrides)
    source_fragment = logical_fragment(source)
    target_fragment = logical_fragment(target)
    region = build_region(
        tensor_id="weight",
        source=source_fragment,
        target=target_fragment,
        overlap_offset=(0,),
        overlap_shape=(8,),
    )

    with pytest.raises(ValueError, match="tensor descriptor mismatch: weight"):
        LogicalWeightTransferPlan(
            model_id="model",
            revision="revision",
            source_placements=(source,),
            target_placements=(target,),
            regions=(region,),
            pipeline_routes=(
                PipelineRouteGroup(
                    source_pp=0,
                    target_pp=0,
                    region_indices=(0,),
                ),
            ),
        )


def test_build_region_rejects_itemsize_mismatch() -> None:
    source = placement("source")
    target = placement("target", itemsize=4, nbytes=32)

    with pytest.raises(ValueError, match="itemsize differ"):
        build_region(
            tensor_id="weight",
            source=logical_fragment(source),
            target=logical_fragment(target),
            overlap_offset=(0,),
            overlap_shape=(8,),
        )


def test_logical_plan_rejects_duplicate_target_boxes() -> None:
    source = placement("source")
    target = placement("target")
    source_fragment = logical_fragment(source)
    target_fragment = logical_fragment(target)
    region = build_region(
        tensor_id="weight",
        source=source_fragment,
        target=target_fragment,
        overlap_offset=(0,),
        overlap_shape=(8,),
    )

    with pytest.raises(ValueError, match="exactly cover target fragment"):
        LogicalWeightTransferPlan(
            model_id="model",
            revision="revision",
            source_placements=(source,),
            target_placements=(target,),
            regions=(region, region),
            pipeline_routes=(
                PipelineRouteGroup(
                    source_pp=0,
                    target_pp=0,
                    region_indices=(0, 1),
                ),
            ),
        )


def test_logical_plan_reports_total_segments() -> None:
    source = placement(
        "source",
        shape=(2, 4),
        offset=(0, 0),
        global_shape=(2, 4),
    )
    targets = (
        placement(
            "target-0",
            shape=(2, 2),
            offset=(0, 0),
            global_shape=(2, 4),
            shard_dims=(1,),
        ),
        placement(
            "target-1",
            shape=(2, 2),
            offset=(0, 2),
            global_shape=(2, 4),
            shard_dims=(1,),
            rank=WeightParallelRank(tp=1),
        ),
    )

    plan = plan_weight_transfer((source,), targets)

    assert plan.total_segments == sum(region.segment_count for region in plan.regions)


def test_logical_plan_digest_is_cached(monkeypatch) -> None:
    from sglang.srt.weight_transfer import contracts as contracts_module

    source = placement("source")
    target = placement("target")
    plan = plan_weight_transfer((source,), (target,))
    digest = plan.digest

    def fail_recompute(_placement):
        raise AssertionError("logical plan digest was recomputed")

    monkeypatch.setattr(
        contracts_module,
        "_placement_identity",
        fail_recompute,
    )

    assert plan.digest == digest


def test_bound_plan_digest_is_cached(monkeypatch) -> None:
    from sglang.srt.weight_transfer import contracts as contracts_module

    plan = bound_plan()
    digest = plan.digest

    def fail_recompute(_value):
        raise AssertionError("bound plan digest was recomputed")

    monkeypatch.setattr(contracts_module.hashlib, "sha256", fail_recompute)

    assert plan.digest == digest


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
