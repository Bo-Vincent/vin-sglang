from __future__ import annotations

from dataclasses import replace
from math import prod

import pytest

from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
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
    LogicalWeightTransferPlan,
    PipelineRouteGroup,
    build_region,
)
from sglang.srt.weight_transfer.planner import plan_weight_transfer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def placement(
    side: str,
    *,
    shape: tuple[int, ...] = (8,),
    offset: tuple[int, ...] = (0,),
    global_shape: tuple[int, ...] = (8,),
    shard_dims: tuple[int, ...] = (),
) -> WeightPlacementManifest:
    fragment_id = f"{side}:weight"
    tensors = (
        WeightPlacementTensor(
            placement_fragment_id=fragment_id,
            tensor_id="weight",
            runtime_name="weight",
            aliases=("weight",),
            global_shape=global_shape,
            global_offset=offset,
            local_shape=shape,
            dtype="bfloat16",
            itemsize=2,
            partition_dim=(shard_dims[0] if len(shard_dims) == 1 else None),
            shard_dims=shard_dims,
            layer_id=0,
            expert_id=None,
            layout_fingerprint="layout:v1",
            nbytes=prod(shape) * 2,
            byte_offset=0,
            rank=WeightParallelRank(),
        ),
    )
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id(tuple(tensors)),
        tensors=tensors,
    )


def runtime_binding(
    manifest: WeightPlacementManifest,
    *,
    address: int,
) -> WeightRuntimeBindingManifest:
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id=f"instance:{manifest.placement_id}",
        generation=1,
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


def test_bound_plan_rejects_extra_region() -> None:
    valid = bound_plan()

    with pytest.raises(ValueError, match="exactly match"):
        replace(valid, regions=(*valid.regions, unexpected_bound_region(valid)))


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
        ),
    )

    plan = plan_weight_transfer((source,), targets)

    assert plan.total_segments == sum(region.segment_count for region in plan.regions)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
