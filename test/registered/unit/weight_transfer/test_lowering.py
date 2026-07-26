from __future__ import annotations

import pytest

from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.api import (
    prepare_weight_load,
    prepare_weight_load_to_local_target,
)
from sglang.srt.weight_transfer.lowering import (
    WeightLoweringLimits,
    iter_bounded_transfer_batches,
    lowering_operation_count,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _placement(
    side: str,
    *,
    global_shape: tuple[int, ...],
    global_offset: tuple[int, ...],
    local_shape: tuple[int, ...],
    shard_dims: tuple[int, ...],
) -> WeightPlacementManifest:
    itemsize = 2
    nbytes = itemsize
    for dimension in local_shape:
        nbytes *= dimension
    tensor = WeightPlacementTensor(
        placement_fragment_id=f"{side}:fragment",
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=global_shape,
        global_offset=global_offset,
        local_shape=local_shape,
        dtype="bfloat16",
        itemsize=itemsize,
        partition_dim=shard_dims[0] if len(shard_dims) == 1 else None,
        shard_dims=shard_dims,
        layer_id=0,
        expert_id=None,
        layout_fingerprint="logical-contiguous:v2",
        nbytes=nbytes,
        byte_offset=0,
        rank=WeightParallelRank(),
    )
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id((tensor,)),
        tensors=(tensor,),
    )


def _binding(
    placement: WeightPlacementManifest,
    address: int,
) -> WeightRuntimeBindingManifest:
    tensor = placement.tensors[0]
    return WeightRuntimeBindingManifest(
        model_id=placement.model_id,
        revision=placement.revision,
        placement_id=placement.placement_id,
        instance_id=f"instance:{address}",
        generation=1,
        lease_id=f"lease:{address}",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=address,
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id=f"worker:{side_from_address(address)}",
                endpoint=f"endpoint:{side_from_address(address)}",
            ),
        ),
    )


def side_from_address(address: int) -> str:
    return f"{address:x}"


def _cross_dim_plan():
    source = _placement(
        "source",
        global_shape=(4, 4),
        global_offset=(0, 0),
        local_shape=(4, 4),
        shard_dims=(),
    )
    target = _placement(
        "target",
        global_shape=(4, 4),
        global_offset=(0, 0),
        local_shape=(4, 2),
        shard_dims=(1,),
    )
    return prepare_weight_load_to_local_target(
        source_placements=(source,),
        source_bindings=(_binding(source, 0x10000),),
        target_placement=target,
        target_binding=_binding(target, 0x20000),
    ).plan


def test_bounded_lowering_never_materializes_an_unbounded_batch() -> None:
    plan = _cross_dim_plan()
    limits = WeightLoweringLimits(
        max_total_operations=4,
        max_batch_operations=2,
        max_batch_bytes=8,
    )

    batches = tuple(iter_bounded_transfer_batches(plan, limits))

    assert lowering_operation_count(plan, limits) == 4
    assert [len(batch.operations) for batch in batches] == [2, 2]
    assert [batch.total_bytes for batch in batches] == [8, 8]
    assert [
        (
            operation.source_offset,
            operation.target_offset,
            operation.nbytes,
        )
        for batch in batches
        for operation in batch.operations
    ] == [
        (0, 0, 4),
        (8, 4, 4),
        (16, 8, 4),
        (24, 12, 4),
    ]


def test_bounded_lowering_splits_large_inner_ranges_by_byte_limit() -> None:
    source = _placement(
        "source",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
    )
    target = _placement(
        "target",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
    )
    plan = prepare_weight_load(
        source_placements=(source,),
        source_bindings=(_binding(source, 0x10000),),
        target_placements=(target,),
        target_bindings=(_binding(target, 0x20000),),
    ).plan
    limits = WeightLoweringLimits(
        max_total_operations=3,
        max_batch_operations=2,
        max_batch_bytes=6,
    )

    batches = tuple(iter_bounded_transfer_batches(plan, limits))

    assert lowering_operation_count(plan, limits) == 3
    assert [batch.total_bytes for batch in batches] == [6, 6, 4]
    assert [
        operation.nbytes for batch in batches for operation in batch.operations
    ] == [6, 6, 4]


def test_bounded_lowering_rejects_total_operation_limit_before_iteration() -> None:
    plan = _cross_dim_plan()
    limits = WeightLoweringLimits(
        max_total_operations=3,
        max_batch_operations=2,
        max_batch_bytes=8,
    )

    with pytest.raises(ValueError, match="total operation limit"):
        tuple(iter_bounded_transfer_batches(plan, limits))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
