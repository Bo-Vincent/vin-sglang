from __future__ import annotations

from math import prod

import pytest

from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    compose_weight_runtime_manifest,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.binding import (
    bind_weight_transfer_plan,
    runtime_manifest_to_parts,
)
from sglang.srt.weight_transfer.contracts import (
    RuntimeWeightLocation,
    StorageWeightLocation,
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
)
from sglang.srt.weight_transfer.planner import plan_weight_transfer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def placement(
    side: str,
    *,
    tensor_id: str = "weight",
    shape: tuple[int, ...] = (8,),
    offset: tuple[int, ...] = (0,),
    global_shape: tuple[int, ...] = (8,),
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
            global_offset=offset,
            local_shape=shape,
            dtype="bfloat16",
            itemsize=2,
            partition_dim=0 if shape != global_shape else None,
            shard_dims=(0,) if shape != global_shape else (),
            layer_id=0,
            expert_id=None,
            layout_fingerprint="layout:v1",
            nbytes=prod(shape) * 2,
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


def runtime_binding(
    manifest: WeightPlacementManifest,
    *,
    address: int,
    generation: int = 1,
    lease_id: str = "lease",
    nbytes_delta: int = 0,
    worker_id: str = "worker",
    endpoint: str = "worker:12345",
    device: str = "cuda:0",
) -> WeightRuntimeBindingManifest:
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id=f"instance:{worker_id}",
        generation=generation,
        lease_id=lease_id,
        fragments=tuple(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=address + index * 0x1000,
                nbytes=tensor.nbytes + nbytes_delta,
                storage_offset=0,
                device=device,
                is_contiguous=True,
                worker_id=worker_id,
                endpoint=endpoint,
            )
            for index, tensor in enumerate(manifest.tensors)
        ),
    )


def storage_binding(
    manifest: WeightPlacementManifest,
    *,
    object_key: str = "weights/revision/object-0",
    object_offset: int = 0,
) -> WeightStorageBindingManifest:
    return WeightStorageBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        storage_id="store:revision",
        provider="mooncake-store",
        fragments=tuple(
            WeightStorageFragmentBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"stored:{tensor.placement_fragment_id}",
                object_key=object_key,
                object_offset=object_offset + index * 0x1000,
                nbytes=tensor.nbytes,
                checksum=f"sha256:{index}",
            )
            for index, tensor in enumerate(manifest.tensors)
        ),
    )


def test_bind_runtime_to_runtime_plan() -> None:
    source = placement("source")
    target = placement("target")
    logical = plan_weight_transfer((source,), (target,))

    bound = bind_weight_transfer_plan(
        logical,
        source_bindings=(runtime_binding(source, address=0x10000),),
        target_bindings=(runtime_binding(target, address=0x20000),),
    )

    assert bound.logical_plan is logical
    assert bound.total_bytes == 16
    assert len(bound.regions) == 1
    assert isinstance(bound.regions[0].source, RuntimeWeightLocation)
    assert isinstance(bound.regions[0].target, RuntimeWeightLocation)
    assert bound.regions[0].source.address == 0x10000
    assert bound.regions[0].target.address == 0x20000
    assert list(bound.regions[0].iter_absolute_segments()) == [(0x10000, 0x20000, 16)]


def test_bind_storage_to_runtime_plan() -> None:
    source = placement("source")
    target = placement("target")
    logical = plan_weight_transfer((source,), (target,))

    bound = bind_weight_transfer_plan(
        logical,
        source_bindings=(storage_binding(source, object_offset=0x20000),),
        target_bindings=(runtime_binding(target, address=0x20000),),
    )

    assert isinstance(bound.regions[0].source, StorageWeightLocation)
    assert bound.regions[0].source.object_key == "weights/revision/object-0"
    assert bound.regions[0].source.object_offset == 0x20000


def test_binding_rejects_placement_identity_mismatch() -> None:
    source = placement("source")
    target = placement("target")
    logical = plan_weight_transfer((source,), (target,))
    wrong = runtime_binding(source, address=0x10000)
    wrong = WeightRuntimeBindingManifest(
        model_id=wrong.model_id,
        revision=wrong.revision,
        placement_id="wrong-placement",
        instance_id=wrong.instance_id,
        generation=wrong.generation,
        lease_id=wrong.lease_id,
        fragments=wrong.fragments,
    )

    with pytest.raises(ValueError, match="placement IDs differ"):
        bind_weight_transfer_plan(
            logical,
            source_bindings=(wrong,),
            target_bindings=(runtime_binding(target, address=0x20000),),
        )


def test_binding_rejects_missing_or_extra_fragments() -> None:
    source = placement("source")
    target = placement("target")
    logical = plan_weight_transfer((source,), (target,))
    incomplete = runtime_binding(source, address=0x10000)
    incomplete = WeightRuntimeBindingManifest(
        model_id=incomplete.model_id,
        revision=incomplete.revision,
        placement_id=incomplete.placement_id,
        instance_id=incomplete.instance_id,
        generation=incomplete.generation,
        lease_id=incomplete.lease_id,
        fragments=runtime_binding(target, address=0x20000).fragments,
    )

    with pytest.raises(ValueError, match="fragment IDs differ"):
        bind_weight_transfer_plan(
            logical,
            source_bindings=(incomplete,),
            target_bindings=(runtime_binding(target, address=0x20000),),
        )


def test_binding_rejects_runtime_range_smaller_than_placement() -> None:
    source = placement("source")
    target = placement("target")
    logical = plan_weight_transfer((source,), (target,))

    with pytest.raises(ValueError, match="byte size differs"):
        bind_weight_transfer_plan(
            logical,
            source_bindings=(
                runtime_binding(source, address=0x10000, nbytes_delta=-2),
            ),
            target_bindings=(runtime_binding(target, address=0x20000),),
        )


def test_binding_rejects_inconsistent_source_generations() -> None:
    source0 = placement(
        "source0",
        rank=WeightParallelRank(dp=0),
    )
    source1 = placement(
        "source1",
        rank=WeightParallelRank(dp=1),
    )
    target0 = placement(
        "target0",
        rank=WeightParallelRank(dp=0),
    )
    target1 = placement(
        "target1",
        rank=WeightParallelRank(dp=1),
    )
    logical = plan_weight_transfer(
        (source0, source1),
        (target0, target1),
    )

    with pytest.raises(ValueError, match="source generations differ"):
        bind_weight_transfer_plan(
            logical,
            source_bindings=(
                runtime_binding(source0, address=0x10000, generation=1),
                runtime_binding(source1, address=0x20000, generation=2),
            ),
            target_bindings=(
                runtime_binding(target0, address=0x30000),
                runtime_binding(target1, address=0x40000),
            ),
        )


def test_binding_rejects_overlapping_target_writes() -> None:
    source_a = placement("source-a", tensor_id="a")
    source_b = placement("source-b", tensor_id="b")
    target_a = placement("target-a", tensor_id="a")
    target_b = placement("target-b", tensor_id="b")
    logical = plan_weight_transfer(
        (source_a, source_b),
        (target_a, target_b),
    )

    with pytest.raises(ValueError, match="overlapping target"):
        bind_weight_transfer_plan(
            logical,
            source_bindings=(
                runtime_binding(source_a, address=0x10000),
                runtime_binding(source_b, address=0x20000),
            ),
            target_bindings=(
                runtime_binding(
                    target_a,
                    address=0x30000,
                    worker_id="target",
                    endpoint="target:1",
                ),
                runtime_binding(
                    target_b,
                    address=0x30008,
                    worker_id="target",
                    endpoint="target:1",
                ),
            ),
        )


def test_binding_rejects_overlapping_runtime_source_and_target_ranges() -> None:
    source = placement("source")
    target = placement("target")
    logical = plan_weight_transfer((source,), (target,))

    with pytest.raises(
        ValueError,
        match="overlapping runtime source and target physical allocations",
    ):
        bind_weight_transfer_plan(
            logical,
            source_bindings=(runtime_binding(source, address=0x10000),),
            target_bindings=(runtime_binding(target, address=0x10008),),
        )


@pytest.mark.parametrize(
    ("worker_id", "endpoint", "device"),
    (
        ("other-worker", "worker:12345", "cuda:0"),
        ("worker", "other-worker:12345", "cuda:0"),
        ("worker", "worker:12345", "cuda:1"),
    ),
)
def test_binding_allows_equal_ranges_in_distinct_runtime_address_spaces(
    worker_id: str,
    endpoint: str,
    device: str,
) -> None:
    source = placement("source")
    target = placement("target")
    logical = plan_weight_transfer((source,), (target,))

    bound = bind_weight_transfer_plan(
        logical,
        source_bindings=(runtime_binding(source, address=0x10000),),
        target_bindings=(
            runtime_binding(
                target,
                address=0x10000,
                worker_id=worker_id,
                endpoint=endpoint,
                device=device,
            ),
        ),
    )

    assert bound.total_bytes == 16


def test_runtime_manifest_compatibility_adapter_round_trips() -> None:
    source = placement("source")
    binding = runtime_binding(source, address=0x10000)
    legacy = compose_weight_runtime_manifest(source, binding)

    parts = runtime_manifest_to_parts(legacy)
    recomposed = compose_weight_runtime_manifest(parts.placement, parts.binding)

    assert parts.binding.placement_id == parts.placement.placement_id
    assert parts.placement.tensors[0].partition_dim is None
    assert parts.placement.tensors[0].shard_dims == ()
    assert recomposed.model_id == legacy.model_id
    assert recomposed.revision == legacy.revision
    assert recomposed.tensors[0].address == legacy.tensors[0].address
    assert recomposed.tensors[0].global_shape == legacy.tensors[0].global_shape


def test_legacy_adapter_keeps_placement_id_across_generations() -> None:
    source = placement("source")
    generation_one = runtime_binding(
        source,
        address=0x10000,
        generation=1,
    )
    generation_two = WeightRuntimeBindingManifest(
        model_id=generation_one.model_id,
        revision=generation_one.revision,
        placement_id=generation_one.placement_id,
        instance_id=generation_one.instance_id,
        generation=2,
        lease_id="lease:2",
        fragments=tuple(
            RuntimeWeightBinding(
                placement_fragment_id=fragment.placement_fragment_id,
                fragment_id=f"{fragment.fragment_id}:generation-2",
                address=fragment.address,
                nbytes=fragment.nbytes,
                storage_offset=fragment.storage_offset,
                device=fragment.device,
                is_contiguous=fragment.is_contiguous,
                worker_id=fragment.worker_id,
                endpoint=fragment.endpoint,
            )
            for fragment in generation_one.fragments
        ),
    )

    first = runtime_manifest_to_parts(
        compose_weight_runtime_manifest(source, generation_one)
    )
    second = runtime_manifest_to_parts(
        compose_weight_runtime_manifest(source, generation_two)
    )

    assert first.placement.placement_id == second.placement.placement_id
    assert (
        first.placement.tensors[0].placement_fragment_id
        == second.placement.tensors[0].placement_fragment_id
    )
    assert (
        first.binding.fragments[0].fragment_id
        != second.binding.fragments[0].fragment_id
    )


def test_binding_rejects_runtime_address_range_overflow() -> None:
    source = placement("source")
    target = placement("target")
    logical = plan_weight_transfer((source,), (target,))

    with pytest.raises(ValueError, match="runtime address range exceeds uint64"):
        bind_weight_transfer_plan(
            logical,
            source_bindings=(
                runtime_binding(
                    source,
                    address=(1 << 64) - source.tensors[0].nbytes + 1,
                ),
            ),
            target_bindings=(runtime_binding(target, address=0x20000),),
        )


def test_storage_binding_rejects_object_range_overflow() -> None:
    with pytest.raises(ValueError, match="storage object range exceeds uint64"):
        WeightStorageFragmentBinding(
            placement_fragment_id="placement-fragment",
            fragment_id="stored-fragment",
            object_key="weights/revision/object",
            object_offset=(1 << 64) - 7,
            nbytes=8,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
