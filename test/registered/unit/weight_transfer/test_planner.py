from __future__ import annotations

import random
from dataclasses import dataclass, replace
from itertools import product
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
from sglang.srt.weight_transfer.binding import bind_weight_transfer_plan
from sglang.srt.weight_transfer.contracts import (
    LogicalWeightTransferRegion,
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
)
from sglang.srt.weight_transfer.planner import (
    WeightPlannerLimits,
    plan_weight_transfer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


MODEL_ID = "qwen-family-moe"
REVISION = "step-42"


@dataclass(frozen=True)
class TensorSpec:
    tensor_id: str
    global_shape: tuple[int, ...]
    shard_dims: tuple[int, ...]
    layer_id: int | None = 0
    expert_id: int | None = None
    dtype: str = "bfloat16"
    itemsize: int = 2
    layout_fingerprint: str = "framework:logical-contiguous:v2"
    aliases: tuple[str, ...] | None = None


Placement = tuple[
    TensorSpec,
    WeightParallelRank,
    tuple[int, ...],
    tuple[int, ...],
]


def tensor_spec(
    tensor_id: str,
    *,
    global_shape: tuple[int, ...],
    shard_dims: tuple[int, ...],
    layer_id: int | None = 0,
    expert_id: int | None = None,
    aliases: tuple[str, ...] | None = None,
) -> TensorSpec:
    return TensorSpec(
        tensor_id=tensor_id,
        global_shape=global_shape,
        shard_dims=shard_dims,
        layer_id=layer_id,
        expert_id=expert_id,
        aliases=aliases,
    )


def build_placements(
    side: str,
    fragments: list[Placement],
    *,
    legacy_partition_only: bool = False,
) -> tuple[WeightPlacementManifest, ...]:
    grouped: dict[WeightParallelRank, list[tuple[TensorSpec, tuple, tuple]]] = {}
    for tensor, rank, offset, shape in fragments:
        grouped.setdefault(rank, []).append((tensor, offset, shape))

    manifests = []
    for rank in sorted(
        grouped,
        key=lambda item: (item.dp, item.pp, item.ep, item.tp),
    ):
        worker_id = f"{side}-d{rank.dp}-p{rank.pp}-e{rank.ep}-t{rank.tp}"
        tensors = []
        for index, (tensor, offset, shape) in enumerate(
            sorted(grouped[rank], key=lambda item: (item[0].tensor_id, item[1]))
        ):
            nbytes = prod(shape) * tensor.itemsize
            tensors.append(
                WeightPlacementTensor(
                    placement_fragment_id=f"{worker_id}:{index}:{tensor.tensor_id}",
                    tensor_id=tensor.tensor_id,
                    runtime_name=tensor.tensor_id,
                    aliases=tensor.aliases or (tensor.tensor_id,),
                    global_shape=tensor.global_shape,
                    global_offset=offset,
                    local_shape=shape,
                    dtype=tensor.dtype,
                    itemsize=tensor.itemsize,
                    partition_dim=(
                        tensor.shard_dims[0] if len(tensor.shard_dims) == 1 else None
                    ),
                    shard_dims=(() if legacy_partition_only else tensor.shard_dims),
                    layer_id=tensor.layer_id,
                    expert_id=tensor.expert_id,
                    layout_fingerprint=tensor.layout_fingerprint,
                    nbytes=nbytes,
                    byte_offset=0,
                    rank=rank,
                )
            )
        manifests.append(
            WeightPlacementManifest(
                model_id=MODEL_ID,
                revision=REVISION,
                placement_id=compute_weight_placement_id(tuple(tensors)),
                tensors=tuple(tensors),
            )
        )
    return tuple(manifests)


def runtime_binding(
    manifest: WeightPlacementManifest,
    *,
    addresses: dict[str, int],
    worker_id: str,
    endpoint: str,
) -> WeightRuntimeBindingManifest:
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id=f"instance:{worker_id}",
        generation=1,
        lease_id=f"lease:{worker_id}",
        fragments=tuple(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=addresses[tensor.placement_fragment_id],
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id=worker_id,
                endpoint=endpoint,
            )
            for tensor in manifest.tensors
        ),
    )


def ep_tp_fragments(
    tensors: tuple[TensorSpec, ...],
    *,
    dp: int,
    pp_owner: dict[str, int],
    ep: int,
    tp: int,
    tp_dim: int,
) -> list[Placement]:
    result = []
    for tensor in tensors:
        assert tensor.global_shape[0] % ep == 0
        assert tensor.global_shape[tp_dim] % tp == 0
        expert_extent = tensor.global_shape[0] // ep
        tp_extent = tensor.global_shape[tp_dim] // tp
        for dp_rank, ep_rank, tp_rank in product(range(dp), range(ep), range(tp)):
            shape = list(tensor.global_shape)
            offset = [0] * len(shape)
            shape[0] = expert_extent
            offset[0] = ep_rank * expert_extent
            shape[tp_dim] = tp_extent
            offset[tp_dim] = tp_rank * tp_extent
            result.append(
                (
                    tensor,
                    WeightParallelRank(
                        dp=dp_rank,
                        pp=pp_owner[tensor.tensor_id],
                        ep=ep_rank,
                        tp=tp_rank,
                    ),
                    tuple(offset),
                    tuple(shape),
                )
            )
    return result


def pp_fragments(
    tensors: tuple[TensorSpec, ...],
    owners: dict[str, int],
) -> list[Placement]:
    return [
        (
            tensor,
            WeightParallelRank(pp=owners[tensor.tensor_id]),
            (0,) * len(tensor.global_shape),
            tensor.global_shape,
        )
        for tensor in tensors
    ]


def placement_tensors(
    manifests: tuple[WeightPlacementManifest, ...],
) -> dict[str, WeightPlacementTensor]:
    return {
        tensor.placement_fragment_id: tensor
        for manifest in manifests
        for tensor in manifest.tensors
    }


def fragment_payload(tensor: WeightPlacementTensor) -> bytearray:
    global_strides = []
    running = 1
    for extent in reversed(tensor.global_shape):
        global_strides.append(running)
        running *= extent
    global_strides.reverse()

    payload = bytearray()
    for local_coordinate in product(*(range(extent) for extent in tensor.local_shape)):
        global_coordinate = tuple(
            begin + local
            for begin, local in zip(
                tensor.global_offset,
                local_coordinate,
                strict=True,
            )
        )
        value = 1 + sum(
            coordinate * stride
            for coordinate, stride in zip(
                global_coordinate,
                global_strides,
                strict=True,
            )
        )
        mask = (1 << (tensor.itemsize * 8)) - 1
        payload.extend((value & mask).to_bytes(tensor.itemsize, "little"))
    return payload


def assert_plan_copies_logical_contents(
    plan,
    source_manifests: tuple[WeightPlacementManifest, ...],
    target_manifests: tuple[WeightPlacementManifest, ...],
) -> None:
    source_tensors = placement_tensors(source_manifests)
    target_tensors = placement_tensors(target_manifests)
    source_payloads = {
        fragment_id: fragment_payload(tensor)
        for fragment_id, tensor in source_tensors.items()
    }
    target_payloads = {
        fragment_id: bytearray(tensor.nbytes)
        for fragment_id, tensor in target_tensors.items()
    }

    for region in plan.regions:
        source = source_payloads[region.source.placement_fragment_id]
        target = target_payloads[region.target.placement_fragment_id]
        for source_offset, target_offset, nbytes in region.iter_segments():
            target[target_offset : target_offset + nbytes] = source[
                source_offset : source_offset + nbytes
            ]

    for fragment_id, tensor in target_tensors.items():
        assert target_payloads[fragment_id] == fragment_payload(tensor)


def brute_force_region_keys(
    source_manifests: tuple[WeightPlacementManifest, ...],
    target_manifests: tuple[WeightPlacementManifest, ...],
) -> set[tuple[str, str, tuple[int, ...], tuple[int, ...]]]:
    result = set()
    for source in placement_tensors(source_manifests).values():
        for target in placement_tensors(target_manifests).values():
            if source.tensor_id != target.tensor_id:
                continue
            overlap_offset = tuple(
                max(source_begin, target_begin)
                for source_begin, target_begin in zip(
                    source.global_offset,
                    target.global_offset,
                    strict=True,
                )
            )
            overlap_end = tuple(
                min(
                    source_begin + source_extent,
                    target_begin + target_extent,
                )
                for source_begin, source_extent, target_begin, target_extent in zip(
                    source.global_offset,
                    source.local_shape,
                    target.global_offset,
                    target.local_shape,
                    strict=True,
                )
            )
            overlap_shape = tuple(
                end - begin
                for begin, end in zip(
                    overlap_offset,
                    overlap_end,
                    strict=True,
                )
            )
            if all(extent > 0 for extent in overlap_shape):
                result.add(
                    (
                        source.placement_fragment_id,
                        target.placement_fragment_id,
                        overlap_offset,
                        overlap_shape,
                    )
                )
    return result


def grid_fragments(
    tensor: TensorSpec,
    ranges_by_dim: tuple[tuple[tuple[int, int], ...], ...],
) -> list[Placement]:
    result = []
    for rank, ranges in enumerate(product(*ranges_by_dim)):
        offset = tuple(begin for begin, _ in ranges)
        shape = tuple(end - begin for begin, end in ranges)
        result.append(
            (
                tensor,
                WeightParallelRank(tp=rank),
                offset,
                shape,
            )
        )
    return result


def random_axis_ranges(
    rng: random.Random,
    shape: tuple[int, ...],
    shard_dims: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    result = []
    for dim, extent in enumerate(shape):
        if dim not in shard_dims:
            result.append(((0, extent),))
            continue
        cut_count = rng.randint(1, min(2, extent - 1))
        cuts = sorted(rng.sample(range(1, extent), cut_count))
        points = (0, *cuts, extent)
        result.append(tuple(zip(points, points[1:])))
    return tuple(result)


@pytest.mark.parametrize(("source_tp", "target_tp"), [(4, 8), (8, 4)])
def test_tp_split_and_merge_preserve_logical_bytes(
    source_tp: int,
    target_tp: int,
) -> None:
    tensor = tensor_spec(
        "layers.0.mlp.down_proj.weight",
        global_shape=(64, 16),
        shard_dims=(0,),
    )
    owner = {tensor.tensor_id: 0}
    sources = build_placements(
        "source",
        ep_tp_fragments(
            (tensor,),
            dp=1,
            pp_owner=owner,
            ep=1,
            tp=source_tp,
            tp_dim=0,
        ),
    )
    targets = build_placements(
        "target",
        ep_tp_fragments(
            (tensor,),
            dp=1,
            pp_owner=owner,
            ep=1,
            tp=target_tp,
            tp_dim=0,
        ),
    )

    plan = plan_weight_transfer(sources, targets)

    assert len(plan.regions) == max(source_tp, target_tp)
    assert all(
        isinstance(region, LogicalWeightTransferRegion) for region in plan.regions
    )
    assert_plan_copies_logical_contents(plan, sources, targets)


def test_legacy_partition_dim_supports_source_target_cross_dimension() -> None:
    source_tensor = tensor_spec(
        "opaque.weight",
        global_shape=(8, 6, 4),
        shard_dims=(0,),
    )
    target_tensor = tensor_spec(
        source_tensor.tensor_id,
        global_shape=source_tensor.global_shape,
        shard_dims=(2,),
    )
    sources = build_placements(
        "source",
        [
            (
                source_tensor,
                WeightParallelRank(tp=rank),
                (rank * 4, 0, 0),
                (4, 6, 4),
            )
            for rank in range(2)
        ],
        legacy_partition_only=True,
    )
    targets = build_placements(
        "target",
        [
            (
                target_tensor,
                WeightParallelRank(tp=rank),
                (0, 0, rank * 2),
                (8, 6, 2),
            )
            for rank in range(2)
        ],
        legacy_partition_only=True,
    )

    plan = plan_weight_transfer(sources, targets)

    assert {
        (tensor.partition_dim, tensor.shard_dims)
        for manifest in sources
        for tensor in manifest.tensors
    } == {(0, ())}
    assert {
        (tensor.partition_dim, tensor.shard_dims)
        for manifest in targets
        for tensor in manifest.tensors
    } == {(2, ())}
    assert len(plan.regions) == 4
    assert_plan_copies_logical_contents(plan, sources, targets)


@pytest.mark.parametrize(
    ("source_owners", "target_owners", "expected_routes"),
    [
        (
            {
                "layers.0.weight": 0,
                "layers.1.weight": 0,
                "layers.2.weight": 1,
                "layers.3.weight": 1,
            },
            {
                "layers.0.weight": 0,
                "layers.1.weight": 1,
                "layers.2.weight": 2,
                "layers.3.weight": 3,
            },
            {(0, 0), (0, 1), (1, 2), (1, 3)},
        ),
        (
            {
                "layers.0.weight": 0,
                "layers.1.weight": 1,
                "layers.2.weight": 2,
                "layers.3.weight": 3,
            },
            {
                "layers.0.weight": 0,
                "layers.1.weight": 0,
                "layers.2.weight": 1,
                "layers.3.weight": 1,
            },
            {(0, 0), (1, 0), (2, 1), (3, 1)},
        ),
    ],
)
def test_pp_routes_follow_explicit_tensor_ownership(
    source_owners: dict[str, int],
    target_owners: dict[str, int],
    expected_routes: set[tuple[int, int]],
) -> None:
    tensors = tuple(
        tensor_spec(
            f"layers.{layer}.weight",
            global_shape=(8,),
            shard_dims=(),
            layer_id=layer,
        )
        for layer in range(4)
    )
    sources = build_placements("source", pp_fragments(tensors, source_owners))
    targets = build_placements("target", pp_fragments(tensors, target_owners))

    plan = plan_weight_transfer(sources, targets)

    assert {
        (route.source_pp, route.target_pp) for route in plan.pipeline_routes
    } == expected_routes
    assert sorted(
        index for route in plan.pipeline_routes for index in route.region_indices
    ) == list(range(len(plan.regions)))
    assert_plan_copies_logical_contents(plan, sources, targets)


@pytest.mark.parametrize(("source_ep", "target_ep"), [(8, 2), (2, 8)])
def test_ep_reshard_uses_leading_expert_coordinate(
    source_ep: int,
    target_ep: int,
) -> None:
    tensor = tensor_spec(
        "layers.0.experts.w1",
        global_shape=(8, 4, 2),
        shard_dims=(0,),
    )
    owners = {tensor.tensor_id: 0}
    sources = build_placements(
        "source",
        ep_tp_fragments(
            (tensor,),
            dp=1,
            pp_owner=owners,
            ep=source_ep,
            tp=1,
            tp_dim=1,
        ),
    )
    targets = build_placements(
        "target",
        ep_tp_fragments(
            (tensor,),
            dp=1,
            pp_owner=owners,
            ep=target_ep,
            tp=1,
            tp_dim=1,
        ),
    )

    plan = plan_weight_transfer(sources, targets)

    assert {region.source.rank.ep for region in plan.regions} == set(range(source_ep))
    assert {region.target.rank.ep for region in plan.regions} == set(range(target_ep))
    assert_plan_copies_logical_contents(plan, sources, targets)


@pytest.mark.parametrize(("source_ep", "target_ep"), [(8, 2), (2, 8)])
def test_explicit_expert_id_routes_each_logical_expert_to_its_ep_owner(
    source_ep: int,
    target_ep: int,
) -> None:
    tensors = tuple(
        tensor_spec(
            f"opaque-{expert_id}",
            global_shape=(4, 2),
            shard_dims=(),
            expert_id=expert_id,
        )
        for expert_id in range(16)
    )
    sources = build_placements(
        "source",
        [
            (
                tensor,
                WeightParallelRank(ep=tensor.expert_id % source_ep),
                (0, 0),
                tensor.global_shape,
            )
            for tensor in tensors
        ],
    )
    targets = build_placements(
        "target",
        [
            (
                tensor,
                WeightParallelRank(ep=tensor.expert_id % target_ep),
                (0, 0),
                tensor.global_shape,
            )
            for tensor in tensors
        ],
    )

    plan = plan_weight_transfer(sources, targets)
    source_by_fragment = placement_tensors(sources)
    target_by_fragment = placement_tensors(targets)

    assert len(plan.regions) == len(tensors)
    for region in plan.regions:
        source = source_by_fragment[region.source.placement_fragment_id]
        target = target_by_fragment[region.target.placement_fragment_id]
        assert source.expert_id == target.expert_id
        assert region.source.rank.ep == source.expert_id % source_ep
        assert region.target.rank.ep == target.expert_id % target_ep
    assert_plan_copies_logical_contents(plan, sources, targets)


@pytest.mark.parametrize("target_dim", [1, 2])
def test_ep_tp_cross_dimension_reshard(target_dim: int) -> None:
    source_tensor = tensor_spec(
        "layers.0.experts.w1",
        global_shape=(4, 6, 8),
        shard_dims=(0,),
    )
    target_tensor = tensor_spec(
        source_tensor.tensor_id,
        global_shape=source_tensor.global_shape,
        shard_dims=(target_dim,),
    )
    sources = build_placements(
        "source",
        [
            (
                source_tensor,
                WeightParallelRank(ep=rank),
                (rank * 2, 0, 0),
                (2, 6, 8),
            )
            for rank in range(2)
        ],
    )
    target_fragments = []
    for rank in range(2):
        shape = list(target_tensor.global_shape)
        offset = [0, 0, 0]
        shape[target_dim] //= 2
        offset[target_dim] = rank * shape[target_dim]
        target_fragments.append(
            (
                target_tensor,
                WeightParallelRank(tp=rank),
                tuple(offset),
                tuple(shape),
            )
        )
    targets = build_placements("target", target_fragments)

    plan = plan_weight_transfer(sources, targets)

    assert len(plan.regions) == 4
    selected = next(
        region
        for region in plan.regions
        if region.source.rank.ep == 0 and region.target.rank.tp == 1
    )
    if target_dim == 1:
        assert selected.overlap_offset == (0, 3, 0)
        assert selected.overlap_shape == (2, 3, 8)
        assert selected.source_base_offset == 48
        assert selected.target_base_offset == 0
        assert selected.inner_bytes == 48
        assert selected.outer_loop_counts == (2,)
        assert selected.source_strides == (96,)
        assert selected.target_strides == (48,)
    else:
        assert selected.overlap_offset == (0, 0, 4)
        assert selected.overlap_shape == (2, 6, 4)
        assert selected.source_base_offset == 8
        assert selected.target_base_offset == 0
        assert selected.inner_bytes == 8
        assert selected.outer_loop_counts == (2, 6)
        assert selected.source_strides == (96, 16)
        assert selected.target_strides == (48, 8)
    assert_plan_copies_logical_contents(plan, sources, targets)


def test_four_axis_reshard_is_complete_and_deterministic() -> None:
    source_tensors = tuple(
        tensor_spec(
            f"layers.{layer}.experts.w1",
            global_shape=(8, 16, 16),
            shard_dims=(0, 1),
            layer_id=layer,
        )
        for layer in range(4)
    )
    target_tensors = tuple(
        tensor_spec(
            tensor.tensor_id,
            global_shape=tensor.global_shape,
            shard_dims=(0, 2),
            layer_id=tensor.layer_id,
        )
        for tensor in source_tensors
    )
    source_owners = {
        tensor.tensor_id: tensor.layer_id // 2 for tensor in source_tensors
    }
    target_owners = {tensor.tensor_id: tensor.layer_id for tensor in target_tensors}
    sources = build_placements(
        "source",
        ep_tp_fragments(
            source_tensors,
            dp=2,
            pp_owner=source_owners,
            ep=8,
            tp=4,
            tp_dim=1,
        ),
    )
    targets = build_placements(
        "target",
        ep_tp_fragments(
            target_tensors,
            dp=4,
            pp_owner=target_owners,
            ep=2,
            tp=8,
            tp_dim=2,
        ),
    )

    first = plan_weight_transfer(sources, targets)
    second = plan_weight_transfer(tuple(reversed(sources)), tuple(reversed(targets)))

    assert first.digest == second.digest
    assert first.total_bytes == 4 * 8 * 16 * 16 * 2 * 4
    assert {region.source.rank.dp for region in first.regions} == {0, 1}
    assert {region.target.rank.dp for region in first.regions} == {0, 1, 2, 3}
    assert {region.source.rank.tp for region in first.regions} == set(range(4))
    assert {region.target.rank.tp for region in first.regions} == set(range(8))
    assert {region.source.rank.ep for region in first.regions} == set(range(8))
    assert {region.target.rank.ep for region in first.regions} == {0, 1}
    assert {(route.source_pp, route.target_pp) for route in first.pipeline_routes} == {
        (0, 0),
        (0, 1),
        (1, 2),
        (1, 3),
    }
    assert_plan_copies_logical_contents(first, sources, targets)


def test_plan_digest_commits_tensor_semantics() -> None:
    float16 = TensorSpec(
        tensor_id="weight",
        global_shape=(8,),
        shard_dims=(0,),
        dtype="float16",
    )
    bfloat16 = TensorSpec(
        tensor_id="weight",
        global_shape=(8,),
        shard_dims=(0,),
        dtype="bfloat16",
    )

    def build_plan(spec: TensorSpec):
        sources = build_placements(
            "source",
            [
                (spec, WeightParallelRank(tp=0), (0,), (4,)),
                (spec, WeightParallelRank(tp=1), (4,), (4,)),
            ],
        )
        targets = build_placements(
            "target",
            [
                (spec, WeightParallelRank(tp=0), (0,), (4,)),
                (spec, WeightParallelRank(tp=1), (4,), (4,)),
            ],
        )
        return plan_weight_transfer(sources, targets)

    assert build_plan(float16).digest != build_plan(bfloat16).digest


def test_cross_dimension_plan_keeps_operation_and_region_counts_bounded(
    monkeypatch,
) -> None:
    source_tensor = tensor_spec(
        "layers.0.experts.w1",
        global_shape=(8, 8192, 8192),
        shard_dims=(0,),
    )
    target_tensor = tensor_spec(
        source_tensor.tensor_id,
        global_shape=source_tensor.global_shape,
        shard_dims=(2,),
    )
    sources = build_placements(
        "source",
        [
            (
                source_tensor,
                WeightParallelRank(ep=rank),
                (rank, 0, 0),
                (1, 8192, 8192),
            )
            for rank in range(8)
        ],
    )
    targets = build_placements(
        "target",
        [
            (
                target_tensor,
                WeightParallelRank(tp=rank),
                (0, 0, rank * 1024),
                (8, 8192, 1024),
            )
            for rank in range(8)
        ],
    )

    def reject_segment_expansion(_self):
        raise AssertionError("planner must not expand a region into segments")

    monkeypatch.setattr(
        LogicalWeightTransferRegion,
        "iter_segments",
        reject_segment_expansion,
    )

    plan = plan_weight_transfer(sources, targets)

    assert len(plan.regions) == 64
    assert len(plan.operations) == len(plan.regions)
    assert {region.segment_count for region in plan.regions} == {8192}
    assert plan.total_segments == 64 * 8192
    assert len(plan.operations) < plan.total_segments


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_tensor_ndim", 2, "tensor rank limit"),
        ("max_candidate_visits", 3, "candidate visit limit"),
        ("max_regions", 3, "region limit"),
        ("max_segments_per_region", 11, "per-region segment limit"),
        ("max_total_segments", 47, "total segment limit"),
    ],
)
def test_planner_enforces_resource_limits(
    field: str,
    value: int,
    message: str,
) -> None:
    source_tensor = tensor_spec(
        "layers.0.experts.w1",
        global_shape=(4, 6, 8),
        shard_dims=(0,),
    )
    target_tensor = tensor_spec(
        source_tensor.tensor_id,
        global_shape=source_tensor.global_shape,
        shard_dims=(2,),
    )
    sources = build_placements(
        "source",
        [
            (
                source_tensor,
                WeightParallelRank(ep=rank),
                (rank * 2, 0, 0),
                (2, 6, 8),
            )
            for rank in range(2)
        ],
    )
    targets = build_placements(
        "target",
        [
            (
                target_tensor,
                WeightParallelRank(tp=rank),
                (0, 0, rank * 4),
                (4, 6, 4),
            )
            for rank in range(2)
        ],
    )
    limits = replace(WeightPlannerLimits(), **{field: value})

    with pytest.raises(ValueError, match=message):
        plan_weight_transfer(sources, targets, limits=limits)


def test_candidate_index_bounds_overlap_checks(monkeypatch) -> None:
    from sglang.srt.weight_transfer import planner as planner_module

    parts = 128
    tensor = tensor_spec(
        "weight",
        global_shape=(parts * 8,),
        shard_dims=(0,),
    )
    sources = build_placements(
        "source",
        [
            (
                tensor,
                WeightParallelRank(tp=rank),
                (rank * 8,),
                (8,),
            )
            for rank in range(parts)
        ],
    )
    targets = build_placements(
        "target",
        [
            (
                tensor,
                WeightParallelRank(tp=rank),
                (rank * 8,),
                (8,),
            )
            for rank in range(parts)
        ],
    )
    original_overlap_box = planner_module._overlap_box
    overlap_checks = 0

    def counting_overlap_box(source, target):
        nonlocal overlap_checks
        overlap_checks += 1
        return original_overlap_box(source, target)

    monkeypatch.setattr(
        planner_module,
        "_overlap_box",
        counting_overlap_box,
    )

    plan = plan_weight_transfer(sources, targets)

    assert len(plan.regions) == parts
    assert overlap_checks <= parts * 2


def test_nd_candidate_index_is_output_sensitive(monkeypatch) -> None:
    from sglang.srt.weight_transfer import planner as planner_module

    parts = 32
    source_tensor = tensor_spec(
        "weight",
        global_shape=(parts, parts),
        shard_dims=(0, 1),
    )
    target_tensor = tensor_spec(
        "weight",
        global_shape=(parts, parts),
        shard_dims=(1,),
    )
    sources = build_placements(
        "source",
        [
            (
                source_tensor,
                WeightParallelRank(tp=column, ep=row),
                (row, column),
                (1, 1),
            )
            for row in range(parts)
            for column in range(parts)
        ],
    )
    targets = build_placements(
        "target",
        [
            (
                target_tensor,
                WeightParallelRank(tp=column),
                (0, column),
                (parts, 1),
            )
            for column in range(parts)
        ],
    )
    original_overlap_box = planner_module._overlap_box
    overlap_checks = 0

    def counting_overlap_box(source, target):
        nonlocal overlap_checks
        overlap_checks += 1
        return original_overlap_box(source, target)

    monkeypatch.setattr(
        planner_module,
        "_overlap_box",
        counting_overlap_box,
    )

    plan = plan_weight_transfer(sources, targets)

    assert len(plan.regions) == parts * parts
    assert overlap_checks <= 2 * len(plan.regions)


def test_candidate_index_matches_brute_force_for_random_1_to_8d_boxes() -> None:
    rng = random.Random(0x5EED)
    cases = 0
    for ndim in range(1, 9):
        for trial in range(8):
            shape = tuple(rng.randint(2, 5) for _ in range(ndim))
            source_dim_count = rng.randint(1, min(3, ndim))
            target_dim_count = rng.randint(1, min(3, ndim))
            source_dims = tuple(sorted(rng.sample(range(ndim), source_dim_count)))
            target_dims = tuple(sorted(rng.sample(range(ndim), target_dim_count)))
            source_tensor = tensor_spec(
                f"random-{ndim}-{trial}",
                global_shape=shape,
                shard_dims=source_dims,
            )
            target_tensor = tensor_spec(
                source_tensor.tensor_id,
                global_shape=shape,
                shard_dims=target_dims,
            )
            sources = build_placements(
                "source",
                grid_fragments(
                    source_tensor,
                    random_axis_ranges(rng, shape, source_dims),
                ),
            )
            targets = build_placements(
                "target",
                grid_fragments(
                    target_tensor,
                    random_axis_ranges(rng, shape, target_dims),
                ),
            )

            plan = plan_weight_transfer(sources, targets)
            actual = {
                (
                    region.source.placement_fragment_id,
                    region.target.placement_fragment_id,
                    region.overlap_offset,
                    region.overlap_shape,
                )
                for region in plan.regions
            }

            assert actual == brute_force_region_keys(sources, targets)
            assert plan.total_bytes == prod(shape) * source_tensor.itemsize
            cases += 1

    assert cases == 64


def test_aliases_with_the_same_physical_source_are_deduplicated() -> None:
    aliases = ("shared.weight", "shared.weight.alias")
    tensor = tensor_spec(
        "shared",
        global_shape=(4,),
        shard_dims=(),
        aliases=aliases,
    )
    sources = build_placements(
        "source",
        [(tensor, WeightParallelRank(), (0,), (4,))],
    )
    targets = build_placements(
        "target",
        [
            (tensor, WeightParallelRank(), (0,), (4,)),
            (tensor, WeightParallelRank(), (0,), (4,)),
        ],
    )
    logical = plan_weight_transfer(sources, targets)
    source_fragment = sources[0].tensors[0].placement_fragment_id
    target_fragments = [tensor.placement_fragment_id for tensor in targets[0].tensors]

    bound = bind_weight_transfer_plan(
        logical,
        source_bindings=(
            runtime_binding(
                sources[0],
                addresses={source_fragment: 0x10000},
                worker_id="source",
                endpoint="source:1",
            ),
        ),
        target_bindings=(
            runtime_binding(
                targets[0],
                addresses={fragment_id: 0x20000 for fragment_id in target_fragments},
                worker_id="target",
                endpoint="target:1",
            ),
        ),
    )

    assert len(logical.regions) == 2
    assert len(bound.operations) == 1
    assert bound.total_bytes == tensor.itemsize * prod(tensor.global_shape)


def test_aliases_with_different_physical_sources_are_not_deduplicated() -> None:
    aliases = ("shared.weight", "shared.weight.alias")
    tensors = tuple(
        tensor_spec(
            tensor_id,
            global_shape=(4,),
            shard_dims=(),
            aliases=aliases,
        )
        for tensor_id in ("logical-a", "logical-b")
    )
    sources = build_placements(
        "source",
        [(tensor, WeightParallelRank(), (0,), (4,)) for tensor in tensors],
    )
    targets = build_placements(
        "target",
        [(tensor, WeightParallelRank(), (0,), (4,)) for tensor in tensors],
    )
    source_fragments = [tensor.placement_fragment_id for tensor in sources[0].tensors]
    target_fragments = [tensor.placement_fragment_id for tensor in targets[0].tensors]
    logical = plan_weight_transfer(sources, targets)

    with pytest.raises(ValueError, match="overlapping target writes"):
        bind_weight_transfer_plan(
            logical,
            source_bindings=(
                runtime_binding(
                    sources[0],
                    addresses={
                        fragment_id: 0x10000 + index * 0x1000
                        for index, fragment_id in enumerate(source_fragments)
                    },
                    worker_id="source",
                    endpoint="source:1",
                ),
            ),
            target_bindings=(
                runtime_binding(
                    targets[0],
                    addresses={
                        fragment_id: 0x20000 for fragment_id in target_fragments
                    },
                    worker_id="target",
                    endpoint="target:1",
                ),
            ),
        )


def test_storage_aliases_with_matching_checksums_are_deduplicated() -> None:
    aliases = ("shared.weight", "shared.weight.alias")
    tensors = tuple(
        tensor_spec(
            tensor_id,
            global_shape=(4,),
            shard_dims=(),
            aliases=aliases,
        )
        for tensor_id in ("logical-a", "logical-b")
    )
    sources = build_placements(
        "source",
        [(tensor, WeightParallelRank(), (0,), (4,)) for tensor in tensors],
    )
    targets = build_placements(
        "target",
        [(tensor, WeightParallelRank(), (0,), (4,)) for tensor in tensors],
    )
    source_fragments = tuple(sources[0].tensors)
    target_fragments = tuple(targets[0].tensors)
    logical = plan_weight_transfer(sources, targets)
    checksum = "sha256:" + "1" * 64
    source_binding = WeightStorageBindingManifest(
        model_id=sources[0].model_id,
        revision=sources[0].revision,
        placement_id=sources[0].placement_id,
        storage_id="weights/revision",
        provider="mooncake-store",
        fragments=tuple(
            WeightStorageFragmentBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"stored:{tensor.placement_fragment_id}",
                object_key=f"weights/revision/{index}",
                object_offset=0,
                nbytes=tensor.nbytes,
                checksum=checksum,
            )
            for index, tensor in enumerate(source_fragments)
        ),
    )

    bound = bind_weight_transfer_plan(
        logical,
        source_bindings=(source_binding,),
        target_bindings=(
            runtime_binding(
                targets[0],
                addresses={
                    tensor.placement_fragment_id: 0x20000 for tensor in target_fragments
                },
                worker_id="target",
                endpoint="target:1",
            ),
        ),
    )

    assert len(logical.regions) == 2
    assert len(bound.operations) == 1
    assert bound.total_bytes == tensors[0].itemsize * prod(tensors[0].global_shape)


def test_storage_aliases_with_different_checksums_are_not_deduplicated() -> None:
    aliases = ("shared.weight", "shared.weight.alias")
    tensors = tuple(
        tensor_spec(
            tensor_id,
            global_shape=(4,),
            shard_dims=(),
            aliases=aliases,
        )
        for tensor_id in ("logical-a", "logical-b")
    )
    sources = build_placements(
        "source",
        [(tensor, WeightParallelRank(), (0,), (4,)) for tensor in tensors],
    )
    targets = build_placements(
        "target",
        [(tensor, WeightParallelRank(), (0,), (4,)) for tensor in tensors],
    )
    source_fragments = tuple(sources[0].tensors)
    target_fragments = tuple(targets[0].tensors)
    source_binding = WeightStorageBindingManifest(
        model_id=sources[0].model_id,
        revision=sources[0].revision,
        placement_id=sources[0].placement_id,
        storage_id="weights/revision",
        provider="mooncake-store",
        fragments=tuple(
            WeightStorageFragmentBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"stored:{tensor.placement_fragment_id}",
                object_key=f"weights/revision/{index}",
                object_offset=0,
                nbytes=tensor.nbytes,
                checksum="sha256:" + str(index) * 64,
            )
            for index, tensor in enumerate(source_fragments)
        ),
    )

    with pytest.raises(ValueError, match="overlapping target writes"):
        bind_weight_transfer_plan(
            plan_weight_transfer(sources, targets),
            source_bindings=(source_binding,),
            target_bindings=(
                runtime_binding(
                    targets[0],
                    addresses={
                        tensor.placement_fragment_id: 0x20000
                        for tensor in target_fragments
                    },
                    worker_id="target",
                    endpoint="target:1",
                ),
            ),
        )


def test_planner_rejects_mixed_rank_placement() -> None:
    first = tensor_spec("first", global_shape=(8,), shard_dims=())
    second = tensor_spec("second", global_shape=(8,), shard_dims=())
    source_parts = build_placements(
        "source-part",
        [
            (first, WeightParallelRank(pp=0), (0,), (8,)),
            (second, WeightParallelRank(pp=1), (0,), (8,)),
        ],
    )
    target_parts = build_placements(
        "target-part",
        [
            (first, WeightParallelRank(pp=0), (0,), (8,)),
            (second, WeightParallelRank(pp=1), (0,), (8,)),
        ],
    )
    source_tensors = tuple(
        tensor for placement in source_parts for tensor in placement.tensors
    )
    source = WeightPlacementManifest(
        model_id=source_parts[0].model_id,
        revision=source_parts[0].revision,
        placement_id=compute_weight_placement_id(tuple(source_tensors)),
        tensors=source_tensors,
    )
    target_tensors = tuple(
        tensor for placement in target_parts for tensor in placement.tensors
    )
    target = WeightPlacementManifest(
        model_id=target_parts[0].model_id,
        revision=target_parts[0].revision,
        placement_id=compute_weight_placement_id(tuple(target_tensors)),
        tensors=target_tensors,
    )

    with pytest.raises(ValueError, match="mixes parallel ranks"):
        plan_weight_transfer((source,), (target,))


@pytest.mark.parametrize("axis", ("dp", "tp", "pp", "ep"))
def test_planner_rejects_negative_parallel_rank(axis: str) -> None:
    tensor = tensor_spec("weight", global_shape=(8,), shard_dims=())
    values = {"dp": 0, "tp": 0, "pp": 0, "ep": 0}
    values[axis] = -1
    source = build_placements(
        "source",
        [(tensor, WeightParallelRank(**values), (0,), (8,))],
    )
    target = build_placements(
        "target",
        [(tensor, WeightParallelRank(), (0,), (8,))],
    )

    with pytest.raises(ValueError, match="parallel rank"):
        plan_weight_transfer(source, target)


def test_planner_fails_closed_on_descriptor_mismatch() -> None:
    source = tensor_spec("weight", global_shape=(8, 8), shard_dims=(0,))
    target = TensorSpec(
        tensor_id=source.tensor_id,
        global_shape=source.global_shape,
        shard_dims=(0,),
        dtype="float16",
    )
    sources = build_placements(
        "source",
        ep_tp_fragments(
            (source,),
            dp=1,
            pp_owner={source.tensor_id: 0},
            ep=1,
            tp=2,
            tp_dim=0,
        ),
    )
    targets = build_placements(
        "target",
        ep_tp_fragments(
            (target,),
            dp=1,
            pp_owner={target.tensor_id: 0},
            ep=1,
            tp=2,
            tp_dim=0,
        ),
    )

    with pytest.raises(ValueError, match="descriptor mismatch"):
        plan_weight_transfer(sources, targets)


def test_planner_fails_closed_on_target_coverage_gap() -> None:
    tensor = tensor_spec("weight", global_shape=(8,), shard_dims=(0,))
    sources = build_placements(
        "source",
        [
            (tensor, WeightParallelRank(tp=0), (0,), (4,)),
            (tensor, WeightParallelRank(tp=1), (4,), (4,)),
        ],
    )
    targets = build_placements(
        "target",
        [
            (tensor, WeightParallelRank(tp=0), (0,), (3,)),
            (tensor, WeightParallelRank(tp=1), (4,), (4,)),
        ],
    )

    with pytest.raises(ValueError, match="not fully covered"):
        plan_weight_transfer(sources, targets)


def test_planner_expected_target_topology_detects_missing_dp_rank() -> None:
    tensor = tensor_spec("weight", global_shape=(8,), shard_dims=())
    sources = build_placements(
        "source",
        [(tensor, WeightParallelRank(), (0,), (8,))],
    )
    targets = build_placements(
        "target",
        [(tensor, WeightParallelRank(dp=dp_rank), (0,), (8,)) for dp_rank in range(3)],
    )
    expected_target_topology = tuple(
        WeightParallelRank(dp=dp_rank) for dp_rank in range(4)
    )

    with pytest.raises(
        ValueError,
        match=r"target placement topology mismatch.*missing.*dp=3",
    ):
        plan_weight_transfer(
            sources,
            targets,
            expected_target_topology=expected_target_topology,
        )


def test_planner_accepts_complete_expected_target_topology() -> None:
    tensor = tensor_spec("weight", global_shape=(8,), shard_dims=())
    sources = build_placements(
        "source",
        [(tensor, WeightParallelRank(), (0,), (8,))],
    )
    targets = build_placements(
        "target",
        [(tensor, WeightParallelRank(dp=dp_rank), (0,), (8,)) for dp_rank in range(4)],
    )
    expected_target_topology = tuple(
        WeightParallelRank(dp=dp_rank) for dp_rank in range(4)
    )

    plan = plan_weight_transfer(
        sources,
        targets,
        expected_target_topology=expected_target_topology,
    )

    assert {region.target.rank.dp for region in plan.regions} == {0, 1, 2, 3}


def test_planner_without_expected_topology_plans_supplied_placements() -> None:
    tensor = tensor_spec("weight", global_shape=(8,), shard_dims=())
    sources = build_placements(
        "source",
        [(tensor, WeightParallelRank(), (0,), (8,))],
    )
    targets = build_placements(
        "target",
        [(tensor, WeightParallelRank(dp=dp_rank), (0,), (8,)) for dp_rank in range(3)],
    )

    plan = plan_weight_transfer(sources, targets)

    assert {region.target.rank.dp for region in plan.regions} == {0, 1, 2}


def test_planner_ignores_incomplete_source_dp_replica() -> None:
    tensor = tensor_spec("weight", global_shape=(8,), shard_dims=(0,))
    sources = build_placements(
        "source",
        [
            (tensor, WeightParallelRank(dp=0, tp=0), (0,), (4,)),
            (tensor, WeightParallelRank(dp=0, tp=1), (4,), (4,)),
            (tensor, WeightParallelRank(dp=1, tp=0), (0,), (4,)),
        ],
    )
    targets = build_placements(
        "target",
        [
            (tensor, WeightParallelRank(dp=0), (0,), (8,)),
            (tensor, WeightParallelRank(dp=1), (0,), (8,)),
        ],
    )

    plan = plan_weight_transfer(sources, targets)

    assert {region.source.rank.dp for region in plan.regions} == {0}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
