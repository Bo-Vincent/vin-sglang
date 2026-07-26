from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import prod
from typing import Sequence

from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.contracts import (
    LogicalPlacementFragment,
    LogicalWeightTransferPlan,
    LogicalWeightTransferRegion,
    PipelineRouteGroup,
    PlacementExecutorGroup,
    build_region,
)

TensorOwner = tuple[int, int | None]
BoxGeometry = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class WeightPlannerLimits:
    max_tensor_ndim: int = 8
    max_candidate_visits: int = 10_000_000
    max_regions: int = 1_000_000
    max_segments_per_region: int = 1_000_000
    max_total_segments: int = 10_000_000

    def __post_init__(self) -> None:
        for name in (
            "max_tensor_ndim",
            "max_candidate_visits",
            "max_regions",
            "max_segments_per_region",
            "max_total_segments",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_WEIGHT_PLANNER_LIMITS = WeightPlannerLimits()


@dataclass(frozen=True)
class _TensorDescriptor:
    tensor_id: str
    global_shape: tuple[int, ...]
    dtype: str
    itemsize: int
    partition_dim: int | None
    shard_dims: tuple[int, ...]
    layer_id: int | None
    expert_id: int | None
    layout_fingerprint: str


@dataclass(frozen=True)
class _CollectedFragment:
    logical: LogicalPlacementFragment
    descriptor: _TensorDescriptor

    @property
    def placement_id(self) -> str:
        return self.logical.placement_id

    @property
    def placement_fragment_id(self) -> str:
        return self.logical.placement_fragment_id

    @property
    def tensor_id(self) -> str:
        return self.logical.tensor_id

    @property
    def global_offset(self) -> tuple[int, ...]:
        return self.logical.global_offset

    @property
    def local_shape(self) -> tuple[int, ...]:
        return self.logical.local_shape

    @property
    def rank(self) -> WeightParallelRank:
        return self.logical.rank


@dataclass(frozen=True)
class _IntervalEntry:
    begin: int
    end: int
    geometry: BoxGeometry


@dataclass(frozen=True)
class _IntervalNode:
    center: int
    by_begin: tuple[_IntervalEntry, ...]
    begins: tuple[int, ...]
    by_end: tuple[_IntervalEntry, ...]
    ends: tuple[int, ...]
    left: _IntervalNode | None
    right: _IntervalNode | None

    @classmethod
    def build(cls, entries: Sequence[_IntervalEntry]) -> _IntervalNode | None:
        if not entries:
            return None
        midpoints = sorted(
            entry.begin + (entry.end - entry.begin) // 2 for entry in entries
        )
        center = midpoints[len(midpoints) // 2]
        left_entries = []
        right_entries = []
        crossing = []
        for entry in entries:
            if entry.end <= center:
                left_entries.append(entry)
            elif entry.begin > center:
                right_entries.append(entry)
            else:
                crossing.append(entry)
        by_begin = tuple(
            sorted(
                crossing,
                key=lambda item: (
                    item.begin,
                    item.end,
                    item.geometry,
                ),
            )
        )
        by_end = tuple(
            sorted(
                crossing,
                key=lambda item: (
                    item.end,
                    item.begin,
                    item.geometry,
                ),
            )
        )
        return cls(
            center=center,
            by_begin=by_begin,
            begins=tuple(entry.begin for entry in by_begin),
            by_end=by_end,
            ends=tuple(entry.end for entry in by_end),
            left=cls.build(left_entries),
            right=cls.build(right_entries),
        )

    def query(
        self,
        begin: int,
        end: int,
        result: list[_IntervalEntry],
    ) -> None:
        if end <= self.center:
            result.extend(self.by_begin[: bisect_left(self.begins, end)])
            if self.left is not None:
                self.left.query(begin, end, result)
            return
        if begin > self.center:
            result.extend(self.by_end[bisect_right(self.ends, begin) :])
            if self.right is not None:
                self.right.query(begin, end, result)
            return

        result.extend(self.by_begin)
        if begin < self.center and self.left is not None:
            self.left.query(begin, end, result)
        if end > self.center and self.right is not None:
            self.right.query(begin, end, result)

    def count(self, begin: int, end: int) -> int:
        if end <= self.center:
            result = bisect_left(self.begins, end)
            if self.left is not None:
                result += self.left.count(begin, end)
            return result
        if begin > self.center:
            result = len(self.by_end) - bisect_right(self.ends, begin)
            if self.right is not None:
                result += self.right.count(begin, end)
            return result

        result = len(self.by_begin)
        if begin < self.center and self.left is not None:
            result += self.left.count(begin, end)
        if end > self.center and self.right is not None:
            result += self.right.count(begin, end)
        return result


class _SourceCandidateIndex:
    def __init__(
        self,
        groups: dict[BoxGeometry, list[_CollectedFragment]],
        descriptor: _TensorDescriptor,
    ) -> None:
        if not groups:
            raise ValueError("source candidate index must not be empty")
        first_geometry = next(iter(groups))
        ndim = len(first_geometry[0])
        self._groups: dict[
            BoxGeometry,
            dict[tuple[int, TensorOwner], _CollectedFragment],
        ] = {
            geometry: self._representatives(descriptor, fragments)
            for geometry, fragments in groups.items()
        }
        self._roots = tuple(
            _IntervalNode.build(
                tuple(
                    _IntervalEntry(
                        begin=offset[dim],
                        end=offset[dim] + shape[dim],
                        geometry=(offset, shape),
                    )
                    for offset, shape in groups
                )
            )
            for dim in range(ndim)
        )

    @staticmethod
    def _representatives(
        descriptor: _TensorDescriptor,
        fragments: Sequence[_CollectedFragment],
    ) -> dict[tuple[int, TensorOwner], _CollectedFragment]:
        result = {}
        for fragment in sorted(fragments, key=_source_sort_key):
            result.setdefault(
                (
                    fragment.rank.dp,
                    _tensor_owner(descriptor, fragment),
                ),
                fragment,
            )
        return result

    def query(
        self,
        target: _CollectedFragment,
        *,
        source_dp: int,
        owner: TensorOwner,
    ) -> tuple[tuple[_CollectedFragment, ...], int]:
        counts = []
        for dim, root in enumerate(self._roots):
            if root is None:
                continue
            begin = target.global_offset[dim]
            counts.append(
                (
                    root.count(
                        begin,
                        begin + target.local_shape[dim],
                    ),
                    dim,
                    root,
                )
            )
        if not counts:
            return (), 0
        _, sweep_dim, root = min(counts, key=lambda item: (item[0], item[1]))
        entries: list[_IntervalEntry] = []
        begin = target.global_offset[sweep_dim]
        root.query(
            begin,
            begin + target.local_shape[sweep_dim],
            entries,
        )
        geometries = sorted(
            {
                entry.geometry
                for entry in entries
                if all(
                    source_begin < target_begin + target_extent
                    and target_begin < source_begin + source_extent
                    for source_begin, source_extent, target_begin, target_extent in zip(
                        entry.geometry[0],
                        entry.geometry[1],
                        target.global_offset,
                        target.local_shape,
                        strict=True,
                    )
                )
            }
        )
        key = (source_dp, owner)
        return (
            tuple(
                representative
                for geometry in geometries
                if (representative := self._groups[geometry].get(key)) is not None
            ),
            len(entries),
        )


def _effective_shard_dims(tensor: WeightPlacementTensor) -> tuple[int, ...]:
    shard_dims = tuple(tensor.shard_dims)
    if not shard_dims and tensor.partition_dim is not None:
        shard_dims = (tensor.partition_dim,)
    if tensor.partition_dim is not None and shard_dims != (tensor.partition_dim,):
        raise ValueError(f"partition_dim conflicts with shard_dims: {tensor.tensor_id}")
    if (
        len(shard_dims) != len(set(shard_dims))
        or tuple(sorted(shard_dims)) != shard_dims
        or any(dim < 0 or dim >= len(tensor.global_shape) for dim in shard_dims)
    ):
        raise ValueError(f"invalid shard dimensions: {tensor.tensor_id}")
    return shard_dims


def _descriptor(tensor: WeightPlacementTensor) -> _TensorDescriptor:
    return _TensorDescriptor(
        tensor_id=tensor.tensor_id,
        global_shape=tuple(tensor.global_shape),
        dtype=tensor.dtype,
        itemsize=tensor.itemsize,
        partition_dim=tensor.partition_dim,
        shard_dims=_effective_shard_dims(tensor),
        layer_id=tensor.layer_id,
        expert_id=tensor.expert_id,
        layout_fingerprint=tensor.layout_fingerprint,
    )


def _descriptor_identity(descriptor: _TensorDescriptor) -> tuple:
    return (
        descriptor.global_shape,
        descriptor.dtype,
        descriptor.itemsize,
        descriptor.layer_id,
        descriptor.expert_id,
        descriptor.layout_fingerprint,
    )


def _validate_descriptor(descriptor: _TensorDescriptor) -> None:
    if (
        not descriptor.tensor_id
        or not descriptor.global_shape
        or any(
            type(extent) is not int or extent <= 0 for extent in descriptor.global_shape
        )
        or type(descriptor.itemsize) is not int
        or descriptor.itemsize <= 0
        or not descriptor.dtype
        or not descriptor.layout_fingerprint
    ):
        raise ValueError(f"invalid tensor descriptor: {descriptor.tensor_id}")


def _validate_fragment_geometry(
    descriptor: _TensorDescriptor,
    tensor: WeightPlacementTensor,
) -> None:
    ndim = len(descriptor.global_shape)
    offset = tuple(tensor.global_offset)
    shape = tuple(tensor.local_shape)
    if (
        len(offset) != ndim
        or len(shape) != ndim
        or any(type(item) is not int for item in (*offset, *shape))
    ):
        raise ValueError(f"fragment rank mismatch: {tensor.placement_fragment_id}")
    if any(
        begin < 0 or extent <= 0 or begin + extent > total
        for begin, extent, total in zip(
            offset,
            shape,
            descriptor.global_shape,
            strict=True,
        )
    ):
        raise ValueError(f"fragment is out of bounds: {tensor.placement_fragment_id}")
    shard_dims = frozenset(descriptor.shard_dims)
    for dim in range(ndim):
        if dim in shard_dims:
            continue
        if offset[dim] != 0 or shape[dim] != descriptor.global_shape[dim]:
            raise ValueError(
                f"fragment uses non-shard dimension {dim}: "
                f"{tensor.placement_fragment_id}"
            )
    expected_nbytes = prod(shape) * descriptor.itemsize
    if tensor.nbytes != expected_nbytes:
        raise ValueError(f"fragment byte size mismatch: {tensor.placement_fragment_id}")
    if tensor.byte_offset < 0 or tensor.byte_offset % descriptor.itemsize:
        raise ValueError(
            f"fragment byte offset is invalid: {tensor.placement_fragment_id}"
        )


def _validate_parallel_rank(rank: WeightParallelRank, label: str) -> None:
    if not isinstance(rank, WeightParallelRank) or any(
        type(value) is not int or value < 0
        for value in (rank.dp, rank.tp, rank.pp, rank.ep)
    ):
        raise ValueError(f"{label} parallel rank is invalid")


def _collect_placements(
    placements: Sequence[WeightPlacementManifest],
    label: str,
) -> tuple[dict[str, _TensorDescriptor], tuple[_CollectedFragment, ...]]:
    if not placements:
        raise ValueError(f"{label} placements must not be empty")
    ordered = tuple(sorted(placements, key=lambda item: item.placement_id))
    first = ordered[0]
    placement_ids = [placement.placement_id for placement in ordered]
    if len(placement_ids) != len(set(placement_ids)):
        raise ValueError(f"duplicate {label} placement ID")
    if any(
        placement.model_id != first.model_id or placement.revision != first.revision
        for placement in ordered
    ):
        raise ValueError(f"{label} placement identity mismatch")

    descriptors: dict[str, _TensorDescriptor] = {}
    side_shards: dict[str, tuple[int, ...]] = {}
    fragments = []
    fragment_ids = set()
    for placement in ordered:
        placement_ranks = set()
        for tensor in placement.tensors:
            _validate_parallel_rank(tensor.rank, label)
            placement_ranks.add(tensor.rank)
            if tensor.placement_fragment_id in fragment_ids:
                raise ValueError(
                    f"duplicate {label} placement fragment: "
                    f"{tensor.placement_fragment_id}"
                )
            fragment_ids.add(tensor.placement_fragment_id)
            descriptor = _descriptor(tensor)
            _validate_descriptor(descriptor)
            previous = descriptors.get(tensor.tensor_id)
            if previous is None:
                descriptors[tensor.tensor_id] = descriptor
                side_shards[tensor.tensor_id] = descriptor.shard_dims
            elif (
                _descriptor_identity(previous) != _descriptor_identity(descriptor)
                or side_shards[tensor.tensor_id] != descriptor.shard_dims
            ):
                raise ValueError(
                    f"inconsistent {label} tensor descriptor: {tensor.tensor_id}"
                )
            _validate_fragment_geometry(descriptor, tensor)
            fragments.append(
                _CollectedFragment(
                    logical=LogicalPlacementFragment(
                        placement_id=placement.placement_id,
                        placement_fragment_id=tensor.placement_fragment_id,
                        tensor_id=tensor.tensor_id,
                        global_offset=tuple(tensor.global_offset),
                        local_shape=tuple(tensor.local_shape),
                        nbytes=tensor.nbytes,
                        rank=tensor.rank,
                        aliases=tuple(tensor.aliases),
                    ),
                    descriptor=descriptor,
                )
            )
        if not placement_ranks:
            raise ValueError(f"{label} placement has no fragments")
        if len(placement_ranks) != 1:
            raise ValueError(f"{label} placement mixes parallel ranks")
    if not fragments:
        raise ValueError(f"{label} placements have no fragments")
    return descriptors, tuple(fragments)


def _box_contains(
    outer_offset: tuple[int, ...],
    outer_shape: tuple[int, ...],
    inner_offset: tuple[int, ...],
    inner_shape: tuple[int, ...],
) -> bool:
    return all(
        outer_begin <= inner_begin
        and inner_begin + inner_extent <= outer_begin + outer_extent
        for outer_begin, outer_extent, inner_begin, inner_extent in zip(
            outer_offset,
            outer_shape,
            inner_offset,
            inner_shape,
            strict=True,
        )
    )


def _boxes_overlap(
    boxes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> bool:
    if len(boxes) < 2:
        return False
    ndim = len(boxes[0][0])
    sweep_dim = max(
        range(ndim),
        key=lambda dim: len(
            {(offset[dim], offset[dim] + shape[dim]) for offset, shape in boxes}
        ),
    )
    ordered = sorted(boxes, key=lambda item: item[0][sweep_dim])
    active: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for offset, shape in ordered:
        begin = offset[sweep_dim]
        active = [
            candidate
            for candidate in active
            if candidate[0][sweep_dim] + candidate[1][sweep_dim] > begin
        ]
        if any(
            all(
                left_begin < right_begin + right_extent
                and right_begin < left_begin + left_extent
                for left_begin, left_extent, right_begin, right_extent in zip(
                    candidate_offset,
                    candidate_shape,
                    offset,
                    shape,
                    strict=True,
                )
            )
            for candidate_offset, candidate_shape in active
        ):
            return True
        active.append((offset, shape))
    return False


def _boxes_exactly_cover(
    container_offset: tuple[int, ...],
    container_shape: tuple[int, ...],
    boxes: Sequence[tuple[tuple[int, ...], tuple[int, ...]]],
) -> bool:
    unique_boxes = tuple(dict.fromkeys(boxes))
    if not unique_boxes:
        return False
    if any(
        not _box_contains(container_offset, container_shape, offset, shape)
        for offset, shape in unique_boxes
    ):
        return False
    if sum(prod(shape) for _, shape in unique_boxes) != prod(container_shape):
        return False
    return not _boxes_overlap(unique_boxes)


def _tensor_owner(
    descriptor: _TensorDescriptor,
    fragment: _CollectedFragment,
) -> TensorOwner:
    return (
        fragment.rank.pp,
        fragment.rank.ep if descriptor.expert_id is not None else None,
    )


def _fragments_cover_tensor(
    descriptor: _TensorDescriptor,
    fragments: Sequence[_CollectedFragment],
) -> bool:
    boxes = tuple(
        dict.fromkeys(
            (fragment.global_offset, fragment.local_shape)
            for fragment in fragments
            if fragment.tensor_id == descriptor.tensor_id
        )
    )
    return _boxes_exactly_cover(
        (0,) * len(descriptor.global_shape),
        descriptor.global_shape,
        boxes,
    )


def _complete_source_replicas(
    descriptors: dict[str, _TensorDescriptor],
    fragments: Sequence[_CollectedFragment],
) -> dict[int, dict[str, TensorOwner]]:
    by_dp_and_tensor: dict[int, dict[str, list[_CollectedFragment]]] = {}
    for fragment in fragments:
        by_dp_and_tensor.setdefault(fragment.rank.dp, {}).setdefault(
            fragment.tensor_id,
            [],
        ).append(fragment)

    replicas: dict[int, dict[str, TensorOwner]] = {}
    for dp_rank in sorted(by_dp_and_tensor):
        owner_by_tensor = {}
        complete = True
        for descriptor in descriptors.values():
            by_owner: dict[TensorOwner, list[_CollectedFragment]] = {}
            for fragment in by_dp_and_tensor[dp_rank].get(
                descriptor.tensor_id,
                (),
            ):
                by_owner.setdefault(
                    _tensor_owner(descriptor, fragment),
                    [],
                ).append(fragment)
            complete_owners = [
                owner
                for owner, owner_fragments in by_owner.items()
                if _fragments_cover_tensor(descriptor, owner_fragments)
            ]
            if not by_owner or len(complete_owners) != len(by_owner):
                complete = False
                break
            owner_by_tensor[descriptor.tensor_id] = min(complete_owners)
        if complete:
            replicas[dp_rank] = owner_by_tensor
    if not replicas:
        raise ValueError(
            "source placements have no complete DP replica; "
            "tensors are not fully covered"
        )
    return replicas


def _validate_supplied_target_coverage(
    descriptors: dict[str, _TensorDescriptor],
    fragments: Sequence[_CollectedFragment],
) -> None:
    by_dp_and_tensor: dict[int, dict[str, list[_CollectedFragment]]] = {}
    for fragment in fragments:
        by_dp_and_tensor.setdefault(fragment.rank.dp, {}).setdefault(
            fragment.tensor_id,
            [],
        ).append(fragment)
    for dp_rank in sorted(by_dp_and_tensor):
        for descriptor in descriptors.values():
            by_owner: dict[TensorOwner, list[_CollectedFragment]] = {}
            for fragment in by_dp_and_tensor[dp_rank].get(
                descriptor.tensor_id,
                (),
            ):
                by_owner.setdefault(
                    _tensor_owner(descriptor, fragment),
                    [],
                ).append(fragment)
            if not by_owner or any(
                not _fragments_cover_tensor(descriptor, owner_fragments)
                for owner_fragments in by_owner.values()
            ):
                raise ValueError(
                    f"target tensor is not fully covered: "
                    f"{descriptor.tensor_id}: dp={dp_rank}"
                )


def _parallel_rank_text(rank: WeightParallelRank) -> str:
    return f"(dp={rank.dp}, tp={rank.tp}, pp={rank.pp}, ep={rank.ep})"


def _validate_expected_target_topology(
    expected_target_topology: Sequence[WeightParallelRank],
    target_fragments: Sequence[_CollectedFragment],
) -> None:
    expected_ranks = tuple(expected_target_topology)
    if not expected_ranks:
        raise ValueError("expected target topology must not be empty")
    for rank in expected_ranks:
        _validate_parallel_rank(rank, "expected target topology")
    if len(expected_ranks) != len(set(expected_ranks)):
        raise ValueError("expected target topology contains duplicate ranks")

    expected = set(expected_ranks)
    supplied = {fragment.rank for fragment in target_fragments}
    missing = sorted(
        expected - supplied,
        key=lambda rank: (rank.dp, rank.pp, rank.ep, rank.tp),
    )
    unexpected = sorted(
        supplied - expected,
        key=lambda rank: (rank.dp, rank.pp, rank.ep, rank.tp),
    )
    if missing or unexpected:
        details = []
        if missing:
            details.append(
                "missing ranks: " + ", ".join(map(_parallel_rank_text, missing))
            )
        if unexpected:
            details.append(
                "unexpected ranks: " + ", ".join(map(_parallel_rank_text, unexpected))
            )
        raise ValueError("target placement topology mismatch; " + "; ".join(details))


def _validate_local_target(
    descriptors: dict[str, _TensorDescriptor],
    fragments: Sequence[_CollectedFragment],
) -> None:
    if not fragments:
        raise ValueError("local target placement has no fragments")
    ranks = {fragment.rank for fragment in fragments}
    if len(ranks) != 1:
        raise ValueError("local target must describe exactly one executor")
    missing = sorted(set(descriptors) - {item.tensor_id for item in fragments})
    if missing:
        raise ValueError("local target is missing fragments: " + ", ".join(missing))


def _validate_tensor_sets(
    source: dict[str, _TensorDescriptor],
    target: dict[str, _TensorDescriptor],
    *,
    local_target: bool,
) -> None:
    source_ids = set(source)
    target_ids = set(target)
    missing_source = sorted(target_ids - source_ids)
    if missing_source:
        raise ValueError(
            "target contains unknown tensors: " + ", ".join(missing_source)
        )
    if not local_target:
        missing_target = sorted(source_ids - target_ids)
        if missing_target:
            raise ValueError("target is missing tensors: " + ", ".join(missing_target))
    for tensor_id in sorted(target_ids):
        if _descriptor_identity(source[tensor_id]) != _descriptor_identity(
            target[tensor_id]
        ):
            raise ValueError(f"tensor descriptor mismatch: {tensor_id}")


def _overlap_box(
    source: _CollectedFragment,
    target: _CollectedFragment,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
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
        end - begin for begin, end in zip(overlap_offset, overlap_end, strict=True)
    )
    if any(extent <= 0 for extent in overlap_shape):
        return None
    return overlap_offset, overlap_shape


def _source_sort_key(fragment: _CollectedFragment) -> tuple:
    rank = fragment.rank
    return (
        rank.dp,
        rank.pp,
        rank.ep,
        rank.tp,
        fragment.placement_id,
        fragment.placement_fragment_id,
    )


def _build_executor_groups(
    placements: Sequence[WeightPlacementManifest],
    regions: Sequence[LogicalWeightTransferRegion],
    side: str,
) -> tuple[PlacementExecutorGroup, ...]:
    placement_by_id = {item.placement_id: item for item in placements}
    indices_by_placement: dict[str, list[int]] = {}
    for index, region in enumerate(regions):
        fragment = region.source if side == "source" else region.target
        indices_by_placement.setdefault(fragment.placement_id, []).append(index)
    groups = []
    for placement_id, indices in indices_by_placement.items():
        placement = placement_by_id[placement_id]
        rank = next(
            (
                tensor.rank
                for tensor in placement.tensors
                if tensor.placement_fragment_id
                in {
                    (
                        regions[index].source.placement_fragment_id
                        if side == "source"
                        else regions[index].target.placement_fragment_id
                    )
                    for index in indices
                }
            ),
            None,
        )
        if rank is None:
            raise ValueError("executor placement has no referenced fragments")
        fragment_ids = tuple(
            sorted(
                {
                    (
                        regions[index].source.placement_fragment_id
                        if side == "source"
                        else regions[index].target.placement_fragment_id
                    )
                    for index in indices
                }
            )
        )
        groups.append(
            PlacementExecutorGroup(
                placement_id=placement_id,
                rank=rank,
                placement_fragment_ids=fragment_ids,
                region_indices=tuple(indices),
            )
        )
    groups.sort(
        key=lambda item: (
            item.rank.dp,
            item.rank.pp,
            item.rank.ep,
            item.rank.tp,
            item.placement_id,
        )
    )
    return tuple(groups)


def _build_pipeline_routes(
    regions: Sequence[LogicalWeightTransferRegion],
) -> tuple[PipelineRouteGroup, ...]:
    by_route: dict[tuple[int, int], list[int]] = {}
    for index, region in enumerate(regions):
        by_route.setdefault(
            (region.source.rank.pp, region.target.rank.pp),
            [],
        ).append(index)
    return tuple(
        PipelineRouteGroup(
            source_pp=source_pp,
            target_pp=target_pp,
            region_indices=tuple(indices),
        )
        for (source_pp, target_pp), indices in sorted(by_route.items())
    )


def _plan(
    source_placements: Sequence[WeightPlacementManifest],
    target_placements: Sequence[WeightPlacementManifest],
    *,
    local_target: bool,
    expected_target_topology: Sequence[WeightParallelRank] | None,
    limits: WeightPlannerLimits,
) -> LogicalWeightTransferPlan:
    source_descriptors, source_fragments = _collect_placements(
        source_placements,
        "source",
    )
    target_descriptors, target_fragments = _collect_placements(
        target_placements,
        "target",
    )
    if any(
        len(descriptor.global_shape) > limits.max_tensor_ndim
        for descriptor in (
            *source_descriptors.values(),
            *target_descriptors.values(),
        )
    ):
        raise ValueError("transfer plan exceeds tensor rank limit")
    source_identity = source_placements[0]
    target_identity = target_placements[0]
    if (
        source_identity.model_id != target_identity.model_id
        or source_identity.revision != target_identity.revision
    ):
        raise ValueError("source and target model identity differ")
    _validate_tensor_sets(
        source_descriptors,
        target_descriptors,
        local_target=local_target,
    )
    if local_target:
        _validate_local_target(target_descriptors, target_fragments)
    else:
        _validate_supplied_target_coverage(
            target_descriptors,
            target_fragments,
        )
        if expected_target_topology is not None:
            _validate_expected_target_topology(
                expected_target_topology,
                target_fragments,
            )

    source_replicas = _complete_source_replicas(
        source_descriptors,
        source_fragments,
    )
    source_dp_ranks = sorted(source_replicas)
    source_dp_by_target_dp = {
        target_dp: source_dp_ranks[target_dp % len(source_dp_ranks)]
        for target_dp in {fragment.rank.dp for fragment in target_fragments}
    }

    candidate_groups: dict[
        str,
        dict[tuple[tuple[int, ...], tuple[int, ...]], list[_CollectedFragment]],
    ] = {}
    for fragment in source_fragments:
        candidate_groups.setdefault(fragment.tensor_id, {}).setdefault(
            (fragment.global_offset, fragment.local_shape),
            [],
        ).append(fragment)
    candidate_indexes = {
        tensor_id: _SourceCandidateIndex(
            groups,
            source_descriptors[tensor_id],
        )
        for tensor_id, groups in candidate_groups.items()
    }

    regions = []
    candidate_visits = 0
    total_segments = 0
    for target in sorted(
        target_fragments,
        key=lambda item: (
            item.placement_id,
            item.placement_fragment_id,
        ),
    ):
        source_dp = source_dp_by_target_dp[target.rank.dp]
        owner = source_replicas[source_dp][target.tensor_id]
        overlaps = []
        candidates, visits = candidate_indexes[target.tensor_id].query(
            target,
            source_dp=source_dp,
            owner=owner,
        )
        candidate_visits += visits
        if candidate_visits > limits.max_candidate_visits:
            raise ValueError("transfer plan exceeds candidate visit limit")
        for source in candidates:
            overlap = _overlap_box(source, target)
            if overlap is not None:
                overlaps.append((*overlap, source))
        overlaps.sort(
            key=lambda item: (
                item[0],
                item[1],
                _source_sort_key(item[2]),
            )
        )
        if not _boxes_exactly_cover(
            target.global_offset,
            target.local_shape,
            tuple((offset, shape) for offset, shape, _ in overlaps),
        ):
            raise ValueError(
                f"target fragment is not fully covered: {target.placement_fragment_id}"
            )
        for overlap_offset, overlap_shape, source in overlaps:
            region = build_region(
                tensor_id=target.tensor_id,
                source=source.logical,
                target=target.logical,
                overlap_offset=overlap_offset,
                overlap_shape=overlap_shape,
            )
            if region.segment_count > limits.max_segments_per_region:
                raise ValueError("transfer plan exceeds per-region segment limit")
            regions.append(region)
            if len(regions) > limits.max_regions:
                raise ValueError("transfer plan exceeds region limit")
            total_segments += region.segment_count
            if total_segments > limits.max_total_segments:
                raise ValueError("transfer plan exceeds total segment limit")

    regions.sort(
        key=lambda item: (
            item.target.placement_id,
            item.target.placement_fragment_id,
            item.target_base_offset,
            item.source.placement_id,
            item.source.placement_fragment_id,
            item.source_base_offset,
        )
    )
    normalized_sources = tuple(
        sorted(source_placements, key=lambda item: item.placement_id)
    )
    normalized_targets = tuple(
        sorted(target_placements, key=lambda item: item.placement_id)
    )
    return LogicalWeightTransferPlan(
        model_id=source_identity.model_id,
        revision=source_identity.revision,
        source_placements=normalized_sources,
        target_placements=normalized_targets,
        regions=tuple(regions),
        source_executors=_build_executor_groups(
            normalized_sources,
            regions,
            "source",
        ),
        target_executors=_build_executor_groups(
            normalized_targets,
            regions,
            "target",
        ),
        pipeline_routes=_build_pipeline_routes(regions),
    )


def plan_weight_transfer(
    source_placements: Sequence[WeightPlacementManifest],
    target_placements: Sequence[WeightPlacementManifest],
    *,
    expected_target_topology: Sequence[WeightParallelRank] | None = None,
    limits: WeightPlannerLimits = DEFAULT_WEIGHT_PLANNER_LIMITS,
) -> LogicalWeightTransferPlan:
    """Plan an address-free reshard for all supplied target placements.

    When ``expected_target_topology`` is provided, it is the exact target
    placement-rank set expected in this full-world call. Missing or unexpected
    ranks are rejected. Without it, completeness is guaranteed only for the
    supplied target placements.
    """

    return _plan(
        source_placements,
        target_placements,
        local_target=False,
        expected_target_topology=expected_target_topology,
        limits=limits,
    )


def plan_weight_transfer_to_local_target(
    source_placements: Sequence[WeightPlacementManifest],
    target_placement: WeightPlacementManifest,
    *,
    limits: WeightPlannerLimits = DEFAULT_WEIGHT_PLANNER_LIMITS,
) -> LogicalWeightTransferPlan:
    """Plan the fragments owned by one target executor."""

    result = _plan(
        source_placements,
        (target_placement,),
        local_target=True,
        expected_target_topology=None,
        limits=limits,
    )
    if len(result.target_executors) != 1:
        raise ValueError("local target must describe exactly one executor")
    return result


def select_weight_storage_placements(
    source_placements: Sequence[WeightPlacementManifest],
) -> tuple[WeightPlacementManifest, ...]:
    """Select one complete, deterministic source replica for persistence."""

    descriptors, fragments = _collect_placements(
        source_placements,
        "source",
    )
    replicas = _complete_source_replicas(descriptors, fragments)
    selected_dp = min(replicas)
    owner_by_tensor = replicas[selected_dp]
    by_geometry: dict[
        tuple[str, tuple[int, ...], tuple[int, ...]],
        list[_CollectedFragment],
    ] = {}
    for fragment in fragments:
        descriptor = descriptors[fragment.tensor_id]
        if (
            fragment.rank.dp != selected_dp
            or _tensor_owner(descriptor, fragment)
            != owner_by_tensor[fragment.tensor_id]
        ):
            continue
        by_geometry.setdefault(
            (
                fragment.tensor_id,
                fragment.global_offset,
                fragment.local_shape,
            ),
            [],
        ).append(fragment)

    selected_fragment_ids = {
        min(group, key=_source_sort_key).placement_fragment_id
        for group in by_geometry.values()
    }
    selected = []
    for placement in sorted(
        source_placements,
        key=lambda item: item.placement_id,
    ):
        tensors = tuple(
            tensor
            for tensor in placement.tensors
            if tensor.placement_fragment_id in selected_fragment_ids
        )
        if tensors:
            placement_id = (
                placement.placement_id
                if tensors == placement.tensors
                else compute_weight_placement_id(tensors)
            )
            selected.append(
                WeightPlacementManifest(
                    model_id=placement.model_id,
                    revision=placement.revision,
                    placement_id=placement_id,
                    tensors=tensors,
                    format_version=placement.format_version,
                )
            )
    if not selected:
        raise ValueError("source placements have no persistable fragments")
    return tuple(selected)
