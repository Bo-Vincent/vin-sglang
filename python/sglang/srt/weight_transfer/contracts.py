from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any, Iterable, Sequence

from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightRuntimeBindingManifest,
    WeightParallelRank,
    WeightPlacementManifest,
)

_UINT64_MAX = (1 << 64) - 1


def _require_nonempty_string(value: object, name: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _require_int_tuple(
    value: object,
    name: str,
    *,
    minimum: int,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must contain integers")
    try:
        result = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must contain integers") from error
    if any(type(item) is not int or item < minimum for item in result):
        raise ValueError(f"{name} must contain integers >= {minimum}")
    return result


def _require_uint64_range(
    offset: int,
    nbytes: int,
    name: str,
) -> None:
    if offset > _UINT64_MAX or nbytes > _UINT64_MAX - offset:
        raise ValueError(f"{name} exceeds uint64")


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
    if not boxes:
        return False
    if any(
        not _box_contains(container_offset, container_shape, offset, shape)
        for offset, shape in boxes
    ):
        return False
    if sum(prod(shape) for _, shape in boxes) != prod(container_shape):
        return False
    return not _boxes_overlap(boxes)


def _require_parallel_rank(rank: object, name: str) -> WeightParallelRank:
    if not isinstance(rank, WeightParallelRank):
        raise ValueError(f"{name} must be a WeightParallelRank")
    if any(
        type(value) is not int or value < 0
        for value in (rank.dp, rank.tp, rank.pp, rank.ep)
    ):
        raise ValueError(f"{name} values must be non-negative integers")
    return rank


def _fragment_itemsize(fragment: LogicalPlacementFragment) -> int:
    elements = prod(fragment.local_shape)
    if elements <= 0 or fragment.nbytes % elements:
        raise ValueError("logical fragment byte size is invalid")
    itemsize = fragment.nbytes // elements
    if itemsize <= 0:
        raise ValueError("logical fragment itemsize is invalid")
    return itemsize


def _canonical_byte_strides(
    shape: tuple[int, ...],
    itemsize: int,
) -> tuple[int, ...]:
    result = []
    running = itemsize
    for extent in reversed(shape):
        result.append(running)
        running *= extent
    return tuple(reversed(result))


def derive_region_geometry(
    source: LogicalPlacementFragment,
    target: LogicalPlacementFragment,
    overlap_offset: tuple[int, ...],
    overlap_shape: tuple[int, ...],
) -> tuple[
    int,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    source_itemsize = _fragment_itemsize(source)
    target_itemsize = _fragment_itemsize(target)
    if source_itemsize != target_itemsize:
        raise ValueError("source and target itemsize differ")

    source_byte_strides = _canonical_byte_strides(
        source.local_shape,
        source_itemsize,
    )
    target_byte_strides = _canonical_byte_strides(
        target.local_shape,
        target_itemsize,
    )
    source_base_offset = sum(
        (overlap_begin - fragment_begin) * stride
        for overlap_begin, fragment_begin, stride in zip(
            overlap_offset,
            source.global_offset,
            source_byte_strides,
            strict=True,
        )
    )
    target_base_offset = sum(
        (overlap_begin - fragment_begin) * stride
        for overlap_begin, fragment_begin, stride in zip(
            overlap_offset,
            target.global_offset,
            target_byte_strides,
            strict=True,
        )
    )

    suffix_begin = len(overlap_shape) - 1
    inner_bytes = overlap_shape[-1] * source_itemsize
    for dim in range(len(overlap_shape) - 2, -1, -1):
        if (
            source_byte_strides[dim] != inner_bytes
            or target_byte_strides[dim] != inner_bytes
        ):
            break
        inner_bytes *= overlap_shape[dim]
        suffix_begin = dim

    return (
        source_base_offset,
        target_base_offset,
        inner_bytes,
        overlap_shape[:suffix_begin],
        source_byte_strides[:suffix_begin],
        target_byte_strides[:suffix_begin],
    )


def _validate_outer_strides(
    counts: tuple[int, ...],
    strides: tuple[int, ...],
    inner_bytes: int,
    side: str,
) -> None:
    span = inner_bytes
    for count, stride in zip(reversed(counts), reversed(strides), strict=True):
        if count > 1 and stride < span:
            raise ValueError(f"{side} strides overlap")
        span += (count - 1) * stride


@dataclass(frozen=True)
class LogicalPlacementFragment:
    placement_id: str
    placement_fragment_id: str
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    nbytes: int
    rank: WeightParallelRank
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("placement_id", "placement_fragment_id", "tensor_id"):
            _require_nonempty_string(getattr(self, name), name)
        global_offset = _require_int_tuple(
            self.global_offset,
            "global_offset",
            minimum=0,
        )
        local_shape = _require_int_tuple(
            self.local_shape,
            "local_shape",
            minimum=1,
        )
        if not global_offset or len(global_offset) != len(local_shape):
            raise ValueError("logical fragment rank is invalid")
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("logical fragment nbytes must be positive")
        _require_parallel_rank(self.rank, "logical fragment parallel rank")
        aliases = tuple(self.aliases)
        if any(type(alias) is not str or not alias for alias in aliases):
            raise ValueError("logical fragment aliases must be non-empty strings")
        if len(aliases) != len(set(aliases)):
            raise ValueError("logical fragment aliases must not contain duplicates")
        object.__setattr__(self, "global_offset", global_offset)
        object.__setattr__(self, "local_shape", local_shape)
        object.__setattr__(self, "aliases", tuple(sorted(aliases)))
        _fragment_itemsize(self)

    @property
    def fragment_id(self) -> str:
        return self.placement_fragment_id


@dataclass(frozen=True)
class LogicalWeightTransferRegion:
    tensor_id: str
    source: LogicalPlacementFragment
    target: LogicalPlacementFragment
    overlap_offset: tuple[int, ...]
    overlap_shape: tuple[int, ...]
    source_base_offset: int
    target_base_offset: int
    inner_bytes: int
    outer_loop_counts: tuple[int, ...]
    source_strides: tuple[int, ...]
    target_strides: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.tensor_id, "tensor_id")
        if (
            self.source.tensor_id != self.tensor_id
            or self.target.tensor_id != self.tensor_id
        ):
            raise ValueError("transfer region tensor mismatch")
        overlap_offset = _require_int_tuple(
            self.overlap_offset,
            "overlap_offset",
            minimum=0,
        )
        overlap_shape = _require_int_tuple(
            self.overlap_shape,
            "overlap_shape",
            minimum=1,
        )
        outer_loop_counts = _require_int_tuple(
            self.outer_loop_counts,
            "outer_loop_counts",
            minimum=1,
        )
        source_strides = _require_int_tuple(
            self.source_strides,
            "source_strides",
            minimum=0,
        )
        target_strides = _require_int_tuple(
            self.target_strides,
            "target_strides",
            minimum=0,
        )
        ndim = len(overlap_offset)
        if (
            ndim == 0
            or len(overlap_shape) != ndim
            or len(self.source.global_offset) != ndim
            or len(self.target.global_offset) != ndim
        ):
            raise ValueError("transfer region logical rank mismatch")
        if not _box_contains(
            self.source.global_offset,
            self.source.local_shape,
            overlap_offset,
            overlap_shape,
        ):
            raise ValueError("transfer region exceeds source fragment")
        if not _box_contains(
            self.target.global_offset,
            self.target.local_shape,
            overlap_offset,
            overlap_shape,
        ):
            raise ValueError("transfer region exceeds target fragment")
        if not (len(outer_loop_counts) == len(source_strides) == len(target_strides)):
            raise ValueError("transfer region outer-loop rank mismatch")
        for name in ("source_base_offset", "target_base_offset", "inner_bytes"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.source_base_offset < 0 or self.target_base_offset < 0:
            raise ValueError("transfer region base offsets must be non-negative")
        if self.inner_bytes <= 0:
            raise ValueError("transfer region inner_bytes must be positive")

        expected = derive_region_geometry(
            self.source,
            self.target,
            overlap_offset,
            overlap_shape,
        )
        actual = (
            self.source_base_offset,
            self.target_base_offset,
            self.inner_bytes,
            outer_loop_counts,
            source_strides,
            target_strides,
        )
        if actual != expected:
            raise ValueError("transfer region geometry is not canonical")
        _validate_outer_strides(
            outer_loop_counts,
            source_strides,
            self.inner_bytes,
            "source",
        )
        _validate_outer_strides(
            outer_loop_counts,
            target_strides,
            self.inner_bytes,
            "target",
        )
        object.__setattr__(self, "overlap_offset", overlap_offset)
        object.__setattr__(self, "overlap_shape", overlap_shape)
        object.__setattr__(self, "outer_loop_counts", outer_loop_counts)
        object.__setattr__(self, "source_strides", source_strides)
        object.__setattr__(self, "target_strides", target_strides)
        self.validate_bounds()

    @property
    def segment_count(self) -> int:
        return prod(self.outer_loop_counts)

    @property
    def total_bytes(self) -> int:
        return self.inner_bytes * self.segment_count

    @property
    def source_offset(self) -> int:
        return self.source_base_offset

    @property
    def target_offset(self) -> int:
        return self.target_base_offset

    @property
    def nbytes(self) -> int:
        return self.inner_bytes

    def validate_bounds(self) -> None:
        source_end = (
            self.source_base_offset
            + sum(
                (count - 1) * stride
                for count, stride in zip(
                    self.outer_loop_counts,
                    self.source_strides,
                    strict=True,
                )
            )
            + self.inner_bytes
        )
        target_end = (
            self.target_base_offset
            + sum(
                (count - 1) * stride
                for count, stride in zip(
                    self.outer_loop_counts,
                    self.target_strides,
                    strict=True,
                )
            )
            + self.inner_bytes
        )
        if source_end > self.source.nbytes:
            raise ValueError("transfer region exceeds source fragment bytes")
        if target_end > self.target.nbytes:
            raise ValueError("transfer region exceeds target fragment bytes")

    def iter_segments(self) -> Iterable[tuple[int, int, int]]:
        if not self.outer_loop_counts:
            yield self.source_base_offset, self.target_base_offset, self.inner_bytes
            return
        for indices in product(*(range(count) for count in self.outer_loop_counts)):
            yield (
                self.source_base_offset
                + sum(
                    index * stride
                    for index, stride in zip(
                        indices,
                        self.source_strides,
                        strict=True,
                    )
                ),
                self.target_base_offset
                + sum(
                    index * stride
                    for index, stride in zip(
                        indices,
                        self.target_strides,
                        strict=True,
                    )
                ),
                self.inner_bytes,
            )


@dataclass(frozen=True)
class PipelineRouteGroup:
    source_pp: int | None
    target_pp: int
    region_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.source_pp is not None and (
            type(self.source_pp) is not int or self.source_pp < 0
        ):
            raise ValueError("source_pp must be non-negative")
        if type(self.target_pp) is not int or self.target_pp < 0:
            raise ValueError("target_pp must be non-negative")
        indices = _require_int_tuple(
            self.region_indices,
            "region_indices",
            minimum=0,
        )
        if len(indices) != len(set(indices)):
            raise ValueError("pipeline route has duplicate region indices")
        object.__setattr__(self, "region_indices", indices)

    @property
    def operation_indices(self) -> tuple[int, ...]:
        return self.region_indices


@dataclass(frozen=True)
class PlacementExecutorGroup:
    placement_id: str
    rank: WeightParallelRank
    placement_fragment_ids: tuple[str, ...]
    region_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.placement_id, "placement_id")
        _require_parallel_rank(self.rank, "executor parallel rank")
        fragment_ids = tuple(self.placement_fragment_ids)
        if (
            not fragment_ids
            or any(type(item) is not str or not item for item in fragment_ids)
            or len(fragment_ids) != len(set(fragment_ids))
        ):
            raise ValueError("executor fragment IDs are invalid")
        indices = _require_int_tuple(
            self.region_indices,
            "region_indices",
            minimum=0,
        )
        object.__setattr__(self, "placement_fragment_ids", fragment_ids)
        object.__setattr__(self, "region_indices", indices)

    @property
    def operation_indices(self) -> tuple[int, ...]:
        return self.region_indices


def _region_identity(region: LogicalWeightTransferRegion) -> tuple:
    return (
        region.tensor_id,
        region.source.placement_id,
        region.source.placement_fragment_id,
        region.target.placement_id,
        region.target.placement_fragment_id,
        region.overlap_offset,
        region.overlap_shape,
        region.source_base_offset,
        region.target_base_offset,
        region.inner_bytes,
        region.outer_loop_counts,
        region.source_strides,
        region.target_strides,
    )


def _placement_identity(placement: WeightPlacementManifest) -> tuple:
    tensors = tuple(
        sorted(
            placement.tensors,
            key=lambda item: (
                item.placement_fragment_id,
                item.tensor_id,
            ),
        )
    )
    return (
        placement.model_id,
        placement.revision,
        placement.placement_id,
        placement.format_version,
        tuple(
            (
                tensor.placement_fragment_id,
                tensor.tensor_id,
                tensor.runtime_name,
                tuple(tensor.aliases),
                tuple(tensor.global_shape),
                tuple(tensor.global_offset),
                tuple(tensor.local_shape),
                tensor.dtype,
                tensor.itemsize,
                tensor.partition_dim,
                tuple(tensor.shard_dims),
                tensor.layer_id,
                tensor.expert_id,
                tensor.layout_fingerprint,
                tensor.nbytes,
                tensor.byte_offset,
                (
                    tensor.rank.dp,
                    tensor.rank.tp,
                    tensor.rank.pp,
                    tensor.rank.ep,
                ),
            )
            for tensor in tensors
        ),
    )


def _placement_fragments(
    placements: Sequence[WeightPlacementManifest],
    side: str,
) -> dict[str, LogicalPlacementFragment]:
    result = {}
    for placement in placements:
        placement_ranks = set()
        for tensor in placement.tensors:
            _require_parallel_rank(
                tensor.rank,
                f"{side} placement parallel rank",
            )
            placement_ranks.add(tensor.rank)
            fragment = LogicalPlacementFragment(
                placement_id=placement.placement_id,
                placement_fragment_id=tensor.placement_fragment_id,
                tensor_id=tensor.tensor_id,
                global_offset=tuple(tensor.global_offset),
                local_shape=tuple(tensor.local_shape),
                nbytes=tensor.nbytes,
                rank=tensor.rank,
                aliases=tuple(tensor.aliases),
            )
            if fragment.placement_fragment_id in result:
                raise ValueError(f"duplicate {side} placement fragment ID")
            result[fragment.placement_fragment_id] = fragment
        if not placement_ranks:
            raise ValueError(f"{side} placement has no fragments")
        if len(placement_ranks) != 1:
            raise ValueError(f"{side} placement mixes parallel ranks")
    return result


def _validate_executor_groups(
    groups: Sequence[PlacementExecutorGroup],
    placements: Sequence[WeightPlacementManifest],
    regions: Sequence[LogicalWeightTransferRegion],
    side: str,
) -> None:
    if not groups:
        return
    placement_by_id = {placement.placement_id: placement for placement in placements}
    covered_indices = []
    for group in groups:
        placement = placement_by_id.get(group.placement_id)
        if placement is None:
            raise ValueError(f"{side} executor placement is unknown")
        expected_fragments = {
            (
                regions[index].source if side == "source" else regions[index].target
            ).placement_fragment_id
            for index in group.region_indices
        }
        if expected_fragments != set(group.placement_fragment_ids):
            raise ValueError(f"{side} executor fragments differ from regions")
        ranks = {
            (regions[index].source if side == "source" else regions[index].target).rank
            for index in group.region_indices
        }
        if ranks != {group.rank}:
            raise ValueError(f"{side} executor rank differs from regions")
        covered_indices.extend(group.region_indices)
    if sorted(covered_indices) != list(range(len(regions))):
        raise ValueError(f"{side} executors must cover each region exactly once")


@dataclass(frozen=True)
class LogicalWeightTransferPlan:
    model_id: str
    revision: str
    source_placements: tuple[WeightPlacementManifest, ...]
    target_placements: tuple[WeightPlacementManifest, ...]
    regions: tuple[LogicalWeightTransferRegion, ...]
    source_executors: tuple[PlacementExecutorGroup, ...] = ()
    target_executors: tuple[PlacementExecutorGroup, ...] = ()
    pipeline_routes: tuple[PipelineRouteGroup, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.model_id, "model_id")
        _require_nonempty_string(self.revision, "revision")
        for name in (
            "source_placements",
            "target_placements",
            "regions",
            "source_executors",
            "target_executors",
            "pipeline_routes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.source_placements or not self.target_placements:
            raise ValueError("logical plan placements must not be empty")
        for side, placements in (
            ("source", self.source_placements),
            ("target", self.target_placements),
        ):
            placement_ids = [placement.placement_id for placement in placements]
            if len(placement_ids) != len(set(placement_ids)):
                raise ValueError(f"duplicate {side} placement ID")
            if any(
                placement.model_id != self.model_id
                or placement.revision != self.revision
                for placement in placements
            ):
                raise ValueError(f"{side} placement identity mismatch")
        if not self.regions:
            raise ValueError("logical plan regions must not be empty")
        source_fragments = _placement_fragments(
            self.source_placements,
            "source",
        )
        target_fragments = _placement_fragments(
            self.target_placements,
            "target",
        )
        boxes_by_target: dict[
            str,
            list[tuple[tuple[int, ...], tuple[int, ...]]],
        ] = {}
        for region in self.regions:
            expected_source = source_fragments.get(region.source.placement_fragment_id)
            expected_target = target_fragments.get(region.target.placement_fragment_id)
            if expected_source != region.source:
                raise ValueError("logical region source differs from source placement")
            if expected_target != region.target:
                raise ValueError("logical region target differs from target placement")
            boxes_by_target.setdefault(
                region.target.placement_fragment_id,
                [],
            ).append((region.overlap_offset, region.overlap_shape))
        for fragment_id, fragment in target_fragments.items():
            if not _boxes_exactly_cover(
                fragment.global_offset,
                fragment.local_shape,
                boxes_by_target.get(fragment_id, ()),
            ):
                raise ValueError(
                    f"regions must exactly cover target fragment: {fragment_id}"
                )
        for group in (*self.source_executors, *self.target_executors):
            if any(index >= len(self.regions) for index in group.region_indices):
                raise ValueError("executor region index is out of range")
        _validate_executor_groups(
            self.source_executors,
            self.source_placements,
            self.regions,
            "source",
        )
        _validate_executor_groups(
            self.target_executors,
            self.target_placements,
            self.regions,
            "target",
        )
        route_indices = [
            index for route in self.pipeline_routes for index in route.region_indices
        ]
        if sorted(route_indices) != list(range(len(self.regions))):
            raise ValueError("pipeline routes must cover each region exactly once")

    @property
    def operations(self) -> tuple[LogicalWeightTransferRegion, ...]:
        return self.regions

    @property
    def total_bytes(self) -> int:
        return sum(region.total_bytes for region in self.regions)

    @property
    def total_segments(self) -> int:
        return sum(region.segment_count for region in self.regions)

    @property
    def digest(self) -> str:
        identity = (
            "sglang-logical-weight-transfer-v1",
            self.model_id,
            self.revision,
            tuple(
                _placement_identity(placement) for placement in self.source_placements
            ),
            tuple(
                _placement_identity(placement) for placement in self.target_placements
            ),
            tuple(_region_identity(region) for region in self.regions),
            tuple(
                (
                    group.placement_id,
                    (
                        group.rank.dp,
                        group.rank.tp,
                        group.rank.pp,
                        group.rank.ep,
                    ),
                    group.placement_fragment_ids,
                    group.region_indices,
                )
                for group in self.source_executors
            ),
            tuple(
                (
                    group.placement_id,
                    (
                        group.rank.dp,
                        group.rank.tp,
                        group.rank.pp,
                        group.rank.ep,
                    ),
                    group.placement_fragment_ids,
                    group.region_indices,
                )
                for group in self.target_executors
            ),
            tuple(
                (route.source_pp, route.target_pp, route.region_indices)
                for route in self.pipeline_routes
            ),
        )
        return hashlib.sha256(repr(identity).encode()).hexdigest()


@dataclass(frozen=True)
class WeightStorageFragmentBinding:
    placement_fragment_id: str
    fragment_id: str
    object_key: str
    object_offset: int
    nbytes: int
    checksum: str | None = None

    def __post_init__(self) -> None:
        for name in ("placement_fragment_id", "fragment_id", "object_key"):
            _require_nonempty_string(getattr(self, name), name)
        if type(self.object_offset) is not int or self.object_offset < 0:
            raise ValueError("object_offset must be a non-negative integer")
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("storage fragment nbytes must be positive")
        _require_uint64_range(
            self.object_offset,
            self.nbytes,
            "storage object range",
        )
        if self.checksum is not None:
            _require_nonempty_string(self.checksum, "checksum")


@dataclass(frozen=True)
class WeightStorageBindingManifest:
    model_id: str
    revision: str
    placement_id: str
    storage_id: str
    provider: str
    fragments: tuple[WeightStorageFragmentBinding, ...]
    format_version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "revision",
            "placement_id",
            "storage_id",
            "provider",
        ):
            _require_nonempty_string(getattr(self, name), name)
        fragments = tuple(self.fragments)
        fragment_ids = [item.placement_fragment_id for item in fragments]
        if (
            not fragments
            or len(fragment_ids) != len(set(fragment_ids))
            or not all(
                isinstance(item, WeightStorageFragmentBinding) for item in fragments
            )
        ):
            raise ValueError("storage binding fragments are invalid")
        if self.format_version != 1:
            raise ValueError("unsupported storage binding format version")
        object.__setattr__(self, "fragments", fragments)


@dataclass(frozen=True)
class RuntimeWeightLocation:
    placement_id: str
    placement_fragment_id: str
    fragment_id: str
    tensor_id: str
    address: int
    nbytes: int
    storage_offset: int
    device: str
    worker_id: str
    endpoint: str
    generation: int
    lease_id: str
    rank: WeightParallelRank
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "placement_id",
            "placement_fragment_id",
            "fragment_id",
            "tensor_id",
            "device",
            "worker_id",
            "endpoint",
            "lease_id",
        ):
            _require_nonempty_string(getattr(self, name), name)
        for name, minimum in (
            ("address", 1),
            ("nbytes", 1),
            ("storage_offset", 0),
            ("generation", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        _require_uint64_range(
            self.address,
            self.nbytes,
            "runtime address range",
        )
        object.__setattr__(
            self,
            "global_offset",
            _require_int_tuple(self.global_offset, "global_offset", minimum=0),
        )
        object.__setattr__(
            self,
            "local_shape",
            _require_int_tuple(self.local_shape, "local_shape", minimum=1),
        )
        object.__setattr__(self, "aliases", tuple(sorted(self.aliases)))


@dataclass(frozen=True)
class StorageWeightLocation:
    placement_id: str
    placement_fragment_id: str
    fragment_id: str
    tensor_id: str
    provider: str
    storage_id: str
    object_key: str
    object_offset: int
    nbytes: int
    checksum: str | None
    rank: WeightParallelRank
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "placement_id",
            "placement_fragment_id",
            "fragment_id",
            "tensor_id",
            "provider",
            "storage_id",
            "object_key",
        ):
            _require_nonempty_string(getattr(self, name), name)
        if type(self.object_offset) is not int or self.object_offset < 0:
            raise ValueError("object_offset must be non-negative")
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("storage location nbytes must be positive")
        _require_uint64_range(
            self.object_offset,
            self.nbytes,
            "storage object range",
        )
        if self.checksum is not None:
            _require_nonempty_string(self.checksum, "checksum")
        object.__setattr__(
            self,
            "global_offset",
            _require_int_tuple(self.global_offset, "global_offset", minimum=0),
        )
        object.__setattr__(
            self,
            "local_shape",
            _require_int_tuple(self.local_shape, "local_shape", minimum=1),
        )
        object.__setattr__(self, "aliases", tuple(sorted(self.aliases)))


PhysicalWeightLocation = RuntimeWeightLocation | StorageWeightLocation


@dataclass(frozen=True)
class BoundWeightTransferRegion:
    logical_region: LogicalWeightTransferRegion
    source: PhysicalWeightLocation
    target: RuntimeWeightLocation

    def __post_init__(self) -> None:
        logical = self.logical_region
        if (
            self.source.placement_id != logical.source.placement_id
            or self.source.placement_fragment_id != logical.source.placement_fragment_id
            or self.source.tensor_id != logical.tensor_id
        ):
            raise ValueError("bound source does not match logical region")
        if (
            self.target.placement_id != logical.target.placement_id
            or self.target.placement_fragment_id != logical.target.placement_fragment_id
            or self.target.tensor_id != logical.tensor_id
        ):
            raise ValueError("bound target does not match logical region")
        self.validate_bounds()

    @property
    def tensor_id(self) -> str:
        return self.logical_region.tensor_id

    @property
    def source_base_offset(self) -> int:
        return self.logical_region.source_base_offset

    @property
    def target_base_offset(self) -> int:
        return self.logical_region.target_base_offset

    @property
    def inner_bytes(self) -> int:
        return self.logical_region.inner_bytes

    @property
    def outer_loop_counts(self) -> tuple[int, ...]:
        return self.logical_region.outer_loop_counts

    @property
    def source_strides(self) -> tuple[int, ...]:
        return self.logical_region.source_strides

    @property
    def target_strides(self) -> tuple[int, ...]:
        return self.logical_region.target_strides

    @property
    def segment_count(self) -> int:
        return self.logical_region.segment_count

    @property
    def total_bytes(self) -> int:
        return self.logical_region.total_bytes

    def validate_bounds(self) -> None:
        source_end = (
            self.source_base_offset
            + sum(
                (count - 1) * stride
                for count, stride in zip(
                    self.outer_loop_counts,
                    self.source_strides,
                    strict=True,
                )
            )
            + self.inner_bytes
        )
        target_end = (
            self.target_base_offset
            + sum(
                (count - 1) * stride
                for count, stride in zip(
                    self.outer_loop_counts,
                    self.target_strides,
                    strict=True,
                )
            )
            + self.inner_bytes
        )
        if source_end > self.source.nbytes:
            raise ValueError("bound region exceeds source byte range")
        if target_end > self.target.nbytes:
            raise ValueError("bound region exceeds target byte range")

    def iter_segments(self) -> Iterable[tuple[int, int, int]]:
        yield from self.logical_region.iter_segments()

    def iter_absolute_segments(self) -> Iterable[tuple[int, int, int]]:
        if not isinstance(self.source, RuntimeWeightLocation):
            raise ValueError("storage source has no absolute runtime address")
        for source_offset, target_offset, nbytes in self.iter_segments():
            yield (
                self.source.address + source_offset,
                self.target.address + target_offset,
                nbytes,
            )


SourceBindingManifest = WeightRuntimeBindingManifest | WeightStorageBindingManifest


def _binding_fragment_index(
    bindings: Sequence[SourceBindingManifest],
    side: str,
) -> dict[
    tuple[str, str],
    tuple[SourceBindingManifest, Any],
]:
    result = {}
    for binding in bindings:
        for fragment in binding.fragments:
            key = (
                binding.placement_id,
                fragment.placement_fragment_id,
            )
            if key in result:
                raise ValueError(f"duplicate {side} binding fragment")
            result[key] = (binding, fragment)
    return result


def _location_matches_logical_fragment(
    location: PhysicalWeightLocation,
    fragment: LogicalPlacementFragment,
) -> bool:
    return (
        location.placement_id == fragment.placement_id
        and location.placement_fragment_id == fragment.placement_fragment_id
        and location.tensor_id == fragment.tensor_id
        and location.nbytes == fragment.nbytes
        and location.rank == fragment.rank
        and location.global_offset == fragment.global_offset
        and location.local_shape == fragment.local_shape
        and location.aliases == fragment.aliases
    )


def _runtime_location_matches_binding(
    location: RuntimeWeightLocation,
    binding: WeightRuntimeBindingManifest,
    fragment: Any,
) -> bool:
    return (
        binding.placement_id == location.placement_id
        and fragment.placement_fragment_id == location.placement_fragment_id
        and fragment.fragment_id == location.fragment_id
        and fragment.address == location.address
        and fragment.nbytes == location.nbytes
        and fragment.storage_offset == location.storage_offset
        and fragment.device == location.device
        and fragment.is_contiguous is True
        and fragment.worker_id == location.worker_id
        and fragment.endpoint == location.endpoint
        and binding.generation == location.generation
        and binding.lease_id == location.lease_id
    )


def _storage_location_matches_binding(
    location: StorageWeightLocation,
    binding: WeightStorageBindingManifest,
    fragment: Any,
) -> bool:
    return (
        binding.placement_id == location.placement_id
        and fragment.placement_fragment_id == location.placement_fragment_id
        and fragment.fragment_id == location.fragment_id
        and binding.provider == location.provider
        and binding.storage_id == location.storage_id
        and fragment.object_key == location.object_key
        and fragment.object_offset == location.object_offset
        and fragment.nbytes == location.nbytes
        and fragment.checksum == location.checksum
    )


def _declared_source_identity(
    region: LogicalWeightTransferRegion,
    binding: SourceBindingManifest,
    fragment: Any,
) -> tuple:
    if isinstance(binding, WeightRuntimeBindingManifest):
        location = (
            "runtime",
            fragment.worker_id,
            fragment.endpoint,
            binding.generation,
            fragment.address + region.source_base_offset,
        )
    else:
        location = (
            "storage",
            binding.provider,
            fragment.object_key,
            fragment.object_offset + region.source_base_offset,
        )
    return (
        location,
        region.inner_bytes,
        region.outer_loop_counts,
        region.source_strides,
    )


def _storage_payload_slice_identity(
    *,
    checksum: str | None,
    nbytes: int,
    source_base_offset: int,
    inner_bytes: int,
    outer_loop_counts: tuple[int, ...],
    source_strides: tuple[int, ...],
) -> tuple | None:
    prefix = "sha256:"
    digest = checksum.removeprefix(prefix) if type(checksum) is str else ""
    if (
        type(checksum) is not str
        or not checksum.startswith(prefix)
        or len(digest) != 64
        or digest != digest.lower()
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return None
    return (
        checksum,
        nbytes,
        source_base_offset,
        inner_bytes,
        outer_loop_counts,
        source_strides,
    )


def _declared_storage_payload_identity(
    region: LogicalWeightTransferRegion,
    binding: SourceBindingManifest,
    fragment: Any,
) -> tuple | None:
    if isinstance(binding, WeightRuntimeBindingManifest):
        return None
    return _storage_payload_slice_identity(
        checksum=fragment.checksum,
        nbytes=fragment.nbytes,
        source_base_offset=region.source_base_offset,
        inner_bytes=region.inner_bytes,
        outer_loop_counts=region.outer_loop_counts,
        source_strides=region.source_strides,
    )


def _declared_target_identity(
    region: LogicalWeightTransferRegion,
    binding: WeightRuntimeBindingManifest,
    fragment: Any,
) -> tuple:
    return (
        fragment.worker_id,
        fragment.endpoint,
        fragment.address + region.target_base_offset,
        region.inner_bytes,
        region.outer_loop_counts,
        region.target_strides,
    )


def _expected_bound_region_identities(
    logical_plan: LogicalWeightTransferPlan,
    source_fragments: dict[
        tuple[str, str],
        tuple[SourceBindingManifest, Any],
    ],
    target_fragments: dict[
        tuple[str, str],
        tuple[SourceBindingManifest, Any],
    ],
) -> tuple[tuple, ...]:
    result = []
    by_target = {}
    for region in logical_plan.regions:
        source_pair = source_fragments.get(
            (
                region.source.placement_id,
                region.source.placement_fragment_id,
            )
        )
        if source_pair is None:
            raise ValueError("bound source is not declared by source bindings")
        target_pair = target_fragments.get(
            (
                region.target.placement_id,
                region.target.placement_fragment_id,
            )
        )
        if target_pair is None or not isinstance(
            target_pair[0],
            WeightRuntimeBindingManifest,
        ):
            raise ValueError("bound target is not declared by target bindings")
        source_identity = _declared_source_identity(
            region,
            *source_pair,
        )
        storage_payload_identity = _declared_storage_payload_identity(
            region,
            *source_pair,
        )
        target_identity = _declared_target_identity(
            region,
            target_pair[0],
            target_pair[1],
        )
        previous = by_target.get(target_identity)
        if previous is None:
            by_target[target_identity] = (
                region,
                source_identity,
                storage_payload_identity,
            )
            result.append(_region_identity(region))
            continue
        (
            previous_region,
            previous_source_identity,
            previous_storage_payload_identity,
        ) = previous
        identical_source_payload = previous_source_identity == source_identity or (
            previous_storage_payload_identity is not None
            and previous_storage_payload_identity == storage_payload_identity
        )
        exact_alias = (
            len(previous_region.target.aliases) > 1
            and previous_region.target.aliases == region.target.aliases
            and previous_region.source.aliases == region.source.aliases
            and identical_source_payload
            and previous_region.overlap_offset == region.overlap_offset
            and previous_region.overlap_shape == region.overlap_shape
        )
        if not exact_alias:
            raise ValueError("declared bindings create overlapping target writes")
    return tuple(result)


def _validate_bound_physical_locations(
    regions: Sequence[BoundWeightTransferRegion],
    source_fragments: dict[
        tuple[str, str],
        tuple[SourceBindingManifest, Any],
    ],
    target_fragments: dict[
        tuple[str, str],
        tuple[SourceBindingManifest, Any],
    ],
) -> None:
    for region in regions:
        source_pair = source_fragments.get(
            (
                region.source.placement_id,
                region.source.placement_fragment_id,
            )
        )
        source_matches = False
        if source_pair is not None:
            binding, fragment = source_pair
            if isinstance(binding, WeightRuntimeBindingManifest):
                source_matches = isinstance(
                    region.source,
                    RuntimeWeightLocation,
                ) and _runtime_location_matches_binding(
                    region.source,
                    binding,
                    fragment,
                )
            else:
                source_matches = isinstance(
                    region.source,
                    StorageWeightLocation,
                ) and _storage_location_matches_binding(
                    region.source,
                    binding,
                    fragment,
                )
        if not source_matches or not _location_matches_logical_fragment(
            region.source,
            region.logical_region.source,
        ):
            raise ValueError("bound source is not declared by source bindings")

        target_pair = target_fragments.get(
            (
                region.target.placement_id,
                region.target.placement_fragment_id,
            )
        )
        target_matches = (
            target_pair is not None
            and isinstance(target_pair[0], WeightRuntimeBindingManifest)
            and _runtime_location_matches_binding(
                region.target,
                target_pair[0],
                target_pair[1],
            )
        )
        if not target_matches or not _location_matches_logical_fragment(
            region.target,
            region.logical_region.target,
        ):
            raise ValueError("bound target is not declared by target bindings")


@dataclass(frozen=True)
class BoundWeightTransferPlan:
    logical_plan: LogicalWeightTransferPlan
    regions: tuple[BoundWeightTransferRegion, ...]
    source_bindings: tuple[SourceBindingManifest, ...]
    target_bindings: tuple[WeightRuntimeBindingManifest, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "source_bindings", tuple(self.source_bindings))
        object.__setattr__(self, "target_bindings", tuple(self.target_bindings))
        if not self.target_bindings:
            raise ValueError("bound plan target bindings must not be empty")
        if not self.regions:
            raise ValueError("bound plan regions must not be empty")
        source_fragments = _binding_fragment_index(
            self.source_bindings,
            "source",
        )
        target_fragments = _binding_fragment_index(
            self.target_bindings,
            "target",
        )
        expected_identities = _expected_bound_region_identities(
            self.logical_plan,
            source_fragments,
            target_fragments,
        )
        actual_identities = tuple(
            _region_identity(region.logical_region) for region in self.regions
        )
        if actual_identities != expected_identities:
            raise ValueError("bound regions must exactly match the logical plan")
        _validate_bound_physical_locations(
            self.regions,
            source_fragments,
            target_fragments,
        )

    @property
    def operations(self) -> tuple[BoundWeightTransferRegion, ...]:
        return self.regions

    @property
    def total_bytes(self) -> int:
        return sum(region.total_bytes for region in self.regions)

    @property
    def total_segments(self) -> int:
        return sum(region.segment_count for region in self.regions)

    @property
    def digest(self) -> str:
        identity = (
            "sglang-bound-weight-transfer-v1",
            self.logical_plan.digest,
            tuple(
                (
                    region.source.fragment_id,
                    region.target.fragment_id,
                    region.source_base_offset,
                    region.target_base_offset,
                    region.inner_bytes,
                    region.outer_loop_counts,
                    region.source_strides,
                    region.target_strides,
                )
                for region in self.regions
            ),
        )
        return hashlib.sha256(repr(identity).encode()).hexdigest()


def build_region(
    *,
    tensor_id: str,
    source: LogicalPlacementFragment,
    target: LogicalPlacementFragment,
    overlap_offset: Sequence[int],
    overlap_shape: Sequence[int],
) -> LogicalWeightTransferRegion:
    normalized_offset = tuple(overlap_offset)
    normalized_shape = tuple(overlap_shape)
    geometry = derive_region_geometry(
        source,
        target,
        normalized_offset,
        normalized_shape,
    )
    return LogicalWeightTransferRegion(
        tensor_id=tensor_id,
        source=source,
        target=target,
        overlap_offset=normalized_offset,
        overlap_shape=normalized_shape,
        source_base_offset=geometry[0],
        target_base_offset=geometry[1],
        inner_bytes=geometry[2],
        outer_loop_counts=geometry[3],
        source_strides=geometry[4],
        target_strides=geometry[5],
    )
