from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from heapq import heappop, heappush
from math import prod
from typing import Iterator, Sequence

Box = tuple[tuple[int, ...], tuple[int, ...]]
DEFAULT_MAX_GEOMETRY_COMPARISONS = 10_000_000
DEFAULT_MAX_GEOMETRY_BOXES = 2_000_000
DEFAULT_MAX_GEOMETRY_EVENTS = 20_000_000
DEFAULT_MAX_GEOMETRY_SORT_WORK = 200_000_000


@dataclass
class _GeometryWorkState:
    comparison_limit: int
    box_limit: int
    event_limit: int
    sort_work_limit: int
    comparisons: int = 0
    boxes: int = 0
    events: int = 0
    sort_work: int = 0


_ExactCoverKey = tuple[tuple[int, ...], tuple[int, ...], tuple[Box, ...]]


@dataclass
class _GeometryRequest:
    state: _GeometryWorkState
    exact_cover_receipts: set[_ExactCoverKey]


_ACTIVE_GEOMETRY_REQUEST: ContextVar[_GeometryRequest | None] = ContextVar(
    "_ACTIVE_GEOMETRY_REQUEST",
    default=None,
)


class GeometryWorkBudget:
    def __init__(
        self,
        limit: int | None = None,
        *,
        max_boxes: int | None = None,
        max_events: int | None = None,
        max_sort_work: int | None = None,
    ) -> None:
        active = _ACTIVE_GEOMETRY_REQUEST.get()
        if (
            active is not None
            and limit is None
            and max_boxes is None
            and max_events is None
            and max_sort_work is None
        ):
            self._state = active.state
            return

        values = (
            (
                "geometry comparison limit",
                DEFAULT_MAX_GEOMETRY_COMPARISONS if limit is None else limit,
            ),
            (
                "geometry box limit",
                DEFAULT_MAX_GEOMETRY_BOXES if max_boxes is None else max_boxes,
            ),
            (
                "geometry event limit",
                DEFAULT_MAX_GEOMETRY_EVENTS if max_events is None else max_events,
            ),
            (
                "geometry sort work limit",
                (
                    DEFAULT_MAX_GEOMETRY_SORT_WORK
                    if max_sort_work is None
                    else max_sort_work
                ),
            ),
        )
        for name, value in values:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._state = _GeometryWorkState(*(value for _, value in values))

    @property
    def limit(self) -> int:
        return self._state.comparison_limit

    @property
    def comparisons(self) -> int:
        return self._state.comparisons

    def consume(self, count: int = 1) -> None:
        self.reserve(comparisons=count)

    def reserve(
        self,
        *,
        comparisons: int = 0,
        boxes: int = 0,
        events: int = 0,
        sort_work: int = 0,
    ) -> None:
        counts = (
            ("geometry comparison", comparisons),
            ("geometry box", boxes),
            ("geometry event", events),
            ("geometry sort work", sort_work),
        )
        for name, count in counts:
            if type(count) is not int or count < 0:
                raise ValueError(f"{name} count must be a non-negative integer")
        state = self._state
        checks = (
            (
                "geometry comparison limit",
                comparisons,
                state.comparisons,
                state.comparison_limit,
            ),
            ("geometry box limit", boxes, state.boxes, state.box_limit),
            ("geometry event limit", events, state.events, state.event_limit),
            (
                "geometry sort work limit",
                sort_work,
                state.sort_work,
                state.sort_work_limit,
            ),
        )
        for name, count, used, limit in checks:
            if count > limit - used:
                raise ValueError(f"{name} exceeded")
        state.comparisons += comparisons
        state.boxes += boxes
        state.events += events
        state.sort_work += sort_work

    @contextmanager
    def request_scope(self) -> Iterator[None]:
        request = _GeometryRequest(self._state, set())
        token = _ACTIVE_GEOMETRY_REQUEST.set(request)
        try:
            yield
        finally:
            _ACTIVE_GEOMETRY_REQUEST.reset(token)


def _sort_work(count: int) -> int:
    if count < 2:
        return 0
    return count * (count - 1).bit_length()


def _has_exact_cover_receipt(
    request: _GeometryRequest,
    container_offset: tuple[int, ...],
    container_shape: tuple[int, ...],
    boxes: tuple[Box, ...],
) -> bool:
    key = (container_offset, container_shape, boxes)
    return key in request.exact_cover_receipts


def _record_exact_cover_receipt(
    request: _GeometryRequest,
    container_offset: tuple[int, ...],
    container_shape: tuple[int, ...],
    boxes: tuple[Box, ...],
) -> None:
    request.exact_cover_receipts.add((container_offset, container_shape, boxes))


class TargetAliasClassification(Enum):
    SNAPSHOT_MISMATCH = "target alias snapshot identity differs"
    TARGET_VIEW_MISMATCH = "overlapping target writes: target views differ"
    ALIAS_NOT_DECLARED = "overlapping target writes: target alias is not declared"
    PARTIAL_OVERLAP = "overlapping target writes: logical regions partially overlap"
    SOURCE_PAYLOAD_MISMATCH = "overlapping target writes: source payload differs"


@dataclass(frozen=True)
class TargetAliasCandidate:
    allocation_identity: tuple
    snapshot_identity: tuple
    target_view_identity: tuple
    logical_box: Box
    source_identity: tuple
    source_payload_identity: tuple | None
    alias_declared: bool


@dataclass(frozen=True)
class TargetAliasConflict:
    classification: TargetAliasClassification
    left_index: int
    right_index: int


@dataclass(frozen=True)
class TargetAliasAnalysis:
    keep_indices: tuple[int, ...]
    conflict: TargetAliasConflict | None = None


def box_contains(
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


def boxes_intersect(left: Box, right: Box) -> bool:
    left_offset, left_shape = left
    right_offset, right_shape = right
    return all(
        left_begin < right_begin + right_extent
        and right_begin < left_begin + left_extent
        for left_begin, left_extent, right_begin, right_extent in zip(
            left_offset,
            left_shape,
            right_offset,
            right_shape,
            strict=True,
        )
    )


def _peak_active_intervals(boxes: Sequence[Box], dim: int) -> int:
    events = []
    for offset, shape in boxes:
        events.append((offset[dim], 1))
        events.append((offset[dim] + shape[dim], -1))
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def find_box_overlap(
    boxes: Sequence[Box],
    *,
    budget: GeometryWorkBudget | None = None,
    _boxes_reserved: bool = False,
) -> tuple[int, int] | None:
    if len(boxes) < 2:
        return None
    budget = budget or GeometryWorkBudget()
    box_count = len(boxes)
    ndim = len(boxes[0][0])
    event_count = ndim * box_count * 2
    budget.reserve(
        boxes=0 if _boxes_reserved else box_count,
        events=event_count,
        sort_work=ndim * _sort_work(box_count * 2) + _sort_work(box_count),
    )
    sweep_dim = min(
        range(ndim),
        key=lambda dim: (_peak_active_intervals(boxes, dim), dim),
    )
    ordered = sorted(
        enumerate(boxes),
        key=lambda item: (
            item[1][0][sweep_dim],
            item[1][0][sweep_dim] + item[1][1][sweep_dim],
            item[0],
        ),
    )
    active_heap: list[tuple[int, int]] = []
    active: dict[int, Box] = {}
    for index, box in ordered:
        begin = box[0][sweep_dim]
        while active_heap and active_heap[0][0] <= begin:
            _, expired_index = heappop(active_heap)
            active.pop(expired_index, None)
        for candidate_index, candidate in active.items():
            budget.consume()
            if boxes_intersect(candidate, box):
                return candidate_index, index
        active[index] = box
        heappush(
            active_heap,
            (begin + box[1][sweep_dim], index),
        )
    return None


def boxes_overlap(
    boxes: Sequence[Box],
    *,
    budget: GeometryWorkBudget | None = None,
    _boxes_reserved: bool = False,
) -> bool:
    return (
        find_box_overlap(
            boxes,
            budget=budget,
            _boxes_reserved=_boxes_reserved,
        )
        is not None
    )


def boxes_exactly_cover(
    container_offset: tuple[int, ...],
    container_shape: tuple[int, ...],
    boxes: Sequence[Box],
    *,
    budget: GeometryWorkBudget | None = None,
    _boxes_reserved: bool = False,
) -> bool:
    budget = budget or GeometryWorkBudget()
    request = _ACTIVE_GEOMETRY_REQUEST.get()
    request_matches = request is not None and budget._state is request.state
    box_count = len(boxes)
    if not box_count:
        return False
    if not _boxes_reserved:
        budget.reserve(boxes=box_count)
    normalized_boxes = tuple(boxes)
    if request_matches and _has_exact_cover_receipt(
        request,
        container_offset,
        container_shape,
        normalized_boxes,
    ):
        return True
    if any(
        not box_contains(container_offset, container_shape, offset, shape)
        for offset, shape in normalized_boxes
    ):
        return False
    if sum(prod(shape) for _, shape in normalized_boxes) != prod(container_shape):
        return False
    result = not boxes_overlap(
        normalized_boxes,
        budget=budget,
        _boxes_reserved=True,
    )
    if result and request_matches:
        _record_exact_cover_receipt(
            request,
            container_offset,
            container_shape,
            normalized_boxes,
        )
    return result


def _payload_conflict(
    indices: Sequence[int],
    candidates: Sequence[TargetAliasCandidate],
) -> tuple[int, int] | None:
    first_index = indices[0]
    first = candidates[first_index]
    source_variant = next(
        (
            index
            for index in indices[1:]
            if candidates[index].source_identity != first.source_identity
        ),
        None,
    )
    if source_variant is None:
        return None

    first_payload = first.source_payload_identity
    if (
        first_payload is None
        or candidates[source_variant].source_payload_identity != first_payload
    ):
        return first_index, source_variant
    payload_variant = next(
        (
            index
            for index in indices[1:]
            if candidates[index].source_payload_identity != first_payload
        ),
        None,
    )
    if payload_variant is None:
        return None
    if candidates[payload_variant].source_identity != first.source_identity:
        return first_index, payload_variant
    return source_variant, payload_variant


def analyze_target_aliases(
    candidates: Sequence[TargetAliasCandidate],
    *,
    budget: GeometryWorkBudget | None = None,
    validate_overlaps: bool = True,
) -> TargetAliasAnalysis:
    budget = budget or GeometryWorkBudget()
    by_allocation: dict[tuple, list[int]] = {}
    for index, candidate in enumerate(candidates):
        by_allocation.setdefault(candidate.allocation_identity, []).append(index)

    keep = set()
    for indices in by_allocation.values():
        first_index = indices[0]
        first = candidates[first_index]
        for index in indices[1:]:
            candidate = candidates[index]
            if candidate.snapshot_identity != first.snapshot_identity:
                return TargetAliasAnalysis(
                    (),
                    TargetAliasConflict(
                        TargetAliasClassification.SNAPSHOT_MISMATCH,
                        first_index,
                        index,
                    ),
                )
            if candidate.target_view_identity != first.target_view_identity:
                return TargetAliasAnalysis(
                    (),
                    TargetAliasConflict(
                        TargetAliasClassification.TARGET_VIEW_MISMATCH,
                        first_index,
                        index,
                    ),
                )

        by_box: dict[Box, list[int]] = {}
        for index in indices:
            by_box.setdefault(candidates[index].logical_box, []).append(index)
        representatives = []
        for duplicate_indices in by_box.values():
            representative = duplicate_indices[0]
            representatives.append(representative)
            keep.add(representative)
            if len(duplicate_indices) < 2:
                continue
            undeclared = next(
                (
                    index
                    for index in duplicate_indices
                    if not candidates[index].alias_declared
                ),
                None,
            )
            if undeclared is not None:
                other = (
                    duplicate_indices[1]
                    if undeclared == representative
                    else representative
                )
                return TargetAliasAnalysis(
                    (),
                    TargetAliasConflict(
                        TargetAliasClassification.ALIAS_NOT_DECLARED,
                        other,
                        undeclared,
                    ),
                )
            payload_conflict = _payload_conflict(
                duplicate_indices,
                candidates,
            )
            if payload_conflict is not None:
                return TargetAliasAnalysis(
                    (),
                    TargetAliasConflict(
                        TargetAliasClassification.SOURCE_PAYLOAD_MISMATCH,
                        *payload_conflict,
                    ),
                )

        if validate_overlaps:
            overlap = find_box_overlap(
                tuple(candidates[index].logical_box for index in representatives),
                budget=budget,
            )
            if overlap is not None:
                return TargetAliasAnalysis(
                    (),
                    TargetAliasConflict(
                        TargetAliasClassification.PARTIAL_OVERLAP,
                        representatives[overlap[0]],
                        representatives[overlap[1]],
                    ),
                )

    return TargetAliasAnalysis(
        tuple(index for index in range(len(candidates)) if index in keep)
    )


def intersect_boxes(left: Box, right: Box) -> Box | None:
    left_offset, left_shape = left
    right_offset, right_shape = right
    offset = tuple(
        max(left_begin, right_begin)
        for left_begin, right_begin in zip(
            left_offset,
            right_offset,
            strict=True,
        )
    )
    end = tuple(
        min(left_begin + left_extent, right_begin + right_extent)
        for left_begin, left_extent, right_begin, right_extent in zip(
            left_offset,
            left_shape,
            right_offset,
            right_shape,
            strict=True,
        )
    )
    shape = tuple(
        box_end - box_begin for box_begin, box_end in zip(offset, end, strict=True)
    )
    if any(extent <= 0 for extent in shape):
        return None
    return offset, shape
