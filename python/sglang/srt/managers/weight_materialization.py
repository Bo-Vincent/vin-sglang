from __future__ import annotations

from collections.abc import Iterable

from sglang.srt.managers.io_struct import WeightMaterializationSessionState

_RETRY_STATES = frozenset(
    {
        WeightMaterializationSessionState.CLEANUP_PENDING,
        WeightMaterializationSessionState.COMPLETION_UNKNOWN,
        WeightMaterializationSessionState.FINALIZE_PENDING,
        WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING,
    }
)
_TERMINAL_STATES = frozenset(
    {
        WeightMaterializationSessionState.DISABLED,
        WeightMaterializationSessionState.FAILED,
        WeightMaterializationSessionState.CONFLICT,
        WeightMaterializationSessionState.NOT_FOUND,
        WeightMaterializationSessionState.SKIPPED,
        WeightMaterializationSessionState.RELEASED,
        WeightMaterializationSessionState.PUBLISHED,
        WeightMaterializationSessionState.PUBLISHED_CLEANUP_FAILED,
    }
)
_PUBLISHED_STATES = frozenset(
    {
        WeightMaterializationSessionState.PUBLISHED,
        WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING,
        WeightMaterializationSessionState.PUBLISHED_CLEANUP_FAILED,
    }
)
_CAPACITY_BLOCKING_STATES = frozenset(
    {
        WeightMaterializationSessionState.CLEANUP_PENDING,
        WeightMaterializationSessionState.COMPLETION_UNKNOWN,
        WeightMaterializationSessionState.FINALIZE_PENDING,
        WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING,
        WeightMaterializationSessionState.PUBLISHED_CLEANUP_FAILED,
    }
)
_REDUCIBLE_STATES = frozenset(
    {
        WeightMaterializationSessionState.DISABLED,
        WeightMaterializationSessionState.PREPARED,
        WeightMaterializationSessionState.FAILED,
        WeightMaterializationSessionState.CONFLICT,
        WeightMaterializationSessionState.NOT_FOUND,
        WeightMaterializationSessionState.CLEANUP_PENDING,
        WeightMaterializationSessionState.COMPLETION_UNKNOWN,
        WeightMaterializationSessionState.FINALIZE_PENDING,
        WeightMaterializationSessionState.SKIPPED,
        WeightMaterializationSessionState.RELEASED,
        WeightMaterializationSessionState.PUBLISHED,
        WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING,
        WeightMaterializationSessionState.PUBLISHED_CLEANUP_FAILED,
    }
)
_REDUCER_PRIORITY = (
    WeightMaterializationSessionState.CONFLICT,
    WeightMaterializationSessionState.FINALIZE_PENDING,
    WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING,
    WeightMaterializationSessionState.PUBLISHED_CLEANUP_FAILED,
    WeightMaterializationSessionState.CLEANUP_PENDING,
    WeightMaterializationSessionState.FAILED,
)
_RELEASED_STATES = frozenset(
    {
        WeightMaterializationSessionState.RELEASED,
        WeightMaterializationSessionState.SKIPPED,
    }
)


def is_retryable_materialization_state(
    state: WeightMaterializationSessionState | str,
) -> bool:
    return WeightMaterializationSessionState(state) in _RETRY_STATES


def is_terminal_materialization_state(
    state: WeightMaterializationSessionState | str,
) -> bool:
    return WeightMaterializationSessionState(state) in _TERMINAL_STATES


def is_published_materialization_state(
    state: WeightMaterializationSessionState | str,
) -> bool:
    return WeightMaterializationSessionState(state) in _PUBLISHED_STATES


def blocks_unresolved_materialization_capacity(
    state: WeightMaterializationSessionState | str,
) -> bool:
    return WeightMaterializationSessionState(state) in _CAPACITY_BLOCKING_STATES


def reduce_materialization_states(
    states: Iterable[object],
    *,
    default: WeightMaterializationSessionState | str,
    completion_unknown: bool = False,
) -> WeightMaterializationSessionState:
    fallback = WeightMaterializationSessionState(default)
    if completion_unknown:
        return WeightMaterializationSessionState.COMPLETION_UNKNOWN

    normalized = set()
    invalid = False
    for state in states:
        try:
            candidate = WeightMaterializationSessionState(state)
        except (TypeError, ValueError):
            invalid = True
            continue
        if candidate not in _REDUCIBLE_STATES:
            invalid = True
            continue
        normalized.add(candidate)
    if invalid:
        return fallback

    for state in _REDUCER_PRIORITY:
        if state in normalized:
            return state
    if WeightMaterializationSessionState.PUBLISHED in normalized and normalized <= {
        WeightMaterializationSessionState.PUBLISHED,
        *_RELEASED_STATES,
    }:
        return WeightMaterializationSessionState.PUBLISHED
    if normalized and normalized <= _RELEASED_STATES:
        return WeightMaterializationSessionState.RELEASED
    if len(normalized) == 1:
        return next(iter(normalized))
    return fallback
