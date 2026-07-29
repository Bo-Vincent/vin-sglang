from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WeightUpdateMutationObservation:
    mutation_started: bool = False


_active_weight_update: ContextVar[WeightUpdateMutationObservation | None] = ContextVar(
    "active_weight_update",
    default=None,
)


def begin_uncoordinated_update(*, full_restore: bool = False) -> None:
    del full_restore
    return None


def finish_uncoordinated_update(token: Any, *, success: bool) -> None:
    del token, success


@contextmanager
def observe_weight_update_mutation() -> Iterator[WeightUpdateMutationObservation]:
    observation = WeightUpdateMutationObservation()
    token = _active_weight_update.set(observation)
    try:
        yield observation
    finally:
        _active_weight_update.reset(token)


def mark_weight_update_mutation_started() -> None:
    observation = _active_weight_update.get()
    if observation is None:
        raise RuntimeError("no coordinated weight update is active")
    observation.mutation_started = True


def _cancel_weight_update(instance: Any, token: Any) -> None:
    cancel = getattr(instance, "cancel_weight_update", None)
    if cancel is None:
        owner = getattr(instance.begin_weight_update, "__self__", None)
        cancel = getattr(owner, "cancel_update", None)
    if cancel is None:
        if token is None:
            return
        raise RuntimeError("coordinated weight updates require a cancel callback")
    cancel(token)


def coordinated_weight_update(
    method=None,
    *,
    full_restore: bool = False,
    full_restore_if: Callable[..., bool] | None = None,
):
    if not isinstance(full_restore, bool):
        raise TypeError("full_restore must be a boolean")
    if full_restore_if is not None and not callable(full_restore_if):
        raise TypeError("full_restore_if must be callable")
    if full_restore and full_restore_if is not None:
        raise ValueError("full_restore and full_restore_if are mutually exclusive")

    def decorate(update_method):
        @wraps(update_method)
        def wrapper(self, *args, **kwargs):
            complete_restore = full_restore
            if full_restore_if is not None:
                complete_restore = full_restore_if(self, *args, **kwargs)
                if not isinstance(complete_restore, bool):
                    raise TypeError("full_restore_if must return a boolean")
            try:
                token = (
                    self.begin_weight_update(full_restore=True)
                    if complete_restore
                    else self.begin_weight_update()
                )
            except Exception as error:
                message = f"Weight update rejected: {error}"
                logger.error(message)
                return False, message

            observation = _active_weight_update.get()
            context_token = None
            if observation is None:
                observation = WeightUpdateMutationObservation()
                context_token = _active_weight_update.set(observation)
            try:
                result = update_method(self, *args, **kwargs)
                success = (
                    isinstance(result, tuple) and len(result) >= 1 and result[0] is True
                )
                mutation_started = observation.mutation_started
            except BaseException as update_error:
                try:
                    mutation_started = observation.mutation_started
                    if mutation_started:
                        self.finish_weight_update(token, success=False)
                    else:
                        _cancel_weight_update(self, token)
                except BaseException as fence_error:
                    raise update_error from fence_error
                raise
            else:
                if success and not mutation_started:
                    _cancel_weight_update(self, token)
                    raise RuntimeError(
                        "successful weight update did not mark mutation start"
                    )
                if mutation_started:
                    self.finish_weight_update(token, success=success)
                else:
                    _cancel_weight_update(self, token)
                return result
            finally:
                if context_token is not None:
                    _active_weight_update.reset(context_token)

        return wrapper

    if method is None:
        return decorate
    return decorate(method)
