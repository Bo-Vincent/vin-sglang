from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Iterator, Mapping

logger = logging.getLogger(__name__)

_PRE_RESERVED_WEIGHT_UPDATES: ContextVar[Mapping[int, Any]] = ContextVar(
    "pre_reserved_weight_updates",
    default={},
)


@contextmanager
def use_pre_reserved_weight_updates(
    reservations: Mapping[int, Any],
) -> Iterator[None]:
    """Expose scheduler-owned reservations to nested worker update calls."""

    token = _PRE_RESERVED_WEIGHT_UPDATES.set(dict(reservations))
    try:
        yield
    finally:
        _PRE_RESERVED_WEIGHT_UPDATES.reset(token)


def _pre_reserved_token(begin_weight_update: Any) -> Any | None:
    coordinator = getattr(begin_weight_update, "__self__", None)
    if coordinator is None:
        return None
    return _PRE_RESERVED_WEIGHT_UPDATES.get().get(id(coordinator))


def begin_uncoordinated_update(*, full_restore: bool = False) -> None:
    del full_restore
    return None


def finish_uncoordinated_update(token: Any, *, success: bool) -> None:
    del token, success


def coordinated_weight_update(method=None, *, full_restore: bool = False):
    if not isinstance(full_restore, bool):
        raise TypeError("full_restore must be a boolean")

    def decorate(update_method):
        @wraps(update_method)
        def wrapper(self, *args, **kwargs):
            reserved_token = _pre_reserved_token(self.begin_weight_update)
            if reserved_token is not None:
                return update_method(self, *args, **kwargs)

            try:
                token = (
                    self.begin_weight_update(full_restore=True)
                    if full_restore
                    else self.begin_weight_update()
                )
            except Exception as error:
                message = f"Weight update rejected: {error}"
                logger.error(message)
                return False, message

            try:
                result = update_method(self, *args, **kwargs)
                success = (
                    isinstance(result, tuple) and len(result) >= 1 and result[0] is True
                )
            except BaseException as update_error:
                try:
                    self.finish_weight_update(token, success=False)
                except BaseException as fence_error:
                    raise update_error from fence_error
                raise
            else:
                self.finish_weight_update(token, success=success)
                return result

        return wrapper

    if method is None:
        return decorate
    return decorate(method)
