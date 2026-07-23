from __future__ import annotations

import logging
from functools import wraps
from inspect import signature
from typing import Any

logger = logging.getLogger(__name__)


def begin_uncoordinated_update(*, full_restore: bool = False) -> None:
    del full_restore
    return None


def finish_uncoordinated_update(token: Any, *, success: bool) -> None:
    del token, success


def _is_complete_disk_restore(update_method, self, args, kwargs) -> bool:
    if update_method.__name__ != "update_weights_from_disk":
        return False

    method_signature = signature(update_method)
    if "weight_name_filter" not in method_signature.parameters:
        return False
    bound = method_signature.bind(self, *args, **kwargs)
    bound.apply_defaults()
    return bound.arguments["weight_name_filter"] is None


def coordinated_weight_update(method=None, *, full_restore: bool = False):
    if not isinstance(full_restore, bool):
        raise TypeError("full_restore must be a boolean")

    def decorate(update_method):
        @wraps(update_method)
        def wrapper(self, *args, **kwargs):
            complete_restore = full_restore or _is_complete_disk_restore(
                update_method, self, args, kwargs
            )
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
