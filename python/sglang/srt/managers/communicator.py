from __future__ import annotations

import asyncio
import copy
import logging
from typing import Callable, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FanOutCommunicator(Generic[T]):
    """Fan-out request + collect response primitive over zmq.

    One send is fanned out to `fan_out` recipients; the caller awaits until
    all `fan_out` responses are collected. Supports two modes:
    - "queueing": requests are serialized; concurrent callers wait in a FIFO queue.
    - "watching": concurrent callers share a single in-flight request and all
      receive the same result when it completes.

    Only one request is in-flight at any time in either mode.
    """

    def __init__(
        self,
        send: Callable[[T], None],
        fan_out: int,
        mode: str = "queueing",
        correlation_attr: str | None = None,
    ):
        self._send = send
        self._fan_out = fan_out
        self._mode = mode
        self._correlation_attr = correlation_attr
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._result_fan_out: Optional[int] = None
        self._result_correlation_value = None
        self._queueing_lock = asyncio.Lock()

        assert mode in ["queueing", "watching"]
        assert correlation_attr is None or correlation_attr

    async def queueing_call(self, obj: T):
        # asyncio.Lock is FIFO-fair: a new caller cannot acquire while earlier
        # callers are still waiting, so requests are strictly serialized in
        # arrival order. It also releases on exception/cancellation, so a
        # failed caller never blocks the callers queued behind it.
        async with self._queueing_lock:
            event = asyncio.Event()
            self._result_event = event
            self._result_values = []
            self._result_fan_out = self._fan_out
            self._result_correlation_value = self._correlation_value(obj)
            try:
                if obj is not None:
                    self._send(obj)
                await event.wait()
                return self._result_values
            finally:
                if self._result_event is event:
                    self._clear_result_state()

    async def watching_call(self, obj):
        if self._result_event is None:
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()
            self._result_fan_out = self._fan_out
            self._result_correlation_value = self._correlation_value(obj)

            if obj is not None:
                self._send(obj)

        # Capture local refs before await -- after event fires, the first
        # awakened coroutine clears shared state; later awaiters use local refs.
        values = self._result_values
        event = self._result_event
        await event.wait()

        result_values = copy.deepcopy(values)
        if self._result_event is event:
            self._clear_result_state()
        return result_values

    async def __call__(self, obj):
        if self._mode == "queueing":
            return await self.queueing_call(obj)
        else:
            return await self.watching_call(obj)

    def set_fan_out(self, fan_out: int):
        self._fan_out = fan_out

    def handle_recv(self, recv_obj: T):
        if (
            self._result_values is None
            or self._result_event is None
            or self._result_fan_out is None
        ):
            logger.debug(
                "Dropping communicator response without active waiter: %s",
                type(recv_obj).__name__,
            )
            return
        correlation_value = self._correlation_value(recv_obj)
        if correlation_value != self._result_correlation_value:
            logger.debug(
                "Dropping communicator response for stale correlation value: %r",
                correlation_value,
            )
            return
        if self._result_event.is_set():
            logger.debug("Dropping communicator response after fan-out completed")
            return
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._result_fan_out:
            self._result_event.set()

    def _correlation_value(self, obj):
        if self._correlation_attr is None:
            return None
        return getattr(obj, self._correlation_attr, None)

    def _clear_result_state(self):
        self._result_event = self._result_values = None
        self._result_fan_out = None
        self._result_correlation_value = None

    @staticmethod
    def merge_results(results):
        all_success = all([r.success for r in results])
        all_message = [r.message for r in results]
        all_message = " | ".join(all_message)
        return all_success, all_message
