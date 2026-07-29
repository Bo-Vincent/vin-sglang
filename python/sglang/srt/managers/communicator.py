from __future__ import annotations

import asyncio
import copy
import logging
import time
from typing import Callable, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FanOutCancelledBeforeDispatch(asyncio.CancelledError):
    pass


class FanOutDeadlineExpiredBeforeDispatch(TimeoutError):
    pass


class FanOutCompletionUnknownError(TimeoutError):
    def __init__(
        self,
        message: str,
        *,
        dispatch_started: bool,
        dispatch_completed: bool,
        partial_results,
        expected_count: int,
    ) -> None:
        super().__init__(message)
        self.completion_unknown = True
        self.dispatch_started = dispatch_started
        self.dispatch_completed = dispatch_completed
        self.partial_results = tuple(partial_results)
        self.received_count = len(self.partial_results)
        self.expected_count = expected_count


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
        responder_attr: str | None = None,
    ):
        self._send = send
        self._fan_out = fan_out
        self._mode = mode
        self._correlation_attr = correlation_attr
        self._responder_attr = responder_attr
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._result_fan_out: Optional[int] = None
        self._result_correlation_value = None
        self._result_responders: set[object] | None = None
        self._result_waiters: int | None = None
        self._result_dispatch_started: bool | None = None
        self._result_dispatch_completed: bool | None = None
        self._queueing_lock = asyncio.Lock()
        self._watching_lock = asyncio.Lock()

        assert mode in ["queueing", "watching"]
        assert correlation_attr is None or correlation_attr
        assert responder_attr is None or responder_attr

    @staticmethod
    def _remaining_timeout(deadline_unix_sec: float | None) -> float | None:
        if deadline_unix_sec is None:
            return None
        if not isinstance(deadline_unix_sec, (int, float)):
            raise TypeError("fan-out deadline must be numeric")
        remaining = float(deadline_unix_sec) - time.time()
        if remaining <= 0:
            raise TimeoutError("fan-out deadline expired before dispatch")
        return remaining

    @classmethod
    async def _acquire_before_deadline(
        cls,
        lock: asyncio.Lock,
        deadline_unix_sec: float | None,
    ) -> None:
        try:
            timeout = cls._remaining_timeout(deadline_unix_sec)
            if timeout is None:
                await lock.acquire()
            else:
                await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError) as error:
            raise FanOutDeadlineExpiredBeforeDispatch(
                "fan-out deadline expired before dispatch"
            ) from error

    async def queueing_call(
        self,
        obj: T,
        *,
        deadline_unix_sec: float | None = None,
    ):
        dispatched = False
        acquired = False
        try:
            await self._acquire_before_deadline(
                self._queueing_lock,
                deadline_unix_sec,
            )
            acquired = True
            try:
                try:
                    self._remaining_timeout(deadline_unix_sec)
                except TimeoutError as error:
                    raise FanOutDeadlineExpiredBeforeDispatch(str(error)) from error
                event = asyncio.Event()
                self._result_event = event
                self._result_values = []
                self._result_fan_out = self._fan_out
                self._result_correlation_value = self._request_correlation_value(obj)
                self._result_responders = set()
                self._result_waiters = 1
                self._result_dispatch_started = False
                self._result_dispatch_completed = False
                values = self._result_values
                fan_out = self._result_fan_out
                try:
                    self._result_dispatch_started = True
                    try:
                        if obj is not None:
                            self._send(obj)
                    except Exception as error:
                        raise FanOutCompletionUnknownError(
                            f"fan-out dispatch completion is unknown: {error}",
                            dispatch_started=True,
                            dispatch_completed=False,
                            partial_results=values,
                            expected_count=fan_out,
                        ) from error
                    self._result_dispatch_completed = True
                    dispatched = True
                    try:
                        timeout = self._remaining_timeout(deadline_unix_sec)
                        await asyncio.wait_for(event.wait(), timeout=timeout)
                    except (TimeoutError, asyncio.TimeoutError) as error:
                        received = len(values)
                        raise FanOutCompletionUnknownError(
                            "fan-out response deadline expired: "
                            f"received {received}/{fan_out}",
                            dispatch_started=True,
                            dispatch_completed=True,
                            partial_results=values,
                            expected_count=fan_out,
                        ) from error
                    return values
                finally:
                    if self._result_event is event:
                        self._clear_result_state()
            finally:
                if acquired:
                    self._queueing_lock.release()
        except asyncio.CancelledError as error:
            if not dispatched:
                raise FanOutCancelledBeforeDispatch() from error
            raise

    async def watching_call(
        self,
        obj,
        *,
        deadline_unix_sec: float | None = None,
    ):
        dispatched = False
        acquired = False
        try:
            await self._acquire_before_deadline(
                self._watching_lock,
                deadline_unix_sec,
            )
            acquired = True
            try:
                try:
                    self._remaining_timeout(deadline_unix_sec)
                except TimeoutError as error:
                    raise FanOutDeadlineExpiredBeforeDispatch(str(error)) from error
                if self._result_event is None:
                    assert self._result_values is None
                    self._result_values = []
                    self._result_event = asyncio.Event()
                    self._result_fan_out = self._fan_out
                    self._result_correlation_value = self._request_correlation_value(
                        obj
                    )
                    self._result_responders = set()
                    self._result_waiters = 0
                    self._result_dispatch_started = True
                    self._result_dispatch_completed = False

                    try:
                        if obj is not None:
                            self._send(obj)
                    except Exception as error:
                        partial_results = tuple(self._result_values)
                        self._clear_result_state()
                        raise FanOutCompletionUnknownError(
                            f"fan-out dispatch completion is unknown: {error}",
                            dispatch_started=True,
                            dispatch_completed=False,
                            partial_results=partial_results,
                            expected_count=self._fan_out,
                        ) from error
                    except BaseException:
                        self._clear_result_state()
                        raise
                    else:
                        self._result_dispatch_completed = True
                        dispatched = True

                # Keep generation-local refs because another waiter may clear
                # shared state after the event fires.
                values = self._result_values
                event = self._result_event
                fan_out = self._result_fan_out
                assert values is not None
                assert event is not None
                assert fan_out is not None
                assert self._result_waiters is not None
                dispatch_started = bool(self._result_dispatch_started)
                dispatch_completed = bool(self._result_dispatch_completed)
                dispatched = dispatch_completed
                self._result_waiters += 1
            finally:
                if acquired:
                    self._watching_lock.release()
            try:
                try:
                    timeout = self._remaining_timeout(deadline_unix_sec)
                    await asyncio.wait_for(event.wait(), timeout=timeout)
                except (TimeoutError, asyncio.TimeoutError) as error:
                    received = len(values)
                    raise FanOutCompletionUnknownError(
                        "fan-out response deadline expired: "
                        f"received {received}/{fan_out}",
                        dispatch_started=dispatch_started,
                        dispatch_completed=dispatch_completed,
                        partial_results=values,
                        expected_count=fan_out,
                    ) from error
                return copy.deepcopy(values)
            finally:
                if self._result_event is event:
                    assert self._result_waiters is not None
                    self._result_waiters -= 1
                    if event.is_set() or self._result_waiters == 0:
                        self._clear_result_state()
        except asyncio.CancelledError as error:
            if not dispatched:
                raise FanOutCancelledBeforeDispatch() from error
            raise

    async def __call__(self, obj, *, deadline_unix_sec: float | None = None):
        if self._mode == "queueing":
            return await self.queueing_call(
                obj,
                deadline_unix_sec=deadline_unix_sec,
            )
        else:
            return await self.watching_call(
                obj,
                deadline_unix_sec=deadline_unix_sec,
            )

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
        if self._responder_attr is not None:
            responder = getattr(recv_obj, self._responder_attr, None)
            if responder is None:
                logger.debug("Dropping communicator response without responder ID")
                return
            assert self._result_responders is not None
            if responder in self._result_responders:
                logger.debug(
                    "Dropping duplicate communicator response from %r",
                    responder,
                )
                return
            self._result_responders.add(responder)
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._result_fan_out:
            self._result_event.set()

    def _correlation_value(self, obj):
        if self._correlation_attr is None:
            return None
        return getattr(obj, self._correlation_attr, None)

    def _request_correlation_value(self, obj):
        value = self._correlation_value(obj)
        if self._correlation_attr is not None and value in (None, ""):
            raise ValueError(f"fan-out request requires {self._correlation_attr}")
        return value

    def _clear_result_state(self):
        self._result_event = self._result_values = None
        self._result_fan_out = None
        self._result_correlation_value = None
        self._result_responders = None
        self._result_waiters = None
        self._result_dispatch_started = None
        self._result_dispatch_completed = None

    @staticmethod
    def merge_results(results):
        all_success = all([r.success for r in results])
        all_message = [r.message for r in results]
        all_message = " | ".join(all_message)
        return all_success, all_message
