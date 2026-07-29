from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sglang.srt.weight_transfer.provider import (
        WeightTransferExecutionContext,
    )

_POLL_INTERVAL_SECONDS = 0.05
_DEFAULT_MAX_WORKERS = 4


class _ThreadedCallState(str, Enum):
    RUNNING = "running"
    ABANDONED = "abandoned"
    COMPLETED = "completed"


class _ThreadedCallAdmissionError(RuntimeError):
    def __init__(self, message: str, *, sealed: bool) -> None:
        super().__init__(message)
        self.sealed = sealed


class _BoundedExecutor:
    def __init__(self, *, max_workers: int, thread_name_prefix: str) -> None:
        if type(max_workers) is not int or max_workers <= 0:
            raise ValueError("max_workers must be a positive integer")
        if type(thread_name_prefix) is not str or not thread_name_prefix:
            raise ValueError("thread_name_prefix must be a non-empty string")
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._admission = threading.BoundedSemaphore(max_workers)
        self._lock = threading.Lock()
        self._sealed = False

    def submit(self, factory: Callable[[], Any]) -> Future[Any]:
        with self._lock:
            if self._sealed:
                raise _ThreadedCallAdmissionError(
                    "threaded call executor is sealed",
                    sealed=True,
                )
            if not self._admission.acquire(blocking=False):
                raise _ThreadedCallAdmissionError(
                    "threaded call executor has no admission capacity",
                    sealed=False,
                )
            try:
                return self._executor.submit(self._run_admitted, factory)
            except BaseException:
                self._admission.release()
                raise

    def _run_admitted(self, factory: Callable[[], Any]) -> Any:
        try:
            return factory()
        finally:
            self._admission.release()

    def seal(self) -> None:
        with self._lock:
            if self._sealed:
                return
            self._sealed = True
            self._executor.shutdown(wait=False)


_SHARED_THREADED_CALL_EXECUTOR = _BoundedExecutor(
    max_workers=_DEFAULT_MAX_WORKERS,
    thread_name_prefix="sglang-weight-transfer",
)


class _ThreadedCall:
    def __init__(self, executor: _BoundedExecutor | None = None) -> None:
        self._executor = executor or _SHARED_THREADED_CALL_EXECUTOR
        self._lock = threading.Lock()
        self._state = _ThreadedCallState.RUNNING
        self._was_abandoned = False
        self._started = False
        self._result: Any = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._future: Future[Any] | None = None
        self._cleanup_on_abandon: Callable[[], None] | None = None
        self._after_done: Callable[[], None] | None = None
        self.done = threading.Event()

    @property
    def state(self) -> _ThreadedCallState:
        with self._lock:
            return self._state

    @property
    def was_abandoned(self) -> bool:
        with self._lock:
            return self._was_abandoned

    @property
    def result(self) -> Any:
        with self._lock:
            return self._result

    @property
    def error(self) -> BaseException | None:
        with self._lock:
            return self._error

    @property
    def thread(self) -> threading.Thread | None:
        with self._lock:
            return self._thread

    def start(
        self,
        factory: Callable[[], Any],
        *,
        thread_name: str,
        before_done: Callable[[], None] | None = None,
        after_done: Callable[[], None] | None = None,
        cleanup_on_abandon: Callable[[], None] | None = None,
    ) -> None:
        self._begin_start(
            after_done=after_done,
            cleanup_on_abandon=cleanup_on_abandon,
        )
        try:
            future = self._executor.submit(
                lambda: self._run(
                    factory,
                    thread_name=thread_name,
                    before_done=before_done,
                )
            )
        except BaseException as error:
            self._complete(result=None, error=error)
        else:
            with self._lock:
                self._future = future

    def start_inline(
        self,
        factory: Callable[[], Any],
        *,
        thread_name: str,
        before_done: Callable[[], None] | None = None,
        after_done: Callable[[], None] | None = None,
        cleanup_on_abandon: Callable[[], None] | None = None,
    ) -> None:
        self._begin_start(
            after_done=after_done,
            cleanup_on_abandon=cleanup_on_abandon,
        )
        self._run(
            factory,
            thread_name=thread_name,
            before_done=before_done,
        )

    def _begin_start(
        self,
        *,
        after_done: Callable[[], None] | None,
        cleanup_on_abandon: Callable[[], None] | None,
    ) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("threaded call has already started")
            self._started = True
            self._cleanup_on_abandon = cleanup_on_abandon
            self._after_done = after_done

    def _run(
        self,
        factory: Callable[[], Any],
        *,
        thread_name: str,
        before_done: Callable[[], None] | None,
    ) -> None:
        thread = threading.current_thread()
        original_name = thread.name
        with self._lock:
            self._thread = thread
        thread.name = thread_name
        result: Any = None
        error: BaseException | None = None
        try:
            result = factory()
        except BaseException as caught:
            error = caught
        finally:
            if before_done is not None:
                try:
                    before_done()
                except BaseException as caught:
                    if error is None:
                        error = caught
            self._complete(result=result, error=error)
            thread.name = original_name

    def _complete(self, *, result: Any, error: BaseException | None) -> None:
        cleanup: Callable[[], None] | None = None
        abandoned = False
        after_done: Callable[[], None] | None
        with self._lock:
            if self._state is _ThreadedCallState.COMPLETED:
                return
            after_done = self._after_done
            if self._state is _ThreadedCallState.ABANDONED:
                abandoned = True
                cleanup = self._cleanup_on_abandon
            else:
                self._result = result
                self._error = error
                self._state = _ThreadedCallState.COMPLETED

        if cleanup is not None:
            try:
                cleanup()
            except BaseException as caught:
                if error is None:
                    error = caught

        if abandoned:
            with self._lock:
                self._result = result
                self._error = error
                self._state = _ThreadedCallState.COMPLETED

        if after_done is not None:
            try:
                after_done()
            except BaseException as caught:
                with self._lock:
                    if self._error is None:
                        self._error = caught
        self.done.set()

    def _abandon(self) -> bool:
        with self._lock:
            if self._state is _ThreadedCallState.COMPLETED:
                return False
            if self._state is _ThreadedCallState.RUNNING:
                self._state = _ThreadedCallState.ABANDONED
                self._was_abandoned = True
            return True

    def result_before(
        self,
        execution_context: WeightTransferExecutionContext,
        *,
        interrupted: Callable[[], BaseException],
    ) -> Any:
        while True:
            if self.done.is_set():
                error = self.error
                if error is not None:
                    raise error
                return self.result
            if execution_context.expired():
                if self._abandon():
                    raise interrupted()
                continue
            self.done.wait(
                timeout=min(
                    _POLL_INTERVAL_SECONDS,
                    execution_context.remaining_seconds(),
                )
            )
