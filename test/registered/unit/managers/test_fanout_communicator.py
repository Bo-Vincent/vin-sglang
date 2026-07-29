"""Unit tests for FanOutCommunicator -- no server, no model loading."""

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.communicator import (
    FanOutCancelledBeforeDispatch,
    FanOutCommunicator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


async def _wait_until(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


class TestQueueingCall(CustomTestCase):
    def test_concurrent_caller_cannot_bypass_queue(self):
        """A new caller arriving in the wakeup window must not overtake a
        queued caller (this interleaving used to raise AssertionError and
        return 500 on concurrent /server_info requests)."""

        async def scenario():
            sent = []
            comm = FanOutCommunicator(send=sent.append, fan_out=1, mode="queueing")

            # A in-flight, B queued behind it.
            task_a = asyncio.create_task(comm("A"))
            await asyncio.sleep(0)
            task_b = asyncio.create_task(comm("B"))
            await asyncio.sleep(0)

            # Complete A, then create C before A's wakeup runs, so C's first
            # step lands between A's cleanup and B's wakeup.
            comm.handle_recv("resp-A")
            task_c = asyncio.create_task(comm("C"))

            # Drive to completion: feed a response whenever one is in flight.
            tasks = [task_a, task_b, task_c]
            for _ in range(100):
                if all(t.done() for t in tasks):
                    break
                if comm._result_event is not None and not comm._result_event.is_set():
                    comm.handle_recv(f"resp-{len(sent)}")
                await asyncio.sleep(0)

            # All callers complete without error, in strict FIFO order.
            await asyncio.gather(*tasks)
            self.assertEqual(sent, ["A", "B", "C"])

        asyncio.run(scenario())

    def test_cancelled_correlated_call_drops_its_late_response(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                mode="queueing",
                correlation_attr="transfer_id",
            )

            task_a = asyncio.create_task(
                comm(SimpleNamespace(transfer_id="transfer-a"))
            )
            await asyncio.sleep(0)
            task_a.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task_a

            task_b = asyncio.create_task(
                comm(SimpleNamespace(transfer_id="transfer-b"))
            )
            await asyncio.sleep(0)
            comm.handle_recv(SimpleNamespace(transfer_id="transfer-a"))
            await asyncio.sleep(0)
            self.assertFalse(task_b.done())

            comm.handle_recv(SimpleNamespace(transfer_id="transfer-b"))
            result = await task_b
            self.assertEqual(
                [item.transfer_id for item in result],
                ["transfer-b"],
            )
            self.assertEqual(
                [item.transfer_id for item in sent],
                ["transfer-a", "transfer-b"],
            )

        asyncio.run(scenario())

    def test_request_id_separates_late_phase_response(self):
        async def scenario():
            comm = FanOutCommunicator(
                send=lambda _: None,
                fan_out=1,
                correlation_attr="request_id",
            )
            commit = asyncio.create_task(
                comm(SimpleNamespace(request_id="commit-request"))
            )
            await asyncio.sleep(0)
            commit.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await commit

            cleanup = asyncio.create_task(
                comm(SimpleNamespace(request_id="cleanup-request"))
            )
            await asyncio.sleep(0)
            comm.handle_recv(SimpleNamespace(request_id="commit-request"))
            await asyncio.sleep(0)
            self.assertFalse(cleanup.done())
            comm.handle_recv(SimpleNamespace(request_id="cleanup-request"))
            await cleanup

        asyncio.run(scenario())

    def test_request_id_separates_late_timeout_retry_for_same_transfer(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                correlation_attr="request_id",
            )
            first_request = SimpleNamespace(
                transfer_id="transfer-1",
                request_id="renew-attempt-1",
            )
            with self.assertRaisesRegex(TimeoutError, "received 0/1"):
                await comm(
                    first_request,
                    deadline_unix_sec=time.time() + 0.01,
                )

            second_request = SimpleNamespace(
                transfer_id="transfer-1",
                request_id="renew-attempt-2",
            )
            retry = asyncio.create_task(
                comm(
                    second_request,
                    deadline_unix_sec=time.time() + 1,
                )
            )
            await _wait_until(lambda: len(sent) == 2)
            comm.handle_recv(
                SimpleNamespace(
                    transfer_id="transfer-1",
                    request_id="renew-attempt-1",
                )
            )
            await asyncio.sleep(0)
            self.assertFalse(retry.done())

            comm.handle_recv(
                SimpleNamespace(
                    transfer_id="transfer-1",
                    request_id="renew-attempt-2",
                )
            )
            result = await retry
            self.assertEqual(
                [item.request_id for item in result],
                ["renew-attempt-2"],
            )
            self.assertEqual(
                [item.request_id for item in sent],
                ["renew-attempt-1", "renew-attempt-2"],
            )

        asyncio.run(scenario())

    def test_duplicate_responder_does_not_complete_fanout(self):
        async def scenario():
            comm = FanOutCommunicator(
                send=lambda _: None,
                fan_out=2,
                correlation_attr="request_id",
                responder_attr="rank",
            )
            task = asyncio.create_task(comm(SimpleNamespace(request_id="request")))
            await asyncio.sleep(0)
            comm.handle_recv(SimpleNamespace(request_id="request", rank=0))
            comm.handle_recv(SimpleNamespace(request_id="request", rank=0))
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            comm.handle_recv(SimpleNamespace(request_id="request", rank=1))
            result = await task
            self.assertEqual([item.rank for item in result], [0, 1])

        asyncio.run(scenario())

    def test_missing_request_correlation_is_rejected_before_dispatch(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                correlation_attr="request_id",
                responder_attr="responder_id",
            )

            with self.assertRaisesRegex(ValueError, "request_id"):
                await comm(
                    SimpleNamespace(request_id=None),
                    deadline_unix_sec=time.time() + 0.01,
                )

            self.assertEqual(sent, [])

        asyncio.run(scenario())

    def test_missing_response_hits_deadline(self):
        async def scenario():
            comm = FanOutCommunicator(send=lambda _: None, fan_out=2)
            with self.assertRaisesRegex(TimeoutError, "received 0/2") as raised:
                await comm(
                    "request",
                    deadline_unix_sec=time.time() + 0.01,
                )

            self.assertEqual(
                type(raised.exception).__name__,
                "FanOutCompletionUnknownError",
            )
            self.assertTrue(raised.exception.completion_unknown)
            self.assertTrue(raised.exception.dispatch_started)
            self.assertTrue(raised.exception.dispatch_completed)
            self.assertEqual(raised.exception.received_count, 0)
            self.assertEqual(raised.exception.expected_count, 2)
            self.assertEqual(raised.exception.partial_results, ())

        asyncio.run(scenario())

    def test_partial_response_timeout_carries_completion_unknown_results(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(send=sent.append, fan_out=2)
            task = asyncio.create_task(
                comm(
                    "request",
                    deadline_unix_sec=time.time() + 0.02,
                )
            )
            await _wait_until(lambda: sent == ["request"])
            comm.handle_recv("rank-0")

            with self.assertRaisesRegex(TimeoutError, "received 1/2") as raised:
                await task

            self.assertTrue(raised.exception.completion_unknown)
            self.assertTrue(raised.exception.dispatch_started)
            self.assertTrue(raised.exception.dispatch_completed)
            self.assertEqual(raised.exception.received_count, 1)
            self.assertEqual(raised.exception.expected_count, 2)
            self.assertEqual(raised.exception.partial_results, ("rank-0",))

        asyncio.run(scenario())

    def test_send_failure_after_dispatch_attempt_is_completion_unknown(self):
        async def scenario():
            sent = []

            def partial_send(obj):
                sent.append(obj)
                raise RuntimeError("send failed after partial dispatch")

            comm = FanOutCommunicator(send=partial_send, fan_out=2)

            with self.assertRaisesRegex(
                TimeoutError,
                "send failed after partial dispatch",
            ) as raised:
                await comm("request")

            self.assertEqual(sent, ["request"])
            self.assertTrue(raised.exception.completion_unknown)
            self.assertTrue(raised.exception.dispatch_started)
            self.assertFalse(raised.exception.dispatch_completed)
            self.assertEqual(raised.exception.received_count, 0)
            self.assertEqual(raised.exception.expected_count, 2)
            self.assertEqual(raised.exception.partial_results, ())

        asyncio.run(scenario())

    def test_response_wait_uses_remaining_deadline(self):
        async def scenario():
            comm = FanOutCommunicator(send=lambda _: None, fan_out=1)

            started = time.monotonic()
            with patch(
                "sglang.srt.managers.communicator.time.time",
                side_effect=[0.0, 0.0, 9.99],
            ):
                with self.assertRaisesRegex(TimeoutError, "received 0/1"):
                    await comm("request", deadline_unix_sec=10.0)

            self.assertLess(time.monotonic() - started, 0.25)

        asyncio.run(scenario())

    def test_expired_request_is_not_dispatched(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(send=sent.append, fan_out=1)

            with self.assertRaisesRegex(TimeoutError, "before dispatch"):
                await comm(
                    "expired",
                    deadline_unix_sec=time.time() - 1,
                )

            self.assertEqual(sent, [])

        asyncio.run(scenario())

    def test_request_expiring_behind_queue_is_not_dispatched(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(send=sent.append, fan_out=1)

            first = asyncio.create_task(comm("first"))
            await asyncio.sleep(0)
            expired = asyncio.create_task(
                comm(
                    "expired",
                    deadline_unix_sec=time.time() + 60,
                )
            )
            await asyncio.sleep(0)

            with patch(
                "sglang.srt.managers.communicator.time.time",
                return_value=time.time() + 120,
            ):
                comm.handle_recv("first-response")
                await first
                with self.assertRaisesRegex(TimeoutError, "before dispatch"):
                    await expired
            self.assertEqual(sent, ["first"])

        asyncio.run(scenario())

    def test_deadline_bounds_queue_lock_wait(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(send=sent.append, fan_out=1)
            await comm._queueing_lock.acquire()
            try:
                started = time.monotonic()
                with self.assertRaisesRegex(TimeoutError, "before dispatch"):
                    await comm(
                        "expired",
                        deadline_unix_sec=time.time() + 0.02,
                    )
                self.assertLess(time.monotonic() - started, 0.25)
                self.assertEqual(sent, [])
            finally:
                comm._queueing_lock.release()

        asyncio.run(scenario())

    def test_cancellation_behind_queue_reports_not_dispatched(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(send=sent.append, fan_out=1)
            first = asyncio.create_task(comm("first"))
            await asyncio.sleep(0)
            queued = asyncio.create_task(comm("queued"))
            await asyncio.sleep(0)

            queued.cancel()
            with self.assertRaises(FanOutCancelledBeforeDispatch):
                await queued

            comm.handle_recv("first-response")
            await first
            self.assertEqual(sent, ["first"])

        asyncio.run(scenario())

    def test_cancelled_send_reports_not_dispatched(self):
        async def scenario():
            def cancel_send(_):
                raise asyncio.CancelledError

            comm = FanOutCommunicator(send=cancel_send, fan_out=1)

            with self.assertRaises(FanOutCancelledBeforeDispatch):
                await comm("request")

        asyncio.run(scenario())


class TestWatchingCall(CustomTestCase):
    def test_expired_request_is_not_dispatched(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                mode="watching",
            )

            with self.assertRaisesRegex(TimeoutError, "before dispatch"):
                await comm(
                    "expired",
                    deadline_unix_sec=time.time() - 1,
                )

            self.assertEqual(sent, [])

        asyncio.run(scenario())

    def test_deadline_is_rechecked_before_dispatch(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                mode="watching",
            )

            with patch(
                "sglang.srt.managers.communicator.time.time",
                side_effect=[0.0, 101.0],
            ):
                with self.assertRaisesRegex(TimeoutError, "before dispatch"):
                    await comm("expired", deadline_unix_sec=100.0)

            self.assertEqual(sent, [])

        asyncio.run(scenario())

    def test_deadline_bounds_watching_lock_wait(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                mode="watching",
            )
            await comm._watching_lock.acquire()
            try:
                started = time.monotonic()
                with self.assertRaisesRegex(TimeoutError, "before dispatch"):
                    await comm(
                        "expired",
                        deadline_unix_sec=time.time() + 0.02,
                    )
                self.assertLess(time.monotonic() - started, 0.25)
                self.assertEqual(sent, [])
            finally:
                comm._watching_lock.release()

        asyncio.run(scenario())

    def test_last_waiter_timeout_does_not_leave_stale_state(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                mode="watching",
            )

            with self.assertRaisesRegex(TimeoutError, "received 0/1"):
                await comm(
                    "first",
                    deadline_unix_sec=time.time() + 0.01,
                )

            second = asyncio.create_task(
                comm(
                    "second",
                    deadline_unix_sec=time.time() + 1,
                )
            )
            await _wait_until(lambda: len(sent) == 2)
            comm.handle_recv("second-response")

            self.assertEqual(await second, ["second-response"])
            self.assertEqual(sent, ["first", "second"])

        asyncio.run(scenario())

    def test_last_waiter_cancellation_does_not_leave_stale_state(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                mode="watching",
            )

            first = asyncio.create_task(comm("first"))
            await asyncio.sleep(0)
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first

            second = asyncio.create_task(comm("second"))
            await asyncio.sleep(0)
            comm.handle_recv("second-response")

            self.assertEqual(await second, ["second-response"])
            self.assertEqual(sent, ["first", "second"])

        asyncio.run(scenario())

    def test_cancelled_waiter_does_not_interrupt_other_waiter(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                mode="watching",
            )

            cancelled = asyncio.create_task(comm("request"))
            waiting = asyncio.create_task(comm("request"))
            await asyncio.sleep(0)
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled

            comm.handle_recv("response")
            self.assertEqual(await waiting, ["response"])
            self.assertEqual(sent, ["request"])

        asyncio.run(scenario())

    def test_timed_out_waiter_does_not_interrupt_other_waiter(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                mode="watching",
            )

            timed_out = asyncio.create_task(
                comm(
                    "request",
                    deadline_unix_sec=time.time() + 0.01,
                )
            )
            waiting = asyncio.create_task(
                comm(
                    "request",
                    deadline_unix_sec=time.time() + 1,
                )
            )
            with self.assertRaisesRegex(TimeoutError, "received 0/1"):
                await timed_out

            comm.handle_recv("response")
            self.assertEqual(await waiting, ["response"])
            self.assertEqual(sent, ["request"])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
