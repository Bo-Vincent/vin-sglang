import asyncio
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import (  # noqa: E402
    multi_tokenizer_mixin as multi_tokenizer_mixin_module,
)
from sglang.srt.managers import (  # noqa: E402
    tokenizer_manager as tokenizer_manager_module,
)
from sglang.srt.managers.io_struct import (  # noqa: E402
    ContinueGenerationReqInput,
    PauseContinueBroadcastReq,
    PauseGenerationReqInput,
    TokenizerWorkerRegistrationReq,
)
from sglang.srt.managers.multi_tokenizer_mixin import (  # noqa: E402
    MultiTokenizerRouter,
    TokenizerWorker,
)
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.utils import TypeBasedDispatcher  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RecordingSocketMapping:
    def __init__(self, events, *, fail_once=None):
        self.events = events
        self.fail_once = fail_once
        self.attempts = defaultdict(int)

    def send_output(self, worker_ipc, output):
        state = output.http_worker_ipc
        key = (state, worker_ipc)
        self.attempts[key] += 1
        if self.fail_once == key and self.attempts[key] == 1:
            self.events.append(("fanout-error", state, worker_ipc))
            raise RuntimeError(f"fanout failed for {worker_ipc}")
        self.events.append(("fanout", state, worker_ipc))

    def clear_socket(self, worker_ipc):
        self.events.append(("socket-closed", None, worker_ipc))


def _make_router(socket_mapping):
    router = object.__new__(MultiTokenizerRouter)
    router.send_to_scheduler = object()
    router.server_args = SimpleNamespace(tokenizer_worker_num=2)
    router.all_worker_ipcs = {"worker-a", "worker-b"}
    pid = multi_tokenizer_mixin_module.os.getpid()
    process_start_time = multi_tokenizer_mixin_module.psutil.Process(pid).create_time()
    worker_a = multi_tokenizer_mixin_module._TokenizerWorkerIdentity(
        ipc_name="worker-a",
        pid=pid,
        process_start_time=process_start_time,
        token="worker-a",
    )
    worker_b = multi_tokenizer_mixin_module._TokenizerWorkerIdentity(
        ipc_name="worker-b",
        pid=pid,
        process_start_time=process_start_time,
        token="worker-b",
    )
    router._worker_registrations = {
        worker_a.ipc_name: worker_a,
        worker_b.ipc_name: worker_b,
    }
    router.socket_mapping = socket_mapping
    router.pause_owners = {"remote-weight-transfer:last"}
    router.active_remote_pause_owner = "remote-weight-transfer:last"
    router.pending_remote_pause_requests = deque()
    router._pause_transitions = {}
    router._pause_poisoned_owners = set()
    router._pause_owner_transitions = {}
    router._pause_owner_workers = {
        "remote-weight-transfer:last": worker_a,
    }
    router._pause_fail_stopped = False
    return router


def _continue_request(owner="remote-weight-transfer:last", origin="worker-a"):
    return ContinueGenerationReqInput(
        rid=owner,
        http_worker_ipc=origin,
        torch_empty_cache=False,
    )


def _identity(request):
    identity = multi_tokenizer_mixin_module._decode_pause_transition(request.rid)
    assert identity is not None
    return identity


def _ack(router, identity, worker_ipc, *, applied=False):
    rid = (
        multi_tokenizer_mixin_module._encode_pause_transition_applied(identity)
        if applied
        else multi_tokenizer_mixin_module._encode_pause_transition(identity)
    )
    router._handle_pause_continue_ack(
        PauseContinueBroadcastReq(
            rid=rid,
            is_pause=identity.expected_state,
            http_worker_ipc=worker_ipc,
        )
    )


def _committed_ack(router, identity, worker_ipc, *, finalize=True):
    router._handle_pause_continue_ack(
        PauseContinueBroadcastReq(
            rid=multi_tokenizer_mixin_module._encode_pause_transition_committed_ack(
                identity
            ),
            is_pause=identity.expected_state,
            http_worker_ipc=worker_ipc,
        )
    )
    transition = router._pause_transitions.get(identity.transition_id)
    if (
        finalize
        and transition is not None
        and transition.committed
        and not transition.commit_pending_workers
    ):
        _finalized_ack(router, identity, transition.origin_worker_ipc)


def _finalized_ack(router, identity, worker_ipc):
    router._handle_pause_continue_ack(
        PauseContinueBroadcastReq(
            rid=multi_tokenizer_mixin_module._encode_pause_transition_finalized_ack(
                identity
            ),
            is_pause=identity.expected_state,
            http_worker_ipc=worker_ipc,
        )
    )


def _worker_registration(
    worker_ipc,
    *,
    pid,
    process_start_time,
    worker_token,
    unregister=False,
):
    return TokenizerWorkerRegistrationReq(
        worker_ipc_name=worker_ipc,
        worker_pid=pid,
        process_start_time=process_start_time,
        worker_token=worker_token,
        unregister=unregister,
    )


async def _start_last_owner_continue(monkeypatch, *, fail_once=None):
    events = []

    async def send_to_scheduler(_socket, request):
        events.append(("scheduler", type(request).__name__, request.rid))

    monkeypatch.setattr(
        multi_tokenizer_mixin_module,
        "async_sock_send",
        send_to_scheduler,
    )
    mapping = _RecordingSocketMapping(events, fail_once=fail_once)
    router = _make_router(mapping)
    request = _continue_request()
    await router._handle_pause_continue_request(request)
    return router, request, _identity(request), mapping, events


def _make_worker(identity):
    worker = object.__new__(TokenizerWorker)
    worker.is_pause = True
    worker.is_pause_cond = asyncio.Condition()
    worker._generation_pause_owners = {identity.owner}
    worker._generation_pause_resume_pending = set()
    worker._prepared_pause_transitions = {identity.transition_id: identity}
    worker._confirmed_pause_transitions = {}
    worker._poisoned_pause_transitions = {}
    worker._latest_pause_transitions = {identity.owner: identity}
    worker._pause_continue_futures = {}
    worker._pause_continue_confirmation_futures = {}
    worker._committed_pause_transitions = {}
    acks = []

    async def dispatch_ack(ack):
        acks.append(ack)

    worker._async_dispatch_to_scheduler = dispatch_ack
    return worker, acks


def test_scheduler_resume_waits_for_all_frozen_workers(monkeypatch) -> None:
    async def scenario():
        router, _request, identity, _mapping, events = await _start_last_owner_continue(
            monkeypatch
        )

        assert not any(event[0] == "scheduler" for event in events)
        _ack(router, identity, "worker-a")
        _ack(router, identity, "worker-b")
        assert not any(event[0] == "scheduler" for event in events)

        _ack(router, identity, "worker-a", applied=True)
        assert not any(event[0] == "scheduler" for event in events)
        _ack(router, identity, "worker-b", applied=True)
        await asyncio.sleep(0)
        _committed_ack(router, identity, "worker-a")
        _committed_ack(router, identity, "worker-b")

        scheduler_index = next(
            index for index, event in enumerate(events) if event[0] == "scheduler"
        )
        confirmed_indices = [
            index
            for index, event in enumerate(events)
            if event[:2]
            == ("fanout", multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED)
        ]
        assert len(confirmed_indices) == 2
        assert max(confirmed_indices) < scheduler_index

    asyncio.run(scenario())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_next_pause_waits_for_irreversible_continue_commit(monkeypatch) -> None:
    async def scenario():
        events = []
        resume_started = asyncio.Event()
        allow_resume = asyncio.Event()

        async def send_to_scheduler(_socket, request):
            events.append(("scheduler", type(request).__name__, request.rid))
            if isinstance(request, ContinueGenerationReqInput):
                resume_started.set()
                await allow_resume.wait()

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        router = _make_router(_RecordingSocketMapping(events))
        continue_request = _continue_request()
        await router._handle_pause_continue_request(continue_request)
        continue_identity = _identity(continue_request)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, continue_identity, worker_ipc)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, continue_identity, worker_ipc, applied=True)
        await resume_started.wait()

        pause_request = PauseGenerationReqInput(
            mode="in_place",
            rid="remote-weight-transfer:next",
            http_worker_ipc="worker-b",
        )
        pause_task = asyncio.create_task(
            router._handle_pause_continue_request(pause_request)
        )
        await asyncio.sleep(0)
        assert pause_task.done() is False
        assert [event[1] for event in events if event[0] == "scheduler"] == [
            "ContinueGenerationReqInput"
        ]

        allow_resume.set()
        await asyncio.sleep(0)
        _committed_ack(router, continue_identity, "worker-a")
        _committed_ack(router, continue_identity, "worker-b")
        await pause_task
        assert [event[1] for event in events if event[0] == "scheduler"] == [
            "ContinueGenerationReqInput",
            "PauseGenerationReqInput",
        ]

        pause_identity = _identity(pause_request)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, pause_identity, worker_ipc)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, pause_identity, worker_ipc, applied=True)
        await asyncio.sleep(0)
        for worker_ipc in ("worker-a", "worker-b"):
            _committed_ack(router, pause_identity, worker_ipc)

    asyncio.run(scenario())


def test_partial_confirmation_send_fails_closed_before_scheduler_resume(
    monkeypatch,
) -> None:
    async def scenario():
        router, _request, identity, _mapping, events = await _start_last_owner_continue(
            monkeypatch,
            fail_once=(
                multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
                "worker-a",
            ),
        )

        _ack(router, identity, "worker-a")
        _ack(router, identity, "worker-b")

        assert not any(event[0] == "scheduler" for event in events)
        assert identity.transition_id not in router._pause_transitions
        assert router.pause_owners == {identity.owner}
        assert router._pause_poisoned_owners == {identity.owner}
        failed_workers = {
            worker
            for kind, state, worker in events
            if kind == "fanout"
            and state == multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED
        }
        assert failed_workers == {"worker-a", "worker-b"}

    asyncio.run(scenario())


def test_partial_commit_send_retries_without_reverting_terminal_state(
    monkeypatch,
) -> None:
    async def scenario():
        router, _request, identity, mapping, events = await _start_last_owner_continue(
            monkeypatch,
            fail_once=(
                multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED,
                "worker-b",
            ),
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RETRY_INTERVAL_SEC",
            0.0,
            raising=False,
        )

        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc, applied=True)
        for _ in range(10):
            await asyncio.sleep(0)
            if (
                mapping.attempts[
                    (
                        multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED,
                        "worker-b",
                    )
                ]
                >= 2
            ):
                break
        for worker_ipc in ("worker-a", "worker-b"):
            _committed_ack(router, identity, worker_ipc)

        assert sum(event[0] == "scheduler" for event in events) == 1
        assert not any(
            event[1] == multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED
            for event in events
            if event[0] in {"fanout", "fanout-error"}
        )
        assert (
            mapping.attempts[
                (multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED, "worker-b")
            ]
            == 2
        )
        assert router.pause_owners == set()
        assert router._pause_poisoned_owners == set()
        assert identity.transition_id not in router._pause_transitions

    asyncio.run(scenario())


def test_permanent_commit_fanout_failure_is_bounded_and_stops_service(
    monkeypatch,
) -> None:
    async def scenario():
        events = []
        stopped = asyncio.Event()

        class PermanentCommitFailureMapping(_RecordingSocketMapping):
            def send_output(self, worker_ipc, output):
                if (
                    output.http_worker_ipc
                    == multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED
                    and worker_ipc == "worker-b"
                ):
                    key = (output.http_worker_ipc, worker_ipc)
                    self.attempts[key] += 1
                    self.events.append(("fanout-error", *key))
                    raise RuntimeError("worker-b unavailable")
                super().send_output(worker_ipc, output)

        def stop_service(_pid, include_parent):
            assert include_parent is True
            stopped.set()

        async def send_to_scheduler(_socket, request):
            events.append(("scheduler", type(request).__name__, request.rid))

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RETRY_INTERVAL_SEC",
            0.005,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC",
            1.0,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "kill_process_tree",
            stop_service,
        )
        mapping = PermanentCommitFailureMapping(events)
        router = _make_router(mapping)
        identity = multi_tokenizer_mixin_module._PauseTransitionIdentity(
            transition_id="commit-near-original-deadline",
            owner="remote-weight-transfer:last",
            action="continue",
            expected_state=False,
            deadline_monotonic_ns=time.monotonic_ns() + 100_000_000,
        )
        request = ContinueGenerationReqInput(
            rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
            http_worker_ipc="worker-a",
            torch_empty_cache=False,
        )
        await router._handle_pause_continue_request(request)
        transition = router._pause_transitions[identity.transition_id]
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc, applied=True)

        await asyncio.wait_for(stopped.wait(), timeout=0.3)
        await asyncio.wait_for(
            router._wait_for_irrevocable_pause_commit(),
            timeout=0.05,
        )

        attempts = mapping.attempts[
            (multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED, "worker-b")
        ]
        assert 1 <= attempts < 100
        assert transition.commit_done.result() is False
        assert router.pause_owners == {identity.owner}
        assert router._pause_poisoned_owners == {identity.owner}
        assert router.active_remote_pause_owner == identity.owner
        assert identity.transition_id not in router._pause_transitions
        failed_workers = {
            worker
            for kind, state, worker in events
            if kind == "fanout"
            and state == multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED
        }
        assert failed_workers == {"worker-a", "worker-b"}

    asyncio.run(scenario())


def test_scheduler_resume_cannot_extend_original_deadline(
    monkeypatch,
) -> None:
    async def scenario():
        events = []
        scheduler_attempts = 0
        resume_started = asyncio.Event()
        never_resume = asyncio.Event()
        stopped = asyncio.Event()

        async def send_to_scheduler(_socket, request):
            nonlocal scheduler_attempts
            scheduler_attempts += 1
            resume_started.set()
            await never_resume.wait()

        def stop_service(_pid, include_parent):
            assert include_parent is True
            stopped.set()

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RETRY_INTERVAL_SEC",
            0.005,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC",
            1.0,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "kill_process_tree",
            stop_service,
        )
        router = _make_router(_RecordingSocketMapping(events))
        identity = multi_tokenizer_mixin_module._PauseTransitionIdentity(
            transition_id="resume-near-original-deadline",
            owner="remote-weight-transfer:last",
            action="continue",
            expected_state=False,
            deadline_monotonic_ns=time.monotonic_ns() + 100_000_000,
        )
        request = ContinueGenerationReqInput(
            rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
            http_worker_ipc="worker-a",
            torch_empty_cache=False,
        )
        await router._handle_pause_continue_request(request)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc, applied=True)

        await asyncio.wait_for(resume_started.wait(), timeout=0.05)
        await asyncio.wait_for(stopped.wait(), timeout=0.3)
        await asyncio.wait_for(
            router._wait_for_irrevocable_pause_commit(),
            timeout=0.05,
        )

        assert 1 <= scheduler_attempts < 100
        assert router.pause_owners == {identity.owner}
        assert router._pause_poisoned_owners == {identity.owner}
        assert identity.transition_id not in router._pause_transitions
        failed_workers = {
            worker
            for kind, state, worker in events
            if kind == "fanout"
            and state == multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED
        }
        assert failed_workers == {"worker-a", "worker-b"}

    asyncio.run(scenario())


def test_confirmation_keeps_admission_closed_until_commit() -> None:
    async def scenario():
        identity = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner="remote-weight-transfer:last",
            action="continue",
            expected_state=False,
        )
        worker_a, _ = _make_worker(identity)
        worker_b, _ = _make_worker(identity)
        confirmed = PauseContinueBroadcastReq(
            rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
            is_pause=False,
            http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
        )

        await TokenizerWorker._apply_pause_continue_broadcast(worker_a, confirmed)
        await TokenizerWorker._apply_pause_continue_broadcast(worker_b, confirmed)
        assert worker_a.is_pause is True
        assert worker_b.is_pause is True
        assert worker_a._generation_pause_owners == {identity.owner}
        assert worker_b._generation_pause_owners == {identity.owner}

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker_a,
            PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED,
            ),
        )
        assert worker_a.is_pause is False
        assert worker_b.is_pause is True

    asyncio.run(scenario())


def test_permanent_scheduler_resume_failure_is_bounded_and_stops_service(
    monkeypatch,
) -> None:
    async def scenario():
        events = []
        stopped = asyncio.Event()
        scheduler_attempts = 0

        async def send_to_scheduler(_socket, _request):
            nonlocal scheduler_attempts
            scheduler_attempts += 1
            raise RuntimeError("scheduler unavailable")

        def stop_service(_pid, include_parent):
            assert include_parent is True
            stopped.set()

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RETRY_INTERVAL_SEC",
            0.001,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC",
            0.01,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "kill_process_tree",
            stop_service,
        )
        router = _make_router(_RecordingSocketMapping(events))
        request = _continue_request()
        await router._handle_pause_continue_request(request)
        identity = _identity(request)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc, applied=True)

        await asyncio.wait_for(stopped.wait(), timeout=0.2)

        assert scheduler_attempts >= 1
        assert scheduler_attempts < 100
        assert router.pause_owners == {identity.owner}
        assert router._pause_poisoned_owners == {identity.owner}
        assert identity.transition_id not in router._pause_transitions
        failed_workers = {
            worker
            for kind, state, worker in events
            if kind == "fanout"
            and state == multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED
        }
        assert failed_workers == {"worker-a", "worker-b"}

    asyncio.run(scenario())


def test_late_committed_apply_and_finalization_ignore_origin_deadline() -> None:
    async def scenario():
        identity = multi_tokenizer_mixin_module._PauseTransitionIdentity(
            transition_id="late-terminal",
            owner="remote-weight-transfer:last",
            action="continue",
            expected_state=False,
            deadline_monotonic_ns=time.monotonic_ns() - 1,
        )
        worker, acks = _make_worker(identity)
        worker.is_pause = False
        worker._generation_pause_owners = set()
        worker._prepared_pause_transitions = {}
        worker._confirmed_pause_transitions = {identity.transition_id: identity}
        pending = asyncio.get_running_loop().create_future()
        worker._pause_continue_futures = {
            identity.transition_id: (identity, pending),
        }

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED,
            ),
        )

        assert pending.done() is False
        assert (
            multi_tokenizer_mixin_module._decode_pause_transition_committed_ack(
                acks[-1].rid
            )
            == identity
        )
        assert worker.is_pause is False
        assert worker._generation_pause_owners == set()
        assert worker._generation_pause_resume_pending == set()
        assert worker._poisoned_pause_transitions == {}

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_FINALIZED,
            ),
        )
        assert pending.result() is True

    asyncio.run(scenario())


def test_duplicate_and_stale_ack_cannot_advance_commit(monkeypatch) -> None:
    async def scenario():
        router, _request, identity, _mapping, events = await _start_last_owner_continue(
            monkeypatch
        )
        stale = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner=identity.owner,
            action=identity.action,
            expected_state=identity.expected_state,
        )

        _ack(router, identity, "worker-a")
        _ack(router, identity, "worker-a")
        _ack(router, stale, "worker-b")
        transition = router._pause_transitions[identity.transition_id]
        assert transition.acked_workers == {"worker-a"}

        _ack(router, identity, "worker-b")
        _ack(router, identity, "worker-a", applied=True)
        _ack(router, identity, "worker-a", applied=True)
        _ack(router, stale, "worker-b", applied=True)
        assert transition.applied_workers == {"worker-a"}
        assert not any(event[0] == "scheduler" for event in events)

        _ack(router, identity, "worker-b", applied=True)
        await asyncio.sleep(0)
        assert sum(event[0] == "scheduler" for event in events) == 1
        _committed_ack(router, identity, "worker-a")
        _committed_ack(router, identity, "worker-b")

    asyncio.run(scenario())


def test_commit_waits_for_every_worker_applied_ack(monkeypatch) -> None:
    async def scenario():
        router, _request, identity, _mapping, events = await _start_last_owner_continue(
            monkeypatch
        )
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc, applied=True)
        await asyncio.sleep(0)

        transition = router._pause_transitions[identity.transition_id]
        assert transition.committed is True
        assert transition.commit_pending_workers == {"worker-a", "worker-b"}
        assert transition.commit_done.done() is False
        assert {
            worker
            for kind, state, worker in events
            if kind == "fanout"
            and state == multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED
        } == {"worker-a", "worker-b"}

        _committed_ack(router, identity, "worker-a", finalize=False)
        assert transition.commit_pending_workers == {"worker-b"}
        assert transition.commit_done.done() is False

        _committed_ack(router, identity, "worker-b", finalize=False)
        assert transition.commit_pending_workers == set()
        assert transition.commit_done.done() is False
        assert router._pause_transitions[identity.transition_id] is transition

        _finalized_ack(router, identity, "worker-a")
        assert transition.commit_done.result() is True
        assert identity.transition_id not in router._pause_transitions
        assert identity.owner not in router._pause_owner_transitions

    asyncio.run(scenario())


def test_finalized_is_retried_until_origin_ack(monkeypatch) -> None:
    async def scenario():
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RETRY_INTERVAL_SEC",
            0.001,
        )
        router, _request, identity, mapping, _events = await _start_last_owner_continue(
            monkeypatch
        )
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc, applied=True)
        await asyncio.sleep(0)
        transition = router._pause_transitions[identity.transition_id]

        for worker_ipc in ("worker-a", "worker-b"):
            _committed_ack(router, identity, worker_ipc, finalize=False)

        await asyncio.sleep(0.01)
        finalized_key = (
            multi_tokenizer_mixin_module._PAUSE_TRANSITION_FINALIZED,
            "worker-a",
        )
        assert mapping.attempts[finalized_key] >= 2
        assert router._pause_transitions[identity.transition_id] is transition
        assert transition.commit_done.done() is False

        _finalized_ack(router, identity, "worker-a")
        assert transition.commit_done.result() is True
        assert identity.transition_id not in router._pause_transitions

        _finalized_ack(router, identity, "worker-a")
        assert transition.commit_done.result() is True

    asyncio.run(scenario())


def test_origin_finalization_wait_is_bounded_and_fail_closed(monkeypatch) -> None:
    async def scenario():
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC",
            0.01,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_CONTINUE_ACK_TIMEOUT_SEC",
            0.01,
        )
        owner = "remote-weight-transfer:lost-finalized"
        worker = object.__new__(TokenizerWorker)
        worker.is_pause = True
        worker.is_pause_cond = asyncio.Condition()
        worker._generation_pause_owners = {owner}
        worker._generation_pause_resume_pending = set()
        worker._pause_continue_futures = {}
        worker._pause_continue_confirmation_futures = {}
        worker._prepared_pause_transitions = {}
        worker._confirmed_pause_transitions = {}
        worker._committed_pause_transitions = {}
        worker._poisoned_pause_transitions = {}
        worker._latest_pause_transitions = {}

        async def dispatch(request):
            if isinstance(request, PauseContinueBroadcastReq):
                return
            identity = _identity(request)
            await TokenizerWorker._apply_pause_continue_broadcast(
                worker,
                PauseContinueBroadcastReq(
                    rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                    is_pause=False,
                ),
            )
            await TokenizerWorker._apply_pause_continue_broadcast(
                worker,
                PauseContinueBroadcastReq(
                    rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                    is_pause=False,
                    http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
                ),
            )
            await TokenizerWorker._apply_pause_continue_broadcast(
                worker,
                PauseContinueBroadcastReq(
                    rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                    is_pause=False,
                    http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED,
                ),
            )

        worker._async_dispatch_to_scheduler = dispatch
        request = ContinueGenerationReqInput(
            rid=owner,
            torch_empty_cache=False,
        )

        with pytest.raises(
            TimeoutError,
            match="pause transition acknowledgement deadline expired",
        ):
            await asyncio.wait_for(
                TokenizerWorker._release_generation_pause(
                    worker,
                    owner,
                    request,
                ),
                timeout=0.2,
            )

        identity = _identity(request)
        assert worker.is_pause is True
        assert worker._generation_pause_owners == {identity.owner}
        assert worker._generation_pause_resume_pending == {identity.owner}
        assert worker._poisoned_pause_transitions == {identity.transition_id: identity}

    asyncio.run(scenario())


def test_cancelled_origin_does_not_ack_late_finalized() -> None:
    async def scenario():
        owner = "remote-weight-transfer:cancelled-finalized"
        worker = object.__new__(TokenizerWorker)
        worker.is_pause = True
        worker.is_pause_cond = asyncio.Condition()
        worker._generation_pause_owners = {owner}
        worker._generation_pause_resume_pending = set()
        worker._pause_continue_futures = {}
        worker._pause_continue_confirmation_futures = {}
        worker._prepared_pause_transitions = {}
        worker._confirmed_pause_transitions = {}
        worker._committed_pause_transitions = {}
        worker._poisoned_pause_transitions = {}
        worker._latest_pause_transitions = {}
        committed = asyncio.Event()
        acknowledgements = []
        captured_identity = None

        async def dispatch(request):
            nonlocal captured_identity
            if isinstance(request, PauseContinueBroadcastReq):
                acknowledgements.append(request)
                return
            identity = _identity(request)
            captured_identity = identity

            async def drive_transition():
                for state in (
                    None,
                    multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
                    multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED,
                ):
                    await TokenizerWorker._apply_pause_continue_broadcast(
                        worker,
                        PauseContinueBroadcastReq(
                            rid=multi_tokenizer_mixin_module._encode_pause_transition(
                                identity
                            ),
                            is_pause=False,
                            http_worker_ipc=state,
                        ),
                    )
                committed.set()

            asyncio.create_task(drive_transition())

        worker._async_dispatch_to_scheduler = dispatch
        request = ContinueGenerationReqInput(
            rid=owner,
            torch_empty_cache=False,
        )
        task = asyncio.create_task(
            TokenizerWorker._release_generation_pause(
                worker,
                owner,
                request,
            )
        )
        await asyncio.wait_for(committed.wait(), timeout=0.1)
        await asyncio.sleep(0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert captured_identity is not None
        identity = captured_identity
        assert worker.is_pause is True
        assert worker._generation_pause_owners == {owner}
        assert worker._generation_pause_resume_pending == {owner}
        assert worker._poisoned_pause_transitions == {identity.transition_id: identity}
        acknowledgements_before_finalized = len(acknowledgements)

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_FINALIZED,
            ),
        )

        assert len(acknowledgements) == acknowledgements_before_finalized
        assert worker.is_pause is True
        assert worker._generation_pause_owners == {owner}

    asyncio.run(scenario())


def test_inflight_finalized_ack_prevents_late_cancellation_rollback() -> None:
    async def scenario():
        owner = "remote-weight-transfer:inflight-finalized"
        worker = object.__new__(TokenizerWorker)
        worker.is_pause = True
        worker.is_pause_cond = asyncio.Condition()
        worker._generation_pause_owners = {owner}
        worker._generation_pause_resume_pending = set()
        worker._pause_continue_futures = {}
        worker._pause_continue_confirmation_futures = {}
        worker._prepared_pause_transitions = {}
        worker._confirmed_pause_transitions = {}
        worker._committed_pause_transitions = {}
        worker._poisoned_pause_transitions = {}
        worker._latest_pause_transitions = {}
        committed = asyncio.Event()
        ack_started = asyncio.Event()
        release_ack = asyncio.Event()
        finalized_acks = []
        captured_identity = None

        async def dispatch(request):
            nonlocal captured_identity
            if isinstance(request, PauseContinueBroadcastReq):
                if (
                    multi_tokenizer_mixin_module._decode_pause_transition_finalized_ack(
                        request.rid
                    )
                    is not None
                ):
                    finalized_acks.append(request)
                    ack_started.set()
                    await release_ack.wait()
                return
            identity = _identity(request)
            captured_identity = identity

            async def drive_transition():
                for state in (
                    None,
                    multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
                    multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED,
                ):
                    await TokenizerWorker._apply_pause_continue_broadcast(
                        worker,
                        PauseContinueBroadcastReq(
                            rid=multi_tokenizer_mixin_module._encode_pause_transition(
                                identity
                            ),
                            is_pause=False,
                            http_worker_ipc=state,
                        ),
                    )
                committed.set()

            asyncio.create_task(drive_transition())

        worker._async_dispatch_to_scheduler = dispatch
        request = ContinueGenerationReqInput(
            rid=owner,
            torch_empty_cache=False,
        )
        origin_task = asyncio.create_task(
            TokenizerWorker._release_generation_pause(
                worker,
                owner,
                request,
            )
        )
        await asyncio.wait_for(committed.wait(), timeout=0.1)
        await asyncio.sleep(0)

        assert captured_identity is not None
        identity = captured_identity
        finalized_task = asyncio.create_task(
            TokenizerWorker._apply_pause_continue_broadcast(
                worker,
                PauseContinueBroadcastReq(
                    rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                    is_pause=False,
                    http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_FINALIZED,
                ),
            )
        )
        await asyncio.wait_for(ack_started.wait(), timeout=0.1)

        origin_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await origin_task

        try:
            assert worker.is_pause is False
            assert worker._generation_pause_owners == set()
            assert worker._generation_pause_resume_pending == set()
            assert worker._poisoned_pause_transitions == {}
        finally:
            release_ack.set()
            await finalized_task

        assert len(finalized_acks) == 1

    asyncio.run(scenario())


def test_worker_exit_after_committed_enqueue_hits_ack_timeout(
    monkeypatch,
) -> None:
    async def scenario():
        stopped = asyncio.Event()
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC",
            0.01,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_TRANSITION_RETRY_INTERVAL_SEC",
            0.001,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "kill_process_tree",
            lambda _pid, include_parent: stopped.set() if include_parent else None,
        )
        router, _request, identity, mapping, _events = await _start_last_owner_continue(
            monkeypatch
        )
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc, applied=True)
        await asyncio.sleep(0)
        _committed_ack(router, identity, "worker-a")

        await asyncio.wait_for(stopped.wait(), timeout=0.2)

        assert (
            mapping.attempts[
                (multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED, "worker-b")
            ]
            >= 1
        )
        assert identity.transition_id not in router._pause_transitions
        assert router._pause_poisoned_owners == {identity.owner}
        assert router._pause_fail_stopped is True

    asyncio.run(scenario())


def test_committed_ack_after_deadline_stops_service(monkeypatch) -> None:
    async def scenario():
        stopped = asyncio.Event()
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "kill_process_tree",
            lambda _pid, include_parent: stopped.set() if include_parent else None,
        )
        router, _request, identity, _mapping, _events = (
            await _start_last_owner_continue(monkeypatch)
        )
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc)
        for worker_ipc in ("worker-a", "worker-b"):
            _ack(router, identity, worker_ipc, applied=True)
        await asyncio.sleep(0)
        transition = router._pause_transitions[identity.transition_id]
        transition.commit_deadline_monotonic_ns = time.monotonic_ns() - 1

        _committed_ack(router, identity, "worker-a")
        _committed_ack(router, identity, "worker-b")

        await asyncio.wait_for(stopped.wait(), timeout=0.1)
        assert transition.commit_done.result() is False
        assert identity.transition_id not in router._pause_transitions

    asyncio.run(scenario())


def test_worker_rejects_stale_committed_identity_without_ack() -> None:
    async def scenario():
        current = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner="remote-weight-transfer:last",
            action="continue",
            expected_state=False,
        )
        stale = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner=current.owner,
            action=current.action,
            expected_state=current.expected_state,
        )
        worker, acks = _make_worker(current)
        worker._confirmed_pause_transitions = {stale.transition_id: stale}

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(stale),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED,
            ),
        )

        assert acks == []
        assert worker.is_pause is True
        assert worker._committed_pause_transitions == {}

    asyncio.run(scenario())


def test_worker_applies_router_registration_state_before_admission() -> None:
    async def scenario():
        worker = object.__new__(TokenizerWorker)
        worker._worker_token = "worker-token"
        worker._router_registration_result = None
        worker._router_registration_future = asyncio.get_running_loop().create_future()
        worker.is_pause = False
        worker.is_pause_cond = asyncio.Condition()

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_worker_registration(
                    worker._worker_token
                ),
                is_pause=True,
                http_worker_ipc=multi_tokenizer_mixin_module._WORKER_REGISTRATION_ACCEPTED,
            ),
        )

        assert worker._router_registration_result is True
        assert worker._router_registration_future.result() is True
        assert worker.is_pause is True

    asyncio.run(scenario())


def test_registration_wait_starts_receive_loop_and_retries(monkeypatch) -> None:
    async def scenario():
        worker = object.__new__(TokenizerWorker)
        worker.worker_id = 101
        worker.tokenizer_ipc_name = "worker-a"
        worker._worker_process_start_time = 1.0
        worker._worker_token = "worker-token"
        worker._router_registration_result = None
        worker._router_registration_future = None
        worker._router_unregistered = False
        worker.is_pause = False
        worker.is_pause_cond = asyncio.Condition()
        worker.recv_from_detokenizer = object()
        worker.soft_watchdog = SimpleNamespace(
            disable=lambda: nullcontext(),
            feed=lambda: None,
        )
        worker._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    PauseContinueBroadcastReq,
                    worker._handle_pause_continue_broadcast,
                )
            ]
        )
        events = []
        receive_task = None
        outputs = asyncio.Queue()

        def start_receive_loop():
            nonlocal receive_task
            events.append("receive-loop")
            receive_task = asyncio.create_task(worker.handle_loop())

        def dispatch(request):
            events.append(("registration", request.worker_token))
            outputs.put_nowait(
                PauseContinueBroadcastReq(
                    rid=multi_tokenizer_mixin_module._encode_worker_registration(
                        worker._worker_token
                    ),
                    is_pause=False,
                    http_worker_ipc=multi_tokenizer_mixin_module._WORKER_REGISTRATION_ACCEPTED,
                )
            )

        worker.auto_create_handle_loop = start_receive_loop
        worker._dispatch_to_scheduler = dispatch
        monkeypatch.setattr(
            tokenizer_manager_module,
            "async_sock_recv",
            lambda _socket: outputs.get(),
        )
        try:
            await TokenizerWorker.wait_for_router_registration(worker, timeout_sec=0.1)
        finally:
            assert receive_task is not None
            receive_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await receive_task

        assert events[0] == "receive-loop"
        assert ("registration", worker._worker_token) in events

    asyncio.run(scenario())


def test_dead_worker_registration_is_replaced_before_next_pause(
    monkeypatch,
) -> None:
    events = []
    router = _make_router(_RecordingSocketMapping(events))
    router.all_worker_ipcs = set()
    router._worker_registrations = {}
    live_pids = {101, 102}
    monkeypatch.setattr(
        multi_tokenizer_mixin_module,
        "_tokenizer_worker_is_alive",
        lambda identity: identity.pid in live_pids,
    )
    old_a = _worker_registration(
        "worker-a",
        pid=101,
        process_start_time=1.0,
        worker_token="old-a",
    )
    worker_b = _worker_registration(
        "worker-b",
        pid=102,
        process_start_time=2.0,
        worker_token="worker-b",
    )
    router._handle_tokenizer_worker_registration(old_a)
    router._handle_tokenizer_worker_registration(worker_b)
    assert router.all_worker_ipcs == {"worker-a", "worker-b"}

    live_pids.remove(101)
    live_pids.add(103)
    replacement = _worker_registration(
        "worker-c",
        pid=103,
        process_start_time=3.0,
        worker_token="replacement-a",
    )
    router._handle_tokenizer_worker_registration(replacement)

    assert router.all_worker_ipcs == {"worker-b", "worker-c"}
    assert set(router._worker_registrations) == {"worker-b", "worker-c"}


def test_dead_remote_pause_owner_stops_service_before_replacement(
    monkeypatch,
) -> None:
    stopped = []
    events = []
    router = _make_router(_RecordingSocketMapping(events))
    router.all_worker_ipcs = {"worker-a", "worker-b"}
    origin = multi_tokenizer_mixin_module._TokenizerWorkerIdentity(
        ipc_name="worker-a",
        pid=101,
        process_start_time=1.0,
        token="origin",
    )
    peer = multi_tokenizer_mixin_module._TokenizerWorkerIdentity(
        ipc_name="worker-b",
        pid=102,
        process_start_time=2.0,
        token="peer",
    )
    router._worker_registrations = {
        origin.ipc_name: origin,
        peer.ipc_name: peer,
    }
    owner = "remote-weight-transfer:committed-owner"
    router.pause_owners = {owner}
    router.active_remote_pause_owner = owner
    router._pause_owner_workers = {owner: origin}
    live_pids = {102, 103}
    monkeypatch.setattr(
        multi_tokenizer_mixin_module,
        "_tokenizer_worker_is_alive",
        lambda identity: identity.pid in live_pids,
    )
    monkeypatch.setattr(
        multi_tokenizer_mixin_module,
        "kill_process_tree",
        lambda pid, include_parent: stopped.append((pid, include_parent)),
    )

    router._handle_tokenizer_worker_registration(
        _worker_registration(
            "worker-c",
            pid=103,
            process_start_time=3.0,
            worker_token="replacement",
        )
    )

    assert stopped == [(multi_tokenizer_mixin_module.os.getpid(), True)]
    assert router.pause_owners == {owner}
    assert router._pause_poisoned_owners == {owner}
    assert set(router._worker_registrations) == {"worker-b"}
    assert "worker-c" not in router.all_worker_ipcs
    assert (
        "fanout",
        multi_tokenizer_mixin_module._WORKER_REGISTRATION_REJECTED,
        "worker-c",
    ) in events


def test_live_extra_worker_is_rejected_without_evicting_membership(
    monkeypatch,
) -> None:
    events = []
    router = _make_router(_RecordingSocketMapping(events))
    router.all_worker_ipcs = set()
    router._worker_registrations = {}
    monkeypatch.setattr(
        multi_tokenizer_mixin_module,
        "_tokenizer_worker_is_alive",
        lambda _identity: True,
    )
    for ipc, pid in (("worker-a", 101), ("worker-b", 102), ("worker-c", 103)):
        router._handle_tokenizer_worker_registration(
            _worker_registration(
                ipc,
                pid=pid,
                process_start_time=float(pid),
                worker_token=ipc,
            )
        )

    assert router.all_worker_ipcs == {"worker-a", "worker-b"}
    assert set(router._worker_registrations) == {"worker-a", "worker-b"}
    assert (
        "fanout",
        multi_tokenizer_mixin_module._WORKER_REGISTRATION_REJECTED,
        "worker-c",
    ) in events
    assert ("socket-closed", None, "worker-c") in events


def test_replacement_registration_waits_for_inflight_transition(
    monkeypatch,
) -> None:
    events = []
    router = _make_router(_RecordingSocketMapping(events))
    router.all_worker_ipcs = {"worker-b"}
    router._worker_registrations = {
        "worker-b": multi_tokenizer_mixin_module._TokenizerWorkerIdentity(
            ipc_name="worker-b",
            pid=102,
            process_start_time=2.0,
            token="worker-b",
        )
    }
    router._pause_transitions = {"inflight": object()}
    monkeypatch.setattr(
        multi_tokenizer_mixin_module,
        "_tokenizer_worker_is_alive",
        lambda _identity: True,
    )
    replacement = _worker_registration(
        "worker-c",
        pid=103,
        process_start_time=3.0,
        worker_token="replacement",
    )

    router._handle_tokenizer_worker_registration(replacement)
    assert not any(
        event[1] == multi_tokenizer_mixin_module._WORKER_REGISTRATION_REJECTED
        for event in events
        if event[0] == "fanout"
    )
    assert "worker-c" not in router.all_worker_ipcs

    router._pause_transitions.clear()
    router._handle_tokenizer_worker_registration(replacement)
    assert router.all_worker_ipcs == {"worker-b", "worker-c"}


def test_replacement_membership_completes_pause_and_continue(monkeypatch) -> None:
    async def scenario():
        events = []

        async def send_to_scheduler(_socket, request):
            events.append(("scheduler", type(request).__name__, request.rid))

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        live_pids = {101, 102}
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_tokenizer_worker_is_alive",
            lambda identity: identity.pid in live_pids,
        )
        router = _make_router(_RecordingSocketMapping(events))
        router.all_worker_ipcs = set()
        router._worker_registrations = {}
        router.pause_owners = set()
        router.active_remote_pause_owner = None
        router._handle_tokenizer_worker_registration(
            _worker_registration(
                "worker-a",
                pid=101,
                process_start_time=1.0,
                worker_token="old-a",
            )
        )
        router._handle_tokenizer_worker_registration(
            _worker_registration(
                "worker-b",
                pid=102,
                process_start_time=2.0,
                worker_token="worker-b",
            )
        )
        live_pids.remove(101)
        live_pids.add(103)
        router._handle_tokenizer_worker_registration(
            _worker_registration(
                "worker-c",
                pid=103,
                process_start_time=3.0,
                worker_token="replacement-a",
            )
        )

        pause = PauseGenerationReqInput(
            rid="remote-weight-transfer:replacement",
            http_worker_ipc="worker-c",
            mode="retract",
        )
        await router._handle_pause_continue_request(pause)
        pause_identity = _identity(pause)
        pause_transition = router._pause_transitions[pause_identity.transition_id]
        assert pause_transition.expected_workers == {"worker-b", "worker-c"}
        for worker_ipc in ("worker-b", "worker-c"):
            _ack(router, pause_identity, worker_ipc)
        for worker_ipc in ("worker-b", "worker-c"):
            _ack(router, pause_identity, worker_ipc, applied=True)
        await asyncio.sleep(0)
        for worker_ipc in ("worker-b", "worker-c"):
            _committed_ack(router, pause_identity, worker_ipc)
        assert router.pause_owners == {pause_identity.owner}
        assert (
            router._pause_owner_workers[pause_identity.owner]
            == router._worker_registrations["worker-c"]
        )

        resume = ContinueGenerationReqInput(
            rid=pause_identity.owner,
            http_worker_ipc="worker-c",
            torch_empty_cache=False,
        )
        await router._handle_pause_continue_request(resume)
        resume_identity = _identity(resume)
        resume_transition = router._pause_transitions[resume_identity.transition_id]
        assert resume_transition.expected_workers == {"worker-b", "worker-c"}
        for worker_ipc in ("worker-b", "worker-c"):
            _ack(router, resume_identity, worker_ipc)
        for worker_ipc in ("worker-b", "worker-c"):
            _ack(router, resume_identity, worker_ipc, applied=True)
        await asyncio.sleep(0)
        for worker_ipc in ("worker-b", "worker-c"):
            _committed_ack(router, resume_identity, worker_ipc)

        assert router.pause_owners == set()
        assert pause_identity.owner not in router._pause_owner_workers
        assert router._pause_transitions == {}

    asyncio.run(scenario())


def test_graceful_unregister_is_identity_fenced(monkeypatch) -> None:
    events = []
    router = _make_router(_RecordingSocketMapping(events))
    router.all_worker_ipcs = set()
    router._worker_registrations = {}
    monkeypatch.setattr(
        multi_tokenizer_mixin_module,
        "_tokenizer_worker_is_alive",
        lambda _identity: True,
    )
    registration = _worker_registration(
        "worker-a",
        pid=101,
        process_start_time=1.0,
        worker_token="worker-a",
    )
    router._handle_tokenizer_worker_registration(registration)

    router._handle_tokenizer_worker_registration(
        _worker_registration(
            "worker-a",
            pid=101,
            process_start_time=1.0,
            worker_token="stale-token",
            unregister=True,
        )
    )
    assert router.all_worker_ipcs == {"worker-a"}

    router._handle_tokenizer_worker_registration(
        _worker_registration(
            "worker-a",
            pid=101,
            process_start_time=1.0,
            worker_token="worker-a",
            unregister=True,
        )
    )
    assert router.all_worker_ipcs == set()
    assert router._worker_registrations == {}


def test_precommit_failure_repauses_applied_worker() -> None:
    async def scenario():
        identity = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner="remote-weight-transfer:last",
            action="continue",
            expected_state=False,
        )
        worker, _ = _make_worker(identity)
        pending = asyncio.get_running_loop().create_future()
        confirmation = asyncio.get_running_loop().create_future()
        worker._pause_continue_futures = {
            identity.transition_id: (identity, pending),
        }
        worker._pause_continue_confirmation_futures = {
            identity.transition_id: (identity, confirmation),
        }

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
            ),
        )
        assert worker.is_pause is True

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=True,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED,
            ),
        )

        with pytest.raises(RuntimeError, match="failed before confirmation"):
            await pending
        assert confirmation.result() is True
        assert worker.is_pause is True
        assert worker._generation_pause_owners == {identity.owner}
        assert worker._generation_pause_resume_pending == {identity.owner}

    asyncio.run(scenario())


def test_partial_pause_failure_resumes_scheduler_and_workers(monkeypatch) -> None:
    async def scenario():
        events = []
        scheduler = SimpleNamespace(
            _engine_paused=False,
            disaggregation_mode=None,
        )

        async def send_to_scheduler(_socket, request):
            events.append(("scheduler", type(request).__name__))
            if isinstance(request, PauseGenerationReqInput):
                Scheduler.pause_generation(scheduler, request)
            else:
                Scheduler.continue_generation(scheduler, request)

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        router = _make_router(_RecordingSocketMapping(events))
        router.pause_owners = set()
        router.active_remote_pause_owner = None
        request = PauseGenerationReqInput(
            mode="in_place",
            rid="remote-weight-transfer:partial",
            http_worker_ipc="worker-a",
        )

        await router._handle_pause_continue_request(request)
        assert scheduler._engine_paused is True
        identity = _identity(request)
        _ack(router, identity, "worker-a")
        router._fail_pause_transition(
            identity.transition_id,
            "worker acknowledgement deadline expired",
        )
        for _ in range(5):
            await asyncio.sleep(0)

        assert [event[1] for event in events if event[0] == "scheduler"] == [
            "PauseGenerationReqInput",
            "ContinueGenerationReqInput",
        ]
        assert scheduler._engine_paused is False
        assert router.pause_owners == set()
        assert router._pause_poisoned_owners == set()
        failed = [
            event
            for event in events
            if event[:2]
            == ("fanout", multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED)
        ]
        assert len(failed) == 2

        worker, _ = _make_worker(identity)
        worker.is_pause = True
        worker._generation_pause_owners = {identity.owner}
        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED,
            ),
        )
        assert worker.is_pause is False
        assert worker._generation_pause_owners == set()
        assert worker._generation_pause_resume_pending == set()

    asyncio.run(scenario())


def test_partial_pause_recovery_failure_stops_the_service(monkeypatch) -> None:
    async def scenario():
        sends = 0
        stopped = []

        async def send_to_scheduler(_socket, _request):
            nonlocal sends
            sends += 1
            if sends > 1:
                raise RuntimeError("scheduler unavailable")

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "kill_process_tree",
            lambda pid, include_parent: stopped.append((pid, include_parent)),
        )
        router = _make_router(_RecordingSocketMapping([]))
        router.pause_owners = set()
        router.active_remote_pause_owner = None
        request = PauseGenerationReqInput(
            mode="in_place",
            rid="remote-weight-transfer:fail-stop",
            http_worker_ipc="worker-a",
        )

        await router._handle_pause_continue_request(request)
        identity = _identity(request)
        router._fail_pause_transition(identity.transition_id, "missing worker ack")
        for _ in range(5):
            await asyncio.sleep(0)

        assert stopped == [(multi_tokenizer_mixin_module.os.getpid(), True)]

    asyncio.run(scenario())


def test_partial_pause_rollback_fanout_failure_stops_the_service(monkeypatch) -> None:
    async def scenario():
        stopped = []

        async def send_to_scheduler(_socket, _request):
            return None

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "kill_process_tree",
            lambda pid, include_parent: stopped.append((pid, include_parent)),
        )
        mapping = _RecordingSocketMapping(
            [],
            fail_once=(
                multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED,
                "worker-a",
            ),
        )
        router = _make_router(mapping)
        router.pause_owners = set()
        router.active_remote_pause_owner = None
        request = PauseGenerationReqInput(
            mode="in_place",
            rid="remote-weight-transfer:fanout-fail-stop",
            http_worker_ipc="worker-a",
        )

        await router._handle_pause_continue_request(request)
        identity = _identity(request)
        router._fail_pause_transition(identity.transition_id, "missing worker ack")
        for _ in range(5):
            await asyncio.sleep(0)

        assert stopped == [(multi_tokenizer_mixin_module.os.getpid(), True)]

    asyncio.run(scenario())


def test_remote_owner_handoff_reconciles_stale_worker_state(monkeypatch) -> None:
    async def scenario():
        first_owner = "remote-weight-transfer:first"
        second_owner = "remote-weight-transfer:second"
        scheduled = []
        delayed_second_pause = []
        original_continue_identities = []
        dropped_second_pause_acks = 0

        async def send_to_scheduler(_socket, request):
            scheduled.append(request)

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )

        worker = object.__new__(TokenizerWorker)
        worker.is_pause = True
        worker.is_pause_cond = asyncio.Condition()
        worker._generation_pause_owners = {first_owner}
        worker._generation_pause_resume_pending = set()
        worker._pause_continue_futures = {}
        worker._pause_continue_confirmation_futures = {}
        worker._prepared_pause_transitions = {}
        worker._confirmed_pause_transitions = {}
        worker._committed_pause_transitions = {}
        worker._poisoned_pause_transitions = {}
        worker._latest_pause_transitions = {}

        class HandoffSocketMapping:
            def send_output(self, worker_ipc, output):
                identity = multi_tokenizer_mixin_module._decode_pause_transition(
                    output.rid
                )
                assert identity is not None
                state = output.http_worker_ipc
                if (
                    worker_ipc == "worker-a"
                    and identity.owner == second_owner
                    and state is None
                ):
                    delayed_second_pause.append(output)
                    return
                if worker_ipc == "worker-a":
                    asyncio.create_task(
                        TokenizerWorker._apply_pause_continue_broadcast(worker, output)
                    )
                    return
                if state is None:
                    router._handle_pause_continue_ack(
                        PauseContinueBroadcastReq(
                            rid=output.rid,
                            is_pause=output.is_pause,
                            http_worker_ipc=worker_ipc,
                        )
                    )
                elif state == multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED:
                    asyncio.get_running_loop().call_soon(
                        router._handle_pause_continue_ack,
                        PauseContinueBroadcastReq(
                            rid=multi_tokenizer_mixin_module._encode_pause_transition_applied(
                                identity
                            ),
                            is_pause=output.is_pause,
                            http_worker_ipc=worker_ipc,
                        ),
                    )
                elif state == multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED:
                    asyncio.get_running_loop().call_soon(
                        router._handle_pause_continue_ack,
                        PauseContinueBroadcastReq(
                            rid=multi_tokenizer_mixin_module._encode_pause_transition_committed_ack(
                                identity
                            ),
                            is_pause=output.is_pause,
                            http_worker_ipc=worker_ipc,
                        ),
                    )

        router = _make_router(HandoffSocketMapping())
        router.pause_owners = {first_owner}
        router.active_remote_pause_owner = first_owner
        router._pause_owner_workers = {
            first_owner: router._worker_registrations["worker-a"],
        }

        async def dispatch_from_worker(request):
            nonlocal dropped_second_pause_acks
            if isinstance(request, PauseContinueBroadcastReq):
                committed_identity = (
                    multi_tokenizer_mixin_module._decode_pause_transition_committed_ack(
                        request.rid
                    )
                )
                if committed_identity is not None:
                    request.http_worker_ipc = "worker-a"
                    router._handle_pause_continue_ack(request)
                    return
                identity = multi_tokenizer_mixin_module._decode_pause_transition(
                    request.rid
                )
                if (
                    identity is not None
                    and identity.owner == second_owner
                    and dropped_second_pause_acks == 0
                ):
                    dropped_second_pause_acks += 1
                    return
                request.http_worker_ipc = "worker-a"
                router._handle_pause_continue_ack(request)
                return
            identity = multi_tokenizer_mixin_module._decode_pause_transition(
                request.rid
            )
            assert identity is not None
            original_continue_identities.append(identity)
            await router._handle_pause_continue_request(request)

        worker._async_dispatch_to_scheduler = dispatch_from_worker

        second_pause = PauseGenerationReqInput(
            mode="in_place",
            rid=second_owner,
            http_worker_ipc="worker-b",
        )
        await router._handle_pause_continue_request(second_pause)
        await router._handle_pause_continue_request(second_pause)
        assert len(delayed_second_pause) == 1

        continue_request = ContinueGenerationReqInput(
            rid=first_owner,
            http_worker_ipc="worker-a",
            torch_empty_cache=False,
        )
        await asyncio.wait_for(
            TokenizerWorker._continue_generation_impl(worker, continue_request),
            timeout=1,
        )

        original_identity = original_continue_identities[0]
        canonical_identity = _identity(continue_request)
        assert original_identity.expected_state is False
        assert canonical_identity.expected_state is True
        assert canonical_identity.transition_id == original_identity.transition_id
        assert router.pause_owners == {second_owner}
        assert router._pause_poisoned_owners == set()
        assert scheduled == []

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            delayed_second_pause[0],
        )
        assert dropped_second_pause_acks == 1
        assert router.active_remote_pause_owner == second_owner
        assert worker.is_pause is True

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            delayed_second_pause[0],
        )
        for _ in range(5):
            await asyncio.sleep(0)

        assert worker.is_pause is True
        assert worker._generation_pause_owners == {second_owner}
        assert router.active_remote_pause_owner == second_owner
        assert router._pause_poisoned_owners == set()

    asyncio.run(scenario())


def test_canonicalized_failure_matches_original_worker_request() -> None:
    async def scenario():
        original = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner="remote-weight-transfer:first",
            action="continue",
            expected_state=False,
        )
        canonical = multi_tokenizer_mixin_module._PauseTransitionIdentity(
            transition_id=original.transition_id,
            owner=original.owner,
            action=original.action,
            expected_state=True,
            deadline_monotonic_ns=original.deadline_monotonic_ns,
        )
        worker = object.__new__(TokenizerWorker)
        worker.is_pause = True
        worker.is_pause_cond = asyncio.Condition()
        worker._generation_pause_owners = {original.owner}
        worker._generation_pause_resume_pending = set()
        pending = asyncio.get_running_loop().create_future()
        confirmation = asyncio.get_running_loop().create_future()
        worker._pause_continue_futures = {original.transition_id: (original, pending)}
        worker._pause_continue_confirmation_futures = {
            original.transition_id: (original, confirmation)
        }
        worker._prepared_pause_transitions = {}
        worker._confirmed_pause_transitions = {}
        worker._committed_pause_transitions = {}
        worker._poisoned_pause_transitions = {}
        worker._latest_pause_transitions = {original.owner: original}

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(canonical),
                is_pause=True,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED,
            ),
        )

        with pytest.raises(RuntimeError, match="failed before confirmation"):
            await asyncio.wait_for(pending, timeout=0.05)
        with pytest.raises(RuntimeError, match="failed before confirmation"):
            await asyncio.wait_for(confirmation, timeout=0.05)
        assert worker.is_pause is True
        assert worker._generation_pause_owners == {original.owner}
        assert worker._generation_pause_resume_pending == {original.owner}
        assert worker._poisoned_pause_transitions == {original.transition_id: canonical}

    asyncio.run(scenario())
