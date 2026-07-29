from __future__ import annotations

import asyncio
import contextvars
import inspect
import time
from types import SimpleNamespace

import msgspec
import pytest

from sglang.srt.configs.load_config import LoadFormat
from sglang.srt.managers.io_struct import (
    WeightSnapshotActivationReqInput,
    WeightSnapshotActivationReqOutput,
    set_weight_snapshot_activation_result,
    weight_snapshot_activation_request_context,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _activation_output(
    request: WeightSnapshotActivationReqInput,
    *,
    state: str,
) -> WeightSnapshotActivationReqOutput:
    return WeightSnapshotActivationReqOutput(
        action=request.action,
        success=True,
        message="Success.",
        phase=request.phase,
        transaction_id=request.transaction_id,
        request_id=request.request_id,
        responder_id="scheduler-0",
        state=state,
    )


def test_activation_phases_share_one_absolute_deadline() -> None:
    calls = []

    async def communicate(request, *, deadline_unix_sec):
        calls.append((request, deadline_unix_sec))
        state = {
            "prepare": "prepared",
            "commit": "serving",
        }[request.phase]
        return [_activation_output(request, state=state)]

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        weight_snapshot_activation_communicator=communicate,
    )

    success, _ = asyncio.run(
        TokenizerControlMixin.update_weight_snapshot_activation(
            manager,
            "activate",
        )
    )

    assert success is True
    assert [request.phase for request, _ in calls] == ["prepare", "commit"]
    deadlines = {deadline for _, deadline in calls}
    assert len(deadlines) == 1
    for request, deadline in calls:
        assert request.deadline_unix_sec == deadline


def test_activation_requires_every_rank_to_observe_serving() -> None:
    phases = []

    async def communicate(request, *, deadline_unix_sec):
        assert request.deadline_unix_sec == deadline_unix_sec
        phases.append(request.phase)
        states = {
            "prepare": ("prepared", "prepared"),
            "commit": ("serving", "prepared"),
            "reconcile": ("serving", "conflict"),
            "abort": ("aborted", "aborted"),
        }[request.phase]
        return [_activation_output(request, state=state) for state in states]

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        weight_snapshot_activation_communicator=communicate,
    )

    success, message = asyncio.run(
        TokenizerControlMixin.update_weight_snapshot_activation(
            manager,
            "activate",
        )
    )

    assert success is False
    assert phases == ["prepare", "commit", "reconcile", "abort"]
    assert "could not be reconciled" in message


def test_expired_activation_request_cannot_reach_pending_owner() -> None:
    events = []

    class Pending:
        def prepare(self, *_args, **_kwargs):
            events.append("prepare")

        def commit(self, *_args, **_kwargs):
            events.append("commit")

        def reconcile(self, *_args, **_kwargs):
            events.append("reconcile")

        def abort(self, *_args, **_kwargs):
            events.append("abort")

    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        pending_weight_snapshot_activation=Pending(),
    )

    def encode() -> bytes:
        return msgspec.msgpack.encode(
            WeightSnapshotActivationReqInput(
                action="activate",
                phase="prepare",
                transaction_id="activation-transaction",
                request_id="expired-request",
                deadline_unix_sec=time.time() - 1,
            )
        )

    payload = contextvars.Context().run(encode)

    def execute() -> None:
        request = msgspec.msgpack.decode(
            payload,
            type=WeightSnapshotActivationReqInput,
        )
        with weight_snapshot_activation_request_context(request):
            ModelRunner.activate_pending_weight_snapshot(runner)

    with pytest.raises(TimeoutError, match="deadline expired"):
        contextvars.Context().run(execute)

    assert events == []


def test_scheduler_batch_keeps_activation_transaction_identity() -> None:
    outputs = []

    def dispatch(request):
        set_weight_snapshot_activation_result(f"{request.phase}-state")
        return WeightSnapshotActivationReqOutput(
            action=request.action,
            success=True,
            message="Success.",
        )

    scheduler = SimpleNamespace(
        session_controller=SimpleNamespace(maybe_reap=lambda _now: None),
        _request_dispatcher=dispatch,
        ipc_channels=SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(
                send_output=lambda output, request: outputs.append((output, request))
            )
        ),
        weight_updater=SimpleNamespace(
            check_pending_remote_instance_weight_transfers=lambda: []
        ),
        flush_wrapper=SimpleNamespace(check_pending=lambda: None),
        external_corpus_manager=None,
    )
    requests = [
        WeightSnapshotActivationReqInput(
            action="activate",
            phase=phase,
            transaction_id=f"transaction-{phase}",
            request_id=f"request-{phase}",
            deadline_unix_sec=time.time() + 30,
        )
        for phase in ("prepare", "commit")
    ]

    Scheduler.process_input_requests(scheduler, requests)

    assert [
        (
            output.request_id,
            output.transaction_id,
            output.phase,
            output.state,
            request.request_id,
        )
        for output, request in outputs
    ] == [
        (
            "request-prepare",
            "transaction-prepare",
            "prepare",
            "prepare-state",
            "request-prepare",
        ),
        (
            "request-commit",
            "transaction-commit",
            "commit",
            "commit-state",
            "request-commit",
        ),
    ]


def test_begin_transfer_keeps_legacy_third_positional_argument() -> None:
    signature = inspect.signature(
        TokenizerControlMixin.begin_remote_instance_weight_transfer
    )
    bound = signature.bind(None, 300, "runtime_v1", "transfer-1")

    assert bound.arguments["transfer_id"] == "transfer-1"
    assert "manifest_revision_semantics" not in bound.arguments


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
