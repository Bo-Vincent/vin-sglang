import asyncio
import sys
from types import SimpleNamespace

import pytest
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints import http_server
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _with_tokenizer_manager(manager, operation):
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    try:
        return asyncio.run(operation())
    finally:
        http_server._global_state = prior_state


def test_http_lists_remote_weight_transfer_sessions() -> None:
    session = {
        "transfer_id": "transfer-1",
        "lease_id": "lease-0",
        "lease_ids": ["lease-0"],
        "generation": 7,
        "deadline_unix_sec": 130.0,
        "expired": True,
        "session_state": "expired",
    }

    async def list_sessions():
        return [session]

    result = _with_tokenizer_manager(
        SimpleNamespace(
            list_remote_instance_weight_transfer_sessions=list_sessions,
        ),
        http_server.list_remote_instance_weight_transfer_sessions,
    )

    assert result == {"sessions": [session]}


def test_http_gets_remote_weight_transfer_session_or_returns_not_found() -> None:
    session = {
        "transfer_id": "transfer-1",
        "lease_id": "lease-0",
        "lease_ids": ["lease-0"],
        "generation": 7,
        "deadline_unix_sec": 130.0,
        "expired": True,
        "session_state": "expired",
    }

    async def get_session(transfer_id, **_identity):
        return session if transfer_id == "transfer-1" else None

    manager = SimpleNamespace(
        get_remote_instance_weight_transfer_session=get_session,
    )
    found = _with_tokenizer_manager(
        manager,
        lambda: http_server.get_remote_instance_weight_transfer_session("transfer-1"),
    )
    missing = _with_tokenizer_manager(
        manager,
        lambda: http_server.get_remote_instance_weight_transfer_session("missing"),
    )

    assert found == session
    assert isinstance(missing, ORJSONResponse)
    assert missing.status_code == 404


def test_http_forwards_fenced_identity_for_status_renew_and_release() -> None:
    calls = []
    session = {
        "transfer_id": "transfer-1",
        "lease_fence": "fence-1",
        "generation": 7,
        "deadline_unix_sec": 130.0,
        "expired": False,
        "session_state": "active",
    }

    async def get_session(transfer_id, *, lease_fence, generation):
        calls.append(("status", transfer_id, lease_fence, generation))
        return session

    async def renew(transfer_id, lease_timeout_sec, *, lease_fence, generation):
        calls.append(
            (
                "renew",
                transfer_id,
                lease_fence,
                generation,
                lease_timeout_sec,
            )
        )
        return True, "Success."

    async def release(transfer_id, *, lease_fence, generation):
        calls.append(("release", transfer_id, lease_fence, generation))
        return True, "Success."

    manager = SimpleNamespace(
        get_remote_instance_weight_transfer_session=get_session,
        renew_remote_instance_weight_transfer=renew,
        release_remote_instance_weight_transfer=release,
    )
    status = _with_tokenizer_manager(
        manager,
        lambda: http_server.get_remote_instance_weight_transfer_session(
            "transfer-1",
            lease_fence="fence-1",
            generation=7,
        ),
    )
    renewed = _with_tokenizer_manager(
        manager,
        lambda: http_server.renew_remote_instance_weight_transfer(
            "transfer-1",
            lease_timeout_sec=60,
            lease_fence="fence-1",
            generation=7,
        ),
    )
    released = _with_tokenizer_manager(
        manager,
        lambda: http_server.release_remote_instance_weight_transfer(
            "transfer-1",
            lease_fence="fence-1",
            generation=7,
        ),
    )

    assert status == session
    assert renewed["deadline_unix_sec"] == 130.0
    assert released == {"success": True, "message": "Success."}
    assert calls == [
        ("status", "transfer-1", "fence-1", 7),
        ("renew", "transfer-1", "fence-1", 7, 60),
        ("status", "transfer-1", "fence-1", 7),
        ("release", "transfer-1", "fence-1", 7),
    ]


def test_http_status_rejects_invalid_fenced_identity() -> None:
    async def get_session(_transfer_id, **_identity):
        raise ValueError("remote weight transfer lease fence is required")

    result = _with_tokenizer_manager(
        SimpleNamespace(
            get_remote_instance_weight_transfer_session=get_session,
        ),
        lambda: http_server.get_remote_instance_weight_transfer_session("transfer-1"),
    )

    assert isinstance(result, ORJSONResponse)
    assert result.status_code == 400


def test_renew_propagates_fanout_completion_unknown_for_reconciliation() -> None:
    async def renew(*_args, **_kwargs):
        error = TimeoutError("fan-out response deadline expired: received 1/2")
        error.completion_unknown = True
        error.dispatch_started = True
        error.dispatch_completed = True
        error.received_count = 1
        error.expected_count = 2
        error.partial_results = ("rank-0",)
        raise error

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        renew_remote_instance_weight_transfer_communicator=renew,
    )

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(
            TokenizerControlMixin.renew_remote_instance_weight_transfer(
                manager,
                "transfer-1",
                lease_timeout_sec=60,
                lease_fence="fence-1",
                generation=7,
            )
        )

    assert raised.value.transfer_id == "transfer-1"
    assert raised.value.session_state == "completion_unknown"
    assert raised.value.lease_fence == "fence-1"
    assert raised.value.generation == 7
    assert raised.value.completion_unknown is True
    assert raised.value.cleanup_pending is False
    assert raised.value.retryable is False
    assert raised.value.reconcile_required is True
    session = manager._remote_weight_transfer_session_index["transfer-1"]
    assert session["session_state"] == "completion_unknown"
    assert session["lease_fence"] == "fence-1"
    assert session["generation"] == 7
    assert session["completion_unknown"] is True
    assert session["reconcile_required"] is True


if __name__ == "__main__":
    pytest_args = ["-x" if argument == "-f" else argument for argument in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *pytest_args]))
