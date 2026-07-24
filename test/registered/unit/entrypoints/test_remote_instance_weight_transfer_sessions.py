import asyncio
from types import SimpleNamespace

from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints import http_server
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

    async def get_session(transfer_id):
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
