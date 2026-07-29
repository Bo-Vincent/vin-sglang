import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest
from fastapi import HTTPException

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.grpc_bridge import RuntimeHandle
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _AsyncContext:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


def _request_manager(status: ServerStatus) -> TokenizerManager:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = SimpleNamespace(language_only=False)
    manager.server_status = status
    manager._startup_admission_enabled = True
    manager._startup_warmup_token = "startup-warmup-token"
    manager.elastic_worker_count = 1
    manager.rid_to_state = {}
    manager.auto_create_handle_loop = Mock()
    manager._set_default_priority = Mock()
    manager.request_logger = Mock()
    manager.tokenizer = None
    manager.is_pause = False
    manager.is_pause_cond = asyncio.Condition()
    manager.model_update_lock = SimpleNamespace(reader_lock=_AsyncContext())
    manager.weight_update_fail_closed = False
    manager._validate_and_resolve_lora = AsyncMock()
    manager._tokenize_one_request = AsyncMock(
        return_value=SimpleNamespace(input_ids=[1, 2, 3])
    )
    manager._send_one_request = Mock()
    manager._init_req_state = lambda obj, _: manager.rid_to_state.update(
        {obj.rid: SimpleNamespace(prompt_token_ids=None)}
    )

    async def wait_one_response(*_args):
        yield {"text": "warm"}

    manager._wait_one_response = wait_one_response
    return manager


def _request(rid: str) -> MagicMock:
    request = MagicMock(spec=GenerateReqInput)
    request.routed_dp_rank = None
    request.is_single = True
    request.rid = rid
    request.return_prompt_token_ids = False
    request.stream = False
    request.normalize_batch_and_arguments = Mock()
    return request


def test_multi_tokenizer_workers_reuse_shared_startup_warmup_token() -> None:
    server_args = SimpleNamespace(_startup_warmup_token="shared-startup-token")
    first = TokenizerManager.__new__(TokenizerManager)
    first.server_args = server_args
    first.init_running_status()
    second = TokenizerManager.__new__(TokenizerManager)
    second.server_args = server_args
    second.init_running_status()

    assert first._startup_warmup_token == "shared-startup-token"
    assert second._startup_warmup_token == "shared-startup-token"


def test_http_generate_route_uses_shared_startup_gate() -> None:
    manager = _request_manager(ServerStatus.Starting)
    obj = _request("http-startup-request")
    raw_request = SimpleNamespace(headers={})
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    try:
        with pytest.raises(HTTPException, match="server is starting") as raised:
            asyncio.run(http_server.generate_request(obj, raw_request))
    finally:
        http_server._global_state = prior_state

    assert raised.value.status_code == 503
    manager._tokenize_one_request.assert_not_awaited()


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/generate",
            {
                "text": "hello",
                "sampling_params": {"max_new_tokens": 1},
                "stream": False,
            },
        ),
        (
            "/generate",
            {
                "text": "hello",
                "sampling_params": {"max_new_tokens": 1},
                "stream": True,
            },
        ),
        (
            "/v1/completions",
            {
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 1,
                "stream": False,
            },
        ),
        (
            "/v1/completions",
            {
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 1,
                "stream": True,
            },
        ),
        (
            "/v1/chat/completions",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
                "stream": False,
            },
        ),
        (
            "/v1/chat/completions",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
                "stream": True,
            },
        ),
        (
            "/v1/responses",
            {
                "model": "test-model",
                "input": "hello",
                "stream": False,
            },
        ),
        (
            "/v1/responses",
            {
                "model": "test-model",
                "input": "hello",
                "stream": True,
            },
        ),
        (
            "/v1/embeddings",
            {
                "model": "test-model",
                "input": "hello",
            },
        ),
        (
            "/v1/classify",
            {
                "model": "test-model",
                "input": "hello",
            },
        ),
        (
            "/v1/score",
            {
                "model": "test-model",
                "query": "hello",
                "items": ["world"],
            },
        ),
        (
            "/v1/rerank",
            {
                "query": "hello",
                "documents": ["world"],
            },
        ),
    ],
)
def test_http_inference_returns_503_before_activation(path, payload) -> None:
    manager = _request_manager(ServerStatus.Starting)

    class UnexpectedServingCall:
        async def handle_request(self, *_args, **_kwargs):
            return httpx.Response(status_code=200)

        async def create_responses(self, *_args, **_kwargs):
            return httpx.Response(status_code=200)

    async def request_once():
        transport = httpx.ASGITransport(
            app=http_server.app,
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.post(path, json=payload)

    prior_state = http_server.get_global_state()
    serving_names = (
        "openai_serving_completion",
        "openai_serving_chat",
        "openai_serving_responses",
        "openai_serving_embedding",
        "openai_serving_classify",
        "openai_serving_score",
        "openai_serving_rerank",
    )
    prior_serving = {
        name: getattr(http_server.app.state, name, None) for name in serving_names
    }
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    for name in serving_names:
        setattr(http_server.app.state, name, UnexpectedServingCall())
    try:
        response = asyncio.run(request_once())
    finally:
        http_server._global_state = prior_state
        for name, prior in prior_serving.items():
            if prior is None:
                delattr(http_server.app.state, name)
            else:
                setattr(http_server.app.state, name, prior)

    assert response.status_code == 503
    assert not response.headers["content-type"].startswith("text/event-stream")
    manager._tokenize_one_request.assert_not_awaited()


def test_native_grpc_runtime_handle_uses_shared_startup_gate() -> None:
    manager = _request_manager(ServerStatus.Starting)
    obj = _request("grpc-startup-request")
    errors = []
    handle = RuntimeHandle.__new__(RuntimeHandle)
    handle.tokenizer_manager = manager
    handle._send_native_error = lambda _callback, message: errors.append(message)

    asyncio.run(handle._run_generate(obj, Mock(), False, None))

    assert errors == ["server is starting; inference is not admitted"]
    manager._tokenize_one_request.assert_not_awaited()


def test_startup_gate_rejects_external_inference_before_normalization() -> None:
    manager = _request_manager(ServerStatus.Starting)
    obj = _request("external-startup-request")
    raw_request = SimpleNamespace(
        headers={"x-sglang-startup-warmup": "not-the-startup-token"}
    )

    async def drive() -> None:
        await manager.generate_request(obj, raw_request).__anext__()

    with pytest.raises(RuntimeError, match="server is starting"):
        asyncio.run(drive())

    obj.normalize_batch_and_arguments.assert_not_called()
    manager._tokenize_one_request.assert_not_awaited()
    assert obj.rid not in manager.rid_to_state


def test_custom_warmup_context_narrowly_bypasses_startup_gate() -> None:
    manager = _request_manager(ServerStatus.Starting)
    obj = _request("custom-warmup")

    async def drive():
        with manager._startup_warmup_admission_bypass():
            return await manager.generate_request(obj).__anext__()

    assert asyncio.run(drive()) == {"text": "warm"}
    manager._tokenize_one_request.assert_awaited_once()
    manager._send_one_request.assert_called_once()


def test_http_warmup_token_narrowly_bypasses_startup_gate() -> None:
    manager = _request_manager(ServerStatus.Starting)
    obj = _request("http-warmup")
    raw_request = SimpleNamespace(
        headers={"x-sglang-startup-warmup": "startup-warmup-token"}
    )

    async def drive():
        return await manager.generate_request(obj, raw_request).__anext__()

    assert asyncio.run(drive()) == {"text": "warm"}
    manager._tokenize_one_request.assert_awaited_once()
    manager._send_one_request.assert_called_once()


def test_matching_warmup_token_cannot_bypass_unhealthy_status() -> None:
    manager = _request_manager(ServerStatus.UnHealthy)
    obj = _request("unhealthy-warmup")
    raw_request = SimpleNamespace(
        headers={"x-sglang-startup-warmup": "startup-warmup-token"}
    )

    async def drive() -> None:
        await manager.generate_request(obj, raw_request).__anext__()

    with pytest.raises(RuntimeError, match="server is unhealthy"):
        asyncio.run(drive())

    manager._tokenize_one_request.assert_not_awaited()


def test_warmup_bypass_still_waits_for_pause_gate() -> None:
    manager = _request_manager(ServerStatus.Starting)
    manager.is_pause = True
    obj = _request("paused-warmup")

    class _ControlledPauseCondition:
        def __init__(self):
            self.wait_started = asyncio.Event()
            self.release = asyncio.Event()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def wait_for(self, _predicate):
            self.wait_started.set()
            await self.release.wait()

    async def drive():
        pause_condition = _ControlledPauseCondition()
        manager.is_pause_cond = pause_condition
        with manager._startup_warmup_admission_bypass():
            task = asyncio.create_task(manager.generate_request(obj).__anext__())
            await pause_condition.wait_started.wait()
            manager._tokenize_one_request.assert_not_awaited()
            manager.is_pause = False
            pause_condition.release.set()
            return await asyncio.wait_for(task, timeout=1)

    assert asyncio.run(drive()) == {"text": "warm"}
    manager._tokenize_one_request.assert_awaited_once()


def test_warmup_bypass_does_not_bypass_model_update_gate() -> None:
    manager = _request_manager(ServerStatus.Starting)
    manager.weight_update_fail_closed = True
    obj = _request("warmup-during-failed-update")

    async def drive() -> None:
        with manager._startup_warmup_admission_bypass():
            await manager.generate_request(obj).__anext__()

    with pytest.raises(RuntimeError, match="partial weight update"):
        asyncio.run(drive())

    manager._tokenize_one_request.assert_not_awaited()
    assert obj.rid not in manager.rid_to_state


def test_process_local_engine_does_not_enable_http_startup_gate() -> None:
    manager = _request_manager(ServerStatus.Starting)
    manager.init_running_status()
    assert manager._startup_admission_enabled is False
    obj = _request("process-local-engine")

    async def drive():
        return await manager.generate_request(obj).__anext__()

    assert asyncio.run(drive()) == {"text": "warm"}
    manager._tokenize_one_request.assert_awaited_once()


def test_partial_weight_update_blocks_inference_before_tokenization() -> None:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = SimpleNamespace(language_only=False)
    manager.server_status = ServerStatus.Up
    manager._startup_admission_enabled = True
    manager._startup_warmup_token = "startup-warmup-token"
    manager.elastic_worker_count = 1
    manager.rid_to_state = {}
    manager.auto_create_handle_loop = Mock()
    manager._set_default_priority = Mock()
    manager.request_logger = Mock()
    manager.tokenizer = None
    manager.is_pause = False
    manager.is_pause_cond = asyncio.Condition()
    manager.model_update_lock = SimpleNamespace(reader_lock=_AsyncContext())
    manager.weight_update_fail_closed = True
    manager._validate_and_resolve_lora = AsyncMock()
    manager._tokenize_one_request = AsyncMock()
    manager._init_req_state = lambda obj, _: manager.rid_to_state.update(
        {obj.rid: object()}
    )

    request = MagicMock(spec=GenerateReqInput)
    request.routed_dp_rank = None
    request.is_single = True
    request.rid = "partial-weight-update"
    request.normalize_batch_and_arguments = Mock()

    async def drive() -> None:
        await manager.generate_request(request).__anext__()

    with pytest.raises(RuntimeError, match="partial weight update"):
        asyncio.run(drive())

    manager._tokenize_one_request.assert_not_awaited()
    assert request.rid not in manager.rid_to_state


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
