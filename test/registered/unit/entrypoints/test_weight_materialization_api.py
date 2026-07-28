import asyncio
import json
import multiprocessing
import sys
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints import engine as engine_module
from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import (
    MaterializeWeightsReqInput,
    UpdateWeightVersionReqInput,
    WeightMaterializationSessionState,
)
from sglang.srt.managers.tokenizer_control_mixin import WeightMaterializationError
from sglang.srt.managers.tokenizer_manager import ServerStatus, TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _with_tokenizer_manager(manager, operation):
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    try:
        return asyncio.run(operation())
    finally:
        http_server._global_state = prior_state


def test_http_materialize_weights_returns_control_plane_result() -> None:
    requests = []
    expected = {
        "materialization_id": "materialize-1",
        "ref": {"snapshot_id": "snapshot-1"},
        "selected_external_dp_rank": 0,
        "total_bytes": 4096,
    }

    async def materialize(obj, request):
        requests.append((obj, request))
        return expected

    obj = MaterializeWeightsReqInput(
        storage_options={"backend": "mooncake"},
        materialization_id="materialize-1",
    )
    result = _with_tokenizer_manager(
        SimpleNamespace(materialize_weights=materialize),
        lambda: http_server.materialize_weights(obj, None),
    )

    assert result == expected
    assert requests == [(obj, None)]


def test_http_materialize_weights_maps_conflict_to_409_with_id_and_state() -> None:
    async def materialize(_obj, _request):
        raise WeightMaterializationError(
            "materialization already active",
            materialization_id="materialize-1",
            session_state="conflict",
        )

    obj = MaterializeWeightsReqInput(
        storage_options={"backend": "mooncake"},
        materialization_id="materialize-1",
    )
    result = _with_tokenizer_manager(
        SimpleNamespace(materialize_weights=materialize),
        lambda: http_server.materialize_weights(obj, None),
    )

    assert isinstance(result, ORJSONResponse)
    assert result.status_code == 409
    assert json.loads(result.body) == {
        "materialization_id": "materialize-1",
        "session_state": "conflict",
        "message": "materialization already active",
    }


async def _raise_materialization_error(error, _request):
    raise error


def test_http_materialize_weights_maps_validation_error_to_400() -> None:
    obj = MaterializeWeightsReqInput(
        storage_options={"backend": "mooncake"},
        materialization_id="materialize-1",
    )
    result = _with_tokenizer_manager(
        SimpleNamespace(
            materialize_weights=lambda _obj, request: _raise_materialization_error(
                ValueError("invalid storage options"),
                request,
            )
        ),
        lambda: http_server.materialize_weights(obj, None),
    )

    assert isinstance(result, ORJSONResponse)
    assert result.status_code == 400


def test_http_materialize_weights_maps_runtime_failure_to_500() -> None:
    obj = MaterializeWeightsReqInput(
        storage_options={"backend": "mooncake"},
        materialization_id="materialize-1",
    )
    result = _with_tokenizer_manager(
        SimpleNamespace(
            materialize_weights=lambda _obj, request: _raise_materialization_error(
                RuntimeError("Store backend failed"),
                request,
            )
        ),
        lambda: http_server.materialize_weights(obj, None),
    )

    assert isinstance(result, ORJSONResponse)
    assert result.status_code == 500


def test_http_materialize_weights_maps_completion_unknown_to_503() -> None:
    obj = MaterializeWeightsReqInput(
        storage_options={"backend": "mooncake"},
        materialization_id="materialize-1",
    )
    error = WeightMaterializationError(
        "Store completion remains unknown",
        materialization_id="materialize-1",
        session_state="completion_unknown",
        completion_ticket="ticket-1",
    )
    result = _with_tokenizer_manager(
        SimpleNamespace(
            materialize_weights=lambda _obj, request: _raise_materialization_error(
                error, request
            )
        ),
        lambda: http_server.materialize_weights(obj, None),
    )

    assert isinstance(result, ORJSONResponse)
    assert result.status_code == 503
    assert json.loads(result.body)["completion_ticket"] == "ticket-1"


def test_http_materialize_weights_maps_every_session_state() -> None:
    expected = {
        WeightMaterializationSessionState.CONFLICT: 409,
        WeightMaterializationSessionState.NOT_FOUND: 404,
        WeightMaterializationSessionState.DISABLED: 503,
        WeightMaterializationSessionState.CLEANUP_PENDING: 503,
        WeightMaterializationSessionState.COMPLETION_UNKNOWN: 503,
        WeightMaterializationSessionState.FINALIZE_PENDING: 503,
        WeightMaterializationSessionState.PUBLISHED_CLEANUP_PENDING: 503,
    }
    obj = MaterializeWeightsReqInput(
        storage_options={"backend": "mooncake"},
        materialization_id="materialize-1",
    )

    for state in WeightMaterializationSessionState:
        error = WeightMaterializationError(
            state.value,
            materialization_id="materialize-1",
            session_state=state.value,
        )
        result = _with_tokenizer_manager(
            SimpleNamespace(
                materialize_weights=lambda _obj, request, error=error: (
                    _raise_materialization_error(error, request)
                )
            ),
            lambda: http_server.materialize_weights(obj, None),
        )
        assert result.status_code == expected.get(state, 500)


def test_engine_materialize_weights_is_a_sync_thin_wrapper() -> None:
    calls = []

    async def materialize(obj, request):
        calls.append((obj, request))
        return {"materialization_id": obj.materialization_id}

    loop = asyncio.new_event_loop()
    engine = SimpleNamespace(
        loop=loop,
        tokenizer_manager=SimpleNamespace(materialize_weights=materialize),
    )
    try:
        result = Engine.materialize_weights(
            engine,
            {"backend": "mooncake"},
            materialization_id="materialize-1",
            source_external_dp_rank=1,
            lease_timeout_sec=60,
        )
    finally:
        loop.close()

    assert result == {"materialization_id": "materialize-1"}
    obj, request = calls[0]
    assert isinstance(obj, MaterializeWeightsReqInput)
    assert obj.storage_options == {"backend": "mooncake"}
    assert obj.materialization_id == "materialize-1"
    assert obj.source_external_dp_rank == 1
    assert obj.lease_timeout_sec == 60
    assert request is None


def test_update_weight_version_cannot_relabel_runtime_artifact() -> None:
    manager = SimpleNamespace(
        server_args=SimpleNamespace(enable_weight_runtime_manifest=True)
    )
    result = _with_tokenizer_manager(
        manager,
        lambda: http_server.update_weight_version(
            UpdateWeightVersionReqInput(new_version="artifact-b"),
            None,
        ),
    )

    assert result.status_code == 409
    assert json.loads(result.body)["success"] is False


def test_http_begin_weight_transfer_preserves_legacy_value_error_response() -> None:
    async def begin(*_args, **_kwargs):
        raise ValueError("lease timeout is invalid")

    result = _with_tokenizer_manager(
        SimpleNamespace(begin_remote_instance_weight_transfer=begin),
        lambda: http_server.begin_remote_instance_weight_transfer(
            lease_timeout_sec=0,
            transfer_id="transfer-1",
        ),
    )

    assert result.status_code == 409
    assert json.loads(result.body) == {
        "transfer_id": None,
        "session_state": "failed",
        "message": "lease timeout is invalid",
    }


def test_http_begin_weight_transfer_preserves_legacy_third_positional_argument() -> (
    None
):
    calls = []

    async def begin(lease_timeout_sec, **kwargs):
        calls.append((lease_timeout_sec, kwargs))
        return {"transfer_id": kwargs["transfer_id"]}

    result = _with_tokenizer_manager(
        SimpleNamespace(begin_remote_instance_weight_transfer=begin),
        lambda: http_server.begin_remote_instance_weight_transfer(
            90,
            "runtime_v1",
            "transfer-legacy",
        ),
    )

    assert result == {"transfer_id": "transfer-legacy"}
    assert calls == [
        (
            90,
            {
                "manifest_format": "runtime_v1",
                "manifest_revision_semantics": "hf_revision_v1",
                "transfer_id": "transfer-legacy",
            },
        )
    ]


def test_http_release_weight_transfer_preserves_legacy_detail_response() -> None:
    async def release(_transfer_id):
        return False, "source snapshot is still in use"

    with pytest.raises(HTTPException) as error:
        _with_tokenizer_manager(
            SimpleNamespace(release_remote_instance_weight_transfer=release),
            lambda: http_server.release_remote_instance_weight_transfer("transfer-1"),
        )

    assert error.value.status_code == 409
    assert error.value.detail == "source snapshot is still in use"


def test_http_release_weight_transfer_returns_fenced_structured_conflict() -> None:
    async def release(_transfer_id, **_identity):
        return False, "source snapshot is still in use"

    result = _with_tokenizer_manager(
        SimpleNamespace(release_remote_instance_weight_transfer=release),
        lambda: http_server.release_remote_instance_weight_transfer(
            "transfer-1",
            lease_fence="fence-1",
            generation=7,
        ),
    )

    assert result.status_code == 409
    assert json.loads(result.body) == {
        "transfer_id": "transfer-1",
        "session_state": "conflict",
        "message": "source snapshot is still in use",
        "lease_fence": "fence-1",
        "generation": 7,
    }


def test_http_renew_weight_transfer_preserves_legacy_detail_response() -> None:
    async def renew(*_args, **_kwargs):
        raise ValueError("lease timeout is invalid")

    with pytest.raises(HTTPException) as error:
        _with_tokenizer_manager(
            SimpleNamespace(renew_remote_instance_weight_transfer=renew),
            lambda: http_server.renew_remote_instance_weight_transfer(
                "transfer-1",
                lease_timeout_sec=0,
            ),
        )

    assert error.value.status_code == 409
    assert error.value.detail == "lease timeout is invalid"


@pytest.mark.parametrize(
    ("operation", "session_state", "cleanup_pending"),
    [
        ("release", "cleanup_pending", True),
        ("renew", "completion_unknown", False),
    ],
)
def test_http_fenced_control_completion_unknown_requires_reconciliation(
    operation,
    session_state,
    cleanup_pending,
) -> None:
    async def fail(*_args, **_kwargs):
        error = RuntimeError("fan-out response deadline expired: received 1/2")
        error.transfer_id = "transfer-1"
        error.session_state = session_state
        error.lease_fence = "fence-1"
        error.generation = 7
        error.completion_unknown = True
        error.cleanup_pending = cleanup_pending
        error.retryable = False
        error.reconcile_required = True
        raise error

    manager = SimpleNamespace(
        release_remote_instance_weight_transfer=fail,
        renew_remote_instance_weight_transfer=fail,
    )

    async def call():
        if operation == "release":
            return await http_server.release_remote_instance_weight_transfer(
                "transfer-1",
                lease_fence="fence-1",
                generation=7,
            )
        return await http_server.renew_remote_instance_weight_transfer(
            "transfer-1",
            lease_timeout_sec=60,
            lease_fence="fence-1",
            generation=7,
        )

    result = _with_tokenizer_manager(manager, call)

    assert result.status_code == 503
    assert json.loads(result.body) == {
        "transfer_id": "transfer-1",
        "session_state": session_state,
        "message": "fan-out response deadline expired: received 1/2",
        "lease_fence": "fence-1",
        "generation": 7,
        "completion_unknown": True,
        "cleanup_pending": cleanup_pending,
        "retryable": False,
        "reconcile_required": True,
    }


def _warmup_server_args():
    return SimpleNamespace(
        checkpoint_engine_wait_weights_before_ready=False,
        is_ep_scale_joiner=False,
        ep_join_mode=None,
        skip_server_warmup=False,
        delete_ckpt_after_loading=False,
        debug_tensor_dump_input_file=None,
    )


def _startup_barrier_process(path, succeed, delay_sec, queue):
    barrier = http_server._MultiWorkerStartupBarrier.attach(path)
    try:
        time.sleep(delay_sec)
        warmed = barrier.arrive_warmup(succeed, timeout_sec=30)
        queue.put(("warmup", succeed, warmed))
        if warmed:
            if barrier.claim_activation():
                barrier.complete_activation()
            activated = barrier.wait_for_activation(timeout_sec=30)
            queue.put(("activated", succeed, activated))
            queue.put(("armed", succeed, None))
            queue.put(("open", succeed, barrier.arm(timeout_sec=30)))
    finally:
        barrier.close()


def _startup_activation_process(
    path,
    worker_id,
    activation_started,
    activation_release,
    arm_delay_sec,
    queue,
):
    barrier = http_server._MultiWorkerStartupBarrier.attach(path)
    manager = SimpleNamespace(server_status=ServerStatus.Starting)
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    original_arm = barrier.arm

    def delayed_arm(*, timeout_sec):
        time.sleep(arm_delay_sec)
        queue.put(("arming", worker_id, barrier.admission_open()))
        return original_arm(timeout_sec=timeout_sec)

    def activate(action):
        queue.put((action, worker_id, barrier.admission_open()))
        if action == "activate":
            activation_started.set()
            if not activation_release.wait(timeout=30):
                raise TimeoutError("activation release timed out")

    barrier.arm = delayed_arm
    try:
        http_server._wait_and_warmup(
            _warmup_server_args(),
            execute_warmup_func=lambda _args: True,
            weight_snapshot_activation=activate,
            startup_barrier=barrier,
        )
        queue.put(
            (
                "done",
                worker_id,
                manager.server_status,
                barrier.admission_open(),
            )
        )
    finally:
        http_server.set_global_state(prior_state)
        barrier.close()


def test_multi_worker_startup_activates_once_before_every_worker_is_armed() -> None:
    barrier = http_server._MultiWorkerStartupBarrier.create(2)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    activation_started = context.Event()
    activation_release = context.Event()
    workers = [
        context.Process(
            target=_startup_activation_process,
            args=(
                barrier.path,
                worker_id,
                activation_started,
                activation_release,
                arm_delay,
                queue,
            ),
        )
        for worker_id, arm_delay in enumerate((0.0, 0.5))
    ]
    try:
        for worker in workers:
            worker.start()
        assert activation_started.wait(timeout=30)
        assert barrier.admission_open() is False
        activation_release.set()

        events = []
        while not any(event[0] == "arming" for event in events):
            events.append(queue.get(timeout=30))
        assert barrier.admission_open() is False
        events.extend(queue.get(timeout=30) for _ in range(6 - len(events)))

        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
    finally:
        barrier.unlink()

    assert [event[0] for event in events].count("activate") == 1
    assert [event[0] for event in events].count("close") == 1
    assert [event[0] for event in events].count("arming") == 2
    done = [event for event in events if event[0] == "done"]
    assert len(done) == 2
    assert all(event[2] is ServerStatus.Up for event in done)
    assert all(event[3] is True for event in done)
    assert all(event[2] is False for event in events if event[0] != "done")


def test_multi_worker_startup_without_snapshot_completes_activation_phase() -> None:
    barrier = http_server._MultiWorkerStartupBarrier.create(1)
    manager = SimpleNamespace(server_status=ServerStatus.Starting)
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    try:
        http_server._wait_and_warmup(
            _warmup_server_args(),
            execute_warmup_func=lambda _args: True,
            startup_barrier=barrier,
        )
    finally:
        http_server.set_global_state(prior_state)

    assert manager.server_status is ServerStatus.Up
    assert barrier.admission_open() is True
    barrier.unlink()


def test_open_startup_barrier_accepts_replacement_worker() -> None:
    barrier = http_server._MultiWorkerStartupBarrier.create(1)
    replacement = None
    try:
        assert barrier.arrive_warmup(True, timeout_sec=1)
        assert barrier.claim_activation() is True
        assert barrier.complete_activation() is True
        assert barrier.wait_for_activation(timeout_sec=1) is True
        assert barrier.arm(timeout_sec=1) is True

        replacement = http_server._MultiWorkerStartupBarrier.attach(barrier.path)
        assert replacement.arrive_warmup(True, timeout_sec=1) is True
        assert replacement.claim_activation() is False
        assert replacement.wait_for_activation(timeout_sec=1) is True
        assert replacement.arm(timeout_sec=1) is True
        assert replacement.admission_open() is True
        assert replacement._read() == (1, 1, 1, 0, replacement._OPEN)
    finally:
        if replacement is not None:
            replacement.close()
        barrier.unlink()


def test_open_startup_barrier_accepts_replacement_process() -> None:
    barrier = http_server._MultiWorkerStartupBarrier.create(1)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    activation_started = context.Event()
    activation_release = context.Event()
    worker = context.Process(
        target=_startup_activation_process,
        args=(
            barrier.path,
            1,
            activation_started,
            activation_release,
            0.0,
            queue,
        ),
    )
    try:
        assert barrier.arrive_warmup(True, timeout_sec=1)
        assert barrier.claim_activation() is True
        assert barrier.complete_activation() is True
        assert barrier.wait_for_activation(timeout_sec=1) is True
        assert barrier.arm(timeout_sec=1) is True

        worker.start()
        worker.join(timeout=30)
        assert worker.exitcode == 0
        events = [queue.get(timeout=1) for _ in range(2)]
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=30)
        barrier.unlink()

    assert events == [
        ("arming", 1, True),
        ("done", 1, ServerStatus.Up, True),
    ]


def test_open_startup_barrier_ignores_replacement_failure() -> None:
    barrier = http_server._MultiWorkerStartupBarrier.create(1)
    try:
        assert barrier.arrive_warmup(True, timeout_sec=1)
        assert barrier.claim_activation() is True
        assert barrier.complete_activation() is True
        assert barrier.wait_for_activation(timeout_sec=1) is True
        assert barrier.arm(timeout_sec=1) is True

        barrier.fail()

        assert barrier.admission_open() is True
        assert barrier._read() == (1, 1, 1, 0, barrier._OPEN)
    finally:
        barrier.unlink()


def test_multi_worker_lifespan_registers_before_serving_and_unregisters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class _Manager:
        worker_id = 7
        server_status = ServerStatus.Starting
        serving_chat_class = staticmethod(lambda *_args, **_kwargs: SimpleNamespace())

        def _enable_startup_admission_gate(self):
            events.append("gate")

        async def wait_for_router_registration(self, *, timeout_sec):
            assert timeout_sec == http_server._MULTI_WORKER_REGISTRATION_TIMEOUT_SEC
            events.append("registered")

        def unregister_from_router(self):
            events.append("unregistered")

    class _Barrier:
        def fail(self):
            events.append("failed")

        def close(self):
            events.append("barrier-closed")

    manager = _Manager()
    barrier = _Barrier()
    args = SimpleNamespace(
        _multi_worker_startup_barrier_path="barrier",
        disaggregation_mode="null",
        enable_metrics=False,
        enable_trace=False,
        grpc_mode=False,
        grpc_port=None,
        load_format="auto",
        sidecar=None,
        smg_grpc_mode=False,
        tool_server=None,
        warmups=None,
    )
    fast_api_app = SimpleNamespace(
        is_single_tokenizer_mode=False,
        state=SimpleNamespace(),
    )
    prior_state = http_server.get_global_state()

    async def init_multi_tokenizer():
        http_server.set_global_state(
            SimpleNamespace(
                tokenizer_manager=manager,
                template_manager=SimpleNamespace(),
                scheduler_info={},
            )
        )
        return args

    for name in (
        "AnthropicServing",
        "OllamaServing",
        "OpenAIServingClassify",
        "OpenAIServingCompletion",
        "OpenAIServingDetokenize",
        "OpenAIServingEmbedding",
        "OpenAIServingRerank",
        "OpenAIServingScore",
        "OpenAIServingTokenize",
        "OpenAIServingTranscription",
    ):
        monkeypatch.setattr(
            http_server, name, lambda *_args, **_kwargs: SimpleNamespace()
        )
    monkeypatch.setattr(http_server, "init_multi_tokenizer", init_multi_tokenizer)
    monkeypatch.setattr(
        http_server._MultiWorkerStartupBarrier,
        "attach",
        classmethod(lambda _cls, _path: barrier),
    )
    monkeypatch.setattr(
        http_server,
        "_wait_and_warmup",
        lambda **_kwargs: events.append("warmup"),
    )

    async def run_lifespan():
        async with http_server.lifespan(fast_api_app):
            events.append("serving")

    try:
        asyncio.run(run_lifespan())
    finally:
        http_server._global_state = prior_state
        http_server._multi_worker_startup_barrier = None

    assert events.index("registered") < events.index("serving")
    assert events[-2:] == ["unregistered", "barrier-closed"]


def test_multi_worker_startup_barrier_propagates_failure() -> None:
    barrier = http_server._MultiWorkerStartupBarrier.create(2)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    workers = [
        context.Process(
            target=_startup_barrier_process,
            args=(barrier.path, succeed, delay, queue),
        )
        for succeed, delay in ((True, 0.0), (False, 0.2))
    ]
    try:
        for worker in workers:
            worker.start()
        events = [queue.get(timeout=30) for _ in range(2)]
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0
    finally:
        barrier.unlink()

    assert sorted(events) == [
        ("warmup", False, False),
        ("warmup", True, False),
    ]


def test_multi_tokenizer_startup_warmup_token_is_stable() -> None:
    server_args = SimpleNamespace()

    token = http_server._ensure_startup_warmup_token(server_args)

    assert token
    assert server_args._startup_warmup_token == token
    assert http_server._ensure_startup_warmup_token(server_args) == token


def test_multi_tokenizer_setup_writes_shared_startup_warmup_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_server_args = []
    shared_memory = SimpleNamespace(unlink=lambda: None)
    tokenizer_manager = SimpleNamespace(
        socket_mapping=SimpleNamespace(clear_all_sockets=lambda: None)
    )
    server_args = SimpleNamespace(
        enable_http2=False,
        enable_metrics=False,
        enable_ssl_refresh=False,
        fastapi_root_path="",
        host="127.0.0.1",
        log_level="info",
        log_level_http=None,
        port=30000,
        ssl_ca_certs=None,
        ssl_certfile=None,
        ssl_keyfile=None,
        ssl_keyfile_password=None,
        tokenizer_worker_num=2,
    )

    def capture_shared_args(_port_args, written_server_args, _scheduler_info):
        assert written_server_args._startup_warmup_token
        captured_server_args.append(written_server_args)
        return shared_memory

    monkeypatch.setattr(
        http_server, "write_data_for_multi_tokenizer", capture_shared_args
    )
    monkeypatch.setattr(http_server, "set_uvicorn_logging_configs", lambda _args: None)
    monkeypatch.setattr(http_server.uvicorn, "run", lambda *_args, **_kwargs: None)

    prior_state = http_server.get_global_state()
    try:
        http_server._setup_and_run_http_server(
            server_args,
            tokenizer_manager,
            SimpleNamespace(),
            SimpleNamespace(),
            [{}],
            None,
        )
    finally:
        http_server._global_state = prior_state

    assert captured_server_args == [server_args]
    worker = TokenizerManager.__new__(TokenizerManager)
    worker.server_args = captured_server_args[0]
    worker.init_running_status()
    assert worker._startup_warmup_token == server_args._startup_warmup_token


class _TrackingStatusManager:
    def __init__(self) -> None:
        self.transitions = []
        self._server_status = ServerStatus.Starting

    @property
    def server_status(self):
        return self._server_status

    @server_status.setter
    def server_status(self, status):
        self.transitions.append(status)
        self._server_status = status

    def _startup_warmup_request_headers(self):
        return {"x-sglang-startup-warmup": "startup-warmup-token"}


def _execute_warmup_server_args(*, disaggregation_mode="null"):
    return SimpleNamespace(
        api_key=None,
        debug_tensor_dump_input_file=None,
        disaggregation_mode=disaggregation_mode,
        dp_size=1,
        language_only=True,
        skip_tokenizer_init=True,
        ssl_verify=lambda: False,
        url=lambda: "http://127.0.0.1:30000",
    )


def test_execute_warmup_does_not_publish_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        status_code=200,
        text="ok",
        json=lambda: {"is_generation": True},
    )
    request_headers = []
    manager = _TrackingStatusManager()
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    monkeypatch.setattr(http_server.time, "sleep", lambda _seconds: None)

    def capture_request_headers(*_args, **kwargs):
        request_headers.append(kwargs["headers"])
        return response

    monkeypatch.setattr(http_server.requests, "get", capture_request_headers)
    monkeypatch.setattr(http_server.requests, "post", capture_request_headers)
    try:
        success = http_server._execute_server_warmup(_execute_warmup_server_args())
    finally:
        http_server._global_state = prior_state

    assert success is True
    assert request_headers == [
        {"x-sglang-startup-warmup": "startup-warmup-token"},
        {"x-sglang-startup-warmup": "startup-warmup-token"},
    ]
    assert manager.transitions == []
    assert manager.server_status is ServerStatus.Starting


def test_pd_warmup_failure_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = SimpleNamespace(
        status_code=200,
        text="ok",
        json=lambda: {"is_generation": True},
    )
    manager = _TrackingStatusManager()
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    monkeypatch.setattr(http_server.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(http_server.requests, "get", lambda *_args, **_kwargs: response)

    async def fail_pd_warmup(**_kwargs):
        return [500]

    monkeypatch.setattr(
        http_server,
        "_send_disaggregation_warmup_requests",
        fail_pd_warmup,
    )
    try:
        success = http_server._execute_server_warmup(
            _execute_warmup_server_args(disaggregation_mode="decode")
        )
    finally:
        http_server._global_state = prior_state

    assert success is False
    assert manager.server_status is ServerStatus.UnHealthy


def test_weight_snapshot_warmup_failure_closes_pending_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    manager = SimpleNamespace(server_status=ServerStatus.Starting)
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    monkeypatch.setattr(http_server, "kill_process_tree", lambda _pid: None)
    try:
        http_server._wait_and_warmup(
            _warmup_server_args(),
            execute_warmup_func=lambda _args: False,
            weight_snapshot_activation=lambda action: events.append(action),
        )
    finally:
        http_server._global_state = prior_state

    assert events == ["close"]
    assert manager.server_status is ServerStatus.UnHealthy


def test_weight_snapshot_weights_ready_timeout_closes_pending_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    killed = []
    manager = SimpleNamespace(
        initial_weights_loaded=False,
        server_status=ServerStatus.Starting,
    )
    args = _warmup_server_args()
    args.checkpoint_engine_wait_weights_before_ready = True
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    monkeypatch.setattr(http_server, "WAIT_WEIGHTS_READY_TIMEOUT", 1)
    monkeypatch.setattr(http_server.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        http_server,
        "kill_process_tree",
        lambda pid: killed.append(pid),
    )
    try:
        http_server._wait_and_warmup(
            args,
            execute_warmup_func=lambda _args: True,
            weight_snapshot_activation=lambda action: events.append(action),
        )
    finally:
        http_server._global_state = prior_state

    assert events == ["close"]
    assert manager.server_status is ServerStatus.UnHealthy
    assert len(killed) == 1


def test_weight_snapshot_custom_warmup_exception_closes_pending_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class _Manager:
        server_status = ServerStatus.Starting
        serving_chat_class = staticmethod(lambda *_args, **_kwargs: SimpleNamespace())

        def _enable_startup_admission_gate(self):
            pass

        @contextmanager
        def _startup_warmup_admission_bypass(self):
            yield

        async def update_weight_snapshot_activation(self, action):
            events.append(action)
            return True, "Success."

    manager = _Manager()
    args = SimpleNamespace(
        disaggregation_mode="null",
        enable_metrics=False,
        enable_trace=False,
        grpc_mode=False,
        grpc_port=None,
        load_format="weight_snapshot",
        sidecar=None,
        smg_grpc_mode=False,
        tool_server=None,
        warmups="custom",
    )
    fast_api_app = SimpleNamespace(
        is_single_tokenizer_mode=True,
        server_args=args,
        state=SimpleNamespace(),
        warmup_thread_kwargs={"server_args": args},
    )
    prior_state = http_server.get_global_state()
    http_server.set_global_state(
        SimpleNamespace(
            tokenizer_manager=manager,
            template_manager=SimpleNamespace(),
            scheduler_info={},
        )
    )
    for name in (
        "AnthropicServing",
        "OllamaServing",
        "OpenAIServingClassify",
        "OpenAIServingCompletion",
        "OpenAIServingDetokenize",
        "OpenAIServingEmbedding",
        "OpenAIServingRerank",
        "OpenAIServingScore",
        "OpenAIServingTokenize",
        "OpenAIServingTranscription",
    ):
        monkeypatch.setattr(
            http_server, name, lambda *_args, **_kwargs: SimpleNamespace()
        )

    async def fail_custom_warmup(*_args, **_kwargs):
        raise RuntimeError("custom warmup failed")

    async def run_lifespan():
        async with http_server.lifespan(fast_api_app):
            raise AssertionError("lifespan must not reach serving")

    monkeypatch.setattr(http_server, "execute_warmups", fail_custom_warmup)
    try:
        with pytest.raises(RuntimeError, match="custom warmup failed"):
            asyncio.run(run_lifespan())
    finally:
        http_server._global_state = prior_state

    assert events == ["close"]
    assert manager.server_status is ServerStatus.UnHealthy


def test_health_generate_rejects_unhealthy_without_promoting_status() -> None:
    manager = SimpleNamespace(
        gracefully_exit=False,
        server_status=ServerStatus.UnHealthy,
    )
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    try:
        response = asyncio.run(
            http_server.health_generate(
                SimpleNamespace(url=SimpleNamespace(path="/health_generate"))
            )
        )
    finally:
        http_server._global_state = prior_state

    assert response.status_code == 503
    assert manager.server_status is ServerStatus.UnHealthy


def test_weight_snapshot_skip_warmup_stays_unready_until_activation() -> None:
    events = []
    manager = _TrackingStatusManager()
    args = _warmup_server_args()
    args.skip_server_warmup = True
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    try:
        http_server._wait_and_warmup(
            args,
            launch_callback=lambda: events.append(("launch", manager.server_status)),
            weight_snapshot_activation=lambda action: events.append(action),
        )
    finally:
        http_server._global_state = prior_state

    assert events == [
        ("launch", ServerStatus.Starting),
        "activate",
        "close",
    ]
    assert manager.transitions == [ServerStatus.Up]


def test_weight_snapshot_activation_finishes_after_warmup() -> None:
    events = []
    manager = SimpleNamespace(server_status=ServerStatus.Starting)
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    try:
        http_server._wait_and_warmup(
            _warmup_server_args(),
            launch_callback=lambda: events.append(("launch", manager.server_status)),
            execute_warmup_func=lambda _args: events.append("warmup") or True,
            weight_snapshot_activation=lambda action: events.append(action),
        )
    finally:
        http_server._global_state = prior_state

    assert events == [
        "warmup",
        ("launch", ServerStatus.Starting),
        "activate",
        "close",
    ]
    assert manager.server_status is ServerStatus.Up


def test_launch_callback_failure_precedes_snapshot_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    killed = []
    manager = SimpleNamespace(server_status=ServerStatus.Starting)
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    monkeypatch.setattr(
        http_server,
        "kill_process_tree",
        lambda pid: killed.append(pid),
    )

    def fail_launch():
        events.append(("launch", manager.server_status))
        raise RuntimeError("launch callback failed")

    try:
        http_server._wait_and_warmup(
            _warmup_server_args(),
            launch_callback=fail_launch,
            execute_warmup_func=lambda _args: True,
            weight_snapshot_activation=lambda action: events.append(action),
        )
    finally:
        http_server._global_state = prior_state

    assert events == [
        ("launch", ServerStatus.Starting),
        "close",
    ]
    assert manager.server_status is ServerStatus.UnHealthy
    assert len(killed) == 1


def test_weight_snapshot_activation_failure_closes_after_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    killed = []
    manager = SimpleNamespace(server_status=ServerStatus.Starting)
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    monkeypatch.setattr(
        http_server,
        "kill_process_tree",
        lambda pid: killed.append(pid),
    )

    def fail_activation(action):
        events.append(action)
        if action == "activate":
            raise RuntimeError("activation failed")

    try:
        http_server._wait_and_warmup(
            _warmup_server_args(),
            launch_callback=lambda: events.append("launch"),
            execute_warmup_func=lambda _args: True,
            weight_snapshot_activation=fail_activation,
        )
    finally:
        http_server._global_state = prior_state

    assert events == ["launch", "activate", "close"]
    assert manager.server_status is ServerStatus.UnHealthy
    assert len(killed) == 1


def test_weight_snapshot_close_failure_happens_after_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    killed = []
    manager = SimpleNamespace(server_status=ServerStatus.Starting)
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    monkeypatch.setattr(
        http_server,
        "kill_process_tree",
        lambda pid: killed.append(pid),
    )

    def fail_close(action):
        events.append(action)
        if action == "close":
            raise RuntimeError("close failed")

    try:
        http_server._wait_and_warmup(
            _warmup_server_args(),
            launch_callback=lambda: events.append("launch"),
            execute_warmup_func=lambda _args: True,
            weight_snapshot_activation=fail_close,
        )
    finally:
        http_server._global_state = prior_state

    assert events == ["launch", "activate", "close"]
    assert manager.server_status is ServerStatus.UnHealthy
    assert len(killed) == 1


def test_weight_snapshot_unknown_activation_is_not_closed_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    killed = []
    manager = SimpleNamespace(server_status=ServerStatus.Starting)
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=manager))
    monkeypatch.setattr(
        http_server,
        "kill_process_tree",
        lambda pid: killed.append(pid),
    )

    def activation(action):
        events.append(action)
        if action == "activate":
            raise http_server._WeightSnapshotActivationCompletionUnknown("timed out")

    try:
        http_server._wait_and_warmup(
            _warmup_server_args(),
            launch_callback=lambda: events.append("launch"),
            execute_warmup_func=lambda _args: True,
            weight_snapshot_activation=activation,
        )
    finally:
        http_server._global_state = prior_state

    assert events == ["launch", "activate"]
    assert manager.server_status is ServerStatus.UnHealthy
    assert len(killed) == 1


def test_engine_rejects_weight_snapshot_before_subprocess_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched = []

    def fail_launch(*_args, **_kwargs):
        launched.append(True)
        raise AssertionError("subprocess launch must not run")

    monkeypatch.setattr(engine_module, "load_plugins", lambda: None)
    monkeypatch.setattr(engine_module.atexit, "register", lambda *_args: None)
    monkeypatch.setattr(Engine, "_launch_subprocesses", fail_launch)

    with pytest.raises(ValueError, match="HTTP server"):
        Engine(server_args=SimpleNamespace(load_format="weight_snapshot"))

    assert launched == []


if __name__ == "__main__":
    pytest_args = ["-x" if argument == "-f" else argument for argument in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *pytest_args]))
