import asyncio
import json
from types import SimpleNamespace

from fastapi.responses import ORJSONResponse

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import MaterializeWeightsReqInput
from sglang.srt.managers.tokenizer_control_mixin import WeightMaterializationError
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
            materialize_weights=lambda _obj, request: (
                _raise_materialization_error(
                    ValueError("invalid storage options"),
                    request,
                )
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
            materialize_weights=lambda _obj, request: (
                _raise_materialization_error(
                    RuntimeError("Store backend failed"),
                    request,
                )
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
    )
    result = _with_tokenizer_manager(
        SimpleNamespace(
            materialize_weights=lambda _obj, request: (
                _raise_materialization_error(error, request)
            )
        ),
        lambda: http_server.materialize_weights(obj, None),
    )

    assert isinstance(result, ORJSONResponse)
    assert result.status_code == 503


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
        )
    finally:
        loop.close()

    assert result == {"materialization_id": "materialize-1"}
    obj, request = calls[0]
    assert isinstance(obj, MaterializeWeightsReqInput)
    assert obj.storage_options == {"backend": "mooncake"}
    assert obj.materialization_id == "materialize-1"
    assert obj.source_external_dp_rank == 1
    assert request is None
