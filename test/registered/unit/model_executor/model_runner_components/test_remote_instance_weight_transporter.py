import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from sglang.srt.entrypoints.engine_info_bootstrap_server import (
    EngineInfoBootstrapServer,
)
from sglang.srt.model_executor.model_runner_components import (
    remote_instance_weight_transporter as transporter_module,
)
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.srt.model_loader import remote_instance_weight_loader_utils as loader_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeEngine:
    pass


class _FakeServerArgs:
    remote_instance_weight_loader_backend = "transfer_engine"
    remote_instance_weight_loader_start_seed_via_transfer_engine = True
    enable_weight_runtime_manifest = True
    dist_init_addr = None
    engine_info_bootstrap_port = 6789
    model_path = "Qwen/Qwen3.5-0.8B"
    revision = "main"

    @staticmethod
    def remote_instance_weight_loader_use_transfer_engine() -> bool:
        return True


def _runtime_manifest(worker_id: str) -> dict:
    return {
        "format_version": 1,
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "instance_id": "sglang:source-session",
        "generation": 1,
        "lease_id": "lease-1",
        "tensors": [
            {
                "fragment_id": "fragment-1",
                "tensor_id": "embed_tokens.weight",
                "runtime_name": "model.embed_tokens.weight",
                "aliases": ["model.embed_tokens.weight"],
                "global_shape": [8, 4],
                "global_offset": [0, 0],
                "local_shape": [4, 4],
                "dtype": "bfloat16",
                "itemsize": 2,
                "partition_dim": 0,
                "layer_id": None,
                "expert_id": None,
                "layout_fingerprint": "qwen3.5:vocab",
                "address": 0x10000,
                "nbytes": 32,
                "byte_offset": 0,
                "stride": [4, 1],
                "storage_offset": 0,
                "device": "cuda",
                "is_contiguous": True,
                "worker_id": worker_id,
                "endpoint": "source-session",
                "rank": {"dp": 0, "tp": 0, "pp": 0, "ep": 0},
                "lease_generation": 1,
            }
        ],
    }


def test_transporter_publishes_legacy_weight_info_without_holding_snapshot(
    monkeypatch,
) -> None:
    payloads = []
    monkeypatch.setattr(
        transporter_module,
        "register_memory_region",
        lambda model, engine: {"model.embed_tokens.weight": (0x10000, 16, 2)},
    )
    monkeypatch.setattr(
        "requests.put",
        lambda url, json, timeout: (
            payloads.append((url, json, timeout))
            or SimpleNamespace(status_code=200, text="OK")
        ),
    )
    transporter = RemoteInstanceWeightTransporter(
        server_args=_FakeServerArgs(),
        get_model=lambda: object(),
        tp_rank=0,
        gpu_id=0,
        dp_rank=0,
        pp_rank=0,
        ep_rank=0,
    )
    transporter.engine = _FakeEngine()
    transporter.session_id = "source-session"

    transporter.maybe_register_and_publish_weight_info()

    assert transporter.weight_info == {"model.embed_tokens.weight": (0x10000, 16, 2)}
    _, payload, timeout = payloads[0]
    assert timeout == 5
    assert payload["tp_rank"] == 0
    assert payload["transfer_engine_info"]["weights_info_dict"] == (
        transporter.weight_info
    )
    assert "weight_runtime_manifest" not in payload["transfer_engine_info"]


def test_transporter_registration_failure_prevents_source_from_becoming_ready(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        transporter_module,
        "register_memory_region",
        lambda model, engine: {"weight": (0x10000, 16, 2)},
    )
    monkeypatch.setattr(
        "requests.put",
        lambda url, json, timeout: SimpleNamespace(
            status_code=503, text="bootstrap unavailable"
        ),
    )
    transporter = RemoteInstanceWeightTransporter(
        server_args=_FakeServerArgs(),
        get_model=lambda: object(),
        tp_rank=0,
        gpu_id=0,
    )
    transporter.engine = _FakeEngine()
    transporter.session_id = "source-session"

    with pytest.raises(RuntimeError, match="bootstrap unavailable"):
        transporter.maybe_register_and_publish_weight_info()


def test_transfer_engine_initialization_failure_is_fatal(monkeypatch) -> None:
    class FailingTransferEngine:
        def initialize(self, *args):
            return -7

        def get_rpc_port(self):
            raise AssertionError(
                "RPC port must not be read after failed initialization"
            )

    engine_module = ModuleType("mooncake.engine")
    engine_module.TransferEngine = FailingTransferEngine
    mooncake_module = ModuleType("mooncake")
    mooncake_module.__path__ = []
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_module)
    monkeypatch.setitem(sys.modules, "mooncake.engine", engine_module)
    monkeypatch.setattr(transporter_module, "get_local_ip_auto", lambda: "127.0.0.1")
    transporter = RemoteInstanceWeightTransporter(
        server_args=_FakeServerArgs(),
        get_model=lambda: object(),
        tp_rank=0,
        gpu_id=0,
    )

    with pytest.raises(RuntimeError, match="initialization failed.*-7"):
        transporter.init_engine()

    assert transporter.session_id == ""


def test_weight_transfer_provider_rejects_a_different_engine() -> None:
    registered_engine = _FakeEngine()
    transporter = RemoteInstanceWeightTransporter(
        server_args=_FakeServerArgs(),
        get_model=lambda: object(),
        tp_rank=0,
        gpu_id=0,
        engine=registered_engine,
    )

    with pytest.raises(RuntimeError, match="different transport instance"):
        transporter.create_weight_transfer_provider(_FakeEngine())


def test_weight_transfer_provider_constructs_mooncake_adapter() -> None:
    from sglang.srt.weight_transfer.mooncake import MooncakeWeightTransferProvider

    engine = _FakeEngine()
    transporter = RemoteInstanceWeightTransporter(
        server_args=_FakeServerArgs(),
        get_model=lambda: object(),
        tp_rank=0,
        gpu_id=0,
        engine=engine,
    )

    provider = transporter.create_weight_transfer_provider(
        engine,
        max_batch_operations=17,
    )

    assert isinstance(provider, MooncakeWeightTransferProvider)
    assert provider.transfer_engine is engine
    assert provider.max_batch_operations == 17


def test_engine_info_bootstrap_keeps_legacy_index(monkeypatch) -> None:
    class FakeUvicornServer:
        should_exit = False

        def __init__(self, config):
            self.config = config

        def run(self):
            return None

    monkeypatch.setattr(
        "sglang.srt.entrypoints.engine_info_bootstrap_server.uvicorn.Server",
        FakeUvicornServer,
    )
    server = EngineInfoBootstrapServer("127.0.0.1", 6789)
    client = TestClient(server.app)
    response = client.put(
        "/register_transfer_engine_info",
        json={
            "tp_rank": 0,
            "transfer_engine_info": {
                "session_id": "source-session",
                "weights_info_dict": {"weight": [0x10000, 16, 2]},
            },
        },
    )

    assert response.status_code == 200
    assert client.get("/get_transfer_engine_info", params={"rank": 0}).json()[
        "remote_instance_transfer_engine_info"
    ] == ["source-session", {"weight": [0x10000, 16, 2]}]


def test_remote_transfer_session_uses_begin_and_release_endpoints(monkeypatch) -> None:
    manifest = _runtime_manifest("source-session/dp0-pp0-ep0-tp0")
    calls = []

    def fake_post(url, params, timeout):
        calls.append(("POST", url, params, timeout))
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [manifest],
            },
        )

    def fake_delete(url, timeout):
        calls.append(("DELETE", url, timeout))
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(loader_utils.requests, "post", fake_post)
    monkeypatch.setattr(loader_utils.requests, "delete", fake_delete)
    monkeypatch.setattr(
        loader_utils, "supports_mooncake_placement_binding_v1", lambda: True
    )

    result = loader_utils.begin_remote_instance_weight_transfer(
        "http://127.0.0.1:30000",
        transfer_id="transfer-1",
    )

    assert result.transfer_id == "transfer-1"
    assert result.manifests == [manifest]
    assert loader_utils.release_remote_instance_weight_transfer(
        "http://127.0.0.1:30000", "transfer-1"
    )
    assert calls == [
        (
            "POST",
            "http://127.0.0.1:30000/remote_instance_weight_transfer",
            {
                "lease_timeout_sec": 300,
                "manifest_format": "placement_binding_v1",
                "transfer_id": "transfer-1",
            },
            30,
        ),
        (
            "DELETE",
            "http://127.0.0.1:30000/remote_instance_weight_transfer/transfer-1",
            30,
        ),
    ]


def test_runtime_manifest_must_stay_inside_registered_memory() -> None:
    transporter = RemoteInstanceWeightTransporter(
        server_args=_FakeServerArgs(),
        get_model=lambda: object(),
        tp_rank=0,
        gpu_id=0,
    )
    transporter.weight_info = {"weight": (0x10000, 16, 2)}
    transporter.validate_runtime_manifest_addresses(
        SimpleNamespace(tensors=[SimpleNamespace(address=0x10008, nbytes=16)])
    )

    with pytest.raises(RuntimeError, match="outside registered"):
        transporter.validate_runtime_manifest_addresses(
            SimpleNamespace(tensors=[SimpleNamespace(address=0x10018, nbytes=16)])
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
