from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.model_runner_components import (
    remote_instance_weight_transporter as transporter_module,
)
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
)


class _ServerArgs:
    dist_init_addr = None
    engine_info_bootstrap_port = 12345
    remote_instance_weight_loader_start_seed_via_transfer_engine = True
    remote_instance_weight_loader_backend = (
        RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE
    )

    @staticmethod
    def remote_instance_weight_loader_use_transfer_engine():
        return True


def _transporter(**overrides):
    values = dict(
        server_args=_ServerArgs(),
        get_model=lambda: object(),
        tp_rank=0,
        gpu_id=0,
        dp_rank=0,
        pp_rank=0,
        ep_rank=0,
    )
    values.update(overrides)
    return RemoteInstanceWeightTransporter(**values)


def test_transporter_registers_physical_memory_without_logical_payload(
    monkeypatch,
) -> None:
    transporter = _transporter()
    transporter.engine = object()
    transporter.session_id = "127.0.0.1:5000"
    registered = {"runtime.weight": (0x1000, 16, 2)}
    monkeypatch.setattr(
        transporter_module,
        "register_memory_region",
        lambda model, engine: registered,
    )
    published = []
    monkeypatch.setattr(
        RemoteInstanceWeightTransporter,
        "_register_to_engine_info_bootstrap",
        lambda self: published.append(True),
    )

    transporter.maybe_register_and_publish_weight_info()

    assert transporter.weight_info == registered
    assert published == [True]


def test_engine_info_bootstrap_publishes_only_legacy_physical_index(
    monkeypatch,
) -> None:
    transporter = _transporter()
    transporter.session_id = "127.0.0.1:5000"
    transporter.weight_info = {"runtime.weight": (0x1000, 16, 2)}
    calls = []

    class _Response:
        status_code = 200
        text = ""

    monkeypatch.setattr(
        "requests.put",
        lambda url, **kwargs: calls.append((url, kwargs)) or _Response(),
    )

    transporter._register_to_engine_info_bootstrap()

    payload = calls[0][1]["json"]
    assert payload == {
        "tp_rank": 0,
        "transfer_engine_info": {
            "session_id": "127.0.0.1:5000",
            "weights_info_dict": {"runtime.weight": (0x1000, 16, 2)},
        },
    }
    assert "placement" not in repr(payload)
    assert "binding" not in repr(payload)


def test_runtime_binding_addresses_must_stay_inside_registered_storage() -> None:
    transporter = _transporter()
    transporter.weight_info = {"runtime.weight": (0x1000, 16, 2)}
    valid = SimpleNamespace(fragments=(SimpleNamespace(address=0x1008, nbytes=16),))
    invalid = SimpleNamespace(fragments=(SimpleNamespace(address=0x1018, nbytes=16),))

    transporter.validate_runtime_binding_inventory_addresses(valid)
    with pytest.raises(RuntimeError, match="outside registered"):
        transporter.validate_runtime_binding_inventory_addresses(invalid)


def test_runtime_binding_validation_requires_registration() -> None:
    transporter = _transporter()

    with pytest.raises(RuntimeError, match="not registered"):
        transporter.validate_runtime_binding_inventory_addresses(
            SimpleNamespace(fragments=())
        )


def test_transfer_engine_initialization_failure_is_fatal(monkeypatch) -> None:
    class _Engine:
        def initialize(self, *args):
            return -1

    fake_module = SimpleNamespace(TransferEngine=_Engine)
    monkeypatch.setitem(__import__("sys").modules, "mooncake.engine", fake_module)
    monkeypatch.setattr(transporter_module, "get_local_ip_auto", lambda: "127.0.0.1")
    transporter = _transporter()

    with pytest.raises(RuntimeError, match="initialization failed"):
        transporter.init_engine()

    assert transporter.engine is None


def test_worker_id_is_ephemeral_and_never_used_as_participant_identity() -> None:
    transporter = _transporter(tp_rank=2, ep_rank=1)
    transporter.session_id = "ephemeral-session"

    assert transporter.worker_id == "ephemeral-session/dp0-pp0-ep1-tp2"
