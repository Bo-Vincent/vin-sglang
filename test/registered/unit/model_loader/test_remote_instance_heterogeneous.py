import contextlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.model_loader import loader as loader_module
from sglang.srt.model_loader.loader import RemoteInstanceModelLoader
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _CompletionUnknownError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _runtime_server_args(monkeypatch):
    monkeypatch.setattr(
        loader_module,
        "get_server_args",
        lambda: SimpleNamespace(torchao_config=None),
    )


@pytest.mark.parametrize("release_success", [True, False])
def test_heterogeneous_loader_builds_local_plan_and_reads_from_source(
    monkeypatch,
    release_success,
) -> None:
    calls = {}
    source_inventory = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "lease_id": "source-runtime-lease",
        "fragments": [SimpleNamespace(fragment_id="source-fragment")],
    }
    target_inventory = {
        "model_id": source_inventory["model_id"],
        "revision": source_inventory["revision"],
        "fragments": [SimpleNamespace(fragment_id="target-fragment")],
    }

    class FakeRuntimeManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            return SimpleNamespace(**inventory)

    class FakeRegistrationLease:
        @classmethod
        def from_fragment(cls, fragment, *, runtime_lease_id=None):
            suffix = f":{runtime_lease_id}" if runtime_lease_id else ""
            return f"lease:{fragment.fragment_id}{suffix}"

    class FakeReader:
        def __init__(self, engine, **kwargs):
            calls["engine"] = engine
            calls["reader_options"] = kwargs

        def execute(self, plan, sources, target, **kwargs):
            calls["execute"] = (plan, sources, target, kwargs)
            return [SimpleNamespace(nbytes=64, operation_count=2, request_count=1)]

    transfer_session = SimpleNamespace(
        transfer_id="transfer-1",
        manifests=[source_inventory],
        lease_timeout_sec=90,
    )

    class FakeCoordinator:
        def __init__(self, seed_url, world_group):
            calls["coordinator"] = (seed_url, world_group)

        def acquire(self):
            calls["acquired"] = calls.get("acquired", 0) + 1
            return transfer_session

        def raise_if_failed(self):
            raise AssertionError("loader must use the fixed readiness gate")

        def ready_for_transfer(self, local_ready):
            calls["ready"] = local_ready
            return local_ready

        def finish(self, *, local_success, local_release_safe=True):
            calls["finish"] = (local_success, local_release_safe)
            return local_success, release_success

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")
    fake_weight_transfer.MemoryRegistrationLease = FakeRegistrationLease
    fake_weight_transfer.MooncakeTransferEngineReader = FakeReader
    fake_weight_transfer.RuntimeManifest = FakeRuntimeManifest
    fake_weight_transfer.TransferEngineError = _CompletionUnknownError

    def plan_runtime_transfer_to_local_target(sources, target):
        calls["plan"] = (sources, target)
        return SimpleNamespace(operations=("compact-operation",))

    fake_weight_transfer.plan_runtime_transfer_to_local_target = (
        plan_runtime_transfer_to_local_target
    )
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", fake_weight_transfer)
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        FakeCoordinator,
    )
    monkeypatch.setattr(
        loader_module,
        "get_world_group",
        lambda: "target-world",
    )
    monkeypatch.setattr(
        loader_module.current_platform,
        "synchronize",
        lambda: calls.setdefault("synchronized", True),
    )
    monkeypatch.setattr(
        loader_module,
        "_post_load_weights",
        lambda model: calls.setdefault("post_loaded", model),
    )

    @contextlib.contextmanager
    def target_builder(**kwargs):
        calls["builder"] = kwargs
        yield target_inventory

    model = object()
    engine = object()
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    success = loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
        model,
        engine,
        "http://seed:30000",
        "target-session",
        target_builder,
    )

    assert success is True
    assert calls["builder"] == {
        "model": model,
        "model_id": source_inventory["model_id"],
        "revision": source_inventory["revision"],
        "instance_id": "sglang:target-session",
        "endpoint": "target-session",
    }
    assert calls["plan"][1].revision == source_inventory["revision"]
    _, _, _, execute_kwargs = calls["execute"]
    assert execute_kwargs == {
        "source_pre_registered": True,
        "source_registrations": ("lease:source-fragment:source-runtime-lease",),
        "target_pre_registered": True,
        "target_registrations": ("lease:target-fragment",),
    }
    assert calls["reader_options"] == {"max_batch_operations": 8192}
    assert calls["synchronized"] is True
    assert calls["post_loaded"] is model
    assert calls["coordinator"] == ("http://seed:30000", "target-world")
    assert calls["acquired"] == 1
    assert calls["ready"] is True
    assert calls["finish"] == (True, True)


def test_post_load_weights_refreshes_gemma_runtime_buffer() -> None:
    norm = GemmaRMSNorm(4)
    norm.weight.data.copy_(torch.tensor([0.5, -0.25, 1.0, 2.0]))
    assert torch.equal(norm.gemma_weight, torch.ones(4))

    loader_module._post_load_weights(norm)

    assert torch.equal(norm.gemma_weight, norm.weight.data + 1.0)


def test_transfer_engine_without_manifest_builder_uses_legacy_loader(
    monkeypatch,
) -> None:
    calls = []
    model = torch.nn.Module()
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    loader.load_config = SimpleNamespace(
        load_format=loader_module.LoadFormat.REMOTE_INSTANCE,
        remote_instance_weight_loader_backend=(
            loader_module.RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE
        ),
        remote_instance_weight_loader_transfer_engine="engine",
        remote_instance_weight_loader_seed_instance_ip="127.0.0.1",
        remote_instance_weight_loader_seed_instance_service_port=30000,
        remote_instance_weight_runtime_manifest_builder=None,
        tp_rank=3,
    )
    monkeypatch.setattr(loader_module, "_get_quantization_config", lambda *args: None)
    monkeypatch.setattr(loader_module, "_initialize_model", lambda *args: model)
    monkeypatch.setattr(loader_module, "register_memory_region", lambda *args: ())
    monkeypatch.setattr(
        loader,
        "load_model_from_remote_instance_by_transfer_engine",
        lambda model, engine, seed_url, tp_rank: (
            calls.append((model, engine, seed_url, tp_rank)) or True
        ),
    )
    monkeypatch.setattr(
        loader,
        "load_model_from_remote_instance_by_transfer_engine_heterogeneous",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy dispatch must not enter manifest loading")
        ),
    )

    loaded = loader.load_model(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )

    assert loaded is model
    assert calls == [
        (model, "engine", "http://127.0.0.1:30000", 3),
    ]


def test_heterogeneous_loader_fails_closed_without_source_manifests(
    monkeypatch,
) -> None:
    class EmptyCoordinator:
        def __init__(self, seed_url, world_group):
            pass

        def acquire(self):
            return None

    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        EmptyCoordinator,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: object())
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    assert (
        loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
            object(), object(), "http://seed:30000", "target-session", object()
        )
        is False
    )


def test_heterogeneous_loader_releases_source_snapshot_after_transfer_failure(
    monkeypatch,
) -> None:
    outcomes = []
    source_inventory = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "lease_id": "source-runtime-lease",
        "fragments": [],
    }

    class FakeCoordinator:
        def __init__(self, seed_url, world_group):
            pass

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1", manifests=[source_inventory]
            )

        def raise_if_failed(self):
            raise AssertionError("loader must use the fixed readiness gate")

        def ready_for_transfer(self, local_ready):
            outcomes.append(("ready", local_ready))
            return local_ready

        def finish(self, *, local_success, local_release_safe=True):
            outcomes.append((local_success, local_release_safe))
            return local_success, True

    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        FakeCoordinator,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: object())

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")

    class FailingRuntimeManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            raise RuntimeError("bad manifest")

    fake_weight_transfer.RuntimeManifest = FailingRuntimeManifest
    fake_weight_transfer.MemoryRegistrationLease = object
    fake_weight_transfer.MooncakeTransferEngineReader = object
    fake_weight_transfer.TransferEngineError = _CompletionUnknownError
    fake_weight_transfer.plan_runtime_transfer_to_local_target = object
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", fake_weight_transfer)
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    assert (
        loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
            object(), object(), "http://seed:30000", "target-session", object()
        )
        is False
    )
    assert outcomes == [("ready", False), (False, True)]


@pytest.mark.parametrize("error_type", [_CompletionUnknownError, RuntimeError])
def test_heterogeneous_loader_keeps_source_lease_when_transfer_completion_is_unknown(
    monkeypatch, error_type
) -> None:
    outcomes = []
    source_inventory = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "lease_id": "source-runtime-lease",
        "fragments": [SimpleNamespace(fragment_id="source-fragment")],
    }
    target_inventory = {
        "model_id": source_inventory["model_id"],
        "revision": source_inventory["revision"],
        "fragments": [SimpleNamespace(fragment_id="target-fragment")],
    }

    class FakeCoordinator:
        def __init__(self, seed_url, world_group):
            pass

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1", manifests=[source_inventory]
            )

        def raise_if_failed(self):
            raise AssertionError("loader must use the fixed readiness gate")

        def ready_for_transfer(self, local_ready):
            outcomes.append(("ready", local_ready))
            return local_ready

        def finish(self, *, local_success, local_release_safe=True):
            outcomes.append((local_success, local_release_safe))
            return False, False

    class FakeRuntimeManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            return SimpleNamespace(**inventory)

    class FakeRegistrationLease:
        @classmethod
        def from_fragment(cls, fragment, *, runtime_lease_id=None):
            return fragment

    class FailingReader:
        def __init__(self, engine, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            raise error_type("completion unknown")

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")
    fake_weight_transfer.MemoryRegistrationLease = FakeRegistrationLease
    fake_weight_transfer.MooncakeTransferEngineReader = FailingReader
    fake_weight_transfer.RuntimeManifest = FakeRuntimeManifest
    fake_weight_transfer.TransferEngineError = _CompletionUnknownError
    fake_weight_transfer.plan_runtime_transfer_to_local_target = (
        lambda sources, target: object()
    )
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", fake_weight_transfer)
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        FakeCoordinator,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: object())

    @contextlib.contextmanager
    def target_builder(**kwargs):
        yield target_inventory

    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    assert (
        loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
            object(),
            object(),
            "http://seed:30000",
            "target-session",
            target_builder,
        )
        is False
    )
    assert outcomes == [("ready", True), (False, False)]


def test_heterogeneous_loader_fails_closed_when_heartbeat_fails_during_transfer(
    monkeypatch,
) -> None:
    state = {"outcomes": []}
    source_inventory = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "lease_id": "source-runtime-lease",
        "fragments": [],
    }
    target_inventory = {
        "model_id": source_inventory["model_id"],
        "revision": source_inventory["revision"],
        "fragments": [],
    }

    class FakeRuntimeManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            return SimpleNamespace(**inventory)

    class FakeCoordinator:
        def __init__(self, seed_url, world_group):
            self.failed = False
            state["coordinator"] = self

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1",
                manifests=[source_inventory],
                lease_timeout_sec=60,
            )

        def raise_if_failed(self):
            if self.failed:
                raise RuntimeError("source lease renew failed")

        def ready_for_transfer(self, local_ready):
            state["readiness"] = local_ready
            return local_ready and not self.failed

        def finish(self, *, local_success, local_release_safe=True):
            if self.failed:
                local_success = False
            state["outcomes"].append((local_success, local_release_safe))
            return False, True

    class FakeReader:
        def __init__(self, engine, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            state["coordinator"].failed = True
            return [SimpleNamespace(nbytes=64, operation_count=1, request_count=1)]

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")
    fake_weight_transfer.MemoryRegistrationLease = SimpleNamespace(
        from_fragment=lambda fragment, **kwargs: fragment
    )
    fake_weight_transfer.MooncakeTransferEngineReader = FakeReader
    fake_weight_transfer.RuntimeManifest = FakeRuntimeManifest
    fake_weight_transfer.TransferEngineError = _CompletionUnknownError
    fake_weight_transfer.plan_runtime_transfer_to_local_target = (
        lambda sources, target: object()
    )
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", fake_weight_transfer)
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        FakeCoordinator,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: object())
    monkeypatch.setattr(loader_module.current_platform, "synchronize", lambda: None)
    monkeypatch.setattr(loader_module, "_post_load_weights", lambda model: None)

    @contextlib.contextmanager
    def target_builder(**kwargs):
        yield target_inventory

    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    assert (
        loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
            object(),
            object(),
            "http://seed:30000",
            "target-session",
            target_builder,
        )
        is False
    )
    assert state["readiness"] is True
    assert state["outcomes"] == [(False, True)]
