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


class _TransferEngineError(RuntimeError):
    pass


class _CompletionUnknownError(_TransferEngineError):
    def __init__(self, message, *, pending_transfer_id="pending-1"):
        super().__init__(message)
        self.pending_transfer_id = pending_transfer_id


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
        "lease_id": "target-runtime-lease",
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
        manifest_format="runtime_v1",
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
    fake_weight_transfer.TransferCompletionUnknownError = _CompletionUnknownError
    fake_weight_transfer.TransferEngineError = _TransferEngineError

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

    class TargetBuilderOwner:
        @contextlib.contextmanager
        def build_remote_instance_target_weight_manifest_session(self, **kwargs):
            del kwargs
            raise AssertionError(
                "runtime_v1 must use the legacy target manifest builder"
            )
            yield

        @contextlib.contextmanager
        def build_remote_instance_target_weight_runtime_manifest(self, **kwargs):
            calls["builder"] = kwargs
            yield target_inventory

    model = object()
    engine = object()
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    target_builder = (
        TargetBuilderOwner().build_remote_instance_target_weight_manifest_session
    )

    success = loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
        model,
        engine,
        "http://seed:30000",
        "target-session",
        target_builder,
    )

    assert success is release_success
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
        "target_registrations": ("lease:target-fragment:target-runtime-lease",),
    }
    assert calls["reader_options"] == {"max_batch_operations": 8192}
    assert calls["synchronized"] is True
    assert calls["post_loaded"] is model
    assert calls["coordinator"] == ("http://seed:30000", "target-world")
    assert calls["acquired"] == 1
    assert calls["ready"] is True
    assert calls["finish"] == (True, True)


def test_heterogeneous_loader_plans_placement_before_acquiring_target_binding(
    monkeypatch,
) -> None:
    events = []
    source_inventory = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "lease_id": "source-runtime-lease",
        "fragments": [SimpleNamespace(fragment_id="source-fragment")],
    }
    source_placement_inventory = SimpleNamespace(
        model_id=source_inventory["model_id"],
        revision=source_inventory["revision"],
        placement_id="source-placement",
    )
    source_binding_inventory = SimpleNamespace(
        model_id=source_inventory["model_id"],
        revision=source_inventory["revision"],
        placement_id="source-placement",
        lease_id="source-runtime-lease",
    )
    placement_inventory = SimpleNamespace(
        model_id=source_inventory["model_id"],
        revision=source_inventory["revision"],
        placement_id="target-placement",
    )
    binding_inventory = SimpleNamespace(
        model_id=source_inventory["model_id"],
        revision=source_inventory["revision"],
        placement_id="target-placement",
        lease_id="target-runtime-lease",
    )
    target_runtime = SimpleNamespace(
        model_id=source_inventory["model_id"],
        revision=source_inventory["revision"],
        lease_id="target-runtime-lease",
        fragments=[SimpleNamespace(fragment_id="target-fragment")],
    )

    class FakeRuntimeManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            return SimpleNamespace(**inventory)

    class FakeTargetPlacementManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            events.append("placement")
            return inventory

    class FakeSourcePlacementManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            events.append("source-placement")
            return inventory

    class FakeRuntimeBindingManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            events.append("binding")
            return inventory

    class FakeRegistrationLease:
        @classmethod
        def from_fragment(cls, fragment, *, runtime_lease_id=None):
            return (fragment.fragment_id, runtime_lease_id)

    class FakeReader:
        def __init__(self, engine, **kwargs):
            del engine, kwargs

        def execute(self, plan, sources, target, **kwargs):
            del plan, sources, target
            assert kwargs["source_registrations"] == (
                ("source-fragment", "source-runtime-lease"),
            )
            assert kwargs["target_registrations"] == (
                ("target-fragment", "target-runtime-lease"),
            )
            events.append("execute")
            return [SimpleNamespace(nbytes=64, operation_count=1, request_count=1)]

    class FakeCoordinator:
        def __init__(self, seed_url, world_group):
            del seed_url, world_group

        def acquire(self):
            return SimpleNamespace(
                manifests=[],
                source_placements=[source_placement_inventory],
                source_bindings=[source_binding_inventory],
                manifest_format="placement_binding_v1",
            )

        def ready_for_transfer(self, local_ready):
            return local_ready

        def finish(self, *, local_success, local_release_safe=True):
            return local_success, local_release_safe

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")
    fake_weight_transfer.MemoryRegistrationLease = FakeRegistrationLease
    fake_weight_transfer.MooncakeTransferEngineReader = FakeReader
    fake_weight_transfer.RuntimeManifest = FakeRuntimeManifest
    fake_weight_transfer.RuntimeBindingManifest = FakeRuntimeBindingManifest
    fake_weight_transfer.SourcePlacementManifest = FakeSourcePlacementManifest
    fake_weight_transfer.TargetPlacementManifest = FakeTargetPlacementManifest
    fake_weight_transfer.TransferCompletionUnknownError = _CompletionUnknownError
    fake_weight_transfer.TransferEngineError = _TransferEngineError

    def bind_runtime_manifest(placement, binding):
        if placement is source_placement_inventory:
            events.append("source-runtime-bind")
            return SimpleNamespace(**source_inventory)
        events.append("target-runtime-bind")
        return target_runtime

    fake_weight_transfer.bind_runtime_manifest = bind_runtime_manifest

    def plan_placement_transfer_to_local_target(sources, placement):
        assert sources == (source_placement_inventory,)
        assert placement is placement_inventory
        events.append("logical-plan")
        return "logical-plan"

    fake_weight_transfer.plan_placement_transfer_to_local_target = (
        plan_placement_transfer_to_local_target
    )

    def bind_logical_transfer_plan(logical, targets, *, source_bindings):
        assert logical == "logical-plan"
        assert targets == (binding_inventory,)
        assert source_bindings == (source_binding_inventory,)
        events.append("plan-bind")
        return SimpleNamespace(operations=("bound-operation",))

    fake_weight_transfer.bind_logical_transfer_plan = bind_logical_transfer_plan
    fake_weight_transfer.plan_runtime_transfer_to_local_target = (
        lambda sources, target: (_ for _ in ()).throw(
            AssertionError("the session path must not use the legacy planner")
        )
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

    class TargetSession:
        placement = placement_inventory

        @contextlib.contextmanager
        def bind(self):
            events.append("binding-lease-open")
            try:
                yield binding_inventory
            finally:
                events.append("binding-lease-close")

    @contextlib.contextmanager
    def target_builder(**kwargs):
        del kwargs
        yield TargetSession()

    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    success = loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
        object(),
        object(),
        "http://seed:30000",
        "target-session",
        target_builder,
    )

    assert success is True
    assert events == [
        "source-placement",
        "binding",
        "source-runtime-bind",
        "placement",
        "logical-plan",
        "binding-lease-open",
        "binding",
        "target-runtime-bind",
        "plan-bind",
        "execute",
        "binding-lease-close",
    ]


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


def test_heterogeneous_loader_blocks_world_when_any_rank_is_quarantined(
    monkeypatch,
) -> None:
    gathered = []

    class World:
        world_size = 2

        def all_gather_object(self, value):
            gathered.append(value)
            return [False, True]

    monkeypatch.setattr(loader_module, "get_world_group", lambda: World())
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        [],
    )
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        lambda *args, **kwargs: pytest.fail(
            "quarantined target world must not acquire a new source lease"
        ),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    assert (
        loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
            object(), object(), "http://seed:30000", "target-session", object()
        )
        is False
    )
    assert gathered == [False]


def test_legacy_loader_drains_ticket_before_releasing_target_model(
    monkeypatch,
) -> None:
    events = []

    class Ticket:
        status = "COMPLETION_UNKNOWN"

        def __init__(self):
            self.results = iter(("COMPLETION_UNKNOWN", "COMPLETED"))

        def drain(self, timeout_ms):
            assert timeout_ms > 0
            events.append(("drain", timeout_ms))
            return next(self.results)

    class Engine:
        def batch_transfer_sync_read_with_ticket(self, *args):
            events.append(("submit", args))
            return Ticket()

        def batch_transfer_sync_read(self, *args):
            raise AssertionError("ticket-capable engine must not use legacy sync API")

    tensor = SimpleNamespace(
        numel=lambda: 4,
        element_size=lambda: 2,
        data_ptr=lambda: 0x2000,
    )
    model = SimpleNamespace(named_parameters=lambda: [("weight", tensor)])
    monkeypatch.setattr(
        loader_module,
        "get_remote_instance_transfer_engine_info_per_rank",
        lambda seed_url, tp_rank: (
            "source-session",
            {"weight": (0x1000, 4, 2)},
        ),
    )
    monkeypatch.setattr(
        loader_module,
        "_post_load_weights",
        lambda loaded_model: events.append(("post_load", loaded_model)),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    success = loader.load_model_from_remote_instance_by_transfer_engine(
        model,
        Engine(),
        "http://seed:30000",
        0,
    )

    assert success is True
    assert [event[0] for event in events] == [
        "submit",
        "drain",
        "drain",
        "post_load",
    ]
    assert events[-1][1] is model


def test_legacy_loader_rejects_failed_drained_ticket(monkeypatch) -> None:
    class Ticket:
        status = "FAILED_DRAINED"

    class Engine:
        def batch_transfer_sync_read_with_ticket(self, *args):
            return Ticket()

    tensor = SimpleNamespace(
        numel=lambda: 4,
        element_size=lambda: 2,
        data_ptr=lambda: 0x2000,
    )
    model = SimpleNamespace(named_parameters=lambda: [("weight", tensor)])
    monkeypatch.setattr(
        loader_module,
        "get_remote_instance_transfer_engine_info_per_rank",
        lambda seed_url, tp_rank: (
            "source-session",
            {"weight": (0x1000, 4, 2)},
        ),
    )
    monkeypatch.setattr(
        loader_module,
        "_post_load_weights",
        lambda loaded_model: pytest.fail("failed transfer must not post-load"),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    assert (
        loader.load_model_from_remote_instance_by_transfer_engine(
            model,
            Engine(),
            "http://seed:30000",
            0,
        )
        is False
    )


def test_legacy_loader_defers_interrupt_until_ticket_is_drained(monkeypatch) -> None:
    events = []

    class Ticket:
        status = "COMPLETION_UNKNOWN"

        def __init__(self):
            self.drain_count = 0

        def drain(self, timeout_ms):
            self.drain_count += 1
            events.append(("drain", self.drain_count))
            if self.drain_count == 1:
                raise KeyboardInterrupt
            return "COMPLETED"

    class Engine:
        def batch_transfer_sync_read_with_ticket(self, *args):
            events.append(("submit", args))
            return Ticket()

    tensor = SimpleNamespace(
        numel=lambda: 4,
        element_size=lambda: 2,
        data_ptr=lambda: 0x2000,
    )
    model = SimpleNamespace(named_parameters=lambda: [("weight", tensor)])
    monkeypatch.setattr(
        loader_module,
        "get_remote_instance_transfer_engine_info_per_rank",
        lambda seed_url, tp_rank: (
            "source-session",
            {"weight": (0x1000, 4, 2)},
        ),
    )
    monkeypatch.setattr(
        loader_module,
        "_post_load_weights",
        lambda loaded_model: pytest.fail("interrupted load must not post-load"),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    with pytest.raises(KeyboardInterrupt):
        loader.load_model_from_remote_instance_by_transfer_engine(
            model,
            Engine(),
            "http://seed:30000",
            0,
        )

    assert [event[0] for event in events] == ["submit", "drain", "drain"]


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
    fake_weight_transfer.TransferCompletionUnknownError = _CompletionUnknownError
    fake_weight_transfer.TransferEngineError = _TransferEngineError
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


@pytest.mark.parametrize("drain_mode", ["terminal", "interrupt", "permanent"])
def test_heterogeneous_loader_drains_unknown_before_releasing_target_and_source(
    monkeypatch,
    drain_mode,
) -> None:
    events = []
    quarantine = []
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        quarantine,
    )
    if drain_mode == "permanent":
        monkeypatch.setattr(
            loader_module,
            "_HETEROGENEOUS_UNKNOWN_DRAIN_MAX_ATTEMPTS",
            2,
        )
        monkeypatch.setattr(
            loader_module,
            "_HETEROGENEOUS_UNKNOWN_DRAIN_TIMEOUT_MS",
            0,
        )
    source_inventory = {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "lease_id": "source-runtime-lease",
        "fragments": [SimpleNamespace(fragment_id="source-fragment")],
    }
    target_inventory = {
        "model_id": source_inventory["model_id"],
        "revision": source_inventory["revision"],
        "lease_id": "target-runtime-lease",
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
            events.append(("ready", local_ready))
            return local_ready

        def finish(self, *, local_success, local_release_safe=True):
            events.append(("finish", local_success, local_release_safe))
            return False, local_release_safe

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
            self.drain_results = iter(("COMPLETION_UNKNOWN", "FAILED_DRAINED"))
            self.interrupted = False

        def execute(self, *args, **kwargs):
            assert kwargs["target_pre_registered"] is True
            events.append("execute")
            raise _CompletionUnknownError(
                "completion unknown",
                pending_transfer_id="pending-1",
            )

        def drain_pending_transfer(self, pending_transfer_id, *, timeout_ms):
            assert pending_transfer_id == "pending-1"
            assert timeout_ms >= 0
            assert events[-1] != "target-close"
            if drain_mode == "interrupt" and not self.interrupted:
                self.interrupted = True
                events.append(("drain-error", "KeyboardInterrupt"))
                raise KeyboardInterrupt
            if drain_mode == "permanent":
                events.append(("drain", "COMPLETION_UNKNOWN"))
                return "COMPLETION_UNKNOWN"
            result = next(self.drain_results)
            events.append(("drain", result))
            return result

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")
    fake_weight_transfer.MemoryRegistrationLease = FakeRegistrationLease
    fake_weight_transfer.MooncakeTransferEngineReader = FailingReader
    fake_weight_transfer.RuntimeManifest = FakeRuntimeManifest
    fake_weight_transfer.TransferCompletionUnknownError = _CompletionUnknownError
    fake_weight_transfer.TransferEngineError = _TransferEngineError
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
        events.append("target-open")
        try:
            yield target_inventory
        finally:
            events.append("target-close")

    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    target_model = object()

    def load():
        return loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
            target_model,
            object(),
            "http://seed:30000",
            "target-session",
            target_builder,
        )

    if drain_mode == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            load()
    else:
        assert load() is False

    expected_events = [
        "target-open",
        ("ready", True),
        "execute",
    ]
    if drain_mode == "interrupt":
        expected_events.append(("drain-error", "KeyboardInterrupt"))
    if drain_mode == "permanent":
        expected_events.extend(
            [
                ("drain", "COMPLETION_UNKNOWN"),
                ("drain", "COMPLETION_UNKNOWN"),
                ("finish", False, False),
            ]
        )
    else:
        expected_events.extend(
            [
                ("drain", "COMPLETION_UNKNOWN"),
                ("drain", "FAILED_DRAINED"),
                "target-close",
                ("finish", False, True),
            ]
        )
    assert events == expected_events
    if drain_mode == "permanent":
        assert len(quarantine) == 1
        assert quarantine[0].pending_transfer_id == "pending-1"
        assert target_model in quarantine[0].owners
        quarantine[0].resources.close()
        assert events[-1] == "target-close"
        quarantine.clear()


@pytest.mark.parametrize("error_type", [_TransferEngineError, RuntimeError])
def test_heterogeneous_loader_releases_source_after_known_transfer_failure(
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
        "lease_id": "target-runtime-lease",
        "fragments": [SimpleNamespace(fragment_id="target-fragment")],
    }

    class FakeCoordinator:
        def __init__(self, seed_url, world_group):
            pass

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1", manifests=[source_inventory]
            )

        def ready_for_transfer(self, local_ready):
            outcomes.append(("ready", local_ready))
            return local_ready

        def finish(self, *, local_success, local_release_safe=True):
            outcomes.append((local_success, local_release_safe))
            return False, local_release_safe

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
            raise error_type("known failure")

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")
    fake_weight_transfer.MemoryRegistrationLease = FakeRegistrationLease
    fake_weight_transfer.MooncakeTransferEngineReader = FailingReader
    fake_weight_transfer.RuntimeManifest = FakeRuntimeManifest
    fake_weight_transfer.TransferCompletionUnknownError = _CompletionUnknownError
    fake_weight_transfer.TransferEngineError = _TransferEngineError
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
    assert outcomes == [("ready", True), (False, True)]


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
        "lease_id": "target-runtime-lease",
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
    fake_weight_transfer.TransferCompletionUnknownError = _CompletionUnknownError
    fake_weight_transfer.TransferEngineError = _TransferEngineError
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
