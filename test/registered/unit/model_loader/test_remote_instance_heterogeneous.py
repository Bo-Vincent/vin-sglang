import builtins
import contextlib
import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    compute_weight_placement_id,
)
from sglang.srt.model_loader import remote_instance_weight_loader_utils
from sglang.srt.model_loader import loader as loader_module
from sglang.srt.model_loader.loader import RemoteInstanceModelLoader
from sglang.srt.weight_transfer.provider import (
    WeightLoadReceipt,
    WeightProviderCapabilities,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_TARGET_MODEL_ID = "Qwen/Qwen3.5-0.8B"
_TARGET_REVISION = "main"


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
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        [],
    )


def _load_heterogeneous(loader, *args, **kwargs):
    kwargs.setdefault("target_model_id", _TARGET_MODEL_ID)
    kwargs.setdefault("target_revision", _TARGET_REVISION)
    return loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
        *args,
        **kwargs,
    )


def test_heterogeneous_loader_rejects_source_model_identity_mismatch() -> None:
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    with pytest.raises(ValueError, match="source manifest identity"):
        loader._require_manifest_identity(
            (
                SimpleNamespace(
                    model_id=_TARGET_MODEL_ID,
                    revision="different-revision",
                ),
            ),
            model_id=_TARGET_MODEL_ID,
            revision=_TARGET_REVISION,
            role="source",
        )


@pytest.mark.parametrize("release_success", [True, False])
def test_heterogeneous_loader_builds_local_plan_and_reads_from_source(
    monkeypatch,
    release_success,
) -> None:
    calls = {}
    source_inventory = {
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_REVISION,
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
            self.world_release_safe = True

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

    success = _load_heterogeneous(
        loader,
        model,
        engine,
        "http://seed:30000",
        "target-session",
        target_builder,
    )

    assert success is release_success
    assert calls["builder"] == {
        "model": model,
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_REVISION,
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
    if release_success:
        assert loader_module._HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE == []
    else:
        quarantine = loader_module._HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE
        assert len(quarantine) == 1
        assert quarantine[0].source_transfer_id == "transfer-1"
        assert quarantine[0].pending_transfer_id == "transfer-1:completed-rank-0"
        assert quarantine[0].terminal_status == "COMPLETED"
        assert quarantine[0].resources_closed is False
        quarantine[0].resources.close()
        quarantine.clear()


def test_heterogeneous_loader_recovers_readiness_failure_without_submission(
    monkeypatch,
) -> None:
    calls = {}
    source_inventory = {
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_REVISION,
        "lease_id": "source-runtime-lease",
        "fragments": [SimpleNamespace(fragment_id="source-fragment")],
    }
    target_inventory = {
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_REVISION,
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
            return (fragment.fragment_id, runtime_lease_id)

    class NoSubmissionReader:
        def __init__(self, engine, **kwargs):
            del engine, kwargs

        def execute(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("readiness failure must prevent DMA submission")

    class FakeCoordinator:
        world_release_safe = False

        def __init__(self, seed_url, world_group):
            del seed_url, world_group

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1",
                manifests=[source_inventory],
                manifest_format="runtime_v1",
            )

        def ready_for_transfer(self, local_ready):
            calls["ready"] = local_ready
            return False

        def finish(self, *, local_success, local_release_safe=True):
            calls["finish"] = (local_success, local_release_safe)
            return False, False

        def release_after_terminal_recovery(
            self,
            *,
            completion_ticket,
            local_terminal_status,
        ):
            calls["recovered"] = (completion_ticket, local_terminal_status)
            return True

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")
    fake_weight_transfer.MemoryRegistrationLease = FakeRegistrationLease
    fake_weight_transfer.MooncakeTransferEngineReader = NoSubmissionReader
    fake_weight_transfer.RuntimeManifest = FakeRuntimeManifest
    fake_weight_transfer.TransferCompletionUnknownError = _CompletionUnknownError
    fake_weight_transfer.TransferEngineError = _TransferEngineError
    fake_weight_transfer.plan_runtime_transfer_to_local_target = (
        lambda sources, target: SimpleNamespace(
            sources=sources,
            target=target,
            operations=("compact-operation",),
        )
    )
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", fake_weight_transfer)
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        FakeCoordinator,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: object())
    monkeypatch.setattr(
        loader_module,
        "_post_load_weights",
        lambda model: pytest.fail("readiness failure must not post-load weights"),
    )

    class TargetBuilderOwner:
        @contextlib.contextmanager
        def build_remote_instance_target_weight_runtime_manifest(self, **kwargs):
            del kwargs
            yield target_inventory

    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    assert (
        _load_heterogeneous(
            loader,
            object(),
            object(),
            "http://seed:30000",
            "target-session",
            TargetBuilderOwner().build_remote_instance_target_weight_runtime_manifest,
        )
        is False
    )

    quarantine = loader_module._HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE
    assert calls["ready"] is True
    assert calls["finish"] == (False, True)
    assert len(quarantine) == 1
    item = quarantine[0]
    assert item.pending_transfer_id == "transfer-1:no-submission-rank-0"
    assert item.terminal_status == "NO_SUBMISSION"
    assert item.resources_closed is False

    monkeypatch.setattr(loader_module, "get_world_group", _MirrorRecoveryWorld)
    assert loader_module.drain_heterogeneous_weight_transfer_quarantine(
        max_attempts=1,
        timeout_ms=0,
    )
    assert calls["recovered"] == (
        "transfer-1:no-submission-rank-0",
        "NO_SUBMISSION",
    )
    assert quarantine == []


def test_heterogeneous_loader_plans_placement_before_acquiring_target_binding(
    monkeypatch,
    caplog,
) -> None:
    caplog.set_level(logging.INFO)
    events = []
    tensor = WeightPlacementTensor(
        placement_fragment_id="source-fragment",
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        dtype="bfloat16",
        itemsize=2,
        partition_dim=None,
        shard_dims=(),
        layer_id=0,
        expert_id=None,
        layout_fingerprint="layout:v1",
        nbytes=16,
        byte_offset=0,
        rank=WeightParallelRank(),
    )
    source_placement_id = compute_weight_placement_id((tensor,))
    source_placement_inventory = WeightPlacementManifest(
        model_id=_TARGET_MODEL_ID,
        revision=_TARGET_REVISION,
        placement_id=source_placement_id,
        tensors=(tensor,),
    )
    source_binding_inventory = WeightRuntimeBindingManifest(
        model_id=_TARGET_MODEL_ID,
        revision=_TARGET_REVISION,
        placement_id=source_placement_id,
        instance_id="source-instance",
        generation=1,
        lease_id="source-runtime-lease",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id="source-fragment",
                fragment_id="source-fragment",
                address=0x10000,
                nbytes=16,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id="source-worker",
                endpoint="source:1",
            ),
        ),
    )
    target_tensor = WeightPlacementTensor(
        placement_fragment_id="target-fragment",
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        dtype="bfloat16",
        itemsize=2,
        partition_dim=None,
        shard_dims=(),
        layer_id=0,
        expert_id=None,
        layout_fingerprint="layout:v1",
        nbytes=16,
        byte_offset=0,
        rank=WeightParallelRank(),
    )
    target_placement_id = compute_weight_placement_id((target_tensor,))
    placement_inventory = WeightPlacementManifest(
        model_id=_TARGET_MODEL_ID,
        revision=_TARGET_REVISION,
        placement_id=target_placement_id,
        tensors=(target_tensor,),
    )
    binding_inventory = WeightRuntimeBindingManifest(
        model_id=_TARGET_MODEL_ID,
        revision=_TARGET_REVISION,
        placement_id=target_placement_id,
        instance_id="target-instance",
        generation=1,
        lease_id="target-runtime-lease",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id="target-fragment",
                fragment_id="target-fragment",
                address=0x20000,
                nbytes=16,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id="target-worker",
                endpoint="target:1",
            ),
        ),
    )
    source_inventory = {
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_REVISION,
        "lease_id": "source-runtime-lease",
        "fragments": [SimpleNamespace(fragment_id="source-fragment")],
    }
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
            return SimpleNamespace(
                model_id=inventory.model_id,
                revision=inventory.revision,
                placement_id=inventory.placement_id,
            )

    class FakeSourcePlacementManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            events.append("source-placement")
            return SimpleNamespace(
                model_id=inventory.model_id,
                revision=inventory.revision,
                placement_id=inventory.placement_id,
            )

    class FakeRuntimeBindingManifest:
        @classmethod
        def from_runtime_inventory(cls, inventory):
            events.append("binding")
            return SimpleNamespace(
                model_id=inventory.model_id,
                revision=inventory.revision,
                placement_id=inventory.placement_id,
                instance_id=inventory.instance_id,
                lease_id=inventory.lease_id,
            )

    class FakeRegistrationLease:
        @classmethod
        def from_fragment(cls, fragment, *, runtime_lease_id=None):
            return (fragment.fragment_id, runtime_lease_id)

    class FakeCoordinator:
        def __init__(self, seed_url, world_group):
            del seed_url, world_group

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1",
                manifests=[],
                source_placements=[source_placement_inventory],
                source_bindings=[source_binding_inventory],
                manifest_format="placement_binding_v1",
            )

        def ready_for_transfer(self, local_ready):
            events.append(("ready", local_ready))
            return local_ready

        def raise_if_failed(self):
            events.append("source-attest")

        def finish(self, *, local_success, local_release_safe=True):
            return local_success, local_release_safe

    fake_weight_transfer = ModuleType("mooncake.weight_transfer")
    fake_weight_transfer.MemoryRegistrationLease = FakeRegistrationLease
    fake_weight_transfer.MooncakeTransferEngineReader = object
    fake_weight_transfer.RuntimeManifest = FakeRuntimeManifest
    fake_weight_transfer.RuntimeBindingManifest = FakeRuntimeBindingManifest
    fake_weight_transfer.SourcePlacementManifest = FakeSourcePlacementManifest
    fake_weight_transfer.TargetPlacementManifest = FakeTargetPlacementManifest
    fake_weight_transfer.TransferCompletionUnknownError = _CompletionUnknownError
    fake_weight_transfer.TransferEngineError = _TransferEngineError

    def bind_runtime_manifest(placement, binding):
        if placement.placement_id == source_placement_id:
            events.append("source-runtime-bind")
            return SimpleNamespace(**source_inventory)
        events.append("target-runtime-bind")
        return target_runtime

    fake_weight_transfer.bind_runtime_manifest = bind_runtime_manifest

    fake_weight_transfer.plan_placement_transfer_to_local_target = (
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("the v4 path must use the SGLang planner")
        )
    )
    fake_weight_transfer.bind_logical_transfer_plan = object
    fake_weight_transfer.plan_runtime_transfer_to_local_target = (
        lambda sources, target: (_ for _ in ()).throw(
            AssertionError("the session path must not use the legacy planner")
        )
    )
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", fake_weight_transfer)
    real_import = builtins.__import__

    def reject_legacy_mooncake_import(name, *args, **kwargs):
        if name == "mooncake.weight_transfer":
            raise AssertionError(
                "placement_binding_v1 must not import the Mooncake legacy runtime"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_legacy_mooncake_import)
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        FakeCoordinator,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: object())
    monkeypatch.setattr(loader_module.current_platform, "synchronize", lambda: None)
    monkeypatch.setattr(loader_module, "_post_load_weights", lambda model: None)

    from sglang.srt.weight_transfer import api as weight_api
    from sglang.srt.weight_transfer import planner as weight_planner

    native_plan = weight_planner.plan_weight_transfer_to_local_target
    native_bind = weight_api.prepare_weight_load_from_plan

    def record_plan(*args, **kwargs):
        events.append("logical-plan")
        return native_plan(*args, **kwargs)

    def record_bind(*args, **kwargs):
        events.append("plan-bind")
        return native_bind(*args, **kwargs)

    monkeypatch.setattr(
        weight_planner,
        "plan_weight_transfer_to_local_target",
        record_plan,
    )
    monkeypatch.setattr(
        weight_api,
        "prepare_weight_load_from_plan",
        record_bind,
    )

    class FakeNativeProvider:
        name = "mooncake-te"

        def probe(self, request):
            events.append("probe")
            return WeightProviderCapabilities(
                provider=self.name,
                load_profiles=frozenset({"runtime_to_runtime"}),
                materialize_profiles=frozenset(),
                supports_nd_regions=True,
                supports_strided_regions=True,
                supports_safe_cancel=False,
                supports_completion_ticket=True,
                supports_transactional_publish=False,
            )

        def prepare(self, request):
            events.append("prepare")
            return request

        def submit(self, request):
            events.append("submit")
            return request

        def wait(self, request):
            events.append("execute")
            return WeightLoadReceipt(
                operation_id=request.operation_id,
                provider=self.name,
                plan_digest=request.plan.digest,
                total_bytes=request.plan.total_bytes,
                region_count=len(request.plan.regions),
                backend_receipts=(SimpleNamespace(nbytes=16, operation_count=1),),
            )

        def synchronize(self, receipt):
            events.append("provider-sync")

        def release(self, prepared, receipt):
            events.append("provider-release")

        def cancel(self, submission):
            raise AssertionError("successful transfer must not be cancelled")

    def provider_factory(engine, **kwargs):
        events.append(("provider", engine, kwargs))
        return FakeNativeProvider()

    class TargetSession:
        placement = placement_inventory

        @contextlib.contextmanager
        def bind(self):
            events.append("binding-lease-open")
            try:
                yield binding_inventory
            finally:
                events.append("binding-lease-close")

        def attest_binding(self, binding):
            assert binding is binding_inventory
            events.append("target-attest")

    @contextlib.contextmanager
    def target_builder(**kwargs):
        del kwargs
        yield TargetSession()

    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    engine = object()
    success = _load_heterogeneous(
        loader,
        object(),
        engine,
        "http://seed:30000",
        "target-session",
        target_builder,
        provider_factory=provider_factory,
    )

    assert success is True
    assert events.index("logical-plan") < events.index("binding-lease-open")
    assert events.index("plan-bind") > events.index("binding-lease-open")
    assert events.index(("ready", True)) < events.index("execute")
    assert events.index("source-attest") < events.index("probe")
    assert events.index("target-attest") < events.index("probe")
    assert events[-1] == "binding-lease-close"
    provider_event = next(
        item for item in events if isinstance(item, tuple) and item[0] == "provider"
    )
    assert provider_event == (
        "provider",
        engine,
        {"max_batch_operations": 8192},
    )
    assert "binding=" in caplog.text
    assert "lowering=" in caplog.text
    assert "data_transfer=" in caplog.text


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
        _load_heterogeneous(
            loader, object(), object(), "http://seed:30000", "target-session", object()
        )
        is False
    )


class _RecoveryExecutor:
    def __init__(self, *statuses):
        self.statuses = iter(statuses)
        self.calls = []

    def drain_completion(self, completion_ticket, *, timeout_ms):
        self.calls.append((completion_ticket, timeout_ms))
        return next(self.statuses)


class _RecoveryResources:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def close(self):
        self.events.append("target-close")
        self.closed = True


class _RecoveryCoordinator:
    def __init__(self, events, *, release_success=True):
        self.events = events
        self.release_success = release_success
        self.calls = []

    def release_after_terminal_recovery(
        self,
        *,
        completion_ticket,
        local_terminal_status,
    ):
        self.calls.append((completion_ticket, local_terminal_status))
        self.events.append(("source-release", local_terminal_status))
        return self.release_success


class _MirrorRecoveryWorld:
    rank_in_group = 0
    world_size = 1

    def __init__(self):
        self.gathers = []
        self.broadcasts = []

    def all_gather_object(self, value):
        self.gathers.append(value)
        return [value]

    def broadcast_object(self, value=None, src=0):
        self.broadcasts.append((value, src))
        return value


class _ScriptedRecoveryWorld:
    rank_in_group = 0
    world_size = 2

    def __init__(self, responses):
        self.responses = iter(responses)
        self.gathers = []

    def all_gather_object(self, value):
        self.gathers.append(value)
        response = next(self.responses)
        return response(value) if callable(response) else response


def _recovery_quarantine_item(
    *,
    source_transfer_id,
    completion_ticket,
    statuses,
    events,
    coordinator=None,
):
    executor = _RecoveryExecutor(*statuses)
    resources = _RecoveryResources(events)
    coordinator = coordinator or _RecoveryCoordinator(events)
    item = SimpleNamespace(
        source_transfer_id=source_transfer_id,
        pending_transfer_id=completion_ticket,
        transfer_executor=executor,
        resources=resources,
        coordinator=coordinator,
        owners=(),
        terminal_status=None,
        resources_closed=False,
    )
    return item, executor, resources, coordinator


def test_drain_heterogeneous_quarantine_keeps_unknown(monkeypatch) -> None:
    events = []
    item, executor, resources, coordinator = _recovery_quarantine_item(
        source_transfer_id="transfer-1",
        completion_ticket="ticket-1",
        statuses=("COMPLETION_UNKNOWN",),
        events=events,
    )
    quarantine = [item]
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        quarantine,
    )
    monkeypatch.setattr(loader_module, "get_world_group", _MirrorRecoveryWorld)

    assert (
        loader_module.drain_heterogeneous_weight_transfer_quarantine(
            max_attempts=1,
            timeout_ms=0,
        )
        is False
    )

    assert quarantine == [item]
    assert item.terminal_status is None
    assert executor.calls == [("ticket-1", 0)]
    assert coordinator.calls == []
    assert resources.closed is False
    assert events == []


def test_drain_heterogeneous_quarantine_waits_for_every_rank_then_releases(
    monkeypatch,
) -> None:
    events = []
    item, executor, resources, coordinator = _recovery_quarantine_item(
        source_transfer_id="transfer-1",
        completion_ticket="local-ticket-1",
        statuses=("COMPLETED",),
        events=events,
    )
    quarantine = [item]
    local_metadata = ((0, "transfer-1", "local-ticket-1"),)
    remote_metadata = ((1, "transfer-1", "remote-ticket-1"),)
    local_status = ((0, "transfer-1", "local-ticket-1", "COMPLETED"),)
    remote_unknown = ((1, "transfer-1", "remote-ticket-1", "COMPLETION_UNKNOWN"),)
    remote_terminal = ((1, "transfer-1", "remote-ticket-1", "FAILED_DRAINED"),)
    local_closed = ((0, "transfer-1", "local-ticket-1", True),)
    remote_closed = ((1, "transfer-1", "remote-ticket-1", True),)
    local_released = ((0, "transfer-1", "local-ticket-1", True),)
    remote_released = ((1, "transfer-1", "remote-ticket-1", True),)
    world = _ScriptedRecoveryWorld(
        (
            [local_metadata, remote_metadata],
            [local_status, remote_unknown],
            [local_metadata, remote_metadata],
            [local_status, remote_terminal],
            [local_closed, remote_closed],
            [local_released, remote_released],
        )
    )
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        quarantine,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: world)

    assert (
        loader_module.drain_heterogeneous_weight_transfer_quarantine(
            max_attempts=1,
            timeout_ms=0,
        )
        is False
    )
    assert quarantine == [item]
    assert item.terminal_status == "COMPLETED"
    assert resources.closed is False
    assert coordinator.calls == []

    assert (
        loader_module.drain_heterogeneous_weight_transfer_quarantine(
            max_attempts=1,
            timeout_ms=0,
        )
        is True
    )

    assert quarantine == []
    assert executor.calls == [("local-ticket-1", 0)]
    assert coordinator.calls == [("local-ticket-1", "COMPLETED")]
    assert resources.closed is True
    assert events == [
        "target-close",
        ("source-release", "COMPLETED"),
    ]


def test_drain_heterogeneous_quarantine_requires_every_rank_release_ack(
    monkeypatch,
) -> None:
    events = []
    item, _, resources, coordinator = _recovery_quarantine_item(
        source_transfer_id="transfer-1",
        completion_ticket="local-ticket-1",
        statuses=("COMPLETED",),
        events=events,
    )
    quarantine = [item]
    local_metadata = ((0, "transfer-1", "local-ticket-1"),)
    remote_metadata = ((1, "transfer-1", "remote-ticket-1"),)
    local_status = ((0, "transfer-1", "local-ticket-1", "COMPLETED"),)
    remote_status = ((1, "transfer-1", "remote-ticket-1", "COMPLETED"),)
    local_closed = ((0, "transfer-1", "local-ticket-1", True),)
    remote_closed = ((1, "transfer-1", "remote-ticket-1", True),)
    local_released = ((0, "transfer-1", "local-ticket-1", True),)
    remote_not_released = ((1, "transfer-1", "remote-ticket-1", False),)
    remote_released = ((1, "transfer-1", "remote-ticket-1", True),)
    world = _ScriptedRecoveryWorld(
        (
            [local_metadata, remote_metadata],
            [local_status, remote_status],
            [local_closed, remote_closed],
            [local_released, remote_not_released],
            [local_metadata, remote_metadata],
            [local_status, remote_status],
            [local_closed, remote_closed],
            [local_released, remote_released],
        )
    )
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        quarantine,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: world)

    assert (
        loader_module.drain_heterogeneous_weight_transfer_quarantine(
            max_attempts=1,
            timeout_ms=0,
        )
        is False
    )
    assert quarantine == [item]
    assert resources.closed is True
    assert coordinator.calls == [("local-ticket-1", "COMPLETED")]

    assert (
        loader_module.drain_heterogeneous_weight_transfer_quarantine(
            max_attempts=1,
            timeout_ms=0,
        )
        is True
    )
    assert quarantine == []
    assert coordinator.calls == [
        ("local-ticket-1", "COMPLETED"),
        ("local-ticket-1", "COMPLETED"),
    ]
    assert events == [
        "target-close",
        ("source-release", "COMPLETED"),
        ("source-release", "COMPLETED"),
    ]


@pytest.mark.parametrize(
    "remote_statuses",
    [
        pytest.param(
            ((1, "transfer-1", "remote-ticket-1", "COMPLETED"),),
            id="count",
        ),
        pytest.param(
            [
                (1, "transfer-1", "remote-ticket-1", "COMPLETED"),
                (1, "transfer-2", "remote-ticket-2", "COMPLETED"),
            ],
            id="container-type",
        ),
        pytest.param(
            (
                (1, "transfer-2", "remote-ticket-2", "COMPLETED"),
                (1, "transfer-1", "remote-ticket-1", "COMPLETED"),
            ),
            id="order",
        ),
        pytest.param(
            (
                (1, "transfer-1", "remote-ticket-1", "SUCCESS"),
                (1, "transfer-2", "remote-ticket-2", "COMPLETED"),
            ),
            id="status",
        ),
        pytest.param(
            (
                (True, "transfer-1", "remote-ticket-1", "COMPLETED"),
                (1, "transfer-2", "remote-ticket-2", "COMPLETED"),
            ),
            id="rank-type",
        ),
    ],
)
def test_drain_heterogeneous_quarantine_rejects_invalid_world_statuses(
    monkeypatch,
    remote_statuses,
) -> None:
    events = []
    first, _, first_resources, first_coordinator = _recovery_quarantine_item(
        source_transfer_id="transfer-1",
        completion_ticket="local-ticket-1",
        statuses=("COMPLETED",),
        events=events,
    )
    second, _, second_resources, second_coordinator = _recovery_quarantine_item(
        source_transfer_id="transfer-2",
        completion_ticket="local-ticket-2",
        statuses=("FAILED_DRAINED",),
        events=events,
    )
    quarantine = [first, second]
    local_metadata = (
        (0, "transfer-1", "local-ticket-1"),
        (0, "transfer-2", "local-ticket-2"),
    )
    remote_metadata = (
        (1, "transfer-1", "remote-ticket-1"),
        (1, "transfer-2", "remote-ticket-2"),
    )
    local_statuses = (
        (0, "transfer-1", "local-ticket-1", "COMPLETED"),
        (0, "transfer-2", "local-ticket-2", "FAILED_DRAINED"),
    )
    world = _ScriptedRecoveryWorld(
        (
            [local_metadata, remote_metadata],
            [local_statuses, remote_statuses],
        )
    )
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        quarantine,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: world)

    assert (
        loader_module.drain_heterogeneous_weight_transfer_quarantine(
            max_attempts=1,
            timeout_ms=0,
        )
        is False
    )

    assert quarantine == [first, second]
    assert first_resources.closed is False
    assert second_resources.closed is False
    assert first_coordinator.calls == []
    assert second_coordinator.calls == []
    assert events == []


def test_world_coordinator_terminal_recovery_release_failure_is_fail_closed(
    monkeypatch,
) -> None:
    release_calls = []
    session = SimpleNamespace(
        transfer_id="transfer-1",
        manifests=[],
        lease_timeout_sec=90,
    )

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self):
            pass

        def raise_if_failed(self):
            pass

        def stop(self):
            pass

    world = _MirrorRecoveryWorld()
    monkeypatch.setattr(
        remote_instance_weight_loader_utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url: session,
    )
    monkeypatch.setattr(
        remote_instance_weight_loader_utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: (
            release_calls.append((seed_url, transfer_id)) or False
        ),
    )
    monkeypatch.setattr(
        remote_instance_weight_loader_utils,
        "RemoteInstanceWeightTransferHeartbeat",
        FakeHeartbeat,
    )
    coordinator = remote_instance_weight_loader_utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        world,
    )

    assert coordinator.acquire() is session
    assert coordinator.finish(
        local_success=False,
        local_release_safe=False,
    ) == (False, False)
    with pytest.raises(ValueError, match="terminal completion status"):
        coordinator.release_after_terminal_recovery(
            completion_ticket="ticket-1",
            local_terminal_status="COMPLETION_UNKNOWN",
        )

    events = []
    item, executor, resources, _ = _recovery_quarantine_item(
        source_transfer_id="transfer-1",
        completion_ticket="ticket-1",
        statuses=("FAILED_DRAINED",),
        events=events,
        coordinator=coordinator,
    )
    quarantine = [item]
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        quarantine,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: world)

    for _ in range(2):
        assert (
            loader_module.drain_heterogeneous_weight_transfer_quarantine(
                max_attempts=1,
                timeout_ms=0,
            )
            is False
        )

    assert release_calls == [
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
    ]
    assert executor.calls == [("ticket-1", 0)]
    assert quarantine == [item]
    assert item.resources_closed is True
    assert resources.closed is True
    assert events == ["target-close"]


def test_heterogeneous_loader_blocks_world_when_any_rank_is_quarantined(
    monkeypatch,
) -> None:
    gathered = []

    class World:
        world_size = 2

        def all_gather_object(self, value):
            gathered.append(value)
            if type(value) is tuple:
                return [(), ((1, "transfer-1", "remote-ticket-1"),)]
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
        _load_heterogeneous(
            loader, object(), object(), "http://seed:30000", "target-session", object()
        )
        is False
    )
    assert gathered == [(), False]


def test_heterogeneous_loader_attempts_recovery_before_quarantine_block(
    monkeypatch,
) -> None:
    events = []
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        [object()],
    )
    monkeypatch.setattr(
        loader_module,
        "drain_heterogeneous_weight_transfer_quarantine",
        lambda **kwargs: events.append(("drain", kwargs)) or False,
        raising=False,
    )
    monkeypatch.setattr(
        loader_module,
        "get_world_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        lambda *args, **kwargs: pytest.fail(
            "blocked load must not acquire a new source lease"
        ),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    assert (
        _load_heterogeneous(
            loader,
            object(),
            object(),
            "http://seed:30000",
            "target-session",
            object(),
        )
        is False
    )
    assert events == [
        (
            "drain",
            {
                "max_attempts": 1,
                "timeout_ms": (loader_module._HETEROGENEOUS_UNKNOWN_DRAIN_TIMEOUT_MS),
            },
        )
    ]


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
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_REVISION,
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
        _load_heterogeneous(
            loader, object(), object(), "http://seed:30000", "target-session", object()
        )
        is False
    )
    assert outcomes == [("ready", False), (False, True)]


@pytest.mark.parametrize(
    "drain_mode",
    ["terminal", "interrupt", "permanent", "missing_ticket", "invalid"],
)
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
    if drain_mode in {"permanent", "invalid"}:
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
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_REVISION,
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
                pending_transfer_id=(
                    None if drain_mode == "missing_ticket" else "pending-1"
                ),
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
            if drain_mode == "invalid":
                events.append(("drain", "INVALID_STATUS"))
                return "INVALID_STATUS"
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
        return _load_heterogeneous(
            loader,
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
    elif drain_mode == "missing_ticket":
        expected_events.append(("finish", False, False))
    elif drain_mode == "invalid":
        expected_events.extend(
            [
                ("drain", "INVALID_STATUS"),
                ("drain", "INVALID_STATUS"),
                ("finish", False, False),
            ]
        )
    else:
        expected_events.extend(
            [
                ("drain", "COMPLETION_UNKNOWN"),
                ("drain", "FAILED_DRAINED"),
                ("finish", False, True),
                "target-close",
            ]
        )
    assert events == expected_events
    if drain_mode in {"permanent", "missing_ticket", "invalid"}:
        assert len(quarantine) == 1
        assert quarantine[0].pending_transfer_id == (
            "transfer-1:completion-unknown-rank-0"
            if drain_mode == "missing_ticket"
            else "pending-1"
        )
        assert quarantine[0].source_transfer_id == "transfer-1"
        assert isinstance(quarantine[0].transfer_executor, FailingReader)
        assert isinstance(quarantine[0].coordinator, FakeCoordinator)
        assert quarantine[0].terminal_status is None
        assert quarantine[0].resources_closed is False
        assert target_model in quarantine[0].owners
        quarantine[0].resources.close()
        assert events[-1] == "target-close"
        quarantine.clear()


@pytest.mark.parametrize(
    ("error_type", "release_safe"),
    [(_TransferEngineError, True), (RuntimeError, False)],
)
def test_heterogeneous_loader_requires_completion_proof_before_release(
    monkeypatch, error_type, release_safe
) -> None:
    outcomes = []
    source_inventory = {
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_REVISION,
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
        _load_heterogeneous(
            loader,
            object(),
            object(),
            "http://seed:30000",
            "target-session",
            target_builder,
        )
        is False
    )
    assert outcomes == [("ready", True), (False, release_safe)]


def test_heterogeneous_loader_fails_closed_when_heartbeat_fails_during_transfer(
    monkeypatch,
) -> None:
    state = {"outcomes": []}
    source_inventory = {
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_REVISION,
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
        _load_heterogeneous(
            loader,
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
