import contextlib
import sys
import time
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
    compose_weight_runtime_manifest,
    compute_weight_placement_id,
)
from sglang.srt.model_loader import loader as loader_module
from sglang.srt.model_loader import remote_instance_weight_loader_utils
from sglang.srt.model_loader.loader import RemoteInstanceModelLoader
from sglang.srt.weight_transfer.binding import runtime_manifest_to_parts
from sglang.srt.weight_transfer.planner import (
    plan_weight_transfer as build_full_world_plan,
)
from sglang.srt.weight_transfer.planner import (
    project_weight_transfer_plan_to_target,
)
from sglang.srt.weight_transfer.provider import (
    WeightLoadReceipt,
    WeightProviderCapabilities,
    WeightTransferExecutionContext,
)
from sglang.srt.weight_transfer.remote_protocol import (
    ARTIFACT_WEIGHT_VERSION_V1,
    HF_REVISION_V1,
    PLACEMENT_BINDING_V1,
    RUNTIME_MANIFEST_V1,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_TARGET_MODEL_ID = "Qwen/Qwen3.5-0.8B"
_TARGET_ARTIFACT_REVISION = "weights-v7"
_TARGET_HF_REVISION = "main"
_REAL_LEGACY_RUNTIME_V1_SUPPORTS_BOUNDED_EXECUTION = (
    loader_module._legacy_runtime_v1_supports_bounded_execution
)


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
    monkeypatch.setattr(
        loader_module,
        "probe_remote_instance_weight_transfer_capabilities",
        lambda **_kwargs: (
            remote_instance_weight_loader_utils.RemoteInstanceWeightTransferCapabilities(
                native_executor=False,
                canonical_adapter=True,
                legacy_planner=True,
            )
        ),
        raising=False,
    )
    monkeypatch.setattr(
        loader_module,
        "_legacy_runtime_v1_supports_bounded_execution",
        lambda _backend: True,
        raising=False,
    )


def _load_heterogeneous(loader, *args, **kwargs):
    kwargs.setdefault("target_model_id", _TARGET_MODEL_ID)
    kwargs.setdefault("target_artifact_revision", _TARGET_ARTIFACT_REVISION)
    kwargs.setdefault("target_hf_revision", _TARGET_HF_REVISION)
    return loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
        *args,
        **kwargs,
    )


def _capability_probe_provider(*, validate_environment=lambda: None):
    return SimpleNamespace(
        name="test-provider",
        bounded_execution_contract_version=1,
        validate_environment=validate_environment,
        probe=lambda request: request,
        prepare=lambda request: request,
        submit=lambda request: request,
        wait=lambda request: request,
        cancel=lambda request: None,
        synchronize=lambda receipt, **kwargs: None,
        release=lambda prepared, receipt: None,
    )


@pytest.mark.parametrize("failure", ["unavailable", "unsupported"])
def test_explicit_provider_probe_does_not_use_legacy_defaults(
    monkeypatch,
    failure,
) -> None:
    def unavailable():
        raise RuntimeError("provider unavailable")

    provider = _capability_probe_provider(
        validate_environment=unavailable if failure == "unavailable" else lambda: None
    )
    if failure == "unsupported":
        provider.release = None
    monkeypatch.setattr(
        remote_instance_weight_loader_utils,
        "_load_legacy_mooncake_weight_backend",
        lambda: pytest.fail("explicit provider probe must not load the legacy backend"),
        raising=False,
    )
    monkeypatch.setattr(
        remote_instance_weight_loader_utils,
        "supports_mooncake_placement_binding_v1",
        lambda: pytest.fail("explicit provider probe must not use legacy capabilities"),
        raising=False,
    )

    capabilities = remote_instance_weight_loader_utils.probe_remote_instance_weight_transfer_capabilities(
        provider=provider,
    )

    assert capabilities.native_executor is False
    assert capabilities.legacy_planner is False


def test_target_artifact_revision_is_not_the_hf_revision(monkeypatch) -> None:
    monkeypatch.setattr(
        loader_module,
        "get_server_args",
        lambda: SimpleNamespace(weight_version="weights-v7"),
    )

    assert loader_module._configured_weight_artifact_revision() == "weights-v7"


def test_legacy_target_revision_alias_preserves_prior_identity() -> None:
    assert loader_module._resolve_target_weight_revisions(
        target_artifact_revision=None,
        target_hf_revision=None,
        target_revision="main",
    ) == ("main", "main")


def test_target_revision_alias_rejects_ambiguous_identity() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        loader_module._resolve_target_weight_revisions(
            target_artifact_revision="weights-v7",
            target_hf_revision="main",
            target_revision="legacy",
        )


@pytest.mark.parametrize(
    "source_revision_semantics",
    ("legacy_hf_unattested", HF_REVISION_V1),
)
def test_artifact_policy_rejects_hf_fallback(source_revision_semantics) -> None:
    with pytest.raises(RuntimeError, match="did not attest artifact weight version"):
        loader_module._resolve_remote_manifest_revision(
            manifest_format=RUNTIME_MANIFEST_V1,
            source_revision_semantics=source_revision_semantics,
            allow_legacy_hf_fallback=False,
            target_artifact_revision="default",
            target_hf_revision=_TARGET_HF_REVISION,
        )


def test_runtime_artifact_attestation_uses_artifact_revision() -> None:
    assert (
        loader_module._resolve_remote_manifest_revision(
            manifest_format=RUNTIME_MANIFEST_V1,
            source_revision_semantics=ARTIFACT_WEIGHT_VERSION_V1,
            allow_legacy_hf_fallback=False,
            target_artifact_revision=_TARGET_ARTIFACT_REVISION,
            target_hf_revision=_TARGET_HF_REVISION,
        )
        == _TARGET_ARTIFACT_REVISION
    )


@pytest.mark.parametrize(
    "source_revision_semantics",
    (HF_REVISION_V1, "legacy_hf_unattested"),
)
def test_explicit_legacy_policy_uses_hf_revision(source_revision_semantics) -> None:
    assert (
        loader_module._resolve_remote_manifest_revision(
            manifest_format=RUNTIME_MANIFEST_V1,
            source_revision_semantics=source_revision_semantics,
            allow_legacy_hf_fallback=True,
            target_artifact_revision=_TARGET_ARTIFACT_REVISION,
            target_hf_revision=_TARGET_HF_REVISION,
        )
        == _TARGET_HF_REVISION
    )


def test_unattested_revision_requires_runtime_hf_compatibility() -> None:
    assert (
        loader_module._resolve_remote_manifest_revision(
            manifest_format=PLACEMENT_BINDING_V1,
            source_revision_semantics="legacy_hf_unattested",
            allow_legacy_hf_fallback=True,
            target_artifact_revision=_TARGET_ARTIFACT_REVISION,
            target_hf_revision=_TARGET_HF_REVISION,
        )
        == _TARGET_HF_REVISION
    )


def test_native_legacy_revision_policy_requires_unambiguous_identity() -> None:
    capabilities = (
        remote_instance_weight_loader_utils.RemoteInstanceWeightTransferCapabilities(
            native_executor=True,
            canonical_adapter=True,
            legacy_planner=False,
        )
    )

    assert loader_module._allow_legacy_hf_manifest_revision(
        capabilities,
        target_artifact_revision="main",
        target_hf_revision="main",
    )
    assert not loader_module._allow_legacy_hf_manifest_revision(
        capabilities,
        target_artifact_revision="weights-v7",
        target_hf_revision="main",
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
            revision=_TARGET_HF_REVISION,
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
        "revision": _TARGET_HF_REVISION,
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
        deadline_unix_sec=time.time() + 120,
    )

    class FakeCoordinator:
        def __init__(self, seed_url, world_group, **_kwargs):
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
        target_artifact_revision=_TARGET_ARTIFACT_REVISION,
        target_hf_revision=_TARGET_HF_REVISION,
    )

    assert success is release_success
    assert calls["builder"] == {
        "model": model,
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_HF_REVISION,
        "instance_id": "sglang:target-session",
        "endpoint": "target-session",
    }
    assert calls["plan"][1].revision == source_inventory["revision"]
    _, _, _, execute_kwargs = calls["execute"]
    execution_context = execute_kwargs.pop("execution_context")
    assert isinstance(execution_context, WeightTransferExecutionContext)
    assert execution_context.remaining_seconds() > 0
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
        "revision": _TARGET_HF_REVISION,
        "lease_id": "source-runtime-lease",
        "fragments": [SimpleNamespace(fragment_id="source-fragment")],
    }
    target_inventory = {
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_HF_REVISION,
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

        def __init__(self, seed_url, world_group, **_kwargs):
            del seed_url, world_group

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1",
                manifests=[source_inventory],
                manifest_format="runtime_v1",
                deadline_unix_sec=time.time() + 120,
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


def test_strict_artifact_identity_rejects_unattested_runtime_before_adapter(
    monkeypatch,
) -> None:
    events = []

    class FakeCoordinator:
        def __init__(self, seed_url, world_group, **_kwargs):
            del seed_url, world_group

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1",
                manifests=[{"model_id": _TARGET_MODEL_ID}],
                source_placements=None,
                source_bindings=None,
                manifest_format="runtime_v1",
                deadline_unix_sec=time.time() + 120,
            )

        def ready_for_transfer(self, local_ready):
            events.append(("ready", local_ready))
            return local_ready

        def finish(self, *, local_success, local_release_safe=True):
            events.append(("finish", local_success, local_release_safe))
            return local_success, local_release_safe

    def fail_adapter(self, inventories):
        del self, inventories
        raise AssertionError("unattested identity must fail before adaptation")

    def reject_legacy(self, **kwargs):
        del self, kwargs
        raise AssertionError("unattested identity must fail before planning")

    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        FakeCoordinator,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: object())
    monkeypatch.setattr(
        loader_module,
        "probe_remote_instance_weight_transfer_capabilities",
        lambda **_kwargs: (
            remote_instance_weight_loader_utils.RemoteInstanceWeightTransferCapabilities(
                native_executor=True,
                canonical_adapter=True,
                legacy_planner=True,
            )
        ),
    )
    monkeypatch.setattr(
        RemoteInstanceModelLoader,
        "_adapt_runtime_v1_source_inventories",
        fail_adapter,
    )
    monkeypatch.setattr(
        RemoteInstanceModelLoader,
        "_prepare_legacy_heterogeneous_weight_load",
        reject_legacy,
    )

    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    success = _load_heterogeneous(
        loader,
        object(),
        object(),
        "http://seed:30000",
        "target-session",
        object(),
        provider_factory=lambda *_args, **_kwargs: object(),
        target_artifact_revision="default",
    )

    assert success is False
    assert events == [
        ("ready", False),
        ("finish", False, True),
    ]


def _legacy_runtime_v1_inventory(partition_dim):
    return {
        "format_version": 1,
        "model_id": _TARGET_MODEL_ID,
        "revision": _TARGET_ARTIFACT_REVISION,
        "instance_id": "source-instance",
        "generation": 1,
        "lease_id": "source-lease",
        "tensors": [
            {
                "fragment_id": "source-fragment",
                "tensor_id": "weight",
                "runtime_name": "weight",
                "aliases": ["weight"],
                "global_shape": [8],
                "global_offset": [0],
                "local_shape": [8],
                "dtype": "bfloat16",
                "itemsize": 2,
                "partition_dim": partition_dim,
                "layer_id": 0,
                "expert_id": None,
                "layout_fingerprint": "layout:v1",
                "address": 0x10000,
                "nbytes": 16,
                "byte_offset": 0,
                "stride": [1],
                "storage_offset": 0,
                "device": "cuda:0",
                "is_contiguous": True,
                "worker_id": "source-worker",
                "endpoint": "source:1",
                "rank": {"dp": 0, "tp": 0, "pp": 0, "ep": 0},
                "lease_generation": 1,
            }
        ],
    }


@pytest.mark.parametrize(
    ("partition_dim", "expected_shard_dims"),
    [(0, (0,)), (None, ())],
)
def test_runtime_v1_adapter_migrates_pre_shard_dims_wire(
    partition_dim,
    expected_shard_dims,
) -> None:
    inventory = _legacy_runtime_v1_inventory(partition_dim)
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    placements, bindings = loader._adapt_runtime_v1_source_inventories((inventory,))

    assert placements[0].tensors[0].shard_dims == expected_shard_dims
    assert placements[0].tensors[0].rank.moe_dp == 0
    assert bindings[0].fragments[0].fragment_id == "source-fragment"
    assert "shard_dims" not in inventory["tensors"][0]


def _canonical_manifest_parts(
    side: str,
    *,
    address: int,
    revision: str = _TARGET_ARTIFACT_REVISION,
    parallel_rank: WeightParallelRank = WeightParallelRank(),
    global_offset: int = 0,
    local_size: int = 8,
):
    sharded = global_offset != 0 or local_size != 8
    tensor = WeightPlacementTensor(
        placement_fragment_id=f"{side}-fragment",
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=(8,),
        global_offset=(global_offset,),
        local_shape=(local_size,),
        dtype="bfloat16",
        itemsize=2,
        partition_dim=0 if sharded else None,
        shard_dims=(0,) if sharded else (),
        layer_id=0,
        expert_id=None,
        layout_fingerprint="layout:v1",
        nbytes=local_size * 2,
        byte_offset=0,
        rank=parallel_rank,
    )
    placement_id = compute_weight_placement_id((tensor,))
    placement = WeightPlacementManifest(
        model_id=_TARGET_MODEL_ID,
        revision=revision,
        placement_id=placement_id,
        tensors=(tensor,),
    )
    binding = WeightRuntimeBindingManifest(
        model_id=_TARGET_MODEL_ID,
        revision=revision,
        placement_id=placement_id,
        instance_id=f"{side}-instance",
        generation=1,
        lease_id=f"{side}-runtime-lease",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=f"{side}-fragment",
                fragment_id=f"{side}-fragment",
                address=address,
                nbytes=local_size * 2,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id=f"{side}-worker",
                endpoint=f"{side}:1",
            ),
        ),
    )
    return placement, binding


def _run_canonical_remote_load(
    monkeypatch,
    *,
    manifest_format: str,
    manifest_revision: str = _TARGET_ARTIFACT_REVISION,
    revision_semantics: str = ARTIFACT_WEIGHT_VERSION_V1,
    allow_legacy_hf_fallback: bool = False,
    target_artifact_revision: str = _TARGET_ARTIFACT_REVISION,
    target_hf_revision: str = _TARGET_HF_REVISION,
    include_session_deadline: bool = True,
    include_coordinator_deadline: bool = True,
    provider_supports_bounded_execution: bool = True,
):
    source_placement, source_binding = _canonical_manifest_parts(
        "source",
        address=0x10000,
        revision=manifest_revision,
    )
    target_placement, target_binding = _canonical_manifest_parts(
        "target",
        address=0x20000,
        revision=manifest_revision,
    )
    source_runtime = compose_weight_runtime_manifest(
        source_placement,
        source_binding,
    )
    calls = {
        "attestations": 0,
        "binding_close": 0,
        "binding_open": 0,
        "coordinator_finish": [],
        "coordinator_ready": [],
        "post_load": [],
        "provider_lifecycle": [],
        "runtime_adapter": [],
        "target_close": 0,
        "target_open": 0,
    }
    shared_deadline_unix_sec = time.time() + 120
    transfer_session = SimpleNamespace(
        transfer_id="transfer-1",
        lease_timeout_sec=300,
        manifests=(source_runtime,) if manifest_format == RUNTIME_MANIFEST_V1 else (),
        source_placements=(
            None if manifest_format == RUNTIME_MANIFEST_V1 else (source_placement,)
        ),
        source_bindings=(
            None if manifest_format == RUNTIME_MANIFEST_V1 else (source_binding,)
        ),
        manifest_format=manifest_format,
        manifest_revision_semantics=revision_semantics,
        allow_legacy_hf_fallback=allow_legacy_hf_fallback,
        deadline_unix_sec=(
            shared_deadline_unix_sec if include_session_deadline else None
        ),
    )
    transfer_handle = (
        remote_instance_weight_loader_utils.RemoteInstanceWeightTransferSessionHandle(
            transfer_id=transfer_session.transfer_id,
            lease_timeout_sec=transfer_session.lease_timeout_sec,
            manifest_format=transfer_session.manifest_format,
            manifest_revision_semantics=(transfer_session.manifest_revision_semantics),
            allow_legacy_hf_fallback=(transfer_session.allow_legacy_hf_fallback),
            deadline_unix_sec=(
                shared_deadline_unix_sec if include_session_deadline else None
            ),
        )
        if manifest_format == PLACEMENT_BINDING_V1
        else transfer_session
    )

    class FakeCoordinator:
        world_release_safe = True

        def __init__(self, seed_url, world_group, **kwargs):
            calls["coordinator"] = (seed_url, world_group, kwargs)
            self.execution_context = (
                WeightTransferExecutionContext(
                    deadline_unix_sec=shared_deadline_unix_sec
                )
                if include_coordinator_deadline
                else None
            )
            self.owner_source_session = (
                transfer_session if manifest_format == PLACEMENT_BINDING_V1 else None
            )

        def acquire(self):
            return transfer_handle

        def clear_owner_source_session(self):
            if self.owner_source_session is not None:
                calls["owner_source_session_cleared"] = (
                    calls.get("owner_source_session_cleared", 0) + 1
                )
            self.owner_source_session = None

        def ready_for_transfer(self, local_ready):
            calls["coordinator_ready"].append(local_ready)
            return local_ready

        def raise_if_failed(self):
            calls["attestations"] += 1

        def finish(self, *, local_success, local_release_safe=True):
            calls["coordinator_finish"].append((local_success, local_release_safe))
            return local_success, local_release_safe

    class FakeNativeProvider:
        name = "mooncake-te"
        bounded_execution_contract_version = 1

        def probe(self, request):
            calls["provider_request"] = request
            calls["provider_lifecycle"].append("probe")
            return WeightProviderCapabilities(
                provider=self.name,
                load_profiles=frozenset({"runtime_to_runtime"}),
                materialize_profiles=frozenset(),
                supports_nd_regions=True,
                supports_strided_regions=True,
                supports_safe_cancel=False,
                supports_completion_ticket=True,
                supports_transactional_publish=False,
                supports_bounded_execution=provider_supports_bounded_execution,
            )

        def prepare(self, request, *, execution_context=None):
            calls["provider_lifecycle"].append("prepare")
            calls.setdefault("execution_contexts", []).append(execution_context)
            return request

        def submit(self, request):
            calls["provider_lifecycle"].append("submit")
            return request

        def wait(self, request, *, execution_context=None):
            calls["provider_lifecycle"].append("wait")
            calls.setdefault("execution_contexts", []).append(execution_context)
            return WeightLoadReceipt(
                operation_id=request.operation_id,
                provider=self.name,
                plan_digest=request.plan.digest,
                total_bytes=request.plan.total_bytes,
                region_count=len(request.plan.regions),
                backend_receipts=(SimpleNamespace(nbytes=16, operation_count=1),),
            )

        def synchronize(self, receipt, *, execution_context=None):
            calls["provider_lifecycle"].append("synchronize")
            calls.setdefault("execution_contexts", []).append(execution_context)

        def release(self, prepared, receipt, *, execution_context=None):
            calls["provider_lifecycle"].append("release")
            calls.setdefault("execution_contexts", []).append(execution_context)

        def cancel(self, submission):
            raise AssertionError("successful transfer must not be cancelled")

    def provider_factory(engine, **kwargs):
        calls["provider_factory"] = (engine, kwargs)
        return FakeNativeProvider()

    class TargetSession:
        placement = target_placement

        @contextlib.contextmanager
        def bind(self):
            calls["binding_open"] += 1
            try:
                yield target_binding
            finally:
                calls["binding_close"] += 1

        def attest_binding(self, binding):
            assert binding is target_binding
            calls["attestations"] += 1

    @contextlib.contextmanager
    def target_builder(**kwargs):
        calls["target_builder"] = kwargs
        calls["target_open"] += 1
        try:
            yield TargetSession()
        finally:
            calls["target_close"] += 1

    real_plan = loader_module.plan_weight_transfer_to_local_target

    def record_plan(source_placements, target):
        calls["local_planner"] = (source_placements, target)
        return real_plan(source_placements, target)

    def record_full_world_plan(
        source_placements,
        target_placements,
        *,
        expected_target_topology,
    ):
        calls["global_planner"] = (
            source_placements,
            target_placements,
            expected_target_topology,
        )
        return build_full_world_plan(
            source_placements,
            target_placements,
            expected_target_topology=expected_target_topology,
        )

    def record_runtime_adapter(manifest):
        calls["runtime_adapter"].append(manifest)
        return runtime_manifest_to_parts(manifest)

    def reject_legacy_backend():
        raise AssertionError("canonical loading must not initialize the legacy backend")

    def record_platform_synchronize():
        calls["platform_synchronize"] = calls.get("platform_synchronize", 0) + 1

    class SingleRankWorld:
        rank_in_group = 0
        world_size = 1

        def gather_object(self, value, dst=0):
            calls.setdefault("gathered_target_placements", []).append((value, dst))
            return [value]

        def scatter_object(self, values, src=0):
            calls.setdefault("scattered_plans", []).append((values, src))
            return values[0]

    world_group = SingleRankWorld()
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        FakeCoordinator,
    )
    monkeypatch.setattr(loader_module, "get_world_group", lambda: world_group)
    monkeypatch.setattr(
        loader_module,
        "probe_remote_instance_weight_transfer_capabilities",
        lambda **_kwargs: (
            remote_instance_weight_loader_utils.RemoteInstanceWeightTransferCapabilities(
                native_executor=True,
                canonical_adapter=True,
                legacy_planner=True,
            )
        ),
    )
    monkeypatch.setattr(
        loader_module,
        "_load_legacy_mooncake_weight_backend",
        reject_legacy_backend,
    )
    monkeypatch.setattr(
        loader_module,
        "plan_weight_transfer_to_local_target",
        record_plan,
    )
    monkeypatch.setattr(
        loader_module,
        "plan_weight_transfer",
        record_full_world_plan,
        raising=False,
    )
    monkeypatch.setattr(
        loader_module,
        "runtime_manifest_to_parts",
        record_runtime_adapter,
    )
    monkeypatch.setattr(
        loader_module.current_platform,
        "synchronize",
        record_platform_synchronize,
    )
    monkeypatch.setattr(
        loader_module,
        "_post_load_weights",
        lambda model: calls["post_load"].append(model),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    model = object()
    engine = object()
    success = _load_heterogeneous(
        loader,
        model,
        engine,
        "http://seed:30000",
        "target-session",
        target_builder,
        provider_factory=provider_factory,
        target_artifact_revision=target_artifact_revision,
        target_hf_revision=target_hf_revision,
    )
    return SimpleNamespace(
        calls=calls,
        engine=engine,
        model=model,
        source_binding=source_binding,
        source_placement=source_placement,
        source_runtime=source_runtime,
        success=success,
        target_binding=target_binding,
        target_placement=target_placement,
        world_group=world_group,
        manifest_format=manifest_format,
        shared_deadline_unix_sec=shared_deadline_unix_sec,
    )


def _assert_canonical_remote_load(
    result,
    *,
    expected_source_placement=None,
    expected_source_binding=None,
    expected_revision: str = _TARGET_ARTIFACT_REVISION,
    expected_legacy_hf_fallback: bool = False,
) -> None:
    expected_source_placement = (
        result.source_placement
        if expected_source_placement is None
        else expected_source_placement
    )
    expected_source_binding = (
        result.source_binding
        if expected_source_binding is None
        else expected_source_binding
    )
    assert result.success is True
    assert result.calls["target_builder"] == {
        "model": result.model,
        "model_id": _TARGET_MODEL_ID,
        "revision": expected_revision,
        "instance_id": "sglang:target-session",
        "endpoint": "target-session",
    }
    if result.manifest_format == PLACEMENT_BINDING_V1:
        assert result.calls["global_planner"] == (
            (expected_source_placement,),
            (result.target_placement,),
            (WeightParallelRank(),),
        )
        assert "local_planner" not in result.calls
        assert result.calls["owner_source_session_cleared"] == 1
    else:
        assert result.calls["local_planner"] == (
            (expected_source_placement,),
            result.target_placement,
        )
        assert "global_planner" not in result.calls
    request = result.calls["provider_request"]
    assert request.plan.logical_plan.revision == expected_revision
    assert request.plan.source_bindings == (expected_source_binding,)
    assert request.plan.target_bindings == (result.target_binding,)
    assert result.calls["provider_factory"] == (
        result.engine,
        {"max_batch_operations": 8192},
    )
    assert result.calls["provider_lifecycle"].count("probe") == 1
    assert result.calls["provider_lifecycle"].count("prepare") == 1
    assert result.calls["provider_lifecycle"].count("submit") == 1
    assert result.calls["provider_lifecycle"].count("wait") == 1
    assert result.calls["provider_lifecycle"].count("synchronize") == 1
    assert result.calls["provider_lifecycle"].count("release") == 1
    contexts = result.calls["execution_contexts"]
    assert len(contexts) == 4
    assert all(
        isinstance(context, WeightTransferExecutionContext) for context in contexts
    )
    assert contexts[0] is contexts[1] is contexts[2]
    assert contexts[0].deadline_unix_sec == result.shared_deadline_unix_sec
    assert 0 < contexts[0].remaining_seconds() <= 120
    assert contexts[3] is not contexts[0]
    assert contexts[3].cancel_signal is None
    assert 0 < contexts[3].remaining_seconds() <= 5
    assert result.calls["coordinator_ready"] == [True]
    assert result.calls["coordinator_finish"] == [(True, True)]
    seed_url, world_group, coordinator_options = result.calls["coordinator"]
    assert seed_url == "http://seed:30000"
    assert world_group is result.world_group
    assert (
        coordinator_options["manifest_revision_semantics"] == ARTIFACT_WEIGHT_VERSION_V1
    )
    assert (
        coordinator_options["allow_legacy_hf_fallback"] is expected_legacy_hf_fallback
    )
    assert result.calls["attestations"] == 4
    assert result.calls["binding_open"] == 1
    assert result.calls["binding_close"] == 1
    assert result.calls["target_open"] == 1
    assert result.calls["target_close"] == 1
    assert result.calls["platform_synchronize"] == 1
    assert result.calls["post_load"] == [result.model]
    assert loader_module._HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE == []


def test_placement_binding_v1_uses_native_planner_provider_and_leases(
    monkeypatch,
) -> None:
    result = _run_canonical_remote_load(
        monkeypatch,
        manifest_format=PLACEMENT_BINDING_V1,
    )

    _assert_canonical_remote_load(result)
    assert result.calls["runtime_adapter"] == []


def test_native_load_rejects_session_without_absolute_deadline(
    monkeypatch,
    caplog,
) -> None:
    result = _run_canonical_remote_load(
        monkeypatch,
        manifest_format=PLACEMENT_BINDING_V1,
        include_session_deadline=False,
        include_coordinator_deadline=False,
    )

    assert result.success is False
    assert result.calls["provider_lifecycle"] == []
    assert result.calls["target_open"] == 0
    assert result.calls["binding_open"] == 0
    assert result.calls["coordinator_ready"] == []
    assert "provider_request" not in result.calls
    assert "absolute deadline" in caplog.text


def test_runtime_v1_adapts_to_native_planner_provider_and_leases(
    monkeypatch,
) -> None:
    result = _run_canonical_remote_load(
        monkeypatch,
        manifest_format=RUNTIME_MANIFEST_V1,
    )
    adapted = runtime_manifest_to_parts(result.source_runtime)

    _assert_canonical_remote_load(
        result,
        expected_source_placement=adapted.placement,
        expected_source_binding=adapted.binding,
    )
    assert result.calls["runtime_adapter"] == [result.source_runtime]


def test_native_provider_rejects_unbounded_capability_during_preflight(
    monkeypatch,
    caplog,
) -> None:
    with caplog.at_level("ERROR"):
        result = _run_canonical_remote_load(
            monkeypatch,
            manifest_format=PLACEMENT_BINDING_V1,
            provider_supports_bounded_execution=False,
        )

    assert result.success is False
    assert result.calls["provider_lifecycle"] == ["probe"]
    assert result.calls["coordinator_ready"] == [False]
    assert result.calls["coordinator_finish"] == [(False, True)]
    assert (
        "native provider requires supports_bounded_execution=true "
        "for bounded execution contract version 1"
    ) in caplog.text


def test_legacy_bounded_execution_requires_contract_version(monkeypatch) -> None:
    monkeypatch.setattr(
        loader_module,
        "_legacy_runtime_v1_supports_bounded_execution",
        _REAL_LEGACY_RUNTIME_V1_SUPPORTS_BOUNDED_EXECUTION,
    )
    backend = SimpleNamespace(supports_bounded_execution=True)

    assert loader_module._legacy_runtime_v1_supports_bounded_execution(backend) is False

    backend.bounded_execution_contract_version = 1
    assert loader_module._legacy_runtime_v1_supports_bounded_execution(backend) is True
    del backend.supports_bounded_execution
    assert loader_module._legacy_runtime_v1_supports_bounded_execution(backend) is False


def test_native_preflight_requires_explicit_bounded_capability(monkeypatch) -> None:
    provider = SimpleNamespace(bounded_execution_contract_version=1)
    monkeypatch.setattr(
        loader_module,
        "preflight_weight_transfer",
        lambda *_args, **_kwargs: SimpleNamespace(
            _capabilities=SimpleNamespace(),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "native provider requires supports_bounded_execution=true "
            "for bounded execution contract version 1"
        ),
    ):
        loader_module._preflight_bounded_native_weight_transfer(
            provider,
            object(),
            attestor=object(),
        )


def test_unbounded_legacy_runtime_v1_fails_before_session_acquire(
    monkeypatch,
    caplog,
) -> None:
    backend = SimpleNamespace()
    monkeypatch.setattr(
        loader_module,
        "get_world_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        loader_module,
        "_load_legacy_mooncake_weight_backend",
        lambda: backend,
    )
    monkeypatch.setattr(
        loader_module,
        "_legacy_runtime_v1_supports_bounded_execution",
        lambda candidate: candidate is not backend,
    )
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        lambda *_args, **_kwargs: pytest.fail(
            "unbounded legacy executor must not acquire a source session"
        ),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    with caplog.at_level("ERROR"):
        success = _load_heterogeneous(
            loader,
            object(),
            object(),
            "http://seed:30000",
            "target-session",
            object(),
        )

    assert success is False
    assert "bounded target-world deadline" in caplog.text


@pytest.mark.parametrize("local_failure", [False, True])
def test_provider_preflight_failure_is_voted_before_session_collectives(
    monkeypatch,
    caplog,
    local_failure,
) -> None:
    phases = []
    remote_error = "configured provider contract mismatch"

    class World:
        rank_in_group = 0
        world_size = 2

        def all_gather_object(self, value, *, phase, execution_context):
            assert isinstance(execution_context, WeightTransferExecutionContext)
            phases.append(phase)
            if phase == "heterogeneous_quarantine.metadata":
                return [value, value]
            if phase == "heterogeneous_quarantine.preflight":
                return [value, value]
            assert phase == "heterogeneous_provider.preflight"
            remote = loader_module._RankLocalProviderPreflightOutcome(
                world_rank=1,
                error=None if local_failure else remote_error,
                capability_fingerprint=(
                    ("native", True, True, False) if local_failure else None
                ),
            )
            return [value, remote]

    def provider_factory(*_args, **_kwargs):
        if local_failure:
            raise RuntimeError("rank-local factory failure")
        return _capability_probe_provider()

    monkeypatch.setattr(loader_module, "get_world_group", lambda: World())
    monkeypatch.setattr(
        loader_module,
        "probe_remote_instance_weight_transfer_capabilities",
        lambda **_kwargs: (
            remote_instance_weight_loader_utils.RemoteInstanceWeightTransferCapabilities(
                native_executor=True,
                canonical_adapter=True,
                legacy_planner=False,
            )
        ),
    )
    monkeypatch.setattr(
        loader_module,
        "_load_legacy_mooncake_weight_backend",
        lambda: pytest.fail("explicit provider failure must not use legacy fallback"),
    )
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        lambda *_args, **_kwargs: pytest.fail(
            "provider preflight failure must not enter session collectives"
        ),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    with caplog.at_level("ERROR"):
        success = _load_heterogeneous(
            loader,
            object(),
            object(),
            "http://seed:30000",
            "target-session",
            object(),
            provider_factory=provider_factory,
        )

    expected_rank = 0 if local_failure else 1
    expected_error = (
        "configured provider factory failed: RuntimeError: rank-local factory failure"
        if local_failure
        else remote_error
    )
    assert success is False
    assert phases == [
        "heterogeneous_quarantine.metadata",
        "heterogeneous_quarantine.preflight",
        "heterogeneous_provider.preflight",
    ]
    assert (
        f"Target-world provider preflight failed at rank {expected_rank}: "
        f"{expected_error}"
    ) in caplog.messages


def test_provider_preflight_rejects_cross_rank_capability_mismatch(
    monkeypatch,
    caplog,
) -> None:
    phases = []

    class World:
        rank_in_group = 0
        world_size = 2

        def all_gather_object(self, value, *, phase, execution_context):
            assert isinstance(execution_context, WeightTransferExecutionContext)
            phases.append(phase)
            if phase == "heterogeneous_quarantine.metadata":
                return [value, value]
            if phase == "heterogeneous_quarantine.preflight":
                return [value, value]
            return [
                value,
                loader_module._RankLocalProviderPreflightOutcome(
                    world_rank=1,
                    capability_fingerprint=("legacy", False, True, True),
                ),
            ]

    monkeypatch.setattr(loader_module, "get_world_group", lambda: World())
    monkeypatch.setattr(
        loader_module,
        "probe_remote_instance_weight_transfer_capabilities",
        lambda **_kwargs: (
            remote_instance_weight_loader_utils.RemoteInstanceWeightTransferCapabilities(
                native_executor=True,
                canonical_adapter=True,
                legacy_planner=False,
            )
        ),
    )
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        lambda *_args, **_kwargs: pytest.fail(
            "capability mismatch must not enter session collectives"
        ),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    with caplog.at_level("ERROR"):
        success = _load_heterogeneous(
            loader,
            object(),
            object(),
            "http://seed:30000",
            "target-session",
            object(),
            provider_factory=lambda *_args, **_kwargs: _capability_probe_provider(),
        )

    assert success is False
    assert phases == [
        "heterogeneous_quarantine.metadata",
        "heterogeneous_quarantine.preflight",
        "heterogeneous_provider.preflight",
    ]
    assert (
        "Target-world provider preflight capability mismatch: "
        "rank 1 differs from rank 0"
    ) in caplog.messages


@pytest.mark.parametrize("failure", ["factory", "capability", "invalid"])
def test_explicit_provider_failure_does_not_fall_back_to_legacy(
    monkeypatch,
    failure,
) -> None:
    calls = []

    def provider_factory(*_args, **_kwargs):
        if failure == "factory":
            raise RuntimeError("provider factory failed")
        return SimpleNamespace(name="incomplete-provider")

    def capability_probe(**kwargs):
        if failure == "capability":
            raise RuntimeError("provider capability probe failed")
        return remote_instance_weight_loader_utils.probe_remote_instance_weight_transfer_capabilities(
            **kwargs
        )

    configured_factory = object() if failure == "invalid" else provider_factory
    monkeypatch.setattr(
        loader_module,
        "get_world_group",
        lambda: SimpleNamespace(world_size=1),
    )
    monkeypatch.setattr(
        loader_module,
        "probe_remote_instance_weight_transfer_capabilities",
        capability_probe,
    )
    monkeypatch.setattr(
        loader_module,
        "_load_legacy_mooncake_weight_backend",
        lambda: calls.append("legacy"),
    )
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        lambda *_args, **_kwargs: calls.append("coordinator"),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)

    success = _load_heterogeneous(
        loader,
        object(),
        object(),
        "http://seed:30000",
        "target-session",
        object(),
        provider_factory=configured_factory,
    )

    assert success is False
    assert calls == []


def test_runtime_v1_legacy_hf_uses_negotiated_session_policy(monkeypatch) -> None:
    result = _run_canonical_remote_load(
        monkeypatch,
        manifest_format=RUNTIME_MANIFEST_V1,
        manifest_revision=_TARGET_HF_REVISION,
        revision_semantics=remote_instance_weight_loader_utils.LEGACY_HF_UNATTESTED,
        allow_legacy_hf_fallback=True,
    )
    adapted = runtime_manifest_to_parts(result.source_runtime)

    _assert_canonical_remote_load(
        result,
        expected_source_placement=adapted.placement,
        expected_source_binding=adapted.binding,
        expected_revision=_TARGET_HF_REVISION,
    )


def test_head_source_placement_manifest_uses_unambiguous_hf_identity(
    monkeypatch,
) -> None:
    result = _run_canonical_remote_load(
        monkeypatch,
        manifest_format=PLACEMENT_BINDING_V1,
        manifest_revision=_TARGET_HF_REVISION,
        revision_semantics=remote_instance_weight_loader_utils.LEGACY_HF_UNATTESTED,
        allow_legacy_hf_fallback=True,
        target_artifact_revision=_TARGET_HF_REVISION,
        target_hf_revision=_TARGET_HF_REVISION,
    )

    _assert_canonical_remote_load(
        result,
        expected_revision=_TARGET_HF_REVISION,
        expected_legacy_hf_fallback=True,
    )


def _tp2_manifest_parts(prefix: str, *, base_address: int):
    rank0 = WeightParallelRank(tp=0)
    rank1 = WeightParallelRank(tp=1)
    return (
        _canonical_manifest_parts(
            f"{prefix}-0",
            address=base_address,
            parallel_rank=rank0,
            global_offset=0,
            local_size=4,
        ),
        _canonical_manifest_parts(
            f"{prefix}-1",
            address=base_address + 0x1000,
            parallel_rank=rank1,
            global_offset=4,
            local_size=4,
        ),
    )


def _target_session(placement, binding, bind_calls):
    class TargetSession:
        def __init__(self):
            self.placement = placement

        @contextlib.contextmanager
        def bind(self):
            bind_calls.append(placement.placement_id)
            yield binding

        def attest_binding(self, actual):
            assert actual == binding

    @contextlib.contextmanager
    def builder(**_kwargs):
        yield TargetSession()

    return builder


def _prepare_distributed_native(
    monkeypatch,
    *,
    world_group,
    coordinator,
    target_parts,
):
    bind_calls = []
    monkeypatch.setattr(
        loader_module,
        "preflight_weight_transfer",
        lambda provider, request, *, attestor: SimpleNamespace(
            provider=provider,
            request=request,
            attestor=attestor,
            _capabilities=SimpleNamespace(
                supports_bounded_execution=True,
            ),
        ),
    )
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    phase_seconds = {
        "source_manifest": 0.0,
        "target_manifest": 0.0,
        "plan": 0.0,
        "binding": 0.0,
    }
    with contextlib.ExitStack() as resources:
        prepared = loader._prepare_distributed_native_heterogeneous_weight_load(
            model=object(),
            coordinator=coordinator,
            world_group=world_group,
            transfer_resources=resources,
            target_manifest_builder=_target_session(
                *target_parts,
                bind_calls,
            ),
            target_model_id=_TARGET_MODEL_ID,
            manifest_revision=_TARGET_ARTIFACT_REVISION,
            local_session_id=f"target-{world_group.rank_in_group}",
            transfer_executor=SimpleNamespace(
                bounded_execution_contract_version=1,
            ),
            phase_seconds=phase_seconds,
        )
    return prepared, bind_calls


def test_placement_binding_root_plans_once_and_scatters_source_subsets(
    monkeypatch,
) -> None:
    source_parts = _tp2_manifest_parts("source", base_address=0x10000)
    target_parts = _tp2_manifest_parts("target", base_address=0x20000)
    planner_calls = []
    projection_calls = []
    real_planner = loader_module.plan_weight_transfer
    real_projection = loader_module.project_weight_transfer_plan_to_targets

    def record_planner(*args, **kwargs):
        planner_calls.append((args, kwargs))
        return real_planner(*args, **kwargs)

    def record_projection(*args, **kwargs):
        projection_calls.append((args, kwargs))
        return real_projection(*args, **kwargs)

    monkeypatch.setattr(loader_module, "plan_weight_transfer", record_planner)
    monkeypatch.setattr(
        loader_module,
        "project_weight_transfer_plan_to_targets",
        record_projection,
    )

    class Coordinator:
        def __init__(self):
            self.owner_source_session = SimpleNamespace(
                source_placements=tuple(item[0] for item in source_parts),
                source_bindings=tuple(item[1] for item in source_parts),
            )
            self.cleared = 0

        def clear_owner_source_session(self):
            self.owner_source_session = None
            self.cleared += 1

        def raise_if_failed(self):
            pass

    coordinator = Coordinator()

    class RootWorld:
        rank_in_group = 0
        world_size = 2

        def gather_object(self, local, dst=0):
            assert dst == 0
            return [
                local,
                loader_module._TargetPlacementEnvelope(
                    world_rank=1,
                    parallel_rank=WeightParallelRank(tp=1),
                    placement=target_parts[1][0],
                ),
            ]

        def scatter_object(self, values, src=0):
            assert src == 0
            self.values = values
            return values[0]

    world = RootWorld()
    prepared, bind_calls = _prepare_distributed_native(
        monkeypatch,
        world_group=world,
        coordinator=coordinator,
        target_parts=target_parts[0],
    )

    assert len(planner_calls) == 1
    assert len(projection_calls) == 1
    assert coordinator.cleared == 1
    assert bind_calls == [target_parts[0][0].placement_id]
    assert prepared.source_placements == (source_parts[0][0],)
    assert prepared.source_bindings == (source_parts[0][1],)
    assert tuple(
        item.placement_id for item in world.values[1].logical_plan.source_placements
    ) == (source_parts[1][0].placement_id,)
    assert world.values[1].source_bindings == (source_parts[1][1],)


def test_placement_binding_central_plan_uses_session_bounded_collectives(
    monkeypatch,
) -> None:
    source_parts = _tp2_manifest_parts("source", base_address=0x10000)
    target_parts = _tp2_manifest_parts("target", base_address=0x20000)
    session = remote_instance_weight_loader_utils.RemoteInstanceWeightTransferSession(
        transfer_id="transfer-1",
        manifests=[],
        lease_timeout_sec=300,
        source_placements=[item[0] for item in source_parts],
        source_bindings=[item[1] for item in source_parts],
        manifest_format=PLACEMENT_BINDING_V1,
        manifest_revision_semantics=ARTIFACT_WEIGHT_VERSION_V1,
    )

    class OriginalWorld:
        rank_in_group = 0
        world_size = 2

        @staticmethod
        def broadcast_object(*_args, **_kwargs):
            raise AssertionError("session must replace synchronous broadcast")

        @staticmethod
        def all_gather_object(*_args, **_kwargs):
            raise AssertionError("session must replace synchronous all-gather")

        @staticmethod
        def gather_object(*_args, **_kwargs):
            raise AssertionError("session must replace synchronous gather")

        @staticmethod
        def scatter_object(*_args, **_kwargs):
            raise AssertionError("session must replace synchronous scatter")

    class BoundedCollectives:
        rank = 0
        world_size = 2

        def __init__(self):
            self.calls = []

        def _record(self, phase, execution_context):
            assert isinstance(execution_context, WeightTransferExecutionContext)
            self.calls.append((phase, execution_context))

        def synchronize_object_collective_deadline(
            self,
            *,
            phase,
            execution_context,
        ):
            assert phase == "remote_instance.acquire.deadline_control"
            assert isinstance(execution_context, WeightTransferExecutionContext)
            self.deadline_context = execution_context
            return execution_context.deadline_unix_sec

        def broadcast_object(
            self,
            value,
            *,
            src,
            phase,
            execution_context,
        ):
            assert src == 0
            self._record(phase, execution_context)
            return value

        def gather_object(
            self,
            value,
            *,
            dst,
            phase,
            execution_context,
        ):
            assert dst == 0
            self._record(phase, execution_context)
            return [
                value,
                loader_module._TargetPlacementEnvelope(
                    world_rank=1,
                    parallel_rank=WeightParallelRank(tp=1),
                    placement=target_parts[1][0],
                ),
            ]

        def scatter_object(
            self,
            values,
            *,
            src,
            phase,
            execution_context,
        ):
            assert src == 0
            self._record(phase, execution_context)
            self.values = values
            return values[0]

        def all_gather_object(
            self,
            value,
            *,
            phase,
            execution_context,
        ):
            self._record(phase, execution_context)
            return [value] * self.world_size

    class NoopHeartbeat:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            pass

        def raise_if_failed(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(
        remote_instance_weight_loader_utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url, **_kwargs: session,
    )
    monkeypatch.setattr(
        remote_instance_weight_loader_utils,
        "RemoteInstanceWeightTransferHeartbeat",
        NoopHeartbeat,
    )
    monkeypatch.setattr(
        remote_instance_weight_loader_utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: True,
    )
    collectives = BoundedCollectives()
    world = OriginalWorld()
    coordinator = remote_instance_weight_loader_utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        world,
        capabilities=(
            remote_instance_weight_loader_utils.RemoteInstanceWeightTransferCapabilities(
                native_executor=True,
                canonical_adapter=True,
                legacy_planner=False,
            )
        ),
        manifest_revision_semantics=ARTIFACT_WEIGHT_VERSION_V1,
        collective_coordinator=collectives,
    )

    coordinator.acquire()
    prepared, bind_calls = _prepare_distributed_native(
        monkeypatch,
        world_group=world,
        coordinator=coordinator,
        target_parts=target_parts[0],
    )

    assert bind_calls == [target_parts[0][0].placement_id]
    assert prepared.source_placements == (source_parts[0][0],)
    assert prepared.source_bindings == (source_parts[0][1],)
    assert tuple(
        item.placement_id
        for item in collectives.values[1].logical_plan.source_placements
    ) == (source_parts[1][0].placement_id,)
    assert [phase for phase, _ in collectives.calls[:3]] == [
        "remote_instance.acquire.broadcast",
        "remote_instance.central_plan.gather",
        "remote_instance.central_plan.scatter",
    ]
    assert all(
        context is coordinator.execution_context for _, context in collectives.calls[:3]
    )


def test_placement_binding_non_root_prepares_without_source_session(
    monkeypatch,
) -> None:
    source_parts = _tp2_manifest_parts("source", base_address=0x10000)
    target_parts = _tp2_manifest_parts("target", base_address=0x20000)
    global_plan = build_full_world_plan(
        tuple(item[0] for item in source_parts),
        tuple(item[0] for item in target_parts),
        expected_target_topology=(
            WeightParallelRank(tp=0),
            WeightParallelRank(tp=1),
        ),
    )
    local_plan = project_weight_transfer_plan_to_target(
        global_plan,
        target_parts[1][0].placement_id,
    )
    local_bindings = tuple(
        loader_module.project_source_bindings(
            local_plan.source_placements,
            tuple(item[1] for item in source_parts),
        )
    )
    local_envelope = loader_module._RankLocalTransferEnvelope(
        world_rank=1,
        parallel_rank=WeightParallelRank(tp=1),
        target_placement_id=target_parts[1][0].placement_id,
        logical_plan=local_plan,
        source_bindings=local_bindings,
    )

    class Coordinator:
        cleared = 0

        @property
        def owner_source_session(self):
            raise AssertionError("non-root must not read the full source session")

        def clear_owner_source_session(self):
            self.cleared += 1

        def raise_if_failed(self):
            pass

    class FollowerWorld:
        rank_in_group = 1
        world_size = 2

        def gather_object(self, local, dst=0):
            assert local.placement == target_parts[1][0]
            return None

        def scatter_object(self, values, src=0):
            assert values is None
            return local_envelope

    coordinator = Coordinator()
    prepared, bind_calls = _prepare_distributed_native(
        monkeypatch,
        world_group=FollowerWorld(),
        coordinator=coordinator,
        target_parts=target_parts[1],
    )

    assert coordinator.cleared == 1
    assert bind_calls == [target_parts[1][0].placement_id]
    assert prepared.source_placements == (source_parts[1][0],)
    assert prepared.source_bindings == (source_parts[1][1],)


@pytest.mark.parametrize(
    "failure",
    ("wrong-rank", "placement-mismatch", "source-binding-closure"),
)
def test_rank_local_envelope_fails_before_target_bind(monkeypatch, failure) -> None:
    source_parts = _tp2_manifest_parts("source", base_address=0x10000)
    target_parts = _tp2_manifest_parts("target", base_address=0x20000)
    global_plan = build_full_world_plan(
        tuple(item[0] for item in source_parts),
        tuple(item[0] for item in target_parts),
        expected_target_topology=(
            WeightParallelRank(tp=0),
            WeightParallelRank(tp=1),
        ),
    )
    selected_target = target_parts[0 if failure == "placement-mismatch" else 1][0]
    local_plan = project_weight_transfer_plan_to_target(
        global_plan,
        selected_target.placement_id,
    )
    local_bindings = tuple(
        loader_module.project_source_bindings(
            local_plan.source_placements,
            tuple(item[1] for item in source_parts),
        )
    )
    if failure == "source-binding-closure":
        local_bindings = tuple(item[1] for item in source_parts)
    envelope = loader_module._RankLocalTransferEnvelope(
        world_rank=0 if failure == "wrong-rank" else 1,
        parallel_rank=WeightParallelRank(tp=1),
        target_placement_id=target_parts[1][0].placement_id,
        logical_plan=local_plan,
        source_bindings=local_bindings,
    )

    class Coordinator:
        owner_source_session = None

        def clear_owner_source_session(self):
            pass

    class FollowerWorld:
        rank_in_group = 1
        world_size = 2

        def gather_object(self, local, dst=0):
            return None

        def scatter_object(self, values, src=0):
            return envelope

    bind_calls = []
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    with contextlib.ExitStack() as resources:
        with pytest.raises(ValueError):
            loader._prepare_distributed_native_heterogeneous_weight_load(
                model=object(),
                coordinator=Coordinator(),
                world_group=FollowerWorld(),
                transfer_resources=resources,
                target_manifest_builder=_target_session(
                    *target_parts[1],
                    bind_calls,
                ),
                target_model_id=_TARGET_MODEL_ID,
                manifest_revision=_TARGET_ARTIFACT_REVISION,
                local_session_id="target-1",
                transfer_executor=object(),
                phase_seconds={
                    "source_manifest": 0.0,
                    "target_manifest": 0.0,
                    "plan": 0.0,
                    "binding": 0.0,
                },
            )
    assert bind_calls == []


def test_duplicate_target_placement_fails_before_target_bind(monkeypatch) -> None:
    source_parts = _tp2_manifest_parts("source", base_address=0x10000)
    target_parts = _tp2_manifest_parts("target", base_address=0x20000)

    class Coordinator:
        def __init__(self):
            self.owner_source_session = SimpleNamespace(
                source_placements=tuple(item[0] for item in source_parts),
                source_bindings=tuple(item[1] for item in source_parts),
            )

        def clear_owner_source_session(self):
            self.owner_source_session = None

    class RootWorld:
        rank_in_group = 0
        world_size = 2

        def gather_object(self, local, dst=0):
            return [
                local,
                loader_module._TargetPlacementEnvelope(
                    world_rank=1,
                    parallel_rank=local.parallel_rank,
                    placement=local.placement,
                ),
            ]

        def scatter_object(self, values, src=0):
            assert all(value.error for value in values)
            return values[0]

    bind_calls = []
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    with contextlib.ExitStack() as resources:
        with pytest.raises(RuntimeError, match="duplicate target placement"):
            loader._prepare_distributed_native_heterogeneous_weight_load(
                model=object(),
                coordinator=Coordinator(),
                world_group=RootWorld(),
                transfer_resources=resources,
                target_manifest_builder=_target_session(
                    *target_parts[0],
                    bind_calls,
                ),
                target_model_id=_TARGET_MODEL_ID,
                manifest_revision=_TARGET_ARTIFACT_REVISION,
                local_session_id="target-0",
                transfer_executor=object(),
                phase_seconds={
                    "source_manifest": 0.0,
                    "target_manifest": 0.0,
                    "plan": 0.0,
                    "binding": 0.0,
                },
            )
    assert bind_calls == []


@pytest.mark.parametrize("failure", ("target-builder", "root-planner"))
def test_distributed_planning_failure_is_scattered_before_bind(
    monkeypatch,
    failure,
) -> None:
    source_parts = _tp2_manifest_parts("source", base_address=0x10000)
    target_parts = _tp2_manifest_parts("target", base_address=0x20000)

    class Coordinator:
        def __init__(self):
            self.owner_source_session = SimpleNamespace(
                source_placements=tuple(item[0] for item in source_parts),
                source_bindings=tuple(item[1] for item in source_parts),
            )
            self.cleared = 0

        def clear_owner_source_session(self):
            self.owner_source_session = None
            self.cleared += 1

    class RootWorld:
        rank_in_group = 0
        world_size = 2

        def __init__(self):
            self.sequence = []
            self.errors = None

        def gather_object(self, local, dst=0):
            self.sequence.append("gather")
            return [
                local,
                loader_module._TargetPlacementEnvelope(
                    world_rank=1,
                    parallel_rank=WeightParallelRank(tp=1),
                    placement=target_parts[1][0],
                ),
            ]

        def scatter_object(self, values, src=0):
            self.sequence.append("scatter")
            self.errors = tuple(value.error for value in values)
            return values[0]

    if failure == "root-planner":
        monkeypatch.setattr(
            loader_module,
            "plan_weight_transfer",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("planner failed " + "x" * 4096)
            ),
        )

    bind_calls = []
    if failure == "target-builder":

        @contextlib.contextmanager
        def target_builder(**_kwargs):
            raise RuntimeError("target placement failed")
            yield

    else:
        target_builder = _target_session(*target_parts[0], bind_calls)

    world = RootWorld()
    loader = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    with contextlib.ExitStack() as resources:
        with pytest.raises(RuntimeError):
            loader._prepare_distributed_native_heterogeneous_weight_load(
                model=object(),
                coordinator=Coordinator(),
                world_group=world,
                transfer_resources=resources,
                target_manifest_builder=target_builder,
                target_model_id=_TARGET_MODEL_ID,
                manifest_revision=_TARGET_ARTIFACT_REVISION,
                local_session_id="target-0",
                transfer_executor=object(),
                phase_seconds={
                    "source_manifest": 0.0,
                    "target_manifest": 0.0,
                    "plan": 0.0,
                    "binding": 0.0,
                },
            )

    assert world.sequence == ["gather", "scatter"]
    assert world.errors[0] == world.errors[1]
    assert len(world.errors[0]) <= 1024
    assert bind_calls == []


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


def test_explicit_provider_without_manifest_builder_fails_closed(
    monkeypatch,
) -> None:
    provider_calls = []
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
        remote_instance_weight_transfer_provider_factory=(
            lambda *_args, **_kwargs: provider_calls.append("provider")
        ),
        tp_rank=3,
    )
    monkeypatch.setattr(loader_module, "_get_quantization_config", lambda *args: None)
    monkeypatch.setattr(loader_module, "_initialize_model", lambda *args: model)
    monkeypatch.setattr(loader_module, "register_memory_region", lambda *args: ())
    monkeypatch.setattr(
        loader,
        "load_model_from_remote_instance_by_transfer_engine",
        lambda *_args, **_kwargs: pytest.fail(
            "explicit provider configuration must not use the legacy loader"
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="configured weight transfer provider requires a runtime manifest builder",
    ):
        loader.load_model(
            model_config=SimpleNamespace(dtype=torch.float32),
            device_config=SimpleNamespace(device="cpu"),
        )

    assert provider_calls == []


def test_heterogeneous_loader_fails_closed_without_source_manifests(
    monkeypatch,
) -> None:
    class EmptyCoordinator:
        def __init__(self, seed_url, world_group, **_kwargs):
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

    def all_gather_object(self, value, *, phase, execution_context):
        assert phase.startswith("heterogeneous_quarantine.")
        assert isinstance(execution_context, WeightTransferExecutionContext)
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

    def all_gather_object(self, value, *, phase, execution_context):
        assert phase.startswith("heterogeneous_quarantine.")
        assert isinstance(execution_context, WeightTransferExecutionContext)
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


def test_drain_heterogeneous_quarantine_fails_closed_on_collective_timeout(
    monkeypatch,
) -> None:
    events = []
    item, _, resources, coordinator = _recovery_quarantine_item(
        source_transfer_id="transfer-1",
        completion_ticket="ticket-1",
        statuses=("COMPLETION_UNKNOWN",),
        events=events,
    )

    class TimedOutWorld:
        rank_in_group = 0
        world_size = 2

        @staticmethod
        def all_gather_object(value, *, phase, execution_context):
            assert value == ((0, "transfer-1", "ticket-1"),)
            assert phase == "heterogeneous_quarantine.metadata"
            assert isinstance(execution_context, WeightTransferExecutionContext)
            raise TimeoutError("collective deadline exceeded")

    quarantine = [item]
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        quarantine,
    )
    monkeypatch.setattr(loader_module, "get_world_group", TimedOutWorld)

    assert (
        loader_module.drain_heterogeneous_weight_transfer_quarantine(
            max_attempts=1,
            timeout_ms=0,
        )
        is False
    )
    assert quarantine == [item]
    assert resources.closed is False
    assert coordinator.calls == []


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
        lambda seed_url, **_kwargs: session,
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

        def all_gather_object(self, value, *, phase, execution_context):
            assert phase.startswith("heterogeneous_quarantine.")
            assert isinstance(execution_context, WeightTransferExecutionContext)
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
    assert len(events) == 1
    name, kwargs = events[0]
    assert name == "drain"
    assert kwargs["max_attempts"] == 1
    assert kwargs["timeout_ms"] == loader_module._HETEROGENEOUS_UNKNOWN_DRAIN_TIMEOUT_MS
    assert isinstance(kwargs["execution_context"], WeightTransferExecutionContext)


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
        "revision": _TARGET_HF_REVISION,
        "lease_id": "source-runtime-lease",
        "fragments": [],
    }

    class FakeCoordinator:
        def __init__(self, seed_url, world_group, **_kwargs):
            pass

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1",
                manifests=[source_inventory],
                deadline_unix_sec=time.time() + 120,
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
    fake_weight_transfer.MemoryRegistrationLease = SimpleNamespace(
        from_fragment=lambda fragment, **_kwargs: fragment
    )
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
        "revision": _TARGET_HF_REVISION,
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
        def __init__(self, seed_url, world_group, **_kwargs):
            pass

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1",
                manifests=[source_inventory],
                deadline_unix_sec=time.time() + 120,
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
        "revision": _TARGET_HF_REVISION,
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
        def __init__(self, seed_url, world_group, **_kwargs):
            pass

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1",
                manifests=[source_inventory],
                deadline_unix_sec=time.time() + 120,
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
        "revision": _TARGET_HF_REVISION,
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
        def __init__(self, seed_url, world_group, **_kwargs):
            self.failed = False
            state["coordinator"] = self

        def acquire(self):
            return SimpleNamespace(
                transfer_id="transfer-1",
                manifests=[source_inventory],
                lease_timeout_sec=60,
                deadline_unix_sec=time.time() + 120,
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
