from __future__ import annotations

import contextvars
import logging
import sys
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from types import SimpleNamespace

import msgspec
import pytest
import torch

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.managers.io_struct import (
    WeightSnapshotActivationReqInput,
    WeightSnapshotActivationReqOutput,
    weight_snapshot_activation_request_context,
)
from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.model_runner_components import load_model_utils
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    load_model_with_memory_saver,
    maybe_enable_ipc_weight_cache,
)
from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    compute_weight_placement_id,
)
from sglang.srt.model_loader import loader as loader_module
from sglang.srt.weight_transfer.api import materialize_weight_snapshot
from sglang.srt.weight_transfer.provider import (
    LocalWeightBufferRegistry,
    LocalWeightTransferProvider,
    WeightStorageDestination,
    WeightTransferCompletionUnknownError,
    WeightTransferExecutionContext,
)
from sglang.srt.weight_transfer.storage import (
    InMemoryWeightStorageCatalog,
    WeightRevisionState,
)
from sglang.srt.weight_transfer.store_runtime import (
    WeightSnapshotBackend,
    WeightSnapshotBackendStatus,
    WeightSnapshotLoadSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_load_config_keeps_legacy_weight_snapshot_barrier_keyword() -> None:
    barrier = object()
    config = LoadConfig(weight_snapshot_world_barrier=barrier)

    assert config.weight_snapshot_world_barrier is barrier


def _placement(side: str) -> WeightPlacementManifest:
    tensor = WeightPlacementTensor(
        placement_fragment_id=f"{side}:fragment",
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        dtype="uint8",
        itemsize=1,
        partition_dim=None,
        shard_dims=(),
        layer_id=0,
        expert_id=None,
        layout_fingerprint="logical-contiguous:uint8:v1",
        nbytes=8,
        byte_offset=0,
        rank=WeightParallelRank(),
    )
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id((tensor,)),
        tensors=(tensor,),
    )


def _binding(
    placement: WeightPlacementManifest,
    address: int,
) -> WeightRuntimeBindingManifest:
    tensor = placement.tensors[0]
    return WeightRuntimeBindingManifest(
        model_id=placement.model_id,
        revision=placement.revision,
        placement_id=placement.placement_id,
        instance_id=f"{placement.placement_id}:instance",
        generation=1,
        lease_id=f"{placement.placement_id}:lease",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"{tensor.placement_fragment_id}:runtime",
                address=address,
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cpu",
                is_contiguous=True,
                worker_id=placement.placement_id,
                endpoint=f"{placement.placement_id}:1",
            ),
        ),
    )


def test_weight_snapshot_load_spec_is_strict_and_canonical() -> None:
    spec = WeightSnapshotLoadSpec.from_mapping(
        {
            "model_id": "model",
            "revision": "revision",
            "catalog_path": "/var/lib/sglang/weights/catalog.json",
            "ref": {
                "provider": "mooncake-store",
                "storage_id": "weights/default/model/revision",
                "manifest_key": "weights/default/model/revision/manifest",
                "manifest_digest": f"sha256:{'a' * 64}",
            },
            "endpoint": "10.0.0.1:12345",
            "load_timeout_sec": 900,
            "mooncake_store": {
                "setup": {
                    "local_hostname": "10.0.0.1",
                    "metadata_server": "etcd://metadata",
                    "global_segment_size": "1GB",
                    "local_buffer_size": "1GB",
                    "protocol": "rdma",
                    "rdma_devices": "mlx5_0",
                    "master_server_addr": "10.0.0.2:50051",
                }
            },
        }
    )

    assert spec.model_id == "model"
    assert spec.ref.storage_id == "weights/default/model/revision"
    assert spec.load_timeout_sec == 900
    with pytest.raises(ValueError, match="unknown weight snapshot load options"):
        WeightSnapshotLoadSpec.from_mapping(
            {
                "model_id": "model",
                "revision": "revision",
                "catalog_path": "/catalog",
                "ref": {
                    "provider": "mooncake-store",
                    "storage_id": "weights",
                    "manifest_key": "weights/manifest",
                    "manifest_digest": f"sha256:{'a' * 64}",
                },
                "unknown": True,
            }
        )


@dataclass
class _TargetResource:
    placement: WeightPlacementManifest
    binding: WeightRuntimeBindingManifest
    attestations: int = 0

    @contextmanager
    def bind(self):
        yield self.binding

    def attest_binding(self, binding):
        assert binding == self.binding
        self.attestations += 1


class _BoundedLocalWeightTransferProvider(LocalWeightTransferProvider):
    bounded_execution_contract_version = 1

    def probe(self, request):
        return replace(
            super().probe(request),
            supports_bounded_execution=True,
        )


class _ContextBoundCatalog:
    def __init__(self, catalog):
        self._catalog = catalog
        self.active = True
        self.close_count = 0

    def __getattr__(self, name):
        if not self.active:
            raise RuntimeError("catalog context is closed")
        return getattr(self._catalog, name)

    def close(self):
        self.close_count += 1
        if self.close_count != 1:
            raise AssertionError("catalog context closed more than once")
        self.active = False


class _PendingActivationProbe:
    def __init__(self) -> None:
        self.events = []

    def activate(self) -> None:
        self.events.append("activate")

    def close(self) -> None:
        self.events.append("close")


class _CoordinatedPendingActivationProbe:
    def __init__(self) -> None:
        self.events = []

    def prepare(self, transaction_id, *, deadline_unix_sec=None):
        self.events.append(("prepare", transaction_id, deadline_unix_sec))
        return SimpleNamespace(state=WeightRevisionState.READY)

    def commit(self, transaction_id, *, deadline_unix_sec=None):
        self.events.append(("commit", transaction_id, deadline_unix_sec))
        return SimpleNamespace(state=WeightRevisionState.SERVING)

    def reconcile(self, transaction_id, *, deadline_unix_sec=None):
        self.events.append(("reconcile", transaction_id, deadline_unix_sec))
        return "serving"

    def abort(self, transaction_id, *, deadline_unix_sec=None):
        self.events.append(("abort", transaction_id, deadline_unix_sec))
        return "aborted"

    def close(self) -> None:
        self.events.append(("close", None))


def _pending_activation_with_failing_cleanup():
    resources = ExitStack()

    def fail_cleanup() -> None:
        raise RuntimeError("backend close failed")

    resources.callback(fail_cleanup)
    return loader_module._PendingWeightSnapshotActivation(
        ref=object(),
        catalog=object(),
        resources=resources,
    )


def test_weight_snapshot_model_loader_loads_final_buffers_before_serving(
    monkeypatch,
    caplog,
) -> None:
    source = _placement("source")
    target = _placement("target")
    source_binding = _binding(source, 0x10000)
    target_binding = _binding(target, 0x20000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    registry.register_runtime(0x20000, bytes(8))
    provider = _BoundedLocalWeightTransferProvider(registry)
    storage_catalog = InMemoryWeightStorageCatalog()
    catalog = _ContextBoundCatalog(storage_catalog)
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
        catalog=storage_catalog,
        publication_id="startup-snapshot",
    )
    target_resource = _TargetResource(target, target_binding)
    events = []

    class Model:
        def eval(self):
            events.append("eval")
            return self

    @contextmanager
    def target_builder(**kwargs):
        events.append(("target", kwargs["model_id"], kwargs["revision"]))
        yield target_resource

    @contextmanager
    def backend_factory(_spec):
        try:
            yield WeightSnapshotBackend(
                provider=provider,
                catalog=catalog,
                endpoint="target:1",
            )
        finally:
            catalog.close()

    config = LoadConfig(
        load_format=LoadFormat.WEIGHT_SNAPSHOT,
        model_loader_extra_config={
            "model_id": "model",
            "revision": "revision",
            "catalog_path": "/unused/catalog.json",
            "ref": {
                "provider": publication.snapshot.ref.provider,
                "storage_id": publication.snapshot.ref.storage_id,
                "manifest_key": publication.snapshot.ref.manifest_key,
                "manifest_digest": publication.snapshot.ref.manifest_digest,
            },
        },
        remote_instance_weight_runtime_manifest_builder=target_builder,
        weight_snapshot_backend_factory=backend_factory,
    )
    monkeypatch.setattr(
        loader_module,
        "_get_quantization_config",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        loader_module,
        "_initialize_model",
        lambda *_args, **_kwargs: Model(),
    )
    monkeypatch.setattr(
        loader_module,
        "_post_load_weights",
        lambda _model: events.append("post_load"),
    )
    monkeypatch.setattr(
        loader_module.current_platform,
        "synchronize",
        lambda: events.append("synchronize"),
    )
    caplog.set_level(logging.INFO, logger=loader_module.__name__)

    snapshot_loader = loader_module.get_model_loader(config)
    loaded = snapshot_loader.load_model(
        model_config=type(
            "ModelConfig",
            (),
            {
                "dtype": torch.float32,
                "model_path": "local-model-path",
                "revision": None,
            },
        )(),
        device_config=type("DeviceConfig", (), {"device": "cpu"})(),
    )

    assert isinstance(loaded, Model)
    assert registry.read_runtime(0x20000, 8) == bytes(range(8))
    assert target_resource.attestations >= 1
    revision_head = catalog.get_revision_head("model", "revision")
    assert revision_head is not None
    assert revision_head.ref == publication.snapshot.ref
    assert revision_head.state is WeightRevisionState.READY
    pending = snapshot_loader.take_pending_weight_snapshot_activation()
    assert pending is not None
    assert snapshot_loader.take_pending_weight_snapshot_activation() is None
    runner = SimpleNamespace(
        server_args=SimpleNamespace(load_format="auto"),
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        model=loaded,
        pending_weight_snapshot_activation=pending,
    )
    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: False)
    model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)

    assert not hasattr(loaded, "_sglang_pending_weight_snapshot")
    assert runner.pending_weight_snapshot_activation is pending
    revision_head = catalog.get_revision_head("model", "revision")
    assert revision_head is not None
    assert revision_head.state is WeightRevisionState.SERVING

    model_runner_module.ModelRunner.close_pending_weight_snapshot_activation(runner)

    assert catalog.close_count == 1
    assert runner.pending_weight_snapshot_activation is None
    with pytest.raises(RuntimeError, match="catalog context is closed"):
        catalog.get_revision_head("model", "revision")
    revision_head = storage_catalog.get_revision_head("model", "revision")
    assert revision_head is not None
    assert revision_head.state is WeightRevisionState.SERVING
    assert events[-3:] == ["synchronize", "post_load", "eval"]
    assert "Loaded weight snapshot:" in caplog.text
    assert "provider=local" in caplog.text
    assert "logical_bytes=8" in caplog.text
    assert "compact_regions=1" in caplog.text


def test_weight_snapshot_rejects_unversioned_provider_before_target_session(
    monkeypatch,
) -> None:
    source = _placement("unversioned-source")
    target = _placement("unversioned-target")
    source_binding = _binding(source, 0x20100)
    target_binding = _binding(target, 0x20200)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x20100, bytes(range(8)))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/unversioned",
            object_prefix="weights/unversioned",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="unversioned-snapshot",
    )
    provider = LocalWeightTransferProvider(registry)
    target_events = []

    class Model:
        def eval(self):
            return self

    @contextmanager
    def target_builder(**_kwargs):
        target_events.append("target")
        yield _TargetResource(target, target_binding)

    @contextmanager
    def backend_factory(_spec):
        yield WeightSnapshotBackend(
            provider=provider,
            catalog=catalog,
            endpoint="target:1",
        )

    config = LoadConfig(
        load_format=LoadFormat.WEIGHT_SNAPSHOT,
        model_loader_extra_config={
            "model_id": "model",
            "revision": "revision",
            "catalog_path": "/unused/catalog.json",
            "ref": {
                "provider": publication.snapshot.ref.provider,
                "storage_id": publication.snapshot.ref.storage_id,
                "manifest_key": publication.snapshot.ref.manifest_key,
                "manifest_digest": publication.snapshot.ref.manifest_digest,
            },
        },
        remote_instance_weight_runtime_manifest_builder=target_builder,
        weight_snapshot_backend_factory=backend_factory,
    )
    monkeypatch.setattr(loader_module, "_get_quantization_config", lambda *_args: None)
    monkeypatch.setattr(
        loader_module,
        "_initialize_model",
        lambda *_args, **_kwargs: Model(),
    )

    snapshot_loader = loader_module.get_model_loader(config)
    with pytest.raises(
        RuntimeError,
        match="weight snapshot provider requires bounded execution contract version 1",
    ):
        snapshot_loader.load_model(
            model_config=SimpleNamespace(
                dtype=torch.float32,
                model_path="local-model-path",
                revision=None,
            ),
            device_config=SimpleNamespace(device="cpu"),
        )

    assert target_events == []


def test_weight_snapshot_custom_bounded_provider_receives_execution_context(
    monkeypatch,
) -> None:
    source = _placement("bounded-source")
    target = _placement("bounded-target")
    source_binding = _binding(source, 0x21000)
    target_binding = _binding(target, 0x22000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x21000, bytes(range(8)))
    registry.register_runtime(0x22000, bytes(8))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/bounded",
            object_prefix="weights/bounded",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="bounded-snapshot",
    )
    contexts = []

    class BoundedProvider(_BoundedLocalWeightTransferProvider):
        def prepare(self, request, *, execution_context=None):
            contexts.append(execution_context)
            return super().prepare(request, execution_context=execution_context)

        def wait(self, submission, *, execution_context=None):
            contexts.append(execution_context)
            return super().wait(
                submission,
                execution_context=execution_context,
            )

        def release(self, prepared, receipt, *, execution_context=None):
            contexts.append(execution_context)
            return super().release(
                prepared,
                receipt,
                execution_context=execution_context,
            )

    provider = BoundedProvider(registry)
    target_resource = _TargetResource(target, target_binding)

    class Model:
        def eval(self):
            return self

    @contextmanager
    def target_builder(**_kwargs):
        yield target_resource

    @contextmanager
    def backend_factory(_spec):
        yield WeightSnapshotBackend(
            provider=provider,
            catalog=catalog,
            endpoint="target:1",
        )

    config = LoadConfig(
        load_format=LoadFormat.WEIGHT_SNAPSHOT,
        model_loader_extra_config={
            "model_id": "model",
            "revision": "revision",
            "catalog_path": "/unused/catalog.json",
            "ref": {
                "provider": publication.snapshot.ref.provider,
                "storage_id": publication.snapshot.ref.storage_id,
                "manifest_key": publication.snapshot.ref.manifest_key,
                "manifest_digest": publication.snapshot.ref.manifest_digest,
            },
        },
        remote_instance_weight_runtime_manifest_builder=target_builder,
        weight_snapshot_backend_factory=backend_factory,
    )
    monkeypatch.setattr(loader_module, "_get_quantization_config", lambda *_args: None)
    monkeypatch.setattr(
        loader_module,
        "_initialize_model",
        lambda *_args, **_kwargs: Model(),
    )
    monkeypatch.setattr(loader_module, "_post_load_weights", lambda _model: None)
    monkeypatch.setattr(loader_module.current_platform, "synchronize", lambda: None)

    snapshot_loader = loader_module.get_model_loader(config)
    snapshot_loader.load_model(
        model_config=SimpleNamespace(
            dtype=torch.float32,
            model_path="local-model-path",
            revision=None,
        ),
        device_config=SimpleNamespace(device="cpu"),
    )

    assert len(contexts) == 3
    assert all(
        isinstance(context, WeightTransferExecutionContext) for context in contexts
    )
    assert contexts[0] is contexts[1]
    assert contexts[2] is not contexts[0]
    assert contexts[2].cancel_signal is None
    assert 0 < contexts[2].remaining_seconds() <= 5
    snapshot_loader.take_pending_weight_snapshot_activation().close()


def test_weight_snapshot_load_format_selects_native_loader() -> None:
    config = LoadConfig(
        load_format=LoadFormat.WEIGHT_SNAPSHOT,
        model_loader_extra_config={
            "model_id": "model",
            "revision": "revision",
            "catalog_path": "/catalog.json",
            "ref": {
                "provider": "mooncake-store",
                "storage_id": "weights",
                "manifest_key": "weights/manifest",
                "manifest_digest": f"sha256:{'a' * 64}",
            },
        },
    )

    assert isinstance(
        loader_module.get_model_loader(config),
        loader_module.WeightSnapshotModelLoader,
    )


def test_weight_snapshot_loader_rejects_ipc_weight_cache() -> None:
    load_config = LoadConfig(
        load_format=LoadFormat.WEIGHT_SNAPSHOT,
        weight_cache_mode="client",
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        maybe_enable_ipc_weight_cache(
            load_config=load_config,
            server_args=type("Args", (), {"weight_cache_mode": "client"})(),
            tp_size=1,
            pp_rank=0,
            tp_rank=0,
        )

    assert load_config.load_format is LoadFormat.WEIGHT_SNAPSHOT


def test_pending_weight_snapshot_is_activated_after_startup(
    monkeypatch,
) -> None:
    pending = _PendingActivationProbe()
    runner = SimpleNamespace(
        server_args=SimpleNamespace(load_format="weight_snapshot"),
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        model=SimpleNamespace(),
        pending_weight_snapshot_activation=pending,
    )
    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: False)

    model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)

    assert pending.events == ["activate"]
    assert runner.pending_weight_snapshot_activation is pending

    model_runner_module.ModelRunner.close_pending_weight_snapshot_activation(runner)
    assert pending.events == ["activate", "close"]
    assert runner.pending_weight_snapshot_activation is None


def test_loaded_model_carries_snapshot_activation_handle(monkeypatch) -> None:
    model = SimpleNamespace()
    pending = _PendingActivationProbe()

    class Loader:
        def load_model(self, **_kwargs):
            return model

        def take_pending_weight_snapshot_activation(self):
            return pending

    class MemorySaver:
        @contextmanager
        def region(self, *_args, **_kwargs):
            yield

    monkeypatch.setattr(
        load_model_utils,
        "get_model_loader",
        lambda **_kwargs: Loader(),
    )
    monkeypatch.setattr(
        load_model_utils,
        "monkey_patch_vllm_parallel_state",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(load_model_utils, "_is_npu", False)

    loaded = load_model_with_memory_saver(
        server_args=SimpleNamespace(
            enable_weights_cpu_backup=False,
            enable_draft_weights_cpu_backup=False,
            weight_cache_mode="off",
        ),
        model_config=object(),
        load_config=LoadConfig(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        device="cpu",
        gpu_id=0,
        memory_saver_adapter=MemorySaver(),
        is_draft_worker=False,
    )

    assert loaded.model is model
    assert loaded.pending_weight_snapshot_activation is pending
    assert not hasattr(model, "_sglang_pending_weight_snapshot")


def test_model_runner_initialization_failure_closes_pending_activation(
    monkeypatch,
) -> None:
    pending = _PendingActivationProbe()
    runner = model_runner_module.ModelRunner.__new__(model_runner_module.ModelRunner)
    runner.pending_weight_snapshot_activation = None
    runner.model_config = object()
    runner.server_args = object()
    runner.ps = SimpleNamespace(moe_ep_size=1, moe_ep_rank=0)
    runner.init_memory_saver_adapter = lambda: None
    runner.maybe_init_remote_instance_transfer_engine = lambda: None
    runner.maybe_init_expert_location_metadata = lambda: None
    runner.maybe_init_lplb_solvers = lambda: None
    runner.maybe_init_eplb_manager = lambda: None
    runner.maybe_init_elastic_ep = lambda: None
    runner.init_token_oracle = lambda: None

    def load_model() -> None:
        runner.model = object()
        runner.pending_weight_snapshot_activation = pending

    runner.load_model = load_model
    monkeypatch.setattr(model_runner_module, "create_sampler", lambda: object())

    def fail_prepare_moe_topk(**_) -> None:
        raise RuntimeError("post-load setup failed")

    monkeypatch.setattr(
        model_runner_module,
        "prepare_moe_topk",
        fail_prepare_moe_topk,
    )

    with pytest.raises(RuntimeError, match="post-load setup failed"):
        model_runner_module.ModelRunner.initialize(runner)

    assert pending.events == ["close"]
    assert runner.pending_weight_snapshot_activation is None


def test_pending_activation_promotes_revision_and_closes_resources() -> None:
    source = _placement("activation-source")
    source_binding = _binding(source, 0x50000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x50000, bytes(range(8)))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/activation",
            object_prefix="weights/activation",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="activation-snapshot",
    )
    cleanup = []
    resources = ExitStack()
    resources.callback(lambda: cleanup.append("closed"))
    pending = loader_module._PendingWeightSnapshotActivation(
        ref=publication.snapshot.ref,
        catalog=catalog,
        resources=resources,
    )

    pending.activate()
    head = catalog.get_revision_head("model", "revision")
    assert head is not None
    assert head.state is WeightRevisionState.SERVING

    pending.close()
    assert cleanup == ["closed"]


def test_pending_activation_uses_prepared_head_for_serving_cas() -> None:
    source = _placement("activation-cas-source")
    source_binding = _binding(source, 0x51000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x51000, bytes(range(8)))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/activation-cas",
            object_prefix="weights/activation-cas",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="activation-cas-snapshot",
    )
    pending = loader_module._PendingWeightSnapshotActivation(
        ref=publication.snapshot.ref,
        catalog=catalog,
        resources=ExitStack(),
    )

    prepared = pending.prepare("activation-transaction")
    committed = pending.commit("activation-transaction")

    assert prepared.state is WeightRevisionState.READY
    assert committed.state is WeightRevisionState.SERVING
    assert committed.generation == prepared.generation + 1
    assert pending.reconcile("activation-transaction") == "serving"


def test_pending_activation_abort_closes_ready_resources() -> None:
    source = _placement("activation-abort-source")
    source_binding = _binding(source, 0x52000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x52000, bytes(range(8)))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/activation-abort",
            object_prefix="weights/activation-abort",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="activation-abort-snapshot",
    )
    cleanup = []
    resources = ExitStack()
    resources.callback(lambda: cleanup.append("closed"))
    pending = loader_module._PendingWeightSnapshotActivation(
        ref=publication.snapshot.ref,
        catalog=catalog,
        resources=resources,
    )
    pending.prepare("activation-transaction")

    assert pending.abort("activation-transaction") == "aborted"
    assert cleanup == ["closed"]
    assert (
        catalog.get_revision_head("model", "revision").state
        is WeightRevisionState.READY
    )


def test_pending_activation_abort_quarantines_serving_resources(
    monkeypatch,
) -> None:
    source = _placement("activation-quarantine-source")
    source_binding = _binding(source, 0x53000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x53000, bytes(range(8)))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/activation-quarantine",
            object_prefix="weights/activation-quarantine",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="activation-quarantine-snapshot",
    )
    cleanup = []
    resources = ExitStack()
    resources.callback(lambda: cleanup.append("closed"))
    pending = loader_module._PendingWeightSnapshotActivation(
        ref=publication.snapshot.ref,
        catalog=catalog,
        resources=resources,
    )
    quarantine = []
    monkeypatch.setattr(
        loader_module,
        "_WEIGHT_SNAPSHOT_ACTIVATION_QUARANTINE",
        quarantine,
    )
    pending.prepare("activation-transaction")
    serving = pending.commit("activation-transaction")
    assert pending.prepare("activation-transaction") == serving

    assert pending.abort("activation-transaction") == "quarantined"
    pending.close()

    assert cleanup == []
    assert quarantine == [pending]
    assert (
        catalog.get_revision_head("model", "revision").state is WeightRevisionState.IDLE
    )


def test_pending_activation_does_not_abort_preexisting_serving_revision(
    monkeypatch,
) -> None:
    source = _placement("activation-preexisting-serving-source")
    source_binding = _binding(source, 0x53500)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x53500, bytes(range(8)))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/activation-preexisting-serving",
            object_prefix="weights/activation-preexisting-serving",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="activation-preexisting-serving-snapshot",
    )
    ready = catalog.get_revision_head("model", "revision")
    serving = catalog.compare_and_set_revision(
        model_id="model",
        revision="revision",
        expected=ready,
        new_ref=publication.snapshot.ref,
        new_state=WeightRevisionState.SERVING,
    )
    cleanup = []
    resources = ExitStack()
    resources.callback(lambda: cleanup.append("closed"))
    pending = loader_module._PendingWeightSnapshotActivation(
        ref=publication.snapshot.ref,
        catalog=catalog,
        resources=resources,
    )
    quarantine = []
    monkeypatch.setattr(
        loader_module,
        "_WEIGHT_SNAPSHOT_ACTIVATION_QUARANTINE",
        quarantine,
    )

    assert pending.prepare("activation-transaction") == serving
    assert pending.abort("activation-transaction") == "aborted"

    assert cleanup == ["closed"]
    assert quarantine == []
    assert catalog.get_revision_head("model", "revision") == serving


def test_pending_activation_does_not_own_external_serving_after_prepare(
    monkeypatch,
) -> None:
    source = _placement("activation-external-serving-source")
    source_binding = _binding(source, 0x53580)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x53580, bytes(range(8)))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/activation-external-serving",
            object_prefix="weights/activation-external-serving",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="activation-external-serving-snapshot",
    )
    cleanup = []
    resources = ExitStack()
    resources.callback(lambda: cleanup.append("closed"))
    pending = loader_module._PendingWeightSnapshotActivation(
        ref=publication.snapshot.ref,
        catalog=catalog,
        resources=resources,
    )
    quarantine = []
    monkeypatch.setattr(
        loader_module,
        "_WEIGHT_SNAPSHOT_ACTIVATION_QUARANTINE",
        quarantine,
    )

    ready = pending.prepare("activation-transaction")
    serving = catalog.compare_and_set_revision(
        model_id="model",
        revision="revision",
        expected=ready,
        new_ref=publication.snapshot.ref,
        new_state=WeightRevisionState.SERVING,
    )
    assert pending.commit("activation-transaction") == serving
    assert pending.abort("activation-transaction") == "aborted"

    assert cleanup == ["closed"]
    assert quarantine == []
    assert catalog.get_revision_head("model", "revision") == serving


def test_pending_activation_rebinds_catalog_deadline_and_rejects_expired_commit() -> (
    None
):
    source = _placement("activation-deadline-source")
    source_binding = _binding(source, 0x53600)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x53600, bytes(range(8)))
    base_catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/activation-deadline",
            object_prefix="weights/activation-deadline",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=base_catalog,
        publication_id="activation-deadline-snapshot",
    )
    contexts = []
    mutations = []

    class CatalogView:
        def __init__(self, context):
            self.context = context

        def get_snapshot(self, ref):
            return base_catalog.get_snapshot(ref)

        def get_revision_head(self, model_id, revision):
            return base_catalog.get_revision_head(model_id, revision)

        def compare_and_set_revision(self, **kwargs):
            mutations.append(kwargs)
            return base_catalog.compare_and_set_revision(**kwargs)

    class DeadlineCatalog:
        def with_execution_context(self, execution_context):
            contexts.append(execution_context)
            return CatalogView(execution_context)

        def __getattr__(self, name):
            return getattr(base_catalog, name)

    pending = loader_module._PendingWeightSnapshotActivation(
        ref=publication.snapshot.ref,
        catalog=DeadlineCatalog(),
        resources=ExitStack(),
    )
    prepare_deadline = time.time() + 30
    pending.prepare(
        "activation-transaction",
        deadline_unix_sec=prepare_deadline,
    )

    with pytest.raises(TimeoutError, match="deadline"):
        pending.commit(
            "activation-transaction",
            deadline_unix_sec=time.time() - 1,
        )

    assert [context.deadline_unix_sec for context in contexts] == [prepare_deadline]
    assert all(
        isinstance(context, WeightTransferExecutionContext) for context in contexts
    )
    assert mutations == []
    assert (
        base_catalog.get_revision_head("model", "revision").state
        is WeightRevisionState.READY
    )


def test_activation_request_context_drives_model_runner_phases(monkeypatch) -> None:
    pending = _CoordinatedPendingActivationProbe()
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        pending_weight_snapshot_activation=pending,
    )
    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: False)

    deadline_unix_sec = time.time() + 30

    def tokenizer_encode(phase, request_id):
        def encode():
            return msgspec.msgpack.encode(
                WeightSnapshotActivationReqInput(
                    action="activate",
                    phase=phase,
                    transaction_id="activation-transaction",
                    request_id=request_id,
                    deadline_unix_sec=deadline_unix_sec,
                )
            )

        return contextvars.Context().run(encode)

    def run_phase(payload):
        def scheduler_round_trip():
            request = msgspec.msgpack.decode(
                payload,
                type=WeightSnapshotActivationReqInput,
            )
            with weight_snapshot_activation_request_context(request):
                model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)
                return WeightSnapshotActivationReqOutput(
                    action="activate",
                    success=True,
                    message="Success.",
                )

        return contextvars.Context().run(scheduler_round_trip)

    prepared = run_phase(
        tokenizer_encode(
            "prepare",
            "prepare-request",
        )
    )
    committed = run_phase(
        tokenizer_encode(
            "commit",
            "commit-request",
        )
    )

    assert prepared.phase == "prepare"
    assert prepared.transaction_id == "activation-transaction"
    assert prepared.request_id == "prepare-request"
    assert prepared.responder_id
    assert prepared.state == "prepared"
    assert committed.phase == "commit"
    assert committed.request_id == "commit-request"
    assert committed.responder_id == prepared.responder_id
    assert committed.state == "serving"
    assert pending.events == [
        ("prepare", "activation-transaction", deadline_unix_sec),
        ("commit", "activation-transaction", deadline_unix_sec),
    ]


def test_structured_commit_requires_nonzero_rank_to_confirm_serving(
    monkeypatch,
) -> None:
    pending = _CoordinatedPendingActivationProbe()
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        pending_weight_snapshot_activation=pending,
    )
    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_runner_module.dist, "get_rank", lambda: 1)
    deadline_unix_sec = time.time() + 30
    request = WeightSnapshotActivationReqInput(
        action="activate",
        phase="commit",
        transaction_id="activation-transaction",
        request_id="commit-request",
        deadline_unix_sec=deadline_unix_sec,
    )

    def run_commit():
        with weight_snapshot_activation_request_context(request):
            model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)
            return WeightSnapshotActivationReqOutput(
                action="activate",
                success=True,
                message="Success.",
            )

    result = contextvars.Context().run(run_commit)

    assert result.state == "serving"
    assert pending.events == [("commit", "activation-transaction", deadline_unix_sec)]


def test_pending_activation_cannot_refresh_loader_deadline() -> None:
    activation = loader_module._PendingWeightSnapshotActivation(
        ref=object(),
        catalog=object(),
        resources=ExitStack(),
        deadline_unix_sec=time.time() - 1,
    )

    with pytest.raises(TimeoutError, match="deadline"):
        activation._catalog_for_deadline(time.time() + 30)


def test_expired_activation_request_does_not_mutate_catalog() -> None:
    source = _placement("activation-expired-source")
    source_binding = _binding(source, 0x53700)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x53700, bytes(range(8)))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/activation-expired",
            object_prefix="weights/activation-expired",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="activation-expired-snapshot",
    )
    pending = loader_module._PendingWeightSnapshotActivation(
        ref=publication.snapshot.ref,
        catalog=catalog,
        resources=ExitStack(),
    )
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        pending_weight_snapshot_activation=pending,
    )

    def run_expired_prepare():
        request = WeightSnapshotActivationReqInput(
            action="activate",
            phase="prepare",
            transaction_id="expired-activation",
            request_id="expired-prepare",
            deadline_unix_sec=time.time() - 1,
        )
        with weight_snapshot_activation_request_context(request):
            model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)

    with pytest.raises(TimeoutError, match="deadline expired"):
        contextvars.Context().run(run_expired_prepare)

    assert (
        catalog.get_revision_head("model", "revision").state
        is WeightRevisionState.READY
    )
    assert pending._transaction_id is None


def test_structured_close_is_local_and_idempotent(monkeypatch) -> None:
    pending = _CoordinatedPendingActivationProbe()
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        pending_weight_snapshot_activation=pending,
    )

    def fail_collective(*_args, **_kwargs):
        raise AssertionError("structured close must not use a collective")

    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        model_runner_module.dist,
        "all_gather_object",
        fail_collective,
    )

    def close(request_id):
        def run():
            request = WeightSnapshotActivationReqInput(
                action="close",
                phase="close",
                transaction_id="close-transaction",
                request_id=request_id,
                deadline_unix_sec=time.time() + 30,
            )
            with weight_snapshot_activation_request_context(request):
                model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)
                return WeightSnapshotActivationReqOutput(
                    action="close",
                    success=True,
                    message="Success.",
                )

        return contextvars.Context().run(run)

    first = close("close-request-1")
    second = close("close-request-2")

    assert pending.events == [("close", None)]
    assert runner.pending_weight_snapshot_activation is None
    assert first.phase == second.phase == "close"
    assert first.state == second.state == "closed"


def test_expired_structured_close_does_not_touch_owner(monkeypatch) -> None:
    pending = _CoordinatedPendingActivationProbe()
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        pending_weight_snapshot_activation=pending,
    )

    def fail_collective(*_args, **_kwargs):
        raise AssertionError("expired structured close must not use a collective")

    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        model_runner_module.dist,
        "all_gather_object",
        fail_collective,
    )

    def run():
        request = WeightSnapshotActivationReqInput(
            action="close",
            phase="close",
            transaction_id="expired-close-transaction",
            request_id="expired-close-request",
            deadline_unix_sec=time.time() - 1,
        )
        with weight_snapshot_activation_request_context(request):
            model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)

    with pytest.raises(TimeoutError, match="deadline expired"):
        contextvars.Context().run(run)

    assert pending.events == []
    assert runner.pending_weight_snapshot_activation is pending


def test_snapshot_activation_failure_closes_loader_resources(monkeypatch) -> None:
    catalog = _ContextBoundCatalog(InMemoryWeightStorageCatalog())
    resources = ExitStack()

    @contextmanager
    def backend_context():
        try:
            yield
        finally:
            catalog.close()

    resources.enter_context(backend_context())
    pending = loader_module._PendingWeightSnapshotActivation(
        ref=object(),
        catalog=catalog,
        resources=resources,
    )
    runner = SimpleNamespace(
        server_args=SimpleNamespace(load_format="weight_snapshot"),
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        model=SimpleNamespace(),
        pending_weight_snapshot_activation=pending,
    )

    def fail_activation(_ref, *, catalog):
        assert catalog.active
        raise RuntimeError("catalog publish failed")

    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        "sglang.srt.weight_transfer.api.mark_weight_snapshot_serving",
        fail_activation,
    )

    with pytest.raises(RuntimeError, match="catalog publish failed"):
        model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)

    assert catalog.close_count == 1
    assert not catalog.active
    assert runner.pending_weight_snapshot_activation is None


def test_snapshot_activation_keeps_serving_after_cleanup_failure(
    monkeypatch,
    caplog,
) -> None:
    pending = _pending_activation_with_failing_cleanup()
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        model=SimpleNamespace(),
        pending_weight_snapshot_activation=pending,
    )
    serving_calls = []
    quarantine = []
    monkeypatch.setattr(
        loader_module,
        "_WEIGHT_SNAPSHOT_CLEANUP_QUARANTINE",
        quarantine,
    )
    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        "sglang.srt.weight_transfer.api.mark_weight_snapshot_serving",
        lambda actual_ref, *, catalog: serving_calls.append((actual_ref, catalog)),
    )

    model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)
    model_runner_module.ModelRunner.close_pending_weight_snapshot_activation(runner)

    assert serving_calls == [(pending.ref, pending.catalog)]
    assert quarantine == [pending]
    assert "retaining the owner for process lifetime" in caplog.text
    assert runner.pending_weight_snapshot_activation is None


def test_snapshot_activation_retries_pending_backend_cleanup() -> None:
    resources = ExitStack()

    def fail_cleanup() -> None:
        raise RuntimeError("backend close failed")

    resources.callback(fail_cleanup)

    class Backend:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self, *, timeout_ms):
            assert timeout_ms == loader_module._WEIGHT_SNAPSHOT_CLEANUP_TIMEOUT_MS
            self.close_calls += 1
            if self.close_calls == 1:
                return WeightSnapshotBackendStatus(
                    terminal=False,
                    pending_tickets=("store/close",),
                )
            return WeightSnapshotBackendStatus(terminal=True, closed=True)

    backend = Backend()
    pending = loader_module._PendingWeightSnapshotActivation(
        ref=object(),
        catalog=object(),
        resources=resources,
        backend=backend,
    )
    runner = SimpleNamespace(pending_weight_snapshot_activation=pending)

    with pytest.raises(RuntimeError, match="cleanup remains pending"):
        model_runner_module.ModelRunner.close_pending_weight_snapshot_activation(runner)
    assert runner.pending_weight_snapshot_activation is pending
    assert pending._state == "cleanup_pending"

    model_runner_module.ModelRunner.close_pending_weight_snapshot_activation(runner)

    assert backend.close_calls == 2
    assert pending._closed is True
    assert runner.pending_weight_snapshot_activation is None


def test_snapshot_activation_preserves_publish_error_when_cleanup_fails(
    monkeypatch,
) -> None:
    pending = _pending_activation_with_failing_cleanup()
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        model=SimpleNamespace(),
        pending_weight_snapshot_activation=pending,
    )
    quarantine = []
    monkeypatch.setattr(
        loader_module,
        "_WEIGHT_SNAPSHOT_CLEANUP_QUARANTINE",
        quarantine,
    )

    def fail_activation(_ref, *, catalog):
        assert catalog is pending.catalog
        raise RuntimeError("catalog publish failed")

    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        "sglang.srt.weight_transfer.api.mark_weight_snapshot_serving",
        fail_activation,
    )

    with pytest.raises(RuntimeError, match="catalog publish failed"):
        model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)

    assert quarantine == [pending]
    assert runner.pending_weight_snapshot_activation is None


def test_nonzero_rank_waits_for_readiness_without_activating(monkeypatch) -> None:
    pending = _PendingActivationProbe()
    runner = SimpleNamespace(
        server_args=SimpleNamespace(load_format="weight_snapshot"),
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        model=SimpleNamespace(),
        pending_weight_snapshot_activation=pending,
    )

    def all_gather_object(outputs, _value):
        outputs[:] = [
            {"ready": True, "error": None},
            {"ready": True, "error": None},
        ]

    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_runner_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(model_runner_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        model_runner_module.dist,
        "all_gather_object",
        all_gather_object,
    )

    model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)

    assert pending.events == []
    assert runner.pending_weight_snapshot_activation is pending


def test_snapshot_activation_closes_backend_when_peer_is_not_ready(
    monkeypatch,
) -> None:
    pending = _PendingActivationProbe()
    runner = SimpleNamespace(
        server_args=SimpleNamespace(load_format="weight_snapshot"),
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        model=SimpleNamespace(),
        pending_weight_snapshot_activation=pending,
    )

    def all_gather_object(outputs, _value):
        outputs[:] = [
            {"ready": True, "error": None},
            {"ready": False, "error": "activation handle is missing"},
        ]

    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_runner_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(model_runner_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        model_runner_module.dist,
        "all_gather_object",
        all_gather_object,
    )

    with pytest.raises(RuntimeError, match="activation handle is missing"):
        model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)

    assert pending.events == ["close"]
    assert runner.pending_weight_snapshot_activation is None


def test_snapshot_activation_has_no_collective_after_rank_zero_commit(
    monkeypatch,
) -> None:
    pending = _PendingActivationProbe()
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        model=SimpleNamespace(),
        pending_weight_snapshot_activation=pending,
    )

    def all_gather_object(outputs, _value):
        outputs[:] = [
            {"ready": True, "error": None},
            {"ready": True, "error": None},
        ]

    monkeypatch.setattr(model_runner_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(model_runner_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(model_runner_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        model_runner_module.dist,
        "all_gather_object",
        all_gather_object,
    )

    model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)

    assert pending.events == ["activate"]
    assert runner.pending_weight_snapshot_activation is pending


def test_non_snapshot_loader_does_not_consume_pending_activation() -> None:
    pending = _PendingActivationProbe()
    runner = SimpleNamespace(
        server_args=SimpleNamespace(load_format="auto"),
        load_config=SimpleNamespace(load_format=LoadFormat.AUTO),
        model=SimpleNamespace(),
        pending_weight_snapshot_activation=pending,
    )

    model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)

    assert pending.events == []
    assert runner.pending_weight_snapshot_activation is pending


def test_weight_snapshot_loader_retains_all_resources_when_completion_is_unknown(
    monkeypatch,
) -> None:
    source = _placement("source")
    target = _placement("target")
    source_binding = _binding(source, 0x30000)
    target_binding = _binding(target, 0x40000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x30000, bytes(range(8)))
    registry.register_runtime(0x40000, bytes(8))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="unknown-startup-snapshot",
    )
    target_resource = _TargetResource(target, target_binding)
    state = {"backend_closed": False, "target_closed": False}

    class CompletionUnknownProvider(_BoundedLocalWeightTransferProvider):
        name = "local"

        def wait(self, submission, *, execution_context=None):
            assert isinstance(execution_context, WeightTransferExecutionContext)
            raise WeightTransferCompletionUnknownError(
                "target completion is unknown",
                provider=self.name,
                phase="wait",
                operation_id=submission.request.operation_id,
                completion_ticket="target-ticket",
            )

    class Model:
        def eval(self):
            return self

    @contextmanager
    def target_builder(**_kwargs):
        try:
            yield target_resource
        finally:
            state["target_closed"] = True

    @contextmanager
    def backend_factory(_spec):
        try:
            yield WeightSnapshotBackend(
                provider=CompletionUnknownProvider(registry),
                catalog=catalog,
                endpoint="target:1",
            )
        finally:
            state["backend_closed"] = True

    config = LoadConfig(
        load_format=LoadFormat.WEIGHT_SNAPSHOT,
        model_loader_extra_config={
            "model_id": "model",
            "revision": "revision",
            "catalog_path": "/unused/catalog.json",
            "ref": {
                "provider": publication.snapshot.ref.provider,
                "storage_id": publication.snapshot.ref.storage_id,
                "manifest_key": publication.snapshot.ref.manifest_key,
                "manifest_digest": publication.snapshot.ref.manifest_digest,
            },
        },
        remote_instance_weight_runtime_manifest_builder=target_builder,
        weight_snapshot_backend_factory=backend_factory,
    )
    monkeypatch.setattr(loader_module, "_get_quantization_config", lambda *_args: None)
    monkeypatch.setattr(
        loader_module,
        "_initialize_model",
        lambda *_args, **_kwargs: Model(),
    )
    quarantine = loader_module._WEIGHT_SNAPSHOT_UNKNOWN_LOAD_QUARANTINE
    initial_size = len(quarantine)
    try:
        with pytest.raises(WeightTransferCompletionUnknownError):
            loader_module.get_model_loader(config).load_model(
                model_config=type(
                    "ModelConfig",
                    (),
                    {
                        "dtype": torch.float32,
                        "model_path": "local-model-path",
                        "revision": None,
                    },
                )(),
                device_config=type("DeviceConfig", (), {"device": "cpu"})(),
            )

        assert len(quarantine) == initial_size + 1
        quarantined_model, resources, ticket = quarantine[-1]
        assert ticket == "target-ticket"
        assert not hasattr(
            quarantined_model,
            "_sglang_pending_weight_snapshot",
        )
        assert state == {"backend_closed": False, "target_closed": False}
    finally:
        while len(quarantine) > initial_size:
            _, resources, _ = quarantine.pop()
            resources.close()

    assert state == {"backend_closed": True, "target_closed": True}


if __name__ == "__main__":
    pytest_args = ["-x" if argument == "-f" else argument for argument in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *pytest_args]))
