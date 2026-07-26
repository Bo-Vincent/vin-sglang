from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
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
)
from sglang.srt.weight_transfer.storage import (
    InMemoryWeightStorageCatalog,
    WeightRevisionState,
)
from sglang.srt.weight_transfer.store_runtime import (
    WeightSnapshotBackend,
    WeightSnapshotLoadSpec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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
    provider = LocalWeightTransferProvider(registry)
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
        catalog=catalog,
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
        yield WeightSnapshotBackend(
            provider=provider,
            catalog=catalog,
            endpoint="target:1",
        )

    def world_barrier():
        head = catalog.get_revision_head("model", "revision")
        assert head is not None
        assert head.state is WeightRevisionState.READY
        events.append("barrier")

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
        weight_snapshot_world_barrier=world_barrier,
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

    loaded = loader_module.get_model_loader(config).load_model(
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
    assert target_resource.attestations == 1
    revision_head = catalog.get_revision_head("model", "revision")
    assert revision_head is not None
    assert revision_head.ref == publication.snapshot.ref
    assert revision_head.state is WeightRevisionState.SERVING
    assert events[-4:] == ["synchronize", "post_load", "eval", "barrier"]
    assert "Loaded weight snapshot:" in caplog.text
    assert "provider=local" in caplog.text
    assert "logical_bytes=8" in caplog.text
    assert "compact_regions=1" in caplog.text


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


def test_world_barrier_failure_does_not_publish_serving(monkeypatch) -> None:
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
        weight_snapshot_world_barrier=lambda: (_ for _ in ()).throw(
            RuntimeError("peer load failed")
        ),
    )
    loader = loader_module.get_model_loader(config)
    serving_calls = []
    monkeypatch.setattr(
        "sglang.srt.weight_transfer.api.mark_weight_snapshot_serving",
        lambda *_args, **_kwargs: serving_calls.append(True),
    )

    with pytest.raises(RuntimeError, match="peer load failed"):
        loader._mark_snapshot_serving(object(), object())

    assert serving_calls == []


@pytest.mark.parametrize("rank,expected_calls", [(0, 1), (1, 0)])
def test_only_world_root_publishes_serving_after_barrier(
    monkeypatch,
    rank,
    expected_calls,
) -> None:
    events = []
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
        weight_snapshot_world_barrier=lambda: events.append("barrier"),
    )
    loader = loader_module.get_model_loader(config)
    monkeypatch.setattr(loader_module, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        loader_module,
        "get_world_group",
        lambda: type("WorldGroup", (), {"rank_in_group": rank})(),
    )
    monkeypatch.setattr(
        "sglang.srt.weight_transfer.api.mark_weight_snapshot_serving",
        lambda *_args, **_kwargs: events.append("serving"),
    )

    loader._mark_snapshot_serving(object(), object())

    assert events == ["barrier", *(["serving"] * expected_calls)]


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

    class CompletionUnknownProvider(LocalWeightTransferProvider):
        name = "local"

        def wait(self, submission):
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
        _, resources, ticket = quarantine[-1]
        assert ticket == "target-ticket"
        assert state == {"backend_closed": False, "target_closed": False}
    finally:
        while len(quarantine) > initial_size:
            _, resources, _ = quarantine.pop()
            resources.close()

    assert state == {"backend_closed": True, "target_closed": True}
