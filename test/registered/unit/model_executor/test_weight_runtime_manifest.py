from __future__ import annotations

import ast
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest

import sglang.srt.model_executor.weight_runtime_manifest as weight_runtime_manifest_module
from sglang.srt.model_executor.weight_runtime_manifest import (
    LogicalTensorView,
    WeightManifestError,
    WeightParallelTopology,
    WeightPlacementManifest,
    WeightRuntimeManifestManager,
    WeightSnapshotCoordinator,
    WeightTargetPlacementManifest,
    compose_weight_runtime_manifest,
    create_sglang_weight_runtime_manifest_manager,
    create_weight_runtime_manifest_manager,
)
from sglang.srt.model_executor.model_runner_components.weight_update_coordination import (
    coordinated_weight_update,
)
from sglang.srt.model_executor.weight_semantics.qwen3_5 import (
    Qwen35WeightSemanticsAdapter,
)
from sglang.srt.model_executor.weight_semantics.qwen3_next import (
    Qwen3NextWeightSemanticsAdapter,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class FakeStorage:
    def __init__(self, address: int) -> None:
        self._address = address

    def data_ptr(self) -> int:
        return self._address


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        address: int = 0x10000,
        dtype: str = "torch.bfloat16",
        itemsize: int = 2,
        device: str = "cpu",
        contiguous: bool = True,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self.is_sparse = False
        self._address = address
        self._itemsize = itemsize
        self._contiguous = contiguous

    def data_ptr(self) -> int:
        return self._address

    def element_size(self) -> int:
        return self._itemsize

    def numel(self) -> int:
        return prod(self.shape)

    def is_contiguous(self) -> bool:
        return self._contiguous

    def stride(self):
        stride = []
        value = 1
        for extent in reversed(self.shape):
            stride.append(value)
            value *= extent
        return tuple(reversed(stride))

    def storage_offset(self) -> int:
        return 0

    def untyped_storage(self) -> FakeStorage:
        return FakeStorage(self._address)


class FakeModel:
    def __init__(self, parameters) -> None:
        self.parameters = parameters

    def named_parameters(self, *, remove_duplicate: bool):
        assert remove_duplicate is False
        return iter(self.parameters)


class CountingFakeModel(FakeModel):
    def __init__(self, parameters) -> None:
        super().__init__(parameters)
        self.physical_collections = 0

    def named_parameters(self, *, remove_duplicate: bool):
        self.physical_collections += 1
        return super().named_parameters(remove_duplicate=remove_duplicate)


class CountingSnapshotCoordinator(WeightSnapshotCoordinator):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_acquisitions = 0

    def acquire_snapshot(self, **kwargs):
        self.snapshot_acquisitions += 1
        return super().acquire_snapshot(**kwargs)


class FakeClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeMoEModel(FakeModel):
    def __init__(self, parameters, *, w13_parameter, up_first: bool) -> None:
        super().__init__(parameters)
        self._moe_module = SimpleNamespace(
            w13_weight=w13_parameter,
            use_flashinfer_trtllm_moe=up_first,
        )

    def modules(self):
        return iter((self, self._moe_module))


class FakeFp8RuntimeModule:
    def __init__(
        self,
        *,
        weight=None,
        weight_scale_inv=None,
        weight_scale=None,
        w13_weight=None,
        w13_weight_scale_inv=None,
        w2_weight=None,
        w2_weight_scale_inv=None,
        block_quant: bool = True,
        weight_block_size=(128, 128),
        is_checkpoint_fp8_serialized: bool = True,
        activation_scheme: str = "dynamic",
        use_mxfp8: bool = False,
        load_up_proj_weight_first: bool = False,
    ) -> None:
        block_size = None if weight_block_size is None else list(weight_block_size)
        quant_config = SimpleNamespace(
            activation_scheme=activation_scheme,
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            use_mxfp8=use_mxfp8,
            weight_block_size=block_size,
        )
        self.block_quant = block_quant
        self.weight_block_size = block_size
        self.quant_method = SimpleNamespace(
            block_quant=block_quant,
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            load_up_proj_weight_first=load_up_proj_weight_first,
            quant_config=quant_config,
            use_mxfp8=use_mxfp8,
            weight_block_size=block_size,
        )
        for name, parameter in (
            ("weight", weight),
            ("weight_scale_inv", weight_scale_inv),
            ("weight_scale", weight_scale),
            ("w13_weight", w13_weight),
            ("w13_weight_scale_inv", w13_weight_scale_inv),
            ("w2_weight", w2_weight),
            ("w2_weight_scale_inv", w2_weight_scale_inv),
        ):
            if parameter is not None:
                setattr(self, name, parameter)


class FakeFp8RuntimeModel(FakeModel):
    def __init__(self, parameters, *, runtime_modules) -> None:
        super().__init__(parameters)
        self._runtime_modules = tuple(runtime_modules)

    def modules(self):
        return iter((self, *self._runtime_modules))


class ReplicatedAdapter:
    def describe_parameter(self, *, names, parameter, topology):
        del topology
        shape = tuple(parameter.shape)
        return (
            LogicalTensorView(
                tensor_id=names[0],
                global_shape=shape,
                global_offset=(0,) * len(shape),
                local_shape=shape,
                partition_dim=None,
                byte_offset=0,
                layer_id=None,
                expert_id=None,
                layout_fingerprint="test:replicated:v1",
            ),
        )


class DummyWeightUpdater:
    def __init__(self, coordinator: WeightSnapshotCoordinator) -> None:
        self.begin_weight_update = coordinator.begin_update
        self.finish_weight_update = coordinator.finish_update
        self.calls = 0

    @coordinated_weight_update
    def update(self, result, *, raise_error: bool = False):
        self.calls += 1
        if raise_error:
            raise RuntimeError("update failed")
        return result


class ProductionShapeWeightUpdater:
    def __init__(self, coordinator: WeightSnapshotCoordinator) -> None:
        self.begin_weight_update = coordinator.begin_update
        self.finish_weight_update = coordinator.finish_update

    @coordinated_weight_update
    def update_weights_from_disk(
        self,
        model_path,
        load_format,
        weight_name_filter=None,
        recapture_cuda_graph=False,
    ):
        del model_path, load_format, weight_name_filter, recapture_cuda_graph
        return True, "disk update complete"

    @coordinated_weight_update
    def update_weights_from_distributed(self, *args, **kwargs):
        del args, kwargs
        return True, "distributed update complete"

    @coordinated_weight_update
    def update_weights_from_tensor(self, *args, **kwargs):
        del args, kwargs
        return True, "tensor update complete"

    @coordinated_weight_update
    def update_weights_from_ipc(self, *args, **kwargs):
        del args, kwargs
        return True, "ipc update complete"


def topology(**overrides) -> WeightParallelTopology:
    values = dict(
        dp_rank=0,
        dp_size=1,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        ep_rank=0,
        ep_size=1,
        moe_tp_rank=0,
        moe_tp_size=1,
        attention_tp_rank=0,
        attention_tp_size=1,
    )
    values.update(overrides)
    return WeightParallelTopology(**values)


def qwen_config(**overrides):
    values = dict(
        model_type="qwen3_5_text",
        hidden_size=8,
        intermediate_size=8,
        moe_intermediate_size=8,
        vocab_size=32,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=2,
        attn_output_gate=False,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        num_experts=8,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def qwen_vision_config(**overrides):
    values = dict(
        hidden_size=8,
        intermediate_size=16,
        num_heads=4,
        num_position_embeddings=32,
        out_hidden_size=8,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def qwen_multimodal_config(**overrides):
    values = dict(
        model_type="qwen3_5",
        text_config=qwen_config(),
        vision_config=qwen_vision_config(),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("partition_dim", "shard_dims", "global_offset", "local_shape", "message"),
    [
        (True, None, (0, 0), (2, 4), "partition"),
        (None, (0, 0), (0, 0), (2, 4), "shard"),
        (None, (1, 0), (0, 0), (2, 4), "shard"),
        (None, (2,), (0, 0), (2, 4), "shard"),
        (0, (1,), (0, 0), (2, 4), "shard"),
        (0, (0,), (0, 1), (2, 3), "non-shard"),
    ],
)
def test_runtime_manifest_rejects_invalid_shard_box_semantics(
    partition_dim,
    shard_dims,
    global_offset,
    local_shape,
    message,
) -> None:
    class StaticAdapter:
        def describe_parameter(self, *, names, parameter, topology):
            del parameter, topology
            return (
                LogicalTensorView(
                    tensor_id=names[0],
                    global_shape=(4, 4),
                    global_offset=global_offset,
                    local_shape=local_shape,
                    partition_dim=partition_dim,
                    byte_offset=0,
                    layer_id=None,
                    expert_id=None,
                    layout_fingerprint="test:shard-box:v2",
                    shard_dims=shard_dims,
                ),
            )

    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", FakeTensor((4, 4)))]),
        adapter=StaticAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )

    with pytest.raises(WeightManifestError, match=message):
        manager.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_snapshot_keeps_aliases_and_rotates_generation_when_pointer_changes() -> None:
    """A replaced Parameter must invalidate plans even when its names stay stable."""
    tensor = FakeTensor((4, 2), address=0x10000)
    model = FakeModel([("z.weight", tensor), ("a.weight", tensor)])
    manager = WeightRuntimeManifestManager(
        model=model,
        adapter=ReplicatedAdapter(),
        topology=topology(
            dp_rank=2,
            dp_size=3,
            tp_rank=3,
            tp_size=4,
            pp_rank=1,
            pp_size=2,
            ep_rank=4,
            ep_size=5,
        ),
        allowed_devices=("cpu",),
    )

    first = manager.snapshot(
        model_id="qwen3.5-0.8b",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )
    manager.release(first.lease_id)
    update_token = manager.coordinator.begin_update()
    tensor._address = 0x20000
    manager.coordinator.finish_update(update_token, success=True)
    manager.coordinator.commit_revision()
    second = manager.snapshot(
        model_id="qwen3.5-0.8b",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )

    assert first.tensors[0].runtime_name == "a.weight"
    assert first.tensors[0].aliases == ("a.weight", "z.weight")
    assert first.tensors[0].address == 0x10000
    assert first.tensors[0].nbytes == 16
    assert first.tensors[0].rank.dp == 2
    assert first.tensors[0].rank.tp == 3
    assert first.tensors[0].rank.pp == 1
    assert first.tensors[0].rank.ep == 4
    assert first.generation == 1
    assert second.generation == 2
    assert second.tensors[0].address == 0x20000
    manager.release(second.lease_id)


def test_target_placement_has_stable_semantics_without_runtime_location() -> None:
    tensor = FakeTensor((4, 2), address=0x10000)
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", tensor)]),
        adapter=ReplicatedAdapter(),
        topology=topology(tp_rank=1, tp_size=2),
        allowed_devices=("cpu",),
    )

    placement = manager.placement(model_id="model", revision="revision")
    payload = msgspec.to_builtins(placement)

    assert placement.format_version == 2
    assert placement.tensors[0].placement_fragment_id
    assert placement.tensors[0].rank.tp == 1
    assert "address" not in payload["tensors"][0]
    assert "worker_id" not in payload["tensors"][0]
    assert "endpoint" not in payload["tensors"][0]
    assert "generation" not in payload
    assert "lease_id" not in payload

    token = manager.coordinator.begin_update()
    tensor._address = 0x20000
    manager.coordinator.finish_update(token, success=True)
    manager.coordinator.commit_revision()
    same_layout = manager.placement(model_id="model", revision="revision")

    assert same_layout == placement


def test_source_and_target_share_one_placement_manifest_contract() -> None:
    assert WeightTargetPlacementManifest is WeightPlacementManifest

    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", FakeTensor((4, 2)))]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )
    parts = manager.snapshot_parts(
        model_id="model",
        revision="revision",
        instance_id="instance",
        worker_id="worker",
        endpoint="endpoint",
    )

    assert isinstance(parts.placement, WeightPlacementManifest)
    assert parts.binding.placement_id == parts.placement.placement_id
    manager.release(parts.binding.lease_id)


def test_local_mooncake_split_capability_requires_the_complete_loader_api() -> None:
    required_apis = {
        "RuntimeBindingManifest",
        "SourcePlacementManifest",
        "TargetPlacementManifest",
        "bind_logical_transfer_plan",
        "bind_runtime_manifest",
        "placement_manifest_from_runtime_manifest",
        "plan_placement_transfer_to_local_target",
        "runtime_binding_from_runtime_manifest",
    }
    api_only_module = SimpleNamespace(**{name: lambda: None for name in required_apis})
    requested_capabilities = []

    def supports(capability):
        requested_capabilities.append(capability)
        return capability == "placement_binding_v1"

    def broken_supports(capability):
        del capability
        raise RuntimeError("mixed Mooncake wheel")

    complete_module = SimpleNamespace(
        supports_weight_transfer_capability=supports,
        **{name: lambda: None for name in required_apis},
    )

    assert (
        weight_runtime_manifest_module.local_mooncake_supports_placement_binding(
            SimpleNamespace()
        )
        is False
    )
    assert (
        weight_runtime_manifest_module.local_mooncake_supports_placement_binding(
            api_only_module
        )
        is False
    )
    assert (
        weight_runtime_manifest_module.local_mooncake_supports_placement_binding(
            SimpleNamespace(
                supports_weight_transfer_capability=lambda capability: False,
                **{name: lambda: None for name in required_apis},
            )
        )
        is False
    )
    assert (
        weight_runtime_manifest_module.local_mooncake_supports_placement_binding(
            SimpleNamespace(
                supports_weight_transfer_capability=True,
                **{name: lambda: None for name in required_apis},
            )
        )
        is False
    )
    assert (
        weight_runtime_manifest_module.local_mooncake_supports_placement_binding(
            SimpleNamespace(
                supports_weight_transfer_capability=broken_supports,
                **{name: lambda: None for name in required_apis},
            )
        )
        is False
    )
    assert (
        weight_runtime_manifest_module.local_mooncake_supports_placement_binding(
            complete_module
        )
        is True
    )
    assert requested_capabilities == ["placement_binding_v1"]
    for missing in required_apis:
        incomplete_module = SimpleNamespace(
            supports_weight_transfer_capability=supports,
            **{
                name: (None if name == missing else lambda: None)
                for name in required_apis
            },
        )
        assert (
            weight_runtime_manifest_module.local_mooncake_supports_placement_binding(
                incomplete_module
            )
            is False
        )


def test_snapshot_parts_use_one_lease_and_one_physical_collection() -> None:
    model = CountingFakeModel([("weight", FakeTensor((4, 2)))])
    coordinator = CountingSnapshotCoordinator()
    manager = WeightRuntimeManifestManager(
        model=model,
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
        coordinator=coordinator,
    )

    parts = manager.snapshot_parts(
        model_id="model",
        revision="revision",
        instance_id="instance",
        worker_id="worker",
        endpoint="endpoint",
    )

    assert coordinator.snapshot_acquisitions == 1
    assert model.physical_collections == 1
    assert parts.placement.revision == parts.binding.revision
    manager.release(parts.binding.lease_id)


def test_snapshot_parts_qualify_revision_from_the_acquired_generation() -> None:
    coordinator = WeightSnapshotCoordinator()
    token = coordinator.begin_update()
    coordinator.finish_update(token, success=True)
    coordinator.commit_revision()
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", FakeTensor((4, 2)))]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
        coordinator=coordinator,
    )

    parts = manager.snapshot_parts(
        model_id="model",
        revision="main",
        instance_id="instance",
        worker_id="worker",
        endpoint="endpoint",
        bind_revision_to_generation=True,
    )

    assert parts.binding.generation == 2
    assert parts.placement.revision == "main@generation-2"
    assert parts.binding.revision == parts.placement.revision
    manager.release(parts.binding.lease_id)


def test_placement_id_depends_only_on_stable_layout_semantics() -> None:
    tensor = FakeTensor((4, 2), address=0x10000)
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", tensor)]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )

    first = manager.snapshot_parts(
        model_id="model-a",
        revision="revision-a",
        instance_id="instance",
        worker_id="worker",
        endpoint="endpoint",
    )
    manager.release(first.binding.lease_id)

    token = manager.coordinator.begin_update()
    tensor._address = 0x20000
    manager.coordinator.finish_update(token, success=True)
    manager.coordinator.commit_revision()

    second = manager.snapshot_parts(
        model_id="model-b",
        revision="revision-b",
        instance_id="instance",
        worker_id="worker",
        endpoint="endpoint",
    )

    assert first.binding.generation != second.binding.generation
    assert first.placement.revision != second.placement.revision
    assert first.placement.placement_id == second.placement.placement_id
    manager.release(second.binding.lease_id)


def test_runtime_binding_refreshes_addresses_and_composes_legacy_snapshot() -> None:
    tensor = FakeTensor((4, 2), address=0x10000)
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", tensor)]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )
    placement = manager.placement(model_id="model", revision="revision")

    binding = manager.snapshot_binding(
        placement=placement,
        instance_id="instance",
        worker_id="worker",
        endpoint="endpoint",
    )
    composed = compose_weight_runtime_manifest(placement, binding)

    assert binding.fragments[0].address == 0x10000
    assert binding.fragments[0].placement_fragment_id == (
        placement.tensors[0].placement_fragment_id
    )
    assert composed.tensors[0].address == 0x10000
    assert composed.tensors[0].fragment_id == binding.fragments[0].fragment_id
    assert composed.lease_id == binding.lease_id
    with pytest.raises(WeightManifestError, match="snapshot lease is active"):
        manager.coordinator.begin_update()

    manager.release(binding.lease_id)
    legacy = manager.snapshot(
        model_id="model",
        revision="revision",
        instance_id="instance",
        worker_id="worker",
        endpoint="endpoint",
    )
    composed_payload = msgspec.to_builtins(composed)
    legacy_payload = msgspec.to_builtins(legacy)
    composed_payload["lease_id"] = "<lease>"
    legacy_payload["lease_id"] = "<lease>"
    assert composed_payload == legacy_payload
    manager.release(legacy.lease_id)


def test_runtime_binding_rejects_a_placement_from_another_manager() -> None:
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", FakeTensor((4, 2)))]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )
    other = WeightRuntimeManifestManager(
        model=FakeModel([("other", FakeTensor((4, 2)))]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )
    placement = other.placement(model_id="model", revision="revision")

    with pytest.raises(WeightManifestError, match="placement"):
        manager.snapshot_binding(
            placement=placement,
            instance_id="instance",
            worker_id="worker",
            endpoint="endpoint",
        )


def test_runtime_binding_rejects_layout_change_but_releases_snapshot_lease() -> None:
    tensor = FakeTensor((4, 2), address=0x10000)
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", tensor)]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )
    placement = manager.placement(model_id="model", revision="revision")
    token = manager.coordinator.begin_update()
    tensor.shape = (2, 4)
    manager.coordinator.finish_update(token, success=True)
    manager.commit_revision()

    with pytest.raises(WeightManifestError, match="layout"):
        manager.snapshot_binding(
            placement=placement,
            instance_id="instance",
            worker_id="worker",
            endpoint="endpoint",
        )

    next_token = manager.coordinator.begin_update()
    manager.coordinator.finish_update(next_token, success=False)


def test_snapshot_lease_blocks_updates_until_explicit_release() -> None:
    coordinator = WeightSnapshotCoordinator()
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", FakeTensor((2, 2)))]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
        coordinator=coordinator,
    )

    snapshot = manager.snapshot(
        model_id="model",
        revision="revision",
        instance_id="instance",
        worker_id="worker",
        endpoint="worker:12345",
    )

    with pytest.raises(WeightManifestError, match="snapshot lease is active"):
        coordinator.begin_update()

    manager.release(snapshot.lease_id)
    token = coordinator.begin_update()
    coordinator.finish_update(token, success=True)
    coordinator.commit_revision()

    next_snapshot = manager.snapshot(
        model_id="model",
        revision="revision-2",
        instance_id="instance",
        worker_id="worker",
        endpoint="worker:12345",
    )
    assert next_snapshot.generation == snapshot.generation + 1
    manager.release(next_snapshot.lease_id)


def test_expired_snapshot_lease_blocks_mutation_until_explicit_release() -> None:
    clock = FakeClock(100.0)
    coordinator = WeightSnapshotCoordinator(clock=clock)

    lease_id, generation = coordinator.acquire_snapshot(lease_timeout_sec=30)
    assert generation == 1

    clock.advance(30)
    assert coordinator.has_snapshot(lease_id)
    with pytest.raises(WeightManifestError, match="snapshot lease is active"):
        coordinator.begin_update()

    clock.advance(3600)
    assert coordinator.has_snapshot(lease_id)
    with pytest.raises(WeightManifestError, match="snapshot lease is active"):
        coordinator.begin_update()

    coordinator.release_snapshot(lease_id)
    token = coordinator.begin_update()
    coordinator.finish_update(token, success=True)
    coordinator.commit_revision()


def test_expired_snapshot_lease_cannot_be_silently_revived() -> None:
    clock = FakeClock(100.0)
    coordinator = WeightSnapshotCoordinator(clock=clock)
    lease_id, _ = coordinator.acquire_snapshot(lease_timeout_sec=30)

    clock.advance(30)
    with pytest.raises(WeightManifestError, match="expired.*explicit release"):
        coordinator.renew_snapshot(lease_id, lease_timeout_sec=30)
    assert coordinator.has_snapshot(lease_id)
    with pytest.raises(WeightManifestError, match="snapshot lease is active"):
        coordinator.begin_update()

    coordinator.release_snapshot(lease_id)
    token = coordinator.begin_update()
    coordinator.finish_update(token, success=True)
    coordinator.commit_revision()


def test_snapshot_lease_status_keeps_expired_lease_until_explicit_release() -> None:
    clock = FakeClock(100.0)
    coordinator = WeightSnapshotCoordinator(clock=clock)
    lease_id, generation = coordinator.acquire_snapshot(lease_timeout_sec=30)

    clock.advance(30)
    statuses = coordinator.list_snapshot_leases()

    assert len(statuses) == 1
    assert statuses[0].lease_id == lease_id
    assert statuses[0].generation == generation
    assert statuses[0].deadline == 130.0
    assert statuses[0].expired is True
    assert coordinator.has_snapshot(lease_id)
    with pytest.raises(WeightManifestError, match="snapshot lease is active"):
        coordinator.begin_update()

    coordinator.release_snapshot(lease_id)
    assert coordinator.list_snapshot_leases() == ()


def test_online_update_coordination_executes_generation_and_failure_contract() -> None:
    coordinator = WeightSnapshotCoordinator()
    updater = DummyWeightUpdater(coordinator)
    lease_id, _ = coordinator.acquire_snapshot()

    rejected = updater.update((True, "ok"))
    assert rejected[0] is False
    assert "snapshot lease is active" in rejected[1]
    assert updater.calls == 0

    coordinator.release_snapshot(lease_id)
    assert updater.update((True, "ok")) == (True, "ok")
    assert coordinator.generation == 2
    coordinator.commit_revision()

    assert updater.update((False, "load failed")) == (False, "load failed")
    assert coordinator.generation == 3
    with pytest.raises(WeightManifestError, match="last weight update failed"):
        coordinator.acquire_snapshot()

    assert updater.update((True, "incremental")) == (True, "incremental")
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.commit_revision()
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.acquire_snapshot()

    restored = coordinator.begin_update(full_restore=True)
    coordinator.finish_update(restored, success=True)
    coordinator.commit_revision()
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 5
    coordinator.release_snapshot(lease_id)


def test_successful_updates_require_explicit_revision_commit() -> None:
    coordinator = WeightSnapshotCoordinator()
    updater = DummyWeightUpdater(coordinator)

    assert updater.update((True, "bucket complete")) == (True, "bucket complete")
    with pytest.raises(WeightManifestError, match="revision commit"):
        coordinator.acquire_snapshot()

    assert coordinator.commit_revision() == 2
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 2
    coordinator.release_snapshot(lease_id)


def test_new_generation_supersedes_uncommitted_success_without_stale_commit() -> None:
    coordinator = WeightSnapshotCoordinator()
    first = coordinator.begin_update()
    first_generation = coordinator.finish_update(first, success=True)

    assert first_generation == 2
    assert coordinator.pending_revision_generation() == first_generation
    second = coordinator.begin_update()
    second_generation = coordinator.finish_update(second, success=True)

    assert second_generation == 3
    assert coordinator.pending_revision_generation() == second_generation
    with pytest.raises(WeightManifestError, match="generation does not match"):
        coordinator.commit_revision(expected_generation=first_generation)
    assert (
        coordinator.commit_revision(expected_generation=second_generation)
        == second_generation
    )
    assert coordinator.pending_revision_generation() is None


def test_commit_and_global_poison_reject_stale_generations() -> None:
    coordinator = WeightSnapshotCoordinator()
    first = coordinator.begin_update()
    first_generation = coordinator.finish_update(first, success=True)
    coordinator.commit_revision(expected_generation=first_generation)

    second = coordinator.begin_update()
    second_generation = coordinator.finish_update(second, success=True)

    with pytest.raises(WeightManifestError, match="generation does not match"):
        coordinator.commit_revision(expected_generation=first_generation)
    with pytest.raises(WeightManifestError, match="generation does not match"):
        coordinator.poison_global_update_failure(expected_generation=first_generation)

    coordinator.poison_global_update_failure(expected_generation=second_generation)
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.commit_revision(expected_generation=second_generation)


def test_completion_fence_precedes_generation_publish_and_snapshot() -> None:
    generations_during_fence = []
    coordinator = None

    def completion_fence() -> None:
        generations_during_fence.append(coordinator.generation)
        with pytest.raises(WeightManifestError, match="update is in progress"):
            coordinator.acquire_snapshot()
        with pytest.raises(WeightManifestError, match="update is in progress"):
            coordinator.commit_revision()

    coordinator = WeightSnapshotCoordinator(completion_fence=completion_fence)
    token = coordinator.begin_update()
    coordinator.finish_update(token, success=True)

    assert generations_during_fence == [1, 1]
    assert coordinator.generation == 2
    with pytest.raises(WeightManifestError, match="revision commit"):
        coordinator.acquire_snapshot()


def test_coordinated_update_fences_before_and_after_weight_mutation() -> None:
    events = []
    coordinator = WeightSnapshotCoordinator(
        completion_fence=lambda: events.append("fence")
    )

    class OrderedUpdater:
        def __init__(self) -> None:
            self.begin_weight_update = coordinator.begin_update
            self.finish_weight_update = coordinator.finish_update

        @coordinated_weight_update
        def update(self):
            events.append("mutation")
            return True, "updated"

    assert OrderedUpdater().update() == (True, "updated")
    assert events == ["fence", "mutation", "fence"]


def test_pre_mutation_fence_failure_cancels_the_update_reservation() -> None:
    def completion_fence() -> None:
        raise RuntimeError("old reader failed to drain")

    coordinator = WeightSnapshotCoordinator(completion_fence=completion_fence)

    with pytest.raises(RuntimeError, match="old reader failed to drain"):
        coordinator.begin_update()

    assert coordinator.generation == 1
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 1
    coordinator.release_snapshot(lease_id)


def test_completion_fence_failure_poisons_the_unpublished_update() -> None:
    fence_calls = 0

    def completion_fence() -> None:
        nonlocal fence_calls
        fence_calls += 1
        if fence_calls == 2:
            raise RuntimeError("device work failed")

    coordinator = WeightSnapshotCoordinator(completion_fence=completion_fence)
    token = coordinator.begin_update()

    with pytest.raises(RuntimeError, match="device work failed"):
        coordinator.finish_update(token, success=True)

    assert coordinator.generation == 2
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_update_and_fence_failure_preserve_both_errors() -> None:
    fence_calls = 0

    def completion_fence() -> None:
        nonlocal fence_calls
        fence_calls += 1
        if fence_calls == 2:
            raise RuntimeError("device fence failed")

    coordinator = WeightSnapshotCoordinator(completion_fence=completion_fence)
    updater = DummyWeightUpdater(coordinator)

    with pytest.raises(RuntimeError, match="update failed") as error:
        updater.update((True, "unused"), raise_error=True)

    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "device fence failed"
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_online_update_exception_poisons_runtime_snapshots() -> None:
    coordinator = WeightSnapshotCoordinator()
    updater = DummyWeightUpdater(coordinator)

    with pytest.raises(RuntimeError, match="update failed"):
        updater.update((True, "unused"), raise_error=True)

    assert coordinator.generation == 2
    with pytest.raises(WeightManifestError, match="last weight update failed"):
        coordinator.acquire_snapshot()

    assert updater.update((True, "incremental")) == (True, "incremental")
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.commit_revision()
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_failed_weight_update_poison_snapshot_until_a_full_update_succeeds() -> None:
    coordinator = WeightSnapshotCoordinator()
    failed = coordinator.begin_update()
    coordinator.finish_update(failed, success=False)

    with pytest.raises(WeightManifestError, match="last weight update failed"):
        coordinator.acquire_snapshot()

    incremental = coordinator.begin_update()
    coordinator.finish_update(incremental, success=True)
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.commit_revision()

    restored = coordinator.begin_update(full_restore=True)
    coordinator.finish_update(restored, success=True)
    with pytest.raises(WeightManifestError, match="revision commit"):
        coordinator.acquire_snapshot()
    coordinator.commit_revision()
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 4
    coordinator.release_snapshot(lease_id)


def test_full_restore_decorator_explicitly_selects_restore_mode() -> None:
    coordinator = WeightSnapshotCoordinator()
    failed = coordinator.begin_update()
    coordinator.finish_update(failed, success=False)

    class FullRestoreUpdater:
        def __init__(self) -> None:
            self.begin_weight_update = coordinator.begin_update
            self.finish_weight_update = coordinator.finish_update

        @coordinated_weight_update(full_restore=True)
        def restore(self):
            return True, "restored"

    assert FullRestoreUpdater().restore() == (True, "restored")
    with pytest.raises(WeightManifestError, match="revision commit"):
        coordinator.acquire_snapshot()
    coordinator.commit_revision()
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 3
    coordinator.release_snapshot(lease_id)


def test_complete_disk_checkpoint_is_a_production_full_restore_path() -> None:
    coordinator = WeightSnapshotCoordinator()
    failed = coordinator.begin_update()
    coordinator.finish_update(failed, success=False)
    updater = ProductionShapeWeightUpdater(coordinator)

    assert updater.update_weights_from_disk("checkpoint", "auto") == (
        True,
        "disk update complete",
    )

    with pytest.raises(WeightManifestError, match="revision commit"):
        coordinator.acquire_snapshot()
    coordinator.commit_revision()
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 3
    coordinator.release_snapshot(lease_id)


@pytest.mark.parametrize(
    ("method_name", "args", "kwargs"),
    [
        (
            "update_weights_from_disk",
            ("checkpoint", "auto"),
            {"weight_name_filter": lambda name: name == "partial.weight"},
        ),
        ("update_weights_from_distributed", (), {}),
        ("update_weights_from_tensor", (), {}),
        ("update_weights_from_ipc", (), {}),
    ],
)
def test_partial_or_unproven_updates_cannot_clear_poison(
    method_name, args, kwargs
) -> None:
    coordinator = WeightSnapshotCoordinator()
    failed = coordinator.begin_update()
    coordinator.finish_update(failed, success=False)
    updater = ProductionShapeWeightUpdater(coordinator)

    result = getattr(updater, method_name)(*args, **kwargs)

    assert result[0] is True
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.commit_revision()
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_global_update_failure_hook_reasserts_sticky_poison() -> None:
    coordinator = WeightSnapshotCoordinator()
    locally_successful = coordinator.begin_update()
    generation = coordinator.finish_update(locally_successful, success=True)

    coordinator.poison_global_update_failure(expected_generation=generation)

    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.commit_revision()
    restored = coordinator.begin_update(full_restore=True)
    coordinator.finish_update(restored, success=True)
    coordinator.commit_revision()
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 3
    coordinator.release_snapshot(lease_id)


def test_full_restore_mode_rejects_non_boolean_values() -> None:
    coordinator = WeightSnapshotCoordinator()

    with pytest.raises(TypeError, match="full_restore must be a boolean"):
        coordinator.begin_update(full_restore="false")
    with pytest.raises(TypeError, match="full_restore must be a boolean"):
        coordinated_weight_update(full_restore="false")

    token = coordinator.begin_update()
    coordinator.cancel_update(token)


def test_finish_update_rejects_non_boolean_success_without_publishing() -> None:
    coordinator = WeightSnapshotCoordinator()
    failed = coordinator.begin_update()
    coordinator.finish_update(failed, success=False)
    restore = coordinator.begin_update(full_restore=True)

    with pytest.raises(TypeError, match="success must be a boolean"):
        coordinator.finish_update(restore, success="false")

    assert coordinator.generation == 2
    coordinator.cancel_update(restore)
    with pytest.raises(WeightManifestError, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_snapshot_releases_lease_when_composition_is_cancelled(monkeypatch) -> None:
    class SnapshotCancelled(BaseException):
        pass

    captured_lease_ids = []

    def cancel_composition(placement, binding):
        del placement
        captured_lease_ids.append(binding.lease_id)
        raise SnapshotCancelled

    monkeypatch.setattr(
        weight_runtime_manifest_module,
        "compose_weight_runtime_manifest",
        cancel_composition,
    )
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", FakeTensor((2, 2)))]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )

    with pytest.raises(SnapshotCancelled):
        manager.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )

    assert len(captured_lease_ids) == 1
    assert not manager.has_lease(captured_lease_ids[0])


def test_uncoordinated_pointer_replacement_fails_closed() -> None:
    tensor = FakeTensor((2, 2), address=0x10000)
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", tensor)]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )
    first = manager.snapshot(
        model_id="model",
        revision="revision",
        instance_id="instance",
        worker_id="worker",
        endpoint="worker:12345",
    )
    manager.release(first.lease_id)
    tensor._address = 0x20000

    with pytest.raises(WeightManifestError, match="outside the update coordinator"):
        manager.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_runtime_fragment_ids_are_unique_across_workers_in_one_instance() -> None:
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", FakeTensor((2, 2), address=0x10000))]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )

    first = manager.snapshot(
        model_id="qwen3.5-0.8b",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )
    second = manager.snapshot(
        model_id="qwen3.5-0.8b",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-1",
        endpoint="worker-1:12345",
    )

    assert first.generation == second.generation
    assert first.tensors[0].fragment_id != second.tensors[0].fragment_id
    manager.release(first.lease_id)
    manager.release(second.lease_id)


def test_snapshot_rejects_noncontiguous_parameter() -> None:
    """A single pointer and byte count cannot describe a strided view safely."""
    manager = WeightRuntimeManifestManager(
        model=FakeModel([("weight", FakeTensor((2, 2), contiguous=False))]),
        adapter=ReplicatedAdapter(),
        topology=topology(),
        allowed_devices=("cpu",),
    )

    with pytest.raises(WeightManifestError, match="non-contiguous"):
        manager.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_qwen_qkv_views_handle_replicated_kv_heads() -> None:
    """KV heads replicate when attention TP exceeds KV heads; Q still shards."""
    adapter = Qwen35WeightSemanticsAdapter(config=qwen_config())
    parameter = FakeTensor((8, 8), itemsize=2)

    views = adapter.describe_parameter(
        names=("layers.0.self_attn.qkv_proj.weight",),
        parameter=parameter,
        topology=topology(
            tp_rank=1,
            tp_size=2,
            attention_tp_rank=1,
            attention_tp_size=2,
        ),
    )

    assert [view.tensor_id for view in views] == [
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
    ]
    assert [view.global_offset for view in views] == [(4, 0), (0, 0), (0, 0)]
    assert [view.local_shape for view in views] == [(4, 8), (2, 8), (2, 8)]
    assert [view.byte_offset for view in views] == [0, 64, 96]


def test_qwen_qkv_replica_ranks_follow_contiguous_tp_groups() -> None:
    adapter = Qwen35WeightSemanticsAdapter(config=qwen_config(num_key_value_heads=2))

    views = adapter.describe_parameter(
        names=("layers.0.self_attn.qkv_proj.weight",),
        parameter=FakeTensor((6, 8), itemsize=2),
        topology=topology(
            tp_rank=2,
            tp_size=4,
            attention_tp_rank=2,
            attention_tp_size=4,
        ),
    )

    assert [view.global_offset for view in views] == [(4, 0), (2, 0), (2, 0)]


def test_qwen_tied_embedding_and_lm_head_publish_both_logical_views() -> None:
    """Tied storage must retain both canonical vocabulary tensor identities."""
    parameter = FakeTensor((16, 8), address=0x20000, itemsize=2)
    manager = WeightRuntimeManifestManager(
        model=FakeModel(
            [
                ("model.embed_tokens.weight", parameter),
                ("lm_head.weight", parameter),
            ]
        ),
        adapter=Qwen35WeightSemanticsAdapter(config=qwen_config()),
        topology=topology(tp_rank=1, tp_size=2),
        allowed_devices=("cpu",),
    )

    manifest = manager.snapshot(
        model_id="qwen3.5-0.8b",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )

    assert [tensor.tensor_id for tensor in manifest.tensors] == [
        "embed_tokens.weight",
        "lm_head.weight",
    ]
    assert {tensor.aliases for tensor in manifest.tensors} == {
        ("lm_head.weight", "model.embed_tokens.weight")
    }
    assert {
        (
            tensor.address,
            tensor.nbytes,
            tensor.byte_offset,
            tensor.storage_offset,
        )
        for tensor in manifest.tensors
    } == {(0x20000, 256, 0, 0)}
    assert {tensor.global_offset for tensor in manifest.tensors} == {(16, 0)}
    assert {tensor.local_shape for tensor in manifest.tensors} == {(16, 8)}


def test_qwen_gate_up_and_down_use_opposite_tp_axes() -> None:
    """Column-parallel gate/up splits rows while row-parallel down splits columns."""
    adapter = Qwen35WeightSemanticsAdapter(config=qwen_config())
    parallel = topology(tp_rank=1, tp_size=2)

    gate_up = adapter.describe_parameter(
        names=("layers.1.mlp.gate_up_proj.weight",),
        parameter=FakeTensor((8, 8), itemsize=2),
        topology=parallel,
    )
    down = adapter.describe_parameter(
        names=("layers.1.mlp.down_proj.weight",),
        parameter=FakeTensor((8, 4), itemsize=2),
        topology=parallel,
    )

    assert [view.tensor_id for view in gate_up] == [
        "layers.1.mlp.gate_proj.weight",
        "layers.1.mlp.up_proj.weight",
    ]
    assert [view.global_offset for view in gate_up] == [(4, 0), (4, 0)]
    assert [view.byte_offset for view in gate_up] == [0, 64]
    assert down[0].global_shape == (8, 8)
    assert down[0].global_offset == (0, 4)
    assert down[0].partition_dim == 1


def test_qwen_rejects_stacked_shared_expert_gate_up_layout() -> None:
    """A stacked shared-expert fusion cannot be exported as flat gate/up views."""
    adapter = Qwen35WeightSemanticsAdapter(
        config=qwen_config(
            model_type="qwen3_5_moe_text",
            shared_expert_intermediate_size=8,
        )
    )

    with pytest.raises(WeightManifestError, match="packed tensor shape mismatch"):
        adapter.describe_parameter(
            names=("layers.2.mlp.shared_expert.gate_up_proj.weight",),
            parameter=FakeTensor((2, 4, 8), itemsize=2),
            topology=topology(tp_rank=1, tp_size=2),
        )


def test_qwen_moe_views_split_ep_ownership_and_expert_tp() -> None:
    """Each fused expert allocation maps into one family logical coordinate."""
    adapter = Qwen35WeightSemanticsAdapter(
        config=qwen_config(model_type="qwen3_5_moe_text")
    )
    parameter = FakeTensor((2, 8, 8), itemsize=2)

    views = adapter.describe_parameter(
        names=("layers.2.mlp.experts.w13_weight",),
        parameter=parameter,
        topology=topology(
            ep_rank=1,
            ep_size=4,
            moe_tp_rank=1,
            moe_tp_size=2,
        ),
    )

    assert [view.expert_id for view in views] == [None, None, None, None]
    assert [view.tensor_id for view in views] == [
        "layers.2.mlp.experts.gate_proj.weight",
        "layers.2.mlp.experts.up_proj.weight",
        "layers.2.mlp.experts.gate_proj.weight",
        "layers.2.mlp.experts.up_proj.weight",
    ]
    assert {view.global_shape for view in views} == {(8, 8, 8)}
    assert [view.global_offset for view in views] == [
        (2, 4, 0),
        (2, 4, 0),
        (3, 4, 0),
        (3, 4, 0),
    ]
    assert {view.local_shape for view in views} == {(1, 4, 8)}
    assert {view.partition_dim for view in views} == {None}
    assert {view.shard_dims for view in views} == {(0, 1)}
    assert [view.byte_offset for view in views] == [0, 64, 128, 192]

    manager = WeightRuntimeManifestManager(
        model=FakeModel([("layers.2.mlp.experts.w13_weight", parameter)]),
        adapter=adapter,
        topology=topology(
            ep_rank=1,
            ep_size=4,
            moe_tp_rank=1,
            moe_tp_size=2,
        ),
        allowed_devices=("cpu",),
    )
    manifest = manager.snapshot(
        model_id="qwen3.5-moe",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )

    assert manifest.format_version == 2
    up_expert_2 = next(
        tensor
        for tensor in manifest.tensors
        if tensor.tensor_id == "layers.2.mlp.experts.up_proj.weight"
        and tensor.global_offset[0] == 2
    )
    assert up_expert_2.stride == (32, 8, 1)
    assert up_expert_2.storage_offset == 32


def test_qwen_moe_down_uses_expert_and_input_logical_axes() -> None:
    adapter = Qwen35WeightSemanticsAdapter(
        config=qwen_config(model_type="qwen3_5_moe_text")
    )

    views = adapter.describe_parameter(
        names=("layers.2.mlp.experts.w2_weight",),
        parameter=FakeTensor((2, 8, 4), itemsize=2),
        topology=topology(
            ep_rank=1,
            ep_size=4,
            moe_tp_rank=1,
            moe_tp_size=2,
        ),
    )

    assert [view.tensor_id for view in views] == [
        "layers.2.mlp.experts.down_proj.weight",
        "layers.2.mlp.experts.down_proj.weight",
    ]
    assert [view.global_shape for view in views] == [(8, 8, 8), (8, 8, 8)]
    assert [view.global_offset for view in views] == [(2, 0, 4), (3, 0, 4)]
    assert [view.local_shape for view in views] == [(1, 8, 4), (1, 8, 4)]
    assert [view.shard_dims for view in views] == [(0, 2), (0, 2)]
    assert [view.partition_dim for view in views] == [None, None]
    assert [view.expert_id for view in views] == [None, None]
    assert [view.byte_offset for view in views] == [0, 64]


def test_qwen_moe_factory_reads_w31_component_order_from_runtime_module() -> None:
    parameter = FakeTensor((2, 8, 8), address=0x40000, itemsize=2)
    model = FakeMoEModel(
        [("layers.2.mlp.experts.w13_weight", parameter)],
        w13_parameter=parameter,
        up_first=True,
    )
    manager = create_weight_runtime_manifest_manager(
        model=model,
        config=qwen_config(model_type="qwen3_5_moe_text"),
        topology=topology(ep_rank=1, ep_size=4, moe_tp_rank=1, moe_tp_size=2),
        allowed_devices=("cpu",),
    )

    manifest = manager.snapshot(
        model_id="qwen3.5-moe",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )
    addresses = {
        (tensor.tensor_id, tensor.global_offset[0]): tensor.address
        for tensor in manifest.tensors
    }

    assert addresses[("layers.2.mlp.experts.up_proj.weight", 2)] == 0x40000
    assert addresses[("layers.2.mlp.experts.gate_proj.weight", 2)] == 0x40000 + 64
    manager.release(manifest.lease_id)


def test_qwen_dense_block_fp8_manifest_publishes_weight_and_scale_views() -> None:
    weight = FakeTensor(
        (256, 128),
        address=0x40000,
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (2, 1),
        address=0x50000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(weight=weight, weight_scale_inv=scale)
    model = FakeFp8RuntimeModel(
        [
            ("layers.1.mlp.gate_up_proj.weight", weight),
            ("layers.1.mlp.gate_up_proj.weight_scale_inv", scale),
        ],
        runtime_modules=(module,),
    )
    manager = create_weight_runtime_manifest_manager(
        model=model,
        config=qwen_config(hidden_size=128, intermediate_size=256),
        topology=topology(tp_rank=1, tp_size=2),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
    )

    manifest = manager.snapshot(
        model_id="qwen3.5-fp8",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )
    tensors = {tensor.tensor_id: tensor for tensor in manifest.tensors}

    assert set(tensors) == {
        "layers.1.mlp.gate_proj.weight",
        "layers.1.mlp.gate_proj.weight_scale_inv",
        "layers.1.mlp.up_proj.weight",
        "layers.1.mlp.up_proj.weight_scale_inv",
    }
    for component in ("gate_proj", "up_proj"):
        weight_view = tensors[f"layers.1.mlp.{component}.weight"]
        scale_view = tensors[f"layers.1.mlp.{component}.weight_scale_inv"]
        assert weight_view.dtype == "float8_e4m3fn"
        assert weight_view.itemsize == 1
        assert weight_view.global_shape == (256, 128)
        assert weight_view.global_offset == (128, 0)
        assert weight_view.local_shape == (128, 128)
        assert scale_view.dtype == "float32"
        assert scale_view.itemsize == 4
        assert scale_view.global_shape == (2, 1)
        assert scale_view.global_offset == (1, 0)
        assert scale_view.local_shape == (1, 1)
        assert scale_view.shard_dims == weight_view.shard_dims == (0,)
        assert scale_view.rank == weight_view.rank

    assert tensors["layers.1.mlp.gate_proj.weight"].address == 0x40000
    assert tensors["layers.1.mlp.up_proj.weight"].address == 0x40000 + 128 * 128
    assert tensors["layers.1.mlp.gate_proj.weight_scale_inv"].address == 0x50000
    assert tensors["layers.1.mlp.up_proj.weight_scale_inv"].address == 0x50000 + 4
    manager.release(manifest.lease_id)


def test_qwen3_block_fp8_manifest_uses_ungated_q_projection_shape() -> None:
    weight = FakeTensor(
        (768, 128),
        address=0x40000,
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (6, 1),
        address=0x60000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(weight=weight, weight_scale_inv=scale)
    manager = create_weight_runtime_manifest_manager(
        model=FakeFp8RuntimeModel(
            [
                ("layers.0.self_attn.qkv_proj.weight", weight),
                ("layers.0.self_attn.qkv_proj.weight_scale_inv", scale),
            ],
            runtime_modules=(module,),
        ),
        config=qwen_config(
            model_type="qwen3_moe",
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=128,
        ),
        topology=topology(),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
        moe_runner_backend="triton",
    )

    manifest = manager.snapshot(
        model_id="qwen3-fp8",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )
    tensors = {tensor.tensor_id: tensor for tensor in manifest.tensors}

    assert tensors["layers.0.self_attn.q_proj.weight"].global_shape == (512, 128)
    assert tensors["layers.0.self_attn.k_proj.weight"].global_shape == (128, 128)
    assert tensors["layers.0.self_attn.v_proj.weight"].global_shape == (128, 128)
    assert tensors["layers.0.self_attn.q_proj.weight_scale_inv"].global_shape == (4, 1)
    assert (
        tensors["layers.0.self_attn.v_proj.weight_scale_inv"].address == 0x60000 + 5 * 4
    )
    manager.release(manifest.lease_id)


def test_qwen3_nonquantized_manifest_uses_ungated_q_projection_shape() -> None:
    parameter = FakeTensor((768, 128), address=0x40000, itemsize=2)
    manager = create_weight_runtime_manifest_manager(
        model=FakeModel([("layers.0.self_attn.qkv_proj.weight", parameter)]),
        config=qwen_config(
            model_type="qwen3",
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=128,
        ),
        topology=topology(),
        allowed_devices=("cpu",),
    )

    manifest = manager.snapshot(
        model_id="qwen3-bf16",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )
    tensors = {tensor.tensor_id: tensor for tensor in manifest.tensors}

    assert tensors["layers.0.self_attn.q_proj.weight"].global_shape == (512, 128)
    assert tensors["layers.0.self_attn.k_proj.weight"].global_shape == (128, 128)
    assert tensors["layers.0.self_attn.v_proj.weight"].global_shape == (128, 128)
    manager.release(manifest.lease_id)


def test_qwen_moe_block_fp8_scale_views_preserve_ep_and_block_tp_coordinates() -> None:
    weight = FakeTensor(
        (2, 512, 256),
        address=0x60000,
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (2, 4, 2),
        address=0xA0000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(
        w13_weight=weight,
        w13_weight_scale_inv=scale,
    )
    model = FakeFp8RuntimeModel(
        [
            ("layers.2.mlp.experts.w13_weight", weight),
            ("layers.2.mlp.experts.w13_weight_scale_inv", scale),
        ],
        runtime_modules=(module,),
    )
    manager = create_weight_runtime_manifest_manager(
        model=model,
        config=qwen_config(
            model_type="qwen3_5_moe_text",
            hidden_size=256,
            moe_intermediate_size=512,
            num_experts=8,
        ),
        topology=topology(
            ep_rank=1,
            ep_size=4,
            moe_tp_rank=1,
            moe_tp_size=2,
        ),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
        moe_runner_backend="triton",
    )

    manifest = manager.snapshot(
        model_id="qwen3.5-moe-fp8",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )
    scales = [
        tensor
        for tensor in manifest.tensors
        if tensor.tensor_id.endswith("weight_scale_inv")
    ]

    assert [(tensor.tensor_id, tensor.global_offset) for tensor in scales] == [
        ("layers.2.mlp.experts.gate_proj.weight_scale_inv", (2, 2, 0)),
        ("layers.2.mlp.experts.gate_proj.weight_scale_inv", (3, 2, 0)),
        ("layers.2.mlp.experts.up_proj.weight_scale_inv", (2, 2, 0)),
        ("layers.2.mlp.experts.up_proj.weight_scale_inv", (3, 2, 0)),
    ]
    assert {tensor.global_shape for tensor in scales} == {(8, 4, 2)}
    assert {tensor.local_shape for tensor in scales} == {(1, 2, 2)}
    assert {tensor.shard_dims for tensor in scales} == {(0, 1)}
    assert {tensor.nbytes for tensor in scales} == {16}

    weights = {
        (tensor.tensor_id, tensor.global_offset[0]): tensor
        for tensor in manifest.tensors
        if tensor.tensor_id.endswith(".weight")
    }
    for scale_view in scales:
        weight_id = scale_view.tensor_id.removesuffix("_scale_inv")
        weight_view = weights[(weight_id, scale_view.global_offset[0])]
        assert scale_view.global_shape == (
            weight_view.global_shape[0],
            weight_view.global_shape[1] // 128,
            weight_view.global_shape[2] // 128,
        )
        assert scale_view.global_offset == (
            weight_view.global_offset[0],
            weight_view.global_offset[1] // 128,
            weight_view.global_offset[2] // 128,
        )
        assert scale_view.local_shape == (
            weight_view.local_shape[0],
            weight_view.local_shape[1] // 128,
            weight_view.local_shape[2] // 128,
        )
    manager.release(manifest.lease_id)


def test_qwen_moe_block_fp8_w31_reorders_weight_and_scale_together() -> None:
    weight = FakeTensor(
        (1, 256, 128),
        address=0xB0000,
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (1, 2, 1),
        address=0xC0000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(
        w13_weight=weight,
        w13_weight_scale_inv=scale,
        load_up_proj_weight_first=True,
    )
    model = FakeFp8RuntimeModel(
        [
            ("layers.2.mlp.experts.w13_weight", weight),
            ("layers.2.mlp.experts.w13_weight_scale_inv", scale),
        ],
        runtime_modules=(module,),
    )
    manager = create_weight_runtime_manifest_manager(
        model=model,
        config=qwen_config(
            model_type="qwen3_5_moe_text",
            hidden_size=128,
            moe_intermediate_size=128,
            num_experts=1,
        ),
        topology=topology(),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
        moe_runner_backend="triton",
    )

    manifest = manager.snapshot(
        model_id="qwen3.5-moe-fp8",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )
    tensors = {tensor.tensor_id: tensor for tensor in manifest.tensors}

    assert tensors["layers.2.mlp.experts.up_proj.weight"].address == 0xB0000
    assert (
        tensors["layers.2.mlp.experts.gate_proj.weight"].address == 0xB0000 + 128 * 128
    )
    assert tensors["layers.2.mlp.experts.up_proj.weight_scale_inv"].address == 0xC0000
    assert (
        tensors["layers.2.mlp.experts.gate_proj.weight_scale_inv"].address
        == 0xC0000 + 4
    )
    for component in ("gate_proj", "up_proj"):
        weight_view = tensors[f"layers.2.mlp.experts.{component}.weight"]
        scale_view = tensors[f"layers.2.mlp.experts.{component}.weight_scale_inv"]
        assert scale_view.global_offset == (
            weight_view.global_offset[0],
            weight_view.global_offset[1] // 128,
            weight_view.global_offset[2] // 128,
        )
    manager.release(manifest.lease_id)


def test_block_fp8_manifest_rejects_online_quantization() -> None:
    weight = FakeTensor(
        (256, 128),
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (2, 1),
        address=0x20000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(
        weight=weight,
        weight_scale_inv=scale,
        is_checkpoint_fp8_serialized=False,
    )
    provider = create_weight_runtime_manifest_manager(
        model=FakeFp8RuntimeModel(
            [
                ("layers.1.mlp.gate_up_proj.weight", weight),
                ("layers.1.mlp.gate_up_proj.weight_scale_inv", scale),
            ],
            runtime_modules=(module,),
        ),
        config=qwen_config(hidden_size=128, intermediate_size=256),
        topology=topology(),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
    )

    with pytest.raises(WeightManifestError, match="(?i)(online|serialized)"):
        provider.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_fp8_manifest_rejects_per_channel_scales() -> None:
    weight = FakeTensor(
        (128, 128),
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (128, 1),
        address=0x20000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(
        weight=weight,
        weight_scale=scale,
        block_quant=False,
        weight_block_size=None,
    )
    provider = create_weight_runtime_manifest_manager(
        model=FakeFp8RuntimeModel(
            [
                ("layers.1.mlp.down_proj.weight", weight),
                ("layers.1.mlp.down_proj.weight_scale", scale),
            ],
            runtime_modules=(module,),
        ),
        config=qwen_config(hidden_size=128, intermediate_size=128),
        topology=topology(),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
    )

    with pytest.raises(WeightManifestError, match="(?i)(channel|block)"):
        provider.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_block_fp8_manifest_rejects_missing_inverse_scale() -> None:
    weight = FakeTensor(
        (256, 128),
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    module = FakeFp8RuntimeModule(weight=weight)
    provider = create_weight_runtime_manifest_manager(
        model=FakeFp8RuntimeModel(
            [("layers.1.mlp.gate_up_proj.weight", weight)],
            runtime_modules=(module,),
        ),
        config=qwen_config(hidden_size=128, intermediate_size=256),
        topology=topology(),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
    )

    with pytest.raises(WeightManifestError, match="(?i)scale"):
        provider.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_block_fp8_manifest_rejects_non_128_block_size() -> None:
    weight = FakeTensor(
        (256, 128),
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (4, 1),
        address=0x20000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(
        weight=weight,
        weight_scale_inv=scale,
        weight_block_size=(64, 128),
    )
    provider = create_weight_runtime_manifest_manager(
        model=FakeFp8RuntimeModel(
            [
                ("layers.1.mlp.gate_up_proj.weight", weight),
                ("layers.1.mlp.gate_up_proj.weight_scale_inv", scale),
            ],
            runtime_modules=(module,),
        ),
        config=qwen_config(hidden_size=128, intermediate_size=128),
        topology=topology(),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
    )

    with pytest.raises(WeightManifestError, match="128"):
        provider.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_block_fp8_manifest_rejects_unaligned_fused_partition() -> None:
    weight = FakeTensor(
        (384, 128),
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (3, 1),
        address=0x20000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(weight=weight, weight_scale_inv=scale)
    provider = create_weight_runtime_manifest_manager(
        model=FakeFp8RuntimeModel(
            [
                ("layers.1.mlp.gate_up_proj.weight", weight),
                ("layers.1.mlp.gate_up_proj.weight_scale_inv", scale),
            ],
            runtime_modules=(module,),
        ),
        config=qwen_config(hidden_size=128, intermediate_size=192),
        topology=topology(),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
    )

    with pytest.raises(WeightManifestError, match="(?i)(align|128)"):
        provider.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_block_fp8_manifest_rejects_swizzled_derived_scale() -> None:
    weight = FakeTensor(
        (256, 128),
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (2, 1),
        address=0x20000,
        dtype="torch.float32",
        itemsize=4,
    )
    swizzled_scale = FakeTensor(
        (2, 1),
        address=0x30000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(weight=weight, weight_scale_inv=scale)
    module.weight_scale_inv_swizzled = swizzled_scale
    provider = create_weight_runtime_manifest_manager(
        model=FakeFp8RuntimeModel(
            [
                ("layers.1.mlp.gate_up_proj.weight", weight),
                ("layers.1.mlp.gate_up_proj.weight_scale_inv", scale),
                (
                    "layers.1.mlp.gate_up_proj.weight_scale_inv_swizzled",
                    swizzled_scale,
                ),
            ],
            runtime_modules=(module,),
        ),
        config=qwen_config(hidden_size=128, intermediate_size=256),
        topology=topology(),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
    )

    with pytest.raises(WeightManifestError, match="(?i)swizzled"):
        provider.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_block_fp8_manifest_rejects_noncanonical_gemm_backend() -> None:
    weight = FakeTensor(
        (128, 128),
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (1, 1),
        address=0x20000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(weight=weight, weight_scale_inv=scale)
    provider = create_weight_runtime_manifest_manager(
        model=FakeFp8RuntimeModel(
            [
                ("layers.1.mlp.down_proj.weight", weight),
                ("layers.1.mlp.down_proj.weight_scale_inv", scale),
            ],
            runtime_modules=(module,),
        ),
        config=qwen_config(hidden_size=128, intermediate_size=128),
        topology=topology(),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="deep_gemm",
    )

    with pytest.raises(WeightManifestError, match="(?i)(triton|backend)"):
        provider.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_block_fp8_manifest_rejects_missing_backend_evidence() -> None:
    weight = FakeTensor(
        (128, 128),
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    scale = FakeTensor(
        (1, 1),
        address=0x20000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(weight=weight, weight_scale_inv=scale)
    provider = create_weight_runtime_manifest_manager(
        model=FakeFp8RuntimeModel(
            [
                ("layers.1.mlp.down_proj.weight", weight),
                ("layers.1.mlp.down_proj.weight_scale_inv", scale),
            ],
            runtime_modules=(module,),
        ),
        config=qwen_config(hidden_size=128, intermediate_size=128),
        topology=topology(),
        allowed_devices=("cpu",),
        quantization="fp8",
    )

    with pytest.raises(WeightManifestError, match="(?i)(triton|backend)"):
        provider.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_qwen3_next_fused_shared_expert_block_fp8_scales_use_logical_axes() -> None:
    w13_weight = FakeTensor(
        (3, 256, 128),
        address=0x30000,
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    w13_scale = FakeTensor(
        (3, 2, 1),
        address=0x50000,
        dtype="torch.float32",
        itemsize=4,
    )
    w2_weight = FakeTensor(
        (3, 128, 128),
        address=0x60000,
        dtype="torch.float8_e4m3fn",
        itemsize=1,
    )
    w2_scale = FakeTensor(
        (3, 1, 1),
        address=0x70000,
        dtype="torch.float32",
        itemsize=4,
    )
    module = FakeFp8RuntimeModule(
        w13_weight=w13_weight,
        w13_weight_scale_inv=w13_scale,
        w2_weight=w2_weight,
        w2_weight_scale_inv=w2_scale,
    )
    model = FakeFp8RuntimeModel(
        [
            ("model.layers.0.mlp.experts.w13_weight", w13_weight),
            (
                "model.layers.0.mlp.experts.w13_weight_scale_inv",
                w13_scale,
            ),
            ("model.layers.0.mlp.experts.w2_weight", w2_weight),
            ("model.layers.0.mlp.experts.w2_weight_scale_inv", w2_scale),
        ],
        runtime_modules=(module,),
    )
    model.num_fused_shared_experts = 1
    manager = create_weight_runtime_manifest_manager(
        model=model,
        config=qwen_config(
            model_type="qwen3_next",
            hidden_size=128,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            num_experts=4,
        ),
        topology=topology(ep_rank=0, ep_size=2),
        allowed_devices=("cpu",),
        quantization="fp8",
        fp8_gemm_backend="triton",
        moe_runner_backend="triton",
    )

    manifest = manager.snapshot(
        model_id="qwen3-next-fp8",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )
    tensors = {tensor.tensor_id: tensor for tensor in manifest.tensors}

    gate = tensors["layers.0.mlp.shared_expert.gate_proj.weight_scale_inv"]
    up = tensors["layers.0.mlp.shared_expert.up_proj.weight_scale_inv"]
    down = tensors["layers.0.mlp.shared_expert.down_proj.weight_scale_inv"]
    assert gate.global_shape == up.global_shape == down.global_shape == (1, 1)
    assert gate.local_shape == up.local_shape == down.local_shape == (1, 1)
    assert gate.address == 0x50000 + 16
    assert up.address == 0x50000 + 20
    assert down.address == 0x70000 + 8
    routed_down = [
        tensor
        for tensor in manifest.tensors
        if tensor.tensor_id == "layers.0.mlp.experts.down_proj.weight_scale_inv"
    ]
    assert [tensor.global_shape for tensor in routed_down] == [(4, 1, 1)] * 2
    assert [tensor.global_offset for tensor in routed_down] == [
        (0, 0, 0),
        (1, 0, 0),
    ]
    assert [tensor.address for tensor in routed_down] == [0x70000, 0x70000 + 4]
    manager.release(manifest.lease_id)


def test_qwen3_next_factory_uses_grouped_gdn_runtime_semantics() -> None:
    parameter = FakeTensor((24, 8), address=0x50000, itemsize=2)
    parameter._sglang_qwen3_next_gdn_layout = "grouped"
    manager = create_weight_runtime_manifest_manager(
        model=FakeModel(
            [("model.layers.0.linear_attn.in_proj_qkvz.weight", parameter)]
        ),
        config=qwen_config(
            model_type="qwen3_next",
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
        ),
        topology=topology(attention_tp_rank=1, attention_tp_size=2),
        allowed_devices=("cpu",),
        moe_runner_backend="triton",
    )

    manifest = manager.snapshot(
        model_id="qwen3-next",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )

    assert len(manifest.tensors) == 1
    tensor = manifest.tensors[0]
    assert tensor.tensor_id == "layers.0.linear_attn.in_proj_qkvz.weight"
    assert tensor.global_shape == (4, 12, 8)
    assert tensor.global_offset == (2, 0, 0)
    assert tensor.local_shape == (2, 12, 8)
    assert tensor.partition_dim == 0
    assert tensor.shard_dims == (0,)
    assert tensor.nbytes == parameter.numel() * parameter.element_size()
    assert tensor.layout_fingerprint == "sglang:qwen3-next:gdn-qkvz-grouped:v1"
    manager.release(manifest.lease_id)


def test_qwen3_next_groups_gdn_ba_by_key_head() -> None:
    adapter = Qwen3NextWeightSemanticsAdapter(
        config=qwen_config(
            model_type="qwen3_next",
            linear_num_key_heads=4,
            linear_num_value_heads=8,
        )
    )

    parameter = FakeTensor((8, 8), itemsize=2)
    parameter._sglang_qwen3_next_gdn_layout = "grouped"
    views = adapter.describe_parameter(
        names=("model.layers.0.linear_attn.in_proj_ba.weight",),
        parameter=parameter,
        topology=topology(attention_tp_rank=1, attention_tp_size=2),
    )

    assert len(views) == 1
    assert views[0].global_shape == (4, 4, 8)
    assert views[0].global_offset == (2, 0, 0)
    assert views[0].local_shape == (2, 4, 8)
    assert views[0].layout_fingerprint == "sglang:qwen3-next:gdn-ba-grouped:v1"


def test_qwen3_next_retags_full_attention_and_uses_contiguous_kv_replicas() -> None:
    adapter = Qwen3NextWeightSemanticsAdapter(
        config=qwen_config(
            model_type="qwen3_next",
            num_key_value_heads=2,
            attn_output_gate=True,
        )
    )

    views = adapter.describe_parameter(
        names=("model.layers.3.self_attn.qkv_proj.weight",),
        parameter=FakeTensor((8, 8), itemsize=2),
        topology=topology(
            tp_rank=2,
            tp_size=4,
            attention_tp_rank=2,
            attention_tp_size=4,
        ),
    )

    assert [view.global_offset for view in views] == [(8, 0), (2, 0), (2, 0)]
    assert {view.layout_fingerprint for view in views} == {"sglang:qwen3-next:qkv:v1"}


def test_qwen3_next_describes_fused_shared_expert_slot() -> None:
    adapter = Qwen3NextWeightSemanticsAdapter(
        config=qwen_config(model_type="qwen3_next", shared_expert_intermediate_size=8),
        num_fused_shared_experts=1,
    )

    w13_views = adapter.describe_parameter(
        names=("model.layers.0.mlp.experts.w13_weight",),
        parameter=FakeTensor((3, 8, 8), itemsize=2),
        topology=topology(ep_rank=0, ep_size=4, moe_tp_rank=0, moe_tp_size=2),
    )
    w2_views = adapter.describe_parameter(
        names=("model.layers.0.mlp.experts.w2_weight",),
        parameter=FakeTensor((3, 8, 4), itemsize=2),
        topology=topology(ep_rank=0, ep_size=4, moe_tp_rank=0, moe_tp_size=2),
    )

    assert [view.expert_id for view in w13_views] == [None] * 6
    assert [view.tensor_id for view in w13_views[:4]] == [
        "layers.0.mlp.experts.gate_proj.weight",
        "layers.0.mlp.experts.up_proj.weight",
        "layers.0.mlp.experts.gate_proj.weight",
        "layers.0.mlp.experts.up_proj.weight",
    ]
    assert [view.shard_dims for view in w13_views[:4]] == [(0, 1)] * 4
    assert [view.tensor_id for view in w13_views[-2:]] == [
        "layers.0.mlp.shared_expert.gate_proj.weight",
        "layers.0.mlp.shared_expert.up_proj.weight",
    ]
    assert [view.byte_offset for view in w13_views] == [0, 64, 128, 192, 256, 320]
    assert [view.expert_id for view in w2_views] == [None, None, None]
    assert [view.tensor_id for view in w2_views[:2]] == [
        "layers.0.mlp.experts.down_proj.weight",
        "layers.0.mlp.experts.down_proj.weight",
    ]
    assert [view.shard_dims for view in w2_views[:2]] == [(0, 2), (0, 2)]
    assert w2_views[-1].tensor_id == ("layers.0.mlp.shared_expert.down_proj.weight")
    assert w2_views[-1].byte_offset == 128


def test_qwen3_next_fused_and_unfused_shared_expert_semantics_match() -> None:
    config = qwen_config(model_type="qwen3_next", shared_expert_intermediate_size=8)
    fused = Qwen3NextWeightSemanticsAdapter(
        config=config,
        num_fused_shared_experts=1,
    )
    unfused = Qwen3NextWeightSemanticsAdapter(config=config)
    parallel = topology(
        tp_rank=1,
        tp_size=2,
        ep_rank=0,
        ep_size=4,
        moe_tp_rank=1,
        moe_tp_size=2,
    )

    fused_gate_up = fused.describe_parameter(
        names=("model.layers.0.mlp.experts.w13_weight",),
        parameter=FakeTensor((3, 8, 8), itemsize=2),
        topology=parallel,
    )[-2:]
    unfused_gate_up = unfused.describe_parameter(
        names=("model.layers.0.mlp.shared_expert.gate_up_proj.weight",),
        parameter=FakeTensor((8, 8), itemsize=2),
        topology=parallel,
    )
    fused_down = fused.describe_parameter(
        names=("model.layers.0.mlp.experts.w2_weight",),
        parameter=FakeTensor((3, 8, 4), itemsize=2),
        topology=parallel,
    )[-1]
    unfused_down = unfused.describe_parameter(
        names=("model.layers.0.mlp.shared_expert.down_proj.weight",),
        parameter=FakeTensor((8, 4), itemsize=2),
        topology=parallel,
    )[0]

    def semantic_key(view):
        return (
            view.tensor_id,
            view.global_shape,
            view.global_offset,
            view.local_shape,
            view.partition_dim,
            view.shard_dims,
            view.expert_id,
            view.layout_fingerprint,
        )

    assert [semantic_key(view) for view in fused_gate_up] == [
        semantic_key(view) for view in unfused_gate_up
    ]
    assert semantic_key(fused_down) == semantic_key(unfused_down)


def test_qwen3_next_rejects_split_checkpoint_gdn_runtime_layout() -> None:
    parameter = FakeTensor((24, 8), itemsize=2)
    parameter._sglang_qwen3_next_gdn_layout = "component"
    adapter = Qwen3NextWeightSemanticsAdapter(
        config=qwen_config(
            model_type="qwen3_next",
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
        )
    )

    with pytest.raises(WeightManifestError, match="marker must explicitly"):
        adapter.describe_parameter(
            names=("model.layers.0.linear_attn.in_proj_qkvz.weight",),
            parameter=parameter,
            topology=topology(attention_tp_rank=1, attention_tp_size=2),
        )


def test_qwen3_next_rejects_missing_gdn_runtime_layout_marker() -> None:
    adapter = Qwen3NextWeightSemanticsAdapter(
        config=qwen_config(
            model_type="qwen3_next",
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
        )
    )

    with pytest.raises(
        WeightManifestError,
        match=r"in_proj_qkvz\.weight: None",
    ):
        adapter.describe_parameter(
            names=("model.layers.0.linear_attn.in_proj_qkvz.weight",),
            parameter=FakeTensor((24, 8), itemsize=2),
            topology=topology(attention_tp_rank=1, attention_tp_size=2),
        )


def test_qwen_expert_manifest_rejects_nontrivial_placement_without_map() -> None:
    manager = create_weight_runtime_manifest_manager(
        model=FakeModel(
            [
                (
                    "model.layers.0.mlp.experts.w13_weight",
                    FakeTensor((2, 8, 8), itemsize=2),
                )
            ]
        ),
        config=qwen_config(model_type="qwen3_next"),
        topology=topology(ep_rank=0, ep_size=4, moe_tp_rank=0, moe_tp_size=2),
        allowed_devices=("cpu",),
        dynamic_expert_placement=True,
        moe_runner_backend="triton",
    )

    with pytest.raises(WeightManifestError, match="explicit expert map"):
        manager.snapshot(
            model_id="qwen3-next",
            revision="step-1",
            instance_id="instance-0",
            worker_id="worker-0",
            endpoint="worker-0:12345",
        )


def test_qwen3_next_factory_rejects_noncanonical_moe_runner_layout() -> None:
    manager = create_weight_runtime_manifest_manager(
        model=FakeModel([]),
        config=qwen_config(model_type="qwen3_next"),
        topology=topology(),
        allowed_devices=("cpu",),
        moe_runner_backend="triton_kernel",
    )

    with pytest.raises(WeightManifestError, match="MoE runner backend"):
        manager.snapshot(
            model_id="qwen3-next",
            revision="step-1",
            instance_id="instance-0",
            worker_id="worker-0",
            endpoint="worker-0:12345",
        )


def test_qwen_multimodal_factory_describes_tp_sharded_vision_parameters() -> None:
    parameters = [
        (
            "visual.patch_embed.proj.weight",
            FakeTensor((8, 3, 2, 2, 2), address=0x10000),
        ),
        (
            "visual.blocks.0.attn.qkv_proj.weight",
            FakeTensor((12, 8), address=0x20000),
        ),
        (
            "visual.blocks.0.attn.qkv_proj.bias",
            FakeTensor((12,), address=0x30000),
        ),
        (
            "visual.blocks.0.attn.proj.weight",
            FakeTensor((8, 4), address=0x40000),
        ),
        (
            "visual.blocks.0.attn.proj.bias",
            FakeTensor((8,), address=0x50000),
        ),
        (
            "visual.blocks.0.mlp.linear_fc1.weight",
            FakeTensor((8, 8), address=0x60000),
        ),
        (
            "visual.blocks.0.mlp.linear_fc2.weight",
            FakeTensor((8, 8), address=0x70000),
        ),
        (
            "model.embed_tokens.weight",
            FakeTensor((16, 8), address=0x80000),
        ),
    ]
    manager = create_weight_runtime_manifest_manager(
        model=FakeModel(parameters),
        config=qwen_multimodal_config(),
        topology=topology(
            tp_rank=1,
            tp_size=2,
            attention_tp_rank=1,
            attention_tp_size=2,
        ),
        allowed_devices=("cpu",),
        is_multimodal=True,
    )

    manifest = manager.snapshot(
        model_id="qwen3.5-0.8b",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )
    tensors = {tensor.tensor_id: tensor for tensor in manifest.tensors}

    assert tensors["visual.patch_embed.proj.weight"].partition_dim is None
    assert tensors["visual.blocks.0.attn.q_proj.weight"].global_offset == (4, 0)
    assert tensors["visual.blocks.0.attn.k_proj.weight"].byte_offset == 64
    assert tensors["visual.blocks.0.attn.v_proj.bias"].byte_offset == 16
    assert tensors["visual.blocks.0.attn.proj.weight"].global_offset == (0, 4)
    assert tensors["visual.blocks.0.mlp.linear_fc1.weight"].global_offset == (
        8,
        0,
    )
    assert tensors["visual.blocks.0.mlp.linear_fc2.weight"].global_offset == (
        0,
        8,
    )
    assert tensors["embed_tokens.weight"].global_offset == (16, 0)
    manager.release(manifest.lease_id)


def test_qwen_multimodal_vision_data_parallel_is_described_as_replicated() -> None:
    manager = create_weight_runtime_manifest_manager(
        model=FakeModel(
            [
                (
                    "visual.blocks.0.attn.qkv_proj.weight",
                    FakeTensor((24, 8), address=0x10000),
                ),
                (
                    "visual.blocks.0.mlp.linear_fc1.weight",
                    FakeTensor((16, 8), address=0x20000),
                ),
            ]
        ),
        config=qwen_multimodal_config(),
        topology=topology(
            tp_rank=1,
            tp_size=2,
            attention_tp_rank=1,
            attention_tp_size=2,
        ),
        allowed_devices=("cpu",),
        is_multimodal=True,
    )

    manifest = manager.snapshot(
        model_id="qwen3.5-0.8b",
        revision="step-1",
        instance_id="instance-0",
        worker_id="worker-0",
        endpoint="worker-0:12345",
    )

    assert {tensor.partition_dim for tensor in manifest.tensors} == {0}
    assert all(
        all(offset == 0 for offset in tensor.global_offset)
        for tensor in manifest.tensors
    )
    assert all(tensor.local_shape == tensor.global_shape for tensor in manifest.tensors)
    manager.release(manifest.lease_id)


def test_qwen_runtime_inventory_matches_mooncake_golden_contract() -> None:
    manager = WeightRuntimeManifestManager(
        model=FakeModel(
            [
                (
                    "layers.10.mlp.experts.w13_weight",
                    FakeTensor((2, 4, 8), address=0x10000, itemsize=2),
                )
            ]
        ),
        adapter=Qwen35WeightSemanticsAdapter(
            config=qwen_config(model_type="qwen3_5_moe_text")
        ),
        topology=topology(
            tp_rank=0,
            tp_size=4,
            pp_rank=1,
            pp_size=2,
            ep_rank=2,
            ep_size=4,
            moe_tp_rank=0,
            moe_tp_size=4,
        ),
        allowed_devices=("cpu",),
    )
    manifest = manager.snapshot(
        model_id="qwen3.5-moe",
        revision="step-42",
        instance_id="source-p1-e2-t0",
        worker_id="source-p1-e2-t0",
        endpoint="source-p1-e2-t0:12345",
    )
    actual = json.loads(msgspec.json.encode(manifest))
    actual["lease_id"] = "<runtime-lease>"
    expected = json.loads(
        Path(
            "test/registered/unit/model_executor/fixtures/"
            "qwen3_5_moe_runtime_manifest.json"
        ).read_text()
    )

    assert actual == expected
    manager.release(manifest.lease_id)


def test_sglang_factory_builds_topology_outside_model_runner() -> None:
    parallel_state = SimpleNamespace(
        dp_rank=None,
        dp_size=1,
        moe_dp_rank=2,
        moe_dp_size=3,
        tp_rank=1,
        tp_size=2,
        pp_rank=1,
        pp_size=2,
        moe_ep_rank=3,
        moe_ep_size=4,
        attn_tp_rank=1,
        attn_tp_size=2,
    )
    parallel = SimpleNamespace(moe_tp_rank=1, moe_tp_size=2)

    manager = create_sglang_weight_runtime_manifest_manager(
        model=FakeModel([]),
        config=qwen_config(),
        parallel_state=parallel_state,
        parallel=parallel,
        allowed_devices=("cpu",),
    )

    assert manager._topology == topology(
        dp_rank=2,
        dp_size=3,
        tp_rank=1,
        tp_size=2,
        pp_rank=1,
        pp_size=2,
        ep_rank=3,
        ep_size=4,
        moe_tp_rank=1,
        moe_tp_size=2,
        attention_tp_rank=1,
        attention_tp_size=2,
    )


def test_qwen_dp_attention_is_rejected_until_vocab_replication_is_described() -> None:
    provider = create_weight_runtime_manifest_manager(
        model=FakeModel([]),
        config=qwen_config(),
        topology=topology(dp_rank=1, dp_size=2, tp_rank=1, tp_size=2),
        allowed_devices=("cpu",),
        dp_attention_enabled=True,
    )

    with pytest.raises(WeightManifestError, match="DP attention"):
        provider.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_unsupported_model_is_lazy_and_fails_only_when_snapshot_is_requested() -> None:
    """Adding the provider must not break normal inference for other model families."""
    provider = create_weight_runtime_manifest_manager(
        model=FakeModel([]),
        config=SimpleNamespace(model_type="deepseek_v3"),
        topology=topology(),
        allowed_devices=("cpu",),
    )

    with pytest.raises(WeightManifestError, match="unsupported model type"):
        provider.snapshot(
            model_id="model",
            revision="revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:12345",
        )


def test_model_runner_provider_is_lazy_after_layout_transforms() -> None:
    """Normal inference must not traverse parameters for an unused exporter."""
    source = Path("python/sglang/srt/model_executor/model_runner.py").read_text()
    tree = ast.parse(source)
    initialize = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "initialize"
    )
    calls = []
    for statement in initialize.body:
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        if isinstance(statement.value.func, ast.Attribute):
            calls.append(statement.value.func.attr)

    assert "init_weight_runtime_manifest_manager" not in calls
    assert "maybe_apply_post_load_model_transforms" in calls
    assert "maybe_init_lora_manager" in calls


def test_model_runner_target_builder_follows_local_mooncake_capability() -> None:
    runner_tree = ast.parse(
        Path("python/sglang/srt/model_executor/model_runner.py").read_text()
    )
    selector = next(
        (
            node
            for node in ast.walk(runner_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_select_remote_instance_target_weight_manifest_builder"
        ),
        None,
    )

    assert selector is not None
    selector_attributes = {
        node.attr for node in ast.walk(selector) if isinstance(node, ast.Attribute)
    }
    assert "build_remote_instance_target_weight_manifest_session" in (
        selector_attributes
    )
    assert "build_remote_instance_target_weight_runtime_manifest" in (
        selector_attributes
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "local_mooncake_supports_placement_binding"
        for node in ast.walk(selector)
    )

    load_model = next(
        node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )
    build_load_config = next(
        node
        for node in ast.walk(load_model)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_load_config"
    )
    builder_value = next(
        keyword.value
        for keyword in build_load_config.keywords
        if keyword.arg == "remote_instance_weight_runtime_manifest_builder"
    )
    assert (
        isinstance(builder_value, ast.Call)
        and isinstance(builder_value.func, ast.Attribute)
        and builder_value.func.attr
        == "_select_remote_instance_target_weight_manifest_builder"
    )


def test_model_runner_rejects_nontrivial_static_expert_placement() -> None:
    source = Path("python/sglang/srt/model_executor/model_runner.py").read_text()

    assert 'self.server_args.init_expert_location != "trivial"' in source
    assert "self.server_args.ep_num_redundant_experts > 0" in source
    assert "moe_runner_backend=self.server_args.moe_runner_backend" in source


def test_qwen3_next_loader_records_gdn_runtime_layout() -> None:
    source = Path("python/sglang/srt/models/qwen3_next.py").read_text()

    assert source.count("_sglang_qwen3_next_gdn_layout") >= 2
    assert '"grouped"' in source
    assert '"component"' in source


def test_model_runner_wires_all_online_updates_to_snapshot_coordinator() -> None:
    runner_tree = ast.parse(
        Path("python/sglang/srt/model_executor/model_runner.py").read_text()
    )
    init_updater = next(
        node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "init_weight_updater"
    )
    coordination_keys = {
        key.value
        for node in ast.walk(init_updater)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert {"begin_weight_update", "finish_weight_update"} <= coordination_keys
    assert any(
        isinstance(node, ast.Attribute)
        and node.attr == "enable_weight_runtime_manifest"
        for node in ast.walk(init_updater)
    )
    coordinator_calls = [
        node
        for node in ast.walk(init_updater)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "WeightSnapshotCoordinator"
    ]
    assert len(coordinator_calls) == 1
    completion_fence = next(
        (
            keyword.value
            for keyword in coordinator_calls[0].keywords
            if keyword.arg == "completion_fence"
        ),
        None,
    )
    assert (
        isinstance(completion_fence, ast.Attribute)
        and isinstance(completion_fence.value, ast.Name)
        and completion_fence.value.id == "current_platform"
        and completion_fence.attr == "synchronize"
    )

    updater_tree = ast.parse(
        Path(
            "python/sglang/srt/model_executor/model_runner_components/weight_updater.py"
        ).read_text()
    )
    public_updates = {
        "update_weights_from_disk",
        "update_weights_from_distributed",
        "update_weights_from_tensor",
        "update_weights_from_ipc",
    }
    methods = {
        node.name: node
        for node in ast.walk(updater_tree)
        if isinstance(node, ast.FunctionDef) and node.name in public_updates
    }
    assert set(methods) == public_updates
    for method in methods.values():
        assert any(
            isinstance(decorator, ast.Name)
            and decorator.id == "coordinated_weight_update"
            for decorator in method.decorator_list
        )
    disk_update = methods["update_weights_from_disk"]
    disk_arguments = [argument.arg for argument in disk_update.args.args]
    filter_index = disk_arguments.index("weight_name_filter")
    default_index = filter_index - (
        len(disk_arguments) - len(disk_update.args.defaults)
    )
    assert isinstance(disk_update.args.defaults[default_index], ast.Constant)
    assert disk_update.args.defaults[default_index].value is None
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "weight_name_filter"
        and any(isinstance(operator, ast.IsNot) for operator in node.ops)
        and any(
            isinstance(comparator, ast.Constant) and comparator.value is None
            for comparator in node.comparators
        )
        for node in ast.walk(disk_update)
    )

    tp_worker_tree = ast.parse(
        Path("python/sglang/srt/managers/tp_worker.py").read_text()
    )
    tp_disk_update = next(
        node
        for node in ast.walk(tp_worker_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "update_weights_from_disk"
    )
    production_calls = [
        node
        for node in ast.walk(tp_disk_update)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update_weights_from_disk"
    ]
    assert len(production_calls) == 1
    assert not any(
        keyword.arg == "weight_name_filter" for keyword in production_calls[0].keywords
    )

    scheduler_tree = ast.parse(
        Path(
            "python/sglang/srt/managers/scheduler_components/weight_updater.py"
        ).read_text()
    )
    scheduler_disk_update = next(
        node
        for node in ast.walk(scheduler_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "update_weights_from_disk"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_run_weight_update_transaction"
        for node in ast.walk(scheduler_disk_update)
    )
    finalize_update = next(
        node
        for node in ast.walk(scheduler_tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_finalize_weight_update"
    )
    finalization_calls = [
        node
        for node in ast.walk(finalize_update)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"commit_revision", "poison_global_update_failure"}
    ]
    assert {node.func.attr for node in finalization_calls} == {
        "commit_revision",
        "poison_global_update_failure",
    }
    assert all(
        any(keyword.arg == "expected_generation" for keyword in call.keywords)
        for call in finalization_calls
    )


def test_model_runner_exposes_cross_rank_failure_poison_hook() -> None:
    runner_tree = ast.parse(
        Path("python/sglang/srt/model_executor/model_runner.py").read_text()
    )
    poison_method = next(
        (
            node
            for node in ast.walk(runner_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "poison_weight_runtime_after_global_failure"
        ),
        None,
    )

    assert poison_method is not None
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "poison_global_update_failure"
        for node in ast.walk(poison_method)
    )
    assert any(
        argument.arg == "expected_generation" for argument in poison_method.args.args
    )


def test_model_runner_releases_manifest_lease_on_cancellation() -> None:
    runner_tree = ast.parse(
        Path("python/sglang/srt/model_executor/model_runner.py").read_text()
    )
    methods = {
        node.name: node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "get_remote_instance_weight_runtime_manifest",
            "get_remote_instance_weight_runtime_manifest_parts",
        }
    }

    assert set(methods) == {
        "get_remote_instance_weight_runtime_manifest",
        "get_remote_instance_weight_runtime_manifest_parts",
    }
    for method in methods.values():
        handlers = [
            node for node in ast.walk(method) if isinstance(node, ast.ExceptHandler)
        ]
        assert any(
            isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
            for handler in handlers
        )


def test_model_runner_keeps_model_revision_independent_from_generation() -> None:
    runner_tree = ast.parse(
        Path("python/sglang/srt/model_executor/model_runner.py").read_text()
    )
    methods = {
        node.name: node
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "get_weight_runtime_manifest",
            "get_weight_runtime_manifest_parts",
        }
    }

    assert set(methods) == {
        "get_weight_runtime_manifest",
        "get_weight_runtime_manifest_parts",
    }
    for method in methods.values():
        assert not any(
            isinstance(node, ast.Attribute) and node.attr == "generation"
            for node in ast.walk(method)
        )
        snapshot_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"snapshot", "snapshot_parts"}
        ]
        assert len(snapshot_calls) == 1
        assert all(
            keyword.arg != "bind_revision_to_generation"
            for keyword in snapshot_calls[0].keywords
        )


def test_runtime_manifest_exporter_is_disabled_by_default() -> None:
    tree = ast.parse(Path("python/sglang/srt/server_args.py").read_text())
    field = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "enable_weight_runtime_manifest"
    )

    assert isinstance(field.value, ast.Constant)
    assert field.value.value is False
