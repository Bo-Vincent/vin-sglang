from __future__ import annotations

import ast
import json
from math import prod
from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest

from sglang.srt.model_executor.weight_runtime_manifest import (
    LogicalTensorView,
    WeightManifestError,
    WeightParallelTopology,
    WeightRuntimeManifestManager,
    WeightSnapshotCoordinator,
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
        itemsize: int = 2,
        device: str = "cpu",
        contiguous: bool = True,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = "torch.bfloat16"
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


class FakeMoEModel(FakeModel):
    def __init__(self, parameters, *, w13_parameter, up_first: bool) -> None:
        super().__init__(parameters)
        self._moe_module = SimpleNamespace(
            w13_weight=w13_parameter,
            use_flashinfer_trtllm_moe=up_first,
        )

    def modules(self):
        return iter((self, self._moe_module))


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


def test_snapshot_lease_ttl_expires_inside_coordinator() -> None:
    now = [100.0]
    coordinator = WeightSnapshotCoordinator(clock=lambda: now[0])

    lease_id, generation = coordinator.acquire_snapshot(lease_timeout_sec=30)
    assert generation == 1

    now[0] = 129.0
    with pytest.raises(WeightManifestError, match="snapshot lease is active"):
        coordinator.begin_update()

    now[0] = 130.0
    token = coordinator.begin_update()
    coordinator.finish_update(token, success=True)
    coordinator.commit_revision()

    with pytest.raises(WeightManifestError, match="does not exist"):
        coordinator.release_snapshot(lease_id)


def test_snapshot_lease_renewal_extends_coordinator_deadline() -> None:
    now = [100.0]
    coordinator = WeightSnapshotCoordinator(clock=lambda: now[0])
    lease_id, _ = coordinator.acquire_snapshot(lease_timeout_sec=30)

    now[0] = 120.0
    coordinator.renew_snapshot(lease_id, lease_timeout_sec=30)

    now[0] = 149.0
    with pytest.raises(WeightManifestError, match="snapshot lease is active"):
        coordinator.begin_update()

    now[0] = 150.0
    token = coordinator.begin_update()
    coordinator.finish_update(token, success=True)
    coordinator.commit_revision()


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

    assert updater.update((True, "recovered")) == (True, "recovered")
    coordinator.commit_revision()
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 4
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


def test_online_update_exception_poisons_runtime_snapshots() -> None:
    coordinator = WeightSnapshotCoordinator()
    updater = DummyWeightUpdater(coordinator)

    with pytest.raises(RuntimeError, match="update failed"):
        updater.update((True, "unused"), raise_error=True)

    assert coordinator.generation == 2
    with pytest.raises(WeightManifestError, match="last weight update failed"):
        coordinator.acquire_snapshot()

    assert updater.update((True, "recovered")) == (True, "recovered")
    coordinator.commit_revision()
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 3
    coordinator.release_snapshot(lease_id)


def test_failed_weight_update_poison_snapshot_until_a_full_update_succeeds() -> None:
    coordinator = WeightSnapshotCoordinator()
    failed = coordinator.begin_update()
    coordinator.finish_update(failed, success=False)

    with pytest.raises(WeightManifestError, match="last weight update failed"):
        coordinator.acquire_snapshot()

    recovered = coordinator.begin_update()
    coordinator.finish_update(recovered, success=True)
    coordinator.commit_revision()
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 3
    coordinator.release_snapshot(lease_id)


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


def test_qwen3_next_factory_uses_grouped_gdn_runtime_semantics() -> None:
    parameter = FakeTensor((24, 8), address=0x50000, itemsize=2)
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

    views = adapter.describe_parameter(
        names=("model.layers.0.linear_attn.in_proj_ba.weight",),
        parameter=FakeTensor((8, 8), itemsize=2),
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
        config=qwen_config(
            model_type="qwen3_next", shared_expert_intermediate_size=8
        ),
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
    assert w2_views[-1].tensor_id == (
        "layers.0.mlp.shared_expert.down_proj.weight"
    )
    assert w2_views[-1].byte_offset == 128


def test_qwen3_next_fused_and_unfused_shared_expert_semantics_match() -> None:
    config = qwen_config(
        model_type="qwen3_next", shared_expert_intermediate_size=8
    )
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

    with pytest.raises(WeightManifestError, match="split-checkpoint GDN"):
        adapter.describe_parameter(
            names=("model.layers.0.linear_attn.in_proj_qkvz.weight",),
            parameter=parameter,
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
