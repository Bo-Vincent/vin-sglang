from __future__ import annotations

from math import prod
from types import SimpleNamespace

import msgspec
import pytest

from sglang.srt.model_executor.weight_inventory import WeightInventoryManager
from sglang.srt.model_executor.weight_inventory_contracts import (
    LogicalParallelAxis,
    LogicalTensorView,
    WeightInventoryError,
    WeightParallelTopology,
)
from sglang.srt.model_executor.weight_inventory_factory import (
    create_weight_inventory_manager,
    topology_from_sglang,
)
from sglang.srt.model_executor.weight_semantics.factory import (
    _runtime_num_fused_shared_experts,
)
from sglang.srt.model_executor.weight_semantics.fp8_block import _retag_weight_view
from sglang.srt.model_executor.weight_semantics.qwen3_5 import (
    Qwen35MultimodalWeightSemanticsAdapter,
    Qwen35WeightSemanticsAdapter,
    _moe_parallel_semantics,
    _shared_moe_parallel_semantics,
)
from sglang.srt.model_executor.weight_semantics.qwen3_next import (
    Qwen3NextWeightSemanticsAdapter,
)
from sglang.srt.model_executor.weight_snapshot import WeightSnapshotCoordinator


class _FakeStorage:
    def __init__(self, address: int, nbytes: int) -> None:
        self.address = address
        self.size = nbytes

    def data_ptr(self) -> int:
        return self.address

    def nbytes(self) -> int:
        return self.size


class _FakeTensor:
    def __init__(
        self,
        shape,
        *,
        address=0x1000,
        storage_address=None,
        storage_nbytes=None,
        storage_offset=0,
        itemsize=2,
        dtype="torch.bfloat16",
    ) -> None:
        self.shape = tuple(shape)
        self._itemsize = itemsize
        self.dtype = dtype
        self.device = SimpleNamespace(type="cpu")
        self.layout = "torch.strided"
        self.is_sparse = False
        self._storage_offset = storage_offset
        self._storage_address = (
            address - storage_offset * itemsize
            if storage_address is None
            else storage_address
        )
        self._address = address
        size = prod(self.shape) * itemsize
        self._storage = _FakeStorage(
            self._storage_address,
            storage_nbytes or storage_offset * itemsize + size,
        )

    def data_ptr(self) -> int:
        return self._address

    def element_size(self) -> int:
        return self._itemsize

    def numel(self) -> int:
        return prod(self.shape)

    def is_contiguous(self) -> bool:
        return True

    def stride(self):
        value = 1
        result = []
        for extent in reversed(self.shape):
            result.append(value)
            value *= extent
        return tuple(reversed(result))

    def storage_offset(self) -> int:
        return self._storage_offset

    def untyped_storage(self):
        return self._storage


class _FakeModel:
    def __init__(self, parameters, *, modules=()) -> None:
        self.parameters = tuple(parameters)
        self._modules = tuple(modules)

    def named_parameters(self, *, remove_duplicate: bool):
        assert remove_duplicate is False
        return iter(self.parameters)

    def modules(self):
        return iter((self, *self._modules))


def _topology(**overrides) -> WeightParallelTopology:
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


def _axes(
    shard_dims=(),
    *,
    ep_mode="replicated",
) -> tuple[LogicalParallelAxis, ...]:
    tp_dims = tuple(dim for dim in shard_dims if not (ep_mode == "split" and dim == 0))
    return (
        LogicalParallelAxis(kind="dp", mode="replicated"),
        LogicalParallelAxis(
            kind="tp",
            mode="split" if tp_dims else "replicated",
            dim=tp_dims[0] if tp_dims else None,
        ),
        LogicalParallelAxis(kind="pp", mode="ownership"),
        LogicalParallelAxis(
            kind="ep",
            mode=ep_mode,
            dim=0 if ep_mode == "split" else None,
        ),
    )


def _view(
    *,
    tensor_id="logical.weight",
    global_shape=(4, 4),
    global_offset=(0, 0),
    local_shape=(4, 4),
    byte_offset=0,
    shard_dims=(),
    parallel_axes=None,
    layout_fingerprint="logical-contiguous",
    aliases=None,
) -> LogicalTensorView:
    kwargs = dict(
        tensor_id=tensor_id,
        global_shape=global_shape,
        global_offset=global_offset,
        local_shape=local_shape,
        byte_offset=byte_offset,
        layer_id=0,
        expert_id=None,
        layout_fingerprint=layout_fingerprint,
        shard_dims=tuple(shard_dims),
        parallel_axes=parallel_axes or _axes(shard_dims),
    )
    if aliases is not None:
        kwargs["aliases"] = aliases
    return LogicalTensorView(**kwargs)


class _Semantics:
    def __init__(self, describe=None) -> None:
        self.describe = describe or (lambda *_args: (_view(),))

    def describe_parameter(self, *, names, parameter, topology):
        return self.describe(names, parameter, topology)


def _manager(
    *,
    model=None,
    semantics=None,
    topology=None,
    coordinator=None,
) -> WeightInventoryManager:
    return WeightInventoryManager(
        model=model or _FakeModel((("runtime.weight", _FakeTensor((4, 4))),)),
        adapter=semantics or _Semantics(),
        topology=topology or _topology(),
        allowed_devices=("cpu",),
        coordinator=coordinator,
    )


def _snapshot(manager, *, suffix="0"):
    return manager.snapshot_inventories(
        model_id="model",
        revision="immutable-revision",
        instance_id=f"instance-{suffix}",
        worker_id=f"worker-{suffix}",
        endpoint=f"worker-{suffix}:1234",
    )


def test_inventory_capture_rejects_a_lease_that_expires_during_scan() -> None:
    now = [100.0]
    coordinator = WeightSnapshotCoordinator(clock=lambda: now[0])

    def describe(*_args):
        now[0] += 31.0
        return (_view(),)

    manager = _manager(
        semantics=_Semantics(describe),
        coordinator=coordinator,
    )

    with pytest.raises(
        WeightInventoryError,
        match="expired during inventory capture",
    ):
        manager.snapshot_inventories(
            model_id="model",
            revision="immutable-revision",
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:1234",
            lease_timeout_sec=30,
        )

    assert coordinator.list_snapshot_leases() == ()


def test_public_placement_identity_ignores_runtime_name_and_allocation() -> None:
    first = _manager(
        model=_FakeModel((("first.runtime.name", _FakeTensor((4, 4), address=0x1000)),))
    )
    second = _manager(
        model=_FakeModel(
            (
                (
                    "renamed.runtime.parameter",
                    _FakeTensor(
                        (4, 4),
                        address=0x9020,
                        storage_address=0x9000,
                        storage_offset=16,
                        storage_nbytes=0x400,
                    ),
                ),
            )
        )
    )

    first_snapshot = _snapshot(first, suffix="a")
    second_snapshot = _snapshot(second, suffix="b")
    try:
        assert first_snapshot.placement == second_snapshot.placement
        assert (
            first_snapshot.binding.fragments[0].address
            != second_snapshot.binding.fragments[0].address
        )
        public = msgspec.to_builtins(first_snapshot.placement.fragments[0])
        assert {"runtime_name", "byte_offset", "partition_dim"}.isdisjoint(public)
    finally:
        first.release(first_snapshot.binding.lease_id)
        second.release(second_snapshot.binding.lease_id)


@pytest.mark.parametrize(
    "describe",
    (
        lambda *_args: (_view(global_offset=(1, 0), local_shape=(3, 4)),),
        lambda *_args: (_view(layout_fingerprint="different-layout"),),
        lambda *_args: (
            _view(tensor_id="logical.weight"),
            _view(tensor_id="logical.alias"),
        ),
    ),
)
def test_logical_box_layout_and_alias_semantics_change_placement_identity(
    describe,
) -> None:
    baseline = _manager().placement_inventory(
        model_id="model", revision="immutable-revision"
    )
    changed = _manager(semantics=_Semantics(describe)).placement_inventory(
        model_id="model", revision="immutable-revision"
    )

    assert baseline.inventory_id != changed.inventory_id


def test_manager_never_infers_logical_aliases_from_private_view_offsets() -> None:
    manager = _manager(
        semantics=_Semantics(
            lambda *_args: (
                _view(tensor_id="logical.left"),
                _view(tensor_id="logical.right"),
            )
        )
    )

    placement = manager.placement_inventory(
        model_id="model",
        revision="immutable-revision",
    )

    assert {fragment.aliases for fragment in placement.fragments} == {()}


def test_semantics_can_declare_logical_aliases_across_private_view_offsets() -> None:
    aliases = ("logical.left", "logical.right")
    manager = _manager(
        model=_FakeModel((("runtime.packed", _FakeTensor((8, 4))),)),
        semantics=_Semantics(
            lambda *_args: (
                _view(tensor_id="logical.left", aliases=aliases),
                _view(
                    tensor_id="logical.right",
                    byte_offset=32,
                    aliases=aliases,
                ),
            )
        ),
    )

    placement = manager.placement_inventory(
        model_id="model",
        revision="immutable-revision",
    )

    assert {fragment.aliases for fragment in placement.fragments} == {aliases}


def test_content_commit_changes_inventory_not_fragment_or_participant() -> None:
    coordinator = WeightSnapshotCoordinator()
    manager = _manager(coordinator=coordinator)
    old = manager.placement_inventory(model_id="model", revision="immutable-revision")
    token = coordinator.begin_update()
    runtime_generation = coordinator.finish_update(token, success=True)
    coordinator.commit_weight_generation(expected_generation=runtime_generation)
    current = manager.placement_inventory(
        model_id="model", revision="immutable-revision"
    )

    assert old.weight_generation == 1
    assert current.weight_generation == 2
    assert old.participant_id == current.participant_id
    assert old.inventory_id != current.inventory_id
    assert old.fragments == current.fragments

    with pytest.raises(WeightInventoryError, match="not locally committed"):
        manager.binding_inventory(
            placement=old,
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:1234",
        )


@pytest.mark.parametrize("revision", ("", "default"))
def test_inventory_entrypoints_reject_ambiguous_content_identity(revision) -> None:
    manager = _manager()

    with pytest.raises(ValueError, match="content-lineage revision"):
        manager.snapshot_inventories(
            model_id="same/checkpoint/path",
            revision=revision,
            instance_id="instance",
            worker_id="worker",
            endpoint="worker:1234",
        )
    with pytest.raises(ValueError, match="content-lineage revision"):
        manager.placement_inventory(
            model_id="same/checkpoint/path",
            revision=revision,
        )
    with pytest.raises(ValueError, match="content-lineage revision"):
        manager.target_layout_inventory(
            model_id="same/checkpoint/path",
            revision=revision,
            desired_weight_generation=1,
        )


def test_restart_at_same_path_requires_new_immutable_revision_for_new_content() -> None:
    first = _manager(
        model=_FakeModel((("runtime.weight", _FakeTensor((4, 4), address=0x1000)),))
    )
    replaced = _manager(
        model=_FakeModel((("runtime.weight", _FakeTensor((4, 4), address=0x9000)),))
    )

    first_placement = first.placement_inventory(
        model_id="same/checkpoint/path",
        revision="checkpoint-sha-a",
    )
    replaced_placement = replaced.placement_inventory(
        model_id="same/checkpoint/path",
        revision="checkpoint-sha-b",
    )

    assert first_placement.participant_id != replaced_placement.participant_id
    assert first_placement.inventory_id != replaced_placement.inventory_id


def test_target_layout_can_describe_desired_uncommitted_generation() -> None:
    manager = _manager()
    placement = manager.target_layout_inventory(
        model_id="model",
        revision="immutable-revision",
        desired_weight_generation=7,
    )
    binding = manager.target_binding_inventory(
        placement=placement,
        instance_id="target-instance",
        worker_id="target-worker",
        endpoint="target:1234",
    )
    try:
        assert placement.weight_generation == 7
        assert binding.placement_inventory_id == placement.inventory_id
    finally:
        manager.release(binding.lease_id)


def test_same_logical_generation_adoption_invalidates_runtime_binding_fence() -> None:
    coordinator = WeightSnapshotCoordinator()
    manager = _manager(coordinator=coordinator)
    first = _snapshot(manager, suffix="before")
    placement_id = first.placement.inventory_id
    first_generation = first.binding.generation
    manager.release(first.binding.lease_id)

    coordinator.adopt_weight_generation(1)
    second = _snapshot(manager, suffix="after")
    try:
        assert second.placement.inventory_id == placement_id
        assert second.binding.generation == first_generation + 1
        assert (
            second.binding.fragments[0].fragment_id
            != first.binding.fragments[0].fragment_id
        )
    finally:
        manager.release(second.binding.lease_id)


def test_active_snapshot_blocks_update_and_generation_adoption() -> None:
    coordinator = WeightSnapshotCoordinator()
    manager = _manager(coordinator=coordinator)
    snapshot = _snapshot(manager)
    try:
        with pytest.raises(WeightInventoryError, match="lease is active"):
            coordinator.begin_update()
        with pytest.raises(WeightInventoryError, match="active lease"):
            coordinator.adopt_weight_generation(7)
    finally:
        manager.release(snapshot.binding.lease_id)


def test_target_snapshot_activation_atomically_releases_lease() -> None:
    coordinator = WeightSnapshotCoordinator()
    manager = _manager(coordinator=coordinator)
    placement = manager.target_layout_inventory(
        model_id="model",
        revision="immutable-revision",
        desired_weight_generation=7,
    )
    binding = manager.target_binding_inventory(
        placement=placement,
        instance_id="target-instance",
        worker_id="target-worker",
        endpoint="target:1234",
    )

    generation = coordinator.adopt_weight_generation_from_snapshot(
        binding.lease_id,
        placement.weight_generation,
    )

    assert generation == 2
    assert coordinator.weight_generation == 7
    assert manager.has_lease(binding.lease_id) is False


def test_target_snapshot_activation_rejects_unrelated_active_lease() -> None:
    coordinator = WeightSnapshotCoordinator()
    manager = _manager(coordinator=coordinator)
    first = _snapshot(manager, suffix="first")
    second = _snapshot(manager, suffix="second")

    try:
        with pytest.raises(WeightInventoryError, match="other active leases"):
            coordinator.adopt_weight_generation_from_snapshot(
                first.binding.lease_id,
                7,
            )
        assert manager.has_lease(first.binding.lease_id)
        assert manager.has_lease(second.binding.lease_id)
        assert coordinator.weight_generation == 1
    finally:
        manager.release(first.binding.lease_id)
        manager.release(second.binding.lease_id)


def test_failed_update_never_publishes_new_logical_generation() -> None:
    coordinator = WeightSnapshotCoordinator()
    token = coordinator.begin_update()
    coordinator.finish_update(token, success=False)

    assert coordinator.weight_generation == 1
    with pytest.raises(WeightInventoryError, match="full successful weight restore"):
        coordinator.commit_weight_generation()
    with pytest.raises(WeightInventoryError):
        coordinator.acquire_snapshot()


@pytest.mark.parametrize(
    "topology",
    (
        _topology(dp_size=2),
        _topology(tp_size=2, attention_tp_size=1),
        _topology(tp_size=4, ep_size=2, moe_tp_size=2, attention_tp_size=4),
    ),
)
def test_inventory_rejects_unrepresentable_sglang_topology(topology) -> None:
    with pytest.raises(WeightInventoryError):
        _manager(topology=topology)


def test_factory_accepts_real_tp2_ep2_and_rejects_impossible_moe_product() -> None:
    config = SimpleNamespace(
        model_type="qwen3_moe",
        num_experts=8,
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
    )
    model = _FakeModel((("model.norm.weight", _FakeTensor((4,))),))
    supported = create_weight_inventory_manager(
        model=model,
        config=config,
        topology=_topology(
            tp_size=2,
            ep_size=2,
            attention_tp_size=2,
        ),
        allowed_devices=("cpu",),
    )
    impossible = create_weight_inventory_manager(
        model=model,
        config=config,
        topology=_topology(
            tp_size=4,
            ep_size=4,
            moe_tp_size=4,
            attention_tp_size=4,
        ),
        allowed_devices=("cpu",),
    )

    assert type(supported).__name__ == "WeightInventoryManager"
    with pytest.raises(WeightInventoryError, match="global TP = EP"):
        impossible.placement_inventory(model_id="model", revision="immutable-revision")


def test_topology_factory_preserves_real_sglang_rank_decomposition() -> None:
    topology = topology_from_sglang(
        parallel_state=SimpleNamespace(
            dp_rank=None,
            dp_size=1,
            moe_dp_rank=0,
            moe_dp_size=1,
            tp_rank=1,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            moe_ep_rank=1,
            moe_ep_size=2,
            attn_tp_rank=1,
            attn_tp_size=2,
        ),
        parallel=SimpleNamespace(moe_tp_rank=0, moe_tp_size=1),
    )

    assert topology.rank() == topology.rank().__class__(tp=1, ep=1)
    assert topology.tp_size == topology.ep_size == 2
    assert topology.moe_tp_size == 1


def test_qwen_moe_semantics_emit_explicit_ep_and_tp_axes() -> None:
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_experts=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
    )
    adapter = Qwen35WeightSemanticsAdapter(
        config=config,
        dynamic_expert_placement=False,
        up_first_w13_parameter_ids=frozenset(),
    )
    parameter = _FakeTensor((4, 16, 4))
    views = adapter.describe_parameter(
        names=("model.layers.0.mlp.experts.w13_weight",),
        parameter=parameter,
        topology=_topology(
            tp_rank=1,
            tp_size=2,
            ep_rank=1,
            ep_size=2,
            attention_tp_rank=1,
            attention_tp_size=2,
        ),
    )

    assert views
    assert all(
        {axis.kind for axis in view.parallel_axes} == {"dp", "tp", "pp", "ep"}
        for view in views
    )
    assert all(
        any(axis.kind == "ep" and axis.mode == "split" for axis in view.parallel_axes)
        for view in views
    )
    assert all(not hasattr(view, "partition_dim") for view in views)


def _qwen35_multimodal_adapter(*, vision_data_parallel: bool):
    return Qwen35MultimodalWeightSemanticsAdapter(
        text_config=SimpleNamespace(
            hidden_size=4,
            intermediate_size=8,
            num_experts=0,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=2,
        ),
        vision_config=SimpleNamespace(
            hidden_size=4,
            num_position_embeddings=8,
        ),
        vision_data_parallel=vision_data_parallel,
    )


def test_qwen35_mm_dp_vision_full_tensor_is_explicit_tp_replica() -> None:
    views = _qwen35_multimodal_adapter(vision_data_parallel=True).describe_parameter(
        names=("visual.pos_embed.weight",),
        parameter=_FakeTensor((8, 4)),
        topology=_topology(
            tp_rank=1,
            tp_size=2,
            attention_tp_rank=1,
            attention_tp_size=2,
        ),
    )

    assert len(views) == 1
    assert views[0].local_shape == views[0].global_shape == (8, 4)
    assert views[0].shard_dims == ()
    assert {axis.kind: axis for axis in views[0].parallel_axes}["tp"].mode == (
        "replicated"
    )


def test_qwen35_tp_vision_rejects_full_tensor_instead_of_guessing_split() -> None:
    with pytest.raises(WeightInventoryError, match="explicit vision TP"):
        _qwen35_multimodal_adapter(vision_data_parallel=False).describe_parameter(
            names=("visual.pos_embed.weight",),
            parameter=_FakeTensor((8, 4)),
            topology=_topology(
                tp_rank=1,
                tp_size=2,
                attention_tp_rank=1,
                attention_tp_size=2,
            ),
        )


def _qwen35_fused_shared_expert_manager(*, runtime_fusion_evidence: bool):
    config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
    )
    modules = (
        (SimpleNamespace(num_fused_shared_experts=1),)
        if runtime_fusion_evidence
        else ()
    )
    model = _FakeModel(
        (
            (
                "model.layers.0.mlp.experts.w13_weight",
                _FakeTensor((5, 16, 4), address=0x1000),
            ),
            (
                "model.layers.0.mlp.experts.w2_weight",
                _FakeTensor((5, 4, 8), address=0x2000),
            ),
        ),
        modules=modules,
    )
    return create_weight_inventory_manager(
        model=model,
        config=config,
        topology=_topology(
            tp_rank=1,
            tp_size=2,
            ep_rank=1,
            ep_size=2,
            attention_tp_rank=1,
            attention_tp_size=2,
        ),
        allowed_devices=("cpu",),
    )


def test_qwen35_fused_shared_expert_is_not_described_as_routed_expert() -> None:
    manager = _qwen35_fused_shared_expert_manager(runtime_fusion_evidence=True)

    placement = manager.placement_inventory(
        model_id="model",
        revision="immutable-revision",
    )

    by_tensor = {}
    for fragment in placement.fragments:
        by_tensor.setdefault(fragment.tensor_id, []).append(fragment)

    routed_prefix = "layers.0.mlp.experts."
    shared_prefix = "layers.0.mlp.shared_expert."
    assert set(by_tensor) == {
        f"{routed_prefix}gate_proj.weight",
        f"{routed_prefix}up_proj.weight",
        f"{routed_prefix}down_proj.weight",
        f"{shared_prefix}gate_proj.weight",
        f"{shared_prefix}up_proj.weight",
        f"{shared_prefix}down_proj.weight",
    }
    assert all(
        len(by_tensor[f"{routed_prefix}{component}.weight"]) == 4
        for component in ("gate_proj", "up_proj", "down_proj")
    )

    shared_shapes = {
        fragment.tensor_id: fragment.global_shape
        for tensor_id, fragments in by_tensor.items()
        if tensor_id.startswith(shared_prefix)
        for fragment in fragments
    }
    assert shared_shapes == {
        f"{shared_prefix}gate_proj.weight": (8, 4),
        f"{shared_prefix}up_proj.weight": (8, 4),
        f"{shared_prefix}down_proj.weight": (4, 8),
    }
    assert all(
        fragment.expert_id is None
        and fragment.global_offset[0] == 0
        and {axis.kind: axis for axis in fragment.parallel_axes}["ep"].mode
        not in {"split", "ownership"}
        for tensor_id, fragments in by_tensor.items()
        if tensor_id.startswith(shared_prefix)
        for fragment in fragments
    )


def test_qwen35_fused_shape_without_runtime_evidence_remains_rejected() -> None:
    manager = _qwen35_fused_shared_expert_manager(runtime_fusion_evidence=False)

    with pytest.raises(WeightInventoryError, match="local expert count mismatch"):
        manager.placement_inventory(
            model_id="model",
            revision="immutable-revision",
        )


@pytest.mark.parametrize("invalid_count", (True, 1.5, "1", -1))
def test_qwen35_fused_shared_expert_rejects_invalid_runtime_count(
    invalid_count,
) -> None:
    model = SimpleNamespace(
        modules=lambda: (SimpleNamespace(num_fused_shared_experts=invalid_count),)
    )

    with pytest.raises(WeightInventoryError, match="fused shared expert count"):
        _runtime_num_fused_shared_experts(model)


def test_qwen_mqa_kv_views_are_explicit_complete_tp_replicas() -> None:
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=2,
        attn_output_gate=False,
    )
    adapter = Qwen35WeightSemanticsAdapter(config=config)
    views = adapter.describe_parameter(
        names=("model.layers.0.self_attn.qkv_proj.weight",),
        parameter=_FakeTensor((8, 8)),
        topology=_topology(
            tp_rank=1,
            tp_size=2,
            attention_tp_rank=1,
            attention_tp_size=2,
        ),
    )

    q_view, k_view, v_view = views
    assert q_view.shard_dims == (0,)
    for view in (k_view, v_view):
        assert view.global_offset == (0, 0)
        assert view.local_shape == view.global_shape == (2, 8)
        assert view.shard_dims == ()
        assert {axis.kind: axis for axis in view.parallel_axes}["tp"].mode == (
            "replicated"
        )


def test_qwen_gqa_partial_kv_replication_fails_before_canonical_translation() -> None:
    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=1,
        attn_output_gate=False,
    )
    adapter = Qwen35WeightSemanticsAdapter(config=config)

    with pytest.raises(WeightInventoryError, match="partially replicated"):
        adapter.describe_parameter(
            names=("model.layers.0.self_attn.qkv_proj.weight",),
            parameter=_FakeTensor((6, 8)),
            topology=_topology(
                tp_rank=1,
                tp_size=4,
                attention_tp_rank=1,
                attention_tp_size=4,
            ),
        )


def test_qwen_moe_tp_only_emits_explicit_replicated_ep_axis() -> None:
    topology = _topology(
        tp_rank=1,
        tp_size=2,
        ep_size=1,
        moe_tp_rank=1,
        moe_tp_size=2,
        attention_tp_rank=1,
        attention_tp_size=2,
    )
    for helper in (_moe_parallel_semantics, _shared_moe_parallel_semantics):
        _, axes = helper(topology, tp_dim=1)
        by_kind = {axis.kind: axis for axis in axes}
        assert set(by_kind) == {"dp", "tp", "pp", "ep"}
        assert by_kind["ep"].mode == "replicated"

    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=8,
        moe_intermediate_size=8,
        num_experts=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=2,
    )
    model = _FakeModel(
        (("model.layers.0.mlp.experts.w13_weight", _FakeTensor((8, 8, 4))),)
    )
    manager = WeightInventoryManager(
        model=model,
        adapter=Qwen35WeightSemanticsAdapter(config=config),
        topology=topology,
        allowed_devices=("cpu",),
    )

    placement = manager.placement_inventory(
        model_id="model",
        revision="immutable-revision",
    )

    assert placement.fragments
    for fragment in placement.fragments:
        axes = {axis.kind: axis for axis in fragment.parallel_axes}
        assert axes["ep"].mode == "replicated"
        assert axes["tp"].mode == "split"


def test_qwen3_next_grouped_gdn_uses_explicit_attention_tp_axis() -> None:
    config = SimpleNamespace(
        num_experts=0,
        hidden_size=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=1,
        linear_value_head_dim=1,
    )
    adapter = Qwen3NextWeightSemanticsAdapter(
        config=config,
        dynamic_expert_placement=False,
        up_first_w13_parameter_ids=frozenset(),
        num_fused_shared_experts=0,
    )
    parameter = _FakeTensor((1, 2, 4))
    parameter._sglang_qwen3_next_gdn_layout = "grouped"
    views = adapter.describe_parameter(
        names=("model.layers.0.linear_attn.in_proj_ba.weight",),
        parameter=parameter,
        topology=_topology(
            tp_rank=1,
            tp_size=2,
            attention_tp_rank=1,
            attention_tp_size=2,
        ),
    )

    assert views[0].shard_dims == (0,)
    assert any(
        axis.kind == "tp" and axis.mode == "split" and axis.dim == 0
        for axis in views[0].parallel_axes
    )


def test_fp8_retag_preserves_explicit_axes_without_legacy_partition_fact() -> None:
    original = _view(shard_dims=(0,))
    retagged = _retag_weight_view(original)

    assert retagged.parallel_axes == original.parallel_axes
    assert retagged.shard_dims == original.shard_dims
    assert "serialized-block-fp8" in retagged.layout_fingerprint
    assert not hasattr(retagged, "partition_dim")
