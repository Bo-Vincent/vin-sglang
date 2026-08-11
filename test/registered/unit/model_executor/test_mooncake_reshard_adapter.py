from __future__ import annotations

import hashlib
from math import prod
from types import SimpleNamespace

import msgspec
import pytest

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.mooncake_reshard_adapter import (
    MooncakeCanonicalReshardAdapter,
    PlacementInventoryParticipant,
)
from sglang.srt.model_executor.weight_inventory import WeightInventoryManager
from sglang.srt.model_executor.weight_inventory_contracts import (
    LogicalParallelAxis,
    LogicalTensorView,
    WeightParallelTopology,
    WeightPlacementInventory,
    WeightRuntimeBindingInventory,
    _participant_id,
    _placement_id,
)
from sglang.srt.model_executor.weight_snapshot import WeightSnapshotCoordinator


class _FakeStorage:
    def __init__(self, address: int, nbytes: int) -> None:
        self._address = address
        self._nbytes = nbytes

    def data_ptr(self) -> int:
        return self._address

    def nbytes(self) -> int:
        return self._nbytes


class _FakeTensor:
    def __init__(
        self,
        shape,
        *,
        address: int = 0x1020,
        storage_address: int = 0x1000,
        storage_nbytes: int = 0x400,
        storage_offset: int = 16,
        itemsize: int = 2,
    ) -> None:
        self.shape = tuple(shape)
        self.dtype = "torch.bfloat16"
        self.device = SimpleNamespace(type="cpu")
        self.layout = "torch.strided"
        self.is_sparse = False
        self._address = address
        self._storage = _FakeStorage(storage_address, storage_nbytes)
        self._storage_offset = storage_offset
        self._itemsize = itemsize

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

    def untyped_storage(self) -> _FakeStorage:
        return self._storage


class _FakeModel:
    def __init__(self, parameters) -> None:
        self._parameters = tuple(parameters)

    def named_parameters(self, *, remove_duplicate: bool):
        assert remove_duplicate is False
        return iter(self._parameters)


class _Capture:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class _FakeParallelTopology(_Capture):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.topology_id = (
            "topology:" + hashlib.sha256(repr(kwargs).encode()).hexdigest()[:16]
        )


class _FakeWeightPlacementManifest(_Capture):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        identity = (
            kwargs["resource_id"],
            kwargs["revision"],
            kwargs["weight_generation"],
            kwargs["placement_set_id"],
        )
        digest = hashlib.sha256(repr(identity).encode()).hexdigest()
        self.placement_id = f"placement:{digest[:24]}"
        self.digest = digest


def _fake_contracts(
    *,
    capabilities=(
        "placement_binding",
        "nd_logical_box",
        "dependent_axis_projection",
        "te_execution",
    ),
):
    return SimpleNamespace(
        OwnershipAxis=type("OwnershipAxis", (_Capture,), {}),
        ParallelRank=type("ParallelRank", (_Capture,), {}),
        ParallelTopology=_FakeParallelTopology,
        PlacementFragment=type("PlacementFragment", (_Capture,), {}),
        ReplicatedAxis=type("ReplicatedAxis", (_Capture,), {}),
        RuntimeBindingFragment=type("RuntimeBindingFragment", (_Capture,), {}),
        SplitAxis=type("SplitAxis", (_Capture,), {}),
        TensorDescriptor=type("TensorDescriptor", (_Capture,), {}),
        TopologyParticipant=type("TopologyParticipant", (_Capture,), {}),
        WeightPlacementManifest=_FakeWeightPlacementManifest,
        WeightPlacementPart=type("WeightPlacementPart", (_Capture,), {}),
        WeightRuntimeBindingManifest=type(
            "WeightRuntimeBindingManifest", (_Capture,), {}
        ),
        supports_weight_reshard_capability=lambda capability: (
            capability in capabilities
        ),
    )


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
            coupled_to="tp" if ep_mode == "coupled" else None,
        ),
    )


def _view(
    *,
    tensor_id="weight",
    global_shape=(4, 4),
    global_offset=(0, 0),
    local_shape=(4, 4),
    shard_dims=(),
    parallel_axes=None,
    byte_offset=0,
    layer_id=None,
    expert_id=None,
    layout_fingerprint="test:canonical-adapter",
):
    return LogicalTensorView(
        tensor_id=tensor_id,
        global_shape=global_shape,
        global_offset=global_offset,
        local_shape=local_shape,
        byte_offset=byte_offset,
        layer_id=layer_id,
        expert_id=expert_id,
        layout_fingerprint=layout_fingerprint,
        shard_dims=tuple(shard_dims),
        parallel_axes=parallel_axes or _axes(shard_dims),
    )


class _StaticSemantics:
    def __init__(self, view_factory) -> None:
        self._view_factory = view_factory

    def describe_parameter(self, *, names, parameter, topology):
        value = self._view_factory(names, parameter, topology)
        return value if isinstance(value, tuple) else (value,)


def _manager(
    *,
    topology=None,
    view_factory=lambda *_args: _view(),
    coordinator=None,
    parameters=None,
) -> WeightInventoryManager:
    return WeightInventoryManager(
        model=_FakeModel(parameters or (("runtime.weight", _FakeTensor((4, 4))),)),
        adapter=_StaticSemantics(view_factory),
        topology=topology or _topology(),
        allowed_devices=("cpu",),
        coordinator=coordinator,
    )


def _snapshot(manager, *, suffix="0", revision="immutable-revision"):
    return manager.snapshot_inventories(
        model_id="model",
        revision=revision,
        instance_id=f"instance-{suffix}",
        worker_id=f"worker-{suffix}",
        endpoint=f"worker-{suffix}:1234",
    )


def _axis_signature(descriptor):
    return tuple(
        (type(axis).__name__, axis.kind, getattr(axis, "dim", None))
        for axis in descriptor.parallel_axes
    )


def test_adapter_is_lazy_and_mooncake_remains_optional(monkeypatch) -> None:
    import sglang.srt.model_executor.mooncake_reshard_adapter as adapter_module

    calls = []

    def fake_import(name):
        calls.append(name)
        raise ImportError("Mooncake is intentionally absent")

    monkeypatch.setattr(adapter_module.importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError, match="Mooncake reshard"):
        adapter_module.load_mooncake_reshard_contracts()
    assert calls == ["mooncake.reshard.weight"]


@pytest.mark.parametrize(
    "missing_capability",
    (
        "placement_binding",
        "nd_logical_box",
        "dependent_axis_projection",
        "te_execution",
    ),
)
def test_adapter_rejects_incomplete_mooncake_runtime_capability(
    monkeypatch,
    missing_capability,
) -> None:
    import sglang.srt.model_executor.mooncake_reshard_adapter as adapter_module

    capabilities = {
        "placement_binding",
        "nd_logical_box",
        "dependent_axis_projection",
        "te_execution",
    } - {missing_capability}
    contracts = _fake_contracts(capabilities=capabilities)
    monkeypatch.setattr(
        adapter_module.importlib,
        "import_module",
        lambda name: contracts,
    )

    with pytest.raises(RuntimeError, match=missing_capability):
        adapter_module.load_mooncake_reshard_contracts()


def test_producer_wire_adapter_canonical_conformance() -> None:
    manager = _manager()
    inventories = _snapshot(manager)
    try:
        placement, bindings = MooncakeCanonicalReshardAdapter(
            contracts=_fake_contracts()
        ).source_placement_and_bindings(
            (msgspec.to_builtins(inventories.placement),),
            (msgspec.to_builtins(inventories.binding),),
        )
    finally:
        manager.release(inventories.binding.lease_id)

    assert placement.resource_id == "model"
    assert placement.weight_generation == 1
    assert len(placement.parts) == 1
    assert bindings[0].generation == 1


def test_adapter_rejects_restart_ambiguous_default_content_identity() -> None:
    manager = _manager()
    inventories = _snapshot(manager)
    try:
        participant_id = _participant_id(
            model_id=inventories.placement.model_id,
            revision="default",
            topology=inventories.placement.topology,
        )
        inventory_id = _placement_id(
            model_id=inventories.placement.model_id,
            revision="default",
            weight_generation=inventories.placement.weight_generation,
            topology=inventories.placement.topology,
            fragments=inventories.placement.fragments,
        )
        placement = WeightPlacementInventory(
            model_id=inventories.placement.model_id,
            revision="default",
            weight_generation=inventories.placement.weight_generation,
            inventory_id=inventory_id,
            participant_id=participant_id,
            topology=inventories.placement.topology,
            fragments=inventories.placement.fragments,
        )
        binding = WeightRuntimeBindingInventory(
            model_id=inventories.binding.model_id,
            revision="default",
            placement_inventory_id=inventory_id,
            instance_id=inventories.binding.instance_id,
            generation=inventories.binding.generation,
            lease_id=inventories.binding.lease_id,
            participant_id=participant_id,
            fragments=inventories.binding.fragments,
        )
        with pytest.raises(ValueError, match="content-lineage revision"):
            MooncakeCanonicalReshardAdapter(
                contracts=_fake_contracts()
            ).source_placement_and_bindings(
                (placement,),
                (binding,),
            )
    finally:
        manager.release(inventories.binding.lease_id)


@pytest.mark.parametrize(
    ("side", "path", "field"),
    (
        ("placement", (), "format_version"),
        ("placement", ("fragments", 0), "runtime_name"),
        ("placement", ("fragments", 0), "byte_offset"),
        ("placement", ("fragments", 0), "partition_dim"),
        ("binding", ("fragments", 0), "unexpected"),
    ),
)
def test_wire_adapter_rejects_unknown_and_legacy_fields(side, path, field) -> None:
    manager = _manager()
    inventories = _snapshot(manager)
    try:
        placement = msgspec.to_builtins(inventories.placement)
        binding = msgspec.to_builtins(inventories.binding)
    finally:
        manager.release(inventories.binding.lease_id)
    target = placement if side == "placement" else binding
    for item in path:
        target = target[item]
    target[field] = "retired-or-unknown"

    with pytest.raises(ValueError, match="invalid SGLang weight"):
        MooncakeCanonicalReshardAdapter(
            contracts=_fake_contracts()
        ).source_placement_and_bindings((placement,), (binding,))


def test_wire_adapter_rejects_binding_unit_conflict() -> None:
    manager = _manager()
    inventories = _snapshot(manager)
    try:
        placement = msgspec.to_builtins(inventories.placement)
        binding = msgspec.to_builtins(inventories.binding)
    finally:
        manager.release(inventories.binding.lease_id)
    binding["fragments"][0]["storage_offset"] += 1

    with pytest.raises(ValueError, match="invalid SGLang weight runtime binding"):
        MooncakeCanonicalReshardAdapter(
            contracts=_fake_contracts()
        ).source_placement_and_bindings((placement,), (binding,))


def test_adapter_accepts_mixed_local_binding_generations() -> None:
    first_coordinator = WeightSnapshotCoordinator()
    second_coordinator = WeightSnapshotCoordinator()
    second_coordinator.adopt_weight_generation(1)
    managers = (
        _manager(
            topology=_topology(
                tp_rank=0,
                tp_size=2,
                attention_tp_rank=0,
                attention_tp_size=2,
            ),
            coordinator=first_coordinator,
            view_factory=lambda _n, _p, topology: _view(
                global_shape=(8, 4),
                global_offset=(topology.tp_rank * 4, 0),
                local_shape=(4, 4),
                shard_dims=(0,),
            ),
        ),
        _manager(
            topology=_topology(
                tp_rank=1,
                tp_size=2,
                attention_tp_rank=1,
                attention_tp_size=2,
            ),
            coordinator=second_coordinator,
            view_factory=lambda _n, _p, topology: _view(
                global_shape=(8, 4),
                global_offset=(topology.tp_rank * 4, 0),
                local_shape=(4, 4),
                shard_dims=(0,),
            ),
        ),
    )
    inventories = tuple(
        _snapshot(manager, suffix=str(index)) for index, manager in enumerate(managers)
    )
    try:
        placement, bindings = MooncakeCanonicalReshardAdapter(
            contracts=_fake_contracts()
        ).source_placement_and_bindings(
            tuple(item.placement for item in inventories),
            tuple(item.binding for item in inventories),
        )
    finally:
        for manager, item in zip(managers, inventories):
            manager.release(item.binding.lease_id)

    assert placement.weight_generation == 1
    assert {binding.generation for binding in bindings} == {1, 2}


def test_adapter_rejects_mixed_logical_weight_generations() -> None:
    first = _manager(
        topology=_topology(tp_size=2, attention_tp_size=2),
        view_factory=lambda _n, _p, topology: _view(
            global_shape=(8, 4),
            global_offset=(topology.tp_rank * 4, 0),
            local_shape=(4, 4),
            shard_dims=(0,),
        ),
    )
    second_coordinator = WeightSnapshotCoordinator()
    second_coordinator.adopt_weight_generation(7)
    second = _manager(
        topology=_topology(
            tp_rank=1,
            tp_size=2,
            attention_tp_rank=1,
            attention_tp_size=2,
        ),
        coordinator=second_coordinator,
        view_factory=lambda _n, _p, topology: _view(
            global_shape=(8, 4),
            global_offset=(topology.tp_rank * 4, 0),
            local_shape=(4, 4),
            shard_dims=(0,),
        ),
    )
    inventories = (_snapshot(first, suffix="0"), _snapshot(second, suffix="1"))
    try:
        with pytest.raises(ValueError, match="weight generations differ"):
            MooncakeCanonicalReshardAdapter(
                contracts=_fake_contracts()
            ).source_placement_and_bindings(
                tuple(item.placement for item in inventories),
                tuple(item.binding for item in inventories),
            )
    finally:
        first.release(inventories[0].binding.lease_id)
        second.release(inventories[1].binding.lease_id)


def test_adapter_rejects_incomplete_sglang_participants() -> None:
    manager = _manager(
        topology=_topology(tp_size=2, attention_tp_size=2),
        view_factory=lambda *_args: _view(
            global_shape=(8, 4),
            local_shape=(4, 4),
            shard_dims=(0,),
        ),
    )
    placement = manager.placement_inventory(
        model_id="model", revision="immutable-revision"
    )

    with pytest.raises(ValueError, match="do not cover the SGLang model world"):
        MooncakeCanonicalReshardAdapter(contracts=_fake_contracts()).placement_manifest(
            (PlacementInventoryParticipant(placement),)
        )


def test_qwen_tp2_ep2_is_representable_with_explicit_ep_coupling() -> None:
    participants = []
    for rank in range(2):
        topology = _topology(
            tp_rank=rank,
            tp_size=2,
            ep_rank=rank,
            ep_size=2,
            attention_tp_rank=rank,
            attention_tp_size=2,
        )
        manager = _manager(
            topology=topology,
            view_factory=lambda _n, _p, topology: _view(
                global_shape=(8, 4),
                global_offset=(topology.tp_rank * 4, 0),
                local_shape=(4, 4),
                shard_dims=(0,),
                parallel_axes=_axes((0,), ep_mode="coupled"),
            ),
        )
        participants.append(
            PlacementInventoryParticipant(
                manager.placement_inventory(
                    model_id="model", revision="immutable-revision"
                )
            )
        )

    manifest = MooncakeCanonicalReshardAdapter(
        contracts=_fake_contracts()
    ).placement_manifest(tuple(participants))

    for part in manifest.parts:
        assert all(axis.kind != "ep" for axis in part.tensors[0].parallel_axes)


def test_adapter_rejects_independent_ep_replication_for_co_mapped_tp_ep() -> None:
    participants = []
    for rank in range(2):
        topology = _topology(
            tp_rank=rank,
            tp_size=2,
            ep_rank=rank,
            ep_size=2,
            attention_tp_rank=rank,
            attention_tp_size=2,
        )
        manager = _manager(
            topology=topology,
            view_factory=lambda _n, _p, topology: _view(
                global_shape=(8, 4),
                global_offset=(topology.tp_rank * 4, 0),
                local_shape=(4, 4),
                shard_dims=(0,),
                parallel_axes=_axes((0,), ep_mode="replicated"),
            ),
        )
        participants.append(
            PlacementInventoryParticipant(
                manager.placement_inventory(
                    model_id="model",
                    revision="immutable-revision",
                )
            )
        )

    with pytest.raises(ValueError, match="coupled TP/EP"):
        MooncakeCanonicalReshardAdapter(contracts=_fake_contracts()).placement_manifest(
            tuple(participants)
        )


def test_qwen_tp2_ep2_conforms_to_installed_mooncake_contracts() -> None:
    canonical = pytest.importorskip("mooncake.reshard.weight")
    participants = []
    for rank in range(2):
        topology = _topology(
            tp_rank=rank,
            tp_size=2,
            ep_rank=rank,
            ep_size=2,
            attention_tp_rank=rank,
            attention_tp_size=2,
        )
        manager = _manager(
            topology=topology,
            view_factory=lambda _n, _p, topology: _view(
                global_shape=(8, 4),
                global_offset=(topology.tp_rank * 4, 0),
                local_shape=(4, 4),
                shard_dims=(0,),
                parallel_axes=_axes((0,), ep_mode="coupled"),
            ),
        )
        participants.append(
            PlacementInventoryParticipant(
                manager.placement_inventory(
                    model_id="model",
                    revision="immutable-revision",
                )
            )
        )

    manifest = MooncakeCanonicalReshardAdapter(contracts=canonical).placement_manifest(
        tuple(participants)
    )

    assert manifest.topology.world_size == 2
    assert manifest.topology.tp_size == manifest.topology.ep_size == 2


def test_adapter_deduplicates_multi_fragment_descriptor() -> None:
    manager = _manager(
        parameters=(("packed.weight", _FakeTensor((8, 4))),),
        view_factory=lambda *_args: (
            _view(
                tensor_id="logical.weight",
                global_shape=(8, 4),
                global_offset=(0, 0),
                local_shape=(4, 4),
                shard_dims=(0,),
            ),
            _view(
                tensor_id="logical.weight",
                global_shape=(8, 4),
                global_offset=(4, 0),
                local_shape=(4, 4),
                shard_dims=(0,),
                byte_offset=32,
            ),
        ),
    )
    placement = manager.placement_inventory(
        model_id="model", revision="immutable-revision"
    )

    manifest = MooncakeCanonicalReshardAdapter(
        contracts=_fake_contracts()
    ).placement_manifest((PlacementInventoryParticipant(placement),))

    assert len(manifest.parts[0].tensors) == 1
    assert len(manifest.parts[0].fragments) == 2


@pytest.mark.parametrize("world_group", [object(), SimpleNamespace(world_size=True)])
def test_target_placement_rejects_implicit_or_non_integer_world_size(
    world_group,
) -> None:
    manager = _manager()
    placement = manager.placement_inventory(
        model_id="model", revision="immutable-revision"
    )

    with pytest.raises(ValueError, match="world_size"):
        MooncakeCanonicalReshardAdapter(
            contracts=_fake_contracts()
        ).gather_target_placement(
            PlacementInventoryParticipant(placement),
            world_group=world_group,
        )


def test_target_placement_requires_real_multi_rank_all_gather() -> None:
    manager = _manager()
    placement = manager.placement_inventory(
        model_id="model", revision="immutable-revision"
    )

    with pytest.raises(ValueError, match="all_gather_object"):
        MooncakeCanonicalReshardAdapter(
            contracts=_fake_contracts()
        ).gather_target_placement(
            PlacementInventoryParticipant(placement),
            world_group=SimpleNamespace(world_size=2),
        )


def test_inventory_rejects_conflicting_multi_fragment_descriptor() -> None:
    manager = _manager(
        parameters=(("packed.weight", _FakeTensor((8, 4))),),
        view_factory=lambda *_args: (
            _view(
                tensor_id="logical.weight",
                global_shape=(8, 4),
                global_offset=(0, 0),
                local_shape=(4, 4),
                shard_dims=(0,),
            ),
            _view(
                tensor_id="logical.weight",
                global_shape=(8, 4),
                global_offset=(4, 0),
                local_shape=(4, 4),
                shard_dims=(0,),
                byte_offset=32,
                layout_fingerprint="conflicting-layout",
            ),
        ),
    )

    with pytest.raises(ValueError, match="logical tensor descriptor"):
        manager.placement_inventory(model_id="model", revision="immutable-revision")


def test_content_generation_changes_placement_but_not_participant() -> None:
    first_coordinator = WeightSnapshotCoordinator()
    seventh_coordinator = WeightSnapshotCoordinator()
    seventh_coordinator.adopt_weight_generation(7)
    first = _manager(coordinator=first_coordinator).placement_inventory(
        model_id="model", revision="immutable-revision"
    )
    seventh = _manager(coordinator=seventh_coordinator).placement_inventory(
        model_id="model", revision="immutable-revision"
    )
    adapter = MooncakeCanonicalReshardAdapter(contracts=_fake_contracts())

    first_manifest = adapter.placement_manifest((PlacementInventoryParticipant(first),))
    seventh_manifest = adapter.placement_manifest(
        (PlacementInventoryParticipant(seventh),)
    )

    assert first.participant_id == seventh.participant_id
    assert first.inventory_id != seventh.inventory_id
    assert first_manifest.placement_id != seventh_manifest.placement_id


def test_adapter_interoperates_with_installed_mooncake_contracts() -> None:
    canonical = pytest.importorskip("mooncake.reshard.weight")
    manager = _manager()
    inventories = _snapshot(manager)
    try:
        placement, bindings = MooncakeCanonicalReshardAdapter(
            contracts=canonical
        ).source_placement_and_bindings(
            (msgspec.to_builtins(inventories.placement),),
            (msgspec.to_builtins(inventories.binding),),
        )
        canonical.validate_runtime_binding(placement, bindings[0])
    finally:
        manager.release(inventories.binding.lease_id)

    assert isinstance(placement, canonical.WeightPlacementManifest)
    assert isinstance(bindings[0], canonical.WeightRuntimeBindingManifest)


def test_source_inventory_renews_lease_after_address_validation() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner.model_config = SimpleNamespace(
        model_path="/node-a/cache/model",
        revision="immutable-revision",
    )
    runner.server_args = SimpleNamespace(
        get_weight_reshard_resource_id=lambda: "canonical-model"
    )
    inventories = SimpleNamespace(binding=SimpleNamespace(lease_id="lease-7"))
    events = []
    runner.remote_instance_weight_transporter = SimpleNamespace(
        session_id="source-session",
        worker_id="source-worker",
        validate_runtime_binding_inventory_addresses=lambda binding: events.append(
            ("validate", binding)
        ),
    )
    runner.get_weight_inventories = lambda **kwargs: (
        events.append(("capture", kwargs)),
        inventories,
    )[1]
    runner.renew_weight_inventory = lambda lease_id, lease_timeout_sec: events.append(
        ("renew", lease_id, lease_timeout_sec)
    )
    runner.release_weight_inventory = lambda lease_id: events.append(
        ("release", lease_id)
    )

    result = runner.get_remote_instance_weight_inventories(
        model_id="canonical-model",
        revision="immutable-revision",
        transfer_id="transfer-7",
        lease_timeout_sec=60,
    )

    assert result is inventories
    assert [event[0] for event in events] == ["capture", "validate", "renew"]
    assert events[0][1]["model_id"] == "canonical-model"
    assert events[-1] == ("renew", "lease-7", 60)
