from __future__ import annotations

from contextlib import contextmanager, nullcontext
from math import prod
from types import SimpleNamespace

import pytest

from sglang.srt.managers.io_struct import (
    BeginRemoteInstanceWeightTransferReqInput,
    ReleaseRemoteInstanceWeightTransferReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.weight_inventory import WeightInventoryManager
from sglang.srt.model_executor.weight_inventory_contracts import (
    LogicalParallelAxis,
    LogicalTensorView,
    WeightInventoryError,
    WeightParallelTopology,
)
from sglang.srt.model_executor.weight_snapshot import WeightSnapshotCoordinator
from sglang.srt.model_loader import loader as loader_module
from sglang.srt.model_loader.loader import RemoteInstanceModelLoader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    RemoteInstanceWeightTransferSession,
)
from sglang.srt.model_loader.weight_reshard_backend import (
    WeightReshardBackendError,
    WeightReshardBackendUnavailableError,
    WeightReshardCompletionUnknownError,
    WeightReshardExecutionResult,
)


class _World:
    world_size = 1
    rank_in_group = 0

    def all_gather_object(self, value):
        return [value]

    def broadcast_object(self, value, src):
        assert src == 0
        return value


class _ActivationPeerFailureWorld:
    world_size = 2
    rank_in_group = 0

    def all_gather_object(self, value):
        if type(value) is bool:
            return [value, False]
        return [
            value,
            {
                "success": False,
                "message": "rank 1 target activation failed",
            },
        ]

    def broadcast_object(self, value, src):
        assert src == 0
        return value


class _Storage:
    def data_ptr(self):
        return 0x1000

    def nbytes(self):
        return 32


class _Tensor:
    shape = (4, 4)
    dtype = "torch.float16"
    device = SimpleNamespace(type="cpu")
    layout = "torch.strided"
    is_sparse = False

    def data_ptr(self):
        return 0x1000

    def element_size(self):
        return 2

    def numel(self):
        return prod(self.shape)

    def is_contiguous(self):
        return True

    def stride(self):
        return (4, 1)

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return _Storage()


class _Model:
    def named_parameters(self, *, remove_duplicate):
        assert remove_duplicate is False
        return iter((("runtime.weight", _Tensor()),))


class _Fp8Tensor:
    device = SimpleNamespace(type="cpu")
    layout = "torch.strided"
    is_sparse = False

    def __init__(self, *, shape, dtype, itemsize, address) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self._itemsize = itemsize
        self._address = address
        self._storage = SimpleNamespace(
            data_ptr=lambda: address,
            nbytes=lambda: prod(self.shape) * itemsize,
        )

    def data_ptr(self):
        return self._address

    def element_size(self):
        return self._itemsize

    def numel(self):
        return prod(self.shape)

    def is_contiguous(self):
        return True

    def stride(self):
        result = []
        running = 1
        for extent in reversed(self.shape):
            result.append(running)
            running *= extent
        return tuple(reversed(result))

    def storage_offset(self):
        return 0

    def untyped_storage(self):
        return self._storage


class _SerializedBlockFp8Model:
    def __init__(self, *, serialized: bool = True) -> None:
        weight = _Fp8Tensor(
            shape=(128, 128),
            dtype="torch.float8_e4m3fn",
            itemsize=1,
            address=0x2000,
        )
        scale = _Fp8Tensor(
            shape=(1, 1),
            dtype="torch.float32",
            itemsize=4,
            address=0x12000,
        )
        quant_config = SimpleNamespace(
            activation_scheme="dynamic",
            is_checkpoint_fp8_serialized=serialized,
            use_mxfp8=False,
            weight_block_size=[128, 128],
        )
        self.quant_module = SimpleNamespace(
            block_quant=True,
            quant_method=SimpleNamespace(
                block_quant=True,
                is_checkpoint_fp8_serialized=serialized,
                quant_config=quant_config,
                use_marlin=False,
                weight_block_size=[128, 128],
            ),
            weight=weight,
            weight_scale_inv=scale,
        )

    def named_parameters(self, *, remove_duplicate):
        assert remove_duplicate is False
        prefix = "model.layers.0.mlp.down_proj"
        return iter(
            (
                (f"{prefix}.weight", self.quant_module.weight),
                (f"{prefix}.weight_scale_inv", self.quant_module.weight_scale_inv),
            )
        )

    def modules(self):
        return (self.quant_module,)


class _Semantics:
    def describe_parameter(self, *, names, parameter, topology):
        assert names == ("runtime.weight",)
        return (
            LogicalTensorView(
                tensor_id="layers.0.weight",
                global_shape=(4, 4),
                global_offset=(0, 0),
                local_shape=(4, 4),
                byte_offset=0,
                layer_id=0,
                expert_id=None,
                layout_fingerprint="logical-contiguous",
                shard_dims=(),
                parallel_axes=tuple(
                    LogicalParallelAxis(
                        kind=kind,
                        mode="ownership" if kind == "pp" else "replicated",
                    )
                    for kind in ("dp", "tp", "pp", "ep")
                ),
            ),
        )


class _InventoryRunner:
    def __init__(self, manager) -> None:
        self.manager = manager

    def get_remote_instance_weight_inventories(self, **kwargs):
        return self.manager.snapshot_inventories(
            model_id=kwargs["model_id"],
            revision=kwargs["revision"],
            instance_id="source-instance",
            worker_id="source-worker",
            endpoint="source:1234",
            lease_timeout_sec=kwargs["lease_timeout_sec"],
        )

    def release_weight_inventory(self, lease_id):
        self.manager.release(lease_id)


def _source_scheduler(monkeypatch):
    coordinator = WeightSnapshotCoordinator()
    coordinator.adopt_weight_generation(7)
    inventory_manager = WeightInventoryManager(
        model=_Model(),
        adapter=_Semantics(),
        topology=WeightParallelTopology(),
        allowed_devices=("cpu",),
        coordinator=coordinator,
    )
    scheduler = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(
            model_runner=_InventoryRunner(inventory_manager),
        ),
        draft_worker=None,
        tp_cpu_group=object(),
        world_cpu_group=object(),
        memory_saver_adapter=object(),
        flush_cache=lambda **_kwargs: True,
        is_fully_idle=lambda: True,
    )
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    output = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="production-chain",
            model_id="model",
            revision="checkpoint-sha",
            lease_timeout_sec=60,
        )
    )
    assert output.success is True
    return (
        scheduler,
        inventory_manager,
        RemoteInstanceWeightTransferSession(
            transfer_id=output.transfer_id,
            placement_inventories=output.placement_inventories,
            binding_inventories=output.binding_inventories,
            lease_timeout_sec=60,
        ),
    )


class _SourceCoordinator:
    instances = []
    forced_finish_result = None

    def __init__(self, seed_url, world_group) -> None:
        self.seed_url = seed_url
        self.world_group = world_group
        self.finish_calls = []
        self.ready_calls = []
        self.session = RemoteInstanceWeightTransferSession(
            transfer_id="source-transfer",
            placement_inventories=[{"weight_generation": 7}],
            binding_inventories=[{"generation": 13}],
            lease_timeout_sec=300,
        )
        self.__class__.instances.append(self)

    def acquire(self):
        return self.session

    def ready_for_transfer(self, local_ready):
        self.ready_calls.append(local_ready)
        return bool(local_ready)

    def finish(self, *, local_success, local_release_safe=True):
        self.finish_calls.append((local_success, local_release_safe))
        if self.forced_finish_result is not None:
            return self.forced_finish_result
        return bool(local_success), bool(local_release_safe)


class _Prepared:
    operation_count = 4
    nbytes = 1024

    def __init__(self) -> None:
        self.completion_unknown_retained = False
        self.closed = False


class _Backend:
    def __init__(self, *, outcome="success", activation_coordinator=None) -> None:
        self.outcome = outcome
        self.activation_coordinator = activation_coordinator
        self.prepared = _Prepared()
        self.events = []
        self.prepare_kwargs = None

    @contextmanager
    def prepare(self, **kwargs):
        self.prepare_kwargs = kwargs
        self.events.append("prepare")
        try:
            yield self.prepared
        finally:
            if not self.prepared.completion_unknown_retained:
                self.close_after_terminal(self.prepared)

    def execute(self, prepared, *, transfer_engine):
        self.events.append("execute")
        if self.outcome.startswith("unknown"):
            raise WeightReshardCompletionUnknownError(
                "completion unknown",
                pending_transfer_id="pending-1",
            )
        if self.outcome == "terminal-error":
            raise WeightReshardBackendError("terminal failure")
        if self.outcome == "unexpected-error":
            raise RuntimeError("backend violated the terminal-error contract")
        if self.outcome == "interrupted":
            raise KeyboardInterrupt
        if self.outcome == "missing-receipt":
            return None
        if self.outcome == "short-receipt":
            return WeightReshardExecutionResult(nbytes=512, segment_count=4)
        return WeightReshardExecutionResult(nbytes=1024, segment_count=4)

    def activate(self, prepared):
        self.events.append("activate")
        if self.outcome == "activation-error":
            raise WeightReshardBackendError("activation failed")
        if self.activation_coordinator is not None:
            self.activation_coordinator.adopt_weight_generation(7)

    def drain_pending_transfer(self, prepared, *, pending_transfer_id, timeout_ms):
        self.events.append("drain")
        return "COMPLETION_UNKNOWN" if self.outcome == "unknown" else "FAILED"

    def retain_completion_unknown(self, prepared):
        self.events.append("retain")
        prepared.completion_unknown_retained = True

    def close_after_terminal(self, prepared):
        if prepared.closed:
            return
        self.events.append("close")
        prepared.closed = True
        prepared.completion_unknown_retained = False


def _loader() -> RemoteInstanceModelLoader:
    result = RemoteInstanceModelLoader.__new__(RemoteInstanceModelLoader)
    result.load_config = SimpleNamespace(
        remote_instance_weight_loader_backend=(
            RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE
        )
    )
    return result


def test_loader_rejects_reshard_inventory_on_legacy_backend() -> None:
    loader = _loader()
    loader.load_config.remote_instance_weight_inventory_builder = object()
    loader.load_config.remote_instance_weight_loader_backend = (
        RemoteInstanceWeightLoaderBackend.NCCL
    )

    with pytest.raises(ValueError, match="requires the transfer_engine backend"):
        loader._validate_reshard_backend_selection()


def test_load_model_passes_canonical_resource_id_instead_of_local_path(
    monkeypatch,
) -> None:
    model = SimpleNamespace(eval=lambda: model)
    loader = _loader()
    loader.load_config = SimpleNamespace(
        load_format=loader_module.LoadFormat.REMOTE_INSTANCE,
        remote_instance_weight_loader_backend=(
            RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE
        ),
        remote_instance_weight_loader_transfer_engine=object(),
        remote_instance_weight_loader_seed_instance_ip="127.0.0.1",
        remote_instance_weight_loader_seed_instance_service_port=30000,
        remote_instance_weight_loader_transfer_engine_session_id="target-session",
        remote_instance_weight_inventory_builder=object(),
        weight_reshard_resource_id="checkpoint-family-42",
    )
    received = {}

    def load_heterogeneous(*args, **kwargs):
        received.update(kwargs)
        return True

    loader.load_model_from_remote_instance_by_transfer_engine_heterogeneous = (
        load_heterogeneous
    )
    monkeypatch.setattr(loader_module, "_get_quantization_config", lambda *_: None)
    monkeypatch.setattr(loader_module, "_initialize_model", lambda *_: model)
    monkeypatch.setattr(
        loader_module, "set_default_torch_dtype", lambda _: nullcontext()
    )

    result = loader.load_model(
        model_config=SimpleNamespace(
            dtype=None,
            model_path="/node-b/cache/model",
            revision="immutable-revision",
        ),
        device_config=SimpleNamespace(device="cpu"),
    )

    assert result is model
    assert received["target_model_id"] == "checkpoint-family-42"
    assert received["target_revision"] == "immutable-revision"


def _patch_runtime(monkeypatch, backend) -> None:
    _SourceCoordinator.instances.clear()
    _SourceCoordinator.forced_finish_result = None
    monkeypatch.setattr(loader_module, "get_world_group", lambda: _World())
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        _SourceCoordinator,
    )
    monkeypatch.setattr(
        loader_module,
        "create_weight_reshard_backend",
        lambda configured: backend,
    )
    monkeypatch.setattr(loader_module.current_platform, "synchronize", lambda: None)
    monkeypatch.setattr(
        loader_module,
        "_post_load_weights",
        lambda model: backend.events.append("post_load"),
    )
    monkeypatch.setattr(loader_module, "get_server_args", lambda: None)
    monkeypatch.setattr(loader_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        [],
    )


def _load(backend, *, target_revision="immutable-revision") -> bool:
    return _loader().load_model_from_remote_instance_by_transfer_engine_heterogeneous(
        model=SimpleNamespace(),
        transfer_engine=object(),
        seed_url="http://source",
        local_session_id="target-session",
        target_inventory_builder=object(),
        target_model_id="model",
        target_revision=target_revision,
    )


def test_production_loader_calls_replaceable_backend_and_activates_after_terminal(
    monkeypatch,
) -> None:
    activation = WeightSnapshotCoordinator()
    backend = _Backend(activation_coordinator=activation)
    _patch_runtime(monkeypatch, backend)

    assert _load(backend) is True

    assert backend.prepare_kwargs["source_placement_inventories"] == (
        {"weight_generation": 7},
    )
    assert backend.prepare_kwargs["source_binding_inventories"] == ({"generation": 13},)
    assert backend.events == [
        "prepare",
        "execute",
        "post_load",
        "activate",
        "close",
    ]
    assert activation.weight_generation == 7
    assert activation.generation == 2
    assert _SourceCoordinator.instances[0].finish_calls == [(True, True)]


def test_scheduler_inventory_to_backend_activation_production_chain(
    monkeypatch,
) -> None:
    scheduler, inventory_manager, session = _source_scheduler(monkeypatch)
    activation = WeightSnapshotCoordinator()
    backend = _Backend(activation_coordinator=activation)

    class SchedulerBackedCoordinator:
        def __init__(self, seed_url, world_group) -> None:
            self.seed_url = seed_url
            self.world_group = world_group

        def acquire(self):
            return session

        def ready_for_transfer(self, local_ready):
            return bool(local_ready)

        def finish(self, *, local_success, local_release_safe=True):
            if not local_release_safe:
                return False, False
            released = scheduler.release_remote_instance_weight_transfer(
                ReleaseRemoteInstanceWeightTransferReqInput(
                    transfer_id=session.transfer_id
                )
            )
            return bool(local_success), released.success

    monkeypatch.setattr(loader_module, "get_world_group", lambda: _World())
    monkeypatch.setattr(
        loader_module,
        "RemoteInstanceWeightTransferWorldCoordinator",
        SchedulerBackedCoordinator,
    )
    monkeypatch.setattr(
        loader_module,
        "create_weight_reshard_backend",
        lambda configured: backend,
    )
    monkeypatch.setattr(loader_module.current_platform, "synchronize", lambda: None)
    monkeypatch.setattr(
        loader_module,
        "_post_load_weights",
        lambda model: backend.events.append("post_load"),
    )
    monkeypatch.setattr(loader_module, "get_server_args", lambda: None)
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE",
        [],
    )

    assert _load(backend, target_revision="checkpoint-sha") is True

    assert backend.prepare_kwargs["source_placement_inventories"] == tuple(
        session.placement_inventories
    )
    assert backend.prepare_kwargs["source_binding_inventories"] == tuple(
        session.binding_inventories
    )
    assert backend.events == [
        "prepare",
        "execute",
        "post_load",
        "activate",
        "close",
    ]
    assert activation.weight_generation == 7
    assert activation.generation == 2
    assert inventory_manager.list_leases() == ()


def test_terminal_backend_failure_does_not_activate_content(monkeypatch) -> None:
    activation = WeightSnapshotCoordinator()
    backend = _Backend(
        outcome="terminal-error",
        activation_coordinator=activation,
    )
    _patch_runtime(monkeypatch, backend)

    assert _load(backend) is False

    assert "activate" not in backend.events
    assert backend.prepared.closed is True
    assert activation.weight_generation == 1
    assert _SourceCoordinator.instances[0].finish_calls == [(False, True)]


@pytest.mark.parametrize("outcome", ["missing-receipt", "short-receipt"])
def test_invalid_execution_receipt_never_reaches_post_load_or_activation(
    monkeypatch,
    outcome,
) -> None:
    activation = WeightSnapshotCoordinator()
    backend = _Backend(outcome=outcome, activation_coordinator=activation)
    _patch_runtime(monkeypatch, backend)

    assert _load(backend) is False

    assert backend.events == ["prepare", "execute", "close"]
    assert activation.weight_generation == 1
    assert _SourceCoordinator.instances[0].finish_calls == [(False, True)]


def test_world_or_source_release_failure_prevents_content_activation(
    monkeypatch,
) -> None:
    activation = WeightSnapshotCoordinator()
    backend = _Backend(activation_coordinator=activation)
    _patch_runtime(monkeypatch, backend)
    _SourceCoordinator.forced_finish_result = (False, False)

    assert _load(backend) is False

    assert "activate" not in backend.events
    assert activation.weight_generation == 1
    assert activation.generation == 1
    assert _SourceCoordinator.instances[0].finish_calls == [(True, True)]


def test_peer_activation_failure_fails_the_entire_target_world(monkeypatch) -> None:
    activation = WeightSnapshotCoordinator()
    backend = _Backend(activation_coordinator=activation)
    _patch_runtime(monkeypatch, backend)
    monkeypatch.setattr(
        loader_module,
        "get_world_group",
        lambda: _ActivationPeerFailureWorld(),
    )

    assert _load(backend) is False

    assert "activate" in backend.events
    assert activation.weight_generation == 7
    assert _SourceCoordinator.instances[0].finish_calls == [(True, True)]


def test_local_activation_failure_is_published_to_the_target_world(monkeypatch) -> None:
    activation = WeightSnapshotCoordinator()
    backend = _Backend(
        outcome="activation-error",
        activation_coordinator=activation,
    )
    _patch_runtime(monkeypatch, backend)

    assert _load(backend) is False

    assert "activate" in backend.events
    assert activation.weight_generation == 1
    assert _SourceCoordinator.instances[0].finish_calls == [(True, True)]


def test_completion_unknown_retains_context_and_blocks_source_release(
    monkeypatch,
) -> None:
    activation = WeightSnapshotCoordinator()
    backend = _Backend(outcome="unknown", activation_coordinator=activation)
    _patch_runtime(monkeypatch, backend)
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_DRAIN_MAX_ATTEMPTS",
        1,
    )

    assert _load(backend) is False

    assert backend.events == ["prepare", "execute", "retain", "drain"]
    assert backend.prepared.closed is False
    assert backend.prepared.completion_unknown_retained is True
    assert activation.weight_generation == 1
    assert _SourceCoordinator.instances[0].finish_calls == [(False, False)]
    assert len(loader_module._HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE) == 1


def test_completion_unknown_releases_resources_after_known_terminal_drain(
    monkeypatch,
) -> None:
    activation = WeightSnapshotCoordinator()
    backend = _Backend(
        outcome="unknown-terminal",
        activation_coordinator=activation,
    )
    _patch_runtime(monkeypatch, backend)
    monkeypatch.setattr(
        loader_module,
        "_HETEROGENEOUS_UNKNOWN_DRAIN_MAX_ATTEMPTS",
        1,
    )

    assert _load(backend) is False

    assert backend.events == ["prepare", "execute", "retain", "drain", "close"]
    assert backend.prepared.closed is True
    assert backend.prepared.completion_unknown_retained is False
    assert activation.weight_generation == 1
    assert _SourceCoordinator.instances[0].finish_calls == [(False, True)]
    assert loader_module._HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE == []


def test_untyped_execute_failure_is_quarantined_and_blocks_source_release(
    monkeypatch,
) -> None:
    backend = _Backend(outcome="unexpected-error")
    _patch_runtime(monkeypatch, backend)

    assert _load(backend) is False

    assert backend.events == ["prepare", "execute", "retain"]
    assert backend.prepared.closed is False
    assert backend.prepared.completion_unknown_retained is True
    assert _SourceCoordinator.instances[0].finish_calls == [(False, False)]
    assert len(loader_module._HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE) == 1


def test_execute_interruption_is_quarantined_before_it_propagates(monkeypatch) -> None:
    backend = _Backend(outcome="interrupted")
    _patch_runtime(monkeypatch, backend)

    with pytest.raises(KeyboardInterrupt):
        _load(backend)

    assert backend.events == ["prepare", "execute", "retain"]
    assert backend.prepared.closed is False
    assert backend.prepared.completion_unknown_retained is True
    assert _SourceCoordinator.instances[0].finish_calls == [(False, False)]
    assert len(loader_module._HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE) == 1


def test_backend_unavailable_fails_before_source_session_acquisition(
    monkeypatch,
) -> None:
    _SourceCoordinator.instances.clear()
    monkeypatch.setattr(loader_module, "get_world_group", lambda: _World())
    monkeypatch.setattr(loader_module, "get_server_args", lambda: None)
    monkeypatch.setattr(
        loader_module,
        "create_weight_reshard_backend",
        lambda configured: (_ for _ in ()).throw(
            WeightReshardBackendUnavailableError("Mooncake unavailable")
        ),
    )

    assert _load(None) is False
    assert _SourceCoordinator.instances == []


@pytest.mark.parametrize(
    "world_group",
    [
        object(),
        SimpleNamespace(world_size=True, rank_in_group=0),
        SimpleNamespace(world_size=1, rank_in_group=0),
        SimpleNamespace(
            world_size=1,
            rank_in_group=1,
            all_gather_object=lambda value: [value],
            broadcast_object=lambda value, src: value,
        ),
    ],
)
def test_invalid_target_world_group_fails_before_backend_or_source_acquisition(
    monkeypatch,
    world_group,
) -> None:
    _SourceCoordinator.instances.clear()
    backend_calls = []
    monkeypatch.setattr(loader_module, "get_world_group", lambda: world_group)
    monkeypatch.setattr(loader_module, "get_server_args", lambda: None)
    monkeypatch.setattr(
        loader_module,
        "create_weight_reshard_backend",
        lambda configured: backend_calls.append(configured),
    )

    assert _load(None) is False
    assert backend_calls == []
    assert _SourceCoordinator.instances == []


def test_placeholder_target_revision_fails_before_backend_or_source_acquisition(
    monkeypatch,
) -> None:
    _SourceCoordinator.instances.clear()
    backend_calls = []
    monkeypatch.setattr(loader_module, "get_world_group", lambda: _World())
    monkeypatch.setattr(loader_module, "get_server_args", lambda: None)
    monkeypatch.setattr(
        loader_module,
        "create_weight_reshard_backend",
        lambda configured: backend_calls.append(configured),
    )

    assert _load(None, target_revision="default") is False
    assert backend_calls == []
    assert _SourceCoordinator.instances == []


def test_post_load_refreshes_runtime_state_only_on_terminal_path() -> None:
    calls = []
    model = SimpleNamespace(
        modules=lambda: (
            SimpleNamespace(refresh_runtime_weight_state=lambda: calls.append("a")),
            SimpleNamespace(refresh_runtime_weight_state=lambda: calls.append("b")),
        ),
        post_load_weights=lambda: calls.append("model"),
    )

    loader_module._post_load_weights(model)

    assert calls == ["a", "b", "model"]


def test_target_snapshot_coordinator_exists_before_target_inventory_binding() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner.server_args = SimpleNamespace(enable_weight_reshard=True)
    runner.init_weight_snapshot_coordinator()

    assert isinstance(runner.weight_snapshot_coordinator, WeightSnapshotCoordinator)
    assert runner.weight_inventory_manager is None


def test_target_builder_allows_idempotent_runtime_state_refresh_modules() -> None:
    runner = ModelRunner.__new__(ModelRunner)
    runner.model_config = SimpleNamespace(quantization=None)
    runner.weight_snapshot_coordinator = WeightSnapshotCoordinator()
    runner.remote_instance_weight_transporter = SimpleNamespace(
        worker_id="target-worker"
    )
    placement = SimpleNamespace(weight_generation=7)
    manager = SimpleNamespace(
        target_layout_inventory=lambda **_kwargs: placement,
    )
    observed = {}

    def create_manager(*, model, coordinator):
        observed["model"] = model
        observed["coordinator"] = coordinator
        return manager

    runner._create_weight_inventory_manager = create_manager
    model = SimpleNamespace(
        modules=lambda: (SimpleNamespace(refresh_runtime_weight_state=lambda: None),)
    )

    with runner.build_remote_instance_target_weight_inventory_session(
        model=model,
        model_id="model",
        revision="immutable-revision",
        weight_generation=7,
        instance_id="target-instance",
        endpoint="target:1234",
    ) as session:
        assert session.placement_inventory is placement

    assert observed == {
        "model": model,
        "coordinator": runner.weight_snapshot_coordinator,
    }


def _serialized_fp8_target_runner(monkeypatch) -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.model_config = SimpleNamespace(
        quantization="fp8",
        hf_config=SimpleNamespace(
            model_type="qwen3",
            hidden_size=128,
            intermediate_size=128,
        ),
    )
    runner.server_args = SimpleNamespace(
        enable_lora=False,
        mm_enable_dp_encoder=False,
        enable_eplb=False,
        elastic_ep_backend=None,
        init_expert_location="trivial",
        ep_num_redundant_experts=0,
        moe_runner_backend="triton",
        fp8_gemm_runner_backend="triton",
        enable_dp_attention=False,
    )
    runner.ps = SimpleNamespace(
        dp_rank=None,
        dp_size=1,
        moe_dp_rank=0,
        moe_dp_size=1,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        attn_tp_rank=0,
        attn_tp_size=1,
    )
    runner.device = "cpu"
    runner.is_multimodal = False
    runner.weight_snapshot_coordinator = WeightSnapshotCoordinator()
    runner.remote_instance_weight_transporter = SimpleNamespace(
        worker_id="target-worker"
    )
    monkeypatch.setattr(
        model_runner_module,
        "get_parallel",
        lambda: SimpleNamespace(moe_tp_rank=0, moe_tp_size=1),
    )
    return runner


def test_target_builder_accepts_serialized_block_fp8_runtime_layout(
    monkeypatch,
) -> None:
    runner = _serialized_fp8_target_runner(monkeypatch)

    with runner.build_remote_instance_target_weight_inventory_session(
        model=_SerializedBlockFp8Model(),
        model_id="model",
        revision="immutable-revision",
        weight_generation=7,
        instance_id="target-instance",
        endpoint="target:1234",
    ) as session:
        fragments = session.placement_inventory.fragments
        assert {fragment.tensor_id for fragment in fragments} == {
            "layers.0.mlp.down_proj.weight",
            "layers.0.mlp.down_proj.weight_scale_inv",
        }
        assert all(
            "serialized-block-fp8" in item.layout_fingerprint for item in fragments
        )
        with session.bind() as binding:
            assert len(binding.fragments) == 2


def test_target_builder_rejects_online_fp8_through_inventory_validation(
    monkeypatch,
) -> None:
    runner = _serialized_fp8_target_runner(monkeypatch)

    with pytest.raises(
        WeightInventoryError,
        match="serialized FP8 checkpoint is required",
    ):
        with runner.build_remote_instance_target_weight_inventory_session(
            model=_SerializedBlockFp8Model(serialized=False),
            model_id="model",
            revision="immutable-revision",
            weight_generation=7,
            instance_id="target-instance",
            endpoint="target:1234",
        ):
            pass
