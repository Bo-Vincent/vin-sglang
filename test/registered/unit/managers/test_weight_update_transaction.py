from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.managers.io_struct import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.scheduler_components import (
    weight_updater as scheduler_weight_updater_module,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_executor.weight_inventory_contracts import WeightInventoryError
from sglang.srt.model_executor.weight_snapshot import WeightSnapshotCoordinator


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _Worker:
    def __init__(self, updater: WeightUpdater, coordinator, model=None) -> None:
        self.updater = updater
        self.model_runner = SimpleNamespace(
            weight_snapshot_coordinator=coordinator,
            model=model,
        )

    def update_weights_from_tensor(self, _request):
        return self.updater.update_weights_from_tensor(
            [("weight", torch.ones(1))],
            load_format="direct",
        )

    def update_weights_from_disk(self, request):
        return self.updater.update_weights_from_disk(
            request.model_path,
            request.load_format,
            revision=request.revision,
            recapture_cuda_graph=False,
        )


class _EagleLikeWorker:
    def __init__(self, draft_worker: _Worker, target_worker: _Worker) -> None:
        self._draft = draft_worker
        self._target = target_worker
        self._draft_worker = SimpleNamespace(
            draft_runner=draft_worker.model_runner,
        )
        self._target_worker = SimpleNamespace(
            model_runner=target_worker.model_runner,
        )

    @property
    def target_worker(self):
        return self._target_worker

    def update_weights_from_tensor(self, request):
        success, message = self._draft.update_weights_from_tensor(request)
        if not success:
            return success, message
        return self._target.update_weights_from_tensor(request)


def _manager(worker, *, memory_saver_adapter=None) -> SchedulerWeightUpdaterManager:
    return SchedulerWeightUpdaterManager(
        tp_worker=worker,
        draft_worker=None,
        tp_cpu_group=object(),
        world_cpu_group=object(),
        memory_saver_adapter=memory_saver_adapter or object(),
        flush_cache=lambda **_kwargs: True,
        is_fully_idle=lambda: True,
    )


def _updater(model, coordinator) -> WeightUpdater:
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=SimpleNamespace(
            dtype=torch.float32,
            model_path="model",
            revision="immutable-revision",
        ),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=lambda *_args, **_kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: object(),
        begin_weight_update=coordinator.begin_update,
        finish_weight_update=coordinator.finish_update,
    )


def _tensor_request(*, disable_draft_model: bool = True):
    return SimpleNamespace(
        disable_draft_model=disable_draft_model,
        flush_cache=False,
        torch_empty_cache=False,
    )


def _install_single_rank_gather(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )


def test_real_weight_updater_commits_only_after_scheduler_finalize(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    model = _Model()
    worker = _Worker(_updater(model, coordinator), coordinator)
    manager = _manager(worker)
    _install_single_rank_gather(monkeypatch)

    result = manager.update_weights_from_tensor(_tensor_request())

    assert result.success is True
    assert coordinator.generation == 2
    assert coordinator.weight_generation == 2
    lease_id, _ = coordinator.acquire_snapshot()
    coordinator.release_snapshot(lease_id)


def test_real_weight_updater_cannot_mutate_during_snapshot_lease() -> None:
    coordinator = WeightSnapshotCoordinator()
    model = _Model()
    updater = _updater(model, coordinator)
    lease_id, _ = coordinator.acquire_snapshot()

    result = updater.update_weights_from_tensor(
        [("weight", torch.ones(1))],
        load_format="direct",
    )

    assert result[0] is False
    assert "lease is active" in result[1]
    assert model.weight.item() == 0
    coordinator.release_snapshot(lease_id)


def test_scheduler_pre_mutation_lease_rejection_does_not_poison_content(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    model = _Model()
    worker = _Worker(_updater(model, coordinator), coordinator)
    manager = _manager(worker)
    _install_single_rank_gather(monkeypatch)
    lease_id, original_generation = coordinator.acquire_snapshot()

    result = manager.update_weights_from_tensor(_tensor_request())

    assert result.success is False
    assert "lease is active" in result.message
    assert coordinator.generation == original_generation
    assert coordinator.weight_generation == 1
    coordinator.release_snapshot(lease_id)
    next_lease_id, next_generation = coordinator.acquire_snapshot()
    assert next_generation == original_generation
    coordinator.release_snapshot(next_lease_id)


def test_remote_rank_reservation_failure_cancels_local_before_mutation(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    model = _Model()
    worker = _Worker(_updater(model, coordinator), coordinator)
    manager = _manager(worker)
    calls = 0

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    gather_call = 0

    def gather_remote_lease_failure(outputs, value, group):
        nonlocal gather_call
        del group
        gather_call += 1
        outputs[0] = value
        outputs[1] = (
            {
                "success": False,
                "message": "a weight inventory lease is active",
            }
            if gather_call == 1
            else value
        )

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        gather_remote_lease_failure,
    )
    original = worker.update_weights_from_tensor

    def mutate(request):
        nonlocal calls
        calls += 1
        return original(request)

    monkeypatch.setattr(worker, "update_weights_from_tensor", mutate)
    result = manager.update_weights_from_tensor(_tensor_request())

    assert result.success is False
    assert "rank 1" in result.message
    assert calls == 0
    assert model.weight.item() == 0
    assert coordinator.generation == 1
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 1
    coordinator.release_snapshot(lease_id)


def test_scheduler_pre_capture_failure_does_not_claim_a_mutation(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    model = _Model()
    worker = _Worker(_updater(model, coordinator), coordinator)
    manager = _manager(worker)
    _install_single_rank_gather(monkeypatch)
    capture_calls = 0
    mutate_calls = 0
    original_capture = SchedulerWeightUpdaterManager._capture_weight_update_generations
    original_mutate = worker.update_weights_from_tensor

    def capture(self, workers):
        nonlocal capture_calls
        capture_calls += 1
        if capture_calls == 1:
            raise RuntimeError("generation read failed")
        return original_capture(self, workers)

    def mutate(request):
        nonlocal mutate_calls
        mutate_calls += 1
        return original_mutate(request)

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_capture_weight_update_generations",
        capture,
    )
    monkeypatch.setattr(worker, "update_weights_from_tensor", mutate)

    result = manager.update_weights_from_tensor(_tensor_request())

    assert result.success is False
    assert "pre-mutation weight generation" in result.message
    assert mutate_calls == 0
    assert coordinator.generation == 1
    assert coordinator.weight_generation == 1
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 1
    coordinator.release_snapshot(lease_id)


def test_ordinary_weight_update_preserves_original_barrier_without_collectives(
    monkeypatch,
) -> None:
    calls = 0
    gather_calls = 0
    barrier_calls = 0

    def mutate(_request):
        nonlocal calls
        calls += 1
        return True, "Success."

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(weight_snapshot_coordinator=None),
        update_weights_from_tensor=mutate,
    )
    manager = _manager(worker)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)

    def gather(outputs, value, group):
        nonlocal gather_calls
        del group
        gather_calls += 1
        outputs[0] = value

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    def barrier(*, group):
        nonlocal barrier_calls
        del group
        barrier_calls += 1

    monkeypatch.setattr(torch.distributed, "barrier", barrier)

    result = manager.update_weights_from_tensor(_tensor_request())

    assert result.success is True
    assert calls == 1
    assert gather_calls == 0
    assert barrier_calls == 1


@pytest.mark.parametrize("source", ["disk", "ipc"])
def test_ordinary_target_success_draft_failure_still_flushes_cache(
    monkeypatch,
    source,
) -> None:
    flush_calls = 0
    barrier_calls = 0

    def flush_cache(**_kwargs):
        nonlocal flush_calls
        flush_calls += 1
        return True

    def barrier(*, group):
        nonlocal barrier_calls
        del group
        barrier_calls += 1

    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    target = SimpleNamespace(
        model_runner=SimpleNamespace(weight_snapshot_coordinator=None),
        update_weights_from_disk=lambda _request: (True, "target updated"),
        update_weights_from_ipc=lambda _request: (True, "target updated"),
    )
    draft = SimpleNamespace(
        model_runner=SimpleNamespace(weight_snapshot_coordinator=None),
        update_weights_from_disk=lambda _request: (False, "draft failed"),
        update_weights_from_ipc=lambda _request: (False, "draft failed"),
    )
    manager = _manager(target)
    manager.draft_worker = draft
    manager.flush_cache = flush_cache
    request = SimpleNamespace(
        model_path="new-model",
        revision=None,
        load_format="auto",
        flush_cache=True,
        torch_empty_cache=False,
    )

    if source == "disk":
        result = manager.update_weights_from_disk(request)
    else:
        result = manager.update_weights_from_ipc(request)

    assert result.success is False
    assert result.message == "draft failed"
    assert flush_calls == 1
    assert barrier_calls == (1 if source == "ipc" else 0)


def test_reshard_update_without_snapshot_coordinator_fails_before_mutation(
    monkeypatch,
) -> None:
    calls = 0

    def mutate(_request):
        nonlocal calls
        calls += 1
        return True, "Success."

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(weight_snapshot_coordinator=None),
        update_weights_from_tensor=mutate,
    )
    manager = _manager(worker)
    manager.scheduler = SimpleNamespace(
        server_args=SimpleNamespace(enable_weight_reshard=True)
    )
    _install_single_rank_gather(monkeypatch)

    result = manager.update_weights_from_tensor(_tensor_request())

    assert result.success is False
    assert "no snapshot coordinator" in result.message
    assert calls == 0


def test_cross_rank_failure_poison_rejects_unproven_disk_restore(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    model = _Model()
    updater = _updater(model, coordinator)
    worker = _Worker(updater, coordinator)
    manager = _manager(worker)
    gather_call = 0

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def gather_with_remote_mutation_failure(outputs, value, group):
        nonlocal gather_call
        del group
        gather_call += 1
        outputs[0] = value
        outputs[1] = (
            {"success": False, "message": "remote mutation failed"}
            if gather_call == 2
            else value
        )

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        gather_with_remote_mutation_failure,
    )

    with pytest.raises(WeightInventoryError, match="unverified.*must restart"):
        manager.update_weights_from_tensor(_tensor_request())

    with pytest.raises(WeightInventoryError, match="full successful weight restore"):
        coordinator.acquire_snapshot()

    _install_single_rank_gather(monkeypatch)
    incremental = manager.update_weights_from_tensor(_tensor_request())
    assert incremental.success is False
    with pytest.raises(WeightInventoryError, match="full successful weight restore"):
        coordinator.acquire_snapshot()

    unproven_restore = manager.update_weights_from_disk(
        SimpleNamespace(
            model_path="restored-model",
            revision="restored-checkpoint-sha",
            load_format="auto",
            recapture_cuda_graph=False,
            flush_cache=False,
            torch_empty_cache=False,
        )
    )

    assert unproven_restore.success is False
    assert "full successful weight restore" in unproven_restore.message
    with pytest.raises(WeightInventoryError, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_single_rank_partial_mutation_fails_stop_before_serving(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    model = _Model()
    worker = _Worker(_updater(model, coordinator), coordinator)
    manager = _manager(worker)
    _install_single_rank_gather(monkeypatch)

    def partially_mutate(_request):
        model.weight.data.fill_(1)
        return False, "weights were partially updated"

    monkeypatch.setattr(worker, "update_weights_from_tensor", partially_mutate)

    with pytest.raises(
        WeightInventoryError,
        match="unverified.*must restart before serving",
    ):
        manager.update_weights_from_tensor(_tensor_request())

    assert model.weight.item() == 1
    assert coordinator.generation == 2
    assert coordinator.weight_generation == 1
    with pytest.raises(WeightInventoryError, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_disk_update_requires_explicit_lineage_before_any_reshard_mutation(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    model = _Model()
    worker = _Worker(_updater(model, coordinator), coordinator)
    manager = _manager(worker)
    _install_single_rank_gather(monkeypatch)
    calls = 0
    original = worker.update_weights_from_disk

    def mutate(request):
        nonlocal calls
        calls += 1
        return original(request)

    monkeypatch.setattr(worker, "update_weights_from_disk", mutate)
    result = manager.update_weights_from_disk(
        SimpleNamespace(
            model_path="new-model",
            revision=None,
            load_format="auto",
            recapture_cuda_graph=False,
            flush_cache=False,
            torch_empty_cache=False,
        )
    )

    assert result.success is False
    assert "content-lineage revision" in result.message
    assert calls == 0
    assert coordinator.generation == 1
    assert coordinator.weight_generation == 1


def test_eagle_tensor_update_commits_target_and_draft_generations(
    monkeypatch,
) -> None:
    draft_coordinator = WeightSnapshotCoordinator()
    target_coordinator = WeightSnapshotCoordinator()
    draft_model = _Model()
    target_model = _Model()
    draft_worker = _Worker(
        _updater(draft_model, draft_coordinator),
        draft_coordinator,
        model=draft_model,
    )
    target_worker = _Worker(
        _updater(target_model, target_coordinator),
        target_coordinator,
        model=target_model,
    )
    eagle_worker = _EagleLikeWorker(draft_worker, target_worker)
    manager = _manager(target_worker)
    manager.draft_worker = eagle_worker
    _install_single_rank_gather(monkeypatch)

    result = manager.update_weights_from_tensor(
        _tensor_request(disable_draft_model=False)
    )

    assert result.success is True
    assert draft_coordinator.generation == 2
    assert draft_coordinator.weight_generation == 2
    assert target_coordinator.generation == 2
    assert target_coordinator.weight_generation == 2


def test_weight_offload_resume_preserves_logical_content_generation(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    model = _Model()
    worker = _Worker(_updater(model, coordinator), coordinator, model=model)
    events = []
    memory_saver = SimpleNamespace(
        pause=lambda tag: events.append(("pause", tag)),
        resume=lambda tag: events.append(("resume", tag)),
    )
    manager = _manager(worker, memory_saver_adapter=memory_saver)
    _install_single_rank_gather(monkeypatch)
    monkeypatch.setattr(torch.distributed, "barrier", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda: SimpleNamespace(synchronize=lambda: events.append(("sync", None))),
    )
    monkeypatch.setattr(
        scheduler_weight_updater_module,
        "_export_static_state",
        lambda loaded_model: (events.append(("export", loaded_model)), "state")[1],
    )
    monkeypatch.setattr(
        scheduler_weight_updater_module,
        "_import_static_state",
        lambda loaded_model, state: events.append(("import", loaded_model, state)),
    )

    manager.release_memory_occupation(
        ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
    )

    assert coordinator.generation == 2
    assert coordinator.weight_generation == 1
    with pytest.raises(WeightInventoryError, match="storage is unavailable"):
        coordinator.acquire_snapshot()

    manager.resume_memory_occupation(
        ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
    )

    assert coordinator.generation == 3
    assert coordinator.weight_generation == 1
    lease_id, binding_generation = coordinator.acquire_snapshot()
    assert binding_generation == 3
    coordinator.release_snapshot(lease_id)
    assert events == [
        ("export", model),
        ("pause", GPU_MEMORY_TYPE_WEIGHTS),
        ("sync", None),
        ("resume", GPU_MEMORY_TYPE_WEIGHTS),
        ("import", model, "state"),
    ]


def test_cross_rank_storage_transition_publishes_one_global_outcome(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    model = _Model()
    worker = _Worker(_updater(model, coordinator), coordinator, model=model)
    memory_saver = SimpleNamespace(
        pause=lambda _tag: None,
        resume=lambda _tag: None,
    )
    manager = _manager(worker, memory_saver_adapter=memory_saver)
    gather_call = 0

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def gather_with_remote_transition_failure(outputs, value, group):
        nonlocal gather_call
        del group
        gather_call += 1
        outputs[0] = value
        outputs[1] = None if gather_call == 1 else "remote offload failed"

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        gather_with_remote_transition_failure,
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(
        scheduler_weight_updater_module,
        "_export_static_state",
        lambda _loaded_model: "state",
    )

    with pytest.raises(
        WeightInventoryError,
        match="weight memory transition failed.*rank 1: remote offload failed",
    ):
        manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        )

    assert gather_call == 2
    assert coordinator.generation == 2
    assert coordinator.weight_generation == 1
    with pytest.raises(WeightInventoryError, match="full successful weight restore"):
        coordinator.acquire_snapshot()
