import asyncio
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.managers import (
    tokenizer_control_mixin as tokenizer_control_mixin_module,
)
from sglang.srt.managers.io_struct import (
    BeginRemoteInstanceWeightTransferReqInput,
    BeginRemoteInstanceWeightTransferReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseRemoteInstanceWeightTransferReqInput,
    RenewRemoteInstanceWeightTransferReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
)
from sglang.srt.managers.scheduler_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import (
    RemoteInstanceWeightTransferBeginError,
    TokenizerControlMixin,
)
from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightManifestError,
    WeightSnapshotCoordinator,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _manager(runner, *, remote_weight_transfer_cpu_group=None):
    if not hasattr(runner, "server_args"):
        runner.server_args = SimpleNamespace(weight_cache_mode="off")
    kwargs = {}
    if remote_weight_transfer_cpu_group is not None:
        kwargs["remote_weight_transfer_cpu_group"] = remote_weight_transfer_cpu_group
    return SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(model_runner=runner),
        draft_worker=None,
        tp_cpu_group=object(),
        world_cpu_group=object(),
        memory_saver_adapter=object(),
        flush_cache=lambda **kwargs: True,
        is_fully_idle=lambda: True,
        **kwargs,
    )


def _manifest(worker_id="source/dp0-pp0-ep0-tp0", lease_id="lease-0"):
    return {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "generation": 1,
        "lease_id": lease_id,
        "tensors": [{"worker_id": worker_id}],
    }


def _placement(dp_rank=0):
    return {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "placement_id": f"source-placement-dp{dp_rank}",
        "tensors": [
            {
                "placement_fragment_id": f"source-fragment-dp{dp_rank}",
                "tensor_id": "model.layers.0.weight",
                "aliases": ["model.layers.0.weight"],
                "global_shape": [8, 8],
                "global_offset": [0, 0],
                "local_shape": [8, 8],
                "dtype": "float16",
                "itemsize": 2,
                "partition_dim": None,
                "shard_dims": [],
                "layer_id": 0,
                "expert_id": None,
                "layout_fingerprint": "dense-row-major",
                "nbytes": 128,
                "byte_offset": 0,
                "rank": {"dp": dp_rank, "tp": 0, "pp": 0, "ep": 0},
            }
        ],
    }


def _binding(dp_rank=0, lease_id="lease-0"):
    return {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main@generation-1",
        "placement_id": f"source-placement-dp{dp_rank}",
        "instance_id": f"source-instance-dp{dp_rank}",
        "generation": 1,
        "lease_id": lease_id,
        "fragments": [
            {
                "placement_fragment_id": f"source-fragment-dp{dp_rank}",
                "fragment_id": f"runtime-fragment-dp{dp_rank}",
                "address": 0x1000 + dp_rank * 0x1000,
                "nbytes": 128,
                "storage_offset": 0,
                "device": "cuda",
                "is_contiguous": True,
                "worker_id": f"source/dp{dp_rank}-pp0-ep0-tp0",
                "endpoint": f"source-session-dp{dp_rank}",
            }
        ],
    }


def test_remote_transfer_collective_does_not_block_scheduler_thread(
    monkeypatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_begin(self, request):
        entered.set()
        assert release.wait(timeout=5)
        return BeginRemoteInstanceWeightTransferReqOutput(
            transfer_id=request.transfer_id,
            success=True,
            message="Success.",
            manifests=[_manifest()],
        )

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "begin_remote_instance_weight_transfer",
        blocking_begin,
    )
    manager = _manager(SimpleNamespace())
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
        lease_timeout_sec=60,
    )

    assert manager.defer_begin_remote_instance_weight_transfer(request) is None
    assert entered.wait(timeout=1)
    assert manager.check_pending_remote_instance_weight_transfers() == []

    release.set()
    deadline = time.monotonic() + 1
    completed = []
    while not completed and time.monotonic() < deadline:
        completed = manager.check_pending_remote_instance_weight_transfers()
        time.sleep(0.01)

    assert len(completed) == 1
    output, completed_request = completed[0]
    assert output.success is True
    assert completed_request is request
    manager.close_remote_instance_weight_transfer_executor()


def test_begin_and_release_remote_transfer_snapshot(monkeypatch) -> None:
    released = []
    manifest = _manifest()
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: manifest,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)

    def all_gather_object(outputs, value, group):
        outputs[0] = value

    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)
    result = manager.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is True
    assert result.manifests == [manifest]
    assert released == []

    release = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )
    assert release.success is True
    assert released == ["lease-0"]


def test_duplicate_begin_returns_the_same_snapshot_without_a_second_lease(
    monkeypatch,
) -> None:
    snapshots = []
    manifest = _manifest()

    def snapshot(**kwargs):
        snapshots.append(kwargs)
        return manifest

    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=snapshot,
        release_weight_runtime_manifest=lambda lease_id: None,
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
    )

    first = manager.begin_remote_instance_weight_transfer(request)
    second = manager.begin_remote_instance_weight_transfer(request)

    assert first.session_state == "created"
    assert second.session_state == "reused"
    assert second.manifests == first.manifests
    assert len(snapshots) == 1
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}


def test_late_begin_after_release_does_not_reacquire_snapshot(monkeypatch) -> None:
    snapshots = []
    released = []
    manifest = _manifest()

    def snapshot(**kwargs):
        snapshots.append(kwargs)
        return manifest

    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=snapshot,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
    )

    assert manager.begin_remote_instance_weight_transfer(request).success is True
    assert (
        manager.release_remote_instance_weight_transfer(
            ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
        ).success
        is True
    )

    replay = manager.begin_remote_instance_weight_transfer(request)

    assert replay.success is False
    assert "already released" in replay.message.lower()
    assert len(snapshots) == 1
    assert released == ["lease-0"]
    assert manager.remote_weight_transfer_leases == {}


def test_late_begin_after_expired_release_does_not_reacquire_snapshot(
    monkeypatch,
) -> None:
    now = [100.0]
    snapshots = []
    released = []
    manifest = _manifest()

    def snapshot(**kwargs):
        snapshots.append(kwargs)
        return manifest

    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=snapshot,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr(weight_updater_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
        lease_timeout_sec=30,
    )

    assert manager.begin_remote_instance_weight_transfer(request).success is True
    now[0] = 131.0
    manager._prune_remote_weight_transfer_bookkeeping()
    assert manager.remote_weight_transfer_expired == {"transfer-1"}
    assert (
        manager.release_remote_instance_weight_transfer(
            ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
        ).success
        is True
    )

    replay = manager.begin_remote_instance_weight_transfer(request)

    assert replay.success is False
    assert "already released" in replay.message.lower()
    assert len(snapshots) == 1
    assert released == ["lease-0"]
    assert manager.remote_weight_transfer_leases == {}


def test_released_transfer_tombstones_are_time_and_count_bounded(
    monkeypatch,
) -> None:
    now = [100.0]
    manager = _manager(SimpleNamespace())
    monkeypatch.setattr(weight_updater_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        weight_updater_module,
        "_REMOTE_WEIGHT_TRANSFER_TOMBSTONE_TTL_SEC",
        10.0,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "_REMOTE_WEIGHT_TRANSFER_TOMBSTONE_LIMIT",
        2,
    )

    manager._complete_remote_weight_transfer_session("transfer-1")
    manager._complete_remote_weight_transfer_session("transfer-2")
    manager._complete_remote_weight_transfer_session("transfer-3")

    assert list(manager.remote_weight_transfer_tombstones) == [
        "transfer-2",
        "transfer-3",
    ]

    now[0] = 111.0
    manager._prune_remote_weight_transfer_bookkeeping()

    assert manager.remote_weight_transfer_tombstones == {}


def test_duplicate_begin_rejects_transfer_id_parameter_mismatch(monkeypatch) -> None:
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: _manifest(),
        release_weight_runtime_manifest=lambda lease_id: None,
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    first = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
    )
    mismatched = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="different",
    )

    assert manager.begin_remote_instance_weight_transfer(first).success is True
    result = manager.begin_remote_instance_weight_transfer(mismatched)

    assert result.success is False
    assert "different parameters" in result.message


def test_begin_rejects_divergent_cached_state_without_acquiring_snapshot(
    monkeypatch,
) -> None:
    snapshots = []
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: snapshots.append(
            kwargs
        ),
        release_weight_runtime_manifest=lambda lease_id: None,
    )
    manager = _manager(runner)
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
    )
    cached = BeginRemoteInstanceWeightTransferReqOutput(
        transfer_id="transfer-1",
        success=True,
        message="Success.",
        manifests=[_manifest()],
    )
    manager._record_remote_weight_transfer_session(request, "lease-0", cached)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 2)

    def all_gather_object(outputs, value, group):
        outputs[:] = [
            value,
            {
                "success": True,
                "message": "Success.",
                "session_state": "created",
            },
        ]

    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)

    result = manager.begin_remote_instance_weight_transfer(request)

    assert result.success is False
    assert result.session_state == "cleanup_pending"
    assert "inconsistent session state" in result.message.lower()
    assert snapshots == []
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}


def test_begin_and_release_split_remote_transfer_snapshot(monkeypatch) -> None:
    released = []
    placement = _placement()
    binding = _binding()
    parts = SimpleNamespace(placement=placement, binding=binding)
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest_parts=lambda **kwargs: parts,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )

    result = manager.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            manifest_format="placement_binding_v1",
        )
    )

    assert result.success is True
    assert result.manifests is None
    assert result.placements == [placement]
    assert result.bindings == [binding]
    assert released == []

    release = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )
    assert release.success is True
    assert released == ["lease-0"]


def test_begin_uses_dedicated_remote_transfer_cpu_group(monkeypatch) -> None:
    manifest = _manifest()
    remote_group = object()
    observed_groups = []
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: manifest,
        release_weight_runtime_manifest=lambda lease_id: None,
    )
    manager = _manager(runner, remote_weight_transfer_cpu_group=remote_group)

    def get_world_size(group):
        observed_groups.append(("world_size", group))
        return 1

    def all_gather_object(outputs, value, group):
        observed_groups.append(("all_gather", group))
        outputs[0] = value

    monkeypatch.setattr("torch.distributed.get_world_size", get_world_size)
    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)

    result = manager.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is True
    assert observed_groups == [
        ("world_size", remote_group),
        ("all_gather", remote_group),
    ]


def test_begin_rolls_back_local_snapshot_when_another_rank_fails(monkeypatch) -> None:
    released = []
    manifest = _manifest()
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: manifest,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 2)

    def all_gather_object(outputs, value, group):
        outputs[:] = [value, {"success": False, "message": "rank 1 failed"}]

    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)
    result = manager.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is False
    assert "rank 1 failed" in result.message
    assert released == ["lease-0"]


def test_begin_rolls_back_local_snapshot_when_collective_fails(monkeypatch) -> None:
    released = []
    manifest = _manifest()
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: manifest,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: (_ for _ in ()).throw(
            RuntimeError("collective failed")
        ),
    )

    result = manager.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is False
    assert "collective failed" in result.message
    assert released == ["lease-0"]


def test_begin_keeps_cleanup_pending_lease_when_rollback_release_fails(
    monkeypatch,
) -> None:
    release_attempts = []
    manifest = _manifest()

    def release(lease_id):
        release_attempts.append(lease_id)
        if len(release_attempts) == 1:
            raise RuntimeError("temporary rollback release failure")

    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: manifest,
        release_weight_runtime_manifest=release,
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 2)

    def all_gather_object(outputs, value, group):
        outputs[:] = [
            value,
            {
                "success": False,
                "message": "rank 1 failed",
                "session_state": "failed",
            },
        ]

    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
    )

    result = manager.begin_remote_instance_weight_transfer(request)

    assert result.success is False
    assert result.session_state == "cleanup_pending"
    assert "temporary rollback release failure" in result.message
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    released = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )

    assert released.success is True
    assert release_attempts == ["lease-0", "lease-0"]
    assert manager.remote_weight_transfer_leases == {}


def test_begin_tracks_snapshot_when_serialization_and_rollback_release_fail(
    monkeypatch,
) -> None:
    release_attempts = []
    snapshot = SimpleNamespace(lease_id="lease-0", generation=7)

    def release(lease_id):
        release_attempts.append(lease_id)
        if len(release_attempts) == 1:
            raise RuntimeError("temporary rollback release failure")

    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: snapshot,
        release_weight_runtime_manifest=release,
    )
    manager = _manager(runner)
    monkeypatch.setattr(
        weight_updater_module.msgspec,
        "to_builtins",
        lambda value: (_ for _ in ()).throw(RuntimeError("serialization failed")),
    )
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
    )

    result = manager.begin_remote_instance_weight_transfer(request)

    assert result.success is False
    assert result.session_state == "cleanup_pending"
    assert "serialization failed" in result.message
    assert "temporary rollback release failure" in result.message
    assert manager.list_remote_instance_weight_transfer_sessions() == [
        {
            "transfer_id": "transfer-1",
            "lease_id": "lease-0",
            "generation": 7,
            "deadline_monotonic_sec": manager.remote_weight_transfer_deadlines[
                "transfer-1"
            ],
            "expired": False,
            "session_state": "cleanup_pending",
        }
    ]

    released = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )

    assert released.success is True
    assert release_attempts == ["lease-0", "lease-0"]
    assert manager.list_remote_instance_weight_transfer_sessions() == []


def test_begin_tracks_snapshot_when_generation_and_rollback_release_fail(
    monkeypatch,
) -> None:
    release_attempts = []
    snapshot = SimpleNamespace(lease_id="lease-0")

    def release(lease_id):
        release_attempts.append(lease_id)
        if len(release_attempts) == 1:
            raise RuntimeError("temporary rollback release failure")

    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: snapshot,
        release_weight_runtime_manifest=release,
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
    )

    result = manager.begin_remote_instance_weight_transfer(request)

    assert result.success is False
    assert result.session_state == "cleanup_pending"
    assert "generation" in result.message
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}

    released = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )

    assert released.success is True
    assert release_attempts == ["lease-0", "lease-0"]
    assert manager.remote_weight_transfer_leases == {}


def test_runtime_revision_commit_ignores_workers_without_manifest_support() -> None:
    SchedulerWeightUpdaterManager._commit_weight_runtime_revision(
        SimpleNamespace(model_runner=SimpleNamespace())
    )


def test_weight_update_does_not_commit_before_all_ranks_succeed(
    monkeypatch,
) -> None:
    events = []

    class RecordingCoordinator(WeightSnapshotCoordinator):
        def commit_revision(self, *, expected_generation=None):
            events.append("commit")
            return super().commit_revision(expected_generation=expected_generation)

    coordinator = RecordingCoordinator()
    runner = SimpleNamespace(
        weight_snapshot_coordinator=coordinator,
        commit_weight_runtime_revision=coordinator.commit_revision,
    )
    manager = _manager(runner)

    def update_weights_from_distributed(request):
        del request
        token = coordinator.begin_update()
        events.append("mutate")
        coordinator.finish_update(token, success=True)
        return True, "local update succeeded"

    manager.tp_worker.update_weights_from_distributed = update_weights_from_distributed
    collective_calls = []

    def all_gather_object(outputs, value, group):
        del group
        collective_calls.append(value)
        events.append("collective")
        if len(collective_calls) == 1:
            outputs[:] = [
                value,
                {"success": False, "message": "remote rank mutation failed"},
            ]
        else:
            outputs[:] = [value, {"success": True, "message": "Success."}]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    result = manager.update_weights_from_distributed(
        UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            flush_cache=False,
        )
    )

    assert result.success is False
    assert "remote rank mutation failed" in result.message
    assert "commit" not in events
    assert len(collective_calls) == 3
    with pytest.raises(WeightManifestError, match="last weight update failed"):
        coordinator.acquire_snapshot()


def test_weight_update_commits_only_after_global_outcome(monkeypatch) -> None:
    events = []

    class RecordingCoordinator(WeightSnapshotCoordinator):
        def commit_revision(self, *, expected_generation=None):
            events.append("commit")
            return super().commit_revision(expected_generation=expected_generation)

    coordinator = RecordingCoordinator()
    runner = SimpleNamespace(
        weight_snapshot_coordinator=coordinator,
        commit_weight_runtime_revision=coordinator.commit_revision,
    )
    manager = _manager(runner)

    def update_weights_from_distributed(request):
        del request
        token = coordinator.begin_update()
        events.append("mutate")
        coordinator.finish_update(token, success=True)
        return True, "local update succeeded"

    manager.tp_worker.update_weights_from_distributed = update_weights_from_distributed

    def all_gather_object(outputs, value, group):
        del group
        events.append("collective")
        outputs[:] = [value, dict(value)]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    result = manager.update_weights_from_distributed(
        UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            flush_cache=False,
        )
    )

    assert result.success is True
    assert events == [
        "mutate",
        "collective",
        "collective",
        "commit",
        "collective",
    ]
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 2
    coordinator.release_snapshot(lease_id)


def test_stale_weight_update_cannot_finalize_a_newer_transaction(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    runner = SimpleNamespace(weight_snapshot_coordinator=coordinator)
    manager = _manager(runner)

    def update_weights_from_distributed(request):
        del request
        token = coordinator.begin_update()
        coordinator.finish_update(token, success=True)
        return True, "transaction A mutated"

    manager.tp_worker.update_weights_from_distributed = update_weights_from_distributed
    collective_calls = []

    def all_gather_object(outputs, value, group):
        del group
        collective_calls.append(value)
        if len(collective_calls) == 1:
            token = coordinator.begin_update()
            coordinator.finish_update(token, success=True)
        outputs[:] = [value, dict(value)]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    result = manager.update_weights_from_distributed(
        UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
            flush_cache=False,
        )
    )

    assert result.success is False
    assert "generation" in result.message
    assert len(collective_calls) == 3
    assert coordinator.pending_revision_generation() == 3

    coordinator.commit_revision(expected_generation=3)
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 3
    coordinator.release_snapshot(lease_id)


def test_weight_update_exception_becomes_collective_outcome_without_barrier(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    runner = SimpleNamespace(
        weight_snapshot_coordinator=coordinator,
        commit_weight_runtime_revision=coordinator.commit_revision,
    )
    manager = _manager(runner)

    def update_weights_from_ipc(request):
        del request
        token = coordinator.begin_update()
        try:
            raise RuntimeError("local mutation raised")
        finally:
            coordinator.finish_update(token, success=False)

    manager.tp_worker.update_weights_from_ipc = update_weights_from_ipc
    collective_calls = []

    def all_gather_object(outputs, value, group):
        del group
        collective_calls.append(value)
        outputs[:] = [value, dict(value)]

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("weight update transaction must not use a bare barrier")
        ),
    )

    result = manager.update_weights_from_ipc(
        UpdateWeightsFromIPCReqInput(zmq_handles={}, flush_cache=False)
    )

    assert result.success is False
    assert "local mutation raised" in result.message
    assert len(collective_calls) == 3
    with pytest.raises(WeightManifestError, match="last weight update failed"):
        coordinator.acquire_snapshot()


def test_weight_memory_release_is_rejected_while_snapshot_lease_is_active(
    monkeypatch,
) -> None:
    events = []
    coordinator = WeightSnapshotCoordinator()
    lease_id, _ = coordinator.acquire_snapshot()
    runner = SimpleNamespace(
        model=object(),
        weight_snapshot_coordinator=coordinator,
    )
    manager = _manager(runner)
    manager.memory_saver_adapter = SimpleNamespace(
        pause=lambda tag: events.append(("pause", tag))
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )

    with pytest.raises(WeightManifestError, match="snapshot lease is active"):
        manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        )

    assert events == []
    assert manager.offload_tags == set()
    coordinator.release_snapshot(lease_id)


def test_weight_memory_release_and_resume_advance_snapshot_generation(
    monkeypatch,
) -> None:
    events = []
    coordinator = WeightSnapshotCoordinator()
    runner = SimpleNamespace(
        model=object(),
        weight_snapshot_coordinator=coordinator,
    )
    manager = _manager(runner)
    manager.memory_saver_adapter = SimpleNamespace(
        pause=lambda tag: events.append(("pause", tag)),
        resume=lambda tag: events.append(("resume", tag)),
    )
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.weight_updater._export_static_state",
        lambda model: {"buffers": []},
    )
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.weight_updater._import_static_state",
        lambda model, state: events.append(("restore", state)),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda group: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda: SimpleNamespace(synchronize=lambda: None),
    )

    manager.release_memory_occupation(
        ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
    )
    with pytest.raises(WeightManifestError, match="revision commit"):
        coordinator.acquire_snapshot()

    manager.resume_memory_occupation(
        ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
    )
    lease_id, generation = coordinator.acquire_snapshot()

    assert generation == 3
    assert events == [
        ("pause", GPU_MEMORY_TYPE_WEIGHTS),
        ("resume", GPU_MEMORY_TYPE_WEIGHTS),
        ("restore", {"buffers": []}),
    ]
    coordinator.release_snapshot(lease_id)


def test_release_keeps_snapshot_lease_available_for_retry(monkeypatch) -> None:
    attempts = []

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)

    def all_gather_object(outputs, value, group):
        outputs[0] = value

    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)

    def release(lease_id):
        attempts.append(lease_id)
        if len(attempts) == 1:
            raise RuntimeError("temporary release failure")

    manager = _manager(SimpleNamespace(release_weight_runtime_manifest=release))
    manager.remote_weight_transfer_leases["transfer-1"] = "lease-0"
    request = ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")

    first = manager.release_remote_instance_weight_transfer(request)
    assert first.success is False
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}

    second = manager.release_remote_instance_weight_transfer(request)
    assert second.success is True
    assert attempts == ["lease-0", "lease-0"]
    assert manager.remote_weight_transfer_leases == {}


def test_expired_remote_transfer_bookkeeping_rejects_silent_renewal(
    monkeypatch,
) -> None:
    now = [100.0]
    renewed = []
    runner = SimpleNamespace(
        renew_weight_runtime_manifest=lambda lease_id, lease_timeout_sec: (
            renewed.append((lease_id, lease_timeout_sec))
        ),
        has_weight_runtime_manifest_lease=lambda lease_id: True,
    )
    manager = _manager(runner)
    monkeypatch.setattr(weight_updater_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    manager._record_remote_weight_transfer_lease("transfer-1", "lease-0", 30)

    now[0] = 131.0
    result = manager.renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            lease_timeout_sec=60,
        )
    )

    assert result.success is False
    assert "expired" in result.message.lower()
    assert renewed == []
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}
    assert manager.remote_weight_transfer_deadlines == {"transfer-1": 130.0}
    assert manager.remote_weight_transfer_expired == {"transfer-1"}


def test_expired_remote_transfer_bookkeeping_still_releases_coordinator_lease(
    monkeypatch,
) -> None:
    now = [100.0]
    released = []
    runner = SimpleNamespace(
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
        has_weight_runtime_manifest_lease=lambda lease_id: True,
    )
    manager = _manager(runner)
    monkeypatch.setattr(weight_updater_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    manager._record_remote_weight_transfer_lease("transfer-1", "lease-0", 30)

    now[0] = 131.0
    manager._prune_remote_weight_transfer_bookkeeping()
    assert manager.remote_weight_transfer_expired == {"transfer-1"}
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}

    result = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )

    assert result.success is True
    assert released == ["lease-0"]
    assert manager.remote_weight_transfer_leases == {}
    assert manager.remote_weight_transfer_deadlines == {}
    assert manager.remote_weight_transfer_expired == set()


def test_scheduler_lists_expired_session_without_releasing_lease(monkeypatch) -> None:
    now = [100.0]
    released = []
    runner = SimpleNamespace(
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr(weight_updater_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    manager._record_remote_weight_transfer_lease(
        "transfer-1",
        "lease-0",
        30,
        generation=7,
    )

    now[0] = 131.0
    assert manager.list_remote_instance_weight_transfer_sessions() == [
        {
            "transfer_id": "transfer-1",
            "lease_id": "lease-0",
            "generation": 7,
            "deadline_monotonic_sec": 130.0,
            "expired": True,
            "session_state": "expired",
        }
    ]
    assert released == []
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}
    assert manager.remote_weight_transfer_deadlines == {"transfer-1": 130.0}

    result = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )

    assert result.success is True
    assert released == ["lease-0"]
    assert manager.list_remote_instance_weight_transfer_sessions() == []


def test_session_record_does_not_extend_snapshot_deadline(monkeypatch) -> None:
    now = [100.0]
    manager = _manager(SimpleNamespace())
    monkeypatch.setattr(weight_updater_module.time, "monotonic", lambda: now[0])
    manager._record_remote_weight_transfer_lease(
        "transfer-1",
        "lease-0",
        30,
        generation=1,
    )

    now[0] = 105.0
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
        lease_timeout_sec=30,
    )
    output = BeginRemoteInstanceWeightTransferReqOutput(
        transfer_id="transfer-1",
        success=True,
        message="Success.",
        session_state="created",
        manifests=[_manifest()],
    )
    manager._record_remote_weight_transfer_session(request, "lease-0", output)

    assert manager.remote_weight_transfer_deadlines == {"transfer-1": 130.0}


def test_failed_renew_retains_lease_for_explicit_release(monkeypatch) -> None:
    released = []

    def renew(lease_id, lease_timeout_sec):
        del lease_id, lease_timeout_sec
        raise RuntimeError("coordinator temporarily unavailable")

    runner = SimpleNamespace(
        renew_weight_runtime_manifest=renew,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
        has_weight_runtime_manifest_lease=lambda lease_id: False,
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    manager.remote_weight_transfer_leases["transfer-1"] = "lease-0"

    renewed = manager.renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            lease_timeout_sec=60,
        )
    )
    assert renewed.success is False
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}

    released_result = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )
    assert released_result.success is True
    assert released == ["lease-0"]
    assert manager.remote_weight_transfer_leases == {}


def _tokenizer_manager(begin_results, release):
    events = []

    async def begin_communicator(request):
        events.append(("begin", request.transfer_id))
        return begin_results

    async def pause(request):
        events.append(("pause", request.mode))

    async def resume(request):
        events.append(("continue", request.torch_empty_cache))

    return SimpleNamespace(
        server_args=SimpleNamespace(
            enable_weight_runtime_manifest=True,
            model_path="Qwen/Qwen3.5-0.8B",
            revision="main",
        ),
        auto_create_handle_loop=lambda: None,
        is_pause=False,
        pause_generation=pause,
        continue_generation=resume,
        begin_remote_instance_weight_transfer_communicator=begin_communicator,
        release_remote_instance_weight_transfer_communicator=release,
        _remote_weight_transfer_events=events,
    )


def test_tokenizer_begin_pauses_only_snapshot_capture() -> None:
    async def release(request):
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                manifests=[_manifest()],
            )
        ],
        release,
    )

    asyncio.run(TokenizerControlMixin.begin_remote_instance_weight_transfer(manager))

    assert [event[0] for event in manager._remote_weight_transfer_events] == [
        "pause",
        "begin",
        "continue",
    ]
    assert manager._remote_weight_transfer_events[-1] == ("continue", False)


def test_tokenizer_begin_resume_failure_keeps_session_discoverable_and_releasable(
    monkeypatch,
) -> None:
    release_requests = []
    resume_attempts = []

    async def release(request):
        release_requests.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                manifests=[_manifest()],
            )
        ],
        release,
    )

    async def pause(request):
        manager._remote_weight_transfer_events.append(("pause", request.mode))
        manager.is_pause = True

    async def fail_resume_once(request):
        del request
        resume_attempts.append("resume")
        manager.is_pause = False
        if len(resume_attempts) == 1:
            raise RuntimeError("source resume failed")

    manager.pause_generation = pause
    manager.continue_generation = fail_resume_once

    with pytest.raises(RemoteInstanceWeightTransferBeginError) as raised:
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                lease_timeout_sec=60,
                transfer_id="transfer-1",
            )
        )

    assert raised.value.transfer_id == "transfer-1"
    assert raised.value.session_state == "cleanup_pending"
    assert "source resume failed" in str(raised.value)
    status = asyncio.run(
        TokenizerControlMixin.get_remote_instance_weight_transfer_session(
            manager, "transfer-1"
        )
    )
    assert status["lease_ids"] == ["lease-0"]
    assert status["session_state"] == "cleanup_pending"

    success, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            manager, "transfer-1"
        )
    )

    assert success is True
    assert release_requests == ["transfer-1"]
    assert resume_attempts == ["resume", "resume"]
    assert manager.is_pause is False
    released = asyncio.run(
        TokenizerControlMixin.get_remote_instance_weight_transfer_session(
            manager, "transfer-1"
        )
    )
    assert released["session_state"] == "released"


def test_tokenizer_begin_cancellation_waits_for_snapshot_and_releases() -> None:
    async def scenario():
        begin_started = asyncio.Event()
        finish_begin = asyncio.Event()
        release_requests = []

        async def release(request):
            release_requests.append(request.transfer_id)
            return [SimpleNamespace(success=True, message="Success.")]

        manager = _tokenizer_manager([], release)

        async def begin(request):
            begin_started.set()
            await finish_begin.wait()
            return [
                SimpleNamespace(
                    transfer_id=request.transfer_id,
                    success=True,
                    message="Success.",
                    session_state="created",
                    manifests=[_manifest()],
                )
            ]

        manager.begin_remote_instance_weight_transfer_communicator = begin
        task = asyncio.create_task(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                lease_timeout_sec=60,
                transfer_id="transfer-cancelled",
            )
        )
        await begin_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        finish_begin.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert release_requests == ["transfer-cancelled"]
        status = (
            await TokenizerControlMixin.get_remote_instance_weight_transfer_session(
                manager, "transfer-cancelled"
            )
        )
        assert status["session_state"] == "released"
        assert manager._remote_weight_transfer_events[-1] == ("continue", False)

    asyncio.run(scenario())


def test_tokenizer_cancelled_reused_begin_keeps_existing_session() -> None:
    async def scenario():
        begin_started = asyncio.Event()
        finish_begin = asyncio.Event()
        release_requests = []

        async def release(request):
            release_requests.append(request.transfer_id)
            return [SimpleNamespace(success=True, message="Success.")]

        manager = _tokenizer_manager(
            [
                SimpleNamespace(
                    transfer_id="transfer-1",
                    success=True,
                    message="Success.",
                    session_state="created",
                    manifests=[_manifest()],
                )
            ],
            release,
        )
        await TokenizerControlMixin.begin_remote_instance_weight_transfer(
            manager,
            lease_timeout_sec=60,
            transfer_id="transfer-1",
        )

        async def reused_begin(request):
            begin_started.set()
            await finish_begin.wait()
            return [
                SimpleNamespace(
                    transfer_id=request.transfer_id,
                    success=True,
                    message="Success.",
                    session_state="reused",
                    manifests=[_manifest()],
                )
            ]

        manager.begin_remote_instance_weight_transfer_communicator = reused_begin
        task = asyncio.create_task(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                lease_timeout_sec=60,
                transfer_id="transfer-1",
            )
        )
        await begin_started.wait()
        task.cancel()
        finish_begin.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert release_requests == []
        status = (
            await TokenizerControlMixin.get_remote_instance_weight_transfer_session(
                manager, "transfer-1"
            )
        )
        assert status["session_state"] == "active"
        assert status["lease_ids"] == ["lease-0"]

    asyncio.run(scenario())


def test_tokenizer_cancelled_created_begin_serializes_same_id_retry() -> None:
    async def scenario():
        session_exists = False
        first_resume_started = asyncio.Event()
        finish_first_resume = asyncio.Event()
        second_begin_started = asyncio.Event()
        begin_calls = 0
        resume_calls = 0
        release_requests = []

        async def release(request):
            nonlocal session_exists
            release_requests.append(request.transfer_id)
            session_exists = False
            return [SimpleNamespace(success=True, message="Success.")]

        manager = _tokenizer_manager([], release)

        async def begin(request):
            nonlocal begin_calls, session_exists
            begin_calls += 1
            if begin_calls == 2:
                second_begin_started.set()
            session_state = "reused" if session_exists else "created"
            session_exists = True
            return [
                SimpleNamespace(
                    transfer_id=request.transfer_id,
                    success=True,
                    message="Success.",
                    session_state=session_state,
                    manifests=[_manifest()],
                )
            ]

        async def pause(request):
            del request
            manager.is_pause = True

        async def resume(request):
            nonlocal resume_calls
            del request
            resume_calls += 1
            if resume_calls == 1:
                first_resume_started.set()
                await finish_first_resume.wait()
            manager.is_pause = False

        manager.begin_remote_instance_weight_transfer_communicator = begin
        manager.pause_generation = pause
        manager.continue_generation = resume

        first = asyncio.create_task(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                lease_timeout_sec=60,
                transfer_id="transfer-1",
            )
        )
        await first_resume_started.wait()
        first.cancel()
        second = asyncio.create_task(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                lease_timeout_sec=60,
                transfer_id="transfer-1",
            )
        )
        await asyncio.sleep(0)
        retry_interleaved = second_begin_started.is_set()

        finish_first_resume.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        result = await second

        assert retry_interleaved is False
        assert result["transfer_id"] == "transfer-1"
        assert begin_calls == 2
        assert release_requests == ["transfer-1"]
        assert session_exists is True

    asyncio.run(scenario())


def test_tokenizer_reused_begin_resume_failure_keeps_existing_session() -> None:
    release_requests = []
    resume_calls = []

    async def release(request):
        release_requests.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                transfer_id="transfer-1",
                success=True,
                message="Success.",
                session_state="created",
                manifests=[_manifest()],
            )
        ],
        release,
    )
    asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            manager,
            lease_timeout_sec=60,
            transfer_id="transfer-1",
        )
    )

    async def reused_begin(request):
        return [
            SimpleNamespace(
                transfer_id=request.transfer_id,
                success=True,
                message="Success.",
                session_state="reused",
                manifests=[_manifest()],
            )
        ]

    manager.begin_remote_instance_weight_transfer_communicator = reused_begin

    async def fail_resume(request):
        del request
        resume_calls.append("resume")
        manager.is_pause = False
        raise RuntimeError("source resume failed")

    manager.continue_generation = fail_resume
    with pytest.raises(RemoteInstanceWeightTransferBeginError) as raised:
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                lease_timeout_sec=60,
                transfer_id="transfer-1",
            )
        )

    assert raised.value.session_state == "reused"
    assert release_requests == []
    assert resume_calls == ["resume"]
    status = asyncio.run(
        TokenizerControlMixin.get_remote_instance_weight_transfer_session(
            manager, "transfer-1"
        )
    )
    assert status["session_state"] == "active"
    assert status["lease_ids"] == ["lease-0"]


def test_tokenizer_cancel_during_resume_finishes_resume_and_tracks_cleanup() -> None:
    async def scenario():
        resume_started = asyncio.Event()
        finish_resume = asyncio.Event()
        release_requests = []

        async def release(request):
            release_requests.append(request.transfer_id)
            return [SimpleNamespace(success=False, message="release still pending")]

        manager = _tokenizer_manager(
            [
                SimpleNamespace(
                    transfer_id="transfer-cancelled",
                    success=True,
                    message="Success.",
                    session_state="created",
                    manifests=[_manifest()],
                )
            ],
            release,
        )

        async def pause(request):
            del request
            manager.is_pause = True

        async def resume(request):
            del request
            resume_started.set()
            await finish_resume.wait()
            manager.is_pause = False

        manager.pause_generation = pause
        manager.continue_generation = resume
        task = asyncio.create_task(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                lease_timeout_sec=60,
                transfer_id="transfer-cancelled",
            )
        )
        await resume_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        finish_resume.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert manager.is_pause is False
        assert release_requests == ["transfer-cancelled"]
        status = (
            await TokenizerControlMixin.get_remote_instance_weight_transfer_session(
                manager, "transfer-cancelled"
            )
        )
        assert status["session_state"] == "cleanup_pending"
        assert status["last_release_success"] is False

    asyncio.run(scenario())


def test_tokenizer_cancel_during_failed_begin_cleanup_finishes_release() -> None:
    async def scenario():
        cleanup_pause_started = asyncio.Event()
        finish_cleanup_pause = asyncio.Event()
        pause_calls = 0
        release_requests = []

        async def release(request):
            release_requests.append(request.transfer_id)
            return [SimpleNamespace(success=True, message="Success.")]

        manager = _tokenizer_manager(
            [
                SimpleNamespace(
                    transfer_id="transfer-cancelled",
                    success=True,
                    message="Success.",
                    session_state="created",
                    manifests=[_manifest()],
                ),
                SimpleNamespace(
                    transfer_id="transfer-cancelled",
                    success=False,
                    message="source worker failed",
                    session_state="failed",
                    manifests=[],
                ),
            ],
            release,
        )

        async def pause(request):
            nonlocal pause_calls
            del request
            pause_calls += 1
            manager.is_pause = True
            if pause_calls == 2:
                cleanup_pause_started.set()
                await finish_cleanup_pause.wait()

        async def resume(request):
            del request
            manager.is_pause = False

        manager.pause_generation = pause
        manager.continue_generation = resume
        task = asyncio.create_task(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                lease_timeout_sec=60,
                transfer_id="transfer-cancelled",
            )
        )
        await cleanup_pause_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        finish_cleanup_pause.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert release_requests == ["transfer-cancelled"]
        assert manager.is_pause is False
        status = (
            await TokenizerControlMixin.get_remote_instance_weight_transfer_session(
                manager, "transfer-cancelled"
            )
        )
        assert status["session_state"] == "released"

    asyncio.run(scenario())


def test_tokenizer_begin_passes_ttl_to_scheduler_without_local_ownership() -> None:
    requests = []

    async def begin(request):
        requests.append(request)
        return [
            SimpleNamespace(
                success=True,
                message="Success.",
                manifests=[_manifest()],
            )
        ]

    async def release(request):
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)
    manager.begin_remote_instance_weight_transfer_communicator = begin

    result = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            manager, lease_timeout_sec=60
        )
    )

    assert requests[0].lease_timeout_sec == 60
    assert result["lease_timeout_sec"] == 60
    assert not hasattr(manager, "_remote_weight_transfer_timeout_tasks")


def test_tokenizer_begin_returns_split_source_manifest() -> None:
    requests = []
    placement = _placement()
    binding = _binding()

    async def begin(request):
        requests.append(request)
        return [
            SimpleNamespace(
                success=True,
                message="Success.",
                manifests=None,
                placements=[placement],
                bindings=[binding],
            )
        ]

    async def release(request):
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)
    manager.begin_remote_instance_weight_transfer_communicator = begin

    result = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            manager,
            lease_timeout_sec=60,
            manifest_format="placement_binding_v1",
        )
    )

    assert requests[0].manifest_format == "placement_binding_v1"
    assert result == {
        "transfer_id": requests[0].transfer_id,
        "source_weight_placements": [placement],
        "source_weight_runtime_bindings": [binding],
        "lease_timeout_sec": 60,
    }


def test_tokenizer_begin_merges_split_dp_replica_manifests() -> None:
    async def release(request):
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                placements=[_placement(dp_rank=0)],
                bindings=[_binding(dp_rank=0, lease_id="lease-dp0")],
            ),
            SimpleNamespace(
                success=True,
                message="Success.",
                placements=[_placement(dp_rank=1)],
                bindings=[_binding(dp_rank=1, lease_id="lease-dp1")],
            ),
        ],
        release,
    )

    result = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            manager, manifest_format="placement_binding_v1"
        )
    )

    assert [
        placement["tensors"][0]["rank"]["dp"]
        for placement in result["source_weight_placements"]
    ] == [0, 1]
    assert [
        binding["fragments"][0]["worker_id"]
        for binding in result["source_weight_runtime_bindings"]
    ] == ["source/dp0-pp0-ep0-tp0", "source/dp1-pp0-ep0-tp0"]
    status = asyncio.run(
        TokenizerControlMixin.get_remote_instance_weight_transfer_session(
            manager, result["transfer_id"]
        )
    )
    assert status["lease_id"] is None
    assert status["lease_ids"] == ["lease-dp0", "lease-dp1"]
    assert status["generation"] == 1


def test_tokenizer_begin_rejects_split_dp_generation_mismatch() -> None:
    released = []

    async def release(request):
        released.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    second_binding = _binding(dp_rank=1, lease_id="lease-dp1")
    second_binding["generation"] = 2
    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                placements=[_placement(dp_rank=0)],
                bindings=[_binding(dp_rank=0, lease_id="lease-dp0")],
            ),
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                placements=[_placement(dp_rank=1)],
                bindings=[second_binding],
            ),
        ],
        release,
    )

    with pytest.raises(RuntimeError, match="one model generation"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager, manifest_format="placement_binding_v1"
            )
        )

    assert len(released) == 1


def test_tokenizer_begin_conflict_does_not_release_existing_session() -> None:
    released = []

    async def release(request):
        released.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=False,
                message="remote weight transfer ID was reused",
                session_state="conflict",
            )
        ],
        release,
    )

    with pytest.raises(RuntimeError, match="ID was reused"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                transfer_id="transfer-1",
            )
        )

    assert released == []


def test_tokenizer_begin_retries_cleanup_for_created_and_failed_dp_results() -> None:
    release_attempts = []

    async def release(request):
        release_attempts.append(request.transfer_id)
        return [
            SimpleNamespace(
                success=len(release_attempts) >= 2,
                message="Success." if len(release_attempts) >= 2 else "retry",
            )
        ]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                manifests=[_manifest(worker_id="source/dp0-pp0-ep0-tp0")],
            ),
            SimpleNamespace(
                success=False,
                message="source rank failed",
                session_state="failed",
            ),
        ],
        release,
    )

    with pytest.raises(RuntimeError, match="source rank failed") as raised:
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                transfer_id="transfer-1",
            )
        )

    assert raised.value.transfer_id == "transfer-1"
    assert raised.value.session_state == "failed"
    assert release_attempts == ["transfer-1", "transfer-1"]


def test_tokenizer_begin_reports_cleanup_pending_when_release_never_succeeds() -> None:
    release_attempts = []

    async def release(request):
        release_attempts.append(request.transfer_id)
        return [SimpleNamespace(success=False, message="still busy")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=False,
                message="snapshot cleanup remains pending",
                session_state="cleanup_pending",
            )
        ],
        release,
    )

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                transfer_id="transfer-1",
            )
        )

    assert raised.value.transfer_id == "transfer-1"
    assert raised.value.session_state == "cleanup_pending"
    assert release_attempts == ["transfer-1", "transfer-1", "transfer-1"]


@pytest.mark.parametrize(
    ("begin_results", "expected_released"),
    [
        pytest.param(
            [
                SimpleNamespace(
                    success=True,
                    message="Success.",
                    session_state="reused",
                    manifests=[_manifest(worker_id="source/dp0-pp0-ep0-tp0")],
                ),
                SimpleNamespace(
                    success=False,
                    message="source rank failed",
                    session_state="failed",
                ),
            ],
            [],
            id="reused-is-not-owned",
        ),
        pytest.param(
            [
                SimpleNamespace(
                    success=False,
                    message="snapshot cleanup remains pending",
                    session_state="cleanup_pending",
                ),
                SimpleNamespace(
                    success=False,
                    message="already released",
                    session_state="released",
                ),
            ],
            ["transfer-1"],
            id="cleanup-pending-is-owned",
        ),
    ],
)
def test_tokenizer_begin_cleans_only_owned_session_states(
    begin_results, expected_released
) -> None:
    released = []

    async def release(request):
        released.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(begin_results, release)

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                transfer_id="transfer-1",
            )
        )

    assert raised.value.transfer_id == "transfer-1"
    assert raised.value.session_state == "failed"
    assert released == expected_released


def test_tokenizer_begin_rejects_duplicate_split_fragment_ids() -> None:
    released = []
    placement = _placement()
    placement["tensors"].append(dict(placement["tensors"][0]))
    binding = _binding()
    binding["fragments"].append(dict(binding["fragments"][0]))

    async def release(request):
        released.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                placements=[placement],
                bindings=[binding],
            )
        ],
        release,
    )

    with pytest.raises(RuntimeError, match="duplicate placement fragment"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager, manifest_format="placement_binding_v1"
            )
        )

    assert len(released) == 1


def test_tokenizer_release_always_fans_out_without_local_session_state() -> None:
    requests = []

    async def release(request):
        requests.append(request)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)
    success, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            manager, "transfer-from-another-worker"
        )
    )

    assert success is True
    assert [request.transfer_id for request in requests] == [
        "transfer-from-another-worker"
    ]


def test_tokenizer_renew_fans_out_without_local_session_state() -> None:
    requests = []

    async def renew(request):
        requests.append(request)
        return [SimpleNamespace(success=True, message="Success.")]

    async def release(request):
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)
    manager.renew_remote_instance_weight_transfer_communicator = renew

    success, _ = asyncio.run(
        TokenizerControlMixin.renew_remote_instance_weight_transfer(
            manager, "transfer-from-another-worker", lease_timeout_sec=60
        )
    )

    assert success is True
    assert requests == [
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-from-another-worker", lease_timeout_sec=60
        )
    ]
    assert manager._remote_weight_transfer_events == []


def test_tokenizer_lists_active_then_expired_session_without_auto_release(
    monkeypatch,
) -> None:
    now = [100.0]
    release_requests = []

    async def release(request):
        release_requests.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                manifests=[_manifest()],
            )
        ],
        release,
    )
    monkeypatch.setattr(tokenizer_control_mixin_module.time, "time", lambda: now[0])

    result = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            manager,
            lease_timeout_sec=60,
            transfer_id="transfer-1",
        )
    )
    assert result["transfer_id"] == "transfer-1"

    sessions = asyncio.run(
        TokenizerControlMixin.list_remote_instance_weight_transfer_sessions(manager)
    )
    assert sessions == [
        {
            "transfer_id": "transfer-1",
            "lease_id": "lease-0",
            "lease_ids": ["lease-0"],
            "generation": 1,
            "manifest_format": "runtime_v1",
            "deadline_unix_sec": 160.0,
            "expired": False,
            "session_state": "active",
            "last_release_attempt_unix_sec": None,
            "last_release_success": None,
            "last_release_message": None,
        }
    ]

    now[0] = 161.0
    status = asyncio.run(
        TokenizerControlMixin.get_remote_instance_weight_transfer_session(
            manager, "transfer-1"
        )
    )
    assert status["expired"] is True
    assert status["session_state"] == "expired"
    assert release_requests == []


def test_tokenizer_failed_manual_release_keeps_discoverable_session(
    monkeypatch,
) -> None:
    now = [100.0]
    release_results = [
        SimpleNamespace(success=False, message="release still unsafe"),
        SimpleNamespace(success=True, message="Success."),
    ]

    async def release(request):
        del request
        return [release_results.pop(0)]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                manifests=[_manifest()],
            )
        ],
        release,
    )
    monkeypatch.setattr(tokenizer_control_mixin_module.time, "time", lambda: now[0])
    asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            manager,
            lease_timeout_sec=60,
            transfer_id="transfer-1",
        )
    )

    now[0] = 161.0
    success, message = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            manager, "transfer-1"
        )
    )
    assert success is False
    assert message == "release still unsafe"
    failed_status = asyncio.run(
        TokenizerControlMixin.get_remote_instance_weight_transfer_session(
            manager, "transfer-1"
        )
    )
    assert failed_status["session_state"] == "expired"
    assert failed_status["last_release_attempt_unix_sec"] == 161.0
    assert failed_status["last_release_success"] is False
    assert failed_status["last_release_message"] == "release still unsafe"

    success, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            manager, "transfer-1"
        )
    )
    assert success is True
    released_status = asyncio.run(
        TokenizerControlMixin.get_remote_instance_weight_transfer_session(
            manager, "transfer-1"
        )
    )
    assert released_status["session_state"] == "released"
    assert released_status["last_release_success"] is True


def test_tokenizer_begin_releases_successful_empty_manifest_response() -> None:
    released = []

    async def release(request):
        released.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                manifests=[],
            )
        ],
        release,
    )

    with pytest.raises(RuntimeError, match="no runtime manifests"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(manager)
        )

    assert len(released) == 1


def test_tokenizer_begin_merges_consistent_dp_replica_manifests() -> None:
    released = []

    async def release(request):
        released.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                manifests=[_manifest(worker_id="source/dp0-pp0-ep0-tp0")],
            ),
            SimpleNamespace(
                success=True,
                message="Success.",
                manifests=[_manifest(worker_id="source/dp1-pp0-ep0-tp0")],
            ),
        ],
        release,
    )

    result = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(manager)
    )

    assert [
        manifest["tensors"][0]["worker_id"]
        for manifest in result["weight_runtime_manifests"]
    ] == ["source/dp0-pp0-ep0-tp0", "source/dp1-pp0-ep0-tp0"]
    assert released == []


def test_tokenizer_begin_rejects_semantically_inconsistent_dp_replica() -> None:
    released = []

    async def release(request):
        released.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    inconsistent = _manifest(worker_id="source/dp1-pp0-ep0-tp0")
    inconsistent["generation"] = 2
    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                manifests=[_manifest(worker_id="source/dp0-pp0-ep0-tp0")],
            ),
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                manifests=[inconsistent],
            ),
        ],
        release,
    )

    with pytest.raises(RuntimeError, match="semantically inconsistent"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(manager)
        )

    assert len(released) == 1


def test_tokenizer_begin_rejects_dp_replica_with_different_shard_dims() -> None:
    released = []

    async def release(request):
        released.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    first = _manifest(worker_id="source/dp0-pp0-ep0-tp0")
    second = _manifest(worker_id="source/dp1-pp0-ep0-tp0")
    first["tensors"][0]["shard_dims"] = [0]
    second["tensors"][0]["shard_dims"] = [1]
    manager = _tokenizer_manager(
        [
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                manifests=[first],
            ),
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                manifests=[second],
            ),
        ],
        release,
    )

    with pytest.raises(RuntimeError, match="semantically inconsistent"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(manager)
        )

    assert len(released) == 1
