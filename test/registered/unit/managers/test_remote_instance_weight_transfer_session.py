import asyncio
import sys
import threading
import time
from collections import deque
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.configs.load_config import LoadFormat
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.managers import multi_tokenizer_mixin as multi_tokenizer_mixin_module
from sglang.srt.managers import (
    tokenizer_control_mixin as tokenizer_control_mixin_module,
)
from sglang.srt.managers.communicator import (
    FanOutCompletionUnknownError,
    FanOutDeadlineExpiredBeforeDispatch,
)
from sglang.srt.managers.io_struct import (
    BeginRemoteInstanceWeightTransferReqInput,
    BeginRemoteInstanceWeightTransferReqOutput,
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    ReleaseMemoryOccupationReqInput,
    ReleaseRemoteInstanceWeightTransferReqInput,
    RenewRemoteInstanceWeightTransferReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    WeightSnapshotActivationReqInput,
    weight_snapshot_activation_request_context,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    MultiTokenizerRouter,
    TokenizerWorker,
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
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightManifestError,
    WeightSnapshotCoordinator,
)
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.srt.weight_transfer.remote_protocol import (
    ARTIFACT_WEIGHT_VERSION_V1,
    HF_REVISION_V1,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _register_live_tokenizer_workers(router):
    pid = multi_tokenizer_mixin_module.os.getpid()
    process_start_time = multi_tokenizer_mixin_module.psutil.Process(pid).create_time()
    workers = {
        ipc_name: multi_tokenizer_mixin_module._TokenizerWorkerIdentity(
            ipc_name=ipc_name,
            pid=pid,
            process_start_time=process_start_time,
            token=ipc_name,
        )
        for ipc_name in ("worker-a", "worker-b")
    }
    router.server_args = SimpleNamespace(tokenizer_worker_num=len(workers))
    router._worker_registrations = workers
    router._pause_owner_transitions = {}
    router._pause_owner_workers = {}
    router._pause_fail_stopped = False
    return workers


def _finalize_router_transition(router, identity):
    transition = router._pause_transitions.get(identity.transition_id)
    if (
        transition is None
        or not transition.committed
        or transition.commit_pending_workers
    ):
        return
    router._handle_pause_continue_ack(
        multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
            rid=multi_tokenizer_mixin_module._encode_pause_transition_finalized_ack(
                identity
            ),
            is_pause=identity.expected_state,
            http_worker_ipc=transition.origin_worker_ipc,
        )
    )


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


class _RemoteTransferCollective:
    def __init__(
        self,
        *,
        rank,
        gather_results,
        broadcast_results=(),
    ):
        self.rank_in_group = rank
        self.world_size = len(gather_results[0])
        self._gather_results = deque(gather_results)
        self._broadcast_results = deque(broadcast_results)
        self.gathered = []
        self.all_gathered = []
        self.broadcasts = []

    def gather_object(self, obj, dst=0):
        assert dst == 0
        self.gathered.append(obj)
        result = list(self._gather_results.popleft())
        result[self.rank_in_group] = obj
        return result if self.rank_in_group == dst else None

    def scatter_object(self, objects=None, src=0):
        assert src == 0
        if self.rank_in_group == src:
            assert len(objects) == self.world_size
            obj = objects[self.rank_in_group]
            self.broadcasts.append(obj)
            return obj
        assert objects is None
        result = self._broadcast_results.popleft()
        self.broadcasts.append(result)
        return result

    def all_gather_object(self, obj):
        self.all_gathered.append(obj)
        return [obj] * self.world_size


class _BoundedRemoteTransferCollective(_RemoteTransferCollective):
    cpu_group = object()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bounded_calls = []

    def gather_object(
        self,
        obj,
        dst=0,
        *,
        phase=None,
        execution_context=None,
    ):
        self.bounded_calls.append((phase, execution_context))
        return super().gather_object(obj, dst=dst)

    def scatter_object(
        self,
        objects=None,
        src=0,
        *,
        phase=None,
        execution_context=None,
    ):
        self.bounded_calls.append((phase, execution_context))
        return super().scatter_object(objects, src=src)

    def all_gather_object(
        self,
        obj,
        *,
        phase=None,
        execution_context=None,
    ):
        self.bounded_calls.append((phase, execution_context))
        return super().all_gather_object(obj)


class _AsymmetricBroadcastFailureCollective:
    def __init__(self, process_group, *, rank, world_size):
        self.process_group = process_group
        self.rank_in_group = rank
        self.world_size = world_size
        self.ranks = list(range(world_size))

    def gather_object(self, obj, dst=0):
        outputs = [None] * self.world_size if self.rank_in_group == dst else None
        torch.distributed.gather_object(
            obj,
            outputs,
            dst=dst,
            group=self.process_group,
        )
        return outputs

    def scatter_object(self, objects=None, src=0):
        if self.rank_in_group == 1:
            raise RuntimeError("injected broadcast failure")
        output = [None]
        torch.distributed.scatter_object_list(
            output,
            objects,
            src=src,
            group=self.process_group,
        )
        return output[0]


@pytest.fixture(autouse=True)
def _adapt_legacy_remote_transfer_collective_fakes(monkeypatch, request):
    if not request.node.name.startswith(
        ("test_begin_", "test_duplicate_begin_", "test_late_begin_")
    ):
        return
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(
        torch.distributed,
        "get_global_rank",
        lambda group, rank: rank,
    )

    def gather_object(value, gathered, dst, group):
        assert dst == 0
        outputs = [None] * torch.distributed.get_world_size(group=group)
        torch.distributed.all_gather_object(outputs, value, group=group)
        if gathered is not None:
            gathered[:] = outputs

    monkeypatch.setattr(torch.distributed, "gather_object", gather_object)
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda payload, src, group: None,
    )


def _manifest(worker_id="source/dp0-pp0-ep0-tp0", lease_id="lease-0"):
    return {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main",
        "generation": 1,
        "lease_id": lease_id,
        "tensors": [{"worker_id": worker_id}],
    }


_BEGIN_REQUEST_TYPE = BeginRemoteInstanceWeightTransferReqInput


def _begin_request(**kwargs):
    kwargs.setdefault("deadline_unix_sec", time.time() + 60)
    return _BEGIN_REQUEST_TYPE(**kwargs)


def _placement(dp_rank=0):
    return {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main",
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
        "revision": "main",
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
    request = _begin_request(
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


@pytest.mark.parametrize("action", ["activate", "close"])
def test_scheduler_dispatches_weight_snapshot_activation_action(action: str) -> None:
    actions = []
    runner = SimpleNamespace(
        activate_pending_weight_snapshot=lambda: actions.append("activate"),
        close_pending_weight_snapshot_activation=lambda: actions.append("close"),
    )
    manager = _manager(runner)

    result = manager.update_weight_snapshot_activation(
        WeightSnapshotActivationReqInput(action=action)
    )

    assert result.success is True
    assert result.action == action
    assert actions == [action]


def test_scheduler_routes_expired_structured_close_without_touching_owner() -> None:
    class Pending:
        close_calls = 0

        def close(self):
            self.close_calls += 1

    pending = Pending()
    runner = SimpleNamespace(
        load_config=SimpleNamespace(load_format=LoadFormat.WEIGHT_SNAPSHOT),
        pending_weight_snapshot_activation=pending,
    )
    runner.activate_pending_weight_snapshot = lambda: (
        model_runner_module.ModelRunner.activate_pending_weight_snapshot(runner)
    )
    runner.close_pending_weight_snapshot_activation = lambda: (
        model_runner_module.ModelRunner.close_pending_weight_snapshot_activation(runner)
    )
    manager = _manager(runner)

    request = WeightSnapshotActivationReqInput(
        action="close",
        phase="close",
        transaction_id="activation-1",
        request_id="close-1",
        deadline_unix_sec=time.time() - 1,
    )
    with weight_snapshot_activation_request_context(request):
        result = manager.update_weight_snapshot_activation(request)

    assert result.success is False
    assert "deadline expired" in result.message
    assert pending.close_calls == 0
    assert runner.pending_weight_snapshot_activation is pending


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
    request = _begin_request(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
        request_id="begin-attempt-1",
    )
    result = manager.begin_remote_instance_weight_transfer(request)

    assert result.success is True
    assert result.manifests == [manifest]
    assert result.request_id == request.request_id
    assert result.external_dp_rank == 0
    assert released == []

    release = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            request_id="release-attempt-1",
        )
    )
    assert release.success is True
    assert release.request_id == "release-attempt-1"
    assert release.external_dp_rank == 0
    assert released == ["lease-0"]


@pytest.mark.parametrize(
    ("deadline_unix_sec", "message"),
    [
        (None, "deadline is required"),
        (time.time() - 1, "deadline is invalid or expired"),
    ],
)
def test_begin_votes_unusable_deadline_before_snapshot_or_manifest_gather(
    monkeypatch,
    deadline_unix_sec,
    message,
) -> None:
    snapshot_calls = []
    vote_calls = []
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
            snapshot_calls.append(kwargs) or _manifest()
        ),
    )
    manager = _manager(runner)

    def all_gather(
        _self,
        value,
        group,
        *,
        phase=None,
        execution_context=None,
    ):
        assert execution_context is not None
        if deadline_unix_sec is None:
            assert execution_context.deadline_unix_sec > time.time()
        else:
            assert execution_context.deadline_unix_sec == deadline_unix_sec
        assert execution_context.deadline_unix_sec <= time.time() + 31
        vote_calls.append((value, group, phase))
        return [value]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_all_gather_remote_transfer_object",
        all_gather,
    )

    result = manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-deadline",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=deadline_unix_sec,
        )
    )

    assert result.success is False
    assert message in result.message.lower()
    assert snapshot_calls == []
    assert len(vote_calls) == 1
    assert vote_calls[0][2] == "remote_instance.source.begin_vote"
    assert vote_calls[0][0]["success"] is False


def test_begin_rechecks_original_deadline_after_vote_before_snapshot(
    monkeypatch,
) -> None:
    now = [100.0]
    snapshot_calls = []
    vote_deadlines = []
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
            snapshot_calls.append(kwargs) or _manifest()
        ),
    )
    manager = _manager(runner)
    monkeypatch.setattr(weight_updater_module.time, "time", lambda: now[0])

    def all_gather(
        _self,
        value,
        group,
        *,
        phase=None,
        execution_context=None,
    ):
        del group, phase
        assert execution_context is not None
        vote_deadlines.append(execution_context.deadline_unix_sec)
        now[0] = 102.0
        return [value]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_all_gather_remote_transfer_object",
        all_gather,
    )

    result = manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-deadline-after-vote",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=101.0,
        )
    )

    assert result.success is False
    assert "deadline expired before snapshot acquisition" in result.message.lower()
    assert vote_deadlines == [101.0]
    assert snapshot_calls == []


def test_begin_votes_local_poison_before_snapshot_or_manifest_gather(
    monkeypatch,
) -> None:
    snapshot_calls = []
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
            snapshot_calls.append(kwargs) or _manifest()
        ),
    )
    manager = _manager(runner)
    manager.remote_weight_transfer_snapshot_poisoned = "snapshot lane poisoned"
    vote_calls = []

    def all_gather(
        _self,
        value,
        group,
        *,
        phase=None,
        execution_context=None,
    ):
        vote_calls.append((value, group, phase, execution_context))
        return [value]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_all_gather_remote_transfer_object",
        all_gather,
    )

    result = manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-poisoned",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is False
    assert "scheduler restart is required" in result.message
    assert snapshot_calls == []
    assert len(vote_calls) == 1
    assert vote_calls[0][2] == "remote_instance.source.begin_vote"
    assert vote_calls[0][0]["poisoned"] is True


def test_begin_votes_local_conflict_before_snapshot_or_manifest_gather(
    monkeypatch,
) -> None:
    snapshot_calls = []
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
            snapshot_calls.append(kwargs) or _manifest()
        ),
    )
    manager = _manager(runner)
    manager._record_remote_weight_transfer_lease(
        "transfer-conflict",
        "existing-lease",
        60,
    )
    vote_calls = []

    def all_gather(
        _self,
        value,
        group,
        *,
        phase=None,
        execution_context=None,
    ):
        vote_calls.append((value, group, phase, execution_context))
        return [value]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_all_gather_remote_transfer_object",
        all_gather,
    )

    result = manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-conflict",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is False
    assert result.session_state == "cleanup_pending"
    assert "already exists" in result.message
    assert snapshot_calls == []
    assert len(vote_calls) == 1
    assert vote_calls[0][2] == "remote_instance.source.begin_vote"


def test_peer_begin_vote_rejection_stops_all_ranks_before_snapshot(
    monkeypatch,
) -> None:
    snapshot_calls = []
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
            snapshot_calls.append(kwargs) or _manifest()
        ),
    )
    manager = _manager(runner)
    vote_calls = []

    def all_gather(
        _self,
        value,
        group,
        *,
        phase=None,
        execution_context=None,
    ):
        vote_calls.append((value, group, phase, execution_context))
        return [
            value,
            {
                "success": False,
                "message": "Remote weight transfer deadline is invalid or expired.",
                "session_state": "failed",
                "poisoned": False,
            },
        ]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_all_gather_remote_transfer_object",
        all_gather,
    )
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_remote_transfer_object",
        lambda *_args, **_kwargs: pytest.fail(
            "manifest gather must not run after begin vote rejection"
        ),
    )

    result = manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-peer-expired",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is False
    assert result.session_state == "failed"
    assert "rank 1" in result.message
    assert "deadline is invalid or expired" in result.message.lower()
    assert snapshot_calls == []
    assert len(vote_calls) == 1


def test_begin_caches_full_manifest_only_on_response_root() -> None:
    root_snapshots = []
    peer_snapshots = []
    root_manifest = _manifest()
    peer_manifest = _manifest(
        worker_id="source/dp0-pp0-ep0-tp1",
        lease_id="lease-1",
    )
    peer_created = {
        "success": True,
        "message": "Success.",
        "session_state": "created",
        "manifest": peer_manifest,
        "manifest_revision_semantics": HF_REVISION_V1,
        "model_id": peer_manifest["model_id"],
        "revision": peer_manifest["revision"],
        "lease_id": peer_manifest["lease_id"],
        "generation": peer_manifest["generation"],
    }
    root_group = _RemoteTransferCollective(
        rank=0,
        gather_results=[[None, peer_created]],
    )
    root_manager = _manager(
        SimpleNamespace(
            get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
                root_snapshots.append(kwargs) or root_manifest
            ),
            release_weight_runtime_manifest=lambda lease_id: None,
        ),
        remote_weight_transfer_cpu_group=root_group,
    )
    request = _begin_request(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
    )

    root_result = root_manager.begin_remote_instance_weight_transfer(request)

    peer_group = _RemoteTransferCollective(
        rank=1,
        gather_results=[[None, None]],
        broadcast_results=root_group.broadcasts,
    )
    peer_manager = _manager(
        SimpleNamespace(
            get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
                peer_snapshots.append(kwargs) or peer_manifest
            ),
            release_weight_runtime_manifest=lambda lease_id: None,
        ),
        remote_weight_transfer_cpu_group=peer_group,
    )
    peer_result = peer_manager.begin_remote_instance_weight_transfer(request)

    assert root_result.manifests == [root_manifest, peer_manifest]
    assert peer_result.success is True
    assert peer_result.manifests is None
    assert root_manager.remote_weight_transfer_sessions["transfer-1"][1] is root_result
    assert peer_manager.remote_weight_transfer_sessions["transfer-1"][1] is None
    assert root_manager.remote_weight_transfer_leases == {"transfer-1": "lease-0"}
    assert peer_manager.remote_weight_transfer_leases == {"transfer-1": "lease-1"}
    assert root_manager.remote_weight_transfer_generations == {"transfer-1": 1}
    assert peer_manager.remote_weight_transfer_generations == {"transfer-1": 1}
    assert len(root_snapshots) == len(peer_snapshots) == 1


def test_source_manifest_control_uses_one_bounded_deadline() -> None:
    root_manifest = _manifest()
    peer_manifest = _manifest(
        worker_id="source/dp0-pp0-ep0-tp1",
        lease_id="lease-1",
    )
    peer_created = {
        "success": True,
        "message": "Success.",
        "session_state": "created",
        "manifest": peer_manifest,
        "manifest_revision_semantics": HF_REVISION_V1,
        "model_id": peer_manifest["model_id"],
        "revision": peer_manifest["revision"],
        "lease_id": peer_manifest["lease_id"],
        "generation": peer_manifest["generation"],
    }
    group = _BoundedRemoteTransferCollective(
        rank=0,
        gather_results=[[None, peer_created]],
    )
    manager = _manager(
        SimpleNamespace(
            get_remote_instance_weight_runtime_manifest=lambda **kwargs: root_manifest,
            release_weight_runtime_manifest=lambda lease_id: None,
        ),
        remote_weight_transfer_cpu_group=group,
    )

    result = manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-1",
            model_id=root_manifest["model_id"],
            revision=root_manifest["revision"],
            lease_timeout_sec=60,
        )
    )

    assert result.success is True
    released, _ = manager._gather_remote_weight_transfer_status(
        success=True,
        message="Success.",
        operation="release",
    )
    assert released is True
    assert [phase for phase, _ in group.bounded_calls] == [
        "remote_instance.source.begin_vote",
        "remote_instance.source.manifest_gather",
        "remote_instance.source.decision_scatter",
        "remote_instance.source.release_gather",
    ]
    contexts = [context for _, context in group.bounded_calls]
    assert 0 < contexts[0].remaining_seconds() <= 30
    assert contexts[1] is contexts[2]
    assert 0 < contexts[1].remaining_seconds() <= 60
    assert 0 < contexts[3].remaining_seconds() <= 30


def test_duplicate_begin_reuses_root_manifest_without_reacquiring_peer_snapshot() -> (
    None
):
    root_snapshots = []
    peer_snapshots = []
    root_manifest = _manifest()
    peer_manifest = _manifest(
        worker_id="source/dp0-pp0-ep0-tp1",
        lease_id="lease-1",
    )
    peer_created = {
        "success": True,
        "message": "Success.",
        "session_state": "created",
        "manifest": peer_manifest,
        "manifest_revision_semantics": HF_REVISION_V1,
        "model_id": peer_manifest["model_id"],
        "revision": peer_manifest["revision"],
        "lease_id": peer_manifest["lease_id"],
        "generation": peer_manifest["generation"],
    }
    root_group = _RemoteTransferCollective(
        rank=0,
        gather_results=[[None, peer_created]],
    )
    root_manager = _manager(
        SimpleNamespace(
            get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
                root_snapshots.append(kwargs) or root_manifest
            ),
            release_weight_runtime_manifest=lambda lease_id: None,
        ),
        remote_weight_transfer_cpu_group=root_group,
    )
    request = _begin_request(
        transfer_id="transfer-1",
        model_id=root_manifest["model_id"],
        revision=root_manifest["revision"],
    )
    root_manager.begin_remote_instance_weight_transfer(request)
    peer_group = _RemoteTransferCollective(
        rank=1,
        gather_results=[[None, None]],
        broadcast_results=root_group.broadcasts,
    )
    peer_manager = _manager(
        SimpleNamespace(
            get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
                peer_snapshots.append(kwargs) or peer_manifest
            ),
            release_weight_runtime_manifest=lambda lease_id: None,
        ),
        remote_weight_transfer_cpu_group=peer_group,
    )
    peer_manager.begin_remote_instance_weight_transfer(request)

    peer_reused = {
        "success": True,
        "message": "Success.",
        "session_state": "reused",
        "manifest_revision_semantics": HF_REVISION_V1,
        "model_id": peer_manifest["model_id"],
        "revision": peer_manifest["revision"],
        "lease_id": peer_manifest["lease_id"],
        "generation": peer_manifest["generation"],
    }
    root_reuse_group = _RemoteTransferCollective(
        rank=0,
        gather_results=[[None, peer_reused]],
    )
    root_manager.remote_weight_transfer_cpu_group = root_reuse_group
    root_result = root_manager.begin_remote_instance_weight_transfer(request)
    peer_reuse_group = _RemoteTransferCollective(
        rank=1,
        gather_results=[[None, None]],
        broadcast_results=root_reuse_group.broadcasts,
    )
    peer_manager.remote_weight_transfer_cpu_group = peer_reuse_group
    peer_result = peer_manager.begin_remote_instance_weight_transfer(request)

    assert root_result.session_state == "reused"
    assert root_result.manifests == [root_manifest, peer_manifest]
    assert peer_result.session_state == "reused"
    assert peer_result.manifests is None
    assert len(root_snapshots) == len(peer_snapshots) == 1


def test_non_root_rolls_back_created_snapshot_after_root_rejects_begin() -> None:
    released = []
    peer_manifest = _manifest(
        worker_id="source/dp0-pp0-ep0-tp1",
        lease_id="lease-1",
    )
    initial_decision = {
        "success": False,
        "message": "source rank 0 failed",
        "session_state": "cleanup_pending",
        "manifest_revision_semantics": None,
    }
    final_decision = {
        **initial_decision,
        "message": "source rank 0 failed",
    }
    peer_group = _RemoteTransferCollective(
        rank=1,
        gather_results=[
            [None, None],
            [None, None],
        ],
        broadcast_results=[initial_decision, final_decision],
    )
    peer_manager = _manager(
        SimpleNamespace(
            get_remote_instance_weight_runtime_manifest=lambda **kwargs: peer_manifest,
            release_weight_runtime_manifest=released.append,
        ),
        remote_weight_transfer_cpu_group=peer_group,
    )

    result = peer_manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-1",
            model_id=peer_manifest["model_id"],
            revision=peer_manifest["revision"],
        )
    )

    assert result.success is False
    assert result.session_state == "cleanup_pending"
    assert released == ["lease-1"]
    assert peer_manager.remote_weight_transfer_leases == {}
    assert len(peer_group.gathered) == 2
    assert len(peer_group.broadcasts) == 2


@pytest.mark.parametrize(
    ("peer_lease_id", "peer_generation", "expected_message"),
    [
        ("different-lease", 1, "lease"),
        ("lease-1", 2, "generation"),
    ],
)
def test_begin_rejects_peer_bookkeeping_that_does_not_match_manifest(
    peer_lease_id,
    peer_generation,
    expected_message,
) -> None:
    released = []
    root_manifest = _manifest()
    peer_manifest = _manifest(
        worker_id="source/dp0-pp0-ep0-tp1",
        lease_id="lease-1",
    )
    peer_created = {
        "success": True,
        "message": "Success.",
        "session_state": "created",
        "manifest": peer_manifest,
        "manifest_revision_semantics": HF_REVISION_V1,
        "model_id": peer_manifest["model_id"],
        "revision": peer_manifest["revision"],
        "lease_id": peer_lease_id,
        "generation": peer_generation,
    }
    root_group = _RemoteTransferCollective(
        rank=0,
        gather_results=[
            [None, peer_created],
            [None, {"success": True, "message": "Success."}],
        ],
    )
    root_manager = _manager(
        SimpleNamespace(
            get_remote_instance_weight_runtime_manifest=lambda **kwargs: root_manifest,
            release_weight_runtime_manifest=released.append,
        ),
        remote_weight_transfer_cpu_group=root_group,
    )

    result = root_manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-1",
            model_id=root_manifest["model_id"],
            revision=root_manifest["revision"],
        )
    )

    assert result.success is False
    assert expected_message in result.message.lower()
    assert released == ["lease-0"]
    assert root_manager.remote_weight_transfer_leases == {}


def _run_remote_transfer_root_gather(rank, world_size, init_method) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    group = None
    try:
        group = torch.distributed.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
        )
        snapshots = []
        released = []
        manifest = _manifest(
            worker_id=f"source/dp0-pp0-ep0-tp{rank}",
            lease_id=f"lease-{rank}",
        )

        def snapshot(**kwargs):
            snapshots.append(kwargs)
            if kwargs["transfer_id"] == "transfer-failed" and rank == 1:
                raise RuntimeError("rank 1 snapshot failed")
            return manifest

        manager = _manager(
            SimpleNamespace(
                get_remote_instance_weight_runtime_manifest=snapshot,
                release_weight_runtime_manifest=released.append,
            ),
            remote_weight_transfer_cpu_group=group,
        )
        request = _begin_request(
            transfer_id="transfer-1",
            model_id=manifest["model_id"],
            revision=manifest["revision"],
        )

        created = manager.begin_remote_instance_weight_transfer(request)
        reused = manager.begin_remote_instance_weight_transfer(request)

        assert created.success is True, (
            rank,
            created.session_state,
            created.message,
        )
        assert reused.success is True, (
            rank,
            reused.session_state,
            reused.message,
        )
        assert len(snapshots) == 1
        cached = manager.remote_weight_transfer_sessions["transfer-1"][1]
        if rank == 0:
            assert created.manifests is not None
            assert len(created.manifests) == world_size
            assert reused.manifests == created.manifests
            assert cached is created
        else:
            assert created.manifests is None
            assert reused.manifests is None
            assert cached is None

        result = manager.release_remote_instance_weight_transfer(
            ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
        )
        assert result.success is True
        assert released == [f"lease-{rank}"]
        assert manager.remote_weight_transfer_sessions == {}

        failed = manager.begin_remote_instance_weight_transfer(
            _begin_request(
                transfer_id="transfer-failed",
                model_id=manifest["model_id"],
                revision=manifest["revision"],
            )
        )
        assert failed.success is False
        assert failed.session_state == "cleanup_pending"
        assert "rank 1 snapshot failed" in failed.message
        assert manager.remote_weight_transfer_leases == {}
        assert released == (
            [f"lease-{rank}", f"lease-{rank}"] if rank == 0 else [f"lease-{rank}"]
        )
    finally:
        if group is not None:
            torch.distributed.destroy_process_group(group)
        torch.distributed.destroy_process_group()


def test_remote_transfer_root_gather_runs_on_two_process_gloo(tmp_path) -> None:
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed Gloo backend is unavailable")
    torch.multiprocessing.spawn(
        _run_remote_transfer_root_gather,
        args=(2, f"file://{tmp_path / 'remote-transfer-gather'}"),
        nprocs=2,
        join=True,
    )


def _run_remote_transfer_asymmetric_collective_failure(
    rank,
    world_size,
    init_method,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    process_group = None
    try:
        process_group = torch.distributed.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
            timeout=timedelta(seconds=3),
        )
        collective = _AsymmetricBroadcastFailureCollective(
            process_group,
            rank=rank,
            world_size=world_size,
        )
        snapshots = []
        released = []
        manifest = _manifest(
            worker_id=f"source/dp0-pp0-ep0-tp{rank}",
            lease_id=f"lease-{rank}",
        )
        manager = _manager(
            SimpleNamespace(
                get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
                    snapshots.append(kwargs) or manifest
                ),
                release_weight_runtime_manifest=released.append,
            ),
            remote_weight_transfer_cpu_group=collective,
        )
        manager.remote_weight_transfer_control_cpu_group = process_group
        request = _begin_request(
            transfer_id="transfer-1",
            model_id=manifest["model_id"],
            revision=manifest["revision"],
        )

        result = manager.begin_remote_instance_weight_transfer(request)
        retry = manager.begin_remote_instance_weight_transfer(
            _begin_request(
                transfer_id="transfer-2",
                model_id=manifest["model_id"],
                revision=manifest["revision"],
            )
        )

        assert result.success is False
        assert manager.remote_weight_transfer_snapshot_poisoned is not None
        assert manager.remote_weight_transfer_leases == {}
        assert released == [f"lease-{rank}"]
        assert retry.success is False
        assert "scheduler restart is required" in retry.message
        assert len(snapshots) == 1
    finally:
        if process_group is not None:
            torch.distributed.destroy_process_group(process_group)
        torch.distributed.destroy_process_group()


def test_asymmetric_collective_failure_poisons_snapshot_lane(tmp_path) -> None:
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed Gloo backend is unavailable")
    torch.multiprocessing.spawn(
        _run_remote_transfer_asymmetric_collective_failure,
        args=(2, f"file://{tmp_path / 'remote-transfer-failure'}"),
        nprocs=2,
        join=True,
    )


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
    request = _begin_request(
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
    request = _begin_request(
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
    request = _begin_request(
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
    first = _begin_request(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
    )
    mismatched = _begin_request(
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
    request = _begin_request(
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
                "poisoned": False,
                "manifest": _manifest(worker_id="source/dp1-pp0-ep0-tp0"),
                "manifest_revision_semantics": HF_REVISION_V1,
                "model_id": "Qwen/Qwen3.5-0.8B",
                "revision": "main",
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
        _begin_request(
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


def test_begin_rejects_mixed_model_rank_revision_semantics(monkeypatch) -> None:
    released = []
    placement = _placement()
    binding = _binding()
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest_parts=lambda **_kwargs: (
            SimpleNamespace(placement=placement, binding=binding)
        ),
        release_weight_runtime_manifest=released.append,
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 2)

    def all_gather_object(outputs, value, group):
        peer = dict(value)
        peer["manifest_revision_semantics"] = HF_REVISION_V1
        outputs[:] = [value, peer]

    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        all_gather_object,
    )

    result = manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            manifest_format="placement_binding_v1",
            manifest_revision_semantics=ARTIFACT_WEIGHT_VERSION_V1,
            request_id="begin-attempt-1",
        )
    )

    assert result.success is False
    assert "incompatible weight identity" in result.message
    assert released == ["lease-0"]


def test_begin_uses_dedicated_remote_transfer_cpu_group() -> None:
    manifest = _manifest()
    remote_group = _RemoteTransferCollective(
        rank=0,
        gather_results=[[None]],
    )
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: manifest,
        release_weight_runtime_manifest=lambda lease_id: None,
    )
    manager = _manager(runner, remote_weight_transfer_cpu_group=remote_group)

    result = manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is True
    assert len(remote_group.gathered) == 1
    assert len(remote_group.broadcasts) == 1


def test_control_lifecycle_uses_group_coordinator_object_collective() -> None:
    renewed = []
    released = []
    control_group = _RemoteTransferCollective(
        rank=0,
        gather_results=[[None]],
    )
    runner = SimpleNamespace(
        renew_weight_runtime_manifest=lambda lease_id, lease_timeout_sec: (
            renewed.append((lease_id, lease_timeout_sec))
        ),
        release_weight_runtime_manifest=released.append,
    )
    manager = _manager(runner)
    manager.remote_weight_transfer_control_cpu_group = control_group
    manager._record_remote_weight_transfer_lease("transfer-1", "lease-0", 60)

    renewal = manager.renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            lease_timeout_sec=60,
        )
    )
    release = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )

    assert renewal.success is True
    assert release.success is True
    assert renewed == [("lease-0", 60)]
    assert released == ["lease-0"]
    assert len(control_group.all_gathered) == 2


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
        outputs[:] = (
            [value, value]
            if "poisoned" in value
            else [value, {"success": False, "message": "rank 1 failed"}]
        )

    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)
    result = manager.begin_remote_instance_weight_transfer(
        _begin_request(
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
    snapshots = []
    manifest = _manifest()
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **kwargs: (
            snapshots.append(kwargs) or manifest
        ),
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)

    def all_gather_object(outputs, value, group):
        if "poisoned" in value:
            outputs[0] = value
            return
        raise RuntimeError("collective failed")

    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        all_gather_object,
    )

    result = manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert result.success is False
    assert "collective failed" in result.message
    assert released == ["lease-0"]
    assert manager.remote_weight_transfer_snapshot_poisoned is not None

    retry = manager.begin_remote_instance_weight_transfer(
        _begin_request(
            transfer_id="transfer-2",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
        )
    )

    assert retry.success is False
    assert "scheduler restart is required" in retry.message
    assert len(snapshots) == 1


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
        outputs[:] = (
            [value, value]
            if "poisoned" in value
            else [
                value,
                {
                    "success": False,
                    "message": "rank 1 failed",
                    "session_state": "failed",
                },
            ]
        )

    monkeypatch.setattr("torch.distributed.all_gather_object", all_gather_object)
    request = _begin_request(
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
    request = _begin_request(
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
    request = _begin_request(
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
        if len(collective_calls) == 2:
            outputs[:] = [
                value,
                {
                    "success": False,
                    "message": "remote rank mutation failed",
                    "mutated_any": True,
                    "generations": [2],
                },
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
    assert len(collective_calls) == 4
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
        "collective",
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
        if len(collective_calls) == 2:
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
    assert len(collective_calls) == 4
    assert coordinator.pending_revision_generation() == 3

    with pytest.raises(WeightManifestError, match="last weight update failed"):
        coordinator.commit_revision(expected_generation=3)
    with pytest.raises(WeightManifestError, match="last weight update failed"):
        coordinator.acquire_snapshot()


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
    assert len(collective_calls) == 4
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
    committed_generations = []

    class RecordingCoordinator(WeightSnapshotCoordinator):
        def commit_revision(self, *, expected_generation=None):
            committed_generations.append(expected_generation)
            return super().commit_revision(expected_generation=expected_generation)

    coordinator = RecordingCoordinator()
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
    assert committed_generations == [3]
    assert events == [
        ("pause", GPU_MEMORY_TYPE_WEIGHTS),
        ("resume", GPU_MEMORY_TYPE_WEIGHTS),
        ("restore", {"buffers": []}),
    ]
    coordinator.release_snapshot(lease_id)


def test_main_update_then_draft_failure_flushes_and_poisons(monkeypatch) -> None:
    main_coordinator = WeightSnapshotCoordinator()
    draft_coordinator = WeightSnapshotCoordinator()
    main_runner = SimpleNamespace(weight_snapshot_coordinator=main_coordinator)
    draft_runner = SimpleNamespace(weight_snapshot_coordinator=draft_coordinator)
    flushes = []
    manager = _manager(main_runner)
    manager.draft_worker = SimpleNamespace(model_runner=draft_runner)
    manager.flush_cache = lambda **kwargs: flushes.append(kwargs) or True

    def update_main(request):
        del request
        token = main_coordinator.begin_update()
        main_coordinator.finish_update(token, success=True)
        return True, "main updated"

    manager.tp_worker.update_weights_from_ipc = update_main
    manager.draft_worker.update_weights_from_ipc = lambda request: (
        False,
        "draft preflight rejected",
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )

    result = manager.update_weights_from_ipc(
        UpdateWeightsFromIPCReqInput(zmq_handles={}, flush_cache=True)
    )

    assert result.success is False
    assert result.fail_closed is True
    assert "draft preflight rejected" in result.message
    assert flushes == [{"empty_cache": False}]
    with pytest.raises(WeightManifestError, match="last weight update failed"):
        main_coordinator.acquire_snapshot()
    with pytest.raises(WeightManifestError, match="last weight update failed"):
        draft_coordinator.acquire_snapshot()


def test_pre_mutation_weight_update_failure_keeps_serving_enabled(monkeypatch) -> None:
    coordinator = WeightSnapshotCoordinator()
    runner = SimpleNamespace(weight_snapshot_coordinator=coordinator)
    manager = _manager(runner)
    manager.tp_worker.update_weights_from_ipc = lambda _request: (
        False,
        "preflight rejected",
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )

    result = manager.update_weights_from_ipc(
        UpdateWeightsFromIPCReqInput(zmq_handles={}, flush_cache=True)
    )

    assert result.success is False
    assert result.fail_closed is False
    lease_id, _generation = coordinator.acquire_snapshot()
    coordinator.release_snapshot(lease_id)


def test_pre_mutation_failure_without_manifest_does_not_fail_closed(
    monkeypatch,
) -> None:
    updater = SimpleNamespace(_sglang_last_weight_mutation_started=False)
    runner = SimpleNamespace(
        weight_snapshot_coordinator=None,
        weight_updater=updater,
    )
    manager = _manager(runner)

    def reject(_request):
        updater._sglang_last_weight_mutation_started = False
        return False, "preflight rejected"

    manager.tp_worker.update_weights_from_ipc = reject
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )

    result = manager.update_weights_from_ipc(
        UpdateWeightsFromIPCReqInput(zmq_handles={}, flush_cache=True)
    )

    assert result.success is False
    assert result.fail_closed is False


def test_control_plane_only_clears_fail_closed_after_full_restore() -> None:
    manager = SimpleNamespace(weight_update_fail_closed=False)
    partial_failure = SimpleNamespace(success=False, fail_closed=True)
    incremental_success = SimpleNamespace(success=True, fail_closed=False)
    full_restore_success = SimpleNamespace(success=True, fail_closed=False)

    tokenizer_control_mixin_module._record_weight_update_safety(
        manager,
        (partial_failure,),
        full_restore=False,
    )
    assert manager.weight_update_fail_closed is True

    tokenizer_control_mixin_module._record_weight_update_safety(
        manager,
        (incremental_success,),
        full_restore=False,
    )
    assert manager.weight_update_fail_closed is True

    tokenizer_control_mixin_module._record_weight_update_safety(
        manager,
        (full_restore_success,),
        full_restore=True,
    )
    assert manager.weight_update_fail_closed is False


def test_weight_memory_resume_rejects_rank_generation_divergence(
    monkeypatch,
) -> None:
    coordinator = WeightSnapshotCoordinator()
    runner = SimpleNamespace(
        model=object(),
        weight_snapshot_coordinator=coordinator,
    )
    manager = _manager(runner)
    manager.offload_tags.add(GPU_MEMORY_TYPE_WEIGHTS)
    manager.stashed_model_static_state = {"buffers": []}
    manager.memory_saver_adapter = SimpleNamespace(resume=lambda tag: None)
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.weight_updater._import_static_state",
        lambda model, state: None,
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda group: None)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    calls = 0

    def all_gather_object(outputs, value, group):
        nonlocal calls
        del group
        calls += 1
        if calls == 1:
            outputs[:] = [value, value]
        elif calls == 2:
            remote = dict(value)
            remote["generations"] = [value["generations"][0] + 1]
            outputs[:] = [value, remote]
        else:
            outputs[:] = [value, value]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    with pytest.raises(WeightManifestError, match="generations differ"):
        manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        )

    with pytest.raises(WeightManifestError, match="last weight update failed"):
        coordinator.acquire_snapshot()


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


def test_renew_echoes_request_id_and_returns_granted_deadline(monkeypatch) -> None:
    monotonic_now = [100.0]
    unix_now = [1000.0]
    renewed = []
    runner = SimpleNamespace(
        renew_weight_runtime_manifest=lambda lease_id, lease_timeout_sec: (
            renewed.append((lease_id, lease_timeout_sec))
        ),
    )
    manager = _manager(runner)
    monkeypatch.setattr(
        weight_updater_module.time,
        "monotonic",
        lambda: monotonic_now[0],
    )
    monkeypatch.setattr(weight_updater_module.time, "time", lambda: unix_now[0])
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    manager._record_remote_weight_transfer_lease("transfer-1", "lease-0", 30)

    result = manager.renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            lease_timeout_sec=60,
            request_id="renew-request-1",
        )
    )

    assert renewed == [("lease-0", 60)]
    assert result.success is True
    assert result.request_id == "renew-request-1"
    assert result.external_dp_rank == 0
    assert result.deadline_unix_sec == 1060.0


def test_stale_lease_identity_cannot_renew_or_release_reused_transfer_id(
    monkeypatch,
) -> None:
    renewed = []
    released = []
    runner = SimpleNamespace(
        renew_weight_runtime_manifest=lambda lease_id, lease_timeout_sec: (
            renewed.append((lease_id, lease_timeout_sec))
        ),
        release_weight_runtime_manifest=released.append,
    )
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    manager._record_remote_weight_transfer_lease(
        "transfer-1",
        "lease-b",
        60,
        generation=2,
        lease_fence="fence-b",
    )

    renewal = manager.renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            lease_timeout_sec=60,
            lease_fence="fence-a",
            generation=1,
            deadline_unix_sec=time.time() + 30,
        )
    )
    release = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            lease_fence="fence-a",
            generation=1,
            deadline_unix_sec=time.time() + 30,
        )
    )

    assert renewal.success is False
    assert release.success is False
    assert renewed == []
    assert released == []
    assert manager.remote_weight_transfer_leases == {"transfer-1": "lease-b"}
    assert manager.remote_weight_transfer_generations == {"transfer-1": 2}


def test_queued_renew_rejects_expired_original_deadline_before_mutation(
    monkeypatch,
) -> None:
    renewed = []
    manager = _manager(
        SimpleNamespace(
            renew_weight_runtime_manifest=lambda lease_id, lease_timeout_sec: (
                renewed.append((lease_id, lease_timeout_sec))
            )
        )
    )
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    manager._record_remote_weight_transfer_lease(
        "transfer-1",
        "lease-1",
        60,
        generation=1,
        lease_fence="fence-1",
    )

    result = manager.renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            lease_timeout_sec=60,
            lease_fence="fence-1",
            generation=1,
            deadline_unix_sec=time.time() - 1,
        )
    )

    assert result.success is False
    assert "deadline" in result.message.lower()
    assert renewed == []


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
    request = _begin_request(
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
    begin_requests = []

    async def begin_communicator(request, *, deadline_unix_sec):
        assert deadline_unix_sec > time.time()
        events.append(("begin", request.transfer_id))
        begin_requests.append(request)
        return begin_results

    async def release_communicator(request, *, deadline_unix_sec):
        assert deadline_unix_sec > time.time()
        return await release(request)

    async def pause(request):
        events.append(("pause", request.mode))

    async def resume(request):
        events.append(("continue", request.torch_empty_cache))

    return SimpleNamespace(
        runtime_weight_revision="default",
        weight_update_fail_closed=False,
        server_args=SimpleNamespace(
            enable_weight_runtime_manifest=True,
            model_path="Qwen/Qwen3.5-0.8B",
            revision="main",
            weight_version="default",
        ),
        auto_create_handle_loop=lambda: None,
        is_pause=False,
        is_pause_cond=asyncio.Condition(),
        model_update_lock=RWLock(),
        pause_generation=pause,
        continue_generation=resume,
        begin_remote_instance_weight_transfer_communicator=begin_communicator,
        release_remote_instance_weight_transfer_communicator=release_communicator,
        _remote_weight_transfer_events=events,
        _remote_weight_transfer_begin_requests=begin_requests,
    )


def _pause_owner_tokenizer_manager(begin, release):
    events = []

    async def begin_communicator(request, *, deadline_unix_sec):
        assert deadline_unix_sec > time.time()
        return await begin(request)

    async def release_communicator(request, *, deadline_unix_sec):
        assert deadline_unix_sec > time.time()
        return await release(request)

    async def dispatch(request):
        if isinstance(request, PauseGenerationReqInput):
            events.append(("pause", request.mode))
        elif isinstance(request, ContinueGenerationReqInput):
            events.append(("continue", request.torch_empty_cache))

    manager = object.__new__(TokenizerManager)
    manager.runtime_weight_revision = "default"
    manager.weight_update_fail_closed = False
    manager.server_args = SimpleNamespace(
        enable_weight_runtime_manifest=True,
        model_path="Qwen/Qwen3.5-0.8B",
        revision="main",
        weight_version="default",
    )
    manager.auto_create_handle_loop = lambda: None
    manager.is_pause = False
    manager.is_pause_cond = asyncio.Condition()
    manager.model_update_lock = RWLock()
    manager._async_dispatch_to_scheduler = dispatch
    manager.begin_remote_instance_weight_transfer_communicator = begin_communicator
    manager.release_remote_instance_weight_transfer_communicator = release_communicator
    manager._remote_weight_transfer_events = events
    return manager


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


def test_tokenizer_release_serializes_with_overlapping_begin_pause() -> None:
    async def scenario():
        release_started = asyncio.Event()
        finish_release = asyncio.Event()
        begin_started = asyncio.Event()
        finish_begin = asyncio.Event()

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

        async def release(_request):
            release_started.set()
            await finish_release.wait()
            return [SimpleNamespace(success=True, message="Success.")]

        manager = _pause_owner_tokenizer_manager(begin, release)
        release_task = asyncio.create_task(
            TokenizerControlMixin.release_remote_instance_weight_transfer(
                manager, "transfer-1"
            )
        )
        await release_started.wait()
        begin_task = asyncio.create_task(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                transfer_id="transfer-2",
            )
        )

        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(begin_started.wait(), timeout=0.05)

            finish_release.set()
            assert (await release_task)[0] is True
            await asyncio.wait_for(begin_started.wait(), timeout=1)
            continue_count = sum(
                event[0] == "continue"
                for event in manager._remote_weight_transfer_events
            )
            await asyncio.sleep(0)
            assert (
                sum(
                    event[0] == "continue"
                    for event in manager._remote_weight_transfer_events
                )
                == continue_count
            )

            finish_begin.set()
            assert (await begin_task)["transfer_id"] == "transfer-2"
        finally:
            finish_release.set()
            finish_begin.set()
            await asyncio.gather(
                release_task,
                begin_task,
                return_exceptions=True,
            )

    asyncio.run(scenario())


def test_tokenizer_admin_pause_survives_overlapping_internal_pause() -> None:
    async def scenario():
        begin_started = asyncio.Event()
        finish_begin = asyncio.Event()

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

        async def release(_request):
            return [SimpleNamespace(success=True, message="Success.")]

        manager = _pause_owner_tokenizer_manager(begin, release)
        begin_task = asyncio.create_task(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                transfer_id="transfer-1",
            )
        )
        await begin_started.wait()

        await manager.pause_generation(PauseGenerationReqInput(mode="retract"))
        finish_begin.set()
        await begin_task

        assert manager.is_pause is True
        assert [event[0] for event in manager._remote_weight_transfer_events] == [
            "pause",
            "pause",
        ]

        await manager.continue_generation(ContinueGenerationReqInput())
        assert manager.is_pause is False
        assert manager._remote_weight_transfer_events[-1][0] == "continue"

    asyncio.run(scenario())


def test_multi_tokenizer_router_serializes_remote_pause_owners(monkeypatch) -> None:
    async def scenario():
        scheduled = []
        outputs = []

        async def send_to_scheduler(_socket, request):
            scheduled.append(request)

        class RecordingSocketMapping:
            def send_output(self, ipc_name, output):
                if ipc_name == "worker-a":
                    outputs.append(output)

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        router = object.__new__(MultiTokenizerRouter)
        router.send_to_scheduler = object()
        router.all_worker_ipcs = {"worker-a", "worker-b"}
        _register_live_tokenizer_workers(router)
        router.socket_mapping = RecordingSocketMapping()
        router.pause_owners = set()
        router.active_remote_pause_owner = None
        router.pending_remote_pause_requests = deque()
        router._pause_transitions = {}
        router._pause_poisoned_owners = set()
        first_owner = "remote-weight-transfer:first"
        second_owner = "remote-weight-transfer:second"

        def ack_all(request):
            identity = multi_tokenizer_mixin_module._decode_pause_transition(
                request.rid
            )
            assert identity is not None
            for worker_ipc in ("worker-a", "worker-b"):
                router._handle_pause_continue_ack(
                    multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                        rid=request.rid,
                        is_pause=identity.expected_state,
                        http_worker_ipc=worker_ipc,
                    )
                )
            transition = router._pause_transitions.get(identity.transition_id)
            if transition is None or not transition.confirmation_sent:
                return
            applied_rid = multi_tokenizer_mixin_module._encode_pause_transition_applied(
                identity
            )
            for worker_ipc in ("worker-a", "worker-b"):
                router._handle_pause_continue_ack(
                    multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                        rid=applied_rid,
                        is_pause=identity.expected_state,
                        http_worker_ipc=worker_ipc,
                    )
                )
            transition = router._pause_transitions.get(identity.transition_id)
            if transition is None or not transition.committed:
                return
            committed_rid = (
                multi_tokenizer_mixin_module._encode_pause_transition_committed_ack(
                    identity
                )
            )
            for worker_ipc in ("worker-a", "worker-b"):
                router._handle_pause_continue_ack(
                    multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                        rid=committed_rid,
                        is_pause=identity.expected_state,
                        http_worker_ipc=worker_ipc,
                    )
                )
            _finalize_router_transition(router, identity)

        first_pause = PauseGenerationReqInput(
            mode="in_place",
            rid=first_owner,
            http_worker_ipc="worker-a",
        )
        await router._handle_pause_continue_request(first_pause)
        ack_all(first_pause)

        second_pause = PauseGenerationReqInput(
            mode="in_place",
            rid=second_owner,
            http_worker_ipc="worker-b",
        )
        await router._handle_pause_continue_request(second_pause)
        ack_all(second_pause)

        admin_pause = PauseGenerationReqInput(
            mode="retract",
            rid="admin",
            http_worker_ipc="worker-b",
        )
        await router._handle_pause_continue_request(admin_pause)
        ack_all(admin_pause)

        first_continue = ContinueGenerationReqInput(
            rid=first_owner,
            http_worker_ipc="worker-a",
        )
        await router._handle_pause_continue_request(first_continue)
        ack_all(first_continue)
        await asyncio.sleep(0)
        ack_all(first_continue)
        await asyncio.sleep(0)
        ack_all(second_pause)

        second_continue = ContinueGenerationReqInput(
            rid=second_owner,
            http_worker_ipc="worker-b",
        )
        await router._handle_pause_continue_request(second_continue)
        ack_all(second_continue)
        await asyncio.sleep(0)
        ack_all(second_continue)

        admin_continue = ContinueGenerationReqInput(
            rid="admin",
            http_worker_ipc="worker-b",
        )
        await router._handle_pause_continue_request(admin_continue)
        ack_all(admin_continue)
        await asyncio.sleep(0)
        ack_all(admin_continue)
        await asyncio.sleep(0)

        assert [
            (type(request), getattr(request, "mode", None)) for request in scheduled
        ] == [
            (PauseGenerationReqInput, "in_place"),
            (PauseGenerationReqInput, "retract"),
            (ContinueGenerationReqInput, None),
        ]
        confirmations = [
            multi_tokenizer_mixin_module._decode_pause_transition(output.rid)
            for output in outputs
            if (
                output.http_worker_ipc
                == multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED
            )
        ]
        assert [(identity.owner, identity.action) for identity in confirmations] == [
            (first_owner, "pause"),
            ("admin", "pause"),
            (first_owner, "continue"),
            (second_owner, "pause"),
            (second_owner, "continue"),
            ("admin", "continue"),
        ]
        assert router.pause_owners == set()
        assert router.active_remote_pause_owner is None

    asyncio.run(scenario())


def test_multi_tokenizer_worker_matches_pause_ack_owner() -> None:
    async def scenario():
        worker = object.__new__(TokenizerWorker)
        worker.is_pause = False
        worker.is_pause_cond = asyncio.Condition()
        worker._generation_pause_owners = set()
        worker._generation_pause_resume_pending = set()
        acks = []

        async def dispatch_ack(ack):
            acks.append(ack)

        worker._async_dispatch_to_scheduler = dispatch_ack
        identity = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner="remote-weight-transfer:second",
            action="pause",
            expected_state=True,
        )
        pending = asyncio.get_running_loop().create_future()
        worker._pause_continue_futures = {
            identity.transition_id: (identity, pending),
        }

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=True,
            ),
        )
        assert worker.is_pause is True
        assert pending.done() is False
        assert len(acks) == 1

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=True,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
            ),
        )
        assert worker.is_pause is True
        assert pending.done() is False
        assert len(acks) == 2

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=True,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED,
            ),
        )
        assert worker.is_pause is True
        assert pending.done() is False
        assert (
            multi_tokenizer_mixin_module._decode_pause_transition_committed_ack(
                acks[-1].rid
            )
            == identity
        )

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=True,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_FINALIZED,
            ),
        )
        assert pending.result() is True

    asyncio.run(scenario())


def test_generation_pause_release_failure_keeps_last_owner_retryable() -> None:
    async def scenario():
        owner = "remote-weight-transfer:retry"
        attempts = 0

        async def continue_generation(_request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("continue dispatch failed")

        manager = SimpleNamespace(
            is_pause=True,
            is_pause_cond=asyncio.Condition(),
            _generation_pause_owners={owner},
            _generation_pause_resume_pending=set(),
            _continue_generation_impl=continue_generation,
        )

        with pytest.raises(RuntimeError, match="continue dispatch failed"):
            await TokenizerControlMixin._release_generation_pause(
                manager,
                owner,
                ContinueGenerationReqInput(torch_empty_cache=False),
            )

        assert manager.is_pause is True
        assert manager._generation_pause_owners == {owner}
        assert manager._generation_pause_resume_pending == set()
        assert manager._generation_continue_unconfirmed == {owner}
        assert not manager._generation_pause_transition_lock.locked()

        await TokenizerControlMixin._release_generation_pause(
            manager,
            owner,
            ContinueGenerationReqInput(torch_empty_cache=False),
        )

        assert attempts == 2
        assert manager.is_pause is False
        assert manager._generation_pause_owners == set()
        assert manager._generation_pause_resume_pending == set()
        assert manager._generation_continue_unconfirmed == set()

    asyncio.run(scenario())


def test_pause_before_dispatch_failure_does_not_leave_a_false_pause_owner() -> None:
    async def scenario():
        owner = "remote-weight-transfer:not-dispatched"
        pause_attempts = 0
        continued = []
        manager = SimpleNamespace(
            is_pause=False,
            is_pause_cond=asyncio.Condition(),
            _generation_pause_owners=set(),
        )

        async def pause_generation(_request):
            nonlocal pause_attempts
            pause_attempts += 1
            if pause_attempts == 1:
                raise FanOutDeadlineExpiredBeforeDispatch("deadline expired")

        async def continue_generation(request):
            continued.append(request.rid)

        manager.pause_generation = pause_generation
        manager.continue_generation = continue_generation

        with pytest.raises(FanOutDeadlineExpiredBeforeDispatch):
            await TokenizerControlMixin._acquire_generation_pause(
                manager,
                owner,
                PauseGenerationReqInput(mode="in_place"),
            )

        assert manager.is_pause is False
        assert manager._generation_pause_owners == set()

        async with TokenizerControlMixin._remote_instance_weight_transfer_pause(
            manager
        ):
            assert pause_attempts == 2
            assert manager.is_pause is True

        assert len(continued) == 1
        assert manager.is_pause is False
        assert manager._generation_pause_owners == set()

    asyncio.run(scenario())


def test_pause_completion_unknown_is_reestablished_before_next_transfer() -> None:
    async def scenario():
        pause_attempts = 0
        continued = []
        owner = "remote-weight-transfer:pause-unknown"

        async def pause_generation(_request):
            nonlocal pause_attempts
            pause_attempts += 1
            if pause_attempts == 1:
                raise FanOutCompletionUnknownError(
                    "pause completion unknown",
                    dispatch_started=True,
                    dispatch_completed=False,
                    partial_results=(),
                    expected_count=1,
                )

        async def continue_generation(request):
            continued.append(request.rid)

        manager = SimpleNamespace(
            is_pause=False,
            is_pause_cond=asyncio.Condition(),
            _generation_pause_owners=set(),
            pause_generation=pause_generation,
            continue_generation=continue_generation,
        )

        with pytest.raises(FanOutCompletionUnknownError):
            await TokenizerControlMixin._acquire_generation_pause(
                manager,
                owner,
                PauseGenerationReqInput(mode="in_place"),
            )

        assert manager._generation_pause_unconfirmed == {owner}
        async with TokenizerControlMixin._remote_instance_weight_transfer_pause(
            manager
        ):
            assert pause_attempts == 2
            assert manager.is_pause is True

        assert continued == [owner]
        assert manager.is_pause is False
        assert manager._generation_pause_owners == set()

    asyncio.run(scenario())


@pytest.mark.parametrize("dispatch_completed", [False, True])
def test_continue_completion_unknown_repauses_before_next_transfer(
    dispatch_completed,
) -> None:
    async def scenario():
        scheduler_paused = False
        pause_attempts = 0
        continue_attempts = 0

        async def pause_generation(_request):
            nonlocal scheduler_paused, pause_attempts
            scheduler_paused = True
            pause_attempts += 1

        async def continue_generation(_request):
            nonlocal scheduler_paused, continue_attempts
            continue_attempts += 1
            if continue_attempts == 1:
                if dispatch_completed:
                    scheduler_paused = False
                raise FanOutCompletionUnknownError(
                    "continue acknowledgement lost",
                    dispatch_started=True,
                    dispatch_completed=dispatch_completed,
                    partial_results=(),
                    expected_count=1,
                )
            scheduler_paused = False

        manager = SimpleNamespace(
            is_pause=False,
            is_pause_cond=asyncio.Condition(),
            _generation_pause_owners=set(),
            pause_generation=pause_generation,
            continue_generation=continue_generation,
        )

        with pytest.raises(FanOutCompletionUnknownError):
            async with TokenizerControlMixin._remote_instance_weight_transfer_pause(
                manager
            ):
                assert scheduler_paused is True

        owner = next(iter(manager._generation_pause_owners))
        assert manager._generation_continue_unconfirmed == {owner}

        async with TokenizerControlMixin._remote_instance_weight_transfer_pause(
            manager
        ):
            assert pause_attempts == 2
            assert scheduler_paused is True

        assert continue_attempts == 2
        assert scheduler_paused is False
        assert manager.is_pause is False
        assert manager._generation_pause_owners == set()

    asyncio.run(scenario())


def test_remote_transfer_reestablishes_pause_for_legacy_pending_owner() -> None:
    async def scenario():
        owners = {
            "remote-weight-transfer:first",
            "remote-weight-transfer:second",
        }
        resumed = []
        paused = []

        async def continue_generation(request):
            resumed.append(request.rid)

        async def pause_generation(request):
            paused.append(request)

        manager = SimpleNamespace(
            is_pause=True,
            is_pause_cond=asyncio.Condition(),
            _generation_pause_owners=set(owners),
            _generation_pause_resume_pending=set(owners),
            continue_generation=continue_generation,
            pause_generation=pause_generation,
        )

        async with TokenizerControlMixin._remote_instance_weight_transfer_pause(
            manager
        ):
            assert manager.is_pause is True
            assert manager._generation_pause_owners == {"remote-weight-transfer:first"}

        assert [request.rid for request in paused] == ["remote-weight-transfer:first"]
        assert resumed == ["remote-weight-transfer:first"]
        assert manager.is_pause is False
        assert manager._generation_pause_owners == set()
        assert manager._generation_pause_resume_pending == set()

    asyncio.run(scenario())


def test_generation_pause_reestablishes_pause_with_poisoned_owner() -> None:
    async def scenario():
        owner = "remote-weight-transfer:new"
        requests = []

        async def pause_generation(request):
            requests.append(request)

        manager = SimpleNamespace(
            is_pause=True,
            is_pause_cond=asyncio.Condition(),
            _generation_pause_owners={
                tokenizer_control_mixin_module._ADMIN_PAUSE_OWNER
            },
            _generation_pause_resume_pending={
                tokenizer_control_mixin_module._ADMIN_PAUSE_OWNER
            },
            _pause_generation_impl=pause_generation,
        )
        request = PauseGenerationReqInput(mode="in_place")

        await TokenizerControlMixin._acquire_generation_pause(
            manager,
            owner,
            request,
        )

        assert requests == [request]
        assert manager.is_pause is True
        assert manager._generation_pause_owners == {
            tokenizer_control_mixin_module._ADMIN_PAUSE_OWNER,
            owner,
        }

    asyncio.run(scenario())


def test_multi_tokenizer_ack_timeout_releases_lock_and_poison_pause(
    monkeypatch,
) -> None:
    async def scenario():
        owner = "remote-weight-transfer:timeout"
        worker = object.__new__(TokenizerWorker)
        worker.is_pause = True
        worker.is_pause_cond = asyncio.Condition()
        worker._generation_pause_owners = {owner}
        worker._generation_pause_resume_pending = set()
        worker._generation_pause_transition_lock = asyncio.Lock()
        worker._pause_continue_futures = {}
        dispatched = []

        async def dispatch(request):
            dispatched.append(request)

        worker._async_dispatch_to_scheduler = dispatch

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "_PAUSE_CONTINUE_ACK_TIMEOUT_SEC",
            0.01,
        )

        with pytest.raises(TimeoutError, match="pause transition acknowledgement"):
            await TokenizerWorker._release_generation_pause(
                worker,
                owner,
                ContinueGenerationReqInput(torch_empty_cache=False),
            )

        assert not worker._generation_pause_transition_lock.locked()
        assert worker.is_pause is True
        assert worker._generation_pause_owners == {owner}
        assert worker._generation_pause_resume_pending == {owner}
        assert worker._pause_continue_futures == {}

        identity = multi_tokenizer_mixin_module._decode_pause_transition(
            dispatched[0].rid
        )
        assert identity is not None
        worker._prepared_pause_transitions = {identity.transition_id: identity}
        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=dispatched[0].rid,
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
            ),
        )
        assert worker.is_pause is True
        assert worker._generation_pause_owners == {owner}

    asyncio.run(scenario())


def test_multi_tokenizer_router_freezes_workers_and_drops_stale_duplicate_ack(
    monkeypatch,
) -> None:
    async def scenario():
        outputs = []

        async def send_to_scheduler(_socket, _request):
            return None

        class RecordingSocketMapping:
            def send_output(self, ipc_name, output):
                outputs.append((ipc_name, output))

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        router = object.__new__(MultiTokenizerRouter)
        router.send_to_scheduler = object()
        router.all_worker_ipcs = {"worker-a", "worker-b"}
        _register_live_tokenizer_workers(router)
        router.socket_mapping = RecordingSocketMapping()
        router.pause_owners = set()
        router.active_remote_pause_owner = None
        router.pending_remote_pause_requests = deque()
        router._pause_transitions = {}
        router._pause_poisoned_owners = set()

        identity = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner="remote-weight-transfer:first",
            action="pause",
            expected_state=True,
        )
        request = PauseGenerationReqInput(
            mode="in_place",
            rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
            http_worker_ipc="worker-a",
        )
        await router._handle_pause_continue_request(request)

        transition = router._pause_transitions[identity.transition_id]
        assert transition.expected_workers == frozenset({"worker-a", "worker-b"})

        router.all_worker_ipcs.add("worker-c")
        router._handle_pause_continue_ack(
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=request.rid,
                is_pause=True,
                http_worker_ipc="worker-c",
            )
        )
        assert transition.acked_workers == set()

        first_ack = multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
            rid=request.rid,
            is_pause=True,
            http_worker_ipc="worker-a",
        )
        router._handle_pause_continue_ack(first_ack)
        router._handle_pause_continue_ack(first_ack)
        assert transition.acked_workers == {"worker-a"}

        stale_identity = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner=identity.owner,
            action=identity.action,
            expected_state=identity.expected_state,
        )
        router._handle_pause_continue_ack(
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(
                    stale_identity
                ),
                is_pause=True,
                http_worker_ipc="worker-b",
            )
        )
        assert transition.acked_workers == {"worker-a"}

        router._handle_pause_continue_ack(
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=request.rid,
                is_pause=True,
                http_worker_ipc="worker-b",
            )
        )
        applied_rid = multi_tokenizer_mixin_module._encode_pause_transition_applied(
            identity
        )
        for worker_ipc in ("worker-a", "worker-b"):
            router._handle_pause_continue_ack(
                multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                    rid=applied_rid,
                    is_pause=True,
                    http_worker_ipc=worker_ipc,
                )
            )
        committed_rid = (
            multi_tokenizer_mixin_module._encode_pause_transition_committed_ack(
                identity
            )
        )
        for worker_ipc in ("worker-a", "worker-b"):
            router._handle_pause_continue_ack(
                multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                    rid=committed_rid,
                    is_pause=True,
                    http_worker_ipc=worker_ipc,
                )
            )
        _finalize_router_transition(router, identity)

        assert identity.transition_id not in router._pause_transitions
        assert all(ipc_name != "worker-c" for ipc_name, _ in outputs)

    asyncio.run(scenario())


def test_multi_tokenizer_router_dispatch_failure_keeps_owner_poisoned(
    monkeypatch,
) -> None:
    async def scenario():
        outputs = []

        async def fail_dispatch(_socket, _request):
            raise RuntimeError("scheduler dispatch failed")

        class RecordingSocketMapping:
            def send_output(self, ipc_name, output):
                outputs.append((ipc_name, output))

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            fail_dispatch,
        )
        router = object.__new__(MultiTokenizerRouter)
        router.send_to_scheduler = object()
        router.all_worker_ipcs = {"worker-a", "worker-b"}
        _register_live_tokenizer_workers(router)
        router.socket_mapping = RecordingSocketMapping()
        router.pause_owners = set()
        router.active_remote_pause_owner = None
        router.pending_remote_pause_requests = deque()
        router._pause_transitions = {}
        router._pause_poisoned_owners = set()

        identity = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner="remote-weight-transfer:dispatch-failure",
            action="pause",
            expected_state=True,
        )
        await router._handle_pause_continue_request(
            PauseGenerationReqInput(
                mode="in_place",
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                http_worker_ipc="worker-a",
            )
        )

        assert router.pause_owners == {identity.owner}
        assert router._pause_poisoned_owners == {identity.owner}
        assert router._pause_transitions == {}
        assert {ipc_name for ipc_name, _ in outputs} == {"worker-a", "worker-b"}
        assert all(
            output.is_pause is True
            and (
                output.http_worker_ipc
                == multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED
            )
            for _, output in outputs
        )

    asyncio.run(scenario())


def test_multi_tokenizer_router_commit_failure_retries_committed_terminal(
    monkeypatch,
) -> None:
    async def scenario():
        async def send_to_scheduler(_socket, _request):
            return None

        class FailingCommitSocketMapping:
            def __init__(self):
                self.failed_once = False

            def send_output(self, ipc_name, output):
                if (
                    not self.failed_once
                    and ipc_name == "worker-b"
                    and (
                        output.http_worker_ipc
                        == multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED
                    )
                ):
                    self.failed_once = True
                    raise RuntimeError("commit dispatch failed")

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        router = object.__new__(MultiTokenizerRouter)
        router.send_to_scheduler = object()
        router.all_worker_ipcs = {"worker-a", "worker-b"}
        _register_live_tokenizer_workers(router)
        socket_mapping = FailingCommitSocketMapping()
        router.socket_mapping = socket_mapping
        router.pause_owners = set()
        router.active_remote_pause_owner = None
        router.pending_remote_pause_requests = deque()
        router._pause_transitions = {}
        router._pause_poisoned_owners = set()

        owner = "remote-weight-transfer:confirmation-failure"
        request = PauseGenerationReqInput(
            mode="in_place",
            rid=owner,
            http_worker_ipc="worker-a",
        )
        await router._handle_pause_continue_request(request)
        identity = multi_tokenizer_mixin_module._decode_pause_transition(request.rid)
        assert identity is not None

        for worker_ipc in ("worker-a", "worker-b"):
            router._handle_pause_continue_ack(
                multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                    rid=request.rid,
                    is_pause=True,
                    http_worker_ipc=worker_ipc,
                )
            )
        applied_rid = multi_tokenizer_mixin_module._encode_pause_transition_applied(
            identity
        )
        for worker_ipc in ("worker-a", "worker-b"):
            router._handle_pause_continue_ack(
                multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                    rid=applied_rid,
                    is_pause=True,
                    http_worker_ipc=worker_ipc,
                )
            )
        await asyncio.sleep(
            multi_tokenizer_mixin_module._PAUSE_TRANSITION_RETRY_INTERVAL_SEC * 2
        )
        committed_rid = (
            multi_tokenizer_mixin_module._encode_pause_transition_committed_ack(
                identity
            )
        )
        for worker_ipc in ("worker-a", "worker-b"):
            router._handle_pause_continue_ack(
                multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                    rid=committed_rid,
                    is_pause=True,
                    http_worker_ipc=worker_ipc,
                )
            )
        _finalize_router_transition(router, identity)
        assert router.pause_owners == {owner}
        assert router._pause_poisoned_owners == set()
        assert identity.transition_id not in router._pause_transitions
        assert socket_mapping.failed_once is True

    asyncio.run(scenario())


def test_multi_tokenizer_router_confirms_idempotent_release_retry(monkeypatch) -> None:
    async def scenario():
        scheduled = []
        outputs = []
        owner = "remote-weight-transfer:retry"

        async def send_to_scheduler(_socket, request):
            scheduled.append(request)

        class RecordingSocketMapping:
            def send_output(self, ipc_name, output):
                outputs.append((ipc_name, output))

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        router = object.__new__(MultiTokenizerRouter)
        router.send_to_scheduler = object()
        router.all_worker_ipcs = {"worker-a", "worker-b"}
        workers = _register_live_tokenizer_workers(router)
        router.socket_mapping = RecordingSocketMapping()
        router.pause_owners = {owner}
        router.active_remote_pause_owner = owner
        router.pending_remote_pause_requests = deque()
        router._pause_transitions = {}
        router._pause_poisoned_owners = {owner}
        router._pause_owner_workers[owner] = workers["worker-a"]

        identity = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner=owner,
            action="continue",
            expected_state=False,
        )
        request = ContinueGenerationReqInput(
            rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
            http_worker_ipc="worker-a",
        )
        await router._handle_pause_continue_request(request)
        for worker_ipc in ("worker-a", "worker-b"):
            router._handle_pause_continue_ack(
                multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                    rid=request.rid,
                    is_pause=False,
                    http_worker_ipc=worker_ipc,
                )
            )
        applied_rid = multi_tokenizer_mixin_module._encode_pause_transition_applied(
            identity
        )
        for worker_ipc in ("worker-a", "worker-b"):
            router._handle_pause_continue_ack(
                multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                    rid=applied_rid,
                    is_pause=False,
                    http_worker_ipc=worker_ipc,
                )
            )
        await asyncio.sleep(0)
        committed_rid = (
            multi_tokenizer_mixin_module._encode_pause_transition_committed_ack(
                identity
            )
        )
        for worker_ipc in ("worker-a", "worker-b"):
            router._handle_pause_continue_ack(
                multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                    rid=committed_rid,
                    is_pause=False,
                    http_worker_ipc=worker_ipc,
                )
            )
        _finalize_router_transition(router, identity)

        assert scheduled == [request]
        assert router._pause_poisoned_owners == set()
        assert identity.transition_id not in router._pause_transitions
        assert any(
            output.http_worker_ipc
            == multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED
            for _, output in outputs
        )

    asyncio.run(scenario())


def test_multi_tokenizer_router_rejects_stale_expected_state(monkeypatch) -> None:
    async def scenario():
        async def send_to_scheduler(_socket, _request):
            return None

        class RecordingSocketMapping:
            def send_output(self, _ipc_name, _output):
                return None

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        router = object.__new__(MultiTokenizerRouter)
        router.send_to_scheduler = object()
        router.all_worker_ipcs = {"worker-a", "worker-b"}
        workers = _register_live_tokenizer_workers(router)
        router.socket_mapping = RecordingSocketMapping()
        owner = "remote-weight-transfer:continue"
        router.pause_owners = {owner}
        router.active_remote_pause_owner = owner
        router.pending_remote_pause_requests = deque()
        router._pause_transitions = {}
        router._pause_poisoned_owners = set()
        router._pause_owner_workers[owner] = workers["worker-a"]

        request = ContinueGenerationReqInput(
            rid=owner,
            http_worker_ipc="worker-a",
        )
        await router._handle_pause_continue_request(request)
        identity = multi_tokenizer_mixin_module._decode_pause_transition(request.rid)
        assert identity is not None
        assert identity.expected_state is False

        for worker_ipc in ("worker-a", "worker-b"):
            router._handle_pause_continue_ack(
                multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                    rid=request.rid,
                    is_pause=False,
                    http_worker_ipc=worker_ipc,
                )
            )
        router.pause_owners.add("admin")
        applied_rid = multi_tokenizer_mixin_module._encode_pause_transition_applied(
            identity
        )
        for worker_ipc in ("worker-a", "worker-b"):
            router._handle_pause_continue_ack(
                multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                    rid=applied_rid,
                    is_pause=False,
                    http_worker_ipc=worker_ipc,
                )
            )

        assert router.pause_owners == {owner, "admin"}
        assert router._pause_poisoned_owners == {owner}
        assert identity.transition_id not in router._pause_transitions

    asyncio.run(scenario())


def test_multi_tokenizer_router_drops_stale_request_before_owner_mutation(
    monkeypatch,
) -> None:
    async def scenario():
        scheduled = []

        async def send_to_scheduler(_socket, request):
            scheduled.append(request)

        class RecordingSocketMapping:
            def send_output(self, _ipc_name, _output):
                raise AssertionError("stale transition must not be broadcast")

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        router = object.__new__(MultiTokenizerRouter)
        router.send_to_scheduler = object()
        router.all_worker_ipcs = {"worker-a", "worker-b"}
        router.socket_mapping = RecordingSocketMapping()
        router.pause_owners = set()
        router.active_remote_pause_owner = None
        router.pending_remote_pause_requests = deque()
        router._pause_transitions = {}
        router._pause_poisoned_owners = set()

        owner = "remote-weight-transfer:stale-request"
        stale = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner=owner,
            action="pause",
            expected_state=True,
        )
        current = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner=owner,
            action="continue",
            expected_state=False,
        )
        router._pause_owner_transitions = {owner: current}

        await router._handle_pause_continue_request(
            PauseGenerationReqInput(
                mode="in_place",
                rid=multi_tokenizer_mixin_module._encode_pause_transition(stale),
                http_worker_ipc="worker-a",
            )
        )

        assert scheduled == []
        assert router.pause_owners == set()
        assert router.active_remote_pause_owner is None
        assert router.pending_remote_pause_requests == deque()
        assert router._pause_transitions == {}
        assert router._pause_poisoned_owners == set()
        assert router._pause_owner_transitions == {owner: current}

    asyncio.run(scenario())


def test_multi_tokenizer_router_retry_reestablishes_pause_before_owner_handoff(
    monkeypatch,
) -> None:
    async def scenario():
        scheduled = []
        dispatch_attempts = 0
        first_owner = "remote-weight-transfer:first"
        second_owner = "remote-weight-transfer:second"

        async def send_to_scheduler(_socket, request):
            nonlocal dispatch_attempts
            dispatch_attempts += 1
            if dispatch_attempts == 1:
                raise RuntimeError("initial pause dispatch failed")
            scheduled.append(request)

        class RecordingSocketMapping:
            def send_output(self, _ipc_name, _output):
                return None

        monkeypatch.setattr(
            multi_tokenizer_mixin_module,
            "async_sock_send",
            send_to_scheduler,
        )
        router = object.__new__(MultiTokenizerRouter)
        router.send_to_scheduler = object()
        router.all_worker_ipcs = {"worker-a", "worker-b"}
        _register_live_tokenizer_workers(router)
        router.socket_mapping = RecordingSocketMapping()
        router.pause_owners = set()
        router.active_remote_pause_owner = None
        router.pending_remote_pause_requests = deque()
        router._pause_transitions = {}
        router._pause_poisoned_owners = set()

        def ack_all(request):
            identity = multi_tokenizer_mixin_module._decode_pause_transition(
                request.rid
            )
            assert identity is not None
            for worker_ipc in ("worker-a", "worker-b"):
                router._handle_pause_continue_ack(
                    multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                        rid=request.rid,
                        is_pause=identity.expected_state,
                        http_worker_ipc=worker_ipc,
                    )
                )
            transition = router._pause_transitions.get(identity.transition_id)
            if transition is None or not transition.confirmation_sent:
                return
            applied_rid = multi_tokenizer_mixin_module._encode_pause_transition_applied(
                identity
            )
            for worker_ipc in ("worker-a", "worker-b"):
                router._handle_pause_continue_ack(
                    multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                        rid=applied_rid,
                        is_pause=identity.expected_state,
                        http_worker_ipc=worker_ipc,
                    )
                )
            transition = router._pause_transitions.get(identity.transition_id)
            if transition is None or not transition.committed:
                return
            committed_rid = (
                multi_tokenizer_mixin_module._encode_pause_transition_committed_ack(
                    identity
                )
            )
            for worker_ipc in ("worker-a", "worker-b"):
                router._handle_pause_continue_ack(
                    multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                        rid=committed_rid,
                        is_pause=identity.expected_state,
                        http_worker_ipc=worker_ipc,
                    )
                )
            _finalize_router_transition(router, identity)

        first_pause = PauseGenerationReqInput(
            mode="in_place",
            rid=first_owner,
            http_worker_ipc="worker-a",
        )
        await router._handle_pause_continue_request(first_pause)
        assert router._pause_poisoned_owners == {first_owner}

        second_pause = PauseGenerationReqInput(
            mode="in_place",
            rid=second_owner,
            http_worker_ipc="worker-b",
        )
        await router._handle_pause_continue_request(second_pause)
        ack_all(second_pause)

        first_continue = ContinueGenerationReqInput(
            rid=first_owner,
            http_worker_ipc="worker-a",
        )
        await router._handle_pause_continue_request(first_continue)
        ack_all(first_continue)
        await asyncio.sleep(0)
        ack_all(first_continue)
        await asyncio.sleep(0)
        ack_all(second_pause)

        assert scheduled == [second_pause]
        assert router.active_remote_pause_owner == second_owner
        assert router._pause_poisoned_owners == set()
        second_identity = multi_tokenizer_mixin_module._decode_pause_transition(
            second_pause.rid
        )
        assert second_identity is not None
        assert second_identity.transition_id not in router._pause_transitions

    asyncio.run(scenario())


def test_multi_tokenizer_worker_rejects_confirmation_after_deadline() -> None:
    async def scenario():
        owner = "remote-weight-transfer:deadline"
        identity = multi_tokenizer_mixin_module._PauseTransitionIdentity(
            transition_id="expired-transition",
            owner=owner,
            action="continue",
            expected_state=False,
            deadline_monotonic_ns=(
                multi_tokenizer_mixin_module.time.monotonic_ns() - 1
            ),
        )
        worker = object.__new__(TokenizerWorker)
        worker.is_pause = True
        worker.is_pause_cond = asyncio.Condition()
        worker._generation_pause_owners = {owner}
        worker._generation_pause_resume_pending = set()
        pending = asyncio.get_running_loop().create_future()
        worker._pause_continue_futures = {
            identity.transition_id: (identity, pending),
        }
        worker._prepared_pause_transitions = {
            identity.transition_id: identity,
        }
        worker._latest_pause_transitions = {owner: identity}
        worker._poisoned_pause_transitions = {}

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
            ),
        )

        with pytest.raises(asyncio.TimeoutError):
            await pending
        assert worker.is_pause is True
        assert worker._generation_pause_owners == {owner}
        assert worker._generation_pause_resume_pending == {owner}
        assert worker._prepared_pause_transitions == {}
        assert worker._poisoned_pause_transitions == {
            identity.transition_id: identity,
        }

    asyncio.run(scenario())


def test_multi_tokenizer_worker_drops_stale_transition_confirmation() -> None:
    async def scenario():
        owner = "remote-weight-transfer:second"
        identity = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner=owner,
            action="continue",
            expected_state=False,
        )
        stale = multi_tokenizer_mixin_module._new_pause_transition_identity(
            owner=owner,
            action="continue",
            expected_state=False,
        )
        worker = object.__new__(TokenizerWorker)
        worker.is_pause = True
        worker.is_pause_cond = asyncio.Condition()
        worker._generation_pause_owners = {owner}
        worker._generation_pause_resume_pending = set()
        pending = asyncio.get_running_loop().create_future()
        worker._pause_continue_futures = {
            identity.transition_id: (identity, pending),
        }
        worker._prepared_pause_transitions = {}
        worker._poisoned_pause_transitions = {}

        async def dispatch_ack(_ack):
            return None

        worker._async_dispatch_to_scheduler = dispatch_ack

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
            ),
        )

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(stale),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
            ),
        )
        assert worker.is_pause is True
        assert worker._generation_pause_owners == {owner}
        assert pending.done() is False

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_CONFIRMED,
            ),
        )
        assert worker.is_pause is True
        assert worker._generation_pause_owners == {owner}
        assert pending.done() is False

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_COMMITTED,
            ),
        )
        assert worker.is_pause is False
        assert pending.done() is False

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(identity),
                is_pause=False,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_FINALIZED,
            ),
        )
        assert pending.result() is True

        await TokenizerWorker._apply_pause_continue_broadcast(
            worker,
            multi_tokenizer_mixin_module.PauseContinueBroadcastReq(
                rid=multi_tokenizer_mixin_module._encode_pause_transition(stale),
                is_pause=True,
                http_worker_ipc=multi_tokenizer_mixin_module._PAUSE_TRANSITION_FAILED,
            ),
        )
        assert worker.is_pause is False
        assert worker._generation_pause_owners == set()
        assert worker._generation_pause_resume_pending == set()

    asyncio.run(scenario())


@pytest.mark.parametrize("enable_weight_runtime_manifest", [False, True])
def test_multi_tokenizer_online_update_returns_controlled_error(
    enable_weight_runtime_manifest,
) -> None:
    async def scenario():
        requests = []

        async def update(request):
            requests.append(request)
            return [SimpleNamespace(success=True, message="Success.")]

        manager = object.__new__(TokenizerManager)
        manager.server_args = SimpleNamespace(
            tokenizer_worker_num=2,
            enable_weight_runtime_manifest=enable_weight_runtime_manifest,
            dp_size=1,
            enable_dp_attention=False,
        )
        manager.auto_create_handle_loop = lambda: None
        manager.abort_request = lambda **_kwargs: None
        manager.is_pause = False
        manager.is_pause_cond = asyncio.Condition()
        manager.model_update_lock = RWLock()
        manager.update_weights_from_distributed_communicator = update
        request = UpdateWeightsFromDistributedReqInput(
            names=[],
            dtypes=[],
            shapes=[],
        )

        with pytest.raises(
            tokenizer_control_mixin_module.fastapi.HTTPException
        ) as raised:
            await TokenizerControlMixin.update_weights_from_distributed(
                manager,
                request,
            )

        assert raised.value.status_code == 409
        assert raised.value.detail == (
            "online weight updates require a single tokenizer worker; "
            "restart with --tokenizer-worker-num 1"
        )
        assert requests == []

    asyncio.run(scenario())


def test_multi_tokenizer_owner_check_returns_controlled_error() -> None:
    manager = SimpleNamespace(
        server_args=SimpleNamespace(
            tokenizer_worker_num=2,
            enable_weight_runtime_manifest=True,
        )
    )

    with pytest.raises(tokenizer_control_mixin_module.fastapi.HTTPException) as raised:
        TokenizerControlMixin._require_single_tokenizer_weight_update_owner(manager)

    assert raised.value.status_code == 409
    assert raised.value.detail == (
        "online weight updates require a single tokenizer worker; "
        "restart with --tokenizer-worker-num 1"
    )


def test_tokenizer_begin_uses_a_new_attempt_id_for_each_fanout() -> None:
    async def release(_request):
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

    for _ in range(2):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                transfer_id="transfer-1",
            )
        )

    requests = manager._remote_weight_transfer_begin_requests
    assert [request.transfer_id for request in requests] == [
        "transfer-1",
        "transfer-1",
    ]
    assert all(request.request_id for request in requests)
    assert requests[0].request_id != requests[1].request_id


def test_tokenizer_begin_timeout_without_identity_keeps_cleanup_pending() -> None:
    releases = []

    async def release(request):
        releases.append(request.transfer_id)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)

    async def timeout_after_dispatch(request, *, deadline_unix_sec):
        del request, deadline_unix_sec
        raise TimeoutError("fan-out response deadline expired: received 0/1")

    manager.begin_remote_instance_weight_transfer_communicator = timeout_after_dispatch

    with pytest.raises(RemoteInstanceWeightTransferBeginError) as raised:
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                transfer_id="transfer-1",
            )
        )

    assert raised.value.session_state == "cleanup_pending"
    assert releases == []


def test_tokenizer_begin_rejects_fail_closed_weight_state() -> None:
    async def release(request):
        del request
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)
    manager.weight_update_fail_closed = True

    with pytest.raises(RuntimeError, match="snapshot export is disabled"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(manager)
        )

    assert manager._remote_weight_transfer_events == []


def test_tokenizer_begin_rejects_mixed_worker_revision_semantics() -> None:
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
                placements=[_placement(dp_rank=0)],
                bindings=[_binding(dp_rank=0, lease_id="lease-dp0")],
                manifest_revision_semantics=ARTIFACT_WEIGHT_VERSION_V1,
            ),
            SimpleNamespace(
                success=True,
                message="Success.",
                session_state="created",
                placements=[_placement(dp_rank=1)],
                bindings=[_binding(dp_rank=1, lease_id="lease-dp1")],
            ),
        ],
        release,
    )

    with pytest.raises(RuntimeError, match="incompatible manifest revision semantics"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                manifest_format="placement_binding_v1",
                manifest_revision_semantics=ARTIFACT_WEIGHT_VERSION_V1,
                transfer_id="transfer-1",
            )
        )

    assert released == ["transfer-1"]


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
            manager,
            "transfer-1",
            lease_fence=status["lease_fence"],
            generation=status["generation"],
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

        async def begin(request, *, deadline_unix_sec):
            assert deadline_unix_sec > time.time()
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

        async def reused_begin(request, *, deadline_unix_sec):
            assert deadline_unix_sec > time.time()
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

        async def begin(request, *, deadline_unix_sec):
            nonlocal begin_calls, session_exists
            assert deadline_unix_sec > time.time()
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

    async def reused_begin(request, *, deadline_unix_sec):
        assert deadline_unix_sec > time.time()
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
            release_requests.append(request)
            return [SimpleNamespace(success=False, message="release still pending")]

        manager = _tokenizer_manager(
            [
                SimpleNamespace(
                    transfer_id="transfer-cancelled",
                    success=True,
                    message="Success.",
                    session_state="created",
                    manifests=[_manifest()],
                    lease_fence="lease-v1:source-authority",
                    generation=1,
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
        assert len(release_requests) == 3
        assert {
            (request.transfer_id, request.lease_fence, request.generation)
            for request in release_requests
        } == {
            ("transfer-cancelled", "lease-v1:source-authority", 1),
        }
        status = (
            await TokenizerControlMixin.get_remote_instance_weight_transfer_session(
                manager, "transfer-cancelled"
            )
        )
        assert status["session_state"] == "cleanup_pending"
        assert status["last_release_success"] is False
        assert status["lease_fence"] == "lease-v1:source-authority"
        assert status["generation"] == 1

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

    async def begin(request, *, deadline_unix_sec):
        assert deadline_unix_sec > time.time()
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

    async def begin(request, *, deadline_unix_sec):
        assert deadline_unix_sec > time.time()
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
        "manifest_revision_semantics": "hf_revision_v1",
        "lease_fence": result["lease_fence"],
        "generation": 1,
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

    with pytest.raises(RuntimeError, match="one model generation") as raised:
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager, manifest_format="placement_binding_v1"
            )
        )

    assert raised.value.session_state == "cleanup_pending"
    assert released == []


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
                lease_fence="fence-1",
                generation=1,
            )
        ],
        release,
    )

    with pytest.raises(RuntimeError) as raised:
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                transfer_id="transfer-1",
                lease_fence="fence-1",
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
                    lease_fence="fence-1",
                    generation=1,
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
                lease_fence="fence-1",
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
    assert requests[0].request_id


def test_tokenizer_release_uses_a_new_attempt_id_for_each_retry() -> None:
    requests = []

    async def release(request):
        requests.append(request)
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)
    for _ in range(2):
        success, _ = asyncio.run(
            TokenizerControlMixin.release_remote_instance_weight_transfer(
                manager,
                "transfer-1",
            )
        )
        assert success is True

    assert requests[0].request_id != requests[1].request_id


def test_tokenizer_renew_uses_unique_request_ids_and_source_deadline() -> None:
    requests = []
    started_at = time.time()
    granted_deadlines = [
        (started_at + 45, started_at + 40),
        (started_at + 55, started_at + 50),
    ]

    async def renew(request, *, deadline_unix_sec):
        assert deadline_unix_sec > time.time()
        assert request.deadline_unix_sec == deadline_unix_sec
        requests.append(request)
        return [
            SimpleNamespace(
                request_id=request.request_id,
                success=True,
                message="Success.",
                deadline_unix_sec=deadline,
            )
            for deadline in granted_deadlines[len(requests) - 1]
        ]

    async def release(request):
        return [SimpleNamespace(success=True, message="Success.")]

    manager = _tokenizer_manager([], release)
    manager.renew_remote_instance_weight_transfer_communicator = renew
    manager._remote_weight_transfer_session_index = {
        "transfer-from-another-worker": {
            "transfer_id": "transfer-from-another-worker",
            "lease_fence": "fence-1",
            "generation": 7,
            "session_state": "active",
            "deadline_unix_sec": time.time() + 120,
        }
    }

    for _ in range(2):
        success, _ = asyncio.run(
            TokenizerControlMixin.renew_remote_instance_weight_transfer(
                manager,
                "transfer-from-another-worker",
                lease_timeout_sec=60,
                lease_fence="fence-1",
                generation=7,
            )
        )
        assert success is True

    assert [request.transfer_id for request in requests] == [
        "transfer-from-another-worker",
        "transfer-from-another-worker",
    ]
    assert all(request.request_id for request in requests)
    assert requests[0].request_id != requests[1].request_id
    assert all(request.lease_fence == "fence-1" for request in requests)
    assert all(request.generation == 7 for request in requests)
    status = asyncio.run(
        TokenizerControlMixin.get_remote_instance_weight_transfer_session(
            manager,
            "transfer-from-another-worker",
        )
    )
    assert status["deadline_unix_sec"] == granted_deadlines[-1][1]
    assert status["session_state"] == "active"
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
            "lease_fence": result["lease_fence"],
            "manifest_format": "runtime_v1",
            "manifest_revision_semantics": "hf_revision_v1",
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
    begin = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            manager,
            lease_timeout_sec=60,
            transfer_id="transfer-1",
        )
    )

    now[0] = 161.0
    success, message = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            manager,
            "transfer-1",
            lease_fence=begin["lease_fence"],
            generation=begin["generation"],
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
            manager,
            "transfer-1",
            lease_fence=begin["lease_fence"],
            generation=begin["generation"],
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
                lease_fence="fence-1",
                generation=1,
            )
        ],
        release,
    )

    with pytest.raises(RuntimeError, match="no runtime manifests"):
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(
                manager,
                lease_fence="fence-1",
            )
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

    with pytest.raises(RuntimeError, match="semantically inconsistent") as raised:
        asyncio.run(
            TokenizerControlMixin.begin_remote_instance_weight_transfer(manager)
        )

    assert raised.value.session_state == "cleanup_pending"
    assert released == []


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


def _tokenizer_manager_for_remote_transfer_scheduler(scheduler):
    async def release(request):
        return [scheduler.release_remote_instance_weight_transfer(request)]

    manager = _tokenizer_manager([], release)

    async def begin(request, *, deadline_unix_sec):
        assert request.deadline_unix_sec == deadline_unix_sec
        return [scheduler.begin_remote_instance_weight_transfer(request)]

    async def renew(request, *, deadline_unix_sec):
        assert request.deadline_unix_sec == deadline_unix_sec
        return [scheduler.renew_remote_instance_weight_transfer(request)]

    async def status(request, *, deadline_unix_sec):
        assert request.deadline_unix_sec == deadline_unix_sec
        return [scheduler.get_remote_instance_weight_transfer_session(request)]

    manager.begin_remote_instance_weight_transfer_communicator = begin
    manager.renew_remote_instance_weight_transfer_communicator = renew
    manager.get_remote_instance_weight_transfer_session_communicator = status
    return manager


def test_begin_cross_tokenizer_workers_control_one_fenced_scheduler_lease(
    monkeypatch,
) -> None:
    released = []
    renewed = []
    manifest = _manifest()
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **_kwargs: manifest,
        release_weight_runtime_manifest=lambda lease_id: released.append(lease_id),
        renew_weight_runtime_manifest=lambda lease_id, lease_timeout_sec: renewed.append(
            (lease_id, lease_timeout_sec)
        ),
    )
    scheduler = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    owner = _tokenizer_manager_for_remote_transfer_scheduler(scheduler)
    peer = _tokenizer_manager_for_remote_transfer_scheduler(scheduler)

    session = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            owner,
            transfer_id="transfer-cross-worker",
            lease_fence="fence-cross-worker",
        )
    )
    identity = {
        "lease_fence": session["lease_fence"],
        "generation": session["generation"],
    }
    deadline_before_stale_control = dict(scheduler.remote_weight_transfer_deadlines)

    with pytest.raises(RuntimeError, match="lease fence"):
        asyncio.run(
            TokenizerControlMixin.get_remote_instance_weight_transfer_session(
                peer,
                "transfer-cross-worker",
            )
        )
    missing_identity_renewed, _ = asyncio.run(
        TokenizerControlMixin.renew_remote_instance_weight_transfer(
            peer,
            "transfer-cross-worker",
            lease_timeout_sec=60,
        )
    )
    missing_identity_released, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            peer,
            "transfer-cross-worker",
        )
    )
    fence_as_id_released, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            peer,
            session["lease_fence"],
            **identity,
        )
    )
    unknown_released, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            peer,
            "unknown-transfer",
            **identity,
        )
    )
    stale_renewed, _ = asyncio.run(
        TokenizerControlMixin.renew_remote_instance_weight_transfer(
            peer,
            "transfer-cross-worker",
            lease_timeout_sec=60,
            lease_fence="stale-fence",
            generation=session["generation"],
        )
    )

    assert missing_identity_renewed is False
    assert missing_identity_released is False
    assert fence_as_id_released is False
    assert unknown_released is False
    assert stale_renewed is False
    assert renewed == []
    assert released == []
    assert scheduler.remote_weight_transfer_deadlines == deadline_before_stale_control
    assert list(scheduler.remote_weight_transfer_leases) == ["transfer-cross-worker"]

    status = asyncio.run(
        TokenizerControlMixin.get_remote_instance_weight_transfer_session(
            peer,
            "transfer-cross-worker",
            **identity,
        )
    )
    assert status["session_state"] == "active"
    assert status["lease_fence"] == session["lease_fence"]
    assert status["lease_fence"].startswith("lease-v1:")

    renewed_ok, _ = asyncio.run(
        TokenizerControlMixin.renew_remote_instance_weight_transfer(
            peer,
            "transfer-cross-worker",
            lease_timeout_sec=60,
            **identity,
        )
    )
    assert renewed_ok is True
    assert renewed == [("lease-0", 60)]

    released_ok, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            peer,
            "transfer-cross-worker",
            **identity,
        )
    )
    assert released_ok is True
    assert released == ["lease-0"]
    assert scheduler.remote_weight_transfer_leases == {}

    stale_tombstone_release, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            peer,
            "transfer-cross-worker",
            lease_fence="stale-fence",
            generation=session["generation"],
        )
    )
    matching_tombstone_release, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            peer,
            "transfer-cross-worker",
            **identity,
        )
    )
    assert stale_tombstone_release is False
    assert matching_tombstone_release is True
    assert released == ["lease-0"]


def test_begin_retry_on_another_tokenizer_worker_recovers_same_fence(
    monkeypatch,
) -> None:
    manifest = _manifest()
    runner = SimpleNamespace(
        get_remote_instance_weight_runtime_manifest=lambda **_kwargs: manifest,
        release_weight_runtime_manifest=lambda _lease_id: None,
    )
    scheduler = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )
    first_worker = _tokenizer_manager_for_remote_transfer_scheduler(scheduler)
    retry_worker = _tokenizer_manager_for_remote_transfer_scheduler(scheduler)
    first = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            first_worker,
            transfer_id="transfer-lost-response",
        )
    )
    retried = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            retry_worker,
            transfer_id="transfer-lost-response",
        )
    )

    assert first["lease_fence"].startswith("lease-v1:")
    assert retried["lease_fence"] == first["lease_fence"]
    assert retried["generation"] == first["generation"]
    assert len(scheduler.remote_weight_transfer_leases) == 1


if __name__ == "__main__":
    pytest_args = ["-x" if argument == "-f" else argument for argument in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *pytest_args]))
