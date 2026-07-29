import asyncio
import time
from types import SimpleNamespace

import pytest

from sglang.srt.managers.io_struct import (
    BeginRemoteInstanceWeightTransferReqInput,
    ReleaseRemoteInstanceWeightTransferReqInput,
    RenewRemoteInstanceWeightTransferReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _SingleRankCollective:
    rank_in_group = 0
    world_size = 1

    @staticmethod
    def all_gather_object(value, **_kwargs):
        return [value]

    @staticmethod
    def gather_object(value, dst=0, **_kwargs):
        assert dst == 0
        return [value]

    @staticmethod
    def scatter_object(values=None, src=0, **_kwargs):
        assert src == 0
        return values[0]


class _RootTwoRankCollective:
    rank_in_group = 0
    world_size = 2
    cpu_group = object()

    def __init__(self):
        self.root_authority = None
        self.peer_manifest_fence = None

    def all_gather_object(self, value, **_kwargs):
        self.root_authority = value["lease_fence"]
        peer = dict(value)
        peer["lease_fence"] = None
        return [value, peer]

    def gather_object(self, value, dst=0, **_kwargs):
        assert dst == 0
        peer_manifest = _manifest(
            "lease-1",
            worker_id="source/dp0-pp0-ep0-tp1",
        )
        self.peer_manifest_fence = self.root_authority
        peer = {
            "success": True,
            "message": "Success.",
            "session_state": "created",
            "manifest_revision_semantics": "hf_revision_v1",
            "model_id": peer_manifest["model_id"],
            "revision": peer_manifest["revision"],
            "lease_id": peer_manifest["lease_id"],
            "generation": peer_manifest["generation"],
            "lease_fence": self.peer_manifest_fence,
            "manifest": peer_manifest,
        }
        return [value, peer]

    @staticmethod
    def scatter_object(values=None, src=0, **_kwargs):
        assert src == 0
        return values[0]


def _manifest(
    lease_id: str,
    *,
    worker_id: str = "source/dp0-pp0-ep0-tp0",
) -> dict:
    return {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "revision": "main",
        "generation": 1,
        "lease_id": lease_id,
        "tensors": [{"worker_id": worker_id}],
    }


def _scheduler(collective=None):
    acquired = []
    released = []
    renewed = []

    def capture(**_kwargs):
        lease_id = f"lease-{len(acquired)}"
        acquired.append(lease_id)
        return _manifest(lease_id)

    runner = SimpleNamespace(
        server_args=SimpleNamespace(weight_cache_mode="off"),
        get_remote_instance_weight_runtime_manifest=capture,
        release_weight_runtime_manifest=released.append,
        renew_weight_runtime_manifest=lambda lease_id, lease_timeout_sec: renewed.append(
            (lease_id, lease_timeout_sec)
        ),
    )
    collective = collective or _SingleRankCollective()
    manager = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(model_runner=runner),
        draft_worker=None,
        tp_cpu_group=collective,
        world_cpu_group=collective,
        remote_weight_transfer_cpu_group=collective,
        remote_weight_transfer_control_cpu_group=collective,
        memory_saver_adapter=object(),
        flush_cache=lambda **_kwargs: True,
        is_fully_idle=lambda: True,
    )
    return manager, acquired, released, renewed


def _tokenizer_manager(scheduler):
    async def begin(request, *, deadline_unix_sec):
        assert request.deadline_unix_sec == deadline_unix_sec
        return [scheduler.begin_remote_instance_weight_transfer(request)]

    async def release(request, *, deadline_unix_sec):
        assert request.deadline_unix_sec == deadline_unix_sec
        return [scheduler.release_remote_instance_weight_transfer(request)]

    async def renew(request, *, deadline_unix_sec):
        assert request.deadline_unix_sec == deadline_unix_sec
        return [scheduler.renew_remote_instance_weight_transfer(request)]

    async def pause(_request):
        return None

    async def resume(_request):
        return None

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
        begin_remote_instance_weight_transfer_communicator=begin,
        release_remote_instance_weight_transfer_communicator=release,
        renew_remote_instance_weight_transfer_communicator=renew,
    )


def test_reacquired_lease_rejects_controls_from_evicted_incarnation() -> None:
    scheduler, acquired, released, renewed = _scheduler()
    first_owner = _tokenizer_manager(scheduler)

    first = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            first_owner,
            transfer_id="transfer-1",
            lease_fence="client-proposal",
        )
    )
    old_identity = {
        "lease_fence": first["lease_fence"],
        "generation": first["generation"],
    }
    old_begin_fence = scheduler.remote_weight_transfer_begin_fences["transfer-1"]
    released_ok, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            first_owner,
            "transfer-1",
            **old_identity,
        )
    )
    assert released_ok is True

    scheduler.remote_weight_transfer_tombstones.clear()
    scheduler.remote_weight_transfer_tombstone_fences.clear()
    scheduler.remote_weight_transfer_tombstone_generations.clear()

    replayed_begin = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
            lease_fence=old_identity["lease_fence"],
        )
    )
    assert replayed_begin.success is False
    assert "begin fence" in replayed_begin.message.lower()
    assert acquired == ["lease-0"]

    replayed_begin_token = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
            lease_fence=old_begin_fence,
        )
    )
    assert replayed_begin_token.success is False
    assert "already consumed" in replayed_begin_token.message.lower()
    assert acquired == ["lease-0"]

    second_owner = _tokenizer_manager(scheduler)
    second = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            second_owner,
            transfer_id="transfer-1",
            lease_fence=old_identity["lease_fence"],
        )
    )

    assert second["lease_fence"] != old_identity["lease_fence"]
    deadline_before_stale_renew = scheduler.remote_weight_transfer_deadlines[
        "transfer-1"
    ]
    stale_renew = scheduler.renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            lease_timeout_sec=60,
            deadline_unix_sec=time.time() + 30,
            **old_identity,
        )
    )
    stale_release = scheduler.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            deadline_unix_sec=time.time() + 30,
            **old_identity,
        )
    )

    assert stale_renew.success is False
    assert stale_release.success is False
    assert renewed == []
    assert released == ["lease-0"]
    assert acquired == ["lease-0", "lease-1"]
    assert scheduler.remote_weight_transfer_leases == {"transfer-1": "lease-1"}
    assert (
        scheduler.remote_weight_transfer_deadlines["transfer-1"]
        == deadline_before_stale_renew
    )


def test_source_root_assigns_one_incarnation_to_all_model_ranks() -> None:
    collective = _RootTwoRankCollective()
    scheduler, _, _, _ = _scheduler(collective)

    result = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
            lease_fence="begin-v1:attempt-1",
        )
    )

    assert result.success is True
    assert result.lease_fence.startswith("lease-v1:")
    assert collective.root_authority == result.lease_fence
    assert collective.peer_manifest_fence == result.lease_fence
    assert scheduler.remote_weight_transfer_fences == {"transfer-1": result.lease_fence}


def test_non_root_session_record_preserves_source_incarnation() -> None:
    scheduler, _, _, _ = _scheduler()
    request = BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="Qwen/Qwen3.5-0.8B",
        revision="main",
        deadline_unix_sec=time.time() + 30,
        lease_fence="begin-v1:attempt-1",
    )
    scheduler._record_remote_weight_transfer_lease(
        request.transfer_id,
        "lease-0",
        request.lease_timeout_sec,
        generation=1,
        lease_fence="lease-v1:source-authority",
        begin_fence=request.lease_fence,
    )

    scheduler._record_remote_weight_transfer_session(
        request,
        "lease-0",
        None,
        generation=1,
    )

    assert scheduler.remote_weight_transfer_fences == {
        "transfer-1": "lease-v1:source-authority"
    }


def test_retry_from_another_tokenizer_recovers_the_active_incarnation() -> None:
    scheduler, acquired, _, _ = _scheduler()
    first_owner = _tokenizer_manager(scheduler)
    retry_owner = _tokenizer_manager(scheduler)

    first = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            first_owner,
            transfer_id="transfer-1",
        )
    )
    retried = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            retry_owner,
            transfer_id="transfer-1",
        )
    )

    assert first["lease_fence"].startswith("lease-v1:")
    assert retried["lease_fence"] == first["lease_fence"]
    assert retried["generation"] == first["generation"]
    assert acquired == ["lease-0"]


def test_legacy_unfenced_lease_remains_renewable_and_releasable() -> None:
    scheduler, _, released, renewed = _scheduler()
    created = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
        )
    )

    renewed_result = scheduler.renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            lease_timeout_sec=60,
            deadline_unix_sec=time.time() + 30,
        )
    )
    released_result = scheduler.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            deadline_unix_sec=time.time() + 30,
        )
    )

    assert created.success is True
    assert created.lease_fence is None
    assert renewed_result.success is True
    assert released_result.success is True
    assert renewed == [("lease-0", 60)]
    assert released == ["lease-0"]


def test_legacy_transfer_id_requires_fence_after_tombstone_expires(
    monkeypatch,
) -> None:
    monotonic_now = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: monotonic_now[0])
    scheduler, acquired, released, _ = _scheduler()

    legacy_begin = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
        )
    )
    first_release = scheduler.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            deadline_unix_sec=time.time() + 30,
        )
    )
    retried_release = scheduler.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            deadline_unix_sec=time.time() + 30,
        )
    )
    tombstoned_begin = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
        )
    )

    assert legacy_begin.success is True
    assert legacy_begin.lease_fence is None
    assert first_release.success is True
    assert retried_release.success is True
    assert tombstoned_begin.success is False
    assert tombstoned_begin.session_state == "released"
    assert acquired == ["lease-0"]
    assert released == ["lease-0"]

    monotonic_now[0] += 301.0
    unfenced_reuse = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
        )
    )

    assert unfenced_reuse.success is False
    assert unfenced_reuse.session_state == "conflict"
    assert "lease fence" in unfenced_reuse.message.lower()
    assert acquired == ["lease-0"]

    fenced_reuse = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
            lease_fence="begin-v1:fenced-reuse",
        )
    )
    stale_legacy_release = scheduler.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            deadline_unix_sec=time.time() + 30,
        )
    )

    assert fenced_reuse.success is True
    assert fenced_reuse.lease_fence.startswith("lease-v1:")
    assert stale_legacy_release.success is False
    assert "lease fence" in stale_legacy_release.message.lower()
    assert acquired == ["lease-0", "lease-1"]
    assert released == ["lease-0"]
    assert scheduler.remote_weight_transfer_leases == {
        "legacy-transfer": "lease-1"
    }


def test_tokenizer_unfenced_controls_cannot_bind_reused_fenced_incarnation() -> None:
    scheduler, acquired, released, renewed = _scheduler()
    owner = _tokenizer_manager(scheduler)

    first = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            owner,
            transfer_id="transfer-1",
        )
    )
    released_ok, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            owner,
            "transfer-1",
            lease_fence=first["lease_fence"],
            generation=first["generation"],
        )
    )
    assert released_ok is True
    scheduler.remote_weight_transfer_tombstones.clear()
    scheduler.remote_weight_transfer_tombstone_fences.clear()
    scheduler.remote_weight_transfer_tombstone_generations.clear()

    second = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            owner,
            transfer_id="transfer-1",
        )
    )
    deadline_before_stale_control = scheduler.remote_weight_transfer_deadlines[
        "transfer-1"
    ]

    with pytest.raises(ValueError, match="lease_fence and generation"):
        asyncio.run(
            TokenizerControlMixin.renew_remote_instance_weight_transfer(
                owner,
                "transfer-1",
                lease_timeout_sec=60,
            )
        )
    with pytest.raises(ValueError, match="lease_fence and generation"):
        asyncio.run(
            TokenizerControlMixin.release_remote_instance_weight_transfer(
                owner,
                "transfer-1",
            )
        )

    assert acquired == ["lease-0", "lease-1"]
    assert released == ["lease-0"]
    assert renewed == []
    assert scheduler.remote_weight_transfer_leases == {"transfer-1": "lease-1"}
    assert (
        scheduler.remote_weight_transfer_deadlines["transfer-1"]
        == deadline_before_stale_control
    )
    assert second["lease_fence"] != first["lease_fence"]


def test_tokenizer_unfenced_controls_remain_compatible_with_legacy_lease() -> None:
    scheduler, _, released, renewed = _scheduler()
    created = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
        )
    )
    peer = _tokenizer_manager(scheduler)

    renewed_ok, _ = asyncio.run(
        TokenizerControlMixin.renew_remote_instance_weight_transfer(
            peer,
            "legacy-transfer",
            lease_timeout_sec=60,
        )
    )
    released_ok, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            peer,
            "legacy-transfer",
        )
    )

    assert created.lease_fence is None
    assert renewed_ok is True
    assert released_ok is True
    assert renewed == [("lease-0", 60)]
    assert released == ["lease-0"]


def test_fenced_retry_upgrades_an_active_legacy_lease_without_reacquiring() -> None:
    scheduler, acquired, released, _ = _scheduler()
    legacy = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="legacy-transfer",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
        )
    )
    retry_owner = _tokenizer_manager(scheduler)

    upgraded = asyncio.run(
        TokenizerControlMixin.begin_remote_instance_weight_transfer(
            retry_owner,
            transfer_id="legacy-transfer",
        )
    )
    released_ok, _ = asyncio.run(
        TokenizerControlMixin.release_remote_instance_weight_transfer(
            retry_owner,
            "legacy-transfer",
            lease_fence=upgraded["lease_fence"],
            generation=upgraded["generation"],
        )
    )

    assert legacy.lease_fence is None
    assert upgraded["lease_fence"].startswith("lease-v1:")
    assert acquired == ["lease-0"]
    assert released_ok is True
    assert released == ["lease-0"]


def test_begin_token_can_only_release_its_current_lease() -> None:
    scheduler, _, released, _ = _scheduler()
    begin_token = "begin-v1:attempt-1"
    created = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
            lease_fence=begin_token,
        )
    )

    renewed = scheduler.renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            lease_timeout_sec=60,
            deadline_unix_sec=time.time() + 30,
            lease_fence=begin_token,
            generation=created.generation,
        )
    )
    released_result = scheduler.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            deadline_unix_sec=time.time() + 30,
            lease_fence=begin_token,
            generation=created.generation,
        )
    )

    assert renewed.success is False
    assert released_result.success is True
    assert released == ["lease-0"]


def test_expired_begin_token_cannot_release_its_current_lease() -> None:
    scheduler, _, released, _ = _scheduler()
    begin_token = "begin-v1:attempt-1"
    created = scheduler.begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            model_id="Qwen/Qwen3.5-0.8B",
            revision="main",
            deadline_unix_sec=time.time() + 30,
            lease_fence=begin_token,
        )
    )
    scheduler.remote_weight_transfer_consumed_begin_fences[begin_token] = (
        "transfer-1",
        time.time() - 1,
    )

    released_result = scheduler.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            deadline_unix_sec=time.time() + 30,
            lease_fence=begin_token,
            generation=created.generation,
        )
    )

    assert released_result.success is False
    assert released == []
    assert scheduler.remote_weight_transfer_leases == {"transfer-1": "lease-0"}


def test_active_lease_accepts_only_its_latest_begin_cleanup_alias() -> None:
    scheduler, acquired, released, _ = _scheduler()

    def begin(begin_fence: str):
        return scheduler.begin_remote_instance_weight_transfer(
            BeginRemoteInstanceWeightTransferReqInput(
                transfer_id="transfer-1",
                model_id="Qwen/Qwen3.5-0.8B",
                revision="main",
                deadline_unix_sec=time.time() + 30,
                lease_fence=begin_fence,
            )
        )

    first = begin("begin-v1:attempt-1")
    duplicate = begin("begin-v1:attempt-1")
    retried = begin("begin-v1:attempt-2")
    stale_duplicate = begin("begin-v1:attempt-1")
    stale_release = scheduler.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            deadline_unix_sec=time.time() + 30,
            lease_fence="begin-v1:attempt-1",
            generation=first.generation,
        )
    )
    current_release = scheduler.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            deadline_unix_sec=time.time() + 30,
            lease_fence="begin-v1:attempt-2",
            generation=first.generation,
        )
    )

    assert first.success is True
    assert duplicate.success is True
    assert retried.success is True
    assert duplicate.lease_fence == first.lease_fence
    assert retried.lease_fence == first.lease_fence
    assert stale_duplicate.success is False
    assert "already consumed" in stale_duplicate.message.lower()
    assert stale_release.success is False
    assert current_release.success is True
    assert acquired == ["lease-0"]
    assert released == ["lease-0"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
