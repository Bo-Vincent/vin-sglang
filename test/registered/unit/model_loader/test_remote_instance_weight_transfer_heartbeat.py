import importlib.util
import sys
import threading
from types import ModuleType, SimpleNamespace

if importlib.util.find_spec("requests") is None:
    requests = ModuleType("requests")
    requests.post = None
    requests.delete = None
    sys.modules["requests"] = requests

from sglang.srt.model_loader import remote_instance_weight_loader_utils as utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_begin_preserves_server_lease_timeout(monkeypatch) -> None:
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert session.lease_timeout_sec == 90


def test_heartbeat_renews_in_background(monkeypatch) -> None:
    background_renewed = threading.Event()
    attempts = []

    def renew(seed_url, transfer_id, lease_timeout_sec):
        attempts.append((seed_url, transfer_id, lease_timeout_sec))
        if len(attempts) >= 2:
            background_renewed.set()
        return True

    monkeypatch.setattr(utils, "renew_remote_instance_weight_transfer", renew)
    heartbeat = utils.RemoteInstanceWeightTransferHeartbeat(
        "http://source",
        "transfer-1",
        lease_timeout_sec=30,
        renew_interval_sec=0.01,
    )

    heartbeat.start()
    try:
        assert attempts[0] == ("http://source", "transfer-1", 30)
        assert background_renewed.wait(timeout=1)
        assert len(attempts) >= 2
        heartbeat.raise_if_failed()
    finally:
        heartbeat.stop()


def test_heartbeat_rejects_an_expired_session_before_loader_uses_it(
    monkeypatch,
) -> None:
    attempted = threading.Event()

    def renew(seed_url, transfer_id, lease_timeout_sec):
        attempted.set()
        return False

    monkeypatch.setattr(utils, "renew_remote_instance_weight_transfer", renew)
    heartbeat = utils.RemoteInstanceWeightTransferHeartbeat(
        "http://source",
        "transfer-1",
        lease_timeout_sec=30,
        renew_interval_sec=0.01,
    )

    try:
        heartbeat.start()
    except RuntimeError as error:
        assert "renew" in str(error).lower()
    else:
        raise AssertionError("initial lease renewal failure must fail closed")
    assert attempted.wait(timeout=1)
    heartbeat.stop()


class _FakeWorldGroup:
    def __init__(
        self,
        *,
        rank: int,
        broadcast_session=None,
        gathered_readiness=None,
        gathered_outcomes=None,
        broadcast_outcome=None,
        readiness_error=None,
    ) -> None:
        self.rank_in_group = rank
        self.world_size = 4
        self.broadcast_session = broadcast_session
        self.gathered_readiness = (
            gathered_readiness
            if gathered_readiness is not None
            else [True] * self.world_size
        )
        self.gathered_outcomes = gathered_outcomes or [(True, True)] * self.world_size
        self.broadcast_outcome = broadcast_outcome
        self.readiness_error = readiness_error
        self.broadcasts = []
        self.gathers = []

    def broadcast_object(self, obj=None, src=0):
        self.broadcasts.append((obj, src))
        if len(self.broadcasts) == 1:
            return obj if self.rank_in_group == src else self.broadcast_session
        return obj if self.rank_in_group == src else self.broadcast_outcome

    def all_gather_object(self, obj):
        self.gathers.append(obj)
        if isinstance(obj, bool):
            if self.readiness_error is not None:
                raise self.readiness_error
            return self.gathered_readiness
        return self.gathered_outcomes


def _session():
    return SimpleNamespace(
        transfer_id="transfer-1",
        manifests=[{"model_id": "model"}],
        lease_timeout_sec=90,
    )


def test_world_transfer_session_owner_acquires_and_releases_once(monkeypatch) -> None:
    calls = []
    session = _session()

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            calls.append("heartbeat-created")

        def start(self):
            calls.append("heartbeat-started")

        def raise_if_failed(self):
            calls.append("heartbeat-checked")

        def stop(self):
            calls.append("heartbeat-stopped")

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url: calls.append("acquire") or session,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: calls.append("release") or True,
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", _FakeWorldGroup(rank=0)
    )

    assert coordinator.acquire() is session
    coordinator.raise_if_failed()
    assert coordinator.ready_for_transfer(True) is True
    world_success, release_success = coordinator.finish(local_success=True)

    assert world_success is True
    assert release_success is True
    assert calls.count("acquire") == 1
    assert calls.count("release") == 1
    assert "heartbeat-started" in calls
    assert "heartbeat-stopped" in calls


def test_world_transfer_acquire_broadcasts_failure_when_cleanup_raises(
    monkeypatch,
) -> None:
    session = _session()

    class FailingHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("heartbeat start failed")

    monkeypatch.setattr(
        utils, "begin_remote_instance_weight_transfer", lambda seed_url: session
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("release failed")),
    )
    monkeypatch.setattr(
        utils, "RemoteInstanceWeightTransferHeartbeat", FailingHeartbeat
    )
    group = _FakeWorldGroup(rank=0)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is None
    assert group.broadcasts == [(None, 0)]


def test_world_transfer_finish_broadcasts_when_release_raises(monkeypatch) -> None:
    session = _session()

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def raise_if_failed(self):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(
        utils, "begin_remote_instance_weight_transfer", lambda seed_url: session
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("release failed")),
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    group = _FakeWorldGroup(rank=0)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is True
    assert coordinator.finish(local_success=True) == (True, False)
    assert group.broadcasts[-1] == ((True, False), 0)


def test_world_transfer_readiness_rejects_partial_world_and_runs_once(
    monkeypatch,
) -> None:
    session = _session()
    monkeypatch.setattr(
        utils, "begin_remote_instance_weight_transfer", lambda seed_url: session
    )
    monkeypatch.setattr(
        utils, "renew_remote_instance_weight_transfer", lambda *args: True
    )
    monkeypatch.setattr(
        utils, "release_remote_instance_weight_transfer", lambda *args: True
    )
    group = _FakeWorldGroup(
        rank=0,
        gathered_readiness=[True, False, True, True],
        gathered_outcomes=[(False, True), (True, True), (True, True), (True, True)],
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is False
    try:
        coordinator.ready_for_transfer(True)
    except RuntimeError as error:
        assert "already checked" in str(error)
    else:
        raise AssertionError("readiness gate must run exactly once")
    assert group.gathers[0] is True
    assert coordinator.finish(local_success=False) == (False, True)


def test_world_transfer_invalid_readiness_keeps_lease_until_ttl(monkeypatch) -> None:
    calls = []
    session = _session()
    monkeypatch.setattr(
        utils, "begin_remote_instance_weight_transfer", lambda seed_url: session
    )
    monkeypatch.setattr(
        utils, "renew_remote_instance_weight_transfer", lambda *args: True
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda *args: calls.append("release") or True,
    )
    group = _FakeWorldGroup(
        rank=0,
        gathered_readiness=[True],
        gathered_outcomes=[(False, False)] * 4,
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is False
    assert coordinator.finish(local_success=False) == (False, False)
    assert calls == []


def test_world_transfer_readiness_collective_failure_keeps_lease_until_ttl(
    monkeypatch,
) -> None:
    calls = []
    session = _session()
    monkeypatch.setattr(
        utils, "begin_remote_instance_weight_transfer", lambda seed_url: session
    )
    monkeypatch.setattr(
        utils, "renew_remote_instance_weight_transfer", lambda *args: True
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda *args: calls.append("release") or True,
    )
    group = _FakeWorldGroup(
        rank=0,
        gathered_outcomes=[(False, False)] * 4,
        readiness_error=RuntimeError("readiness collective failed"),
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is False
    assert coordinator.finish(local_success=False) == (False, False)
    assert calls == []


def test_world_transfer_session_keeps_lease_until_ttl_when_completion_is_unknown(
    monkeypatch,
) -> None:
    calls = []
    session = _session()

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def raise_if_failed(self):
            pass

        def stop(self):
            calls.append("heartbeat-stopped")

    monkeypatch.setattr(
        utils, "begin_remote_instance_weight_transfer", lambda seed_url: session
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: calls.append("release") or True,
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        _FakeWorldGroup(
            rank=0,
            gathered_outcomes=[(False, False)] * 4,
        ),
    )

    assert coordinator.acquire() is session
    world_success, release_success = coordinator.finish(
        local_success=False,
        local_release_safe=False,
    )

    assert world_success is False
    assert release_success is False
    assert calls == ["heartbeat-stopped"]


def test_world_transfer_session_follower_reuses_broadcast_and_never_calls_source(
    monkeypatch,
) -> None:
    session = _session()
    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url: (_ for _ in ()).throw(
            AssertionError("only world rank zero may acquire")
        ),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: (_ for _ in ()).throw(
            AssertionError("only world rank zero may release")
        ),
    )
    group = _FakeWorldGroup(
        rank=2,
        broadcast_session=session,
        gathered_outcomes=[
            (True, True),
            (False, True),
            (True, True),
            (True, True),
        ],
        broadcast_outcome=(False, True),
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", group
    )

    assert coordinator.acquire() is session
    world_success, release_success = coordinator.finish(local_success=True)

    assert world_success is False
    assert release_success is True


def test_world_transfer_session_rejects_invalid_collective_outcomes(
    monkeypatch,
) -> None:
    calls = []
    session = _session()

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def raise_if_failed(self):
            pass

        def stop(self):
            calls.append("heartbeat-stopped")

    monkeypatch.setattr(
        utils, "begin_remote_instance_weight_transfer", lambda seed_url: session
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: calls.append("release") or True,
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source",
        _FakeWorldGroup(rank=0, gathered_outcomes=[(True, True)]),
    )

    assert coordinator.acquire() is session
    assert coordinator.finish(local_success=True) == (False, False)
    assert calls == ["heartbeat-stopped"]
