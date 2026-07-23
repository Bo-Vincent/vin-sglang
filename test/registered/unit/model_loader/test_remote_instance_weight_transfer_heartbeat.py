import importlib.util
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

if importlib.util.find_spec("requests") is None:
    requests = ModuleType("requests")
    requests.post = None
    requests.delete = None
    sys.modules["requests"] = requests

from sglang.srt.model_loader import remote_instance_weight_loader_utils as utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def deterministic_transfer_id(monkeypatch) -> None:
    monkeypatch.setattr(
        utils.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="transfer-1"),
    )


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


def test_placement_binding_negotiation_uses_the_runtime_builder_probe(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        utils,
        "local_mooncake_supports_placement_binding",
        lambda: calls.append("probe") or True,
    )

    assert utils.supports_mooncake_placement_binding_v1() is True
    assert calls == ["probe"]


def test_begin_requests_and_parses_split_source_manifest(monkeypatch) -> None:
    calls = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "source_weight_placements": [{"placement_id": "source-placement"}],
                "source_weight_runtime_bindings": [
                    {
                        "placement_id": "source-placement",
                        "lease_id": "source-lease",
                    }
                ],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: True)
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert calls[0][1]["params"] == {
        "lease_timeout_sec": 90,
        "manifest_format": "placement_binding_v1",
        "transfer_id": "transfer-1",
    }
    assert session.manifests == []
    assert session.source_placements == [{"placement_id": "source-placement"}]
    assert session.source_bindings == [
        {"placement_id": "source-placement", "lease_id": "source-lease"}
    ]
    assert session.manifest_format == "placement_binding_v1"


def test_begin_falls_back_to_runtime_manifest_when_capability_is_missing(
    monkeypatch,
) -> None:
    calls = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: False)
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert calls[0][1]["params"] == {
        "lease_timeout_sec": 90,
        "manifest_format": "runtime_v1",
        "transfer_id": "transfer-1",
    }
    assert session.manifests == [{"model_id": "model"}]
    assert session.source_placements is None
    assert session.source_bindings is None
    assert session.manifest_format == "runtime_v1"


def test_begin_retries_runtime_manifest_once_for_unsupported_split_format(
    monkeypatch,
) -> None:
    calls = []

    class UnsupportedResponse:
        status_code = 422
        text = ""

        @staticmethod
        def json():
            return {
                "detail": [
                    {
                        "type": "literal_error",
                        "loc": ["query", "manifest_format"],
                        "msg": "Input should be 'runtime_v1'",
                    }
                ],
            }

    class RuntimeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return UnsupportedResponse() if len(calls) == 1 else RuntimeResponse()

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: True)
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert [call[1]["params"]["manifest_format"] for call in calls] == [
        "placement_binding_v1",
        "runtime_v1",
    ]
    assert session.transfer_id == "transfer-1"
    assert session.manifest_format == "runtime_v1"


def test_begin_retries_runtime_manifest_once_for_explicit_conflict_format(
    monkeypatch,
) -> None:
    calls = []

    class UnsupportedResponse:
        status_code = 409
        text = "unsupported manifest_format=placement_binding_v1"

        @staticmethod
        def json():
            return {"detail": "unsupported manifest_format=placement_binding_v1"}

    class RuntimeResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return UnsupportedResponse() if len(calls) == 1 else RuntimeResponse()

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: True)
    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert [call[1]["params"]["manifest_format"] for call in calls] == [
        "placement_binding_v1",
        "runtime_v1",
    ]
    assert session.transfer_id == "transfer-1"


def test_begin_does_not_retry_unrelated_conflict(monkeypatch) -> None:
    calls = []
    released = []

    class Response:
        status_code = 409
        text = "a weight snapshot lease is active"

        @staticmethod
        def json():
            return {"detail": "a weight snapshot lease is active"}

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: True)
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Response(),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or True,
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source", lease_timeout_sec=90
        )
        is None
    )
    assert len(calls) == 1
    assert released == []


def test_begin_retries_release_for_structured_cleanup_pending_response(
    monkeypatch,
) -> None:
    release_attempts = []

    class Response:
        status_code = 409
        text = "snapshot cleanup remains pending"

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "session_state": "cleanup_pending",
                "message": "snapshot cleanup remains pending",
            }

    def release(seed_url, transfer_id):
        release_attempts.append((seed_url, transfer_id))
        return len(release_attempts) >= 2

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: False)
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(utils, "release_remote_instance_weight_transfer", release)

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source",
            lease_timeout_sec=90,
        )
        is None
    )
    assert release_attempts == [
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
    ]


def test_begin_does_not_release_structured_conflict_response(monkeypatch) -> None:
    released = []

    class Response:
        status_code = 409
        text = "transfer ID conflict"

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "session_state": "conflict",
                "message": "transfer ID conflict",
            }

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: False)
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or True,
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source",
            lease_timeout_sec=90,
        )
        is None
    )
    assert released == []


def test_begin_does_not_retry_split_request_after_server_error(monkeypatch) -> None:
    calls = []

    class Response:
        status_code = 503
        text = "temporarily unavailable"

        @staticmethod
        def json():
            return {"detail": "temporarily unavailable"}

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: True)
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Response(),
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source", lease_timeout_sec=90
        )
        is None
    )
    assert len(calls) == 1


def test_begin_does_not_release_unsupported_response_without_ownership_state(
    monkeypatch,
) -> None:
    calls = []
    released = []

    class Response:
        status_code = 422
        text = "unsupported manifest_format=placement_binding_v1"

        @staticmethod
        def json():
            return {
                "detail": "unsupported manifest_format=placement_binding_v1",
                "transfer_id": "transfer-rejected",
            }

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: True)
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Response(),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or True,
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source", lease_timeout_sec=90
        )
        is None
    )
    assert len(calls) == 1
    assert released == []


def test_begin_reuses_legacy_runtime_session_returned_for_split_request(
    monkeypatch,
) -> None:
    calls = []

    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: True)
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Response(),
    )

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert len(calls) == 1
    assert session.transfer_id == "transfer-1"
    assert session.manifests == [{"model_id": "model"}]
    assert session.manifest_format == "runtime_v1"


def test_begin_releases_invalid_split_session_and_fails_closed(
    monkeypatch,
) -> None:
    calls = []
    released = []

    class InvalidSplitResponse:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "source_weight_placements": [{"placement_id": "source-placement"}],
                "source_weight_runtime_bindings": [],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return InvalidSplitResponse()

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: True)
    monkeypatch.setattr(utils.requests, "post", post)
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or False,
    )

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert released == [
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
    ]
    assert len(calls) == 1
    assert calls[0][1]["params"]["manifest_format"] == "placement_binding_v1"
    assert session is None


def test_begin_releases_transfer_id_when_payload_validation_fails(
    monkeypatch,
) -> None:
    released = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [],
                "lease_timeout_sec": 90,
            }

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: False)
    monkeypatch.setattr(utils.requests, "post", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or True,
    )

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", lease_timeout_sec=90
    )

    assert session is None
    assert released == [("http://source", "transfer-1")]


def test_begin_retries_response_loss_with_the_same_target_generated_id(
    monkeypatch,
) -> None:
    calls = []
    released = []

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "transfer_id": "transfer-1",
                "weight_runtime_manifests": [{"model_id": "model"}],
                "lease_timeout_sec": 90,
            }

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise RuntimeError("response lost")
        return Response()

    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: False)
    monkeypatch.setattr(utils.requests, "post", post)
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append((seed_url, transfer_id)) or True,
    )

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        lease_timeout_sec=90,
    )

    assert session.transfer_id == "transfer-1"
    assert len(calls) == 2
    assert {call[1]["params"]["transfer_id"] for call in calls} == {"transfer-1"}
    assert released == []


def test_begin_releases_known_id_after_repeated_response_loss(monkeypatch) -> None:
    released = []
    monkeypatch.setattr(utils, "supports_mooncake_placement_binding_v1", lambda: False)
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("response lost")),
    )

    def release(seed_url, transfer_id):
        released.append((seed_url, transfer_id))
        return len(released) >= 2

    monkeypatch.setattr(utils, "release_remote_instance_weight_transfer", release)

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source",
            lease_timeout_sec=90,
        )
        is None
    )
    assert released == [
        ("http://source", "transfer-1"),
        ("http://source", "transfer-1"),
    ]


def test_begin_rejects_explicit_empty_transfer_id(monkeypatch) -> None:
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: pytest.fail("invalid ID must fail before HTTP"),
    )

    with pytest.raises(ValueError, match="non-empty string"):
        utils.begin_remote_instance_weight_transfer(
            "http://source",
            transfer_id="",
        )


def test_renew_timeout_is_strictly_inside_remaining_lease_window(
    monkeypatch,
) -> None:
    calls = []

    class Response:
        status_code = 200

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        return Response()

    monkeypatch.setattr(utils.requests, "post", post)

    assert utils.renew_remote_instance_weight_transfer(
        "http://source",
        "transfer-1",
        lease_timeout_sec=30,
        remaining_lease_sec=2.0,
    )
    assert 0 < calls[0][1]["timeout"] < 2.0


def test_heartbeat_renews_in_background(monkeypatch) -> None:
    background_renewed = threading.Event()
    attempts = []

    def renew(seed_url, transfer_id, lease_timeout_sec, **kwargs):
        del kwargs
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

    def renew(seed_url, transfer_id, lease_timeout_sec, **kwargs):
        del seed_url, transfer_id, lease_timeout_sec, kwargs
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


def test_world_transfer_session_releases_source_when_broadcast_fails(
    monkeypatch,
) -> None:
    calls = []
    session = _session()

    class FailingWorldGroup(_FakeWorldGroup):
        def broadcast_object(self, obj=None, src=0):
            del obj, src
            raise RuntimeError("broadcast failed")

    class FakeHeartbeat:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self):
            calls.append("heartbeat-started")

        def raise_if_failed(self):
            return None

        def stop(self):
            calls.append("heartbeat-stopped")

    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url: session,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: (
            calls.append(("release", seed_url, transfer_id)) or True
        ),
    )
    monkeypatch.setattr(utils, "RemoteInstanceWeightTransferHeartbeat", FakeHeartbeat)
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", FailingWorldGroup(rank=0)
    )

    try:
        coordinator.acquire()
    except RuntimeError as error:
        assert "broadcast failed" in str(error)
    else:
        raise AssertionError("broadcast failure must escape acquire")

    assert calls == [
        "heartbeat-started",
        "heartbeat-stopped",
        ("release", "http://source", "transfer-1"),
    ]


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
        utils, "renew_remote_instance_weight_transfer", lambda *args, **kwargs: True
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


def test_world_transfer_invalid_readiness_requires_explicit_release(
    monkeypatch,
) -> None:
    calls = []
    session = _session()
    monkeypatch.setattr(
        utils, "begin_remote_instance_weight_transfer", lambda seed_url: session
    )
    monkeypatch.setattr(
        utils, "renew_remote_instance_weight_transfer", lambda *args, **kwargs: True
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


def test_world_transfer_readiness_collective_failure_requires_recovery(
    monkeypatch,
) -> None:
    calls = []
    session = _session()
    monkeypatch.setattr(
        utils, "begin_remote_instance_weight_transfer", lambda seed_url: session
    )
    monkeypatch.setattr(
        utils, "renew_remote_instance_weight_transfer", lambda *args, **kwargs: True
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


def test_world_transfer_unknown_completion_requires_explicit_release(
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
