from __future__ import annotations

import pytest

from sglang.srt.model_loader import remote_instance_weight_loader_utils as utils


class _Response:
    def __init__(self, status_code=200, payload=None, text="") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _current_payload(*, transfer_id="transfer-1", lease_timeout_sec=90):
    return {
        "transfer_id": transfer_id,
        "success": True,
        "message": "Success.",
        "session_state": "created",
        "placement_inventories": [{"inventory_id": "placement"}],
        "binding_inventories": [{"generation": 3}],
        "lease_timeout_sec": lease_timeout_sec,
    }


def test_begin_uses_one_endpoint_semantics_and_parses_current_payload(
    monkeypatch,
) -> None:
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return _Response(payload=_current_payload())

    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source",
        lease_timeout_sec=60,
        transfer_id="transfer-1",
    )

    assert session.transfer_id == "transfer-1"
    assert session.lease_timeout_sec == 90
    assert session.placement_inventories == [{"inventory_id": "placement"}]
    assert session.binding_inventories == [{"generation": 3}]
    assert calls[0][0].endswith("/remote_instance_weight_transfer")
    assert calls[0][1]["params"] == {
        "lease_timeout_sec": 60,
        "transfer_id": "transfer-1",
    }


@pytest.mark.parametrize(
    "legacy_field",
    (
        "weight_runtime_manifests",
        "source_weight_placements",
        "source_weight_runtime_bindings",
        "manifest_format",
    ),
)
def test_begin_rejects_legacy_or_extra_payload_and_releases_owner_session(
    monkeypatch,
    legacy_field,
) -> None:
    payload = _current_payload()
    payload[legacy_field] = []
    releases = []
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: _Response(payload=payload),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: releases.append(transfer_id) or True,
    )

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", transfer_id="transfer-1"
    )

    assert session is None
    assert releases == ["transfer-1"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("success", False),
        ("session_state", "unknown"),
        ("placement_inventories", ["not-an-inventory"]),
        ("binding_inventories", ["not-an-inventory"]),
    ),
)
def test_begin_rejects_semantically_invalid_current_payload(
    monkeypatch,
    field,
    value,
) -> None:
    payload = _current_payload()
    payload[field] = value
    releases = []
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *args, **kwargs: _Response(payload=payload),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: releases.append(transfer_id) or True,
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source",
            transfer_id="transfer-1",
        )
        is None
    )
    assert releases == ["transfer-1"]


def test_begin_retries_response_loss_with_same_transfer_id(monkeypatch) -> None:
    seen_ids = []

    def post(url, **kwargs):
        seen_ids.append(kwargs["params"]["transfer_id"])
        if len(seen_ids) == 1:
            raise TimeoutError("response lost")
        return _Response(payload=_current_payload(transfer_id=seen_ids[-1]))

    monkeypatch.setattr(utils.requests, "post", post)

    session = utils.begin_remote_instance_weight_transfer(
        "http://source", transfer_id="stable-transfer"
    )

    assert session.transfer_id == "stable-transfer"
    assert seen_ids == ["stable-transfer", "stable-transfer"]


def test_begin_releases_cleanup_pending_but_not_conflict(monkeypatch) -> None:
    releases = []
    responses = iter(
        (
            _Response(
                status_code=409,
                payload={
                    "transfer_id": "owned",
                    "session_state": "cleanup_pending",
                },
            ),
            _Response(
                status_code=409,
                payload={
                    "transfer_id": "conflict",
                    "session_state": "conflict",
                },
            ),
        )
    )
    monkeypatch.setattr(utils.requests, "post", lambda *a, **k: next(responses))
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: releases.append(transfer_id) or True,
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source", transfer_id="owned"
        )
        is None
    )
    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source", transfer_id="conflict"
        )
        is None
    )
    assert releases == ["owned"]


def test_begin_rejects_mismatched_transfer_id_and_cleans_both_ids(
    monkeypatch,
) -> None:
    releases = []
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda *a, **k: _Response(payload=_current_payload(transfer_id="other")),
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: releases.append(transfer_id) or True,
    )

    assert (
        utils.begin_remote_instance_weight_transfer(
            "http://source", transfer_id="expected"
        )
        is None
    )
    assert releases == ["other", "expected"]


def test_renew_timeout_stays_inside_remaining_lease(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        utils.requests,
        "post",
        lambda url, **kwargs: calls.append(kwargs) or _Response(),
    )

    assert utils.renew_remote_instance_weight_transfer(
        "http://source",
        "transfer-1",
        300,
        remaining_lease_sec=2.5,
    )

    assert 0 < calls[0]["timeout"] < 2.5


def test_heartbeat_renews_and_surfaces_failure(monkeypatch) -> None:
    renewals = []
    monkeypatch.setattr(
        utils,
        "renew_remote_instance_weight_transfer",
        lambda *args, **kwargs: renewals.append((args, kwargs)) or True,
    )
    heartbeat = utils.RemoteInstanceWeightTransferHeartbeat(
        "http://source",
        "transfer-1",
        lease_timeout_sec=30,
        renew_interval_sec=1,
    )

    heartbeat.start()
    heartbeat.stop()

    assert renewals
    heartbeat.raise_if_failed()


class _World:
    rank_in_group = 0
    world_size = 1

    def __init__(self) -> None:
        self.gathered = []

    def broadcast_object(self, value, src):
        return value

    def all_gather_object(self, value):
        self.gathered.append(value)
        return [value]


def test_world_coordinator_releases_only_after_terminal_success(monkeypatch) -> None:
    session = utils.RemoteInstanceWeightTransferSession(
        transfer_id="transfer-1",
        placement_inventories=[{}],
        binding_inventories=[{}],
        lease_timeout_sec=30,
    )
    released = []
    monkeypatch.setattr(
        utils,
        "begin_remote_instance_weight_transfer",
        lambda seed_url: session,
    )
    monkeypatch.setattr(
        utils.RemoteInstanceWeightTransferHeartbeat,
        "start",
        lambda self: None,
    )
    monkeypatch.setattr(
        utils.RemoteInstanceWeightTransferHeartbeat,
        "stop",
        lambda self: None,
    )
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda seed_url, transfer_id: released.append(transfer_id) or True,
    )
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", _World()
    )

    assert coordinator.acquire() is session
    assert coordinator.ready_for_transfer(True) is True
    assert coordinator.finish(local_success=True) == (True, True)
    assert released == ["transfer-1"]


def test_world_coordinator_keeps_source_leased_for_completion_unknown(
    monkeypatch,
) -> None:
    world = _World()
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", world
    )
    coordinator._acquired = True
    coordinator.session = utils.RemoteInstanceWeightTransferSession(
        transfer_id="transfer-1",
        placement_inventories=[{}],
        binding_inventories=[{}],
        lease_timeout_sec=30,
    )
    releases = []
    monkeypatch.setattr(
        utils,
        "release_remote_instance_weight_transfer",
        lambda *args: releases.append(args) or True,
    )

    assert coordinator.finish(
        local_success=False,
        local_release_safe=False,
    ) == (False, False)
    assert releases == []


def test_world_coordinator_rejects_boolean_coercion_at_coordination_boundary() -> None:
    coordinator = utils.RemoteInstanceWeightTransferWorldCoordinator(
        "http://source", _World()
    )
    coordinator._acquired = True
    coordinator.session = utils.RemoteInstanceWeightTransferSession(
        transfer_id="transfer-1",
        placement_inventories=[{}],
        binding_inventories=[{}],
        lease_timeout_sec=30,
    )

    with pytest.raises(TypeError, match="local_ready"):
        coordinator.ready_for_transfer(1)
    with pytest.raises(TypeError, match="local_success"):
        coordinator.finish(local_success=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transfer_id": "", "placement_inventories": [{}], "binding_inventories": [{}]},
        {
            "transfer_id": "transfer-1",
            "placement_inventories": [],
            "binding_inventories": [],
        },
        {
            "transfer_id": "transfer-1",
            "placement_inventories": [{}],
            "binding_inventories": [],
        },
    ],
)
def test_remote_weight_transfer_session_rejects_invalid_wire_shape(kwargs) -> None:
    with pytest.raises(ValueError):
        utils.RemoteInstanceWeightTransferSession(
            lease_timeout_sec=30,
            **kwargs,
        )
