from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from sglang.srt.model_loader import mooncake_reshard_backend as backend_module
from sglang.srt.model_loader.mooncake_reshard_backend import (
    MooncakeWeightReshardBackend,
)
from sglang.srt.model_loader.weight_reshard_backend import (
    WeightReshardCompletionUnknownError,
)


class _Adapter:
    def __init__(self) -> None:
        self.source_placement = SimpleNamespace(
            resource_id="model",
            revision="revision",
            weight_generation=7,
        )
        self.target_placement = SimpleNamespace(
            resource_id="model",
            revision="revision",
        )

    def source_placement_and_bindings(self, *args):
        return self.source_placement, ()

    def gather_target_placement(self, *args, **kwargs):
        return self.target_placement

    def runtime_binding_manifest(self, *args, **kwargs):
        return "target-binding"


class _UnknownReader:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def execute(self, *args, **kwargs):
        raise backend_module.TransferCompletionUnknownError(
            "completion remains unknown",
            pending_transfer_id="pending-1",
        )


def _prepared_context(monkeypatch, events: list[str]):
    backend = MooncakeWeightReshardBackend()
    backend._adapter = _Adapter()
    monkeypatch.setattr(
        backend_module,
        "PlacementInventoryParticipant",
        lambda placement: placement,
    )

    def plan_placement_transfer_to_local_target(*args):
        events.append("plan")
        return SimpleNamespace(operations=(object(),))

    monkeypatch.setattr(
        backend_module,
        "plan_placement_transfer_to_local_target",
        plan_placement_transfer_to_local_target,
    )
    monkeypatch.setattr(
        backend_module,
        "bind_logical_transfer_plan",
        lambda plan, *args, **kwargs: plan,
    )

    target_resource = SimpleNamespace(
        placement_inventory=SimpleNamespace(participant_id="target-participant")
    )

    @contextmanager
    def bind():
        events.append("bind")
        try:
            yield "target-binding-inventory"
        finally:
            events.append("binding-close")

    target_resource.bind = bind

    @contextmanager
    def target_inventory_builder(**kwargs):
        try:
            yield target_resource
        finally:
            events.append("target-close")

    return backend, backend.prepare(
        model=object(),
        source_placement_inventories=(),
        source_binding_inventories=(),
        target_inventory_builder=target_inventory_builder,
        target_model_id="model",
        target_revision="revision",
        target_instance_id="instance",
        target_endpoint="endpoint",
        world_group=object(),
    )


def test_provider_retains_resources_before_propagating_completion_unknown(
    monkeypatch,
) -> None:
    events: list[str] = []
    backend, prepared_context = _prepared_context(monkeypatch, events)
    monkeypatch.setattr(
        backend_module,
        "MooncakeTransferEngineReader",
        _UnknownReader,
    )

    with prepared_context as prepared:
        with pytest.raises(
            WeightReshardCompletionUnknownError,
            match="completion is unknown",
        ) as raised:
            backend.execute(prepared, transfer_engine=object())
        assert raised.value.pending_transfer_id == "pending-1"

    assert prepared.completion_unknown_retained is True
    assert events == ["plan", "bind"]

    backend.close_after_terminal(prepared)
    assert prepared.completion_unknown_retained is False
    assert events == ["plan", "bind", "binding-close", "target-close"]


def test_provider_closes_resources_on_normal_context_exit(monkeypatch) -> None:
    events: list[str] = []
    _, prepared_context = _prepared_context(monkeypatch, events)

    with prepared_context as prepared:
        assert prepared.closed is False
        assert events == ["plan", "bind"]

    assert prepared.closed is True
    assert events == ["plan", "bind", "binding-close", "target-close"]
