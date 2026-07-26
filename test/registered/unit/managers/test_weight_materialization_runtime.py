from __future__ import annotations

import os
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.scheduler_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.managers.io_struct import (
    CommitWeightMaterializationReqInput,
    CommitWeightMaterializationReqOutput,
    PrepareWeightMaterializationReqInput,
    PrepareWeightMaterializationReqOutput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.provider import (
    WeightPayloadIdentity,
    WeightStorageDestination,
    WeightTransferCompletionUnknownError,
)
from sglang.srt.weight_transfer.storage import WeightStorageRef
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Source:
    def __init__(
        self,
        *,
        dp_rank: int = 0,
        lease_id: str | None = None,
        revision: str = "main",
    ) -> None:
        lease_id = lease_id or f"lease-dp{dp_rank}"
        fragment_id = f"fragment-dp{dp_rank}"
        tensor = WeightPlacementTensor(
            placement_fragment_id=fragment_id,
            tensor_id="model.layers.0.weight",
            runtime_name="model.layers.0.weight",
            aliases=("model.layers.0.weight",),
            global_shape=(8, 8),
            global_offset=(0, 0),
            local_shape=(8, 8),
            dtype="float16",
            itemsize=2,
            partition_dim=None,
            shard_dims=(),
            layer_id=0,
            expert_id=None,
            layout_fingerprint="dense-row-major",
            nbytes=128,
            byte_offset=0,
            rank=WeightParallelRank(dp=dp_rank),
        )
        placement_id = compute_weight_placement_id((tensor,))
        self.placement = WeightPlacementManifest(
            model_id="Qwen/Qwen3.5-0.8B",
            revision=revision,
            placement_id=placement_id,
            tensors=(tensor,),
        )
        self.binding = WeightRuntimeBindingManifest(
            model_id=self.placement.model_id,
            revision=revision,
            placement_id=placement_id,
            instance_id=f"instance-dp{dp_rank}",
            generation=7,
            lease_id=lease_id,
            fragments=(
                RuntimeWeightBinding(
                    placement_fragment_id=fragment_id,
                    fragment_id=f"runtime-{fragment_id}",
                    address=0x1000 + dp_rank * 0x1000,
                    nbytes=128,
                    storage_offset=0,
                    device="cuda",
                    is_contiguous=True,
                    worker_id=f"worker-dp{dp_rank}",
                    endpoint=f"endpoint-dp{dp_rank}",
                ),
            ),
        )
        self.payload_identity = WeightPayloadIdentity.create(
            (self.placement,),
            {fragment_id: f"sha256:{'a' * 64}"},
        )
        self.released = False
        self.quarantined = False
        self.release_calls = 0

    def payload_checksum(self, location) -> str:
        assert (
            location.placement_fragment_id
            == self.placement.tensors[0].placement_fragment_id
        )
        return self.payload_identity.fragments[0].checksum

    def release(self) -> None:
        if self.quarantined:
            raise RuntimeError("quarantined source cannot be released")
        self.release_calls += 1
        self.released = True

    def quarantine(self, error: WeightTransferCompletionUnknownError) -> None:
        self.quarantined = True
        self.operation_id = error.operation_id
        self.completion_ticket = error.completion_ticket


def _manager(source: _Source, *, external_dp_rank: int = 0):
    collective_group = object()
    runner = SimpleNamespace(
        capture_runtime_weight_snapshot_source=lambda **_kwargs: source,
    )
    return SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(model_runner=runner),
        draft_worker=None,
        tp_cpu_group=object(),
        world_cpu_group=object(),
        memory_saver_adapter=object(),
        flush_cache=lambda **_kwargs: True,
        is_fully_idle=lambda: True,
        remote_weight_transfer_cpu_group=collective_group,
        weight_materialization_cpu_group=collective_group,
        scheduler=SimpleNamespace(ps=SimpleNamespace(dp_rank=external_dp_rank)),
    )


def _prepare_request(
    *,
    materialization_id: str = "materialize-0",
    revision: str = "main",
):
    return SimpleNamespace(
        materialization_id=materialization_id,
        model_id="Qwen/Qwen3.5-0.8B",
        revision=revision,
    )


def _commit_request(
    *,
    materialization_id: str = "materialize-0",
    selected_external_dp_rank: int | None = 0,
    storage_options=None,
):
    return SimpleNamespace(
        materialization_id=materialization_id,
        selected_external_dp_rank=selected_external_dp_rank,
        storage_options=storage_options or {"catalog_path": "/catalog"},
    )


def _install_world(monkeypatch, gathered=None) -> None:
    gathered = gathered or ()
    gather_calls = 0
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group: 1 + len(gathered),
    )

    def all_gather_object(outputs, value, group):
        nonlocal gather_calls
        values = (
            [value, *gathered] if gather_calls == 0 else [value] * (1 + len(gathered))
        )
        gather_calls += 1
        assert len(outputs) == len(values)
        outputs[:] = values

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)


def _successful_capture(source: _Source) -> dict:
    return {
        "success": True,
        "message": "Success.",
        "placement": source.placement,
        "binding": source.binding,
        "payload_identity": source.payload_identity,
    }


def test_model_runner_captures_unbounded_runtime_snapshot(monkeypatch) -> None:
    calls = []
    manager = object()
    runner = SimpleNamespace(
        model=object(),
        weight_runtime_manifest_manager=manager,
        remote_instance_weight_transporter=SimpleNamespace(
            worker_id="worker-3",
            session_id="session-9",
        ),
        init_weight_runtime_manifest_manager=lambda: calls.append("init"),
    )
    expected = object()

    def capture(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        "sglang.srt.weight_transfer.runtime.RuntimeWeightSnapshotSource.capture",
        capture,
    )

    result = model_runner_module.ModelRunner.capture_runtime_weight_snapshot_source(
        runner,
        materialization_id="materialize-7",
        model_id="model",
        revision="revision",
    )

    assert result is expected
    assert calls[0] == "init"
    assert calls[1] == {
        "model": runner.model,
        "manager": manager,
        "model_id": "model",
        "revision": "revision",
        "instance_id": "sglang:session-9:materialize-7",
        "worker_id": "worker-3",
        "endpoint": "session-9",
        "lease_timeout_sec": None,
    }


def test_model_runner_materialization_does_not_require_active_te_session(
    monkeypatch,
) -> None:
    calls = []
    runner = SimpleNamespace(
        model=object(),
        gpu_id=3,
        weight_runtime_manifest_manager=object(),
        remote_instance_weight_transporter=SimpleNamespace(
            worker_id="",
            session_id="",
        ),
        init_weight_runtime_manifest_manager=lambda: None,
    )
    monkeypatch.setattr(
        "sglang.srt.weight_transfer.runtime.RuntimeWeightSnapshotSource.capture",
        lambda **kwargs: calls.append(kwargs) or object(),
    )

    model_runner_module.ModelRunner.capture_runtime_weight_snapshot_source(
        runner,
        materialization_id="materialize-7",
        model_id="model",
        revision="revision",
    )

    runtime_id = f"sglang:runtime:{os.getpid()}:3"
    assert calls[0]["instance_id"] == f"{runtime_id}:materialize-7"
    assert calls[0]["worker_id"] == runtime_id
    assert calls[0]["endpoint"] == f"local://{runtime_id}"


def test_non_dp_runtime_maps_missing_external_dp_rank_to_zero(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    manager.scheduler.ps.dp_rank = None
    _install_world(monkeypatch)

    result = manager.prepare_weight_materialization(_prepare_request())

    assert result.success is True
    assert result.external_dp_rank == 0


def test_prepare_releases_local_capture_when_any_world_rank_fails(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(
        monkeypatch,
        gathered=(
            {
                "success": False,
                "message": "rank 1 capture failed",
                "placement": None,
                "binding": None,
                "payload_identity": None,
            },
        ),
    )

    result = manager.prepare_weight_materialization(_prepare_request())

    assert result.success is False
    assert "rank 1 capture failed" in result.message
    assert source.released is True
    assert source.release_calls == 1
    assert "materialize-0" not in manager.weight_materialization_sessions


def test_prepare_rejects_multi_rank_world_without_isolated_group(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    manager.remote_weight_transfer_cpu_group = None
    manager.weight_materialization_cpu_group = None
    _install_world(monkeypatch, gathered=(_successful_capture(_Source(dp_rank=1)),))

    result = manager.prepare_weight_materialization(_prepare_request())

    assert result.success is False
    assert (
        "multi-rank weight materialization requires an isolated CPU process group"
        in result.message
    )
    assert source.released is True


def test_prepare_internal_dp_selection_can_leave_no_local_placements(
    monkeypatch,
) -> None:
    selected_source = _Source(dp_rank=0)
    local_source = _Source(dp_rank=1)
    manager = _manager(local_source)
    _install_world(
        monkeypatch,
        gathered=(_successful_capture(selected_source),),
    )

    result = manager.prepare_weight_materialization(_prepare_request())

    assert result.success is True, result.message
    assert result.total_bytes == 128
    assert result.logical_payload_digest.startswith("sha256:")
    session = manager.weight_materialization_sessions["materialize-0"]
    assert session.local_selected_placement_ids == ()
    assert tuple(
        tensor.rank.dp
        for placement in session.selected_placements
        for tensor in placement.tensors
    ) == (0,)
    assert local_source.released is False


def test_unselected_external_dp_releases_without_opening_store(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source, external_dp_rank=1)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    monkeypatch.setattr(
        weight_updater_module,
        "open_weight_snapshot_write_backend",
        lambda *_args, **_kwargs: pytest.fail("Store must not be opened"),
        raising=False,
    )

    result = manager.commit_weight_materialization(
        _commit_request(selected_external_dp_rank=0)
    )

    assert result.success is True
    assert result.selected is False
    assert result.ref is None
    assert source.released is True


def test_selected_external_dp_with_empty_local_ids_joins_store_collectives(
    monkeypatch,
) -> None:
    selected_source = _Source(dp_rank=0)
    local_source = _Source(dp_rank=1)
    manager = _manager(local_source)
    _install_world(
        monkeypatch,
        gathered=(_successful_capture(selected_source),),
    )
    assert manager.prepare_weight_materialization(_prepare_request()).success
    lifecycle = {}
    _install_fake_selected_backend(
        monkeypatch,
        materialize=lambda *_args, **_kwargs: pytest.fail(
            "empty local rank must not use the owned-source helper"
        ),
        lifecycle=lifecycle,
    )
    expected_ref = WeightStorageRef(
        provider="mooncake-store",
        storage_id="model/revision",
        manifest_key="manifest",
        manifest_digest=f"sha256:{'b' * 64}",
    )
    calls = []

    def materialize_without_local_source(**kwargs):
        calls.append(kwargs)
        assert local_source.released is False
        kwargs["attestor"].attest(
            SimpleNamespace(
                source_placements=kwargs["source_placements"],
                source_bindings=kwargs["source_bindings"],
            )
        )
        return SimpleNamespace(snapshot=SimpleNamespace(ref=expected_ref))

    monkeypatch.setattr(
        weight_updater_module,
        "materialize_weight_snapshot",
        materialize_without_local_source,
    )

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is True, result.message
    assert result.selected is True
    assert len(calls) == 1
    assert lifecycle.get("closed", False) is False
    assert len(manager.weight_storage_owners) == 1
    manager.commit_weight_materialization(
        _commit_request(selected_external_dp_rank=None)
    )
    assert lifecycle.get("closed", False) is False
    manager.close_remote_instance_weight_transfer_executor()
    assert lifecycle["closed"] is True


def test_selected_external_dp_uses_distributed_root_catalog_backend(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    group = object()
    manager.remote_weight_transfer_cpu_group = group
    manager.weight_materialization_cpu_group = group
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    calls = {}
    destination = WeightStorageDestination(
        provider="mooncake-store",
        storage_id="model/revision",
        object_prefix="objects/model/revision",
    )

    class FakeCoordinator:
        def __init__(self, requested_group):
            calls["group"] = requested_group

    class FakeWriteSpec:
        @classmethod
        def from_mapping(cls, value):
            calls["storage_options"] = value
            return SimpleNamespace(destination=destination)

    backend = SimpleNamespace(provider=object(), catalog=object())

    @contextmanager
    def open_backend(
        spec,
        *,
        local_placement_ids,
        payload_checksum_verifier,
        coordinator,
    ):
        calls["open"] = (
            spec,
            local_placement_ids,
            payload_checksum_verifier,
            coordinator,
        )
        try:
            yield backend
        finally:
            calls["closed"] = True

    expected_ref = WeightStorageRef(
        provider="mooncake-store",
        storage_id="model/revision",
        manifest_key="manifest",
        manifest_digest=f"sha256:{'b' * 64}",
    )

    def materialize(source_arg, **kwargs):
        calls["materialize_count"] = calls.get("materialize_count", 0) + 1
        calls["materialize"] = (source_arg, kwargs)
        assert source_arg.released is False
        assert kwargs["release_source"] is False
        return SimpleNamespace(snapshot=SimpleNamespace(ref=expected_ref))

    monkeypatch.setattr(
        weight_updater_module,
        "TorchDistributedWeightStoreCoordinator",
        FakeCoordinator,
        raising=False,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "WeightSnapshotWriteSpec",
        FakeWriteSpec,
        raising=False,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "open_weight_snapshot_write_backend",
        open_backend,
        raising=False,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "materialize_distributed_runtime_weight_snapshot",
        materialize,
        raising=False,
    )

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is True
    assert result.selected is True
    assert result.session_state == "published"
    assert result.ref == {
        "provider": "mooncake-store",
        "storage_id": "model/revision",
        "manifest_key": "manifest",
        "manifest_digest": f"sha256:{'b' * 64}",
    }
    assert calls["group"] is group
    assert calls["open"][1] == (source.placement.placement_id,)
    assert calls["open"][2].__self__ is source
    assert calls["materialize"][0] is source
    assert calls["materialize"][1]["publication_id"] == "materialize-0"
    assert source.released is True
    assert calls.get("closed", False) is False
    session = manager.weight_materialization_sessions["materialize-0"]
    assert session.source is None
    assert session.backend_owner is None
    assert len(manager.weight_storage_owners) == 1

    replay = manager.commit_weight_materialization(_commit_request())
    conflict = manager.commit_weight_materialization(
        _commit_request(storage_options={"catalog_path": "/other"})
    )
    assert replay == result
    assert calls["materialize_count"] == 1
    assert conflict.success is False
    assert conflict.session_state == "conflict"
    assert calls.get("closed", False) is False

    cleanup = manager.commit_weight_materialization(
        _commit_request(selected_external_dp_rank=None)
    )
    assert cleanup.success is True
    assert cleanup.session_state == "released"
    assert calls.get("closed", False) is False
    manager.close_remote_instance_weight_transfer_executor()
    assert calls["closed"] is True


def test_successful_materialization_results_are_bounded_and_replayable(
    monkeypatch,
) -> None:
    sources = {
        f"materialize-{index}": _Source(lease_id=f"lease-{index}") for index in range(3)
    }
    runner = SimpleNamespace(
        capture_runtime_weight_snapshot_source=lambda **kwargs: sources[
            kwargs["materialization_id"]
        ],
    )
    manager = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(model_runner=runner),
        draft_worker=None,
        tp_cpu_group=object(),
        world_cpu_group=object(),
        memory_saver_adapter=object(),
        flush_cache=lambda **_kwargs: True,
        is_fully_idle=lambda: True,
        remote_weight_transfer_cpu_group=object(),
        weight_materialization_cpu_group=object(),
        scheduler=SimpleNamespace(ps=SimpleNamespace(dp_rank=0)),
    )
    _install_world(monkeypatch)
    monkeypatch.setattr(
        weight_updater_module,
        "_WEIGHT_MATERIALIZATION_TERMINAL_LIMIT",
        2,
    )
    destination = WeightStorageDestination(
        provider="mooncake-store",
        storage_id="model/revision",
        object_prefix="objects/model/revision",
    )
    lifecycle = {"opens": 0, "closes": 0, "materializations": 0}

    class FakeWriteSpec:
        def __init__(self):
            self.destination = destination

        @classmethod
        def from_mapping(cls, _value):
            return cls()

    @contextmanager
    def open_backend(*_args, **_kwargs):
        lifecycle["opens"] += 1
        try:
            yield SimpleNamespace(provider=object(), catalog=object())
        finally:
            lifecycle["closes"] += 1

    def materialize(source, **kwargs):
        lifecycle["materializations"] += 1
        assert source.released is False
        assert kwargs["release_source"] is False
        materialization_id = kwargs["publication_id"]
        assert materialization_id in sources
        return SimpleNamespace(
            snapshot=SimpleNamespace(
                ref=WeightStorageRef(
                    provider="mooncake-store",
                    storage_id="model/revision",
                    manifest_key="manifest/model/revision",
                    manifest_digest=f"sha256:{'b' * 64}",
                )
            )
        )

    monkeypatch.setattr(
        weight_updater_module,
        "TorchDistributedWeightStoreCoordinator",
        lambda _group: object(),
    )
    monkeypatch.setattr(
        weight_updater_module,
        "WeightSnapshotWriteSpec",
        FakeWriteSpec,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "open_weight_snapshot_write_backend",
        open_backend,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "materialize_distributed_runtime_weight_snapshot",
        materialize,
    )

    outputs = {}
    for materialization_id in sources:
        prepare = manager.prepare_weight_materialization(
            _prepare_request(materialization_id=materialization_id)
        )
        assert prepare.success
        outputs[materialization_id] = manager.commit_weight_materialization(
            _commit_request(
                materialization_id=materialization_id,
                storage_options=(
                    {"catalog_path": "/other-store-deployment"}
                    if materialization_id == "materialize-1"
                    else None
                ),
            )
        )

    assert tuple(manager.weight_materialization_sessions) == (
        "materialize-1",
        "materialize-2",
    )
    assert lifecycle == {"opens": 3, "closes": 1, "materializations": 3}
    assert len(manager.weight_storage_owners) == 2
    replay = manager.commit_weight_materialization(
        _commit_request(materialization_id="materialize-2")
    )
    assert replay == outputs["materialize-2"]
    assert lifecycle["materializations"] == 3
    manager.close_remote_instance_weight_transfer_executor()
    assert lifecycle["closes"] == 3


def test_storage_owner_capacity_rejects_before_opening_another_backend(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    monkeypatch.setattr(weight_updater_module, "_WEIGHT_STORAGE_OWNER_LIMIT", 1)
    retained_owner = weight_updater_module._WeightStorageBackendOwner()
    manager.weight_storage_owners[
        (
            "model",
            "revision",
            f"sha256:{'a' * 64}",
            "mooncake-store",
            "storage",
            "manifest",
            f"sha256:{'b' * 64}",
        )
    ] = ({}, retained_owner)
    monkeypatch.setattr(
        weight_updater_module,
        "open_weight_snapshot_write_backend",
        lambda *_args, **_kwargs: pytest.fail("backend must not open"),
    )

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is False
    assert result.session_state == "failed"
    assert "owner limit reached" in result.message
    assert source.released is True


def test_completion_unknown_session_counts_toward_storage_owner_capacity(
    monkeypatch,
) -> None:
    first_source = _Source(lease_id="lease-first")
    second_source = _Source(lease_id="lease-second")
    sources = {
        "materialize-first": first_source,
        "materialize-second": second_source,
    }
    manager = _manager(first_source)
    manager.tp_worker.model_runner.capture_runtime_weight_snapshot_source = (
        lambda **kwargs: sources[kwargs["materialization_id"]]
    )
    _install_world(monkeypatch)
    monkeypatch.setattr(weight_updater_module, "_WEIGHT_STORAGE_OWNER_LIMIT", 1)
    materialization_calls = []

    def materialize(source_arg, **kwargs):
        materialization_calls.append(kwargs["publication_id"])
        error = WeightTransferCompletionUnknownError(
            "commit outcome is unknown",
            provider="mooncake-store",
            phase="commit",
            operation_id=kwargs["publication_id"],
            completion_ticket=f"ticket-{kwargs['publication_id']}",
        )
        source_arg.quarantine(error)
        raise error

    _install_fake_selected_backend(monkeypatch, materialize=materialize)

    assert manager.prepare_weight_materialization(
        _prepare_request(materialization_id="materialize-first")
    ).success
    first = manager.commit_weight_materialization(
        _commit_request(materialization_id="materialize-first")
    )
    assert first.completion_unknown is True

    assert manager.prepare_weight_materialization(
        _prepare_request(materialization_id="materialize-second")
    ).success
    second = manager.commit_weight_materialization(
        _commit_request(materialization_id="materialize-second")
    )

    assert second.success is False
    assert second.session_state == "failed"
    assert "owner limit reached" in second.message
    assert materialization_calls == ["materialize-first"]
    assert first_source.quarantined is True
    assert second_source.released is True


def test_exit_stack_close_failure_remains_fail_closed(monkeypatch) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    session = manager.weight_materialization_sessions["materialize-0"]

    close_calls = 0

    @contextmanager
    def failing_backend():
        nonlocal close_calls
        yield object()
        close_calls += 1
        raise RuntimeError("close failed permanently")

    owner = weight_updater_module._WeightStorageBackendOwner()
    owner.enter_context(failing_backend())
    session.backend_owner = owner

    assert manager._close_materialization_backend(session) == "close failed permanently"
    assert session.backend_owner is owner
    assert manager._close_materialization_backend(session) == "close failed permanently"
    assert session.backend_owner is owner
    assert close_calls == 1


def test_source_release_failure_is_reported_for_the_whole_model_world(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    session = manager.weight_materialization_sessions["materialize-0"]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        lambda _self, value, *, operation: [
            value,
            {
                "error": "peer source release failed",
                "completion_unknown": False,
            },
        ],
    )

    errors, completion_unknown = manager._release_materialization_source_world(
        session,
        operation="test source release",
    )

    assert errors == ["rank 1: peer source release failed"]
    assert completion_unknown is False
    assert source.released is True
    assert session.source is None


def test_cleanup_keeps_backend_when_peer_source_is_quarantined(monkeypatch) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    session = manager.weight_materialization_sessions["materialize-0"]
    session.source = None

    class Owner:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    owner = Owner()
    session.backend_owner = owner

    def gather(_self, value, *, operation):
        assert operation == "cleanup source release"
        return [
            value,
            {
                "error": "completion-unknown runtime source remains quarantined",
                "completion_unknown": True,
            },
        ]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    output = manager._cleanup_weight_materialization_session(
        _commit_request(selected_external_dp_rank=None),
        session,
    )

    assert output.success is False
    assert output.session_state == "completion_unknown"
    assert output.completion_unknown is True
    assert session.backend_owner is owner
    assert owner.closed is False


def test_selected_known_failure_releases_source(monkeypatch) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    lifecycle = {}
    _install_fake_selected_backend(
        monkeypatch,
        materialize=lambda source_arg, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("upload failed")
        ),
        lifecycle=lifecycle,
    )

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is False
    assert result.session_state == "failed"
    assert "upload failed" in result.message
    assert result.ref is None
    assert source.released is True
    assert lifecycle["closed"] is True


def test_post_publication_peer_release_failure_retains_all_backend_owners(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    lifecycle = {}
    expected_ref = WeightStorageRef(
        provider="mooncake-store",
        storage_id="model/revision",
        manifest_key="manifest",
        manifest_digest=f"sha256:{'b' * 64}",
    )

    def materialize(source_arg, **kwargs):
        assert source_arg.released is False
        assert kwargs["release_source"] is False
        return SimpleNamespace(snapshot=SimpleNamespace(ref=expected_ref))

    _install_fake_selected_backend(
        monkeypatch,
        materialize=materialize,
        lifecycle=lifecycle,
    )

    def gather(_self, value, *, operation):
        if operation in {
            "retained Store owner capacity",
            "Store backend setup",
            "Store materialization outcome",
            "retained Store owner decision",
            "cleanup source release",
            "cleanup Store backend close ownership",
            "cleanup Store backend close status",
        }:
            return [value, value]
        assert operation == "post-publication source release"
        return [
            value,
            {
                "error": "peer source release failed",
                "completion_unknown": False,
            },
        ]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is False
    assert result.session_state == "cleanup_pending"
    assert "peer source release failed" in result.message
    assert source.released is True
    session = manager.weight_materialization_sessions["materialize-0"]
    assert session.backend_owner is None
    assert len(manager.weight_storage_owners) == 1
    assert lifecycle.get("closed", False) is False

    cleanup = manager.commit_weight_materialization(
        _commit_request(selected_external_dp_rank=None)
    )
    assert cleanup.success is True
    assert cleanup.session_state == "released"
    assert len(manager.weight_storage_owners) == 1
    assert lifecycle.get("closed", False) is False


def test_selected_completion_unknown_quarantines_and_replays_terminal_output(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    calls = []
    lifecycle = {}

    def materialize(source_arg, **_kwargs):
        calls.append(source_arg)
        error = WeightTransferCompletionUnknownError(
            "commit outcome is unknown",
            provider="mooncake-store",
            phase="commit",
            operation_id="materialize-0",
            completion_ticket="ticket-0",
        )
        source_arg.quarantine(error)
        raise error

    _install_fake_selected_backend(
        monkeypatch,
        materialize=materialize,
        lifecycle=lifecycle,
    )
    request = _commit_request()

    first = manager.commit_weight_materialization(request)
    second = manager.commit_weight_materialization(request)

    assert first.success is False
    assert first.session_state == "completion_unknown"
    assert first.completion_ticket == "ticket-0"
    assert first.ref is None
    assert second == first
    assert calls == [source]
    assert source.quarantined is True
    assert source.released is False
    assert lifecycle.get("closed", False) is False

    manager.close_remote_instance_weight_transfer_executor()
    assert lifecycle["closed"] is True


def test_same_materialization_id_is_idempotent_and_rejects_conflicts(
    monkeypatch,
) -> None:
    source = _Source()
    captures = []
    manager = _manager(source)
    manager.tp_worker.model_runner.capture_runtime_weight_snapshot_source = (
        lambda **kwargs: captures.append(kwargs) or source
    )
    _install_world(monkeypatch)
    request = _prepare_request()

    first = manager.prepare_weight_materialization(request)
    replay = manager.prepare_weight_materialization(request)
    conflict = manager.prepare_weight_materialization(
        _prepare_request(revision="other")
    )

    assert replay == first
    assert len(captures) == 1
    assert conflict.success is False
    assert conflict.session_state == "conflict"
    assert source.released is False


def test_prepare_fails_all_ranks_when_peer_merge_fails(monkeypatch) -> None:
    source = _Source(dp_rank=0)
    peer = _Source(dp_rank=1)
    manager = _manager(source)

    def gather(_self, value, *, operation):
        if operation == "prepare status":
            return [value, _successful_capture(peer)]
        if operation == "prepare merge status":
            return [
                value,
                {
                    "success": False,
                    "message": "peer merge failed",
                },
            ]
        assert operation == "failed prepare merge source release"
        return [value, value]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    output = manager.prepare_weight_materialization(_prepare_request())

    assert output.success is False
    assert output.session_state == "failed"
    assert "rank 1: peer merge failed" in output.message
    assert source.released is True


def test_prepare_cleanup_session_is_created_on_all_ranks_for_peer_failure(
    monkeypatch,
) -> None:
    source = _Source(dp_rank=0)
    peer = _Source(dp_rank=1)
    manager = _manager(source)

    def gather(_self, value, *, operation):
        if operation == "prepare status":
            return [value, _successful_capture(peer)]
        if operation == "prepare merge status":
            return [
                value,
                {
                    "success": False,
                    "message": "peer merge failed",
                },
            ]
        assert operation == "failed prepare merge source release"
        return [
            value,
            {
                "error": "peer source release failed",
                "completion_unknown": False,
            },
        ]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    output = manager.prepare_weight_materialization(_prepare_request())

    assert output.success is False
    assert output.session_state == "cleanup_pending"
    assert "peer source release failed" in output.message
    assert source.released is True
    session = manager.weight_materialization_sessions["materialize-0"]
    assert session.source is None
    assert session.state == "cleanup_pending"


def test_pending_failures_return_materialization_output_types() -> None:
    manager = _manager(_Source())
    prepare = PrepareWeightMaterializationReqInput(
        materialization_id="prepare-failure",
        model_id="model",
        revision="revision",
    )
    commit = CommitWeightMaterializationReqInput(
        materialization_id="commit-failure",
        selected_external_dp_rank=0,
        storage_options={},
    )
    for request in (prepare, commit):
        future = Future()
        future.set_exception(RuntimeError("background failed"))
        pending = (
            manager.weight_materialization_pending
            if isinstance(
                request,
                (
                    PrepareWeightMaterializationReqInput,
                    CommitWeightMaterializationReqInput,
                ),
            )
            else manager.remote_weight_transfer_pending
        )
        pending.append((future, request))

    completed = manager.check_pending_remote_instance_weight_transfers()

    assert isinstance(completed[0][0], PrepareWeightMaterializationReqOutput)
    assert isinstance(completed[1][0], CommitWeightMaterializationReqOutput)
    assert all(output.success is False for output, _request in completed)
    assert all("background failed" in output.message for output, _request in completed)


def test_blocked_materialization_does_not_starve_lease_renewal(
    monkeypatch,
) -> None:
    entered_materialization = threading.Event()
    release_materialization = threading.Event()
    renewed = threading.Event()
    manager = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace()),
        draft_worker=None,
        tp_cpu_group=object(),
        world_cpu_group=object(),
        memory_saver_adapter=object(),
        flush_cache=lambda **_kwargs: True,
        is_fully_idle=lambda: True,
        remote_weight_transfer_cpu_group=object(),
        weight_materialization_cpu_group=object(),
    )

    def blocking_commit(_self, request):
        entered_materialization.set()
        assert release_materialization.wait(timeout=5)
        return CommitWeightMaterializationReqOutput(
            materialization_id=request.materialization_id,
            success=True,
            message="Success.",
            selected=True,
            session_state="published",
        )

    def renew(_self, request):
        renewed.set()
        return SimpleNamespace(
            transfer_id=request.transfer_id,
            success=True,
            message="Success.",
        )

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "commit_weight_materialization",
        blocking_commit,
    )
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "renew_remote_instance_weight_transfer",
        renew,
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda *, group: None)

    manager.defer_commit_weight_materialization(
        CommitWeightMaterializationReqInput(
            materialization_id="blocked-commit",
            selected_external_dp_rank=0,
            storage_options={},
        )
    )
    assert entered_materialization.wait(timeout=1)
    manager.defer_renew_remote_instance_weight_transfer(
        SimpleNamespace(transfer_id="renew-while-store-is-blocked")
    )

    assert renewed.wait(timeout=1)
    release_materialization.set()
    manager.close_remote_instance_weight_transfer_executor()


def test_scheduler_creates_distinct_remote_and_materialization_groups(
    monkeypatch,
) -> None:
    from sglang.srt.managers import scheduler as scheduler_module

    created_groups = []
    manager_kwargs = {}

    def new_group(*, ranks, backend):
        group = object()
        created_groups.append((tuple(ranks), backend, group))
        return group

    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    monkeypatch.setattr(
        scheduler_module,
        "SchedulerWeightUpdaterManager",
        lambda **kwargs: manager_kwargs.update(kwargs) or object(),
    )
    scheduler = SimpleNamespace(
        server_args=SimpleNamespace(enable_weight_runtime_manifest=True),
        world_group=SimpleNamespace(ranks=[0, 1], cpu_group=object()),
        tp_worker=object(),
        draft_worker=None,
        tp_cpu_group=object(),
        memory_saver_adapter=object(),
        flush_cache=lambda **_kwargs: True,
        is_fully_idle=lambda: True,
        metrics_collector=object(),
    )

    scheduler_module.Scheduler.init_weight_updater(scheduler)

    assert len(created_groups) == 2
    assert created_groups[0][:2] == ((0, 1), "gloo")
    assert created_groups[1][:2] == ((0, 1), "gloo")
    assert manager_kwargs["remote_weight_transfer_cpu_group"] is created_groups[0][2]
    assert manager_kwargs["weight_materialization_cpu_group"] is created_groups[1][2]


def test_materialization_collectives_are_serialized_inside_background_lane(
    monkeypatch,
) -> None:
    manager = _manager(_Source())
    group = object()
    manager.weight_materialization_cpu_group = group
    events = []

    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda *, group: events.append(("barrier", group)),
    )
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "prepare_weight_materialization",
        lambda _self, request: events.append(("prepare", request.materialization_id)),
    )
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "commit_weight_materialization",
        lambda _self, request: events.append(("commit", request.materialization_id)),
    )

    manager.defer_prepare_weight_materialization(
        PrepareWeightMaterializationReqInput(
            materialization_id="materialize-prepare",
            model_id="model",
            revision="revision",
        )
    )
    manager.defer_commit_weight_materialization(
        CommitWeightMaterializationReqInput(
            materialization_id="materialize-commit",
            selected_external_dp_rank=0,
            storage_options={},
        )
    )
    manager.close_remote_instance_weight_transfer_executor()

    assert events == [
        ("barrier", group),
        ("prepare", "materialize-prepare"),
        ("barrier", group),
        ("commit", "materialize-commit"),
    ]


def _install_fake_selected_backend(
    monkeypatch,
    *,
    materialize,
    lifecycle=None,
) -> None:
    lifecycle = lifecycle if lifecycle is not None else {}
    destination = WeightStorageDestination(
        provider="mooncake-store",
        storage_id="model/revision",
        object_prefix="objects/model/revision",
    )

    class FakeWriteSpec:
        def __init__(self):
            self.destination = destination

        @classmethod
        def from_mapping(cls, _value):
            return cls()

    @contextmanager
    def open_backend(*_args, **_kwargs):
        try:
            yield SimpleNamespace(provider=object(), catalog=object())
        finally:
            lifecycle["closed"] = True

    monkeypatch.setattr(
        weight_updater_module,
        "TorchDistributedWeightStoreCoordinator",
        lambda _group: object(),
        raising=False,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "WeightSnapshotWriteSpec",
        FakeWriteSpec,
        raising=False,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "open_weight_snapshot_write_backend",
        open_backend,
        raising=False,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "materialize_distributed_runtime_weight_snapshot",
        materialize,
        raising=False,
    )
