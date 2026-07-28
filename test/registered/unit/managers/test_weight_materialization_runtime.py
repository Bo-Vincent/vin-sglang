from __future__ import annotations

import os
import sys
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.io_struct import (
    BeginRemoteInstanceWeightTransferReqInput,
    CommitWeightMaterializationReqInput,
    CommitWeightMaterializationReqOutput,
    PrepareWeightMaterializationReqInput,
    PrepareWeightMaterializationReqOutput,
    ReleaseRemoteInstanceWeightTransferReqInput,
    RenewRemoteInstanceWeightTransferReqInput,
    WeightMaterializationSessionState,
)
from sglang.srt.managers.scheduler_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.model_executor import model_runner as model_runner_module
from sglang.srt.model_executor.weight_runtime_manifest import (
    MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC,
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer import runtime as weight_transfer_runtime
from sglang.srt.weight_transfer.distributed import (
    TorchDistributedWeightStoreCoordinator,
)
from sglang.srt.weight_transfer.provider import (
    WeightPayloadIdentity,
    WeightStorageDestination,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
    WeightTransferExecutionContext,
    WeightTransferReleaseError,
    WeightTransferTerminalProof,
    WeightTransferTerminalStatus,
)
from sglang.srt.weight_transfer.runtime import RuntimeWeightSnapshotSource
from sglang.srt.weight_transfer.storage import WeightStorageRef
from sglang.srt.weight_transfer.store_runtime import WeightSnapshotBackendStatus
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _terminal_backend(**attributes):
    return SimpleNamespace(
        seal=lambda: (),
        quiesce=lambda **_kwargs: WeightSnapshotBackendStatus(terminal=True),
        **attributes,
    )


@pytest.fixture(autouse=True)
def _publish_materialized_candidate(monkeypatch):
    monkeypatch.setattr(
        weight_updater_module,
        "publish_weight_snapshot",
        lambda candidate, **_kwargs: SimpleNamespace(snapshot=candidate.snapshot),
    )


class _Source:
    def __init__(
        self,
        *,
        dp_rank: int = 0,
        tp_rank: int | None = None,
        tp_size: int = 1,
        lease_id: str | None = None,
        revision: str = "main",
    ) -> None:
        if tp_rank is not None and not 0 <= tp_rank < tp_size:
            raise ValueError("tp_rank must be inside tp_size")
        rank_suffix = (
            f"dp{dp_rank}"
            if tp_rank is None
            else f"dp{dp_rank}-tp{tp_rank}-of-{tp_size}"
        )
        lease_id = lease_id or f"lease-{rank_suffix}"
        fragment_id = f"fragment-{rank_suffix}"
        global_shape = (8, 8) if tp_rank is None else (8 * tp_size, 8)
        global_offset = (0, 0) if tp_rank is None else (8 * tp_rank, 0)
        tensor = WeightPlacementTensor(
            placement_fragment_id=fragment_id,
            tensor_id="model.layers.0.weight",
            runtime_name="model.layers.0.weight",
            aliases=("model.layers.0.weight",),
            global_shape=global_shape,
            global_offset=global_offset,
            local_shape=(8, 8),
            dtype="float16",
            itemsize=2,
            partition_dim=None if tp_rank is None else 0,
            shard_dims=() if tp_rank is None else (0,),
            layer_id=0,
            expert_id=None,
            layout_fingerprint="dense-row-major",
            nbytes=128,
            byte_offset=0,
            rank=WeightParallelRank(
                dp=dp_rank,
                tp=0 if tp_rank is None else tp_rank,
            ),
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
            instance_id=f"instance-{rank_suffix}",
            generation=7,
            lease_id=lease_id,
            fragments=(
                RuntimeWeightBinding(
                    placement_fragment_id=fragment_id,
                    fragment_id=f"runtime-{fragment_id}",
                    address=(
                        0x1000
                        + dp_rank * 0x100000
                        + (0 if tp_rank is None else tp_rank * 0x1000)
                    ),
                    nbytes=128,
                    storage_offset=0,
                    device="cuda",
                    is_contiguous=True,
                    worker_id=f"worker-{rank_suffix}",
                    endpoint=f"endpoint-{rank_suffix}",
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
        self.provider_name = error.provider
        self.completion_ticket = error.completion_ticket

    def attest(self, request, *, request_binding=None) -> None:
        binding = request_binding or self.binding
        assert binding in request.source_bindings

    def resolve_quarantine(self, proof) -> None:
        assert proof.operation_id == self.operation_id
        assert proof.provider == self.provider_name
        assert proof.completion_ticket == self.completion_ticket
        self.quarantined = False
        self.release()


class _DeferredSource(_Source):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.captured_payload_identity = self.payload_identity
        self.payload_identity = None
        self.hash_calls = 0

    def capture_payload_identity(self, *, execution_context=None):
        assert execution_context is not None
        self.hash_calls += 1
        self.payload_identity = self.captured_payload_identity
        return self.payload_identity


def _manager(source: _Source, *, external_dp_rank: int = 0):
    collective_group = object()
    runner = SimpleNamespace(
        capture_runtime_weight_snapshot_source=lambda **_kwargs: source,
    )
    manager = SchedulerWeightUpdaterManager(
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
    manager.weight_materialization_execution_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 300
    )
    return manager


def _prepare_request(
    *,
    materialization_id: str = "materialize-0",
    revision: str = "main",
    deadline_unix_sec: float | None = None,
):
    return SimpleNamespace(
        materialization_id=materialization_id,
        request_id=f"{materialization_id}-prepare",
        model_id="Qwen/Qwen3.5-0.8B",
        revision=revision,
        lease_timeout_sec=300,
        deadline_unix_sec=deadline_unix_sec,
    )


def _commit_request(
    *,
    materialization_id: str = "materialize-0",
    selected_external_dp_rank: int | None = 0,
    storage_options=None,
    phase: str | None = None,
    request_id: str | None = None,
    deadline_unix_sec: float | None = None,
):
    return SimpleNamespace(
        materialization_id=materialization_id,
        request_id=request_id or f"{materialization_id}-commit",
        selected_external_dp_rank=selected_external_dp_rank,
        storage_options=storage_options or {"catalog_path": "/catalog"},
        phase=phase or ("cleanup" if selected_external_dp_rank is None else "commit"),
        deadline_unix_sec=deadline_unix_sec,
    )


def _install_world(monkeypatch, gathered=None):
    gathered = gathered or ()
    gather_calls = 0
    world_size = 1 + len(gathered)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group: world_size,
    )

    def all_gather_object(outputs, value, group):
        nonlocal gather_calls
        values = [value] * world_size
        gather_calls += 1
        assert len(outputs) == len(values)
        outputs[:] = values

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    class Coordinator:
        rank = 0

        def __init__(self):
            self.world_size = world_size
            self.root_gather_values = []
            self.root_scatter_values = []

        def gather_object_to_root(self, value, **_kwargs):
            self.root_gather_values.append(value)
            return (value, *gathered)

        def all_gather_object(self, value, **_kwargs):
            return [value] * world_size

        def scatter_object_from_root(self, values, **_kwargs):
            self.root_scatter_values.append(values)
            return values[0]

    coordinator = Coordinator()
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_weight_materialization_store_coordinator",
        lambda _self: coordinator,
    )
    return coordinator


def _successful_capture(source: _Source) -> dict:
    return {
        "success": True,
        "message": "Success.",
        "placement": source.placement,
        "binding": source.binding,
        "payload_identity": source.payload_identity,
    }


def test_model_runner_captures_runtime_snapshot_with_finite_lease(monkeypatch) -> None:
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
    execution_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 30,
        cancel_signal=threading.Event(),
    )

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
        lease_timeout_sec=300,
        execution_context=execution_context,
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
        "lease_timeout_sec": 300,
        "execution_context": execution_context,
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
        lease_timeout_sec=300,
    )

    runtime_id = f"sglang:runtime:{os.getpid()}:3"
    assert calls[0]["instance_id"] == f"{runtime_id}:materialize-7"
    assert calls[0]["worker_id"] == runtime_id
    assert calls[0]["endpoint"] == f"local://{runtime_id}"


def test_runtime_wrapper_quarantines_generic_unknown_until_terminal_proof(
    monkeypatch,
) -> None:
    captured = _Source()
    released = []
    manager = SimpleNamespace(
        has_lease=lambda _lease_id: True,
        release=released.append,
    )
    source = RuntimeWeightSnapshotSource(
        model=object(),
        manager=manager,
        parts=SimpleNamespace(
            placement=captured.placement,
            binding=captured.binding,
        ),
        payload_hasher=lambda _location: (
            captured.payload_identity.fragments[0].checksum
        ),
        payload_identity=captured.payload_identity,
    )

    class GenericCompletionUnknown(WeightTransferError):
        @property
        def completion_ticket(self):
            return "ticket-runtime"

    def materialize(**_kwargs):
        raise GenericCompletionUnknown(
            "runtime completion is unknown",
            code="BACKEND_FAILURE",
            provider="mooncake-store",
            phase="wait",
            operation_id="materialize-runtime",
            retryable=False,
            completion_known=False,
            cleanup_required=True,
        )

    monkeypatch.setattr(weight_transfer_runtime, "materialize_weights", materialize)

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
        weight_transfer_runtime.materialize_runtime_weights(
            source,
            destination=WeightStorageDestination(
                provider="mooncake-store",
                storage_id="model/revision",
                object_prefix="model/revision",
            ),
            provider=SimpleNamespace(),
        )

    assert raised.value.completion_ticket == "ticket-runtime"
    assert source.quarantined is True
    assert source.released is False
    assert released == []

    source.resolve_quarantine(
        WeightTransferTerminalProof(
            operation_id="materialize-runtime",
            provider="mooncake-store",
            completion_ticket="ticket-runtime",
            status=WeightTransferTerminalStatus.COMPLETED,
        )
    )

    assert source.quarantined is False
    assert source.released is True
    assert released == [captured.binding.lease_id]


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
    commit = manager.commit_weight_materialization(_commit_request())
    assert commit.success is False
    assert commit.session_state == WeightMaterializationSessionState.NOT_FOUND


def test_prepare_rejects_multi_rank_world_without_isolated_group(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    manager.remote_weight_transfer_cpu_group = None
    manager.weight_materialization_cpu_group = None
    manager.weight_materialization_execution_context = None
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


@pytest.mark.parametrize(
    ("local_dp_rank", "expected_hash_calls"),
    [(0, 1), (1, 0)],
)
def test_prepare_hashes_only_the_selected_internal_dp_replica(
    monkeypatch,
    local_dp_rank,
    expected_hash_calls,
) -> None:
    local_source = _DeferredSource(dp_rank=local_dp_rank)
    peer_source = _DeferredSource(dp_rank=1 - local_dp_rank)
    capture_kwargs = []
    runner = SimpleNamespace(
        capture_runtime_weight_snapshot_source=lambda **kwargs: (
            capture_kwargs.append(kwargs) or local_source
        ),
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
    manager.weight_materialization_execution_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 300
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)

    class Coordinator:
        rank = 0
        world_size = 2

        def __init__(self) -> None:
            self.gather_calls = 0
            self.scatter_calls = 0
            self.gathered_values = []

        def all_gather_object(self, value, **_kwargs):
            return [value, value]

        def gather_object_to_root(self, value, **_kwargs):
            self.gather_calls += 1
            self.gathered_values.append(dict(value))
            if self.gather_calls == 1:
                remote = {
                    "success": True,
                    "message": "Success.",
                    "placement": peer_source.placement,
                    "binding": peer_source.binding,
                    "payload_identity": None,
                }
            else:
                remote = {
                    "success": True,
                    "message": "Success.",
                    "payload_identity": (
                        peer_source.captured_payload_identity
                        if peer_source.placement.tensors[0].rank.dp == 0
                        else None
                    ),
                }
            return (value, remote)

        def scatter_object_from_root(self, values, **_kwargs):
            self.scatter_calls += 1
            return values[0]

    coordinator = Coordinator()
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_weight_materialization_store_coordinator",
        lambda _self: coordinator,
    )

    result = manager.prepare_weight_materialization(_prepare_request())

    assert result.success is True, result.message
    assert capture_kwargs[0]["defer_payload_identity"] is True
    assert local_source.hash_calls == expected_hash_calls
    assert coordinator.gather_calls == 2
    assert coordinator.scatter_calls == 2
    assert set(coordinator.gathered_values[0]) == {
        "success",
        "message",
        "placement",
        "binding",
        "payload_identity",
    }
    assert coordinator.gathered_values[0]["payload_identity"] is None
    assert set(coordinator.gathered_values[1]) == {
        "success",
        "message",
        "payload_identity",
    }
    session = manager.weight_materialization_sessions["materialize-0"]
    assert bool(session.local_selected_placement_ids) is (local_dp_rank == 0)


def test_prepare_root_scatter_keeps_nonroot_source_state_linear(monkeypatch) -> None:
    world_size = 8
    sources = tuple(
        _Source(tp_rank=rank, tp_size=world_size) for rank in range(world_size)
    )
    manager = _manager(sources[0])
    coordinator = _install_world(
        monkeypatch,
        gathered=tuple(_successful_capture(source) for source in sources[1:]),
    )

    result = manager.prepare_weight_materialization(_prepare_request())

    assert result.success is True, result.message
    assert len(coordinator.root_gather_values) == 2
    assert len(coordinator.root_scatter_values) == 2
    packets = coordinator.root_scatter_values[-1]
    assert len(packets) == world_size
    root_selection = packets[0]["selection"]
    assert len(root_selection.placements) == world_size
    summaries = {packet["selection"].summary for packet in packets}
    assert len(summaries) == 1
    nonroot_record_count = 0
    for packet in packets[1:]:
        selection = packet["selection"]
        assert len(selection.placements) == 1
        assert len(selection.bindings) == 1
        assert len(selection.payload_identity.fragments) == 1
        nonroot_record_count += len(selection.placements)
    assert len(root_selection.placements) + nonroot_record_count <= 2 * world_size - 1
    session = manager.weight_materialization_sessions["materialize-0"]
    assert len(session.selected_placements) == world_size


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
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_materialization_deadline_expired_world",
        lambda _self, _deadline: (False, None),
    )
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

    def materialize_without_local_source(request, **kwargs):
        calls.append((request, kwargs))
        assert local_source.released is False
        assert all(
            placement.placement_id != local_source.placement.placement_id
            for placement in request.source_placements
        )
        return SimpleNamespace(snapshot=SimpleNamespace(ref=expected_ref))

    monkeypatch.setattr(
        weight_updater_module,
        "materialize_weight_snapshot_candidate",
        materialize_without_local_source,
    )

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is True, result.message
    assert result.selected is True
    assert len(calls) == 1
    assert lifecycle["closed"] is True
    manager.commit_weight_materialization(
        _commit_request(selected_external_dp_rank=None)
    )
    assert lifecycle["closed"] is True
    manager.close_remote_instance_weight_transfer_executor()
    assert lifecycle["closed"] is True


def test_selected_external_dp_uses_distributed_root_catalog_backend(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    store_coordinator_factory = (
        SchedulerWeightUpdaterManager._weight_materialization_store_coordinator
    )
    group = object()
    manager.remote_weight_transfer_cpu_group = group
    manager.weight_materialization_cpu_group = group
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_weight_materialization_store_coordinator",
        store_coordinator_factory,
    )
    manager.weight_materialization_coordinator = None
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

    backend = _terminal_backend(
        provider=SimpleNamespace(name="mooncake-store"),
        catalog=object(),
    )

    @contextmanager
    def open_backend(
        spec,
        *,
        local_placement_ids,
        payload_checksum_verifier,
        coordinator,
        execution_context=None,
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

    def materialize(request, **kwargs):
        calls["materialize_count"] = calls.get("materialize_count", 0) + 1
        calls["materialize"] = (request, kwargs)
        assert source.released is False
        assert request.operation_id == "materialize-0"
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
        "preflight_weight_transfer",
        lambda _provider, request, *, attestor: attestor.attest(request) or object(),
        raising=False,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "materialize_weight_snapshot_candidate",
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
    assert calls["materialize"][0].operation_id == "materialize-0"
    assert source.released is True
    assert calls["closed"] is True
    session = manager.weight_materialization_sessions["materialize-0"]
    assert session.source is None
    assert session.backend_owner is None
    assert session.backend is None

    replay = manager.commit_weight_materialization(_commit_request())
    conflict = manager.commit_weight_materialization(
        _commit_request(storage_options={"catalog_path": "/other"})
    )
    assert replay == result
    assert calls["materialize_count"] == 1
    assert conflict.success is False
    assert conflict.session_state == "conflict"
    assert calls["closed"] is True

    cleanup = manager.commit_weight_materialization(
        _commit_request(selected_external_dp_rank=None)
    )
    assert cleanup.success is True
    assert cleanup.session_state == "published"
    assert cleanup.ref == result.ref
    assert calls["closed"] is True
    manager.close_remote_instance_weight_transfer_executor()
    assert calls["closed"] is True


def test_materialization_publishes_only_after_world_outcome_consensus(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    events = []
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    events.clear()

    def gather(_self, value, *, operation):
        events.append(operation)
        return [value]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    expected_ref = WeightStorageRef(
        provider="mooncake-store",
        storage_id="model/revision",
        manifest_key="manifest",
        manifest_digest=f"sha256:{'b' * 64}",
    )

    def materialize(*_args, **_kwargs):
        events.append("candidate")
        return SimpleNamespace(snapshot=SimpleNamespace(ref=expected_ref))

    _install_fake_selected_backend(monkeypatch, materialize=materialize)

    def publish(candidate, **_kwargs):
        events.append("publish")
        return SimpleNamespace(snapshot=candidate.snapshot)

    monkeypatch.setattr(weight_updater_module, "publish_weight_snapshot", publish)

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is True
    assert events == [
        "commit request state",
        "commit deadline vote",
        "unresolved materialization capacity",
        "Store backend setup",
        "Store materialization preflight",
        "candidate",
        "Store materialization outcome",
        "publish",
        "post-publication backend close ownership",
        "post-publication backend close readiness",
        "post-publication source resolution",
        "post-publication backend close status",
    ]


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
            yield _terminal_backend(
                provider=SimpleNamespace(name="mooncake-store"),
                catalog=object(),
            )
        finally:
            lifecycle["closes"] += 1

    def materialize(request, **_kwargs):
        lifecycle["materializations"] += 1
        materialization_id = request.operation_id
        assert sources[materialization_id].released is False
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
        "preflight_weight_transfer",
        lambda _provider, request, *, attestor: attestor.attest(request) or object(),
    )
    monkeypatch.setattr(
        weight_updater_module,
        "materialize_weight_snapshot_candidate",
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

    evicted = manager.commit_weight_materialization(
        _commit_request(materialization_id="materialize-0")
    )
    assert evicted.success is False
    assert evicted.session_state == WeightMaterializationSessionState.NOT_FOUND
    assert "was not prepared" in evicted.message
    assert lifecycle == {"opens": 3, "closes": 3, "materializations": 3}
    replay = manager.commit_weight_materialization(
        _commit_request(materialization_id="materialize-2")
    )
    assert replay == outputs["materialize-2"]
    assert lifecycle["materializations"] == 3
    manager.close_remote_instance_weight_transfer_executor()
    assert lifecycle["closes"] == 3


def test_completion_unknown_session_counts_toward_unresolved_limit(
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
    monkeypatch.setattr(
        weight_updater_module,
        "_WEIGHT_MATERIALIZATION_UNRESOLVED_LIMIT",
        1,
    )
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
    assert "unresolved weight materialization limit reached" in second.message
    assert materialization_calls == ["materialize-first"]
    assert first_source.quarantined is True
    assert second_source.released is True


def test_exit_stack_close_failure_is_sticky_and_observable(monkeypatch) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    session = manager.weight_materialization_sessions["materialize-0"]

    close_calls = 0

    @contextmanager
    def failing_backend():
        nonlocal close_calls
        yield _terminal_backend()
        close_calls += 1
        raise RuntimeError("close failed permanently")

    owner = weight_updater_module._WeightStorageBackendOwner()
    owner.enter_context(failing_backend())
    session.backend_owner = owner

    assert manager._close_materialization_backend(session) == "close failed permanently"
    assert session.backend_owner is owner
    assert owner.terminal_error == "close failed permanently"
    assert manager._close_materialization_backend(session) == "close failed permanently"
    assert close_calls == 1


def test_published_cleanup_failure_is_not_upgraded_on_retry(monkeypatch) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    session = manager.weight_materialization_sessions["materialize-0"]
    ref = {"provider": "mooncake-store", "storage_id": "model/revision"}
    session.publication_ref = dict(ref)
    session.commit_output = CommitWeightMaterializationReqOutput(
        materialization_id="materialize-0",
        request_id="materialize-0-commit",
        success=True,
        message="Success.",
        external_dp_rank=0,
        selected=True,
        ref=dict(ref),
        session_state="published",
    )
    session.state = "published"

    close_calls = 0

    @contextmanager
    def failing_backend():
        nonlocal close_calls
        yield _terminal_backend()
        close_calls += 1
        raise RuntimeError("backend close has no retry path")

    owner = weight_updater_module._WeightStorageBackendOwner()
    session.backend = owner.enter_context(failing_backend())
    session.backend_owner = owner
    cleanup_request = _commit_request(selected_external_dp_rank=None)

    first = manager._cleanup_weight_materialization_session(
        cleanup_request,
        session,
    )
    second = manager._cleanup_weight_materialization_session(
        cleanup_request,
        session,
    )

    assert first.success is False
    assert second.success is False
    assert first.session_state == "published_cleanup_failed"
    assert second.session_state == "published_cleanup_failed"
    assert "backend close has no retry path" in first.message
    assert second.message == first.message
    assert session.backend_owner is owner
    assert session.backend is not None
    assert owner.terminal_error == "backend close has no retry path"
    assert close_calls == 1


def test_published_cleanup_retries_completion_unknown_store_close(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    session = manager.weight_materialization_sessions["materialize-0"]
    ref = {"provider": "mooncake-store", "storage_id": "model/revision"}
    session.publication_ref = dict(ref)
    session.commit_output = CommitWeightMaterializationReqOutput(
        materialization_id="materialize-0",
        request_id="materialize-0-commit",
        success=True,
        message="Success.",
        external_dp_rank=0,
        selected=True,
        ref=dict(ref),
        session_state="published",
    )
    session.state = "published"

    class Backend:
        def __init__(self):
            self.close_complete = False
            self.close_timeouts = []

        def seal(self):
            return ()

        def quiesce(self, *, timeout_ms):
            assert timeout_ms >= 0
            return WeightSnapshotBackendStatus(terminal=True)

        def close(self, *, timeout_ms):
            self.close_timeouts.append(timeout_ms)
            if not self.close_complete:
                return WeightSnapshotBackendStatus(
                    terminal=False,
                    pending_tickets=("mooncake-store/close",),
                )
            return WeightSnapshotBackendStatus(terminal=True, closed=True)

    backend = Backend()

    @contextmanager
    def closing_backend():
        yield backend
        status = backend.close(timeout_ms=0)
        if not status.closed:
            error = RuntimeError("Mooncake Store close is still pending")
            error.completion_unknown = True
            raise error

    owner = weight_updater_module._WeightStorageBackendOwner()
    session.backend = owner.enter_context(closing_backend())
    session.backend_owner = owner

    first = manager._cleanup_weight_materialization_session(
        _commit_request(
            request_id="close-timeout",
            deadline_unix_sec=time.time() + 0.05,
        ),
        session,
    )

    assert first.success is False
    assert first.completion_unknown is True
    assert first.session_state == "published_cleanup_pending"
    assert source.released is True
    assert session.backend_owner is owner
    assert session.backend is backend
    assert owner.terminal_error is None

    backend.close_complete = True
    second = manager.commit_weight_materialization(
        _commit_request(
            request_id="close-retry",
            deadline_unix_sec=time.time() + 1,
        )
    )

    assert second.success is True
    assert second.completion_unknown is False
    assert second.session_state == "published"
    assert second.ref == ref
    assert session.backend_owner is None
    assert session.backend is None
    assert owner.closed is True
    assert backend.close_timeouts[0] == 0
    assert 0 < backend.close_timeouts[-1] <= 1000


def test_cleanup_drains_native_calls_before_closing_backend(monkeypatch) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    session = manager.weight_materialization_sessions["materialize-0"]
    ref = {"provider": "mooncake-store", "storage_id": "model/revision"}
    session.publication_ref = dict(ref)
    session.commit_output = CommitWeightMaterializationReqOutput(
        materialization_id="materialize-0",
        request_id="materialize-0-commit",
        success=True,
        message="Published; cleanup remains pending.",
        external_dp_rank=0,
        selected=True,
        ref=dict(ref),
        session_state="published_cleanup_pending",
    )
    session.state = "published_cleanup_pending"

    class Backend:
        def __init__(self):
            self.pending = True
            self.complete_on_drain = False
            self.drain_timeouts = []
            self.sealed = False

        def pending_tickets(self):
            if not self.pending:
                return ()
            return ("materialize-0/upload",)

        def seal(self):
            self.sealed = True
            return self.pending_tickets()

        def quiesce(self, *, timeout_ms):
            self.drain_timeouts.append(timeout_ms)
            if self.complete_on_drain and timeout_ms > 0:
                self.pending = False
            pending = self.pending_tickets()
            return WeightSnapshotBackendStatus(
                terminal=not pending,
                pending_tickets=pending,
            )

    class Owner:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    backend = Backend()
    owner = Owner()
    session.backend = backend
    session.backend_owner = owner

    first = manager._cleanup_weight_materialization_session(
        _commit_request(
            selected_external_dp_rank=None,
            request_id="pending-native-cleanup",
            deadline_unix_sec=time.time() + 0.05,
        ),
        session,
    )

    assert first.success is False
    assert first.completion_unknown is True
    assert first.session_state == "published_cleanup_pending"
    assert "pending calls" in first.message
    assert owner.close_calls == 0
    assert source.released is False
    assert session.backend_owner is owner
    assert session.backend is not None
    assert backend.sealed is True
    assert len(backend.drain_timeouts) == 1
    assert 0 <= backend.drain_timeouts[0] <= 50

    backend.complete_on_drain = True
    session.deadline_unix_sec = time.time() - 1
    second = manager._cleanup_weight_materialization_session(
        _commit_request(
            selected_external_dp_rank=None,
            request_id="terminal-native-cleanup",
            deadline_unix_sec=time.time() + 1,
        ),
        session,
    )

    assert second.success is True
    assert second.completion_unknown is False
    assert second.session_state == "published"
    assert owner.close_calls == 1
    assert source.released is True
    assert session.backend_owner is None
    assert session.backend is None


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


def test_release_failure_preserves_publication_for_same_id_finalize(
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
    calls = []

    def materialize(source_arg, **_kwargs):
        calls.append("finalize" if calls else "publish")
        publication = SimpleNamespace(snapshot=SimpleNamespace(ref=expected_ref))
        if len(calls) == 1:
            error = WeightTransferReleaseError(
                "provider release failed after publication",
                receipt=None,
                provider="mooncake-store",
                operation_id="materialize-0",
            )
            error.publication = publication
            raise error
        return publication

    _install_fake_selected_backend(
        monkeypatch,
        materialize=materialize,
        lifecycle=lifecycle,
    )

    first = manager.commit_weight_materialization(
        _commit_request(request_id="publish-request")
    )

    assert first.success is False
    assert first.session_state == "finalize_pending"
    assert first.ref is None
    assert "provider release failed after publication" in first.message
    assert source.released is False
    assert lifecycle.get("closed") is not True

    cleanup = manager.commit_weight_materialization(
        _commit_request(
            selected_external_dp_rank=None,
            request_id="cleanup-request",
        )
    )

    assert cleanup.success is False
    assert cleanup.session_state == "finalize_pending"
    assert cleanup.ref is None
    assert source.released is False
    assert lifecycle.get("closed") is not True

    second = manager.commit_weight_materialization(
        _commit_request(
            request_id="finalize-request",
            phase="recover",
        )
    )

    assert second.success is True
    assert second.session_state == "published"
    assert second.ref == manager._weight_storage_ref_builtins(expected_ref)
    assert calls == ["publish", "finalize"]
    assert source.released is True
    assert lifecycle["closed"] is True


def test_commit_rejects_partial_session_ownership_before_mutation(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success

    def gather(_self, value, *, operation):
        assert operation == "commit request state"
        peer = dict(value)
        peer.update(
            present=False,
            request_identity=None,
            commit_identity=None,
            state=None,
            has_commit_output=False,
            publication_ref=None,
        )
        return [value, peer]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is False
    assert result.completion_unknown is True
    assert result.session_state == "completion_unknown"
    assert source.released is False


def test_commit_rejects_divergent_recovery_state_before_mutation(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success

    def gather(_self, value, *, operation):
        assert operation == "commit request state"
        peer = dict(value)
        peer["state"] = "completion_unknown"
        return [value, peer]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is False
    assert result.completion_unknown is True
    assert result.session_state == "completion_unknown"
    assert source.released is False


def test_post_publication_peer_release_failure_closes_store_backend(
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
    materialize_calls = 0

    def materialize(source_arg, **kwargs):
        nonlocal materialize_calls
        materialize_calls += 1
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
            "commit request state",
            "commit deadline vote",
            "unresolved materialization capacity",
            "Store backend setup",
            "Store materialization preflight",
            "Store materialization outcome",
            "cleanup source release",
            "cleanup Store backend close ownership",
            "cleanup Store backend close readiness",
            "cleanup Store backend close status",
            "post-publication backend close ownership",
            "post-publication backend close readiness",
            "post-publication backend close status",
        }:
            return [value, value]
        assert operation == "post-publication source resolution"
        return [
            value,
            {
                "error": "peer source release failed",
            },
        ]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is True
    assert result.session_state == "published_cleanup_pending"
    assert "peer source release failed" in result.message
    assert result.ref == manager._weight_storage_ref_builtins(expected_ref)
    assert source.released is True
    session = manager.weight_materialization_sessions["materialize-0"]
    assert session.backend_owner is None
    assert session.backend is None
    assert lifecycle["closed"] is True

    replay = manager.commit_weight_materialization(_commit_request())
    assert replay.success is True
    assert replay.session_state == "published"
    assert replay.ref == result.ref
    assert materialize_calls == 1

    cleanup = manager.commit_weight_materialization(
        _commit_request(selected_external_dp_rank=None)
    )
    assert cleanup.success is True
    assert cleanup.session_state == "published"
    assert cleanup.ref == result.ref
    assert lifecycle["closed"] is True

    final_replay = manager.commit_weight_materialization(_commit_request())
    assert final_replay.success is True
    assert final_replay.session_state == "published"
    assert final_replay.ref == result.ref
    assert materialize_calls == 1


def test_retryable_cleanup_failure_is_not_pruned_as_terminal(monkeypatch) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    expected_ref = WeightStorageRef(
        provider="mooncake-store",
        storage_id="model/revision",
        manifest_key="manifest",
        manifest_digest=f"sha256:{'b' * 64}",
    )
    _install_fake_selected_backend(
        monkeypatch,
        materialize=lambda *_args, **_kwargs: SimpleNamespace(
            snapshot=SimpleNamespace(ref=expected_ref)
        ),
    )
    published = manager.commit_weight_materialization(_commit_request())
    assert published.session_state == "published"

    fail_cleanup_once = True

    def gather(_self, value, *, operation):
        nonlocal fail_cleanup_once
        if operation == "cleanup Store backend close ownership" and fail_cleanup_once:
            fail_cleanup_once = False
            raise RuntimeError("cleanup consensus unavailable")
        return [value]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    first = manager.commit_weight_materialization(
        _commit_request(
            selected_external_dp_rank=None,
            request_id="cleanup-failed",
        )
    )
    monkeypatch.setattr(
        weight_updater_module,
        "_WEIGHT_MATERIALIZATION_TERMINAL_LIMIT",
        0,
    )
    second = manager.commit_weight_materialization(
        _commit_request(
            selected_external_dp_rank=None,
            request_id="cleanup-retry",
        )
    )

    assert first.success is False
    assert first.completion_unknown is True
    assert first.session_state == "published_cleanup_pending"
    assert second.success is True
    assert second.session_state == "published"
    assert second.ref == published.ref


def test_selected_completion_unknown_recovers_with_same_materialization_id(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    calls = []
    lifecycle = {}
    expected_ref = WeightStorageRef(
        provider="mooncake-store",
        storage_id="model/revision",
        manifest_key="manifest",
        manifest_digest=f"sha256:{'b' * 64}",
    )

    def materialize(source_arg, **_kwargs):
        calls.append("submit")
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

    def transfer(request, **_kwargs):
        if not source.quarantined:
            return materialize(
                source,
                publication_id=request.operation_id,
            )
        calls.append("recover")
        return SimpleNamespace(snapshot=SimpleNamespace(ref=expected_ref))

    monkeypatch.setattr(
        weight_updater_module,
        "materialize_weight_snapshot_candidate",
        transfer,
    )
    request = _commit_request()

    first = manager.commit_weight_materialization(request)
    cleanup = manager.commit_weight_materialization(
        _commit_request(selected_external_dp_rank=None)
    )
    second = manager.commit_weight_materialization(request)

    assert first.success is False
    assert first.session_state == "completion_unknown"
    assert first.completion_ticket == "ticket-0"
    assert first.ref is None
    assert cleanup.completion_unknown is True
    assert cleanup.phase == "cleanup"
    assert second.success is True
    assert second.ref == {
        "provider": "mooncake-store",
        "storage_id": "model/revision",
        "manifest_key": "manifest",
        "manifest_digest": f"sha256:{'b' * 64}",
    }
    assert calls == ["submit", "recover"]
    assert source.quarantined is False
    assert source.released is True
    assert lifecycle["closed"] is True


def test_base_completion_unknown_quarantines_then_recovers_source(monkeypatch) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    lifecycle = {}
    calls = []
    expected_ref = WeightStorageRef(
        provider="mooncake-store",
        storage_id="model/revision",
        manifest_key="manifest",
        manifest_digest=f"sha256:{'b' * 64}",
    )

    class GenericCompletionUnknown(WeightTransferError):
        @property
        def completion_ticket(self):
            return "ticket-0"

    def materialize(source_arg, **_kwargs):
        if source_arg.quarantined:
            calls.append("recover")
            return SimpleNamespace(snapshot=SimpleNamespace(ref=expected_ref))
        calls.append("submit")
        raise GenericCompletionUnknown(
            "provider outcome is unknown",
            code="BACKEND_FAILURE",
            provider="mooncake-store",
            phase="wait",
            operation_id="materialize-0",
            retryable=False,
            completion_known=False,
            cleanup_required=True,
        )

    _install_fake_selected_backend(
        monkeypatch,
        materialize=materialize,
        lifecycle=lifecycle,
    )

    request = _commit_request()
    first = manager.commit_weight_materialization(request)
    cleanup = manager.commit_weight_materialization(
        _commit_request(selected_external_dp_rank=None)
    )
    second = manager.commit_weight_materialization(request)

    assert first.success is False
    assert first.session_state == "completion_unknown"
    assert first.completion_unknown is True
    assert first.completion_ticket == "ticket-0"
    assert cleanup.completion_unknown is True
    assert second.success is True
    assert second.session_state == "published"
    assert source.quarantined is False
    assert source.released is True
    assert lifecycle["closed"] is True
    assert calls == ["submit", "recover"]


def test_materialization_outcome_gather_failure_retains_source_for_retry(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    lifecycle = {}
    calls = []
    expected_ref = WeightStorageRef(
        provider="mooncake-store",
        storage_id="model/revision",
        manifest_key="manifest",
        manifest_digest=f"sha256:{'b' * 64}",
    )

    def materialize(source_arg, **_kwargs):
        calls.append("materialize")
        assert source_arg.released is False
        return SimpleNamespace(snapshot=SimpleNamespace(ref=expected_ref))

    _install_fake_selected_backend(
        monkeypatch,
        materialize=materialize,
        lifecycle=lifecycle,
    )
    published = []
    monkeypatch.setattr(
        weight_updater_module,
        "publish_weight_snapshot",
        lambda candidate, **_kwargs: (
            published.append(candidate) or SimpleNamespace(snapshot=candidate.snapshot)
        ),
    )
    outcome_calls = 0

    def gather(_self, value, *, operation):
        nonlocal outcome_calls
        if operation == "commit request state":
            return [value]
        if operation == "Store materialization outcome":
            outcome_calls += 1
            if outcome_calls == 1:
                raise RuntimeError("outcome collective failed")
        return [value]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )
    request = _commit_request()

    first = manager.commit_weight_materialization(request)

    assert first.success is False
    assert first.session_state == "completion_unknown"
    assert first.completion_ticket is None
    assert source.released is False
    assert published == []

    second = manager.commit_weight_materialization(request)

    assert second.success is True
    assert second.session_state == "published"
    assert source.released is True
    assert lifecycle["closed"] is True
    assert calls == ["materialize", "materialize"]
    assert len(published) == 1


def test_divergent_materialized_candidates_are_never_published(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    expected_ref = WeightStorageRef(
        provider="mooncake-store",
        storage_id="model/revision",
        manifest_key="manifest",
        manifest_digest=f"sha256:{'b' * 64}",
    )

    _install_fake_selected_backend(
        monkeypatch,
        materialize=lambda *_args, **_kwargs: SimpleNamespace(
            snapshot=SimpleNamespace(ref=expected_ref)
        ),
    )
    monkeypatch.setattr(
        weight_updater_module,
        "publish_weight_snapshot",
        lambda *_args, **_kwargs: pytest.fail(
            "divergent candidates must not be published"
        ),
    )

    def gather(_self, value, *, operation):
        if operation != "Store materialization outcome":
            return [value, value]
        peer = dict(value)
        peer["snapshot_digest"] = f"sha256:{'c' * 64}"
        return [value, peer]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    result = manager.commit_weight_materialization(_commit_request())

    assert result.success is False
    assert result.completion_unknown is True
    assert result.session_state == "completion_unknown"
    assert "different snapshot digests" in result.message
    assert source.released is False


def test_materialization_rejects_divergent_recovery_tickets(monkeypatch) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    lifecycle = {}
    calls = []
    expected_ref = WeightStorageRef(
        provider="mooncake-store",
        storage_id="model/revision",
        manifest_key="manifest",
        manifest_digest=f"sha256:{'b' * 64}",
    )

    def materialize(source_arg, **_kwargs):
        if not calls:
            calls.append("submit")
            error = WeightTransferCompletionUnknownError(
                "commit outcome is unknown",
                provider="mooncake-store",
                phase="commit",
                operation_id="materialize-0",
                completion_ticket="ticket-0",
            )
            source_arg.quarantine(error)
            raise error
        calls.append("recover")
        return SimpleNamespace(snapshot=SimpleNamespace(ref=expected_ref))

    _install_fake_selected_backend(
        monkeypatch,
        materialize=materialize,
        lifecycle=lifecycle,
    )
    outcome_calls = 0

    def gather(_self, value, *, operation):
        nonlocal outcome_calls
        if operation == "commit request state":
            return [value, value]
        if operation != "Store materialization outcome":
            return [value, value]
        outcome_calls += 1
        if outcome_calls == 1:
            peer = dict(value)
            peer["completion_ticket"] = "ticket-1"
            return [value, peer]
        return [value, value]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )
    request = _commit_request()

    first = manager.commit_weight_materialization(request)

    assert first.success is False
    assert first.session_state == "completion_unknown"
    assert first.completion_ticket is None
    assert "different materialization recovery tickets" in first.message
    assert source.quarantined is True

    second = manager.commit_weight_materialization(request)

    assert second.success is True
    assert source.quarantined is False
    assert source.released is True
    assert lifecycle["closed"] is True
    assert calls == ["submit", "recover"]


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


def test_prepare_passes_active_context_to_runtime_snapshot_capture(
    monkeypatch,
) -> None:
    source = _Source()
    captures = []
    manager = _manager(source)
    manager.tp_worker.model_runner.capture_runtime_weight_snapshot_source = (
        lambda **kwargs: captures.append(kwargs) or source
    )
    _install_world(monkeypatch)
    execution_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 30,
        cancel_signal=threading.Event(),
    )
    manager.weight_materialization_execution_context = execution_context

    result = manager.prepare_weight_materialization(_prepare_request())

    assert result.success is True
    assert captures[0]["execution_context"] is execution_context


def test_prepare_fails_all_ranks_when_peer_merge_fails(monkeypatch) -> None:
    source = _Source(dp_rank=0)
    peer = _Source(dp_rank=1)
    manager = _manager(source)
    _install_world(
        monkeypatch,
        gathered=(_successful_capture(peer),),
    )

    def gather(_self, value, *, operation):
        if operation == "prepare session vote":
            return [value, value]
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
    _install_world(
        monkeypatch,
        gathered=(_successful_capture(peer),),
    )

    def gather(_self, value, *, operation):
        if operation == "prepare session vote":
            return [value, value]
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
        request_id="prepare-failure-request",
        model_id="model",
        revision="revision",
        lease_timeout_sec=300,
    )
    commit = CommitWeightMaterializationReqInput(
        materialization_id="commit-failure",
        request_id="commit-failure-request",
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


@pytest.mark.parametrize("isolated_control_group", [False, True])
def test_blocked_begin_does_not_starve_remote_transfer_control(
    monkeypatch,
    isolated_control_group,
) -> None:
    begin_started = threading.Event()
    finish_begin = threading.Event()
    renewed = threading.Event()
    released = threading.Event()
    snapshot_group = object()
    control_group = object() if isolated_control_group else snapshot_group

    def renew_manifest(lease_id, *, lease_timeout_sec):
        assert lease_id == "active-lease"
        assert lease_timeout_sec == 300
        renewed.set()

    def release_manifest(lease_id):
        assert lease_id == "active-lease"
        released.set()

    runner = SimpleNamespace(
        renew_weight_runtime_manifest=renew_manifest,
        release_weight_runtime_manifest=release_manifest,
    )
    manager = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(model_runner=runner),
        draft_worker=None,
        tp_cpu_group=object(),
        world_cpu_group=object(),
        memory_saver_adapter=object(),
        flush_cache=lambda **_kwargs: True,
        is_fully_idle=lambda: True,
        remote_weight_transfer_cpu_group=snapshot_group,
        remote_weight_transfer_control_cpu_group=control_group,
        weight_materialization_cpu_group=object(),
    )
    manager._record_remote_weight_transfer_lease(
        "active-transfer",
        "active-lease",
        300,
    )

    def blocking_begin(_self, request):
        begin_started.set()
        assert finish_begin.wait(timeout=5)
        return SimpleNamespace(
            transfer_id=request.transfer_id,
            success=True,
            message="Success.",
        )

    def all_gather_object(outputs, value, *, group):
        assert group is control_group
        outputs[0] = value

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "begin_remote_instance_weight_transfer",
        blocking_begin,
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda *, group: 1,
    )
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    manager.defer_begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="blocked-begin",
            model_id="model",
            revision="revision",
        )
    )
    assert begin_started.wait(timeout=1)

    manager.defer_renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(transfer_id="active-transfer")
    )
    manager.defer_release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="active-transfer")
    )

    try:
        assert renewed.wait(timeout=1)
        assert released.wait(timeout=1)
        assert "active-transfer" not in manager.remote_weight_transfer_leases
    finally:
        finish_begin.set()
        manager.close_remote_instance_weight_transfer_executor()


def test_release_cannot_race_an_inflight_begin(monkeypatch) -> None:
    begin_started = threading.Event()
    finish_begin = threading.Event()
    control_group = object()
    manager = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(model_runner=SimpleNamespace()),
        draft_worker=None,
        tp_cpu_group=object(),
        world_cpu_group=object(),
        memory_saver_adapter=object(),
        flush_cache=lambda **_kwargs: True,
        is_fully_idle=lambda: True,
        remote_weight_transfer_cpu_group=object(),
        remote_weight_transfer_control_cpu_group=control_group,
        weight_materialization_cpu_group=object(),
    )

    def blocking_begin(_self, request):
        begin_started.set()
        assert finish_begin.wait(timeout=5)
        return SimpleNamespace(
            transfer_id=request.transfer_id,
            success=True,
            message="Success.",
        )

    def all_gather_object(outputs, value, *, group):
        assert group is control_group
        outputs[0] = value

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "begin_remote_instance_weight_transfer",
        blocking_begin,
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda *, group: 1,
    )
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    manager.defer_begin_remote_instance_weight_transfer(
        BeginRemoteInstanceWeightTransferReqInput(
            transfer_id="same-transfer",
            model_id="model",
            revision="revision",
        )
    )
    assert begin_started.wait(timeout=1)
    manager.defer_release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="same-transfer")
    )

    try:
        output = manager.remote_weight_transfer_pending[-1][0].result(timeout=1)
        assert output.success is False
        assert "begin is still in progress" in output.message
        assert "same-transfer" not in manager.remote_weight_transfer_tombstones
    finally:
        finish_begin.set()
        manager.close_remote_instance_weight_transfer_executor()


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
            request_id=request.request_id,
            success=True,
            message="Success.",
            external_dp_rank=0,
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
    monkeypatch.setattr(
        torch.distributed,
        "monitored_barrier",
        lambda *, group, timeout, wait_all_ranks: None,
    )
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        lambda _self, value, *, operation: [value],
    )

    manager.defer_commit_weight_materialization(
        CommitWeightMaterializationReqInput(
            materialization_id="blocked-commit",
            request_id="blocked-commit-request",
            selected_external_dp_rank=0,
            storage_options={},
            deadline_unix_sec=time.time() + 30,
        )
    )
    assert entered_materialization.wait(timeout=1)
    manager.defer_renew_remote_instance_weight_transfer(
        SimpleNamespace(transfer_id="renew-while-store-is-blocked")
    )

    assert renewed.wait(timeout=1)
    release_materialization.set()
    manager.close_remote_instance_weight_transfer_executor()


def test_rank_local_terminal_condition_is_voted_before_collective_barrier(
    monkeypatch,
) -> None:
    manager = _manager(_Source())
    events = []
    barrier_calls = []
    vote_calls = []
    cancel_signal = threading.Event()
    cancel_signal.set()

    monkeypatch.setattr(
        torch.distributed,
        "monitored_barrier",
        lambda *, group, timeout, wait_all_ranks: (
            events.append("barrier"),
            barrier_calls.append((group, timeout, wait_all_ranks)),
        ),
    )

    def gather(_self, value, *, operation):
        events.append("vote")
        vote_calls.append((value, operation))
        return [
            value,
            {
                "cancelled": False,
                "deadline_expired": False,
                "error": None,
                "poisoned": None,
            },
        ]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    with pytest.raises(RuntimeError, match="cancelled"):
        manager._run_weight_materialization(
            lambda _request: pytest.fail("terminal request must not execute"),
            _commit_request(deadline_unix_sec=time.time() + 5),
            cancel_signal,
        )

    assert events == ["vote"]
    assert barrier_calls == []
    assert len(vote_calls) == 1
    assert vote_calls[0][1] == "materialization admission vote"
    assert vote_calls[0][0] == {
        "cancelled": True,
        "deadline_expired": False,
        "error": None,
        "poisoned": None,
    }
    assert manager.weight_materialization_poisoned is None


@pytest.mark.parametrize(
    ("deadline_unix_sec", "message"),
    [
        (None, "deadline is required"),
        (time.time() - 1, "deadline does not leave enough time"),
    ],
)
def test_materialization_admission_votes_unusable_deadline_before_barrier(
    monkeypatch,
    deadline_unix_sec,
    message,
) -> None:
    manager = _manager(_Source())
    barrier_calls = []
    vote_calls = []

    monkeypatch.setattr(
        torch.distributed,
        "monitored_barrier",
        lambda **kwargs: barrier_calls.append(kwargs),
    )

    def gather(_self, value, *, operation):
        control_context = manager.weight_materialization_execution_context
        assert control_context is not None
        assert control_context.deadline_unix_sec > time.time()
        vote_calls.append((value, operation))
        return [
            value,
            {
                "cancelled": False,
                "deadline_expired": False,
                "error": None,
                "poisoned": None,
            },
        ]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    with pytest.raises(RuntimeError, match=message):
        manager._run_weight_materialization(
            lambda _request: pytest.fail("materialization operation must not execute"),
            _commit_request(deadline_unix_sec=deadline_unix_sec),
            threading.Event(),
        )

    assert barrier_calls == []
    assert len(vote_calls) == 1
    assert vote_calls[0][1] == "materialization admission vote"
    assert vote_calls[0][0]["cancelled"] is False
    assert vote_calls[0][0]["error"] is not None


def test_peer_materialization_preflight_rejection_stops_all_ranks_before_barrier(
    monkeypatch,
) -> None:
    manager = _manager(_Source())
    barrier_calls = []
    vote_calls = []
    monkeypatch.setattr(
        torch.distributed,
        "monitored_barrier",
        lambda **kwargs: barrier_calls.append(kwargs),
    )

    def gather(_self, value, *, operation):
        vote_calls.append((value, operation))
        return [
            value,
            {
                "cancelled": False,
                "deadline_expired": True,
                "error": (
                    "weight materialization deadline does not leave enough time "
                    "for admission"
                ),
                "poisoned": None,
            },
        ]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    with pytest.raises(RuntimeError, match="rank 1.*deadline"):
        manager._run_weight_materialization(
            lambda _request: pytest.fail("materialization operation must not execute"),
            _commit_request(deadline_unix_sec=time.time() + 30),
            threading.Event(),
        )

    assert len(vote_calls) == 1
    assert barrier_calls == []


def test_multi_rank_materialization_gather_requires_execution_context(
    monkeypatch,
) -> None:
    manager = _manager(_Source())
    manager.weight_materialization_execution_context = None
    raw_collectives = []
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda *, group: 2,
    )

    def raw_all_gather(outputs, value, *, group):
        raw_collectives.append((outputs, value, group))
        outputs[:] = [value, value]

    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        raw_all_gather,
    )

    with pytest.raises(RuntimeError, match="execution context is required"):
        manager._gather_weight_materialization_objects(
            {"ready": True},
            operation="context regression",
        )

    assert raw_collectives == []


def test_single_rank_materialization_gather_does_not_require_context(
    monkeypatch,
) -> None:
    manager = _manager(_Source())
    manager.weight_materialization_execution_context = None
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda *, group: 1,
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda *_args, **_kwargs: pytest.fail("single-rank path must not gather"),
    )

    value = {"ready": True}
    assert manager._gather_weight_materialization_objects(
        value,
        operation="single-rank regression",
    ) == [value]


@pytest.mark.parametrize(
    "method_name",
    [
        "_gather_weight_materialization_sources_to_root",
        "_scatter_weight_materialization_sources_from_root",
    ],
)
def test_multi_rank_store_materialization_helpers_require_execution_context(
    monkeypatch,
    method_name,
) -> None:
    manager = _manager(_Source())
    manager.weight_materialization_execution_context = None
    coordinator_calls = []
    manager.weight_materialization_coordinator = SimpleNamespace(
        poisoned=False,
        gather_object_to_root=lambda *args, **kwargs: (
            coordinator_calls.append(("gather", args, kwargs)) or ()
        ),
        scatter_object_from_root=lambda *args, **kwargs: (
            coordinator_calls.append(("scatter", args, kwargs)) or None
        ),
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda *, group: 2,
    )

    method = getattr(manager, method_name)
    with pytest.raises(RuntimeError, match="execution context is required"):
        method(
            {"ready": True} if "gather" in method_name else None,
            operation="Store context regression",
        )

    assert coordinator_calls == []


def test_commit_deadline_vote_uses_bounded_cleanup_deadline_with_real_coordinator(
    monkeypatch,
) -> None:
    manager = _manager(_Source())
    group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *, group: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *, group: 1)
    coordinator = TorchDistributedWeightStoreCoordinator(group=group)
    original_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() - 1,
        cancel_signal=threading.Event(),
    )
    manager.weight_materialization_execution_context = original_context
    calls = []

    def gather(_self, value, *, operation):
        control_context = manager.weight_materialization_execution_context
        calls.append((value, operation, control_context))
        return coordinator.all_gather_object(
            value,
            phase=operation,
            execution_context=control_context,
        )

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    expired, error = manager._materialization_deadline_expired_world(
        original_context.deadline_unix_sec
    )

    assert expired is True
    assert error is None
    assert calls[0][1] == "commit deadline vote"
    assert calls[0][2] is not original_context
    assert calls[0][2].deadline_unix_sec > time.time()
    assert calls[0][2].deadline_unix_sec <= time.time() + 31
    assert coordinator.poisoned is False
    assert manager.weight_materialization_execution_context is original_context


def test_poisoned_materialization_lane_votes_before_rank_local_cleanup(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    session = manager.weight_materialization_sessions["materialize-0"]

    class Owner:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    owner = Owner()
    session.backend_owner = owner
    session.backend = _terminal_backend()
    manager.weight_materialization_poisoned = (
        "weight materialization process group is unusable"
    )
    monkeypatch.setattr(
        torch.distributed,
        "monitored_barrier",
        lambda **_kwargs: pytest.fail("poisoned cleanup must not enter a barrier"),
    )
    admission_votes = []

    def gather(_self, value, *, operation):
        admission_votes.append((value, operation))
        return [value]

    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        gather,
    )

    manager.defer_commit_weight_materialization(
        CommitWeightMaterializationReqInput(
            materialization_id="materialize-0",
            request_id="poisoned-cleanup",
            selected_external_dp_rank=None,
            storage_options={},
            phase="cleanup",
            deadline_unix_sec=time.time() + 1,
        )
    )
    output = manager.weight_materialization_pending[-1][0].result(timeout=1)

    assert output.success is False
    assert output.completion_unknown is True
    assert output.session_state == "cleanup_pending"
    assert "rank-local cleanup" in output.message
    assert source.released is True
    assert owner.closed is True
    assert session.source is None
    assert session.backend_owner is None
    assert session.backend is None
    assert len(admission_votes) == 1
    assert admission_votes[0][1] == "materialization admission vote"
    assert admission_votes[0][0]["poisoned"] is not None
    manager.close_remote_instance_weight_transfer_executor()


def test_stalled_bounded_provider_deadline_releases_lane_for_cleanup(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    provider_started = threading.Event()
    provider_finished = threading.Event()

    def materialize(source_arg, *, execution_context, publication_id, **_kwargs):
        assert source_arg is source
        assert execution_context is not None
        provider_started.set()
        while not execution_context.expired():
            time.sleep(min(0.005, execution_context.remaining_seconds()))
        provider_finished.set()
        raise WeightTransferError(
            "bounded provider deadline expired",
            code="DEADLINE_EXCEEDED",
            provider="mooncake-store",
            phase="wait",
            operation_id=publication_id,
            retryable=False,
            completion_known=True,
            cleanup_required=True,
        )

    _install_fake_selected_backend(monkeypatch, materialize=materialize)
    monkeypatch.setattr(
        torch.distributed,
        "monitored_barrier",
        lambda *, group, timeout, wait_all_ranks: None,
    )
    manager.defer_commit_weight_materialization(
        CommitWeightMaterializationReqInput(
            materialization_id="materialize-0",
            request_id="stalled-commit",
            selected_external_dp_rank=0,
            storage_options={"catalog_path": "/catalog"},
            deadline_unix_sec=time.time() + 1.1,
        )
    )
    assert provider_started.wait(timeout=1)
    manager.defer_commit_weight_materialization(
        CommitWeightMaterializationReqInput(
            materialization_id="materialize-0",
            request_id="queued-cleanup",
            selected_external_dp_rank=None,
            storage_options={"catalog_path": "/catalog"},
            phase="cleanup",
            deadline_unix_sec=time.time() + 3,
        )
    )

    completed = []
    wait_deadline = time.monotonic() + 3
    while (
        not any(request.request_id == "queued-cleanup" for _, request in completed)
        and time.monotonic() < wait_deadline
    ):
        completed.extend(manager.check_pending_remote_instance_weight_transfers())
        time.sleep(0.005)

    assert provider_finished.is_set()
    assert [request.request_id for _, request in completed] == [
        "stalled-commit",
        "queued-cleanup",
    ]
    cleanup = completed[-1][0]
    assert cleanup.phase == "cleanup"
    manager.close_remote_instance_weight_transfer_executor()


def test_executor_close_cancels_stalled_provider_and_returns_bounded(
    monkeypatch,
) -> None:
    source = _Source()
    manager = _manager(source)
    _install_world(monkeypatch)
    assert manager.prepare_weight_materialization(_prepare_request()).success
    provider_started = threading.Event()
    provider_saw_cancel = threading.Event()
    provider_finished = threading.Event()
    emergency_release = threading.Event()
    lifecycle = {}

    def materialize(source_arg, *, execution_context, publication_id, **_kwargs):
        assert source_arg is source
        assert execution_context is not None
        provider_started.set()
        try:
            while not emergency_release.wait(timeout=0.005):
                if execution_context.cancelled():
                    provider_saw_cancel.set()
            raise WeightTransferError(
                "stalled provider released after shutdown",
                code="CANCELLED",
                provider="mooncake-store",
                phase="wait",
                operation_id=publication_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            )
        finally:
            provider_finished.set()

    _install_fake_selected_backend(
        monkeypatch,
        materialize=materialize,
        lifecycle=lifecycle,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "_WEIGHT_MATERIALIZATION_SHUTDOWN_TIMEOUT_SEC",
        0.05,
        raising=False,
    )
    monkeypatch.setattr(
        torch.distributed,
        "monitored_barrier",
        lambda *, group, timeout, wait_all_ranks: None,
    )
    manager.defer_commit_weight_materialization(
        CommitWeightMaterializationReqInput(
            materialization_id="materialize-0",
            request_id="shutdown-stalled-commit",
            selected_external_dp_rank=0,
            storage_options={"catalog_path": "/catalog"},
            deadline_unix_sec=time.time() + 5,
        )
    )
    assert provider_started.wait(timeout=1)

    shutdown_done = threading.Event()
    shutdown_errors = []

    def shutdown():
        try:
            manager.close_remote_instance_weight_transfer_executor()
        except BaseException as error:
            shutdown_errors.append(error)
        finally:
            shutdown_done.set()

    shutdown_thread = threading.Thread(target=shutdown)
    started = time.monotonic()
    shutdown_thread.start()
    try:
        assert shutdown_done.wait(timeout=0.5)
        assert provider_saw_cancel.wait(timeout=0.5)
        assert source.released is False
        assert lifecycle.get("closed") is not True
        assert manager.weight_materialization_active_id == "materialize-0"
    finally:
        emergency_release.set()
        shutdown_thread.join(timeout=1)

    assert not shutdown_thread.is_alive()
    assert time.monotonic() - started < 0.75
    assert shutdown_errors == []
    assert provider_finished.wait(timeout=1)
    output = manager.weight_materialization_pending[-1][0].result(timeout=1)
    assert output.success is False
    assert source.released is True
    assert lifecycle["closed"] is True
    assert manager.weight_materialization_active_id is None


def test_scheduler_control_groups_cover_public_weight_operation_deadline(
    monkeypatch,
) -> None:
    from sglang.srt.managers import scheduler as scheduler_module

    created_groups = []
    manager_kwargs = {}

    def new_group(*, ranks, backend, timeout):
        group = object()
        created_groups.append((tuple(ranks), backend, timeout, group))
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

    assert len(created_groups) == 3
    assert created_groups[0][:2] == ((0, 1), "gloo")
    assert created_groups[1][:2] == ((0, 1), "gloo")
    assert created_groups[2][:2] == ((0, 1), "gloo")
    assert {group[2] for group in created_groups} == {created_groups[0][2]}
    assert created_groups[0][2].total_seconds() == (
        MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
    )
    assert manager_kwargs["remote_weight_transfer_cpu_group"] is created_groups[0][3]
    assert (
        manager_kwargs["remote_weight_transfer_control_cpu_group"]
        is created_groups[1][3]
    )
    assert manager_kwargs["weight_materialization_cpu_group"] is created_groups[2][3]


def test_materialization_collectives_are_serialized_inside_background_lane(
    monkeypatch,
) -> None:
    manager = _manager(_Source())
    group = object()
    manager.weight_materialization_cpu_group = group
    events = []

    monkeypatch.setattr(
        torch.distributed,
        "monitored_barrier",
        lambda *, group, timeout, wait_all_ranks: events.append(("barrier", group)),
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
    monkeypatch.setattr(
        SchedulerWeightUpdaterManager,
        "_gather_weight_materialization_objects",
        lambda _self, value, *, operation: [value],
    )

    manager.defer_prepare_weight_materialization(
        PrepareWeightMaterializationReqInput(
            materialization_id="materialize-prepare",
            request_id="materialize-prepare-request",
            model_id="model",
            revision="revision",
            lease_timeout_sec=300,
            deadline_unix_sec=time.time() + 30,
        )
    )
    manager.defer_commit_weight_materialization(
        CommitWeightMaterializationReqInput(
            materialization_id="materialize-commit",
            request_id="materialize-commit-request",
            selected_external_dp_rank=0,
            storage_options={},
            deadline_unix_sec=time.time() + 30,
        )
    )
    for future, _request in manager.weight_materialization_pending:
        future.result(timeout=1)
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
            yield _terminal_backend(
                provider=SimpleNamespace(name="mooncake-store"),
                catalog=object(),
            )
        finally:
            lifecycle["closed"] = True

    attestors = {}

    def prepare(
        *,
        source_placements,
        source_bindings,
        destination,
        payload_identity,
        operation_id,
        source_placements_are_selected=False,
    ):
        return SimpleNamespace(
            source_placements=tuple(source_placements),
            source_bindings=tuple(source_bindings),
            destination=destination,
            payload_identity=payload_identity,
            operation_id=operation_id,
            source_placements_are_selected=source_placements_are_selected,
        )

    def preflight(_provider, request, *, attestor):
        attestor.attest(request)
        attestors[id(request)] = attestor
        return object()

    def materialize_snapshot(request, **kwargs):
        source = getattr(attestors[id(request)], "source", None)
        return materialize(
            source,
            publication_id=request.operation_id,
            release_source=False,
            execution_context=kwargs.get("execution_context"),
        )

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
        "prepare_weight_materialization",
        prepare,
        raising=False,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "preflight_weight_transfer",
        preflight,
        raising=False,
    )
    monkeypatch.setattr(
        weight_updater_module,
        "materialize_weight_snapshot_candidate",
        materialize_snapshot,
        raising=False,
    )


if __name__ == "__main__":
    pytest_args = ["-x" if argument == "-f" else argument for argument in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *pytest_args]))
