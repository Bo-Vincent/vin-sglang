from __future__ import annotations

import asyncio
from types import SimpleNamespace

import msgspec

from sglang.srt.managers.io_struct import (
    BeginRemoteInstanceWeightTransferReqInput,
    BeginRemoteInstanceWeightTransferReqOutput,
    ReleaseRemoteInstanceWeightTransferReqInput,
    RenewRemoteInstanceWeightTransferReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.model_executor import weight_inventory_contracts as contracts


def _axes() -> tuple[contracts.LogicalParallelAxis, ...]:
    return (
        contracts.LogicalParallelAxis(kind="dp", mode="replicated"),
        contracts.LogicalParallelAxis(kind="tp", mode="split", dim=0),
        contracts.LogicalParallelAxis(kind="pp", mode="ownership"),
        contracts.LogicalParallelAxis(kind="ep", mode="replicated"),
    )


def _inventory_pair(
    *,
    tp_rank=0,
    tp_size=1,
    weight_generation=7,
    binding_generation=1,
    lease_id=None,
):
    topology = contracts.WeightParallelTopology(
        tp_rank=tp_rank,
        tp_size=tp_size,
        attention_tp_rank=tp_rank,
        attention_tp_size=tp_size,
    )
    rank = topology.rank()
    local_rows = 4
    facts = {
        "tensor_id": "layers.0.weight",
        "aliases": (),
        "global_shape": (local_rows * tp_size, 4),
        "global_offset": (local_rows * tp_rank, 0),
        "local_shape": (local_rows, 4),
        "dtype": "float16",
        "itemsize": 2,
        "shard_dims": (0,),
        "parallel_axes": _axes(),
        "layer_id": 0,
        "expert_id": None,
        "layout_fingerprint": "logical-contiguous",
        "nbytes": 32,
        "rank": rank,
    }
    fragment = contracts.WeightPlacementInventoryFragment(
        placement_fragment_id=contracts._placement_fragment_id(**facts),
        **facts,
    )
    placement_facts = {
        "model_id": "model",
        "revision": "immutable-revision",
        "weight_generation": weight_generation,
        "topology": topology,
        "fragments": (fragment,),
    }
    placement = contracts.WeightPlacementInventory(
        inventory_id=contracts._placement_id(**placement_facts),
        participant_id=contracts._participant_id(
            model_id="model",
            revision="immutable-revision",
            topology=topology,
        ),
        **placement_facts,
    )
    address = 0x1000 + tp_rank * 0x1000
    runtime_fragment = contracts.WeightRuntimeBindingInventoryFragment(
        placement_fragment_id=fragment.placement_fragment_id,
        fragment_id=f"runtime-{tp_rank}-{binding_generation}",
        address=address,
        nbytes=32,
        storage_offset=0,
        itemsize=2,
        local_shape=(4, 4),
        strides_bytes=(8, 2),
        storage_address=address,
        storage_nbytes=32,
        storage_offset_bytes=0,
        device="cuda",
        is_contiguous=True,
        worker_id=f"worker-{tp_rank}",
        endpoint=f"worker-{tp_rank}:1234",
    )
    binding = contracts.WeightRuntimeBindingInventory(
        model_id="model",
        revision="immutable-revision",
        placement_inventory_id=placement.inventory_id,
        instance_id=f"instance-{tp_rank}",
        generation=binding_generation,
        lease_id=lease_id or f"lease-{tp_rank}",
        participant_id=placement.participant_id,
        fragments=(runtime_fragment,),
    )
    return contracts.WeightPlacementBindingInventories(
        placement=placement,
        binding=binding,
    )


class _Runner:
    def __init__(self, inventories) -> None:
        self.inventories = inventories
        self.capture_calls = []
        self.released = []
        self.renewed = []

    def get_remote_instance_weight_inventories(self, **kwargs):
        self.capture_calls.append(kwargs)
        return self.inventories

    def release_weight_inventory(self, lease_id):
        self.released.append(lease_id)

    def renew_weight_inventory(self, lease_id, *, lease_timeout_sec):
        self.renewed.append((lease_id, lease_timeout_sec))


def _manager(runner) -> SchedulerWeightUpdaterManager:
    return SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(model_runner=runner),
        draft_worker=None,
        tp_cpu_group=object(),
        world_cpu_group=object(),
        memory_saver_adapter=object(),
        flush_cache=lambda **_kwargs: True,
        is_fully_idle=lambda: True,
    )


def _single_rank_collectives(monkeypatch) -> None:
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 1)
    monkeypatch.setattr(
        "torch.distributed.all_gather_object",
        lambda outputs, value, group: outputs.__setitem__(0, value),
    )


def _request(*, revision="immutable-revision"):
    return BeginRemoteInstanceWeightTransferReqInput(
        transfer_id="transfer-1",
        model_id="model",
        revision=revision,
        lease_timeout_sec=60,
    )


def test_scheduler_captures_current_inventory_and_releases_lease(monkeypatch) -> None:
    runner = _Runner(_inventory_pair())
    manager = _manager(runner)
    _single_rank_collectives(monkeypatch)

    output = manager.begin_remote_instance_weight_transfer(_request())

    assert output.success is True
    assert output.session_state == "created"
    assert len(output.placement_inventories) == 1
    assert len(output.binding_inventories) == 1
    assert output.placement_inventories[0]["weight_generation"] == 7
    assert output.binding_inventories[0]["generation"] == 1

    released = manager.release_remote_instance_weight_transfer(
        ReleaseRemoteInstanceWeightTransferReqInput(transfer_id="transfer-1")
    )
    assert released.success is True
    assert runner.released == ["lease-0"]


def test_duplicate_begin_reuses_exact_inventory_without_new_lease(monkeypatch) -> None:
    runner = _Runner(_inventory_pair())
    manager = _manager(runner)
    _single_rank_collectives(monkeypatch)

    first = manager.begin_remote_instance_weight_transfer(_request())
    second = manager.begin_remote_instance_weight_transfer(_request())

    assert first.session_state == "created"
    assert second.session_state == "reused"
    assert second.placement_inventories == first.placement_inventories
    assert second.binding_inventories == first.binding_inventories
    assert len(runner.capture_calls) == 1


def test_transfer_id_reuse_with_different_revision_fails_closed(monkeypatch) -> None:
    runner = _Runner(_inventory_pair())
    manager = _manager(runner)
    _single_rank_collectives(monkeypatch)
    assert manager.begin_remote_instance_weight_transfer(_request()).success

    conflict = manager.begin_remote_instance_weight_transfer(
        _request(revision="different-revision")
    )

    assert conflict.success is False
    assert conflict.session_state == "conflict"
    assert "different parameters" in conflict.message


def test_scheduler_rejects_inventory_identity_mismatch(monkeypatch) -> None:
    runner = _Runner(_inventory_pair())
    manager = _manager(runner)
    _single_rank_collectives(monkeypatch)

    output = manager.begin_remote_instance_weight_transfer(
        _request(revision="different-revision")
    )

    assert output.success is False
    assert "requested model identity" in output.message
    assert runner.released == ["lease-0"]


def test_scheduler_rejects_placeholder_revision_before_inventory_capture(
    monkeypatch,
) -> None:
    runner = _Runner(_inventory_pair())
    manager = _manager(runner)
    _single_rank_collectives(monkeypatch)

    output = manager.begin_remote_instance_weight_transfer(_request(revision="default"))

    assert output.success is False
    assert "content-lineage revision" in output.message
    assert runner.capture_calls == []


def test_scheduler_accepts_mixed_local_binding_generations(monkeypatch) -> None:
    local = _inventory_pair(tp_rank=0, tp_size=2, binding_generation=3)
    remote = _inventory_pair(tp_rank=1, tp_size=2, binding_generation=19)
    runner = _Runner(local)
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 2)

    def gather(outputs, value, group):
        outputs[:] = [
            value,
            {
                "success": True,
                "message": "Success.",
                "session_state": "created",
                "placement_inventory": msgspec.to_builtins(remote.placement),
                "binding_inventory": msgspec.to_builtins(remote.binding),
            },
        ]

    monkeypatch.setattr("torch.distributed.all_gather_object", gather)

    output = manager.begin_remote_instance_weight_transfer(_request())

    assert output.success is True
    assert {item["generation"] for item in output.binding_inventories} == {3, 19}
    assert {item["weight_generation"] for item in output.placement_inventories} == {7}


def test_scheduler_rejects_mixed_logical_weight_generations(monkeypatch) -> None:
    local = _inventory_pair(tp_rank=0, tp_size=2, weight_generation=7)
    remote = _inventory_pair(tp_rank=1, tp_size=2, weight_generation=8)
    runner = _Runner(local)
    manager = _manager(runner)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group: 2)

    def gather(outputs, value, group):
        outputs[:] = [
            value,
            {
                "success": True,
                "message": "Success.",
                "session_state": "created",
                "placement_inventory": msgspec.to_builtins(remote.placement),
                "binding_inventory": msgspec.to_builtins(remote.binding),
            },
        ]

    monkeypatch.setattr("torch.distributed.all_gather_object", gather)

    output = manager.begin_remote_instance_weight_transfer(_request())

    assert output.success is False
    assert "logical weight generation" in output.message
    assert runner.released == ["lease-0"]


def test_scheduler_rejects_binding_geometry_before_backend(monkeypatch) -> None:
    pair = _inventory_pair()
    payload = msgspec.to_builtins(pair.binding)
    payload["fragments"][0]["itemsize"] = 1
    runner = _Runner(
        {
            "placement": msgspec.to_builtins(pair.placement),
            "binding": payload,
        }
    )
    manager = _manager(runner)
    _single_rank_collectives(monkeypatch)

    output = manager.begin_remote_instance_weight_transfer(_request())

    assert output.success is False
    assert "self-consistent" in output.message


def test_renew_uses_the_local_runtime_lease(monkeypatch) -> None:
    runner = _Runner(_inventory_pair())
    manager = _manager(runner)
    _single_rank_collectives(monkeypatch)
    assert manager.begin_remote_instance_weight_transfer(_request()).success

    renewed = manager.renew_remote_instance_weight_transfer(
        RenewRemoteInstanceWeightTransferReqInput(
            transfer_id="transfer-1",
            lease_timeout_sec=120,
        )
    )

    assert renewed.success is True
    assert runner.renewed == [("lease-0", 120)]


def test_io_request_has_no_schema_selector() -> None:
    fields = {field.name for field in msgspec.structs.fields(_request().__class__)}

    assert "manifest_format" not in fields
    assert fields >= {"transfer_id", "model_id", "revision", "lease_timeout_sec"}


def test_tokenizer_deadline_starts_after_inventory_capture(monkeypatch) -> None:
    now = [100.0]

    class Manager:
        is_pause = True
        server_args = SimpleNamespace(
            model_path="/node-a/cache/model",
            revision="immutable-revision",
            get_weight_reshard_resource_id=lambda: "canonical-model",
        )
        captured_request = None

        async def begin_remote_instance_weight_transfer_communicator(self, request):
            self.captured_request = request
            now[0] = 130.0
            pair = _inventory_pair()
            return [
                BeginRemoteInstanceWeightTransferReqOutput(
                    transfer_id=request.transfer_id,
                    success=True,
                    message="Success.",
                    session_state="created",
                    placement_inventories=[msgspec.to_builtins(pair.placement)],
                    binding_inventories=[msgspec.to_builtins(pair.binding)],
                )
            ]

    manager = Manager()
    monkeypatch.setattr(
        "sglang.srt.managers.tokenizer_control_mixin.time.time", lambda: now[0]
    )

    output = asyncio.run(
        TokenizerControlMixin._begin_remote_instance_weight_transfer(
            manager,
            lease_timeout_sec=60,
            transfer_id="transfer-1",
        )
    )

    assert output["success"] is True
    assert manager.captured_request.model_id == "canonical-model"
    assert (
        manager._remote_weight_transfer_session_index["transfer-1"]["deadline_unix_sec"]
        == 190.0
    )
