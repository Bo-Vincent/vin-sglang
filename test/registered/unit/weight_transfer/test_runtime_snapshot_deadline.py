from __future__ import annotations

from dataclasses import dataclass

import pytest
import sglang.srt.weight_transfer.runtime as runtime_module
import torch
from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    WeightRuntimeManifestParts,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.contracts import RuntimeWeightLocation
from sglang.srt.weight_transfer.provider import (
    WeightPayloadIdentity,
    WeightTransferExecutionContext,
)
from sglang.srt.weight_transfer.runtime import (
    RuntimeWeightPayloadHasher,
    RuntimeWeightSnapshotSource,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _parts() -> WeightRuntimeManifestParts:
    tensor = WeightPlacementTensor(
        placement_fragment_id="fragment",
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=(1,),
        global_offset=(0,),
        local_shape=(1,),
        dtype="float32",
        itemsize=4,
        partition_dim=None,
        shard_dims=(),
        layer_id=0,
        expert_id=None,
        layout_fingerprint="logical-contiguous:float32:v1",
        nbytes=4,
        byte_offset=0,
        rank=WeightParallelRank(),
    )
    placement = WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id((tensor,)),
        tensors=(tensor,),
    )
    binding = WeightRuntimeBindingManifest(
        model_id=placement.model_id,
        revision=placement.revision,
        placement_id=placement.placement_id,
        instance_id="source-instance",
        generation=1,
        lease_id="source-lease",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id="runtime-fragment",
                address=1,
                nbytes=4,
                storage_offset=0,
                device="cpu",
                is_contiguous=True,
                worker_id="worker",
                endpoint="local",
            ),
        ),
    )
    return WeightRuntimeManifestParts(placement=placement, binding=binding)


@dataclass
class _Manager:
    parts: WeightRuntimeManifestParts
    clock: dict[str, float]
    released: bool = False

    def snapshot_parts(self, **_kwargs) -> WeightRuntimeManifestParts:
        self.clock["now"] += 3.0
        return self.parts

    def has_lease(self, lease_id: str) -> bool:
        return lease_id == self.parts.binding.lease_id and not self.released

    def release(self, lease_id: str) -> None:
        assert self.has_lease(lease_id)
        self.released = True


def _captured_source() -> RuntimeWeightSnapshotSource:
    parts = _parts()
    payload_identity = WeightPayloadIdentity.create(
        (parts.placement,),
        {"fragment": f"sha256:{'a' * 64}"},
    )
    return RuntimeWeightSnapshotSource(
        model=object(),
        manager=_Manager(parts, {"now": 100.0}),
        parts=parts,
        payload_hasher=object(),
        payload_identity=payload_identity,
    )


def test_materialize_runtime_weights_passes_execution_context(monkeypatch) -> None:
    source = _captured_source()
    context = WeightTransferExecutionContext(deadline_unix_sec=10**10)
    receipt = object()
    contexts = []

    def record_materialization(**kwargs):
        contexts.append(kwargs["execution_context"])
        return receipt

    monkeypatch.setattr(
        runtime_module,
        "materialize_weights",
        record_materialization,
    )

    result = runtime_module.materialize_runtime_weights(
        source,
        destination=object(),
        provider=object(),
        execution_context=context,
    )

    assert result is receipt
    assert contexts == [context]
    assert source.released is True


def test_materialize_runtime_snapshot_passes_execution_context(monkeypatch) -> None:
    source = _captured_source()
    context = WeightTransferExecutionContext(deadline_unix_sec=10**10)
    publication = object()
    contexts = []

    def record_materialization(**kwargs):
        contexts.append(kwargs["execution_context"])
        return publication

    monkeypatch.setattr(
        runtime_module,
        "materialize_weight_snapshot",
        record_materialization,
    )

    result = runtime_module.materialize_runtime_weight_snapshot(
        source,
        destination=object(),
        provider=object(),
        catalog=object(),
        execution_context=context,
        release_source=False,
    )

    assert result is publication
    assert contexts == [context]
    assert source.released is False
    source.release()


def test_distributed_runtime_snapshot_passes_execution_context(monkeypatch) -> None:
    source = _captured_source()
    context = WeightTransferExecutionContext(deadline_unix_sec=10**10)
    publication = object()
    contexts = []

    def record_materialization(**kwargs):
        contexts.append(kwargs["execution_context"])
        return publication

    monkeypatch.setattr(
        runtime_module,
        "materialize_weight_snapshot",
        record_materialization,
    )

    result = runtime_module.materialize_distributed_runtime_weight_snapshot(
        source,
        global_placements=(source.placement,),
        global_bindings=(source.binding,),
        payload_identity=source.payload_identity,
        destination=object(),
        provider=object(),
        catalog=object(),
        execution_context=context,
        release_source=False,
    )

    assert result is publication
    assert contexts == [context]
    assert source.released is False
    source.release()


def test_snapshot_hash_deadline_starts_before_snapshot_parts(monkeypatch) -> None:
    clock = {"now": 100.0}
    manager = _Manager(_parts(), clock)
    hash_deadlines = []

    class RecordingHasher:
        def __init__(self, _model, *, chunk_bytes):
            assert chunk_bytes > 0

        def __call__(self, _location, *, execution_context):
            hash_deadlines.append(execution_context.deadline_unix_sec)
            return f"sha256:{'a' * 64}"

    monkeypatch.setattr(runtime_module.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        runtime_module,
        "RuntimeWeightPayloadHasher",
        RecordingHasher,
    )

    source = RuntimeWeightSnapshotSource.capture(
        model=object(),
        manager=manager,
        model_id="model",
        revision="revision",
        instance_id="source-instance",
        worker_id="worker",
        endpoint="local",
        lease_timeout_sec=5,
        execution_context=WeightTransferExecutionContext(
            deadline_unix_sec=200.0,
        ),
    )

    assert hash_deadlines == [105.0]
    source.release()


def test_deferred_snapshot_hash_keeps_the_original_lease_deadline(monkeypatch) -> None:
    clock = {"now": 100.0}
    manager = _Manager(_parts(), clock)
    hash_deadlines = []

    class RecordingHasher:
        def __init__(self, _model, *, chunk_bytes):
            assert chunk_bytes > 0

        def __call__(self, _location, *, execution_context):
            hash_deadlines.append(execution_context.deadline_unix_sec)
            return f"sha256:{'a' * 64}"

    monkeypatch.setattr(runtime_module.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        runtime_module,
        "RuntimeWeightPayloadHasher",
        RecordingHasher,
    )

    source = RuntimeWeightSnapshotSource.capture(
        model=object(),
        manager=manager,
        model_id="model",
        revision="revision",
        instance_id="source-instance",
        worker_id="worker",
        endpoint="local",
        lease_timeout_sec=5,
        execution_context=WeightTransferExecutionContext(
            deadline_unix_sec=200.0,
        ),
        defer_payload_identity=True,
    )

    assert source.payload_identity is None
    assert hash_deadlines == []
    source.capture_payload_identity(
        execution_context=WeightTransferExecutionContext(
            deadline_unix_sec=200.0,
        )
    )
    assert hash_deadlines == [105.0]
    source.release()


def test_payload_hasher_resolves_subranges_without_scanning_all_parameters() -> None:
    model = torch.nn.Module()
    model.register_parameter(
        "weight",
        torch.nn.Parameter(torch.arange(16, dtype=torch.float32)),
    )
    hasher = RuntimeWeightPayloadHasher(model)
    parameter = model.get_parameter("weight")
    location = RuntimeWeightLocation(
        placement_id="placement",
        placement_fragment_id="fragment",
        fragment_id="runtime-fragment",
        tensor_id="weight",
        address=parameter.data_ptr() + 4 * parameter.element_size(),
        nbytes=4 * parameter.element_size(),
        storage_offset=4,
        device="cpu",
        worker_id="worker",
        endpoint="local",
        generation=1,
        lease_id="lease",
        rank=WeightParallelRank(),
        global_offset=(4,),
        local_shape=(4,),
    )

    class RejectLinearScan:
        def __iter__(self):
            raise AssertionError("payload range lookup must use its address index")

    hasher._ranges = RejectLinearScan()

    current, begin = hasher._resolve(location)

    assert current.address == parameter.data_ptr()
    assert begin == 4 * parameter.element_size()


def test_payload_hasher_address_index_handles_aliases_and_bounds() -> None:
    model = torch.nn.Module()
    base = torch.nn.Parameter(torch.arange(16, dtype=torch.float32))
    same_start_short = torch.nn.Parameter(base.detach()[:8])
    inner_alias = torch.nn.Parameter(base.detach()[4:12])
    other = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    model.register_parameter("base", base)
    model.register_parameter("same_start_short", same_start_short)
    model.register_parameter("inner_alias", inner_alias)
    model.register_parameter("other", other)
    hasher = RuntimeWeightPayloadHasher(model)

    def location(address: int, nbytes: int, *, device: str = "cpu"):
        return RuntimeWeightLocation(
            placement_id="placement",
            placement_fragment_id=f"fragment-{address}-{nbytes}",
            fragment_id=f"runtime-fragment-{address}-{nbytes}",
            tensor_id="weight",
            address=address,
            nbytes=nbytes,
            storage_offset=0,
            device=device,
            worker_id="worker",
            endpoint="local",
            generation=1,
            lease_id="lease",
            rank=WeightParallelRank(),
            global_offset=(0,),
            local_shape=(nbytes,),
        )

    current, begin = hasher._resolve(
        location(
            inner_alias.data_ptr(),
            inner_alias.numel() * inner_alias.element_size(),
        )
    )
    assert current.address == inner_alias.data_ptr()
    assert current.nbytes == inner_alias.numel() * inner_alias.element_size()
    assert begin == 0

    tail_address = base.data_ptr() + 12 * base.element_size()
    current, begin = hasher._resolve(location(tail_address, 4 * base.element_size()))
    assert current.address == base.data_ptr()
    assert current.nbytes == base.numel() * base.element_size()
    assert begin == 12 * base.element_size()

    max_end = max(item.address + item.nbytes for item in hasher._ranges)
    with pytest.raises(ValueError, match="not owned"):
        hasher._resolve(location(max_end, base.element_size()))
    with pytest.raises(ValueError, match="not owned"):
        hasher._resolve(location(base.data_ptr(), base.element_size(), device="cuda"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
