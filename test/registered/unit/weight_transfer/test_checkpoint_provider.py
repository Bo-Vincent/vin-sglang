from __future__ import annotations

import ctypes
import hashlib
from math import prod

import pytest

from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.api import load_weights_to_local_target
from sglang.srt.weight_transfer.checkpoint_provider import (
    CheckpointLoadStats,
    CheckpointProviderState,
    CheckpointStorageToRuntimeProvider,
)
from sglang.srt.weight_transfer.contracts import (
    RuntimeWeightLocation,
    StorageWeightLocation,
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
)
from sglang.srt.weight_transfer.lowering import WeightLoweringLimits
from sglang.srt.weight_transfer.provider import (
    WeightTargetLoadMode,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DEFAULT_CHECKSUM = object()


class _TargetAttestor:
    def __init__(self, binding: WeightRuntimeBindingManifest) -> None:
        self.binding = binding

    def attest(self, request) -> None:
        assert request.plan.target_bindings == (self.binding,)


def _placement(
    side: str,
    *,
    global_shape: tuple[int, ...],
    global_offset: tuple[int, ...],
    local_shape: tuple[int, ...],
    shard_dims: tuple[int, ...],
    rank: WeightParallelRank,
) -> WeightPlacementManifest:
    tensor = WeightPlacementTensor(
        placement_fragment_id=(
            f"{side}:weight:{rank.dp}:{rank.tp}:{rank.pp}:{rank.ep}"
        ),
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=global_shape,
        global_offset=global_offset,
        local_shape=local_shape,
        dtype="uint8",
        itemsize=1,
        partition_dim=shard_dims[0] if len(shard_dims) == 1 else None,
        shard_dims=shard_dims,
        layer_id=0,
        expert_id=None,
        layout_fingerprint="logical-contiguous:v2",
        nbytes=prod(local_shape),
        byte_offset=0,
        rank=rank,
    )
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id((tensor,)),
        tensors=(tensor,),
    )


def _storage_binding(
    placement: WeightPlacementManifest,
    *,
    provider: str,
    object_key: str,
    object_offset: int,
    payload: bytes,
    checksum: str | None | object = _DEFAULT_CHECKSUM,
) -> WeightStorageBindingManifest:
    tensor = placement.tensors[0]
    return WeightStorageBindingManifest(
        model_id=placement.model_id,
        revision=placement.revision,
        placement_id=placement.placement_id,
        storage_id=f"{provider}:revision",
        provider=provider,
        fragments=(
            WeightStorageFragmentBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"stored:{tensor.placement_fragment_id}",
                object_key=object_key,
                object_offset=object_offset,
                nbytes=len(payload),
                checksum=(
                    f"sha256:{hashlib.sha256(payload).hexdigest()}"
                    if checksum is _DEFAULT_CHECKSUM
                    else checksum
                ),
            ),
        ),
    )


def _runtime_binding(
    placement: WeightPlacementManifest,
    *,
    address: int,
    device: str,
) -> WeightRuntimeBindingManifest:
    tensor = placement.tensors[0]
    return WeightRuntimeBindingManifest(
        model_id=placement.model_id,
        revision=placement.revision,
        placement_id=placement.placement_id,
        instance_id=f"target:{address}",
        generation=1,
        lease_id=f"lease:{address}",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=address,
                nbytes=tensor.nbytes,
                storage_offset=0,
                device=device,
                is_contiguous=True,
                worker_id="target-worker",
                endpoint="target-worker:1",
            ),
        ),
    )


def _load(
    *,
    source_placements: tuple[WeightPlacementManifest, ...],
    source_bindings: tuple[WeightStorageBindingManifest, ...],
    target_placement: WeightPlacementManifest,
    target_binding: WeightRuntimeBindingManifest,
    provider: CheckpointStorageToRuntimeProvider,
):
    return load_weights_to_local_target(
        source_placements=source_placements,
        source_bindings=source_bindings,
        target_placement=target_placement,
        target_binding=target_binding,
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
        attestor=_TargetAttestor(target_binding),
    )


def test_checkpoint_provider_requires_runtime_target_attestation() -> None:
    source = _placement(
        "source",
        global_shape=(1,),
        global_offset=(0,),
        local_shape=(1,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(1,),
        global_offset=(0,),
        local_shape=(1,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    source_binding = _storage_binding(
        source,
        provider="checkpoint",
        object_key="/unused",
        object_offset=0,
        payload=b"\x00",
    )

    with pytest.raises(WeightTransferError, match="attestor is required") as raised:
        load_weights_to_local_target(
            source_placements=(source,),
            source_bindings=(source_binding,),
            target_placement=target,
            target_binding=_runtime_binding(target, address=0x1000, device="cpu"),
            provider=CheckpointStorageToRuntimeProvider(),
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.code == "ATTESTATION_REQUIRED"
    assert raised.value.phase == "attest"


def test_local_checkpoint_provider_loads_cross_dim_nd_bytes_exactly(
    tmp_path,
) -> None:
    source_0 = _placement(
        "source",
        global_shape=(4, 4),
        global_offset=(0, 0),
        local_shape=(2, 4),
        shard_dims=(0,),
        rank=WeightParallelRank(tp=0),
    )
    source_1 = _placement(
        "source",
        global_shape=(4, 4),
        global_offset=(2, 0),
        local_shape=(2, 4),
        shard_dims=(0,),
        rank=WeightParallelRank(tp=1),
    )
    target = _placement(
        "target",
        global_shape=(4, 4),
        global_offset=(0, 0),
        local_shape=(4, 2),
        shard_dims=(1,),
        rank=WeightParallelRank(tp=0),
    )
    payload_0 = bytes(range(8))
    payload_1 = bytes(range(8, 16))
    path_0 = tmp_path / "shard-0.bin"
    path_1 = tmp_path / "shard-1.bin"
    path_0.write_bytes(b"abc" + payload_0)
    path_1.write_bytes(b"defgh" + payload_1)
    target_buffer = ctypes.create_string_buffer(target.tensors[0].nbytes)
    provider = CheckpointStorageToRuntimeProvider()

    receipt = _load(
        source_placements=(source_0, source_1),
        source_bindings=(
            _storage_binding(
                source_0,
                provider="checkpoint",
                object_key=str(path_0),
                object_offset=3,
                payload=payload_0,
            ),
            _storage_binding(
                source_1,
                provider="checkpoint",
                object_key=str(path_1),
                object_offset=5,
                payload=payload_1,
            ),
        ),
        target_placement=target,
        target_binding=_runtime_binding(
            target,
            address=ctypes.addressof(target_buffer),
            device="cpu",
        ),
        provider=provider,
    )

    assert bytes(target_buffer.raw) == bytes((0, 1, 4, 5, 8, 9, 12, 13))
    assert receipt.total_bytes == 8
    assert receipt.region_count == 2
    assert isinstance(receipt.backend_receipts[0], CheckpointLoadStats)
    assert provider.lifecycle == (
        CheckpointProviderState.PREPARED,
        CheckpointProviderState.SUBMITTED,
        CheckpointProviderState.COMPLETED,
        CheckpointProviderState.RELEASED,
    )


def test_bad_checksum_is_rejected_before_any_target_write() -> None:
    source = _placement(
        "source",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    payload = bytes(range(8))
    writes: list[tuple[int, bytes]] = []

    with pytest.raises(WeightTransferError) as caught:
        _load(
            source_placements=(source,),
            source_bindings=(
                _storage_binding(
                    source,
                    provider="oss",
                    object_key="oss://bucket/model.bin",
                    object_offset=11,
                    payload=payload,
                    checksum=f"sha256:{'0' * 64}",
                ),
            ),
            target_placement=target,
            target_binding=_runtime_binding(
                target,
                address=0x20000,
                device="cuda:0",
            ),
            provider=CheckpointStorageToRuntimeProvider(
                range_reader=lambda location, offset, nbytes: payload[
                    offset
                    - location.object_offset : offset
                    - location.object_offset
                    + nbytes
                ],
                source_version_reader=lambda location: "version-1",
                target_writer=lambda target, offset, data: writes.append(
                    (target.address + offset, data)
                ),
            ),
        )

    assert writes == []
    assert caught.value.code == "CHECKSUM_MISMATCH"
    assert caught.value.phase == "prepare"
    assert caught.value.completion_known is True
    assert caught.value.cleanup_required is False


def test_checksummed_injected_reader_requires_source_version_fencing() -> None:
    source = _placement(
        "source",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    payload = bytes(range(8))
    writes = []

    with pytest.raises(WeightTransferError) as caught:
        _load(
            source_placements=(source,),
            source_bindings=(
                _storage_binding(
                    source,
                    provider="oss",
                    object_key="oss://bucket/model.bin",
                    object_offset=0,
                    payload=payload,
                ),
            ),
            target_placement=target,
            target_binding=_runtime_binding(
                target,
                address=0x25000,
                device="cuda:0",
            ),
            provider=CheckpointStorageToRuntimeProvider(
                range_reader=lambda location, offset, nbytes: payload[
                    offset : offset + nbytes
                ],
                target_writer=lambda target, offset, data: writes.append(data),
            ),
        )

    assert writes == []
    assert caught.value.code == "SOURCE_VERSION_REQUIRED"
    assert caught.value.phase == "prepare"


def test_provider_lowers_large_nd_plan_into_bounded_batches() -> None:
    rows = 17
    source = _placement(
        "source",
        global_shape=(rows, 2),
        global_offset=(0, 0),
        local_shape=(rows, 2),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(rows, 2),
        global_offset=(0, 0),
        local_shape=(rows, 1),
        shard_dims=(1,),
        rank=WeightParallelRank(),
    )
    payload = bytes(range(rows * 2))
    target_payload = bytearray(rows)
    provider = CheckpointStorageToRuntimeProvider(
        range_reader=lambda location, offset, nbytes: payload[
            offset - location.object_offset : offset - location.object_offset + nbytes
        ],
        source_version_reader=lambda location: "version-1",
        target_writer=lambda target, offset, data: target_payload.__setitem__(
            slice(offset, offset + len(data)),
            data,
        ),
        lowering_limits=WeightLoweringLimits(
            max_total_operations=rows,
            max_batch_operations=3,
            max_batch_bytes=3,
        ),
    )

    receipt = _load(
        source_placements=(source,),
        source_bindings=(
            _storage_binding(
                source,
                provider="oss",
                object_key="oss://bucket/model.bin",
                object_offset=7,
                payload=payload,
            ),
        ),
        target_placement=target,
        target_binding=_runtime_binding(target, address=0x30000, device="cuda:0"),
        provider=provider,
    )

    stats = receipt.backend_receipts[0]
    assert isinstance(stats, CheckpointLoadStats)
    assert stats.operation_count == rows
    assert stats.batch_count == 6
    assert stats.max_batch_operations == 3
    assert stats.max_batch_bytes == 3
    assert target_payload == payload[::2]


def test_lowering_operation_limit_is_rejected_before_target_mutation() -> None:
    source = _placement(
        "source",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    payload = bytes(range(8))
    writes = []

    with pytest.raises(WeightTransferError) as caught:
        _load(
            source_placements=(source,),
            source_bindings=(
                _storage_binding(
                    source,
                    provider="oss",
                    object_key="oss://bucket/model.bin",
                    object_offset=0,
                    payload=payload,
                ),
            ),
            target_placement=target,
            target_binding=_runtime_binding(
                target,
                address=0x35000,
                device="cuda:0",
            ),
            provider=CheckpointStorageToRuntimeProvider(
                range_reader=lambda location, offset, nbytes: payload[
                    offset : offset + nbytes
                ],
                source_version_reader=lambda location: "version-1",
                target_writer=lambda target, offset, data: writes.append(data),
                lowering_limits=WeightLoweringLimits(
                    max_total_operations=3,
                    max_batch_operations=2,
                    max_batch_bytes=2,
                ),
            ),
        )

    assert writes == []
    assert caught.value.code == "LOWERING_LIMIT_EXCEEDED"
    assert caught.value.phase == "prepare"
    assert caught.value.cleanup_required is False


def test_injected_oss_reader_receives_object_byte_ranges() -> None:
    source = _placement(
        "source",
        global_shape=(6,),
        global_offset=(0,),
        local_shape=(6,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(6,),
        global_offset=(0,),
        local_shape=(6,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    payload = b"weight"
    calls: list[tuple[str, str, int, int]] = []
    target_payload = bytearray(6)

    def read_oss(
        location: StorageWeightLocation,
        object_offset: int,
        nbytes: int,
    ) -> bytes:
        calls.append(
            (
                location.provider,
                location.object_key,
                object_offset,
                nbytes,
            )
        )
        relative = object_offset - location.object_offset
        return payload[relative : relative + nbytes]

    _load(
        source_placements=(source,),
        source_bindings=(
            _storage_binding(
                source,
                provider="oss",
                object_key="oss://bucket/model.bin",
                object_offset=4096,
                payload=payload,
            ),
        ),
        target_placement=target,
        target_binding=_runtime_binding(target, address=0x40000, device="cuda:0"),
        provider=CheckpointStorageToRuntimeProvider(
            range_reader=read_oss,
            source_version_reader=lambda location: "version-1",
            target_writer=lambda target, offset, data: target_payload.__setitem__(
                slice(offset, offset + len(data)),
                data,
            ),
            checksum_chunk_bytes=4,
        ),
    )

    assert target_payload == payload
    assert calls == [
        ("oss", "oss://bucket/model.bin", 4096, 4),
        ("oss", "oss://bucket/model.bin", 4100, 2),
        ("oss", "oss://bucket/model.bin", 4096, 6),
    ]


def test_sync_writer_failure_has_known_completion_and_releases_lifecycle() -> None:
    source = _placement(
        "source",
        global_shape=(4, 2),
        global_offset=(0, 0),
        local_shape=(4, 2),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(4, 2),
        global_offset=(0, 0),
        local_shape=(4, 1),
        shard_dims=(1,),
        rank=WeightParallelRank(),
    )
    payload = bytes(range(8))
    writes = 0

    def fail_second_write(target, offset, data) -> None:
        nonlocal writes
        del target, offset, data
        writes += 1
        if writes == 2:
            raise RuntimeError("device write failed")

    provider = CheckpointStorageToRuntimeProvider(
        range_reader=lambda location, offset, nbytes: payload[
            offset - location.object_offset : offset - location.object_offset + nbytes
        ],
        source_version_reader=lambda location: "version-1",
        target_writer=fail_second_write,
        lowering_limits=WeightLoweringLimits(
            max_total_operations=4,
            max_batch_operations=1,
        ),
    )

    with pytest.raises(WeightTransferError) as caught:
        _load(
            source_placements=(source,),
            source_bindings=(
                _storage_binding(
                    source,
                    provider="oss",
                    object_key="oss://bucket/model.bin",
                    object_offset=0,
                    payload=payload,
                ),
            ),
            target_placement=target,
            target_binding=_runtime_binding(
                target,
                address=0x50000,
                device="cuda:0",
            ),
            provider=provider,
        )

    assert writes == 2
    assert caught.value.code == "TARGET_WRITE_FAILED"
    assert caught.value.phase == "submit"
    assert caught.value.completion_known is True
    assert caught.value.cleanup_required is True
    assert provider.lifecycle == (
        CheckpointProviderState.PREPARED,
        CheckpointProviderState.SUBMITTED,
        CheckpointProviderState.FAILED,
        CheckpointProviderState.RELEASED,
    )


def test_injected_writer_preserves_completion_unknown_error() -> None:
    source = _placement(
        "source",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    payload = bytes(range(8))
    provider = CheckpointStorageToRuntimeProvider(
        range_reader=lambda location, offset, nbytes: payload[offset : offset + nbytes],
        source_version_reader=lambda location: "version-1",
        target_writer=lambda target, offset, data: (_ for _ in ()).throw(
            WeightTransferCompletionUnknownError(
                "writer completion is unknown",
                provider="checkpoint",
                phase="submit",
                operation_id="backend-operation",
                completion_ticket="writer-ticket",
            )
        ),
    )

    with pytest.raises(WeightTransferCompletionUnknownError) as caught:
        _load(
            source_placements=(source,),
            source_bindings=(
                _storage_binding(
                    source,
                    provider="oss",
                    object_key="oss://bucket/model.bin",
                    object_offset=0,
                    payload=payload,
                ),
            ),
            target_placement=target,
            target_binding=_runtime_binding(
                target,
                address=0x55000,
                device="cuda:0",
            ),
            provider=provider,
        )

    assert caught.value.completion_known is False
    assert caught.value.completion_ticket == "writer-ticket"
    assert provider.lifecycle == (
        CheckpointProviderState.PREPARED,
        CheckpointProviderState.SUBMITTED,
        CheckpointProviderState.FAILED,
    )


def test_source_version_change_is_rejected_before_affected_target_write() -> None:
    source = _placement(
        "source",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    payload = bytes(range(8))
    version_calls = 0
    writes = []

    def source_version(location: StorageWeightLocation) -> str:
        nonlocal version_calls
        del location
        version_calls += 1
        return "version-1" if version_calls <= 2 else "version-2"

    with pytest.raises(WeightTransferError) as caught:
        _load(
            source_placements=(source,),
            source_bindings=(
                _storage_binding(
                    source,
                    provider="oss",
                    object_key="oss://bucket/model.bin",
                    object_offset=0,
                    payload=payload,
                ),
            ),
            target_placement=target,
            target_binding=_runtime_binding(
                target,
                address=0x60000,
                device="cuda:0",
            ),
            provider=CheckpointStorageToRuntimeProvider(
                range_reader=lambda location, offset, nbytes: payload[
                    offset : offset + nbytes
                ],
                source_version_reader=source_version,
                target_writer=lambda target, offset, data: writes.append(data),
            ),
        )

    assert writes == []
    assert caught.value.code == "SOURCE_VERSION_CHANGED"
    assert caught.value.phase == "submit"
    assert caught.value.completion_known is True
    assert caught.value.cleanup_required is False


def test_checksumless_source_is_version_fenced_before_each_target_write() -> None:
    source = _placement(
        "source",
        global_shape=(4, 2),
        global_offset=(0, 0),
        local_shape=(4, 2),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(4, 2),
        global_offset=(0, 0),
        local_shape=(4, 1),
        shard_dims=(1,),
        rank=WeightParallelRank(),
    )
    payload = bytes(range(8))
    version_calls = 0
    writes = []

    def source_version(_location: StorageWeightLocation) -> str:
        nonlocal version_calls
        version_calls += 1
        return "version-1" if version_calls <= 3 else "version-2"

    with pytest.raises(WeightTransferError) as caught:
        _load(
            source_placements=(source,),
            source_bindings=(
                _storage_binding(
                    source,
                    provider="oss",
                    object_key="oss://bucket/model.bin",
                    object_offset=0,
                    payload=payload,
                    checksum=None,
                ),
            ),
            target_placement=target,
            target_binding=_runtime_binding(
                target,
                address=0x65000,
                device="cuda:0",
            ),
            provider=CheckpointStorageToRuntimeProvider(
                range_reader=lambda location, offset, nbytes: payload[
                    offset : offset + nbytes
                ],
                source_version_reader=source_version,
                target_writer=lambda target, offset, data: writes.append(
                    (offset, data)
                ),
            ),
        )

    assert writes == [(0, b"\x00")]
    assert caught.value.code == "SOURCE_VERSION_CHANGED"
    assert caught.value.phase == "submit"
    assert caught.value.cleanup_required is True


def test_checksumless_injected_reader_requires_source_version_fencing() -> None:
    source = _placement(
        "source",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    target = _placement(
        "target",
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        shard_dims=(),
        rank=WeightParallelRank(),
    )
    payload = bytes(range(8))
    writes = []

    with pytest.raises(WeightTransferError) as caught:
        _load(
            source_placements=(source,),
            source_bindings=(
                _storage_binding(
                    source,
                    provider="oss",
                    object_key="oss://bucket/model.bin",
                    object_offset=0,
                    payload=payload,
                    checksum=None,
                ),
            ),
            target_placement=target,
            target_binding=_runtime_binding(
                target,
                address=0x70000,
                device="cuda:0",
            ),
            provider=CheckpointStorageToRuntimeProvider(
                range_reader=lambda location, offset, nbytes: payload[
                    offset : offset + nbytes
                ],
                target_writer=lambda target, offset, data: writes.append(data),
            ),
        )

    assert writes == []
    assert caught.value.code == "SOURCE_VERSION_REQUIRED"
    assert caught.value.phase == "prepare"


@pytest.mark.parametrize("target_offset", [-1, 8])
def test_checkpoint_writer_rejects_out_of_bounds_range_before_mutation(
    target_offset: int,
) -> None:
    writes = []
    provider = CheckpointStorageToRuntimeProvider(
        range_reader=lambda source, offset, nbytes: bytes(nbytes),
        source_version_reader=lambda source: "version-1",
        target_writer=lambda target, offset, payload: writes.append(
            (target, offset, payload)
        ),
    )
    target = RuntimeWeightLocation(
        placement_id="placement",
        placement_fragment_id="placement-fragment",
        fragment_id="runtime-fragment",
        tensor_id="weight",
        address=0x80000,
        nbytes=8,
        storage_offset=0,
        device="cuda:0",
        worker_id="worker",
        endpoint="worker:1",
        generation=1,
        lease_id="lease",
        rank=WeightParallelRank(),
        global_offset=(0,),
        local_shape=(8,),
    )

    with pytest.raises(ValueError, match="target range exceeds runtime binding"):
        provider._write_target(target, target_offset, b"x")

    assert writes == []


def test_checkpoint_provider_does_not_advertise_backend_cancellation() -> None:
    provider = CheckpointStorageToRuntimeProvider()

    assert provider.probe(None).supports_safe_cancel is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
