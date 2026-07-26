from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest
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
    LocalWeightBufferRegistry,
    LocalWeightTransferProvider,
    WeightPayloadIdentity,
    WeightStorageDestination,
    WeightTransferCompletionUnknownError,
    WeightTransferTerminalProof,
    WeightTransferTerminalStatus,
)
from sglang.srt.weight_transfer.runtime import (
    RuntimeWeightPayloadHasher,
    RuntimeWeightSnapshotSource,
    materialize_distributed_runtime_weight_snapshot,
    materialize_runtime_weight_snapshot,
    quarantined_runtime_weight_snapshots,
)
from sglang.srt.weight_transfer.storage import InMemoryWeightStorageCatalog
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(8, dtype=torch.float32),
            requires_grad=False,
        )


def _location(model: _Model, *, offset: int, nbytes: int) -> RuntimeWeightLocation:
    return RuntimeWeightLocation(
        placement_id="placement",
        placement_fragment_id="fragment",
        fragment_id="runtime-fragment",
        tensor_id="weight",
        address=model.weight.data_ptr() + offset,
        nbytes=nbytes,
        storage_offset=0,
        device="cpu",
        worker_id="worker",
        endpoint="local",
        generation=1,
        lease_id="lease",
        rank=WeightParallelRank(),
        global_offset=(0,),
        local_shape=(nbytes,),
        aliases=("weight",),
    )


def test_runtime_payload_hasher_hashes_exact_address_range_in_bounded_chunks() -> None:
    model = _Model()
    payload = model.weight.detach().reshape(-1).view(torch.uint8).numpy().tobytes()
    hasher = RuntimeWeightPayloadHasher(model, chunk_bytes=3)

    assert hasher(_location(model, offset=5, nbytes=11)) == (
        f"sha256:{hashlib.sha256(payload[5:16]).hexdigest()}"
    )
    with pytest.raises(ValueError, match="not owned by the runtime model"):
        hasher(_location(model, offset=len(payload) - 2, nbytes=8))


def _parts(model: _Model) -> WeightRuntimeManifestParts:
    nbytes = model.weight.numel() * model.weight.element_size()
    tensor = WeightPlacementTensor(
        placement_fragment_id="fragment",
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=(model.weight.numel(),),
        global_offset=(0,),
        local_shape=(model.weight.numel(),),
        dtype="float32",
        itemsize=model.weight.element_size(),
        partition_dim=None,
        shard_dims=(),
        layer_id=0,
        expert_id=None,
        layout_fingerprint="logical-contiguous:float32:v1",
        nbytes=nbytes,
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
                address=model.weight.data_ptr(),
                nbytes=nbytes,
                storage_offset=0,
                device="cpu",
                is_contiguous=True,
                worker_id="worker",
                endpoint="local",
            ),
        ),
    )
    return WeightRuntimeManifestParts(placement=placement, binding=binding)


def _tp_parts(
    model: _Model,
    *,
    tp_rank: int,
) -> WeightRuntimeManifestParts:
    nbytes = model.weight.numel() * model.weight.element_size()
    fragment_id = f"fragment:{tp_rank}"
    tensor = WeightPlacementTensor(
        placement_fragment_id=fragment_id,
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=(model.weight.numel() * 2,),
        global_offset=(model.weight.numel() * tp_rank,),
        local_shape=(model.weight.numel(),),
        dtype="float32",
        itemsize=model.weight.element_size(),
        partition_dim=0,
        shard_dims=(0,),
        layer_id=0,
        expert_id=None,
        layout_fingerprint="logical-contiguous:float32:v1",
        nbytes=nbytes,
        byte_offset=0,
        rank=WeightParallelRank(tp=tp_rank),
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
        instance_id=f"source-instance:{tp_rank}",
        generation=1,
        lease_id=f"source-lease:{tp_rank}",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=fragment_id,
                fragment_id=f"runtime-fragment:{tp_rank}",
                address=model.weight.data_ptr(),
                nbytes=nbytes,
                storage_offset=0,
                device="cpu",
                is_contiguous=True,
                worker_id=f"worker:{tp_rank}",
                endpoint=f"local:{tp_rank}",
            ),
        ),
    )
    return WeightRuntimeManifestParts(placement=placement, binding=binding)


@dataclass
class _Manager:
    parts: WeightRuntimeManifestParts
    attestations: int = 0
    released: bool = False

    def snapshot_parts(self, **_kwargs) -> WeightRuntimeManifestParts:
        return self.parts

    def attest_binding(self, binding: WeightRuntimeBindingManifest) -> None:
        assert binding == self.parts.binding
        assert not self.released
        self.attestations += 1

    def has_lease(self, lease_id: str) -> bool:
        return lease_id == self.parts.binding.lease_id and not self.released

    def release(self, lease_id: str) -> None:
        assert self.has_lease(lease_id)
        self.released = True


def test_runtime_snapshot_source_materializes_then_releases_source_lease() -> None:
    model = _Model()
    parts = _parts(model)
    manager = _Manager(parts)
    source = RuntimeWeightSnapshotSource.capture(
        model=model,
        manager=manager,
        model_id="model",
        revision="revision",
        instance_id="source-instance",
        worker_id="worker",
        endpoint="local",
    )
    registry = LocalWeightBufferRegistry()
    payload = model.weight.detach().reshape(-1).view(torch.uint8).numpy().tobytes()
    registry.register_runtime(model.weight.data_ptr(), payload)
    provider = LocalWeightTransferProvider(registry)
    catalog = InMemoryWeightStorageCatalog()

    publication = materialize_runtime_weight_snapshot(
        source,
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/model/revision",
            object_prefix="weights/model/revision",
        ),
        provider=provider,
        catalog=catalog,
        publication_id="publication",
    )

    assert publication.snapshot is not None
    assert catalog.get_snapshot(publication.snapshot.ref) == publication.snapshot
    assert manager.attestations == 1
    assert manager.released is True
    assert source.released is True


@pytest.mark.parametrize("release_source", [True, False])
def test_distributed_runtime_snapshot_uses_global_manifests_and_local_lease(
    release_source,
) -> None:
    models = (_Model(), _Model())
    managers = (
        _Manager(_tp_parts(models[0], tp_rank=0)),
        _Manager(_tp_parts(models[1], tp_rank=1)),
    )
    sources = tuple(
        RuntimeWeightSnapshotSource.capture(
            model=model,
            manager=manager,
            model_id="model",
            revision="revision",
            instance_id=f"source-instance:{rank}",
            worker_id=f"worker:{rank}",
            endpoint=f"local:{rank}",
        )
        for rank, (model, manager) in enumerate(zip(models, managers, strict=True))
    )
    placements = tuple(source.placement for source in sources)
    bindings = tuple(source.binding for source in sources)
    checksums = {
        fragment.placement_fragment_id: fragment.checksum
        for source in sources
        for fragment in source.payload_identity.fragments
    }
    identity = WeightPayloadIdentity.create(placements, checksums)
    registry = LocalWeightBufferRegistry()
    for model in models:
        registry.register_runtime(
            model.weight.data_ptr(),
            model.weight.detach().view(torch.uint8).numpy().tobytes(),
        )
    catalog = InMemoryWeightStorageCatalog()

    publication = materialize_distributed_runtime_weight_snapshot(
        sources[0],
        global_placements=placements,
        global_bindings=bindings,
        payload_identity=identity,
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/model/revision",
            object_prefix="weights/model/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="distributed-publication",
        release_source=release_source,
    )

    assert publication.snapshot.placements == tuple(
        sorted(placements, key=lambda placement: placement.placement_id)
    )
    assert sources[0].released is release_source
    assert managers[0].released is release_source
    assert sources[1].released is False
    if not release_source:
        sources[0].release()
    sources[1].release()


def test_distributed_runtime_snapshot_accepts_partial_local_projection() -> None:
    class TwoWeightModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Parameter(
                torch.arange(4, dtype=torch.float32),
                requires_grad=False,
            )
            self.second = torch.nn.Parameter(
                torch.arange(4, 8, dtype=torch.float32),
                requires_grad=False,
            )

    model = TwoWeightModel()
    tensors = tuple(
        WeightPlacementTensor(
            placement_fragment_id=f"fragment:{name}",
            tensor_id=name,
            runtime_name=name,
            aliases=(name,),
            global_shape=(parameter.numel(),),
            global_offset=(0,),
            local_shape=(parameter.numel(),),
            dtype="float32",
            itemsize=parameter.element_size(),
            partition_dim=None,
            shard_dims=(),
            layer_id=0,
            expert_id=None,
            layout_fingerprint="logical-contiguous:float32:v1",
            nbytes=parameter.numel() * parameter.element_size(),
            byte_offset=0,
            rank=WeightParallelRank(tp=1),
        )
        for name, parameter in model.named_parameters()
    )
    placement = WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id(tensors),
        tensors=tensors,
    )
    parameters = dict(model.named_parameters())
    binding = WeightRuntimeBindingManifest(
        model_id=placement.model_id,
        revision=placement.revision,
        placement_id=placement.placement_id,
        instance_id="source-instance:1",
        generation=1,
        lease_id="source-lease:1",
        fragments=tuple(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=parameters[tensor.tensor_id].data_ptr(),
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cpu",
                is_contiguous=True,
                worker_id="worker:1",
                endpoint="local:1",
            )
            for tensor in tensors
        ),
    )
    manager = _Manager(
        WeightRuntimeManifestParts(
            placement=placement,
            binding=binding,
        )
    )
    source = RuntimeWeightSnapshotSource.capture(
        model=model,
        manager=manager,
        model_id="model",
        revision="revision",
        instance_id="source-instance:1",
        worker_id="worker:1",
        endpoint="local:1",
    )
    selected_placement = WeightPlacementManifest(
        model_id=placement.model_id,
        revision=placement.revision,
        placement_id=compute_weight_placement_id((tensors[1],)),
        tensors=(tensors[1],),
    )
    selected_binding = WeightRuntimeBindingManifest(
        model_id=binding.model_id,
        revision=binding.revision,
        placement_id=selected_placement.placement_id,
        instance_id=binding.instance_id,
        generation=binding.generation,
        lease_id=binding.lease_id,
        fragments=(binding.fragments[1],),
    )
    selected_identity = source.payload_identity.select((selected_placement,))
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(
        model.second.data_ptr(),
        model.second.detach().view(torch.uint8).numpy().tobytes(),
    )

    publication = materialize_distributed_runtime_weight_snapshot(
        source,
        global_placements=(selected_placement,),
        global_bindings=(selected_binding,),
        payload_identity=selected_identity,
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/model/revision",
            object_prefix="weights/model/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=InMemoryWeightStorageCatalog(),
        publication_id="partial-publication",
    )

    assert publication.snapshot.placements == (selected_placement,)
    assert manager.attestations == 1
    assert source.released is True


class _CompletionUnknownProvider(LocalWeightTransferProvider):
    name = "completion-unknown-local"

    def wait(self, submission):
        raise WeightTransferCompletionUnknownError(
            "completion is not observable",
            provider=self.name,
            phase="wait",
            operation_id=submission.request.operation_id,
            completion_ticket="pending-transfer",
        )


def test_runtime_snapshot_source_retains_lease_until_terminal_proof() -> None:
    model = _Model()
    manager = _Manager(_parts(model))
    source = RuntimeWeightSnapshotSource.capture(
        model=model,
        manager=manager,
        model_id="model",
        revision="revision",
        instance_id="source-instance",
        worker_id="worker",
        endpoint="local",
    )
    registry = LocalWeightBufferRegistry()
    payload = model.weight.detach().reshape(-1).view(torch.uint8).numpy().tobytes()
    registry.register_runtime(model.weight.data_ptr(), payload)
    provider = _CompletionUnknownProvider(registry)

    with pytest.raises(WeightTransferCompletionUnknownError):
        materialize_runtime_weight_snapshot(
            source,
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/model/revision",
                object_prefix="weights/model/revision",
            ),
            provider=provider,
            catalog=InMemoryWeightStorageCatalog(),
            publication_id="unknown-publication",
        )

    assert source.quarantined is True
    assert source in quarantined_runtime_weight_snapshots()
    assert manager.released is False
    with pytest.raises(RuntimeError, match="cannot be released"):
        source.release()

    source.resolve_quarantine(
        WeightTransferTerminalProof(
            operation_id=source.operation_id,
            provider=provider.name,
            completion_ticket="pending-transfer",
            status=WeightTransferTerminalStatus.COMPLETED,
        )
    )

    assert source.quarantined is False
    assert source.released is True
    assert manager.released is True
    assert source not in quarantined_runtime_weight_snapshots()
