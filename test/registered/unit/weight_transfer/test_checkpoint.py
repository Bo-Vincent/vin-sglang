from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pytest

from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    WeightRuntimeManifestParts,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.checkpoint import (
    SemanticCheckpointSource,
    load_checkpoint_weights,
    materialize_checkpoint_weight_snapshot,
    materialize_checkpoint_weights,
)
from sglang.srt.weight_transfer.contracts import (
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
)
from sglang.srt.weight_transfer.provider import (
    LocalWeightBufferRegistry,
    LocalWeightTransferProvider,
    WeightPayloadIdentity,
    WeightStorageDestination,
    WeightTargetLoadMode,
    WeightTransferCompletionUnknownError,
    WeightTransferTerminalProof,
    WeightTransferTerminalStatus,
)
from sglang.srt.weight_transfer.runtime import (
    RuntimeWeightSnapshotSource,
    quarantined_runtime_weight_snapshots,
)
from sglang.srt.weight_transfer.storage import (
    InMemoryWeightStorageCatalog,
    WeightSnapshotPublicationState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def placement(
    side: str,
    *,
    layout: str = "layout:v1",
) -> WeightPlacementManifest:
    tensors = (
        WeightPlacementTensor(
            placement_fragment_id=f"{side}:fragment",
            tensor_id="weight",
            runtime_name="weight",
            aliases=("weight",),
            global_shape=(8,),
            global_offset=(0,),
            local_shape=(8,),
            dtype="uint8",
            itemsize=1,
            partition_dim=None,
            shard_dims=(),
            layer_id=0,
            expert_id=None,
            layout_fingerprint=layout,
            nbytes=8,
            byte_offset=0,
            rank=WeightParallelRank(),
        ),
    )
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id(tuple(tensors)),
        tensors=tensors,
    )


def runtime_binding(
    manifest: WeightPlacementManifest,
    address: int,
) -> WeightRuntimeBindingManifest:
    tensor = manifest.tensors[0]
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id=f"{manifest.placement_id}:instance",
        generation=1,
        lease_id=f"{manifest.placement_id}:lease",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=address,
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cpu",
                is_contiguous=True,
                worker_id=manifest.placement_id,
                endpoint=f"{manifest.placement_id}:1",
            ),
        ),
    )


@dataclass
class _RuntimeOwner:
    binding: WeightRuntimeBindingManifest
    attestations: int = 0
    released: bool = False

    def attest_binding(self, binding: WeightRuntimeBindingManifest) -> None:
        assert binding == self.binding
        assert not self.released
        self.attestations += 1

    def has_lease(self, lease_id: str) -> bool:
        return lease_id == self.binding.lease_id and not self.released

    def release(self, lease_id: str) -> None:
        assert self.has_lease(lease_id)
        self.released = True


def owned_runtime_source(
    manifest: WeightPlacementManifest,
    binding: WeightRuntimeBindingManifest,
) -> tuple[RuntimeWeightSnapshotSource, _RuntimeOwner]:
    checksum = f"sha256:{hashlib.sha256(bytes(range(8))).hexdigest()}"
    identity = WeightPayloadIdentity.create(
        (manifest,),
        {
            manifest.tensors[0].placement_fragment_id: checksum,
        },
    )
    owner = _RuntimeOwner(binding)
    return (
        RuntimeWeightSnapshotSource(
            model=object(),
            manager=owner,
            parts=WeightRuntimeManifestParts(
                placement=manifest,
                binding=binding,
            ),
            payload_hasher=lambda _location: checksum,
            payload_identity=identity,
        ),
        owner,
    )


def checkpoint_source(
    manifest: WeightPlacementManifest,
) -> SemanticCheckpointSource:
    tensor = manifest.tensors[0]
    return SemanticCheckpointSource(
        placements=(manifest,),
        bindings=(
            WeightStorageBindingManifest(
                model_id=manifest.model_id,
                revision=manifest.revision,
                placement_id=manifest.placement_id,
                storage_id="checkpoint:revision",
                provider="checkpoint",
                fragments=(
                    WeightStorageFragmentBinding(
                        placement_fragment_id=tensor.placement_fragment_id,
                        fragment_id=f"stored:{tensor.placement_fragment_id}",
                        object_key="checkpoint/model.safetensors",
                        object_offset=128,
                        nbytes=tensor.nbytes,
                    ),
                ),
            ),
        ),
    )


def test_checkpoint_without_sidecar_uses_framework_loader() -> None:
    calls = []

    result = load_checkpoint_weights(
        source=None,
        target_placements=(),
        target_bindings=(),
        provider=None,
        framework_loader=lambda: calls.append("framework") or "loaded",
        target_mode=WeightTargetLoadMode.COLD_START,
    )

    assert result == "loaded"
    assert calls == ["framework"]


def test_semantic_checkpoint_loads_through_nd_provider() -> None:
    source = placement("source")
    target = placement("target")
    target_binding = runtime_binding(target, 0x20000)
    registry = LocalWeightBufferRegistry()
    registry.storage_objects["checkpoint/model.safetensors"] = bytearray(
        128
    ) + bytearray(range(8))
    registry.register_runtime(0x20000, bytearray(8))

    receipt = load_checkpoint_weights(
        source=checkpoint_source(source),
        target_placements=(target,),
        target_bindings=(target_binding,),
        provider=LocalWeightTransferProvider(registry),
        framework_loader=lambda: pytest.fail(
            "semantic sidecars must not use the framework fallback"
        ),
        target_mode=WeightTargetLoadMode.COLD_START,
    )

    assert receipt.total_bytes == 8
    assert registry.read_runtime(0x20000, 8) == bytes(range(8))


def test_semantic_checkpoint_forwards_runtime_attestation() -> None:
    source = placement("source")
    target = placement("target")
    target_binding = runtime_binding(target, 0x20000)
    registry = LocalWeightBufferRegistry()
    registry.storage_objects["checkpoint/model.safetensors"] = bytearray(
        128
    ) + bytearray(range(8))
    registry.register_runtime(0x20000, bytearray(8))
    requests = []

    load_checkpoint_weights(
        source=checkpoint_source(source),
        target_placements=(target,),
        target_bindings=(target_binding,),
        provider=LocalWeightTransferProvider(registry),
        framework_loader=None,
        target_mode=WeightTargetLoadMode.COLD_START,
        attestor=type(
            "RecordingAttestor",
            (),
            {"attest": lambda self, request: requests.append(request)},
        )(),
    )

    assert len(requests) == 1
    assert requests[0].profile == "storage_to_runtime"


def test_invalid_semantic_checkpoint_does_not_fallback() -> None:
    source = placement("source")
    target = placement("target", layout="layout:v2")
    calls = []

    with pytest.raises(ValueError, match="descriptor mismatch"):
        load_checkpoint_weights(
            source=checkpoint_source(source),
            target_placements=(target,),
            target_bindings=(runtime_binding(target, 0x20000),),
            provider=LocalWeightTransferProvider(LocalWeightBufferRegistry()),
            framework_loader=lambda: calls.append("framework"),
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert calls == []


def test_checkpoint_without_sidecar_can_materialize_via_runtime_export() -> None:
    source = placement("source")
    source_binding = runtime_binding(source, 0x10000)
    runtime_source, owner = owned_runtime_source(source, source_binding)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytearray(range(8)))
    calls = []

    receipt = materialize_checkpoint_weights(
        source=None,
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights:revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        framework_load_and_export=lambda: (
            calls.append("framework"),
            runtime_source,
        )[1],
    )

    assert calls == ["framework"]
    assert owner.released is True
    assert receipt.total_bytes == 8
    assert receipt.stored_placements == (source,)
    assert len(registry.storage_objects) == 1


def test_checkpoint_runtime_export_rejects_unowned_manifest_tuple() -> None:
    source = placement("source")
    source_binding = runtime_binding(source, 0x10000)

    with pytest.raises(ValueError, match="owned runtime snapshot"):
        materialize_checkpoint_weights(
            source=None,
            destination=WeightStorageDestination(
                provider="local",
                storage_id="weights:revision",
                object_prefix="weights/revision",
            ),
            provider=LocalWeightTransferProvider(LocalWeightBufferRegistry()),
            framework_load_and_export=lambda: ((source,), (source_binding,)),
        )


def test_checkpoint_materialization_forwards_runtime_attestation() -> None:
    source = placement("source")
    source_binding = runtime_binding(source, 0x10000)
    runtime_source, owner = owned_runtime_source(source, source_binding)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytearray(range(8)))
    requests = []

    materialize_checkpoint_weights(
        source=None,
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights:revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        framework_load_and_export=lambda: runtime_source,
        attestor=type(
            "RecordingAttestor",
            (),
            {"attest": lambda self, request: requests.append(request)},
        )(),
    )

    assert len(requests) == 1
    assert owner.attestations == 1
    assert owner.released is True
    assert requests[0].profile == "runtime_to_storage"


def test_semantic_checkpoint_can_publish_store_snapshot() -> None:
    source = placement("source")
    registry = LocalWeightBufferRegistry()
    registry.storage_objects["checkpoint/model.safetensors"] = bytearray(
        128
    ) + bytearray(range(8))
    catalog = InMemoryWeightStorageCatalog()

    publication = materialize_checkpoint_weight_snapshot(
        source=checkpoint_source(source),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights:revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="checkpoint-to-store",
        framework_load_and_export=lambda: pytest.fail(
            "semantic checkpoint must not use the framework fallback"
        ),
    )

    assert publication.state is WeightSnapshotPublicationState.PUBLISHED
    assert publication.snapshot.ref.manifest_key == "weights/revision/manifest"
    assert catalog.get_snapshot(publication.snapshot.ref) == publication.snapshot
    assert publication.snapshot.placements == (source,)


def test_checkpoint_without_sidecar_can_publish_runtime_export() -> None:
    source = placement("source")
    source_binding = runtime_binding(source, 0x10000)
    runtime_source, owner = owned_runtime_source(source, source_binding)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytearray(range(8)))
    catalog = InMemoryWeightStorageCatalog()
    calls = []

    publication = materialize_checkpoint_weight_snapshot(
        source=None,
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights:revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="runtime-export-to-store",
        framework_load_and_export=lambda: (
            calls.append("framework"),
            runtime_source,
        )[1],
    )

    assert calls == ["framework"]
    assert owner.released is True
    assert publication.state is WeightSnapshotPublicationState.PUBLISHED
    assert publication.snapshot.placements == (source,)


def test_checkpoint_snapshot_forwards_runtime_attestation() -> None:
    source = placement("source")
    source_binding = runtime_binding(source, 0x10000)
    runtime_source, owner = owned_runtime_source(source, source_binding)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytearray(range(8)))
    requests = []

    publication = materialize_checkpoint_weight_snapshot(
        source=None,
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights:revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=InMemoryWeightStorageCatalog(),
        publication_id="attested-runtime-export",
        framework_load_and_export=lambda: runtime_source,
        attestor=type(
            "RecordingAttestor",
            (),
            {"attest": lambda self, request: requests.append(request)},
        )(),
    )

    assert publication.state is WeightSnapshotPublicationState.PUBLISHED
    assert owner.attestations == 1
    assert owner.released is True
    assert len(requests) == 1
    assert requests[0].profile == "runtime_to_storage"


class _CompletionUnknownProvider(LocalWeightTransferProvider):
    name = "local"

    def wait(self, submission):
        raise WeightTransferCompletionUnknownError(
            "checkpoint Store completion is unknown",
            provider=self.name,
            phase="wait",
            operation_id=submission.request.operation_id,
            completion_ticket="checkpoint-ticket",
        )


def test_checkpoint_runtime_owner_is_quarantined_until_terminal_proof() -> None:
    manifest = placement("source")
    binding = runtime_binding(manifest, 0x10000)
    runtime_source, owner = owned_runtime_source(manifest, binding)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytearray(range(8)))

    with pytest.raises(WeightTransferCompletionUnknownError):
        materialize_checkpoint_weight_snapshot(
            source=None,
            destination=WeightStorageDestination(
                provider="local",
                storage_id="weights:revision",
                object_prefix="weights/revision",
            ),
            provider=_CompletionUnknownProvider(registry),
            catalog=InMemoryWeightStorageCatalog(),
            publication_id="checkpoint-unknown",
            framework_load_and_export=lambda: runtime_source,
        )

    assert owner.released is False
    assert runtime_source in quarantined_runtime_weight_snapshots()
    runtime_source.resolve_quarantine(
        WeightTransferTerminalProof(
            operation_id=runtime_source.operation_id,
            provider="local",
            completion_ticket="checkpoint-ticket",
            status=WeightTransferTerminalStatus.COMPLETED,
        )
    )
    assert owner.released is True


def test_semantic_checkpoint_sidecar_has_stable_bytes_and_file_roundtrip(
    tmp_path,
) -> None:
    source = checkpoint_source(placement("source"))

    payload = source.to_json_bytes()
    restored = SemanticCheckpointSource.from_json_bytes(payload)
    path = tmp_path / "model.weights.json"
    restored.write_sidecar(path)

    assert restored == source
    assert restored.to_json_bytes() == payload
    assert SemanticCheckpointSource.read_sidecar(path) == source
    assert path.read_bytes() == payload


@pytest.mark.parametrize(
    "mutate",
    [
        lambda document: {**document, "unexpected": True},
        lambda document: {**document, "version": 999},
        lambda document: {
            **document,
            "bindings": [
                {
                    **document["bindings"][0],
                    "fragments": [
                        {
                            **document["bindings"][0]["fragments"][0],
                            "unknown": "field",
                        }
                    ],
                }
            ],
        },
        lambda document: {
            **document,
            "bindings": [
                {
                    **document["bindings"][0],
                    "model_id": "different-model",
                }
            ],
        },
        lambda document: {
            **document,
            "placements": [
                {
                    **document["placements"][0],
                    "tensors": [
                        {
                            **document["placements"][0]["tensors"][0],
                            "nbytes": 7,
                        }
                    ],
                }
            ],
        },
    ],
)
def test_semantic_checkpoint_sidecar_rejects_schema_corruption(mutate) -> None:
    payload = checkpoint_source(placement("source")).to_json_bytes()
    document = json.loads(payload)
    corrupted = json.dumps(mutate(document), separators=(",", ":")).encode()

    with pytest.raises(ValueError, match="sidecar"):
        SemanticCheckpointSource.from_json_bytes(corrupted)


def test_semantic_checkpoint_sidecar_enforces_size_bounds(tmp_path) -> None:
    source = checkpoint_source(placement("source"))
    payload = source.to_json_bytes()
    path = tmp_path / "oversized.weights.json"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="size"):
        source.to_json_bytes(max_bytes=len(payload) - 1)
    with pytest.raises(ValueError, match="size"):
        SemanticCheckpointSource.from_json_bytes(
            payload,
            max_bytes=len(payload) - 1,
        )
    with pytest.raises(ValueError, match="size"):
        SemanticCheckpointSource.read_sidecar(
            path,
            max_bytes=len(payload) - 1,
        )


def test_semantic_checkpoint_sidecar_rejects_truncated_json() -> None:
    payload = checkpoint_source(placement("source")).to_json_bytes()

    with pytest.raises(ValueError, match="sidecar"):
        SemanticCheckpointSource.from_json_bytes(payload[:-1])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
