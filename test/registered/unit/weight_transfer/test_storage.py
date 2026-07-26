from __future__ import annotations

from dataclasses import replace

import pytest

from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.contracts import (
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
)
from sglang.srt.weight_transfer.storage import (
    InMemoryWeightStorageCatalog,
    StoredWeightSnapshot,
    WeightRevisionHead,
    WeightRevisionState,
    WeightSnapshotPublicationState,
    WeightStorageRef,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def placement(index: int) -> WeightPlacementManifest:
    fragment_id = f"placement:{index}:fragment"
    tensor = WeightPlacementTensor(
        placement_fragment_id=fragment_id,
        tensor_id=f"weight:{index}",
        runtime_name=f"weight_{index}",
        aliases=(f"weight_{index}",),
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        dtype="bfloat16",
        itemsize=2,
        partition_dim=None,
        shard_dims=(),
        layer_id=index,
        expert_id=None,
        layout_fingerprint="layout:v1",
        nbytes=16,
        byte_offset=0,
        rank=WeightParallelRank(pp=index),
    )
    tensors = (tensor,)
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id(tuple(tensors)),
        tensors=tensors,
    )


def multi_fragment_placement() -> WeightPlacementManifest:
    tensors = tuple(
        WeightPlacementTensor(
            placement_fragment_id=f"multi:fragment:{index}",
            tensor_id=f"multi:weight:{index}",
            runtime_name=f"multi_weight_{index}",
            aliases=(f"multi_weight_{index}", f"multi_alias_{index}"),
            global_shape=(8,),
            global_offset=(0,),
            local_shape=(8,),
            dtype="bfloat16",
            itemsize=2,
            partition_dim=None,
            shard_dims=(),
            layer_id=index,
            expert_id=None,
            layout_fingerprint="layout:v1",
            nbytes=16,
            byte_offset=index * 16,
            rank=WeightParallelRank(),
        )
        for index in range(2)
    )
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id(tuple(tensors)),
        tensors=tensors,
    )


def storage_binding(
    manifest: WeightPlacementManifest,
    *,
    provider: str = "test-store",
    storage_id: str = "weights/revision",
    object_key: str | None = None,
    object_offset: int = 0,
    nbytes_delta: int = 0,
    checksum: str | None = None,
) -> WeightStorageBindingManifest:
    tensor = manifest.tensors[0]
    return WeightStorageBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        storage_id=storage_id,
        provider=provider,
        fragments=(
            WeightStorageFragmentBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"stored:{tensor.placement_fragment_id}",
                object_key=object_key or f"objects/{manifest.placement_id}",
                object_offset=object_offset,
                nbytes=tensor.nbytes + nbytes_delta,
                checksum=checksum,
            ),
        ),
    )


def multi_fragment_binding(
    manifest: WeightPlacementManifest,
    *,
    reverse: bool = False,
) -> WeightStorageBindingManifest:
    fragments = tuple(
        WeightStorageFragmentBinding(
            placement_fragment_id=tensor.placement_fragment_id,
            fragment_id=f"stored:{tensor.placement_fragment_id}",
            object_key="objects/multi",
            object_offset=index * 16,
            nbytes=tensor.nbytes,
        )
        for index, tensor in enumerate(
            sorted(
                manifest.tensors,
                key=lambda item: item.placement_fragment_id,
            )
        )
    )
    return WeightStorageBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        storage_id="weights/revision",
        provider="test-store",
        fragments=tuple(reversed(fragments)) if reverse else fragments,
    )


def snapshot(
    placements: tuple[WeightPlacementManifest, ...] | None = None,
    bindings: tuple[WeightStorageBindingManifest, ...] | None = None,
) -> StoredWeightSnapshot:
    placements = placements or (placement(0), placement(1))
    bindings = bindings or tuple(storage_binding(item) for item in placements)
    return StoredWeightSnapshot.create(
        provider="test-store",
        storage_id="weights/revision",
        manifest_key="weights/revision/manifest.json",
        placements=placements,
        storage_bindings=bindings,
    )


def test_storage_ref_requires_canonical_sha256_digest() -> None:
    with pytest.raises(ValueError, match="canonical sha256"):
        WeightStorageRef(
            provider="test-store",
            storage_id="weights/revision",
            manifest_key="weights/revision/manifest.json",
            manifest_digest="not-a-digest",
        )


def test_snapshot_digest_is_canonical_and_order_independent() -> None:
    placements = (placement(0), placement(1))
    bindings = tuple(storage_binding(item) for item in placements)

    forward = snapshot(placements, bindings)
    reverse = snapshot(tuple(reversed(placements)), tuple(reversed(bindings)))

    assert forward.digest == reverse.digest
    assert forward.ref.manifest_digest == forward.digest
    assert forward.digest.startswith("sha256:")
    assert [item.placement_id for item in reverse.placements] == [
        item.placement_id
        for item in sorted(placements, key=lambda item: item.placement_id)
    ]


def test_canonical_snapshot_identity_is_binding_order_independent() -> None:
    placement = multi_fragment_placement()
    forward = snapshot(
        (placement,),
        (multi_fragment_binding(placement),),
    )
    reverse = snapshot(
        (placement,),
        (multi_fragment_binding(placement, reverse=True),),
    )

    assert forward.digest == reverse.digest
    assert forward == reverse

    catalog = InMemoryWeightStorageCatalog()
    pending = catalog.prepare_publish("publish-retry", forward)
    assert catalog.prepare_publish("publish-retry", reverse) == pending


def test_snapshot_digest_commits_ref_and_manifest_contents() -> None:
    original = snapshot()
    changed_binding = storage_binding(
        original.placements[0],
        checksum="sha256:" + "1" * 64,
    )
    changed = snapshot(
        original.placements,
        (changed_binding, original.storage_bindings[1]),
    )

    assert changed.digest != original.digest
    with pytest.raises(ValueError, match="digest"):
        StoredWeightSnapshot(
            ref=replace(
                original.ref,
                manifest_key="weights/revision/other-manifest.json",
            ),
            placements=original.placements,
            storage_bindings=original.storage_bindings,
            digest=original.digest,
        )


def test_snapshot_rejects_ref_digest_identity_mismatch() -> None:
    original = snapshot()

    with pytest.raises(ValueError, match="manifest digest"):
        StoredWeightSnapshot(
            ref=replace(
                original.ref,
                manifest_digest="sha256:" + "f" * 64,
            ),
            placements=original.placements,
            storage_bindings=original.storage_bindings,
            digest=original.digest,
        )


@pytest.mark.parametrize(
    "bindings, message",
    [
        (lambda manifests: (storage_binding(manifests[0]),), "correspond"),
        (
            lambda manifests: (
                storage_binding(manifests[0], provider="other-store"),
                storage_binding(manifests[1]),
            ),
            "provider",
        ),
        (
            lambda manifests: (
                storage_binding(manifests[0], storage_id="other-revision"),
                storage_binding(manifests[1]),
            ),
            "storage_id",
        ),
        (
            lambda manifests: (
                storage_binding(manifests[0], nbytes_delta=1),
                storage_binding(manifests[1]),
            ),
            "byte size",
        ),
    ],
)
def test_snapshot_rejects_incomplete_or_inconsistent_bindings(
    bindings,
    message: str,
) -> None:
    placements = (placement(0), placement(1))

    with pytest.raises(ValueError, match=message):
        snapshot(placements, bindings(placements))


def test_snapshot_rejects_duplicate_placement_and_binding_ids() -> None:
    first = placement(0)
    duplicate_tensors = tuple(first.tensors)
    duplicate_placement = WeightPlacementManifest(
        model_id=first.model_id,
        revision=first.revision,
        placement_id=compute_weight_placement_id(tuple(duplicate_tensors)),
        tensors=duplicate_tensors,
    )
    first_binding = storage_binding(first)

    with pytest.raises(ValueError, match="duplicate placement ID"):
        snapshot(
            (first, duplicate_placement),
            (first_binding, first_binding),
        )

    with pytest.raises(ValueError, match="duplicate storage binding"):
        snapshot(
            (first,),
            (first_binding, first_binding),
        )


def test_snapshot_rejects_overlapping_and_overflowing_object_ranges() -> None:
    placements = (placement(0), placement(1))
    overlapping = tuple(
        storage_binding(item, object_key="objects/shared", object_offset=0)
        for item in placements
    )

    with pytest.raises(ValueError, match="overlap"):
        snapshot(placements, overlapping)

    uint64_upper_half = storage_binding(
        placements[0],
        object_offset=1 << 63,
    )
    stored = snapshot((placements[0],), (uint64_upper_half,))
    assert stored.storage_bindings[0].fragments[0].object_offset == 1 << 63

    overflow = storage_binding(placements[0])
    object.__setattr__(
        overflow.fragments[0],
        "object_offset",
        (1 << 64) - 8,
    )
    with pytest.raises(ValueError, match="range"):
        snapshot((placements[0],), (overflow,))


def test_catalog_publishes_atomically_and_queries_by_full_ref() -> None:
    stored = snapshot()
    catalog = InMemoryWeightStorageCatalog()

    pending = catalog.prepare_publish("publish-1", stored)
    assert pending.state is WeightSnapshotPublicationState.PENDING
    assert catalog.get_snapshot(stored.ref) is None
    assert catalog.get_publication("publish-1") == pending
    assert catalog.recoverable_publications() == (pending,)

    published = catalog.publish("publish-1")
    assert published.state is WeightSnapshotPublicationState.PUBLISHED
    assert catalog.publish("publish-1") == published
    assert catalog.get_snapshot(stored.ref) == stored
    assert (
        catalog.get_snapshot(
            replace(stored.ref, manifest_key="weights/revision/missing.json")
        )
        is None
    )
    assert catalog.recoverable_publications() == ()


def test_catalog_abort_is_terminal_and_cannot_hide_published_snapshot() -> None:
    stored = snapshot()
    catalog = InMemoryWeightStorageCatalog()
    catalog.prepare_publish("aborted", stored)

    aborted = catalog.abort("aborted")
    assert aborted.state is WeightSnapshotPublicationState.ABORTED
    assert catalog.abort("aborted") == aborted
    assert catalog.get_snapshot(stored.ref) is None
    with pytest.raises(ValueError, match="aborted"):
        catalog.publish("aborted")

    catalog.prepare_publish("published", stored)
    catalog.publish("published")
    with pytest.raises(ValueError, match="published"):
        catalog.abort("published")
    assert catalog.get_snapshot(stored.ref) == stored


def test_catalog_restores_publication_state_for_recovery() -> None:
    pending_snapshot = snapshot()
    published_snapshot = StoredWeightSnapshot.create(
        provider="test-store",
        storage_id="weights/other-revision",
        manifest_key="weights/other-revision/manifest.json",
        placements=pending_snapshot.placements,
        storage_bindings=tuple(
            WeightStorageBindingManifest(
                model_id=item.model_id,
                revision=item.revision,
                placement_id=item.placement_id,
                storage_id="weights/other-revision",
                provider=item.provider,
                fragments=item.fragments,
            )
            for item in pending_snapshot.storage_bindings
        ),
    )
    catalog = InMemoryWeightStorageCatalog()
    pending = catalog.prepare_publish("pending", pending_snapshot)
    catalog.prepare_publish("published", published_snapshot)
    published = catalog.publish("published")

    restored = InMemoryWeightStorageCatalog(publications=catalog.export_publications())

    assert restored.get_publication("pending") == pending
    assert restored.get_publication("published") == published
    assert restored.recoverable_publications() == (pending,)
    assert restored.get_snapshot(pending_snapshot.ref) is None
    assert restored.get_snapshot(published_snapshot.ref) == published_snapshot


def test_revision_head_cas_requires_published_matching_snapshot() -> None:
    stored = snapshot()
    catalog = InMemoryWeightStorageCatalog()

    with pytest.raises(ValueError, match="published"):
        catalog.compare_and_set_revision(
            model_id="model",
            revision="revision",
            expected=None,
            new_ref=stored.ref,
            new_state=WeightRevisionState.READY,
        )

    catalog.prepare_publish("published", stored)
    catalog.publish("published")
    with pytest.raises(ValueError, match="model or revision"):
        catalog.compare_and_set_revision(
            model_id="other-model",
            revision="revision",
            expected=None,
            new_ref=stored.ref,
            new_state=WeightRevisionState.READY,
        )


def test_revision_head_cas_is_generation_checked_and_restartable() -> None:
    stored = snapshot()
    catalog = InMemoryWeightStorageCatalog()
    catalog.prepare_publish("published", stored)
    catalog.publish("published")

    ready = catalog.compare_and_set_revision(
        model_id="model",
        revision="revision",
        expected=None,
        new_ref=stored.ref,
        new_state=WeightRevisionState.READY,
    )
    assert ready == WeightRevisionHead(
        model_id="model",
        revision="revision",
        ref=stored.ref,
        generation=1,
        state=WeightRevisionState.READY,
    )
    assert (
        catalog.compare_and_set_revision(
            model_id="model",
            revision="revision",
            expected=None,
            new_ref=stored.ref,
            new_state=WeightRevisionState.SERVING,
        )
        is None
    )
    assert catalog.get_revision_head("model", "revision") == ready

    serving = catalog.compare_and_set_revision(
        model_id="model",
        revision="revision",
        expected=ready,
        new_ref=stored.ref,
        new_state=WeightRevisionState.SERVING,
    )
    assert serving is not None
    assert serving.generation == 2
    assert serving.state is WeightRevisionState.SERVING

    restored = InMemoryWeightStorageCatalog(
        publications=catalog.export_publications(),
        revision_heads=catalog.export_revision_heads(),
    )
    assert restored.get_revision_head("model", "revision") == serving


def test_revision_head_is_create_only_for_a_model_revision() -> None:
    first = snapshot()
    second = StoredWeightSnapshot.create(
        provider="test-store",
        storage_id="weights/second-copy",
        manifest_key="weights/second-copy/manifest.json",
        placements=first.placements,
        storage_bindings=tuple(
            WeightStorageBindingManifest(
                model_id=binding.model_id,
                revision=binding.revision,
                placement_id=binding.placement_id,
                storage_id="weights/second-copy",
                provider=binding.provider,
                fragments=binding.fragments,
            )
            for binding in first.storage_bindings
        ),
    )
    catalog = InMemoryWeightStorageCatalog()
    for publication_id, stored in (("first", first), ("second", second)):
        catalog.prepare_publish(publication_id, stored)
        catalog.publish(publication_id)
    ready = catalog.compare_and_set_revision(
        model_id="model",
        revision="revision",
        expected=None,
        new_ref=first.ref,
        new_state=WeightRevisionState.READY,
    )

    with pytest.raises(ValueError, match="immutable"):
        catalog.compare_and_set_revision(
            model_id="model",
            revision="revision",
            expected=ready,
            new_ref=second.ref,
            new_state=WeightRevisionState.READY,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
