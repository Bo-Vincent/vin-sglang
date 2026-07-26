from __future__ import annotations

import json
from pathlib import Path

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
    StoredWeightSnapshot,
    WeightMaterializationAttemptState,
    WeightMaterializationIntent,
    WeightRevisionState,
    WeightSnapshotPublicationState,
    weight_placement_set_digest,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _catalog(path: Path):
    from sglang.srt.weight_transfer.storage_file import (
        FileWeightStorageCatalog,
    )

    return FileWeightStorageCatalog(path)


def _snapshot(index: int) -> StoredWeightSnapshot:
    storage_id = f"weights/revision-{index}"
    placement_fragment_id = f"placement-{index}:fragment"
    tensors = (
        WeightPlacementTensor(
            placement_fragment_id=placement_fragment_id,
            tensor_id=f"weight-{index}",
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
        ),
    )
    placement = WeightPlacementManifest(
        model_id="model",
        revision=f"revision-{index}",
        placement_id=compute_weight_placement_id(tuple(tensors)),
        tensors=tensors,
    )
    binding = WeightStorageBindingManifest(
        model_id=placement.model_id,
        revision=placement.revision,
        placement_id=placement.placement_id,
        storage_id=storage_id,
        provider="test-store",
        fragments=(
            WeightStorageFragmentBinding(
                placement_fragment_id=placement_fragment_id,
                fragment_id=f"stored:{placement_fragment_id}",
                object_key=f"{storage_id}/payload/{placement_fragment_id}",
                object_offset=0,
                nbytes=16,
                checksum="sha256:" + f"{index + 1:x}" * 64,
            ),
        ),
    )
    return StoredWeightSnapshot.create(
        provider="test-store",
        storage_id=storage_id,
        manifest_key=f"{storage_id}/manifest.json",
        placements=(placement,),
        storage_bindings=(binding,),
    )


def _intent(snapshot: StoredWeightSnapshot) -> WeightMaterializationIntent:
    fragments = tuple(
        fragment
        for binding in snapshot.storage_bindings
        for fragment in binding.fragments
    )
    placement = snapshot.placements[0]
    return WeightMaterializationIntent(
        provider=snapshot.ref.provider,
        storage_id=snapshot.ref.storage_id,
        object_prefix=snapshot.ref.storage_id,
        model_id=placement.model_id,
        revision=placement.revision,
        source_digest=weight_placement_set_digest(snapshot.placements),
        total_bytes=sum(fragment.nbytes for fragment in fragments),
        fragment_count=len(fragments),
    )


def test_reopens_preparing_materialization_with_completion_ticket(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.json"
    snapshot = _snapshot(0)
    catalog = _catalog(path)
    catalog.begin_materialization("materialize-0", _intent(snapshot))
    expected = catalog.set_materialization_completion_ticket(
        "materialize-0",
        "ticket:materialize-0",
    )

    reopened = _catalog(path)

    assert expected.state is WeightMaterializationAttemptState.PREPARING
    assert reopened.get_materialization("materialize-0") == expected
    assert reopened.recoverable_materializations() == (expected,)


def test_reopens_materialized_pending_and_published_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.json"
    pending_snapshot = _snapshot(0)
    published_snapshot = _snapshot(1)
    catalog = _catalog(path)
    catalog.begin_materialization(
        "materialize-1",
        _intent(published_snapshot),
    )
    materialized = catalog.complete_materialization(
        "materialize-1",
        published_snapshot,
    )
    pending = catalog.prepare_publish("publish-0", pending_snapshot)
    catalog.prepare_publish("publish-1", published_snapshot)
    published = catalog.publish("publish-1")

    reopened = _catalog(path)

    assert materialized.state is WeightMaterializationAttemptState.MATERIALIZED
    assert reopened.get_materialization("materialize-1") == materialized
    assert reopened.get_publication("publish-0") == pending
    assert reopened.get_publication("publish-1") == published
    assert reopened.recoverable_publications() == (pending,)
    assert reopened.get_snapshot(pending_snapshot.ref) is None
    assert reopened.get_snapshot(published_snapshot.ref) == published_snapshot
    assert published.state is WeightSnapshotPublicationState.PUBLISHED


def test_rejects_materialized_snapshot_from_another_intent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.json"
    catalog = _catalog(path)
    first_snapshot = _snapshot(0)
    second_snapshot = _snapshot(1)
    for index, stored in enumerate((first_snapshot, second_snapshot)):
        materialization_id = f"materialize-{index}"
        catalog.begin_materialization(materialization_id, _intent(stored))
        catalog.complete_materialization(materialization_id, stored)

    payload = json.loads(path.read_bytes())
    payload["materializations"][0]["snapshot"] = payload["materializations"][1][
        "snapshot"
    ]
    encoded = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(encoded)

    with pytest.raises(ValueError, match="invalid catalog state"):
        _catalog(path)

    assert path.read_bytes() == encoded


def test_two_existing_instances_reload_before_writes_without_lost_updates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.json"
    first = _catalog(path)
    second = _catalog(path)
    first_snapshot = _snapshot(0)
    second_snapshot = _snapshot(1)

    first.begin_materialization("materialize-0", _intent(first_snapshot))
    second.begin_materialization("materialize-1", _intent(second_snapshot))
    first.set_materialization_completion_ticket(
        "materialize-0",
        "ticket:materialize-0",
    )
    second.prepare_publish("publish-1", second_snapshot)
    first.prepare_publish("publish-0", first_snapshot)

    reopened = _catalog(path)

    assert reopened.get_materialization("materialize-0") is not None
    assert reopened.get_materialization("materialize-1") is not None
    assert reopened.get_publication("publish-0") is not None
    assert reopened.get_publication("publish-1") is not None


def test_revision_head_cas_survives_restart_and_concurrent_instances(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.json"
    first = _catalog(path)
    second = _catalog(path)
    stored = _snapshot(0)
    first.prepare_publish("publish-0", stored)
    first.publish("publish-0")

    ready = second.compare_and_set_revision(
        model_id="model",
        revision="revision-0",
        expected=None,
        new_ref=stored.ref,
        new_state=WeightRevisionState.READY,
    )
    assert ready is not None
    assert (
        first.compare_and_set_revision(
            model_id="model",
            revision="revision-0",
            expected=None,
            new_ref=stored.ref,
            new_state=WeightRevisionState.SERVING,
        )
        is None
    )
    serving = first.compare_and_set_revision(
        model_id="model",
        revision="revision-0",
        expected=ready,
        new_ref=stored.ref,
        new_state=WeightRevisionState.SERVING,
    )

    reopened = _catalog(path)
    assert serving is not None
    assert serving.generation == 2
    assert reopened.get_revision_head("model", "revision-0") == serving
    assert reopened.export_revision_heads() == (serving,)


def test_failed_atomic_replace_preserves_previous_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.weight_transfer import storage_file

    path = tmp_path / "catalog.json"
    catalog = _catalog(path)
    catalog.begin_materialization("materialize-0", _intent(_snapshot(0)))
    original = path.read_bytes()

    def fail_replace(source, destination):
        del source, destination
        raise OSError("replace unavailable")

    monkeypatch.setattr(storage_file.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace unavailable"):
        catalog.begin_materialization("materialize-1", _intent(_snapshot(1)))

    assert path.read_bytes() == original
    assert catalog.get_materialization("materialize-0") is not None
    assert catalog.get_materialization("materialize-1") is None
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_post_replace_directory_fsync_failure_reconciles_committed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "catalog.json"
    catalog = _catalog(path)
    stored = _snapshot(0)
    catalog.prepare_publish("publish-0", stored)
    catalog.publish("publish-0")
    ready = catalog.compare_and_set_revision(
        model_id="model",
        revision="revision-0",
        expected=None,
        new_ref=stored.ref,
        new_state=WeightRevisionState.READY,
    )
    assert ready is not None

    monkeypatch.setattr(
        catalog,
        "_fsync_directory",
        lambda: (_ for _ in ()).throw(OSError("directory fsync unavailable")),
    )
    serving = catalog.compare_and_set_revision(
        model_id="model",
        revision="revision-0",
        expected=ready,
        new_ref=stored.ref,
        new_state=WeightRevisionState.SERVING,
    )

    assert serving is not None
    assert serving.state is WeightRevisionState.SERVING
    assert _catalog(path).get_revision_head("model", "revision-0") == serving


def test_corrupt_catalog_fails_closed_without_using_cached_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "catalog.json"
    catalog = _catalog(path)
    catalog.begin_materialization("materialize-0", _intent(_snapshot(0)))
    path.write_bytes(b"{not-json")
    corrupt = path.read_bytes()

    with pytest.raises(ValueError, match="invalid catalog"):
        catalog.get_materialization("materialize-0")

    assert path.read_bytes() == corrupt


@pytest.mark.parametrize(
    "payload",
    [
        {
            "format": "unknown-catalog",
            "version": 1,
            "materializations": [],
            "publications": [],
        },
        {
            "format": "sglang-weight-storage-catalog",
            "version": 2,
            "materializations": [],
            "publications": [],
        },
    ],
)
def test_unknown_catalog_format_fails_closed(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    path = tmp_path / "catalog.json"
    encoded = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(encoded)

    with pytest.raises(ValueError, match="unsupported catalog"):
        _catalog(path)

    assert path.read_bytes() == encoded


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
