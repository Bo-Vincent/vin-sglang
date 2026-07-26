from __future__ import annotations

from dataclasses import replace

import msgspec
import pytest

from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    WeightSnapshotCoordinator,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.api import (
    load_weight_snapshot,
    materialize_weight_snapshot,
)
from sglang.srt.weight_transfer.provider import (
    LocalWeightBufferRegistry,
    LocalWeightTransferProvider,
    WeightTargetLoadMode,
    WeightTargetLoadSession,
    WeightTargetLoadState,
    WeightTransferTerminalProof,
    WeightTransferTerminalStatus,
    WeightStorageDestination,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
)
from sglang.srt.weight_transfer.storage import (
    InMemoryWeightStorageCatalog,
    WeightMaterializationAttemptState,
    WeightRevisionState,
    WeightSnapshotPublicationState,
    weight_source_snapshot_digest,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def placement(side: str) -> WeightPlacementManifest:
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
            layout_fingerprint="layout:v1",
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


def binding(
    manifest: WeightPlacementManifest,
    address: int,
    *,
    generation: int = 1,
    lease_id: str | None = None,
) -> WeightRuntimeBindingManifest:
    tensor = manifest.tensors[0]
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id=f"{manifest.placement_id}:instance",
        generation=generation,
        lease_id=lease_id or f"{manifest.placement_id}:lease",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"{tensor.placement_fragment_id}:runtime",
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


def target_binding(
    coordinator: WeightSnapshotCoordinator,
    manifest: WeightPlacementManifest,
    address: int,
    *,
    full_restore: bool = False,
) -> WeightRuntimeBindingManifest:
    lease_id, generation = coordinator.acquire_target_snapshot(
        full_restore=full_restore
    )
    return binding(
        manifest,
        address,
        generation=generation,
        lease_id=lease_id,
    )


def test_materialized_snapshot_reopens_by_ref_and_loads_target() -> None:
    source = placement("source")
    target = placement("target")
    source_binding = binding(source, 0x10000)
    coordinator = WeightSnapshotCoordinator()
    target_runtime_binding = target_binding(coordinator, target, 0x20000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    registry.register_runtime(0x20000, bytes(8))
    provider = LocalWeightTransferProvider(registry)
    catalog = InMemoryWeightStorageCatalog()
    target_session = WeightTargetLoadSession(
        target_bindings=(target_runtime_binding,),
        owners=(registry,),
        coordinator=coordinator,
    )

    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
        catalog=catalog,
        publication_id="publish-1",
    )

    assert publication.state is WeightSnapshotPublicationState.PUBLISHED
    assert catalog.get_snapshot(publication.snapshot.ref) == publication.snapshot
    ready = catalog.get_revision_head(source.model_id, source.revision)
    assert ready is not None
    assert ready.ref == publication.snapshot.ref
    assert ready.state is WeightRevisionState.READY

    receipt = load_weight_snapshot(
        publication.snapshot.ref,
        catalog=catalog,
        target_placements=(target,),
        target_bindings=(target_runtime_binding,),
        provider=provider,
        target_mode=WeightTargetLoadMode.LIVE_UPDATE,
        target_session=target_session,
    )

    assert receipt.total_bytes == 8
    assert registry.read_runtime(0x20000, 8) == bytes(range(8))
    assert target_session.state is WeightTargetLoadState.TRANSFERRED
    with pytest.raises(Exception, match="weight update is in progress"):
        coordinator.acquire_snapshot()
    pending_generation = target_session.mark_ready()
    assert target_session.state is WeightTargetLoadState.READY
    with pytest.raises(Exception, match="explicit revision commit"):
        coordinator.acquire_snapshot()
    assert target_session.activate() == pending_generation
    assert target_session.require_ready() is receipt
    lease_id, _ = coordinator.acquire_snapshot()
    coordinator.release_snapshot(lease_id)
    target_session.fail(RuntimeError("peer rank failed after local activation"))
    assert target_session.state is WeightTargetLoadState.POISONED
    with pytest.raises(Exception, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_snapshot_load_rejects_published_ref_without_ready_head() -> None:
    source = placement("source")
    target = placement("target")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    registry.register_runtime(0x20000, bytes(8))
    provider = LocalWeightTransferProvider(registry)
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
        catalog=catalog,
        publication_id="published-without-ready-head",
    )
    unready_catalog = InMemoryWeightStorageCatalog(
        materializations=catalog.export_materializations(),
        publications=catalog.export_publications(),
    )

    with pytest.raises(ValueError, match="READY or SERVING"):
        load_weight_snapshot(
            publication.snapshot.ref,
            catalog=unready_catalog,
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )


def test_materialization_retry_repairs_published_snapshot_ready_head() -> None:
    source = placement("source")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = LocalWeightTransferProvider(registry)
    catalog = InMemoryWeightStorageCatalog()
    destination = WeightStorageDestination(
        provider="local",
        storage_id="weights/revision",
        object_prefix="weights/revision",
    )
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=destination,
        provider=provider,
        catalog=catalog,
        publication_id="repair-ready-head",
    )
    recovered = InMemoryWeightStorageCatalog(
        materializations=catalog.export_materializations(),
        publications=catalog.export_publications(),
    )
    assert recovered.get_revision_head(source.model_id, source.revision) is None

    replay = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=destination,
        provider=provider,
        catalog=recovered,
        publication_id="repair-ready-head",
    )

    assert replay == publication
    ready = recovered.get_revision_head(source.model_id, source.revision)
    assert ready is not None
    assert ready.ref == publication.snapshot.ref
    assert ready.state is WeightRevisionState.READY


def test_target_session_rejects_forged_receipt_and_poisons_runtime() -> None:
    class ForgedReceiptProvider(LocalWeightTransferProvider):
        def wait(self, submission):
            receipt = super().wait(submission)
            return replace(receipt, total_bytes=receipt.total_bytes + 1)

    source = placement("source")
    target = placement("target")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    registry.register_runtime(0x20000, bytes(8))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="forged-receipt",
    )
    coordinator = WeightSnapshotCoordinator()
    target_runtime_binding = target_binding(coordinator, target, 0x20000)
    session = WeightTargetLoadSession(
        target_bindings=(target_runtime_binding,),
        owners=(registry,),
        coordinator=coordinator,
    )

    with pytest.raises(ValueError, match="completion is invalid"):
        load_weight_snapshot(
            publication.snapshot.ref,
            catalog=catalog,
            target_placements=(target,),
            target_bindings=(target_runtime_binding,),
            provider=ForgedReceiptProvider(registry),
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
            target_session=session,
        )

    assert session.state is WeightTargetLoadState.POISONED
    assert session.update_token is None
    with pytest.raises(Exception, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_target_completion_fence_failure_clears_token_and_poisons() -> None:
    source = placement("source")
    target = placement("target")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    registry.register_runtime(0x20000, bytes(8))
    provider = LocalWeightTransferProvider(registry)
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
        catalog=catalog,
        publication_id="completion-fence-failure",
    )
    fence_calls = 0

    def completion_fence() -> None:
        nonlocal fence_calls
        fence_calls += 1
        if fence_calls == 2:
            raise RuntimeError("target device fence failed")

    coordinator = WeightSnapshotCoordinator(completion_fence=completion_fence)
    target_runtime_binding = target_binding(coordinator, target, 0x20000)
    session = WeightTargetLoadSession(
        target_bindings=(target_runtime_binding,),
        owners=(registry,),
        coordinator=coordinator,
    )
    load_weight_snapshot(
        publication.snapshot.ref,
        catalog=catalog,
        target_placements=(target,),
        target_bindings=(target_runtime_binding,),
        provider=provider,
        target_mode=WeightTargetLoadMode.LIVE_UPDATE,
        target_session=session,
    )

    with pytest.raises(RuntimeError, match="target device fence failed"):
        session.mark_ready()

    assert session.state is WeightTargetLoadState.POISONED
    assert session.update_token is None
    assert coordinator.generation == 2
    with pytest.raises(Exception, match="full successful weight restore"):
        coordinator.acquire_snapshot()


def test_catalog_publish_failure_keeps_recoverable_snapshot_without_rewrite() -> None:
    class FailOnceCatalog(InMemoryWeightStorageCatalog):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def publish(self, publication_id):
            if self.fail:
                self.fail = False
                raise RuntimeError("catalog publish unavailable")
            return super().publish(publication_id)

    source = placement("source")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = LocalWeightTransferProvider(registry)
    catalog = FailOnceCatalog()

    with pytest.raises(RuntimeError, match="catalog publish unavailable"):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(source_binding,),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/revision",
                object_prefix="weights/revision",
            ),
            provider=provider,
            catalog=catalog,
            publication_id="publish-1",
        )

    pending = catalog.get_publication("publish-1")
    assert pending is not None
    assert pending.state is WeightSnapshotPublicationState.PENDING
    stored_object_count = len(registry.storage_objects)

    published = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
        catalog=catalog,
        publication_id="publish-1",
    )

    assert published.state is WeightSnapshotPublicationState.PUBLISHED
    assert len(registry.storage_objects) == stored_object_count


def test_release_failure_keeps_committed_snapshot_recoverable() -> None:
    class FailReleaseOnceProvider(LocalWeightTransferProvider):
        def __init__(self, registry):
            super().__init__(registry)
            self.fail_release = True

        def release(self, prepared, receipt):
            if self.fail_release:
                self.fail_release = False
                raise RuntimeError("finalize unavailable")
            return super().release(prepared, receipt)

    source = placement("source")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = FailReleaseOnceProvider(registry)
    catalog = InMemoryWeightStorageCatalog()
    destination = WeightStorageDestination(
        provider=provider.name,
        storage_id="weights/revision",
        object_prefix="weights/revision",
    )

    with pytest.raises(WeightTransferError, match="finalize unavailable") as raised:
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(source_binding,),
            destination=destination,
            provider=provider,
            catalog=catalog,
            publication_id="publish-release",
        )

    assert raised.value.code == "RELEASE_FAILED"
    assert raised.value.completion_known is True
    pending = catalog.get_publication("publish-release")
    assert pending is not None
    assert pending.state is WeightSnapshotPublicationState.PENDING
    stored_object_count = len(registry.storage_objects)

    published = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=destination,
        provider=provider,
        catalog=catalog,
        publication_id="publish-release",
    )

    assert published.state is WeightSnapshotPublicationState.PUBLISHED
    assert len(registry.storage_objects) == stored_object_count


def test_materialization_journal_precedes_provider_probe() -> None:
    events = []

    class RecordingCatalog(InMemoryWeightStorageCatalog):
        def begin_materialization(self, materialization_id, intent):
            events.append("begin")
            return super().begin_materialization(materialization_id, intent)

    class RecordingProvider(LocalWeightTransferProvider):
        def probe(self, request):
            events.append("probe")
            return super().probe(request)

    source = placement("source")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = RecordingProvider(registry)

    materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
        catalog=RecordingCatalog(),
        publication_id="journal-order",
    )

    assert events[:2] == ["begin", "probe"]


def test_commit_before_catalog_completion_recovers_without_rewriting() -> None:
    class FailCompleteOnceCatalog(InMemoryWeightStorageCatalog):
        def __init__(self):
            super().__init__()
            self.fail = True

        def complete_materialization(self, materialization_id, snapshot):
            if self.fail:
                self.fail = False
                raise RuntimeError("catalog completion unavailable")
            return super().complete_materialization(materialization_id, snapshot)

    source = placement("source")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = LocalWeightTransferProvider(registry)
    catalog = FailCompleteOnceCatalog()
    destination = WeightStorageDestination(
        provider=provider.name,
        storage_id="weights/revision",
        object_prefix="weights/revision",
    )

    with pytest.raises(RuntimeError, match="catalog completion unavailable"):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(source_binding,),
            destination=destination,
            provider=provider,
            catalog=catalog,
            publication_id="commit-before-catalog",
        )

    attempt = catalog.get_materialization("commit-before-catalog")
    assert attempt is not None
    assert attempt.state is WeightMaterializationAttemptState.PREPARING
    stored_object_count = len(registry.storage_objects)

    published = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=destination,
        provider=provider,
        catalog=catalog,
        publication_id="commit-before-catalog",
    )

    assert published.state is WeightSnapshotPublicationState.PUBLISHED
    assert len(registry.storage_objects) == stored_object_count
    assert (
        catalog.get_materialization("commit-before-catalog").state
        is WeightMaterializationAttemptState.MATERIALIZED
    )


def test_completion_unknown_persists_recovery_ticket_without_aborting() -> None:
    class CompletionUnknownProvider(LocalWeightTransferProvider):
        def wait(self, submission):
            super().wait(submission)
            raise WeightTransferCompletionUnknownError(
                "commit response lost",
                provider=self.name,
                phase="wait",
                operation_id=submission.request.operation_id,
                completion_ticket="ticket:commit-response-lost",
            )

    source = placement("source")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    catalog = InMemoryWeightStorageCatalog()

    with pytest.raises(WeightTransferCompletionUnknownError):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(source_binding,),
            destination=WeightStorageDestination(
                provider="local",
                storage_id="weights/revision",
                object_prefix="weights/revision",
            ),
            provider=CompletionUnknownProvider(registry),
            catalog=catalog,
            publication_id="completion-unknown",
        )

    attempt = catalog.get_materialization("completion-unknown")
    assert attempt is not None
    assert attempt.state is WeightMaterializationAttemptState.PREPARING
    assert attempt.completion_ticket == "ticket:commit-response-lost"
    assert attempt.intent.source_snapshot_digest == weight_source_snapshot_digest(
        (source,),
        (source_binding,),
    )
    assert catalog.get_publication("completion-unknown") is None


def test_source_snapshot_digest_binds_generation_lease_and_fragments_not_address() -> (
    None
):
    source = placement("source")
    first = binding(source, 0x10000)
    relocated = msgspec.structs.replace(
        first,
        fragments=(
            msgspec.structs.replace(
                first.fragments[0],
                address=0x20000,
            ),
        ),
    )
    next_generation = msgspec.structs.replace(
        relocated,
        generation=2,
        lease_id="source:lease:2",
        fragments=(
            msgspec.structs.replace(
                relocated.fragments[0],
                fragment_id="source:fragment:generation-2",
            ),
        ),
    )

    first_digest = weight_source_snapshot_digest((source,), (first,))

    assert weight_source_snapshot_digest((source,), (relocated,)) == first_digest
    assert weight_source_snapshot_digest((source,), (next_generation,)) != first_digest


def test_load_rejects_unpublished_or_wrong_snapshot_ref_before_transfer() -> None:
    source = placement("source")
    target = placement("target")
    source_binding = binding(source, 0x10000)
    coordinator = WeightSnapshotCoordinator()
    target_runtime_binding = target_binding(coordinator, target, 0x20000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    registry.register_runtime(0x20000, bytes(8))
    provider = LocalWeightTransferProvider(registry)
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
        catalog=catalog,
        publication_id="publish-1",
    )
    wrong_ref = replace(
        publication.snapshot.ref,
        manifest_key="weights/revision/other-manifest",
    )

    with pytest.raises(ValueError, match="published weight snapshot"):
        load_weight_snapshot(
            wrong_ref,
            catalog=catalog,
            target_placements=(target,),
            target_bindings=(target_runtime_binding,),
            provider=provider,
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
            target_session=WeightTargetLoadSession(
                target_bindings=(target_runtime_binding,),
                owners=(registry,),
                coordinator=coordinator,
            ),
        )

    assert registry.read_runtime(0x20000, 8) == bytes(8)


def test_partial_snapshot_load_poisoned_target_cannot_be_activated() -> None:
    class PartialLoadProvider(LocalWeightTransferProvider):
        def _execute_load(self, request):
            target = request.plan.regions[0].target
            self.registry.write_runtime(target.address, b"\x11\x22\x33\x44")
            raise RuntimeError("second range failed")

    source = placement("source")
    target = placement("target")
    source_binding = binding(source, 0x10000)
    coordinator = WeightSnapshotCoordinator()
    target_runtime_binding = target_binding(coordinator, target, 0x20000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    registry.register_runtime(0x20000, bytes(8))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="partial-load",
    )
    session = WeightTargetLoadSession(
        target_bindings=(target_runtime_binding,),
        owners=(registry,),
        coordinator=coordinator,
    )

    with pytest.raises(WeightTransferError, match="second range failed") as raised:
        load_weight_snapshot(
            publication.snapshot.ref,
            catalog=catalog,
            target_placements=(target,),
            target_bindings=(target_runtime_binding,),
            provider=PartialLoadProvider(registry),
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
            target_session=session,
        )

    assert raised.value.completion_known is True
    assert session.state is WeightTargetLoadState.POISONED
    assert session.failure is raised.value
    assert session.owners == (registry,)
    assert registry.read_runtime(0x20000, 8) == b"\x11\x22\x33\x44\x00\x00\x00\x00"
    with pytest.raises(Exception, match="full successful weight restore"):
        session.coordinator.acquire_snapshot()
    with pytest.raises(RuntimeError, match="not ready"):
        session.require_ready()

    restore_binding = target_binding(
        coordinator,
        target,
        0x20000,
        full_restore=True,
    )
    incremental_session = WeightTargetLoadSession(
        target_bindings=(restore_binding,),
        owners=(registry,),
        coordinator=coordinator,
        full_restore=False,
    )
    with pytest.raises(Exception, match="restore-only target binding"):
        load_weight_snapshot(
            publication.snapshot.ref,
            catalog=catalog,
            target_placements=(target,),
            target_bindings=(restore_binding,),
            provider=LocalWeightTransferProvider(registry),
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
            target_session=incremental_session,
        )
    coordinator.release_snapshot(restore_binding.lease_id)

    restore_binding = target_binding(
        coordinator,
        target,
        0x20000,
        full_restore=True,
    )
    restore_session = WeightTargetLoadSession(
        target_bindings=(restore_binding,),
        owners=(registry,),
        coordinator=coordinator,
        full_restore=True,
    )
    restored = load_weight_snapshot(
        publication.snapshot.ref,
        catalog=catalog,
        target_placements=(target,),
        target_bindings=(restore_binding,),
        provider=LocalWeightTransferProvider(registry),
        target_mode=WeightTargetLoadMode.LIVE_UPDATE,
        target_session=restore_session,
    )
    restored_generation = restore_session.mark_ready()
    assert restore_session.activate() == restored_generation
    assert restore_session.require_ready() is restored
    assert registry.read_runtime(0x20000, 8) == bytes(range(8))
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == restored_generation
    coordinator.release_snapshot(lease_id)


def test_prewrite_load_failure_aborts_without_poisoning_target() -> None:
    class RejectAttestor:
        def attest(self, _request) -> None:
            raise RuntimeError("source lease rejected")

    source = placement("source")
    target = placement("target")
    source_binding = binding(source, 0x10000)
    coordinator = WeightSnapshotCoordinator()
    target_runtime_binding = target_binding(coordinator, target, 0x20000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    registry.register_runtime(0x20000, bytes(8))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="prewrite-failure",
    )
    session = WeightTargetLoadSession(
        target_bindings=(target_runtime_binding,),
        owners=(registry,),
        coordinator=coordinator,
    )

    with pytest.raises(RuntimeError, match="source lease rejected"):
        load_weight_snapshot(
            publication.snapshot.ref,
            catalog=catalog,
            target_placements=(target,),
            target_bindings=(target_runtime_binding,),
            provider=LocalWeightTransferProvider(registry),
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
            target_session=session,
            attestor=RejectAttestor(),
        )

    assert session.state is WeightTargetLoadState.ABORTED
    assert registry.read_runtime(0x20000, 8) == bytes(8)
    lease_id, generation = coordinator.acquire_snapshot()
    assert generation == 1
    coordinator.release_snapshot(lease_id)


def test_completion_unknown_snapshot_load_quarantines_target_owner() -> None:
    class UnknownLoadProvider(LocalWeightTransferProvider):
        def wait(self, submission):
            super().wait(submission)
            raise WeightTransferCompletionUnknownError(
                "load response lost",
                provider=self.name,
                phase="wait",
                operation_id=submission.request.operation_id,
                completion_ticket="load-ticket",
            )

    source = placement("source")
    target = placement("target")
    source_binding = binding(source, 0x10000)
    coordinator = WeightSnapshotCoordinator()
    target_runtime_binding = target_binding(coordinator, target, 0x20000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    registry.register_runtime(0x20000, bytes(8))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="unknown-load",
    )
    session = WeightTargetLoadSession(
        target_bindings=(target_runtime_binding,),
        owners=(registry,),
        coordinator=coordinator,
    )

    with pytest.raises(WeightTransferCompletionUnknownError):
        load_weight_snapshot(
            publication.snapshot.ref,
            catalog=catalog,
            target_placements=(target,),
            target_bindings=(target_runtime_binding,),
            provider=UnknownLoadProvider(registry),
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
            target_session=session,
        )

    assert session.state is WeightTargetLoadState.QUARANTINED
    assert session.completion_ticket == "load-ticket"
    assert session.owners == (registry,)
    with pytest.raises(Exception, match="weight update is in progress"):
        session.coordinator.acquire_snapshot()
    with pytest.raises(RuntimeError, match="not ready"):
        session.require_ready()
    pending_generation = session.resolve_quarantine(
        WeightTransferTerminalProof(
            operation_id=session.operation_id,
            provider="local",
            completion_ticket="load-ticket",
            status=WeightTransferTerminalStatus.FAILED,
        )
    )
    assert pending_generation == 2
    assert session.state is WeightTargetLoadState.POISONED
    with pytest.raises(Exception, match="full successful weight restore"):
        session.coordinator.acquire_snapshot()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
