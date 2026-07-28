from __future__ import annotations

import hashlib
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
    materialize_weight_snapshot_candidate,
    preflight_weight_transfer,
    prepare_weight_materialization,
    publish_weight_snapshot,
)
from sglang.srt.weight_transfer.provider import (
    LocalWeightBufferRegistry,
    LocalWeightTransferProvider,
    WeightPayloadIdentity,
    WeightStorageDestination,
    WeightTargetLoadMode,
    WeightTargetLoadSession,
    WeightTargetLoadState,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
    WeightTransferReleaseError,
    WeightTransferTerminalProof,
    WeightTransferTerminalStatus,
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


def test_materialized_candidate_is_not_loadable_before_publication() -> None:
    source = placement("source")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = LocalWeightTransferProvider(registry)
    catalog = InMemoryWeightStorageCatalog()

    candidate = materialize_weight_snapshot_candidate(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
        catalog=catalog,
        publication_id="candidate-1",
    )

    assert candidate.state is WeightMaterializationAttemptState.MATERIALIZED
    assert candidate.snapshot is not None
    assert catalog.get_snapshot(candidate.snapshot.ref) is None
    assert catalog.get_publication(candidate.materialization_id) is None
    assert catalog.get_revision_head(source.model_id, source.revision) is None

    publication = publish_weight_snapshot(candidate, catalog=catalog)

    assert publication.state is WeightSnapshotPublicationState.PUBLISHED
    assert catalog.get_snapshot(candidate.snapshot.ref) == candidate.snapshot
    head = catalog.get_revision_head(source.model_id, source.revision)
    assert head is not None
    assert head.ref == candidate.snapshot.ref
    assert head.state is WeightRevisionState.READY


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


def test_target_session_rejects_forged_receipt_and_quarantines_runtime() -> None:
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

    with pytest.raises(
        WeightTransferCompletionUnknownError,
        match="receipt",
    ) as raised:
        load_weight_snapshot(
            publication.snapshot.ref,
            catalog=catalog,
            target_placements=(target,),
            target_bindings=(target_runtime_binding,),
            provider=ForgedReceiptProvider(registry),
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
            target_session=session,
        )

    assert raised.value.code == "COMPLETION_UNKNOWN"
    assert session.state is WeightTargetLoadState.QUARANTINED
    assert session.update_token is not None
    assert session.owners == (registry,)
    with pytest.raises(Exception, match="weight update is in progress"):
        coordinator.acquire_snapshot()


@pytest.mark.parametrize("manifest_key", ("", "weights/revision"))
def test_materialization_rejects_non_descendant_manifest_locator_as_unknown(
    manifest_key: str,
) -> None:
    class InvalidManifestProvider(LocalWeightTransferProvider):
        def __init__(self, registry):
            super().__init__(registry)
            self.cancel_calls = 0
            self.release_calls = 0

        def wait(self, submission):
            receipt = super().wait(submission)
            return replace(receipt, manifest_key=manifest_key)

        def cancel(self, submission):
            self.cancel_calls += 1
            return super().cancel(submission)

        def release(self, prepared, receipt, *, execution_context=None):
            self.release_calls += 1
            return super().release(
                prepared,
                receipt,
                execution_context=execution_context,
            )

    source = placement("source")
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = InvalidManifestProvider(registry)
    catalog = InMemoryWeightStorageCatalog()

    with pytest.raises(WeightTransferCompletionUnknownError):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/revision",
                object_prefix="weights/revision",
            ),
            provider=provider,
            catalog=catalog,
            publication_id=f"invalid-manifest-{manifest_key!r}",
        )

    assert provider.cancel_calls == 0
    assert provider.release_calls == 0
    attempt = catalog.get_materialization(f"invalid-manifest-{manifest_key!r}")
    assert attempt is not None
    assert attempt.state is WeightMaterializationAttemptState.PREPARING


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


def test_revision_cas_loser_rolls_back_its_publication() -> None:
    source = placement("source")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = LocalWeightTransferProvider(registry)
    catalog = InMemoryWeightStorageCatalog()

    winner = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/winner",
            object_prefix="weights/winner",
        ),
        provider=provider,
        catalog=catalog,
        publication_id="winner",
    )

    with pytest.raises(ValueError, match="READY"):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(source_binding,),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/loser",
                object_prefix="weights/loser",
            ),
            provider=provider,
            catalog=catalog,
            publication_id="loser",
        )

    loser = catalog.get_publication("loser")
    assert loser is not None
    assert loser.state is WeightSnapshotPublicationState.ABORTED
    assert catalog.get_snapshot(loser.snapshot.ref) is None
    head = catalog.get_revision_head(source.model_id, source.revision)
    assert head is not None
    assert head.ref == winner.snapshot.ref


def test_snapshot_materialization_reuses_preflight_for_the_same_request() -> None:
    class CountingProvider(LocalWeightTransferProvider):
        requires_runtime_attestation = True

        def __init__(self, registry):
            super().__init__(registry)
            self.probe_calls = 0

        def probe(self, request):
            self.probe_calls += 1
            return super().probe(request)

    class CountingAttestor:
        def __init__(self):
            self.calls = 0

        def attest(self, request):
            assert request.profile == "runtime_to_storage"
            self.calls += 1

    source = placement("source")
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = CountingProvider(registry)
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/preflight",
            object_prefix="weights/preflight",
        ),
        operation_id="snapshot-preflight",
    )
    attestor = CountingAttestor()
    preflight = preflight_weight_transfer(
        provider,
        request,
        attestor=attestor,
    )

    publication = materialize_weight_snapshot(
        request,
        provider=provider,
        catalog=InMemoryWeightStorageCatalog(),
        preflight=preflight,
    )

    assert publication.publication_id == request.operation_id
    assert provider.probe_calls == 1
    assert attestor.calls == 2


def test_snapshot_materialization_rejects_preflight_for_another_request() -> None:
    source = placement("source")
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = LocalWeightTransferProvider(registry)
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/preflight",
            object_prefix="weights/preflight",
        ),
        operation_id="snapshot-preflight",
    )
    preflight = preflight_weight_transfer(provider, request)
    catalog = InMemoryWeightStorageCatalog()

    with pytest.raises(ValueError, match="preflight"):
        materialize_weight_snapshot(
            replace(request, operation_id="another-operation"),
            provider=provider,
            catalog=catalog,
            preflight=preflight,
        )

    assert catalog.get_materialization("another-operation") is None


def test_release_failure_keeps_committed_snapshot_recoverable() -> None:
    class FailReleaseOnceProvider(LocalWeightTransferProvider):
        def __init__(self, registry):
            super().__init__(registry)
            self.fail_release = True
            self.execute_calls = 0
            self.release_calls = 0

        def _execute_materialize(self, request):
            self.execute_calls += 1
            return super()._execute_materialize(request)

        def release(self, prepared, receipt):
            self.release_calls += 1
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
    attempt = catalog.get_materialization("publish-release")
    assert attempt is not None
    assert attempt.state is WeightMaterializationAttemptState.PREPARING
    assert catalog.get_publication("publish-release") is None
    stored_object_count = len(registry.storage_objects)
    assert provider.execute_calls == 1
    assert provider.release_calls == 1

    candidate = materialize_weight_snapshot_candidate(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=destination,
        provider=provider,
        catalog=catalog,
        publication_id="publish-release",
    )

    assert candidate.state is WeightMaterializationAttemptState.MATERIALIZED
    assert catalog.get_publication("publish-release") is None
    assert len(registry.storage_objects) == stored_object_count
    assert provider.execute_calls == 1
    assert provider.release_calls == 2

    published = publish_weight_snapshot(candidate, catalog=catalog)

    assert published.state is WeightSnapshotPublicationState.PUBLISHED


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


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_known_materialization_failure_cleans_or_retains_recovery_ticket(
    cleanup_fails: bool,
) -> None:
    class KnownFailureProvider(LocalWeightTransferProvider):
        def __init__(self, registry):
            super().__init__(registry)
            self.discard_calls = []

        def probe(self, request):
            return replace(
                super().probe(request),
                supports_completion_ticket=True,
            )

        def materialization_recovery_ticket(self, prepared):
            return "ticket:known-failure"

        def wait(self, submission):
            super().wait(submission)
            raise WeightTransferError(
                "known materialization failure",
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="wait",
                operation_id=submission.request.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            )

        def discard_materialization_recovery(
            self,
            request,
            *,
            completion_ticket,
            execution_context=None,
        ):
            self.discard_calls.append(completion_ticket)
            if cleanup_fails:
                raise RuntimeError("recovery cleanup failed")

    source = placement("source")
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    provider = KnownFailureProvider(registry)
    catalog = InMemoryWeightStorageCatalog()

    with pytest.raises(
        WeightTransferReleaseError if cleanup_fails else WeightTransferError,
        match="known materialization failure",
    ):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/revision",
                object_prefix="weights/revision",
            ),
            provider=provider,
            catalog=catalog,
            publication_id=f"known-failure-{cleanup_fails}",
        )

    attempt = catalog.get_materialization(f"known-failure-{cleanup_fails}")
    assert attempt is not None
    assert provider.discard_calls == ["ticket:known-failure"]
    if cleanup_fails:
        assert attempt.state is WeightMaterializationAttemptState.PREPARING
        assert attempt.completion_ticket == "ticket:known-failure"
    else:
        assert attempt.state is WeightMaterializationAttemptState.ABORTED
        assert attempt.completion_ticket is None


@pytest.mark.parametrize("base_error", [False, True])
def test_prepared_recovery_ticket_reaches_completion_unknown_source(
    base_error: bool,
) -> None:
    class PreparedTicketProvider(LocalWeightTransferProvider):
        def probe(self, request):
            return replace(
                super().probe(request),
                supports_completion_ticket=True,
            )

        def materialization_recovery_ticket(self, prepared):
            return "ticket:prepared"

        def wait(self, submission):
            super().wait(submission)
            if base_error:
                raise WeightTransferError(
                    "commit response lost",
                    code="BACKEND_FAILURE",
                    provider=self.name,
                    phase="wait",
                    operation_id=submission.request.operation_id,
                    retryable=False,
                    completion_known=False,
                    cleanup_required=True,
                )
            raise WeightTransferCompletionUnknownError(
                "commit response lost",
                provider=self.name,
                phase="wait",
                operation_id=submission.request.operation_id,
            )

    source = placement("source")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(range(8)))
    catalog = InMemoryWeightStorageCatalog()

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(source_binding,),
            destination=WeightStorageDestination(
                provider="local",
                storage_id="weights/revision",
                object_prefix="weights/revision",
            ),
            provider=PreparedTicketProvider(registry),
            catalog=catalog,
            publication_id=f"prepared-ticket-{base_error}",
        )

    assert raised.value.completion_ticket == "ticket:prepared"
    attempt = catalog.get_materialization(f"prepared-ticket-{base_error}")
    assert attempt is not None
    assert attempt.state is WeightMaterializationAttemptState.PREPARING
    assert attempt.completion_ticket == "ticket:prepared"


def test_recovery_ticket_allows_runtime_rebinding_to_reach_provider() -> None:
    class CompletionUnknownProvider(LocalWeightTransferProvider):
        def __init__(self, registry):
            super().__init__(registry)
            self.recover_calls = 0
            self.recovered_request = None

        def wait(self, submission):
            super().wait(submission)
            raise WeightTransferCompletionUnknownError(
                "commit response lost",
                provider=self.name,
                phase="wait",
                operation_id=submission.request.operation_id,
                completion_ticket="ticket:source-snapshot",
            )

        def recover_materialization(self, request, *, completion_ticket=None):
            self.recover_calls += 1
            self.recovered_request = request
            receipt = super().recover_materialization(
                request,
                completion_ticket=completion_ticket,
            )
            assert receipt is not None
            return replace(receipt, completion_ticket=completion_ticket)

        def discard_materialization_recovery(
            self,
            request,
            *,
            completion_ticket,
            execution_context=None,
        ):
            del request, completion_ticket, execution_context

    source = placement("source")
    registry = LocalWeightBufferRegistry()
    payload = bytes(range(8))
    registry.register_runtime(0x10000, payload)
    registry.register_runtime(0x20000, payload)
    provider = CompletionUnknownProvider(registry)
    catalog = InMemoryWeightStorageCatalog()
    destination = WeightStorageDestination(
        provider=provider.name,
        storage_id="weights/revision",
        object_prefix="weights/revision",
    )
    original_binding = binding(source, 0x10000)
    rebound_binding = binding(
        source,
        0x20000,
        generation=2,
        lease_id="source:restart-lease:2",
    )
    rebound_binding = msgspec.structs.replace(
        rebound_binding,
        instance_id="source:restart-instance",
        fragments=(
            msgspec.structs.replace(
                rebound_binding.fragments[0],
                fragment_id="source:fragment:restart",
                worker_id="source:restart-worker",
                endpoint="source:restart-endpoint",
            ),
        ),
    )
    payload_identity = WeightPayloadIdentity.create(
        (source,),
        {
            source.tensors[0].placement_fragment_id: (
                f"sha256:{hashlib.sha256(payload).hexdigest()}"
            )
        },
    )

    with pytest.raises(WeightTransferCompletionUnknownError):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(original_binding,),
            destination=destination,
            provider=provider,
            catalog=catalog,
            payload_identity=payload_identity,
            publication_id="strict-source-snapshot",
        )

    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(rebound_binding,),
        destination=destination,
        provider=provider,
        catalog=catalog,
        payload_identity=payload_identity,
        publication_id="strict-source-snapshot",
    )

    assert weight_source_snapshot_digest(
        (source,),
        (original_binding,),
    ) != weight_source_snapshot_digest((source,), (rebound_binding,))
    assert provider.recover_calls == 1
    assert provider.recovered_request is not None
    assert provider.recovered_request.source_bindings == (rebound_binding,)
    assert publication.state is WeightSnapshotPublicationState.PUBLISHED


def test_runtime_rebinding_without_ticket_fails_before_provider() -> None:
    class FailCompleteOnceCatalog(InMemoryWeightStorageCatalog):
        def __init__(self):
            super().__init__()
            self.fail = True

        def complete_materialization(self, materialization_id, snapshot):
            if self.fail:
                self.fail = False
                raise RuntimeError("catalog completion unavailable")
            return super().complete_materialization(materialization_id, snapshot)

    class RecoveryCountingProvider(LocalWeightTransferProvider):
        def __init__(self, registry):
            super().__init__(registry)
            self.recover_calls = 0

        def recover_materialization(self, request, *, completion_ticket=None):
            self.recover_calls += 1
            return super().recover_materialization(
                request,
                completion_ticket=completion_ticket,
            )

    source = placement("source")
    payload = bytes(range(8))
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, payload)
    registry.register_runtime(0x20000, payload)
    provider = RecoveryCountingProvider(registry)
    catalog = FailCompleteOnceCatalog()
    destination = WeightStorageDestination(
        provider=provider.name,
        storage_id="weights/revision",
        object_prefix="weights/revision",
    )
    payload_identity = WeightPayloadIdentity.create(
        (source,),
        {
            source.tensors[0].placement_fragment_id: (
                f"sha256:{hashlib.sha256(payload).hexdigest()}"
            )
        },
    )

    with pytest.raises(RuntimeError, match="catalog completion unavailable"):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            destination=destination,
            provider=provider,
            catalog=catalog,
            payload_identity=payload_identity,
            publication_id="rebind-without-ticket",
        )

    attempt = catalog.get_materialization("rebind-without-ticket")
    assert attempt is not None
    assert attempt.completion_ticket is None
    with pytest.raises(ValueError, match="original intent"):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(
                binding(
                    source,
                    0x20000,
                    generation=2,
                    lease_id="source:restart-lease:2",
                ),
            ),
            destination=destination,
            provider=provider,
            catalog=catalog,
            payload_identity=payload_identity,
            publication_id="rebind-without-ticket",
        )

    assert provider.recover_calls == 0


def test_recovery_does_not_treat_missing_payload_digest_as_a_wildcard() -> None:
    class CompletionUnknownProvider(LocalWeightTransferProvider):
        def __init__(self, registry):
            super().__init__(registry)
            self.recover_calls = 0

        def wait(self, submission):
            super().wait(submission)
            raise WeightTransferCompletionUnknownError(
                "commit response lost",
                provider=self.name,
                phase="wait",
                operation_id=submission.request.operation_id,
                completion_ticket="ticket:payload-digest",
            )

        def recover_materialization(self, request, *, completion_ticket=None):
            self.recover_calls += 1
            return super().recover_materialization(
                request,
                completion_ticket=completion_ticket,
            )

    source = placement("source")
    source_binding = binding(source, 0x10000)
    registry = LocalWeightBufferRegistry()
    payload = bytes(range(8))
    registry.register_runtime(0x10000, payload)
    provider = CompletionUnknownProvider(registry)
    catalog = InMemoryWeightStorageCatalog()
    destination = WeightStorageDestination(
        provider=provider.name,
        storage_id="weights/revision",
        object_prefix="weights/revision",
    )

    with pytest.raises(WeightTransferCompletionUnknownError):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(source_binding,),
            destination=destination,
            provider=provider,
            catalog=catalog,
            publication_id="strict-payload-digest",
        )

    with pytest.raises(ValueError, match="original intent"):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(
                binding(
                    source,
                    0x10000,
                    generation=2,
                    lease_id="source:restart-lease:2",
                ),
            ),
            destination=destination,
            provider=provider,
            catalog=catalog,
            publication_id="strict-payload-digest",
        )

    payload_identity = WeightPayloadIdentity.create(
        (source,),
        {
            source.tensors[0].placement_fragment_id: (
                f"sha256:{hashlib.sha256(payload).hexdigest()}"
            )
        },
    )
    with pytest.raises(ValueError, match="original intent"):
        materialize_weight_snapshot(
            source_placements=(source,),
            source_bindings=(source_binding,),
            destination=destination,
            provider=provider,
            catalog=catalog,
            payload_identity=payload_identity,
            publication_id="strict-payload-digest",
        )

    assert provider.recover_calls == 0


@pytest.mark.parametrize(
    "field",
    (
        "instance_id",
        "generation",
        "lease_id",
        "fragment_id",
        "storage_offset",
        "device",
        "is_contiguous",
        "worker_id",
        "endpoint",
    ),
)
def test_source_snapshot_digest_binds_runtime_identity_except_address(
    field: str,
) -> None:
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
    first_digest = weight_source_snapshot_digest((source,), (first,))
    assert weight_source_snapshot_digest((source,), (relocated,)) == first_digest

    fragment = relocated.fragments[0]
    replacements = {
        "instance_id": msgspec.structs.replace(
            relocated,
            instance_id="source:instance:2",
        ),
        "generation": msgspec.structs.replace(relocated, generation=2),
        "lease_id": msgspec.structs.replace(
            relocated,
            lease_id="source:lease:2",
        ),
        "fragment_id": msgspec.structs.replace(
            relocated,
            fragments=(
                msgspec.structs.replace(
                    fragment,
                    fragment_id="source:fragment:2",
                ),
            ),
        ),
        "storage_offset": msgspec.structs.replace(
            relocated,
            fragments=(msgspec.structs.replace(fragment, storage_offset=1),),
        ),
        "device": msgspec.structs.replace(
            relocated,
            fragments=(msgspec.structs.replace(fragment, device="cuda:1"),),
        ),
        "is_contiguous": msgspec.structs.replace(
            relocated,
            fragments=(msgspec.structs.replace(fragment, is_contiguous=False),),
        ),
        "worker_id": msgspec.structs.replace(
            relocated,
            fragments=(msgspec.structs.replace(fragment, worker_id="source:worker:2"),),
        ),
        "endpoint": msgspec.structs.replace(
            relocated,
            fragments=(
                msgspec.structs.replace(fragment, endpoint="source:endpoint:2"),
            ),
        ),
    }
    assert (
        weight_source_snapshot_digest((source,), (replacements[field],)) != first_digest
    )


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
