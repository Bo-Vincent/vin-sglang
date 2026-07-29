from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import sglang.srt.weight_transfer as weight_transfer
from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    WeightSnapshotCoordinator,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer import api as weight_transfer_api
from sglang.srt.weight_transfer.api import (
    execute_weight_materialization,
    load_weights,
    mark_weight_snapshot_serving,
    materialize_weight_snapshot,
    prepare_weight_materialization,
)
from sglang.srt.weight_transfer.contracts import (
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
)
from sglang.srt.weight_transfer.provider import (
    LocalWeightBufferRegistry,
    LocalWeightTransferProvider,
    WeightLoadReceipt,
    WeightMaterializeReceipt,
    WeightProviderCapabilities,
    WeightStorageDestination,
    WeightTargetLoadMode,
    WeightTargetLoadSession,
    WeightTargetLoadState,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
    WeightTransferExecutionContext,
    WeightTransferReleaseError,
)
from sglang.srt.weight_transfer.storage import (
    InMemoryWeightStorageCatalog,
    WeightMaterializationAttemptState,
    WeightRevisionHead,
    WeightRevisionState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LEGACY_PUBLIC_EXPORTS = frozenset(
    {
        "BoundWeightTransferPlan",
        "BoundWeightTransferRegion",
        "CheckpointLoadStats",
        "CheckpointProviderState",
        "CheckpointStorageToRuntimeProvider",
        "DEFAULT_WEIGHT_PLANNER_LIMITS",
        "FileWeightStorageCatalog",
        "FlatWeightTransferOperation",
        "InMemoryWeightStorageCatalog",
        "LocalWeightStoreDistributedCoordinator",
        "LogicalPlacementFragment",
        "LogicalWeightTransferPlan",
        "LogicalWeightTransferRegion",
        "MooncakeWeightStoreProvider",
        "MooncakeWeightTransferProvider",
        "PipelineRouteGroup",
        "PlacementExecutorGroup",
        "RootWeightStorageCatalog",
        "RuntimeRangeWriter",
        "RuntimeWeightLocation",
        "RuntimeWeightPayloadHasher",
        "RuntimeWeightSnapshotSource",
        "SemanticCheckpointSource",
        "StorageRangeReader",
        "StorageVersionReader",
        "StorageWeightLocation",
        "StoredWeightSnapshot",
        "TorchDistributedWeightStoreCoordinator",
        "WeightLoadReceipt",
        "WeightLoadRequest",
        "WeightLoweringLimits",
        "WeightMaterializationAttempt",
        "WeightMaterializationAttemptState",
        "WeightMaterializationIntent",
        "WeightMaterializeReceipt",
        "WeightMaterializeRequest",
        "WeightPayloadFragmentIdentity",
        "WeightPayloadIdentity",
        "WeightPlannerLimits",
        "WeightProviderCapabilities",
        "WeightProviderReceipt",
        "WeightProviderRequest",
        "WeightRevisionHead",
        "WeightRevisionState",
        "WeightSnapshotBackend",
        "WeightSnapshotBackendFactory",
        "WeightSnapshotLoadSpec",
        "WeightSnapshotPublication",
        "WeightSnapshotPublicationState",
        "WeightSnapshotWriteSpec",
        "WeightStorageBindingManifest",
        "WeightStorageCatalog",
        "WeightStorageDestination",
        "WeightStorageFragmentBinding",
        "WeightStorageRef",
        "WeightStoreDistributedCoordinator",
        "WeightStoreDistributedError",
        "WeightStoreUploadOutcome",
        "WeightTargetActivationController",
        "WeightTargetLoadMode",
        "WeightTargetLoadSession",
        "WeightTargetLoadState",
        "WeightTransferAttestor",
        "WeightTransferBatch",
        "WeightTransferCompletionUnknownError",
        "WeightTransferError",
        "WeightTransferProvider",
        "WeightTransferReleaseError",
        "WeightTransferTerminalProof",
        "WeightTransferTerminalStatus",
        "bind_weight_source",
        "bind_weight_transfer_plan",
        "execute_weight_load",
        "execute_weight_materialization",
        "iter_bounded_transfer_batches",
        "load_checkpoint_weights",
        "load_weight_snapshot",
        "load_weights",
        "load_weights_to_local_target",
        "lowering_operation_count",
        "mark_weight_snapshot_serving",
        "materialize_checkpoint_weight_snapshot",
        "materialize_checkpoint_weights",
        "materialize_distributed_runtime_weight_snapshot",
        "materialize_runtime_weight_snapshot",
        "materialize_runtime_weights",
        "materialize_weight_snapshot",
        "materialize_weights",
        "open_weight_snapshot_backend",
        "open_weight_snapshot_write_backend",
        "plan_weight_transfer",
        "plan_weight_transfer_to_local_target",
        "prepare_weight_load",
        "prepare_weight_load_from_plan",
        "prepare_weight_load_to_local_target",
        "prepare_weight_materialization",
        "project_source_bindings",
        "quarantined_runtime_weight_snapshots",
        "runtime_manifest_to_parts",
        "select_weight_storage_placements",
        "weight_placement_set_digest",
        "weight_source_snapshot_digest",
        "weight_stored_payload_digest",
    }
)


@pytest.mark.parametrize(
    "name",
    [
        "WeightLoadRequest",
        "WeightMaterializationCompletionTicketProvider",
        "WeightMaterializationRecoveryProvider",
        "WeightMaterializeRequest",
        "WeightProviderReceipt",
        "WeightProviderRequest",
        "WeightTransferAttestor",
        "WeightTransferProvider",
    ],
)
def test_public_facade_exports_provider_contract(name: str) -> None:
    assert name in weight_transfer.__all__
    assert getattr(weight_transfer, name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "WeightTransferPreflight",
        "WeightTransferExecutionContext",
        "mark_weight_snapshot_serving",
        "preflight_weight_transfer",
    ],
)
def test_public_facade_exports_current_contract(name: str) -> None:
    assert name in weight_transfer.__all__
    assert getattr(weight_transfer, name) is not None


@pytest.mark.parametrize(
    "name",
    [
        "PhysicalWeightLocation",
        "SourceBindingManifest",
        "WeightRuntimeBindingManifest",
        "WeightTransferCancellationSignal",
    ],
)
def test_public_facade_exports_signature_contract(name: str) -> None:
    assert name in weight_transfer.__all__
    assert getattr(weight_transfer, name) is not None


def test_public_facade_preserves_complete_legacy_exports() -> None:
    missing = _LEGACY_PUBLIC_EXPORTS.difference(weight_transfer.__all__)
    unresolved = {
        name
        for name in _LEGACY_PUBLIC_EXPORTS
        if getattr(weight_transfer, name, None) is None
    }

    assert not missing
    assert not unresolved


@pytest.mark.parametrize(
    "name",
    ["LocalWeightBufferRegistry", "LocalWeightTransferProvider"],
)
def test_public_facade_does_not_export_reference_provider(name: str) -> None:
    assert name not in weight_transfer.__all__


def placement(side: str) -> WeightPlacementManifest:
    tensor = WeightPlacementTensor(
        placement_fragment_id=f"{side}:fragment",
        tensor_id="weight",
        runtime_name="weight",
        aliases=("weight",),
        global_shape=(8,),
        global_offset=(0,),
        local_shape=(8,),
        dtype="bfloat16",
        itemsize=2,
        partition_dim=None,
        shard_dims=(),
        layer_id=0,
        expert_id=None,
        layout_fingerprint="layout:v1",
        nbytes=16,
        byte_offset=0,
        rank=WeightParallelRank(),
    )
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id((tensor,)),
        tensors=(tensor,),
    )


def binding(
    manifest: WeightPlacementManifest,
    address: int,
    *,
    nbytes_delta: int = 0,
) -> WeightRuntimeBindingManifest:
    tensor = manifest.tensors[0]
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id=f"instance:{manifest.placement_id}",
        generation=1,
        lease_id=f"lease:{manifest.placement_id}",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=address,
                nbytes=tensor.nbytes + nbytes_delta,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id=manifest.placement_id,
                endpoint=f"{manifest.placement_id}:12345",
            ),
        ),
    )


def placement_for_dp(side: str, dp_rank: int) -> WeightPlacementManifest:
    base = placement(side)
    tensor = msgspec.structs.replace(
        base.tensors[0],
        placement_fragment_id=f"{side}:dp{dp_rank}:fragment",
        rank=WeightParallelRank(dp=dp_rank),
    )
    return msgspec.structs.replace(
        base,
        placement_id=compute_weight_placement_id((tensor,)),
        tensors=(tensor,),
    )


def test_local_target_load_projects_source_bindings_for_subset_plan() -> None:
    base = placement("source")
    second = msgspec.structs.replace(
        base.tensors[0],
        placement_fragment_id="source:layer1:fragment",
        tensor_id="weight.layer1",
        runtime_name="weight.layer1",
        layer_id=1,
        byte_offset=16,
    )
    source = msgspec.structs.replace(
        base,
        placement_id=compute_weight_placement_id((base.tensors[0], second)),
        tensors=(base.tensors[0], second),
    )
    target_tensor = msgspec.structs.replace(
        base.tensors[0],
        placement_fragment_id="target:layer0:fragment",
    )
    target = msgspec.structs.replace(
        base,
        placement_id=compute_weight_placement_id((target_tensor,)),
        tensors=(target_tensor,),
    )
    source_binding = WeightRuntimeBindingManifest(
        model_id=source.model_id,
        revision=source.revision,
        placement_id=source.placement_id,
        instance_id="source-instance",
        generation=1,
        lease_id="source-lease",
        fragments=tuple(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=0x10000 + index * 0x100,
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id="source-worker",
                endpoint="source-worker:12345",
            )
            for index, tensor in enumerate(source.tensors)
        ),
    )

    request = weight_transfer_api.prepare_weight_load_to_local_target(
        source_placements=(source,),
        source_bindings=(source_binding,),
        target_placement=target,
        target_binding=binding(target, 0x20000),
    )

    source_fragments = {
        region.source.placement_fragment_id for region in request.plan.regions
    }
    assert source_fragments == {base.tensors[0].placement_fragment_id}
    assert request.plan.logical_plan.source_placements[0].placement_id != (
        source.placement_id
    )


class RecordingProvider:
    name = "recording"
    requires_runtime_attestation = False

    def __init__(
        self,
        *,
        fail_known: bool = False,
        fail_unknown: bool = False,
        fail_unknown_base: bool = False,
        interrupt_wait: bool = False,
        fail_cancel: bool = False,
        fail_release: bool = False,
        max_total_bytes: int | None = None,
        max_total_operations: int | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.fail_known = fail_known
        self.fail_unknown = fail_unknown
        self.fail_unknown_base = fail_unknown_base
        self.interrupt_wait = interrupt_wait
        self.fail_cancel = fail_cancel
        self.fail_release = fail_release
        self.max_total_bytes = max_total_bytes
        self.max_total_operations = max_total_operations
        self.events = [] if events is None else events

    def probe(self, request):
        self.events.append("probe")
        return WeightProviderCapabilities(
            provider=self.name,
            load_profiles=frozenset({"runtime_to_runtime"}),
            materialize_profiles=frozenset(),
            supports_nd_regions=True,
            supports_strided_regions=True,
            supports_safe_cancel=True,
            supports_completion_ticket=True,
            supports_transactional_publish=False,
            max_regions=1024,
            max_segments_per_region=1_000_000,
            max_total_operations=self.max_total_operations,
            max_total_bytes=self.max_total_bytes,
        )

    def prepare(self, request, *, execution_context=None):
        del execution_context
        self.events.append("prepare")
        return request

    def submit(self, prepared):
        self.events.append("submit")
        return prepared

    def wait(self, submission, *, execution_context=None):
        del execution_context
        self.events.append("wait")
        if self.fail_unknown:
            raise WeightTransferCompletionUnknownError(
                "completion is unknown",
                provider=self.name,
                phase="wait",
                operation_id=submission.operation_id,
            )
        if self.fail_unknown_base:
            raise WeightTransferError(
                "completion is unknown",
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="wait",
                operation_id=submission.operation_id,
                retryable=True,
                completion_known=False,
                cleanup_required=True,
            )
        if self.interrupt_wait:
            raise KeyboardInterrupt("interrupted while waiting")
        if self.fail_known:
            raise WeightTransferError(
                "known failure",
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="wait",
                operation_id=submission.operation_id,
                retryable=True,
                completion_known=True,
                cleanup_required=True,
            )
        return WeightLoadReceipt(
            operation_id=submission.operation_id,
            provider=self.name,
            plan_digest=submission.plan.digest,
            total_bytes=submission.plan.total_bytes,
            region_count=len(submission.plan.regions),
        )

    def cancel(self, submission):
        self.events.append("cancel")
        if self.fail_cancel:
            raise TimeoutError("cancel did not reach a terminal state")

    def synchronize(self, receipt, *, execution_context=None):
        del execution_context
        self.events.append("synchronize")

    def release(self, prepared, receipt, *, execution_context=None):
        del execution_context
        self.events.append("release")
        if self.fail_release:
            raise RuntimeError("release failed")


class RecordingMaterializeProvider(RecordingProvider):
    def __init__(self, manifest_key: str) -> None:
        super().__init__()
        self.manifest_key = manifest_key

    def probe(self, request):
        self.events.append("probe")
        return WeightProviderCapabilities(
            provider=self.name,
            load_profiles=frozenset(),
            materialize_profiles=frozenset({"runtime_to_storage"}),
            supports_nd_regions=True,
            supports_strided_regions=True,
            supports_safe_cancel=True,
            supports_completion_ticket=False,
            supports_transactional_publish=True,
        )

    def wait(self, submission):
        self.events.append("wait")
        bindings = tuple(
            WeightStorageBindingManifest(
                model_id=placement.model_id,
                revision=placement.revision,
                placement_id=placement.placement_id,
                storage_id=submission.destination.storage_id,
                provider=self.name,
                fragments=tuple(
                    WeightStorageFragmentBinding(
                        placement_fragment_id=tensor.placement_fragment_id,
                        fragment_id=f"stored:{tensor.placement_fragment_id}",
                        object_key=(
                            f"{submission.destination.object_prefix}/payload/"
                            f"{tensor.placement_fragment_id}"
                        ),
                        object_offset=0,
                        nbytes=tensor.nbytes,
                    )
                    for tensor in placement.tensors
                ),
            )
            for placement in submission.source_placements
        )
        return WeightMaterializeReceipt(
            operation_id=submission.operation_id,
            provider=self.name,
            manifest_key=self.manifest_key,
            stored_placements=submission.source_placements,
            storage_bindings=bindings,
            total_bytes=submission.total_bytes,
            fragment_count=len(submission.source_locations),
        )


def test_serving_revision_cannot_be_demoted_to_ready() -> None:
    source = placement("source")
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(16))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="serving-is-monotonic",
    )
    serving = mark_weight_snapshot_serving(publication.snapshot.ref, catalog=catalog)

    with pytest.raises(ValueError, match="invalid revision transition"):
        catalog.compare_and_set_revision(
            model_id=serving.model_id,
            revision=serving.revision,
            expected=serving,
            new_ref=serving.ref,
            new_state=WeightRevisionState.READY,
        )


def test_mark_weight_snapshot_serving_preserves_head_return_type() -> None:
    source = placement("source")
    registry = LocalWeightBufferRegistry()
    registry.register_runtime(0x10000, bytes(16))
    catalog = InMemoryWeightStorageCatalog()
    publication = materialize_weight_snapshot(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        provider=LocalWeightTransferProvider(registry),
        catalog=catalog,
        publication_id="serving-head-compatibility",
    )

    head = mark_weight_snapshot_serving(publication.snapshot.ref, catalog=catalog)

    assert isinstance(head, WeightRevisionHead)
    assert head.state is WeightRevisionState.SERVING


def test_materialization_is_not_published_before_provider_finalize() -> None:
    class FinalizeFailureProvider(RecordingMaterializeProvider):
        def probe(self, request):
            capabilities = super().probe(request)
            return replace(capabilities, supports_completion_ticket=True)

        def materialization_recovery_ticket(self, prepared):
            return "finalize-ticket"

        def recover_materialization(self, request, *, completion_ticket=None):
            raise AssertionError("recovery is not expected during the first attempt")

        def wait(self, submission):
            return replace(
                super().wait(submission),
                completion_ticket="finalize-ticket",
            )

        def release(self, prepared, receipt):
            super().release(prepared, receipt)
            raise RuntimeError("finalize failed")

    source = placement("source")
    provider = FinalizeFailureProvider("weights/revision/manifest")
    catalog = InMemoryWeightStorageCatalog()
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        operation_id="finalize-failure",
    )

    with pytest.raises(WeightTransferReleaseError):
        materialize_weight_snapshot(
            request,
            provider=provider,
            catalog=catalog,
        )

    attempt = catalog.get_materialization(request.operation_id)
    assert attempt is not None
    assert attempt.state is WeightMaterializationAttemptState.PREPARING
    assert attempt.completion_ticket == "finalize-ticket"
    assert catalog.get_publication(request.operation_id) is None
    assert catalog.get_revision_head("model", "revision") is None


def test_prepare_materialization_accepts_a_selected_local_source_closure() -> None:
    def sharded_source(side: str, tp_rank: int) -> WeightPlacementManifest:
        base = placement(side)
        tensor = msgspec.structs.replace(
            base.tensors[0],
            placement_fragment_id=f"{side}:tp{tp_rank}:fragment",
            global_shape=(16,),
            global_offset=(tp_rank * 8,),
            local_shape=(8,),
            partition_dim=0,
            shard_dims=(0,),
            rank=WeightParallelRank(tp=tp_rank),
        )
        return msgspec.structs.replace(
            base,
            placement_id=compute_weight_placement_id((tensor,)),
            tensors=(tensor,),
        )

    source = sharded_source("source-tp1", 1)

    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider="recording",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        source_placements_are_selected=True,
    )

    assert request.source_placements == (source,)


def test_load_weights_runs_provider_lifecycle_in_order() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider()

    receipt = load_weights(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        target_placements=(target,),
        target_bindings=(binding(target, 0x20000),),
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
    )

    assert receipt.total_bytes == 16
    assert [name for name, _ in receipt.provider_phase_seconds] == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "synchronize",
        "release",
    ]
    assert all(seconds >= 0 for _, seconds in receipt.provider_phase_seconds)
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "synchronize",
        "release",
    ]


@pytest.mark.parametrize(
    "forge",
    [
        lambda receipt: None,
        lambda receipt: replace(receipt, operation_id="other-operation"),
        lambda receipt: replace(receipt, provider="other-provider"),
        lambda receipt: replace(receipt, plan_digest="sha256:" + "0" * 64),
        lambda receipt: replace(receipt, total_bytes=receipt.total_bytes + 1),
        lambda receipt: replace(receipt, region_count=receipt.region_count + 1),
    ],
)
def test_cold_start_invalid_receipt_retains_provider_resources(
    forge,
) -> None:
    class ForgedReceiptProvider(RecordingProvider):
        def wait(self, submission):
            return forge(super().wait(submission))

    source = placement("source")
    target = placement("target")
    provider = ForgedReceiptProvider()

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.completion_known is False
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
    ]


def test_load_requires_an_explicit_cold_start_or_live_update_mode() -> None:
    source = placement("source")
    target = placement("target")
    source_binding = binding(source, 0x10000)
    target_binding = binding(target, 0x20000)
    provider = RecordingProvider()

    with pytest.raises(TypeError, match="target_mode"):
        load_weights(
            source_placements=(source,),
            source_bindings=(source_binding,),
            target_placements=(target,),
            target_bindings=(target_binding,),
            provider=provider,
        )
    with pytest.raises(ValueError, match="requires a target load session"):
        load_weights(
            source_placements=(source,),
            source_bindings=(source_binding,),
            target_placements=(target,),
            target_bindings=(target_binding,),
            provider=provider,
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
        )
    with pytest.raises(ValueError, match="must not use a live target session"):
        load_weights(
            source_placements=(source,),
            source_bindings=(source_binding,),
            target_placements=(target,),
            target_bindings=(target_binding,),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            target_session=WeightTargetLoadSession(
                target_bindings=(target_binding,),
                owners=(object(),),
                coordinator=WeightSnapshotCoordinator(),
            ),
        )

    assert provider.events == []


def test_preflight_binding_failure_never_calls_provider() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider()

    with pytest.raises(ValueError, match="byte size differs"):
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000, nbytes_delta=-2),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert provider.events == []


def test_known_failure_cancels_and_releases() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(fail_known=True)

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.completion_known is True
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "cancel",
        "release",
    ]


def test_release_failure_preserves_the_execution_error() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(fail_known=True, fail_release=True)

    with pytest.raises(WeightTransferReleaseError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.code == "RELEASE_FAILED"
    assert isinstance(raised.value.__cause__, WeightTransferError)
    assert isinstance(raised.value.release_error, RuntimeError)
    assert str(raised.value.release_error) == "release failed"
    assert provider.events[-2:] == ["cancel", "release"]


def test_completion_unknown_retains_provider_resources() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(fail_unknown=True)

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.completion_known is False
    assert isinstance(raised.value.__cause__, WeightTransferError)
    assert provider.events == ["probe", "prepare", "submit", "wait"]


def test_base_completion_unknown_error_retains_provider_resources() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(fail_unknown_base=True)

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.completion_known is False
    assert provider.events == ["probe", "prepare", "submit", "wait"]


def test_generic_completion_unknown_preserves_ticket_without_abort() -> None:
    class MaterializeProvider(RecordingProvider):
        def probe(self, request):
            self.events.append("probe")
            return WeightProviderCapabilities(
                provider=self.name,
                load_profiles=frozenset(),
                materialize_profiles=frozenset({"runtime_to_storage"}),
                supports_nd_regions=True,
                supports_strided_regions=True,
                supports_safe_cancel=True,
                supports_completion_ticket=True,
                supports_transactional_publish=True,
            )

        def materialization_recovery_ticket(self, prepared):
            self.events.append("ticket")
            return "ticket-generated"

        def wait(self, submission):
            self.events.append("wait")
            raise WeightTransferError(
                "completion is unknown",
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="wait",
                operation_id=submission.operation_id,
                retryable=True,
                completion_known=False,
                cleanup_required=True,
            )

        def recover_materialization(self, request, *, completion_ticket):
            self.events.append(f"recover:{completion_ticket}")
            raise WeightTransferError(
                "recovery completion is unknown",
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="recover",
                operation_id=request.operation_id,
                retryable=True,
                completion_known=False,
                cleanup_required=True,
            )

    class Catalog:
        def __init__(self):
            self.attempt = None
            self.abort_calls = []

        def get_materialization(self, materialization_id):
            return self.attempt

        def begin_materialization(self, materialization_id, intent):
            self.attempt = SimpleNamespace(
                intent=intent,
                completion_ticket=None,
                state=object(),
            )
            return self.attempt

        def get_publication(self, materialization_id):
            return None

        def set_materialization_completion_ticket(
            self,
            materialization_id,
            completion_ticket,
        ):
            self.attempt.completion_ticket = completion_ticket

        def abort_materialization(self, materialization_id):
            self.abort_calls.append(materialization_id)

    source = placement("source")
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider="recording",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        operation_id="materialize-generic-unknown",
    )
    provider = MaterializeProvider()
    catalog = Catalog()

    with pytest.raises(WeightTransferCompletionUnknownError) as first:
        materialize_weight_snapshot(
            request,
            provider=provider,
            catalog=catalog,
        )

    assert first.value.completion_ticket == "ticket-generated"
    assert catalog.attempt.completion_ticket == "ticket-generated"
    assert catalog.abort_calls == []

    with pytest.raises(WeightTransferCompletionUnknownError) as recovered:
        materialize_weight_snapshot(
            request,
            provider=provider,
            catalog=catalog,
        )

    assert recovered.value.completion_ticket == "ticket-generated"
    assert catalog.attempt.completion_ticket == "ticket-generated"
    assert catalog.abort_calls == []
    assert provider.events == [
        "probe",
        "prepare",
        "ticket",
        "submit",
        "wait",
        "recover:ticket-generated",
    ]


def test_cancel_failure_after_interrupt_becomes_completion_unknown() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(interrupt_wait=True, fail_cancel=True)

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.completion_known is False
    assert raised.value.__cause__.__class__ is KeyboardInterrupt
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "cancel",
    ]


def test_capability_limit_fails_before_prepare() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider(max_total_bytes=8)

    with pytest.raises(WeightTransferError, match="byte limit"):
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert provider.events == ["probe"]


def test_materialization_operation_limit_fails_before_prepare() -> None:
    class MaterializeProvider(RecordingProvider):
        def probe(self, request):
            self.events.append("probe")
            return WeightProviderCapabilities(
                provider=self.name,
                load_profiles=frozenset(),
                materialize_profiles=frozenset({"runtime_to_storage"}),
                supports_nd_regions=True,
                supports_strided_regions=True,
                supports_safe_cancel=True,
                supports_completion_ticket=False,
                supports_transactional_publish=True,
                max_total_operations=1,
            )

    source = placement("source")
    first_tensor = source.tensors[0]
    second_tensor = msgspec.structs.replace(
        first_tensor,
        placement_fragment_id="source:fragment:copy",
        tensor_id="weight.copy",
        runtime_name="weight.copy",
        aliases=("weight.copy",),
        byte_offset=first_tensor.nbytes,
    )
    source = msgspec.structs.replace(
        source,
        placement_id=compute_weight_placement_id((first_tensor, second_tensor)),
        tensors=(first_tensor, second_tensor),
    )
    source_binding = binding(source, 0x10000)
    source_binding = msgspec.structs.replace(
        source_binding,
        fragments=(
            source_binding.fragments[0],
            msgspec.structs.replace(
                source_binding.fragments[0],
                placement_fragment_id=second_tensor.placement_fragment_id,
                fragment_id="runtime:source:fragment:copy",
                address=0x20000,
            ),
        ),
    )
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(source_binding,),
        destination=WeightStorageDestination(
            provider="recording",
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
    )
    provider = MaterializeProvider()

    with pytest.raises(WeightTransferError, match="operation limit"):
        execute_weight_materialization(request, provider=provider)

    assert provider.events == ["probe"]


def test_materialization_accepts_provider_manifest_key_under_prefix() -> None:
    source = placement("source")
    provider = RecordingMaterializeProvider("weights/revision/index.json")
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
    )

    receipt = execute_weight_materialization(request, provider=provider)

    assert receipt.manifest_key == "weights/revision/index.json"
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "synchronize",
        "release",
    ]


def test_materialization_rejects_manifest_key_outside_prefix_as_unknown() -> None:
    source = placement("source")
    provider = RecordingMaterializeProvider("weights/other/index.json")
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
    )

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
        execute_weight_materialization(request, provider=provider)

    assert raised.value.completion_known is False
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
    ]


def test_provider_capability_uses_only_bounded_execution_name() -> None:
    capabilities = WeightProviderCapabilities(
        provider="bounded",
        load_profiles=frozenset(),
        materialize_profiles=frozenset({"runtime_to_storage"}),
        supports_nd_regions=True,
        supports_strided_regions=True,
        supports_safe_cancel=False,
        supports_completion_ticket=True,
        supports_transactional_publish=True,
        supports_bounded_execution=True,
    )

    assert capabilities.supports_bounded_execution is True
    assert not hasattr(capabilities, "supports_bounded_wait")


def test_completion_ticket_capability_requires_recovery_protocol() -> None:
    class TicketOnlyProvider(RecordingMaterializeProvider):
        def probe(self, request):
            return replace(
                super().probe(request),
                supports_completion_ticket=True,
            )

        def materialization_recovery_ticket(self, prepared):
            return "ticket"

    source = placement("source")
    provider = TicketOnlyProvider("weights/revision/index.json")
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
    )

    with pytest.raises(ValueError, match="recovery protocol"):
        execute_weight_materialization(request, provider=provider)

    assert provider.events == ["probe"]


def test_bounded_execution_rejects_unbounded_provider_before_prepare() -> None:
    class MaterializeProvider(RecordingProvider):
        def probe(self, request):
            self.events.append("probe")
            return WeightProviderCapabilities(
                provider=self.name,
                load_profiles=frozenset(),
                materialize_profiles=frozenset({"runtime_to_storage"}),
                supports_nd_regions=True,
                supports_strided_regions=True,
                supports_safe_cancel=True,
                supports_completion_ticket=False,
                supports_transactional_publish=True,
            )

    source = placement("source")
    provider = MaterializeProvider()
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
    )

    with pytest.raises(WeightTransferError) as raised:
        execute_weight_materialization(
            request,
            provider=provider,
            execution_context=WeightTransferExecutionContext(deadline_unix_sec=10**10),
        )

    assert raised.value.code == "UNBOUNDED_PROVIDER"
    assert provider.events == ["probe"]


def test_cancelled_context_reports_cancelled_before_prepare() -> None:
    class BoundedProvider(RecordingProvider):
        def probe(self, request):
            return replace(
                super().probe(request),
                supports_bounded_execution=True,
            )

    source = placement("source")
    target = placement("target")
    provider = BoundedProvider()

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            execution_context=WeightTransferExecutionContext(
                deadline_unix_sec=10**10,
                cancel_signal=SimpleNamespace(is_set=lambda: True),
            ),
        )

    assert raised.value.code == "CANCELLED"
    assert raised.value.completion_known is True
    assert provider.events == ["probe"]


def test_materialization_cleans_ticket_without_sink_before_submit() -> None:
    class SlowTicketProvider(RecordingMaterializeProvider):
        def __init__(self):
            super().__init__("weights/revision/manifest")
            self.events = []

        def probe(self, request):
            return replace(
                super().probe(request),
                supports_completion_ticket=True,
                supports_bounded_execution=True,
            )

        def prepare(self, request, *, execution_context=None):
            del execution_context
            self.events.append("prepare")
            time.sleep(0.03)
            return request

        def materialization_recovery_ticket(self, prepared):
            self.events.append("ticket")
            return "ticket"

        def recover_materialization(
            self,
            request,
            *,
            completion_ticket=None,
            execution_context=None,
        ):
            del request, completion_ticket, execution_context
            return None

        def discard_materialization_recovery(
            self,
            request,
            *,
            completion_ticket,
            execution_context=None,
        ):
            del request, completion_ticket, execution_context
            self.events.append("discard")

    source = placement("source")
    provider = SlowTicketProvider()
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
        operation_id="ticket-cleanup-before-submit",
    )

    with pytest.raises(WeightTransferError) as raised:
        execute_weight_materialization(
            request,
            provider=provider,
            execution_context=WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + 0.01,
            ),
        )

    assert raised.value.code == "DEADLINE_EXCEEDED"
    assert provider.events == ["probe", "prepare", "ticket", "release", "discard"]


def test_bounded_execution_does_not_downgrade_legacy_synchronize() -> None:
    class LegacySynchronizeProvider(RecordingProvider):
        def probe(self, request):
            return replace(
                super().probe(request),
                supports_bounded_execution=True,
            )

        def synchronize(self, receipt):
            return super().synchronize(receipt)

    source = placement("source")
    target = placement("target")
    provider = LegacySynchronizeProvider()

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            execution_context=WeightTransferExecutionContext(
                deadline_unix_sec=10**10,
            ),
        )

    assert raised.value.code == "BACKEND_FAILURE"
    assert isinstance(raised.value.__cause__, TypeError)
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "cancel",
        "release",
    ]


def test_bounded_execution_uses_terminal_context_for_release() -> None:
    class MaterializeProvider(RecordingProvider):
        def __init__(self):
            super().__init__()
            self.contexts = []

        def probe(self, request):
            self.events.append("probe")
            return WeightProviderCapabilities(
                provider=self.name,
                load_profiles=frozenset(),
                materialize_profiles=frozenset({"runtime_to_storage"}),
                supports_nd_regions=True,
                supports_strided_regions=True,
                supports_safe_cancel=True,
                supports_completion_ticket=False,
                supports_transactional_publish=True,
                supports_bounded_execution=True,
            )

        def wait(self, submission, *, execution_context=None):
            self.events.append("wait")
            self.contexts.append(execution_context)
            raise WeightTransferError(
                "known bounded failure",
                code="BACKEND_FAILURE",
                provider=self.name,
                phase="wait",
                operation_id=submission.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            )

        def release(self, prepared, receipt, *, execution_context=None):
            self.events.append("release")
            self.contexts.append(execution_context)

    source = placement("source")
    provider = MaterializeProvider()
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
    )
    context = WeightTransferExecutionContext(deadline_unix_sec=10**10)

    with pytest.raises(WeightTransferError, match="known bounded failure"):
        execute_weight_materialization(
            request,
            provider=provider,
            execution_context=context,
        )

    assert provider.contexts[0] is context
    release_context = provider.contexts[1]
    assert release_context is not context
    assert release_context.cancel_signal is None
    assert 0 < release_context.remaining_seconds()
    assert (
        release_context.remaining_seconds()
        <= weight_transfer_api._PROVIDER_TERMINAL_RELEASE_TIMEOUT_SEC
    )
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "cancel",
        "release",
    ]


def test_bounded_load_context_reaches_provider_lifecycle() -> None:
    class LoadProvider(RecordingProvider):
        def __init__(self):
            super().__init__()
            self.contexts = []

        def probe(self, request):
            return replace(
                super().probe(request),
                supports_bounded_execution=True,
            )

        def prepare(self, request, *, execution_context=None):
            self.contexts.append(execution_context)
            return super().prepare(request)

        def wait(self, submission, *, execution_context=None):
            self.contexts.append(execution_context)
            return super().wait(submission)

        def synchronize(self, receipt, *, execution_context=None):
            self.contexts.append(execution_context)
            return super().synchronize(receipt)

        def release(self, prepared, receipt, *, execution_context=None):
            self.contexts.append(execution_context)
            return super().release(prepared, receipt)

    source = placement("source")
    target = placement("target")
    provider = LoadProvider()
    context = WeightTransferExecutionContext(deadline_unix_sec=10**10)

    receipt = load_weights(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        target_placements=(target,),
        target_bindings=(binding(target, 0x20000),),
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
        execution_context=context,
    )

    assert receipt.total_bytes == 16
    assert all(received is context for received in provider.contexts[:3])
    release_context = provider.contexts[3]
    assert release_context is not context
    assert release_context.cancel_signal is None
    assert 0 < release_context.remaining_seconds()
    assert (
        release_context.remaining_seconds()
        <= weight_transfer_api._PROVIDER_TERMINAL_RELEASE_TIMEOUT_SEC
    )


def test_expired_context_after_prepare_prevents_submit() -> None:
    class SlowPrepareProvider(RecordingProvider):
        def probe(self, request):
            return replace(
                super().probe(request),
                supports_bounded_execution=True,
            )

        def prepare(self, request, *, execution_context=None):
            self.events.append("prepare")
            time.sleep(0.03)
            return request

    source = placement("source")
    target = placement("target")
    provider = SlowPrepareProvider()

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            execution_context=WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + 0.01,
            ),
        )

    assert raised.value.code == "DEADLINE_EXCEEDED"
    assert raised.value.completion_known is True
    assert provider.events == ["probe", "prepare", "release"]


def test_completion_known_deadline_survives_terminal_release() -> None:
    class DeadlineProvider(RecordingProvider):
        def __init__(self):
            super().__init__()
            self.release_contexts = []

        def probe(self, request):
            return replace(
                super().probe(request),
                supports_bounded_execution=True,
            )

        def wait(self, submission, *, execution_context=None):
            self.events.append("wait")
            time.sleep(0.03)
            raise WeightTransferError(
                "provider stopped before starting the commit",
                code="DEADLINE_EXCEEDED",
                provider=self.name,
                phase="wait",
                operation_id=submission.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            )

        def release(self, prepared, receipt, *, execution_context=None):
            self.events.append("release")
            self.release_contexts.append(execution_context)
            if execution_context is None or execution_context.expired():
                raise TimeoutError("release reused the expired business context")

    source = placement("source")
    target = placement("target")
    provider = DeadlineProvider()
    context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 0.01,
    )

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            execution_context=context,
        )

    assert not isinstance(raised.value, WeightTransferReleaseError)
    assert raised.value.code == "DEADLINE_EXCEEDED"
    assert raised.value.completion_known is True
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "cancel",
        "release",
    ]
    release_context = provider.release_contexts[0]
    assert release_context is not context
    assert release_context.cancel_signal is None
    assert 0 < release_context.remaining_seconds()
    assert (
        release_context.remaining_seconds()
        <= weight_transfer_api._PROVIDER_TERMINAL_RELEASE_TIMEOUT_SEC
    )


def test_completion_known_cancellation_survives_terminal_release() -> None:
    cancelled = {"value": False}

    class CancelledProvider(RecordingProvider):
        def __init__(self):
            super().__init__()
            self.release_contexts = []

        def probe(self, request):
            return replace(
                super().probe(request),
                supports_bounded_execution=True,
            )

        def wait(self, submission, *, execution_context=None):
            self.events.append("wait")
            cancelled["value"] = True
            raise WeightTransferError(
                "provider cancelled before starting the commit",
                code="CANCELLED",
                provider=self.name,
                phase="wait",
                operation_id=submission.operation_id,
                retryable=False,
                completion_known=True,
                cleanup_required=True,
            )

        def release(self, prepared, receipt, *, execution_context=None):
            self.events.append("release")
            self.release_contexts.append(execution_context)
            if execution_context is None or execution_context.expired():
                raise TimeoutError("release reused the cancelled business context")

    source = placement("source")
    target = placement("target")
    provider = CancelledProvider()
    context = WeightTransferExecutionContext(
        deadline_unix_sec=10**10,
        cancel_signal=SimpleNamespace(is_set=lambda: cancelled["value"]),
    )

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            execution_context=context,
        )

    assert not isinstance(raised.value, WeightTransferReleaseError)
    assert raised.value.code == "CANCELLED"
    assert raised.value.completion_known is True
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "wait",
        "cancel",
        "release",
    ]
    release_context = provider.release_contexts[0]
    assert release_context is not context
    assert release_context.cancel_signal is None
    assert 0 < release_context.remaining_seconds()
    assert (
        release_context.remaining_seconds()
        <= weight_transfer_api._PROVIDER_TERMINAL_RELEASE_TIMEOUT_SEC
    )


def test_post_submit_deadline_without_safe_cancel_quarantines_owner() -> None:
    class UnsafeSlowSubmitProvider(RecordingProvider):
        def probe(self, request):
            return replace(
                super().probe(request),
                supports_safe_cancel=False,
                supports_bounded_execution=True,
            )

        def submit(self, prepared):
            self.events.append("submit")
            time.sleep(0.03)
            return None

    source = placement("source")
    target = placement("target")
    provider = UnsafeSlowSubmitProvider()
    owner = object()
    coordinator = WeightSnapshotCoordinator()
    lease_id, generation = coordinator.acquire_target_snapshot(full_restore=False)
    target_binding = msgspec.structs.replace(
        binding(target, 0x20000),
        generation=generation,
        lease_id=lease_id,
    )
    session = WeightTargetLoadSession(
        target_bindings=(target_binding,),
        owners=(owner,),
        coordinator=coordinator,
    )

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(target_binding,),
            provider=provider,
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
            target_session=session,
            execution_context=WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + 0.01,
            ),
        )

    assert (
        raised.value.completion_known,
        provider.events,
        session.state,
    ) == (
        False,
        ["probe", "prepare", "submit"],
        WeightTargetLoadState.QUARANTINED,
    )
    assert isinstance(raised.value, WeightTransferCompletionUnknownError)
    assert isinstance(raised.value.__cause__, WeightTransferError)
    assert raised.value.__cause__.code == "DEADLINE_EXCEEDED"
    assert session.owners == (owner,)
    assert session.update_token is not None


def test_post_submit_deadline_safe_cancel_releases_provider() -> None:
    class SafeSlowSubmitProvider(RecordingProvider):
        def probe(self, request):
            return replace(
                super().probe(request),
                supports_bounded_execution=True,
            )

        def submit(self, prepared):
            self.events.append("submit")
            time.sleep(0.03)
            return None

    source = placement("source")
    target = placement("target")
    provider = SafeSlowSubmitProvider()

    with pytest.raises(WeightTransferError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            execution_context=WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + 0.01,
            ),
        )

    assert raised.value.code == "DEADLINE_EXCEEDED"
    assert raised.value.completion_known is True
    assert provider.events == [
        "probe",
        "prepare",
        "submit",
        "cancel",
        "release",
    ]


def test_provider_reported_deadline_unknown_does_not_cancel_or_release() -> None:
    class ProviderDeadlineUnknown(RecordingProvider):
        def wait(self, submission, *, execution_context=None):
            del execution_context
            self.events.append("wait")
            raise WeightTransferError(
                "provider deadline completion is unknown",
                code="DEADLINE_EXCEEDED",
                provider=self.name,
                phase="wait",
                operation_id=submission.operation_id,
                retryable=False,
                completion_known=False,
                cleanup_required=True,
            )

    source = placement("source")
    target = placement("target")
    provider = ProviderDeadlineUnknown()

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.completion_known is False
    assert provider.events == ["probe", "prepare", "submit", "wait"]


class RecordingAttestor:
    def __init__(
        self,
        events: list[str],
        *,
        error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.error = error

    def attest(self, request) -> None:
        assert request.profile == "runtime_to_runtime"
        self.events.append("attest")
        if self.error is not None:
            raise self.error


def test_preflight_reuses_probe_and_reattests_before_execution() -> None:
    class OneShotProvider(RecordingProvider):
        def probe(self, request):
            if "probe" in self.events:
                raise AssertionError("probe ran twice")
            return super().probe(request)

    source = placement("source")
    target = placement("target")
    request = weight_transfer_api.prepare_weight_load(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        target_placements=(target,),
        target_bindings=(binding(target, 0x20000),),
    )
    events = []
    provider = OneShotProvider(events=events)
    attestor = RecordingAttestor(events)

    preflight = weight_transfer_api.preflight_weight_transfer(
        provider,
        request,
        attestor=attestor,
    )
    receipt = weight_transfer_api.execute_weight_load(
        request,
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
        attestor=attestor,
        preflight=preflight,
    )

    assert receipt.operation_id == request.operation_id
    assert events == [
        "attest",
        "probe",
        "attest",
        "prepare",
        "submit",
        "wait",
        "synchronize",
        "release",
    ]


def test_preflight_returns_public_opaque_contract() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider()
    request = weight_transfer_api.prepare_weight_load(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        target_placements=(target,),
        target_bindings=(binding(target, 0x20000),),
    )

    preflight = weight_transfer_api.preflight_weight_transfer(provider, request)

    assert type(preflight) is weight_transfer.WeightTransferPreflight


def test_prepare_weight_load_projects_unselected_source_dp_replica() -> None:
    sources = tuple(placement_for_dp("source", dp_rank) for dp_rank in range(2))
    target = placement_for_dp("target", 0)

    request = weight_transfer_api.prepare_weight_load(
        source_placements=sources,
        source_bindings=tuple(
            binding(source, 0x10000 + index * 0x1000)
            for index, source in enumerate(sources)
        ),
        target_placements=(target,),
        target_bindings=(binding(target, 0x20000),),
    )

    logical = request.plan.logical_plan
    referenced = {
        (region.source.placement_id, region.source.placement_fragment_id)
        for region in logical.regions
    }
    planned = {
        (source.placement_id, tensor.placement_fragment_id)
        for source in logical.source_placements
        for tensor in source.tensors
    }
    bound = {
        (source.placement_id, fragment.placement_fragment_id)
        for source in request.plan.source_bindings
        for fragment in source.fragments
    }

    assert planned == referenced
    assert bound == referenced
    assert {region.source.rank.dp for region in logical.regions} == {0}


def test_materialization_preflight_reuses_canonical_request(monkeypatch) -> None:
    source = placement("source")
    provider = RecordingMaterializeProvider("weights/revision/index.json")
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
    )
    validate = weight_transfer_api.validate_weight_materialize_request
    calls = 0

    def count_validate(actual_request) -> None:
        nonlocal calls
        calls += 1
        validate(actual_request)

    monkeypatch.setattr(
        weight_transfer_api,
        "validate_weight_materialize_request",
        count_validate,
    )

    preflight = weight_transfer_api.preflight_weight_transfer(provider, request)
    execute_weight_materialization(
        request,
        provider=provider,
        preflight=preflight,
    )

    assert calls == 1


def test_preflight_rejects_materialization_location_drift_before_probe() -> None:
    source = placement("source")
    provider = RecordingMaterializeProvider("weights/revision/index.json")
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
    )
    object.__setattr__(
        request,
        "source_locations",
        (replace(request.source_locations[0], address=0xDEADBEEF),),
    )

    with pytest.raises(ValueError, match="source locations differ"):
        weight_transfer_api.preflight_weight_transfer(provider, request)

    assert provider.events == []


def test_execute_revalidates_materialization_after_preflight() -> None:
    source = placement("source")
    provider = RecordingMaterializeProvider("weights/revision/index.json")
    request = prepare_weight_materialization(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/revision",
            object_prefix="weights/revision",
        ),
    )
    preflight = weight_transfer_api.preflight_weight_transfer(provider, request)
    object.__setattr__(
        request,
        "source_locations",
        (replace(request.source_locations[0], address=0xDEADBEEF),),
    )

    with pytest.raises(ValueError, match="preflight"):
        execute_weight_materialization(
            request,
            provider=provider,
            preflight=preflight,
        )

    assert provider.events == ["probe"]


def test_execute_rejects_forged_or_rebound_preflight_before_prepare() -> None:
    source = placement("source")
    target = placement("target")
    request = weight_transfer_api.prepare_weight_load(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        target_placements=(target,),
        target_bindings=(binding(target, 0x20000),),
    )
    provider = RecordingProvider()
    preflight = weight_transfer_api.preflight_weight_transfer(provider, request)

    with pytest.raises(ValueError, match="preflight"):
        weight_transfer_api.execute_weight_load(
            request,
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            preflight=object(),
        )
    with pytest.raises(ValueError, match="preflight"):
        weight_transfer_api.execute_weight_load(
            replace(request, operation_id="other-operation"),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            preflight=preflight,
        )
    with pytest.raises(ValueError, match="preflight"):
        weight_transfer_api.execute_weight_load(
            request,
            provider=RecordingProvider(),
            target_mode=WeightTargetLoadMode.COLD_START,
            preflight=preflight,
        )

    assert provider.events == ["probe"]


def test_attestor_runs_before_provider_probe() -> None:
    source = placement("source")
    target = placement("target")
    events = []
    provider = RecordingProvider(events=events)

    receipt = load_weights(
        source_placements=(source,),
        source_bindings=(binding(source, 0x10000),),
        target_placements=(target,),
        target_bindings=(binding(target, 0x20000),),
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
        attestor=RecordingAttestor(events),
    )

    assert events[:2] == ["attest", "probe"]
    assert receipt.provider_phase_seconds[0][0] == "attest"


def test_attestation_failure_never_probes_provider() -> None:
    source = placement("source")
    target = placement("target")
    events = []
    provider = RecordingProvider(events=events)

    with pytest.raises(ValueError, match="source lease was revoked"):
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            attestor=RecordingAttestor(
                events,
                error=ValueError("source lease was revoked"),
            ),
        )

    assert events == ["attest"]


def test_required_attestation_fails_before_provider_probe() -> None:
    source = placement("source")
    target = placement("target")
    provider = RecordingProvider()
    provider.requires_runtime_attestation = True

    with pytest.raises(WeightTransferError, match="attestor is required") as raised:
        load_weights(
            source_placements=(source,),
            source_bindings=(binding(source, 0x10000),),
            target_placements=(target,),
            target_bindings=(binding(target, 0x20000),),
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
        )

    assert raised.value.code == "ATTESTATION_REQUIRED"
    assert raised.value.phase == "attest"
    assert provider.events == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
