from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable, Sequence, cast

from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
)
from sglang.srt.weight_transfer.binding import (
    bind_weight_source,
    bind_weight_transfer_plan,
    project_source_bindings,
)
from sglang.srt.weight_transfer.contracts import (
    LogicalWeightTransferPlan,
    RuntimeWeightLocation,
    SourceBindingManifest,
)
from sglang.srt.weight_transfer.planner import (
    plan_weight_transfer,
    plan_weight_transfer_to_local_target,
    select_weight_storage_placements,
)
from sglang.srt.weight_transfer.provider import (
    WeightLoadReceipt,
    WeightLoadRequest,
    WeightMaterializeReceipt,
    WeightMaterializeRequest,
    WeightPayloadIdentity,
    WeightProviderCapabilities,
    WeightProviderReceipt,
    WeightStorageDestination,
    WeightTargetLoadMode,
    WeightTargetLoadSession,
    WeightTransferAttestor,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
    WeightTransferProvider,
    WeightTransferReleaseError,
    new_operation_id,
)
from sglang.srt.weight_transfer.storage import (
    StoredWeightSnapshot,
    WeightMaterializationAttemptState,
    WeightMaterializationIntent,
    WeightRevisionHead,
    WeightRevisionState,
    WeightSnapshotPublication,
    WeightSnapshotPublicationState,
    WeightStorageCatalog,
    WeightStorageRef,
    weight_placement_set_digest,
    weight_source_snapshot_digest,
)


def _snapshot_identity(
    snapshot: StoredWeightSnapshot,
) -> tuple[str, str]:
    identities = {
        (placement.model_id, placement.revision) for placement in snapshot.placements
    }
    if len(identities) != 1:
        raise ValueError("weight snapshot has no canonical model revision")
    return next(iter(identities))


def _require_loadable_revision_head(
    catalog: WeightStorageCatalog,
    snapshot: StoredWeightSnapshot,
) -> WeightRevisionHead:
    model_id, revision = _snapshot_identity(snapshot)
    head = catalog.get_revision_head(model_id, revision)
    if (
        head is None
        or head.ref != snapshot.ref
        or head.state
        not in {
            WeightRevisionState.READY,
            WeightRevisionState.SERVING,
        }
    ):
        raise ValueError(
            "weight snapshot revision head must reference this ref in "
            "READY or SERVING state"
        )
    return head


def _mark_publication_ready(
    catalog: WeightStorageCatalog,
    publication: WeightSnapshotPublication,
) -> WeightSnapshotPublication:
    snapshot = publication.snapshot
    model_id, revision = _snapshot_identity(snapshot)
    head = catalog.get_revision_head(model_id, revision)
    if head is None:
        head = catalog.compare_and_set_revision(
            model_id=model_id,
            revision=revision,
            expected=None,
            new_ref=snapshot.ref,
            new_state=WeightRevisionState.READY,
        )
        if head is None:
            head = catalog.get_revision_head(model_id, revision)
    if (
        head is None
        or head.ref != snapshot.ref
        or head.state
        not in {
            WeightRevisionState.READY,
            WeightRevisionState.SERVING,
        }
    ):
        raise ValueError("published weight snapshot could not enter READY state")
    return publication


def mark_weight_snapshot_serving(
    ref: WeightStorageRef,
    *,
    catalog: WeightStorageCatalog,
) -> WeightRevisionHead:
    """Atomically mark a fully initialized snapshot as serving."""

    snapshot = catalog.get_snapshot(ref)
    if snapshot is None:
        raise ValueError("published weight snapshot was not found")
    head = _require_loadable_revision_head(catalog, snapshot)
    if head.state is WeightRevisionState.SERVING:
        return head
    updated = catalog.compare_and_set_revision(
        model_id=head.model_id,
        revision=head.revision,
        expected=head,
        new_ref=ref,
        new_state=WeightRevisionState.SERVING,
    )
    if updated is not None:
        return updated
    reconciled = _require_loadable_revision_head(catalog, snapshot)
    if reconciled.state is not WeightRevisionState.SERVING:
        raise RuntimeError("weight snapshot could not enter SERVING state")
    return reconciled


def _capability_error(
    message: str,
    *,
    provider: str,
    operation_id: str,
) -> WeightTransferError:
    return WeightTransferError(
        message,
        code="UNSUPPORTED_CAPABILITY",
        provider=provider,
        phase="probe",
        operation_id=operation_id,
        retryable=False,
        completion_known=True,
        cleanup_required=False,
    )


def _validate_load_capabilities(
    request: WeightLoadRequest,
    capabilities: WeightProviderCapabilities,
) -> None:
    if request.profile not in capabilities.load_profiles:
        raise _capability_error(
            f"provider does not support load profile {request.profile}",
            provider=capabilities.provider,
            operation_id=request.operation_id,
        )
    if not capabilities.supports_nd_regions:
        raise _capability_error(
            "provider does not support N-D regions",
            provider=capabilities.provider,
            operation_id=request.operation_id,
        )
    if (
        any(region.outer_loop_counts for region in request.plan.regions)
        and not capabilities.supports_strided_regions
    ):
        raise _capability_error(
            "provider does not support strided regions",
            provider=capabilities.provider,
            operation_id=request.operation_id,
        )
    if (
        capabilities.max_regions is not None
        and len(request.plan.regions) > capabilities.max_regions
    ):
        raise _capability_error(
            "transfer plan exceeds provider region limit",
            provider=capabilities.provider,
            operation_id=request.operation_id,
        )
    if capabilities.max_segments_per_region is not None and any(
        region.segment_count > capabilities.max_segments_per_region
        for region in request.plan.regions
    ):
        raise _capability_error(
            "transfer plan exceeds provider segment limit",
            provider=capabilities.provider,
            operation_id=request.operation_id,
        )
    if (
        capabilities.max_total_operations is not None
        and request.plan.total_segments > capabilities.max_total_operations
    ):
        raise _capability_error(
            "transfer plan exceeds provider total operation limit",
            provider=capabilities.provider,
            operation_id=request.operation_id,
        )
    if (
        capabilities.max_total_bytes is not None
        and request.plan.total_bytes > capabilities.max_total_bytes
    ):
        raise _capability_error(
            "transfer plan exceeds provider byte limit",
            provider=capabilities.provider,
            operation_id=request.operation_id,
        )


def _validate_materialize_capabilities(
    request: WeightMaterializeRequest,
    capabilities: WeightProviderCapabilities,
) -> None:
    if request.profile not in capabilities.materialize_profiles:
        raise _capability_error(
            f"provider does not support materialize profile {request.profile}",
            provider=capabilities.provider,
            operation_id=request.operation_id,
        )
    if not capabilities.supports_transactional_publish:
        raise _capability_error(
            "provider does not support transactional publish",
            provider=capabilities.provider,
            operation_id=request.operation_id,
        )
    if (
        capabilities.max_total_bytes is not None
        and request.total_bytes > capabilities.max_total_bytes
    ):
        raise _capability_error(
            "materialization exceeds provider byte limit",
            provider=capabilities.provider,
            operation_id=request.operation_id,
        )


def _execute(
    provider: WeightTransferProvider,
    request: WeightLoadRequest | WeightMaterializeRequest,
    *,
    attestor: WeightTransferAttestor | None = None,
    target_session: WeightTargetLoadSession | None = None,
    completion_ticket_sink: Callable[[str], None] | None = None,
) -> WeightProviderReceipt:
    provider_phase_seconds = []
    requires_attestation = getattr(
        provider,
        "requires_runtime_attestation",
        True,
    )
    if type(requires_attestation) is not bool:
        raise ValueError("provider requires_runtime_attestation must be a boolean")
    if requires_attestation and attestor is None:
        raise WeightTransferError(
            "a runtime attestor is required by this provider",
            code="ATTESTATION_REQUIRED",
            provider=provider.name,
            phase="attest",
            operation_id=request.operation_id,
            retryable=False,
            completion_known=True,
            cleanup_required=False,
        )
    if attestor is not None:
        phase_started = time.perf_counter()
        attestor.attest(request)
        provider_phase_seconds.append(("attest", time.perf_counter() - phase_started))
    phase_started = time.perf_counter()
    capabilities = provider.probe(request)
    if isinstance(request, WeightLoadRequest):
        _validate_load_capabilities(request, capabilities)
    else:
        _validate_materialize_capabilities(request, capabilities)
    provider_phase_seconds.append(("probe", time.perf_counter() - phase_started))

    prepared = None
    submission = None
    receipt = None
    try:
        phase_started = time.perf_counter()
        prepared = provider.prepare(request)
        provider_phase_seconds.append(("prepare", time.perf_counter() - phase_started))
        if (
            isinstance(request, WeightMaterializeRequest)
            and capabilities.supports_completion_ticket
        ):
            ticket_factory = getattr(
                provider,
                "materialization_recovery_ticket",
                None,
            )
            if not callable(ticket_factory):
                raise ValueError(
                    "provider advertises completion tickets without a ticket factory"
                )
            completion_ticket = ticket_factory(prepared)
            if type(completion_ticket) is not str or not completion_ticket:
                raise ValueError(
                    "provider returned an invalid materialization recovery ticket"
                )
            if completion_ticket_sink is not None:
                completion_ticket_sink(completion_ticket)
        phase_started = time.perf_counter()
        if target_session is not None:
            target_session.mark_mutating()
        submission = provider.submit(prepared)
        provider_phase_seconds.append(("submit", time.perf_counter() - phase_started))
        phase_started = time.perf_counter()
        receipt = provider.wait(submission)
        provider_phase_seconds.append(("wait", time.perf_counter() - phase_started))
        phase_started = time.perf_counter()
        provider.synchronize(receipt)
        provider_phase_seconds.append(
            ("synchronize", time.perf_counter() - phase_started)
        )
    except BaseException as error:
        if isinstance(error, WeightTransferError) and not error.completion_known:
            raise

        release_safe = submission is None
        cancel_error = None
        if submission is not None and capabilities.supports_safe_cancel:
            try:
                provider.cancel(submission)
                release_safe = True
            except BaseException as caught:
                cancel_error = caught
        if isinstance(error, WeightTransferError) and error.completion_known:
            release_safe = True
        if not release_safe:
            detail = str(error) or error.__class__.__name__
            if cancel_error is not None:
                detail = f"{detail}; cancel did not confirm completion: {cancel_error}"
            raise WeightTransferCompletionUnknownError(
                detail,
                provider=capabilities.provider,
                phase="cancel" if cancel_error is not None else "execute",
                operation_id=request.operation_id,
            ) from error

        if prepared is not None:
            try:
                provider.release(prepared, None)
            except BaseException:
                pass
        if isinstance(error, WeightTransferError):
            raise
        raise WeightTransferError(
            str(error),
            code="BACKEND_FAILURE",
            provider=capabilities.provider,
            phase="execute",
            operation_id=request.operation_id,
            retryable=False,
            completion_known=True,
            cleanup_required=True,
        ) from error
    completed_receipt = replace(
        receipt,
        provider_phase_seconds=tuple(provider_phase_seconds),
    )
    phase_started = time.perf_counter()
    try:
        provider.release(prepared, completed_receipt)
    except BaseException as error:
        raise WeightTransferReleaseError(
            str(error),
            receipt=completed_receipt,
        ) from error
    return replace(
        completed_receipt,
        provider_phase_seconds=(
            *completed_receipt.provider_phase_seconds,
            ("release", time.perf_counter() - phase_started),
        ),
    )


def prepare_weight_load(
    *,
    source_placements: Sequence[WeightPlacementManifest],
    source_bindings: Sequence[SourceBindingManifest],
    target_placements: Sequence[WeightPlacementManifest],
    target_bindings: Sequence[WeightRuntimeBindingManifest],
) -> WeightLoadRequest:
    """Plan and bind a load without probing or calling a provider."""

    logical_plan = plan_weight_transfer(source_placements, target_placements)
    return prepare_weight_load_from_plan(
        logical_plan,
        source_bindings=source_bindings,
        target_bindings=target_bindings,
    )


def prepare_weight_load_to_local_target(
    *,
    source_placements: Sequence[WeightPlacementManifest],
    source_bindings: Sequence[SourceBindingManifest],
    target_placement: WeightPlacementManifest,
    target_binding: WeightRuntimeBindingManifest,
) -> WeightLoadRequest:
    """Plan and bind the fragments owned by one local target executor."""

    logical_plan = plan_weight_transfer_to_local_target(
        source_placements,
        target_placement,
    )
    return prepare_weight_load_from_plan(
        logical_plan,
        source_bindings=source_bindings,
        target_bindings=(target_binding,),
    )


def prepare_weight_load_from_plan(
    logical_plan: LogicalWeightTransferPlan,
    *,
    source_bindings: Sequence[SourceBindingManifest],
    target_bindings: Sequence[WeightRuntimeBindingManifest],
) -> WeightLoadRequest:
    """Bind a previously planned target world without provider side effects."""

    bound_plan = bind_weight_transfer_plan(
        logical_plan,
        source_bindings=source_bindings,
        target_bindings=target_bindings,
    )
    profile = (
        "runtime_to_runtime"
        if isinstance(bound_plan.regions[0].source, RuntimeWeightLocation)
        else "storage_to_runtime"
    )
    return WeightLoadRequest(
        operation_id=new_operation_id(),
        plan=bound_plan,
        profile=profile,
    )


def execute_weight_load(
    request: WeightLoadRequest,
    *,
    provider: WeightTransferProvider,
    target_mode: WeightTargetLoadMode,
    attestor: WeightTransferAttestor | None = None,
    target_session: WeightTargetLoadSession | None = None,
) -> WeightLoadReceipt:
    """Execute one preflight-validated load request."""

    if not isinstance(target_mode, WeightTargetLoadMode):
        raise ValueError("target_mode must explicitly select cold start or live update")
    if target_mode is WeightTargetLoadMode.COLD_START:
        if target_session is not None:
            raise ValueError("cold-start loads must not use a live target session")
    elif target_session is None:
        raise ValueError("live target mutation requires a target load session")

    if target_session is not None:
        target_session.begin(request, provider_name=provider.name)
    try:
        receipt = cast(
            WeightLoadReceipt,
            _execute(
                provider,
                request,
                attestor=attestor,
                target_session=target_session,
            ),
        )
        if target_session is not None:
            target_session.complete_transfer(receipt)
    except BaseException as error:
        if target_session is not None:
            target_session.fail(error)
        raise
    return receipt


def load_weights(
    *,
    source_placements: Sequence[WeightPlacementManifest],
    source_bindings: Sequence[SourceBindingManifest],
    target_placements: Sequence[WeightPlacementManifest],
    target_bindings: Sequence[WeightRuntimeBindingManifest],
    provider: WeightTransferProvider,
    target_mode: WeightTargetLoadMode,
    attestor: WeightTransferAttestor | None = None,
    target_session: WeightTargetLoadSession | None = None,
) -> WeightLoadReceipt:
    """Plan, bind, and execute a provider-neutral heterogeneous load."""

    request = prepare_weight_load(
        source_placements=source_placements,
        source_bindings=source_bindings,
        target_placements=target_placements,
        target_bindings=target_bindings,
    )
    return execute_weight_load(
        request,
        provider=provider,
        target_mode=target_mode,
        attestor=attestor,
        target_session=target_session,
    )


def load_weights_to_local_target(
    *,
    source_placements: Sequence[WeightPlacementManifest],
    source_bindings: Sequence[SourceBindingManifest],
    target_placement: WeightPlacementManifest,
    target_binding: WeightRuntimeBindingManifest,
    provider: WeightTransferProvider,
    target_mode: WeightTargetLoadMode,
    attestor: WeightTransferAttestor | None = None,
    target_session: WeightTargetLoadSession | None = None,
) -> WeightLoadReceipt:
    """Plan, bind, and load the fragments owned by one local executor."""

    request = prepare_weight_load_to_local_target(
        source_placements=source_placements,
        source_bindings=source_bindings,
        target_placement=target_placement,
        target_binding=target_binding,
    )
    return execute_weight_load(
        request,
        provider=provider,
        target_mode=target_mode,
        attestor=attestor,
        target_session=target_session,
    )


def prepare_weight_materialization(
    *,
    source_placements: Sequence[WeightPlacementManifest],
    source_bindings: Sequence[SourceBindingManifest],
    destination: WeightStorageDestination,
    payload_identity: WeightPayloadIdentity | None = None,
    operation_id: str | None = None,
) -> WeightMaterializeRequest:
    """Validate and select one complete source replica without calling a provider."""

    normalized_placements = tuple(
        sorted(source_placements, key=lambda item: item.placement_id)
    )
    normalized_bindings = tuple(source_bindings)
    bind_weight_source(
        normalized_placements,
        normalized_bindings,
    )
    stored_placements = select_weight_storage_placements(normalized_placements)
    stored_bindings = project_source_bindings(
        stored_placements,
        normalized_bindings,
    )
    locations = bind_weight_source(stored_placements, stored_bindings)
    selected_payload_identity = (
        None if payload_identity is None else payload_identity.select(stored_placements)
    )
    profile = (
        "runtime_to_storage"
        if isinstance(locations[0], RuntimeWeightLocation)
        else "storage_to_storage"
    )
    return WeightMaterializeRequest(
        operation_id=operation_id or new_operation_id(),
        source_placements=stored_placements,
        source_bindings=stored_bindings,
        source_locations=locations,
        destination=destination,
        profile=profile,
        payload_identity=selected_payload_identity,
    )


def execute_weight_materialization(
    request: WeightMaterializeRequest,
    *,
    provider: WeightTransferProvider,
    attestor: WeightTransferAttestor | None = None,
    completion_ticket_sink: Callable[[str], None] | None = None,
) -> WeightMaterializeReceipt:
    """Execute one previously validated materialization request."""

    return cast(
        WeightMaterializeReceipt,
        _execute(
            provider,
            request,
            attestor=attestor,
            completion_ticket_sink=completion_ticket_sink,
        ),
    )


def materialize_weights(
    *,
    source_placements: Sequence[WeightPlacementManifest],
    source_bindings: Sequence[SourceBindingManifest],
    destination: WeightStorageDestination,
    provider: WeightTransferProvider,
    payload_identity: WeightPayloadIdentity | None = None,
    attestor: WeightTransferAttestor | None = None,
) -> WeightMaterializeReceipt:
    """Write one validated source snapshot to a transactional destination."""

    request = prepare_weight_materialization(
        source_placements=source_placements,
        source_bindings=source_bindings,
        destination=destination,
        payload_identity=payload_identity,
    )
    return execute_weight_materialization(
        request,
        provider=provider,
        attestor=attestor,
    )


def materialize_weight_snapshot(
    *,
    source_placements: Sequence[WeightPlacementManifest],
    source_bindings: Sequence[SourceBindingManifest],
    destination: WeightStorageDestination,
    provider: WeightTransferProvider,
    catalog: WeightStorageCatalog,
    payload_identity: WeightPayloadIdentity | None = None,
    publication_id: str | None = None,
    attestor: WeightTransferAttestor | None = None,
) -> WeightSnapshotPublication:
    """Materialize payloads, then publish their semantic storage snapshot."""

    if provider.name != destination.provider:
        raise ValueError(
            "weight transfer provider differs from storage destination provider"
        )
    materialization_id = publication_id or new_operation_id()
    request = prepare_weight_materialization(
        source_placements=source_placements,
        source_bindings=source_bindings,
        destination=destination,
        payload_identity=payload_identity,
        operation_id=materialization_id,
    )
    source = request.source_placements[0]
    intent = WeightMaterializationIntent(
        provider=destination.provider,
        storage_id=destination.storage_id,
        object_prefix=destination.object_prefix,
        model_id=source.model_id,
        revision=source.revision,
        source_digest=weight_placement_set_digest(request.source_placements),
        total_bytes=request.total_bytes,
        fragment_count=len(request.source_locations),
        source_snapshot_digest=weight_source_snapshot_digest(
            request.source_placements,
            request.source_bindings,
        ),
        payload_digest=(
            None
            if request.payload_identity is None
            else request.payload_identity.payload_digest
        ),
    )
    previous_attempt = catalog.get_materialization(materialization_id)
    recover_existing_attempt = previous_attempt is not None
    if previous_attempt is not None and previous_attempt.completion_ticket is not None:
        previous_intent = previous_attempt.intent
        if (
            previous_intent.provider != intent.provider
            or previous_intent.storage_id != intent.storage_id
            or previous_intent.object_prefix != intent.object_prefix
            or previous_intent.model_id != intent.model_id
            or previous_intent.revision != intent.revision
            or previous_intent.source_digest != intent.source_digest
            or previous_intent.total_bytes != intent.total_bytes
            or previous_intent.fragment_count != intent.fragment_count
            or (
                previous_intent.payload_digest is not None
                and previous_intent.payload_digest != intent.payload_digest
            )
        ):
            raise ValueError(
                "materialization recovery request differs from its original intent"
            )
        attempt = previous_attempt
    else:
        attempt = catalog.begin_materialization(materialization_id, intent)
    current = catalog.get_publication(materialization_id)
    if current is not None:
        if attempt.state is not WeightMaterializationAttemptState.MATERIALIZED:
            raise ValueError(
                "snapshot publication exists before materialization completed"
            )
        if current.snapshot != attempt.snapshot:
            raise ValueError("snapshot publication differs from its materialization")
        if current.state is WeightSnapshotPublicationState.PENDING:
            return _mark_publication_ready(
                catalog,
                catalog.publish(materialization_id),
            )
        if current.state is WeightSnapshotPublicationState.PUBLISHED:
            return _mark_publication_ready(catalog, current)
        raise ValueError("aborted publication cannot be retried")
    if attempt.state is WeightMaterializationAttemptState.MATERIALIZED:
        pending = catalog.prepare_publish(
            materialization_id,
            attempt.snapshot,
        )
        return _mark_publication_ready(
            catalog,
            catalog.publish(pending.publication_id),
        )

    release_error = None
    try:
        if attempt.completion_ticket is not None:
            recover = getattr(provider, "recover_materialization", None)
            if not callable(recover):
                raise WeightTransferCompletionUnknownError(
                    "provider cannot recover an interrupted materialization",
                    provider=provider.name,
                    phase="recover",
                    operation_id=materialization_id,
                    completion_ticket=attempt.completion_ticket,
                )
            receipt = recover(
                request,
                completion_ticket=attempt.completion_ticket,
            )
            if receipt is None:
                raise WeightTransferCompletionUnknownError(
                    "provider did not resolve the persisted completion ticket",
                    provider=provider.name,
                    phase="recover",
                    operation_id=materialization_id,
                    completion_ticket=attempt.completion_ticket,
                )
        elif recover_existing_attempt:
            recover = getattr(provider, "recover_materialization", None)
            receipt = (
                None
                if not callable(recover)
                else recover(
                    request,
                    completion_ticket=None,
                )
            )
            if receipt is None:
                receipt = execute_weight_materialization(
                    request,
                    provider=provider,
                    attestor=attestor,
                    completion_ticket_sink=lambda ticket: (
                        catalog.set_materialization_completion_ticket(
                            materialization_id,
                            ticket,
                        )
                    ),
                )
        else:
            receipt = execute_weight_materialization(
                request,
                provider=provider,
                attestor=attestor,
                completion_ticket_sink=lambda ticket: (
                    catalog.set_materialization_completion_ticket(
                        materialization_id,
                        ticket,
                    )
                ),
            )
    except WeightTransferReleaseError as error:
        if not isinstance(error.receipt, WeightMaterializeReceipt):
            raise
        receipt = error.receipt
        release_error = error
    except WeightTransferCompletionUnknownError as error:
        if error.completion_ticket is not None:
            catalog.set_materialization_completion_ticket(
                materialization_id,
                error.completion_ticket,
            )
        raise
    except WeightTransferError:
        catalog.abort_materialization(materialization_id)
        raise
    if receipt.completion_ticket is not None:
        catalog.set_materialization_completion_ticket(
            materialization_id,
            receipt.completion_ticket,
        )
    snapshot = _snapshot_from_materialize_receipt(destination, receipt)
    completed = catalog.complete_materialization(
        materialization_id,
        snapshot,
    )
    if completed.snapshot != snapshot:
        raise ValueError("catalog materialization snapshot differs")
    pending = catalog.prepare_publish(
        materialization_id,
        snapshot,
    )
    if release_error is not None:
        release_error.publication = pending
        raise release_error
    return _mark_publication_ready(
        catalog,
        catalog.publish(pending.publication_id),
    )


def _snapshot_from_materialize_receipt(
    destination: WeightStorageDestination,
    receipt: WeightMaterializeReceipt,
) -> StoredWeightSnapshot:
    return StoredWeightSnapshot.create(
        provider=destination.provider,
        storage_id=destination.storage_id,
        manifest_key=receipt.manifest_key,
        placements=receipt.stored_placements,
        storage_bindings=receipt.storage_bindings,
    )


def load_weight_snapshot(
    ref: WeightStorageRef,
    *,
    catalog: WeightStorageCatalog,
    target_placements: Sequence[WeightPlacementManifest],
    target_bindings: Sequence[WeightRuntimeBindingManifest],
    provider: WeightTransferProvider,
    target_mode: WeightTargetLoadMode,
    target_session: WeightTargetLoadSession | None = None,
    attestor: WeightTransferAttestor | None = None,
) -> WeightLoadReceipt:
    """Resolve a published snapshot ref, then plan and load it."""

    snapshot = catalog.get_snapshot(ref)
    if snapshot is None:
        raise ValueError("published weight snapshot was not found")
    _require_loadable_revision_head(catalog, snapshot)
    if provider.name != ref.provider:
        raise ValueError(
            "weight transfer provider differs from storage snapshot provider"
        )
    if len(target_placements) != 1 or len(target_bindings) != 1:
        raise ValueError(
            "weight snapshot loading requires exactly one local target "
            "placement and binding"
        )
    return load_weights_to_local_target(
        source_placements=snapshot.placements,
        source_bindings=snapshot.storage_bindings,
        target_placement=target_placements[0],
        target_binding=target_bindings[0],
        provider=provider,
        target_mode=target_mode,
        attestor=attestor,
        target_session=target_session,
    )
