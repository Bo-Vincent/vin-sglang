from __future__ import annotations

import time
from dataclasses import dataclass, replace
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
    WeightMaterializationCompletionTicketProvider,
    WeightMaterializationRecoveryCleanupProvider,
    WeightMaterializationRecoveryProvider,
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
    WeightTransferExecutionContext,
    WeightTransferProvider,
    WeightTransferReleaseError,
    new_operation_id,
    validate_weight_materialize_request,
)
from sglang.srt.weight_transfer.storage import (
    StoredWeightSnapshot,
    WeightMaterializationAttempt,
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

_PROVIDER_TERMINAL_RELEASE_TIMEOUT_SEC = 5.0


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
    if head is not None and head.ref != snapshot.ref:
        catalog.abort(publication.publication_id)
        raise ValueError("published weight snapshot could not enter READY state")
    if publication.state is WeightSnapshotPublicationState.PENDING:
        publication = catalog.publish(publication.publication_id)
    elif publication.state is not WeightSnapshotPublicationState.PUBLISHED:
        raise ValueError("aborted publication cannot enter READY state")
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
        catalog.abort(publication.publication_id)
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
        capabilities.max_total_operations is not None
        and len(request.source_locations) > capabilities.max_total_operations
    ):
        raise _capability_error(
            "materialization exceeds provider total operation limit",
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


def _invalid_receipt(
    message: str,
    *,
    provider: str,
    operation_id: str,
) -> WeightTransferError:
    return WeightTransferError(
        f"provider receipt is invalid: {message}",
        code="INVALID_RECEIPT",
        provider=provider,
        phase="wait",
        operation_id=operation_id,
        retryable=False,
        completion_known=False,
        cleanup_required=True,
    )


def _validate_provider_receipt(
    request: WeightLoadRequest | WeightMaterializeRequest,
    receipt: object,
    *,
    provider: str,
    completion_ticket: str | None,
) -> WeightProviderReceipt:
    if isinstance(request, WeightLoadRequest):
        if not isinstance(receipt, WeightLoadReceipt):
            raise _invalid_receipt(
                "load operation returned the wrong receipt type",
                provider=provider,
                operation_id=request.operation_id,
            )
        if (
            receipt.operation_id != request.operation_id
            or receipt.provider != provider
            or receipt.plan_digest != request.plan.digest
            or receipt.total_bytes != request.plan.total_bytes
            or receipt.region_count != len(request.plan.regions)
        ):
            raise _invalid_receipt(
                "load receipt differs from the request",
                provider=provider,
                operation_id=request.operation_id,
            )
        return receipt

    if not isinstance(receipt, WeightMaterializeReceipt):
        raise _invalid_receipt(
            "materialization returned the wrong receipt type",
            provider=provider,
            operation_id=request.operation_id,
        )
    manifest_prefix = f"{request.destination.object_prefix.rstrip('/')}/"
    manifest_key_is_valid = (
        isinstance(receipt.manifest_key, str)
        and receipt.manifest_key.startswith(manifest_prefix)
        and len(receipt.manifest_key) > len(manifest_prefix)
    )
    if (
        receipt.operation_id != request.operation_id
        or receipt.provider != provider
        or not manifest_key_is_valid
        or receipt.stored_placements != request.source_placements
        or receipt.total_bytes != request.total_bytes
        or receipt.fragment_count != len(request.source_locations)
        or receipt.completion_ticket != completion_ticket
    ):
        raise _invalid_receipt(
            "materialization receipt differs from the request",
            provider=provider,
            operation_id=request.operation_id,
        )
    try:
        _snapshot_from_materialize_receipt(request.destination, receipt)
    except (TypeError, ValueError) as error:
        raise _invalid_receipt(
            str(error),
            provider=provider,
            operation_id=request.operation_id,
        ) from error
    return receipt


_PREFLIGHT_SEAL = object()


@dataclass(frozen=True, slots=True)
class WeightTransferPreflight:
    """Validated provider and request binding for one execution."""

    _provider: WeightTransferProvider
    _request: WeightLoadRequest | WeightMaterializeRequest
    _attestor: WeightTransferAttestor | None
    _capabilities: WeightProviderCapabilities
    _phase_seconds: tuple[tuple[str, float], ...]
    _request_state: tuple
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _PREFLIGHT_SEAL:
            raise ValueError("weight transfer preflight cannot be constructed directly")


def _request_state(
    request: WeightLoadRequest | WeightMaterializeRequest,
) -> tuple:
    if isinstance(request, WeightLoadRequest):
        return (
            request.operation_id,
            request.plan,
            request.profile,
        )
    return (
        request.operation_id,
        request.source_placements,
        request.source_bindings,
        request.source_locations,
        request.destination,
        request.profile,
        request.payload_identity,
    )


def _validate_request_contract(
    request: WeightLoadRequest | WeightMaterializeRequest,
) -> None:
    if isinstance(request, WeightMaterializeRequest):
        validate_weight_materialize_request(request)
    elif not isinstance(request, WeightLoadRequest):
        raise ValueError("weight transfer request is invalid")


def _require_materialization_recovery_provider(
    provider: WeightTransferProvider,
) -> WeightMaterializationRecoveryProvider:
    if not isinstance(provider, WeightMaterializationRecoveryProvider):
        raise ValueError("provider does not implement materialization recovery")
    return provider


def _require_materialization_completion_ticket_provider(
    provider: WeightTransferProvider,
) -> WeightMaterializationCompletionTicketProvider:
    if not isinstance(provider, WeightMaterializationCompletionTicketProvider):
        raise ValueError(
            "provider advertises completion tickets without the recovery protocol"
        )
    return provider


def preflight_weight_transfer(
    provider: WeightTransferProvider,
    request: WeightLoadRequest | WeightMaterializeRequest,
    *,
    attestor: WeightTransferAttestor | None = None,
) -> WeightTransferPreflight:
    """Validate one request without entering provider collectives."""

    _validate_request_contract(request)
    phase_seconds = []
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
        phase_seconds.append(("attest", time.perf_counter() - phase_started))

    phase_started = time.perf_counter()
    capabilities = provider.probe(request)
    if capabilities.provider != provider.name:
        raise _capability_error(
            "provider capability identity differs from provider name",
            provider=provider.name,
            operation_id=request.operation_id,
        )
    if isinstance(request, WeightLoadRequest):
        _validate_load_capabilities(request, capabilities)
    else:
        _validate_materialize_capabilities(request, capabilities)
        if capabilities.supports_completion_ticket:
            _require_materialization_completion_ticket_provider(provider)
    phase_seconds.append(("probe", time.perf_counter() - phase_started))
    return WeightTransferPreflight(
        _provider=provider,
        _request=request,
        _attestor=attestor,
        _capabilities=capabilities,
        _phase_seconds=tuple(phase_seconds),
        _request_state=_request_state(request),
        _seal=_PREFLIGHT_SEAL,
    )


def _validate_preflight(
    preflight: object,
    *,
    provider: WeightTransferProvider,
    request: WeightLoadRequest | WeightMaterializeRequest,
    attestor: WeightTransferAttestor | None,
) -> WeightTransferPreflight:
    if (
        type(preflight) is not WeightTransferPreflight
        or preflight._seal is not _PREFLIGHT_SEAL
        or preflight._provider is not provider
        or preflight._request is not request
        or (attestor is not None and preflight._attestor is not attestor)
        or preflight._request_state != _request_state(request)
    ):
        raise ValueError(
            "weight transfer preflight is invalid or bound to another execution"
        )
    return preflight


def _completion_unknown_error(
    error: WeightTransferError,
    *,
    provider: str,
    operation_id: str,
    completion_ticket: str | None,
) -> WeightTransferCompletionUnknownError:
    reported_ticket = getattr(error, "completion_ticket", None)
    detail = str(error) or error.__class__.__name__
    if (
        completion_ticket is not None
        and reported_ticket is not None
        and reported_ticket != completion_ticket
    ):
        detail += "; provider returned a different completion ticket"
    return WeightTransferCompletionUnknownError(
        detail,
        provider=provider,
        phase=error.phase,
        operation_id=operation_id,
        completion_ticket=completion_ticket or reported_ticket,
    )


def _release_provider(
    provider: WeightTransferProvider,
    prepared,
    receipt: WeightProviderReceipt | None,
    execution_context: WeightTransferExecutionContext | None,
) -> None:
    release_context = _terminal_execution_context(execution_context)
    if release_context is None:
        provider.release(prepared, receipt)
    else:
        provider.release(
            prepared,
            receipt,
            execution_context=release_context,
        )


def _terminal_execution_context(
    execution_context: WeightTransferExecutionContext | None,
) -> WeightTransferExecutionContext | None:
    if execution_context is None:
        return None
    now = time.time()
    deadline_unix_sec = execution_context.deadline_unix_sec
    if deadline_unix_sec <= now:
        deadline_unix_sec = now + _PROVIDER_TERMINAL_RELEASE_TIMEOUT_SEC
    else:
        deadline_unix_sec = min(
            deadline_unix_sec,
            now + _PROVIDER_TERMINAL_RELEASE_TIMEOUT_SEC,
        )
    return WeightTransferExecutionContext(deadline_unix_sec=deadline_unix_sec)


def _discard_materialization_recovery(
    provider: WeightTransferProvider,
    request: WeightMaterializeRequest,
    completion_ticket: str | None,
    execution_context: WeightTransferExecutionContext | None,
) -> None:
    if completion_ticket is None:
        return
    if not isinstance(provider, WeightMaterializationRecoveryCleanupProvider):
        raise ValueError(
            "provider returned a completion ticket without recovery cleanup"
        )
    terminal_context = _terminal_execution_context(execution_context)
    if terminal_context is None:
        provider.discard_materialization_recovery(
            request,
            completion_ticket=completion_ticket,
        )
    else:
        provider.discard_materialization_recovery(
            request,
            completion_ticket=completion_ticket,
            execution_context=terminal_context,
        )


def _finalize_materialization_recovery(
    provider: WeightTransferProvider,
    catalog: WeightStorageCatalog,
    request: WeightMaterializeRequest,
    attempt: WeightMaterializationAttempt,
    execution_context: WeightTransferExecutionContext | None,
) -> WeightMaterializationAttempt:
    completion_ticket = attempt.completion_ticket
    if completion_ticket is None:
        return attempt
    try:
        _discard_materialization_recovery(
            provider,
            request,
            completion_ticket,
            execution_context,
        )
        return catalog.clear_materialization_completion_ticket(
            attempt.materialization_id,
            completion_ticket,
        )
    except BaseException as error:
        release_error = WeightTransferReleaseError(
            str(error),
            receipt=None,
            provider=provider.name,
            operation_id=request.operation_id,
            release_error=error,
        )
        release_error.materialized_candidate = attempt
        raise release_error from error


def _execute(
    provider: WeightTransferProvider,
    request: WeightLoadRequest | WeightMaterializeRequest,
    *,
    attestor: WeightTransferAttestor | None = None,
    target_session: WeightTargetLoadSession | None = None,
    completion_ticket_sink: Callable[[str], None] | None = None,
    preflight: WeightTransferPreflight | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
) -> WeightProviderReceipt:
    prepared = None
    submit_returned = False
    local_post_submit_interruption = False

    def require_live_execution(phase: str) -> None:
        nonlocal local_post_submit_interruption
        if execution_context is None:
            return
        cancelled = execution_context.cancelled()
        if not cancelled and execution_context.remaining_seconds() > 0:
            return
        local_post_submit_interruption = submit_returned
        reason = "cancelled" if cancelled else "deadline expired"
        raise WeightTransferError(
            f"weight transfer {reason} before {phase}",
            code="CANCELLED" if cancelled else "DEADLINE_EXCEEDED",
            provider=capabilities.provider,
            phase=phase,
            operation_id=request.operation_id,
            retryable=False,
            completion_known=not submit_returned,
            cleanup_required=prepared is not None,
        )

    reused_preflight = preflight is not None
    if preflight is None:
        validated_preflight = preflight_weight_transfer(
            provider,
            request,
            attestor=attestor,
        )
    else:
        validated_preflight = _validate_preflight(
            preflight,
            provider=provider,
            request=request,
            attestor=attestor,
        )
    capabilities = validated_preflight._capabilities
    provider_phase_seconds = list(validated_preflight._phase_seconds)
    execution_attestor = validated_preflight._attestor
    if reused_preflight and execution_attestor is not None:
        phase_started = time.perf_counter()
        execution_attestor.attest(request)
        provider_phase_seconds.append(("attest", time.perf_counter() - phase_started))
    if execution_context is not None and not capabilities.supports_bounded_execution:
        raise WeightTransferError(
            "provider does not support bounded transfer execution",
            code="UNBOUNDED_PROVIDER",
            provider=capabilities.provider,
            phase="preflight",
            operation_id=request.operation_id,
            retryable=False,
            completion_known=True,
            cleanup_required=False,
        )
    require_live_execution("execution")

    submission = None
    receipt = None
    completion_ticket = None
    completion_ticket_sink_succeeded = False

    def discard_known_materialization_recovery() -> None:
        if (
            not isinstance(request, WeightMaterializeRequest)
            or completion_ticket is None
            or completion_ticket_sink_succeeded
        ):
            return
        _discard_materialization_recovery(
            provider,
            request,
            completion_ticket,
            execution_context,
        )

    try:
        phase_started = time.perf_counter()
        if execution_context is None:
            prepared = provider.prepare(request)
        else:
            prepared = provider.prepare(
                request,
                execution_context=execution_context,
            )
        provider_phase_seconds.append(("prepare", time.perf_counter() - phase_started))
        if (
            isinstance(request, WeightMaterializeRequest)
            and capabilities.supports_completion_ticket
        ):
            recovery_provider = _require_materialization_completion_ticket_provider(
                provider
            )
            completion_ticket = recovery_provider.materialization_recovery_ticket(
                prepared
            )
            if type(completion_ticket) is not str or not completion_ticket:
                raise ValueError(
                    "provider returned an invalid materialization recovery ticket"
                )
            if completion_ticket_sink is not None:
                completion_ticket_sink(completion_ticket)
                completion_ticket_sink_succeeded = True
        require_live_execution("submit")
        phase_started = time.perf_counter()
        if target_session is not None:
            target_session.mark_mutating()
        submission = provider.submit(prepared)
        submit_returned = True
        provider_phase_seconds.append(("submit", time.perf_counter() - phase_started))
        require_live_execution("wait")
        phase_started = time.perf_counter()
        if execution_context is None:
            receipt = provider.wait(submission)
        else:
            receipt = provider.wait(
                submission,
                execution_context=execution_context,
            )
        provider_phase_seconds.append(("wait", time.perf_counter() - phase_started))
        receipt = _validate_provider_receipt(
            request,
            receipt,
            provider=capabilities.provider,
            completion_ticket=completion_ticket,
        )
        phase_started = time.perf_counter()
        if execution_context is None:
            provider.synchronize(receipt)
        else:
            provider.synchronize(
                receipt,
                execution_context=execution_context,
            )
        provider_phase_seconds.append(
            ("synchronize", time.perf_counter() - phase_started)
        )
    except BaseException as error:
        if (
            isinstance(error, WeightTransferError)
            and not error.completion_known
            and not local_post_submit_interruption
        ):
            raise _completion_unknown_error(
                error,
                provider=capabilities.provider,
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            ) from error

        release_safe = not submit_returned
        cancel_error = None
        if submit_returned and capabilities.supports_safe_cancel:
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
                completion_ticket=completion_ticket,
            ) from error

        if prepared is not None:
            try:
                _release_provider(
                    provider,
                    prepared,
                    None,
                    execution_context,
                )
            except BaseException as release_error:
                try:
                    discard_known_materialization_recovery()
                except BaseException as cleanup_error:
                    raise WeightTransferCompletionUnknownError(
                        f"{error}; provider release failed: {release_error}; "
                        f"materialization recovery cleanup failed: {cleanup_error}",
                        provider=capabilities.provider,
                        phase="release",
                        operation_id=request.operation_id,
                        completion_ticket=completion_ticket,
                    ) from cleanup_error
                raise WeightTransferReleaseError(
                    f"{error}; provider release failed: {release_error}",
                    receipt=None,
                    provider=capabilities.provider,
                    operation_id=request.operation_id,
                    release_error=release_error,
                ) from error
        try:
            discard_known_materialization_recovery()
        except BaseException as cleanup_error:
            raise WeightTransferCompletionUnknownError(
                f"{error}; materialization recovery cleanup failed: {cleanup_error}",
                provider=capabilities.provider,
                phase="cleanup",
                operation_id=request.operation_id,
                completion_ticket=completion_ticket,
            ) from cleanup_error
        if local_post_submit_interruption:
            assert isinstance(error, WeightTransferError)
            raise WeightTransferError(
                str(error),
                code=error.code,
                provider=error.provider,
                phase=error.phase,
                operation_id=error.operation_id,
                retryable=error.retryable,
                completion_known=True,
                cleanup_required=error.cleanup_required,
            ) from error
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
        _release_provider(
            provider,
            prepared,
            completed_receipt,
            execution_context,
        )
    except BaseException as error:
        raise WeightTransferReleaseError(
            str(error),
            receipt=completed_receipt,
            release_error=error,
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
    projected_source_bindings = project_source_bindings(
        logical_plan.source_placements,
        source_bindings,
    )
    return prepare_weight_load_from_plan(
        logical_plan,
        source_bindings=projected_source_bindings,
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
    projected_source_bindings = project_source_bindings(
        logical_plan.source_placements,
        source_bindings,
    )
    return prepare_weight_load_from_plan(
        logical_plan,
        source_bindings=projected_source_bindings,
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
    preflight: WeightTransferPreflight | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
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
                preflight=preflight,
                execution_context=execution_context,
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
    execution_context: WeightTransferExecutionContext | None = None,
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
        execution_context=execution_context,
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
    execution_context: WeightTransferExecutionContext | None = None,
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
        execution_context=execution_context,
    )


def prepare_weight_materialization(
    *,
    source_placements: Sequence[WeightPlacementManifest],
    source_bindings: Sequence[SourceBindingManifest],
    destination: WeightStorageDestination,
    payload_identity: WeightPayloadIdentity | None = None,
    operation_id: str | None = None,
    source_placements_are_selected: bool = False,
) -> WeightMaterializeRequest:
    """Validate a Store source without calling a provider.

    Normal callers provide the full runtime world and let this function select
    one complete source replica. Distributed Store workers may instead pass the
    root-selected local placement closure.
    """

    normalized_placements = tuple(
        sorted(source_placements, key=lambda item: item.placement_id)
    )
    normalized_bindings = tuple(source_bindings)
    bind_weight_source(
        normalized_placements,
        normalized_bindings,
    )
    stored_placements = (
        normalized_placements
        if source_placements_are_selected
        else select_weight_storage_placements(normalized_placements)
    )
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
    preflight: WeightTransferPreflight | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
) -> WeightMaterializeReceipt:
    """Execute one previously validated materialization request."""

    receipt = cast(
        WeightMaterializeReceipt,
        _execute(
            provider,
            request,
            attestor=attestor,
            completion_ticket_sink=completion_ticket_sink,
            preflight=preflight,
            execution_context=execution_context,
        ),
    )
    if completion_ticket_sink is None:
        try:
            _discard_materialization_recovery(
                provider,
                request,
                receipt.completion_ticket,
                execution_context,
            )
        except BaseException as error:
            raise WeightTransferReleaseError(
                str(error),
                receipt=receipt,
                release_error=error,
            ) from error
    return receipt


def materialize_weights(
    *,
    source_placements: Sequence[WeightPlacementManifest],
    source_bindings: Sequence[SourceBindingManifest],
    destination: WeightStorageDestination,
    provider: WeightTransferProvider,
    payload_identity: WeightPayloadIdentity | None = None,
    attestor: WeightTransferAttestor | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
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
        execution_context=execution_context,
    )


def materialize_weight_snapshot_candidate(
    request: WeightMaterializeRequest | None = None,
    *,
    provider: WeightTransferProvider,
    catalog: WeightStorageCatalog,
    source_placements: Sequence[WeightPlacementManifest] | None = None,
    source_bindings: Sequence[SourceBindingManifest] | None = None,
    destination: WeightStorageDestination | None = None,
    payload_identity: WeightPayloadIdentity | None = None,
    publication_id: str | None = None,
    attestor: WeightTransferAttestor | None = None,
    preflight: WeightTransferPreflight | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
) -> WeightMaterializationAttempt:
    """Materialize payloads without making the snapshot loadable."""

    if request is None:
        if preflight is not None:
            raise ValueError(
                "a prebuilt request is required with a weight transfer preflight"
            )
        if source_placements is None or source_bindings is None or destination is None:
            raise ValueError(
                "snapshot materialization requires a request or source inputs"
            )
        materialization_id = publication_id or new_operation_id()
        request = prepare_weight_materialization(
            source_placements=source_placements,
            source_bindings=source_bindings,
            destination=destination,
            payload_identity=payload_identity,
            operation_id=materialization_id,
        )
    else:
        if not isinstance(request, WeightMaterializeRequest):
            raise ValueError("snapshot materialization request is invalid")
        if (
            source_placements is not None
            or source_bindings is not None
            or destination is not None
            or payload_identity is not None
        ):
            raise ValueError(
                "a prebuilt request cannot be combined with snapshot source inputs"
            )
        if publication_id is not None and publication_id != request.operation_id:
            raise ValueError("publication ID differs from the materialization request")
        materialization_id = request.operation_id
        destination = request.destination

    if provider.name != destination.provider:
        raise ValueError(
            "weight transfer provider differs from storage destination provider"
        )
    if execution_context is not None:
        with_execution_context = getattr(
            catalog,
            "with_execution_context",
            None,
        )
        if callable(with_execution_context):
            catalog = with_execution_context(execution_context)
    if preflight is not None:
        _validate_preflight(
            preflight,
            provider=provider,
            request=request,
            attestor=attestor,
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
    if previous_attempt is not None:
        recovery_identity_matches = previous_attempt.intent == intent
        if (
            not recovery_identity_matches
            and previous_attempt.completion_ticket is not None
        ):
            recovery_identity_matches = (
                previous_attempt.intent.matches_durable_recovery(intent)
            )
        if not recovery_identity_matches:
            raise ValueError(
                "materialization recovery request differs from its original intent"
            )
    if previous_attempt is not None and previous_attempt.completion_ticket is not None:
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
        if current.state is WeightSnapshotPublicationState.ABORTED:
            raise ValueError("aborted publication cannot be retried")
        return _finalize_materialization_recovery(
            provider,
            catalog,
            request,
            attempt,
            execution_context,
        )
    if attempt.state is WeightMaterializationAttemptState.MATERIALIZED:
        return _finalize_materialization_recovery(
            provider,
            catalog,
            request,
            attempt,
            execution_context,
        )

    try:
        if attempt.completion_ticket is not None:
            recovery_provider = _require_materialization_recovery_provider(provider)
            if execution_context is None:
                receipt = recovery_provider.recover_materialization(
                    request,
                    completion_ticket=attempt.completion_ticket,
                )
            else:
                receipt = recovery_provider.recover_materialization(
                    request,
                    completion_ticket=attempt.completion_ticket,
                    execution_context=execution_context,
                )
            if receipt is None:
                raise WeightTransferCompletionUnknownError(
                    "provider did not resolve the persisted completion ticket",
                    provider=provider.name,
                    phase="recover",
                    operation_id=materialization_id,
                    completion_ticket=attempt.completion_ticket,
                )
            receipt = cast(
                WeightMaterializeReceipt,
                _validate_provider_receipt(
                    request,
                    receipt,
                    provider=provider.name,
                    completion_ticket=attempt.completion_ticket,
                ),
            )
        elif recover_existing_attempt:
            recover = getattr(provider, "recover_materialization", None)
            if not callable(recover):
                receipt = None
            elif execution_context is None:
                receipt = recover(
                    request,
                    completion_ticket=None,
                )
            else:
                receipt = recover(
                    request,
                    completion_ticket=None,
                    execution_context=execution_context,
                )
            if receipt is not None:
                receipt = cast(
                    WeightMaterializeReceipt,
                    _validate_provider_receipt(
                        request,
                        receipt,
                        provider=provider.name,
                        completion_ticket=None,
                    ),
                )
            if receipt is None:
                receipt = execute_weight_materialization(
                    request,
                    provider=provider,
                    attestor=attestor,
                    preflight=preflight,
                    completion_ticket_sink=lambda ticket: (
                        catalog.set_materialization_completion_ticket(
                            materialization_id,
                            ticket,
                        )
                    ),
                    execution_context=execution_context,
                )
        else:
            receipt = execute_weight_materialization(
                request,
                provider=provider,
                attestor=attestor,
                preflight=preflight,
                completion_ticket_sink=lambda ticket: (
                    catalog.set_materialization_completion_ticket(
                        materialization_id,
                        ticket,
                    )
                ),
                execution_context=execution_context,
            )
    except WeightTransferReleaseError as error:
        if not isinstance(error.receipt, WeightMaterializeReceipt):
            raise
        receipt = error.receipt
        if receipt.completion_ticket is not None:
            catalog.set_materialization_completion_ticket(
                materialization_id,
                receipt.completion_ticket,
            )
        raise
    except WeightTransferError as error:
        if not error.completion_known:
            completion_unknown = _completion_unknown_error(
                error,
                provider=provider.name,
                operation_id=materialization_id,
                completion_ticket=attempt.completion_ticket,
            )
            if completion_unknown.completion_ticket is not None:
                catalog.set_materialization_completion_ticket(
                    materialization_id,
                    completion_unknown.completion_ticket,
                )
            raise completion_unknown from error
        current_attempt = catalog.get_materialization(materialization_id)
        completion_ticket = (
            None if current_attempt is None else current_attempt.completion_ticket
        )
        if completion_ticket is not None:
            try:
                _discard_materialization_recovery(
                    provider,
                    request,
                    completion_ticket,
                    execution_context,
                )
                catalog.clear_materialization_completion_ticket(
                    materialization_id,
                    completion_ticket,
                )
            except BaseException as cleanup_error:
                release_error = WeightTransferReleaseError(
                    f"{error}; materialization recovery cleanup failed: "
                    f"{cleanup_error}",
                    receipt=None,
                    provider=provider.name,
                    operation_id=materialization_id,
                    release_error=cleanup_error,
                )
                release_error.materialized_candidate = current_attempt
                raise release_error from error
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
    return _finalize_materialization_recovery(
        provider,
        catalog,
        request,
        completed,
        execution_context,
    )


def publish_weight_snapshot(
    candidate: WeightMaterializationAttempt,
    *,
    catalog: WeightStorageCatalog,
    execution_context: WeightTransferExecutionContext | None = None,
) -> WeightSnapshotPublication:
    """Publish a materialized candidate and advance its revision to READY."""

    if (
        not isinstance(candidate, WeightMaterializationAttempt)
        or candidate.state is not WeightMaterializationAttemptState.MATERIALIZED
        or candidate.snapshot is None
    ):
        raise ValueError("weight snapshot candidate is not materialized")
    if execution_context is not None:
        with_execution_context = getattr(catalog, "with_execution_context", None)
        if callable(with_execution_context):
            catalog = with_execution_context(execution_context)
    current_attempt = catalog.get_materialization(candidate.materialization_id)
    if current_attempt != candidate:
        raise ValueError("weight snapshot candidate differs from catalog state")
    current = catalog.get_publication(candidate.materialization_id)
    if current is not None:
        if current.snapshot != candidate.snapshot:
            raise ValueError("snapshot publication differs from its materialization")
        if current.state is WeightSnapshotPublicationState.ABORTED:
            raise ValueError("aborted publication cannot be retried")
        return _mark_publication_ready(catalog, current)
    pending = catalog.prepare_publish(
        candidate.materialization_id,
        candidate.snapshot,
    )
    return _mark_publication_ready(catalog, pending)


def materialize_weight_snapshot(
    request: WeightMaterializeRequest | None = None,
    *,
    provider: WeightTransferProvider,
    catalog: WeightStorageCatalog,
    source_placements: Sequence[WeightPlacementManifest] | None = None,
    source_bindings: Sequence[SourceBindingManifest] | None = None,
    destination: WeightStorageDestination | None = None,
    payload_identity: WeightPayloadIdentity | None = None,
    publication_id: str | None = None,
    attestor: WeightTransferAttestor | None = None,
    preflight: WeightTransferPreflight | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
) -> WeightSnapshotPublication:
    """Materialize and publish one storage snapshot."""

    candidate = materialize_weight_snapshot_candidate(
        request,
        provider=provider,
        catalog=catalog,
        source_placements=source_placements,
        source_bindings=source_bindings,
        destination=destination,
        payload_identity=payload_identity,
        publication_id=publication_id,
        attestor=attestor,
        preflight=preflight,
        execution_context=execution_context,
    )
    return publish_weight_snapshot(
        candidate,
        catalog=catalog,
        execution_context=execution_context,
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
    execution_context: WeightTransferExecutionContext | None = None,
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
        execution_context=execution_context,
    )
