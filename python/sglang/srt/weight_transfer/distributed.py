from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from sglang.srt.distributed.bounded_object_collectives import (
    BoundedObjectCollectiveCoordinator,
    _RootCallEnvelope,
)
from sglang.srt.distributed.bounded_object_collectives import (
    _PendingCollective as _BoundedPendingCollective,
)
from sglang.srt.distributed.bounded_object_collectives import (
    _SerializedCollectiveValue as _BoundedSerializedCollectiveValue,
)
from sglang.srt.weight_transfer._threaded_call import (
    _BoundedExecutor,
    _ThreadedCall,
)
from sglang.srt.weight_transfer.provider import WeightTransferExecutionContext
from sglang.srt.weight_transfer.storage import (
    InMemoryWeightStorageCatalog,
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

__all__ = [
    "LocalWeightStoreDistributedCoordinator",
    "RootWeightStorageCatalog",
    "TorchDistributedWeightStoreCoordinator",
    "WeightStoreDistributedCoordinator",
    "WeightStoreDistributedError",
    "WeightStorePreflightOutcome",
    "WeightStoreUploadOutcome",
]

_ROOT_RANK = 0
_ROOT_CALL_VERSION = 1
_COLLECTIVE_VALUE_VERSION = 1
_DEFAULT_MAX_OBJECT_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_AGGREGATE_BYTES = 1024 * 1024 * 1024
_DEFAULT_MAX_RESIDENT_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_CHUNK_BYTES = 1024 * 1024
_DEFAULT_MAX_COLLECTIVE_MEMBERS = 65536
_DEFAULT_COLLECTIVE_TIMEOUT_SEC = 300.0
_ROOT_FACTORY_TERMINAL_BROADCAST_GRACE_SEC = 5.0
_ROOT_FACTORY_EXECUTOR = _BoundedExecutor(
    max_workers=4,
    thread_name_prefix="sglang-weight-store-root",
)

_PendingCollective = _BoundedPendingCollective
_SerializedCollectiveValue = _BoundedSerializedCollectiveValue


class WeightStoreDistributedError(RuntimeError):
    def __init__(
        self,
        phase: str,
        message: str,
        *,
        completion_unknown: bool = False,
    ) -> None:
        if type(phase) is not str or not phase:
            raise ValueError("distributed error phase must be a non-empty string")
        if type(message) is not str or not message:
            raise ValueError("distributed error message must be a non-empty string")
        if type(completion_unknown) is not bool:
            raise ValueError("distributed completion_unknown must be a boolean")
        super().__init__(message)
        self.phase = phase
        self.completion_unknown = completion_unknown


@dataclass(frozen=True)
class WeightStorePreflightOutcome:
    rank: int
    error: str | None
    completion_unknown: bool = False

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("preflight outcome rank must be a non-negative integer")
        if self.error is not None and (type(self.error) is not str or not self.error):
            raise ValueError(
                "preflight outcome error must be None or a non-empty string"
            )
        if type(self.completion_unknown) is not bool:
            raise ValueError("preflight completion_unknown must be a boolean")
        if self.completion_unknown and self.error is None:
            raise ValueError("unknown preflight completion requires an error")


@dataclass(frozen=True)
class WeightStoreUploadOutcome:
    rank: int
    placement_ids: tuple[str, ...]
    receipts: tuple[Any, ...]
    error: str | None
    completion_unknown: bool = False

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("upload outcome rank must be a non-negative integer")

        if isinstance(self.placement_ids, (str, bytes, bytearray)):
            raise ValueError("upload outcome placement IDs must be a sequence")
        try:
            placement_ids = tuple(self.placement_ids)
        except TypeError as error:
            raise ValueError(
                "upload outcome placement IDs must be a sequence"
            ) from error
        if any(
            type(placement_id) is not str or not placement_id
            for placement_id in placement_ids
        ):
            raise ValueError("upload outcome placement IDs must be non-empty strings")
        if len(placement_ids) != len(set(placement_ids)):
            raise ValueError("upload outcome placement IDs must not contain duplicates")

        if isinstance(self.receipts, (str, bytes, bytearray)):
            raise ValueError("upload outcome receipts must be a sequence")
        try:
            receipts = tuple(self.receipts)
        except TypeError as error:
            raise ValueError("upload outcome receipts must be a sequence") from error

        if self.error is not None and (type(self.error) is not str or not self.error):
            raise ValueError("upload outcome error must be None or a non-empty string")
        if type(self.completion_unknown) is not bool:
            raise ValueError("upload completion_unknown must be a boolean")
        if self.completion_unknown and self.error is None:
            raise ValueError("unknown upload completion requires an error")

        object.__setattr__(self, "placement_ids", placement_ids)
        object.__setattr__(self, "receipts", receipts)


class WeightStoreDistributedCoordinator(Protocol):
    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    def run_root(
        self,
        phase: str,
        factory: Callable[[], Any],
        *,
        discard_result: bool = False,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any: ...

    def prepare_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any: ...

    def gather_object_to_root(
        self,
        value: Any,
        *,
        phase: str,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> tuple[Any, ...] | None: ...

    def scatter_object_from_root(
        self,
        values: tuple[Any, ...] | list[Any] | None,
        *,
        phase: str,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any: ...

    def exchange_preflight_outcome(
        self,
        outcome: WeightStorePreflightOutcome,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> tuple[WeightStorePreflightOutcome, ...]: ...

    def exchange_upload_outcome(
        self,
        outcome: WeightStoreUploadOutcome,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> tuple[WeightStoreUploadOutcome, ...] | None: ...

    def commit_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any: ...

    def abort_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None: ...

    def finalize_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class _CatalogProjectionRequest:
    rank: int
    materialization_id: str
    placement_ids: tuple[str, ...]
    intent: WeightMaterializationIntent


class RootWeightStorageCatalog:
    """Run every catalog transition on rank zero and broadcast its result."""

    def __init__(
        self,
        catalog: WeightStorageCatalog | None,
        coordinator: WeightStoreDistributedCoordinator,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
        _local_catalog: InMemoryWeightStorageCatalog | None = None,
        _projection_requests: (
            dict[
                str,
                _CatalogProjectionRequest,
            ]
            | None
        ) = None,
    ) -> None:
        if coordinator is None or not callable(getattr(coordinator, "run_root", None)):
            raise ValueError("root catalog coordinator is invalid")
        if coordinator.rank == _ROOT_RANK and catalog is None:
            raise ValueError("root catalog is required on rank zero")
        self._catalog = catalog
        self._coordinator = coordinator
        self._local_catalog = _local_catalog or InMemoryWeightStorageCatalog()
        self._projection_requests = (
            {} if _projection_requests is None else _projection_requests
        )
        self._projection_enabled = bool(self._projection_requests)
        if execution_context is not None and not isinstance(
            execution_context,
            WeightTransferExecutionContext,
        ):
            raise ValueError("weight storage execution context is invalid")
        self._execution_context = execution_context

    def with_execution_context(
        self,
        execution_context: WeightTransferExecutionContext,
    ) -> RootWeightStorageCatalog:
        return RootWeightStorageCatalog(
            self._catalog,
            self._coordinator,
            execution_context=execution_context,
            _local_catalog=self._local_catalog,
            _projection_requests=self._projection_requests,
        )

    def _run(
        self,
        phase: str,
        factory: Callable[[WeightStorageCatalog], Any],
        *,
        discard_result: bool = False,
    ) -> Any:
        def root_factory() -> Any:
            if self._catalog is None:
                raise RuntimeError("root catalog is unavailable")
            return factory(self._catalog)

        if self._execution_context is None:
            return self._coordinator.run_root(
                phase,
                root_factory,
                discard_result=discard_result,
            )
        return self._coordinator.run_root(
            phase,
            root_factory,
            discard_result=discard_result,
            execution_context=self._execution_context,
        )

    def prime_materialization_request(self, request: Any) -> None:
        source_placements = tuple(request.source_placements)
        source_bindings = tuple(request.source_bindings)
        source = source_placements[0]
        payload_identity = request.payload_identity
        intent = WeightMaterializationIntent(
            provider=request.destination.provider,
            storage_id=request.destination.storage_id,
            object_prefix=request.destination.object_prefix,
            model_id=source.model_id,
            revision=source.revision,
            source_digest=weight_placement_set_digest(source_placements),
            total_bytes=request.total_bytes,
            fragment_count=len(request.source_locations),
            source_snapshot_digest=weight_source_snapshot_digest(
                source_placements,
                source_bindings,
            ),
            payload_digest=(
                None if payload_identity is None else payload_identity.payload_digest
            ),
        )
        self._projection_requests[request.operation_id] = _CatalogProjectionRequest(
            rank=self._coordinator.rank,
            materialization_id=request.operation_id,
            placement_ids=tuple(
                placement.placement_id for placement in source_placements
            ),
            intent=intent,
        )
        self._projection_enabled = True

    @staticmethod
    def _project_snapshot(
        snapshot: StoredWeightSnapshot,
        placement_ids: tuple[str, ...],
    ) -> StoredWeightSnapshot:
        placement_id_set = set(placement_ids)
        placements = tuple(
            placement
            for placement in snapshot.placements
            if placement.placement_id in placement_id_set
        )
        bindings = tuple(
            binding
            for binding in snapshot.storage_bindings
            if binding.placement_id in placement_id_set
        )
        if len(placements) != len(placement_id_set) or len(bindings) != len(
            placement_id_set
        ):
            raise ValueError("root catalog snapshot projection is incomplete")
        return StoredWeightSnapshot.create(
            provider=snapshot.ref.provider,
            storage_id=snapshot.ref.storage_id,
            manifest_key=snapshot.ref.manifest_key,
            placements=placements,
            storage_bindings=bindings,
        )

    def _projection_world(
        self,
        phase: str,
        materialization_id: str,
        factory: Callable[[WeightStorageCatalog], Any],
        projector: Callable[[Any, _CatalogProjectionRequest], Any],
    ) -> Any:
        request = self._projection_requests.get(materialization_id)
        if request is None:
            raise ValueError("rank-local catalog projection was not primed")
        gathered = (
            self._coordinator.gather_object_to_root(
                request,
                phase=f"{phase}.gather",
            )
            if self._execution_context is None
            else self._coordinator.gather_object_to_root(
                request,
                phase=f"{phase}.gather",
                execution_context=self._execution_context,
            )
        )
        packets = None

        def build_packets(catalog: WeightStorageCatalog) -> None:
            nonlocal packets
            if (
                not isinstance(gathered, (tuple, list))
                or len(gathered) != self._coordinator.world_size
            ):
                raise ValueError("root catalog projection world is incomplete")
            value = factory(catalog)
            projected = []
            for rank, item in enumerate(gathered):
                if (
                    not isinstance(item, _CatalogProjectionRequest)
                    or item.rank != rank
                    or item.materialization_id != materialization_id
                ):
                    raise ValueError("root catalog projection request is invalid")
                projected.append(projector(value, item))
            packets = tuple(projected)

        self._run(phase, build_packets, discard_result=True)
        return (
            self._coordinator.scatter_object_from_root(
                packets,
                phase=f"{phase}.scatter",
            )
            if self._execution_context is None
            else self._coordinator.scatter_object_from_root(
                packets,
                phase=f"{phase}.scatter",
                execution_context=self._execution_context,
            )
        )

    def begin_materialization(
        self,
        materialization_id: str,
        intent: WeightMaterializationIntent,
    ) -> WeightMaterializationAttempt:
        if not self._projection_enabled or not isinstance(
            intent,
            WeightMaterializationIntent,
        ):
            return self._run(
                "catalog.begin_materialization",
                lambda catalog: catalog.begin_materialization(
                    materialization_id,
                    intent,
                ),
            )
        self._run(
            "catalog.begin_materialization",
            lambda catalog: catalog.begin_materialization(
                materialization_id,
                intent,
            ),
            discard_result=True,
        )
        return self._local_catalog.begin_materialization(
            materialization_id,
            intent,
        )

    def complete_materialization(
        self,
        materialization_id: str,
        snapshot: StoredWeightSnapshot,
    ) -> WeightMaterializationAttempt:
        if not self._projection_enabled or not isinstance(
            snapshot,
            StoredWeightSnapshot,
        ):
            return self._run(
                "catalog.complete_materialization",
                lambda catalog: catalog.complete_materialization(
                    materialization_id,
                    snapshot,
                ),
            )
        self._run(
            "catalog.complete_materialization",
            lambda catalog: catalog.complete_materialization(
                materialization_id,
                snapshot,
            ),
            discard_result=True,
        )
        return self._local_catalog.complete_materialization(
            materialization_id,
            snapshot,
        )

    def abort_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt:
        if not self._projection_enabled:
            return self._run(
                "catalog.abort_materialization",
                lambda catalog: catalog.abort_materialization(materialization_id),
            )
        self._run(
            "catalog.abort_materialization",
            lambda catalog: catalog.abort_materialization(materialization_id),
            discard_result=True,
        )
        return self._local_catalog.abort_materialization(materialization_id)

    def set_materialization_completion_ticket(
        self,
        materialization_id: str,
        completion_ticket: str,
    ) -> WeightMaterializationAttempt:
        if not self._projection_enabled:
            return self._run(
                "catalog.set_materialization_completion_ticket",
                lambda catalog: catalog.set_materialization_completion_ticket(
                    materialization_id,
                    completion_ticket,
                ),
            )
        self._run(
            "catalog.set_materialization_completion_ticket",
            lambda catalog: catalog.set_materialization_completion_ticket(
                materialization_id,
                completion_ticket,
            ),
            discard_result=True,
        )
        return self._local_catalog.set_materialization_completion_ticket(
            materialization_id,
            completion_ticket,
        )

    def clear_materialization_completion_ticket(
        self,
        materialization_id: str,
        completion_ticket: str,
    ) -> WeightMaterializationAttempt:
        if not self._projection_enabled:
            return self._run(
                "catalog.clear_materialization_completion_ticket",
                lambda catalog: catalog.clear_materialization_completion_ticket(
                    materialization_id,
                    completion_ticket,
                ),
            )
        self._run(
            "catalog.clear_materialization_completion_ticket",
            lambda catalog: catalog.clear_materialization_completion_ticket(
                materialization_id,
                completion_ticket,
            ),
            discard_result=True,
        )
        return self._local_catalog.clear_materialization_completion_ticket(
            materialization_id,
            completion_ticket,
        )

    def get_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt | None:
        if not self._projection_enabled:
            return self._run(
                "catalog.get_materialization",
                lambda catalog: catalog.get_materialization(materialization_id),
            )

        def project(
            attempt: WeightMaterializationAttempt | None,
            request: _CatalogProjectionRequest,
        ) -> WeightMaterializationAttempt | None:
            if attempt is None:
                return None
            snapshot = (
                None
                if attempt.snapshot is None
                else self._project_snapshot(
                    attempt.snapshot,
                    request.placement_ids,
                )
            )
            return WeightMaterializationAttempt(
                materialization_id=materialization_id,
                intent=request.intent,
                state=attempt.state,
                snapshot=snapshot,
                completion_ticket=attempt.completion_ticket,
            )

        attempt = self._projection_world(
            "catalog.get_materialization",
            materialization_id,
            lambda catalog: catalog.get_materialization(materialization_id),
            project,
        )
        if attempt is None:
            return None
        local = self._local_catalog.get_materialization(materialization_id)
        if local is None:
            local = self._local_catalog.begin_materialization(
                materialization_id,
                attempt.intent,
            )
        if attempt.completion_ticket is not None and local.completion_ticket is None:
            local = self._local_catalog.set_materialization_completion_ticket(
                materialization_id,
                attempt.completion_ticket,
            )
        if attempt.state is WeightMaterializationAttemptState.MATERIALIZED:
            assert attempt.snapshot is not None
            local = self._local_catalog.complete_materialization(
                materialization_id,
                attempt.snapshot,
            )
        elif attempt.state is WeightMaterializationAttemptState.ABORTED:
            local = self._local_catalog.abort_materialization(materialization_id)
        return local

    def recoverable_materializations(
        self,
    ) -> tuple[WeightMaterializationAttempt, ...]:
        if not self._projection_enabled:
            return self._run(
                "catalog.recoverable_materializations",
                lambda catalog: catalog.recoverable_materializations(),
            )
        self._run(
            "catalog.recoverable_materializations",
            lambda catalog: catalog.recoverable_materializations(),
            discard_result=True,
        )
        return self._local_catalog.recoverable_materializations()

    def prepare_publish(
        self,
        publication_id: str,
        snapshot: StoredWeightSnapshot,
    ) -> WeightSnapshotPublication:
        if not self._projection_enabled or not isinstance(
            snapshot,
            StoredWeightSnapshot,
        ):
            return self._run(
                "catalog.prepare_publish",
                lambda catalog: catalog.prepare_publish(
                    publication_id,
                    snapshot,
                ),
            )
        self._run(
            "catalog.prepare_publish",
            lambda catalog: catalog.prepare_publish(
                publication_id,
                snapshot,
            ),
            discard_result=True,
        )
        return self._local_catalog.prepare_publish(
            publication_id,
            snapshot,
        )

    def publish(self, publication_id: str) -> WeightSnapshotPublication:
        if not self._projection_enabled:
            return self._run(
                "catalog.publish",
                lambda catalog: catalog.publish(publication_id),
            )
        self._run(
            "catalog.publish",
            lambda catalog: catalog.publish(publication_id),
            discard_result=True,
        )
        return self._local_catalog.publish(publication_id)

    def abort(self, publication_id: str) -> WeightSnapshotPublication:
        if not self._projection_enabled:
            return self._run(
                "catalog.abort",
                lambda catalog: catalog.abort(publication_id),
            )
        self._run(
            "catalog.abort",
            lambda catalog: catalog.abort(publication_id),
            discard_result=True,
        )
        return self._local_catalog.abort(publication_id)

    def get_snapshot(
        self,
        ref: WeightStorageRef,
    ) -> StoredWeightSnapshot | None:
        return self._run(
            "catalog.get_snapshot",
            lambda catalog: catalog.get_snapshot(ref),
        )

    def get_publication(
        self,
        publication_id: str,
    ) -> WeightSnapshotPublication | None:
        if not self._projection_enabled:
            return self._run(
                "catalog.get_publication",
                lambda catalog: catalog.get_publication(publication_id),
            )

        def project(
            publication: WeightSnapshotPublication | None,
            request: _CatalogProjectionRequest,
        ) -> WeightSnapshotPublication | None:
            if publication is None:
                return None
            return WeightSnapshotPublication(
                publication_id=publication.publication_id,
                snapshot=self._project_snapshot(
                    publication.snapshot,
                    request.placement_ids,
                ),
                state=publication.state,
            )

        publication = self._projection_world(
            "catalog.get_publication",
            publication_id,
            lambda catalog: catalog.get_publication(publication_id),
            project,
        )
        if publication is None:
            return None
        local = self._local_catalog.get_publication(publication_id)
        if local is None:
            local = self._local_catalog.prepare_publish(
                publication_id,
                publication.snapshot,
            )
        if publication.state is WeightSnapshotPublicationState.PUBLISHED:
            local = self._local_catalog.publish(publication_id)
        elif publication.state is WeightSnapshotPublicationState.ABORTED:
            local = self._local_catalog.abort(publication_id)
        return local

    def recoverable_publications(
        self,
    ) -> tuple[WeightSnapshotPublication, ...]:
        if not self._projection_enabled:
            return self._run(
                "catalog.recoverable_publications",
                lambda catalog: catalog.recoverable_publications(),
            )
        self._run(
            "catalog.recoverable_publications",
            lambda catalog: catalog.recoverable_publications(),
            discard_result=True,
        )
        return self._local_catalog.recoverable_publications()

    def get_revision_head(
        self,
        model_id: str,
        revision: str,
    ) -> WeightRevisionHead | None:
        return self._run(
            "catalog.get_revision_head",
            lambda catalog: catalog.get_revision_head(model_id, revision),
        )

    def compare_and_set_revision(
        self,
        *,
        model_id: str,
        revision: str,
        expected: WeightRevisionHead | None,
        new_ref: WeightStorageRef,
        new_state: WeightRevisionState,
    ) -> WeightRevisionHead | None:
        return self._run(
            "catalog.compare_and_set_revision",
            lambda catalog: catalog.compare_and_set_revision(
                model_id=model_id,
                revision=revision,
                expected=expected,
                new_ref=new_ref,
                new_state=new_state,
            ),
        )


def _format_exception(error: BaseException) -> str:
    error_type = type(error).__name__
    try:
        message = str(error)
    except BaseException:
        message = ""
    if not message:
        return error_type
    return f"{error_type}: {message}"


def _check_execution_context(
    phase: str,
    execution_context: WeightTransferExecutionContext | None,
) -> None:
    if execution_context is not None and execution_context.expired():
        raise WeightStoreDistributedError(
            phase,
            f"{phase} exceeded the weight transfer deadline",
            completion_unknown=False,
        )


def _call_local_factory(
    phase: str,
    factory: Callable[[], Any],
    *,
    discard_result: bool,
) -> Any:
    try:
        result = factory()
    except BaseException as error:
        raise WeightStoreDistributedError(
            phase,
            _format_exception(error),
            completion_unknown=bool(getattr(error, "completion_unknown", False)),
        ) from error
    if discard_result:
        return None
    return result


def _validate_root_call(
    phase: object,
    factory: object,
    *,
    discard_result: object,
) -> None:
    if type(phase) is not str or not phase:
        raise ValueError("root call phase must be a non-empty string")
    if not callable(factory):
        raise ValueError("root call factory must be callable")
    if type(discard_result) is not bool:
        raise ValueError("discard_result must be a boolean")


def _validate_outcomes(
    values: list[Any] | tuple[Any, ...],
    *,
    world_size: int,
) -> tuple[WeightStoreUploadOutcome, ...]:
    phase = "exchange_upload_outcome"
    if len(values) != world_size:
        raise WeightStoreDistributedError(
            phase,
            "invalid gathered outcomes: world size does not match",
        )

    outcomes = []
    for index, value in enumerate(values):
        if type(value) is not WeightStoreUploadOutcome:
            raise WeightStoreDistributedError(
                phase,
                f"invalid gathered outcome at index {index}",
            )
        try:
            outcome = WeightStoreUploadOutcome(
                rank=value.rank,
                placement_ids=value.placement_ids,
                receipts=value.receipts,
                error=value.error,
                completion_unknown=value.completion_unknown,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise WeightStoreDistributedError(
                phase,
                f"invalid gathered outcome at index {index}: {error}",
            ) from error
        outcomes.append(outcome)

    ordered = tuple(sorted(outcomes, key=lambda outcome: outcome.rank))
    ranks = tuple(outcome.rank for outcome in ordered)
    if ranks != tuple(range(world_size)):
        raise WeightStoreDistributedError(
            phase,
            "invalid gathered outcomes: ranks must be unique and complete",
        )

    placement_owners: dict[str, int] = {}
    for outcome in ordered:
        for placement_id in outcome.placement_ids:
            owner = placement_owners.setdefault(placement_id, outcome.rank)
            if owner != outcome.rank:
                raise WeightStoreDistributedError(
                    phase,
                    f"invalid gathered outcomes: duplicate placement ID {placement_id}",
                )
    return ordered


def _validate_preflight_outcomes(
    values: list[Any] | tuple[Any, ...],
    *,
    world_size: int,
) -> tuple[WeightStorePreflightOutcome, ...]:
    phase = "exchange_preflight_outcome"
    if len(values) != world_size:
        raise WeightStoreDistributedError(
            phase,
            "invalid gathered preflight outcomes: world size does not match",
        )

    outcomes = []
    for index, value in enumerate(values):
        if type(value) is not WeightStorePreflightOutcome:
            raise WeightStoreDistributedError(
                phase,
                f"invalid gathered preflight outcome at index {index}",
            )
        try:
            outcome = WeightStorePreflightOutcome(
                rank=value.rank,
                error=value.error,
                completion_unknown=value.completion_unknown,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise WeightStoreDistributedError(
                phase,
                f"invalid gathered preflight outcome at index {index}: {error}",
            ) from error
        outcomes.append(outcome)

    ordered = tuple(sorted(outcomes, key=lambda outcome: outcome.rank))
    if tuple(outcome.rank for outcome in ordered) != tuple(range(world_size)):
        raise WeightStoreDistributedError(
            phase,
            "invalid gathered preflight outcomes: ranks must be unique and complete",
        )
    return ordered


class LocalWeightStoreDistributedCoordinator:
    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def run_root(
        self,
        phase: str,
        factory: Callable[[], Any],
        *,
        discard_result: bool = False,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any:
        _validate_root_call(
            phase,
            factory,
            discard_result=discard_result,
        )
        _check_execution_context(phase, execution_context)
        result = _call_local_factory(
            phase,
            factory,
            discard_result=discard_result,
        )
        _check_execution_context(phase, execution_context)
        return result

    def prepare_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any:
        return self.run_root(
            "prepare_upload",
            factory,
            discard_result=False,
            execution_context=execution_context,
        )

    def gather_object_to_root(
        self,
        value: Any,
        *,
        phase: str,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> tuple[Any, ...]:
        _check_execution_context(phase, execution_context)
        return (value,)

    def scatter_object_from_root(
        self,
        values: tuple[Any, ...] | list[Any] | None,
        *,
        phase: str,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any:
        _check_execution_context(phase, execution_context)
        if (
            not isinstance(values, (tuple, list))
            or isinstance(values, (str, bytes, bytearray))
            or len(values) != 1
        ):
            raise WeightStoreDistributedError(
                phase,
                "root scatter values must match the process-group world size",
            )
        return values[0]

    def exchange_preflight_outcome(
        self,
        outcome: WeightStorePreflightOutcome,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> tuple[WeightStorePreflightOutcome, ...]:
        del execution_context
        return _validate_preflight_outcomes((outcome,), world_size=1)

    def exchange_upload_outcome(
        self,
        outcome: WeightStoreUploadOutcome,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> tuple[WeightStoreUploadOutcome, ...]:
        del execution_context
        return _validate_outcomes((outcome,), world_size=1)

    def commit_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any:
        return self.run_root(
            "commit_upload",
            factory,
            discard_result=False,
            execution_context=execution_context,
        )

    def abort_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        self.run_root(
            "abort_upload",
            factory,
            discard_result=True,
            execution_context=execution_context,
        )

    def finalize_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        self.run_root(
            "finalize_upload",
            factory,
            discard_result=True,
            execution_context=execution_context,
        )


class TorchDistributedWeightStoreCoordinator(BoundedObjectCollectiveCoordinator):
    def __init__(
        self,
        group: Any = None,
        *,
        max_object_bytes: int = _DEFAULT_MAX_OBJECT_BYTES,
        max_aggregate_bytes: int = _DEFAULT_MAX_AGGREGATE_BYTES,
        max_resident_bytes: int = _DEFAULT_MAX_RESIDENT_BYTES,
        chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
        max_collective_members: int = _DEFAULT_MAX_COLLECTIVE_MEMBERS,
    ) -> None:
        super().__init__(
            group=group,
            max_object_bytes=max_object_bytes,
            max_aggregate_bytes=max_aggregate_bytes,
            max_resident_bytes=max_resident_bytes,
            chunk_bytes=chunk_bytes,
            max_collective_members=max_collective_members,
            error_type=WeightStoreDistributedError,
            context_factory=lambda: WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + _DEFAULT_COLLECTIVE_TIMEOUT_SEC,
            ),
            unresolved_label="weight Store",
        )
        self._root_factory_calls: set[_ThreadedCall] = set()
        self._root_factory_calls_lock = threading.Lock()

    def _retire_root_factory_call(self, call: _ThreadedCall) -> None:
        with self._root_factory_calls_lock:
            self._root_factory_calls.discard(call)

    def _root_factory_interrupted(
        self,
        phase: str,
        execution_context: WeightTransferExecutionContext,
    ) -> WeightStoreDistributedError:
        reason = (
            "was cancelled before completion"
            if execution_context.cancelled()
            else "did not finish before the deadline"
        )
        return WeightStoreDistributedError(
            phase,
            f"root factory {reason}",
            completion_unknown=True,
        )

    def _decode_root_factory_call(self, envelope: Any, *, phase: str) -> Any:
        try:
            return self._decode_root_call(envelope, phase=phase)
        except WeightStoreDistributedError as error:
            if error.completion_unknown:
                raise self._poison(phase, str(error)) from error
            raise

    def _run_root_factory(
        self,
        phase: str,
        factory: Callable[[], Any],
        *,
        discard_result: bool,
        execution_context: WeightTransferExecutionContext,
    ) -> Any:
        terminal_context = WeightTransferExecutionContext(
            deadline_unix_sec=(
                execution_context.deadline_unix_sec
                + _ROOT_FACTORY_TERMINAL_BROADCAST_GRACE_SEC
            ),
        )
        envelope = None
        if self.rank == _ROOT_RANK:
            try:
                call = _ThreadedCall(_ROOT_FACTORY_EXECUTOR)
                with self._root_factory_calls_lock:
                    self._root_factory_calls.add(call)
                call.start(
                    factory,
                    thread_name=f"weight-store-root-{phase}",
                    after_done=lambda: self._retire_root_factory_call(call),
                )
                result = call.result_before(
                    execution_context,
                    interrupted=lambda: self._root_factory_interrupted(
                        phase,
                        execution_context,
                    ),
                )
            except BaseException as error:
                envelope = _RootCallEnvelope(
                    version=_ROOT_CALL_VERSION,
                    phase=phase,
                    succeeded=False,
                    result=None,
                    error=_format_exception(error),
                    completion_unknown=bool(
                        getattr(error, "completion_unknown", False)
                    ),
                )
            else:
                envelope = _RootCallEnvelope(
                    version=_ROOT_CALL_VERSION,
                    phase=phase,
                    succeeded=True,
                    result=None if discard_result else result,
                    error=None,
                    completion_unknown=False,
                )

        if self.world_size == 1:
            return self._decode_root_factory_call(envelope, phase=phase)
        envelope = self._broadcast_object(
            envelope,
            phase=phase,
            execution_context=terminal_context,
        )
        return self._decode_root_factory_call(envelope, phase=phase)

    def run_root(
        self,
        phase: str,
        factory: Callable[[], Any],
        *,
        discard_result: bool = False,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any:
        _validate_root_call(
            phase,
            factory,
            discard_result=discard_result,
        )
        self._require_healthy(phase)
        if self.world_size == 1:
            _check_execution_context(phase, execution_context)
            context = execution_context or WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + _DEFAULT_COLLECTIVE_TIMEOUT_SEC,
            )
        else:
            context = self._collective_context(phase, execution_context)
        return self._run_root_factory(
            phase,
            factory,
            discard_result=discard_result,
            execution_context=context,
        )

    def prepare_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any:
        return self.run_root(
            "prepare_upload",
            factory,
            discard_result=False,
            execution_context=execution_context,
        )

    def exchange_preflight_outcome(
        self,
        outcome: WeightStorePreflightOutcome,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> tuple[WeightStorePreflightOutcome, ...]:
        phase = "exchange_preflight_outcome"
        return self._gather_validated_to_root(
            outcome,
            phase=phase,
            execution_context=execution_context,
            validator=lambda gathered: _validate_preflight_outcomes(
                gathered,
                world_size=self.world_size,
            ),
            error_prefix="torch.distributed preflight exchange failed",
        )

    def exchange_upload_outcome(
        self,
        outcome: WeightStoreUploadOutcome,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> tuple[WeightStoreUploadOutcome, ...] | None:
        phase = "exchange_upload_outcome"
        return self._gather_validated_to_root(
            outcome,
            phase=phase,
            execution_context=execution_context,
            validator=lambda gathered: _validate_outcomes(
                gathered,
                world_size=self.world_size,
            ),
            error_prefix="torch.distributed outcome exchange failed",
            root_result_only=True,
        )

    def commit_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> Any:
        return self.run_root(
            "commit_upload",
            factory,
            discard_result=False,
            execution_context=execution_context,
        )

    def abort_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        self.run_root(
            "abort_upload",
            factory,
            discard_result=True,
            execution_context=execution_context,
        )

    def finalize_upload(
        self,
        factory: Callable[[], Any],
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        self.run_root(
            "finalize_upload",
            factory,
            discard_result=True,
            execution_context=execution_context,
        )
