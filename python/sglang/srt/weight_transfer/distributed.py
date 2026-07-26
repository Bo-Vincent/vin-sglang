from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from sglang.srt.weight_transfer.storage import (
    StoredWeightSnapshot,
    WeightMaterializationAttempt,
    WeightMaterializationIntent,
    WeightRevisionHead,
    WeightRevisionState,
    WeightSnapshotPublication,
    WeightStorageCatalog,
    WeightStorageRef,
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


class WeightStoreDistributedError(RuntimeError):
    def __init__(self, phase: str, message: str) -> None:
        if type(phase) is not str or not phase:
            raise ValueError("distributed error phase must be a non-empty string")
        if type(message) is not str or not message:
            raise ValueError("distributed error message must be a non-empty string")
        super().__init__(message)
        self.phase = phase


@dataclass(frozen=True)
class WeightStorePreflightOutcome:
    rank: int
    error: str | None

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 0:
            raise ValueError("preflight outcome rank must be a non-negative integer")
        if self.error is not None and (type(self.error) is not str or not self.error):
            raise ValueError(
                "preflight outcome error must be None or a non-empty string"
            )


@dataclass(frozen=True)
class WeightStoreUploadOutcome:
    rank: int
    placement_ids: tuple[str, ...]
    receipts: tuple[Any, ...]
    error: str | None

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
    ) -> Any: ...

    def prepare_upload(self, factory: Callable[[], Any]) -> Any: ...

    def exchange_preflight_outcome(
        self,
        outcome: WeightStorePreflightOutcome,
    ) -> tuple[WeightStorePreflightOutcome, ...]: ...

    def exchange_upload_outcome(
        self,
        outcome: WeightStoreUploadOutcome,
    ) -> tuple[WeightStoreUploadOutcome, ...]: ...

    def commit_upload(self, factory: Callable[[], Any]) -> Any: ...

    def abort_upload(self, factory: Callable[[], Any]) -> None: ...

    def finalize_upload(self, factory: Callable[[], Any]) -> None: ...


class RootWeightStorageCatalog:
    """Run every catalog transition on rank zero and broadcast its result."""

    def __init__(
        self,
        catalog: WeightStorageCatalog | None,
        coordinator: WeightStoreDistributedCoordinator,
    ) -> None:
        if coordinator is None or not callable(getattr(coordinator, "run_root", None)):
            raise ValueError("root catalog coordinator is invalid")
        if coordinator.rank == _ROOT_RANK and catalog is None:
            raise ValueError("root catalog is required on rank zero")
        self._catalog = catalog
        self._coordinator = coordinator

    def _run(self, phase: str, factory: Callable[[WeightStorageCatalog], Any]) -> Any:
        def root_factory() -> Any:
            if self._catalog is None:
                raise RuntimeError("root catalog is unavailable")
            return factory(self._catalog)

        return self._coordinator.run_root(phase, root_factory)

    def begin_materialization(
        self,
        materialization_id: str,
        intent: WeightMaterializationIntent,
    ) -> WeightMaterializationAttempt:
        return self._run(
            "catalog.begin_materialization",
            lambda catalog: catalog.begin_materialization(
                materialization_id,
                intent,
            ),
        )

    def complete_materialization(
        self,
        materialization_id: str,
        snapshot: StoredWeightSnapshot,
    ) -> WeightMaterializationAttempt:
        return self._run(
            "catalog.complete_materialization",
            lambda catalog: catalog.complete_materialization(
                materialization_id,
                snapshot,
            ),
        )

    def abort_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt:
        return self._run(
            "catalog.abort_materialization",
            lambda catalog: catalog.abort_materialization(materialization_id),
        )

    def set_materialization_completion_ticket(
        self,
        materialization_id: str,
        completion_ticket: str,
    ) -> WeightMaterializationAttempt:
        return self._run(
            "catalog.set_materialization_completion_ticket",
            lambda catalog: catalog.set_materialization_completion_ticket(
                materialization_id,
                completion_ticket,
            ),
        )

    def get_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt | None:
        return self._run(
            "catalog.get_materialization",
            lambda catalog: catalog.get_materialization(materialization_id),
        )

    def recoverable_materializations(
        self,
    ) -> tuple[WeightMaterializationAttempt, ...]:
        return self._run(
            "catalog.recoverable_materializations",
            lambda catalog: catalog.recoverable_materializations(),
        )

    def prepare_publish(
        self,
        publication_id: str,
        snapshot: StoredWeightSnapshot,
    ) -> WeightSnapshotPublication:
        return self._run(
            "catalog.prepare_publish",
            lambda catalog: catalog.prepare_publish(
                publication_id,
                snapshot,
            ),
        )

    def publish(self, publication_id: str) -> WeightSnapshotPublication:
        return self._run(
            "catalog.publish",
            lambda catalog: catalog.publish(publication_id),
        )

    def abort(self, publication_id: str) -> WeightSnapshotPublication:
        return self._run(
            "catalog.abort",
            lambda catalog: catalog.abort(publication_id),
        )

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
        return self._run(
            "catalog.get_publication",
            lambda catalog: catalog.get_publication(publication_id),
        )

    def recoverable_publications(
        self,
    ) -> tuple[WeightSnapshotPublication, ...]:
        return self._run(
            "catalog.recoverable_publications",
            lambda catalog: catalog.recoverable_publications(),
        )

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


@dataclass(frozen=True)
class _RootCallEnvelope:
    version: int
    phase: str
    succeeded: bool
    result: Any
    error: str | None


def _format_exception(error: BaseException) -> str:
    error_type = type(error).__name__
    try:
        message = str(error)
    except BaseException:
        message = ""
    if not message:
        return error_type
    return f"{error_type}: {message}"


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


def _decode_root_call(
    value: Any,
    *,
    phase: str,
) -> Any:
    if type(value) is not _RootCallEnvelope:
        raise WeightStoreDistributedError(
            phase,
            "invalid root broadcast response",
        )
    if (
        value.version != _ROOT_CALL_VERSION
        or value.phase != phase
        or type(value.succeeded) is not bool
    ):
        raise WeightStoreDistributedError(
            phase,
            "invalid root broadcast response",
        )
    if value.succeeded:
        if value.error is not None:
            raise WeightStoreDistributedError(
                phase,
                "invalid root broadcast success response",
            )
        return value.result
    if value.result is not None or type(value.error) is not str or not value.error:
        raise WeightStoreDistributedError(
            phase,
            "invalid root broadcast error response",
        )
    raise WeightStoreDistributedError(phase, value.error)


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
    ) -> Any:
        _validate_root_call(
            phase,
            factory,
            discard_result=discard_result,
        )
        return _call_local_factory(
            phase,
            factory,
            discard_result=discard_result,
        )

    def prepare_upload(self, factory: Callable[[], Any]) -> Any:
        return self.run_root(
            "prepare_upload",
            factory,
            discard_result=False,
        )

    def exchange_preflight_outcome(
        self,
        outcome: WeightStorePreflightOutcome,
    ) -> tuple[WeightStorePreflightOutcome, ...]:
        return _validate_preflight_outcomes((outcome,), world_size=1)

    def exchange_upload_outcome(
        self,
        outcome: WeightStoreUploadOutcome,
    ) -> tuple[WeightStoreUploadOutcome, ...]:
        return _validate_outcomes((outcome,), world_size=1)

    def commit_upload(self, factory: Callable[[], Any]) -> Any:
        return self.run_root(
            "commit_upload",
            factory,
            discard_result=False,
        )

    def abort_upload(self, factory: Callable[[], Any]) -> None:
        self.run_root(
            "abort_upload",
            factory,
            discard_result=True,
        )

    def finalize_upload(self, factory: Callable[[], Any]) -> None:
        self.run_root(
            "finalize_upload",
            factory,
            discard_result=True,
        )


class TorchDistributedWeightStoreCoordinator:
    def __init__(self, group: Any = None) -> None:
        phase = "initialize"
        try:
            distributed = importlib.import_module("torch.distributed")
        except BaseException as error:
            raise WeightStoreDistributedError(
                phase,
                f"torch.distributed is unavailable: {_format_exception(error)}",
            ) from error

        required = (
            "all_gather_object",
            "broadcast_object_list",
            "get_rank",
            "get_world_size",
            "is_initialized",
        )
        missing = [
            name for name in required if not callable(getattr(distributed, name, None))
        ]
        if missing:
            raise WeightStoreDistributedError(
                phase,
                "torch.distributed is missing required APIs: " + ", ".join(missing),
            )

        try:
            initialized = distributed.is_initialized()
        except BaseException as error:
            raise WeightStoreDistributedError(
                phase,
                "torch.distributed initialization check failed: "
                f"{_format_exception(error)}",
            ) from error
        if type(initialized) is not bool or not initialized:
            raise WeightStoreDistributedError(
                phase,
                "torch.distributed must be initialized",
            )

        try:
            rank = distributed.get_rank(group=group)
            world_size = distributed.get_world_size(group=group)
        except BaseException as error:
            raise WeightStoreDistributedError(
                phase,
                f"torch.distributed rank discovery failed: {_format_exception(error)}",
            ) from error
        if type(world_size) is not int or world_size <= 0:
            raise WeightStoreDistributedError(
                phase,
                "torch.distributed world size must be a positive integer",
            )
        if type(rank) is not int or not 0 <= rank < world_size:
            raise WeightStoreDistributedError(
                phase,
                "torch.distributed rank is outside the process group",
            )

        self._distributed = distributed
        self._group = group
        self._rank = rank
        self._world_size = world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def _run_root_factory(
        self,
        phase: str,
        factory: Callable[[], Any],
        *,
        discard_result: bool,
    ) -> Any:
        envelope = None
        if self.rank == _ROOT_RANK:
            try:
                result = factory()
            except BaseException as error:
                envelope = _RootCallEnvelope(
                    version=_ROOT_CALL_VERSION,
                    phase=phase,
                    succeeded=False,
                    result=None,
                    error=_format_exception(error),
                )
            else:
                envelope = _RootCallEnvelope(
                    version=_ROOT_CALL_VERSION,
                    phase=phase,
                    succeeded=True,
                    result=None if discard_result else result,
                    error=None,
                )

        object_list = [envelope]
        source = (
            {"src": _ROOT_RANK} if self._group is None else {"group_src": _ROOT_RANK}
        )
        try:
            self._distributed.broadcast_object_list(
                object_list,
                group=self._group,
                **source,
            )
        except BaseException as error:
            raise WeightStoreDistributedError(
                phase,
                f"torch.distributed root broadcast failed: {_format_exception(error)}",
            ) from error
        return _decode_root_call(object_list[0], phase=phase)

    def run_root(
        self,
        phase: str,
        factory: Callable[[], Any],
        *,
        discard_result: bool = False,
    ) -> Any:
        _validate_root_call(
            phase,
            factory,
            discard_result=discard_result,
        )
        return self._run_root_factory(
            phase,
            factory,
            discard_result=discard_result,
        )

    def prepare_upload(self, factory: Callable[[], Any]) -> Any:
        return self.run_root(
            "prepare_upload",
            factory,
            discard_result=False,
        )

    def exchange_preflight_outcome(
        self,
        outcome: WeightStorePreflightOutcome,
    ) -> tuple[WeightStorePreflightOutcome, ...]:
        gathered: list[Any] = [None] * self.world_size
        try:
            self._distributed.all_gather_object(
                gathered,
                outcome,
                group=self._group,
            )
        except BaseException as error:
            raise WeightStoreDistributedError(
                "exchange_preflight_outcome",
                "torch.distributed preflight exchange failed: "
                f"{_format_exception(error)}",
            ) from error
        return _validate_preflight_outcomes(gathered, world_size=self.world_size)

    def exchange_upload_outcome(
        self,
        outcome: WeightStoreUploadOutcome,
    ) -> tuple[WeightStoreUploadOutcome, ...]:
        gathered: list[Any] = [None] * self.world_size
        try:
            self._distributed.all_gather_object(
                gathered,
                outcome,
                group=self._group,
            )
        except BaseException as error:
            raise WeightStoreDistributedError(
                "exchange_upload_outcome",
                "torch.distributed outcome exchange failed: "
                f"{_format_exception(error)}",
            ) from error
        return _validate_outcomes(gathered, world_size=self.world_size)

    def commit_upload(self, factory: Callable[[], Any]) -> Any:
        return self.run_root(
            "commit_upload",
            factory,
            discard_result=False,
        )

    def abort_upload(self, factory: Callable[[], Any]) -> None:
        self.run_root(
            "abort_upload",
            factory,
            discard_result=True,
        )

    def finalize_upload(self, factory: Callable[[], Any]) -> None:
        self.run_root(
            "finalize_upload",
            factory,
            discard_result=True,
        )
