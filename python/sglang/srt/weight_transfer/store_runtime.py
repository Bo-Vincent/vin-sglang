from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Iterator, Mapping, Protocol, Sequence

from sglang.srt.weight_transfer._threaded_call import (
    _BoundedExecutor,
    _ThreadedCall,
    _ThreadedCallAdmissionError,
)
from sglang.srt.weight_transfer.contracts import RuntimeWeightLocation
from sglang.srt.weight_transfer.distributed import (
    RootWeightStorageCatalog,
    WeightStoreDistributedCoordinator,
    WeightStoreDistributedError,
    WeightStorePreflightOutcome,
)
from sglang.srt.weight_transfer.mooncake_store import (
    MooncakeWeightStoreProvider,
    WeightStoreNativeCallState,
)
from sglang.srt.weight_transfer.provider import (
    WeightStorageDestination,
    WeightTransferExecutionContext,
    WeightTransferProvider,
)
from sglang.srt.weight_transfer.storage import (
    WeightStorageCatalog,
    WeightStorageRef,
)
from sglang.srt.weight_transfer.storage_file import FileWeightStorageCatalog

_LOAD_SPEC_KEYS = frozenset(
    {
        "model_id",
        "revision",
        "catalog_path",
        "ref",
        "endpoint",
        "instance_id",
        "load_timeout_sec",
        "mooncake_store",
    }
)
_REF_KEYS = frozenset(
    {
        "provider",
        "storage_id",
        "manifest_key",
        "manifest_digest",
    }
)
_MOONCAKE_STORE_KEYS = frozenset(
    {
        "setup",
        "key_prefix",
        "namespace",
        "max_range_bytes",
        "max_ranges_per_request",
        "max_region_segments",
        "max_total_operations",
        "target_pre_registered",
    }
)
_WRITE_SPEC_KEYS = frozenset(
    {
        "catalog_path",
        "destination",
        "endpoint",
        "mooncake_store",
        "provider_options",
    }
)
_DESTINATION_KEYS = frozenset(
    {
        "provider",
        "storage_id",
        "object_prefix",
    }
)
_MOONCAKE_STORE_WRITE_KEYS = frozenset(
    {
        "setup",
        "key_prefix",
        "namespace",
        "max_range_bytes",
        "max_ranges_per_request",
        "max_region_segments",
        "max_total_operations",
        "source_pre_registered",
    }
)

_STORE_LIFECYCLE_EXECUTOR = _BoundedExecutor(
    max_workers=2,
    thread_name_prefix="sglang-weight-store-lifecycle",
)
_STORE_TERMINAL_CONTROL_TIMEOUT_SEC = 5.0
_PENDING_LIFECYCLE_CALLS: set[_StoreLifecycleCall] = set()
_PENDING_LIFECYCLE_CALLS_LOCK = threading.Lock()


class _StoreLifecycleInterrupted(RuntimeError):
    def __init__(self, phase: str, *, started: bool) -> None:
        super().__init__(f"Mooncake Store {phase} did not finish before the deadline")
        self.completion_unknown = started


@dataclass(eq=False)
class _StoreLifecycleCall:
    phase: str
    owner: Any
    factory: Callable[[], Any]
    cleanup_on_abandon: Callable[[], Any] | None = None
    _call: _ThreadedCall = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._call = _ThreadedCall(_STORE_LIFECYCLE_EXECUTOR)

    @property
    def done(self) -> threading.Event:
        return self._call.done

    def start(self) -> None:
        with _PENDING_LIFECYCLE_CALLS_LOCK:
            _PENDING_LIFECYCLE_CALLS.add(self)

        def forget_pending_call() -> None:
            with _PENDING_LIFECYCLE_CALLS_LOCK:
                _PENDING_LIFECYCLE_CALLS.discard(self)

        self._call.start(
            self.factory,
            thread_name=f"sglang-weight-store-{self.phase}",
            after_done=forget_pending_call,
            cleanup_on_abandon=self.cleanup_on_abandon,
        )

    def result_before(
        self,
        execution_context: WeightTransferExecutionContext,
    ) -> Any:
        return self._call.result_before(
            execution_context,
            interrupted=lambda: _StoreLifecycleInterrupted(
                self.phase,
                started=True,
            ),
        )


def _run_store_lifecycle_call(
    phase: str,
    owner: Any,
    factory: Callable[[], Any],
    execution_context: WeightTransferExecutionContext | None,
    *,
    cleanup_on_abandon: Callable[[], Any] | None = None,
    start_when_expired: bool = False,
) -> Any:
    if execution_context is None:
        return factory()
    if execution_context.expired() and not start_when_expired:
        raise _StoreLifecycleInterrupted(phase, started=False)
    call = _StoreLifecycleCall(
        phase=phase,
        owner=owner,
        factory=factory,
        cleanup_on_abandon=cleanup_on_abandon,
    )
    call.start()
    return call.result_before(execution_context)


def _owner_has_pending_lifecycle_call(owner: Any) -> bool:
    with _PENDING_LIFECYCLE_CALLS_LOCK:
        return any(call.owner is owner for call in _PENDING_LIFECYCLE_CALLS)


def _store_terminal_execution_context(
    execution_context: WeightTransferExecutionContext | None,
) -> WeightTransferExecutionContext | None:
    if execution_context is None:
        return None
    return WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + _STORE_TERMINAL_CONTROL_TIMEOUT_SEC,
    )


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str or not key for key in value):
        raise ValueError(f"{name} keys must be non-empty strings")
    return value


def _require_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _non_negative_timeout(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], name: str):
    unknown = set(mapping).difference(allowed)
    if unknown:
        raise ValueError(f"unknown {name}: {sorted(unknown)}")


@dataclass(frozen=True)
class WeightSnapshotLoadSpec:
    model_id: str
    revision: str
    catalog_path: str
    ref: WeightStorageRef
    endpoint: str | None = None
    instance_id: str | None = None
    mooncake_store: Mapping[str, Any] | None = None
    load_timeout_sec: int = 1800

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> WeightSnapshotLoadSpec:
        mapping = _require_mapping(value, "weight snapshot load options")
        _reject_unknown(
            mapping,
            _LOAD_SPEC_KEYS,
            "weight snapshot load options",
        )
        ref_mapping = _require_mapping(mapping.get("ref"), "weight snapshot ref")
        _reject_unknown(ref_mapping, _REF_KEYS, "weight snapshot ref options")
        ref = WeightStorageRef(
            provider=_require_string(ref_mapping.get("provider"), "ref.provider"),
            storage_id=_require_string(
                ref_mapping.get("storage_id"),
                "ref.storage_id",
            ),
            manifest_key=_require_string(
                ref_mapping.get("manifest_key"),
                "ref.manifest_key",
            ),
            manifest_digest=_require_string(
                ref_mapping.get("manifest_digest"),
                "ref.manifest_digest",
            ),
        )
        endpoint = mapping.get("endpoint")
        if endpoint is not None:
            endpoint = _require_string(endpoint, "endpoint")
        instance_id = mapping.get("instance_id")
        if instance_id is not None:
            instance_id = _require_string(instance_id, "instance_id")
        mooncake_store = mapping.get("mooncake_store")
        if mooncake_store is not None:
            mooncake_store = dict(
                _require_mapping(mooncake_store, "mooncake_store options")
            )
            _reject_unknown(
                mooncake_store,
                _MOONCAKE_STORE_KEYS,
                "Mooncake Store options",
            )
            setup = _require_mapping(
                mooncake_store.get("setup"),
                "Mooncake Store setup",
            )
            mooncake_store["setup"] = dict(setup)
        return cls(
            model_id=_require_string(mapping.get("model_id"), "model_id"),
            revision=_require_string(mapping.get("revision"), "revision"),
            catalog_path=_require_string(
                mapping.get("catalog_path"),
                "catalog_path",
            ),
            ref=ref,
            endpoint=endpoint,
            instance_id=instance_id,
            load_timeout_sec=_positive_integer(
                mapping,
                "load_timeout_sec",
                1800,
            ),
            mooncake_store=mooncake_store,
        )


@dataclass(frozen=True)
class WeightSnapshotWriteSpec:
    catalog_path: str
    destination: WeightStorageDestination
    endpoint: str | None = None
    mooncake_store: Mapping[str, Any] | None = None
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        legacy_options = (
            None if self.mooncake_store is None else dict(self.mooncake_store)
        )
        provider_options = dict(self.provider_options)
        if (
            legacy_options is not None
            and provider_options
            and legacy_options != provider_options
        ):
            raise ValueError(
                "mooncake_store and provider_options are mutually exclusive"
            )
        if legacy_options is not None:
            if self.destination.provider != MooncakeWeightStoreProvider.name:
                raise ValueError(
                    "non-Mooncake writes do not accept mooncake_store options"
                )
            provider_options = dict(legacy_options)
        object.__setattr__(self, "provider_options", provider_options)
        if self.destination.provider == MooncakeWeightStoreProvider.name:
            object.__setattr__(self, "mooncake_store", provider_options)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> WeightSnapshotWriteSpec:
        mapping = _require_mapping(value, "weight snapshot write options")
        _reject_unknown(
            mapping,
            _WRITE_SPEC_KEYS,
            "weight snapshot write options",
        )

        destination_mapping = _require_mapping(
            mapping.get("destination"),
            "weight snapshot destination",
        )
        _reject_unknown(
            destination_mapping,
            _DESTINATION_KEYS,
            "weight snapshot destination options",
        )
        provider = _require_string(
            destination_mapping.get("provider"),
            "destination.provider",
        )
        destination = WeightStorageDestination(
            provider=provider,
            storage_id=_require_string(
                destination_mapping.get("storage_id"),
                "destination.storage_id",
            ),
            object_prefix=_require_string(
                destination_mapping.get("object_prefix"),
                "destination.object_prefix",
            ),
        )

        endpoint = mapping.get("endpoint")
        if endpoint is not None:
            endpoint = _require_string(endpoint, "endpoint")

        legacy_mooncake_options = mapping.get("mooncake_store")
        provider_options = dict(
            _require_mapping(
                mapping.get("provider_options", {}),
                "provider_options",
            )
        )
        if legacy_mooncake_options is not None and provider_options:
            raise ValueError(
                "mooncake_store and provider_options are mutually exclusive"
            )
        if provider == MooncakeWeightStoreProvider.name:
            provider_options = dict(
                _require_mapping(
                    (
                        legacy_mooncake_options
                        if legacy_mooncake_options is not None
                        else provider_options
                    ),
                    "Mooncake Store provider options",
                )
            )
            _reject_unknown(
                provider_options,
                _MOONCAKE_STORE_WRITE_KEYS,
                "Mooncake Store write options",
            )
            provider_options["setup"] = dict(
                _require_mapping(
                    provider_options.get("setup"),
                    "Mooncake Store setup",
                )
            )
            for name in ("key_prefix", "namespace"):
                if name in provider_options:
                    _require_string(provider_options[name], name)
            for name in (
                "max_range_bytes",
                "max_ranges_per_request",
                "max_region_segments",
                "max_total_operations",
            ):
                if name in provider_options:
                    _positive_integer(provider_options, name, 1)
            source_pre_registered = provider_options.get(
                "source_pre_registered",
                False,
            )
            if type(source_pre_registered) is not bool:
                raise ValueError("source_pre_registered must be a boolean")
        elif legacy_mooncake_options is not None:
            raise ValueError("non-Mooncake writes do not accept mooncake_store options")

        return cls(
            catalog_path=_require_string(
                mapping.get("catalog_path"),
                "catalog_path",
            ),
            destination=destination,
            endpoint=endpoint,
            provider_options=provider_options,
        )


@dataclass(frozen=True)
class WeightSnapshotBackendStatus:
    terminal: bool
    pending_tickets: tuple[str, ...] = ()
    closed: bool = False

    def __post_init__(self) -> None:
        if type(self.terminal) is not bool or type(self.closed) is not bool:
            raise ValueError("backend status flags must be booleans")
        tickets = tuple(self.pending_tickets)
        if any(type(ticket) is not str or not ticket for ticket in tickets):
            raise ValueError("pending backend tickets must be non-empty strings")
        if self.terminal and tickets:
            raise ValueError("a terminal backend cannot have pending tickets")
        if self.closed and not self.terminal:
            raise ValueError("a closed backend must be terminal")
        object.__setattr__(self, "pending_tickets", tickets)


class WeightSnapshotBackendLifecycle(Protocol):
    def seal(self) -> tuple[str, ...]: ...

    def quiesce(self, *, timeout_ms: int) -> WeightSnapshotBackendStatus: ...

    def close(
        self,
        *,
        timeout_ms: int | None,
    ) -> WeightSnapshotBackendStatus: ...


class _NoopWeightSnapshotBackendLifecycle:
    def seal(self) -> tuple[str, ...]:
        return ()

    def quiesce(self, *, timeout_ms: int) -> WeightSnapshotBackendStatus:
        _non_negative_timeout(timeout_ms, "backend quiesce timeout_ms")
        return WeightSnapshotBackendStatus(terminal=True)

    def close(
        self,
        *,
        timeout_ms: int | None,
    ) -> WeightSnapshotBackendStatus:
        if timeout_ms is not None:
            _non_negative_timeout(timeout_ms, "backend close timeout_ms")
        return WeightSnapshotBackendStatus(terminal=True, closed=True)


@dataclass(frozen=True)
class WeightSnapshotBackend:
    provider: WeightTransferProvider
    catalog: WeightStorageCatalog
    endpoint: str
    lifecycle: WeightSnapshotBackendLifecycle = field(
        default_factory=_NoopWeightSnapshotBackendLifecycle,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.provider is None or self.catalog is None:
            raise ValueError("weight snapshot backend is incomplete")
        _require_string(self.endpoint, "weight snapshot backend endpoint")
        for method_name in ("seal", "quiesce", "close"):
            if not callable(getattr(self.lifecycle, method_name, None)):
                raise ValueError("weight snapshot backend lifecycle is invalid")

    def seal(self) -> tuple[str, ...]:
        """Stop new backend admission while preserving drainable calls."""

        return tuple(self.lifecycle.seal())

    def quiesce(self, *, timeout_ms: int) -> WeightSnapshotBackendStatus:
        """Drain admitted calls within one total monotonic timeout budget."""

        _non_negative_timeout(timeout_ms, "backend quiesce timeout_ms")
        return self.lifecycle.quiesce(timeout_ms=timeout_ms)

    def close(
        self,
        *,
        timeout_ms: int | None = None,
    ) -> WeightSnapshotBackendStatus:
        """Seal, quiesce, and close without claiming native cancellation."""

        if timeout_ms is not None:
            _non_negative_timeout(timeout_ms, "backend close timeout_ms")
        return self.lifecycle.close(timeout_ms=timeout_ms)


class _MooncakeStoreBackendLifecycle:
    _CLOSE_TICKET = "mooncake-store/close"

    def __init__(self, store: Any, provider: MooncakeWeightStoreProvider) -> None:
        self._store = store
        self._provider = provider
        self._lock = threading.Lock()
        self._close_call: _ThreadedCall | None = None
        self._closed = False

    @staticmethod
    def _pending_tickets(statuses: Sequence[Any]) -> tuple[str, ...]:
        return tuple(
            f"{status.operation_id}/{status.phase}"
            for status in statuses
            if status.state is WeightStoreNativeCallState.PENDING
        )

    def seal(self) -> tuple[str, ...]:
        statuses = self._provider.seal_native_calls_for_close()
        return self._pending_tickets(statuses)

    def quiesce(self, *, timeout_ms: int) -> WeightSnapshotBackendStatus:
        _non_negative_timeout(timeout_ms, "backend quiesce timeout_ms")
        statuses = self._provider.drain_pending_calls(timeout_ms=timeout_ms)
        pending = self._pending_tickets(statuses)
        return WeightSnapshotBackendStatus(
            terminal=not pending,
            pending_tickets=pending,
        )

    def close(
        self,
        *,
        timeout_ms: int | None,
    ) -> WeightSnapshotBackendStatus:
        if timeout_ms is not None:
            _non_negative_timeout(timeout_ms, "backend close timeout_ms")
        deadline = None if timeout_ms is None else time.monotonic() + timeout_ms / 1000
        self.seal()
        quiesced = self.quiesce(
            timeout_ms=0 if deadline is None else self._remaining_ms(deadline)
        )
        if not quiesced.terminal:
            return quiesced
        self.seal()

        with self._lock:
            if self._closed:
                return WeightSnapshotBackendStatus(terminal=True, closed=True)
            call = self._close_call
            if call is None:
                call = _ThreadedCall(_STORE_LIFECYCLE_EXECUTOR)
                self._close_call = call
                start = True
            else:
                start = False

        if start:
            if deadline is None:
                call.start_inline(
                    self._store.close,
                    thread_name="sglang-weight-store-close",
                )
            else:
                call.start(
                    self._store.close,
                    thread_name="sglang-weight-store-close",
                )

        try:
            if deadline is None:
                call.done.wait()
            else:
                execution_context = WeightTransferExecutionContext(
                    deadline_unix_sec=time.time()
                    + max(0.0, deadline - time.monotonic())
                )
                call.result_before(
                    execution_context,
                    interrupted=lambda: _StoreLifecycleInterrupted(
                        "close",
                        started=True,
                    ),
                )
        except _StoreLifecycleInterrupted:
            return WeightSnapshotBackendStatus(
                terminal=False,
                pending_tickets=(self._CLOSE_TICKET,),
            )
        except _ThreadedCallAdmissionError:
            with self._lock:
                if self._close_call is call:
                    self._close_call = None
            raise

        error = call.error
        if error is not None:
            raise error
        with self._lock:
            self._closed = True
        return WeightSnapshotBackendStatus(terminal=True, closed=True)

    @staticmethod
    def _remaining_ms(deadline: float) -> int:
        return int(max(0.0, deadline - time.monotonic()) * 1000)


WeightSnapshotBackendFactory = Callable[
    [WeightSnapshotLoadSpec],
    Any,
]


class WeightSnapshotWriteBackendFactory(Protocol):
    def __call__(
        self,
        spec: WeightSnapshotWriteSpec,
        /,
        *,
        local_placement_ids: Sequence[str],
        payload_checksum_verifier: Callable[[RuntimeWeightLocation], str],
        coordinator: WeightStoreDistributedCoordinator,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> ContextManager[WeightSnapshotBackend]: ...


_WEIGHT_SNAPSHOT_WRITE_BACKENDS: dict[str, WeightSnapshotWriteBackendFactory] = {}


def register_weight_snapshot_write_backend(
    provider: str,
    factory: WeightSnapshotWriteBackendFactory,
    *,
    replace: bool = False,
) -> None:
    _require_string(provider, "weight snapshot writer provider")
    if not callable(factory):
        raise TypeError("weight snapshot writer factory must be callable")
    if provider in _WEIGHT_SNAPSHOT_WRITE_BACKENDS and not replace:
        raise ValueError(f"weight snapshot writer is already registered: {provider}")
    _WEIGHT_SNAPSHOT_WRITE_BACKENDS[provider] = factory


def _positive_integer(
    options: Mapping[str, Any],
    name: str,
    default: int,
) -> int:
    value = options.get(name, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _format_backend_error(error: BaseException) -> str:
    error_type = type(error).__name__
    try:
        message = str(error)
    except BaseException:
        message = ""
    return error_type if not message else f"{error_type}: {message}"


def _rank_qualified_store_setup(
    setup: Mapping[str, Any],
    *,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("Store world_size must be a positive integer")
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError("Store rank must be an integer inside world_size")

    qualified = dict(setup)
    if world_size > 1:
        hostname = _require_string(
            qualified.get("local_hostname"),
            "Mooncake Store local_hostname",
        )
        qualified["local_hostname"] = f"{hostname}-rank-{rank}"
    return qualified


def _close_store(
    store: Any | None,
    *,
    backend: WeightSnapshotBackend | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
) -> BaseException | None:
    if store is None:
        return None
    if backend is not None:
        timeout_ms = (
            None
            if execution_context is None
            else int(execution_context.remaining_seconds() * 1000)
        )
        try:
            status = backend.close(timeout_ms=timeout_ms)
        except BaseException as error:
            return error
        if status.closed:
            return None
        pending = ", ".join(status.pending_tickets) or "unknown"
        error = RuntimeError(
            (
                "Mooncake Store backend has pending calls: "
                f"{pending}; quiesce the backend before close"
            )
            if not status.terminal
            else "Mooncake Store backend reached terminal state without closing"
        )
        error.completion_unknown = not status.terminal
        return error
    close = getattr(store, "close", None)
    if not callable(close):
        return None
    if _owner_has_pending_lifecycle_call(store):
        error = RuntimeError("Mooncake Store lifecycle call is still pending")
        error.completion_unknown = True
        return error
    try:
        _run_store_lifecycle_call(
            "close",
            store,
            close,
            execution_context,
            start_when_expired=True,
        )
    except BaseException as error:
        return error
    return None


def _outcome_error_message(
    outcomes: Sequence[WeightStorePreflightOutcome],
) -> str | None:
    failures = tuple(outcome for outcome in outcomes if outcome.error is not None)
    if not failures:
        return None
    return "; ".join(f"rank {outcome.rank}: {outcome.error}" for outcome in failures)


def _outcome_completion_unknown(
    outcomes: Sequence[WeightStorePreflightOutcome],
) -> bool:
    return any(outcome.completion_unknown for outcome in outcomes)


def _combined_lifecycle_error(
    phase: str,
    primary_error: BaseException,
    close_error: BaseException,
) -> RuntimeError:
    error = RuntimeError(
        f"{_format_backend_error(primary_error)}; "
        f"Mooncake Store {phase} cleanup failed: {_format_backend_error(close_error)}"
    )
    error.completion_unknown = bool(
        getattr(primary_error, "completion_unknown", False)
        or getattr(close_error, "completion_unknown", False)
    )
    return error


@contextmanager
def open_weight_snapshot_backend(
    spec: WeightSnapshotLoadSpec,
    *,
    rank: int = 0,
    world_size: int = 1,
    execution_context: WeightTransferExecutionContext | None = None,
) -> Iterator[WeightSnapshotBackend]:
    """Open the default Mooncake Store backend for one startup load."""

    if not isinstance(spec, WeightSnapshotLoadSpec):
        raise ValueError("weight snapshot load spec is invalid")
    if spec.ref.provider != MooncakeWeightStoreProvider.name:
        raise ValueError(
            "no default weight snapshot backend for provider "
            f"{spec.ref.provider!r}; inject weight_snapshot_backend_factory"
        )
    if spec.mooncake_store is None:
        raise ValueError("Mooncake Store load requires mooncake_store options")

    options = spec.mooncake_store
    setup = _rank_qualified_store_setup(
        _require_mapping(options.get("setup"), "Mooncake Store setup"),
        rank=rank,
        world_size=world_size,
    )
    endpoint = spec.endpoint or _require_string(
        setup.get("local_hostname"),
        "Mooncake Store local_hostname",
    )
    try:
        from mooncake.store import MooncakeDistributedStore
        from mooncake.weight_transfer import WeightStore
    except Exception as error:
        raise RuntimeError(
            "Mooncake Store weight snapshot loading is unavailable"
        ) from error

    if execution_context is None:
        execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=time.time() + spec.load_timeout_sec
        )
    store = MooncakeDistributedStore()
    backend: WeightSnapshotBackend | None = None
    primary_error: BaseException | None = None
    try:
        result = _run_store_lifecycle_call(
            "setup",
            store,
            lambda: store.setup(setup),
            execution_context,
            cleanup_on_abandon=getattr(store, "close", None),
        )
        if result != 0:
            raise RuntimeError(f"MooncakeDistributedStore setup failed: {result}")
        weight_store = WeightStore(
            store,
            key_prefix=str(options.get("key_prefix", "weights")),
            max_range_bytes=_positive_integer(
                options,
                "max_range_bytes",
                64 * 1024 * 1024,
            ),
            max_ranges_per_request=_positive_integer(
                options,
                "max_ranges_per_request",
                1024,
            ),
            max_region_segments=_positive_integer(
                options,
                "max_region_segments",
                1_000_000,
            ),
        )
        target_pre_registered = options.get("target_pre_registered", False)
        if type(target_pre_registered) is not bool:
            raise ValueError("target_pre_registered must be a boolean")
        provider = MooncakeWeightStoreProvider(
            weight_store,
            namespace=str(options.get("namespace", "default")),
            target_pre_registered=target_pre_registered,
            prepare_upload_is_local=True,
            max_total_operations=_positive_integer(
                options,
                "max_total_operations",
                10_000_000,
            ),
        )
        backend = WeightSnapshotBackend(
            provider=provider,
            catalog=FileWeightStorageCatalog(spec.catalog_path),
            endpoint=endpoint,
            lifecycle=_MooncakeStoreBackendLifecycle(store, provider),
        )
        yield backend
    except BaseException as error:
        primary_error = error
        raise
    finally:
        close_error = _close_store(
            store,
            backend=backend,
            execution_context=_store_terminal_execution_context(execution_context),
        )
        if close_error is not None:
            if primary_error is None:
                raise close_error
            raise _combined_lifecycle_error(
                "reader",
                primary_error,
                close_error,
            ) from primary_error


@contextmanager
def _open_mooncake_weight_snapshot_write_backend(
    spec: WeightSnapshotWriteSpec,
    *,
    local_placement_ids: Sequence[str],
    payload_checksum_verifier: Callable[[RuntimeWeightLocation], str],
    coordinator: WeightStoreDistributedCoordinator,
    execution_context: WeightTransferExecutionContext | None = None,
) -> Iterator[WeightSnapshotBackend]:
    """Open a writer whose Store remains live until the caller exits the context."""

    if coordinator is None:
        raise ValueError("weight snapshot write coordinator is required")

    store: Any | None = None
    provider: MooncakeWeightStoreProvider | None = None
    backend: WeightSnapshotBackend | None = None
    endpoint: str | None = None
    local_setup_error: BaseException | None = None
    try:
        if not isinstance(spec, WeightSnapshotWriteSpec):
            raise ValueError("weight snapshot write spec is invalid")
        if spec.destination.provider != MooncakeWeightStoreProvider.name:
            raise ValueError("weight snapshot destination must use mooncake-store")
        if not callable(payload_checksum_verifier):
            raise ValueError("payload checksum verifier must be callable")

        options = spec.provider_options
        if not options:
            raise ValueError("Mooncake Store writes require provider_options")
        setup = _rank_qualified_store_setup(
            _require_mapping(options.get("setup"), "Mooncake Store setup"),
            rank=coordinator.rank,
            world_size=coordinator.world_size,
        )
        endpoint = spec.endpoint or _require_string(
            setup.get("local_hostname"),
            "Mooncake Store local_hostname",
        )
        try:
            from mooncake.store import MooncakeDistributedStore
            from mooncake.weight_transfer import WeightStore
        except Exception as error:
            raise RuntimeError(
                "Mooncake Store weight snapshot writing is unavailable"
            ) from error

        store = MooncakeDistributedStore()
        result = _run_store_lifecycle_call(
            "setup",
            store,
            lambda: store.setup(setup),
            execution_context,
            cleanup_on_abandon=getattr(store, "close", None),
        )
        if result != 0:
            raise RuntimeError(f"MooncakeDistributedStore setup failed: {result}")

        weight_store = WeightStore(
            store,
            key_prefix=str(options.get("key_prefix", "weights")),
            max_range_bytes=_positive_integer(
                options,
                "max_range_bytes",
                64 * 1024 * 1024,
            ),
            max_ranges_per_request=_positive_integer(
                options,
                "max_ranges_per_request",
                1024,
            ),
            max_region_segments=_positive_integer(
                options,
                "max_region_segments",
                1_000_000,
            ),
        )
        source_pre_registered = options.get("source_pre_registered", False)
        if type(source_pre_registered) is not bool:
            raise ValueError("source_pre_registered must be a boolean")

        provider = MooncakeWeightStoreProvider(
            weight_store,
            namespace=str(options.get("namespace", "default")),
            local_placement_ids=local_placement_ids,
            coordinator=coordinator,
            payload_checksum_verifier=payload_checksum_verifier,
            source_pre_registered=source_pre_registered,
            prepare_upload_is_local=True,
            max_total_operations=_positive_integer(
                options,
                "max_total_operations",
                10_000_000,
            ),
        )
    except BaseException as error:
        local_setup_error = error

    try:
        setup_outcome = WeightStorePreflightOutcome(
            rank=coordinator.rank,
            error=(
                None
                if local_setup_error is None
                else _format_backend_error(local_setup_error)
            ),
            completion_unknown=bool(
                getattr(local_setup_error, "completion_unknown", False)
            ),
        )
        setup_outcomes = (
            coordinator.exchange_preflight_outcome(setup_outcome)
            if execution_context is None
            else coordinator.exchange_preflight_outcome(
                setup_outcome,
                execution_context=execution_context,
            )
        )
    except BaseException as exchange_error:
        cleanup_context = _store_terminal_execution_context(execution_context)
        cleanup_error = _close_store(
            store,
            execution_context=cleanup_context,
        )
        failure = _format_backend_error(exchange_error)
        if cleanup_error is not None:
            failure = f"{failure}; cleanup: {_format_backend_error(cleanup_error)}"
        raise WeightStoreDistributedError(
            "initialize_write_backend",
            failure,
            completion_unknown=bool(
                getattr(exchange_error, "completion_unknown", False)
                or getattr(cleanup_error, "completion_unknown", False)
            ),
        ) from exchange_error

    setup_failure = _outcome_error_message(setup_outcomes)
    if setup_failure is not None:
        cleanup_context = _store_terminal_execution_context(execution_context)
        cleanup_error = _close_store(
            store,
            execution_context=cleanup_context,
        )
        cleanup_outcome = WeightStorePreflightOutcome(
            rank=coordinator.rank,
            error=(
                None if cleanup_error is None else _format_backend_error(cleanup_error)
            ),
            completion_unknown=bool(
                getattr(cleanup_error, "completion_unknown", False)
            ),
        )
        try:
            cleanup_outcomes = (
                coordinator.exchange_preflight_outcome(cleanup_outcome)
                if cleanup_context is None
                else coordinator.exchange_preflight_outcome(
                    cleanup_outcome,
                    execution_context=cleanup_context,
                )
            )
        except BaseException as cleanup_exchange_error:
            cleanup_failure = _format_backend_error(cleanup_exchange_error)
            cleanup_completion_unknown = bool(
                getattr(cleanup_exchange_error, "completion_unknown", False)
                or getattr(cleanup_error, "completion_unknown", False)
            )
        else:
            cleanup_failure = _outcome_error_message(cleanup_outcomes)
            cleanup_completion_unknown = _outcome_completion_unknown(cleanup_outcomes)
        failure = setup_failure
        if cleanup_failure is not None:
            failure = f"{failure}; cleanup: {cleanup_failure}"
        raise WeightStoreDistributedError(
            "initialize_write_backend",
            failure,
            completion_unknown=(
                _outcome_completion_unknown(setup_outcomes)
                or cleanup_completion_unknown
            ),
        ) from local_setup_error
    if store is None or provider is None or endpoint is None:
        _close_store(
            store,
            execution_context=_store_terminal_execution_context(execution_context),
        )
        raise WeightStoreDistributedError(
            "initialize_write_backend",
            "distributed setup succeeded without a complete local backend",
        )

    primary_error: BaseException | None = None
    try:
        root_catalog: WeightStorageCatalog | None = None

        def initialize_catalog() -> None:
            nonlocal root_catalog
            root_catalog = FileWeightStorageCatalog(spec.catalog_path)

        if execution_context is None:
            coordinator.run_root(
                "catalog.initialize",
                initialize_catalog,
                discard_result=True,
            )
        else:
            coordinator.run_root(
                "catalog.initialize",
                initialize_catalog,
                discard_result=True,
                execution_context=execution_context,
            )
        catalog = RootWeightStorageCatalog(
            root_catalog,
            coordinator,
            execution_context=execution_context,
        )
        backend = WeightSnapshotBackend(
            provider=provider,
            catalog=catalog,
            endpoint=endpoint,
            lifecycle=_MooncakeStoreBackendLifecycle(store, provider),
        )
        yield backend
    except BaseException as error:
        primary_error = error
        raise
    finally:
        current_execution_context = execution_context
        if provider is not None:
            provider_execution_context = provider.current_execution_context()
            if provider_execution_context is not None:
                current_execution_context = provider_execution_context
        terminal_context = _store_terminal_execution_context(
            current_execution_context,
        )
        close_error = _close_store(
            store,
            backend=backend,
            execution_context=terminal_context,
        )
        try:
            close_outcome = WeightStorePreflightOutcome(
                rank=coordinator.rank,
                error=(
                    None if close_error is None else _format_backend_error(close_error)
                ),
                completion_unknown=bool(
                    getattr(close_error, "completion_unknown", False)
                ),
            )
            close_outcomes = (
                coordinator.exchange_preflight_outcome(close_outcome)
                if terminal_context is None
                else coordinator.exchange_preflight_outcome(
                    close_outcome,
                    execution_context=terminal_context,
                )
            )
        except BaseException:
            if primary_error is None:
                raise
        else:
            close_failure = _outcome_error_message(close_outcomes)
            if primary_error is None and close_failure is not None:
                raise WeightStoreDistributedError(
                    "close_write_backend",
                    close_failure,
                    completion_unknown=_outcome_completion_unknown(close_outcomes),
                )


_WEIGHT_SNAPSHOT_WRITE_BACKENDS[MooncakeWeightStoreProvider.name] = (
    _open_mooncake_weight_snapshot_write_backend
)


@contextmanager
def open_weight_snapshot_write_backend(
    spec: WeightSnapshotWriteSpec,
    *,
    local_placement_ids: Sequence[str],
    payload_checksum_verifier: Callable[[RuntimeWeightLocation], str],
    coordinator: WeightStoreDistributedCoordinator,
    execution_context: WeightTransferExecutionContext | None = None,
) -> Iterator[WeightSnapshotBackend]:
    if not isinstance(spec, WeightSnapshotWriteSpec):
        raise ValueError("weight snapshot write spec is invalid")
    provider_name = spec.destination.provider
    factory = _WEIGHT_SNAPSHOT_WRITE_BACKENDS.get(provider_name)
    if factory is None:
        raise ValueError(f"no weight snapshot writer registered for {provider_name!r}")

    factory_args = {
        "local_placement_ids": local_placement_ids,
        "payload_checksum_verifier": payload_checksum_verifier,
        "coordinator": coordinator,
    }
    if execution_context is not None:
        factory_args["execution_context"] = execution_context
    context = factory(spec, **factory_args)
    with context as backend:
        if not isinstance(backend, WeightSnapshotBackend):
            raise TypeError("weight snapshot writer returned an invalid backend")
        if backend.provider.name != provider_name:
            raise ValueError(
                "weight snapshot writer provider differs from the destination"
            )
        yield backend
