from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from sglang.srt.weight_transfer.contracts import RuntimeWeightLocation
from sglang.srt.weight_transfer.distributed import (
    RootWeightStorageCatalog,
    WeightStoreDistributedCoordinator,
    WeightStoreDistributedError,
    WeightStorePreflightOutcome,
)
from sglang.srt.weight_transfer.mooncake_store import (
    MooncakeWeightStoreProvider,
)
from sglang.srt.weight_transfer.provider import (
    WeightStorageDestination,
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
            mooncake_store=mooncake_store,
        )


@dataclass(frozen=True)
class WeightSnapshotWriteSpec:
    catalog_path: str
    destination: WeightStorageDestination
    endpoint: str | None
    mooncake_store: Mapping[str, Any]

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
        if provider != MooncakeWeightStoreProvider.name:
            raise ValueError("destination.provider must be 'mooncake-store'")
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

        mooncake_store = dict(
            _require_mapping(
                mapping.get("mooncake_store"),
                "mooncake_store options",
            )
        )
        _reject_unknown(
            mooncake_store,
            _MOONCAKE_STORE_WRITE_KEYS,
            "Mooncake Store write options",
        )
        mooncake_store["setup"] = dict(
            _require_mapping(
                mooncake_store.get("setup"),
                "Mooncake Store setup",
            )
        )
        for name in ("key_prefix", "namespace"):
            if name in mooncake_store:
                _require_string(mooncake_store[name], name)
        for name in (
            "max_range_bytes",
            "max_ranges_per_request",
            "max_region_segments",
            "max_total_operations",
        ):
            if name in mooncake_store:
                _positive_integer(mooncake_store, name, 1)
        source_pre_registered = mooncake_store.get(
            "source_pre_registered",
            False,
        )
        if type(source_pre_registered) is not bool:
            raise ValueError("source_pre_registered must be a boolean")

        return cls(
            catalog_path=_require_string(
                mapping.get("catalog_path"),
                "catalog_path",
            ),
            destination=destination,
            endpoint=endpoint,
            mooncake_store=mooncake_store,
        )


@dataclass(frozen=True)
class WeightSnapshotBackend:
    provider: WeightTransferProvider
    catalog: WeightStorageCatalog
    endpoint: str

    def __post_init__(self) -> None:
        if self.provider is None or self.catalog is None:
            raise ValueError("weight snapshot backend is incomplete")
        _require_string(self.endpoint, "weight snapshot backend endpoint")


WeightSnapshotBackendFactory = Callable[
    [WeightSnapshotLoadSpec],
    Any,
]


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


def _close_store(store: Any | None) -> BaseException | None:
    if store is None:
        return None
    close = getattr(store, "close", None)
    if not callable(close):
        return None
    try:
        close()
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


@contextmanager
def open_weight_snapshot_backend(
    spec: WeightSnapshotLoadSpec,
    *,
    rank: int = 0,
    world_size: int = 1,
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

    store = MooncakeDistributedStore()
    result = store.setup(setup)
    if result != 0:
        raise RuntimeError(f"MooncakeDistributedStore setup failed: {result}")
    try:
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
        backend = WeightSnapshotBackend(
            provider=MooncakeWeightStoreProvider(
                weight_store,
                namespace=str(options.get("namespace", "default")),
                target_pre_registered=target_pre_registered,
                max_total_operations=_positive_integer(
                    options,
                    "max_total_operations",
                    10_000_000,
                ),
            ),
            catalog=FileWeightStorageCatalog(spec.catalog_path),
            endpoint=endpoint,
        )
        yield backend
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


@contextmanager
def open_weight_snapshot_write_backend(
    spec: WeightSnapshotWriteSpec,
    *,
    local_placement_ids: Sequence[str],
    payload_checksum_verifier: Callable[[RuntimeWeightLocation], str],
    coordinator: WeightStoreDistributedCoordinator,
) -> Iterator[WeightSnapshotBackend]:
    """Open a writer whose Store remains live until the caller exits the context."""

    if coordinator is None:
        raise ValueError("weight snapshot write coordinator is required")

    store: Any | None = None
    provider: WeightTransferProvider | None = None
    endpoint: str | None = None
    local_setup_error: BaseException | None = None
    try:
        if not isinstance(spec, WeightSnapshotWriteSpec):
            raise ValueError("weight snapshot write spec is invalid")
        if spec.destination.provider != MooncakeWeightStoreProvider.name:
            raise ValueError("weight snapshot destination must use mooncake-store")
        if not callable(payload_checksum_verifier):
            raise ValueError("payload checksum verifier must be callable")

        options = spec.mooncake_store
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
        result = store.setup(setup)
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
            max_total_operations=_positive_integer(
                options,
                "max_total_operations",
                10_000_000,
            ),
        )
    except BaseException as error:
        local_setup_error = error

    try:
        setup_outcomes = coordinator.exchange_preflight_outcome(
            WeightStorePreflightOutcome(
                rank=coordinator.rank,
                error=(
                    None
                    if local_setup_error is None
                    else _format_backend_error(local_setup_error)
                ),
            )
        )
    except BaseException:
        _close_store(store)
        raise

    setup_failure = _outcome_error_message(setup_outcomes)
    if setup_failure is not None:
        _close_store(store)
        raise WeightStoreDistributedError(
            "initialize_write_backend",
            setup_failure,
        ) from local_setup_error
    if store is None or provider is None or endpoint is None:
        _close_store(store)
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

        coordinator.run_root(
            "catalog.initialize",
            initialize_catalog,
            discard_result=True,
        )
        backend = WeightSnapshotBackend(
            provider=provider,
            catalog=RootWeightStorageCatalog(
                root_catalog,
                coordinator,
            ),
            endpoint=endpoint,
        )
        yield backend
    except BaseException as error:
        primary_error = error
        raise
    finally:
        close_error = _close_store(store)
        try:
            close_outcomes = coordinator.exchange_preflight_outcome(
                WeightStorePreflightOutcome(
                    rank=coordinator.rank,
                    error=(
                        None
                        if close_error is None
                        else _format_backend_error(close_error)
                    ),
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
                )
