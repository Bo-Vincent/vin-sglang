# SPDX-License-Identifier: Apache-2.0

import enum
import importlib.util
import logging
import math
import threading
import time
import uuid
from bisect import bisect_right
from dataclasses import dataclass, replace
from typing import Any, List

import requests
from sglang.srt.model_executor.weight_runtime_manifest import (
    DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC,
    MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC,
)
from sglang.srt.weight_transfer.binding import runtime_manifest_to_parts
from sglang.srt.weight_transfer.provider import WeightTransferExecutionContext
from sglang.srt.weight_transfer.remote_protocol import (
    HF_REVISION_V1,
    PLACEMENT_BINDING_V1,
    RUNTIME_MANIFEST_V1,
    validate_manifest_revision_semantics,
)

logger = logging.getLogger(__name__)

LEGACY_HF_UNATTESTED = "legacy_hf_unattested"
BOUNDED_EXECUTION_CONTRACT_VERSION = 1
_MAX_REMOTE_TRANSFER_NEGOTIATION_STATES = 4
_MAX_REMOTE_TRANSFER_TRANSPORT_ATTEMPTS = 3
_CAPABILITY_PROBE_UNSET = object()
_BOUNDED_EXECUTION_CAPABILITY_UNSET = object()
_BUILTIN_BOUNDED_EXECUTION_ADAPTERS = frozenset(
    {
        (
            "sglang.srt.weight_transfer.mooncake",
            "MooncakeWeightTransferProvider",
        ),
        (
            "sglang.srt.weight_transfer.mooncake_store",
            "MooncakeWeightStoreProvider",
        ),
    }
)
_MOONCAKE_PLACEMENT_BINDING_APIS = (
    "RuntimeBindingManifest",
    "SourcePlacementManifest",
    "TargetPlacementManifest",
    "bind_logical_transfer_plan",
    "bind_runtime_manifest",
    "placement_manifest_from_runtime_manifest",
    "plan_placement_transfer_to_local_target",
    "runtime_binding_from_runtime_manifest",
)


class RemoteInstanceWeightLoaderBackend(str, enum.Enum):
    NCCL = "nccl"
    TRANSFER_ENGINE = "transfer_engine"
    MODELEXPRESS = "modelexpress"


@dataclass(frozen=True, slots=True)
class RemoteInstanceWeightTransferSession:
    transfer_id: str
    manifests: list[dict]
    lease_timeout_sec: int
    source_placements: list[dict] | None = None
    source_bindings: list[dict] | None = None
    manifest_format: str = RUNTIME_MANIFEST_V1
    manifest_revision_semantics: str = LEGACY_HF_UNATTESTED
    allow_legacy_hf_fallback: bool = False
    deadline_unix_sec: float | None = None
    lease_fence: str | None = None
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class RemoteInstanceWeightTransferSessionHandle:
    transfer_id: str
    lease_timeout_sec: int
    manifest_format: str
    manifest_revision_semantics: str
    allow_legacy_hf_fallback: bool
    deadline_unix_sec: float | None = None
    lease_fence: str | None = None
    generation: int | None = None


@dataclass(frozen=True, slots=True)
class RemoteInstanceWeightLeaseRenewal:
    deadline_unix_sec: float | None


@dataclass(frozen=True, slots=True)
class _RemoteTransferNegotiationState:
    manifest_format: str
    send_revision_semantics: bool
    send_lease_fence: bool


@dataclass(frozen=True, slots=True)
class RemoteInstanceWeightTransferCapabilities:
    native_executor: bool
    canonical_adapter: bool
    legacy_planner: bool
    native_contract_error: str | None = None
    legacy_contract_error: str | None = None

    @property
    def supports_placement_binding_v1(self) -> bool:
        return self.native_executor

    @property
    def supports_runtime_v1(self) -> bool:
        return (self.native_executor and self.canonical_adapter) or self.legacy_planner


def get_missing_legacy_runtime_v1_apis(backend: Any) -> tuple[str, ...]:
    exception_types = (
        "TransferCompletionUnknownError",
        "TransferEngineError",
    )
    missing = [
        name
        for name in (
            "MooncakeTransferEngineReader",
            "plan_runtime_transfer_to_local_target",
        )
        if not callable(getattr(backend, name, None))
    ]
    for owner_name, method_name in (
        ("MemoryRegistrationLease", "from_fragment"),
        ("RuntimeManifest", "from_runtime_inventory"),
    ):
        owner = getattr(backend, owner_name, None)
        if not callable(getattr(owner, method_name, None)):
            missing.append(f"{owner_name}.{method_name}")
    for name in exception_types:
        value = getattr(backend, name, None)
        if not isinstance(value, type) or not issubclass(value, Exception):
            missing.append(name)
    return tuple(missing)


def get_missing_native_weight_provider_apis(provider: Any) -> tuple[str, ...]:
    missing = [
        name
        for name in (
            "probe",
            "prepare",
            "submit",
            "wait",
            "cancel",
            "synchronize",
            "release",
        )
        if not callable(getattr(provider, name, None))
    ]
    if type(getattr(provider, "name", None)) is not str or not provider.name:
        missing.append("name")
    return tuple(missing)


def _bounded_execution_contract_version(executor: Any) -> Any:
    version = getattr(executor, "bounded_execution_contract_version", None)
    if version is not None:
        return version
    executor_type = type(executor)
    if (
        executor_type.__module__,
        executor_type.__name__,
    ) in _BUILTIN_BOUNDED_EXECUTION_ADAPTERS:
        return BOUNDED_EXECUTION_CONTRACT_VERSION
    return None


def bounded_execution_contract_error(
    executor: Any,
    *,
    role: str,
    supports_bounded_execution: Any = _BOUNDED_EXECUTION_CAPABILITY_UNSET,
) -> str | None:
    version = _bounded_execution_contract_version(executor)
    if type(version) is not int or version != BOUNDED_EXECUTION_CONTRACT_VERSION:
        return (
            f"{role} requires bounded execution contract version "
            f"{BOUNDED_EXECUTION_CONTRACT_VERSION}"
        )
    if (
        supports_bounded_execution is not _BOUNDED_EXECUTION_CAPABILITY_UNSET
        and supports_bounded_execution is not True
    ):
        return (
            f"{role} requires supports_bounded_execution=true for bounded "
            f"execution contract version {BOUNDED_EXECUTION_CONTRACT_VERSION}"
        )
    return None


def require_bounded_execution_contract(
    executor: Any,
    *,
    role: str,
    supports_bounded_execution: Any = _BOUNDED_EXECUTION_CAPABILITY_UNSET,
) -> None:
    error = bounded_execution_contract_error(
        executor,
        role=role,
        supports_bounded_execution=supports_bounded_execution,
    )
    if error is not None:
        raise RuntimeError(error)


def _load_legacy_mooncake_weight_backend() -> Any | None:
    try:
        from mooncake import weight_transfer
    except Exception:
        return None
    return weight_transfer


def supports_mooncake_placement_binding_v1() -> bool:
    """Deprecated compatibility probe; use the structured capability probe."""
    backend = _load_legacy_mooncake_weight_backend()
    if backend is None:
        return False
    try:
        supports = getattr(backend, "supports_weight_transfer_capability", None)
        return (
            callable(supports)
            and supports(PLACEMENT_BINDING_V1) is True
            and all(
                callable(getattr(backend, name, None))
                for name in _MOONCAKE_PLACEMENT_BINDING_APIS
            )
        )
    except Exception:
        return False


def probe_remote_instance_weight_transfer_capabilities(
    *,
    provider: Any = _CAPABILITY_PROBE_UNSET,
    legacy_backend: Any = _CAPABILITY_PROBE_UNSET,
) -> RemoteInstanceWeightTransferCapabilities:
    use_legacy_defaults = (
        provider is _CAPABILITY_PROBE_UNSET
        and legacy_backend is _CAPABILITY_PROBE_UNSET
    )
    if provider is _CAPABILITY_PROBE_UNSET:
        provider = None
    if legacy_backend is _CAPABILITY_PROBE_UNSET:
        legacy_backend = (
            _load_legacy_mooncake_weight_backend() if use_legacy_defaults else None
        )
    native_contract_error = None
    native_executor = provider is not None and not (
        get_missing_native_weight_provider_apis(provider)
    )
    if native_executor:
        native_contract_error = bounded_execution_contract_error(
            provider,
            role="native provider",
        )
        native_executor = native_contract_error is None
    validate_environment = getattr(provider, "validate_environment", None)
    if native_executor and callable(validate_environment):
        try:
            validate_environment()
        except Exception:
            logger.warning(
                "Configured weight transfer provider is unavailable",
                exc_info=True,
            )
            native_executor = False
            native_contract_error = "native provider environment validation failed"
    canonical_adapter = callable(runtime_manifest_to_parts)
    legacy_contract_error = None
    legacy_planner = legacy_backend is not None and not (
        get_missing_legacy_runtime_v1_apis(legacy_backend)
    )
    if legacy_planner:
        legacy_contract_error = bounded_execution_contract_error(
            legacy_backend,
            role="legacy backend",
            supports_bounded_execution=getattr(
                legacy_backend,
                "supports_bounded_execution",
                None,
            ),
        )
        legacy_planner = legacy_contract_error is None
    if (
        use_legacy_defaults
        and legacy_planner
        and supports_mooncake_placement_binding_v1()
    ):
        native_executor = True
    return RemoteInstanceWeightTransferCapabilities(
        native_executor=native_executor,
        canonical_adapter=canonical_adapter,
        legacy_planner=legacy_planner,
        native_contract_error=native_contract_error,
        legacy_contract_error=legacy_contract_error,
    )


class _SessionBoundWorldGroupCollectives:
    _METHOD_NAMES = (
        "broadcast_object",
        "all_gather_object",
        "gather_object",
        "scatter_object",
    )

    def __init__(
        self,
        world_group: Any,
        collective_coordinator: Any,
        *,
        execution_context_getter,
        execution_context_setter,
        failure_callback,
    ) -> None:
        rank = getattr(world_group, "rank_in_group", None)
        world_size = getattr(world_group, "world_size", None)
        collective_rank = getattr(
            collective_coordinator,
            "rank_in_group",
            getattr(collective_coordinator, "rank", None),
        )
        if rank != collective_rank or world_size != getattr(
            collective_coordinator, "world_size", None
        ):
            raise ValueError(
                "bounded collective coordinator does not match the target world"
            )
        self.world_group = world_group
        self.collective_coordinator = collective_coordinator
        self.execution_context_getter = execution_context_getter
        self.execution_context_setter = execution_context_setter
        self.failure_callback = failure_callback
        self._broadcast_count = 0
        self._all_gather_count = 0
        self._original_methods = {}
        self._bounded_methods = {
            name: getattr(collective_coordinator, name) for name in self._METHOD_NAMES
        }
        self._installed = False
        self._install()

    def _install(self) -> None:
        installed = []
        try:
            instance_vars = vars(self.world_group)
            for name in self._METHOD_NAMES:
                self._original_methods[name] = (
                    name in instance_vars,
                    instance_vars.get(name),
                )
                setattr(self.world_group, name, getattr(self, name))
                installed.append(name)
        except BaseException:
            for name in reversed(installed):
                had_instance_value, original = self._original_methods[name]
                if had_instance_value:
                    setattr(self.world_group, name, original)
                else:
                    delattr(self.world_group, name)
            raise
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        for name in reversed(self._METHOD_NAMES):
            had_instance_value, original = self._original_methods[name]
            if had_instance_value:
                setattr(self.world_group, name, original)
            else:
                delattr(self.world_group, name)
        self._installed = False

    def _run(self, phase: str, operation):
        try:
            return operation(self.execution_context_getter())
        except BaseException as error:
            try:
                self.failure_callback(phase, error)
            except Exception:
                logger.exception(
                    "Failed to quarantine remote-instance collective failure"
                )
            raise

    def broadcast_object(self, obj: Any = None, src: int = 0) -> Any:
        if self._broadcast_count == 0:
            synchronize_deadline = getattr(
                self.collective_coordinator,
                "synchronize_object_collective_deadline",
                None,
            )
            if callable(synchronize_deadline):
                phase = "remote_instance.acquire.deadline_control"
                self._run(
                    phase,
                    lambda context: self.execution_context_setter(
                        synchronize_deadline(
                            phase=phase,
                            execution_context=context,
                        )
                    ),
                )
        phase = (
            "remote_instance.acquire.broadcast"
            if self._broadcast_count == 0
            else (
                "remote_instance.finish.terminal_broadcast"
                if self._broadcast_count == 1
                else "remote_instance.recovery.terminal_broadcast"
            )
        )
        self._broadcast_count += 1
        return self._run(
            phase,
            lambda context: self._bounded_methods["broadcast_object"](
                obj,
                src=src,
                phase=phase,
                execution_context=context,
            ),
        )

    def all_gather_object(self, obj: Any) -> list[Any]:
        phase = (
            "remote_instance.readiness.gather"
            if self._all_gather_count == 0
            else "remote_instance.finish.gather"
        )
        self._all_gather_count += 1
        return self._run(
            phase,
            lambda context: self._bounded_methods["all_gather_object"](
                obj,
                phase=phase,
                execution_context=context,
            ),
        )

    def gather_object(self, obj: Any, dst: int = 0) -> list[Any] | None:
        phase = "remote_instance.central_plan.gather"
        return self._run(
            phase,
            lambda context: self._bounded_methods["gather_object"](
                obj,
                dst=dst,
                phase=phase,
                execution_context=context,
            ),
        )

    def scatter_object(
        self,
        objects: list[Any] | tuple[Any, ...] | None,
        src: int = 0,
    ) -> Any:
        phase = "remote_instance.central_plan.scatter"
        return self._run(
            phase,
            lambda context: self._bounded_methods["scatter_object"](
                objects,
                src=src,
                phase=phase,
                execution_context=context,
            ),
        )


class RemoteInstanceWeightTransferHeartbeat:
    def __init__(
        self,
        seed_url: str,
        transfer_id: str,
        *,
        lease_timeout_sec: int,
        renew_interval_sec: float | None = None,
        execution_context: WeightTransferExecutionContext | None = None,
        lease_fence: str | None = None,
        generation: int | None = None,
    ) -> None:
        if not seed_url or not transfer_id:
            raise ValueError("remote weight transfer identifiers must not be empty")
        if (
            isinstance(lease_timeout_sec, bool)
            or not isinstance(lease_timeout_sec, int)
            or lease_timeout_sec <= 0
        ):
            raise ValueError("lease_timeout_sec must be a positive integer")
        interval = (
            max(1.0, lease_timeout_sec / 3)
            if renew_interval_sec is None
            else renew_interval_sec
        )
        if interval <= 0 or interval >= lease_timeout_sec:
            raise ValueError(
                "renew_interval_sec must be positive and shorter than the lease"
            )
        self.seed_url = seed_url
        self.transfer_id = transfer_id
        self.lease_timeout_sec = lease_timeout_sec
        self.renew_interval_sec = interval
        self._stop_event = threading.Event()
        self._failure_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._lease_deadline = time.monotonic() + lease_timeout_sec
        self.execution_context = execution_context
        self.lease_fence = lease_fence
        self.generation = generation

    def _remaining_session_seconds(self) -> float:
        if self.execution_context is None:
            return math.inf
        return self.execution_context.remaining_seconds()

    def _renew(self) -> bool:
        remaining_session_sec = self._remaining_session_seconds()
        remaining_lease_sec = min(
            self._lease_deadline - time.monotonic(),
            remaining_session_sec,
        )
        if remaining_lease_sec <= 0:
            return False
        requested_lease_timeout_sec = self.lease_timeout_sec
        if math.isfinite(remaining_session_sec):
            if (
                remaining_session_sec
                < MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
            ):
                return remaining_lease_sec >= remaining_session_sec
            requested_lease_timeout_sec = min(
                self.lease_timeout_sec,
                int(remaining_session_sec),
            )
        renewal = renew_remote_instance_weight_transfer_lease(
            self.seed_url,
            self.transfer_id,
            requested_lease_timeout_sec,
            remaining_lease_sec=remaining_lease_sec,
            lease_fence=self.lease_fence,
            generation=self.generation,
            execution_context=self.execution_context,
        )
        if renewal is None:
            return False
        deadline_unix_sec = getattr(renewal, "deadline_unix_sec", None)
        if deadline_unix_sec is None:
            self._lease_deadline = time.monotonic() + requested_lease_timeout_sec
            return True
        granted_lease_sec = deadline_unix_sec - time.time()
        if math.isfinite(remaining_session_sec):
            granted_lease_sec = min(
                granted_lease_sec,
                remaining_session_sec,
            )
        if granted_lease_sec <= 0:
            return False
        self._lease_deadline = time.monotonic() + granted_lease_sec
        return True

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("remote weight transfer heartbeat was already started")
        if not self._renew():
            raise RuntimeError("initial source weight transfer lease renew failed")
        self._thread = threading.Thread(
            target=self._run,
            name=f"weight-transfer-heartbeat-{self.transfer_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                remaining_session_sec = self._remaining_session_seconds()
                if remaining_session_sec <= 0:
                    raise RuntimeError(
                        "remote weight transfer session deadline exceeded"
                    )
                wait_seconds = min(
                    self.renew_interval_sec,
                    remaining_session_sec,
                )
                if self._stop_event.wait(wait_seconds):
                    return
                if (
                    self.execution_context is not None
                    and self.execution_context.expired()
                ):
                    raise RuntimeError(
                        "remote weight transfer session deadline exceeded"
                    )
                if not self._renew():
                    raise RuntimeError("source weight transfer lease renew failed")
            except BaseException as error:
                with self._failure_lock:
                    if self._failure is None:
                        self._failure = error
                self._stop_event.set()
                return

    def raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError(
                f"Remote weight transfer lease renewal failed: {failure}"
            ) from failure

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is None:
            return
        join_timeout = min(
            35.0,
            max(1.0, self._remaining_session_seconds() + 1.0),
        )
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():
            raise RuntimeError("remote weight transfer heartbeat did not stop")


class RemoteInstanceWeightTransferWorldCoordinator:
    """Share one source snapshot lease across the target model world."""

    def __init__(
        self,
        seed_url: str,
        world_group: Any,
        *,
        capabilities: RemoteInstanceWeightTransferCapabilities | None = None,
        manifest_revision_semantics: str = HF_REVISION_V1,
        allow_legacy_hf_fallback: bool = False,
        session_timeout_sec: int = (
            DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
        ),
        collective_coordinator: Any | None = None,
    ) -> None:
        if (
            isinstance(session_timeout_sec, bool)
            or not isinstance(session_timeout_sec, int)
            or session_timeout_sec <= 0
        ):
            raise ValueError("session_timeout_sec must be a positive integer")
        self.seed_url = seed_url
        self.world_group = world_group
        self.capabilities = (
            probe_remote_instance_weight_transfer_capabilities()
            if capabilities is None
            else capabilities
        )
        self.manifest_revision_semantics = manifest_revision_semantics
        self.allow_legacy_hf_fallback = allow_legacy_hf_fallback
        self.session_timeout_sec = session_timeout_sec
        self.execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=time.time() + session_timeout_sec,
        )
        self.is_owner = world_group.rank_in_group == 0
        self.session: (
            RemoteInstanceWeightTransferSession
            | RemoteInstanceWeightTransferSessionHandle
            | None
        ) = None
        self._owner_source_session: RemoteInstanceWeightTransferSession | None = None
        self.heartbeat: RemoteInstanceWeightTransferHeartbeat | None = None
        self._acquired = False
        self._finished = False
        self._readiness_checked = False
        self._release_safe = True
        self._world_release_safe = False
        self._source_release_confirmed = False
        self._collective_poisoned = False
        self._session_collectives = None
        if (
            collective_coordinator is None
            and getattr(
                world_group,
                "cpu_group",
                None,
            )
            is not None
        ):
            collective_coordinator = world_group
        if collective_coordinator is not None and world_group.world_size > 1:
            self._session_collectives = _SessionBoundWorldGroupCollectives(
                world_group,
                collective_coordinator,
                execution_context_getter=lambda: self.execution_context,
                execution_context_setter=self._set_execution_context_deadline,
                failure_callback=self._handle_collective_failure,
            )

    def _handle_collective_failure(
        self,
        phase: str,
        error: BaseException,
    ) -> None:
        self._collective_poisoned = True
        self._release_safe = False
        self._world_release_safe = False
        self._stop_heartbeat()
        logger.critical(
            "Remote-instance target-world collective %s failed under the "
            "session deadline; the process group is poisoned and scheduler "
            "restart is required: %s: %s",
            phase,
            type(error).__name__,
            error,
        )

    def _restore_session_collectives(self) -> None:
        if self._session_collectives is not None:
            self._session_collectives.restore()

    def _set_execution_context_deadline(self, deadline_unix_sec: float) -> None:
        self.execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=deadline_unix_sec,
        )

    def _set_source_session_deadline(
        self,
        local_session: RemoteInstanceWeightTransferSession | Any,
    ) -> float:
        source_timeout_sec = getattr(local_session, "lease_timeout_sec", None)
        if (
            isinstance(source_timeout_sec, bool)
            or not isinstance(source_timeout_sec, int)
            or source_timeout_sec <= 0
        ):
            raise ValueError("source session lease timeout is invalid")
        deadline_unix_sec = min(
            self.execution_context.deadline_unix_sec,
            time.time() + source_timeout_sec,
        )
        self.execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=deadline_unix_sec,
        )
        return deadline_unix_sec

    @staticmethod
    def _remaining_session_timeout_sec(
        deadline_unix_sec: float,
        source_timeout_sec: int,
    ) -> int:
        return max(
            1,
            min(
                source_timeout_sec,
                int(max(0.0, deadline_unix_sec - time.time())),
            ),
        )

    def _share_source_session(
        self,
        local_session: RemoteInstanceWeightTransferSession | Any,
        deadline_unix_sec: float,
    ) -> RemoteInstanceWeightTransferSession | Any:
        if not isinstance(local_session, RemoteInstanceWeightTransferSession):
            return local_session
        remaining_timeout_sec = self._remaining_session_timeout_sec(
            deadline_unix_sec,
            local_session.lease_timeout_sec,
        )
        if local_session.manifest_format == PLACEMENT_BINDING_V1:
            self._owner_source_session = local_session
            return RemoteInstanceWeightTransferSessionHandle(
                transfer_id=local_session.transfer_id,
                lease_timeout_sec=remaining_timeout_sec,
                manifest_format=local_session.manifest_format,
                manifest_revision_semantics=local_session.manifest_revision_semantics,
                allow_legacy_hf_fallback=local_session.allow_legacy_hf_fallback,
                deadline_unix_sec=deadline_unix_sec,
                lease_fence=local_session.lease_fence,
                generation=local_session.generation,
            )
        return replace(
            local_session,
            lease_timeout_sec=remaining_timeout_sec,
            deadline_unix_sec=deadline_unix_sec,
        )

    def _adopt_shared_session_deadline(self) -> None:
        deadline_unix_sec = getattr(self.session, "deadline_unix_sec", None)
        if deadline_unix_sec is None:
            return
        if (
            isinstance(deadline_unix_sec, bool)
            or not isinstance(deadline_unix_sec, (int, float))
            or not math.isfinite(deadline_unix_sec)
            or deadline_unix_sec <= 0
        ):
            raise ValueError("target-world session deadline is invalid")
        if deadline_unix_sec != self.execution_context.deadline_unix_sec:
            self.execution_context = WeightTransferExecutionContext(
                deadline_unix_sec=float(deadline_unix_sec),
            )

    def acquire(
        self,
    ) -> (
        RemoteInstanceWeightTransferSession
        | RemoteInstanceWeightTransferSessionHandle
        | None
    ):
        if self._acquired:
            raise RuntimeError("remote weight transfer world session already acquired")
        self._acquired = True

        local_session = None
        if self.is_owner:
            try:
                local_session = begin_remote_instance_weight_transfer(
                    self.seed_url,
                    lease_timeout_sec=self.session_timeout_sec,
                    capabilities=self.capabilities,
                    manifest_revision_semantics=self.manifest_revision_semantics,
                    allow_legacy_hf_fallback=self.allow_legacy_hf_fallback,
                    execution_context=self.execution_context,
                )
            except Exception:
                logger.exception("Failed to acquire the source weight transfer session")
            if local_session is not None:
                try:
                    deadline_unix_sec = self._set_source_session_deadline(local_session)
                    self.heartbeat = RemoteInstanceWeightTransferHeartbeat(
                        self.seed_url,
                        local_session.transfer_id,
                        lease_timeout_sec=local_session.lease_timeout_sec,
                        execution_context=self.execution_context,
                        lease_fence=getattr(local_session, "lease_fence", None),
                        generation=getattr(local_session, "generation", None),
                    )
                    self.heartbeat.start()
                except Exception:
                    logger.exception("Failed to start remote weight transfer heartbeat")
                    _best_effort_release_invalid_transfer(
                        self.seed_url,
                        local_session.transfer_id,
                        attempts=3,
                        lease_fence=getattr(local_session, "lease_fence", None),
                        generation=getattr(local_session, "generation", None),
                        execution_context=self.execution_context,
                    )
                    self.heartbeat = None
                    local_session = None

        shared_session = local_session
        if local_session is not None:
            shared_session = self._share_source_session(
                local_session,
                deadline_unix_sec,
            )

        try:
            self.session = self.world_group.broadcast_object(shared_session, src=0)
        except Exception:
            self._owner_source_session = None
            self._stop_heartbeat()
            if self.is_owner and local_session is not None:
                _best_effort_release_invalid_transfer(
                    self.seed_url,
                    local_session.transfer_id,
                    attempts=3,
                    lease_fence=getattr(local_session, "lease_fence", None),
                    generation=getattr(local_session, "generation", None),
                    execution_context=self.execution_context,
                )
            logger.exception("Failed to broadcast the source weight transfer session")
            raise
        if self.session is None:
            self._restore_session_collectives()
            return None
        try:
            self._adopt_shared_session_deadline()
        except Exception:
            self._handle_collective_failure(
                "remote_instance.acquire.deadline",
                RuntimeError("target-world session deadline is invalid"),
            )
            raise
        return self.session

    @property
    def owner_source_session(self) -> RemoteInstanceWeightTransferSession | None:
        return self._owner_source_session if self.is_owner else None

    def clear_owner_source_session(self) -> None:
        self._owner_source_session = None

    def raise_if_failed(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat.raise_if_failed()

    def ready_for_transfer(self, local_ready: bool) -> bool:
        """Run the single target-world gate before any rank starts DMA."""
        if not self._acquired:
            raise RuntimeError("remote weight transfer world session was not acquired")
        if self._finished:
            raise RuntimeError("remote weight transfer world session already finished")
        if self._readiness_checked:
            raise RuntimeError("remote weight transfer readiness was already checked")
        self._readiness_checked = True
        if self.session is None:
            return False

        ready = bool(local_ready)
        if self.heartbeat is not None:
            try:
                self.heartbeat.raise_if_failed()
            except Exception:
                ready = False
                logger.exception(
                    "Source lease heartbeat failed before the target-world "
                    "transfer readiness gate"
                )

        try:
            gathered_readiness = self.world_group.all_gather_object(ready)
        except Exception:
            self._release_safe = False
            logger.exception(
                "Failed to gather target transfer readiness; source mutation "
                "remains blocked until explicit release or recovery"
            )
            return False

        readiness_valid = len(
            gathered_readiness
        ) == self.world_group.world_size and all(
            isinstance(value, bool) for value in gathered_readiness
        )
        if not readiness_valid:
            self._release_safe = False
            logger.error(
                "Target world returned invalid transfer readiness; source mutation "
                "remains blocked until explicit release or recovery"
            )
            return False
        return all(gathered_readiness)

    def finish(
        self,
        *,
        local_success: bool,
        local_release_safe: bool = True,
    ) -> tuple[bool, bool]:
        if not self._acquired:
            raise RuntimeError("remote weight transfer world session was not acquired")
        if self._finished:
            raise RuntimeError("remote weight transfer world session already finished")
        self._finished = True
        self._owner_source_session = None
        if self.session is None:
            self._restore_session_collectives()
            return False, True

        local_release_safe = bool(local_release_safe) and self._release_safe

        if self.heartbeat is not None:
            try:
                self.heartbeat.raise_if_failed()
            except Exception:
                local_success = False
                logger.exception(
                    "Remote weight transfer heartbeat failed before world sync"
                )

        try:
            gathered_outcomes = self.world_group.all_gather_object(
                (bool(local_success), bool(local_release_safe))
            )
        except Exception:
            logger.exception(
                "Failed to gather target transfer outcomes; source mutation "
                "remains blocked until explicit release or recovery"
            )
            return False, False

        outcomes_valid = len(gathered_outcomes) == self.world_group.world_size and all(
            isinstance(outcome, tuple)
            and len(outcome) == 2
            and isinstance(outcome[0], bool)
            and isinstance(outcome[1], bool)
            for outcome in gathered_outcomes
        )
        if not outcomes_valid:
            logger.error(
                "Target world returned invalid transfer outcomes; source mutation "
                "remains blocked until explicit release or recovery"
            )
            return False, False

        release_safe = all(outcome[1] for outcome in gathered_outcomes)
        self._world_release_safe = release_safe
        world_success = release_safe and all(
            outcome[0] for outcome in gathered_outcomes
        )
        release_success = True
        if self.is_owner:
            if release_safe:
                heartbeat_stopped = self._stop_heartbeat()
                if not heartbeat_stopped:
                    world_success = False
                if not self._source_release_confirmed:
                    try:
                        release_kwargs = {}
                        if getattr(self.session, "lease_fence", None) is not None:
                            release_kwargs = {
                                "lease_fence": self.session.lease_fence,
                                "generation": self.session.generation,
                                "execution_context": self.execution_context,
                            }
                        self._source_release_confirmed = bool(
                            release_remote_instance_weight_transfer(
                                self.seed_url,
                                self.session.transfer_id,
                                **release_kwargs,
                            )
                        )
                        if not self._source_release_confirmed:
                            logger.error(
                                "Failed to release source weight transfer %s; "
                                "source mutation remains blocked until explicit "
                                "release or recovery",
                                self.session.transfer_id,
                            )
                    except Exception:
                        release_success = False
                        logger.exception(
                            "Failed to release source weight transfer %s; source "
                            "mutation remains blocked until explicit release or "
                            "recovery",
                            self.session.transfer_id,
                        )
                release_success = self._source_release_confirmed
            else:
                release_success = False
                logger.error(
                    "Keeping source weight transfer %s leased because a target "
                    "rank could not confirm transfer completion; source mutation "
                    "remains blocked until explicit release or recovery",
                    self.session.transfer_id,
                )

        try:
            outcome = self.world_group.broadcast_object(
                (world_success, release_success) if self.is_owner else None,
                src=0,
            )
        except Exception:
            logger.exception("Failed to broadcast the target transfer outcome")
            return False, False
        if (
            not isinstance(outcome, tuple)
            or len(outcome) != 2
            or not all(isinstance(value, bool) for value in outcome)
        ):
            logger.error("Target world returned an invalid weight transfer outcome")
            return False, False
        if outcome[1]:
            self._restore_session_collectives()
        return outcome

    @property
    def world_release_safe(self) -> bool:
        return self._world_release_safe

    def release_after_terminal_recovery(
        self,
        *,
        completion_ticket: str,
        local_terminal_status: str,
    ) -> bool:
        if not self._acquired or not self._finished or self.session is None:
            raise RuntimeError(
                "remote weight transfer is not waiting for terminal recovery"
            )
        if type(completion_ticket) is not str or not completion_ticket:
            raise ValueError("completion_ticket must be a non-empty string")
        if local_terminal_status not in {
            "COMPLETED",
            "FAILED_DRAINED",
            "NO_SUBMISSION",
        }:
            raise ValueError(
                "local terminal completion status must be COMPLETED, "
                "FAILED_DRAINED, or NO_SUBMISSION"
            )
        if self._collective_poisoned:
            logger.critical(
                "Cannot recover source weight transfer %s through a poisoned "
                "target-world process group; the source lease must expire and "
                "the scheduler must restart",
                self.session.transfer_id,
            )
            return False

        local_result = None
        if self.is_owner:
            heartbeat_stopped = self._stop_heartbeat()
            if not heartbeat_stopped:
                logger.error(
                    "Heartbeat did not stop cleanly for recovered source "
                    "weight transfer %s; retrying the idempotent source release",
                    self.session.transfer_id,
                )
            if not self._source_release_confirmed:
                try:
                    release_kwargs = {}
                    if getattr(self.session, "lease_fence", None) is not None:
                        release_kwargs = {
                            "lease_fence": self.session.lease_fence,
                            "generation": self.session.generation,
                            "execution_context": self.execution_context,
                        }
                    self._source_release_confirmed = bool(
                        release_remote_instance_weight_transfer(
                            self.seed_url,
                            self.session.transfer_id,
                            **release_kwargs,
                        )
                    )
                except Exception:
                    self._source_release_confirmed = False
                    logger.exception(
                        "Failed to release recovered source weight transfer %s",
                        self.session.transfer_id,
                    )
            local_result = self._source_release_confirmed

        try:
            result = self.world_group.broadcast_object(
                local_result,
                src=0,
            )
        except Exception:
            logger.exception("Failed to broadcast recovered source release outcome")
            return False
        if type(result) is not bool:
            logger.error(
                "Target world returned invalid recovered source release outcome"
            )
            return False
        if result:
            self._world_release_safe = True
            self._restore_session_collectives()
        return result

    def _stop_heartbeat(self) -> bool:
        if self.heartbeat is None:
            return True
        heartbeat = self.heartbeat
        self.heartbeat = None
        stopped = True
        try:
            heartbeat.stop()
        except Exception:
            stopped = False
            logger.exception("Remote weight transfer heartbeat did not stop cleanly")
        try:
            heartbeat.raise_if_failed()
        except Exception:
            logger.exception(
                "Remote weight transfer heartbeat had already failed before stopping"
            )
        return stopped


def trigger_init_weights_send_group_for_remote_instance_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    # Only support loading weights from instance with same parallelism strategy.
    # Per TP rank pair between seed and dst instances will build a communication group for sending weights.
    # i.e. seed TP 0 <-> dst TP 0, seed TP 1 <-> dst TP 1, etc.
    # Each communication group will have a world size 2.
    try:
        requests.post(
            f"{seed_instance_service_url}/init_weights_send_group_for_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_rank": 0,
                "world_size": 2,
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
                "backend": "nccl",
            },
        )
    except Exception as e:
        logger.error(
            f"Failed to trigger init_weights_send_group_for_remote_instance_request to seed instance {seed_instance_service_url}: {e}."
        )
        raise


def trigger_transferring_weights_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    try:
        requests.post(
            f"{seed_instance_service_url}/send_weights_to_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
            },
        )
    except Exception as e:
        logger.error(f"Failed to trigger send weights to remote instance request: {e}")
        raise


def get_remote_instance_transfer_engine_info_per_rank(seed_url: str, rank: int):
    try:
        response = requests.get(
            f"{seed_url}/get_remote_instance_transfer_engine_info",
            params={
                "rank": rank,
            },
        )

        if response.status_code == 200:
            data = response.json()

            if "remote_instance_transfer_engine_info" in data:
                return data["remote_instance_transfer_engine_info"]
            else:
                logger.error(
                    "Failed to get `remote_instance_transfer_engine_info` in response."
                )
                return None, None
        else:
            logger.error(f"request.get failed: {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None, None


def _unsupported_manifest_format_response(response) -> bool:
    if response.status_code not in (400, 409, 422):
        return False
    details = str(getattr(response, "text", ""))
    try:
        details = f"{details} {response.json()}"
    except Exception:
        pass
    details = details.lower()
    names_format = "manifest_format" in details or PLACEMENT_BINDING_V1 in details
    rejects_format = any(
        marker in details
        for marker in (
            "unsupported",
            "not supported",
            "unexpected",
            "unknown",
            "extra_forbidden",
            "literal_error",
            "input should be",
            "invalid",
        )
    )
    return names_format and rejects_format


def _unsupported_revision_semantics_response(response) -> bool:
    if response.status_code not in (400, 409, 422):
        return False
    details = str(getattr(response, "text", ""))
    try:
        details = f"{details} {response.json()}"
    except Exception:
        pass
    details = details.lower()
    return "manifest_revision_semantics" in details and any(
        marker in details
        for marker in (
            "unsupported",
            "not supported",
            "unexpected",
            "unknown",
            "extra_forbidden",
            "literal_error",
            "input should be",
            "invalid",
        )
    )


def _unsupported_lease_fence_response(response) -> bool:
    if response.status_code not in (400, 409, 422):
        return False
    details = str(getattr(response, "text", ""))
    try:
        details = f"{details} {response.json()}"
    except Exception:
        pass
    details = details.lower()
    return "lease_fence" in details and any(
        marker in details
        for marker in (
            "unsupported",
            "not supported",
            "unexpected",
            "unknown",
            "extra_forbidden",
            "invalid",
        )
    )


def _best_effort_release_invalid_transfer(
    seed_url: str,
    transfer_id: str,
    *,
    attempts: int = 1,
    lease_fence: str | None = None,
    generation: int | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
    resolve_fence: bool = False,
) -> bool:
    if lease_fence is None and resolve_fence:
        request_timeout_sec = _remote_transfer_http_timeout(execution_context)
        if request_timeout_sec is not None:
            try:
                response = requests.get(
                    f"{seed_url}/remote_instance_weight_transfer/{transfer_id}",
                    timeout=request_timeout_sec,
                )
                if response.status_code == 200:
                    payload = response.json()
                    if isinstance(payload, dict):
                        candidate_fence = payload.get("lease_fence")
                        candidate_generation = payload.get("generation")
                        if type(candidate_fence) is str and candidate_fence:
                            lease_fence = candidate_fence
                            if type(candidate_generation) is int:
                                generation = candidate_generation
            except Exception:
                logger.exception(
                    "Failed to resolve remote weight transfer lease fence for %s",
                    transfer_id,
                )
    for _ in range(attempts):
        try:
            if lease_fence is None and generation is None:
                released = release_remote_instance_weight_transfer(
                    seed_url,
                    transfer_id,
                )
            else:
                released = release_remote_instance_weight_transfer(
                    seed_url,
                    transfer_id,
                    lease_fence=lease_fence,
                    generation=generation,
                    execution_context=execution_context,
                )
            if released:
                return True
        except Exception:
            logger.exception(
                "Failed to clean up invalid remote weight transfer %s",
                transfer_id,
            )
    logger.error(
        "Failed to clean up invalid remote weight transfer %s; source mutation "
        "remains blocked until explicit release or recovery",
        transfer_id,
    )
    return False


def _remote_transfer_http_timeout(
    execution_context: WeightTransferExecutionContext | None,
    *,
    reserve_cleanup: bool = False,
) -> float | None:
    if execution_context is None:
        return 30.0
    remaining_sec = execution_context.remaining_seconds()
    if remaining_sec <= 0:
        return None
    if reserve_cleanup:
        remaining_sec /= 2
    return min(30.0, remaining_sec)


def begin_remote_instance_weight_transfer(
    seed_url: str,
    lease_timeout_sec: int = 300,
    *,
    transfer_id: str | None = None,
    lease_fence: str | None = None,
    capabilities: RemoteInstanceWeightTransferCapabilities | None = None,
    manifest_revision_semantics: str = HF_REVISION_V1,
    allow_legacy_hf_fallback: bool = False,
    execution_context: WeightTransferExecutionContext | None = None,
):
    if transfer_id is None:
        transfer_id = uuid.uuid4().hex
    if not isinstance(transfer_id, str) or not transfer_id:
        raise ValueError("transfer_id must be a non-empty string")
    lease_fence = lease_fence or uuid.uuid4().hex
    if type(lease_fence) is not str or not lease_fence:
        raise ValueError("lease_fence must be a non-empty string")
    if capabilities is None:
        capabilities = probe_remote_instance_weight_transfer_capabilities()
    if not (
        capabilities.supports_placement_binding_v1 or capabilities.supports_runtime_v1
    ):
        logger.error(
            "No executable remote-instance weight transfer manifest path is available."
        )
        return None
    manifest_format = (
        PLACEMENT_BINDING_V1
        if capabilities.supports_placement_binding_v1
        else RUNTIME_MANIFEST_V1
    )
    validate_manifest_revision_semantics(
        manifest_format,
        manifest_revision_semantics,
    )
    allow_runtime_fallback = (
        manifest_format == PLACEMENT_BINDING_V1 and capabilities.supports_runtime_v1
    )
    negotiation_state = _RemoteTransferNegotiationState(
        manifest_format=manifest_format,
        send_revision_semantics=True,
        send_lease_fence=True,
    )
    attempted_states = set()
    transport_attempt = 0

    while True:
        if negotiation_state not in attempted_states:
            if len(attempted_states) >= _MAX_REMOTE_TRANSFER_NEGOTIATION_STATES:
                logger.error(
                    "Remote weight transfer compatibility negotiation exhausted "
                    "%d states",
                    _MAX_REMOTE_TRANSFER_NEGOTIATION_STATES,
                )
                return None
            attempted_states.add(negotiation_state)
            transport_attempt = 0
        transport_attempt += 1
        try:
            request_timeout_sec = _remote_transfer_http_timeout(
                execution_context,
                reserve_cleanup=True,
            )
            if request_timeout_sec is None:
                break
            params = {
                "lease_timeout_sec": lease_timeout_sec,
                "manifest_format": negotiation_state.manifest_format,
                "transfer_id": transfer_id,
            }
            if negotiation_state.send_revision_semantics:
                params["manifest_revision_semantics"] = manifest_revision_semantics
            if negotiation_state.send_lease_fence:
                params["lease_fence"] = lease_fence
            response = requests.post(
                f"{seed_url}/remote_instance_weight_transfer",
                params=params,
                timeout=request_timeout_sec,
            )
            if response.status_code != 200:
                try:
                    error_payload = response.json()
                    response_transfer_id = error_payload.get("transfer_id")
                    response_session_state = error_payload.get("session_state")
                    response_lease_fence = error_payload.get("lease_fence")
                    response_generation = error_payload.get("generation")
                except Exception:
                    response_transfer_id = None
                    response_session_state = None
                    response_lease_fence = None
                    response_generation = None
                if response_transfer_id == transfer_id and response_session_state in {
                    "created",
                    "cleanup_pending",
                }:
                    _best_effort_release_invalid_transfer(
                        seed_url,
                        response_transfer_id,
                        attempts=3,
                        lease_fence=response_lease_fence,
                        generation=response_generation,
                        execution_context=execution_context,
                        resolve_fence=True,
                    )
                if (
                    negotiation_state.send_revision_semantics
                    and allow_legacy_hf_fallback
                    and _unsupported_revision_semantics_response(response)
                ):
                    negotiation_state = replace(
                        negotiation_state,
                        send_revision_semantics=False,
                    )
                    continue
                if (
                    negotiation_state.send_lease_fence
                    and _unsupported_lease_fence_response(response)
                ):
                    negotiation_state = replace(
                        negotiation_state,
                        send_lease_fence=False,
                    )
                    continue
                if (
                    allow_runtime_fallback
                    and response_transfer_id is None
                    and _unsupported_manifest_format_response(response)
                ):
                    negotiation_state = replace(
                        negotiation_state,
                        manifest_format=RUNTIME_MANIFEST_V1,
                    )
                    allow_runtime_fallback = False
                    continue
                logger.error(
                    "Failed to begin remote weight transfer: %s: %s",
                    response.status_code,
                    getattr(response, "text", ""),
                )
                return None

            payload = response.json()
            response_transfer_id = payload.get("transfer_id")
            response_lease_fence = payload.get("lease_fence")
            response_generation = payload.get("generation")
            if execution_context is not None and execution_context.expired():
                _best_effort_release_invalid_transfer(
                    seed_url,
                    response_transfer_id or transfer_id,
                    attempts=3,
                    lease_fence=response_lease_fence,
                    generation=response_generation,
                    execution_context=execution_context,
                    resolve_fence=True,
                )
                return None
            if response_transfer_id != transfer_id:
                if response_transfer_id:
                    _best_effort_release_invalid_transfer(
                        seed_url,
                        response_transfer_id,
                        lease_fence=response_lease_fence,
                        generation=response_generation,
                        execution_context=execution_context,
                        resolve_fence=True,
                    )
                _best_effort_release_invalid_transfer(
                    seed_url,
                    transfer_id,
                    attempts=3,
                    execution_context=execution_context,
                    resolve_fence=True,
                )
                logger.error(
                    "Remote instance returned a different transfer ID: %r != %r",
                    response_transfer_id,
                    transfer_id,
                )
                return None
            manifests = payload.get("weight_runtime_manifests") or []
            source_placements = payload.get("source_weight_placements")
            source_bindings = payload.get("source_weight_runtime_bindings")
            server_lease_timeout_sec = payload.get(
                "lease_timeout_sec", lease_timeout_sec
            )
            lease_timeout_valid = (
                not isinstance(server_lease_timeout_sec, bool)
                and isinstance(server_lease_timeout_sec, int)
                and server_lease_timeout_sec > 0
            )

            split_manifest_valid = (
                bool(source_placements)
                and bool(source_bindings)
                and len(source_placements) == len(source_bindings)
            )
            runtime_manifest_valid = bool(manifests)
            if split_manifest_valid and capabilities.supports_placement_binding_v1:
                actual_manifest_format = PLACEMENT_BINDING_V1
            elif runtime_manifest_valid and capabilities.supports_runtime_v1:
                actual_manifest_format = RUNTIME_MANIFEST_V1
            else:
                actual_manifest_format = negotiation_state.manifest_format
            actual_revision_semantics = payload.get(
                "manifest_revision_semantics",
                LEGACY_HF_UNATTESTED,
            )
            if actual_revision_semantics != LEGACY_HF_UNATTESTED:
                try:
                    validate_manifest_revision_semantics(
                        actual_manifest_format,
                        actual_revision_semantics,
                    )
                except ValueError:
                    actual_revision_semantics = ""
            payload_valid = (
                lease_timeout_valid
                and (
                    split_manifest_valid
                    if actual_manifest_format == PLACEMENT_BINDING_V1
                    else runtime_manifest_valid and capabilities.supports_runtime_v1
                )
                and bool(actual_revision_semantics)
            )
            if not payload_valid:
                if transfer_id:
                    _best_effort_release_invalid_transfer(
                        seed_url,
                        transfer_id,
                        attempts=3,
                        lease_fence=response_lease_fence,
                        generation=response_generation,
                        execution_context=execution_context,
                        resolve_fence=True,
                    )
                logger.error("Remote instance returned an incomplete transfer session.")
                return None

            return RemoteInstanceWeightTransferSession(
                transfer_id=transfer_id,
                manifests=manifests,
                lease_timeout_sec=server_lease_timeout_sec,
                source_placements=source_placements,
                source_bindings=source_bindings,
                manifest_format=actual_manifest_format,
                manifest_revision_semantics=actual_revision_semantics,
                allow_legacy_hf_fallback=allow_legacy_hf_fallback,
                lease_fence=(
                    response_lease_fence
                    if type(response_lease_fence) is str and response_lease_fence
                    else None
                ),
                generation=(
                    response_generation
                    if type(response_generation) is int and response_generation > 0
                    else None
                ),
            )
        except Exception as error:
            logger.error("Failed to begin remote weight transfer: %s", error)
            if transport_attempt < _MAX_REMOTE_TRANSFER_TRANSPORT_ATTEMPTS:
                continue
            _best_effort_release_invalid_transfer(
                seed_url,
                transfer_id,
                attempts=3,
                execution_context=execution_context,
                resolve_fence=True,
            )
            return None

    return None


def release_remote_instance_weight_transfer(
    seed_url: str,
    transfer_id: str,
    *,
    lease_fence: str | None = None,
    generation: int | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
) -> bool:
    request_timeout_sec = _remote_transfer_http_timeout(execution_context)
    if request_timeout_sec is None:
        return False
    params = {}
    if lease_fence is not None:
        params["lease_fence"] = lease_fence
    if generation is not None:
        params["generation"] = generation
    try:
        kwargs = {"timeout": request_timeout_sec}
        if params:
            kwargs["params"] = params
        response = requests.delete(
            f"{seed_url}/remote_instance_weight_transfer/{transfer_id}", **kwargs
        )
        if response.status_code == 200:
            return True
        logger.error(
            "Failed to release remote weight transfer %s: %s: %s",
            transfer_id,
            response.status_code,
            response.text,
        )
    except Exception as error:
        logger.error(
            "Failed to release remote weight transfer %s: %s",
            transfer_id,
            error,
        )
    return False


def renew_remote_instance_weight_transfer_lease(
    seed_url: str,
    transfer_id: str,
    lease_timeout_sec: int,
    *,
    remaining_lease_sec: float | None = None,
    lease_fence: str | None = None,
    generation: int | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
) -> RemoteInstanceWeightLeaseRenewal | None:
    remaining_lease_sec = (
        float(lease_timeout_sec)
        if remaining_lease_sec is None
        else float(remaining_lease_sec)
    )
    if not math.isfinite(remaining_lease_sec) or remaining_lease_sec <= 0:
        logger.error(
            "Cannot renew remote weight transfer %s after its lease window elapsed",
            transfer_id,
        )
        return None
    request_timeout_sec = min(30.0, remaining_lease_sec / 2)
    bounded_timeout_sec = _remote_transfer_http_timeout(execution_context)
    if bounded_timeout_sec is None:
        return None
    request_timeout_sec = min(request_timeout_sec, bounded_timeout_sec)
    if request_timeout_sec <= 0:
        return None
    try:
        params = {"lease_timeout_sec": lease_timeout_sec}
        if lease_fence is not None:
            params["lease_fence"] = lease_fence
        if generation is not None:
            params["generation"] = generation
        response = requests.post(
            f"{seed_url}/remote_instance_weight_transfer/{transfer_id}/renew",
            params=params,
            timeout=request_timeout_sec,
        )
        if response.status_code == 200:
            try:
                payload = response.json()
            except (AttributeError, ValueError):
                payload = {}
            deadline_unix_sec = (
                payload.get("deadline_unix_sec") if isinstance(payload, dict) else None
            )
            if deadline_unix_sec is not None and (
                isinstance(deadline_unix_sec, bool)
                or not isinstance(deadline_unix_sec, (int, float))
                or not math.isfinite(deadline_unix_sec)
                or deadline_unix_sec <= 0
            ):
                logger.error(
                    "Remote weight transfer %s returned an invalid lease deadline",
                    transfer_id,
                )
                return None
            return RemoteInstanceWeightLeaseRenewal(
                deadline_unix_sec=(
                    None if deadline_unix_sec is None else float(deadline_unix_sec)
                )
            )
        logger.error(
            "Failed to renew remote weight transfer %s: %s: %s",
            transfer_id,
            response.status_code,
            response.text,
        )
    except Exception as error:
        logger.error(
            "Failed to renew remote weight transfer %s: %s",
            transfer_id,
            error,
        )
    return None


def renew_remote_instance_weight_transfer(
    seed_url: str,
    transfer_id: str,
    lease_timeout_sec: int,
    *,
    remaining_lease_sec: float | None = None,
) -> bool:
    return (
        renew_remote_instance_weight_transfer_lease(
            seed_url,
            transfer_id,
            lease_timeout_sec,
            remaining_lease_sec=remaining_lease_sec,
        )
        is not None
    )


def register_memory_region(model, transfer_engine):
    if importlib.util.find_spec("torch") is None:
        return register_memory_region_v1(model, transfer_engine)
    else:
        return register_memory_region_v2(model, transfer_engine)


def register_memory_region_v1(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    for name, weight in model.named_parameters():
        ret = transfer_engine.register_memory(
            weight.data_ptr(), weight.numel() * weight.element_size()
        )
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight {name}, error: {ret}"
            )
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )

    end_tic = time.time()
    logger.debug(f"Register memory region time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict


def register_memory_region_v2(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    weight_ranges = []
    for name, weight in model.named_parameters():
        address = int(weight.data_ptr())
        numel = int(weight.numel())
        itemsize = int(weight.element_size())
        nbytes = numel * itemsize
        if address <= 0 or nbytes <= 0:
            raise RuntimeError(f"weight {name} has no registerable storage")
        weight_mr_dict[name] = (address, numel, itemsize)
        weight_ranges.append((name, address, address + nbytes))

    import torch

    memory_snapshot = torch.cuda.memory.memory_snapshot()
    active_blocks = []
    for segment_index, segment in enumerate(memory_snapshot):
        for block in segment.get("blocks", []):
            address = block.get("address", -1)
            size = block.get("size", -1)
            if (
                type(address) is int
                and type(size) is int
                and address >= 0
                and size > 0
                and block.get("state") == "active_allocated"
            ):
                active_blocks.append((address, address + size, segment_index))
    active_blocks.sort()
    block_starts = [block[0] for block in active_blocks]
    selected_indices = set()
    for name, begin, end in weight_ranges:
        index = bisect_right(block_starts, begin) - 1
        if index < 0 or end > active_blocks[index][1]:
            raise RuntimeError(
                f"weight {name} is not covered by an active CUDA allocation"
            )
        selected_indices.add(index)

    weight_blocks_for_reg_mr = []
    for index in sorted(selected_indices):
        begin, end, segment_index = active_blocks[index]
        if (
            weight_blocks_for_reg_mr
            and weight_blocks_for_reg_mr[-1][1] == begin
            and weight_blocks_for_reg_mr[-1][2] == segment_index
        ):
            previous_begin, _, _ = weight_blocks_for_reg_mr[-1]
            weight_blocks_for_reg_mr[-1] = (
                previous_begin,
                end,
                segment_index,
            )
        else:
            weight_blocks_for_reg_mr.append((begin, end, segment_index))

    for begin, end, _ in weight_blocks_for_reg_mr:
        ret = transfer_engine.register_memory(begin, end - begin)
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight block at address {begin} "
                f"with size {end - begin}, error: {ret}"
            )

    end_tic = time.time()
    logger.debug(f"Register memory region v2 time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict
