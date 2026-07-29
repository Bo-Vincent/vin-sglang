from __future__ import annotations

import sys
import threading
import time
import types
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import sglang.srt.weight_transfer.store_runtime as store_runtime
from sglang.srt.weight_transfer.distributed import (
    LocalWeightStoreDistributedCoordinator,
    RootWeightStorageCatalog,
    WeightStoreDistributedError,
    WeightStorePreflightOutcome,
)
from sglang.srt.weight_transfer.mooncake_store import MooncakeWeightStoreProvider
from sglang.srt.weight_transfer.provider import WeightTransferExecutionContext
from sglang.srt.weight_transfer.store_runtime import (
    WeightSnapshotBackend,
    WeightSnapshotBackendStatus,
    WeightSnapshotLoadSpec,
    WeightSnapshotWriteSpec,
    open_weight_snapshot_backend,
    open_weight_snapshot_write_backend,
    register_weight_snapshot_write_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _spec(tmp_path, *, provider: str = "mooncake-store"):
    return WeightSnapshotLoadSpec.from_mapping(
        {
            "model_id": "model",
            "revision": "revision",
            "catalog_path": str(tmp_path / "catalog.json"),
            "ref": {
                "provider": provider,
                "storage_id": "weights/default/model/revision",
                "manifest_key": "weights/default/model/revision/manifest",
                "manifest_digest": f"sha256:{'a' * 64}",
            },
            "mooncake_store": {
                "setup": {
                    "local_hostname": "target-worker",
                    "metadata_server": "http://metadata",
                    "protocol": "tcp",
                },
                "namespace": "serving",
                "key_prefix": "model-weights",
                "max_range_bytes": 4096,
                "max_ranges_per_request": 7,
                "max_region_segments": 29,
                "max_total_operations": 31,
                "target_pre_registered": True,
            },
        }
    )


def _write_spec(tmp_path, **overrides: Any):
    value = {
        "catalog_path": str(tmp_path / "write-catalog.json"),
        "destination": {
            "provider": "mooncake-store",
            "storage_id": "model/revision",
            "object_prefix": "objects/model/revision",
        },
        "endpoint": "source-worker",
        "provider_options": {
            "setup": {
                "local_hostname": "source-worker",
                "metadata_server": "http://metadata",
                "protocol": "tcp",
            },
            "namespace": "serving",
            "key_prefix": "model-weights",
            "max_range_bytes": 4096,
            "max_ranges_per_request": 7,
            "max_region_segments": 29,
            "max_total_operations": 31,
            "source_pre_registered": True,
        },
    }
    value.update(overrides)
    return WeightSnapshotWriteSpec.from_mapping(value)


def _install_fake_mooncake(
    monkeypatch,
    *,
    setup_result: int = 0,
    setup_error: BaseException | None = None,
    close_error: BaseException | None = None,
    setup_started: threading.Event | None = None,
    release_setup: threading.Event | None = None,
    close_started: threading.Event | None = None,
    release_close: threading.Event | None = None,
):
    stores = []
    weight_stores = []

    class FakeDistributedStore:
        def __init__(self):
            self.setup_options = None
            self.closed = False
            self.setup_thread = None
            self.close_thread = None
            stores.append(self)

        def setup(self, options):
            self.setup_thread = threading.get_ident()
            self.setup_options = options
            if setup_started is not None:
                setup_started.set()
            if release_setup is not None:
                release_setup.wait(timeout=5)
            if setup_error is not None:
                raise setup_error
            return setup_result

        def close(self):
            self.close_thread = threading.get_ident()
            self.closed = True
            if close_started is not None:
                close_started.set()
            if release_close is not None:
                release_close.wait(timeout=5)
            if close_error is not None:
                raise close_error

    class FakeWeightStore:
        def __init__(self, store, **options):
            self.store = store
            self.options = options
            self.max_ranges_per_request = options["max_ranges_per_request"]
            self.max_region_segments = options["max_region_segments"]
            weight_stores.append(self)

    mooncake_package = types.ModuleType("mooncake")
    mooncake_package.__path__ = []
    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = FakeDistributedStore
    transfer_module = types.ModuleType("mooncake.weight_transfer")
    transfer_module.WeightStore = FakeWeightStore
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", transfer_module)
    return stores, weight_stores


@contextmanager
def _isolated_write_backend_registry():
    registry = store_runtime._WEIGHT_SNAPSHOT_WRITE_BACKENDS
    original = dict(registry)
    try:
        yield registry
    finally:
        registry.clear()
        registry.update(original)


class _SequencedCoordinator:
    world_size = 2

    def __init__(
        self,
        *,
        rank: int = 0,
        remote_setup_error: str | None = None,
        remote_close_error: str | None = None,
        remote_setup_completion_unknown: bool = False,
        remote_close_completion_unknown: bool = False,
    ) -> None:
        self.rank = rank
        self.remote_setup_error = remote_setup_error
        self.remote_close_error = remote_close_error
        self.remote_setup_completion_unknown = remote_setup_completion_unknown
        self.remote_close_completion_unknown = remote_close_completion_unknown
        self.exchange_calls = 0
        self.exchange_contexts = []
        self.root_calls: list[tuple[str, bool]] = []

    def exchange_preflight_outcome(self, outcome, *, execution_context=None):
        self.exchange_contexts.append(execution_context)
        remote_error = (
            self.remote_setup_error
            if self.exchange_calls == 0
            else self.remote_close_error
        )
        remote_completion_unknown = (
            self.remote_setup_completion_unknown
            if self.exchange_calls == 0
            else self.remote_close_completion_unknown
        )
        self.exchange_calls += 1
        remote = WeightStorePreflightOutcome(
            rank=1 - self.rank,
            error=remote_error,
            completion_unknown=remote_completion_unknown,
        )
        return tuple(sorted((outcome, remote), key=lambda item: item.rank))

    def run_root(self, phase, factory, *, discard_result=False):
        self.root_calls.append((phase, discard_result))
        if self.rank != 0:
            return None
        result = factory()
        return None if discard_result else result


def test_write_spec_parses_strict_destination_and_store_options(tmp_path) -> None:
    spec = _write_spec(tmp_path)

    assert spec.catalog_path == str(tmp_path / "write-catalog.json")
    assert spec.destination.provider == "mooncake-store"
    assert spec.destination.storage_id == "model/revision"
    assert spec.destination.object_prefix == "objects/model/revision"
    assert spec.endpoint == "source-worker"
    assert spec.provider_options["source_pre_registered"] is True
    assert spec.mooncake_store == spec.provider_options


def test_load_spec_preserves_legacy_positional_field_order(tmp_path) -> None:
    parsed = _spec(tmp_path)

    legacy = WeightSnapshotLoadSpec(
        parsed.model_id,
        parsed.revision,
        parsed.catalog_path,
        parsed.ref,
        parsed.endpoint,
        parsed.instance_id,
        parsed.mooncake_store,
    )
    explicit_timeout = WeightSnapshotLoadSpec(
        parsed.model_id,
        parsed.revision,
        parsed.catalog_path,
        parsed.ref,
        parsed.endpoint,
        parsed.instance_id,
        parsed.mooncake_store,
        17,
    )

    assert legacy.mooncake_store == parsed.mooncake_store
    assert legacy.load_timeout_sec == 1800
    assert explicit_timeout.mooncake_store == parsed.mooncake_store
    assert explicit_timeout.load_timeout_sec == 17


def test_write_spec_normalizes_legacy_and_new_constructor_aliases(tmp_path) -> None:
    parsed = _write_spec(tmp_path)
    options = dict(parsed.provider_options)

    positional = WeightSnapshotWriteSpec(
        parsed.catalog_path,
        parsed.destination,
        parsed.endpoint,
        options,
    )
    legacy_keyword = WeightSnapshotWriteSpec(
        parsed.catalog_path,
        parsed.destination,
        parsed.endpoint,
        mooncake_store=options,
    )
    provider_keyword = WeightSnapshotWriteSpec(
        parsed.catalog_path,
        parsed.destination,
        parsed.endpoint,
        provider_options=options,
    )

    for spec in (positional, legacy_keyword, provider_keyword):
        assert spec.mooncake_store == options
        assert spec.provider_options == options

    updated = replace(provider_keyword, endpoint="source-worker-2")
    assert updated.endpoint == "source-worker-2"
    assert updated.mooncake_store == options
    assert updated.provider_options == options


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"unknown": True}, "unknown weight snapshot write options"),
        (
            {
                "destination": {
                    "provider": "mooncake-store",
                    "storage_id": "model/revision",
                    "object_prefix": "objects",
                    "manifest_key": "forbidden",
                }
            },
            "unknown weight snapshot destination",
        ),
        (
            {
                "provider_options": {},
                "mooncake_store": {
                    "setup": {"local_hostname": "source"},
                    "target_pre_registered": True,
                },
            },
            "unknown Mooncake Store write options",
        ),
    ],
)
def test_write_spec_rejects_unknown_fields(
    tmp_path,
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _write_spec(tmp_path, **overrides)


def test_write_spec_accepts_registered_provider_options(tmp_path) -> None:
    spec = _write_spec(
        tmp_path,
        destination={
            "provider": "checkpoint-test",
            "storage_id": "model/revision",
            "object_prefix": "objects",
        },
        provider_options={"bucket": "weights"},
    )

    assert spec.destination.provider == "checkpoint-test"
    assert spec.provider_options == {"bucket": "weights"}


def test_write_spec_rejects_both_mooncake_option_forms(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="mutually exclusive",
    ):
        _write_spec(
            tmp_path,
            mooncake_store={"setup": {"local_hostname": "source"}},
        )


def test_write_spec_accepts_legacy_mooncake_options(tmp_path) -> None:
    options = dict(_write_spec(tmp_path).provider_options)

    spec = _write_spec(
        tmp_path,
        provider_options={},
        mooncake_store=options,
    )

    assert spec.provider_options == options
    assert spec.mooncake_store == options


def test_write_spec_rejects_mooncake_options_for_other_provider(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="non-Mooncake writes do not accept mooncake_store options",
    ):
        _write_spec(
            tmp_path,
            destination={
                "provider": "checkpoint-test",
                "storage_id": "model/revision",
                "object_prefix": "objects",
            },
            provider_options={},
            mooncake_store={"setup": {"local_hostname": "source"}},
        )


def test_write_spec_direct_construction_defaults_provider_options(tmp_path) -> None:
    parsed = _write_spec(tmp_path)

    spec = WeightSnapshotWriteSpec(
        parsed.catalog_path,
        parsed.destination,
        parsed.endpoint,
    )

    assert spec.provider_options == {}
    assert spec.mooncake_store == {}


def test_threaded_call_timeout_completion_cleanup_runs_once() -> None:
    from sglang.srt.weight_transfer._threaded_call import (
        _BoundedExecutor,
        _ThreadedCall,
        _ThreadedCallState,
    )

    executor = _BoundedExecutor(
        max_workers=1,
        thread_name_prefix="test-weight-transfer",
    )
    call = _ThreadedCall(executor)
    factory_ready = threading.Barrier(2)
    release_factory = threading.Event()
    cleanup_started = threading.Event()
    release_cleanup = threading.Event()
    cleanup_calls = 0

    def factory():
        factory_ready.wait(timeout=2)
        release_factory.wait(timeout=2)
        return "finished"

    def cleanup():
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_started.set()
        release_cleanup.wait(timeout=2)

    call.start(
        factory,
        thread_name="test-timeout-cleanup",
        cleanup_on_abandon=cleanup,
    )
    factory_ready.wait(timeout=2)

    with pytest.raises(RuntimeError, match="timed out"):
        call.result_before(
            store_runtime.WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + 0.02
            ),
            interrupted=lambda: RuntimeError("timed out"),
        )
    assert call.state is _ThreadedCallState.ABANDONED

    release_factory.set()
    assert cleanup_started.wait(timeout=2)
    with pytest.raises(RuntimeError, match="timed out again"):
        call.result_before(
            store_runtime.WeightTransferExecutionContext(
                deadline_unix_sec=time.time() - 1
            ),
            interrupted=lambda: RuntimeError("timed out again"),
        )

    release_cleanup.set()
    assert call.done.wait(timeout=2)
    assert cleanup_calls == 1
    assert call.state is _ThreadedCallState.COMPLETED
    assert call.was_abandoned is True
    executor.seal()


def test_threaded_call_publishes_done_after_completion_callback() -> None:
    from sglang.srt.weight_transfer._threaded_call import (
        _BoundedExecutor,
        _ThreadedCall,
        _ThreadedCallState,
    )

    executor = _BoundedExecutor(
        max_workers=1,
        thread_name_prefix="test-weight-transfer",
    )
    callback_started = threading.Event()
    release_callback = threading.Event()
    call = _ThreadedCall(executor)

    def after_done():
        callback_started.set()
        release_callback.wait(timeout=2)

    call.start(
        lambda: "finished",
        thread_name="test-completion-callback",
        after_done=after_done,
    )

    assert callback_started.wait(timeout=1)
    assert call.state is _ThreadedCallState.COMPLETED
    assert not call.done.is_set()
    release_callback.set()
    assert call.done.wait(timeout=1)
    assert call.result == "finished"
    executor.seal()


def test_registered_write_backend_is_provider_neutral(
    tmp_path,
) -> None:
    spec = _write_spec(
        tmp_path,
        destination={
            "provider": "checkpoint-test",
            "storage_id": "model/revision",
            "object_prefix": "objects",
        },
        provider_options={"bucket": "weights"},
    )
    calls = []

    @contextmanager
    def factory(received_spec, **kwargs):
        calls.append((received_spec, kwargs))
        yield WeightSnapshotBackend(
            provider=SimpleNamespace(name="checkpoint-test"),
            catalog=SimpleNamespace(),
            endpoint="checkpoint://writer",
        )

    with _isolated_write_backend_registry():
        register_weight_snapshot_write_backend("checkpoint-test", factory)
        with open_weight_snapshot_write_backend(
            spec,
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: "checksum",
            coordinator=LocalWeightStoreDistributedCoordinator(),
        ) as backend:
            assert backend.endpoint == "checkpoint://writer"

    assert calls[0][0] is spec
    assert calls[0][1]["local_placement_ids"] == ("placement-0",)


def test_backend_default_lifecycle_is_provider_neutral() -> None:
    backend = WeightSnapshotBackend(
        provider=SimpleNamespace(name="checkpoint-test"),
        catalog=SimpleNamespace(),
        endpoint="checkpoint://writer",
    )

    assert backend.seal() == ()
    assert backend.quiesce(timeout_ms=0) == WeightSnapshotBackendStatus(terminal=True)
    assert backend.close(timeout_ms=0) == WeightSnapshotBackendStatus(
        terminal=True,
        closed=True,
    )


@pytest.mark.parametrize("initial_timeout_ms", [0, 20])
def test_backend_close_retains_ticket_until_lifecycle_call_drains(
    initial_timeout_ms,
) -> None:
    close_started = threading.Event()
    release_close = threading.Event()

    class Provider:
        def seal_native_calls_for_close(self):
            return ()

        def drain_pending_calls(self, *, timeout_ms):
            assert timeout_ms >= 0
            return ()

    class Store:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1
            close_started.set()
            release_close.wait(timeout=2)

    provider = Provider()
    store = Store()
    backend = WeightSnapshotBackend(
        provider=provider,
        catalog=SimpleNamespace(),
        endpoint="store://writer",
        lifecycle=store_runtime._MooncakeStoreBackendLifecycle(
            store,
            provider,
        ),
    )

    pending = backend.close(timeout_ms=initial_timeout_ms)

    assert close_started.wait(timeout=1)
    assert pending == WeightSnapshotBackendStatus(
        terminal=False,
        pending_tickets=("mooncake-store/close",),
    )
    assert store.close_calls == 1

    release_close.set()
    assert backend.close(timeout_ms=1000) == WeightSnapshotBackendStatus(
        terminal=True,
        closed=True,
    )
    assert store.close_calls == 1


def test_backend_close_retries_after_admission_rejection(monkeypatch) -> None:
    from sglang.srt.weight_transfer._threaded_call import (
        _BoundedExecutor,
        _ThreadedCall,
        _ThreadedCallAdmissionError,
    )

    executor = _BoundedExecutor(
        max_workers=1,
        thread_name_prefix="test-store-close",
    )
    monkeypatch.setattr(store_runtime, "_STORE_LIFECYCLE_EXECUTOR", executor)
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    blocker = _ThreadedCall(executor)
    blocker.start(
        lambda: (blocker_started.set(), release_blocker.wait(timeout=2)),
        thread_name="test-store-close-blocker",
    )
    assert blocker_started.wait(timeout=1)

    class Provider:
        def seal_native_calls_for_close(self):
            return ()

        def drain_pending_calls(self, *, timeout_ms):
            assert timeout_ms >= 0
            return ()

    class Store:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    store = Store()
    backend = WeightSnapshotBackend(
        provider=Provider(),
        catalog=SimpleNamespace(),
        endpoint="store://writer",
        lifecycle=store_runtime._MooncakeStoreBackendLifecycle(
            store,
            Provider(),
        ),
    )

    with pytest.raises(_ThreadedCallAdmissionError):
        backend.close(timeout_ms=100)
    assert store.close_calls == 0

    release_blocker.set()
    assert blocker.done.wait(timeout=1)
    assert backend.close(timeout_ms=1000).closed is True
    assert store.close_calls == 1
    executor.seal()


def test_write_backend_rejects_duplicate_builtin_registration() -> None:
    @contextmanager
    def factory(_spec, **_kwargs):
        yield

    with _isolated_write_backend_registry():
        with pytest.raises(ValueError, match="already registered"):
            register_weight_snapshot_write_backend("mooncake-store", factory)


def test_write_backend_replace_overrides_builtin(
    tmp_path,
) -> None:
    calls = []

    @contextmanager
    def factory(
        spec,
        *,
        local_placement_ids,
        payload_checksum_verifier,
        coordinator,
    ):
        calls.append(
            (
                spec,
                local_placement_ids,
                payload_checksum_verifier,
                coordinator,
            )
        )
        yield WeightSnapshotBackend(
            provider=SimpleNamespace(name="mooncake-store"),
            catalog=SimpleNamespace(),
            endpoint="override://writer",
        )

    coordinator = LocalWeightStoreDistributedCoordinator()

    def verifier(_location):
        return "checksum"

    with _isolated_write_backend_registry():
        register_weight_snapshot_write_backend(
            "mooncake-store",
            factory,
            replace=True,
        )

        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=verifier,
            coordinator=coordinator,
        ) as backend:
            assert backend.endpoint == "override://writer"

    assert calls == [
        (
            _write_spec(tmp_path),
            ("placement-0",),
            verifier,
            coordinator,
        )
    ]


def test_write_backend_registry_restores_builtin_and_cleans_extensions() -> None:
    registry = store_runtime._WEIGHT_SNAPSHOT_WRITE_BACKENDS
    builtin_factory = registry["mooncake-store"]

    @contextmanager
    def replacement(_spec, **_kwargs):
        yield

    with _isolated_write_backend_registry():
        register_weight_snapshot_write_backend(
            "mooncake-store",
            replacement,
            replace=True,
        )
        register_weight_snapshot_write_backend("checkpoint-test", replacement)
        assert registry["mooncake-store"] is replacement
        assert registry["checkpoint-test"] is replacement

    assert registry["mooncake-store"] is builtin_factory
    assert "checkpoint-test" not in registry


def test_write_backend_rejects_unknown_provider(tmp_path) -> None:
    spec = _write_spec(
        tmp_path,
        destination={
            "provider": "unknown",
            "storage_id": "model/revision",
            "object_prefix": "objects",
        },
        provider_options={},
    )

    with pytest.raises(
        ValueError,
        match="no weight snapshot writer registered for 'unknown'",
    ):
        with open_weight_snapshot_write_backend(
            spec,
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: "checksum",
            coordinator=LocalWeightStoreDistributedCoordinator(),
        ):
            pass


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("source_pre_registered", 1, "source_pre_registered"),
        ("max_range_bytes", 0, "max_range_bytes"),
        ("max_ranges_per_request", True, "max_ranges_per_request"),
        ("namespace", "", "namespace"),
        ("key_prefix", 3, "key_prefix"),
    ],
)
def test_write_spec_rejects_invalid_store_options(
    tmp_path,
    option: str,
    value: Any,
    message: str,
) -> None:
    base = _write_spec(tmp_path).provider_options
    options = dict(base)
    options[option] = value

    with pytest.raises(ValueError, match=message):
        _write_spec(tmp_path, provider_options=options)


def test_default_write_backend_uses_root_catalog_and_configured_provider(
    monkeypatch,
    tmp_path,
) -> None:
    stores, weight_stores = _install_fake_mooncake(monkeypatch)
    coordinator = LocalWeightStoreDistributedCoordinator()

    def verifier(_location):
        return f"sha256:{'b' * 64}"

    with open_weight_snapshot_write_backend(
        _write_spec(tmp_path),
        local_placement_ids=("placement-0",),
        payload_checksum_verifier=verifier,
        coordinator=coordinator,
    ) as backend:
        assert isinstance(backend.provider, MooncakeWeightStoreProvider)
        assert isinstance(backend.catalog, RootWeightStorageCatalog)
        assert backend.endpoint == "source-worker"
        assert backend.provider.namespace == "serving"
        assert backend.provider.local_placement_ids == frozenset({"placement-0"})
        assert backend.provider.payload_checksum_verifier is verifier
        assert backend.provider.coordinator is coordinator
        assert backend.provider.source_pre_registered is True
        assert backend.provider.max_total_operations == 31
        assert stores[0].closed is False

    assert stores[0].setup_options == {
        "local_hostname": "source-worker",
        "metadata_server": "http://metadata",
        "protocol": "tcp",
    }
    assert stores[0].closed is True
    assert weight_stores[0].options == {
        "key_prefix": "model-weights",
        "max_range_bytes": 4096,
        "max_ranges_per_request": 7,
        "max_region_segments": 29,
    }


def test_write_backend_store_lifetime_is_owned_by_outer_exit_stack(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(monkeypatch)
    with ExitStack() as owner:
        backend = owner.enter_context(
            open_weight_snapshot_write_backend(
                _write_spec(tmp_path),
                local_placement_ids=("placement-0",),
                payload_checksum_verifier=lambda _location: f"sha256:{'b' * 64}",
                coordinator=LocalWeightStoreDistributedCoordinator(),
            )
        )

        assert isinstance(backend.provider, MooncakeWeightStoreProvider)
        assert stores[0].closed is False

    assert stores[0].closed is True


def test_write_backend_setup_and_close_remain_synchronous(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(monkeypatch)
    lifecycle_thread = threading.get_ident()

    with open_weight_snapshot_write_backend(
        _write_spec(tmp_path),
        local_placement_ids=("placement-0",),
        payload_checksum_verifier=lambda _location: f"sha256:{'b' * 64}",
        coordinator=LocalWeightStoreDistributedCoordinator(),
    ):
        assert stores[0].setup_thread == lifecycle_thread

    assert stores[0].close_thread == lifecycle_thread


def test_write_backend_probe_failure_uses_terminal_context_for_close(
    monkeypatch,
    tmp_path,
) -> None:
    close_started = threading.Event()
    release_close = threading.Event()
    _install_fake_mooncake(
        monkeypatch,
        close_started=close_started,
        release_close=release_close,
    )
    close_contexts = []
    close_store = store_runtime._close_store

    def record_close(store, *, backend, execution_context):
        close_contexts.append(execution_context)
        return close_store(
            store,
            backend=backend,
            execution_context=execution_context,
        )

    monkeypatch.setattr(store_runtime, "_close_store", record_close)
    context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 0.05,
    )
    release_timer = threading.Timer(0.3, release_close.set)
    release_timer.start()
    started = time.monotonic()
    elapsed = None

    def fail_probe(_request):
        time.sleep(0.06)
        raise RuntimeError("probe failed")

    try:
        with pytest.raises(RuntimeError, match="probe failed"):
            with open_weight_snapshot_write_backend(
                _write_spec(tmp_path),
                local_placement_ids=("placement-0",),
                payload_checksum_verifier=lambda _location: f"sha256:{'b' * 64}",
                coordinator=LocalWeightStoreDistributedCoordinator(),
                execution_context=context,
            ) as backend:
                monkeypatch.setattr(
                    backend.provider,
                    "probe",
                    fail_probe,
                )
                backend.provider.probe(object())
    finally:
        elapsed = time.monotonic() - started
        release_close.set()
        release_timer.cancel()
        release_timer.join(timeout=1)

    assert close_started.wait(timeout=1)
    assert len(close_contexts) == 1
    close_context = close_contexts[0]
    assert close_context is not context
    assert close_context.deadline_unix_sec > time.time()
    assert close_context.deadline_unix_sec <= (
        time.time() + store_runtime._STORE_TERMINAL_CONTROL_TIMEOUT_SEC
    )
    assert elapsed >= 0.25


def test_write_backend_refuses_close_with_pending_native_call(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(monkeypatch)
    release_call = threading.Event()
    native_call_started = threading.Event()
    provider = None
    started = time.monotonic()
    try:
        with pytest.raises(WeightStoreDistributedError) as raised:
            with open_weight_snapshot_write_backend(
                _write_spec(tmp_path),
                local_placement_ids=("placement-0",),
                payload_checksum_verifier=lambda _location: f"sha256:{'b' * 64}",
                coordinator=LocalWeightStoreDistributedCoordinator(),
            ) as backend:
                provider = backend.provider
                call = provider._get_or_start_native_call(
                    "materialize-1",
                    "upload",
                    lambda: (
                        native_call_started.set(),
                        release_call.wait(timeout=5),
                    )[-1],
                )
                assert native_call_started.wait(timeout=1)
                assert call.thread is not None
                assert call.thread.daemon is False
                assert call.owner is provider
                assert len(provider.pending_native_calls()) == 1
    finally:
        release_call.set()

    assert time.monotonic() - started < 0.5
    assert raised.value.phase == "close_write_backend"
    assert "pending calls" in str(raised.value)
    assert stores[0].closed is False
    assert provider is not None
    drained = provider.drain_pending_calls(timeout_ms=1000)
    assert len(drained) == 1
    assert drained[0].state.value == "succeeded"
    assert call.owner is None


def test_write_backend_public_lifecycle_seals_quiesces_and_closes(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(monkeypatch)
    release_call = threading.Event()

    with open_weight_snapshot_write_backend(
        _write_spec(tmp_path),
        local_placement_ids=("placement-0",),
        payload_checksum_verifier=lambda _location: f"sha256:{'b' * 64}",
        coordinator=LocalWeightStoreDistributedCoordinator(),
    ) as backend:
        backend.provider._get_or_start_native_call(
            "materialize-1",
            "upload",
            lambda: release_call.wait(timeout=2),
        )

        assert backend.seal() == ("materialize-1/upload",)
        pending = backend.quiesce(timeout_ms=0)
        assert pending == WeightSnapshotBackendStatus(
            terminal=False,
            pending_tickets=("materialize-1/upload",),
        )
        assert stores[0].closed is False

        release_call.set()
        closed = backend.close(timeout_ms=1000)
        assert closed == WeightSnapshotBackendStatus(
            terminal=True,
            closed=True,
        )
        assert stores[0].closed is True

    assert stores[0].closed is True


def test_write_backend_does_not_construct_file_catalog_on_non_root(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(monkeypatch)
    catalog_calls: list[str] = []
    coordinator = _SequencedCoordinator(rank=1)

    monkeypatch.setattr(
        "sglang.srt.weight_transfer.store_runtime.FileWeightStorageCatalog",
        lambda path: catalog_calls.append(path),
    )

    with open_weight_snapshot_write_backend(
        _write_spec(tmp_path),
        local_placement_ids=("placement-1",),
        payload_checksum_verifier=lambda _location: f"sha256:{'c' * 64}",
        coordinator=coordinator,
    ) as backend:
        assert isinstance(backend.catalog, RootWeightStorageCatalog)

    assert catalog_calls == []
    assert stores[0].setup_options["local_hostname"] == "source-worker-rank-1"
    assert coordinator.root_calls == [("catalog.initialize", True)]
    assert coordinator.exchange_calls == 2


def test_write_backend_fails_all_ranks_on_remote_setup_failure_and_closes(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(monkeypatch)
    coordinator = _SequencedCoordinator(
        remote_setup_error="RuntimeError: rank 1 setup failed",
    )

    with pytest.raises(WeightStoreDistributedError) as raised:
        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: f"sha256:{'c' * 64}",
            coordinator=coordinator,
        ):
            pass

    assert raised.value.phase == "initialize_write_backend"
    assert "rank 1 setup failed" in str(raised.value)
    assert stores[0].closed is True
    assert coordinator.root_calls == []
    assert coordinator.exchange_calls == 2


def test_write_backend_setup_failure_uses_aligned_terminal_cleanup_context(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(monkeypatch)
    coordinator = _SequencedCoordinator(
        remote_setup_error="RuntimeError: rank 1 setup failed",
    )
    close_contexts = []
    close_store = store_runtime._close_store

    def record_close(store, *, execution_context):
        close_contexts.append(execution_context)
        return close_store(
            store,
            execution_context=execution_context,
        )

    monkeypatch.setattr(store_runtime, "_close_store", record_close)
    business_context = WeightTransferExecutionContext(deadline_unix_sec=time.time() + 1)

    with pytest.raises(WeightStoreDistributedError):
        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: f"sha256:{'c' * 64}",
            coordinator=coordinator,
            execution_context=business_context,
        ):
            pass

    assert stores[0].closed is True
    assert coordinator.exchange_contexts[0] is business_context
    assert len(close_contexts) == 1
    cleanup_context = close_contexts[0]
    assert cleanup_context is coordinator.exchange_contexts[1]
    assert cleanup_context is not business_context
    assert cleanup_context.deadline_unix_sec > time.time()
    assert cleanup_context.deadline_unix_sec <= (
        time.time() + store_runtime._STORE_TERMINAL_CONTROL_TIMEOUT_SEC
    )


def test_write_backend_preserves_unknown_remote_setup_completion(
    monkeypatch,
    tmp_path,
) -> None:
    _install_fake_mooncake(monkeypatch)
    coordinator = _SequencedCoordinator(
        remote_setup_error="setup timed out",
        remote_setup_completion_unknown=True,
    )

    with pytest.raises(WeightStoreDistributedError) as raised:
        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: f"sha256:{'c' * 64}",
            coordinator=coordinator,
        ):
            pass

    assert raised.value.phase == "initialize_write_backend"
    assert raised.value.completion_unknown is True


def test_write_backend_broadcasts_remote_close_failure(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(monkeypatch)
    coordinator = _SequencedCoordinator(
        remote_close_error="RuntimeError: rank 1 close failed",
    )

    with pytest.raises(WeightStoreDistributedError) as raised:
        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: f"sha256:{'c' * 64}",
            coordinator=coordinator,
        ):
            pass

    assert raised.value.phase == "close_write_backend"
    assert "rank 1 close failed" in str(raised.value)
    assert stores[0].closed is True
    assert coordinator.exchange_calls == 2


def test_write_backend_preserves_unknown_remote_close_completion(
    monkeypatch,
    tmp_path,
) -> None:
    _install_fake_mooncake(monkeypatch)
    coordinator = _SequencedCoordinator(
        remote_close_error="close timed out",
        remote_close_completion_unknown=True,
    )

    with pytest.raises(WeightStoreDistributedError) as raised:
        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: f"sha256:{'c' * 64}",
            coordinator=coordinator,
        ):
            pass

    assert raised.value.phase == "close_write_backend"
    assert raised.value.completion_unknown is True


def test_write_backend_broadcasts_catalog_initialization_failure_and_closes(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(monkeypatch)

    def fail_catalog(_path):
        raise RuntimeError("catalog open failed")

    monkeypatch.setattr(
        "sglang.srt.weight_transfer.store_runtime.FileWeightStorageCatalog",
        fail_catalog,
    )

    with pytest.raises(RuntimeError, match="catalog open failed"):
        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: f"sha256:{'c' * 64}",
            coordinator=LocalWeightStoreDistributedCoordinator(),
        ):
            pass

    assert stores[0].closed is True


@pytest.mark.parametrize(
    "setup_failure",
    [RuntimeError("setup raised"), 9],
)
def test_write_backend_closes_store_when_setup_fails(
    monkeypatch,
    tmp_path,
    setup_failure: BaseException | int,
) -> None:
    options = (
        {"setup_error": setup_failure}
        if isinstance(setup_failure, BaseException)
        else {"setup_result": setup_failure}
    )
    stores, _ = _install_fake_mooncake(monkeypatch, **options)

    with pytest.raises(RuntimeError, match="setup"):
        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: f"sha256:{'d' * 64}",
            coordinator=LocalWeightStoreDistributedCoordinator(),
        ):
            pass

    assert stores[0].closed is True


def test_write_backend_merges_setup_and_cleanup_close_failures(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(
        monkeypatch,
        setup_error=RuntimeError("setup raised"),
        close_error=RuntimeError("cleanup close failed"),
    )

    with pytest.raises(WeightStoreDistributedError) as raised:
        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: f"sha256:{'d' * 64}",
            coordinator=LocalWeightStoreDistributedCoordinator(),
        ):
            pass

    assert raised.value.phase == "initialize_write_backend"
    assert "setup raised" in str(raised.value)
    assert "cleanup close failed" in str(raised.value)
    assert raised.value.completion_unknown is False
    assert stores[0].closed is True


def test_write_backend_marks_unresolved_cleanup_close_unknown(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        store_runtime,
        "_STORE_TERMINAL_CONTROL_TIMEOUT_SEC",
        0.05,
    )
    close_started = threading.Event()
    release_close = threading.Event()
    stores, _ = _install_fake_mooncake(
        monkeypatch,
        setup_error=RuntimeError("setup raised"),
        close_started=close_started,
        release_close=release_close,
    )
    spec = _write_spec(tmp_path)
    context = store_runtime.WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 0.05
    )
    try:
        with pytest.raises(WeightStoreDistributedError) as raised:
            with open_weight_snapshot_write_backend(
                spec,
                local_placement_ids=("placement-0",),
                payload_checksum_verifier=lambda _location: f"sha256:{'d' * 64}",
                coordinator=LocalWeightStoreDistributedCoordinator(),
                execution_context=context,
            ):
                pass
    finally:
        release_close.set()

    assert close_started.wait(timeout=1)
    assert raised.value.phase == "initialize_write_backend"
    assert "setup raised" in str(raised.value)
    assert raised.value.completion_unknown is True
    assert stores[0].closed is True


def test_write_backend_merges_setup_exchange_and_unresolved_close_failure(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        store_runtime,
        "_STORE_TERMINAL_CONTROL_TIMEOUT_SEC",
        0.05,
    )
    close_started = threading.Event()
    release_close = threading.Event()
    stores, _ = _install_fake_mooncake(
        monkeypatch,
        close_started=close_started,
        release_close=release_close,
    )

    class FailingSetupExchangeCoordinator(_SequencedCoordinator):
        def exchange_preflight_outcome(self, outcome, **kwargs):
            del outcome, kwargs
            raise RuntimeError("setup exchange failed")

    context = store_runtime.WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 0.05
    )
    try:
        with pytest.raises(WeightStoreDistributedError) as raised:
            with open_weight_snapshot_write_backend(
                _write_spec(tmp_path),
                local_placement_ids=("placement-0",),
                payload_checksum_verifier=lambda _location: f"sha256:{'d' * 64}",
                coordinator=FailingSetupExchangeCoordinator(),
                execution_context=context,
            ):
                pass
    finally:
        release_close.set()

    assert close_started.wait(timeout=1)
    assert raised.value.phase == "initialize_write_backend"
    assert "setup exchange failed" in str(raised.value)
    assert "cleanup" in str(raised.value)
    assert raised.value.completion_unknown is True
    assert stores[0].closed is True


def test_write_backend_close_error_does_not_mask_body_error(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(
        monkeypatch,
        close_error=RuntimeError("close failed"),
    )

    with pytest.raises(ValueError, match="body failed"):
        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: f"sha256:{'e' * 64}",
            coordinator=LocalWeightStoreDistributedCoordinator(),
        ):
            raise ValueError("body failed")

    assert stores[0].closed is True


def test_write_backend_surfaces_close_error_without_primary_failure(
    monkeypatch,
    tmp_path,
) -> None:
    _install_fake_mooncake(
        monkeypatch,
        close_error=RuntimeError("close failed"),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        with open_weight_snapshot_write_backend(
            _write_spec(tmp_path),
            local_placement_ids=("placement-0",),
            payload_checksum_verifier=lambda _location: f"sha256:{'f' * 64}",
            coordinator=LocalWeightStoreDistributedCoordinator(),
        ):
            pass


def test_default_mooncake_backend_uses_configured_store_and_closes(
    monkeypatch,
    tmp_path,
) -> None:
    stores = []
    weight_stores = []

    class FakeDistributedStore:
        def __init__(self):
            self.setup_options = None
            self.closed = False
            stores.append(self)

        def setup(self, options):
            self.setup_options = options
            return 0

        def close(self):
            self.closed = True

    class FakeWeightStore:
        def __init__(self, store, **options):
            self.store = store
            self.options = options
            self.max_ranges_per_request = options["max_ranges_per_request"]
            self.max_region_segments = options["max_region_segments"]
            weight_stores.append(self)

    mooncake_package = types.ModuleType("mooncake")
    mooncake_package.__path__ = []
    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = FakeDistributedStore
    transfer_module = types.ModuleType("mooncake.weight_transfer")
    transfer_module.WeightStore = FakeWeightStore
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", transfer_module)

    with open_weight_snapshot_backend(_spec(tmp_path)) as backend:
        assert isinstance(backend.provider, MooncakeWeightStoreProvider)
        assert backend.endpoint == "target-worker"
        assert backend.provider.namespace == "serving"
        assert backend.provider.target_pre_registered is True
        assert backend.provider.max_total_operations == 31
        assert stores[0].closed is False

    assert stores[0].setup_options == {
        "local_hostname": "target-worker",
        "metadata_server": "http://metadata",
        "protocol": "tcp",
    }
    assert stores[0].closed is True
    assert weight_stores[0].options == {
        "key_prefix": "model-weights",
        "max_range_bytes": 4096,
        "max_ranges_per_request": 7,
        "max_region_segments": 29,
    }


def test_default_mooncake_backend_qualifies_multi_rank_store_hostname(
    monkeypatch,
    tmp_path,
) -> None:
    stores, _ = _install_fake_mooncake(monkeypatch)

    with open_weight_snapshot_backend(
        _spec(tmp_path),
        rank=1,
        world_size=2,
    ):
        pass

    assert stores[0].setup_options["local_hostname"] == "target-worker-rank-1"


def test_read_backend_uses_caller_deadline_for_setup_and_close(
    monkeypatch,
    tmp_path,
) -> None:
    _install_fake_mooncake(monkeypatch)
    contexts = []
    run_call = store_runtime._run_store_lifecycle_call
    close_store = store_runtime._close_store

    def record_call(phase, store, factory, execution_context, **kwargs):
        contexts.append((phase, execution_context))
        return run_call(
            phase,
            store,
            factory,
            execution_context,
            **kwargs,
        )

    def record_close(store, *, backend, execution_context):
        contexts.append(("close", execution_context))
        return close_store(
            store,
            backend=backend,
            execution_context=execution_context,
        )

    monkeypatch.setattr(store_runtime, "_run_store_lifecycle_call", record_call)
    monkeypatch.setattr(store_runtime, "_close_store", record_close)
    context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 30,
    )

    with open_weight_snapshot_backend(
        _spec(tmp_path),
        execution_context=context,
    ):
        pass

    assert contexts[0] == ("setup", context)
    assert contexts[1][0] == "close"
    assert contexts[1][1] is not context
    assert contexts[1][1].deadline_unix_sec > time.time()


@pytest.mark.parametrize(
    "setup_failure",
    [RuntimeError("reader setup raised"), 9],
)
def test_read_backend_closes_store_when_setup_fails(
    monkeypatch,
    tmp_path,
    setup_failure: BaseException | int,
) -> None:
    options = (
        {"setup_error": setup_failure}
        if isinstance(setup_failure, BaseException)
        else {"setup_result": setup_failure}
    )
    stores, _ = _install_fake_mooncake(monkeypatch, **options)

    with pytest.raises(RuntimeError, match="setup"):
        with open_weight_snapshot_backend(_spec(tmp_path)):
            pass

    assert stores[0].closed is True


def test_read_backend_marks_blocked_setup_cleanup_unknown(
    monkeypatch,
    tmp_path,
) -> None:
    setup_started = threading.Event()
    release_setup = threading.Event()
    stores, _ = _install_fake_mooncake(
        monkeypatch,
        setup_started=setup_started,
        release_setup=release_setup,
    )
    spec = replace(_spec(tmp_path), load_timeout_sec=0.05)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError) as raised:
            with open_weight_snapshot_backend(spec):
                pass
    finally:
        release_setup.set()

    assert setup_started.wait(timeout=1)
    assert time.monotonic() - started < 0.5
    assert raised.value.completion_unknown is True
    deadline = time.monotonic() + 1
    while not stores[0].closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stores[0].closed is True


def test_default_backend_rejects_non_mooncake_provider_before_setup(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="no default weight snapshot backend"):
        with open_weight_snapshot_backend(_spec(tmp_path, provider="checkpoint")):
            pass


def test_default_backend_closes_store_when_provider_construction_fails(
    monkeypatch,
    tmp_path,
) -> None:
    store_state = {"closed": False}

    class FakeDistributedStore:
        def setup(self, _options):
            return 0

        def close(self):
            store_state["closed"] = True

    class FailingWeightStore:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("weight store construction failed")

    mooncake_package = types.ModuleType("mooncake")
    mooncake_package.__path__ = []
    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = FakeDistributedStore
    transfer_module = types.ModuleType("mooncake.weight_transfer")
    transfer_module.WeightStore = FailingWeightStore
    monkeypatch.setitem(sys.modules, "mooncake", mooncake_package)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    monkeypatch.setitem(sys.modules, "mooncake.weight_transfer", transfer_module)

    with pytest.raises(RuntimeError, match="weight store construction failed"):
        with open_weight_snapshot_backend(_spec(tmp_path)):
            pass
    assert store_state["closed"] is True


if __name__ == "__main__":
    pytest_args = ["-x" if argument == "-f" else argument for argument in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *pytest_args]))
