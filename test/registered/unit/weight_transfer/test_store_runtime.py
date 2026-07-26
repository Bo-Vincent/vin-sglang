from __future__ import annotations

import sys
import types
from contextlib import ExitStack
from typing import Any

import pytest

from sglang.srt.weight_transfer.distributed import (
    LocalWeightStoreDistributedCoordinator,
    RootWeightStorageCatalog,
    WeightStoreDistributedError,
    WeightStorePreflightOutcome,
)
from sglang.srt.weight_transfer.mooncake_store import MooncakeWeightStoreProvider
from sglang.srt.weight_transfer.store_runtime import (
    WeightSnapshotLoadSpec,
    WeightSnapshotWriteSpec,
    open_weight_snapshot_backend,
    open_weight_snapshot_write_backend,
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
        "mooncake_store": {
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
):
    stores = []
    weight_stores = []

    class FakeDistributedStore:
        def __init__(self):
            self.setup_options = None
            self.closed = False
            stores.append(self)

        def setup(self, options):
            self.setup_options = options
            if setup_error is not None:
                raise setup_error
            return setup_result

        def close(self):
            self.closed = True
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


class _SequencedCoordinator:
    world_size = 2

    def __init__(
        self,
        *,
        rank: int = 0,
        remote_setup_error: str | None = None,
        remote_close_error: str | None = None,
    ) -> None:
        self.rank = rank
        self.remote_setup_error = remote_setup_error
        self.remote_close_error = remote_close_error
        self.exchange_calls = 0
        self.root_calls: list[tuple[str, bool]] = []

    def exchange_preflight_outcome(self, outcome):
        remote_error = (
            self.remote_setup_error
            if self.exchange_calls == 0
            else self.remote_close_error
        )
        self.exchange_calls += 1
        remote = WeightStorePreflightOutcome(
            rank=1 - self.rank,
            error=remote_error,
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
    assert spec.mooncake_store["source_pre_registered"] is True


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
                "mooncake_store": {
                    "setup": {"local_hostname": "source"},
                    "target_pre_registered": True,
                }
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


def test_write_spec_rejects_non_mooncake_destination(tmp_path) -> None:
    with pytest.raises(ValueError, match="destination.provider"):
        _write_spec(
            tmp_path,
            destination={
                "provider": "checkpoint",
                "storage_id": "model/revision",
                "object_prefix": "objects",
            },
        )


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
    base = _write_spec(tmp_path).mooncake_store
    options = dict(base)
    options[option] = value

    with pytest.raises(ValueError, match=message):
        _write_spec(tmp_path, mooncake_store=options)


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
    assert coordinator.exchange_calls == 1


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
