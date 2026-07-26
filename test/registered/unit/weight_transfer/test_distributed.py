from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from importlib import import_module
from typing import Any
from unittest.mock import call, patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MODULE_NAME = "sglang.srt.weight_transfer.distributed"
try:
    distributed = import_module(_MODULE_NAME)
except ModuleNotFoundError as error:
    if error.name != _MODULE_NAME:
        raise
    distributed = None
    _IMPORT_ERROR: ModuleNotFoundError | None = error
else:
    _IMPORT_ERROR = None

requires_distributed = pytest.mark.skipif(
    distributed is None,
    reason="distributed coordinator is not implemented",
)


@dataclass
class _BroadcastBus:
    payloads: list[Any] = field(default_factory=list)


class _FakeDistributed:
    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        group: Any,
        bus: _BroadcastBus,
        initialized: bool = True,
        gathered: list[Any] | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.bus = bus
        self.initialized = initialized
        self.gathered = gathered
        self.broadcast_index = 0
        self.broadcast_calls: list[tuple[int | None, int | None, Any]] = []
        self.gather_inputs: list[Any] = []

    def is_initialized(self) -> bool:
        return self.initialized

    def get_rank(self, group: Any = None) -> int:
        assert group is self.group
        return self.rank

    def get_world_size(self, group: Any = None) -> int:
        assert group is self.group
        return self.world_size

    def broadcast_object_list(
        self,
        object_list: list[Any],
        src: int | None = None,
        group: Any = None,
        *,
        group_src: int | None = None,
    ) -> None:
        assert group is self.group
        assert (src is None) != (group_src is None)
        source_rank = group_src if group_src is not None else src
        assert source_rank == 0
        index = self.broadcast_index
        if self.rank == 0:
            assert index == len(self.bus.payloads)
            self.bus.payloads.append(object_list[0])
        else:
            object_list[0] = self.bus.payloads[index]
        self.broadcast_index += 1
        self.broadcast_calls.append((src, group_src, group))

    def all_gather_object(
        self,
        object_list: list[Any],
        value: Any,
        group: Any = None,
    ) -> None:
        assert group is self.group
        assert self.gathered is not None
        self.gather_inputs.append(value)
        object_list[:] = self.gathered


def _require_module() -> Any:
    assert distributed is not None
    return distributed


def _make_torch_coordinators(
    *,
    world_size: int,
    gathered: list[Any] | None = None,
) -> tuple[list[Any], list[_FakeDistributed], _BroadcastBus, Any]:
    module = _require_module()
    group = object()
    bus = _BroadcastBus()
    backends = [
        _FakeDistributed(
            rank=rank,
            world_size=world_size,
            group=group,
            bus=bus,
            gathered=gathered,
        )
        for rank in range(world_size)
    ]
    with patch("importlib.import_module", side_effect=backends) as loader:
        coordinators = [
            module.TorchDistributedWeightStoreCoordinator(group=group)
            for _ in range(world_size)
        ]
    assert loader.call_args_list == [
        call("torch.distributed") for _ in range(world_size)
    ]
    return coordinators, backends, bus, group


def _factory_for_rank(
    rank: int,
    calls: list[int],
    result: Any,
):
    def factory() -> Any:
        calls.append(rank)
        if rank != 0:
            raise AssertionError("non-root factory was called")
        return result

    return factory


def test_distributed_module_is_available() -> None:
    assert _IMPORT_ERROR is None


@requires_distributed
def test_error_preserves_phase_and_message() -> None:
    module = _require_module()

    error = module.WeightStoreDistributedError("prepare_upload", "failed")

    assert error.phase == "prepare_upload"
    assert str(error) == "failed"


@requires_distributed
def test_upload_outcome_is_frozen_and_preserves_metadata() -> None:
    module = _require_module()
    receipt = object()
    outcome = module.WeightStoreUploadOutcome(
        rank=2,
        placement_ids=("placement-b", "placement-a"),
        receipts=(receipt,),
        error=None,
    )

    assert outcome.rank == 2
    assert outcome.placement_ids == ("placement-b", "placement-a")
    assert outcome.receipts == (receipt,)
    assert outcome.error is None
    with pytest.raises(FrozenInstanceError):
        outcome.rank = 3


@requires_distributed
def test_preflight_outcome_is_frozen_and_preserves_metadata() -> None:
    module = _require_module()
    outcome = module.WeightStorePreflightOutcome(
        rank=2,
        error="ValueError: local plan mismatch",
    )

    assert outcome.rank == 2
    assert outcome.error == "ValueError: local plan mismatch"
    with pytest.raises(FrozenInstanceError):
        outcome.rank = 3


@requires_distributed
@pytest.mark.parametrize("rank", [-1, True, 1.5, "1"])
def test_upload_outcome_rejects_invalid_rank(rank: Any) -> None:
    module = _require_module()

    with pytest.raises(ValueError, match="rank"):
        module.WeightStoreUploadOutcome(
            rank=rank,
            placement_ids=(),
            receipts=(),
            error=None,
        )


@requires_distributed
@pytest.mark.parametrize(
    "placement_ids",
    [
        ("",),
        (1,),
        ("placement-a", "placement-a"),
    ],
)
def test_upload_outcome_rejects_invalid_or_duplicate_placement_ids(
    placement_ids: tuple[Any, ...],
) -> None:
    module = _require_module()

    with pytest.raises(ValueError, match="placement"):
        module.WeightStoreUploadOutcome(
            rank=0,
            placement_ids=placement_ids,
            receipts=(),
            error=None,
        )


@requires_distributed
@pytest.mark.parametrize("error", ["", 1])
def test_upload_outcome_rejects_invalid_error(error: Any) -> None:
    module = _require_module()

    with pytest.raises(ValueError, match="error"):
        module.WeightStoreUploadOutcome(
            rank=0,
            placement_ids=(),
            receipts=(),
            error=error,
        )


@requires_distributed
def test_local_coordinator_executes_every_factory_and_exchanges_outcome() -> None:
    module = _require_module()
    coordinator = module.LocalWeightStoreDistributedCoordinator()
    calls: list[str] = []
    plan = object()
    manifest = object()
    outcome = module.WeightStoreUploadOutcome(
        rank=0,
        placement_ids=("placement-0",),
        receipts=("receipt-0",),
        error=None,
    )
    preflight = module.WeightStorePreflightOutcome(rank=0, error=None)

    assert coordinator.rank == 0
    assert coordinator.world_size == 1
    assert (
        coordinator.prepare_upload(lambda: calls.append("prepare_upload") or plan)
        is plan
    )
    assert coordinator.exchange_preflight_outcome(preflight) == (preflight,)
    assert coordinator.exchange_upload_outcome(outcome) == (outcome,)
    assert (
        coordinator.commit_upload(lambda: calls.append("commit_upload") or manifest)
        is manifest
    )
    assert coordinator.abort_upload(lambda: calls.append("abort_upload")) is None
    assert coordinator.finalize_upload(lambda: calls.append("finalize_upload")) is None
    assert calls == [
        "prepare_upload",
        "commit_upload",
        "abort_upload",
        "finalize_upload",
    ]


@requires_distributed
def test_local_coordinator_run_root_supports_custom_phases_and_discard() -> None:
    module = _require_module()
    coordinator = module.LocalWeightStoreDistributedCoordinator()
    calls: list[str] = []

    assert coordinator.run_root(
        "catalog.lookup",
        lambda: calls.append("lookup") or {"value": 1},
    ) == {"value": 1}
    assert (
        coordinator.run_root(
            "catalog.cleanup",
            lambda: calls.append("cleanup") or object(),
            discard_result=True,
        )
        is None
    )
    assert calls == ["lookup", "cleanup"]


@requires_distributed
@pytest.mark.parametrize("phase", ["", None, 1, True])
def test_local_coordinator_run_root_rejects_invalid_phase(phase: Any) -> None:
    module = _require_module()
    coordinator = module.LocalWeightStoreDistributedCoordinator()
    called = False

    def factory() -> None:
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="phase"):
        coordinator.run_root(phase, factory)

    assert called is False


@requires_distributed
@pytest.mark.parametrize(
    "method_name",
    [
        "prepare_upload",
        "commit_upload",
        "abort_upload",
        "finalize_upload",
    ],
)
def test_local_coordinator_wraps_factory_errors(method_name: str) -> None:
    module = _require_module()
    coordinator = module.LocalWeightStoreDistributedCoordinator()

    def fail() -> None:
        raise RuntimeError("root failed")

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        getattr(coordinator, method_name)(fail)

    assert raised.value.phase == method_name
    assert "root failed" in str(raised.value)


@requires_distributed
def test_torch_coordinator_requires_initialized_distributed() -> None:
    module = _require_module()
    group = object()
    backend = _FakeDistributed(
        rank=0,
        world_size=1,
        group=group,
        bus=_BroadcastBus(),
        initialized=False,
    )

    with patch("importlib.import_module", return_value=backend):
        with pytest.raises(
            module.WeightStoreDistributedError,
            match="initialized",
        ) as raised:
            module.TorchDistributedWeightStoreCoordinator(group=group)

    assert raised.value.phase == "initialize"


@requires_distributed
def test_torch_coordinator_broadcasts_prepare_and_commit_results_from_root() -> None:
    plan = {"kind": "upload-plan"}
    manifest = {"kind": "committed-manifest"}
    coordinators, backends, _, group = _make_torch_coordinators(world_size=3)
    prepare_calls: list[int] = []
    commit_calls: list[int] = []

    prepare_results = [
        coordinator.prepare_upload(_factory_for_rank(rank, prepare_calls, plan))
        for rank, coordinator in enumerate(coordinators)
    ]
    commit_results = [
        coordinator.commit_upload(_factory_for_rank(rank, commit_calls, manifest))
        for rank, coordinator in enumerate(coordinators)
    ]

    assert prepare_results == [plan, plan, plan]
    assert commit_results == [manifest, manifest, manifest]
    assert prepare_calls == [0]
    assert commit_calls == [0]
    assert [coordinator.rank for coordinator in coordinators] == [0, 1, 2]
    assert all(coordinator.world_size == 3 for coordinator in coordinators)
    assert all(
        backend.broadcast_calls == [(None, 0, group), (None, 0, group)]
        for backend in backends
    )


@requires_distributed
def test_torch_coordinator_run_root_broadcasts_custom_result_once() -> None:
    result = {"kind": "catalog-result"}
    coordinators, backends, bus, group = _make_torch_coordinators(world_size=3)
    calls: list[int] = []

    results = [
        coordinator.run_root(
            "catalog.custom",
            _factory_for_rank(rank, calls, result),
        )
        for rank, coordinator in enumerate(coordinators)
    ]

    assert results == [result, result, result]
    assert calls == [0]
    assert len(bus.payloads) == 1
    assert bus.payloads[0].phase == "catalog.custom"
    assert all(backend.broadcast_calls == [(None, 0, group)] for backend in backends)


@requires_distributed
def test_root_catalog_runs_every_catalog_method_on_root_and_broadcasts() -> None:
    module = _require_module()
    coordinators, _, bus, _ = _make_torch_coordinators(world_size=2)
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class RecordingCatalog:
        def __getattr__(self, name: str):
            def method(*args: Any, **kwargs: Any) -> Any:
                calls.append((name, args, kwargs))
                return {
                    "method": name,
                    "args": args,
                    "kwargs": kwargs,
                }

            return method

    root_catalog = module.RootWeightStorageCatalog(
        RecordingCatalog(),
        coordinators[0],
    )
    non_root_catalog = module.RootWeightStorageCatalog(None, coordinators[1])
    invocations = [
        ("begin_materialization", ("materialization", "intent"), {}),
        (
            "complete_materialization",
            ("materialization", "snapshot"),
            {},
        ),
        ("abort_materialization", ("materialization",), {}),
        (
            "set_materialization_completion_ticket",
            ("materialization", "ticket"),
            {},
        ),
        ("get_materialization", ("materialization",), {}),
        ("recoverable_materializations", (), {}),
        ("prepare_publish", ("publication", "snapshot"), {}),
        ("publish", ("publication",), {}),
        ("abort", ("publication",), {}),
        ("get_snapshot", ("ref",), {}),
        ("get_publication", ("publication",), {}),
        ("recoverable_publications", (), {}),
        ("get_revision_head", ("model", "revision"), {}),
        (
            "compare_and_set_revision",
            (),
            {
                "model_id": "model",
                "revision": "revision",
                "expected": "old-head",
                "new_ref": "new-ref",
                "new_state": "ready",
            },
        ),
    ]

    for method_name, args, kwargs in invocations:
        root_result = getattr(root_catalog, method_name)(*args, **kwargs)
        non_root_result = getattr(non_root_catalog, method_name)(*args, **kwargs)
        assert non_root_result == root_result

    expected_names = [method_name for method_name, _, _ in invocations]
    assert [name for name, _, _ in calls] == expected_names
    phases = [payload.phase for payload in bus.payloads]
    assert phases == [f"catalog.{name}" for name in expected_names]
    assert len(phases) == len(set(phases))


@requires_distributed
def test_root_catalog_requires_catalog_only_on_root() -> None:
    module = _require_module()
    coordinators, _, _, _ = _make_torch_coordinators(world_size=2)

    with pytest.raises(ValueError, match="root catalog"):
        module.RootWeightStorageCatalog(None, coordinators[0])

    module.RootWeightStorageCatalog(None, coordinators[1])


@requires_distributed
def test_torch_coordinator_runs_abort_and_finalize_only_on_root() -> None:
    coordinators, backends, _, group = _make_torch_coordinators(world_size=3)

    for method_name in ("abort_upload", "finalize_upload"):
        calls: list[int] = []
        results = [
            getattr(coordinator, method_name)(_factory_for_rank(rank, calls, object()))
            for rank, coordinator in enumerate(coordinators)
        ]
        assert results == [None, None, None]
        assert calls == [0]

    assert all(
        backend.broadcast_calls == [(None, 0, group), (None, 0, group)]
        for backend in backends
    )


@requires_distributed
def test_torch_coordinator_returns_complete_outcomes_sorted_by_rank() -> None:
    module = _require_module()
    outcomes = [
        module.WeightStoreUploadOutcome(
            rank=rank,
            placement_ids=(f"placement-{rank}",),
            receipts=(f"receipt-{rank}",),
            error=None,
        )
        for rank in range(3)
    ]
    gathered = [outcomes[2], outcomes[0], outcomes[1]]
    coordinators, backends, _, _ = _make_torch_coordinators(
        world_size=3,
        gathered=gathered,
    )

    results = [
        coordinator.exchange_upload_outcome(outcomes[rank])
        for rank, coordinator in enumerate(coordinators)
    ]

    assert results == [tuple(outcomes), tuple(outcomes), tuple(outcomes)]
    assert [backend.gather_inputs for backend in backends] == [
        [outcomes[0]],
        [outcomes[1]],
        [outcomes[2]],
    ]
    assert all(
        isinstance(value, module.WeightStoreUploadOutcome)
        for backend in backends
        for value in backend.gather_inputs
    )


@requires_distributed
def test_torch_coordinator_returns_complete_preflight_outcomes_sorted_by_rank() -> None:
    module = _require_module()
    outcomes = [
        module.WeightStorePreflightOutcome(
            rank=rank,
            error=None if rank != 1 else "ValueError: invalid local plan",
        )
        for rank in range(3)
    ]
    gathered = [outcomes[2], outcomes[0], outcomes[1]]
    coordinators, backends, _, _ = _make_torch_coordinators(
        world_size=3,
        gathered=gathered,
    )

    results = [
        coordinator.exchange_preflight_outcome(outcomes[rank])
        for rank, coordinator in enumerate(coordinators)
    ]

    assert results == [tuple(outcomes), tuple(outcomes), tuple(outcomes)]
    assert [backend.gather_inputs for backend in backends] == [
        [outcomes[0]],
        [outcomes[1]],
        [outcomes[2]],
    ]


@requires_distributed
@pytest.mark.parametrize(
    "case",
    ["unknown", "duplicate-rank", "duplicate-placement"],
)
def test_torch_coordinator_fails_closed_on_invalid_gathered_outcomes(
    case: str,
) -> None:
    module = _require_module()
    first = module.WeightStoreUploadOutcome(
        rank=0,
        placement_ids=("placement-0",),
        receipts=(),
        error=None,
    )
    if case == "unknown":
        gathered = [first, object()]
    elif case == "duplicate-rank":
        gathered = [first, first]
    else:
        gathered = [
            first,
            module.WeightStoreUploadOutcome(
                rank=1,
                placement_ids=("placement-0",),
                receipts=(),
                error=None,
            ),
        ]
    coordinators, _, _, _ = _make_torch_coordinators(
        world_size=2,
        gathered=gathered,
    )

    errors = []
    for rank, coordinator in enumerate(coordinators):
        local = first
        if rank == 1:
            local = module.WeightStoreUploadOutcome(
                rank=1,
                placement_ids=("local-1",),
                receipts=(),
                error=None,
            )
        with pytest.raises(module.WeightStoreDistributedError) as raised:
            coordinator.exchange_upload_outcome(local)
        errors.append(raised.value)

    assert all(error.phase == "exchange_upload_outcome" for error in errors)
    assert len({str(error) for error in errors}) == 1


@requires_distributed
@pytest.mark.parametrize(
    "method_name",
    [
        "prepare_upload",
        "commit_upload",
        "abort_upload",
        "finalize_upload",
    ],
)
def test_root_factory_error_is_broadcast_consistently_to_every_rank(
    method_name: str,
) -> None:
    module = _require_module()
    coordinators, _, _, _ = _make_torch_coordinators(world_size=3)
    calls: list[int] = []

    def factory_for_rank(rank: int):
        def factory() -> None:
            calls.append(rank)
            if rank != 0:
                raise AssertionError("non-root factory was called")
            raise ValueError("root failed")

        return factory

    errors = []
    for rank, coordinator in enumerate(coordinators):
        with pytest.raises(module.WeightStoreDistributedError) as raised:
            getattr(coordinator, method_name)(factory_for_rank(rank))
        errors.append(raised.value)

    assert calls == [0]
    assert all(error.phase == method_name for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert "root failed" in str(errors[0])


@requires_distributed
def test_torch_coordinator_fails_closed_on_unknown_broadcast_structure() -> None:
    module = _require_module()
    group = object()
    bus = _BroadcastBus(payloads=[object()])
    backend = _FakeDistributed(
        rank=1,
        world_size=2,
        group=group,
        bus=bus,
    )
    with patch("importlib.import_module", return_value=backend):
        coordinator = module.TorchDistributedWeightStoreCoordinator(group=group)

    def non_root_factory() -> None:
        raise AssertionError("non-root factory was called")

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        coordinator.prepare_upload(non_root_factory)

    assert raised.value.phase == "prepare_upload"
    assert "invalid" in str(raised.value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
