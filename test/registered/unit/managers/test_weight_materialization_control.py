import asyncio
from types import SimpleNamespace

import pytest

from sglang.srt.managers.io_struct import (
    CommitWeightMaterializationReqInput,
    CommitWeightMaterializationReqOutput,
    MaterializeWeightsReqInput,
    PrepareWeightMaterializationReqInput,
    PrepareWeightMaterializationReqOutput,
)
from sglang.srt.managers.tokenizer_control_mixin import (
    TokenizerControlMixin,
    WeightMaterializationError,
    _COMMUNICATOR_SPECS,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ReaderLock:
    def __init__(self):
        self.depth = 0
        self.entries = 0

    @property
    def reader_lock(self):
        owner = self

        class _Context:
            async def __aenter__(self):
                owner.depth += 1
                owner.entries += 1

            async def __aexit__(self, exc_type, exc, traceback):
                owner.depth -= 1

        return _Context()

    @property
    def writer_lock(self):
        raise AssertionError("materialization must not take the writer lock")


def _manager(prepare, commit, *, dp_size=2, enabled=True):
    lock = _ReaderLock()
    manager = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_weight_runtime_manifest=enabled,
            model_path="Qwen/Qwen3-0.6B",
            revision="main",
            dp_size=dp_size,
        ),
        model_update_lock=lock,
        prepare_weight_materialization_communicator=prepare,
        commit_weight_materialization_communicator=commit,
        auto_create_handle_loop=lambda: None,
    )
    return manager, lock


def _prepare_output(
    rank,
    *,
    success=True,
    message="prepared",
    generation=7,
    digest="sha256:logical",
    total_bytes=4096,
    state="prepared",
    materialization_id="materialize-1",
):
    return PrepareWeightMaterializationReqOutput(
        materialization_id=materialization_id,
        success=success,
        message=message,
        external_dp_rank=rank,
        generation=generation if success else None,
        logical_payload_digest=digest if success else None,
        total_bytes=total_bytes if success else None,
        session_state=state,
    )


def _commit_output(
    rank,
    *,
    selected=False,
    ref=None,
    success=True,
    message="committed",
    state="published",
    materialization_id="materialize-1",
    completion_unknown=False,
    completion_ticket=None,
):
    return CommitWeightMaterializationReqOutput(
        materialization_id=materialization_id,
        success=success,
        message=message,
        external_dp_rank=rank,
        selected=selected,
        ref=ref,
        completion_unknown=completion_unknown,
        completion_ticket=completion_ticket,
        session_state=state,
    )


def _request(**overrides):
    values = {
        "storage_options": {"backend": "mooncake", "namespace": "weights"},
        "materialization_id": "materialize-1",
    }
    values.update(overrides)
    return MaterializeWeightsReqInput(**values)


def test_registers_correlated_prepare_and_commit_communicators() -> None:
    specs = {spec[0]: spec for spec in _COMMUNICATOR_SPECS}

    assert specs["prepare_weight_materialization"][2:] == (
        "queueing",
        "materialization_id",
    )
    assert specs["commit_weight_materialization"][2:] == (
        "queueing",
        "materialization_id",
    )


def test_out_of_order_prepare_selects_lowest_rank_and_holds_reader_lock() -> None:
    requests = []
    manager = None

    async def prepare(request):
        requests.append(request)
        assert manager.model_update_lock.depth == 1
        return [_prepare_output(1), _prepare_output(0)]

    async def commit(request):
        requests.append(request)
        assert manager.model_update_lock.depth == 1
        return [
            _commit_output(1),
            _commit_output(
                0,
                selected=True,
                ref={"namespace": "weights", "snapshot_id": "snapshot-1"},
            ),
        ]

    manager, lock = _manager(prepare, commit)
    result = asyncio.run(
        TokenizerControlMixin.materialize_weights(manager, _request(), None)
    )

    assert result == {
        "materialization_id": "materialize-1",
        "ref": {"namespace": "weights", "snapshot_id": "snapshot-1"},
        "selected_external_dp_rank": 0,
        "total_bytes": 4096,
    }
    assert lock.entries == 1
    assert lock.depth == 0
    assert isinstance(requests[0], PrepareWeightMaterializationReqInput)
    assert requests[0].model_id == "Qwen/Qwen3-0.6B"
    assert requests[0].revision == "main"
    assert isinstance(requests[1], CommitWeightMaterializationReqInput)
    assert requests[1].selected_external_dp_rank == 0


def test_uses_explicit_source_external_dp_rank() -> None:
    commits = []

    async def prepare(_request):
        return [_prepare_output(0), _prepare_output(1)]

    async def commit(request):
        commits.append(request)
        return [
            _commit_output(0),
            _commit_output(
                1,
                selected=True,
                ref={"snapshot_id": "snapshot-1"},
            ),
        ]

    manager, _ = _manager(prepare, commit)
    result = asyncio.run(
        TokenizerControlMixin.materialize_weights(
            manager,
            _request(source_external_dp_rank=1),
            None,
        )
    )

    assert result["selected_external_dp_rank"] == 1
    assert commits[0].selected_external_dp_rank == 1


def test_prepare_failure_triggers_cleanup_without_returning_a_ref() -> None:
    commits = []

    async def prepare(_request):
        return [
            _prepare_output(0),
            _prepare_output(
                1,
                success=False,
                message="capture failed",
                state="failed",
            ),
        ]

    async def commit(request):
        commits.append(request)
        assert request.selected_external_dp_rank is None
        return [
            _commit_output(0, state="released"),
            _commit_output(1, state="released"),
        ]

    manager, _ = _manager(prepare, commit)
    with pytest.raises(WeightMaterializationError, match="capture failed") as raised:
        asyncio.run(
            TokenizerControlMixin.materialize_weights(manager, _request(), None)
        )

    assert raised.value.materialization_id == "materialize-1"
    assert len(commits) == 1
    assert commits[0].selected_external_dp_rank is None


@pytest.mark.parametrize(
    "second",
    [
        _prepare_output(0),
        _prepare_output(1, generation=8),
        _prepare_output(1, digest="sha256:other"),
        _prepare_output(1, total_bytes=8192),
    ],
)
def test_duplicate_or_inconsistent_prepare_results_fail_closed(second) -> None:
    commits = []

    async def prepare(_request):
        return [_prepare_output(0), second]

    async def commit(request):
        commits.append(request)
        return [
            _commit_output(0, state="released"),
            _commit_output(1, state="released"),
        ]

    manager, _ = _manager(prepare, commit)
    with pytest.raises(WeightMaterializationError):
        asyncio.run(
            TokenizerControlMixin.materialize_weights(manager, _request(), None)
        )

    assert len(commits) == 1
    assert commits[0].selected_external_dp_rank is None


def test_multiple_commit_refs_fail_closed_then_cleanup() -> None:
    commits = []

    async def prepare(_request):
        return [_prepare_output(0), _prepare_output(1)]

    async def commit(request):
        commits.append(request)
        if request.selected_external_dp_rank is None:
            return [
                _commit_output(0, state="released"),
                _commit_output(1, state="released"),
            ]
        return [
            _commit_output(
                0,
                selected=True,
                ref={"snapshot_id": "snapshot-1"},
            ),
            _commit_output(1, ref={"snapshot_id": "snapshot-1"}),
        ]

    manager, _ = _manager(prepare, commit)
    with pytest.raises(WeightMaterializationError, match="exactly one") as raised:
        asyncio.run(
            TokenizerControlMixin.materialize_weights(manager, _request(), None)
        )

    assert raised.value.materialization_id == "materialize-1"
    assert [request.selected_external_dp_rank for request in commits] == [0, None]


def test_cancellation_waits_for_cleanup() -> None:
    prepare_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    commits = []

    async def prepare(_request):
        prepare_started.set()
        await asyncio.Future()

    async def commit(request):
        commits.append(request)
        await asyncio.sleep(0)
        cleanup_finished.set()
        return [
            _commit_output(0, state="released"),
            _commit_output(1, state="released"),
        ]

    manager, _ = _manager(prepare, commit)

    async def scenario():
        task = asyncio.create_task(
            TokenizerControlMixin.materialize_weights(manager, _request(), None)
        )
        await prepare_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cleanup_finished.is_set()

    asyncio.run(scenario())
    assert len(commits) == 1
    assert commits[0].selected_external_dp_rank is None


@pytest.mark.parametrize(
    "obj",
    [
        _request(materialization_id=""),
        _request(source_external_dp_rank=-1),
        _request(source_external_dp_rank=True),
        _request(source_external_dp_rank=2),
    ],
)
def test_rejects_invalid_id_or_source_rank_before_fanout(obj) -> None:
    async def unexpected(_request):
        raise AssertionError("invalid request must not fan out")

    manager, _ = _manager(unexpected, unexpected)
    with pytest.raises(ValueError):
        asyncio.run(TokenizerControlMixin.materialize_weights(manager, obj, None))


def test_requires_runtime_manifest_feature_flag() -> None:
    async def unexpected(_request):
        raise AssertionError("disabled feature must not fan out")

    manager, _ = _manager(unexpected, unexpected, enabled=False)
    with pytest.raises(WeightMaterializationError, match="enable-weight-runtime"):
        asyncio.run(
            TokenizerControlMixin.materialize_weights(manager, _request(), None)
        )
