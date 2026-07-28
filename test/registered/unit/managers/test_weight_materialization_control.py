import asyncio
import contextvars
import sys
from types import SimpleNamespace

import fastapi
import msgspec
import pytest
from sglang.srt.managers import io_struct as io_struct_module
from sglang.srt.managers.communicator import (
    FanOutCancelledBeforeDispatch,
    FanOutDeadlineExpiredBeforeDispatch,
)
from sglang.srt.managers.io_struct import (
    CommitWeightMaterializationReqInput,
    CommitWeightMaterializationReqOutput,
    MaterializeWeightsReqInput,
    PrepareWeightMaterializationReqInput,
    PrepareWeightMaterializationReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    WeightMaterializationSessionState,
    WeightSnapshotActivationReqInput,
    WeightSnapshotActivationReqOutput,
    weight_update_request_context,
)
from sglang.srt.managers.tokenizer_control_mixin import (
    _COMMUNICATOR_SPECS,
    TokenizerControlMixin,
    WeightMaterializationError,
    _call_weight_update_communicator,
    _validate_next_weight_revision,
)
from sglang.srt.managers.weight_materialization import (
    is_published_materialization_state,
    is_retryable_materialization_state,
    is_terminal_materialization_state,
    reduce_materialization_states,
)
from sglang.srt.utils.aio_rwlock import RWLock
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

    @property
    def snapshot_reader_lock(self):
        return self.reader_lock


class _WriterLock:
    def __init__(self):
        self.depth = 0

    @property
    def writer_lock(self):
        owner = self

        class _Context:
            async def __aenter__(self):
                owner.depth += 1

            async def __aexit__(self, exc_type, exc, traceback):
                owner.depth -= 1

        return _Context()


def _manager(prepare, commit, *, dp_size=2, enabled=True):
    lock = _ReaderLock()
    manager = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_weight_runtime_manifest=enabled,
            model_path="Qwen/Qwen3-0.6B",
            revision="main",
            weight_version="weights-v1",
            dp_size=dp_size,
        ),
        model_update_lock=lock,
        runtime_weight_revision="weights-v1",
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
    request_id="request-1",
):
    return PrepareWeightMaterializationReqOutput(
        materialization_id=materialization_id,
        request_id=request_id,
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
    request_id="request-1",
    phase="commit",
):
    return CommitWeightMaterializationReqOutput(
        materialization_id=materialization_id,
        request_id=request_id,
        success=success,
        message=message,
        external_dp_rank=rank,
        selected=selected,
        ref=ref,
        completion_unknown=completion_unknown,
        completion_ticket=completion_ticket,
        session_state=state,
        phase=phase,
    )


def _request(**overrides):
    values = {
        "storage_options": {"backend": "mooncake", "namespace": "weights"},
        "materialization_id": "materialize-1",
    }
    values.update(overrides)
    return MaterializeWeightsReqInput(**values)


def test_materialization_session_state_classification_preserves_wire_values() -> None:
    state = WeightMaterializationSessionState

    assert state.PREPARED.value == "prepared"
    assert state.COMPLETION_UNKNOWN.value == "completion_unknown"
    assert state.FINALIZE_PENDING.value == "finalize_pending"
    assert state.PUBLISHED.value == "published"
    assert state.PUBLISHED_CLEANUP_PENDING.value == "published_cleanup_pending"
    assert state.PUBLISHED_CLEANUP_FAILED.value == "published_cleanup_failed"

    assert is_retryable_materialization_state(state.CLEANUP_PENDING)
    assert is_retryable_materialization_state(state.COMPLETION_UNKNOWN)
    assert is_retryable_materialization_state(state.FINALIZE_PENDING)
    assert is_retryable_materialization_state(state.PUBLISHED_CLEANUP_PENDING)
    assert not is_retryable_materialization_state(state.PUBLISHED_CLEANUP_FAILED)
    assert is_terminal_materialization_state(state.PUBLISHED)
    assert is_terminal_materialization_state(state.PUBLISHED_CLEANUP_FAILED)
    assert is_published_materialization_state(state.PUBLISHED)
    assert is_published_materialization_state(state.PUBLISHED_CLEANUP_PENDING)
    assert not is_published_materialization_state(state.FINALIZE_PENDING)


def test_materialization_session_state_reducer_is_fail_closed() -> None:
    state = WeightMaterializationSessionState
    reduce_states = reduce_materialization_states

    assert (
        reduce_states(
            ("published", "released", "skipped"),
            default=state.FAILED,
        )
        == state.PUBLISHED
    )
    assert (
        reduce_states(
            ("failed", "conflict"),
            default=state.FAILED,
        )
        == state.CONFLICT
    )
    assert (
        reduce_states(
            ("finalize_pending", "skipped"),
            default=state.FAILED,
        )
        == state.FINALIZE_PENDING
    )
    assert (
        reduce_states(
            ("published",),
            default=state.FAILED,
            completion_unknown=True,
        )
        == state.COMPLETION_UNKNOWN
    )
    assert (
        reduce_states(
            ("future_state",),
            default=state.FAILED,
        )
        == state.FAILED
    )
    assert (
        reduce_states(
            ("published", "future_state"),
            default=state.FAILED,
        )
        == state.FAILED
    )
    assert (
        reduce_states(
            ("published", "unknown"),
            default=state.FAILED,
        )
        == state.FAILED
    )
    assert (
        reduce_states(
            ("committing",),
            default=state.FAILED,
        )
        == state.FAILED
    )


def test_runtime_manifests_require_a_new_weight_revision() -> None:
    manager, _ = _manager(None, None)

    with pytest.raises(ValueError, match="require weight_version"):
        _validate_next_weight_revision(manager, None)
    with pytest.raises(ValueError, match="new weight artifact"):
        _validate_next_weight_revision(manager, "weights-v1")

    _validate_next_weight_revision(manager, "weights-v2")
    manager.server_args.enable_weight_runtime_manifest = False
    _validate_next_weight_revision(manager, None)


def test_runtime_revision_does_not_follow_a_metadata_only_version_change() -> None:
    manager, _ = _manager(None, None)

    manager.server_args.weight_version = "metadata-only"

    with pytest.raises(ValueError, match="new weight artifact"):
        _validate_next_weight_revision(manager, "weights-v1")
    _validate_next_weight_revision(manager, "metadata-only")


def test_weight_revision_is_published_inside_writer_transaction() -> None:
    async def scenario():
        lock = _WriterLock()
        revisions = []

        async def update(_request):
            assert lock.depth == 1
            return [
                UpdateWeightsFromDistributedReqOutput(
                    success=True,
                    message="updated",
                )
            ]

        def publish_revision(revision):
            assert lock.depth == 1
            revisions.append(revision)
            manager.runtime_weight_revision = revision

        manager = SimpleNamespace(
            server_args=SimpleNamespace(
                dp_size=1,
                enable_dp_attention=False,
                enable_weight_runtime_manifest=True,
            ),
            runtime_weight_revision="weights-v1",
            weight_update_fail_closed=False,
            model_update_lock=lock,
            is_pause_cond=asyncio.Condition(),
            is_pause=False,
            update_weights_from_distributed_communicator=update,
            _require_single_tokenizer_weight_update_owner=lambda: None,
            auto_create_handle_loop=lambda: None,
            abort_request=lambda **_kwargs: None,
            _update_weight_version_if_provided=publish_revision,
        )
        result = await TokenizerControlMixin.update_weights_from_distributed(
            manager,
            UpdateWeightsFromDistributedReqInput(
                names=["weight"],
                dtypes=["float16"],
                shapes=[[1]],
                weight_version="weights-v2",
            ),
        )

        assert result[0] is True
        assert revisions == ["weights-v2"]
        assert manager.runtime_weight_revision == "weights-v2"
        assert lock.depth == 0

    asyncio.run(scenario())


def test_concurrent_updates_cannot_publish_the_same_revision_twice() -> None:
    async def scenario():
        updates = []

        async def update(_request):
            updates.append("updated")
            await asyncio.sleep(0)
            return [
                UpdateWeightsFromDistributedReqOutput(
                    success=True,
                    message="updated",
                )
            ]

        manager = SimpleNamespace(
            server_args=SimpleNamespace(
                dp_size=1,
                enable_dp_attention=False,
                enable_weight_runtime_manifest=True,
            ),
            runtime_weight_revision="weights-v1",
            weight_update_fail_closed=False,
            model_update_lock=RWLock(),
            is_pause_cond=asyncio.Condition(),
            is_pause=False,
            update_weights_from_distributed_communicator=update,
            _require_single_tokenizer_weight_update_owner=lambda: None,
            auto_create_handle_loop=lambda: None,
            abort_request=lambda **_kwargs: None,
        )

        def publish_revision(revision):
            manager.runtime_weight_revision = revision

        manager._update_weight_version_if_provided = publish_revision
        request = UpdateWeightsFromDistributedReqInput(
            names=["weight"],
            dtypes=["float16"],
            shapes=[[1]],
            weight_version="weights-v2",
        )
        results = await asyncio.gather(
            TokenizerControlMixin.update_weights_from_distributed(manager, request),
            TokenizerControlMixin.update_weights_from_distributed(manager, request),
            return_exceptions=True,
        )

        assert len(updates) == 1
        assert sum(isinstance(result, ValueError) for result in results) == 1
        assert manager.runtime_weight_revision == "weights-v2"

    asyncio.run(scenario())


def test_not_dispatched_update_does_not_fail_closed() -> None:
    manager = SimpleNamespace(weight_update_fail_closed=False)

    async def communicator(_request):
        raise FanOutCancelledBeforeDispatch

    with pytest.raises(FanOutCancelledBeforeDispatch):
        asyncio.run(_call_weight_update_communicator(manager, communicator, object()))

    assert manager.weight_update_fail_closed is False


def test_online_weight_update_retry_gets_a_new_request_id() -> None:
    manager = SimpleNamespace(weight_update_fail_closed=False)
    request = UpdateWeightsFromDistributedReqInput(
        names=["weight"],
        dtypes=["float16"],
        shapes=[[1]],
    )
    request_ids = []

    async def communicator(actual):
        request_ids.append(actual.request_id)
        return [
            UpdateWeightsFromDistributedReqOutput(
                success=True,
                message="updated",
                request_id=actual.request_id,
                responder_id="scheduler-0",
            )
        ]

    asyncio.run(_call_weight_update_communicator(manager, communicator, request))
    asyncio.run(_call_weight_update_communicator(manager, communicator, request))

    assert all(request_ids)
    assert request_ids[0] != request_ids[1]


@pytest.mark.parametrize(
    ("request_factory", "output_type"),
    [
        (
            lambda: UpdateWeightsFromDistributedReqInput(
                names=["weight"],
                dtypes=["float16"],
                shapes=[[1]],
                request_id="distributed-request",
            ),
            UpdateWeightsFromDistributedReqOutput,
        ),
        (
            lambda: UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[b"tensor"],
                request_id="tensor-request",
            ),
            UpdateWeightsFromTensorReqOutput,
        ),
        (
            lambda: UpdateWeightsFromIPCReqInput(
                zmq_handles={"gpu-0": "ipc://weights"},
                request_id="ipc-request",
            ),
            UpdateWeightsFromIPCReqOutput,
        ),
    ],
)
def test_online_weight_update_ack_inherits_request_and_responder(
    request_factory,
    output_type,
) -> None:
    def tokenizer_encode():
        request_input = request_factory()
        return (
            type(request_input),
            request_input.request_id,
            msgspec.msgpack.encode(request_input),
        )

    request_type, request_id, payload = contextvars.Context().run(tokenizer_encode)

    def scheduler_round_trip():
        decoded = msgspec.msgpack.decode(payload, type=request_type)
        with weight_update_request_context(decoded):
            output = output_type(success=True, message="updated")
        return decoded, output

    decoded, output = contextvars.Context().run(scheduler_round_trip)

    assert decoded.request_id == request_id
    assert output.request_id == decoded.request_id
    assert output.responder_id


def test_online_weight_update_responder_identity_refreshes_after_fork(
    monkeypatch,
) -> None:
    monkeypatch.setattr(io_struct_module.os, "getpid", lambda: 101)
    parent = io_struct_module._stable_control_responder_id()
    monkeypatch.setattr(io_struct_module.os, "getpid", lambda: 202)
    child = io_struct_module._stable_control_responder_id()

    assert parent.endswith(":101")
    assert child.endswith(":202")
    assert parent != child


def test_legacy_multi_tokenizer_online_update_is_rejected() -> None:
    manager = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_weight_runtime_manifest=False,
            tokenizer_worker_num=2,
        )
    )

    with pytest.raises(fastapi.HTTPException) as raised:
        TokenizerControlMixin._require_single_tokenizer_weight_update_owner(manager)

    assert raised.value.status_code == 409
    assert "single tokenizer worker" in raised.value.detail


def test_materialization_rejects_fail_closed_source_under_reader_lock() -> None:
    async def unexpected(*_args, **_kwargs):
        raise AssertionError("materialization communicator was called")

    manager, lock = _manager(unexpected, unexpected)
    manager.weight_update_fail_closed = True

    with pytest.raises(RuntimeError, match="snapshot export is disabled"):
        asyncio.run(
            TokenizerControlMixin.materialize_weights(manager, _request(), None)
        )

    assert lock.entries == 1
    assert lock.depth == 0


def test_snapshot_reader_keeps_inference_serving_without_starving_writer() -> None:
    async def exercise() -> None:
        lock = RWLock()
        snapshot_entered = asyncio.Event()
        release_snapshot = asyncio.Event()
        inference_entered = asyncio.Event()
        release_inference = asyncio.Event()
        writer_entered = asyncio.Event()
        release_writer = asyncio.Event()
        queued_snapshot_entered = asyncio.Event()

        async def hold_snapshot() -> None:
            async with lock.snapshot_reader_lock:
                snapshot_entered.set()
                await release_snapshot.wait()

        async def hold_inference() -> None:
            async with lock.reader_lock:
                inference_entered.set()
                await release_inference.wait()

        async def hold_writer() -> None:
            async with lock.writer_lock:
                writer_entered.set()
                await release_writer.wait()

        async def queue_snapshot() -> None:
            async with lock.snapshot_reader_lock:
                queued_snapshot_entered.set()

        snapshot = asyncio.create_task(hold_snapshot())
        await asyncio.wait_for(snapshot_entered.wait(), timeout=1)
        writer = asyncio.create_task(hold_writer())
        while lock._waiting_writers == 0:
            await asyncio.sleep(0)
        inference = asyncio.create_task(hold_inference())
        await asyncio.wait_for(inference_entered.wait(), timeout=1)

        queued_snapshot = asyncio.create_task(queue_snapshot())
        await asyncio.sleep(0)
        assert not queued_snapshot_entered.is_set()

        release_inference.set()
        release_snapshot.set()
        await asyncio.wait_for(writer_entered.wait(), timeout=1)
        assert not queued_snapshot_entered.is_set()

        release_writer.set()
        await asyncio.wait_for(queued_snapshot_entered.wait(), timeout=1)
        await asyncio.gather(snapshot, inference, writer, queued_snapshot)

    asyncio.run(exercise())


def test_registers_request_correlated_weight_control_communicators() -> None:
    specs = {spec[0]: spec for spec in _COMMUNICATOR_SPECS}

    for name in (
        "update_weights_from_distributed",
        "update_weights_from_tensor",
        "update_weights_from_ipc",
        "weight_snapshot_activation",
    ):
        assert specs[name][2:] == (
            "queueing",
            "request_id",
            "responder_id",
        )
    assert specs["begin_remote_instance_weight_transfer"][2:] == (
        "queueing",
        "request_id",
        "external_dp_rank",
    )
    assert specs["release_remote_instance_weight_transfer"][2:] == (
        "queueing",
        "request_id",
        "external_dp_rank",
    )
    assert specs["renew_remote_instance_weight_transfer"][2:] == (
        "queueing",
        "request_id",
        "external_dp_rank",
    )
    assert specs["prepare_weight_materialization"][2:] == (
        "queueing",
        "request_id",
        "external_dp_rank",
    )
    assert specs["commit_weight_materialization"][2:] == (
        "queueing",
        "request_id",
        "external_dp_rank",
    )


def _activation_output(
    request: WeightSnapshotActivationReqInput,
    responder_id: str,
    *,
    success: bool = True,
    state: str,
    message: str = "Success.",
) -> WeightSnapshotActivationReqOutput:
    return WeightSnapshotActivationReqOutput(
        action=request.action,
        phase=request.phase,
        transaction_id=request.transaction_id,
        request_id=request.request_id,
        responder_id=responder_id,
        state=state,
        success=success,
        message=message,
    )


def test_weight_snapshot_activation_reconciles_partial_commit() -> None:
    requests = []

    async def activate(request, **_kwargs):
        requests.append(request)
        if request.phase == "prepare":
            return [
                _activation_output(request, "scheduler-0", state="prepared"),
                _activation_output(request, "scheduler-1", state="prepared"),
            ]
        if request.phase == "commit":
            return [
                _activation_output(request, "scheduler-0", state="serving"),
                _activation_output(
                    request,
                    "scheduler-1",
                    success=False,
                    state="commit_unknown",
                    message="ACK lost",
                ),
            ]
        assert request.phase == "reconcile"
        return [
            _activation_output(request, "scheduler-0", state="serving"),
            _activation_output(request, "scheduler-1", state="serving"),
        ]

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        weight_snapshot_activation_communicator=activate,
    )
    success, message = asyncio.run(
        TokenizerControlMixin.update_weight_snapshot_activation(manager, "activate")
    )

    assert success is True
    assert "reconciled" in message
    assert [request.phase for request in requests] == [
        "prepare",
        "commit",
        "reconcile",
    ]
    assert len({request.transaction_id for request in requests}) == 1
    assert len({request.request_id for request in requests}) == 3


def test_weight_snapshot_activation_aborts_after_prepare_failure() -> None:
    requests = []

    async def activate(request, **_kwargs):
        requests.append(request)
        if request.phase == "prepare":
            return [
                _activation_output(request, "scheduler-0", state="prepared"),
                _activation_output(
                    request,
                    "scheduler-1",
                    success=False,
                    state="not_ready",
                    message="missing activation handle",
                ),
            ]
        assert request.phase == "abort"
        return [
            _activation_output(request, "scheduler-0", state="aborted"),
            _activation_output(request, "scheduler-1", state="aborted"),
        ]

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        weight_snapshot_activation_communicator=activate,
    )
    success, message = asyncio.run(
        TokenizerControlMixin.update_weight_snapshot_activation(manager, "activate")
    )

    assert success is False
    assert "prepare failed" in message
    assert [request.phase for request in requests] == ["prepare", "abort"]


def test_weight_snapshot_activation_quarantines_unreconciled_commit() -> None:
    requests = []

    async def activate(request, **_kwargs):
        requests.append(request)
        if request.phase == "prepare":
            state = "prepared"
            success = True
        elif request.phase == "commit":
            state = "commit_unknown"
            success = False
        elif request.phase == "reconcile":
            state = "conflict"
            success = False
        else:
            assert request.phase == "abort"
            state = "quarantined"
            success = False
        return [
            _activation_output(
                request,
                "scheduler-0",
                success=success,
                state=state,
                message=state,
            )
        ]

    manager = SimpleNamespace(
        auto_create_handle_loop=lambda: None,
        weight_snapshot_activation_communicator=activate,
    )
    success, message = asyncio.run(
        TokenizerControlMixin.update_weight_snapshot_activation(manager, "activate")
    )

    assert success is False
    assert "quarantined" in message
    assert [request.phase for request in requests] == [
        "prepare",
        "commit",
        "reconcile",
        "abort",
    ]


def test_out_of_order_prepare_selects_lowest_rank_and_holds_reader_lock() -> None:
    requests = []
    manager = None

    async def prepare(request, **_kwargs):
        requests.append(request)
        assert manager.model_update_lock.depth == 1
        return [_prepare_output(1), _prepare_output(0)]

    async def commit(request, **_kwargs):
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

    assert result["materialization_id"] == "materialize-1"
    assert result["ref"] == {
        "namespace": "weights",
        "snapshot_id": "snapshot-1",
    }
    assert result["selected_external_dp_rank"] == 0
    assert result["total_bytes"] == 4096
    assert result["session_state"] == "published"
    assert result["cleanup_state"] is None
    assert result["completion_unknown"] is False
    assert result["completion_ticket"] is None
    assert lock.entries == 1
    assert lock.depth == 0
    assert isinstance(requests[0], PrepareWeightMaterializationReqInput)
    assert requests[0].model_id == "Qwen/Qwen3-0.6B"
    assert requests[0].revision == "weights-v1"
    assert requests[0].request_id
    assert isinstance(requests[1], CommitWeightMaterializationReqInput)
    assert requests[1].selected_external_dp_rank == 0
    assert requests[1].request_id != requests[0].request_id


def test_published_result_survives_best_effort_cleanup() -> None:
    commits = []
    ref = {"namespace": "weights", "snapshot_id": "snapshot-1"}

    async def prepare(_request, **_kwargs):
        return [_prepare_output(0), _prepare_output(1)]

    async def commit(request, **_kwargs):
        commits.append(request)
        if request.selected_external_dp_rank is None:
            return [
                _commit_output(
                    0,
                    selected=True,
                    ref=ref,
                    state="published",
                    phase="cleanup",
                ),
                _commit_output(1, state="released", phase="cleanup"),
            ]
        return [
            _commit_output(
                0,
                selected=True,
                ref=ref,
                state="published_cleanup_pending",
            ),
            _commit_output(1, state="skipped"),
        ]

    manager, _ = _manager(prepare, commit)
    result = asyncio.run(
        TokenizerControlMixin.materialize_weights(manager, _request(), None)
    )

    assert result["ref"] == ref
    assert [request.selected_external_dp_rank for request in commits] == [0, None]


def test_uses_explicit_source_external_dp_rank() -> None:
    commits = []

    async def prepare(_request, **_kwargs):
        return [_prepare_output(0), _prepare_output(1)]

    async def commit(request, **_kwargs):
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

    async def prepare(_request, **_kwargs):
        return [
            _prepare_output(0),
            _prepare_output(
                1,
                success=False,
                message="capture failed",
                state="failed",
            ),
        ]

    async def commit(request, **_kwargs):
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

    async def prepare(_request, **_kwargs):
        return [_prepare_output(0), second]

    async def commit(request, **_kwargs):
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

    async def prepare(_request, **_kwargs):
        return [_prepare_output(0), _prepare_output(1)]

    async def commit(request, **_kwargs):
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

    async def prepare(_request, **_kwargs):
        prepare_started.set()
        await asyncio.Future()

    async def commit(request, **_kwargs):
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
    "error_type",
    [
        FanOutCancelledBeforeDispatch,
        FanOutDeadlineExpiredBeforeDispatch,
    ],
)
def test_before_dispatch_failure_does_not_broadcast_cleanup(error_type) -> None:
    commits = []

    async def prepare(_request, **_kwargs):
        raise error_type

    async def commit(request, **_kwargs):
        commits.append(request)
        raise AssertionError("cleanup must not be broadcast before dispatch")

    manager, _ = _manager(prepare, commit)

    with pytest.raises(error_type):
        asyncio.run(
            TokenizerControlMixin.materialize_weights(manager, _request(), None)
        )

    assert commits == []


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


if __name__ == "__main__":
    pytest_args = ["-x" if argument == "-f" else argument for argument in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *pytest_args]))
