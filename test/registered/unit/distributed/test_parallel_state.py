"""
Test file to verify the correctness of parallel group calculations.

This test validates that the parallel group initialization creates the correct
groups for different parallelism configurations including:
- Tensor parallelism (TP)
- Pipeline parallelism (PP)
- Attention context parallelism (attn_cp)
- Attention data parallelism (attn_dp)
- MoE expert parallelism (EP)
- MoE data parallelism (moe_dp)

These tests call the ACTUAL initialize_model_parallel() function with mocked
distributed backend to verify the group construction logic.

## How These Tests Work

initialize_model_parallel() creates ALL groups for ALL ranks in a single call.
For example, when creating TP groups with tp_size=2 and world_size=8:

    group_ranks = [[0,1], [2,3], [4,5], [6,7]]  # ALL groups created
    _TP = init_model_parallel_group(group_ranks, local_rank, ...)

ALL ranks call this function and get the same complete group structure. Each rank
then figures out which specific group(s) it belongs to.

Our tests:
1. Mock the distributed backend (no real GPUs needed)
2. Mock init_model_parallel_group to capture the group_ranks parameter
3. Call the real initialize_model_parallel()
4. Verify group_ranks contains the expected complete group structure

We only need to simulate rank 0 because we're testing the group creation logic,
not the per-rank group membership logic.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd")

# Import the actual parallel_state module
parallel_state = pytest.importorskip("sglang.srt.distributed.parallel_state")


def _run_bounded_object_collective_process(
    rank,
    init_method,
    result_connection,
):
    import os

    import torch.distributed
    from sglang.srt.weight_transfer.provider import WeightTransferExecutionContext

    try:
        torch.distributed.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=2,
        )
        coordinator = _make_group_coordinator([0, 1], rank_in_group=rank)
        coordinator.cpu_group = torch.distributed.group.WORLD
        normal_context = WeightTransferExecutionContext(
            deadline_unix_sec=time.time() + 10.0,
        )
        shared_deadline = coordinator.synchronize_object_collective_deadline(
            phase="test.deadline",
            execution_context=normal_context,
        )
        normal_context = WeightTransferExecutionContext(
            deadline_unix_sec=shared_deadline,
        )
        broadcast = coordinator.broadcast_object(
            {"owner": 1, "payload": "x" * 4097} if rank == 1 else None,
            src=1,
            phase="test.broadcast",
            execution_context=normal_context,
        )
        all_gather = coordinator.all_gather_object(
            {"rank": rank, "payload": "y" * (rank * 3073 + 1)},
            phase="test.all_gather",
            execution_context=normal_context,
        )
        gather = coordinator.gather_object(
            {"rank": rank, "payload": "z" * (rank * 2053 + 1)},
            dst=1,
            phase="test.gather",
            execution_context=normal_context,
        )
        scatter = coordinator.scatter_object(
            (
                [
                    {"target": 0, "payload": "short"},
                    {"target": 1, "payload": "w" * 5003},
                ]
                if rank == 1
                else None
            ),
            src=1,
            phase="test.scatter",
            execution_context=normal_context,
        )
        result = {
            "rank": rank,
            "shared_deadline": shared_deadline,
            "broadcast": broadcast,
            "all_gather": all_gather,
            "gather": gather,
            "scatter": scatter,
        }
        if rank == 1:
            time.sleep(1.5)
        else:
            stuck_context = WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + 0.35,
            )
            started = time.monotonic()
            try:
                coordinator.all_gather_object(
                    "rank-0-only",
                    phase="test.stuck_rank",
                    execution_context=stuck_context,
                )
            except BaseException as error:
                result["timeout_elapsed"] = time.monotonic() - started
                result["completion_unknown"] = bool(
                    getattr(error, "completion_unknown", False)
                )
                result["timeout_error"] = str(error)
            else:
                result["timeout_error"] = None
            retry_started = time.monotonic()
            try:
                coordinator.all_gather_object(
                    "must-fail-fast",
                    phase="test.poisoned",
                    execution_context=WeightTransferExecutionContext(
                        deadline_unix_sec=time.time() + 10.0,
                    ),
                )
            except BaseException as error:
                result["retry_elapsed"] = time.monotonic() - retry_started
                result["retry_error"] = str(error)
            else:
                result["retry_error"] = None
            result["poisoned"] = coordinator.bounded_object_collectives_poisoned
        result_connection.send(result)
    except BaseException as error:
        result_connection.send(
            {
                "rank": rank,
                "worker_error": f"{type(error).__name__}: {error}",
            }
        )
    finally:
        result_connection.close()
        os._exit(0)


@pytest.mark.skipif(
    not parallel_state.torch.distributed.is_available()
    or not parallel_state.torch.distributed.is_gloo_available(),
    reason="Gloo is required for bounded CPU object collectives",
)
def test_bounded_object_collectives_use_real_work_and_timeout_stuck_rank(
    tmp_path,
):
    process_context = parallel_state.torch.multiprocessing.get_context("spawn")
    init_method = f"file://{tmp_path / 'bounded-object-collective-init'}"
    processes = []
    result_connections = []
    for rank in range(2):
        result_connection, child_connection = process_context.Pipe(duplex=False)
        process = process_context.Process(
            target=_run_bounded_object_collective_process,
            args=(rank, init_method, child_connection),
        )
        process.start()
        child_connection.close()
        processes.append(process)
        result_connections.append(result_connection)

    try:
        for process in processes:
            process.join(timeout=30)
        assert all(not process.is_alive() for process in processes)
        assert all(process.exitcode == 0 for process in processes)
        assert all(connection.poll(1.0) for connection in result_connections)
        results = sorted(
            (connection.recv() for connection in result_connections),
            key=lambda result: result["rank"],
        )
    finally:
        for connection in result_connections:
            connection.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all("worker_error" not in result for result in results)
    assert results[0]["shared_deadline"] == results[1]["shared_deadline"]
    assert [result["broadcast"] for result in results] == [
        {"owner": 1, "payload": "x" * 4097},
        {"owner": 1, "payload": "x" * 4097},
    ]
    expected_gathered = [
        {"rank": 0, "payload": "y"},
        {"rank": 1, "payload": "y" * 3074},
    ]
    assert [result["all_gather"] for result in results] == [
        expected_gathered,
        expected_gathered,
    ]
    assert results[0]["gather"] is None
    assert results[1]["gather"] == [
        {"rank": 0, "payload": "z"},
        {"rank": 1, "payload": "z" * 2054},
    ]
    assert [result["scatter"] for result in results] == [
        {"target": 0, "payload": "short"},
        {"target": 1, "payload": "w" * 5003},
    ]
    assert 0.2 <= results[0]["timeout_elapsed"] < 1.5
    assert results[0]["completion_unknown"] is True
    assert "deadline exceeded" in results[0]["timeout_error"]
    assert results[0]["poisoned"] is True
    assert results[0]["retry_elapsed"] < 0.2
    assert "poisoned" in results[0]["retry_error"]


def _make_group_coordinator(ranks, rank_in_group):
    coordinator = object.__new__(parallel_state.GroupCoordinator)
    coordinator.ranks = ranks
    coordinator.world_size = len(ranks)
    coordinator.rank_in_group = rank_in_group
    coordinator.cpu_group = object()
    return coordinator


def test_bounded_object_collectives_live_in_distributed_layer():
    assert (
        parallel_state._BoundedObjectCollectiveCoordinator.__module__
        == "sglang.srt.distributed.bounded_object_collectives"
    )


def test_bounded_object_collective_maintenance_has_internal_owners():
    assert not hasattr(
        parallel_state._BoundedObjectCollectiveCoordinator,
        "_serialized_value",
    )
    assert not hasattr(
        parallel_state.GroupCoordinator,
        "reap_bounded_object_collectives",
    )


def test_gather_object_returns_values_only_on_root(monkeypatch):
    root = _make_group_coordinator([2, 5, 9], rank_in_group=1)
    peer = _make_group_coordinator([2, 5, 9], rank_in_group=0)
    calls = []

    def gather_object(obj, outputs, *, dst, group):
        calls.append((obj, outputs, dst, group))
        if outputs is not None:
            outputs[:] = ["rank-2", "rank-5", "rank-9"]

    monkeypatch.setattr(
        parallel_state.torch.distributed, "gather_object", gather_object
    )

    assert root.gather_object("rank-5", dst=1) == [
        "rank-2",
        "rank-5",
        "rank-9",
    ]
    assert peer.gather_object("rank-2", dst=1) is None
    assert calls[0][2:] == (5, root.cpu_group)
    assert calls[1][1] is None
    assert calls[1][2:] == (5, peer.cpu_group)


def test_scatter_object_uses_global_source_rank(monkeypatch):
    root = _make_group_coordinator([3, 7, 11], rank_in_group=1)
    peer = _make_group_coordinator([3, 7, 11], rank_in_group=2)
    calls = []

    def scatter_object_list(outputs, inputs, *, src, group):
        calls.append((inputs, src, group))
        outputs[0] = "root-value" if inputs is not None else "peer-value"

    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "scatter_object_list",
        scatter_object_list,
    )

    values = ["for-rank-3", "for-rank-7", "for-rank-11"]
    assert root.scatter_object(values, src=1) == "root-value"
    assert peer.scatter_object(None, src=1) == "peer-value"
    assert calls == [
        (values, 7, root.cpu_group),
        (None, 7, peer.cpu_group),
    ]


def test_object_collectives_bypass_single_rank(monkeypatch):
    coordinator = _make_group_coordinator([4], rank_in_group=0)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("single-rank collective reached torch.distributed")

    monkeypatch.setattr(
        parallel_state.torch.distributed, "gather_object", unexpected_call
    )
    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "scatter_object_list",
        unexpected_call,
    )

    assert coordinator.gather_object("only", dst=0) == ["only"]
    assert coordinator.scatter_object(["only"], src=0) == "only"


@pytest.mark.parametrize(
    "method, root_name", [("gather_object", "dst"), ("scatter_object", "src")]
)
@pytest.mark.parametrize("root", [-1, 3])
def test_object_collectives_reject_invalid_root(method, root_name, root):
    coordinator = _make_group_coordinator([2, 5, 9], rank_in_group=0)

    with pytest.raises(AssertionError, match=f"Invalid {root_name} rank"):
        getattr(coordinator, method)("value", **{root_name: root})


def test_scatter_object_rejects_wrong_source_list_length(monkeypatch):
    coordinator = _make_group_coordinator([2, 5, 9], rank_in_group=1)
    called = False

    def scatter_object_list(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "scatter_object_list",
        scatter_object_list,
    )

    with pytest.raises(AssertionError, match="Expected 3 objects"):
        coordinator.scatter_object(["rank-2", "rank-5"], src=1)
    assert not called


def test_object_collectives_use_bounded_backend_with_execution_context(monkeypatch):
    coordinator = _make_group_coordinator([2, 5, 9], rank_in_group=0)
    execution_context = SimpleNamespace(
        deadline_unix_sec=1234.5,
        expired=lambda: False,
        cancelled=lambda: False,
        remaining_seconds=lambda: 10.0,
    )
    calls = []

    class BoundedCollectives:
        def synchronize_deadline(self, **kwargs):
            calls.append(("deadline", None, kwargs))
            return execution_context.deadline_unix_sec

        def broadcast_object(self, value, **kwargs):
            calls.append(("broadcast", value, kwargs))
            return "broadcast-result"

        def all_gather_object(self, value, **kwargs):
            calls.append(("all-gather", value, kwargs))
            return ["all-gather-result"]

        def gather_object(self, value, **kwargs):
            calls.append(("gather", value, kwargs))
            return ["gather-result"]

        def scatter_object(self, values, **kwargs):
            calls.append(("scatter", values, kwargs))
            return "scatter-result"

    coordinator._bounded_object_collective_coordinator = BoundedCollectives()

    assert (
        coordinator.synchronize_object_collective_deadline(
            phase="session.deadline",
            execution_context=execution_context,
        )
        == execution_context.deadline_unix_sec
    )

    def unexpected_sync_collective(*args, **kwargs):
        raise AssertionError("bounded collective reached a synchronous object API")

    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "broadcast_object_list",
        unexpected_sync_collective,
    )
    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "all_gather_object",
        unexpected_sync_collective,
    )
    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "gather_object",
        unexpected_sync_collective,
    )
    monkeypatch.setattr(
        parallel_state.torch.distributed,
        "scatter_object_list",
        unexpected_sync_collective,
    )

    assert (
        coordinator.broadcast_object(
            "broadcast",
            src=0,
            phase="session.acquire",
            execution_context=execution_context,
        )
        == "broadcast-result"
    )
    assert coordinator.all_gather_object(
        "all-gather",
        phase="session.readiness",
        execution_context=execution_context,
    ) == ["all-gather-result"]
    assert coordinator.gather_object(
        "gather",
        dst=0,
        phase="session.plan.gather",
        execution_context=execution_context,
    ) == ["gather-result"]
    assert (
        coordinator.scatter_object(
            ["rank-2", "rank-5", "rank-9"],
            src=0,
            phase="session.plan.scatter",
            execution_context=execution_context,
        )
        == "scatter-result"
    )

    assert [call[:2] for call in calls] == [
        ("deadline", None),
        ("broadcast", "broadcast"),
        ("all-gather", "all-gather"),
        ("gather", "gather"),
        ("scatter", ["rank-2", "rank-5", "rank-9"]),
    ]
    assert [call[2]["phase"] for call in calls] == [
        "session.deadline",
        "session.acquire",
        "session.readiness",
        "session.plan.gather",
        "session.plan.scatter",
    ]
    assert all(call[2]["execution_context"] is execution_context for call in calls)
    assert calls[1][2]["src"] == 0
    assert calls[3][2]["dst"] == 0
    assert calls[4][2]["src"] == 0


def test_bounded_object_collective_unknown_completion_aborts_and_poisons_group():
    coordinator = _make_group_coordinator([2, 5, 9], rank_in_group=1)
    execution_context = SimpleNamespace(
        expired=lambda: False,
        cancelled=lambda: False,
        remaining_seconds=lambda: 10.0,
    )
    abort_calls = []
    backend_calls = []

    class CompletionUnknown(RuntimeError):
        completion_unknown = True

    class CpuGroup:
        def abort(self):
            abort_calls.append("abort")

    class BoundedCollectives:
        def all_gather_object(self, value, **kwargs):
            backend_calls.append((value, kwargs))
            raise CompletionUnknown("rank 2 did not enter the collective")

    coordinator.cpu_group = CpuGroup()
    coordinator._bounded_object_collective_coordinator = BoundedCollectives()

    with pytest.raises(CompletionUnknown, match="rank 2"):
        coordinator.all_gather_object(
            True,
            phase="session.readiness",
            execution_context=execution_context,
        )
    with pytest.raises(RuntimeError, match="poisoned"):
        coordinator.all_gather_object(
            False,
            phase="session.finish",
            execution_context=execution_context,
        )

    assert abort_calls == ["abort"]
    assert len(backend_calls) == 1
    assert coordinator.bounded_object_collectives_poisoned is True


def test_parallel_group_construction_tp8_attn_cp2():
    """
    Test parallel group construction for 8 GPU configuration with:
    - tensor_model_parallel_size = 8
    - attention_context_model_parallel_size = 2

    Expected groups based on docstring example:
        1 tensor model-parallel group:
            [g0, g1, g2, g3, g4, g5, g6, g7]
        4 attention context-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]

    This test calls the ACTUAL initialize_model_parallel() and verifies the groups.

    Note: We simulate only rank 0 here, but initialize_model_parallel() creates
    ALL groups for ALL ranks in a single call. We capture these groups via mocking
    and verify the complete group structure.
    """
    world_size = 8

    # Mock the distributed backend
    # Note: get_rank() returns 0 because we're testing from a single process,
    # but initialize_model_parallel() still creates all groups for all ranks
    with (
        patch.object(parallel_state, "_WORLD", None),
        patch.object(parallel_state, "_TP", None),
        patch.object(parallel_state, "_ATTN_CP", None),
        patch.object(parallel_state, "_ATTN_TP", None),
        patch.object(parallel_state, "_PP", None),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_backend", return_value="nccl"),
    ):
        # Mock init_model_parallel_group to capture the groups being created
        created_groups = {}

        def mock_init_model_parallel_group(group_ranks, local_rank, backend, **kwargs):
            group_name = kwargs.get("group_name", "unknown")
            created_groups[group_name] = group_ranks

            # Create a mock group object
            mock_group = Mock()
            mock_group.device_group = Mock()
            return mock_group

        with (
            patch.object(
                parallel_state,
                "init_model_parallel_group",
                side_effect=mock_init_model_parallel_group,
            ),
            patch.object(parallel_state, "get_world_group") as mock_world_group,
        ):
            # Mock world group
            mock_world = Mock()
            mock_world.device_group = Mock()
            mock_world.local_rank = 0
            mock_world_group.return_value = mock_world

            # Call the actual function
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=8,
                pipeline_model_parallel_size=1,
                attention_context_model_parallel_size=2,
            )

            # Verify TP groups
            tp_groups = created_groups.get("tp", [])
            assert len(tp_groups) == 1, f"Expected 1 TP group, got {len(tp_groups)}"
            assert tp_groups[0] == [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ], f"Wrong TP group: {tp_groups[0]}"

            # Verify ATTN_CP groups
            attn_cp_groups = created_groups.get("attn_cp", [])
            assert len(attn_cp_groups) == 4, (
                f"Expected 4 ATTN_CP groups, got {len(attn_cp_groups)}"
            )
            expected_attn_cp = [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            assert attn_cp_groups == expected_attn_cp, (
                f"Wrong ATTN_CP groups: {attn_cp_groups}"
            )

            print("TP=8, Attn CP=2 group construction verified")

            # Cleanup
            parallel_state.destroy_model_parallel()


def test_parallel_group_construction_tp8_moe_ep4_cp2():
    """
    Test parallel group construction for 8 GPU configuration with:
    - tensor_model_parallel_size = 8
    - expert_model_parallel_size = 4
    - moe_data_model_parallel_size = 2

    Expected groups:
        1 tensor model-parallel group:
            [g0, g1, g2, g3, g4, g5, g6, g7]
        2 MoE expert-parallel groups:
            [g0, g1, g2, g3], [g4, g5, g6, g7]
        4 MoE data-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]
    """
    world_size = 8

    # Mock the distributed backend
    with (
        patch.object(parallel_state, "_WORLD", None),
        patch.object(parallel_state, "_TP", None),
        patch.object(parallel_state, "_MOE_EP", None),
        patch.object(parallel_state, "_MOE_DP", None),
        patch.object(parallel_state, "_MOE_TP", None),
        patch.object(parallel_state, "_PP", None),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=world_size),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.get_backend", return_value="nccl"),
    ):
        # Mock init_model_parallel_group to capture the groups being created
        created_groups = {}

        def mock_init_model_parallel_group(group_ranks, local_rank, backend, **kwargs):
            group_name = kwargs.get("group_name", "unknown")
            created_groups[group_name] = group_ranks

            # Create a mock group object
            mock_group = Mock()
            mock_group.device_group = Mock()
            return mock_group

        with (
            patch.object(
                parallel_state,
                "init_model_parallel_group",
                side_effect=mock_init_model_parallel_group,
            ),
            patch.object(parallel_state, "get_world_group") as mock_world_group,
        ):
            # Mock world group
            mock_world = Mock()
            mock_world.device_group = Mock()
            mock_world.local_rank = 0
            mock_world_group.return_value = mock_world

            # Call the actual function
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=8,
                expert_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                moe_data_model_parallel_size=2,
            )

            # Verify TP groups
            tp_groups = created_groups.get("tp", [])
            assert len(tp_groups) == 1, f"Expected 1 TP group, got {len(tp_groups)}"
            assert tp_groups[0] == [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ], f"Wrong TP group: {tp_groups[0]}"

            # Verify MOE_EP groups
            moe_ep_groups = created_groups.get("moe_ep", [])
            assert len(moe_ep_groups) == 2, (
                f"Expected 2 MOE_EP groups, got {len(moe_ep_groups)}"
            )
            expected_moe_ep = [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
            assert moe_ep_groups == expected_moe_ep, (
                f"Wrong MOE_EP groups: {moe_ep_groups}"
            )

            # Verify MOE_DP groups
            moe_dp_groups = created_groups.get("moe_dp", [])
            assert len(moe_dp_groups) == 4, (
                f"Expected 4 MOE_DP groups, got {len(moe_dp_groups)}"
            )
            expected_moe_dp = [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
            assert moe_dp_groups == expected_moe_dp, (
                f"Wrong MOE_DP groups: {moe_dp_groups}"
            )

            print("TP=8, MoE EP=4, MoE CP=2 group construction verified")

            # Cleanup
            parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    # Registered tests are launched as scripts with pytest arguments.
    import sys

    import pytest

    pytest_args = ["-x" if argument == "-f" else argument for argument in sys.argv[1:]]
    sys.exit(pytest.main([__file__, *pytest_args]))
