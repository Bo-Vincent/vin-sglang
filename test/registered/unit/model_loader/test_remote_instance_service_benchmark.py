import importlib.util
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_PATH = REPO_ROOT / "test/manual/benchmark_remote_instance_service_startup.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_remote_instance_service_startup", BENCHMARK_PATH
)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _args(**overrides):
    values = dict(
        python="python",
        model="model",
        content_revision="checkpoint-sha-42",
        source_tp_size=4,
        target_tp_size=2,
        source_pp_size=1,
        target_pp_size=1,
        source_dp_size=1,
        target_dp_size=1,
        source_ep_size=4,
        target_ep_size=2,
        mem_fraction_static=0.8,
        attention_backend="",
        mm_attention_backend="",
        sampling_backend="",
        disable_custom_all_reduce=False,
        disable_shared_experts_fusion=True,
        moe_runner_backend="triton",
        modes=("cold", "legacy", "reshard"),
        bootstrap_port=31999,
        source_port=31000,
        protocol="tcp",
        transport_device="",
        request_timeout_s=1,
        prompt="test",
        max_new_tokens=1,
        sampling_seed=0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _server(tmp_path, process, port=32000):
    log_path = tmp_path / "server.log"
    log_path.write_text("", encoding="utf-8")
    return benchmark.ServerProcess(
        process=process,
        log_file=SimpleNamespace(close=lambda: None),
        log_path=log_path,
        started_at=time.perf_counter(),
        port=port,
    )


class _FakeProcess:
    def __init__(self, pid=1234, poll_results=None):
        self.pid = pid
        self.returncode = None
        self._poll_results = iter(poll_results or [None])
        self._last_poll = None

    def poll(self):
        try:
            self._last_poll = next(self._poll_results)
        except StopIteration:
            pass
        self.returncode = self._last_poll
        return self._last_poll


class _Response:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"text": "ok"}


def test_start_server_rejects_occupied_port_before_spawn(monkeypatch, tmp_path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]

        def unexpected_popen(*args, **kwargs):
            raise AssertionError("Popen must not run for an occupied port")

        monkeypatch.setattr(benchmark.subprocess, "Popen", unexpected_popen)
        with pytest.raises(RuntimeError, match=rf"port {port} .*already in use"):
            benchmark._start_server(
                _args(),
                gpus="0",
                port=port,
                load_mode="cold",
                log_path=tmp_path / "occupied.log",
            )


def test_wait_ready_rejects_a_child_that_exits_after_health(
    monkeypatch, tmp_path
) -> None:
    process = _FakeProcess(poll_results=[None, None, None, None, 17])
    server = _server(tmp_path, process)
    monkeypatch.setattr(
        benchmark, "_listening_port_owner_pids", lambda port: {process.pid}
    )
    monkeypatch.setattr(benchmark, "_process_tree_pids", lambda root_pid: {root_pid})
    monkeypatch.setattr(benchmark.requests, "get", lambda *args, **kwargs: _Response())

    with pytest.raises(RuntimeError, match="server exited with 17"):
        benchmark._wait_ready(server, server.port, timeout_s=0.1)


def test_wait_ready_rejects_listener_outside_spawned_process_tree(
    monkeypatch, tmp_path
) -> None:
    process = _FakeProcess(pid=1234)
    server = _server(tmp_path, process)
    monkeypatch.setattr(
        benchmark, "_listening_port_owner_pids", lambda port: {9999}, raising=False
    )
    monkeypatch.setattr(
        benchmark, "_process_tree_pids", lambda root_pid: {root_pid}, raising=False
    )

    def unexpected_get(*args, **kwargs):
        raise AssertionError("health must not be sent to an unexpected listener owner")

    monkeypatch.setattr(benchmark.requests, "get", unexpected_get)

    with pytest.raises(RuntimeError, match="listener owner.*9999.*process tree"):
        benchmark._wait_ready(server, server.port, timeout_s=0.1)


def test_generate_rechecks_owner_and_process_after_response(
    monkeypatch, tmp_path
) -> None:
    process = _FakeProcess(poll_results=[None, None, 23])
    server = _server(tmp_path, process)
    monkeypatch.setattr(
        benchmark, "_listening_port_owner_pids", lambda port: {process.pid}
    )
    monkeypatch.setattr(benchmark, "_process_tree_pids", lambda root_pid: {root_pid})
    monkeypatch.setattr(benchmark.requests, "post", lambda *args, **kwargs: _Response())

    with pytest.raises(RuntimeError, match="server exited with 23"):
        benchmark._generate(_args(), server)


def test_server_identity_is_auditable(monkeypatch, tmp_path) -> None:
    process = _FakeProcess(pid=1234)
    server = _server(tmp_path, process)
    monkeypatch.setattr(
        benchmark, "_listening_port_owner_pids", lambda port: {1235}, raising=False
    )
    monkeypatch.setattr(
        benchmark,
        "_process_tree_pids",
        lambda root_pid: {root_pid, 1235},
        raising=False,
    )

    assert benchmark._assert_server_identity(server) == {
        "root_pid": 1234,
        "port": 32000,
        "process_tree_pids": [1234, 1235],
        "listener_owner_pids": [1235],
    }


def test_server_command_describes_dp_and_ep_topology() -> None:
    args = _args(source_dp_size=2, target_dp_size=3)

    source = benchmark._server_command(args, port=31000, load_mode="source")
    target = benchmark._server_command(args, port=32000, load_mode="reshard")

    assert source[source.index("--revision") + 1] == "checkpoint-sha-42"
    assert target[target.index("--revision") + 1] == "checkpoint-sha-42"

    assert source[source.index("--dp-size") + 1] == "2"
    assert source[source.index("--ep-size") + 1] == "4"
    assert target[target.index("--dp-size") + 1] == "3"
    assert target[target.index("--ep-size") + 1] == "2"
    assert "--disable-shared-experts-fusion" in source
    assert "--disable-shared-experts-fusion" in target
    assert "--enable-deterministic-inference" in source
    assert "--enable-deterministic-inference" in target


def test_legacy_mode_is_skipped_for_heterogeneous_dp_or_ep() -> None:
    executed, skipped = benchmark._eligible_modes(
        _args(source_tp_size=2, target_tp_size=2)
    )

    assert executed == ["cold", "reshard"]
    assert [item["mode"] for item in skipped] == ["legacy"]
    assert "source_ep_size=4" in skipped[0]["reason"]
    assert "target_ep_size=2" in skipped[0]["reason"]


def test_reuse_modes_are_skipped_when_both_sides_use_dp_two() -> None:
    executed, skipped = benchmark._eligible_modes(
        _args(
            source_tp_size=2,
            target_tp_size=2,
            source_ep_size=2,
            target_ep_size=2,
            source_dp_size=2,
            target_dp_size=2,
        )
    )

    assert executed == ["cold"]
    assert [item["mode"] for item in skipped] == ["legacy", "reshard"]
    assert all("source_dp_size=2" in item["reason"] for item in skipped)


def test_heterogeneous_reuse_uses_cold_target_as_correctness_baseline() -> None:
    cold_target = {"text": "cold-target-output"}

    assert benchmark._ordered_modes(("reshard", "cold")) == ["cold", "reshard"]
    assert benchmark._target_expected_response("cold", None) is None
    assert benchmark._target_expected_response("reshard", cold_target) is cold_target
    with pytest.raises(RuntimeError, match="cold target baseline"):
        benchmark._target_expected_response("reshard", None)


def test_response_parity_compares_logprobs_with_tolerance() -> None:
    response = {
        "text": "42",
        "meta_info": {
            "input_token_logprobs": [[None, 1, None], [-0.25, 2, None]],
            "output_token_logprobs": [[-0.125, 3, None]],
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "finish_reason": {"type": "length", "length": 1},
        },
    }
    expected = benchmark._deterministic_response(response)
    close = {
        **expected,
        "output_token_logprobs": [-0.1250005],
    }
    wrong = {
        **expected,
        "output_token_logprobs": [-0.25],
    }

    assert expected["input_token_logprobs"] == [None, -0.25]
    assert benchmark._responses_match(expected, close, atol=1e-5, rtol=1e-5)
    assert not benchmark._responses_match(expected, wrong, atol=1e-5, rtol=1e-5)


def test_source_probe_p95_must_stay_within_baseline_distribution(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="source probe p95"):
        benchmark._assert_iteration_consistency(
            mode="reshard",
            iteration=0,
            measurements=[],
            source_probe={
                "success_count": 5,
                "error_count": 0,
                "mismatch_count": 0,
                "latency_p95_s": 3.1,
            },
            source_baseline_p95_s=1.0,
            max_source_probe_p95_ratio=3.0,
            min_source_probe_samples=5,
            responses_path=tmp_path / "responses.jsonl",
        )


def test_source_probe_requires_enough_samples_for_p95(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="fewer than 5"):
        benchmark._assert_iteration_consistency(
            mode="reshard",
            iteration=0,
            measurements=[],
            source_probe={
                "success_count": 4,
                "error_count": 0,
                "mismatch_count": 0,
                "latency_p95_s": 0.5,
            },
            source_baseline_p95_s=1.0,
            max_source_probe_p95_ratio=3.0,
            min_source_probe_samples=5,
            responses_path=tmp_path / "responses.jsonl",
        )


def test_reshard_log_contract_requires_the_heterogeneous_loader(tmp_path) -> None:
    log_path = tmp_path / "reshard.log"
    log_path.write_text(
        "TransferEngine memory regions have been successfully registered.\n"
        "Loaded heterogeneous remote-instance weights: "
        "transfer_id=transfer-1, "
        "release_success=true, bytes=4096, "
        "compact_operations=2, segments=4, elapsed=0.1s\n",
        encoding="utf-8",
    )

    evidence = benchmark._assert_reuse_log_contract("reshard", log_path)

    assert evidence["passed"] is True
    assert (
        "Loaded heterogeneous remote-instance weights:" in evidence["required_markers"]
    )


def test_reshard_log_contract_rejects_missing_success_marker(tmp_path) -> None:
    log_path = tmp_path / "reshard.log"
    log_path.write_text(
        "TransferEngine memory regions have been successfully registered.\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing required log marker"):
        benchmark._assert_reuse_log_contract("reshard", log_path)


@pytest.mark.parametrize(
    "failure_marker",
    [
        "Fallback load_format to 'auto'",
        "using a retired target inventory builder",
        "Heterogeneous remote-instance weight loading failed",
        "completion remains unknown",
        "Loaded weights but failed to release source transfer session",
        "Remote weight transfer lease renewal failed",
    ],
)
def test_reshard_log_contract_rejects_fallback_and_transfer_failures(
    tmp_path, failure_marker
) -> None:
    log_path = tmp_path / "reshard.log"
    log_path.write_text(
        "TransferEngine memory regions have been successfully registered.\n"
        "Loaded heterogeneous remote-instance weights: "
        "transfer_id=transfer-1, "
        "release_success=true, bytes=4096, "
        "compact_operations=2, segments=4, elapsed=0.1s\n"
        f"{failure_marker}\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="forbidden log marker"):
        benchmark._assert_reuse_log_contract("reshard", log_path)


def test_execution_schedule_balances_reuse_mode_order() -> None:
    assert benchmark._execution_schedule(("reshard", "cold", "legacy"), 4) == [
        ["cold", "reshard", "legacy"],
        ["cold", "legacy", "reshard"],
        ["cold", "reshard", "legacy"],
        ["cold", "legacy", "reshard"],
    ]


def test_parse_args_counts_only_eligible_reuse_modes(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            "model",
            "--content-revision",
            "revision-1",
            "--source-gpus",
            "0,1",
            "--target-gpus",
            "2,3,4,5",
            "--source-tp-size",
            "2",
            "--target-tp-size",
            "4",
            "--iterations",
            "1",
            "--drop-page-cache",
        ],
    )

    assert benchmark.parse_args().iterations == 1


def test_parse_args_requires_complete_reuse_order_cycles(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            "model",
            "--content-revision",
            "revision-1",
            "--source-gpus",
            "0,1",
            "--target-gpus",
            "2,3",
            "--source-tp-size",
            "2",
            "--target-tp-size",
            "2",
            "--iterations",
            "3",
            "--drop-page-cache",
        ],
    )

    with pytest.raises(SystemExit):
        benchmark.parse_args()
    assert "complete reuse-mode ordering cycles" in capsys.readouterr().err
