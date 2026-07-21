import importlib.util
import sys
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
        modes=("cold", "legacy", "manifest"),
        bootstrap_port=31999,
        source_port=31000,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_server_command_describes_dp_and_ep_topology() -> None:
    args = _args(source_dp_size=2, target_dp_size=3)

    source = benchmark._server_command(args, port=31000, load_mode="source")
    target = benchmark._server_command(args, port=32000, load_mode="manifest")

    assert source[source.index("--dp-size") + 1] == "2"
    assert source[source.index("--ep-size") + 1] == "4"
    assert target[target.index("--dp-size") + 1] == "3"
    assert target[target.index("--ep-size") + 1] == "2"
    assert "--disable-shared-experts-fusion" in source
    assert "--disable-shared-experts-fusion" in target


def test_legacy_mode_is_skipped_for_heterogeneous_dp_or_ep() -> None:
    executed, skipped = benchmark._eligible_modes(
        _args(source_tp_size=2, target_tp_size=2)
    )

    assert executed == ["cold", "manifest"]
    assert [item["mode"] for item in skipped] == ["legacy"]
    assert "source_ep_size=4" in skipped[0]["reason"]
    assert "target_ep_size=2" in skipped[0]["reason"]


def test_legacy_mode_is_skipped_when_both_sides_use_dp_two() -> None:
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

    assert executed == ["cold", "manifest"]
    assert [item["mode"] for item in skipped] == ["legacy"]
    assert "source_dp_size=2" in skipped[0]["reason"]


def test_heterogeneous_reuse_uses_cold_target_as_correctness_baseline() -> None:
    cold_target = {"text": "cold-target-output"}

    assert benchmark._ordered_modes(("manifest", "cold")) == ["cold", "manifest"]
    assert benchmark._target_expected_response("cold", None) is None
    assert benchmark._target_expected_response("manifest", cold_target) is cold_target
    with pytest.raises(RuntimeError, match="cold target baseline"):
        benchmark._target_expected_response("manifest", None)


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
            mode="manifest",
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
            mode="manifest",
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


def test_execution_schedule_balances_reuse_mode_order() -> None:
    assert benchmark._execution_schedule(("manifest", "cold", "legacy"), 4) == [
        ["cold", "manifest", "legacy"],
        ["cold", "legacy", "manifest"],
        ["cold", "manifest", "legacy"],
        ["cold", "legacy", "manifest"],
    ]


def test_parse_args_counts_only_eligible_reuse_modes(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            "model",
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
