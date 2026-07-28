from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_PATH = REPO_ROOT / "test/manual/benchmark_weight_snapshot_store.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_weight_snapshot_store",
    BENCHMARK_PATH,
)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def test_probe_window_uses_request_overlap_not_completion_only() -> None:
    window_start = 100.0
    window_end = 200.0

    assert benchmark._probe_overlaps_window(
        {
            "started_at_s": 150.0,
            "completed_at_s": 220.0,
        },
        window_start,
        window_end,
    )
    assert not benchmark._probe_overlaps_window(
        {
            "started_at_s": 201.0,
            "completed_at_s": 220.0,
        },
        window_start,
        window_end,
    )


def test_store_continuity_controls_process_exit_code() -> None:
    failed = {
        "summary": {
            "source_serving_continuity_passed": False,
        }
    }
    passed = {
        "summary": {
            "source_serving_continuity_passed": True,
        }
    }

    assert benchmark._continuity_exit_code(failed, report_only=False) == 2
    assert benchmark._continuity_exit_code(failed, report_only=True) == 0
    assert benchmark._continuity_exit_code(passed, report_only=False) == 0


def test_baseline_discards_warmups_and_uses_multiple_measured_samples(
    monkeypatch,
) -> None:
    calls = []
    responses = [
        {"stable": "cold-0"},
        {"stable": "cold-1"},
        {"stable": "warmed"},
        {"stable": "warmed"},
        {"stable": "warmed"},
    ]
    ticks = iter((10.0, 10.2, 20.0, 20.3, 30.0, 30.4))

    def post(url, payload, timeout):
        calls.append((url, payload, timeout))
        return responses[len(calls) - 1]

    monkeypatch.setattr(benchmark, "_post_json", post)
    monkeypatch.setattr(benchmark, "_stable_response", lambda response: response)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))

    baseline, latencies = benchmark._collect_warmed_baseline(
        "http://source/generate",
        {"request": True},
        timeout=1,
        warmup_samples=2,
        sample_count=3,
    )

    assert baseline == {"stable": "warmed"}
    assert latencies == pytest.approx([0.2, 0.3, 0.4])
    assert len(calls) == 5


def test_continuity_gate_rejects_unfinished_probe_worker() -> None:
    complete = {
        "started_at_s": 100.0,
        "completed_at_s": 101.0,
        "latency_s": 1.0,
        "success": True,
        "consistent": True,
    }
    unfinished = {
        "started_at_s": 101.0,
        "completed_at_s": None,
        "success": None,
        "consistent": None,
    }

    assert not benchmark._source_serving_continuity_passed(
        in_window=[complete, unfinished],
        min_samples=1,
        materialization_published=True,
        post_consistent=True,
        probe_worker_stopped=False,
        latency_limit_s=2.0,
    )
    assert benchmark._source_serving_continuity_passed(
        in_window=[complete],
        min_samples=1,
        materialization_published=True,
        post_consistent=True,
        probe_worker_stopped=True,
        latency_limit_s=2.0,
    )


def test_probe_latency_limit_uses_warmed_ratio_and_absolute_cap() -> None:
    assert benchmark._probe_latency_limit(
        warmed_baseline_p95_s=0.4,
        max_ratio=10,
        absolute_limit_s=3,
    ) == pytest.approx(3)
    assert benchmark._probe_latency_limit(
        warmed_baseline_p95_s=0.2,
        max_ratio=10,
        absolute_limit_s=3,
    ) == pytest.approx(2)


if __name__ == "__main__":
    pytest_args = ["-x" if argument == "-f" else argument for argument in sys.argv[1:]]
    raise SystemExit(pytest.main([__file__, *pytest_args]))
