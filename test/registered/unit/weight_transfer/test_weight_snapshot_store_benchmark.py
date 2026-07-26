from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
