# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_router_pressure_microbench.py"
)
SPEC = importlib.util.spec_from_file_location("router_pressure_analyzer", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load analyzer from {MODULE_PATH}")
analyzer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyzer)


def write_sample(root: Path, scenario: str, ns_per_op: list[float]) -> None:
    group, phase, identifier = scenario.split("/", 2)
    sample_dir = root / "criterion" / f"router_pressure_{group}" / phase / identifier / "new"
    sample_dir.mkdir(parents=True)
    iters = [float(index + 1) * 100.0 for index in range(len(ns_per_op))]
    times = [iteration * latency for iteration, latency in zip(iters, ns_per_op)]
    (sample_dir / "sample.json").write_text(
        json.dumps({"sampling_mode": "Linear", "iters": iters, "times": times})
    )


class SampleParsingTests(unittest.TestCase):
    def test_load_run_normalizes_numeric_endpoint_ids_and_percentiles(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_sample(root, "prefill_p2/policy_only/64", [10, 20, 30, 40, 50])

            rows = analyzer.load_run(root)

        row = rows["prefill_p2/policy_only/endpoints=64"]
        self.assertEqual(row.samples, 5)
        self.assertEqual(row.p50_ns, 30)
        self.assertEqual(row.p95_ns, 48)
        self.assertEqual(row.p99_ns, 49.6)
        self.assertAlmostEqual(row.throughput_per_s, 1_000_000_000 / 30)

    def test_load_run_preserves_cache_and_snapshot_identifiers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            write_sample(
                root,
                "cache/request_path/endpoints=256,top_k=32",
                [100, 110, 120],
            )
            write_sample(root, "snapshot/capture/endpoints=256,dp=8", [80, 90, 100])

            rows = analyzer.load_run(root)

        self.assertIn("cache/request_path/endpoints=256,top_k=32", rows)
        self.assertIn("snapshot/capture/endpoints=256,dp=8", rows)

    def test_allocation_log_is_keyed_by_scenario(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "alloc.csv"
            path.write_text(
                "prefill_p2/policy_only/endpoints=64,2.000000,17.000000\n"
                "cache/request_path/endpoints=256,top_k=32,2888.000000,466059.000000\n"
            )

            rows = analyzer.load_allocations(path)

        self.assertEqual(rows["prefill_p2/policy_only/endpoints=64"].allocations, 2)
        self.assertEqual(
            rows["cache/request_path/endpoints=256,top_k=32"].bytes, 466_059
        )

    def test_perf_log_extracts_cpu_validity_and_counters(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "perf.csv"
            path.write_text(
                "54926.64,msec,task-clock,54926642966,100.00,0.966,CPUs utilized\n"
                "211370251273,,cycles,54926642966,100.00,3.848,GHz\n"
                "650528303867,,instructions,54926642966,100.00,3.08,insn per cycle\n"
                "3470982,,cache-misses,54926642966,100.00,,\n"
                "16363,,context-switches,54926642966,100.00,297.906,/sec\n"
                "0,,cpu-migrations,54926642966,100.00,0.000,/sec\n"
            )

            perf = analyzer.load_perf(path)

        self.assertEqual(perf.task_clock_ms, 54_926.64)
        self.assertEqual(perf.cpu_utilized, 0.966)
        self.assertEqual(perf.cycles, 211_370_251_273)
        self.assertEqual(perf.instructions, 650_528_303_867)
        self.assertEqual(perf.cache_misses, 3_470_982)
        self.assertEqual(perf.cpu_migrations, 0)

    def test_perf_validity_rejects_migration_or_non_single_cpu_run(self):
        valid = analyzer.PerfStats(1, 0.98, 1, 1, 1, 1, 0)
        migrated = analyzer.PerfStats(1, 0.98, 1, 1, 1, 1, 2)
        underutilized = analyzer.PerfStats(1, 0.75, 1, 1, 1, 1, 0)

        self.assertTrue(analyzer.evaluate_perf("valid", valid).passed)
        self.assertFalse(analyzer.evaluate_perf("migrated", migrated).passed)
        self.assertFalse(analyzer.evaluate_perf("under", underutilized).passed)

    def test_aggregate_perf_uses_the_median_of_each_counter(self):
        with tempfile.TemporaryDirectory() as raw:
            paths = []
            for index, value in enumerate((10, 30, 20)):
                path = Path(raw) / f"perf-{index}.csv"
                path.write_text(
                    f"{value},msec,task-clock,1,100.00,1.0,CPUs utilized\n"
                    f"{value},,cycles,1,100.00,,\n"
                    f"{value},,instructions,1,100.00,,\n"
                    f"{value},,cache-misses,1,100.00,,\n"
                    f"{value},,context-switches,1,100.00,,\n"
                    "0,,cpu-migrations,1,100.00,,\n"
                )
                paths.append(path)

            perf = analyzer.aggregate_perf(paths)

        self.assertEqual(perf.task_clock_ms, 20)
        self.assertEqual(perf.cycles, 20)
        self.assertEqual(perf.cpu_migrations, 0)


class AggregationTests(unittest.TestCase):
    def test_aggregate_runs_reports_rsd_of_run_level_p95(self):
        runs = [
            {"p": analyzer.SampleStats(30, 10, 20, 100, 0)},
            {"p": analyzer.SampleStats(30, 10, 22, 100, 0)},
            {"p": analyzer.SampleStats(30, 10, 18, 100, 0)},
        ]

        aggregate = analyzer.aggregate_runs(runs)["p"]

        self.assertEqual(aggregate.p95_ns, 20)
        expected_rsd = math.sqrt(((20 - 20) ** 2 + (22 - 20) ** 2 + (18 - 20) ** 2) / 3) / 20
        self.assertAlmostEqual(aggregate.p95_rsd, expected_rsd)

    def test_compare_variants_uses_p95_ratio(self):
        base = {"p": analyzer.AggregateStats(10, 100, 0.01, 1000)}
        candidate = {"p": analyzer.AggregateStats(10, 109, 0.01, 900)}

        comparison = analyzer.compare_variants(base, candidate)

        self.assertAlmostEqual(comparison["p"].p95_ratio, 1.09)


class AcceptanceTests(unittest.TestCase):
    def test_acceptance_flags_regression_absolute_and_scaling_failures(self):
        step2 = {
            "prefill_p2/policy_only/endpoints=64": analyzer.AggregateStats(10, 100, 0.01, 1),
            "prefill_p2/policy_only/endpoints=256": analyzer.AggregateStats(10, 100, 0.01, 1),
        }
        common = {
            "prefill_p2/policy_only/endpoints=64": analyzer.AggregateStats(10, 105, 0.01, 1),
            "prefill_p2/policy_only/endpoints=256": analyzer.AggregateStats(10, 140, 0.01, 1),
        }
        rich = {
            "prefill_p2/policy_only/endpoints=64": analyzer.AggregateStats(10, 100, 0.01, 1),
            "prefill_p2/policy_only/endpoints=256": analyzer.AggregateStats(10, 170, 0.01, 1),
            "snapshot/capture/endpoints=256,dp=1": analyzer.AggregateStats(
                10, 101_000, 0.01, 1
            ),
            "prefill_p2/request_path/endpoints=256": analyzer.AggregateStats(
                10, 151_000, 0.01, 1
            ),
            "session/request_path/endpoints=256": analyzer.AggregateStats(
                10, 149_000, 0.01, 1
            ),
            "decode/request_path/endpoints=256": analyzer.AggregateStats(
                10, 149_000, 0.01, 1
            ),
            "cache/request_path/endpoints=256,top_k=32": analyzer.AggregateStats(
                10, 301_000, 0.01, 1
            ),
        }

        checks = analyzer.evaluate_acceptance(step2, common, rich)
        failed_names = {check.name for check in checks if not check.passed}

        self.assertIn("step3_common_vs_step2/prefill_p2/policy_only/endpoints=256", failed_names)
        self.assertIn("step3_rich_vs_common/prefill_p2/policy_only/endpoints=256", failed_names)
        self.assertIn("absolute/snapshot_256_dp1", failed_names)
        self.assertIn("absolute/prefill_p2_request_256", failed_names)
        self.assertIn("absolute/cache_request_256_top32", failed_names)
        self.assertIn("scaling/prefill_p2_policy_64_to_256", failed_names)

    def test_acceptance_marks_high_rsd_for_confirmation(self):
        rich = {
            "prefill_p2/policy_only/endpoints=64": analyzer.AggregateStats(10, 100, 0.11, 1),
            "prefill_p2/policy_only/endpoints=256": analyzer.AggregateStats(10, 100, 0.01, 1),
        }

        checks = analyzer.evaluate_stability(rich)

        check = next(
            check
            for check in checks
            if check.name == "stability/prefill_p2/policy_only/endpoints=64"
        )
        self.assertFalse(check.passed)
        self.assertIn("confirmation", check.detail)


if __name__ == "__main__":
    unittest.main()
