# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_simulator_http_fleet_e2e.py"
)


def load_analyzer():
    spec = importlib.util.spec_from_file_location("simulator_http_analyzer", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import analyzer from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SimulatorHttpFleetAnalyzerTest(unittest.TestCase):
    def create_result(self, root: Path, *, repeats: int = 3, errors: int = 0) -> None:
        root.mkdir(parents=True, exist_ok=True)
        cases = []
        for repeat in range(repeats):
            case = {
                "endpoint_count": 32,
                "policy": "cache_aware",
                "workload": "tracelab_multiturn",
                "repeat": repeat,
            }
            cases.append(case | {"name": f"case-{repeat}"})
            directory = root / f"case-{repeat}"
            directory.mkdir()
            (directory / "case.json").write_text(json.dumps(case))
            (directory / "COMPLETE").write_text("ok\n")
            (directory / "summary.json").write_text(
                json.dumps(
                    {
                        "request_count": 32,
                        "request_errors": errors,
                        "throughput_rps": 8.0 + repeat,
                        "completion_tps": 128.0 + repeat,
                        "ttft_ms": {"mean": 10.0 + repeat, "p95": 12.0 + repeat},
                        "e2e_ms": {"mean": 40.0 + repeat, "p95": 45.0 + repeat},
                        "cache": {"hit_rate": 0.8, "hit_tokens": 800, "total_effective_tokens": 1000},
                        "worker_cv": 0.2,
                        "policy_reasons": {"cache_candidate": 32.0},
                        "native_cache_audit": {
                            "cache_candidate_decisions": 32,
                            "monitor_decisions": 32,
                            "router_local_decisions": 0,
                            "zero_snapshot_decisions": 0,
                            "actual_cache_metrics": 1,
                            "fallback_power_of_two_decisions": 0,
                            "fallback_power_of_two_proposals": 0,
                            "fallback_monitor_decisions": 0,
                            "fallback_router_local_decisions": 0,
                            "fallback_zero_snapshot_decisions": 0,
                        },
                    }
                )
            )
        (root / "manifest.json").write_text(
            json.dumps({"contract": {"repeats": repeats, "cases": cases}})
        )
        (root / "RUN_COMPLETE").write_text("ok\n")

    def test_analyzer_groups_three_repeats_by_endpoint_policy_and_workload(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_result(root)

            analysis = analyzer.analyze_results(root)

        self.assertEqual(len(analysis["groups"]), 1)
        group = analysis["groups"][0]
        self.assertEqual(group["endpoint_count"], 32)
        self.assertEqual(group["policy"], "cache_aware")
        self.assertEqual(group["workload"], "tracelab_multiturn")
        self.assertEqual(group["repeat_count"], 3)
        self.assertEqual(group["metrics"]["throughput_rps"]["median"], 9.0)

    def test_analyzer_rejects_missing_or_nonzero_request_errors(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_result(root, errors=1)
            with self.assertRaisesRegex(RuntimeError, "request_errors"):
                analyzer.analyze_results(root)

            self.create_result(root / "incomplete")
            (root / "incomplete" / "case-2" / "COMPLETE").unlink()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                analyzer.analyze_results(root / "incomplete")

    def test_analyzer_reports_ttft_e2e_tps_hit_rate_cv_and_policy_reasons(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_result(root)

            group = analyzer.analyze_results(root)["groups"][0]

        self.assertEqual(
            set(group["metrics"]),
            {
                "throughput_rps",
                "completion_tps",
                "ttft_mean_ms",
                "ttft_p95_ms",
                "e2e_mean_ms",
                "e2e_p95_ms",
                "cache_hit_rate",
                "worker_cv",
            },
        )
        self.assertEqual(group["policy_reasons"], {"cache_candidate": 96.0})
        self.assertEqual(group["native_cache_audit"]["cache_candidate_decisions"], 96)

    def test_analyzer_marks_simulator_values_as_predicted_not_gpu_measurements(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_result(root)

            analysis = analyzer.analyze_results(root)

        self.assertEqual(analysis["measurement_kind"], "simulator_predicted_relative")
        self.assertIn("not GPU", analysis["measurement_notice"])

    def test_analyzer_marks_zmq_reason_as_not_emitted(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_result(root)
            manifest = json.loads((root / "manifest.json").read_text())
            for expected in manifest["contract"]["cases"]:
                expected["policy"] = "cache_aware_zmq"
                directory = root / expected["name"]
                case = json.loads((directory / "case.json").read_text())
                case["policy"] = "cache_aware_zmq"
                (directory / "case.json").write_text(json.dumps(case))
                summary = json.loads((directory / "summary.json").read_text())
                summary["policy_reasons"] = {}
                summary["zmq_policy_audit"] = {
                    "lookup_decisions": 32,
                    "cache_holder_selections": 32,
                    "threshold_fallbacks": 0,
                    "load_imbalance_fallbacks": 0,
                    "block_size_fallbacks": 0,
                }
                (directory / "summary.json").write_text(json.dumps(summary))
            (root / "manifest.json").write_text(json.dumps(manifest))

            group = analyzer.analyze_results(root)["groups"][0]

        self.assertEqual(group["policy_reasons"], {})
        self.assertEqual(group["policy_reason_observability"], "not_emitted_by_policy")

    def test_analyzer_requires_and_aggregates_power_of_two_monitor_audit(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_result(root)
            manifest = json.loads((root / "manifest.json").read_text())
            for expected in manifest["contract"]["cases"]:
                expected["policy"] = "power_of_two"
                directory = root / expected["name"]
                case = json.loads((directory / "case.json").read_text())
                case["policy"] = "power_of_two"
                (directory / "case.json").write_text(json.dumps(case))
                summary = json.loads((directory / "summary.json").read_text())
                summary["policy_reasons"] = {"primary": 32.0}
                summary["power_of_two_audit"] = {
                    "power_of_two_decisions": 32,
                    "monitor_decisions": 32,
                    "router_local_decisions": 0,
                    "zero_snapshot_decisions": 0,
                }
                (directory / "summary.json").write_text(json.dumps(summary))
            (root / "manifest.json").write_text(json.dumps(manifest))

            group = analyzer.analyze_results(root)["groups"][0]

        self.assertEqual(group["power_of_two_audit"]["power_of_two_decisions"], 96)
        self.assertEqual(group["power_of_two_audit"]["monitor_decisions"], 96)

    def test_analyzer_requires_and_aggregates_zmq_path_audit(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_result(root)
            manifest = json.loads((root / "manifest.json").read_text())
            for expected in manifest["contract"]["cases"]:
                expected["policy"] = "cache_aware_zmq"
                directory = root / expected["name"]
                case = json.loads((directory / "case.json").read_text())
                case["policy"] = "cache_aware_zmq"
                (directory / "case.json").write_text(json.dumps(case))
                summary = json.loads((directory / "summary.json").read_text())
                summary["policy_reasons"] = {}
                summary["zmq_policy_audit"] = {
                    "lookup_decisions": 32,
                    "cache_holder_selections": 32,
                    "threshold_fallbacks": 0,
                    "load_imbalance_fallbacks": 0,
                    "block_size_fallbacks": 0,
                }
                (directory / "summary.json").write_text(json.dumps(summary))
            (root / "manifest.json").write_text(json.dumps(manifest))

            group = analyzer.analyze_results(root)["groups"][0]

        self.assertEqual(group["zmq_policy_audit"]["lookup_decisions"], 96)
        self.assertEqual(group["zmq_policy_audit"]["cache_holder_selections"], 96)

    def test_analyzer_requires_and_aggregates_shortest_ttft_monitor_audit(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_result(root)
            manifest = json.loads((root / "manifest.json").read_text())
            for expected in manifest["contract"]["cases"]:
                expected["policy"] = "shortest_ttft"
                directory = root / expected["name"]
                case = json.loads((directory / "case.json").read_text())
                case["policy"] = "shortest_ttft"
                (directory / "case.json").write_text(json.dumps(case))
                summary = json.loads((directory / "summary.json").read_text())
                summary["policy_reasons"] = {"shortest_ttft": 32.0}
                summary["shortest_ttft_audit"] = {
                    "shortest_ttft_decisions": 32,
                    "monitor_decisions": 32,
                    "monitor_fallback_decisions": 8,
                    "native_admission_guard_covered_decisions": 32,
                    "router_local_decisions": 0,
                    "zero_snapshot_decisions": 0,
                    "admission_evaluated_candidates": 64,
                    "admission_rejected_candidates": 4,
                    "outstanding_guard_evaluated_candidates": 60,
                    "outstanding_guard_rejected_candidates": 2,
                }
                (directory / "summary.json").write_text(json.dumps(summary))
            (root / "manifest.json").write_text(json.dumps(manifest))

            group = analyzer.analyze_results(root)["groups"][0]

        self.assertEqual(group["shortest_ttft_audit"]["shortest_ttft_decisions"], 96)
        self.assertEqual(group["shortest_ttft_audit"]["monitor_decisions"], 96)
        self.assertEqual(group["shortest_ttft_audit"]["monitor_fallback_decisions"], 24)
        self.assertEqual(group["shortest_ttft_audit"]["native_admission_guard_covered_decisions"], 96)

    def test_analyzer_rejects_shortest_ttft_without_fresh_monitor_audit(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.create_result(root, repeats=1)
            manifest = json.loads((root / "manifest.json").read_text())
            expected = manifest["contract"]["cases"][0]
            expected["policy"] = "shortest_ttft"
            directory = root / expected["name"]
            case = json.loads((directory / "case.json").read_text())
            case["policy"] = "shortest_ttft"
            (directory / "case.json").write_text(json.dumps(case))
            summary = json.loads((directory / "summary.json").read_text())
            summary["policy_reasons"] = {"shortest_ttft": 32.0}
            summary["shortest_ttft_audit"] = {
                "shortest_ttft_decisions": 32,
                "monitor_decisions": 31,
                "monitor_fallback_decisions": 0,
                "native_admission_guard_covered_decisions": 32,
                "router_local_decisions": 1,
                "zero_snapshot_decisions": 0,
                "admission_evaluated_candidates": 64,
                "admission_rejected_candidates": 4,
                "outstanding_guard_evaluated_candidates": 60,
                "outstanding_guard_rejected_candidates": 2,
            }
            (directory / "summary.json").write_text(json.dumps(summary))
            (root / "manifest.json").write_text(json.dumps(manifest))

            with self.assertRaisesRegex(RuntimeError, "Shortest-TTFT"):
                analyzer.analyze_results(root)

    def test_analyzer_combines_confirmation_repeats_without_replacing_primary(self):
        analyzer = load_analyzer()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "primary"
            confirmation = root / "confirmation"
            self.create_result(primary, repeats=3)
            self.create_result(confirmation, repeats=2)

            group = analyzer.analyze_results(
                primary, confirmation_results_dir=confirmation
            )["groups"][0]

        self.assertEqual(group["repeat_count"], 5)
        self.assertEqual(group["primary_repeat_count"], 3)
        self.assertEqual(group["confirmation_repeat_count"], 2)
        self.assertEqual(group["native_cache_audit"]["cache_candidate_decisions"], 160)


if __name__ == "__main__":
    unittest.main()
