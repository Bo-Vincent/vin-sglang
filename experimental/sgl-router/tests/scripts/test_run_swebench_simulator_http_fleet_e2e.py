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
    / "run_swebench_simulator_http_fleet_e2e.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("swebench_simulator", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SWEbenchSimulatorHttpFleetContractTest(unittest.TestCase):
    def test_fixed_256_worker_matrix_has_five_policies_and_three_repeats(self):
        runner = load_runner()

        cases = runner.build_cases(runner.DEFAULT_POLICIES, repeats=3)

        self.assertEqual(runner.WORKER_COUNT, 256)
        self.assertEqual(len(cases), 15)
        self.assertEqual({case.endpoint_count for case in cases}, {256})
        self.assertEqual({case.policy for case in cases}, set(runner.DEFAULT_POLICIES))
        self.assertIn("original_shortest_ttft", runner.DEFAULT_POLICIES)
        self.assertEqual({case.workload for case in cases}, {"swebench_prompt_shape"})

    def test_replay_request_uses_unique_task_routing_key(self):
        runner = load_runner()
        task = runner.SWEbenchTask(
            instance_id="task-1",
            repo="acme/project",
            base_commit="abc",
            version="1.0",
            problem_statement="Fix it.",
            source_index=0,
        )

        request = runner.replay_request(task, context_repetitions=2)

        self.assertEqual(request["routing_key"], "task-1")
        self.assertIn("acme/project", request["prompt"])
        self.assertEqual(request["task_index"], 0)

    def test_runner_rejects_any_worker_count_other_than_256(self):
        runner = load_runner()

        with self.assertRaises(SystemExit):
            runner.parse_args(["--worker-count", "128"])

    def test_runner_defaults_to_public_prompt_shape_contract(self):
        runner = load_runner()

        arguments = runner.parse_args([])

        self.assertEqual(arguments.request_rate, 64.0)
        self.assertEqual(arguments.output_tokens, 64)
        self.assertEqual(arguments.repeats, 3)
        self.assertEqual(arguments.worker_page_size, 1)
        self.assertIsNone(arguments.task_limit)
        self.assertFalse(hasattr(arguments, "indexer_bridge"))
        self.assertGreaterEqual(
            runner.DEFAULT_AGENT_CONTEXT_REPETITIONS,
            16,
        )
        self.assertEqual(runner.CACHE_AWARE_MIN_MATCHED_TOKENS, 1024)
        self.assertFalse(arguments.execute)

    def test_runtime_probe_checks_the_simulator_import_boundary(self):
        runner = load_runner()

        command = runner.simulator_runtime_probe_command("/python")

        self.assertEqual(command[:2], ["/python", "-c"])
        self.assertIn("aiconfigurator.sdk.models", command[2])
        self.assertIn("transformers.image_processing_backends", command[2])

    def test_native_cache_measurement_records_guard_and_admission_controls(self):
        runner = load_runner()
        before = "\n".join(
            (
                "sgl_router_cache_admission_evaluated_total 4",
                "sgl_router_cache_admission_rejected_total 2",
                "sgl_router_cache_pressure_guard_compared_total 3",
                "sgl_router_cache_pressure_guard_override_total 1",
            )
        )
        after = "\n".join(
            (
                "sgl_router_cache_admission_evaluated_total 16",
                "sgl_router_cache_admission_rejected_total 7",
                "sgl_router_cache_pressure_guard_compared_total 13",
                "sgl_router_cache_pressure_guard_override_total 5",
            )
        )

        self.assertEqual(
            runner.measurement_policy_controls("cache_aware", before, after),
            {
                "admission_evaluated_candidates": 12,
                "admission_rejected_candidates": 5,
                "pressure_guard_compared_pairs": 10,
                "pressure_guard_overrides": 4,
            },
        )
        self.assertIsNone(
            runner.measurement_policy_controls("power_of_two", before, after)
        )

    def test_complete_case_requires_the_recorded_uniform_warmup(self):
        runner = load_runner()
        case = runner.SWEbenchCase(policy="cache_aware", repeat=0)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / case.name
            directory.mkdir()
            (directory / "COMPLETE").write_text("ok\n")
            (directory / "case.json").write_text(json.dumps(runner.asdict(case)))
            (directory / "summary.json").write_text("{}\n")
            (directory / "requests.jsonl").write_text("{}\n")

            self.assertFalse(runner.case_complete(directory, case))

            (directory / "warmup_requests.jsonl").write_text("{}\n")
            self.assertTrue(runner.case_complete(directory, case))


if __name__ == "__main__":
    unittest.main()
