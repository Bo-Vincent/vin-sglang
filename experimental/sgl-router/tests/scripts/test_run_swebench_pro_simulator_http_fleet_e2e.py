# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_swebench_pro_simulator_http_fleet_e2e.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("swebench_pro_simulator", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SWEbenchProSimulatorHttpFleetContractTest(unittest.TestCase):
    def test_fixed_256_worker_matrix_has_four_policies_and_three_repeats(self):
        runner = load_runner()

        cases = runner.build_cases(runner.DEFAULT_POLICIES, repeats=3)

        self.assertEqual(runner.WORKER_COUNT, 256)
        self.assertEqual(len(cases), 12)
        self.assertEqual({case.endpoint_count for case in cases}, {256})
        self.assertEqual({case.policy for case in cases}, set(runner.DEFAULT_POLICIES))
        self.assertEqual({case.workload for case in cases}, {"swebench_pro_prompt_shape"})

    def test_replay_request_uses_unique_task_routing_key(self):
        runner = load_runner()
        task = runner.SWEbenchProTask(
            instance_id="task-1",
            repo="acme/project",
            base_commit="abc",
            problem_statement="Fix it.",
            requirements="Keep compatibility.",
            interface="src/main.py",
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
        self.assertTrue(arguments.require_indexer_success)
        self.assertGreaterEqual(
            runner.DEFAULT_AGENT_CONTEXT_REPETITIONS,
            16,
        )
        self.assertEqual(runner.CACHE_AWARE_MIN_MATCHED_TOKENS, 1024)
        self.assertFalse(arguments.execute)

    def test_indexer_success_is_required_unless_explicitly_disabled_for_diagnosis(self):
        runner = load_runner()

        self.assertTrue(runner.parse_args([]).require_indexer_success)
        self.assertFalse(
            runner.parse_args(["--no-require-indexer-success"]).require_indexer_success
        )

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


if __name__ == "__main__":
    unittest.main()
