# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_tracelab_simulator_http_fleet_e2e.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("tracelab_simulator_http_fleet", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TraceLabSimulatorHttpFleetContractTest(unittest.TestCase):
    def test_fixed_256_worker_matrix_has_four_policies_and_three_repeats(self):
        runner = load_runner()

        cases = runner.build_cases(runner.DEFAULT_POLICIES, repeats=3)

        self.assertEqual(runner.WORKER_COUNT, 256)
        self.assertEqual(len(cases), 12)
        self.assertEqual({case.endpoint_count for case in cases}, {256})
        self.assertEqual({case.policy for case in cases}, set(runner.DEFAULT_POLICIES))
        self.assertEqual({case.workload for case in cases}, {"tracelab_session_local"})

    def test_replay_phase_partition_keeps_session_warmup_before_measurements(self):
        runner = load_runner()
        rounds = (
            runner.ReplayRound("a", 0, 1024, 0, 1024, 8, 1, True),
            runner.ReplayRound("a", 1, 1536, 1024, 512, 8, 2, False),
            runner.ReplayRound("a", 2, 2048, 1536, 512, 8, 3, False),
            runner.ReplayRound("a", 3, 2560, 2048, 512, 8, 4, False),
            runner.ReplayRound("b", 4, 1024, 0, 1024, 8, 5, True),
            runner.ReplayRound("b", 5, 1792, 1024, 768, 8, 6, False),
            runner.ReplayRound("b", 6, 2304, 1792, 512, 8, 7, False),
            runner.ReplayRound("b", 7, 2816, 2304, 512, 8, 8, False),
        )

        warmups, measurements = runner.partition_replay_rounds(rounds)

        self.assertEqual([round_.round_index for round_ in warmups["a"]], [0])
        self.assertEqual([round_.round_index for round_ in measurements["a"]], [1, 2, 3])
        self.assertEqual([round_.round_index for round_ in warmups["b"]], [4])
        self.assertEqual([round_.round_index for round_ in measurements["b"]], [5, 6, 7])

    def test_selection_contract_uses_fixed_256_sessions_and_trace_output_geometry(self):
        runner = load_runner()

        selection = runner.default_selection_config()
        round_ = runner.ReplayRound("a", 1, 16000, 12000, 4000, 683, 7, False)

        self.assertEqual(selection.session_count, 256)
        self.assertEqual(runner.max_tokens_for_round(round_, max_total_tokens=32768), 683)

    def test_runner_rejects_any_worker_count_other_than_256(self):
        runner = load_runner()

        with self.assertRaises(SystemExit):
            runner.parse_args(["--worker-count", "128"])

    def test_native_cache_policy_scales_indexer_query_concurrency_to_fleet_size(self):
        runner = load_runner()

        arguments = runner.policy_args("cache_aware", "http://127.0.0.1:50551")

        position = arguments.index("--kv-indexer-query-max-inflight")
        self.assertEqual(arguments[position + 1], "256")

    def test_router_command_serializes_integer_timeout_flags(self):
        runner = load_runner()
        args = SimpleNamespace(
            router_binary=Path("/router"),
            router_port=30480,
            model_path=Path("/model"),
            tokenizer_path=Path("/tokenizer"),
            request_timeout_seconds=360.0,
            stale_request_timeout_seconds=420.0,
        )
        case = runner.TraceLabCase(policy="power_of_two", repeat=0)

        command = runner.router_command(
            args,
            case,
            ("http://127.0.0.1:17000",),
            "http://127.0.0.1:50581",
        )

        timeout_index = command.index("--request-timeout-secs")
        stale_timeout_index = command.index("--stale-request-timeout-secs")
        self.assertEqual(command[timeout_index + 1], "360")
        self.assertEqual(command[stale_timeout_index + 1], "420")


if __name__ == "__main__":
    unittest.main()
