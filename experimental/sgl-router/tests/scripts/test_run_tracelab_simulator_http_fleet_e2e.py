# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import tempfile
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
    def test_measurement_audit_uses_last_policy_decisions_when_router_log_flushes_late(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            router_log = Path(directory) / "router.log"
            router_log.write_text(
                "\n".join(
                    ["prefill policy decision policy=PowerOfTwo warmup"] * 2
                    + ["prefill policy decision policy=PowerOfTwo measurement"] * 3
                )
                + "\n"
            )
            decisions = runner.measurement_decision_log(
                router_log, policy_marker="policy=PowerOfTwo", expected_decisions=3
            )
        self.assertEqual(decisions.count("measurement"), 3)
        self.assertNotIn("warmup", decisions)

    def test_native_cache_audit_uses_measurement_candidate_delta(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            router_log = Path(directory) / "router.log"
            router_log.write_text(
                "\n".join(
                    ["cache candidate winner warmup"] * 2
                    + ["cache candidate winner measurement"] * 3
                )
                + "\n"
            )

            decisions = runner.cache_candidate_audit_log(
                router_log, {"cache_candidate": 3.0}
            )

        self.assertEqual(decisions.count("measurement"), 3)
        self.assertNotIn("warmup", decisions)

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

    def test_native_cache_policy_uses_explicit_indexer_budget(self):
        runner = load_runner()

        arguments = runner.policy_args(
            "cache_aware",
            "http://127.0.0.1:50551",
            query_timeout_ms=10_000,
            query_max_inflight=256,
        )

        timeout_position = arguments.index("--kv-indexer-query-timeout-ms")
        inflight_position = arguments.index("--kv-indexer-query-max-inflight")
        self.assertEqual(arguments[timeout_position + 1], "10000")
        self.assertEqual(arguments[inflight_position + 1], "256")

    def test_runner_defaults_to_audited_indexer_budget(self):
        runner = load_runner()

        arguments = runner.parse_args([])

        self.assertEqual(arguments.kv_indexer_query_timeout_ms, 10_000)
        self.assertEqual(arguments.kv_indexer_query_max_inflight, 256)
        self.assertEqual(arguments.kv_indexer_max_concurrent_streams, 512)
        self.assertEqual(arguments.warmup_request_rate, 1.0)
        self.assertEqual(arguments.pressure_guard_seed_holders, 2)
        self.assertEqual(arguments.pressure_guard_seed_request_rate, 64.0)
        self.assertEqual(arguments.indexer_drain_quiet_seconds, 5.0)
        self.assertFalse(arguments.require_indexer_success)

    def test_pressure_guard_seed_assigns_two_distinct_replicas_per_session(self):
        runner = load_runner()

        targets = runner.pressure_guard_seed_targets(
            ("session-a", "session-b", "session-c"),
            ("http://worker-0", "http://worker-1", "http://worker-2", "http://worker-3"),
            holders_per_session=2,
        )

        self.assertEqual(
            targets["session-a"], ("http://worker-0", "http://worker-1")
        )
        self.assertEqual(
            targets["session-b"], ("http://worker-2", "http://worker-3")
        )
        self.assertEqual(
            targets["session-c"], ("http://worker-0", "http://worker-1")
        )
        self.assertTrue(all(len(set(urls)) == 2 for urls in targets.values()))

    def test_pressure_guard_seed_uses_one_deterministic_warmup_session(self):
        runner = load_runner()
        warmups = {
            "session-c": ("c-warmup",),
            "session-a": ("a-warmup",),
            "session-b": ("b-warmup",),
        }

        selected = runner.pressure_guard_seed_warmups(warmups)

        self.assertEqual(selected, {"session-a": ("a-warmup",)})

    def test_runner_rejects_non_positive_warmup_request_rate(self):
        runner = load_runner()

        with self.assertRaises(SystemExit):
            runner.parse_args(["--warmup-request-rate", "0"])

    def test_indexer_bridge_drain_requires_a_quiet_successful_apply_window(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            bridge_log = Path(directory) / "bridge.log"
            bridge_log.write_text(runner.BRIDGE_APPLY_LOG_MARKER + "\n")
            clock = iter((0.0, 0.0, 5.0))

            result = runner.wait_for_indexer_bridge_drain(
                (bridge_log,),
                quiet_seconds=5.0,
                timeout_seconds=10.0,
                poll_seconds=0.1,
                sleep=lambda _: None,
                monotonic=lambda: next(clock),
            )

        self.assertEqual(result["applied_batches"], 1)
        self.assertEqual(result["bridge_failures"], 0)

    def test_indexer_bridge_drain_rejects_bridge_reconnect(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            bridge_log = Path(directory) / "bridge.log"
            bridge_log.write_text(runner.BRIDGE_FAILURE_LOG_MARKER + "\n")

            with self.assertRaisesRegex(RuntimeError, "bridge failure"):
                runner.wait_for_indexer_bridge_drain(
                    (bridge_log,),
                    quiet_seconds=1.0,
                    timeout_seconds=2.0,
                    poll_seconds=0.1,
                    sleep=lambda _: None,
                    monotonic=lambda: 0.0,
                )

    def test_indexer_query_summary_reports_latency_and_failures(self):
        runner = load_runner()
        before = ""
        after = "\n".join(
            [
                'sgl_router_kv_indexer_query_duration_seconds_bucket{model_id="m",outcome="success",le="0.01"} 1',
                'sgl_router_kv_indexer_query_duration_seconds_bucket{model_id="m",outcome="success",le="0.1"} 3',
                'sgl_router_kv_indexer_query_duration_seconds_bucket{model_id="m",outcome="success",le="+Inf"} 3',
                'sgl_router_kv_indexer_query_duration_seconds_sum{model_id="m",outcome="success"} 0.13',
                'sgl_router_kv_indexer_query_duration_seconds_count{model_id="m",outcome="success"} 3',
                'sgl_router_kv_indexer_query_duration_seconds_bucket{model_id="m",outcome="timeout",le="2"} 1',
                'sgl_router_kv_indexer_query_duration_seconds_bucket{model_id="m",outcome="timeout",le="+Inf"} 1',
                'sgl_router_kv_indexer_query_duration_seconds_sum{model_id="m",outcome="timeout"} 2.0',
                'sgl_router_kv_indexer_query_duration_seconds_count{model_id="m",outcome="timeout"} 1',
            ]
        )

        summary = runner.indexer_query_summary(before, after)

        self.assertEqual(summary["query_count"], 4)
        self.assertEqual(summary["success_count"], 3)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["outcomes"]["success"]["p95_ms"], 100.0)
        self.assertEqual(summary["outcomes"]["timeout"]["mean_ms"], 2_000.0)

    def test_indexer_query_success_gate_rejects_any_fail_open_outcome(self):
        runner = load_runner()

        with self.assertRaisesRegex(RuntimeError, "failed"):
            runner.require_indexer_query_success(
                {
                    "query_count": 256,
                    "success_count": 255,
                    "failure_count": 1,
                    "outcomes": {"success": {"count": 255}, "timeout": {"count": 1}},
                },
                expected_queries=256,
                phase="warmup",
            )

    def test_zmq_lookup_summary_distinguishes_matched_and_empty(self):
        runner = load_runner()
        after = "\n".join(
            [
                'sgl_router_zmq_prefix_lookup_duration_seconds_bucket{model_id="m",outcome="matched",le="0.001"} 1',
                'sgl_router_zmq_prefix_lookup_duration_seconds_bucket{model_id="m",outcome="matched",le="+Inf"} 1',
                'sgl_router_zmq_prefix_lookup_duration_seconds_sum{model_id="m",outcome="matched"} 0.001',
                'sgl_router_zmq_prefix_lookup_duration_seconds_count{model_id="m",outcome="matched"} 1',
                'sgl_router_zmq_prefix_lookup_duration_seconds_bucket{model_id="m",outcome="empty",le="0.0001"} 1',
                'sgl_router_zmq_prefix_lookup_duration_seconds_bucket{model_id="m",outcome="empty",le="+Inf"} 1',
                'sgl_router_zmq_prefix_lookup_duration_seconds_sum{model_id="m",outcome="empty"} 0.0001',
                'sgl_router_zmq_prefix_lookup_duration_seconds_count{model_id="m",outcome="empty"} 1',
            ]
        )

        summary = runner.zmq_prefix_lookup_summary("", after)

        self.assertEqual(summary["lookup_count"], 2)
        self.assertEqual(summary["matched_count"], 1)
        self.assertEqual(summary["empty_count"], 1)
        self.assertEqual(summary["outcomes"]["matched"]["p95_ms"], 1.0)

    def test_router_command_serializes_integer_timeout_flags(self):
        runner = load_runner()
        args = SimpleNamespace(
            router_binary=Path("/router"),
            router_port=30480,
            model_path=Path("/model"),
            tokenizer_path=Path("/tokenizer"),
            request_timeout_seconds=360.0,
            stale_request_timeout_seconds=420.0,
            kv_indexer_query_timeout_ms=2_000,
            kv_indexer_query_max_inflight=32,
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
