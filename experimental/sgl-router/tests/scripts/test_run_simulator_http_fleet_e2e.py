# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_simulator_http_fleet_e2e.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("simulator_http_fleet", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runner from {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SimulatorHttpFleetContractTest(unittest.TestCase):
    def test_health_probe_retries_a_single_client_timeout_within_global_window(self):
        runner = load_runner()

        class SlowThenReadyHandler(BaseHTTPRequestHandler):
            request_count = 0

            def do_GET(self):
                type(self).request_count += 1
                if type(self).request_count == 1:
                    time.sleep(3.1)
                self.send_response(200)
                self.end_headers()
                try:
                    self.wfile.write(b"ok")
                except BrokenPipeError:
                    pass

            def log_message(self, _format, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), SlowThenReadyHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            asyncio.run(
                runner.wait_http_urls(
                    (f"http://127.0.0.1:{server.server_port}/health",), timeout=5.0
                )
            )
        finally:
            server.shutdown()
            thread.join(timeout=2.0)
            server.server_close()

        self.assertGreaterEqual(SlowThenReadyHandler.request_count, 2)

    def test_default_matrix_covers_requested_endpoint_counts_and_policies(self):
        runner = load_runner()

        self.assertEqual(runner.DEFAULT_ENDPOINT_COUNTS, (32, 128, 256, 512, 1024))
        self.assertEqual(
            runner.DEFAULT_POLICIES,
            ("power_of_two", "cache_aware", "cache_aware_zmq", "shortest_ttft"),
        )
        cases = runner.build_cases(
            endpoint_counts=runner.DEFAULT_ENDPOINT_COUNTS,
            policies=runner.DEFAULT_POLICIES,
            workloads=("tracelab_multiturn", "multi_holder_pressure"),
            repeats=3,
        )
        self.assertEqual(len(cases), 120)
        self.assertEqual({case.endpoint_count for case in cases}, set(runner.DEFAULT_ENDPOINT_COUNTS))
        self.assertEqual({case.policy for case in cases}, set(runner.DEFAULT_POLICIES))

    def test_simulator_worker_command_is_cpu_blocking_and_uses_unique_ports(self):
        runner = load_runner()
        first = runner.worker_spec(
            0,
            http_base_port=31_000,
            kv_base_port=51_000,
            dist_base_port=53_000,
        )
        second = runner.worker_spec(
            1,
            http_base_port=31_000,
            kv_base_port=51_000,
            dist_base_port=53_000,
        )

        command, environment = runner.simulator_worker_command(
            first,
            python="python3",
            simulator_config=Path("/sim/replay.json"),
            model_path=Path("/sim/model"),
            tokenizer_path=Path("/sim/tokenizer"),
            max_total_tokens=8192,
            max_running_requests=32,
        )

        self.assertNotEqual(first.http_port, second.http_port)
        self.assertNotEqual(first.kv_port, second.kv_port)
        self.assertNotEqual(first.load_port, second.load_port)
        self.assertEqual(first.load_port, first.kv_port + 1)
        self.assertNotEqual(first.dist_port, second.dist_port)
        self.assertIn("sglang_simulator.simulation.sglang.launch_server", command)
        self.assertIn("--enable-cache-report", command)
        self.assertIn("--enable-metrics", command)
        self.assertIn("--load-publish-endpoint", command)
        self.assertEqual(command[command.index("--load-publish-endpoint") + 1], "auto")
        self.assertEqual(
            command[command.index("--load-snapshot-publish-interval") + 1], "1"
        )
        self.assertIn("--dist-init-addr", command)
        self.assertIn("127.0.0.1:53000", command)
        self.assertIn("--chat-template", command)
        self.assertIn("chatml", command)
        self.assertEqual(environment["SGLANG_USE_CPU_ENGINE"], "1")
        self.assertEqual(environment["SGLANG_SIMULATOR_OUTPUT_MODE"], "BLOCKING")

    def test_auto_port_layout_skips_a_candidate_with_an_occupied_worker_port(self):
        runner = load_runner()
        blocked_port = runner.DEFAULT_PORT_LAYOUTS[0].kv_base_port + 18

        layout = runner.select_available_port_layout(
            endpoint_count=1_024,
            candidates=runner.DEFAULT_PORT_LAYOUTS,
            reserved_ports=(30_380, 50_551),
            port_is_available=lambda port: port != blocked_port,
        )

        self.assertEqual(layout, runner.DEFAULT_PORT_LAYOUTS[1])

    def test_256_worker_tier_falls_back_when_a_primary_port_is_occupied(self):
        runner = load_runner()
        candidates = runner.auto_port_layout_candidates(256)
        blocked_port = candidates[0].kv_base_port + 2 * 79 + 1

        layout = runner.select_available_port_layout(
            endpoint_count=256,
            candidates=candidates,
            reserved_ports=(30_380, 50_551),
            port_is_available=lambda port: port != blocked_port,
        )

        self.assertEqual(layout, candidates[1])

    def test_port_layout_waits_for_a_previous_fleet_socket_to_be_released(self):
        runner = load_runner()
        checks = 0
        sleeps = []

        def port_is_available(_port):
            nonlocal checks
            checks += 1
            return checks > 1

        layout = runner.wait_for_available_port_layout(
            endpoint_count=4,
            candidates=(runner.DEFAULT_PORT_LAYOUTS[0],),
            reserved_ports=(),
            timeout=1.0,
            port_is_available=port_is_available,
            sleep=lambda duration: sleeps.append(duration),
            monotonic=iter((0.0, 0.0)).__next__,
        )

        self.assertEqual(layout, runner.DEFAULT_PORT_LAYOUTS[0])
        self.assertEqual(sleeps, [0.5])

    def test_default_tier_layouts_do_not_reuse_ports_or_enter_ephemeral_range(self):
        runner = load_runner()
        used_ports = set()

        for endpoint_count in runner.DEFAULT_ENDPOINT_COUNTS:
            layout = runner.auto_port_layout_candidates(endpoint_count)[0]
            ports = set(layout.ports(endpoint_count))

            self.assertLess(max(ports), 32_768)
            self.assertFalse(used_ports.intersection(ports))
            used_ports.update(ports)

    def test_cache_aware_audit_rejects_missing_indexer_or_fresh_monitor(self):
        runner = load_runner()
        good = {
            "cache_candidate_decisions": 1,
            "monitor_decisions": 1,
            "router_local_decisions": 0,
            "zero_snapshot_decisions": 0,
            "actual_cache_metrics": 1,
        }
        runner.require_native_cache_audit(good, expected_decisions=1)

        for key in ("cache_candidate_decisions", "monitor_decisions", "actual_cache_metrics"):
            broken = dict(good)
            broken[key] = 0
            with self.assertRaisesRegex(RuntimeError, key):
                runner.require_native_cache_audit(broken, expected_decisions=1)

    def test_cache_aware_audit_requires_fresh_monitor_for_p2_fallback(self):
        runner = load_runner()
        audit = {
            "cache_candidate_decisions": 1,
            "monitor_decisions": 1,
            "router_local_decisions": 0,
            "zero_snapshot_decisions": 0,
            "actual_cache_metrics": 1,
            "fallback_power_of_two_decisions": 1,
            "fallback_power_of_two_proposals": 1,
            "fallback_monitor_decisions": 1,
            "fallback_router_local_decisions": 0,
            "fallback_zero_snapshot_decisions": 0,
        }
        runner.require_native_cache_audit(audit, expected_decisions=2)

        audit["fallback_monitor_decisions"] = 0
        with self.assertRaisesRegex(RuntimeError, "fallback has no fresh LoadMonitor"):
            runner.require_native_cache_audit(audit, expected_decisions=2)

    def test_policy_arguments_keep_native_indexer_and_monitor_paths_distinct(self):
        runner = load_runner()

        native = runner.policy_args("cache_aware", "http://127.0.0.1:50551")
        shortest = runner.policy_args("shortest_ttft", "http://127.0.0.1:50551")
        zmq = runner.policy_args("cache_aware_zmq", "http://127.0.0.1:50551")
        power_of_two = runner.policy_args("power_of_two", "http://127.0.0.1:50551")

        self.assertIn("--kv-indexer-endpoint", native)
        self.assertIn("--kv-indexer-endpoint", shortest)
        self.assertNotIn("--kv-indexer-endpoint", zmq)
        self.assertNotIn("--kv-indexer-endpoint", power_of_two)
        self.assertEqual(power_of_two, ["--policy", "power_of_two"])

    def test_native_policy_arguments_accept_a_guard_tuning_profile(self):
        runner = load_runner()

        tuning = runner.CacheAwareTuning(
            min_matched_tokens=2_048,
            candidate_min_workers=4,
            candidate_ratio=0.125,
            candidate_max_workers=64,
            switch_margin_tokens=8_192,
            pressure_abs_threshold_tokens=512,
            pressure_abs_threshold_ms=75.0,
            pressure_rel_threshold=1.2,
        )
        arguments = runner.policy_args(
            "cache_aware",
            "http://127.0.0.1:50551",
            cache_aware_tuning=tuning,
        )

        def value(flag):
            return arguments[arguments.index(flag) + 1]

        self.assertEqual(value("--cache-affinity-min-matched-tokens"), "2048")
        self.assertEqual(value("--cache-candidate-max-workers"), "64")
        self.assertEqual(value("--cache-switch-margin-tokens"), "8192")
        self.assertEqual(value("--pressure-abs-threshold-ms"), "75.0")
        self.assertEqual(value("--pressure-rel-threshold"), "1.2")

    def test_native_policy_arguments_can_enable_a_global_p2_backup(self):
        runner = load_runner()
        tuning = runner.CacheAwareTuning(
            min_matched_tokens=1_024,
            candidate_min_workers=2,
            candidate_ratio=0.05,
            candidate_max_workers=32,
            switch_margin_tokens=1_024,
            pressure_abs_threshold_tokens=8_192,
            pressure_abs_threshold_ms=None,
            pressure_rel_threshold=1.5,
            global_backup=True,
        )

        arguments = runner.policy_args(
            "cache_aware",
            "http://127.0.0.1:50551",
            cache_aware_tuning=tuning,
        )

        self.assertIn("--cache-aware-global-backup", arguments)

    def test_execution_contract_hashes_the_actual_runner_interpreter_and_config(self):
        runner = load_runner()

        contract = runner.execution_artifact_contract(
            runner_script=SCRIPT,
            python=sys.executable,
            simulator_config=SCRIPT,
            simulator_dependency_root=SCRIPT.parent,
            argv=("--policies", "cache_aware"),
        )

        self.assertEqual(contract["runner_script_sha256"], runner.sha256_file(SCRIPT))
        self.assertEqual(contract["simulator_config_sha256"], runner.sha256_file(SCRIPT))
        self.assertEqual(contract["simulator_dependency_root_sha256"], runner.sha256_tree(SCRIPT.parent))
        self.assertEqual(contract["runner_argv"], ["--policies", "cache_aware"])
        self.assertTrue(Path(contract["python_executable"]).is_file())

    def test_cache_aware_control_summary_reports_admission_and_guard_deltas(self):
        runner = load_runner()
        before = "\n".join(
            (
                "sgl_router_cache_admission_evaluated_total 4",
                "sgl_router_cache_admission_rejected_total 3",
                "sgl_router_cache_pressure_guard_compared_total 7",
                "sgl_router_cache_pressure_guard_override_total 2",
            )
        )
        after = "\n".join(
            (
                "sgl_router_cache_admission_evaluated_total 16",
                "sgl_router_cache_admission_rejected_total 8",
                "sgl_router_cache_pressure_guard_compared_total 19",
                "sgl_router_cache_pressure_guard_override_total 6",
            )
        )

        self.assertEqual(
            runner.cache_aware_control_summary(before, after),
            {
                "admission_evaluated_candidates": 12,
                "admission_rejected_candidates": 5,
                "pressure_guard_compared_pairs": 12,
                "pressure_guard_overrides": 4,
            },
        )

    def test_indexer_query_limit_tracks_the_runner_concurrency_bound(self):
        runner = load_runner()

        self.assertEqual(runner.indexer_query_concurrency(32), 32)
        self.assertEqual(runner.indexer_query_concurrency(128), 128)
        self.assertEqual(runner.indexer_query_concurrency(256), 256)
        self.assertEqual(runner.indexer_query_concurrency(512), 256)
        self.assertEqual(runner.indexer_query_concurrency(1024), 256)

        native = runner.policy_args(
            "cache_aware",
            "http://127.0.0.1:50551",
            indexer_query_max_inflight=256,
        )
        position = native.index("--kv-indexer-query-max-inflight")
        self.assertEqual(native[position + 1], "256")

    def test_router_command_forwards_native_indexer_timeout(self):
        runner = load_runner()
        args = runner.parse_args(["--kv-indexer-query-timeout-ms", "10000"])
        args.router_binary = Path("/router")
        args.router_port = 30_480
        args.model_path = Path("/model")
        args.tokenizer_path = Path("/tokenizer")

        command = runner.router_command(
            args,
            runner.Case(
                endpoint_count=256,
                policy="cache_aware",
                workload="multi_holder_pressure",
                repeat=0,
            ),
            ("http://127.0.0.1:17000",),
            "http://127.0.0.1:50551",
        )

        position = command.index("--kv-indexer-query-timeout-ms")
        self.assertEqual(command[position + 1], "10000")

    def test_worker_environment_keeps_the_simulator_runtime_isolated(self):
        runner = load_runner()

        environment = runner.simulator_environment(
            simulator_site=Path("/sim/site"),
            simulator_dependency_root=Path("/sim/deps"),
            source_root=Path("/sim/source"),
            simulator_config=Path("/sim/replay.json"),
        )

        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["SGLANG_USE_CPU_ENGINE"], "1")
        self.assertEqual(environment["SGLANG_SIMULATOR_OUTPUT_MODE"], "BLOCKING")
        self.assertTrue(environment["PYTHONPATH"].startswith("/sim/site:"))
        self.assertIn("/sim/deps", environment["PYTHONPATH"])

    def test_zero_cache_holders_selects_router_seeded_cache(self):
        runner = load_runner()

        args = runner.parse_args(
            ["--endpoint-counts", "4", "--max-cache-holders", "0"]
        )

        self.assertEqual(args.max_cache_holders, 0)

    def test_runner_rejects_worker_fleet_reuse(self):
        runner = load_runner()

        with self.assertRaises(SystemExit):
            runner.parse_args(["--endpoint-counts", "128", "--reuse-worker-fleet"])

    def test_worker_cache_flush_uses_each_worker_control_endpoint(self):
        runner = load_runner()

        self.assertEqual(
            runner.worker_cache_flush_urls(
                ("http://127.0.0.1:31000", "http://127.0.0.1:31001")
            ),
            ("http://127.0.0.1:31000/flush_cache", "http://127.0.0.1:31001/flush_cache"),
        )

    def test_cache_audit_reads_ansi_colored_router_logs(self):
        runner = load_runner()
        log = (
            "cache candidate winner "
            "prefill_pressure_source\x1b[0m\x1b[2m=\x1b[0m\"estimated_prefill_queue_ms\" "
            "load_snapshot_version\x1b[0m\x1b[2m=\x1b[0m42\n"
        )

        self.assertEqual(
            runner.cache_monitor_usage(log),
            {
                "cache_candidate_decisions": 1,
                "monitor_decisions": 1,
                "router_local_decisions": 0,
                "zero_snapshot_decisions": 0,
                "actual_cache_metrics": 0,
            },
        )

    def test_power_of_two_audit_requires_fresh_snapshot_for_every_pair(self):
        runner = load_runner()
        log = (
            'prefill policy decision policy=PowerOfTwo prefill_pressure_source="estimated_prefill_queue_ms" load_snapshot_version=41\n'
            'prefill policy decision policy=PowerOfTwo prefill_pressure_source="estimated_prefill_queue_ms" load_snapshot_version=42\n'
        )

        audit = runner.power_of_two_monitor_usage(log)

        self.assertEqual(
            audit,
            {
                "power_of_two_decisions": 2,
                "monitor_decisions": 2,
                "router_local_decisions": 0,
                "zero_snapshot_decisions": 0,
            },
        )
        runner.require_power_of_two_audit(audit, expected_decisions=2)

        native_queue = runner.power_of_two_monitor_usage(
            'prefill policy decision policy=PowerOfTwo prefill_pressure_source="native_queue_tokens" load_snapshot_version=43\n'
        )
        runner.require_power_of_two_audit(native_queue, expected_decisions=1)

        broken = dict(audit)
        broken["zero_snapshot_decisions"] = 1
        with self.assertRaisesRegex(RuntimeError, "zero snapshot"):
            runner.require_power_of_two_audit(broken, expected_decisions=2)

        local = runner.power_of_two_monitor_usage(
            'prefill policy decision policy=PowerOfTwo prefill_pressure_source="router_local" load_snapshot_version=42\n'
        )
        with self.assertRaisesRegex(RuntimeError, "fresh monitor"):
            runner.require_power_of_two_audit(local, expected_decisions=1)

    def test_prefill_power_of_two_audit_counts_final_decisions_only(self):
        runner = load_runner()
        log = (
            "prefill policy decision policy=PowerOfTwo selected=http://worker-a\n"
            "cache candidate winner policy=CacheAffinity selected=http://worker-b\n"
            "prefill policy decision policy=SessionAffinity selected=http://worker-c\n"
        )

        self.assertEqual(runner.prefill_power_of_two_decisions(log), 1)

    def test_zmq_audit_accounts_for_every_selection_path(self):
        runner = load_runner()
        log = (
            "cache-aware-zmq: load-balance check considered engine_load_workers=256 engine_load_expected=256\n"
            "cache-aware-zmq match_prefix matched_blocks=32 match_rate=0.8\n"
            "cache-aware-zmq: selected worker by cache overlap\n"
            "cache-aware-zmq: load-balance check considered engine_load_workers=256 engine_load_expected=256\n"
            "cache-aware-zmq match_prefix matched_blocks=4 match_rate=0.1\n"
            "cache-aware-zmq: overlap below threshold, falling back to min-load\n"
            "cache-aware-zmq: load-balance check considered engine_load_workers=256 engine_load_expected=256\n"
            "cache-aware-zmq: load imbalance fast-path\n"
        )

        audit = runner.zmq_policy_usage(log)

        self.assertEqual(
            audit,
            {
                "load_balance_checks": 3,
                "complete_engine_load_checks": 3,
                "incomplete_engine_load_checks": 0,
                "lookup_decisions": 2,
                "cache_holder_selections": 1,
                "threshold_fallbacks": 1,
                "load_imbalance_fallbacks": 1,
                "block_size_fallbacks": 0,
            },
        )
        runner.require_zmq_policy_audit(audit, expected_decisions=3)

        broken = dict(audit)
        broken["cache_holder_selections"] = 0
        with self.assertRaisesRegex(RuntimeError, "lookup outcomes"):
            runner.require_zmq_policy_audit(broken, expected_decisions=3)

    def test_shortest_ttft_audit_requires_fresh_monitor_for_every_winner(self):
        runner = load_runner()
        log = (
            "shortest TTFT candidate winner "
            "prefill_pressure_source=\"estimated_prefill_queue_ms\" "
            "load_snapshot_version=41\n"
            "shortest TTFT candidate winner "
            "prefill_pressure_source\x1b[0m\x1b[2m=\x1b[0m\"estimated_prefill_queue_ms\" "
            "load_snapshot_version=42\n"
        )

        audit = runner.shortest_ttft_monitor_usage(log)

        self.assertEqual(
            audit,
            {
                "shortest_ttft_decisions": 2,
                "monitor_decisions": 2,
                "monitor_fallback_decisions": 0,
                "router_local_decisions": 0,
                "zero_snapshot_decisions": 0,
            },
        )
        runner.require_shortest_ttft_audit(audit, expected_decisions=2)

    def test_shortest_ttft_audit_rejects_missing_monitor_evidence(self):
        runner = load_runner()
        good = {
            "shortest_ttft_decisions": 1,
            "monitor_decisions": 1,
            "monitor_fallback_decisions": 0,
            "router_local_decisions": 0,
            "zero_snapshot_decisions": 0,
        }
        runner.require_shortest_ttft_audit(good, expected_decisions=1)

        for key in ("shortest_ttft_decisions", "monitor_decisions"):
            broken = dict(good)
            broken[key] = 0
            with self.assertRaisesRegex(RuntimeError, key):
                runner.require_shortest_ttft_audit(broken, expected_decisions=1)
        for key in ("router_local_decisions", "zero_snapshot_decisions"):
            broken = dict(good)
            broken[key] = 1
            with self.assertRaisesRegex(RuntimeError, key):
                runner.require_shortest_ttft_audit(broken, expected_decisions=1)

    def test_shortest_ttft_audit_rejects_winner_without_load_monitor_fields(self):
        runner = load_runner()

        with self.assertRaisesRegex(RuntimeError, "load-monitor audit fields"):
            runner.shortest_ttft_monitor_usage("shortest TTFT candidate winner\n")

    def test_completed_case_records_zero_request_errors(self):
        runner = load_runner()

        summary = runner.summarize_case(
            [{"ttft_ms": 1.0, "e2e_ms": 2.0, "completion_tokens": 3}],
            elapsed_seconds=1.0,
            cache={"hit_rate": 1.0},
            worker_success={"http://127.0.0.1:31000": 1.0},
            worker_urls=("http://127.0.0.1:31000",),
            policy_reasons={"primary": 1.0},
        )

        self.assertEqual(summary["request_errors"], 0)

    def test_resume_archives_incomplete_case_without_deleting_its_evidence(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "case"
            directory.mkdir()
            (directory / "case.json").write_text("{}\n")
            (directory / "router.log").write_text("partial evidence\n")

            archived = runner.archive_incomplete_case(directory)

            self.assertFalse(directory.exists())
            self.assertEqual(archived.name, "case.attempt-0")
            self.assertEqual((archived / "router.log").read_text(), "partial evidence\n")
            self.assertTrue((archived / "RESUME_ARCHIVED").is_file())

    def test_worker_startup_batches_bound_concurrent_initialization(self):
        runner = load_runner()

        self.assertEqual(
            tuple(runner.batched((0, 1, 2, 3, 4), 2)),
            ((0, 1), (2, 3), (4,)),
        )

    def test_reusable_fleet_groups_cases_by_endpoint_count(self):
        runner = load_runner()
        cases = runner.build_cases(
            endpoint_counts=(4, 8),
            policies=("power_of_two",),
            workloads=("tracelab_multiturn",),
            repeats=2,
        )

        groups = runner.group_cases_by_endpoint_count(cases)

        self.assertEqual(tuple(endpoint_count for endpoint_count, _ in groups), (4, 8))
        self.assertEqual(tuple(len(group) for _, group in groups), (2, 2))

    def test_case_runner_invokes_a_fresh_case_boundary_each_time(self):
        runner = load_runner()
        cases = runner.build_cases(
            endpoint_counts=(4, 8),
            policies=("power_of_two",),
            workloads=("tracelab_multiturn",),
            repeats=1,
        )
        sleeps = []
        started = []
        original_run = runner.run_case
        original_sleep = runner.time.sleep
        try:
            runner.run_case = lambda _args, case: started.append(case.name)
            runner.time.sleep = lambda seconds: sleeps.append(seconds)
            with tempfile.TemporaryDirectory() as temporary:
                runner.run_cases(
                    SimpleNamespace(
                        results_dir=Path(temporary),
                        control_plane_quiesce_seconds=16.0,
                    ),
                    cases,
                )
        finally:
            runner.run_case = original_run
            runner.time.sleep = original_sleep

        self.assertEqual(started, [case.name for case in cases])
        self.assertEqual(sleeps, [16.0, 16.0])


if __name__ == "__main__":
    unittest.main()
