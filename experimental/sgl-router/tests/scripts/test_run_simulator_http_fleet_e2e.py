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
            reporter_base_port=47_000,
            kv_base_port=51_000,
            dist_base_port=53_000,
        )
        second = runner.worker_spec(
            1,
            http_base_port=31_000,
            reporter_base_port=47_000,
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
        self.assertNotEqual(first.reporter_port, second.reporter_port)
        self.assertNotEqual(first.kv_port, second.kv_port)
        self.assertNotEqual(first.dist_port, second.dist_port)
        self.assertIn("sglang_simulator.simulation.sglang.launch_server", command)
        self.assertIn("--enable-cache-report", command)
        self.assertIn("--enable-metrics", command)
        self.assertIn("--load-reporter-port", command)
        self.assertIn("--dist-init-addr", command)
        self.assertIn("127.0.0.1:53000", command)
        self.assertIn("--chat-template", command)
        self.assertIn("chatml", command)
        self.assertEqual(environment["SGLANG_USE_CPU_ENGINE"], "1")
        self.assertEqual(environment["SGLANG_SIMULATOR_OUTPUT_MODE"], "BLOCKING")

    def test_auto_port_layout_skips_a_candidate_with_an_occupied_worker_port(self):
        runner = load_runner()
        blocked_port = runner.DEFAULT_PORT_LAYOUTS[0].reporter_base_port + 18

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
        blocked_port = candidates[0].reporter_base_port + 79

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
        runner.require_native_cache_audit(good)

        for key in ("cache_candidate_decisions", "monitor_decisions", "actual_cache_metrics"):
            broken = dict(good)
            broken[key] = 0
            with self.assertRaisesRegex(RuntimeError, key):
                runner.require_native_cache_audit(broken)

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

    def test_worker_environment_keeps_the_simulator_runtime_isolated(self):
        runner = load_runner()

        environment = runner.simulator_environment(
            simulator_site=Path("/sim/site"),
            source_root=Path("/sim/source"),
            simulator_config=Path("/sim/replay.json"),
        )

        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(environment["SGLANG_USE_CPU_ENGINE"], "1")
        self.assertEqual(environment["SGLANG_SIMULATOR_OUTPUT_MODE"], "BLOCKING")
        self.assertTrue(environment["PYTHONPATH"].startswith("/sim/site:"))

    def test_zero_cache_holders_selects_router_seeded_cache(self):
        runner = load_runner()

        args = runner.parse_args(
            ["--endpoint-counts", "4", "--max-cache-holders", "0"]
        )

        self.assertEqual(args.max_cache_holders, 0)

    def test_reused_worker_fleet_quiesces_router_reporter_leases_between_cases(self):
        runner = load_runner()

        args = runner.parse_args(["--endpoint-counts", "128", "--reuse-worker-fleet"])

        self.assertEqual(args.control_plane_quiesce_seconds, 16.0)

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
            "prefill_pressure_source\x1b[0m\x1b[2m=\x1b[0m\"monitor_fallback\" "
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

    def test_reused_fleet_quiesces_before_reusing_worker_reporter_ports(self):
        runner = load_runner()
        cases = runner.build_cases(
            endpoint_counts=(4, 8),
            policies=("power_of_two",),
            workloads=("tracelab_multiturn",),
            repeats=1,
        )
        sleeps = []
        started = []
        stopped = []
        original_start = runner.start_worker_fleet
        original_run = runner.run_case
        original_stop = runner.stop_processes
        original_sleep = runner.time.sleep
        try:
            runner.start_worker_fleet = lambda _args, count, _directory: (
                [count],
                [f"fleet-{count}"],
            )
            runner.run_case = lambda _args, case, *, worker_fleet: started.append(
                (case.endpoint_count, tuple(worker_fleet[0]))
            )
            runner.stop_processes = lambda processes: stopped.append(tuple(processes))
            runner.time.sleep = lambda seconds: sleeps.append(seconds)
            with tempfile.TemporaryDirectory() as temporary:
                runner.run_cases(
                    SimpleNamespace(
                        resume=False,
                        reuse_worker_fleet=True,
                        results_dir=Path(temporary),
                        control_plane_quiesce_seconds=16.0,
                    ),
                    cases,
                )
        finally:
            runner.start_worker_fleet = original_start
            runner.run_case = original_run
            runner.stop_processes = original_stop
            runner.time.sleep = original_sleep

        self.assertEqual(started, [(4, (4,)), (8, (8,))])
        self.assertEqual(stopped, [("fleet-4",), ("fleet-8",)])
        self.assertEqual(sleeps, [16.0])


if __name__ == "__main__":
    unittest.main()
