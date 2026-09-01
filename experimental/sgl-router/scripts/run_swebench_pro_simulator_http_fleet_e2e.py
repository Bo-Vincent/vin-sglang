#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""在 256 个 HTTP Simulator worker 上重放公开 SWE-bench Pro task prompt。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_simulator_http_fleet_e2e as fleet
import run_tracelab_simulator_http_fleet_e2e as tracelab
from swebench_pro_replay import (
    DEFAULT_AGENT_CONTEXT_REPETITIONS,
    SWEbenchProTask,
    agent_prefix,
    build_replay_manifest,
    build_task_prompt,
    fetch_public_test_rows,
    load_dataset_cache,
    select_tasks,
    sha256_file as dataset_sha256_file,
    write_dataset_cache,
)


WORKER_COUNT = 256
DEFAULT_REQUEST_RATE = 64.0
DEFAULT_OUTPUT_TOKENS = 64
DEFAULT_POLICIES = fleet.DEFAULT_POLICIES
CACHE_AWARE_MIN_MATCHED_TOKENS = 1024


@dataclass(frozen=True)
class SWEbenchCase:
    policy: str
    repeat: int
    endpoint_count: int = WORKER_COUNT
    workload: str = "swebench_pro_prompt_shape"

    @property
    def name(self) -> str:
        return f"swebench-pro-{self.endpoint_count}w-{self.policy}-r{self.repeat}"


def parse_csv(value: str) -> tuple[str, ...]:
    return fleet.parse_csv(value)


def build_cases(policies: Sequence[str], *, repeats: int) -> tuple[SWEbenchCase, ...]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return tuple(
        SWEbenchCase(policy=policy, repeat=repeat)
        for policy in policies
        for repeat in range(repeats)
    )


def measurement_policy_controls(
    policy: str, router_before: str, router_after: str
) -> dict[str, int] | None:
    if policy != "cache_aware":
        return None
    return fleet.cache_aware_control_summary(router_before, router_after)


def replay_request(
    task: SWEbenchProTask, *, context_repetitions: int = DEFAULT_AGENT_CONTEXT_REPETITIONS
) -> dict[str, object]:
    return {
        "task_index": task.source_index,
        "instance_id": task.instance_id,
        "repo": task.repo,
        "routing_key": task.instance_id,
        "prompt": build_task_prompt(task, context_repetitions=context_repetitions),
    }


def simulator_runtime_probe_command(python: str) -> list[str]:
    return fleet.simulator_runtime_probe_command(python)


def validate_simulator_runtime(args: argparse.Namespace) -> None:
    """兼容 SWE-bench runner 的现有入口，实际复用共享的运行时预检。"""
    fleet.validate_simulator_runtime(args)


def validate_agent_context_prefix(
    tasks: Sequence[SWEbenchProTask],
    *,
    tokenizer_path: Path,
    context_repetitions: int,
    require_cache_candidate: bool,
) -> dict[str, int]:
    """确认固定共享前缀自身足够进入 Native Cache-Aware 的候选域。"""
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError("SWE-bench Pro Simulator runner requires tokenizers") from error
    tokenizer_file = tokenizer_path / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise RuntimeError(f"tokenizer.json does not exist: {tokenizer_file}")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    counts = {
        task.repo: len(tokenizer.encode(agent_prefix(task.repo, context_repetitions=context_repetitions)).ids)
        for task in tasks
    }
    minimum = min(counts.values())
    maximum = max(counts.values())
    if require_cache_candidate and minimum < CACHE_AWARE_MIN_MATCHED_TOKENS:
        raise RuntimeError(
            "SWE-bench Pro agent prefix is below Native Cache-Aware's matched-token "
            f"threshold: min={minimum} required={CACHE_AWARE_MIN_MATCHED_TOKENS}"
        )
    return {"min": minimum, "max": maximum}


async def replay_tasks(
    tasks: Sequence[SWEbenchProTask],
    *,
    router_url: str,
    model: str,
    request_rate: float,
    output_tokens: int,
    timeout_seconds: float,
    max_total_tokens: int,
    context_repetitions: int,
) -> list[dict[str, object]]:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("SWE-bench Pro Simulator runner requires aiohttp") from error
    requests = [replay_request(task, context_repetitions=context_repetitions) for task in tasks]
    gate = fleet.RateGate(request_rate)
    semaphore = asyncio.Semaphore(fleet.MAX_HTTP_REQUEST_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def send(index: int, task: SWEbenchProTask) -> dict[str, object]:
            await gate.wait_turn()
            async with semaphore:
                request = requests[index]
                round_ = tracelab.ReplayRound(
                    session_id=task.instance_id,
                    round_index=0,
                    input_tokens=0,
                    prefix_tokens=0,
                    newly_append_tokens=0,
                    output_tokens=output_tokens,
                    source_line=task.source_index,
                    is_warmup=False,
                )
                result = await tracelab.stream_chat_request(
                    session=session,
                    router_url=router_url,
                    model=model,
                    prompt=str(request["prompt"]),
                    round_=round_,
                    timeout_seconds=timeout_seconds,
                    max_total_tokens=max_total_tokens,
                )
                result.update(
                    {
                        "task_index": task.source_index,
                        "instance_id": task.instance_id,
                        "repo": task.repo,
                    }
                )
                return result

        return list(await asyncio.gather(*(send(index, task) for index, task in enumerate(tasks))))


def case_complete(directory: Path, case: SWEbenchCase) -> bool:
    try:
        return (
            (directory / "COMPLETE").read_text().strip() == "ok"
            and json.loads((directory / "case.json").read_text()) == asdict(case)
            and (directory / "summary.json").stat().st_size > 0
            and (directory / "requests.jsonl").stat().st_size > 0
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False


def run_case(
    args: argparse.Namespace,
    case: SWEbenchCase,
    tasks: Sequence[SWEbenchProTask],
    worker_fleet: tuple[Sequence[fleet.WorkerSpec], Sequence[fleet.ManagedProcess]],
) -> None:
    directory = args.results_dir / case.name
    if directory.exists():
        if args.resume and case_complete(directory, case):
            print(f"skip complete {case.name}", flush=True)
            return
        if not args.resume:
            raise FileExistsError(f"incomplete or incompatible case directory: {directory}")
        archived = fleet.archive_incomplete_case(directory)
        print(f"preserved incomplete {case.name} at {archived.name}; retrying", flush=True)
    directory.mkdir(parents=True, exist_ok=False)
    fleet.atomic_write_text(
        directory / "case.json", json.dumps(asdict(case), indent=2, sort_keys=True) + "\n"
    )
    fleet.atomic_write_text(args.results_dir / "CURRENT", case.name + "\n")

    specs, workers = worker_fleet
    managed: list[fleet.ManagedProcess] = []
    try:
        fleet.ensure_worker_fleet_healthy(specs, workers)
        managed = tracelab.start_case_control_plane(args, case, directory, specs)
        worker_urls = [spec.url for spec in specs]
        asyncio.run(fleet.flush_worker_caches(worker_urls))
        time.sleep(args.control_plane_quiesce_seconds)

        worker_before = asyncio.run(fleet.fetch_texts(tuple(f"{url}/metrics" for url in worker_urls)))
        router_before = asyncio.run(
            fleet.fetch_texts((f"http://127.0.0.1:{args.router_port}/metrics",))
        )[0]
        router_log = directory / "router.log"
        log_offset = router_log.stat().st_size
        fleet.atomic_write_text(directory / "router.measurement_before.prom", router_before)

        started = time.monotonic()
        requests = asyncio.run(
            replay_tasks(
                tasks,
                router_url=f"http://127.0.0.1:{args.router_port}",
                model=str(args.model_path),
                request_rate=args.request_rate,
                output_tokens=args.output_tokens,
                timeout_seconds=args.request_timeout_seconds,
                max_total_tokens=args.max_total_tokens,
                context_repetitions=args.agent_context_repetitions,
            )
        )
        elapsed_seconds = time.monotonic() - started
        if len(requests) != len(tasks):
            raise RuntimeError(f"{case.name} measured {len(requests)} requests, expected {len(tasks)}")

        worker_after = asyncio.run(fleet.fetch_texts(tuple(f"{url}/metrics" for url in worker_urls)))
        router_after = asyncio.run(
            fleet.fetch_texts((f"http://127.0.0.1:{args.router_port}/metrics",))
        )[0]
        cache = fleet.prefill_cache_summary(worker_before, worker_after)
        worker_success = fleet.metric_delta(
            fleet.worker_success_counts(router_before), fleet.worker_success_counts(router_after)
        )
        cache_aware_controls = measurement_policy_controls(
            case.policy, router_before, router_after
        )
        if cache_aware_controls is not None:
            fleet.require_cache_aware_controls(cache_aware_controls)
        reasons = fleet.metric_delta(
            fleet.policy_reason_counts(router_before, case.policy),
            fleet.policy_reason_counts(router_after, case.policy),
        )
        zmq_lookup = tracelab.zmq_prefix_lookup_summary(router_before, router_after)

        audit: dict[str, int] | None = None
        shortest_ttft_audit: dict[str, int] | None = None
        power_of_two_audit: dict[str, int] | None = None
        zmq_policy_audit: dict[str, int] | None = None
        decision_log = router_log.read_text(errors="replace")[log_offset:]
        if case.policy == "cache_aware":
            audit = fleet.cache_monitor_usage(decision_log)
            audit["actual_cache_metrics"] = int(float(cache["total_effective_tokens"]) > 0.0)
            fallback_audit = fleet.power_of_two_monitor_usage(decision_log)
            audit.update(
                {
                    "fallback_power_of_two_decisions": fleet.prefill_power_of_two_decisions(
                        decision_log
                    ),
                    "fallback_power_of_two_proposals": fallback_audit[
                        "power_of_two_decisions"
                    ],
                    "fallback_monitor_decisions": fallback_audit["monitor_decisions"],
                    "fallback_router_local_decisions": fallback_audit[
                        "router_local_decisions"
                    ],
                    "fallback_zero_snapshot_decisions": fallback_audit[
                        "zero_snapshot_decisions"
                    ],
                }
            )
            fleet.require_native_cache_audit(audit, expected_decisions=len(tasks))
        elif case.policy == "shortest_ttft":
            shortest_ttft_audit = fleet.shortest_ttft_monitor_usage(decision_log)
            fleet.require_shortest_ttft_audit(
                shortest_ttft_audit, expected_decisions=len(tasks)
            )
        elif case.policy == "power_of_two":
            power_of_two_audit = fleet.power_of_two_monitor_usage(decision_log)
            fleet.require_power_of_two_audit(
                power_of_two_audit, expected_decisions=len(tasks)
            )
        elif case.policy == "cache_aware_zmq":
            zmq_policy_audit = fleet.zmq_policy_usage(decision_log)
            fleet.require_zmq_policy_audit(zmq_policy_audit, expected_decisions=len(tasks))
        summary = fleet.summarize_case(
            requests,
            elapsed_seconds=elapsed_seconds,
            cache=cache,
            worker_success=worker_success,
            worker_urls=worker_urls,
            policy_reasons=reasons,
        )
        summary["native_cache_audit"] = audit
        summary["cache_aware_controls"] = cache_aware_controls
        summary["shortest_ttft_audit"] = shortest_ttft_audit
        summary["power_of_two_audit"] = power_of_two_audit
        summary["zmq_policy_audit"] = zmq_policy_audit
        summary["zmq_prefix_lookup"] = zmq_lookup if case.policy == "cache_aware_zmq" else None
        summary["swebench_pro_measurement_count"] = len(tasks)
        fleet.atomic_write_text(
            directory / "requests.jsonl",
            "".join(json.dumps(request, sort_keys=True) + "\n" for request in requests),
        )
        fleet.atomic_write_text(directory / "router.prom", router_after)
        fleet.atomic_write_text(
            directory / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        fleet.atomic_write_text(directory / "COMPLETE", "ok\n")
    finally:
        fleet.stop_processes(managed)
        fleet.wait_for_control_plane_quiescence(args)


def run_cases(
    args: argparse.Namespace, cases: Sequence[SWEbenchCase], tasks: Sequence[SWEbenchProTask]
) -> None:
    pending = [case for case in cases if not (args.resume and case_complete(args.results_dir / case.name, case))]
    if not pending:
        return
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] run {case.name}", flush=True)
        fleet_directory = args.results_dir / "worker-fleets" / case.name
        specs, workers = fleet.start_worker_fleet(args, WORKER_COUNT, fleet_directory)
        try:
            run_case(args, case, tasks, (specs, workers))
        finally:
            fleet.stop_processes(workers)
            fleet.wait_for_control_plane_quiescence(args)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--router-binary", type=Path)
    parser.add_argument("--router-cwd", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--simulator-site", type=Path)
    parser.add_argument("--simulator-dependency-root", type=Path)
    parser.add_argument("--simulator-config", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--agent-context-repetitions", type=int, default=DEFAULT_AGENT_CONTEXT_REPETITIONS)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--worker-count", type=int, default=WORKER_COUNT)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--request-rate", type=float, default=DEFAULT_REQUEST_RATE)
    fleet.add_cache_aware_tuning_arguments(parser)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--http-base-port", type=int)
    parser.add_argument("--kv-base-port", type=int)
    parser.add_argument("--dist-base-port", type=int)
    parser.add_argument("--router-port", type=int, default=30_380)
    parser.add_argument("--max-total-tokens", type=int, default=32_768)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--worker-page-size", type=int, default=1)
    parser.add_argument("--worker-start-batch-size", type=int, default=16)
    parser.add_argument("--worker-port-layout-wait-timeout", type=float, default=90.0)
    parser.add_argument("--worker-start-timeout", type=float, default=300.0)
    parser.add_argument("--router-start-timeout", type=float, default=180.0)
    parser.add_argument("--router-settle-seconds", type=float, default=35.0)
    parser.add_argument("--control-plane-quiesce-seconds", type=float, default=16.0)
    parser.add_argument("--request-timeout-seconds", type=int, default=360)
    parser.add_argument("--stale-request-timeout-seconds", type=int, default=420)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.policies = parse_csv(args.policies)
    unsupported = set(args.policies) - set(DEFAULT_POLICIES)
    if unsupported:
        parser.error(f"unsupported policies: {sorted(unsupported)}")
    if args.worker_count != WORKER_COUNT:
        parser.error(f"--worker-count must be {WORKER_COUNT} for this contract")
    if args.repeats <= 0 or args.request_rate <= 0.0 or args.output_tokens <= 0:
        parser.error("--repeats, --request-rate, and --output-tokens must be positive")
    if args.task_limit is not None and args.task_limit <= 0:
        parser.error("--task-limit must be positive")
    if args.agent_context_repetitions <= 0:
        parser.error("--agent-context-repetitions must be positive")
    if (
        args.max_total_tokens <= 0
        or args.max_running_requests <= 0
        or args.worker_page_size <= 0
        or args.worker_start_batch_size <= 0
        or args.request_timeout_seconds <= 0
        or args.stale_request_timeout_seconds <= 0
    ):
        parser.error("capacity and timeout arguments must be positive")
    try:
        args.cache_aware_tuning = fleet.cache_aware_tuning_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        configured = fleet.configured_port_layout(args)
    except ValueError as error:
        parser.error(str(error))
    if configured is not None and any(port > 65_535 for port in configured.ports(WORKER_COUNT)):
        parser.error("every worker port range must fit in 1..65535")
    if args.execute:
        required = (
            "source_root", "router_binary", "router_cwd",
            "python", "simulator_site", "simulator_dependency_root", "simulator_config", "model_path", "tokenizer_path",
            "dataset_cache", "results_dir",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error("--execute requires: " + ", ".join("--" + name.replace("_", "-") for name in missing))
        for name in required:
            value = getattr(args, name)
            if isinstance(value, Path):
                setattr(args, name, value.resolve())
    return args


def create_or_verify_replay_manifest(
    results_dir: Path,
    tasks: Sequence[SWEbenchProTask],
    dataset_cache: Path,
    *,
    context_repetitions: int,
) -> Path:
    path = results_dir / "swebench-pro-replay-manifest.json"
    expected = build_replay_manifest(tasks, dataset_cache, context_repetitions=context_repetitions)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid SWE-bench Pro replay manifest: {path}") from error
        if existing != expected:
            raise RuntimeError(f"SWE-bench Pro replay manifest mismatch: {path}")
    else:
        fleet.atomic_write_text(path, json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: Iterable[str] | None = None) -> int:
    invocation = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    args = parse_args(invocation)
    cases = build_cases(args.policies, repeats=args.repeats)
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "worker_count": WORKER_COUNT, "cases": [asdict(case) | {"name": case.name} for case in cases]}, indent=2, sort_keys=True))
        return 0
    assert args.dataset_cache is not None
    assert args.results_dir is not None
    if args.download_dataset:
        write_dataset_cache(args.dataset_cache, fetch_public_test_rows(task_limit=args.task_limit))
    validate_simulator_runtime(args)
    tasks = select_tasks(load_dataset_cache(args.dataset_cache), task_limit=args.task_limit)
    agent_prefix_tokens = validate_agent_context_prefix(
        tasks,
        tokenizer_path=args.tokenizer_path,
        context_repetitions=args.agent_context_repetitions,
        require_cache_candidate="cache_aware" in args.policies,
    )
    replay_manifest = create_or_verify_replay_manifest(
        args.results_dir,
        tasks,
        args.dataset_cache,
        context_repetitions=args.agent_context_repetitions,
    )
    contract = {
        "schema_version": 2,
        "source_commit": fleet.read_source_commit(args.source_root),
        "router_binary_sha256": fleet.sha256_file(args.router_binary),
        "worker_count": WORKER_COUNT,
        "policies": list(args.policies),
        "repeats": args.repeats,
        "request_rate": args.request_rate,
        "output_tokens": args.output_tokens,
        "max_total_tokens": args.max_total_tokens,
        "max_running_requests": args.max_running_requests,
        "worker_page_size": args.worker_page_size,
        "dataset_cache": str(args.dataset_cache),
        "dataset_cache_sha256": dataset_sha256_file(args.dataset_cache),
        "task_count": len(tasks),
        "agent_context_repetitions": args.agent_context_repetitions,
        "agent_prefix_tokens": agent_prefix_tokens,
        "replay_manifest_sha256": fleet.sha256_file(replay_manifest),
        "measurement_kind": "simulator_predicted_relative",
        "control_plane_quiesce_seconds": args.control_plane_quiesce_seconds,
        "case_isolation": {
            "worker_lifecycle": "fresh_per_case",
            "cache_reset": "flush_before_measurement",
            "control_plane": "stop_all_then_quiesce",
        },
        "cache_aware_tuning": asdict(args.cache_aware_tuning),
        "execution_artifacts": fleet.execution_artifact_contract(
            runner_script=Path(__file__),
            python=args.python,
            simulator_config=args.simulator_config,
            simulator_dependency_root=args.simulator_dependency_root,
            simulator_runtime_python_root=args.source_root / "python",
            argv=invocation,
        ),
        "python": args.python,
        "simulator_pythonpath": os.environ.get("PYTHONPATH", ""),
        "simulator_site": str(args.simulator_site),
        "simulator_dependency_root": str(args.simulator_dependency_root),
        "simulator_config": str(args.simulator_config),
        "cases": [asdict(case) | {"name": case.name} for case in cases],
    }
    fleet.write_or_verify_manifest(args.results_dir, contract, resume=args.resume)
    run_cases(args, cases, tasks)
    fleet.atomic_write_text(args.results_dir / "RUN_COMPLETE", "ok\n")
    fleet.atomic_write_text(args.results_dir / "CURRENT", "complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
