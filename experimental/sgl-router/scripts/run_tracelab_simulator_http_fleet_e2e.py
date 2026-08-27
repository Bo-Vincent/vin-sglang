# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""在固定 256 个 HTTP Simulator worker 上重放 TraceLab session slice。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_simulator_http_fleet_e2e as fleet
from tracelab_replay import (
    ReplayRound,
    ReplaySelectionConfig,
    VirtualTokenPlan,
    build_virtual_token_plan,
    find_reversible_token_ids,
    load_trace_rows,
    reconstruct_virtual_prompt,
    select_replay_slice,
    sha256_file as replay_sha256_file,
    write_replay_manifest,
)


WORKER_COUNT = 256
ROUNDS_PER_SESSION = 4
DEFAULT_REQUEST_RATE = 64.0
DEFAULT_POLICIES = fleet.DEFAULT_POLICIES
TRACE_PROVIDER = "codex"
TRACE_SELECTION_SEED = 20260822
TRACE_MIN_INPUT_TOKENS = 1024
TRACE_MAX_INPUT_TOKENS = 16384
TRACE_MIN_PREFIX_TOKENS = 1024
TRACE_MAX_APPEND_TOKENS = 4096
MAX_HTTP_REQUEST_CONCURRENCY = fleet.MAX_HTTP_REQUEST_CONCURRENCY


@dataclass(frozen=True)
class TraceLabCase:
    policy: str
    repeat: int
    endpoint_count: int = WORKER_COUNT
    workload: str = "tracelab_session_local"

    @property
    def name(self) -> str:
        return f"tracelab-{self.endpoint_count}w-{self.policy}-r{self.repeat}"


class TokenizersAdapter:
    """把 ``tokenizers.Tokenizer`` 适配为 TraceLab replay 所需的最小接口。"""

    def __init__(self, tokenizer_path: Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as error:
            raise RuntimeError(
                "TraceLab Simulator replay requires the Python tokenizers package"
            ) from error
        tokenizer_json = tokenizer_path / "tokenizer.json"
        if not tokenizer_json.is_file():
            raise RuntimeError(f"tokenizer.json does not exist: {tokenizer_json}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
        self.vocab_size = self._tokenizer.get_vocab_size()

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del clean_up_tokenization_spaces
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids


def parse_csv(value: str) -> tuple[str, ...]:
    return fleet.parse_csv(value)


def default_selection_config() -> ReplaySelectionConfig:
    return ReplaySelectionConfig(
        session_count=WORKER_COUNT,
        rounds_per_session=ROUNDS_PER_SESSION,
        seed=TRACE_SELECTION_SEED,
        min_input_tokens=TRACE_MIN_INPUT_TOKENS,
        max_input_tokens=TRACE_MAX_INPUT_TOKENS,
        min_prefix_tokens=TRACE_MIN_PREFIX_TOKENS,
        max_append_tokens=TRACE_MAX_APPEND_TOKENS,
        provider=TRACE_PROVIDER,
    )


def build_cases(policies: Sequence[str], *, repeats: int) -> tuple[TraceLabCase, ...]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return tuple(
        TraceLabCase(policy=policy, repeat=repeat)
        for policy in policies
        for repeat in range(repeats)
    )


def partition_replay_rounds(
    rounds: Sequence[ReplayRound],
) -> tuple[
    dict[str, tuple[ReplayRound, ...]],
    dict[str, tuple[ReplayRound, ...]],
]:
    """按 session 分开首轮 Router warmup 与后续测量，并拒绝乱序 trace。"""
    by_session: dict[str, list[ReplayRound]] = {}
    for round_ in rounds:
        by_session.setdefault(round_.session_id, []).append(round_)

    warmups: dict[str, tuple[ReplayRound, ...]] = {}
    measurements: dict[str, tuple[ReplayRound, ...]] = {}
    for session_id, turns in by_session.items():
        ordered = tuple(sorted(turns, key=lambda item: (item.round_index, item.source_line)))
        if len(ordered) != ROUNDS_PER_SESSION:
            raise ValueError(f"session {session_id} does not contain {ROUNDS_PER_SESSION} turns")
        if not ordered[0].is_warmup or any(turn.is_warmup for turn in ordered[1:]):
            raise ValueError(f"session {session_id} does not have exactly one leading warmup")
        if any(
            later.round_index != earlier.round_index + 1
            for earlier, later in zip(ordered, ordered[1:])
        ):
            raise ValueError(f"session {session_id} has non-contiguous rounds")
        warmups[session_id] = (ordered[0],)
        measurements[session_id] = ordered[1:]
    return warmups, measurements


def expected_measurement_count(rounds: Sequence[ReplayRound]) -> int:
    return sum(not round_.is_warmup for round_ in rounds)


def max_tokens_for_round(round_: ReplayRound, *, max_total_tokens: int) -> int:
    if round_.output_tokens <= 0:
        raise ValueError(f"TraceLab round has no output tokens: {round_.source_line}")
    if round_.input_tokens + round_.output_tokens > max_total_tokens:
        raise ValueError(
            "TraceLab round exceeds Simulator context capacity: "
            f"input={round_.input_tokens} output={round_.output_tokens} "
            f"max_total_tokens={max_total_tokens}"
        )
    return round_.output_tokens


def policy_args(policy: str, indexer_endpoint: str) -> list[str]:
    return fleet.policy_args(
        policy,
        indexer_endpoint,
        indexer_query_max_inflight=WORKER_COUNT,
    )


def router_command(
    args: argparse.Namespace,
    case: TraceLabCase,
    worker_urls: Sequence[str],
    indexer_endpoint: str,
) -> list[str]:
    return [
        str(args.router_binary),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.router_port),
        "--model-id",
        str(args.model_path),
        "--tokenizer-path",
        str(args.tokenizer_path / "tokenizer.json"),
        "--worker-urls",
        *worker_urls,
        "--request-timeout-secs",
        str(int(args.request_timeout_seconds)),
        "--stale-request-timeout-secs",
        str(int(args.stale_request_timeout_seconds)),
        "--load-monitor",
        *policy_args(case.policy, indexer_endpoint),
    ]


def start_case_control_plane(
    args: argparse.Namespace,
    case: TraceLabCase,
    directory: Path,
    specs: Sequence[fleet.WorkerSpec],
) -> list[fleet.ManagedProcess]:
    indexer_endpoint = f"http://127.0.0.1:{args.indexer_port}"
    managed: list[fleet.ManagedProcess] = []
    try:
        if fleet.needs_external_indexer(case.policy):
            indexer = fleet.start_process(
                "kv-indexer-server",
                [str(args.indexer_server)],
                directory / "kv-indexer-server.log",
                cwd=args.router_cwd,
                env={
                    "KV_INDEXER_LISTEN_ADDR": f"127.0.0.1:{args.indexer_port}",
                    "KV_INDEXER_PREFIX_QUERY_MAX_INFLIGHT": str(WORKER_COUNT),
                    "KV_INDEXER_MAX_CONCURRENT_STREAMS": str(WORKER_COUNT),
                },
            )
            managed.append(indexer)
            fleet.wait_tcp(
                "127.0.0.1",
                args.indexer_port,
                timeout=args.indexer_start_timeout,
                process=indexer.process,
            )
            for spec in specs:
                managed.append(
                    fleet.start_process(
                        f"kv-indexer-bridge-{spec.index}",
                        [str(args.indexer_bridge)],
                        directory / "bridges" / f"{spec.index}.log",
                        cwd=args.router_cwd,
                        env={
                            "KV_INDEXER_WORKER_ID": spec.worker_id,
                            "KV_INDEXER_WORKER_ADDRESS": spec.url,
                            "KV_INDEXER_ENDPOINT": indexer_endpoint,
                            "SGLANG_KV_EVENT_ENDPOINT": f"tcp://127.0.0.1:{spec.kv_port}",
                            "SGLANG_KV_EVENT_TOPIC": "kv",
                            "KV_INDEXER_CLEAR_TIERS": "HBM,DRAM,SSD",
                        },
                    )
                )
        router_environment: dict[str, str] = {}
        if case.policy in {"cache_aware", "shortest_ttft"}:
            router_environment["RUST_LOG"] = "info,sgl_router::server::routes::chat=debug"
        router = fleet.start_process(
            f"router-{case.policy}",
            router_command(args, case, [spec.url for spec in specs], indexer_endpoint),
            directory / "router.log",
            cwd=args.router_cwd,
            env=router_environment,
        )
        managed.append(router)
        asyncio.run(
            fleet.wait_http_urls(
                (f"http://127.0.0.1:{args.router_port}/healthz",),
                timeout=args.router_start_timeout,
            )
        )
        time.sleep(args.router_settle_seconds)
        return managed
    except Exception:
        fleet.stop_processes(managed)
        raise


async def stream_chat_request(
    *,
    session: object,
    router_url: str,
    model: str,
    prompt: str,
    round_: ReplayRound,
    timeout_seconds: float,
    max_total_tokens: int,
) -> dict[str, object]:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("TraceLab Simulator runner requires aiohttp") from error
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens_for_round(round_, max_total_tokens=max_total_tokens),
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.monotonic()
    first_token_at: float | None = None
    completion_tokens: int | None = None
    async with session.post(
        f"{router_url}/v1/chat/completions",
        json=payload,
        headers={"X-SMG-Routing-Key": round_.session_id},
    ) as response:
        if response.status != 200:
            body = await response.text()
            raise RuntimeError(
                f"TraceLab {round_.session_id}/{round_.round_index} returned "
                f"HTTP {response.status}: {body[:400]}"
            )
        buffer = b""
        async for chunk in response.content.iter_chunked(8192):
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                choices = event.get("choices")
                if isinstance(choices, list) and choices:
                    delta = choices[0].get("delta")
                    content = delta.get("content") if isinstance(delta, dict) else None
                    if isinstance(content, str) and content and first_token_at is None:
                        first_token_at = time.monotonic()
                usage = event.get("usage")
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                    completion_tokens = int(usage["completion_tokens"])
    finished = time.monotonic()
    if first_token_at is None:
        first_token_at = finished
    return {
        "session_id": round_.session_id,
        "round_index": round_.round_index,
        "source_line": round_.source_line,
        "trace_input_tokens": round_.input_tokens,
        "trace_prefix_tokens": round_.prefix_tokens,
        "trace_append_tokens": round_.newly_append_tokens,
        "trace_output_tokens": round_.output_tokens,
        "ttft_ms": (first_token_at - started) * 1000.0,
        "e2e_ms": (finished - started) * 1000.0,
        "completion_tokens": completion_tokens,
    }


async def replay_phase(
    phase: Mapping[str, Sequence[ReplayRound]],
    *,
    prompts: Mapping[tuple[str, int, int], str],
    router_url: str,
    model: str,
    request_rate: float,
    timeout_seconds: float,
    max_total_tokens: int,
) -> list[dict[str, object]]:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("TraceLab Simulator runner requires aiohttp") from error
    gate = fleet.RateGate(request_rate)
    semaphore = asyncio.Semaphore(MAX_HTTP_REQUEST_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def replay_one_session(turns: Sequence[ReplayRound]) -> list[dict[str, object]]:
            values: list[dict[str, object]] = []
            for round_ in turns:
                await gate.wait_turn()
                key = (round_.session_id, round_.round_index, round_.source_line)
                try:
                    prompt = prompts[key]
                except KeyError as error:
                    raise RuntimeError(f"missing virtual prompt for {key}") from error
                async with semaphore:
                    values.append(
                        await stream_chat_request(
                            session=session,
                            router_url=router_url,
                            model=model,
                            prompt=prompt,
                            round_=round_,
                            timeout_seconds=timeout_seconds,
                            max_total_tokens=max_total_tokens,
                        )
                    )
            return values

        nested = await asyncio.gather(
            *(replay_one_session(turns) for _, turns in sorted(phase.items()))
        )
    return [value for values in nested for value in values]


def build_virtual_prompts(
    rounds: Sequence[ReplayRound], tokenizer: TokenizersAdapter
) -> dict[tuple[str, int, int], str]:
    required = len({round_.session_id for round_ in rounds}) + len(rounds)
    safe_token_ids = find_reversible_token_ids(tokenizer, minimum=required)
    token_plan: VirtualTokenPlan = build_virtual_token_plan(
        rounds, safe_token_ids=safe_token_ids
    )
    return {
        (round_.session_id, round_.round_index, round_.source_line): reconstruct_virtual_prompt(
            round_, tokenizer, token_plan=token_plan
        )
        for round_ in rounds
    }


def create_or_verify_replay_manifest(
    results_dir: Path,
    *,
    trace: Path,
    selection: ReplaySelectionConfig,
) -> tuple[ReplayRound, ...]:
    rounds = select_replay_slice(load_trace_rows(trace), selection)
    trace_sha256 = replay_sha256_file(trace)
    payload = {
        "schema_version": 1,
        "trace_sha256": trace_sha256,
        "selection": asdict(selection),
        "rounds": [asdict(round_) for round_ in rounds],
    }
    manifest = results_dir / "tracelab-replay-manifest.json"
    if manifest.exists():
        try:
            existing = json.loads(manifest.read_text())
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid replay manifest: {manifest}") from error
        if existing != payload:
            raise RuntimeError(f"TraceLab replay manifest mismatch: {manifest}")
    else:
        write_replay_manifest(manifest, rounds, selection, trace_sha256=trace_sha256)
    return rounds


def case_complete(directory: Path, case: TraceLabCase) -> bool:
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
    case: TraceLabCase,
    rounds: Sequence[ReplayRound],
    prompts: Mapping[tuple[str, int, int], str],
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
        managed = start_case_control_plane(args, case, directory, specs)
        worker_urls = [spec.url for spec in specs]
        asyncio.run(fleet.flush_worker_caches(worker_urls))
        time.sleep(args.indexer_settle_seconds)

        warmups, measurements = partition_replay_rounds(rounds)
        asyncio.run(
            replay_phase(
                warmups,
                prompts=prompts,
                router_url=f"http://127.0.0.1:{args.router_port}",
                model=str(args.model_path),
                request_rate=args.request_rate,
                timeout_seconds=args.request_timeout_seconds,
                max_total_tokens=args.max_total_tokens,
            )
        )
        time.sleep(args.indexer_settle_seconds)

        worker_before = asyncio.run(
            fleet.fetch_texts(tuple(f"{url}/metrics" for url in worker_urls))
        )
        router_before = asyncio.run(
            fleet.fetch_texts((f"http://127.0.0.1:{args.router_port}/metrics",))
        )[0]
        router_log = directory / "router.log"
        log_offset = router_log.stat().st_size
        fleet.atomic_write_text(directory / "router.measurement_before.prom", router_before)

        started = time.monotonic()
        requests = asyncio.run(
            replay_phase(
                measurements,
                prompts=prompts,
                router_url=f"http://127.0.0.1:{args.router_port}",
                model=str(args.model_path),
                request_rate=args.request_rate,
                timeout_seconds=args.request_timeout_seconds,
                max_total_tokens=args.max_total_tokens,
            )
        )
        elapsed_seconds = time.monotonic() - started
        expected = expected_measurement_count(rounds)
        if len(requests) != expected:
            raise RuntimeError(f"{case.name} measured {len(requests)} requests, expected {expected}")

        worker_after = asyncio.run(
            fleet.fetch_texts(tuple(f"{url}/metrics" for url in worker_urls))
        )
        router_after = asyncio.run(
            fleet.fetch_texts((f"http://127.0.0.1:{args.router_port}/metrics",))
        )[0]
        cache = fleet.prefill_cache_summary(worker_before, worker_after)
        worker_success = fleet.metric_delta(
            fleet.worker_success_counts(router_before), fleet.worker_success_counts(router_after)
        )
        reasons = fleet.metric_delta(
            fleet.policy_reason_counts(router_before, case.policy),
            fleet.policy_reason_counts(router_after, case.policy),
        )
        audit: dict[str, int] | None = None
        if case.policy == "cache_aware":
            audit = fleet.cache_monitor_usage(router_log.read_text(errors="replace")[log_offset:])
            audit["actual_cache_metrics"] = int(
                float(cache["total_effective_tokens"]) > 0.0
            )
            fleet.require_native_cache_audit(audit)

        summary = fleet.summarize_case(
            requests,
            elapsed_seconds=elapsed_seconds,
            cache=cache,
            worker_success=worker_success,
            worker_urls=worker_urls,
            policy_reasons=reasons,
        )
        summary["native_cache_audit"] = audit
        summary["trace_measurement_count"] = expected
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
    args: argparse.Namespace,
    cases: Sequence[TraceLabCase],
    rounds: Sequence[ReplayRound],
    prompts: Mapping[tuple[str, int, int], str],
) -> None:
    pending = [
        case
        for case in cases
        if not (args.resume and case_complete(args.results_dir / case.name, case))
    ]
    if not pending:
        return
    fleet_directory = args.results_dir / "worker-fleet" / f"{WORKER_COUNT}w"
    specs, workers = fleet.start_worker_fleet(args, WORKER_COUNT, fleet_directory)
    try:
        for index, case in enumerate(cases, start=1):
            print(f"[{index}/{len(cases)}] run {case.name}", flush=True)
            run_case(args, case, rounds, prompts, (specs, workers))
    finally:
        fleet.stop_processes(workers)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--router-binary", type=Path)
    parser.add_argument("--router-cwd", type=Path)
    parser.add_argument("--indexer-server", type=Path)
    parser.add_argument("--indexer-bridge", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--simulator-site", type=Path)
    parser.add_argument("--simulator-config", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--worker-count", type=int, default=WORKER_COUNT)
    parser.add_argument("--sessions", type=int, default=WORKER_COUNT)
    parser.add_argument("--rounds-per-session", type=int, default=ROUNDS_PER_SESSION)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--request-rate", type=float, default=DEFAULT_REQUEST_RATE)
    parser.add_argument("--http-base-port", type=int)
    parser.add_argument("--reporter-base-port", type=int)
    parser.add_argument("--kv-base-port", type=int)
    parser.add_argument("--dist-base-port", type=int)
    parser.add_argument("--router-port", type=int, default=30_380)
    parser.add_argument("--indexer-port", type=int, default=50_551)
    parser.add_argument("--max-total-tokens", type=int, default=32_768)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument("--worker-start-batch-size", type=int, default=16)
    parser.add_argument("--worker-port-layout-wait-timeout", type=float, default=90.0)
    parser.add_argument("--indexer-start-timeout", type=float, default=60.0)
    parser.add_argument("--worker-start-timeout", type=float, default=300.0)
    parser.add_argument("--router-start-timeout", type=float, default=180.0)
    parser.add_argument("--router-settle-seconds", type=float, default=35.0)
    parser.add_argument("--indexer-settle-seconds", type=float, default=4.0)
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
    if args.sessions != WORKER_COUNT:
        parser.error(f"--sessions must be {WORKER_COUNT} for this contract")
    if args.rounds_per_session != ROUNDS_PER_SESSION:
        parser.error(f"--rounds-per-session must be {ROUNDS_PER_SESSION}")
    if args.repeats <= 0 or args.request_rate <= 0.0:
        parser.error("--repeats and --request-rate must be positive")
    if (
        args.max_total_tokens <= 0
        or args.max_running_requests <= 0
        or args.worker_start_batch_size <= 0
    ):
        parser.error("worker capacity arguments must be positive")
    if (
        args.worker_port_layout_wait_timeout < 0.0
        or args.router_settle_seconds < 0.0
        or args.indexer_settle_seconds < 0.0
        or args.control_plane_quiesce_seconds < 0.0
        or args.request_timeout_seconds <= 0.0
        or args.stale_request_timeout_seconds <= 0.0
    ):
        parser.error("timeout values are invalid")
    try:
        configured = fleet.configured_port_layout(args)
    except ValueError as error:
        parser.error(str(error))
    if configured is not None and any(
        port > 65_535 for port in configured.ports(WORKER_COUNT)
    ):
        parser.error("every worker port range must fit in 1..65535")
    if args.execute:
        required = (
            "source_root",
            "router_binary",
            "router_cwd",
            "indexer_server",
            "indexer_bridge",
            "python",
            "simulator_site",
            "simulator_config",
            "model_path",
            "tokenizer_path",
            "trace",
            "results_dir",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(
                "--execute requires: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        for name in required:
            value = getattr(args, name)
            if isinstance(value, Path):
                setattr(args, name, value.resolve())
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cases = build_cases(args.policies, repeats=args.repeats)
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "worker_count": WORKER_COUNT,
                    "sessions": WORKER_COUNT,
                    "measurement_requests_per_case": WORKER_COUNT * (ROUNDS_PER_SESSION - 1),
                    "cases": [asdict(case) | {"name": case.name} for case in cases],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    assert args.results_dir is not None
    assert args.trace is not None
    rounds = create_or_verify_replay_manifest(
        args.results_dir, trace=args.trace, selection=default_selection_config()
    )
    warmups, _ = partition_replay_rounds(rounds)
    if len(warmups) != WORKER_COUNT:
        raise RuntimeError(f"expected {WORKER_COUNT} TraceLab sessions, got {len(warmups)}")
    for round_ in rounds:
        max_tokens_for_round(round_, max_total_tokens=args.max_total_tokens)
    tokenizer = TokenizersAdapter(args.tokenizer_path)
    prompts = build_virtual_prompts(rounds, tokenizer)
    contract = {
        "schema_version": 1,
        "source_commit": fleet.read_source_commit(args.source_root),
        "router_binary_sha256": fleet.sha256_file(args.router_binary),
        "indexer_server_sha256": fleet.sha256_file(args.indexer_server),
        "indexer_bridge_sha256": fleet.sha256_file(args.indexer_bridge),
        "worker_count": WORKER_COUNT,
        "policies": list(args.policies),
        "repeats": args.repeats,
        "request_rate": args.request_rate,
        "max_total_tokens": args.max_total_tokens,
        "max_running_requests": args.max_running_requests,
        "trace": str(args.trace),
        "trace_sha256": replay_sha256_file(args.trace),
        "trace_selection": asdict(default_selection_config()),
        "replay_manifest_sha256": fleet.sha256_file(
            args.results_dir / "tracelab-replay-manifest.json"
        ),
        "output_tokens": "per TraceLab round",
        "control_plane_quiesce_seconds": args.control_plane_quiesce_seconds,
        "simulator_config": str(args.simulator_config),
        "cases": [asdict(case) | {"name": case.name} for case in cases],
    }
    fleet.write_or_verify_manifest(args.results_dir, contract, resume=args.resume)
    run_cases(args, cases, rounds, prompts)
    fleet.atomic_write_text(args.results_dir / "RUN_COMPLETE", "ok\n")
    fleet.atomic_write_text(args.results_dir / "CURRENT", "complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
