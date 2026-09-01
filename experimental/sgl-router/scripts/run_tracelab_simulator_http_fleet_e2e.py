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
DEFAULT_WARMUP_REQUEST_RATE = 1.0
DEFAULT_PRESSURE_GUARD_SEED_HOLDERS = 2
DEFAULT_PRESSURE_GUARD_SEED_REQUEST_RATE = 64.0
DEFAULT_POLICIES = fleet.DEFAULT_POLICIES
TRACE_PROVIDER = "codex"
TRACE_SELECTION_SEED = 20260822
TRACE_MIN_INPUT_TOKENS = 1024
TRACE_MAX_INPUT_TOKENS = 16384
TRACE_MIN_PREFIX_TOKENS = 1024
TRACE_MAX_APPEND_TOKENS = 4096
MAX_HTTP_REQUEST_CONCURRENCY = fleet.MAX_HTTP_REQUEST_CONCURRENCY
BRIDGE_APPLY_LOG_MARKER = "applied KV event batch to Indexer"
BRIDGE_FAILURE_LOG_MARKER = "bridge session lost; reconnecting"


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


def bridge_log_progress(log_paths: Sequence[Path]) -> tuple[int, int]:
    applied_batches = 0
    bridge_failures = 0
    for path in log_paths:
        text = path.read_text(errors="replace") if path.exists() else ""
        applied_batches += text.count(BRIDGE_APPLY_LOG_MARKER)
        bridge_failures += text.count(BRIDGE_FAILURE_LOG_MARKER)
    return applied_batches, bridge_failures


def wait_for_indexer_bridge_drain(
    log_paths: Sequence[Path],
    *,
    quiet_seconds: float,
    timeout_seconds: float,
    poll_seconds: float,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> dict[str, int | float]:
    deadline = monotonic() + timeout_seconds
    applied_batches, bridge_failures = bridge_log_progress(log_paths)
    if bridge_failures:
        raise RuntimeError(f"bridge failure before Indexer drain: {bridge_failures}")
    last_change = monotonic()
    while True:
        applied_now, bridge_failures = bridge_log_progress(log_paths)
        if bridge_failures:
            raise RuntimeError(f"bridge failure before Indexer drain: {bridge_failures}")
        now = monotonic()
        if applied_now != applied_batches:
            applied_batches = applied_now
            last_change = now
        if applied_batches and now - last_change >= quiet_seconds:
            return {
                "applied_batches": applied_batches,
                "bridge_failures": bridge_failures,
                "quiet_seconds": quiet_seconds,
            }
        if now >= deadline:
            raise RuntimeError(
                "Indexer bridge did not reach a successful apply quiet window "
                f"within {timeout_seconds}s"
            )
        sleep(min(poll_seconds, deadline - now))


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


INDEXER_QUERY_DURATION_METRIC = "sgl_router_kv_indexer_query_duration_seconds"
ZMQ_PREFIX_LOOKUP_DURATION_METRIC = "sgl_router_zmq_prefix_lookup_duration_seconds"


def policy_args(
    policy: str,
    indexer_endpoint: str,
    *,
    query_timeout_ms: int,
    query_max_inflight: int,
) -> list[str]:
    return fleet.policy_args(
        policy,
        indexer_endpoint,
        indexer_query_timeout_ms=query_timeout_ms,
        indexer_query_max_inflight=query_max_inflight,
    )


def indexer_query_summary(
    before: str,
    after: str,
    *,
    metric: str = INDEXER_QUERY_DURATION_METRIC,
) -> dict[str, object]:
    def aggregate(metric: str, snapshot: str) -> dict[str, float]:
        values: dict[str, float] = {}
        for labels, value in fleet.metric_samples(snapshot, metric):
            outcome = labels.get("outcome")
            if outcome:
                values[outcome] = values.get(outcome, 0.0) + value
        return values

    def buckets(snapshot: str) -> dict[tuple[str, str], float]:
        values: dict[tuple[str, str], float] = {}
        for labels, value in fleet.metric_samples(snapshot, f"{metric}_bucket"):
            outcome = labels.get("outcome")
            bound = labels.get("le")
            if outcome and bound:
                key = (outcome, bound)
                values[key] = values.get(key, 0.0) + value
        return values

    counts = fleet.metric_delta(
        aggregate(f"{metric}_count", before),
        aggregate(f"{metric}_count", after),
    )
    sums = fleet.metric_delta(
        aggregate(f"{metric}_sum", before),
        aggregate(f"{metric}_sum", after),
    )
    before_buckets = buckets(before)
    after_buckets = buckets(after)
    bucket_delta = {
        key: after_buckets.get(key, 0.0) - before_buckets.get(key, 0.0)
        for key in set(before_buckets) | set(after_buckets)
    }

    outcomes: dict[str, dict[str, float | int | None]] = {}
    for outcome, count in sorted(counts.items()):
        if count <= 0.0:
            continue
        finite_bounds = sorted(
            (float(bound), value)
            for (bucket_outcome, bound), value in bucket_delta.items()
            if bucket_outcome == outcome and bound != "+Inf" and value >= 0.0
        )

        def quantile_ms(quantile: float) -> float | None:
            target = count * quantile
            for bound, cumulative in finite_bounds:
                if cumulative >= target:
                    return bound * 1_000.0
            return None

        outcomes[outcome] = {
            "count": int(round(count)),
            "mean_ms": 1_000.0 * sums.get(outcome, 0.0) / count,
            "p50_ms": quantile_ms(0.50),
            "p95_ms": quantile_ms(0.95),
        }

    query_count = sum(int(values["count"]) for values in outcomes.values())
    success_count = int(outcomes.get("success", {}).get("count", 0))
    return {
        "query_count": query_count,
        "success_count": success_count,
        "failure_count": query_count - success_count,
        "outcomes": outcomes,
    }


def zmq_prefix_lookup_summary(before: str, after: str) -> dict[str, object]:
    summary = indexer_query_summary(before, after, metric=ZMQ_PREFIX_LOOKUP_DURATION_METRIC)
    return {
        "lookup_count": summary["query_count"],
        "matched_count": int(summary["outcomes"].get("matched", {}).get("count", 0)),
        "empty_count": int(summary["outcomes"].get("empty", {}).get("count", 0)),
        "outcomes": summary["outcomes"],
    }


def require_indexer_query_success(
    summary: Mapping[str, object], *, expected_queries: int, phase: str
) -> None:
    query_count = int(summary["query_count"])
    success_count = int(summary["success_count"])
    failure_count = int(summary["failure_count"])
    if query_count != expected_queries:
        raise RuntimeError(
            f"Indexer {phase} query count was {query_count}, expected {expected_queries}"
        )
    if failure_count != 0 or success_count != expected_queries:
        raise RuntimeError(
            f"Indexer {phase} queries failed: success={success_count} failure={failure_count}"
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
        *policy_args(
            case.policy,
            indexer_endpoint,
            query_timeout_ms=args.kv_indexer_query_timeout_ms,
            query_max_inflight=args.kv_indexer_query_max_inflight,
        ),
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
                    "KV_INDEXER_PREFIX_QUERY_MAX_INFLIGHT": str(
                        args.kv_indexer_query_max_inflight
                    ),
                    "KV_INDEXER_MAX_CONCURRENT_STREAMS": str(
                        args.kv_indexer_max_concurrent_streams
                    ),
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
                            "RUST_LOG": "info,sgl_kv_indexer::bridge=debug",
                        },
                    )
                )
        router_environment = {
            "RUST_LOG": (
                "info,sgl_router::server::routes::chat=debug,"
                "sgl_router::policies::power_of_two=debug,"
                "sgl_router::policies::cache_aware_zmq=debug"
            )
        }
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


def pressure_guard_seed_targets(
    session_ids: Sequence[str],
    worker_urls: Sequence[str],
    *,
    holders_per_session: int,
) -> dict[str, tuple[str, ...]]:
    """为每个 TraceLab session 固定分配多个独立 cache replica。"""
    if holders_per_session < 2:
        raise ValueError("pressure-guard seed requires at least two holders")
    if len(worker_urls) < holders_per_session:
        raise ValueError("worker fleet cannot satisfy pressure-guard replica count")
    return {
        session_id: tuple(
            worker_urls[(index * holders_per_session + offset) % len(worker_urls)]
            for offset in range(holders_per_session)
        )
        for index, session_id in enumerate(sorted(session_ids))
    }


def pressure_guard_seed_warmups(
    warmups: Mapping[str, Sequence[ReplayRound]],
) -> dict[str, Sequence[ReplayRound]]:
    """选择一个确定 session 作为 pressure guard 的双副本预条件。

    正常 router warmup 先按 V3 语义为整条 TraceLab slice 建立单副本 cache。
    仅对一个 session 补两个直接 worker replica，既可让 guard 观察到完整 pair，
    又避免预条件本身挤掉整支 fleet 的正常 warmup cache。
    """
    if not warmups:
        raise ValueError("pressure-guard seed requires at least one warmup session")
    session_id = min(warmups)
    return {session_id: warmups[session_id]}


async def seed_pressure_guard_replicas(
    warmups: Mapping[str, Sequence[ReplayRound]],
    *,
    prompts: Mapping[tuple[str, int, int], str],
    worker_urls: Sequence[str],
    model: str,
    request_rate: float,
    timeout_seconds: float,
    max_total_tokens: int,
    holders_per_session: int,
) -> int:
    """在 router warmup 前为每个 session 预置两个同前缀的 worker。"""
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("TraceLab Simulator runner requires aiohttp") from error
    targets = pressure_guard_seed_targets(
        tuple(warmups), worker_urls, holders_per_session=holders_per_session
    )
    gate = fleet.RateGate(request_rate)
    semaphore = asyncio.Semaphore(MAX_HTTP_REQUEST_CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def seed_one(round_: ReplayRound, worker_url: str) -> None:
            await gate.wait_turn()
            key = (round_.session_id, round_.round_index, round_.source_line)
            try:
                prompt = prompts[key]
            except KeyError as error:
                raise RuntimeError(f"missing virtual prompt for {key}") from error
            async with semaphore:
                await stream_chat_request(
                    session=session,
                    router_url=worker_url,
                    model=model,
                    prompt=prompt,
                    round_=round_,
                    timeout_seconds=timeout_seconds,
                    max_total_tokens=max_total_tokens,
                )

        await asyncio.gather(
            *(
                seed_one(turns[0], worker_url)
                for session_id, turns in sorted(warmups.items())
                for worker_url in targets[session_id]
            )
        )
    return sum(len(urls) for urls in targets.values())


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


def measurement_decision_log(
    router_log: Path, *, policy_marker: str, expected_decisions: int
) -> str:
    """从完整 router 日志取测量阶段最后的策略决策。

    Router 子进程的 stdout 可能在 warmup/measurement 边界后才批量 flush，
    因此文件 offset 不能作为阶段边界。请求按阶段串行执行，最后
    ``expected_decisions`` 条指定策略决策就是 measurement 的审计输入。
    """
    lines = []
    for line in router_log.read_text(errors="replace").splitlines():
        normalized = fleet.ANSI_ESCAPE_RE.sub("", line)
        if policy_marker in normalized:
            lines.append(line)
    if len(lines) < expected_decisions:
        raise RuntimeError(
            f"{policy_marker} audit has {len(lines)} decisions, expected at least {expected_decisions}"
        )
    return "\n".join(lines[-expected_decisions:]) + "\n"


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
        router_warmup_before = asyncio.run(
            fleet.fetch_texts((f"http://127.0.0.1:{args.router_port}/metrics",))
        )[0]
        fleet.atomic_write_text(directory / "router.warmup_before.prom", router_warmup_before)
        asyncio.run(
            replay_phase(
                warmups,
                prompts=prompts,
                router_url=f"http://127.0.0.1:{args.router_port}",
                model=str(args.model_path),
                request_rate=args.warmup_request_rate,
                timeout_seconds=args.request_timeout_seconds,
                max_total_tokens=args.max_total_tokens,
            )
        )
        if fleet.needs_external_indexer(case.policy):
            drain = wait_for_indexer_bridge_drain(
                tuple(directory / "bridges" / f"{spec.index}.log" for spec in specs),
                quiet_seconds=args.indexer_drain_quiet_seconds,
                timeout_seconds=args.indexer_drain_timeout_seconds,
                poll_seconds=args.indexer_drain_poll_seconds,
            )
            fleet.atomic_write_text(
                directory / "indexer_drain.json",
                json.dumps(drain, indent=2, sort_keys=True) + "\n",
            )
        else:
            time.sleep(args.indexer_settle_seconds)

        guard_seed_warmups = pressure_guard_seed_warmups(warmups)
        seeded_requests = asyncio.run(
            seed_pressure_guard_replicas(
                guard_seed_warmups,
                prompts=prompts,
                worker_urls=worker_urls,
                model=str(args.model_path),
                request_rate=args.pressure_guard_seed_request_rate,
                timeout_seconds=args.request_timeout_seconds,
                max_total_tokens=args.max_total_tokens,
                holders_per_session=args.pressure_guard_seed_holders,
            )
        )
        if fleet.needs_external_indexer(case.policy):
            seed_drain = wait_for_indexer_bridge_drain(
                tuple(directory / "bridges" / f"{spec.index}.log" for spec in specs),
                quiet_seconds=args.indexer_drain_quiet_seconds,
                timeout_seconds=args.indexer_drain_timeout_seconds,
                poll_seconds=args.indexer_drain_poll_seconds,
            )
            seed_drain["seeded_requests"] = seeded_requests
            seed_drain["seeded_sessions"] = len(guard_seed_warmups)
            fleet.atomic_write_text(
                directory / "pressure_guard_seed_drain.json",
                json.dumps(seed_drain, indent=2, sort_keys=True) + "\n",
            )
        else:
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
        warmup_indexer_query = indexer_query_summary(router_warmup_before, router_before)

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
        cache_aware_controls = (
            fleet.cache_aware_control_summary(router_before, router_after)
            if case.policy == "cache_aware"
            else None
        )
        if cache_aware_controls is not None:
            fleet.require_cache_aware_controls(cache_aware_controls)
        worker_success = fleet.metric_delta(
            fleet.worker_success_counts(router_before), fleet.worker_success_counts(router_after)
        )
        reasons = fleet.metric_delta(
            fleet.policy_reason_counts(router_before, case.policy),
            fleet.policy_reason_counts(router_after, case.policy),
        )
        measurement_indexer_query = indexer_query_summary(router_before, router_after)
        measurement_zmq_lookup = zmq_prefix_lookup_summary(router_before, router_after)
        if args.require_indexer_success and fleet.needs_external_indexer(case.policy):
            require_indexer_query_success(
                warmup_indexer_query,
                expected_queries=sum(len(turns) for turns in warmups.values()),
                phase="warmup",
            )
            require_indexer_query_success(
                measurement_indexer_query,
                expected_queries=expected,
                phase="measurement",
            )
        audit: dict[str, int] | None = None
        shortest_ttft_audit: dict[str, int] | None = None
        power_of_two_audit: dict[str, int] | None = None
        zmq_policy_audit: dict[str, int] | None = None
        if case.policy == "cache_aware":
            decision_log = measurement_decision_log(
                router_log, policy_marker="cache candidate winner", expected_decisions=expected
            )
            audit = fleet.cache_monitor_usage(decision_log)
            audit["actual_cache_metrics"] = int(
                float(cache["total_effective_tokens"]) > 0.0
            )
            fleet.require_native_cache_audit(audit, expected_decisions=expected)
        elif case.policy == "shortest_ttft":
            decision_log = measurement_decision_log(
                router_log, policy_marker="shortest TTFT candidate winner", expected_decisions=expected
            )
            shortest_ttft_audit = fleet.shortest_ttft_monitor_usage(decision_log)
            fleet.require_shortest_ttft_audit(
                shortest_ttft_audit, expected_decisions=expected
            )
        elif case.policy == "power_of_two":
            decision_log = measurement_decision_log(
                router_log, policy_marker="policy=PowerOfTwo", expected_decisions=expected
            )
            power_of_two_audit = fleet.power_of_two_monitor_usage(decision_log)
            fleet.require_power_of_two_audit(
                power_of_two_audit, expected_decisions=expected
            )
        elif case.policy == "cache_aware_zmq":
            decision_log = measurement_decision_log(
                router_log, policy_marker="cache-aware-zmq", expected_decisions=expected
            )
            zmq_policy_audit = fleet.zmq_policy_usage(decision_log)
            fleet.require_zmq_policy_audit(zmq_policy_audit, expected_decisions=expected)

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
        summary["indexer_query"] = (
            {
                "warmup": warmup_indexer_query,
                "measurement": measurement_indexer_query,
            }
            if fleet.needs_external_indexer(case.policy)
            else None
        )
        summary["zmq_prefix_lookup"] = (
            measurement_zmq_lookup if case.policy == "cache_aware_zmq" else None
        )
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
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] run {case.name}", flush=True)
        fleet_directory = args.results_dir / "worker-fleets" / case.name
        specs, workers = fleet.start_worker_fleet(args, WORKER_COUNT, fleet_directory)
        try:
            run_case(args, case, rounds, prompts, (specs, workers))
        finally:
            fleet.stop_processes(workers)
            fleet.wait_for_control_plane_quiescence(args)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--router-binary", type=Path)
    parser.add_argument("--router-cwd", type=Path)
    parser.add_argument("--indexer-server", type=Path)
    parser.add_argument("--indexer-bridge", type=Path)
    parser.add_argument("--python")
    parser.add_argument("--simulator-site", type=Path)
    parser.add_argument("--simulator-dependency-root", type=Path)
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
    parser.add_argument(
        "--warmup-request-rate", type=float, default=DEFAULT_WARMUP_REQUEST_RATE
    )
    parser.add_argument(
        "--pressure-guard-seed-holders",
        type=int,
        default=DEFAULT_PRESSURE_GUARD_SEED_HOLDERS,
    )
    parser.add_argument(
        "--pressure-guard-seed-request-rate",
        type=float,
        default=DEFAULT_PRESSURE_GUARD_SEED_REQUEST_RATE,
    )
    parser.add_argument("--http-base-port", type=int)
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
    parser.add_argument("--indexer-drain-quiet-seconds", type=float, default=5.0)
    parser.add_argument("--indexer-drain-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--indexer-drain-poll-seconds", type=float, default=0.5)
    parser.add_argument("--control-plane-quiesce-seconds", type=float, default=16.0)
    parser.add_argument("--request-timeout-seconds", type=int, default=360)
    parser.add_argument("--stale-request-timeout-seconds", type=int, default=420)
    parser.add_argument("--kv-indexer-query-timeout-ms", type=int, default=10_000)
    parser.add_argument("--kv-indexer-query-max-inflight", type=int, default=256)
    parser.add_argument("--kv-indexer-max-concurrent-streams", type=int, default=512)
    parser.add_argument("--require-indexer-success", action="store_true")
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
    if (
        args.repeats <= 0
        or args.request_rate <= 0.0
        or args.warmup_request_rate <= 0.0
        or args.pressure_guard_seed_request_rate <= 0.0
    ):
        parser.error(
            "--repeats, --request-rate, --warmup-request-rate, and "
            "--pressure-guard-seed-request-rate must be positive"
        )
    if args.pressure_guard_seed_holders < 2:
        parser.error("--pressure-guard-seed-holders must be at least two")
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
        or args.indexer_drain_quiet_seconds <= 0.0
        or args.indexer_drain_timeout_seconds <= 0.0
        or args.indexer_drain_poll_seconds <= 0.0
        or args.control_plane_quiesce_seconds < 0.0
        or args.request_timeout_seconds <= 0.0
        or args.stale_request_timeout_seconds <= 0.0
        or args.kv_indexer_query_timeout_ms <= 0
        or args.kv_indexer_query_max_inflight <= 0
        or args.kv_indexer_max_concurrent_streams <= 0
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
            "simulator_dependency_root",
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
    invocation = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    args = parse_args(invocation)
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
    fleet.validate_simulator_runtime(args)
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
        "schema_version": 3,
        "source_commit": fleet.read_source_commit(args.source_root),
        "router_binary_sha256": fleet.sha256_file(args.router_binary),
        "indexer_server_sha256": fleet.sha256_file(args.indexer_server),
        "indexer_bridge_sha256": fleet.sha256_file(args.indexer_bridge),
        "worker_count": WORKER_COUNT,
        "policies": list(args.policies),
        "repeats": args.repeats,
        "request_rate": args.request_rate,
        "warmup_request_rate": args.warmup_request_rate,
        "pressure_guard_seed_holders": args.pressure_guard_seed_holders,
        "pressure_guard_seed_request_rate": args.pressure_guard_seed_request_rate,
        "indexer_drain_quiet_seconds": args.indexer_drain_quiet_seconds,
        "indexer_drain_timeout_seconds": args.indexer_drain_timeout_seconds,
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
        "case_isolation": {
            "worker_lifecycle": "fresh_per_case",
            "cache_reset": "flush_before_warmup",
            "control_plane": "stop_all_then_quiesce",
        },
        "kv_indexer_query_timeout_ms": args.kv_indexer_query_timeout_ms,
        "kv_indexer_query_max_inflight": args.kv_indexer_query_max_inflight,
        "kv_indexer_max_concurrent_streams": args.kv_indexer_max_concurrent_streams,
        "require_indexer_success": args.require_indexer_success,
        "execution_artifacts": fleet.execution_artifact_contract(
            runner_script=Path(__file__),
            python=args.python,
            simulator_config=args.simulator_config,
            simulator_dependency_root=args.simulator_dependency_root,
            simulator_runtime_python_root=args.source_root / "python",
            argv=invocation,
        ),
        "simulator_dependency_root": str(args.simulator_dependency_root),
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
