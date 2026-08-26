# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""以真实 SGLang Simulator HTTP worker 运行 Router fleet 合同。"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

DEFAULT_ENDPOINT_COUNTS = (32, 128, 256, 512, 1024)
DEFAULT_POLICIES = (
    "power_of_two",
    "cache_aware",
    "cache_aware_zmq",
    "shortest_ttft",
)
DEFAULT_WORKLOADS = ("tracelab_multiturn", "multi_holder_pressure")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class WorkerSpec:
    index: int
    http_port: int
    reporter_port: int
    kv_port: int
    dist_port: int

    @property
    def worker_id(self) -> str:
        return f"simulator-{self.index:04d}"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.http_port}"


@dataclass(frozen=True)
class Case:
    endpoint_count: int
    policy: str
    workload: str
    repeat: int

    @property
    def name(self) -> str:
        return (
            f"{self.workload}-{self.endpoint_count}w-"
            f"{self.policy}-r{self.repeat}"
        )


def parse_csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("expected at least one comma-separated value")
    return items


def parse_endpoint_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item) for item in parse_csv(value))
    if any(count <= 0 for count in counts):
        raise ValueError("endpoint counts must be positive")
    if len(set(counts)) != len(counts):
        raise ValueError("endpoint counts must be unique")
    return counts


def build_cases(
    *,
    endpoint_counts: Sequence[int],
    policies: Sequence[str],
    workloads: Sequence[str],
    repeats: int,
) -> tuple[Case, ...]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return tuple(
        Case(
            endpoint_count=endpoint_count,
            policy=policy,
            workload=workload,
            repeat=repeat,
        )
        for endpoint_count in endpoint_counts
        for policy in policies
        for workload in workloads
        for repeat in range(repeats)
    )


def worker_spec(
    index: int,
    *,
    http_base_port: int,
    reporter_base_port: int,
    kv_base_port: int,
    dist_base_port: int,
) -> WorkerSpec:
    if index < 0:
        raise ValueError("worker index must be non-negative")
    return WorkerSpec(
        index=index,
        http_port=http_base_port + index,
        reporter_port=reporter_base_port + index,
        kv_port=kv_base_port + index,
        dist_port=dist_base_port + index,
    )


def simulator_worker_command(
    spec: WorkerSpec,
    *,
    python: str,
    simulator_config: Path,
    model_path: Path,
    tokenizer_path: Path,
    max_total_tokens: int,
    max_running_requests: int,
) -> tuple[list[str], dict[str, str]]:
    kv_events = json.dumps(
        {
            "endpoint": f"tcp://*:{spec.kv_port}",
            "publisher": "zmq",
            "topic": "kv",
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    command = [
        python,
        "-m",
        "sglang_simulator.simulation.sglang.launch_server",
        "--model-path",
        str(model_path),
        "--sim-config-path",
        str(simulator_config),
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.http_port),
        "--tokenizer-path",
        str(tokenizer_path),
        "--chat-template",
        "chatml",
        "--max-total-tokens",
        str(max_total_tokens),
        "--max-running-requests",
        str(max_running_requests),
        "--disable-overlap-schedule",
        "--load-reporter-port",
        str(spec.reporter_port),
        "--dist-init-addr",
        f"127.0.0.1:{spec.dist_port}",
        "--enable-cache-report",
        "--enable-metrics",
        "--kv-events-config",
        kv_events,
    ]
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "SGLANG_USE_CPU_ENGINE": "1",
        "SGLANG_SIMULATOR_CONFIG_PATH": str(simulator_config),
        "SGLANG_SIMULATOR_OUTPUT_MODE": "BLOCKING",
    }
    return command, environment


def simulator_environment(
    *,
    simulator_site: Path,
    source_root: Path,
    simulator_config: Path,
) -> dict[str, str]:
    """返回只覆盖 Simulator worker 的运行时环境。"""
    inherited = os.environ.get("PYTHONPATH")
    paths = [
        str(simulator_site),
        str(source_root / "tools" / "sglang-simulator" / "src"),
        str(source_root / "python"),
    ]
    if inherited:
        paths.append(inherited)
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join(paths),
        "SGLANG_SIMULATOR_CONFIG_PATH": str(simulator_config),
        "SGLANG_SIMULATOR_OUTPUT_MODE": "BLOCKING",
        "SGLANG_USE_CPU_ENGINE": "1",
    }


def policy_args(policy: str, indexer_endpoint: str) -> list[str]:
    if policy == "power_of_two":
        return ["--policy", policy]
    if policy == "cache_aware_zmq":
        return [
            "--policy",
            policy,
            "--cache-threshold",
            "0.5",
            "--balance-abs-threshold",
            "32",
            "--balance-rel-threshold",
            "1.1",
        ]
    if policy == "cache_aware":
        return [
            "--policy",
            policy,
            "--kv-indexer-endpoint",
            indexer_endpoint,
            "--kv-indexer-query-timeout-ms",
            "100",
            "--cache-affinity-min-matched-tokens",
            "1024",
            "--cache-candidate-min-workers",
            "2",
            "--cache-candidate-ratio",
            "0.05",
            "--cache-candidate-max-workers",
            "32",
            "--cache-switch-margin-tokens",
            "1024",
            "--pressure-abs-threshold-tokens",
            "8192",
            "--pressure-rel-threshold",
            "1.5",
        ]
    if policy == "shortest_ttft":
        return [
            "--policy",
            policy,
            "--kv-indexer-endpoint",
            indexer_endpoint,
            "--kv-indexer-query-timeout-ms",
            "100",
        ]
    raise ValueError(f"unsupported policy: {policy}")


def require_native_cache_audit(audit: dict[str, int]) -> None:
    required_positive = (
        "cache_candidate_decisions",
        "monitor_decisions",
        "actual_cache_metrics",
    )
    for key in required_positive:
        if audit.get(key, 0) <= 0:
            raise RuntimeError(f"{key} must be positive for native cache-aware")
    if audit.get("monitor_decisions") != audit.get("cache_candidate_decisions"):
        raise RuntimeError("monitor_decisions must cover every cache_candidate_decisions")
    for key in ("router_local_decisions", "zero_snapshot_decisions"):
        if audit.get(key, 0) != 0:
            raise RuntimeError(f"{key} must be zero for native cache-aware")


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen[bytes]
    log_handle: object


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending-{os.getpid()}")
    pending.write_text(text)
    os.replace(pending, path)


def start_process(
    name: str,
    command: Sequence[str],
    log_path: Path,
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
) -> ManagedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    child_env = os.environ.copy()
    if env:
        child_env.update(env)
    process = subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=child_env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return ManagedProcess(name=name, process=process, log_handle=log_handle)


def stop_processes(processes: Iterable[ManagedProcess]) -> None:
    active = [managed for managed in processes if managed.process.poll() is None]
    for managed in active:
        try:
            os.killpg(managed.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30.0
    for managed in active:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            managed.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(managed.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            managed.process.wait(timeout=10)
    for managed in processes:
        managed.log_handle.close()


async def wait_http_urls(urls: Sequence[str], *, timeout: float) -> None:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("HTTP fleet runner requires aiohttp") from error

    deadline = time.monotonic() + timeout
    pending = set(urls)
    connector = aiohttp.TCPConnector(limit=min(256, max(16, len(urls))))
    client_timeout = aiohttp.ClientTimeout(total=3)
    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        while pending and time.monotonic() < deadline:
            batch = tuple(pending)

            async def probe(url: str) -> tuple[str, bool]:
                try:
                    async with session.get(url) as response:
                        return url, response.status < 500
                except aiohttp.ClientError:
                    return url, False

            for url, ready in await asyncio.gather(*(probe(url) for url in batch)):
                if ready:
                    pending.discard(url)
            if pending:
                await asyncio.sleep(0.5)
    if pending:
        sample = ", ".join(sorted(pending)[:5])
        raise TimeoutError(f"timed out waiting for {len(pending)} HTTP endpoints: {sample}")


def wait_tcp(host: str, port: int, *, timeout: float, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"process exited while waiting for {host}:{port}")
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError as error:
            last_error = str(error)
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {host}:{port}: {last_error}")


async def fetch_texts(urls: Sequence[str]) -> list[str]:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("HTTP fleet runner requires aiohttp") from error

    connector = aiohttp.TCPConnector(limit=min(256, max(16, len(urls))))
    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def fetch(url: str) -> str:
            async with session.get(url) as response:
                if response.status != 200:
                    raise RuntimeError(f"GET {url} returned HTTP {response.status}")
                return await response.text()

        return list(await asyncio.gather(*(fetch(url) for url in urls)))


def metric_samples(text: str, name: str) -> list[tuple[dict[str, str], float]]:
    samples: list[tuple[dict[str, str], float]] = []
    prefix = f"{name}{{"
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith(prefix):
            continue
        metric, raw_value = line.rsplit(" ", 1)
        labels: dict[str, str] = {}
        for field in metric[len(prefix) : -1].split(","):
            key, value = field.split("=", 1)
            labels[key] = value.strip('"')
        samples.append((labels, float(raw_value)))
    return samples


def metric_delta(
    before: Mapping[str, float], after: Mapping[str, float]
) -> dict[str, float]:
    return {
        key: after.get(key, 0.0) - before.get(key, 0.0)
        for key in sorted(set(before) | set(after))
        if after.get(key, 0.0) - before.get(key, 0.0) != 0.0
    }


def policy_reason_counts(metrics: str, policy: str) -> dict[str, float]:
    return {
        labels["reason"]: value
        for labels, value in metric_samples(metrics, "sgl_router_policy_decisions_total")
        if labels.get("policy") == policy and labels.get("reason")
    }


def worker_success_counts(metrics: str) -> dict[str, float]:
    return {
        labels["worker_url"]: value
        for labels, value in metric_samples(metrics, "sgl_router_worker_requests_total")
        if labels.get("outcome") == "success" and labels.get("worker_url")
    }


def prefill_cache_summary(before_metrics: Sequence[str], after_metrics: Sequence[str]) -> dict[str, object]:
    if len(before_metrics) != len(after_metrics):
        raise ValueError("worker metric snapshots must have equal lengths")

    def by_mode(metrics: Sequence[str]) -> dict[str, float]:
        values: dict[str, float] = {}
        for text in metrics:
            for labels, value in metric_samples(text, "sglang:prefill_effective_tokens_total"):
                mode = labels.get("mode")
                if mode:
                    values[mode] = values.get(mode, 0.0) + value
        return values

    values = metric_delta(by_mode(before_metrics), by_mode(after_metrics))
    if any(value < 0.0 for value in values.values()):
        raise RuntimeError("prefill effective-token counter decreased")
    values = {mode: value for mode, value in values.items() if value > 0.0}
    total = sum(values.values())
    hit = sum(
        value
        for mode, value in values.items()
        if mode in {"device_hit", "host_hit", "storage_hit"}
    )
    return {
        "tokens_by_mode": values,
        "total_effective_tokens": total,
        "hit_tokens": hit,
        "hit_rate": 0.0 if total == 0.0 else hit / total,
    }


def router_command(
    args: argparse.Namespace,
    case: Case,
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
        "180",
        "--stale-request-timeout-secs",
        "240",
        "--load-monitor",
        *policy_args(case.policy, indexer_endpoint),
    ]


def worker_specs(args: argparse.Namespace, endpoint_count: int) -> list[WorkerSpec]:
    return [
        worker_spec(
            index,
            http_base_port=args.http_base_port,
            reporter_base_port=args.reporter_base_port,
            kv_base_port=args.kv_base_port,
            dist_base_port=args.dist_base_port,
        )
        for index in range(endpoint_count)
    ]


def start_case_stack(
    args: argparse.Namespace, case: Case, directory: Path
) -> tuple[list[WorkerSpec], list[ManagedProcess]]:
    specs = worker_specs(args, case.endpoint_count)
    indexer_endpoint = f"http://127.0.0.1:{args.indexer_port}"
    managed: list[ManagedProcess] = []
    try:
        indexer = start_process(
            "kv-indexer-server",
            [str(args.indexer_server)],
            directory / "kv-indexer-server.log",
            cwd=args.router_cwd,
            env={"KV_INDEXER_LISTEN_ADDR": f"127.0.0.1:{args.indexer_port}"},
        )
        managed.append(indexer)
        wait_tcp(
            "127.0.0.1",
            args.indexer_port,
            timeout=args.indexer_start_timeout,
            process=indexer.process,
        )

        for spec in specs:
            managed.append(
                start_process(
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

        environment = simulator_environment(
            simulator_site=args.simulator_site,
            source_root=args.source_root,
            simulator_config=args.simulator_config,
        )
        workers: list[ManagedProcess] = []
        for spec in specs:
            command, worker_environment = simulator_worker_command(
                spec,
                python=args.python,
                simulator_config=args.simulator_config,
                model_path=args.model_path,
                tokenizer_path=args.tokenizer_path,
                max_total_tokens=args.max_total_tokens,
                max_running_requests=args.max_running_requests,
            )
            worker_environment.update(environment)
            worker = start_process(
                f"worker-{spec.index}",
                command,
                directory / "workers" / f"{spec.index}.log",
                cwd=args.source_root,
                env=worker_environment,
            )
            managed.append(worker)
            workers.append(worker)
        asyncio.run(
            wait_http_urls(
                tuple(f"{spec.url}/health" for spec in specs),
                timeout=args.worker_start_timeout,
            )
        )
        for spec, worker in zip(specs, workers):
            if worker.process.poll() is not None:
                raise RuntimeError(f"worker {spec.index} exited during startup")

        router_environment: dict[str, str] = {}
        if case.policy in {"cache_aware", "shortest_ttft"}:
            router_environment["RUST_LOG"] = "info,sgl_router::server::routes::chat=debug"
        router = start_process(
            f"router-{case.policy}",
            router_command(args, case, [spec.url for spec in specs], indexer_endpoint),
            directory / "router.log",
            cwd=args.router_cwd,
            env=router_environment,
        )
        managed.append(router)
        asyncio.run(
            wait_http_urls(
                (f"http://127.0.0.1:{args.router_port}/healthz",),
                timeout=args.router_start_timeout,
            )
        )
        time.sleep(args.router_settle_seconds)
        return specs, managed
    except Exception:
        stop_processes(managed)
        raise


def cache_monitor_usage(router_log: str) -> dict[str, int]:
    counts = {
        "cache_candidate_decisions": 0,
        "monitor_decisions": 0,
        "router_local_decisions": 0,
        "zero_snapshot_decisions": 0,
        "actual_cache_metrics": 0,
    }
    for line in router_log.splitlines():
        if "cache candidate winner" not in line:
            continue
        normalized = ANSI_ESCAPE_RE.sub("", line)
        source_marker = "prefill_pressure_source="
        version_marker = "load_snapshot_version="
        if source_marker not in normalized or version_marker not in normalized:
            raise RuntimeError("cache candidate winner has no load-monitor audit fields")
        source = normalized.split(source_marker, 1)[1].split()[0].strip('"')
        version = normalized.split(version_marker, 1)[1].split()[0].strip('"')
        counts["cache_candidate_decisions"] += 1
        if source in {"estimated_prefill_queue_ms", "monitor_fallback"}:
            counts["monitor_decisions"] += 1
        elif source == "router_local":
            counts["router_local_decisions"] += 1
        else:
            raise RuntimeError(f"unknown cache pressure source: {source}")
        if version == "0":
            counts["zero_snapshot_decisions"] += 1
    return counts


class RateGate:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0.0:
            raise ValueError("requests_per_second must be positive")
        self.interval = 1.0 / requests_per_second
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def wait_turn(self) -> None:
        loop = asyncio.get_running_loop()
        async with self.lock:
            now = loop.time()
            scheduled = max(now, self.next_at)
            self.next_at = scheduled + self.interval
        delay = scheduled - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)


def long_prefix(workload: str) -> str:
    scope = "multiturn" if workload == "tracelab_multiturn" else "shared-holder"
    return " ".join(f"{scope}_prefix_{index}" for index in range(1536))


def worker_cache_flush_urls(worker_urls: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{worker_url}/flush_cache" for worker_url in worker_urls)


async def flush_worker_caches(worker_urls: Sequence[str]) -> None:
    """丢弃 Simulator 启动 warmup 留下、且早于 bridge 订阅的 KV 状态。"""
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("HTTP fleet runner requires aiohttp") from error
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def flush(url: str) -> None:
            async with session.post(url) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"cache flush {url} returned {response.status}: {text[:400]}")
                await response.read()

        await asyncio.gather(*(flush(url) for url in worker_cache_flush_urls(worker_urls)))


async def post_direct_warmups(
    worker_urls: Sequence[str],
    *,
    model: str,
    prefix: str,
    holder_count: int,
) -> None:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("HTTP fleet runner requires aiohttp") from error
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def warm(url: str, holder: int) -> None:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"{prefix} warmup holder={holder}",
                    }
                ],
                "max_tokens": 4,
                "temperature": 0,
            }
            async with session.post(f"{url}/v1/chat/completions", json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"warmup {url} returned {response.status}: {text[:400]}")
                await response.read()

        await asyncio.gather(
            *(warm(url, index) for index, url in enumerate(worker_urls[:holder_count]))
        )


async def stream_requests(
    *,
    router_url: str,
    model: str,
    workload: str,
    request_count: int,
    qps: float,
    output_tokens: int,
) -> list[dict[str, object]]:
    try:
        import aiohttp
    except ImportError as error:
        raise RuntimeError("HTTP fleet runner requires aiohttp") from error
    prefix = long_prefix(workload)
    gate = RateGate(qps)
    semaphore = asyncio.Semaphore(256)
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def send(index: int) -> dict[str, object]:
            await gate.wait_turn()
            async with semaphore:
                session_id = f"{workload}-session-{index % max(8, request_count // 8)}"
                suffix = (
                    f"turn={index % 4} session={session_id} request={index}"
                    if workload == "tracelab_multiturn"
                    else f"shared_request={index}"
                )
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": f"{prefix} {suffix}"}],
                    "max_tokens": output_tokens,
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
                    headers={"X-SMG-Routing-Key": session_id},
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(
                            f"request {index} returned HTTP {response.status}: {text[:400]}"
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
                            choices = event.get("choices") if isinstance(event, dict) else None
                            if isinstance(choices, list) and choices:
                                delta = choices[0].get("delta")
                                content = delta.get("content") if isinstance(delta, dict) else None
                                if isinstance(content, str) and content and first_token_at is None:
                                    first_token_at = time.monotonic()
                            usage = event.get("usage") if isinstance(event, dict) else None
                            if isinstance(usage, dict) and isinstance(
                                usage.get("completion_tokens"), int
                            ):
                                completion_tokens = int(usage["completion_tokens"])
                finished = time.monotonic()
                if first_token_at is None:
                    first_token_at = finished
                return {
                    "request_index": index,
                    "session_id": session_id,
                    "ttft_ms": (first_token_at - started) * 1000.0,
                    "e2e_ms": (finished - started) * 1000.0,
                    "completion_tokens": completion_tokens,
                }

        return list(await asyncio.gather(*(send(index) for index in range(request_count))))


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile_value)))
    return ordered[position]


def summarize_case(
    requests: Sequence[Mapping[str, object]],
    *,
    elapsed_seconds: float,
    cache: Mapping[str, object],
    worker_success: Mapping[str, float],
    worker_urls: Sequence[str],
    policy_reasons: Mapping[str, float],
) -> dict[str, object]:
    if elapsed_seconds <= 0.0:
        raise ValueError("elapsed_seconds must be positive")
    ttft = [float(request["ttft_ms"]) for request in requests]
    e2e = [float(request["e2e_ms"]) for request in requests]
    completion_tokens = sum(
        int(request["completion_tokens"])
        for request in requests
        if isinstance(request.get("completion_tokens"), int)
    )
    worker_counts = [worker_success.get(worker_url, 0.0) for worker_url in worker_urls]
    mean = sum(worker_counts) / len(worker_counts)
    variance = sum((value - mean) ** 2 for value in worker_counts) / len(worker_counts)
    return {
        "request_count": len(requests),
        "request_errors": 0,
        "elapsed_seconds": elapsed_seconds,
        "throughput_rps": len(requests) / elapsed_seconds,
        "completion_tps": completion_tokens / elapsed_seconds,
        "ttft_ms": {"mean": sum(ttft) / len(ttft), "p95": percentile(ttft, 0.95)},
        "e2e_ms": {"mean": sum(e2e) / len(e2e), "p95": percentile(e2e, 0.95)},
        "cache": dict(cache),
        "worker_success": dict(worker_success),
        "worker_cv": 0.0 if mean == 0.0 else variance**0.5 / mean,
        "policy_reasons": dict(policy_reasons),
    }


def read_source_commit(source_root: Path) -> str:
    value = os.environ.get("SGLANG_SOURCE_COMMIT", "").strip()
    if value:
        return value
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "archive-without-git"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_or_verify_manifest(
    results_dir: Path, contract: Mapping[str, object], *, resume: bool
) -> None:
    manifest = results_dir / "manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        if payload.get("contract") != dict(contract):
            raise RuntimeError(f"manifest contract mismatch: {manifest}")
        if not resume:
            raise FileExistsError(f"results already exist; pass --resume: {results_dir}")
        return
    atomic_write_text(
        manifest,
        json.dumps({"contract": dict(contract)}, indent=2, sort_keys=True) + "\n",
    )


def case_complete(directory: Path, case: Case) -> bool:
    try:
        return (
            (directory / "COMPLETE").read_text().strip() == "ok"
            and json.loads((directory / "case.json").read_text()) == asdict(case)
            and (directory / "summary.json").stat().st_size > 0
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False


def archive_incomplete_case(directory: Path) -> Path:
    """保留失败 attempt，再把规范 case 路径交还给同合同的续跑。"""
    for attempt in range(1_000_000):
        archived = directory.with_name(f"{directory.name}.attempt-{attempt}")
        if not archived.exists():
            os.replace(directory, archived)
            atomic_write_text(archived / "RESUME_ARCHIVED", "incomplete case preserved for resume\n")
            return archived
    raise RuntimeError(f"too many archived attempts for {directory}")


def run_case(args: argparse.Namespace, case: Case) -> None:
    directory = args.results_dir / case.name
    if directory.exists():
        if args.resume and case_complete(directory, case):
            print(f"skip complete {case.name}", flush=True)
            return
        if not args.resume:
            raise FileExistsError(f"incomplete or incompatible case directory: {directory}")
        archived = archive_incomplete_case(directory)
        print(f"preserved incomplete {case.name} at {archived.name}; retrying", flush=True)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_write_text(directory / "case.json", json.dumps(asdict(case), indent=2, sort_keys=True) + "\n")
    atomic_write_text(args.results_dir / "CURRENT", case.name + "\n")
    managed: list[ManagedProcess] = []
    try:
        specs, managed = start_case_stack(args, case, directory)
        worker_urls = [spec.url for spec in specs]
        holder_count = min(args.max_cache_holders, case.endpoint_count)
        asyncio.run(flush_worker_caches(worker_urls))
        time.sleep(args.indexer_settle_seconds)
        asyncio.run(
            post_direct_warmups(
                worker_urls,
                model=str(args.model_path),
                prefix=long_prefix(case.workload),
                holder_count=holder_count,
            )
        )
        time.sleep(args.indexer_settle_seconds)
        worker_before = asyncio.run(fetch_texts(tuple(f"{url}/metrics" for url in worker_urls)))
        router_before = asyncio.run(fetch_texts((f"http://127.0.0.1:{args.router_port}/metrics",)))[0]
        worker_loads = asyncio.run(fetch_texts(tuple(f"{url}/v1/loads" for url in worker_urls)))
        atomic_write_text(directory / "router.measurement_before.prom", router_before)
        atomic_write_text(directory / "worker_loads_before.json", json.dumps(worker_loads, indent=2) + "\n")
        router_log = directory / "router.log"
        log_offset = router_log.stat().st_size
        started = time.monotonic()
        requests = asyncio.run(
            stream_requests(
                router_url=f"http://127.0.0.1:{args.router_port}",
                model=str(args.model_path),
                workload=case.workload,
                request_count=case.endpoint_count,
                qps=max(1.0, args.qps_per_worker * case.endpoint_count),
                output_tokens=args.output_tokens,
            )
        )
        elapsed_seconds = time.monotonic() - started
        worker_after = asyncio.run(fetch_texts(tuple(f"{url}/metrics" for url in worker_urls)))
        router_after = asyncio.run(fetch_texts((f"http://127.0.0.1:{args.router_port}/metrics",)))[0]
        cache = prefill_cache_summary(worker_before, worker_after)
        worker_success = metric_delta(
            worker_success_counts(router_before), worker_success_counts(router_after)
        )
        reasons = metric_delta(
            policy_reason_counts(router_before, case.policy),
            policy_reason_counts(router_after, case.policy),
        )
        audit: dict[str, int] | None = None
        if case.policy == "cache_aware":
            audit = cache_monitor_usage(router_log.read_text(errors="replace")[log_offset:])
            audit["actual_cache_metrics"] = int(
                float(cache["total_effective_tokens"]) > 0.0
            )
            require_native_cache_audit(audit)
        summary = summarize_case(
            requests,
            elapsed_seconds=elapsed_seconds,
            cache=cache,
            worker_success=worker_success,
            worker_urls=worker_urls,
            policy_reasons=reasons,
        )
        summary["native_cache_audit"] = audit
        summary["holder_count"] = holder_count
        atomic_write_text(
            directory / "requests.jsonl",
            "".join(json.dumps(request, sort_keys=True) + "\n" for request in requests),
        )
        atomic_write_text(directory / "router.prom", router_after)
        atomic_write_text(
            directory / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        atomic_write_text(directory / "COMPLETE", "ok\n")
    finally:
        stop_processes(managed)


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
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--endpoint-counts",
        default=",".join(str(count) for count in DEFAULT_ENDPOINT_COUNTS),
    )
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--http-base-port", type=int, default=31_000)
    parser.add_argument("--reporter-base-port", type=int, default=47_000)
    parser.add_argument("--kv-base-port", type=int, default=51_000)
    parser.add_argument("--dist-base-port", type=int, default=53_000)
    parser.add_argument("--router-port", type=int, default=30_380)
    parser.add_argument("--indexer-port", type=int, default=50_551)
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--max-running-requests", type=int, default=32)
    parser.add_argument(
        "--max-cache-holders",
        type=int,
        default=8,
        help="direct-warm worker count; 0 uses the first routed request as the cache seed",
    )
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--qps-per-worker", type=float, default=0.25)
    parser.add_argument("--indexer-start-timeout", type=float, default=60.0)
    parser.add_argument("--worker-start-timeout", type=float, default=300.0)
    parser.add_argument("--router-start-timeout", type=float, default=180.0)
    parser.add_argument("--router-settle-seconds", type=float, default=35.0)
    parser.add_argument("--indexer-settle-seconds", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.endpoint_counts = parse_endpoint_counts(args.endpoint_counts)
    args.policies = parse_csv(args.policies)
    args.workloads = parse_csv(args.workloads)
    unsupported = set(args.policies) - set(DEFAULT_POLICIES)
    if unsupported:
        parser.error(f"unsupported policies: {sorted(unsupported)}")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.max_total_tokens <= 0 or args.max_running_requests <= 0:
        parser.error("worker capacity arguments must be positive")
    if args.max_cache_holders < 0 or args.output_tokens <= 0 or args.qps_per_worker <= 0.0:
        parser.error("cache holders must be non-negative; output and QPS must be positive")
    if args.router_settle_seconds < 0.0 or args.indexer_settle_seconds < 0.0:
        parser.error("settle durations must be non-negative")
    largest = max(args.endpoint_counts)
    for base in (
        args.http_base_port,
        args.reporter_base_port,
        args.kv_base_port,
        args.dist_base_port,
    ):
        if not 1 <= base <= 65535 - (largest - 1):
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
            "results_dir",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"--execute requires: {', '.join('--' + name.replace('_', '-') for name in missing)}")
        for name in required:
            value = getattr(args, name)
            if isinstance(value, Path):
                setattr(args, name, value.resolve())
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    cases = build_cases(
        endpoint_counts=args.endpoint_counts,
        policies=args.policies,
        workloads=args.workloads,
        repeats=args.repeats,
    )
    if not args.execute:
        print(json.dumps({"mode": "dry-run", "cases": [asdict(case) | {"name": case.name} for case in cases]}, indent=2, sort_keys=True))
        return 0
    contract = {
        "schema_version": 1,
        "source_commit": read_source_commit(args.source_root),
        "router_binary_sha256": sha256_file(args.router_binary),
        "indexer_server_sha256": sha256_file(args.indexer_server),
        "indexer_bridge_sha256": sha256_file(args.indexer_bridge),
        "endpoint_counts": list(args.endpoint_counts),
        "policies": list(args.policies),
        "workloads": list(args.workloads),
        "repeats": args.repeats,
        "max_total_tokens": args.max_total_tokens,
        "max_running_requests": args.max_running_requests,
        "output_tokens": args.output_tokens,
        "qps_per_worker": args.qps_per_worker,
        "max_cache_holders": args.max_cache_holders,
        "simulator_config": str(args.simulator_config),
        "cases": [asdict(case) | {"name": case.name} for case in cases],
    }
    write_or_verify_manifest(args.results_dir, contract, resume=args.resume)
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] run {case.name}", flush=True)
        run_case(args, case)
    atomic_write_text(args.results_dir / "RUN_COMPLETE", "ok\n")
    atomic_write_text(args.results_dir / "CURRENT", "complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
