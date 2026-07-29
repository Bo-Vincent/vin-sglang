"""Compare cold checkpoint startup with two remote weight-reuse paths.

The benchmark keeps a source SGLang service online while target services start
in one of three modes:

* ``cold``: load the target from checkpoint files after dropping page cache.
* ``legacy``: use the homogeneous remote-instance TransferEngine path.
* ``manifest``: use the runtime-manifest path, including heterogeneous TP.

Every deterministic source and target response is appended to
``responses.jsonl``. Source responses must equal the source baseline; reuse
targets must equal a cold-loaded target with the same target topology. Both
comparisons require exact generated text, input/output token IDs, token counts,
and finish reason, plus input/output token logprobs within explicit tolerances.
Dynamic metadata such as request IDs, cache counters, and latency is retained in
the raw response but intentionally excluded.

Homogeneous example (runs all three modes by default):

PYTHONPATH=python python test/manual/benchmark_remote_instance_service_startup.py \
  --model /models/Qwen3.5-0.8B --source-gpus 0,1 --target-gpus 2,3 \
  --source-tp-size 2 --target-tp-size 2 --drop-page-cache --iterations 6

Large-model legacy example (runtime-manifest semantics may be model-specific):

PYTHONPATH=python python test/manual/benchmark_remote_instance_service_startup.py \
  --model /models/Qwen2-72B --source-gpus 0,1 --target-gpus 2,3 \
  --source-tp-size 2 --target-tp-size 2 --modes cold,legacy \
  --drop-page-cache --iterations 6

Heterogeneous example (legacy is reported as ineligible when TPs differ):

PYTHONPATH=python python test/manual/benchmark_remote_instance_service_startup.py \
  --model /models/Qwen3.5-0.8B --source-gpus 0,1 --target-gpus 2,3,4,5 \
  --source-tp-size 2 --target-tp-size 4 --drop-page-cache --iterations 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import requests

ALL_MODES = ("cold", "legacy", "manifest")
REUSE_MODES = ("legacy", "manifest")
MIN_PERCENTILE_SAMPLES = 5
MIN_MEASURED_ITERATIONS = 5
SERVER_TERMINATION_GRACE_S = 30.0
SERVER_KILL_TIMEOUT_S = 30.0
REUSE_REQUIRED_LOG_MARKERS = {
    "legacy": (
        "Loading weights from remote instance ...",
        "TransferEngine memory regions have been successfully registered.",
    ),
    "manifest": (
        "TransferEngine memory regions have been successfully registered.",
        "Loaded heterogeneous remote-instance weights:",
        "manifest_format=placement_binding_v1",
        "transfer_id=",
        "release_success=true",
    ),
}
REUSE_FORBIDDEN_LOG_MARKERS = (
    "Fallback load_format to 'auto'",
    "using the runtime-v1 target builder",
    "manifest_format=runtime_v1",
    "Cannot acquire remote weight transfer session",
    "Heterogeneous remote-instance weight loading failed",
    "Failed to load weights from remote instance via transfer engine",
    "completion remains unknown",
    "Failed to finish remote weight transfer session",
    "Loaded weights but failed to release source transfer session",
    "Remote weight transfer lease renewal failed",
    "source weight transfer lease renew failed",
)


def _p95(values: list[float]) -> float | None:
    if len(values) < MIN_PERCENTILE_SAMPLES:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def _coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < MIN_PERCENTILE_SAMPLES:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return None
    return statistics.pstdev(values) / mean


def _series_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": statistics.median(values),
        "p95": _p95(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "cv": _coefficient_of_variation(values),
    }


@dataclass
class ServerProcess:
    process: subprocess.Popen
    log_file: Any
    log_path: Path
    started_at: float
    port: int
    ready_identity: dict[str, Any] | None = None


class ResponseRecorder:
    """Persist raw inference responses as they arrive, including probe traffic."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._sequence = 0

    def record_response(
        self,
        *,
        mode: str,
        iteration: int,
        endpoint: str,
        phase: str,
        latency_s: float,
        response: Any,
        deterministic_response: dict[str, Any],
        expected: dict[str, Any] | None,
        logprob_atol: float,
        logprob_rtol: float,
        server_identity: dict[str, Any],
    ) -> dict[str, Any]:
        consistent = expected is None or _responses_match(
            deterministic_response,
            expected,
            atol=logprob_atol,
            rtol=logprob_rtol,
        )
        entry = {
            "kind": "response",
            "mode": mode,
            "iteration": iteration,
            "endpoint": endpoint,
            "phase": phase,
            "captured_at_unix_s": time.time(),
            "latency_s": latency_s,
            "consistent_with_expected": consistent,
            "deterministic_response": deterministic_response,
            "server_identity": server_identity,
            "raw_response": response,
        }
        return self._write(entry)

    def record_error(
        self,
        *,
        mode: str,
        iteration: int,
        endpoint: str,
        phase: str,
        error: BaseException,
    ) -> dict[str, Any]:
        return self._write(
            {
                "kind": "error",
                "mode": mode,
                "iteration": iteration,
                "endpoint": endpoint,
                "phase": phase,
                "captured_at_unix_s": time.time(),
                "error": repr(error),
            }
        )

    def close(self) -> None:
        self._file.close()

    def _write(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            entry["sequence"] = self._sequence
            self._sequence += 1
            self._file.write(json.dumps(entry, sort_keys=True) + "\n")
            self._file.flush()
        return entry


class SourceProbe:
    """Continuously verify source inference while a target service starts."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        mode: str,
        iteration: int,
        expected: dict[str, Any],
        recorder: ResponseRecorder,
        server: ServerProcess,
    ) -> None:
        self.args = args
        self.mode = mode
        self.iteration = iteration
        self.expected = expected
        self.recorder = recorder
        self.server = server
        self.latencies: list[float] = []
        self.errors: list[str] = []
        self.mismatches: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=self.args.request_timeout_s + 5)
        if self._thread.is_alive():
            self.errors.append("source probe thread did not stop before timeout")
        ordered = sorted(self.latencies)
        return {
            "success_count": len(ordered),
            "error_count": len(self.errors),
            "mismatch_count": len(self.mismatches),
            "errors": self.errors[:5],
            "mismatches": self.mismatches[:5],
            "latency_p50_s": statistics.median(ordered) if ordered else None,
            "latency_p95_s": _p95(ordered),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                measurement = _generate_and_record(
                    self.args,
                    server=self.server,
                    mode=self.mode,
                    iteration=self.iteration,
                    endpoint="source",
                    phase="during_target_start",
                    expected=self.expected,
                    recorder=self.recorder,
                )
                self.latencies.append(measurement["latency_s"])
                if not measurement["consistent_with_expected"]:
                    self.mismatches.append(
                        {
                            "sequence": measurement["response_sequence"],
                            "deterministic_response": measurement[
                                "deterministic_response"
                            ],
                        }
                    )
            except Exception as error:
                self.errors.append(repr(error))
            self._stop.wait(self.args.probe_interval_s)


def _parse_gpus(value: str) -> str:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or any(not item.isdigit() for item in values):
        raise argparse.ArgumentTypeError("GPU list must contain comma-separated IDs")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GPU list must not contain duplicates")
    return ",".join(values)


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(modes) - set(ALL_MODES))
    if not modes:
        raise argparse.ArgumentTypeError("at least one benchmark mode is required")
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown benchmark modes {unknown}; choose from {list(ALL_MODES)}"
        )
    if len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("benchmark modes must not contain duplicates")
    return modes


def _drop_page_cache() -> None:
    os.sync()
    Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="ascii")


def _server_command(
    args: argparse.Namespace,
    *,
    port: int,
    load_mode: str,
) -> list[str]:
    is_source = load_mode == "source"
    tp_size = args.source_tp_size if is_source else args.target_tp_size
    pp_size = args.source_pp_size if is_source else args.target_pp_size
    dp_size = args.source_dp_size if is_source else args.target_dp_size
    ep_size = args.source_ep_size if is_source else args.target_ep_size
    command = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--tokenizer-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        str(tp_size),
        "--pp-size",
        str(pp_size),
        "--dp-size",
        str(dp_size),
        "--ep-size",
        str(ep_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--disable-cuda-graph",
    ]
    if args.attention_backend:
        command.extend(["--attention-backend", args.attention_backend])
    if args.mm_attention_backend:
        command.extend(["--mm-attention-backend", args.mm_attention_backend])
    if args.sampling_backend:
        command.extend(["--sampling-backend", args.sampling_backend])
    if args.disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")
    if args.disable_shared_experts_fusion:
        command.append("--disable-shared-experts-fusion")
    if args.moe_runner_backend:
        command.extend(["--moe-runner-backend", args.moe_runner_backend])
    if is_source:
        command.extend(
            [
                "--remote-instance-weight-loader-start-seed-via-transfer-engine",
                "--engine-info-bootstrap-port",
                str(args.bootstrap_port),
            ]
        )
        if "manifest" in args.modes:
            command.append("--enable-weight-runtime-manifest")
    elif load_mode in REUSE_MODES:
        command.extend(
            [
                "--load-format",
                "remote_instance",
                "--remote-instance-weight-loader-backend",
                "transfer_engine",
                "--remote-instance-weight-loader-seed-instance-ip",
                "127.0.0.1",
                "--remote-instance-weight-loader-seed-instance-service-port",
                str(args.source_port),
            ]
        )
        if load_mode == "manifest":
            command.append("--enable-weight-runtime-manifest")
    elif load_mode != "cold":
        raise ValueError(f"unknown load mode: {load_mode}")
    return command


def _start_server(
    args: argparse.Namespace,
    *,
    gpus: str,
    port: int,
    load_mode: str,
    log_path: Path,
) -> ServerProcess:
    _assert_port_available(port)
    if load_mode == "source":
        _assert_port_available(args.bootstrap_port)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpus
    environment["MOONCAKE_PROTOCOL"] = args.protocol
    if args.transport_device:
        environment["MOONCAKE_DEVICE"] = args.transport_device
    environment["PYTHONUNBUFFERED"] = "1"
    log_file = log_path.open("w", encoding="utf-8")
    started_at = time.perf_counter()
    try:
        process = subprocess.Popen(
            _server_command(args, port=port, load_mode=load_mode),
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise
    return ServerProcess(
        process=process,
        log_file=log_file,
        log_path=log_path,
        started_at=started_at,
        port=port,
    )


def _listening_port_owner_pids(port: int) -> set[int]:
    loopback_hosts = {
        "0.0.0.0",
        "127.0.0.1",
        "::",
        "::1",
        "::ffff:127.0.0.1",
    }
    owners = set()
    for connection in psutil.net_connections(kind="inet"):
        local_address = connection.laddr
        if not local_address or connection.status != psutil.CONN_LISTEN:
            continue
        if local_address.port != port or local_address.ip not in loopback_hosts:
            continue
        if connection.pid is not None:
            owners.add(connection.pid)
    return owners


def _process_tree_pids(root_pid: int) -> set[int]:
    try:
        root = psutil.Process(root_pid)
        return {root_pid, *(child.pid for child in root.children(recursive=True))}
    except (psutil.AccessDenied, psutil.NoSuchProcess) as error:
        raise RuntimeError(
            f"cannot inspect spawned server process tree rooted at PID {root_pid}"
        ) from error


def _assert_process_alive(server: ServerProcess) -> None:
    returncode = server.process.poll()
    if returncode is not None:
        raise RuntimeError(
            f"server exited with {returncode}:\n{_tail(server.log_path)}"
        )


def _assert_port_available(port: int) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    except OSError as error:
        owners = sorted(_listening_port_owner_pids(port))
        owner_text = f" by listening PID(s) {owners}" if owners else ""
        raise RuntimeError(f"port {port} is already in use{owner_text}") from error


def _wait_port_released(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            _assert_port_available(port)
            return
        except RuntimeError as error:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise RuntimeError(
                    f"port {port} was not released within {timeout_s:.1f}s"
                ) from error
            time.sleep(min(0.1, remaining_s))


def _assert_server_identity(
    server: ServerProcess, owner_pids: set[int] | None = None
) -> dict[str, Any]:
    _assert_process_alive(server)
    if owner_pids is None:
        owner_pids = _listening_port_owner_pids(server.port)
    process_tree_pids = _process_tree_pids(server.process.pid)
    unexpected_owners = owner_pids - process_tree_pids
    if unexpected_owners:
        raise RuntimeError(
            f"listener owner PID(s) {sorted(unexpected_owners)} for port "
            f"{server.port} are outside spawned process tree "
            f"{sorted(process_tree_pids)}"
        )
    if not owner_pids:
        raise RuntimeError(
            f"port {server.port} has no auditable listener owner for spawned "
            f"process tree {sorted(process_tree_pids)}"
        )
    _assert_process_alive(server)
    return {
        "root_pid": server.process.pid,
        "port": server.port,
        "process_tree_pids": sorted(process_tree_pids),
        "listener_owner_pids": sorted(owner_pids),
    }


def _tail(path: Path, lines: int = 80) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError as error:
        return repr(error)


def _assert_reuse_log_contract(mode: str, log_path: Path) -> dict[str, Any]:
    if mode not in REUSE_MODES:
        raise ValueError(f"log contract is only defined for reuse modes: {mode}")
    text = log_path.read_text(errors="replace")
    forbidden = [marker for marker in REUSE_FORBIDDEN_LOG_MARKERS if marker in text]
    if forbidden:
        raise RuntimeError(
            f"{mode} log contains forbidden log marker(s) {forbidden}: {log_path}"
        )
    required = REUSE_REQUIRED_LOG_MARKERS[mode]
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            f"{mode} log is missing required log marker(s) {missing}: {log_path}"
        )
    return {
        "passed": True,
        "required_markers": list(required),
        "forbidden_markers_checked": list(REUSE_FORBIDDEN_LOG_MARKERS),
    }


def _parse_manifest_transfer_metrics(
    log_path: Path,
    *,
    expected_target_ranks: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    marker = "Loaded heterogeneous remote-instance weights:"
    lines = [
        line
        for line in log_path.read_text(errors="replace").splitlines()
        if marker in line
    ]
    if not lines:
        raise RuntimeError(f"manifest transfer metrics are missing: {log_path}")

    rank_metrics = []
    for line in lines:
        rank_match = re.search(r"\btarget_rank=(\d+),", line)
        match = re.search(
            r"\bbytes=(\d+), "
            r"compact_operations=(\d+), segments=(\d+), "
            r"elapsed=([0-9]+(?:\.[0-9]+)?)s; phases: (.+)$",
            line,
        )
        if match is None:
            raise RuntimeError(
                f"manifest transfer metrics are malformed in {log_path}: {line}"
            )
        phases_s = {
            name: float(value)
            for name, value in re.findall(
                r"\b([a-z_]+)=([0-9]+(?:\.[0-9]+)?)s", match.group(5)
            )
        }
        rank_metrics.append(
            {
                "target_rank": (
                    None if rank_match is None else int(rank_match.group(1))
                ),
                "logical_bytes": int(match.group(1)),
                "compact_operations": int(match.group(2)),
                "segments": int(match.group(3)),
                "elapsed_s": float(match.group(4)),
                "phases_s": phases_s,
            }
        )

    rank_ids = [metric["target_rank"] for metric in rank_metrics]
    if len(rank_metrics) > 1 and (
        any(rank is None for rank in rank_ids) or len(set(rank_ids)) != len(rank_ids)
    ):
        raise RuntimeError(
            "manifest transfer metrics need unique target_rank values for "
            f"multiple rank records in {log_path}"
        )

    if expected_target_ranks is not None:
        if any(rank is None for rank in rank_ids) or tuple(sorted(rank_ids)) != tuple(
            expected_target_ranks
        ):
            raise RuntimeError(
                "manifest transfer metrics do not cover the expected target ranks: "
                f"expected={expected_target_ranks}, actual={tuple(sorted(rank_ids))}"
            )

    logical_bytes = sum(metric["logical_bytes"] for metric in rank_metrics)
    elapsed_s = max(metric["elapsed_s"] for metric in rank_metrics)
    phases_s = {
        phase: max(
            metric["phases_s"][phase]
            for metric in rank_metrics
            if phase in metric["phases_s"]
        )
        for phase in sorted(
            {phase for metric in rank_metrics for phase in metric["phases_s"]}
        )
    }
    data_transfer_s = phases_s.get("data_transfer")
    if data_transfer_s is None or data_transfer_s <= 0 or elapsed_s <= 0:
        raise RuntimeError(
            f"manifest transfer durations must be positive in {log_path}: "
            f"elapsed={elapsed_s}, data_transfer={data_transfer_s}"
        )

    data_transfer_gb_per_s = logical_bytes / data_transfer_s / 1e9
    return {
        "logical_bytes": logical_bytes,
        "compact_operations": sum(
            metric["compact_operations"] for metric in rank_metrics
        ),
        "segments": sum(metric["segments"] for metric in rank_metrics),
        "elapsed_s": elapsed_s,
        "phases_s": phases_s,
        "data_transfer_logical_gb_per_s": data_transfer_gb_per_s,
        "data_transfer_logical_gbps": data_transfer_gb_per_s * 8,
        "end_to_end_logical_gb_per_s": logical_bytes / elapsed_s / 1e9,
        "rank_metrics": rank_metrics,
    }


def _wait_ready(server: ServerProcess, port: int, timeout_s: float) -> float:
    if port != server.port:
        raise ValueError(
            f"readiness port {port} does not match spawned server port {server.port}"
        )
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health_generate"
    while time.monotonic() < deadline:
        _assert_process_alive(server)
        owner_pids = _listening_port_owner_pids(server.port)
        if not owner_pids:
            time.sleep(0.2)
            continue
        before_health = _assert_server_identity(server, owner_pids)
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                server.ready_identity = {
                    "before_health": before_health,
                    "after_health": _assert_server_identity(server),
                }
                return time.perf_counter() - server.started_at
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise TimeoutError(
        f"server did not become ready in {timeout_s}s:\n{_tail(server.log_path)}"
    )


def _inference_request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "text": args.prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "sampling_seed": args.sampling_seed,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }


def _token_ids(meta_info: dict[str, Any], field: str) -> list[int]:
    entries = meta_info.get(field)
    if not isinstance(entries, list):
        raise ValueError(f"response meta_info.{field} must be a list")
    token_ids = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(f"invalid {field}[{index}] entry: {entry!r}")
        token_id = entry[1]
        if not isinstance(token_id, int):
            raise ValueError(f"invalid {field}[{index}] token ID: {token_id!r}")
        token_ids.append(token_id)
    return token_ids


def _token_logprobs(meta_info: dict[str, Any], field: str) -> list[float | None]:
    entries = meta_info.get(field)
    if not isinstance(entries, list):
        raise ValueError(f"response meta_info.{field} must be a list")
    values = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(f"invalid {field}[{index}] entry: {entry!r}")
        value = entry[0]
        if value is None:
            values.append(None)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
        else:
            raise ValueError(f"invalid {field}[{index}] logprob: {value!r}")
    return values


def _deterministic_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict) or not isinstance(response.get("text"), str):
        raise ValueError("generate response must be an object with a string text field")
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError("generate response must contain a meta_info object")
    return {
        "text": response["text"],
        "input_token_ids": _token_ids(meta_info, "input_token_logprobs"),
        "output_token_ids": _token_ids(meta_info, "output_token_logprobs"),
        "input_token_logprobs": _token_logprobs(meta_info, "input_token_logprobs"),
        "output_token_logprobs": _token_logprobs(meta_info, "output_token_logprobs"),
        "prompt_tokens": meta_info.get("prompt_tokens"),
        "completion_tokens": meta_info.get("completion_tokens"),
        "finish_reason": meta_info.get("finish_reason"),
    }


def _responses_match(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> bool:
    logprob_fields = ("input_token_logprobs", "output_token_logprobs")
    if actual.keys() != expected.keys():
        return False
    for key in actual.keys() - set(logprob_fields):
        if actual[key] != expected[key]:
            return False
    for field in logprob_fields:
        actual_values = actual[field]
        expected_values = expected[field]
        if len(actual_values) != len(expected_values):
            return False
        for actual_value, expected_value in zip(actual_values, expected_values):
            if actual_value is None or expected_value is None:
                if actual_value is not expected_value:
                    return False
            elif not math.isclose(
                actual_value,
                expected_value,
                abs_tol=atol,
                rel_tol=rtol,
            ):
                return False
    return True


def _generate(
    args: argparse.Namespace, server: ServerProcess
) -> tuple[float, Any, dict[str, Any]]:
    before_request = _assert_server_identity(server)
    started = time.perf_counter()
    response = requests.post(
        f"http://127.0.0.1:{server.port}/generate",
        json=_inference_request(args),
        timeout=args.request_timeout_s,
    )
    response.raise_for_status()
    latency_s = time.perf_counter() - started
    after_response = _assert_server_identity(server)
    return (
        latency_s,
        response.json(),
        {
            "before_request": before_request,
            "after_response": after_response,
        },
    )


def _generate_and_record(
    args: argparse.Namespace,
    *,
    server: ServerProcess,
    mode: str,
    iteration: int,
    endpoint: str,
    phase: str,
    expected: dict[str, Any] | None,
    recorder: ResponseRecorder,
) -> dict[str, Any]:
    try:
        latency_s, response, server_identity = _generate(args, server)
        deterministic_response = _deterministic_response(response)
    except Exception as error:
        recorder.record_error(
            mode=mode,
            iteration=iteration,
            endpoint=endpoint,
            phase=phase,
            error=error,
        )
        raise
    record = recorder.record_response(
        mode=mode,
        iteration=iteration,
        endpoint=endpoint,
        phase=phase,
        latency_s=latency_s,
        response=response,
        deterministic_response=deterministic_response,
        expected=expected,
        logprob_atol=args.logprob_atol,
        logprob_rtol=args.logprob_rtol,
        server_identity=server_identity,
    )
    return {
        "latency_s": latency_s,
        "response_sequence": record["sequence"],
        "consistent_with_expected": record["consistent_with_expected"],
        "deterministic_response": deterministic_response,
        "server_identity": server_identity,
    }


def _collect_source_baseline(
    args: argparse.Namespace,
    recorder: ResponseRecorder,
    source_server: ServerProcess,
) -> dict[str, Any]:
    warmup = _generate_and_record(
        args,
        server=source_server,
        mode="source",
        iteration=-1,
        endpoint="source",
        phase="baseline_warmup",
        expected=None,
        recorder=recorder,
    )
    expected = warmup["deterministic_response"]
    measurements = []
    for sample in range(args.source_baseline_samples):
        measurement = _generate_and_record(
            args,
            server=source_server,
            mode="source",
            iteration=-1,
            endpoint="source",
            phase=f"baseline_sample_{sample}",
            expected=expected,
            recorder=recorder,
        )
        if not measurement["consistent_with_expected"]:
            raise RuntimeError(
                f"source baseline sample {sample} changed deterministic response"
            )
        measurements.append(measurement)
        if sample + 1 < args.source_baseline_samples:
            time.sleep(args.probe_interval_s)
    latencies = [measurement["latency_s"] for measurement in measurements]
    return {
        "deterministic_response": expected,
        "warmup_response_sequence": warmup["response_sequence"],
        "response_sequences": [
            measurement["response_sequence"] for measurement in measurements
        ],
        "sample_count": len(latencies),
        "latency_p50_s": statistics.median(latencies),
        "latency_p95_s": _p95(latencies),
    }


def _process_group_alive(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_process_group_exit(process_group: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while _process_group_alive(process_group):
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return False
        time.sleep(min(0.1, remaining_s))
    return True


def _stop_server(server: ServerProcess) -> None:
    process_group = server.process.pid

    def signal_group(sig: signal.Signals) -> None:
        try:
            os.killpg(process_group, sig)
        except ProcessLookupError:
            pass

    try:
        signal_group(signal.SIGTERM)
        term_deadline = time.monotonic() + SERVER_TERMINATION_GRACE_S
        if server.process.poll() is None:
            try:
                server.process.wait(timeout=max(0.0, term_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass

        group_exited = _wait_process_group_exit(
            process_group,
            timeout_s=max(0.0, term_deadline - time.monotonic()),
        )
        if not group_exited:
            signal_group(signal.SIGKILL)
            try:
                server.process.wait(timeout=SERVER_KILL_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pass
            if not _wait_process_group_exit(
                process_group, timeout_s=SERVER_KILL_TIMEOUT_S
            ):
                raise RuntimeError(
                    f"server process group {process_group} did not exit after SIGKILL"
                )
        _wait_port_released(server.port, timeout_s=30.0)
    finally:
        server.log_file.close()


def _assert_iteration_consistency(
    *,
    mode: str,
    iteration: int,
    measurements: list[tuple[str, dict[str, Any]]],
    source_probe: dict[str, Any],
    source_baseline_p95_s: float,
    max_source_probe_p95_ratio: float,
    min_source_probe_samples: int,
    responses_path: Path,
) -> None:
    failures = [
        label
        for label, measurement in measurements
        if not measurement["consistent_with_expected"]
    ]
    if source_probe["success_count"] == 0:
        failures.append("source probe produced no successful inference")
    if source_probe["error_count"]:
        failures.append(f"source probe errors={source_probe['error_count']}")
    if source_probe["mismatch_count"]:
        failures.append(f"source probe mismatches={source_probe['mismatch_count']}")
    if source_probe["success_count"] < min_source_probe_samples:
        failures.append(
            f"source probe samples={source_probe['success_count']} fewer than "
            f"{min_source_probe_samples}"
        )
    source_probe_p95_s = source_probe.get("latency_p95_s")
    source_probe_p95_limit_s = source_baseline_p95_s * max_source_probe_p95_ratio
    if source_probe_p95_s is not None and source_probe_p95_s > source_probe_p95_limit_s:
        failures.append(
            "source probe p95 "
            f"{source_probe_p95_s:.6f}s exceeds "
            f"{source_probe_p95_limit_s:.6f}s"
        )
    if failures:
        raise RuntimeError(
            f"{mode} iteration {iteration} failed strict inference consistency: "
            f"{failures}; inspect {responses_path}"
        )


def _run_target(
    args: argparse.Namespace,
    *,
    source_server: ServerProcess,
    mode: str,
    iteration: int,
    output_dir: Path,
    source_baseline: dict[str, Any],
    source_baseline_p95_s: float,
    target_expected: dict[str, Any] | None,
    recorder: ResponseRecorder,
    expected_target_ranks: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if args.drop_page_cache:
        _drop_page_cache()

    before = _generate_and_record(
        args,
        server=source_server,
        mode=mode,
        iteration=iteration,
        endpoint="source",
        phase="before_target_start",
        expected=source_baseline,
        recorder=recorder,
    )
    probe = SourceProbe(
        args,
        mode=mode,
        iteration=iteration,
        expected=source_baseline,
        recorder=recorder,
        server=source_server,
    )
    server: ServerProcess | None = None
    probe_summary: dict[str, Any] | None = None
    probe.start()
    try:
        server = _start_server(
            args,
            gpus=args.target_gpus,
            port=args.target_port,
            load_mode=mode,
            log_path=output_dir / f"{mode}-{iteration}.log",
        )
        ready_s = _wait_ready(server, args.target_port, args.timeout_s)
        target = _generate_and_record(
            args,
            server=server,
            mode=mode,
            iteration=iteration,
            endpoint="target",
            phase="after_target_ready",
            expected=target_expected,
            recorder=recorder,
        )
        probe_summary = probe.stop()
        after = _generate_and_record(
            args,
            server=source_server,
            mode=mode,
            iteration=iteration,
            endpoint="source",
            phase="after_target_ready",
            expected=source_baseline,
            recorder=recorder,
        )
        _assert_iteration_consistency(
            mode=mode,
            iteration=iteration,
            measurements=[
                ("source before target start", before),
                ("target after ready", target),
                ("source after target ready", after),
            ],
            source_probe=probe_summary,
            source_baseline_p95_s=source_baseline_p95_s,
            max_source_probe_p95_ratio=args.max_source_probe_p95_ratio,
            min_source_probe_samples=args.min_source_probe_samples,
            responses_path=recorder.path,
        )
        reuse_log_contract = (
            _assert_reuse_log_contract(mode, server.log_path)
            if mode in REUSE_MODES
            else None
        )
        transfer_metrics = (
            _parse_manifest_transfer_metrics(
                server.log_path,
                expected_target_ranks=expected_target_ranks,
            )
            if mode == "manifest"
            else None
        )
        return {
            "mode": mode,
            "iteration": iteration,
            "spawn_to_ready_s": ready_s,
            "first_generation_s": target["latency_s"],
            "target_deterministic_response": target["deterministic_response"],
            "response_sequences": {
                "source_before": before["response_sequence"],
                "target": target["response_sequence"],
                "source_after": after["response_sequence"],
            },
            "source_probe": probe_summary,
            "reuse_log_contract": reuse_log_contract,
            "transfer_metrics": transfer_metrics,
            "server_identity": {
                "root_pid": server.process.pid,
                "port": server.port,
                "ready": server.ready_identity,
                "generation": target["server_identity"],
            },
            "log": str(server.log_path),
        }
    finally:
        if probe.is_alive():
            probe.stop()
        if server is not None:
            _stop_server(server)
        time.sleep(args.inter_run_delay_s)


def _mode_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [record["spawn_to_ready_s"] for record in records]
    first_generation = [record["first_generation_s"] for record in records]
    summary = {
        "iterations": len(records),
        "spawn_to_ready_p50_s": statistics.median(ready),
        "spawn_to_ready_p95_s": _p95(ready),
        "spawn_to_ready_mean_s": statistics.mean(ready),
        "spawn_to_ready_cv": _coefficient_of_variation(ready),
        "first_generation_p50_s": statistics.median(first_generation),
        "first_generation_p95_s": _p95(first_generation),
        "first_generation_mean_s": statistics.mean(first_generation),
        "first_generation_cv": _coefficient_of_variation(first_generation),
        "source_probe_success_count": sum(
            record["source_probe"]["success_count"] for record in records
        ),
        "source_probe_error_count": sum(
            record["source_probe"]["error_count"] for record in records
        ),
        "source_probe_mismatch_count": sum(
            record["source_probe"]["mismatch_count"] for record in records
        ),
    }
    transfer_metrics = [
        record["transfer_metrics"]
        for record in records
        if record.get("transfer_metrics") is not None
    ]
    if transfer_metrics:
        phases = sorted(
            {name for metrics in transfer_metrics for name in metrics["phases_s"]}
        )
        summary["transfer"] = {
            "iterations": len(transfer_metrics),
            "logical_bytes": _series_summary(
                [metrics["logical_bytes"] for metrics in transfer_metrics]
            ),
            "compact_operations": _series_summary(
                [metrics["compact_operations"] for metrics in transfer_metrics]
            ),
            "segments": _series_summary(
                [metrics["segments"] for metrics in transfer_metrics]
            ),
            "elapsed_s": _series_summary(
                [metrics["elapsed_s"] for metrics in transfer_metrics]
            ),
            "data_transfer_logical_gb_per_s_p50": statistics.median(
                [
                    metrics["data_transfer_logical_gb_per_s"]
                    for metrics in transfer_metrics
                ]
            ),
            "data_transfer_logical_gb_per_s_p95": _p95(
                [
                    metrics["data_transfer_logical_gb_per_s"]
                    for metrics in transfer_metrics
                ]
            ),
            "data_transfer_logical_gb_per_s_cv": _coefficient_of_variation(
                [
                    metrics["data_transfer_logical_gb_per_s"]
                    for metrics in transfer_metrics
                ]
            ),
            "data_transfer_logical_gbps_p50": statistics.median(
                [metrics["data_transfer_logical_gbps"] for metrics in transfer_metrics]
            ),
            "end_to_end_logical_gb_per_s_p50": statistics.median(
                [metrics["end_to_end_logical_gb_per_s"] for metrics in transfer_metrics]
            ),
            "phases_s": {
                phase: _series_summary(
                    [
                        metrics["phases_s"][phase]
                        for metrics in transfer_metrics
                        if phase in metrics["phases_s"]
                    ]
                )
                for phase in phases
            },
        }
        per_rank = {}
        for metrics in transfer_metrics:
            for rank_metric in metrics.get("rank_metrics", ()):
                rank = rank_metric["target_rank"]
                if rank is None:
                    continue
                per_rank.setdefault(rank, []).append(rank_metric)
        if per_rank:
            summary["transfer"]["per_rank"] = {
                str(rank): {
                    metric_name: _series_summary(
                        [metric[metric_name] for metric in rank_records]
                    )
                    for metric_name in (
                        "logical_bytes",
                        "compact_operations",
                        "segments",
                        "elapsed_s",
                    )
                    for rank_records in (records_for_rank,)
                }
                for rank, records_for_rank in sorted(per_rank.items())
            }
    return summary


def _reuse_comparison(
    cold: dict[str, Any],
    reuse: dict[str, Any],
    *,
    max_reuse_to_cold_ratio: float,
) -> dict[str, Any]:
    cold_p50 = cold["spawn_to_ready_p50_s"]
    cold_p95 = cold["spawn_to_ready_p95_s"]
    cold_mean = cold["spawn_to_ready_mean_s"]
    reuse_p50 = reuse["spawn_to_ready_p50_s"]
    reuse_p95 = reuse["spawn_to_ready_p95_s"]
    reuse_mean = reuse["spawn_to_ready_mean_s"]
    threshold_s = cold_p50 * max_reuse_to_cold_ratio
    return {
        "cold_spawn_to_ready_p50_s": cold_p50,
        "reuse_spawn_to_ready_p50_s": reuse_p50,
        "cold_spawn_to_ready_p95_s": cold_p95,
        "reuse_spawn_to_ready_p95_s": reuse_p95,
        "cold_spawn_to_ready_mean_s": cold_mean,
        "reuse_spawn_to_ready_mean_s": reuse_mean,
        "p50_speedup": cold_p50 / reuse_p50,
        "p95_speedup": (
            cold_p95 / reuse_p95
            if cold_p95 is not None and reuse_p95 is not None
            else None
        ),
        "mean_speedup": cold_mean / reuse_mean,
        "reuse_to_cold_p50_ratio": reuse_p50 / cold_p50,
        "p50_improvement_ratio": (cold_p50 - reuse_p50) / cold_p50,
        "max_reuse_to_cold_ratio": max_reuse_to_cold_ratio,
        "reuse_p50_must_be_lte_s": threshold_s,
        "passes_significant_improvement_threshold": reuse_p50 <= threshold_s,
    }


def _eligible_modes(args: argparse.Namespace) -> tuple[list[str], list[dict[str, str]]]:
    executed = []
    skipped = []
    for mode in args.modes:
        if mode == "legacy" and (
            args.source_dp_size != 1
            or args.target_dp_size != 1
            or args.source_tp_size != args.target_tp_size
            or args.source_pp_size != args.target_pp_size
            or args.source_dp_size != args.target_dp_size
            or args.source_ep_size != args.target_ep_size
        ):
            skipped.append(
                {
                    "mode": "legacy",
                    "reason": (
                        "legacy remote-instance reuse requires homogeneous "
                        "TP/PP/DP/EP; "
                        f"source_tp_size={args.source_tp_size}, "
                        f"target_tp_size={args.target_tp_size}, "
                        f"source_pp_size={args.source_pp_size}, "
                        f"target_pp_size={args.target_pp_size}, "
                        f"source_dp_size={args.source_dp_size}, "
                        f"target_dp_size={args.target_dp_size}, "
                        f"source_ep_size={args.source_ep_size}, "
                        f"target_ep_size={args.target_ep_size}"
                    ),
                }
            )
        else:
            executed.append(mode)
    return executed, skipped


def _ordered_modes(modes) -> list[str]:
    modes = list(modes)
    return (["cold"] if "cold" in modes else []) + [
        mode for mode in modes if mode != "cold"
    ]


def _execution_schedule(modes, iterations: int) -> list[list[str]]:
    ordered = _ordered_modes(modes)
    if not ordered:
        return []
    if iterations % len(ordered):
        raise ValueError(
            "iterations must form complete rotation blocks for the "
            f"{len(ordered)} executed modes"
        )
    schedule = []
    for iteration in range(iterations):
        offset = iteration % len(ordered)
        schedule.append(ordered[offset:] + ordered[:offset])
    return schedule


def _target_expected_response(
    mode: str, cold_target_baseline: dict[str, Any] | None
) -> dict[str, Any] | None:
    if mode == "cold":
        return None
    if cold_target_baseline is None:
        raise RuntimeError(f"{mode} requires a cold target baseline")
    return cold_target_baseline


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    result_path = output_dir / "benchmark-result.json"
    recorder = ResponseRecorder(responses_path)
    expected_target_ranks = tuple(
        range(args.target_tp_size * args.target_pp_size * args.target_dp_size)
    )
    source: ServerProcess | None = None
    result: dict[str, Any] | None = None
    try:
        source = _start_server(
            args,
            gpus=args.source_gpus,
            port=args.source_port,
            load_mode="source",
            log_path=output_dir / "source.log",
        )
        source_ready_s = _wait_ready(source, args.source_port, args.timeout_s)
        baseline = _collect_source_baseline(args, recorder, source)
        source_baseline = baseline["deterministic_response"]
        executed_modes, skipped_modes = _eligible_modes(args)
        executed_modes = _ordered_modes(executed_modes)
        execution_schedule = _execution_schedule(executed_modes, args.iterations)
        records: dict[str, list[dict[str, Any]]] = {mode: [] for mode in executed_modes}
        warmup_records: dict[str, dict[str, Any]] = {}
        cold_target_baselines = []
        cold_target_baseline = None
        for mode in executed_modes:
            warmup_record = _run_target(
                args,
                source_server=source,
                mode=mode,
                iteration=-1,
                output_dir=output_dir,
                source_baseline=source_baseline,
                source_baseline_p95_s=baseline["latency_p95_s"],
                target_expected=_target_expected_response(mode, cold_target_baseline),
                recorder=recorder,
                expected_target_ranks=expected_target_ranks,
            )
            warmup_records[mode] = warmup_record
            if mode == "cold":
                cold_target_baseline = warmup_record["target_deterministic_response"]

        for iteration, iteration_modes in enumerate(execution_schedule):
            for mode in iteration_modes:
                record = _run_target(
                    args,
                    source_server=source,
                    mode=mode,
                    iteration=iteration,
                    output_dir=output_dir,
                    source_baseline=source_baseline,
                    source_baseline_p95_s=baseline["latency_p95_s"],
                    target_expected=(
                        cold_target_baseline
                        if mode == "cold"
                        else _target_expected_response(mode, cold_target_baseline)
                    ),
                    recorder=recorder,
                    expected_target_ranks=expected_target_ranks,
                )
                records[mode].append(record)
                if mode == "cold":
                    cold_target_baseline = record["target_deterministic_response"]
                    cold_target_baselines.append(cold_target_baseline)

        by_mode = {
            mode: _mode_summary(mode_records) for mode, mode_records in records.items()
        }
        comparisons = {
            f"{mode}_vs_cold": _reuse_comparison(
                by_mode["cold"],
                by_mode[mode],
                max_reuse_to_cold_ratio=args.max_reuse_to_cold_ratio,
            )
            for mode in REUSE_MODES
            if mode in by_mode
        }
        threshold_results = [
            comparison["passes_significant_improvement_threshold"]
            for comparison in comparisons.values()
        ]
        reuse_log_results = [
            record["reuse_log_contract"]["passed"]
            for mode, mode_records in records.items()
            if mode in REUSE_MODES
            for record in mode_records
        ]
        result = {
            "schema_version": 1,
            "model": args.model,
            "topology": {
                "source_tp_size": args.source_tp_size,
                "target_tp_size": args.target_tp_size,
                "source_pp_size": args.source_pp_size,
                "target_pp_size": args.target_pp_size,
                "source_dp_size": args.source_dp_size,
                "target_dp_size": args.target_dp_size,
                "source_ep_size": args.source_ep_size,
                "target_ep_size": args.target_ep_size,
                "source_gpus": args.source_gpus,
                "target_gpus": args.target_gpus,
            },
            "modes_requested": list(args.modes),
            "modes_executed": executed_modes,
            "warmup_mode_order": executed_modes,
            "mode_execution_schedule": execution_schedule,
            "skipped_modes": skipped_modes,
            "iterations": args.iterations,
            "page_cache_dropped_before_each_target": args.drop_page_cache,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "measurement_boundary": "target process spawn to health_generate ready",
            "deterministic_inference": {
                "request": _inference_request(args),
                "comparison": (
                    "source responses are compared with the source baseline; "
                    "reuse targets are compared with the cold target at the "
                    "same target topology using exact text, token IDs, token "
                    "counts, and finish reason plus tolerance-bounded token "
                    "logprobs"
                ),
                "logprob_atol": args.logprob_atol,
                "logprob_rtol": args.logprob_rtol,
                "source_baseline_warmup_response_sequence": baseline[
                    "warmup_response_sequence"
                ],
                "source_baseline_response_sequences": baseline["response_sequences"],
                "source_baseline": source_baseline,
                "cold_target_baselines": cold_target_baselines,
            },
            "source": {
                "spawn_to_ready_s": source_ready_s,
                "server_identity": {
                    "root_pid": source.process.pid,
                    "port": source.port,
                    "ready": source.ready_identity,
                },
                "baseline_sample_count": baseline["sample_count"],
                "baseline_generation_latency_p50_s": baseline["latency_p50_s"],
                "baseline_generation_latency_p95_s": baseline["latency_p95_s"],
                "max_probe_p95_ratio_to_baseline": (args.max_source_probe_p95_ratio),
                "min_probe_samples_per_target_start": args.min_source_probe_samples,
                "log": str(source.log_path),
            },
            "records": records,
            "warmup_records": warmup_records,
            "summary": {
                "by_mode": by_mode,
                "comparisons": comparisons,
                "significant_improvement_threshold": {
                    "metric": "spawn_to_ready_p50_s",
                    "condition": "reuse <= cold * max_reuse_to_cold_ratio",
                    "max_reuse_to_cold_ratio": args.max_reuse_to_cold_ratio,
                },
                "all_executed_reuse_modes_pass_threshold": (
                    all(threshold_results) if threshold_results else None
                ),
                "strict_response_consistency_passed": True,
                "source_serving_continuity_passed": True,
                "strict_reuse_log_contract_passed": (
                    all(reuse_log_results) if reuse_log_results else None
                ),
            },
            "artifacts": {
                "result_json": str(result_path),
                "responses_jsonl": str(responses_path),
            },
        }
    finally:
        if source is not None:
            _stop_server(source)
        recorder.close()

    if result is None:
        raise RuntimeError("benchmark exited without producing a result")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _benchmark_exit_code(
    result: dict[str, Any],
    *,
    report_only: bool,
) -> int:
    if report_only:
        return 0
    by_mode = result.get("summary", {}).get("by_mode", {})
    measured_counts = [
        summary.get("iterations", 0)
        for mode, summary in by_mode.items()
        if mode in result.get("modes_executed", ())
    ]
    if not measured_counts or min(measured_counts) < MIN_MEASURED_ITERATIONS:
        return 2
    passed = result["summary"]["all_executed_reuse_modes_pass_threshold"]
    reuse_requested = bool(set(result.get("modes_requested", ())) & set(REUSE_MODES))
    return 2 if passed is False or (reuse_requested and passed is None) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cold checkpoint startup, homogeneous remote reuse, and "
            "runtime-manifest heterogeneous reuse."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python"))
    parser.add_argument("--source-gpus", type=_parse_gpus, required=True)
    parser.add_argument("--target-gpus", type=_parse_gpus, required=True)
    parser.add_argument("--source-tp-size", type=int, required=True)
    parser.add_argument("--target-tp-size", type=int, required=True)
    parser.add_argument("--source-pp-size", type=int, default=1)
    parser.add_argument("--target-pp-size", type=int, default=1)
    parser.add_argument("--source-dp-size", type=int, default=1)
    parser.add_argument("--target-dp-size", type=int, default=1)
    parser.add_argument("--source-ep-size", type=int, default=1)
    parser.add_argument("--target-ep-size", type=int, default=1)
    parser.add_argument("--modes", type=_parse_modes, default=ALL_MODES)
    parser.add_argument("--source-port", type=int, default=31000)
    parser.add_argument("--target-port", type=int, default=32000)
    parser.add_argument("--bootstrap-port", type=int, default=31999)
    parser.add_argument("--protocol", default="rdma")
    parser.add_argument("--transport-device", default="")
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--mm-attention-backend", default="")
    parser.add_argument("--sampling-backend", default="")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--disable-shared-experts-fusion", action="store_true")
    parser.add_argument("--moe-runner-backend", default="")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--timeout-s", type=float, default=1200)
    parser.add_argument("--request-timeout-s", type=float, default=120)
    parser.add_argument("--probe-interval-s", type=float, default=0.2)
    parser.add_argument("--inter-run-delay-s", type=float, default=2)
    parser.add_argument(
        "--prompt",
        default="A deterministic topology test 7319: The next integer after 41 is",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--source-baseline-samples", type=int, default=5)
    parser.add_argument("--min-source-probe-samples", type=int, default=5)
    parser.add_argument("--logprob-atol", type=float, default=1e-4)
    parser.add_argument("--logprob-rtol", type=float, default=1e-4)
    parser.add_argument(
        "--max-source-probe-p95-ratio",
        type=float,
        default=3.0,
        help="Fail when source probe p95 exceeds this multiple of baseline latency.",
    )
    parser.add_argument(
        "--max-reuse-to-cold-ratio",
        type=float,
        default=0.8,
        help=(
            "Significant-improvement gate applied to spawn-to-ready p50 "
            "(default: reuse <= cold * 0.8)."
        ),
    )
    parser.add_argument("--drop-page-cache", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write measurements without failing the process on the speed gate.",
    )
    parser.add_argument("--output-dir", default="remote-instance-service-benchmark")
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if not args.report_only and args.iterations < MIN_MEASURED_ITERATIONS:
        parser.error(
            f"iterations must be at least {MIN_MEASURED_ITERATIONS} "
            "unless --report-only is set"
        )
    if (
        min(
            args.source_tp_size,
            args.target_tp_size,
            args.source_pp_size,
            args.target_pp_size,
            args.source_dp_size,
            args.target_dp_size,
            args.source_ep_size,
            args.target_ep_size,
        )
        <= 0
    ):
        parser.error("source/target TP, PP, DP, and EP sizes must be positive")
    source_world_size = args.source_tp_size * args.source_pp_size * args.source_dp_size
    target_world_size = args.target_tp_size * args.target_pp_size * args.target_dp_size
    if len(args.source_gpus.split(",")) != source_world_size:
        parser.error(
            "source-gpus count must equal source-tp-size * source-pp-size * "
            "source-dp-size"
        )
    if len(args.target_gpus.split(",")) != target_world_size:
        parser.error(
            "target-gpus count must equal target-tp-size * target-pp-size * "
            "target-dp-size"
        )
    if set(args.source_gpus.split(",")) & set(args.target_gpus.split(",")):
        parser.error("source and target GPU sets must not overlap")
    if len({args.source_port, args.target_port, args.bootstrap_port}) != 3:
        parser.error("source-port, target-port, and bootstrap-port must be distinct")
    if any(mode in args.modes for mode in REUSE_MODES) and "cold" not in args.modes:
        parser.error("reuse modes require cold mode for speedup comparison")
    if "cold" in args.modes and not args.drop_page_cache:
        parser.error("cold mode requires --drop-page-cache")
    if not 0 < args.mem_fraction_static < 1:
        parser.error("mem-fraction-static must be between 0 and 1")
    if args.timeout_s <= 0 or args.request_timeout_s <= 0:
        parser.error("timeout-s and request-timeout-s must be positive")
    if args.probe_interval_s <= 0 or args.inter_run_delay_s < 0:
        parser.error("probe interval must be positive and inter-run delay nonnegative")
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    if args.source_baseline_samples < MIN_PERCENTILE_SAMPLES:
        parser.error(
            f"source-baseline-samples must be at least {MIN_PERCENTILE_SAMPLES}"
        )
    if args.min_source_probe_samples < MIN_PERCENTILE_SAMPLES:
        parser.error(
            f"min-source-probe-samples must be at least {MIN_PERCENTILE_SAMPLES}"
        )
    if args.logprob_atol < 0 or args.logprob_rtol < 0:
        parser.error("logprob tolerances must be nonnegative")
    if args.max_source_probe_p95_ratio <= 0:
        parser.error("max-source-probe-p95-ratio must be positive")
    if not 0 < args.max_reuse_to_cold_ratio <= 1:
        parser.error("max-reuse-to-cold-ratio must be in (0, 1]")
    eligible_modes, _ = _eligible_modes(args)
    try:
        _execution_schedule(eligible_modes, args.iterations)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    benchmark_result = run(parsed_args)
    print("REMOTE_INSTANCE_SERVICE_BENCHMARK_JSON=" + json.dumps(benchmark_result))
    raise SystemExit(
        _benchmark_exit_code(
            benchmark_result,
            report_only=parsed_args.report_only,
        )
    )
