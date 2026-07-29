"""Measure live weight materialization while the source keeps serving."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error


def _generation_request(prompt: str) -> dict[str, Any]:
    return {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 8,
            "sampling_seed": 0,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }


def _stable_response(response: dict[str, Any]) -> dict[str, Any]:
    meta = response["meta_info"]
    return {
        "text": response["text"],
        "output_ids": response["output_ids"],
        "completion_tokens": meta["completion_tokens"],
        "finish_reason": meta["finish_reason"],
        "output_token_logprobs": meta["output_token_logprobs"],
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * percentile) - 1))
    return ordered[index]


def _probe_overlaps_window(
    probe: dict[str, Any],
    window_start_s: float,
    window_end_s: float,
) -> bool:
    completed_at_s = probe.get("completed_at_s")
    return (
        probe["started_at_s"] <= window_end_s
        and (completed_at_s is None or completed_at_s >= window_start_s)
    )


def _collect_warmed_baseline(
    url: str,
    generation: dict[str, Any],
    *,
    timeout: float,
    warmup_samples: int,
    sample_count: int,
) -> tuple[dict[str, Any], list[float]]:
    for _ in range(warmup_samples):
        _post_json(url, generation, timeout)

    baseline = None
    latencies = []
    for _ in range(sample_count):
        started = time.perf_counter()
        response = _stable_response(_post_json(url, generation, timeout))
        latencies.append(time.perf_counter() - started)
        if baseline is None:
            baseline = response
        elif response != baseline:
            raise RuntimeError("warmed baseline responses are inconsistent")
    assert baseline is not None
    return baseline, latencies


def _probe_latency_limit(
    *,
    warmed_baseline_p95_s: float,
    max_ratio: float,
    absolute_limit_s: float,
) -> float:
    return min(warmed_baseline_p95_s * max_ratio, absolute_limit_s)


def _source_serving_continuity_passed(
    *,
    in_window: list[dict[str, Any]],
    min_samples: int,
    materialization_published: bool,
    post_consistent: bool,
    probe_worker_stopped: bool,
    latency_limit_s: float,
) -> bool:
    return (
        probe_worker_stopped
        and len(in_window) >= min_samples
        and all(probe.get("completed_at_s") is not None for probe in in_window)
        and all(probe.get("success") is True for probe in in_window)
        and all(probe.get("consistent") is True for probe in in_window)
        and materialization_published
        and all(
            probe.get("latency_s") is not None
            and probe["latency_s"] <= latency_limit_s
            for probe in in_window
        )
        and post_consistent
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_url = args.source_url.rstrip("/")
    generation = _generation_request(args.prompt)
    baseline, baseline_latencies_s = _collect_warmed_baseline(
        f"{source_url}/generate",
        generation,
        timeout=args.timeout,
        warmup_samples=args.baseline_warmup_samples,
        sample_count=args.baseline_samples,
    )
    baseline_latency_s = statistics.median(baseline_latencies_s)
    baseline_latency_p95_s = _percentile(baseline_latencies_s, 0.95)

    probes: list[dict[str, Any]] = []
    probes_lock = threading.Lock()
    stop = threading.Event()

    def probe_source() -> None:
        sequence = 0
        while not stop.is_set():
            started = time.perf_counter()
            record: dict[str, Any] = {
                "sequence": sequence,
                "started_at_s": time.time(),
                "completed_at_s": None,
                "success": None,
                "consistent": None,
            }
            with probes_lock:
                probes.append(record)
            completed: dict[str, Any] = {}
            try:
                response = _post_json(
                    f"{source_url}/generate",
                    generation,
                    args.timeout,
                )
                completed["success"] = True
                completed["consistent"] = _stable_response(response) == baseline
            except Exception as error:
                completed["success"] = False
                completed["consistent"] = False
                completed["error"] = str(error)
            completed["latency_s"] = time.perf_counter() - started
            completed["completed_at_s"] = time.time()
            with probes_lock:
                record.update(completed)
            sequence += 1
            stop.wait(args.probe_interval)

    thread = threading.Thread(target=probe_source, daemon=True)
    thread.start()
    time.sleep(args.probe_interval)

    store_setup = {
        "local_hostname": args.local_hostname,
        "metadata_server": args.metadata_server,
        "global_segment_size": args.store_buffer_bytes,
        "local_buffer_size": args.store_buffer_bytes,
        "protocol": args.protocol,
        "master_server_addr": args.master_server,
    }
    if args.rdma_devices:
        store_setup["rdma_devices"] = args.rdma_devices
    request = {
        "materialization_id": args.materialization_id,
        "storage_options": {
            "catalog_path": args.catalog_path,
            "destination": {
                "provider": "mooncake-store",
                "storage_id": args.storage_id,
                "object_prefix": args.storage_id,
            },
            "endpoint": args.local_hostname,
            "mooncake_store": {
                "setup": store_setup,
                "namespace": args.namespace,
                "key_prefix": args.key_prefix,
                "max_range_bytes": args.max_range_bytes,
                "max_ranges_per_request": args.max_ranges_per_request,
                "max_region_segments": args.max_region_segments,
                "max_total_operations": args.max_total_operations,
                "source_pre_registered": False,
            },
        },
    }

    materialize_started = time.perf_counter()
    materialize_started_at = time.time()
    try:
        materialize_response = _post_json(
            f"{source_url}/materialize_weights",
            request,
            args.materialize_timeout,
        )
    finally:
        materialize_finished_at = time.time()
        materialize_e2e_s = time.perf_counter() - materialize_started
        time.sleep(args.probe_interval)
        stop.set()
        thread.join(timeout=args.timeout)
    probe_worker_stopped = not thread.is_alive()

    post = None
    post_error = None
    if probe_worker_stopped:
        try:
            post_raw = _post_json(f"{source_url}/generate", generation, args.timeout)
            post = _stable_response(post_raw)
        except Exception as error:
            post_error = str(error)
    else:
        post_error = "source probe worker did not stop before its timeout"
    with probes_lock:
        probe_records = [dict(probe) for probe in probes]
    in_window = [
        probe
        for probe in probe_records
        if _probe_overlaps_window(
            probe,
            materialize_started_at,
            materialize_finished_at,
        )
    ]
    latencies = [
        probe["latency_s"]
        for probe in in_window
        if probe.get("latency_s") is not None
    ]
    successful = [probe for probe in in_window if probe.get("success") is True]
    materialization_published = (
        isinstance(materialize_response, dict)
        and materialize_response.get("session_state") == "published"
        and materialize_response.get("completion_unknown") is False
        and isinstance(materialize_response.get("ref"), dict)
    )
    probe_latency_limit_s = _probe_latency_limit(
        warmed_baseline_p95_s=baseline_latency_p95_s,
        max_ratio=args.max_probe_latency_ratio,
        absolute_limit_s=args.max_probe_latency_seconds,
    )
    post_consistent = post == baseline
    source_serving_continuity_passed = _source_serving_continuity_passed(
        in_window=in_window,
        min_samples=args.min_source_probe_samples,
        materialization_published=materialization_published,
        post_consistent=post_consistent,
        probe_worker_stopped=probe_worker_stopped,
        latency_limit_s=probe_latency_limit_s,
    )
    result = {
        "source_url": source_url,
        "model_id": args.model_id,
        "revision": args.revision,
        "materialization_id": args.materialization_id,
        "materialize_e2e_s": materialize_e2e_s,
        "baseline_latency_s": baseline_latency_s,
        "baseline_latency_p95_s": baseline_latency_p95_s,
        "baseline_latencies_s": baseline_latencies_s,
        "baseline_warmup_samples": args.baseline_warmup_samples,
        "materialize_response": materialize_response,
        "materialization_published": materialization_published,
        "baseline": baseline,
        "post_materialization": post,
        "post_materialization_error": post_error,
        "post_materialization_consistent": post_consistent,
        "source_during_materialization": {
            "requests": len(in_window),
            "successes": len(successful),
            "consistent": sum(bool(probe["consistent"]) for probe in in_window),
            "failures": len(in_window) - len(successful),
            "latency_p50_s": statistics.median(latencies) if latencies else None,
            "latency_p95_s": _percentile(latencies, 0.95) if latencies else None,
            "latency_max_s": max(latencies) if latencies else None,
            "latency_limit_s": probe_latency_limit_s,
            "probe_worker_stopped": probe_worker_stopped,
        },
        "summary": {
            "source_serving_continuity_passed": (source_serving_continuity_passed),
            "gate": (
                "at least the configured number of source requests overlap "
                "materialization; "
                "all overlapping requests succeed and match the baseline; "
                "the materialization is published without unknown completion; "
                "the probe worker has no unfinished request; source probe max "
                "latency stays within the warmed ratio and absolute cap; "
                "the post-materialization response matches the baseline"
            ),
        },
        "probes": probe_records,
        "request": request,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _continuity_exit_code(
    result: dict[str, Any],
    *,
    report_only: bool,
) -> int:
    if report_only:
        return 0
    passed = result["summary"]["source_serving_continuity_passed"]
    return 0 if passed else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", default="default")
    parser.add_argument("--catalog-path", required=True)
    parser.add_argument("--storage-id", required=True)
    parser.add_argument("--materialization-id", required=True)
    parser.add_argument("--metadata-server", required=True)
    parser.add_argument("--master-server", required=True)
    parser.add_argument("--local-hostname", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--namespace", default="default")
    parser.add_argument("--key-prefix", default="weights")
    parser.add_argument("--protocol", default="rdma")
    parser.add_argument("--rdma-devices", default="")
    parser.add_argument("--store-buffer-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--max-range-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--max-ranges-per-request", type=int, default=1024)
    parser.add_argument("--max-region-segments", type=int, default=1_000_000)
    parser.add_argument("--max-total-operations", type=int, default=10_000_000)
    parser.add_argument("--baseline-warmup-samples", type=int, default=2)
    parser.add_argument("--baseline-samples", type=int, default=5)
    parser.add_argument("--probe-interval", type=float, default=0.05)
    parser.add_argument("--min-source-probe-samples", type=int, default=5)
    parser.add_argument("--max-probe-latency-ratio", type=float, default=10.0)
    parser.add_argument("--max-probe-latency-seconds", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--materialize-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write measurements without failing the process on continuity.",
    )
    parser.add_argument(
        "--prompt",
        default="A deterministic Store reuse test: The next integer after 41 is",
    )
    args = parser.parse_args()
    if args.store_buffer_bytes <= 0:
        parser.error("--store-buffer-bytes must be positive")
    if args.probe_interval <= 0:
        parser.error("--probe-interval must be positive")
    if args.baseline_warmup_samples < 0:
        parser.error("--baseline-warmup-samples must be non-negative")
    if args.baseline_samples <= 0:
        parser.error("--baseline-samples must be positive")
    if args.min_source_probe_samples <= 0:
        parser.error("--min-source-probe-samples must be positive")
    if args.max_probe_latency_ratio <= 0:
        parser.error("--max-probe-latency-ratio must be positive")
    if args.max_probe_latency_seconds <= 0:
        parser.error("--max-probe-latency-seconds must be positive")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    output = run(parsed_args)
    print("WEIGHT_SNAPSHOT_STORE_JSON=" + json.dumps(output, sort_keys=True))
    raise SystemExit(
        _continuity_exit_code(
            output,
            report_only=parsed_args.report_only,
        )
    )
