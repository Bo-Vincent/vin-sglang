# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""汇总真实 HTTP Simulator fleet 的可恢复 Router 实验结果。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


MEASUREMENT_KIND = "simulator_predicted_relative"
MEASUREMENT_NOTICE = (
    "Simulator values are predicted relative results, not GPU measurements. "
    "Use the real GPU E2E report for absolute TTFT, E2E, and throughput."
)
METRIC_PATHS = {
    "throughput_rps": ("throughput_rps",),
    "completion_tps": ("completion_tps",),
    "ttft_mean_ms": ("ttft_ms", "mean"),
    "ttft_p95_ms": ("ttft_ms", "p95"),
    "e2e_mean_ms": ("e2e_ms", "mean"),
    "e2e_p95_ms": ("e2e_ms", "p95"),
    "cache_hit_rate": ("cache", "hit_rate"),
    "worker_cv": ("worker_cv",),
}
NATIVE_CACHE_AUDIT_FIELDS = (
    "cache_candidate_decisions",
    "monitor_decisions",
    "router_local_decisions",
    "zero_snapshot_decisions",
    "actual_cache_metrics",
)


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise RuntimeError(f"missing required result file: {path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON result file: {path}") from error


def number_at(summary: Mapping[str, object], path: tuple[str, ...]) -> float:
    value: object = summary
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise RuntimeError(f"summary is missing metric: {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise RuntimeError(f"summary metric is not finite: {'.'.join(path)}")
    return float(value)


def require_complete_case(root: Path, expected: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    name = expected.get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeError("manifest case has no name")
    directory = root / name
    complete = directory / "COMPLETE"
    if (complete.read_text() if complete.is_file() else "") != "ok\n":
        raise RuntimeError(f"incomplete case: {name}")
    case = read_json(directory / "case.json")
    summary = read_json(directory / "summary.json")
    if not isinstance(case, dict) or not isinstance(summary, dict):
        raise RuntimeError(f"case or summary has invalid object shape: {name}")
    for field in ("endpoint_count", "policy", "workload", "repeat"):
        if case.get(field) != expected.get(field):
            raise RuntimeError(f"manifest mismatch in {name}: {field}")
    return case, summary


def validate_summary(case: Mapping[str, object], summary: Mapping[str, object]) -> None:
    request_count = summary.get("request_count")
    if not isinstance(request_count, int) or request_count <= 0:
        raise RuntimeError("summary request_count must be positive")
    request_errors = summary.get("request_errors")
    if not isinstance(request_errors, int) or request_errors != 0:
        raise RuntimeError("summary request_errors must be exactly zero")
    for path in METRIC_PATHS.values():
        number_at(summary, path)
    reasons = summary.get("policy_reasons")
    if not isinstance(reasons, Mapping):
        raise RuntimeError("summary policy_reasons must be an object")
    if not reasons and case.get("policy") != "cache_aware_zmq":
        raise RuntimeError("summary policy_reasons must not be empty")
    for reason, count in reasons.items():
        if not isinstance(reason, str) or not isinstance(count, (int, float)) or count < 0:
            raise RuntimeError("summary policy_reasons contains an invalid entry")
    if case.get("policy") != "cache_aware":
        return
    audit = summary.get("native_cache_audit")
    if not isinstance(audit, Mapping):
        raise RuntimeError("native Cache-Aware case has no audit")
    for field in NATIVE_CACHE_AUDIT_FIELDS:
        if not isinstance(audit.get(field), int):
            raise RuntimeError(f"native Cache-Aware audit has no integer {field}")
    if audit["cache_candidate_decisions"] <= 0 or audit["monitor_decisions"] <= 0:
        raise RuntimeError("native Cache-Aware audit has no cache/monitor decision")
    if audit["monitor_decisions"] != audit["cache_candidate_decisions"]:
        raise RuntimeError("native Cache-Aware monitor does not cover every cache decision")
    if audit["actual_cache_metrics"] <= 0:
        raise RuntimeError("native Cache-Aware case has no cache metric evidence")
    if audit["router_local_decisions"] != 0 or audit["zero_snapshot_decisions"] != 0:
        raise RuntimeError("native Cache-Aware case fell back from LoadMonitor")


def median_rsd(values: Iterable[float]) -> dict[str, float]:
    sample = tuple(values)
    if not sample:
        raise RuntimeError("metric has no samples")
    mean = statistics.fmean(sample)
    deviation = statistics.pstdev(sample)
    return {
        "median": float(statistics.median(sample)),
        "mean": mean,
        "rsd_percent": 0.0 if mean == 0.0 and deviation == 0.0 else deviation / abs(mean) * 100.0,
    }


def load_result_groups(
    results_dir: Path,
) -> dict[tuple[int, str, str], list[tuple[dict[str, object], dict[str, object]]]]:
    results_dir = results_dir.resolve()
    run_complete = results_dir / "RUN_COMPLETE"
    if (run_complete.read_text() if run_complete.is_file() else "") != "ok\n":
        raise RuntimeError("results RUN_COMPLETE is missing or not ok")
    manifest = read_json(results_dir / "manifest.json")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("contract"), Mapping):
        raise RuntimeError("manifest has no contract")
    contract = manifest["contract"]
    expected_cases = contract.get("cases")
    repeats = contract.get("repeats")
    if not isinstance(expected_cases, list) or not expected_cases:
        raise RuntimeError("manifest has no expected cases")
    if not isinstance(repeats, int) or repeats <= 0:
        raise RuntimeError("manifest repeats must be positive")

    grouped: dict[tuple[int, str, str], list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    for expected in expected_cases:
        if not isinstance(expected, Mapping):
            raise RuntimeError("manifest case must be an object")
        case, summary = require_complete_case(results_dir, expected)
        validate_summary(case, summary)
        endpoint_count = case.get("endpoint_count")
        policy = case.get("policy")
        workload = case.get("workload")
        if not isinstance(endpoint_count, int) or not isinstance(policy, str) or not isinstance(workload, str):
            raise RuntimeError("case has an invalid grouping key")
        grouped[(endpoint_count, policy, workload)].append((case, summary))

    for (endpoint_count, policy, workload), samples in sorted(grouped.items()):
        observed_repeats = {case["repeat"] for case, _ in samples}
        if observed_repeats != set(range(repeats)):
            raise RuntimeError(f"incomplete repeats for {endpoint_count}/{policy}/{workload}")
    return grouped


def analyze_results(
    results_dir: Path, *, confirmation_results_dir: Path | None = None
) -> dict[str, object]:
    primary = load_result_groups(results_dir)
    confirmation = (
        load_result_groups(confirmation_results_dir)
        if confirmation_results_dir is not None
        else {}
    )
    extra_confirmation = set(confirmation) - set(primary)
    if extra_confirmation:
        raise RuntimeError(
            f"confirmation has groups absent from primary results: {sorted(extra_confirmation)}"
        )

    groups = []
    for (endpoint_count, policy, workload), primary_samples in sorted(primary.items()):
        confirmation_samples = confirmation.get((endpoint_count, policy, workload), [])
        samples = primary_samples + confirmation_samples
        metrics = {
            name: median_rsd(number_at(summary, path) for _, summary in samples)
            for name, path in METRIC_PATHS.items()
        }
        reasons: dict[str, float] = defaultdict(float)
        audit_totals: dict[str, int] = defaultdict(int)
        reason_presence = {bool(summary["policy_reasons"]) for _, summary in samples}
        if len(reason_presence) != 1:
            raise RuntimeError(f"inconsistent policy-reason observability for {policy}")
        for _, summary in samples:
            for reason, count in summary["policy_reasons"].items():
                reasons[str(reason)] += float(count)
            audit = summary.get("native_cache_audit")
            if isinstance(audit, Mapping):
                for field in NATIVE_CACHE_AUDIT_FIELDS:
                    audit_totals[field] += int(audit[field])
        group = {
            "endpoint_count": endpoint_count,
            "policy": policy,
            "workload": workload,
            "repeat_count": len(samples),
            "primary_repeat_count": len(primary_samples),
            "confirmation_repeat_count": len(confirmation_samples),
            "metrics": metrics,
            "policy_reasons": dict(sorted(reasons.items())),
            "policy_reason_observability": (
                "emitted" if reason_presence.pop() else "not_emitted_by_policy"
            ),
        }
        if audit_totals:
            group["native_cache_audit"] = dict(audit_totals)
        groups.append(group)

    return {
        "measurement_kind": MEASUREMENT_KIND,
        "measurement_notice": MEASUREMENT_NOTICE,
        "case_count": sum(len(samples) for samples in primary.values()),
        "group_count": len(groups),
        "groups": groups,
    }


def markdown_report(analysis: Mapping[str, object]) -> str:
    lines = [
        "# Router HTTP Simulator Fleet 分析",
        "",
        str(analysis["measurement_notice"]),
        "",
        "| Workers | Policy | Workload | TTFT p95 (ms) | E2E p95 (ms) | TPS | KV hit | Worker CV |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in analysis["groups"]:
        metrics = group["metrics"]
        lines.append(
            "| {endpoint_count} | {policy} | {workload} | {ttft:.3f} | {e2e:.3f} | {tps:.3f} | {hit:.2%} | {cv:.3f} |".format(
                endpoint_count=group["endpoint_count"],
                policy=group["policy"],
                workload=group["workload"],
                ttft=metrics["ttft_p95_ms"]["median"],
                e2e=metrics["e2e_p95_ms"]["median"],
                tps=metrics["completion_tps"]["median"],
                hit=metrics["cache_hit_rate"]["median"],
                cv=metrics["worker_cv"]["median"],
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirmation-results-dir", type=Path)
    args = parser.parse_args(argv)
    analysis = analyze_results(
        args.results_dir, confirmation_results_dir=args.confirmation_results_dir
    )
    output_dir = args.output_dir or args.results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    (output_dir / "analysis.md").write_text(markdown_report(analysis))
    print(json.dumps(analysis, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
