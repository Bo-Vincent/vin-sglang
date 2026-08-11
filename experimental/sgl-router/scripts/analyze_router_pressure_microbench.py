#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze Router-only Criterion pressure benchmarks and enforce gates."""

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence


class SampleStats(NamedTuple):
    samples: int
    p50_ns: float
    p95_ns: float
    p99_ns: float
    throughput_per_s: float


class AggregateStats(NamedTuple):
    p50_ns: float
    p95_ns: float
    p95_rsd: float
    throughput_per_s: float


class AllocationStats(NamedTuple):
    allocations: float
    bytes: float


class ComparisonStats(NamedTuple):
    p95_ratio: float
    p50_ratio: float


class PerfStats(NamedTuple):
    task_clock_ms: float
    cpu_utilized: float
    cycles: float
    instructions: float
    cache_misses: float
    context_switches: float
    cpu_migrations: float


class AcceptanceCheck(NamedTuple):
    name: str
    passed: bool
    detail: str


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def normalize_scenario(group: str, phase: str, identifier: str) -> str:
    prefix = "router_pressure_"
    if not group.startswith(prefix):
        raise ValueError(f"unexpected Criterion group {group!r}")
    family = group[len(prefix) :]
    if identifier.isdigit():
        identifier = f"endpoints={identifier}"
    return f"{family}/{phase}/{identifier}"


def load_run(root: Path) -> Dict[str, SampleStats]:
    criterion = root / "criterion"
    if not criterion.is_dir():
        raise ValueError(f"missing Criterion directory: {criterion}")
    rows: Dict[str, SampleStats] = {}
    for sample_path in sorted(criterion.rglob("new/sample.json")):
        parts = sample_path.relative_to(criterion).parts
        if len(parts) < 5:
            raise ValueError(f"unexpected sample path: {sample_path}")
        group, phase = parts[0], parts[1]
        identifier = "/".join(parts[2:-2])
        scenario = normalize_scenario(group, phase, identifier)
        payload = json.loads(sample_path.read_text())
        iterations = payload.get("iters")
        times = payload.get("times")
        if not isinstance(iterations, list) or not isinstance(times, list):
            raise ValueError(f"sample lacks iters/times arrays: {sample_path}")
        if not iterations or len(iterations) != len(times):
            raise ValueError(f"sample has inconsistent arrays: {sample_path}")
        per_operation = []
        for iteration, elapsed in zip(iterations, times):
            if iteration <= 0 or elapsed < 0:
                raise ValueError(f"sample has invalid timing values: {sample_path}")
            per_operation.append(float(elapsed) / float(iteration))
        p50 = percentile(per_operation, 0.50)
        rows[scenario] = SampleStats(
            samples=len(per_operation),
            p50_ns=p50,
            p95_ns=percentile(per_operation, 0.95),
            p99_ns=percentile(per_operation, 0.99),
            throughput_per_s=1_000_000_000.0 / p50 if p50 else math.inf,
        )
    if not rows:
        raise ValueError(f"no Router pressure samples below {criterion}")
    return rows


def load_allocations(path: Path) -> Dict[str, AllocationStats]:
    rows: Dict[str, AllocationStats] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            scenario, allocations, allocated_bytes = raw.rsplit(",", 2)
            rows[scenario] = AllocationStats(float(allocations), float(allocated_bytes))
        except ValueError as error:
            raise ValueError(f"invalid allocation row {path}:{line_number}: {raw}") from error
    if not rows:
        raise ValueError(f"no allocation rows in {path}")
    return rows


def load_perf(path: Path) -> PerfStats:
    counters: Dict[str, float] = {}
    cpu_utilized = math.nan
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.split(",")
        if len(fields) < 3:
            raise ValueError(f"invalid perf row in {path}: {raw}")
        try:
            value = float(fields[0])
        except ValueError as error:
            raise ValueError(f"invalid perf value in {path}: {raw}") from error
        event = fields[2]
        counters[event] = value
        if event == "task-clock" and len(fields) >= 6:
            cpu_utilized = float(fields[5])
    required = {
        "task-clock",
        "cycles",
        "instructions",
        "cache-misses",
        "context-switches",
        "cpu-migrations",
    }
    missing = required - set(counters)
    if missing or not math.isfinite(cpu_utilized):
        raise ValueError(f"incomplete perf data in {path}: missing={sorted(missing)}")
    return PerfStats(
        task_clock_ms=counters["task-clock"],
        cpu_utilized=cpu_utilized,
        cycles=counters["cycles"],
        instructions=counters["instructions"],
        cache_misses=counters["cache-misses"],
        context_switches=counters["context-switches"],
        cpu_migrations=counters["cpu-migrations"],
    )


def aggregate_runs(runs: Sequence[Dict[str, SampleStats]]) -> Dict[str, AggregateStats]:
    if not runs:
        raise ValueError("at least one run is required")
    expected = set(runs[0])
    for index, run in enumerate(runs[1:], start=2):
        if set(run) != expected:
            missing = sorted(expected - set(run))
            extra = sorted(set(run) - expected)
            raise ValueError(f"run {index} scenario mismatch: missing={missing}, extra={extra}")
    aggregate = {}
    for scenario in sorted(expected):
        p50_values = [run[scenario].p50_ns for run in runs]
        p95_values = [run[scenario].p95_ns for run in runs]
        throughput_values = [run[scenario].throughput_per_s for run in runs]
        p95_mean = statistics.fmean(p95_values)
        aggregate[scenario] = AggregateStats(
            p50_ns=statistics.median(p50_values),
            p95_ns=statistics.median(p95_values),
            p95_rsd=(statistics.pstdev(p95_values) / p95_mean if p95_mean else 0.0),
            throughput_per_s=statistics.median(throughput_values),
        )
    return aggregate


def aggregate_allocations(paths: Iterable[Path]) -> Dict[str, AllocationStats]:
    runs = [load_allocations(path) for path in paths]
    if not runs:
        return {}
    expected = set(runs[0])
    for index, run in enumerate(runs[1:], start=2):
        if set(run) != expected:
            raise ValueError(f"allocation run {index} has a different scenario set")
    return {
        scenario: AllocationStats(
            statistics.median(run[scenario].allocations for run in runs),
            statistics.median(run[scenario].bytes for run in runs),
        )
        for scenario in sorted(expected)
    }


def aggregate_perf(paths: Iterable[Path]) -> PerfStats:
    runs = [load_perf(path) for path in paths]
    if not runs:
        return PerfStats(*(math.nan for _ in range(7)))
    return PerfStats(
        *(
            statistics.median(getattr(run, field) for run in runs)
            for field in PerfStats._fields
        )
    )


def compare_variants(
    baseline: Dict[str, AggregateStats], candidate: Dict[str, AggregateStats]
) -> Dict[str, ComparisonStats]:
    missing = set(baseline) - set(candidate)
    if missing:
        raise ValueError(f"candidate is missing scenarios: {sorted(missing)}")
    return {
        scenario: ComparisonStats(
            p95_ratio=candidate[scenario].p95_ns / baseline_row.p95_ns,
            p50_ratio=candidate[scenario].p50_ns / baseline_row.p50_ns,
        )
        for scenario, baseline_row in baseline.items()
    }


def ratio_check(
    name: str,
    rows: Dict[str, AggregateStats],
    numerator: str,
    denominator: str,
    maximum: float,
) -> AcceptanceCheck:
    if numerator not in rows or denominator not in rows:
        return AcceptanceCheck(name, False, "missing required scenario")
    ratio = rows[numerator].p95_ns / rows[denominator].p95_ns
    return AcceptanceCheck(name, ratio <= maximum, f"ratio={ratio:.4f}, limit={maximum:.4f}")


def absolute_check(
    name: str, rows: Dict[str, AggregateStats], scenario: str, maximum_ns: float
) -> AcceptanceCheck:
    if scenario not in rows:
        return AcceptanceCheck(name, False, f"missing required scenario {scenario}")
    actual = rows[scenario].p95_ns
    return AcceptanceCheck(
        name,
        actual <= maximum_ns,
        f"p95_us={actual / 1_000.0:.4f}, limit_us={maximum_ns / 1_000.0:.4f}",
    )


def evaluate_stability(rows: Dict[str, AggregateStats]) -> List[AcceptanceCheck]:
    return [
        AcceptanceCheck(
            f"stability/{scenario}",
            row.p95_rsd <= 0.10,
            (
                f"p95_rsd={row.p95_rsd:.4f}"
                if row.p95_rsd <= 0.10
                else f"p95_rsd={row.p95_rsd:.4f}; confirmation required"
            ),
        )
        for scenario, row in sorted(rows.items())
    ]


def evaluate_perf(name: str, perf: PerfStats) -> AcceptanceCheck:
    passed = 0.90 <= perf.cpu_utilized <= 1.05 and perf.cpu_migrations == 0
    return AcceptanceCheck(
        f"perf/{name}",
        passed,
        (
            f"cpu_utilized={perf.cpu_utilized:.4f}, "
            f"cpu_migrations={perf.cpu_migrations:.0f}, "
            f"task_clock_ms={perf.task_clock_ms:.2f}"
        ),
    )


def evaluate_acceptance(
    step2: Dict[str, AggregateStats],
    step3_common: Dict[str, AggregateStats],
    step3_rich: Dict[str, AggregateStats],
) -> List[AcceptanceCheck]:
    checks: List[AcceptanceCheck] = []
    for label, baseline, candidate in [
        ("step3_common_vs_step2", step2, step3_common),
        ("step3_rich_vs_common", step3_common, step3_rich),
    ]:
        comparison = compare_variants(baseline, candidate)
        for scenario, row in sorted(comparison.items()):
            checks.append(
                AcceptanceCheck(
                    f"{label}/{scenario}",
                    row.p95_ratio <= 1.10,
                    f"p95_ratio={row.p95_ratio:.4f}, limit=1.1000",
                )
            )

    checks.extend(
        [
            absolute_check(
                "absolute/snapshot_256_dp1",
                step3_rich,
                "snapshot/capture/endpoints=256,dp=1",
                100_000,
            ),
            absolute_check(
                "absolute/prefill_p2_request_256",
                step3_rich,
                "prefill_p2/request_path/endpoints=256",
                150_000,
            ),
            absolute_check(
                "absolute/session_request_256",
                step3_rich,
                "session/request_path/endpoints=256",
                150_000,
            ),
            absolute_check(
                "absolute/decode_request_256",
                step3_rich,
                "decode/request_path/endpoints=256",
                150_000,
            ),
            absolute_check(
                "absolute/cache_request_256_top32",
                step3_rich,
                "cache/request_path/endpoints=256,top_k=32",
                300_000,
            ),
        ]
    )

    for family in ("prefill_p2", "session", "decode"):
        checks.append(
            ratio_check(
                f"scaling/{family}_policy_64_to_256",
                step3_rich,
                f"{family}/policy_only/endpoints=256",
                f"{family}/policy_only/endpoints=64",
                1.5,
            )
        )
    checks.append(
        ratio_check(
            "scaling/snapshot_dp1_64_to_256",
            step3_rich,
            "snapshot/capture/endpoints=256,dp=1",
            "snapshot/capture/endpoints=64,dp=1",
            6.0,
        )
    )
    for top_k in (4, 16, 32):
        checks.append(
            ratio_check(
                f"scaling/cache_policy_top{top_k}_64_to_256",
                step3_rich,
                f"cache/policy_only/endpoints=256,top_k={top_k}",
                f"cache/policy_only/endpoints=64,top_k={top_k}",
                6.0,
            )
        )
    return checks


def stats_dict(rows: Dict[str, NamedTuple]) -> Dict[str, dict]:
    return {scenario: row._asdict() for scenario, row in sorted(rows.items())}


def write_report(
    output_dir: Path,
    variants: Dict[str, Dict[str, AggregateStats]],
    allocations: Dict[str, Dict[str, AllocationStats]],
    perf: Dict[str, PerfStats],
    checks: Sequence[AcceptanceCheck],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    passed = all(check.passed for check in checks)
    payload = {
        "passed": passed,
        "variants": {name: stats_dict(rows) for name, rows in variants.items()},
        "allocations": {name: stats_dict(rows) for name, rows in allocations.items()},
        "perf": {name: row._asdict() for name, row in perf.items()},
        "checks": [check._asdict() for check in checks],
    }
    (output_dir / "analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Router-only Pressure Microbenchmark Analysis",
        "",
        f"Overall: **{'PASS' if passed else 'FAIL'}**",
        "",
        "## Acceptance checks",
        "",
        "| Check | Result | Detail |",
        "| --- | --- | --- |",
    ]
    for check in checks:
        lines.append(f"| `{check.name}` | {'PASS' if check.passed else 'FAIL'} | {check.detail} |")
    lines.extend(["", "## Step 3 rich-pressure latency", "", "| Scenario | p50 us | p95 us | RSD | ops/s |", "| --- | ---: | ---: | ---: | ---: |"])
    for scenario, row in sorted(variants["step3_rich"].items()):
        lines.append(
            f"| `{scenario}` | {row.p50_ns / 1_000:.4f} | {row.p95_ns / 1_000:.4f} | "
            f"{row.p95_rsd:.2%} | {row.throughput_per_s:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Process CPU validity",
            "",
            "| Variant | Task clock ms | CPUs utilized | Migrations | Cycles | Instructions | Cache misses |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in perf.items():
        lines.append(
            f"| `{name}` | {row.task_clock_ms:.2f} | {row.cpu_utilized:.4f} | "
            f"{row.cpu_migrations:.0f} | {row.cycles:.0f} | {row.instructions:.0f} | "
            f"{row.cache_misses:.0f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("step2", "step3-common", "step3-rich"):
        parser.add_argument(f"--{name}-run", action="append", type=Path, required=True)
        parser.add_argument(f"--{name}-alloc", action="append", type=Path, default=[])
        parser.add_argument(f"--{name}-perf", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_paths = {
        "step2": args.step2_run,
        "step3_common": args.step3_common_run,
        "step3_rich": args.step3_rich_run,
    }
    alloc_paths = {
        "step2": args.step2_alloc,
        "step3_common": args.step3_common_alloc,
        "step3_rich": args.step3_rich_alloc,
    }
    perf_paths = {
        "step2": args.step2_perf,
        "step3_common": args.step3_common_perf,
        "step3_rich": args.step3_rich_perf,
    }
    variants = {
        name: aggregate_runs([load_run(path) for path in paths])
        for name, paths in run_paths.items()
    }
    allocations = {
        name: aggregate_allocations(paths) for name, paths in alloc_paths.items()
    }
    perf = {name: aggregate_perf(paths) for name, paths in perf_paths.items()}
    checks = []
    for rows in variants.values():
        checks.extend(evaluate_stability(rows))
    checks.extend(
        evaluate_acceptance(
            variants["step2"], variants["step3_common"], variants["step3_rich"]
        )
    )
    for name, paths in perf_paths.items():
        checks.extend(
            evaluate_perf(f"{name}/run-{index}", load_perf(path))
            for index, path in enumerate(paths, start=1)
        )
    write_report(args.output_dir, variants, allocations, perf, checks)
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
