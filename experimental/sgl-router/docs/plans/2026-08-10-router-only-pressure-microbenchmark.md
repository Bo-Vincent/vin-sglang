# Router-only Pressure Microbenchmark Implementation Plan

> **历史计划。** v4 的 `router_pressure` 已改为直接构造 `EngineLoadTable`，不再通过
> 旧 gRPC LoadMonitor 注入负载；本文的变体和指标不能解释为当前 benchmark 合同。

**Goal:** Quantify Step 3 pressure collection/consumption overhead at 8, 64, and 256 endpoints, and fix any measured hot-path regression in the semantic commit that introduced it.

**Architecture:** Add one Criterion bench that exercises the public Router APIs with real LoadMonitor state seeded through the load-reporting gRPC contract. Step 2 captures the existing diagnostic snapshot; Step 3 captures the scheduling snapshot consumed by the Router hot path. Measure snapshot capture, policy-only selection, and snapshot-plus-policy decisions separately; emit allocation counts alongside Criterion samples and collect process CPU counters with `perf stat`. Run the same bench on the Step 2 base, the Step 3 fallback-metric build, and the Step 3 rich-pressure build.

**Tech Stack:** Rust, Criterion, Tokio/Tonic, Linux `taskset` and `perf`, Python JSON analysis.

---

## Acceptance contract

- Endpoint counts: 8, 64, and 256; one DP rank for the main matrix and eight ranks for the snapshot scaling confirmation.
- Policies: Prefill Power-of-Two, Session-Aware plus Admission/Guard, Cache-Aware top-K (4, 16, and 32), and Decode Power-of-Two plus Decode Guard.
- Views: `snapshot-only`, `policy-only`, and `snapshot + policy` for every policy family.
- Comparisons:
  - Step 3 fallback metrics versus Step 2.
  - Step 3 rich pressure metrics versus Step 3 fallback metrics.
- Three independent formal runs. Add confirmation runs only when a primary metric has RSD greater than 10%.
- No panic, missing selection, out-of-domain selection, or malformed result is accepted.
- Core policy p95 regression must be at most 10% for both comparisons.
- At 256 endpoints, snapshot p95 must be at most 100 microseconds; complete P2, Session, and Decode decisions at most 150 microseconds; Cache top-32 at most 300 microseconds.
- Pair policy-only latency from 64 to 256 endpoints must scale by at most 1.5x. Snapshot and fixed-K Cache paths may scale by at most 6x.
- Allocation count/bytes per operation and `perf stat` task-clock, cycles, instructions, and cache misses are diagnostic gates: any superlinear allocation growth or unexplained CPU increase requires root-cause analysis before completion.

### Task 1: Add the benchmark contract

**Files:**
- Modify: `Cargo.toml`
- Create: `benches/router_pressure.rs`

1. Add a failing compile check for the not-yet-registered `router_pressure` bench.
2. Register the bench and a benchmark-only `step3-pressure` feature.
3. Build fixtures through public Router APIs; do not add production-only test hooks.
4. Seed common counters on all revisions and compile rich counters only under `step3-pressure`.
5. Add a counting allocator that is enabled only during a separate allocation pass.
6. Run `cargo bench --bench router_pressure --no-run` and the existing bench compile checks.

### Task 2: Add reproducible result analysis

**Files:**
- Create: `scripts/analyze_router_pressure_microbench.py`
- Create: `tests/scripts/test_router_pressure_microbench.py`

1. Write tests for Criterion sample parsing, p50/p95/p99, throughput, RSD, allocation parsing, comparison ratios, scaling checks, and acceptance failures.
2. Confirm the tests fail before the analyzer exists.
3. Implement the minimum analyzer needed by the tests.
4. Run the focused Python test and `py_compile`.

### Task 3: Run the Step 2 and Step 3 matrix

**Artifacts:**
- Local staging: `.vin_stage/router-pressure-microbench-20260810/`
- Remote source/results: `/root/router-policy-bench/router-pressure-microbench-20260810/`

1. Create isolated Step 2 and Step 3 worktrees without touching protected untracked files.
2. Apply the identical benchmark-only patch to Step 2.
3. Build release benches locally or on the remote host with an explicit Rust `PATH`.
4. Transfer sources with `rsync` only.
5. Pin runs to one physical core on one NUMA node.
6. Run Step 2 common, Step 3 common, and Step 3 rich-pressure three times each under `perf stat`.
7. Preserve Criterion raw samples, allocation records, command lines, environment data, and commit hashes.

### Task 4: Fix measured blockers in history

**Files:** Determined by the failing benchmark.

1. For each failing contract, write a focused regression/performance test that fails on the current semantic commit.
2. Implement the smallest correction.
3. Fix up the correction into the self-owned commit that introduced the issue; do not modify dependency commits.
4. Re-run the complete matrix because code changes invalidate prior performance data.
5. Use `git range-diff` to prove dependency commits remain equivalent.

### Task 5: Final verification and report

**Files:**
- Create: `docs/router-policy-pressure-microbenchmark-report.md`
- Modify: `BENCHMARKS.md`

1. Run final analysis and verify every acceptance check.
2. Run `cargo fmt --check`, `cargo clippy --all-targets --all-features`, and all Rust/Python tests on the final tree.
3. Verify every self-owned commit independently where its dependency boundary permits it.
4. Review `origin/main..HEAD` as a community owner; only evidence-backed findings may block completion.
5. Record exact base/head, machine identity, CPU pinning, raw artifact locations, results, fixes, and remaining non-production boundaries.
