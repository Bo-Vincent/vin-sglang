# Simulator HTTP Fleet Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 使用 #33824 的 CPU SGLang Simulator 启动真实 HTTP worker fleet，并在 32、128、256、512、1024 个 logical worker 下比较四个 Router policy。

**Architecture:** 每个 logical worker 是一个独立的 SGLang Simulator HTTP server，保留其 scheduler、cache 和请求生命周期。一个真实 `sgl-router` 连接全部 worker；Native Cache-Aware 额外连接 KV Indexer server 和每 worker 一个 bridge，并消费 LoadMonitor reporter。先以 4 worker 验证 HTTP、LoadMonitor、KV event、Indexer 和 Router 的协议闭环，再跑规模矩阵。旧的 Rust `simulator_fleet.rs` 仍作为 policy-only 回归，不和本次 HTTP fleet 结果混用。

**Tech Stack:** SGLang Simulator #33824、`experimental/sgl-router`、KV Indexer、LoadMonitor、Python `unittest`、远端 `h20-8-usa` CPU runtime。

---

### Task 1: 建立干净的四 policy 评估基线

**Files:**

- Modify: Git history of `codex/router-simulator-scale-eval` only
- Test: `experimental/sgl-router` Rust policy and proxy suites

**Step 1: 记录当前基线和上游 Simulator 提交**

Run:

```bash
git status --short --branch
git log --oneline -12
```

Expected: 工作树干净，Simulator 的六个提交和 `vin/rust-v3` 基线可追溯。

**Step 2: 引入最终 Cache-Aware 审计提交**

Run:

```bash
git cherry-pick e30c62fb362beeb7579756df928ee8fdae0c42a9
```

Expected: Cache-Aware admission/guard metrics 可由本次实验直接采集；不改变 winner 选择语义。

**Step 3: 引入 Shortest-TTFT 实验 policy**

Run:

```bash
git cherry-pick 095008c89e
```

Expected: 四个 requested policy 均可由同一 Router binary 创建。若与审计提交冲突，只保留原 policy 语义和审计字段；不重写其他 Router 代码。

**Step 4: 验证 policy 基线**

Run:

```bash
cargo fmt --check
cargo test --all-targets --no-fail-fast
```

Expected: 所有 Rust 测试通过，再进入 fleet harness 开发。

### Task 2: 定义并测试 Simulator fleet 合同

**Files:**

- Create: `experimental/sgl-router/scripts/run_simulator_http_fleet_e2e.py`
- Create: `experimental/sgl-router/tests/scripts/test_run_simulator_http_fleet_e2e.py`

**Step 1: 写失败的合同测试**

覆盖以下纯函数和 CLI 行为：

```python
def test_default_matrix_includes_requested_endpoint_counts_and_policies():
    assert build_cases() == expected_cases_for(
        endpoints=(32, 128, 256, 512, 1024),
        policies=("power_of_two", "cache_aware", "cache_aware_zmq", "shortest_ttft"),
    )

def test_simulator_worker_command_uses_cpu_blocking_and_unique_ports():
    command, env = simulator_worker_command(...)
    assert "sglang_simulator.simulation.sglang.launch_server" in command
    assert env["SGLANG_USE_CPU_ENGINE"] == "1"
    assert env["SGLANG_SIMULATOR_OUTPUT_MODE"] == "BLOCKING"

def test_cache_aware_case_requires_indexer_and_load_monitor_audit():
    assert validate_case_artifacts(...) is None
```

**Step 2: 运行测试并确认 RED**

Run:

```bash
python3 -m unittest experimental/sgl-router/tests/scripts/test_run_simulator_http_fleet_e2e.py -v
```

Expected: 因 runner 尚不存在或合同函数缺失而失败。

**Step 3: 实现最小 runner**

实现范围仅包括：

- endpoint count：`32,128,256,512,1024`；
- policy：`power_of_two`、`cache_aware`、`cache_aware_zmq`、`shortest_ttft`；
- Simulator worker 以 CPU `BLOCKING` mode 启动，worker 各有唯一 HTTP、LoadMonitor、KV event port；
- Cache-Aware/Shortest 启动 Indexer；所有 policy 使用同一 worker fleet 和 QPS 密度；
- measurement 前 warmup，采集 Router、worker 和 Simulator 指标；
- raw result 中保存合同、源码/二进制 hash、case command、错误、worker 分布和 policy reason；
- native cache case fail-closed：缺 Indexer candidate、fresh LoadMonitor 或 actual simulated cache counter 时该 case 失败。

**Step 4: 运行测试并确认 GREEN**

Run:

```bash
python3 -m unittest experimental/sgl-router/tests/scripts/test_run_simulator_http_fleet_e2e.py -v
python3 -m py_compile experimental/sgl-router/scripts/run_simulator_http_fleet_e2e.py
```

Expected: 测试通过且脚本可导入。

### Task 3: 实现结果分析器和目标压力 workload

**Files:**

- Create: `experimental/sgl-router/scripts/analyze_simulator_http_fleet_e2e.py`
- Create: `experimental/sgl-router/tests/scripts/test_analyze_simulator_http_fleet_e2e.py`
- Modify: `experimental/sgl-router/scripts/run_simulator_http_fleet_e2e.py`

**Step 1: 写失败的分析器测试**

覆盖：

```python
def test_analyzer_groups_three_repeats_by_endpoint_policy_and_workload(): ...
def test_analyzer_rejects_missing_or_nonzero_request_errors(): ...
def test_analyzer_reports_ttft_e2e_tps_hit_rate_cv_and_policy_reasons(): ...
def test_analyzer_marks_simulator_values_as_predicted_not_gpu_measurements(): ...
```

**Step 2: 运行 RED**

Run:

```bash
python3 -m unittest experimental/sgl-router/tests/scripts/test_analyze_simulator_http_fleet_e2e.py -v
```

Expected: 因 analyzer 不存在而失败。

**Step 3: 实现最小 analyzer 和第二 workload**

保留两个 workload：

- `tracelab_multiturn`：128 session × 4 turn，首轮 warmup；
- `multi_holder_pressure`：同一 prefix 在多个 Simulator worker 预热，候选间制造 queue skew，验证 admission reject、Guard compare 与 Guard override。

分析输出每个 endpoint/policy/workload 的三轮中位数、RSD、TTFT、E2E、TPS、实际 simulated KV hit、worker CV、candidate/fallback reason 和 Cache-Aware admission/guard counter。任何错误、fatal、OOM、缺 repeat 或缺期望计数时 fail closed。

**Step 4: 运行 GREEN**

Run:

```bash
python3 -m unittest \
  experimental/sgl-router/tests/scripts/test_run_simulator_http_fleet_e2e.py \
  experimental/sgl-router/tests/scripts/test_analyze_simulator_http_fleet_e2e.py -v
python3 -m py_compile \
  experimental/sgl-router/scripts/run_simulator_http_fleet_e2e.py \
  experimental/sgl-router/scripts/analyze_simulator_http_fleet_e2e.py
```

Expected: 所有 harness 单测通过。

### Task 4: 远端 4-worker 协议 smoke test

**Files:**

- Use: `experimental/sgl-router/scripts/run_simulator_http_fleet_e2e.py`
- Results: `/root/router-policy-bench/results/simulator-http-fleet-smoke-<date>/`

**Step 1: rsync 源码和固定制品到 `h20-8-usa`**

不覆盖既有结果；只使用 `rsync`，不使用 `scp`。

**Step 2: 运行 Simulator 上游 CPU serving 回归**

Run:

```bash
python3 -m pytest -q \
  tools/sglang-simulator/test/test_simulation_sglang_runner.py \
  tools/sglang-simulator/test/test_simulation_sglang_serving.py
```

Expected: Simulator 的 scheduler/cache/HTTP serving 回归通过。

**Step 3: 执行 4-worker smoke**

Run:

```bash
python3 experimental/sgl-router/scripts/run_simulator_http_fleet_e2e.py \
  --endpoint-counts 4 --repeats 1 --workloads tracelab_multiturn,multi_holder_pressure \
  --execute
```

Expected: 每个 policy 都得到 HTTP 响应；Native Cache-Aware 的 Indexer、fresh LoadMonitor、实际 simulated cache metric 证据均存在；失败时停止，不进入规模矩阵。

### Task 5: 运行正式规模矩阵

**Files:**

- Results: `/root/router-policy-bench/results/simulator-http-fleet-32-1024-<date>/`

**Step 1: 启动可恢复正式任务**

Run:

```bash
python3 experimental/sgl-router/scripts/run_simulator_http_fleet_e2e.py \
  --endpoint-counts 32,128,256,512,1024 \
  --policies power_of_two,cache_aware,cache_aware_zmq,shortest_ttft \
  --workloads tracelab_multiturn,multi_holder_pressure \
  --repeats 3 --execute --resume
```

Expected: 5 endpoint tiers × 4 policies × 2 workloads × 3 repeats = 120 completed cases。每一 tier 的 QPS 按 0.25 request/s/worker 缩放，并保留固定 replay seed。

**Step 2: 运行中持续审计**

检查 `CURRENT`、完成 case 数、非空 request error、fatal/OOM、CPU/RSS、Router snapshot age、Indexer bridge 存活和 worker 健康。失败只能在合同不变且既有结果可恢复时 `--resume`。

**Step 3: 正式分析**

Run:

```bash
python3 experimental/sgl-router/scripts/analyze_simulator_http_fleet_e2e.py \
  --results-dir /root/router-policy-bench/results/simulator-http-fleet-32-1024-<date>
```

Expected: 完整三轮、零错误、全部验收项和每组统计结果；高 RSD 的组保留主数据并补独立 confirmation repeats。

### Task 6: 写入报告并完成审计

**Files:**

- Create: `experimental/sgl-router/docs/router-policy-simulator-http-fleet-report.md`
- Modify: `experimental/sgl-router/docs/router-policy-simulator-large-fleet-report.md`
- Modify: `experimental/sgl-router/BENCHMARKS.md`

**Step 1: 写入新报告**

报告必须明确区分：

- 当前新结果：真实 Router + 多个 Simulator HTTP worker + LoadMonitor + Indexer；
- 旧 `simulator_fleet.rs`：policy-only logical-time 测试；
- 真实 8×L20 E2E：唯一可用于 GPU 绝对性能的锚点；
- Simulator TTFT/E2E/TPS：predictor 驱动的 relative policy result，不可表述为真实 GPU 性能。

**Step 2: 同步原始结果并验证本地副本**

使用 `rsync` 复制到 `.vin_stage`，核对 `manifest`、`RUN_COMPLETE`、case 数、analysis 和 source identity。

**Step 3: 最终验证**

Run:

```bash
python3 -m unittest \
  experimental/sgl-router/tests/scripts/test_run_simulator_http_fleet_e2e.py \
  experimental/sgl-router/tests/scripts/test_analyze_simulator_http_fleet_e2e.py -v
cargo fmt --check
cargo test --all-targets --no-fail-fast
git diff --check
```

Expected: harness、Router、Simulator 及文档引用均可复现；不推送、不删除既有结果。
