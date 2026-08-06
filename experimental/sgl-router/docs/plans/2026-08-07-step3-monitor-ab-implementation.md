# Router Step 3 Monitor A/B Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 整理 Step 1/2/3 提交边界，并在一晚内完成 Step 2 与 Step 3 Router 的 48-case 配对 A/B。

**Architecture:** 保持同一套 Step 3 Engine/Reporter、Indexer、模型和 4P+4D runtime，仅切换 Step 2 与 Step 3 Router binary。实验 runner 在 repo 外的 `.vin_stage` 中生成冻结 manifest、交错执行 A/B，并把正式原始结果通过 `rsync` 同步回本地。

**Tech Stack:** Rust、Python、protobuf/gRPC、SGLang Benchmark、systemd、jq、rsync。

---

### Task 1: 冻结当前提交与工作树

**Files:**
- Inspect: `experimental/sgl-router/src/**`
- Inspect: `python/sglang/srt/load_reporter/**`
- Inspect: `experimental/sgl-router/docs/**`

1. 记录当前 HEAD、`origin/main`、自有提交和未跟踪文件。
2. 创建本地备份 ref，不改工作树。
3. 用 `git diff --check` 和路径清单确认 Step 3 diff 范围。
4. 记录受保护的 `benchmark-results` 和已有未跟踪探针，后续不删除、不覆盖。

### Task 2: 整理 Step 3 Reporter 提交

**Files:**
- Modify: `experimental/sgl-router/proto/load_monitor.proto`
- Modify: `python/sglang/srt/load_reporter/proto/load_monitor.proto`
- Modify: `python/sglang/srt/load_reporter/proto/load_monitor_pb2.py`
- Modify: `python/sglang/srt/load_reporter/store.py`
- Modify: `python/sglang/srt/load_reporter/README.md`
- Create: `test/registered/unit/load_reporter/test_pressure_metrics.py`

1. 运行 Reporter 定向测试并确认通过。
2. 确认两份 proto 完全一致且新字段均为 optional。
3. 只暂存 Reporter/proto/test 文件。
4. 使用规定身份提交 `feat(load-reporter): expose Step 3 pressure metrics`。

### Task 3: 整理 Step 3 Router 提交

**Files:**
- Modify: `experimental/sgl-router/src/config/cli.rs`
- Modify: `experimental/sgl-router/src/config/types.rs`
- Modify: `experimental/sgl-router/src/load_monitor/mod.rs`
- Modify: `experimental/sgl-router/src/load_monitor/tests.rs`
- Modify: `experimental/sgl-router/src/policies/admission.rs`
- Modify: `experimental/sgl-router/src/policies/cache_aware.rs`
- Modify: `experimental/sgl-router/src/policies/mod.rs`
- Modify: `experimental/sgl-router/src/policies/session_aware.rs`

1. 运行 LoadMonitor 派生、reset/restart、Prefill Guard 和 Decode pressure 定向测试。
2. 运行 `cargo fmt -- --check`、`cargo clippy --all-targets -- -D warnings` 和
   `cargo test --all-targets`。
3. 只暂存 Router Step 3 文件。
4. 提交 `feat(sgl-router): consume Step 3 load metrics in policy selection`。

### Task 4: 整理 Step 3 文档提交

**Files:**
- Modify: `experimental/sgl-router/docs/router-policy-step1-integration.md`
- Modify: `experimental/sgl-router/docs/router-policy-step1-step2-design.md`
- Create: `experimental/sgl-router/docs/router-policy-step3-load-monitor-design.md`
- Create: `experimental/sgl-router/docs/plans/2026-08-07-step3-monitor-ab-design.md`
- Create: `experimental/sgl-router/docs/plans/2026-08-07-step3-monitor-ab-implementation.md`

1. 确认 Step 1/2 文档只增加 Step 3 边界引用，不改变原语义。
2. 检查文档中的字段、默认值和 CLI 与源码一致。
3. 提交 `docs(sgl-router): document Step 3 monitor integration`。

### Task 5: 刷新主线并保持提交边界

1. `git fetch origin main`，记录新 `origin/main`。
2. 对完整集成分支做机械 rebase；冲突只解决为等价迁移，不改依赖 PR 语义。
3. 用 `git range-diff` 分别审计依赖提交、Step 1、Step 2 和 Step 3。
4. 确认 rebase 后 Step 2 tip 是 A，Step 3 tip 是 B。
5. 重新运行路径 diff、`git diff --check` 和提交身份审计。

### Task 6: 构建配对基线并验证协议兼容

1. 为 A/B 创建只读 source archive，记录 tree hash。
2. 在 H20 上分别构建 Step 2 Router 和 Step 3 Router。
3. Engine/Reporter 使用 B source 启动 4P+4D、Indexer 和 Mooncake TCP。
4. 用 A Router 接收 B Reporter，验证 optional unknown fields 不影响 gRPC ingestion。
5. 用 B Router 验证非零 Prefill/Decode counter 和 fresh snapshot。
6. 两个 smoke 均要求 HTTP 200、无 LoadMonitor error、fatal 或 OOM。

### Task 7: 实现并验证短时 benchmark harness

**Files:**
- Create outside repo: `.vin_stage/router-step3-monitor-ab-20260807/ab-harness/scripts/run_step3_monitor_ab.py`
- Create outside repo: `.vin_stage/router-step3-monitor-ab-20260807/ab-harness/scripts/analyze_step3_monitor_ab.py`
- Create outside repo: `.vin_stage/router-step3-monitor-ab-20260807/ab-harness/test_contract.py`
- Create outside repo: `.vin_stage/router-step3-monitor-ab-20260807/ab-harness/test_analyzer.py`

1. 先写 contract tests，覆盖 48 个唯一 case、A/B 交错顺序、固定 QPS/repeat/workload、
   immutable manifest、resume 和结果目录隔离。
2. 运行 contract tests，确认缺少实现时失败。
3. 实现最小 runner/analyzer，复用历史 workload 参数和 SGLang benchmark 输出格式。
4. 再运行 contract tests 和 `py_compile`，确认通过。
5. 执行 dry-run，核验恰好 48 case 且预计时长不超过一晚。

### Task 8: 运行正式 48-case A/B

1. 在独立结果目录启动可恢复 systemd service。
2. 运行中检查 COMPLETE、CURRENT、请求错误、LoadMonitor error、fatal/OOM 和 GPU。
3. 不改变冻结参数；失败时先诊断，只在合同一致且结果可恢复时 `--resume`。
4. 48/48 后运行 analyzer。
5. 仅对结论敏感且主指标 RSD > 10% 的组补确认轮，最多 12 case。

### Task 9: 与历史数据对照并收尾

1. 从历史 108-case 提取 `random/power_of_two`、`session_hotspot/session_aware`、
   `hot_prefix/cache_aware`、`multiturn/cache_aware`。
2. 报告当晚配对 A/B 为主、历史跨时间对照为辅。
3. 将正式结果通过 `rsync` 同步到 `.vin_stage`，不写受保护的 repo benchmark-results。
4. 更新 Step 3 文档和 `BENCHMARKS.md`，通过 fixup/rebase 落入 Step 3 docs 提交。
5. 使用 review-as-community-owner 审查自有 Step 3 diff。
6. 重新运行 Rust、Python、manifest、analyzer 和 `git diff --check` 全部门禁。
7. 不 push；报告最终提交、数据、改善/中性/退化结论和所有非生产边界。
