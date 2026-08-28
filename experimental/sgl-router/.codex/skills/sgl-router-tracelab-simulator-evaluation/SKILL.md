---
name: sgl-router-tracelab-simulator-evaluation
description: Use when 需要运行或审查基于 TraceLab、Indexer、LoadMonitor 的可复现 256-worker SGLang HTTP Simulator 测试。
---

# SGLang Router TraceLab Simulator Evaluation

使用本 Skill 复现 256-worker TraceLab HTTP Simulator 对照，或核验其结果。详细命令、
环境变量和验收项见
[`docs/sop/router-tracelab-256-http-simulator-sop.md`](../../../docs/sop/router-tracelab-256-http-simulator-sop.md)。

## 固定入口

- Replay runner：`scripts/run_tracelab_simulator_http_fleet_e2e.py`
- Analyzer：`scripts/analyze_simulator_http_fleet_e2e.py`
- Trace 选择与虚拟 prompt：`scripts/tracelab_replay.py`
- Simulator runtime：`tools/sglang-simulator/`
- 对应测试：`tests/scripts/test_{tracelab_replay,run_tracelab_simulator_http_fleet_e2e,analyze_simulator_http_fleet_e2e}.py`

## 必须保持的合同

- 使用 `vin/rust-v3-simulator-test` 或它的干净临时 worktree；运行期间不改动测试源码。
- 固定 256 worker、256 session、每 session 4 turn、64 QPS、四策略各三轮。
- Native `cache_aware` 与 `shortest_ttft` 使用 `--require-indexer-success`，预算固定为
  10s / 256 in-flight / 512 streams。
- 先跑脚本测试和 dry-run；实际运行前确认 source commit、release binary 和 trace SHA。
- 远端源码与结果只用 `rsync` 传输；使用新的结果目录，不覆盖或删除既有结果。
- 失败时仅在合同、制品和结果目录均不变时使用 `--resume`；否则建立新的结果目录。

## 结果判断

正式结果要求 `RUN_COMPLETE=ok`、全部 case `COMPLETE=ok`、零请求错误、零 fatal/OOM。
Analyzer 必须通过 Native Cache-Aware audit：每个 cache candidate 有 Monitor 决策和真实 cache
metric，且没有 `router_local` 或 zero-snapshot fallback。主指标三轮 RSD 高于 10% 时，保留主轮，
以独立目录补两轮 confirmation。

Simulator 输出只能标记为 `simulator_predicted_relative`：它用于控制面容量、候选选择和策略
相对行为，真实 GPU 的 TTFT、E2E 与吞吐应通过独立 GPU E2E 合同判断。
