# 进度日志

## 2026-09-01

- 创建本地无 upstream 分支 `vin/shortest-ttft`，worktree 为 `/Users/gaobo/Documents/mooncake/.worktrees/shortest-ttft`。
- 锚定 V4 基线 `20a491d1d311553bbab3f22e19bbafb86ef3c0cc`。
- 已确认 RTP-LLM 参考实现需要缓存命中和 engine 排队时间，后续实现不能把已有 cache-aware/admission 代码当作依赖。
- 已确认 #34608 的三帧负载协议、`LoadStat` 字段和 `/server_info` 的端点发现字段；下一步是写失败测试。
- 新增 `component::policies::shortest_ttft::engine_queue_can_outweigh_a_better_prefix_match`，并用绝对 Rust toolchain 跑到预期红灯：`E0432 could not find shortest_ttft in policies`。
- 最小 `ShortestTtftPolicy` 与 `EngineLoadTable` 已实现；同一测试已通过（1 passed）。
- `monitor_consumes_a_gauge_and_removes_worker_state` 已按 TDD 转绿（1 passed），验证真实 load PUB、SUB、状态清理链路。
- `shortest_ttft_builds_via_factory` 已按 TDD 转绿（1 passed），策略可作为独立 `PolicyKind` 选择。
- `fetch_resolves_shortest_ttft_load_endpoint` 已按 TDD 转绿（1 passed），`/server_info` load 端点解析完成。
- 新增 manager 生命周期测试，预期红灯为 `E0425 run_with_config_and_monitor not found`。
- 实现 monitor-aware worker-manager 入口：Added 后基于 `/server_info` 启动独立 load subscriber，Removed 时 cancel/join 后清除状态；对应生命周期测试转绿。
- `main` 仅为 `PolicyKind::ShortestTtft` 创建 monitor，并把同一 table 同时交给 factory 和 worker manager；退出时显式 shutdown monitor。
- CLI 已接受 `--policy shortest_ttft`，同时拒绝把它作为 sticky fallback；两项测试转绿。
- 增加 RTP candidate-window 公平性、DP gauge 完整性/过期、publisher sequence 重置、移除后迟到 gauge 的组件回归覆盖。
- 提交前已重新运行 `cargo test --no-fail-fast`：450 library + 3 binary + 46 component + 68 proxy 测试全部通过；`cargo fmt --check` 和 `git diff --check` 通过。新策略模块未导入 `cache_aware_zmq` 或 admission，分支 `vin/shortest-ttft` 确认无 upstream。
