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
- 已从 `vin/shortest-ttft` 建立隔离 worktree
  `/Users/gaobo/Documents/mooncake/.worktrees/shortest-ttft-indexer`，分支
  `vin/shortest-ttft-indexer`；基线全量 router 测试通过。
- 已提交 Indexer 变体设计：`184a7c3dcc1 docs(router): design shortest TTFT indexer variant`。
- 已写入实施计划；下一步按 TDD 新增 external signal 覆盖 local tree 与 CLI
  解析的失败测试。
- 当前 shell 未配置 cargo PATH，已定位本机固定 Rust 1.90 toolchain；后续测试
  使用其绝对路径，不安装或修改全局环境。
- TDD 红灯已确认：`external_indexer_match_overrides_the_local_tree` 当前仍选本地
  tree 的 worker A，而不是 external signal 的 worker B；CLI 测试也因
  `ModelConfig.shortest_ttft` 尚不存在而以 `E0609` 失败。
- 已完成最小配置和 policy 接线；新增 model config 字段导致的构造字面量编译
  缺项已逐处补为 `None`，不改变既有测试的策略配置。
- 定向验证已通过：7 个 Shortest-TTFT 组件测试、2 个 Shortest-TTFT Indexer
  CLI 测试，以及既有 external Indexer proxy 测试。
- 完整 `cargo test --no-fail-fast` 已通过：452 library、3 binary、48 component、
  68 proxy 测试；`cargo fmt --all -- --check`、`git diff --check` 通过。当前正做
  最终只读审查，随后提交并从 vin 推送。
