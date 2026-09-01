# Shortest-TTFT 独立策略计划

## 目标

在本地分支 `vin/shortest-ttft` 上，从 V4 基线 `20a491d1d311553bbab3f22e19bbafb86ef3c0cc` 独立实现贴近 RTP-LLM 的 Shortest-TTFT 路由策略；不复用或改动现有 cache-aware/admission-guard 策略框架，也不推送远端。

## 验收标准

- 新策略可通过配置选择，并且现有策略行为不变。
- 分数语义是 `预填充估算时间(请求 token - 命中 token) + engine 排队时间`，且采用 RTP-LLM 的候选集与公平性选择。
- 缓存命中与 engine load 各自通过最小、独立的输入路径取得；不导入现有 `cache_aware_zmq` 或 admission 模块。
- load 状态的 worker 加入、移除、发布者重启与过期数据行为具备定向测试。
- 定向 Rust 测试、格式检查与隔离扫描通过；分支无 upstream 且未推送。

## 阶段

| 阶段 | 状态 | 内容 |
| --- | --- | --- |
| 1. 基线与算法合同 | 完成 | 已锁定 RTP-LLM 语义、V4 基线和 #34608 wire contract。 |
| 2. 测试先行 | 完成 | 策略、monitor、factory、`/server_info`、worker 生命周期和 CLI 均已红绿验证。 |
| 3. 最小实现 | 完成 | 独立 policy、load table、monitor、config/factory、端点解析和 Router 启停接线完成。 |
| 4. 验证 | 完成 | 全量 Rust 测试、格式、diff 和独立性检查均已通过；分支确认无 upstream。 |

## 决策与限制

- 分支基线必须保持为 `20a491d1d311553bbab3f22e19bbafb86ef3c0cc`，不从脏的 `vin/rust-v4` 分叉。
- RTP-LLM Shortest-TTFT 实际依赖缓存命中和 `runningQueueTime`，因此两个输入都属于策略所需合同。
- 本任务仅本地保存；禁止 `git push`、创建 PR 或远端 CI。

## 错误记录

| 错误 | 次数 | 处理 |
| --- | --- | --- |
| 无 | 0 | — |
| `run_with_config_and_monitor` 不存在 | 1 | 新增显式可选 monitor 入口，旧入口保留 `None`。 |
| 在 `experimental/sgl-router` 子目录错误引用工作树相对路径 | 1 | 仅只读 `nl` 命令失败；改用该子目录下的 `src/...` 路径，无代码影响。 |
| 在工作树根目录执行 Cargo 命令 | 1 | 根目录没有 `Cargo.toml`；改在 `experimental/sgl-router` 运行，未产生文件改动。 |
