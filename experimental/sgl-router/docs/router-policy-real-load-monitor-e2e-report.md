# Router Policy：真实 LoadMonitor 生效 E2E 验证

日期：2026-08-25

## 结论

本轮补齐了此前 virtual-fleet 测试缺少的证据：Router 并非只消费测试构造的
`LoadMonitorSnapshot`，而是已在真实 8×L20 SGLang engine 上完成以下闭环：

```text
真实 Engine 负载与队列
  → worker load reporter gRPC（45000..45007）
  → Router --load-monitor 的 outbound session
  → 每请求捕获的非零 LoadMonitor snapshot
  → Session-Aware / Cache-Aware shared admission
  → Pressure Guard 改变最终 worker
```

核心压力 proof 通过：Session primary `31007` 的真实 worker queue 达到 `3` 后，
同一 session 的下一请求切换到 `31006`；Router 日志给出
`reason=BackupPressureGuard` 和 `load_snapshot_version=80`。`session_pressure_backup`
恰好增加 1，`session_admission_backup` 增量为 0。因此这不是 hard Admission 容量拒绝，
而是 LoadMonitor 压力数据驱动的软逃逸。

这证明 Router 的 LoadMonitor 接入、Cache/Session policy 消费和 Prefill Pressure Guard
在真实 GPU runtime 中可用。它不是一次端到端性能 benchmark，也不证明当前 source 与
`sglang-kernel==0.4.6.post1+cu129` 在 L20 上兼容，详见第 5 节。

## 1. 固定实验合同

| 项目 | 值 |
| --- | --- |
| Router source worktree | `codex/router-simulator-scale-eval@54fd67dad1` |
| Router binary SHA-256 | `9377c8ff0a1dc6d64b4f7959175f29fcf10c9af5f1b0e10c591ad55c99afc32c` |
| Router / Indexer build | release、`cargo build --locked` |
| Lock artifact SHA-256 | `9a997034df2e806c50cc56745df72781c847d5b8d6ec29e6f9f7c90ea0d650aa` |
| 模型 / 硬件 | `Qwen/Qwen2.5-7B-Instruct`，8 × NVIDIA L20，单卡 worker |
| Worker 设置 | `--enable-metrics --enable-cache-report --load-reporter-port 45000..45007` |
| Router 设置 | `--load-monitor`，`session_aware` / `cache_aware`，Decode P2 |
| Indexer | 本机 `kv-indexer-server` + 8 个 bridge，100 ms query timeout |
| Engine 运行形状 | Triton attention，CUDA Graph 禁用，Torch fused-op fallback |

`experimental/sgl-router/Cargo.lock` 是当前 worktree 中被忽略的构建输入；本轮仅在远端
隔离 source archive 中放入同一 `0facd` 基线的已验证 lock，不修改或提交该文件。

## 2. 真实 reporter 和 Cache/Session preflight

最终 preflight 使用 8 个真实 worker，先检查每个 `/server_info` 的 reporter port，再启动
Router，执行 Cache-Aware、`cache_aware_zmq`、Sticky 与 Session-Aware 的最小真实请求。

| 验收项 | 结果 |
| --- | --- |
| 8 个 reporter port | `45000` 到 `45007`，逐一与 worker 配置一致 |
| Router snapshot versions | `15, 23, 23, 23, 287, 287`，均为真实请求日志提取 |
| Cache-Aware / Indexer | warm 在 `31000` / `31001` 的两个长 prefix 均回到对应 holder |
| Cache-Aware 决策日志 | `CacheCandidate`，`input_tokens=4125`，`matched_prefix_tokens=4125`，`uncached_tokens=0`，snapshot `287` |
| Session-Aware | 同一 session 的三轮均落到 `31002` |
| preflight 结果 | `ok=true` |

这一步同时证明：worker 启动参数不是孤立存在；Router 已实际连 reporter 并在
Cache-Aware 的真实 Indexer 命中请求上读取 snapshot。对于 SGLang 的 Engine Prometheus
metrics 约定，参见 [SGLang Observability 文档](https://docs.sglang.ai/advanced_features/observability.html)。

## 3. 真实 Pressure Guard 因果 proof

只验证“snapshot 非零”仍不足以证明 Monitor 影响 policy。因此第二个 probe 固定以下
因果链：

1. 将所有 worker 的 `max_running_requests` 固定为 8；
2. 用 Session-Aware 的第一轮请求确定 primary；
3. 对该 primary 直接并发发送 4 个长、互不复用的真实请求；
4. 在 worker `/metrics` 观测到 queue depth 至少 2 后，等待一个 reporter period；
5. 发送同一 session 的第二轮请求，并检查 Router reason、选择 worker 与 snapshot version。

最终结构化结果如下：

| 字段 | 实测值 | 含义 |
| --- | --- | --- |
| initial primary | `http://127.0.0.1:31007` | 第一轮 Session-Aware primary |
| max primary queue reqs | `3` | 真实 Engine `/metrics` 观测值 |
| pressure escape worker | `http://127.0.0.1:31006` | 第二轮实际选择的不同 worker |
| `session_pressure_backup` delta | `+1` | 恰好一次 Pressure Guard escape |
| `session_admission_backup` delta | `0` | 没有把 hard Admission 当作 Guard 成功 |
| max snapshot version | `80` | Guard 决策消费了非零实时 snapshot |
| Router debug reason | `BackupPressureGuard` | 最终决策原因 |
| complete / fatal scan | `RUN_COMPLETE=ok` / `{}` | 无 fatal、OOM 或遗留 worker |

关键日志等价于：

```text
primary=31007
backup=31006
selected=31006
reason=BackupPressureGuard
load_snapshot_version=80
```

由于 `pressure_guard_prefers_backup` 只能在 primary、backup 都有 fresh snapshot report，
且压力差跨过配置阈值时返回 true，上述结果同时证明 Engine report 被 Router 接收、判断为
fresh，并真正参与 Session-Aware 的最终选择。

## 4. 验证范围

本轮可以确认：

- `--load-monitor` 确实建立了 Router 发起的 worker reporter 会话；
- Router request path 确实捕获并消费真实 snapshot，而不是本地伪造值；
- Cache-Aware 的 Indexer 命中路径与真实 LoadMonitor 可以同时工作；
- Session-Aware 的 Guard 在 real queue pressure 下能软切换到 backup；
- 该 escape 不是 hard Admission 满容量 fallback；
- 两组最终结果均完成且没有 fatal/OOM。

本轮不应外推为：

- 完整 QPS/TTFT/TPS 性能收益或大规模 endpoint 容量结果；
- Cache-Aware 在两种不同 cache-depth 之间由 Guard 覆盖 cache benefit 的量化阈值；
- HiCache prefetch/eviction/offload、PD transfer-aware Decode、Bucket/SLO 或异构硬件行为；
- 当前 source 的新 kernel wheel 已通过 L20 runtime 兼容性。

## 5. 运行时兼容性边界

当前 source 指定 `sglang-kernel==0.4.6.post1`。官方 CUDA 12.9 索引中的
`0.4.6.post1+cu129` wheel 在本机 L20（SM89）探测时只提供 `sm90/sm100` common ops，
并触发 Torch ABI symbol load failure；因此不能用它运行本轮 GPU proof。

为只验证 Router/Monitor 数据流，实验环境使用了已在该机器 import 通过的
`sglang-kernel==0.4.5+cu129`，并仅在隔离 wrapper 中设置
`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`。当前 Python source、Router binary、Reporter
和 policy 都来自本轮 source；但该 kernel workaround 意味着本报告**不覆盖新 kernel ABI**。
生产验证必须在有 SM89-compatible 0.4.6.post1 kernel artifact 的环境重新执行相同合同，
且不设置 skip 变量。

## 6. 制品

本地只读原始结果：

```text
/Users/gaobo/Documents/mooncake/.vin_stage/load-monitor-real-effect-20260825/results/preflight-r3
/Users/gaobo/Documents/mooncake/.vin_stage/load-monitor-real-effect-20260825/results/session-guard
```

其中 `preflight-r3/preflight/preflight.json` 记录 reporter ports、snapshot versions 和
Cache/Session landing；`session-guard/result.json` 记录 queue、reason delta、最终 worker、
snapshot version 和二进制/runner provenance。失败的 r1（Transformers API）和 r2（新 kernel
版本门槛）目录也保留在远端，未被删除或并入本报告。
