# Router Policy 大规模 Virtual Fleet 模拟报告

日期：2026-08-25
状态：完成 CPU 逻辑时间验证；不构成真实 GPU fleet 的生产性能结论

## 1. 结论

[SGLang Simulator #33824](https://github.com/sgl-project/sglang/pull/33824)
可以用于这次实验，但不能直接把一个 Simulator engine 当成数百个 Router worker。
该 PR 的 Simulator 复用了 SGLang scheduler、缓存运行时和延迟预测器，但一个进程仍只
代表一个 engine。为验证 Router 在大候选域中的实际决策行为，本实验复用 Router 的
`PowerOfTwoChoicesPolicy`、`CacheAwarePolicy`、`CandidateDomain`、Admission 和
Pressure Guard，在测试中维护 8 / 64 / 256 个 virtual worker 的 queue/cache 状态。

结果支持两个结论：

- Cache-Aware 在预置可复用前缀的 workload 中始终得到 100% cache hit；相对 P2，在
  `20k rps` 的逻辑高压下，8 worker 的预测吞吐提高 **2.55x**、p95 logical TTFT 降低
  **72.7%**，64 worker 为 **2.21x**。
- 这不是“Cache-Aware 同时实现全局均衡”的证明。没有足够 queue pressure 时，它会稳定地
  选中同一组 cache holder；因此 worker CV 可达 1.732。Pressure Guard 只在 cache
  候选域内扩展低压候选，不承诺把流量均匀铺到整个 fleet。

接口/算法路径和 CPU 开销的 smoke 检查为 **GO**；将 4 个 cache replica 静态映射到
256 worker 的策略直接作为生产 fleet 配置为 **NO-GO**。生产前仍需要真实 Indexer、
LoadMonitor stream、HTTP Router 和 GPU endpoint 的规模压测。

## 2. 目的与边界

本实验回答的是：在候选域从 8 扩大到 64、256 时，现有 P2 / Cache-Aware 的真实 Rust
决策路径是否仍能正确选择 worker、按缓存收益降低 logical prefill cost，并在高压力下
避免一直绑定明显拥堵的 cache holder。

它没有验证以下内容：

- 256 个真实 SGLang server、GPU forward、端到端 HTTP latency 或真实吞吐；
- Indexer RPC / LoadMonitor stream 的网络开销和并发更新；
- HiCache prefetch、eviction、offload、KV transfer，及 PD/Decode/ITL policy；
- 真实热点分布、cache eviction 后的命中变化或 256 endpoint 的 CPU 内存占用。

报告中的 `TTFT` 是 **logical-time prefill/TTFT proxy**：命中 7/8 个 prefill block
使用 1 ms replay cost，完全未命中使用 8 ms replay cost。两项数值来自 #33824 的示例
replay table，而不是 H20 或任一 GPU 的实测延迟。

## 3. 测试对象和合同

临时集成的 Simulator 代码来自 #33824 的 6 个原始提交；没有推送、没有改动
`personal/vin/rust-v3`。最后一个上游修复必须保留：它通过 `apply_simulator_server_args`
构造不可变的当前 `ServerArgs`，否则 Simulator runner 不能启动。

### 3.1 上游 Simulator 回归

在 `h20-8-usa` 的 CPU engine 模式下运行，使用当前 SGLang 所需的
`transformers==5.12.1` 和独立 Python 依赖目录：

| 套件 | 结果 | 说明 |
| --- | ---: | --- |
| `test_simulation_sglang_runner.py` | 1 / 1 passed | 实际启动 `sglang::scheduler` |
| serving / cache-hit / offline-blocking | 7 / 7 passed | Scheduler、cache runtime、serving 路径 |
| 合计 | **8 / 8 passed** | CPU-only，无 GPU forward |

### 3.2 Virtual Fleet 正式矩阵

每个 release process 运行下列矩阵一次；共三个独立 process。

| 维度 | 值 |
| --- | --- |
| endpoint 数 | 8、64、256 |
| 策略 | `power_of_two`、`cache_aware` |
| 到达率 | 500 rps（2.0 ms 间隔）、20k rps（0.05 ms 间隔） |
| 每个内部 repeat | 4,096 请求 |
| 每个 group 的样本 | 9（3 process × 3 repeat），36,864 请求 |
| 前缀 | 256 个 hot prefix，每个预置 4 个 cache replica |
| 请求 | 8,192 token / 8 blocks；每次命中 7 blocks |
| Admission / Guard | 正式 `resolve_cache_candidates` / `resolve_prefill`；Cache-Aware 开启 Pressure Guard |

每次请求构造新的不可变 `LoadMonitorSnapshot` 和真实 `PrefixOutcome` / `PrefixMatch`，调用
生产 Router policy，再将选择结果反馈到 virtual worker 的 `available_at_ms` 与 cache 集合。
因此它检查的是 Router 当前策略、Admission 和 Guard 的组合行为，而不是另写一套近似打分器。

正式原始结果在本机：
`/Users/gaobo/Documents/mooncake/.vin_stage/router-simulator-scale-eval-results-20260825/virtual-fleet-release-v2-500-20k/`。
此前将 0.5 ms 错标为低负载的两组目录已保留但没有纳入本文分析。

## 4. 正式结果

以下是每个 group 的 9 个样本中位数；所有 `invalid_selections=0`，主指标 RSD 均小于
2.1%。`decision p95` 包含测试构造 snapshot / prefix signal 的成本，仅作 smoke 指标。

| Workers | Arrival | Policy | Hit rate | mean / p95 logical TTFT (ms) | logical TPS | Worker CV | decision p95 (us) |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: |
| 8 | 500 rps | P2 | 77.73% | 2.588 / 8.0 | 500.1 | 0.044 | 0.343 |
| 8 | 500 rps | Cache-Aware | 100% | 1.0 / 1.0 | 500.1 | 1.732 | 3.157 |
| 64 | 500 rps | P2 | 16.58% | 6.845 / 8.0 | 499.6 | 0.120 | 0.984 |
| 64 | 500 rps | Cache-Aware | 100% | 1.0 / 1.0 | 500.1 | 1.732 | 5.924 |
| 256 | 500 rps | P2 | 4.42% | 7.691 / 8.0 | 499.6 | 0.244 | 3.244 |
| 256 | 500 rps | Cache-Aware | 100% | 1.0 / 1.0 | 500.1 | 1.732 | 14.130 |
| 8 | 20k rps | P2 | 77.93% | 681.492 / 1072.5 | 3128.9 | 0.030 | 0.343 |
| 8 | 20k rps | Cache-Aware | 100% | 154.300 / 292.6 | 7994.5 | 0.000 | 3.319 |
| 64 | 20k rps | P2 | 16.55% | 130.093 / 228.8 | 9026.0 | 0.047 | 0.992 |
| 64 | 20k rps | Cache-Aware | 100% | 1.0 / 1.0 | 19907.7 | 1.000 | 5.988 |
| 256 | 20k rps | P2 | 4.32% | 8.924 / 14.0 | 18614.0 | 0.153 | 3.244 |
| 256 | 20k rps | Cache-Aware | 100% | 1.0 / 1.0 | 19907.7 | 1.732 | 14.149 |

在低到达率下，两策略都能完成 500 rps，因此吞吐相同；差别是 Cache-Aware 的命中和
logical TTFT。在高压下，缓存收益转化为更短的服务时间：8 worker 和 64 worker 的
P2 出现明显排队，Cache-Aware 分别为 2.55x 和 2.21x 的 logical TPS。

256 worker / 20k rps 的 P2 已接近到达率，因此 Cache-Aware 的相对吞吐收益缩小到约
1.07x；它仍保持 1 ms logical p95，而 P2 为 14 ms。

## 5. 对策略设计的含义

### Cache-Aware 的收益和边界

这里的全部前缀都有 4 个预热 replica，所以所有 Cache-Aware 决策原因都是
`CacheCandidate`。它证明了“top-K cache candidate → admission → guard → 选 primary”
路径可在 256 worker 候选域工作，但**没有**覆盖没有 winner 时的 `bucket + P2` fallback。

稳定的 cache holder 选择是预期行为，不是 P2 的替代：

```text
有足够低压 cache holder
  → 保留缓存收益最高的候选

cache holder 过压
  → Pressure Guard 在 cache candidate 中扩大/切换到较低压候选

没有有效 cache winner
  → 才退化到该请求候选范围内的 P2
```

因此，worker CV 不能单独作为 Cache-Aware 成败标准。低压的 CV=1.732 是 4 个
cache replica 的稳定分布结果；在 64 worker / 20k rps 下 Guard 已把 CV 降到 1.0，
而 256 worker 时这 4 个 replica 自身仍未超压，Guard 无需扩域。

### 热路径规模信号

由 8 扩展到 256 worker，测试内 `decision p95` 从约 0.34 us 增长至：P2 3.24 us、
Cache-Aware 14.15 us。Cache-Aware 的增长来自构造和处理 4 个 cache match，而不是全局
扫描 256 worker。它与已有 Router-only pressure microbenchmark 的结论一致：候选域、
top-K 和快照访问应保持有界；这个测试本身不替代正式的 allocation / CPU profile。

## 6. 可复现性和后续验证

测试实现在：
`tests/component/policies/simulator_fleet.rs`。它包含：

- 8 / 64 / 256 的 known-selection 与 cache-benefit 覆盖；
- 高压力下 Cache-Aware 相对低压自身会扩散的行为断言；
- 可通过 `SGL_ROUTER_SIMULATOR_FLEET_REPORT` 导出 JSON 矩阵。

下一轮应使用 #33824 的配置/replay 能力做参数敏感性分析，同时单独执行真实 Router
规模测试：8 / 64 / 256 HTTP endpoint、真实 Indexer top-K、LoadMonitor 增量更新和
Router RPS。验收指标至少包括 Router decision p50/p95、allocation、CPU、snapshot age、
cache hit、TTFT、吞吐、worker CV 和 fallback reason。只有该矩阵通过后，才能判断
Cache-Aware 在大规模 GPU fleet 上是否可生产启用。
