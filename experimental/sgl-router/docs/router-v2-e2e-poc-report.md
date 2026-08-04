# Router v2 E2E PoC 报告

日期：2026-08-04

分支：`vin/router-v2`

测试源码树：`0699b2ce75d5feb8f27d153263f2449d12312fbb`（报告回填前；回填后代码内容未变）

范围：`experimental/sgl-router` 的 Step 1 Policy/Admission/Guard，以及 Step 2
静态 Prefill Bucket 接口验证

## 结论

- **自研代码的接口、功能与 CPU 回归 GO。** `cargo fmt --check` 通过；Rust 测试
  `516 + 3 + 51 + 70` 全部通过。
- **真实 GPU Router 热路径 GO。** P2 与 ScorePolicy 主轮共 `576/576` 请求成功，
  P2 确认轮再完成 `192/192`；Session、Cache、Bucket、TTFT tier 和
  global-first 共 17 个定向请求全部成功。请求错误、fatal 和 OOM 均为零。
- **没有观察到新 Policy 框架的明显性能负担。** 同一工作负载下，ScorePolicy 相比
  P2 的请求吞吐为 `+0.20%`，P95 E2E 为 `-7.15%`，P95 ITL 为 `-15.97%`。
  这是一组固定顺序的 PoC 数据，只能说明没有明显回归，不能据此宣称 ScorePolicy
  一定优于 P2。
- **静态同构 4+4 Bucket 仍不建议直接用于生产。** Bucket 候选域约束和 SLO tier
  选择正确，但把同构容量硬拆为两半会降低无 SLO 流量的可用容量。生产 Bucket 仍需
  Registry Bucket Index、动态 Admission 和 overflow/fallback。

## 环境与方法

- 实际硬件为 **8 x NVIDIA L20 46 GiB**，不是 H20；模型为本地
  `Qwen2.5-7B-Instruct`，单卡单 endpoint，`mem_fraction_static=0.85`。
- Router 使用上述源码树的 release 二进制。worker 使用机器上已验证的 `e2e-v2`
  runtime，通过真实 OpenAI Chat HTTP 请求执行模型；Router 与 worker runtime 并非
  完全同 revision，因此本报告是 Router Policy 验证，不是整套 SGLang 性能认证。
- P2 和 ScorePolicy 都使用 512 input / 32 output、96 requests、QPS 16、
  max concurrency 32、warmup 8，固定跑三轮。P2 因主轮 P95 TTFT RSD 为 11.38%，
  保留主数据并追加两轮相同参数的确认实验。
- Cache-Aware 使用符合 `MatchExternalKvPrefix` gRPC wire contract 的隔离测试
  Indexer，把真实 prefix holder 指向 worker 7；请求仍经当前 Router 和真实 GPU worker
  执行。它验证 Indexer 信号消费和 Cache Policy，但不等价于生产 Redis/bridge Indexer。
- 原始结果通过 `rsync` 保存于未提交目录
  `benchmark-results/router-v2-e2e-poc-20260804/final-head-0699b2ce/`。

## 性能结果

下表为三轮均值；吞吐单位分别为 req/s 和 token/s，延迟单位为 ms。

| Policy | 成功/总数 | 请求吞吐 | 总 token 吞吐 | P95 E2E | P95 TTFT | P95 ITL | KV hit-rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Power-of-Two | 288/288 | 14.948 | 8,131.79 | 964.16 | 161.57 | 26.56 | 5.67% |
| ScorePolicy（load-based） | 288/288 | 14.979 | 8,148.40 | 895.18 | 161.38 | 22.32 | 5.77% |

除 P2 主轮 P95 TTFT 外，所有主指标 RSD 均小于 10%。P2 两轮确认结果为
`192/192` 成功，P95 TTFT 均值 `178.43 ms`、RSD `8.51%`；吞吐 RSD `0.43%`，
P95 E2E RSD `5.05%`，P95 ITL RSD `0.97%`。确认轮没有复现持续性高抖动。

## 策略路径检查

| 路径 | 验证结果 |
| --- | --- |
| Session-Aware + Stable Pair | 同一 session 的 10 个请求全部命中 worker 6；首请求后均复用 1,052 cached tokens |
| Cache-Aware + Indexer signal | Indexer 指定的 worker 7 被选中，并实际复用 1,052 cached tokens |
| Prompt Bucket | short 请求只进入 worker 0–3，long 请求只进入 worker 4–7 |
| TTFT SLO tier | 无 SLO 选择 rank 更低的 cheap bucket；100 ms TTFT SLO 选择 fast bucket |
| global-first affinity | long 请求先落到 long bucket；同 session 的 short 请求保留跨 Bucket primary |

这些结果说明当前数据流按预期工作：

```text
Request
  -> Bucket / Eligible candidate domain
  -> P2 or Session/Cache affinity primary + backup
  -> Admission and Guard
  -> real GPU worker
```

## 验证边界

- Cache-Aware 已验证真实 Router、真实 GPU KV 复用和 Indexer RPC 消费；生产
  Redis/bridge Indexer 的容量、时效性和故障降级不在本次 E2E 范围内。
- LoadMonitor snapshot、Prefill Admission/Guard 和 `estimated_prefill_queue_ms` 降级由
  component/proxy 测试覆盖；本轮没有启动生产 LoadMonitor 数据链路。
- Decode policy/guard 的候选过滤、P2 选择和 retraction 处理由 component/proxy 测试
  覆盖；本轮 Plain-worker 拓扑不构成真实 P/D 分离性能实验。
- 当前 Bucket 是静态配置 PoC。动态 pod 归属、Bucket overflow、异构硬件 profile 和
  TPS Bucket 属于后续增量，不能从本报告推导生产 GO。
- 结果目录中的 `failure*.json` 是验证 harness 对 benchmark JSON、Prometheus label 和
  nullable cache detail 的兼容问题记录；对应请求与 Router 均成功，修正 harness 后按
  append-only 方式重跑并完成，没有覆盖原始失败证据。
