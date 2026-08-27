# Router Policy：256 Worker HTTP Simulator 四策略对比

日期：2026-08-27
状态：已完成；`simulator_predicted_relative`

## 1. 结论

本轮在一台机器上启动了 **256 个真实 HTTP Simulator endpoint**，并通过真实
`sgl-router`、LoadMonitor、KV Indexer 和 ZMQ KV event bridge 跑完四种策略：
`power_of_two`、Native `cache_aware`、`cache_aware_zmq`、`shortest_ttft`。

- 四策略、两种 workload、三轮主矩阵共 **24/24** 完成，所有请求错误和
  `policy_selection_failed` 均为 0。
- Native `cache_aware` 的 Cache-Aware + LoadMonitor 主路径完整生效：两个 workload
  合计 1,535 次 cache candidate 决策全部使用 Monitor 快照；`router_local=0`、
  `zero_snapshot=0`。
- 两种 cache-aware 策略均取得接近 100% 的实际 KV hit。ZMQ 版本在本合同中的 p95
  TTFT 最低；Native 版本的 p95 TTFT 高 16%–40%，但三轮波动很小，且完整覆盖了新的
  Indexer + LoadMonitor 路径。
- `shortest_ttft` 也保持了接近 100% 的 KV hit，但选择明显更集中，TTFT/E2E 都落后于
  两个 Cache-Aware 基线。其 p95 波动超过 10%，独立确认轮也复现了不稳定性，因此不把它
  作为这一阶段的默认 Prefill policy。

这是一项 **Router 控制面规模实验**：HTTP、Router、Indexer、LoadMonitor 和 256 endpoint
都是真实进程；模型前向使用 SGLang Simulator 的 CPU 延迟预测。因此表中的 TTFT、E2E 和
TPS 只用于同一合同内的相对比较，不能替代真实 GPU 集群的绝对性能数据。

## 2. 合同

| 项目 | 取值 |
| --- | --- |
| endpoint 数 | 256 个单进程 Simulator worker，CPU engine |
| Router / policy | `sgl-router`；P2、Native Cache-Aware、ZMQ Cache-Aware、Shortest-TTFT |
| 主矩阵 | 2 workload × 4 policy × 3 repeat = 24 cases |
| 每个 case | 256 请求，16 completion tokens，64 requests/s |
| Cache warmup | 每个 case 清空 cache 后，将长前缀直接预热到前 8 个 worker |
| 负载控制面 | 所有 Router 均启用 LoadMonitor；Native Cache-Aware / Shortest-TTFT 使用外部 KV Indexer |
| 产品源码 | `b228c3792b33a9a9287bbf2d3515deee4f548564` |
| Router 二进制 SHA-256 | `d5f18245ae6e2d6b14865757555d6879592ce2543d258df900293825cd0a4abf` |
| 运行机 | `h20-8-usa`；本实验不使用 GPU forward |

两种 workload 均为可复现的合成请求：

| Workload | 请求形态 | 验证重点 |
| --- | --- | --- |
| `tracelab_multiturn` | 32 个 `session_id` 各复用 8 次，所有请求共享长前缀，suffix 带 `turn` 标记 | 多轮会话形态下的前缀复用与策略选择 |
| `multi_holder_pressure` | 同一长前缀预热到 8 个 holder，请求使用不同 suffix | 多个 cache holder 同时可命中时的选择与负载分布 |

`tracelab_multiturn` 的 `turn` 是请求标签，不表示前一轮生成内容被后一轮消费；本轮不把它
表述为真实 agent trace 回放。`multi_holder_pressure` 没有注入固定的队列拥堵，因而验证的是
多 holder 选择，不单独证明强制过载下 Guard 的逃逸效果。

## 3. 主矩阵结果

表中数值为三轮中位数。`TPS` 是 completion tokens/s；`Worker CV` 越小表示请求分布越均衡。
Cache affinity 预期会提高 CV，因为它会把请求留在可复用 KV 的 holder 上。

### 3.1 `tracelab_multiturn`

| Policy | TTFT p95 (ms) | E2E p95 (ms) | TPS | KV hit | Worker CV | 主指标最大 RSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 110.6 | 154.0 | 1,000.3 | 37.87% | 0.984 | 2.02% |
| Native Cache-Aware | 102.5 | 140.7 | 997.0 | 99.94% | 5.799 | 1.60% |
| Cache-Aware ZMQ | **88.1** | **129.8** | **1,000.7** | 99.94% | 5.732 | 2.56% |
| Shortest-TTFT | 303.8 | 694.9 | 977.1 | 99.94% | 15.969 | 16.36% |

### 3.2 `multi_holder_pressure`

| Policy | TTFT p95 (ms) | E2E p95 (ms) | TPS | KV hit | Worker CV | 主指标最大 RSD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P2 | 134.2 | 175.7 | 992.9 | 36.71% | 0.948 | 1.48% |
| Native Cache-Aware | 147.0 | 216.2 | 989.8 | 99.98% | 5.633 | 1.64% |
| Cache-Aware ZMQ | **104.9** | **144.9** | **997.6** | 99.98% | 5.701 | 4.92% |
| Shortest-TTFT | 334.7 | 702.8 | 967.3 | 99.59% | 11.007 | 14.56% |

观察：

- Native Cache-Aware 相比 P2 显著提高 KV hit；在 `tracelab_multiturn` 中，p95 TTFT 从
  110.6 ms 降至 102.5 ms，E2E p95 从 154.0 ms 降至 140.7 ms。
- Cache-Aware ZMQ 在该 Simulator 合同中取得最低延迟。Native Cache-Aware 已使用真实
  Indexer + Monitor 路径，结果与 ZMQ 的差距反映当前两条控制面实现的整体开销，不外推为
  真实 GPU fleet 上的固定差距。
- `multi_holder_pressure` 中 Native Cache-Aware 为缓存收益保留多个 holder，TTFT 不以 P2
  为目标；这是缓存收益与均衡之间的策略取舍，需在真实 GPU 压力下继续校准 Guard 阈值。
- Shortest-TTFT 有极高 hit，但 CV 达到 11–16，说明其选择比 Cache-Aware 更集中；本实验
  没有证明这种集中能换来更好的端到端体验。

## 4. 策略路径审计

### Native Cache-Aware

Analyzer 对 Native policy 强制以下门槛：每个 cache candidate 必须有 cache 指标和
LoadMonitor 决策，且不能退化为 Router 本地负载或零版本快照。主矩阵通过：

| Workload | Cache candidate | Monitor 决策 | Router local | Zero snapshot | 其他 reason |
| --- | ---: | ---: | ---: | ---: | --- |
| `tracelab_multiturn` | 768 | 768 | 0 | 0 | 无 |
| `multi_holder_pressure` | 767 | 767 | 0 | 0 | `no_cache_candidate=1` |

### 其他策略

- P2：`tracelab_multiturn` 的 768 次决策均为 `primary`；`multi_holder_pressure` 有
  750 次 `primary` 与 18 次 `admission_backup`。
- Cache-Aware ZMQ：该 policy 没有发出 Router policy reason metric，因此报告保留
  `not_emitted_by_policy`，不将它补写为推测 reason。
- Shortest-TTFT：主轮中 `tracelab_multiturn` 为 768 次 `shortest_ttft_cache`；
  `multi_holder_pressure` 为 765 次 `shortest_ttft_cache`、3 次
  `shortest_ttft_p2_fallback`。全部请求成功完成。

## 5. Shortest-TTFT 独立确认轮

Shortest-TTFT 主轮的 TTFT/E2E p95 RSD 超过 10%，因此以相同产品二进制和运行合同另跑
2 repeats（2 workload × 2 = 4 cases）。确认轮独立保存，主三轮不被替换。五次样本合并仅用于
判断波动，不改变主矩阵表中的三轮中位数。

| Workload | 主轮 TTFT/E2E p95 RSD | 确认轮 TTFT/E2E p95 RSD | 五次合并 TTFT/E2E p95 RSD |
| --- | ---: | ---: | ---: |
| `tracelab_multiturn` | 16.36% / 10.38% | 70.13% / 39.96% | 56.39% / 28.31% |
| `multi_holder_pressure` | 14.56% / 12.64% | 8.11% / 2.30% | 18.59% / 16.55% |

确认轮表明 Shortest-TTFT 的波动不是由主轮一次偶发异常造成。特别是
`tracelab_multiturn` 的两次确认波动较大；后续若继续推进该策略，应先锁定预测模型、并发和
Decode 压力信号，再判断其得分函数本身的收益。

## 6. 验证与制品

- 主矩阵：`RUN_COMPLETE=ok`、24/24 `COMPLETE`、零非空 `request_errors`、零
  `policy_selection_failed`。
- 确认轮：`RUN_COMPLETE=ok`、4/4 `COMPLETE`、零非空 `request_errors`、零
  `policy_selection_failed`。
- 主轮 analyzer 与确认轮 analyzer 都通过；本地合并 analyzer 保留每组的
  `primary_repeat_count`、`confirmation_repeat_count` 和总 `repeat_count`。
- 主轮和确认轮通过 `rsync --checksum --dry-run` 与远端逐文件核对，无差异。
- 本地原始制品：
  `/Users/gaobo/Documents/mooncake/.vin_stage/router-simulator-http-256-four-policy-unknown-fallback-20260827/`。

本报告与真实 8×L20 E2E 报告互补：前者验证真实 GPU 端到端体验，本文验证 256 endpoint
下 Router HTTP / Indexer / LoadMonitor 控制面的策略行为与可观测性。
