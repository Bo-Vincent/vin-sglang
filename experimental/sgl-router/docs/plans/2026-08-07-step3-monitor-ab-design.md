# Router Step 3 Monitor A/B 设计

日期：2026-08-07
状态：冻结
范围：`experimental/sgl-router`

## 目标

完成两件事：

1. 把 Router Policy 的 Step 1、Step 2、Step 3 保持为可审计的代码与提交边界；
2. 在同一台 8 卡 H20 上，用数小时内结束的配对实验判断 Step 3 Monitor 是否改善
   RT、TTFT、ITL、吞吐、KV hit rate 和 worker balance。

实验允许得到“改善”“中性”或“退化”结论，不以证明收益为前提。

## 代码边界

### Step 1：同构 Policy、Admission 和 Guard

保持现有五个提交：

```text
dce5c26702  Policy Proposal contracts
41819b0e98  shared Admission and Guard
ea1bcddf16  bounded Session-Aware
bbd9d4d816  bounded Cache-Aware and Indexer
130789bc44  Decode Policy and Step 1 integration
```

Step 1 不包含 Bucket，也不包含 Step 3 的新 LoadMonitor 指标。

### Step 2：Bucket 和 SLO

保持现有提交：

```text
7701026663  Prefill/Decode Bucket, SLO and ordered fallback
```

Step 2 只改变候选域和 fallback，不拥有 Policy、Admission 或 Guard。

### Step 3：Monitor 到 Policy/Guard

整理为三个语义提交：

```text
de70a84d2c  feat(load-reporter): expose Step 3 pressure metrics
7cf2ce8453  feat(sgl-router): consume Step 3 load metrics in policy selection
本 docs 提交  docs(sgl-router): document and evaluate Step 3 monitor integration
```

第一项负责 Scheduler 原始 gauge/counter 到 protobuf；第二项负责 Router 派生、降级和
Policy/Guard 消费；第三项负责设计、配置、验证和 A/B 结果。Step 1/2 提交不吸收 Step 3
逻辑。

## A/B 隔离方式

正式实验冻结两个 Router 构建：

```text
A / baseline  = 7701026663（rebase 后的 Step 2 tip）
B / candidate = e211fc49b7（正式 A/B 使用的完整 Step 3 代码树）
```

正式 A/B 完成后只补充了 CLI 参数校验与实验报告，未改变实验使用的
Policy/Monitor 性能路径；因此这里保留实际被测 candidate 哈希，而不是最终文档提交哈希。

Engine、模型、Indexer 和 Reporter 始终使用 Step 3 source。A 的旧 protobuf 会忽略新增
optional 字段，B 会派生并消费它们。每个 case 只重启 Router，不重启 4P+4D Engine，避免
模型加载、GPU 温度和 Engine cache 重建污染对照。

开始实验前必须验证：

- A 能接收新 Reporter 的报告且请求成功；
- A/B 除 Step 3 Router diff 外，构建参数、Engine、数据集和 Router 参数相同；
- A 不识别或消费 Step 3 字段，B 能观测到非零连续 counter 和 snapshot version。

## 短时矩阵

使用四个能够映射到历史 108-case 数据的 panel：

| Panel | Workload | Prefill Policy | 主要观察点 |
|---|---|---|---|
| Load | `random` | `power_of_two` | queue-time load selection |
| Session | `session_hotspot` | `session_aware` soft | affinity 与 Pressure Guard |
| Cache | `hot_prefix` | `cache_aware` | 热点 cache 与负载竞争 |
| Cache | `multiturn` | `cache_aware` | Agent-like prefix reuse |

固定维度：

```text
variants      = A, B
request_rates = 8, 12
repeats       = 3
cases         = 2 * 4 * 2 * 3 = 48
```

同一 `(panel, workload, rate, repeat)` 内配对运行 A/B；偶数 repeat 先 A，奇数 repeat 先 B，
减少时间漂移偏差。预计正式矩阵 3–5 小时。只有会改变结论且三轮主指标 RSD > 10% 的组才补
确认轮，最多 12 个 case，保证整晚可以结束。

Step 2 Bucket 在该矩阵中不启用。实验目标是隔离 Step 3，而不是重复 Step 2 Bucket 验证。

## 指标与判断

每个 case 保存：

- RT、TTFT、ITL 的 p50/p95/p99；
- request/input/output throughput；
- KV hit rate；
- worker request CV；
- Policy decision reason；
- LoadMonitor snapshot、原始 counter 增量和报告错误；
- request error、fatal、OOM 和 GPU 健康。

主结论按同一配对组的 B 相对 A 计算。历史 108-case 的匹配 workload 只作为跨时间辅助参照，
不能替代当晚 A/B。

验收规则：

1. 48/48 case 完成，三轮齐全，请求错误、fatal、OOM 和 LoadMonitor error 为 0；
2. B 有非零 Step 3 counter 和 fresh snapshot，A/B reason 均符合对应 Policy；
3. 报告每个 workload 的性能 delta 和 RSD，不只汇总平均值；
4. 若 B 在主要负载组改善 TTFT/RT 或 balance 且吞吐无显著回退，可判定有收益；
5. 若变化小于噪声或只在单轮出现，结论为中性；
6. 任一稳定的吞吐或 tail latency 退化必须原样报告。

## 边界

本实验不证明：

- 异构 Bucket 或 SLO capacity；
- 跨机 RDMA/RoCE；
- 长时间 soak 和生产容量；
- Transfer-Aware Decode 或 Reservation。

它只回答：在同构单机 4P+4D、相同 Policy 和 workload 下，Step 3 的实时负载指标是否比
Step 2 的 waiting/running/token fallback 产生更好的 Router 决策。
