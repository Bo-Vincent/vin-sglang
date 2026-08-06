# Router Policy Step 3 LoadMonitor A/B 报告

日期：2026-08-07
结论：功能与兼容性 GO；性能结果为条件性正向，不支持“所有场景均提升”的结论。

## 1. 实验问题

本实验比较同一个 Step 2 Router 在接入 Step 3 LoadMonitor 指标前后的差异：

- baseline：`7701026663`，只使用 Step 2 已有负载信息；
- candidate：`e211fc49b7`，增加 Prefill queue-time 与 Decode queue/step/KV pressure；
- 两组共用 Step 3 Engine、Reporter、模型、Indexer 和 workload；
- Bucket 关闭，避免把 Bucket 候选域变化混入指标效果；
- 单机 8×H20，4 Prefill + 4 Decode，Mooncake TCP，Qwen2.5-7B-Instruct。

正式主轮包含 4 个 workload、QPS 8/12、两个 variant、三轮，共 48 cases。
baseline/candidate 按 repeat 交错执行。对主指标 RSD 超过 10% 的 scope，按波动从高到低
选择三组，并为两个 variant 成对补 r3/r4，共 12 个确认 cases；原三轮不被替换。

## 2. 验收结果

主轮 48/48、确认轮 12/12 均完成。以下检查全部通过：

- 三轮主数据完整，确认轮 repeat IDs 为 3/4；
- 请求错误、LoadMonitor 错误、fatal 和 OOM 均为 0；
- prompt token、KV hit-rate 和 policy reason 完整；
- Step 3 protobuf 字段完整，24 个 candidate records 的四组新 counter delta 均为正；
- 独立 proof 中 baseline/candidate 的 Router 决策均读到非零 LoadMonitor snapshot，最大
  version 分别为 13 和 11。

candidate 主轮累计观测到：

```text
Prefill uncached tokens = 13,493,230
Prefill busy time      = 6,341,987,612 us
Decode steps           = 300,879
Decode step time       = 6,858,024,297 us
```

这证明新字段不只是 Reporter 侧存在，而是经过 LoadMonitor 进入了真实 Router 决策视图。

## 3. 性能结果

正值表示 candidate 相比 baseline 更好。补确认轮的三个 scope 使用五轮合并结果，其余使用
正式主轮三轮结果。

| 场景 | QPS | RT p95 | TTFT p95 | ITL p95 | 请求吞吐 | KV hit | Worker CV |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cache hot-prefix | 8 | +0.30% | -0.65% | +0.25% | -0.03% | +0.00% | +30.84% |
| Cache hot-prefix | 12 | +0.71% | +0.61% | +0.77% | +0.11% | +0.34% | -0.34% |
| Cache multiturn | 8 | +4.19% | +3.30% | +6.09% | +1.00% | +0.02% | -17.94% |
| Cache multiturn | 12 | -0.55% | +2.80% | +3.48% | +3.26% | +0.75% | -19.89% |
| P2 random load | 8 | +0.33% | +2.61% | +0.14% | -0.02% | +5.31% | +13.59% |
| P2 random load | 12 | +0.04% | +0.59% | +0.51% | -0.00% | -2.95% | -9.32% |
| Session hotspot | 8 | +10.38% | +32.29% | +6.79% | +3.30% | +0.05% | +20.49% |
| Session hotspot | 12 | +0.17% | -9.70% | +4.14% | -1.10% | +0.03% | -43.72% |

结果可以分为三类：

1. P2 random 与 Cache hot-prefix 基本中性，主要指标变化多在 1% 内，说明新增指标没有在
   普通负载和强 cache affinity 场景引入明显开销。
2. Cache multiturn QPS 8 与 Session QPS 8 的 RT、TTFT、ITL 同时改善；这是更细压力信号
   最有价值的场景。
3. Session QPS 12 的 ITL 改善，但 TTFT 回退 9.70%、吞吐回退 1.10%，且 candidate TTFT
   p95 的五轮 RSD 为 43.5%。这个结果不能用于宣称稳定回归，也不能被忽略为已解决。

## 4. 波动与历史数据边界

确认轮后，Cache multiturn QPS 8 和 Session QPS 8 的 candidate 确认轮 TTFT RSD 已低于
10%，但它们的 baseline 或五轮合并值仍有明显波动。Session QPS 12 的 baseline/candidate
均未稳定。Cache multiturn QPS 12 的 baseline TTFT RSD 为 11.4%，因为只补最高三组，未再
追加确认轮。因此 affinity workload 的尾延迟结果应视为方向性证据，不是精确收益承诺。

历史 Router v2 数据只作二级参照。当前 baseline 相比历史数据的 TTFT、RT 和吞吐存在明显
环境/trace 漂移，部分 TTFT 差异达到数倍；它不能建立 Step 3 的因果关系。正式结论只采用
同一轮内、相同 Engine/Reporter/workload 的成对 A/B。

## 5. 结论与后续

- 接口、兼容、降级、真实指标采集和 Router 消费链路：GO。
- 普通 P2 与稳定 cache-hit 场景的额外负担：未观察到明显回归。
- 新指标带来的性能收益：在 QPS 8 affinity workload 上成立，在 QPS 12 Session 场景不稳定。
- 不应把本结果描述为“Step 3 全面提升吞吐和尾延迟”。生产启用前应针对高压 Session workload
  校准 queue-time/Pressure Guard 阈值，并在更长 steady-state trace 上复测。
- 本实验不覆盖异构硬件、Bucket/SLO、RDMA、跨机网络或生产容量，不改变 Step 2 Bucket 的
  独立验证结论。

## 6. 结果位置

远端原始结果：

```text
/root/router-policy-bench/results/step3-monitor-ab-48-v2-20260807
/root/router-policy-bench/results/step3-monitor-ab-proof-v5-20260807
/root/router-policy-bench/results/step3-monitor-ab-confirmation-12-20260807
```

正式合并分析：

```text
/root/router-policy-bench/results/step3-monitor-ab-48-v2-20260807/analysis-formal-with-confirmation
```

本地副本统一保存到仓库外：

```text
/Users/gaobo/Documents/mooncake/.vin_stage/router-step3-monitor-ab-results-20260807
```
