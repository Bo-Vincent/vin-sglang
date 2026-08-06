# Router v2 Policy / PD E2E 验证报告

日期：2026-08-05

分支：`vin/router-v2`

GPU 验证源码：`ffca5ece7012b56a9acc7c252dc2cc6d6b4e1ce8`。报告回填只修改
文档；Router 实现和已验证 release binary 未变化，binary SHA256 为
`331fba37d4f20683e059b15e87e3c36da9efdd5214aa5d988213a3ca44a785fd`。

范围：Step 1 的 P/D Policy、Admission、Guard、Session/Cache-Aware，以及 Step 2
静态 Bucket 接口；包含同构性能矩阵和单机真实 4P+4D PD 数据路径。

## 结论

- **接口、正确性和默认兼容性 GO。** 最终实现的全量 Rust 测试、Clippy、格式检查和
  Python harness 合同均通过；现有策略默认行为未被替换，新策略为显式选择。
- **正式同构 E2E GO。** 108/108 个主 case、14/14 个 RSD 确认 case 成功；请求错误、
  LoadMonitor 错误、fatal 和 OOM 均为零，9 项 analyzer acceptance 全部通过。
- **新的 Cache-Aware 能消费真实 Indexer Top-K。** 它逐候选计算 matched/uncached
  work、执行 Admission 和有界 tournament，直接产生一个 winner；没有可准入 cache
  candidate 时才降级到 P2。
- **Session/Cache 与负载 co-design 的方向有效，但新策略不是旧 affinity 策略的无条件
  性能替代。** 相比 P2，cache multiturn 和 session workload 明显改善；相比
  `cache_aware_zmq`，新 Cache-Aware 在 multiturn 仍有差距，需要后续基于真实 trace
  调参。当前策略为 opt-in，因此该差距不改变本 PR 的正确性结论。
- **真实单机 PD 路径 GO。** 4 Prefill + 4 Decode、Indexer、LoadMonitor、P 侧
  Cache-Aware、D 侧 P2 和 Mooncake TCP KV transfer 全部贯通。
- **静态同构 Bucket 仍是实验能力。** 本结果不把静态 4+4 切分解释为生产 GO；动态
  Registry Bucket Index、SLO profile、ordered overflow 和异构集群仍属后续增量。

## 环境与合同

- 硬件：8 x NVIDIA L20 46 GiB；模型：本地 `Qwen2.5-7B-Instruct`；每卡一个
  endpoint，`mem_fraction_static=0.85`。
- 主矩阵：QPS 8/12，三轮；Cache panel 为 4 workloads x 3 policies，Session panel
  为 2 workloads x 3 policies，共 108 cases。
- Cache 对照：`power_of_two`、`cache_aware_zmq`、新 `cache_aware`；Session 对照：
  `power_of_two`、`sticky`、新 `session_aware`。
- 新 Cache-Aware 参数：Indexer timeout 25 ms，`min_matched_tokens=1024`，候选数
  `max(8, ceil(5% x P workers))` 且上限 32，work margin 1,024 tokens。
- 原始结果未提交到仓库，通过 `rsync` 保存于工作区外的本地验证目录；目录名保留
  commit 前缀和实验类型，便于与报告中的源码版本和合同逐项核对。

## 正式 E2E 结果

### Cache-Aware 对 P2

以下均为三轮均值的相对变化；延迟负数表示更低。

| Workload | QPS | 请求吞吐 | P95 RT | P95 TTFT | P95 ITL | KV hit-rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hot_prefix | 8 | +0.06% | -13.42% | -3.27% | +0.80% | +26.98% |
| hot_prefix | 12 | +0.06% | -14.69% | -10.54% | -1.29% | +24.70% |
| shared_prefix | 8 | +0.03% | -19.29% | -30.66% | +1.30% | +66.31% |
| shared_prefix | 12 | +0.20% | -26.37% | -40.07% | -15.75% | +64.29% |
| multiturn | 8 | +34.06% | -32.92% | -60.64% | -24.70% | +186.90% |
| multiturn | 12 | +33.92% | -31.48% | -47.72% | -68.07% | +171.55% |

Random workload 的吞吐和 P95 RT/ITL 基本持平，未观察到无有效 affinity 时的明显
退化。相较 `cache_aware_zmq`，hot/shared prefix 的吞吐基本持平，但 P95 RT/TTFT
约高 1%–6%；multiturn 的吞吐低 9.09%–17.41%、P95 RT 高 17.09%–26.76%。因此
第一阶段结论是“新架构可用且优于无 affinity 的 P2”，而不是“已取代 ZMQ 策略”。

### Session-Aware 对 P2

| Workload | QPS | 请求吞吐 | P95 RT | P95 TTFT | P95 ITL | KV hit-rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| session_hotspot | 8 | +15.76% | -32.33% | -59.21% | -37.41% | +59.11% |
| session_hotspot | 12 | +20.51% | -31.45% | -59.40% | -48.78% | +61.22% |
| session_interleaved | 8 | +43.50% | -45.10% | -73.15% | -23.45% | +220.94% |
| session_interleaved | 12 | +56.89% | -40.38% | -71.77% | -46.26% | +231.87% |

与 strict `sticky` 相比，新 Session-Aware 会为 Admission/Pressure 留出逃逸空间，
因此不能期望每个 trace 都获得更高 cache locality；它提供的是 bounded affinity，而不是
替换 strict sticky 的同义实现。

### 稳定性与 acceptance

- 主轮 108 records、36 groups、24 comparisons；所有 policy reason、prompt tokens、
  KV hit-rate 和三轮合同均有效。
- 7 个主轮 group 的某项主指标 RSD 超过 10%，原三轮全部保留，并在独立目录补 r3/r4
  共 14 cases。确认轮本身 14/14、零错误且 acceptance PASS。
- 新 `cache_aware` / `session_aware` 的相关高 RSD group 在确认轮稳定；仍有 3 个 P2
  baseline 的 P95 ITL group 抖动，作为环境噪声保留，不删除或挑选数据。

## 定向真实路径

### LoadMonitor Pressure Guard

Admission-safe 压力探针把 session primary 的 waiting queue 提升到 3，同时保留
`max_running_requests=8` 的硬准入余量。Router 观察到 fresh snapshot 59，决策计数为：

```text
session_admission_backup = 0
session_pressure_backup  = 1
```

最终请求从原 session primary `31000` 逃逸到低压力 `31004`。这证明 Pressure Guard
消费真实 LoadMonitor snapshot，而不是由 hard Admission 提前替它做出决定。

### Cross-Bucket Cache candidate

定向请求实测 token 数为 2,245，Indexer 提供 1,280 matched tokens，得到 965
uncached tokens；跨 Bucket cache candidate 通过 target-specific Admission 并以
`cache_candidate` reason 被选中。该检查证明 Bucket 尚未实现完整生产 fallback 时，
Cache-Aware 仍按真实 extend work，而不是仅按原始 input length 判断候选。

### 单机 4P+4D PD

验证拓扑和结果：

```text
4 Prefill (32200..32203)
  -> Cache-Aware + Indexer + LoadMonitor
  -> real Mooncake TCP KV transfer
4 Decode (32300..32303)
  -> Decode Power-of-Two
```

- 首请求产生 `no_cache_candidate=1`，重复请求产生 `cache_candidate=1`；两次 P 请求均
  落到同一 Prefill，重复请求引擎侧报告 6,328 cached tokens。
- 两次请求各产生一个 Decode P2 决策；每次 primary/backup 均不同，response header 与
  Router 最终 D 日志一致。
- 最大 LoadSnapshot version 为 32；KV transfer count 为 2，合计 692.234 MiB。
- 8/8 worker 日志都证明 `MC_FORCE_TCP=1`；bootstrap failure 和 transfer failure 均为 0。
- service `Result=success`、`RUN_COMPLETE=ok`、`fatal_scan={}`；GPU、端口均释放，
  kernel Xid 为零。

Mooncake 使用官方 CUDA<13 wheel `mooncake-transfer-engine==0.3.12.post1`，wheel
SHA256 为 `fe72c75c03b198f19becb2f9c7b8c736fecffce65033336f010a14893c5fd61f`；
它以隔离 overlay 方式注入 benchmark worker，没有修改共享 venv。

## 验证边界

- PD 测试证明同机真实 P/D KV 数据路径，不证明跨机 RDMA/RoCE 带宽、拓扑 tier 或
  transfer-aware D scoring；本轮 D Policy 明确为 P2。
- 正式性能矩阵为同构 L20。异构硬件 Bucket 的离线 SLO profile、rank、capacity
  admission 和 ordered fallback 需要 Step 2 后续实验。
- 当前实现没有 Reservation/预扣；接口允许后续 policy/admission 增量接入，但本 PR
  不把它声明为已实现。
- `estimated_prefill_queue_ms` 只在设计中保留为后续高精度输入；本 PR 没有对应生产
  字段或消费路径，当前比较器使用 `num_waiting_uncached_tokens`、请求数和本地
  active-load，并在 snapshot 缺失或过期时降级。
- 失败的 Pressure/PD harness 运行均按 append-only 保留。对应失败分别来自 hard
  Admission 掩盖 Guard、ANSI 日志解析，不是 Router 请求或 KV transfer 失败；修复后的
  独立结果目录通过原始集成路径复验。
