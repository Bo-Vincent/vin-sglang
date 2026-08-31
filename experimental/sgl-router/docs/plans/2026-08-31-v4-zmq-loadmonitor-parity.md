# V4 ZMQ LoadMonitor 语义对齐

## 目标

在不恢复 gRPC Load Reporter 的前提下，让 V4 的 `cache_aware` 经由
`#34608` 的 ZMQ LoadStat 消费与 V3 相同的 native cache-aware 负载语义：
hard admission、Prefill pressure guard、freshness 与 DP-rank 完整性。

V3 工作树仅作为只读语义基准，不修改、不构建、不提交。

## 已确认的缺口

当前 ZMQ `LoadStat` 只发送 running/waiting/KV used/KV capacity 四项。V4
因此删除了 V3 的 `max_running_requests`、`num_total_tokens`、
`num_waiting_uncached_tokens` 与 queue-time 推导，导致 admission 退化为
KV capacity 判断，Cache-Aware pressure guard 也被移除。这不满足性能比较
的策略完整性要求。

## 设计

1. 保持现有三帧 ZMQ 格式与前四个字段的位置不变；在 `LoadStat` 尾部追加
   V3 RankLoad 对应字段。旧 Router 继续忽略尾部字段，新 Router 对旧短帧
   标为不完整并回退本地负载，不能把不完整监控当作正常 monitor。
2. Python publisher 直接从现有 `LoadSnapshot` 填充扩展字段，不增加新的
   scheduler 采集路径。连续累计 counter 由 Router 按每个 `(worker, dp_rank)`
   的相邻样本计算 Prefill 吞吐与 estimated queue time；reset、缺失或非递增
   counter 不产生派生指标。
3. Rust `EngineLoadTable` 只在所有已声明 DP rank 都 fresh 且语义完整时聚合
   `EngineWorkerLoad`。聚合及 pressure comparison 与 V3 `AggregateLoad` 的
   字段/降级次序保持一致；不完整、缺 rank 或 stale 均全体回退 router-local。
4. 恢复 V3 的 hard admission：running request cap、total token cap、以及
   cache-candidate 的 pending uncached-token cap。恢复 cache near-tie 的
   pressure-guard override 和 Pair proposal 的 guard hints。
5. 添加 wire、聚合、admission 与 guard 的定向测试。运行时验收使用独立的
   多 cache-holder 压力 case，必须观察到 fresh full monitor、admission
   rejection、guard comparison 和 guard override；普通 TraceLab 单候选 case
   只能作为功能/吞吐样本，不能独立证明 guard。

## 验收

- Python LoadStat 单测证明扩展字段从 `LoadSnapshot` 原样发布。
- Rust 单测证明短帧/缺字段 fail closed，完整多 rank 样本产生 V3 等价聚合和
  queue-time 派生值。
- Rust admission 单测证明 request cap、token cap、pending work 与 pressure
  override 均实际改变选择。
- H20-2 完整 256-worker case 的 manifest 锁定二进制 SHA、参数、worker 生命周期
  与 monitor coverage；所有比较候选必须为 fresh/full，且 admission/guard
  计数均为正、请求无错误。
