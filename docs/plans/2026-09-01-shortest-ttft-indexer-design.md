# Shortest-TTFT Indexer 设计

## 目标

在 `vin/shortest-ttft-indexer` 上保留已验证的独立 Shortest-TTFT
算法和 LoadStat monitor，只将缓存命中来源替换为 V4 外部 KV Indexer。
本地 `HashTree` 不再参与该分支运行时的 Shortest-TTFT 缓存评分。

## 已确认的 V4 合同

V4 ingress 已能把 `GrpcPrefixIndex::match_prefix` 的异步结果转换成
`ExternalPrefixSignal`，并随 `SelectionContext` 交给同步 policy。
该 signal 携带每个 engine address 的 `matched_prefix_blocks` 与查询的
总 block 数。Indexer 的不可达、过载、超时和过大查询会成为 `Empty`；
拒绝型协议错误仍拒绝请求。这一既有语义保持不变。

## 方案

1. 新增仅供 `--policy shortest_ttft` 使用的独立 Indexer 配置与 CLI 参数：
   `--shortest-ttft-indexer-endpoint`、`--shortest-ttft-indexer-query-timeout-ms`
   和 `--shortest-ttft-indexer-query-max-inflight`。
2. `main` 根据当前 policy 选择 Indexer 配置。Shortest-TTFT 使用现有
   `GrpcPrefixIndex` 与既有 ingress query，不复用 `CacheAwareConfig`、
   cache 阈值或 admission 逻辑。
3. Shortest-TTFT 收到 `ExternalPrefixSignal` 时，只由 signal 中与当前
   候选 URL 对应的最大 `matched_prefix_blocks` 计算 `hit_tokens`；信号
   为空、无匹配或 block size 未知时所有候选均为零命中，绝不回读本地
   `HashTree`。
4. 外部 signal 不存在时保留 policy 的本地-tree 路径，以维持直接单元调用
   和未配置 Indexer 的测试兼容性；该新分支的生产 indexer 配置则保证
   ingress 总会提供一个 authoritative signal。
5. LoadStat monitor、`prefill + queue pressure` 评分、前 30% 候选窗和
   最近未调度公平性保持不变。

## 验收

- `--policy shortest_ttft` 只接受新的 shortest-TTFT Indexer 参数；
  cache-aware 的 Indexer 参数仍只属于 `cache_aware_zmq`。
- 外部命中覆盖本地 tree 的矛盾内容；无匹配不会回退本地 tree。
- indexer 异常沿用 V4 的 Empty / rejected 语义，TTFT 队列评分保持可用。
- 全量 Rust 测试、格式检查和独立性扫描通过，再从 `vin` 推送分支。
