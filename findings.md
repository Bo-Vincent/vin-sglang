# 调研记录

## RTP-LLM 参考实现

- 源码：`RTP-LLM/github-opensource/rtp_llm/flexlb/flexlb-sync/.../ShortestTTFTStrategy.java`。
- 本地参考版本：`2aff4342cbb94544fd015d1b21d3a6e8e7b4d728`。
- 单 worker 估算：`TaskInfo.estimatePrefillTimeMs(seqLen, hitCacheTokens) + runningQueueTime`。
- 命中 token = `block_size * prefix_match_blocks`。
- 先按 TTFT 与最近调度时间排序，取前 30%（至少 1 个）；在与最小 TTFT 足够接近的候选中选择最久未被调度者。

## V4 基线

- `20a491d1d3` 已提供 KV-event `HashTree`、block-size oracle、请求 token 与 #34608 的 Python `LoadStat` 发布器。
- 该基线没有独立的 Rust load consumer；现有 `cache_aware_zmq` 不应作为本策略依赖。
- 因此实现应以独立模块接收 #34608 load socket，并为策略提供只读快照。

## #34608 load 合同

- Python publisher 的帧为 `[b"load", 8 字节大端 sequence, msgpack LoadStat]`。
- `LoadStat` 的数组负载是 `["LoadStat", num_running_reqs, num_waiting_reqs, num_tokens, max_total_num_tokens, attn_dp_rank]`。
- `/server_info` 在 `kv_events` 内发布 `load_endpoint_port_base` 与 `load_topic`；DP rank 使用 `port_base + rank`，端点 host 必须按 worker 的可连接地址解析。
- Load 是瞬时 gauge：publisher 重启后 sequence 可以回退，不能套用 KV event 的严格重放去重语义；每个 DP rank 过期都会使该 worker 的 engine-load 快照不可用。

## 已验证的本地行为

- `engine_queue_can_outweigh_a_better_prefix_match` 已通过：全前缀命中但 engine queue pressure 为 9 的 worker，不会压过无命中且空闲的 worker。
- 当前实现的 RTP 近似为 `floor((10 * input_tokens - 7 * hit_tokens) / 10) + running + waiting`；#34608 未提供毫秒级 queue-time，因此第二项是可获得的运行/等待请求压力，而非伪造的毫秒数据。
- `EngineLoadMonitor` 使用独立的 ZMQ SUB task，不复用 KV-event 的 cursor/replay 状态；worker 删除先 cancel/join 再清空 table，因此迟到消息不能恢复已删除 worker 的 load。
- Factory 使用 `build_registry_with_engine_load` 显式接收 table；旧 `build_registry` 只保留给不创建 monitor 的兼容调用点，避免两个组件意外拥有不同状态。
- `ServerInfo` 从 `kv_events.load_endpoint_port_base` / `load_topic` 生成 `LoadEndpointConfig`，并与 KV publisher 复用唯一的 wildcard-host 到 worker URL host 的解析规则。
- `main` 只在 `--policy shortest_ttft` 时创建 `EngineLoadMonitor`；它与 policy 共用同一 `EngineLoadTable`，进程退出时会 cancel/join 全部独立 subscriber。
- 组件测试还覆盖 DP rank 缺失/过期、publisher sequence 重置和 worker 移除后的迟到 gauge；这三类情况不会产生错误的远端 queue 聚合。

## Shortest-TTFT Indexer 变体

- V4 ingress 的 `chat.rs` 在配置 `GrpcPrefixIndex`、具备 request token 和 block
  size 时调用 `match_prefix`，并将 `PrefixOutcome` 封装为
  `SelectionContext::external_prefix`。
- `PrefixOutcome::Matched` 以每个 engine address 的
  `matched_prefix_blocks` 表示命中；同一 address 的多个匹配必须取最大值。
- `Overloaded`、`Timeout`、`Unreachable` 与 `QueryTooLarge` 已由 ingress 转为
  `Empty`，而 `Rejected` 仍返回请求错误。Shortest-TTFT 只消费该统一 signal，
  不应复制或改变其异步错误语义。
- 既有 `--kv-indexer-*` 配置存放于 `CacheAwareConfig`；Shortest-TTFT 需要
  独立配置入口，避免从 cache-aware/admission 框架取得状态或阈值。
