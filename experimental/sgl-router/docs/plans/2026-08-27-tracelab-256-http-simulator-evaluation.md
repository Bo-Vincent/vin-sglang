# TraceLab 256-worker HTTP Simulator 评估设计

## 目标

在 256 个 SGLang CPU Simulator HTTP worker 上，使用公开 TraceLab v0.0.2 的
`codex` agent-coding session 几何，比较 `power_of_two`、Native `cache_aware`、
`cache_aware_zmq` 与 `shortest_ttft` 的相对 TTFT、E2E、completion TPS、实际 KV hit
和 worker CV。

## 数据合同

- Trace：`syfi_coding_trace.jsonl.gz`，SHA-256
  `11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4fb65`。
- 筛选：`provider=codex`、连续 4 turn、输入 1,024–16,384 tokens、measurement
  prefix 至少 1,024 tokens、append 至多 4,096 tokens。
- 当前数据中有 327 个合格 session；正式 slice 固定选 256 个，按固定 seed 排序。
- 每个 session 第 1 turn 经 Router 自然执行为 warmup；后 3 turn 为 measurement。
  每 policy / repeat 有 256 warmup 和 768 measurement。
- 公开 TraceLab 没有 prompt 文本。replay 只保留 session、turn 顺序和 token 几何，并
  用 session-local、可逆的虚拟 token 链重建 prompt；它不伪造跨 session 公共 prefix。
- Simulator 和 Router 使用远端已有的完整 `Qwen2.5-7B-Instruct` tokenizer。旧的
  6-token fixture 会把真实 prompt 压成 `[UNK]`，不能用于此合同。

## 架构

```text
TraceLab JSONL
  -> 256-session replay manifest
  -> virtual prompt reconstruction
  -> 256 Simulator HTTP workers
  -> Router (P2 / Native Cache-Aware / ZMQ / Shortest-TTFT)
  -> worker cache metrics + Router metrics + Indexer / LoadMonitor audit
  -> analyzer / report
```

新增的 Simulator TraceLab runner 不复用原 GPU TraceLab runner 的 worker lifecycle：
后者固定为 8 GPU endpoint。它复用 replay adapter 的 session 串行约束，在每个 case
中执行以下步骤：

1. 启动或复用 256 个 CPU Simulator worker；每个 case flush cache。
2. 仅对 Native Cache-Aware / Shortest-TTFT 启动 Indexer 与 256 个 bridge。
3. 通过 Router 重放 256 个 warmup turn，等待 KV event / Indexer 收敛。
4. 记录 measurement 前 worker/router 快照；重放每个 session 后 3 个 turn，session 内
   串行、session 间并发，全球请求率固定为 64 QPS，输出长度沿用 TraceLab round。
5. 记录 measurement 后快照，生成请求、cache、worker、reason 和 Native audit 结果。

## 正式矩阵与验收

```text
256 workers
× 4 policies
× 3 repeats
= 12 cases
```

每个 group 输出中位数和 RSD：TTFT mean/p95、E2E mean/p95、completion TPS、KV hit
rate、worker CV、policy reasons。Native Cache-Aware 必须满足：每个 cache candidate
有真实 cache metric 和 LoadMonitor 决策，且 `router_local=0`、`zero_snapshot=0`。
所有 case 必须有 `RUN_COMPLETE`、`COMPLETE`、零请求错误和零 fatal/OOM。

若 TTFT p95、E2E p95、TPS 或 KV hit 的三轮 RSD 超过 10%，在独立目录以相同二进制和
合同补两轮，再由 analyzer 保留主三轮并合并展示确认样本。

## 实现与测试

1. 从已验证的 TraceLab adapter 引入 session slice、virtual token plan 和 prompt
   reconstruction；新增测试覆盖 256-session 的 deterministic selection。
2. 新增 Simulator TraceLab runner；测试 warmup/measurement 分离、session 串行、无
   人工 fixed-prefix prewarm、metric snapshot 和 12-case dry-run。
3. 新增对应 analyzer；测试 3-repeat 聚合、零错误 gate、Native audit 与 confirmation
   合并。
4. 先执行全部 Python 测试和 256-session dry-run，再在 `h20-8-usa` 执行正式 E2E。

## 边界

这是 `simulator_predicted_relative` 实验：它验证 256 endpoint 下 Router、Indexer、
LoadMonitor 和策略的相对行为；不将 CPU Simulator 的 TTFT/E2E/TPS 写成真实 GPU
性能结论。此前人工长 prefix 的 256-worker 结果保留为独立控制面压力测试，不与本合同
合并。
