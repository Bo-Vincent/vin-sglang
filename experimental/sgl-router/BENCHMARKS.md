# sgl-router microbench harness + SMG comparison

This file pairs with `experimental/sgl-router/benches/` and the SMG
Criterion harnesses at:

- `~/smg_workspace/smg/model_gateway/benches/radix_tree_benchmark.rs`
- `~/smg_workspace/smg/model_gateway/benches/manual_policy_benchmark.rs`
- `~/smg_workspace/smg/model_gateway/benches/router_registry_bench.rs`
- `sgl-model-gateway/benches/*` (in-tree mirror of SMG, same code)

## Scope

These are CPU-bound microbenches that don't need GPUs — they target
routing-decision latency only. The full E2E throughput comparison
(genai-bench at 4×H200 against a real SGLang fleet) is **not** part of
this file; it requires a real GPU cluster and is tracked separately.

## How to run

sgl-router:
```bash
cd experimental/sgl-router
cargo bench --bench tree_lookup     -- --sample-size 30 --measurement-time 3
cargo bench --bench policy_select   -- --sample-size 30 --measurement-time 3
```

SMG (the gateway being deprecated):
```bash
cd ~/smg_workspace/smg/model_gateway
cargo bench --bench radix_tree_benchmark -- --sample-size 30 --measurement-time 3 \
    'token_match_10w_4096tok|token_insert_10w_4096tok'
cargo bench --bench manual_policy_benchmark
```

For the quick runs whose numbers are reproduced below: drop
`--sample-size` to 10 and `--measurement-time` to 2 (Criterion will
warn about reduced statistical confidence but the order-of-magnitude
comparison stands).

## Quick-run Data Points (M1 MacBook, release profile)

These are NOT the real acceptance numbers — they're a sanity check
that the sgl-router routing primitives are in the same ballpark as the
SMG ones they replace. Real targets come from the cluster-scale
comparison and are tracked separately.

### Cache-aware lookup (`HashTree` vs SMG `TokenTree`)

| Bench | sgl-router | SMG TokenTree | Notes |
|---|---|---|---|
| Insert 64 blocks for 1 worker (medium case) | `hashtree_insert/128` ≈ 21.5 µs | `token_insert_10w_4096tok` ≈ 1.05 µs | Numbers not directly comparable — SMG counts per-token insert, sgl-router counts per-block insert. SMG inserts 4096 tokens at a fixed `block_size`; sgl-router inserts 128 pre-hashed `i64` block-hashes. The hashing step (`compute_block_hashes`) is upstream of `HashTree` and not measured here. |
| Match request prefix | `hashtree_match_prefix/w64_bpw128_q64` ≈ 47 ns | `token_match_10w_4096tok` ≈ 1.24 µs | sgl-router's match is a short-circuit walk over `i64` hashes; SMG's match tokenizes + hashes per-call. The fair comparison includes `compute_block_hashes` cost (~ tens of µs depending on prompt length). |

**Read carefully.** The 26× difference at the match step is not the
end-to-end speedup an operator should expect — `compute_block_hashes`
upstream dominates in real traffic. The number proves that sgl-router's
tree walk is no slower than SMG's, which is what the `routing-decision
latency p50 ≤ 1.10× SMG` acceptance criterion targets.

### Policy selection (non-cache-aware)

| Policy | n=4 workers | n=16 | n=64 | n=256 | SMG equivalent |
|---|---|---|---|---|---|
| `round_robin`     | 2.5 ns | 2.5 ns | 2.5 ns | 2.5 ns | SMG round-robin is O(1) — same shape. |
| `random`          | — | — | — | — | `SliceRandom::choose` call — O(1), matching SMG's O(1) `rand::random()` call. |
| `power_of_two`    | — | — | — | — | Two distinct indices sampled directly - O(1), matching SMG's shape (2× rand + 2× load read). |

Both `random` and `power_of_two` are now O(1), ensuring consistent
performance regardless of worker count.
TODO: Add a regression guard for the O(n) shape. Although `policy_select`
measures the metric, nothing currently runs or gates on these results.

## Router v2 policy / PD E2E

正式 108-case Policy 对照、RSD 确认轮、真实 LoadMonitor Pressure Guard、Indexer
Cache candidate 和单机 4P+4D Mooncake KV transfer 的环境、结果与边界见
[Router v2 Policy / PD E2E 报告](docs/router-v2-e2e-poc-report.md)。该验证证明 Step 1
Policy/Admission/Guard 数据流可用；静态 Bucket 仍是 Step 2 实验接口，不代表同构
4+4 切分或异构生产 Bucket 已达到 production GO。

## Step 3 LoadMonitor A/B

Step 2 baseline 与 Step 3 多维 LoadMonitor candidate 的 48-case 正式主轮、12-case RSD
确认轮、真实 4P+4D snapshot proof、性能结果和适用边界见
[Step 3 LoadMonitor A/B 报告](docs/router-policy-step3-monitor-ab-report.md)。该结果证明新指标
链路可用且多数场景中性或改善，但高压 Session workload 仍有高波动 TTFT 回退，不能外推为
所有 workload 的稳定收益。

## Router Policy 热路径 Microbenchmark

8 / 64 / 256 endpoints 下 scheduling snapshot、P2、Session-Aware、Cache-Aware top-K
和 Decode Policy 的三轮 Router-only latency、allocation、CPU、扩展性门槛与适用边界见
[Router Policy 热路径 Microbenchmark 报告](docs/router-policy-pressure-microbenchmark-report.md)。
正式 V6 共 211 项验收并全部通过；该结果不替代高 Router RPS 和并发 LoadMonitor 写入
条件下的容量验证。

## Pre-deprecation calibration runbook

Before deleting SMG, every routing-latency metric in the slim-design
spec needs a real-cluster measurement. The bench-harness here is the
small-scale, CPU-only complement; it catches algorithmic regressions in
the routing primitives without burning GPU time. Pair both: this file
in pre-commit / CI tier-2, the real-cluster e2e in the
`pr-test-rust.yml` matrix entry.

## Simulator large-fleet policy evaluation

临时引入 SGLang Simulator #33824 后，使用真实 Rust P2 / Cache-Aware 决策路径和
virtual worker queue/cache 闭环完成的 8 / 64 / 256 worker CPU logical-time 矩阵、原始
结果和适用边界见
[Router Policy 大规模 Virtual Fleet 模拟报告](docs/router-policy-simulator-large-fleet-report.md)。
该实验验证策略行为与规模边界，不替代真实 GPU、HTTP、Indexer RPC 或 LoadMonitor stream
的生产吞吐验证。

## Router Policy 真实 LoadMonitor E2E

真实 8×L20 worker reporter、Router outbound LoadMonitor session、Indexer Cache-Aware
preflight 与 Session-Aware Pressure Guard 的功能性闭环证据见
[Router Policy 真实 LoadMonitor E2E 验证](docs/router-policy-real-load-monitor-e2e-report.md)。
该验证证明真实 snapshot 能改变最终 worker，不替代正式性能矩阵，也不覆盖 L20 对当前
`sglang-kernel` 新版本的 ABI 兼容性。
