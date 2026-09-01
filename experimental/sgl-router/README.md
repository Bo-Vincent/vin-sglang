# sgl-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

Serves a single model and routes across its workers. Exposes
`/v1/tokenize`, `/v1/detokenize`, `/v1/models`, `/v1/chat/completions`
(buffered and SSE), plus `/healthz` / `/readyz` and `/metrics`. Worker
pools come from either a static URL list or Kubernetes EndpointSlice
discovery.

## Building

```bash
cd experimental/sgl-router
cargo build --release
```

## Running

The router is configured entirely through CLI flags (run
`sgl-router --help` for the full list). It serves exactly one model, so
`--model-id` is required, along with exactly one discovery backend.
`--tokenizer-path` is optional: give it a local `tokenizer.json` path or a
HuggingFace repo id, and when omitted the router downloads the tokenizer
for `--model-id` from HuggingFace (honoring `HF_TOKEN` / `HF_HOME`).

Static worker list:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000
```

Kubernetes EndpointSlice discovery:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --service-discovery \
  --service-discovery-namespace prod \
  --selector app=engines-qwen3
```

Omit `--service-discovery-namespace` to watch all namespaces (requires
cluster-wide RBAC). For prefill/decode disaggregation, replace `--selector`
with `--prefill-selector` and `--decode-selector`.

External KV indexer as the cache-aware signal source:

```bash
sgl-router \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy cache_aware_zmq \
  --kv-indexer-endpoint http://10.0.0.10:50051 \
  --kv-indexer-query-timeout-ms 100 \
  --kv-indexer-query-max-inflight 32
```

The existing cache-aware policy and thresholds are reused. When configured, the
Indexer replaces the Router-local radix tree as the cache signal. A successful
query with no usable match selects by minimum active load; connection failures,
timeouts, local admission rejection, and server rejection fail the Router
request with `503` rather than silently switching signals. The timeout and local
concurrency bound default to 100ms and 32 respectively.

## 独立 Shortest-TTFT 策略

`--policy shortest_ttft` 实现 RTP-LLM 相近的 Shortest-TTFT 选择：对每个
worker 计算 `floor((10 * input_tokens - 7 * hit_tokens) / 10)`，再加上该
engine 的 queue pressure；随后取 TTFT 最小的前 30%（至少一个）并在相近
候选中优先最久未被调度的 worker。

```bash
sgl-router \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy shortest_ttft \
  --shortest-ttft-indexer-endpoint http://10.0.0.10:50051 \
  --shortest-ttft-indexer-query-timeout-ms 100 \
  --shortest-ttft-indexer-query-max-inflight 32
```

上述 Shortest-TTFT 专用 endpoint 使 V4 外部 Indexer 的每 worker prefix match
成为缓存命中的 authoritative 信号：`Empty`、超时、过载或不可达均为零命中，
不会回读本地 `HashTree`；拒绝型协议错误仍按请求错误返回。未配置该 endpoint
时仅保留直接单元调用与旧部署的本地 tree 兼容路径。engine load 则通过独立的
`LoadStat` SUB monitor 读取 worker `/server_info` 中声明的
`kv_events.load_endpoint_port_base` 和 `load_topic`。它不启用或复用
`cache_aware_zmq` 的策略/游标/admission 逻辑，也不使用 cache-aware 的
调优参数。当前 #34608 的发布负载是 running/waiting request gauge，不含
RTP-LLM 的毫秒级 `runningQueueTime`；某个 DP rank 缺失或过期时，策略会
回退到 router 的本地 in-flight 计数，而不是聚合不完整的远端数据。

## License

Apache-2.0.
