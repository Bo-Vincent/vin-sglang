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

## Engine 发布的负载快照

Router 不再连接旧的 Load Reporter/LoadMonitor gRPC 服务。支持 #34608 的
Worker 通过 `/server_info.kv_events` 广播 `load_endpoint_port_base` 和
`load_topic`；Router 为每个 DP rank 建立 ZMQ SUB，并只接受三帧
`[topic, sequence, msgpack LoadStat]` 消息。

每个 `LoadStat` 的四个可信字段是运行请求数、等待请求数、已用 KV tokens
和 KV 容量。Router 只有在同一 Worker 的所有已广播 rank 都有新鲜快照时才
使用 `running + waiting`；缺失、过期或格式错误时会回退到 Router 本地的
in-flight 计数。不会把请求数解释成 token、排队时延或旧协议的压力指标。

## Session、Cache 与 Score 策略

`power_of_two`、`session_aware`、`cache_aware` 和 `score_policy` 在健康的
Prefill 候选范围中产生候选，再通过共享准入逻辑决定最终 Worker。`cache_aware`
只读取入口阶段准备好的 Indexer 结果，不会在选择热路径发起同步 Indexer RPC。
`fused_score` 保留为兼容名称；`load_based` 是 Router 本地 active-load 的软分，
不是 Engine queue/token 的硬准入结论。

## External KV indexer (cache_aware_zmq)

External KV indexer as the cache-aware signal source for the legacy
`cache_aware_zmq` policy:

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

## License

Apache-2.0.
