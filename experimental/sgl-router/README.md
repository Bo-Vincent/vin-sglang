# sgl-router

Slim, KV-aware, OpenAI-compatible router for SGLang workers.

Serves a single model and routes across its workers. Exposes
`/v1/tokenize`, `/v1/detokenize`, `/v1/models`, `/v1/chat/completions`
(buffered and SSE), plus `/healthz` / `/readyz` and `/metrics`. Worker pools
come from either a static URL list or Kubernetes EndpointSlice discovery.

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

## Engine-reported load monitoring

Load monitoring is disabled by default. When enabled, the Router first binds
an independent gRPC listener, then asks every discovered worker to start or
renew reporting through `/v1/start_reporting`. Port `0` is supported and the
actual bound port is sent to the engine:

```bash
sgl-router \
  --host 0.0.0.0 --port 30000 \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy round_robin \
  --load-monitor \
  --load-monitor-bind-host 0.0.0.0 \
  --load-monitor-bind-port 0 \
  --load-monitor-report-ip 10.0.0.10
```

`--load-monitor-report-ip` is required and must be reachable from the engine.
The first version uses a fixed 1-second report interval, 3-second freshness
window, 15-second lease, and 2-second registration timeout. This change keeps
routing policies unchanged; the immutable snapshot is the read-only boundary
for follow-up scheduling integrations.

The monitor maintains an internal immutable, versioned snapshot with worker
freshness, source and sequence metadata, complete DP-rank values, and aggregate
load. This snapshot is intentionally not exposed as a public HTTP endpoint;
follow-up scheduling policies consume it inside the Router process.

The Router intentionally sends no `Authorization` header to
`/v1/start_reporting`. It is therefore compatible with an unauthenticated
open-source or fake engine. Engine builds that enforce `ADMIN_FORCE` on this
endpoint currently reject registration with 401/403; authenticated reporting
is outside this Router-only change.

## Legacy Cache-Aware ZMQ policy

The existing `cache_aware_zmq` policy can additionally consume an external
KV-indexer signal while keeping its local ZMQ subscription and legacy
threshold behavior:

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

When configured, the Indexer replaces the Router-local radix tree as the cache
signal. A successful query with no usable match, a connection failure, a
timeout, or local admission rejection falls back to minimum-load routing;
contract rejections fail the request with `503`. The timeout and local
concurrency bound default to 100ms and 32 respectively.

## Session-Aware and Cache-Aware policies

`power_of_two`, `session_aware`, `cache_aware`, and `score_policy` are the new
Step 1 policies. They produce a primary plus an optional backup inside the
current healthy Prefill candidate range. The Router then applies shared hard
admission (fresh engine running/KV capacity when available); affinity policies
may additionally apply Cache Benefit / Pressure Guard before the final worker
is dispatched.

```bash
sgl-router \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy session_aware \
  --session-id-header x-session-id \
  --session-idle-secs 600 \
  --session-eviction-interval-secs 60 \
  --stable-pair \
  --affinity-mode soft
```

```bash
sgl-router \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy cache_aware \
  --kv-indexer-endpoint http://10.0.0.10:50051 \
  --stable-pair
```

`cache_aware` only reads the indexer result prepared at ingress; it never
issues a synchronous Indexer RPC from the selection path. Missing, stale, or
disabled LoadMonitor data does not hard-reject registry-healthy workers.
Bucket/SLO policy is not enabled by these flags yet: it will replace the
candidate range before policy selection, not rewrite these policies. The
legacy `sticky` and `cache_aware_zmq` policies keep their existing direct
dispatch behavior; they do not silently opt into this new shared layer.

## Score policy

`score_policy` is a top-level policy, parallel to `power_of_two`,
`session_aware`, and `cache_aware`. It uses the generic score
composition implementation internally; it does not turn Session or Cache
affinity into a collection of flags. As a new top-level policy it participates
in shared hard Prefill admission, but it has no affinity backup or soft
Cache/Pressure Guard. The compatibility spelling `fused_score` retains its
existing behavior.

```bash
sgl-router \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy score_policy \
  --fuse prefix_cache=2.0,load_based=0.3
```

When `--fuse` is omitted, `score_policy` uses `prefix_cache,load_based`.
`load_based` is currently a router-local active-load **soft score**, not an
Engine LoadMonitor prefill-queue estimate or a hard admission rule. The
upstream spelling `fused_score` remains available for compatibility; use
`score_policy` for new Step 1 configuration.

## License

Apache-2.0.
