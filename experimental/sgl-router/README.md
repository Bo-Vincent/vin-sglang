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

## Routing policies

The first-phase policy set is:

- `power_of_two`: existing implementation. Randomly samples two workers and
  selects the one with fewer router-local in-flight requests.
- `session_aware`: maps the configured Session-ID header to one primary
  worker. Strict mode always keeps a healthy primary; soft mode may
  temporarily use one backup when Pressure Guard trips, without rewriting the
  session mapping.
- `cache_aware`: uses the existing KV-event `HashTree` to find the workers
  holding the longest reusable prefix. It compares one cache primary with one
  backup; Cache Benefit and Pressure Guard can short-circuit to the backup.

`session_aware` and `cache_aware` are independent top-level policies and never
run together for the same request. The existing `sticky` and
`cache_aware_zmq` policies remain available and retain their existing
behavior.

Both new affinity policies use the same bounded backup rules:

- By default, the backup is selected with Power-of-Two after excluding the
  primary.
- `--affinity-stable-pair true` instead chooses a deterministic backup for the
  same Session-ID or matched prefix.
- Candidate load uses per-worker active prefill tokens only when both
  candidates have that signal. Otherwise, both candidates fall back together
  to router-local in-flight request count.
- Pressure Guard selects the backup only when the primary exceeds both the
  absolute and relative thresholds.

Session-Aware example:

```bash
sgl-router \
  --model-id qwen3 \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy session_aware \
  --session-id-header x-sgl-session-id \
  --session-strict false \
  --affinity-stable-pair true
```

The Session-ID map is local to one router process and idle entries expire.
Configure expiration with `--session-idle-secs` and
`--session-eviction-interval-secs`.

Cache-Aware example:

```bash
sgl-router \
  --model-id qwen3 \
  --tokenizer-path /models/qwen3/tokenizer.json \
  --worker-urls http://10.0.0.1:30000 http://10.0.0.2:30000 \
  --policy cache_aware \
  --cache-benefit true \
  --affinity-pressure-guard true \
  --affinity-pressure-abs-threshold 8192 \
  --affinity-pressure-rel-threshold 1.5
```

For Cache Benefit:

```text
cache_saved_work      = reusable prefix tokens on the cache primary
remaining_uncached_work = request tokens - cache_saved_work
```

When enabled, it uses the cache primary only if
`cache_saved_work > remaining_uncached_work`; otherwise it selects the
backup. If request tokens, worker block size, or KV-event placement are
unavailable, Cache-Aware safely degrades to Power-of-Two.
If the globally longest prefix is held only by an ineligible worker,
Cache-Aware falls back to the longest shorter prefix that still has an
eligible holder.

The new policies expose
`sgl_router_policy_decisions_total{policy,reason}`. Labels are fixed code-path
values and never contain Session-ID, prompt content, or worker URL.

First phase does not include SLO gating, DP-aware routing, KV prefetch,
eviction/offload control, or other Orchestrator actions.

## License

Apache-2.0.
