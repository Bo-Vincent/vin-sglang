# TraceLab 256-worker HTTP Simulator 复现 SOP

适用场景：在单机启动 256 个 SGLang CPU Simulator HTTP worker，比较
`power_of_two`、Native `cache_aware`、`cache_aware_zmq` 和 `shortest_ttft`。
该测试验证 Router、KV Indexer、LoadMonitor 与 KV event 路径的控制面行为；结果统一标记为
`simulator_predicted_relative`，不作为真实 GPU 的绝对性能数据。

可在 Codex 中使用仓库内 Skill：
[`sgl-router-tracelab-simulator-evaluation`](../../.codex/skills/sgl-router-tracelab-simulator-evaluation/SKILL.md)。
Skill 随仓库版本化；若本地 Codex 未自动发现项目内 Skill，可一次性链接到本机 Skill 目录：

```bash
ln -s "$(pwd)/.codex/skills/sgl-router-tracelab-simulator-evaluation" \
  "${CODEX_HOME:-$HOME/.codex}/skills/sgl-router-tracelab-simulator-evaluation"
```

## 代码位置

| 用途 | 文件 |
| --- | --- |
| 256-worker TraceLab runner | `experimental/sgl-router/scripts/run_tracelab_simulator_http_fleet_e2e.py` |
| 结果 analyzer | `experimental/sgl-router/scripts/analyze_simulator_http_fleet_e2e.py` |
| Trace 选择和虚拟 prompt | `experimental/sgl-router/scripts/tracelab_replay.py` |
| Simulator runtime | `tools/sglang-simulator/` |
| runner 单测 | `experimental/sgl-router/tests/scripts/test_run_tracelab_simulator_http_fleet_e2e.py` |
| analyzer 单测 | `experimental/sgl-router/tests/scripts/test_analyze_simulator_http_fleet_e2e.py` |
| replay 单测 | `experimental/sgl-router/tests/scripts/test_tracelab_replay.py` |

`run_simulator_http_fleet_e2e.py` 是另一套通用 synthetic workload runner；本 SOP 的公开
TraceLab 重放使用 `run_tracelab_simulator_http_fleet_e2e.py`。

## 固定实验合同

| 项目 | 固定值 |
| --- | --- |
| Trace | TraceLab v0.0.2 `syfi_coding_trace.jsonl.gz`，SHA-256 `11ce51ec0a25e3d1d95b025bca2f7d1647e47571eb7cc968acd5fc64d4b4fb65` |
| Slice | `provider=codex`；256 session；每 session 连续 4 turn；seed `20260822` |
| 请求 | 每 case 256 个 warmup、768 个 measurement；session 内串行、session 间并发 |
| Worker | 256 个 CPU Simulator HTTP worker；`max_total_tokens=32768`、`max_running_requests=32` |
| 策略 | `power_of_two,cache_aware,cache_aware_zmq,shortest_ttft` |
| 负载与轮数 | 64 QPS；每策略 3 repeats，共 12 case |
| Indexer | query timeout 10s；max in-flight 256；max concurrent streams 512；外部查询必须全部成功 |

公开 TraceLab 没有原始 prompt 和 KV hash。runner 依据 session、轮次和 token 几何重建无碰撞的
session-local 虚拟 prompt；它不模拟跨 session 的 system prompt、tool schema 或文档共享。

## 前置条件

1. checkout `vin/rust-v3-simulator-test`，或从该分支创建干净临时 worktree。
2. 选择已预留 CPU、内存和端口范围的 Linux host；运行期间不要并行启动第二组 256-worker
   fleet。
3. 远端具备 Rust、Python、`aiohttp`、`tokenizers`、SGLang Simulator、完整
   `Qwen/Qwen2.5-7B-Instruct` tokenizer，以及 TraceLab 数据文件。
4. 传输源码或原始结果时使用 `rsync`；每次正式运行使用新的结果目录，保留已有结果。

## 1. 本地预检

在仓库根目录执行：

```bash
cd experimental/sgl-router

python3 -m unittest \
  tests/scripts/test_tracelab_replay.py \
  tests/scripts/test_run_tracelab_simulator_http_fleet_e2e.py \
  tests/scripts/test_analyze_simulator_http_fleet_e2e.py -v

python3 scripts/run_tracelab_simulator_http_fleet_e2e.py \
  --policies power_of_two,cache_aware,cache_aware_zmq,shortest_ttft \
  --repeats 3 \
  --request-rate 64
```

预期 dry-run 输出 12 个 case、256 worker、256 session 和每 case 768 个 measurement 请求。

记录待测源代码身份：

```bash
git rev-parse HEAD
git status --short
```

## 2. 同步与构建

设置已获授权的测试主机和唯一 run id。不要把含结果的远端目录回写到源码目录。

```bash
export BENCH_HOST=<authorized-benchmark-host>
export LOCAL_SGLANG_ROOT="$(git rev-parse --show-toplevel)"
export REMOTE_RUN=sgl-router-bench/tracelab-256-<run-id>
export SGLANG_SOURCE_COMMIT="$(git -C "$LOCAL_SGLANG_ROOT" rev-parse HEAD)"

ssh "$BENCH_HOST" "mkdir -p \"\$HOME/$REMOTE_RUN/source\" \"\$HOME/$REMOTE_RUN/results\""
rsync -a --exclude .git --exclude target \
  "$LOCAL_SGLANG_ROOT/" "$BENCH_HOST:~/$REMOTE_RUN/source/"

ssh "$BENCH_HOST"
```

登录远端后构建 Router 和 Indexer：

```bash
export REMOTE_RUN=sgl-router-bench/tracelab-256-<run-id>
export REMOTE_ROOT="$HOME/$REMOTE_RUN"
export REMOTE_SGLANG_ROOT="$REMOTE_ROOT/source"
export SGL_ROUTER_ROOT="$REMOTE_SGLANG_ROOT/experimental/sgl-router"
# 使用本地预检打印的 commit；源码 rsync 时不复制 .git。
export SGLANG_SOURCE_COMMIT=<local-source-commit>

cd "$SGL_ROUTER_ROOT"
cargo build --release --bin sgl-router
cargo build --release --manifest-path sgl-kv-indexer/Cargo.toml --bins
```

## 3. 远端执行

先按测试机实际安装位置设置以下变量；所有路径都应指向同一份远端 source 和已验证运行时。

```bash
export PYTHON_BIN=/path/to/python
export SIMULATOR_SITE=/path/to/python/site-packages
export SIMULATOR_CONFIG="$REMOTE_SGLANG_ROOT/tools/sglang-simulator/examples/sim_configs/replay.json"
export MODEL_PATH=/path/to/Qwen2.5-7B-Instruct
export TOKENIZER_PATH=/path/to/Qwen2.5-7B-Instruct
export TRACE_FILE=/path/to/syfi_coding_trace.jsonl.gz
export RESULT_DIR="$REMOTE_ROOT/results/tracelab-256-<run-id>"
```

`PYTHON_BIN` 是启动 runner 和 worker 的 Python，`SIMULATOR_SITE` 会被优先加入 worker
的 `PYTHONPATH`。两者可以来自同一已验证环境，也可以分别提供 runner 依赖和 SGLang
Simulator 所需的兼容依赖。执行前必须验证它们组合后可以导入 Simulator：

```bash
PYTHONPATH="$SIMULATOR_SITE:$REMOTE_SGLANG_ROOT/tools/sglang-simulator/src:$REMOTE_SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -c '
import aiohttp, tokenizers, aiconfigurator
import transformers.image_processing_backends
from sglang_simulator.simulation.sglang.launch_server import validate_launch_runtime
validate_launch_runtime()
print("simulator import preflight: ok")
'
```

确认路径与端口可用后，从 `SGL_ROUTER_ROOT` 运行：

```bash
"$PYTHON_BIN" scripts/run_tracelab_simulator_http_fleet_e2e.py \
  --source-root "$REMOTE_SGLANG_ROOT" \
  --router-binary "$SGL_ROUTER_ROOT/target/release/sgl-router" \
  --router-cwd "$SGL_ROUTER_ROOT" \
  --indexer-server "$SGL_ROUTER_ROOT/target/release/kv-indexer-server" \
  --indexer-bridge "$SGL_ROUTER_ROOT/target/release/kv-indexer-bridge" \
  --python "$PYTHON_BIN" \
  --simulator-site "$SIMULATOR_SITE" \
  --simulator-config "$SIMULATOR_CONFIG" \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER_PATH" \
  --trace "$TRACE_FILE" \
  --results-dir "$RESULT_DIR" \
  --policies power_of_two,cache_aware,cache_aware_zmq,shortest_ttft \
  --repeats 3 \
  --request-rate 64 \
  --kv-indexer-query-timeout-ms 10000 \
  --kv-indexer-query-max-inflight 256 \
  --kv-indexer-max-concurrent-streams 512 \
  --require-indexer-success \
  --execute
```

长时间运行应由测试机的任务管理方式保持存活。若运行中断，先读取当前 case、日志、
`request_errors`、fatal/OOM 和 `RUN_COMPLETE`。只有结果目录 manifest、源码 commit、二进制
SHA 和所有参数完全一致时才追加 `--resume`；不满足时创建新的 `RESULT_DIR`。

## 4. 分析与验收

完成后执行：

```bash
"$PYTHON_BIN" scripts/analyze_simulator_http_fleet_e2e.py \
  --results-dir "$RESULT_DIR" \
  --output-dir "$RESULT_DIR/analysis"
```

验收条件：

- 根目录为 `RUN_COMPLETE=ok`，12 个 case 均为 `COMPLETE=ok`；
- 每个 case 的 `request_errors=0`，日志无 fatal 或 OOM；
- Native `cache_aware` 的每个 cache candidate 都有真实 cache metric 和 LoadMonitor 决策，
  且 `router_local_decisions=0`、`zero_snapshot_decisions=0`；
- analyzer 产出 TTFT、E2E、TPS、吞吐、KV hit、worker CV 与 policy reason；
- 主指标三轮 RSD 高于 10% 时，保留主轮，在新目录仅对波动策略补两轮，再使用
  `--confirmation-results-dir` 合并分析。

同步原始结果时只追加到本地独立制品目录：

```bash
rsync -a "$BENCH_HOST:~/$REMOTE_RUN/results/tracelab-256-<run-id>/" \
  /path/to/local-artifacts/tracelab-256-<run-id>/
```

## 常见结果解释

- Native `cache_aware` 的高 KV hit 与高 TTFT 可以同时出现：外部 Indexer 查询在选择前同步
  等待，其时间应从 `sgl_router_kv_indexer_query_duration_seconds` 单独读取。
- `cache_aware_zmq` 的本地 lookup 命中不等价于连续 device KV 命中；同时检查实际 KV hit、
  worker CV 和 ZMQ lookup 指标。
- 256-worker 数值不替代真实 GPU 基准。对 GPU TTFT、E2E、TPS 和吞吐结论，另行运行真实
  SGLang worker 的固定合同。
