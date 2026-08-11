# Router Policy 热路径 Microbenchmark 报告

## 结论

本轮 Router-only 验证通过：8、64、256 endpoints 下，Step 3 的多维压力指标没有给
P2、Session-Aware、Cache-Aware 或 Decode Policy 引入不可接受的决策开销。

- 正式矩阵共 9 轮、342 个 Criterion scenario samples，211/211 项验收全部通过。
- 三种变体的最大三轮 p95 RSD 分别为 4.12%、2.80% 和 2.12%，不需要确认轮。
- 256 endpoints 下，Step 3 rich-pressure 的 request-path p95 为：P2 35.22 us、
  Session 35.77 us、Decode 35.69 us、Cache top-32 110.36 us。
- 256 endpoints 的 scheduling snapshot p95 为 29.59 us；它仍随 endpoint 数量线性
  增长，但 64 到 256 的增长为 3.99x，符合预期并低于 6x 门槛。
- rich-pressure 相对 common-metrics 的最大 p95 增量为 9.04%，发生在 8 endpoints 的
  0.8 us 级 snapshot；256 endpoints 的各主路径增量均不超过 1.36%。
- 所有 9 轮均固定在一个 CPU 上，CPU 利用率为 0.967--0.971，CPU migration 为 0。

因此，本结果支持将 scheduling snapshot、候选成员索引和 Session 快路径作为当前
Step 3 实现保留。它证明的是单 Router 决策热路径在 256 endpoints 内可接受，不等价于
已经证明高 Router RPS、并发 LoadMonitor 写入或多 Router 实例下的生产容量。

## 验证对象

对比三个版本：

| 变体 | 含义 |
| --- | --- |
| Step 2 | 正式实验使用 `7701026663a0457e31a790de06ac8c7a1d7aa8a3`；rebase 后的等价提交为 `614b6402fc`，均为 Bucket/SLO 后、Step 3 前的基线 |
| Step 3 common | 当前策略代码，只填充原有通用负载字段 |
| Step 3 rich | 当前策略代码，同时填充 Prefill queue、Decode queue/retraction、Decode step 和 active-token 字段 |

Step 2 和 Step 3 使用同一份 benchmark 源码和同一份 `Cargo.lock`。benchmark 源码
SHA-256 为 `94e9ac6acf35045bc74290fe83424fb3b3030f7d541c58c9467cbdebf66c13c0`。
分支 rebase 到 `origin/main@a58fa0388e` 后，`experimental/sgl-router` 和
`experimental/sgl-kv-indexer` 的 tracked 文件与远端 exact-head 验证源码 checksum
完全一致；Router 相关源码树相对正式实验提交没有内容变化，因此无需重跑性能矩阵。

覆盖以下决策路径：

- scheduling snapshot：8 / 64 / 256 endpoints，DP=1 / 8；
- Prefill Power-of-Two：policy-only 和 request-path；
- Session-Aware：稳定命中，policy-only 和 request-path；
- Cache-Aware：top-K=4 / 16 / 32，policy-only 和 request-path；
- Decode Power-of-Two：policy-only 和 request-path。

`policy-only` 复用已构造的 snapshot 和候选域，观察策略本身；`request-path` 每次重新
捕获 snapshot 并构造候选域，更接近 Router 每请求路径。Cache 场景还包含既有 prefix
候选解析和有界 top-K 决策。

## 环境与方法

| 项目 | 值 |
| --- | --- |
| 机器 | `h20-8-usa`，Alibaba Cloud KVM |
| CPU | Intel Xeon Gold 6462C，2 sockets，64 cores / 128 threads |
| 内存 | 991 GiB |
| CPU 固定 | `taskset -c 2` |
| Criterion | warm-up 200 ms，measurement 1 s，30 samples |
| 重复 | 每个变体独立三轮 |
| 辅助观测 | 独立 allocation pass；`perf stat` 记录 task-clock、cycles、instructions、cache-misses、context-switches、cpu-migrations |
| GPU | 未使用；这是纯 CPU Router benchmark |

每个 fixture 通过公开 LoadMonitor gRPC reporting 接口写入负载数据，不直接修改内部
store。正式 runner 运行顺序为每轮 Step 2、Step 3 common、Step 3 rich，避免只比较
不同时间段的机器状态。

## 正式结果

### 256 endpoints 主路径

| 场景 | Step 2 p95 | Step 3 common p95 | Step 3 rich p95 | rich / common |
| --- | ---: | ---: | ---: | ---: |
| Snapshot，DP=1 | 140.87 us | 29.20 us | 29.59 us | 1.014x |
| P2 policy-only | 1.927 us | 0.199 us | 0.200 us | 1.003x |
| P2 request-path | 144.28 us | 35.47 us | 35.22 us | 0.993x |
| Session policy-only | 2.918 us | 0.368 us | 0.367 us | 0.996x |
| Session request-path | 144.87 us | 35.69 us | 35.77 us | 1.002x |
| Decode policy-only | 2.531 us | 0.240 us | 0.241 us | 1.003x |
| Decode request-path | 144.01 us | 35.67 us | 35.69 us | 1.000x |
| Cache top-32 policy-only | 86.06 us | 80.06 us | 81.22 us | 1.015x |
| Cache top-32 request-path | 227.08 us | 109.29 us | 110.36 us | 1.010x |

Step 3 相对 Step 2 的主要改善来自专用 `SchedulingSnapshot`：策略不再为每个请求构造
诊断时间字符串、rank 详情、URL/model 列表和排序结果。多维压力字段本身的成本由
rich/common 对比隔离；在 256 endpoints 下，其增量约为 0%--1.5%。

### 扩展性与绝对门槛

| 检查 | 结果 | 门槛 |
| --- | ---: | ---: |
| Snapshot 256，DP=1 | 29.59 us | <= 100 us |
| P2 request-path 256 | 35.22 us | <= 150 us |
| Session request-path 256 | 35.77 us | <= 150 us |
| Decode request-path 256 | 35.69 us | <= 150 us |
| Cache top-32 request-path 256 | 110.36 us | <= 300 us |
| P2 policy-only，64 -> 256 | 1.034x | <= 1.5x |
| Session policy-only，64 -> 256 | 1.015x | <= 1.5x |
| Decode policy-only，64 -> 256 | 1.030x | <= 1.5x |
| Snapshot，64 -> 256 | 3.985x | <= 6x |
| Cache top-4 / 16 / 32，64 -> 256 | 3.853x / 3.635x / 3.365x | <= 6x |

P2、Session 和 Decode 的 policy-only 路径在 64 到 256 endpoints 之间基本保持常数；
request-path 的增长主要来自必须复制一份不可变 scheduling snapshot。Cache-Aware 需要
处理 Indexer 返回的候选并比较 top-K，因此仍随 endpoint/candidate 数量增长。

### 分配量

| 256 endpoints 场景 | Step 2 alloc/op | Step 3 alloc/op | Step 2 bytes/op | Step 3 bytes/op |
| --- | ---: | ---: | ---: | ---: |
| Snapshot，DP=1 | 2051 | 264 | 257318 | 265084 |
| P2 request-path | 2055 | 269 | 259389 | 271779 |
| Session request-path | 2058 | 272 | 259429 | 271819 |
| Decode request-path | 2055 | 269 | 259389 | 271779 |
| Cache top-32 request-path | 2888 | 1099 | 384139 | 382017 |

专用 snapshot 将非 Cache request-path 的分配次数减少约 87%，Cache top-32 减少约
62%。P2/Session/Decode 的 bytes/op 比 Step 2 高约 5%，因为 Step 3 的 `AggregateLoad`
包含新增的可选压力字段；rich/common 的分配量和字节数完全相同，说明填充这些字段不会
再触发额外热路径分配。

## 根据失败样本完成的修复

正式 V6 之前的预跑暴露并修复了以下问题：

1. 诊断型 `LoadMonitorSnapshot` 会复制并排序所有 worker/rank 元数据，256 endpoints
   snapshot 超过 140 us。新增 scheduling-only snapshot 后降到约 30 us，同时保留原
   diagnostic API。
2. Admission/Guard 对候选归属做重复 Worker-ID 线性扫描。`CandidateDomain` 现在维护
   request-local 成员索引；小池使用指针数组，大池使用指针集合，并保留等价 Worker-ID
   的兼容降级。
3. Session hit 每次线性查找已分配 worker。Assignment 现在保存 `Weak<Worker>`，候选域
   成员校验通过时直接命中；registry 已替换或没有成员索引时仍按 Worker-ID 降级。
4. rich-pressure Cache path 曾重复扫描相同候选以判断指标完整性。`FreshLoadLookup` 改为
   单次捕获本地负载和指标能力，消除了 20% 以上的预跑回退。
5. benchmark fixture 的容量上限和 Session 命中位置曾造成无关的全域 fallback 与随机
   线性查找波动。正式合同固定容量和中间 worker 命中位置，使 Step 2/Step 3 比较的工作
   量一致。受影响的旧目录均保留但没有并入正式分析。

## 验收与产物

正式 analyzer 输出：

- `passed=true`；
- 211/211 checks PASS；
- 9/9 perf runs 的 CPU 利用率在 0.90--1.05 内；
- 9/9 perf runs 的 CPU migration 为 0；
- 未发现 fatal、OOM、panic 或 benchmark error；
- 342/342 Criterion samples、9/9 allocation logs、9/9 perf logs 完整。

代码验证：

- `cargo fmt -- --check`；
- `cargo clippy --locked --all-targets --all-features -- -D warnings`；
- Router `cargo test --locked --all-targets --all-features`：559 + 4 + 55 + 74 tests；
- Indexer tests：52 tests；
- analyzer tests：10 tests；
- LoadReporter 单元测试：17 tests、20 subtests；
- LoadReporter 单 owner E2E：2 个 tokenizer workers、1 次生成、1 条 gRPC report
  stream，1 test PASS；
- Python `py_compile` 和 `git diff --check`。

LoadReporter E2E 使用本地 `Qwen2.5-7B-Instruct`。远端预编译
`sglang-kernel=0.4.5+cu129` 低于当前源码声明的最低版本，测试通过源码提供的
`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` 进入运行时；模型加载、CUDA graph、生成和
report stream 均实际执行。这只用于验证本次 Python 生命周期改动，不构成旧 kernel
版本的生产兼容性声明。

原始结果未提交到仓库：

- 远端：`/root/router-policy-bench/results/router-pressure-microbench-final-v6-locked-20260810`
- 本地：`/Users/gaobo/Documents/mooncake/.vin_stage/router-pressure-microbench-results-v6-20260810`
- 正式分析：上述目录下的 `analysis-formal/analysis.json` 和 `analysis-formal/report.md`

## 边界

本报告的 p95 是 Criterion batch-normalized per-operation samples 的 p95，再对三轮取中位
数；它用于发现算法和分配回退，不是线上单请求服务尾延迟。

本轮没有覆盖：

- 多 Router 进程或高并发 Router RPS；
- snapshot 读锁与持续 LoadMonitor report 写入的竞争；
- HTTP 解析、tokenization、Indexer RPC、worker 网络与模型执行；
- 跨 NUMA core 调度、CPU oversubscription 或生产容器限额；
- 256 以上 endpoints。

若生产池会超过 256 endpoints，下一步应补并发 Router-only load benchmark：固定 endpoint
规模后逐步提高 Router 线程和请求速率，同时让 LoadMonitor 并发写入，观察 decision
p50/p95/p99、锁等待、CPU、allocation rate 和吞吐饱和点。
