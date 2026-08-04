# Router v2 执行计划

日期：2026-08-04
状态：执行中
分支：`vin/router-v2`

本计划执行 [Router Policy Step 1 与 Step 2 方案设计](../router-policy-step1-step2-design.md)。
实现严格限于 `experimental/sgl-router` 的 policy 与静态 Bucket 层：没有现成来源的
LoadMonitor 指标不新增采集，不实现 Reservation、投机发送或 Transfer-Aware。

## 完成标准

- Step 1 的 Prefill 与 Decode 都经过 `proposal → admission → guard → final`；
- Step 2 只以静态 Bucket 配置替换候选域来源，P/D policy 本身不感知 Bucket；
- 所有 Bucket retry 重新运行 proposal、admission、guard；
- `sticky`、`cache_aware_zmq` 保持原路径；
- 定向单测、全量 Rust 测试与 H20 的功能/E2E PoC 都有可复核结果。

## 提交切分与测试先行顺序

### 0. 分支与设计基线

1. 在个人远端创建 `vin/router-v2`；提交冻结设计与本执行计划。
2. 复核 `origin/main` 集成基线、个人 remote 和提交身份。

### 1. Step 1：统一 Prefill 决策链

1. **先写失败测试**：候选域内 P2 pair、primary/backup admission、当前域 fallback、
   Cache Benefit 与 Pressure Guard 的短路顺序。
2. 将 `CandidateRange` 演进为可复用的 `CandidateDomain`，但 Step 1 仅创建
   `Global(Prefill)`；保留兼容别名或最小适配，避免无关重构。
3. 明确拆分 P Admission（硬容量）和 P Guard（软切换）。LoadSnapshot 缺失/过期时
   降级 local active-load，不产生硬拒绝。
4. 接入 P2、Session-Aware、Cache-Aware 的 proposal；Score Policy 仍是独立软评分
   policy，但也经过 Router 共享硬 Admission；其 primary 不通过时走 domain fallback。
5. 扩展 route/component 测试，证明最终发送的是 `FinalDecision.selected`。

### 2. Step 1：Decode P2 与 PD 决策链

1. **先写失败测试**：PD 的 Decode P2 提出两个不同 D；D primary 失败切 backup；
   两者通过时由 D Guard 选较低压力者；legacy host-affinity 行为保持可选。
2. 新增独立 `DecodePolicy` 和 factory/config，默认新策略为 `power_of_two`；既有
   `decode_with_affinity` 收敛为 `legacy_host_affinity` 实现，而不是写死路径。
3. 在 `Final P` 后解析 `Global(Decode)` 并执行 D Admission、D Guard，最后才创建
   PD dispatch。仅使用既有 running/KV/request/active-load 输入。
4. 不虚构 transfer time、decode queue time、retraction 或 TPS profile；实际 dispatch
   生命周期只登记最终 D 的 local decode active-load。

### 3. Step 2：静态 Bucket、SLO 与 fallback

1. **先写失败测试**：按 P input length 与 D peak sequence length 构成域；同一候选集
   按唯一 `rank`；`slo_first` 先尝试 SLO eligible、尽后才降级；retry 重新产生 pair。
2. 增加小型静态 Bucket 配置：`id`、`stage`、`rank`、worker IDs、长度/context 兼容性、
   可选 TTFT/TPS 离线 profile 与 pending 限制。没有 Bucket 配置时恒为 GlobalDomain。
3. `ttft_slo_policy`、`tps_slo_policy` 各自支持 `disabled | best_effort | slo_first`；
   请求的 SLO 仅在明确提供且 Bucket 有相应离线 profile 时启用 profile 筛选。
4. 实现 `affinity_aware_range = bucket | global-first | global`。跨 Bucket primary 需在
   primary 自己所在 Bucket 校验静态兼容/SLO，再经过动态 admission；失败按语义回到
   target Bucket 的 P2/Stable Pair 路径。
5. PD 中 P/D 各自解析 Bucket domain、独立执行 policy/admission/guard；不引入
   transfer-aware 或 P/D 联合全局搜索。

### 4. 验证与 PoC

1. 本地运行 formatter、定向/全量 Rust 测试、`git diff --check`；若本机无 Rust
   toolchain，只记录实际环境限制，不以此替代远端验证。
2. 同步当前分支到 H20；运行同一测试集及两个 PoC：无 Bucket 的同构 baseline 与
   静态异构/逻辑 Bucket 矩阵。检查请求错误、fatal/OOM、RT、TTFT、TPS、吞吐、KV hit
   与路由理由。
3. 结论必须分开说明：接口/功能是否 GO，及某一 Bucket 配置是否达到生产性能 GO；
   不把同构静态拆分的结果泛化为 Bucket 架构结论。

## 迁移边界

未来指标与能力只在明确来源到位后按以下位置接入：

- `estimated_prefill_queue_ms`：共享 P 压力比较器的更高精度层；
- Decode retraction/queue/step 与 P→D transfer：D Guard 的可选 cost model；
- Reservation：`RoutePlan → Reservation Hook → existing dispatch`；
- 新 policy：实现 P 或 D policy trait 并注册，复用 domain/admission/guard。

这些增量不应修改 Bucket selector，也不应将硬约束变成 policy score。
