# Shortest-TTFT V4 Indexer 实施计划

> **执行说明：** 本计划按 `superpowers:executing-plans` 的小步验证方式执行。

**目标：** 在 `vin/shortest-ttft-indexer` 保持现有 Shortest-TTFT 算法、
LoadStat monitor、候选窗口与公平性不变；仅由 V4 外部 KV Indexer 为其提供
缓存命中结果。

**架构：** `main` 仅在 `--policy shortest_ttft` 且配置了新的专用 endpoint 时
创建现有 `GrpcPrefixIndex`。既有 ingress 将查询结果写入
`SelectionContext::external_prefix`；Shortest-TTFT 优先使用该 authoritative
signal 计算每个候选 URL 的命中 token。signal 存在时绝不读取本地 `HashTree`；
无 signal 时保留直接单元调用与未配置 endpoint 的兼容行为。

**技术栈：** Rust、clap、`sgl_kv_indexer::GrpcPrefixIndex`、现有 tokio 测试。

---

## 文件与职责

- 修改 `experimental/sgl-router/src/config/types.rs`：新增最小、独立的
  `ShortestTtftConfig`，复用通用 endpoint 值类型但不引用 cache-aware 配置。
- 修改 `experimental/sgl-router/src/config/cli.rs`：增加三项 shortest-TTFT
  专属 Indexer 参数及其作用域/数值校验。
- 修改 `experimental/sgl-router/src/main.rs`：按 policy 选择 Indexer endpoint，
  保持既有 cache-aware 创建路径不变。
- 修改 `experimental/sgl-router/src/policies/shortest_ttft.rs`：有 external
  signal 时按 address 的最大 `matched_prefix_blocks` 计算命中；无命中为零。
- 修改 `experimental/sgl-router/tests/component/policies/shortest_ttft.rs`：
  先验证外部 signal 覆盖相反的本地 tree。
- 修改 `experimental/sgl-router/src/config/cli.rs` 的单元测试：验证新 flag 的
  解析和错误作用域。
- 必要时更新各处 `ModelConfig` 字面量与 README 的启动示例。

## 阶段 1：先写失败测试

1. 在 Shortest-TTFT 组件测试构造相反的本地 tree 与 V4 `PrefixOutcome::Matched`：
   local 命中 A、external 命中 B，断言 B 被选中。
2. 在 CLI 测试中传入新 endpoint 与超时/并发参数，断言解析结果包含
   `model.shortest_ttft`；再断言非 `shortest_ttft` policy 拒绝这些参数。
3. 运行精确测试，预期此时失败，因为 external signal 尚未被 Shortest-TTFT
   消费，新参数也未定义。记录实际失败输出。

## 阶段 2：最小实现

1. 为 model config 添加 shortest-TTFT 私有配置，并补齐构造字面量。
2. 解析、检查新参数；保留原 `--kv-indexer-*` 参数只作用于
   `cache_aware_zmq`。
3. 在 main 中根据 `PolicyKind` 选出 endpoint，使用既有 `GrpcPrefixIndex`。
4. 在 policy 中实现 external signal 到命中 token 的映射：相同 address 取最大
   prefix blocks，乘 block size 后钳制到 input token 数；empty/未知 block size
   都返回零。
5. 格式化并重跑阶段 1 测试，直到通过。

## 阶段 3：回归验证与交付

1. 运行定向配置、Shortest-TTFT 与 external-indexer proxy 测试。
2. 运行 `cargo test --no-fail-fast`、`cargo fmt --check`、`git diff --check`。
3. 扫描 Shortest-TTFT 未依赖 `cache_aware_zmq` 或 admission；检查 diff 和
   worktree 状态。
4. 用规定 identity 提交；用 git bundle + rsync 传到 `vin`，在
   `/nvme/vin-sglang` 从 `personal` remote 推送 `vin/shortest-ttft-indexer`。

## 验收合同

- 指定 external signal 时，Shortest-TTFT 不会回读 local `HashTree`。
- `Empty`、超时、过载、不可达都由既有 ingress 转为零命中；拒绝型协议错误仍
  拒绝请求。
- 缓存命中来源外，LoadStat 评分、公平性和其余 policy 均不改变。
- 推送前的本地验证结果与最终远端 ref 可复查。
