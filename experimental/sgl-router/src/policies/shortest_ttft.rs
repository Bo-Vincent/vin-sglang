// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! 独立的 RTP-LLM 风格 Shortest-TTFT 路由策略。
//!
//! 该策略只读取两类输入：KV-event 树给出的每个 worker 前缀命中，以及
//! engine load monitor 给出的实际运行/等待请求数。它不依赖
//! `cache_aware_zmq`、admission 或 eligibility filter；缺少或过期的
//! monitor 数据只回退到 worker 的本地 in-flight 计数。

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::warn;
use zeromq::{Socket, SocketRecv, SubSocket, ZmqMessage};

use crate::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree,
};
use crate::policies::{Policy, SelectionContext};
use crate::workers::Worker;

const RTP_CANDIDATE_PERCENT_NUMERATOR: usize = 3;
const RTP_CANDIDATE_PERCENT_DENOMINATOR: usize = 10;
const RTP_TTFT_THRESHOLD_PERCENTAGE: f64 = 0.1;
const RTP_STDDEV_THRESHOLD_FACTOR: f64 = 0.5;

/// #34608 `LoadStat` 的稳定字段。每个 DP rank 都发布一个独立 gauge。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadStat {
    pub num_running_reqs: u64,
    pub num_waiting_reqs: u64,
    pub num_tokens: u64,
    pub max_total_num_tokens: u64,
}

#[derive(Clone, Debug)]
struct LoadEntry {
    load: LoadStat,
    captured_at: Instant,
}

#[derive(Default)]
struct EngineLoadState {
    by_rank: HashMap<(String, u32), LoadEntry>,
    expected: HashSet<(String, u32)>,
}

/// 每个 `(worker_url, DP rank)` 的最近 engine gauge。
///
/// 对一个 worker，只要任一已发现 DP rank 缺失或过期，就不返回部分聚合；
/// 否则一条沉默 rank 会让该 worker 看起来比实际更空闲。锁只覆盖单次状态
/// 拷贝，绝不跨越网络 I/O 或回调边界。
pub struct EngineLoadTable {
    state: RwLock<EngineLoadState>,
    freshness: Duration,
}

impl fmt::Debug for EngineLoadTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.read();
        f.debug_struct("EngineLoadTable")
            .field("rank_entries", &state.by_rank.len())
            .field("expected_ranks", &state.expected.len())
            .field("freshness", &self.freshness)
            .finish()
    }
}

impl EngineLoadTable {
    pub fn new() -> Arc<Self> {
        Self::with_freshness(Duration::from_secs(2))
    }

    pub fn with_freshness(freshness: Duration) -> Arc<Self> {
        Arc::new(Self {
            state: RwLock::new(EngineLoadState::default()),
            freshness,
        })
    }

    /// 标记一个由 `/server_info` 宣告的 DP rank。注册先于消息到达，故冷启动
    /// 期间返回 `None`，由策略使用本地计数回退。
    pub fn mark_expected_rank(&self, worker_url: &str, dp_rank: u32) {
        self.state
            .write()
            .expected
            .insert((worker_url.to_string(), dp_rank));
    }

    /// 写入一个新 gauge。Load 是瞬时状态而不是增量，最新值直接覆盖旧值。
    pub fn set(&self, worker_url: &str, dp_rank: u32, load: LoadStat, captured_at: Instant) {
        self.state.write().by_rank.insert(
            (worker_url.to_string(), dp_rank),
            LoadEntry { load, captured_at },
        );
    }

    /// 删除 worker 的期望 rank 和已缓存 gauge，避免旧端点重加后污染新实例。
    pub fn remove_worker(&self, worker_url: &str) {
        let mut state = self.state.write();
        state.expected.retain(|(url, _)| url != worker_url);
        state.by_rank.retain(|(url, _), _| url != worker_url);
    }

    /// 聚合所有完整、未过期 DP rank 的 queue pressure。
    pub fn queue_pressure(&self, worker_url: &str, now: Instant) -> Option<u64> {
        let state = self.state.read();
        let ranks: Vec<u32> = state
            .expected
            .iter()
            .filter_map(|(url, rank)| (url == worker_url).then_some(*rank))
            .collect();
        if ranks.is_empty() {
            return None;
        }

        ranks.into_iter().try_fold(0_u64, |total, rank| {
            let entry = state.by_rank.get(&(worker_url.to_string(), rank))?;
            if now.saturating_duration_since(entry.captured_at) > self.freshness {
                return None;
            }
            Some(
                total.saturating_add(
                    entry
                        .load
                        .num_running_reqs
                        .saturating_add(entry.load.num_waiting_reqs),
                ),
            )
        })
    }
}

/// 已由 worker `/server_info` 解析并规范化的 load PUB 端点。
///
/// `host` 必须是 Router 可连接的地址而不是 engine 的 wildcard bind
/// 地址；`port_base + dp_rank` 是 #34608 的唯一端点导出规则。
#[derive(Clone, Debug)]
pub struct LoadEndpointConfig {
    pub host: String,
    pub port_base: u16,
    pub topic: String,
    pub dp_size: u32,
}

struct LoadSubscriberHandle {
    cancel: CancellationToken,
    join: JoinHandle<()>,
}

/// 一个 policy 私有的 #34608 load socket 管理器。
///
/// monitor 与 KV-event subscriber 不共享任务或状态：前者是可覆盖的 gauge，
/// 后者是有序的 cache mutation stream。二者混用 cursor/replay 语义会让
/// publisher 重启后的第一个有效 load 永远被错误忽略。
pub struct EngineLoadMonitor {
    table: Arc<EngineLoadTable>,
    handles: AsyncMutex<HashMap<(String, u32), LoadSubscriberHandle>>,
}

impl fmt::Debug for EngineLoadMonitor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EngineLoadMonitor")
            .field("table", &self.table)
            .finish_non_exhaustive()
    }
}

impl EngineLoadMonitor {
    pub fn new(table: Arc<EngineLoadTable>) -> Arc<Self> {
        Arc::new(Self {
            table,
            handles: AsyncMutex::new(HashMap::new()),
        })
    }

    /// 为一个 worker 的所有 DP rank 启动独立 SUB task。
    ///
    /// 不在 handles 锁内执行 socket I/O；SUB task 自己持有 socket，删除时
    /// 先取消并 join，最后清 table，避免已移除 worker 的迟到消息重新写回。
    pub async fn add_worker(&self, worker_url: &str, config: LoadEndpointConfig) {
        let mut handles = self.handles.lock().await;
        for dp_rank in 0..config.dp_size {
            let key = (worker_url.to_string(), dp_rank);
            if handles.contains_key(&key) {
                continue;
            }
            let port = match u16::try_from(config.port_base as u32 + dp_rank) {
                Ok(port) => port,
                Err(_) => {
                    warn!(
                        worker_url,
                        dp_rank,
                        port_base = config.port_base,
                        "shortest-ttft: load port overflows u16; skipping rank"
                    );
                    continue;
                }
            };
            let endpoint = format!("tcp://{}:{port}", config.host);
            let cancel = CancellationToken::new();
            let task_cancel = cancel.clone();
            let task_table = Arc::clone(&self.table);
            let task_worker_url = worker_url.to_string();
            let task_topic = config.topic.clone();
            let join = tokio::spawn(async move {
                run_load_subscriber(
                    task_worker_url,
                    dp_rank,
                    endpoint,
                    task_topic,
                    task_table,
                    task_cancel,
                )
                .await;
            });
            self.table.mark_expected_rank(worker_url, dp_rank);
            handles.insert(key, LoadSubscriberHandle { cancel, join });
        }
    }

    /// 终止并等待 worker 的所有 rank，再清除它的 cached/expected state。
    pub async fn remove_worker(&self, worker_url: &str) {
        let drained = {
            let mut handles = self.handles.lock().await;
            let keys: Vec<(String, u32)> = handles
                .keys()
                .filter(|(url, _)| url == worker_url)
                .cloned()
                .collect();
            keys.into_iter()
                .filter_map(|key| handles.remove(&key))
                .collect::<Vec<_>>()
        };
        for handle in drained {
            handle.cancel.cancel();
            if let Err(error) = handle.join.await {
                warn!(worker_url, %error, "shortest-ttft: load subscriber did not join cleanly");
            }
        }
        self.table.remove_worker(worker_url);
    }

    /// 关闭所有 socket。没有一个 task 能在 join 后继续写入 table。
    pub async fn shutdown(&self) {
        let drained = {
            let mut handles = self.handles.lock().await;
            handles.drain().collect::<Vec<_>>()
        };
        let mut workers = HashSet::new();
        for ((worker_url, _), handle) in drained {
            workers.insert(worker_url);
            handle.cancel.cancel();
            if let Err(error) = handle.join.await {
                warn!(%error, "shortest-ttft: load subscriber did not join during shutdown");
            }
        }
        for worker_url in workers {
            self.table.remove_worker(&worker_url);
        }
    }
}

async fn run_load_subscriber(
    worker_url: String,
    dp_rank: u32,
    endpoint: String,
    topic: String,
    table: Arc<EngineLoadTable>,
    cancel: CancellationToken,
) {
    let mut sub = SubSocket::new();
    tokio::select! {
        _ = cancel.cancelled() => return,
        result = sub.connect(&endpoint) => {
            if let Err(error) = result {
                warn!(worker_url, dp_rank, %endpoint, %error, "shortest-ttft: load SUB connect failed");
                return;
            }
        }
    }
    tokio::select! {
        _ = cancel.cancelled() => return,
        result = sub.subscribe(&topic) => {
            if let Err(error) = result {
                warn!(worker_url, dp_rank, %endpoint, %error, "shortest-ttft: load SUB subscribe failed");
                return;
            }
        }
    }

    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            result = sub.recv() => match result {
                Ok(message) => {
                    if let Some(load) = decode_load_message(&message) {
                        table.set(&worker_url, dp_rank, load, Instant::now());
                    }
                }
                Err(error) => {
                    warn!(worker_url, dp_rank, %endpoint, %error, "shortest-ttft: load SUB receive failed");
                    tokio::task::yield_now().await;
                }
            }
        }
    }
}

/// Decode #34608's three frames. Sequence is validated for shape but never
/// used to filter a gauge: a restarted publisher legitimately restarts it.
fn decode_load_message(message: &ZmqMessage) -> Option<LoadStat> {
    if message.len() != 3 {
        return None;
    }
    let sequence: [u8; 8] = message.get(1)?.as_ref().try_into().ok()?;
    let _sequence = i64::from_be_bytes(sequence);
    let payload = message.get(2)?;
    let (tag, running, waiting, tokens, capacity, _rank): (
        String,
        u64,
        u64,
        u64,
        u64,
        Option<u64>,
    ) = rmp_serde::from_slice(payload.as_ref()).ok()?;
    (tag == "LoadStat").then_some(LoadStat {
        num_running_reqs: running,
        num_waiting_reqs: waiting,
        num_tokens: tokens,
        max_total_num_tokens: capacity,
    })
}

#[derive(Clone)]
struct ScoredWorker {
    worker: Arc<Worker>,
    ttft: u64,
    last_selected: u64,
}

/// 与 RTP-LLM `ShortestTTFTStrategy` 对齐的 Router policy。
pub struct ShortestTtftPolicy {
    tree: Arc<HashTree>,
    block_size: Arc<BlockSizeOracle>,
    engine_load: Arc<EngineLoadTable>,
    last_selected: Mutex<HashMap<String, u64>>,
    selection_clock: AtomicU64,
}

impl fmt::Debug for ShortestTtftPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShortestTtftPolicy")
            .field("tree_nodes", &self.tree.node_count())
            .field("engine_load", &self.engine_load)
            .finish()
    }
}

impl ShortestTtftPolicy {
    pub fn new(
        tree: Arc<HashTree>,
        block_size: Arc<BlockSizeOracle>,
        engine_load: Arc<EngineLoadTable>,
    ) -> Self {
        Self {
            tree,
            block_size,
            engine_load,
            last_selected: Mutex::new(HashMap::new()),
            selection_clock: AtomicU64::new(0),
        }
    }

    /// RTP 的预填充近似：`tokens - 0.7 * cache_hit_tokens`。
    fn estimate_prefill_time(tokens: u64, hit_cache_tokens: u64) -> u64 {
        tokens
            .saturating_mul(10)
            .saturating_sub(hit_cache_tokens.saturating_mul(7))
            / 10
    }

    /// V4 ingress 已完成异步 Indexer 查询时，signal 是缓存命中的唯一来源。
    /// `Some(empty)` 与 `None` 必须区分：前者代表 authoritative 的零命中，
    /// 绝不能回读本地 HashTree；后者仅保留给未配置 Indexer 的兼容调用路径。
    fn external_matched_tokens(&self, ctx: &SelectionContext<'_>) -> Option<HashMap<String, u64>> {
        let signal = ctx.external_prefix()?;
        let Some(tokens) = ctx.request_tokens() else {
            return Some(HashMap::new());
        };
        let Some(block_size) = self.block_size.get() else {
            return Some(HashMap::new());
        };
        let sgl_kv_indexer::PrefixOutcome::Matched { matches, .. } = &signal.outcome else {
            return Some(HashMap::new());
        };

        let input_tokens = tokens.len() as u64;
        let queried_blocks = signal.query_blocks as u64;
        let mut hit_tokens: HashMap<String, u64> = HashMap::new();
        for matched in matches {
            let tokens = (matched.matched_prefix_blocks as u64)
                .min(queried_blocks)
                .saturating_mul(block_size as u64)
                .min(input_tokens);
            hit_tokens
                .entry(matched.address.clone())
                .and_modify(|current| *current = (*current).max(tokens))
                .or_insert(tokens);
        }
        Some(hit_tokens)
    }

    fn matched_tokens(&self, ctx: &SelectionContext<'_>) -> HashMap<String, u64> {
        if let Some(matched_tokens) = self.external_matched_tokens(ctx) {
            return matched_tokens;
        }
        let Some(tokens) = ctx.request_tokens() else {
            return HashMap::new();
        };
        let Some(block_size) = self.block_size.get() else {
            return HashMap::new();
        };
        let hashes = if self.block_size.is_bigram() {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        if hashes.is_empty() {
            return HashMap::new();
        }

        let matched = self.tree.match_prefix(None, &hashes);
        let hit_tokens = (matched.matched_blocks as u64)
            .saturating_mul(block_size as u64)
            .min(tokens.len() as u64);
        matched
            .workers
            .into_iter()
            .map(|worker| (worker.url, hit_tokens))
            .collect()
    }

    fn choose(&self, mut scored: Vec<ScoredWorker>) -> Option<Arc<Worker>> {
        if scored.is_empty() {
            return None;
        }
        scored.sort_by(|left, right| {
            left.ttft
                .cmp(&right.ttft)
                .then_with(|| left.last_selected.cmp(&right.last_selected))
                .then_with(|| left.worker.url.cmp(&right.worker.url))
        });

        let candidate_count = (scored.len() * RTP_CANDIDATE_PERCENT_NUMERATOR
            / RTP_CANDIDATE_PERCENT_DENOMINATOR)
            .max(1);
        let candidates = &scored[..candidate_count];
        let min_ttft = candidates[0].ttft;
        let average =
            candidates.iter().map(|item| item.ttft as f64).sum::<f64>() / candidates.len() as f64;
        let stddev = (candidates
            .iter()
            .map(|item| (item.ttft as f64 - average).powi(2))
            .sum::<f64>()
            / candidates.len() as f64)
            .sqrt();
        let threshold =
            (average * RTP_TTFT_THRESHOLD_PERCENTAGE).max(stddev * RTP_STDDEV_THRESHOLD_FACTOR);
        let chosen = candidates
            .iter()
            .filter(|item| (item.ttft as f64 - min_ttft as f64).abs() <= threshold)
            .min_by(|left, right| {
                left.last_selected
                    .cmp(&right.last_selected)
                    .then_with(|| left.worker.url.cmp(&right.worker.url))
            })
            .unwrap_or(&candidates[0]);

        let selected_at = self.selection_clock.fetch_add(1, Ordering::Relaxed) + 1;
        self.last_selected
            .lock()
            .insert(chosen.worker.url.clone(), selected_at);
        Some(Arc::clone(&chosen.worker))
    }
}

impl Policy for ShortestTtftPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let matched_tokens = self.matched_tokens(ctx);
        let input_tokens = ctx.request_tokens().map_or(0, |tokens| tokens.len() as u64);
        let last_selected = self.last_selected.lock();
        let now = Instant::now();
        let scored = workers
            .iter()
            .map(|worker| {
                let hit_tokens = matched_tokens.get(&worker.url).copied().unwrap_or(0);
                let queue_pressure = self
                    .engine_load
                    .queue_pressure(&worker.url, now)
                    .unwrap_or_else(|| worker.active_load() as u64);
                ScoredWorker {
                    worker: Arc::clone(worker),
                    ttft: Self::estimate_prefill_time(input_tokens, hit_tokens)
                        .saturating_add(queue_pressure),
                    last_selected: last_selected.get(&worker.url).copied().unwrap_or(0),
                }
            })
            .collect();
        drop(last_selected);
        self.choose(scored)
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }
}
