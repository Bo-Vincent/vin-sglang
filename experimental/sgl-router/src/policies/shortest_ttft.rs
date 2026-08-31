// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! 图中 Shortest-TTFT baseline 的候选排序与并发热点规避。

use crate::discovery::WorkerId;
use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
use crate::policies::{
    estimate_matched_prefix_tokens, CacheCandidate, Policy, PrefillProposal, SelectionContext,
    ShortestTtftCandidateProposal,
};
use crate::workers::Worker;
use dashmap::DashMap;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;

/// 图中 outstanding guard 的固定实验阈值。它不是当前 Cache-Aware 的
/// pressure guard，也不按 policy 调参。
pub const DEFAULT_OUTSTANDING_UNCACHED_TOKENS_THRESHOLD: u64 = 16_384;

/// 已通过 Router hard admission 的一个 baseline 候选。
#[derive(Debug, Clone, PartialEq)]
pub struct ShortestTtftCandidate {
    pub worker_id: String,
    pub matched_prefix_tokens: u64,
    pub uncached_tokens: u64,
    pub queue_ms: f64,
    pub outstanding_uncached_tokens: u64,
}

impl ShortestTtftCandidate {
    /// 图中模型：每个未命中 token 按 1ms 估算 Prefill，再加排队时间。
    pub fn estimated_ttft_ms(&self) -> f64 {
        self.uncached_tokens as f64 + self.queue_ms
    }
}

/// TTFT 排名中的 top 候选与相似阈值。
#[derive(Debug, Clone, PartialEq)]
pub struct ShortestTtftRanking {
    pub top_candidates: Vec<ShortestTtftCandidate>,
    pub min_ttft_ms: f64,
    pub similarity_margin_ms: f64,
}

/// 最终选中的 baseline 候选和其估计值。
#[derive(Debug, Clone, PartialEq)]
pub struct ShortestTtftSelection {
    pub worker_id: String,
    pub estimated_ttft_ms: f64,
    pub outstanding_guard_fallback: bool,
    pub cas_retries: u32,
}

/// 独立顶层 Shortest-TTFT policy 的并发状态。
#[derive(Debug, Default)]
pub struct ShortestTtftPolicy {
    last_selected: DashMap<WorkerId, AtomicU64>,
    next_selection_stamp: AtomicU64,
}

impl ShortestTtftPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    fn candidate_proposal(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<ShortestTtftCandidateProposal> {
        let input_tokens = ctx.input_tokens()?;
        let signal = ctx.external_prefix()?;
        if signal.query_blocks == 0 || workers.is_empty() {
            return None;
        }

        let matched_by_address: HashMap<&str, u32> = match &signal.outcome {
            sgl_kv_indexer::PrefixOutcome::Empty => HashMap::new(),
            sgl_kv_indexer::PrefixOutcome::Matched { matches, .. } => matches
                .iter()
                .map(|entry| (entry.address.as_str(), entry.matched_prefix_blocks))
                .collect(),
        };
        let candidates = workers
            .iter()
            .filter_map(|worker| {
                let matched_prefix_tokens = matched_by_address
                    .get(worker.url.as_str())
                    .copied()
                    .map(|blocks| {
                        estimate_matched_prefix_tokens(input_tokens, signal.query_blocks, blocks)
                    })
                    .unwrap_or(0);
                let candidate = CacheCandidate {
                    worker: Arc::clone(worker),
                    matched_prefix_tokens,
                    uncached_tokens: input_tokens.saturating_sub(matched_prefix_tokens),
                    candidate_range_id: ctx.candidate_range_id().to_string(),
                    max_pending_prefill_tokens: None,
                };
                ctx.bind_prefill_cache_candidate(candidate)
            })
            .collect::<Vec<_>>();
        (!candidates.is_empty()).then_some(ShortestTtftCandidateProposal {
            candidates,
            outstanding_uncached_tokens_threshold: DEFAULT_OUTSTANDING_UNCACHED_TOKENS_THRESHOLD,
        })
    }

    fn try_claim(&self, worker_id: &WorkerId) -> bool {
        let stamp = self
            .next_selection_stamp
            .fetch_add(1, AtomicOrdering::Relaxed)
            .saturating_add(1);
        let slot = self
            .last_selected
            .entry(worker_id.clone())
            .or_insert_with(|| AtomicU64::new(0));
        let observed = slot.load(AtomicOrdering::Acquire);
        slot.compare_exchange(
            observed,
            stamp,
            AtomicOrdering::AcqRel,
            AtomicOrdering::Acquire,
        )
        .is_ok()
    }
}

impl Policy for ShortestTtftPolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        PowerOfTwoChoicesPolicy::new().select(workers, ctx)
    }

    fn propose_prefill(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillProposal> {
        self.candidate_proposal(workers, ctx)
            .map(PrefillProposal::ShortestTtftCandidates)
            .or_else(|| {
                PowerOfTwoChoicesPolicy::new()
                    .propose(workers, ctx)
                    .map(PrefillProposal::Pair)
            })
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }

    fn try_claim_shortest_ttft(&self, worker_id: &WorkerId) -> bool {
        self.try_claim(worker_id)
    }
}

fn compare_ttft(left: &ShortestTtftCandidate, right: &ShortestTtftCandidate) -> Ordering {
    left.estimated_ttft_ms()
        .total_cmp(&right.estimated_ttft_ms())
        .then_with(|| left.worker_id.cmp(&right.worker_id))
}

fn top_candidate_count(total: usize) -> usize {
    if total <= 3 {
        total
    } else {
        ((total * 3).div_ceil(10)).max(2)
    }
}

/// 按 baseline 的 top-30% 与 similarity band 规则对候选排序。
pub fn rank_shortest_ttft(candidates: &[ShortestTtftCandidate]) -> Option<ShortestTtftRanking> {
    let mut ranked = candidates.to_vec();
    ranked.sort_by(compare_ttft);
    let count = top_candidate_count(ranked.len());
    let top_candidates = ranked.into_iter().take(count).collect::<Vec<_>>();
    let fastest = top_candidates.first()?;
    let min_ttft_ms = fastest.estimated_ttft_ms();
    let mean = top_candidates
        .iter()
        .map(ShortestTtftCandidate::estimated_ttft_ms)
        .sum::<f64>()
        / top_candidates.len() as f64;
    let stddev = (top_candidates
        .iter()
        .map(|candidate| {
            let delta = candidate.estimated_ttft_ms() - mean;
            delta * delta
        })
        .sum::<f64>()
        / top_candidates.len() as f64)
        .sqrt();
    Some(ShortestTtftRanking {
        top_candidates,
        min_ttft_ms,
        similarity_margin_ms: (min_ttft_ms * 0.2).max(stddev * 0.5),
    })
}

fn choose_similar_candidate(ranking: &ShortestTtftRanking) -> Option<&ShortestTtftCandidate> {
    let fastest = ranking.top_candidates.first()?;
    let max_ttft_ms = ranking.min_ttft_ms + ranking.similarity_margin_ms;
    let cache_leader = ranking
        .top_candidates
        .iter()
        .take_while(|candidate| candidate.estimated_ttft_ms() <= max_ttft_ms)
        .max_by(|left, right| {
            left.matched_prefix_tokens
                .cmp(&right.matched_prefix_tokens)
                .then_with(|| compare_ttft(right, left))
        })?;
    if cache_leader.matched_prefix_tokens > fastest.matched_prefix_tokens {
        Some(cache_leader)
    } else {
        Some(fastest)
    }
}

/// Outstanding guard 是软筛选：所有 worker 都超阈值时恢复完整候选域。
pub fn apply_outstanding_guard(
    candidates: &[ShortestTtftCandidate],
    outstanding_threshold_tokens: u64,
) -> Vec<ShortestTtftCandidate> {
    let guarded = candidates
        .iter()
        .filter(|candidate| {
            candidate
                .outstanding_uncached_tokens
                .saturating_add(candidate.uncached_tokens)
                <= outstanding_threshold_tokens
        })
        .cloned()
        .collect::<Vec<_>>();
    if guarded.is_empty() {
        candidates.to_vec()
    } else {
        guarded
    }
}

/// 选择 TTFT winner；CAS 失败时将该热点候选移除并重算下一 winner。
pub fn select_shortest_ttft(
    candidates: &[ShortestTtftCandidate],
    outstanding_threshold_tokens: u64,
    mut try_claim: impl FnMut(&str) -> bool,
) -> Option<ShortestTtftSelection> {
    let guarded = apply_outstanding_guard(candidates, outstanding_threshold_tokens);
    let outstanding_guard_fallback = guarded.len() == candidates.len()
        && candidates.iter().all(|candidate| {
            candidate
                .outstanding_uncached_tokens
                .saturating_add(candidate.uncached_tokens)
                > outstanding_threshold_tokens
        });
    let mut remaining = guarded;
    let mut cas_retries = 0;
    while let Some(ranking) = rank_shortest_ttft(&remaining) {
        let winner = choose_similar_candidate(&ranking)?;
        if try_claim(&winner.worker_id) {
            return Some(ShortestTtftSelection {
                worker_id: winner.worker_id.clone(),
                estimated_ttft_ms: winner.estimated_ttft_ms(),
                outstanding_guard_fallback,
                cas_retries,
            });
        }
        cas_retries = cas_retries.saturating_add(1);
        remaining.retain(|candidate| candidate.worker_id != winner.worker_id);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::ExternalPrefixSignal;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".into())],
            bootstrap_port: None,
        }))
    }

    fn candidate(
        worker_id: &str,
        matched_prefix_tokens: u64,
        uncached_tokens: u64,
        queue_ms: f64,
        outstanding_uncached_tokens: u64,
    ) -> ShortestTtftCandidate {
        ShortestTtftCandidate {
            worker_id: worker_id.to_string(),
            matched_prefix_tokens,
            uncached_tokens,
            queue_ms,
            outstanding_uncached_tokens,
        }
    }

    #[test]
    fn scores_uncached_prefill_plus_queue() {
        let candidates = vec![
            candidate("cache", 900, 100, 200.0, 0),
            candidate("fast", 100, 200, 10.0, 0),
        ];

        let selected = select_shortest_ttft(&candidates, 1_000, |_| true).unwrap();

        assert_eq!(selected.worker_id, "fast");
        assert_eq!(selected.estimated_ttft_ms, 210.0);
    }

    #[test]
    fn uses_top_thirty_percent_with_minimum_of_two() {
        let candidates = vec![
            candidate("w0", 0, 100, 0.0, 0),
            candidate("w1", 0, 110, 0.0, 0),
            candidate("w2", 0, 120, 0.0, 0),
            candidate("w3", 0, 130, 0.0, 0),
            candidate("w4", 0, 140, 0.0, 0),
            candidate("w5", 0, 150, 0.0, 0),
            candidate("w6", 0, 160, 0.0, 0),
            candidate("w7", 0, 170, 0.0, 0),
        ];

        let ranked = rank_shortest_ttft(&candidates).unwrap();

        assert_eq!(ranked.top_candidates.len(), 3);
        assert_eq!(ranked.top_candidates[0].worker_id, "w0");
        assert_eq!(ranked.top_candidates[2].worker_id, "w2");
    }

    #[test]
    fn prefers_more_cache_only_inside_similar_ttft_band() {
        let candidates = vec![
            candidate("fast", 100, 200, 0.0, 0),
            candidate("cache", 400, 210, 0.0, 0),
            candidate("slow", 900, 300, 0.0, 0),
            candidate("slower", 900, 400, 0.0, 0),
        ];

        let selected = select_shortest_ttft(&candidates, 1_000, |_| true).unwrap();

        assert_eq!(selected.worker_id, "cache");
    }

    #[test]
    fn falls_back_to_full_set_when_outstanding_guard_filters_every_candidate() {
        let candidates = vec![
            candidate("w0", 100, 100, 0.0, 950),
            candidate("w1", 200, 200, 0.0, 900),
        ];

        let guarded = apply_outstanding_guard(&candidates, 100);

        assert_eq!(guarded.len(), 2);
        assert_eq!(guarded[0].worker_id, "w0");
        assert_eq!(guarded[1].worker_id, "w1");
    }

    #[test]
    fn retries_next_winner_when_hotspot_cas_loses_race() {
        let candidates = vec![
            candidate("first", 0, 100, 0.0, 0),
            candidate("second", 0, 110, 0.0, 0),
        ];
        let mut attempts = Vec::new();

        let selected = select_shortest_ttft(&candidates, 1_000, |worker_id| {
            attempts.push(worker_id.to_string());
            worker_id == "second"
        })
        .unwrap();

        assert_eq!(attempts, vec!["first", "second"]);
        assert_eq!(selected.worker_id, "second");
        assert_eq!(selected.cas_retries, 1);
    }

    #[test]
    fn marks_full_outstanding_guard_fallback() {
        let candidates = vec![
            candidate("w0", 0, 100, 0.0, 950),
            candidate("w1", 0, 200, 0.0, 900),
        ];

        let selected = select_shortest_ttft(&candidates, 100, |_| true).unwrap();

        assert!(selected.outstanding_guard_fallback);
    }

    #[test]
    fn authoritative_empty_indexer_result_keeps_all_zero_hit_workers() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second")];
        let signal = ExternalPrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Empty,
            query_blocks: 8,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(8_000)
            .with_external_prefix(Some(&signal));
        let policy = ShortestTtftPolicy::new();

        let PrefillProposal::ShortestTtftCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("an authoritative empty answer still has baseline candidates")
        else {
            panic!("an authoritative empty answer must not degrade to P2");
        };

        assert_eq!(proposal.candidates.len(), 2);
        assert!(proposal
            .candidates
            .iter()
            .all(|candidate| candidate.matched_prefix_tokens == 0));
        assert!(proposal
            .candidates
            .iter()
            .all(|candidate| candidate.uncached_tokens == 8_000));
    }

    #[test]
    fn no_external_prefix_signal_degrades_to_power_of_two() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second")];
        let ctx = SelectionContext::new(&model, None).with_input_tokens(8_000);
        let policy = ShortestTtftPolicy::new();

        let PrefillProposal::Pair(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("no external signal must retain a P2 fallback")
        else {
            panic!("no external signal must not become synthetic cache candidates");
        };

        assert_eq!(proposal.kind, crate::policies::ProposalKind::PowerOfTwo);
    }
}
