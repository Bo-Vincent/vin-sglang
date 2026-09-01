// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! 基于本地 ZMQ KV radix tree 结果生成有界 Cache-Aware 候选集。

use crate::config::AffinityConfig;
use crate::policies::admission::FreshLoadLookup;
use crate::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree,
};
use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
use crate::policies::{
    estimate_matched_prefix_tokens, CacheCandidate, CacheCandidateProposal, Policy,
    PrefillProposal, ProposalKind, SelectionContext, SelectionProposal,
};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: AffinityConfig,
    tree: Arc<HashTree>,
    block_size_oracle: Arc<BlockSizeOracle>,
}

impl CacheAwarePolicy {
    pub fn new(
        config: AffinityConfig,
        tree: Arc<HashTree>,
        block_size_oracle: Arc<BlockSizeOracle>,
    ) -> Self {
        Self {
            config,
            tree,
            block_size_oracle,
        }
    }

    /// One local-tree descent returns the deepest contiguous cache prefix per
    /// worker URL. DP ranks of a worker collapse to their best depth, matching
    /// the established ZMQ radix-tree policy.
    fn local_prefix_depths(
        &self,
        ctx: &SelectionContext<'_>,
    ) -> Option<(usize, HashMap<String, u32>)> {
        let tokens = ctx.request_tokens().filter(|tokens| !tokens.is_empty())?;
        let block_size = self.block_size_oracle.get()?;
        let hashes = if self.block_size_oracle.is_bigram() {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        if hashes.is_empty() {
            return None;
        }

        let mut depths: HashMap<String, u32> = HashMap::new();
        for (worker, depth) in self.tree.prefix_depths(None, &hashes) {
            let depth = u32::try_from(depth).unwrap_or(u32::MAX);
            depths
                .entry(worker.url)
                .and_modify(|current| *current = (*current).max(depth))
                .or_insert(depth);
        }
        Some((hashes.len(), depths))
    }

    fn cache_candidate_proposal(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<CacheCandidateProposal> {
        let input_tokens = ctx.input_tokens()?;
        let (query_blocks, matched_by_url) = self.local_prefix_depths(ctx)?;
        if workers.is_empty() {
            return None;
        }

        let mut candidates = Vec::new();
        for worker in workers {
            let Some(matched_prefix_blocks) = matched_by_url.get(&worker.url).copied() else {
                continue;
            };
            if matched_prefix_blocks == 0 {
                continue;
            }
            let matched_prefix_tokens =
                estimate_matched_prefix_tokens(input_tokens, query_blocks, matched_prefix_blocks);
            if !self.passes_cache_gate(input_tokens, matched_prefix_tokens) {
                continue;
            }
            let candidate = CacheCandidate {
                worker: Arc::clone(worker),
                matched_prefix_tokens,
                uncached_tokens: input_tokens.saturating_sub(matched_prefix_tokens),
                candidate_range_id: ctx.candidate_range_id().to_string(),
                max_pending_prefill_tokens: None,
            };
            let Some(candidate) = ctx.bind_prefill_cache_candidate(candidate) else {
                continue;
            };
            candidates.push(candidate);
        }

        let limit = self.candidate_limit(workers.len());
        if limit == 0 {
            return None;
        }
        let loads = FreshLoadLookup::new(
            ctx.load_snapshot(),
            candidates.iter().map(|candidate| &candidate.worker),
        );
        if candidates.len() > limit {
            candidates.select_nth_unstable_by(limit, |left, right| {
                compare_candidate_seed(left, right, &loads)
            });
            candidates.truncate(limit);
        }
        candidates.sort_by(|left, right| compare_candidate_seed(left, right, &loads));
        if candidates.is_empty() {
            return None;
        }
        Some(CacheCandidateProposal {
            candidates,
            cache_switch_margin_tokens: self.config.cache_switch_margin_tokens,
            enable_pressure_guard: self.config.pressure_guard,
            pressure_abs_threshold_tokens: self.config.pressure_abs_threshold_tokens,
            pressure_abs_threshold_ms: self.config.pressure_abs_threshold_ms,
            pressure_rel_threshold: self.config.pressure_rel_threshold,
        })
    }

    fn passes_cache_gate(&self, input_tokens: u64, matched_prefix_tokens: u64) -> bool {
        self.config
            .cache_affinity_min_matched_tokens
            .is_none_or(|minimum| matched_prefix_tokens >= minimum)
            && self
                .config
                .cache_affinity_min_match_ratio
                .is_none_or(|minimum| {
                    input_tokens > 0
                        && matched_prefix_tokens as f64 / input_tokens as f64 >= minimum
                })
    }

    fn candidate_limit(&self, worker_count: usize) -> usize {
        let proportional = (self.config.cache_candidate_ratio.clamp(0.0, 1.0) * worker_count as f64)
            .ceil() as usize;
        worker_count
            .min(self.config.cache_candidate_max_workers)
            .min(self.config.cache_candidate_min_workers.max(proportional))
    }
}

fn compare_candidate_seed(
    left: &CacheCandidate,
    right: &CacheCandidate,
    loads: &FreshLoadLookup<'_>,
) -> Ordering {
    right
        .matched_prefix_tokens
        .cmp(&left.matched_prefix_tokens)
        .then_with(|| loads.compare_prefill_pressure(&left.worker, &right.worker))
        .then_with(|| left.worker.id.0.cmp(&right.worker.id.0))
}

impl Policy for CacheAwarePolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        self.propose(workers, ctx).map(|proposal| proposal.primary)
    }

    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        match self.propose_prefill(workers, ctx)? {
            PrefillProposal::Pair(proposal) => Some(proposal),
            PrefillProposal::CacheCandidates(proposal) => {
                let candidate = proposal.candidates.into_iter().next()?;
                Some(
                    SelectionProposal::primary(candidate.worker)
                        .with_kind(ProposalKind::CacheAffinity),
                )
            }
            PrefillProposal::ShortestTtftCandidates(_) => None,
        }
    }

    fn propose_prefill(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillProposal> {
        if ctx.affinity_lookup_enabled() {
            if let Some(proposal) = self.cache_candidate_proposal(workers, ctx) {
                return Some(PrefillProposal::CacheCandidates(proposal));
            }
        }
        PowerOfTwoChoicesPolicy::new()
            .propose(workers, ctx)
            .map(PrefillProposal::Pair)
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_token_estimate_caps_untrusted_block_count() {
        assert_eq!(estimate_matched_prefix_tokens(80, 8, 99), 80);
    }
}
