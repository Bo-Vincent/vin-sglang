// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! 基于 ingress 已完成 KV Indexer 查询的 Cache-Aware policy。
//!
//! 这个 policy 不访问 indexer 网络；它只消费 [`ExternalPrefixSignal`]，并把
//! 结果和 Router 提供的候选域求交。缓存命中仍要经过共享 Admission 与 Guard。

use crate::config::{AffinityConfig, AffinityMode};
use crate::policies::admission::compare_prefill_pressure;
use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
use crate::policies::session_aware::affinity_backup;
use crate::policies::{
    ExternalPrefixSignal, GuardHints, Policy, ProposalKind, SelectionContext, SelectionProposal,
};
use crate::workers::Worker;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Debug)]
pub struct CacheAwarePolicy {
    config: AffinityConfig,
}

impl CacheAwarePolicy {
    pub fn new(config: AffinityConfig) -> Self {
        Self { config }
    }

    fn affinity_proposal(
        &self,
        primary: Arc<Worker>,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
        signal: &ExternalPrefixSignal,
        matched_prefix_blocks: u32,
    ) -> SelectionProposal {
        let affinity_key = cache_affinity_key(ctx, signal.query_blocks, matched_prefix_blocks);
        let backup = affinity_backup(
            workers,
            &primary,
            &affinity_key,
            ctx.candidate_range_id(),
            self.config.stable_pair,
            ctx,
        );
        let proposal = match backup {
            Some(backup) => SelectionProposal::with_backup(primary, backup),
            None => SelectionProposal::primary(primary),
        };
        proposal
            .with_kind(ProposalKind::CacheAffinity)
            .with_guard_hints(GuardHints {
                matched_prefix_tokens: estimate_matched_prefix_tokens(
                    ctx.input_tokens(),
                    signal.query_blocks,
                    matched_prefix_blocks,
                ),
                enable_cache_benefit: self.config.cache_benefit,
                enable_pressure_guard: self.config.pressure_guard
                    && self.config.mode == AffinityMode::Soft,
                pressure_abs_threshold_tokens: self.config.pressure_abs_threshold_tokens,
                pressure_rel_threshold: self.config.pressure_rel_threshold,
            })
    }
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
        if !ctx.affinity_lookup_enabled() {
            return PowerOfTwoChoicesPolicy::new().propose(workers, ctx);
        }
        let Some((primary, matched_prefix_blocks, signal)) = best_prefix_holder(workers, ctx)
        else {
            return PowerOfTwoChoicesPolicy::new().propose(workers, ctx);
        };
        Some(self.affinity_proposal(primary, workers, ctx, signal, matched_prefix_blocks))
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        true
    }

    fn is_bucket_affinity_policy(&self) -> bool {
        true
    }
}

fn best_prefix_holder<'a>(
    workers: &[Arc<Worker>],
    ctx: &'a SelectionContext<'_>,
) -> Option<(Arc<Worker>, u32, &'a ExternalPrefixSignal)> {
    let signal = ctx.external_prefix()?;
    let sgl_kv_indexer::PrefixOutcome::Matched { matches, .. } = &signal.outcome else {
        return None;
    };
    if signal.query_blocks == 0 {
        return None;
    }
    let best_blocks = matches
        .iter()
        .filter(|entry| workers.iter().any(|worker| worker.url == entry.address))
        .map(|entry| entry.matched_prefix_blocks)
        .max()?;
    let primary = workers
        .iter()
        .filter(|worker| {
            matches.iter().any(|entry| {
                entry.address == worker.url && entry.matched_prefix_blocks == best_blocks
            })
        })
        .min_by(|left, right| compare_prefill_pressure(left, right, ctx.load_snapshot()))
        .cloned()?;
    Some((primary, best_blocks, signal))
}

fn estimate_matched_prefix_tokens(
    input_tokens: Option<u64>,
    query_blocks: usize,
    matched_prefix_blocks: u32,
) -> Option<u64> {
    let input_tokens = input_tokens?;
    let query_blocks = u64::try_from(query_blocks).ok()?.max(1);
    Some(input_tokens.saturating_mul(u64::from(matched_prefix_blocks)) / query_blocks)
}

/// cache 命中的确切 token 前缀不可从当前 indexer block 协议直接得到。这里使用
/// request token 序列和命中块数构造稳定 key，仅用于 stable pair 的 backup；
/// 它不参与 prefix 真实性判断。
fn cache_affinity_key(
    ctx: &SelectionContext<'_>,
    query_blocks: usize,
    matched_prefix_blocks: u32,
) -> String {
    let mut hasher = DefaultHasher::new();
    matched_prefix_blocks.hash(&mut hasher);
    if let Some(tokens) = ctx.request_tokens() {
        let matched_tokens = if query_blocks == 0 {
            0
        } else {
            tokens.len().saturating_mul(matched_prefix_blocks as usize) / query_blocks
        };
        tokens[..matched_tokens.min(tokens.len())].hash(&mut hasher);
    }
    format!(
        "blocks:{matched_prefix_blocks}:hash:{:016x}",
        hasher.finish()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::ModelId;

    #[test]
    fn stable_pair_key_is_bounded_and_ignores_the_uncached_suffix() {
        let model = ModelId("model".into());
        let first = [1, 2, 3, 4];
        let second = [1, 2, 9, 9];
        let first_ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&first));
        let second_ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&second));

        let first_key = cache_affinity_key(&first_ctx, 2, 1);
        let second_key = cache_affinity_key(&second_ctx, 2, 1);
        assert_eq!(
            first_key, second_key,
            "the same cached prefix must keep the same stable backup"
        );

        let long_tokens = vec![7; 100_000];
        let long_ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&long_tokens));
        assert!(
            cache_affinity_key(&long_ctx, 2, 1).len() <= 64,
            "stable-pair bookkeeping must not allocate in proportion to prompt length"
        );
    }
}
