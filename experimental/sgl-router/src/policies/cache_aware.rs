// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::BoundedCacheAwareConfig;
use crate::policies::active_load::ActiveLoadRegistry;
use crate::policies::bounded::{
    power_of_two_excluding, pressure_guard_trips, stable_backup, AffinityLoadView,
};
use crate::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree,
};
use crate::policies::{request_tokens_for, Policy, SelectionContext};
use crate::server::metrics::MetricsRegistry;
use crate::tokenizer::TokenizerRegistry;
use crate::workers::Worker;
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

/// Bounded Cache-Aware routing. Unlike `cache_aware_zmq`, this policy always
/// reasons over one cache primary and one bounded backup.
pub struct CacheAwarePolicy {
    config: BoundedCacheAwareConfig,
    tree: Arc<HashTree>,
    tokenizers: Arc<TokenizerRegistry>,
    block_size_oracle: Arc<BlockSizeOracle>,
    load: AffinityLoadView,
    metrics: OnceLock<Arc<MetricsRegistry>>,
}

impl CacheAwarePolicy {
    pub fn new(
        config: BoundedCacheAwareConfig,
        tree: Arc<HashTree>,
        tokenizers: Arc<TokenizerRegistry>,
        block_size_oracle: Arc<BlockSizeOracle>,
        active: Arc<ActiveLoadRegistry>,
    ) -> Self {
        Self {
            config,
            tree,
            tokenizers,
            block_size_oracle,
            load: AffinityLoadView::new(active),
            metrics: OnceLock::new(),
        }
    }

    fn record(&self, reason: &'static str) {
        if let Some(metrics) = self.metrics.get() {
            metrics.record_policy_decision("cache_aware", reason);
        }
    }

    fn fallback(&self, workers: &[Arc<Worker>]) -> Option<Arc<Worker>> {
        let selected = power_of_two_excluding(workers, None, &self.load);
        if selected.is_some() {
            self.record("no_affinity");
        }
        selected
    }

    fn cache_saved_work(
        matched_blocks: usize,
        block_size: usize,
        token_len: usize,
        is_bigram: bool,
    ) -> usize {
        let logical_work = matched_blocks
            .saturating_mul(block_size)
            .saturating_add(usize::from(is_bigram && matched_blocks > 0));
        logical_work.min(token_len)
    }
}

impl Policy for CacheAwarePolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        if workers.is_empty() {
            return None;
        }

        let fallback_ids;
        let tokens: &[u32] = match ctx.request_tokens() {
            Some(tokens) if !tokens.is_empty() => tokens,
            _ => {
                let Some(body) = ctx.request_body() else {
                    return self.fallback(workers);
                };
                let Ok(value) = serde_json::from_slice::<serde_json::Value>(body) else {
                    return self.fallback(workers);
                };
                let Some(routed) = request_tokens_for(&self.tokenizers, ctx.model(), &value) else {
                    return self.fallback(workers);
                };
                fallback_ids = routed.ids;
                &fallback_ids
            }
        };
        let Some(block_size) = self.block_size_oracle.get() else {
            return self.fallback(workers);
        };
        let is_bigram = self.block_size_oracle.is_bigram();
        let block_hashes = if is_bigram {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        if block_hashes.is_empty() {
            return self.fallback(workers);
        }

        let eligible_urls: HashSet<&str> =
            workers.iter().map(|worker| worker.url.as_str()).collect();
        let matched = self
            .tree
            .match_prefix_for_workers(None, &block_hashes, &eligible_urls);
        if matched.matched_blocks == 0 || matched.workers.is_empty() {
            return self.fallback(workers);
        }
        let matched_urls: HashSet<&str> = matched
            .workers
            .iter()
            .map(|worker| worker.url.as_str())
            .collect();
        let holders: Vec<Arc<Worker>> = workers
            .iter()
            .filter(|worker| matched_urls.contains(worker.url.as_str()))
            .cloned()
            .collect();
        let Some(primary) = power_of_two_excluding(&holders, None, &self.load) else {
            return self.fallback(workers);
        };

        let mut affinity_key = Vec::with_capacity(matched.matched_blocks * 8);
        for hash in block_hashes.iter().take(matched.matched_blocks) {
            affinity_key.extend_from_slice(&hash.to_le_bytes());
        }
        let backup = if self.config.stable_pair {
            stable_backup(workers, primary.url.as_str(), &affinity_key)
        } else {
            power_of_two_excluding(workers, Some(primary.url.as_str()), &self.load)
        };
        let Some(backup) = backup else {
            self.record("primary_only");
            return Some(primary);
        };

        let cache_saved_work = Self::cache_saved_work(
            matched.matched_blocks,
            block_size as usize,
            tokens.len(),
            is_bigram,
        );
        let remaining_uncached_work = tokens.len().saturating_sub(cache_saved_work);
        if self.config.cache_benefit && cache_saved_work <= remaining_uncached_work {
            self.record("cache_benefit");
            return Some(backup);
        }

        if self.config.pressure_guard {
            let (primary_pressure, backup_pressure) = self.load.pair_pressure(&primary, &backup);
            if pressure_guard_trips(
                primary_pressure,
                backup_pressure,
                self.config.pressure_abs_threshold,
                self.config.pressure_rel_threshold,
            ) {
                self.record("pressure_guard");
                return Some(backup);
            }
        }

        self.record("cache_primary");
        Some(primary)
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.metrics.set(metrics);
    }
}

impl std::fmt::Debug for CacheAwarePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheAwarePolicy")
            .field("config", &self.config)
            .field("tree_nodes", &self.tree.node_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::BoundedCacheAwareConfig;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::active_load::ActiveLoadRegistry;
    use crate::policies::kv_events::{compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId};
    use crate::policies::{Policy, SelectionContext};
    use crate::tokenizer::TokenizerRegistry;
    use crate::workers::Worker;
    use std::sync::Arc;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }))
    }

    fn policy(
        config: BoundedCacheAwareConfig,
        tree: Arc<HashTree>,
        active: Arc<ActiveLoadRegistry>,
    ) -> CacheAwarePolicy {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(4).unwrap();
        CacheAwarePolicy::new(
            config,
            tree,
            Arc::new(TokenizerRegistry::default()),
            oracle,
            active,
        )
    }

    fn seed(tree: &HashTree, worker: &Worker, tokens: &[u32], blocks: usize) {
        let hashes = compute_block_hashes(tokens, 4);
        tree.insert(
            &KvWorkerId::new(worker.url.clone(), 0),
            None,
            &hashes[..blocks],
        );
    }

    fn select(policy: &CacheAwarePolicy, workers: &[Arc<Worker>], tokens: &[u32]) -> Arc<Worker> {
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(tokens));
        policy.select(workers, &ctx).unwrap()
    }

    #[test]
    fn cache_benefit_keeps_primary_when_saved_work_is_larger() {
        let tree = Arc::new(HashTree::new());
        let active = ActiveLoadRegistry::with_defaults();
        let workers = vec![worker("w0"), worker("w1")];
        let tokens: Vec<u32> = (0..12).collect();
        seed(&tree, &workers[0], &tokens, 2); // 8 saved, 4 uncached
        let policy = policy(BoundedCacheAwareConfig::default(), tree, active);

        assert_eq!(select(&policy, &workers, &tokens).url, workers[0].url);
    }

    #[test]
    fn cache_benefit_uses_backup_when_uncached_work_is_not_smaller() {
        let tree = Arc::new(HashTree::new());
        let active = ActiveLoadRegistry::with_defaults();
        let workers = vec![worker("w0"), worker("w1")];
        let tokens: Vec<u32> = (0..12).collect();
        seed(&tree, &workers[0], &tokens, 1); // 4 saved, 8 uncached
        let policy = policy(BoundedCacheAwareConfig::default(), tree, active);

        assert_eq!(select(&policy, &workers, &tokens).url, workers[1].url);
    }

    #[test]
    fn pressure_guard_escapes_a_materially_hot_cache_primary() {
        let tree = Arc::new(HashTree::new());
        let active = ActiveLoadRegistry::with_defaults();
        let workers = vec![worker("w0"), worker("w1")];
        let tokens: Vec<u32> = (0..12).collect();
        seed(&tree, &workers[0], &tokens, 2);
        let config = BoundedCacheAwareConfig {
            pressure_abs_threshold: 100,
            ..BoundedCacheAwareConfig::default()
        };
        let policy = policy(config, tree, Arc::clone(&active));
        let _hot = active.register(workers[0].id.clone(), workers[0].url.clone(), 1000, 0);
        let _cool = active.register(workers[1].id.clone(), workers[1].url.clone(), 1, 0);

        assert_eq!(select(&policy, &workers, &tokens).url, workers[1].url);
    }

    #[test]
    fn missing_request_tokens_degrades_to_power_of_two() {
        let tree = Arc::new(HashTree::new());
        let active = ActiveLoadRegistry::with_defaults();
        let workers = vec![worker("w0"), worker("w1")];
        let policy = policy(BoundedCacheAwareConfig::default(), tree, active);
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);

        assert!(policy.select(&workers, &ctx).is_some());
    }

    #[test]
    fn disabled_cache_benefit_keeps_the_cache_primary() {
        let tree = Arc::new(HashTree::new());
        let active = ActiveLoadRegistry::with_defaults();
        let workers = vec![worker("w0"), worker("w1")];
        let tokens: Vec<u32> = (0..12).collect();
        seed(&tree, &workers[0], &tokens, 1);
        let config = BoundedCacheAwareConfig {
            cache_benefit: false,
            pressure_guard: false,
            ..BoundedCacheAwareConfig::default()
        };
        let policy = policy(config, tree, active);

        assert_eq!(select(&policy, &workers, &tokens).url, workers[0].url);
    }

    #[test]
    fn pressure_guard_requires_both_thresholds_at_policy_level() {
        let tree = Arc::new(HashTree::new());
        let active = ActiveLoadRegistry::with_defaults();
        let workers = vec![worker("w0"), worker("w1")];
        let tokens: Vec<u32> = (0..12).collect();
        seed(&tree, &workers[0], &tokens, 2);
        let config = BoundedCacheAwareConfig {
            pressure_abs_threshold: 100,
            pressure_rel_threshold: 1.5,
            ..BoundedCacheAwareConfig::default()
        };
        let policy = policy(config, tree, Arc::clone(&active));
        let _primary = active.register(workers[0].id.clone(), workers[0].url.clone(), 1000, 0);
        let _backup = active.register(workers[1].id.clone(), workers[1].url.clone(), 800, 0);

        assert_eq!(select(&policy, &workers, &tokens).url, workers[0].url);
    }

    #[test]
    fn stable_pair_produces_the_same_backup_for_the_same_prefix() {
        let tree = Arc::new(HashTree::new());
        let active = ActiveLoadRegistry::with_defaults();
        let workers = vec![worker("w0"), worker("w1"), worker("w2")];
        let tokens: Vec<u32> = (0..12).collect();
        seed(&tree, &workers[0], &tokens, 1);
        let config = BoundedCacheAwareConfig {
            stable_pair: true,
            ..BoundedCacheAwareConfig::default()
        };
        let policy = policy(config, tree, active);
        let first = select(&policy, &workers, &tokens);

        assert_ne!(first.url, workers[0].url);
        for _ in 0..20 {
            assert_eq!(select(&policy, &workers, &tokens).url, first.url);
        }
    }

    #[test]
    fn cache_benefit_short_circuits_with_one_final_reason() {
        let tree = Arc::new(HashTree::new());
        let active = ActiveLoadRegistry::with_defaults();
        let workers = vec![worker("w0"), worker("w1")];
        let tokens: Vec<u32> = (0..12).collect();
        seed(&tree, &workers[0], &tokens, 1);
        let policy = policy(BoundedCacheAwareConfig::default(), tree, active);
        let metrics = MetricsRegistry::new();
        policy.attach_metrics(Arc::clone(&metrics));

        assert_eq!(select(&policy, &workers, &tokens).url, workers[1].url);
        let rendered = metrics.render();
        assert!(rendered.contains(
            r#"sgl_router_policy_decisions_total{policy="cache_aware",reason="cache_benefit"} 1"#
        ));
        assert!(!rendered.contains(
            r#"sgl_router_policy_decisions_total{policy="cache_aware",reason="pressure_guard"}"#
        ));
    }

    #[test]
    fn bigram_saved_work_includes_the_shared_boundary_token() {
        assert_eq!(CacheAwarePolicy::cache_saved_work(2, 4, 12, true), 9);
        assert_eq!(CacheAwarePolicy::cache_saved_work(2, 4, 12, false), 8);
        assert_eq!(CacheAwarePolicy::cache_saved_work(3, 4, 12, true), 12);
    }

    #[test]
    fn falls_back_to_a_shorter_prefix_on_an_eligible_worker() {
        let tree = Arc::new(HashTree::new());
        let active = ActiveLoadRegistry::with_defaults();
        let unhealthy = worker("unhealthy");
        let healthy = worker("healthy");
        let no_cache = worker("no-cache");
        let tokens: Vec<u32> = (0..12).collect();
        seed(&tree, &unhealthy, &tokens, 3);
        seed(&tree, &healthy, &tokens, 2);
        let config = BoundedCacheAwareConfig {
            pressure_guard: false,
            ..BoundedCacheAwareConfig::default()
        };
        let policy = policy(config, tree, Arc::clone(&active));
        let _healthy_load = active.register(healthy.id.clone(), healthy.url.clone(), 100, 0);
        let _no_cache_load = active.register(no_cache.id.clone(), no_cache.url.clone(), 0, 0);

        let selected = select(&policy, &[Arc::clone(&healthy), no_cache], &tokens);

        assert_eq!(selected.url, healthy.url);
    }
}
