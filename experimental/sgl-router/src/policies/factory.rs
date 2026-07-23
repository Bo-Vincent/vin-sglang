// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::{Config, ModelConfig, PolicyKind};
use crate::discovery::ModelId;
use crate::policies::{
    active_load::ActiveLoadRegistry,
    cache_aware::CacheAwarePolicy,
    cache_aware_zmq::CacheAwareZmqPolicy,
    kv_events::{BlockSizeOracle, HashTree},
    load_based::LoadBasedPolicy,
    power_of_two::PowerOfTwoChoicesPolicy,
    random::RandomPolicy,
    round_robin::RoundRobinPolicy,
    session_aware::SessionAwarePolicy,
    sticky::StickyPolicy,
    Policy, PolicyRegistry,
};
use crate::tokenizer::TokenizerRegistry;
use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;

/// Build a dependency-free policy for use as the sticky-session fallback
/// (keyless requests + initial pin of a new key). `Cli::into_config`
/// validates `--sticky-fallback-policy` to one of these four, so the
/// `CacheAwareZmq`/`Sticky` arms are never reached in practice.
fn build_sticky_fallback(kind: PolicyKind) -> Arc<dyn Policy> {
    match kind {
        PolicyKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        PolicyKind::Random => Arc::new(RandomPolicy::new()),
        PolicyKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
        PolicyKind::LoadBased => Arc::new(LoadBasedPolicy::new()),
        PolicyKind::SessionAware
        | PolicyKind::CacheAware
        | PolicyKind::CacheAwareZmq
        | PolicyKind::Sticky => {
            unreachable!("sticky fallback is validated to be dependency-free in Cli::into_config")
        }
    }
}

/// Construct a [`StickyPolicy`] from a model's `sticky` config (or
/// defaults). Shared by `build_policy` and the test shim so the duration
/// conversion + fallback wiring live in one place.
fn build_sticky(model: &ModelConfig) -> Arc<dyn Policy> {
    let s = model.sticky.clone().unwrap_or_default();
    Arc::new(StickyPolicy::new(
        Duration::from_secs(s.idle_secs),
        Duration::from_secs(s.eviction_interval_secs),
        build_sticky_fallback(s.fallback_policy),
    ))
}

/// Construct a policy for a single model from its [`ModelConfig`] and the
/// process-shared `HashTree` + `TokenizerRegistry` + `BlockSizeOracle`.
///
/// The tree, tokenizer registry, and oracle are consulted by both
/// cache-aware variants. Callers pass shared instances together with the
/// shared active-load registry.
pub fn build_policy(
    model: &ModelConfig,
    tree: Arc<HashTree>,
    tokenizers: Arc<TokenizerRegistry>,
    block_size_oracle: Arc<BlockSizeOracle>,
    active_load: Arc<ActiveLoadRegistry>,
) -> Arc<dyn Policy> {
    match model.policy {
        PolicyKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        PolicyKind::Random => Arc::new(RandomPolicy::new()),
        PolicyKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
        PolicyKind::LoadBased => Arc::new(LoadBasedPolicy::new()),
        PolicyKind::SessionAware => Arc::new(SessionAwarePolicy::new(
            model.session_aware.clone().unwrap_or_default(),
            active_load,
        )),
        PolicyKind::CacheAware => Arc::new(CacheAwarePolicy::new(
            model.bounded_cache_aware.unwrap_or_default(),
            tree,
            tokenizers,
            block_size_oracle,
            active_load,
        )),
        PolicyKind::CacheAwareZmq => {
            let cache_cfg = model.cache_aware.unwrap_or_default();
            Arc::new(CacheAwareZmqPolicy::new(
                cache_cfg,
                tree,
                tokenizers,
                block_size_oracle,
            ))
        }
        PolicyKind::Sticky => build_sticky(model),
    }
}

/// Test-only constructor that supplies empty process dependencies.
#[cfg(test)]
pub fn build_policy_kind_only(kind: PolicyKind) -> Arc<dyn Policy> {
    match kind {
        PolicyKind::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        PolicyKind::Random => Arc::new(RandomPolicy::new()),
        PolicyKind::PowerOfTwo => Arc::new(PowerOfTwoChoicesPolicy::new()),
        PolicyKind::LoadBased => Arc::new(LoadBasedPolicy::new()),
        PolicyKind::SessionAware => Arc::new(SessionAwarePolicy::new(
            crate::config::SessionAwareConfig::default(),
            ActiveLoadRegistry::with_defaults(),
        )),
        PolicyKind::CacheAware => Arc::new(CacheAwarePolicy::new(
            crate::config::BoundedCacheAwareConfig::default(),
            Arc::new(HashTree::new()),
            Arc::new(TokenizerRegistry::default()),
            BlockSizeOracle::new(),
            ActiveLoadRegistry::with_defaults(),
        )),
        PolicyKind::CacheAwareZmq => {
            // Provide an empty tree + empty tokenizer registry + fresh
            // oracle so the test policy is constructible. Production
            // callers go through `build_policy` with the real
            // process-shared instances.
            Arc::new(CacheAwareZmqPolicy::new(
                crate::config::CacheAwareConfig::default(),
                Arc::new(HashTree::new()),
                Arc::new(TokenizerRegistry::default()),
                BlockSizeOracle::new(),
            ))
        }
        PolicyKind::Sticky => {
            let s = crate::config::StickyConfig::default();
            Arc::new(StickyPolicy::new(
                Duration::from_secs(s.idle_secs),
                Duration::from_secs(s.eviction_interval_secs),
                build_sticky_fallback(s.fallback_policy),
            ))
        }
    }
}

pub fn build_registry(
    cfg: &Config,
    tree: Arc<HashTree>,
    tokenizers: Arc<TokenizerRegistry>,
    block_size_oracle: Arc<BlockSizeOracle>,
    active_load: Arc<ActiveLoadRegistry>,
) -> Result<PolicyRegistry> {
    let reg = PolicyRegistry::default();
    let m = &cfg.model;
    reg.insert(
        ModelId(m.id.clone()),
        build_policy(
            m,
            Arc::clone(&tree),
            Arc::clone(&tokenizers),
            Arc::clone(&block_size_oracle),
            active_load,
        ),
    );
    Ok(reg)
}

/// Convenience for tests + non-cache-aware callers: builds a registry with
/// a fresh, empty `HashTree` and an empty `TokenizerRegistry`. The
/// cache-aware policies then degrade through their normal missing-signal
/// fallbacks, which is what dependency-free tests expect.
///
/// Production callers go through [`build_registry`] with the real
/// process-shared instances.
pub fn build_registry_with_defaults(cfg: &Config) -> Result<PolicyRegistry> {
    build_registry(
        cfg,
        Arc::new(HashTree::new()),
        Arc::new(TokenizerRegistry::default()),
        BlockSizeOracle::new(),
        ActiveLoadRegistry::with_defaults(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ProxyConfig, ServerConfig,
        StaticUrlsDiscoveryConfig,
    };

    use crate::config::PolicyKind;

    fn cfg_with_model(id: &str, policy: PolicyKind) -> Config {
        Config {
            server: ServerConfig {
                host: "0".into(),
                port: 0,
            },
            observability: Default::default(),
            model: ModelConfig {
                id: id.into(),
                tokenizer_path: "/tmp/x".into(),
                policy,
                circuit_breaker: None,
                cache_aware: None,
                bounded_cache_aware: None,
                session_aware: None,
                sticky: None,
            },
            discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
                urls: vec!["http://placeholder:0".into()],
            }),
            proxy: ProxyConfig::default(),
            active_load: ActiveLoadConfig::default(),
        }
    }

    #[test]
    fn build_policy_kind_only_covers_all_variants() {
        // Trivially total — the match is exhaustive over `PolicyKind`.
        let _ = build_policy_kind_only(PolicyKind::RoundRobin);
        let _ = build_policy_kind_only(PolicyKind::Random);
        let _ = build_policy_kind_only(PolicyKind::PowerOfTwo);
        let _ = build_policy_kind_only(PolicyKind::LoadBased);
        let _ = build_policy_kind_only(PolicyKind::SessionAware);
        let _ = build_policy_kind_only(PolicyKind::CacheAware);
        let _ = build_policy_kind_only(PolicyKind::CacheAwareZmq);
        let _ = build_policy_kind_only(PolicyKind::Sticky);
    }

    #[test]
    fn registry_assigns_configured_model() {
        let cfg = cfg_with_model("qwen", PolicyKind::RoundRobin);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(
            &cfg,
            tree,
            tokenizers,
            BlockSizeOracle::new(),
            ActiveLoadRegistry::with_defaults(),
        )
        .unwrap();
        assert!(reg.get(&ModelId("qwen".into())).is_some());
        assert!(reg.get(&ModelId("missing".into())).is_none());
    }

    #[test]
    fn cache_aware_zmq_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::CacheAwareZmq);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(
            &cfg,
            tree,
            tokenizers,
            BlockSizeOracle::new(),
            ActiveLoadRegistry::with_defaults(),
        )
        .unwrap();
        let p = reg.get(&ModelId("modelA".into())).unwrap();
        // Down-cast probe via Debug — cheaper than carrying a type-tag
        // on the trait. Pinning the debug repr is fine because the field
        // name is part of the file's public test surface.
        let dbg = format!("{p:?}");
        assert!(
            dbg.contains("CacheAwareZmqPolicy"),
            "expected CacheAwareZmqPolicy debug repr, got: {dbg}",
        );
    }

    #[test]
    fn session_aware_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::SessionAware);
        let reg = build_registry_with_defaults(&cfg).unwrap();
        let policy = reg.get(&ModelId("modelA".into())).unwrap();

        assert!(format!("{policy:?}").contains("SessionAwarePolicy"));
    }

    #[test]
    fn bounded_cache_aware_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::CacheAware);
        let reg = build_registry_with_defaults(&cfg).unwrap();
        let policy = reg.get(&ModelId("modelA".into())).unwrap();

        assert!(format!("{policy:?}").contains("CacheAwarePolicy"));
    }

    #[test]
    fn load_based_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::LoadBased);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(
            &cfg,
            tree,
            tokenizers,
            BlockSizeOracle::new(),
            ActiveLoadRegistry::with_defaults(),
        )
        .unwrap();
        let p = reg.get(&ModelId("modelA".into())).unwrap();
        let dbg = format!("{p:?}");
        assert!(
            dbg.contains("LoadBasedPolicy"),
            "expected LoadBasedPolicy debug repr, got: {dbg}",
        );
    }

    #[test]
    fn sticky_builds_via_factory() {
        let cfg = cfg_with_model("modelA", PolicyKind::Sticky);
        let tree = Arc::new(HashTree::new());
        let tokenizers = Arc::new(TokenizerRegistry::default());
        let reg = build_registry(
            &cfg,
            tree,
            tokenizers,
            BlockSizeOracle::new(),
            ActiveLoadRegistry::with_defaults(),
        )
        .unwrap();
        let p = reg.get(&ModelId("modelA".into())).unwrap();
        let dbg = format!("{p:?}");
        assert!(
            dbg.contains("StickyPolicy"),
            "expected StickyPolicy debug repr, got: {dbg}",
        );
    }
}
