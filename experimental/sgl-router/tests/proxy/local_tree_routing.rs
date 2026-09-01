// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Full HTTP routing paths backed by the local ZMQ radix tree.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use serde_json::json;
use sgl_router::config::{AffinityConfig, PolicyKind};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::engine_load::{EngineLoadTable, LoadStat, NativeCacheRankLoad};
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::kv_events::{
    compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId,
};
use sgl_router::policies::request_tokens_for;
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use tower::ServiceExt;

use crate::common::cache_aware_fixture::{config, MODEL};

fn monitored_load(total_prefill_uncached_tokens: u64, total_prefill_busy_us: u64) -> LoadStat {
    LoadStat {
        num_running_reqs: 0,
        num_waiting_reqs: 0,
        num_tokens: 0,
        max_total_num_tokens: 65_536,
        native_cache: Some(NativeCacheRankLoad {
            num_waiting_uncached_tokens: 0,
            num_total_tokens: 0,
            max_running_requests: 64,
            total_prefill_uncached_tokens,
            total_prefill_busy_us,
        }),
    }
}
use crate::common::mock_worker::MockWorker;

#[tokio::test]
async fn shortest_ttft_routes_from_the_local_radix_tree() {
    let cached = MockWorker::start(vec![]).await;
    let uncached = MockWorker::start(vec![]).await;
    let mut cfg = config();
    cfg.model.policy = PolicyKind::ShortestTtft;
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello there friend"}],
    });
    let tokens = request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &body)
        .expect("test prompt tokenizes");
    let hashes = compute_block_hashes(&tokens.ids, 1);
    assert!(!hashes.is_empty());
    let tree = Arc::new(HashTree::new());
    tree.insert(&KvWorkerId::new(cached.url.clone(), 0), None, &hashes);
    let registry = Arc::new(WorkerRegistry::default());
    for url in [&cached.url, &uncached.url] {
        registry
            .add(WorkerSpec {
                id: WorkerId(url.clone()),
                url: url.clone(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId(MODEL.into())],
                bootstrap_port: None,
            })
            .unwrap();
    }
    let oracle = BlockSizeOracle::new();
    let engine_load = EngineLoadTable::new();
    let captured_at = Instant::now();
    for url in [&cached.url, &uncached.url] {
        engine_load.mark_expected_rank(url, 0);
        engine_load.set(url, 0, monitored_load(1_000, 1_000), captured_at);
        engine_load.set(url, 0, monitored_load(2_000, 2_000), captured_at);
    }
    oracle.try_set(1).unwrap();
    let policies = Arc::new(
        build_registry(
            &cfg,
            tree,
            Arc::clone(&tokenizers),
            Arc::clone(&oracle),
            Arc::clone(&engine_load),
        )
        .unwrap(),
    );
    let mut ctx = AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        policies,
    );
    ctx.block_size_oracle = oracle;
    ctx.engine_load = engine_load;

    let app = build_router(Arc::new(ctx));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(cached.captured.lock().unwrap().last_body.is_some());
    assert!(uncached.captured.lock().unwrap().last_body.is_none());
}

#[tokio::test]
async fn cache_aware_routes_from_the_local_radix_tree_without_prefix_index() {
    let cached = MockWorker::start(vec![]).await;
    let uncached = MockWorker::start(vec![]).await;
    let mut cfg = config();
    cfg.model.policy = PolicyKind::CacheAware;
    cfg.model.cache_aware = None;
    cfg.model.affinity = Some(AffinityConfig {
        cache_affinity_min_matched_tokens: Some(0),
        cache_candidate_min_workers: 1,
        cache_candidate_ratio: 1.0,
        cache_candidate_max_workers: 1,
        ..Default::default()
    });
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "local radix cache hit"}],
    });
    let tokens = request_tokens_for(&tokenizers, &ModelId(MODEL.into()), &body)
        .expect("test prompt tokenizes");
    let hashes = compute_block_hashes(&tokens.ids, 1);
    assert!(!hashes.is_empty());

    let tree = Arc::new(HashTree::new());
    tree.insert(&KvWorkerId::new(cached.url.clone(), 0), None, &hashes);
    let registry = Arc::new(WorkerRegistry::default());
    for url in [&cached.url, &uncached.url] {
        registry
            .add(WorkerSpec {
                id: WorkerId(url.clone()),
                url: url.clone(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId(MODEL.into())],
                bootstrap_port: None,
            })
            .unwrap();
    }
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();
    let policies = Arc::new(
        build_registry(
            &cfg,
            tree,
            Arc::clone(&tokenizers),
            Arc::clone(&oracle),
            EngineLoadTable::new(),
        )
        .unwrap(),
    );
    let mut ctx = AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        registry,
        policies,
    );
    ctx.block_size_oracle = oracle;

    let app = build_router(Arc::new(ctx));
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert!(cached.captured.lock().unwrap().last_body.is_some());
    assert!(uncached.captured.lock().unwrap().last_body.is_none());
}
