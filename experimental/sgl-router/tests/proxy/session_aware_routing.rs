// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP-path coverage for the new `session_aware` policy.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use sgl_router::config::{
    ActiveLoadConfig, Config, DiscoveryBackend, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, SessionAwareConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::active_load::ActiveLoadRegistry;
use sgl_router::policies::factory::build_registry;
use sgl_router::policies::kv_events::{BlockSizeOracle, HashTree};
use sgl_router::proxy::Proxy;
use sgl_router::server::app::build_router;
use sgl_router::server::app_context::AppContext;
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tower::ServiceExt;

use crate::common::mock_worker::MockWorker;

fn build_context(header: &str, worker_urls: &[String]) -> Arc<AppContext> {
    let cfg = Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            id: "tiny".into(),
            tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
            policy: PolicyKind::SessionAware,
            circuit_breaker: None,
            cache_aware: None,
            bounded_cache_aware: None,
            session_aware: Some(SessionAwareConfig {
                header_name: header.into(),
                strict: true,
                idle_secs: 3600,
                eviction_interval_secs: 3600,
                ..SessionAwareConfig::default()
            }),
            sticky: None,
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig::default(),
        active_load: ActiveLoadConfig::default(),
    };
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let workers = Arc::new(WorkerRegistry::default());
    for (index, url) in worker_urls.iter().enumerate() {
        workers
            .add(WorkerSpec {
                id: WorkerId(format!("w{index}")),
                url: url.clone(),
                mode: WorkerMode::Plain,
                model_ids: vec![ModelId("tiny".into())],
                bootstrap_port: None,
            })
            .unwrap();
    }
    let active_load = ActiveLoadRegistry::with_defaults();
    let policies = Arc::new(
        build_registry(
            &cfg,
            Arc::new(HashTree::new()),
            Arc::clone(&tokenizers),
            BlockSizeOracle::new(),
            Arc::clone(&active_load),
        )
        .unwrap(),
    );
    Arc::new(AppContext::with_active_load(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(5)).unwrap()),
        workers,
        policies,
        active_load,
    ))
}

fn request(header: Option<(&str, &str)>) -> Request<Body> {
    let mut request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some((name, value)) = header {
        request = request.header(name, value);
    }
    request
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "tiny",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": false
            }))
            .unwrap(),
        ))
        .unwrap()
}

fn successful_dispatches(metrics: &str) -> HashMap<String, u64> {
    let mut counts = HashMap::new();
    for line in metrics
        .lines()
        .filter(|line| line.starts_with("sgl_router_worker_requests_total{"))
        .filter(|line| line.contains(r#"outcome="success""#))
    {
        let Some(after) = line.split(r#"worker_url=""#).nth(1) else {
            continue;
        };
        let Some((url, _)) = after.split_once('"') else {
            continue;
        };
        let count = line
            .rsplit_once(' ')
            .and_then(|(_, count)| count.parse::<u64>().ok())
            .unwrap_or(0);
        *counts.entry(url.to_string()).or_insert(0) += count;
    }
    counts
}

#[tokio::test]
async fn configured_session_header_pins_repeated_requests() {
    let w0 = MockWorker::start(vec![]).await;
    let w1 = MockWorker::start(vec![]).await;
    let ctx = build_context("x-agent-session", &[w0.url, w1.url]);
    let app = build_router(Arc::clone(&ctx));

    for _ in 0..4 {
        let response = app
            .clone()
            .oneshot(request(Some(("x-agent-session", "session-1"))))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    let metrics = ctx.metrics.render();
    let used: Vec<_> = successful_dispatches(&metrics)
        .into_values()
        .filter(|count| *count > 0)
        .collect();
    assert_eq!(used, vec![4], "{metrics}");
    assert!(metrics.contains(
        r#"sgl_router_policy_decisions_total{policy="session_aware",reason="assigned"} 1"#
    ));
    assert!(metrics.contains(
        r#"sgl_router_policy_decisions_total{policy="session_aware",reason="session_primary"} 3"#
    ));
}

#[tokio::test]
async fn unconfigured_header_is_treated_as_no_session() {
    let w0 = MockWorker::start(vec![]).await;
    let w1 = MockWorker::start(vec![]).await;
    let ctx = build_context("x-agent-session", &[w0.url, w1.url]);
    let app = build_router(Arc::clone(&ctx));

    let response = app
        .oneshot(request(Some(("x-sgl-session-id", "ignored"))))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(ctx.metrics.render().contains(
        r#"sgl_router_policy_decisions_total{policy="session_aware",reason="no_session"} 1"#
    ));
}
