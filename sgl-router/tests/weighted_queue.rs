// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    response::IntoResponse,
    routing::post,
    Json,
};
use futures::{future::BoxFuture, FutureExt};
use serde_json::{json, Value};
use sgl_router::config::{
    Config, DiscoveryBackend, InflightLoadConfig, ModelConfig, ObservabilityConfig, PolicyKind,
    ProxyConfig, ServerConfig, StaticUrlsDiscoveryConfig,
};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::factory::build_registry_with_defaults;
use sgl_router::proxy::Proxy;
use sgl_router::server::{
    app::build_router, app_context::AppContext, weighted_queue::WeightedQueueConfig,
};
use sgl_router::tokenizer::TokenizerRegistry;
use sgl_router::workers::WorkerRegistry;
use std::{
    num::NonZeroU32,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::{mpsc, oneshot};
use tower::ServiceExt;

fn config(max_attempts: u32) -> Config {
    Config {
        server: ServerConfig {
            host: "0".into(),
            port: 0,
            ..Default::default()
        },
        observability: ObservabilityConfig::default(),
        model: ModelConfig {
            aliases: Vec::new(),
            id: "tiny".into(),
            tokenizer_path: Some("tests/fixtures/tiny_tokenizer.json".into()),
            disable_input_ids_forwarding: false,
            tokenizer: Default::default(),
            policy: PolicyKind::RoundRobin,
            decode_policy: Default::default(),
            dp_aware: false,
            dp_rank_policy: Default::default(),
            decode_dp_rank_policy: None,
            bucket_config: None,
            reorg_buckets: None,
            reorg_admission: Default::default(),
            circuit_breaker: None,
            cache_aware: None,
            sticky: None,
            affinity: None,
            fused: None,
            eligibility: None,
            sampling_overrides: Default::default(),
            default_chat_template_kwargs: Default::default(),
        },
        discovery: DiscoveryBackend::StaticUrls(StaticUrlsDiscoveryConfig {
            urls: vec!["http://placeholder:0".into()],
        }),
        proxy: ProxyConfig {
            max_attempts: NonZeroU32::new(max_attempts).unwrap(),
            ..Default::default()
        },
        router_inflight_load: InflightLoadConfig::default(),
    }
}

struct Arrival {
    forwarded_weight: Option<String>,
    id: String,
    release: oneshot::Sender<()>,
}
struct Engine {
    url: String,
    arrivals: mpsc::UnboundedReceiver<Arrival>,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Engine {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn engine() -> Engine {
    engine_status(StatusCode::OK).await
}
async fn engine_status(status: StatusCode) -> Engine {
    let (tx, rx) = mpsc::unbounded_channel();
    let app = axum::Router::new()
        .route("/v1/chat/completions", post(fake_chat))
        .with_state((tx, status));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Engine {
        url,
        arrivals: rx,
        task,
    }
}
async fn fake_chat(
    State((tx, status)): State<(mpsc::UnboundedSender<Arrival>, StatusCode)>,
    headers: HeaderMap,
    Json(v): Json<Value>,
) -> axum::response::Response {
    let (release, wait) = oneshot::channel();
    tx.send(Arrival {
        forwarded_weight: headers
            .get("x-sgl-request-weight")
            .map(|value| value.to_str().unwrap().to_owned()),
        id: v["messages"][0]["content"].as_str().unwrap().into(),
        release,
    })
    .unwrap();
    if v["stream"] == true {
        let stream = futures::stream::once(async move {
            let _ = wait.await;
            if status != StatusCode::OK {
                return Err(std::io::Error::other("engine stream failed"));
            }
            Ok::<_, std::io::Error>(Bytes::from_static(b"data: [DONE]\n\n"))
        });
        return (
            [("content-type", "text/event-stream")],
            Body::from_stream(stream),
        )
            .into_response();
    }
    let _ = wait.await;
    if status != StatusCode::OK {
        return (status, Json(json!({"error":"retry"}))).into_response();
    }
    Json(json!({"id":"test","object":"chat.completion","model":"tiny","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]})).into_response()
}
fn limits(queued: usize, timeout: u64) -> WeightedQueueConfig {
    WeightedQueueConfig {
        max_inflight: 1,
        max_queued_requests: queued,
        max_queued_body_bytes: 1_000_000,
        queue_timeout_ms: timeout,
    }
}
fn context(engine: &Engine, queue: Option<WeightedQueueConfig>) -> Arc<AppContext> {
    let mut cfg = config(1);
    cfg.server.weighted_queue = queue;
    let tokenizers = Arc::new(TokenizerRegistry::load_from_config(&cfg).unwrap());
    let registry = Arc::new(WorkerRegistry::default());
    registry
        .add(WorkerSpec {
            id: WorkerId("fake".into()),
            url: engine.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            ..Default::default()
        })
        .unwrap();
    let policies = Arc::new(build_registry_with_defaults(&cfg).unwrap());
    Arc::new(AppContext::new(
        cfg,
        tokenizers,
        Arc::new(Proxy::new(Duration::from_secs(10)).unwrap()),
        registry,
        policies,
    ))
}
fn request(id: &str, weight: Option<&str>, stream: bool) -> Request<Body> {
    let mut req = Request::post("/v1/chat/completions").header("content-type", "application/json");
    if let Some(weight) = weight {
        req = req.header("x-sgl-request-weight", weight);
    }
    req.body(Body::from(
        json!({"model":"tiny", "messages":[{"role":"user","content":id}],"stream":stream})
            .to_string(),
    ))
    .unwrap()
}
type ResponseFuture = BoxFuture<'static, axum::response::Response>;
fn dispatch(ctx: &Arc<AppContext>, req: Request<Body>) -> ResponseFuture {
    let app = build_router(ctx.clone());
    async move { app.oneshot(req).await.unwrap() }.boxed()
}
fn enqueue(ctx: &Arc<AppContext>, id: &str, weight: &str) -> ResponseFuture {
    let mut future = dispatch(ctx, request(id, Some(weight), false));
    // Poll ingress synchronously through admission; no worker may be selected yet.
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    future
}
async fn arrival(engine: &mut Engine) -> Arrival {
    tokio::time::timeout(Duration::from_secs(5), engine.arrivals.recv())
        .await
        .unwrap()
        .unwrap()
}
async fn finish(future: ResponseFuture) {
    let response = tokio::time::timeout(Duration::from_secs(5), future)
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
}
fn error(response: &axum::response::Response, status: StatusCode, code: &str) {
    assert_eq!(response.status(), status);
    assert_eq!(response.headers()["x-router-error-code"], code);
}

#[tokio::test]
async fn weighted_public_requests_reorder_and_keep_low_weight_progressing() {
    let mut engine = engine().await;
    let ctx = context(&engine, Some(limits(128, 10_000)));
    let queue = ctx.weighted_queue.as_ref().unwrap();
    let blocker = queue
        .acquire(1, 0, tokio::time::Instant::now() + Duration::from_secs(10))
        .await
        .unwrap();
    let four = enqueue(&ctx, "four", "4");
    let two = enqueue(&ctx, "two", "2");
    let one = enqueue(&ctx, "one", "1");
    let mut tasks = vec![
        tokio::spawn(finish(four)),
        tokio::spawn(finish(two)),
        tokio::spawn(finish(one)),
    ];
    drop(blocker);
    let a = arrival(&mut engine).await;
    assert_eq!(a.id, "four");
    assert_eq!(a.forwarded_weight, None);
    let high = enqueue(&ctx, "high", "100");
    tasks.push(tokio::spawn(finish(high)));
    a.release.send(()).unwrap();
    let a = arrival(&mut engine).await;
    assert_eq!(
        a.id, "high",
        "a fresh high weight may overtake queued weight two"
    );
    let mut trace = vec!["four".to_string(), a.id.clone()];
    let mut current = a;
    let mut saw_two = false;
    let mut saw_one = false;
    // Serving weight two leaves two high requests with the same virtual finish tag.
    // Each subsequent progress step may therefore require two high completions.
    for _ in 0..200 {
        let high = enqueue(&ctx, "high", "100");
        tasks.push(tokio::spawn(finish(high)));
        current.release.send(()).unwrap();
        current = arrival(&mut engine).await;
        trace.push(current.id.clone());
        saw_two |= current.id == "two";
        saw_one |= current.id == "one";
        if saw_two && saw_one {
            break;
        }
    }
    assert!(
        saw_two && saw_one,
        "low weights must progress under continuous high arrivals; saw_two={saw_two}, saw_one={saw_one}, arrivals={trace:?}"
    );
    current.release.send(()).unwrap();
    // The two displaced low-weight requests leave two high requests queued.
    for _ in 0..2 {
        let a = arrival(&mut engine).await;
        a.release.send(()).unwrap();
    }
    for task in tasks {
        task.await.unwrap();
    }
}

#[tokio::test]
async fn public_rejections_and_cancelled_waiters_never_reach_engine() {
    let mut engine = engine().await;
    let ctx = context(&engine, Some(limits(1, 100)));
    let queue = ctx.weighted_queue.as_ref().unwrap();
    let blocker = queue
        .acquire(1, 0, tokio::time::Instant::now() + Duration::from_secs(10))
        .await
        .unwrap();
    for value in ["0", "101", "-1", "1.5", "", "abc"] {
        error(
            &dispatch(&ctx, request("invalid", Some(value), false)).await,
            StatusCode::BAD_REQUEST,
            "invalid_request_weight",
        );
    }
    let mut duplicate = request("duplicate", Some("1"), false);
    duplicate
        .headers_mut()
        .append("x-sgl-request-weight", "2".parse().unwrap());
    error(
        &dispatch(&ctx, duplicate).await,
        StatusCode::BAD_REQUEST,
        "invalid_request_weight",
    );
    let waiter = enqueue(&ctx, "cancelled", "1");
    error(
        &dispatch(&ctx, request("full", Some("1"), false)).await,
        StatusCode::TOO_MANY_REQUESTS,
        "weighted_queue_full",
    );
    drop(waiter);
    let timed = enqueue(&ctx, "timeout", "1");
    error(
        &tokio::time::timeout(Duration::from_secs(3), timed)
            .await
            .unwrap(),
        StatusCode::GATEWAY_TIMEOUT,
        "weighted_queue_timeout",
    );
    let closed = enqueue(&ctx, "closed", "1");
    ctx.mark_not_ready();
    error(
        &closed.await,
        StatusCode::SERVICE_UNAVAILABLE,
        "weighted_queue_unavailable",
    );
    drop(blocker);
    assert!(matches!(
        engine.arrivals.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
}

#[tokio::test]
async fn disabled_queue_ignores_weight_and_preserves_body_limit() {
    let mut engine = engine().await;
    let ctx = context(&engine, None);
    let running = tokio::spawn(finish(dispatch(
        &ctx,
        request("ignored", Some("invalid"), false),
    )));
    let a = arrival(&mut engine).await;
    assert_eq!(a.id, "ignored");
    assert_eq!(a.forwarded_weight.as_deref(), Some("invalid"));
    a.release.send(()).unwrap();
    running.await.unwrap();
    let oversized = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("x".repeat(1024)))
        .unwrap();
    let response = sgl_router::server::app::build_router_with_body_limit(ctx.clone(), 512)
        .oneshot(oversized)
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
    assert!(engine.arrivals.try_recv().is_err());
}

#[tokio::test]
async fn stream_headers_keep_slot_until_eof_or_client_drop() {
    for (cancel, stream_error) in [(false, false), (true, false), (false, true)] {
        let mut engine = engine_status(if stream_error {
            StatusCode::BAD_GATEWAY
        } else {
            StatusCode::OK
        })
        .await;
        let ctx = context(&engine, Some(limits(1, 10_000)));
        let stream = tokio::spawn(dispatch(&ctx, request("stream", None, true)));
        let a = arrival(&mut engine).await;
        let response = tokio::time::timeout(Duration::from_secs(5), stream)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let waiting = enqueue(&ctx, "next", "1");
        error(
            &dispatch(&ctx, request("full", None, false)).await,
            StatusCode::TOO_MANY_REQUESTS,
            "weighted_queue_full",
        );
        assert!(engine.arrivals.try_recv().is_err());
        let next = tokio::spawn(async move {
            let response = waiting.await;
            let expected = if stream_error {
                StatusCode::BAD_GATEWAY
            } else {
                StatusCode::OK
            };
            assert_eq!(response.status(), expected);
            axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
        });
        if cancel {
            drop(response);
            let _ = a.release.send(());
        } else {
            a.release.send(()).unwrap();
            let body = axum::body::to_bytes(response.into_body(), usize::MAX).await;
            assert_eq!(body.is_err(), stream_error);
        }
        let a = arrival(&mut engine).await;
        assert_eq!(a.id, "next");
        a.release.send(()).unwrap();
        next.await.unwrap();
    }
}

#[tokio::test]
async fn retry_keeps_one_admission_slot_across_worker_attempts() {
    let mut good = engine().await;
    let mut bad = engine_status(StatusCode::SERVICE_UNAVAILABLE).await;
    let mut ctx = context(&good, Some(limits(1, 10_000)));
    Arc::get_mut(&mut ctx).unwrap().config.proxy.max_attempts = NonZeroU32::new(2).unwrap();
    ctx.registry
        .add(WorkerSpec {
            id: WorkerId("retry-failure".into()),
            url: bad.url.clone(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("tiny".into())],
            ..Default::default()
        })
        .unwrap();
    let mut observed_retry = false;
    for _ in 0..4 {
        let running = tokio::spawn(finish(dispatch(&ctx, request("retry", None, false))));
        let (first, failed) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::select! {
                arrival = good.arrivals.recv() => (arrival.unwrap(), false),
                arrival = bad.arrivals.recv() => (arrival.unwrap(), true),
            }
        })
        .await
        .unwrap();
        let waiting = enqueue(&ctx, "cancelled", "1");
        error(
            &dispatch(&ctx, request("full", None, false)).await,
            StatusCode::TOO_MANY_REQUESTS,
            "weighted_queue_full",
        );
        first.release.send(()).unwrap();
        if failed {
            let second = arrival(&mut good).await;
            assert_eq!(second.id, "retry");
            // Retrying neither releases the original slot nor acquires a second one.
            error(
                &dispatch(&ctx, request("still-full", None, false)).await,
                StatusCode::TOO_MANY_REQUESTS,
                "weighted_queue_full",
            );
            assert!(bad.arrivals.try_recv().is_err());
            drop(waiting);
            second.release.send(()).unwrap();
            observed_retry = true;
        } else {
            drop(waiting);
        }
        running.await.unwrap();
        if observed_retry {
            break;
        }
    }
    assert!(
        observed_retry,
        "round robin must exercise the failing worker"
    );
    let permit = tokio::time::timeout(
        Duration::from_secs(5),
        ctx.weighted_queue.as_ref().unwrap().acquire(
            1,
            0,
            tokio::time::Instant::now() + Duration::from_secs(5),
        ),
    )
    .await
    .unwrap()
    .unwrap();
    drop(permit);
}

#[tokio::test]
async fn request_budget_includes_queue_and_retry_backoff() {
    let mut bad = engine_status(StatusCode::SERVICE_UNAVAILABLE).await;
    let mut ctx = context(&bad, Some(limits(1, 10_000)));
    let mutable = Arc::get_mut(&mut ctx).unwrap();
    mutable.config.proxy.max_attempts = NonZeroU32::new(2).unwrap();
    mutable.config.proxy.initial_backoff_ms = 1_000;
    mutable.config.proxy.max_backoff_ms = 1_000;
    mutable.proxy = Arc::new(Proxy::new(Duration::from_millis(100)).unwrap());
    let blocker = ctx
        .weighted_queue
        .as_ref()
        .unwrap()
        .acquire(1, 0, tokio::time::Instant::now() + Duration::from_secs(5))
        .await
        .unwrap();
    let started = tokio::time::Instant::now();
    let queued = enqueue(&ctx, "budget", "1");
    let running = tokio::spawn(queued);
    // Deliberately consume most of the request's wall-clock budget in admission.
    tokio::time::sleep(Duration::from_millis(80)).await;
    drop(blocker);
    let first = arrival(&mut bad).await;
    assert_eq!(first.id, "budget");
    first.release.send(()).unwrap();
    let response = tokio::time::timeout_at(started + Duration::from_millis(350), running)
        .await
        .expect("queue wait and retry backoff must share the 100ms request budget")
        .unwrap();
    assert!(response.status().is_server_error());
    axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(bad.arrivals.try_recv().is_err());
}
