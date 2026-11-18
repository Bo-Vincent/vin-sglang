use axum::{
    body::{Body, Bytes},
    extract::Request,
    http::{HeaderValue, StatusCode},
    response::Response,
    routing::any,
    Router,
};
use clap::Parser;
use futures::StreamExt;
use sgl_router::global_router::{build_router, GlobalRouterArgs};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

struct Server {
    url: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn serve(app: Router) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Server { url, task }
}
async fn gateway(routes: &[(&str, &str)], extra: &[&str]) -> Server {
    let mut argv = vec!["sgl-global-router".to_string()];
    for (model, url) in routes {
        argv.extend(["--model-route".into(), format!("{model}={url}")]);
    }
    argv.extend(extra.iter().map(|s| s.to_string()));
    serve(build_router(GlobalRouterArgs::try_parse_from(argv).unwrap()).unwrap()).await
}

#[tokio::test]
async fn routes_models_and_preserves_wire_request_and_downstream_error() {
    let a = serve(Router::new().fallback(any(|req: Request| async move {
        let uri = req.uri().to_string();
        let headers = req.headers().clone();
        let body = axum::body::to_bytes(req.into_body(), 1024).await.unwrap();
        assert_eq!(uri, "/v1/chat/completions?x=1&x=%2F");
        assert_eq!(headers["authorization"], "Bearer secret");
        assert_eq!(headers["anthropic-version"], "2023-06-01");
        assert!(!headers.contains_key("x-remove"));
        assert_eq!(body, "{ \"model\": \"a\", \"messages\": [] }");
        let mut response = Response::new(Body::from("downstream error"));
        *response.status_mut() = StatusCode::TOO_MANY_REQUESTS;
        response
            .headers_mut()
            .insert("retry-after", HeaderValue::from_static("9"));
        response
            .headers_mut()
            .insert("connection", HeaderValue::from_static("x-secret"));
        response
            .headers_mut()
            .insert("x-secret", HeaderValue::from_static("hidden"));
        response
    })))
    .await;
    let b = serve(Router::new().fallback(any(|| async { "model b" }))).await;
    let g = gateway(&[("a", &a.url), ("b", &b.url)], &[]).await;
    let client = reqwest::Client::new();
    let response = client
        .post(format!("{}/v1/chat/completions?x=1&x=%2F", g.url))
        .header("authorization", "Bearer secret")
        .header("anthropic-version", "2023-06-01")
        .header("connection", "x-remove")
        .header("x-remove", "hidden")
        .body("{ \"model\": \"a\", \"messages\": [] }")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 429);
    assert_eq!(response.headers()["retry-after"], "9");
    assert!(!response.headers().contains_key("x-secret"));
    assert_eq!(response.text().await.unwrap(), "downstream error");
    for path in [
        "/generate",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/classify",
        "/v1/tokenize",
        "/v1/detokenize",
        "/v1/rerank",
    ] {
        assert_eq!(
            client
                .post(format!("{}{path}", g.url))
                .json(&serde_json::json!({"model":"b"}))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
            "model b"
        );
    }
    let models: serde_json::Value = client
        .get(format!("{}/v1/models", g.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        models["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["a", "b"]
    );
}

#[tokio::test]
async fn missing_unknown_default_invalid_body_and_private_paths() {
    let downstream = serve(Router::new().fallback(any(|| async { "default" }))).await;
    let g = gateway(&[("a", &downstream.url)], &[]).await;
    let default = gateway(
        &[("a", &downstream.url)],
        &["--default-model", "a", "--max-body-bytes", "64"],
    )
    .await;
    let client = reqwest::Client::new();
    for (body, status) in [
        ("{}", 400),
        ("{\"model\":\"unknown\"}", 404),
        ("{\"model\":null}", 400),
        ("[]", 400),
        ("invalid", 400),
    ] {
        assert_eq!(
            client
                .post(format!("{}/generate", g.url))
                .body(body)
                .send()
                .await
                .unwrap()
                .status(),
            status
        );
    }
    assert_eq!(
        client
            .post(format!("{}/generate", default.url))
            .body("{}")
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
        "default"
    );
    assert_eq!(
        client
            .post(format!("{}/generate", default.url))
            .json(&serde_json::json!({"model":"unknown"}))
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
    assert_eq!(
        client
            .post(format!("{}/generate", default.url))
            .body(" ".repeat(65))
            .send()
            .await
            .unwrap()
            .status(),
        413
    );
    for path in [
        "/workers",
        "/flush_cache",
        "/internal/kv_snapshot",
        "/model_info",
    ] {
        assert_eq!(
            client
                .post(format!("{}{path}", g.url))
                .body("{}")
                .send()
                .await
                .unwrap()
                .status(),
            404
        );
    }
    for path in ["/healthz", "/readyz"] {
        assert_eq!(
            client
                .get(format!("{}{path}", g.url))
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
    }
}

#[test]
fn rejects_invalid_configuration() {
    for routes in [
        vec!["a=http://localhost", "a=http://localhost:2"],
        vec!["=http://localhost"],
        vec!["a=ftp://localhost"],
        vec!["a=http://localhost/path"],
        vec!["a=http://user:pass@localhost"],
        vec!["a=http://localhost?x=1"],
    ] {
        let mut argv = vec!["sgl-global-router".to_string()];
        for route in routes {
            argv.extend(["--model-route".into(), route.into()]);
        }
        assert!(build_router(GlobalRouterArgs::try_parse_from(argv).unwrap()).is_err());
    }
    assert!(GlobalRouterArgs::try_parse_from(["sgl-global-router"]).is_err());
    for extra in [
        ["--default-model", "missing"],
        ["--ttft-timeout-secs", "0"],
        ["--idle-timeout-secs", "0"],
        ["--ttft-timeout-secs", "18446744073709551615"],
        ["--max-body-bytes", "0"],
    ] {
        assert!(build_router(
            GlobalRouterArgs::try_parse_from([
                "sgl-global-router",
                "--model-route",
                "a=http://localhost",
                extra[0],
                extra[1]
            ])
            .unwrap()
        )
        .is_err());
    }
}

#[tokio::test]
async fn incremental_stream_outlives_ttft_and_releases_downstream_on_cancel() {
    for content_type in ["text/event-stream", "application/octet-stream"] {
        check_incremental_stream(content_type).await;
    }
}

async fn check_incremental_stream(content_type: &'static str) {
    let dropped = Arc::new(AtomicUsize::new(0));
    struct Guard(Arc<AtomicUsize>);
    impl Drop for Guard {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let marker = dropped.clone();
    let downstream = serve(Router::new().fallback(any(move || {
        let guard = Guard(marker.clone());
        async move {
            let stream = futures::stream::unfold((guard, 0), |(guard, n)| async move {
                tokio::time::sleep(Duration::from_millis(150)).await;
                Some((
                    Ok::<_, std::io::Error>(Bytes::from(format!("data: {n}\n\n"))),
                    (guard, n + 1),
                ))
            });
            Response::builder()
                .header("content-type", content_type)
                .body(Body::from_stream(stream))
                .unwrap()
        }
    })))
    .await;
    let g = gateway(
        &[("a", &downstream.url)],
        &["--ttft-timeout-secs", "1", "--idle-timeout-secs", "1"],
    )
    .await;
    let response = reqwest::Client::new()
        .post(format!("{}/generate", g.url))
        .json(&serde_json::json!({"model":"a","stream":true}))
        .send()
        .await
        .unwrap();
    let mut stream = response.bytes_stream();
    for n in 0..9 {
        let chunk = tokio::time::timeout(Duration::from_millis(500), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(chunk, format!("data: {n}\n\n"));
    }
    drop(stream);
    tokio::time::timeout(Duration::from_secs(3), async {
        while dropped.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn large_request_and_compressed_response_keep_their_original_bytes() {
    use std::io::Write;
    let original = format!(
        r#"{{"model":"unknown","mo\u0064el":"\u0061","messages":[{{"content":"{}"}}],"opaque":1e300}}"#,
        "large input ".repeat(12_000)
    );
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder
        .write_all(b"compressed downstream response")
        .unwrap();
    let compressed = encoder.finish().unwrap();
    let expected_body = original.clone();
    let response_bytes = compressed.clone();
    let downstream = serve(Router::new().fallback(any(move |req: Request| {
        let expected_body = expected_body.clone();
        let response_bytes = response_bytes.clone();
        async move {
            assert_eq!(req.headers()["accept-encoding"], "gzip");
            assert_eq!(
                axum::body::to_bytes(req.into_body(), 2_000_000)
                    .await
                    .unwrap(),
                expected_body
            );
            Response::builder()
                .header("content-encoding", "gzip")
                .header("content-length", response_bytes.len())
                .body(Body::from(response_bytes))
                .unwrap()
        }
    })))
    .await;
    let g = gateway(&[("a", &downstream.url)], &["--default-model", "a"]).await;
    let client = reqwest::Client::new();
    for body in [
        r#"{"model":null}"#,
        r#"{"model":3}"#,
        r#"{"model":"a","opaque":[1,]}"#,
        r#"{"model":"a","opaque":1e400}"#,
        r#"{"model":"a","opaque":"\uD800"}"#,
    ] {
        assert_eq!(
            client
                .post(format!("{}/generate", g.url))
                .body(body)
                .send()
                .await
                .unwrap()
                .status(),
            400
        );
    }
    let response = client
        .post(format!("{}/generate", g.url))
        .header("accept-encoding", "gzip")
        .body(original)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-encoding"], "gzip");
    assert_eq!(response.content_length(), Some(compressed.len() as u64));
    assert_eq!(response.bytes().await.unwrap(), compressed);
}

#[tokio::test]
async fn ttft_idle_connection_failures_and_redirects() {
    let downstream = serve(Router::new().fallback(any(|req: Request| async move {
        match req.uri().query().unwrap_or("") {
            "headers" => {
                tokio::time::sleep(Duration::from_secs(2)).await;
                Response::new(Body::empty())
            }
            "redirect" => Response::builder()
                .status(307)
                .header("location", "http://127.0.0.1:1/generate")
                .body(Body::from("redirect"))
                .unwrap(),
            mode => {
                let first = if mode == "first" { 2000 } else { 0 };
                let stream = futures::stream::unfold(0, move |n| async move {
                    tokio::time::sleep(Duration::from_millis(if n == 0 { first } else { 2000 }))
                        .await;
                    Some((
                        Ok::<_, std::io::Error>(Bytes::from_static(b"data: first\n\n")),
                        n + 1,
                    ))
                });
                Response::new(Body::from_stream(stream))
            }
        }
    })))
    .await;
    let g = gateway(
        &[("a", &downstream.url)],
        &["--ttft-timeout-secs", "1", "--idle-timeout-secs", "1"],
    )
    .await;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    for mode in ["headers", "first"] {
        assert_eq!(
            client
                .post(format!("{}/generate?{mode}", g.url))
                .body("{\"model\":\"a\"}")
                .send()
                .await
                .unwrap()
                .status(),
            504
        );
    }
    let response = client
        .post(format!("{}/generate?redirect", g.url))
        .body("{\"model\":\"a\"}")
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 307);
    assert_eq!(response.text().await.unwrap(), "redirect");
    let response = client
        .post(format!("{}/generate?idle", g.url))
        .body("{\"model\":\"a\"}")
        .send()
        .await
        .unwrap();
    let mut stream = response.bytes_stream();
    assert!(stream.next().await.unwrap().is_ok());
    assert!(tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
        .unwrap()
        .is_err());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let g = gateway(&[("a", &format!("http://{addr}"))], &[]).await;
    assert_eq!(
        client
            .post(format!("{}/generate", g.url))
            .body("{\"model\":\"a\"}")
            .send()
            .await
            .unwrap()
            .status(),
        502
    );
}
