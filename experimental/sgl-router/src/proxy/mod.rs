// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! HTTP proxy — forwards requests to the upstream SGLang worker.

pub mod sse;

use crate::health::circuit_breaker::CircuitBreaker;
use crate::server::error::ApiError;
use crate::server::header_utils::should_forward_request_header;
use anyhow::Context;
use axum::body::Body;
use axum::http::{HeaderMap, HeaderName, HeaderValue, Response};
use bytes::Bytes;
use reqwest::{Client, Url};
use std::sync::Arc;
use std::time::Duration;

// SGLang's Uvicorn server closes idle HTTP/1.1 connections after five seconds
// by default. Keep the Router's pool lifetime below that value so a POST never
// reuses a connection the worker has already closed. POST forwarding is not
// retried automatically because the worker may have observed the request body.
const UPSTREAM_POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(4);

/// Parse a worker URL emitted by discovery.  On failure, trip the worker's
/// circuit breaker so the malformed worker drops out of subsequent
/// `healthy_workers_for(...)` selection, then surface the error as
/// `ApiError::WorkerMisconfigured`.
fn parse_worker_url(worker_url: &str, breaker: &CircuitBreaker) -> Result<Url, ApiError> {
    Url::parse(worker_url).map_err(|e| {
        breaker.record_failure();
        ApiError::WorkerMisconfigured {
            worker: worker_url.to_string(),
            source: anyhow::Error::new(e).context("parse worker URL"),
        }
    })
}

fn build_proxy_client() -> Result<Client, anyhow::Error> {
    build_proxy_client_with_pool_idle_timeout(UPSTREAM_POOL_IDLE_TIMEOUT)
}

fn build_proxy_client_with_pool_idle_timeout(
    pool_idle_timeout: Duration,
) -> Result<Client, anyhow::Error> {
    Client::builder()
        .pool_max_idle_per_host(64)
        .pool_idle_timeout(pool_idle_timeout)
        .connect_timeout(Duration::from_secs(5))
        .build()
        .context("build reqwest client")
}

#[derive(Debug)]
pub struct Proxy {
    pub client: Client,
    /// Wall-clock timeout applied to non-streaming upstream requests. Streaming
    /// requests deliberately do not use this (long generations are valid).
    pub request_timeout: Duration,
}

impl Proxy {
    /// Build a proxy. `request_timeout` is the per-request wall-clock budget for
    /// non-streaming forwards. Connect timeout is hard-coded to 5 s — even a
    /// streaming request fails fast at TCP setup if the worker is unreachable.
    pub fn new(request_timeout: Duration) -> Result<Self, anyhow::Error> {
        let client = build_proxy_client()?;
        Ok(Self {
            client,
            request_timeout,
        })
    }

    /// Classify a reqwest error into the right `ApiError` variant, given an
    /// explicit worker URL. Called from the breaker-gated `forward_*_to`
    /// methods, which carry per-request worker URLs (not a single proxy-level
    /// URL).
    ///
    /// Walks the full source chain to detect timeouts, because reqwest wraps
    /// hyper which wraps `std::io::Error` — a top-level `is_timeout()` check
    /// misses both the wrapped reqwest timeout and the `io::ErrorKind::TimedOut`
    /// cases.
    fn classify_reqwest_error_for(worker: Url, e: reqwest::Error, path: &str) -> ApiError {
        let source = anyhow::Error::new(e).context(format!("worker {worker}: post {path}"));
        let is_timeout = source.chain().any(|c| {
            c.downcast_ref::<reqwest::Error>()
                .is_some_and(|r| r.is_timeout())
        }) || source.chain().any(|c| {
            c.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::TimedOut)
        });
        if is_timeout {
            ApiError::UpstreamTimeout { worker }
        } else {
            ApiError::UpstreamUnreachable { worker, source }
        }
    }

    /// Breaker-gated JSON POST: checks `breaker.allow()` first, records
    /// success/failure based on response status, and returns
    /// `ApiError::BreakerOpen` immediately when the breaker is Open.
    ///
    /// `worker_url` is the discovery-emitted worker URL string. It's parsed
    /// to [`reqwest::Url`] internally so we can use [`Url::join`] for clean
    /// path concatenation (no double-slash) and pass a typed URL to the
    /// split error variants (`UpstreamUnreachable` / `UpstreamTimeout` /
    /// `UpstreamStatus`).
    pub async fn forward_json_to(
        &self,
        worker_url: &str,
        breaker: &CircuitBreaker,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .timeout(self.request_timeout);
        let resp = req.send().await.map_err(|e| {
            breaker.record_failure();
            Self::classify_reqwest_error_for(worker_url.clone(), e, path)
        })?;
        let status = resp.status();
        // Defer breaker recording until after the body completes — a
        // worker that returns 2xx headers and then drops mid-body is
        // still failing the request, and crediting it as healthy lets
        // a misbehaving worker stay eligible. For 5xx the early bail is
        // safe (no body to consume meaningfully), but we still wait
        // until after the read attempt to record exactly once.
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(
                    upstream = %url,
                    status = %status,
                    error = ?e,
                    "upstream dropped connection mid-body",
                );
                breaker.record_failure();
                return Err(ApiError::UpstreamStatus { status });
            }
        };
        if status.is_server_error() {
            breaker.record_failure();
        } else {
            breaker.record_success();
        }
        let mut out = Response::new(Body::from(bytes));
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        Ok(out)
    }

    /// Breaker-gated streaming POST: checks `breaker.allow()` first, records
    /// success/failure, and returns `ApiError::BreakerOpen` when Open.
    ///
    /// `stream_guards` — when `Some`, the value is threaded into the SSE
    /// pump task and held for the entire body lifetime (headers → last byte
    /// / client disconnect).  The proxy does not inspect the boxed value; it
    /// relies entirely on `Drop` semantics, so callers typically pack
    /// `(LoadGuard, ActiveLoadGuard)` here. This keeps both the per-worker
    /// `active_requests` counter and the per-request active-load entry alive
    /// for the full streaming lifetime — without which a long-running SSE
    /// response would under-report load.
    // Each parameter is a distinct, required input to a single upstream
    // forward (target, breaker, path, headers, body, plus the two
    // streaming-lifetime callbacks). Bundling them into a struct purely to
    // satisfy the arg-count heuristic would add indirection without clarity.
    #[allow(clippy::too_many_arguments)]
    pub async fn forward_streaming_to(
        &self,
        worker_url: &str,
        breaker: &Arc<CircuitBreaker>,
        path: &str,
        headers: &HeaderMap,
        body: Bytes,
        stream_guards: Option<Box<dyn Send + 'static>>,
        on_first_byte: Option<Box<dyn FnOnce() + Send + 'static>>,
    ) -> Result<Response<Body>, ApiError> {
        if !breaker.allow() {
            return Err(ApiError::BreakerOpen {
                worker: worker_url.to_string(),
            });
        }
        let worker_url = parse_worker_url(worker_url, breaker)?;
        let url = worker_url.join(path).map_err(|e| {
            ApiError::Internal(anyhow::Error::new(e).context(format!("join worker path {path}")))
        })?;
        let mut req = self.client.post(url.clone()).body(body);
        for (k, v) in headers {
            if should_forward_request_header(k) {
                req = req.header(k, v);
            }
        }
        req = req
            .header("content-type", "application/json")
            .header("accept", "text/event-stream");
        let resp = req.send().await.map_err(|e| {
            breaker.record_failure();
            Self::classify_reqwest_error_for(worker_url.clone(), e, path)
        })?;
        let status = resp.status();
        let upstream_ct = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/json")
            .to_string();
        let content_type = if status.is_success() {
            "text/event-stream".to_string()
        } else {
            upstream_ct
        };
        // Breaker recording is deferred to the pump's completion hook so
        // an upstream that returns 2xx headers and then drops mid-stream
        // is recorded as a failure. For 5xx headers we record_failure
        // up front and skip the pump hook (the body we surface is the
        // error response — its stream completing is not a worker win).
        let on_complete: Option<Box<dyn FnOnce(bool) + Send + 'static>> =
            if status.is_server_error() {
                breaker.record_failure();
                None
            } else {
                let breaker_for_hook = Arc::clone(breaker);
                Some(Box::new(move |ok| {
                    if ok {
                        breaker_for_hook.record_success();
                    } else {
                        breaker_for_hook.record_failure();
                    }
                }))
            };
        // Only record TTFT for successful streams — a 4xx/5xx error body
        // streaming back is not a generated token, so drop the hook for
        // non-2xx responses.
        let first_byte_hook = if status.is_success() {
            on_first_byte
        } else {
            None
        };
        let body = sse::bytes_stream_to_body(
            resp.bytes_stream(),
            stream_guards,
            on_complete,
            first_byte_hook,
        );
        let mut out = Response::new(body);
        *out.status_mut() = status;
        out.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_str(&content_type)
                .unwrap_or_else(|_| HeaderValue::from_static("application/json")),
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    async fn read_request_headers(stream: &mut TcpStream) {
        let mut bytes = Vec::new();
        let mut chunk = [0_u8; 1024];
        loop {
            let read = stream.read(&mut chunk).await.unwrap();
            assert!(read > 0, "client closed before sending HTTP headers");
            bytes.extend_from_slice(&chunk[..read]);
            if bytes.windows(4).any(|window| window == b"\r\n\r\n") {
                return;
            }
        }
    }

    async fn write_keep_alive_response(stream: &mut TcpStream) {
        stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: keep-alive\r\n\r\nok")
            .await
            .unwrap();
        stream.flush().await.unwrap();
    }

    #[tokio::test]
    async fn new_returns_result_not_panic() {
        let p = Proxy::new(Duration::from_secs(5)).unwrap();
        assert_eq!(p.request_timeout, Duration::from_secs(5));
    }

    #[tokio::test]
    async fn pool_discards_idle_connection_before_a_closed_worker_connection_is_reused() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut first, _) = listener.accept().await.unwrap();
            read_request_headers(&mut first).await;
            write_keep_alive_response(&mut first).await;
            tokio::time::sleep(Duration::from_millis(60)).await;
            drop(first);

            let (mut second, _) = tokio::time::timeout(Duration::from_secs(1), listener.accept())
                .await
                .expect("client must open a new connection after the worker closes idle keep-alive")
                .unwrap();
            read_request_headers(&mut second).await;
            write_keep_alive_response(&mut second).await;
        });

        let client = build_proxy_client_with_pool_idle_timeout(Duration::from_millis(20)).unwrap();
        let url = format!("http://{address}/v1/chat/completions");
        assert_eq!(
            client
                .post(&url)
                .body("first")
                .send()
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::OK
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(
            client
                .post(&url)
                .body("second")
                .send()
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::OK
        );
        server.await.unwrap();
    }
}
