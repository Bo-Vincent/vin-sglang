// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Static model dispatch to independent, single-model sgl-router services.

use std::{collections::BTreeMap, net::IpAddr, sync::Arc, time::Duration};

use anyhow::{bail, Context};
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Request, State},
    http::{HeaderMap, Uri},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use futures::StreamExt;
use reqwest::{Client, Url};
use serde_json::Value;

use crate::server::error::ApiError;

#[derive(Debug, Parser)]
#[command(
    name = "sgl-global-router",
    about = "Dispatch models to independent sgl-router services"
)]
pub struct GlobalRouterArgs {
    #[arg(long, default_value = "0.0.0.0")]
    pub host: IpAddr,
    #[arg(long, default_value_t = 30000)]
    pub port: u16,
    /// Repeat MODEL=http(s)://host:port; each model has exactly one service URL.
    #[arg(long, required = true)]
    pub model_route: Vec<String>,
    /// Route requests without a model to this configured model.
    #[arg(long)]
    pub default_model: Option<String>,
    /// Deadline from downstream dispatch through the first response body chunk.
    #[arg(long, default_value_t = 300)]
    pub ttft_timeout_secs: u64,
    /// Maximum silence between downstream response body chunks.
    #[arg(long, default_value_t = 60)]
    pub idle_timeout_secs: u64,
    #[arg(long, default_value_t = 16_777_216)]
    pub max_body_bytes: usize,
}

struct Gateway {
    routes: BTreeMap<String, Url>,
    default_model: Option<String>,
    client: Client,
    ttft: Duration,
    idle: Duration,
    max_body_bytes: usize,
}

pub fn build_router(args: GlobalRouterArgs) -> anyhow::Result<Router> {
    if args.ttft_timeout_secs == 0 || args.idle_timeout_secs == 0 || args.max_body_bytes == 0 {
        bail!("timeouts and max-body-bytes must be positive");
    }
    for seconds in [args.ttft_timeout_secs, args.idle_timeout_secs] {
        if std::time::Instant::now()
            .checked_add(Duration::from_secs(seconds))
            .is_none()
        {
            bail!("timeout exceeds the supported clock range");
        }
    }
    let mut routes = BTreeMap::new();
    for route in args.model_route {
        let (model, address) = route
            .split_once('=')
            .context("model-route must be MODEL=URL")?;
        if model.trim().is_empty() || model.trim() != model || model.chars().any(char::is_control) {
            bail!(
                "model-route requires a nonempty model without surrounding whitespace or controls"
            );
        }
        let url = Url::parse(address).context("invalid model-route URL")?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.path() != "/"
            || url.query().is_some()
            || url.fragment().is_some()
        {
            bail!("model-route URL must be an HTTP(S) origin without credentials, path, query or fragment");
        }
        if routes.insert(model.to_string(), url).is_some() {
            bail!("duplicate model-route: {model}");
        }
    }
    if routes.is_empty() {
        bail!("at least one model-route is required");
    }
    if args
        .default_model
        .as_ref()
        .is_some_and(|model| !routes.contains_key(model))
    {
        bail!("default-model must name a configured model-route");
    }
    let gateway = Arc::new(Gateway {
        routes,
        default_model: args.default_model,
        client: Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(Duration::from_secs(5))
            .build()?,
        ttft: Duration::from_secs(args.ttft_timeout_secs),
        idle: Duration::from_secs(args.idle_timeout_secs),
        max_body_bytes: args.max_body_bytes,
    });
    let mut router = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/readyz", get(|| async { "ok" }))
        .route("/v1/models", get(models));
    for path in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/classify",
        "/v1/tokenize",
        "/v1/detokenize",
    ] {
        router = router.route(path, post(forward));
    }
    Ok(router
        .route("/generate", post(forward).put(forward))
        .route("/v1/rerank", post(forward).put(forward))
        .layer(DefaultBodyLimit::max(args.max_body_bytes))
        .with_state(gateway))
}

async fn models(State(gateway): State<Arc<Gateway>>) -> Json<Value> {
    Json(
        serde_json::json!({"object":"list", "data": gateway.routes.keys().map(|model|
        serde_json::json!({"id":model,"object":"model","created":0,"owned_by":"sgl-global-router"})
    ).collect::<Vec<_>>()}),
    )
}

// Connection can name additional hop-by-hop fields; remove those on both legs.
fn end_to_end_headers(headers: &HeaderMap) -> HeaderMap {
    let named: Vec<_> = headers
        .get_all("connection")
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| {
            value
                .split(',')
                .map(|name| name.trim().to_ascii_lowercase())
        })
        .collect();
    let mut result = headers.clone();
    for name in [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "proxy-connection",
    ] {
        result.remove(name);
    }
    for name in named {
        result.remove(name);
    }
    result
}

async fn forward(State(gateway): State<Arc<Gateway>>, request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let body = match axum::body::to_bytes(body, gateway.max_body_bytes).await {
        Ok(body) => body,
        Err(error) => {
            return (axum::http::StatusCode::PAYLOAD_TOO_LARGE, error.to_string()).into_response()
        }
    };
    match dispatch(&gateway, parts.method, parts.uri, parts.headers, body).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn dispatch(
    gateway: &Gateway,
    method: axum::http::Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let json: Value =
        serde_json::from_slice(&body).map_err(|error| ApiError::BadRequest(error.to_string()))?;
    let object = json
        .as_object()
        .ok_or_else(|| ApiError::BadRequest("request body must be a JSON object".into()))?;
    let model = match object.get("model") {
        Some(Value::String(model)) => model.as_str(),
        Some(_) => return Err(ApiError::BadRequest("model must be a string".into())),
        None => gateway.default_model.as_deref().ok_or_else(|| {
            ApiError::BadRequest("model is required unless default-model is configured".into())
        })?,
    };
    let origin = gateway
        .routes
        .get(model)
        .ok_or_else(|| ApiError::ModelNotFound(model.to_string()))?;
    let mut target = origin.clone();
    target.set_path(uri.path());
    target.set_query(uri.query());
    let timeout_error = || ApiError::UpstreamTimeout {
        worker: origin.clone(),
    };
    let upstream_error = |error: reqwest::Error| {
        if error.is_timeout() {
            timeout_error()
        } else {
            ApiError::UpstreamUnreachable {
                worker: origin.clone(),
                source: error.into(),
            }
        }
    };
    let deadline = tokio::time::Instant::now() + gateway.ttft;
    let response = tokio::time::timeout_at(
        deadline,
        gateway
            .client
            .request(method, target)
            .headers(end_to_end_headers(&headers))
            .body(body)
            .send(),
    )
    .await
    .map_err(|_| timeout_error())?
    .map_err(upstream_error)?;
    let status = response.status();
    let headers = end_to_end_headers(response.headers());
    let mut stream = response.bytes_stream();
    // Commit downstream status only after the first chunk, so TTFT can return 504.
    let first = tokio::time::timeout_at(deadline, stream.next())
        .await
        .map_err(|_| timeout_error())?
        .transpose()
        .map_err(upstream_error)?;
    let idle = gateway.idle;
    // No worker hooks are needed here; body polling supplies backpressure and cancellation.
    let stream = futures::stream::try_unfold(stream, move |mut stream| async move {
        match tokio::time::timeout(idle, stream.next()).await {
            Ok(Some(chunk)) => chunk
                .map(|bytes| Some((bytes, stream)))
                .map_err(std::io::Error::other),
            Ok(None) => Ok(None),
            Err(_) => Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                "downstream response idle timeout",
            )),
        }
    });
    let stream = futures::stream::iter(first.map(Ok::<_, std::io::Error>)).chain(stream);
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    Ok(response)
}
