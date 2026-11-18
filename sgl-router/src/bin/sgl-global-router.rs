// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use axum::serve::ListenerExt;
use clap::Parser;
use sgl_router::global_router::{build_router, GlobalRouterArgs};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = GlobalRouterArgs::parse();
    let address = (args.host, args.port);
    let app = build_router(args)?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    let listener = listener.tap_io(|stream| {
        if let Err(error) = stream.set_nodelay(true) {
            eprintln!("failed to set TCP_NODELAY: {error}");
        }
    });
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
