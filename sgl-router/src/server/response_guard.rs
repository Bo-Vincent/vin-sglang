// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use super::weighted_queue::QueuePermit;
use axum::body::{Body, Bytes, HttpBody};
use futures::Stream;
use http_body::{Frame, SizeHint};
use std::pin::Pin;
use std::task::{Context, Poll};

pub(super) struct Guarded<T> {
    inner: T,
    permit: Option<QueuePermit>,
}

impl<T> Guarded<T> {
    pub(super) fn new(inner: T, permit: QueuePermit) -> Self {
        Self {
            inner,
            permit: Some(permit),
        }
    }
}

pub(super) fn body(body: Body, permit: QueuePermit) -> Body {
    if body.is_end_stream() {
        return body;
    }
    Body::new(Guarded::new(body, permit))
}

impl HttpBody for Guarded<Body> {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_frame(cx);
        if matches!(&result, Poll::Ready(None | Some(Err(_)))) || this.inner.is_end_stream() {
            this.permit.take();
        }
        result
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

impl<S, T, E> Stream for Guarded<S>
where
    S: Stream<Item = Result<T, E>> + Unpin,
{
    type Item = Result<T, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_next(cx);
        if matches!(&result, Poll::Ready(None | Some(Err(_)))) {
            this.permit.take();
        }
        result
    }
}
