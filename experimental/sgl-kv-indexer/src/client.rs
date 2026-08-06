// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Router-facing client for advisory external KV prefix lookup.

use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;
use tonic::transport::{Channel, Endpoint};

use crate::pb::kv_indexer_client::KvIndexerClient;
use crate::pb::{ExternalKvNodeMatch, MatchExternalKvRequest};

/// One worker's reusable contiguous request prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixMatch {
    pub worker_id: String,
    pub matched_prefix_blocks: u32,
}

/// Why an external lookup did not provide a usable signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NoSignalReason {
    Empty,
    Unreachable,
    Timeout,
    Saturated,
    Rejected,
}

/// The Router treats the index as advisory: every lookup failure has the same
/// safe effect as a cache miss and never fails inference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixOutcome {
    Matched { matches: Vec<PrefixMatch> },
    NoSignal(NoSignalReason),
}

/// Router-side bounds for one KV Indexer endpoint.
#[derive(Debug, Clone)]
pub struct PrefixIndexConfig {
    pub endpoint: String,
    pub query_deadline: Duration,
    pub max_inflight: usize,
}

/// gRPC implementation of the advisory prefix index.
#[derive(Debug)]
pub struct GrpcPrefixIndex {
    channel: Option<Channel>,
    deadline: Duration,
    inflight: Arc<Semaphore>,
}

impl GrpcPrefixIndex {
    /// Invalid endpoint syntax is deliberately converted into a later
    /// `NoSignal(Unreachable)`: indexer availability must not prevent Router
    /// startup or requests from reaching a healthy worker.
    pub fn new(config: PrefixIndexConfig) -> Self {
        let channel = Endpoint::from_shared(config.endpoint)
            .ok()
            .map(|endpoint| endpoint.connect_lazy());
        Self {
            channel,
            deadline: config.query_deadline,
            inflight: Arc::new(Semaphore::new(config.max_inflight.max(1))),
        }
    }

    /// Query request-order block hashes and derive each worker's contiguous
    /// prefix from the Indexer's block-placement response.
    async fn query(&self, hashes: Vec<String>) -> PrefixOutcome {
        if hashes.is_empty() {
            return PrefixOutcome::NoSignal(NoSignalReason::Empty);
        }
        let Some(channel) = self.channel.clone() else {
            return PrefixOutcome::NoSignal(NoSignalReason::Unreachable);
        };
        let Ok(_permit) = Arc::clone(&self.inflight).try_acquire_owned() else {
            return PrefixOutcome::NoSignal(NoSignalReason::Saturated);
        };

        let mut client = KvIndexerClient::new(channel);
        let request = MatchExternalKvRequest {
            hashes: hashes.clone(),
            count_as_hit: true,
        };
        match tokio::time::timeout(self.deadline, client.match_external_kv(request)).await {
            Err(_) => PrefixOutcome::NoSignal(NoSignalReason::Timeout),
            Ok(Err(status)) if status.code() == tonic::Code::Unavailable => {
                PrefixOutcome::NoSignal(NoSignalReason::Unreachable)
            }
            Ok(Err(_)) => PrefixOutcome::NoSignal(NoSignalReason::Rejected),
            Ok(Ok(response)) => {
                let mut matches = response
                    .into_inner()
                    .matches
                    .into_iter()
                    .filter_map(|node| prefix_match(&hashes, node))
                    .collect::<Vec<_>>();
                matches.sort_by(|left, right| {
                    right
                        .matched_prefix_blocks
                        .cmp(&left.matched_prefix_blocks)
                        .then_with(|| left.worker_id.cmp(&right.worker_id))
                });
                if matches.is_empty() {
                    PrefixOutcome::NoSignal(NoSignalReason::Empty)
                } else {
                    PrefixOutcome::Matched { matches }
                }
            }
        }
    }
}

/// Router-facing prefix lookup seam. The production client and proxy tests use
/// the same advisory contract: missing or failed cache information never makes
/// a healthy request fail.
#[tonic::async_trait]
pub trait PrefixIndex: Send + Sync + Debug {
    async fn match_prefix(&self, hashes: Vec<String>) -> PrefixOutcome;
}

#[tonic::async_trait]
impl PrefixIndex for GrpcPrefixIndex {
    async fn match_prefix(&self, hashes: Vec<String>) -> PrefixOutcome {
        self.query(hashes).await
    }
}

fn prefix_match(hashes: &[String], node: ExternalKvNodeMatch) -> Option<PrefixMatch> {
    let held: HashSet<&str> = node
        .hashes_by_tier
        .iter()
        .flat_map(|tier| tier.hashes.iter().map(String::as_str))
        .collect();
    let matched_prefix_blocks = hashes
        .iter()
        .take_while(|hash| held.contains(hash.as_str()))
        .count()
        .try_into()
        .unwrap_or(u32::MAX);
    (matched_prefix_blocks > 0).then_some(PrefixMatch {
        worker_id: node.worker_id,
        matched_prefix_blocks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb::TierHashes;

    #[test]
    fn derives_only_the_contiguous_request_prefix() {
        let hashes = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let node = ExternalKvNodeMatch {
            worker_id: "p0".into(),
            address: "unused-by-router".into(),
            hashes_by_tier: vec![TierHashes {
                tier: 1,
                hashes: vec!["a".into(), "b".into(), "d".into()],
            }],
        };
        assert_eq!(
            prefix_match(&hashes, node),
            Some(PrefixMatch {
                worker_id: "p0".into(),
                matched_prefix_blocks: 2,
            })
        );
    }
}
