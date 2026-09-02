// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Cache-Aware prefix signals derived from the Router-local KV-event tree.

use super::ExternalPrefixSignal;
use crate::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree,
};
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Reads one Router-local radix tree and normalizes its per-worker contiguous
/// depths into the same signal shape consumed by the native Cache-Aware
/// policy for an Indexer lookup.
#[derive(Clone, Debug)]
pub struct RadixTreePrefixProvider {
    tree: Arc<HashTree>,
    block_size_oracle: Arc<BlockSizeOracle>,
}

impl RadixTreePrefixProvider {
    pub fn new(tree: Arc<HashTree>, block_size_oracle: Arc<BlockSizeOracle>) -> Self {
        Self {
            tree,
            block_size_oracle,
        }
    }

    /// Produces one best contiguous depth per worker URL. A worker may report
    /// multiple DP ranks; the deepest rank is its usable Cache-Aware depth.
    pub fn match_request_tokens(&self, tokens: &[u32]) -> Option<ExternalPrefixSignal> {
        let block_size = self.block_size_oracle.get()?;
        let hashes = if self.block_size_oracle.is_bigram() {
            compute_block_hashes_bigram(tokens, block_size as usize)
        } else {
            compute_block_hashes(tokens, block_size as usize)
        };
        if hashes.is_empty() {
            return None;
        }

        let mut depth_by_url = BTreeMap::<String, u32>::new();
        for (worker, depth) in self.tree.prefix_depths(None, &hashes) {
            let depth = u32::try_from(depth).unwrap_or(u32::MAX);
            depth_by_url
                .entry(worker.url)
                .and_modify(|current| *current = (*current).max(depth))
                .or_insert(depth);
        }
        let best_prefix_blocks = depth_by_url.values().copied().max()?;
        let matches = depth_by_url
            .into_iter()
            .map(|(address, matched_prefix_blocks)| PrefixMatch {
                worker_id: address.clone(),
                address,
                matched_prefix_blocks,
            })
            .collect();
        Some(ExternalPrefixSignal {
            outcome: PrefixOutcome::Matched {
                matches,
                best_prefix_blocks,
            },
            query_blocks: hashes.len(),
        })
    }
}
