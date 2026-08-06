// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Candidate-domain seam reserved for Step 2 Bucket/SLO routing.
//!
//! Step 1 deliberately exposes only the global P/D domains. Keeping this
//! adapter in the request path lets Step 2 replace domain construction without
//! changing policy, Admission, Guard, or dispatch control flow.

use crate::config::BucketConfig;
use crate::policies::admission::CandidateDomain;
use crate::policies::CacheCandidate;
use crate::workers::Worker;
use std::sync::Arc;

/// Request facts consumed by the optional Step 2 domain resolver.
#[derive(Debug, Clone, Copy)]
pub struct BucketRequest {
    pub input_tokens: u64,
    pub expected_peak_sequence_tokens: Option<u64>,
    pub ttft_slo_ms: Option<u64>,
    pub tps_slo: Option<f64>,
}

/// Step 1 global-domain adapter. Step 2 activates static Bucket resolution.
#[derive(Debug, Clone, Default)]
pub struct BucketSelector;

impl BucketSelector {
    pub fn new(_config: Option<BucketConfig>) -> Self {
        Self
    }

    pub fn is_enabled(&self) -> bool {
        false
    }

    pub fn prefill_domains(
        &self,
        workers: &[Arc<Worker>],
        _request: BucketRequest,
    ) -> Vec<CandidateDomain> {
        vec![CandidateDomain::global_prefill(workers)]
    }

    pub fn decode_domains(
        &self,
        workers: &[Arc<Worker>],
        _request: BucketRequest,
    ) -> Vec<CandidateDomain> {
        vec![CandidateDomain::global_decode(workers)]
    }

    pub fn bind_prefill_cache_candidate(
        &self,
        mut candidate: CacheCandidate,
        _request: BucketRequest,
    ) -> Option<CacheCandidate> {
        candidate.candidate_range_id = "global".to_string();
        candidate.max_pending_prefill_tokens = None;
        Some(candidate)
    }

    pub fn prefill_affinity_domain(
        &self,
        _workers: &[Arc<Worker>],
        _primary: &Arc<Worker>,
        _request: BucketRequest,
    ) -> Option<CandidateDomain> {
        None
    }
}
