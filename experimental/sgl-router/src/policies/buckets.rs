// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! 静态 Bucket 到候选域的纯转换。
//!
//! 该模块不比较 worker 压力，也不做 Admission。它只根据请求长度、离线 SLO
//! profile 和唯一 rank 产生有序的 CandidateDomain；Router 对每个 domain 重新
//! 运行 policy → admission → guard，因而不会复用上一 Bucket 的 primary/backup。

use crate::config::{BucketConfig, BucketSpec, BucketStage, SloBucketPolicy};
use crate::policies::admission::CandidateDomain;
use crate::workers::Worker;
use std::sync::Arc;

/// Bucket 选择所需、且 ingress 可实际给出的请求事实。
#[derive(Debug, Clone, Copy)]
pub struct BucketRequest {
    pub input_tokens: u64,
    pub expected_peak_sequence_tokens: Option<u64>,
    pub ttft_slo_ms: Option<u64>,
    pub tps_slo: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct BucketSelector {
    config: Option<BucketConfig>,
}

impl BucketSelector {
    pub fn new(config: Option<BucketConfig>) -> Self {
        Self { config }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.is_some()
    }

    pub fn prefill_domains(
        &self,
        workers: &[Arc<Worker>],
        request: BucketRequest,
    ) -> Vec<CandidateDomain> {
        let Some(config) = &self.config else {
            return vec![CandidateDomain::global_prefill(workers)];
        };
        self.ordered_specs(
            BucketStage::Prefill,
            request,
            config.ttft_slo_policy,
            |spec| prefill_compatible(spec, request.input_tokens),
            |spec| ttft_eligible(spec, request.ttft_slo_ms),
        )
        .into_iter()
        .filter_map(|spec| {
            let members = members(workers, spec);
            (!members.is_empty()).then(|| {
                CandidateDomain::bucket_prefill(
                    spec.id.clone(),
                    members,
                    spec.max_pending_prefill_tokens,
                )
            })
        })
        .collect()
    }

    pub fn decode_domains(
        &self,
        workers: &[Arc<Worker>],
        request: BucketRequest,
    ) -> Vec<CandidateDomain> {
        let Some(config) = &self.config else {
            return vec![CandidateDomain::global_decode(workers)];
        };
        self.ordered_specs(
            BucketStage::Decode,
            request,
            config.tps_slo_policy,
            |spec| {
                decode_compatible(
                    spec,
                    request.input_tokens,
                    request.expected_peak_sequence_tokens,
                )
            },
            |spec| tps_eligible(spec, request.tps_slo),
        )
        .into_iter()
        .filter_map(|spec| {
            let members = members(workers, spec);
            (!members.is_empty()).then(|| CandidateDomain::bucket_decode(spec.id.clone(), members))
        })
        .collect()
    }

    /// 为全局 Session/Cache primary 找到它**自己**所属的 Prefill Bucket。
    ///
    /// `global-first` / `global` 的目的正是允许一个已命中的 primary 跨过当前
    /// 请求的 prompt-length target Bucket。因此此处不检查 `min/max_input_tokens`
    /// （那是 normal target selection 的分桶条件）；仍严格检查真实 runtime
    /// context 上限、Hard TTFT profile 和当前活跃 worker membership。返回的
    /// Domain 随后仍要完整经过 policy → Admission → Guard。
    pub fn prefill_affinity_domain(
        &self,
        workers: &[Arc<Worker>],
        primary: &Arc<Worker>,
        request: BucketRequest,
    ) -> Option<CandidateDomain> {
        let config = self.config.as_ref()?;
        let spec = config.buckets.iter().find(|spec| {
            spec.stage == BucketStage::Prefill
                && spec.worker_ids.iter().any(|id| id == &primary.id.0)
                && spec
                    .max_context_tokens
                    .is_none_or(|max_context| request.input_tokens <= max_context)
                && (config.ttft_slo_policy != SloBucketPolicy::SloFirst
                    || ttft_eligible(spec, request.ttft_slo_ms))
        })?;
        let members = members(workers, spec);
        members
            .iter()
            .any(|worker| worker.id == primary.id)
            .then(|| {
                CandidateDomain::bucket_prefill(
                    spec.id.clone(),
                    members,
                    spec.max_pending_prefill_tokens,
                )
            })
    }

    fn ordered_specs<'a>(
        &'a self,
        stage: BucketStage,
        _request: BucketRequest,
        slo_policy: SloBucketPolicy,
        compatible: impl Fn(&BucketSpec) -> bool,
        slo_eligible: impl Fn(&BucketSpec) -> bool,
    ) -> Vec<&'a BucketSpec> {
        let Some(config) = &self.config else {
            return Vec::new();
        };
        let mut compatible_specs: Vec<&BucketSpec> = config
            .buckets
            .iter()
            .filter(|spec| spec.stage == stage && compatible(spec))
            .collect();
        compatible_specs.sort_by(|left, right| {
            left.rank
                .cmp(&right.rank)
                .then_with(|| left.id.cmp(&right.id))
        });

        if slo_policy != SloBucketPolicy::SloFirst {
            return compatible_specs;
        }

        let mut eligible = Vec::new();
        let mut degraded = Vec::new();
        for spec in compatible_specs {
            if slo_eligible(spec) {
                eligible.push(spec);
            } else {
                degraded.push(spec);
            }
        }
        eligible.extend(degraded);
        eligible
    }
}

fn members(workers: &[Arc<Worker>], spec: &BucketSpec) -> Vec<Arc<Worker>> {
    workers
        .iter()
        .filter(|worker| spec.worker_ids.iter().any(|id| id == &worker.id.0))
        .cloned()
        .collect()
}

fn prefill_compatible(spec: &BucketSpec, input_tokens: u64) -> bool {
    within(input_tokens, spec.min_input_tokens, spec.max_input_tokens)
        && spec
            .max_context_tokens
            .is_none_or(|max_context| input_tokens <= max_context)
}

fn decode_compatible(
    spec: &BucketSpec,
    input_tokens: u64,
    expected_peak_sequence_tokens: Option<u64>,
) -> bool {
    let Some(expected_peak_sequence_tokens) = expected_peak_sequence_tokens else {
        // output budget 不可信时只允许显式 catch-all Decode Bucket；不能伪造
        // precise sequence length 然后送往一个短上下文 Runtime。
        return spec.min_sequence_tokens.is_none()
            && spec.max_sequence_tokens.is_none()
            && spec
                .max_context_tokens
                .is_none_or(|max_context| input_tokens <= max_context);
    };
    within(
        expected_peak_sequence_tokens,
        spec.min_sequence_tokens,
        spec.max_sequence_tokens,
    ) && spec
        .max_context_tokens
        .is_none_or(|max_context| expected_peak_sequence_tokens <= max_context)
}

fn within(value: u64, min: Option<u64>, max: Option<u64>) -> bool {
    min.is_none_or(|min| value >= min) && max.is_none_or(|max| value <= max)
}

fn ttft_eligible(spec: &BucketSpec, request_ttft_slo_ms: Option<u64>) -> bool {
    let Some(request_ttft_slo_ms) = request_ttft_slo_ms else {
        return true;
    };
    spec.ttft_p95_at_capacity_ms
        .is_some_and(|p95| p95 <= request_ttft_slo_ms)
}

fn tps_eligible(spec: &BucketSpec, request_tps_slo: Option<f64>) -> bool {
    let Some(request_tps_slo) = request_tps_slo else {
        return true;
    };
    spec.tps_p05_at_capacity
        .is_some_and(|p05| p05 >= request_tps_slo)
}
