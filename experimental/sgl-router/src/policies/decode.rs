// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Decode policy 的独立扩展点。
//!
//! Decode 不复用 Prefill 的 Session/Cache policy。它只在 Final Prefill 已确定后，
//! 对 Router 提供的 Decode candidate domain 提出 primary/backup；共享 D Admission
//! 和 Guard 再负责最终选择。

use crate::config::DecodePolicyKind;
use crate::load_monitor::LoadMonitorSnapshot;
use crate::policies::admission::{compare_decode_pressure, CandidateDomain};
use crate::policies::registry::select_decode_with_affinity;
use crate::policies::{ProposalKind, SelectionProposal};
use rand::Rng;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct DecodeSelectionContext<'a> {
    load_snapshot: Option<&'a LoadMonitorSnapshot>,
    prefill_url: Option<&'a str>,
}

impl<'a> DecodeSelectionContext<'a> {
    pub fn new() -> Self {
        Self {
            load_snapshot: None,
            prefill_url: None,
        }
    }

    /// 复用 ingress 一次性捕获的 snapshot；Decode policy 不自行向 LoadMonitor
    /// 查询。没有 fresh 数据时压力比较器自动降级到 local active-load。
    pub fn with_load_snapshot(mut self, load_snapshot: &'a LoadMonitorSnapshot) -> Self {
        self.load_snapshot = Some(load_snapshot);
        self
    }

    pub fn load_snapshot(&self) -> Option<&LoadMonitorSnapshot> {
        self.load_snapshot
    }

    /// `legacy_host_affinity` 的兼容输入。新的 P2 policy 不读取该值。
    pub fn with_prefill_url(mut self, prefill_url: &'a str) -> Self {
        self.prefill_url = Some(prefill_url);
        self
    }

    pub fn prefill_url(&self) -> Option<&str> {
        self.prefill_url
    }
}

pub trait DecodePolicy: Send + Sync + std::fmt::Debug {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal>;
}

/// Step 1 的 Decode 默认：在当前 Decode domain 内随机采样两个 worker，并由
/// 当前可靠的 D 压力比较器排列 primary/backup。它不会做 P→D transfer 估算。
#[derive(Debug, Default)]
pub struct DecodePowerOfTwoPolicy;

impl DecodePowerOfTwoPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl DecodePolicy for DecodePowerOfTwoPolicy {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        match domain.workers.len() {
            0 => None,
            1 => Some(
                SelectionProposal::primary(Arc::clone(&domain.workers[0]))
                    .with_kind(ProposalKind::PowerOfTwo),
            ),
            len => {
                let mut rng = rand::thread_rng();
                let i = rng.gen_range(0..len);
                let mut j = rng.gen_range(0..len - 1);
                if j >= i {
                    j += 1;
                }
                let left = &domain.workers[i];
                let right = &domain.workers[j];
                let (primary, backup) =
                    if compare_decode_pressure(left, right, ctx.load_snapshot()).is_gt() {
                        (Arc::clone(right), Arc::clone(left))
                    } else {
                        (Arc::clone(left), Arc::clone(right))
                    };
                Some(
                    SelectionProposal::with_backup(primary, backup)
                        .with_kind(ProposalKind::PowerOfTwo),
                )
            }
        }
    }
}

/// 旧 PD same-host Decode 选择的兼容 policy。
///
/// 它只提出一个 primary，因而不会因为新 Decode Guard 改写既有 host-affinity
/// 语义；primary 不通过 D Admission 时，仍由共享 domain fallback 选择一个可用 D。
#[derive(Debug, Default)]
pub struct LegacyHostAffinityDecodePolicy;

impl DecodePolicy for LegacyHostAffinityDecodePolicy {
    fn propose(
        &self,
        domain: &CandidateDomain,
        ctx: &DecodeSelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        let prefill_url = ctx.prefill_url()?;
        select_decode_with_affinity(prefill_url, &domain.workers).map(SelectionProposal::primary)
    }
}

/// 仅构造 role-local Decode policy；它不触碰 Prefill policy factory，也不承担
/// Bucket/SLO/transfer 选择。后者由 Router 先给它一个 CandidateDomain。
pub fn build_decode_policy(kind: DecodePolicyKind) -> Box<dyn DecodePolicy> {
    match kind {
        DecodePolicyKind::PowerOfTwo => Box::new(DecodePowerOfTwoPolicy::new()),
        DecodePolicyKind::LegacyHostAffinity => Box::new(LegacyHostAffinityDecodePolicy),
    }
}
