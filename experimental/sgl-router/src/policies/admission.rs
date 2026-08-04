// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefill 共享 Admission 与 Guard。
//!
//! policy 只提出 `primary + backup`；本模块先应用硬容量约束，再在已准入的
//! 两个候选之间应用可选软 Guard。未来 Bucket 只需要替换
//! [`CandidateRange`] 的 worker 集、标识和可选 pending 上限。

use crate::load_monitor::{AggregateLoad, LoadMonitorSnapshot};
use crate::policies::{GuardHints, SelectionProposal};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::sync::Arc;

/// Router 在进入 policy 之前定义的候选域。
///
/// Step 1 固定使用 [`Self::global`]。Bucket/SLO 增量只能在这里缩小候选域，
/// 不能在 policy 对全局 worker 做完选择后再反向修正。
pub struct CandidateRange<'a> {
    pub id: &'a str,
    pub workers: &'a [Arc<Worker>],
    pub max_pending_prefill_tokens: Option<u64>,
}

impl<'a> CandidateRange<'a> {
    pub fn global(workers: &'a [Arc<Worker>]) -> Self {
        Self {
            id: "global",
            workers,
            max_pending_prefill_tokens: None,
        }
    }
}

/// 最终 worker 的来源，用于日志、指标和后续 Reservation 的输入。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    Primary,
    BackupPrimaryAdmission,
    BackupCacheBenefit,
    BackupPressureGuard,
    RangeFallback,
}

/// Admission / Guard 结束后的不可变选择结果。
#[derive(Clone)]
pub struct FinalDecision {
    pub selected: Arc<Worker>,
    pub primary: Arc<Worker>,
    pub backup: Option<Arc<Worker>>,
    pub reason: DecisionReason,
    pub candidate_range_id: String,
    pub load_snapshot_version: u64,
}

/// 解析一次 Prefill proposal。
///
/// `request_input_tokens` 使用 ingress 已有的 token 数或保守字节估算值。没有
/// fresh LoadMonitor 数据时，健康候选不做 hard reject；这保证 reporter 未启用、
/// 暂时缺报或过期时不会把整个 worker 池误判为不可用。
pub fn resolve_prefill(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &LoadMonitorSnapshot,
) -> Option<FinalDecision> {
    if !contains_worker(range, &proposal.primary) {
        return None;
    }
    let backup = proposal
        .backup
        .as_ref()
        .filter(|worker| contains_worker(range, worker))
        .cloned();
    let primary_admitted = is_admitted(range, &proposal.primary, request_input_tokens, snapshot);
    let backup_admitted = backup
        .as_ref()
        .is_some_and(|worker| is_admitted(range, worker, request_input_tokens, snapshot));

    let (selected, reason) = match (primary_admitted, backup.as_ref(), backup_admitted) {
        (true, Some(backup), true) => {
            if cache_benefit_prefers_backup(&proposal.guard_hints, request_input_tokens) {
                (Arc::clone(backup), DecisionReason::BackupCacheBenefit)
            } else if pressure_guard_prefers_backup(
                &proposal.primary,
                backup,
                &proposal.guard_hints,
                snapshot,
            ) {
                (Arc::clone(backup), DecisionReason::BackupPressureGuard)
            } else {
                (Arc::clone(&proposal.primary), DecisionReason::Primary)
            }
        }
        (true, _, _) => (Arc::clone(&proposal.primary), DecisionReason::Primary),
        (false, Some(backup), true) => (Arc::clone(backup), DecisionReason::BackupPrimaryAdmission),
        _ => range_fallback(range, proposal, request_input_tokens, snapshot)?,
    };

    Some(FinalDecision {
        selected,
        primary: Arc::clone(&proposal.primary),
        backup,
        reason,
        candidate_range_id: range.id.to_string(),
        load_snapshot_version: snapshot.version,
    })
}

fn contains_worker(range: &CandidateRange<'_>, candidate: &Arc<Worker>) -> bool {
    range.workers.iter().any(|worker| worker.id == candidate.id)
}

/// 只把 fresh engine snapshot 作为 hard 容量依据。缺失或过期 report 退化为
/// registry 已完成的健康筛选，避免把 LoadMonitor 可用性变成请求可用性的单点。
fn is_admitted(
    range: &CandidateRange<'_>,
    worker: &Arc<Worker>,
    request_input_tokens: u64,
    snapshot: &LoadMonitorSnapshot,
) -> bool {
    let Some(load) = snapshot.fresh_load(&worker.id) else {
        return true;
    };

    load.num_running_reqs.saturating_add(1) <= load.max_running_requests
        && load.num_total_tokens.saturating_add(request_input_tokens)
            <= load.max_total_num_tokens
        && range.max_pending_prefill_tokens.map_or(true, |limit| {
            load.num_waiting_uncached_tokens
                .saturating_add(request_input_tokens)
                <= limit
        })
}

fn cache_benefit_prefers_backup(hints: &GuardHints, request_input_tokens: u64) -> bool {
    if !hints.enable_cache_benefit {
        return false;
    }
    let Some(saved) = hints.matched_prefix_tokens else {
        return false;
    };
    saved <= request_input_tokens.saturating_sub(saved)
}

/// 压力逃逸只在两个候选都有 fresh snapshot 时成立；这不是 queue-ms 预测，
/// 仅比较可直接采集的等待未命中 token 压力。
fn pressure_guard_prefers_backup(
    primary: &Arc<Worker>,
    backup: &Arc<Worker>,
    hints: &GuardHints,
    snapshot: &LoadMonitorSnapshot,
) -> bool {
    if !hints.enable_pressure_guard {
        return false;
    }
    let (Some(primary_load), Some(backup_load)) = (
        snapshot.fresh_load(&primary.id),
        snapshot.fresh_load(&backup.id),
    ) else {
        return false;
    };
    let primary_pressure = primary_load.num_waiting_uncached_tokens;
    let backup_pressure = backup_load.num_waiting_uncached_tokens;
    primary_pressure.saturating_sub(backup_pressure) > hints.pressure_abs_threshold_tokens
        && (primary_pressure as f64)
            > (backup_pressure as f64) * hints.pressure_rel_threshold
}

fn range_fallback(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &LoadMonitorSnapshot,
) -> Option<(Arc<Worker>, DecisionReason)> {
    proposal
        .eligible_workers
        .as_deref()
        .unwrap_or(range.workers)
        .iter()
        .filter(|worker| contains_worker(range, worker))
        .filter(|worker| is_admitted(range, worker, request_input_tokens, snapshot))
        .min_by(|left, right| compare_prefill_pressure(left, right, Some(snapshot)))
        .cloned()
        .map(|worker| (worker, DecisionReason::RangeFallback))
}

/// 在两者都带有 fresh report 时使用相同精度的多维压力比较；否则退化为
/// router local active-load，避免将一个有 snapshot 的 worker 与一个没有
/// snapshot 的 worker 的不同精度字段混在一个排序键中。
/// 在同一候选域上比较 Prefill 压力。P2、Cache-Aware 并列命中和 Admission
/// fallback 共用这个函数，因此它们不会各自解释一套 LoadMonitor 字段。
/// `None` 代表调用点没有可用快照，退化为 Router 本地 active-load。
pub(crate) fn compare_prefill_pressure(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    snapshot: Option<&LoadMonitorSnapshot>,
) -> Ordering {
    match snapshot.and_then(|snapshot| {
        Some((
            snapshot.fresh_load(&left.id)?,
            snapshot.fresh_load(&right.id)?,
        ))
    }) {
        Some((left_load, right_load)) => load_pressure_key(&left_load)
            .cmp(&load_pressure_key(&right_load))
            .then_with(|| left.active_load().cmp(&right.active_load()))
            .then_with(|| left.id.0.cmp(&right.id.0)),
        _ => left
            .active_load()
            .cmp(&right.active_load())
            .then_with(|| left.id.0.cmp(&right.id.0)),
    }
}

fn load_pressure_key(load: &AggregateLoad) -> (u64, u64, u64) {
    (
        load.num_waiting_uncached_tokens,
        load.num_waiting_reqs,
        load.num_running_reqs,
    )
}
