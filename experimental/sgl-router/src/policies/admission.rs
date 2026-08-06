// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefill / Decode 共享 Admission 与 Guard。
//!
//! Pair proposal 先做容量检查再执行 Guard；Cache-Aware 候选走有界比较。

use crate::discovery::WorkerId;
use crate::load_monitor::{AggregateLoad, SchedulingSnapshot};
use crate::policies::{
    CacheCandidate, CacheCandidateProposal, CandidateMembership, GuardHints, SelectionProposal,
};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

/// Prefill policy 的候选域及其容量上限。
pub struct CandidateRange<'a> {
    pub id: &'a str,
    pub workers: &'a [Arc<Worker>],
    pub max_pending_prefill_tokens: Option<u64>,
    membership: Option<&'a CandidateMembership>,
}

impl<'a> CandidateRange<'a> {
    pub fn global(workers: &'a [Arc<Worker>]) -> Self {
        Self {
            id: "global",
            workers,
            max_pending_prefill_tokens: None,
            membership: None,
        }
    }

    pub fn membership(&self) -> Option<&CandidateMembership> {
        self.membership
    }
}

/// Router 在 policy 前解析出的角色化候选域。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStage {
    Prefill,
    Decode,
}

#[derive(Clone)]
pub struct CandidateDomain {
    pub id: String,
    pub stage: RoutingStage,
    pub workers: Vec<Arc<Worker>>,
    pub max_pending_prefill_tokens: Option<u64>,
    membership: CandidateMembership,
}

impl CandidateDomain {
    pub fn global_prefill(workers: &[Arc<Worker>]) -> Self {
        Self::new("global", RoutingStage::Prefill, workers.to_vec(), None)
    }

    pub fn global_decode(workers: &[Arc<Worker>]) -> Self {
        Self::new("global", RoutingStage::Decode, workers.to_vec(), None)
    }

    pub fn bucket_prefill(
        id: impl Into<String>,
        workers: Vec<Arc<Worker>>,
        max_pending_prefill_tokens: Option<u64>,
    ) -> Self {
        Self::new(
            id,
            RoutingStage::Prefill,
            workers,
            max_pending_prefill_tokens,
        )
    }

    pub fn bucket_decode(id: impl Into<String>, workers: Vec<Arc<Worker>>) -> Self {
        Self::new(id, RoutingStage::Decode, workers, None)
    }

    fn new(
        id: impl Into<String>,
        stage: RoutingStage,
        workers: Vec<Arc<Worker>>,
        max_pending_prefill_tokens: Option<u64>,
    ) -> Self {
        let membership = CandidateMembership::from_workers(&workers);
        Self {
            id: id.into(),
            stage,
            workers,
            max_pending_prefill_tokens,
            membership,
        }
    }

    /// 将统一 Domain 投影为 Prefill Admission 使用的借用视图。
    pub fn prefill_range(&self) -> Option<CandidateRange<'_>> {
        (self.stage == RoutingStage::Prefill).then(|| CandidateRange {
            id: self.id.as_str(),
            workers: &self.workers,
            max_pending_prefill_tokens: self.max_pending_prefill_tokens,
            membership: Some(&self.membership),
        })
    }
}

/// 最终 worker 的选择原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    Primary,
    CacheCandidate,
    BackupPrimaryAdmission,
    BackupPressureGuard,
    RangeFallback,
}

/// Resolve Cache-Aware candidates inside one margin above the admitted
/// minimum-work floor. The winner is final; `None` starts no-hit fallback.
pub fn resolve_cache_candidates(
    proposal: &CacheCandidateProposal,
    request_input_tokens: u64,
    snapshot: &SchedulingSnapshot,
) -> Option<FinalDecision> {
    let loads = FreshLoadLookup::new(
        Some(snapshot),
        proposal
            .candidates
            .iter()
            .map(|candidate| &candidate.worker),
    );
    let admitted: Vec<&CacheCandidate> = proposal
        .candidates
        .iter()
        .filter(|candidate| is_cache_candidate_admitted(candidate, request_input_tokens, &loads))
        .collect();
    let work_floor = admitted
        .iter()
        .copied()
        .min_by_key(|candidate| candidate.uncached_tokens)?;
    let near_tie_ceiling = work_floor
        .uncached_tokens
        .saturating_add(proposal.cache_switch_margin_tokens);
    let mut winner = work_floor;
    for candidate in admitted {
        if candidate.uncached_tokens > near_tie_ceiling {
            continue;
        }
        if compare_cache_candidates(winner, candidate, proposal, &loads).is_gt() {
            winner = candidate;
        }
    }
    Some(FinalDecision {
        selected: Arc::clone(&winner.worker),
        primary: Arc::clone(&winner.worker),
        backup: None,
        reason: DecisionReason::CacheCandidate,
        candidate_range_id: winner.candidate_range_id.clone(),
        load_snapshot_version: snapshot.version,
    })
}

/// Admission / Guard 的最终结果。
#[derive(Clone)]
pub struct FinalDecision {
    pub selected: Arc<Worker>,
    pub primary: Arc<Worker>,
    pub backup: Option<Arc<Worker>>,
    pub reason: DecisionReason,
    pub candidate_range_id: String,
    pub load_snapshot_version: u64,
}

/// 解析 Prefill proposal。缺少 fresh LoadMonitor 数据时退化为 registry 健康结果。
pub fn resolve_prefill(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &SchedulingSnapshot,
) -> Option<FinalDecision> {
    if !contains_worker(range, &proposal.primary) {
        return None;
    }
    let backup = proposal
        .backup
        .as_ref()
        .filter(|worker| contains_worker(range, worker))
        .cloned();
    let primary_admitted = is_proposal_worker_eligible(proposal, &proposal.primary)
        && is_admitted(range, &proposal.primary, request_input_tokens, snapshot);
    let backup_admitted = backup.as_ref().is_some_and(|worker| {
        is_proposal_worker_eligible(proposal, worker)
            && is_admitted(range, worker, request_input_tokens, snapshot)
    });

    let (selected, reason) = match (primary_admitted, backup.as_ref(), backup_admitted) {
        (true, Some(backup), true) => {
            if pressure_guard_prefers_backup(
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

/// 解析 Decode proposal。缺少 fresh snapshot 时不做容量拒绝。
pub fn resolve_decode(
    domain: &CandidateDomain,
    proposal: &SelectionProposal,
    request_kv_tokens: u64,
    snapshot: &SchedulingSnapshot,
) -> Option<FinalDecision> {
    if domain.stage != RoutingStage::Decode || !contains_domain_worker(domain, &proposal.primary) {
        return None;
    }
    let backup = proposal
        .backup
        .as_ref()
        .filter(|worker| contains_domain_worker(domain, worker))
        .cloned();
    let primary_admitted = is_decode_admitted(&proposal.primary, request_kv_tokens, snapshot);
    let backup_admitted = backup
        .as_ref()
        .is_some_and(|worker| is_decode_admitted(worker, request_kv_tokens, snapshot));

    let (selected, reason) = match (primary_admitted, backup.as_ref(), backup_admitted) {
        (true, Some(backup), true) => {
            if compare_decode_pressure(&proposal.primary, backup, Some(snapshot)).is_gt() {
                (Arc::clone(backup), DecisionReason::BackupPressureGuard)
            } else {
                (Arc::clone(&proposal.primary), DecisionReason::Primary)
            }
        }
        (true, _, _) => (Arc::clone(&proposal.primary), DecisionReason::Primary),
        (false, Some(backup), true) => (Arc::clone(backup), DecisionReason::BackupPrimaryAdmission),
        _ => decode_domain_fallback(domain, request_kv_tokens, snapshot)?,
    };

    Some(FinalDecision {
        selected,
        primary: Arc::clone(&proposal.primary),
        backup,
        reason,
        candidate_range_id: domain.id.clone(),
        load_snapshot_version: snapshot.version,
    })
}

fn contains_worker(range: &CandidateRange<'_>, candidate: &Arc<Worker>) -> bool {
    // Policy 通常克隆 Domain 内的 Arc；ID 扫描仅兼容等价的独立 Worker 实例。
    range
        .membership
        .is_some_and(|membership| membership.contains(candidate))
        || range.workers.iter().any(|worker| worker.id == candidate.id)
}

fn contains_domain_worker(domain: &CandidateDomain, candidate: &Arc<Worker>) -> bool {
    domain.membership.contains(candidate)
        || domain
            .workers
            .iter()
            .any(|worker| worker.id == candidate.id)
}

fn is_proposal_worker_eligible(proposal: &SelectionProposal, candidate: &Arc<Worker>) -> bool {
    proposal
        .eligible_workers
        .as_ref()
        .is_none_or(|workers| workers.iter().any(|worker| worker.id == candidate.id))
}

/// 只有 fresh engine snapshot 参与容量判断。
fn is_admitted(
    range: &CandidateRange<'_>,
    worker: &Arc<Worker>,
    request_input_tokens: u64,
    snapshot: &SchedulingSnapshot,
) -> bool {
    let load = snapshot.fresh_load(&worker.id);
    is_prefill_admitted_with_load(range, request_input_tokens, load)
}

fn is_prefill_admitted_with_load(
    range: &CandidateRange<'_>,
    request_input_tokens: u64,
    load: Option<&AggregateLoad>,
) -> bool {
    let Some(load) = load else {
        return true;
    };
    load.num_running_reqs.saturating_add(1) <= load.max_running_requests
        && load.num_total_tokens.saturating_add(request_input_tokens) <= load.max_total_num_tokens
        && range.max_pending_prefill_tokens.is_none_or(|limit| {
            load.num_waiting_uncached_tokens
                .saturating_add(request_input_tokens)
                <= limit
        })
}

fn is_decode_admitted(
    worker: &Arc<Worker>,
    request_kv_tokens: u64,
    snapshot: &SchedulingSnapshot,
) -> bool {
    let load = snapshot.fresh_load(&worker.id);
    is_decode_admitted_with_load(request_kv_tokens, load)
}

fn is_decode_admitted_with_load(request_kv_tokens: u64, load: Option<&AggregateLoad>) -> bool {
    let Some(load) = load else {
        return true;
    };
    load.num_running_reqs.saturating_add(1) <= load.max_running_requests
        && load.num_total_tokens.saturating_add(request_kv_tokens) <= load.max_total_num_tokens
}

fn is_cache_candidate_admitted(
    candidate: &CacheCandidate,
    request_input_tokens: u64,
    loads: &FreshLoadLookup<'_>,
) -> bool {
    let Some(load) = loads.get(&candidate.worker.id) else {
        return true;
    };
    load.num_running_reqs.saturating_add(1) <= load.max_running_requests
        && load.num_total_tokens.saturating_add(request_input_tokens) <= load.max_total_num_tokens
        && candidate.max_pending_prefill_tokens.is_none_or(|limit| {
            load.num_waiting_uncached_tokens
                .saturating_add(candidate.uncached_tokens)
                <= limit
        })
}

/// Compare admitted cache candidates. `Less` means `left` is preferred.
fn compare_cache_candidates(
    left: &CacheCandidate,
    right: &CacheCandidate,
    proposal: &CacheCandidateProposal,
    loads: &FreshLoadLookup<'_>,
) -> Ordering {
    let work_delta = left.uncached_tokens.abs_diff(right.uncached_tokens);
    if work_delta > proposal.cache_switch_margin_tokens {
        return left
            .uncached_tokens
            .cmp(&right.uncached_tokens)
            .then_with(|| loads.compare_prefill_pressure(&left.worker, &right.worker))
            .then_with(|| left.worker.id.0.cmp(&right.worker.id.0));
    }

    // Pressure can overturn cache benefit only inside the near-tie margin.
    if proposal.enable_pressure_guard {
        if materially_more_pressured(
            &left.worker,
            &right.worker,
            proposal.pressure_abs_threshold_tokens,
            proposal.pressure_abs_threshold_ms,
            proposal.pressure_rel_threshold,
            loads,
        ) {
            return Ordering::Greater;
        }
        if materially_more_pressured(
            &right.worker,
            &left.worker,
            proposal.pressure_abs_threshold_tokens,
            proposal.pressure_abs_threshold_ms,
            proposal.pressure_rel_threshold,
            loads,
        ) {
            return Ordering::Less;
        }
    }

    left.uncached_tokens
        .cmp(&right.uncached_tokens)
        .then_with(|| loads.compare_prefill_pressure(&left.worker, &right.worker))
        .then_with(|| left.worker.id.0.cmp(&right.worker.id.0))
}

fn materially_more_pressured(
    candidate: &Arc<Worker>,
    other: &Arc<Worker>,
    absolute_threshold: u64,
    absolute_threshold_ms: Option<f64>,
    relative_threshold: f64,
    loads: &FreshLoadLookup<'_>,
) -> bool {
    let (Some(candidate_load), Some(other_load)) = (
        loads.comparable_get(&candidate.id),
        loads.comparable_get(&other.id),
    ) else {
        return false;
    };
    if let (Some(absolute_threshold_ms), true) =
        (absolute_threshold_ms, loads.compare_prefill_queue_ms)
    {
        let candidate_pressure = candidate_load
            .estimated_prefill_queue_ms
            .expect("complete candidate set has queue estimates");
        let other_pressure = other_load
            .estimated_prefill_queue_ms
            .expect("complete candidate set has queue estimates");
        return candidate_pressure - other_pressure > absolute_threshold_ms
            && candidate_pressure > other_pressure * relative_threshold;
    }
    let candidate_pressure = candidate_load.num_waiting_uncached_tokens;
    let other_pressure = other_load.num_waiting_uncached_tokens;
    candidate_pressure.saturating_sub(other_pressure) > absolute_threshold
        && candidate_pressure as f64 > other_pressure as f64 * relative_threshold
}

/// Request-local lookup for fresh LoadMonitor data and captured local load.
pub(crate) struct FreshLoadLookup<'a> {
    snapshot: Option<&'a SchedulingSnapshot>,
    local_active_by_worker_id: HashMap<String, usize>,
    compare_aggregate: bool,
    compare_prefill_queue_ms: bool,
    compare_decode_queues: bool,
    compare_decode_step_ms: bool,
    compare_decode_active_tokens: bool,
}

impl<'a> FreshLoadLookup<'a> {
    pub(crate) fn new<'w>(
        snapshot: Option<&'a SchedulingSnapshot>,
        workers: impl IntoIterator<Item = &'w Arc<Worker>>,
    ) -> Self {
        let mut local_active_by_worker_id = HashMap::new();
        let mut compare_aggregate = snapshot.is_some();
        let mut compare_prefill_queue_ms = true;
        let mut compare_decode_queues = true;
        let mut compare_decode_step_ms = true;
        let mut compare_decode_active_tokens = true;
        // Capture local load and snapshot capability in one pass. Mixed
        // fresh/stale sets use local load only to preserve transitivity.
        for worker in workers {
            local_active_by_worker_id.insert(worker.id.0.clone(), worker.active_load());
            let Some(load) = snapshot.and_then(|snapshot| snapshot.fresh_load(&worker.id)) else {
                compare_aggregate = false;
                continue;
            };
            compare_prefill_queue_ms &= load.estimated_prefill_queue_ms.is_some();
            compare_decode_queues &= load.decode_prealloc_queue_reqs.is_some()
                && load.decode_transfer_queue_reqs.is_some()
                && load.decode_retracted_queue_reqs.is_some();
            compare_decode_step_ms &= load.mean_decode_step_ms.is_some();
            compare_decode_active_tokens &= load.num_active_tokens.is_some();
        }
        compare_aggregate &= !local_active_by_worker_id.is_empty();
        compare_prefill_queue_ms &= compare_aggregate;
        compare_decode_queues &= compare_aggregate;
        compare_decode_step_ms &= compare_aggregate;
        compare_decode_active_tokens &= compare_aggregate;
        Self {
            snapshot,
            local_active_by_worker_id,
            compare_aggregate,
            compare_prefill_queue_ms,
            compare_decode_queues,
            compare_decode_step_ms,
            compare_decode_active_tokens,
        }
    }

    pub(crate) fn get(&self, worker_id: &WorkerId) -> Option<&AggregateLoad> {
        self.snapshot?.fresh_load(worker_id)
    }

    fn comparable_get(&self, worker_id: &WorkerId) -> Option<&AggregateLoad> {
        if self.compare_aggregate {
            self.get(worker_id)
        } else {
            None
        }
    }

    pub(crate) fn compare_prefill_pressure(
        &self,
        left: &Arc<Worker>,
        right: &Arc<Worker>,
    ) -> Ordering {
        let left_local = self
            .local_active_by_worker_id
            .get(left.id.0.as_str())
            .copied()
            .unwrap_or(usize::MAX);
        let right_local = self
            .local_active_by_worker_id
            .get(right.id.0.as_str())
            .copied()
            .unwrap_or(usize::MAX);
        match (
            self.comparable_get(&left.id),
            self.comparable_get(&right.id),
        ) {
            (Some(left_load), Some(right_load)) => {
                compare_prefill_aggregate(left_load, right_load, self.compare_prefill_queue_ms)
                    .then_with(|| left_local.cmp(&right_local))
            }
            _ => left_local.cmp(&right_local),
        }
    }

    fn compare_decode_pressure(&self, left: &Arc<Worker>, right: &Arc<Worker>) -> Ordering {
        let left_local = self
            .local_active_by_worker_id
            .get(left.id.0.as_str())
            .copied()
            .unwrap_or(usize::MAX);
        let right_local = self
            .local_active_by_worker_id
            .get(right.id.0.as_str())
            .copied()
            .unwrap_or(usize::MAX);
        match (
            self.comparable_get(&left.id),
            self.comparable_get(&right.id),
        ) {
            (Some(left_load), Some(right_load)) => compare_decode_aggregate(
                left_load,
                right_load,
                self.compare_decode_queues,
                self.compare_decode_step_ms,
                self.compare_decode_active_tokens,
            )
            .then_with(|| left_local.cmp(&right_local)),
            _ => left_local.cmp(&right_local),
        }
    }
}

/// 两个候选都有 fresh snapshot 时才允许压力逃逸。
fn pressure_guard_prefers_backup(
    primary: &Arc<Worker>,
    backup: &Arc<Worker>,
    hints: &GuardHints,
    snapshot: &SchedulingSnapshot,
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
    if let (Some(absolute_threshold_ms), Some(primary_queue_ms), Some(backup_queue_ms)) = (
        hints.pressure_abs_threshold_ms,
        primary_load.estimated_prefill_queue_ms,
        backup_load.estimated_prefill_queue_ms,
    ) {
        return primary_queue_ms - backup_queue_ms > absolute_threshold_ms
            && primary_queue_ms > backup_queue_ms * hints.pressure_rel_threshold;
    }
    let primary_pressure = primary_load.num_waiting_uncached_tokens;
    let backup_pressure = backup_load.num_waiting_uncached_tokens;
    primary_pressure.saturating_sub(backup_pressure) > hints.pressure_abs_threshold_tokens
        && (primary_pressure as f64) > (backup_pressure as f64) * hints.pressure_rel_threshold
}

fn range_fallback(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &SchedulingSnapshot,
) -> Option<(Arc<Worker>, DecisionReason)> {
    let candidates = proposal
        .eligible_workers
        .as_deref()
        .unwrap_or(range.workers);
    let admission_loads = FreshLoadLookup::new(Some(snapshot), candidates.iter());
    let admitted: Vec<Arc<Worker>> = candidates
        .iter()
        .filter(|worker| contains_worker(range, worker))
        .filter(|worker| {
            is_prefill_admitted_with_load(
                range,
                request_input_tokens,
                admission_loads.get(&worker.id),
            )
        })
        .cloned()
        .collect();
    let comparison_loads = FreshLoadLookup::new(Some(snapshot), admitted.iter());
    admitted
        .into_iter()
        .min_by(|left, right| comparison_loads.compare_prefill_pressure(left, right))
        .map(|worker| (worker, DecisionReason::RangeFallback))
}

fn decode_domain_fallback(
    domain: &CandidateDomain,
    request_kv_tokens: u64,
    snapshot: &SchedulingSnapshot,
) -> Option<(Arc<Worker>, DecisionReason)> {
    let admission_loads = FreshLoadLookup::new(Some(snapshot), domain.workers.iter());
    let admitted: Vec<Arc<Worker>> = domain
        .workers
        .iter()
        .filter(|worker| {
            is_decode_admitted_with_load(request_kv_tokens, admission_loads.get(&worker.id))
        })
        .cloned()
        .collect();
    let comparison_loads = FreshLoadLookup::new(Some(snapshot), admitted.iter());
    admitted
        .into_iter()
        .min_by(|left, right| comparison_loads.compare_decode_pressure(left, right))
        .map(|worker| (worker, DecisionReason::RangeFallback))
}

/// 比较 Prefill 压力；任一 fresh report 缺失时退化为本地 active-load。
pub(crate) fn compare_prefill_pressure(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    snapshot: Option<&SchedulingSnapshot>,
) -> Ordering {
    match snapshot.and_then(|snapshot| {
        Some((
            snapshot.fresh_load(&left.id)?,
            snapshot.fresh_load(&right.id)?,
        ))
    }) {
        Some((left_load, right_load)) => compare_prefill_aggregate(
            left_load,
            right_load,
            left_load.estimated_prefill_queue_ms.is_some()
                && right_load.estimated_prefill_queue_ms.is_some(),
        )
        .then_with(|| left.active_load().cmp(&right.active_load())),
        _ => left.active_load().cmp(&right.active_load()),
    }
}

fn fallback_prefill_pressure_key(load: &AggregateLoad) -> (u64, u64, u64) {
    (
        load.num_waiting_uncached_tokens,
        load.num_waiting_reqs,
        load.num_running_reqs,
    )
}

fn compare_prefill_aggregate(
    left: &AggregateLoad,
    right: &AggregateLoad,
    compare_queue_ms: bool,
) -> Ordering {
    if compare_queue_ms {
        return left
            .estimated_prefill_queue_ms
            .expect("queue estimate availability was checked")
            .total_cmp(
                &right
                    .estimated_prefill_queue_ms
                    .expect("queue estimate availability was checked"),
            )
            .then_with(|| {
                fallback_prefill_pressure_key(left).cmp(&fallback_prefill_pressure_key(right))
            });
    }
    fallback_prefill_pressure_key(left).cmp(&fallback_prefill_pressure_key(right))
}

/// 比较 Decode 队列与执行压力，最后使用本地 active-load。
pub(crate) fn compare_decode_pressure(
    left: &Arc<Worker>,
    right: &Arc<Worker>,
    snapshot: Option<&SchedulingSnapshot>,
) -> Ordering {
    match snapshot.and_then(|snapshot| {
        Some((
            snapshot.fresh_load(&left.id)?,
            snapshot.fresh_load(&right.id)?,
        ))
    }) {
        Some((left_load, right_load)) => compare_decode_aggregate(
            left_load,
            right_load,
            left_load.decode_prealloc_queue_reqs.is_some()
                && right_load.decode_prealloc_queue_reqs.is_some()
                && left_load.decode_transfer_queue_reqs.is_some()
                && right_load.decode_transfer_queue_reqs.is_some()
                && left_load.decode_retracted_queue_reqs.is_some()
                && right_load.decode_retracted_queue_reqs.is_some(),
            left_load.mean_decode_step_ms.is_some() && right_load.mean_decode_step_ms.is_some(),
            left_load.num_active_tokens.is_some() && right_load.num_active_tokens.is_some(),
        )
        .then_with(|| left.active_load().cmp(&right.active_load())),
        None => left.active_load().cmp(&right.active_load()),
    }
}

fn compare_decode_aggregate(
    left: &AggregateLoad,
    right: &AggregateLoad,
    compare_queues: bool,
    compare_step_ms: bool,
    compare_active_tokens: bool,
) -> Ordering {
    let mut ordering = Ordering::Equal;
    if compare_queues {
        ordering = left
            .decode_retracted_queue_reqs
            .expect("decode queue availability was checked")
            .cmp(
                &right
                    .decode_retracted_queue_reqs
                    .expect("decode queue availability was checked"),
            )
            .then_with(|| {
                let left_incoming = left
                    .decode_prealloc_queue_reqs
                    .expect("decode queue availability was checked")
                    .saturating_add(
                        left.decode_transfer_queue_reqs
                            .expect("decode queue availability was checked"),
                    );
                let right_incoming = right
                    .decode_prealloc_queue_reqs
                    .expect("decode queue availability was checked")
                    .saturating_add(
                        right
                            .decode_transfer_queue_reqs
                            .expect("decode queue availability was checked"),
                    );
                left_incoming.cmp(&right_incoming)
            });
    }
    let left_running =
        u128::from(left.num_running_reqs).saturating_mul(u128::from(right.max_running_requests));
    let right_running =
        u128::from(right.num_running_reqs).saturating_mul(u128::from(left.max_running_requests));
    ordering = ordering.then_with(|| left_running.cmp(&right_running));
    if compare_step_ms {
        ordering = ordering.then_with(|| {
            left.mean_decode_step_ms
                .expect("decode step availability was checked")
                .total_cmp(
                    &right
                        .mean_decode_step_ms
                        .expect("decode step availability was checked"),
                )
        });
    }
    if compare_active_tokens {
        let left_active = u128::from(
            left.num_active_tokens
                .expect("active-token availability was checked"),
        )
        .saturating_mul(u128::from(right.max_total_num_tokens));
        let right_active = u128::from(
            right
                .num_active_tokens
                .expect("active-token availability was checked"),
        )
        .saturating_mul(u128::from(left.max_total_num_tokens));
        ordering = ordering.then_with(|| left_active.cmp(&right_active));
    } else {
        let left_kv = u128::from(left.num_total_tokens)
            .saturating_mul(u128::from(right.max_total_num_tokens));
        let right_kv = u128::from(right.num_total_tokens)
            .saturating_mul(u128::from(left.max_total_num_tokens));
        ordering = ordering.then_with(|| left_kv.cmp(&right_kv));
    }
    ordering
        .then_with(|| left.num_running_reqs.cmp(&right.num_running_reqs))
        .then_with(|| {
            if compare_active_tokens {
                left.num_active_tokens.cmp(&right.num_active_tokens)
            } else {
                left.num_total_tokens.cmp(&right.num_total_tokens)
            }
        })
}
