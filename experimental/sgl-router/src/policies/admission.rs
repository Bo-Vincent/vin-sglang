// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefill / Decode 共享 Admission 与 Guard。
//!
//! Pair proposal 先做容量检查再执行 Guard；Cache-Aware 候选走有界比较。

use crate::load_monitor::{AggregateLoad, Freshness, LoadMonitorSnapshot};
use crate::policies::{CacheCandidate, CacheCandidateProposal, GuardHints, SelectionProposal};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

/// Prefill policy 的候选域及其容量上限。
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
}

impl CandidateDomain {
    pub fn global_prefill(workers: &[Arc<Worker>]) -> Self {
        Self {
            id: "global".to_string(),
            stage: RoutingStage::Prefill,
            workers: workers.to_vec(),
            max_pending_prefill_tokens: None,
        }
    }

    pub fn global_decode(workers: &[Arc<Worker>]) -> Self {
        Self {
            id: "global".to_string(),
            stage: RoutingStage::Decode,
            workers: workers.to_vec(),
            max_pending_prefill_tokens: None,
        }
    }

    pub fn bucket_prefill(
        id: impl Into<String>,
        workers: Vec<Arc<Worker>>,
        max_pending_prefill_tokens: Option<u64>,
    ) -> Self {
        Self {
            id: id.into(),
            stage: RoutingStage::Prefill,
            workers,
            max_pending_prefill_tokens,
        }
    }

    pub fn bucket_decode(id: impl Into<String>, workers: Vec<Arc<Worker>>) -> Self {
        Self {
            id: id.into(),
            stage: RoutingStage::Decode,
            workers,
            max_pending_prefill_tokens: None,
        }
    }

    /// 将统一 Domain 投影为 Prefill Admission 使用的借用视图。
    pub fn prefill_range(&self) -> Option<CandidateRange<'_>> {
        (self.stage == RoutingStage::Prefill).then(|| CandidateRange {
            id: self.id.as_str(),
            workers: &self.workers,
            max_pending_prefill_tokens: self.max_pending_prefill_tokens,
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
    snapshot: &LoadMonitorSnapshot,
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
    snapshot: &LoadMonitorSnapshot,
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
    range.workers.iter().any(|worker| worker.id == candidate.id)
}

fn contains_domain_worker(domain: &CandidateDomain, candidate: &Arc<Worker>) -> bool {
    domain
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
    snapshot: &LoadMonitorSnapshot,
) -> bool {
    let load = snapshot.fresh_load(&worker.id);
    is_prefill_admitted_with_load(range, request_input_tokens, load.as_ref())
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
    snapshot: &LoadMonitorSnapshot,
) -> bool {
    let load = snapshot.fresh_load(&worker.id);
    is_decode_admitted_with_load(request_kv_tokens, load.as_ref())
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
    if materially_more_pressured(
        &left.worker,
        &right.worker,
        proposal.pressure_abs_threshold_tokens,
        proposal.pressure_rel_threshold,
        loads,
    ) {
        return Ordering::Greater;
    }
    if materially_more_pressured(
        &right.worker,
        &left.worker,
        proposal.pressure_abs_threshold_tokens,
        proposal.pressure_rel_threshold,
        loads,
    ) {
        return Ordering::Less;
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
    relative_threshold: f64,
    loads: &FreshLoadLookup<'_>,
) -> bool {
    let (Some(candidate_load), Some(other_load)) = (
        loads.comparable_get(&candidate.id),
        loads.comparable_get(&other.id),
    ) else {
        return false;
    };
    let candidate_pressure = candidate_load.num_waiting_uncached_tokens;
    let other_pressure = other_load.num_waiting_uncached_tokens;
    candidate_pressure.saturating_sub(other_pressure) > absolute_threshold
        && candidate_pressure as f64 > other_pressure as f64 * relative_threshold
}

/// Request-local O(1) lookup for fresh LoadMonitor and captured local load.
pub(crate) struct FreshLoadLookup<'a> {
    by_worker_id: HashMap<&'a str, &'a AggregateLoad>,
    local_active_by_worker_id: HashMap<String, usize>,
    compare_aggregate: bool,
}

impl<'a> FreshLoadLookup<'a> {
    pub(crate) fn new<'w>(
        snapshot: Option<&'a LoadMonitorSnapshot>,
        workers: impl IntoIterator<Item = &'w Arc<Worker>>,
    ) -> Self {
        // Snapshot atomics once so sort comparisons remain stable.
        let local_active_by_worker_id: HashMap<String, usize> = workers
            .into_iter()
            .map(|worker| (worker.id.0.clone(), worker.active_load()))
            .collect();
        let by_worker_id: HashMap<&'a str, &'a AggregateLoad> = snapshot
            .into_iter()
            .flat_map(|snapshot| snapshot.workers.iter())
            .filter(|worker| {
                worker.freshness == Freshness::Fresh
                    && local_active_by_worker_id.contains_key(worker.worker_id.as_str())
            })
            .filter_map(|worker| {
                worker
                    .aggregate
                    .as_ref()
                    .map(|aggregate| (worker.worker_id.as_str(), aggregate))
            })
            .collect();
        // Mixed fresh/stale sets use local load only to preserve transitivity.
        let compare_aggregate = !local_active_by_worker_id.is_empty()
            && by_worker_id.len() == local_active_by_worker_id.len();
        Self {
            by_worker_id,
            local_active_by_worker_id,
            compare_aggregate,
        }
    }

    pub(crate) fn get(&self, worker_id: &crate::discovery::WorkerId) -> Option<&AggregateLoad> {
        self.by_worker_id.get(worker_id.0.as_str()).copied()
    }

    fn comparable_get(&self, worker_id: &crate::discovery::WorkerId) -> Option<&AggregateLoad> {
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
            (Some(left_load), Some(right_load)) => load_pressure_key(left_load)
                .cmp(&load_pressure_key(right_load))
                .then_with(|| left_local.cmp(&right_local)),
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
            (Some(left_load), Some(right_load)) => compare_decode_aggregate(left_load, right_load)
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
        && (primary_pressure as f64) > (backup_pressure as f64) * hints.pressure_rel_threshold
}

fn range_fallback(
    range: &CandidateRange<'_>,
    proposal: &SelectionProposal,
    request_input_tokens: u64,
    snapshot: &LoadMonitorSnapshot,
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
    snapshot: &LoadMonitorSnapshot,
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
            .then_with(|| left.active_load().cmp(&right.active_load())),
        _ => left.active_load().cmp(&right.active_load()),
    }
}

fn load_pressure_key(load: &AggregateLoad) -> (u64, u64, u64) {
    (
        load.num_waiting_uncached_tokens,
        load.num_waiting_reqs,
        load.num_running_reqs,
    )
}

/// 比较 Decode running/KV 占用比例，最后使用本地 active-load。
pub(crate) fn compare_decode_pressure(
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
        Some((left_load, right_load)) => compare_decode_aggregate(&left_load, &right_load)
            .then_with(|| left.active_load().cmp(&right.active_load())),
        None => left.active_load().cmp(&right.active_load()),
    }
}

fn compare_decode_aggregate(left: &AggregateLoad, right: &AggregateLoad) -> Ordering {
    let left_running =
        u128::from(left.num_running_reqs).saturating_mul(u128::from(right.max_running_requests));
    let right_running =
        u128::from(right.num_running_reqs).saturating_mul(u128::from(left.max_running_requests));
    let left_kv =
        u128::from(left.num_total_tokens).saturating_mul(u128::from(right.max_total_num_tokens));
    let right_kv =
        u128::from(right.num_total_tokens).saturating_mul(u128::from(left.max_total_num_tokens));
    left_running
        .cmp(&right_running)
        .then_with(|| left_kv.cmp(&right_kv))
        .then_with(|| left.num_running_reqs.cmp(&right.num_running_reqs))
        .then_with(|| left.num_total_tokens.cmp(&right.num_total_tokens))
}
