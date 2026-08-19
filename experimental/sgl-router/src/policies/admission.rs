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
    if let Some(absolute_threshold_ms) = absolute_threshold_ms.filter(|_| {
        loads.compare_prefill_queue_ms
            && candidate_load.estimated_prefill_queue_ms.is_some()
            && other_load.estimated_prefill_queue_ms.is_some()
    }) {
        let candidate_pressure = candidate_load
            .estimated_prefill_queue_ms
            .expect("queue estimate availability was checked");
        let other_pressure = other_load
            .estimated_prefill_queue_ms
            .expect("queue estimate availability was checked");
        return candidate_pressure - other_pressure > absolute_threshold_ms
            && candidate_pressure > other_pressure * relative_threshold;
    }
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
    compare_prefill_queue_ms: bool,
    compare_decode_queues: bool,
    compare_decode_step_ms: bool,
    compare_decode_active_tokens: bool,
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
        let compare_prefill_queue_ms = compare_aggregate
            && by_worker_id
                .values()
                .all(|load| load.estimated_prefill_queue_ms.is_some());
        let compare_decode_queues = compare_aggregate
            && by_worker_id.values().all(|load| {
                load.decode_retracted_queue_reqs.is_some()
                    && load.decode_prealloc_queue_reqs.is_some()
                    && load.decode_transfer_queue_reqs.is_some()
            });
        let compare_decode_step_ms = compare_aggregate
            && by_worker_id
                .values()
                .all(|load| load.mean_decode_step_ms.is_some());
        let compare_decode_active_tokens = compare_aggregate
            && by_worker_id
                .values()
                .all(|load| load.num_active_tokens.is_some());
        Self {
            by_worker_id,
            local_active_by_worker_id,
            compare_aggregate,
            compare_prefill_queue_ms,
            compare_decode_queues,
            compare_decode_step_ms,
            compare_decode_active_tokens,
        }
    }

    pub(crate) fn get(&self, worker_id: &crate::discovery::WorkerId) -> Option<&'a AggregateLoad> {
        self.by_worker_id.get(worker_id.0.as_str()).copied()
    }

    fn comparable_get(&self, worker_id: &crate::discovery::WorkerId) -> Option<&'a AggregateLoad> {
        if self.compare_aggregate {
            self.get(worker_id)
        } else {
            None
        }
    }

    /// Both inputs are a pure function of the worker id, so resolving once per
    /// candidate is equivalent to resolving inside every comparison.
    fn pressure_key(&self, worker: &Arc<Worker>) -> PressureKey<'a> {
        PressureKey {
            load: self.comparable_get(&worker.id),
            local_active: self
                .local_active_by_worker_id
                .get(worker.id.0.as_str())
                .copied()
                .unwrap_or(usize::MAX),
        }
    }

    fn compare_prefill_keys(&self, left: &PressureKey<'a>, right: &PressureKey<'a>) -> Ordering {
        match (left.load, right.load) {
            (Some(left_load), Some(right_load)) => {
                compare_prefill_aggregate(left_load, right_load, self.compare_prefill_queue_ms)
                    .then_with(|| left.local_active.cmp(&right.local_active))
            }
            _ => left.local_active.cmp(&right.local_active),
        }
    }

    fn compare_decode_keys(&self, left: &PressureKey<'a>, right: &PressureKey<'a>) -> Ordering {
        match (left.load, right.load) {
            (Some(left_load), Some(right_load)) => compare_decode_aggregate(
                left_load,
                right_load,
                self.compare_decode_queues,
                self.compare_decode_step_ms,
                self.compare_decode_active_tokens,
            )
            .then_with(|| left.local_active.cmp(&right.local_active)),
            _ => left.local_active.cmp(&right.local_active),
        }
    }

    pub(crate) fn compare_prefill_pressure(
        &self,
        left: &Arc<Worker>,
        right: &Arc<Worker>,
    ) -> Ordering {
        self.compare_prefill_keys(&self.pressure_key(left), &self.pressure_key(right))
    }

    /// Cheapest candidate under `compare`, resolving each candidate's inputs
    /// exactly once. Ties keep the earlier candidate, matching `min_by`.
    fn min_by_pressure_key(
        &self,
        candidates: Vec<Arc<Worker>>,
        compare: impl Fn(&Self, &PressureKey<'a>, &PressureKey<'a>) -> Ordering,
    ) -> Option<Arc<Worker>> {
        let mut candidates = candidates.into_iter();
        let first = candidates.next()?;
        let second = match candidates.next() {
            Some(candidate) => candidate,
            None => return Some(first),
        };
        let mut best_key = self.pressure_key(&first);
        let mut best = first;
        let second_key = self.pressure_key(&second);
        if compare(self, &second_key, &best_key).is_lt() {
            best_key = second_key;
            best = second;
        }
        for candidate in candidates {
            let key = self.pressure_key(&candidate);
            if compare(self, &key, &best_key).is_lt() {
                best_key = key;
                best = candidate;
            }
        }
        Some(best)
    }
}

struct PressureKey<'a> {
    load: Option<&'a AggregateLoad>,
    local_active: usize,
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
    if let Some(absolute_threshold_ms) = hints.pressure_abs_threshold_ms.filter(|_| {
        primary_load.estimated_prefill_queue_ms.is_some()
            && backup_load.estimated_prefill_queue_ms.is_some()
    }) {
        let primary_pressure = primary_load
            .estimated_prefill_queue_ms
            .expect("queue estimate availability was checked");
        let backup_pressure = backup_load
            .estimated_prefill_queue_ms
            .expect("queue estimate availability was checked");
        return primary_pressure - backup_pressure > absolute_threshold_ms
            && primary_pressure > backup_pressure * hints.pressure_rel_threshold;
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
    comparison_loads
        .min_by_pressure_key(admitted, FreshLoadLookup::compare_prefill_keys)
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
    comparison_loads
        .min_by_pressure_key(admitted, FreshLoadLookup::compare_decode_keys)
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
        Some((left_load, right_load)) => compare_prefill_aggregate(
            &left_load,
            &right_load,
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
        Some((left_load, right_load)) => compare_decode_aggregate(
            &left_load,
            &right_load,
            left_load.decode_retracted_queue_reqs.is_some()
                && right_load.decode_retracted_queue_reqs.is_some()
                && left_load.decode_prealloc_queue_reqs.is_some()
                && right_load.decode_prealloc_queue_reqs.is_some()
                && left_load.decode_transfer_queue_reqs.is_some()
                && right_load.decode_transfer_queue_reqs.is_some(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::load_monitor::WorkerSnapshot;
    use crate::policies::SelectionProposal;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".into())],
            bootstrap_port: None,
        }))
    }

    fn load(waiting_tokens: u64, running_requests: u64) -> AggregateLoad {
        AggregateLoad {
            num_running_reqs: running_requests,
            num_waiting_uncached_tokens: waiting_tokens,
            num_total_tokens: 1_024 + running_requests,
            max_total_num_tokens: 8_192,
            max_running_requests: 64,
            ..Default::default()
        }
    }

    fn snapshot(entries: &[(&Arc<Worker>, AggregateLoad)]) -> LoadMonitorSnapshot {
        LoadMonitorSnapshot {
            enabled: true,
            version: 1,
            captured_at: None,
            workers: entries
                .iter()
                .map(|(worker, aggregate)| WorkerSnapshot {
                    worker_id: worker.id.0.clone(),
                    url: worker.url.clone(),
                    mode: worker.mode(),
                    model_ids: worker
                        .model_ids
                        .iter()
                        .map(|model| model.0.clone())
                        .collect(),
                    freshness: Freshness::Fresh,
                    source_instance_id: None,
                    sequence_id: None,
                    report_time_unix_ms: None,
                    last_error: None,
                    received_at: None,
                    expires_at: None,
                    aggregate: Some(aggregate.clone()),
                    ranks: Vec::new(),
                })
                .collect(),
        }
    }

    #[test]
    fn pressure_key_fold_keeps_prefill_decode_and_tie_order() {
        let first = worker("first");
        let best = worker("best");
        let last = worker("last");
        let loads = snapshot(&[
            (&first, load(20, 3)),
            (&best, load(10, 1)),
            (&last, load(30, 2)),
        ]);
        let candidates = vec![Arc::clone(&first), Arc::clone(&best), Arc::clone(&last)];
        let lookup = FreshLoadLookup::new(Some(&loads), candidates.iter());

        assert_eq!(
            lookup
                .min_by_pressure_key(candidates.clone(), FreshLoadLookup::compare_prefill_keys)
                .expect("non-empty candidates")
                .id,
            best.id
        );
        assert_eq!(
            lookup
                .min_by_pressure_key(candidates, FreshLoadLookup::compare_decode_keys)
                .expect("non-empty candidates")
                .id,
            best.id
        );

        let tied = vec![Arc::clone(&first), Arc::clone(&last)];
        let tied_snapshot = snapshot(&[(&first, load(1, 1)), (&last, load(1, 1))]);
        let tied_lookup = FreshLoadLookup::new(Some(&tied_snapshot), tied.iter());
        assert_eq!(
            tied_lookup
                .min_by_pressure_key(tied, FreshLoadLookup::compare_prefill_keys)
                .expect("non-empty candidates")
                .id,
            first.id
        );
    }

    #[test]
    fn prefill_fallback_filters_capacity_before_ranking() {
        let busy = worker("busy");
        let quick = worker("quick");
        let roomy = worker("roomy");
        let domain = CandidateDomain::bucket_prefill(
            "bucket",
            vec![Arc::clone(&busy), Arc::clone(&quick), Arc::clone(&roomy)],
            Some(1_000),
        );
        let range = domain.prefill_range().expect("prefill domain");
        let snapshot = snapshot(&[
            (&busy, load(500, 1)),
            (&quick, load(800, 1)),
            (&roomy, load(10, 1)),
        ]);

        let decision = resolve_prefill(&range, &SelectionProposal::primary(busy), 900, &snapshot)
            .expect("roomy remains admitted");
        assert_eq!(decision.reason, DecisionReason::RangeFallback);
        assert_eq!(decision.selected.id, roomy.id);
    }

    #[test]
    fn prefill_pressure_prefers_estimated_queue_when_available() {
        let token_light_but_slow = worker("token-light-but-slow");
        let token_heavy_but_fast = worker("token-heavy-but-fast");
        let mut slow = load(1, 1);
        slow.estimated_prefill_queue_ms = Some(100.0);
        let mut fast = load(10_000, 1);
        fast.estimated_prefill_queue_ms = Some(10.0);
        let snapshot = snapshot(&[(&token_light_but_slow, slow), (&token_heavy_but_fast, fast)]);

        assert!(compare_prefill_pressure(
            &token_light_but_slow,
            &token_heavy_but_fast,
            Some(&snapshot),
        )
        .is_gt());
    }

    #[test]
    fn prefill_pressure_uses_token_fallback_when_queue_estimate_is_incomplete() {
        let token_light = worker("token-light");
        let token_heavy = worker("token-heavy");
        let mut light = load(1, 1);
        light.estimated_prefill_queue_ms = Some(100.0);
        let heavy = load(10_000, 1);
        let snapshot = snapshot(&[(&token_light, light), (&token_heavy, heavy)]);

        assert!(compare_prefill_pressure(&token_light, &token_heavy, Some(&snapshot)).is_lt());
    }

    #[test]
    fn prefill_candidate_set_uses_token_fallback_when_any_queue_estimate_is_missing() {
        let token_light = worker("token-light");
        let token_heavy = worker("token-heavy");
        let incomplete = worker("incomplete");
        let mut light = load(1, 1);
        light.estimated_prefill_queue_ms = Some(100.0);
        let mut heavy = load(10_000, 1);
        heavy.estimated_prefill_queue_ms = Some(10.0);
        let incomplete_load = load(20_000, 1);
        let snapshot = snapshot(&[
            (&token_light, light),
            (&token_heavy, heavy),
            (&incomplete, incomplete_load),
        ]);
        let candidates = vec![
            Arc::clone(&token_light),
            Arc::clone(&token_heavy),
            Arc::clone(&incomplete),
        ];
        let lookup = FreshLoadLookup::new(Some(&snapshot), candidates.iter());

        assert_eq!(
            lookup
                .min_by_pressure_key(candidates, FreshLoadLookup::compare_prefill_keys)
                .expect("non-empty candidates")
                .id,
            token_light.id,
        );
    }

    #[test]
    fn prefill_pressure_guard_uses_queue_threshold_when_configured() {
        let primary = worker("primary");
        let backup = worker("backup");
        let mut primary_load = load(1, 1);
        primary_load.estimated_prefill_queue_ms = Some(100.0);
        let mut backup_load = load(10_000, 1);
        backup_load.estimated_prefill_queue_ms = Some(10.0);
        let snapshot = snapshot(&[(&primary, primary_load), (&backup, backup_load)]);
        let hints = GuardHints {
            enable_pressure_guard: true,
            pressure_abs_threshold_tokens: u64::MAX,
            pressure_abs_threshold_ms: Some(20.0),
            pressure_rel_threshold: 1.5,
        };

        assert!(pressure_guard_prefers_backup(
            &primary, &backup, &hints, &snapshot
        ));
    }

    #[test]
    fn cache_guard_uses_queue_threshold_when_configured() {
        let cached_but_slow = worker("cached-but-slow");
        let less_cached_but_fast = worker("less-cached-but-fast");
        let mut slow = load(1, 1);
        slow.estimated_prefill_queue_ms = Some(100.0);
        let mut fast = load(10_000, 1);
        fast.estimated_prefill_queue_ms = Some(10.0);
        let snapshot = snapshot(&[(&cached_but_slow, slow), (&less_cached_but_fast, fast)]);
        let proposal = CacheCandidateProposal {
            candidates: vec![
                CacheCandidate {
                    worker: Arc::clone(&cached_but_slow),
                    matched_prefix_tokens: 90,
                    uncached_tokens: 10,
                    candidate_range_id: "global".into(),
                    max_pending_prefill_tokens: None,
                },
                CacheCandidate {
                    worker: Arc::clone(&less_cached_but_fast),
                    matched_prefix_tokens: 80,
                    uncached_tokens: 20,
                    candidate_range_id: "global".into(),
                    max_pending_prefill_tokens: None,
                },
            ],
            cache_switch_margin_tokens: 32,
            enable_pressure_guard: true,
            pressure_abs_threshold_tokens: u64::MAX,
            pressure_abs_threshold_ms: Some(20.0),
            pressure_rel_threshold: 1.5,
        };

        let decision = resolve_cache_candidates(&proposal, 100, &snapshot)
            .expect("both candidates are admitted");
        assert_eq!(decision.selected.id, less_cached_but_fast.id);
    }

    #[test]
    fn decode_pressure_prefers_retraction_then_incoming_queue() {
        let retracted = worker("retracted");
        let clear = worker("clear");
        let mut retracted_load = load(10, 1);
        retracted_load.decode_retracted_queue_reqs = Some(1);
        retracted_load.decode_prealloc_queue_reqs = Some(0);
        retracted_load.decode_transfer_queue_reqs = Some(0);
        let mut clear_load = load(10, 1);
        clear_load.decode_retracted_queue_reqs = Some(0);
        clear_load.decode_prealloc_queue_reqs = Some(0);
        clear_load.decode_transfer_queue_reqs = Some(0);
        let retraction_snapshot = snapshot(&[(&retracted, retracted_load), (&clear, clear_load)]);

        assert!(compare_decode_pressure(&retracted, &clear, Some(&retraction_snapshot),).is_gt());

        let incoming = worker("incoming");
        let idle = worker("idle");
        let mut incoming_load = load(10, 1);
        incoming_load.decode_retracted_queue_reqs = Some(0);
        incoming_load.decode_prealloc_queue_reqs = Some(2);
        incoming_load.decode_transfer_queue_reqs = Some(3);
        let mut idle_load = load(10, 1);
        idle_load.decode_retracted_queue_reqs = Some(0);
        idle_load.decode_prealloc_queue_reqs = Some(0);
        idle_load.decode_transfer_queue_reqs = Some(0);
        let incoming_snapshot = snapshot(&[(&incoming, incoming_load), (&idle, idle_load)]);

        assert!(compare_decode_pressure(&incoming, &idle, Some(&incoming_snapshot)).is_gt());
    }

    #[test]
    fn decode_pressure_uses_step_time_then_active_tokens_before_total_tokens() {
        let slow_step = worker("slow-step");
        let fast_step = worker("fast-step");
        let mut slow_load = load(10, 1);
        slow_load.decode_retracted_queue_reqs = Some(0);
        slow_load.decode_prealloc_queue_reqs = Some(0);
        slow_load.decode_transfer_queue_reqs = Some(0);
        slow_load.mean_decode_step_ms = Some(20.0);
        slow_load.num_active_tokens = Some(10);
        let mut fast_load = slow_load.clone();
        fast_load.mean_decode_step_ms = Some(5.0);
        let step_snapshot = snapshot(&[(&slow_step, slow_load), (&fast_step, fast_load)]);

        assert!(compare_decode_pressure(&slow_step, &fast_step, Some(&step_snapshot)).is_gt());

        let active = worker("active");
        let quiet = worker("quiet");
        let mut active_load = load(10, 1);
        active_load.decode_retracted_queue_reqs = Some(0);
        active_load.decode_prealloc_queue_reqs = Some(0);
        active_load.decode_transfer_queue_reqs = Some(0);
        active_load.mean_decode_step_ms = Some(5.0);
        active_load.num_active_tokens = Some(100);
        let mut quiet_load = active_load.clone();
        quiet_load.num_active_tokens = Some(10);
        let active_snapshot = snapshot(&[(&active, active_load), (&quiet, quiet_load)]);

        assert!(compare_decode_pressure(&active, &quiet, Some(&active_snapshot)).is_gt());
    }
}
