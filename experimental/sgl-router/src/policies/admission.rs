// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::load_monitor::{AggregateLoad, Freshness, LoadMonitorSnapshot};
use crate::policies::{GuardHints, SelectionProposal};
use crate::workers::Worker;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    Primary,
    BackupPrimaryAdmission,
    BackupPressureGuard,
    RangeFallback,
}

#[derive(Clone)]
pub struct FinalDecision {
    pub selected: Arc<Worker>,
    pub primary: Arc<Worker>,
    pub backup: Option<Arc<Worker>>,
    pub reason: DecisionReason,
    pub candidate_range_id: String,
    pub load_snapshot_version: u64,
}

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

fn contains_worker(range: &CandidateRange<'_>, candidate: &Arc<Worker>) -> bool {
    range.workers.iter().any(|worker| worker.id == candidate.id)
}

fn is_proposal_worker_eligible(proposal: &SelectionProposal, candidate: &Arc<Worker>) -> bool {
    proposal
        .eligible_workers
        .as_ref()
        .is_none_or(|workers| workers.iter().any(|worker| worker.id == candidate.id))
}

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
}

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
