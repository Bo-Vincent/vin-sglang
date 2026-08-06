// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Decode policy 的最小可观察契约。
//!
//! 这些测试只覆盖 Router 已有的 running/KV/local active-load 输入；不把尚未
//! 采集的 transfer、decode queue 或 retraction 指标伪造成可用数据。

use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::load_monitor::{AggregateLoad, Freshness, LoadMonitorSnapshot, WorkerSnapshot};
use sgl_router::policies::admission::{resolve_decode, CandidateDomain, DecisionReason};
use sgl_router::policies::decode::{
    DecodePolicy, DecodePowerOfTwoPolicy, DecodeSelectionContext, LegacyHostAffinityDecodePolicy,
};
use sgl_router::policies::SelectionProposal;
use sgl_router::workers::Worker;
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn worker(id: &str) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}:30000"),
        mode: WorkerMode::Decode,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }))
}

fn snapshot(entries: &[(&Arc<Worker>, AggregateLoad)]) -> LoadMonitorSnapshot {
    LoadMonitorSnapshot {
        enabled: true,
        version: 7,
        captured_at: Some("2026-08-04T00:00:00Z".into()),
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
fn decode_p2_proposes_a_distinct_lower_pressure_primary_and_backup() {
    let busy = worker("busy");
    let idle = worker("idle");
    busy.active_requests.store(8, Ordering::Relaxed);
    idle.active_requests.store(1, Ordering::Relaxed);
    let domain = CandidateDomain::global_decode(&[Arc::clone(&busy), Arc::clone(&idle)]);
    let ctx = DecodeSelectionContext::new();

    let proposal = DecodePowerOfTwoPolicy::new()
        .propose(&domain, &ctx)
        .expect("two decode candidates must produce a proposal");

    assert_eq!(proposal.primary.id, idle.id);
    assert_eq!(
        proposal.backup.expect("P2 keeps the other sample").id,
        busy.id
    );
}

#[test]
fn legacy_host_affinity_remains_an_explicit_single_primary_compatibility_policy() {
    let same_host = worker("host-a");
    let other_host = worker("host-b");
    let domain = CandidateDomain::global_decode(&[Arc::clone(&same_host), other_host]);
    let ctx = DecodeSelectionContext::new().with_prefill_url("http://host-a:9999");

    let proposal = LegacyHostAffinityDecodePolicy
        .propose(&domain, &ctx)
        .expect("legacy policy selects one compatible decode worker");

    assert_eq!(proposal.primary.id, same_host.id);
    assert!(
        proposal.backup.is_none(),
        "legacy semantics do not invent a backup"
    );
}

#[test]
fn decode_admission_uses_backup_before_scanning_domain() {
    let primary = worker("primary");
    let backup = worker("backup");
    let fallback = worker("fallback");
    let domain = CandidateDomain::global_decode(&[
        Arc::clone(&primary),
        Arc::clone(&backup),
        Arc::clone(&fallback),
    ]);
    let loads = snapshot(&[
        (
            &primary,
            AggregateLoad {
                num_running_reqs: 4,
                max_running_requests: 4,
                max_total_num_tokens: 1_000,
                ..Default::default()
            },
        ),
        (
            &backup,
            AggregateLoad {
                max_running_requests: 4,
                max_total_num_tokens: 1_000,
                ..Default::default()
            },
        ),
        (
            &fallback,
            AggregateLoad {
                max_running_requests: 4,
                max_total_num_tokens: 1_000,
                ..Default::default()
            },
        ),
    ]);
    let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

    let decision =
        resolve_decode(&domain, &proposal, 64, &loads).expect("admitted backup must be selected");

    assert_eq!(decision.selected.id, backup.id);
    assert_eq!(decision.reason, DecisionReason::BackupPrimaryAdmission);
}

#[test]
fn decode_guard_can_escape_a_primary_to_lower_dynamic_pressure_backup() {
    let primary = worker("primary");
    let backup = worker("backup");
    let domain = CandidateDomain::global_decode(&[Arc::clone(&primary), Arc::clone(&backup)]);
    let loads = snapshot(&[
        (
            &primary,
            AggregateLoad {
                num_running_reqs: 3,
                max_running_requests: 8,
                num_total_tokens: 900,
                max_total_num_tokens: 2_000,
                ..Default::default()
            },
        ),
        (
            &backup,
            AggregateLoad {
                num_running_reqs: 1,
                max_running_requests: 8,
                num_total_tokens: 100,
                max_total_num_tokens: 2_000,
                ..Default::default()
            },
        ),
    ]);
    let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

    let decision =
        resolve_decode(&domain, &proposal, 64, &loads).expect("both candidates are admitted");

    assert_eq!(decision.selected.id, backup.id);
    assert_eq!(decision.reason, DecisionReason::BackupPressureGuard);
}
