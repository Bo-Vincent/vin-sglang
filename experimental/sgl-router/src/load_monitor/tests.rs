use super::*;
use crate::discovery::{ModelId, WorkerSpec};
use crate::policies::admission::{
    resolve_decode, resolve_prefill, CandidateDomain, CandidateRange, DecisionReason,
};
use crate::policies::{GuardHints, SelectionProposal};

/// Builds one Router worker for store and snapshot tests.
fn test_worker(id: &str, mode: WorkerMode) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.to_string()),
        url: format!("http://{id}:30000"),
        mode,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }))
}

/// Builds a healthy report with one valid rank.
fn test_report(origin: &str, source: &str, sequence: u64, mode: WorkerMode) -> LoadReport {
    LoadReport {
        source_instance_id: source.to_string(),
        sequence_id: sequence,
        report_time_unix_ms: 123,
        worker: Some(proto::Worker {
            worker_addr: origin.to_string(),
            worker_type: worker_type_for_mode(mode) as i32,
            model: Some("model".to_string()),
            zone: None,
        }),
        status: ReportStatus::Healthy as i32,
        last_error: None,
        ranks: vec![RankLoad {
            dp_rank: 0,
            snapshot_time_unix_ms: 123,
            num_running_reqs: 2,
            num_waiting_reqs: 3,
            num_waiting_uncached_tokens: 4,
            num_used_tokens: 20,
            num_total_tokens: 24,
            num_active_tokens: Some(24),
            max_total_num_tokens: 100,
            max_running_requests: 10,
            total_prefill_uncached_tokens: Some(0),
            total_prefill_busy_us: Some(0),
            decode_prealloc_queue_reqs: Some(0),
            decode_transfer_queue_reqs: Some(0),
            decode_retracted_queue_reqs: Some(0),
            total_decode_steps: Some(0),
            total_decode_step_us: Some(0),
            token_usage: 0.2,
            gen_throughput: 5.0,
            cache_hit_rate: 0.5,
            utilization: 0.7,
        }],
    }
}

/// Creates an enabled in-memory monitor without starting network servers.
fn test_monitor() -> LoadMonitor {
    LoadMonitor::new_enabled(
        LoadMonitorConfig {
            enabled: true,
            bind_host: "127.0.0.1".to_string(),
            bind_port: 0,
            report_ip: Some("127.0.0.1".to_string()),
        },
        12345,
    )
    .unwrap()
}

/// Waits until the in-memory snapshot reaches an expected sequence number.
async fn wait_for_sequence(monitor: &LoadMonitor, expected: u64) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if monitor.snapshot().workers[0].sequence_id == Some(expected) {
                return;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("load snapshot did not reach sequence {expected}"));
}

/// Disabled snapshots preserve the documented empty state.
#[test]
fn disabled_snapshot_is_empty() {
    let snapshot = LoadMonitor::disabled().snapshot();
    assert!(!snapshot.enabled);
    assert_eq!(snapshot.version, 0);
    assert!(snapshot.captured_at.is_none());
    assert!(snapshot.workers.is_empty());
}

/// Scheduling captures only fresh aggregate state and keeps diagnostic ranks
/// on the existing snapshot API.
#[tokio::test]
async fn scheduling_snapshot_contains_only_fresh_aggregates() {
    let monitor = test_monitor();
    let fresh = test_worker("fresh", WorkerMode::Prefill);
    let stale = test_worker("stale", WorkerMode::Prefill);
    monitor
        .reconcile(vec![Arc::clone(&fresh), Arc::clone(&stale)])
        .await;

    let fresh_binding = monitor
        .begin_stream(test_report(
            "fresh:30000",
            "fresh-source",
            1,
            WorkerMode::Prefill,
        ))
        .unwrap();
    let mut stale_report = test_report("stale:30000", "stale-source", 1, WorkerMode::Prefill);
    stale_report.status = ReportStatus::Stale as i32;
    let stale_binding = monitor.begin_stream(stale_report).unwrap();

    let scheduling = monitor.scheduling_snapshot();

    assert!(scheduling.enabled);
    assert_eq!(scheduling.fresh_loads.len(), 1);
    assert!(scheduling.fresh_load(&fresh.id).is_some());
    assert!(scheduling.fresh_load(&stale.id).is_none());
    assert_eq!(monitor.snapshot().workers.len(), 2);

    monitor.end_stream(&fresh_binding);
    monitor.end_stream(&stale_binding);
    monitor.stop_registrations().await;
}

/// Rank aggregation sums counters and generation throughput.
#[test]
fn aggregate_sums_rank_loads() {
    let first = validate_rank(&test_report("w:30000", "s", 1, WorkerMode::Plain).ranks[0]).unwrap();
    let mut second = first.clone();
    second.dp_rank = 1;
    let aggregate = aggregate_ranks(&[first, second]);
    assert_eq!(aggregate.total_requests, 10);
    assert_eq!(aggregate.free_tokens, 160);
    assert_eq!(aggregate.available_slots, 16);
    assert_eq!(aggregate.gen_throughput, 10.0);
}

/// Invalid counts, capacity relations, duplicate ranks, and floats are
/// rejected before any worker snapshot can be replaced.
#[test]
fn rejects_invalid_rank_contract_categories() {
    let base = test_report("worker:30000", "source", 1, WorkerMode::Plain);
    let mut cases = Vec::new();

    let mut negative_count = base.clone();
    negative_count.ranks[0].num_running_reqs = -1;
    cases.push(("negative count", negative_count));

    let mut token_capacity = base.clone();
    token_capacity.ranks[0].num_used_tokens = 101;
    cases.push(("token capacity", token_capacity));

    let mut request_capacity = base.clone();
    request_capacity.ranks[0].num_running_reqs = 11;
    cases.push(("request capacity", request_capacity));

    let mut duplicate_rank = base.clone();
    duplicate_rank.ranks.push(duplicate_rank.ranks[0]);
    cases.push(("duplicate rank", duplicate_rank));

    let mut infinite_metric = base;
    infinite_metric.ranks[0].utilization = f64::INFINITY;
    cases.push(("infinite metric", infinite_metric));

    for (category, report) in cases {
        let error = validate_report(&report, SystemTime::now()).unwrap_err();
        assert_eq!(
            error.code(),
            tonic::Code::InvalidArgument,
            "category {category}"
        );
    }
}

/// Negative engine timestamps are rejected even though Router receipt
/// time remains authoritative for freshness.
#[test]
fn rejects_negative_report_timestamp() {
    let mut report = test_report("w:30000", "s", 1, WorkerMode::Plain);
    report.report_time_unix_ms = -1;
    assert!(validate_report(&report, SystemTime::now()).is_err());
}

/// Unreachable reports carry only an explanatory error and no rank set.
#[test]
fn validates_unreachable_report_shape() {
    let mut report = test_report("w:30000", "s", 1, WorkerMode::Plain);
    report.status = ReportStatus::Unreachable as i32;
    report.last_error = Some("scheduler unavailable".to_string());
    assert!(validate_report(&report, SystemTime::now()).is_err());

    report.ranks.clear();
    assert!(validate_report(&report, SystemTime::now()).is_ok());
    report.last_error = None;
    assert!(validate_report(&report, SystemTime::now()).is_err());
}

/// Duplicate and out-of-order sequences leave the latest accepted report.
#[tokio::test]
async fn ignores_non_increasing_sequence_without_closing_stream() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    let binding = monitor
        .begin_stream(test_report("worker:30000", "source", 2, WorkerMode::Plain))
        .unwrap();
    monitor
        .apply_stream_report(
            &binding,
            test_report("worker:30000", "source", 1, WorkerMode::Plain),
        )
        .unwrap();
    assert_eq!(monitor.snapshot().workers[0].sequence_id, Some(2));
    monitor.stop_registrations().await;
}

/// Duplicate discovery origins remain visible but cannot bind a report
/// stream to an arbitrary WorkerId.
#[tokio::test]
async fn duplicate_origin_rejects_stream_binding() {
    let monitor = test_monitor();
    let first = test_worker("worker", WorkerMode::Plain);
    let second = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("worker-copy".to_string()),
        url: first.url.clone(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }));
    monitor.reconcile(vec![first, second]).await;

    let error = monitor
        .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error.message().contains("duplicate worker origin"));
    assert_eq!(monitor.snapshot().workers.len(), 2);
    monitor.stop_registrations().await;
}

/// First-message lookup and subsequent identity checks reject unknown,
/// role-mismatched, or changing report streams.
#[tokio::test]
async fn stream_identity_is_bound_and_immutable() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;

    let unknown = test_report("unknown:30000", "source", 1, WorkerMode::Plain);
    assert!(monitor.begin_stream(unknown).is_err());
    let wrong_role = test_report("worker:30000", "source", 1, WorkerMode::Decode);
    assert!(monitor.begin_stream(wrong_role).is_err());

    let binding = monitor
        .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
        .unwrap();
    let changed_source = test_report("worker:30000", "other-source", 2, WorkerMode::Plain);
    assert!(monitor
        .apply_stream_report(&binding, changed_source)
        .is_err());
    let changed_origin = test_report("other:30000", "source", 2, WorkerMode::Plain);
    assert!(monitor
        .apply_stream_report(&binding, changed_origin)
        .is_err());
    monitor.end_stream(&binding);
    monitor.stop_registrations().await;
}

/// A new source takes ownership and permanently retires the previous one.
#[tokio::test]
async fn new_source_retires_old_source_until_worker_recreated() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    let old = monitor
        .begin_stream(test_report("worker:30000", "old", 1, WorkerMode::Plain))
        .unwrap();
    let _new = monitor
        .begin_stream(test_report("worker:30000", "new", 1, WorkerMode::Plain))
        .unwrap();
    assert!(monitor
        .apply_stream_report(
            &old,
            test_report("worker:30000", "old", 2, WorkerMode::Plain)
        )
        .is_err());
    assert!(monitor
        .begin_stream(test_report("worker:30000", "old", 3, WorkerMode::Plain))
        .is_err());
    monitor.stop_registrations().await;
}

/// A reconnect from the same engine incarnation takes ownership without
/// allowing the old stream or its cleanup path to disturb the new session.
#[tokio::test]
async fn same_source_reconnect_supersedes_old_session() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    let old = monitor
        .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
        .unwrap();
    let new = monitor
        .begin_stream(test_report("worker:30000", "source", 2, WorkerMode::Plain))
        .unwrap();

    let error = monitor
        .apply_stream_report(
            &old,
            test_report("worker:30000", "source", 3, WorkerMode::Plain),
        )
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::Aborted);

    monitor.end_stream(&old);
    monitor
        .apply_stream_report(
            &new,
            test_report("worker:30000", "source", 3, WorkerMode::Plain),
        )
        .unwrap();
    assert_eq!(monitor.snapshot().workers[0].sequence_id, Some(3));
    monitor.stop_registrations().await;
}

/// Healthy reports with zero capacity retain diagnostics but are stale.
#[tokio::test]
async fn zero_capacity_healthy_report_is_locally_stale() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    let mut report = test_report("worker:30000", "source", 1, WorkerMode::Plain);
    report.ranks[0].num_running_reqs = 0;
    report.ranks[0].num_used_tokens = 0;
    report.ranks[0].max_running_requests = 0;
    report.ranks[0].max_total_num_tokens = 0;
    monitor.begin_stream(report).unwrap();
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.workers[0].freshness, Freshness::Stale);
    assert_eq!(snapshot.workers[0].ranks.len(), 1);
    assert_eq!(
        snapshot.workers[0].last_error.as_deref(),
        Some("engine reported a rank with zero token or request capacity")
    );
    monitor.stop_registrations().await;
}

/// Freshness expiration uses Router receipt time rather than engine time.
#[tokio::test]
async fn freshness_expires_from_router_receipt_time() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    monitor
        .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
        .unwrap();
    {
        let mut store = monitor.inner.store.write();
        store
            .workers
            .get_mut(&WorkerId("worker".to_string()))
            .unwrap()
            .report
            .as_mut()
            .unwrap()
            .received_at = SystemTime::now() - STALE_AFTER - Duration::from_millis(1);
    }
    assert_eq!(monitor.snapshot().workers[0].freshness, Freshness::Stale);
    monitor.stop_registrations().await;
}

/// An owned snapshot cannot change after a later report replaces the store.
#[tokio::test]
async fn captured_snapshot_is_immutable_across_updates() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    let binding = monitor
        .begin_stream(test_report("worker:30000", "source", 1, WorkerMode::Plain))
        .unwrap();
    let first = monitor.snapshot();
    monitor
        .apply_stream_report(
            &binding,
            test_report("worker:30000", "source", 2, WorkerMode::Plain),
        )
        .unwrap();
    let second = monitor.snapshot();
    assert_eq!(first.workers[0].sequence_id, Some(1));
    assert_eq!(second.workers[0].sequence_id, Some(2));
    assert!(second.version > first.version);
    monitor.stop_registrations().await;
}

/// Consecutive reports from one source derive prefill queue time and decode
/// step time from monotonic engine counters.
#[tokio::test]
async fn derives_pressure_metrics_from_consecutive_reports() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;

    let mut first = test_report("worker:30000", "source", 1, WorkerMode::Plain);
    first.ranks[0].total_prefill_uncached_tokens = Some(10_000);
    first.ranks[0].total_prefill_busy_us = Some(5_000_000);
    first.ranks[0].total_decode_steps = Some(100);
    first.ranks[0].total_decode_step_us = Some(2_000_000);
    let binding = monitor.begin_stream(first).unwrap();
    assert!(monitor.snapshot().workers[0]
        .aggregate
        .as_ref()
        .unwrap()
        .estimated_prefill_queue_ms
        .is_none());

    let mut second = test_report("worker:30000", "source", 2, WorkerMode::Plain);
    second.ranks[0].num_waiting_uncached_tokens = 800;
    second.ranks[0].num_active_tokens = Some(18);
    second.ranks[0].decode_prealloc_queue_reqs = Some(2);
    second.ranks[0].decode_transfer_queue_reqs = Some(3);
    second.ranks[0].decode_retracted_queue_reqs = Some(1);
    second.ranks[0].total_prefill_uncached_tokens = Some(12_000);
    second.ranks[0].total_prefill_busy_us = Some(5_500_000);
    second.ranks[0].total_decode_steps = Some(120);
    second.ranks[0].total_decode_step_us = Some(2_200_000);
    monitor.apply_stream_report(&binding, second).unwrap();

    let snapshot = monitor.snapshot();
    let aggregate = snapshot.workers[0].aggregate.as_ref().unwrap();
    assert_eq!(aggregate.num_active_tokens, Some(18));
    assert_eq!(aggregate.decode_prealloc_queue_reqs, Some(2));
    assert_eq!(aggregate.decode_transfer_queue_reqs, Some(3));
    assert_eq!(aggregate.decode_retracted_queue_reqs, Some(1));
    assert_eq!(aggregate.prefill_throughput_tokens_per_s, Some(4_000.0));
    assert_eq!(aggregate.estimated_prefill_queue_ms, Some(200.0));
    assert_eq!(aggregate.mean_decode_step_ms, Some(10.0));
    monitor.stop_registrations().await;
}

/// Derived Prefill queue time is consumed by Pressure Guard, not merely
/// published in the snapshot.
#[tokio::test]
async fn derived_prefill_queue_time_changes_guard_decision() {
    let monitor = test_monitor();
    let primary = test_worker("primary", WorkerMode::Prefill);
    let backup = test_worker("backup", WorkerMode::Prefill);
    monitor
        .reconcile(vec![Arc::clone(&primary), Arc::clone(&backup)])
        .await;

    let mut primary_first = test_report("primary:30000", "p-source", 1, WorkerMode::Prefill);
    primary_first.ranks[0].total_prefill_uncached_tokens = Some(1_000);
    primary_first.ranks[0].total_prefill_busy_us = Some(1_000_000);
    let primary_binding = monitor.begin_stream(primary_first).unwrap();

    let mut backup_first = test_report("backup:30000", "b-source", 1, WorkerMode::Prefill);
    backup_first.ranks[0].total_prefill_uncached_tokens = Some(1_000);
    backup_first.ranks[0].total_prefill_busy_us = Some(1_000_000);
    let backup_binding = monitor.begin_stream(backup_first).unwrap();

    let mut primary_second = test_report("primary:30000", "p-source", 2, WorkerMode::Prefill);
    primary_second.ranks[0].num_waiting_uncached_tokens = 100;
    primary_second.ranks[0].total_prefill_uncached_tokens = Some(1_100);
    primary_second.ranks[0].total_prefill_busy_us = Some(2_000_000);
    monitor
        .apply_stream_report(&primary_binding, primary_second)
        .unwrap();

    let mut backup_second = test_report("backup:30000", "b-source", 2, WorkerMode::Prefill);
    backup_second.ranks[0].num_waiting_uncached_tokens = 500;
    backup_second.ranks[0].total_prefill_uncached_tokens = Some(2_000);
    backup_second.ranks[0].total_prefill_busy_us = Some(1_100_000);
    monitor
        .apply_stream_report(&backup_binding, backup_second)
        .unwrap();

    let snapshot = monitor.scheduling_snapshot();
    let range_workers = vec![Arc::clone(&primary), Arc::clone(&backup)];
    let range = CandidateRange::global(&range_workers);
    let token_guard = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup))
        .with_guard_hints(GuardHints {
            enable_pressure_guard: true,
            pressure_abs_threshold_tokens: 10,
            pressure_abs_threshold_ms: None,
            pressure_rel_threshold: 2.0,
        });
    let token_decision = resolve_prefill(&range, &token_guard, 1, &snapshot).unwrap();
    assert_eq!(token_decision.selected.id, primary.id);

    let queue_time_guard =
        SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup)).with_guard_hints(
            GuardHints {
                enable_pressure_guard: true,
                pressure_abs_threshold_tokens: 10,
                pressure_abs_threshold_ms: Some(100.0),
                pressure_rel_threshold: 2.0,
            },
        );
    let queue_time_decision = resolve_prefill(&range, &queue_time_guard, 1, &snapshot).unwrap();
    assert_eq!(queue_time_decision.selected.id, backup.id);
    assert_eq!(
        queue_time_decision.reason,
        DecisionReason::BackupPressureGuard
    );

    monitor.stop_registrations().await;
}

/// Decode queue gauges and step counters flow through LoadMonitor into the
/// Decode pressure ordering.
#[tokio::test]
async fn decode_pressure_metrics_change_policy_decision() {
    let monitor = test_monitor();
    let primary = test_worker("primary", WorkerMode::Decode);
    let backup = test_worker("backup", WorkerMode::Decode);
    monitor
        .reconcile(vec![Arc::clone(&primary), Arc::clone(&backup)])
        .await;

    let mut primary_first = test_report("primary:30000", "p-source", 1, WorkerMode::Decode);
    primary_first.ranks[0].total_decode_steps = Some(100);
    primary_first.ranks[0].total_decode_step_us = Some(1_000_000);
    let primary_binding = monitor.begin_stream(primary_first).unwrap();

    let mut backup_first = test_report("backup:30000", "b-source", 1, WorkerMode::Decode);
    backup_first.ranks[0].total_decode_steps = Some(100);
    backup_first.ranks[0].total_decode_step_us = Some(1_000_000);
    let backup_binding = monitor.begin_stream(backup_first).unwrap();

    let mut primary_second = test_report("primary:30000", "p-source", 2, WorkerMode::Decode);
    primary_second.ranks[0].decode_retracted_queue_reqs = Some(1);
    primary_second.ranks[0].total_decode_steps = Some(110);
    primary_second.ranks[0].total_decode_step_us = Some(1_200_000);
    monitor
        .apply_stream_report(&primary_binding, primary_second)
        .unwrap();

    let mut backup_second = test_report("backup:30000", "b-source", 2, WorkerMode::Decode);
    backup_second.ranks[0].decode_retracted_queue_reqs = Some(0);
    backup_second.ranks[0].total_decode_steps = Some(110);
    backup_second.ranks[0].total_decode_step_us = Some(1_100_000);
    monitor
        .apply_stream_report(&backup_binding, backup_second)
        .unwrap();

    let snapshot = monitor.scheduling_snapshot();
    let domain = CandidateDomain::global_decode(&[Arc::clone(&primary), Arc::clone(&backup)]);
    let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));
    let decision = resolve_decode(&domain, &proposal, 1, &snapshot).unwrap();
    assert_eq!(decision.selected.id, backup.id);
    assert_eq!(decision.reason, DecisionReason::BackupPressureGuard);

    monitor.stop_registrations().await;
}

/// Counter resets and producer restarts invalidate only the derived sample;
/// the fresh direct gauges remain available.
#[tokio::test]
async fn counter_reset_and_source_restart_drop_derived_metrics() {
    let monitor = test_monitor();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;
    let mut first = test_report("worker:30000", "source", 1, WorkerMode::Plain);
    first.ranks[0].total_prefill_uncached_tokens = Some(1_000);
    first.ranks[0].total_prefill_busy_us = Some(1_000);
    first.ranks[0].total_decode_steps = Some(100);
    first.ranks[0].total_decode_step_us = Some(100_000);
    let binding = monitor.begin_stream(first).unwrap();

    let mut reset = test_report("worker:30000", "source", 2, WorkerMode::Plain);
    reset.ranks[0].num_active_tokens = Some(12);
    reset.ranks[0].total_prefill_uncached_tokens = Some(10);
    reset.ranks[0].total_prefill_busy_us = Some(10);
    reset.ranks[0].total_decode_steps = Some(1);
    reset.ranks[0].total_decode_step_us = Some(1_000);
    monitor.apply_stream_report(&binding, reset).unwrap();
    let reset_snapshot = monitor.snapshot();
    let reset_load = reset_snapshot.workers[0].aggregate.as_ref().unwrap();
    assert_eq!(reset_load.num_active_tokens, Some(12));
    assert!(reset_load.estimated_prefill_queue_ms.is_none());
    assert!(reset_load.mean_decode_step_ms.is_none());

    monitor.end_stream(&binding);
    let restarted = monitor
        .begin_stream(test_report(
            "worker:30000",
            "new-source",
            1,
            WorkerMode::Plain,
        ))
        .unwrap();
    let restarted_snapshot = monitor.snapshot();
    let restarted_load = restarted_snapshot.workers[0].aggregate.as_ref().unwrap();
    assert!(restarted_load.estimated_prefill_queue_ms.is_none());
    assert!(restarted_load.mean_decode_step_ms.is_none());
    monitor.end_stream(&restarted);
    monitor.stop_registrations().await;
}

/// Optional fields keep a rolling upgrade from rejecting older reporters.
#[test]
fn accepts_legacy_rank_without_step3_metrics() {
    let mut report = test_report("worker:30000", "source", 1, WorkerMode::Plain);
    let rank = &mut report.ranks[0];
    rank.num_active_tokens = None;
    rank.total_prefill_uncached_tokens = None;
    rank.total_prefill_busy_us = None;
    rank.decode_prealloc_queue_reqs = None;
    rank.decode_transfer_queue_reqs = None;
    rank.decode_retracted_queue_reqs = None;
    rank.total_decode_steps = None;
    rank.total_decode_step_us = None;

    let accepted = validate_report(&report, SystemTime::now()).unwrap();
    assert_eq!(accepted.aggregate.num_active_tokens, None);
    assert_eq!(accepted.aggregate.estimated_prefill_queue_ms, None);
    assert_eq!(accepted.aggregate.mean_decode_step_ms, None);
}

#[test]
fn rejects_incomplete_step3_counter_pairs() {
    let mut prefill = test_report("worker:30000", "source", 1, WorkerMode::Plain);
    prefill.ranks[0].total_prefill_busy_us = None;
    let error = validate_report(&prefill, SystemTime::now()).unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error
        .message()
        .contains("prefill cumulative counters must be reported together"));

    let mut decode = test_report("worker:30000", "source", 1, WorkerMode::Plain);
    decode.ranks[0].total_decode_step_us = None;
    let error = validate_report(&decode, SystemTime::now()).unwrap_err();
    assert_eq!(error.code(), tonic::Code::InvalidArgument);
    assert!(error
        .message()
        .contains("decode cumulative counters must be reported together"));
}

/// Exercises actual HTTP registration, an ephemeral gRPC listener, stream
/// ingestion, and immutable snapshot publication without engine auth.
#[tokio::test]
async fn fake_engine_registration_and_grpc_report_form_complete_loop() {
    use axum::routing::post;
    use axum::{Json, Router};
    use tokio::sync::mpsc;

    let (registration_tx, mut registration_rx) = mpsc::channel(4);
    let app = Router::new().route(
        START_REPORTING_PATH,
        post(
            move |headers: axum::http::HeaderMap, Json(body): Json<serde_json::Value>| {
                let registration_tx = registration_tx.clone();
                async move {
                    registration_tx
                        .send((
                            body,
                            headers.contains_key(axum::http::header::AUTHORIZATION),
                        ))
                        .await
                        .unwrap();
                    axum::http::StatusCode::OK
                }
            },
        ),
    );
    let engine_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let engine_addr = engine_listener.local_addr().unwrap();
    let engine_server = tokio::spawn(async move {
        axum::serve(engine_listener, app).await.unwrap();
    });

    let config = LoadMonitorConfig {
        enabled: true,
        bind_host: "127.0.0.1".to_string(),
        bind_port: 0,
        report_ip: Some("127.0.0.1".to_string()),
    };
    let (monitor, grpc) = bind_and_serve(config).await.unwrap();
    let worker = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("worker".to_string()),
        url: format!("http://{engine_addr}"),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }));
    monitor.reconcile(vec![worker]).await;

    let (registration, has_authorization) =
        tokio::time::timeout(Duration::from_secs(3), registration_rx.recv())
            .await
            .unwrap()
            .unwrap();
    assert!(!has_authorization, "Router must not send a Bearer token");
    assert_eq!(registration["ip"], "127.0.0.1");
    assert_eq!(
        registration["port"].as_u64(),
        Some(grpc.local_addr().port() as u64)
    );
    assert_eq!(registration["report_interval_ms"].as_u64(), Some(1000));
    assert_eq!(registration["lease_ttl_ms"].as_u64(), Some(15000));

    let mut client = proto::load_monitor_service_client::LoadMonitorServiceClient::connect(
        format!("http://{}", grpc.local_addr()),
    )
    .await
    .unwrap();
    let report = test_report(
        &engine_addr.to_string(),
        "fake-engine",
        1,
        WorkerMode::Plain,
    );
    client
        .report(tokio_stream::iter(vec![report]))
        .await
        .unwrap();
    let snapshot = monitor.snapshot();
    assert_eq!(snapshot.workers[0].freshness, Freshness::Fresh);
    assert_eq!(
        snapshot.workers[0]
            .aggregate
            .as_ref()
            .unwrap()
            .total_requests,
        5
    );
    let renewal = tokio::time::timeout(Duration::from_secs(3), registration_rx.recv())
        .await
        .unwrap();
    assert!(renewal.is_some(), "Router must renew the reporting lease");

    grpc.shutdown(&monitor).await;
    engine_server.abort();
}

/// Exercises the network service with two overlapping streams from the same
/// engine incarnation, matching a reconnect while the old HTTP/2 stream is
/// still half-open in the Router.
#[tokio::test]
async fn grpc_same_source_reconnect_aborts_old_stream_and_keeps_new_stream() {
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    let config = LoadMonitorConfig {
        enabled: true,
        bind_host: "127.0.0.1".to_string(),
        bind_port: 0,
        report_ip: Some("127.0.0.1".to_string()),
    };
    let (monitor, grpc) = bind_and_serve(config).await.unwrap();
    monitor
        .reconcile(vec![test_worker("worker", WorkerMode::Plain)])
        .await;

    let client = proto::load_monitor_service_client::LoadMonitorServiceClient::connect(format!(
        "http://{}",
        grpc.local_addr()
    ))
    .await
    .unwrap();

    let (old_tx, old_rx) = mpsc::channel(4);
    let mut old_client = client.clone();
    let old_rpc = tokio::spawn(async move { old_client.report(ReceiverStream::new(old_rx)).await });
    old_tx
        .send(test_report(
            "worker:30000",
            "same-engine",
            1,
            WorkerMode::Plain,
        ))
        .await
        .unwrap();
    wait_for_sequence(&monitor, 1).await;

    let (new_tx, new_rx) = mpsc::channel(4);
    let mut new_client = client;
    let new_rpc = tokio::spawn(async move { new_client.report(ReceiverStream::new(new_rx)).await });
    new_tx
        .send(test_report(
            "worker:30000",
            "same-engine",
            2,
            WorkerMode::Plain,
        ))
        .await
        .unwrap();
    wait_for_sequence(&monitor, 2).await;

    let old_error = tokio::time::timeout(Duration::from_secs(1), old_rpc)
        .await
        .expect("superseded stream must be closed promptly")
        .unwrap()
        .unwrap_err();
    assert_eq!(old_error.code(), tonic::Code::Aborted);

    new_tx
        .send(test_report(
            "worker:30000",
            "same-engine",
            3,
            WorkerMode::Plain,
        ))
        .await
        .unwrap();
    wait_for_sequence(&monitor, 3).await;

    drop(old_tx);
    drop(new_tx);
    new_rpc.await.unwrap().unwrap();
    monitor.stop_registrations().await;
    grpc.shutdown(&monitor).await;
}

/// Retryable HTTP responses back off, terminal 4xx responses pause until
/// the next topology reconcile, and removal clears monitor state.
#[tokio::test]
async fn registration_retry_terminal_response_and_removal_reconcile() {
    use axum::routing::post;
    use axum::Router;
    use tokio::sync::mpsc;

    let attempts = Arc::new(AtomicUsize::new(0));
    let (attempt_tx, mut attempt_rx) = mpsc::channel(8);
    let attempts_for_handler = Arc::clone(&attempts);
    let app = Router::new().route(
        START_REPORTING_PATH,
        post(move || {
            let attempt_tx = attempt_tx.clone();
            let attempt = attempts_for_handler.fetch_add(1, Ordering::AcqRel) + 1;
            async move {
                attempt_tx.send(attempt).await.unwrap();
                match attempt {
                    1 => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    2 => axum::http::StatusCode::TOO_MANY_REQUESTS,
                    3 => axum::http::StatusCode::BAD_REQUEST,
                    _ => axum::http::StatusCode::OK,
                }
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let engine_addr = listener.local_addr().unwrap();
    let engine_server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let monitor = LoadMonitor::new_enabled(
        LoadMonitorConfig {
            enabled: true,
            bind_host: "127.0.0.1".to_string(),
            bind_port: 0,
            report_ip: Some("127.0.0.1".to_string()),
        },
        3456,
    )
    .unwrap();
    let worker = Arc::new(Worker::new(WorkerSpec {
        id: WorkerId("worker".to_string()),
        url: format!("http://{engine_addr}"),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("model".to_string())],
        bootstrap_port: None,
    }));
    monitor.reconcile(vec![Arc::clone(&worker)]).await;

    for expected in 1..=3 {
        let actual = tokio::time::timeout(Duration::from_secs(3), attempt_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(actual, expected);
    }
    assert!(
        tokio::time::timeout(Duration::from_millis(1200), attempt_rx.recv())
            .await
            .is_err(),
        "terminal 4xx must pause registration until topology reconcile"
    );

    monitor.reconcile(vec![worker]).await;
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), attempt_rx.recv())
            .await
            .unwrap(),
        Some(4)
    );
    monitor.reconcile(Vec::new()).await;
    assert!(monitor.snapshot().workers.is_empty());

    monitor.stop_registrations().await;
    engine_server.abort();
}
