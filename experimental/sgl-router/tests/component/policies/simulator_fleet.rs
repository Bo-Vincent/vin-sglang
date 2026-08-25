// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! CPU virtual-fleet coverage for Router policy decisions.

use serde::Serialize;
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use sgl_router::config::AffinityConfig;
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::load_monitor::{AggregateLoad, Freshness, LoadMonitorSnapshot, WorkerSnapshot};
use sgl_router::policies::admission::{resolve_cache_candidates, resolve_prefill, CandidateDomain};
use sgl_router::policies::cache_aware::CacheAwarePolicy;
use sgl_router::policies::power_of_two::PowerOfTwoChoicesPolicy;
use sgl_router::policies::{ExternalPrefixSignal, Policy, PrefillProposal, SelectionContext};
use sgl_router::workers::Worker;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

const INPUT_TOKENS: u64 = 8_192;
const QUERY_BLOCKS: usize = 8;
const CACHED_BLOCKS: u32 = 7;
const HOT_PREFIXES: usize = 256;
const REPLICAS_PER_PREFIX: usize = 4;
const ARRIVAL_INTERVAL_MS: f64 = 2.0;
const MODEL: &str = "router-simulator-fleet";

#[derive(Clone, Copy)]
enum PolicyKind {
    PowerOfTwo,
    CacheAware,
}

#[derive(Debug, Serialize)]
struct VirtualMetrics {
    requests: usize,
    cache_hit_rate: f64,
    p95_ttft_ms: f64,
    mean_ttft_ms: f64,
    simulated_throughput_rps: f64,
    worker_cv: f64,
    decision_p95_us: f64,
    invalid_selections: usize,
    reasons: BTreeMap<String, usize>,
}

#[derive(Serialize)]
struct MatrixRow {
    endpoint_count: usize,
    scenario: &'static str,
    policy: &'static str,
    repeat: usize,
    metrics: VirtualMetrics,
}

#[derive(Default)]
struct VirtualWorker {
    available_at_ms: f64,
    cached_prefixes: BTreeSet<usize>,
    selections: usize,
}

struct VirtualFleet {
    workers: Vec<VirtualWorker>,
}

impl VirtualFleet {
    fn new(endpoint_count: usize) -> Self {
        let mut workers = (0..endpoint_count)
            .map(|_| VirtualWorker::default())
            .collect::<Vec<_>>();
        for prefix in 0..HOT_PREFIXES {
            for replica in 0..REPLICAS_PER_PREFIX.min(endpoint_count) {
                let worker = (prefix * REPLICAS_PER_PREFIX + replica) % endpoint_count;
                workers[worker].cached_prefixes.insert(prefix);
            }
        }
        Self { workers }
    }

    fn snapshot(&self, workers: &[Arc<Worker>], now_ms: f64, version: u64) -> LoadMonitorSnapshot {
        let workers = workers
            .iter()
            .zip(&self.workers)
            .map(|(worker, state)| {
                let queue_ms = (state.available_at_ms - now_ms).max(0.0);
                let running = u64::from(queue_ms > 0.0);
                WorkerSnapshot {
                    worker_id: worker.id.0.clone(),
                    url: worker.url.clone(),
                    mode: WorkerMode::Prefill,
                    model_ids: vec![MODEL.to_string()],
                    freshness: Freshness::Fresh,
                    source_instance_id: Some("virtual-simulator".to_string()),
                    sequence_id: Some(version),
                    report_time_unix_ms: Some(version as i64),
                    last_error: None,
                    received_at: None,
                    expires_at: None,
                    aggregate: Some(AggregateLoad {
                        rank_count: 1,
                        num_running_reqs: running,
                        num_waiting_reqs: running,
                        total_requests: running.saturating_mul(2),
                        num_waiting_uncached_tokens: (queue_ms * 1_024.0) as u64,
                        num_used_tokens: 0,
                        num_total_tokens: 0,
                        num_active_tokens: Some(0),
                        max_total_num_tokens: 1_000_000,
                        max_running_requests: 1_024,
                        decode_prealloc_queue_reqs: Some(0),
                        decode_transfer_queue_reqs: Some(0),
                        decode_retracted_queue_reqs: Some(0),
                        free_tokens: 1_000_000,
                        available_slots: 1_024 - running,
                        prefill_throughput_tokens_per_s: Some(1_024_000.0),
                        estimated_prefill_queue_ms: Some(queue_ms),
                        mean_decode_step_ms: Some(0.0),
                        request_utilization: running as f64 / 1_024.0,
                        weighted_token_usage: 0.0,
                        max_rank_token_usage: 0.0,
                        gen_throughput: 0.0,
                    }),
                    ranks: Vec::new(),
                }
            })
            .collect();
        LoadMonitorSnapshot {
            enabled: true,
            version,
            captured_at: None,
            workers,
        }
    }

    fn prefix_signal(&self, workers: &[Arc<Worker>], prefix: usize) -> ExternalPrefixSignal {
        let matches = workers
            .iter()
            .zip(&self.workers)
            .filter(|(_, state)| state.cached_prefixes.contains(&prefix))
            .map(|(worker, _)| PrefixMatch {
                address: worker.url.clone(),
                matched_prefix_blocks: CACHED_BLOCKS,
                worker_id: worker.id.0.clone(),
            })
            .collect::<Vec<_>>();
        let outcome = if matches.is_empty() {
            PrefixOutcome::Empty
        } else {
            PrefixOutcome::Matched {
                best_prefix_blocks: CACHED_BLOCKS,
                matches,
            }
        };
        ExternalPrefixSignal {
            outcome,
            query_blocks: QUERY_BLOCKS,
        }
    }

    fn dispatch(&mut self, worker: usize, prefix: usize, now_ms: f64) -> (bool, f64) {
        let state = &mut self.workers[worker];
        let cache_hit = state.cached_prefixes.contains(&prefix);
        let prefill_ms = replay_prefill_ms(cache_hit);
        let queue_ms = (state.available_at_ms - now_ms).max(0.0);
        state.available_at_ms = now_ms.max(state.available_at_ms) + prefill_ms;
        state.cached_prefixes.insert(prefix);
        state.selections += 1;
        (cache_hit, queue_ms + prefill_ms)
    }
}

// #33824 示例 replay table 中 `[1, 3] -> 1 ms` 与 `[8, 0] -> 8 ms`。
// 本测试把请求分成八个等大的 prefill blocks：命中七个 block 时走 1 ms，
// 无 prefix 命中时走 8 ms。它是 logical-time 的 prefill/TTFT proxy，不是 GPU 延迟声明。
fn replay_prefill_ms(cache_hit: bool) -> f64 {
    if cache_hit {
        1.0
    } else {
        8.0
    }
}

fn router_workers(endpoint_count: usize) -> Vec<Arc<Worker>> {
    (0..endpoint_count)
        .map(|index| {
            Arc::new(Worker::new(WorkerSpec {
                id: WorkerId(format!("virtual-{index:04}")),
                url: format!("http://127.0.0.1:{}", 30_000 + index),
                mode: WorkerMode::Prefill,
                model_ids: vec![ModelId(MODEL.to_string())],
                bootstrap_port: None,
            }))
        })
        .collect()
}

fn percentile(mut values: Vec<f64>, percentile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index]
}

fn coefficient_of_variation(values: &[usize]) -> f64 {
    let mean = values.iter().sum::<usize>() as f64 / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| (*value as f64 - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt() / mean
}

fn run_virtual_fleet(endpoint_count: usize, kind: PolicyKind, requests: usize) -> VirtualMetrics {
    run_virtual_fleet_with_arrival(endpoint_count, kind, requests, ARRIVAL_INTERVAL_MS)
}

fn run_virtual_fleet_with_arrival(
    endpoint_count: usize,
    kind: PolicyKind,
    requests: usize,
    arrival_interval_ms: f64,
) -> VirtualMetrics {
    let workers = router_workers(endpoint_count);
    let worker_positions = workers
        .iter()
        .enumerate()
        .map(|(index, worker)| (worker.id.0.clone(), index))
        .collect::<BTreeMap<_, _>>();
    let domain = CandidateDomain::global_prefill(&workers);
    let range = domain.prefill_range().expect("prefill domain");
    let model = ModelId(MODEL.to_string());
    let power_of_two = PowerOfTwoChoicesPolicy::new();
    let cache_aware = CacheAwarePolicy::new(AffinityConfig {
        cache_affinity_min_matched_tokens: Some(1_024),
        cache_candidate_min_workers: REPLICAS_PER_PREFIX,
        cache_candidate_ratio: 0.0,
        cache_candidate_max_workers: REPLICAS_PER_PREFIX,
        cache_switch_margin_tokens: 1_024,
        pressure_guard: true,
        pressure_abs_threshold_ms: Some(1.0),
        pressure_rel_threshold: 1.25,
        ..Default::default()
    });
    let mut fleet = VirtualFleet::new(endpoint_count);
    let mut ttft_ms = Vec::with_capacity(requests);
    let mut decision_us = Vec::with_capacity(requests);
    let mut cache_hits = 0usize;
    let mut invalid_selections = 0usize;
    let mut reasons = BTreeMap::new();

    for request in 0..requests {
        let now_ms = request as f64 * arrival_interval_ms;
        let prefix = request % HOT_PREFIXES;
        let snapshot = fleet.snapshot(&workers, now_ms, request as u64 + 1);
        let started = Instant::now();
        let decision = match kind {
            PolicyKind::PowerOfTwo => {
                let ctx = SelectionContext::new(&model, None)
                    .with_input_tokens(INPUT_TOKENS)
                    .with_load_snapshot(&snapshot)
                    .with_candidate_range_id(range.id);
                let proposal = power_of_two
                    .propose(&domain.workers, &ctx)
                    .expect("power-of-two proposal");
                resolve_prefill(&range, &proposal, INPUT_TOKENS, &snapshot)
                    .expect("power-of-two decision")
            }
            PolicyKind::CacheAware => {
                let signal = fleet.prefix_signal(&workers, prefix);
                let ctx = SelectionContext::new(&model, None)
                    .with_input_tokens(INPUT_TOKENS)
                    .with_external_prefix(Some(&signal))
                    .with_load_snapshot(&snapshot)
                    .with_candidate_range_id(range.id);
                match cache_aware
                    .propose_prefill(&domain.workers, &ctx)
                    .expect("cache-aware proposal")
                {
                    PrefillProposal::CacheCandidates(proposal) => {
                        resolve_cache_candidates(&proposal, INPUT_TOKENS, &snapshot)
                            .expect("cache candidate decision")
                    }
                    PrefillProposal::Pair(proposal) => {
                        resolve_prefill(&range, &proposal, INPUT_TOKENS, &snapshot)
                            .expect("cache fallback decision")
                    }
                }
            }
        };
        decision_us.push(started.elapsed().as_secs_f64() * 1_000_000.0);
        *reasons.entry(format!("{:?}", decision.reason)).or_insert(0) += 1;
        let Some(&worker) = worker_positions.get(&decision.selected.id.0) else {
            invalid_selections += 1;
            continue;
        };
        let (cache_hit, ttft) = fleet.dispatch(worker, prefix, now_ms);
        cache_hits += usize::from(cache_hit);
        ttft_ms.push(ttft);
    }

    let makespan_ms = fleet
        .workers
        .iter()
        .map(|worker| worker.available_at_ms)
        .fold(0.0, f64::max);
    VirtualMetrics {
        requests,
        cache_hit_rate: cache_hits as f64 / requests as f64,
        p95_ttft_ms: percentile(ttft_ms.clone(), 0.95),
        mean_ttft_ms: ttft_ms.iter().sum::<f64>() / ttft_ms.len() as f64,
        simulated_throughput_rps: requests as f64 * 1_000.0 / makespan_ms,
        worker_cv: coefficient_of_variation(
            &fleet
                .workers
                .iter()
                .map(|worker| worker.selections)
                .collect::<Vec<_>>(),
        ),
        decision_p95_us: percentile(decision_us, 0.95),
        invalid_selections,
        reasons,
    }
}

fn write_matrix_if_requested(rows: &[MatrixRow]) {
    let Ok(path) = std::env::var("SGL_ROUTER_SIMULATOR_FLEET_REPORT") else {
        return;
    };
    let encoded = serde_json::to_vec_pretty(rows).expect("serialize virtual fleet matrix");
    std::fs::write(path, encoded).expect("write virtual fleet matrix");
}

#[test]
fn cache_aware_reuses_replicated_prefixes_without_increasing_simulated_ttft() {
    let power_of_two = run_virtual_fleet(64, PolicyKind::PowerOfTwo, 2_048);
    let cache_aware = run_virtual_fleet(64, PolicyKind::CacheAware, 2_048);

    assert!(cache_aware.cache_hit_rate > power_of_two.cache_hit_rate + 0.30);
    assert!(cache_aware.p95_ttft_ms <= power_of_two.p95_ttft_ms);
}

#[test]
fn virtual_fleet_selects_only_known_workers_at_large_endpoint_counts() {
    for endpoint_count in [8, 64, 256] {
        let result = run_virtual_fleet(endpoint_count, PolicyKind::CacheAware, 2_048);
        assert_eq!(result.requests, 2_048);
        assert_eq!(result.invalid_selections, 0);
    }
}

#[test]
fn cache_aware_uses_pressure_to_spread_equivalent_cache_replicas() {
    let low_pressure_cache_aware = run_virtual_fleet(64, PolicyKind::CacheAware, 4_096);
    let power_of_two = run_virtual_fleet_with_arrival(64, PolicyKind::PowerOfTwo, 4_096, 0.05);
    let cache_aware = run_virtual_fleet_with_arrival(64, PolicyKind::CacheAware, 4_096, 0.05);

    assert!(cache_aware.cache_hit_rate > power_of_two.cache_hit_rate + 0.30);
    assert!(cache_aware.worker_cv < low_pressure_cache_aware.worker_cv);
    assert!(cache_aware.p95_ttft_ms <= power_of_two.p95_ttft_ms);
}

#[test]
fn virtual_fleet_matrix_records_policy_reasons_and_metrics() {
    let mut rows = Vec::new();
    for endpoint_count in [8, 64, 256] {
        for (scenario, arrival_interval_ms) in [("arrival_500_rps", 2.0), ("arrival_20k_rps", 0.05)]
        {
            for (policy, kind) in [
                ("power_of_two", PolicyKind::PowerOfTwo),
                ("cache_aware", PolicyKind::CacheAware),
            ] {
                for repeat in 0..3 {
                    let metrics = run_virtual_fleet_with_arrival(
                        endpoint_count,
                        kind,
                        4_096,
                        arrival_interval_ms,
                    );
                    assert_eq!(metrics.invalid_selections, 0);
                    if matches!(kind, PolicyKind::CacheAware) {
                        assert!(metrics.reasons.contains_key("CacheCandidate"));
                    }
                    rows.push(MatrixRow {
                        endpoint_count,
                        scenario,
                        policy,
                        repeat,
                        metrics,
                    });
                }
            }
        }
    }
    assert_eq!(rows.len(), 36);
    write_matrix_if_requested(&rows);
}
