// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Router-only pressure-path microbenchmark.
//!
//! The fixture constructs the same #34608 engine-load table that the ZMQ load
//! subscriber fills in production. Policy-only cases reuse one captured
//! snapshot; `request_path` cases include a fresh snapshot capture on every
//! decision.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use sgl_kv_indexer::{PrefixMatch, PrefixOutcome};
use sgl_router::config::AffinityConfig;
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::admission::{
    resolve_cache_candidates, resolve_decode, resolve_prefill, CandidateDomain,
};
use sgl_router::policies::cache_aware::CacheAwarePolicy;
use sgl_router::policies::decode::{DecodePolicy, DecodePowerOfTwoPolicy, DecodeSelectionContext};
use sgl_router::policies::engine_load::{EngineLoadSnapshot, EngineLoadTable, LoadStat};
use sgl_router::policies::power_of_two::PowerOfTwoChoicesPolicy;
use sgl_router::policies::session_aware::SessionAwarePolicy;
use sgl_router::policies::{
    ExternalPrefixSignal, Policy, PrefillProposal, ProposalKind, SelectionContext,
};
use sgl_router::workers::Worker;
use std::alloc::{GlobalAlloc, Layout, System};
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const ENDPOINT_COUNTS: &[usize] = &[8, 64, 256];
const CACHE_TOP_K: &[usize] = &[4, 16, 32];
const MODEL: &str = "router-pressure-bench";
const INPUT_TOKENS: u64 = 32_768;
const MAX_TOTAL_TOKENS: u64 = 262_144;

struct CountingAllocator;

static ALLOC_ENABLED: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ALLOC_ENABLED.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        // SAFETY: Delegates the unchanged layout to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if ALLOC_ENABLED.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        // SAFETY: Delegates the unchanged layout to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` came from the delegated system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ALLOC_ENABLED.load(Ordering::Relaxed) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        }
        // SAFETY: Delegates the original allocation and new size unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

struct Fixture {
    loads: Arc<EngineLoadTable>,
    workers: Vec<Arc<Worker>>,
    snapshot: EngineLoadSnapshot,
    model: ModelId,
}

fn capture_policy_snapshot(loads: &EngineLoadTable) -> EngineLoadSnapshot {
    loads.capture_snapshot(Instant::now())
}

fn worker(index: usize, mode: WorkerMode) -> Arc<Worker> {
    let host = format!("127.0.{}.{}", index / 254, index % 254 + 1);
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(format!("worker-{index:04}")),
        url: format!("http://{host}:{}", 30_000 + index),
        mode,
        model_ids: vec![ModelId(MODEL.to_string())],
        bootstrap_port: None,
    }))
}

fn build_fixture(endpoint_count: usize, ranks: usize, mode: WorkerMode) -> Fixture {
    let loads = EngineLoadTable::new();
    let workers: Vec<_> = (0..endpoint_count)
        .map(|index| worker(index, mode))
        .collect();
    let now = Instant::now();
    for (index, worker) in workers.iter().enumerate() {
        for rank in 0..ranks {
            loads.mark_expected_rank(&worker.url, rank as u32);
            loads.set(
                &worker.url,
                rank as u32,
                LoadStat {
                    num_running_reqs: ((index + rank) % 32) as u64,
                    num_waiting_reqs: ((index * 3 + rank) % 24) as u64,
                    num_tokens: 32_768 + ((index * 257 + rank * 101) % 65_536) as u64,
                    max_total_num_tokens: MAX_TOTAL_TOKENS,
                },
                now,
            );
        }
    }
    let snapshot = capture_policy_snapshot(&loads);
    assert!(workers
        .iter()
        .all(|worker| snapshot.fresh_load_for_url(&worker.url).is_some()));
    Fixture {
        loads,
        workers,
        snapshot,
        model: ModelId(MODEL.to_string()),
    }
}

fn measure_allocations<T>(iterations: u64, mut operation: impl FnMut() -> T) -> (f64, f64) {
    ALLOC_COUNT.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    ALLOC_ENABLED.store(true, Ordering::Release);
    for _ in 0..iterations {
        let value = operation();
        black_box(&value);
        drop(value);
    }
    ALLOC_ENABLED.store(false, Ordering::Release);
    (
        ALLOC_COUNT.load(Ordering::Relaxed) as f64 / iterations as f64,
        ALLOC_BYTES.load(Ordering::Relaxed) as f64 / iterations as f64,
    )
}

fn record_allocations<T>(name: &str, mut operation: impl FnMut() -> T) {
    let Ok(path) = std::env::var("SGL_ROUTER_ALLOC_LOG") else {
        return;
    };
    let (allocations, bytes) = measure_allocations(100, &mut operation);
    let mut output = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open allocation log");
    writeln!(output, "{name},{allocations:.6},{bytes:.6}").expect("write allocation record");
}

fn prefill_decision(
    policy: &dyn Policy,
    fixture: &Fixture,
    domain: &CandidateDomain,
    snapshot: &EngineLoadSnapshot,
    session_id: Option<&str>,
) -> WorkerId {
    let range = domain
        .prefill_range()
        .expect("prefill benchmark requires a prefill domain");
    let ctx = SelectionContext::new(&fixture.model, None)
        .with_session_id(session_id)
        .with_input_tokens(INPUT_TOKENS)
        .with_load_snapshot(snapshot)
        .with_candidate_range_id(range.id);
    let proposal = policy
        .propose(&domain.workers, &ctx)
        .expect("prefill policy must propose a worker");
    let decision = resolve_prefill(&range, &proposal, INPUT_TOKENS, snapshot)
        .expect("prefill decision must stay in its domain");
    decision.selected.id.clone()
}

fn initialized_session_policy(fixture: &Fixture) -> SessionAwarePolicy {
    let policy = SessionAwarePolicy::new(AffinityConfig::default());
    let ctx = SelectionContext::new(&fixture.model, None)
        .with_session_id(Some("stable-session"))
        .with_input_tokens(INPUT_TOKENS)
        .with_load_snapshot(&fixture.snapshot);
    let primary = &fixture.workers[fixture.workers.len() / 2];
    policy.commit_prefill_selection(&ctx, ProposalKind::PowerOfTwo, primary);
    policy
}

fn cache_signal(workers: &[Arc<Worker>]) -> ExternalPrefixSignal {
    let matches = workers
        .iter()
        .enumerate()
        .map(|(index, worker)| PrefixMatch {
            matched_prefix_blocks: (workers.len() - index) as u32,
            // Route on the worker URL: the #33370 indexer contract matches
            // PrefixMatch.address against registered worker URLs.
            address: worker.url.clone(),
            worker_id: worker.id.0.clone(),
        })
        .collect();
    let best_prefix_blocks = workers.len() as u32;
    ExternalPrefixSignal {
        outcome: PrefixOutcome::Matched {
            matches,
            best_prefix_blocks,
        },
        query_blocks: workers.len(),
    }
}

fn cache_decision(
    policy: &CacheAwarePolicy,
    fixture: &Fixture,
    signal: &ExternalPrefixSignal,
    snapshot: &EngineLoadSnapshot,
) -> WorkerId {
    let ctx = SelectionContext::new(&fixture.model, None)
        .with_input_tokens(INPUT_TOKENS)
        .with_external_prefix(Some(signal))
        .with_load_snapshot(snapshot);
    let PrefillProposal::CacheCandidates(proposal) = policy
        .propose_prefill(&fixture.workers, &ctx)
        .expect("cache policy must propose candidates")
    else {
        panic!("cache benchmark must exercise the candidate path");
    };
    resolve_cache_candidates(&proposal, INPUT_TOKENS, snapshot)
        .expect("cache candidates must pass admission")
        .selected
        .id
        .clone()
}

fn decode_decision(domain: &CandidateDomain, snapshot: &EngineLoadSnapshot) -> WorkerId {
    let ctx = DecodeSelectionContext::new().with_load_snapshot(snapshot);
    let proposal = DecodePowerOfTwoPolicy::new()
        .propose(domain, &ctx)
        .expect("decode policy must propose a worker");
    resolve_decode(domain, &proposal, INPUT_TOKENS, snapshot)
        .expect("decode decision must stay in its domain")
        .selected
        .id
        .clone()
}

fn bench_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("router_pressure/snapshot");
    group.throughput(Throughput::Elements(1));
    for &endpoint_count in ENDPOINT_COUNTS {
        let fixture = build_fixture(endpoint_count, 1, WorkerMode::Prefill);
        let name = format!("endpoints={endpoint_count},dp=1");
        record_allocations(&format!("snapshot/{name}"), || {
            capture_policy_snapshot(&fixture.loads)
        });
        group.bench_with_input(
            BenchmarkId::new("capture", &name),
            &endpoint_count,
            |b, _| b.iter(|| black_box(capture_policy_snapshot(&fixture.loads))),
        );
    }
    for &endpoint_count in ENDPOINT_COUNTS {
        let fixture = build_fixture(endpoint_count, 8, WorkerMode::Prefill);
        let name = format!("endpoints={endpoint_count},dp=8");
        record_allocations(&format!("snapshot/{name}"), || {
            capture_policy_snapshot(&fixture.loads)
        });
        group.bench_with_input(
            BenchmarkId::new("capture", &name),
            &endpoint_count,
            |b, _| b.iter(|| black_box(capture_policy_snapshot(&fixture.loads))),
        );
    }
    group.finish();
}

fn bench_prefill_p2(c: &mut Criterion) {
    let mut group = c.benchmark_group("router_pressure/prefill_p2");
    group.throughput(Throughput::Elements(1));
    for &endpoint_count in ENDPOINT_COUNTS {
        let fixture = build_fixture(endpoint_count, 1, WorkerMode::Prefill);
        let domain = CandidateDomain::global_prefill(&fixture.workers);
        let policy = PowerOfTwoChoicesPolicy::new();
        let policy_name = format!("policy_only/endpoints={endpoint_count}");
        record_allocations(&format!("prefill_p2/{policy_name}"), || {
            prefill_decision(&policy, &fixture, &domain, &fixture.snapshot, None)
        });
        group.bench_function(BenchmarkId::new("policy_only", endpoint_count), |b| {
            b.iter(|| {
                black_box(prefill_decision(
                    &policy,
                    &fixture,
                    &domain,
                    &fixture.snapshot,
                    None,
                ))
            })
        });

        let full_fixture = build_fixture(endpoint_count, 1, WorkerMode::Prefill);
        let full_policy = PowerOfTwoChoicesPolicy::new();
        let full_name = format!("request_path/endpoints={endpoint_count}");
        record_allocations(&format!("prefill_p2/{full_name}"), || {
            let snapshot = capture_policy_snapshot(&full_fixture.loads);
            let domain = CandidateDomain::global_prefill(&full_fixture.workers);
            prefill_decision(&full_policy, &full_fixture, &domain, &snapshot, None)
        });
        group.bench_function(BenchmarkId::new("request_path", endpoint_count), |b| {
            b.iter(|| {
                let snapshot = capture_policy_snapshot(&full_fixture.loads);
                let domain = CandidateDomain::global_prefill(&full_fixture.workers);
                black_box(prefill_decision(
                    &full_policy,
                    &full_fixture,
                    &domain,
                    &snapshot,
                    None,
                ))
            })
        });
    }
    group.finish();
}

fn bench_session(c: &mut Criterion) {
    let mut group = c.benchmark_group("router_pressure/session");
    group.throughput(Throughput::Elements(1));
    for &endpoint_count in ENDPOINT_COUNTS {
        let fixture = build_fixture(endpoint_count, 1, WorkerMode::Prefill);
        let domain = CandidateDomain::global_prefill(&fixture.workers);
        let policy = initialized_session_policy(&fixture);
        let policy_name = format!("policy_only/endpoints={endpoint_count}");
        record_allocations(&format!("session/{policy_name}"), || {
            prefill_decision(
                &policy,
                &fixture,
                &domain,
                &fixture.snapshot,
                Some("stable-session"),
            )
        });
        group.bench_function(BenchmarkId::new("policy_only", endpoint_count), |b| {
            b.iter(|| {
                black_box(prefill_decision(
                    &policy,
                    &fixture,
                    &domain,
                    &fixture.snapshot,
                    Some("stable-session"),
                ))
            })
        });

        let full_fixture = build_fixture(endpoint_count, 1, WorkerMode::Prefill);
        let full_policy = initialized_session_policy(&full_fixture);
        let full_name = format!("request_path/endpoints={endpoint_count}");
        record_allocations(&format!("session/{full_name}"), || {
            let snapshot = capture_policy_snapshot(&full_fixture.loads);
            let domain = CandidateDomain::global_prefill(&full_fixture.workers);
            prefill_decision(
                &full_policy,
                &full_fixture,
                &domain,
                &snapshot,
                Some("stable-session"),
            )
        });
        group.bench_function(BenchmarkId::new("request_path", endpoint_count), |b| {
            b.iter(|| {
                let snapshot = capture_policy_snapshot(&full_fixture.loads);
                let domain = CandidateDomain::global_prefill(&full_fixture.workers);
                black_box(prefill_decision(
                    &full_policy,
                    &full_fixture,
                    &domain,
                    &snapshot,
                    Some("stable-session"),
                ))
            })
        });
    }
    group.finish();
}

fn bench_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("router_pressure/cache");
    group.throughput(Throughput::Elements(1));
    for &endpoint_count in ENDPOINT_COUNTS {
        for &top_k in CACHE_TOP_K.iter().filter(|&&top_k| top_k <= endpoint_count) {
            let fixture = build_fixture(endpoint_count, 1, WorkerMode::Prefill);
            let signal = cache_signal(&fixture.workers);
            let config = AffinityConfig {
                cache_affinity_min_matched_tokens: Some(0),
                cache_candidate_min_workers: top_k,
                cache_candidate_ratio: 0.0,
                cache_candidate_max_workers: top_k,
                ..Default::default()
            };
            let policy = CacheAwarePolicy::new(config.clone());
            let policy_name = format!("policy_only/endpoints={endpoint_count},top_k={top_k}");
            record_allocations(&format!("cache/{policy_name}"), || {
                cache_decision(&policy, &fixture, &signal, &fixture.snapshot)
            });
            group.bench_function(
                BenchmarkId::new(
                    "policy_only",
                    format!("endpoints={endpoint_count},top_k={top_k}"),
                ),
                |b| {
                    b.iter(|| {
                        black_box(cache_decision(
                            &policy,
                            &fixture,
                            &signal,
                            &fixture.snapshot,
                        ))
                    })
                },
            );

            let full_fixture = build_fixture(endpoint_count, 1, WorkerMode::Prefill);
            let full_signal = cache_signal(&full_fixture.workers);
            let full_policy = CacheAwarePolicy::new(config);
            let full_name = format!("request_path/endpoints={endpoint_count},top_k={top_k}");
            record_allocations(&format!("cache/{full_name}"), || {
                let snapshot = capture_policy_snapshot(&full_fixture.loads);
                cache_decision(&full_policy, &full_fixture, &full_signal, &snapshot)
            });
            group.bench_function(
                BenchmarkId::new(
                    "request_path",
                    format!("endpoints={endpoint_count},top_k={top_k}"),
                ),
                |b| {
                    b.iter(|| {
                        let snapshot = capture_policy_snapshot(&full_fixture.loads);
                        black_box(cache_decision(
                            &full_policy,
                            &full_fixture,
                            &full_signal,
                            &snapshot,
                        ))
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("router_pressure/decode");
    group.throughput(Throughput::Elements(1));
    for &endpoint_count in ENDPOINT_COUNTS {
        let fixture = build_fixture(endpoint_count, 1, WorkerMode::Decode);
        let domain = CandidateDomain::global_decode(&fixture.workers);
        let policy_name = format!("policy_only/endpoints={endpoint_count}");
        record_allocations(&format!("decode/{policy_name}"), || {
            decode_decision(&domain, &fixture.snapshot)
        });
        group.bench_function(BenchmarkId::new("policy_only", endpoint_count), |b| {
            b.iter(|| black_box(decode_decision(&domain, &fixture.snapshot)))
        });

        let full_fixture = build_fixture(endpoint_count, 1, WorkerMode::Decode);
        let full_name = format!("request_path/endpoints={endpoint_count}");
        record_allocations(&format!("decode/{full_name}"), || {
            let snapshot = capture_policy_snapshot(&full_fixture.loads);
            let domain = CandidateDomain::global_decode(&full_fixture.workers);
            decode_decision(&domain, &snapshot)
        });
        group.bench_function(BenchmarkId::new("request_path", endpoint_count), |b| {
            b.iter(|| {
                let snapshot = capture_policy_snapshot(&full_fixture.loads);
                let domain = CandidateDomain::global_decode(&full_fixture.workers);
                black_box(decode_decision(&domain, &snapshot))
            })
        });
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(1))
        .sample_size(30)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_snapshot, bench_prefill_p2, bench_session, bench_cache, bench_decode
}
criterion_main!(benches);
