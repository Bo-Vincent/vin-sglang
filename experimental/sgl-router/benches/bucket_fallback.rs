// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Degraded-path admission and bucket-membership microbenchmarks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use sgl_router::config::{BucketConfig, BucketSpec, BucketStage, SloBucketPolicy};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::admission::{resolve_decode, resolve_prefill, CandidateDomain};
use sgl_router::policies::buckets::{BucketRequest, BucketSelector};
use sgl_router::policies::engine_load::{EngineLoadSnapshot, EngineWorkerLoad};
use sgl_router::policies::SelectionProposal;
use sgl_router::workers::Worker;
use std::{collections::HashMap, sync::Arc, time::Instant};

const FLEET: usize = 64;

fn worker(index: usize) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(format!("worker-{index:03}")),
        url: format!("http://127.0.0.1:{}", 30_000 + index),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("bucket-bench".into())],
        bootstrap_port: None,
    }))
}

fn workers(count: usize) -> Vec<Arc<Worker>> {
    (0..count).map(worker).collect()
}

fn snapshot(workers: &[Arc<Worker>]) -> EngineLoadSnapshot {
    let captured_at = Instant::now();
    EngineLoadSnapshot::from_workers(
        1,
        workers
            .iter()
            .enumerate()
            .map(|(index, worker)| {
                (
                    worker.url.clone(),
                    EngineWorkerLoad {
                        num_running_reqs: if index == 0 { 64 } else { index as u64 % 8 },
                        num_waiting_reqs: index as u64 % 16,
                        num_tokens: 1_024 + index as u64,
                        max_total_num_tokens: 262_144,
                        captured_at,
                    },
                )
            })
            .collect::<HashMap<_, _>>(),
    )
}

fn bench_admission_fallback(c: &mut Criterion) {
    let workers = workers(FLEET);
    let snapshot = snapshot(&workers);
    let proposal = SelectionProposal::primary(Arc::clone(&workers[0]));
    let prefill = CandidateDomain::global_prefill(&workers);
    let decode = CandidateDomain::global_decode(&workers);

    c.bench_function("bucket_fallback/prefill/64", |b| {
        let range = prefill.prefill_range().expect("prefill domain");
        b.iter(|| black_box(resolve_prefill(&range, &proposal, 4_096, &snapshot)))
    });
    c.bench_function("bucket_fallback/decode/64", |b| {
        b.iter(|| black_box(resolve_decode(&decode, &proposal, 4_096, &snapshot)))
    });
}

fn bucket_config(member_count: usize) -> BucketConfig {
    BucketConfig {
        buckets: vec![BucketSpec {
            id: "bucket".into(),
            stage: BucketStage::Prefill,
            rank: 0,
            worker_ids: (0..member_count)
                .map(|index| format!("worker-{index:03}"))
                .collect(),
            min_extend_tokens: None,
            max_extend_tokens: None,
            min_sequence_tokens: None,
            max_sequence_tokens: None,
            max_context_tokens: None,
            ttft_p95_at_capacity_ms: None,
            tps_p05_at_capacity: None,
            max_pending_prefill_tokens: None,
        }],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }
}

fn bench_membership_index(c: &mut Criterion) {
    let workers = workers(256);
    let request = BucketRequest {
        input_tokens: 4_096,
        expected_peak_sequence_tokens: None,
        ttft_slo_ms: None,
        tps_slo: None,
    };
    let mut group = c.benchmark_group("bucket_membership");
    // Compare each cardinality against the same case on the base revision.
    // Four stays on the scan path; five and above use the precomputed index.
    for member_count in [4, 5, 8, 9] {
        let selector = BucketSelector::new(Some(bucket_config(member_count)));
        group.bench_function(BenchmarkId::from_parameter(member_count), |b| {
            b.iter(|| black_box(selector.prefill_domains(&workers, request)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_admission_fallback, bench_membership_index);
criterion_main!(benches);
