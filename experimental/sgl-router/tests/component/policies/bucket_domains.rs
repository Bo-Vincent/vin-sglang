// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! 静态 Bucket 仅决定候选域和 fallback 顺序；不在这里做 worker 评分或动态
//! admission。最终选择仍由 P/D policy、Admission 与 Guard 完成。

use sgl_router::config::{BucketConfig, BucketSpec, BucketStage, SloBucketPolicy};
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::buckets::{BucketRequest, BucketSelector};
use sgl_router::workers::Worker;
use std::sync::Arc;

fn worker(id: &str, mode: WorkerMode) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}:30000"),
        mode,
        model_ids: vec![ModelId("m".into())],
        bootstrap_port: None,
    }))
}

fn bucket(id: &str, stage: BucketStage, rank: u32, worker_ids: &[&str]) -> BucketSpec {
    BucketSpec {
        id: id.into(),
        stage,
        rank,
        worker_ids: worker_ids.iter().map(|id| (*id).into()).collect(),
        min_input_tokens: None,
        max_input_tokens: None,
        min_sequence_tokens: None,
        max_sequence_tokens: None,
        max_context_tokens: None,
        ttft_p95_at_capacity_ms: None,
        tps_p05_at_capacity: None,
        max_pending_prefill_tokens: None,
    }
}

#[test]
fn prefill_slo_first_tries_eligible_buckets_by_rank_before_degrading() {
    let fast = worker("fast", WorkerMode::Prefill);
    let cheap = worker("cheap", WorkerMode::Prefill);
    let mut cheap_bucket = bucket("cheap", BucketStage::Prefill, 10, &["cheap"]);
    cheap_bucket.ttft_p95_at_capacity_ms = Some(300);
    let mut fast_bucket = bucket("fast", BucketStage::Prefill, 20, &["fast"]);
    fast_bucket.ttft_p95_at_capacity_ms = Some(80);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![cheap_bucket, fast_bucket],
        ttft_slo_policy: SloBucketPolicy::SloFirst,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }));

    let domains = selector.prefill_domains(
        &[cheap, fast],
        BucketRequest {
            input_tokens: 256,
            expected_peak_sequence_tokens: None,
            ttft_slo_ms: Some(100),
            tps_slo: None,
        },
    );

    assert_eq!(
        domains
            .iter()
            .map(|domain| domain.id.as_str())
            .collect::<Vec<_>>(),
        ["fast", "cheap"],
        "eligible buckets come first; non-eligible buckets are the explicit SLO-degraded fallback"
    );
}

#[test]
fn global_affinity_domain_can_cross_target_prompt_length_but_not_hard_ttft() {
    let short = worker("short", WorkerMode::Prefill);
    let long = worker("long", WorkerMode::Prefill);
    let mut short_bucket = bucket("p-short", BucketStage::Prefill, 10, &["short"]);
    short_bucket.max_input_tokens = Some(64);
    short_bucket.max_context_tokens = Some(4_096);
    short_bucket.ttft_p95_at_capacity_ms = Some(80);
    let mut long_bucket = bucket("p-long", BucketStage::Prefill, 20, &["long"]);
    long_bucket.min_input_tokens = Some(65);
    long_bucket.max_context_tokens = Some(4_096);
    long_bucket.ttft_p95_at_capacity_ms = Some(300);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![short_bucket, long_bucket],
        ttft_slo_policy: SloBucketPolicy::SloFirst,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }));
    let workers = vec![Arc::clone(&short), Arc::clone(&long)];
    let request = BucketRequest {
        input_tokens: 256,
        expected_peak_sequence_tokens: None,
        ttft_slo_ms: Some(100),
        tps_slo: None,
    };

    assert_eq!(
        selector
            .prefill_domains(&workers, request)
            .iter()
            .map(|domain| domain.id.as_str())
            .collect::<Vec<_>>(),
        ["p-long"],
        "normal target selection follows prompt-length compatibility"
    );
    assert_eq!(
        selector
            .prefill_affinity_domain(&workers, &short, request)
            .expect("a cross-Bucket primary may use its own runtime/SLO profile")
            .id,
        "p-short"
    );
    assert!(
        selector
            .prefill_affinity_domain(&workers, &long, request)
            .is_none(),
        "an affinity primary whose own Hard TTFT profile misses the request SLO is rejected"
    );
}

#[test]
fn decode_bucket_uses_peak_sequence_length_then_tps_profile_and_rank() {
    let short = worker("short", WorkerMode::Decode);
    let long = worker("long", WorkerMode::Decode);
    let mut short_bucket = bucket("short", BucketStage::Decode, 10, &["short"]);
    short_bucket.max_sequence_tokens = Some(1_024);
    short_bucket.tps_p05_at_capacity = Some(80.0);
    let mut long_bucket = bucket("long", BucketStage::Decode, 20, &["long"]);
    long_bucket.max_sequence_tokens = Some(8_192);
    long_bucket.tps_p05_at_capacity = Some(40.0);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![short_bucket, long_bucket],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::SloFirst,
    }));

    let domains = selector.decode_domains(
        &[short, long],
        BucketRequest {
            input_tokens: 256,
            expected_peak_sequence_tokens: Some(900),
            ttft_slo_ms: None,
            tps_slo: Some(60.0),
        },
    );

    assert_eq!(domains.len(), 2);
    assert_eq!(domains[0].id, "short");
    assert_eq!(domains[1].id, "long");
}

#[test]
fn missing_bucket_configuration_keeps_the_global_domain() {
    let p = worker("p", WorkerMode::Prefill);
    let d = worker("d", WorkerMode::Decode);
    let selector = BucketSelector::new(None);
    let facts = BucketRequest {
        input_tokens: 128,
        expected_peak_sequence_tokens: Some(512),
        ttft_slo_ms: Some(100),
        tps_slo: Some(20.0),
    };

    let prefill = selector.prefill_domains(&[p], facts);
    let decode = selector.decode_domains(&[d], facts);

    assert_eq!(prefill.len(), 1);
    assert_eq!(prefill[0].id, "global");
    assert_eq!(decode.len(), 1);
    assert_eq!(decode[0].id, "global");
}

#[test]
fn decode_catch_all_still_rejects_input_beyond_runtime_context() {
    let d = worker("d", WorkerMode::Decode);
    let mut catch_all = bucket("d-catch-all", BucketStage::Decode, 10, &["d"]);
    catch_all.max_context_tokens = Some(1_024);
    let selector = BucketSelector::new(Some(BucketConfig {
        buckets: vec![catch_all],
        ttft_slo_policy: SloBucketPolicy::Disabled,
        tps_slo_policy: SloBucketPolicy::Disabled,
    }));

    let domains = selector.decode_domains(
        &[d],
        BucketRequest {
            input_tokens: 2_048,
            expected_peak_sequence_tokens: None,
            ttft_slo_ms: None,
            tps_slo: None,
        },
    );

    assert!(
        domains.is_empty(),
        "an unknown output budget does not erase the known input context requirement"
    );
}
