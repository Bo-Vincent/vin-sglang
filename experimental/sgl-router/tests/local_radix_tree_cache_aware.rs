// SPDX-License-Identifier: Apache-2.0

use std::sync::{atomic::Ordering, Arc};

use sgl_router::config::AffinityConfig;
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::cache_aware::CacheAwarePolicy;
use sgl_router::policies::kv_events::{
    compute_block_hashes, compute_block_hashes_bigram, BlockSizeOracle, HashTree, KvWorkerId,
};
use sgl_router::policies::{Policy, PrefillProposal, ProposalKind, SelectionContext};
use sgl_router::workers::Worker;

fn worker(id: &str) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: id.into(),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    }))
}

#[test]
fn local_radix_tree_produces_per_worker_cache_candidates() {
    let tokens = [11u32, 12, 13, 14];
    let hashes = compute_block_hashes(&tokens, 1);
    let tree = Arc::new(HashTree::new());
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();
    let deep = worker("http://deep");
    let shallow = worker("http://shallow");
    tree.insert(&KvWorkerId::new(deep.url.clone(), 0), None, &hashes);
    tree.insert(&KvWorkerId::new(shallow.url.clone(), 0), None, &hashes[..2]);

    tree.insert(&KvWorkerId::new(deep.url.clone(), 1), None, &hashes[..3]);
    let policy = CacheAwarePolicy::new(
        AffinityConfig {
            cache_affinity_min_matched_tokens: Some(0),
            cache_candidate_min_workers: 2,
            cache_candidate_ratio: 1.0,
            cache_candidate_max_workers: 2,
            ..Default::default()
        },
        Arc::clone(&tree),
        oracle,
    );
    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None)
        .with_input_tokens(tokens.len() as u64)
        .with_request_tokens(Some(&tokens));
    let PrefillProposal::CacheCandidates(proposal) = policy
        .propose_prefill(&[Arc::clone(&deep), Arc::clone(&shallow)], &ctx)
        .expect("local tree hit must propose cache candidates")
    else {
        panic!("expected cache candidates")
    };

    assert_eq!(proposal.candidates.len(), 2);
    assert_eq!(proposal.candidates[0].worker.url, deep.url);
    assert_eq!(proposal.candidates[0].matched_prefix_tokens, 4);
    assert_eq!(proposal.candidates[1].worker.url, shallow.url);
    assert_eq!(proposal.candidates[1].matched_prefix_tokens, 2);
}

#[test]
fn local_radix_tree_candidates_still_run_admission_and_pressure_guard() {
    use std::collections::HashMap;
    use std::time::Instant;

    use sgl_router::policies::admission::resolve_cache_candidates;
    use sgl_router::policies::engine_load::{EngineLoadSnapshot, NativeCacheWorkerLoad};

    let tokens = [21u32, 22, 23, 24];
    let hashes = compute_block_hashes(&tokens, 1);
    let tree = Arc::new(HashTree::new());
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();
    let hot = worker("http://hot");
    let cool = worker("http://cool");
    tree.insert(&KvWorkerId::new(hot.url.clone(), 0), None, &hashes);
    tree.insert(&KvWorkerId::new(cool.url.clone(), 0), None, &hashes[..3]);

    let policy = CacheAwarePolicy::new(
        AffinityConfig {
            cache_affinity_min_matched_tokens: Some(0),
            cache_candidate_min_workers: 2,
            cache_candidate_ratio: 1.0,
            cache_candidate_max_workers: 2,
            cache_switch_margin_tokens: 1,
            pressure_abs_threshold_tokens: 1,
            pressure_rel_threshold: 1.0,
            ..Default::default()
        },
        Arc::clone(&tree),
        oracle,
    );
    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None)
        .with_input_tokens(tokens.len() as u64)
        .with_request_tokens(Some(&tokens));
    let PrefillProposal::CacheCandidates(proposal) = policy
        .propose_prefill(&[Arc::clone(&hot), Arc::clone(&cool)], &ctx)
        .expect("local tree hit must propose cache candidates")
    else {
        panic!("expected cache candidates")
    };

    let captured_at = Instant::now();
    let snapshot = EngineLoadSnapshot::from_native_cache_workers(
        7,
        HashMap::from([
            (
                hot.url.clone(),
                NativeCacheWorkerLoad {
                    num_running_reqs: 1,
                    num_waiting_reqs: 1,
                    num_waiting_uncached_tokens: 100,
                    num_used_tokens: 0,
                    num_total_tokens: 0,
                    max_total_num_tokens: 1_000,
                    max_running_requests: 64,
                    prefill_throughput_tokens_per_s: None,
                    estimated_prefill_queue_ms: None,
                    captured_at,
                },
            ),
            (
                cool.url.clone(),
                NativeCacheWorkerLoad {
                    num_running_reqs: 1,
                    num_waiting_reqs: 0,
                    num_waiting_uncached_tokens: 0,
                    num_used_tokens: 0,
                    num_total_tokens: 0,
                    max_total_num_tokens: 1_000,
                    max_running_requests: 64,
                    prefill_throughput_tokens_per_s: None,
                    estimated_prefill_queue_ms: None,
                    captured_at,
                },
            ),
        ]),
    );
    let resolution = resolve_cache_candidates(&proposal, tokens.len() as u64, &snapshot);

    assert_eq!(
        resolution.decision.expect("admitted decision").selected.url,
        cool.url
    );
    assert_eq!(resolution.admission_evaluated_candidates, 2);
    assert_eq!(resolution.admission_rejected_candidates, 0);
    assert_eq!(resolution.pressure_guard_compared_pairs, 1);
    assert_eq!(resolution.pressure_guard_overrides, 1);
}

fn local_policy(config: AffinityConfig, tree: Arc<HashTree>) -> CacheAwarePolicy {
    let oracle = BlockSizeOracle::new();
    oracle.try_set(1).unwrap();
    CacheAwarePolicy::new(config, tree, oracle)
}

#[test]
fn local_radix_tree_candidate_bound_keeps_the_best_k() {
    let tokens: Vec<u32> = (1..=64).collect();
    let hashes = compute_block_hashes(&tokens, 1);
    let tree = Arc::new(HashTree::new());
    let workers: Vec<_> = (0..64)
        .map(|index| worker(&format!("http://w{index:02}")))
        .collect();
    for (index, worker) in workers.iter().enumerate() {
        tree.insert(
            &KvWorkerId::new(worker.url.clone(), 0),
            None,
            &hashes[..=index],
        );
    }

    let policy = local_policy(
        AffinityConfig {
            cache_affinity_min_matched_tokens: Some(0),
            cache_candidate_min_workers: 4,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 4,
            ..Default::default()
        },
        Arc::clone(&tree),
    );
    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None)
        .with_input_tokens(64_000)
        .with_request_tokens(Some(&tokens));
    let PrefillProposal::CacheCandidates(proposal) = policy
        .propose_prefill(&workers, &ctx)
        .expect("local tree hits must produce candidates")
    else {
        panic!("expected cache candidates")
    };

    assert_eq!(proposal.candidates.len(), 4);
    assert_eq!(
        proposal
            .candidates
            .iter()
            .map(|candidate| candidate.matched_prefix_tokens)
            .collect::<Vec<_>>(),
        vec![64_000, 63_000, 62_000, 61_000]
    );
}

#[test]
fn local_radix_tree_equal_hits_use_captured_local_load_then_worker_id() {
    let tokens = [31u32, 32, 33, 34];
    let hashes = compute_block_hashes(&tokens, 1);
    let tree = Arc::new(HashTree::new());
    let workers: Vec<_> = (0..8)
        .map(|index| {
            let worker = worker(&format!("http://w{index}"));
            worker.active_requests.store(8 - index, Ordering::Relaxed);
            worker
        })
        .collect();
    for worker in &workers {
        tree.insert(&KvWorkerId::new(worker.url.clone(), 0), None, &hashes);
    }

    let policy = local_policy(
        AffinityConfig {
            cache_affinity_min_matched_tokens: Some(0),
            cache_candidate_min_workers: 2,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 2,
            ..Default::default()
        },
        Arc::clone(&tree),
    );
    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None)
        .with_input_tokens(tokens.len() as u64)
        .with_request_tokens(Some(&tokens));
    let PrefillProposal::CacheCandidates(proposal) = policy
        .propose_prefill(&workers, &ctx)
        .expect("equal local tree hits must retain candidates")
    else {
        panic!("expected cache candidates")
    };

    assert_eq!(
        proposal
            .candidates
            .iter()
            .map(|candidate| candidate.worker.url.as_str())
            .collect::<Vec<_>>(),
        vec!["http://w7", "http://w6"]
    );
}

#[test]
fn local_radix_tree_cache_gates_keep_and_semantics() {
    let tokens = [41u32, 42, 43, 44, 45, 46, 47, 48];
    let hashes = compute_block_hashes(&tokens, 1);
    let tree = Arc::new(HashTree::new());
    let half = worker("http://half");
    let below_ratio = worker("http://below-ratio");
    tree.insert(&KvWorkerId::new(half.url.clone(), 0), None, &hashes[..4]);
    tree.insert(
        &KvWorkerId::new(below_ratio.url.clone(), 0),
        None,
        &hashes[..3],
    );

    let policy = local_policy(
        AffinityConfig {
            cache_affinity_min_matched_tokens: Some(30),
            cache_affinity_min_match_ratio: Some(0.5),
            cache_candidate_min_workers: 2,
            cache_candidate_max_workers: 2,
            ..Default::default()
        },
        Arc::clone(&tree),
    );
    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None)
        .with_input_tokens(80)
        .with_request_tokens(Some(&tokens));
    let PrefillProposal::CacheCandidates(proposal) = policy
        .propose_prefill(&[Arc::clone(&half), Arc::clone(&below_ratio)], &ctx)
        .expect("one local hit satisfies both lower bounds")
    else {
        panic!("expected cache candidates")
    };

    assert_eq!(proposal.candidates.len(), 1);
    assert_eq!(proposal.candidates[0].worker.url, half.url);
}

#[test]
fn local_radix_tree_default_gate_rejects_a_weak_prefix() {
    let tokens = [51u32, 52, 53, 54, 55, 56, 57, 58];
    let hashes = compute_block_hashes(&tokens, 1);
    let tree = Arc::new(HashTree::new());
    let weak = worker("http://weak");
    tree.insert(&KvWorkerId::new(weak.url.clone(), 0), None, &hashes[..3]);

    let policy = local_policy(AffinityConfig::default(), Arc::clone(&tree));
    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None)
        .with_input_tokens(80)
        .with_request_tokens(Some(&tokens));
    let proposal = policy
        .propose_prefill(&[weak], &ctx)
        .expect("a weak local hit must degrade to P2");

    assert!(matches!(proposal, PrefillProposal::Pair(_)));
}

#[test]
fn local_radix_tree_default_gate_accepts_a_long_prefix() {
    let tokens: Vec<u32> = (1..=4_125).collect();
    let hashes = compute_block_hashes(&tokens, 1);
    let tree = Arc::new(HashTree::new());
    let holder = worker("http://holder");
    tree.insert(
        &KvWorkerId::new(holder.url.clone(), 0),
        None,
        &hashes[..2_048],
    );

    let policy = local_policy(AffinityConfig::default(), Arc::clone(&tree));
    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None)
        .with_input_tokens(tokens.len() as u64)
        .with_request_tokens(Some(&tokens));
    let PrefillProposal::CacheCandidates(proposal) = policy
        .propose_prefill(&[Arc::clone(&holder)], &ctx)
        .expect("the local long-prefix hit must pass the default lower bound")
    else {
        panic!("expected cache candidates")
    };

    assert_eq!(proposal.candidates[0].worker.url, holder.url);
    assert_eq!(proposal.candidates[0].matched_prefix_tokens, 2_048);
    assert_eq!(proposal.candidates[0].uncached_tokens, 2_077);
}

#[test]
fn local_radix_tree_miss_falls_back_to_power_of_two() {
    let tokens = [61u32, 62, 63, 64];
    let tree = Arc::new(HashTree::new());
    let policy = local_policy(AffinityConfig::default(), tree);
    let model = ModelId("tiny".into());
    let first = worker("http://first");
    let second = worker("http://second");
    let ctx = SelectionContext::new(&model, None)
        .with_input_tokens(tokens.len() as u64)
        .with_request_tokens(Some(&tokens));
    let proposal = policy
        .propose(&[first, second], &ctx)
        .expect("a local tree miss must fall back to P2");

    assert_eq!(proposal.kind, ProposalKind::PowerOfTwo);
    assert!(proposal.backup.is_some());
}

#[test]
fn local_radix_tree_uses_the_worker_bigram_hash_mode() {
    let tokens = [71u32, 72, 73, 74, 75, 76];
    let hashes = compute_block_hashes_bigram(&tokens, 2);
    let tree = Arc::new(HashTree::new());
    let holder = worker("http://bigram");
    tree.insert(&KvWorkerId::new(holder.url.clone(), 0), None, &hashes);

    let oracle = BlockSizeOracle::new();
    oracle.try_set(2).unwrap();
    oracle.set_bigram(true);
    let policy = CacheAwarePolicy::new(
        AffinityConfig {
            cache_affinity_min_matched_tokens: Some(0),
            cache_candidate_min_workers: 1,
            cache_candidate_ratio: 1.0,
            cache_candidate_max_workers: 1,
            ..Default::default()
        },
        Arc::clone(&tree),
        oracle,
    );
    let model = ModelId("tiny".into());
    let ctx = SelectionContext::new(&model, None)
        .with_input_tokens(tokens.len() as u64)
        .with_request_tokens(Some(&tokens));
    let PrefillProposal::CacheCandidates(proposal) = policy
        .propose_prefill(&[Arc::clone(&holder)], &ctx)
        .expect("bigram local-tree hit must produce a candidate")
    else {
        panic!("expected cache candidates")
    };

    assert_eq!(proposal.candidates.len(), 1);
    assert_eq!(proposal.candidates[0].worker.url, holder.url);
    assert_eq!(
        proposal.candidates[0].matched_prefix_tokens,
        tokens.len() as u64
    );
}
