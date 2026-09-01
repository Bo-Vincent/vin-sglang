// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! RTP-LLM Shortest-TTFT 的独立策略合同。

use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use sgl_router::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
use sgl_router::policies::kv_events::{
    compute_block_hashes, BlockSizeOracle, HashTree, KvWorkerId,
};
use sgl_router::policies::shortest_ttft::{
    EngineLoadMonitor, EngineLoadTable, LoadEndpointConfig, LoadStat, ShortestTtftPolicy,
};
use sgl_router::policies::{ExternalPrefixSignal, Policy, SelectionContext};
use sgl_router::workers::Worker;
use zeromq::SocketSend;

use super::zmq_helpers::make_pub_bound;

fn worker(id: &str) -> Arc<Worker> {
    Arc::new(Worker::new(WorkerSpec {
        id: WorkerId(id.into()),
        url: format!("http://{id}:30000"),
        mode: WorkerMode::Plain,
        model_ids: vec![ModelId("tiny".into())],
        bootstrap_port: None,
    }))
}

fn load(running: u64, waiting: u64) -> LoadStat {
    LoadStat {
        num_running_reqs: running,
        num_waiting_reqs: waiting,
        num_tokens: 0,
        max_total_num_tokens: 0,
    }
}

fn load_message(seq: i64, running: u64, waiting: u64) -> zeromq::ZmqMessage {
    let mut payload = Vec::new();
    rmp::encode::write_array_len(&mut payload, 6).unwrap();
    rmp::encode::write_str(&mut payload, "LoadStat").unwrap();
    rmp::encode::write_uint(&mut payload, running).unwrap();
    rmp::encode::write_uint(&mut payload, waiting).unwrap();
    rmp::encode::write_uint(&mut payload, 0).unwrap();
    rmp::encode::write_uint(&mut payload, 0).unwrap();
    rmp::encode::write_nil(&mut payload).unwrap();

    let mut message = zeromq::ZmqMessage::from(Bytes::from_static(b"load"));
    message.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
    message.push_back(Bytes::from(payload));
    message
}

#[test]
fn engine_queue_can_outweigh_a_better_prefix_match() {
    let model = ModelId("tiny".into());
    let worker_a = worker("worker-a");
    let worker_b = worker("worker-b");
    let workers = vec![Arc::clone(&worker_a), Arc::clone(&worker_b)];
    let tokens = [11_u32, 12, 13, 14, 15, 16, 17, 18];

    let block_size = BlockSizeOracle::new();
    block_size.try_set(2).unwrap();
    let tree = Arc::new(HashTree::new());
    let hashes = compute_block_hashes(&tokens, 2);
    tree.insert(&KvWorkerId::new(worker_a.url.clone(), 0), None, &hashes);

    let engine_load = EngineLoadTable::with_freshness(Duration::from_secs(1));
    engine_load.mark_expected_rank(&worker_a.url, 0);
    engine_load.mark_expected_rank(&worker_b.url, 0);
    let now = Instant::now();
    engine_load.set(&worker_a.url, 0, load(3, 6), now);
    engine_load.set(&worker_b.url, 0, load(0, 0), now);

    let policy = ShortestTtftPolicy::new(tree, block_size, engine_load);
    let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&tokens));

    assert_eq!(
        policy.select(&workers, &ctx).unwrap().url,
        worker_b.url,
        "RTP-LLM 的 TTFT 估算必须把 engine 队列压力叠加到缓存命中收益上"
    );
}

#[test]
fn external_indexer_match_overrides_the_local_tree() {
    let model = ModelId("tiny".into());
    let worker_a = worker("worker-a");
    let worker_b = worker("worker-b");
    let workers = vec![Arc::clone(&worker_a), Arc::clone(&worker_b)];
    let tokens = [11_u32, 12, 13, 14, 15, 16, 17, 18];

    let block_size = BlockSizeOracle::new();
    block_size.try_set(2).unwrap();
    let tree = Arc::new(HashTree::new());
    let hashes = compute_block_hashes(&tokens, 2);
    tree.insert(&KvWorkerId::new(worker_a.url.clone(), 0), None, &hashes);

    let signal = ExternalPrefixSignal {
        outcome: sgl_kv_indexer::PrefixOutcome::Matched {
            matches: vec![sgl_kv_indexer::PrefixMatch {
                address: worker_b.url.clone(),
                matched_prefix_blocks: 4,
                worker_id: "worker-b".into(),
            }],
            best_prefix_blocks: 4,
        },
        query_blocks: 4,
    };
    let policy = ShortestTtftPolicy::new(tree, block_size, EngineLoadTable::new());
    let ctx = SelectionContext::new(&model, None)
        .with_request_tokens(Some(&tokens))
        .with_external_prefix(Some(&signal));

    assert_eq!(
        policy.select(&workers, &ctx).unwrap().url,
        worker_b.url,
        "V4 Indexer signal 存在时，Shortest-TTFT 必须忽略相反的本地 HashTree 命中"
    );
}

#[test]
fn external_indexer_empty_result_does_not_fall_back_to_the_local_tree() {
    let model = ModelId("tiny".into());
    let worker_a = worker("worker-a");
    let worker_b = worker("worker-b");
    let workers = vec![Arc::clone(&worker_a), Arc::clone(&worker_b)];
    let tokens = [11_u32, 12, 13, 14, 15, 16, 17, 18];

    let block_size = BlockSizeOracle::new();
    block_size.try_set(2).unwrap();
    let tree = Arc::new(HashTree::new());
    let hashes = compute_block_hashes(&tokens, 2);
    tree.insert(&KvWorkerId::new(worker_a.url.clone(), 0), None, &hashes);

    let signal = ExternalPrefixSignal {
        outcome: sgl_kv_indexer::PrefixOutcome::Empty,
        query_blocks: 4,
    };
    let policy = ShortestTtftPolicy::new(tree, block_size, EngineLoadTable::new());
    let load_guard = worker_a.load_guard();
    let ctx = SelectionContext::new(&model, None)
        .with_request_tokens(Some(&tokens))
        .with_external_prefix(Some(&signal));

    assert_eq!(
        policy.select(&workers, &ctx).unwrap().url,
        worker_b.url,
        "authoritative 的 Empty 必须是零命中，不能回退至本地 HashTree"
    );
    drop(load_guard);
}

#[test]
fn partial_or_stale_dp_gauges_are_not_aggregated() {
    let table = EngineLoadTable::with_freshness(Duration::from_millis(10));
    let worker_url = "http://worker:30000";
    let now = Instant::now();
    table.mark_expected_rank(worker_url, 0);
    table.mark_expected_rank(worker_url, 1);
    table.set(worker_url, 0, load(2, 1), now);
    assert_eq!(
        table.queue_pressure(worker_url, now),
        None,
        "缺任一 DP rank 时不能把局部 gauge 当作完整 engine 负载"
    );

    table.set(
        worker_url,
        1,
        load(3, 4),
        now.checked_sub(Duration::from_secs(1)).unwrap(),
    );
    assert_eq!(
        table.queue_pressure(worker_url, now),
        None,
        "过期 gauge 必须整体降级，不能和新 rank 混合"
    );
}

#[test]
fn rtp_candidate_window_uses_last_selection_for_fairness() {
    let model = ModelId("tiny".into());
    let workers: Vec<_> = (0..10)
        .map(|index| worker(&format!("worker-{index}")))
        .collect();
    let policy = ShortestTtftPolicy::new(
        Arc::new(HashTree::new()),
        BlockSizeOracle::new(),
        EngineLoadTable::new(),
    );
    let ctx = SelectionContext::new(&model, None).with_request_tokens(Some(&[1_u32]));

    let selected: Vec<_> = (0..3)
        .map(|_| policy.select(&workers, &ctx).unwrap().url.clone())
        .collect();
    assert_eq!(
        selected,
        vec![
            "http://worker-0:30000",
            "http://worker-1:30000",
            "http://worker-2:30000",
        ],
        "RTP 的前 30% 候选窗内，TTFT 相近时必须优先选择最久未被调度的 worker"
    );
}

#[tokio::test]
async fn monitor_consumes_a_gauge_and_removes_worker_state() {
    let (mut publisher, port) = make_pub_bound().await;
    let table = EngineLoadTable::with_freshness(Duration::from_secs(1));
    let monitor = EngineLoadMonitor::new(Arc::clone(&table));
    let worker_url = "http://127.0.0.1:30000";
    monitor
        .add_worker(
            worker_url,
            LoadEndpointConfig {
                host: "127.0.0.1".into(),
                port_base: port,
                topic: "load".into(),
                dp_size: 1,
            },
        )
        .await;

    tokio::time::sleep(Duration::from_millis(150)).await;
    publisher
        .send(load_message(1, 3, 4))
        .await
        .expect("send LoadStat gauge");

    let deadline = Instant::now() + Duration::from_secs(3);
    while table.queue_pressure(worker_url, Instant::now()) != Some(7) {
        assert!(
            Instant::now() < deadline,
            "monitor did not receive LoadStat"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    monitor.remove_worker(worker_url).await;
    assert_eq!(
        table.queue_pressure(worker_url, Instant::now()),
        None,
        "removed workers must not retain stale monitor state"
    );
    publisher
        .send(load_message(2, 9, 9))
        .await
        .expect("send late gauge after worker removal");
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        table.queue_pressure(worker_url, Instant::now()),
        None,
        "已取消并 join 的 subscriber 不能让迟到 gauge 恢复已移除 worker 的状态"
    );
    monitor.shutdown().await;
}

#[tokio::test]
async fn monitor_accepts_a_publisher_sequence_reset() {
    let (mut publisher, port) = make_pub_bound().await;
    let table = EngineLoadTable::with_freshness(Duration::from_secs(1));
    let monitor = EngineLoadMonitor::new(Arc::clone(&table));
    let worker_url = "http://127.0.0.1:30001";
    monitor
        .add_worker(
            worker_url,
            LoadEndpointConfig {
                host: "127.0.0.1".into(),
                port_base: port,
                topic: "load".into(),
                dp_size: 1,
            },
        )
        .await;

    tokio::time::sleep(Duration::from_millis(150)).await;
    publisher
        .send(load_message(42, 5, 0))
        .await
        .expect("send pre-restart gauge");
    let deadline = Instant::now() + Duration::from_secs(3);
    while table.queue_pressure(worker_url, Instant::now()) != Some(5) {
        assert!(
            Instant::now() < deadline,
            "monitor did not receive first gauge"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // #34608 的 sequence 只给 KV event 的有序消费使用；load 是覆盖型 gauge。
    // 发布者重启后序号可以回到 0，Router 必须接收新的低序号样本。
    publisher
        .send(load_message(0, 1, 2))
        .await
        .expect("send post-restart gauge");
    let deadline = Instant::now() + Duration::from_secs(3);
    while table.queue_pressure(worker_url, Instant::now()) != Some(3) {
        assert!(
            Instant::now() < deadline,
            "monitor ignored the publisher sequence reset"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    monitor.shutdown().await;
}
