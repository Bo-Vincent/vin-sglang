// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
    time::Duration,
};

use serde::Deserialize;
use tokio::{sync::oneshot, time::Instant};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightedQueueConfig {
    pub max_inflight: usize,
    #[serde(default = "default_requests")]
    pub max_queued_requests: usize,
    #[serde(default = "default_bytes")]
    pub max_queued_body_bytes: usize,
    #[serde(default = "default_timeout")]
    pub queue_timeout_ms: u64,
}
fn default_requests() -> usize {
    1024
}
fn default_bytes() -> usize {
    67_108_864
}
fn default_timeout() -> u64 {
    30_000
}

impl WeightedQueueConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_inflight == 0
            || self.max_queued_requests == 0
            || self.max_queued_body_bytes == 0
            || self.queue_timeout_ms == 0
        {
            return Err("weighted queue limits and timeout must be positive".into());
        }
        if Instant::now()
            .checked_add(Duration::from_millis(self.queue_timeout_ms))
            .is_none()
        {
            return Err("weighted queue timeout exceeds supported duration".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueError {
    Full,
    Timeout,
    Closed,
    InvalidWeight,
}
impl fmt::Display for QueueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Full => "weighted queue is full",
            Self::Timeout => "weighted queue wait timed out",
            Self::Closed => "weighted queue is closed",
            Self::InvalidWeight => "request weight must be between 1 and 100",
        })
    }
}
impl std::error::Error for QueueError {}

type Key = (u128, u128);
type Delivery = (
    oneshot::Sender<Result<QueuePermit, QueueError>>,
    Result<QueuePermit, QueueError>,
);
struct Waiter {
    bytes: usize,
    deadline: Instant,
    sender: oneshot::Sender<Result<QueuePermit, QueueError>>,
}
#[derive(Default)]
struct State {
    closed: bool,
    inflight: usize,
    bytes: usize,
    progress: u128,
    sequence: u128,
    waiting: BTreeMap<Key, Waiter>,
}
pub struct WeightedQueue {
    config: WeightedQueueConfig,
    state: Mutex<State>,
}
pub struct QueuePermit {
    queue: Option<Arc<WeightedQueue>>,
}
struct Ticket {
    queue: Arc<WeightedQueue>,
    key: Key,
}

impl WeightedQueue {
    pub fn new(config: WeightedQueueConfig) -> Result<Arc<Self>, String> {
        config.validate()?;
        Ok(Arc::new(Self {
            config,
            state: Mutex::new(State::default()),
        }))
    }

    pub async fn acquire(
        self: &Arc<Self>,
        weight: u8,
        body_bytes: usize,
        deadline: Instant,
    ) -> Result<QueuePermit, QueueError> {
        if !(1..=100).contains(&weight) {
            return Err(QueueError::InvalidWeight);
        }
        let now = Instant::now();
        let deadline = deadline.min(
            now.checked_add(Duration::from_millis(self.config.queue_timeout_ms))
                .ok_or(QueueError::Timeout)?,
        );
        let (receiver, ticket) = {
            let mut state = self.state.lock().unwrap();
            if state.closed {
                return Err(QueueError::Closed);
            }
            if deadline <= Instant::now() {
                return Err(QueueError::Timeout);
            }
            let finish = state
                .progress
                .checked_add(1_000_000u128.div_ceil(u128::from(weight)))
                .ok_or(QueueError::Closed)?;
            if state.waiting.is_empty() && state.inflight < self.config.max_inflight {
                state.inflight += 1;
                state.progress = finish;
                return Ok(QueuePermit {
                    queue: Some(self.clone()),
                });
            }
            let bytes = state
                .bytes
                .checked_add(body_bytes)
                .ok_or(QueueError::Full)?;
            if state.waiting.len() >= self.config.max_queued_requests
                || bytes > self.config.max_queued_body_bytes
            {
                return Err(QueueError::Full);
            }
            let sequence = state.sequence.checked_add(1).ok_or(QueueError::Closed)?;
            state.sequence = sequence;
            let key = (finish, sequence);
            let (sender, receiver) = oneshot::channel();
            state.waiting.insert(
                key,
                Waiter {
                    bytes: body_bytes,
                    deadline,
                    sender,
                },
            );
            state.bytes = bytes;
            (
                receiver,
                Ticket {
                    queue: self.clone(),
                    key,
                },
            )
        };
        let result = tokio::select! {
            biased;
            result = receiver => result.unwrap_or(Err(QueueError::Closed)),
            _ = tokio::time::sleep_until(deadline) => Err(QueueError::Timeout),
        };
        drop(ticket);
        result
    }

    // Claim capacity under the lock; deliver permits only after releasing it.
    fn pump(self: &Arc<Self>, state: &mut State) -> Vec<Delivery> {
        let mut deliveries = Vec::new();
        while !state.closed && state.inflight < self.config.max_inflight {
            let Some(((finish, _), waiter)) = state.waiting.pop_first() else {
                break;
            };
            state.bytes -= waiter.bytes;
            if waiter.deadline <= Instant::now() {
                deliveries.push((waiter.sender, Err(QueueError::Timeout)));
                continue;
            }
            state.progress = state.progress.max(finish);
            state.inflight += 1;
            deliveries.push((
                waiter.sender,
                Ok(QueuePermit {
                    queue: Some(self.clone()),
                }),
            ));
        }
        deliveries
    }

    fn deliver(mut deliveries: Vec<Delivery>) {
        while let Some((sender, result)) = deliveries.pop() {
            if let Err(Ok(mut permit)) = sender.send(result) {
                // Disarm before releasing so cancelled deliveries do not recurse.
                let queue = permit.queue.take().unwrap();
                deliveries.extend(queue.release());
            }
        }
    }

    fn release(self: &Arc<Self>) -> Vec<Delivery> {
        let mut state = self.state.lock().unwrap();
        state.inflight -= 1;
        self.pump(&mut state)
    }

    pub fn close(&self) {
        let waiting = {
            let mut state = self.state.lock().unwrap();
            state.closed = true;
            state.bytes = 0;
            std::mem::take(&mut state.waiting)
        };
        for (_, waiter) in waiting {
            let _ = waiter.sender.send(Err(QueueError::Closed));
        }
    }
}
impl Drop for Ticket {
    fn drop(&mut self) {
        let mut state = self.queue.state.lock().unwrap();
        if let Some(waiter) = state.waiting.remove(&self.key) {
            state.bytes -= waiter.bytes;
        }
    }
}
impl Drop for QueuePermit {
    fn drop(&mut self) {
        if let Some(queue) = self.queue.take() {
            WeightedQueue::deliver(queue.release());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::poll;

    fn queue(capacity: usize) -> Arc<WeightedQueue> {
        WeightedQueue::new(WeightedQueueConfig {
            max_inflight: capacity,
            max_queued_requests: 16,
            max_queued_body_bytes: 16,
            queue_timeout_ms: 30_000,
        })
        .unwrap()
    }
    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(60)
    }

    #[test]
    fn configuration_defaults_and_validation() {
        let config: WeightedQueueConfig = serde_json::from_str(r#"{"max_inflight":2}"#).unwrap();
        assert_eq!(config.max_queued_requests, 1024);
        assert_eq!(config.max_queued_body_bytes, 67_108_864);
        assert_eq!(config.queue_timeout_ms, 30_000);
        assert!(config.validate().is_ok());
        assert!(
            serde_json::from_str::<WeightedQueueConfig>(r#"{"max_inflight":2,"unknown":1}"#)
                .is_err()
        );
        assert!(serde_json::from_str::<WeightedQueueConfig>("{}").is_err());
        assert!(WeightedQueueConfig {
            max_inflight: 0,
            ..config
        }
        .validate()
        .is_err());
    }

    #[tokio::test]
    async fn weighted_order_and_new_arrivals_use_current_progress() {
        let q = queue(1);
        let held = q.acquire(1, 1, deadline()).await.unwrap();
        let mut low = Box::pin(q.acquire(1, 1, deadline()));
        let mut middle = Box::pin(q.acquire(2, 1, deadline()));
        let mut high = Box::pin(q.acquire(4, 1, deadline()));
        assert!(poll!(&mut low).is_pending());
        assert!(poll!(&mut middle).is_pending());
        assert!(poll!(&mut high).is_pending());
        drop(held);
        let high = high.await.unwrap();
        assert!(poll!(&mut middle).is_pending());
        let mut newcomer = Box::pin(q.acquire(100, 1, deadline()));
        assert!(poll!(&mut newcomer).is_pending());
        drop(high);
        let newcomer = newcomer.await.unwrap();
        drop(newcomer);
        let mut held = middle.await.unwrap();
        // Repeated high-weight arrivals eventually finish behind the old request.
        for _ in 0..100 {
            let mut next = Box::pin(q.acquire(100, 1, deadline()));
            assert!(poll!(&mut next).is_pending());
            drop(held);
            if let std::task::Poll::Ready(permit) = poll!(&mut low) {
                drop(permit.unwrap());
                drop(next.await.unwrap());
                return;
            }
            held = next.await.unwrap();
        }
        panic!("old request did not receive a permit");
    }

    #[tokio::test]
    async fn equal_weights_are_fifo_and_cancellation_reclaims_budget() {
        let q = queue(1);
        let held = q.acquire(1, 1, deadline()).await.unwrap();
        let mut first = Box::pin(q.acquire(2, 10, deadline()));
        let mut second = Box::pin(q.acquire(2, 6, deadline()));
        assert!(poll!(&mut first).is_pending());
        assert!(poll!(&mut second).is_pending());
        assert!(matches!(
            q.acquire(1, 1, deadline()).await,
            Err(QueueError::Full)
        ));
        drop(held);
        let first = first.await.unwrap();
        assert!(poll!(&mut second).is_pending());
        drop(second);
        assert_eq!(q.state.lock().unwrap().bytes, 0);
        drop(first);
        assert_eq!(q.state.lock().unwrap().inflight, 0);
    }

    #[tokio::test(start_paused = true)]
    async fn timeout_close_and_invalid_weight() {
        let q = queue(1);
        assert!(matches!(
            q.acquire(0, 1, deadline()).await,
            Err(QueueError::InvalidWeight)
        ));
        let held = q.acquire(1, 1, deadline()).await.unwrap();
        assert!(matches!(
            q.acquire(1, 1, deadline()).await,
            Err(QueueError::Timeout)
        ));
        assert_eq!(q.state.lock().unwrap().bytes, 0);
        let mut waiter = Box::pin(q.acquire(1, 1, deadline()));
        assert!(poll!(&mut waiter).is_pending());
        q.close();
        assert!(matches!(waiter.await, Err(QueueError::Closed)));
        assert!(matches!(
            q.acquire(1, 1, deadline()).await,
            Err(QueueError::Closed)
        ));
        drop(held);
        assert_eq!(q.state.lock().unwrap().inflight, 0);
    }

    #[tokio::test]
    async fn cancelled_delivery_releases_claimed_capacity() {
        let q = queue(1);
        let mut held = q.acquire(1, 1, deadline()).await.unwrap();
        let mut waiter = Box::pin(q.acquire(1, 1, deadline()));
        assert!(poll!(&mut waiter).is_pending());
        // Reproduce cancellation between claiming capacity and sending the permit.
        let deliveries = {
            let mut state = q.state.lock().unwrap();
            state.inflight -= 1;
            q.pump(&mut state)
        };
        held.queue.take();
        drop(held);
        drop(waiter);
        WeightedQueue::deliver(deliveries);
        assert_eq!(q.state.lock().unwrap().inflight, 0);
        let permit = q.acquire(1, 1, deadline()).await.unwrap();
        drop(permit);
    }

    #[tokio::test(start_paused = true)]
    async fn request_limit_and_expired_waiter_cannot_claim_capacity() {
        let q = WeightedQueue::new(WeightedQueueConfig {
            max_inflight: 1,
            max_queued_requests: 1,
            max_queued_body_bytes: 16,
            queue_timeout_ms: 30_000,
        })
        .unwrap();
        let held = q.acquire(1, 1, deadline()).await.unwrap();
        let mut waiter = Box::pin(q.acquire(1, 1, Instant::now() + Duration::from_secs(1)));
        assert!(poll!(&mut waiter).is_pending());
        assert!(matches!(
            q.acquire(1, 0, deadline()).await,
            Err(QueueError::Full)
        ));
        tokio::time::advance(Duration::from_secs(2)).await;
        drop(held);
        assert!(matches!(waiter.await, Err(QueueError::Timeout)));
        assert_eq!(q.state.lock().unwrap().inflight, 0);
        assert_eq!(q.state.lock().unwrap().bytes, 0);
    }

    #[tokio::test]
    async fn close_after_claim_preserves_existing_grant() {
        let q = queue(1);
        let mut held = q.acquire(1, 1, deadline()).await.unwrap();
        let mut waiter = Box::pin(q.acquire(1, 1, deadline()));
        assert!(poll!(&mut waiter).is_pending());
        let deliveries = q.release();
        held.queue.take();
        q.close();
        WeightedQueue::deliver(deliveries);
        drop(waiter.await.unwrap());
        assert_eq!(q.state.lock().unwrap().inflight, 0);
    }

    #[tokio::test]
    async fn concurrent_capacity_never_overshoots() {
        let q = queue(3);
        let mut tasks = Vec::new();
        for _ in 0..16 {
            let q = q.clone();
            tasks.push(tokio::spawn(async move {
                let permit = q.acquire(1, 1, deadline()).await.unwrap();
                assert!(q.state.lock().unwrap().inflight <= 3);
                tokio::task::yield_now().await;
                drop(permit);
            }));
        }
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(q.state.lock().unwrap().inflight, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn public_acquire_cancellation_preserves_all_capacity() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let q = WeightedQueue::new(WeightedQueueConfig {
            max_inflight: 4,
            max_queued_requests: 64,
            max_queued_body_bytes: 64,
            queue_timeout_ms: 30_000,
        })
        .unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let mut held = Vec::new();
        for _ in 0..4 {
            held.push(q.acquire(1, 0, deadline()).await.unwrap());
        }
        let mut waiters = Vec::new();
        for weight in 1..=48 {
            let queue = q.clone();
            let active = active.clone();
            let mut waiter = Box::pin(async move {
                let permit = queue.acquire(weight, 1, deadline()).await.unwrap();
                assert!(active.fetch_add(1, Ordering::SeqCst) < 4);
                tokio::task::yield_now().await;
                active.fetch_sub(1, Ordering::SeqCst);
                drop(permit);
            });
            assert!(poll!(&mut waiter).is_pending());
            waiters.push(waiter);
        }
        let mut tasks = Vec::new();
        for (index, waiter) in waiters.into_iter().enumerate() {
            if index % 2 == 0 {
                drop(waiter);
            } else {
                tasks.push(tokio::spawn(waiter));
            }
        }
        drop(held);
        for task in tasks {
            task.await.unwrap();
        }
        assert_eq!(active.load(Ordering::SeqCst), 0);
        let mut recovered = Vec::new();
        for _ in 0..4 {
            recovered.push(
                tokio::time::timeout(Duration::from_secs(5), q.acquire(1, 0, deadline()))
                    .await
                    .unwrap()
                    .unwrap(),
            );
        }
        let mut fifth = Box::pin(q.acquire(1, 0, deadline()));
        assert!(poll!(&mut fifth).is_pending());
        drop(fifth);
        drop(recovered);
    }
}
