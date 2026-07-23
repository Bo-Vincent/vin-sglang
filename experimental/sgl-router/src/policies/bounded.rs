// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::policies::active_load::ActiveLoadRegistry;
use crate::workers::Worker;
use rand::Rng;
use sha2::{Digest, Sha256};
use std::sync::Arc;

/// Read-only load view shared by the new affinity policies.
///
/// Prefill-token load is comparable only when both candidates have a registry
/// slot. If either side is unknown, both candidates degrade to the same
/// router-local in-flight request-count level.
#[derive(Debug, Clone)]
pub(crate) struct AffinityLoadView {
    active: Arc<ActiveLoadRegistry>,
}

impl AffinityLoadView {
    pub(crate) fn new(active: Arc<ActiveLoadRegistry>) -> Self {
        Self { active }
    }

    pub(crate) fn pair_pressure(&self, first: &Worker, second: &Worker) -> (usize, usize) {
        if self.active.is_known(&first.id) && self.active.is_known(&second.id) {
            (
                self.active.prefill_load(&first.id),
                self.active.prefill_load(&second.id),
            )
        } else {
            (first.active_load(), second.active_load())
        }
    }
}

/// Produce a low-cost backup without ever returning the excluded primary.
pub(crate) fn power_of_two_excluding(
    workers: &[Arc<Worker>],
    excluded_url: Option<&str>,
    load: &AffinityLoadView,
) -> Option<Arc<Worker>> {
    match workers {
        [] => return None,
        [only] => {
            return (excluded_url != Some(only.url.as_str())).then(|| Arc::clone(only));
        }
        [first, second] => {
            return match (
                excluded_url == Some(first.url.as_str()),
                excluded_url == Some(second.url.as_str()),
            ) {
                (true, true) => None,
                (true, false) => Some(Arc::clone(second)),
                (false, true) => Some(Arc::clone(first)),
                (false, false) => {
                    let (first_pressure, second_pressure) = load.pair_pressure(first, second);
                    Some(Arc::clone(if first_pressure <= second_pressure {
                        first
                    } else {
                        second
                    }))
                }
            };
        }
        _ => {}
    }

    let mut rng = rand::thread_rng();
    let (first_idx, second_idx) = sample_two_indices(workers, excluded_url, &mut rng)?;
    let first = &workers[first_idx];
    let second = &workers[second_idx];
    let (first_pressure, second_pressure) = load.pair_pressure(first, second);
    Some(Arc::clone(if first_pressure <= second_pressure {
        first
    } else {
        second
    }))
}

/// Uniformly sample two distinct indices while rejecting the optional
/// primary. Production passes a registry-derived slice with unique URLs and
/// at least two eligible workers, so rejection sampling is allocation-free
/// with constant expected work.
fn sample_two_indices<R: Rng + ?Sized>(
    workers: &[Arc<Worker>],
    excluded_url: Option<&str>,
    rng: &mut R,
) -> Option<(usize, usize)> {
    if workers.len() < 2 {
        return None;
    }
    let sample_one = |excluded_index: Option<usize>, rng: &mut R| {
        // With unique registry URLs and at most one excluded primary this
        // succeeds in constant expected time. The bounded retry keeps the
        // helper total for malformed duplicate-URL input; only that exceptional
        // path scans for a valid slot.
        for _ in 0..64 {
            let index = rng.gen_range(0..workers.len());
            if excluded_index != Some(index) && excluded_url != Some(workers[index].url.as_str()) {
                return Some(index);
            }
        }
        workers.iter().enumerate().find_map(|(index, worker)| {
            (excluded_index != Some(index) && excluded_url != Some(worker.url.as_str()))
                .then_some(index)
        })
    };
    let first = sample_one(None, rng)?;
    let second = sample_one(Some(first), rng)?;
    Some((first, second))
}

/// Deterministically select one non-primary worker for an affinity key.
pub(crate) fn stable_backup(
    workers: &[Arc<Worker>],
    primary_url: &str,
    affinity_key: &[u8],
) -> Option<Arc<Worker>> {
    let mut candidates: Vec<&Arc<Worker>> = workers
        .iter()
        .filter(|worker| worker.url != primary_url)
        .collect();
    candidates.sort_unstable_by(|left, right| left.url.cmp(&right.url));
    if candidates.is_empty() {
        return None;
    }
    let digest = Sha256::digest(affinity_key);
    let slot = u64::from_le_bytes(digest[..8].try_into().expect("sha256 has at least 8 bytes"));
    Some(Arc::clone(candidates[slot as usize % candidates.len()]))
}

/// Both thresholds are required so small absolute noise and large relative
/// changes near zero cannot independently trigger an affinity escape.
pub(crate) fn pressure_guard_trips(
    primary: usize,
    backup: usize,
    absolute_threshold: usize,
    relative_threshold: f32,
) -> bool {
    primary.saturating_sub(backup) > absolute_threshold
        && (primary as f64) > (backup as f64 * relative_threshold as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::active_load::ActiveLoadRegistry;
    use crate::workers::Worker;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::sync::Arc;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }))
    }

    #[test]
    fn power_of_two_backup_excludes_primary() {
        let primary = worker("w0");
        let backup = worker("w1");
        let load = AffinityLoadView::new(ActiveLoadRegistry::with_defaults());

        let selected = power_of_two_excluding(
            &[Arc::clone(&primary), Arc::clone(&backup)],
            Some(primary.url.as_str()),
            &load,
        )
        .expect("one non-primary candidate remains");

        assert_eq!(selected.url, backup.url);
    }

    #[test]
    fn stable_backup_is_repeatable_and_never_primary() {
        let workers = vec![worker("w0"), worker("w1"), worker("w2")];
        let first = stable_backup(&workers, workers[0].url.as_str(), b"session-42").unwrap();
        assert_ne!(first.url, workers[0].url);
        for _ in 0..100 {
            assert_eq!(
                stable_backup(&workers, workers[0].url.as_str(), b"session-42")
                    .unwrap()
                    .url,
                first.url
            );
        }
    }

    #[test]
    fn pair_pressure_uses_prefill_tokens_when_both_workers_are_known() {
        let active = ActiveLoadRegistry::with_defaults();
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _g0 = active.register(w0.id.clone(), w0.url.clone(), 500, 0);
        let _g1 = active.register(w1.id.clone(), w1.url.clone(), 10, 0);
        let load = AffinityLoadView::new(active);

        assert_eq!(load.pair_pressure(&w0, &w1), (500, 10));
    }

    #[test]
    fn pair_pressure_degrades_both_candidates_when_one_is_unknown() {
        let active = ActiveLoadRegistry::with_defaults();
        let w0 = worker("w0");
        let w1 = worker("w1");
        let _active_only_for_w0 = active.register(w0.id.clone(), w0.url.clone(), 500, 0);
        let _request_on_w1 = w1.load_guard();
        let load = AffinityLoadView::new(active);

        assert_eq!(load.pair_pressure(&w0, &w1), (0, 1));
    }

    #[test]
    fn pressure_guard_requires_absolute_and_relative_thresholds() {
        assert!(pressure_guard_trips(1000, 100, 500, 1.5));
        assert!(!pressure_guard_trips(550, 100, 500, 1.5));
        assert!(!pressure_guard_trips(1000, 800, 100, 1.5));
    }

    #[test]
    fn power_of_two_sampling_is_uniform_after_excluding_primary() {
        let workers = vec![
            worker("w0"),
            worker("w1"),
            worker("w2"),
            worker("w3"),
            worker("w4"),
        ];
        let mut rng = StdRng::seed_from_u64(42);
        let mut counts = [0usize; 5];
        const DRAWS: usize = 60_000;
        for _ in 0..DRAWS {
            let (first, second) =
                sample_two_indices(&workers, Some(workers[0].url.as_str()), &mut rng).unwrap();
            assert_ne!(first, second);
            assert_ne!(first, 0);
            assert_ne!(second, 0);
            counts[first] += 1;
            counts[second] += 1;
        }

        let expected = DRAWS / 2;
        for count in &counts[1..] {
            assert!(
                count.abs_diff(expected) < 1_000,
                "eligible sampling is biased: {counts:?}"
            );
        }
    }
}
