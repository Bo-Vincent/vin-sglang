// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use crate::config::SessionAwareConfig;
use crate::policies::active_load::{
    spawn_sweeper, ActiveLoadRegistry, Clock, JanitorHandle, SystemTimeClock,
};
use crate::policies::bounded::{
    power_of_two_excluding, pressure_guard_trips, stable_backup, AffinityLoadView,
};
use crate::policies::{Policy, SelectionContext};
use crate::server::metrics::MetricsRegistry;
use crate::workers::Worker;
use dashmap::DashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
struct SessionAssignment {
    primary_url: String,
    backup_url: Option<String>,
    last_seen: Instant,
}

#[derive(Debug)]
struct SessionState {
    assignments: DashMap<String, SessionAssignment>,
    clock: Arc<dyn Clock>,
    idle: Duration,
}

impl SessionState {
    fn sweep_expired(&self) -> usize {
        let now = self.clock.now();
        let mut removed = 0;
        self.assignments.retain(|_, assignment| {
            let keep = now.saturating_duration_since(assignment.last_seen) <= self.idle;
            if !keep {
                removed += 1;
            }
            keep
        });
        removed
    }
}

/// Session-ID affinity with an optional bounded-load escape.
pub struct SessionAwarePolicy {
    config: SessionAwareConfig,
    state: Arc<SessionState>,
    load: AffinityLoadView,
    metrics: OnceLock<Arc<MetricsRegistry>>,
    _janitor: Option<JanitorHandle>,
}

impl SessionAwarePolicy {
    pub fn new(config: SessionAwareConfig, active: Arc<ActiveLoadRegistry>) -> Self {
        let state = Arc::new(SessionState {
            assignments: DashMap::new(),
            clock: Arc::new(SystemTimeClock),
            idle: Duration::from_secs(config.idle_secs),
        });
        let _janitor = if tokio::runtime::Handle::try_current().is_ok() {
            let swept = Arc::clone(&state);
            Some(spawn_sweeper(
                move || swept.sweep_expired(),
                Duration::from_secs(config.eviction_interval_secs),
                "session-aware-eviction",
            ))
        } else {
            None
        };
        Self {
            config,
            state,
            load: AffinityLoadView::new(active),
            metrics: OnceLock::new(),
            _janitor,
        }
    }

    #[cfg(test)]
    fn new_without_sweeper(config: SessionAwareConfig, active: Arc<ActiveLoadRegistry>) -> Self {
        Self::with_clock(config, active, Arc::new(SystemTimeClock))
    }

    #[cfg(test)]
    fn with_clock(
        config: SessionAwareConfig,
        active: Arc<ActiveLoadRegistry>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            state: Arc::new(SessionState {
                assignments: DashMap::new(),
                clock,
                idle: Duration::from_secs(config.idle_secs),
            }),
            config,
            load: AffinityLoadView::new(active),
            metrics: OnceLock::new(),
            _janitor: None,
        }
    }

    fn choose_backup(
        &self,
        workers: &[Arc<Worker>],
        primary: &Worker,
        key: &[u8],
    ) -> Option<Arc<Worker>> {
        if self.config.stable_pair {
            stable_backup(workers, primary.url.as_str(), key)
        } else {
            power_of_two_excluding(workers, Some(primary.url.as_str()), &self.load)
        }
    }

    fn record(&self, reason: &'static str) {
        if let Some(metrics) = self.metrics.get() {
            metrics.record_policy_decision("session_aware", reason);
        }
    }

    fn assign(
        &self,
        workers: &[Arc<Worker>],
        key: &str,
        reason: &'static str,
    ) -> Option<Arc<Worker>> {
        let primary = power_of_two_excluding(workers, None, &self.load)?;
        let backup = self
            .config
            .stable_pair
            .then(|| self.choose_backup(workers, &primary, key.as_bytes()))
            .flatten();
        self.state.assignments.insert(
            key.to_string(),
            SessionAssignment {
                primary_url: primary.url.clone(),
                backup_url: backup.as_ref().map(|worker| worker.url.clone()),
                last_seen: self.state.clock.now(),
            },
        );
        self.record(reason);
        Some(primary)
    }

    #[cfg(test)]
    fn sweep_expired(&self) -> usize {
        self.state.sweep_expired()
    }

    #[cfg(test)]
    fn assignment_count(&self) -> usize {
        self.state.assignments.len()
    }
}

impl Policy for SessionAwarePolicy {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>> {
        let Some(key) = ctx.routing_key().filter(|key| !key.is_empty()) else {
            let selected = power_of_two_excluding(workers, None, &self.load);
            if selected.is_some() {
                self.record("no_session");
            }
            return selected;
        };

        let Some(mut assignment) = self.state.assignments.get_mut(key) else {
            return self.assign(workers, key, "assigned");
        };
        let Some(primary) = workers
            .iter()
            .find(|worker| worker.url == assignment.primary_url)
            .cloned()
        else {
            drop(assignment);
            return self.assign(workers, key, "remap");
        };
        assignment.last_seen = self.state.clock.now();

        if self.config.strict || !self.config.pressure_guard {
            self.record("session_primary");
            return Some(primary);
        }

        let backup = if self.config.stable_pair {
            let configured_backup = assignment
                .backup_url
                .as_deref()
                .and_then(|url| workers.iter().find(|worker| worker.url == url).cloned());
            let backup =
                configured_backup.or_else(|| self.choose_backup(workers, &primary, key.as_bytes()));
            if assignment.backup_url.as_deref() != backup.as_ref().map(|worker| worker.url.as_str())
            {
                assignment.backup_url = backup.as_ref().map(|worker| worker.url.clone());
            }
            backup
        } else {
            self.choose_backup(workers, &primary, key.as_bytes())
        };
        drop(assignment);

        let Some(backup) = backup else {
            self.record("session_primary");
            return Some(primary);
        };
        let (primary_pressure, backup_pressure) = self.load.pair_pressure(&primary, &backup);
        if pressure_guard_trips(
            primary_pressure,
            backup_pressure,
            self.config.pressure_abs_threshold,
            self.config.pressure_rel_threshold,
        ) {
            self.record("pressure_guard");
            Some(backup)
        } else {
            self.record("session_primary");
            Some(primary)
        }
    }

    fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.metrics.set(metrics);
    }
}

impl std::fmt::Debug for SessionAwarePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionAwarePolicy")
            .field("config", &self.config)
            .field("assignments", &self.state.assignments.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SessionAwareConfig;
    use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
    use crate::policies::active_load::{ActiveLoadRegistry, MockClock};
    use crate::policies::{Policy, SelectionContext};
    use crate::workers::Worker;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }))
    }

    fn config(strict: bool) -> SessionAwareConfig {
        SessionAwareConfig {
            strict,
            pressure_abs_threshold: 100,
            pressure_rel_threshold: 1.5,
            ..SessionAwareConfig::default()
        }
    }

    #[test]
    fn same_session_keeps_the_same_primary() {
        let active = ActiveLoadRegistry::with_defaults();
        let policy = SessionAwarePolicy::new_without_sweeper(config(false), active);
        let workers = vec![worker("w0"), worker("w1"), worker("w2")];
        let model = ModelId("m".into());
        let ctx = SelectionContext::with_routing_key(&model, None, Some("session-1"));

        let first = policy.select(&workers, &ctx).unwrap();
        for _ in 0..20 {
            assert_eq!(policy.select(&workers, &ctx).unwrap().url, first.url);
        }
    }

    #[test]
    fn strict_session_ignores_pressure_escape() {
        let active = ActiveLoadRegistry::with_defaults();
        let policy = SessionAwarePolicy::new_without_sweeper(config(true), Arc::clone(&active));
        let workers = vec![worker("w0"), worker("w1")];
        let model = ModelId("m".into());
        let ctx = SelectionContext::with_routing_key(&model, None, Some("session-2"));
        let primary = policy.select(&workers, &ctx).unwrap();
        let backup = workers.iter().find(|w| w.url != primary.url).unwrap();
        let _hot = active.register(primary.id.clone(), primary.url.clone(), 1000, 0);
        let _cool = active.register(backup.id.clone(), backup.url.clone(), 1, 0);

        assert_eq!(policy.select(&workers, &ctx).unwrap().url, primary.url);
    }

    #[test]
    fn soft_session_uses_backup_without_rewriting_primary() {
        let active = ActiveLoadRegistry::with_defaults();
        let policy = SessionAwarePolicy::new_without_sweeper(config(false), Arc::clone(&active));
        let workers = vec![worker("w0"), worker("w1")];
        let model = ModelId("m".into());
        let ctx = SelectionContext::with_routing_key(&model, None, Some("session-3"));
        let primary = policy.select(&workers, &ctx).unwrap();
        let backup = workers.iter().find(|w| w.url != primary.url).unwrap();
        let hot = active.register(primary.id.clone(), primary.url.clone(), 1000, 0);
        let cool = active.register(backup.id.clone(), backup.url.clone(), 1, 0);

        assert_eq!(policy.select(&workers, &ctx).unwrap().url, backup.url);
        drop(hot);
        drop(cool);
        assert_eq!(policy.select(&workers, &ctx).unwrap().url, primary.url);
    }

    #[test]
    fn idle_session_assignment_is_swept() {
        let clock = Arc::new(MockClock::new(Instant::now()));
        let active = ActiveLoadRegistry::with_defaults();
        let policy = SessionAwarePolicy::with_clock(
            config(false),
            active,
            Arc::clone(&clock) as Arc<dyn Clock>,
        );
        let workers = vec![worker("w0"), worker("w1")];
        let model = ModelId("m".into());
        let ctx = SelectionContext::with_routing_key(&model, None, Some("old"));
        policy.select(&workers, &ctx).unwrap();
        assert_eq!(policy.assignment_count(), 1);

        clock.advance(Duration::from_secs(601));
        assert_eq!(policy.sweep_expired(), 1);
        assert_eq!(policy.assignment_count(), 0);
    }

    #[test]
    fn missing_session_key_degrades_without_creating_an_assignment() {
        let active = ActiveLoadRegistry::with_defaults();
        let policy = SessionAwarePolicy::new_without_sweeper(config(false), active);
        let workers = vec![worker("w0"), worker("w1")];
        let model = ModelId("m".into());
        let ctx = SelectionContext::new(&model, None);

        assert!(policy.select(&workers, &ctx).is_some());
        assert_eq!(policy.assignment_count(), 0);
    }

    #[test]
    fn non_stable_mode_does_not_persist_a_backup() {
        let active = ActiveLoadRegistry::with_defaults();
        let policy = SessionAwarePolicy::new_without_sweeper(config(false), active);
        let workers = vec![worker("w0"), worker("w1"), worker("w2")];
        let model = ModelId("m".into());
        let ctx = SelectionContext::with_routing_key(&model, None, Some("dynamic-backup"));

        policy.select(&workers, &ctx).unwrap();
        assert!(
            policy
                .state
                .assignments
                .get("dynamic-backup")
                .unwrap()
                .backup_url
                .is_none(),
            "Power-of-Two backup must be sampled again on each decision"
        );
    }

    #[test]
    fn non_stable_mode_reexplores_power_of_two_backups() {
        let active = ActiveLoadRegistry::with_defaults();
        let policy = SessionAwarePolicy::new_without_sweeper(config(false), Arc::clone(&active));
        let workers = vec![worker("w0"), worker("w1"), worker("w2")];
        let model = ModelId("m".into());
        let ctx = SelectionContext::with_routing_key(&model, None, Some("reexplore"));
        let primary = policy.select(&workers, &ctx).unwrap();
        let backups: Vec<_> = workers
            .iter()
            .filter(|worker| worker.url != primary.url)
            .cloned()
            .collect();

        let primary_load = active.register(primary.id.clone(), primary.url.clone(), 1000, 0);
        let first_hot = active.register(backups[0].id.clone(), backups[0].url.clone(), 900, 0);
        let first_cool = active.register(backups[1].id.clone(), backups[1].url.clone(), 1, 0);
        assert_eq!(policy.select(&workers, &ctx).unwrap().url, backups[1].url);
        drop(primary_load);
        drop(first_hot);
        drop(first_cool);

        let _primary_load = active.register(primary.id.clone(), primary.url.clone(), 1000, 0);
        let _second_cool = active.register(backups[0].id.clone(), backups[0].url.clone(), 1, 0);
        let _second_hot = active.register(backups[1].id.clone(), backups[1].url.clone(), 900, 0);
        assert_eq!(policy.select(&workers, &ctx).unwrap().url, backups[0].url);
    }

    #[test]
    fn removed_primary_is_remapped_and_counted() {
        let active = ActiveLoadRegistry::with_defaults();
        let policy = SessionAwarePolicy::new_without_sweeper(config(false), active);
        let workers = vec![worker("w0"), worker("w1"), worker("w2")];
        let model = ModelId("m".into());
        let ctx = SelectionContext::with_routing_key(&model, None, Some("session-remap"));
        let metrics = MetricsRegistry::new();
        policy.attach_metrics(Arc::clone(&metrics));
        let first = policy.select(&workers, &ctx).unwrap();
        let survivors: Vec<_> = workers
            .iter()
            .filter(|worker| worker.url != first.url)
            .cloned()
            .collect();

        let remapped = policy.select(&survivors, &ctx).unwrap();
        assert_ne!(remapped.url, first.url);
        let rendered = metrics.render();
        assert!(rendered.contains(
            r#"sgl_router_policy_decisions_total{policy="session_aware",reason="assigned"} 1"#
        ));
        assert!(rendered.contains(
            r#"sgl_router_policy_decisions_total{policy="session_aware",reason="remap"} 1"#
        ));
    }

    #[test]
    fn removed_backup_is_replaced_in_the_assignment() {
        let active = ActiveLoadRegistry::with_defaults();
        let stable_config = SessionAwareConfig {
            stable_pair: true,
            ..config(false)
        };
        let policy = SessionAwarePolicy::new_without_sweeper(stable_config, active);
        let workers = vec![worker("w0"), worker("w1"), worker("w2")];
        let model = ModelId("m".into());
        let ctx = SelectionContext::with_routing_key(&model, None, Some("session-backup"));
        let primary = policy.select(&workers, &ctx).unwrap();
        let old_backup = policy
            .state
            .assignments
            .get("session-backup")
            .unwrap()
            .backup_url
            .clone()
            .unwrap();
        let eligible: Vec<_> = workers
            .iter()
            .filter(|worker| worker.url != old_backup)
            .cloned()
            .collect();

        assert_eq!(policy.select(&eligible, &ctx).unwrap().url, primary.url);
        let replacement = policy
            .state
            .assignments
            .get("session-backup")
            .unwrap()
            .backup_url
            .clone()
            .unwrap();
        assert_ne!(replacement, old_backup);
        assert!(eligible.iter().any(|worker| worker.url == replacement));
    }
}
