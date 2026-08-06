// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod active_load;
pub mod admission;
pub mod cache_aware;
pub mod cache_aware_zmq;
pub mod factory;
pub mod kv_events;
pub mod load_based;
pub mod power_of_two;
pub mod random;
pub mod registry;
pub mod round_robin;
pub mod scoring;
pub mod session_aware;
pub mod sticky;

use crate::discovery::ModelId;
use crate::load_monitor::LoadMonitorSnapshot;
use crate::policies::scoring::{EligibilityFilter, ScoringPolicy};
use crate::server::metrics::MetricsRegistry;
use crate::tokenizer::{adapter, TokenizerRegistry};
use crate::workers::Worker;
use dashmap::DashMap;
use std::sync::Arc;

/// Tokens produced once at ingress for a request. Consumed by the
/// cache-aware selection decision and, when `engine_equivalent`, forwarded
/// to the engine as `input_ids` so the engine skips its own prompt
/// tokenization (the router and engine would otherwise tokenize the same
/// prompt twice in the same cluster).
pub struct RequestTokens {
    /// The prompt token ids.
    pub ids: Vec<u32>,
    /// True only when the ids were produced via the model's chat encoder —
    /// i.e. they match what the engine would tokenize from the chat
    /// template. False for the raw-prompt fallback, where the engine must
    /// tokenize the text itself, so the ids are NOT safe to forward.
    pub engine_equivalent: bool,
}

/// External Indexer answer prepared once on the async ingress path for the
/// synchronous cache-aware policy.
pub struct ExternalPrefixSignal {
    pub outcome: sgl_kv_indexer::PrefixOutcome,
    pub query_blocks: usize,
}

/// Produce the routing tokens — and whether they are engine-equivalent —
/// from an already-parsed request body, using the shared tokenizer registry.
///
/// Tokenization is a property of the MODEL (does it have a chat encoder?),
/// not of the routing policy, so this lives here as a free function the
/// ingress calls directly with `ctx.tokenizers` — every policy (sticky,
/// round-robin, cache-aware) shares one tokenize. The cache-aware policy also
/// calls it as a body-tokenize fallback for callers that didn't pre-tokenize.
///
/// Chat requests (`messages`) on a model that has a chat encoder are rendered
/// through that encoder and tokenized the way the engine does, so the query
/// hashes match the engine's cached blocks (chat-templated tokens) AND the ids
/// are safe to hand the engine as `input_ids` (`engine_equivalent = true`).
/// Everything else — `/v1/completions` (`prompt`), `/generate` (`text`), or a
/// chat model without an encoder — tokenizes the raw extracted prompt text;
/// those ids only match the engine after it applies its own template, so they
/// are NOT engine-equivalent. A failed encoder render/encode falls through to
/// the raw path rather than failing the request.
pub fn request_tokens_for(
    tokenizers: &TokenizerRegistry,
    model_id: &ModelId,
    value: &serde_json::Value,
) -> Option<RequestTokens> {
    if tokenizers.has_chat_encoder(&model_id.0) {
        if let Some(messages) = value.get("messages").filter(|m| m.is_array()) {
            if let Some(ids) = tokenizers.encode_chat(&model_id.0, messages) {
                return Some(RequestTokens {
                    ids,
                    engine_equivalent: true,
                });
            }
        }
    }
    let text = extract_prompt_text_from_value(value)?;
    let ids = tokenize_text(tokenizers, model_id, &text)?;
    Some(RequestTokens {
        ids,
        engine_equivalent: false,
    })
}

/// Tokenize `text` for `model_id` via the shared registry. Returns `None` if
/// no tokenizer is loaded (the model_id may be misconfigured) or if encoding
/// fails / yields no tokens. An encode error logs at WARN (a loaded-but-erroring
/// tokenizer silently disables the offload); the no-text / empty-output paths
/// are expected and stay quiet.
fn tokenize_text(
    tokenizers: &TokenizerRegistry,
    model_id: &ModelId,
    text: &str,
) -> Option<Vec<u32>> {
    let tokenizer = tokenizers.get(&model_id.0)?;
    match adapter::encode(&tokenizer, text) {
        Ok(ids) if !ids.is_empty() => Some(ids),
        Ok(_) => None,
        Err(e) => {
            // WARN, not DEBUG: a tokenizer that is loaded but consistently
            // erroring silently turns the whole tokenization offload into a
            // no-op, so the failure must be visible above DEBUG. Sustained
            // failure logs once per request; the volume signal is the
            // `sgl_router_ingress_tokenize_errors_total` counter (which the
            // chat handler bumps on the chat-encode failure), so no
            // rate-limiter here.
            tracing::warn!(
                model = %model_id,
                error = %e,
                "ingress tokenize failed; routing/forwarding skips this prompt",
            );
            None
        }
    }
}

/// Extract a raw prompt-text candidate from an already-parsed JSON request
/// body. Returns `None` when there's no routable text field; the caller then
/// skips tokenization. This is the raw path — chat requests on a model with a
/// chat encoder are rendered via the encoder instead (see [`request_tokens_for`]).
///
/// Supported shapes (in priority order):
///   1. `"prompt": "..."` — `/v1/completions`-style.
///   2. `"prompt": ["...", "..."]` — `/v1/completions` array form;
///      concatenated with `"\n"`.
///   3. `"messages": [{"content": "..."}]` — `/v1/chat/completions`
///      with string content; concatenated with `"\n"`.
///   4. `"messages": [{"content": [{"text": "..."}]}]` — chat with
///      multimodal content blocks; text-only blocks concatenated.
///   5. `"text": "..."` — SGLang `/generate` native form.
///
/// Anything else yields `None`.
pub(crate) fn extract_prompt_text_from_value(v: &serde_json::Value) -> Option<String> {
    if let Some(s) = v.get("prompt").and_then(|p| p.as_str()) {
        return Some(s.to_string());
    }
    if let Some(arr) = v.get("prompt").and_then(|p| p.as_array()) {
        let parts: Vec<&str> = arr.iter().filter_map(|x| x.as_str()).collect();
        if !parts.is_empty() {
            return Some(parts.join("\n"));
        }
    }
    if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
        let mut buf = String::new();
        for m in msgs {
            match m.get("content") {
                Some(serde_json::Value::String(s)) => {
                    if !buf.is_empty() {
                        buf.push('\n');
                    }
                    buf.push_str(s);
                }
                Some(serde_json::Value::Array(parts)) => {
                    for part in parts {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            if !buf.is_empty() {
                                buf.push('\n');
                            }
                            buf.push_str(t);
                        }
                    }
                }
                _ => {}
            }
        }
        if !buf.is_empty() {
            return Some(buf);
        }
    }
    if let Some(s) = v.get("text").and_then(|t| t.as_str()) {
        return Some(s.to_string());
    }
    None
}

/// Selection input — carries the request body and the routing tokens
/// (computed once at ingress) so cache-aware policies can hash prefix
/// tokens without reshaping the [`Policy`] trait or re-tokenizing.  Today's
/// load-only policies (round-robin, random, power-of-two, load-based) read
/// only `workers`; sticky reads `routing_key`.
///
/// Constructed via [`Self::new`] / [`Self::with_routing_key`]; the
/// ingress-computed tokens are attached with [`Self::with_request_tokens`].
/// Accessors expose immutable references so callers cannot mutate the model
/// id or swap in a different body without going through the constructor.
#[derive(Clone)]
pub struct SelectionContext<'a> {
    model: &'a ModelId,
    request_body: Option<&'a [u8]>,
    routing_key: Option<&'a str>,
    request_tokens: Option<&'a [u32]>,
    external_prefix: Option<&'a ExternalPrefixSignal>,
    session_id: Option<&'a str>,
    candidate_range_id: &'a str,
    input_tokens: Option<u64>,
    load_snapshot: Option<&'a LoadMonitorSnapshot>,
    affinity_lookup_enabled: bool,
    affinity_assignment_enabled: bool,
}

impl<'a> SelectionContext<'a> {
    pub fn new(model: &'a ModelId, request_body: Option<&'a [u8]>) -> Self {
        Self {
            model,
            request_body,
            routing_key: None,
            request_tokens: None,
            external_prefix: None,
            session_id: None,
            candidate_range_id: "global",
            input_tokens: None,
            load_snapshot: None,
            affinity_lookup_enabled: true,
            affinity_assignment_enabled: true,
        }
    }

    pub fn with_routing_key(
        model: &'a ModelId,
        request_body: Option<&'a [u8]>,
        routing_key: Option<&'a str>,
    ) -> Self {
        Self {
            model,
            request_body,
            routing_key,
            request_tokens: None,
            external_prefix: None,
            session_id: None,
            candidate_range_id: "global",
            input_tokens: None,
            load_snapshot: None,
            affinity_lookup_enabled: true,
            affinity_assignment_enabled: true,
        }
    }

    /// Attach the ingress-computed routing tokens. When present, the
    /// cache-aware policy consumes these instead of re-parsing and
    /// re-tokenizing the body.
    pub fn with_request_tokens(mut self, request_tokens: Option<&'a [u32]>) -> Self {
        self.request_tokens = request_tokens;
        self
    }

    pub fn with_external_prefix(
        mut self,
        external_prefix: Option<&'a ExternalPrefixSignal>,
    ) -> Self {
        self.external_prefix = external_prefix;
        self
    }

    pub fn with_session_id(mut self, session_id: Option<&'a str>) -> Self {
        self.session_id = session_id;
        self
    }

    pub fn with_candidate_range_id(mut self, candidate_range_id: &'a str) -> Self {
        self.candidate_range_id = candidate_range_id;
        self
    }
    pub fn with_input_tokens(mut self, input_tokens: u64) -> Self {
        self.input_tokens = Some(input_tokens);
        self
    }

    pub fn with_load_snapshot(mut self, load_snapshot: &'a LoadMonitorSnapshot) -> Self {
        self.load_snapshot = Some(load_snapshot);
        self
    }

    pub fn without_affinity_lookup(mut self) -> Self {
        self.affinity_lookup_enabled = false;
        self.affinity_assignment_enabled = false;
        self
    }

    pub fn without_affinity_assignment(mut self) -> Self {
        self.affinity_assignment_enabled = false;
        self
    }

    pub fn model(&self) -> &ModelId {
        self.model
    }

    pub fn request_body(&self) -> Option<&[u8]> {
        self.request_body
    }

    pub fn routing_key(&self) -> Option<&str> {
        self.routing_key
    }

    /// Ingress-precomputed routing tokens, if any. `None` means the policy
    /// must derive tokens itself (e.g. a caller that didn't pre-tokenize).
    pub fn request_tokens(&self) -> Option<&[u32]> {
        self.request_tokens
    }

    pub fn external_prefix(&self) -> Option<&ExternalPrefixSignal> {
        self.external_prefix
    }
    pub fn session_id(&self) -> Option<&str> {
        self.session_id
    }

    pub fn candidate_range_id(&self) -> &str {
        self.candidate_range_id
    }
    pub fn input_tokens(&self) -> Option<u64> {
        self.input_tokens
    }

    pub fn load_snapshot(&self) -> Option<&LoadMonitorSnapshot> {
        self.load_snapshot
    }

    pub fn affinity_lookup_enabled(&self) -> bool {
        self.affinity_lookup_enabled
    }

    pub fn affinity_assignment_enabled(&self) -> bool {
        self.affinity_assignment_enabled
    }
}

/// A policy's ordered candidates for the shared routing pipeline.
#[derive(Clone)]
pub struct SelectionProposal {
    pub primary: Arc<Worker>,
    pub backup: Option<Arc<Worker>>,
    pub kind: ProposalKind,
    pub guard_hints: GuardHints,
    pub eligible_workers: Option<Vec<Arc<Worker>>>,
}

#[derive(Clone)]
pub struct CacheCandidate {
    pub worker: Arc<Worker>,
    pub matched_prefix_tokens: u64,
    pub uncached_tokens: u64,
    pub candidate_range_id: String,
    pub max_pending_prefill_tokens: Option<u64>,
}

#[derive(Clone)]
pub struct CacheCandidateProposal {
    pub candidates: Vec<CacheCandidate>,
    pub cache_switch_margin_tokens: u64,
    pub pressure_abs_threshold_tokens: u64,
    pub pressure_rel_threshold: f64,
}

#[derive(Clone)]
pub enum PrefillProposal {
    Pair(SelectionProposal),
    CacheCandidates(CacheCandidateProposal),
}

impl PrefillProposal {
    pub fn with_eligible_workers(self, workers: Vec<Arc<Worker>>) -> Self {
        match self {
            Self::Pair(proposal) => Self::Pair(proposal.with_eligible_workers(workers)),
            Self::CacheCandidates(mut proposal) => {
                proposal.candidates.retain(|candidate| {
                    workers
                        .iter()
                        .any(|worker| worker.id == candidate.worker.id)
                });
                Self::CacheCandidates(proposal)
            }
        }
    }
}

impl SelectionProposal {
    pub fn primary(primary: Arc<Worker>) -> Self {
        Self {
            primary,
            backup: None,
            kind: ProposalKind::Generic,
            guard_hints: GuardHints::default(),
            eligible_workers: None,
        }
    }

    pub fn with_backup(primary: Arc<Worker>, backup: Arc<Worker>) -> Self {
        Self {
            primary,
            backup: Some(backup),
            kind: ProposalKind::PowerOfTwo,
            guard_hints: GuardHints::default(),
            eligible_workers: None,
        }
    }

    pub fn with_kind(mut self, kind: ProposalKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_guard_hints(mut self, guard_hints: GuardHints) -> Self {
        self.guard_hints = guard_hints;
        self
    }

    pub fn with_eligible_workers(mut self, workers: Vec<Arc<Worker>>) -> Self {
        self.eligible_workers = Some(workers);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalKind {
    Generic,
    PowerOfTwo,
    SessionAffinity,
    CacheAffinity,
    Score,
}

#[derive(Debug, Clone)]
pub struct GuardHints {
    pub enable_pressure_guard: bool,
    pub pressure_abs_threshold_tokens: u64,
    pub pressure_rel_threshold: f64,
}

impl Default for GuardHints {
    fn default() -> Self {
        Self {
            enable_pressure_guard: false,
            pressure_abs_threshold_tokens: 0,
            pressure_rel_threshold: 1.0,
        }
    }
}

pub trait Policy: Send + Sync + std::fmt::Debug {
    fn select(&self, workers: &[Arc<Worker>], ctx: &SelectionContext<'_>) -> Option<Arc<Worker>>;

    fn propose(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<SelectionProposal> {
        self.select(workers, ctx).map(SelectionProposal::primary)
    }

    fn propose_prefill(
        &self,
        workers: &[Arc<Worker>],
        ctx: &SelectionContext<'_>,
    ) -> Option<PrefillProposal> {
        self.propose(workers, ctx).map(PrefillProposal::Pair)
    }

    fn uses_shared_prefill_admission(&self) -> bool {
        false
    }

    fn commit_prefill_selection(
        &self,
        _ctx: &SelectionContext<'_>,
        _proposal_kind: ProposalKind,
        _selected: &Arc<Worker>,
    ) {
    }

    fn is_bucket_affinity_policy(&self) -> bool {
        false
    }

    /// Whether this policy's ROUTING decision needs the request tokens (i.e.
    /// it routes by prompt prefix). Ingress tokenization itself is no longer
    /// gated on this — that is a model property (`has_chat_encoder`) decided at
    /// ingress via [`request_tokens_for`]. This flag is the EXTRA gate that
    /// keeps the cache-aware policy's RAW-prompt routing path alive: a
    /// cache-aware model with no chat encoder still wants its `/v1/completions`
    /// /`text` prompt tokenized for tree matching, which `has_chat_encoder`
    /// alone would not trigger. Default `false` (load-only + sticky route
    /// without prefix tokens); only the cache-aware policy overrides it.
    fn needs_request_tokens(&self) -> bool {
        false
    }

    /// Attach the process metrics registry after construction. Default is a
    /// no-op — only policies that emit metrics (cache-aware-zmq's
    /// `sgl_router_overlap_blocks`) override it. Mirrors
    /// `ActiveLoadRegistry::attach_metrics`: the registry is built after the
    /// policies, so it is injected here rather than passed to the constructor.
    fn attach_metrics(&self, _metrics: Arc<MetricsRegistry>) {}

    /// Per-worker preference view, `None` when the decision cannot reduce to
    /// one (rotation, sampling, a veto); opt in by implementing
    /// [`ScoringPolicy`].
    fn as_scoring(&self) -> Option<&dyn ScoringPolicy> {
        None
    }

    /// Per-worker eligibility view, `None` when this policy imposes no hard
    /// constraint. A scoring policy opts in via [`ScoringPolicy::as_filter`];
    /// a filter-only policy overrides this directly.
    fn as_filter(&self) -> Option<&dyn EligibilityFilter> {
        None
    }

    /// Whether this policy can be a term of a fused sum. DERIVED and never
    /// hand-written, so it cannot claim a capability the policy lacks.
    fn can_fuse(&self) -> bool {
        self.as_scoring().is_some()
    }

    /// Whether this policy can be a `--filter` entry. DERIVED, same reasoning
    /// as [`Self::can_fuse`].
    fn can_filter(&self) -> bool {
        self.as_filter().is_some()
    }
}

#[derive(Debug, Default)]
pub struct PolicyRegistry {
    by_model: DashMap<ModelId, Arc<dyn Policy>>,
}

impl PolicyRegistry {
    pub fn insert(&self, model: ModelId, policy: Arc<dyn Policy>) {
        self.by_model.insert(model, policy);
    }

    pub fn get(&self, model: &ModelId) -> Option<Arc<dyn Policy>> {
        self.by_model.get(model).map(|p| p.clone())
    }

    /// Inject the metrics registry into every registered policy. Called once
    /// at startup (after the registry is built) so metrics-emitting policies
    /// can record into the shared registry.
    pub fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        for entry in self.by_model.iter() {
            entry.value().attach_metrics(Arc::clone(&metrics));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AffinityConfig;
    use crate::discovery::{WorkerId, WorkerMode, WorkerSpec};
    use crate::load_monitor::{AggregateLoad, Freshness, LoadMonitorSnapshot, WorkerSnapshot};
    use crate::policies::admission::{
        resolve_cache_candidates, resolve_prefill, CandidateRange, DecisionReason,
    };
    use crate::policies::cache_aware::CacheAwarePolicy;
    use crate::policies::power_of_two::PowerOfTwoChoicesPolicy;
    use crate::policies::session_aware::SessionAwarePolicy;

    fn worker(id: &str) -> Arc<Worker> {
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(id.into()),
            url: format!("http://{id}:30000"),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("model".into())],
            bootstrap_port: None,
        }))
    }

    fn snapshot(entries: &[(&Arc<Worker>, AggregateLoad)]) -> LoadMonitorSnapshot {
        LoadMonitorSnapshot {
            enabled: true,
            version: 1,
            captured_at: Some("2026-08-04T00:00:00Z".into()),
            workers: entries
                .iter()
                .map(|(worker, aggregate)| WorkerSnapshot {
                    worker_id: worker.id.0.clone(),
                    url: worker.url.clone(),
                    mode: worker.mode(),
                    model_ids: worker
                        .model_ids
                        .iter()
                        .map(|model| model.0.clone())
                        .collect(),
                    freshness: Freshness::Fresh,
                    source_instance_id: None,
                    sequence_id: None,
                    report_time_unix_ms: None,
                    last_error: None,
                    received_at: None,
                    expires_at: None,
                    aggregate: Some(aggregate.clone()),
                    ranks: Vec::new(),
                })
                .collect(),
        }
    }

    #[test]
    fn default_proposal_preserves_legacy_single_worker_selection() {
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None);
        let only = worker("only");
        let policy = PowerOfTwoChoicesPolicy::new();

        let proposal = policy
            .propose(&[Arc::clone(&only)], &ctx)
            .expect("one candidate must produce a proposal");
        assert_eq!(proposal.primary.id, only.id);
        assert!(proposal.backup.is_none());
    }

    #[test]
    fn power_of_two_proposal_keeps_the_other_sample_as_backup() {
        let model = ModelId("model".into());
        let ctx = SelectionContext::new(&model, None);
        let workers = vec![worker("first"), worker("second")];
        let policy = PowerOfTwoChoicesPolicy::new();

        let proposal = policy
            .propose(&workers, &ctx)
            .expect("two candidates must produce a proposal");
        let backup = proposal.backup.expect("P2 must retain its second sample");

        assert_ne!(proposal.primary.id, backup.id);
        assert_eq!(proposal.kind, ProposalKind::PowerOfTwo);
    }

    #[test]
    fn prefill_proposal_adapter_keeps_existing_pair_semantics() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second")];
        let policy = PowerOfTwoChoicesPolicy::new();
        let ctx = SelectionContext::new(&model, None);

        let proposal = policy
            .propose_prefill(&workers, &ctx)
            .expect("P2 must produce a prefill proposal");

        let PrefillProposal::Pair(pair) = proposal else {
            panic!("P2 must retain pair semantics");
        };
        assert_eq!(pair.kind, ProposalKind::PowerOfTwo);
        assert!(pair.backup.is_some());
    }

    #[test]
    fn power_of_two_orders_its_sample_with_fresh_load_monitor_snapshot() {
        let model = ModelId("model".into());
        let busy = worker("busy");
        let idle = worker("idle");
        let workers = vec![Arc::clone(&busy), Arc::clone(&idle)];
        let load_snapshot = snapshot(&[
            (
                &busy,
                AggregateLoad {
                    num_waiting_uncached_tokens: 512,
                    max_running_requests: 8,
                    max_total_num_tokens: 4_096,
                    ..Default::default()
                },
            ),
            (
                &idle,
                AggregateLoad {
                    num_waiting_uncached_tokens: 16,
                    max_running_requests: 8,
                    max_total_num_tokens: 4_096,
                    ..Default::default()
                },
            ),
        ]);
        let ctx = SelectionContext::new(&model, None).with_load_snapshot(&load_snapshot);

        let proposal = PowerOfTwoChoicesPolicy::new()
            .propose(&workers, &ctx)
            .expect("two candidates must produce a proposal");

        assert_eq!(proposal.primary.id, idle.id);
        assert_eq!(
            proposal.backup.expect("P2 keeps its other sample").id,
            busy.id
        );
    }

    #[test]
    fn admission_uses_admitted_backup_before_scanning_candidate_range() {
        let primary = worker("primary");
        let backup = worker("backup");
        let fallback = worker("fallback");
        let workers = vec![
            Arc::clone(&primary),
            Arc::clone(&backup),
            Arc::clone(&fallback),
        ];
        let snapshot = snapshot(&[
            (
                &primary,
                AggregateLoad {
                    num_running_reqs: 4,
                    max_running_requests: 4,
                    num_total_tokens: 10,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &backup,
                AggregateLoad {
                    max_running_requests: 4,
                    num_total_tokens: 10,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &fallback,
                AggregateLoad {
                    max_running_requests: 4,
                    num_total_tokens: 10,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
        ]);
        let range = CandidateRange::global(&workers);
        let proposal = SelectionProposal::with_backup(Arc::clone(&primary), Arc::clone(&backup));

        let decision = resolve_prefill(&range, &proposal, 32, &snapshot)
            .expect("an admitted backup must be selected");

        assert_eq!(decision.selected.id, backup.id);
        assert_eq!(decision.reason, DecisionReason::BackupPrimaryAdmission);
    }

    #[test]
    fn disabled_monitor_does_not_hard_reject_a_registry_healthy_primary() {
        let primary = worker("primary");
        let workers = vec![Arc::clone(&primary)];
        let snapshot = LoadMonitorSnapshot {
            enabled: false,
            version: 0,
            captured_at: None,
            workers: Vec::new(),
        };

        let decision = resolve_prefill(
            &CandidateRange::global(&workers),
            &SelectionProposal::primary(Arc::clone(&primary)),
            1_000_000,
            &snapshot,
        )
        .expect("disabled reporting must preserve the healthy registry candidate");

        assert_eq!(decision.selected.id, primary.id);
        assert_eq!(decision.reason, DecisionReason::Primary);
    }

    #[test]
    fn pressure_guard_uses_fresh_waiting_uncached_tokens_not_local_load() {
        let primary = worker("primary");
        let backup = worker("backup");
        let workers = vec![Arc::clone(&primary), Arc::clone(&backup)];
        let snapshot = snapshot(&[
            (
                &primary,
                AggregateLoad {
                    num_waiting_uncached_tokens: 200,
                    max_running_requests: 4,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
            (
                &backup,
                AggregateLoad {
                    num_waiting_uncached_tokens: 20,
                    max_running_requests: 4,
                    max_total_num_tokens: 1_000,
                    ..Default::default()
                },
            ),
        ]);
        let proposal =
            SelectionProposal::with_backup(primary, backup).with_guard_hints(GuardHints {
                enable_pressure_guard: true,
                pressure_abs_threshold_tokens: 100,
                pressure_rel_threshold: 2.0,
            });

        let decision = resolve_prefill(&CandidateRange::global(&workers), &proposal, 80, &snapshot)
            .expect("both candidates fit capacity");

        assert_eq!(decision.reason, DecisionReason::BackupPressureGuard);
    }

    #[test]
    fn cache_candidates_keep_bounded_target_specific_uncached_work() {
        let model = ModelId("model".into());
        let hot = worker("hot");
        let warm = worker("warm");
        let workers = vec![Arc::clone(&hot), Arc::clone(&warm)];
        let signal = ExternalPrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches: vec![
                    sgl_kv_indexer::PrefixMatch {
                        address: "http://gone:30000".into(),
                        matched_prefix_blocks: 8,
                        worker_id: "gone".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        address: hot.url.clone(),
                        matched_prefix_blocks: 6,
                        worker_id: "hot".into(),
                    },
                    sgl_kv_indexer::PrefixMatch {
                        address: warm.url.clone(),
                        matched_prefix_blocks: 4,
                        worker_id: "warm".into(),
                    },
                ],
                best_prefix_blocks: 8,
            },
            query_blocks: 8,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(8_000)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig {
            cache_candidate_min_workers: 2,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 2,
            ..Default::default()
        });

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("routable matches must produce cache candidates")
        else {
            panic!("cache hits must retain candidate-set semantics");
        };

        assert_eq!(proposal.candidates.len(), 2);
        assert_eq!(proposal.candidates[0].worker.id, hot.id);
        assert_eq!(proposal.candidates[0].matched_prefix_tokens, 6_000);
        assert_eq!(proposal.candidates[0].uncached_tokens, 2_000);
        assert_eq!(proposal.candidates[1].worker.id, warm.id);
        assert_eq!(proposal.candidates[1].matched_prefix_tokens, 4_000);
        assert_eq!(proposal.candidates[1].uncached_tokens, 4_000);
    }

    #[test]
    fn cache_candidate_bound_keeps_the_best_k_from_a_large_match_set() {
        let model = ModelId("model".into());
        let workers: Vec<Arc<Worker>> = (0..64)
            .map(|index| worker(&format!("w{index:02}")))
            .collect();
        let matches = workers
            .iter()
            .enumerate()
            .map(|(index, worker)| sgl_kv_indexer::PrefixMatch {
                address: worker.url.clone(),
                matched_prefix_blocks: (index + 1) as u32,
                worker_id: worker.id.0.clone(),
            })
            .collect();
        let signal = ExternalPrefixSignal {
            outcome: sgl_kv_indexer::PrefixOutcome::Matched {
                matches,
                best_prefix_blocks: 64,
            },
            query_blocks: 64,
        };
        let ctx = SelectionContext::new(&model, None)
            .with_input_tokens(64_000)
            .with_external_prefix(Some(&signal));
        let policy = CacheAwarePolicy::new(AffinityConfig {
            cache_affinity_min_matched_tokens: Some(0),
            cache_candidate_min_workers: 4,
            cache_candidate_ratio: 0.0,
            cache_candidate_max_workers: 4,
            ..Default::default()
        });

        let PrefillProposal::CacheCandidates(proposal) = policy
            .propose_prefill(&workers, &ctx)
            .expect("the bounded best candidates must survive")
        else {
            panic!("cache hits must retain candidate-set semantics");
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

    fn cache_candidate(
        worker: &Arc<Worker>,
        matched_prefix_tokens: u64,
        uncached_tokens: u64,
        max_pending_prefill_tokens: Option<u64>,
    ) -> CacheCandidate {
        CacheCandidate {
            worker: Arc::clone(worker),
            matched_prefix_tokens,
            uncached_tokens,
            candidate_range_id: "global".into(),
            max_pending_prefill_tokens,
        }
    }

    #[test]
    fn cache_tournament_skips_inadmissible_matches_and_returns_no_backup() {
        let full = worker("full");
        let winner = worker("winner");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&full, 90, 10, None),
                cache_candidate(&winner, 70, 30, None),
            ],
            cache_switch_margin_tokens: 16,
            pressure_abs_threshold_tokens: 1_024,
            pressure_rel_threshold: 1.5,
        };
        let loads = snapshot(&[
            (
                &full,
                AggregateLoad {
                    num_running_reqs: 8,
                    max_running_requests: 8,
                    max_total_num_tokens: 10_000,
                    ..AggregateLoad::default()
                },
            ),
            (
                &winner,
                AggregateLoad {
                    max_running_requests: 8,
                    max_total_num_tokens: 10_000,
                    ..AggregateLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads)
            .expect("a later admitted cache match must survive");

        assert_eq!(decision.selected.id, winner.id);
        assert_eq!(decision.primary.id, winner.id);
        assert!(decision.backup.is_none());
        assert_eq!(decision.reason, DecisionReason::CacheCandidate);
    }

    #[test]
    fn cache_tournament_pressure_vetoes_a_small_cache_gain() {
        let congested = worker("congested");
        let idle = worker("idle");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&congested, 90, 10, None),
                cache_candidate(&idle, 80, 20, None),
            ],
            cache_switch_margin_tokens: 32,
            pressure_abs_threshold_tokens: 100,
            pressure_rel_threshold: 1.5,
        };
        let loads = snapshot(&[
            (
                &congested,
                AggregateLoad {
                    num_waiting_uncached_tokens: 1_000,
                    max_running_requests: 8,
                    max_total_num_tokens: 10_000,
                    ..AggregateLoad::default()
                },
            ),
            (
                &idle,
                AggregateLoad {
                    num_waiting_uncached_tokens: 10,
                    max_running_requests: 8,
                    max_total_num_tokens: 10_000,
                    ..AggregateLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads).unwrap();
        assert_eq!(decision.selected.id, idle.id);
    }

    #[test]
    fn cache_tournament_keeps_a_material_cache_gain_despite_pressure() {
        let hot = worker("hot");
        let idle = worker("idle");
        let proposal = CacheCandidateProposal {
            candidates: vec![
                cache_candidate(&hot, 90, 10, None),
                cache_candidate(&idle, 20, 80, None),
            ],
            cache_switch_margin_tokens: 32,
            pressure_abs_threshold_tokens: 100,
            pressure_rel_threshold: 1.5,
        };
        let loads = snapshot(&[
            (
                &hot,
                AggregateLoad {
                    num_waiting_uncached_tokens: 1_000,
                    max_running_requests: 8,
                    max_total_num_tokens: 10_000,
                    ..AggregateLoad::default()
                },
            ),
            (
                &idle,
                AggregateLoad {
                    num_waiting_uncached_tokens: 10,
                    max_running_requests: 8,
                    max_total_num_tokens: 10_000,
                    ..AggregateLoad::default()
                },
            ),
        ]);

        let decision = resolve_cache_candidates(&proposal, 100, &loads).unwrap();
        assert_eq!(
            decision.selected.id, hot.id,
            "pressure may break a near tie, but must not erase a material cache-work gain"
        );
    }

    #[test]
    fn session_affinity_reuses_primary_and_stable_backup() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second"), worker("third")];
        let policy = SessionAwarePolicy::new(AffinityConfig {
            stable_pair: true,
            ..Default::default()
        });
        let ctx = SelectionContext::new(&model, None).with_session_id(Some("session-a"));

        let first = policy
            .propose(&workers, &ctx)
            .expect("a new session must get an initial P2 proposal");
        policy.commit_prefill_selection(&ctx, first.kind, &first.primary);
        let second = policy
            .propose(&workers, &ctx)
            .expect("a mapped session must produce an affinity proposal");
        let third = policy
            .propose(&workers, &ctx)
            .expect("the session assignment must remain stable");

        assert_eq!(second.kind, ProposalKind::SessionAffinity);
        assert_eq!(second.primary.id, first.primary.id);
        assert_eq!(third.primary.id, second.primary.id);
        assert_eq!(
            third.backup.expect("stable pair has backup").id,
            second.backup.expect("stable pair has backup").id,
        );
    }

    #[test]
    fn new_session_commits_the_admitted_backup() {
        let model = ModelId("model".into());
        let workers = vec![worker("first"), worker("second")];
        let policy = SessionAwarePolicy::new(AffinityConfig::default());
        let ctx = SelectionContext::new(&model, None).with_session_id(Some("session-a"));
        let proposal = policy
            .propose(&workers, &ctx)
            .expect("a new session produces a P2 proposal");
        let backup = proposal
            .backup
            .clone()
            .expect("two workers retain a backup");
        let loads = snapshot(&[
            (
                &proposal.primary,
                AggregateLoad {
                    num_running_reqs: 1,
                    max_running_requests: 1,
                    max_total_num_tokens: 4_096,
                    ..Default::default()
                },
            ),
            (
                &backup,
                AggregateLoad {
                    max_running_requests: 1,
                    max_total_num_tokens: 4_096,
                    ..Default::default()
                },
            ),
        ]);
        let decision = resolve_prefill(&CandidateRange::global(&workers), &proposal, 32, &loads)
            .expect("the admitted backup must become final P");
        assert_eq!(decision.selected.id, backup.id);
        policy.commit_prefill_selection(&ctx, proposal.kind, &decision.selected);

        let mapped = policy
            .propose(&workers, &ctx)
            .expect("the next turn must reuse the actual first-turn worker");
        assert_eq!(mapped.kind, ProposalKind::SessionAffinity);
        assert_eq!(mapped.primary.id, backup.id);
    }
}
