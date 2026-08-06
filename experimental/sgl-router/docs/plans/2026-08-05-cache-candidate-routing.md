# Cache candidate routing implementation plan

**Goal:** Replace Cache-Aware's single affinity primary plus arbitrary backup with one bounded global prefix-match candidate tournament, while preserving the existing Step 1 admission/guard framework and making Step 2 Bucket routing the no-hit fallback.

**Architecture:** `Policy::propose_prefill` is the compatibility seam: existing policies return their current pair proposal, while Cache-Aware returns target-specific cache candidates carrying `H` and `E=L-H`. The Router admits and reduces those candidates once; if none survives, it re-enters the normal Bucket/Global domains with affinity lookup disabled and P2 selection. Session-Aware, P2, Decode, legacy sticky, and `cache_aware_zmq` retain their current contracts.

**Tech Stack:** Rust 2021, `sgl-router`, `sgl-kv-indexer::PrefixOutcome`, `LoadMonitorSnapshot`, existing proxy/component test harnesses, and a Python E2E harness on a dedicated 8-GPU test host.

---

## Completion contract

- Cache-Aware performs one Indexer query and considers a bounded Top-K of routable matches.
- Each cache candidate has `L`, coarse `H`, and target-specific `E=L-H`.
- Configured minimum matched-token and match-ratio gates are lower bounds; all configured gates must pass.
- Every candidate independently passes hard compatibility and Prefill Admission before entering the pairwise tournament.
- Cache-Aware returns one final P and no policy backup; pressure is a comparator input, not a late backup switch.
- If no cache candidate survives, the Router restarts from the first no-hit Prefill domain with `E=L` and P2/Load-Aware selection.
- Step 1 uses a catch-all domain; Step 2 may provide ordered Buckets. Cache-Aware has no `bucket/global-rebind/global-preserve` Session mode.
- Session-Aware keeps its range, stable pair, and optional pressure guard behavior.
- Existing `sticky` and `cache_aware_zmq` behavior is unchanged.

### Task 1: Define the cache-candidate proposal contract

**Files:**

- Modify: `src/policies/mod.rs`
- Modify: `src/policies/scoring/mod.rs`
- Test: `src/policies/mod.rs`

**Step 1: Write the failing tests**

Add tests proving that a Cache-Aware prefill proposal carries multiple candidates with `matched_prefix_tokens` and `uncached_tokens`, while the default policy adapter still returns the existing pair proposal.

**Step 2: Run the tests and verify RED**

Run:

```bash
cargo test --lib policies::tests::cache_candidates_ -- --nocapture
```

Expected: compile/test failure because the cache-candidate proposal type and `Policy::propose_prefill` do not exist.

**Step 3: Implement the minimum interface**

Add a `PrefillProposal` enum with existing pair and bounded cache-candidate variants. Add a default `Policy::propose_prefill` adapter and make `Pipeline` preserve the filtered eligible set for both variants.

**Step 4: Run GREEN**

Run the same command and expect all selected tests to pass.

### Task 2: Build bounded candidates and configurable gates

**Files:**

- Modify: `src/config/cli.rs`
- Modify: `src/config/types.rs`
- Modify: `src/policies/cache_aware.rs`
- Modify: `src/policies/factory.rs`
- Test: `src/config/cli.rs`
- Test: `src/policies/cache_aware.rs`
- Test: `src/policies/mod.rs`

**Step 1: Write the failing tests**

Cover:

- non-routable matches are ignored;
- `H = L * min(matched_blocks, query_blocks) / query_blocks` and `E = L-H`;
- minimum matched tokens and ratio use AND semantics when both are configured;
- `K=min(N,K_max,max(K_min,ceil(ratio*N)))` bounds attempted candidates;
- equal-hit candidates prefer lower fresh Prefill pressure before deterministic worker-id tie-break;
- no signal or no candidate passing the gates produces the no-hit proposal, not a cache backup.

**Step 2: Run RED**

```bash
cargo test --lib cache_candidate -- --nocapture
cargo test --lib config::cli::tests::cache_candidate -- --nocapture
```

**Step 3: Implement the minimum code**

Keep Cache-Aware configuration semantically independent: CLI rejects Session-only range/stable-pair
fields and Cache policy reads only candidate/pressure fields. The rebased Step 1 `ModelConfig` shape
is left unchanged, so those fields remain in the existing `AffinityConfig` carrier rather than
forcing mechanical edits into dependency commits. Keep Indexer RPC behavior unchanged and bound
only Router candidate attempts in Step 1.

**Step 4: Run GREEN**

Run both commands and expect zero failures.

### Task 3: Admit and reduce candidates without a backup

**Files:**

- Modify: `src/policies/admission.rs`
- Test: `src/policies/mod.rs`

**Step 1: Write the failing tests**

Cover:

- full `L` is used for max-context/KV safety and candidate-specific `E` for pending Prefill capacity;
- an inadmissible first match does not hide a later admissible match;
- admitted candidates establish one global minimum-E floor before pressure comparison;
- near-tie pressure switches cannot chain beyond one margin from that global floor;
- materially worse pressure can veto a small cache gain;
- otherwise lower `E` wins, and near-equal `E` uses lower pressure;
- the returned Cache-Aware `FinalDecision` has no backup;
- no admitted cache candidate returns `None` so the caller can enter no-hit fallback.

**Step 2: Run RED**

```bash
cargo test --lib cache_tournament -- --nocapture
```

**Step 3: Implement the reducer**

Expose one candidate-admission helper and one pure pairwise comparator. First collect the bounded admitted set, anchor the near-tie band to its global minimum E, and only reduce candidates inside that band. Do not change the existing pair resolver used by Session/P2/Decode.

**Step 4: Run GREEN**

Run the selected tests and the complete `cargo test --lib` suite.

### Task 4: Route cache candidates before Bucket fallback

**Files:**

- Modify: `src/policies/buckets.rs`
- Modify: `src/server/routes/chat.rs`
- Modify: `tests/component/policies/bucket_domains.rs`
- Modify: `tests/proxy/bucket_routing.rs`

**Step 1: Write the failing tests**

Cover:

- Cache-Aware performs one global match before normal Prefill Bucket selection;
- a cross-Bucket match uses candidate `E` for the Bucket work range but full `L` for context compatibility;
- a candidate that fails its own Bucket SLO or hard Admission is skipped;
- when all cache candidates fail, fallback restarts at the first ordered no-hit Bucket and uses P2;
- no Bucket config uses the global domain for the same fallback;
- Session-Aware still follows its configured range semantics.

**Step 2: Run RED**

```bash
cargo test --test component bucket_domains -- --nocapture
cargo test --test proxy bucket_routing -- --nocapture
```

**Step 3: Implement the route flow**

Move Cache-Aware into a dedicated cache-candidate-first branch. Rename the old cross-Bucket helper to describe candidate work compatibility. Keep the existing domain retry loop for P2, Session, Score, and Cache no-hit fallback.

**Step 4: Run GREEN**

Run the selected component/proxy tests and then all Rust tests.

### Task 5: Align Step 1 and Step 2 documentation and commit ownership

**Files:**

- Modify: `docs/router-policy-step1-step2-design.md`
- Modify: `docs/router-policy-step1-integration.md`
- Modify: `docs/plans/2026-08-04-router-v2-execution.md`
- Modify: `README.md`
- Add: `docs/plans/2026-08-05-cache-candidate-routing.md`

**Step 1: Update the docs**

State explicitly:

- Step 1 implements the bounded cache-candidate interface, gates, Admission, tournament, no-hit global fallback, P/D guard, and LoadMonitor downgrade path.
- Step 2 adds work-aware Bucket compatibility/SLO/rank fallback; Bucket is not required for Step 1.
- Cache-Aware has one fixed global-match semantic and no backup; Session remains a separate affinity policy with range and backup controls.

**Step 2: Validate the docs**

```bash
git diff --check
rg -n "Cache-Aware.*(global-rebind|global-preserve|stable pair|backup)" README.md docs src/config
```

Expected: no stale statement claiming that new Cache-Aware exposes a range mode or stable backup.

**Step 3: Place changes into their original commits**

Create `fixup!` commits for the rebased Step 1, docs, and Step 2 commits; run interactive autosquash from the first owned commit. Use `git range-diff` to prove the dependency commits remain patch-equivalent.

### Task 6: Full verification and community-owner review

**Files:**

- Modify only files in our owned commits when a verified finding requires it.
- Update: `docs/router-v2-e2e-poc-report.md` with a new section; preserve historical results.

**Step 1: Local/static verification**

```bash
cargo fmt --check
cargo test --lib
cargo test --bin sgl-router
cargo test --test component
cargo test --test proxy
python3 -m unittest discover -s tests/scripts -p 'test_*.py'
python3 -m py_compile scripts/*.py
git diff --check
```

**Step 2: Remote single-machine PD verification**

Use `rsync` to deploy the exact source tree to the dedicated GPU test host. Start at least one Prefill and one Decode endpoint, verify P selection, D selection, bootstrap handoff, Cache-Aware hit/no-hit behavior, admission fallback, request completion, request errors, fatal/OOM, and route reasons.

**Step 3: Full E2E/performance verification**

Run repeatable comparisons for P2, old `cache_aware_zmq`, new Cache-Aware, Session-Aware, and the relevant Step 2 Bucket cases. Record RT, TTFT, ITL/TPS, throughput, KV hit rate, worker CV, reason counts, RSD, errors, fatal/OOM, and GPU health. Preserve raw results and use confirmation rounds for unstable primary metrics.

**Step 4: Review exact base..head**

Use `review-as-community-owner` against the latest `origin/main`. Every finding needs an exact line, reachable trigger, impact, and reproduction/test. Fix only findings in our owned commits, add a failing regression test first, and autosquash each fix into its owning commit.

**Step 5: Final audit**

Re-run all verification commands, verify author/committer identity, verify no `fixup!` commit remains, verify the branch is not pushed, and leave the worktree ready for user review.
