# Native heterogeneous weight conversion implementation plan

## Baseline

- Branch: `vin/sglang-weight-v4`
- Base: `288705ffdd0ea16d38b98676dac3412984cd3583`
- v3 worktree remains untouched.
- Existing manifest regression baseline: 90 tests passing on the remote Python
  3.12 environment.

## Task 1: Red tests for core contracts

Create:

- `test/registered/unit/weight_transfer/test_contracts.py`
- `test/registered/unit/weight_transfer/test_binding.py`

Cover:

- canonical N-D region invariants and lazy segment iteration;
- runtime and storage binding identity;
- bounds, lease/generation, alias, and target-write overlap failures;
- old `partition_dim` normalization.

Run the tests before adding implementation and record the expected import
failure.

## Task 2: Core contracts

Create:

- `python/sglang/srt/weight_transfer/__init__.py`
- `python/sglang/srt/weight_transfer/contracts.py`
- `python/sglang/srt/weight_transfer/binding.py`

Reuse `WeightPlacementManifest` and `WeightRuntimeBindingManifest`. Add
address-free logical plans, storage bindings, bound locations, executor groups,
PP route groups, validation errors, and plan digests.

## Task 3: Red tests for the N-D planner

Create:

- `test/registered/unit/weight_transfer/test_planner.py`

Add explicit red/green cases for:

- TP4 to TP8 and TP8 to TP4;
- PP2 to PP4 and PP4 to PP2;
- EP8 to EP2 and EP2 to EP8;
- dim0 to dim1 and dim0 to dim2;
- the exact four-axis acceptance case;
- incomplete DP replicas, target holes, overlapping boxes, descriptor/layout
  mismatch, and deterministic planning;
- cross-dimension stress with O(box intersections) regions.

## Task 4: Native planner

Create:

- `python/sglang/srt/weight_transfer/planner.py`

Implement normalization, geometry validation, exact N-D coverage, complete DP
replica selection, explicit PP/EP ownership, candidate interval indexing,
logical-box intersection, canonical region lowering, executor grouping, and PP
routes. The module must not import Mooncake.

## Task 5: Red tests for Weight I/O

Create:

- `test/registered/unit/weight_transfer/test_provider.py`
- `test/registered/unit/weight_transfer/test_api.py`

Cover capability negotiation, lifecycle ordering, local runtime and storage
copies, transactional materialization, abort, known cancellation, completion
unknown, and zero provider calls on preflight failure.

## Task 6: Provider and storage APIs

Create:

- `python/sglang/srt/weight_transfer/provider.py`
- `python/sglang/srt/weight_transfer/api.py`

Implement structured capabilities and errors, local reference execution,
`load_weights`, and a separate `materialize_weights` transaction.

Add a framework checkpoint adapter that delegates named tensors to the existing
model loader. Direct checkpoint/OSS range planning requires an explicit
placement sidecar.

## Task 7: Mooncake adapter and live integration

Create:

- `python/sglang/srt/weight_transfer/mooncake.py`
- `test/registered/unit/weight_transfer/test_mooncake_provider.py`

Modify:

- `python/sglang/srt/model_loader/loader.py`
- relevant existing remote-instance tests

Translate bound SGLang regions to Mooncake executable regions lazily. Keep
Mooncake TE registrations, batching, tickets, and completion quarantine. Remove
Mooncake planner/binder imports from the placement-binding path.

## Task 8: Regression and E2E

Run:

- the new weight-transfer unit suite;
- the 90-test runtime-manifest suite;
- remote-instance lifecycle, heartbeat, transporter, and service benchmark
  suites;
- Mooncake SGLang contract tests;
- local byte-exact runtime/store E2E;
- GPU live G2G TP and cross-dimension cases;
- a serving cold-start versus weight-reuse comparison.

Record exact commits, hardware, model, commands, timings, output equality, and
unverified scale limits.

## Task 9: Performance

Measure separately:

- manifest export;
- planning;
- binding;
- backend lowering;
- data transfer;
- synchronization and post-load;
- end-to-end spawn-to-ready.

Use warmups and repeated measured runs. Validate bytes on every run and report
median, p95, coefficient of variation, logical throughput, wire throughput when
available, and source-serving latency impact.

## Task 10: Review and publish

Run formatters and focused static checks. Review fail-closed behavior,
operation-count bounds, optional dependency boundaries, and v3 regressions.

Set and verify the repository-local author and committer identity, create
logical commits, push only `vin/sglang-weight-v4` to the personal remote, and
verify the remote HEAD. Do not open a community pull request.
