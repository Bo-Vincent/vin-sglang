# SGLang native heterogeneous weight conversion

## Goal

Implement the first three phases of the heterogeneous weight conversion RFC in
SGLang without making Mooncake, NCCL M2N, or a model naming convention part of
the core contract.

The implementation must:

- keep runtime manifests as the source of truth for model semantics and
  physical locations;
- plan TP, PP, EP, and DP changes jointly with address-free N-D logical boxes;
- bind a logical plan to the current runtime or storage locations only after
  planning;
- execute the same plan through pluggable providers;
- expose separate APIs for loading weights into a runtime and materializing
  weights into persistent storage;
- preserve the v3 snapshot lease, generation, bounds, readiness, and
  completion-unknown behavior.

## Ownership

SGLang owns:

- model adapters and canonical tensor identity;
- `WeightPlacementManifest` and `WeightRuntimeBindingManifest`;
- the canonical N-D planner and validation rules;
- target allocation, post-load processing, world barriers, and activation;
- provider negotiation and structured operation results.

Providers own:

- physical transfer submission and completion;
- backend-specific batching, scatter, or strided lowering;
- persistent-object writes and atomic publication where applicable;
- backend tickets, cancellation, and drain behavior.

Mooncake remains an optional TE/Store provider. It is not imported by the
planner or core contracts.

## Package layout

```text
python/sglang/srt/weight_transfer/
  contracts.py       logical, runtime, and storage contracts
  planner.py         address-free N-D planning
  binding.py         runtime/storage binding and physical validation
  provider.py        capabilities, lifecycle, and local reference provider
  api.py             load_weights() and materialize_weights()
  mooncake.py        optional Mooncake TE/Store adapter
```

Existing manifests in
`sglang.srt.model_executor.weight_runtime_manifest` remain the public runtime
source. The new package consumes them directly.

## Core contracts

### Placement

`WeightPlacementManifest` remains address-free. Each fragment carries:

- stable tensor and placement-fragment IDs;
- global shape, local logical box, dtype, and item size;
- layer/expert semantics and layout fingerprint;
- TP/PP/EP/DP rank metadata;
- legacy `partition_dim` and canonical `shard_dims`.

Legacy `partition_dim` is normalized to a one-dimensional shard set. Source and
target shard dimensions may differ.

### Binding

`WeightRuntimeBindingManifest` binds placement fragments to a concrete runtime
generation, lease, address range, worker, and endpoint.

`WeightStorageBindingManifest` binds the same placement fragments to provider
objects and byte ranges. Store, L3, checkpoint, and OSS are represented by
provider-neutral object references. Checkpoint data may use direct range
planning only when a compatible semantic placement sidecar exists.

### Logical plan

`LogicalWeightTransferPlan` contains no runtime address, object key, lease, or
backend handle. Each region contains:

- source and target placement-fragment IDs;
- overlap offset and shape;
- source and target base offsets;
- maximal common contiguous `inner_bytes`;
- outer loop counts;
- independent source and target strides.

Regions are grouped by `(src_pp, dst_pp)`. The planner emits one region per
logical-box intersection and never expands rows or elements into operations.

### Bound plan

The binder attaches current runtime or storage locations to a logical plan and
validates:

- model, revision, placement ID, generation, and lease identity;
- fragment completeness and exact placement-to-binding correspondence;
- source and target address/object bounds;
- complete target coverage;
- no overlapping target writes;
- deterministic alias deduplication only when source and target byte geometry
  is identical.

## Joint planning

TP and EP are represented by N-D logical boxes. PP is explicit tensor/layer
ownership routing. DP selects a complete source replica and does not shard
weight contents.

For each source DP rank, the planner first proves that every tensor has a
complete owner. An owner is `(pp, ep)` only for tensors with an explicit
`expert_id`; packed expert tensors use the expert logical axis and do not treat
the EP rank as ownership. Target DP rank `d` selects complete source replica
`source_dps[d % len(source_dps)]`.

For every target fragment, the planner:

1. selects the source DP replica and explicit tensor owner;
2. queries source fragments of the same tensor whose logical boxes may overlap;
3. computes N-D intersections;
4. proves that the intersections cover the target fragment exactly;
5. derives the canonical strided region geometry.

There is no intermediate TP, PP, or EP layout and no all-gather buffer.

## Weight I/O

Two facade operations are exposed:

```python
load_weights(source, target_runtime, executor, options) -> LoadReceipt
materialize_weights(source, destination, writer, options) -> MaterializeReceipt
```

`load_weights` plans and binds source data to already allocated final target
buffers. Runtime-to-runtime and semantic Store-to-runtime paths use the N-D
logical plan.

`materialize_weights` creates a separate transactional write request. It does
not pretend that a storage upload is a target runtime reshard. A writer must
prepare the destination, write all fragments, verify durable receipts, and
publish or abort atomically.

Checkpoint and OSS paths without a semantic sidecar continue through the
framework named-tensor loader so that model-specific quantization, fusion,
packing, swizzling, and post-load processing remain correct.

## Provider lifecycle

```text
probe -> prepare -> submit -> wait/cancel -> synchronize -> release
```

Capability negotiation completes before the first target write. Providers
declare supported source/target kinds, N-D or strided support, limits, safe
cancellation, ticket/drain behavior, and transactional publication.

Known failures are cancelled and released. Completion-unknown failures retain
the target buffers, bindings, registrations, source lease, and provider ticket
until a later drain establishes a terminal state.

The local reference provider executes runtime and storage ranges against
registered byte buffers. It is used for deterministic correctness tests and is
not a production transport.

## Integration

The existing remote-instance path keeps acquisition, heartbeat, target-world
readiness, post-load, quarantine, and release coordination. Its
`placement_binding_v1` branch switches from the Mooncake planner/binder to the
SGLang planner/binder, then passes the bound plan to the Mooncake provider.

The `runtime_v1` path remains available through a compatibility adapter.
Ordinary checkpoint loading and non-heterogeneous remote updates are unchanged.

## Verification

Required correctness cases:

- TP4 to TP8 and TP8 to TP4;
- PP2 to PP4 and PP4 to PP2;
- EP8 to EP2 and EP2 to EP8;
- `[experts, out, in]` dim0 to dim1 and dim0 to dim2;
- TP4/PP2/EP8/DP2 to TP8/PP4/EP2/DP4;
- runtime, storage, checkpoint-sidecar, and legacy manifest sources;
- byte-exact target contents, exact coverage, and no overlapping writes;
- bounded region count under cross-dimension stress.

Required lifecycle cases:

- stale generation, revoked lease, placement mismatch, address overflow, and
  capability mismatch fail before provider submission;
- known provider failure cancels and cleans up;
- completion unknown does not release protected resources;
- storage materialization publishes only after all writes are durable and
  aborts known failures.

Serving E2E compares cold-start and reuse outputs, token IDs, finish reason, and
logprobs. Performance reports planning, binding, transfer, synchronization,
post-load, and end-to-end spawn-to-ready time separately.
