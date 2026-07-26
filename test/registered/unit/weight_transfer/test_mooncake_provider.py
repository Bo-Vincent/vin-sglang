from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.api import (
    execute_weight_load,
    prepare_weight_load,
    prepare_weight_load_from_plan,
)
from sglang.srt.weight_transfer.mooncake import (
    MooncakeWeightTransferCompletionUnknownError,
    MooncakeWeightTransferProvider,
)
from sglang.srt.weight_transfer.planner import (
    plan_weight_transfer_to_local_target,
)
from sglang.srt.weight_transfer.provider import WeightTargetLoadMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclass(frozen=True)
class FakeDescriptor:
    tensor_id: str
    global_shape: tuple[int, ...]
    dtype: str
    itemsize: int
    partition_dim: int | None
    layer_id: int | None
    expert_id: int | None
    layout_fingerprint: str
    shard_dims: tuple[int, ...]


@dataclass(frozen=True)
class FakePlacementFragment:
    placement_fragment_id: str
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    nbytes: int
    rank: WeightParallelRank
    aliases: tuple[str, ...]


class FakePlacementManifest:
    @classmethod
    def from_runtime_inventory(cls, inventory):
        descriptors = {}
        fragments = []
        for tensor in inventory.tensors:
            descriptors[tensor.tensor_id] = FakeDescriptor(
                tensor_id=tensor.tensor_id,
                global_shape=tuple(tensor.global_shape),
                dtype=tensor.dtype,
                itemsize=tensor.itemsize,
                partition_dim=tensor.partition_dim,
                layer_id=tensor.layer_id,
                expert_id=tensor.expert_id,
                layout_fingerprint=tensor.layout_fingerprint,
                shard_dims=tuple(tensor.shard_dims),
            )
            fragments.append(
                FakePlacementFragment(
                    placement_fragment_id=tensor.placement_fragment_id,
                    tensor_id=tensor.tensor_id,
                    global_offset=tuple(tensor.global_offset),
                    local_shape=tuple(tensor.local_shape),
                    nbytes=tensor.nbytes,
                    rank=tensor.rank,
                    aliases=tuple(tensor.aliases),
                )
            )
        return SimpleNamespace(
            model_id=inventory.model_id,
            revision=inventory.revision,
            placement_id=inventory.placement_id,
            tensors=tuple(descriptors.values()),
            fragments=tuple(fragments),
        )


class FakeRuntimeBindingManifest:
    @classmethod
    def from_runtime_inventory(cls, inventory):
        return SimpleNamespace(
            model_id=inventory.model_id,
            revision=inventory.revision,
            placement_id=inventory.placement_id,
            instance_id=inventory.instance_id,
            generation=inventory.generation,
            lease_id=inventory.lease_id,
            fragments=inventory.fragments,
        )


class FakeTransferCompletionUnknownError(RuntimeError):
    def __init__(self, message, pending_transfer_id="pending-1"):
        super().__init__(message)
        self.pending_transfer_id = pending_transfer_id


class FakeTransferEngineError(RuntimeError):
    pass


class AllowAllAttestor:
    def attest(self, _request) -> None:
        pass


ALLOW_ALL_ATTESTOR = AllowAllAttestor()


class FakeLogicalTransferPlan:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTransferRegion:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakePipelineRouteGroup:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeValue:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@dataclass(frozen=True)
class FakeParallelRank:
    dp: int = 0
    tp: int = 0
    pp: int = 0
    ep: int = 0


class FakeRuntimeLeaseSnapshot:
    @classmethod
    def from_fragment(cls, fragment):
        return SimpleNamespace(
            fragment_id=fragment.fragment_id,
            lease_generation=fragment.lease_generation,
        )


@dataclass(frozen=True)
class FakeRegistrationLease:
    fragment_id: str
    worker_id: str
    address: int
    nbytes: int
    lease_generation: int
    runtime_lease_id: str | None = None


def placement(
    side: str,
    *,
    shard_dim: int,
    rank: int,
    legacy_partition_only: bool = False,
) -> WeightPlacementManifest:
    global_shape = (4, 6, 8)
    local_shape = list(global_shape)
    global_offset = [0, 0, 0]
    local_shape[shard_dim] //= 2
    global_offset[shard_dim] = rank * local_shape[shard_dim]
    tensor = WeightPlacementTensor(
        placement_fragment_id=f"{side}:{rank}:fragment",
        tensor_id="experts.w1",
        runtime_name="experts.w1",
        aliases=("experts.w1",),
        global_shape=global_shape,
        global_offset=tuple(global_offset),
        local_shape=tuple(local_shape),
        dtype="bfloat16",
        itemsize=2,
        partition_dim=shard_dim,
        shard_dims=() if legacy_partition_only else (shard_dim,),
        layer_id=0,
        expert_id=None,
        layout_fingerprint="layout:v1",
        nbytes=2 * local_shape[0] * local_shape[1] * local_shape[2],
        byte_offset=0,
        rank=WeightParallelRank(
            ep=rank if shard_dim == 0 else 0,
            tp=rank if shard_dim != 0 else 0,
        ),
    )
    tensors = (tensor,)
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id(tuple(tensors)),
        tensors=tuple(tensors),
    )


def binding(
    manifest: WeightPlacementManifest,
    *,
    address: int,
) -> WeightRuntimeBindingManifest:
    tensor = manifest.tensors[0]
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id=f"instance:{manifest.placement_id}",
        generation=1,
        lease_id="lease:1",
        fragments=(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=address,
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id=manifest.placement_id,
                endpoint=f"{manifest.placement_id}:12345",
            ),
        ),
    )


def aliased_placements() -> tuple[
    WeightPlacementManifest,
    WeightPlacementManifest,
]:
    aliases = ("shared.weight", "shared.weight_alias")

    def tensor(
        *,
        fragment_id: str,
        runtime_name: str,
    ) -> WeightPlacementTensor:
        return WeightPlacementTensor(
            placement_fragment_id=fragment_id,
            tensor_id="shared.weight",
            runtime_name=runtime_name,
            aliases=aliases,
            global_shape=(4,),
            global_offset=(0,),
            local_shape=(4,),
            dtype="uint8",
            itemsize=1,
            partition_dim=None,
            shard_dims=(),
            layer_id=0,
            expert_id=None,
            layout_fingerprint="layout:v1",
            nbytes=4,
            byte_offset=0,
            rank=WeightParallelRank(),
        )

    source_tensors = (
        tensor(
            fragment_id="source:shared:fragment",
            runtime_name="shared.weight",
        ),
    )
    target_tensors = (
        tensor(
            fragment_id="target:shared:fragment",
            runtime_name="shared.weight",
        ),
        tensor(
            fragment_id="target:shared-alias:fragment",
            runtime_name="shared.weight_alias",
        ),
    )
    return (
        WeightPlacementManifest(
            model_id="model",
            revision="revision",
            placement_id=compute_weight_placement_id(source_tensors),
            tensors=source_tensors,
        ),
        WeightPlacementManifest(
            model_id="model",
            revision="revision",
            placement_id=compute_weight_placement_id(target_tensors),
            tensors=target_tensors,
        ),
    )


def aliased_target_binding(
    manifest: WeightPlacementManifest,
    *,
    address: int,
) -> WeightRuntimeBindingManifest:
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id="instance:target-alias",
        generation=1,
        lease_id="lease:1",
        fragments=tuple(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=("runtime:a" if index == 0 else "runtime:z"),
                address=address,
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id="target",
                endpoint="target:12345",
            )
            for index, tensor in enumerate(manifest.tensors)
        ),
    )


def mixed_tp_placement(
    side: str,
    *,
    rank: int,
    tp_size: int,
) -> WeightPlacementManifest:
    sharded_extent = 8 // tp_size
    tensors = (
        WeightPlacementTensor(
            placement_fragment_id=f"{side}:{rank}:sharded",
            tensor_id="sharded.weight",
            runtime_name="sharded.weight",
            aliases=("sharded.weight",),
            global_shape=(8, 4),
            global_offset=(rank * sharded_extent, 0),
            local_shape=(sharded_extent, 4),
            dtype="uint8",
            itemsize=1,
            partition_dim=0,
            shard_dims=(0,),
            layer_id=0,
            expert_id=None,
            layout_fingerprint="layout:v1",
            nbytes=sharded_extent * 4,
            byte_offset=0,
            rank=WeightParallelRank(tp=rank),
        ),
        WeightPlacementTensor(
            placement_fragment_id=f"{side}:{rank}:replicated",
            tensor_id="replicated.weight",
            runtime_name="replicated.weight",
            aliases=("replicated.weight",),
            global_shape=(4,),
            global_offset=(0,),
            local_shape=(4,),
            dtype="uint8",
            itemsize=1,
            partition_dim=None,
            shard_dims=(),
            layer_id=0,
            expert_id=None,
            layout_fingerprint="layout:v1",
            nbytes=4,
            byte_offset=0,
            rank=WeightParallelRank(tp=rank),
        ),
    )
    return WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id(tensors),
        tensors=tensors,
    )


def multi_tensor_binding(
    manifest: WeightPlacementManifest,
    *,
    address: int,
) -> WeightRuntimeBindingManifest:
    return WeightRuntimeBindingManifest(
        model_id=manifest.model_id,
        revision=manifest.revision,
        placement_id=manifest.placement_id,
        instance_id=f"instance:{manifest.placement_id}",
        generation=1,
        lease_id=f"lease:{manifest.placement_id}",
        fragments=tuple(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=address + index * 0x100,
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id=manifest.placement_id,
                endpoint=f"{manifest.placement_id}:12345",
            )
            for index, tensor in enumerate(manifest.tensors)
        ),
    )


def fake_backend(calls, *, completion_unknown=False):
    class FakeReader:
        def __init__(self, engine, **kwargs):
            calls["reader"] = (engine, kwargs)

        def execute(self, plan, sources, target, **kwargs):
            calls["execute"] = (plan, sources, target, kwargs)
            if completion_unknown:
                raise FakeTransferCompletionUnknownError("unknown")
            return (
                SimpleNamespace(
                    nbytes=sum(
                        operation.inner_bytes * prod(operation.outer_loop_counts)
                        for operation in plan.operations
                    ),
                    operation_count=sum(
                        prod(operation.outer_loop_counts)
                        for operation in plan.operations
                    ),
                ),
            )

        def drain_pending_transfer(self, pending_transfer_id, *, timeout_ms):
            calls["drain"] = (pending_transfer_id, timeout_ms)
            return "FAILED_DRAINED"

    def forbidden_bind(*args, **kwargs):
        del args, kwargs
        raise AssertionError("SGLang provider must not invoke Mooncake binding")

    return SimpleNamespace(
        ExecutorTransferPlan=FakeValue,
        ParallelRank=FakeParallelRank,
        PipelineRouteGroup=FakePipelineRouteGroup,
        RuntimeFragment=FakeValue,
        RuntimeLeaseSnapshot=FakeRuntimeLeaseSnapshot,
        RuntimeManifest=FakeValue,
        TensorDescriptor=FakeDescriptor,
        TransferPlan=FakeValue,
        TransferRegion=FakeTransferRegion,
        MemoryRegistrationLease=FakeRegistrationLease,
        bind_logical_transfer_plan=forbidden_bind,
        bind_runtime_manifest=forbidden_bind,
        MooncakeTransferEngineReader=FakeReader,
        TransferCompletionUnknownError=FakeTransferCompletionUnknownError,
        TransferEngineError=FakeTransferEngineError,
    )


def copying_backend(memory: dict[int, bytearray], calls: dict):
    backend = fake_backend(calls)

    class CopyingReader:
        def __init__(self, engine, **kwargs):
            calls["reader"] = (engine, kwargs)

        def execute(self, plan, sources, target, **kwargs):
            calls["execute"] = (plan, sources, target, kwargs)
            transferred = 0
            operation_count = 0
            for operation in plan.operations:
                source = memory[operation.source.address]
                destination = memory[operation.target.address]
                for outer_index in product(
                    *(range(count) for count in operation.outer_loop_counts)
                ):
                    source_offset = operation.source_base_offset + sum(
                        index * stride
                        for index, stride in zip(
                            outer_index,
                            operation.source_strides,
                            strict=True,
                        )
                    )
                    target_offset = operation.target_base_offset + sum(
                        index * stride
                        for index, stride in zip(
                            outer_index,
                            operation.target_strides,
                            strict=True,
                        )
                    )
                    destination[
                        target_offset : target_offset + operation.inner_bytes
                    ] = source[source_offset : source_offset + operation.inner_bytes]
                    transferred += operation.inner_bytes
                    operation_count += 1
            return (
                SimpleNamespace(
                    nbytes=transferred,
                    operation_count=operation_count,
                ),
            )

    backend.MooncakeTransferEngineReader = CopyingReader
    return backend


def _logical_fragment_payload(
    manifest: WeightPlacementManifest,
) -> bytes:
    tensor = manifest.tensors[0]
    payload = bytearray()
    for local_index in product(*(range(extent) for extent in tensor.local_shape)):
        global_index = tuple(
            offset + index
            for offset, index in zip(
                tensor.global_offset,
                local_index,
                strict=True,
            )
        )
        linear_index = 0
        for index, extent in zip(
            global_index,
            tensor.global_shape,
            strict=True,
        ):
            linear_index = linear_index * extent + index
        payload.extend(linear_index.to_bytes(tensor.itemsize, "little"))
    return bytes(payload)


def test_mooncake_provider_translates_native_regions_without_replanning(
    monkeypatch,
) -> None:
    sources = tuple(placement("source", shard_dim=0, rank=rank) for rank in range(2))
    targets = tuple(placement("target", shard_dim=2, rank=rank) for rank in range(2))
    source_bindings = tuple(
        binding(manifest, address=0x10000 + index * 0x1000)
        for index, manifest in enumerate(sources)
    )
    target_bindings = tuple(
        binding(manifest, address=0x20000 + index * 0x1000)
        for index, manifest in enumerate(targets)
    )
    request = prepare_weight_load(
        source_placements=sources,
        source_bindings=source_bindings,
        target_placements=targets,
        target_bindings=target_bindings,
    )
    calls = {}
    backend = fake_backend(calls)
    monkeypatch.setattr(
        MooncakeWeightTransferProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    provider = MooncakeWeightTransferProvider(
        "engine",
        source_registrations=("source-mr",),
        target_registrations=("target-mr",),
    )

    with pytest.raises(
        Exception,
        match="one local target manifest",
    ):
        execute_weight_load(
            request,
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            attestor=ALLOW_ALL_ATTESTOR,
        )

    local_request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(sources, targets[0]),
        source_bindings=source_bindings,
        target_bindings=(target_bindings[0],),
    )
    receipt = execute_weight_load(
        local_request,
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
        attestor=ALLOW_ALL_ATTESTOR,
    )

    assert receipt.total_bytes == 192
    assert receipt.region_count == len(local_request.plan.regions)
    executable_plan = calls["execute"][0]
    assert len(executable_plan.operations) == len(local_request.plan.regions)
    assert all(
        operation.overlap_shape == (2, 6, 4) for operation in executable_plan.operations
    )
    assert len(executable_plan.source_executors) == 2
    assert len(executable_plan.target_executors) == 1
    assert "logical" not in calls
    assert "bindings" not in calls
    _, _, _, execute_kwargs = calls["execute"]
    assert execute_kwargs == {
        "source_pre_registered": True,
        "source_registrations": ("source-mr",),
        "target_pre_registered": True,
        "target_registrations": ("target-mr",),
    }


def test_mooncake_provider_executes_cross_dim_regions_byte_exactly(
    monkeypatch,
) -> None:
    sources = tuple(placement("source", shard_dim=0, rank=rank) for rank in range(2))
    target = placement("target", shard_dim=2, rank=0)
    source_bindings = tuple(
        binding(manifest, address=0x10000 + index * 0x1000)
        for index, manifest in enumerate(sources)
    )
    target_binding = binding(target, address=0x20000)
    memory = {
        source_binding.fragments[0].address: bytearray(
            _logical_fragment_payload(source)
        )
        for source, source_binding in zip(
            sources,
            source_bindings,
            strict=True,
        )
    }
    memory[target_binding.fragments[0].address] = bytearray(target.tensors[0].nbytes)
    calls = {}
    backend = copying_backend(memory, calls)
    monkeypatch.setattr(
        MooncakeWeightTransferProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(sources, target),
        source_bindings=source_bindings,
        target_bindings=(target_binding,),
    )

    receipt = execute_weight_load(
        request,
        provider=MooncakeWeightTransferProvider("in-memory-te"),
        target_mode=WeightTargetLoadMode.COLD_START,
        attestor=ALLOW_ALL_ATTESTOR,
    )

    assert receipt.total_bytes == target.tensors[0].nbytes
    assert bytes(memory[target_binding.fragments[0].address]) == (
        _logical_fragment_payload(target)
    )
    assert calls["execute"][0].operations


def test_mooncake_provider_exposes_pending_ticket_for_drain(monkeypatch) -> None:
    source = placement("source", shard_dim=0, rank=0)
    source_peer = placement("source", shard_dim=0, rank=1)
    target = placement("target", shard_dim=2, rank=0)
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(
            (source, source_peer),
            target,
        ),
        source_bindings=(
            binding(source, address=0x10000),
            binding(source_peer, address=0x11000),
        ),
        target_bindings=(binding(target, address=0x20000),),
    )
    calls = {}
    backend = fake_backend(calls, completion_unknown=True)
    monkeypatch.setattr(
        MooncakeWeightTransferProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    provider = MooncakeWeightTransferProvider("engine")

    with pytest.raises(MooncakeWeightTransferCompletionUnknownError) as raised:
        execute_weight_load(
            request,
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert raised.value.pending_transfer_id == "pending-1"
    assert provider.drain_pending_transfer("pending-1", timeout_ms=10) == (
        "FAILED_DRAINED"
    )
    assert calls["drain"] == ("pending-1", 10)


def test_mooncake_provider_derives_registration_leases_from_bound_plan(
    monkeypatch,
) -> None:
    sources = tuple(placement("source", shard_dim=0, rank=rank) for rank in range(2))
    target = placement("target", shard_dim=2, rank=0)
    source_bindings = tuple(
        binding(manifest, address=0x10000 + index * 0x1000)
        for index, manifest in enumerate(sources)
    )
    target_binding = binding(target, address=0x20000)
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(sources, target),
        source_bindings=source_bindings,
        target_bindings=(target_binding,),
    )
    calls = {}
    backend = fake_backend(calls)
    monkeypatch.setattr(
        MooncakeWeightTransferProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )

    execute_weight_load(
        request,
        provider=MooncakeWeightTransferProvider("engine"),
        target_mode=WeightTargetLoadMode.COLD_START,
        attestor=ALLOW_ALL_ATTESTOR,
    )

    _, _, _, execute_kwargs = calls["execute"]
    source_registrations = execute_kwargs["source_registrations"]
    target_registrations = execute_kwargs["target_registrations"]
    assert {
        (
            registration.fragment_id,
            registration.worker_id,
            registration.address,
            registration.nbytes,
            registration.lease_generation,
            registration.runtime_lease_id,
        )
        for registration in source_registrations
    } == {
        (
            binding.fragments[0].fragment_id,
            binding.fragments[0].worker_id,
            binding.fragments[0].address,
            binding.fragments[0].nbytes,
            binding.generation,
            binding.lease_id,
        )
        for binding in source_bindings
    }
    assert target_registrations == (
        FakeRegistrationLease(
            fragment_id=target_binding.fragments[0].fragment_id,
            worker_id=target_binding.fragments[0].worker_id,
            address=target_binding.fragments[0].address,
            nbytes=target_binding.fragments[0].nbytes,
            lease_generation=target_binding.generation,
            runtime_lease_id=target_binding.lease_id,
        ),
    )


def test_real_mooncake_contract_accepts_direct_bound_plan() -> None:
    pytest.importorskip("mooncake.weight_transfer")
    sources = tuple(placement("source", shard_dim=0, rank=rank) for rank in range(2))
    target = placement("target", shard_dim=2, rank=0)
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(sources, target),
        source_bindings=tuple(
            binding(source, address=0x10000 + rank * 0x1000)
            for rank, source in enumerate(sources)
        ),
        target_bindings=(binding(target, address=0x20000),),
    )

    provider = MooncakeWeightTransferProvider("contract-only-engine")
    prepared = provider.prepare(request)

    assert prepared.executable_plan.total_bytes == request.plan.total_bytes
    assert len(prepared.executable_plan.operations) == len(request.plan.regions)
    assert len(prepared.source_manifests) == 2
    assert prepared.target_manifest.placement_id == target.placement_id


def test_real_mooncake_contract_normalizes_legacy_partition_dim() -> None:
    pytest.importorskip("mooncake.weight_transfer")
    source = placement(
        "source",
        shard_dim=0,
        rank=0,
        legacy_partition_only=True,
    )
    source_peer = placement(
        "source",
        shard_dim=0,
        rank=1,
        legacy_partition_only=True,
    )
    target = placement("target", shard_dim=2, rank=0)
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(
            (source, source_peer),
            target,
        ),
        source_bindings=(
            binding(source, address=0x10000),
            binding(source_peer, address=0x11000),
        ),
        target_bindings=(binding(target, address=0x20000),),
    )

    prepared = MooncakeWeightTransferProvider("contract-only-engine").prepare(request)

    assert all(
        descriptor.effective_shard_dims == (0,)
        for manifest in prepared.source_manifests
        for descriptor in manifest.tensors
    )


def test_real_mooncake_contract_accepts_alias_deduplicated_bound_plan() -> None:
    pytest.importorskip("mooncake.weight_transfer")
    from mooncake.weight_transfer.planner import resolve_executor_plans

    source, target = aliased_placements()
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target((source,), target),
        source_bindings=(binding(source, address=0x10000),),
        target_bindings=(aliased_target_binding(target, address=0x20000),),
    )

    assert len(request.plan.logical_plan.regions) == 2
    assert len(request.plan.regions) == 1

    prepared = MooncakeWeightTransferProvider("contract-only-engine").prepare(request)

    assert len(prepared.executable_plan.operations) == 1
    assert all(
        index == 0
        for executor in (
            *prepared.executable_plan.source_executors,
            *prepared.executable_plan.target_executors,
        )
        for index in executor.operation_indices
    )
    assert tuple(
        route.operation_indices for route in prepared.executable_plan.pipeline_routes
    ) == ((0,),)
    assert (
        len(
            resolve_executor_plans(
                prepared.executable_plan,
                prepared.target_manifest,
                "target",
            )
        )
        == 1
    )
    assert (
        len(
            resolve_executor_plans(
                prepared.executable_plan,
                prepared.source_manifests[0],
                "source",
            )
        )
        == 1
    )


@pytest.mark.parametrize("target_rank", (0, 2))
def test_real_mooncake_contract_projects_local_source_snapshots(
    target_rank: int,
) -> None:
    pytest.importorskip("mooncake.weight_transfer")
    from mooncake.weight_transfer.planner import resolve_executor_plans

    sources = tuple(
        mixed_tp_placement("source", rank=rank, tp_size=2) for rank in range(2)
    )
    target = mixed_tp_placement("target", rank=target_rank, tp_size=4)
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(sources, target),
        source_bindings=tuple(
            multi_tensor_binding(source, address=0x10000 + rank * 0x1000)
            for rank, source in enumerate(sources)
        ),
        target_bindings=(multi_tensor_binding(target, address=0x20000),),
    )

    prepared = MooncakeWeightTransferProvider("contract-only-engine").prepare(request)

    expected_instances = {
        executor.instance_id for executor in prepared.executable_plan.source_executors
    }
    assert {
        manifest.instance_id for manifest in prepared.source_manifests
    } == expected_instances
    for manifest in prepared.source_manifests:
        executors = resolve_executor_plans(
            prepared.executable_plan,
            manifest,
            "source",
        )
        assert len(executors) == 1
        assert executors[0].fragment_ids == tuple(
            fragment.fragment_id
            for fragment in sorted(
                manifest.fragments,
                key=lambda item: item.fragment_id,
            )
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
