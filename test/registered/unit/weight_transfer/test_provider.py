from __future__ import annotations

from dataclasses import replace
from inspect import Parameter, signature
from itertools import product
from math import prod

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
    load_weights,
    materialize_weights,
    prepare_weight_materialization,
)
from sglang.srt.weight_transfer.binding import project_source_bindings
from sglang.srt.weight_transfer.checkpoint_provider import (
    CheckpointStorageToRuntimeProvider,
)
from sglang.srt.weight_transfer.mooncake import MooncakeWeightTransferProvider
from sglang.srt.weight_transfer.mooncake_store import MooncakeWeightStoreProvider
from sglang.srt.weight_transfer.planner import (
    select_weight_storage_placements,
)
from sglang.srt.weight_transfer.provider import (
    LocalWeightBufferRegistry,
    LocalWeightTransferProvider,
    WeightMaterializeRequest,
    WeightProviderCapabilities,
    WeightStorageDestination,
    WeightTargetLoadMode,
    WeightTransferProvider,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def placements(
    side: str,
    *,
    global_shape: tuple[int, int],
    shard_dim: int,
    parts: int,
    dp: int = 1,
) -> tuple[WeightPlacementManifest, ...]:
    result = []
    for dp_rank in range(dp):
        for rank in range(parts):
            shape = list(global_shape)
            offset = [0, 0]
            shape[shard_dim] //= parts
            offset[shard_dim] = rank * shape[shard_dim]
            worker = f"{side}-d{dp_rank}-t{rank}"
            tensor = WeightPlacementTensor(
                placement_fragment_id=f"{worker}:fragment",
                tensor_id="weight",
                runtime_name="weight",
                aliases=("weight",),
                global_shape=global_shape,
                global_offset=tuple(offset),
                local_shape=tuple(shape),
                dtype="uint8",
                itemsize=1,
                partition_dim=shard_dim,
                shard_dims=(shard_dim,),
                layer_id=0,
                expert_id=None,
                layout_fingerprint="layout:v1",
                nbytes=prod(shape),
                byte_offset=0,
                rank=WeightParallelRank(dp=dp_rank, tp=rank),
            )
            tensors = (tensor,)
            result.append(
                WeightPlacementManifest(
                    model_id="model",
                    revision="revision",
                    placement_id=compute_weight_placement_id(tuple(tensors)),
                    tensors=tensors,
                )
            )
    return tuple(result)


def bindings(
    manifests: tuple[WeightPlacementManifest, ...],
    *,
    address_base: int,
) -> tuple[WeightRuntimeBindingManifest, ...]:
    result = []
    for index, manifest in enumerate(manifests):
        tensor = manifest.tensors[0]
        address = address_base + index * 0x1000
        result.append(
            WeightRuntimeBindingManifest(
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
                        device="cpu",
                        is_contiguous=True,
                        worker_id=manifest.placement_id,
                        endpoint=f"{manifest.placement_id}:1",
                    ),
                ),
            )
        )
    return tuple(result)


def payload(tensor: WeightPlacementTensor) -> bytearray:
    result = bytearray()
    for local in product(*(range(extent) for extent in tensor.local_shape)):
        global_coordinate = tuple(
            begin + coordinate
            for begin, coordinate in zip(
                tensor.global_offset,
                local,
                strict=True,
            )
        )
        result.append(
            global_coordinate[0] * tensor.global_shape[1] + global_coordinate[1]
        )
    return result


def register_runtime_buffers(
    registry: LocalWeightBufferRegistry,
    manifests: tuple[WeightPlacementManifest, ...],
    runtime_bindings: tuple[WeightRuntimeBindingManifest, ...],
    *,
    initialize: bool,
) -> dict[str, bytearray]:
    result = {}
    placement_by_id = {item.placement_id: item for item in manifests}
    for binding in runtime_bindings:
        tensor = placement_by_id[binding.placement_id].tensors[0]
        fragment = binding.fragments[0]
        buffer = payload(tensor) if initialize else bytearray(tensor.nbytes)
        registry.register_runtime(fragment.address, buffer)
        result[binding.placement_id] = buffer
    return result


def test_local_reference_provider_does_not_claim_bounded_execution() -> None:
    provider = LocalWeightTransferProvider(LocalWeightBufferRegistry())

    assert provider.probe(object()).supports_bounded_execution is False


@pytest.mark.parametrize(
    "provider_type",
    [
        WeightTransferProvider,
        LocalWeightTransferProvider,
        CheckpointStorageToRuntimeProvider,
        MooncakeWeightTransferProvider,
        MooncakeWeightStoreProvider,
    ],
)
def test_provider_synchronize_accepts_execution_context(provider_type) -> None:
    parameter = signature(provider_type.synchronize).parameters["execution_context"]

    assert parameter.kind is Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize(
    "name",
    [
        "supports_nd_regions",
        "supports_strided_regions",
        "supports_safe_cancel",
        "supports_completion_ticket",
        "supports_transactional_publish",
        "supports_bounded_execution",
    ],
)
def test_provider_capability_booleans_are_strict(name: str) -> None:
    values = {
        "provider": "strict",
        "load_profiles": frozenset(),
        "materialize_profiles": frozenset(),
        "supports_nd_regions": True,
        "supports_strided_regions": True,
        "supports_safe_cancel": False,
        "supports_completion_ticket": False,
        "supports_transactional_publish": True,
        "supports_bounded_execution": False,
    }
    values[name] = 1

    with pytest.raises(ValueError, match=f"{name} must be a boolean"):
        WeightProviderCapabilities(**values)


def test_provider_capability_profiles_are_immutable() -> None:
    load_profiles = {"runtime_to_runtime"}
    materialize_profiles = {"runtime_to_storage"}

    capabilities = WeightProviderCapabilities(
        provider="strict",
        load_profiles=load_profiles,
        materialize_profiles=materialize_profiles,
        supports_nd_regions=True,
        supports_strided_regions=True,
        supports_safe_cancel=False,
        supports_completion_ticket=False,
        supports_transactional_publish=True,
    )
    load_profiles.clear()
    materialize_profiles.clear()

    assert capabilities.load_profiles == frozenset({"runtime_to_runtime"})
    assert capabilities.materialize_profiles == frozenset({"runtime_to_storage"})


def test_materialize_request_normalizes_sequences_and_validates_locations() -> None:
    sources = placements("source", global_shape=(8, 8), shard_dim=0, parts=2)
    source_bindings = bindings(sources, address_base=0x100000)
    prepared = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=source_bindings,
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights:revision",
            object_prefix="weights/revision",
        ),
    )
    mutable_placements = list(prepared.source_placements)
    mutable_bindings = list(prepared.source_bindings)
    mutable_locations = list(prepared.source_locations)

    request = WeightMaterializeRequest(
        operation_id=prepared.operation_id,
        source_placements=mutable_placements,
        source_bindings=mutable_bindings,
        source_locations=mutable_locations,
        destination=prepared.destination,
        profile=prepared.profile,
    )
    mutable_placements.clear()
    mutable_bindings.clear()
    mutable_locations.clear()

    assert request.source_placements == prepared.source_placements
    assert request.source_bindings == prepared.source_bindings
    assert request.source_locations == prepared.source_locations
    assert isinstance(request.source_placements, tuple)
    assert isinstance(request.source_bindings, tuple)
    assert isinstance(request.source_locations, tuple)

    with pytest.raises(ValueError, match="source locations differ"):
        replace(
            request,
            source_locations=(
                replace(request.source_locations[0], address=0xDEADBEEF),
                *request.source_locations[1:],
            ),
        )


def test_local_provider_executes_cross_dimension_runtime_load() -> None:
    sources = placements("source", global_shape=(8, 8), shard_dim=0, parts=4)
    targets = placements("target", global_shape=(8, 8), shard_dim=1, parts=8)
    source_bindings = bindings(sources, address_base=0x100000)
    target_bindings = bindings(targets, address_base=0x200000)
    registry = LocalWeightBufferRegistry()
    register_runtime_buffers(
        registry,
        sources,
        source_bindings,
        initialize=True,
    )
    target_buffers = register_runtime_buffers(
        registry,
        targets,
        target_bindings,
        initialize=False,
    )
    provider = LocalWeightTransferProvider(registry)

    receipt = load_weights(
        source_placements=sources,
        source_bindings=source_bindings,
        target_placements=targets,
        target_bindings=target_bindings,
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
    )

    assert receipt.total_bytes == 64
    assert receipt.region_count == 32
    for manifest in targets:
        assert target_buffers[manifest.placement_id] == payload(manifest.tensors[0])


def test_materialize_keeps_fragments_separate_and_round_trips() -> None:
    sources = placements("source", global_shape=(8, 8), shard_dim=0, parts=4)
    targets = placements("target", global_shape=(8, 8), shard_dim=1, parts=2)
    source_bindings = bindings(sources, address_base=0x100000)
    target_bindings = bindings(targets, address_base=0x200000)
    registry = LocalWeightBufferRegistry()
    register_runtime_buffers(
        registry,
        sources,
        source_bindings,
        initialize=True,
    )
    target_buffers = register_runtime_buffers(
        registry,
        targets,
        target_bindings,
        initialize=False,
    )
    provider = LocalWeightTransferProvider(registry)

    materialized = materialize_weights(
        source_placements=sources,
        source_bindings=source_bindings,
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights:revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
    )

    assert materialized.total_bytes == 64
    assert materialized.stored_placements == tuple(
        sorted(sources, key=lambda item: item.placement_id)
    )
    assert len(materialized.storage_bindings) == 4
    assert sum(len(binding.fragments) for binding in materialized.storage_bindings) == 4
    assert len(registry.storage_objects) == 4
    assert {len(value) for value in registry.storage_objects.values()} == {16}

    loaded = load_weights(
        source_placements=sources,
        source_bindings=materialized.storage_bindings,
        target_placements=targets,
        target_bindings=target_bindings,
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
    )

    assert loaded.total_bytes == 64
    for manifest in targets:
        assert target_buffers[manifest.placement_id] == payload(manifest.tensors[0])


def test_materialize_writes_only_one_complete_dp_replica() -> None:
    sources = placements(
        "source",
        global_shape=(8, 8),
        shard_dim=0,
        parts=4,
        dp=2,
    )
    source_bindings = bindings(sources, address_base=0x100000)
    registry = LocalWeightBufferRegistry()
    register_runtime_buffers(
        registry,
        sources,
        source_bindings,
        initialize=True,
    )
    provider = LocalWeightTransferProvider(registry)

    materialized = materialize_weights(
        source_placements=sources,
        source_bindings=source_bindings,
        destination=WeightStorageDestination(
            provider="local",
            storage_id="weights:revision",
            object_prefix="weights/revision",
        ),
        provider=provider,
    )

    assert materialized.total_bytes == 64
    assert materialized.fragment_count == 4
    assert len(materialized.stored_placements) == 4
    assert {
        tensor.rank.dp
        for placement in materialized.stored_placements
        for tensor in placement.tensors
    } == {0}
    assert len(materialized.storage_bindings) == 4
    assert len(registry.storage_objects) == 4


def test_storage_selection_reidentifies_partial_ep_placements() -> None:
    placements_with_bindings = []
    for ep_rank in range(2):
        rank = WeightParallelRank(ep=ep_rank)
        shared = WeightPlacementTensor(
            placement_fragment_id=f"shared:ep{ep_rank}",
            tensor_id="shared.weight",
            runtime_name="shared.weight",
            aliases=("shared.weight",),
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
            rank=rank,
        )
        expert = WeightPlacementTensor(
            placement_fragment_id=f"expert:{ep_rank}",
            tensor_id=f"expert.{ep_rank}.weight",
            runtime_name=f"expert.{ep_rank}.weight",
            aliases=(f"expert.{ep_rank}.weight",),
            global_shape=(4,),
            global_offset=(0,),
            local_shape=(4,),
            dtype="uint8",
            itemsize=1,
            partition_dim=None,
            shard_dims=(),
            layer_id=0,
            expert_id=ep_rank,
            layout_fingerprint="layout:v1",
            nbytes=4,
            byte_offset=0,
            rank=rank,
        )
        tensors = (shared, expert)
        placement = WeightPlacementManifest(
            model_id="model",
            revision="revision",
            placement_id=compute_weight_placement_id(tuple(tensors)),
            tensors=tensors,
        )
        binding = WeightRuntimeBindingManifest(
            model_id=placement.model_id,
            revision=placement.revision,
            placement_id=placement.placement_id,
            instance_id=f"instance:ep{ep_rank}",
            generation=1,
            lease_id="lease:1",
            fragments=tuple(
                RuntimeWeightBinding(
                    placement_fragment_id=tensor.placement_fragment_id,
                    fragment_id=f"runtime:{tensor.placement_fragment_id}",
                    address=0x100000 + ep_rank * 0x1000 + index * 0x100,
                    nbytes=tensor.nbytes,
                    storage_offset=0,
                    device="cpu",
                    is_contiguous=True,
                    worker_id=f"worker:ep{ep_rank}",
                    endpoint=f"worker:ep{ep_rank}:1",
                )
                for index, tensor in enumerate(tensors)
            ),
        )
        placements_with_bindings.append((placement, binding))

    source_placements = tuple(item[0] for item in placements_with_bindings)
    source_bindings = tuple(item[1] for item in placements_with_bindings)
    selected = select_weight_storage_placements(source_placements)
    projected = project_source_bindings(selected, source_bindings)

    assert sum(len(item.tensors) for item in selected) == 3
    assert all(
        placement.placement_id == compute_weight_placement_id(placement.tensors)
        for placement in selected
    )
    assert {item.placement_id for item in projected} == {
        item.placement_id for item in selected
    }
    assert sum(len(item.fragments) for item in projected) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
