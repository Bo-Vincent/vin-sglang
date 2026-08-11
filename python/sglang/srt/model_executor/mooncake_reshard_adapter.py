"""SGLang-owned adapter for Mooncake canonical reshard contracts."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import msgspec

from sglang.srt.model_executor.weight_inventory_contracts import (
    LogicalParallelAxis,
    WeightParallelRank,
    WeightParallelTopology,
    WeightPlacementBindingInventories,
    WeightPlacementInventory,
    WeightPlacementInventoryFragment,
    WeightRuntimeBindingInventory,
    validate_remote_weight_lineage,
    validate_weight_topology_representability,
)

_MOONCAKE_MODULE = "mooncake.reshard.weight"
_REQUIRED_CAPABILITIES = (
    "placement_binding",
    "nd_logical_box",
    "dependent_axis_projection",
    "te_execution",
)
_REQUIRED_CONTRACTS = (
    "OwnershipAxis",
    "ParallelRank",
    "ParallelTopology",
    "PlacementFragment",
    "ReplicatedAxis",
    "RuntimeBindingFragment",
    "SplitAxis",
    "TensorDescriptor",
    "TopologyParticipant",
    "WeightPlacementManifest",
    "WeightPlacementPart",
    "WeightRuntimeBindingManifest",
)


def load_mooncake_reshard_contracts() -> Any:
    """Load Mooncake only when heterogeneous resharding is requested."""

    try:
        module = importlib.import_module(_MOONCAKE_MODULE)
    except Exception as error:
        raise RuntimeError(
            "Mooncake reshard support requires mooncake.reshard.weight"
        ) from error
    missing = [name for name in _REQUIRED_CONTRACTS if not hasattr(module, name)]
    supports = getattr(module, "supports_weight_reshard_capability", None)
    missing_capabilities = (
        list(_REQUIRED_CAPABILITIES)
        if not callable(supports)
        else [
            capability
            for capability in _REQUIRED_CAPABILITIES
            if supports(capability) is not True
        ]
    )
    if missing or missing_capabilities:
        details = []
        if missing:
            details.append(f"missing contracts: {', '.join(missing)}")
        if missing_capabilities:
            details.append("missing capabilities: " + ", ".join(missing_capabilities))
        detail = f"; {'; '.join(details)}" if details else ""
        raise RuntimeError(
            "Mooncake reshard planner and TransferEngine contracts are unavailable"
            + detail
        )
    return module


@dataclass(frozen=True)
class PlacementInventoryParticipant:
    """One SGLang worker's address-free placement inventory."""

    placement: WeightPlacementInventory

    def __post_init__(self) -> None:
        if not isinstance(self.placement, WeightPlacementInventory):
            raise TypeError("placement must be an SGLang WeightPlacementInventory")

    @property
    def participant_id(self) -> str:
        return self.placement.participant_id


def _coerce_placement(value: Any) -> WeightPlacementInventory:
    if isinstance(value, WeightPlacementInventory):
        return value
    try:
        return msgspec.convert(value, type=WeightPlacementInventory, strict=True)
    except Exception as error:
        raise ValueError("invalid SGLang weight placement inventory") from error


def _coerce_binding(value: Any) -> WeightRuntimeBindingInventory:
    if isinstance(value, WeightRuntimeBindingInventory):
        return value
    try:
        return msgspec.convert(
            value,
            type=WeightRuntimeBindingInventory,
            strict=True,
        )
    except Exception as error:
        raise ValueError("invalid SGLang weight runtime binding inventory") from error


def placement_participants_from_inventories(
    placements: Sequence[Any],
    bindings: Sequence[Any],
) -> tuple[PlacementInventoryParticipant, ...]:
    """Pair source wire inventories without asking Mooncake to inspect them."""

    if len(placements) != len(bindings) or not placements:
        raise ValueError("source placement and binding inventories must be paired")
    result = []
    for raw_placement, raw_binding in zip(placements, bindings):
        placement = _coerce_placement(raw_placement)
        binding = _coerce_binding(raw_binding)
        WeightPlacementBindingInventories(
            placement=placement,
            binding=binding,
        )
        result.append(PlacementInventoryParticipant(placement))
    return tuple(result)


class MooncakeCanonicalReshardAdapter:
    """Translate explicit SGLang metadata into canonical Mooncake contracts."""

    def __init__(self, *, contracts: Any | None = None) -> None:
        self.contracts = contracts or load_mooncake_reshard_contracts()

    def placement_manifest(
        self,
        participants: Sequence[PlacementInventoryParticipant],
    ) -> Any:
        items = tuple(participants)
        if not items:
            raise ValueError("placement participants must not be empty")
        first = items[0].placement
        identity = validate_remote_weight_lineage(
            model_id=first.model_id,
            revision=first.revision,
        )
        weight_generation = first.weight_generation
        topology_sizes = _topology_sizes(first.topology)
        topology_semantics = _topology_semantics(first.topology)
        participant_ids = [item.participant_id for item in items]
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("duplicate placement participant_id")
        ranks = [item.placement.topology.rank() for item in items]
        if len(ranks) != len(set(ranks)):
            raise ValueError("duplicate placement parallel rank")
        for item in items:
            validate_weight_topology_representability(item.placement.topology)
            if (item.placement.model_id, item.placement.revision) != identity:
                raise ValueError("placement participant identities differ")
            if item.placement.weight_generation != weight_generation:
                raise ValueError("placement participant weight generations differ")
            if _topology_sizes(item.placement.topology) != topology_sizes:
                raise ValueError("placement participant topologies differ")
            if _topology_semantics(item.placement.topology) != topology_semantics:
                raise ValueError("placement participant subgroup topologies differ")
        _validate_sglang_participant_coverage(items)
        _validate_cross_participant_descriptors(items)
        _validate_coupled_parallel_axes(items)
        _validate_moe_rank_decomposition(items)

        canonical_participants = tuple(
            self.contracts.TopologyParticipant(
                participant_id=item.participant_id,
                rank=self._rank(item.placement.topology.rank()),
            )
            for item in sorted(items, key=lambda value: value.participant_id)
        )
        topology = self.contracts.ParallelTopology(
            tp_size=topology_sizes[0],
            pp_size=topology_sizes[1],
            ep_size=topology_sizes[2],
            dp_size=topology_sizes[3],
            participants=canonical_participants,
        )
        placement_set_id = _placement_set_id(
            model_id=identity[0],
            revision=identity[1],
            weight_generation=weight_generation,
            participants=items,
        )
        parts = tuple(
            self._placement_part(
                item,
                resource_id=identity[0],
                revision=identity[1],
                weight_generation=weight_generation,
                placement_set_id=placement_set_id,
                topology_id=topology.topology_id,
            )
            for item in sorted(items, key=lambda value: value.participant_id)
        )
        return self.contracts.WeightPlacementManifest(
            resource_id=identity[0],
            revision=identity[1],
            weight_generation=weight_generation,
            placement_set_id=placement_set_id,
            topology=topology,
            parts=parts,
        )

    def source_placement_and_bindings(
        self,
        placement_inventories: Sequence[Any],
        binding_inventories: Sequence[Any],
    ) -> tuple[Any, tuple[Any, ...]]:
        """Convert a complete source snapshot after SGLang pairs its inventories."""

        participants = placement_participants_from_inventories(
            placement_inventories,
            binding_inventories,
        )
        bindings = tuple(_coerce_binding(item) for item in binding_inventories)
        placement = self.placement_manifest(participants)
        return placement, self.runtime_binding_manifests(
            bindings,
            placement=placement,
            placement_inventories=tuple(
                participant.placement for participant in participants
            ),
        )

    def gather_target_placement(
        self,
        local: PlacementInventoryParticipant,
        *,
        world_group: Any,
    ) -> Any:
        """All-gather address-free target parts before constructing the manifest."""

        world_size = getattr(world_group, "world_size", None)
        if type(world_size) is not int or world_size <= 0:
            raise ValueError(
                "target world_group must expose a positive integer world_size"
            )
        all_gather = getattr(world_group, "all_gather_object", None)
        if world_size > 1 and not callable(all_gather):
            raise ValueError(
                "multi-rank target world_group must expose all_gather_object()"
            )
        gathered = tuple(all_gather(local)) if world_size > 1 else (local,)
        if len(gathered) != world_size:
            raise ValueError("target placement all-gather returned the wrong size")
        if not all(
            isinstance(item, PlacementInventoryParticipant) for item in gathered
        ):
            raise ValueError("target placement all-gather returned invalid metadata")
        return self.placement_manifest(gathered)

    def runtime_binding_manifest(
        self,
        binding: Any,
        *,
        placement: Any,
        placement_inventory: Any,
    ) -> Any:
        source = _coerce_binding(binding)
        inventory = _coerce_placement(placement_inventory)
        WeightPlacementBindingInventories(
            placement=inventory,
            binding=source,
        )
        if source.model_id != placement.resource_id:
            raise ValueError("runtime binding resource_id differs from placement")
        if source.revision != placement.revision:
            raise ValueError("runtime binding revision differs from placement")
        canonical_participant_id = source.participant_id
        canonical_part = next(
            (
                part
                for part in placement.parts
                if part.participant_id == canonical_participant_id
            ),
            None,
        )
        if canonical_part is None:
            raise ValueError("runtime binding participant is absent from placement")
        expected_fragment_ids = {
            fragment.placement_fragment_id for fragment in canonical_part.fragments
        }
        actual_fragment_ids = {
            fragment.placement_fragment_id for fragment in source.fragments
        }
        if actual_fragment_ids != expected_fragment_ids:
            raise ValueError("runtime binding fragments differ from placement part")
        fragments = tuple(
            self.contracts.RuntimeBindingFragment(
                placement_fragment_id=fragment.placement_fragment_id,
                fragment_id=fragment.fragment_id,
                address=fragment.address,
                nbytes=fragment.nbytes,
                worker_id=fragment.worker_id,
                endpoint=fragment.endpoint,
                device=fragment.device,
                itemsize=fragment.itemsize,
                local_shape=fragment.local_shape,
                strides_bytes=fragment.strides_bytes,
                storage_address=fragment.storage_address,
                storage_nbytes=fragment.storage_nbytes,
                storage_offset_bytes=fragment.storage_offset_bytes,
            )
            for fragment in source.fragments
        )
        return self.contracts.WeightRuntimeBindingManifest(
            resource_id=source.model_id,
            placement_id=placement.placement_id,
            placement_digest=placement.digest,
            instance_id=source.instance_id,
            generation=source.generation,
            lease_id=source.lease_id,
            revision=source.revision,
            participant_id=canonical_participant_id,
            fragments=fragments,
        )

    def runtime_binding_manifests(
        self,
        bindings: Iterable[Any],
        *,
        placement: Any,
        placement_inventories: Iterable[Any],
    ) -> tuple[Any, ...]:
        binding_items = tuple(_coerce_binding(binding) for binding in bindings)
        inventory_items = tuple(
            _coerce_placement(inventory) for inventory in placement_inventories
        )
        if len(binding_items) != len(inventory_items) or not binding_items:
            raise ValueError("runtime binding inventories must be paired")
        participant_ids = [binding.participant_id for binding in binding_items]
        if len(participant_ids) != len(set(participant_ids)):
            raise ValueError("duplicate runtime binding participant_id")
        expected = {part.participant_id for part in placement.parts if part.fragments}
        if set(participant_ids) != expected:
            raise ValueError("runtime binding participant coverage differs")
        return tuple(
            self.runtime_binding_manifest(
                binding,
                placement=placement,
                placement_inventory=inventory,
            )
            for binding, inventory in zip(binding_items, inventory_items)
        )

    def _rank(self, rank: WeightParallelRank) -> Any:
        return self.contracts.ParallelRank(
            dp=rank.dp,
            tp=rank.tp,
            pp=rank.pp,
            ep=rank.ep,
        )

    def _placement_part(
        self,
        participant: PlacementInventoryParticipant,
        *,
        resource_id: str,
        revision: str,
        weight_generation: int,
        placement_set_id: str,
        topology_id: str,
    ) -> Any:
        placement = participant.placement
        topology_rank = placement.topology.rank()
        if any(item.rank != topology_rank for item in placement.fragments):
            raise ValueError("placement fragment rank differs from inventory topology")
        descriptors_by_id = {}
        descriptor_signatures = {}
        for item in placement.fragments:
            signature = (
                item.global_shape,
                item.dtype,
                item.itemsize,
                item.shard_dims,
                item.parallel_axes,
                item.layout_fingerprint,
                item.layer_id,
                item.expert_id,
            )
            previous = descriptor_signatures.setdefault(item.tensor_id, signature)
            if previous != signature:
                raise ValueError(
                    f"SGLang tensor descriptors disagree: {item.tensor_id}"
                )
            if item.tensor_id not in descriptors_by_id:
                descriptors_by_id[item.tensor_id] = self._descriptor(
                    item,
                )
        descriptors = tuple(
            descriptors_by_id[tensor_id] for tensor_id in sorted(descriptors_by_id)
        )
        fragments = tuple(
            self.contracts.PlacementFragment(
                tensor_id=item.tensor_id,
                global_offset=item.global_offset,
                local_shape=item.local_shape,
                nbytes=item.nbytes,
                rank=self._rank(item.rank),
                aliases=item.aliases,
                placement_fragment_id=item.placement_fragment_id,
            )
            for item in placement.fragments
        )
        return self.contracts.WeightPlacementPart(
            resource_id=resource_id,
            revision=revision,
            weight_generation=weight_generation,
            placement_set_id=placement_set_id,
            topology_id=topology_id,
            participant_id=participant.participant_id,
            rank=self._rank(placement.topology.rank()),
            tensors=descriptors,
            fragments=fragments,
        )

    def _descriptor(
        self,
        tensor: WeightPlacementInventoryFragment,
    ) -> Any:
        axes = [
            self._axis(axis) for axis in tensor.parallel_axes if axis.mode != "coupled"
        ]
        return self.contracts.TensorDescriptor(
            tensor_id=tensor.tensor_id,
            global_shape=tensor.global_shape,
            dtype=tensor.dtype,
            itemsize=tensor.itemsize,
            shard_dims=tensor.shard_dims,
            layout_fingerprint=tensor.layout_fingerprint,
            parallel_axes=tuple(axes),
            layer_id=tensor.layer_id,
            expert_id=tensor.expert_id,
        )

    def _axis(self, axis: LogicalParallelAxis) -> Any:
        if axis.mode == "split":
            return self.contracts.SplitAxis(kind=axis.kind, dim=axis.dim)
        if axis.mode == "replicated":
            return self.contracts.ReplicatedAxis(kind=axis.kind)
        if axis.mode == "ownership":
            return self.contracts.OwnershipAxis(kind=axis.kind)
        raise ValueError(f"unsupported logical parallel mode: {axis.mode}")


def _placement_set_id(
    *,
    model_id: str,
    revision: str,
    weight_generation: int,
    participants: Sequence[PlacementInventoryParticipant],
) -> str:
    topology = participants[0].placement.topology
    ranks = tuple(
        sorted(
            (
                item.participant_id,
                item.placement.topology.dp_rank,
                item.placement.topology.tp_rank,
                item.placement.topology.pp_rank,
                item.placement.topology.ep_rank,
            )
            for item in participants
        )
    )
    identity = (
        "sglang-weight-placement-set",
        model_id,
        revision,
        weight_generation,
        topology.dp_size,
        topology.tp_size,
        topology.pp_size,
        topology.ep_size,
        ranks,
    )
    digest = hashlib.sha256(msgspec.json.encode(identity)).hexdigest()
    return f"sglang-weight-placement-set:sha256:{digest}"


def _topology_sizes(topology: WeightParallelTopology) -> tuple[int, int, int, int]:
    return (
        topology.tp_size,
        topology.pp_size,
        topology.ep_size,
        topology.dp_size,
    )


def _topology_semantics(topology: WeightParallelTopology) -> tuple[int, ...]:
    return (
        topology.dp_size,
        topology.tp_size,
        topology.pp_size,
        topology.ep_size,
        topology.moe_tp_size,
        topology.attention_tp_size,
    )


def _validate_sglang_participant_coverage(
    participants: Sequence[PlacementInventoryParticipant],
) -> None:
    topology = participants[0].placement.topology
    expected = {
        (pp_rank, tp_rank)
        for pp_rank in range(topology.pp_size)
        for tp_rank in range(topology.tp_size)
    }
    actual = {
        (item.placement.topology.pp_rank, item.placement.topology.tp_rank)
        for item in participants
    }
    if actual != expected or len(participants) != len(expected):
        raise ValueError("placement participants do not cover the SGLang model world")
    for item in participants:
        current = item.placement.topology
        if current.dp_rank != 0:
            raise ValueError("SGLang weight placement does not support DP ranks")
        if current.moe_tp_size == 1 and current.moe_tp_rank != 0:
            raise ValueError("replicated MoE-TP coordinate must have rank zero")
        expected_ep_rank = current.tp_rank if current.ep_size > 1 else 0
        if current.ep_rank != expected_ep_rank:
            raise ValueError(
                "SGLang EP coordinate is inconsistent with the global TP rank"
            )


def _validate_cross_participant_descriptors(
    participants: Sequence[PlacementInventoryParticipant],
) -> None:
    descriptors: dict[str, tuple[Any, ...]] = {}
    for participant in participants:
        for fragment in participant.placement.fragments:
            signature = (
                fragment.aliases,
                fragment.global_shape,
                fragment.dtype,
                fragment.itemsize,
                fragment.shard_dims,
                fragment.parallel_axes,
                fragment.layer_id,
                fragment.expert_id,
                fragment.layout_fingerprint,
            )
            previous = descriptors.setdefault(fragment.tensor_id, signature)
            if previous != signature:
                raise ValueError(
                    "placement participants disagree on logical tensor descriptor: "
                    f"{fragment.tensor_id}"
                )


def _validate_coupled_parallel_axes(
    participants: Sequence[PlacementInventoryParticipant],
) -> None:
    topology = participants[0].placement.topology
    if topology.ep_size == 1:
        return
    co_mapped = (
        topology.moe_tp_size == 1
        and topology.tp_size == topology.ep_size
        and all(
            item.placement.topology.tp_rank == item.placement.topology.ep_rank
            for item in participants
        )
    )
    if not co_mapped:
        return

    axes_by_tensor = {
        fragment.tensor_id: {axis.kind: axis for axis in fragment.parallel_axes}
        for participant in participants
        for fragment in participant.placement.fragments
    }
    for tensor_id, axes in axes_by_tensor.items():
        coupled = [axis for axis in (axes["tp"], axes["ep"]) if axis.mode == "coupled"]
        if len(coupled) != 1:
            raise ValueError(
                "co-mapped SGLang TP/EP placement requires exactly one coupled "
                f"TP/EP axis: {tensor_id}"
            )


def _validate_moe_rank_decomposition(
    participants: Sequence[PlacementInventoryParticipant],
) -> None:
    has_ep_split = any(
        axis.kind == "ep" and axis.mode == "split"
        for participant in participants
        for fragment in participant.placement.fragments
        for axis in fragment.parallel_axes
    )
    if not has_ep_split:
        return
    topology = participants[0].placement.topology
    expected_tp_size = topology.ep_size * topology.moe_tp_size
    if topology.tp_size != expected_tp_size:
        raise ValueError(
            "SGLang MoE placement requires global TP = EP * MoE-TP; "
            f"got {topology.tp_size} != {topology.ep_size} * "
            f"{topology.moe_tp_size}"
        )


__all__ = [
    "MooncakeCanonicalReshardAdapter",
    "PlacementInventoryParticipant",
    "load_mooncake_reshard_contracts",
    "placement_participants_from_inventories",
]
