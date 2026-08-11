from __future__ import annotations

import hashlib
from math import prod
from typing import Any, Protocol

import msgspec

DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC = 300
MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC = 30
MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC = 3600


def validate_remote_instance_weight_transfer_lease_timeout(
    lease_timeout_sec: int,
) -> int:
    if isinstance(lease_timeout_sec, bool) or not isinstance(lease_timeout_sec, int):
        raise ValueError("lease_timeout_sec must be an integer")
    if not (
        MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
        <= lease_timeout_sec
        <= MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
    ):
        raise ValueError(
            "lease_timeout_sec must be between "
            f"{MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC} and "
            f"{MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC}"
        )
    return lease_timeout_sec


def validate_remote_weight_lineage(
    *,
    model_id: str,
    revision: str | None,
) -> tuple[str, str]:
    """Validate the stable lineage shared by one runtime weight generation.

    SGLang does not hash model bytes here. The revision distinguishes restart or
    checkpoint lineages where the runtime ``weight_generation`` counter may begin
    again. The complete content identity is the later
    ``(model_id, revision, weight_generation)`` tuple; the revision alone is not a
    byte-content proof.
    """

    if (
        type(model_id) is not str
        or not model_id.strip()
        or type(revision) is not str
        or not revision.strip()
        or revision.strip().lower() == "default"
    ):
        raise ValueError(
            "remote weight reshard requires a model ID and an explicit, "
            "non-placeholder content-lineage revision"
        )
    return model_id, revision


def validate_remote_weight_source_identity(
    *,
    requested_model_id: str,
    requested_revision: str | None,
    loaded_model_id: str,
    loaded_revision: str | None,
) -> tuple[str, str]:
    """Require a remote export request to match the source's loaded content."""

    requested = validate_remote_weight_lineage(
        model_id=requested_model_id,
        revision=requested_revision,
    )
    try:
        loaded = validate_remote_weight_lineage(
            model_id=loaded_model_id,
            revision=loaded_revision,
        )
    except ValueError as error:
        raise WeightInventoryError(
            "the source server has no explicit content-lineage revision"
        ) from error
    if requested != loaded:
        raise WeightInventoryError(
            "remote weight request identity does not match the source model"
        )
    return requested


class WeightInventoryError(RuntimeError):
    pass


class WeightParallelRank(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    dp: int = 0
    tp: int = 0
    pp: int = 0
    ep: int = 0

    def __post_init__(self) -> None:
        values = (self.dp, self.tp, self.pp, self.ep)
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("parallel ranks must be non-negative integers")


class WeightParallelTopology(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    dp_rank: int = 0
    dp_size: int = 1
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1
    moe_tp_rank: int = 0
    moe_tp_size: int = 1
    attention_tp_rank: int = 0
    attention_tp_size: int = 1

    def __post_init__(self) -> None:
        ranks = (
            self.dp_rank,
            self.tp_rank,
            self.pp_rank,
            self.ep_rank,
            self.moe_tp_rank,
            self.attention_tp_rank,
        )
        sizes = (
            self.dp_size,
            self.tp_size,
            self.pp_size,
            self.ep_size,
            self.moe_tp_size,
            self.attention_tp_size,
        )
        if any(type(rank) is not int or rank < 0 for rank in ranks):
            raise ValueError("parallel ranks must be non-negative integers")
        if any(type(size) is not int or size <= 0 for size in sizes):
            raise ValueError("parallel sizes must be positive integers")
        if any(rank >= size for rank, size in zip(ranks, sizes)):
            raise ValueError("parallel rank is outside its topology")

    def rank(self) -> WeightParallelRank:
        return WeightParallelRank(
            dp=self.dp_rank,
            tp=self.tp_rank,
            pp=self.pp_rank,
            ep=self.ep_rank,
        )


class LogicalParallelAxis(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """SGLang-owned explicit parallel semantics for one logical tensor."""

    kind: str
    mode: str
    dim: int | None = None
    coupled_to: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("dp", "tp", "pp", "ep"):
            raise ValueError(f"unsupported logical parallel axis: {self.kind}")
        if self.mode not in ("split", "replicated", "ownership", "coupled"):
            raise ValueError(f"unsupported logical parallel mode: {self.mode}")
        if self.mode == "split":
            if type(self.dim) is not int or self.dim < 0:
                raise ValueError("split logical parallel axis requires a dimension")
        elif self.dim is not None:
            raise ValueError("non-split logical parallel axis cannot have a dimension")
        if self.mode == "coupled":
            if {self.kind, self.coupled_to} != {"tp", "ep"}:
                raise ValueError("only TP and EP axes can be explicitly coupled")
        elif self.coupled_to is not None:
            raise ValueError("only a coupled axis can name coupled_to")


def validate_weight_topology_representability(
    topology: WeightParallelTopology,
) -> None:
    """Reject subgroup coordinates that one canonical participant rank cannot encode."""

    if topology.dp_size != 1:
        raise WeightInventoryError(
            "weight inventories do not yet support DP or MoE-DP participants"
        )
    if (
        topology.attention_tp_size != topology.tp_size
        or topology.attention_tp_rank != topology.tp_rank
    ):
        raise WeightInventoryError(
            "attention TP subgroup coordinates must match global TP coordinates"
        )
    if topology.ep_size > 1 and topology.moe_tp_size > 1:
        raise WeightInventoryError(
            "hybrid EP and MoE-TP cannot be represented by one participant rank"
        )
    if topology.ep_size == 1 and topology.moe_tp_size not in (
        1,
        topology.tp_size,
    ):
        raise WeightInventoryError(
            "MoE-TP coordinates must be replicated or match global TP coordinates"
        )
    if (
        topology.moe_tp_size == topology.tp_size
        and topology.moe_tp_rank != topology.tp_rank
    ):
        raise WeightInventoryError(
            "MoE-TP rank must match global TP rank when MoE tensors are TP split"
        )


class LogicalTensorView(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Private SGLang view linking model semantics to local parameter storage."""

    tensor_id: str
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    byte_offset: int
    layer_id: int | None
    expert_id: int | None
    layout_fingerprint: str
    shard_dims: tuple[int, ...]
    parallel_axes: tuple[LogicalParallelAxis, ...]
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.aliases and (
            tuple(sorted(self.aliases)) != self.aliases
            or len(self.aliases) < 2
            or len(set(self.aliases)) != len(self.aliases)
            or self.tensor_id not in self.aliases
            or any(type(alias) is not str or not alias for alias in self.aliases)
        ):
            raise ValueError("aliases must be a canonical logical tensor ID group")


class WeightPlacementInventoryFragment(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Address-free logical placement facts produced by one SGLang worker."""

    placement_fragment_id: str
    tensor_id: str
    aliases: tuple[str, ...]
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    dtype: str
    itemsize: int
    shard_dims: tuple[int, ...]
    parallel_axes: tuple[LogicalParallelAxis, ...]
    layer_id: int | None
    expert_id: int | None
    layout_fingerprint: str
    nbytes: int
    rank: WeightParallelRank

    def __post_init__(self) -> None:
        if not self.placement_fragment_id or not self.tensor_id:
            raise ValueError("placement fragment identifiers must not be empty")
        if not self.dtype or not self.layout_fingerprint:
            raise ValueError("placement fragment layout must not be empty")
        ndim = len(self.global_shape)
        if (
            ndim == 0
            or len(self.global_offset) != ndim
            or len(self.local_shape) != ndim
            or any(type(value) is not int or value <= 0 for value in self.global_shape)
            or any(type(value) is not int or value < 0 for value in self.global_offset)
            or any(type(value) is not int or value <= 0 for value in self.local_shape)
        ):
            raise ValueError("placement fragment has an invalid logical box")
        if any(
            offset + extent > total
            for offset, extent, total in zip(
                self.global_offset, self.local_shape, self.global_shape
            )
        ):
            raise ValueError("placement fragment logical box is out of bounds")
        if (
            type(self.itemsize) is not int
            or self.itemsize <= 0
            or type(self.nbytes) is not int
            or self.nbytes != prod(self.local_shape) * self.itemsize
        ):
            raise ValueError("placement fragment byte units are inconsistent")
        if (
            tuple(sorted(self.shard_dims)) != self.shard_dims
            or len(set(self.shard_dims)) != len(self.shard_dims)
            or any(
                type(dim) is not int or not 0 <= dim < ndim for dim in self.shard_dims
            )
        ):
            raise ValueError("shard_dims must be sorted, unique, and in bounds")
        split_dims = tuple(
            sorted(
                axis.dim
                for axis in self.parallel_axes
                if axis.mode == "split" and axis.dim is not None
            )
        )
        axis_kinds = tuple(axis.kind for axis in self.parallel_axes)
        if len(axis_kinds) != len(set(axis_kinds)):
            raise ValueError("parallel_axes must contain at most one axis per kind")
        if set(axis_kinds) != {"dp", "tp", "pp", "ep"}:
            raise ValueError("parallel_axes must explicitly describe dp/tp/pp/ep")
        axes_by_kind = {axis.kind: axis for axis in self.parallel_axes}
        for axis in self.parallel_axes:
            if axis.mode != "coupled":
                continue
            coupled_to = axes_by_kind.get(axis.coupled_to)
            if coupled_to is None or coupled_to.mode == "coupled":
                raise ValueError("a coupled axis requires one explicit primary axis")
        if split_dims != self.shard_dims:
            raise ValueError("parallel_axes conflict with shard_dims")
        if self.aliases:
            if (
                tuple(sorted(self.aliases)) != self.aliases
                or len(self.aliases) < 2
                or len(set(self.aliases)) != len(self.aliases)
                or self.tensor_id not in self.aliases
                or any(type(alias) is not str or not alias for alias in self.aliases)
            ):
                raise ValueError("aliases must be a canonical logical tensor ID group")
        if not isinstance(self.rank, WeightParallelRank):
            raise ValueError("rank must be a WeightParallelRank")
        if self.placement_fragment_id != _placement_fragment_id(
            tensor_id=self.tensor_id,
            aliases=self.aliases,
            global_shape=self.global_shape,
            global_offset=self.global_offset,
            local_shape=self.local_shape,
            dtype=self.dtype,
            itemsize=self.itemsize,
            shard_dims=self.shard_dims,
            parallel_axes=self.parallel_axes,
            layer_id=self.layer_id,
            expert_id=self.expert_id,
            layout_fingerprint=self.layout_fingerprint,
            nbytes=self.nbytes,
            rank=self.rank,
        ):
            raise ValueError("placement fragment identity does not match its facts")


class WeightPlacementInventory(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    model_id: str
    revision: str
    weight_generation: int
    inventory_id: str
    participant_id: str
    topology: WeightParallelTopology
    fragments: tuple[WeightPlacementInventoryFragment, ...]

    def __post_init__(self) -> None:
        if not all(
            (self.model_id, self.revision, self.inventory_id, self.participant_id)
        ):
            raise ValueError("placement inventory identifiers must not be empty")
        if type(self.weight_generation) is not int or self.weight_generation <= 0:
            raise ValueError("weight_generation must be a positive integer")
        if not isinstance(self.topology, WeightParallelTopology):
            raise ValueError("topology must be a WeightParallelTopology")
        if not self.fragments:
            raise ValueError("placement inventory must contain fragments")
        fragment_ids = tuple(item.placement_fragment_id for item in self.fragments)
        if len(fragment_ids) != len(set(fragment_ids)):
            raise ValueError("placement inventory fragment IDs must be unique")
        topology_rank = self.topology.rank()
        if any(fragment.rank != topology_rank for fragment in self.fragments):
            raise ValueError("placement fragment rank differs from inventory topology")
        for fragment in self.fragments:
            for axis in fragment.parallel_axes:
                if axis.mode != "coupled":
                    continue
                if getattr(self.topology, f"{axis.kind}_size") != getattr(
                    self.topology, f"{axis.coupled_to}_size"
                ) or getattr(fragment.rank, axis.kind) != getattr(
                    fragment.rank, axis.coupled_to
                ):
                    raise ValueError(
                        "coupled parallel axis differs from its primary coordinate"
                    )
        descriptor_signatures: dict[str, tuple[Any, ...]] = {}
        for fragment in self.fragments:
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
            previous = descriptor_signatures.setdefault(fragment.tensor_id, signature)
            if previous != signature:
                raise ValueError(
                    "placement fragments disagree on logical tensor descriptor"
                )
        expected_participant_id = _participant_id(
            model_id=self.model_id,
            revision=self.revision,
            topology=self.topology,
        )
        if self.participant_id != expected_participant_id:
            raise ValueError("placement inventory participant identity is unstable")
        if self.inventory_id != _placement_id(
            model_id=self.model_id,
            revision=self.revision,
            weight_generation=self.weight_generation,
            topology=self.topology,
            fragments=self.fragments,
        ):
            raise ValueError("placement inventory identity does not match its facts")


class WeightRuntimeBindingInventoryFragment(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    placement_fragment_id: str
    fragment_id: str
    address: int
    nbytes: int
    storage_offset: int
    itemsize: int
    local_shape: tuple[int, ...]
    strides_bytes: tuple[int, ...]
    storage_address: int
    storage_nbytes: int
    storage_offset_bytes: int
    device: str
    is_contiguous: bool
    worker_id: str
    endpoint: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.placement_fragment_id,
                self.fragment_id,
                self.device,
                self.worker_id,
                self.endpoint,
            )
        ):
            raise ValueError("runtime binding fragment identifiers must not be empty")
        if (
            type(self.address) is not int
            or self.address <= 0
            or type(self.storage_address) is not int
            or self.storage_address <= 0
            or type(self.storage_nbytes) is not int
            or self.storage_nbytes <= 0
        ):
            raise ValueError("runtime binding addresses must be positive integers")
        if (
            type(self.itemsize) is not int
            or self.itemsize <= 0
            or type(self.nbytes) is not int
            or self.nbytes != prod(self.local_shape) * self.itemsize
        ):
            raise ValueError("runtime binding byte units are inconsistent")
        if not self.local_shape or any(
            type(extent) is not int or extent <= 0 for extent in self.local_shape
        ):
            raise ValueError(
                "runtime binding local_shape must contain positive extents"
            )
        if self.strides_bytes != _contiguous_strides_bytes(
            self.local_shape, self.itemsize
        ):
            raise ValueError("runtime binding has invalid contiguous strides")
        if (
            type(self.storage_offset) is not int
            or type(self.storage_offset_bytes) is not int
            or self.storage_offset < 0
            or self.storage_offset_bytes < 0
        ):
            raise ValueError("storage offsets must be non-negative")
        if self.storage_offset_bytes != self.storage_offset * self.itemsize:
            raise ValueError("storage_offset conflicts with storage_offset_bytes")
        if self.address != self.storage_address + self.storage_offset_bytes:
            raise ValueError("address conflicts with storage offset")
        if self.storage_offset_bytes + self.nbytes > self.storage_nbytes:
            raise ValueError("runtime binding exceeds its storage")
        if self.is_contiguous is not True:
            raise ValueError("runtime binding fragment must be contiguous")


class WeightRuntimeBindingInventory(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    model_id: str
    revision: str
    placement_inventory_id: str
    instance_id: str
    generation: int
    lease_id: str
    participant_id: str
    fragments: tuple[WeightRuntimeBindingInventoryFragment, ...]

    def __post_init__(self) -> None:
        if not all(
            (
                self.model_id,
                self.revision,
                self.placement_inventory_id,
                self.instance_id,
                self.lease_id,
                self.participant_id,
            )
        ):
            raise ValueError("runtime binding inventory identifiers must not be empty")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("runtime binding generation must be positive")
        if not self.fragments:
            raise ValueError("runtime binding inventory must contain fragments")
        placement_ids = tuple(item.placement_fragment_id for item in self.fragments)
        fragment_ids = tuple(item.fragment_id for item in self.fragments)
        if len(placement_ids) != len(set(placement_ids)):
            raise ValueError("runtime binding placement fragment IDs must be unique")
        if len(fragment_ids) != len(set(fragment_ids)):
            raise ValueError("runtime binding fragment IDs must be unique")


class WeightPlacementBindingInventories(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    placement: WeightPlacementInventory
    binding: WeightRuntimeBindingInventory

    def __post_init__(self) -> None:
        if (
            self.placement.model_id != self.binding.model_id
            or self.placement.revision != self.binding.revision
            or self.placement.inventory_id != self.binding.placement_inventory_id
            or self.placement.participant_id != self.binding.participant_id
        ):
            raise ValueError("placement and runtime binding inventories disagree")
        placement_ids = {
            item.placement_fragment_id for item in self.placement.fragments
        }
        binding_ids = {item.placement_fragment_id for item in self.binding.fragments}
        if placement_ids != binding_ids:
            raise ValueError("runtime binding does not cover the placement inventory")
        placement_by_id = {
            item.placement_fragment_id: item for item in self.placement.fragments
        }
        for runtime_fragment in self.binding.fragments:
            placement_fragment = placement_by_id[runtime_fragment.placement_fragment_id]
            if (
                runtime_fragment.nbytes != placement_fragment.nbytes
                or runtime_fragment.itemsize != placement_fragment.itemsize
                or runtime_fragment.local_shape != placement_fragment.local_shape
            ):
                raise ValueError(
                    "runtime binding fragment geometry differs from placement fragment"
                )


class WeightSemanticsAdapter(Protocol):
    def describe_parameter(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]: ...


def _placement_fragment_id(
    *,
    tensor_id: str,
    aliases: tuple[str, ...],
    global_shape: tuple[int, ...],
    global_offset: tuple[int, ...],
    local_shape: tuple[int, ...],
    dtype: str,
    itemsize: int,
    shard_dims: tuple[int, ...],
    parallel_axes: tuple[LogicalParallelAxis, ...],
    layer_id: int | None,
    expert_id: int | None,
    layout_fingerprint: str,
    nbytes: int,
    rank: WeightParallelRank,
) -> str:
    identity = (
        "sglang-weight-placement-fragment",
        tensor_id,
        aliases,
        global_shape,
        global_offset,
        local_shape,
        dtype,
        itemsize,
        shard_dims,
        parallel_axes,
        layer_id,
        expert_id,
        layout_fingerprint,
        nbytes,
        rank,
    )
    return hashlib.sha256(msgspec.json.encode(identity)).hexdigest()[:24]


def _placement_id(
    *,
    model_id: str,
    revision: str,
    weight_generation: int,
    topology: WeightParallelTopology,
    fragments: tuple[WeightPlacementInventoryFragment, ...],
) -> str:
    identity = (
        "sglang-weight-placement-inventory",
        model_id,
        revision,
        weight_generation,
        topology,
        tuple(sorted(fragments, key=lambda item: item.placement_fragment_id)),
    )
    return hashlib.sha256(msgspec.json.encode(identity)).hexdigest()[:32]


def _participant_id(
    *,
    model_id: str,
    revision: str,
    topology: WeightParallelTopology,
) -> str:
    identity = (
        "sglang-weight-participant",
        model_id,
        revision,
        topology.dp_rank,
        topology.dp_size,
        topology.tp_rank,
        topology.tp_size,
        topology.pp_rank,
        topology.pp_size,
        topology.ep_rank,
        topology.ep_size,
        topology.moe_tp_rank,
        topology.moe_tp_size,
        topology.attention_tp_rank,
        topology.attention_tp_size,
    )
    digest = hashlib.sha256(msgspec.json.encode(identity)).hexdigest()
    return f"sglang-weight-participant:sha256:{digest}"


def _contiguous_strides_bytes(
    local_shape: tuple[int, ...], itemsize: int
) -> tuple[int, ...]:
    stride = itemsize
    result = []
    for extent in reversed(local_shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


__all__ = [
    "DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC",
    "LogicalParallelAxis",
    "LogicalTensorView",
    "WeightInventoryError",
    "WeightParallelRank",
    "WeightParallelTopology",
    "WeightPlacementBindingInventories",
    "WeightPlacementInventory",
    "WeightPlacementInventoryFragment",
    "WeightRuntimeBindingInventory",
    "WeightRuntimeBindingInventoryFragment",
    "WeightSemanticsAdapter",
    "validate_remote_instance_weight_transfer_lease_timeout",
    "validate_remote_weight_lineage",
    "validate_remote_weight_source_identity",
    "validate_weight_topology_representability",
]
