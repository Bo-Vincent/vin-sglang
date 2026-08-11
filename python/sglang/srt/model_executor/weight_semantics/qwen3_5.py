from __future__ import annotations

import re
from math import prod
from typing import Any, Sequence

import msgspec

from sglang.srt.model_executor.weight_inventory_contracts import (
    LogicalParallelAxis,
    LogicalTensorView,
    WeightInventoryError,
    WeightParallelTopology,
)

_LAYER_PATTERN = re.compile(r"(?:^|\.)(layers\.(\d+)\..*)")


def _shape(parameter: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in parameter.shape)


def _itemsize(parameter: Any) -> int:
    return int(parameter.element_size())


def _canonical_name(name: str) -> str:
    layer_match = _LAYER_PATTERN.search(name)
    if layer_match is not None:
        return layer_match.group(1)
    for suffix in ("embed_tokens.weight", "lm_head.weight", "norm.weight"):
        if name.endswith(suffix):
            return suffix
    for prefix in ("language_model.model.", "language_model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _layer_id(name: str) -> int | None:
    match = _LAYER_PATTERN.search(name)
    return int(match.group(2)) if match is not None else None


def _replace_suffix(name: str, old: str, new: str) -> str:
    if not name.endswith(old):
        raise WeightInventoryError(f"invalid Qwen3.5 parameter name: {name}")
    return name[: -len(old)] + new


def _parallel_axes(
    shard_dims: tuple[int, ...],
    *,
    expert_id: int | None,
) -> tuple[LogicalParallelAxis, ...]:
    axes = [
        LogicalParallelAxis(kind="dp", mode="replicated"),
        LogicalParallelAxis(kind="pp", mode="ownership"),
    ]
    tp_dims = shard_dims
    if len(shard_dims) == 2 and shard_dims[0] == 0:
        axes.append(LogicalParallelAxis(kind="ep", mode="split", dim=0))
        tp_dims = shard_dims[1:]
    elif expert_id is not None:
        axes.append(LogicalParallelAxis(kind="ep", mode="ownership"))
    else:
        axes.append(LogicalParallelAxis(kind="ep", mode="replicated"))
    if len(tp_dims) > 1:
        raise WeightInventoryError("Qwen3.5 view has ambiguous TP shard dimensions")
    axes.append(
        LogicalParallelAxis(
            kind="tp",
            mode="split" if tp_dims else "replicated",
            dim=tp_dims[0] if tp_dims else None,
        )
    )
    return tuple(axes)


def _moe_parallel_semantics(
    topology: WeightParallelTopology,
    *,
    tp_dim: int,
) -> tuple[tuple[int, ...], tuple[LogicalParallelAxis, ...]]:
    shard_dims = []
    axes = [
        LogicalParallelAxis(kind="dp", mode="replicated"),
        LogicalParallelAxis(kind="pp", mode="ownership"),
    ]
    if topology.ep_size > 1:
        shard_dims.append(0)
        axes.append(LogicalParallelAxis(kind="ep", mode="split", dim=0))
    else:
        axes.append(LogicalParallelAxis(kind="ep", mode="replicated"))
    if topology.moe_tp_size > 1:
        shard_dims.append(tp_dim)
        axes.append(LogicalParallelAxis(kind="tp", mode="split", dim=tp_dim))
    else:
        axes.append(LogicalParallelAxis(kind="tp", mode="replicated"))
    return tuple(sorted(shard_dims)), tuple(axes)


def _shared_moe_parallel_semantics(
    topology: WeightParallelTopology,
    *,
    tp_dim: int,
) -> tuple[tuple[int, ...], tuple[LogicalParallelAxis, ...]]:
    axes = [
        LogicalParallelAxis(kind="dp", mode="replicated"),
        LogicalParallelAxis(kind="pp", mode="ownership"),
    ]
    axes.append(LogicalParallelAxis(kind="ep", mode="replicated"))
    if topology.moe_tp_size > 1:
        axes.append(LogicalParallelAxis(kind="tp", mode="split", dim=tp_dim))
        return (tp_dim,), tuple(axes)
    axes.append(LogicalParallelAxis(kind="tp", mode="replicated"))
    return (), tuple(axes)


def _apply_qwen_coupled_parallel_semantics(
    views: tuple[LogicalTensorView, ...],
    topology: WeightParallelTopology,
) -> tuple[LogicalTensorView, ...]:
    """Describe the TP/EP rank co-map without claiming independent replicas."""

    if topology.ep_size == 1:
        return views
    if (
        topology.moe_tp_size != 1
        or topology.tp_size != topology.ep_size
        or topology.tp_rank != topology.ep_rank
    ):
        raise WeightInventoryError(
            "Qwen TP/EP coupled semantics require equal co-mapped TP and EP ranks"
        )

    result = []
    for view in views:
        axes = {axis.kind: axis for axis in view.parallel_axes}
        if set(axes) != {"dp", "tp", "pp", "ep"}:
            raise WeightInventoryError(
                f"Qwen logical view does not explicitly describe four axes: "
                f"{view.tensor_id}"
            )
        coupled_kind, primary_kind = (
            ("tp", "ep") if axes["ep"].mode in {"split", "ownership"} else ("ep", "tp")
        )
        parallel_axes = tuple(
            (
                LogicalParallelAxis(
                    kind=axis.kind,
                    mode="coupled",
                    coupled_to=primary_kind,
                )
                if axis.kind == coupled_kind
                else axis
            )
            for axis in view.parallel_axes
        )
        result.append(msgspec.structs.replace(view, parallel_axes=parallel_axes))
    return tuple(result)


def _view(
    *,
    tensor_id: str,
    global_shape: tuple[int, ...],
    global_offset: tuple[int, ...],
    local_shape: tuple[int, ...],
    shard_dims: tuple[int, ...],
    byte_offset: int,
    layer_id: int | None,
    expert_id: int | None = None,
    layout: str,
    parallel_axes: tuple[LogicalParallelAxis, ...] | None = None,
    aliases: tuple[str, ...] = (),
) -> LogicalTensorView:
    return LogicalTensorView(
        tensor_id=tensor_id,
        global_shape=global_shape,
        global_offset=global_offset,
        local_shape=local_shape,
        byte_offset=byte_offset,
        layer_id=layer_id,
        expert_id=expert_id,
        layout_fingerprint=f"sglang:qwen3.5:{layout}:v1",
        shard_dims=shard_dims,
        parallel_axes=(
            _parallel_axes(shard_dims, expert_id=expert_id)
            if parallel_axes is None
            else parallel_axes
        ),
        aliases=aliases,
    )


def _split_dim_zero(
    *,
    parameter: Any,
    tensor_ids: Sequence[str],
    global_extents: Sequence[int],
    ranks: Sequence[int],
    sizes: Sequence[int],
    layer_id: int | None,
    layout: str,
    replicated: Sequence[bool] | None = None,
    aliases: tuple[str, ...] = (),
) -> tuple[LogicalTensorView, ...]:
    if not (len(tensor_ids) == len(global_extents) == len(ranks) == len(sizes)):
        raise WeightInventoryError("invalid packed Qwen3.5 tensor description")
    replicated = tuple(replicated or (False,) * len(tensor_ids))
    if len(replicated) != len(tensor_ids):
        raise WeightInventoryError("invalid packed Qwen3.5 replication description")
    shape = _shape(parameter)
    tail = shape[1:]
    local_extents = []
    for extent, rank, size in zip(global_extents, ranks, sizes):
        if size <= 0 or rank < 0 or rank >= size or extent % size != 0:
            raise WeightInventoryError("Qwen3.5 tensor is not evenly partitionable")
        local_extents.append(extent // size)
    if not shape or shape[0] != sum(local_extents):
        raise WeightInventoryError(
            f"Qwen3.5 packed tensor shape mismatch: {shape}, {local_extents}"
        )

    byte_offset = 0
    result = []
    for tensor_id, extent, local_extent, rank, size, is_replicated in zip(
        tensor_ids, global_extents, local_extents, ranks, sizes, replicated
    ):
        if is_replicated and (rank != 0 or size != 1):
            raise WeightInventoryError(
                "a replicated Qwen3.5 packed view must contain the full tensor"
            )
        local_shape = (local_extent, *tail)
        result.append(
            _view(
                tensor_id=tensor_id,
                global_shape=(extent, *tail),
                global_offset=(rank * local_extent, *((0,) * len(tail))),
                local_shape=local_shape,
                shard_dims=() if is_replicated else (0,),
                byte_offset=byte_offset,
                layer_id=layer_id,
                layout=layout,
                aliases=aliases,
            )
        )
        byte_offset += prod(local_shape) * _itemsize(parameter)
    return tuple(result)


def _split_dim_zero_or_full(
    *,
    parameter: Any,
    tensor_ids: Sequence[str],
    global_extents: Sequence[int],
    global_tail: tuple[int, ...],
    rank: int,
    size: int,
    layer_id: int | None,
    layout: str,
) -> tuple[LogicalTensorView, ...]:
    if len(tensor_ids) != len(global_extents) or size <= 0 or not 0 <= rank < size:
        raise WeightInventoryError("invalid Qwen3.5 VL tensor description")
    if any(extent <= 0 or extent % size != 0 for extent in global_extents):
        raise WeightInventoryError("Qwen3.5 VL tensor is not evenly partitionable")

    shape = _shape(parameter)
    local_extents = tuple(extent // size for extent in global_extents)
    sharded_shape = (sum(local_extents), *global_tail)
    full_shape = (sum(global_extents), *global_tail)
    if shape == sharded_shape:
        extents = local_extents
        offsets = tuple(rank * extent for extent in local_extents)
    elif shape == full_shape:
        extents = tuple(global_extents)
        offsets = (0,) * len(global_extents)
    else:
        raise WeightInventoryError(
            f"Qwen3.5 VL packed tensor shape mismatch: {shape}, "
            f"expected {sharded_shape} or {full_shape}"
        )

    byte_offset = 0
    result = []
    for tensor_id, global_extent, local_extent, offset in zip(
        tensor_ids, global_extents, extents, offsets
    ):
        local_shape = (local_extent, *global_tail)
        result.append(
            _view(
                tensor_id=tensor_id,
                global_shape=(global_extent, *global_tail),
                global_offset=(offset, *((0,) * len(global_tail))),
                local_shape=local_shape,
                shard_dims=(0,),
                byte_offset=byte_offset,
                layer_id=layer_id,
                layout=layout,
            )
        )
        byte_offset += prod(local_shape) * _itemsize(parameter)
    return tuple(result)


def _row_parallel_or_full_view(
    *,
    parameter: Any,
    tensor_id: str,
    global_shape: tuple[int, int],
    rank: int,
    size: int,
    layer_id: int | None,
    layout: str,
) -> tuple[LogicalTensorView, ...]:
    if size <= 0 or not 0 <= rank < size or global_shape[1] % size != 0:
        raise WeightInventoryError(
            f"Qwen3.5 VL tensor is not TP divisible: {tensor_id}"
        )
    shape = _shape(parameter)
    sharded_shape = (global_shape[0], global_shape[1] // size)
    if shape == sharded_shape:
        offset = rank * sharded_shape[1]
    elif shape == global_shape:
        offset = 0
    else:
        raise WeightInventoryError(
            f"Qwen3.5 VL row tensor shape mismatch: {tensor_id}: {shape}, "
            f"expected {sharded_shape} or {global_shape}"
        )
    return (
        _view(
            tensor_id=tensor_id,
            global_shape=global_shape,
            global_offset=(0, offset),
            local_shape=shape,
            shard_dims=(1,),
            byte_offset=0,
            layer_id=layer_id,
            layout=layout,
        ),
    )


def _replicated_view(
    *,
    parameter: Any,
    tensor_id: str,
    layer_id: int | None,
    layout: str,
    expected_shape: tuple[int, ...] | None = None,
) -> tuple[LogicalTensorView, ...]:
    shape = _shape(parameter)
    if expected_shape is not None and shape != expected_shape:
        raise WeightInventoryError(
            f"Qwen3.5 replicated tensor shape mismatch: {tensor_id}: {shape}, "
            f"expected {expected_shape}"
        )
    return (
        _view(
            tensor_id=tensor_id,
            global_shape=shape,
            global_offset=(0,) * len(shape),
            local_shape=shape,
            shard_dims=(),
            byte_offset=0,
            layer_id=layer_id,
            layout=layout,
        ),
    )


def _row_parallel_view(
    *,
    parameter: Any,
    tensor_id: str,
    global_shape: tuple[int, int],
    rank: int,
    size: int,
    layer_id: int | None,
    layout: str,
) -> tuple[LogicalTensorView, ...]:
    if global_shape[1] % size != 0:
        raise WeightInventoryError(f"Qwen3.5 tensor is not TP divisible: {tensor_id}")
    local_shape = (global_shape[0], global_shape[1] // size)
    if _shape(parameter) != local_shape:
        raise WeightInventoryError(
            f"Qwen3.5 row tensor shape mismatch: {tensor_id}: {_shape(parameter)}"
        )
    return (
        _view(
            tensor_id=tensor_id,
            global_shape=global_shape,
            global_offset=(0, rank * local_shape[1]),
            local_shape=local_shape,
            shard_dims=(1,),
            byte_offset=0,
            layer_id=layer_id,
            layout=layout,
        ),
    )


class Qwen35WeightSemanticsAdapter:
    def __init__(
        self,
        *,
        config: Any,
        dynamic_expert_placement: bool = False,
        up_first_w13_parameter_ids: Sequence[int] = (),
        num_fused_shared_experts: int = 0,
    ) -> None:
        self._config = config
        self._dynamic_expert_placement = dynamic_expert_placement
        self._up_first_w13_parameter_ids = frozenset(up_first_w13_parameter_ids)
        if num_fused_shared_experts not in (0, 1):
            raise WeightInventoryError(
                "Qwen3.5 weight inventories support at most one fused shared expert"
            )
        self._num_fused_shared_experts = num_fused_shared_experts

    def describe_parameter(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]:
        return _apply_qwen_coupled_parallel_semantics(
            self._describe_parameter_uncoupled(
                names=names,
                parameter=parameter,
                topology=topology,
            ),
            topology,
        )

    def _describe_parameter_uncoupled(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]:
        canonical_names = tuple(dict.fromkeys(_canonical_name(name) for name in names))
        name = canonical_names[0]
        layer_id = _layer_id(name)
        aliases = tuple(sorted(canonical_names)) if len(canonical_names) > 1 else ()
        if aliases and set(aliases) != {"embed_tokens.weight", "lm_head.weight"}:
            raise WeightInventoryError(
                f"Qwen3.5 has an unsupported logical parameter alias group: {aliases}"
            )

        if name.endswith("experts.w13_weight"):
            return self._moe_w13(
                name=name,
                parameter=parameter,
                topology=topology,
                layer_id=layer_id,
            )
        if name.endswith("experts.w2_weight"):
            return self._moe_w2(
                name=name,
                parameter=parameter,
                topology=topology,
                layer_id=layer_id,
            )
        if name.endswith("qkv_proj.weight"):
            return self._qkv(
                name=name,
                parameter=parameter,
                topology=topology,
                layer_id=layer_id,
            )
        if name.endswith("in_proj_qkvz.weight"):
            return self._gdn_qkvz(
                name=name,
                parameter=parameter,
                topology=topology,
                layer_id=layer_id,
            )
        if name.endswith("in_proj_ba.weight"):
            return self._gdn_ba(
                name=name,
                parameter=parameter,
                topology=topology,
                layer_id=layer_id,
            )
        if name.endswith("conv1d.weight"):
            return self._gdn_conv(
                name=name,
                parameter=parameter,
                topology=topology,
                layer_id=layer_id,
            )
        if name.endswith("gate_up_proj.weight"):
            intermediate = self._intermediate_size(name)
            return _split_dim_zero(
                parameter=parameter,
                tensor_ids=(
                    _replace_suffix(name, "gate_up_proj.weight", "gate_proj.weight"),
                    _replace_suffix(name, "gate_up_proj.weight", "up_proj.weight"),
                ),
                global_extents=(intermediate, intermediate),
                ranks=(topology.tp_rank, topology.tp_rank),
                sizes=(topology.tp_size, topology.tp_size),
                layer_id=layer_id,
                layout="gate-up",
            )
        if name.endswith("down_proj.weight"):
            return _row_parallel_view(
                parameter=parameter,
                tensor_id=name,
                global_shape=(
                    int(self._config.hidden_size),
                    self._intermediate_size(name),
                ),
                rank=topology.tp_rank,
                size=topology.tp_size,
                layer_id=layer_id,
                layout="row-parallel",
            )
        if name.endswith("o_proj.weight"):
            head_dim = self._head_dim()
            return _row_parallel_view(
                parameter=parameter,
                tensor_id=name,
                global_shape=(
                    int(self._config.hidden_size),
                    int(self._config.num_attention_heads) * head_dim,
                ),
                rank=topology.attention_tp_rank,
                size=topology.attention_tp_size,
                layer_id=layer_id,
                layout="attention-row",
            )
        if name.endswith("out_proj.weight"):
            value_dim = int(self._config.linear_value_head_dim) * int(
                self._config.linear_num_value_heads
            )
            return _row_parallel_view(
                parameter=parameter,
                tensor_id=name,
                global_shape=(int(self._config.hidden_size), value_dim),
                rank=topology.attention_tp_rank,
                size=topology.attention_tp_size,
                layer_id=layer_id,
                layout="gdn-row",
            )
        if name in ("embed_tokens.weight", "lm_head.weight"):
            return tuple(
                view
                for tensor_id in canonical_names
                if tensor_id in ("embed_tokens.weight", "lm_head.weight")
                for view in _split_dim_zero(
                    parameter=parameter,
                    tensor_ids=(tensor_id,),
                    global_extents=(int(self._config.vocab_size),),
                    ranks=(topology.tp_rank,),
                    sizes=(topology.tp_size,),
                    layer_id=None,
                    layout="vocab-parallel",
                    aliases=aliases,
                )
            )
        if name.endswith(("A_log", "dt_bias")):
            extent = int(self._config.linear_num_value_heads)
            return _split_dim_zero(
                parameter=parameter,
                tensor_ids=(name,),
                global_extents=(extent,),
                ranks=(topology.attention_tp_rank,),
                sizes=(topology.attention_tp_size,),
                layer_id=layer_id,
                layout="gdn-head",
            )
        if name.endswith(
            (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "q_norm.weight",
                "k_norm.weight",
                "norm.weight",
                "gate.weight",
                "shared_expert_gate.weight",
            )
        ):
            shape = _shape(parameter)
            return (
                _view(
                    tensor_id=name,
                    global_shape=shape,
                    global_offset=(0,) * len(shape),
                    local_shape=shape,
                    shard_dims=(),
                    byte_offset=0,
                    layer_id=layer_id,
                    layout="replicated",
                ),
            )
        raise WeightInventoryError(f"unsupported Qwen3.5 parameter: {names[0]}")

    def _head_dim(self) -> int:
        configured = getattr(self._config, "head_dim", None)
        if configured:
            return int(configured)
        return int(self._config.hidden_size) // int(self._config.num_attention_heads)

    def _intermediate_size(self, name: str) -> int:
        if "shared_expert" in name and hasattr(
            self._config, "shared_expert_intermediate_size"
        ):
            return int(self._config.shared_expert_intermediate_size)
        return int(self._config.intermediate_size)

    def _has_attention_output_gate(self) -> bool:
        return bool(getattr(self._config, "attn_output_gate", True))

    def _qkv(
        self, *, name, parameter, topology, layer_id
    ) -> tuple[LogicalTensorView, ...]:
        head_dim = self._head_dim()
        q_extent = int(self._config.num_attention_heads) * head_dim
        if self._has_attention_output_gate():
            q_extent *= 2
        kv_heads = int(self._config.num_key_value_heads)
        kv_extent = kv_heads * head_dim
        kv_partitions = min(topology.attention_tp_size, kv_heads)
        kv_replicas = topology.attention_tp_size // kv_partitions
        if kv_replicas > 1 and kv_partitions > 1:
            raise WeightInventoryError(
                "partially replicated Qwen3.5 KV shards are not represented by "
                "the current weight inventory contract"
            )
        kv_rank = topology.attention_tp_rank // kv_replicas
        return _split_dim_zero(
            parameter=parameter,
            tensor_ids=(
                _replace_suffix(name, "qkv_proj.weight", "q_proj.weight"),
                _replace_suffix(name, "qkv_proj.weight", "k_proj.weight"),
                _replace_suffix(name, "qkv_proj.weight", "v_proj.weight"),
            ),
            global_extents=(q_extent, kv_extent, kv_extent),
            ranks=(
                topology.attention_tp_rank,
                kv_rank,
                kv_rank,
            ),
            sizes=(topology.attention_tp_size, kv_partitions, kv_partitions),
            layer_id=layer_id,
            layout="qkv",
            replicated=(False, kv_partitions == 1, kv_partitions == 1),
        )

    def _gdn_qkvz(
        self, *, name, parameter, topology, layer_id
    ) -> tuple[LogicalTensorView, ...]:
        key_dim = int(self._config.linear_key_head_dim) * int(
            self._config.linear_num_key_heads
        )
        value_dim = int(self._config.linear_value_head_dim) * int(
            self._config.linear_num_value_heads
        )
        suffixes = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "z_proj.weight")
        return _split_dim_zero(
            parameter=parameter,
            tensor_ids=tuple(
                _replace_suffix(name, "in_proj_qkvz.weight", suffix)
                for suffix in suffixes
            ),
            global_extents=(key_dim, key_dim, value_dim, value_dim),
            ranks=(topology.attention_tp_rank,) * 4,
            sizes=(topology.attention_tp_size,) * 4,
            layer_id=layer_id,
            layout="gdn-qkvz",
        )

    def _gdn_ba(
        self, *, name, parameter, topology, layer_id
    ) -> tuple[LogicalTensorView, ...]:
        extent = int(self._config.linear_num_value_heads)
        return _split_dim_zero(
            parameter=parameter,
            tensor_ids=(
                _replace_suffix(name, "in_proj_ba.weight", "b_proj.weight"),
                _replace_suffix(name, "in_proj_ba.weight", "a_proj.weight"),
            ),
            global_extents=(extent, extent),
            ranks=(topology.attention_tp_rank,) * 2,
            sizes=(topology.attention_tp_size,) * 2,
            layer_id=layer_id,
            layout="gdn-ba",
        )

    def _gdn_conv(
        self, *, name, parameter, topology, layer_id
    ) -> tuple[LogicalTensorView, ...]:
        key_dim = int(self._config.linear_key_head_dim) * int(
            self._config.linear_num_key_heads
        )
        value_dim = int(self._config.linear_value_head_dim) * int(
            self._config.linear_num_value_heads
        )
        return _split_dim_zero(
            parameter=parameter,
            tensor_ids=(
                _replace_suffix(name, "conv1d.weight", "conv_q.weight"),
                _replace_suffix(name, "conv1d.weight", "conv_k.weight"),
                _replace_suffix(name, "conv1d.weight", "conv_v.weight"),
            ),
            global_extents=(key_dim, key_dim, value_dim),
            ranks=(topology.attention_tp_rank,) * 3,
            sizes=(topology.attention_tp_size,) * 3,
            layer_id=layer_id,
            layout="gdn-conv",
        )

    def _expert_ids(
        self, *, parameter: Any, topology: WeightParallelTopology
    ) -> tuple[int, ...]:
        if self._dynamic_expert_placement:
            raise WeightInventoryError(
                "dynamic Qwen3.5 expert placement requires an explicit expert map"
            )
        num_experts = int(self._config.num_experts)
        if num_experts % topology.ep_size != 0:
            raise WeightInventoryError(
                "Qwen3.5 experts are not evenly EP partitionable"
            )
        local_experts = num_experts // topology.ep_size
        expected_slots = local_experts + self._num_fused_shared_experts
        if not _shape(parameter) or _shape(parameter)[0] != expected_slots:
            raise WeightInventoryError(
                f"Qwen3.5 local expert count mismatch: {_shape(parameter)}, "
                f"expected {expected_slots} slots"
            )
        start = topology.ep_rank * local_experts
        return tuple(range(start, start + local_experts))

    def _shared_intermediate_size(self) -> int:
        shared_intermediate = int(
            getattr(self._config, "shared_expert_intermediate_size", 0)
        )
        intermediate = int(self._config.moe_intermediate_size)
        if self._num_fused_shared_experts and shared_intermediate != intermediate:
            raise WeightInventoryError(
                "Qwen3.5 fused shared expert must match routed intermediate size"
            )
        return shared_intermediate

    def _moe_w13(
        self, *, name, parameter, topology, layer_id
    ) -> tuple[LogicalTensorView, ...]:
        expert_ids = self._expert_ids(parameter=parameter, topology=topology)
        shape = _shape(parameter)
        intermediate = int(self._config.moe_intermediate_size)
        if intermediate % topology.moe_tp_size != 0:
            raise WeightInventoryError("Qwen3.5 expert tensor is not TP divisible")
        local_intermediate = intermediate // topology.moe_tp_size
        expected = (
            len(expert_ids) + self._num_fused_shared_experts,
            local_intermediate * 2,
            int(self._config.hidden_size),
        )
        if shape != expected:
            raise WeightInventoryError(
                f"Qwen3.5 w13 tensor shape mismatch: {shape}, expected {expected}"
            )
        prefix = name[: -len("experts.w13_weight")]
        expert_bytes = prod(shape[1:]) * _itemsize(parameter)
        component_bytes = local_intermediate * shape[2] * _itemsize(parameter)
        components = (
            ("up_proj", "gate_proj")
            if id(parameter) in self._up_first_w13_parameter_ids
            else ("gate_proj", "up_proj")
        )
        result = []
        num_experts = int(self._config.num_experts)
        shard_dims, parallel_axes = _moe_parallel_semantics(topology, tp_dim=1)
        for local_index, expert_id in enumerate(expert_ids):
            base = local_index * expert_bytes
            for component_index, component in enumerate(components):
                result.append(
                    _view(
                        tensor_id=f"{prefix}experts.{component}.weight",
                        global_shape=(num_experts, intermediate, shape[2]),
                        global_offset=(
                            expert_id,
                            topology.moe_tp_rank * local_intermediate,
                            0,
                        ),
                        local_shape=(1, local_intermediate, shape[2]),
                        byte_offset=base + component_index * component_bytes,
                        layer_id=layer_id,
                        expert_id=None,
                        layout="moe-w13",
                        shard_dims=shard_dims,
                        parallel_axes=parallel_axes,
                    )
                )
        if self._num_fused_shared_experts:
            shared_intermediate = self._shared_intermediate_size()
            shared_base = len(expert_ids) * expert_bytes
            shared_shard_dims, shared_axes = _shared_moe_parallel_semantics(
                topology, tp_dim=0
            )
            for component_index, component in enumerate(components):
                result.append(
                    _view(
                        tensor_id=f"{prefix}shared_expert.{component}.weight",
                        global_shape=(shared_intermediate, shape[2]),
                        global_offset=(
                            topology.moe_tp_rank * local_intermediate,
                            0,
                        ),
                        local_shape=(local_intermediate, shape[2]),
                        shard_dims=shared_shard_dims,
                        byte_offset=(shared_base + component_index * component_bytes),
                        layer_id=layer_id,
                        expert_id=None,
                        layout="gate-up",
                        parallel_axes=shared_axes,
                    )
                )
        return tuple(result)

    def _moe_w2(
        self, *, name, parameter, topology, layer_id
    ) -> tuple[LogicalTensorView, ...]:
        expert_ids = self._expert_ids(parameter=parameter, topology=topology)
        shape = _shape(parameter)
        intermediate = int(self._config.moe_intermediate_size)
        if intermediate % topology.moe_tp_size != 0:
            raise WeightInventoryError("Qwen3.5 expert tensor is not TP divisible")
        local_intermediate = intermediate // topology.moe_tp_size
        expected = (
            len(expert_ids) + self._num_fused_shared_experts,
            int(self._config.hidden_size),
            local_intermediate,
        )
        if shape != expected:
            raise WeightInventoryError(
                f"Qwen3.5 w2 tensor shape mismatch: {shape}, expected {expected}"
            )
        prefix = name[: -len("experts.w2_weight")]
        expert_bytes = prod(shape[1:]) * _itemsize(parameter)
        num_experts = int(self._config.num_experts)
        shard_dims, parallel_axes = _moe_parallel_semantics(topology, tp_dim=2)
        result = [
            _view(
                tensor_id=f"{prefix}experts.down_proj.weight",
                global_shape=(num_experts, shape[1], intermediate),
                global_offset=(
                    expert_id,
                    0,
                    topology.moe_tp_rank * local_intermediate,
                ),
                local_shape=(1, shape[1], local_intermediate),
                byte_offset=local_index * expert_bytes,
                layer_id=layer_id,
                expert_id=None,
                layout="moe-w2",
                shard_dims=shard_dims,
                parallel_axes=parallel_axes,
            )
            for local_index, expert_id in enumerate(expert_ids)
        ]
        if self._num_fused_shared_experts:
            shared_intermediate = self._shared_intermediate_size()
            shared_shard_dims, shared_axes = _shared_moe_parallel_semantics(
                topology, tp_dim=1
            )
            result.append(
                _view(
                    tensor_id=f"{prefix}shared_expert.down_proj.weight",
                    global_shape=(shape[1], shared_intermediate),
                    global_offset=(
                        0,
                        topology.moe_tp_rank * local_intermediate,
                    ),
                    local_shape=(shape[1], local_intermediate),
                    shard_dims=shared_shard_dims,
                    byte_offset=len(expert_ids) * expert_bytes,
                    layer_id=layer_id,
                    expert_id=None,
                    layout="row-parallel",
                    parallel_axes=shared_axes,
                )
            )
        return tuple(result)


class Qwen35VisionWeightSemanticsAdapter:
    def __init__(self, *, config: Any, data_parallel: bool = False) -> None:
        self._config = config
        self._data_parallel = data_parallel

    def describe_parameter(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]:
        views = self._describe_parameter(
            names=names,
            parameter=parameter,
            topology=topology,
        )
        if self._data_parallel:
            result = []
            for view in views:
                if view.global_offset != (0,) * len(view.global_shape) or (
                    view.local_shape != view.global_shape
                ):
                    raise WeightInventoryError(
                        "Qwen3.5 MM-DP vision requires a complete tensor replica: "
                        f"{view.tensor_id}"
                    )
                result.append(
                    msgspec.structs.replace(
                        view,
                        shard_dims=(),
                        parallel_axes=_parallel_axes((), expert_id=None),
                    )
                )
            return tuple(result)

        if topology.attention_tp_size > 1:
            for view in views:
                if view.shard_dims and view.local_shape == view.global_shape:
                    raise WeightInventoryError(
                        "Qwen3.5 explicit vision TP received a full tensor instead "
                        f"of a shard: {view.tensor_id}"
                    )
        return views

    def _describe_parameter(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]:
        canonical_names = tuple(
            dict.fromkeys(self._canonical_name(name) for name in names)
        )
        if len(canonical_names) > 1:
            raise WeightInventoryError(
                "Qwen3.5 vision has an unsupported logical parameter alias group: "
                f"{tuple(sorted(canonical_names))}"
            )
        name = canonical_names[0]
        hidden = int(self._config.hidden_size)
        rank = 0 if self._data_parallel else topology.attention_tp_rank
        size = 1 if self._data_parallel else topology.attention_tp_size

        if name == "visual.patch_embed.proj.weight":
            return _replicated_view(
                parameter=parameter,
                tensor_id=name,
                layer_id=None,
                layout="vision-patch",
                expected_shape=(
                    hidden,
                    int(self._config.in_channels),
                    int(self._config.temporal_patch_size),
                    int(self._config.patch_size),
                    int(self._config.patch_size),
                ),
            )
        if name == "visual.patch_embed.proj.bias":
            return _replicated_view(
                parameter=parameter,
                tensor_id=name,
                layer_id=None,
                layout="vision-patch-bias",
                expected_shape=(hidden,),
            )
        if name == "visual.pos_embed.weight":
            return _split_dim_zero_or_full(
                parameter=parameter,
                tensor_ids=(name,),
                global_extents=(int(self._config.num_position_embeddings),),
                global_tail=(hidden,),
                rank=rank,
                size=size,
                layer_id=None,
                layout="vision-position",
            )
        if ".blocks." in name and name.endswith(
            ("attn.qkv_proj.weight", "attn.qkv_proj.bias")
        ):
            suffix = "weight" if name.endswith(".weight") else "bias"
            global_tail = (hidden,) if suffix == "weight" else ()
            return _split_dim_zero_or_full(
                parameter=parameter,
                tensor_ids=tuple(
                    _replace_suffix(
                        name,
                        f"qkv_proj.{suffix}",
                        f"{component}_proj.{suffix}",
                    )
                    for component in ("q", "k", "v")
                ),
                global_extents=(hidden, hidden, hidden),
                global_tail=global_tail,
                rank=rank,
                size=size,
                layer_id=None,
                layout=f"vision-qkv-{suffix}",
            )
        if ".blocks." in name and name.endswith("attn.proj.weight"):
            return _row_parallel_or_full_view(
                parameter=parameter,
                tensor_id=name,
                global_shape=(hidden, hidden),
                rank=rank,
                size=size,
                layer_id=None,
                layout="vision-attention-row",
            )
        if name.endswith("attn.proj.bias"):
            return _replicated_view(
                parameter=parameter,
                tensor_id=name,
                layer_id=None,
                layout="vision-attention-bias",
                expected_shape=(hidden,),
            )
        if name.endswith("linear_fc1.weight"):
            output_size, input_size = self._mlp_shape(name, first=True)
            return _split_dim_zero_or_full(
                parameter=parameter,
                tensor_ids=(name,),
                global_extents=(output_size,),
                global_tail=(input_size,),
                rank=rank,
                size=size,
                layer_id=None,
                layout="vision-mlp-column",
            )
        if name.endswith("linear_fc1.bias"):
            output_size, _ = self._mlp_shape(name, first=True)
            return _split_dim_zero_or_full(
                parameter=parameter,
                tensor_ids=(name,),
                global_extents=(output_size,),
                global_tail=(),
                rank=rank,
                size=size,
                layer_id=None,
                layout="vision-mlp-column-bias",
            )
        if name.endswith("linear_fc2.weight"):
            output_size, input_size = self._mlp_shape(name, first=False)
            return _row_parallel_or_full_view(
                parameter=parameter,
                tensor_id=name,
                global_shape=(output_size, input_size),
                rank=rank,
                size=size,
                layer_id=None,
                layout="vision-mlp-row",
            )
        if name.endswith("linear_fc2.bias"):
            output_size, _ = self._mlp_shape(name, first=False)
            return _replicated_view(
                parameter=parameter,
                tensor_id=name,
                layer_id=None,
                layout="vision-mlp-row-bias",
                expected_shape=(output_size,),
            )
        if name.endswith(
            (
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
                "merger.norm.weight",
                "merger.norm.bias",
            )
        ) or ("deepstack_merger_list." in name and ".norm." in name):
            return _replicated_view(
                parameter=parameter,
                tensor_id=name,
                layer_id=None,
                layout="vision-norm",
            )
        raise WeightInventoryError(f"unsupported Qwen3.5 vision parameter: {names[0]}")

    @staticmethod
    def _canonical_name(name: str) -> str:
        if name.startswith("model.visual."):
            return "visual." + name.removeprefix("model.visual.")
        return name

    def _mlp_shape(self, name: str, *, first: bool) -> tuple[int, int]:
        hidden = int(self._config.hidden_size)
        if ".blocks." in name:
            intermediate = int(self._config.intermediate_size)
            return (intermediate, hidden) if first else (hidden, intermediate)
        merged_hidden = hidden * int(self._config.spatial_merge_size) ** 2
        output = merged_hidden if first else int(self._config.out_hidden_size)
        return output, merged_hidden


class Qwen35MultimodalWeightSemanticsAdapter:
    def __init__(
        self,
        *,
        text_config: Any,
        vision_config: Any,
        vision_data_parallel: bool = False,
        dynamic_expert_placement: bool = False,
        up_first_w13_parameter_ids: Sequence[int] = (),
        num_fused_shared_experts: int = 0,
    ) -> None:
        self._text = Qwen35WeightSemanticsAdapter(
            config=text_config,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameter_ids,
            num_fused_shared_experts=num_fused_shared_experts,
        )
        self._vision = Qwen35VisionWeightSemanticsAdapter(
            config=vision_config,
            data_parallel=vision_data_parallel,
        )

    def describe_parameter(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]:
        if all(name.startswith(("visual.", "model.visual.")) for name in names):
            return _apply_qwen_coupled_parallel_semantics(
                self._vision.describe_parameter(
                    names=names,
                    parameter=parameter,
                    topology=topology,
                ),
                topology,
            )
        if any(name.startswith(("visual.", "model.visual.")) for name in names):
            raise WeightInventoryError("Qwen3.5 aliases mix vision and text parameters")
        return self._text.describe_parameter(
            names=names,
            parameter=parameter,
            topology=topology,
        )
