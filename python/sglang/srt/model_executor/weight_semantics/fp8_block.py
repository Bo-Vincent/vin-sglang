from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang.srt.model_executor.weight_runtime_manifest import (
    LogicalTensorView,
    WeightManifestError,
    WeightParallelTopology,
    WeightSemanticsAdapter,
)

_BLOCK_SHAPE = (128, 128)
_DERIVED_PARAMETER_NAMES = (
    "weight_scale_inv_deepgemm",
    "weight_scale_inv_shuffled",
    "weight_scale_inv_swizzled",
    "w13_weight_scale_inv_deepgemm",
    "w13_weight_scale_inv_shuffled",
    "w13_weight_scale_inv_swizzled",
    "w2_weight_scale_inv_deepgemm",
    "w2_weight_scale_inv_shuffled",
    "w2_weight_scale_inv_swizzled",
)
_WEIGHT_SCALE_ROLES = (
    ("weight", "weight_scale_inv"),
    ("w13_weight", "w13_weight_scale_inv"),
    ("w2_weight", "w2_weight_scale_inv"),
)


@dataclass(frozen=True)
class _BlockFp8Pair:
    weight: Any
    weight_names: tuple[str, ...]
    scale: Any
    scale_names: tuple[str, ...]


def _dtype_name(parameter: Any) -> str:
    return str(parameter.dtype).removeprefix("torch.")


def _shape(parameter: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in parameter.shape)


def _getattr_chain(*objects: Any, name: str, default: Any = None) -> Any:
    for item in objects:
        if item is not None and hasattr(item, name):
            return getattr(item, name)
    return default


def _unravel_index(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinates = [0] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        coordinates[axis] = index % shape[axis]
        index //= shape[axis]
    if index:
        raise WeightManifestError("FP8 logical view offset exceeds weight storage")
    return tuple(coordinates)


def _ravel_index(coordinates: tuple[int, ...], shape: tuple[int, ...]) -> int:
    index = 0
    for coordinate, extent in zip(coordinates, shape):
        if coordinate < 0 or coordinate >= extent:
            raise WeightManifestError("FP8 scale offset exceeds scale storage")
        index = index * extent + coordinate
    return index


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _retag_weight_view(view: LogicalTensorView) -> LogicalTensorView:
    return LogicalTensorView(
        tensor_id=view.tensor_id,
        global_shape=view.global_shape,
        global_offset=view.global_offset,
        local_shape=view.local_shape,
        partition_dim=view.partition_dim,
        byte_offset=view.byte_offset,
        layer_id=view.layer_id,
        expert_id=view.expert_id,
        expert_axis=view.expert_axis,
        layout_fingerprint=(
            f"{view.layout_fingerprint}|serialized-block-fp8:e4m3fn:128x128:weight:v1"
        ),
        shard_dims=view.shard_dims,
    )


def _scale_tensor_id(weight_tensor_id: str) -> str:
    if not weight_tensor_id.endswith(".weight"):
        raise WeightManifestError(
            f"FP8 logical weight has no canonical scale name: {weight_tensor_id}"
        )
    return f"{weight_tensor_id}_scale_inv"


def _scale_view(
    *,
    pair: _BlockFp8Pair,
    weight_view: LogicalTensorView,
) -> LogicalTensorView:
    weight_shape = _shape(pair.weight)
    scale_shape = _shape(pair.scale)
    if len(weight_shape) < 2:
        raise WeightManifestError("block FP8 weights must have at least two axes")

    physical_block_axes = (len(weight_shape) - 2, len(weight_shape) - 1)
    logical_rank = len(weight_view.global_shape)
    if logical_rank < 2:
        raise WeightManifestError("block FP8 logical views must have at least two axes")
    logical_block_axes = (logical_rank - 2, logical_rank - 1)
    expected_scale_shape = (
        *weight_shape[:-2],
        _ceil_div(weight_shape[-2], _BLOCK_SHAPE[0]),
        _ceil_div(weight_shape[-1], _BLOCK_SHAPE[1]),
    )
    if scale_shape != expected_scale_shape:
        raise WeightManifestError(
            "block FP8 inverse scale shape does not match its runtime weight: "
            f"{scale_shape}, expected {expected_scale_shape}"
        )

    weight_itemsize = int(pair.weight.element_size())
    if weight_view.byte_offset % weight_itemsize:
        raise WeightManifestError("FP8 logical view is not element aligned")
    physical_weight_offset = _unravel_index(
        weight_view.byte_offset // weight_itemsize,
        weight_shape,
    )

    physical_scale_offset = list(physical_weight_offset)
    for block_index, axis in enumerate(physical_block_axes):
        block = _BLOCK_SHAPE[block_index]
        if physical_weight_offset[axis] % block:
            raise WeightManifestError(
                "FP8 packed component boundary is not aligned to a 128x128 block"
            )
        physical_scale_offset[axis] //= block

    global_shape = list(weight_view.global_shape)
    global_offset = list(weight_view.global_offset)
    local_shape = list(weight_view.local_shape)
    for block_index, axis in enumerate(logical_block_axes):
        block = _BLOCK_SHAPE[block_index]
        offset = weight_view.global_offset[axis]
        extent = weight_view.local_shape[axis]
        total = weight_view.global_shape[axis]
        if offset % block:
            raise WeightManifestError(
                "FP8 logical shard offset is not aligned to a 128x128 block"
            )
        if offset + extent < total and extent % block:
            raise WeightManifestError(
                "FP8 logical shard extent is not aligned to a 128x128 block"
            )
        global_shape[axis] = _ceil_div(total, block)
        global_offset[axis] = offset // block
        local_shape[axis] = _ceil_div(offset + extent, block) - offset // block

    return LogicalTensorView(
        tensor_id=_scale_tensor_id(weight_view.tensor_id),
        global_shape=tuple(global_shape),
        global_offset=tuple(global_offset),
        local_shape=tuple(local_shape),
        partition_dim=weight_view.partition_dim,
        byte_offset=(
            _ravel_index(tuple(physical_scale_offset), scale_shape)
            * int(pair.scale.element_size())
        ),
        layer_id=weight_view.layer_id,
        expert_id=weight_view.expert_id,
        expert_axis=weight_view.expert_axis,
        layout_fingerprint=(
            f"{weight_view.layout_fingerprint}|serialized-block-fp8:"
            "fp32-inverse-scale:128x128:v1"
        ),
        shard_dims=weight_view.shard_dims,
    )


class SerializedBlockFp8WeightSemanticsAdapter:
    def __init__(
        self,
        *,
        delegate: WeightSemanticsAdapter,
        pairs: tuple[_BlockFp8Pair, ...],
    ) -> None:
        self._delegate = delegate
        self._weight_pairs = {id(pair.weight): pair for pair in pairs}
        self._scale_pairs = {id(pair.scale): pair for pair in pairs}

    def describe_parameter(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]:
        pair = self._weight_pairs.get(id(parameter))
        if pair is not None:
            return tuple(
                _retag_weight_view(view)
                for view in self._delegate.describe_parameter(
                    names=names,
                    parameter=parameter,
                    topology=topology,
                )
            )

        pair = self._scale_pairs.get(id(parameter))
        if pair is not None:
            weight_views = self._delegate.describe_parameter(
                names=pair.weight_names,
                parameter=pair.weight,
                topology=topology,
            )
            return tuple(
                _scale_view(pair=pair, weight_view=view) for view in weight_views
            )

        if _dtype_name(parameter).startswith("float8_"):
            raise WeightManifestError(
                f"FP8 runtime parameter has no inverse block scale: {names[0]}"
            )
        return self._delegate.describe_parameter(
            names=names,
            parameter=parameter,
            topology=topology,
        )


def create_serialized_block_fp8_adapter(
    *,
    model: Any,
    delegate: WeightSemanticsAdapter,
    fp8_gemm_backend: str | None,
    moe_runner_backend: str | None,
) -> SerializedBlockFp8WeightSemanticsAdapter:
    if fp8_gemm_backend != "triton":
        raise WeightManifestError(
            "block FP8 weight manifests require the canonical triton GEMM "
            f"backend; got {fp8_gemm_backend!r}"
        )

    parameter_names: dict[int, list[str]] = {}
    parameters: dict[int, Any] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        parameter_names.setdefault(id(parameter), []).append(name)
        parameters[id(parameter)] = parameter

    pairs: dict[int, _BlockFp8Pair] = {}
    modules = getattr(model, "modules", None)
    if modules is not None:
        for module in modules():
            quant_method = getattr(module, "quant_method", None)
            quant_config = getattr(quant_method, "quant_config", None)
            for derived_name in _DERIVED_PARAMETER_NAMES:
                if getattr(module, derived_name, None) is not None:
                    raise WeightManifestError(
                        f"derived or swizzled FP8 layout is unsupported: {derived_name}"
                    )

            for weight_role, scale_role in _WEIGHT_SCALE_ROLES:
                weight = getattr(module, weight_role, None)
                if weight is None or not _dtype_name(weight).startswith("float8_"):
                    continue
                if id(weight) in pairs:
                    continue
                if _dtype_name(weight) != "float8_e4m3fn":
                    raise WeightManifestError(
                        "block FP8 weight manifests require float8_e4m3fn weights"
                    )
                if bool(
                    _getattr_chain(
                        quant_method,
                        quant_config,
                        name="use_mxfp8",
                        default=False,
                    )
                ):
                    raise WeightManifestError("MXFP8 weight manifests are unsupported")
                if not bool(
                    _getattr_chain(
                        quant_method,
                        module,
                        name="block_quant",
                        default=False,
                    )
                ):
                    raise WeightManifestError(
                        "per-channel or per-tensor FP8 scales are unsupported; "
                        "block quantization is required"
                    )
                if not bool(
                    _getattr_chain(
                        quant_method,
                        quant_config,
                        name="is_checkpoint_fp8_serialized",
                        default=False,
                    )
                ):
                    raise WeightManifestError(
                        "online FP8 quantization is unsupported; a serialized FP8 "
                        "checkpoint is required"
                    )
                activation_scheme = _getattr_chain(
                    quant_config,
                    name="activation_scheme",
                    default="dynamic",
                )
                if activation_scheme != "dynamic":
                    raise WeightManifestError(
                        "block FP8 weight manifests require dynamic activation"
                    )
                block_shape = _getattr_chain(
                    quant_method,
                    quant_config,
                    module,
                    name="weight_block_size",
                )
                if tuple(block_shape or ()) != _BLOCK_SHAPE:
                    raise WeightManifestError(
                        "block FP8 weight manifests require 128x128 blocks"
                    )
                if bool(getattr(quant_method, "use_marlin", False)):
                    raise WeightManifestError(
                        "Marlin-derived FP8 weight layouts are unsupported"
                    )
                if bool(getattr(weight, "is_shuffled", False)):
                    raise WeightManifestError("shuffled FP8 weights are unsupported")
                if (
                    weight_role in ("w13_weight", "w2_weight")
                    and moe_runner_backend != "triton"
                ):
                    raise WeightManifestError(
                        "block FP8 MoE weight manifests require the canonical "
                        f"triton runner backend; got {moe_runner_backend!r}"
                    )

                scale = getattr(module, scale_role, None)
                if scale is None:
                    alternate_scale = getattr(
                        module,
                        scale_role.removesuffix("_inv"),
                        None,
                    )
                    qualifier = "per-channel " if alternate_scale is not None else ""
                    raise WeightManifestError(
                        f"{qualifier}FP8 weight is missing {scale_role}"
                    )
                if _dtype_name(scale) != "float32" or bool(
                    getattr(scale, "format_ue8m0", False)
                ):
                    raise WeightManifestError(
                        "block FP8 inverse scales must use canonical float32 storage"
                    )
                if (
                    id(weight) not in parameter_names
                    or id(scale) not in parameter_names
                ):
                    raise WeightManifestError(
                        "FP8 weight and inverse scale must be runtime parameters"
                    )
                pairs[id(weight)] = _BlockFp8Pair(
                    weight=weight,
                    weight_names=tuple(sorted(parameter_names[id(weight)])),
                    scale=scale,
                    scale_names=tuple(sorted(parameter_names[id(scale)])),
                )

    unpaired = [
        names[0]
        for parameter_id, names in parameter_names.items()
        if _dtype_name(parameters[parameter_id]).startswith("float8_")
        and parameter_id not in pairs
    ]
    if unpaired:
        raise WeightManifestError(
            f"FP8 runtime parameter has no inverse block scale: {unpaired[0]}"
        )
    if not pairs:
        raise WeightManifestError(
            "FP8 quantization is enabled but no serialized block FP8 weights were found"
        )

    scale_ids = [id(pair.scale) for pair in pairs.values()]
    if len(scale_ids) != len(set(scale_ids)):
        raise WeightManifestError(
            "multiple FP8 weights unexpectedly share one inverse scale allocation"
        )

    return SerializedBlockFp8WeightSemanticsAdapter(
        delegate=delegate,
        pairs=tuple(pairs.values()),
    )
