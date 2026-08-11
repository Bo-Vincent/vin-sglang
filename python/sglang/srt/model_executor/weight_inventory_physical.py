from __future__ import annotations

from typing import Any

import msgspec

from sglang.srt.model_executor.weight_inventory_contracts import WeightInventoryError


class PhysicalParameter(msgspec.Struct, frozen=True, kw_only=True):
    names: tuple[str, ...]
    parameter: Any
    address: int
    nbytes: int
    storage_address: int
    storage_nbytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: str
    itemsize: int
    device: str


class PhysicalFragmentLookup(msgspec.Struct, frozen=True, kw_only=True):
    physical_names: tuple[str, ...]
    view_byte_offset: int


def contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(shape)
    value = 1
    for index in range(len(shape) - 1, -1, -1):
        result[index] = value
        value *= shape[index]
    return tuple(result)


def collect_physical_parameters(
    *,
    model: Any,
    allowed_devices: frozenset[str],
) -> tuple[PhysicalParameter, ...]:
    grouped: dict[tuple, tuple[Any, list[str]]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        key = _storage_key(parameter)
        if key not in grouped:
            grouped[key] = (parameter, [])
        grouped[key][1].append(name)
    result = [
        _inspect_parameter(
            names=tuple(sorted(names)),
            parameter=parameter,
            allowed_devices=allowed_devices,
        )
        for parameter, names in grouped.values()
    ]
    result.sort(key=lambda item: item.names)
    return tuple(result)


def physical_signature(physical: tuple[PhysicalParameter, ...]) -> tuple:
    return tuple(
        (
            item.names,
            item.address,
            item.nbytes,
            item.storage_address,
            item.storage_nbytes,
            item.shape,
            item.stride,
            item.storage_offset,
            item.dtype,
            item.device,
        )
        for item in physical
    )


def physical_layout_signature(physical: tuple[PhysicalParameter, ...]) -> tuple:
    return tuple(
        (
            item.names,
            item.nbytes,
            item.storage_nbytes,
            item.shape,
            item.stride,
            item.dtype,
            item.itemsize,
        )
        for item in physical
    )


def _dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _storage_key(parameter: Any) -> tuple:
    return (
        int(parameter.untyped_storage().data_ptr()),
        int(parameter.storage_offset()),
        tuple(int(value) for value in parameter.shape),
        tuple(int(value) for value in parameter.stride()),
        _dtype_name(parameter.dtype),
    )


def _inspect_parameter(
    *,
    names: tuple[str, ...],
    parameter: Any,
    allowed_devices: frozenset[str],
) -> PhysicalParameter:
    runtime_name = names[0]
    if getattr(parameter, "is_sparse", False):
        raise WeightInventoryError(f"sparse parameter is unsupported: {runtime_name}")
    layout = getattr(parameter, "layout", None)
    if layout is not None and str(layout) not in ("strided", "torch.strided"):
        raise WeightInventoryError(
            f"non-strided parameter is unsupported: {runtime_name}"
        )
    if not parameter.is_contiguous():
        raise WeightInventoryError(
            f"non-contiguous parameter is unsupported: {runtime_name}"
        )

    device = str(parameter.device.type)
    if device not in allowed_devices:
        raise WeightInventoryError(
            f"parameter device is unsupported: {runtime_name}: {device}"
        )
    shape = tuple(int(value) for value in parameter.shape)
    itemsize = int(parameter.element_size())
    nbytes = int(parameter.numel()) * itemsize
    address = int(parameter.data_ptr())
    if address <= 0 or itemsize <= 0 or nbytes <= 0:
        raise WeightInventoryError(
            f"parameter has no transferable storage: {runtime_name}"
        )
    storage = parameter.untyped_storage()
    storage_address = int(storage.data_ptr())
    storage_nbytes_method = getattr(storage, "nbytes", None)
    storage_offset = int(parameter.storage_offset())
    storage_nbytes = (
        int(storage_nbytes_method())
        if callable(storage_nbytes_method)
        else storage_offset * itemsize + nbytes
    )
    storage_offset_bytes = storage_offset * itemsize
    if (
        storage_address <= 0
        or storage_nbytes <= 0
        or storage_offset_bytes < 0
        or storage_address + storage_offset_bytes != address
        or storage_offset_bytes + nbytes > storage_nbytes
    ):
        raise WeightInventoryError(
            f"parameter storage bounds are invalid: {runtime_name}"
        )
    return PhysicalParameter(
        names=names,
        parameter=parameter,
        address=address,
        nbytes=nbytes,
        storage_address=storage_address,
        storage_nbytes=storage_nbytes,
        shape=shape,
        stride=tuple(int(value) for value in parameter.stride()),
        storage_offset=storage_offset,
        dtype=_dtype_name(parameter.dtype),
        itemsize=itemsize,
        device=device,
    )


__all__ = [
    "PhysicalFragmentLookup",
    "PhysicalParameter",
    "collect_physical_parameters",
    "contiguous_stride",
    "physical_layout_signature",
    "physical_signature",
]
