from __future__ import annotations

from typing import Any, Sequence

from sglang.srt.model_executor.weight_inventory import (
    UnavailableWeightInventoryManager,
    WeightInventoryManager,
)
from sglang.srt.model_executor.weight_inventory_contracts import (
    WeightInventoryError,
    WeightParallelTopology,
)
from sglang.srt.model_executor.weight_snapshot import WeightSnapshotCoordinator


def topology_from_sglang(
    *,
    parallel_state: Any,
    parallel: Any,
) -> WeightParallelTopology:
    if parallel_state.dp_rank is not None:
        dp_rank = parallel_state.dp_rank
        dp_size = parallel_state.dp_size
    else:
        dp_rank = parallel_state.moe_dp_rank or 0
        dp_size = parallel_state.moe_dp_size
    return WeightParallelTopology(
        dp_rank=dp_rank,
        dp_size=dp_size,
        tp_rank=parallel_state.tp_rank,
        tp_size=parallel_state.tp_size,
        pp_rank=parallel_state.pp_rank,
        pp_size=parallel_state.pp_size,
        ep_rank=parallel_state.moe_ep_rank,
        ep_size=parallel_state.moe_ep_size,
        moe_tp_rank=parallel.moe_tp_rank,
        moe_tp_size=parallel.moe_tp_size,
        attention_tp_rank=parallel_state.attn_tp_rank,
        attention_tp_size=parallel_state.attn_tp_size,
    )


def create_sglang_weight_inventory_manager(
    *,
    model: Any,
    config: Any,
    parallel_state: Any,
    parallel: Any,
    allowed_devices: Sequence[str] = ("cuda",),
    quantization: str | None = None,
    lora_enabled: bool = False,
    is_multimodal: bool = False,
    vision_data_parallel: bool = False,
    dynamic_expert_placement: bool = False,
    moe_runner_backend: str | None = None,
    fp8_gemm_backend: str | None = None,
    dp_attention_enabled: bool = False,
    coordinator: WeightSnapshotCoordinator | None = None,
):
    return create_weight_inventory_manager(
        model=model,
        config=config,
        topology=topology_from_sglang(
            parallel_state=parallel_state,
            parallel=parallel,
        ),
        allowed_devices=allowed_devices,
        quantization=quantization,
        lora_enabled=lora_enabled,
        is_multimodal=is_multimodal,
        vision_data_parallel=vision_data_parallel,
        dynamic_expert_placement=dynamic_expert_placement,
        moe_runner_backend=moe_runner_backend,
        fp8_gemm_backend=fp8_gemm_backend,
        dp_attention_enabled=dp_attention_enabled,
        coordinator=coordinator,
    )


def create_weight_inventory_manager(
    *,
    model: Any,
    config: Any,
    topology: WeightParallelTopology,
    allowed_devices: Sequence[str] = ("cuda",),
    quantization: str | None = None,
    lora_enabled: bool = False,
    is_multimodal: bool = False,
    vision_data_parallel: bool = False,
    dynamic_expert_placement: bool = False,
    moe_runner_backend: str | None = None,
    fp8_gemm_backend: str | None = None,
    dp_attention_enabled: bool = False,
    coordinator: WeightSnapshotCoordinator | None = None,
):
    if quantization not in (None, "fp8"):
        return UnavailableWeightInventoryManager(
            f"quantized weight inventories are unsupported: {quantization}"
        )
    if lora_enabled:
        return UnavailableWeightInventoryManager(
            "LoRA weight inventories are unsupported"
        )
    if dp_attention_enabled:
        return UnavailableWeightInventoryManager(
            "DP attention weight inventories are unsupported"
        )

    from sglang.srt.model_executor.weight_semantics.factory import (
        create_weight_semantics_adapter,
    )

    try:
        adapter = create_weight_semantics_adapter(
            model=model,
            config=config,
            topology=topology,
            quantization=quantization,
            is_multimodal=is_multimodal,
            vision_data_parallel=vision_data_parallel,
            dynamic_expert_placement=dynamic_expert_placement,
            moe_runner_backend=moe_runner_backend,
            fp8_gemm_backend=fp8_gemm_backend,
        )
        return WeightInventoryManager(
            model=model,
            adapter=adapter,
            topology=topology,
            allowed_devices=allowed_devices,
            coordinator=coordinator,
        )
    except WeightInventoryError as error:
        return UnavailableWeightInventoryManager(str(error))


__all__ = [
    "create_sglang_weight_inventory_manager",
    "create_weight_inventory_manager",
    "topology_from_sglang",
]
