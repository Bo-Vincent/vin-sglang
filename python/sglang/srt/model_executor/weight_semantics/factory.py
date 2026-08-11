from __future__ import annotations

from typing import Any

from sglang.srt.model_executor.weight_inventory_contracts import (
    WeightInventoryError,
    WeightParallelTopology,
    WeightSemanticsAdapter,
)


def create_weight_semantics_adapter(
    *,
    model: Any,
    config: Any,
    topology: WeightParallelTopology,
    quantization: str | None,
    is_multimodal: bool,
    vision_data_parallel: bool,
    dynamic_expert_placement: bool,
    moe_runner_backend: str | None,
    fp8_gemm_backend: str | None,
) -> WeightSemanticsAdapter:
    """Select the SGLang model-owned logical weight semantics adapter."""

    model_type = getattr(config, "model_type", None)
    text_model_types = (
        "qwen3",
        "qwen3_moe",
        "qwen3_5_text",
        "qwen3_5_moe_text",
        "qwen3_next",
    )
    multimodal_model_types = ("qwen3_5", "qwen3_5_moe")
    if is_multimodal and model_type not in multimodal_model_types:
        raise WeightInventoryError(
            f"unsupported multimodal model type for weight inventories: {model_type}"
        )
    if not is_multimodal and model_type not in text_model_types:
        raise WeightInventoryError(
            f"unsupported model type for weight inventories: {model_type}"
        )
    if model_type == "qwen3_next" and moe_runner_backend != "triton":
        raise WeightInventoryError(
            "Qwen3-Next weight inventories require the canonical triton MoE "
            f"runner backend; got {moe_runner_backend!r}"
        )

    from sglang.srt.model_executor.weight_semantics.qwen3 import (
        Qwen3WeightSemanticsAdapter,
    )
    from sglang.srt.model_executor.weight_semantics.qwen3_5 import (
        Qwen35MultimodalWeightSemanticsAdapter,
        Qwen35WeightSemanticsAdapter,
    )
    from sglang.srt.model_executor.weight_semantics.qwen3_next import (
        Qwen3NextWeightSemanticsAdapter,
    )

    up_first_w13_parameter_ids = _up_first_w13_parameter_ids(model)
    num_fused_shared_experts = _runtime_num_fused_shared_experts(model)
    semantic_config = getattr(config, "text_config", None) if is_multimodal else config
    if semantic_config is None:
        raise WeightInventoryError("multimodal model config is missing text_config")
    if int(getattr(semantic_config, "num_experts", 0) or 0) > 0:
        _validate_sglang_moe_rank_decomposition(topology)

    if is_multimodal:
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            raise WeightInventoryError(
                "Qwen3.5 multimodal config is missing vision_config"
            )
        adapter: WeightSemanticsAdapter = Qwen35MultimodalWeightSemanticsAdapter(
            text_config=semantic_config,
            vision_config=vision_config,
            vision_data_parallel=vision_data_parallel,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameter_ids,
            num_fused_shared_experts=num_fused_shared_experts,
        )
    elif model_type == "qwen3_next":
        adapter = Qwen3NextWeightSemanticsAdapter(
            config=config,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameter_ids,
            num_fused_shared_experts=num_fused_shared_experts,
        )
    elif model_type in ("qwen3", "qwen3_moe"):
        adapter = Qwen3WeightSemanticsAdapter(
            config=config,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameter_ids,
        )
    else:
        adapter = Qwen35WeightSemanticsAdapter(
            config=config,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameter_ids,
            num_fused_shared_experts=num_fused_shared_experts,
        )

    if quantization == "fp8":
        from sglang.srt.model_executor.weight_semantics.fp8_block import (
            create_serialized_block_fp8_adapter,
        )

        adapter = create_serialized_block_fp8_adapter(
            model=model,
            delegate=adapter,
            fp8_gemm_backend=fp8_gemm_backend,
            moe_runner_backend=moe_runner_backend,
        )
    return adapter


def _up_first_w13_parameter_ids(model: Any) -> frozenset[int]:
    result = set()
    modules = getattr(model, "modules", None)
    if modules is None:
        return frozenset()
    for module in modules():
        parameter = getattr(module, "w13_weight", None)
        if parameter is None:
            continue
        quant_method = getattr(module, "quant_method", None)
        if bool(getattr(module, "use_flashinfer_trtllm_moe", False)) or bool(
            getattr(quant_method, "load_up_proj_weight_first", False)
        ):
            result.add(id(parameter))
    return frozenset(result)


def _runtime_num_fused_shared_experts(model: Any) -> int:
    modules = getattr(model, "modules", None)
    candidates = modules() if callable(modules) else (model,)
    positive_counts = set()
    for module in candidates:
        value = getattr(module, "num_fused_shared_experts", None)
        if value is None:
            continue
        if type(value) is not int:
            raise WeightInventoryError(
                "runtime fused shared expert count must be an integer"
            )
        count = value
        if count < 0:
            raise WeightInventoryError(
                f"invalid runtime fused shared expert count: {count}"
            )
        if count:
            positive_counts.add(count)
    if len(positive_counts) > 1:
        raise WeightInventoryError(
            "inconsistent runtime fused shared expert counts: "
            f"{tuple(sorted(positive_counts))}"
        )
    return next(iter(positive_counts), 0)


def _validate_sglang_moe_rank_decomposition(
    topology: WeightParallelTopology,
) -> None:
    expected_tp_size = topology.ep_size * topology.moe_tp_size
    if topology.tp_size != expected_tp_size:
        raise WeightInventoryError(
            "SGLang MoE topology must satisfy global TP = EP * MoE-TP when "
            f"MoE-DP is disabled; got {topology.tp_size} != "
            f"{topology.ep_size} * {topology.moe_tp_size}"
        )


__all__ = ["create_weight_semantics_adapter"]
