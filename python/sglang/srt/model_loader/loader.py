# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/model_executor/model_loader/loader.py

from __future__ import annotations

# ruff: noqa: SIM117
import collections
import concurrent.futures
import dataclasses
import fnmatch
import gc
import glob
import hashlib
import importlib
import json
import logging
import math
import os
import re
import shutil
import socket
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager, suppress
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
)

import huggingface_hub
import msgspec
import numpy as np
import torch
from sglang.srt.constants import GIB_BYTES
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    LEGACY_HF_UNATTESTED,
    RemoteInstanceWeightLoaderBackend,
    RemoteInstanceWeightTransferWorldCoordinator,
    bounded_execution_contract_error,
    get_missing_legacy_runtime_v1_apis,
    get_remote_instance_transfer_engine_info_per_rank,
    probe_remote_instance_weight_transfer_capabilities,
    register_memory_region,
    require_bounded_execution_contract,
)
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import get_available_gpu_memory

# Try to import accelerate (optional dependency)
try:
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate.utils import get_max_memory

    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False
    infer_auto_device_map = None
    init_empty_weights = None
    get_max_memory = None

from huggingface_hub import HfApi, hf_hub_download
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.connector import (
    ConnectorType,
    create_remote_connector,
    get_connector_type,
)
from sglang.srt.connector.utils import parse_model_name
from sglang.srt.distributed import (
    model_parallel_is_initialized,
)
from sglang.srt.distributed.parallel_state import get_world_group
from sglang.srt.environ import envs
from sglang.srt.layers.modelopt_utils import QUANT_CFG_CHOICES
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightParallelRank,
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
    WeightRuntimeManifest,
)
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    trigger_transferring_weights_request,
)
from sglang.srt.model_loader.utils import (
    get_model_architecture,
    set_default_torch_dtype,
)
from sglang.srt.model_loader.weight_utils import (
    buffered_multi_thread_safetensors_weights_iterator,
    download_safetensors_index_file_from_hf,
    download_weights_from_hf,
    fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    get_gguf_extra_tensor_names,
    get_quant_config,
    gguf_quant_weights_iterator,
    initialize_dummy_weights,
    maybe_add_mtp_safetensors,
    multi_thread_pt_weights_iterator,
    np_cache_weights_iterator,
    pt_weights_iterator,
    safetensors_weights_iterator,
    set_runai_streamer_env,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import (
    get_bool_env_var,
    get_device_capability,
    is_npu,
    is_pin_memory_available,
    rank0_log,
    set_weight_attrs,
)
from sglang.srt.utils.common import is_cuda_alike, temp_set_env
from sglang.srt.weight_transfer.api import (
    execute_weight_load,
    preflight_weight_transfer,
    prepare_weight_load_from_plan,
)
from sglang.srt.weight_transfer.binding import (
    project_source_bindings,
    runtime_manifest_to_parts,
)
from sglang.srt.weight_transfer.planner import (
    plan_weight_transfer,
    plan_weight_transfer_to_local_target,
    project_weight_transfer_plan_to_targets,
)
from sglang.srt.weight_transfer.provider import (
    WeightLoadRequest,
    WeightTargetLoadMode,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
    WeightTransferExecutionContext,
    WeightTransferProvider,
)
from sglang.srt.weight_transfer.remote_protocol import (
    ARTIFACT_WEIGHT_VERSION_V1,
    HF_REVISION_V1,
    PLACEMENT_BINDING_V1,
    RUNTIME_MANIFEST_V1,
    validate_manifest_revision_semantics,
)
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

# Constants for memory management
DEFAULT_GPU_MEMORY_FRACTION_FOR_CALIBRATION = (
    0.8  # Reserve 20% GPU memory headroom for ModelOpt calibration
)

if TYPE_CHECKING:
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

_is_npu = is_npu()
# ModelOpt: QUANT_CFG_CHOICES is imported from modelopt_utils.py
# which contains the complete mapping of quantization config choices

logger = logging.getLogger(__name__)

_LEGACY_UNKNOWN_TRANSFER_QUARANTINE = []
_HETEROGENEOUS_UNKNOWN_DRAIN_MAX_ATTEMPTS = 30
_HETEROGENEOUS_UNKNOWN_DRAIN_TIMEOUT_MS = 1000
_HETEROGENEOUS_QUARANTINE_COORDINATION_TIMEOUT_SEC = 5.0


def _configured_weight_artifact_revision() -> str:
    server_args = get_server_args()
    revision = (
        "default"
        if server_args is None
        else getattr(server_args, "weight_version", None)
    )
    if type(revision) is not str or not revision:
        raise RuntimeError("the target weight artifact revision is not initialized")
    return revision


def _resolve_target_weight_revisions(
    *,
    target_artifact_revision: str | None,
    target_hf_revision: str | None,
    target_revision: str | None,
) -> tuple[str, str]:
    if target_revision is not None:
        if target_artifact_revision is not None or target_hf_revision is not None:
            raise ValueError(
                "target_revision cannot be combined with explicit artifact or "
                "Hugging Face revisions"
            )
        target_artifact_revision = target_revision
        target_hf_revision = target_revision
    if type(target_artifact_revision) is not str or not target_artifact_revision:
        raise ValueError("target artifact revision must be a non-empty string")
    if type(target_hf_revision) is not str or not target_hf_revision:
        raise ValueError("target Hugging Face revision must be a non-empty string")
    return target_artifact_revision, target_hf_revision


def _resolve_remote_manifest_revision(
    *,
    manifest_format: str,
    source_revision_semantics: str,
    allow_legacy_hf_fallback: bool,
    target_artifact_revision: str,
    target_hf_revision: str,
) -> str:
    if source_revision_semantics == LEGACY_HF_UNATTESTED:
        if not allow_legacy_hf_fallback:
            raise RuntimeError(
                "source did not attest artifact weight version "
                f"{target_artifact_revision}"
            )
        return target_hf_revision

    validate_manifest_revision_semantics(
        manifest_format,
        source_revision_semantics,
    )
    if source_revision_semantics == ARTIFACT_WEIGHT_VERSION_V1:
        return target_artifact_revision
    if not allow_legacy_hf_fallback:
        raise RuntimeError(
            f"source did not attest artifact weight version {target_artifact_revision}"
        )
    return target_hf_revision


def _allow_legacy_hf_manifest_revision(
    capabilities,
    *,
    target_artifact_revision: str,
    target_hf_revision: str,
) -> bool:
    if capabilities.legacy_planner and not capabilities.native_executor:
        return True
    return (
        capabilities.supports_runtime_v1
        and target_artifact_revision == target_hf_revision
    )


@dataclasses.dataclass
class _HeterogeneousUnknownTransferQuarantine:
    source_transfer_id: str
    pending_transfer_id: str | None
    transfer_executor: Any
    resources: ExitStack
    coordinator: Any
    owners: tuple[Any, ...]
    terminal_status: str | None = None
    resources_closed: bool = False


@dataclasses.dataclass(frozen=True)
class _RemoteInstanceWeightLoadAttestor:
    coordinator: Any
    target_resource: Any
    target_binding: Any

    def attest(self, request: Any) -> None:
        self.coordinator.raise_if_failed()
        if tuple(request.plan.target_bindings) != (self.target_binding,):
            raise RuntimeError(
                "weight load request target binding changed before execution"
            )
        attest_binding = getattr(self.target_resource, "attest_binding", None)
        if not callable(attest_binding):
            raise RuntimeError("target runtime does not support binding attestation")
        attest_binding(self.target_binding)


class _LegacyMooncakeWeightBackend(Protocol):
    RuntimeManifest: Any
    MemoryRegistrationLease: Any
    MooncakeTransferEngineReader: Callable[..., Any]
    TransferCompletionUnknownError: type[BaseException]
    TransferEngineError: type[Exception]
    plan_runtime_transfer_to_local_target: Callable[..., Any]
    bounded_execution_contract_version: int
    supports_bounded_execution: bool


@dataclasses.dataclass(frozen=True)
class _PreparedNativeHeterogeneousWeightLoad:
    source_placements: tuple[WeightPlacementManifest, ...]
    source_bindings: tuple[WeightRuntimeBindingManifest, ...]
    target_placement: WeightPlacementManifest
    target_binding: WeightRuntimeBindingManifest
    load_request: WeightLoadRequest
    load_attestor: _RemoteInstanceWeightLoadAttestor
    load_preflight: object
    transfer_executor: WeightTransferProvider


@dataclasses.dataclass(frozen=True)
class _TargetPlacementEnvelope:
    world_rank: int
    parallel_rank: WeightParallelRank | None
    placement: WeightPlacementManifest | None
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class _RankLocalTransferEnvelope:
    world_rank: int
    parallel_rank: WeightParallelRank | None
    target_placement_id: str | None
    logical_plan: Any | None
    source_bindings: tuple[WeightRuntimeBindingManifest, ...] = ()
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class _RankLocalProviderPreflightOutcome:
    world_rank: int
    error: str | None = None
    capability_fingerprint: tuple[str, bool, bool, bool] | None = None


def _compact_target_planning_error(error: BaseException) -> str:
    message = f"{type(error).__name__}: {error}"
    return message[:1024]


def _compact_provider_preflight_error(stage: str, error: BaseException) -> str:
    return f"{stage} failed: {type(error).__name__}: {error}"[:1024]


def _vote_provider_preflight(
    world_group: Any,
    local_error: str | None,
    capability_fingerprint: tuple[str, bool, bool, bool] | None,
) -> bool:
    world_size = getattr(world_group, "world_size", 1)
    local_rank = getattr(world_group, "rank_in_group", 0 if world_size == 1 else None)
    if (
        type(world_size) is not int
        or world_size <= 0
        or type(local_rank) is not int
        or not 0 <= local_rank < world_size
    ):
        logger.error("Target-world provider preflight rank metadata is invalid")
        return False

    local_outcome = _RankLocalProviderPreflightOutcome(
        world_rank=local_rank,
        error=local_error,
        capability_fingerprint=capability_fingerprint,
    )
    if world_size == 1:
        outcomes = [local_outcome]
    else:
        execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=(
                time.time() + _HETEROGENEOUS_QUARANTINE_COORDINATION_TIMEOUT_SEC
            )
        )
        try:
            outcomes = world_group.all_gather_object(
                local_outcome,
                phase="heterogeneous_provider.preflight",
                execution_context=execution_context,
            )
        except Exception:
            logger.exception("Cannot coordinate target-world provider preflight")
            return False

    by_rank = {}
    if isinstance(outcomes, (list, tuple)) and len(outcomes) == world_size:
        for outcome in outcomes:
            if (
                not isinstance(outcome, _RankLocalProviderPreflightOutcome)
                or type(outcome.world_rank) is not int
                or not 0 <= outcome.world_rank < world_size
                or outcome.world_rank in by_rank
                or (
                    outcome.error is not None
                    and (type(outcome.error) is not str or not outcome.error)
                )
                or (
                    outcome.error is None
                    and (
                        not isinstance(outcome.capability_fingerprint, tuple)
                        or len(outcome.capability_fingerprint) != 4
                        or outcome.capability_fingerprint[0] not in {"native", "legacy"}
                        or any(
                            type(value) is not bool
                            for value in outcome.capability_fingerprint[1:]
                        )
                    )
                )
            ):
                break
            by_rank[outcome.world_rank] = outcome
    if tuple(sorted(by_rank)) != tuple(range(world_size)):
        logger.error("Target-world provider preflight vote is invalid")
        return False

    for rank in range(world_size):
        error = by_rank[rank].error
        if error is not None:
            logger.error(
                "Target-world provider preflight failed at rank %d: %s",
                rank,
                error,
            )
            return False
    reference_fingerprint = by_rank[0].capability_fingerprint
    for rank in range(1, world_size):
        if by_rank[rank].capability_fingerprint != reference_fingerprint:
            logger.error(
                "Target-world provider preflight capability mismatch: "
                "rank %d differs from rank 0",
                rank,
            )
            return False
    return True


def _preflight_bounded_native_weight_transfer(
    provider: WeightTransferProvider,
    request: WeightLoadRequest,
    *,
    attestor: _RemoteInstanceWeightLoadAttestor,
) -> object:
    preflight = preflight_weight_transfer(
        provider,
        request,
        attestor=attestor,
    )
    capabilities = getattr(preflight, "_capabilities", None)
    require_bounded_execution_contract(
        provider,
        role="native provider",
        supports_bounded_execution=getattr(
            capabilities,
            "supports_bounded_execution",
            None,
        ),
    )
    return preflight


def _placement_parallel_rank(
    placement: WeightPlacementManifest,
) -> WeightParallelRank:
    ranks = {tensor.rank for tensor in placement.tensors}
    if len(ranks) != 1:
        raise ValueError("target placement must belong to one parallel rank")
    return next(iter(ranks))


@dataclasses.dataclass(frozen=True)
class _PreparedLegacyHeterogeneousWeightLoad:
    plan: Any
    source_manifests: tuple[Any, ...]
    source_registrations: tuple[Any, ...]
    target_registrations: tuple[Any, ...]
    target_manifest: Any
    backend: _LegacyMooncakeWeightBackend


_PreparedHeterogeneousWeightLoad = (
    _PreparedNativeHeterogeneousWeightLoad | _PreparedLegacyHeterogeneousWeightLoad
)


def _load_legacy_mooncake_weight_backend() -> _LegacyMooncakeWeightBackend:
    try:
        backend = importlib.import_module("mooncake.weight_transfer")
    except ImportError as error:
        raise RuntimeError(
            "Mooncake legacy runtime_v1 support is unavailable"
        ) from error

    missing = get_missing_legacy_runtime_v1_apis(backend)
    if missing:
        raise RuntimeError(
            "Mooncake legacy runtime_v1 support is missing APIs: " + ", ".join(missing)
        )
    return cast(_LegacyMooncakeWeightBackend, backend)


def _legacy_runtime_v1_supports_bounded_execution(
    backend: _LegacyMooncakeWeightBackend,
) -> bool:
    return (
        bounded_execution_contract_error(
            backend,
            role="legacy backend",
            supports_bounded_execution=getattr(
                backend,
                "supports_bounded_execution",
                None,
            ),
        )
        is None
    )


_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE: list[
    _HeterogeneousUnknownTransferQuarantine
] = []
_WEIGHT_SNAPSHOT_UNKNOWN_LOAD_QUARANTINE: list[tuple[Any, ExitStack, Any]] = []
_WEIGHT_SNAPSHOT_CLEANUP_QUARANTINE: list[Any] = []
_WEIGHT_SNAPSHOT_ACTIVATION_QUARANTINE: list[Any] = []
_WEIGHT_SNAPSHOT_CLEANUP_TIMEOUT_MS = 5_000


def _transfer_completion_status_name(status: Any) -> str:
    name = getattr(status, "name", None)
    if isinstance(name, str):
        return name
    try:
        value = int(status)
    except (TypeError, ValueError):
        value = None
    by_value = {
        0: "COMPLETED",
        -1: "FAILED_DRAINED",
        -2: "COMPLETION_UNKNOWN",
    }
    if value in by_value:
        return by_value[value]
    text = str(status)
    for candidate in by_value.values():
        if candidate in text:
            return candidate
    return text


_HETEROGENEOUS_TERMINAL_STATUSES = frozenset(
    {"COMPLETED", "FAILED_DRAINED", "NO_SUBMISSION"}
)
_HETEROGENEOUS_ALL_COMPLETION_STATUSES = frozenset(
    {
        *_HETEROGENEOUS_TERMINAL_STATUSES,
        "COMPLETION_UNKNOWN",
    }
)


def _drain_quarantined_transfer(
    item: _HeterogeneousUnknownTransferQuarantine,
    *,
    max_attempts: int,
    timeout_ms: int,
) -> str:
    if item.terminal_status in _HETEROGENEOUS_TERMINAL_STATUSES:
        return item.terminal_status
    if not item.pending_transfer_id:
        return "COMPLETION_UNKNOWN"
    for _ in range(max_attempts):
        try:
            drain_completion = getattr(
                item.transfer_executor,
                "drain_completion",
                None,
            )
            if callable(drain_completion):
                status = drain_completion(
                    item.pending_transfer_id,
                    timeout_ms=timeout_ms,
                )
            else:
                status = item.transfer_executor.drain_pending_transfer(
                    item.pending_transfer_id,
                    timeout_ms=timeout_ms,
                )
        except Exception:
            logger.exception(
                "Failed to drain quarantined heterogeneous transfer %s",
                item.pending_transfer_id,
            )
            continue
        status_name = _transfer_completion_status_name(status)
        if status_name not in _HETEROGENEOUS_ALL_COMPLETION_STATUSES:
            logger.error(
                "Quarantined heterogeneous transfer %s returned invalid "
                "completion status %r",
                item.pending_transfer_id,
                status,
            )
            continue
        if status_name in _HETEROGENEOUS_TERMINAL_STATUSES:
            item.terminal_status = status_name
            return status_name
    return "COMPLETION_UNKNOWN"


def _validate_quarantine_metadata(
    gathered: Any,
    *,
    world_size: int,
    item_count: int,
) -> tuple[tuple[tuple[int, str, str], ...], ...]:
    if not isinstance(gathered, (list, tuple)) or len(gathered) != world_size:
        raise ValueError("invalid quarantine metadata world size")
    normalized = []
    source_order = None
    for rank, row in enumerate(gathered):
        if type(row) is not tuple or len(row) != item_count:
            raise ValueError("invalid quarantine metadata container")
        checked = []
        for entry in row:
            if (
                type(entry) is not tuple
                or len(entry) != 3
                or type(entry[0]) is not int
                or entry[0] != rank
                or type(entry[1]) is not str
                or not entry[1]
                or type(entry[2]) is not str
                or not entry[2]
            ):
                raise ValueError("invalid quarantine metadata entry")
            checked.append(entry)
        checked = tuple(checked)
        current_order = tuple(entry[1] for entry in checked)
        if source_order is None:
            source_order = current_order
        elif current_order != source_order:
            raise ValueError("quarantine source transfer order differs")
        normalized.append(checked)
    return tuple(normalized)


def _validate_quarantine_statuses(
    gathered: Any,
    *,
    metadata: tuple[tuple[tuple[int, str, str], ...], ...],
) -> tuple[tuple[tuple[int, str, str, str], ...], ...]:
    if not isinstance(gathered, (list, tuple)) or len(gathered) != len(metadata):
        raise ValueError("invalid quarantine status world size")
    normalized = []
    for rank, (row, expected_row) in enumerate(zip(gathered, metadata, strict=True)):
        if type(row) is not tuple or len(row) != len(expected_row):
            raise ValueError("invalid quarantine status container")
        checked = []
        for entry, expected in zip(row, expected_row, strict=True):
            if (
                type(entry) is not tuple
                or len(entry) != 4
                or type(entry[0]) is not int
                or entry[:3] != expected
                or type(entry[3]) is not str
                or entry[3] not in _HETEROGENEOUS_ALL_COMPLETION_STATUSES
            ):
                raise ValueError("invalid quarantine status entry")
            checked.append(entry)
        normalized.append(tuple(checked))
    return tuple(normalized)


def _validate_quarantine_closed(
    gathered: Any,
    *,
    metadata: tuple[tuple[tuple[int, str, str], ...], ...],
) -> tuple[tuple[tuple[int, str, str, bool], ...], ...]:
    if not isinstance(gathered, (list, tuple)) or len(gathered) != len(metadata):
        raise ValueError("invalid quarantine close world size")
    normalized = []
    for row, expected_row in zip(gathered, metadata, strict=True):
        if type(row) is not tuple or len(row) != len(expected_row):
            raise ValueError("invalid quarantine close container")
        checked = []
        for entry, expected in zip(row, expected_row, strict=True):
            if (
                type(entry) is not tuple
                or len(entry) != 4
                or entry[:3] != expected
                or type(entry[3]) is not bool
            ):
                raise ValueError("invalid quarantine close entry")
            checked.append(entry)
        normalized.append(tuple(checked))
    return tuple(normalized)


def _validate_quarantine_releases(
    gathered: Any,
    *,
    metadata: tuple[tuple[tuple[int, str, str], ...], ...],
) -> tuple[tuple[tuple[int, str, str, bool], ...], ...]:
    if not isinstance(gathered, (list, tuple)) or len(gathered) != len(metadata):
        raise ValueError("invalid quarantine release world size")
    normalized = []
    for row, expected_row in zip(gathered, metadata, strict=True):
        if type(row) is not tuple or len(row) != len(expected_row):
            raise ValueError("invalid quarantine release container")
        checked = []
        for entry, expected in zip(row, expected_row, strict=True):
            if (
                type(entry) is not tuple
                or len(entry) != 4
                or entry[:3] != expected
                or type(entry[3]) is not bool
            ):
                raise ValueError("invalid quarantine release entry")
            checked.append(entry)
        normalized.append(tuple(checked))
    return tuple(normalized)


def drain_heterogeneous_weight_transfer_quarantine(
    *,
    max_attempts: int = 1,
    timeout_ms: int = _HETEROGENEOUS_UNKNOWN_DRAIN_TIMEOUT_MS,
    execution_context: WeightTransferExecutionContext | None = None,
) -> bool:
    """Drain a quarantined target world without releasing live DMA buffers."""

    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    if type(timeout_ms) is not int or timeout_ms < 0:
        raise ValueError("timeout_ms must be a non-negative integer")
    if execution_context is None:
        execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=(
                time.time() + _HETEROGENEOUS_QUARANTINE_COORDINATION_TIMEOUT_SEC
            )
        )

    world_group = get_world_group()
    world_size = getattr(world_group, "world_size", 1)
    rank = getattr(world_group, "rank_in_group", 0)
    local_items = tuple(_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE)
    if world_size == 1 and not local_items:
        return True
    local_metadata = tuple(
        (
            rank,
            item.source_transfer_id,
            item.pending_transfer_id,
        )
        for item in local_items
    )
    try:
        gathered_metadata = world_group.all_gather_object(
            local_metadata,
            phase="heterogeneous_quarantine.metadata",
            execution_context=execution_context,
        )
        metadata = _validate_quarantine_metadata(
            gathered_metadata,
            world_size=world_size,
            item_count=len(local_items),
        )
    except Exception:
        logger.exception("Failed to validate quarantined target-world metadata")
        return False
    if not local_items:
        return True

    local_statuses = tuple(
        (
            rank,
            item.source_transfer_id,
            item.pending_transfer_id,
            _drain_quarantined_transfer(
                item,
                max_attempts=max_attempts,
                timeout_ms=timeout_ms,
            ),
        )
        for item in local_items
    )
    try:
        gathered_statuses = world_group.all_gather_object(
            local_statuses,
            phase="heterogeneous_quarantine.status",
            execution_context=execution_context,
        )
        statuses = _validate_quarantine_statuses(
            gathered_statuses,
            metadata=metadata,
        )
    except Exception:
        logger.exception("Failed to validate quarantined target-world statuses")
        return False

    terminal_indices = tuple(
        index
        for index in range(len(local_items))
        if all(row[index][3] in _HETEROGENEOUS_TERMINAL_STATUSES for row in statuses)
    )
    if not terminal_indices:
        return False

    locally_closed = []
    for index, item in enumerate(local_items):
        if index in terminal_indices and not item.resources_closed:
            try:
                item.resources.close()
            except Exception:
                logger.exception(
                    "Failed to close quarantined target resources for %s",
                    item.source_transfer_id,
                )
            else:
                item.resources_closed = True
        locally_closed.append(
            (
                rank,
                item.source_transfer_id,
                item.pending_transfer_id,
                item.resources_closed,
            )
        )
    try:
        gathered_closed = world_group.all_gather_object(
            tuple(locally_closed),
            phase="heterogeneous_quarantine.closed",
            execution_context=execution_context,
        )
        closed = _validate_quarantine_closed(
            gathered_closed,
            metadata=metadata,
        )
    except Exception:
        logger.exception("Failed to validate quarantined target-world resource closure")
        return False

    local_releases = []
    for index, item in enumerate(local_items):
        release_success = False
        if index not in terminal_indices or not all(row[index][3] for row in closed):
            local_releases.append(
                (
                    rank,
                    item.source_transfer_id,
                    item.pending_transfer_id,
                    release_success,
                )
            )
            continue
        item = local_items[index]
        try:
            release_success = item.coordinator.release_after_terminal_recovery(
                completion_ticket=item.pending_transfer_id,
                local_terminal_status=statuses[rank][index][3],
            )
        except Exception:
            logger.exception(
                "Failed to release recovered source transfer %s",
                item.source_transfer_id,
            )
        local_releases.append(
            (
                rank,
                item.source_transfer_id,
                item.pending_transfer_id,
                bool(release_success),
            )
        )

    try:
        gathered_releases = world_group.all_gather_object(
            tuple(local_releases),
            phase="heterogeneous_quarantine.released",
            execution_context=execution_context,
        )
        releases = _validate_quarantine_releases(
            gathered_releases,
            metadata=metadata,
        )
    except Exception:
        logger.exception("Failed to validate quarantined target-world source release")
        return False

    released = {
        id(local_items[index])
        for index in terminal_indices
        if all(row[index][3] for row in releases)
    }
    if released:
        _HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE[:] = [
            item
            for item in _HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE
            if id(item) not in released
        ]
    return not _HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE


@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_infos: Dict[str, Dict] = {}

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_data = p.data
            device_data = p.data.to(target_device)
            original_infos[name] = dict(
                device=p.device,
                original_data=original_data,
                device_data=device_data,
            )
            p.data = device_data
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        pin_memory = is_pin_memory_available()
        for name, p in module.named_parameters():
            if name in original_infos:
                original_info = original_infos[name]
                device_data = original_info["device_data"]
                original_data = original_info["original_data"]
                original_device: torch.device = original_info["device"]

                if (
                    (device_data.device == p.data.device)
                    and (device_data.data_ptr() == p.data.data_ptr())
                    and (device_data.shape == p.data.shape)
                    and (device_data.dtype == p.data.dtype)
                ):
                    original_data.copy_(p.data.to(original_data.device))
                    p.data = original_data
                elif original_device.type == "cpu":
                    # `torch.empty_like` does not support `pin_memory` argument
                    cpu_data = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device="cpu",
                        pin_memory=pin_memory,
                    )
                    cpu_data.copy_(p.data)
                    p.data = cpu_data
                else:
                    p.data = p.data.to(original_device)
        # New parameters or parameters already on target device are untouched


logger = logging.getLogger(__name__)


def _get_quantization_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
) -> Optional[QuantizationConfig]:
    """Get the quantization config."""
    model_class, _ = get_model_architecture(model_config)
    packed_modules_mapping = getattr(model_class, "packed_modules_mapping", {})
    remap_prefix = getattr(model_class, "remap_prefix", None)
    # TODO: we should remove this code and switch to the packed_modules_mapping declared inside the modeling files
    if model_config.quantization == "quark":
        packed_modules_mapping.update(
            {
                "gate_up_proj": ["gate_proj", "up_proj"],
                "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
            }
        )

    if _is_npu:
        packed_modules_mapping.update(
            {
                "visual": {
                    "qkv_proj": ["qkv"],
                    "gate_up_proj": ["gate_proj", "up_proj"],
                },
                "vision_model": {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                    "proj": ["out_proj"],
                },
                "model": {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                    "gate_up_proj": ["gate_proj", "up_proj"],
                    "fused_qkv_a_proj_with_mqa": [
                        "q_a_proj",
                        "kv_a_proj_with_mqa",
                    ],
                },
            }
        )

    if model_config.quantization is not None:
        quant_config = get_quant_config(
            model_config, load_config, packed_modules_mapping, remap_prefix
        )
        # (yizhang2077) workaround for nvidia/Llama-4-Maverick-17B-128E-Eagle3
        if quant_config is None:
            return None
        # Carry DSV4 expert layout into quant configs so downstream readers don't read env.
        from sglang.srt.layers.quantization.fp8 import Fp8Config

        if isinstance(quant_config, Fp8Config):
            quant_config.is_fp4_experts = model_config.is_fp4_experts
            quant_config.dequant_fp4_to_fp8 = envs.SGLANG_DSV4_FP4_DEQUANT.get()
            # Handle hybrid NVFP4 moe (nvidia/DeepSeek-V4-Pro-NVFP4)
            nvfp4_meta = model_config.nvfp4_moe_meta
            if nvfp4_meta is not None:
                from sglang.srt.layers.quantization.modelopt_quant import (
                    HybridFp8NvFp4Config,
                    ModelOptFp4Config,
                )

                # MTP MoE layers (model.decoder.*) are not NVFP4 quantized.
                nvfp4_exclude_modules = list(
                    nvfp4_meta.get("exclude_modules") or []
                ) + ["model.decoder.*"]
                nvfp4_config = ModelOptFp4Config(
                    is_checkpoint_nvfp4_serialized=True,
                    group_size=int(nvfp4_meta["group_size"]),
                    exclude_modules=nvfp4_exclude_modules,
                    packed_modules_mapping=quant_config.packed_modules_mapping,
                )
                quant_config = HybridFp8NvFp4Config(
                    fp8_config=quant_config, nvfp4_config=nvfp4_config
                )
        elif quant_config.get_name() == "humming":
            quant_config.is_fp4_experts = model_config.is_fp4_experts
        if not _is_npu:
            major, minor = get_device_capability()

            if major is not None and minor is not None:
                assert 0 <= minor < 10
                capability = major * 10 + minor
                if capability < quant_config.get_min_capability():
                    raise ValueError(
                        f"The quantization method {model_config.quantization} "
                        "is not supported for the current GPU. "
                        f"Minimum capability: {quant_config.get_min_capability()}. "
                        f"Current capability: {capability}."
                    )
        supported_dtypes = quant_config.get_supported_act_dtypes()
        if model_config.dtype not in supported_dtypes:
            raise ValueError(
                f"{model_config.dtype} is not supported for quantization "
                f"method {model_config.quantization}. Supported dtypes: "
                f"{supported_dtypes}"
            )
        hf_to_sglang_mapper = getattr(model_class, "hf_to_sglang_mapper", None)
        # pass mappings by reference to quant_config
        if hf_to_sglang_mapper is not None and quant_config is not None:
            quant_config.apply_weight_name_mapper(hf_to_sglang_mapper)
        return quant_config
    return None


def _initialize_model(
    model_config: ModelConfig,
    load_config: LoadConfig,
    quant_config: Optional[QuantizationConfig] = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    model_class, _ = get_model_architecture(model_config)
    kwargs = {
        "config": model_config.hf_config,
        "quant_config": quant_config,
    }

    # Only add sparse head kwargs if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set()
    if envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.is_set():
        kwargs["sparse_head"] = envs.SGLANG_EMBEDDINGS_SPARSE_HEAD.get()
        kwargs["model_path"] = model_config.model_path

    if load_config.draft_model_idx is not None:
        kwargs["draft_model_idx"] = load_config.draft_model_idx

    return model_class(**kwargs)


def _post_load_weights(model: nn.Module) -> None:
    # Loaders that bypass `model.load_weights()` (dummy / sharded state / remote instance /
    # remote fs) must trigger the model's post-load fixup explicitly; `model.load_weights()`
    # would normally do it internally. NextN subclasses override the method to fill in
    # `is_nextn=True`, so the loader doesn't need to know.
    for module in model.modules():
        refresh = getattr(module, "refresh_runtime_weight_state", None)
        if callable(refresh):
            refresh()
    if hasattr(model, "post_load_weights"):
        model.post_load_weights()


class WeightSnapshotActivation(Protocol):
    def activate(self) -> None: ...

    def close(self) -> None: ...


class BaseModelLoader(ABC):
    """Base class for model loaders."""

    def __init__(self, load_config: LoadConfig):
        self.load_config = load_config

    @abstractmethod
    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    @abstractmethod
    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        """Load a model with the given configurations."""
        raise NotImplementedError

    def take_pending_weight_snapshot_activation(
        self,
    ) -> WeightSnapshotActivation | None:
        return None


class DefaultModelLoader(BaseModelLoader):
    """Model loader that can load different file types from disk."""

    # default number of thread when enable multithread weight loading
    DEFAULT_NUM_THREADS = 8

    _MTP_PATTERN = re.compile(r"model\.mtp\.layers\.(\d+)\.")

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: Optional[str]
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        model_config: Optional[ModelConfig] = None
        """The model configuration (for checking architecture, etc)."""

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            return cls(
                model_config.model_path,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                model_config=model_config,
            )

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"enable_multithread_load", "num_threads"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

    def _maybe_download_from_modelscope(
        self, model: str, revision: Optional[str]
    ) -> str:
        """Download model from ModelScope hub if SGLANG_USE_MODELSCOPE is True.

        Returns the path to the downloaded model, or the original model path if
        not downloaded from ModelScope."""
        if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
            # download model from ModelScope hub,
            # lazy import so that modelscope is not required for normal use.
            # pylint: disable=C.
            from modelscope.hub.snapshot_download import snapshot_download

            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=self.load_config.download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=self.load_config.ignore_patterns,
                )
            else:
                model_path = model
            return model_path
        return model

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        model_name_or_path = self._maybe_download_from_modelscope(
            model_name_or_path, revision
        )

        is_local = os.path.isdir(model_name_or_path)
        load_format = self.load_config.load_format
        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        # Some quantized models use .pt files for storing the weights.
        if load_format == LoadFormat.AUTO:
            allow_patterns = ["*.safetensors", "*.bin"]
        elif (
            load_format == LoadFormat.SAFETENSORS
            or load_format == LoadFormat.FASTSAFETENSORS
        ):
            use_safetensors = True
            allow_patterns = ["*.safetensors"]
        elif load_format == LoadFormat.MISTRAL:
            use_safetensors = True
            allow_patterns = ["consolidated*.safetensors"]
            index_file = "consolidated.safetensors.index.json"
        elif load_format == LoadFormat.PT:
            allow_patterns = ["*.pt"]
        elif load_format == LoadFormat.NPCACHE:
            allow_patterns = ["*.bin"]
        elif load_format == LoadFormat.DUMMY:
            raise ValueError(
                "DUMMY load_format should use DummyModelLoader and not call _prepare_weights"
            )
        else:
            raise ValueError(f"Unknown load_format: {load_format}")

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if not is_local:
            hf_folder = download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        else:
            hf_folder = model_name_or_path

        server_args = get_server_args()
        if server_args and server_args.model_checksum is not None:
            from sglang.srt.utils.model_file_verifier import verify

            checksums_source = server_args.model_checksum or model_name_or_path
            verify(model_path=hf_folder, checksums_source=checksums_source)

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            if not is_local:
                download_safetensors_index_file_from_hf(
                    model_name_or_path,
                    index_file,
                    self.load_config.download_dir,
                    revision,
                )
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        # Sort and optionally stagger weight files (see SGLANG_SORT_WEIGHT_FILES).
        # k=-1: no sort; k=0: sort only; k>0: sort + stagger by (tp_rank*k).
        k = envs.SGLANG_SORT_WEIGHT_FILES.get()
        if k >= 0:
            hf_weights_files.sort()
            if k > 0:
                tp_size = get_parallel().tp_size
                if tp_size > 1:
                    tp_rank = get_parallel().tp_rank
                    group_size = tp_size * k
                    staggered: List[str] = []
                    for i in range(0, len(hf_weights_files), group_size):
                        group = hf_weights_files[i : i + group_size]
                        n = len(group)
                        staggered.extend(group[(j + tp_rank * k) % n] for j in range(n))
                    hf_weights_files = staggered

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: Source
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        extra_config = self.load_config.model_loader_extra_config
        use_multithread = extra_config.get("enable_multithread_load", True)
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision, source.fall_back_to_pt
        )

        if use_safetensors and source.model_config is not None:
            hf_weights_files = maybe_add_mtp_safetensors(
                hf_weights_files,
                hf_folder,
                "model.safetensors.index.json",
                source.model_config.hf_config,
            )

        if self.load_config.load_format == LoadFormat.NPCACHE:
            # Currently np_cache only support *.bin checkpoints
            assert use_safetensors is False
            weights_iterator = np_cache_weights_iterator(
                source.model_or_path,
                self.load_config.download_dir,
                hf_folder,
                hf_weights_files,
            )
        elif use_safetensors:
            server_args = get_server_args()
            weight_loader_disable_mmap = server_args.weight_loader_disable_mmap
            weight_loader_prefetch = server_args.weight_loader_prefetch_checkpoints
            prefetch_num_threads = server_args.weight_loader_prefetch_num_threads
            weight_loader_drop_cache_after_load = (
                server_args.weight_loader_drop_cache_after_load
            )

            # Prefetch and multi-threaded loading both read the same shards,
            # competing for I/O on shared/network storage. When prefetch is
            # active (mmap path, not FASTSAFETENSORS) and the user didn't
            # explicitly request multi-threaded loading, fall back to the
            # single-threaded loader and let prefetch feed the page cache.
            # Setting enable_multithread_load or num_threads in
            # --model-loader-extra-config opts out (the latter is consumed
            # only by the multi-threaded iterator, so it signals intent);
            # e.g. local NVMe, where prefetch is a no-op and multi-threading
            # helps.
            if (
                weight_loader_prefetch
                and not weight_loader_disable_mmap
                and self.load_config.load_format != LoadFormat.FASTSAFETENSORS
                and use_multithread
                and not (
                    {"enable_multithread_load", "num_threads"} & extra_config.keys()
                )
            ):
                logger.warning(
                    "--weight-loader-prefetch-checkpoints is enabled; falling "
                    "back to single-threaded weight loading to avoid I/O "
                    "oversubscription with the prefetch threads. Set "
                    "enable_multithread_load=true in --model-loader-extra-config "
                    "to keep multi-threaded loading."
                )
                use_multithread = False

            if self.load_config.load_format == LoadFormat.FASTSAFETENSORS:
                weights_iterator = fastsafetensors_weights_iterator(
                    hf_weights_files,
                )
            elif use_multithread:
                weights_iterator = buffered_multi_thread_safetensors_weights_iterator(
                    hf_weights_files,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                    disable_mmap=weight_loader_disable_mmap,
                    prefetch=weight_loader_prefetch,
                    prefetch_num_threads=prefetch_num_threads,
                    drop_cache_after_load=weight_loader_drop_cache_after_load,
                )
            else:
                weights_iterator = safetensors_weights_iterator(
                    hf_weights_files,
                    disable_mmap=weight_loader_disable_mmap,
                    prefetch=weight_loader_prefetch,
                    prefetch_num_threads=prefetch_num_threads,
                    drop_cache_after_load=weight_loader_drop_cache_after_load,
                )

        else:
            if use_multithread:
                weights_iterator = multi_thread_pt_weights_iterator(
                    hf_weights_files,
                    max_workers=extra_config.get(
                        "num_threads", self.DEFAULT_NUM_THREADS
                    ),
                )
            else:
                weights_iterator = pt_weights_iterator(hf_weights_files)

        if self.load_config.draft_model_idx is not None:
            return self._filter_mtp_weights(
                weights_iterator, source.prefix, self.load_config.draft_model_idx
            )

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    @classmethod
    def _filter_mtp_weights(
        cls, weights_iterator, prefix: str, draft_model_idx: int
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Filter MTP weights to keep only the specified draft model layer
        and remap it to layer 0. Yields lazily so the upstream buffered
        iterator's sliding window actually bounds CPU memory — eager
        materialization caused page-reclaim hangs on large MoE checkpoints
        with multi-layer EAGLE."""
        for name, tensor in weights_iterator:
            match = cls._MTP_PATTERN.match(name)
            if match is not None:
                idx = int(match.group(1))
                if idx != draft_model_idx:
                    continue
                new_name = name.replace(match.group(), "model.mtp.layers.0.")
            else:
                new_name = name
            yield (prefix + new_name, tensor)

    def _get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:

        primary_weights = DefaultModelLoader.Source.init_new(model_config, model)
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[DefaultModelLoader.Source], getattr(model, "secondary_weights", ())
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(
            model_config.model_path, model_config.revision, fall_back_to_pt=True
        )

    def _load_modelopt_base_model(self, model_config: ModelConfig) -> nn.Module:
        """Load and prepare the base model for ModelOpt quantization.

        This method handles the common model loading logic shared between
        DefaultModelLoader (conditional) and ModelOptModelLoader (dedicated).
        """
        if not HAS_ACCELERATE:
            raise ImportError(
                "accelerate is required for ModelOpt quantization. "
                "Please install it with: pip install accelerate"
            )

        try:
            hf_config = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=True,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        except (KeyError, ValueError):
            from sglang.srt.utils.hf_transformers_utils import get_config

            hf_config = get_config(
                model_config.model_path,
                trust_remote_code=True,
            )
        with init_empty_weights():
            torch_dtype = getattr(hf_config, "torch_dtype", torch.float16)
            model = AutoModelForCausalLM.from_config(
                hf_config, torch_dtype=torch_dtype, trust_remote_code=True
            )
        max_memory = get_max_memory()
        inferred_device_map = infer_auto_device_map(model, max_memory=max_memory)

        on_cpu = "cpu" in inferred_device_map.values()
        model_kwargs = {"torch_dtype": "auto"}
        device_map = "auto"

        if on_cpu:
            for device in max_memory.keys():
                if isinstance(device, int):
                    max_memory[device] *= DEFAULT_GPU_MEMORY_FRACTION_FOR_CALIBRATION

            logger.warning(
                "Model does not fit to the GPU mem. "
                f"We apply the following memory limit for calibration: \n{max_memory}\n"
                f"If you hit GPU OOM issue, please adjust the memory fraction "
                f"(currently {DEFAULT_GPU_MEMORY_FRACTION_FOR_CALIBRATION}) or "
                "reduce the calibration `batch_size` manually."
            )
            model_kwargs["max_memory"] = max_memory

        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_path,
            config=hf_config,
            device_map=device_map,
            **model_kwargs,
            trust_remote_code=True,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
        )
        # Handle both legacy modelopt_quant and unified quantization flags
        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Legacy approach
            quant_choice_str = model_config.modelopt_quant
            rank0_log(f"ModelOpt quantization requested (legacy): {quant_choice_str}")
        else:
            # Unified approach - extract quantization type
            quant_choice_str = model_config._get_modelopt_quant_type()
            rank0_log(
                f"ModelOpt quantization requested (unified): {model_config.quantization} -> {quant_choice_str}"
            )

        if not isinstance(quant_choice_str, str):
            raise TypeError(
                f"Quantization type must be a string (e.g., 'fp8'), "
                f"got {type(quant_choice_str)}"
            )

        return model

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Load base model using shared method
            model = self._load_modelopt_base_model(model_config)
            # Note: DefaultModelLoader doesn't do additional quantization processing
            # For full ModelOpt quantization, use ModelOptModelLoader
            return model.eval()

        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            self.load_weights_and_postprocess(
                model, self._get_all_weights(model_config, model), target_device
            )

        self.counter_after_loading_weights = time.perf_counter()
        return model.eval()

    @staticmethod
    def load_weights_and_postprocess(model, weights, target_device):
        # Used in tests to verify memory savings when using online quantization.
        if is_cuda_alike():
            peak_memory = torch.cuda.max_memory_allocated()
            logger.debug(
                "Peak GPU memory before loading weights: %s GiB",
                f"{peak_memory / GIB_BYTES:.3f}",
            )
            memory_start = get_available_gpu_memory(
                target_device.type, gpu_id=torch.cuda.current_device()
            )

        quant_config = getattr(model, "quant_config", None)
        is_nvfp4_online = getattr(quant_config, "is_nvfp4_online", False)

        if is_nvfp4_online:
            # Scope exact FP4 quantization math to load-time conversion only;
            # restore the original environment before serving starts.
            with temp_set_env(FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH="1"):
                model.load_weights(weights)
            if target_device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        else:
            model.load_weights(weights)

        # Used in tests to verify memory savings when using online quantization.
        if is_cuda_alike():
            memory_end = get_available_gpu_memory(
                target_device.type, gpu_id=torch.cuda.current_device()
            )
            logger.debug(
                "Memory increase during load_weights: %s GiB",
                f"{memory_start - memory_end:.3f}",
            )

        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)


class LayeredModelLoader(DefaultModelLoader):
    """Model loader that loads weights layer by layer so that one can quantize a
    layer before loading another to make the peak memory envelope smaller."""

    def __init__(self, load_config: LoadConfig):
        # Back to the default load format
        load_config.load_format = LoadFormat.AUTO
        super().__init__(load_config)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
        from sglang.srt.runtime_context import get_server_args

        torchao_config = get_server_args().torchao_config
        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            # Create model on meta device
            with torch.device("meta"):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            # Check model's layered load support
            if not hasattr(model, "load_weights_to_module"):
                raise ValueError(
                    "LayeredModelLoader requires the model to have a "
                    "`load_weights_to_module` method. "
                    f"{model_config.model_path} does not support it."
                )

            # Get all weights from disk
            weights = self._get_all_weights(model_config, model)

            # Helper function to recursively fill the weights of a module
            def fill_module(module, fqn: List[str], weights):
                """
                fqn: list of strings representing the fully qualified name of `module`.
                """
                # Layer by layer
                for name, submod in module.named_children():
                    fill_module(submod, fqn + [name], weights)

                # First materialize on target device
                module.to_empty(device=target_device, recurse=False)
                fqn_path = ".".join(fqn)
                # Fill weights
                model.load_weights_to_module(
                    fqn_path,
                    weights,
                )
                # Quantize weights if applicable
                if torchao_config and "proj" in fqn_path:
                    # Note: `None` here is needed to indicate no filter, see
                    # `apply_torchao_config_to_model` for details.
                    apply_torchao_config_to_model(module, torchao_config, None)

            # Start calling on root module
            fill_module(model, [], weights)

        if torchao_config:
            model.torchao_applied = True

        return model.eval()


class QuantizedRLModelLoader(DefaultModelLoader):
    """
    Model loader for RL training with FP8 quantization (profile-free, native SGLang).

    Workflow:
      1. Initial load: Load base model → Record state → Apply FP8 quantization
      2. Training Actor in full precision
      3. Reload: Trainer sends full precision weights → Quantize to FP8 → Copy to original memory
      4. Use torch.as_strided to preserve memory locations across reloads

    Usage:
      --model-path Qwen/Qwen2.5-7B --quantization fp8 --load-format flash_rl
    """

    # Parameter attributes to record for weight reloading
    RECORDED_LOADER_KEYS = [
        "weight_loader",
        "load_qkv_weight",
        "load_column_parallel_weight",
        "load_row_parallel_weight",
        "load_merged_column_weight",
        "output_dim",
        "input_dim",
        "_assert_and_load",
    ]

    # Parameters to skip during FP8 quantization (matches FlashRL's exclude_list)
    SKIP_QUANTIZATION_PARAMS = [
        "weight_scale",
        "input_scale",
        "output_scale",
        ".bias",
        "lm_head.weight",
        "model.norm.weight",
        "embed_tokens",  # BF16 params
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
        "projector",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",  # LayerNorms
    ]

    # Stacked parameters (Qwen2): shards loaded separately, then combined
    STACKED_PARAMS_MAPPING = [
        ("qkv_proj", ["q_proj", "k_proj", "v_proj"]),
        ("gate_up_proj", ["gate_proj", "up_proj"]),
    ]
    _QKV_SHARD_ALIASES = {
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
    }

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        logger.info("[QuantizedRL] Profile-free FP8 quantization enabled")
        self._initial_load_complete = False

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ):
        """Standard weight preparation using base model path."""
        logger.info(f"[QuantizedRL] Loading from base model: {model_name_or_path}")
        temp_config = LoadConfig(load_format=LoadFormat.AUTO)
        temp_loader = DefaultModelLoader(temp_config)
        return temp_loader._prepare_weights(
            model_name_or_path, revision, fall_back_to_pt
        )

    @staticmethod
    def _bind_method_to_cls(func, obj):
        """Bind function to object instance (for weight_loader methods)."""
        import types

        if hasattr(func, "__self__") or not callable(func):
            return func
        return types.MethodType(func, obj)

    def load_weights_and_postprocess(self, model, weights, target_device):
        """
        Initial load: Load BF16 → Record state → Apply FP8 quantization.
        Called ONCE during model initialization.
        """
        logger.info("[QuantizedRL] Initial load with FP8 quantization")

        original_load_weights = model.load_weights

        def load_weights_proxy(weights):
            if QuantizedRLModelLoader.is_reload_scenario(model):
                logger.info("[QuantizedRL] Using fast path reload in load_weights")
                QuantizedRLModelLoader.rebinding_and_load_weights(
                    model, original_load_weights, weights
                )
            else:
                original_load_weights(weights)

        model.load_weights = load_weights_proxy

        model.load_weights(weights)
        original_weights = dict(model.named_parameters())

        # Record pre-quantization state (shape/stride) for torch.as_strided reset

        model.original_weights_rebuild_keys = {}
        for name, p in original_weights.items():
            model.original_weights_rebuild_keys[name] = {
                "shape": p.shape,
                "stride": p.stride(),
                "dtype": p.dtype,
                "nbytes": p.untyped_storage().nbytes(),
            }

        # Record parameter attributes (weight_loader, etc.) before quantization
        recorded_loader = {
            k: dict() for k in QuantizedRLModelLoader.RECORDED_LOADER_KEYS
        }
        for name, p in original_weights.items():
            for key in QuantizedRLModelLoader.RECORDED_LOADER_KEYS:
                if hasattr(p, key):
                    attr = getattr(p, key)
                    if not callable(attr):
                        recorded_loader[key][name] = attr
                    elif hasattr(attr, "__self__") and p is attr.__self__:
                        recorded_loader[key][name] = attr.__func__  # Store unbound
                    else:
                        recorded_loader[key][name] = attr
        model.recorded_loader = recorded_loader

        # Apply FP8 quantization (creates new Parameters, loses attributes)
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        model.flash_rl_initial_load_complete = True
        self._initial_load_complete = True
        logger.info("[QuantizedRL] Initial load complete")

    @staticmethod
    def is_reload_scenario(model):
        """Check if model is ready for reloading (initial load completed)."""
        return (
            hasattr(model, "original_weights_rebuild_keys")
            and hasattr(model, "recorded_loader")
            and getattr(model, "flash_rl_initial_load_complete", False)
        )

    @staticmethod
    def _is_stacked_param(name):
        """Check if parameter is stacked (qkv_proj, gate_up_proj)."""
        for stacked_name, _ in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING:
            if stacked_name in name:
                return True
        return False

    @staticmethod
    def _resolve_stacked_info(name: str) -> Tuple[str, Optional[str], Optional[Any]]:
        for target, shard_names in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING:
            for idx, shard in enumerate(shard_names):
                if shard in name:
                    shard_id = (
                        QuantizedRLModelLoader._QKV_SHARD_ALIASES.get(shard, shard)
                        if target == "qkv_proj"
                        else idx
                    )
                    return name.replace(shard, target), target, shard_id
        return name, None, None

    @staticmethod
    def _store_quantized_scale(
        scale_store: Dict[str, Union[torch.Tensor, Dict[Any, torch.Tensor]]],
        name: str,
        scale: torch.Tensor,
    ) -> None:
        param_name, stacked_key, shard_id = (
            QuantizedRLModelLoader._resolve_stacked_info(name)
        )
        if stacked_key is None:
            scale_store[param_name] = scale
        else:
            shard_dict = scale_store.setdefault(param_name, {})
            assert isinstance(shard_dict, dict)
            shard_dict[shard_id] = scale

    @staticmethod
    def _apply_scale_update(
        all_params: Dict[str, torch.nn.Parameter],
        param_name: str,
        scale_info: Union[torch.Tensor, Dict[Any, torch.Tensor], None],
    ) -> None:
        if scale_info is None:
            return
        # Get tp rank and size
        tp_rank = get_parallel().tp_rank
        tp_size = get_parallel().tp_size

        def _get_tp_sharded_scale(full_scale_tensor):
            """Get tp sharded scale from full scale tensor"""
            if tp_size == 1:
                return full_scale_tensor

            full_dim = full_scale_tensor.shape[0]
            shard_dim = full_dim // tp_size
            start_idx = tp_rank * shard_dim
            end_idx = start_idx + shard_dim
            return full_scale_tensor[start_idx:end_idx]

        if param_name.endswith(".weight"):
            scale_param_name = f"{param_name[:-7]}.weight_scale"
        else:
            scale_param_name = f"{param_name}.weight_scale"

        scale_param = all_params.get(scale_param_name)
        if scale_param is None:
            logger.warning(
                "[QuantizedRL] Scale parameter not found: %s", scale_param_name
            )
            return
        if isinstance(scale_info, torch.Tensor):
            new_scale = scale_info.t().contiguous()
            if scale_param.data.shape == new_scale.shape:
                scale_param.data.copy_(new_scale)
            else:
                logger.warning(
                    "[QuantizedRL] Scale shape mismatch for %s: expected %s, got %s",
                    scale_param_name,
                    scale_param.data.shape,
                    new_scale.shape,
                )
        else:
            stacked_key = next(
                (
                    target
                    for target, _ in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING
                    if target in param_name
                ),
                None,
            )
            shard_names = next(
                (
                    names
                    for target, names in QuantizedRLModelLoader.STACKED_PARAMS_MAPPING
                    if target == stacked_key
                ),
                [],
            )
            rows_per_shard = scale_param.data.shape[-1] // max(len(shard_names), 1)
            if rows_per_shard * len(shard_names) != scale_param.data.shape[-1]:
                logger.warning(
                    f"Scale param shape {scale_param.data.shape[-1]} not divisible by {len(shard_names)}"
                )
            offset = 0
            for idx, shard in enumerate(shard_names):
                shard_id = (
                    QuantizedRLModelLoader._QKV_SHARD_ALIASES.get(shard, shard)
                    if stacked_key == "qkv_proj"
                    else idx
                )
                shard_scale = scale_info.get(shard_id)
                shard_scale = _get_tp_sharded_scale(shard_scale)
                if shard_scale is None:
                    offset += rows_per_shard
                    continue
                shard_rows = shard_scale.shape[0]
                start = offset
                end = start + shard_rows
                scale_param.data[..., start:end] = shard_scale.t().contiguous()
                offset = end

    @staticmethod
    def rebinding_and_load_weights(model, first_time_load_weights, weights):
        """
        Reload: VERL sends BF16 → Quantize to FP8 → Copy to original memory.

        Flow: Reset params → Restore attributes → Quantize in iterator → Load → Copy back
        """
        logger.info("[QuantizedRL] Reload: Updating weights with FP8 quantization")

        weights_list = list(weights)
        updated_param_names, is_last_update = (
            QuantizedRLModelLoader._get_updated_params(weights_list, model)
        )

        # Save current FP8 parameter data pointers
        existing_params = dict(model.named_parameters())
        current_param_data = {}
        for name in updated_param_names:
            if name in existing_params:
                current_param_data[name] = existing_params[name].data

        # Reset to pre-quantization shape using torch.as_strided
        # Keeps same storage, just changes view - critical for memory preservation
        for name, rebuild_info in model.original_weights_rebuild_keys.items():
            if name in updated_param_names and name in existing_params:
                existing_params[name].data = torch.as_strided(
                    # Note: avoid clone here
                    existing_params[name].data.clone(),
                    rebuild_info["shape"],
                    rebuild_info["stride"],
                )

        # Restore weight loader attributes (only if missing)
        for k, loader_dict in model.recorded_loader.items():
            for param_name, loader in loader_dict.items():
                if param_name in updated_param_names and param_name in existing_params:
                    param = existing_params[param_name]
                    if not hasattr(param, k):
                        if callable(loader):
                            if hasattr(loader, "__self__"):
                                setattr(param, k, loader)
                            else:
                                setattr(
                                    param,
                                    k,
                                    QuantizedRLModelLoader._bind_method_to_cls(
                                        loader, param
                                    ),
                                )
                        else:
                            setattr(param, k, loader)

        del existing_params

        # Quantize BF16 weights to FP8 in iterator (before weight_loader)
        # Store scales for later update
        quantized_scales: Dict[str, Union[torch.Tensor, Dict[Any, torch.Tensor]]] = {}

        def quantize_weights_iterator(weights_iter):
            """Quantize individual shards before weight_loader stacks them."""
            from sglang.kernels.ops.quantization.fp8_kernel import (
                per_token_group_quant_fp8,
            )

            for name, weight in weights_iter:
                if any(
                    skip in name
                    for skip in QuantizedRLModelLoader.SKIP_QUANTIZATION_PARAMS
                ):
                    logger.info(f"[QuantizedRL] Skip: {name} ({weight.dtype})")
                    yield (name, weight)
                elif weight.dtype in [torch.bfloat16, torch.float32, torch.float16]:
                    qweight, scale = per_token_group_quant_fp8(weight, weight.shape[-1])
                    logger.info(f"[QuantizedRL] Quantize: {name} {weight.dtype}→FP8")
                    QuantizedRLModelLoader._store_quantized_scale(
                        quantized_scales, name, scale
                    )
                    yield (name, qweight)
                else:
                    logger.info(f"[QuantizedRL] Keep: {name} ({weight.dtype})")
                    yield (name, weight)

        # Load quantized weights (weight_loader stacks FP8 shards)
        first_time_load_weights(quantize_weights_iterator(iter(weights_list)))

        # Copy back to original FP8 memory locations and update scales
        all_params = dict(model.named_parameters())

        for name in updated_param_names:
            if name not in all_params or name not in current_param_data:
                continue
            if any(
                skip in name for skip in QuantizedRLModelLoader.SKIP_QUANTIZATION_PARAMS
            ):
                continue

            new_param = all_params[name]
            old_fp8_data = current_param_data[name]

            # Handle embeddings/lm_head (BF16) and quantized weights (FP8)
            if "embed_tokens" in name or "lm_head" in name:
                old_fp8_data.copy_(new_param.data)
                new_param.data = old_fp8_data
            elif (
                new_param.dtype == torch.float8_e4m3fn
                and old_fp8_data.dtype == torch.float8_e4m3fn
            ):
                # FP8: Use strided view for transposed storage
                strided_data = torch.as_strided(
                    new_param.data, old_fp8_data.shape, old_fp8_data.stride()
                )
                old_fp8_data.copy_(strided_data)
                new_param.data = old_fp8_data
                QuantizedRLModelLoader._apply_scale_update(
                    all_params,
                    name,
                    quantized_scales.get(name),
                )
            elif new_param.dtype == old_fp8_data.dtype:
                # Same dtype (LayerNorm, etc.): Direct copy
                old_fp8_data.copy_(new_param.data)
                new_param.data = old_fp8_data
            else:
                raise RuntimeError(
                    f"Unexpected dtype mismatch for {name}: "
                    f"new={new_param.dtype}, old={old_fp8_data.dtype}"
                )

        # Cleanup
        del current_param_data
        if is_last_update:
            gc.collect()
            current_platform.empty_cache()

        logger.info("[QuantizedRL] Reload complete")
        return updated_param_names, is_last_update

    @staticmethod
    def _get_updated_params(weights_list, model):
        """Identify which parameters need updating from incoming weights."""
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(model.named_parameters())
        updated_params = set()
        is_last_update = False

        for name, _ in weights_list:
            if name == "lm_head.weight":
                is_last_update = True

            if any(
                skip in name for skip in QuantizedRLModelLoader.SKIP_QUANTIZATION_PARAMS
            ):
                continue

            from sglang.srt.layers.utils import get_layer_id

            # Skip params outside layer range (for pipeline parallelism)
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(model, "start_layer")
                and (layer_id < model.start_layer or layer_id >= model.end_layer)
            ):
                continue

            # Skip tied embeddings and vision tower params
            if (
                hasattr(model, "config")
                and model.config.tie_word_embeddings
                and "lm_head.weight" in name
            ):
                continue
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue

            # Map stacked param shards (q/k/v_proj → qkv_proj)
            mapped = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name in name:
                    name = name.replace(weight_name, param_name)
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    updated_params.add(name)
                    mapped = True
                    break

            if not mapped:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict:
                    updated_params.add(name)

        return list(updated_params), is_last_update


class DummyModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        if get_bool_env_var("SGL_CPU_QUANTIZATION"):
            return load_model_with_cpu_quantization(
                self, model_config=model_config, device_config=device_config
            )

        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            # NOTE(woosuk): For accurate performance evaluation, we assign
            # random values to the weights.
            initialize_dummy_weights(model)

            _post_load_weights(model)

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # Skip FusedMoE layers already quantized during init (FP8 or FP4)
                    if (
                        hasattr(module, "is_weights_quantized")
                        and module.is_weights_quantized()
                    ):
                        continue
                    quant_method.process_weights_after_loading(module)

        return model.eval()


class ShardedStateLoader(BaseModelLoader):
    """
    Model loader that directly loads each worker's model state dict, which
    enables a fast load path for large tensor-parallel models where each worker
    only needs to read its own shard rather than the entire checkpoint. See
    `examples/runtime/engine/save_sharded_state.py` for creating a sharded checkpoint.
    """

    DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = (
            {}
            if load_config.model_loader_extra_config is None
            else load_config.model_loader_extra_config.copy()
        )
        self.pattern = extra_config.pop("pattern", self.DEFAULT_PATTERN)
        if extra_config:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{load_config.model_loader_extra_config.keys()}"
            )

    @staticmethod
    def _filter_subtensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter out all tensors that share the same memory or a subset of the
        memory of another tensor.
        """
        same_storage_groups: Dict[Any, List[Tuple[str, torch.Tensor]]] = (
            collections.defaultdict(list)
        )
        for key, tensor in tensors.items():
            if tensor.numel():
                ptr = tensor.untyped_storage().data_ptr()
                same_storage_groups[tensor.device, ptr].append((key, tensor))

        def get_end_ptr(tensor: torch.Tensor) -> int:
            return tensor.view(-1)[-1].data_ptr() + tensor.element_size()

        result: Dict[str, torch.Tensor] = {}
        for group in same_storage_groups.values():
            for k, t in group:
                if not t.is_contiguous():
                    # End-pointer dedup assumes a flat view; non-contiguous
                    # tensors (e.g. produced by
                    # ``.transpose(...).contiguous().transpose(...)`` in some
                    # quant ``post_load_weights`` paths) cannot be flattened
                    # via ``view(-1)``. Include them directly; downstream
                    # writers call ``.contiguous()`` before save.
                    result[k] = t
                    continue
                a, b = t.data_ptr(), get_end_ptr(t)
                for k2, t2 in group:
                    if not t2.is_contiguous():
                        continue
                    a2, b2 = t2.data_ptr(), get_end_ptr(t2)
                    if a < a2 or b2 < b:
                        continue
                    if a2 < a or b < b2 or not t.is_contiguous():
                        break  # t2 covers strictly more memory than t.
                    if k2 < k:
                        # Same tensors, keep the one with the smaller key.
                        break
                else:
                    result[k] = t
        return result

    def _prepare_weights(self, model_name_or_path: str, revision: Optional[str]):
        if os.path.isdir(model_name_or_path):
            return model_name_or_path
        else:
            allow_patterns = ["*.safetensors"]
            return download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                allow_patterns,
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path, model_config.revision)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        from safetensors.torch import safe_open

        local_model_path = self._prepare_weights(
            model_config.model_path, model_config.revision
        )

        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, quant_config)
                for _, module in model.named_modules():
                    quant_method = getattr(module, "quant_method", None)
                    if quant_method is not None:
                        quant_method.process_weights_after_loading(module)
            rank = get_parallel().tp_rank
            pattern = os.path.join(
                local_model_path,
                self.pattern.format(rank=rank, part="*"),
            )
            filepaths = glob.glob(pattern)
            if not filepaths:
                # TODO: support un-sharded checkpoints too
                raise ValueError(
                    f"Could not find checkpoint files '{pattern}', only "
                    f"pre-sharded checkpoints are currently supported!"
                )
            state_dict = self._filter_subtensors(model.state_dict())
            for path in filepaths:
                with safe_open(path, framework="pt") as f:
                    for key in f.keys():  # noqa: SIM118
                        tensor = f.get_tensor(key)
                        # If loading with LoRA enabled, additional padding may
                        # be added to certain parameters. We only load into a
                        # narrowed view of the parameter data.
                        param_data = state_dict[key].data
                        param_shape = state_dict[key].shape
                        for dim, size in enumerate(tensor.shape):
                            if size < param_shape[dim]:
                                param_data = param_data.narrow(dim, 0, size)
                        if tensor.shape != param_shape:
                            logger.warning(
                                "loading tensor of shape %s into "
                                "parameter '%s' of shape %s",
                                tensor.shape,
                                key,
                                param_shape,
                            )
                        param_data.copy_(tensor)
                        state_dict.pop(key)
            if state_dict:
                raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

            _post_load_weights(model)

        return model.eval()

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from safetensors.torch import save_file

        if pattern is None:
            pattern = ShardedStateLoader.DEFAULT_PATTERN
        rank = get_parallel().tp_rank
        part_idx = 0
        total_size = 0
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        state_dict_part: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            param_size = tensor.nelement() * tensor.element_size()
            if max_size is not None and total_size + param_size > max_size:
                filename = pattern.format(rank=rank, part=part_idx)
                save_file(
                    state_dict_part,
                    os.path.join(path, filename),
                )
                part_idx += 1
                total_size = 0
                state_dict_part = {}
            state_dict_part[key] = tensor
            total_size += param_size
        if len(state_dict_part) > 0:
            filename = pattern.format(rank=rank, part=part_idx)
            save_file(
                state_dict_part,
                os.path.join(path, filename),
            )


class PreshardedModelLoader(DefaultModelLoader):
    """Dump/reload post-process weights under ``<model_path>/presharded/<subdir>/``.

    Optional roots in ``model_loader_extra_config`` (subdir still appended):
    ``presharded_path`` (target), ``draft_presharded_path`` (speculative draft).
    Dump dir must be shared across ranks/nodes.
    """

    DEFAULT_SUBDIR = "presharded"
    MAX_FILE_BYTES = 20 * (1024**3)
    CHECKSUM_FILENAME = "checksum.json"
    READY_FILENAME = "READY"
    TMP_SUBDIR = "_tmp_presharding"
    PLAN_VERSION = 1
    DEFAULT_HASH_NUM_THREADS = 8
    _CONTENT_HASH_HEX_LEN = 32

    def __init__(self, load_config: LoadConfig):
        extra = (
            {}
            if load_config.model_loader_extra_config is None
            else dict(load_config.model_loader_extra_config)
        )
        self._presharded_path_override = extra.pop("presharded_path", None)
        self._draft_presharded_path_override = extra.pop("draft_presharded_path", None)
        self._max_file_bytes = int(extra.pop("max_file_bytes", self.MAX_FILE_BYTES))
        self._hash_num_threads = int(
            extra.pop("hash_num_threads", self.DEFAULT_HASH_NUM_THREADS)
        )
        self._verify_on_load = bool(extra.pop("verify_on_load", False))
        load_config.model_loader_extra_config = extra
        load_config.load_format = LoadFormat.AUTO
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        presharded_dir = self._presharded_dir(model_config)
        if not self._presharded_ready(presharded_dir):
            super().download_model(model_config)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        shard_config = self._collect_shard_config(model_config)
        presharded_dir = self._presharded_dir(model_config, shard_config)
        if self._presharded_ready(presharded_dir) and self._shard_config_matches(
            presharded_dir, shard_config
        ):
            logger.info("Loading from presharded checkpoint at %s", presharded_dir)
            return self._load_from_presharded(
                model_config, device_config, presharded_dir
            )
        logger.info(
            "No presharded checkpoint at %s; doing first-time load and dump.",
            presharded_dir,
        )
        return self._first_time_load_and_dump(
            model_config, device_config, presharded_dir, shard_config
        )

    @classmethod
    def _presharded_ready(cls, presharded_dir: str) -> bool:
        return os.path.isfile(os.path.join(presharded_dir, cls.READY_FILENAME))

    def _presharded_dir(
        self,
        model_config: ModelConfig,
        shard_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        if shard_config is None:
            shard_config = self._collect_shard_config(model_config)
        subfolder = self._build_subfolder_name(shard_config)
        if model_config.is_draft_model:
            root = self._draft_presharded_path_override
        else:
            root = self._presharded_path_override
        if root is None:
            root = os.path.join(model_config.model_path, self.DEFAULT_SUBDIR)
        return os.path.join(root, subfolder)

    def _collect_shard_config(self, model_config: ModelConfig) -> Dict[str, Any]:
        def _safe(fn) -> int:
            try:
                return fn()
            except (AssertionError, AttributeError, RuntimeError):
                return 1

        parallel = get_parallel()
        server_args = get_server_args()
        return {
            "tp": _safe(lambda: parallel.tp_size),
            "dp": _safe(lambda: parallel.moe_dp_size),
            "ep": _safe(lambda: parallel.moe_ep_size),
            "pp": _safe(lambda: parallel.pp_size),
            "moe_dense_tp_size": server_args.moe_dense_tp_size,
            "moe_dp_size": server_args.moe_dp_size,
            "enable_dp_lm_head": server_args.enable_dp_lm_head,
            "enable_fp32_lm_head": server_args.enable_fp32_lm_head,
            "quantization": model_config.quantization,
            "model_dtype": str(model_config.dtype),
            "ep_num_redundant_experts": server_args.ep_num_redundant_experts,
            "enable_eplb": server_args.enable_eplb,
            "init_expert_location": self._normalize_init_expert_location(
                server_args.init_expert_location
            ),
            "structural_signature": self._compute_structural_signature(model_config),
        }

    @staticmethod
    def _normalize_init_expert_location(value: Optional[str]) -> Optional[str]:
        if value is None or value == "trivial":
            return value
        if value.endswith((".json", ".pt")) and os.path.isfile(value):
            h = hashlib.sha1()
            with open(value, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            return f"file:{os.path.basename(value)}:sha1:{h.hexdigest()[:16]}"
        return value

    def _build_subfolder_name(self, shard_config: Dict[str, Any]) -> str:
        combined = hashlib.sha1(
            json.dumps(shard_config, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"TP-{shard_config['tp']}-sig-{combined}"

    def _shard_config_matches(
        self, presharded_dir: str, shard_config: Dict[str, Any]
    ) -> bool:
        try:
            with open(os.path.join(presharded_dir, self.CHECKSUM_FILENAME)) as f:
                stored = json.load(f).get("shard_config")
        except (OSError, ValueError):
            stored = None
        current = json.loads(json.dumps(shard_config))
        if stored == current:
            return True
        logger.warning(
            "Presharded checkpoint at %s was dumped with a different shard "
            "config than the current launch (stored=%s, current=%s). "
            "Treating as a cache miss and re-dumping.",
            presharded_dir,
            stored,
            current,
        )
        return False

    def _compute_structural_signature(self, model_config: ModelConfig) -> Optional[str]:
        local_sig = self._compute_local_structural_signature(model_config)
        return self._make_rank_invariant_structural_signature(local_sig)

    def _compute_local_structural_signature(
        self, model_config: ModelConfig
    ) -> Optional[str]:
        from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT

        def _clear_meta_rope_cache() -> None:
            meta_keys = [
                k
                for k, v in _ROPE_DICT.items()
                if any(p.device.type == "meta" for p in v.parameters())
                or any(b.device.type == "meta" for b in v.buffers())
            ]
            for k in meta_keys:
                del _ROPE_DICT[k]

        try:
            quant_config = _get_quantization_config(model_config, self.load_config)
            with set_default_torch_dtype(model_config.dtype):
                with torch.device("meta"):
                    meta_model = _initialize_model(
                        model_config, self.load_config, quant_config
                    )
                state_dict = meta_model.state_dict()
                sig_input = sorted(
                    (name, tuple(t.shape), str(t.dtype))
                    for name, t in state_dict.items()
                )
            del meta_model
            return self._hash_structural_signature(sig_input)
        except Exception as e:
            logger.warning(
                "Failed to build structural signature for presharded cache key "
                "(model_type=%s): %s",
                getattr(
                    getattr(model_config, "hf_config", None), "model_type", "unknown"
                ),
                e,
            )
            return None
        finally:
            _clear_meta_rope_cache()

    @classmethod
    def _make_rank_invariant_structural_signature(
        cls, local_sig: Optional[str]
    ) -> Optional[str]:
        try:
            from sglang.srt.distributed import get_world_group

            group = get_world_group()
            if group.world_size <= 1:
                return local_sig
            all_sigs = group.all_gather_object(local_sig)
        except (AssertionError, AttributeError, RuntimeError):
            return local_sig

        if all(s is None for s in all_sigs):
            return None
        return hashlib.sha1(repr(all_sigs).encode()).hexdigest()[:16]

    @staticmethod
    def _hash_structural_signature(
        sig_input: List[Tuple[str, Tuple[int, ...], str]],
    ) -> str:
        h = hashlib.sha1(repr(sig_input).encode())
        return h.hexdigest()[:16]

    @staticmethod
    def _world_rank_and_size() -> Tuple[int, int]:
        from sglang.srt.distributed import get_world_group

        try:
            g = get_world_group()
            return g.rank_in_group, g.world_size
        except (AssertionError, AttributeError):
            return 0, 1

    @staticmethod
    def _world_barrier() -> None:
        from sglang.srt.distributed import get_world_group

        try:
            get_world_group().barrier()
        except (AssertionError, AttributeError):
            pass

    @staticmethod
    def _new_content_hasher():
        import xxhash

        return xxhash.xxh3_128()

    @staticmethod
    def _hash_tensor(tensor: torch.Tensor) -> str:
        # CPU copy so concurrent dump workers cannot race CUDA D2H hashing.
        t = tensor.detach()
        prefix = str(tuple(t.shape)).encode() + str(t.dtype).encode()
        h = PreshardedModelLoader._new_content_hasher()
        h.update(prefix)

        if t.numel() == 0:
            return h.hexdigest()

        cpu = t.contiguous().to(device="cpu", copy=True).contiguous()
        flat_u8 = cpu.reshape(-1).view(torch.uint8)
        h.update(memoryview(flat_u8.numpy()))
        return h.hexdigest()

    def _verify_rank_checksum(
        self,
        verify_hashes: List[Tuple[str, str]],
        plan: Dict[str, Any],
        rank: int,
        presharded_dir: str,
    ) -> None:
        expected = plan.get("rank_checksums", {}).get(str(rank))
        if expected is None:
            raise ValueError(
                f"Plan at {presharded_dir} has no rank_checksums entry for "
                f"rank {rank}; cannot verify. Set "
                f"--model-loader-extra-config '{{\"verify_on_load\": false}}' "
                f"to skip verification, or re-dump the checkpoint."
            )

        total = 0
        for name, content_hash in verify_hashes:
            d = PreshardedModelLoader._fold_name_content_digest(name, content_hash)
            total = (total + int.from_bytes(d[:8], "big")) & 0xFFFFFFFFFFFFFFFF
        actual = format(total, "016x")

        if actual != expected:
            raise ValueError(
                f"Rank-{rank} checksum mismatch for presharded checkpoint at "
                f"{presharded_dir}: expected {expected}, got {actual}. The "
                f"checkpoint files may be corrupted; re-dump or skip "
                f"verification with --model-loader-extra-config "
                f"'{{\"verify_on_load\": false}}'."
            )

    @staticmethod
    def _fold_name_content_digest(name: str, content_hash: str) -> bytes:
        h = PreshardedModelLoader._new_content_hasher()
        h.update((name + ":" + content_hash).encode("utf-8"))
        return h.digest()

    @staticmethod
    def _collect_extra_tensors(model: nn.Module) -> Dict[str, torch.Tensor]:
        seen: set = set()
        param_storages: set = set()
        for name, tensor in model.state_dict().items():
            seen.add(name)
            if tensor.numel() > 0:
                param_storages.add((tensor.device, tensor.untyped_storage().data_ptr()))
        extras: Dict[str, torch.Tensor] = {}
        for module_name, module in model.named_modules():
            prefix = f"{module_name}." if module_name else ""
            for attr_name in list(vars(module).keys()):
                if attr_name.startswith("_"):
                    continue
                try:
                    val = getattr(module, attr_name)
                except AttributeError:
                    continue
                if isinstance(val, torch.Tensor) and not isinstance(
                    val, torch.nn.Parameter
                ):
                    full_name = f"{prefix}{attr_name}"
                    if full_name in seen:
                        continue
                    if val.numel() > 0:
                        key = (val.device, val.untyped_storage().data_ptr())
                        if key in param_storages:
                            continue
                    extras[full_name] = val
        return extras

    @staticmethod
    def _rebind_parameter_aliases(model: nn.Module) -> None:
        for _, module in model.named_modules():
            gemma_w = getattr(module, "gemma_weight", None)
            weight = getattr(module, "weight", None)
            if (
                isinstance(gemma_w, torch.Tensor)
                and isinstance(weight, torch.nn.Parameter)
                and gemma_w.shape == weight.shape
            ):
                torch.add(weight.data, 1.0, out=gemma_w)

            attn = getattr(module, "attn", None)
            conv1d = getattr(module, "conv1d", None)
            if attn is None:
                continue
            if hasattr(module, "A_log") and hasattr(attn, "A_log"):
                attn.A_log = module.A_log
            if hasattr(module, "dt_bias") and hasattr(attn, "dt_bias"):
                attn.dt_bias = module.dt_bias
            if conv1d is None:
                continue
            cweight = getattr(conv1d, "weight", None)
            if cweight is not None and hasattr(attn, "conv_weights"):
                if cweight.dim() == 3 and cweight.size(1) == 1:
                    attn.conv_weights = cweight.view(cweight.size(0), cweight.size(2))
                else:
                    attn.conv_weights = (
                        cweight.squeeze() if cweight.dim() > 2 else cweight
                    )
            if hasattr(conv1d, "bias") and hasattr(attn, "bias"):
                attn.bias = conv1d.bias

    def _ensure_presharded_dir_writable(self, presharded_dir: str) -> None:
        rank, _ = self._world_rank_and_size()
        try:
            os.makedirs(presharded_dir, exist_ok=True)
            if rank == 0:
                probe = os.path.join(presharded_dir, ".presharded_write_probe")
                last_err: Optional[OSError] = None
                for _ in range(5):
                    try:
                        with open(probe, "w") as f:
                            f.write("ok")
                        os.unlink(probe)
                        last_err = None
                        break
                    except OSError as e:
                        last_err = e
                        os.makedirs(presharded_dir, exist_ok=True)
                        time.sleep(0.05)
                if last_err is not None:
                    raise last_err
        except OSError as e:
            raise RuntimeError(
                f"Presharded dump directory is not writable: {presharded_dir}. "
                "Set model_loader_extra_config "
                '\'{"presharded_path": "..."}\' (or draft_presharded_path for '
                "the draft model) to a writable shared filesystem path. "
                f"Original error: {e}"
            ) from e
        self._world_barrier()

    def _first_time_load_and_dump(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        presharded_dir: str,
        shard_config: Dict[str, Any],
    ) -> nn.Module:
        self._ensure_presharded_dir_writable(presharded_dir)
        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(model_config, self.load_config, quant_config)
            self.load_weights_and_postprocess(
                model,
                self._get_all_weights(model_config, model),
                target_device,
            )

            state_dict = dict(model.state_dict())
            extras = self._collect_extra_tensors(model)
            self._dump_state_to_disk(state_dict, extras, presharded_dir, shard_config)
            del state_dict
            del extras
            gc.collect()

        self.counter_after_loading_weights = time.perf_counter()
        return model.eval()

    def _dump_state_to_disk(
        self,
        state_dict: Dict[str, torch.Tensor],
        extras: Dict[str, torch.Tensor],
        presharded_dir: str,
        shard_config: Dict[str, Any],
    ) -> None:
        rank, world_size = self._world_rank_and_size()
        tmp_dir = os.path.join(presharded_dir, self.TMP_SUBDIR)
        if rank == 0:
            ready_path = os.path.join(presharded_dir, self.READY_FILENAME)
            if os.path.isfile(ready_path):
                os.unlink(ready_path)
            os.makedirs(tmp_dir, exist_ok=True)
        self._world_barrier()

        items: List[Tuple[str, torch.Tensor, bool]] = []
        items.extend((n, t, False) for n, t in state_dict.items())
        items.extend((n, t, True) for n, t in extras.items())

        def _entry(item: Tuple[str, torch.Tensor, bool]) -> Tuple[str, Dict[str, Any]]:
            name, tensor, is_extra = item
            return name, {
                "checksum": self._hash_tensor(tensor),
                "size": tensor.numel() * tensor.element_size(),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "is_extra": is_extra,
            }

        manifest: Dict[str, Dict[str, Any]] = {}
        num_workers = min(max(1, len(items)), self._hash_num_threads)
        if num_workers <= 1:
            for it in items:
                name, info = _entry(it)
                manifest[name] = info
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers,
                thread_name_prefix="presharded-hash",
            ) as ex:
                for name, info in ex.map(_entry, items):
                    manifest[name] = info

        with open(os.path.join(tmp_dir, f"manifest_{rank:05d}.json"), "w") as f:
            json.dump(manifest, f)
        self._world_barrier()

        if rank == 0:
            plan = self._build_dump_plan(world_size, tmp_dir, self._max_file_bytes)
            plan["shard_config"] = shard_config
            with open(os.path.join(presharded_dir, self.CHECKSUM_FILENAME), "w") as f:
                json.dump(plan, f, indent=2)
        self._world_barrier()

        with open(os.path.join(presharded_dir, self.CHECKSUM_FILENAME)) as f:
            plan = json.load(f)
        all_tensors = {**state_dict, **extras}
        self._dump_files_for_rank(all_tensors, plan, rank, presharded_dir)
        self._world_barrier()

        if rank == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            ready_path = os.path.join(presharded_dir, self.READY_FILENAME)
            with open(ready_path, "w") as f:
                json.dump(
                    {
                        "plan_version": self.PLAN_VERSION,
                        "world_size": world_size,
                        "created_at": time.time(),
                    },
                    f,
                )
        self._world_barrier()

    @staticmethod
    def _make_filename(
        file_id: int, rank_list: Tuple[int, ...], is_common: bool
    ) -> str:
        if is_common:
            return f"model-{file_id:05d}-common.safetensor"
        rank_str = ",".join(f"{r:03d}" for r in rank_list)
        return f"model-{file_id:05d}-rank-{rank_str}.safetensor"

    @classmethod
    def _build_dump_plan(
        cls, world_size: int, tmp_dir: str, max_file_bytes: int
    ) -> Dict[str, Any]:
        rank_to_manifest: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for r in range(world_size):
            manifest_path = os.path.join(tmp_dir, f"manifest_{r:05d}.json")
            try:
                with open(manifest_path) as f:
                    rank_to_manifest[r] = json.load(f)
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Rank {r} did not write {manifest_path}. The presharded "
                    "dump directory must be on a filesystem shared by all "
                    "ranks/nodes (set presharded_path / draft_presharded_path "
                    "to a shared path if model_path is node-local)."
                ) from e

        checksum_to_entries: Dict[str, List[Tuple[int, str, Dict[str, Any]]]] = (
            collections.defaultdict(list)
        )
        name_to_is_extra: Dict[Tuple[int, str], bool] = {}
        for r, manifest in rank_to_manifest.items():
            for name, info in manifest.items():
                checksum_to_entries[info["checksum"]].append((r, name, info))
                name_to_is_extra[(r, name)] = bool(info.get("is_extra", False))

        tensor_records: List[Dict[str, Any]] = []
        for checksum, entries in checksum_to_entries.items():
            sizes = {info["size"] for _, _, info in entries}
            if len(sizes) != 1:
                raise RuntimeError(
                    f"Checksum {checksum} maps to inconsistent sizes {sizes}; "
                    f"this indicates a hash collision or stale manifest."
                )
            size = next(iter(sizes))
            ranks = sorted({r for r, _, _ in entries})
            rank_to_names: Dict[str, List[str]] = collections.defaultdict(list)
            for r, n, _ in entries:
                rank_to_names[str(r)].append(n)
            tensor_records.append(
                {
                    "checksum": checksum,
                    "size": size,
                    "rank_list": ranks,
                    "rank_to_names": {k: sorted(v) for k, v in rank_to_names.items()},
                }
            )

        by_rank_list: Dict[Tuple[int, ...], List[Dict[str, Any]]] = (
            collections.defaultdict(list)
        )
        for rec in tensor_records:
            by_rank_list[tuple(rec["rank_list"])].append(rec)

        files: List[Dict[str, Any]] = []
        next_file_id = 0
        for rank_tuple, recs in by_rank_list.items():
            recs.sort(key=lambda r: -r["size"])
            is_common = len(rank_tuple) == world_size and rank_tuple == tuple(
                range(world_size)
            )
            writer_load = {wr: 0 for wr in rank_tuple}
            writer_records: Dict[int, List[Dict[str, Any]]] = {
                wr: [] for wr in rank_tuple
            }
            for rec in recs:
                wr = min(rank_tuple, key=lambda r: writer_load[r])
                writer_records[wr].append(rec)
                writer_load[wr] += rec["size"]

            for wr, wr_recs in writer_records.items():
                cur_size = 0
                cur_tensors: List[Dict[str, Any]] = []
                for rec in wr_recs:
                    if cur_tensors and cur_size + rec["size"] > max_file_bytes:
                        files.append(
                            {
                                "filename": cls._make_filename(
                                    next_file_id, rank_tuple, is_common
                                ),
                                "writer_rank": wr,
                                "rank_list": (None if is_common else list(rank_tuple)),
                                "is_common": is_common,
                                "tensors": cur_tensors,
                            }
                        )
                        next_file_id += 1
                        cur_size = 0
                        cur_tensors = []
                    cur_tensors.append(
                        {
                            "stored_key": rec["checksum"],
                            "size": rec["size"],
                            "rank_to_names": rec["rank_to_names"],
                        }
                    )
                    cur_size += rec["size"]
                if cur_tensors:
                    files.append(
                        {
                            "filename": cls._make_filename(
                                next_file_id, rank_tuple, is_common
                            ),
                            "writer_rank": wr,
                            "rank_list": (None if is_common else list(rank_tuple)),
                            "is_common": is_common,
                            "tensors": cur_tensors,
                        }
                    )
                    next_file_id += 1

        rank_to_reads: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
        for f in files:
            for t in f["tensors"]:
                for r_str, names in t["rank_to_names"].items():
                    for name in names:
                        rank_to_reads[int(r_str)].append(
                            {
                                "filename": f["filename"],
                                "stored_key": t["stored_key"],
                                "name": name,
                                "is_extra": name_to_is_extra.get(
                                    (int(r_str), name), False
                                ),
                            }
                        )

        rank_checksums: Dict[str, str] = {}
        for r in range(world_size):
            total = 0
            for rec in rank_to_reads.get(r, []):
                d = cls._fold_name_content_digest(rec["name"], rec["stored_key"])
                total = (total + int.from_bytes(d[:8], "big")) & 0xFFFFFFFFFFFFFFFF
            rank_checksums[str(r)] = format(total, "016x")

        return {
            "version": cls.PLAN_VERSION,
            "world_size": world_size,
            "files": files,
            "rank_to_reads": {str(r): v for r, v in rank_to_reads.items()},
            "rank_checksums": rank_checksums,
        }

    def _dump_files_for_rank(
        self,
        state_dict: Dict[str, torch.Tensor],
        plan: Dict[str, Any],
        rank: int,
        presharded_dir: str,
    ) -> None:
        from safetensors.torch import save_file

        for f in plan["files"]:
            if f["writer_rank"] != rank:
                continue
            tensors_to_save: Dict[str, torch.Tensor] = {}
            for t in f["tensors"]:
                names_for_this_rank = t["rank_to_names"].get(str(rank))
                if not names_for_this_rank:
                    raise RuntimeError(
                        f"writer_rank {rank} is missing tensor {t['stored_key']} "
                        f"for file {f['filename']}; plan is inconsistent."
                    )
                name_for_this_rank = names_for_this_rank[0]
                tensor = (
                    state_dict[name_for_this_rank]
                    .detach()
                    .to(device="cpu", copy=False)
                    .contiguous()
                )
                tensors_to_save[t["stored_key"]] = tensor
            save_file(tensors_to_save, os.path.join(presharded_dir, f["filename"]))

    @staticmethod
    def _read_presharded_file(
        full_path: str, stored_keys: List[str]
    ) -> Dict[str, torch.Tensor]:
        from safetensors.torch import safe_open

        with safe_open(full_path, framework="pt") as fh:
            return {key: fh.get_tensor(key) for key in stored_keys}

    def _apply_presharded_file(
        self,
        *,
        items: List[Dict[str, Any]],
        cached: Dict[str, torch.Tensor],
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        target_device: torch.device,
        loaded_param_keys: set,
        verify_hashes: List[Tuple[str, str]],
    ) -> None:
        if self._verify_on_load:
            keys = list(cached.keys())
            n_workers = min(max(1, len(keys)), self._hash_num_threads)

            def _hash_one(key, _cached=cached):
                return key, self._hash_tensor(_cached[key])

            if n_workers <= 1:
                key_to_hash = dict(_hash_one(k) for k in keys)
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_workers,
                    thread_name_prefix="presharded-verify",
                ) as ex:
                    key_to_hash = dict(ex.map(_hash_one, keys))
            for r in items:
                verify_hashes.append((r["name"], key_to_hash[r["stored_key"]]))

        for r in items:
            tensor = cached[r["stored_key"]]
            if r.get("is_extra"):
                module_path, _, attr_name = r["name"].rpartition(".")
                module = model.get_submodule(module_path) if module_path else model
                if hasattr(module, attr_name):
                    try:
                        delattr(module, attr_name)
                    except AttributeError:
                        pass
                setattr(module, attr_name, tensor.to(target_device))
                continue
            if r["name"] not in state_dict:
                continue
            param_data = state_dict[r["name"]].data
            param_shape = state_dict[r["name"]].shape
            for dim, size in enumerate(tensor.shape):
                if size < param_shape[dim]:
                    param_data = param_data.narrow(dim, 0, size)
            if tensor.shape != param_data.shape:
                raise ValueError(
                    f"Presharded tensor shape mismatch for '{r['name']}': "
                    f"dumped {tuple(tensor.shape)} vs parameter slice "
                    f"{tuple(param_data.shape)} (full param {tuple(param_shape)}). "
                    "Re-dump with matching quant/parallel config, or set "
                    "verify_on_load and check process_weights_after_loading."
                )
            param_data.copy_(tensor)
            loaded_param_keys.add(r["name"])

        cached.clear()
        del cached

    def _load_from_presharded(
        self,
        model_config: ModelConfig,
        device_config: DeviceConfig,
        presharded_dir: str,
    ) -> nn.Module:
        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(model_config, self.load_config, quant_config)

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)

            rank, _ = self._world_rank_and_size()
            with open(os.path.join(presharded_dir, self.CHECKSUM_FILENAME)) as f:
                plan = json.load(f)
            if plan.get("version") != self.PLAN_VERSION:
                raise ValueError(
                    f"Unsupported presharded plan version {plan.get('version')!r} "
                    f"at {presharded_dir}; expected {self.PLAN_VERSION}."
                )

            state_dict = dict(model.state_dict())
            reads = plan.get("rank_to_reads", {}).get(str(rank), [])

            by_file: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
            for r in reads:
                by_file[r["filename"]].append(r)

            loaded_param_keys: set = set()
            verify_hashes: List[Tuple[str, str]] = []
            for filename, items in by_file.items():
                stored_keys = list(dict.fromkeys(r["stored_key"] for r in items))
                cached = self._read_presharded_file(
                    os.path.join(presharded_dir, filename), stored_keys
                )
                self._apply_presharded_file(
                    items=items,
                    cached=cached,
                    model=model,
                    state_dict=state_dict,
                    target_device=target_device,
                    loaded_param_keys=loaded_param_keys,
                    verify_hashes=verify_hashes,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            loaded_storages: set = set()
            for k in loaded_param_keys:
                t = state_dict[k]
                if t.numel() > 0:
                    loaded_storages.add((t.device, t.untyped_storage().data_ptr()))
            missing = []
            for k, t in state_dict.items():
                if k in loaded_param_keys:
                    continue
                if t.numel() == 0:
                    continue
                storage_key = (t.device, t.untyped_storage().data_ptr())
                if storage_key not in loaded_storages:
                    missing.append(k)
            if missing:
                raise ValueError(
                    f"Missing keys {tuple(sorted(missing))} in presharded "
                    f"checkpoint at {presharded_dir}."
                )

            self._rebind_parameter_aliases(model)

            if self._verify_on_load:
                self._verify_rank_checksum(verify_hashes, plan, rank, presharded_dir)

        self.counter_after_loading_weights = time.perf_counter()
        return model.eval()


class BitsAndBytesModelLoader(BaseModelLoader):
    """Model loader to load model weights with BitAndBytes quantization."""

    possible_config_file_names = ["adapter_config.json"]

    default_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".fc1.",
        ".fc2.",
        ".dense.",
        ".query_key_value.",
        ".qkv_proj.",
        ".dense_h_to_4h.",
        ".dense_4h_to_h.",
        ".out_proj.",
    ]

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

        # we don't need to quantize the whole model, only the target modules
        # that are specified in the adapter config file. If the adapter config
        # file is not provided, we will quantize the default modules.
        if (
            not load_config.model_loader_extra_config
            or "qlora_adapter_name_or_path" not in load_config.model_loader_extra_config
        ):
            self.target_modules = []
            return

        qlora_adapter = load_config.model_loader_extra_config[
            "qlora_adapter_name_or_path"
        ]

        config_file_path = self._get_config_file(qlora_adapter)

        with open(config_file_path, "r") as f:
            config = json.load(f)
            self.target_modules = config["target_modules"]

    def _get_config_file(self, qlora_adapter: str) -> str:
        is_local = os.path.isdir(qlora_adapter)
        config_file_path = None
        if is_local:
            for file in self.possible_config_file_names:
                config_file_path = os.path.join(qlora_adapter, file)
                if os.path.exists(config_file_path):
                    break
        else:
            hf_api = HfApi()
            repo_files = hf_api.list_repo_files(repo_id=qlora_adapter)
            for file in self.possible_config_file_names:
                if file in repo_files:
                    config_file_path = hf_hub_download(
                        repo_id=qlora_adapter, filename=file
                    )
                    break

        if not config_file_path:
            raise ValueError(f"Cannot find adapter config file in {qlora_adapter}")

        return config_file_path

    def _get_weight_files(
        self,
        model_name_or_path: str,
        allowed_patterns: List[str],
        revision: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        """Retrieve weight files. Download the files if necessary.

        Return the weight files and the file pattern."""
        is_local = os.path.isdir(model_name_or_path)

        if is_local:
            for pattern in allowed_patterns:
                weight_files = glob.glob(os.path.join(model_name_or_path, pattern))
                if weight_files:
                    return weight_files, pattern
        else:
            hf_api = HfApi()
            repo_files = hf_api.list_repo_files(repo_id=model_name_or_path)
            for pattern in allowed_patterns:
                matching_files = fnmatch.filter(repo_files, pattern)
                if matching_files:
                    hf_folder = download_weights_from_hf(
                        model_name_or_path,
                        self.load_config.download_dir,
                        [pattern],
                        revision,
                        ignore_patterns=self.load_config.ignore_patterns,
                    )
                    return glob.glob(os.path.join(hf_folder, pattern)), pattern

        raise RuntimeError(f"No model weights found in: `{model_name_or_path}`")

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str]
    ) -> Tuple[List[str], bool]:
        """Prepare weight files for the model."""

        allowed_patterns = ["*.safetensors", "*.bin", "*.pt"]

        hf_weights_files, matched_pattern = self._get_weight_files(
            model_name_or_path, allowed_patterns, revision
        )

        if matched_pattern != "*.safetensors":
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_weights_files, matched_pattern == "*.safetensors"

    def _hf_weight_iter(self, hf_weights_files, use_safetensors: bool):
        if use_safetensors:
            return safetensors_weights_iterator(hf_weights_files)
        else:
            return pt_weights_iterator(hf_weights_files)

    def _get_quantized_weights_iterator(
        self,
        model_name_or_path: str,
        revision: Optional[str],
        pre_quant: bool,
        load_8bit: bool,
    ) -> Tuple[Generator[Tuple[str, torch.Tensor], None, None], Dict[str, Any]]:
        """Get an iterator to the model weights with bitsandbytes quantization,
        as well as the quantization state dictionary."""

        # only load the bitsandbytes module when needed
        try:
            import bitsandbytes

            if bitsandbytes.__version__ < "0.44.0":
                raise ImportError(
                    "bitsandbytes version is wrong. Please "
                    "install bitsandbytes>=0.44.0."
                )
        except ImportError as err:
            raise ImportError(
                "Please install bitsandbytes>=0.44.0 via "
                "`pip install bitsandbytes>=0.44.0` to use "
                "bitsandbytes quantizer."
            ) from err

        hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision
        )

        quant_state_dict: Dict[str, Any] = {}

        if pre_quant:
            if load_8bit:
                return (
                    self._quantized_8bit_generator(
                        hf_weights_files, use_safetensors, quant_state_dict
                    ),
                    quant_state_dict,
                )
            else:
                return (
                    self._quantized_4bit_generator(
                        hf_weights_files, use_safetensors, quant_state_dict
                    ),
                    quant_state_dict,
                )

        return (
            self._unquantized_generator(
                hf_weights_files, use_safetensors, quant_state_dict
            ),
            quant_state_dict,
        )

    def _is_8bit_weight_name(self, weight_name: str):
        quantized_suffix = {".scb", ".weight_format"}
        return any(weight_name.lower().endswith(suffix) for suffix in quantized_suffix)

    def _is_4bit_weight_name(self, weight_name: str):
        quantized_suffix = {
            "absmax",
            "quant_map",
            "nested_absmax",
            "nested_quant_map",
            "bitsandbytes",
        }
        suffix = weight_name.split(".")[-1]
        return any(q_suffix in suffix for q_suffix in quantized_suffix)

    def _quantized_8bit_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):
            if not weight_name.lower().endswith(".scb"):
                continue

            weight_key = weight_name.lower().replace(".scb", ".weight")
            quant_state_dict[weight_key] = weight_tensor

        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):
            if self._is_8bit_weight_name(weight_name):
                continue

            if weight_name in quant_state_dict:
                set_weight_attrs(weight_tensor, {"load_in_8bit": True})
                yield weight_name, weight_tensor
            else:
                yield weight_name, weight_tensor

    def _quantized_4bit_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        from bitsandbytes.functional import QuantState

        # First iterate over all quant state weights
        weight_iterator = self._hf_weight_iter(hf_weights_files, use_safetensors)
        temp_state_dict = {}
        for weight_name, weight_tensor in weight_iterator:
            if not self._is_4bit_weight_name(weight_name):
                continue
            # bitsandbytes library requires
            # weight.quant_state.bitsandbytes__* in CPU
            if "quant_state.bitsandbytes" in weight_name:
                temp_state_dict[weight_name] = weight_tensor.cpu().data
            else:
                temp_state_dict[weight_name] = weight_tensor

        # Closure to parse quant_state for each prequant weight
        def _parse_quant_state(param_name: str, temp_state_dict: Dict) -> QuantState:
            quant_state = {}
            for k in temp_state_dict:
                if param_name + "." in k:
                    quant_state[k] = temp_state_dict[k]

            return QuantState.from_dict(quant_state, device="cuda")

        # Second iterate over all prequant and normal weights
        # pre quantized weights would have a quant_state
        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):
            if self._is_4bit_weight_name(weight_name):
                continue

            if (f"{weight_name}.quant_state.bitsandbytes__nf4" in temp_state_dict) or (
                f"{weight_name}.quant_state.bitsandbytes__fp4" in temp_state_dict
            ):
                quant_state = _parse_quant_state(weight_name, temp_state_dict)
                quant_state_dict[weight_name] = quant_state
                yield weight_name, weight_tensor
            else:
                yield weight_name, weight_tensor

    def _unquantized_generator(
        self, hf_weights_files, use_safetensors, quant_state_dict
    ) -> Generator:
        from bitsandbytes.functional import quantize_4bit

        tp_size = get_parallel().tp_size
        tp_rank = get_parallel().tp_rank

        for weight_name, weight_tensor in self._hf_weight_iter(
            hf_weights_files, use_safetensors
        ):
            if any(
                target_module in weight_name for target_module in self.target_modules
            ) and weight_name.endswith(".weight"):
                weight_name = weight_name.replace(".weight", ".qweight")

                if any(
                    module in weight_name
                    for module in self.column_parallel_weights_modules
                ):
                    total_size = weight_tensor.size(-1)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[..., start_index:end_index]

                else:
                    total_size = weight_tensor.size(0)
                    start_index = total_size // tp_size * tp_rank
                    end_index = total_size // tp_size * (tp_rank + 1)
                    weight_sub_tensor = weight_tensor[start_index:end_index, ...]

                # bitsandbytes requires data in GPU
                if weight_sub_tensor.is_cuda:
                    loaded_weight = weight_sub_tensor
                else:
                    loaded_weight = weight_sub_tensor.cuda()

                # remove the following after the issue is fixed:
                # https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1342
                if loaded_weight.is_contiguous() is False:
                    loaded_weight = loaded_weight.contiguous()

                with set_default_torch_dtype(torch.float32):
                    processed_weight, quant_state = quantize_4bit(
                        loaded_weight, compress_statistics=True, quant_type="nf4"
                    )

                quant_state_dict[weight_name] = quant_state
            else:
                processed_weight = weight_tensor

            yield weight_name, processed_weight

    def _load_weights(self, model_config: ModelConfig, model: nn.Module) -> None:
        if not hasattr(model, "load_weights"):
            raise AttributeError(
                "The required method 'load_weights' is not defined in class"
                f" {type(model).__name__}."
            )

        if not hasattr(model, "bitsandbytes_stacked_params_mapping"):
            raise AttributeError(
                f"Model {type(model).__name__} does not support BitsAndBytes "
                "quantization yet."
            )

        if len(self.target_modules) == 0:
            if hasattr(model, "default_bitsandbytes_target_modules"):
                self.target_modules = model.default_bitsandbytes_target_modules
            else:
                self.target_modules = self.default_target_modules

        if hasattr(model, "column_parallel_weights_modules"):
            self.column_parallel_weights_modules = model.column_parallel_weights_modules
        else:
            self.column_parallel_weights_modules = []

        self.model_type = type(model).__name__

        logger.info(
            "Loading weights with BitsAndBytes quantization.  May take a while ..."
        )

        quant_config = getattr(model_config.hf_config, "quantization_config", None)

        pre_quant = False
        if quant_config is not None:
            quant_method = quant_config.get("quant_method")
            if quant_method == "bitsandbytes":
                pre_quant = True
            else:
                raise ValueError(
                    f"BitsAndBytes loader does not support {quant_method} quantization"
                )

        # The quant_states in pre_quantized models cannot work with a split
        # weight tensor. So TP does not work with pre_quantized bnb models.
        if pre_quant and get_parallel().tp_size > 1:
            raise ValueError(
                "Prequant BitsAndBytes models with TP is not supported."
                "Please try with PP."
            )

        load_8bit = False
        if pre_quant:
            load_8bit = quant_config.get("load_in_8bit", False)

        qweight_iterator, quant_state_dict = self._get_quantized_weights_iterator(
            model_config.model_path, model_config.revision, pre_quant, load_8bit
        )

        model.load_weights(qweight_iterator)

        current_platform.empty_cache()

        param_dict = dict(model.named_parameters())
        stacked_quant_state_dict: Dict[str, Dict[int, Any]] = {}
        model_type = model_config.hf_config.model_type
        for quant_param_name in quant_state_dict:
            non_stacked_param_name = quant_param_name
            if model_type == "mllama" and "vision_model" in quant_param_name:
                # adapt to VisionAttention
                quant_param_name = quant_param_name.replace(
                    "self_attn.o_proj", "self_attn.proj"
                )
            shard_index = 0
            for shard_name, (
                weight_name,
                index,
            ) in model.bitsandbytes_stacked_params_mapping.items():
                if (
                    model_type in ["qwen2_vl", "qwen2_5_vl"]
                    and "visual" in quant_param_name
                ):
                    break
                if shard_name in quant_param_name:
                    shard_index = index
                    quant_param_name = quant_param_name.replace(shard_name, weight_name)
                    break

            if (
                model_type in ["qwen2_vl", "qwen2_5_vl"]
                and "visual" in quant_param_name
            ):
                quant_param_name = quant_param_name.replace(
                    r"attn.qkv.", r"attn.qkv_proj."
                )

            if quant_param_name not in param_dict:
                raise ValueError(
                    f"Parameter {quant_param_name} not found in the model."
                )

            if quant_param_name not in stacked_quant_state_dict:
                stacked_quant_state_dict[quant_param_name] = {}

            stacked_quant_state_dict[quant_param_name][shard_index] = quant_state_dict[
                non_stacked_param_name
            ]

        # save quant_states and offsets as the attributes of the parameters
        for param_name, param in param_dict.items():
            if param_name in stacked_quant_state_dict:
                quant_states = stacked_quant_state_dict[param_name]
                set_weight_attrs(param, {"bnb_quant_state": quant_states})

                pack_ratio = getattr(param, "pack_factor", -1)
                if pack_ratio == -1:
                    raise ValueError(f"pack_factor not set for parameter {param_name}.")

                num_elements = [0] * len(quant_states)
                for seq, quant_state in quant_states.items():
                    num_elements[seq] = math.prod(quant_state.shape) // pack_ratio

                offsets = np.concatenate(([0], np.cumsum(num_elements)))
                # Make torch infer_schema happy(Compatible with vLLM)
                offsets = torch.tensor(offsets).cpu()
                set_weight_attrs(param, {"bnb_shard_offsets": offsets})

                if load_8bit:
                    set_weight_attrs(
                        param, {"matmul_state": [None] * len(quant_states)}
                    )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path, model_config.revision)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

                self._load_weights(model_config, model)

        return model.eval()


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(self, model_name_or_path: str):
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        else:
            raise ValueError(f"{model_name_or_path} is not a file.")

    def _get_gguf_weights_map(self, model_config: ModelConfig):
        """
        GGUF uses this naming convention for their tensors from HF checkpoint:
        `blk.N.BB.weight` and `blk.N.BB.bias`
        where N signifies the block number of a layer, and BB signifies the
        attention/mlp layer components.
        See "Standardized tensor names" in
        https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
        """

        # only load the gguf module when needed
        try:
            import gguf

            # FIXME: add version check for gguf
        except ImportError as err:
            raise ImportError(
                "Please install gguf via `pip install gguf` to use gguf quantizer."
            ) from err

        config = model_config.hf_config
        model_type = config.model_type
        # hack: ggufs have a different name than transformers
        if model_type == "cohere":
            model_type = "command-r"
        elif model_type == "qwen3_moe":
            model_type = "qwen3moe"
        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")
        num_layers = config.num_hidden_layers
        name_map = gguf.get_tensor_name_map(arch, num_layers)
        with torch.device("meta"):
            dummy_model = AutoModelForCausalLM.from_config(config)
        state_dict = dummy_model.state_dict()

        gguf_to_hf_name_map = {}
        for hf_name in state_dict:
            name, suffix = hf_name.rsplit(".", 1)
            gguf_name = name_map.get_name(name)
            gguf_to_hf_name_map[f"{gguf_name}.{suffix}"] = hf_name
        return gguf_to_hf_name_map

    def _get_weights_iterator(
        self, model_name_or_path: str, gguf_to_hf_name_map: Dict[str, str]
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        return gguf_quant_weights_iterator(model_name_or_path, gguf_to_hf_name_map)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        local_model_path = self._prepare_weights(model_config.model_path)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        if "lm_head.weight" in get_gguf_extra_tensor_names(
            local_model_path, gguf_weights_map
        ):
            model_config.hf_config.update({"tie_word_embeddings": True})

        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(model_config, self.load_config, quant_config)
            model.load_weights(
                self._get_weights_iterator(local_model_path, gguf_weights_map)
            )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)
        return model


@dataclasses.dataclass(frozen=True)
class _ColdStartWeightLoadAttestor:
    target_resource: Any
    target_binding: Any

    def attest(self, request: Any) -> None:
        if tuple(request.plan.target_bindings) != (self.target_binding,):
            raise RuntimeError(
                "weight snapshot target binding changed before execution"
            )
        attest_binding = getattr(self.target_resource, "attest_binding", None)
        if not callable(attest_binding):
            raise RuntimeError("target runtime does not support binding attestation")
        attest_binding(self.target_binding)


@dataclasses.dataclass
class _PendingWeightSnapshotActivation:
    """Own loader resources until startup activation finishes."""

    ref: Any
    catalog: Any
    resources: ExitStack
    backend: Any | None = None
    deadline_unix_sec: float | None = None
    _transaction_id: str | None = dataclasses.field(default=None, init=False)
    _prepared_head: Any = dataclasses.field(default=None, init=False, repr=False)
    _owns_serving_transition: bool = dataclasses.field(
        default=False,
        init=False,
        repr=False,
    )
    _committed_head: Any = dataclasses.field(default=None, init=False, repr=False)
    _state: str = dataclasses.field(default="loaded", init=False)
    _closed: bool = dataclasses.field(default=False, init=False, repr=False)
    _quarantined: bool = dataclasses.field(default=False, init=False, repr=False)

    def _catalog_for_deadline(self, deadline_unix_sec: float | None):
        deadlines = [
            deadline
            for deadline in (self.deadline_unix_sec, deadline_unix_sec)
            if deadline is not None
        ]
        if not deadlines:
            return self.catalog
        if any(
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            for deadline in deadlines
        ):
            raise ValueError("weight snapshot activation deadline is invalid")
        execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=min(deadlines)
        )
        if execution_context.expired():
            raise TimeoutError("weight snapshot activation deadline expired")
        with_execution_context = getattr(
            self.catalog,
            "with_execution_context",
            None,
        )
        if callable(with_execution_context):
            return with_execution_context(execution_context)
        return self.catalog

    def _loadable_head(self, catalog=None):
        from sglang.srt.weight_transfer.storage import WeightRevisionState

        catalog = self.catalog if catalog is None else catalog
        snapshot = catalog.get_snapshot(self.ref)
        if snapshot is None:
            raise ValueError("published weight snapshot was not found")
        identities = {
            (placement.model_id, placement.revision)
            for placement in snapshot.placements
        }
        if len(identities) != 1:
            raise ValueError("weight snapshot has no canonical model revision")
        model_id, revision = next(iter(identities))
        head = catalog.get_revision_head(model_id, revision)
        if (
            head is None
            or head.ref != self.ref
            or head.state
            not in {
                WeightRevisionState.READY,
                WeightRevisionState.SERVING,
            }
        ):
            raise RuntimeError("weight snapshot revision head is not READY or SERVING")
        return head

    def _require_transaction(self, transaction_id: str) -> None:
        if type(transaction_id) is not str or not transaction_id:
            raise ValueError("activation transaction_id must be a non-empty string")
        if self._transaction_id not in (None, transaction_id):
            raise RuntimeError("weight snapshot activation transaction conflicts")
        self._transaction_id = transaction_id

    def prepare(
        self,
        transaction_id: str,
        *,
        deadline_unix_sec: float | None = None,
    ):
        from sglang.srt.weight_transfer.storage import WeightRevisionState

        self._require_transaction(transaction_id)
        if self._closed or self._quarantined:
            raise RuntimeError("weight snapshot activation owner is unavailable")
        catalog = self._catalog_for_deadline(deadline_unix_sec)
        head = self._loadable_head(catalog)
        if self._owns_serving_transition:
            if self._committed_head is None or head != self._committed_head:
                raise RuntimeError(
                    "owned weight snapshot SERVING revision changed after commit"
                )
            self._prepared_head = head
            self._state = "serving"
            return head
        self._prepared_head = head
        self._owns_serving_transition = False
        self._committed_head = None
        self._state = (
            "serving" if head.state is WeightRevisionState.SERVING else "prepared"
        )
        return head

    def commit(
        self,
        transaction_id: str,
        *,
        deadline_unix_sec: float | None = None,
    ):
        from sglang.srt.weight_transfer.storage import WeightRevisionState

        self._require_transaction(transaction_id)
        if self._prepared_head is None:
            raise RuntimeError("weight snapshot activation was not prepared")
        catalog = self._catalog_for_deadline(deadline_unix_sec)
        current = self._loadable_head(catalog)
        if current.state is WeightRevisionState.SERVING:
            self._state = "serving"
            return current
        if current != self._prepared_head:
            raise RuntimeError("weight snapshot revision head changed after prepare")
        updated = catalog.compare_and_set_revision(
            model_id=current.model_id,
            revision=current.revision,
            expected=self._prepared_head,
            new_ref=self.ref,
            new_state=WeightRevisionState.SERVING,
        )
        if updated is None:
            updated = self._loadable_head(catalog)
            if updated.state is not WeightRevisionState.SERVING:
                raise RuntimeError("weight snapshot SERVING CAS did not commit")
        else:
            self._committed_head = updated
            self._owns_serving_transition = True
        self._prepared_head = updated
        self._state = "serving"
        return updated

    def reconcile(
        self,
        transaction_id: str,
        *,
        deadline_unix_sec: float | None = None,
    ) -> str:
        from sglang.srt.weight_transfer.storage import WeightRevisionState

        self._require_transaction(transaction_id)
        catalog = self._catalog_for_deadline(deadline_unix_sec)
        try:
            current = self._loadable_head(catalog)
        except Exception:
            return "conflict"
        if current.state is WeightRevisionState.SERVING:
            self._prepared_head = current
            self._state = "serving"
            return "serving"
        if current == self._prepared_head:
            self._state = "prepared"
            return "prepared"
        return "conflict"

    def abort(
        self,
        transaction_id: str,
        *,
        deadline_unix_sec: float | None = None,
    ) -> str:
        from sglang.srt.weight_transfer.storage import WeightRevisionState

        self._require_transaction(transaction_id)
        if self._closed:
            return "aborted"
        catalog = self._catalog_for_deadline(deadline_unix_sec)
        try:
            current = self._loadable_head(catalog)
        except Exception:
            self.quarantine(catalog=catalog)
            return "quarantined"
        if self._owns_serving_transition:
            if current == self._committed_head:
                self.quarantine(current, catalog=catalog)
            else:
                self.quarantine(catalog=catalog)
            return "quarantined"
        if current.state is WeightRevisionState.SERVING:
            self._state = "aborted"
            self.close()
            return "aborted"
        if self._prepared_head is not None and current != self._prepared_head:
            self.quarantine(catalog=catalog)
            return "quarantined"
        self._state = "aborted"
        self.close()
        return "aborted"

    def quarantine(self, current=None, *, catalog=None) -> None:
        from sglang.srt.weight_transfer.storage import WeightRevisionState

        if self._quarantined:
            return
        catalog = self.catalog if catalog is None else catalog
        self._quarantined = True
        self._state = "quarantined"
        _WEIGHT_SNAPSHOT_ACTIVATION_QUARANTINE.append(self)
        try:
            if current is None:
                current = self._loadable_head(catalog)
            if (
                self._owns_serving_transition
                and self._committed_head is not None
                and current == self._committed_head
                and current.state is WeightRevisionState.SERVING
            ):
                updated = catalog.compare_and_set_revision(
                    model_id=current.model_id,
                    revision=current.revision,
                    expected=current,
                    new_ref=self.ref,
                    new_state=WeightRevisionState.IDLE,
                )
                if updated is None:
                    observed = catalog.get_revision_head(
                        current.model_id,
                        current.revision,
                    )
                    if (
                        observed is not None
                        and observed.ref == self.ref
                        and observed.state is WeightRevisionState.SERVING
                    ):
                        logger.error(
                            "Quarantined weight snapshot remains SERVING after "
                            "revision CAS conflict"
                        )
        except Exception:
            logger.exception(
                "Failed to move quarantined weight snapshot out of SERVING"
            )

    def activate(self) -> None:
        from sglang.srt.weight_transfer.api import mark_weight_snapshot_serving

        mark_weight_snapshot_serving(
            self.ref,
            catalog=self.catalog,
        )
        self._state = "serving"

    def close(self) -> None:
        if self._closed or self._quarantined:
            return
        if self._state == "cleanup_pending" and self.backend is not None:
            try:
                status = self.backend.close(
                    timeout_ms=_WEIGHT_SNAPSHOT_CLEANUP_TIMEOUT_MS,
                )
                if not status.closed:
                    retry_error = RuntimeError(
                        "weight snapshot backend cleanup remains pending: "
                        + ", ".join(status.pending_tickets)
                    )
                    retry_error.completion_unknown = True
                    raise retry_error
            except Exception:
                logger.exception(
                    "Weight snapshot activation cleanup retry remains pending"
                )
                raise
            self._closed = True
            self._state = "closed"
            return
        try:
            self.resources.close()
        except Exception as error:
            if self.backend is not None:
                try:
                    status = self.backend.close(
                        timeout_ms=_WEIGHT_SNAPSHOT_CLEANUP_TIMEOUT_MS,
                    )
                    if not status.closed:
                        retry_error = RuntimeError(
                            "weight snapshot backend cleanup remains pending: "
                            + ", ".join(status.pending_tickets)
                        )
                        retry_error.completion_unknown = True
                        raise retry_error
                except Exception as retry_error:
                    if self not in _WEIGHT_SNAPSHOT_CLEANUP_QUARANTINE:
                        _WEIGHT_SNAPSHOT_CLEANUP_QUARANTINE.append(self)
                    self._state = "cleanup_pending"
                    logger.exception(
                        "Weight snapshot activation cleanup remains pending; "
                        "retaining the owner for retry"
                    )
                    raise retry_error from error
                logger.warning(
                    "Weight snapshot activation cleanup completed on backend retry"
                )
            else:
                if self not in _WEIGHT_SNAPSHOT_CLEANUP_QUARANTINE:
                    _WEIGHT_SNAPSHOT_CLEANUP_QUARANTINE.append(self)
                logger.exception(
                    "Weight snapshot activation finished, but loader resource "
                    "cleanup failed; retaining the owner for process lifetime"
                )
        self._closed = True
        if self._state != "aborted":
            self._state = "closed"


class WeightSnapshotModelLoader(BaseModelLoader):
    """Load a published semantic snapshot into final non-serving buffers."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        from sglang.srt.weight_transfer.store_runtime import WeightSnapshotLoadSpec

        self.spec = WeightSnapshotLoadSpec.from_mapping(
            cast(dict[str, Any], load_config.model_loader_extra_config or {})
        )
        self._pending_activation: _PendingWeightSnapshotActivation | None = None

    def take_pending_weight_snapshot_activation(
        self,
    ) -> WeightSnapshotActivation | None:
        pending = self._pending_activation
        self._pending_activation = None
        return pending

    def download_model(self, model_config: ModelConfig) -> None:
        del model_config

    def _backend_context(
        self,
        execution_context: WeightTransferExecutionContext,
    ):
        from sglang.srt.weight_transfer.store_runtime import (
            open_weight_snapshot_backend,
        )

        factory = self.load_config.weight_snapshot_backend_factory
        if callable(factory):
            return factory(self.spec)
        if not model_parallel_is_initialized():
            return open_weight_snapshot_backend(
                self.spec,
                execution_context=execution_context,
            )
        world_group = get_world_group()
        return open_weight_snapshot_backend(
            self.spec,
            rank=world_group.rank_in_group,
            world_size=world_group.world_size,
            execution_context=execution_context,
        )

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        from sglang.srt.model_executor.weight_runtime_manifest import (
            WeightPlacementManifest,
            WeightRuntimeBindingManifest,
        )
        from sglang.srt.weight_transfer.api import (
            load_weight_snapshot,
        )
        from sglang.srt.weight_transfer.provider import (
            WeightTargetLoadMode,
            WeightTransferCompletionUnknownError,
        )
        from sglang.srt.weight_transfer.store_runtime import WeightSnapshotBackend

        target_builder = (
            self.load_config.remote_instance_weight_runtime_manifest_builder
        )
        if not callable(target_builder):
            raise RuntimeError(
                "weight snapshot loading requires --enable-weight-runtime-manifest"
            )
        if self._pending_activation is not None:
            raise RuntimeError(
                "weight snapshot activation is still owned by the loader"
            )

        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

        execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=time.time() + self.spec.load_timeout_sec,
        )
        resources = ExitStack()
        retain_resources = False
        try:
            backend = resources.enter_context(self._backend_context(execution_context))
            if not isinstance(backend, WeightSnapshotBackend):
                raise ValueError(
                    "weight snapshot backend factory returned an invalid backend"
                )
            if backend.provider.name != self.spec.ref.provider:
                raise ValueError(
                    "weight snapshot provider differs from the storage ref"
                )
            require_bounded_execution_contract(
                backend.provider,
                role="weight snapshot provider",
            )
            snapshot = backend.catalog.get_snapshot(self.spec.ref)
            if snapshot is None:
                raise ValueError("published weight snapshot was not found")
            source_identities = {
                (placement.model_id, placement.revision)
                for placement in snapshot.placements
            }
            expected_identity = (self.spec.model_id, self.spec.revision)
            if source_identities != {expected_identity}:
                raise ValueError(
                    "published weight snapshot model identity differs from load spec"
                )

            instance_id = self.spec.instance_id or (
                f"sglang-weight-snapshot:{os.getpid()}:{id(model)}"
            )
            target_resource = resources.enter_context(
                target_builder(
                    model=model,
                    model_id=self.spec.model_id,
                    revision=self.spec.revision,
                    instance_id=instance_id,
                    endpoint=backend.endpoint,
                )
            )
            target_placement = getattr(target_resource, "placement", None)
            bind = getattr(target_resource, "bind", None)
            if not isinstance(
                target_placement, WeightPlacementManifest
            ) or not callable(bind):
                raise ValueError(
                    "weight snapshot target builder did not return placement/bind"
                )
            target_binding = resources.enter_context(bind())
            if not isinstance(target_binding, WeightRuntimeBindingManifest):
                raise ValueError("weight snapshot target binding is invalid")

            load_started = time.perf_counter()
            receipt = load_weight_snapshot(
                self.spec.ref,
                catalog=backend.catalog,
                target_placements=(target_placement,),
                target_bindings=(target_binding,),
                provider=backend.provider,
                target_mode=WeightTargetLoadMode.COLD_START,
                attestor=_ColdStartWeightLoadAttestor(
                    target_resource=target_resource,
                    target_binding=target_binding,
                ),
                execution_context=execution_context,
            )
            phase_text = ", ".join(
                f"{name}={seconds:.4f}s"
                for name, seconds in receipt.provider_phase_seconds
            )
            logger.info(
                "Loaded weight snapshot: provider=%s, logical_bytes=%d, "
                "compact_regions=%d, elapsed=%.4fs, phases=[%s]",
                receipt.provider,
                receipt.total_bytes,
                receipt.region_count,
                time.perf_counter() - load_started,
                phase_text,
            )
            current_platform.synchronize()
            _post_load_weights(model)
            model = model.eval()
            self._pending_activation = _PendingWeightSnapshotActivation(
                ref=self.spec.ref,
                catalog=backend.catalog,
                resources=resources,
                backend=backend,
                deadline_unix_sec=execution_context.deadline_unix_sec,
            )
            resources = ExitStack()
        except WeightTransferCompletionUnknownError as error:
            retain_resources = True
            _WEIGHT_SNAPSHOT_UNKNOWN_LOAD_QUARANTINE.append(
                (model, resources, error.completion_ticket)
            )
            raise
        finally:
            if not retain_resources:
                resources.close()
        return model


class RemoteInstanceModelLoader(BaseModelLoader):
    """Model loader that can load Tensors from remote sglang instance."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )
        self.remote_instance_transfer_engine_weight_info = None

    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        logger.info("Loading weights from remote instance ...")
        load_config = self.load_config

        assert load_config.load_format == LoadFormat.REMOTE_INSTANCE, (
            f"Model loader {self.load_config.load_format} is not supported for "
            f"load format {load_config.load_format}"
        )

        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, quant_config)

        if (
            load_config.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            model_weights = f"instance://{load_config.remote_instance_weight_loader_seed_instance_ip}:{load_config.remote_instance_weight_loader_send_weights_group_ports[load_config.tp_rank]}"
            with create_remote_connector(model_weights, device_config.device) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.INSTANCE:
                    self.load_model_from_remote_instance_by_nccl(
                        model, client, model_config, device_config
                    )
                else:
                    raise ValueError(
                        f"Unsupported connector type {connector_type} for "
                        f"remote tensor model loading."
                    )
        elif (
            load_config.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE
        ):
            if load_config.remote_instance_weight_loader_transfer_engine is None:
                raise RuntimeError(
                    "Transfer engine is not initialized for remote instance "
                    "model loader with `transfer_engine` backend. "
                )
            logger.info(
                "TransferEngine registering memory regions (this may take a few seconds)..."
            )
            # register memory region
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                model, load_config.remote_instance_weight_loader_transfer_engine
            )
            logger.info(
                "TransferEngine memory regions have been successfully registered."
            )

            # transfer weights
            seed_url = f"http://{load_config.remote_instance_weight_loader_seed_instance_ip}:{load_config.remote_instance_weight_loader_seed_instance_service_port}"
            provider_factory = getattr(
                load_config,
                "remote_instance_weight_transfer_provider_factory",
                None,
            )
            if load_config.remote_instance_weight_runtime_manifest_builder is not None:
                success = self.load_model_from_remote_instance_by_transfer_engine_heterogeneous(
                    model,
                    load_config.remote_instance_weight_loader_transfer_engine,
                    seed_url,
                    load_config.remote_instance_weight_loader_transfer_engine_session_id,
                    load_config.remote_instance_weight_runtime_manifest_builder,
                    target_model_id=model_config.model_path,
                    target_artifact_revision=_configured_weight_artifact_revision(),
                    target_hf_revision=model_config.revision or "default",
                    provider_factory=provider_factory,
                )
            else:
                if provider_factory is not None:
                    raise RuntimeError(
                        "configured weight transfer provider requires a runtime "
                        "manifest builder"
                    )
                success = self.load_model_from_remote_instance_by_transfer_engine(
                    model,
                    load_config.remote_instance_weight_loader_transfer_engine,
                    seed_url,
                    load_config.tp_rank,
                )
            if not success:
                raise RuntimeError(
                    "Failed to load weights from remote instance via transfer engine."
                )
        elif (
            load_config.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.MODELEXPRESS
        ):
            try:
                from modelexpress.engines.sglang.loader import MxModelLoader
            except ImportError as exc:
                raise ImportError(
                    "ModelExpress support requires the 'modelexpress' "
                    "package. Install it in the SGLang image."
                ) from exc

            model = MxModelLoader(load_config).load_model(
                model=model,
                model_config=model_config,
                device_config=device_config,
            )
        else:
            raise ValueError("Invalid remote instance weight loader backend.")

        return model.eval()

    def load_model_from_remote_instance_by_nccl(
        self, model, client, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        load_config = self.load_config
        instance_ip = socket.gethostbyname(socket.gethostname())
        start_build_group_tic = time.time()
        client.build_group(
            gpu_id=device_config.gpu_id,
            tp_rank=load_config.tp_rank,
            instance_ip=instance_ip,
        )
        current_platform.synchronize()
        end_build_group_tic = time.time()
        logger.debug(
            f"finish building group for remote instance, time used: {(end_build_group_tic - start_build_group_tic):.4f}s"
        )

        if load_config.tp_rank == 0:
            t = threading.Thread(
                target=trigger_transferring_weights_request,
                args=(
                    load_config.remote_instance_weight_loader_seed_instance_ip,
                    load_config.remote_instance_weight_loader_seed_instance_service_port,
                    load_config.remote_instance_weight_loader_send_weights_group_ports,
                    instance_ip,
                ),
            )
            t.start()

        start_get_weights_tic = time.time()
        with set_default_torch_dtype(model_config.dtype):
            for _, tensor in model.named_parameters():
                torch.distributed.broadcast(
                    tensor.data,
                    src=0,
                    group=client._model_update_group,
                )
            current_platform.synchronize()

            _post_load_weights(model)
        end_get_weights_tic = time.time()
        logger.debug(
            f"finish getting all weights from remote instance, time used: {(end_get_weights_tic - start_get_weights_tic):.4f}s"
        )
        # destroy the process group after loading weights
        torch.distributed.distributed_c10d.destroy_process_group(
            client._model_update_group
        )
        current_platform.empty_cache()

    def load_model_from_remote_instance_by_transfer_engine(
        self, model, transfer_engine, seed_url, tp_rank
    ) -> bool:
        # get remote weights metadata from source instance
        seed_transfer_engine_session_id, seed_transfer_engine_weight_info = (
            get_remote_instance_transfer_engine_info_per_rank(seed_url, tp_rank)
        )
        if (
            seed_transfer_engine_session_id is None
            or seed_transfer_engine_weight_info is None
        ):
            logger.error("Cannot get transfer engine session or weight info.")
            return False

        # prepare local/remote RDMA keys
        seed_ptr_list = []
        client_ptr_list = []
        client_len_list = []
        for name, tensor in model.named_parameters():
            weight_info = seed_transfer_engine_weight_info.get(name, None)
            if weight_info is None:
                logger.error(f"Cannot find weight info for {name}.")
                return False

            seed_ptr, seed_numel, seed_element_size = weight_info
            if (
                seed_numel != tensor.numel()
                or seed_element_size != tensor.element_size()
            ):
                logger.error(
                    f"Weight info does not match for {name}, "
                    f"expected ({seed_numel}, {seed_element_size}), "
                    f"got ({tensor.numel()}, {tensor.element_size()})"
                )
                return False
            client_ptr = tensor.data_ptr()
            client_len = tensor.numel() * tensor.element_size()
            seed_ptr_list.append(seed_ptr)
            client_ptr_list.append(client_ptr)
            client_len_list.append(client_len)

        # Prefer the ticket API so target parameters remain alive until the
        # native transfer reaches a known terminal state.
        ticket_method = getattr(
            transfer_engine,
            "batch_transfer_sync_read_with_ticket",
            None,
        )
        if callable(ticket_method):
            ticket = ticket_method(
                seed_transfer_engine_session_id,
                client_ptr_list,
                seed_ptr_list,
                client_len_list,
            )
            status = _transfer_completion_status_name(ticket.status)
            if status == "COMPLETION_UNKNOWN":
                logger.error(
                    "Legacy remote weight transfer completion is unknown; "
                    "retaining the target model while draining the native ticket"
                )
            pending_interrupt = None
            while status == "COMPLETION_UNKNOWN":
                try:
                    status = _transfer_completion_status_name(ticket.drain(1000))
                except BaseException as error:
                    if isinstance(error, Exception):
                        logger.exception(
                            "Failed to query the legacy remote weight transfer "
                            "ticket; target parameters remain quarantined"
                        )
                        time.sleep(1)
                    else:
                        if pending_interrupt is None:
                            pending_interrupt = error
                        logger.warning(
                            "Deferring process interruption until the legacy "
                            "remote weight transfer reaches a terminal state"
                        )
            if pending_interrupt is not None:
                raise pending_interrupt
            ret = 0 if status == "COMPLETED" else -1
        else:
            ret = transfer_engine.batch_transfer_sync_read(
                seed_transfer_engine_session_id,
                client_ptr_list,
                seed_ptr_list,
                client_len_list,
            )
            if ret == -2:
                _LEGACY_UNKNOWN_TRANSFER_QUARANTINE.append(
                    (model, transfer_engine, tuple(client_ptr_list))
                )
                logger.error(
                    "Legacy Transfer Engine returned completion unknown without "
                    "a ticket API; retaining the target model for process lifetime"
                )
                return False
        if ret < 0:
            logger.error("batch transfer failed, error: %s", ret)
            return False

        _post_load_weights(model)

        return True

    @staticmethod
    def _runtime_v1_target_manifest_builder(target_manifest_builder):
        owner = getattr(target_manifest_builder, "__self__", None)
        legacy_builder = getattr(
            owner,
            "build_remote_instance_target_weight_runtime_manifest",
            None,
        )
        return legacy_builder if callable(legacy_builder) else target_manifest_builder

    @staticmethod
    def _require_manifest_identity(
        manifests,
        *,
        model_id: str,
        revision: str,
        role: str,
    ) -> None:
        mismatches = {
            (manifest.model_id, manifest.revision)
            for manifest in manifests
            if (manifest.model_id, manifest.revision) != (model_id, revision)
        }
        if mismatches:
            raise ValueError(
                f"{role} manifest identity does not match target model "
                f"{model_id}@{revision}: {sorted(mismatches)}"
            )

    @staticmethod
    def _coerce_weight_manifest_inventory(inventory, manifest_type):
        if isinstance(inventory, manifest_type):
            return inventory
        return msgspec.convert(inventory, type=manifest_type)

    @staticmethod
    def _migrate_runtime_v1_inventory(inventory):
        if not isinstance(inventory, collections.abc.Mapping):
            return inventory
        if inventory.get("format_version", 1) != 1:
            return inventory

        migrated = dict(inventory)
        migrated["format_version"] = 1
        tensors = inventory.get("tensors")
        if not isinstance(tensors, (list, tuple)):
            return migrated

        migrated_tensors = []
        for tensor in tensors:
            if (
                isinstance(tensor, collections.abc.Mapping)
                and "shard_dims" not in tensor
            ):
                tensor = dict(tensor)
                partition_dim = tensor.get("partition_dim")
                tensor["shard_dims"] = () if partition_dim is None else (partition_dim,)
            migrated_tensors.append(tensor)
        migrated["tensors"] = tuple(migrated_tensors)
        return migrated

    def _adapt_runtime_v1_source_inventories(self, inventories):
        if not inventories:
            raise ValueError("source runtime manifest inventories must not be empty")
        parts = tuple(
            runtime_manifest_to_parts(
                self._coerce_weight_manifest_inventory(
                    self._migrate_runtime_v1_inventory(inventory),
                    WeightRuntimeManifest,
                )
            )
            for inventory in inventories
        )
        return (
            tuple(part.placement for part in parts),
            tuple(part.binding for part in parts),
        )

    def _plan_rank_local_transfer_envelopes(
        self,
        *,
        gathered_targets,
        owner_source_session,
        target_model_id,
        manifest_revision,
        world_size,
        phase_seconds,
    ) -> list[_RankLocalTransferEnvelope]:
        if (
            not isinstance(gathered_targets, (list, tuple))
            or len(gathered_targets) != world_size
        ):
            raise ValueError("target placement gather is incomplete")

        targets = []
        placement_ids = set()
        parallel_ranks = set()
        for expected_rank, item in enumerate(gathered_targets):
            if not isinstance(item, _TargetPlacementEnvelope):
                raise ValueError("target placement envelope is invalid")
            if item.world_rank != expected_rank:
                raise ValueError("target placement envelope rank differs")
            if item.error is not None:
                raise RuntimeError(
                    f"target rank {expected_rank} placement failed: {item.error}"
                )
            if not isinstance(
                item.placement, WeightPlacementManifest
            ) or not isinstance(item.parallel_rank, WeightParallelRank):
                raise ValueError("target placement envelope is incomplete")
            if _placement_parallel_rank(item.placement) != item.parallel_rank:
                raise ValueError("target placement parallel rank differs")
            if item.placement.placement_id in placement_ids:
                raise ValueError("duplicate target placement")
            if item.parallel_rank in parallel_ranks:
                raise ValueError("duplicate target parallel rank")
            placement_ids.add(item.placement.placement_id)
            parallel_ranks.add(item.parallel_rank)
            targets.append(item)

        if owner_source_session is None:
            raise RuntimeError("target root does not own the source manifest payload")
        source_placement_inventories = getattr(
            owner_source_session,
            "source_placements",
            None,
        )
        source_binding_inventories = getattr(
            owner_source_session,
            "source_bindings",
            None,
        )
        source_started = time.perf_counter()
        if (
            not source_placement_inventories
            or not source_binding_inventories
            or len(source_placement_inventories) != len(source_binding_inventories)
        ):
            raise ValueError(
                "source placement and runtime binding inventories must be paired"
            )
        source_placements = tuple(
            self._coerce_weight_manifest_inventory(
                inventory,
                WeightPlacementManifest,
            )
            for inventory in source_placement_inventories
        )
        source_bindings = tuple(
            self._coerce_weight_manifest_inventory(
                inventory,
                WeightRuntimeBindingManifest,
            )
            for inventory in source_binding_inventories
        )
        self._require_manifest_identity(
            source_placements,
            model_id=target_model_id,
            revision=manifest_revision,
            role="source",
        )
        source_bindings = tuple(
            project_source_bindings(source_placements, source_bindings)
        )
        phase_seconds["source_manifest"] = time.perf_counter() - source_started

        target_placements = tuple(item.placement for item in targets)
        expected_topology = tuple(item.parallel_rank for item in targets)
        plan_started = time.perf_counter()
        global_plan = plan_weight_transfer(
            source_placements,
            target_placements,
            expected_target_topology=expected_topology,
        )
        local_plans = project_weight_transfer_plan_to_targets(global_plan)
        result = {}
        for item in targets:
            local_plan = local_plans[item.placement.placement_id]
            local_bindings = tuple(
                project_source_bindings(
                    local_plan.source_placements,
                    source_bindings,
                )
            )
            result[item.world_rank] = _RankLocalTransferEnvelope(
                world_rank=item.world_rank,
                parallel_rank=item.parallel_rank,
                target_placement_id=item.placement.placement_id,
                logical_plan=local_plan,
                source_bindings=local_bindings,
            )
        phase_seconds["plan"] = time.perf_counter() - plan_started
        return [result[rank] for rank in range(world_size)]

    def _prepare_distributed_native_heterogeneous_weight_load(
        self,
        *,
        model,
        coordinator,
        world_group,
        transfer_resources,
        target_manifest_builder,
        target_model_id,
        manifest_revision,
        local_session_id,
        transfer_executor,
        phase_seconds,
    ) -> _PreparedNativeHeterogeneousWeightLoad:
        if transfer_executor is None:
            raise RuntimeError(
                "placement_binding_v1 requires a configured weight transfer provider"
            )
        local_rank = getattr(world_group, "rank_in_group", None)
        world_size = getattr(world_group, "world_size", None)
        if (
            type(local_rank) is not int
            or type(world_size) is not int
            or not 0 <= local_rank < world_size
        ):
            raise ValueError("target world rank metadata is invalid")

        target_resource = None
        target_placement = None
        parallel_rank = None
        target_error = None
        target_started = time.perf_counter()
        try:
            target_resource = transfer_resources.enter_context(
                target_manifest_builder(
                    model=model,
                    model_id=target_model_id,
                    revision=manifest_revision,
                    instance_id=f"sglang:{local_session_id}",
                    endpoint=local_session_id,
                )
            )
            if not hasattr(target_resource, "placement") or not callable(
                getattr(target_resource, "bind", None)
            ):
                raise ValueError(
                    "placement_binding_v1 requires a target manifest session "
                    "with placement and bind()"
                )
            target_placement = self._coerce_weight_manifest_inventory(
                target_resource.placement,
                WeightPlacementManifest,
            )
            self._require_manifest_identity(
                (target_placement,),
                model_id=target_model_id,
                revision=manifest_revision,
                role="target",
            )
            parallel_rank = _placement_parallel_rank(target_placement)
        except BaseException as error:
            target_error = _compact_target_planning_error(error)
        phase_seconds["target_manifest"] = time.perf_counter() - target_started

        local_target = _TargetPlacementEnvelope(
            world_rank=local_rank,
            parallel_rank=parallel_rank,
            placement=target_placement,
            error=target_error,
        )
        scattered = None
        try:
            gathered_targets = world_group.gather_object(local_target, dst=0)
            scatter_inputs = None
            if local_rank == 0:
                try:
                    scatter_inputs = self._plan_rank_local_transfer_envelopes(
                        gathered_targets=gathered_targets,
                        owner_source_session=coordinator.owner_source_session,
                        target_model_id=target_model_id,
                        manifest_revision=manifest_revision,
                        world_size=world_size,
                        phase_seconds=phase_seconds,
                    )
                except BaseException as error:
                    compact_error = _compact_target_planning_error(error)
                    scatter_inputs = [
                        _RankLocalTransferEnvelope(
                            world_rank=rank,
                            parallel_rank=None,
                            target_placement_id=None,
                            logical_plan=None,
                            error=compact_error,
                        )
                        for rank in range(world_size)
                    ]
            scattered = world_group.scatter_object(scatter_inputs, src=0)
            gathered_targets = None
            scatter_inputs = None
        finally:
            clear_owner_source_session = getattr(
                coordinator,
                "clear_owner_source_session",
                None,
            )
            if callable(clear_owner_source_session):
                clear_owner_source_session()

        if not isinstance(scattered, _RankLocalTransferEnvelope):
            raise ValueError("rank-local transfer envelope is invalid")
        if scattered.world_rank != local_rank:
            raise ValueError("rank-local transfer envelope rank differs")
        if scattered.error is not None:
            raise RuntimeError(scattered.error)
        if target_resource is None or target_placement is None or parallel_rank is None:
            raise RuntimeError("local target placement was not built")
        if (
            scattered.parallel_rank != parallel_rank
            or scattered.target_placement_id != target_placement.placement_id
            or scattered.logical_plan is None
        ):
            raise ValueError("rank-local transfer envelope target differs")

        logical_plan = scattered.logical_plan
        if logical_plan.target_placements != (target_placement,):
            raise ValueError("rank-local transfer target placement differs")
        if len(logical_plan.target_executors) != 1:
            raise ValueError("rank-local transfer must have one target executor")
        target_executor = logical_plan.target_executors[0]
        if (
            target_executor.placement_id != target_placement.placement_id
            or target_executor.rank != parallel_rank
        ):
            raise ValueError("rank-local target executor differs")
        source_bindings = tuple(scattered.source_bindings)
        if (
            tuple(
                project_source_bindings(
                    logical_plan.source_placements,
                    source_bindings,
                )
            )
            != source_bindings
        ):
            raise ValueError("rank-local source binding closure differs")

        binding_started = time.perf_counter()
        target_binding = self._coerce_weight_manifest_inventory(
            transfer_resources.enter_context(target_resource.bind()),
            WeightRuntimeBindingManifest,
        )
        load_request = prepare_weight_load_from_plan(
            logical_plan,
            source_bindings=source_bindings,
            target_bindings=(target_binding,),
        )
        phase_seconds["binding"] = time.perf_counter() - binding_started
        load_attestor = _RemoteInstanceWeightLoadAttestor(
            coordinator=coordinator,
            target_resource=target_resource,
            target_binding=target_binding,
        )
        load_preflight = _preflight_bounded_native_weight_transfer(
            transfer_executor,
            load_request,
            attestor=load_attestor,
        )
        return _PreparedNativeHeterogeneousWeightLoad(
            source_placements=logical_plan.source_placements,
            source_bindings=source_bindings,
            target_placement=target_placement,
            target_binding=target_binding,
            load_request=load_request,
            load_attestor=load_attestor,
            load_preflight=load_preflight,
            transfer_executor=transfer_executor,
        )

    def _prepare_native_heterogeneous_weight_load(
        self,
        *,
        model,
        coordinator,
        transfer_resources,
        source_placement_inventories,
        source_binding_inventories,
        target_manifest_builder,
        target_model_id,
        manifest_revision,
        local_session_id,
        transfer_executor,
        phase_seconds,
    ) -> _PreparedNativeHeterogeneousWeightLoad:
        if transfer_executor is None:
            raise RuntimeError(
                "placement_binding_v1 requires a configured weight transfer provider"
            )
        source_started = time.perf_counter()
        if (
            not source_placement_inventories
            or not source_binding_inventories
            or len(source_placement_inventories) != len(source_binding_inventories)
        ):
            raise ValueError(
                "source placement and runtime binding inventories must be paired"
            )
        source_placements = tuple(
            self._coerce_weight_manifest_inventory(
                inventory,
                WeightPlacementManifest,
            )
            for inventory in source_placement_inventories
        )
        source_bindings = tuple(
            self._coerce_weight_manifest_inventory(
                inventory,
                WeightRuntimeBindingManifest,
            )
            for inventory in source_binding_inventories
        )
        phase_seconds["source_manifest"] = time.perf_counter() - source_started
        self._require_manifest_identity(
            source_placements,
            model_id=target_model_id,
            revision=manifest_revision,
            role="source",
        )

        target_started = time.perf_counter()
        target_resource = transfer_resources.enter_context(
            target_manifest_builder(
                model=model,
                model_id=target_model_id,
                revision=manifest_revision,
                instance_id=f"sglang:{local_session_id}",
                endpoint=local_session_id,
            )
        )
        if not hasattr(target_resource, "placement") or not callable(
            getattr(target_resource, "bind", None)
        ):
            raise ValueError(
                "placement_binding_v1 requires a target manifest session with "
                "placement and bind()"
            )
        target_placement = self._coerce_weight_manifest_inventory(
            target_resource.placement,
            WeightPlacementManifest,
        )
        self._require_manifest_identity(
            (target_placement,),
            model_id=target_model_id,
            revision=manifest_revision,
            role="target",
        )
        phase_seconds["target_manifest"] = time.perf_counter() - target_started

        plan_started = time.perf_counter()
        logical_plan = plan_weight_transfer_to_local_target(
            source_placements,
            target_placement,
        )
        phase_seconds["plan"] = time.perf_counter() - plan_started

        binding_started = time.perf_counter()
        target_binding = self._coerce_weight_manifest_inventory(
            transfer_resources.enter_context(target_resource.bind()),
            WeightRuntimeBindingManifest,
        )
        load_request = prepare_weight_load_from_plan(
            logical_plan,
            source_bindings=source_bindings,
            target_bindings=(target_binding,),
        )
        phase_seconds["binding"] = time.perf_counter() - binding_started
        load_attestor = _RemoteInstanceWeightLoadAttestor(
            coordinator=coordinator,
            target_resource=target_resource,
            target_binding=target_binding,
        )
        load_preflight = _preflight_bounded_native_weight_transfer(
            transfer_executor,
            load_request,
            attestor=load_attestor,
        )
        return _PreparedNativeHeterogeneousWeightLoad(
            source_placements=source_placements,
            source_bindings=source_bindings,
            target_placement=target_placement,
            target_binding=target_binding,
            load_request=load_request,
            load_attestor=load_attestor,
            load_preflight=load_preflight,
            transfer_executor=transfer_executor,
        )

    def _prepare_legacy_heterogeneous_weight_load(
        self,
        *,
        model,
        transfer_resources,
        inventories,
        target_manifest_builder,
        target_model_id,
        manifest_revision,
        local_session_id,
        phase_seconds,
    ) -> _PreparedLegacyHeterogeneousWeightLoad:
        source_started = time.perf_counter()
        backend = _load_legacy_mooncake_weight_backend()
        if not inventories:
            raise ValueError("source runtime manifest inventories must not be empty")
        source_manifests = tuple(
            backend.RuntimeManifest.from_runtime_inventory(inventory)
            for inventory in inventories
        )
        phase_seconds["source_manifest"] = time.perf_counter() - source_started
        self._require_manifest_identity(
            source_manifests,
            model_id=target_model_id,
            revision=manifest_revision,
            role="source",
        )

        target_started = time.perf_counter()
        selected_target_builder = self._runtime_v1_target_manifest_builder(
            target_manifest_builder
        )
        target_resource = transfer_resources.enter_context(
            selected_target_builder(
                model=model,
                model_id=target_model_id,
                revision=manifest_revision,
                instance_id=f"sglang:{local_session_id}",
                endpoint=local_session_id,
            )
        )
        target_manifest = backend.RuntimeManifest.from_runtime_inventory(
            target_resource
        )
        self._require_manifest_identity(
            (target_manifest,),
            model_id=target_model_id,
            revision=manifest_revision,
            role="target",
        )
        phase_seconds["target_manifest"] = time.perf_counter() - target_started

        plan_started = time.perf_counter()
        plan = backend.plan_runtime_transfer_to_local_target(
            source_manifests,
            target_manifest,
        )
        phase_seconds["plan"] = time.perf_counter() - plan_started
        source_registrations = tuple(
            backend.MemoryRegistrationLease.from_fragment(
                fragment,
                runtime_lease_id=manifest.lease_id,
            )
            for manifest in source_manifests
            for fragment in manifest.fragments
        )
        target_registrations = tuple(
            backend.MemoryRegistrationLease.from_fragment(
                fragment,
                runtime_lease_id=target_manifest.lease_id,
            )
            for fragment in target_manifest.fragments
        )
        return _PreparedLegacyHeterogeneousWeightLoad(
            plan=plan,
            source_manifests=source_manifests,
            source_registrations=source_registrations,
            target_registrations=target_registrations,
            target_manifest=target_manifest,
            backend=backend,
        )

    @staticmethod
    def _execute_native_heterogeneous_weight_load(
        prepared: _PreparedNativeHeterogeneousWeightLoad,
        phase_seconds: dict[str, float],
        execution_context: WeightTransferExecutionContext,
    ) -> Iterable[Any]:
        load_receipt = execute_weight_load(
            prepared.load_request,
            provider=prepared.transfer_executor,
            target_mode=WeightTargetLoadMode.COLD_START,
            attestor=prepared.load_attestor,
            preflight=prepared.load_preflight,
            execution_context=execution_context,
        )
        provider_phases = dict(load_receipt.provider_phase_seconds)
        phase_seconds["attestation"] += provider_phases.get("attest", 0.0)
        phase_seconds["provider_probe"] = provider_phases.get("probe", 0.0)
        phase_seconds["lowering"] = provider_phases.get("prepare", 0.0)
        phase_seconds["data_transfer"] = provider_phases.get(
            "submit", 0.0
        ) + provider_phases.get("wait", 0.0)
        phase_seconds["provider_synchronize"] = provider_phases.get("synchronize", 0.0)
        phase_seconds["provider_release"] = provider_phases.get("release", 0.0)
        return load_receipt.backend_receipts

    @staticmethod
    def _execute_legacy_heterogeneous_weight_load(
        prepared: _PreparedLegacyHeterogeneousWeightLoad,
        transfer_executor: Any,
        execution_context: WeightTransferExecutionContext,
    ) -> Iterable[Any]:
        if execution_context.expired():
            raise TimeoutError("remote weight transfer deadline expired")
        return transfer_executor.execute(
            prepared.plan,
            prepared.source_manifests,
            prepared.target_manifest,
            source_pre_registered=True,
            source_registrations=prepared.source_registrations,
            target_pre_registered=True,
            target_registrations=prepared.target_registrations,
            execution_context=execution_context,
        )

    def load_model_from_remote_instance_by_transfer_engine_heterogeneous(
        self,
        model,
        transfer_engine,
        seed_url,
        local_session_id,
        target_manifest_builder,
        *,
        target_model_id: str,
        target_artifact_revision: str | None = None,
        target_hf_revision: str | None = None,
        target_revision: str | None = None,
        provider_factory=None,
    ) -> bool:
        target_artifact_revision, target_hf_revision = _resolve_target_weight_revisions(
            target_artifact_revision=target_artifact_revision,
            target_hf_revision=target_hf_revision,
            target_revision=target_revision,
        )
        world_group = get_world_group()
        quarantine_execution_context = WeightTransferExecutionContext(
            deadline_unix_sec=(
                time.time() + _HETEROGENEOUS_QUARANTINE_COORDINATION_TIMEOUT_SEC
            )
        )
        try:
            drain_heterogeneous_weight_transfer_quarantine(
                max_attempts=1,
                timeout_ms=_HETEROGENEOUS_UNKNOWN_DRAIN_TIMEOUT_MS,
                execution_context=quarantine_execution_context,
            )
        except Exception:
            logger.exception("Failed to recover a previous heterogeneous transfer")
        local_quarantined = bool(_HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE)
        world_size = getattr(world_group, "world_size", 1)
        if world_size > 1:
            try:
                quarantine_flags = world_group.all_gather_object(
                    local_quarantined,
                    phase="heterogeneous_quarantine.preflight",
                    execution_context=quarantine_execution_context,
                )
            except Exception:
                logger.exception("Cannot coordinate the target-world quarantine state")
                return False
            if len(quarantine_flags) != world_size or not all(
                type(flag) is bool for flag in quarantine_flags
            ):
                logger.error("Target world returned invalid quarantine state")
                return False
            world_quarantined = any(quarantine_flags)
        else:
            world_quarantined = local_quarantined
        if world_quarantined:
            logger.error(
                "A previous heterogeneous transfer remains completion-unknown; "
                "restart the target process before another remote weight load"
            )
            return False
        server_args = get_server_args()
        if server_args is not None and getattr(server_args, "torchao_config", None):
            logger.error(
                "Heterogeneous remote-instance loading does not support "
                "--torchao-config because source and target layouts differ."
            )
            return False
        started = time.perf_counter()
        phase_seconds = {
            "acquire": 0.0,
            "source_manifest": 0.0,
            "target_manifest": 0.0,
            "plan": 0.0,
            "binding": 0.0,
            "provider_setup": 0.0,
            "attestation": 0.0,
            "provider_probe": 0.0,
            "lowering": 0.0,
            "data_transfer": 0.0,
            "provider_synchronize": 0.0,
            "provider_release": 0.0,
            "transfer": 0.0,
            "synchronize": 0.0,
            "post_load": 0.0,
            "release": 0.0,
        }
        provider_started = time.perf_counter()
        native_provider = None
        legacy_backend = None
        capabilities = None
        provider_preflight_error = None
        if provider_factory is not None:
            if not callable(provider_factory):
                provider_preflight_error = (
                    "configured weight transfer provider factory is not callable"
                )
            else:
                try:
                    native_provider = provider_factory(
                        transfer_engine,
                        max_batch_operations=8192,
                    )
                except Exception as error:
                    provider_preflight_error = _compact_provider_preflight_error(
                        "configured provider factory",
                        error,
                    )
            if provider_preflight_error is None:
                try:
                    capabilities = probe_remote_instance_weight_transfer_capabilities(
                        provider=native_provider,
                    )
                except Exception as error:
                    provider_preflight_error = _compact_provider_preflight_error(
                        "configured provider capability probe",
                        error,
                    )
            if (
                provider_preflight_error is None
                and capabilities is not None
                and not capabilities.native_executor
            ):
                provider_preflight_error = (
                    capabilities.native_contract_error
                    or "configured provider does not satisfy the native provider contract"
                )
        else:
            try:
                legacy_backend = _load_legacy_mooncake_weight_backend()
            except RuntimeError as error:
                legacy_backend = None
                provider_preflight_error = _compact_provider_preflight_error(
                    "legacy backend setup",
                    error,
                )
            if provider_preflight_error is None:
                try:
                    capabilities = probe_remote_instance_weight_transfer_capabilities(
                        legacy_backend=legacy_backend,
                    )
                except Exception as error:
                    provider_preflight_error = _compact_provider_preflight_error(
                        "legacy backend capability probe",
                        error,
                    )
        phase_seconds["provider_setup"] = time.perf_counter() - provider_started
        if (
            provider_preflight_error is None
            and capabilities is not None
            and capabilities.legacy_planner
            and not capabilities.native_executor
            and (
                legacy_backend is None
                or not _legacy_runtime_v1_supports_bounded_execution(legacy_backend)
            )
        ):
            provider_preflight_error = (
                "Mooncake legacy runtime_v1 execution cannot satisfy the bounded "
                "target-world deadline; configure a native weight transfer provider"
            )
        if (
            provider_preflight_error is None
            and capabilities is not None
            and legacy_backend is not None
            and not capabilities.legacy_planner
        ):
            provider_preflight_error = (
                capabilities.legacy_contract_error
                or "legacy backend does not satisfy the runtime_v1 provider contract"
            )
        if provider_preflight_error is None and capabilities is None:
            provider_preflight_error = (
                "provider capability probe returned no capabilities"
            )
        capability_fingerprint = None
        if provider_preflight_error is None:
            capability_fingerprint = (
                "native" if provider_factory is not None else "legacy",
                capabilities.native_executor,
                capabilities.canonical_adapter,
                capabilities.legacy_planner,
            )
        if not _vote_provider_preflight(
            world_group,
            provider_preflight_error,
            capability_fingerprint,
        ):
            return False
        requested_revision_semantics = (
            ARTIFACT_WEIGHT_VERSION_V1
            if capabilities.supports_placement_binding_v1
            else HF_REVISION_V1
        )
        allow_legacy_hf_fallback = _allow_legacy_hf_manifest_revision(
            capabilities,
            target_artifact_revision=target_artifact_revision,
            target_hf_revision=target_hf_revision,
        )
        coordinator = RemoteInstanceWeightTransferWorldCoordinator(
            seed_url,
            world_group,
            capabilities=capabilities,
            manifest_revision_semantics=requested_revision_semantics,
            allow_legacy_hf_fallback=allow_legacy_hf_fallback,
        )
        acquire_started = time.perf_counter()
        try:
            transfer_session = coordinator.acquire()
        except Exception:
            logger.exception("Cannot acquire remote weight transfer session")
            return False
        phase_seconds["acquire"] = time.perf_counter() - acquire_started
        if transfer_session is None:
            logger.error("Cannot acquire remote weight transfer session.")
            return False
        deadline_unix_sec = getattr(transfer_session, "deadline_unix_sec", None)
        if deadline_unix_sec is None:
            coordinator_context = getattr(coordinator, "execution_context", None)
            deadline_unix_sec = getattr(
                coordinator_context,
                "deadline_unix_sec",
                None,
            )
        if deadline_unix_sec is None:
            logger.error(
                "Remote weight transfer session is missing the required "
                "absolute deadline"
            )
            return False
        try:
            execution_context = WeightTransferExecutionContext(
                deadline_unix_sec=deadline_unix_sec,
            )
        except ValueError:
            logger.exception("Target-world transfer session deadline is invalid")
            return False
        inventories = getattr(transfer_session, "manifests", ())
        manifest_format = getattr(
            transfer_session,
            "manifest_format",
            RUNTIME_MANIFEST_V1,
        )
        revision_semantics = getattr(
            transfer_session,
            "manifest_revision_semantics",
            LEGACY_HF_UNATTESTED,
        )
        negotiated_legacy_hf_fallback = getattr(
            transfer_session,
            "allow_legacy_hf_fallback",
            allow_legacy_hf_fallback,
        )
        identity_error = None
        try:
            manifest_revision = _resolve_remote_manifest_revision(
                manifest_format=manifest_format,
                source_revision_semantics=revision_semantics,
                allow_legacy_hf_fallback=negotiated_legacy_hf_fallback,
                target_artifact_revision=target_artifact_revision,
                target_hf_revision=target_hf_revision,
            )
        except (RuntimeError, ValueError) as error:
            identity_error = error
            manifest_revision = target_hf_revision
        transfer_success = False
        release_safe = True
        release_success = False
        receipts = ()
        plan = None
        transfer_executor = native_provider
        prepared: _PreparedHeterogeneousWeightLoad | None = None
        execution_manifest_format = manifest_format
        completion_unknown_errors = (WeightTransferCompletionUnknownError,)
        legacy_transfer_errors = ()
        pending_transfer_id = None
        local_terminal_status = "NO_SUBMISSION"
        finish_completed = False

        def finish_before_target_release() -> None:
            nonlocal finish_completed
            nonlocal release_success
            nonlocal transfer_success
            if finish_completed:
                return
            finish_completed = True
            clear_owner_source_session = getattr(
                coordinator,
                "clear_owner_source_session",
                None,
            )
            if callable(clear_owner_source_session):
                clear_owner_source_session()
            release_started = time.perf_counter()
            try:
                transfer_success, release_success = coordinator.finish(
                    local_success=transfer_success,
                    local_release_safe=release_safe,
                )
            except Exception:
                transfer_success = False
                release_success = False
                logger.exception("Failed to finish remote weight transfer session")
            phase_seconds["release"] = time.perf_counter() - release_started

            world_release_safe = getattr(
                coordinator,
                "world_release_safe",
                release_success and release_safe,
            )
            if world_release_safe and release_success:
                return

            ticket = pending_transfer_id
            if not ticket:
                rank = getattr(world_group, "rank_in_group", 0)
                ticket_kind = (
                    local_terminal_status.lower().replace("_", "-")
                    if local_terminal_status in _HETEROGENEOUS_TERMINAL_STATUSES
                    else "completion-unknown"
                )
                ticket = f"{transfer_session.transfer_id}:{ticket_kind}-rank-{rank}"
            quarantine_resources = transfer_resources.pop_all()
            _HETEROGENEOUS_UNKNOWN_TRANSFER_QUARANTINE.append(
                _HeterogeneousUnknownTransferQuarantine(
                    source_transfer_id=transfer_session.transfer_id,
                    pending_transfer_id=ticket,
                    transfer_executor=transfer_executor,
                    resources=quarantine_resources,
                    coordinator=coordinator,
                    owners=(
                        model,
                        transfer_engine,
                        transfer_executor,
                        transfer_session,
                        prepared,
                    ),
                    terminal_status=local_terminal_status,
                )
            )
            logger.critical(
                "Target world did not prove terminal completion for source "
                "transfer %s; target resources and source lease remain "
                "quarantined",
                transfer_session.transfer_id,
            )

        try:
            with ExitStack() as transfer_resources:
                planning_error = None
                try:
                    if identity_error is not None:
                        raise identity_error
                    if manifest_format == PLACEMENT_BINDING_V1:
                        if not capabilities.native_executor:
                            raise RuntimeError(
                                "placement_binding_v1 requires the native "
                                "weight transfer executor"
                            )
                        prepared = (
                            self._prepare_distributed_native_heterogeneous_weight_load(
                                model=model,
                                coordinator=coordinator,
                                world_group=world_group,
                                transfer_resources=transfer_resources,
                                target_manifest_builder=target_manifest_builder,
                                target_model_id=target_model_id,
                                manifest_revision=manifest_revision,
                                local_session_id=local_session_id,
                                transfer_executor=transfer_executor,
                                phase_seconds=phase_seconds,
                            )
                        )
                    elif manifest_format == RUNTIME_MANIFEST_V1:
                        if (
                            capabilities.native_executor
                            and capabilities.canonical_adapter
                        ):
                            (
                                adapted_source_placements,
                                adapted_source_bindings,
                            ) = self._adapt_runtime_v1_source_inventories(inventories)
                            prepared = self._prepare_native_heterogeneous_weight_load(
                                model=model,
                                coordinator=coordinator,
                                transfer_resources=transfer_resources,
                                source_placement_inventories=(
                                    adapted_source_placements
                                ),
                                source_binding_inventories=(adapted_source_bindings),
                                target_manifest_builder=target_manifest_builder,
                                target_model_id=target_model_id,
                                manifest_revision=manifest_revision,
                                local_session_id=local_session_id,
                                transfer_executor=transfer_executor,
                                phase_seconds=phase_seconds,
                            )
                            execution_manifest_format = PLACEMENT_BINDING_V1
                        elif capabilities.legacy_planner:
                            prepared = self._prepare_legacy_heterogeneous_weight_load(
                                model=model,
                                transfer_resources=transfer_resources,
                                inventories=inventories,
                                target_manifest_builder=target_manifest_builder,
                                target_model_id=target_model_id,
                                manifest_revision=manifest_revision,
                                local_session_id=local_session_id,
                                phase_seconds=phase_seconds,
                            )
                        else:
                            raise RuntimeError(
                                "runtime_v1 requires either the SGLang canonical "
                                "adapter with native executor or the explicit "
                                "Mooncake legacy planner API"
                            )
                    else:
                        raise ValueError(
                            f"unsupported source manifest format: {manifest_format}"
                        )
                    if isinstance(prepared, _PreparedNativeHeterogeneousWeightLoad):
                        plan = prepared.load_request.plan
                        transfer_executor = prepared.transfer_executor
                    else:
                        plan = prepared.plan
                        completion_unknown_errors = (
                            WeightTransferCompletionUnknownError,
                            prepared.backend.TransferCompletionUnknownError,
                        )
                        legacy_transfer_errors = (prepared.backend.TransferEngineError,)
                except BaseException as error:
                    planning_error = error

                transfer_resources.callback(finish_before_target_release)
                world_ready = coordinator.ready_for_transfer(planning_error is None)
                if planning_error is not None:
                    raise planning_error
                if not world_ready:
                    raise RuntimeError(
                        "target world is not ready for heterogeneous weight transfer"
                    )

                transfer_started = time.perf_counter()
                release_safe = False
                local_terminal_status = None
                try:
                    if isinstance(prepared, _PreparedNativeHeterogeneousWeightLoad):
                        receipts = self._execute_native_heterogeneous_weight_load(
                            prepared,
                            phase_seconds,
                            execution_context,
                        )
                    elif isinstance(prepared, _PreparedLegacyHeterogeneousWeightLoad):
                        transfer_executor = (
                            prepared.backend.MooncakeTransferEngineReader(
                                transfer_engine,
                                max_batch_operations=8192,
                            )
                        )
                        receipts = self._execute_legacy_heterogeneous_weight_load(
                            prepared,
                            transfer_executor,
                            execution_context,
                        )
                    else:
                        raise RuntimeError(
                            "heterogeneous weight load execution was not prepared"
                        )
                except completion_unknown_errors as error:
                    pending_transfer_id = getattr(
                        error,
                        "completion_ticket",
                        None,
                    ) or getattr(error, "pending_transfer_id", None)
                    if not pending_transfer_id:
                        logger.critical(
                            "Completion-unknown transfer has no drain ticket; "
                            "resources remain quarantined for process lifetime"
                        )
                        raise
                    logger.error(
                        "Transfer completion is unknown; retaining the target model, "
                        "runtime binding, registrations, and source lease while "
                        "draining %s",
                        pending_transfer_id,
                    )
                    pending_interrupt = None
                    for _ in range(_HETEROGENEOUS_UNKNOWN_DRAIN_MAX_ATTEMPTS):
                        try:
                            drain_completion = getattr(
                                transfer_executor,
                                "drain_completion",
                                None,
                            )
                            if callable(drain_completion):
                                terminal_status = drain_completion(
                                    pending_transfer_id,
                                    timeout_ms=(
                                        _HETEROGENEOUS_UNKNOWN_DRAIN_TIMEOUT_MS
                                    ),
                                )
                            else:
                                terminal_status = (
                                    transfer_executor.drain_pending_transfer(
                                        pending_transfer_id,
                                        timeout_ms=(
                                            _HETEROGENEOUS_UNKNOWN_DRAIN_TIMEOUT_MS
                                        ),
                                    )
                                )
                        except BaseException as drain_error:
                            if isinstance(drain_error, Exception):
                                logger.exception(
                                    "Failed to query pending transfer %s; target "
                                    "resources remain quarantined",
                                    pending_transfer_id,
                                )
                                time.sleep(1)
                            else:
                                if pending_interrupt is None:
                                    pending_interrupt = drain_error
                                logger.warning(
                                    "Deferring process interruption until pending "
                                    "transfer %s reaches a terminal state",
                                    pending_transfer_id,
                                )
                            continue
                        status_name = _transfer_completion_status_name(terminal_status)
                        if status_name == "COMPLETION_UNKNOWN":
                            continue
                        if status_name not in _HETEROGENEOUS_TERMINAL_STATUSES:
                            logger.error(
                                "Pending transfer %s returned invalid "
                                "completion status %r",
                                pending_transfer_id,
                                terminal_status,
                            )
                            continue
                        local_terminal_status = status_name
                        release_safe = True
                        if pending_interrupt is not None:
                            raise pending_interrupt
                        terminal_error_type = (
                            prepared.backend.TransferEngineError
                            if isinstance(
                                prepared,
                                _PreparedLegacyHeterogeneousWeightLoad,
                            )
                            else RuntimeError
                        )
                        raise terminal_error_type(
                            "heterogeneous transfer became terminal after an "
                            f"unknown completion state: {status_name}"
                        ) from error
                    logger.critical(
                        "Pending transfer %s did not reach a terminal state after "
                        "%d drain attempts; the target-world finish barrier will "
                        "retain resources and the source lease",
                        pending_transfer_id,
                        _HETEROGENEOUS_UNKNOWN_DRAIN_MAX_ATTEMPTS,
                    )
                    if pending_interrupt is not None:
                        raise pending_interrupt
                    raise
                except WeightTransferError as error:
                    release_safe = error.completion_known
                    if release_safe:
                        local_terminal_status = "FAILED_DRAINED"
                    raise
                except legacy_transfer_errors:
                    release_safe = True
                    local_terminal_status = "FAILED_DRAINED"
                    raise
                except Exception:
                    release_safe = False
                    raise
                release_safe = True
                local_terminal_status = "COMPLETED"
                phase_seconds["transfer"] = time.perf_counter() - transfer_started
                if isinstance(prepared, _PreparedLegacyHeterogeneousWeightLoad):
                    phase_seconds["data_transfer"] = phase_seconds["transfer"]
                synchronize_started = time.perf_counter()
                current_platform.synchronize()
                phase_seconds["synchronize"] = time.perf_counter() - synchronize_started
                post_load_started = time.perf_counter()
                _post_load_weights(model)
                phase_seconds["post_load"] = time.perf_counter() - post_load_started
                transfer_success = True
        except completion_unknown_errors:
            release_safe = False
            logger.exception(
                "Heterogeneous remote-instance transfer completion remains unknown; "
                "target resources and source lease must remain quarantined"
            )
        except WeightTransferError as error:
            release_safe = error.completion_known
            if release_safe:
                logger.exception(
                    "Heterogeneous remote-instance weight loading failed with "
                    "a known terminal completion state"
                )
            else:
                logger.exception(
                    "Heterogeneous remote-instance weight loading failed without "
                    "a terminal completion proof; resources remain quarantined"
                )
        except legacy_transfer_errors:
            release_safe = True
            logger.exception(
                "Heterogeneous remote-instance Transfer Engine loading failed with "
                "a known terminal completion state"
            )
        except Exception:
            logger.exception("Heterogeneous remote-instance weight loading failed")
        finally:
            if not finish_completed:
                finish_before_target_release()

        if not transfer_success:
            logger.error(
                "Heterogeneous remote-instance loading phases before failure: %s",
                ", ".join(
                    f"{name}={seconds:.4f}s" for name, seconds in phase_seconds.items()
                ),
            )
            return False
        if not release_success:
            logger.error(
                "Loaded weights but failed to release source transfer session %s; "
                "failing remote-instance startup because source mutation remains "
                "blocked until explicit release or recovery",
                transfer_session.transfer_id,
            )
            return False

        try:
            target_rank = int(get_parallel().world_rank)
        except (AssertionError, AttributeError, RuntimeError):
            target_rank = int(getattr(world_group, "rank_in_group", 0))
        logger.info(
            "Loaded heterogeneous remote-instance weights: "
            "manifest_format=%s, transfer_id=%s, target_rank=%d, "
            "release_success=true, bytes=%d, "
            "compact_operations=%d, segments=%d, elapsed=%.4fs; "
            "phases: %s",
            execution_manifest_format,
            transfer_session.transfer_id,
            target_rank,
            sum(receipt.nbytes for receipt in receipts),
            len(plan.operations),
            sum(receipt.operation_count for receipt in receipts),
            time.perf_counter() - started,
            ", ".join(
                f"{name}={seconds:.4f}s" for name, seconds in phase_seconds.items()
            ),
        )
        return True


class RemoteModelLoader(BaseModelLoader):
    """Model loader that can load Tensors from remote database."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # TODO @DellCurry: move to s3 connector only
        set_runai_streamer_env(load_config)

    def _get_weights_iterator_kv(
        self,
        client,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights from remote storage."""
        assert get_connector_type(client) == ConnectorType.KV
        rank = get_parallel().tp_rank
        return client.weight_iterator(rank)

    def _get_weights_iterator_fs(
        self,
        client,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights from remote storage."""
        assert get_connector_type(client) == ConnectorType.FS
        return client.weight_iterator()

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    @staticmethod
    def save_model(
        model: torch.nn.Module,
        model_path: str,
        url: str,
    ) -> None:
        with create_remote_connector(url) as client:
            assert get_connector_type(client) == ConnectorType.KV
            model_name = parse_model_name(url)
            rank = get_parallel().tp_rank
            state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
            for key, tensor in state_dict.items():
                r_key = f"{model_name}/keys/rank_{rank}/{key}"
                client.set(r_key, tensor)

            for root, _, files in os.walk(model_path):
                for file_name in files:
                    # ignore hidden files
                    if file_name.startswith("."):
                        continue
                    if os.path.splitext(file_name)[1] in (".json", ".py"):
                        file_path = os.path.join(root, file_name)
                        with open(file_path, encoding="utf-8") as file:
                            file_content = file.read()
                            f_key = f"{model_name}/files/{file_name}"
                            client.setstr(f_key, file_content)

    def _load_model_from_remote_kv(
        self, model: nn.Module, model_config: ModelConfig, client
    ):
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)
        weights_iterator = self._get_weights_iterator_kv(client)
        state_dict = ShardedStateLoader._filter_subtensors(model.state_dict())
        for key, tensor in weights_iterator:
            # If loading with LoRA enabled, additional padding may
            # be added to certain parameters. We only load into a
            # narrowed view of the parameter data.
            param_data = state_dict[key].data
            param_shape = state_dict[key].shape
            for dim, size in enumerate(tensor.shape):
                if size < param_shape[dim]:
                    param_data = param_data.narrow(dim, 0, size)
            if tensor.shape != param_shape:
                logger.warning(
                    "loading tensor of shape %s into parameter '%s' of shape %s",
                    tensor.shape,
                    key,
                    param_shape,
                )
            param_data.copy_(tensor)
            state_dict.pop(key)
        if state_dict:
            raise ValueError(f"Missing keys {tuple(state_dict)} in loaded state!")

        _post_load_weights(model)

    def _load_model_from_remote_fs(
        self, model, client, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            model.load_weights(self._get_weights_iterator_fs(client))

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(module, target_device):
                        quant_method.process_weights_after_loading(module)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:
        logger.info("Loading weights from remote storage ...")
        start = time.perf_counter()
        load_config = self.load_config

        assert load_config.load_format == LoadFormat.REMOTE, (
            f"Model loader {self.load_config.load_format} is not supported for "
            f"load format {load_config.load_format}"
        )

        model_weights = model_config.model_path
        if hasattr(model_config, "model_weights"):
            model_weights = model_config.model_weights

        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with torch.device(device_config.device):
                model = _initialize_model(model_config, self.load_config, quant_config)

            with create_remote_connector(
                model_weights, device=device_config.device
            ) as client:
                connector_type = get_connector_type(client)
                if connector_type == ConnectorType.KV:
                    self._load_model_from_remote_kv(model, model_config, client)
                elif connector_type == ConnectorType.FS:
                    self._load_model_from_remote_fs(
                        model, client, model_config, device_config
                    )

        end = time.perf_counter()
        logger.info("Loaded weights from remote storage in %.2f seconds.", end - start)
        return model.eval()


def load_model_with_cpu_quantization(
    self,
    *,
    model_config: ModelConfig,
    device_config: DeviceConfig,
) -> nn.Module:
    target_device = torch.device(device_config.device)
    quant_config = _get_quantization_config(model_config, self.load_config)
    with set_default_torch_dtype(model_config.dtype):
        model = _initialize_model(
            model_config,
            self.load_config,
            quant_config,
        )

        if not isinstance(self, DummyModelLoader):
            model.load_weights(self._get_all_weights(model_config, model))

        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                # When quant methods need to process weights after loading
                # (for repacking, quantizing, etc), they expect parameters
                # to be on the global target device. This scope is for the
                # case where cpu offloading is used, where we will move the
                # parameters onto device for processing and back off after.
                with device_loading_context(module, target_device):
                    quant_method.process_weights_after_loading(module)

        model.to(target_device)

    return model.eval()


class IncModelLoader(DefaultModelLoader):
    """
    Model loader that applies Intel AutoRound quantization
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        logger.info("IncModelLoader: Loading model...")

        # Check if model is already quantized
        if model_config._is_already_quantized():
            logger.info("Model is already quantized, loading directly...")
            # Use default loading for pre-quantized models
            return super().load_model(
                model_config=model_config, device_config=device_config
            )

        quant_model = self._autoround_quantization_workflow(model_config, device_config)

        target_device = torch.device(device_config.device)

        # Return autoround model for offline quantization mode
        if self.load_config.inc_save_path is not None:
            quant_model.to(target_device)
            return quant_model.eval()

        model_config.hf_config = quant_model.config
        quant_config = _get_quantization_config(model_config, self.load_config)

        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            self.load_weights_and_postprocess(
                model, iter(quant_model.state_dict().items()), target_device
            )
        return model.eval()

    def _parse_quantization(self, quantization: str):
        """Map quantization to AutoRound's scheme and format."""
        AR_QUANT_CFG_CHOICES = {
            "auto-round-int8": ("INT8", "llm_compressor"),
        }
        quant_cfg = AR_QUANT_CFG_CHOICES.get(quantization)
        if not quant_cfg:
            raise ValueError(
                f"Invalid quantization choice: '{quantization}'. "
                f"Available choices: {list(AR_QUANT_CFG_CHOICES.keys())}"
            )
        return quant_cfg

    def _autoround_quantization_workflow(
        self, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        """Auto-round quantization workflow: quantize, save checkpoint, then return model."""
        try:
            from auto_round import AutoRound
        except ImportError:
            logger.error(
                "auto-round library not found. "
                "Please install it using `pip install auto-round` to use AutoRound quantization."
            )
            raise

        scheme, format = self._parse_quantization(model_config.quantization)

        try:
            autoround = AutoRound(
                model_config.model_path,
                scheme=scheme,
                iters=self.load_config.inc_tuning_iters,
                disable_opt_rtn=self.load_config.inc_disable_opt_rtn,
                low_cpu_mem_usage=False,
            )
            if self.load_config.inc_save_path is not None:
                logger.info("Offline quantization mode: Will quantize and save")
                model, _ = autoround.quantize_and_save(
                    output_dir=self.load_config.inc_save_path, format=format
                )
                return model
            else:
                logger.info("Online quantization mode: Will quantize and skip saving")
                # Use a temporary directory and discard it so nothing is persisted in online mode.
                with tempfile.TemporaryDirectory() as tmp_save_dir:
                    model, _ = autoround.quantize_and_save(
                        output_dir=tmp_save_dir, format=format
                    )
                return model
        except Exception as e:
            raise ValueError(f"AutoRound quantization failed: {e}")


class ModelOptModelLoader(DefaultModelLoader):
    """
    Model loader that applies NVIDIA Model Optimizer quantization
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        # Any ModelOpt specific initialization if needed

    def _setup_modelopt_quantization(
        self,
        model,
        tokenizer,
        quant_cfg,
        quantized_ckpt_restore_path: str | None = None,
        quantized_ckpt_save_path: str | None = None,
        export_path: str | None = None,
    ) -> None:
        """
        Set up ModelOpt quantization for the given model.

        Args:
            model: The model to quantize
            tokenizer: The tokenizer associated with the model
            quant_cfg: The quantization configuration
            quantized_ckpt_restore_path: Path to restore quantized checkpoint from
            quantized_ckpt_save_path: Path to save quantized checkpoint to
            export_path: Path to export the quantized model in HuggingFace format

        Raises:
            ImportError: If ModelOpt is not available
            Exception: If quantization setup fails
        """
        try:
            import modelopt.torch.opt as mto
            import modelopt.torch.quantization as mtq
            from modelopt.torch.quantization.utils import is_quantized
        except ImportError as e:
            raise ImportError(
                "ModelOpt is not available. Please install modelopt."
            ) from e

        if is_quantized(model):
            rank0_log("Model is already quantized, skipping quantization setup.")
            return
        # Restore from checkpoint if provided
        if quantized_ckpt_restore_path:
            try:
                mto.restore(model, quantized_ckpt_restore_path)
                rank0_log(
                    f"Restored quantized model from {quantized_ckpt_restore_path}"
                )

                # Export model if path provided (even when restoring from checkpoint)
                self._maybe_export_modelopt(model, export_path)
                return
            except Exception as e:
                logger.warning(
                    f"Failed to restore from {quantized_ckpt_restore_path}: {e}"
                )
                rank0_log("Proceeding with calibration-based quantization...")

        # Set up calibration-based quantization
        try:
            # Left padding tends to work better for batched generation with decoder-only LMs
            with suppress(Exception):
                tokenizer.padding_side = "left"

            from modelopt.torch.utils.dataset_utils import (
                create_forward_loop,
                get_dataset_dataloader,
            )

            # Create calibration dataloader
            calib_dataloader = get_dataset_dataloader(
                dataset_name="cnn_dailymail",  # TODO: Consider making this configurable
                tokenizer=tokenizer,
                batch_size=36,  # TODO: Consider making this configurable
                num_samples=512,  # TODO: Consider making this configurable
                device=model.device,
                include_labels=False,
            )

            calibrate_loop = create_forward_loop(dataloader=calib_dataloader)

            # Apply quantization
            mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)

            if not model_parallel_is_initialized() or get_parallel().tp_rank == 0:
                mtq.print_quant_summary(model)

            # Save checkpoint if path provided
            if quantized_ckpt_save_path:
                try:
                    mto.save(model, quantized_ckpt_save_path)
                    rank0_log(f"Quantized model saved to {quantized_ckpt_save_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to save quantized checkpoint to {quantized_ckpt_save_path}: {e}"
                    )

            # Export model if path provided
            self._maybe_export_modelopt(model, export_path)

        except Exception as e:
            raise Exception(f"Failed to set up ModelOpt quantization: {e}") from e

    def _maybe_export_modelopt(self, model, export_path: str | None) -> None:
        """Export model to HuggingFace format if export_path is provided."""
        if export_path:
            try:
                # Get the original model path from the model config
                original_model_path = getattr(self, "_original_model_path", None)
                self._export_modelopt_checkpoint(
                    model, export_path, original_model_path
                )
                rank0_log(
                    f"Quantized model exported to HuggingFace format at {export_path}"
                )
            except Exception as e:
                rank0_log(
                    f"Warning: Failed to export quantized model to {export_path}: {e}"
                )

    def _export_modelopt_checkpoint(
        self,
        model,
        export_path: str,
        model_path: str = None,
        trust_remote_code: bool = True,
    ) -> None:
        """
        Export the quantized model to HuggingFace format using ModelOpt export API.

        Args:
            model: The quantized model to export
            export_path: Directory path to export the model to
            model_path: Path to the original model (for tokenizer export)
            trust_remote_code: Whether to trust remote code for tokenizer loading

        Raises:
            ImportError: If ModelOpt export functionality is not available
            Exception: If export fails
        """
        try:
            from modelopt.torch.export import export_hf_checkpoint
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "ModelOpt export functionality is not available. "
                "Please ensure you have the latest version of modelopt installed."
            ) from e

        # Create export directory if it doesn't exist
        os.makedirs(export_path, exist_ok=True)

        # Export the quantized model
        export_hf_checkpoint(model, export_dir=export_path)

        # Export the tokenizer if model_path is provided
        if model_path:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code
                )
                tokenizer.save_pretrained(export_path)
                rank0_log(f"Tokenizer exported to {export_path}")
            except Exception as e:
                rank0_log(f"Warning: Failed to export tokenizer: {e}")

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        logger.info("ModelOptModelLoader: Loading base model...")

        # Store the original model path for tokenizer export
        self._original_model_path = model_config.model_path

        # Check if model is already quantized
        if model_config._is_already_quantized():
            logger.info("Model is already quantized, loading directly...")
            # Use default loading for pre-quantized models
            return super().load_model(
                model_config=model_config, device_config=device_config
            )

        # TODO: Quantize-and-serve mode has been disabled at the ModelConfig level
        # All quantization now uses the standard workflow (quantize + export/save)
        logger.info("Standard quantization mode: Will quantize and export/save")
        return self._standard_quantization_workflow(model_config, device_config)

    def _standard_quantization_workflow(
        self, model_config: ModelConfig, device_config: DeviceConfig
    ) -> nn.Module:
        """Standard quantization workflow: quantize, save checkpoint, export, then return model."""
        # Use shared method from parent class to load base model for quantization
        model = self._load_modelopt_base_model(model_config)

        # Import ModelOpt modules
        try:
            import modelopt.torch.quantization as mtq
        except ImportError:
            logger.error(
                "NVIDIA Model Optimizer (modelopt) library not found. "
                "Please install it to use ModelOpt quantization."
            )
            raise

        # Handle both old modelopt_quant and new unified quantization flags
        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Legacy modelopt_quant flag
            quant_choice_str = model_config.modelopt_quant
        else:
            # Unified quantization flag - extract the type (fp8/fp4)
            quant_choice_str = model_config._get_modelopt_quant_type()

        quant_cfg_name = QUANT_CFG_CHOICES.get(quant_choice_str)
        if not quant_cfg_name:
            raise ValueError(
                f"Invalid quantization choice: '{quant_choice_str}'. "
                f"Available choices: {list(QUANT_CFG_CHOICES.keys())}"
            )

        try:
            # getattr will fetch the config object, e.g., mtq.FP8_DEFAULT_CFG
            quant_cfg = getattr(mtq, quant_cfg_name)
        except AttributeError:
            raise AttributeError(
                f"ModelOpt quantization config '{quant_cfg_name}' not found. "
                "Please verify the ModelOpt library installation."
            )

        logger.info(
            f"Quantizing model with ModelOpt using config: mtq.{quant_cfg_name}"
        )

        # Get ModelOpt configuration from LoadConfig
        modelopt_config = self.load_config.modelopt_config
        quantized_ckpt_restore_path = (
            modelopt_config.checkpoint_restore_path if modelopt_config else None
        )
        quantized_ckpt_save_path = (
            modelopt_config.checkpoint_save_path if modelopt_config else None
        )
        export_path = modelopt_config.export_path if modelopt_config else None
        tokenizer = AutoTokenizer.from_pretrained(
            model_config.model_path, use_fast=True
        )

        try:
            self._setup_modelopt_quantization(
                model,
                tokenizer,
                quant_cfg,
                quantized_ckpt_restore_path=quantized_ckpt_restore_path,
                quantized_ckpt_save_path=quantized_ckpt_save_path,
                export_path=export_path,
            )
        except Exception as e:
            logger.warning(f"ModelOpt quantization failed: {e}")
            rank0_log("Proceeding without quantization...")

        return model.eval()


class RunaiModelStreamerLoader(BaseModelLoader):
    """
    Model loader that uses Runai Model Streamer to load a model.

    Supports fast model loading from SSDs, shared filesystems and object storage (S3, GCS, Azure blob) with weight streaming.

    Configuration (via load_config.model_loader_extra_config):
        - distributed (bool): Enable distributed streaming - True by default for url paths (object storage)
        - concurrency (int): Number of concurrent downloads
        - memory_limit (int): Memory limit for streaming buffer

    Note: Metadata files must be pre-downloaded via
    ObjectStorageModel.download_and_get_path() before instantiation.
    """

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        revision: Optional[str]
        """The optional model revision."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        model_config: Optional[ModelConfig] = None
        """The model configuration (for checking architecture, etc)."""

        @classmethod
        def init_new(cls, model_config: ModelConfig, model):
            model_weights = model_config.model_path
            if hasattr(model_config, "model_weights"):
                model_weights = model_config.model_weights
            return cls(
                model_weights,
                model_config.revision,
                prefix="",
                fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                model_config=model_config,
            )

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra_config = load_config.model_loader_extra_config
        allowed_keys = {"distributed", "concurrency", "memory_limit"}
        unexpected_keys = set(extra_config.keys()) - allowed_keys

        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format "
                f"{load_config.load_format}: "
                f"{unexpected_keys}"
            )

        set_runai_streamer_env(load_config)

        self._is_distributed = None
        if load_config.model_loader_extra_config:
            extra_config = load_config.model_loader_extra_config

            if "distributed" in extra_config and isinstance(
                extra_config.get("distributed"), bool
            ):
                self._is_distributed = extra_config.get("distributed")

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str]
    ) -> Tuple[str, List[str]]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        from sglang.srt.utils.runai_utils import is_runai_obj_uri, list_safetensors

        is_object_storage_path = is_runai_obj_uri(model_name_or_path)
        if self._is_distributed is None:
            self._is_distributed = is_object_storage_path
        is_local = os.path.isdir(model_name_or_path)
        safetensors_pattern = "*.safetensors"
        index_file = SAFE_WEIGHTS_INDEX_NAME

        hf_folder = (
            model_name_or_path
            if (is_local or is_object_storage_path)
            else download_weights_from_hf(
                model_name_or_path,
                self.load_config.download_dir,
                [safetensors_pattern],
                revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        )

        server_args = get_server_args()
        if server_args and server_args.model_checksum is not None:
            from sglang.srt.utils.model_file_verifier import verify

            checksums_source = server_args.model_checksum or model_name_or_path
            verify(model_path=hf_folder, checksums_source=checksums_source)

        hf_weights_files = list_safetensors(path=hf_folder)

        # For models like Mistral-7B-Instruct-v0.3
        # there are both sharded safetensors files and a consolidated
        # safetensors file. Using both breaks.
        # Here, we download the `model.safetensors.index.json` and filter
        # any files not found in the index.
        if not is_local and not is_object_storage_path:
            download_safetensors_index_file_from_hf(
                model_name_or_path,
                index_file,
                self.load_config.download_dir,
                revision,
            )
        hf_weights_files = filter_duplicate_safetensors_files(
            hf_weights_files, hf_folder, index_file
        )

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files

    def _get_weights_iterator(
        self, source: Source
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        from sglang.srt.model_loader.weight_utils import (
            runai_safetensors_weights_iterator,
        )

        hf_folder, hf_weights_files = self._prepare_weights(
            source.model_or_path, source.revision
        )

        if source.model_config is not None:
            hf_weights_files = maybe_add_mtp_safetensors(
                hf_weights_files,
                hf_folder,
                "model.safetensors.index.json",
                source.model_config.hf_config,
            )

        weights_iterator = runai_safetensors_weights_iterator(
            hf_weights_files, self._is_distributed, self.target_device_str
        )

        if self.load_config.draft_model_idx is not None:
            import re

            def filter_weights(original_weights_iterator):
                pattern = r"model.mtp.layers.(\d+)."
                for name, tensor in original_weights_iterator:
                    group = re.match(pattern, name)
                    if group is not None:
                        idx = int(group.group(1))
                        if idx != self.load_config.draft_model_idx:
                            continue
                        new_name = name.replace(group.group(), "model.mtp.layers.0.")
                    else:
                        new_name = name
                    yield (new_name, tensor)

            weights_iterator = filter_weights(weights_iterator)

        def apply_prefix(original_weights_iterator):
            yield from (
                (source.prefix + name, tensor)
                for (name, tensor) in original_weights_iterator
            )

        return apply_prefix(weights_iterator)

    def _get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:

        primary_weights = RunaiModelStreamerLoader.Source.init_new(model_config, model)
        yield from self._get_weights_iterator(primary_weights)

        secondary_weights = cast(
            Iterable[RunaiModelStreamerLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config.model_path, model_config.revision)

    def load_model(
        self,
        *,
        model_config: ModelConfig,
        device_config: DeviceConfig,
    ) -> nn.Module:

        if hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant:
            # Load base model using shared method
            raise NotImplementedError(
                "Runai Model Streamer Loader does not support ModelOpt quantization yet"
            )

        assert device_config.device_type in ("cuda", "cpu"), (
            f"Runai Model Streamer only supports CUDA and CPU, "
            f"got {device_config.device_type}"
        )

        if device_config.device_type == "cuda":
            self.target_device_str = (
                device_config.device_type + ":" + str(device_config.gpu_id)
            )
        else:
            self.target_device_str = "cpu"

        target_device = torch.device(device_config.device)
        quant_config = _get_quantization_config(model_config, self.load_config)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = _initialize_model(
                    model_config,
                    self.load_config,
                    quant_config,
                )

            DefaultModelLoader.load_weights_and_postprocess(
                model, self._get_all_weights(model_config, model), target_device
            )

        return model.eval()


def get_model_loader(
    load_config: LoadConfig, model_config: Optional[ModelConfig] = None
) -> BaseModelLoader:
    """Get a model loader based on the load format."""

    if load_config.load_format == LoadFormat.DUMMY:
        return DummyModelLoader(load_config)

    if model_config and model_config.quantization in ["auto-round-int8"]:
        logger.info("Using IncModelLoader due to AutoRound quantization config.")
        return IncModelLoader(load_config)

    # ModelOptModelLoader's local-copy quantize-and-export workflow doesn't apply
    # to non-local loaders. These loaders own their weight transport path and still
    # initialize the model with ModelOpt quantization config where applicable.
    model_optloader_allowed = model_config and load_config.load_format not in (
        LoadFormat.RUNAI_STREAMER,
        LoadFormat.REMOTE_INSTANCE,
        LoadFormat.WEIGHT_SNAPSHOT,
    )

    if model_optloader_allowed and (
        (hasattr(model_config, "modelopt_quant") and model_config.modelopt_quant)
        or model_config.quantization
        in ["modelopt_fp8", "modelopt_fp4", "modelopt_mixed", "modelopt"]
    ):
        logger.info("Using ModelOptModelLoader due to ModelOpt quantization config.")
        return ModelOptModelLoader(load_config)

    # Use ModelOptModelLoader for unified quantization flags
    if (
        model_optloader_allowed
        and hasattr(model_config, "quantization")
        and model_config.quantization
        in ["modelopt_fp8", "modelopt_fp4", "modelopt_mixed"]
    ):
        if model_config._is_already_quantized():
            logger.info(
                f"Using ModelOptModelLoader for pre-quantized model: {model_config.quantization}"
            )
        else:
            logger.info(
                f"Using ModelOptModelLoader for quantization: {model_config.quantization}"
            )
        return ModelOptModelLoader(load_config)

    if isinstance(load_config.load_format, type):
        return load_config.load_format(load_config)

    if load_config.load_format == LoadFormat.SHARDED_STATE:
        return ShardedStateLoader(load_config)

    if load_config.load_format == LoadFormat.PRESHARDED:
        return PreshardedModelLoader(load_config)

    if load_config.load_format == LoadFormat.BITSANDBYTES:
        return BitsAndBytesModelLoader(load_config)

    if load_config.load_format == LoadFormat.GGUF:
        return GGUFModelLoader(load_config)

    if load_config.load_format == LoadFormat.LAYERED:
        return LayeredModelLoader(load_config)

    # Check for FLASH_RL format early
    # FP8 approach: BF16/FP16 model with native FP8 quantization
    if load_config.load_format == LoadFormat.FLASH_RL:
        logger.info(
            "Using QuantizedRLModelLoader for RL training with native FP8 quantization."
        )
        logger.info(
            "FP8 approach: Model loads with native SGLang FP8 quantization. "
            "Same model path for both training and inference."
        )

        # Set quantization to FP8 for native SGLang support
        if model_config and not model_config.quantization:
            logger.info(
                "QuantizedRL: Setting quantization to fp8 (native SGLang support). "
                "Model will be loaded with FP8 infrastructure"
            )
            model_config.quantization = "fp8"

        return QuantizedRLModelLoader(load_config)

    if load_config.load_format == LoadFormat.REMOTE:
        return RemoteModelLoader(load_config)

    if load_config.load_format == LoadFormat.REMOTE_INSTANCE:
        return RemoteInstanceModelLoader(load_config)

    if load_config.load_format == LoadFormat.WEIGHT_SNAPSHOT:
        return WeightSnapshotModelLoader(load_config)

    if load_config.load_format == LoadFormat.PRIVATE:
        import importlib

        try:
            module = importlib.import_module("sglang.private.private_model_loader")
            return module.PrivateModelLoader(load_config)
        except ImportError:
            raise ValueError("Failed to import sglang.private.private_model_loader")

    if load_config.load_format == LoadFormat.RUNAI_STREAMER:
        return RunaiModelStreamerLoader(load_config)

    if load_config.load_format == LoadFormat.IPC_CACHE:
        from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
        from sglang.srt.weight_cache.protocol import (
            compute_global_rank,
            get_socket_path,
        )

        if load_config.weight_cache_socket:
            socket_path = load_config.weight_cache_socket
        else:
            from sglang.srt.runtime_context import get_parallel

            ps = get_parallel()
            global_rank = compute_global_rank(ps.tp_size, ps.pp_rank, ps.tp_rank)
            socket_path = get_socket_path(global_rank=global_rank)
        return IpcModelLoader(
            load_config=load_config,
            socket_path=socket_path,
            weight_cache_mode=load_config.weight_cache_mode,
            fallback_load_format=load_config.fallback_load_format,
        )

    return DefaultModelLoader(load_config)
