from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import fastapi
from sglang.srt.managers.communicator import (
    FanOutCancelledBeforeDispatch,
    FanOutCommunicator,
    FanOutCompletionUnknownError,
    FanOutDeadlineExpiredBeforeDispatch,
)
from sglang.srt.managers.io_struct import (
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BeginRemoteInstanceWeightTransferReqInput,
    BeginRemoteInstanceWeightTransferReqOutput,
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    CommitWeightMaterializationReqInput,
    CommitWeightMaterializationReqOutput,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetRemoteInstanceWeightTransferSessionReqInput,
    GetRemoteInstanceWeightTransferSessionReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    LoRAUpdateOutput,
    MaterializeWeightsReqInput,
    OpenSessionReqInput,
    PauseGenerationReqInput,
    PrepareWeightMaterializationReqInput,
    PrepareWeightMaterializationReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ReleaseRemoteInstanceWeightTransferReqInput,
    ReleaseRemoteInstanceWeightTransferReqOutput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    RenewRemoteInstanceWeightTransferReqInput,
    RenewRemoteInstanceWeightTransferReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    ScaleElasticEPReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    SlowDownReqInput,
    SlowDownReqOutput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
    WeightMaterializationSessionState,
    WeightSnapshotActivationReqInput,
    WeightSnapshotActivationReqOutput,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.managers.weight_materialization import (
    is_published_materialization_state,
    reduce_materialization_states,
)
from sglang.srt.model_executor.weight_runtime_manifest import (
    DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC,
    validate_remote_instance_weight_transfer_lease_timeout,
)
from sglang.srt.server_args import LoRARef, ServerArgs
from sglang.srt.utils import (
    get_bool_env_var,
    normalize_serialized_named_tensor_payloads,
)
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.srt.weight_transfer.remote_protocol import (
    ARTIFACT_WEIGHT_VERSION_V1,
    HF_REVISION_V1,
    PLACEMENT_BINDING_V1,
    RUNTIME_MANIFEST_V1,
    validate_manifest_revision_semantics,
)
from sglang.utils import TypeBasedDispatcher

if TYPE_CHECKING:
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class RemoteInstanceWeightTransferBeginError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        transfer_id: str,
        session_state: str,
    ) -> None:
        super().__init__(message)
        self.transfer_id = transfer_id
        self.session_state = session_state


class RemoteInstanceWeightTransferControlError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        transfer_id: str,
        session_state: str,
        lease_fence: str | None,
        generation: int | None,
    ) -> None:
        super().__init__(message)
        self.transfer_id = transfer_id
        self.session_state = session_state
        self.lease_fence = lease_fence
        self.generation = generation
        self.completion_unknown = True
        self.cleanup_pending = session_state == "cleanup_pending"
        self.retryable = False
        self.reconcile_required = True


class WeightMaterializationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        materialization_id: str,
        session_state: WeightMaterializationSessionState | str,
        completion_ticket: str | None = None,
    ) -> None:
        super().__init__(message)
        self.materialization_id = materialization_id
        self.session_state = WeightMaterializationSessionState(session_state)
        self.completion_ticket = completion_ticket


_REMOTE_WEIGHT_TRANSFER_SESSION_INDEX_LIMIT = 4096
_CONTROL_CLEANUP_TIMEOUT_SEC = 60
_ADMIN_PAUSE_OWNER = "admin"
_REMOTE_WEIGHT_TRANSFER_PAUSE_OWNER_PREFIX = "remote-weight-transfer:"
_REMOTE_WEIGHT_TRANSFER_BEGIN_FENCE_PREFIX = "begin-v1:"


def _remote_weight_transfer_session_index(manager) -> Dict[str, Dict[str, Any]]:
    # Discovery only; scheduler leases remain authoritative for release and updates.
    index = getattr(manager, "_remote_weight_transfer_session_index", None)
    if index is None:
        index = {}
        setattr(manager, "_remote_weight_transfer_session_index", index)
    return index


def _remote_weight_transfer_lease_identity(payloads) -> Tuple[List[str], int | None]:
    lease_ids = sorted(
        {
            payload["lease_id"]
            for payload in payloads
            if isinstance(payload, dict) and payload.get("lease_id")
        }
    )
    generations = {
        payload["generation"]
        for payload in payloads
        if isinstance(payload, dict) and payload.get("generation") is not None
    }
    generation = next(iter(generations)) if len(generations) == 1 else None
    return lease_ids, generation


def _remote_weight_transfer_result_payloads(results, manifest_format: str) -> List:
    field = "bindings" if manifest_format == PLACEMENT_BINDING_V1 else "manifests"
    return [
        payload
        for result in results
        for payload in (getattr(result, field, None) or ())
    ]


def _remote_weight_transfer_created_by_request(results) -> bool:
    states = [getattr(result, "session_state", "unknown") for result in (results or ())]
    if any(state in {"conflict", "expired"} for state in states):
        return False
    return any(
        (getattr(result, "success", False) and state != "reused")
        or state in {"created", "cleanup_pending"}
        for result, state in zip(results, states)
    )


def _remote_weight_transfer_begin_lock(manager) -> asyncio.Lock:
    lock = getattr(manager, "_remote_weight_transfer_begin_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(manager, "_remote_weight_transfer_begin_lock", lock)
    return lock


def _remember_remote_weight_transfer_session(
    manager,
    *,
    transfer_id: str,
    manifest_format: str,
    manifest_revision_semantics: str,
    deadline_unix_sec: float | None,
    payloads,
    session_state: str,
    lease_fence: str | None = None,
) -> Dict[str, Any]:
    index = _remote_weight_transfer_session_index(manager)
    existing = index.get(transfer_id, {})
    lease_ids, generation = _remote_weight_transfer_lease_identity(payloads)
    lease_ids = lease_ids or list(existing.get("lease_ids", ()))
    record = {
        "transfer_id": transfer_id,
        "lease_id": lease_ids[0] if len(lease_ids) == 1 else None,
        "lease_ids": lease_ids,
        "generation": (
            generation if generation is not None else existing.get("generation")
        ),
        "manifest_format": manifest_format,
        "manifest_revision_semantics": manifest_revision_semantics,
        "deadline_unix_sec": deadline_unix_sec,
        "expired": session_state == "expired",
        "session_state": session_state,
        "last_release_attempt_unix_sec": existing.get("last_release_attempt_unix_sec"),
        "last_release_success": existing.get("last_release_success"),
        "last_release_message": existing.get("last_release_message"),
    }
    current_fence = lease_fence or existing.get("lease_fence")
    if current_fence is not None:
        record["lease_fence"] = current_fence
    index[transfer_id] = record
    while len(index) > _REMOTE_WEIGHT_TRANSFER_SESSION_INDEX_LIMIT:
        released_id = next(
            (
                item_id
                for item_id, item in index.items()
                if item["session_state"] == "released"
            ),
            None,
        )
        if released_id is None:
            break
        index.pop(released_id)
    return record


def _resolve_remote_weight_transfer_control_identity(
    manager,
    transfer_reference: str,
) -> tuple[str, Dict[str, Any] | None]:
    if type(transfer_reference) is not str or not transfer_reference:
        raise ValueError("transfer_id must be a non-empty string")
    index = _remote_weight_transfer_session_index(manager)
    direct = _refresh_remote_weight_transfer_session(manager, transfer_reference)
    if direct is not None:
        return transfer_reference, direct
    matches = [
        transfer_id
        for transfer_id, record in index.items()
        if record.get("lease_fence") == transfer_reference
    ]
    if len(matches) > 1:
        raise RuntimeError("remote weight transfer lease fence is ambiguous")
    if matches:
        transfer_id = matches[0]
        return transfer_id, _refresh_remote_weight_transfer_session(
            manager, transfer_id
        )
    return transfer_reference, None


def _resolve_unfenced_remote_weight_transfer_control(
    manager,
    transfer_reference: str,
) -> str:
    transfer_id, session = _resolve_remote_weight_transfer_control_identity(
        manager,
        transfer_reference,
    )
    if session is not None and session.get("lease_fence") is not None:
        raise ValueError(
            "lease_fence and generation are required for a fenced "
            "remote weight transfer"
        )
    return transfer_id


def _validate_remote_weight_transfer_control_identity(
    lease_fence: str | None,
    generation: int | None,
) -> None:
    if (lease_fence is None) != (generation is None):
        raise ValueError("lease_fence and generation must be provided together")
    if lease_fence is not None and (type(lease_fence) is not str or not lease_fence):
        raise ValueError("lease_fence must be a non-empty string")
    if generation is not None and (type(generation) is not int or generation <= 0):
        raise ValueError("generation must be a positive integer")


def _refresh_remote_weight_transfer_session(
    manager, transfer_id: str
) -> Dict[str, Any] | None:
    index = _remote_weight_transfer_session_index(manager)
    current = index.get(transfer_id)
    if current is None:
        return None
    record = dict(current)
    deadline = record.get("deadline_unix_sec")
    if (
        record["session_state"] != "released"
        and deadline is not None
        and deadline <= time.time()
    ):
        record["expired"] = True
        record["session_state"] = "expired"
        index[transfer_id] = record
    return dict(record)


def _record_remote_weight_transfer_release(
    manager,
    *,
    transfer_id: str,
    attempted_at: float,
    success: bool,
    message: str,
    completion_unknown: bool = False,
    lease_fence: str | None = None,
    generation: int | None = None,
) -> None:
    index = _remote_weight_transfer_session_index(manager)
    record = _refresh_remote_weight_transfer_session(manager, transfer_id)
    if record is None:
        record = {
            "transfer_id": transfer_id,
            "lease_id": None,
            "lease_ids": [],
            "generation": None,
            "manifest_format": None,
            "deadline_unix_sec": None,
            "expired": False,
            "session_state": "release_failed",
        }
    record["last_release_attempt_unix_sec"] = attempted_at
    record["last_release_success"] = success
    record["last_release_message"] = message
    if lease_fence is not None:
        record["lease_fence"] = lease_fence
    if generation is not None:
        record["generation"] = generation
    if success:
        record["session_state"] = "released"
    elif completion_unknown:
        record["session_state"] = "cleanup_pending"
        record["completion_unknown"] = True
        record["reconcile_required"] = True
    index[transfer_id] = record


def _record_remote_weight_transfer_completion_unknown(
    manager,
    *,
    transfer_id: str,
    session_state: str,
    lease_fence: str | None,
    generation: int | None,
) -> None:
    index = _remote_weight_transfer_session_index(manager)
    record = _refresh_remote_weight_transfer_session(manager, transfer_id)
    if record is None:
        record = {
            "transfer_id": transfer_id,
            "lease_id": None,
            "lease_ids": [],
            "generation": None,
            "manifest_format": None,
            "deadline_unix_sec": None,
            "expired": False,
            "last_release_attempt_unix_sec": None,
            "last_release_success": None,
            "last_release_message": None,
        }
    record["session_state"] = session_state
    record["completion_unknown"] = True
    record["reconcile_required"] = True
    if lease_fence is not None:
        record["lease_fence"] = lease_fence
    if generation is not None:
        record["generation"] = generation
    index[transfer_id] = record


def _record_remote_weight_transfer_renewal(
    manager,
    *,
    transfer_id: str,
    deadline_unix_sec: float,
) -> None:
    index = _remote_weight_transfer_session_index(manager)
    record = _refresh_remote_weight_transfer_session(manager, transfer_id)
    if record is None:
        record = {
            "transfer_id": transfer_id,
            "lease_id": None,
            "lease_ids": [],
            "generation": None,
            "manifest_format": None,
            "last_release_attempt_unix_sec": None,
            "last_release_success": None,
            "last_release_message": None,
        }
    record["deadline_unix_sec"] = deadline_unix_sec
    record["expired"] = False
    record["session_state"] = "active"
    index[transfer_id] = record


async def _finish_control_task(task: asyncio.Task):
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def _fan_out_may_have_dispatched(error: BaseException) -> bool:
    return not isinstance(
        error,
        (
            FanOutCancelledBeforeDispatch,
            FanOutDeadlineExpiredBeforeDispatch,
        ),
    )


async def _release_uncertain_remote_weight_transfer(
    manager,
    *,
    transfer_id: str,
    lease_fence: str | None = None,
    generation: int | None = None,
    attempts: int = 3,
) -> bool:
    if lease_fence is None or generation is None:
        return False
    for _ in range(attempts):
        try:
            (
                success,
                _,
            ) = await TokenizerControlMixin.release_remote_instance_weight_transfer(
                manager,
                transfer_id,
                lease_fence=lease_fence,
                generation=generation,
            )
        except Exception:
            logger.exception(
                "Failed to clean up uncertain remote weight transfer %s",
                transfer_id,
            )
            success = False
        if success:
            return True
    return False


def _weight_materialization_active_ids(manager) -> set[str]:
    active_ids = getattr(manager, "_weight_materialization_active_ids", None)
    if active_ids is None:
        active_ids = set()
        manager._weight_materialization_active_ids = active_ids
    return active_ids


def _serving_weight_revision(manager) -> str:
    revision = getattr(manager, "runtime_weight_revision", None)
    if type(revision) is not str or not revision:
        raise RuntimeError("the serving weight revision is not initialized")
    return revision


def _hf_model_revision(manager) -> str:
    revision = getattr(manager.server_args, "revision", None) or "default"
    if type(revision) is not str or not revision:
        raise RuntimeError("the Hugging Face model revision is invalid")
    return revision


def _require_weight_snapshot_export_allowed(manager) -> None:
    if getattr(manager, "weight_update_fail_closed", False):
        raise RuntimeError(
            "weight snapshot export is disabled after an incomplete weight update"
        )


def _validate_next_weight_revision(manager, revision: str | None) -> None:
    if not manager.server_args.enable_weight_runtime_manifest:
        return
    if type(revision) is not str or not revision:
        raise ValueError(
            "online weight updates require weight_version when runtime manifests "
            "are enabled"
        )
    if revision == _serving_weight_revision(manager):
        raise ValueError("weight_version must identify a new weight artifact")


def _record_weight_update_safety(
    manager,
    results,
    *,
    full_restore: bool,
) -> None:
    results = tuple(results)
    if any(getattr(result, "fail_closed", False) for result in results):
        manager.weight_update_fail_closed = True
    elif full_restore and results and all(result.success for result in results):
        manager.weight_update_fail_closed = False


def _finish_weight_update_transaction(
    manager,
    results,
    *,
    weight_version: str | None,
    full_restore: bool,
) -> tuple[bool, str]:
    _record_weight_update_safety(manager, results, full_restore=full_restore)
    success, message = FanOutCommunicator.merge_results(results)
    if success and weight_version is not None:
        manager._update_weight_version_if_provided(weight_version)
        message += f" Weight version updated to {weight_version}."
    return success, message


async def _call_weight_update_communicator(manager, communicator, request):
    if hasattr(request, "request_id"):
        request.request_id = uuid.uuid4().hex
    try:
        return await communicator(request)
    except (
        FanOutCancelledBeforeDispatch,
        FanOutDeadlineExpiredBeforeDispatch,
    ):
        raise
    except BaseException:
        manager.weight_update_fail_closed = True
        raise


def _weight_materialization_fan_out(manager) -> int:
    communicator = manager.prepare_weight_materialization_communicator
    fan_out = getattr(communicator, "_fan_out", None)
    if type(fan_out) is int and fan_out > 0:
        return fan_out
    return int(getattr(manager.server_args, "dp_size", 1))


def _weight_materialization_result_state(
    results,
    default: WeightMaterializationSessionState | str,
) -> WeightMaterializationSessionState:
    return reduce_materialization_states(
        (getattr(result, "session_state", None) for result in results),
        default=default,
        completion_unknown=any(
            getattr(result, "completion_unknown", False) for result in results
        ),
    )


def _ordered_weight_materialization_results(
    results,
    *,
    materialization_id: str,
    expected_fan_out: int,
    phase: str,
    allow_identical_duplicates: bool,
):
    if not results:
        raise WeightMaterializationError(
            f"{phase} returned no responses",
            materialization_id=materialization_id,
            session_state=WeightMaterializationSessionState.FAILED,
        )

    by_rank = {}
    for result in results:
        if result.materialization_id != materialization_id:
            raise WeightMaterializationError(
                f"{phase} returned a mismatched materialization_id",
                materialization_id=materialization_id,
                session_state=WeightMaterializationSessionState.CONFLICT,
            )
        rank = result.external_dp_rank
        if type(rank) is not int or rank < 0 or rank >= expected_fan_out:
            raise WeightMaterializationError(
                f"{phase} returned invalid external_dp_rank {rank!r}",
                materialization_id=materialization_id,
                session_state=WeightMaterializationSessionState.CONFLICT,
            )
        by_rank.setdefault(rank, []).append(result)

    expected_ranks = set(range(expected_fan_out))
    if set(by_rank) != expected_ranks:
        raise WeightMaterializationError(
            f"{phase} responses do not cover external DP ranks "
            f"{sorted(expected_ranks)}",
            materialization_id=materialization_id,
            session_state=WeightMaterializationSessionState.CONFLICT,
        )

    ordered = []
    for rank in sorted(by_rank):
        rank_results = by_rank[rank]
        if len(rank_results) > 1:
            if not allow_identical_duplicates or any(
                result != rank_results[0] for result in rank_results[1:]
            ):
                raise WeightMaterializationError(
                    f"{phase} returned duplicate responses for external DP rank {rank}",
                    materialization_id=materialization_id,
                    session_state=WeightMaterializationSessionState.CONFLICT,
                )
        ordered.append(rank_results[0])
    return ordered


async def _cleanup_weight_materialization(
    manager,
    *,
    materialization_id: str,
    storage_options: Dict[str, Any],
) -> WeightMaterializationSessionState:
    deadline_unix_sec = time.time() + _CONTROL_CLEANUP_TIMEOUT_SEC
    request = CommitWeightMaterializationReqInput(
        materialization_id=materialization_id,
        request_id=uuid.uuid4().hex,
        selected_external_dp_rank=None,
        storage_options=storage_options,
        phase="cleanup",
        deadline_unix_sec=deadline_unix_sec,
    )
    cleanup_task = asyncio.create_task(
        manager.commit_weight_materialization_communicator(
            request,
            deadline_unix_sec=deadline_unix_sec,
        )
    )
    try:
        results = await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
        results = await _finish_control_task(cleanup_task)
    except Exception:
        logger.exception(
            "Failed to clean up weight materialization %s",
            materialization_id,
        )
        return WeightMaterializationSessionState.CLEANUP_PENDING

    if not results or any(not result.success for result in results):
        logger.error(
            "Weight materialization cleanup did not complete for %s: %s",
            materialization_id,
            " | ".join(getattr(result, "message", "") for result in results),
        )
        return _weight_materialization_result_state(
            results,
            WeightMaterializationSessionState.CLEANUP_PENDING,
        )
    return _weight_materialization_result_state(
        results,
        WeightMaterializationSessionState.CLEANUP_PENDING,
    )


_RUNTIME_TENSOR_SEMANTIC_FIELDS = (
    "tensor_id",
    "aliases",
    "global_shape",
    "global_offset",
    "local_shape",
    "dtype",
    "itemsize",
    "partition_dim",
    "shard_dims",
    "layer_id",
    "expert_id",
    "layout_fingerprint",
    "nbytes",
    "byte_offset",
    "stride",
    "storage_offset",
    "device",
    "is_contiguous",
)


def _freeze_runtime_manifest_value(value):
    if isinstance(value, dict):
        return tuple(
            sorted(
                (key, _freeze_runtime_manifest_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_runtime_manifest_value(item) for item in value)
    return value


def _runtime_manifest_group_signature(manifests) -> tuple:
    signatures = []
    for manifest in manifests:
        tensors = manifest.get("tensors") or ()
        tensor_signatures = []
        for tensor in tensors:
            rank = tensor.get("rank") or {}
            tensor_signatures.append(
                (
                    tuple(
                        _freeze_runtime_manifest_value(tensor.get(field))
                        for field in _RUNTIME_TENSOR_SEMANTIC_FIELDS
                    ),
                    tuple(rank.get(axis) for axis in ("tp", "pp", "ep")),
                )
            )
        signatures.append(
            (
                manifest.get("model_id"),
                manifest.get("revision"),
                manifest.get("generation"),
                manifest.get("format_version"),
                tuple(sorted(tensor_signatures)),
            )
        )
    return tuple(sorted(signatures))


def _merge_runtime_manifest_groups(groups) -> list:
    if not groups or any(not group for group in groups):
        raise RuntimeError("source workers returned no runtime manifests")
    expected_signature = _runtime_manifest_group_signature(groups[0])
    if any(
        _runtime_manifest_group_signature(group) != expected_signature
        for group in groups[1:]
    ):
        raise RuntimeError(
            "source DP replicas returned semantically inconsistent runtime manifests"
        )

    manifests = [manifest for group in groups for manifest in group]
    worker_ids = []
    for manifest in manifests:
        manifest_worker_ids = {
            tensor.get("worker_id") for tensor in manifest.get("tensors") or ()
        }
        if None in manifest_worker_ids or len(manifest_worker_ids) != 1:
            raise RuntimeError(
                "each source runtime manifest must describe exactly one worker"
            )
        worker_ids.append(next(iter(manifest_worker_ids)))
    if len(set(worker_ids)) != len(worker_ids):
        raise RuntimeError("source runtime manifest worker IDs are not unique")
    return manifests


def _merge_placement_binding_groups(
    placement_groups, binding_groups
) -> tuple[list, list]:
    if (
        not placement_groups
        or not binding_groups
        or len(placement_groups) != len(binding_groups)
        or any(not group for group in placement_groups)
        or any(not group for group in binding_groups)
    ):
        raise RuntimeError("source workers returned no placement/binding manifests")
    if any(
        len(placements) != len(bindings)
        for placements, bindings in zip(placement_groups, binding_groups)
    ):
        raise RuntimeError("source placement and binding counts do not match")

    expected_signature = _runtime_manifest_group_signature(placement_groups[0])
    if any(
        _runtime_manifest_group_signature(group) != expected_signature
        for group in placement_groups[1:]
    ):
        raise RuntimeError(
            "source DP replicas returned semantically inconsistent placements"
        )

    placements = [placement for group in placement_groups for placement in group]
    bindings = [binding for group in binding_groups for binding in group]
    worker_ids = []
    generations = set()
    for placement, binding in zip(placements, bindings):
        placement_identity = (
            placement.get("model_id"),
            placement.get("revision"),
            placement.get("placement_id"),
        )
        binding_identity = (
            binding.get("model_id"),
            binding.get("revision"),
            binding.get("placement_id"),
        )
        if placement_identity != binding_identity:
            raise RuntimeError("source runtime binding does not match its placement")

        placement_records = placement.get("tensors") or ()
        binding_records = binding.get("fragments") or ()
        placement_fragment_ids = [
            tensor.get("placement_fragment_id") for tensor in placement_records
        ]
        binding_fragment_ids = [
            fragment.get("placement_fragment_id") for fragment in binding_records
        ]
        if len(placement_fragment_ids) != len(set(placement_fragment_ids)):
            raise RuntimeError("source placement has duplicate placement fragment IDs")
        if len(binding_fragment_ids) != len(set(binding_fragment_ids)):
            raise RuntimeError(
                "source runtime binding has duplicate placement fragment IDs"
            )
        placement_fragments = {
            tensor.get("placement_fragment_id"): tensor.get("nbytes")
            for tensor in placement_records
        }
        binding_fragments = {
            fragment.get("placement_fragment_id"): fragment.get("nbytes")
            for fragment in binding_records
        }
        if (
            None in placement_fragments
            or None in binding_fragments
            or placement_fragments != binding_fragments
        ):
            raise RuntimeError(
                "source runtime binding fragments do not match placement"
            )

        fragment_worker_ids = {
            fragment.get("worker_id") for fragment in binding.get("fragments") or ()
        }
        if None in fragment_worker_ids or len(fragment_worker_ids) != 1:
            raise RuntimeError(
                "each source runtime binding must describe exactly one worker"
            )
        worker_ids.append(next(iter(fragment_worker_ids)))
        generations.add(binding.get("generation"))

    if len(set(worker_ids)) != len(worker_ids):
        raise RuntimeError("source runtime binding worker IDs are not unique")
    if None in generations or len(generations) != 1:
        raise RuntimeError(
            "source placement and bindings do not describe one model generation"
        )
    return placements, bindings


# Declarative spec: (attr_name_prefix, response_type[, mode, correlation_attr])
# Each entry creates self.{prefix}_communicator and registers
# response_type -> communicator.handle_recv in the dispatch table.
_COMMUNICATOR_SPECS = [
    ("init_weights_update_group", InitWeightsUpdateGroupReqOutput),
    ("destroy_weights_update_group", DestroyWeightsUpdateGroupReqOutput),
    (
        "update_weights_from_distributed",
        UpdateWeightsFromDistributedReqOutput,
        "queueing",
        "request_id",
        "responder_id",
    ),
    (
        "init_weights_send_group_for_remote_instance",
        InitWeightsSendGroupForRemoteInstanceReqOutput,
    ),
    ("send_weights_to_remote_instance", SendWeightsToRemoteInstanceReqOutput),
    (
        "begin_remote_instance_weight_transfer",
        BeginRemoteInstanceWeightTransferReqOutput,
        "queueing",
        "request_id",
        "external_dp_rank",
    ),
    (
        "get_remote_instance_weight_transfer_session",
        GetRemoteInstanceWeightTransferSessionReqOutput,
        "queueing",
        "request_id",
        "external_dp_rank",
    ),
    (
        "release_remote_instance_weight_transfer",
        ReleaseRemoteInstanceWeightTransferReqOutput,
        "queueing",
        "request_id",
        "external_dp_rank",
    ),
    (
        "renew_remote_instance_weight_transfer",
        RenewRemoteInstanceWeightTransferReqOutput,
        "queueing",
        "request_id",
        "external_dp_rank",
    ),
    (
        "prepare_weight_materialization",
        PrepareWeightMaterializationReqOutput,
        "queueing",
        "request_id",
        "external_dp_rank",
    ),
    (
        "commit_weight_materialization",
        CommitWeightMaterializationReqOutput,
        "queueing",
        "request_id",
        "external_dp_rank",
    ),
    (
        "weight_snapshot_activation",
        WeightSnapshotActivationReqOutput,
        "queueing",
        "request_id",
        "responder_id",
    ),
    (
        "update_weights_from_tensor",
        UpdateWeightsFromTensorReqOutput,
        "queueing",
        "request_id",
        "responder_id",
    ),
    (
        "update_weights_from_ipc",
        UpdateWeightsFromIPCReqOutput,
        "queueing",
        "request_id",
        "responder_id",
    ),
    ("get_weights_by_name", GetWeightsByNameReqOutput),
    ("release_memory_occupation", ReleaseMemoryOccupationReqOutput),
    ("resume_memory_occupation", ResumeMemoryOccupationReqOutput),
    ("check_weights", CheckWeightsReqOutput),
    ("slow_down", SlowDownReqOutput),
    ("flush_cache", FlushCacheReqOutput),
    ("add_external_corpus", AddExternalCorpusReqOutput),
    ("remove_external_corpus", RemoveExternalCorpusReqOutput),
    ("list_external_corpora", ListExternalCorporaReqOutput),
    ("clear_hicache_storage", ClearHiCacheReqOutput),
    ("attach_hicache_storage", AttachHiCacheStorageReqOutput),
    ("detach_hicache_storage", DetachHiCacheStorageReqOutput),
    ("profile", ProfileReqOutput),
    ("get_internal_state", GetInternalStateReqOutput),
    ("set_internal_state", SetInternalStateReqOutput),
    ("expert_distribution", ExpertDistributionReqOutput),
    ("update_lora_adapter", LoRAUpdateOutput),
    ("dumper_control", DumperControlReqOutput),
    ("scale_elastic_ep", ScaleElasticEPReqOutput),
]


class TokenizerControlMixin:
    """Mixin for TokenizerManager's control-plane operations (weights, cache, lora,
    profile, internal state, etc.) -- everything that talks to the scheduler via
    FanOutCommunicator, as opposed to data-plane inference requests multiplexed by rid.
    """

    def _require_single_tokenizer_weight_update_owner(
        self: TokenizerManager,
    ) -> None:
        if getattr(self.server_args, "tokenizer_worker_num", 1) != 1:
            raise fastapi.HTTPException(
                status_code=409,
                detail=(
                    "online weight updates require a single tokenizer worker; "
                    "restart with --tokenizer-worker-num 1"
                ),
            )

    def _get_generation_pause_transition_lock(self: TokenizerManager) -> asyncio.Lock:
        lock = getattr(self, "_generation_pause_transition_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._generation_pause_transition_lock = lock
        return lock

    def _get_generation_pause_owners(self: TokenizerManager) -> set[str]:
        owners = getattr(self, "_generation_pause_owners", None)
        if owners is None:
            owners = set()
            if getattr(self, "is_pause", False):
                owners.add(_ADMIN_PAUSE_OWNER)
            self._generation_pause_owners = owners
        return owners

    def _get_generation_pause_resume_pending(
        self: TokenizerManager,
    ) -> set[str]:
        owners = getattr(self, "_generation_pause_resume_pending", None)
        if owners is None:
            owners = set()
            self._generation_pause_resume_pending = owners
        return owners

    def _get_generation_pause_unconfirmed(
        self: TokenizerManager,
    ) -> set[str]:
        owners = getattr(self, "_generation_pause_unconfirmed", None)
        if owners is None:
            owners = set()
            self._generation_pause_unconfirmed = owners
        return owners

    def _get_generation_continue_unconfirmed(
        self: TokenizerManager,
    ) -> set[str]:
        owners = getattr(self, "_generation_continue_unconfirmed", None)
        if owners is None:
            owners = set()
            self._generation_continue_unconfirmed = owners
        return owners

    def _migrate_generation_pause_resume_pending(
        self: TokenizerManager,
    ) -> tuple[set[str], set[str]]:
        legacy = TokenizerControlMixin._get_generation_pause_resume_pending(self)
        pause_unconfirmed = TokenizerControlMixin._get_generation_pause_unconfirmed(
            self
        )
        continue_unconfirmed = (
            TokenizerControlMixin._get_generation_continue_unconfirmed(self)
        )
        latest = getattr(self, "_latest_pause_transitions", {})
        committed = getattr(self, "_committed_pause_transitions", {})

        for pending in (pause_unconfirmed, continue_unconfirmed):
            for owner in tuple(pending):
                identity = latest.get(owner)
                if (
                    identity is not None
                    and committed.get(identity.transition_id) == identity
                ):
                    pending.discard(owner)

        for owner in tuple(legacy):
            identity = latest.get(owner)
            if (
                identity is not None
                and committed.get(identity.transition_id) == identity
            ):
                legacy.discard(owner)
                continue
            if identity is not None and identity.action == "continue":
                continue_unconfirmed.add(owner)
            else:
                pause_unconfirmed.add(owner)
            legacy.discard(owner)
        return pause_unconfirmed, continue_unconfirmed

    async def _acquire_generation_pause(
        self: TokenizerManager,
        owner: str,
        obj: PauseGenerationReqInput,
    ) -> None:
        async with TokenizerControlMixin._get_generation_pause_transition_lock(self):
            owners = TokenizerControlMixin._get_generation_pause_owners(self)
            pause_unconfirmed, continue_unconfirmed = (
                TokenizerControlMixin._migrate_generation_pause_resume_pending(self)
            )
            was_owner = owner in owners
            should_dispatch = (
                not owners
                or owner == _ADMIN_PAUSE_OWNER
                or bool(pause_unconfirmed)
                or bool(continue_unconfirmed)
            )
            async with self.is_pause_cond:
                owners.add(owner)
                self.is_pause = True

            if not should_dispatch:
                return

            obj.rid = owner
            pause_impl = getattr(self, "_pause_generation_impl", None)
            if pause_impl is None:
                pause_impl = self.pause_generation
            try:
                await pause_impl(obj)
            except BaseException as error:
                async with self.is_pause_cond:
                    if _fan_out_may_have_dispatched(error):
                        pause_unconfirmed.add(owner)
                        continue_unconfirmed.discard(owner)
                    elif not was_owner:
                        owners.discard(owner)
                        pause_unconfirmed.discard(owner)
                        continue_unconfirmed.discard(owner)
                    self.is_pause = bool(owners)
                    if not self.is_pause:
                        self.is_pause_cond.notify_all()
                raise
            pause_unconfirmed.clear()
            continue_unconfirmed.clear()

    async def _release_generation_pause(
        self: TokenizerManager,
        owner: str,
        obj: ContinueGenerationReqInput,
    ) -> None:
        async with TokenizerControlMixin._get_generation_pause_transition_lock(self):
            owners = TokenizerControlMixin._get_generation_pause_owners(self)
            pause_unconfirmed, continue_unconfirmed = (
                TokenizerControlMixin._migrate_generation_pause_resume_pending(self)
            )
            if owner not in owners:
                pause_unconfirmed.discard(owner)
                continue_unconfirmed.discard(owner)
                return

            async with self.is_pause_cond:
                if owners - {owner}:
                    owners.remove(owner)
                    pause_unconfirmed.discard(owner)
                    continue_unconfirmed.discard(owner)
                    self.is_pause = True
                    return

            obj.rid = owner
            continue_impl = getattr(self, "_continue_generation_impl", None)
            if continue_impl is None:
                continue_impl = self.continue_generation
            try:
                await continue_impl(obj)
            except BaseException as error:
                async with self.is_pause_cond:
                    if _fan_out_may_have_dispatched(error):
                        pause_unconfirmed.discard(owner)
                        continue_unconfirmed.add(owner)
                    else:
                        continue_unconfirmed.discard(owner)
                    self.is_pause = True
                raise

            async with self.is_pause_cond:
                owners.remove(owner)
                pause_unconfirmed.clear()
                continue_unconfirmed.clear()
                self.is_pause = bool(owners)
                if not self.is_pause:
                    self.is_pause_cond.notify_all()

    @asynccontextmanager
    async def _remote_instance_weight_transfer_pause(self: TokenizerManager):
        lock = getattr(self, "_remote_weight_transfer_pause_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._remote_weight_transfer_pause_lock = lock

        async with lock:
            release_pause = getattr(self, "_release_generation_pause", None)
            if release_pause is None:

                async def release_pause(owner, obj):
                    return await TokenizerControlMixin._release_generation_pause(
                        self,
                        owner,
                        obj,
                    )

            acquire_pause = getattr(self, "_acquire_generation_pause", None)
            if acquire_pause is None:

                async def acquire_pause(owner, obj):
                    return await TokenizerControlMixin._acquire_generation_pause(
                        self,
                        owner,
                        obj,
                    )

            pause_unconfirmed, continue_unconfirmed = (
                TokenizerControlMixin._migrate_generation_pause_resume_pending(self)
            )
            owners = TokenizerControlMixin._get_generation_pause_owners(self)
            unconfirmed = pause_unconfirmed | continue_unconfirmed
            for stale_owner in unconfirmed - owners:
                pause_unconfirmed.discard(stale_owner)
                continue_unconfirmed.discard(stale_owner)

            remote_owners = sorted(
                owner
                for owner in owners
                if owner.startswith(_REMOTE_WEIGHT_TRANSFER_PAUSE_OWNER_PREFIX)
            )
            if remote_owners:
                owner = remote_owners[0]
                for pending_owner in remote_owners[1:]:
                    await release_pause(
                        pending_owner,
                        ContinueGenerationReqInput(torch_empty_cache=False),
                    )
                if owner in pause_unconfirmed or owner in continue_unconfirmed:
                    await acquire_pause(
                        owner,
                        PauseGenerationReqInput(mode="in_place"),
                    )
            else:
                owner = (
                    f"{_REMOTE_WEIGHT_TRANSFER_PAUSE_OWNER_PREFIX}{uuid.uuid4().hex}"
                )
                await acquire_pause(
                    owner,
                    PauseGenerationReqInput(mode="in_place"),
                )
            try:
                yield
            finally:
                resume_task = asyncio.create_task(
                    release_pause(
                        owner,
                        ContinueGenerationReqInput(torch_empty_cache=False),
                    )
                )
                await _finish_control_task(resume_task)

    def init_communicators(self: TokenizerManager, server_args: ServerArgs):
        dispatch_pairs = []
        for spec in _COMMUNICATOR_SPECS:
            name, resp_type = spec[0], spec[1]
            mode = spec[2] if len(spec) > 2 else "queueing"
            correlation_attr = spec[3] if len(spec) > 3 else None
            responder_attr = spec[4] if len(spec) > 4 else None
            comm = FanOutCommunicator(
                self._dispatch_to_scheduler,
                server_args.dp_size,
                mode,
                correlation_attr,
                responder_attr,
            )
            setattr(self, f"{name}_communicator", comm)
            dispatch_pairs.append((resp_type, comm.handle_recv))
        self._result_dispatcher += TypeBasedDispatcher(dispatch_pairs)

    def update_control_communicator_fan_out(self: TokenizerManager, worker_count: int):
        primary_group_control = (
            self.server_args.enable_dp_attention
            and not self.server_args.enable_dp_attention_local_control_broadcast
        )
        if primary_group_control:
            control_fan_out = (
                worker_count + self.server_args.tp_size - 1
            ) // self.server_args.tp_size
        else:
            control_fan_out = worker_count

        for spec in _COMMUNICATOR_SPECS:
            getattr(self, f"{spec[0]}_communicator").set_fan_out(worker_count)

        self.get_internal_state_communicator.set_fan_out(control_fan_out)

    async def add_external_corpus(
        self: TokenizerManager, obj: AddExternalCorpusReqInput
    ) -> AddExternalCorpusReqOutput:
        self.auto_create_handle_loop()
        if self.server_args.speculative_algorithm != "NGRAM":
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        truncated = False
        try:
            if not obj.corpus_id:
                import uuid

                obj.corpus_id = uuid.uuid4().hex
            if obj.file_path is not None:
                from sglang.srt.speculative.cpp_ngram.external_corpus import (
                    iter_external_corpus_chunks,
                )

                max_tokens = (
                    self.server_args.speculative_ngram_external_corpus_max_tokens
                )
                obj.token_chunks = list(
                    iter_external_corpus_chunks(
                        obj.file_path, self.tokenizer, max_tokens
                    )
                )
            elif obj.documents is not None:
                from sglang.srt.speculative.cpp_ngram.external_corpus import (
                    SEPARATOR_TOKEN,
                )

                max_tokens = (
                    self.server_args.speculative_ngram_external_corpus_max_tokens
                )
                token_chunks = []
                total_tokens = 0
                has_prev = False
                for doc in obj.documents:
                    if not doc:
                        continue
                    token_ids = list(
                        self.tokenizer.encode(doc, add_special_tokens=False)
                    )
                    if not token_ids:
                        continue
                    if has_prev:
                        token_ids = [SEPARATOR_TOKEN] + token_ids
                    if total_tokens + len(token_ids) > max_tokens:
                        truncated = True
                        break
                    token_chunks.append(token_ids)
                    total_tokens += len(token_ids)
                    has_prev = True
                obj.token_chunks = token_chunks
            else:
                return AddExternalCorpusReqOutput(
                    success=False,
                    message="Either file_path or documents must be provided.",
                )
            obj.file_path = None
            obj.documents = None
            results = await self.add_external_corpus_communicator(obj)
            all_success, all_message = FanOutCommunicator.merge_results(results)
            if truncated and all_success:
                all_message += f" (truncated: exceeded {max_tokens} token limit)"
            return AddExternalCorpusReqOutput(
                success=all_success,
                corpus_id=results[0].corpus_id if all_success else "",
                message=all_message,
                loaded_token_count=results[0].loaded_token_count if all_success else 0,
            )
        except Exception as e:
            return AddExternalCorpusReqOutput(success=False, message=str(e))

    async def remove_external_corpus(
        self: TokenizerManager, corpus_id: str
    ) -> RemoveExternalCorpusReqOutput:
        self.auto_create_handle_loop()
        if self.server_args.speculative_algorithm != "NGRAM":
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        results = await self.remove_external_corpus_communicator(
            RemoveExternalCorpusReqInput(corpus_id=corpus_id)
        )
        all_success, all_message = FanOutCommunicator.merge_results(results)
        return RemoveExternalCorpusReqOutput(success=all_success, message=all_message)

    async def list_external_corpora(
        self: TokenizerManager,
    ) -> ListExternalCorporaReqOutput:
        self.auto_create_handle_loop()
        if self.server_args.speculative_algorithm != "NGRAM":
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        results = await self.list_external_corpora_communicator(
            ListExternalCorporaReqInput()
        )
        all_success, all_message = FanOutCommunicator.merge_results(results)
        # Merge corpus token counts from all DP ranks (each rank loads the same set).
        corpus_token_counts = results[0].corpus_token_counts if all_success else {}
        return ListExternalCorporaReqOutput(
            success=all_success,
            corpus_token_counts=corpus_token_counts,
            message=all_message,
        )

    async def flush_cache(
        self: TokenizerManager, timeout_s: Optional[float] = None
    ) -> FlushCacheReqOutput:
        self.auto_create_handle_loop()
        return (
            await self.flush_cache_communicator(FlushCacheReqInput(timeout_s=timeout_s))
        )[0]

    async def update_weight_snapshot_activation(
        self: TokenizerManager,
        action: str,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        if action not in {"activate", "close"}:
            raise ValueError("weight snapshot activation action is invalid")

        transaction_id = uuid.uuid4().hex
        transaction_deadline_unix_sec = time.time() + _CONTROL_CLEANUP_TIMEOUT_SEC

        async def run_phase(
            phase: str,
            *,
            request_action: str = "activate",
            deadline_unix_sec: float = transaction_deadline_unix_sec,
        ):
            return await self.weight_snapshot_activation_communicator(
                WeightSnapshotActivationReqInput(
                    action=request_action,
                    phase=phase,
                    transaction_id=transaction_id,
                    request_id=uuid.uuid4().hex,
                    deadline_unix_sec=deadline_unix_sec,
                ),
                deadline_unix_sec=deadline_unix_sec,
            )

        def phase_succeeded(results, phase: str, states: set[str]) -> bool:
            return bool(results) and all(
                result.success
                and result.phase == phase
                and result.transaction_id == transaction_id
                and result.state in states
                for result in results
            )

        async def abort(message: str) -> Tuple[bool, str]:
            cleanup_deadline_unix_sec = time.time() + _CONTROL_CLEANUP_TIMEOUT_SEC
            try:
                results = await run_phase(
                    "abort",
                    deadline_unix_sec=cleanup_deadline_unix_sec,
                )
            except BaseException as error:
                return False, f"{message}; abort completion is unknown: {error}"
            states = {result.state for result in results}
            if phase_succeeded(results, "abort", {"aborted"}):
                return False, message
            if "quarantined" in states:
                return False, f"{message}; activation resources are quarantined"
            return (
                False,
                f"{message}; abort failed: {FanOutCommunicator.merge_results(results)[1]}",
            )

        if action == "close":
            results = await run_phase("close", request_action="close")
            return FanOutCommunicator.merge_results(results)

        try:
            prepared = await run_phase("prepare")
        except BaseException as error:
            return await abort(f"weight snapshot activation prepare failed: {error}")
        if not phase_succeeded(prepared, "prepare", {"prepared", "serving"}):
            return await abort(
                "weight snapshot activation prepare failed: "
                + FanOutCommunicator.merge_results(prepared)[1]
            )

        commit_message = ""
        try:
            committed = await run_phase("commit")
            commit_message = FanOutCommunicator.merge_results(committed)[1]
            if phase_succeeded(
                committed,
                "commit",
                {"serving"},
            ):
                return True, commit_message
        except BaseException as error:
            commit_message = str(error)

        try:
            reconciled = await run_phase("reconcile")
        except BaseException as error:
            return await abort(
                "weight snapshot activation commit could not be reconciled: "
                f"{commit_message}; reconcile error: {error}"
            )
        if phase_succeeded(reconciled, "reconcile", {"serving"}):
            return True, "weight snapshot activation reconciled as SERVING"
        return await abort(
            "weight snapshot activation commit could not be reconciled: "
            f"{commit_message}; " + FanOutCommunicator.merge_results(reconciled)[1]
        )

    async def clear_hicache_storage(self: TokenizerManager) -> ClearHiCacheReqOutput:
        """Clear the hierarchical cache storage."""
        self.auto_create_handle_loop()
        # Delegate to the scheduler to handle HiCacheStorage clearing
        return (await self.clear_hicache_storage_communicator(ClearHiCacheReqInput()))[
            0
        ]

    async def attach_hicache_storage(
        self: TokenizerManager,
        hicache_storage_backend: str,
        hicache_storage_backend_extra_config_json: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> AttachHiCacheStorageReqOutput:
        """Attach (enable) HiCache storage backend at runtime."""
        self.auto_create_handle_loop()
        results = await self.attach_hicache_storage_communicator(
            AttachHiCacheStorageReqInput(
                hicache_storage_backend=hicache_storage_backend,
                hicache_storage_backend_extra_config_json=hicache_storage_backend_extra_config_json,
                hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
                hicache_write_policy=hicache_write_policy,
            )
        )

        all_success, all_message = FanOutCommunicator.merge_results(results)
        out = AttachHiCacheStorageReqOutput(success=all_success, message=all_message)
        # TODO: partial rollback if failed
        if all_success:
            # Keep tokenizer side server_info consistent with scheduler side.
            hicache_fields = {"hicache_storage_backend": hicache_storage_backend}
            if hicache_storage_backend_extra_config_json is not None:
                hicache_fields["hicache_storage_backend_extra_config"] = (
                    hicache_storage_backend_extra_config_json
                )
            if hicache_storage_prefetch_policy is not None:
                hicache_fields["hicache_storage_prefetch_policy"] = (
                    hicache_storage_prefetch_policy
                )
            if hicache_write_policy is not None:
                hicache_fields["hicache_write_policy"] = hicache_write_policy
            self.server_args.override("tokenizer.attach_hicache", **hicache_fields)
        return out

    async def detach_hicache_storage(
        self: TokenizerManager,
    ) -> DetachHiCacheStorageReqOutput:
        """Detach (disable) HiCache storage backend at runtime."""
        self.auto_create_handle_loop()
        results = await self.detach_hicache_storage_communicator(
            DetachHiCacheStorageReqInput()
        )

        all_success, all_message = FanOutCommunicator.merge_results(results)
        out = DetachHiCacheStorageReqOutput(success=all_success, message=all_message)
        # TODO: partial rollback if failed
        if all_success:
            self.server_args.override(
                "tokenizer.detach_hicache",
                hicache_storage_backend=None,
                hicache_storage_backend_extra_config=None,
            )
        return out

    async def start_profile(
        self: TokenizerManager,
        req: Optional[ProfileReq] = None,
    ):
        self.auto_create_handle_loop()
        req = req or ProfileReq()
        req.req_type = ProfileReqType.START_PROFILE
        env_with_stack: bool = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
        req.with_stack = (
            False if req.with_stack is False or env_with_stack is False else True
        )
        env_record_shapes: bool = get_bool_env_var(
            "SGLANG_PROFILE_RECORD_SHAPES", "true"
        )
        req.record_shapes = (req.record_shapes is not False) and env_record_shapes
        req.profile_id = req.profile_id or str(time.time())
        return await self._execute_profile(req)

    async def stop_profile(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
        return await self._execute_profile(req)

    async def _execute_profile(self: TokenizerManager, req: ProfileReq):
        result = (await self.profile_communicator(req))[0]
        if not result.success:
            raise RuntimeError(result.message)
        return result

    async def start_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.START_RECORD)
        await self.expert_distribution_communicator(req)

    async def stop_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.STOP_RECORD)
        await self.expert_distribution_communicator(req)

    async def dump_expert_distribution_record(self: TokenizerManager):
        self.auto_create_handle_loop()
        req = ExpertDistributionReq(action=ExpertDistributionReqType.DUMP_RECORD)
        await self.expert_distribution_communicator(req)

    async def init_weights_update_group(
        self: TokenizerManager,
        obj: InitWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert self.server_args.dp_size == 1 or self.server_args.enable_dp_attention, (
            "dp_size must be 1 or dp attention must be enabled for update weights from distributed"
        )

        results = await self.init_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def destroy_weights_update_group(
        self: TokenizerManager,
        obj: DestroyWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert self.server_args.dp_size == 1 or self.server_args.enable_dp_attention, (
            "dp_size must be 1 or dp attention must be enabled for destroy parameter update group"
        )

        results = await self.destroy_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def update_weights_from_distributed(
        self: TokenizerManager,
        obj: UpdateWeightsFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self._require_single_tokenizer_weight_update_owner()
        self.auto_create_handle_loop()
        assert self.server_args.dp_size == 1 or self.server_args.enable_dp_attention, (
            "dp_size must be 1 or dp attention must be enabled for update weights from distributed"
        )
        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        async def update_locked():
            async with self.model_update_lock.writer_lock:
                _validate_next_weight_revision(self, obj.weight_version)
                results = await _call_weight_update_communicator(
                    self,
                    self.update_weights_from_distributed_communicator,
                    obj,
                )
                return _finish_weight_update_transaction(
                    self,
                    results,
                    weight_version=obj.weight_version,
                    full_restore=False,
                )

        # Hold is_pause_cond while updating to prevent unpause from racing.
        async with self.is_pause_cond:
            if self.is_pause:
                return await update_locked()
        return await update_locked()

    async def init_weights_send_group_for_remote_instance(
        self: TokenizerManager,
        obj: InitWeightsSendGroupForRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        # TODO: support DP
        assert self.server_args.dp_size == 1, (
            "dp_size must be 1 for init_weights_send_group_for_remote_instance"
        )
        result = (
            await self.init_weights_send_group_for_remote_instance_communicator(obj)
        )[0]
        return result.success, result.message

    async def send_weights_to_remote_instance(
        self: TokenizerManager,
        obj: SendWeightsToRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        # TODO: support DP
        assert self.server_args.dp_size == 1, (
            "dp_size must be 1 for send_weights_to_remote_instance"
        )
        result = (await self.send_weights_to_remote_instance_communicator(obj))[0]
        return result.success, result.message

    async def materialize_weights(
        self: TokenizerManager,
        obj: MaterializeWeightsReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Dict[str, Any]:
        materialization_id = (
            uuid.uuid4().hex
            if obj.materialization_id is None
            else obj.materialization_id
        )
        if type(materialization_id) is not str or not materialization_id.strip():
            raise ValueError("materialization_id must be a non-empty string")
        if not isinstance(obj.storage_options, dict):
            raise ValueError("storage_options must be a dictionary")
        validate_remote_instance_weight_transfer_lease_timeout(obj.lease_timeout_sec)
        deadline_unix_sec = time.time() + obj.lease_timeout_sec

        if not self.server_args.enable_weight_runtime_manifest:
            raise WeightMaterializationError(
                "weight materialization requires --enable-weight-runtime-manifest",
                materialization_id=materialization_id,
                session_state=WeightMaterializationSessionState.DISABLED,
            )

        self.auto_create_handle_loop()
        expected_fan_out = _weight_materialization_fan_out(self)
        selected_rank = obj.source_external_dp_rank
        if selected_rank is not None and (
            type(selected_rank) is not int
            or selected_rank < 0
            or selected_rank >= expected_fan_out
        ):
            raise ValueError(
                f"source_external_dp_rank must be an integer in [0, {expected_fan_out})"
            )

        active_ids = _weight_materialization_active_ids(self)
        if materialization_id in active_ids:
            raise WeightMaterializationError(
                "weight materialization is already active",
                materialization_id=materialization_id,
                session_state=WeightMaterializationSessionState.CONFLICT,
            )
        active_ids.add(materialization_id)
        storage_options = dict(obj.storage_options)
        cleanup_candidate = False

        try:
            async with self.model_update_lock.snapshot_reader_lock:
                _require_weight_snapshot_export_allowed(self)
                try:
                    prepare_results = (
                        await self.prepare_weight_materialization_communicator(
                            PrepareWeightMaterializationReqInput(
                                materialization_id=materialization_id,
                                request_id=uuid.uuid4().hex,
                                model_id=self.server_args.model_path,
                                revision=_serving_weight_revision(self),
                                lease_timeout_sec=obj.lease_timeout_sec,
                                deadline_unix_sec=deadline_unix_sec,
                            ),
                            deadline_unix_sec=deadline_unix_sec,
                        )
                    )
                    cleanup_candidate = True
                    prepared = _ordered_weight_materialization_results(
                        prepare_results,
                        materialization_id=materialization_id,
                        expected_fan_out=expected_fan_out,
                        phase="weight materialization prepare",
                        allow_identical_duplicates=True,
                    )
                    prepare_failures = [
                        result.message for result in prepared if not result.success
                    ]
                    if prepare_failures:
                        raise WeightMaterializationError(
                            "weight materialization prepare failed: "
                            + " | ".join(prepare_failures),
                            materialization_id=materialization_id,
                            session_state=_weight_materialization_result_state(
                                prepared,
                                WeightMaterializationSessionState.FAILED,
                            ),
                        )

                    generations = {result.generation for result in prepared}
                    digests = {result.logical_payload_digest for result in prepared}
                    byte_counts = {result.total_bytes for result in prepared}
                    if (
                        len(generations) != 1
                        or None in generations
                        or len(digests) != 1
                        or None in digests
                        or not next(iter(digests))
                        or len(byte_counts) != 1
                        or None in byte_counts
                    ):
                        raise WeightMaterializationError(
                            "source DP replicas returned inconsistent generation, "
                            "logical payload digest, or total bytes",
                            materialization_id=materialization_id,
                            session_state=WeightMaterializationSessionState.CONFLICT,
                        )
                    generation = next(iter(generations))
                    logical_payload_digest = next(iter(digests))
                    total_bytes = next(iter(byte_counts))
                    if (
                        type(generation) is not int
                        or generation < 0
                        or type(logical_payload_digest) is not str
                        or not logical_payload_digest
                        or type(total_bytes) is not int
                        or total_bytes < 0
                    ):
                        raise WeightMaterializationError(
                            "source DP replicas returned invalid generation, "
                            "logical payload digest, or total bytes",
                            materialization_id=materialization_id,
                            session_state=WeightMaterializationSessionState.CONFLICT,
                        )

                    selected_rank = (
                        prepared[0].external_dp_rank
                        if selected_rank is None
                        else selected_rank
                    )
                    commit_results = (
                        await self.commit_weight_materialization_communicator(
                            CommitWeightMaterializationReqInput(
                                materialization_id=materialization_id,
                                request_id=uuid.uuid4().hex,
                                selected_external_dp_rank=selected_rank,
                                storage_options=storage_options,
                                deadline_unix_sec=deadline_unix_sec,
                            ),
                            deadline_unix_sec=deadline_unix_sec,
                        )
                    )
                    committed = _ordered_weight_materialization_results(
                        commit_results,
                        materialization_id=materialization_id,
                        expected_fan_out=expected_fan_out,
                        phase="weight materialization commit",
                        allow_identical_duplicates=False,
                    )
                    commit_failures = [
                        result.message
                        for result in committed
                        if not result.success or result.completion_unknown
                    ]
                    if commit_failures:
                        completion_tickets = {
                            result.completion_ticket
                            for result in committed
                            if result.completion_ticket
                        }
                        raise WeightMaterializationError(
                            "weight materialization commit failed: "
                            + " | ".join(commit_failures),
                            materialization_id=materialization_id,
                            session_state=_weight_materialization_result_state(
                                committed,
                                WeightMaterializationSessionState.FAILED,
                            ),
                            completion_ticket=(
                                next(iter(completion_tickets))
                                if len(completion_tickets) == 1
                                else None
                            ),
                        )

                    refs = [
                        result.ref for result in committed if result.ref is not None
                    ]
                    if len(refs) != 1:
                        raise WeightMaterializationError(
                            "weight materialization commit must return exactly one "
                            "storage ref",
                            materialization_id=materialization_id,
                            session_state=WeightMaterializationSessionState.CONFLICT,
                        )
                    selected_results = [
                        result for result in committed if result.selected
                    ]
                    if (
                        len(selected_results) != 1
                        or selected_results[0].external_dp_rank != selected_rank
                        or selected_results[0].ref != refs[0]
                        or any(
                            result.selected or result.ref is not None
                            for result in committed
                            if result.external_dp_rank != selected_rank
                        )
                    ):
                        raise WeightMaterializationError(
                            "weight materialization commit returned inconsistent "
                            "source selection or refs",
                            materialization_id=materialization_id,
                            session_state=WeightMaterializationSessionState.CONFLICT,
                        )
                    if not isinstance(refs[0], dict) or not refs[0]:
                        raise WeightMaterializationError(
                            "weight materialization commit returned an invalid "
                            "storage ref",
                            materialization_id=materialization_id,
                            session_state=WeightMaterializationSessionState.CONFLICT,
                        )

                    selected_result = selected_results[0]
                    cleanup_state = None
                    selected_state = WeightMaterializationSessionState(
                        selected_result.session_state
                    )
                    if (
                        is_published_materialization_state(selected_state)
                        and selected_state
                        is not WeightMaterializationSessionState.PUBLISHED
                    ):
                        cleanup_state = await _cleanup_weight_materialization(
                            self,
                            materialization_id=materialization_id,
                            storage_options=storage_options,
                        )
                    session_state = cleanup_state or selected_result.session_state
                    return {
                        "materialization_id": materialization_id,
                        "ref": dict(refs[0]),
                        "selected_external_dp_rank": selected_rank,
                        "total_bytes": total_bytes,
                        "session_state": session_state,
                        "cleanup_state": cleanup_state,
                        "completion_unknown": (
                            session_state
                            == WeightMaterializationSessionState.COMPLETION_UNKNOWN
                        ),
                        "completion_ticket": selected_result.completion_ticket,
                    }
                except (
                    FanOutCancelledBeforeDispatch,
                    FanOutDeadlineExpiredBeforeDispatch,
                ):
                    if cleanup_candidate:
                        await _cleanup_weight_materialization(
                            self,
                            materialization_id=materialization_id,
                            storage_options=storage_options,
                        )
                    raise
                except asyncio.CancelledError:
                    await _cleanup_weight_materialization(
                        self,
                        materialization_id=materialization_id,
                        storage_options=storage_options,
                    )
                    raise
                except WeightMaterializationError as error:
                    cleanup_state = await _cleanup_weight_materialization(
                        self,
                        materialization_id=materialization_id,
                        storage_options=storage_options,
                    )
                    if cleanup_state in (
                        WeightMaterializationSessionState.CLEANUP_PENDING,
                        WeightMaterializationSessionState.COMPLETION_UNKNOWN,
                    ):
                        error.session_state = cleanup_state
                    raise
                except Exception as error:
                    cleanup_state = await _cleanup_weight_materialization(
                        self,
                        materialization_id=materialization_id,
                        storage_options=storage_options,
                    )
                    raise WeightMaterializationError(
                        f"weight materialization failed: {error}",
                        materialization_id=materialization_id,
                        session_state=(
                            cleanup_state
                            if cleanup_state
                            in (
                                WeightMaterializationSessionState.CLEANUP_PENDING,
                                WeightMaterializationSessionState.COMPLETION_UNKNOWN,
                            )
                            else WeightMaterializationSessionState.FAILED
                        ),
                    ) from error
        finally:
            active_ids.discard(materialization_id)

    async def begin_remote_instance_weight_transfer(
        self: TokenizerManager,
        lease_timeout_sec: int = (
            DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
        ),
        manifest_format: str = RUNTIME_MANIFEST_V1,
        transfer_id: str | None = None,
        *,
        manifest_revision_semantics: str = HF_REVISION_V1,
        lease_fence: str | None = None,
    ) -> dict:
        """Pause for snapshot capture, then serve while the lease is held."""
        if not self.server_args.enable_weight_runtime_manifest:
            raise RuntimeError(
                "remote heterogeneous weight reuse requires "
                "--enable-weight-runtime-manifest"
            )
        validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        validate_manifest_revision_semantics(
            manifest_format,
            manifest_revision_semantics,
        )
        if transfer_id is not None and (
            type(transfer_id) is not str or not transfer_id
        ):
            raise ValueError("transfer_id must be a non-empty string")
        if lease_fence is not None and (
            type(lease_fence) is not str or not lease_fence
        ):
            raise ValueError("lease_fence must be a non-empty string")
        _require_weight_snapshot_export_allowed(self)

        self.auto_create_handle_loop()
        transfer_id = transfer_id or uuid.uuid4().hex
        # The caller value is accepted for wire compatibility, but a new lease
        # always starts with a source-issued begin token.
        lease_fence = (
            f"{_REMOTE_WEIGHT_TRANSFER_BEGIN_FENCE_PREFIX}{secrets.token_urlsafe(32)}"
        )
        async with _remote_weight_transfer_begin_lock(self):
            return await TokenizerControlMixin._begin_remote_instance_weight_transfer(
                self,
                lease_timeout_sec=lease_timeout_sec,
                manifest_format=manifest_format,
                manifest_revision_semantics=manifest_revision_semantics,
                transfer_id=transfer_id,
                lease_fence=lease_fence,
            )

    async def _begin_remote_instance_weight_transfer(
        self: TokenizerManager,
        *,
        lease_timeout_sec: int,
        manifest_format: str,
        manifest_revision_semantics: str,
        transfer_id: str,
        lease_fence: str,
    ) -> dict:
        deadline_unix_sec = time.time() + lease_timeout_sec
        results = None

        async def capture_snapshot():
            nonlocal results
            async with TokenizerControlMixin._remote_instance_weight_transfer_pause(
                self
            ):
                async with self.model_update_lock.reader_lock:
                    _require_weight_snapshot_export_allowed(self)
                    revision = (
                        _serving_weight_revision(self)
                        if manifest_revision_semantics == ARTIFACT_WEIGHT_VERSION_V1
                        else _hf_model_revision(self)
                    )
                    request = BeginRemoteInstanceWeightTransferReqInput(
                        transfer_id=transfer_id,
                        model_id=self.server_args.model_path,
                        revision=revision,
                        lease_timeout_sec=lease_timeout_sec,
                        manifest_format=manifest_format,
                        manifest_revision_semantics=manifest_revision_semantics,
                        request_id=uuid.uuid4().hex,
                        deadline_unix_sec=deadline_unix_sec,
                        lease_fence=lease_fence,
                    )
                    results = (
                        await self.begin_remote_instance_weight_transfer_communicator(
                            request,
                            deadline_unix_sec=deadline_unix_sec,
                        )
                    )
            return results

        capture_task = asyncio.create_task(capture_snapshot())
        try:
            results = await asyncio.shield(capture_task)
        except asyncio.CancelledError:
            capture_error = None
            try:
                results = await _finish_control_task(capture_task)
            except BaseException as error:
                capture_error = error

            cleanup_candidate = _remote_weight_transfer_created_by_request(
                results or ()
            )
            cleanup_uncertain = (
                capture_error is not None
                and _fan_out_may_have_dispatched(capture_error)
            )
            if cleanup_candidate or cleanup_uncertain:
                cleanup_payloads = _remote_weight_transfer_result_payloads(
                    results or (), manifest_format
                )
                _, payload_generation = _remote_weight_transfer_lease_identity(
                    cleanup_payloads
                )
                reported_fences = {
                    getattr(result, "lease_fence", None)
                    for result in (results or ())
                    if getattr(result, "lease_fence", None) is not None
                }
                cleanup_lease_fence = (
                    next(iter(reported_fences))
                    if len(reported_fences) == 1
                    else (lease_fence if not reported_fences else None)
                )
                reported_generations = {
                    getattr(result, "generation", None)
                    for result in (results or ())
                    if getattr(result, "generation", None) is not None
                }
                cleanup_generation = (
                    next(iter(reported_generations))
                    if len(reported_generations) == 1
                    else (payload_generation if not reported_generations else None)
                )
                if (
                    cleanup_generation is not None
                    and payload_generation is not None
                    and cleanup_generation != payload_generation
                ):
                    cleanup_generation = None
                _remember_remote_weight_transfer_session(
                    self,
                    transfer_id=transfer_id,
                    manifest_format=manifest_format,
                    manifest_revision_semantics=manifest_revision_semantics,
                    deadline_unix_sec=deadline_unix_sec,
                    payloads=cleanup_payloads,
                    session_state="cleanup_pending",
                    lease_fence=cleanup_lease_fence or lease_fence,
                )
                cleanup_task = asyncio.create_task(
                    _release_uncertain_remote_weight_transfer(
                        self,
                        transfer_id=transfer_id,
                        lease_fence=cleanup_lease_fence,
                        generation=cleanup_generation,
                    )
                )
                try:
                    await _finish_control_task(cleanup_task)
                except Exception:
                    logger.exception(
                        "Failed to clean up cancelled remote weight transfer %s",
                        transfer_id,
                    )
            raise
        except Exception as error:
            if results is None:
                if not _fan_out_may_have_dispatched(error):
                    raise
                _remember_remote_weight_transfer_session(
                    self,
                    transfer_id=transfer_id,
                    manifest_format=manifest_format,
                    manifest_revision_semantics=manifest_revision_semantics,
                    deadline_unix_sec=deadline_unix_sec,
                    payloads=(),
                    session_state="cleanup_pending",
                    lease_fence=lease_fence,
                )
                cleanup_succeeded = await _release_uncertain_remote_weight_transfer(
                    self,
                    transfer_id=transfer_id,
                    lease_fence=lease_fence,
                )
                raise RemoteInstanceWeightTransferBeginError(
                    f"Source snapshot response was lost after dispatch: {error}",
                    transfer_id=transfer_id,
                    session_state=(
                        "failed" if cleanup_succeeded else "cleanup_pending"
                    ),
                ) from error
            session_states = [
                getattr(result, "session_state", "unknown") for result in results
            ]
            cleanup_pending = _remote_weight_transfer_created_by_request(results)
            if not cleanup_pending:
                if results and all(state == "reused" for state in session_states):
                    session_state = "reused"
                elif "conflict" in session_states:
                    session_state = "conflict"
                elif "expired" in session_states:
                    session_state = "expired"
                else:
                    session_state = "failed"
                raise RemoteInstanceWeightTransferBeginError(
                    f"Failed to resume source generation after snapshot capture: "
                    f"{error}",
                    transfer_id=transfer_id,
                    session_state=session_state,
                ) from error
            _remember_remote_weight_transfer_session(
                self,
                transfer_id=transfer_id,
                manifest_format=manifest_format,
                manifest_revision_semantics=manifest_revision_semantics,
                deadline_unix_sec=deadline_unix_sec,
                payloads=_remote_weight_transfer_result_payloads(
                    results, manifest_format
                ),
                session_state="cleanup_pending",
                lease_fence=lease_fence,
            )
            raise RemoteInstanceWeightTransferBeginError(
                f"Failed to resume source generation after snapshot capture: {error}",
                transfer_id=transfer_id,
                session_state="cleanup_pending",
            ) from error
        failures = [result.message for result in results if not result.success]
        reported_fences = {
            getattr(result, "lease_fence", None)
            for result in results
            if getattr(result, "lease_fence", None) is not None
        }
        if len(reported_fences) > 1:
            failures.append("source workers returned inconsistent lease fences")
        authoritative_lease_fence = (
            next(iter(reported_fences)) if reported_fences else lease_fence
        )
        lease_fence = authoritative_lease_fence
        manifests = None
        placements = None
        bindings = None
        if not results:
            failures.append("source workers returned no transfer responses")
        elif not failures:
            result_semantics = {
                getattr(result, "manifest_revision_semantics", HF_REVISION_V1)
                for result in results
            }
            if result_semantics != {manifest_revision_semantics}:
                failures.append(
                    "source workers returned incompatible manifest revision semantics"
                )
            else:
                try:
                    if manifest_format == PLACEMENT_BINDING_V1:
                        placements, bindings = _merge_placement_binding_groups(
                            [result.placements for result in results],
                            [result.bindings for result in results],
                        )
                    else:
                        manifests = _merge_runtime_manifest_groups(
                            [result.manifests for result in results]
                        )
                except RuntimeError as error:
                    failures.append(str(error))
        raw_session_payloads = _remote_weight_transfer_result_payloads(
            results, manifest_format
        )
        session_payloads = (
            bindings if manifest_format == PLACEMENT_BINDING_V1 else manifests
        ) or []
        _, payload_generation = _remote_weight_transfer_lease_identity(
            raw_session_payloads
        )
        reported_generations = {
            getattr(result, "generation", None)
            for result in results
            if getattr(result, "generation", None) is not None
        }
        if len(reported_generations) > 1:
            failures.append("source workers returned inconsistent snapshot generation")
        generation = (
            next(iter(reported_generations))
            if reported_generations
            else payload_generation
        )
        if (
            generation is not None
            and payload_generation is not None
            and generation != payload_generation
        ):
            failures.append(
                "source snapshot generation does not match scheduler authority"
            )
        if failures:
            session_states = [
                getattr(result, "session_state", "unknown") for result in results
            ]
            cleanup_candidate = _remote_weight_transfer_created_by_request(results)
            cleanup_succeeded = False
            cleanup_cancellation = None
            if cleanup_candidate:
                _remember_remote_weight_transfer_session(
                    self,
                    transfer_id=transfer_id,
                    manifest_format=manifest_format,
                    manifest_revision_semantics=manifest_revision_semantics,
                    deadline_unix_sec=deadline_unix_sec,
                    payloads=raw_session_payloads,
                    session_state="cleanup_pending",
                    lease_fence=lease_fence,
                )
                for _ in range(3):
                    cleanup_task = asyncio.create_task(
                        TokenizerControlMixin.release_remote_instance_weight_transfer(
                            self,
                            transfer_id,
                            lease_fence=lease_fence,
                            generation=generation,
                        )
                    )
                    try:
                        (
                            cleanup_succeeded,
                            _,
                        ) = await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError as error:
                        cleanup_cancellation = cleanup_cancellation or error
                        try:
                            cleanup_succeeded, _ = await _finish_control_task(
                                cleanup_task
                            )
                        except Exception:
                            logger.exception(
                                "Failed to clean up remote weight transfer %s",
                                transfer_id,
                            )
                    except Exception:
                        logger.exception(
                            "Failed to clean up remote weight transfer %s",
                            transfer_id,
                        )
                    if cleanup_succeeded:
                        break
            if cleanup_candidate:
                session_state = "failed" if cleanup_succeeded else "cleanup_pending"
            elif "conflict" in session_states:
                session_state = "conflict"
            elif "expired" in session_states:
                session_state = "expired"
            elif "released" in session_states:
                session_state = "released"
            else:
                session_state = "failed"
            if session_state in {"cleanup_pending", "expired"}:
                _remember_remote_weight_transfer_session(
                    self,
                    transfer_id=transfer_id,
                    manifest_format=manifest_format,
                    manifest_revision_semantics=manifest_revision_semantics,
                    deadline_unix_sec=deadline_unix_sec,
                    payloads=_remote_weight_transfer_result_payloads(
                        results, manifest_format
                    ),
                    session_state=session_state,
                    lease_fence=lease_fence,
                )
            if cleanup_cancellation is not None:
                raise cleanup_cancellation
            raise RemoteInstanceWeightTransferBeginError(
                " | ".join(failures),
                transfer_id=transfer_id,
                session_state=session_state,
            )
        if manifest_format == PLACEMENT_BINDING_V1:
            assert placements is not None and bindings is not None
            session_payloads = bindings
            response = {
                "transfer_id": transfer_id,
                "source_weight_placements": placements,
                "source_weight_runtime_bindings": bindings,
                "lease_timeout_sec": lease_timeout_sec,
                "manifest_revision_semantics": manifest_revision_semantics,
            }
        else:
            assert manifests is not None
            session_payloads = manifests
            response = {
                "transfer_id": transfer_id,
                "weight_runtime_manifests": manifests,
                "lease_timeout_sec": lease_timeout_sec,
                "manifest_revision_semantics": manifest_revision_semantics,
            }
        if generation is None:
            raise RuntimeError(
                "source workers returned no authoritative snapshot generation"
            )
        response["lease_fence"] = authoritative_lease_fence
        response["generation"] = generation
        reused = bool(results) and all(
            getattr(result, "session_state", "created") == "reused"
            for result in results
        )
        existing = _refresh_remote_weight_transfer_session(self, transfer_id)
        _remember_remote_weight_transfer_session(
            self,
            transfer_id=transfer_id,
            manifest_format=manifest_format,
            manifest_revision_semantics=manifest_revision_semantics,
            deadline_unix_sec=(
                existing.get("deadline_unix_sec")
                if reused and existing is not None
                else (None if reused else deadline_unix_sec)
            ),
            payloads=session_payloads,
            session_state="reused" if reused and existing is None else "active",
            lease_fence=authoritative_lease_fence,
        )
        return response

    async def release_remote_instance_weight_transfer(
        self: TokenizerManager,
        transfer_id: str,
        *,
        lease_fence: str | None = None,
        generation: int | None = None,
    ) -> Tuple[bool, str]:
        _validate_remote_weight_transfer_control_identity(lease_fence, generation)
        self.auto_create_handle_loop()
        if lease_fence is None:
            transfer_id = _resolve_unfenced_remote_weight_transfer_control(
                self,
                transfer_id,
            )
        else:
            if type(transfer_id) is not str or not transfer_id:
                raise ValueError("transfer_id must be a non-empty string")
        attempted_at = time.time()
        request_id = uuid.uuid4().hex
        deadline_unix_sec = time.time() + _CONTROL_CLEANUP_TIMEOUT_SEC
        completion_unknown_error = None
        try:
            async with TokenizerControlMixin._remote_instance_weight_transfer_pause(
                self
            ):
                results = (
                    await self.release_remote_instance_weight_transfer_communicator(
                        ReleaseRemoteInstanceWeightTransferReqInput(
                            transfer_id=transfer_id,
                            request_id=request_id,
                            lease_fence=lease_fence,
                            generation=generation,
                            deadline_unix_sec=deadline_unix_sec,
                        ),
                        deadline_unix_sec=deadline_unix_sec,
                    )
                )
            success, message = FanOutCommunicator.merge_results(results)
        except Exception as error:
            success, message = False, str(error)
            if isinstance(error, FanOutCompletionUnknownError) or bool(
                getattr(error, "completion_unknown", False)
            ):
                completion_unknown_error = error
        _record_remote_weight_transfer_release(
            self,
            transfer_id=transfer_id,
            attempted_at=attempted_at,
            success=success,
            message=message,
            completion_unknown=completion_unknown_error is not None,
            lease_fence=lease_fence,
            generation=generation,
        )
        log = logger.info if success else logger.warning
        log(
            "Explicit remote weight transfer release: "
            "transfer_id=%s success=%s message=%s",
            transfer_id,
            success,
            message,
        )
        if completion_unknown_error is not None:
            raise RemoteInstanceWeightTransferControlError(
                message,
                transfer_id=transfer_id,
                session_state="cleanup_pending",
                lease_fence=lease_fence,
                generation=generation,
            ) from completion_unknown_error
        return success, message

    async def renew_remote_instance_weight_transfer(
        self: TokenizerManager,
        transfer_id: str,
        lease_timeout_sec: int = (
            DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
        ),
        *,
        lease_fence: str | None = None,
        generation: int | None = None,
    ) -> Tuple[bool, str]:
        validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        _validate_remote_weight_transfer_control_identity(lease_fence, generation)
        self.auto_create_handle_loop()
        if lease_fence is None:
            transfer_id = _resolve_unfenced_remote_weight_transfer_control(
                self,
                transfer_id,
            )
        elif type(transfer_id) is not str or not transfer_id:
            raise ValueError("transfer_id must be a non-empty string")
        legacy_deadline_unix_sec = time.time() + lease_timeout_sec
        request_id = uuid.uuid4().hex
        granted_deadline_unix_sec = None
        deadline_unix_sec = time.time() + min(
            lease_timeout_sec, _CONTROL_CLEANUP_TIMEOUT_SEC
        )
        completion_unknown_error = None
        try:
            results = await self.renew_remote_instance_weight_transfer_communicator(
                RenewRemoteInstanceWeightTransferReqInput(
                    transfer_id=transfer_id,
                    lease_timeout_sec=lease_timeout_sec,
                    request_id=request_id,
                    lease_fence=lease_fence,
                    generation=generation,
                    deadline_unix_sec=deadline_unix_sec,
                ),
                deadline_unix_sec=deadline_unix_sec,
            )
            success, message = FanOutCommunicator.merge_results(results)
            if success:
                granted_deadlines = [
                    getattr(result, "deadline_unix_sec", None) for result in results
                ]
                present_deadlines = [
                    deadline for deadline in granted_deadlines if deadline is not None
                ]
                if present_deadlines and len(present_deadlines) != len(
                    granted_deadlines
                ):
                    success = False
                    message = (
                        "Source replicas returned mixed legacy and authoritative "
                        "lease deadlines."
                    )
                elif any(
                    isinstance(deadline, bool)
                    or not isinstance(deadline, (int, float))
                    or not math.isfinite(deadline)
                    or deadline <= 0
                    for deadline in present_deadlines
                ):
                    success = False
                    message = "Source replicas returned an invalid lease deadline."
                elif present_deadlines:
                    granted_deadline_unix_sec = min(present_deadlines)
                else:
                    granted_deadline_unix_sec = legacy_deadline_unix_sec
        except Exception as error:
            success, message = False, str(error)
            if isinstance(error, FanOutCompletionUnknownError) or bool(
                getattr(error, "completion_unknown", False)
            ):
                completion_unknown_error = error
        if completion_unknown_error is not None:
            _record_remote_weight_transfer_completion_unknown(
                self,
                transfer_id=transfer_id,
                session_state="completion_unknown",
                lease_fence=lease_fence,
                generation=generation,
            )
            raise RemoteInstanceWeightTransferControlError(
                message,
                transfer_id=transfer_id,
                session_state="completion_unknown",
                lease_fence=lease_fence,
                generation=generation,
            ) from completion_unknown_error
        if success:
            assert granted_deadline_unix_sec is not None
            _record_remote_weight_transfer_renewal(
                self,
                transfer_id=transfer_id,
                deadline_unix_sec=granted_deadline_unix_sec,
            )
        return success, message

    async def list_remote_instance_weight_transfer_sessions(
        self: TokenizerManager,
    ) -> List[Dict[str, Any]]:
        index = _remote_weight_transfer_session_index(self)
        sessions = [
            _refresh_remote_weight_transfer_session(self, transfer_id)
            for transfer_id in sorted(index)
        ]
        return [session for session in sessions if session is not None]

    async def get_remote_instance_weight_transfer_session(
        self: TokenizerManager,
        transfer_id: str,
        *,
        lease_fence: str | None = None,
        generation: int | None = None,
    ) -> Dict[str, Any] | None:
        if type(transfer_id) is not str or not transfer_id:
            raise ValueError("transfer_id must be a non-empty string")
        _validate_remote_weight_transfer_control_identity(lease_fence, generation)
        local_session = _refresh_remote_weight_transfer_session(self, transfer_id)
        communicator = getattr(
            self,
            "get_remote_instance_weight_transfer_session_communicator",
            None,
        )
        if communicator is None:
            return local_session

        self.auto_create_handle_loop()
        if lease_fence is None and local_session is not None:
            lease_fence = local_session.get("lease_fence")
            generation = local_session.get("generation")
        deadline_unix_sec = time.time() + _CONTROL_CLEANUP_TIMEOUT_SEC
        request_id = uuid.uuid4().hex
        results = await communicator(
            GetRemoteInstanceWeightTransferSessionReqInput(
                transfer_id=transfer_id,
                request_id=request_id,
                lease_fence=lease_fence,
                generation=generation,
                deadline_unix_sec=deadline_unix_sec,
            ),
            deadline_unix_sec=deadline_unix_sec,
        )
        success, message = FanOutCommunicator.merge_results(results)
        if not success:
            if results and all(
                getattr(result, "session_state", "unknown") == "unknown"
                for result in results
            ):
                return None
            raise RuntimeError(message)

        states = {result.session_state for result in results}
        fences = {result.lease_fence for result in results}
        generations = {result.generation for result in results}
        if len(states) != 1 or len(fences) != 1 or len(generations) != 1:
            raise RuntimeError(
                "source workers returned inconsistent transfer session identity"
            )
        session_state = next(iter(states))
        authoritative_fence = next(iter(fences))
        authoritative_generation = next(iter(generations))
        deadlines = [
            result.deadline_unix_sec
            for result in results
            if result.deadline_unix_sec is not None
        ]
        deadline = min(deadlines) if deadlines else None

        record = dict(local_session or {})
        record.update(
            {
                "transfer_id": transfer_id,
                "lease_fence": authoritative_fence,
                "generation": authoritative_generation,
                "deadline_unix_sec": deadline,
                "expired": session_state == "expired",
                "session_state": session_state,
            }
        )
        _remote_weight_transfer_session_index(self)[transfer_id] = record
        return dict(record)

    async def update_weights_from_tensor(
        self: TokenizerManager,
        obj: UpdateWeightsFromTensorReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self._require_single_tokenizer_weight_update_owner()
        self.auto_create_handle_loop()
        assert self.server_args.dp_size == 1 or self.server_args.enable_dp_attention, (
            "dp_size must be 1 or dp attention must be enabled for update weights from tensor"
        )
        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        obj.serialized_named_tensors = normalize_serialized_named_tensor_payloads(
            obj.serialized_named_tensors
        )

        async def update_locked():
            async with self.model_update_lock.writer_lock:
                _validate_next_weight_revision(self, obj.weight_version)
                results = await _call_weight_update_communicator(
                    self,
                    self.update_weights_from_tensor_communicator,
                    obj,
                )
                return _finish_weight_update_transaction(
                    self,
                    results,
                    weight_version=obj.weight_version,
                    full_restore=False,
                )

        async with self.is_pause_cond:
            if self.is_pause:
                return await update_locked()
        return await update_locked()

    async def update_weights_from_ipc(
        self: TokenizerManager,
        obj: UpdateWeightsFromIPCReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Update weights via IPC for checkpoint-engine integration."""
        self._require_single_tokenizer_weight_update_owner()
        self.auto_create_handle_loop()
        try:
            # For now, we only support single data parallel instance
            assert (
                self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
            ), (
                "dp_size must be 1 or dp attention must be enabled for update weights from IPC"
            )
            logger.info("Starting IPC weight update")

            async def update_locked():
                async with self.model_update_lock.writer_lock:
                    _validate_next_weight_revision(self, obj.weight_version)
                    results = await _call_weight_update_communicator(
                        self,
                        self.update_weights_from_ipc_communicator,
                        obj,
                    )
                    return _finish_weight_update_transaction(
                        self,
                        results,
                        weight_version=obj.weight_version,
                        full_restore=False,
                    )

            async with self.is_pause_cond:
                if self.is_pause:
                    return await update_locked()
            return await update_locked()
        except Exception as e:
            error_msg = f"IPC weight update failed: {str(e)}"
            logger.error(error_msg)
            success, message = False, error_msg

        return success, message

    async def _unload_lora_adapter_locked(
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,
    ) -> UnloadLoRAAdapterReqOutput:
        assert self.lora_update_lock.locked(), (
            "self.lora_update_lock must be locked in order for self._unload_lora_adapter_locked() to be called"
        )

        # Unregister the LoRA adapter from the registry to stop new requests for this adapter
        # from being started.
        lora_id = await self.lora_registry.unregister(obj.lora_name)
        obj.lora_id = lora_id

        # Initiate the actual unloading operation at the backend processes only after all
        # ongoing requests using this LoRA adapter are finished.
        await self.lora_registry.wait_for_unload(lora_id)
        result = (await self.update_lora_adapter_communicator(obj))[0]

        return result

    async def load_lora_adapter(
        self: TokenizerManager,
        obj: LoadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert self.server_args.dp_size == 1, (
                "dp_size must be 1 for dynamic lora loading"
            )
            logger.info(
                "Start load Lora adapter. Lora name=%s, path=%s",
                obj.lora_name,
                obj.lora_path,
            )

            async with self.lora_update_lock:
                # Generate new uniquely identifiable LoRARef object.
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path=obj.lora_path,
                    pinned=obj.pinned,
                )

                # Trigger the actual loading operation at the backend processes.
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]

                # Register the LoRA adapter only after loading is successful.
                if result.success:
                    await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter

                if self.server_args.max_loaded_loras is not None:
                    while (
                        self.lora_registry.num_registered_loras
                        > self.server_args.max_loaded_loras
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True
                        )
                        if lru_lora_name is None:
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "
                                f"LoRA registry is: {self.lora_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "
                            f"max allowed: {self.server_args.max_loaded_loras})"
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_lora_name]

                return result
        except ValueError as e:
            return LoadLoRAAdapterReqOutput(
                success=False,
                error_message=str(e),
            )

    async def load_lora_adapter_from_tensors(
        self: TokenizerManager,
        obj: LoadLoRAAdapterFromTensorsReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert self.server_args.dp_size == 1, (
                "dp_size must be 1 for dynamic lora loading"
            )
            logger.info(
                "Start load Lora adapter from tensors. Lora name=%s",
                obj.lora_name,
            )

            async with self.lora_update_lock:
                new_adapter = LoRARef(
                    lora_name=obj.lora_name,
                    lora_path="__tensor__",
                    pinned=obj.pinned,
                )
                obj.lora_id = new_adapter.lora_id
                result = (await self.update_lora_adapter_communicator(obj))[0]

                if result.success:
                    await self.lora_registry.register(new_adapter)
                    self.lora_ref_cache[obj.lora_name] = new_adapter
                if self.server_args.max_loaded_loras is not None:
                    while (
                        self.lora_registry.num_registered_loras
                        > self.server_args.max_loaded_loras
                    ):
                        lru_lora_name = await self.lora_registry.lru_lora_name(
                            exclude_pinned=True
                        )
                        if lru_lora_name is None:
                            raise ValueError(
                                "Didn't find any LoRA adapters when trying to evict LRU LoRA adapter. "
                                f"LoRA registry is: {self.lora_registry._registry}"
                            )

                        logger.info(
                            f"Unloading least recently used LoRA adapter '{lru_lora_name}' "
                            f"(current number of adapters: {self.lora_registry.num_registered_loras}, "
                            f"max allowed: {self.server_args.max_loaded_loras})"
                        )

                        unload_result = await self._unload_lora_adapter_locked(
                            UnloadLoRAAdapterReqInput(lora_name=lru_lora_name)
                        )
                        if not unload_result.success:
                            raise ValueError(
                                f"Error while unloading LRU LoRA adapter '{lru_lora_name}': "
                                f"{unload_result.error_message}"
                            )
                        del result.loaded_adapters[lru_lora_name]

                return result
        except ValueError as e:
            return LoadLoRAAdapterFromTensorsReqOutput(
                success=False,
                error_message=str(e),
            )

    async def unload_lora_adapter(
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,
        _: Optional[fastapi.Request] = None,
    ) -> UnloadLoRAAdapterReqOutput:
        self.auto_create_handle_loop()

        try:
            if not self.server_args.enable_lora:
                raise ValueError(
                    "LoRA is not enabled. Please set `--enable-lora` to enable LoRA."
                )

            assert obj.lora_name is not None, (
                "lora_name must be provided to unload LoRA adapter"
            )

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert self.server_args.dp_size == 1, (
                "dp_size must be 1 for dynamic lora loading"
            )
            logger.info(
                "Start unload Lora adapter. Lora name=%s",
                obj.lora_name,
            )

            async with self.lora_update_lock:
                return await self._unload_lora_adapter_locked(obj)
        except ValueError as e:
            return UnloadLoRAAdapterReqOutput(success=False, error_message=str(e))

    async def get_weights_by_name(
        self: TokenizerManager,
        obj: GetWeightsByNameReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        results = await self.get_weights_by_name_communicator(obj)
        all_parameters = [r.parameter for r in results]
        if self.server_args.dp_size == 1:
            return all_parameters[0]
        else:
            return all_parameters

    async def release_memory_occupation(
        self: TokenizerManager,
        obj: ReleaseMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.release_memory_occupation_communicator(obj)

    async def resume_memory_occupation(
        self: TokenizerManager,
        obj: ResumeMemoryOccupationReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.resume_memory_occupation_communicator(obj)

    async def check_weights(
        self: TokenizerManager,
        obj: CheckWeightsReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str, Optional[List[Dict]], Optional[str]]:
        self.auto_create_handle_loop()
        results = await self.check_weights_communicator(obj)
        success, message = FanOutCommunicator.merge_results(results)
        ranks: Optional[List[Dict]] = None
        per_engine_checksum: Optional[str] = None
        if any(r.payload is not None for r in results):
            rank_infos: List[ChecksumInfo] = []
            for r in results:
                if r.payload is not None:
                    rank_infos.extend(r.payload)
            h = hashlib.sha256()
            for info in rank_infos:
                h.update(info.per_gpu_checksum.encode())
            per_engine_checksum = h.hexdigest()
            ranks = [msgspec_to_builtins(info) for info in rank_infos]
        return success, message, ranks, per_engine_checksum

    async def slow_down(
        self: TokenizerManager,
        obj: SlowDownReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        await self.slow_down_communicator(obj)

    async def get_internal_state(self: TokenizerManager) -> List[Dict[Any, Any]]:
        self.auto_create_handle_loop()
        req = GetInternalStateReq()
        responses: List[
            GetInternalStateReqOutput
        ] = await self.get_internal_state_communicator(req)
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def set_internal_state(
        self: TokenizerManager, obj: SetInternalStateReq
    ) -> List[bool]:
        self.auto_create_handle_loop()
        responses: List[
            SetInternalStateReqOutput
        ] = await self.set_internal_state_communicator(obj)
        return [res.updated for res in responses]

    async def dumper_control(
        self: TokenizerManager, obj: DumperControlReqInput
    ) -> List[DumperControlReqOutput]:
        self.auto_create_handle_loop()
        return await self.dumper_control_communicator(obj)

    async def get_loads(
        self: TokenizerManager,
        include: Optional[List[str]] = None,
        dp_rank: Optional[int] = None,
    ) -> List[LoadSnapshot]:
        """
        Get load snapshots for /v1/loads endpoint.

        Args:
            include: List of sections to include. Options: core, memory, spec, lora, disagg, queues, all
            dp_rank: Optional filter for specific DP rank

        Returns:
            List of LoadSnapshot, one per scheduler (filtered by dp_rank if specified)
        """
        self.auto_create_handle_loop()
        if dp_rank is not None and (
            dp_rank < 0 or dp_rank >= self.elastic_worker_count
        ):
            return []

        reader = self.load_snapshot_reader
        if dp_rank is not None:
            load = reader.read(dp_rank)
            results = [load] if load is not None else []
        else:
            results = reader.read_all()

        return results

    async def open_session(
        self: TokenizerManager,
        obj: OpenSessionReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        self.auto_create_handle_loop()
        if obj.streaming:
            if not self.server_args.enable_streaming_session:
                raise ValueError(
                    "Streaming sessions are disabled. "
                    "Please relaunch with --enable-streaming-session."
                )

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        future = asyncio.Future()
        self.session_futures[obj.session_id] = future
        self._dispatch_to_scheduler(obj)

        try:
            return await future
        finally:
            self.session_futures.pop(obj.session_id, None)

    async def close_session(
        self: TokenizerManager,
        obj: CloseSessionReqInput,
        request: Optional[fastapi.Request] = None,
    ):
        await self._async_dispatch_to_scheduler(obj)

    def _update_weight_version_if_provided(
        self: TokenizerManager, weight_version: Optional[str]
    ) -> None:
        """Update weight version if provided."""
        if weight_version is not None:
            self.runtime_weight_revision = weight_version
            self.server_args.override(
                "tokenizer.weight_version", weight_version=weight_version
            )
