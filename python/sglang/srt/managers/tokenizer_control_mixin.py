from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import fastapi

from sglang.srt.managers.communicator import FanOutCommunicator
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
    OpenSessionReqInput,
    PauseGenerationReqInput,
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
)
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang.srt.model_executor.weight_inventory_contracts import (
    DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC,
    validate_remote_instance_weight_transfer_lease_timeout,
    validate_remote_weight_lineage,
)
from sglang.srt.server_args import LoRARef, ServerArgs
from sglang.srt.utils import (
    get_bool_env_var,
    normalize_serialized_named_tensor_payloads,
)
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
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


_REMOTE_WEIGHT_TRANSFER_SESSION_INDEX_LIMIT = 4096


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


def _remote_weight_transfer_result_payloads(results) -> List:
    return [
        payload
        for result in results
        for payload in (getattr(result, "binding_inventories", None) or ())
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
    deadline_unix_sec: float | None,
    payloads,
    session_state: str,
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
        "deadline_unix_sec": deadline_unix_sec,
        "expired": session_state == "expired",
        "session_state": session_state,
        "last_release_attempt_unix_sec": existing.get("last_release_attempt_unix_sec"),
        "last_release_success": existing.get("last_release_success"),
        "last_release_message": existing.get("last_release_message"),
    }
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
) -> None:
    index = _remote_weight_transfer_session_index(manager)
    record = _refresh_remote_weight_transfer_session(manager, transfer_id)
    if record is None:
        record = {
            "transfer_id": transfer_id,
            "lease_id": None,
            "lease_ids": [],
            "generation": None,
            "deadline_unix_sec": None,
            "expired": False,
            "session_state": "release_failed",
        }
    record["last_release_attempt_unix_sec"] = attempted_at
    record["last_release_success"] = success
    record["last_release_message"] = message
    if success:
        record["session_state"] = "released"
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


def _merge_inventory_result_groups(results) -> tuple[list, list]:
    if len(results) != 1:
        raise RuntimeError(
            "remote heterogeneous weight reuse does not support DP source groups"
        )
    placements = results[0].placement_inventories
    bindings = results[0].binding_inventories
    if not placements or not bindings or len(placements) != len(bindings):
        raise RuntimeError(
            "source workers returned invalid placement/binding inventories"
        )
    return list(placements), list(bindings)


# Declarative spec: (attr_name_prefix, response_type[, mode, correlation_attr])
# Each entry creates self.{prefix}_communicator and registers
# response_type -> communicator.handle_recv in the dispatch table.
_COMMUNICATOR_SPECS = [
    ("init_weights_update_group", InitWeightsUpdateGroupReqOutput),
    ("destroy_weights_update_group", DestroyWeightsUpdateGroupReqOutput),
    ("update_weights_from_distributed", UpdateWeightsFromDistributedReqOutput),
    (
        "init_weights_send_group_for_remote_instance",
        InitWeightsSendGroupForRemoteInstanceReqOutput,
    ),
    ("send_weights_to_remote_instance", SendWeightsToRemoteInstanceReqOutput),
    (
        "begin_remote_instance_weight_transfer",
        BeginRemoteInstanceWeightTransferReqOutput,
        "queueing",
        "transfer_id",
    ),
    (
        "release_remote_instance_weight_transfer",
        ReleaseRemoteInstanceWeightTransferReqOutput,
        "queueing",
        "transfer_id",
    ),
    (
        "renew_remote_instance_weight_transfer",
        RenewRemoteInstanceWeightTransferReqOutput,
        "queueing",
        "transfer_id",
    ),
    ("update_weights_from_tensor", UpdateWeightsFromTensorReqOutput),
    ("update_weights_from_ipc", UpdateWeightsFromIPCReqOutput),
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

    @asynccontextmanager
    async def _remote_instance_weight_transfer_pause(self: TokenizerManager):
        owns_pause = not getattr(self, "is_pause", False)
        if owns_pause:
            await self.pause_generation(PauseGenerationReqInput(mode="in_place"))
        try:
            yield
        finally:
            if owns_pause:
                resume_task = asyncio.create_task(
                    self.continue_generation(
                        ContinueGenerationReqInput(torch_empty_cache=False)
                    )
                )
                await _finish_control_task(resume_task)

    def init_communicators(self: TokenizerManager, server_args: ServerArgs):
        dispatch_pairs = []
        for spec in _COMMUNICATOR_SPECS:
            name, resp_type = spec[0], spec[1]
            mode = spec[2] if len(spec) > 2 else "queueing"
            correlation_attr = spec[3] if len(spec) > 3 else None
            comm = FanOutCommunicator(
                self._dispatch_to_scheduler,
                server_args.dp_size,
                mode,
                correlation_attr,
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
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        results = await self.init_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def destroy_weights_update_group(
        self: TokenizerManager,
        obj: DestroyWeightsUpdateGroupReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for destroy parameter update group"

        results = await self.destroy_weights_update_group_communicator(obj)
        return FanOutCommunicator.merge_results(results)

    async def update_weights_from_distributed(
        self: TokenizerManager,
        obj: UpdateWeightsFromDistributedReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from distributed"

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        # Hold is_pause_cond while updating to prevent unpause from racing.
        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                results = await self.update_weights_from_distributed_communicator(obj)

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.update_weights_from_distributed_communicator(obj)

        success, message = FanOutCommunicator.merge_results(results)
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def init_weights_send_group_for_remote_instance(
        self: TokenizerManager,
        obj: InitWeightsSendGroupForRemoteInstanceReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        # TODO: support DP
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for init_weights_send_group_for_remote_instance"
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
        assert (
            self.server_args.dp_size == 1
        ), "dp_size must be 1 for send_weights_to_remote_instance"
        result = (await self.send_weights_to_remote_instance_communicator(obj))[0]
        return result.success, result.message

    async def begin_remote_instance_weight_transfer(
        self: TokenizerManager,
        lease_timeout_sec: int = (
            DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
        ),
        transfer_id: str | None = None,
    ) -> dict:
        """Pause for inventory capture, then serve while the lease is held."""
        if not self.server_args.enable_weight_reshard:
            raise RuntimeError(
                "remote heterogeneous weight reuse requires --enable-weight-reshard"
            )
        validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        if transfer_id is not None and (
            type(transfer_id) is not str or not transfer_id
        ):
            raise ValueError("transfer_id must be a non-empty string")

        self.auto_create_handle_loop()
        transfer_id = transfer_id or uuid.uuid4().hex
        async with _remote_weight_transfer_begin_lock(self):
            return await TokenizerControlMixin._begin_remote_instance_weight_transfer(
                self,
                lease_timeout_sec=lease_timeout_sec,
                transfer_id=transfer_id,
            )

    async def _begin_remote_instance_weight_transfer(
        self: TokenizerManager,
        *,
        lease_timeout_sec: int,
        transfer_id: str,
    ) -> dict:
        model_id, revision = validate_remote_weight_lineage(
            model_id=self.server_args.get_weight_reshard_resource_id(),
            revision=getattr(self.server_args, "revision", None),
        )
        request = BeginRemoteInstanceWeightTransferReqInput(
            transfer_id=transfer_id,
            model_id=model_id,
            revision=revision,
            lease_timeout_sec=lease_timeout_sec,
        )
        results = None
        deadline_unix_sec = None

        async def capture_inventories():
            nonlocal deadline_unix_sec, results
            async with TokenizerControlMixin._remote_instance_weight_transfer_pause(
                self
            ):
                results = await self.begin_remote_instance_weight_transfer_communicator(
                    request
                )
                deadline_unix_sec = time.time() + lease_timeout_sec
            return results

        capture_task = asyncio.create_task(capture_inventories())
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
            if cleanup_candidate or capture_error is not None:
                _remember_remote_weight_transfer_session(
                    self,
                    transfer_id=transfer_id,
                    deadline_unix_sec=(
                        deadline_unix_sec
                        if deadline_unix_sec is not None
                        else time.time() + lease_timeout_sec
                    ),
                    payloads=_remote_weight_transfer_result_payloads(results or ()),
                    session_state="cleanup_pending",
                )
            if cleanup_candidate:
                cleanup_task = asyncio.create_task(
                    TokenizerControlMixin.release_remote_instance_weight_transfer(
                        self, transfer_id
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
                raise
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
                    "Failed to resume source generation after inventory capture: "
                    f"{error}",
                    transfer_id=transfer_id,
                    session_state=session_state,
                ) from error
            _remember_remote_weight_transfer_session(
                self,
                transfer_id=transfer_id,
                deadline_unix_sec=deadline_unix_sec,
                payloads=_remote_weight_transfer_result_payloads(results),
                session_state="cleanup_pending",
            )
            raise RemoteInstanceWeightTransferBeginError(
                f"Failed to resume source generation after inventory capture: {error}",
                transfer_id=transfer_id,
                session_state="cleanup_pending",
            ) from error

        failures = [result.message for result in results if not result.success]
        placements = None
        bindings = None
        if not results:
            failures.append("source workers returned no transfer responses")
        elif not failures:
            try:
                placements, bindings = _merge_inventory_result_groups(results)
            except RuntimeError as error:
                failures.append(str(error))
        if failures:
            session_states = [
                getattr(result, "session_state", "unknown") for result in results
            ]
            cleanup_candidate = _remote_weight_transfer_created_by_request(results)
            cleanup_succeeded = False
            cleanup_cancellation = None
            if cleanup_candidate:
                for _ in range(3):
                    cleanup_task = asyncio.create_task(
                        TokenizerControlMixin.release_remote_instance_weight_transfer(
                            self,
                            transfer_id,
                        )
                    )
                    try:
                        cleanup_succeeded, _ = await asyncio.shield(cleanup_task)
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
                    deadline_unix_sec=deadline_unix_sec,
                    payloads=_remote_weight_transfer_result_payloads(results),
                    session_state=session_state,
                )
            if cleanup_cancellation is not None:
                raise cleanup_cancellation
            raise RemoteInstanceWeightTransferBeginError(
                " | ".join(failures),
                transfer_id=transfer_id,
                session_state=session_state,
            )

        assert placements is not None and bindings is not None
        reused = all(
            getattr(result, "session_state", "created") == "reused"
            for result in results
        )
        existing = _refresh_remote_weight_transfer_session(self, transfer_id)
        _remember_remote_weight_transfer_session(
            self,
            transfer_id=transfer_id,
            deadline_unix_sec=(
                existing.get("deadline_unix_sec")
                if reused and existing is not None
                else (None if reused else deadline_unix_sec)
            ),
            payloads=bindings,
            session_state="reused" if reused and existing is None else "active",
        )
        return {
            "transfer_id": transfer_id,
            "success": True,
            "message": "Success.",
            "session_state": "reused" if reused else "created",
            "placement_inventories": placements,
            "binding_inventories": bindings,
            "lease_timeout_sec": lease_timeout_sec,
        }

    async def release_remote_instance_weight_transfer(
        self: TokenizerManager, transfer_id: str
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        attempted_at = time.time()
        try:
            async with TokenizerControlMixin._remote_instance_weight_transfer_pause(
                self
            ):
                results = (
                    await self.release_remote_instance_weight_transfer_communicator(
                        ReleaseRemoteInstanceWeightTransferReqInput(
                            transfer_id=transfer_id
                        )
                    )
                )
            success, message = FanOutCommunicator.merge_results(results)
        except Exception as error:
            success, message = False, str(error)
        _record_remote_weight_transfer_release(
            self,
            transfer_id=transfer_id,
            attempted_at=attempted_at,
            success=success,
            message=message,
        )
        log = logger.info if success else logger.warning
        log(
            "Explicit remote weight transfer release: "
            "transfer_id=%s success=%s message=%s",
            transfer_id,
            success,
            message,
        )
        return success, message

    async def renew_remote_instance_weight_transfer(
        self: TokenizerManager,
        transfer_id: str,
        lease_timeout_sec: int = (
            DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
        ),
    ) -> Tuple[bool, str]:
        validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        self.auto_create_handle_loop()
        deadline_unix_sec = None
        try:
            async with TokenizerControlMixin._remote_instance_weight_transfer_pause(
                self
            ):
                results = await self.renew_remote_instance_weight_transfer_communicator(
                    RenewRemoteInstanceWeightTransferReqInput(
                        transfer_id=transfer_id,
                        lease_timeout_sec=lease_timeout_sec,
                    )
                )
                deadline_unix_sec = time.time() + lease_timeout_sec
            success, message = FanOutCommunicator.merge_results(results)
        except Exception as error:
            success, message = False, str(error)
        if success:
            assert deadline_unix_sec is not None
            _record_remote_weight_transfer_renewal(
                self,
                transfer_id=transfer_id,
                deadline_unix_sec=deadline_unix_sec,
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
        self: TokenizerManager, transfer_id: str
    ) -> Dict[str, Any] | None:
        if type(transfer_id) is not str or not transfer_id:
            raise ValueError("transfer_id must be a non-empty string")
        return _refresh_remote_weight_transfer_session(self, transfer_id)

    async def update_weights_from_tensor(
        self: TokenizerManager,
        obj: UpdateWeightsFromTensorReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        self.auto_create_handle_loop()
        assert (
            self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
        ), "dp_size must be 1 or dp attention must be enabled for update weights from tensor"

        if obj.abort_all_requests:
            self.abort_request(abort_all=True)

        obj.serialized_named_tensors = normalize_serialized_named_tensor_payloads(
            obj.serialized_named_tensors
        )

        async with self.is_pause_cond:
            is_paused = self.is_pause
            if is_paused:
                results = await self.update_weights_from_tensor_communicator(obj)

        if not is_paused:
            async with self.model_update_lock.writer_lock:
                results = await self.update_weights_from_tensor_communicator(obj)

        success, message = FanOutCommunicator.merge_results(results)
        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def update_weights_from_ipc(
        self: TokenizerManager,
        obj: UpdateWeightsFromIPCReqInput,
        request: Optional[fastapi.Request] = None,
    ) -> Tuple[bool, str]:
        """Update weights via IPC for checkpoint-engine integration."""
        self.auto_create_handle_loop()
        try:
            # For now, we only support single data parallel instance
            assert (
                self.server_args.dp_size == 1 or self.server_args.enable_dp_attention
            ), "dp_size must be 1 or dp attention must be enabled for update weights from IPC"
            logger.info("Starting IPC weight update")

            async with self.is_pause_cond:
                is_paused = self.is_pause
                if is_paused:
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]
                    success, message = result.success, result.message

            if not is_paused:
                async with self.model_update_lock.writer_lock:
                    result = (await self.update_weights_from_ipc_communicator(obj))[0]
                    success, message = result.success, result.message
        except Exception as e:
            error_msg = f"IPC weight update failed: {str(e)}"
            logger.error(error_msg)
            success, message = False, error_msg

        if success and obj.weight_version is not None:
            self._update_weight_version_if_provided(obj.weight_version)
            message += f" Weight version updated to {obj.weight_version}."

        return success, message

    async def _unload_lora_adapter_locked(
        self: TokenizerManager,
        obj: UnloadLoRAAdapterReqInput,
    ) -> UnloadLoRAAdapterReqOutput:
        assert (
            self.lora_update_lock.locked()
        ), "self.lora_update_lock must be locked in order for self._unload_lora_adapter_locked() to be called"

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
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
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

            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
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

            assert (
                obj.lora_name is not None
            ), "lora_name must be provided to unload LoRA adapter"

            # TODO (lifuhuang): Remove this after we verify that dynamic lora loading works
            # with dp_size > 1.
            assert (
                self.server_args.dp_size == 1
            ), "dp_size must be 1 for dynamic lora loading"
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
        responses: List[GetInternalStateReqOutput] = (
            await self.get_internal_state_communicator(req)
        )
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def set_internal_state(
        self: TokenizerManager, obj: SetInternalStateReq
    ) -> List[bool]:
        self.auto_create_handle_loop()
        responses: List[SetInternalStateReqOutput] = (
            await self.set_internal_state_communicator(obj)
        )
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
            self.server_args.override(
                "tokenizer.weight_version", weight_version=weight_version
            )
