from __future__ import annotations

import hashlib
import logging
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.io_struct import (
    BeginRemoteInstanceWeightTransferReqInput,
    BeginRemoteInstanceWeightTransferReqOutput,
    ChecksumInfo,
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ReleaseRemoteInstanceWeightTransferReqInput,
    ReleaseRemoteInstanceWeightTransferReqOutput,
    RenewRemoteInstanceWeightTransferReqInput,
    RenewRemoteInstanceWeightTransferReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from sglang.srt.model_executor.weight_runtime_manifest import WeightManifestError

logger = logging.getLogger(__name__)


def _get_draft_model_runner(draft_worker):
    # DFlash / FrozenKVMTP workers expose draft_model_runner directly
    runner = getattr(draft_worker, "draft_model_runner", None)
    if runner is not None:
        return runner
    # EAGLEWorkerV2: _draft_worker.draft_runner
    inner = getattr(draft_worker, "_draft_worker", None)
    if inner is not None:
        runner = getattr(inner, "draft_runner", None)
        if runner is not None:
            return runner
    return None


def _merge_checksum_payloads(target: Dict, draft: Dict) -> Dict:
    merged_checksums = dict(target["checksums"])
    for name, chk in draft["checksums"].items():
        merged_checksums[f"draft.{name}"] = chk
    h = hashlib.sha256()
    for name in sorted(merged_checksums):
        h.update(name.encode())
        h.update(merged_checksums[name].encode())
    target["checksums"] = merged_checksums
    target["per_gpu_checksum"] = h.hexdigest()
    return target


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    world_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    remote_weight_transfer_cpu_group: Any = None
    scheduler: Optional[Any] = None
    metrics_collector: Optional[Any] = None
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
    remote_weight_transfer_leases: Dict[str, str] = field(default_factory=dict)
    remote_weight_transfer_deadlines: Dict[str, float] = field(default_factory=dict)
    remote_weight_transfer_expired: set[str] = field(default_factory=set)
    remote_weight_transfer_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    remote_weight_transfer_executor: Optional[ThreadPoolExecutor] = field(
        default=None, init=False, repr=False
    )
    remote_weight_transfer_pending: List[Tuple[Future, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    @contextmanager
    def _coordinate_weight_memory_transition(
        self,
        *,
        enabled: bool,
        commit_revision: bool,
    ) -> Iterator[None]:
        coordinator = getattr(
            self.tp_worker.model_runner,
            "weight_snapshot_coordinator",
            None,
        )
        if not enabled or coordinator is None:
            yield
            return

        token = None
        local_error = None
        try:
            token = coordinator.begin_update()
        except Exception as error:
            local_error = str(error)

        try:
            world_size = torch.distributed.get_world_size(group=self.world_cpu_group)
            gathered_errors = [None] * world_size
            torch.distributed.all_gather_object(
                gathered_errors,
                local_error,
                group=self.world_cpu_group,
            )
        except Exception as error:
            if token is not None:
                coordinator.cancel_update(token)
            raise WeightManifestError(
                f"failed to coordinate weight memory transition: {error}"
            ) from error

        failures = [
            f"rank {rank}: {error}"
            for rank, error in enumerate(gathered_errors)
            if error is not None
        ]
        if failures:
            if token is not None:
                coordinator.cancel_update(token)
            raise WeightManifestError(
                "weight memory transition reservation failed: " + " | ".join(failures)
            )

        assert token is not None
        success = False
        try:
            yield
            success = True
        finally:
            coordinator.finish_update(token, success=success)
            if success and commit_revision:
                coordinator.commit_revision()

    def _prune_remote_weight_transfer_bookkeeping(self) -> None:
        now = time.monotonic()
        with self.remote_weight_transfer_lock:
            expired = [
                transfer_id
                for transfer_id, deadline in self.remote_weight_transfer_deadlines.items()
                if deadline <= now
            ]
            for transfer_id in expired:
                self.remote_weight_transfer_deadlines.pop(transfer_id, None)
                self.remote_weight_transfer_expired.add(transfer_id)

    def _get_remote_weight_transfer_lease(self, transfer_id: str) -> str | None:
        self._prune_remote_weight_transfer_bookkeeping()
        with self.remote_weight_transfer_lock:
            return self.remote_weight_transfer_leases.get(transfer_id)

    def _record_remote_weight_transfer_lease(
        self,
        transfer_id: str,
        lease_id: str,
        lease_timeout_sec: int,
    ) -> None:
        with self.remote_weight_transfer_lock:
            self.remote_weight_transfer_leases[transfer_id] = lease_id
            self.remote_weight_transfer_deadlines[transfer_id] = (
                time.monotonic() + lease_timeout_sec
            )
            self.remote_weight_transfer_expired.discard(transfer_id)

    def _forget_remote_weight_transfer_lease(self, transfer_id: str) -> None:
        with self.remote_weight_transfer_lock:
            self.remote_weight_transfer_leases.pop(transfer_id, None)
            self.remote_weight_transfer_deadlines.pop(transfer_id, None)
            self.remote_weight_transfer_expired.discard(transfer_id)

    @contextmanager
    def _observe_weight_load(self, source: str) -> Iterator[None]:
        # Edge-trigger weight_load_duration_seconds at the end of each
        # update_weights_from_* call. Engine is paused during the update so
        # the periodic log_stats path can't carry this.
        # `source` distinguishes disk vs distributed vs tensor vs ipc.
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.metrics_collector is not None:
                self.metrics_collector.observe_weight_load(
                    time.perf_counter() - t0, source
                )

    def flush_cache_after_weight_update(self, recv_req) -> None:
        if recv_req.flush_cache:
            flush_cache_success = self.flush_cache(
                empty_cache=recv_req.torch_empty_cache
            )
            assert flush_cache_success, "Cache flush failed after updating weights"

    @staticmethod
    def _commit_weight_runtime_revision(worker) -> None:
        runner = getattr(worker, "model_runner", None)
        if getattr(runner, "weight_snapshot_coordinator", None) is None:
            return
        runner.commit_weight_runtime_revision()

    @staticmethod
    def _weight_snapshot_coordinator(worker):
        if worker is None:
            return None
        runner = getattr(worker, "model_runner", None)
        if runner is None:
            runner = _get_draft_model_runner(worker)
        return getattr(runner, "weight_snapshot_coordinator", None)

    def _weight_update_coordinators(self, workers) -> tuple:
        coordinators = []
        seen = set()
        for worker in workers:
            coordinator = self._weight_snapshot_coordinator(worker)
            if coordinator is None or id(coordinator) in seen:
                continue
            coordinators.append(coordinator)
            seen.add(id(coordinator))
        return tuple(coordinators)

    def _capture_weight_update_generations(self, workers) -> tuple:
        return tuple(
            (coordinator, coordinator.generation)
            for coordinator in self._weight_update_coordinators(workers)
        )

    @staticmethod
    def _validate_weight_update_generations(
        generation_mapping,
        *,
        require_pending_revision: bool,
    ) -> None:
        failures = []
        for coordinator, expected_generation in generation_mapping:
            current_generation = coordinator.generation
            if current_generation != expected_generation:
                failures.append(
                    "weight update generation changed from "
                    f"{expected_generation} to {current_generation}"
                )
                continue
            if require_pending_revision:
                pending_generation = coordinator.pending_revision_generation()
                if pending_generation != expected_generation:
                    failures.append(
                        "weight update generation "
                        f"{expected_generation} is not pending revision commit"
                    )
        if failures:
            raise WeightManifestError(" | ".join(failures))

    @staticmethod
    def _finalize_weight_update(generation_mapping, *, commit: bool) -> None:
        failures = []
        for coordinator, expected_generation in generation_mapping:
            try:
                if commit:
                    coordinator.commit_revision(expected_generation=expected_generation)
                else:
                    coordinator.poison_global_update_failure(
                        expected_generation=expected_generation
                    )
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")
        if failures:
            raise WeightManifestError(" | ".join(failures))

    @staticmethod
    def _poison_weight_update_best_effort(generation_mapping) -> None:
        for coordinator, expected_generation in generation_mapping:
            try:
                coordinator.poison_global_update_failure(
                    expected_generation=expected_generation
                )
            except Exception:
                logger.exception("Failed to poison a partial weight update")

    def _gather_weight_update_outcome(
        self,
        *,
        success: bool,
        message: str,
        phase: str,
    ) -> Tuple[bool, str]:
        local_outcome = {"success": bool(success), "message": str(message)}
        world_size = torch.distributed.get_world_size(group=self.world_cpu_group)
        gathered = [None] * world_size
        torch.distributed.all_gather_object(
            gathered,
            local_outcome,
            group=self.world_cpu_group,
        )

        failures = []
        for rank, outcome in enumerate(gathered):
            if (
                not isinstance(outcome, dict)
                or not isinstance(outcome.get("success"), bool)
                or not isinstance(outcome.get("message"), str)
            ):
                failures.append(f"rank {rank}: invalid {phase} outcome")
            elif not outcome["success"]:
                failures.append(f"rank {rank}: {outcome['message']}")
        if failures:
            return False, " | ".join(failures)
        return True, "Success."

    def _run_weight_update_transaction(
        self,
        *,
        operation: str,
        mutate: Callable[[], Tuple[bool, str]],
        workers,
        recv_req,
    ) -> Tuple[bool, str]:
        try:
            local_success, local_message = mutate()
            if not isinstance(local_success, bool):
                raise TypeError("weight update success outcome must be a boolean")
            local_message = str(local_message)
        except Exception as error:
            local_success = False
            local_message = f"{type(error).__name__}: {error}"

        try:
            generation_mapping = self._capture_weight_update_generations(workers)
        except Exception as error:
            generation_mapping = ()
            local_success = False
            local_message = (
                f"{local_message} | failed to capture weight update generation: "
                f"{type(error).__name__}: {error}"
            )

        if local_success:
            try:
                self.flush_cache_after_weight_update(recv_req)
            except Exception as error:
                local_success = False
                local_message = f"{type(error).__name__}: {error}"

        try:
            mutation_success, mutation_message = self._gather_weight_update_outcome(
                success=local_success,
                message=local_message,
                phase=f"{operation} mutation",
            )
        except Exception as error:
            self._poison_weight_update_best_effort(generation_mapping)
            return (
                False,
                f"Failed to gather {operation} mutation outcomes: "
                f"{type(error).__name__}: {error}",
            )

        try:
            self._validate_weight_update_generations(
                generation_mapping,
                require_pending_revision=mutation_success,
            )
            local_ready_success = True
            local_ready_message = "Success."
        except Exception as error:
            local_ready_success = False
            local_ready_message = f"{type(error).__name__}: {error}"

        try:
            ready_success, ready_message = self._gather_weight_update_outcome(
                success=local_ready_success,
                message=local_ready_message,
                phase=f"{operation} finalize readiness",
            )
        except Exception as error:
            self._poison_weight_update_best_effort(generation_mapping)
            return (
                False,
                f"Failed to gather {operation} finalize readiness outcomes: "
                f"{type(error).__name__}: {error}",
            )

        try:
            self._finalize_weight_update(
                generation_mapping,
                commit=mutation_success and ready_success,
            )
            local_finalize_success = True
            local_finalize_message = "Success."
        except Exception as error:
            local_finalize_success = False
            local_finalize_message = f"{type(error).__name__}: {error}"

        try:
            finalize_success, finalize_message = self._gather_weight_update_outcome(
                success=local_finalize_success,
                message=local_finalize_message,
                phase=f"{operation} finalize",
            )
        except Exception as error:
            self._poison_weight_update_best_effort(generation_mapping)
            return (
                False,
                f"Failed to gather {operation} finalize outcomes: "
                f"{type(error).__name__}: {error}",
            )

        if not finalize_success:
            self._poison_weight_update_best_effort(generation_mapping)
        if not mutation_success:
            return False, mutation_message
        if not ready_success:
            return False, ready_message
        if not finalize_success:
            return False, finalize_message
        return True, local_message

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        """In-place update of the weights from disk."""
        with self._observe_weight_load("disk"):
            workers = [self.tp_worker]
            if self.draft_worker is not None:
                workers.append(self.draft_worker)

            def mutate():
                success, message = self.tp_worker.update_weights_from_disk(recv_req)
                if success and self.draft_worker is not None:
                    success, message = self.draft_worker.update_weights_from_disk(
                        recv_req
                    )
                return success, message

            success, message = self._run_weight_update_transaction(
                operation="disk weight update",
                mutate=mutate,
                workers=workers,
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightFromDiskReqOutput(
                success=success, message=message, num_paused_requests=0
            )

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success=success, message=message)

    def destroy_weights_update_group(
        self,
        recv_req: DestroyWeightsUpdateGroupReqInput,
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success=success, message=message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        with self._observe_weight_load("distributed"):
            success, message = self._run_weight_update_transaction(
                operation="distributed weight update",
                mutate=lambda: self.tp_worker.update_weights_from_distributed(recv_req),
                workers=(self.tp_worker,),
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightsFromDistributedReqOutput(
                success=success, message=message
            )

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        """Update the online model parameter from tensors."""
        with self._observe_weight_load("tensor"):
            if recv_req.disable_draft_model:
                worker = self.tp_worker
            else:
                worker = self.draft_worker or self.tp_worker
            success, message = self._run_weight_update_transaction(
                operation="tensor weight update",
                mutate=lambda: worker.update_weights_from_tensor(recv_req),
                workers=(worker,),
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightsFromTensorReqOutput(success=success, message=message)

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        with self._observe_weight_load("ipc"):
            workers = [self.tp_worker]
            if self.draft_worker is not None:
                workers.append(self.draft_worker)

            def mutate():
                success, message = self.tp_worker.update_weights_from_ipc(recv_req)
                if success and self.draft_worker is not None:
                    success, message = self.draft_worker.update_weights_from_ipc(
                        recv_req
                    )
                return success, message

            success, message = self._run_weight_update_transaction(
                operation="IPC weight update",
                mutate=mutate,
                workers=workers,
                recv_req=recv_req,
            )
            if not success:
                logger.error(message)
            return UpdateWeightsFromIPCReqOutput(success=success, message=message)

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter=parameter)

    def _defer_remote_instance_weight_transfer(self, operation, recv_req) -> None:
        if self.remote_weight_transfer_executor is None:
            self.remote_weight_transfer_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sglang-weight-transfer",
            )
        future = self.remote_weight_transfer_executor.submit(operation, recv_req)
        self.remote_weight_transfer_pending.append((future, recv_req))

    def defer_begin_remote_instance_weight_transfer(
        self, recv_req: BeginRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.begin_remote_instance_weight_transfer, recv_req
        )

    def defer_release_remote_instance_weight_transfer(
        self, recv_req: ReleaseRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.release_remote_instance_weight_transfer, recv_req
        )

    def defer_renew_remote_instance_weight_transfer(
        self, recv_req: RenewRemoteInstanceWeightTransferReqInput
    ) -> None:
        self._defer_remote_instance_weight_transfer(
            self.renew_remote_instance_weight_transfer, recv_req
        )

    @staticmethod
    def _remote_instance_weight_transfer_failure(recv_req, error: Exception):
        kwargs = {
            "transfer_id": recv_req.transfer_id,
            "success": False,
            "message": str(error),
        }
        if isinstance(recv_req, BeginRemoteInstanceWeightTransferReqInput):
            return BeginRemoteInstanceWeightTransferReqOutput(**kwargs)
        if isinstance(recv_req, ReleaseRemoteInstanceWeightTransferReqInput):
            return ReleaseRemoteInstanceWeightTransferReqOutput(**kwargs)
        return RenewRemoteInstanceWeightTransferReqOutput(**kwargs)

    def check_pending_remote_instance_weight_transfers(self):
        self._prune_remote_weight_transfer_bookkeeping()
        completed = []
        remaining = []
        for future, recv_req in self.remote_weight_transfer_pending:
            if not future.done():
                remaining.append((future, recv_req))
                continue
            try:
                output = future.result()
            except Exception as error:
                logger.exception("Remote instance weight transfer control failed")
                output = self._remote_instance_weight_transfer_failure(recv_req, error)
            completed.append((output, recv_req))
        self.remote_weight_transfer_pending = remaining
        return completed

    def close_remote_instance_weight_transfer_executor(self) -> None:
        if self.remote_weight_transfer_executor is None:
            return
        self.remote_weight_transfer_executor.shutdown(wait=True)
        self.remote_weight_transfer_executor = None

    def begin_remote_instance_weight_transfer(
        self, recv_req: BeginRemoteInstanceWeightTransferReqInput
    ) -> BeginRemoteInstanceWeightTransferReqOutput:
        """Acquire one address-stable snapshot on every model rank."""
        collective_group = self.remote_weight_transfer_cpu_group or self.world_cpu_group
        local_snapshot = None
        split_manifest = recv_req.manifest_format == "placement_binding_v1"
        try:
            if recv_req.manifest_format not in ("runtime_v1", "placement_binding_v1"):
                raise RuntimeError(
                    f"unsupported source manifest format: {recv_req.manifest_format}"
                )
            if self._get_remote_weight_transfer_lease(recv_req.transfer_id) is not None:
                raise RuntimeError(
                    f"remote weight transfer already exists: {recv_req.transfer_id}"
                )
            if split_manifest:
                local_snapshot = self.tp_worker.model_runner.get_remote_instance_weight_runtime_manifest_parts(
                    model_id=recv_req.model_id,
                    revision=recv_req.revision,
                    transfer_id=recv_req.transfer_id,
                    lease_timeout_sec=recv_req.lease_timeout_sec,
                )
                placement = (
                    local_snapshot["placement"]
                    if isinstance(local_snapshot, dict)
                    else local_snapshot.placement
                )
                binding = (
                    local_snapshot["binding"]
                    if isinstance(local_snapshot, dict)
                    else local_snapshot.binding
                )
                placement_payload = (
                    placement
                    if isinstance(placement, dict)
                    else msgspec.to_builtins(placement)
                )
                binding_payload = (
                    binding
                    if isinstance(binding, dict)
                    else msgspec.to_builtins(binding)
                )
                local_result = {
                    "success": True,
                    "message": "Success.",
                    "placement": placement_payload,
                    "binding": binding_payload,
                }
            else:
                local_snapshot = self.tp_worker.model_runner.get_remote_instance_weight_runtime_manifest(
                    model_id=recv_req.model_id,
                    revision=recv_req.revision,
                    transfer_id=recv_req.transfer_id,
                    lease_timeout_sec=recv_req.lease_timeout_sec,
                )
                local_payload = (
                    local_snapshot
                    if isinstance(local_snapshot, dict)
                    else msgspec.to_builtins(local_snapshot)
                )
                local_result = {
                    "success": True,
                    "message": "Success.",
                    "manifest": local_payload,
                }
        except Exception as error:
            local_result = {
                "success": False,
                "message": str(error),
            }

        try:
            world_size = torch.distributed.get_world_size(group=collective_group)
            gathered = [None] * world_size
            torch.distributed.all_gather_object(
                gathered, local_result, group=collective_group
            )
        except Exception as error:
            if local_snapshot is not None:
                self.tp_worker.model_runner.release_weight_runtime_manifest(
                    self._remote_transfer_snapshot_lease_id(
                        local_snapshot, split_manifest=split_manifest
                    )
                )
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=f"Failed to gather source runtime manifests: {error}",
            )

        failures = [item["message"] for item in gathered if not item["success"]]
        manifests = None
        placements = None
        bindings = None
        if not failures:
            try:
                if split_manifest:
                    placements = [item["placement"] for item in gathered]
                    bindings = [item["binding"] for item in gathered]
                    self._validate_remote_transfer_parts(
                        placements, bindings, world_size
                    )
                else:
                    manifests = [item["manifest"] for item in gathered]
                    self._validate_remote_transfer_manifests(manifests, world_size)
            except Exception as error:
                failures.append(str(error))

        if failures:
            if local_snapshot is not None:
                self.tp_worker.model_runner.release_weight_runtime_manifest(
                    self._remote_transfer_snapshot_lease_id(
                        local_snapshot, split_manifest=split_manifest
                    )
                )
            return BeginRemoteInstanceWeightTransferReqOutput(
                transfer_id=recv_req.transfer_id,
                success=False,
                message=" | ".join(failures),
            )

        local_lease_id = self._remote_transfer_snapshot_lease_id(
            local_snapshot, split_manifest=split_manifest
        )
        self._record_remote_weight_transfer_lease(
            recv_req.transfer_id,
            local_lease_id,
            recv_req.lease_timeout_sec,
        )
        return BeginRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=True,
            message="Success.",
            manifests=manifests,
            placements=placements,
            bindings=bindings,
        )

    @staticmethod
    def _remote_transfer_snapshot_lease_id(snapshot, *, split_manifest: bool) -> str:
        if split_manifest:
            binding = (
                snapshot["binding"] if isinstance(snapshot, dict) else snapshot.binding
            )
            return (
                binding["lease_id"] if isinstance(binding, dict) else binding.lease_id
            )
        return snapshot["lease_id"] if isinstance(snapshot, dict) else snapshot.lease_id

    def _gather_remote_weight_transfer_status(
        self, *, success: bool, message: str, operation: str
    ) -> Tuple[bool, str]:
        local_result = {"success": success, "message": message}
        collective_group = self.remote_weight_transfer_cpu_group or self.world_cpu_group
        try:
            world_size = torch.distributed.get_world_size(group=collective_group)
            gathered = [None] * world_size
            torch.distributed.all_gather_object(
                gathered, local_result, group=collective_group
            )
        except Exception as error:
            return False, f"Failed to gather source {operation} results: {error}"

        failures = [item["message"] for item in gathered if not item["success"]]
        if failures:
            return False, " | ".join(failures)
        return True, "Success."

    def renew_remote_instance_weight_transfer(
        self, recv_req: RenewRemoteInstanceWeightTransferReqInput
    ) -> RenewRemoteInstanceWeightTransferReqOutput:
        lease_id = self._get_remote_weight_transfer_lease(recv_req.transfer_id)
        if lease_id is None:
            local_success = False
            local_message = "Remote weight transfer does not exist or has expired."
        else:
            try:
                self.tp_worker.model_runner.renew_weight_runtime_manifest(
                    lease_id,
                    lease_timeout_sec=recv_req.lease_timeout_sec,
                )
                self._record_remote_weight_transfer_lease(
                    recv_req.transfer_id,
                    lease_id,
                    recv_req.lease_timeout_sec,
                )
                local_success = True
                local_message = "Success."
            except Exception as error:
                local_success = False
                local_message = str(error)

        success, message = self._gather_remote_weight_transfer_status(
            success=local_success,
            message=local_message,
            operation="renewal",
        )
        return RenewRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=success,
            message=message,
        )

    @staticmethod
    def _validate_remote_transfer_manifests(manifests, world_size: int) -> None:
        if len(manifests) != world_size:
            raise RuntimeError(
                f"expected {world_size} source manifests, got {len(manifests)}"
            )
        if any(not manifest.get("tensors") for manifest in manifests):
            raise RuntimeError("every source rank must publish at least one tensor")

        worker_ids = {
            tensor["worker_id"]
            for manifest in manifests
            for tensor in manifest["tensors"][:1]
        }
        if len(worker_ids) != world_size:
            raise RuntimeError("source runtime manifest worker IDs are not unique")

        identities = {
            (
                manifest.get("model_id"),
                manifest.get("revision"),
                manifest.get("generation"),
            )
            for manifest in manifests
        }
        if len(identities) != 1:
            raise RuntimeError(
                "source runtime manifests do not describe one model generation"
            )

    @staticmethod
    def _validate_remote_transfer_parts(placements, bindings, world_size: int) -> None:
        if len(placements) != world_size or len(bindings) != world_size:
            raise RuntimeError(
                "source placement and binding counts must match the model world"
            )
        if any(not placement.get("tensors") for placement in placements):
            raise RuntimeError("every source rank must publish at least one tensor")
        if any(not binding.get("fragments") for binding in bindings):
            raise RuntimeError("every source rank must publish at least one binding")

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
                raise RuntimeError(
                    "source runtime binding does not match its placement"
                )

        worker_ids = []
        for binding in bindings:
            fragment_worker_ids = {
                fragment.get("worker_id") for fragment in binding["fragments"]
            }
            if None in fragment_worker_ids or len(fragment_worker_ids) != 1:
                raise RuntimeError(
                    "each source runtime binding must describe exactly one worker"
                )
            worker_ids.append(next(iter(fragment_worker_ids)))
        if len(set(worker_ids)) != world_size:
            raise RuntimeError("source runtime binding worker IDs are not unique")

        identities = {
            (
                placement.get("model_id"),
                placement.get("revision"),
                binding.get("generation"),
            )
            for placement, binding in zip(placements, bindings)
        }
        if len(identities) != 1:
            raise RuntimeError(
                "source placement and bindings do not describe one model generation"
            )

    def release_remote_instance_weight_transfer(
        self, recv_req: ReleaseRemoteInstanceWeightTransferReqInput
    ) -> ReleaseRemoteInstanceWeightTransferReqOutput:
        lease_id = self._get_remote_weight_transfer_lease(recv_req.transfer_id)
        if lease_id is None:
            local_success = True
            local_message = "Remote weight transfer was already released."
        else:
            try:
                self.tp_worker.model_runner.release_weight_runtime_manifest(lease_id)
                self._forget_remote_weight_transfer_lease(recv_req.transfer_id)
                local_success = True
                local_message = "Success."
            except Exception as error:
                local_success = False
                local_message = str(error)

        success, message = self._gather_remote_weight_transfer_status(
            success=local_success,
            message=local_message,
            operation="release",
        )
        return ReleaseRemoteInstanceWeightTransferReqOutput(
            transfer_id=recv_req.transfer_id,
            success=success,
            message=message,
        )

    def release_memory_occupation(self, recv_req: ReleaseMemoryOccupationReqInput):
        assert self.is_fully_idle(), (
            "release_memory_occupation should be called only when server is idle."
        )

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        with self._coordinate_weight_memory_transition(
            enabled=GPU_MEMORY_TYPE_WEIGHTS in tags,
            commit_revision=False,
        ):
            for tag in tags:
                self.offload_tags.add(tag)

            if GPU_MEMORY_TYPE_KV_CACHE in tags:
                scheduler = self.scheduler
                if scheduler is not None:
                    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                        for queue_name in (
                            "disagg_decode_transfer_queue",
                            "disagg_decode_prealloc_queue",
                        ):
                            queue = getattr(scheduler, queue_name, None)
                            if queue is not None:
                                queue.release_memory_occupation()
                    elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                        queue = getattr(
                            scheduler, "disagg_prefill_bootstrap_queue", None
                        )
                        if queue is not None:
                            queue.release_memory_occupation()
                self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
                self.flush_cache()

            if GPU_MEMORY_TYPE_WEIGHTS in tags:
                self.stashed_model_static_state = _export_static_state(
                    self.tp_worker.model_runner.model
                )
                torch.distributed.barrier(self.tp_cpu_group)
                self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

            if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
                self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

            torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(self, recv_req: ResumeMemoryOccupationReqInput):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        with self._coordinate_weight_memory_transition(
            enabled=GPU_MEMORY_TYPE_WEIGHTS in tags,
            commit_revision=True,
        ):
            for tag in tags:
                self.offload_tags.remove(tag)

            if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
                self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

            if GPU_MEMORY_TYPE_WEIGHTS in tags:
                self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
                torch.distributed.barrier(self.tp_cpu_group)
                _import_static_state(
                    self.tp_worker.model_runner.model,
                    self.stashed_model_static_state,
                )
                del self.stashed_model_static_state

            if GPU_MEMORY_TYPE_KV_CACHE in tags:
                self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)
                scheduler = self.scheduler
                if scheduler is not None:
                    if scheduler.disaggregation_mode == DisaggregationMode.DECODE:
                        for queue_name in (
                            "disagg_decode_transfer_queue",
                            "disagg_decode_prealloc_queue",
                        ):
                            queue = getattr(scheduler, queue_name, None)
                            if queue is not None:
                                queue.resume_memory_occupation()
                    elif scheduler.disaggregation_mode == DisaggregationMode.PREFILL:
                        queue = getattr(
                            scheduler, "disagg_prefill_bootstrap_queue", None
                        )
                        if queue is not None:
                            queue.resume_memory_occupation()

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self, recv_req: CheckWeightsReqInput):
        try:
            payload = self.tp_worker.model_runner.check_weights(
                action=recv_req.action, allow_quant_error=recv_req.allow_quant_error
            )

            if self.draft_worker is not None:
                draft_runner = _get_draft_model_runner(self.draft_worker)
                if draft_runner is not None:
                    draft_payload = draft_runner.check_weights(
                        action=recv_req.action,
                        allow_quant_error=recv_req.allow_quant_error,
                    )
                    if payload is not None and draft_payload is not None:
                        payload = _merge_checksum_payloads(payload, draft_payload)

            tp_size = torch.distributed.get_world_size(group=self.tp_cpu_group)
            if tp_size > 1 and payload is not None:
                all_payloads = [None] * tp_size
                torch.distributed.all_gather_object(
                    all_payloads, payload, group=self.tp_cpu_group
                )
                payload = all_payloads
            if payload is not None:
                # Normalize to one ChecksumInfo per rank so the wire shape is a
                # uniform List[ChecksumInfo] (tp==1 becomes a single-element list).
                per_rank = payload if isinstance(payload, list) else [payload]
                payload = [msgspec.convert(p, ChecksumInfo) for p in per_rank]
            return CheckWeightsReqOutput(
                success=True, message="Success.", payload=payload
            )
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self, params):
        url = params["url"]

        self.tp_worker.model_runner.weight_exporter.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert draft_url is not None, (
                "draft_url must be provided when draft model is enabled"
            )
            self.draft_worker.model_runner.weight_exporter.save_remote_model(draft_url)

    def save_sharded_model(self, params):
        self.tp_worker.model_runner.weight_exporter.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    with torch.inference_mode():
        self_named_buffers = dict(model.named_buffers())
        for name, tensor in static_params["buffers"]:
            self_named_buffers[name][...] = tensor
