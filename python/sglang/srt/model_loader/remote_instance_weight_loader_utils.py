# SPDX-License-Identifier: Apache-2.0

import enum
import importlib.util
import logging
import math
import threading
import time
import uuid
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, List

import requests

from sglang.srt.model_executor.weight_inventory_contracts import (
    validate_remote_instance_weight_transfer_lease_timeout,
)

logger = logging.getLogger(__name__)


class RemoteInstanceWeightLoaderBackend(str, enum.Enum):
    NCCL = "nccl"
    TRANSFER_ENGINE = "transfer_engine"
    MODELEXPRESS = "modelexpress"


@dataclass(frozen=True, slots=True)
class RemoteInstanceWeightTransferSession:
    transfer_id: str
    placement_inventories: list[dict]
    binding_inventories: list[dict]
    lease_timeout_sec: int

    def __post_init__(self) -> None:
        if type(self.transfer_id) is not str or not self.transfer_id:
            raise ValueError("transfer_id must be a non-empty string")
        if (
            type(self.placement_inventories) is not list
            or type(self.binding_inventories) is not list
            or not self.placement_inventories
            or len(self.placement_inventories) != len(self.binding_inventories)
            or not all(type(item) is dict for item in self.placement_inventories)
            or not all(type(item) is dict for item in self.binding_inventories)
        ):
            raise ValueError(
                "placement and binding inventories must be paired non-empty dict lists"
            )
        validate_remote_instance_weight_transfer_lease_timeout(self.lease_timeout_sec)


class RemoteInstanceWeightTransferHeartbeat:
    def __init__(
        self,
        seed_url: str,
        transfer_id: str,
        *,
        lease_timeout_sec: int,
        renew_interval_sec: float | None = None,
    ) -> None:
        if not seed_url or not transfer_id:
            raise ValueError("remote weight transfer identifiers must not be empty")
        if (
            isinstance(lease_timeout_sec, bool)
            or not isinstance(lease_timeout_sec, int)
            or lease_timeout_sec <= 0
        ):
            raise ValueError("lease_timeout_sec must be a positive integer")
        interval = (
            max(1.0, lease_timeout_sec / 3)
            if renew_interval_sec is None
            else renew_interval_sec
        )
        if interval <= 0 or interval >= lease_timeout_sec:
            raise ValueError(
                "renew_interval_sec must be positive and shorter than the lease"
            )
        self.seed_url = seed_url
        self.transfer_id = transfer_id
        self.lease_timeout_sec = lease_timeout_sec
        self.renew_interval_sec = interval
        self._stop_event = threading.Event()
        self._failure_lock = threading.Lock()
        self._failure: BaseException | None = None
        self._thread: threading.Thread | None = None
        self._lease_deadline = time.monotonic() + lease_timeout_sec

    def _renew(self) -> bool:
        remaining_lease_sec = self._lease_deadline - time.monotonic()
        if remaining_lease_sec <= 0:
            return False
        renewed = renew_remote_instance_weight_transfer(
            self.seed_url,
            self.transfer_id,
            self.lease_timeout_sec,
            remaining_lease_sec=remaining_lease_sec,
        )
        if renewed:
            self._lease_deadline = time.monotonic() + self.lease_timeout_sec
        return renewed

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("remote weight transfer heartbeat was already started")
        if not self._renew():
            raise RuntimeError("initial source weight transfer lease renew failed")
        self._thread = threading.Thread(
            target=self._run,
            name=f"weight-transfer-heartbeat-{self.transfer_id[:8]}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.renew_interval_sec):
            try:
                if not self._renew():
                    raise RuntimeError("source weight transfer lease renew failed")
            except BaseException as error:
                with self._failure_lock:
                    if self._failure is None:
                        self._failure = error
                self._stop_event.set()
                return

    def raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError(
                f"Remote weight transfer lease renewal failed: {failure}"
            ) from failure

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is None:
            return
        self._thread.join(timeout=35)
        if self._thread.is_alive():
            raise RuntimeError("remote weight transfer heartbeat did not stop")


def validate_remote_instance_weight_transfer_world_group(world_group: Any) -> int:
    world_size = getattr(world_group, "world_size", None)
    rank_in_group = getattr(world_group, "rank_in_group", None)
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("target world_group must expose a positive integer world_size")
    if (
        type(rank_in_group) is not int
        or rank_in_group < 0
        or rank_in_group >= world_size
    ):
        raise ValueError("target world_group must expose a valid integer rank_in_group")
    for method_name in ("all_gather_object", "broadcast_object"):
        if not callable(getattr(world_group, method_name, None)):
            raise ValueError(f"target world_group must expose {method_name}()")
    return world_size


class RemoteInstanceWeightTransferWorldCoordinator:
    """Share one source snapshot lease across the target model world."""

    def __init__(self, seed_url: str, world_group: Any) -> None:
        self.seed_url = seed_url
        self.world_group = world_group
        self.world_size = validate_remote_instance_weight_transfer_world_group(
            world_group
        )
        self.is_owner = world_group.rank_in_group == 0
        self.session: RemoteInstanceWeightTransferSession | None = None
        self.heartbeat: RemoteInstanceWeightTransferHeartbeat | None = None
        self._acquired = False
        self._finished = False
        self._readiness_checked = False
        self._release_safe = True

    def acquire(self) -> RemoteInstanceWeightTransferSession | None:
        if self._acquired:
            raise RuntimeError("remote weight transfer world session already acquired")
        self._acquired = True

        local_session = None
        if self.is_owner:
            try:
                local_session = begin_remote_instance_weight_transfer(self.seed_url)
            except Exception:
                logger.exception("Failed to acquire the source weight transfer session")
            if local_session is not None:
                try:
                    self.heartbeat = RemoteInstanceWeightTransferHeartbeat(
                        self.seed_url,
                        local_session.transfer_id,
                        lease_timeout_sec=local_session.lease_timeout_sec,
                    )
                    self.heartbeat.start()
                except Exception:
                    logger.exception("Failed to start remote weight transfer heartbeat")
                    _best_effort_release_invalid_transfer(
                        self.seed_url,
                        local_session.transfer_id,
                        attempts=3,
                    )
                    self.heartbeat = None
                    local_session = None

        try:
            self.session = self.world_group.broadcast_object(local_session, src=0)
            if self.session is not None and not isinstance(
                self.session, RemoteInstanceWeightTransferSession
            ):
                raise ValueError(
                    "target world broadcast an invalid weight transfer session"
                )
        except Exception:
            self._stop_heartbeat()
            if self.is_owner and local_session is not None:
                _best_effort_release_invalid_transfer(
                    self.seed_url,
                    local_session.transfer_id,
                    attempts=3,
                )
            logger.exception("Failed to broadcast the source weight transfer session")
            raise
        return self.session

    def raise_if_failed(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat.raise_if_failed()

    def ready_for_transfer(self, local_ready: bool) -> bool:
        """Run the single target-world gate before any rank starts DMA."""
        if type(local_ready) is not bool:
            raise TypeError("local_ready must be a bool")
        if not self._acquired:
            raise RuntimeError("remote weight transfer world session was not acquired")
        if self._finished:
            raise RuntimeError("remote weight transfer world session already finished")
        if self._readiness_checked:
            raise RuntimeError("remote weight transfer readiness was already checked")
        self._readiness_checked = True
        if self.session is None:
            return False

        ready = local_ready
        if self.heartbeat is not None:
            try:
                self.heartbeat.raise_if_failed()
            except Exception:
                ready = False
                logger.exception(
                    "Source lease heartbeat failed before the target-world "
                    "transfer readiness gate"
                )

        try:
            gathered_readiness = self.world_group.all_gather_object(ready)
        except Exception:
            self._release_safe = False
            logger.exception(
                "Failed to gather target transfer readiness; source mutation "
                "remains blocked until explicit release or recovery"
            )
            return False

        readiness_valid = len(gathered_readiness) == self.world_size and all(
            isinstance(value, bool) for value in gathered_readiness
        )
        if not readiness_valid:
            self._release_safe = False
            logger.error(
                "Target world returned invalid transfer readiness; source mutation "
                "remains blocked until explicit release or recovery"
            )
            return False
        return all(gathered_readiness)

    def finish(
        self,
        *,
        local_success: bool,
        local_release_safe: bool = True,
    ) -> tuple[bool, bool]:
        if type(local_success) is not bool or type(local_release_safe) is not bool:
            raise TypeError("local_success and local_release_safe must be bools")
        if not self._acquired:
            raise RuntimeError("remote weight transfer world session was not acquired")
        if self._finished:
            raise RuntimeError("remote weight transfer world session already finished")
        self._finished = True
        if self.session is None:
            return False, True

        local_release_safe = local_release_safe and self._release_safe

        if self.heartbeat is not None:
            try:
                self.heartbeat.raise_if_failed()
            except Exception:
                local_success = False
                logger.exception(
                    "Remote weight transfer heartbeat failed before world sync"
                )

        try:
            gathered_outcomes = self.world_group.all_gather_object(
                (local_success, local_release_safe)
            )
        except Exception:
            self._stop_heartbeat()
            logger.exception(
                "Failed to gather target transfer outcomes; source mutation "
                "remains blocked until explicit release or recovery"
            )
            return False, False

        outcomes_valid = len(gathered_outcomes) == self.world_size and all(
            isinstance(outcome, tuple)
            and len(outcome) == 2
            and isinstance(outcome[0], bool)
            and isinstance(outcome[1], bool)
            for outcome in gathered_outcomes
        )
        if not outcomes_valid:
            self._stop_heartbeat()
            logger.error(
                "Target world returned invalid transfer outcomes; source mutation "
                "remains blocked until explicit release or recovery"
            )
            return False, False

        release_safe = all(outcome[1] for outcome in gathered_outcomes)
        world_success = release_safe and all(
            outcome[0] for outcome in gathered_outcomes
        )
        release_success = True
        if self.is_owner:
            if not self._stop_heartbeat():
                world_success = False
            if release_safe:
                try:
                    release_success = release_remote_instance_weight_transfer(
                        self.seed_url, self.session.transfer_id
                    )
                    if not release_success:
                        logger.error(
                            "Failed to release source weight transfer %s; source "
                            "mutation remains blocked until explicit release or "
                            "recovery",
                            self.session.transfer_id,
                        )
                except Exception:
                    release_success = False
                    logger.exception(
                        "Failed to release source weight transfer %s; source "
                        "mutation remains blocked until explicit release or recovery",
                        self.session.transfer_id,
                    )
            else:
                release_success = False
                logger.error(
                    "Keeping source weight transfer %s leased because a target "
                    "rank could not confirm transfer completion; source mutation "
                    "remains blocked until explicit release or recovery",
                    self.session.transfer_id,
                )

        try:
            outcome = self.world_group.broadcast_object(
                (world_success, release_success) if self.is_owner else None,
                src=0,
            )
        except Exception:
            logger.exception("Failed to broadcast the target transfer outcome")
            return False, False
        if (
            not isinstance(outcome, tuple)
            or len(outcome) != 2
            or not all(isinstance(value, bool) for value in outcome)
        ):
            logger.error("Target world returned an invalid weight transfer outcome")
            return False, False
        return outcome

    def _stop_heartbeat(self) -> bool:
        if self.heartbeat is None:
            return True
        try:
            self.heartbeat.stop()
            self.heartbeat.raise_if_failed()
        except Exception:
            logger.exception("Remote weight transfer heartbeat failed while stopping")
            return False
        finally:
            self.heartbeat = None
        return True


def trigger_init_weights_send_group_for_remote_instance_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    # Only support loading weights from instance with same parallelism strategy.
    # Per TP rank pair between seed and dst instances will build a communication group for sending weights.
    # i.e. seed TP 0 <-> dst TP 0, seed TP 1 <-> dst TP 1, etc.
    # Each communication group will have a world size 2.
    try:
        requests.post(
            f"{seed_instance_service_url}/init_weights_send_group_for_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_rank": 0,
                "world_size": 2,
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
                "backend": "nccl",
            },
        )
    except Exception as e:
        logger.error(
            f"Failed to trigger init_weights_send_group_for_remote_instance_request to seed instance {seed_instance_service_url}: {e}."
        )
        raise


def trigger_transferring_weights_request(
    remote_instance_weight_loader_seed_instance_ip: str,
    remote_instance_weight_loader_seed_instance_service_port: int,
    remote_instance_weight_loader_send_weights_group_ports: List[int],
    remote_instance_weight_loader_client_id: str,
):
    seed_instance_service_url = f"http://{remote_instance_weight_loader_seed_instance_ip}:{remote_instance_weight_loader_seed_instance_service_port}"
    try:
        requests.post(
            f"{seed_instance_service_url}/send_weights_to_remote_instance",
            json={
                "master_address": remote_instance_weight_loader_seed_instance_ip,
                "ports": (
                    ",".join(
                        str(p)
                        for p in remote_instance_weight_loader_send_weights_group_ports
                    )
                ),
                "group_name": f"send_weights_{remote_instance_weight_loader_client_id}",
            },
        )
    except Exception as e:
        logger.error(f"Failed to trigger send weights to remote instance request: {e}")
        raise


def get_remote_instance_transfer_engine_info_per_rank(seed_url: str, rank: int):
    try:
        response = requests.get(
            f"{seed_url}/get_remote_instance_transfer_engine_info",
            params={
                "rank": rank,
            },
        )

        if response.status_code == 200:
            data = response.json()

            if "remote_instance_transfer_engine_info" in data:
                return data["remote_instance_transfer_engine_info"]
            else:
                logger.error(
                    "Failed to get `remote_instance_transfer_engine_info` in response."
                )
                return None, None
        else:
            logger.error(f"request.get failed: {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Exception: {e}")
        return None, None


def _best_effort_release_invalid_transfer(
    seed_url: str,
    transfer_id: str,
    *,
    attempts: int = 1,
) -> bool:
    for _ in range(attempts):
        try:
            if release_remote_instance_weight_transfer(seed_url, transfer_id):
                return True
        except Exception:
            logger.exception(
                "Failed to clean up invalid remote weight transfer %s",
                transfer_id,
            )
    logger.error(
        "Failed to clean up invalid remote weight transfer %s; source mutation "
        "remains blocked until explicit release or recovery",
        transfer_id,
    )
    return False


def begin_remote_instance_weight_transfer(
    seed_url: str,
    lease_timeout_sec: int = 300,
    *,
    transfer_id: str | None = None,
):
    if transfer_id is None:
        transfer_id = uuid.uuid4().hex
    if not isinstance(transfer_id, str) or not transfer_id:
        raise ValueError("transfer_id must be a non-empty string")

    for attempt in range(3):
        try:
            response = requests.post(
                f"{seed_url}/remote_instance_weight_transfer",
                params={
                    "lease_timeout_sec": lease_timeout_sec,
                    "transfer_id": transfer_id,
                },
                timeout=30,
            )
            if response.status_code != 200:
                try:
                    error_payload = response.json()
                    response_transfer_id = error_payload.get("transfer_id")
                    response_session_state = error_payload.get("session_state")
                except Exception:
                    response_transfer_id = None
                    response_session_state = None
                if response_transfer_id == transfer_id and response_session_state in {
                    "created",
                    "cleanup_pending",
                }:
                    _best_effort_release_invalid_transfer(
                        seed_url,
                        response_transfer_id,
                        attempts=3,
                    )
                logger.error(
                    "Failed to begin remote weight transfer: %s: %s",
                    response.status_code,
                    getattr(response, "text", ""),
                )
                return None

            payload = response.json()
            response_transfer_id = payload.get("transfer_id")
            if response_transfer_id != transfer_id:
                if response_transfer_id:
                    _best_effort_release_invalid_transfer(
                        seed_url,
                        response_transfer_id,
                    )
                _best_effort_release_invalid_transfer(
                    seed_url,
                    transfer_id,
                    attempts=3,
                )
                logger.error(
                    "Remote instance returned a different transfer ID: %r != %r",
                    response_transfer_id,
                    transfer_id,
                )
                return None

            allowed_payload_fields = {
                "transfer_id",
                "success",
                "message",
                "session_state",
                "placement_inventories",
                "binding_inventories",
                "lease_timeout_sec",
            }
            placements = payload.get("placement_inventories")
            bindings = payload.get("binding_inventories")
            server_lease_timeout_sec = payload.get(
                "lease_timeout_sec", lease_timeout_sec
            )
            try:
                validate_remote_instance_weight_transfer_lease_timeout(
                    server_lease_timeout_sec
                )
                lease_timeout_valid = True
            except ValueError:
                lease_timeout_valid = False
            payload_valid = (
                set(payload) == allowed_payload_fields
                and payload.get("success") is True
                and type(payload.get("message")) is str
                and payload.get("session_state") in {"created", "reused"}
                and isinstance(placements, list)
                and isinstance(bindings, list)
                and bool(placements)
                and len(placements) == len(bindings)
                and all(type(item) is dict for item in placements)
                and all(type(item) is dict for item in bindings)
                and lease_timeout_valid
            )
            if not payload_valid:
                _best_effort_release_invalid_transfer(
                    seed_url,
                    transfer_id,
                    attempts=3,
                )
                logger.error("Remote instance returned an invalid inventory session.")
                return None

            return RemoteInstanceWeightTransferSession(
                transfer_id=transfer_id,
                placement_inventories=placements,
                binding_inventories=bindings,
                lease_timeout_sec=server_lease_timeout_sec,
            )
        except Exception as error:
            logger.error("Failed to begin remote weight transfer: %s", error)
            if attempt + 1 < 3:
                continue
            _best_effort_release_invalid_transfer(
                seed_url,
                transfer_id,
                attempts=3,
            )
            return None

    return None


def release_remote_instance_weight_transfer(seed_url: str, transfer_id: str) -> bool:
    try:
        response = requests.delete(
            f"{seed_url}/remote_instance_weight_transfer/{transfer_id}",
            timeout=30,
        )
        if response.status_code == 200:
            return True
        logger.error(
            "Failed to release remote weight transfer %s: %s: %s",
            transfer_id,
            response.status_code,
            response.text,
        )
    except Exception as error:
        logger.error(
            "Failed to release remote weight transfer %s: %s",
            transfer_id,
            error,
        )
    return False


def renew_remote_instance_weight_transfer(
    seed_url: str,
    transfer_id: str,
    lease_timeout_sec: int,
    *,
    remaining_lease_sec: float | None = None,
) -> bool:
    remaining_lease_sec = (
        float(lease_timeout_sec)
        if remaining_lease_sec is None
        else float(remaining_lease_sec)
    )
    if not math.isfinite(remaining_lease_sec) or remaining_lease_sec <= 0:
        logger.error(
            "Cannot renew remote weight transfer %s after its lease window elapsed",
            transfer_id,
        )
        return False
    request_timeout_sec = min(30.0, remaining_lease_sec / 2)
    if request_timeout_sec <= 0:
        return False
    try:
        response = requests.post(
            f"{seed_url}/remote_instance_weight_transfer/{transfer_id}/renew",
            params={"lease_timeout_sec": lease_timeout_sec},
            timeout=request_timeout_sec,
        )
        if response.status_code == 200:
            return True
        logger.error(
            "Failed to renew remote weight transfer %s: %s: %s",
            transfer_id,
            response.status_code,
            response.text,
        )
    except Exception as error:
        logger.error(
            "Failed to renew remote weight transfer %s: %s",
            transfer_id,
            error,
        )
    return False


def register_memory_region(model, transfer_engine):
    if importlib.util.find_spec("torch") is None:
        return register_memory_region_v1(model, transfer_engine)
    else:
        return register_memory_region_v2(model, transfer_engine)


def register_memory_region_v1(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    for name, weight in model.named_parameters():
        ret = transfer_engine.register_memory(
            weight.data_ptr(), weight.numel() * weight.element_size()
        )
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight {name}, error: {ret}"
            )
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )

    end_tic = time.time()
    logger.debug(f"Register memory region time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict


def register_memory_region_v2(model, transfer_engine):
    start_tic = time.time()

    weight_mr_dict = {}
    weight_ranges = []
    for name, weight in model.named_parameters():
        address = int(weight.data_ptr())
        numel = int(weight.numel())
        itemsize = int(weight.element_size())
        nbytes = numel * itemsize
        if address <= 0 or nbytes <= 0:
            raise RuntimeError(f"weight {name} has no registerable storage")
        weight_mr_dict[name] = (address, numel, itemsize)
        weight_ranges.append((name, address, address + nbytes))

    import torch

    memory_snapshot = torch.cuda.memory.memory_snapshot()
    active_blocks = []
    for segment_index, segment in enumerate(memory_snapshot):
        for block in segment.get("blocks", []):
            address = block.get("address", -1)
            size = block.get("size", -1)
            if (
                type(address) is int
                and type(size) is int
                and address >= 0
                and size > 0
                and block.get("state") == "active_allocated"
            ):
                active_blocks.append((address, address + size, segment_index))
    active_blocks.sort()
    block_starts = [block[0] for block in active_blocks]
    selected_indices = set()
    for name, begin, end in weight_ranges:
        index = bisect_right(block_starts, begin) - 1
        if index < 0 or end > active_blocks[index][1]:
            raise RuntimeError(
                f"weight {name} is not covered by an active CUDA allocation"
            )
        selected_indices.add(index)

    weight_blocks_for_reg_mr = []
    for index in sorted(selected_indices):
        begin, end, segment_index = active_blocks[index]
        if (
            weight_blocks_for_reg_mr
            and weight_blocks_for_reg_mr[-1][1] == begin
            and weight_blocks_for_reg_mr[-1][2] == segment_index
        ):
            previous_begin, _, _ = weight_blocks_for_reg_mr[-1]
            weight_blocks_for_reg_mr[-1] = (
                previous_begin,
                end,
                segment_index,
            )
        else:
            weight_blocks_for_reg_mr.append((begin, end, segment_index))

    for begin, end, _ in weight_blocks_for_reg_mr:
        ret = transfer_engine.register_memory(begin, end - begin)
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight block at address {begin} "
                f"with size {end - begin}, error: {ret}"
            )

    end_tic = time.time()
    logger.debug(f"Register memory region v2 time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict
