# SPDX-License-Identifier: Apache-2.0

import enum
import importlib
import importlib.util
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, List

import requests

logger = logging.getLogger(__name__)


class RemoteInstanceWeightLoaderBackend(str, enum.Enum):
    NCCL = "nccl"
    TRANSFER_ENGINE = "transfer_engine"
    MODELEXPRESS = "modelexpress"


@dataclass(frozen=True, slots=True)
class RemoteInstanceWeightTransferSession:
    transfer_id: str
    manifests: list[dict]
    lease_timeout_sec: int


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

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("remote weight transfer heartbeat was already started")
        if not renew_remote_instance_weight_transfer(
            self.seed_url,
            self.transfer_id,
            self.lease_timeout_sec,
        ):
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
                renewed = renew_remote_instance_weight_transfer(
                    self.seed_url,
                    self.transfer_id,
                    self.lease_timeout_sec,
                )
                if not renewed:
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


class RemoteInstanceWeightTransferWorldCoordinator:
    """Share one source snapshot lease across the target model world."""

    def __init__(self, seed_url: str, world_group: Any) -> None:
        self.seed_url = seed_url
        self.world_group = world_group
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
                    try:
                        release_remote_instance_weight_transfer(
                            self.seed_url, local_session.transfer_id
                        )
                    except Exception:
                        logger.exception(
                            "Failed to clean up the source weight transfer after "
                            "heartbeat startup; waiting for lease TTL"
                        )
                    self.heartbeat = None
                    local_session = None

        try:
            self.session = self.world_group.broadcast_object(local_session, src=0)
        except Exception:
            self._stop_heartbeat()
            logger.exception(
                "Failed to broadcast the source weight transfer session; "
                "keeping its lease until TTL"
            )
            raise
        return self.session

    def raise_if_failed(self) -> None:
        if self.heartbeat is not None:
            self.heartbeat.raise_if_failed()

    def ready_for_transfer(self, local_ready: bool) -> bool:
        """Run the single target-world gate before any rank starts DMA."""
        if not self._acquired:
            raise RuntimeError("remote weight transfer world session was not acquired")
        if self._finished:
            raise RuntimeError("remote weight transfer world session already finished")
        if self._readiness_checked:
            raise RuntimeError("remote weight transfer readiness was already checked")
        self._readiness_checked = True
        if self.session is None:
            return False

        ready = bool(local_ready)
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
                "Failed to gather target transfer readiness; keeping the source "
                "lease until TTL"
            )
            return False

        readiness_valid = len(
            gathered_readiness
        ) == self.world_group.world_size and all(
            isinstance(value, bool) for value in gathered_readiness
        )
        if not readiness_valid:
            self._release_safe = False
            logger.error(
                "Target world returned invalid transfer readiness; keeping the "
                "source lease until TTL"
            )
            return False
        return all(gathered_readiness)

    def finish(
        self,
        *,
        local_success: bool,
        local_release_safe: bool = True,
    ) -> tuple[bool, bool]:
        if not self._acquired:
            raise RuntimeError("remote weight transfer world session was not acquired")
        if self._finished:
            raise RuntimeError("remote weight transfer world session already finished")
        self._finished = True
        if self.session is None:
            return False, True

        local_release_safe = bool(local_release_safe) and self._release_safe

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
                (bool(local_success), bool(local_release_safe))
            )
        except Exception:
            self._stop_heartbeat()
            logger.exception(
                "Failed to gather target transfer outcomes; keeping the source "
                "lease until TTL"
            )
            return False, False

        outcomes_valid = len(gathered_outcomes) == self.world_group.world_size and all(
            isinstance(outcome, tuple)
            and len(outcome) == 2
            and isinstance(outcome[0], bool)
            and isinstance(outcome[1], bool)
            for outcome in gathered_outcomes
        )
        if not outcomes_valid:
            self._stop_heartbeat()
            logger.error(
                "Target world returned invalid transfer outcomes; keeping the "
                "source lease until TTL"
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
                except Exception:
                    release_success = False
                    logger.exception(
                        "Failed to release source weight transfer %s; waiting for "
                        "lease TTL",
                        self.session.transfer_id,
                    )
            else:
                release_success = False
                logger.error(
                    "Keeping source weight transfer %s leased until TTL because "
                    "a target rank could not confirm transfer completion",
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


def begin_remote_instance_weight_transfer(seed_url: str, lease_timeout_sec: int = 300):
    try:
        response = requests.post(
            f"{seed_url}/remote_instance_weight_transfer",
            params={"lease_timeout_sec": lease_timeout_sec},
            timeout=30,
        )
        if response.status_code != 200:
            logger.error(
                "Failed to begin remote weight transfer: %s: %s",
                response.status_code,
                response.text,
            )
            return None
        payload = response.json()
        transfer_id = payload.get("transfer_id")
        manifests = payload.get("weight_runtime_manifests")
        server_lease_timeout_sec = payload.get("lease_timeout_sec", lease_timeout_sec)
        if (
            not transfer_id
            or not manifests
            or isinstance(server_lease_timeout_sec, bool)
            or not isinstance(server_lease_timeout_sec, int)
            or server_lease_timeout_sec <= 0
        ):
            logger.error("Remote instance returned an incomplete transfer session.")
            return None
        return RemoteInstanceWeightTransferSession(
            transfer_id=transfer_id,
            manifests=manifests,
            lease_timeout_sec=server_lease_timeout_sec,
        )
    except Exception as error:
        logger.error(f"Failed to begin remote weight transfer: {error}")
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
    seed_url: str, transfer_id: str, lease_timeout_sec: int
) -> bool:
    try:
        response = requests.post(
            f"{seed_url}/remote_instance_weight_transfer/{transfer_id}/renew",
            params={"lease_timeout_sec": lease_timeout_sec},
            timeout=30,
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
    weight_addr_set = set()
    for name, weight in model.named_parameters():
        weight_mr_dict[name] = (
            weight.data_ptr(),
            weight.numel(),
            weight.element_size(),
        )
        weight_addr_set.add(weight.data_ptr())

    import torch

    memory_snapshot = torch.cuda.memory.memory_snapshot()
    weight_blocks_for_reg_mr = []
    # Blocks in each segment have continuous physical addresses,
    # so they can be merged for memory registration.
    for segment in memory_snapshot:
        current_weight_block = None
        blocks = segment.get("blocks", [])
        for block in blocks:
            address = block.get("address", -1)
            size = block.get("size", -1)
            state = block.get("state", "")
            if address < 0 or size < 0 or state == "":
                continue
            # Only register active allocated memory blocks that hold weights.
            if state == "active_allocated":
                if address in weight_addr_set:
                    if current_weight_block is None:
                        current_weight_block = (address, size)
                    elif current_weight_block[0] + current_weight_block[1] == address:
                        current_weight_block = (
                            current_weight_block[0],
                            current_weight_block[1] + size,
                        )
                    else:
                        weight_blocks_for_reg_mr.append(current_weight_block)
                        current_weight_block = (address, size)
        if current_weight_block is not None:
            weight_blocks_for_reg_mr.append(current_weight_block)

    # Register merged memory blocks that hold weights.
    for weight_block in weight_blocks_for_reg_mr:
        address, size = weight_block
        ret = transfer_engine.register_memory(address, size)
        if ret != 0:
            raise RuntimeError(
                f"register memory failed for weight block at address {address} with size {size}, error: {ret}"
            )

    end_tic = time.time()
    logger.debug(f"Register memory region v2 time: {(end_tic - start_tic):.4f}s")
    return weight_mr_dict
