# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Mixin classes and utils for multi-http-worker mode
This file uses multiple processes to handle requests and tokenization, reducing the overhead of python and http server.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as multiprocessing
import os
import pickle
import signal
import sys
import threading
import time
import uuid
import zlib
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import psutil
import setproctitle
import zmq
import zmq.asyncio
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.disagg_service import start_disagg_service
from sglang.srt.managers.io_struct import (
    BaseBatchReq,
    BaseReq,
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
    ContinueGenerationReqInput,
    FreezeGCReq,
    PauseContinueBroadcastReq,
    PauseGenerationReqInput,
    TokenizerWorkerRegistrationReq,
    async_sock_recv,
    async_sock_send,
    sock_recv,
    sock_send,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.load_snapshot import (
    create_load_snapshot_reader,
    zmq_reader_owner,
)
from sglang.srt.managers.tokenizer_control_mixin import (
    _ADMIN_PAUSE_OWNER,
    _REMOTE_WEIGHT_TRANSFER_PAUSE_OWNER_PREFIX,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    kill_itself_when_parent_died,
    kill_process_tree,
)
from sglang.srt.utils.network import get_zmq_socket
from sglang.utils import get_exception_traceback

if TYPE_CHECKING:
    from sglang.srt.managers.detokenizer_manager import DetokenizerManager

logger = logging.getLogger(__name__)

_PAUSE_CONTINUE_ACK_TIMEOUT_SEC = 30.0
_PAUSE_TRANSITION_PREFIX = "pause-transition-v1|"
_PAUSE_TRANSITION_APPLIED_PREFIX = "pause-transition-applied-v1|"
_PAUSE_TRANSITION_COMMITTED_ACK_PREFIX = "pause-transition-committed-v1|"
_PAUSE_TRANSITION_FINALIZED_ACK_PREFIX = "pause-transition-finalized-v1|"
_PAUSE_TRANSITION_CONFIRMED = "__pause_transition_confirmed__"
_PAUSE_TRANSITION_COMMITTED = "__pause_transition_committed__"
_PAUSE_TRANSITION_FINALIZED = "__pause_transition_finalized__"
_PAUSE_TRANSITION_FAILED = "__pause_transition_failed__"
_PAUSE_TRANSITION_RETRY_INTERVAL_SEC = 0.05
_PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC = 5.0
_WORKER_REGISTRATION_PREFIX = "tokenizer-worker-registration-v1|"
_WORKER_REGISTRATION_ACCEPTED = "__worker_registration_accepted__"
_WORKER_REGISTRATION_REJECTED = "__worker_registration_rejected__"
_WORKER_REGISTRATION_RETRY_INTERVAL_SEC = 0.05


@dataclass(frozen=True)
class _PauseTransitionIdentity:
    transition_id: str
    owner: str
    action: str
    expected_state: bool
    deadline_monotonic_ns: int


@dataclass
class _RouterPauseTransition:
    identity: _PauseTransitionIdentity
    origin_worker_ipc: str
    origin_worker: _TokenizerWorkerIdentity
    expected_workers: frozenset[str]
    acked_workers: set[str] = field(default_factory=set)
    applied_workers: set[str] = field(default_factory=set)
    confirmation_sent: bool = False
    scheduler_resume_request: Optional[ContinueGenerationReqInput] = None
    commit_started: bool = False
    committed: bool = False
    commit_pending_workers: set[str] = field(default_factory=set)
    commit_deadline_monotonic_ns: Optional[int] = None
    commit_retry_handle: Optional[asyncio.TimerHandle] = None
    commit_task: Optional[asyncio.Task] = None
    commit_done: Optional[asyncio.Future] = None
    timeout_handle: Optional[asyncio.TimerHandle] = None
    scheduler_pause_dispatched: bool = False


@dataclass(frozen=True)
class _TokenizerWorkerIdentity:
    ipc_name: str
    pid: int
    process_start_time: float
    token: str


def _new_pause_transition_identity(
    *,
    owner: str,
    action: str,
    expected_state: bool,
) -> _PauseTransitionIdentity:
    if action not in {"pause", "continue"}:
        raise ValueError(f"invalid pause transition action: {action}")
    return _PauseTransitionIdentity(
        transition_id=uuid.uuid4().hex,
        owner=owner,
        action=action,
        expected_state=expected_state,
        deadline_monotonic_ns=time.monotonic_ns()
        + int(_PAUSE_CONTINUE_ACK_TIMEOUT_SEC * 1_000_000_000),
    )


def _encode_pause_transition(identity: _PauseTransitionIdentity) -> str:
    return (
        f"{_PAUSE_TRANSITION_PREFIX}{identity.transition_id}|{identity.action}|"
        f"{int(identity.expected_state)}|{identity.deadline_monotonic_ns}|"
        f"{identity.owner}"
    )


def _decode_pause_transition(
    value: Optional[str],
) -> Optional[_PauseTransitionIdentity]:
    if not value or not value.startswith(_PAUSE_TRANSITION_PREFIX):
        return None
    fields = value[len(_PAUSE_TRANSITION_PREFIX) :].split("|", 4)
    if len(fields) != 5:
        return None
    transition_id, action, expected_state, deadline, owner = fields
    if (
        not transition_id
        or not owner
        or action not in {"pause", "continue"}
        or expected_state not in {"0", "1"}
    ):
        return None
    try:
        deadline_monotonic_ns = int(deadline)
    except ValueError:
        return None
    if deadline_monotonic_ns <= 0:
        return None
    return _PauseTransitionIdentity(
        transition_id=transition_id,
        owner=owner,
        action=action,
        expected_state=expected_state == "1",
        deadline_monotonic_ns=deadline_monotonic_ns,
    )


def _encode_pause_transition_applied(identity: _PauseTransitionIdentity) -> str:
    encoded = _encode_pause_transition(identity)
    return _PAUSE_TRANSITION_APPLIED_PREFIX + encoded[len(_PAUSE_TRANSITION_PREFIX) :]


def _decode_pause_transition_applied(
    value: Optional[str],
) -> Optional[_PauseTransitionIdentity]:
    if not value or not value.startswith(_PAUSE_TRANSITION_APPLIED_PREFIX):
        return None
    encoded = _PAUSE_TRANSITION_PREFIX + value[len(_PAUSE_TRANSITION_APPLIED_PREFIX) :]
    return _decode_pause_transition(encoded)


def _encode_pause_transition_committed_ack(
    identity: _PauseTransitionIdentity,
) -> str:
    encoded = _encode_pause_transition(identity)
    return (
        _PAUSE_TRANSITION_COMMITTED_ACK_PREFIX
        + encoded[len(_PAUSE_TRANSITION_PREFIX) :]
    )


def _decode_pause_transition_committed_ack(
    value: Optional[str],
) -> Optional[_PauseTransitionIdentity]:
    if not value or not value.startswith(_PAUSE_TRANSITION_COMMITTED_ACK_PREFIX):
        return None
    encoded = (
        _PAUSE_TRANSITION_PREFIX + value[len(_PAUSE_TRANSITION_COMMITTED_ACK_PREFIX) :]
    )
    return _decode_pause_transition(encoded)


def _encode_pause_transition_finalized_ack(
    identity: _PauseTransitionIdentity,
) -> str:
    encoded = _encode_pause_transition(identity)
    return (
        _PAUSE_TRANSITION_FINALIZED_ACK_PREFIX
        + encoded[len(_PAUSE_TRANSITION_PREFIX) :]
    )


def _decode_pause_transition_finalized_ack(
    value: Optional[str],
) -> Optional[_PauseTransitionIdentity]:
    if not value or not value.startswith(_PAUSE_TRANSITION_FINALIZED_ACK_PREFIX):
        return None
    encoded = (
        _PAUSE_TRANSITION_PREFIX + value[len(_PAUSE_TRANSITION_FINALIZED_ACK_PREFIX) :]
    )
    return _decode_pause_transition(encoded)


def _encode_worker_registration(token: str) -> str:
    return f"{_WORKER_REGISTRATION_PREFIX}{token}"


def _decode_worker_registration(value: Optional[str]) -> Optional[str]:
    if not value or not value.startswith(_WORKER_REGISTRATION_PREFIX):
        return None
    token = value[len(_WORKER_REGISTRATION_PREFIX) :]
    return token or None


def _tokenizer_worker_is_alive(identity: _TokenizerWorkerIdentity) -> bool:
    try:
        process = psutil.Process(identity.pid)
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and abs(process.create_time() - identity.process_start_time) < 1e-6
        )
    except (psutil.Error, OSError):
        return False


def _same_pause_transition_correlation(
    left: _PauseTransitionIdentity,
    right: _PauseTransitionIdentity,
) -> bool:
    return (
        left.transition_id == right.transition_id
        and left.owner == right.owner
        and left.action == right.action
        and left.deadline_monotonic_ns == right.deadline_monotonic_ns
    )


class SocketMapping:
    def __init__(self):
        self._zmq_context = zmq.Context()
        self._mapping: Dict[str, zmq.Socket] = {}

    def clear_all_sockets(self):
        for socket in self._mapping.values():
            socket.close()
        self._mapping.clear()

    def clear_socket(self, ipc_name: str) -> None:
        socket = self._mapping.pop(ipc_name, None)
        if socket is not None:
            socket.close()

    def _register_ipc_mapping(self, ipc_name: str, is_tokenizer: bool):
        type_str = "tokenizer" if is_tokenizer else "detokenizer"
        if ipc_name in self._mapping:
            logger.warning(f"{type_str} already registered {ipc_name=}, skipping...")
            return
        logger.info(f"Registering {type_str} {ipc_name=} in SocketMapping...")
        socket = get_zmq_socket(self._zmq_context, zmq.PUSH, ipc_name, False)
        self._mapping[ipc_name] = socket

    def send_output(self, ipc_name: str, output: Any, is_tokenizer: bool = False):
        if ipc_name is None:
            # Some unhandled cases
            logger.warning(f"IPC name is None, output type={type(output)}, skipping...")
            return

        if ipc_name not in self._mapping:
            self._register_ipc_mapping(ipc_name, is_tokenizer=is_tokenizer)
        sock_send(self._mapping[ipc_name], output)


def _extract_field_by_index(
    output: Any, field_name: str, index: int, check_length: bool = True
) -> Any:
    """Extract a field value from output by index, handling None and length checks.

    Args:
        output: The output object containing the field
        field_name: The name of the field to extract
        index: The index to access in the field list
        check_length: If True, check both field existence and length. If False, only check field existence.

    Returns:
        A list containing the field value at index, or None if not available.
    """
    field = getattr(output, field_name, None)
    if field is None:
        return None

    should_wrap_result = field_name in ("customized_info", "time_stats")
    if should_wrap_result:
        field = unwrap_from_pickle(field)
        if field is None:
            return None

    if isinstance(field, dict):
        new_field = {}
        for k, v in field.items():
            if len(v) > index:
                new_field[k] = [v[index]] if should_wrap_result else v[index]
            else:
                new_field[k] = [None] if should_wrap_result else None
        if should_wrap_result:
            return wrap_as_pickle(new_field) if new_field else None
        return new_field

    if check_length:
        if len(field) <= index:
            return None

    new_field = [field[index]]
    return wrap_as_pickle(new_field) if should_wrap_result else new_field


def _handle_output_by_index(output, i):
    """NOTE: A maintainable method is better here."""
    if isinstance(output, BatchTokenIDOutput):
        new_output = BatchTokenIDOutput(
            rids=[output.rids[i]],
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),
            spec_num_correct_drafts=_extract_field_by_index(
                output, "spec_num_correct_drafts", i
            ),
            spec_correct_drafts_histogram=_extract_field_by_index(
                output, "spec_correct_drafts_histogram", i
            ),
            spec_num_block_accept_tokens=_extract_field_by_index(
                output, "spec_num_block_accept_tokens", i
            ),
            spec_num_cap_tokens=_extract_field_by_index(
                output, "spec_num_cap_tokens", i
            ),
            spec_cap_lens_histogram=_extract_field_by_index(
                output, "spec_cap_lens_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            decoded_texts=_extract_field_by_index(output, "decoded_texts", i),
            decode_ids=_extract_field_by_index(output, "decode_ids", i),
            read_offsets=_extract_field_by_index(output, "read_offsets", i),
            output_ids=_extract_field_by_index(output, "output_ids", i),
            skip_special_tokens=_extract_field_by_index(
                output, "skip_special_tokens", i
            ),
            spaces_between_special_tokens=_extract_field_by_index(
                output, "spaces_between_special_tokens", i
            ),
            no_stop_trim=_extract_field_by_index(output, "no_stop_trim", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            cached_tokens_details=_extract_field_by_index(
                output, "cached_tokens_details", i
            ),
            image_tokens=_extract_field_by_index(output, "image_tokens", i),
            audio_tokens=_extract_field_by_index(output, "audio_tokens", i),
            video_tokens=_extract_field_by_index(output, "video_tokens", i),
            input_token_logprobs_val=_extract_field_by_index(
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_token_sampling_mask=_extract_field_by_index(
                output, "output_token_sampling_mask", i, check_length=False
            ),
            output_token_sampling_logprobs=_extract_field_by_index(
                output, "output_token_sampling_logprobs", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(
                output, "output_hidden_states", i, check_length=False
            ),
            routed_experts=_extract_field_by_index(
                output, "routed_experts", i, check_length=False
            ),
            indexer_topk=_extract_field_by_index(
                output, "indexer_topk", i, check_length=False
            ),
            retraction_counts=_extract_field_by_index(output, "retraction_counts", i),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            token_steps=_extract_field_by_index(
                output, "token_steps", i, check_length=False
            ),
            customized_info=_extract_field_by_index(
                output, "customized_info", i, check_length=False
            ),
            dp_ranks=_extract_field_by_index(output, "dp_ranks", i, check_length=False),
        )
    elif isinstance(output, BatchEmbeddingOutput):
        new_output = BatchEmbeddingOutput(
            rids=[output.rids[i]],
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            embeddings=_extract_field_by_index(output, "embeddings", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
    elif isinstance(output, BatchStrOutput):
        new_output = BatchStrOutput(
            rids=[output.rids[i]],
            spec_verify_ct=_extract_field_by_index(output, "spec_verify_ct", i),
            spec_num_correct_drafts=_extract_field_by_index(
                output, "spec_num_correct_drafts", i
            ),
            spec_correct_drafts_histogram=_extract_field_by_index(
                output, "spec_correct_drafts_histogram", i
            ),
            spec_num_block_accept_tokens=_extract_field_by_index(
                output, "spec_num_block_accept_tokens", i
            ),
            spec_num_cap_tokens=_extract_field_by_index(
                output, "spec_num_cap_tokens", i
            ),
            spec_cap_lens_histogram=_extract_field_by_index(
                output, "spec_cap_lens_histogram", i
            ),
            time_stats=_extract_field_by_index(output, "time_stats", i),
            finished_reasons=_extract_field_by_index(output, "finished_reasons", i),
            output_strs=_extract_field_by_index(output, "output_strs", i),
            output_ids=_extract_field_by_index(output, "output_ids", i),
            prompt_tokens=_extract_field_by_index(output, "prompt_tokens", i),
            completion_tokens=_extract_field_by_index(output, "completion_tokens", i),
            reasoning_tokens=_extract_field_by_index(output, "reasoning_tokens", i),
            cached_tokens=_extract_field_by_index(output, "cached_tokens", i),
            cached_tokens_details=_extract_field_by_index(
                output, "cached_tokens_details", i
            ),
            image_tokens=_extract_field_by_index(output, "image_tokens", i),
            audio_tokens=_extract_field_by_index(output, "audio_tokens", i),
            video_tokens=_extract_field_by_index(output, "video_tokens", i),
            input_token_logprobs_val=_extract_field_by_index(
                output, "input_token_logprobs_val", i, check_length=False
            ),
            input_token_logprobs_idx=_extract_field_by_index(
                output, "input_token_logprobs_idx", i, check_length=False
            ),
            output_token_logprobs_val=_extract_field_by_index(
                output, "output_token_logprobs_val", i, check_length=False
            ),
            output_token_logprobs_idx=_extract_field_by_index(
                output, "output_token_logprobs_idx", i, check_length=False
            ),
            input_top_logprobs_val=_extract_field_by_index(
                output, "input_top_logprobs_val", i, check_length=False
            ),
            input_top_logprobs_idx=_extract_field_by_index(
                output, "input_top_logprobs_idx", i, check_length=False
            ),
            output_top_logprobs_val=_extract_field_by_index(
                output, "output_top_logprobs_val", i, check_length=False
            ),
            output_top_logprobs_idx=_extract_field_by_index(
                output, "output_top_logprobs_idx", i, check_length=False
            ),
            input_token_ids_logprobs_val=_extract_field_by_index(
                output, "input_token_ids_logprobs_val", i, check_length=False
            ),
            input_token_ids_logprobs_idx=_extract_field_by_index(
                output, "input_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_ids_logprobs_val=_extract_field_by_index(
                output, "output_token_ids_logprobs_val", i, check_length=False
            ),
            output_token_ids_logprobs_idx=_extract_field_by_index(
                output, "output_token_ids_logprobs_idx", i, check_length=False
            ),
            output_token_entropy_val=_extract_field_by_index(
                output, "output_token_entropy_val", i, check_length=False
            ),
            output_token_sampling_mask=_extract_field_by_index(
                output, "output_token_sampling_mask", i, check_length=False
            ),
            output_token_sampling_logprobs=_extract_field_by_index(
                output, "output_token_sampling_logprobs", i, check_length=False
            ),
            output_hidden_states=_extract_field_by_index(
                output, "output_hidden_states", i, check_length=False
            ),
            routed_experts=_extract_field_by_index(
                output, "routed_experts", i, check_length=False
            ),
            indexer_topk=_extract_field_by_index(
                output, "indexer_topk", i, check_length=False
            ),
            customized_info=_extract_field_by_index(
                output, "customized_info", i, check_length=False
            ),
            dp_ranks=_extract_field_by_index(output, "dp_ranks", i, check_length=False),
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=_extract_field_by_index(output, "retraction_counts", i),
            token_steps=_extract_field_by_index(
                output, "token_steps", i, check_length=False
            ),
        )
    else:
        new_output = output
    return new_output


class MultiHttpWorkerDetokenizerMixin:
    """Mixin class for DetokenizerManager"""

    def maybe_clear_socket_mapping(self: DetokenizerManager):
        if hasattr(self, "socket_mapping"):
            self.socket_mapping.clear_all_sockets()

    def multi_http_worker_event_loop(self: DetokenizerManager):
        """The event loop that handles requests, for multi multi-http-worker mode"""
        self.socket_mapping = SocketMapping()
        # Watchdog wiring mirrors DetokenizerManager.event_loop: the watchdog is
        # paused while waiting for input and fed once per processed message.
        while True:
            with self.soft_watchdog.disable():
                recv_obj = sock_recv(self.recv_from_scheduler)
            output = self._request_dispatcher(recv_obj)
            if output is not None:
                # Fan out the output back to the originating tokenizer worker(s).
                # In multi-detokenizer mode the upstream MultiDetokenizerRouter may
                # forward either batched or single requests, so handle both shapes.
                if isinstance(recv_obj, BaseBatchReq):
                    for i, ipc_name in enumerate(recv_obj.http_worker_ipcs):
                        new_output = _handle_output_by_index(output, i)
                        self.socket_mapping.send_output(
                            ipc_name, new_output, is_tokenizer=True
                        )
                elif isinstance(recv_obj, BaseReq):
                    self.socket_mapping.send_output(
                        recv_obj.http_worker_ipc, output, is_tokenizer=True
                    )
                else:
                    raise ValueError(
                        f"multi_http_worker_event_loop got unexpected req type {type(recv_obj)}"
                    )
            self.soft_watchdog.feed()


class MultiTokenizerRouter:
    """A router between tokenizer managers and the scheduler/detokenizer manager.

    Forward: tokenizer managers → router → scheduler.
    Backward: detokenizer manager → router → tokenizer managers.
    Also broadcasts pause/continue to all tokenizer managers for consistent is_pause state.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        self.server_args = server_args
        context = zmq.asyncio.Context(3)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )
        self.receive_from_worker = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_worker_ipc_name, True
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._task = asyncio.run_coroutine_threadsafe(
            self.router_worker_obj(), self._loop
        )
        self._handle_task = asyncio.run_coroutine_threadsafe(
            print_exception_wrapper(self.handle_loop), self._loop
        )

        # In multi-tokenizer mode the N TokenizerWorker processes cannot each
        # bind the zmq PULL socket used for load snapshots, so the single
        # MultiTokenizerRouter process owns it (zmq -> SHM) and the workers
        # read SHM only. Drain it event-driven via the socket's fd instead of
        # polling on a timer.
        self.load_snapshot_reader = None
        if zmq_reader_owner(server_args, "MultiTokenizerRouter"):
            self.load_snapshot_reader = create_load_snapshot_reader(
                server_args, port_args, caller="MultiTokenizerRouter"
            )
            self._loop.call_soon_threadsafe(self._register_load_snapshot_reader)

        self.disaggregation_bootstrap_server = start_disagg_service(self.server_args)

        # Worker IPC names for pause/continue broadcasting
        self.all_worker_ipcs: set[str] = set()
        self._worker_registrations: Dict[str, _TokenizerWorkerIdentity] = {}
        self.pause_owners: set[str] = set()
        self.active_remote_pause_owner: Optional[str] = None
        self.pending_remote_pause_requests = deque()
        self._pause_transitions: Dict[str, _RouterPauseTransition] = {}
        self._pause_poisoned_owners: set[str] = set()
        self._pause_owner_transitions: Dict[str, _PauseTransitionIdentity] = {}
        self._pause_owner_workers: Dict[str, _TokenizerWorkerIdentity] = {}
        self._pause_fail_stopped = False
        # Shared socket mapping (both coroutines run on self._loop, so safe)
        self.socket_mapping = SocketMapping()

    def _run_loop(self):
        self._loop.run_forever()

    def _register_load_snapshot_reader(self):
        """Drain zmq load snapshots into SHM whenever the PULL socket is readable.

        zmq exposes an edge-triggered fd; ``poll()`` drains it until empty, which
        also re-arms the fd, so TokenizerWorkers reading SHM stay up to date
        without any timer.
        """
        assert self.load_snapshot_reader is not None
        self._loop.add_reader(
            self.load_snapshot_reader.fileno(), self.load_snapshot_reader.poll
        )
        # Drain anything already queued before the fd was registered.
        self.load_snapshot_reader.poll()

    async def router_worker_obj(self):
        """Forward path: workers → scheduler, with pause/continue broadcast."""
        while True:
            recv_obj = await async_sock_recv(self.receive_from_worker)

            if isinstance(recv_obj, TokenizerWorkerRegistrationReq):
                self._handle_tokenizer_worker_registration(recv_obj)
                continue

            if isinstance(recv_obj, PauseContinueBroadcastReq):
                self._handle_pause_continue_ack(recv_obj)
                continue

            if isinstance(
                recv_obj, (PauseGenerationReqInput, ContinueGenerationReqInput)
            ):
                await self._handle_pause_continue_request(recv_obj)
                continue

            await async_sock_send(self.send_to_scheduler, recv_obj)

    @staticmethod
    def _registration_identity(
        request: TokenizerWorkerRegistrationReq,
    ) -> Optional[_TokenizerWorkerIdentity]:
        if (
            type(request.worker_ipc_name) is not str
            or not request.worker_ipc_name
            or type(request.worker_pid) is not int
            or request.worker_pid <= 0
            or type(request.process_start_time) not in {float, int}
            or request.process_start_time <= 0
            or type(request.worker_token) is not str
            or not request.worker_token
        ):
            return None
        return _TokenizerWorkerIdentity(
            ipc_name=request.worker_ipc_name,
            pid=request.worker_pid,
            process_start_time=float(request.process_start_time),
            token=request.worker_token,
        )

    def _ensure_worker_registration_state(self) -> None:
        if not hasattr(self, "_worker_registrations"):
            self._worker_registrations = {}

    def _drop_tokenizer_worker(
        self,
        identity: _TokenizerWorkerIdentity,
        *,
        reason: str,
    ) -> None:
        self._ensure_pause_transition_state()
        if self._worker_registrations.get(identity.ipc_name) != identity:
            return
        self._worker_registrations.pop(identity.ipc_name, None)
        self.all_worker_ipcs.discard(identity.ipc_name)
        self.socket_mapping.clear_socket(identity.ipc_name)
        for owner, owner_worker in tuple(self._pause_owner_workers.items()):
            if owner_worker == identity:
                self._stop_for_lost_pause_owner(owner, identity, reason)
        logger.info(
            "Router removed tokenizer worker %s (pid=%s): %s",
            identity.ipc_name,
            identity.pid,
            reason,
        )

    def _prune_dead_tokenizer_workers(self) -> None:
        self._ensure_worker_registration_state()
        for identity in tuple(self._worker_registrations.values()):
            if not _tokenizer_worker_is_alive(identity):
                self._drop_tokenizer_worker(identity, reason="process is not live")

    def _send_worker_registration_result(
        self,
        identity: _TokenizerWorkerIdentity,
        *,
        accepted: bool,
    ) -> None:
        try:
            self.socket_mapping.send_output(
                identity.ipc_name,
                PauseContinueBroadcastReq(
                    rid=_encode_worker_registration(identity.token),
                    is_pause=bool(self.pause_owners or self._pause_poisoned_owners),
                    http_worker_ipc=(
                        _WORKER_REGISTRATION_ACCEPTED
                        if accepted
                        else _WORKER_REGISTRATION_REJECTED
                    ),
                ),
            )
        except Exception:
            logger.exception(
                "Failed to publish tokenizer worker registration result to %s",
                identity.ipc_name,
            )
        finally:
            if not accepted:
                self.socket_mapping.clear_socket(identity.ipc_name)

    def _handle_tokenizer_worker_registration(
        self,
        request: TokenizerWorkerRegistrationReq,
    ) -> None:
        self._ensure_pause_transition_state()
        self._ensure_worker_registration_state()
        identity = self._registration_identity(request)
        if identity is None:
            logger.error("Rejected tokenizer worker registration with invalid identity")
            return
        current = self._worker_registrations.get(identity.ipc_name)
        if request.unregister:
            if current == identity:
                self._drop_tokenizer_worker(identity, reason="graceful unregister")
            return

        self._prune_dead_tokenizer_workers()
        if self._pause_fail_stopped:
            self._send_worker_registration_result(identity, accepted=False)
            return
        current = self._worker_registrations.get(identity.ipc_name)
        if current == identity:
            self._send_worker_registration_result(identity, accepted=True)
            return
        if current is not None:
            self._send_worker_registration_result(identity, accepted=False)
            return
        if self._pause_transitions:
            return
        configured_workers = getattr(
            getattr(self, "server_args", None),
            "tokenizer_worker_num",
            0,
        )
        if (
            not _tokenizer_worker_is_alive(identity)
            or configured_workers <= 0
            or len(self._worker_registrations) >= configured_workers
        ):
            self._send_worker_registration_result(identity, accepted=False)
            return

        self._worker_registrations[identity.ipc_name] = identity
        self.all_worker_ipcs.add(identity.ipc_name)
        logger.info(
            "Router registered tokenizer worker %s (pid=%s, total=%s)",
            identity.ipc_name,
            identity.pid,
            len(self._worker_registrations),
        )
        self._send_worker_registration_result(identity, accepted=True)

    def _ensure_pause_transition_state(self) -> None:
        if not hasattr(self, "_pause_transitions"):
            self._pause_transitions = {}
        if not hasattr(self, "_pause_poisoned_owners"):
            self._pause_poisoned_owners = set()
        if not hasattr(self, "_pause_owner_transitions"):
            self._pause_owner_transitions = {}
        if not hasattr(self, "_pause_owner_workers"):
            self._pause_owner_workers = {}
        if not hasattr(self, "_pause_fail_stopped"):
            self._pause_fail_stopped = False

    def _stop_for_lost_pause_owner(
        self,
        owner: str,
        worker: _TokenizerWorkerIdentity,
        reason: str,
    ) -> None:
        self._pause_poisoned_owners.add(owner)
        if self._pause_fail_stopped:
            return
        self._pause_fail_stopped = True
        logger.critical(
            "Tokenizer worker %s (pid=%s) lost pause owner %s; stopping the "
            "service: %s",
            worker.ipc_name,
            worker.pid,
            owner,
            reason,
        )
        kill_process_tree(os.getpid(), include_parent=True)

    def _send_pause_transition(
        self,
        transition: _RouterPauseTransition,
        *,
        state: Optional[str] = None,
        effective_state: Optional[bool] = None,
    ) -> bool:
        broadcast = PauseContinueBroadcastReq(
            rid=_encode_pause_transition(transition.identity),
            is_pause=(
                effective_state
                if effective_state is not None
                else (
                    True
                    if state == _PAUSE_TRANSITION_FAILED
                    else transition.identity.expected_state
                )
            ),
            http_worker_ipc=state,
        )
        sent_to_all = True
        worker_ipcs = sorted(
            transition.expected_workers,
            key=lambda ipc_name: (
                (
                    state == _PAUSE_TRANSITION_CONFIRMED
                    or state == _PAUSE_TRANSITION_COMMITTED
                )
                and ipc_name == transition.origin_worker_ipc
            ),
        )
        for ipc_name in worker_ipcs:
            try:
                self.socket_mapping.send_output(ipc_name, broadcast)
            except Exception:
                sent_to_all = False
                logger.exception(
                    "Failed to send pause transition %s to %s",
                    transition.identity.transition_id,
                    ipc_name,
                )
                if state in {
                    _PAUSE_TRANSITION_CONFIRMED,
                    _PAUSE_TRANSITION_COMMITTED,
                }:
                    break
        return sent_to_all

    def _fail_pause_transition(
        self,
        transition_id: str,
        message: str,
    ) -> None:
        transition = self._pause_transitions.get(transition_id)
        if transition is None or transition.commit_started:
            return
        if transition.timeout_handle is not None:
            transition.timeout_handle.cancel()
            transition.timeout_handle = None
        if transition.commit_retry_handle is not None:
            transition.commit_retry_handle.cancel()
        current = self._pause_owner_transitions.get(transition.identity.owner)
        if current != transition.identity:
            self._pause_transitions.pop(transition_id, None)
            return
        if (
            transition.identity.action == "pause"
            and transition.scheduler_pause_dispatched
        ):
            transition.commit_started = True
            transition.commit_done = asyncio.get_running_loop().create_future()
            transition.commit_task = asyncio.create_task(
                self._recover_failed_pause_transition(transition, message)
            )
            return
        self._pause_transitions.pop(transition_id, None)
        self._pause_poisoned_owners.add(transition.identity.owner)
        logger.error(
            "Pause transition %s for %s failed: %s",
            transition.identity.transition_id,
            transition.identity.owner,
            message,
        )
        self._send_pause_transition(
            transition,
            state=_PAUSE_TRANSITION_FAILED,
        )

    async def _recover_failed_pause_transition(
        self,
        transition: _RouterPauseTransition,
        message: str,
    ) -> None:
        identity = transition.identity
        remaining_owners = self.pause_owners - {identity.owner}
        if not remaining_owners:
            try:
                await asyncio.wait_for(
                    async_sock_send(
                        self.send_to_scheduler,
                        ContinueGenerationReqInput(
                            rid=identity.owner,
                            torch_empty_cache=False,
                        ),
                    ),
                    timeout=_PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.critical(
                    "Failed to recover pause transition %s; stopping the service",
                    identity.transition_id,
                    exc_info=True,
                )
                kill_process_tree(os.getpid(), include_parent=True)
                return

        self.pause_owners.discard(identity.owner)
        if self.active_remote_pause_owner == identity.owner:
            self.active_remote_pause_owner = None
        self._pause_poisoned_owners.discard(identity.owner)
        if self._pause_owner_transitions.get(identity.owner) == identity:
            self._pause_owner_transitions.pop(identity.owner, None)
        if (
            identity.action == "pause"
            and self._pause_owner_workers.get(identity.owner)
            == transition.origin_worker
        ):
            self._pause_owner_workers.pop(identity.owner, None)
        self._pause_transitions.pop(identity.transition_id, None)
        logger.error(
            "Pause transition %s for %s failed and was rolled back: %s",
            identity.transition_id,
            identity.owner,
            message,
        )
        if not self._send_pause_transition(
            transition,
            state=_PAUSE_TRANSITION_FAILED,
            effective_state=bool(remaining_owners),
        ):
            logger.critical(
                "Failed to publish pause rollback %s; stopping the service",
                identity.transition_id,
            )
            kill_process_tree(os.getpid(), include_parent=True)
            return
        if transition.commit_done is not None and not transition.commit_done.done():
            transition.commit_done.set_result(False)
        if (
            self.active_remote_pause_owner is None
            and self.pending_remote_pause_requests
        ):
            await self._promote_next_remote_pause()

    def _expire_pause_transition(self, transition_id: str) -> None:
        transition = self._pause_transitions.get(transition_id)
        if transition is None or transition.commit_started:
            return
        if time.monotonic_ns() < transition.identity.deadline_monotonic_ns:
            delay = (
                transition.identity.deadline_monotonic_ns - time.monotonic_ns()
            ) / 1_000_000_000
            transition.timeout_handle = asyncio.get_running_loop().call_later(
                max(0.0, delay),
                self._expire_pause_transition,
                transition_id,
            )
            return
        self._fail_pause_transition(
            transition_id,
            "worker acknowledgement deadline expired",
        )

    def _register_pause_transition(
        self,
        identity: _PauseTransitionIdentity,
        origin_worker_ipc: Optional[str],
    ) -> Optional[_RouterPauseTransition]:
        self._ensure_pause_transition_state()
        if getattr(self, "_worker_registrations", None):
            self._prune_dead_tokenizer_workers()
        if self._pause_fail_stopped:
            return None
        expected_workers = frozenset(self.all_worker_ipcs)
        origin_worker = getattr(self, "_worker_registrations", {}).get(
            origin_worker_ipc
        )
        configured_workers = getattr(
            getattr(self, "server_args", None),
            "tokenizer_worker_num",
            len(expected_workers),
        )
        if (
            origin_worker_ipc is None
            or origin_worker_ipc not in expected_workers
            or origin_worker is None
            or not expected_workers
            or len(expected_workers) != configured_workers
            or time.monotonic_ns() >= identity.deadline_monotonic_ns
        ):
            return None
        is_remote_owner = identity.owner.startswith(
            _REMOTE_WEIGHT_TRANSFER_PAUSE_OWNER_PREFIX
        )
        if is_remote_owner:
            owner_worker = self._pause_owner_workers.get(identity.owner)
            if identity.action == "continue":
                if owner_worker != origin_worker:
                    return None
            elif owner_worker is not None and owner_worker != origin_worker:
                return None
        existing = self._pause_transitions.get(identity.transition_id)
        if existing is not None:
            if (
                existing.identity == identity
                and existing.origin_worker_ipc == origin_worker_ipc
            ):
                return existing
            return None
        previous = self._pause_owner_transitions.get(identity.owner)
        if previous is not None and previous != identity:
            if previous.deadline_monotonic_ns >= identity.deadline_monotonic_ns:
                return None
            stale = self._pause_transitions.get(previous.transition_id)
            if stale is not None and stale.commit_started:
                return None
            stale = self._pause_transitions.pop(previous.transition_id, None)
            if stale is not None:
                self._pause_poisoned_owners.add(identity.owner)
                if stale.timeout_handle is not None:
                    stale.timeout_handle.cancel()
        transition = _RouterPauseTransition(
            identity=identity,
            origin_worker_ipc=origin_worker_ipc,
            origin_worker=origin_worker,
            expected_workers=expected_workers,
        )
        self._pause_transitions[identity.transition_id] = transition
        self._pause_owner_transitions[identity.owner] = identity
        if is_remote_owner and identity.action == "pause":
            self._pause_owner_workers[identity.owner] = origin_worker
        delay = (identity.deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000
        transition.timeout_handle = asyncio.get_running_loop().call_later(
            max(0.0, delay),
            self._expire_pause_transition,
            identity.transition_id,
        )
        return transition

    def _poison_unregistered_pause_transition(
        self,
        identity: _PauseTransitionIdentity,
    ) -> None:
        current = self._pause_owner_transitions.get(identity.owner)
        if (
            current is not None
            and current != identity
            and current.deadline_monotonic_ns > identity.deadline_monotonic_ns
        ):
            return
        self._pause_owner_transitions[identity.owner] = identity
        self._pause_poisoned_owners.add(identity.owner)

    def _reject_pause_transition(
        self,
        identity: _PauseTransitionIdentity,
        origin_worker_ipc: Optional[str],
        message: str,
    ) -> None:
        transition = self._register_pause_transition(identity, origin_worker_ipc)
        if transition is None:
            self._poison_unregistered_pause_transition(identity)
            return
        self._fail_pause_transition(identity.transition_id, message)

    async def _dispatch_pause_transition_to_scheduler(
        self,
        identity: _PauseTransitionIdentity,
        request: PauseGenerationReqInput | ContinueGenerationReqInput,
    ) -> None:
        timeout = max(
            0.0,
            (identity.deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000,
        )
        await asyncio.wait_for(
            async_sock_send(self.send_to_scheduler, request),
            timeout=timeout,
        )
        transition = self._pause_transitions.get(identity.transition_id)
        if (
            transition is not None
            and transition.identity == identity
            and identity.action == "pause"
        ):
            transition.scheduler_pause_dispatched = True

    def _complete_pause_transition(self, transition_id: str) -> None:
        transition = self._pause_transitions.get(transition_id)
        if transition is None or transition.confirmation_sent:
            return
        if (
            self._pause_owner_transitions.get(transition.identity.owner)
            != transition.identity
        ):
            self._pause_transitions.pop(transition_id, None)
            return
        current_expected_state = (
            True
            if transition.identity.action == "pause"
            else bool(self.pause_owners - {transition.identity.owner})
        )
        if transition.identity.expected_state != current_expected_state:
            self._fail_pause_transition(
                transition_id,
                "transition expected_state became stale before commit",
            )
            return
        if not self._send_pause_transition(
            transition,
            state=_PAUSE_TRANSITION_CONFIRMED,
        ):
            self._fail_pause_transition(
                transition_id,
                "confirmation broadcast failed",
            )
            return
        transition.confirmation_sent = True

    def _finish_pause_transition_commit(
        self,
        transition: _RouterPauseTransition,
    ) -> None:
        transition_id = transition.identity.transition_id
        if (
            self._pause_transitions.get(transition_id) is not transition
            or transition.committed
        ):
            return
        transition.committed = True
        identity = transition.identity
        was_poisoned = identity.owner in self._pause_poisoned_owners
        if identity.action == "continue":
            self.pause_owners.discard(identity.owner)
            if self.active_remote_pause_owner == identity.owner:
                self.active_remote_pause_owner = None
        else:
            self.pause_owners.add(identity.owner)
        self._pause_poisoned_owners.discard(identity.owner)
        transition.commit_pending_workers = set(transition.expected_workers)
        transition.commit_deadline_monotonic_ns = min(
            identity.deadline_monotonic_ns,
            time.monotonic_ns()
            + int(_PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC * 1_000_000_000),
        )
        self._send_committed_pause_transition(transition_id)

        if (
            identity.action == "continue"
            and self.active_remote_pause_owner is None
            and self.pending_remote_pause_requests
        ):
            asyncio.create_task(
                self._promote_next_remote_pause(
                    force_scheduler_pause=was_poisoned,
                )
            )

    async def _resume_scheduler_and_finish_pause_transition(
        self,
        transition: _RouterPauseTransition,
    ) -> None:
        request = transition.scheduler_resume_request
        assert request is not None
        deadline_monotonic_ns = min(
            transition.identity.deadline_monotonic_ns,
            time.monotonic_ns()
            + int(_PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC * 1_000_000_000),
        )
        last_error: Optional[BaseException] = None
        while (
            self._pause_transitions.get(transition.identity.transition_id) is transition
        ):
            remaining = (deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(
                    async_sock_send(self.send_to_scheduler, request),
                    timeout=remaining,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                last_error = error
                error_text = str(error) or type(error).__name__
                logger.warning(
                    "Failed to resume the scheduler for committed pause "
                    "transition %s; retrying: %s",
                    transition.identity.transition_id,
                    error_text,
                )
                remaining = (
                    deadline_monotonic_ns - time.monotonic_ns()
                ) / 1_000_000_000
                if remaining <= 0:
                    break
                await asyncio.sleep(
                    min(_PAUSE_TRANSITION_RETRY_INTERVAL_SEC, remaining)
                )
                continue
            self._finish_pause_transition_commit(transition)
            return
        message = "scheduler resume deadline expired"
        if last_error is not None:
            message = (
                "scheduler resume failed: "
                f"{str(last_error) or type(last_error).__name__}"
            )
        self._stop_failed_pause_transition_commit(
            transition,
            message,
        )

    def _stop_failed_pause_transition_commit(
        self,
        transition: _RouterPauseTransition,
        message: str,
    ) -> None:
        identity = transition.identity
        if (
            self._pause_transitions.get(identity.transition_id) is not transition
            or transition.committed
        ):
            return
        self._pause_transitions.pop(identity.transition_id, None)
        self._pause_poisoned_owners.add(identity.owner)
        logger.critical(
            "Pause transition %s could not resume the scheduler; stopping the "
            "service: %s",
            identity.transition_id,
            message,
        )
        self._send_pause_transition(
            transition,
            state=_PAUSE_TRANSITION_FAILED,
            effective_state=True,
        )
        if transition.commit_done is not None and not transition.commit_done.done():
            transition.commit_done.set_result(False)
        kill_process_tree(os.getpid(), include_parent=True)

    def _send_committed_pause_transition(self, transition_id: str) -> None:
        transition = self._pause_transitions.get(transition_id)
        if transition is None or not transition.committed:
            return
        transition.commit_retry_handle = None
        if transition.commit_deadline_monotonic_ns is None:
            transition.commit_deadline_monotonic_ns = min(
                transition.identity.deadline_monotonic_ns,
                time.monotonic_ns()
                + int(_PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC * 1_000_000_000),
            )
        if time.monotonic_ns() >= transition.commit_deadline_monotonic_ns:
            self._stop_failed_committed_pause_transition(transition)
            return
        broadcast = PauseContinueBroadcastReq(
            rid=_encode_pause_transition(transition.identity),
            is_pause=transition.identity.expected_state,
            http_worker_ipc=_PAUSE_TRANSITION_COMMITTED,
        )
        for worker_ipc in sorted(tuple(transition.commit_pending_workers)):
            if time.monotonic_ns() >= transition.commit_deadline_monotonic_ns:
                self._stop_failed_committed_pause_transition(transition)
                return
            try:
                self.socket_mapping.send_output(worker_ipc, broadcast)
            except Exception as error:
                logger.warning(
                    "Failed to send committed pause transition %s to %s; retrying: %s",
                    transition.identity.transition_id,
                    worker_ipc,
                    error,
                )
                continue
        if transition.commit_pending_workers:
            remaining_ns = transition.commit_deadline_monotonic_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                self._stop_failed_committed_pause_transition(transition)
                return
            transition.commit_retry_handle = asyncio.get_running_loop().call_later(
                min(
                    _PAUSE_TRANSITION_RETRY_INTERVAL_SEC,
                    remaining_ns / 1_000_000_000,
                ),
                self._send_committed_pause_transition,
                transition_id,
            )
            return

    def _finalize_committed_pause_transition(
        self,
        transition: _RouterPauseTransition,
    ) -> None:
        identity = transition.identity
        if (
            self._pause_transitions.get(identity.transition_id) is not transition
            or not transition.committed
            or transition.commit_pending_workers
        ):
            return
        if transition.commit_retry_handle is not None:
            transition.commit_retry_handle.cancel()
            transition.commit_retry_handle = None
        self._send_finalized_pause_transition(identity.transition_id)

    def _send_finalized_pause_transition(self, transition_id: str) -> None:
        transition = self._pause_transitions.get(transition_id)
        if (
            transition is None
            or not transition.committed
            or transition.commit_pending_workers
        ):
            return
        transition.commit_retry_handle = None
        deadline = transition.commit_deadline_monotonic_ns
        if deadline is None or time.monotonic_ns() >= deadline:
            self._stop_failed_committed_pause_transition(transition)
            return
        try:
            self.socket_mapping.send_output(
                transition.origin_worker_ipc,
                PauseContinueBroadcastReq(
                    rid=_encode_pause_transition(transition.identity),
                    is_pause=transition.identity.expected_state,
                    http_worker_ipc=_PAUSE_TRANSITION_FINALIZED,
                ),
            )
        except Exception:
            logger.warning(
                "Failed to send finalized pause transition %s; retrying",
                transition.identity.transition_id,
                exc_info=True,
            )
        remaining_ns = deadline - time.monotonic_ns()
        if remaining_ns <= 0:
            self._stop_failed_committed_pause_transition(transition)
            return
        transition.commit_retry_handle = asyncio.get_running_loop().call_later(
            min(
                _PAUSE_TRANSITION_RETRY_INTERVAL_SEC,
                remaining_ns / 1_000_000_000,
            ),
            self._send_finalized_pause_transition,
            transition_id,
        )

    def _stop_failed_committed_pause_transition(
        self,
        transition: _RouterPauseTransition,
    ) -> None:
        identity = transition.identity
        if (
            self._pause_transitions.get(identity.transition_id) is not transition
            or not transition.committed
        ):
            return
        if transition.commit_retry_handle is not None:
            transition.commit_retry_handle.cancel()
            transition.commit_retry_handle = None
        self._pause_transitions.pop(identity.transition_id, None)
        self.pause_owners.add(identity.owner)
        self._pause_poisoned_owners.add(identity.owner)
        self._pause_fail_stopped = True
        if identity.owner.startswith(_REMOTE_WEIGHT_TRANSFER_PAUSE_OWNER_PREFIX):
            self.active_remote_pause_owner = identity.owner
        logger.critical(
            "Committed pause transition %s could not reach all workers; stopping "
            "the service",
            identity.transition_id,
        )
        self._send_pause_transition(
            transition,
            state=_PAUSE_TRANSITION_FAILED,
            effective_state=True,
        )
        if transition.commit_done is not None and not transition.commit_done.done():
            transition.commit_done.set_result(False)
        kill_process_tree(os.getpid(), include_parent=True)

    def _commit_pause_transition(self, transition_id: str) -> None:
        transition = self._pause_transitions.get(transition_id)
        if transition is None or transition.commit_started:
            return
        if time.monotonic_ns() >= transition.identity.deadline_monotonic_ns:
            self._fail_pause_transition(
                transition_id,
                "worker applied acknowledgement deadline expired",
            )
            return
        if (
            self._pause_owner_transitions.get(transition.identity.owner)
            != transition.identity
        ):
            self._pause_transitions.pop(transition_id, None)
            return
        current_expected_state = (
            True
            if transition.identity.action == "pause"
            else bool(self.pause_owners - {transition.identity.owner})
        )
        if transition.identity.expected_state != current_expected_state:
            self._fail_pause_transition(
                transition_id,
                "transition expected_state became stale before commit",
            )
            return
        transition.commit_started = True
        transition.commit_done = asyncio.get_running_loop().create_future()
        if transition.timeout_handle is not None:
            transition.timeout_handle.cancel()
            transition.timeout_handle = None
        if transition.scheduler_resume_request is not None:
            transition.commit_task = asyncio.create_task(
                self._resume_scheduler_and_finish_pause_transition(transition)
            )
            return
        self._finish_pause_transition_commit(transition)

    async def _wait_for_irrevocable_pause_commit(self) -> None:
        while True:
            commit_done = next(
                (
                    transition.commit_done
                    for transition in self._pause_transitions.values()
                    if transition.commit_started
                ),
                None,
            )
            if commit_done is None:
                return
            await asyncio.shield(commit_done)

    def _handle_pause_transition_applied_ack(
        self,
        recv_obj: PauseContinueBroadcastReq,
        identity: _PauseTransitionIdentity,
    ) -> None:
        transition = self._pause_transitions.get(identity.transition_id)
        worker_ipc = recv_obj.http_worker_ipc
        if (
            transition is None
            or not transition.confirmation_sent
            or transition.identity != identity
            or self._pause_owner_transitions.get(identity.owner) != identity
            or transition.commit_started
            or recv_obj.is_pause != identity.expected_state
            or worker_ipc not in transition.expected_workers
            or worker_ipc in transition.applied_workers
            or time.monotonic_ns() >= identity.deadline_monotonic_ns
        ):
            return
        transition.applied_workers.add(worker_ipc)
        if transition.applied_workers == set(transition.expected_workers):
            self._commit_pause_transition(identity.transition_id)

    def _handle_pause_transition_committed_ack(
        self,
        recv_obj: PauseContinueBroadcastReq,
        identity: _PauseTransitionIdentity,
    ) -> None:
        transition = self._pause_transitions.get(identity.transition_id)
        worker_ipc = recv_obj.http_worker_ipc
        if (
            transition is None
            or not transition.committed
            or transition.identity != identity
            or recv_obj.is_pause != identity.expected_state
            or worker_ipc not in transition.commit_pending_workers
        ):
            return
        if (
            transition.commit_deadline_monotonic_ns is None
            or time.monotonic_ns() >= transition.commit_deadline_monotonic_ns
        ):
            self._stop_failed_committed_pause_transition(transition)
            return
        transition.commit_pending_workers.remove(worker_ipc)
        if not transition.commit_pending_workers:
            self._finalize_committed_pause_transition(transition)

    def _handle_pause_transition_finalized_ack(
        self,
        recv_obj: PauseContinueBroadcastReq,
        identity: _PauseTransitionIdentity,
    ) -> None:
        transition = self._pause_transitions.get(identity.transition_id)
        if (
            transition is None
            or not transition.committed
            or transition.commit_pending_workers
            or transition.identity != identity
            or recv_obj.is_pause != identity.expected_state
            or recv_obj.http_worker_ipc != transition.origin_worker_ipc
        ):
            return
        deadline = transition.commit_deadline_monotonic_ns
        if deadline is None or time.monotonic_ns() >= deadline:
            self._stop_failed_committed_pause_transition(transition)
            return
        if transition.commit_retry_handle is not None:
            transition.commit_retry_handle.cancel()
            transition.commit_retry_handle = None
        self._pause_transitions.pop(identity.transition_id, None)
        if self._pause_owner_transitions.get(identity.owner) == identity:
            self._pause_owner_transitions.pop(identity.owner, None)
        if (
            identity.action == "continue"
            and self._pause_owner_workers.get(identity.owner)
            == transition.origin_worker
        ):
            self._pause_owner_workers.pop(identity.owner, None)
        if transition.commit_done is not None and not transition.commit_done.done():
            transition.commit_done.set_result(True)

    def _handle_pause_continue_ack(
        self,
        recv_obj: PauseContinueBroadcastReq,
    ) -> None:
        self._ensure_pause_transition_state()
        finalized_identity = _decode_pause_transition_finalized_ack(recv_obj.rid)
        if finalized_identity is not None:
            self._handle_pause_transition_finalized_ack(
                recv_obj,
                finalized_identity,
            )
            return
        committed_identity = _decode_pause_transition_committed_ack(recv_obj.rid)
        if committed_identity is not None:
            self._handle_pause_transition_committed_ack(
                recv_obj,
                committed_identity,
            )
            return
        applied_identity = _decode_pause_transition_applied(recv_obj.rid)
        if applied_identity is not None:
            self._handle_pause_transition_applied_ack(
                recv_obj,
                applied_identity,
            )
            return
        identity = _decode_pause_transition(recv_obj.rid)
        if identity is None:
            return
        transition = self._pause_transitions.get(identity.transition_id)
        worker_ipc = recv_obj.http_worker_ipc
        if (
            transition is None
            or transition.identity != identity
            or self._pause_owner_transitions.get(identity.owner) != identity
            or transition.commit_started
            or transition.confirmation_sent
            or recv_obj.is_pause != identity.expected_state
            or worker_ipc not in transition.expected_workers
            or worker_ipc in transition.acked_workers
            or time.monotonic_ns() >= identity.deadline_monotonic_ns
        ):
            return
        transition.acked_workers.add(worker_ipc)
        if transition.acked_workers == set(transition.expected_workers):
            current_expected_state = (
                True
                if identity.action == "pause"
                else bool(self.pause_owners - {identity.owner})
            )
            if identity.expected_state != current_expected_state:
                self._fail_pause_transition(
                    identity.transition_id,
                    "transition expected_state became stale",
                )
                return
            if (
                identity.action == "pause"
                and identity.owner.startswith(
                    _REMOTE_WEIGHT_TRANSFER_PAUSE_OWNER_PREFIX
                )
                and self.active_remote_pause_owner not in (None, identity.owner)
            ):
                return
            self._complete_pause_transition(identity.transition_id)

    async def _promote_next_remote_pause(
        self,
        *,
        force_scheduler_pause: bool = False,
    ) -> None:
        while self.pending_remote_pause_requests:
            next_request = self.pending_remote_pause_requests.popleft()
            identity = _decode_pause_transition(next_request.rid)
            if identity is None:
                continue
            if time.monotonic_ns() >= identity.deadline_monotonic_ns:
                self._reject_pause_transition(
                    identity,
                    next_request.http_worker_ipc,
                    "queued pause transition deadline expired",
                )
                continue
            transition = self._pause_transitions.get(identity.transition_id)
            if transition is not None:
                self.active_remote_pause_owner = identity.owner
                if force_scheduler_pause:
                    try:
                        await self._dispatch_pause_transition_to_scheduler(
                            identity,
                            next_request,
                        )
                    except Exception as error:
                        self._fail_pause_transition(
                            identity.transition_id,
                            f"scheduler dispatch failed: {error}",
                        )
                        return
                if transition.acked_workers == set(transition.expected_workers):
                    self._complete_pause_transition(identity.transition_id)
                return
            await self._handle_pause_continue_request(next_request)
            return

    async def _handle_pause_continue_request(
        self,
        recv_obj: PauseGenerationReqInput | ContinueGenerationReqInput,
    ) -> None:
        self._ensure_pause_transition_state()
        if self._pause_fail_stopped:
            return
        await self._wait_for_irrevocable_pause_commit()
        is_pause_request = isinstance(recv_obj, PauseGenerationReqInput)
        identity = _decode_pause_transition(recv_obj.rid)
        if identity is None:
            owner = recv_obj.rid or _ADMIN_PAUSE_OWNER
            remaining = (
                self.pause_owners if is_pause_request else self.pause_owners - {owner}
            )
            identity = _new_pause_transition_identity(
                owner=owner,
                action="pause" if is_pause_request else "continue",
                expected_state=True if is_pause_request else bool(remaining),
            )
            recv_obj.rid = _encode_pause_transition(identity)
        owner = identity.owner
        is_remote_owner = owner.startswith(_REMOTE_WEIGHT_TRANSFER_PAUSE_OWNER_PREFIX)
        current = self._pause_owner_transitions.get(owner)
        has_registered_correlation = (
            current is not None
            and _same_pause_transition_correlation(current, identity)
        )
        if has_registered_correlation and current != identity:
            identity = current
            recv_obj.rid = _encode_pause_transition(identity)
        if (
            current is not None
            and current != identity
            and current.deadline_monotonic_ns >= identity.deadline_monotonic_ns
        ):
            return

        expected_action = "pause" if is_pause_request else "continue"
        if identity.action != expected_action:
            self._reject_pause_transition(
                identity,
                recv_obj.http_worker_ipc,
                "transition action does not match request type",
            )
            return

        if is_pause_request:
            if not identity.expected_state:
                self._reject_pause_transition(
                    identity,
                    recv_obj.http_worker_ipc,
                    "pause transition expected_state must be paused",
                )
                return
            if is_remote_owner and self.active_remote_pause_owner not in (None, owner):
                if all(
                    (queued := _decode_pause_transition(request.rid)) is None
                    or queued.owner != owner
                    for request in self.pending_remote_pause_requests
                ):
                    self.pause_owners.add(owner)
                    self.pending_remote_pause_requests.append(recv_obj)
                    transition = self._register_pause_transition(
                        identity,
                        recv_obj.http_worker_ipc,
                    )
                    if transition is None:
                        self._poison_unregistered_pause_transition(identity)
                        return
                    self._send_pause_transition(transition)
                return

            was_paused = bool(self.pause_owners)
            self.pause_owners.add(owner)
            if is_remote_owner:
                self.active_remote_pause_owner = owner

            transition = self._register_pause_transition(
                identity,
                recv_obj.http_worker_ipc,
            )
            if transition is None:
                self._poison_unregistered_pause_transition(identity)
                return
            scheduler_pause_uncertain = bool(self._pause_poisoned_owners) or any(
                pending.identity.action == "continue"
                and not pending.identity.expected_state
                for pending in self._pause_transitions.values()
            )
            should_forward = (
                not was_paused
                or owner == _ADMIN_PAUSE_OWNER
                or scheduler_pause_uncertain
            )
            if should_forward and recv_obj.mode != "abort":
                try:
                    await self._dispatch_pause_transition_to_scheduler(
                        identity,
                        recv_obj,
                    )
                except Exception as error:
                    self._fail_pause_transition(
                        identity.transition_id,
                        f"scheduler dispatch failed: {error}",
                    )
                    return
            self._send_pause_transition(transition)
            return

        if (
            is_remote_owner
            and owner != self.active_remote_pause_owner
            and owner in self.pause_owners
        ):
            self.pending_remote_pause_requests = deque(
                request
                for request in self.pending_remote_pause_requests
                if (
                    (queued := _decode_pause_transition(request.rid)) is None
                    or queued.owner != owner
                )
            )
            for transition_id, pending in tuple(self._pause_transitions.items()):
                if (
                    pending.identity.owner == owner
                    and pending.identity.action == "pause"
                ):
                    self._pause_transitions.pop(transition_id)
                    if pending.timeout_handle is not None:
                        pending.timeout_handle.cancel()

        expected_state = bool(self.pause_owners - {owner})
        if identity.expected_state != expected_state:
            if has_registered_correlation:
                self._reject_pause_transition(
                    identity,
                    recv_obj.http_worker_ipc,
                    "continue transition expected_state is stale",
                )
                return
            # The router owns the aggregate pause state across workers.
            identity = _PauseTransitionIdentity(
                transition_id=identity.transition_id,
                owner=identity.owner,
                action=identity.action,
                expected_state=expected_state,
                deadline_monotonic_ns=identity.deadline_monotonic_ns,
            )
            recv_obj.rid = _encode_pause_transition(identity)

        transition = self._register_pause_transition(
            identity,
            recv_obj.http_worker_ipc,
        )
        if transition is None:
            self._poison_unregistered_pause_transition(identity)
            return
        if not expected_state:
            transition.scheduler_resume_request = recv_obj
        self._send_pause_transition(transition)

    async def handle_loop(self):
        """Backward path: detokenizer → route results to correct worker."""
        while True:
            recv_obj = await async_sock_recv(self.recv_from_detokenizer)
            await self._distribute_result_to_workers(recv_obj)

    async def _distribute_result_to_workers(self, recv_obj):
        if isinstance(recv_obj, BaseReq):
            ipc_names = [recv_obj.http_worker_ipc]
        elif isinstance(recv_obj, BaseBatchReq):
            ipc_names = recv_obj.http_worker_ipcs
        else:
            raise ValueError(f"Unknown recv_obj type: {type(recv_obj)}")

        for i, ipc_name in enumerate(ipc_names):
            new_recv_obj = _handle_output_by_index(recv_obj, i)
            self.socket_mapping.send_output(ipc_name, new_recv_obj)


class MultiDetokenizerRouter:
    """Route scheduler outputs to one of N DetokenizerManager workers.

    Each request is pinned to a worker by hashing its ``http_worker_ipc`` with
    ``zlib.crc32`` (deterministic across runs), so all outputs of the same rid
    always land on the same detokenizer and ``decode_status`` stays consistent.
    """

    def __init__(self, ipc_name_list: List[str], port_args: PortArgs):
        self.ipc_name_list = ipc_name_list
        self.num_workers = len(ipc_name_list)
        self.socket_mapping = SocketMapping()
        context = zmq.Context(2)
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )

    def _pick(self, key: str) -> str:
        return self.ipc_name_list[zlib.crc32(key.encode()) % self.num_workers]

    def _send(self, ipc_name: str, obj: Any) -> None:
        self.socket_mapping.send_output(ipc_name, obj, is_tokenizer=False)

    def event_loop(self):
        while True:
            recv_obj = sock_recv(self.recv_from_scheduler)

            # FreezeGCReq must freeze every detokenizer process.
            if isinstance(recv_obj, FreezeGCReq):
                for ipc in self.ipc_name_list:
                    self._send(ipc, recv_obj)
                continue

            # Single request: route by its own http_worker_ipc.
            if isinstance(recv_obj, BaseReq):
                assert recv_obj.http_worker_ipc is not None, (
                    f"Single req {recv_obj.rid=} missing http_worker_ipc"
                )
                self._send(self._pick(recv_obj.http_worker_ipc), recv_obj)
                continue

            # Batch request.
            if isinstance(recv_obj, BaseBatchReq):
                # Idle/no-op batch (rids=[]): broadcast to all detokenizers
                if not recv_obj.rids:
                    for ipc in self.ipc_name_list:
                        self._send(ipc, recv_obj)
                    continue

                ipcs = recv_obj.http_worker_ipcs
                assert (
                    ipcs is not None
                    and len(ipcs) == len(recv_obj.rids)
                    and all(x is not None for x in ipcs)
                ), f"Batch req {recv_obj.rids=} has invalid http_worker_ipcs"

                # Split per-item and route each by its own ipc.
                for i, ipc_key in enumerate(ipcs):
                    one = _handle_output_by_index(recv_obj, i)
                    if one is recv_obj:
                        raise TypeError(f"Cannot split {type(recv_obj)}")
                    one.http_worker_ipcs = [ipc_key]
                    self._send(self._pick(ipc_key), one)
                continue

            raise ValueError(
                f"MultiDetokenizerRouter got unsupported type {type(recv_obj)}"
            )


def run_multi_detokenizer_router_process(
    ipc_name_list: List[str],
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::detokenizer_router")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    router = None
    try:
        router = MultiDetokenizerRouter(ipc_name_list, port_args)
        router.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"MultiDetokenizerRouter hit an exception: {traceback}")
        if router is not None:
            router.socket_mapping.clear_all_sockets()
        parent_process.send_signal(signal.SIGQUIT)


class TokenizerWorker(TokenizerManager):
    """Tokenizer Worker in multi-http-worker mode"""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        setproctitle.setproctitle(f"sglang::tokenizer_worker:{os.getpid()}")
        import torch

        torch.set_num_threads(1)
        super().__init__(
            server_args,
            port_args,
            start_pd_bootstrap_service=False,
        )

        self.worker_id = os.getpid()
        self.tokenizer_ipc_name = port_args.tokenizer_ipc_name
        self._worker_process_start_time = psutil.Process(self.worker_id).create_time()
        self._worker_token = uuid.uuid4().hex
        self._router_registration_result: Optional[bool] = None
        self._router_registration_future: Optional[asyncio.Future] = None
        self._router_unregistered = False

        # For PD disaggregation
        self.disaggregation_transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        self._pause_continue_futures: Dict[
            str, tuple[_PauseTransitionIdentity, asyncio.Future]
        ] = {}
        self._pause_continue_confirmation_futures: Dict[
            str, tuple[_PauseTransitionIdentity, asyncio.Future]
        ] = {}
        self._prepared_pause_transitions: Dict[str, _PauseTransitionIdentity] = {}
        self._confirmed_pause_transitions: Dict[str, _PauseTransitionIdentity] = {}
        self._committed_pause_transitions: Dict[str, _PauseTransitionIdentity] = {}
        self._poisoned_pause_transitions: Dict[str, _PauseTransitionIdentity] = {}
        self._accepted_pause_transition_finalizations: Dict[
            str, _PauseTransitionIdentity
        ] = {}
        self._latest_pause_transitions: Dict[str, _PauseTransitionIdentity] = {}

        # Register PauseContinueBroadcastReq in the result dispatcher so
        # handle_loop routes it to _handle_pause_continue_broadcast
        from sglang.utils import TypeBasedDispatcher

        self._result_dispatcher += TypeBasedDispatcher(
            [(PauseContinueBroadcastReq, self._handle_pause_continue_broadcast)]
        )
        self._dispatch_to_scheduler(self._worker_registration_request())

    def _worker_registration_request(
        self,
        *,
        unregister: bool = False,
    ) -> TokenizerWorkerRegistrationReq:
        return TokenizerWorkerRegistrationReq(
            worker_ipc_name=self.tokenizer_ipc_name,
            worker_pid=self.worker_id,
            process_start_time=self._worker_process_start_time,
            worker_token=self._worker_token,
            unregister=unregister,
        )

    async def wait_for_router_registration(self, *, timeout_sec: float) -> None:
        if self._router_registration_result is True:
            return
        if self._router_registration_result is False:
            raise RuntimeError("tokenizer worker registration was rejected")
        self.auto_create_handle_loop()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._router_registration_future = future
        if self._router_registration_result is not None:
            future.set_result(self._router_registration_result)
        deadline = loop.time() + timeout_sec
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError("tokenizer worker registration timed out")
                try:
                    accepted = await asyncio.wait_for(
                        asyncio.shield(future),
                        timeout=min(
                            _WORKER_REGISTRATION_RETRY_INTERVAL_SEC,
                            remaining,
                        ),
                    )
                    break
                except asyncio.TimeoutError:
                    self._dispatch_to_scheduler(self._worker_registration_request())
        finally:
            if self._router_registration_future is future:
                self._router_registration_future = None
            if not future.done():
                future.cancel()
        if not accepted:
            raise RuntimeError("tokenizer worker registration was rejected")

    def unregister_from_router(self) -> None:
        if self._router_unregistered:
            return
        self._router_unregistered = True
        self._dispatch_to_scheduler(self._worker_registration_request(unregister=True))

    async def _apply_worker_registration_result(
        self,
        obj: PauseContinueBroadcastReq,
        token: str,
    ) -> None:
        if token != self._worker_token:
            return
        if obj.http_worker_ipc == _WORKER_REGISTRATION_ACCEPTED:
            accepted = True
            async with self.is_pause_cond:
                self.is_pause = obj.is_pause
                if not self.is_pause:
                    self.is_pause_cond.notify_all()
        elif obj.http_worker_ipc == _WORKER_REGISTRATION_REJECTED:
            accepted = False
        else:
            return
        self._router_registration_result = accepted
        future = self._router_registration_future
        if future is not None and not future.done():
            future.set_result(accepted)

    async def _acquire_generation_pause(
        self,
        owner: str,
        obj: PauseGenerationReqInput,
    ) -> None:
        async with self._get_generation_pause_transition_lock():
            owners = self._get_generation_pause_owners()
            resume_pending = self._get_generation_pause_resume_pending()
            async with self.is_pause_cond:
                owners.add(owner)
                self.is_pause = True
            obj.rid = owner
            try:
                await self._pause_generation_impl(obj)
            except BaseException:
                resume_pending.add(owner)
                async with self.is_pause_cond:
                    self.is_pause = True
                raise

    async def _release_generation_pause(
        self,
        owner: str,
        obj: ContinueGenerationReqInput,
    ) -> None:
        async with self._get_generation_pause_transition_lock():
            owners = self._get_generation_pause_owners()
            resume_pending = self._get_generation_pause_resume_pending()
            if owner not in owners:
                resume_pending.discard(owner)
                return
            obj.rid = owner
            try:
                await self._continue_generation_impl(obj)
            except BaseException:
                if owner in owners:
                    resume_pending.add(owner)
                    async with self.is_pause_cond:
                        self.is_pause = True
                raise

    async def _run_pause_transition(
        self,
        obj: PauseGenerationReqInput | ContinueGenerationReqInput,
        *,
        owner: str,
        action: str,
        expected_state: bool,
    ) -> None:
        loop = asyncio.get_running_loop()
        identity = _new_pause_transition_identity(
            owner=owner,
            action=action,
            expected_state=expected_state,
        )
        future = loop.create_future()
        confirmation_future = loop.create_future()
        confirmation_received = False
        self._remember_pause_transition(identity)
        self._pause_continue_futures[identity.transition_id] = (identity, future)
        confirmation_futures = getattr(
            self,
            "_pause_continue_confirmation_futures",
            None,
        )
        if confirmation_futures is None:
            confirmation_futures = {}
            self._pause_continue_confirmation_futures = confirmation_futures
        confirmation_futures[identity.transition_id] = (
            identity,
            confirmation_future,
        )
        accepted_finalizations = self._get_accepted_pause_transition_finalizations()
        obj.rid = _encode_pause_transition(identity)
        try:
            timeout = max(
                0.0,
                (identity.deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000,
            )
            await asyncio.wait_for(
                self._async_dispatch_to_scheduler(obj),
                timeout=timeout,
            )
            timeout = max(
                0.0,
                (identity.deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000,
            )
            await asyncio.wait_for(
                asyncio.shield(confirmation_future),
                timeout=timeout,
            )
            confirmation_received = True
            terminal_timeout = max(
                _PAUSE_TRANSITION_RECOVERY_TIMEOUT_SEC,
                (identity.deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000,
            )
            await asyncio.wait_for(
                asyncio.shield(future),
                timeout=terminal_timeout,
            )
        except asyncio.TimeoutError as error:
            if (
                confirmation_received
                and accepted_finalizations.get(identity.transition_id) != identity
            ):
                await self._hold_generation_paused(identity)
            elif not confirmation_received:
                self._poison_pause_transition(identity)
            raise TimeoutError(
                "pause transition acknowledgement deadline expired"
            ) from error
        except BaseException:
            if (
                confirmation_received
                and accepted_finalizations.get(identity.transition_id) != identity
            ):
                await self._hold_generation_paused(identity)
            elif not confirmation_received:
                self._poison_pause_transition(identity)
            if future.done() and not future.cancelled():
                future.exception()
            raise
        finally:
            current = self._pause_continue_futures.get(identity.transition_id)
            if current is not None and current[1] is future:
                self._pause_continue_futures.pop(identity.transition_id)
            confirmation = confirmation_futures.get(identity.transition_id)
            if confirmation is not None and confirmation[1] is confirmation_future:
                confirmation_futures.pop(identity.transition_id)
            if not future.done():
                future.cancel()
            if not confirmation_future.done():
                confirmation_future.cancel()
            if accepted_finalizations.get(identity.transition_id) == identity:
                accepted_finalizations.pop(identity.transition_id, None)

    def _get_accepted_pause_transition_finalizations(
        self,
    ) -> Dict[str, _PauseTransitionIdentity]:
        accepted = getattr(
            self,
            "_accepted_pause_transition_finalizations",
            None,
        )
        if accepted is None:
            accepted = {}
            self._accepted_pause_transition_finalizations = accepted
        return accepted

    async def _hold_generation_paused(
        self,
        identity: _PauseTransitionIdentity,
    ) -> None:
        self._poison_pause_transition(identity)
        self._get_generation_pause_resume_pending().add(identity.owner)
        async with self.is_pause_cond:
            self._get_generation_pause_owners().add(identity.owner)
            self.is_pause = True

    def _poison_pause_transition(
        self,
        identity: _PauseTransitionIdentity,
    ) -> None:
        poisoned = getattr(self, "_poisoned_pause_transitions", None)
        if poisoned is None:
            poisoned = {}
            self._poisoned_pause_transitions = poisoned
        if len(poisoned) >= 4096:
            poisoned.pop(next(iter(poisoned)))
        poisoned[identity.transition_id] = identity
        prepared = getattr(self, "_prepared_pause_transitions", None)
        if prepared is not None:
            prepared.pop(identity.transition_id, None)
        confirmed = getattr(self, "_confirmed_pause_transitions", None)
        if confirmed is not None:
            confirmed.pop(identity.transition_id, None)

    def _remember_pause_transition(
        self,
        identity: _PauseTransitionIdentity,
    ) -> None:
        latest = getattr(self, "_latest_pause_transitions", None)
        if latest is None:
            latest = {}
            self._latest_pause_transitions = latest
        previous = latest.get(identity.owner)
        if (
            previous is not None
            and previous != identity
            and _same_pause_transition_correlation(previous, identity)
        ):
            for state_name in (
                "_prepared_pause_transitions",
                "_confirmed_pause_transitions",
                "_committed_pause_transitions",
                "_poisoned_pause_transitions",
            ):
                state = getattr(self, state_name, None)
                if state is not None and state.get(identity.transition_id) == previous:
                    state[identity.transition_id] = identity
            for future_name in (
                "_pause_continue_futures",
                "_pause_continue_confirmation_futures",
            ):
                futures = getattr(self, future_name, None)
                pending = (
                    None if futures is None else futures.get(identity.transition_id)
                )
                if pending is not None and pending[0] == previous:
                    futures[identity.transition_id] = (identity, pending[1])
            latest[identity.owner] = identity
            return
        if previous is not None and previous != identity:
            pending = getattr(self, "_pause_continue_futures", {}).get(
                previous.transition_id
            )
            if pending is not None and not pending[1].done():
                pending[1].set_exception(
                    RuntimeError("pause transition superseded by owner retry")
                )
            getattr(self, "_prepared_pause_transitions", {}).pop(
                previous.transition_id,
                None,
            )
            getattr(self, "_confirmed_pause_transitions", {}).pop(
                previous.transition_id,
                None,
            )
        if len(latest) >= 4096 and identity.owner not in latest:
            latest.pop(next(iter(latest)))
        latest[identity.owner] = identity

    async def _pause_generation_impl(self, obj: PauseGenerationReqInput):
        owner = obj.rid or _ADMIN_PAUSE_OWNER
        await self._run_pause_transition(
            obj,
            owner=owner,
            action="pause",
            expected_state=True,
        )

        if obj.mode == "abort":
            # Abort polling: only the originator checks its own lock state
            while True:
                self.abort_request(abort_all=True)
                is_locked = await self.model_update_lock.is_locked()
                if not is_locked:
                    break
                await asyncio.sleep(1.0)

    async def _continue_generation_impl(self, obj: ContinueGenerationReqInput):
        owner = obj.rid or _ADMIN_PAUSE_OWNER
        owners = self._get_generation_pause_owners()
        await self._run_pause_transition(
            obj,
            owner=owner,
            action="continue",
            expected_state=bool(owners - {owner}),
        )

    def _handle_pause_continue_broadcast(self, obj: PauseContinueBroadcastReq):
        """Called from handle_loop when a broadcast arrives from the router."""
        loop = asyncio.get_event_loop()
        loop.create_task(self._apply_pause_continue_broadcast(obj))

    async def _apply_pause_continue_broadcast(self, obj: PauseContinueBroadcastReq):
        registration_token = _decode_worker_registration(obj.rid)
        if registration_token is not None:
            await self._apply_worker_registration_result(
                obj,
                registration_token,
            )
            return
        identity = _decode_pause_transition(obj.rid)
        if identity is None:
            return
        prepared_transitions = getattr(
            self,
            "_prepared_pause_transitions",
            None,
        )
        if prepared_transitions is None:
            prepared_transitions = {}
            self._prepared_pause_transitions = prepared_transitions
        confirmed_transitions = getattr(
            self,
            "_confirmed_pause_transitions",
            None,
        )
        if confirmed_transitions is None:
            confirmed_transitions = {}
            self._confirmed_pause_transitions = confirmed_transitions
        committed_transitions = getattr(
            self,
            "_committed_pause_transitions",
            None,
        )
        if committed_transitions is None:
            committed_transitions = {}
            self._committed_pause_transitions = committed_transitions
        latest_transitions = getattr(
            self,
            "_latest_pause_transitions",
            None,
        )
        if latest_transitions is None:
            latest_transitions = {}
            self._latest_pause_transitions = latest_transitions
        poisoned_transitions = getattr(
            self,
            "_poisoned_pause_transitions",
            {},
        )
        latest = latest_transitions.get(identity.owner)
        if (
            latest is not None
            and latest != identity
            and _same_pause_transition_correlation(latest, identity)
        ):
            self._remember_pause_transition(identity)
        if obj.http_worker_ipc == _PAUSE_TRANSITION_FAILED:
            if committed_transitions.get(identity.transition_id) == identity:
                return
            latest = latest_transitions.get(identity.owner)
            if latest is not None and latest != identity:
                return
            self._remember_pause_transition(identity)
            prepared_transitions.pop(identity.transition_id, None)
            confirmed_transitions.pop(identity.transition_id, None)
            self._poison_pause_transition(identity)
            owners = self._get_generation_pause_owners()
            resume_pending = self._get_generation_pause_resume_pending()
            async with self.is_pause_cond:
                if identity.action == "pause":
                    owners.discard(identity.owner)
                    resume_pending.discard(identity.owner)
                    self.is_pause = obj.is_pause
                    if not self.is_pause:
                        self.is_pause_cond.notify_all()
                else:
                    owners.add(identity.owner)
                    resume_pending.add(identity.owner)
                    self.is_pause = True
            pending = self._pause_continue_futures.get(identity.transition_id)
            if pending is not None and pending[0] == identity and not pending[1].done():
                pending[1].set_exception(
                    RuntimeError("pause transition failed before confirmation")
                )
            confirmation = getattr(
                self,
                "_pause_continue_confirmation_futures",
                {},
            ).get(identity.transition_id)
            if (
                confirmation is not None
                and confirmation[0] == identity
                and not confirmation[1].done()
            ):
                confirmation[1].set_exception(
                    RuntimeError("pause transition failed before confirmation")
                )
            return

        if obj.http_worker_ipc == _PAUSE_TRANSITION_CONFIRMED:
            if latest_transitions.get(identity.owner) != identity:
                return
            if poisoned_transitions.get(identity.transition_id) == identity:
                return
            prepared = prepared_transitions.get(identity.transition_id)
            pending = self._pause_continue_futures.get(identity.transition_id)
            if prepared != identity:
                return
            if time.monotonic_ns() >= identity.deadline_monotonic_ns:
                self._poison_pause_transition(identity)
                self._get_generation_pause_resume_pending().add(identity.owner)
                async with self.is_pause_cond:
                    self._get_generation_pause_owners().add(identity.owner)
                    self.is_pause = True
                if (
                    pending is not None
                    and pending[0] == identity
                    and not pending[1].done()
                ):
                    pending[1].set_exception(asyncio.TimeoutError())
                return
            prepared_transitions.pop(identity.transition_id, None)
            confirmed_transitions[identity.transition_id] = identity
            await self._async_dispatch_to_scheduler(
                PauseContinueBroadcastReq(
                    rid=_encode_pause_transition_applied(identity),
                    is_pause=identity.expected_state,
                )
            )
            confirmation = getattr(
                self,
                "_pause_continue_confirmation_futures",
                {},
            ).get(identity.transition_id)
            if (
                confirmation is not None
                and confirmation[0] == identity
                and not confirmation[1].done()
            ):
                confirmation[1].set_result(True)
            return

        if obj.http_worker_ipc == _PAUSE_TRANSITION_FINALIZED:
            if (
                latest_transitions.get(identity.owner) != identity
                or committed_transitions.get(identity.transition_id) != identity
                or poisoned_transitions.get(identity.transition_id) == identity
            ):
                return
            accepted_finalizations = self._get_accepted_pause_transition_finalizations()
            if (
                len(accepted_finalizations) >= 4096
                and identity.transition_id not in accepted_finalizations
            ):
                accepted_finalizations.pop(next(iter(accepted_finalizations)))
            accepted_finalizations[identity.transition_id] = identity
            await self._async_dispatch_to_scheduler(
                PauseContinueBroadcastReq(
                    rid=_encode_pause_transition_finalized_ack(identity),
                    is_pause=identity.expected_state,
                )
            )
            pending = self._pause_continue_futures.get(identity.transition_id)
            if pending is not None and pending[0] == identity and not pending[1].done():
                pending[1].set_result(True)
            return

        if obj.http_worker_ipc == _PAUSE_TRANSITION_COMMITTED:
            if latest_transitions.get(identity.owner) != identity or (
                confirmed_transitions.get(identity.transition_id) != identity
                and committed_transitions.get(identity.transition_id) != identity
            ):
                return
            pending = self._pause_continue_futures.get(identity.transition_id)
            confirmed_transitions.pop(identity.transition_id, None)
            if (
                len(committed_transitions) >= 4096
                and identity.transition_id not in committed_transitions
            ):
                committed_transitions.pop(next(iter(committed_transitions)))
            committed_transitions[identity.transition_id] = identity
            owners = self._get_generation_pause_owners()
            resume_pending = self._get_generation_pause_resume_pending()
            async with self.is_pause_cond:
                if identity.action == "pause":
                    owners.add(identity.owner)
                else:
                    owners.discard(identity.owner)
                self.is_pause = identity.expected_state
                if not self.is_pause:
                    self.is_pause_cond.notify_all()
            resume_pending.discard(identity.owner)
            confirmation = getattr(
                self,
                "_pause_continue_confirmation_futures",
                {},
            ).get(identity.transition_id)
            if (
                confirmation is not None
                and confirmation[0] == identity
                and not confirmation[1].done()
            ):
                confirmation[1].set_result(True)
            await self._async_dispatch_to_scheduler(
                PauseContinueBroadcastReq(
                    rid=_encode_pause_transition_committed_ack(identity),
                    is_pause=identity.expected_state,
                )
            )
            return

        if obj.is_pause != identity.expected_state:
            return
        latest = latest_transitions.get(identity.owner)
        if (
            latest is not None
            and latest != identity
            and not _same_pause_transition_correlation(latest, identity)
            and latest.deadline_monotonic_ns >= identity.deadline_monotonic_ns
        ):
            return
        self._remember_pause_transition(identity)
        if poisoned_transitions.get(identity.transition_id) == identity:
            async with self.is_pause_cond:
                self._get_generation_pause_owners().add(identity.owner)
                self.is_pause = True
            return
        prepared_transitions[identity.transition_id] = identity
        async with self.is_pause_cond:
            self._get_generation_pause_owners().add(identity.owner)
            self.is_pause = True

        await self._async_dispatch_to_scheduler(
            PauseContinueBroadcastReq(
                rid=obj.rid,
                is_pause=identity.expected_state,
            )
        )


def get_tokenizer_worker_class(server_args: ServerArgs) -> Type[TokenizerWorker]:
    worker_class = server_args.get_tokenizer_worker_class()
    if not isinstance(worker_class, type) or not issubclass(
        worker_class, TokenizerWorker
    ):
        raise TypeError(
            "ServerArgs.get_tokenizer_worker_class() must return a TokenizerWorker "
            f"subclass, got {worker_class!r}"
        )

    return worker_class


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"MultiTokenizerRouter hit an exception: {traceback}")
        if hasattr(func, "__self__") and isinstance(
            func.__self__, MultiTokenizerRouter
        ):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


def get_main_process_id() -> int:
    """Get the main process ID."""
    return multiprocessing.current_process()._parent_pid


def write_to_shared_memory(obj, name: str) -> shared_memory.SharedMemory:
    """Write data to shared memory"""
    serialized = pickle.dumps(obj)
    size = len(serialized)
    try:
        # Try to open existing shared memory
        shm = shared_memory.SharedMemory(name=name)
        # If size is insufficient, close and recreate
        if shm.size < size:
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(create=True, size=size, name=name)
    except FileNotFoundError:
        # If not present, create new shared memory
        shm = shared_memory.SharedMemory(create=True, size=size, name=name)

    shm.buf[:size] = serialized
    return shm


def read_from_shared_memory(name: str) -> Any:
    """Read data from shared memory"""
    try:
        shm = shared_memory.SharedMemory(name=name)
        data = pickle.loads(bytes(shm.buf))
        shm.close()
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Shared memory {name} not found")


def write_data_for_multi_tokenizer(
    port_args: PortArgs, server_args: ServerArgs, scheduler_info: Dict
):
    """Write args information to share memory for multi-tokenizer"""
    # get main process ID
    main_pid = get_main_process_id()
    current_pid = os.getpid()
    logger.info(f"main process ID: {main_pid}, current process ID: {current_pid}")
    args = (port_args, server_args, scheduler_info)
    args_shm = write_to_shared_memory(args, f"multi_tokenizer_args_{current_pid}")
    args_shm.close()

    return args_shm
