from __future__ import annotations

import importlib
import math
import pickle
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

_COLLECTIVE_VALUE_VERSION = 1
_ROOT_CALL_VERSION = 1
_GATHER_VERSION = 1
_GATHER_STATUS_OK = 0
_GATHER_STATUS_MESSAGE_LIMIT = 1
_GATHER_STATUS_AGGREGATE_LIMIT = 2
_GATHER_HEADER_FIELDS = 4
_SCATTER_VERSION = 1
_SCATTER_STATUS_OK = 0
_SCATTER_STATUS_VALUES_INVALID = 1
_SCATTER_STATUS_MESSAGE_LIMIT = 2
_SCATTER_STATUS_AGGREGATE_LIMIT = 3
_SCATTER_HEADER_FIELDS = 5
_DEADLINE_VERSION = 1
_DEADLINE_WIRE_BYTES = 12
_DEFAULT_MAX_OBJECT_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_AGGREGATE_BYTES = 1024 * 1024 * 1024
_DEFAULT_MAX_RESIDENT_BYTES = 2 * 1024 * 1024 * 1024
_DEFAULT_CHUNK_BYTES = 1024 * 1024
_DEFAULT_MAX_COLLECTIVE_MEMBERS = 65536
_DEFAULT_COLLECTIVE_TIMEOUT_SEC = 300.0
_COLLECTIVE_POLL_INTERVAL_SEC = 0.01
_MAX_CONTROL_TENSOR_BYTES = _SCATTER_HEADER_FIELDS * 8
_COLLECTIVE_REAPER_INTERVAL_SEC = 0.05
_DECODE_STATUS_VERSION = 1
_DECODE_STATUS_FIELDS = 2


class BoundedObjectCollectiveError(RuntimeError):
    def __init__(
        self,
        phase: str,
        message: str,
        *,
        completion_unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.completion_unknown = completion_unknown


class _ExecutionContext(Protocol):
    deadline_unix_sec: float

    def cancelled(self) -> bool: ...

    def expired(self) -> bool: ...

    def remaining_seconds(self) -> float: ...


class _SerializationLimitExceeded(RuntimeError):
    pass


class _BoundedPickleWriter:
    def __init__(self, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self._max_bytes = max_bytes
        self._buffer = bytearray()

    @property
    def size(self) -> int:
        return len(self._buffer)

    def write(self, data: bytes | bytearray | memoryview) -> int:
        size = len(data)
        if size > self._max_bytes - len(self._buffer):
            raise _SerializationLimitExceeded(
                "serialized object exceeds the configured size limit"
            )
        self._buffer.extend(data)
        return size

    def getvalue(self) -> bytes:
        return bytes(self._buffer)


@dataclass(frozen=True)
class _SerializedCollectiveValue:
    version: int
    phase: str
    succeeded: bool
    value: Any
    error: str | None


@dataclass(frozen=True)
class _RootCallEnvelope:
    version: int
    phase: str
    succeeded: bool
    result: Any
    error: str | None
    completion_unknown: bool


@dataclass
class _PendingCollective:
    work: Any | None
    tensors: tuple[Any, ...]


def _collective_chunk_bytes(
    *,
    configured_chunk_bytes: int,
    max_object_bytes: int,
    max_aggregate_bytes: int,
    world_size: int,
) -> int:
    if type(world_size) is not int or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    return min(
        configured_chunk_bytes,
        max_object_bytes,
        max(1, max_aggregate_bytes // (world_size + 1)),
    )


def _format_exception(error: BaseException) -> str:
    error_type = type(error).__name__
    try:
        message = str(error)
    except BaseException:
        message = ""
    return f"{error_type}: {message}" if message else error_type


def _pickle_with_limit(value: Any, *, max_bytes: int) -> bytes:
    writer = _BoundedPickleWriter(max_bytes)
    pickle.Pickler(writer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return writer.getvalue()


def _serialize_collective_value(
    value: Any,
    *,
    phase: str,
    max_bytes: int,
) -> bytes:
    envelope = _SerializedCollectiveValue(
        version=_COLLECTIVE_VALUE_VERSION,
        phase=phase,
        succeeded=True,
        value=value,
        error=None,
    )
    try:
        return _pickle_with_limit(envelope, max_bytes=max_bytes)
    except _SerializationLimitExceeded:
        raise
    except BaseException as error:
        failure = _SerializedCollectiveValue(
            version=_COLLECTIVE_VALUE_VERSION,
            phase=phase,
            succeeded=False,
            value=None,
            error=_format_exception(error),
        )
        return _pickle_with_limit(failure, max_bytes=max_bytes)


class BoundedObjectCollectiveCoordinator:
    """Deadline-aware object collectives with bounded bytes and member count."""

    def __init__(
        self,
        group: Any = None,
        *,
        max_object_bytes: int = _DEFAULT_MAX_OBJECT_BYTES,
        max_aggregate_bytes: int = _DEFAULT_MAX_AGGREGATE_BYTES,
        max_resident_bytes: int = _DEFAULT_MAX_RESIDENT_BYTES,
        chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
        max_collective_members: int = _DEFAULT_MAX_COLLECTIVE_MEMBERS,
        error_type: type[RuntimeError] = BoundedObjectCollectiveError,
        context_factory: Callable[[], _ExecutionContext] | None = None,
        unresolved_label: str = "bounded object",
    ) -> None:
        phase = "initialize"
        if type(max_object_bytes) is not int or max_object_bytes <= 0:
            raise ValueError("max_object_bytes must be a positive integer")
        if type(max_aggregate_bytes) is not int or max_aggregate_bytes <= 0:
            raise ValueError("max_aggregate_bytes must be a positive integer")
        if type(max_resident_bytes) is not int or max_resident_bytes <= 0:
            raise ValueError("max_resident_bytes must be a positive integer")
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        if type(max_collective_members) is not int or max_collective_members <= 0:
            raise ValueError("max_collective_members must be a positive integer")
        if not isinstance(error_type, type) or not issubclass(error_type, RuntimeError):
            raise ValueError("error_type must be a RuntimeError type")
        if context_factory is not None and not callable(context_factory):
            raise ValueError("context_factory must be callable")
        if type(unresolved_label) is not str or not unresolved_label:
            raise ValueError("unresolved_label must be a non-empty string")

        try:
            distributed = importlib.import_module("torch.distributed")
        except BaseException as error:
            raise error_type(
                phase,
                f"torch.distributed is unavailable: {_format_exception(error)}",
            ) from error
        required = (
            "all_gather",
            "broadcast",
            "gather",
            "get_rank",
            "get_world_size",
            "is_initialized",
        )
        missing = [
            name for name in required if not callable(getattr(distributed, name, None))
        ]
        if missing:
            raise error_type(
                phase,
                "torch.distributed is missing required APIs: " + ", ".join(missing),
            )
        try:
            initialized = distributed.is_initialized()
        except BaseException as error:
            raise error_type(
                phase,
                "torch.distributed initialization check failed: "
                f"{_format_exception(error)}",
            ) from error
        if type(initialized) is not bool or not initialized:
            raise error_type(phase, "torch.distributed must be initialized")
        try:
            rank = distributed.get_rank(group=group)
            world_size = distributed.get_world_size(group=group)
        except BaseException as error:
            raise error_type(
                phase,
                f"torch.distributed rank discovery failed: {_format_exception(error)}",
            ) from error
        if type(world_size) is not int or world_size <= 0:
            raise error_type(
                phase,
                "torch.distributed world size must be a positive integer",
            )
        if type(rank) is not int or not 0 <= rank < world_size:
            raise error_type(
                phase,
                "torch.distributed rank is outside the process group",
            )
        if world_size > max_collective_members:
            raise ValueError(
                "process-group world size exceeds the collective member limit"
            )
        if max_aggregate_bytes < max(
            world_size * 8,
            _MAX_CONTROL_TENSOR_BYTES,
        ):
            raise ValueError(
                "max_aggregate_bytes is too small for collective size metadata"
            )
        if max_resident_bytes < _MAX_CONTROL_TENSOR_BYTES:
            raise ValueError(
                "max_resident_bytes is too small for collective control metadata"
            )

        self._distributed = distributed
        self._group = group
        self._rank = rank
        self._world_size = world_size
        self._max_object_bytes = max_object_bytes
        self._max_aggregate_bytes = max_aggregate_bytes
        self._max_resident_bytes = max_resident_bytes
        self._max_collective_members = max_collective_members
        self._chunk_bytes = min(
            _collective_chunk_bytes(
                configured_chunk_bytes=chunk_bytes,
                max_object_bytes=max_object_bytes,
                max_aggregate_bytes=max_aggregate_bytes,
                world_size=world_size,
            ),
            max(1, max_resident_bytes // (world_size + 2)),
        )
        self._error_type = error_type
        self._context_factory = context_factory
        self._unresolved_label = unresolved_label
        self._torch: Any | None = None
        self._poisoned: str | None = None
        self._pending_collectives: list[_PendingCollective] = []
        self._pending_collectives_lock = threading.Lock()
        self._reaper_thread: threading.Thread | None = None

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def poisoned(self) -> bool:
        return self._poisoned is not None

    def _error(
        self,
        phase: str,
        message: str,
        *,
        completion_unknown: bool = False,
    ) -> RuntimeError:
        return self._error_type(
            phase,
            message,
            completion_unknown=completion_unknown,
        )

    def reap_completed_collectives(self) -> int:
        with self._pending_collectives_lock:
            collectives = self._pending_collectives
            self._pending_collectives = []
        reaped = 0
        pending: list[_PendingCollective] = []
        for collective in collectives:
            work = collective.work
            if work is None:
                # No handle means completion cannot be proven.
                pending.append(collective)
                continue
            is_completed = getattr(work, "is_completed", None)
            wait = getattr(work, "wait", None)
            if not callable(is_completed) or not callable(wait):
                pending.append(collective)
                continue
            try:
                completed = bool(is_completed())
            except BaseException:
                pending.append(collective)
                continue
            if not completed:
                pending.append(collective)
                continue
            try:
                wait()
            except BaseException as error:
                # is_completed() is the ownership boundary. A terminal backend
                # failure no longer retains references to caller buffers.
                error.__traceback__ = None
                error.__cause__ = None
                error.__context__ = None
                reaped += 1
                continue
            reaped += 1
        with self._pending_collectives_lock:
            self._pending_collectives.extend(pending)
        return reaped

    def _reap_pending_collectives(self) -> None:
        while True:
            time.sleep(_COLLECTIVE_REAPER_INTERVAL_SEC)
            self.reap_completed_collectives()
            with self._pending_collectives_lock:
                if not any(
                    collective.work is not None
                    for collective in self._pending_collectives
                ):
                    self._reaper_thread = None
                    return

    def _retain_pending_collective(
        self,
        work: Any | None,
        tensors: tuple[Any, ...],
    ) -> None:
        with self._pending_collectives_lock:
            self._pending_collectives.append(
                _PendingCollective(work=work, tensors=tensors)
            )
            if work is None or (
                self._reaper_thread is not None and self._reaper_thread.is_alive()
            ):
                return
            self._reaper_thread = threading.Thread(
                target=self._reap_pending_collectives,
                name="bounded-object-collective-reaper",
                daemon=True,
            )
            self._reaper_thread.start()

    def _require_healthy(self, phase: str) -> None:
        if self._poisoned is not None:
            raise self._error(
                phase,
                self._poisoned,
                completion_unknown=True,
            )

    def _collective_context(
        self,
        phase: str,
        execution_context: _ExecutionContext | None,
    ) -> _ExecutionContext:
        context = execution_context
        if context is None:
            if self._context_factory is None:
                raise self._error(phase, "an execution context is required")
            context = self._context_factory()
        if context.expired():
            reason = "cancelled" if context.cancelled() else "deadline exceeded"
            raise self._poison(phase, reason)
        return context

    def _poison(
        self,
        phase: str,
        detail: str,
        *,
        work: Any | None = None,
        tensors: tuple[Any, ...] = (),
    ) -> RuntimeError:
        if work is not None or tensors:
            self._retain_pending_collective(work, tensors)
        if self._poisoned is None:
            self._poisoned = (
                f"{phase} left the {self._unresolved_label} process group "
                f"unresolved: {detail}; scheduler restart is required"
            )
        return self._error(
            phase,
            self._poisoned,
            completion_unknown=True,
        )

    def _torch_module(self, phase: str) -> Any:
        if self._torch is not None:
            return self._torch
        try:
            torch = importlib.import_module("torch")
        except BaseException as error:
            raise self._error(
                phase,
                f"torch is unavailable: {_format_exception(error)}",
            ) from error
        for name in ("empty", "frombuffer", "tensor", "uint8", "int64"):
            if not hasattr(torch, name):
                raise self._error(phase, f"torch is missing required API: {name}")
        self._torch = torch
        return torch

    def _try_serialized_value(
        self,
        value: Any,
        *,
        phase: str,
        max_bytes: int | None = None,
    ) -> tuple[bytes | None, int]:
        limit = self._max_object_bytes if max_bytes is None else max_bytes
        try:
            payload = _serialize_collective_value(
                value,
                phase=phase,
                max_bytes=limit,
            )
        except _SerializationLimitExceeded:
            return None, limit + 1
        return payload, len(payload)

    def _require_resident_budget(self, phase: str, estimated_bytes: int) -> None:
        if estimated_bytes > self._max_resident_bytes:
            raise self._error(
                phase,
                "collective resident buffers exceed the configured size limit",
            )

    def _decode_serialized_value(
        self,
        payload: bytes | bytearray | memoryview,
        *,
        phase: str,
    ) -> Any:
        try:
            envelope = pickle.loads(payload)
        except BaseException as error:
            raise self._error(
                phase,
                f"invalid serialized collective payload: {_format_exception(error)}",
            ) from error
        if type(envelope) is not _SerializedCollectiveValue:
            raise self._error(phase, "invalid serialized collective envelope")
        if envelope.version != _COLLECTIVE_VALUE_VERSION or envelope.phase != phase:
            raise self._error(
                phase,
                "serialized collective envelope does not match the operation",
            )
        if envelope.succeeded:
            if envelope.error is not None:
                raise self._error(
                    phase,
                    "invalid serialized collective success envelope",
                )
            return envelope.value
        if envelope.value is not None or not envelope.error:
            raise self._error(
                phase,
                "invalid serialized collective failure envelope",
            )
        raise self._error(phase, envelope.error)

    def _decode_consensus(
        self,
        *,
        phase: str,
        execution_context: _ExecutionContext,
        local_error: BaseException | None,
    ) -> None:
        if self.world_size == 1:
            if local_error is not None:
                raise local_error
            return
        torch = self._torch_module(phase)
        status = torch.tensor(
            [_DECODE_STATUS_VERSION, int(local_error is not None)],
            dtype=torch.int64,
            device="cpu",
        )
        statuses = [
            torch.empty(_DECODE_STATUS_FIELDS, dtype=torch.int64, device="cpu")
            for _ in range(self.world_size)
        ]
        self._start_all_gather(
            f"{phase}.decode_status",
            statuses,
            status,
            execution_context,
            retained=(),
        )
        decoded = tuple(
            tuple(int(value) for value in item.tolist()) for item in statuses
        )
        if any(
            len(item) != _DECODE_STATUS_FIELDS
            or item[0] != _DECODE_STATUS_VERSION
            or item[1] not in (0, 1)
            for item in decoded
        ):
            raise self._error(phase, "invalid collective decode status")
        failed_ranks = tuple(rank for rank, item in enumerate(decoded) if item[1] == 1)
        if not failed_ranks:
            return
        detail = f"collective payload decoding failed on ranks {failed_ranks}"
        if local_error is not None:
            detail = f"{detail}: {local_error}"
        raise self._error(phase, detail)

    def _decode_payloads_with_consensus(
        self,
        payloads: list[bytearray],
        *,
        phase: str,
        execution_context: _ExecutionContext,
    ) -> list[Any]:
        values: list[Any] = []
        local_error: BaseException | None = None
        try:
            for index, payload in enumerate(payloads):
                values.append(self._decode_serialized_value(payload, phase=phase))
                payloads[index] = bytearray()
        except BaseException as error:
            local_error = error
        finally:
            payloads.clear()
        self._decode_consensus(
            phase=phase,
            execution_context=execution_context,
            local_error=local_error,
        )
        return values

    def _decode_payload_with_consensus(
        self,
        payload: bytes | bytearray,
        *,
        phase: str,
        execution_context: _ExecutionContext,
    ) -> Any:
        value = None
        local_error: BaseException | None = None
        try:
            value = self._decode_serialized_value(payload, phase=phase)
        except BaseException as error:
            local_error = error
        self._decode_consensus(
            phase=phase,
            execution_context=execution_context,
            local_error=local_error,
        )
        return value

    def _byte_tensor(
        self,
        payload: bytes,
        *,
        padded_size: int | None = None,
        phase: str,
    ) -> tuple[Any, bytearray]:
        torch = self._torch_module(phase)
        backing = bytearray(payload)
        size = len(payload) if padded_size is None else padded_size
        if size < len(payload):
            raise self._error(phase, "tensor size is smaller than its payload")
        if not payload:
            tensor = torch.empty(size, dtype=torch.uint8, device="cpu")
            if size:
                tensor.zero_()
            return tensor, backing
        source = torch.frombuffer(backing, dtype=torch.uint8)
        if size == len(payload):
            return source, backing
        tensor = torch.empty(size, dtype=torch.uint8, device="cpu")
        tensor.zero_()
        tensor[: len(payload)].copy_(source)
        return tensor, backing

    @staticmethod
    def _tensor_bytes(tensor: Any, size: int) -> bytes:
        return tensor[:size].contiguous().numpy().tobytes()

    def _wait_collective(
        self,
        phase: str,
        work: Any,
        tensors: tuple[Any, ...],
        execution_context: _ExecutionContext,
    ) -> None:
        is_completed = getattr(work, "is_completed", None)
        wait = getattr(work, "wait", None)
        if not callable(is_completed) or not callable(wait):
            raise self._poison(
                phase,
                "async collective returned an invalid Work handle",
                work=work,
                tensors=tensors,
            )
        while True:
            try:
                complete = bool(is_completed())
            except BaseException as error:
                raise self._poison(
                    phase,
                    _format_exception(error),
                    work=work,
                    tensors=tensors,
                ) from error
            if complete:
                try:
                    wait()
                except BaseException as error:
                    raise self._poison(
                        phase,
                        _format_exception(error),
                        work=work,
                        tensors=tensors,
                    ) from error
                return
            if execution_context.expired():
                reason = (
                    "cancelled"
                    if execution_context.cancelled()
                    else "deadline exceeded"
                )
                raise self._poison(
                    phase,
                    reason,
                    work=work,
                    tensors=tensors,
                )
            time.sleep(
                min(
                    _COLLECTIVE_POLL_INTERVAL_SEC,
                    execution_context.remaining_seconds(),
                )
            )

    def _source_args(self, root: int = 0) -> dict[str, int]:
        return {"src": root} if self._group is None else {"group_src": root}

    def _destination_args(self, root: int = 0) -> dict[str, int]:
        return {"dst": root} if self._group is None else {"group_dst": root}

    def _start_all_gather(
        self,
        phase: str,
        outputs: list[Any],
        value: Any,
        execution_context: _ExecutionContext,
        *,
        retained: tuple[Any, ...],
    ) -> None:
        self._collective_context(phase, execution_context)
        try:
            work = self._distributed.all_gather(
                outputs,
                value,
                group=self._group,
                async_op=True,
            )
        except BaseException as error:
            raise self._poison(
                phase,
                _format_exception(error),
                tensors=(*retained, value, *outputs),
            ) from error
        self._wait_collective(
            phase,
            work,
            (*retained, value, *outputs),
            execution_context,
        )

    def _gather_tensor(
        self,
        phase: str,
        value: Any,
        outputs: list[Any] | None,
        execution_context: _ExecutionContext,
        *,
        dst: int,
        retained: tuple[Any, ...],
        error_prefix: str,
    ) -> None:
        tensors = (
            (*retained, value) if outputs is None else (*retained, value, *outputs)
        )
        self._collective_context(phase, execution_context)
        try:
            work = self._distributed.gather(
                value,
                gather_list=outputs,
                group=self._group,
                async_op=True,
                **self._destination_args(dst),
            )
        except BaseException as error:
            raise self._poison(
                phase,
                f"{error_prefix}: {_format_exception(error)}",
                tensors=tensors,
            ) from error
        self._wait_collective(phase, work, tensors, execution_context)

    def _start_broadcast(
        self,
        phase: str,
        tensor: Any,
        execution_context: _ExecutionContext,
        *,
        src: int = 0,
        retained: tuple[Any, ...],
        error_prefix: str | None = None,
    ) -> None:
        self._collective_context(phase, execution_context)
        try:
            work = self._distributed.broadcast(
                tensor,
                group=self._group,
                async_op=True,
                **self._source_args(src),
            )
        except BaseException as error:
            detail = _format_exception(error)
            if error_prefix is not None:
                detail = f"{error_prefix}: {detail}"
            raise self._poison(
                phase,
                detail,
                tensors=(*retained, tensor),
            ) from error
        self._wait_collective(
            phase,
            work,
            (*retained, tensor),
            execution_context,
        )

    def _start_scatter(
        self,
        phase: str,
        output: Any,
        inputs: list[Any] | None,
        execution_context: _ExecutionContext,
        *,
        src: int,
        retained: tuple[Any, ...],
    ) -> None:
        tensors = (
            (*retained, output) if inputs is None else (*retained, output, *inputs)
        )
        self._collective_context(phase, execution_context)
        scatter = getattr(self._distributed, "scatter", None)
        if not callable(scatter):
            raise self._error(
                phase,
                "torch.distributed is missing required API: scatter",
            )
        try:
            work = scatter(
                output,
                scatter_list=inputs,
                group=self._group,
                async_op=True,
                **self._source_args(src),
            )
        except BaseException as error:
            raise self._poison(
                phase,
                _format_exception(error),
                tensors=tensors,
            ) from error
        self._wait_collective(phase, work, tensors, execution_context)

    def _validate_root(self, root: int, *, name: str) -> None:
        if type(root) is not int or not 0 <= root < self.world_size:
            raise ValueError(f"{name} must be a valid group-local rank")

    def synchronize_deadline(
        self,
        *,
        phase: str,
        execution_context: _ExecutionContext,
        src: int = 0,
    ) -> float:
        self._validate_root(src, name="src")
        self._require_healthy(phase)
        context = self._collective_context(phase, execution_context)
        if self.world_size == 1:
            return float(context.deadline_unix_sec)
        wire = (
            struct.pack(
                "!Id",
                _DEADLINE_VERSION,
                float(context.deadline_unix_sec),
            )
            if self.rank == src
            else bytes(_DEADLINE_WIRE_BYTES)
        )
        tensor, backing = self._byte_tensor(wire, phase=phase)
        self._start_broadcast(
            phase,
            tensor,
            context,
            src=src,
            retained=(backing,),
            error_prefix="Failed to synchronize bounded collective deadline",
        )
        version, deadline_unix_sec = struct.unpack(
            "!Id",
            self._tensor_bytes(tensor, _DEADLINE_WIRE_BYTES),
        )
        if (
            version != _DEADLINE_VERSION
            or not math.isfinite(deadline_unix_sec)
            or deadline_unix_sec <= 0
        ):
            raise self._error(
                phase,
                "invalid bounded collective deadline response",
            )
        return deadline_unix_sec

    def all_gather_object(
        self,
        value: Any,
        *,
        phase: str,
        execution_context: _ExecutionContext | None = None,
    ) -> list[Any]:
        self._require_healthy(phase)
        if self.world_size == 1:
            self._collective_context(phase, execution_context)
            return [value]
        context = self._collective_context(phase, execution_context)
        torch = self._torch_module(phase)
        payload, local_size_value = self._try_serialized_value(value, phase=phase)
        local_size = torch.tensor(
            [local_size_value],
            dtype=torch.int64,
            device="cpu",
        )
        sizes_output = [
            torch.empty(1, dtype=torch.int64, device="cpu")
            for _ in range(self.world_size)
        ]
        self._start_all_gather(
            f"{phase}.size",
            sizes_output,
            local_size,
            context,
            retained=(),
        )
        sizes = tuple(int(item.item()) for item in sizes_output)
        if any(size <= 0 or size > self._max_object_bytes for size in sizes):
            raise self._error(
                phase,
                "serialized collective payload exceeds the configured size limit",
            )
        total_size = sum(sizes)
        if total_size > self._max_aggregate_bytes:
            raise self._error(
                phase,
                "serialized collective aggregate exceeds the configured size limit",
            )
        assert payload is not None
        self._require_resident_budget(
            phase,
            total_size
            + len(payload)
            + (self.world_size + 2) * self._chunk_bytes
            + self.world_size * 8,
        )
        received = [bytearray(size) for size in sizes]
        rounds = (max(sizes) + self._chunk_bytes - 1) // self._chunk_bytes
        for round_index in range(rounds):
            offset = round_index * self._chunk_bytes
            local_chunk, backing = self._byte_tensor(
                payload[offset : offset + self._chunk_bytes],
                padded_size=self._chunk_bytes,
                phase=phase,
            )
            gathered_chunks = [
                torch.empty(self._chunk_bytes, dtype=torch.uint8, device="cpu")
                for _ in range(self.world_size)
            ]
            self._start_all_gather(
                f"{phase}.payload",
                gathered_chunks,
                local_chunk,
                context,
                retained=(backing,),
            )
            for buffer, size, tensor in zip(
                received,
                sizes,
                gathered_chunks,
                strict=True,
            ):
                valid_size = min(self._chunk_bytes, max(0, size - offset))
                if valid_size:
                    buffer[offset : offset + valid_size] = self._tensor_bytes(
                        tensor,
                        valid_size,
                    )
            del gathered_chunks, local_chunk, backing
        return self._decode_payloads_with_consensus(
            received,
            phase=phase,
            execution_context=context,
        )

    def broadcast_object(
        self,
        value: Any,
        *,
        src: int,
        phase: str,
        execution_context: _ExecutionContext | None = None,
    ) -> Any:
        self._validate_root(src, name="src")
        self._require_healthy(phase)
        context = self._collective_context(phase, execution_context)
        if self.world_size == 1:
            return value
        torch = self._torch_module(phase)
        payload: bytes | None = None
        size = 0
        if self.rank == src:
            payload, size = self._try_serialized_value(value, phase=phase)
        size_tensor = torch.tensor([size], dtype=torch.int64, device="cpu")
        self._start_broadcast(
            f"{phase}.size",
            size_tensor,
            context,
            src=src,
            retained=(),
        )
        size = int(size_tensor.item())
        if size <= 0 or size > self._max_object_bytes:
            raise self._error(
                phase,
                "serialized collective payload exceeds the configured size limit",
            )
        self._require_resident_budget(phase, size * 3 + self.world_size * 8)
        wire = payload if payload is not None else b""
        payload_tensor, backing = self._byte_tensor(
            wire,
            padded_size=size,
            phase=phase,
        )
        self._start_broadcast(
            f"{phase}.payload",
            payload_tensor,
            context,
            src=src,
            retained=(backing,),
        )
        return self._decode_payload_with_consensus(
            self._tensor_bytes(payload_tensor, size),
            phase=phase,
            execution_context=context,
        )

    def _broadcast_object(
        self,
        value: Any,
        *,
        phase: str,
        execution_context: _ExecutionContext,
    ) -> Any:
        return self.broadcast_object(
            value,
            src=0,
            phase=phase,
            execution_context=execution_context,
        )

    @staticmethod
    def _gather_status_error(status: int) -> str:
        if status == _GATHER_STATUS_MESSAGE_LIMIT:
            return "serialized collective payload exceeds the configured size limit"
        if status == _GATHER_STATUS_AGGREGATE_LIMIT:
            return "serialized collective aggregate exceeds the configured size limit"
        return "invalid root gather admission response"

    def _broadcast_chunked_object(
        self,
        value: Any,
        *,
        phase: str,
        execution_context: _ExecutionContext,
        error_prefix: str,
        src: int = 0,
    ) -> Any:
        torch = self._torch_module(phase)
        root_value = value
        payload: bytes | None = None
        size = 0
        if self.rank == src:
            payload, size = self._try_serialized_value(
                value,
                phase=phase,
                max_bytes=self._max_aggregate_bytes,
            )
            if payload is None:
                root_value = _RootCallEnvelope(
                    version=_ROOT_CALL_VERSION,
                    phase=phase,
                    succeeded=False,
                    result=None,
                    error=(
                        "serialized terminal status exceeds the configured "
                        "aggregate size limit"
                    ),
                    completion_unknown=False,
                )
                payload, size = self._try_serialized_value(
                    root_value,
                    phase=phase,
                    max_bytes=self._max_aggregate_bytes,
                )
        size_tensor = torch.tensor([size], dtype=torch.int64, device="cpu")
        self._start_broadcast(
            f"{phase}.terminal_size",
            size_tensor,
            execution_context,
            src=src,
            retained=(),
            error_prefix=error_prefix,
        )
        size = int(size_tensor.item())
        if size <= 0 or size > self._max_aggregate_bytes:
            raise self._error(
                phase,
                "serialized terminal status exceeds the configured aggregate size limit",
            )
        self._require_resident_budget(
            phase,
            size + (0 if payload is None else len(payload)) + 2 * self._chunk_bytes,
        )
        received = None if self.rank == src else bytearray(size)
        wire = payload if payload is not None else b""
        for offset in range(0, size, self._chunk_bytes):
            wire_size = min(self._chunk_bytes, size - offset)
            tensor, backing = self._byte_tensor(
                wire[offset : offset + wire_size],
                padded_size=wire_size,
                phase=phase,
            )
            self._start_broadcast(
                f"{phase}.terminal_payload",
                tensor,
                execution_context,
                src=src,
                retained=(backing,),
                error_prefix=error_prefix,
            )
            if received is not None:
                received[offset : offset + wire_size] = self._tensor_bytes(
                    tensor,
                    wire_size,
                )
        if received is None:
            local_value = root_value
            local_error = None
        else:
            local_value = None
            local_error = None
            try:
                local_value = self._decode_serialized_value(received, phase=phase)
            except BaseException as error:
                local_error = error
        self._decode_consensus(
            phase=phase,
            execution_context=execution_context,
            local_error=local_error,
        )
        return local_value

    def _decode_root_call(self, value: Any, *, phase: str) -> Any:
        if type(value) is not _RootCallEnvelope:
            raise self._error(phase, "invalid root broadcast response")
        if (
            value.version != _ROOT_CALL_VERSION
            or value.phase != phase
            or type(value.succeeded) is not bool
        ):
            raise self._error(phase, "invalid root broadcast response")
        if value.succeeded:
            if value.error is not None or value.completion_unknown is not False:
                raise self._error(
                    phase,
                    "invalid root broadcast success response",
                )
            return value.result
        if value.result is not None or type(value.error) is not str or not value.error:
            raise self._error(phase, "invalid root broadcast error response")
        if type(value.completion_unknown) is not bool:
            raise self._error(phase, "invalid root broadcast error response")
        raise self._error(
            phase,
            value.error,
            completion_unknown=value.completion_unknown,
        )

    def _gather_validated_to_root(
        self,
        value: Any,
        *,
        phase: str,
        execution_context: _ExecutionContext | None,
        validator: Callable[[list[Any]], Any],
        error_prefix: str,
        root_result_only: bool = False,
        dst: int = 0,
    ) -> Any:
        self._validate_root(dst, name="dst")
        self._require_healthy(phase)
        context = self._collective_context(phase, execution_context)
        if self.world_size == 1:
            return validator([value])
        torch = self._torch_module(phase)
        payload, local_size_value = self._try_serialized_value(value, phase=phase)
        local_size = torch.tensor(
            [local_size_value],
            dtype=torch.int64,
            device="cpu",
        )
        gathered_sizes = (
            [
                torch.empty(1, dtype=torch.int64, device="cpu")
                for _ in range(self.world_size)
            ]
            if self.rank == dst
            else None
        )
        self._gather_tensor(
            f"{phase}.size",
            local_size,
            gathered_sizes,
            context,
            dst=dst,
            retained=(),
            error_prefix=error_prefix,
        )
        sizes: tuple[int, ...] = ()
        status = _GATHER_STATUS_OK
        rounds = 0
        total_size = 0
        if gathered_sizes is not None:
            sizes = tuple(int(item.item()) for item in gathered_sizes)
            if any(size <= 0 or size > self._max_object_bytes for size in sizes):
                status = _GATHER_STATUS_MESSAGE_LIMIT
            else:
                total_size = sum(sizes)
                if total_size > self._max_aggregate_bytes:
                    status = _GATHER_STATUS_AGGREGATE_LIMIT
                else:
                    rounds = (max(sizes) + self._chunk_bytes - 1) // self._chunk_bytes
        admission = torch.tensor(
            [_GATHER_VERSION, status, rounds, total_size],
            dtype=torch.int64,
            device="cpu",
        )
        self._start_broadcast(
            f"{phase}.admission",
            admission,
            context,
            src=dst,
            retained=(),
            error_prefix=error_prefix,
        )
        version, status, rounds, total_size = (int(item) for item in admission.tolist())
        if version != _GATHER_VERSION:
            raise self._error(phase, "invalid root gather admission response")
        if status != _GATHER_STATUS_OK:
            raise self._error(phase, self._gather_status_error(status))
        max_rounds = (
            self._max_object_bytes + self._chunk_bytes - 1
        ) // self._chunk_bytes
        if (
            rounds <= 0
            or rounds > max_rounds
            or total_size <= 0
            or total_size > self._max_aggregate_bytes
        ):
            raise self._error(phase, "invalid root gather admission response")
        assert payload is not None
        local_resident = len(payload) + 2 * self._chunk_bytes
        if self.rank == dst:
            local_resident += total_size + self.world_size * self._chunk_bytes
        self._require_resident_budget(phase, local_resident)
        received = [bytearray(size) for size in sizes] if self.rank == dst else None
        for round_index in range(rounds):
            offset = round_index * self._chunk_bytes
            local_chunk, backing = self._byte_tensor(
                payload[offset : offset + self._chunk_bytes],
                padded_size=self._chunk_bytes,
                phase=phase,
            )
            gathered_chunks = (
                [
                    torch.empty(
                        self._chunk_bytes,
                        dtype=torch.uint8,
                        device="cpu",
                    )
                    for _ in range(self.world_size)
                ]
                if self.rank == dst
                else None
            )
            self._gather_tensor(
                f"{phase}.payload",
                local_chunk,
                gathered_chunks,
                context,
                dst=dst,
                retained=(backing,),
                error_prefix=error_prefix,
            )
            if received is not None and gathered_chunks is not None:
                for buffer, size, tensor in zip(
                    received,
                    sizes,
                    gathered_chunks,
                    strict=True,
                ):
                    valid_size = min(self._chunk_bytes, max(0, size - offset))
                    if valid_size:
                        buffer[offset : offset + valid_size] = self._tensor_bytes(
                            tensor,
                            valid_size,
                        )
                del gathered_chunks
            del local_chunk, backing
        root_result = None
        terminal = None
        if received is not None:
            try:
                gathered = []
                for index, buffer in enumerate(received):
                    gathered.append(self._decode_serialized_value(buffer, phase=phase))
                    received[index] = bytearray()
                root_result = validator(gathered)
            except BaseException as error:
                terminal = _RootCallEnvelope(
                    version=_ROOT_CALL_VERSION,
                    phase=phase,
                    succeeded=False,
                    result=None,
                    error=(
                        str(error)
                        if isinstance(error, self._error_type)
                        else _format_exception(error)
                    ),
                    completion_unknown=bool(
                        getattr(error, "completion_unknown", False)
                    ),
                )
            else:
                terminal = _RootCallEnvelope(
                    version=_ROOT_CALL_VERSION,
                    phase=phase,
                    succeeded=True,
                    result=None if root_result_only else root_result,
                    error=None,
                    completion_unknown=False,
                )
            finally:
                received.clear()
        terminal = self._broadcast_chunked_object(
            terminal,
            phase=phase,
            execution_context=context,
            error_prefix=error_prefix,
            src=dst,
        )
        self._decode_root_call(terminal, phase=phase)
        if root_result_only:
            return root_result if self.rank == dst else None
        return root_result if self.rank == dst else terminal.result

    def gather_object(
        self,
        value: Any,
        *,
        dst: int,
        phase: str,
        execution_context: _ExecutionContext | None = None,
    ) -> list[Any] | None:
        return self._gather_validated_to_root(
            value,
            phase=phase,
            execution_context=execution_context,
            validator=list,
            error_prefix="Failed to gather bounded objects",
            root_result_only=True,
            dst=dst,
        )

    def gather_object_to_root(
        self,
        value: Any,
        *,
        phase: str,
        execution_context: _ExecutionContext | None = None,
    ) -> tuple[Any, ...] | None:
        return self._gather_validated_to_root(
            value,
            phase=phase,
            execution_context=execution_context,
            validator=tuple,
            error_prefix="torch.distributed root gather failed",
            root_result_only=True,
        )

    @staticmethod
    def _scatter_status_error(status: int) -> str:
        if status == _SCATTER_STATUS_VALUES_INVALID:
            return "root scatter values must match the process-group world size"
        if status == _SCATTER_STATUS_MESSAGE_LIMIT:
            return "serialized collective payload exceeds the configured size limit"
        if status == _SCATTER_STATUS_AGGREGATE_LIMIT:
            return "serialized collective aggregate exceeds the configured size limit"
        return "invalid root scatter admission response"

    def scatter_object(
        self,
        values: list[Any] | tuple[Any, ...] | None,
        *,
        src: int,
        phase: str,
        execution_context: _ExecutionContext | None = None,
    ) -> Any:
        self._validate_root(src, name="src")
        self._require_healthy(phase)
        context = self._collective_context(phase, execution_context)
        if self.world_size == 1:
            if (
                not isinstance(values, (tuple, list))
                or isinstance(values, (str, bytes, bytearray))
                or len(values) != 1
            ):
                raise self._error(
                    phase,
                    "root scatter values must match the process-group world size",
                )
            return values[0]
        torch = self._torch_module(phase)
        payloads: tuple[bytes, ...] = ()
        sizes: tuple[int, ...] = ()
        status = _SCATTER_STATUS_OK
        rounds = 0
        total_size = 0
        max_size = 0
        if self.rank == src:
            if (
                not isinstance(values, (tuple, list))
                or isinstance(values, (str, bytes, bytearray))
                or len(values) != self.world_size
            ):
                status = _SCATTER_STATUS_VALUES_INVALID
            else:
                serialized: list[bytes] = []
                for value in values:
                    remaining = self._max_aggregate_bytes - total_size
                    if remaining <= 0:
                        status = _SCATTER_STATUS_AGGREGATE_LIMIT
                        break
                    payload, size = self._try_serialized_value(
                        value,
                        phase=phase,
                        max_bytes=min(self._max_object_bytes, remaining),
                    )
                    if payload is None:
                        status = (
                            _SCATTER_STATUS_AGGREGATE_LIMIT
                            if remaining < self._max_object_bytes
                            else _SCATTER_STATUS_MESSAGE_LIMIT
                        )
                        break
                    serialized.append(payload)
                    total_size += size
                if status == _SCATTER_STATUS_OK:
                    payloads = tuple(serialized)
                    sizes = tuple(len(payload) for payload in payloads)
                    max_size = max(sizes)
                    rounds = (max_size + self._chunk_bytes - 1) // self._chunk_bytes
        admission = torch.tensor(
            [_SCATTER_VERSION, status, rounds, total_size, max_size],
            dtype=torch.int64,
            device="cpu",
        )
        self._start_broadcast(
            f"{phase}.admission",
            admission,
            context,
            src=src,
            retained=(),
            error_prefix="torch.distributed root scatter failed",
        )
        version, status, rounds, total_size, max_size = (
            int(item) for item in admission.tolist()
        )
        if version != _SCATTER_VERSION:
            raise self._error(phase, "invalid root scatter admission response")
        if status != _SCATTER_STATUS_OK:
            raise self._error(phase, self._scatter_status_error(status))
        max_rounds = (
            self._max_object_bytes + self._chunk_bytes - 1
        ) // self._chunk_bytes
        if (
            rounds <= 0
            or rounds > max_rounds
            or total_size <= 0
            or total_size > self._max_aggregate_bytes
            or max_size <= 0
            or max_size > self._max_object_bytes
        ):
            raise self._error(phase, "invalid root scatter admission response")
        local_size_tensor = torch.empty(1, dtype=torch.int64, device="cpu")
        size_inputs = (
            [torch.tensor([size], dtype=torch.int64, device="cpu") for size in sizes]
            if self.rank == src
            else None
        )
        self._start_scatter(
            f"{phase}.size",
            local_size_tensor,
            size_inputs,
            context,
            src=src,
            retained=(),
        )
        local_size = int(local_size_tensor.item())
        if local_size <= 0 or local_size > self._max_object_bytes:
            raise self._error(phase, "invalid root scatter payload size")
        resident_bytes = local_size + 2 * self._chunk_bytes
        if self.rank == src:
            resident_bytes += total_size + self.world_size * self._chunk_bytes
        self._require_resident_budget(phase, resident_bytes)
        received = bytearray(local_size)
        for round_index in range(rounds):
            offset = round_index * self._chunk_bytes
            output = torch.empty(
                self._chunk_bytes,
                dtype=torch.uint8,
                device="cpu",
            )
            inputs = None
            retained: tuple[Any, ...] = ()
            if self.rank == src:
                tensors = []
                backings = []
                for payload in payloads:
                    tensor, backing = self._byte_tensor(
                        payload[offset : offset + self._chunk_bytes],
                        padded_size=self._chunk_bytes,
                        phase=phase,
                    )
                    tensors.append(tensor)
                    backings.append(backing)
                inputs = tensors
                retained = tuple(backings)
            self._start_scatter(
                f"{phase}.payload",
                output,
                inputs,
                context,
                src=src,
                retained=retained,
            )
            valid_size = min(self._chunk_bytes, max(0, local_size - offset))
            if valid_size:
                received[offset : offset + valid_size] = self._tensor_bytes(
                    output,
                    valid_size,
                )
            del output, inputs, retained
        return self._decode_payload_with_consensus(
            received,
            phase=phase,
            execution_context=context,
        )

    def scatter_object_from_root(
        self,
        values: tuple[Any, ...] | list[Any] | None,
        *,
        phase: str,
        execution_context: _ExecutionContext | None = None,
    ) -> Any:
        return self.scatter_object(
            values,
            src=0,
            phase=phase,
            execution_context=execution_context,
        )
