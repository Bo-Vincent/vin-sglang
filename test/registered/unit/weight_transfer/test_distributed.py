from __future__ import annotations

import gc
import pickle
import threading
import time
import weakref
from dataclasses import FrozenInstanceError, dataclass, field
from datetime import timedelta
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest.mock import call, patch

import pytest
from sglang.srt.distributed.bounded_object_collectives import (
    _DEFAULT_MAX_OBJECT_BYTES,
    _serialize_collective_value,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_MODULE_NAME = "sglang.srt.weight_transfer.distributed"
try:
    distributed = import_module(_MODULE_NAME)
except ModuleNotFoundError as error:
    if error.name != _MODULE_NAME:
        raise
    distributed = None
    _IMPORT_ERROR: ModuleNotFoundError | None = error
else:
    _IMPORT_ERROR = None

requires_distributed = pytest.mark.skipif(
    distributed is None,
    reason="distributed coordinator is not implemented",
)


def _serialized_value(value: Any, *, phase: str) -> bytes:
    return _serialize_collective_value(
        value,
        phase=phase,
        max_bytes=_DEFAULT_MAX_OBJECT_BYTES,
    )


@dataclass
class _BroadcastBus:
    payloads: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class _TensorCollectiveCall:
    kind: str
    input_dtype: Any
    output_dtypes: tuple[Any, ...]
    async_op: bool
    src: int | None = None
    group_src: int | None = None


@dataclass(frozen=True)
class _RootGatherCall:
    kind: str
    input_numel: int
    output_numels: tuple[int, ...]
    async_op: bool


class _FakeWork:
    def __init__(
        self,
        completions: tuple[bool, ...] = (True,),
        *,
        poll_error: BaseException | None = None,
        wait_error: BaseException | None = None,
        on_poll: Any = None,
    ) -> None:
        self._completions = list(completions)
        self._last_completion = completions[-1] if completions else False
        self._poll_error = poll_error
        self._wait_error = wait_error
        self._on_poll = on_poll
        self.events: list[str] = []
        self.wait_count = 0

    def is_completed(self) -> bool:
        self.events.append("is_completed")
        if self._on_poll is not None:
            self._on_poll()
        if self._poll_error is not None:
            raise self._poll_error
        if self._completions:
            self._last_completion = self._completions.pop(0)
        return self._last_completion

    def wait(self) -> None:
        self.events.append("wait")
        self.wait_count += 1
        if self._wait_error is not None:
            raise self._wait_error


class _FakeDistributed:
    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        group: Any,
        bus: _BroadcastBus,
        initialized: bool = True,
        gathered: list[Any] | None = None,
        object_gather_error: BaseException | None = None,
        object_broadcast_error: BaseException | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.bus = bus
        self.initialized = initialized
        self.gathered = gathered
        self.object_gather_error = object_gather_error
        self.object_broadcast_error = object_broadcast_error
        self.broadcast_index = 0
        self.broadcast_calls: list[tuple[int | None, int | None, Any]] = []
        self.gather_inputs: list[Any] = []
        self.tensor_collective_calls: list[_TensorCollectiveCall] = []

    def is_initialized(self) -> bool:
        return self.initialized

    def get_rank(self, group: Any = None) -> int:
        assert group is self.group
        return self.rank

    def get_world_size(self, group: Any = None) -> int:
        assert group is self.group
        return self.world_size

    def broadcast_object_list(
        self,
        object_list: list[Any],
        src: int | None = None,
        group: Any = None,
        *,
        group_src: int | None = None,
    ) -> None:
        assert group is self.group
        assert (src is None) != (group_src is None)
        if self.object_broadcast_error is not None:
            raise self.object_broadcast_error
        source_rank = group_src if group_src is not None else src
        assert source_rank == 0
        index = self.broadcast_index
        if self.rank == 0:
            assert index == len(self.bus.payloads)
            self.bus.payloads.append(object_list[0])
        else:
            object_list[0] = self.bus.payloads[index]
        self.broadcast_index += 1
        self.broadcast_calls.append((src, group_src, group))

    def all_gather_object(
        self,
        object_list: list[Any],
        value: Any,
        group: Any = None,
    ) -> None:
        assert group is self.group
        if self.object_gather_error is not None:
            raise self.object_gather_error
        assert self.gathered is not None
        self.gather_inputs.append(value)
        object_list[:] = self.gathered

    def all_gather(
        self,
        outputs: list[Any],
        value: Any,
        group: Any = None,
        *,
        async_op: bool = False,
    ) -> Any:
        assert group is self.group
        self.tensor_collective_calls.append(
            _TensorCollectiveCall(
                kind="all_gather",
                input_dtype=value.dtype,
                output_dtypes=tuple(output.dtype for output in outputs),
                async_op=async_op,
            )
        )
        if value.dtype == import_module("torch").int64 and value.numel() == 2:
            for output in outputs:
                output.copy_(value)
            return _FakeWork()
        raise AssertionError("legacy path used tensor all_gather")

    def gather(
        self,
        value: Any,
        gather_list: list[Any] | None = None,
        dst: int | None = None,
        group: Any = None,
        *,
        group_dst: int | None = None,
        async_op: bool = False,
    ) -> Any:
        del value, gather_list, dst, group_dst, async_op
        assert group is self.group
        if self.object_gather_error is not None:
            raise self.object_gather_error
        raise AssertionError("legacy path used tensor gather")

    def broadcast(
        self,
        tensor: Any,
        src: int | None = None,
        group: Any = None,
        *,
        group_src: int | None = None,
        async_op: bool = False,
    ) -> Any:
        assert group is self.group
        assert (src is None) != (group_src is None)
        if self.object_broadcast_error is not None:
            raise self.object_broadcast_error
        self.tensor_collective_calls.append(
            _TensorCollectiveCall(
                kind="broadcast",
                input_dtype=tensor.dtype,
                output_dtypes=(),
                async_op=async_op,
                src=src,
                group_src=group_src,
            )
        )
        source_rank = group_src if group_src is not None else src
        assert source_rank == 0
        index = self.broadcast_index
        if self.rank == 0:
            assert index == len(self.bus.payloads)
            self.bus.payloads.append(tensor.clone())
        else:
            tensor.copy_(self.bus.payloads[index])
        self.broadcast_index += 1
        return _FakeWork()


class _TensorDistributed(_FakeDistributed):
    def __init__(
        self,
        *,
        torch: Any,
        rank: int,
        world_size: int,
        group: Any,
        gather_payloads: tuple[bytes, ...] = (),
        broadcast_payload: bytes | None = None,
        works: tuple[_FakeWork, ...] = (),
        launch_errors: dict[int, BaseException] | None = None,
        decode_fail_ranks: tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            rank=rank,
            world_size=world_size,
            group=group,
            bus=_BroadcastBus(),
        )
        self.torch = torch
        self.gather_payloads = gather_payloads
        self.broadcast_payload = broadcast_payload
        self.works = list(works)
        self.launch_errors = dict(launch_errors or {})
        self.decode_fail_ranks = frozenset(decode_fail_ranks)
        self.tensor_refs: list[weakref.ReferenceType[Any]] = []
        self._payload_offset = 0

    def _next_work(self) -> _FakeWork:
        if self.works:
            return self.works.pop(0)
        return _FakeWork()

    def _record_tensor_refs(self, *tensors: Any) -> None:
        self.tensor_refs = [weakref.ref(tensor) for tensor in tensors]

    def _maybe_raise_launch_error(self) -> None:
        index = len(self.tensor_collective_calls) - 1
        error = self.launch_errors.get(index)
        if error is not None:
            raise error

    def all_gather(
        self,
        outputs: list[Any],
        value: Any,
        group: Any = None,
        *,
        async_op: bool = False,
    ) -> _FakeWork:
        assert group is self.group
        self.tensor_collective_calls.append(
            _TensorCollectiveCall(
                kind="all_gather",
                input_dtype=value.dtype,
                output_dtypes=tuple(output.dtype for output in outputs),
                async_op=async_op,
            )
        )
        self._record_tensor_refs(value, *outputs)
        self._maybe_raise_launch_error()
        if value.dtype == self.torch.int64:
            if value.numel() == 2:
                local_failed = int(value[1].item())
                for rank, output in enumerate(outputs):
                    output.copy_(
                        self.torch.tensor(
                            [
                                1,
                                (
                                    local_failed
                                    if rank == self.rank
                                    else int(rank in self.decode_fail_ranks)
                                ),
                            ],
                            dtype=self.torch.int64,
                        )
                    )
            else:
                assert len(outputs) == len(self.gather_payloads)
                for output, payload in zip(outputs, self.gather_payloads, strict=True):
                    output.fill_(len(payload))
        else:
            assert value.dtype == self.torch.uint8
            for output, payload in zip(outputs, self.gather_payloads, strict=True):
                output.zero_()
                chunk = payload[
                    self._payload_offset : self._payload_offset + value.numel()
                ]
                if chunk:
                    output[: len(chunk)].copy_(
                        self.torch.tensor(list(chunk), dtype=self.torch.uint8)
                    )
            self._payload_offset += value.numel()
        return self._next_work()

    def broadcast(
        self,
        tensor: Any,
        src: int | None = None,
        group: Any = None,
        *,
        group_src: int | None = None,
        async_op: bool = False,
    ) -> _FakeWork:
        assert group is self.group
        assert self.broadcast_payload is not None
        assert (src is None) != (group_src is None)
        self.tensor_collective_calls.append(
            _TensorCollectiveCall(
                kind="broadcast",
                input_dtype=tensor.dtype,
                output_dtypes=(),
                async_op=async_op,
                src=src,
                group_src=group_src,
            )
        )
        self._record_tensor_refs(tensor)
        self._maybe_raise_launch_error()
        if tensor.dtype == self.torch.int64:
            tensor.fill_(len(self.broadcast_payload))
        else:
            assert tensor.dtype == self.torch.uint8
            tensor.copy_(
                self.torch.tensor(
                    list(self.broadcast_payload),
                    dtype=self.torch.uint8,
                )
            )
        return self._next_work()


class _RootGatherDistributed(_FakeDistributed):
    def __init__(
        self,
        *,
        torch: Any,
        rank: int,
        world_size: int,
        group: Any,
        gather_payloads: tuple[bytes, ...],
        broadcast_values: tuple[tuple[int, ...] | bytes, ...] = (),
    ) -> None:
        super().__init__(
            rank=rank,
            world_size=world_size,
            group=group,
            bus=_BroadcastBus(),
        )
        self.torch = torch
        self.gather_payloads = gather_payloads
        self.broadcast_values = broadcast_values
        self.gather_calls: list[_RootGatherCall] = []
        self.tensor_broadcast_numels: list[int] = []
        self._payload_round = 0
        self._tensor_broadcast_index = 0
        self.decode_consensus_calls = 0

    def all_gather_object(
        self,
        object_list: list[Any],
        value: Any,
        group: Any = None,
    ) -> None:
        del object_list, value, group
        raise AssertionError("outcome exchange used all_gather_object")

    def all_gather(
        self,
        outputs: list[Any],
        value: Any,
        group: Any = None,
        *,
        async_op: bool = False,
    ) -> Any:
        assert group is self.group
        assert value.dtype == self.torch.int64
        assert value.numel() == 2
        for output in outputs:
            output.copy_(self.torch.tensor([1, 0], dtype=self.torch.int64))
        self.decode_consensus_calls += 1
        return _FakeWork() if async_op else None

    def gather(
        self,
        value: Any,
        gather_list: list[Any] | None = None,
        dst: int | None = None,
        group: Any = None,
        *,
        group_dst: int | None = None,
        async_op: bool = False,
    ) -> _FakeWork | None:
        assert group is self.group
        assert (dst is None) != (group_dst is None)
        assert (group_dst if group_dst is not None else dst) == 0
        outputs = () if gather_list is None else tuple(gather_list)
        self.gather_calls.append(
            _RootGatherCall(
                kind="gather",
                input_numel=value.numel(),
                output_numels=tuple(output.numel() for output in outputs),
                async_op=async_op,
            )
        )
        if gather_list is not None:
            if value.dtype == self.torch.int64:
                assert len(gather_list) == len(self.gather_payloads)
                for output, payload in zip(
                    gather_list,
                    self.gather_payloads,
                    strict=True,
                ):
                    output.fill_(len(payload))
            else:
                assert value.dtype == self.torch.uint8
                offset = self._payload_round * value.numel()
                for output, payload in zip(
                    gather_list,
                    self.gather_payloads,
                    strict=True,
                ):
                    output.zero_()
                    chunk = payload[offset : offset + value.numel()]
                    if chunk:
                        output[: len(chunk)].copy_(
                            self.torch.tensor(
                                list(chunk),
                                dtype=self.torch.uint8,
                            )
                        )
                self._payload_round += 1
        return _FakeWork() if async_op else None

    def broadcast(
        self,
        tensor: Any,
        src: int | None = None,
        group: Any = None,
        *,
        group_src: int | None = None,
        async_op: bool = False,
    ) -> _FakeWork | None:
        assert group is self.group
        assert (src is None) != (group_src is None)
        assert (group_src if group_src is not None else src) == 0
        self.tensor_broadcast_numels.append(tensor.numel())
        if self.rank != 0:
            value = self.broadcast_values[self._tensor_broadcast_index]
            if tensor.dtype == self.torch.int64:
                assert isinstance(value, tuple)
                tensor.copy_(self.torch.tensor(value, dtype=self.torch.int64))
            else:
                assert tensor.dtype == self.torch.uint8
                assert isinstance(value, bytes)
                tensor.copy_(
                    self.torch.tensor(
                        list(value),
                        dtype=self.torch.uint8,
                    )
                )
        self._tensor_broadcast_index += 1
        return _FakeWork() if async_op else None


def _require_module() -> Any:
    assert distributed is not None
    return distributed


def _make_torch_coordinators(
    *,
    world_size: int,
    gathered: list[Any] | None = None,
) -> tuple[list[Any], list[_FakeDistributed], _BroadcastBus, Any]:
    module = _require_module()
    group = object()
    bus = _BroadcastBus()
    backends = [
        _FakeDistributed(
            rank=rank,
            world_size=world_size,
            group=group,
            bus=bus,
            gathered=gathered,
        )
        for rank in range(world_size)
    ]
    with patch("importlib.import_module", side_effect=backends) as loader:
        coordinators = [
            module.TorchDistributedWeightStoreCoordinator(group=group)
            for _ in range(world_size)
        ]
    assert loader.call_args_list == [
        call("torch.distributed") for _ in range(world_size)
    ]
    return coordinators, backends, bus, group


def _broadcast_bus_envelopes(bus: _BroadcastBus) -> list[Any]:
    module = _require_module()
    assert len(bus.payloads) % 2 == 0
    envelopes = []
    for index in range(0, len(bus.payloads), 2):
        size = int(bus.payloads[index].item())
        payload = bytes(bus.payloads[index + 1][:size].tolist())
        serialized = pickle.loads(payload)
        assert type(serialized) is module._SerializedCollectiveValue
        envelopes.append(serialized.value)
    return envelopes


def _make_tensor_coordinator(
    *,
    rank: int,
    world_size: int,
    gather_payloads: tuple[bytes, ...] = (),
    broadcast_payload: bytes | None = None,
    works: tuple[_FakeWork, ...] = (),
    launch_errors: dict[int, BaseException] | None = None,
    decode_fail_ranks: tuple[int, ...] = (),
    max_object_bytes: int = 64 * 1024 * 1024,
    max_aggregate_bytes: int = 1024 * 1024 * 1024,
    max_resident_bytes: int = 2 * 1024 * 1024 * 1024,
    chunk_bytes: int = 1024 * 1024,
    max_collective_members: int = 65536,
) -> tuple[Any, _TensorDistributed, Any]:
    module = _require_module()
    torch = import_module("torch")
    group = object()
    backend = _TensorDistributed(
        torch=torch,
        rank=rank,
        world_size=world_size,
        group=group,
        gather_payloads=gather_payloads,
        broadcast_payload=broadcast_payload,
        works=works,
        launch_errors=launch_errors,
        decode_fail_ranks=decode_fail_ranks,
    )
    with patch("importlib.import_module", return_value=backend) as loader:
        coordinator = module.TorchDistributedWeightStoreCoordinator(
            group=group,
            max_object_bytes=max_object_bytes,
            max_aggregate_bytes=max_aggregate_bytes,
            max_resident_bytes=max_resident_bytes,
            chunk_bytes=chunk_bytes,
            max_collective_members=max_collective_members,
        )
    loader.assert_called_once_with("torch.distributed")
    return coordinator, backend, torch


def _make_root_gather_coordinator(
    *,
    rank: int,
    world_size: int,
    gather_payloads: tuple[bytes, ...],
    broadcast_values: tuple[tuple[int, ...] | bytes, ...] = (),
    max_object_bytes: int,
    max_aggregate_bytes: int,
    chunk_bytes: int,
) -> tuple[Any, _RootGatherDistributed, Any]:
    module = _require_module()
    torch = import_module("torch")
    group = object()
    backend = _RootGatherDistributed(
        torch=torch,
        rank=rank,
        world_size=world_size,
        group=group,
        gather_payloads=gather_payloads,
        broadcast_values=broadcast_values,
    )
    with patch("importlib.import_module", return_value=backend):
        coordinator = module.TorchDistributedWeightStoreCoordinator(
            group=group,
            max_object_bytes=max_object_bytes,
            max_aggregate_bytes=max_aggregate_bytes,
            chunk_bytes=chunk_bytes,
        )
    return coordinator, backend, torch


def _root_gather_broadcast_values(
    *,
    module: Any,
    phase: str,
    gather_payloads: tuple[bytes, ...],
    terminal: Any,
    chunk_bytes: int,
) -> tuple[tuple[int, ...] | bytes, ...]:
    rounds = (max(map(len, gather_payloads)) + chunk_bytes - 1) // chunk_bytes
    terminal_payload = _serialized_value(
        terminal,
        phase=phase,
    )
    return (
        (1, 0, rounds, sum(map(len, gather_payloads))),
        (len(terminal_payload),),
        *(
            terminal_payload[offset : offset + chunk_bytes]
            for offset in range(0, len(terminal_payload), chunk_bytes)
        ),
    )


def _execution_context(
    *,
    cancel_signal: threading.Event | None = None,
) -> Any:
    module = _require_module()
    return module.WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 60,
        cancel_signal=cancel_signal,
    )


def test_bounded_pickle_writer_stops_at_configured_limit() -> None:
    bounded = import_module("sglang.srt.distributed.bounded_object_collectives")
    writer = bounded._BoundedPickleWriter(128)

    with pytest.raises(bounded._SerializationLimitExceeded):
        writer.write(b"x" * 129)

    assert writer.size == 0


def test_bounded_pickle_writer_uses_one_bounded_buffer() -> None:
    bounded = import_module("sglang.srt.distributed.bounded_object_collectives")
    writer = bounded._BoundedPickleWriter(4096)

    for _ in range(4096):
        assert writer.write(b"x") == 1

    assert writer.size == 4096
    assert isinstance(writer._buffer, bytearray)
    assert len(writer._buffer) == writer.size
    assert writer.getvalue() == b"x" * 4096


def test_collective_chunk_budget_caps_world_sized_tensor_allocation() -> None:
    bounded = import_module("sglang.srt.distributed.bounded_object_collectives")

    assert (
        bounded._collective_chunk_bytes(
            configured_chunk_bytes=1024 * 1024,
            max_object_bytes=64 * 1024 * 1024,
            max_aggregate_bytes=64 * 1024 * 1024,
            world_size=128,
        )
        == (64 * 1024 * 1024) // 129
    )


def _run_gloo_bounded_collectives(
    rank: int,
    world_size: int,
    init_method: str,
) -> None:
    torch = import_module("torch")
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    control_group = None
    try:
        control_group = torch.distributed.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
        )
        module = import_module(_MODULE_NAME)
        coordinator = module.TorchDistributedWeightStoreCoordinator(control_group)
        values = (
            {"rank": 0, "payload": "short"},
            {"rank": 1, "payload": ["longer", "value", 1]},
        )

        gathered = coordinator.all_gather_object(
            values[rank],
            phase="gloo.gather",
        )
        assert gathered == list(values)
        root_only = coordinator.gather_object_to_root(
            values[rank],
            phase="gloo.root_gather",
        )
        assert root_only == (values if rank == 0 else None)
        scatter_values = (
            tuple(
                {
                    "rank": target_rank,
                    "operations": (
                        tuple(f"operation-{index}" for index in range(world_size))
                        if target_rank == 0
                        else (f"operation-{target_rank}",)
                    ),
                    "summary": {
                        "operation_count": world_size,
                        "digest": "sha256:" + "a" * 64,
                    },
                }
                for target_rank in range(world_size)
            )
            if rank == 0
            else None
        )
        if scatter_values is not None:
            assert sum(len(item["operations"]) for item in scatter_values) == (
                2 * world_size - 1
            )
        local_selection = coordinator.scatter_object_from_root(
            scatter_values,
            phase="gloo.local_scatter",
        )
        assert local_selection["rank"] == rank
        assert len(local_selection["operations"]) == (world_size if rank == 0 else 1)
        assert local_selection["summary"]["operation_count"] == world_size
        preflight = module.WeightStorePreflightOutcome(
            rank=rank,
            error=(None if rank == 0 else "RuntimeError: injected rank failure"),
        )
        preflight_outcomes = coordinator.exchange_preflight_outcome(preflight)
        assert len(preflight_outcomes) == world_size
        with pytest.raises(module.WeightStoreDistributedError) as root_failure:
            coordinator.run_root(
                "gloo.rank_failure",
                lambda: (_ for _ in ()).throw(
                    RuntimeError(
                        "; ".join(
                            outcome.error
                            for outcome in preflight_outcomes
                            if outcome.error is not None
                        )
                    )
                ),
            )
        assert "injected rank failure" in str(root_failure.value)
        trace = (
            "root_gather",
            "rank_local_scatter",
            "preflight_exchange",
            "root_failure",
        )
        assert (
            coordinator.all_gather_object(
                trace,
                phase="gloo.failure_trace",
            )
            == [trace] * world_size
        )
        outcomes = tuple(
            module.WeightStoreUploadOutcome(
                rank=outcome_rank,
                placement_ids=(f"placement-{outcome_rank}",),
                receipts=("short" if outcome_rank == 0 else "remote-" + "x" * 4097,),
                error=None,
            )
            for outcome_rank in range(world_size)
        )
        exchanged = coordinator.exchange_upload_outcome(outcomes[rank])
        assert exchanged == (outcomes if rank == 0 else None)
        assert coordinator.run_root(
            "gloo.root",
            lambda: {"root": rank, "value": "published"},
        ) == {"root": 0, "value": "published"}
    finally:
        if control_group is not None:
            torch.distributed.destroy_process_group(control_group)
        torch.distributed.destroy_process_group()


def _run_gloo_decode_consensus(
    rank: int,
    world_size: int,
    init_method: str,
) -> None:
    torch = import_module("torch")
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    control_group = None
    try:
        control_group = torch.distributed.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
        )
        module = import_module(_MODULE_NAME)
        coordinator = module.TorchDistributedWeightStoreCoordinator(control_group)
        original_decode = coordinator._decode_serialized_value
        if rank == 1:

            def fail_decode(payload: bytes, *, phase: str) -> Any:
                del payload, phase
                raise ValueError("injected rank-local decode failure")

            coordinator._decode_serialized_value = fail_decode
        try:
            with pytest.raises(
                module.WeightStoreDistributedError,
                match="decoding failed on ranks",
            ) as raised:
                coordinator.all_gather_object(
                    {"rank": rank},
                    phase="gloo.decode_failure",
                )
            assert raised.value.completion_unknown is False
        finally:
            coordinator._decode_serialized_value = original_decode

        assert coordinator.all_gather_object(
            {"rank": rank},
            phase="gloo.after_decode_failure",
        ) == [{"rank": item} for item in range(world_size)]
    finally:
        if control_group is not None:
            torch.distributed.destroy_process_group(control_group)
        torch.distributed.destroy_process_group()


def _run_gloo_skewed_timeout(
    rank: int,
    world_size: int,
    init_method: str,
    marker_path: str,
) -> None:
    torch = import_module("torch")
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    control_group = None
    try:
        control_group = torch.distributed.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
        )
        module = import_module(_MODULE_NAME)
        coordinator = module.TorchDistributedWeightStoreCoordinator(control_group)
        context = module.WeightTransferExecutionContext(
            deadline_unix_sec=time.time() + 0.5,
        )
        if rank == 1:
            time.sleep(0.75)
        outcome = module.WeightStoreUploadOutcome(
            rank=rank,
            placement_ids=(f"placement-{rank}",),
            receipts=(f"receipt-{rank}",),
            error=None,
        )

        with pytest.raises(module.WeightStoreDistributedError) as raised:
            coordinator.exchange_upload_outcome(
                outcome,
                execution_context=context,
            )

        assert raised.value.completion_unknown is True
        assert coordinator.poisoned is True

        def commit() -> None:
            Path(marker_path).write_text("committed")

        with pytest.raises(module.WeightStoreDistributedError) as retry:
            coordinator.commit_upload(commit)
        assert retry.value.completion_unknown is True
        assert not Path(marker_path).exists()
    finally:
        if control_group is not None:
            torch.distributed.destroy_process_group(control_group)
        torch.distributed.destroy_process_group()


def _run_gloo_chunked_timeout_reap(
    rank: int,
    world_size: int,
    init_method: str,
) -> None:
    torch = import_module("torch")
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=10),
    )
    control_group = None
    try:
        control_group = torch.distributed.new_group(
            ranks=list(range(world_size)),
            backend="gloo",
        )
        bounded = import_module("sglang.srt.distributed.bounded_object_collectives")
        module = import_module(_MODULE_NAME)
        coordinator = bounded.BoundedObjectCollectiveCoordinator(
            control_group,
            max_object_bytes=8 * 1024,
            max_aggregate_bytes=16 * 1024,
            chunk_bytes=64,
        )
        context = module.WeightTransferExecutionContext(
            deadline_unix_sec=time.time() + 10,
        )
        values = coordinator.all_gather_object(
            {"rank": rank, "payload": "x" * (1024 + rank * 1024)},
            phase="gloo.chunked",
            execution_context=context,
        )
        assert [value["rank"] for value in values] == list(range(world_size))
        assert coordinator._chunk_bytes == 64

        torch.distributed.barrier(group=control_group)
        value = torch.tensor([rank], dtype=torch.int64)
        outputs = [torch.empty(1, dtype=torch.int64) for _ in range(world_size)]
        if rank == 1:
            time.sleep(0.3)
        context = module.WeightTransferExecutionContext(
            deadline_unix_sec=time.time() + (0.15 if rank == 0 else 5.0),
        )
        if rank == 0:
            with pytest.raises(
                bounded.BoundedObjectCollectiveError,
                match="deadline exceeded",
            ):
                coordinator._start_all_gather(
                    "gloo.timeout_reap",
                    outputs,
                    value,
                    context,
                    retained=(),
                )
            deadline = time.monotonic() + 5.0
            while coordinator._pending_collectives and time.monotonic() < deadline:
                time.sleep(0.01)
            assert coordinator._pending_collectives == []
        else:
            coordinator._start_all_gather(
                "gloo.timeout_reap",
                outputs,
                value,
                context,
                retained=(),
            )
        torch.distributed.barrier(group=control_group)
    finally:
        if control_group is not None:
            torch.distributed.destroy_process_group(control_group)
        torch.distributed.destroy_process_group()


def _factory_for_rank(
    rank: int,
    calls: list[int],
    result: Any,
):
    def factory() -> Any:
        calls.append(rank)
        if rank != 0:
            raise AssertionError("non-root factory was called")
        return result

    return factory


def test_distributed_module_is_available() -> None:
    assert _IMPORT_ERROR is None


@requires_distributed
def test_weight_store_coordinator_reuses_distributed_primitive() -> None:
    module = _require_module()
    common_methods = {
        "_byte_tensor",
        "_collective_context",
        "_decode_serialized_value",
        "_poison",
        "_start_all_gather",
        "_start_broadcast",
        "_start_scatter",
        "_wait_collective",
        "all_gather_object",
        "gather_object_to_root",
        "reap_completed_collectives",
        "scatter_object_from_root",
    }

    assert common_methods.isdisjoint(
        module.TorchDistributedWeightStoreCoordinator.__dict__
    )


@requires_distributed
def test_error_preserves_phase_and_message() -> None:
    module = _require_module()

    error = module.WeightStoreDistributedError("prepare_upload", "failed")

    assert error.phase == "prepare_upload"
    assert str(error) == "failed"


@requires_distributed
def test_upload_outcome_is_frozen_and_preserves_metadata() -> None:
    module = _require_module()
    receipt = object()
    outcome = module.WeightStoreUploadOutcome(
        rank=2,
        placement_ids=("placement-b", "placement-a"),
        receipts=(receipt,),
        error=None,
    )

    assert outcome.rank == 2
    assert outcome.placement_ids == ("placement-b", "placement-a")
    assert outcome.receipts == (receipt,)
    assert outcome.error is None
    with pytest.raises(FrozenInstanceError):
        outcome.rank = 3


@requires_distributed
def test_preflight_outcome_is_frozen_and_preserves_metadata() -> None:
    module = _require_module()
    outcome = module.WeightStorePreflightOutcome(
        rank=2,
        error="ValueError: local plan mismatch",
    )

    assert outcome.rank == 2
    assert outcome.error == "ValueError: local plan mismatch"
    with pytest.raises(FrozenInstanceError):
        outcome.rank = 3


@requires_distributed
@pytest.mark.parametrize("rank", [-1, True, 1.5, "1"])
def test_upload_outcome_rejects_invalid_rank(rank: Any) -> None:
    module = _require_module()

    with pytest.raises(ValueError, match="rank"):
        module.WeightStoreUploadOutcome(
            rank=rank,
            placement_ids=(),
            receipts=(),
            error=None,
        )


@requires_distributed
@pytest.mark.parametrize(
    "placement_ids",
    [
        ("",),
        (1,),
        ("placement-a", "placement-a"),
    ],
)
def test_upload_outcome_rejects_invalid_or_duplicate_placement_ids(
    placement_ids: tuple[Any, ...],
) -> None:
    module = _require_module()

    with pytest.raises(ValueError, match="placement"):
        module.WeightStoreUploadOutcome(
            rank=0,
            placement_ids=placement_ids,
            receipts=(),
            error=None,
        )


@requires_distributed
@pytest.mark.parametrize("error", ["", 1])
def test_upload_outcome_rejects_invalid_error(error: Any) -> None:
    module = _require_module()

    with pytest.raises(ValueError, match="error"):
        module.WeightStoreUploadOutcome(
            rank=0,
            placement_ids=(),
            receipts=(),
            error=error,
        )


@requires_distributed
def test_local_coordinator_executes_every_factory_and_exchanges_outcome() -> None:
    module = _require_module()
    coordinator = module.LocalWeightStoreDistributedCoordinator()
    calls: list[str] = []
    plan = object()
    manifest = object()
    outcome = module.WeightStoreUploadOutcome(
        rank=0,
        placement_ids=("placement-0",),
        receipts=("receipt-0",),
        error=None,
    )
    preflight = module.WeightStorePreflightOutcome(rank=0, error=None)

    assert coordinator.rank == 0
    assert coordinator.world_size == 1
    assert (
        coordinator.prepare_upload(lambda: calls.append("prepare_upload") or plan)
        is plan
    )
    assert coordinator.exchange_preflight_outcome(preflight) == (preflight,)
    assert coordinator.exchange_upload_outcome(outcome) == (outcome,)
    assert (
        coordinator.commit_upload(lambda: calls.append("commit_upload") or manifest)
        is manifest
    )
    assert coordinator.abort_upload(lambda: calls.append("abort_upload")) is None
    assert coordinator.finalize_upload(lambda: calls.append("finalize_upload")) is None
    assert calls == [
        "prepare_upload",
        "commit_upload",
        "abort_upload",
        "finalize_upload",
    ]


@requires_distributed
def test_local_coordinator_root_gather_and_scatter_preserve_world_size_one() -> None:
    module = _require_module()
    coordinator = module.LocalWeightStoreDistributedCoordinator()

    assert coordinator.gather_object_to_root(
        {"records": ("only",)},
        phase="local.gather",
    ) == ({"records": ("only",)},)
    assert coordinator.scatter_object_from_root(
        ({"records": ("only",)},),
        phase="local.scatter",
    ) == {"records": ("only",)}


@requires_distributed
def test_root_catalog_projects_full_snapshot_to_rank_local_state() -> None:
    module = _require_module()
    manifest_module = import_module("sglang.srt.model_executor.weight_runtime_manifest")
    contracts = import_module("sglang.srt.weight_transfer.contracts")
    storage = import_module("sglang.srt.weight_transfer.storage")
    placements = []
    bindings = []
    for index in range(2):
        tensor = manifest_module.WeightPlacementTensor(
            placement_fragment_id=f"placement:{index}:fragment",
            tensor_id=f"weight:{index}",
            runtime_name=f"weight_{index}",
            aliases=(f"weight_{index}",),
            global_shape=(8,),
            global_offset=(0,),
            local_shape=(8,),
            dtype="bfloat16",
            itemsize=2,
            partition_dim=None,
            shard_dims=(),
            layer_id=index,
            expert_id=None,
            layout_fingerprint="layout:v1",
            nbytes=16,
            byte_offset=0,
            rank=manifest_module.WeightParallelRank(pp=index),
        )
        placement = manifest_module.WeightPlacementManifest(
            model_id="model",
            revision="revision",
            placement_id=manifest_module.compute_weight_placement_id((tensor,)),
            tensors=(tensor,),
        )
        placements.append(placement)
        bindings.append(
            contracts.WeightStorageBindingManifest(
                model_id=placement.model_id,
                revision=placement.revision,
                placement_id=placement.placement_id,
                storage_id="weights/revision",
                provider="test-store",
                fragments=(
                    contracts.WeightStorageFragmentBinding(
                        placement_fragment_id=tensor.placement_fragment_id,
                        fragment_id=f"stored:{tensor.placement_fragment_id}",
                        object_key=f"objects/{placement.placement_id}",
                        object_offset=0,
                        nbytes=tensor.nbytes,
                        checksum=None,
                    ),
                ),
            )
        )
    full_snapshot = storage.StoredWeightSnapshot.create(
        provider="test-store",
        storage_id="weights/revision",
        manifest_key="weights/revision/manifest",
        placements=tuple(placements),
        storage_bindings=tuple(bindings),
    )
    intent = storage.WeightMaterializationIntent(
        provider="test-store",
        storage_id="weights/revision",
        object_prefix="weights/revision",
        model_id="model",
        revision="revision",
        source_digest=storage.weight_placement_set_digest(tuple(placements)),
        total_bytes=32,
        fragment_count=2,
    )
    root_state = storage.InMemoryWeightStorageCatalog()
    root_state.begin_materialization("materialize", intent)
    root_state.complete_materialization("materialize", full_snapshot)
    coordinator = module.LocalWeightStoreDistributedCoordinator()
    catalog = module.RootWeightStorageCatalog(root_state, coordinator)
    local_intent = storage.WeightMaterializationIntent(
        provider="test-store",
        storage_id="weights/revision",
        object_prefix="weights/revision",
        model_id="model",
        revision="revision",
        source_digest=storage.weight_placement_set_digest((placements[0],)),
        total_bytes=16,
        fragment_count=1,
    )
    catalog._projection_requests["materialize"] = module._CatalogProjectionRequest(
        rank=0,
        materialization_id="materialize",
        placement_ids=(placements[0].placement_id,),
        intent=local_intent,
    )
    catalog._projection_enabled = True

    projected = catalog.get_materialization("materialize")

    assert projected is not None
    assert projected.snapshot is not None
    assert projected.intent == local_intent
    assert projected.snapshot.placements == (placements[0],)
    assert len(root_state.get_materialization("materialize").snapshot.placements) == 2


@requires_distributed
def test_local_coordinator_run_root_supports_custom_phases_and_discard() -> None:
    module = _require_module()
    coordinator = module.LocalWeightStoreDistributedCoordinator()
    calls: list[str] = []

    assert coordinator.run_root(
        "catalog.lookup",
        lambda: calls.append("lookup") or {"value": 1},
    ) == {"value": 1}
    assert (
        coordinator.run_root(
            "catalog.cleanup",
            lambda: calls.append("cleanup") or object(),
            discard_result=True,
        )
        is None
    )
    assert calls == ["lookup", "cleanup"]


@requires_distributed
@pytest.mark.parametrize("phase", ["", None, 1, True])
def test_local_coordinator_run_root_rejects_invalid_phase(phase: Any) -> None:
    module = _require_module()
    coordinator = module.LocalWeightStoreDistributedCoordinator()
    called = False

    def factory() -> None:
        nonlocal called
        called = True

    with pytest.raises(ValueError, match="phase"):
        coordinator.run_root(phase, factory)

    assert called is False


@requires_distributed
@pytest.mark.parametrize(
    "method_name",
    [
        "prepare_upload",
        "commit_upload",
        "abort_upload",
        "finalize_upload",
    ],
)
def test_local_coordinator_wraps_factory_errors(method_name: str) -> None:
    module = _require_module()
    coordinator = module.LocalWeightStoreDistributedCoordinator()

    def fail() -> None:
        raise RuntimeError("root failed")

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        getattr(coordinator, method_name)(fail)

    assert raised.value.phase == method_name
    assert "root failed" in str(raised.value)


@requires_distributed
def test_torch_coordinator_requires_initialized_distributed() -> None:
    module = _require_module()
    group = object()
    backend = _FakeDistributed(
        rank=0,
        world_size=1,
        group=group,
        bus=_BroadcastBus(),
        initialized=False,
    )

    with patch("importlib.import_module", return_value=backend):
        with pytest.raises(
            module.WeightStoreDistributedError,
            match="initialized",
        ) as raised:
            module.TorchDistributedWeightStoreCoordinator(group=group)

    assert raised.value.phase == "initialize"


@requires_distributed
def test_torch_coordinator_without_context_uses_bounded_collectives() -> None:
    module = _require_module()
    gather_phase = "default.gather"
    values = ("rank-0", {"rank": 1})
    payloads = tuple(
        _serialized_value(
            value,
            phase=gather_phase,
        )
        for value in values
    )
    root_phase = "default.root"
    root_result = {"root": True}
    root_envelope = module._RootCallEnvelope(
        version=module._ROOT_CALL_VERSION,
        phase=root_phase,
        succeeded=True,
        result=root_result,
        error=None,
        completion_unknown=False,
    )
    broadcast_payload = _serialized_value(
        root_envelope,
        phase=root_phase,
    )
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=payloads,
        broadcast_payload=broadcast_payload,
    )

    assert coordinator.all_gather_object(
        values[0],
        phase=gather_phase,
    ) == list(values)
    assert coordinator.run_root(root_phase, lambda: root_result) == root_result
    assert backend.gather_inputs == []
    assert backend.broadcast_calls == []
    assert backend.tensor_collective_calls
    assert all(call.async_op for call in backend.tensor_collective_calls)


@requires_distributed
def test_none_context_collective_launch_error_poisons() -> None:
    module = _require_module()
    phase = "default.failure"
    payloads = tuple(
        _serialized_value(
            f"rank-{rank}",
            phase=phase,
        )
        for rank in range(2)
    )
    coordinator, _, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=payloads,
        launch_errors={0: RuntimeError("launch failed")},
    )

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        coordinator.all_gather_object("rank-0", phase=phase)

    assert raised.value.phase == f"{phase}.size"
    assert raised.value.completion_unknown is True
    assert coordinator.poisoned is True


@requires_distributed
def test_bounded_all_gather_uses_size_and_payload_tensor_collectives() -> None:
    phase = "bounded.gather"
    values = (
        "a",
        {"rank": 1, "values": [1, 2, 3]},
        ["longer", "payload", {"rank": 2}],
    )
    payloads = tuple(
        _serialized_value(
            value,
            phase=phase,
        )
        for value in values
    )
    assert len({len(payload) for payload in payloads}) == len(payloads)
    coordinator, backend, torch = _make_tensor_coordinator(
        rank=1,
        world_size=len(values),
        gather_payloads=payloads,
    )

    gathered = coordinator.all_gather_object(
        values[1],
        phase=phase,
        execution_context=_execution_context(),
    )

    assert gathered == list(values)
    assert backend.tensor_collective_calls == [
        _TensorCollectiveCall(
            kind="all_gather",
            input_dtype=torch.int64,
            output_dtypes=(torch.int64,) * len(values),
            async_op=True,
        ),
        _TensorCollectiveCall(
            kind="all_gather",
            input_dtype=torch.uint8,
            output_dtypes=(torch.uint8,) * len(values),
            async_op=True,
        ),
        _TensorCollectiveCall(
            kind="all_gather",
            input_dtype=torch.int64,
            output_dtypes=(torch.int64,) * len(values),
            async_op=True,
        ),
    ]
    assert backend.gather_inputs == []
    assert backend.broadcast_calls == []


@requires_distributed
def test_bounded_root_broadcast_uses_size_and_payload_tensor_collectives() -> None:
    module = _require_module()
    phase = "bounded.root"
    root_result = {"revision": "r1", "placements": ["short", "longer-value"]}
    envelope = module._RootCallEnvelope(
        version=module._ROOT_CALL_VERSION,
        phase=phase,
        succeeded=True,
        result=root_result,
        error=None,
        completion_unknown=False,
    )
    payload = _serialized_value(
        envelope,
        phase=phase,
    )
    coordinator, backend, torch = _make_tensor_coordinator(
        rank=1,
        world_size=2,
        broadcast_payload=payload,
    )

    result = coordinator.run_root(
        phase,
        lambda: pytest.fail("non-root factory was called"),
        execution_context=_execution_context(),
    )

    assert result == root_result
    assert backend.tensor_collective_calls == [
        _TensorCollectiveCall(
            kind="broadcast",
            input_dtype=torch.int64,
            output_dtypes=(),
            async_op=True,
            group_src=0,
        ),
        _TensorCollectiveCall(
            kind="broadcast",
            input_dtype=torch.uint8,
            output_dtypes=(),
            async_op=True,
            group_src=0,
        ),
        _TensorCollectiveCall(
            kind="all_gather",
            input_dtype=torch.int64,
            output_dtypes=(torch.int64, torch.int64),
            async_op=True,
        ),
    ]
    assert backend.broadcast_calls == []
    assert backend.gather_inputs == []


@requires_distributed
def test_bounded_collectives_run_on_two_process_gloo(tmp_path: Path) -> None:
    torch = import_module("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed Gloo backend is unavailable")

    torch.multiprocessing.spawn(
        _run_gloo_bounded_collectives,
        args=(2, f"file://{tmp_path / 'gloo-store'}"),
        nprocs=2,
        join=True,
    )


@requires_distributed
def test_gloo_skewed_timeout_poisons_every_rank_and_blocks_commit(
    tmp_path: Path,
) -> None:
    torch = import_module("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed Gloo backend is unavailable")

    torch.multiprocessing.spawn(
        _run_gloo_skewed_timeout,
        args=(
            2,
            f"file://{tmp_path / 'gloo-timeout-store'}",
            str(tmp_path / "commit-marker"),
        ),
        nprocs=2,
        join=True,
    )


@requires_distributed
def test_gloo_rank_local_decode_failure_keeps_collective_sequence(
    tmp_path: Path,
) -> None:
    torch = import_module("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed Gloo backend is unavailable")

    torch.multiprocessing.spawn(
        _run_gloo_decode_consensus,
        args=(2, f"file://{tmp_path / 'gloo-decode-consensus'}"),
        nprocs=2,
        join=True,
    )


@requires_distributed
def test_gloo_chunked_collective_reaps_completed_timed_out_work(
    tmp_path: Path,
) -> None:
    torch = import_module("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed Gloo backend is unavailable")

    torch.multiprocessing.spawn(
        _run_gloo_chunked_timeout_reap,
        args=(2, f"file://{tmp_path / 'gloo-timeout-reap'}"),
        nprocs=2,
        join=True,
    )


@requires_distributed
def test_bounded_collective_waits_only_after_work_completes() -> None:
    phase = "bounded.poll"
    values = ("rank-0", "rank-1")
    payloads = tuple(
        _serialized_value(
            value,
            phase=phase,
        )
        for value in values
    )
    size_work = _FakeWork((False, False, True))
    payload_work = _FakeWork()
    coordinator, _, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=payloads,
        works=(size_work, payload_work),
    )

    with patch("sglang.srt.distributed.bounded_object_collectives.time.sleep") as sleep:
        assert coordinator.all_gather_object(
            values[0],
            phase=phase,
            execution_context=_execution_context(),
        ) == list(values)

    assert size_work.events == [
        "is_completed",
        "is_completed",
        "is_completed",
        "wait",
    ]
    assert size_work.wait_count == 1
    assert payload_work.events == ["is_completed", "wait"]
    assert payload_work.wait_count == 1
    assert sleep.call_count == 2


@requires_distributed
@pytest.mark.parametrize("interruption", ["deadline", "cancel"])
def test_bounded_collective_interrupted_before_launch_poisons(
    interruption: str,
) -> None:
    module = _require_module()
    phase = f"bounded.prelaunch.{interruption}"
    payload = _serialized_value(
        "value",
        phase=phase,
    )
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=(payload, payload),
    )
    cancel_signal = threading.Event()
    if interruption == "cancel":
        cancel_signal.set()
        context = module.WeightTransferExecutionContext(
            deadline_unix_sec=time.time() + 60,
            cancel_signal=cancel_signal,
        )
    else:
        context = module.WeightTransferExecutionContext(
            deadline_unix_sec=time.time() - 1,
        )

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        coordinator.all_gather_object(
            "value",
            phase=phase,
            execution_context=context,
        )

    assert raised.value.phase == phase
    assert raised.value.completion_unknown is True
    assert ("cancelled" if interruption == "cancel" else "deadline exceeded") in str(
        raised.value
    )
    assert coordinator.poisoned is True
    assert coordinator._pending_collectives == []
    assert backend.tensor_collective_calls == []


@requires_distributed
@pytest.mark.parametrize(
    ("failure", "detail"),
    [
        ("deadline", "deadline exceeded"),
        ("cancel", "cancelled"),
        ("launch", "RuntimeError: launch failed"),
        ("poll", "RuntimeError: poll failed"),
        ("wait", "RuntimeError: wait failed"),
    ],
)
def test_bounded_collective_unknown_completion_poisons_and_retains_objects(
    failure: str,
    detail: str,
) -> None:
    module = _require_module()
    phase = f"bounded.failure.{failure}"
    values = ("rank-0", "rank-1")
    payloads = tuple(
        _serialized_value(
            value,
            phase=phase,
        )
        for value in values
    )
    cancel_signal = threading.Event()
    size_work: _FakeWork | None
    launch_errors: dict[int, BaseException] = {}
    if failure == "cancel":
        size_work = _FakeWork((False,), on_poll=cancel_signal.set)
    elif failure == "launch":
        size_work = None
        launch_errors[0] = RuntimeError("launch failed")
    elif failure == "poll":
        size_work = _FakeWork(poll_error=RuntimeError("poll failed"))
    elif failure == "wait":
        size_work = _FakeWork(wait_error=RuntimeError("wait failed"))
    else:
        size_work = _FakeWork((False,))
    works = () if size_work is None else (size_work,)
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=payloads,
        works=works,
        launch_errors=launch_errors,
    )
    context = _execution_context(cancel_signal=cancel_signal)

    if failure == "deadline":
        with patch.object(
            type(context),
            "expired",
            side_effect=(False, False, True),
        ):
            with pytest.raises(module.WeightStoreDistributedError) as raised:
                coordinator.all_gather_object(
                    values[0],
                    phase=phase,
                    execution_context=context,
                )
    else:
        with pytest.raises(module.WeightStoreDistributedError) as raised:
            coordinator.all_gather_object(
                values[0],
                phase=phase,
                execution_context=context,
            )

    error = raised.value
    assert error.phase == f"{phase}.size"
    assert error.completion_unknown is True
    assert detail in str(error)
    assert "scheduler restart is required" in str(error)
    assert coordinator.poisoned is True
    assert len(coordinator._pending_collectives) == 1
    pending = coordinator._pending_collectives[0]
    assert pending.work is size_work
    assert len(pending.tensors) == 3
    tensor_refs = tuple(backend.tensor_refs)
    error.__traceback__ = None
    error.__cause__ = None
    error.__context__ = None
    del raised
    gc.collect()
    assert tensor_refs
    assert all(reference() is not None for reference in tensor_refs)
    assert tuple(reference() for reference in tensor_refs) == pending.tensors


@requires_distributed
def test_completed_pending_collectives_release_retained_objects() -> None:
    module = _require_module()
    payload = _serialized_value(
        "value",
        phase="bounded.reap",
    )
    coordinator, _, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=(payload, payload),
    )
    retained = object()
    work = _FakeWork((False, True))
    coordinator._pending_collectives.append(
        module._PendingCollective(work=work, tensors=(retained,))
    )

    assert coordinator.reap_completed_collectives() == 0
    assert len(coordinator._pending_collectives) == 1
    assert coordinator.reap_completed_collectives() == 1
    assert coordinator._pending_collectives == []
    assert work.wait_count == 1


@requires_distributed
def test_completed_work_with_wait_error_releases_retained_objects() -> None:
    module = _require_module()
    payload = _serialized_value(
        "value",
        phase="bounded.reap-failed",
    )
    coordinator, _, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=(payload, payload),
    )
    work = _FakeWork(
        (True,),
        wait_error=RuntimeError("collective failed"),
    )

    class Retained:
        pass

    retained = Retained()
    retained_ref = weakref.ref(retained)
    coordinator._pending_collectives.append(
        module._PendingCollective(work=work, tensors=(retained,))
    )
    del retained

    assert coordinator.reap_completed_collectives() == 1
    gc.collect()
    assert retained_ref() is None
    assert coordinator._pending_collectives == []
    assert work.wait_count == 1


@requires_distributed
def test_reaper_stops_after_terminal_work_failure() -> None:
    payload = _serialized_value(
        "value",
        phase="bounded.reaper-terminal",
    )
    coordinator, _, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=(payload, payload),
    )
    work = _FakeWork(
        (True,),
        wait_error=RuntimeError("terminal collective failure"),
    )

    class Retained:
        pass

    retained = Retained()
    retained_ref = weakref.ref(retained)
    coordinator._retain_pending_collective(work, (retained,))
    del retained

    deadline = time.monotonic() + 2
    while coordinator._reaper_thread is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    gc.collect()

    assert coordinator._reaper_thread is None
    assert coordinator._pending_collectives == []
    assert retained_ref() is None
    assert work.wait_count == 1


@requires_distributed
def test_unknown_work_handle_keeps_retained_objects_alive() -> None:
    module = _require_module()
    payload = _serialized_value(
        "value",
        phase="bounded.unknown-work",
    )
    coordinator, _, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=(payload, payload),
    )

    class Retained:
        pass

    retained = Retained()
    retained_ref = weakref.ref(retained)
    coordinator._pending_collectives.append(
        module._PendingCollective(work=None, tensors=(retained,))
    )
    del retained

    assert coordinator.reap_completed_collectives() == 0
    gc.collect()
    assert retained_ref() is not None
    assert len(coordinator._pending_collectives) == 1


@requires_distributed
def test_coordinator_rejects_unbounded_collective_member_count() -> None:
    with pytest.raises(ValueError, match="collective member limit"):
        _make_tensor_coordinator(
            rank=0,
            world_size=3,
            max_collective_members=2,
        )


@requires_distributed
def test_payload_launch_failure_retains_backing_storage() -> None:
    module = _require_module()
    phase = "bounded.payload-launch"
    values = ("rank-0", {"rank": 1})
    payloads = tuple(
        _serialized_value(
            value,
            phase=phase,
        )
        for value in values
    )
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=payloads,
        works=(_FakeWork(),),
        launch_errors={1: RuntimeError("payload launch failed")},
    )

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        coordinator.all_gather_object(
            values[0],
            phase=phase,
            execution_context=_execution_context(),
        )

    assert raised.value.phase == f"{phase}.payload"
    assert raised.value.completion_unknown is True
    assert coordinator.poisoned is True
    assert len(coordinator._pending_collectives) == 1
    pending = coordinator._pending_collectives[0]
    assert pending.work is None
    assert any(isinstance(value, bytearray) for value in pending.tensors)
    assert all(reference() is not None for reference in backend.tensor_refs)
    assert (
        tuple(reference() for reference in backend.tensor_refs) == pending.tensors[1:]
    )


@requires_distributed
def test_broadcast_payload_launch_failure_retains_backing_storage() -> None:
    module = _require_module()
    phase = "bounded.broadcast-launch"
    envelope = module._RootCallEnvelope(
        version=module._ROOT_CALL_VERSION,
        phase=phase,
        succeeded=True,
        result={"root": True},
        error=None,
        completion_unknown=False,
    )
    payload = _serialized_value(
        envelope,
        phase=phase,
    )
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=1,
        world_size=2,
        broadcast_payload=payload,
        works=(_FakeWork(),),
        launch_errors={1: RuntimeError("broadcast payload launch failed")},
    )

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        coordinator.run_root(
            phase,
            lambda: pytest.fail("non-root factory was called"),
            execution_context=_execution_context(),
        )

    assert raised.value.phase == f"{phase}.payload"
    assert raised.value.completion_unknown is True
    assert coordinator.poisoned is True
    pending = coordinator._pending_collectives[0]
    assert pending.work is None
    assert any(isinstance(value, bytearray) for value in pending.tensors)
    assert all(reference() is not None for reference in backend.tensor_refs)
    assert (
        tuple(reference() for reference in backend.tensor_refs) == pending.tensors[1:]
    )


@requires_distributed
@pytest.mark.parametrize(
    ("failure", "expected_message", "expected_call_count"),
    [
        (
            "size",
            "serialized collective payload exceeds the configured size limit",
            1,
        ),
        (
            "version",
            "serialized collective envelope does not match the operation",
            3,
        ),
        (
            "decode",
            "invalid serialized collective payload",
            3,
        ),
    ],
)
def test_bounded_collective_wire_failures_do_not_poison(
    failure: str,
    expected_message: str,
    expected_call_count: int,
) -> None:
    module = _require_module()
    phase = f"bounded.wire.{failure}"
    local_payload = _serialized_value(
        "local",
        phase=phase,
    )
    max_object_bytes = 64 * 1024 * 1024
    if failure == "size":
        payloads = (local_payload, b"x" * (len(local_payload) + 1))
        max_object_bytes = 32
    elif failure == "version":
        payload = pickle.dumps(
            module._SerializedCollectiveValue(
                version=module._COLLECTIVE_VALUE_VERSION + 1,
                phase=phase,
                succeeded=True,
                value="value",
                error=None,
            ),
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        payloads = (local_payload, payload)
    else:
        payloads = (local_payload, b"not-a-pickle")
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=payloads,
        max_object_bytes=max_object_bytes,
    )

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        coordinator.all_gather_object(
            "local",
            phase=phase,
            execution_context=_execution_context(),
        )

    assert raised.value.phase == phase
    assert raised.value.completion_unknown is False
    assert expected_message in str(raised.value)
    assert coordinator.poisoned is False
    assert coordinator._pending_collectives == []
    assert len(backend.tensor_collective_calls) == expected_call_count


@requires_distributed
def test_rank_local_decode_failure_reaches_terminal_consensus() -> None:
    module = _require_module()
    phase = "bounded.decode-consensus"
    values = ("rank-0", "rank-1")
    payloads = tuple(
        _serialized_value(
            value,
            phase=phase,
        )
        for value in values
    )
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=payloads,
        decode_fail_ranks=(1,),
    )

    with pytest.raises(
        module.WeightStoreDistributedError,
        match=r"decoding failed on ranks \(1,\)",
    ) as raised:
        coordinator.all_gather_object(
            values[0],
            phase=phase,
            execution_context=_execution_context(),
        )

    assert raised.value.completion_unknown is False
    assert coordinator.poisoned is False
    assert len(backend.tensor_collective_calls) == 3


@requires_distributed
def test_all_gather_rejects_resident_buffers_before_payload_allocation() -> None:
    module = _require_module()
    phase = "bounded.resident-limit"
    values = ("x" * 64, "y" * 64)
    payloads = tuple(
        _serialized_value(
            value,
            phase=phase,
        )
        for value in values
    )
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=payloads,
        max_object_bytes=max(map(len, payloads)) + 1,
        max_aggregate_bytes=sum(map(len, payloads)) + 1,
        max_resident_bytes=128,
        chunk_bytes=64,
    )

    with pytest.raises(
        module.WeightStoreDistributedError,
        match="resident buffers exceed",
    ) as raised:
        coordinator.all_gather_object(
            values[0],
            phase=phase,
            execution_context=_execution_context(),
        )

    assert raised.value.completion_unknown is False
    assert len(backend.tensor_collective_calls) == 1


@requires_distributed
def test_none_context_all_gather_enforces_aggregate_limit() -> None:
    module = _require_module()
    phase = "default.aggregate-limit"
    payloads = tuple(
        _serialized_value(
            f"rank-{rank}",
            phase=phase,
        )
        for rank in range(2)
    )
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=payloads,
        max_aggregate_bytes=sum(map(len, payloads)) - 1,
    )

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        coordinator.all_gather_object("rank-0", phase=phase)

    assert raised.value.phase == phase
    assert raised.value.completion_unknown is False
    assert "aggregate exceeds" in str(raised.value)
    assert coordinator.poisoned is False
    assert len(backend.tensor_collective_calls) == 1
    assert backend.tensor_collective_calls[0].async_op is True


@requires_distributed
def test_poisoned_coordinator_fails_fast_without_touching_backend() -> None:
    module = _require_module()
    phase = "bounded.poison"
    payload = _serialized_value(
        "value",
        phase=phase,
    )
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=(payload, payload),
        launch_errors={0: RuntimeError("launch failed")},
    )
    context = _execution_context()

    with pytest.raises(module.WeightStoreDistributedError) as first:
        coordinator.all_gather_object(
            "value",
            phase=phase,
            execution_context=context,
        )

    backend_calls = tuple(backend.tensor_collective_calls)
    backend.launch_errors.clear()
    with pytest.raises(module.WeightStoreDistributedError) as second:
        coordinator.all_gather_object(
            "value",
            phase="bounded.after-poison",
            execution_context=context,
        )

    assert first.value.completion_unknown is True
    assert second.value.phase == "bounded.after-poison"
    assert second.value.completion_unknown is True
    assert str(second.value) == str(first.value)
    assert tuple(backend.tensor_collective_calls) == backend_calls


@requires_distributed
def test_poisoned_fast_path_does_not_poll_pending_work() -> None:
    module = _require_module()
    payload = _serialized_value(
        "value",
        phase="bounded.poisoned-pending",
    )
    coordinator, backend, _ = _make_tensor_coordinator(
        rank=0,
        world_size=2,
        gather_payloads=(payload, payload),
    )
    work = _FakeWork(on_poll=lambda: pytest.fail("pending Work was polled"))
    coordinator._pending_collectives.append(
        module._PendingCollective(work=work, tensors=(object(),))
    )
    coordinator._poisoned = "injected poison"

    with pytest.raises(module.WeightStoreDistributedError, match="injected poison"):
        coordinator.all_gather_object(
            "value",
            phase="bounded.poisoned-fast-path",
            execution_context=_execution_context(),
        )

    assert work.events == []
    assert backend.tensor_collective_calls == []


@requires_distributed
def test_torch_coordinator_broadcasts_prepare_and_commit_results_from_root() -> None:
    plan = {"kind": "upload-plan"}
    manifest = {"kind": "committed-manifest"}
    coordinators, backends, _, _ = _make_torch_coordinators(world_size=3)
    prepare_calls: list[int] = []
    commit_calls: list[int] = []

    prepare_results = [
        coordinator.prepare_upload(_factory_for_rank(rank, prepare_calls, plan))
        for rank, coordinator in enumerate(coordinators)
    ]
    commit_results = [
        coordinator.commit_upload(_factory_for_rank(rank, commit_calls, manifest))
        for rank, coordinator in enumerate(coordinators)
    ]

    assert prepare_results == [plan, plan, plan]
    assert commit_results == [manifest, manifest, manifest]
    assert prepare_calls == [0]
    assert commit_calls == [0]
    assert [coordinator.rank for coordinator in coordinators] == [0, 1, 2]
    assert all(coordinator.world_size == 3 for coordinator in coordinators)
    assert all(backend.broadcast_calls == [] for backend in backends)
    assert all(
        len(backend.tensor_collective_calls) == 6
        and all(call.async_op for call in backend.tensor_collective_calls)
        for backend in backends
    )


@requires_distributed
def test_torch_coordinator_run_root_broadcasts_custom_result_once() -> None:
    result = {"kind": "catalog-result"}
    coordinators, backends, bus, _ = _make_torch_coordinators(world_size=3)
    calls: list[int] = []

    results = [
        coordinator.run_root(
            "catalog.custom",
            _factory_for_rank(rank, calls, result),
        )
        for rank, coordinator in enumerate(coordinators)
    ]

    assert results == [result, result, result]
    assert calls == [0]
    assert [envelope.phase for envelope in _broadcast_bus_envelopes(bus)] == [
        "catalog.custom"
    ]
    assert all(backend.broadcast_calls == [] for backend in backends)
    assert all(
        len(backend.tensor_collective_calls) == 3
        and all(call.async_op for call in backend.tensor_collective_calls)
        for backend in backends
    )


@requires_distributed
def test_root_factory_deadline_broadcasts_unknown_completion_to_every_rank() -> None:
    module = _require_module()
    coordinators, _, bus, _ = _make_torch_coordinators(world_size=3)
    factory_entered = threading.Event()
    release_factory = threading.Event()
    calls: list[int] = []

    def factory_for_rank(rank: int):
        def factory() -> None:
            calls.append(rank)
            if rank != 0:
                raise AssertionError("non-root factory was called")
            factory_entered.set()
            release_factory.wait(timeout=5)

        return factory

    errors = []
    root_context = module.WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 0.05,
    )
    with pytest.raises(module.WeightStoreDistributedError) as raised:
        coordinators[0].run_root(
            "catalog.deadline",
            factory_for_rank(0),
            execution_context=root_context,
        )
    errors.append(raised.value)
    assert factory_entered.is_set()

    envelopes = _broadcast_bus_envelopes(bus)
    assert len(envelopes) == 1
    assert envelopes[0].completion_unknown is True

    for rank in range(1, len(coordinators)):
        with pytest.raises(module.WeightStoreDistributedError) as raised:
            coordinators[rank].run_root(
                "catalog.deadline",
                factory_for_rank(rank),
                execution_context=module.WeightTransferExecutionContext(
                    deadline_unix_sec=time.time() + 1,
                ),
            )
        errors.append(raised.value)

    assert calls == [0]
    assert all(error.completion_unknown for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert all(coordinator.poisoned for coordinator in coordinators)

    assert len(coordinators[0]._root_factory_calls) == 1
    release_factory.set()
    deadline = time.monotonic() + 1
    while (
        coordinators[0]._root_factory_calls
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert coordinators[0]._root_factory_calls == set()


@requires_distributed
def test_root_catalog_runs_every_catalog_method_on_root_and_broadcasts() -> None:
    module = _require_module()
    coordinators, _, bus, _ = _make_torch_coordinators(world_size=2)
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class RecordingCatalog:
        def __getattr__(self, name: str):
            def method(*args: Any, **kwargs: Any) -> Any:
                calls.append((name, args, kwargs))
                return {
                    "method": name,
                    "args": args,
                    "kwargs": kwargs,
                }

            return method

    root_catalog = module.RootWeightStorageCatalog(
        RecordingCatalog(),
        coordinators[0],
    )
    non_root_catalog = module.RootWeightStorageCatalog(None, coordinators[1])
    invocations = [
        ("begin_materialization", ("materialization", "intent"), {}),
        (
            "complete_materialization",
            ("materialization", "snapshot"),
            {},
        ),
        ("abort_materialization", ("materialization",), {}),
        (
            "set_materialization_completion_ticket",
            ("materialization", "ticket"),
            {},
        ),
        ("get_materialization", ("materialization",), {}),
        ("recoverable_materializations", (), {}),
        ("prepare_publish", ("publication", "snapshot"), {}),
        ("publish", ("publication",), {}),
        ("abort", ("publication",), {}),
        ("get_snapshot", ("ref",), {}),
        ("get_publication", ("publication",), {}),
        ("recoverable_publications", (), {}),
        ("get_revision_head", ("model", "revision"), {}),
        (
            "compare_and_set_revision",
            (),
            {
                "model_id": "model",
                "revision": "revision",
                "expected": "old-head",
                "new_ref": "new-ref",
                "new_state": "ready",
            },
        ),
    ]

    for method_name, args, kwargs in invocations:
        root_result = getattr(root_catalog, method_name)(*args, **kwargs)
        non_root_result = getattr(non_root_catalog, method_name)(*args, **kwargs)
        assert non_root_result == root_result

    expected_names = [method_name for method_name, _, _ in invocations]
    assert [name for name, _, _ in calls] == expected_names
    phases = [envelope.phase for envelope in _broadcast_bus_envelopes(bus)]
    assert phases == [f"catalog.{name}" for name in expected_names]
    assert len(phases) == len(set(phases))


@requires_distributed
def test_root_catalog_requires_catalog_only_on_root() -> None:
    module = _require_module()
    coordinators, _, _, _ = _make_torch_coordinators(world_size=2)

    with pytest.raises(ValueError, match="root catalog"):
        module.RootWeightStorageCatalog(None, coordinators[0])

    module.RootWeightStorageCatalog(None, coordinators[1])


@requires_distributed
def test_torch_coordinator_runs_abort_and_finalize_only_on_root() -> None:
    coordinators, backends, _, _ = _make_torch_coordinators(world_size=3)

    for method_name in ("abort_upload", "finalize_upload"):
        calls: list[int] = []
        results = [
            getattr(coordinator, method_name)(_factory_for_rank(rank, calls, object()))
            for rank, coordinator in enumerate(coordinators)
        ]
        assert results == [None, None, None]
        assert calls == [0]

    assert all(backend.broadcast_calls == [] for backend in backends)
    assert all(
        len(backend.tensor_collective_calls) == 6
        and all(call.async_op for call in backend.tensor_collective_calls)
        for backend in backends
    )


@requires_distributed
def test_torch_coordinator_returns_complete_outcomes_sorted_by_rank() -> None:
    module = _require_module()
    phase = "exchange_upload_outcome"
    chunk_bytes = 128
    outcomes = tuple(
        module.WeightStoreUploadOutcome(
            rank=rank,
            placement_ids=(f"placement-{rank}",),
            receipts=("x" * 4096 if rank == 1 else f"receipt-{rank}",),
            error=None,
        )
        for rank in range(3)
    )
    gathered = (outcomes[2], outcomes[0], outcomes[1])
    payloads = tuple(
        _serialized_value(
            outcome,
            phase=phase,
        )
        for outcome in gathered
    )
    terminal = module._RootCallEnvelope(
        version=module._ROOT_CALL_VERSION,
        phase=phase,
        succeeded=True,
        result=None,
        error=None,
        completion_unknown=False,
    )
    broadcast_values = _root_gather_broadcast_values(
        module=module,
        phase=phase,
        gather_payloads=payloads,
        terminal=terminal,
        chunk_bytes=chunk_bytes,
    )

    results = []
    backends = []
    for rank in range(len(outcomes)):
        coordinator, backend, _ = _make_root_gather_coordinator(
            rank=rank,
            world_size=len(outcomes),
            gather_payloads=payloads,
            broadcast_values=(() if rank == 0 else broadcast_values),
            max_object_bytes=max(map(len, payloads)) + 1,
            max_aggregate_bytes=max(
                sum(map(len, payloads)),
                sum(
                    len(value) if isinstance(value, bytes) else 0
                    for value in broadcast_values
                ),
            )
            + 1024,
            chunk_bytes=chunk_bytes,
        )
        results.append(coordinator.exchange_upload_outcome(outcomes[rank]))
        backends.append(backend)

    assert results == [outcomes, None, None]
    assert all(backend.gather_calls for backend in backends)
    assert all(call.async_op for backend in backends for call in backend.gather_calls)
    assert all(backend.gather_inputs == [] for backend in backends)
    compact_payload = _serialized_value(
        terminal,
        phase=phase,
    )
    expected_terminal_chunks = (len(compact_payload) + chunk_bytes - 1) // chunk_bytes
    assert len(backends[0].tensor_broadcast_numels) == 2 + expected_terminal_chunks


@requires_distributed
def test_torch_coordinator_returns_complete_preflight_outcomes_sorted_by_rank() -> None:
    module = _require_module()
    phase = "exchange_preflight_outcome"
    chunk_bytes = 128
    outcomes = tuple(
        module.WeightStorePreflightOutcome(
            rank=rank,
            error=None if rank != 1 else "ValueError: invalid local plan",
        )
        for rank in range(3)
    )
    gathered = (outcomes[2], outcomes[0], outcomes[1])
    payloads = tuple(
        _serialized_value(
            outcome,
            phase=phase,
        )
        for outcome in gathered
    )
    terminal = module._RootCallEnvelope(
        version=module._ROOT_CALL_VERSION,
        phase=phase,
        succeeded=True,
        result=outcomes,
        error=None,
        completion_unknown=False,
    )
    broadcast_values = _root_gather_broadcast_values(
        module=module,
        phase=phase,
        gather_payloads=payloads,
        terminal=terminal,
        chunk_bytes=chunk_bytes,
    )

    results = []
    backends = []
    for rank in range(len(outcomes)):
        coordinator, backend, _ = _make_root_gather_coordinator(
            rank=rank,
            world_size=len(outcomes),
            gather_payloads=payloads,
            broadcast_values=(() if rank == 0 else broadcast_values),
            max_object_bytes=max(map(len, payloads)) + 1,
            max_aggregate_bytes=sum(map(len, payloads)) + 4096,
            chunk_bytes=chunk_bytes,
        )
        results.append(coordinator.exchange_preflight_outcome(outcomes[rank]))
        backends.append(backend)

    assert results == [outcomes, outcomes, outcomes]
    assert all(backend.gather_calls for backend in backends)
    assert all(backend.gather_inputs == [] for backend in backends)


@requires_distributed
def test_outcome_exchange_root_receives_exact_skewed_payloads() -> None:
    module = _require_module()
    phase = "exchange_upload_outcome"
    chunk_bytes = 32
    outcomes = tuple(
        module.WeightStoreUploadOutcome(
            rank=rank,
            placement_ids=(f"placement-{rank}",),
            receipts=(
                {
                    "generation": 100 + rank,
                    "manifest_digest": f"sha256:manifest-{rank}",
                    "checksum": f"sha256:payload-{rank}",
                    "payload": "x" * (rank * 73),
                },
            ),
            error=None,
        )
        for rank in range(7)
    )
    wire_order = (outcomes[6], outcomes[0], *outcomes[1:6])
    payloads = tuple(
        _serialized_value(
            outcome,
            phase=phase,
        )
        for outcome in wire_order
    )
    coordinator, backend, _ = _make_root_gather_coordinator(
        rank=0,
        world_size=len(outcomes),
        gather_payloads=payloads,
        max_object_bytes=max(map(len, payloads)) + 1,
        max_aggregate_bytes=sum(map(len, payloads)) + 1024,
        chunk_bytes=chunk_bytes,
    )

    result = coordinator.exchange_upload_outcome(
        outcomes[0],
        execution_context=_execution_context(),
    )

    assert result == outcomes
    payload_calls = backend.gather_calls[1:]
    assert payload_calls
    assert all(
        call.output_numels == (chunk_bytes,) * len(outcomes) for call in payload_calls
    )


@requires_distributed
def test_outcome_exchange_non_root_receives_only_compact_terminal() -> None:
    module = _require_module()
    phase = "exchange_upload_outcome"
    world_size = 129
    rank = world_size - 1
    chunk_bytes = 64
    outcomes = tuple(
        module.WeightStoreUploadOutcome(
            rank=outcome_rank,
            placement_ids=(f"placement-{outcome_rank}",),
            receipts=(
                (
                    "remote-large-" + "x" * (chunk_bytes * 131)
                    if outcome_rank == 1
                    else f"receipt-{outcome_rank}"
                ),
            ),
            error=None,
        )
        for outcome_rank in range(world_size)
    )
    payloads = tuple(
        _serialized_value(
            outcome,
            phase=phase,
        )
        for outcome in outcomes
    )
    max_rounds = (max(map(len, payloads)) + chunk_bytes - 1) // chunk_bytes
    terminal = module._RootCallEnvelope(
        version=module._ROOT_CALL_VERSION,
        phase=phase,
        succeeded=True,
        result=None,
        error=None,
        completion_unknown=False,
    )
    terminal_payload = _serialized_value(
        terminal,
        phase=phase,
    )
    terminal_chunks = tuple(
        terminal_payload[offset : offset + chunk_bytes]
        for offset in range(0, len(terminal_payload), chunk_bytes)
    )
    broadcast_values: tuple[tuple[int, ...] | bytes, ...] = (
        (1, 0, max_rounds, sum(map(len, payloads))),
        (len(terminal_payload),),
        *terminal_chunks,
    )
    coordinator, backend, _ = _make_root_gather_coordinator(
        rank=rank,
        world_size=world_size,
        gather_payloads=payloads,
        broadcast_values=broadcast_values,
        max_object_bytes=max(map(len, payloads)) + 1,
        max_aggregate_bytes=max(
            sum(map(len, payloads)),
            len(terminal_payload),
        )
        + 1024,
        chunk_bytes=chunk_bytes,
    )

    result = coordinator.exchange_upload_outcome(
        outcomes[rank],
        execution_context=_execution_context(),
    )

    assert result is None
    assert backend.gather_calls
    assert all(call.output_numels == () for call in backend.gather_calls)
    assert max(call.input_numel for call in backend.gather_calls) <= chunk_bytes
    assert max(backend.tensor_broadcast_numels) <= chunk_bytes
    assert len(terminal_payload) < chunk_bytes * 8
    assert world_size > chunk_bytes
    assert max(map(len, payloads)) > chunk_bytes * 100


@requires_distributed
@pytest.mark.parametrize(
    ("limit_kind", "expected_message"),
    [
        ("message", "payload exceeds the configured size limit"),
        ("aggregate", "aggregate exceeds the configured size limit"),
    ],
)
def test_outcome_exchange_rejects_limits_before_payload_gather(
    limit_kind: str,
    expected_message: str,
) -> None:
    module = _require_module()
    phase = "exchange_upload_outcome"
    outcomes = tuple(
        module.WeightStoreUploadOutcome(
            rank=rank,
            placement_ids=(f"placement-{rank}",),
            receipts=("x" * (rank + 1),),
            error=None,
        )
        for rank in range(3)
    )
    payloads = tuple(
        _serialized_value(
            outcome,
            phase=phase,
        )
        for outcome in outcomes
    )
    max_object_bytes = max(map(len, payloads)) + 1
    max_aggregate_bytes = sum(map(len, payloads)) + 1
    if limit_kind == "message":
        max_object_bytes = min(map(len, payloads)) - 1
    else:
        max_aggregate_bytes = sum(map(len, payloads)) - 1
    coordinator, backend, _ = _make_root_gather_coordinator(
        rank=0,
        world_size=len(outcomes),
        gather_payloads=payloads,
        max_object_bytes=max_object_bytes,
        max_aggregate_bytes=max_aggregate_bytes,
        chunk_bytes=32,
    )

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        coordinator.exchange_upload_outcome(
            outcomes[0],
            execution_context=_execution_context(),
        )

    assert raised.value.phase == phase
    assert expected_message in str(raised.value)
    assert raised.value.completion_unknown is False
    assert len(backend.gather_calls) == 1
    assert backend.gather_calls[0].input_numel == 1
    assert backend.tensor_broadcast_numels == [4]
    assert coordinator.poisoned is False


@requires_distributed
@pytest.mark.parametrize(
    ("failure", "expected_phase"),
    [
        ("gather", "exchange_upload_outcome.size"),
        ("broadcast", "exchange_upload_outcome.admission"),
    ],
)
def test_outcome_exchange_launch_failure_poisons_and_retains_tensors(
    failure: str,
    expected_phase: str,
) -> None:
    module = _require_module()
    phase = "exchange_upload_outcome"
    outcomes = tuple(
        module.WeightStoreUploadOutcome(
            rank=rank,
            placement_ids=(f"placement-{rank}",),
            receipts=(f"receipt-{rank}",),
            error=None,
        )
        for rank in range(2)
    )
    payloads = tuple(
        _serialized_value(
            outcome,
            phase=phase,
        )
        for outcome in outcomes
    )
    coordinator, backend, _ = _make_root_gather_coordinator(
        rank=0,
        world_size=len(outcomes),
        gather_payloads=payloads,
        max_object_bytes=max(map(len, payloads)) + 1,
        max_aggregate_bytes=sum(map(len, payloads)) + 1024,
        chunk_bytes=32,
    )
    failed_method = "gather" if failure == "gather" else "broadcast"

    with patch.object(
        backend,
        failed_method,
        side_effect=RuntimeError(f"{failure} launch failed"),
    ):
        with pytest.raises(module.WeightStoreDistributedError) as raised:
            coordinator.exchange_upload_outcome(
                outcomes[0],
                execution_context=_execution_context(),
            )

    assert raised.value.phase == expected_phase
    assert raised.value.completion_unknown is True
    assert f"{failure} launch failed" in str(raised.value)
    assert "scheduler restart is required" in str(raised.value)
    assert coordinator.poisoned is True
    assert len(coordinator._pending_collectives) == 1
    pending = coordinator._pending_collectives[0]
    assert pending.work is None
    assert pending.tensors


@requires_distributed
@pytest.mark.parametrize(
    "case",
    ["unknown", "duplicate-rank", "duplicate-placement"],
)
def test_torch_coordinator_fails_closed_on_invalid_gathered_outcomes(
    case: str,
) -> None:
    module = _require_module()
    phase = "exchange_upload_outcome"
    chunk_bytes = 128
    first = module.WeightStoreUploadOutcome(
        rank=0,
        placement_ids=("placement-0",),
        receipts=(),
        error=None,
    )
    if case == "unknown":
        gathered = [first, object()]
    elif case == "duplicate-rank":
        gathered = [first, first]
    else:
        gathered = [
            first,
            module.WeightStoreUploadOutcome(
                rank=1,
                placement_ids=("placement-0",),
                receipts=(),
                error=None,
            ),
        ]
    payloads = tuple(
        _serialized_value(
            outcome,
            phase=phase,
        )
        for outcome in gathered
    )
    with pytest.raises(module.WeightStoreDistributedError) as validation:
        module._validate_outcomes(gathered, world_size=2)
    terminal = module._RootCallEnvelope(
        version=module._ROOT_CALL_VERSION,
        phase=phase,
        succeeded=False,
        result=None,
        error=str(validation.value),
        completion_unknown=False,
    )
    broadcast_values = _root_gather_broadcast_values(
        module=module,
        phase=phase,
        gather_payloads=payloads,
        terminal=terminal,
        chunk_bytes=chunk_bytes,
    )

    errors = []
    for rank in range(2):
        coordinator, _, _ = _make_root_gather_coordinator(
            rank=rank,
            world_size=2,
            gather_payloads=payloads,
            broadcast_values=(() if rank == 0 else broadcast_values),
            max_object_bytes=max(map(len, payloads)) + 1,
            max_aggregate_bytes=sum(map(len, payloads)) + 4096,
            chunk_bytes=chunk_bytes,
        )
        local = first
        if rank == 1:
            local = module.WeightStoreUploadOutcome(
                rank=1,
                placement_ids=("local-1",),
                receipts=(),
                error=None,
            )
        with pytest.raises(module.WeightStoreDistributedError) as raised:
            coordinator.exchange_upload_outcome(local)
        errors.append(raised.value)

    assert all(error.phase == phase for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert "invalid gathered" in str(errors[0])


@requires_distributed
@pytest.mark.parametrize(
    "method_name",
    [
        "prepare_upload",
        "commit_upload",
        "abort_upload",
        "finalize_upload",
    ],
)
def test_root_factory_error_is_broadcast_consistently_to_every_rank(
    method_name: str,
) -> None:
    module = _require_module()
    coordinators, _, _, _ = _make_torch_coordinators(world_size=3)
    calls: list[int] = []

    def factory_for_rank(rank: int):
        def factory() -> None:
            calls.append(rank)
            if rank != 0:
                raise AssertionError("non-root factory was called")
            raise ValueError("root failed")

        return factory

    errors = []
    for rank, coordinator in enumerate(coordinators):
        with pytest.raises(module.WeightStoreDistributedError) as raised:
            getattr(coordinator, method_name)(factory_for_rank(rank))
        errors.append(raised.value)

    assert calls == [0]
    assert all(error.phase == method_name for error in errors)
    assert len({str(error) for error in errors}) == 1
    assert "root failed" in str(errors[0])


@requires_distributed
def test_torch_coordinator_fails_closed_on_unknown_broadcast_structure() -> None:
    module = _require_module()
    torch = import_module("torch")
    group = object()
    phase = "prepare_upload"
    payload = _serialized_value(
        object(),
        phase=phase,
    )
    bus = _BroadcastBus(
        payloads=[
            torch.tensor([len(payload)], dtype=torch.int64),
            torch.tensor(list(payload), dtype=torch.uint8),
        ]
    )
    backend = _FakeDistributed(
        rank=1,
        world_size=2,
        group=group,
        bus=bus,
    )
    with patch("importlib.import_module", return_value=backend):
        coordinator = module.TorchDistributedWeightStoreCoordinator(group=group)

    def non_root_factory() -> None:
        raise AssertionError("non-root factory was called")

    with pytest.raises(module.WeightStoreDistributedError) as raised:
        coordinator.prepare_upload(non_root_factory)

    assert raised.value.phase == phase
    assert "invalid" in str(raised.value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
