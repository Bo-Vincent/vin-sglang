from __future__ import annotations

import hashlib
import time
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightPlacementManifest,
    WeightRuntimeBindingManifest,
    WeightRuntimeManifestParts,
)
from sglang.srt.weight_transfer.api import (
    materialize_weight_snapshot,
    materialize_weights,
)
from sglang.srt.weight_transfer.binding import (
    bind_weight_source,
    project_source_bindings,
)
from sglang.srt.weight_transfer.contracts import RuntimeWeightLocation
from sglang.srt.weight_transfer.provider import (
    WeightMaterializeReceipt,
    WeightPayloadIdentity,
    WeightStorageDestination,
    WeightTransferAttestor,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
    WeightTransferExecutionContext,
    WeightTransferProvider,
    WeightTransferTerminalProof,
)
from sglang.srt.weight_transfer.storage import (
    WeightSnapshotPublication,
    WeightStorageCatalog,
)


@dataclass(frozen=True)
class _RuntimeParameterBytes:
    address: int
    nbytes: int
    device: str
    value: torch.Tensor


class RuntimeWeightPayloadHasher:
    """Hash runtime parameter ranges with bounded host staging."""

    def __init__(
        self,
        model: Any,
        *,
        chunk_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        named_parameters = getattr(model, "named_parameters", None)
        if not callable(named_parameters):
            raise ValueError("runtime model does not expose named parameters")

        ranges: dict[tuple[int, int, str], _RuntimeParameterBytes] = {}
        for _, parameter in named_parameters(remove_duplicate=False):
            if not isinstance(parameter, torch.Tensor):
                raise ValueError("runtime parameter is not a tensor")
            if not parameter.is_contiguous():
                raise ValueError("runtime parameter must be contiguous")
            address = int(parameter.data_ptr())
            nbytes = int(parameter.numel()) * int(parameter.element_size())
            device = str(parameter.device.type)
            if address <= 0 or nbytes <= 0:
                raise ValueError("runtime parameter has no transferable storage")
            ranges.setdefault(
                (address, nbytes, device),
                _RuntimeParameterBytes(
                    address=address,
                    nbytes=nbytes,
                    device=device,
                    value=parameter.detach(),
                ),
            )
        if not ranges:
            raise ValueError("runtime model has no transferable parameters")
        self._ranges = tuple(
            sorted(
                ranges.values(),
                key=lambda item: (item.device, item.address, item.nbytes),
            )
        )
        self._exact_ranges = {
            (item.device, item.address, item.nbytes): item for item in self._ranges
        }
        ranges_by_device: dict[str, list[_RuntimeParameterBytes]] = {}
        for item in self._ranges:
            ranges_by_device.setdefault(item.device, []).append(item)
        self._range_index = {}
        for device, device_ranges in ranges_by_device.items():
            starts = tuple(item.address for item in device_ranges)
            max_end = 0
            max_index = 0
            prefix_max_indices = []
            for index, item in enumerate(device_ranges):
                end = item.address + item.nbytes
                if end > max_end:
                    max_end = end
                    max_index = index
                prefix_max_indices.append(max_index)
            self._range_index[device] = (
                tuple(device_ranges),
                starts,
                tuple(prefix_max_indices),
            )
        self.chunk_bytes = chunk_bytes

    def _resolve(self, location: RuntimeWeightLocation) -> tuple[Any, int]:
        if not isinstance(location, RuntimeWeightLocation):
            raise ValueError("runtime payload location is invalid")
        exact = self._exact_ranges.get(
            (location.device, location.address, location.nbytes)
        )
        if exact is not None:
            return exact, 0
        index = self._range_index.get(location.device)
        if index is None:
            raise ValueError("runtime payload range is not owned by the runtime model")
        ranges, starts, prefix_max_indices = index
        end = location.address + location.nbytes
        position = bisect_right(starts, location.address) - 1
        if position >= 0:
            current = ranges[prefix_max_indices[position]]
            if end <= current.address + current.nbytes:
                return current, location.address - current.address
        raise ValueError("runtime payload range is not owned by the runtime model")

    def __call__(
        self,
        location: RuntimeWeightLocation,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> str:
        current, begin = self._resolve(location)
        byte_view = current.value.reshape(-1).view(torch.uint8)
        digest = hashlib.sha256()
        for offset in range(0, location.nbytes, self.chunk_bytes):
            _check_payload_hash_context(execution_context)
            nbytes = min(self.chunk_bytes, location.nbytes - offset)
            chunk = byte_view.narrow(0, begin + offset, nbytes)
            if chunk.device.type != "cpu":
                chunk = chunk.to(device="cpu", non_blocking=False)
            digest.update(chunk.contiguous().numpy().tobytes())
        return f"sha256:{digest.hexdigest()}"


def _check_payload_hash_context(
    execution_context: WeightTransferExecutionContext | None,
) -> None:
    if execution_context is None:
        return
    cancelled = execution_context.cancelled()
    if not cancelled and execution_context.remaining_seconds() > 0:
        return
    reason = "cancelled" if cancelled else "deadline exceeded"
    raise TimeoutError(f"runtime payload hashing {reason}")


_RUNTIME_WEIGHT_SNAPSHOT_QUARANTINE: list[RuntimeWeightSnapshotSource] = []


@dataclass
class RuntimeWeightSnapshotSource:
    """Own one runtime snapshot lease through Store materialization."""

    model: Any
    manager: Any
    parts: WeightRuntimeManifestParts
    payload_hasher: RuntimeWeightPayloadHasher
    payload_identity: WeightPayloadIdentity | None
    released: bool = False
    quarantined: bool = False
    operation_id: str | None = None
    provider_name: str | None = None
    completion_ticket: str | None = None
    hash_deadline_unix_sec: float | None = None

    @classmethod
    def capture(
        cls,
        *,
        model: Any,
        manager: Any,
        model_id: str,
        revision: str,
        instance_id: str,
        worker_id: str,
        endpoint: str,
        lease_timeout_sec: int | None = None,
        checksum_chunk_bytes: int = 64 * 1024 * 1024,
        execution_context: WeightTransferExecutionContext | None = None,
        defer_payload_identity: bool = False,
    ) -> RuntimeWeightSnapshotSource:
        if type(defer_payload_identity) is not bool:
            raise ValueError("defer_payload_identity must be a boolean")
        lease_deadline = (
            time.time() + lease_timeout_sec
            if type(lease_timeout_sec) is int and lease_timeout_sec > 0
            else None
        )
        parts = manager.snapshot_parts(
            model_id=model_id,
            revision=revision,
            instance_id=instance_id,
            worker_id=worker_id,
            endpoint=endpoint,
            lease_timeout_sec=lease_timeout_sec,
        )
        try:
            if not isinstance(parts, WeightRuntimeManifestParts):
                raise ValueError("runtime manifest manager returned invalid parts")
            hash_execution_context = execution_context
            if lease_deadline is not None:
                if (
                    hash_execution_context is None
                    or lease_deadline < hash_execution_context.deadline_unix_sec
                ):
                    hash_execution_context = WeightTransferExecutionContext(
                        deadline_unix_sec=lease_deadline,
                        cancel_signal=(
                            None
                            if execution_context is None
                            else execution_context.cancel_signal
                        ),
                    )
            hasher = RuntimeWeightPayloadHasher(
                model,
                chunk_bytes=checksum_chunk_bytes,
            )
            source = cls(
                model=model,
                manager=manager,
                parts=parts,
                payload_hasher=hasher,
                payload_identity=None,
                hash_deadline_unix_sec=(
                    None
                    if hash_execution_context is None
                    else hash_execution_context.deadline_unix_sec
                ),
            )
            if not defer_payload_identity:
                source.capture_payload_identity(
                    execution_context=hash_execution_context,
                )
            return source
        except BaseException:
            has_lease = getattr(manager, "has_lease", None)
            if not callable(has_lease) or has_lease(parts.binding.lease_id):
                manager.release(parts.binding.lease_id)
            raise

    @property
    def placement(self):
        return self.parts.placement

    @property
    def binding(self) -> WeightRuntimeBindingManifest:
        return self.parts.binding

    def capture_payload_identity(
        self,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> WeightPayloadIdentity:
        if self.released:
            raise RuntimeError("runtime weight snapshot source is released")
        if self.quarantined:
            raise RuntimeError("runtime weight snapshot source is quarantined")
        if self.payload_identity is not None:
            return self.payload_identity
        if self.hash_deadline_unix_sec is not None and (
            execution_context is None
            or self.hash_deadline_unix_sec < execution_context.deadline_unix_sec
        ):
            execution_context = WeightTransferExecutionContext(
                deadline_unix_sec=self.hash_deadline_unix_sec,
                cancel_signal=(
                    None
                    if execution_context is None
                    else execution_context.cancel_signal
                ),
            )
        locations = bind_weight_source(
            (self.parts.placement,),
            (self.parts.binding,),
        )
        identity = WeightPayloadIdentity.create(
            (self.parts.placement,),
            {
                location.placement_fragment_id: self.payload_hasher(
                    location,
                    execution_context=execution_context,
                )
                for location in locations
            },
        )
        self.payload_identity = identity
        return identity

    def payload_checksum(
        self,
        location: RuntimeWeightLocation,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> str:
        if self.released:
            raise RuntimeError("runtime weight snapshot source is released")
        if execution_context is None:
            return self.payload_hasher(location)
        return self.payload_hasher(
            location,
            execution_context=execution_context,
        )

    def attest_payload_identity(
        self,
        request: Any,
        *,
        execution_context: WeightTransferExecutionContext | None = None,
    ) -> None:
        _check_payload_hash_context(execution_context)
        source_fragments = {
            fragment.placement_fragment_id: fragment
            for fragment in self.binding.fragments
        }
        matching_bindings = tuple(
            binding
            for binding in request.source_bindings
            if isinstance(binding, WeightRuntimeBindingManifest)
            and binding.model_id == self.binding.model_id
            and binding.revision == self.binding.revision
            and binding.instance_id == self.binding.instance_id
            and binding.generation == self.binding.generation
            and binding.lease_id == self.binding.lease_id
            and all(
                source_fragments.get(fragment.placement_fragment_id) == fragment
                for fragment in binding.fragments
            )
        )
        if len(matching_bindings) != 1:
            raise RuntimeError(
                "materialization binding differs from the captured snapshot"
            )
        request_binding = matching_bindings[0]
        self.attest(request, request_binding=request_binding)
        matching_placements = tuple(
            placement
            for placement in request.source_placements
            if placement.placement_id == request_binding.placement_id
        )
        if len(matching_placements) != 1:
            raise RuntimeError(
                "materialization placements differ from the captured snapshot"
            )
        local_placement = matching_placements[0]
        source_tensors = {
            tensor.placement_fragment_id: tensor for tensor in self.placement.tensors
        }
        if any(
            source_tensors.get(tensor.placement_fragment_id) != tensor
            for tensor in local_placement.tensors
        ):
            raise RuntimeError(
                "materialization placements differ from the captured snapshot"
            )
        payload_identity = getattr(request, "payload_identity", None)
        if not isinstance(payload_identity, WeightPayloadIdentity):
            raise RuntimeError("materialization payload identity is invalid")
        captured_identity = self.payload_identity
        if not isinstance(captured_identity, WeightPayloadIdentity):
            raise RuntimeError("runtime snapshot payload identity was not captured")
        if payload_identity.select((local_placement,)) != captured_identity.select(
            (local_placement,)
        ):
            raise RuntimeError(
                "materialization payload identity differs from the captured snapshot"
            )
        _check_payload_hash_context(execution_context)

    def attest(
        self,
        request: Any,
        *,
        request_binding: WeightRuntimeBindingManifest | None = None,
    ) -> None:
        if self.released:
            raise RuntimeError("runtime weight snapshot source is released")
        expected_binding = request_binding or self.binding
        if expected_binding not in tuple(request.source_bindings):
            raise RuntimeError(
                "materialization request differs from the runtime snapshot"
            )
        if request_binding is not None and request_binding != self.binding:
            source_fragments = {
                fragment.placement_fragment_id: fragment
                for fragment in self.binding.fragments
            }
            if (
                request_binding.model_id != self.binding.model_id
                or request_binding.revision != self.binding.revision
                or request_binding.instance_id != self.binding.instance_id
                or request_binding.generation != self.binding.generation
                or request_binding.lease_id != self.binding.lease_id
                or any(
                    source_fragments.get(fragment.placement_fragment_id) != fragment
                    for fragment in request_binding.fragments
                )
            ):
                raise RuntimeError(
                    "projected materialization binding differs from the runtime snapshot"
                )
        self.manager.attest_binding(self.binding)

    def release(self) -> None:
        if self.released:
            return
        if self.quarantined:
            raise RuntimeError("completion-unknown runtime snapshot cannot be released")
        self._release_lease()

    def _release_lease(self) -> None:
        has_lease = getattr(self.manager, "has_lease", None)
        if not callable(has_lease) or has_lease(self.binding.lease_id):
            self.manager.release(self.binding.lease_id)
        self.released = True

    def quarantine(self, error: WeightTransferCompletionUnknownError) -> None:
        if not isinstance(error, WeightTransferCompletionUnknownError):
            raise TypeError("runtime snapshot quarantine requires completion unknown")
        self.quarantined = True
        self.operation_id = error.operation_id
        self.provider_name = error.provider
        self.completion_ticket = error.completion_ticket
        if self not in _RUNTIME_WEIGHT_SNAPSHOT_QUARANTINE:
            _RUNTIME_WEIGHT_SNAPSHOT_QUARANTINE.append(self)

    def resolve_quarantine(self, proof: WeightTransferTerminalProof) -> None:
        if not self.quarantined or self.released:
            raise RuntimeError("runtime weight snapshot is not quarantined")
        if not isinstance(proof, WeightTransferTerminalProof):
            raise ValueError("runtime weight snapshot requires a terminal proof")
        if (
            proof.operation_id != self.operation_id
            or proof.provider != self.provider_name
            or proof.completion_ticket != self.completion_ticket
        ):
            raise ValueError(
                "terminal proof differs from the quarantined materialization"
            )
        self._release_lease()
        self.quarantined = False
        _RUNTIME_WEIGHT_SNAPSHOT_QUARANTINE.remove(self)


def quarantined_runtime_weight_snapshots() -> tuple[RuntimeWeightSnapshotSource, ...]:
    return tuple(_RUNTIME_WEIGHT_SNAPSHOT_QUARANTINE)


def _completion_unknown_error(
    error: WeightTransferError,
) -> WeightTransferCompletionUnknownError:
    if isinstance(error, WeightTransferCompletionUnknownError):
        return error
    return WeightTransferCompletionUnknownError(
        str(error) or error.__class__.__name__,
        provider=error.provider,
        phase=error.phase,
        operation_id=error.operation_id,
        completion_ticket=getattr(error, "completion_ticket", None),
    )


@dataclass(frozen=True)
class _RuntimeSourceAttestor:
    source: RuntimeWeightSnapshotSource
    additional: WeightTransferAttestor | None
    request_binding: WeightRuntimeBindingManifest | None = None

    def attest(self, request: Any) -> None:
        self.source.attest(
            request,
            request_binding=self.request_binding,
        )
        if self.additional is not None:
            self.additional.attest(request)


def _captured_payload_identity(
    source: RuntimeWeightSnapshotSource,
) -> WeightPayloadIdentity:
    identity = source.payload_identity
    if not isinstance(identity, WeightPayloadIdentity):
        raise RuntimeError("runtime snapshot payload identity was not captured")
    return identity


def materialize_runtime_weights(
    source: RuntimeWeightSnapshotSource,
    *,
    destination: WeightStorageDestination,
    provider: WeightTransferProvider,
    additional_attestor: WeightTransferAttestor | None = None,
    execution_context: WeightTransferExecutionContext | None = None,
) -> WeightMaterializeReceipt:
    """Materialize one owned runtime snapshot without publishing a catalog ref."""

    if not isinstance(source, RuntimeWeightSnapshotSource):
        raise ValueError("runtime weight snapshot source is invalid")
    if source.released or source.quarantined:
        raise RuntimeError("runtime weight snapshot source is not materializable")
    payload_identity = _captured_payload_identity(source)
    try:
        receipt = materialize_weights(
            source_placements=(source.placement,),
            source_bindings=(source.binding,),
            destination=destination,
            provider=provider,
            payload_identity=payload_identity,
            attestor=_RuntimeSourceAttestor(source, additional_attestor),
            execution_context=execution_context,
        )
    except WeightTransferError as error:
        if not error.completion_known:
            completion_unknown = _completion_unknown_error(error)
            source.quarantine(completion_unknown)
            if completion_unknown is error:
                raise
            raise completion_unknown from error
        source.release()
        raise
    except BaseException:
        source.release()
        raise
    source.release()
    return receipt


def materialize_runtime_weight_snapshot(
    source: RuntimeWeightSnapshotSource,
    *,
    destination: WeightStorageDestination,
    provider: WeightTransferProvider,
    catalog: WeightStorageCatalog,
    publication_id: str | None = None,
    additional_attestor: WeightTransferAttestor | None = None,
    release_source: bool = True,
    execution_context: WeightTransferExecutionContext | None = None,
) -> WeightSnapshotPublication:
    """Publish one captured runtime snapshot, optionally retaining its lease."""

    if not isinstance(source, RuntimeWeightSnapshotSource):
        raise ValueError("runtime weight snapshot source is invalid")
    if source.released or source.quarantined:
        raise RuntimeError("runtime weight snapshot source is not materializable")
    payload_identity = _captured_payload_identity(source)
    return _materialize_runtime_weight_snapshot(
        source,
        source_placements=(source.placement,),
        source_bindings=(source.binding,),
        payload_identity=payload_identity,
        destination=destination,
        provider=provider,
        catalog=catalog,
        publication_id=publication_id,
        additional_attestor=additional_attestor,
        release_source=release_source,
        execution_context=execution_context,
    )


def materialize_distributed_runtime_weight_snapshot(
    source: RuntimeWeightSnapshotSource,
    *,
    global_placements: Sequence[WeightPlacementManifest],
    global_bindings: Sequence[WeightRuntimeBindingManifest],
    payload_identity: WeightPayloadIdentity,
    destination: WeightStorageDestination,
    provider: WeightTransferProvider,
    catalog: WeightStorageCatalog,
    publication_id: str | None = None,
    additional_attestor: WeightTransferAttestor | None = None,
    release_source: bool = True,
    execution_context: WeightTransferExecutionContext | None = None,
) -> WeightSnapshotPublication:
    """Publish a global snapshot, optionally retaining this rank's lease."""

    placements = tuple(global_placements)
    bindings = tuple(global_bindings)
    if not isinstance(source, RuntimeWeightSnapshotSource):
        raise ValueError("runtime weight snapshot source is invalid")
    if source.released or source.quarantined:
        raise RuntimeError("runtime weight snapshot source is not materializable")
    if not isinstance(payload_identity, WeightPayloadIdentity):
        raise ValueError("global runtime payload identity is invalid")
    bind_weight_source(placements, bindings)

    local_fragment_ids = {
        tensor.placement_fragment_id for tensor in source.placement.tensors
    }
    local_placements = tuple(
        placement
        for placement in placements
        if any(
            tensor.placement_fragment_id in local_fragment_ids
            for tensor in placement.tensors
        )
    )
    if len(local_placements) != 1 or any(
        tensor.placement_fragment_id not in local_fragment_ids
        for placement in local_placements
        for tensor in placement.tensors
    ):
        raise ValueError(
            "global runtime snapshot does not contain one local source projection"
        )
    local_binding = next(
        (
            binding
            for binding in bindings
            if binding.placement_id == local_placements[0].placement_id
        ),
        None,
    )
    projected_bindings = project_source_bindings(
        local_placements,
        (source.binding,),
    )
    if local_binding is None or projected_bindings != (local_binding,):
        raise ValueError(
            "global runtime binding differs from the local source projection"
        )
    local_payload_identity = _captured_payload_identity(source)
    if payload_identity.select(local_placements) != local_payload_identity.select(
        local_placements
    ):
        raise ValueError(
            "global runtime payload identity differs from the local source"
        )
    return _materialize_runtime_weight_snapshot(
        source,
        source_placements=placements,
        source_bindings=bindings,
        payload_identity=payload_identity,
        destination=destination,
        provider=provider,
        catalog=catalog,
        publication_id=publication_id,
        additional_attestor=additional_attestor,
        request_binding=local_binding,
        release_source=release_source,
        execution_context=execution_context,
    )


def _materialize_runtime_weight_snapshot(
    source: RuntimeWeightSnapshotSource,
    *,
    source_placements: Sequence[WeightPlacementManifest],
    source_bindings: Sequence[WeightRuntimeBindingManifest],
    payload_identity: WeightPayloadIdentity,
    destination: WeightStorageDestination,
    provider: WeightTransferProvider,
    catalog: WeightStorageCatalog,
    publication_id: str | None,
    additional_attestor: WeightTransferAttestor | None,
    request_binding: WeightRuntimeBindingManifest | None = None,
    release_source: bool = True,
    execution_context: WeightTransferExecutionContext | None = None,
) -> WeightSnapshotPublication:
    if type(release_source) is not bool:
        raise ValueError("release_source must be a boolean")
    try:
        publication = materialize_weight_snapshot(
            source_placements=source_placements,
            source_bindings=source_bindings,
            destination=destination,
            provider=provider,
            catalog=catalog,
            payload_identity=payload_identity,
            publication_id=publication_id,
            attestor=_RuntimeSourceAttestor(
                source,
                additional_attestor,
                request_binding,
            ),
            execution_context=execution_context,
        )
    except WeightTransferError as error:
        if not error.completion_known:
            completion_unknown = _completion_unknown_error(error)
            source.quarantine(completion_unknown)
            if completion_unknown is error:
                raise
            raise completion_unknown from error
        if release_source:
            source.release()
        raise
    except BaseException:
        if release_source:
            source.release()
        raise
    if release_source:
        source.release()
    return publication
