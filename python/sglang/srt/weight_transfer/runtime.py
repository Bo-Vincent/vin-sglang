from __future__ import annotations

import hashlib
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
                key=lambda item: (item.nbytes, item.address, item.device),
            )
        )
        self.chunk_bytes = chunk_bytes

    def _resolve(self, location: RuntimeWeightLocation) -> tuple[Any, int]:
        if not isinstance(location, RuntimeWeightLocation):
            raise ValueError("runtime payload location is invalid")
        end = location.address + location.nbytes
        for current in self._ranges:
            if (
                current.device == location.device
                and current.address <= location.address
                and end <= current.address + current.nbytes
            ):
                return current, location.address - current.address
        raise ValueError("runtime payload range is not owned by the runtime model")

    def __call__(self, location: RuntimeWeightLocation) -> str:
        current, begin = self._resolve(location)
        byte_view = current.value.reshape(-1).view(torch.uint8)
        digest = hashlib.sha256()
        for offset in range(0, location.nbytes, self.chunk_bytes):
            nbytes = min(self.chunk_bytes, location.nbytes - offset)
            chunk = byte_view.narrow(0, begin + offset, nbytes)
            if chunk.device.type != "cpu":
                chunk = chunk.to(device="cpu", non_blocking=False)
            digest.update(chunk.contiguous().numpy().tobytes())
        return f"sha256:{digest.hexdigest()}"


_RUNTIME_WEIGHT_SNAPSHOT_QUARANTINE: list[RuntimeWeightSnapshotSource] = []


@dataclass
class RuntimeWeightSnapshotSource:
    """Own one runtime snapshot lease through Store materialization."""

    model: Any
    manager: Any
    parts: WeightRuntimeManifestParts
    payload_hasher: RuntimeWeightPayloadHasher
    payload_identity: WeightPayloadIdentity
    released: bool = False
    quarantined: bool = False
    operation_id: str | None = None
    provider_name: str | None = None
    completion_ticket: str | None = None

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
    ) -> RuntimeWeightSnapshotSource:
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
            hasher = RuntimeWeightPayloadHasher(
                model,
                chunk_bytes=checksum_chunk_bytes,
            )
            locations = bind_weight_source(
                (parts.placement,),
                (parts.binding,),
            )
            payload_identity = WeightPayloadIdentity.create(
                (parts.placement,),
                {
                    location.placement_fragment_id: hasher(location)
                    for location in locations
                },
            )
            return cls(
                model=model,
                manager=manager,
                parts=parts,
                payload_hasher=hasher,
                payload_identity=payload_identity,
            )
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

    def payload_checksum(self, location: RuntimeWeightLocation) -> str:
        if self.released:
            raise RuntimeError("runtime weight snapshot source is released")
        return self.payload_hasher(location)

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

    def _quarantine(self, error: WeightTransferCompletionUnknownError) -> None:
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


def materialize_runtime_weights(
    source: RuntimeWeightSnapshotSource,
    *,
    destination: WeightStorageDestination,
    provider: WeightTransferProvider,
    additional_attestor: WeightTransferAttestor | None = None,
) -> WeightMaterializeReceipt:
    """Materialize one owned runtime snapshot without publishing a catalog ref."""

    if not isinstance(source, RuntimeWeightSnapshotSource):
        raise ValueError("runtime weight snapshot source is invalid")
    if source.released or source.quarantined:
        raise RuntimeError("runtime weight snapshot source is not materializable")
    try:
        receipt = materialize_weights(
            source_placements=(source.placement,),
            source_bindings=(source.binding,),
            destination=destination,
            provider=provider,
            payload_identity=source.payload_identity,
            attestor=_RuntimeSourceAttestor(source, additional_attestor),
        )
    except WeightTransferCompletionUnknownError as error:
        source._quarantine(error)
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
) -> WeightSnapshotPublication:
    """Publish one captured runtime snapshot, optionally retaining its lease."""

    if not isinstance(source, RuntimeWeightSnapshotSource):
        raise ValueError("runtime weight snapshot source is invalid")
    if source.released or source.quarantined:
        raise RuntimeError("runtime weight snapshot source is not materializable")
    return _materialize_runtime_weight_snapshot(
        source,
        source_placements=(source.placement,),
        source_bindings=(source.binding,),
        payload_identity=source.payload_identity,
        destination=destination,
        provider=provider,
        catalog=catalog,
        publication_id=publication_id,
        additional_attestor=additional_attestor,
        release_source=release_source,
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
    if payload_identity.select(local_placements) != source.payload_identity.select(
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
        )
    except WeightTransferCompletionUnknownError as error:
        source._quarantine(error)
        raise
    except BaseException:
        if release_source:
            source.release()
        raise
    if release_source:
        source.release()
    return publication
