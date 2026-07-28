from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import msgspec
from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightManifestError,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
)
from sglang.srt.weight_transfer.api import (
    load_weights_to_local_target,
    materialize_weight_snapshot,
    materialize_weights,
)
from sglang.srt.weight_transfer.contracts import (
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
)
from sglang.srt.weight_transfer.provider import (
    WeightLoadReceipt,
    WeightMaterializeReceipt,
    WeightPayloadIdentity,
    WeightStorageDestination,
    WeightTargetLoadMode,
    WeightTargetLoadSession,
    WeightTransferAttestor,
    WeightTransferProvider,
)
from sglang.srt.weight_transfer.runtime import (
    RuntimeWeightSnapshotSource,
    materialize_runtime_weight_snapshot,
    materialize_runtime_weights,
)
from sglang.srt.weight_transfer.storage import (
    WeightSnapshotPublication,
    WeightStorageCatalog,
)

T = TypeVar("T")
_SIDECAR_FORMAT = "sglang-semantic-checkpoint"
_SIDECAR_VERSION = 1
_SIDECAR_MOE_DP_VERSION = 2
_DEFAULT_MAX_SIDECAR_BYTES = 64 * 1024 * 1024
_MAX_SIDECAR_BYTES = 256 * 1024 * 1024


class _RankRecord(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
    omit_defaults=True,
):
    dp: int
    tp: int
    pp: int
    ep: int
    moe_dp: int = 0


class _PlacementTensorRecord(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    placement_fragment_id: str
    tensor_id: str
    runtime_name: str
    aliases: tuple[str, ...]
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    dtype: str
    itemsize: int
    partition_dim: int | None
    shard_dims: tuple[int, ...]
    layer_id: int | None
    expert_id: int | None
    layout_fingerprint: str
    nbytes: int
    byte_offset: int
    rank: _RankRecord
    expert_axis: int | None = None


class _PlacementRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    model_id: str
    revision: str
    placement_id: str
    tensors: tuple[_PlacementTensorRecord, ...]
    format_version: int


class _StorageFragmentRecord(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    placement_fragment_id: str
    fragment_id: str
    object_key: str
    object_offset: int
    nbytes: int
    checksum: str | None


class _StorageBindingRecord(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    model_id: str
    revision: str
    placement_id: str
    storage_id: str
    provider: str
    fragments: tuple[_StorageFragmentRecord, ...]
    format_version: int


class _SemanticCheckpointEnvelope(
    msgspec.Struct,
    frozen=True,
    forbid_unknown_fields=True,
):
    format: str
    version: int
    placements: tuple[_PlacementRecord, ...]
    bindings: tuple[_StorageBindingRecord, ...]


def _validate_sidecar_limit(max_bytes: int) -> int:
    if type(max_bytes) is not int or max_bytes <= 0 or max_bytes > _MAX_SIDECAR_BYTES:
        raise ValueError(
            f"sidecar size limit must be between 1 and {_MAX_SIDECAR_BYTES} bytes"
        )
    return max_bytes


def _rank_record(rank: WeightParallelRank) -> _RankRecord:
    return _RankRecord(
        dp=rank.dp,
        tp=rank.tp,
        pp=rank.pp,
        ep=rank.ep,
        moe_dp=rank.moe_dp,
    )


def _placement_record(placement: WeightPlacementManifest) -> _PlacementRecord:
    return _PlacementRecord(
        model_id=placement.model_id,
        revision=placement.revision,
        placement_id=placement.placement_id,
        tensors=tuple(
            _PlacementTensorRecord(
                placement_fragment_id=tensor.placement_fragment_id,
                tensor_id=tensor.tensor_id,
                runtime_name=tensor.runtime_name,
                aliases=tensor.aliases,
                global_shape=tensor.global_shape,
                global_offset=tensor.global_offset,
                local_shape=tensor.local_shape,
                dtype=tensor.dtype,
                itemsize=tensor.itemsize,
                partition_dim=tensor.partition_dim,
                shard_dims=tensor.shard_dims,
                layer_id=tensor.layer_id,
                expert_id=tensor.expert_id,
                expert_axis=tensor.expert_axis,
                layout_fingerprint=tensor.layout_fingerprint,
                nbytes=tensor.nbytes,
                byte_offset=tensor.byte_offset,
                rank=_rank_record(tensor.rank),
            )
            for tensor in placement.tensors
        ),
        format_version=placement.format_version,
    )


def _binding_record(
    binding: WeightStorageBindingManifest,
) -> _StorageBindingRecord:
    return _StorageBindingRecord(
        model_id=binding.model_id,
        revision=binding.revision,
        placement_id=binding.placement_id,
        storage_id=binding.storage_id,
        provider=binding.provider,
        fragments=tuple(
            _StorageFragmentRecord(
                placement_fragment_id=fragment.placement_fragment_id,
                fragment_id=fragment.fragment_id,
                object_key=fragment.object_key,
                object_offset=fragment.object_offset,
                nbytes=fragment.nbytes,
                checksum=fragment.checksum,
            )
            for fragment in binding.fragments
        ),
        format_version=binding.format_version,
    )


def _placement_from_record(record: _PlacementRecord) -> WeightPlacementManifest:
    return WeightPlacementManifest(
        model_id=record.model_id,
        revision=record.revision,
        placement_id=record.placement_id,
        tensors=tuple(
            WeightPlacementTensor(
                placement_fragment_id=tensor.placement_fragment_id,
                tensor_id=tensor.tensor_id,
                runtime_name=tensor.runtime_name,
                aliases=tensor.aliases,
                global_shape=tensor.global_shape,
                global_offset=tensor.global_offset,
                local_shape=tensor.local_shape,
                dtype=tensor.dtype,
                itemsize=tensor.itemsize,
                partition_dim=tensor.partition_dim,
                shard_dims=tensor.shard_dims,
                layer_id=tensor.layer_id,
                expert_id=tensor.expert_id,
                expert_axis=tensor.expert_axis,
                layout_fingerprint=tensor.layout_fingerprint,
                nbytes=tensor.nbytes,
                byte_offset=tensor.byte_offset,
                rank=WeightParallelRank(
                    dp=tensor.rank.dp,
                    tp=tensor.rank.tp,
                    pp=tensor.rank.pp,
                    ep=tensor.rank.ep,
                    moe_dp=tensor.rank.moe_dp,
                ),
            )
            for tensor in record.tensors
        ),
        format_version=record.format_version,
    )


def _binding_from_record(
    record: _StorageBindingRecord,
) -> WeightStorageBindingManifest:
    return WeightStorageBindingManifest(
        model_id=record.model_id,
        revision=record.revision,
        placement_id=record.placement_id,
        storage_id=record.storage_id,
        provider=record.provider,
        fragments=tuple(
            WeightStorageFragmentBinding(
                placement_fragment_id=fragment.placement_fragment_id,
                fragment_id=fragment.fragment_id,
                object_key=fragment.object_key,
                object_offset=fragment.object_offset,
                nbytes=fragment.nbytes,
                checksum=fragment.checksum,
            )
            for fragment in record.fragments
        ),
        format_version=record.format_version,
    )


@dataclass(frozen=True)
class SemanticCheckpointSource:
    """Checkpoint or OSS ranges paired with an explicit placement sidecar."""

    placements: tuple[WeightPlacementManifest, ...]
    bindings: tuple[WeightStorageBindingManifest, ...]

    def __post_init__(self) -> None:
        placements = tuple(self.placements)
        bindings = tuple(self.bindings)
        if not placements or not bindings:
            raise ValueError(
                "semantic checkpoint placements and bindings must not be empty"
            )
        if not all(
            isinstance(placement, WeightPlacementManifest) for placement in placements
        ) or not all(
            isinstance(binding, WeightStorageBindingManifest) for binding in bindings
        ):
            raise ValueError("semantic checkpoint manifests are invalid")
        placement_ids = [placement.placement_id for placement in placements]
        binding_ids = [binding.placement_id for binding in bindings]
        if (
            len(placement_ids) != len(set(placement_ids))
            or len(binding_ids) != len(set(binding_ids))
            or set(placement_ids) != set(binding_ids)
        ):
            raise ValueError("semantic checkpoint placement and binding IDs must match")
        placements_by_id = {
            placement.placement_id: placement for placement in placements
        }
        for binding in bindings:
            placement = placements_by_id[binding.placement_id]
            if (
                binding.model_id != placement.model_id
                or binding.revision != placement.revision
            ):
                raise ValueError(
                    "semantic checkpoint binding identity differs from placement"
                )
            expected_fragments = {
                tensor.placement_fragment_id: tensor.nbytes
                for tensor in placement.tensors
            }
            actual_fragments = {
                fragment.placement_fragment_id: fragment.nbytes
                for fragment in binding.fragments
            }
            if expected_fragments != actual_fragments:
                raise ValueError(
                    "semantic checkpoint binding fragments differ from placement"
                )
        object.__setattr__(self, "placements", placements)
        object.__setattr__(self, "bindings", bindings)

    def to_json_bytes(
        self,
        *,
        max_bytes: int = _DEFAULT_MAX_SIDECAR_BYTES,
    ) -> bytes:
        max_bytes = _validate_sidecar_limit(max_bytes)
        payload = msgspec.json.encode(
            _SemanticCheckpointEnvelope(
                format=_SIDECAR_FORMAT,
                version=(
                    _SIDECAR_MOE_DP_VERSION
                    if any(
                        tensor.rank.moe_dp
                        for placement in self.placements
                        for tensor in placement.tensors
                    )
                    else _SIDECAR_VERSION
                ),
                placements=tuple(
                    _placement_record(placement) for placement in self.placements
                ),
                bindings=tuple(_binding_record(binding) for binding in self.bindings),
            )
        )
        if len(payload) > max_bytes:
            raise ValueError(
                f"semantic checkpoint sidecar size exceeds {max_bytes} bytes"
            )
        return payload

    @classmethod
    def from_json_bytes(
        cls,
        payload: bytes | bytearray | memoryview,
        *,
        max_bytes: int = _DEFAULT_MAX_SIDECAR_BYTES,
    ) -> SemanticCheckpointSource:
        max_bytes = _validate_sidecar_limit(max_bytes)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise ValueError("semantic checkpoint sidecar must be bytes-like")
        payload_size = len(payload)
        if payload_size <= 0 or payload_size > max_bytes:
            raise ValueError(
                f"semantic checkpoint sidecar size must be between 1 and "
                f"{max_bytes} bytes"
            )
        encoded = bytes(payload)
        try:
            envelope = msgspec.json.decode(
                encoded,
                type=_SemanticCheckpointEnvelope,
            )
            if envelope.format != _SIDECAR_FORMAT or envelope.version not in (
                _SIDECAR_VERSION,
                _SIDECAR_MOE_DP_VERSION,
            ):
                raise ValueError("unsupported semantic checkpoint sidecar")
            return cls(
                placements=tuple(
                    _placement_from_record(placement)
                    for placement in envelope.placements
                ),
                bindings=tuple(
                    _binding_from_record(binding) for binding in envelope.bindings
                ),
            )
        except (
            msgspec.DecodeError,
            TypeError,
            ValueError,
            WeightManifestError,
        ) as error:
            raise ValueError("invalid semantic checkpoint sidecar") from error

    def write_sidecar(
        self,
        path: str | Path,
        *,
        max_bytes: int = _DEFAULT_MAX_SIDECAR_BYTES,
    ) -> None:
        Path(path).write_bytes(self.to_json_bytes(max_bytes=max_bytes))

    @classmethod
    def read_sidecar(
        cls,
        path: str | Path,
        *,
        max_bytes: int = _DEFAULT_MAX_SIDECAR_BYTES,
    ) -> SemanticCheckpointSource:
        max_bytes = _validate_sidecar_limit(max_bytes)
        sidecar_path = Path(path)
        try:
            with sidecar_path.open("rb") as sidecar_file:
                payload = sidecar_file.read(max_bytes + 1)
        except OSError as error:
            raise ValueError("semantic checkpoint sidecar is unavailable") from error
        if len(payload) <= 0 or len(payload) > max_bytes:
            raise ValueError(
                f"semantic checkpoint sidecar size must be between 1 and "
                f"{max_bytes} bytes"
            )
        return cls.from_json_bytes(payload, max_bytes=max_bytes)


def load_checkpoint_weights(
    *,
    source: SemanticCheckpointSource | None,
    target_placements: Sequence[WeightPlacementManifest],
    target_bindings: Sequence[WeightRuntimeBindingManifest],
    provider: WeightTransferProvider | None,
    framework_loader: Callable[[], T] | None,
    target_mode: WeightTargetLoadMode,
    attestor: WeightTransferAttestor | None = None,
    target_session: WeightTargetLoadSession | None = None,
) -> WeightLoadReceipt | T:
    """Load a semantic sidecar directly or delegate to the framework loader."""

    if source is None:
        if target_mode is not WeightTargetLoadMode.COLD_START:
            raise ValueError(
                "framework checkpoint fallback is only valid during cold start"
            )
        if target_session is not None:
            raise ValueError(
                "framework checkpoint fallback must not use a live target session"
            )
        if framework_loader is None:
            raise ValueError(
                "checkpoint loading without a semantic sidecar requires "
                "a framework loader"
            )
        return framework_loader()
    if provider is None:
        raise ValueError("semantic checkpoint loading requires a provider")
    if len(target_placements) != 1 or len(target_bindings) != 1:
        raise ValueError(
            "semantic checkpoint loading requires exactly one local target "
            "placement and binding"
        )
    return load_weights_to_local_target(
        source_placements=source.placements,
        source_bindings=source.bindings,
        target_placement=target_placements[0],
        target_binding=target_bindings[0],
        provider=provider,
        target_mode=target_mode,
        attestor=attestor,
        target_session=target_session,
    )


def materialize_checkpoint_weights(
    *,
    source: SemanticCheckpointSource | None,
    destination: WeightStorageDestination,
    provider: WeightTransferProvider,
    framework_load_and_export: Callable[[], RuntimeWeightSnapshotSource] | None,
    payload_identity: WeightPayloadIdentity | None = None,
    attestor: WeightTransferAttestor | None = None,
) -> WeightMaterializeReceipt:
    """Write semantic ranges directly, or materialize an owned SGLang runtime."""

    if source is None:
        if framework_load_and_export is None:
            raise ValueError(
                "checkpoint materialization without a semantic sidecar "
                "requires a framework load-and-export callback"
            )
        runtime_source = framework_load_and_export()
        if not isinstance(runtime_source, RuntimeWeightSnapshotSource):
            raise ValueError(
                "framework load-and-export must return an owned runtime snapshot"
            )
        if (
            payload_identity is not None
            and payload_identity != runtime_source.payload_identity
        ):
            runtime_source.release()
            raise ValueError(
                "checkpoint payload identity differs from runtime snapshot"
            )
        return materialize_runtime_weights(
            runtime_source,
            destination=destination,
            provider=provider,
            additional_attestor=attestor,
        )

    return materialize_weights(
        source_placements=source.placements,
        source_bindings=source.bindings,
        destination=destination,
        provider=provider,
        payload_identity=payload_identity,
        attestor=attestor,
    )


def materialize_checkpoint_weight_snapshot(
    *,
    source: SemanticCheckpointSource | None,
    destination: WeightStorageDestination,
    provider: WeightTransferProvider,
    catalog: WeightStorageCatalog,
    framework_load_and_export: Callable[[], RuntimeWeightSnapshotSource] | None,
    publication_id: str | None = None,
    payload_identity: WeightPayloadIdentity | None = None,
    attestor: WeightTransferAttestor | None = None,
) -> WeightSnapshotPublication:
    """Publish semantic ranges or an owned runtime as a reusable snapshot."""

    if source is None:
        if framework_load_and_export is None:
            raise ValueError(
                "checkpoint snapshot materialization without a semantic "
                "sidecar requires a framework load-and-export callback"
            )
        runtime_source = framework_load_and_export()
        if not isinstance(runtime_source, RuntimeWeightSnapshotSource):
            raise ValueError(
                "framework load-and-export must return an owned runtime snapshot"
            )
        if (
            payload_identity is not None
            and payload_identity != runtime_source.payload_identity
        ):
            runtime_source.release()
            raise ValueError(
                "checkpoint payload identity differs from runtime snapshot"
            )
        return materialize_runtime_weight_snapshot(
            runtime_source,
            destination=destination,
            provider=provider,
            catalog=catalog,
            publication_id=publication_id,
            additional_attestor=attestor,
        )

    return materialize_weight_snapshot(
        source_placements=source.placements,
        source_bindings=source.bindings,
        destination=destination,
        provider=provider,
        catalog=catalog,
        payload_identity=payload_identity,
        publication_id=publication_id,
        attestor=attestor,
    )
