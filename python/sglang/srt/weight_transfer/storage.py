from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from math import prod
from threading import RLock
from typing import Iterable, Protocol, Sequence

from sglang.srt.model_executor.weight_runtime_manifest import (
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
)
from sglang.srt.weight_transfer.contracts import (
    WeightStorageBindingManifest,
    WeightStorageFragmentBinding,
)

_MAX_STORAGE_RANGE_END = (1 << 64) - 1
_SNAPSHOT_FORMAT = "sglang-stored-weight-snapshot-v1"
_MATERIALIZATION_FORMAT = "sglang-weight-materialization-intent-v1"


def _require_nonempty_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_canonical_digest(value: object, name: str) -> str:
    value = _require_nonempty_string(value, name)
    prefix = "sha256:"
    payload = value.removeprefix(prefix)
    if (
        not value.startswith(prefix)
        or len(payload) != 64
        or payload != payload.lower()
        or any(character not in "0123456789abcdef" for character in payload)
    ):
        raise ValueError(f"{name} must be a canonical sha256 digest")
    return value


def _require_bounded_range(offset: object, nbytes: object, name: str) -> None:
    if type(offset) is not int or offset < 0:
        raise ValueError(f"{name} offset must be a non-negative integer")
    if type(nbytes) is not int or nbytes <= 0:
        raise ValueError(f"{name} byte size must be positive")
    if offset > _MAX_STORAGE_RANGE_END - nbytes:
        raise ValueError(f"{name} range exceeds the supported 64-bit bound")


def _validate_placement_tensor(tensor: WeightPlacementTensor) -> None:
    for name in (
        "placement_fragment_id",
        "tensor_id",
        "runtime_name",
        "dtype",
        "layout_fingerprint",
    ):
        _require_nonempty_string(getattr(tensor, name), name)
    if (
        type(tensor.itemsize) is not int
        or tensor.itemsize <= 0
        or type(tensor.nbytes) is not int
        or tensor.nbytes <= 0
    ):
        raise ValueError("placement tensor byte size is invalid")
    shape = tuple(tensor.global_shape)
    offset = tuple(tensor.global_offset)
    local_shape = tuple(tensor.local_shape)
    if (
        not shape
        or len(offset) != len(shape)
        or len(local_shape) != len(shape)
        or any(type(extent) is not int or extent <= 0 for extent in shape)
        or any(type(begin) is not int or begin < 0 for begin in offset)
        or any(type(extent) is not int or extent <= 0 for extent in local_shape)
        or any(
            begin + extent > global_extent
            for begin, extent, global_extent in zip(
                offset,
                local_shape,
                shape,
                strict=True,
            )
        )
    ):
        raise ValueError("placement tensor logical bounds are invalid")
    if prod(local_shape) * tensor.itemsize != tensor.nbytes:
        raise ValueError("placement tensor byte size differs from logical shape")
    _require_bounded_range(
        tensor.byte_offset,
        tensor.nbytes,
        "placement tensor",
    )
    aliases = tuple(tensor.aliases)
    if any(type(alias) is not str or not alias for alias in aliases) or len(
        aliases
    ) != len(set(aliases)):
        raise ValueError("placement tensor aliases are invalid")
    shard_dims = tuple(tensor.shard_dims)
    if (
        any(type(dim) is not int or dim < 0 or dim >= len(shape) for dim in shard_dims)
        or tuple(sorted(shard_dims)) != shard_dims
        or len(shard_dims) != len(set(shard_dims))
    ):
        raise ValueError("placement tensor shard dimensions are invalid")
    if tensor.partition_dim is not None and shard_dims != (tensor.partition_dim,):
        raise ValueError("placement tensor partition dimension is inconsistent")
    rank_values = (
        tensor.rank.dp,
        tensor.rank.tp,
        tensor.rank.pp,
        tensor.rank.ep,
    )
    if any(type(value) is not int or value < 0 for value in rank_values):
        raise ValueError("placement tensor parallel rank is invalid")


def _validate_and_normalize_snapshot(
    ref: WeightStorageRef,
    placements: Sequence[WeightPlacementManifest],
    storage_bindings: Sequence[WeightStorageBindingManifest],
) -> tuple[
    tuple[WeightPlacementManifest, ...],
    tuple[WeightStorageBindingManifest, ...],
]:
    if not isinstance(ref, WeightStorageRef):
        raise ValueError("ref must be a WeightStorageRef")
    placements = tuple(placements)
    storage_bindings = tuple(storage_bindings)
    if not placements:
        raise ValueError("stored snapshot placements must not be empty")
    if not storage_bindings:
        raise ValueError("stored snapshot bindings must not be empty")
    if any(not isinstance(item, WeightPlacementManifest) for item in placements):
        raise ValueError("stored snapshot placements are invalid")
    if any(
        not isinstance(item, WeightStorageBindingManifest) for item in storage_bindings
    ):
        raise ValueError("stored snapshot bindings are invalid")

    placement_ids = [item.placement_id for item in placements]
    if len(placement_ids) != len(set(placement_ids)):
        raise ValueError("duplicate placement ID in stored snapshot")
    binding_ids = [item.placement_id for item in storage_bindings]
    if len(binding_ids) != len(set(binding_ids)):
        raise ValueError("duplicate storage binding placement ID")
    if set(placement_ids) != set(binding_ids):
        raise ValueError("placements and storage bindings must correspond exactly")

    model_revisions = {(item.model_id, item.revision) for item in placements}
    if len(model_revisions) != 1:
        raise ValueError("stored placements must share model and revision")

    placement_by_id = {}
    placement_fragment_ids: set[str] = set()
    for placement in placements:
        for name in ("model_id", "revision", "placement_id"):
            _require_nonempty_string(getattr(placement, name), name)
        if not placement.tensors:
            raise ValueError("stored placement must contain fragments")
        local_ids = []
        for tensor in placement.tensors:
            _validate_placement_tensor(tensor)
            fragment_id = tensor.placement_fragment_id
            local_ids.append(fragment_id)
            if fragment_id in placement_fragment_ids:
                raise ValueError("duplicate placement fragment ID in stored snapshot")
            placement_fragment_ids.add(fragment_id)
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("duplicate placement fragment ID")
        placement_by_id[placement.placement_id] = placement

    provider_fragment_ids: set[str] = set()
    ranges_by_object: dict[str, list[tuple[int, int]]] = {}
    for binding in storage_bindings:
        placement = placement_by_id[binding.placement_id]
        if (
            binding.model_id != placement.model_id
            or binding.revision != placement.revision
        ):
            raise ValueError("storage binding model or revision differs from placement")
        if binding.provider != ref.provider:
            raise ValueError("storage binding provider differs from ref")
        if binding.storage_id != ref.storage_id:
            raise ValueError("storage binding storage_id differs from ref")

        tensor_by_id = {
            tensor.placement_fragment_id: tensor for tensor in placement.tensors
        }
        fragment_by_id = {
            fragment.placement_fragment_id: fragment for fragment in binding.fragments
        }
        if set(tensor_by_id) != set(fragment_by_id):
            raise ValueError(
                "placement fragments and storage bindings must correspond exactly"
            )
        for fragment_id, fragment in fragment_by_id.items():
            if fragment.fragment_id in provider_fragment_ids:
                raise ValueError("duplicate provider fragment ID in stored snapshot")
            provider_fragment_ids.add(fragment.fragment_id)
            tensor = tensor_by_id[fragment_id]
            if fragment.nbytes != tensor.nbytes:
                raise ValueError("storage binding byte size differs from placement")
            _require_bounded_range(
                fragment.object_offset,
                fragment.nbytes,
                "storage object",
            )
            ranges_by_object.setdefault(fragment.object_key, []).append(
                (
                    fragment.object_offset,
                    fragment.object_offset + fragment.nbytes,
                )
            )

    for ranges in ranges_by_object.values():
        ranges.sort()
        if any(
            current_begin < previous_end
            for (_, previous_end), (current_begin, _) in zip(
                ranges,
                ranges[1:],
            )
        ):
            raise ValueError("storage object ranges overlap")

    return (
        tuple(sorted(placements, key=lambda item: item.placement_id)),
        tuple(sorted(storage_bindings, key=lambda item: item.placement_id)),
    )


def _tensor_identity(tensor: WeightPlacementTensor) -> dict[str, object]:
    rank_identity = {
        "dp": tensor.rank.dp,
        "ep": tensor.rank.ep,
        "pp": tensor.rank.pp,
        "tp": tensor.rank.tp,
    }
    if tensor.rank.moe_dp:
        rank_identity["moe_dp"] = tensor.rank.moe_dp
    return {
        "aliases": sorted(tensor.aliases),
        "byte_offset": tensor.byte_offset,
        "dtype": tensor.dtype,
        "expert_id": tensor.expert_id,
        "expert_axis": tensor.expert_axis,
        "global_offset": list(tensor.global_offset),
        "global_shape": list(tensor.global_shape),
        "itemsize": tensor.itemsize,
        "layer_id": tensor.layer_id,
        "layout_fingerprint": tensor.layout_fingerprint,
        "local_shape": list(tensor.local_shape),
        "nbytes": tensor.nbytes,
        "partition_dim": tensor.partition_dim,
        "placement_fragment_id": tensor.placement_fragment_id,
        "rank": rank_identity,
        "runtime_name": tensor.runtime_name,
        "shard_dims": list(tensor.shard_dims),
        "tensor_id": tensor.tensor_id,
    }


def _placement_identity(
    placement: WeightPlacementManifest,
) -> dict[str, object]:
    return {
        "format_version": placement.format_version,
        "model_id": placement.model_id,
        "placement_id": placement.placement_id,
        "revision": placement.revision,
        "tensors": [
            _tensor_identity(tensor)
            for tensor in sorted(
                placement.tensors,
                key=lambda item: item.placement_fragment_id,
            )
        ],
    }


def _fragment_binding_identity(
    fragment: WeightStorageFragmentBinding,
) -> dict[str, object]:
    return {
        "checksum": fragment.checksum,
        "fragment_id": fragment.fragment_id,
        "nbytes": fragment.nbytes,
        "object_key": fragment.object_key,
        "object_offset": fragment.object_offset,
        "placement_fragment_id": fragment.placement_fragment_id,
    }


def _storage_binding_identity(
    binding: WeightStorageBindingManifest,
) -> dict[str, object]:
    return {
        "format_version": binding.format_version,
        "fragments": [
            _fragment_binding_identity(fragment)
            for fragment in sorted(
                binding.fragments,
                key=lambda item: item.placement_fragment_id,
            )
        ],
        "model_id": binding.model_id,
        "placement_id": binding.placement_id,
        "provider": binding.provider,
        "revision": binding.revision,
        "storage_id": binding.storage_id,
    }


def _snapshot_digest(
    *,
    provider: str,
    storage_id: str,
    manifest_key: str,
    placements: Sequence[WeightPlacementManifest],
    storage_bindings: Sequence[WeightStorageBindingManifest],
) -> str:
    identity = {
        "format": _SNAPSHOT_FORMAT,
        "manifest_key": manifest_key,
        "placements": [_placement_identity(placement) for placement in placements],
        "provider": provider,
        "storage_bindings": [
            _storage_binding_identity(binding) for binding in storage_bindings
        ],
        "storage_id": storage_id,
    }
    payload = json.dumps(
        identity,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def weight_placement_set_digest(
    placements: Sequence[WeightPlacementManifest],
) -> str:
    placements = tuple(placements)
    if not placements:
        raise ValueError("weight placements must not be empty")
    placement_ids = [placement.placement_id for placement in placements]
    if len(placement_ids) != len(set(placement_ids)):
        raise ValueError("weight placements contain duplicate placement IDs")
    model_revisions = {
        (placement.model_id, placement.revision) for placement in placements
    }
    if len(model_revisions) != 1:
        raise ValueError("weight placements describe different revisions")
    for placement in placements:
        if not placement.tensors:
            raise ValueError("weight placement tensors must not be empty")
        for tensor in placement.tensors:
            _validate_placement_tensor(tensor)
    payload = json.dumps(
        {
            "format": _MATERIALIZATION_FORMAT,
            "placements": [
                _placement_identity(placement)
                for placement in sorted(
                    placements,
                    key=lambda item: item.placement_id,
                )
            ],
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _source_binding_identity(
    binding: WeightRuntimeBindingManifest | WeightStorageBindingManifest,
) -> dict:
    if isinstance(binding, WeightRuntimeBindingManifest):
        return {
            "kind": "runtime",
            "model_id": binding.model_id,
            "revision": binding.revision,
            "placement_id": binding.placement_id,
            "instance_id": binding.instance_id,
            "generation": binding.generation,
            "lease_id": binding.lease_id,
            "format_version": binding.format_version,
            "fragments": [
                {
                    "placement_fragment_id": fragment.placement_fragment_id,
                    "fragment_id": fragment.fragment_id,
                    "nbytes": fragment.nbytes,
                    "storage_offset": fragment.storage_offset,
                    "device": fragment.device,
                    "is_contiguous": fragment.is_contiguous,
                    "worker_id": fragment.worker_id,
                    "endpoint": fragment.endpoint,
                }
                for fragment in sorted(
                    binding.fragments,
                    key=lambda item: item.placement_fragment_id,
                )
            ],
        }
    return {
        "kind": "storage",
        "model_id": binding.model_id,
        "revision": binding.revision,
        "placement_id": binding.placement_id,
        "storage_id": binding.storage_id,
        "provider": binding.provider,
        "format_version": binding.format_version,
        "fragments": [
            {
                "placement_fragment_id": fragment.placement_fragment_id,
                "fragment_id": fragment.fragment_id,
                "object_key": fragment.object_key,
                "object_offset": fragment.object_offset,
                "nbytes": fragment.nbytes,
                "checksum": fragment.checksum,
            }
            for fragment in sorted(
                binding.fragments,
                key=lambda item: item.placement_fragment_id,
            )
        ],
    }


def weight_source_snapshot_digest(
    placements: Sequence[WeightPlacementManifest],
    bindings: Sequence[WeightRuntimeBindingManifest | WeightStorageBindingManifest],
) -> str:
    """Digest one strict source snapshot without coupling it to raw addresses."""

    placements = tuple(placements)
    bindings = tuple(bindings)
    placement_digest = weight_placement_set_digest(placements)
    placement_by_id = {placement.placement_id: placement for placement in placements}
    binding_by_id = {binding.placement_id: binding for binding in bindings}
    if len(binding_by_id) != len(bindings) or set(binding_by_id) != set(
        placement_by_id
    ):
        raise ValueError("weight source bindings differ from placements")
    for placement_id, placement in placement_by_id.items():
        binding = binding_by_id[placement_id]
        if (
            placement.model_id != binding.model_id
            or placement.revision != binding.revision
        ):
            raise ValueError("weight source binding model identity differs")
        placement_fragments = {
            tensor.placement_fragment_id: tensor for tensor in placement.tensors
        }
        binding_fragments = {
            fragment.placement_fragment_id: fragment for fragment in binding.fragments
        }
        if (
            len(placement_fragments) != len(placement.tensors)
            or len(binding_fragments) != len(binding.fragments)
            or set(placement_fragments) != set(binding_fragments)
        ):
            raise ValueError("weight source binding fragments differ")
        if any(
            binding_fragments[fragment_id].nbytes != tensor.nbytes
            for fragment_id, tensor in placement_fragments.items()
        ):
            raise ValueError("weight source binding byte size differs")
    payload = json.dumps(
        {
            "format": "sglang-weight-source-snapshot-v1",
            "placement_digest": placement_digest,
            "bindings": [
                _source_binding_identity(binding_by_id[placement_id])
                for placement_id in sorted(placement_by_id)
            ],
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def weight_stored_payload_digest(
    placements: Sequence[WeightPlacementManifest],
    bindings: Sequence[WeightStorageBindingManifest],
) -> str:
    placements = tuple(placements)
    bindings = tuple(bindings)
    placement_by_id = {placement.placement_id: placement for placement in placements}
    binding_by_id = {binding.placement_id: binding for binding in bindings}
    if (
        len(placement_by_id) != len(placements)
        or len(binding_by_id) != len(bindings)
        or set(placement_by_id) != set(binding_by_id)
    ):
        raise ValueError("stored payload placements and bindings differ")
    fragments = []
    for placement_id in sorted(placement_by_id):
        placement = placement_by_id[placement_id]
        binding = binding_by_id[placement_id]
        stored_by_id = {
            fragment.placement_fragment_id: fragment for fragment in binding.fragments
        }
        if len(stored_by_id) != len(binding.fragments):
            raise ValueError("stored payload has duplicate fragments")
        for tensor in placement.tensors:
            stored = stored_by_id.get(tensor.placement_fragment_id)
            if stored is None or stored.nbytes != tensor.nbytes:
                raise ValueError("stored payload fragment differs from placement")
            checksum = _require_canonical_digest(
                stored.checksum,
                "stored payload checksum",
            )
            fragments.append(
                {
                    "placement_fragment_id": tensor.placement_fragment_id,
                    "tensor_id": tensor.tensor_id,
                    "global_offset": tensor.global_offset,
                    "local_shape": tensor.local_shape,
                    "nbytes": tensor.nbytes,
                    "checksum": checksum,
                }
            )
    payload = json.dumps(
        {
            "format": "sglang-weight-payload-v1",
            "fragments": sorted(
                fragments,
                key=lambda item: item["placement_fragment_id"],
            ),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class WeightStorageRef:
    provider: str
    storage_id: str
    manifest_key: str
    manifest_digest: str

    def __post_init__(self) -> None:
        for name in ("provider", "storage_id", "manifest_key"):
            _require_nonempty_string(getattr(self, name), name)
        _require_canonical_digest(
            self.manifest_digest,
            "manifest_digest",
        )


@dataclass(frozen=True, eq=False)
class StoredWeightSnapshot:
    ref: WeightStorageRef
    placements: tuple[WeightPlacementManifest, ...]
    storage_bindings: tuple[WeightStorageBindingManifest, ...]
    digest: str

    def __post_init__(self) -> None:
        placements, storage_bindings = _validate_and_normalize_snapshot(
            self.ref,
            self.placements,
            self.storage_bindings,
        )
        digest = _require_canonical_digest(self.digest, "digest")
        if self.ref.manifest_digest != digest:
            raise ValueError("storage ref manifest digest differs from snapshot digest")
        expected = _snapshot_digest(
            provider=self.ref.provider,
            storage_id=self.ref.storage_id,
            manifest_key=self.ref.manifest_key,
            placements=placements,
            storage_bindings=storage_bindings,
        )
        if digest != expected:
            raise ValueError("stored snapshot digest is not canonical")
        object.__setattr__(self, "placements", placements)
        object.__setattr__(self, "storage_bindings", storage_bindings)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StoredWeightSnapshot) and self.ref == other.ref

    def __hash__(self) -> int:
        return hash(self.ref)

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        storage_id: str,
        manifest_key: str,
        placements: Sequence[WeightPlacementManifest],
        storage_bindings: Sequence[WeightStorageBindingManifest],
    ) -> StoredWeightSnapshot:
        provisional_ref = WeightStorageRef(
            provider=provider,
            storage_id=storage_id,
            manifest_key=manifest_key,
            manifest_digest="sha256:" + "0" * 64,
        )
        placements, storage_bindings = _validate_and_normalize_snapshot(
            provisional_ref,
            placements,
            storage_bindings,
        )
        digest = _snapshot_digest(
            provider=provider,
            storage_id=storage_id,
            manifest_key=manifest_key,
            placements=placements,
            storage_bindings=storage_bindings,
        )
        return cls(
            ref=replace(provisional_ref, manifest_digest=digest),
            placements=placements,
            storage_bindings=storage_bindings,
            digest=digest,
        )


class WeightRevisionState(str, Enum):
    READY = "ready"
    SERVING = "serving"
    IDLE = "idle"
    EVICTING = "evicting"
    EVICTED = "evicted"


@dataclass(frozen=True)
class WeightRevisionHead:
    model_id: str
    revision: str
    ref: WeightStorageRef
    generation: int
    state: WeightRevisionState

    def __post_init__(self) -> None:
        _require_nonempty_string(self.model_id, "model_id")
        _require_nonempty_string(self.revision, "revision")
        if not isinstance(self.ref, WeightStorageRef):
            raise ValueError("revision head ref must be a WeightStorageRef")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("revision head generation must be positive")
        try:
            state = WeightRevisionState(self.state)
        except ValueError as error:
            raise ValueError("revision head state is invalid") from error
        object.__setattr__(self, "state", state)


_REVISION_STATE_TRANSITIONS = {
    WeightRevisionState.READY: frozenset(
        {
            WeightRevisionState.READY,
            WeightRevisionState.SERVING,
            WeightRevisionState.IDLE,
            WeightRevisionState.EVICTING,
        }
    ),
    WeightRevisionState.SERVING: frozenset(
        {
            WeightRevisionState.SERVING,
            WeightRevisionState.IDLE,
        }
    ),
    WeightRevisionState.IDLE: frozenset(
        {
            WeightRevisionState.IDLE,
            WeightRevisionState.SERVING,
            WeightRevisionState.EVICTING,
        }
    ),
    WeightRevisionState.EVICTING: frozenset(
        {
            WeightRevisionState.EVICTING,
            WeightRevisionState.EVICTED,
        }
    ),
    WeightRevisionState.EVICTED: frozenset({WeightRevisionState.EVICTED}),
}


@dataclass(frozen=True)
class WeightMaterializationIntent:
    provider: str
    storage_id: str
    object_prefix: str
    model_id: str
    revision: str
    source_digest: str
    total_bytes: int
    fragment_count: int
    source_snapshot_digest: str | None = None
    payload_digest: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "provider",
            "storage_id",
            "object_prefix",
            "model_id",
            "revision",
        ):
            _require_nonempty_string(getattr(self, name), name)
        _require_canonical_digest(self.source_digest, "source_digest")
        if self.source_snapshot_digest is not None:
            _require_canonical_digest(
                self.source_snapshot_digest,
                "source_snapshot_digest",
            )
        if self.payload_digest is not None:
            _require_canonical_digest(self.payload_digest, "payload_digest")
        if type(self.total_bytes) is not int or self.total_bytes <= 0:
            raise ValueError("materialization total_bytes must be positive")
        if type(self.fragment_count) is not int or self.fragment_count <= 0:
            raise ValueError("materialization fragment_count must be positive")

    def matches_durable_recovery(self, other: object) -> bool:
        """Compare provider-neutral identity that survives runtime rebinding."""

        if (
            not isinstance(other, WeightMaterializationIntent)
            or self.payload_digest is None
            or other.payload_digest is None
        ):
            return False
        return (
            self.provider,
            self.storage_id,
            self.object_prefix,
            self.model_id,
            self.revision,
            self.source_digest,
            self.total_bytes,
            self.fragment_count,
            self.payload_digest,
        ) == (
            other.provider,
            other.storage_id,
            other.object_prefix,
            other.model_id,
            other.revision,
            other.source_digest,
            other.total_bytes,
            other.fragment_count,
            other.payload_digest,
        )


class WeightMaterializationAttemptState(str, Enum):
    PREPARING = "preparing"
    MATERIALIZED = "materialized"
    ABORTED = "aborted"


@dataclass(frozen=True)
class WeightMaterializationAttempt:
    materialization_id: str
    intent: WeightMaterializationIntent
    state: WeightMaterializationAttemptState
    snapshot: StoredWeightSnapshot | None = None
    completion_ticket: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(
            self.materialization_id,
            "materialization_id",
        )
        if not isinstance(self.intent, WeightMaterializationIntent):
            raise ValueError("materialization intent is invalid")
        try:
            state = WeightMaterializationAttemptState(self.state)
        except ValueError as error:
            raise ValueError("materialization attempt state is invalid") from error
        object.__setattr__(self, "state", state)
        if state is WeightMaterializationAttemptState.MATERIALIZED:
            if not isinstance(self.snapshot, StoredWeightSnapshot):
                raise ValueError("materialized attempt requires a stored snapshot")
        elif self.snapshot is not None:
            raise ValueError("non-materialized attempt must not contain a snapshot")
        if self.completion_ticket is not None:
            _require_nonempty_string(
                self.completion_ticket,
                "completion_ticket",
            )

    @property
    def recoverable(self) -> bool:
        return self.state is WeightMaterializationAttemptState.PREPARING


class WeightSnapshotPublicationState(str, Enum):
    PENDING = "pending"
    PUBLISHED = "published"
    ABORTED = "aborted"


@dataclass(frozen=True)
class WeightSnapshotPublication:
    publication_id: str
    snapshot: StoredWeightSnapshot
    state: WeightSnapshotPublicationState

    def __post_init__(self) -> None:
        _require_nonempty_string(self.publication_id, "publication_id")
        if not isinstance(self.snapshot, StoredWeightSnapshot):
            raise ValueError("publication snapshot is invalid")
        try:
            state = WeightSnapshotPublicationState(self.state)
        except ValueError as error:
            raise ValueError("publication state is invalid") from error
        object.__setattr__(self, "state", state)

    @property
    def recoverable(self) -> bool:
        return self.state is WeightSnapshotPublicationState.PENDING


class WeightStorageCatalog(Protocol):
    def begin_materialization(
        self,
        materialization_id: str,
        intent: WeightMaterializationIntent,
    ) -> WeightMaterializationAttempt: ...

    def complete_materialization(
        self,
        materialization_id: str,
        snapshot: StoredWeightSnapshot,
    ) -> WeightMaterializationAttempt: ...

    def abort_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt: ...

    def set_materialization_completion_ticket(
        self,
        materialization_id: str,
        completion_ticket: str,
    ) -> WeightMaterializationAttempt: ...

    def clear_materialization_completion_ticket(
        self,
        materialization_id: str,
        completion_ticket: str,
    ) -> WeightMaterializationAttempt: ...

    def get_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt | None: ...

    def recoverable_materializations(
        self,
    ) -> tuple[WeightMaterializationAttempt, ...]: ...

    def prepare_publish(
        self,
        publication_id: str,
        snapshot: StoredWeightSnapshot,
    ) -> WeightSnapshotPublication: ...

    def publish(self, publication_id: str) -> WeightSnapshotPublication: ...

    def abort(self, publication_id: str) -> WeightSnapshotPublication: ...

    def get_snapshot(
        self,
        ref: WeightStorageRef,
    ) -> StoredWeightSnapshot | None: ...

    def get_publication(
        self,
        publication_id: str,
    ) -> WeightSnapshotPublication | None: ...

    def recoverable_publications(
        self,
    ) -> tuple[WeightSnapshotPublication, ...]: ...

    def get_revision_head(
        self,
        model_id: str,
        revision: str,
    ) -> WeightRevisionHead | None: ...

    def compare_and_set_revision(
        self,
        *,
        model_id: str,
        revision: str,
        expected: WeightRevisionHead | None,
        new_ref: WeightStorageRef,
        new_state: WeightRevisionState,
    ) -> WeightRevisionHead | None: ...


class InMemoryWeightStorageCatalog:
    def __init__(
        self,
        *,
        materializations: Iterable[WeightMaterializationAttempt] = (),
        publications: Iterable[WeightSnapshotPublication] = (),
        revision_heads: Iterable[WeightRevisionHead] = (),
    ) -> None:
        self._lock = RLock()
        self._materializations: dict[str, WeightMaterializationAttempt] = {}
        self._publications: dict[str, WeightSnapshotPublication] = {}
        self._publication_by_ref: dict[WeightStorageRef, str] = {}
        self._snapshots: dict[WeightStorageRef, StoredWeightSnapshot] = {}
        self._revision_heads: dict[tuple[str, str], WeightRevisionHead] = {}
        for materialization in materializations:
            self._restore_materialization(materialization)
        for publication in publications:
            self._restore(publication)
        for revision_head in revision_heads:
            self._restore_revision_head(revision_head)

    def _restore_materialization(
        self,
        materialization: WeightMaterializationAttempt,
    ) -> None:
        if not isinstance(materialization, WeightMaterializationAttempt):
            raise ValueError("restored materialization is invalid")
        if materialization.materialization_id in self._materializations:
            raise ValueError("duplicate restored materialization ID")
        if materialization.state is WeightMaterializationAttemptState.MATERIALIZED:
            self._validate_materialized_snapshot(
                materialization.intent,
                materialization.snapshot,
            )
        self._materializations[materialization.materialization_id] = materialization

    def _restore(self, publication: WeightSnapshotPublication) -> None:
        if not isinstance(publication, WeightSnapshotPublication):
            raise ValueError("restored publication is invalid")
        if publication.publication_id in self._publications:
            raise ValueError("duplicate restored publication ID")
        ref = publication.snapshot.ref
        if (
            publication.state is not WeightSnapshotPublicationState.ABORTED
            and ref in self._publication_by_ref
        ):
            raise ValueError("duplicate restored snapshot ref")
        self._publications[publication.publication_id] = publication
        if publication.state is not WeightSnapshotPublicationState.ABORTED:
            self._publication_by_ref[ref] = publication.publication_id
        if publication.state is WeightSnapshotPublicationState.PUBLISHED:
            self._snapshots[ref] = publication.snapshot

    def _restore_revision_head(self, revision_head: WeightRevisionHead) -> None:
        if not isinstance(revision_head, WeightRevisionHead):
            raise ValueError("restored revision head is invalid")
        key = (revision_head.model_id, revision_head.revision)
        if key in self._revision_heads:
            raise ValueError("duplicate restored revision head")
        self._validate_revision_snapshot(key, revision_head.ref)
        self._revision_heads[key] = revision_head

    def prepare_publish(
        self,
        publication_id: str,
        snapshot: StoredWeightSnapshot,
    ) -> WeightSnapshotPublication:
        _require_nonempty_string(publication_id, "publication_id")
        if not isinstance(snapshot, StoredWeightSnapshot):
            raise ValueError("snapshot must be a StoredWeightSnapshot")
        with self._lock:
            current = self._publications.get(publication_id)
            if current is not None:
                if current.snapshot != snapshot:
                    raise ValueError(
                        "publication ID already identifies another snapshot"
                    )
                return current
            if snapshot.ref in self._publication_by_ref:
                raise ValueError("snapshot ref already has a publication")
            publication = WeightSnapshotPublication(
                publication_id=publication_id,
                snapshot=snapshot,
                state=WeightSnapshotPublicationState.PENDING,
            )
            self._publications[publication_id] = publication
            self._publication_by_ref[snapshot.ref] = publication_id
            return publication

    def begin_materialization(
        self,
        materialization_id: str,
        intent: WeightMaterializationIntent,
    ) -> WeightMaterializationAttempt:
        _require_nonempty_string(materialization_id, "materialization_id")
        if not isinstance(intent, WeightMaterializationIntent):
            raise ValueError("intent must be a WeightMaterializationIntent")
        with self._lock:
            current = self._materializations.get(materialization_id)
            if current is not None:
                if current.intent != intent:
                    raise ValueError(
                        "materialization ID already identifies another intent"
                    )
                if current.state is WeightMaterializationAttemptState.ABORTED:
                    raise ValueError("aborted materialization cannot be retried")
                return current
            attempt = WeightMaterializationAttempt(
                materialization_id=materialization_id,
                intent=intent,
                state=WeightMaterializationAttemptState.PREPARING,
            )
            self._materializations[materialization_id] = attempt
            return attempt

    def complete_materialization(
        self,
        materialization_id: str,
        snapshot: StoredWeightSnapshot,
    ) -> WeightMaterializationAttempt:
        if not isinstance(snapshot, StoredWeightSnapshot):
            raise ValueError("snapshot must be a StoredWeightSnapshot")
        with self._lock:
            current = self._require_materialization(materialization_id)
            if current.state is WeightMaterializationAttemptState.ABORTED:
                raise ValueError("aborted materialization cannot complete")
            if current.state is WeightMaterializationAttemptState.MATERIALIZED:
                if current.snapshot != snapshot:
                    raise ValueError(
                        "materialization already completed with another snapshot"
                    )
                return current
            self._validate_materialized_snapshot(current.intent, snapshot)
            materialized = replace(
                current,
                state=WeightMaterializationAttemptState.MATERIALIZED,
                snapshot=snapshot,
            )
            self._materializations[materialization_id] = materialized
            return materialized

    def abort_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt:
        with self._lock:
            current = self._require_materialization(materialization_id)
            if current.state is WeightMaterializationAttemptState.ABORTED:
                return current
            if current.state is WeightMaterializationAttemptState.MATERIALIZED:
                raise ValueError("materialized attempt cannot be aborted")
            aborted = replace(
                current,
                state=WeightMaterializationAttemptState.ABORTED,
                completion_ticket=None,
            )
            self._materializations[materialization_id] = aborted
            return aborted

    def set_materialization_completion_ticket(
        self,
        materialization_id: str,
        completion_ticket: str,
    ) -> WeightMaterializationAttempt:
        _require_nonempty_string(completion_ticket, "completion_ticket")
        with self._lock:
            current = self._require_materialization(materialization_id)
            if current.state is not WeightMaterializationAttemptState.PREPARING:
                raise ValueError(
                    "completion ticket requires a preparing materialization"
                )
            if (
                current.completion_ticket is not None
                and current.completion_ticket != completion_ticket
            ):
                raise ValueError(
                    "materialization already has another completion ticket"
                )
            updated = replace(
                current,
                completion_ticket=completion_ticket,
            )
            self._materializations[materialization_id] = updated
            return updated

    def clear_materialization_completion_ticket(
        self,
        materialization_id: str,
        completion_ticket: str,
    ) -> WeightMaterializationAttempt:
        _require_nonempty_string(completion_ticket, "completion_ticket")
        with self._lock:
            current = self._require_materialization(materialization_id)
            if current.completion_ticket is None:
                return current
            if current.completion_ticket != completion_ticket:
                raise ValueError("materialization has another completion ticket")
            updated = replace(current, completion_ticket=None)
            self._materializations[materialization_id] = updated
            return updated

    def get_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt | None:
        _require_nonempty_string(materialization_id, "materialization_id")
        with self._lock:
            return self._materializations.get(materialization_id)

    def recoverable_materializations(
        self,
    ) -> tuple[WeightMaterializationAttempt, ...]:
        with self._lock:
            return tuple(
                attempt
                for _, attempt in sorted(self._materializations.items())
                if attempt.recoverable
            )

    def publish(self, publication_id: str) -> WeightSnapshotPublication:
        with self._lock:
            current = self._require_publication(publication_id)
            if current.state is WeightSnapshotPublicationState.PUBLISHED:
                return current
            if current.state is WeightSnapshotPublicationState.ABORTED:
                raise ValueError("aborted publication cannot be published")
            existing = self._snapshots.get(current.snapshot.ref)
            if existing is not None and existing != current.snapshot:
                raise ValueError("published snapshot ref identity conflicts")
            published = replace(
                current,
                state=WeightSnapshotPublicationState.PUBLISHED,
            )
            self._publications[publication_id] = published
            self._snapshots[current.snapshot.ref] = current.snapshot
            return published

    def abort(self, publication_id: str) -> WeightSnapshotPublication:
        with self._lock:
            current = self._require_publication(publication_id)
            if current.state is WeightSnapshotPublicationState.ABORTED:
                return current
            if current.state is WeightSnapshotPublicationState.PUBLISHED:
                if any(
                    head.ref == current.snapshot.ref
                    for head in self._revision_heads.values()
                ):
                    raise ValueError(
                        "published publication referenced by a revision cannot be aborted"
                    )
                self._snapshots.pop(current.snapshot.ref, None)
            aborted = replace(
                current,
                state=WeightSnapshotPublicationState.ABORTED,
            )
            self._publications[publication_id] = aborted
            self._publication_by_ref.pop(current.snapshot.ref, None)
            return aborted

    def get_snapshot(
        self,
        ref: WeightStorageRef,
    ) -> StoredWeightSnapshot | None:
        if not isinstance(ref, WeightStorageRef):
            raise ValueError("ref must be a WeightStorageRef")
        with self._lock:
            return self._snapshots.get(ref)

    def get_publication(
        self,
        publication_id: str,
    ) -> WeightSnapshotPublication | None:
        _require_nonempty_string(publication_id, "publication_id")
        with self._lock:
            return self._publications.get(publication_id)

    def recoverable_publications(
        self,
    ) -> tuple[WeightSnapshotPublication, ...]:
        with self._lock:
            return tuple(
                publication
                for _, publication in sorted(self._publications.items())
                if publication.recoverable
            )

    def get_revision_head(
        self,
        model_id: str,
        revision: str,
    ) -> WeightRevisionHead | None:
        key = (
            _require_nonempty_string(model_id, "model_id"),
            _require_nonempty_string(revision, "revision"),
        )
        with self._lock:
            return self._revision_heads.get(key)

    def compare_and_set_revision(
        self,
        *,
        model_id: str,
        revision: str,
        expected: WeightRevisionHead | None,
        new_ref: WeightStorageRef,
        new_state: WeightRevisionState,
    ) -> WeightRevisionHead | None:
        key = (
            _require_nonempty_string(model_id, "model_id"),
            _require_nonempty_string(revision, "revision"),
        )
        if expected is not None and not isinstance(expected, WeightRevisionHead):
            raise ValueError("expected must be a WeightRevisionHead or None")
        if (
            expected is not None
            and (
                expected.model_id,
                expected.revision,
            )
            != key
        ):
            raise ValueError("expected revision head identity differs from CAS key")
        if not isinstance(new_ref, WeightStorageRef):
            raise ValueError("new_ref must be a WeightStorageRef")
        try:
            new_state = WeightRevisionState(new_state)
        except ValueError as error:
            raise ValueError("new revision state is invalid") from error

        with self._lock:
            current = self._revision_heads.get(key)
            if current != expected:
                return None
            self._validate_revision_snapshot(key, new_ref)
            if current is None:
                if new_state is not WeightRevisionState.READY:
                    raise ValueError("new revision head must start in READY state")
                generation = 1
            else:
                if current.ref != new_ref:
                    raise ValueError("published model revision ref is immutable")
                if new_state not in _REVISION_STATE_TRANSITIONS[current.state]:
                    raise ValueError(
                        f"invalid revision transition {current.state.value} "
                        f"to {new_state.value}"
                    )
                if current.state is new_state:
                    return current
                generation = current.generation + 1
            updated = WeightRevisionHead(
                model_id=key[0],
                revision=key[1],
                ref=new_ref,
                generation=generation,
                state=new_state,
            )
            self._revision_heads[key] = updated
            return updated

    def export_publications(
        self,
    ) -> tuple[WeightSnapshotPublication, ...]:
        with self._lock:
            return tuple(
                publication for _, publication in sorted(self._publications.items())
            )

    def export_materializations(
        self,
    ) -> tuple[WeightMaterializationAttempt, ...]:
        with self._lock:
            return tuple(
                attempt for _, attempt in sorted(self._materializations.items())
            )

    def export_revision_heads(self) -> tuple[WeightRevisionHead, ...]:
        with self._lock:
            return tuple(head for _, head in sorted(self._revision_heads.items()))

    def _validate_revision_snapshot(
        self,
        key: tuple[str, str],
        ref: WeightStorageRef,
    ) -> None:
        snapshot = self._snapshots.get(ref)
        if snapshot is None:
            raise ValueError("revision head ref must identify a published snapshot")
        identities = {
            (placement.model_id, placement.revision)
            for placement in snapshot.placements
        }
        if identities != {key}:
            raise ValueError(
                "revision head model or revision differs from published snapshot"
            )

    @staticmethod
    def _validate_materialized_snapshot(
        intent: WeightMaterializationIntent,
        snapshot: StoredWeightSnapshot,
    ) -> None:
        if (
            snapshot.ref.provider != intent.provider
            or snapshot.ref.storage_id != intent.storage_id
            or not snapshot.ref.manifest_key.startswith(
                intent.object_prefix.rstrip("/") + "/"
            )
        ):
            raise ValueError("materialized snapshot differs from its storage intent")
        model_revisions = {
            (placement.model_id, placement.revision)
            for placement in snapshot.placements
        }
        if model_revisions != {(intent.model_id, intent.revision)}:
            raise ValueError("materialized snapshot differs from its model revision")
        if weight_placement_set_digest(snapshot.placements) != intent.source_digest:
            raise ValueError("materialized snapshot differs from its source placements")
        fragments = [
            fragment
            for binding in snapshot.storage_bindings
            for fragment in binding.fragments
        ]
        if (
            sum(fragment.nbytes for fragment in fragments) != intent.total_bytes
            or len(fragments) != intent.fragment_count
        ):
            raise ValueError(
                "materialized snapshot differs from its source byte inventory"
            )
        if (
            intent.payload_digest is not None
            and weight_stored_payload_digest(
                snapshot.placements,
                snapshot.storage_bindings,
            )
            != intent.payload_digest
        ):
            raise ValueError("materialized snapshot differs from its payload identity")

    def _require_materialization(
        self,
        materialization_id: str,
    ) -> WeightMaterializationAttempt:
        _require_nonempty_string(materialization_id, "materialization_id")
        materialization = self._materializations.get(materialization_id)
        if materialization is None:
            raise KeyError(f"unknown materialization: {materialization_id}")
        return materialization

    def _require_publication(
        self,
        publication_id: str,
    ) -> WeightSnapshotPublication:
        _require_nonempty_string(publication_id, "publication_id")
        publication = self._publications.get(publication_id)
        if publication is None:
            raise KeyError(f"unknown publication: {publication_id}")
        return publication
