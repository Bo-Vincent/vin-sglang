from __future__ import annotations

import hashlib
import threading
import time
from math import prod
from typing import Any, Callable, Protocol, Sequence
from uuid import uuid4

import msgspec

DEFAULT_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC = 300
MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC = 30
MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC = 3600

_WEIGHT_RUNTIME_MANIFEST_FORMAT_VERSION = 2
_WEIGHT_PLACEMENT_MANIFEST_FORMAT_VERSION = 2
_WEIGHT_RUNTIME_BINDING_MANIFEST_FORMAT_VERSION = 1

_MOONCAKE_PLACEMENT_BINDING_CAPABILITY = "placement_binding_v1"
_MOONCAKE_PLACEMENT_BINDING_APIS = (
    "RuntimeBindingManifest",
    "SourcePlacementManifest",
    "TargetPlacementManifest",
    "bind_logical_transfer_plan",
    "bind_runtime_manifest",
    "placement_manifest_from_runtime_manifest",
    "plan_placement_transfer_to_local_target",
    "runtime_binding_from_runtime_manifest",
)


def local_mooncake_supports_placement_binding(
    weight_transfer_module: Any | None = None,
) -> bool:
    try:
        if weight_transfer_module is None:
            from mooncake import weight_transfer as weight_transfer_module
        supports = getattr(
            weight_transfer_module,
            "supports_weight_transfer_capability",
            None,
        )
        if (
            not callable(supports)
            or supports(_MOONCAKE_PLACEMENT_BINDING_CAPABILITY) is not True
        ):
            return False
        return all(
            callable(getattr(weight_transfer_module, name, None))
            for name in _MOONCAKE_PLACEMENT_BINDING_APIS
        )
    except Exception:
        return False


def validate_remote_instance_weight_transfer_lease_timeout(
    lease_timeout_sec: int,
) -> int:
    if isinstance(lease_timeout_sec, bool) or not isinstance(lease_timeout_sec, int):
        raise ValueError("lease_timeout_sec must be an integer")
    if not (
        MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
        <= lease_timeout_sec
        <= MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC
    ):
        raise ValueError(
            "lease_timeout_sec must be between "
            f"{MIN_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC} and "
            f"{MAX_REMOTE_INSTANCE_WEIGHT_TRANSFER_LEASE_TIMEOUT_SEC}"
        )
    return lease_timeout_sec


class WeightManifestError(RuntimeError):
    pass


def _validate_manifest_format_version(
    format_version: int,
    *,
    expected: int,
    manifest_name: str,
) -> None:
    if type(format_version) is not int or format_version != expected:
        raise WeightManifestError(
            f"unsupported {manifest_name} format_version: {format_version}"
        )


class WeightParallelRank(msgspec.Struct, frozen=True, kw_only=True):
    dp: int = 0
    tp: int = 0
    pp: int = 0
    ep: int = 0


class WeightParallelTopology(msgspec.Struct, frozen=True, kw_only=True):
    dp_rank: int = 0
    dp_size: int = 1
    tp_rank: int = 0
    tp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    ep_rank: int = 0
    ep_size: int = 1
    moe_tp_rank: int = 0
    moe_tp_size: int = 1
    attention_tp_rank: int = 0
    attention_tp_size: int = 1

    def __post_init__(self) -> None:
        ranks = (
            self.dp_rank,
            self.tp_rank,
            self.pp_rank,
            self.ep_rank,
            self.moe_tp_rank,
            self.attention_tp_rank,
        )
        sizes = (
            self.dp_size,
            self.tp_size,
            self.pp_size,
            self.ep_size,
            self.moe_tp_size,
            self.attention_tp_size,
        )
        if any(rank < 0 for rank in ranks) or any(size <= 0 for size in sizes):
            raise ValueError("parallel ranks and sizes must be positive")
        if any(rank >= size for rank, size in zip(ranks, sizes)):
            raise ValueError("parallel rank is outside its topology")

    def rank(self) -> WeightParallelRank:
        return WeightParallelRank(
            dp=self.dp_rank,
            tp=self.tp_rank,
            pp=self.pp_rank,
            ep=self.ep_rank,
        )


class LogicalTensorView(msgspec.Struct, frozen=True, kw_only=True):
    tensor_id: str
    global_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    partition_dim: int | None
    byte_offset: int
    layer_id: int | None
    expert_id: int | None
    layout_fingerprint: str
    shard_dims: tuple[int, ...] | None = None


class RuntimeWeightTensor(msgspec.Struct, frozen=True, kw_only=True):
    fragment_id: str
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
    address: int
    nbytes: int
    byte_offset: int
    stride: tuple[int, ...]
    storage_offset: int
    device: str
    is_contiguous: bool
    worker_id: str
    endpoint: str
    rank: WeightParallelRank
    lease_generation: int


class WeightRuntimeManifest(msgspec.Struct, frozen=True, kw_only=True):
    model_id: str
    revision: str
    instance_id: str
    generation: int
    lease_id: str
    tensors: tuple[RuntimeWeightTensor, ...]
    format_version: int = _WEIGHT_RUNTIME_MANIFEST_FORMAT_VERSION

    def __post_init__(self) -> None:
        _validate_manifest_format_version(
            self.format_version,
            expected=_WEIGHT_RUNTIME_MANIFEST_FORMAT_VERSION,
            manifest_name="runtime manifest",
        )


class WeightPlacementTensor(msgspec.Struct, frozen=True, kw_only=True):
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
    rank: WeightParallelRank


class WeightPlacementManifest(msgspec.Struct, frozen=True, kw_only=True):
    model_id: str
    revision: str
    placement_id: str
    tensors: tuple[WeightPlacementTensor, ...]
    format_version: int = _WEIGHT_PLACEMENT_MANIFEST_FORMAT_VERSION

    def __post_init__(self) -> None:
        _validate_manifest_format_version(
            self.format_version,
            expected=_WEIGHT_PLACEMENT_MANIFEST_FORMAT_VERSION,
            manifest_name="placement manifest",
        )
        _validate_weight_placement_manifest(self)


# Compatibility name used by the target-side session introduced in v2.
WeightTargetPlacementManifest = WeightPlacementManifest


class RuntimeWeightBinding(msgspec.Struct, frozen=True, kw_only=True):
    placement_fragment_id: str
    fragment_id: str
    address: int
    nbytes: int
    storage_offset: int
    device: str
    is_contiguous: bool
    worker_id: str
    endpoint: str


class WeightRuntimeBindingManifest(msgspec.Struct, frozen=True, kw_only=True):
    model_id: str
    revision: str
    placement_id: str
    instance_id: str
    generation: int
    lease_id: str
    fragments: tuple[RuntimeWeightBinding, ...]
    format_version: int = _WEIGHT_RUNTIME_BINDING_MANIFEST_FORMAT_VERSION

    def __post_init__(self) -> None:
        _validate_manifest_format_version(
            self.format_version,
            expected=_WEIGHT_RUNTIME_BINDING_MANIFEST_FORMAT_VERSION,
            manifest_name="runtime binding manifest",
        )
        _validate_weight_runtime_binding_manifest(self)


class WeightRuntimeManifestParts(msgspec.Struct, frozen=True, kw_only=True):
    placement: WeightPlacementManifest
    binding: WeightRuntimeBindingManifest


class WeightSemanticsAdapter(Protocol):
    def describe_parameter(
        self,
        *,
        names: tuple[str, ...],
        parameter: Any,
        topology: WeightParallelTopology,
    ) -> tuple[LogicalTensorView, ...]: ...


class _PhysicalParameter(msgspec.Struct, frozen=True, kw_only=True):
    names: tuple[str, ...]
    parameter: Any
    address: int
    nbytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: str
    itemsize: int
    device: str


def _dtype_name(dtype: Any) -> str:
    value = str(dtype)
    return value.removeprefix("torch.")


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    result = [0] * len(shape)
    value = 1
    for index in range(len(shape) - 1, -1, -1):
        result[index] = value
        value *= shape[index]
    return tuple(result)


def _storage_key(parameter: Any) -> tuple:
    return (
        int(parameter.untyped_storage().data_ptr()),
        int(parameter.storage_offset()),
        tuple(int(value) for value in parameter.shape),
        tuple(int(value) for value in parameter.stride()),
        _dtype_name(parameter.dtype),
    )


def _inspect_parameter(
    *,
    names: tuple[str, ...],
    parameter: Any,
    allowed_devices: frozenset[str],
) -> _PhysicalParameter:
    runtime_name = names[0]
    if getattr(parameter, "is_sparse", False):
        raise WeightManifestError(f"sparse parameter is unsupported: {runtime_name}")
    layout = getattr(parameter, "layout", None)
    if layout is not None and str(layout) not in ("strided", "torch.strided"):
        raise WeightManifestError(
            f"non-strided parameter is unsupported: {runtime_name}"
        )
    if not parameter.is_contiguous():
        raise WeightManifestError(
            f"non-contiguous parameter is unsupported: {runtime_name}"
        )

    device = str(parameter.device.type)
    if device not in allowed_devices:
        raise WeightManifestError(
            f"parameter device is unsupported: {runtime_name}: {device}"
        )
    shape = tuple(int(value) for value in parameter.shape)
    itemsize = int(parameter.element_size())
    nbytes = int(parameter.numel()) * itemsize
    address = int(parameter.data_ptr())
    if address <= 0 or itemsize <= 0 or nbytes <= 0:
        raise WeightManifestError(
            f"parameter has no transferable storage: {runtime_name}"
        )
    return _PhysicalParameter(
        names=names,
        parameter=parameter,
        address=address,
        nbytes=nbytes,
        shape=shape,
        stride=tuple(int(value) for value in parameter.stride()),
        storage_offset=int(parameter.storage_offset()),
        dtype=_dtype_name(parameter.dtype),
        itemsize=itemsize,
        device=device,
    )


def _view_shard_dims(view: LogicalTensorView) -> tuple[int, ...]:
    if view.shard_dims is None:
        return () if view.partition_dim is None else (view.partition_dim,)
    shard_dims = view.shard_dims
    if (
        not isinstance(shard_dims, tuple)
        or any(type(dim) is not int for dim in shard_dims)
        or tuple(sorted(shard_dims)) != shard_dims
        or len(set(shard_dims)) != len(shard_dims)
    ):
        raise WeightManifestError(f"invalid shard axes for {view.tensor_id}")
    if view.partition_dim is not None and shard_dims != (view.partition_dim,):
        raise WeightManifestError(
            f"partition axis conflicts with shard axes for {view.tensor_id}"
        )
    return shard_dims


def _validate_view(view: LogicalTensorView, physical: _PhysicalParameter) -> int:
    ndim = len(view.global_shape)
    if (
        not view.tensor_id
        or not view.layout_fingerprint
        or len(view.global_offset) != ndim
        or len(view.local_shape) != ndim
    ):
        raise WeightManifestError(
            f"invalid logical view for {physical.names[0]}: {view.tensor_id}"
        )
    if view.partition_dim is not None and (
        type(view.partition_dim) is not int or not 0 <= view.partition_dim < ndim
    ):
        raise WeightManifestError(f"invalid partition axis for {view.tensor_id}")
    shard_dims = _view_shard_dims(view)
    if any(dim < 0 or dim >= ndim for dim in shard_dims):
        raise WeightManifestError(f"invalid shard axes for {view.tensor_id}")
    for offset, extent, total in zip(
        view.global_offset, view.local_shape, view.global_shape
    ):
        if offset < 0 or extent <= 0 or offset + extent > total:
            raise WeightManifestError(f"view is out of bounds: {view.tensor_id}")
    for dim, (offset, extent, total) in enumerate(
        zip(view.global_offset, view.local_shape, view.global_shape)
    ):
        if dim not in shard_dims and (offset != 0 or extent != total):
            raise WeightManifestError(
                f"view uses a non-shard axis: {view.tensor_id}: {dim}"
            )
    nbytes = prod(view.local_shape) * physical.itemsize
    if (
        view.byte_offset < 0
        or view.byte_offset % physical.itemsize != 0
        or view.byte_offset + nbytes > physical.nbytes
    ):
        raise WeightManifestError(f"view exceeds parameter storage: {view.tensor_id}")
    return nbytes


def _fragment_id_from_placement(
    *,
    instance_id: str,
    worker_id: str,
    generation: int,
    placement: WeightPlacementTensor,
) -> str:
    value = (
        f"{instance_id}|{worker_id}|{generation}|{placement.tensor_id}|"
        f"{placement.global_offset}|{placement.local_shape}|"
        f"{placement.byte_offset}"
    ).encode()
    return hashlib.sha256(value).hexdigest()[:24]


def _placement_fragment_id(
    *,
    view: LogicalTensorView,
    names: tuple[str, ...],
    dtype: str,
    itemsize: int,
    rank: WeightParallelRank,
) -> str:
    value = (
        "weight-placement-v1",
        view.tensor_id,
        view.global_shape,
        view.global_offset,
        view.local_shape,
        view.partition_dim,
        _view_shard_dims(view),
        view.byte_offset,
        view.layer_id,
        view.expert_id,
        view.layout_fingerprint,
        names,
        dtype,
        itemsize,
        rank.dp,
        rank.tp,
        rank.pp,
        rank.ep,
    )
    return hashlib.sha256(repr(value).encode()).hexdigest()[:24]


def compute_weight_placement_id(
    tensors: Sequence[WeightPlacementTensor],
) -> str:
    tensors = tuple(tensors)
    if not tensors or not all(
        isinstance(tensor, WeightPlacementTensor) for tensor in tensors
    ):
        raise ValueError("placement tensors must not be empty")
    identity = (
        "weight-placement-v2",
        tensors,
    )
    return hashlib.sha256(msgspec.json.encode(identity)).hexdigest()[:32]


def _validate_weight_placement_manifest(
    manifest: WeightPlacementManifest,
) -> None:
    if not manifest.placement_id:
        raise WeightManifestError("placement_id must not be empty")
    if not manifest.tensors:
        raise WeightManifestError("placement tensors must not be empty")
    fragment_ids = tuple(tensor.placement_fragment_id for tensor in manifest.tensors)
    if any(not fragment_id for fragment_id in fragment_ids):
        raise WeightManifestError("placement fragment identity must not be empty")
    if len(fragment_ids) != len(set(fragment_ids)):
        raise WeightManifestError("duplicate placement fragment identity")
    if manifest.placement_id != compute_weight_placement_id(manifest.tensors):
        raise WeightManifestError(
            "placement_id does not match canonical tensor geometry"
        )


def _validate_weight_runtime_binding_manifest(
    manifest: WeightRuntimeBindingManifest,
) -> None:
    if not manifest.placement_id:
        raise WeightManifestError("placement_id must not be empty")
    if type(manifest.generation) is not int or manifest.generation <= 0:
        raise WeightManifestError("generation must be a positive integer")
    if not manifest.fragments:
        raise WeightManifestError("runtime binding fragments must not be empty")
    placement_fragment_ids = tuple(
        fragment.placement_fragment_id for fragment in manifest.fragments
    )
    runtime_fragment_ids = tuple(
        fragment.fragment_id for fragment in manifest.fragments
    )
    if any(not fragment_id for fragment_id in placement_fragment_ids):
        raise WeightManifestError("placement fragment identity must not be empty")
    if any(not fragment_id for fragment_id in runtime_fragment_ids):
        raise WeightManifestError("runtime fragment identity must not be empty")
    if len(placement_fragment_ids) != len(set(placement_fragment_ids)):
        raise WeightManifestError("duplicate placement fragment identity")
    if len(runtime_fragment_ids) != len(set(runtime_fragment_ids)):
        raise WeightManifestError("duplicate runtime fragment identity")


def _placement_id(
    *,
    tensors: tuple[WeightPlacementTensor, ...],
) -> str:
    return compute_weight_placement_id(tensors)


def _physical_signature(physical: tuple[_PhysicalParameter, ...]) -> tuple:
    return tuple(
        (
            item.names,
            item.address,
            item.nbytes,
            item.shape,
            item.stride,
            item.storage_offset,
            item.dtype,
            item.device,
        )
        for item in physical
    )


def _physical_layout_signature(physical: tuple[_PhysicalParameter, ...]) -> tuple:
    return tuple(
        (
            item.names,
            item.nbytes,
            item.shape,
            item.stride,
            item.dtype,
            item.itemsize,
        )
        for item in physical
    )


def compose_weight_runtime_manifest(
    placement: WeightPlacementManifest,
    binding: WeightRuntimeBindingManifest,
) -> WeightRuntimeManifest:
    _validate_weight_placement_manifest(placement)
    _validate_weight_runtime_binding_manifest(binding)
    if placement.model_id != binding.model_id:
        raise WeightManifestError("placement and runtime binding model_id differ")
    if placement.revision != binding.revision:
        raise WeightManifestError("placement and runtime binding revision differ")
    if placement.placement_id != binding.placement_id:
        raise WeightManifestError("placement and runtime binding IDs differ")

    binding_by_id = {}
    for fragment in binding.fragments:
        if fragment.placement_fragment_id in binding_by_id:
            raise WeightManifestError(
                "runtime binding has duplicate placement fragment"
            )
        binding_by_id[fragment.placement_fragment_id] = fragment
    placement_ids = {tensor.placement_fragment_id for tensor in placement.tensors}
    unknown = set(binding_by_id) - placement_ids
    missing = placement_ids - set(binding_by_id)
    if unknown:
        raise WeightManifestError("runtime binding has an unknown placement fragment")
    if missing:
        raise WeightManifestError("runtime binding is missing a placement fragment")

    tensors = []
    for item in placement.tensors:
        fragment = binding_by_id[item.placement_fragment_id]
        if fragment.nbytes != item.nbytes:
            raise WeightManifestError(
                "runtime binding byte size differs from placement"
            )
        tensors.append(
            RuntimeWeightTensor(
                fragment_id=fragment.fragment_id,
                tensor_id=item.tensor_id,
                runtime_name=item.runtime_name,
                aliases=item.aliases,
                global_shape=item.global_shape,
                global_offset=item.global_offset,
                local_shape=item.local_shape,
                dtype=item.dtype,
                itemsize=item.itemsize,
                partition_dim=item.partition_dim,
                shard_dims=item.shard_dims,
                layer_id=item.layer_id,
                expert_id=item.expert_id,
                layout_fingerprint=item.layout_fingerprint,
                address=fragment.address,
                nbytes=fragment.nbytes,
                byte_offset=item.byte_offset,
                stride=_contiguous_stride(item.local_shape),
                storage_offset=fragment.storage_offset,
                device=fragment.device,
                is_contiguous=fragment.is_contiguous,
                worker_id=fragment.worker_id,
                endpoint=fragment.endpoint,
                rank=item.rank,
                lease_generation=binding.generation,
            )
        )
    return WeightRuntimeManifest(
        model_id=placement.model_id,
        revision=placement.revision,
        instance_id=binding.instance_id,
        generation=binding.generation,
        lease_id=binding.lease_id,
        tensors=tuple(tensors),
    )


class _SnapshotLease(msgspec.Struct, kw_only=True):
    generation: int
    deadline: float | None
    expired: bool = False
    restore_only: bool = False


class WeightSnapshotLeaseStatus(msgspec.Struct, frozen=True, kw_only=True):
    lease_id: str
    generation: int
    deadline: float | None
    expired: bool
    restore_only: bool = False


class WeightSnapshotCoordinator:
    """Serializes in-place updates with address-bearing runtime snapshots."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        completion_fence: Callable[[], None] | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._clock = clock
        self._completion_fence = completion_fence or (lambda: None)
        self._generation = 1
        self._healthy = True
        self._poisoned = False
        self._needs_revision_commit = False
        self._last_update_success = True
        self._update_token: str | None = None
        self._update_full_restore = False
        self._update_fence_pending = False
        self._pending_full_restore_commit = False
        self._leases: dict[str, _SnapshotLease] = {}

    def _refresh_expired_leases_locked(self) -> None:
        now = self._clock()
        for lease in self._leases.values():
            if lease.deadline is not None and lease.deadline <= now:
                lease.expired = True

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def begin_update(self, *, full_restore: bool = False) -> str:
        if not isinstance(full_restore, bool):
            raise TypeError("full_restore must be a boolean")
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None:
                raise WeightManifestError("a weight update is already in progress")
            if self._leases:
                raise WeightManifestError("a weight snapshot lease is active")
            token = uuid4().hex
            expected_generation = self._generation
            self._update_token = token
            self._update_full_restore = full_restore
            self._update_fence_pending = True

        try:
            self._completion_fence()
        except BaseException:
            with self._lock:
                if token == self._update_token:
                    self._update_token = None
                    self._update_full_restore = False
                    self._update_fence_pending = False
            raise

        with self._lock:
            if (
                token != self._update_token
                or expected_generation != self._generation
                or not self._update_fence_pending
            ):
                raise WeightManifestError(
                    "weight update reservation changed during completion fence"
                )
            self._update_fence_pending = False
            return token

    def begin_update_from_snapshot(
        self,
        lease_id: str,
        expected_generation: int,
        *,
        full_restore: bool = False,
    ) -> str:
        """Atomically consume a target binding lease and reserve mutation."""

        if type(lease_id) is not str or not lease_id:
            raise ValueError("lease_id must be a non-empty string")
        if (
            isinstance(expected_generation, bool)
            or not isinstance(expected_generation, int)
            or expected_generation <= 0
        ):
            raise ValueError("expected_generation must be a positive integer")
        if not isinstance(full_restore, bool):
            raise TypeError("full_restore must be a boolean")
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None:
                raise WeightManifestError("a weight update is already in progress")
            lease = self._leases.get(lease_id)
            if lease is None or lease.expired:
                raise WeightManifestError("target binding lease is not active")
            if (
                lease.generation != expected_generation
                or expected_generation != self._generation
            ):
                raise WeightManifestError("target binding generation is stale")
            if lease.restore_only and not full_restore:
                raise WeightManifestError(
                    "restore-only target binding requires a full restore"
                )
            if len(self._leases) != 1:
                raise WeightManifestError("another weight snapshot lease is active")
            del self._leases[lease_id]
            token = uuid4().hex
            self._update_token = token
            self._update_full_restore = full_restore
            self._update_fence_pending = True

        try:
            self._completion_fence()
        except BaseException:
            with self._lock:
                if token == self._update_token:
                    self._update_token = None
                    self._update_full_restore = False
                    self._update_fence_pending = False
            raise

        with self._lock:
            if (
                token != self._update_token
                or expected_generation != self._generation
                or not self._update_fence_pending
            ):
                raise WeightManifestError(
                    "target update reservation changed during completion fence"
                )
            self._update_fence_pending = False
            return token

    def begin_target_update(
        self,
        binding: WeightRuntimeBindingManifest,
        *,
        full_restore: bool,
    ) -> str:
        if not isinstance(binding, WeightRuntimeBindingManifest):
            raise WeightManifestError("runtime binding has an invalid type")
        return self.begin_update_from_snapshot(
            binding.lease_id,
            binding.generation,
            full_restore=full_restore,
        )

    def finish_update(self, token: str, *, success: bool) -> int:
        if not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        with self._lock:
            if not token or token != self._update_token:
                raise WeightManifestError("weight update token does not match")
            if self._update_fence_pending:
                raise WeightManifestError(
                    "weight update completion fence is in progress"
                )
            self._update_fence_pending = True

        try:
            self._completion_fence()
        except BaseException:
            with self._lock:
                self._publish_update_locked(success=False)
            raise

        with self._lock:
            self._publish_update_locked(success=success)
            return self._generation

    def _publish_update_locked(self, *, success: bool) -> None:
        self._generation += 1
        self._healthy = False
        self._needs_revision_commit = True
        self._last_update_success = bool(success)
        if not success:
            self._poisoned = True
        self._pending_full_restore_commit = bool(success and self._update_full_restore)
        self._update_token = None
        self._update_full_restore = False
        self._update_fence_pending = False

    def cancel_update(self, token: str) -> None:
        """Cancel a reservation before any local weight mutation starts."""
        with self._lock:
            if not token or token != self._update_token:
                raise WeightManifestError("weight update token does not match")
            if self._update_fence_pending:
                raise WeightManifestError(
                    "weight update completion fence is in progress"
                )
            self._update_token = None
            self._update_full_restore = False

    def pending_revision_generation(self) -> int | None:
        with self._lock:
            if not self._needs_revision_commit:
                return None
            return self._generation

    def poison_global_update_failure(self, *, expected_generation: int) -> None:
        """Fail closed after an upper-layer cross-rank update transaction fails."""
        if isinstance(expected_generation, bool) or not isinstance(
            expected_generation, int
        ):
            raise TypeError("expected_generation must be an integer")
        if expected_generation <= 0:
            raise ValueError("expected_generation must be positive")
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None:
                raise WeightManifestError("a weight update is in progress")
            if expected_generation != self._generation:
                raise WeightManifestError("weight update generation does not match")
            if self._leases:
                raise WeightManifestError("a weight snapshot lease is active")
            self._healthy = False
            self._poisoned = True
            self._needs_revision_commit = True
            self._last_update_success = False
            self._pending_full_restore_commit = False

    def commit_revision(self, *, expected_generation: int | None = None) -> int:
        if expected_generation is not None:
            if isinstance(expected_generation, bool) or not isinstance(
                expected_generation, int
            ):
                raise TypeError("expected_generation must be an integer")
            if expected_generation <= 0:
                raise ValueError("expected_generation must be positive")
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None:
                raise WeightManifestError("a weight update is in progress")
            if (
                expected_generation is not None
                and expected_generation != self._generation
            ):
                raise WeightManifestError("weight update generation does not match")
            if self._leases:
                raise WeightManifestError("a weight snapshot lease is active")
            if not self._needs_revision_commit:
                return self._generation
            if not self._last_update_success:
                raise WeightManifestError(
                    "the last weight update failed; "
                    "a full successful weight restore is required"
                )
            if self._poisoned and not self._pending_full_restore_commit:
                raise WeightManifestError(
                    "the last weight update failed; "
                    "a full successful weight restore is required"
                )
            if self._pending_full_restore_commit:
                self._poisoned = False
            self._healthy = True
            self._needs_revision_commit = False
            self._pending_full_restore_commit = False
            return self._generation

    def acquire_snapshot(
        self, *, lease_timeout_sec: int | None = None
    ) -> tuple[str, int]:
        if lease_timeout_sec is not None:
            validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None:
                raise WeightManifestError("a weight update is in progress")
            if self._poisoned:
                if self._pending_full_restore_commit and self._last_update_success:
                    raise WeightManifestError(
                        "successful full weight restore requires "
                        "an explicit revision commit"
                    )
                raise WeightManifestError(
                    "the last weight update failed; "
                    "a full successful weight restore is required"
                )
            if not self._healthy:
                if self._needs_revision_commit and self._last_update_success:
                    raise WeightManifestError(
                        "updated weights require an explicit revision commit"
                    )
                raise WeightManifestError(
                    "the last weight update failed; "
                    "a full successful weight restore is required"
                )
            lease_id = uuid4().hex
            deadline = (
                None if lease_timeout_sec is None else self._clock() + lease_timeout_sec
            )
            self._leases[lease_id] = _SnapshotLease(
                generation=self._generation,
                deadline=deadline,
            )
            return lease_id, self._generation

    def acquire_target_snapshot(
        self,
        *,
        full_restore: bool,
        lease_timeout_sec: int | None = None,
    ) -> tuple[str, int]:
        """Issue a write-only target lease, including for poisoned restores."""

        if not isinstance(full_restore, bool):
            raise TypeError("full_restore must be a boolean")
        if not full_restore:
            return self.acquire_snapshot(lease_timeout_sec=lease_timeout_sec)
        if lease_timeout_sec is not None:
            validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        with self._lock:
            self._refresh_expired_leases_locked()
            if self._update_token is not None:
                raise WeightManifestError("a weight update is in progress")
            if self._leases:
                raise WeightManifestError("a weight snapshot lease is active")
            lease_id = uuid4().hex
            deadline = (
                None if lease_timeout_sec is None else self._clock() + lease_timeout_sec
            )
            self._leases[lease_id] = _SnapshotLease(
                generation=self._generation,
                deadline=deadline,
                restore_only=True,
            )
            return lease_id, self._generation

    def renew_snapshot(self, lease_id: str, *, lease_timeout_sec: int) -> None:
        validate_remote_instance_weight_transfer_lease_timeout(lease_timeout_sec)
        with self._lock:
            self._refresh_expired_leases_locked()
            lease = self._leases.get(lease_id)
            if lease is None:
                raise WeightManifestError("weight snapshot lease does not exist")
            if lease.expired:
                raise WeightManifestError(
                    "weight snapshot lease expired and requires explicit release"
                )
            lease.deadline = self._clock() + lease_timeout_sec

    def attest_snapshot(self, lease_id: str, generation: int) -> None:
        if not lease_id:
            raise WeightManifestError("weight snapshot lease ID must not be empty")
        if type(generation) is not int or generation <= 0:
            raise WeightManifestError(
                "weight snapshot generation must be a positive integer"
            )
        with self._lock:
            self._refresh_expired_leases_locked()
            lease = self._leases.get(lease_id)
            if lease is None:
                raise WeightManifestError("weight snapshot lease does not exist")
            if lease.expired:
                raise WeightManifestError("weight snapshot lease expired")
            if lease.restore_only:
                raise WeightManifestError(
                    "restore-only target binding cannot attest source weights"
                )
            if lease.generation != generation or self._generation != generation:
                raise WeightManifestError("weight snapshot generation is stale")

    def has_snapshot(self, lease_id: str) -> bool:
        with self._lock:
            self._refresh_expired_leases_locked()
            return lease_id in self._leases

    def list_snapshot_leases(self) -> tuple[WeightSnapshotLeaseStatus, ...]:
        with self._lock:
            self._refresh_expired_leases_locked()
            return tuple(
                WeightSnapshotLeaseStatus(
                    lease_id=lease_id,
                    generation=lease.generation,
                    deadline=lease.deadline,
                    expired=lease.expired,
                    restore_only=lease.restore_only,
                )
                for lease_id, lease in sorted(self._leases.items())
            )

    def release_snapshot(self, lease_id: str) -> None:
        with self._lock:
            self._refresh_expired_leases_locked()
            if lease_id not in self._leases:
                raise WeightManifestError("weight snapshot lease does not exist")
            del self._leases[lease_id]

    def invalidate(self) -> None:
        token = self.begin_update()
        generation = self.finish_update(token, success=True)
        self.commit_revision(expected_generation=generation)

    def poison_uncoordinated_mutation(self, lease_id: str) -> None:
        with self._lock:
            self._refresh_expired_leases_locked()
            if lease_id not in self._leases:
                raise WeightManifestError("weight snapshot lease does not exist")
            del self._leases[lease_id]
            self._generation += 1
            self._healthy = False
            self._poisoned = True
            self._needs_revision_commit = True
            self._last_update_success = False
            self._pending_full_restore_commit = False


class WeightRuntimeManifestManager:
    def __init__(
        self,
        *,
        model: Any,
        adapter: WeightSemanticsAdapter,
        topology: WeightParallelTopology,
        allowed_devices: Sequence[str] = ("cuda",),
        coordinator: WeightSnapshotCoordinator | None = None,
    ) -> None:
        self._model = model
        self._adapter = adapter
        self._topology = topology
        self._allowed_devices = frozenset(allowed_devices)
        self.coordinator = coordinator or WeightSnapshotCoordinator()
        self._last_signature: tuple | None = None
        self._last_generation: int | None = None
        self._placements: dict[tuple[str, str], WeightPlacementManifest] = {}
        self._placement_layouts: dict[tuple[str, str], tuple] = {}
        self._issued_bindings: dict[str, WeightRuntimeBindingManifest] = {}
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        self.coordinator.invalidate()
        with self._lock:
            self._issued_bindings.clear()

    @property
    def generation(self) -> int:
        return self.coordinator.generation

    def release(self, lease_id: str) -> None:
        self.coordinator.release_snapshot(lease_id)
        with self._lock:
            self._issued_bindings.pop(lease_id, None)

    def renew(self, lease_id: str, *, lease_timeout_sec: int) -> None:
        self.coordinator.renew_snapshot(lease_id, lease_timeout_sec=lease_timeout_sec)

    def has_lease(self, lease_id: str) -> bool:
        return self.coordinator.has_snapshot(lease_id)

    def attest_binding(self, binding: WeightRuntimeBindingManifest) -> None:
        if not isinstance(binding, WeightRuntimeBindingManifest):
            raise WeightManifestError("runtime binding has an invalid type")
        if not binding.fragments:
            raise WeightManifestError("runtime binding fragments must not be empty")
        workers = {fragment.worker_id for fragment in binding.fragments}
        endpoints = {fragment.endpoint for fragment in binding.fragments}
        if (
            len(workers) != 1
            or len(endpoints) != 1
            or not next(iter(workers))
            or not next(iter(endpoints))
        ):
            raise WeightManifestError(
                "runtime binding fragments must use one worker_id and endpoint"
            )
        self.coordinator.attest_snapshot(
            binding.lease_id,
            binding.generation,
        )
        key = (binding.model_id, binding.revision)
        with self._lock:
            placement = self._placements.get(key)
            if placement is None or placement.placement_id != binding.placement_id:
                raise WeightManifestError("runtime binding placement is not current")
            issued_binding = self._issued_bindings.get(binding.lease_id)
            if issued_binding != binding:
                raise WeightManifestError(
                    "runtime binding differs from issued runtime binding"
                )
            physical = self._collect_physical_parameters()
            if self._placement_layouts.get(key) != _physical_layout_signature(physical):
                raise WeightManifestError(
                    "runtime binding physical layout differs from placement"
                )
            self._accept_physical_snapshot(
                physical=physical,
                lease_id=binding.lease_id,
                generation=binding.generation,
            )
            expected_fragments = self._build_binding_fragments(
                placement=placement,
                physical=physical,
                instance_id=binding.instance_id,
                worker_id=issued_binding.fragments[0].worker_id,
                endpoint=issued_binding.fragments[0].endpoint,
                generation=binding.generation,
            )
            if expected_fragments != binding.fragments:
                raise WeightManifestError(
                    "runtime binding differs from current parameter storage"
                )
        self.coordinator.attest_snapshot(
            binding.lease_id,
            binding.generation,
        )

    def list_leases(self) -> tuple[WeightSnapshotLeaseStatus, ...]:
        return self.coordinator.list_snapshot_leases()

    def commit_revision(self, *, expected_generation: int | None = None) -> int:
        return self.coordinator.commit_revision(
            expected_generation=expected_generation,
        )

    def cancel_update(self, token: str) -> None:
        self.coordinator.cancel_update(token)

    def finish_update(self, token: str, *, success: bool) -> int:
        return self.coordinator.finish_update(token, success=success)

    def poison_global_update_failure(self, *, expected_generation: int) -> None:
        self.coordinator.poison_global_update_failure(
            expected_generation=expected_generation,
        )

    def begin_target_update(
        self,
        binding: WeightRuntimeBindingManifest,
        *,
        full_restore: bool,
    ) -> str:
        """Validate and atomically upgrade an issued binding into write access."""

        if not isinstance(binding, WeightRuntimeBindingManifest):
            raise WeightManifestError("runtime binding has an invalid type")
        key = (binding.model_id, binding.revision)
        try:
            with self._lock:
                placement = self._placements.get(key)
                issued_binding = self._issued_bindings.get(binding.lease_id)
                if (
                    placement is None
                    or placement.placement_id != binding.placement_id
                    or issued_binding != binding
                ):
                    raise WeightManifestError(
                        "target binding was not issued for the current placement"
                    )
                physical = self._collect_physical_parameters()
                if self._placement_layouts.get(key) != _physical_layout_signature(
                    physical
                ):
                    raise WeightManifestError(
                        "target binding physical layout differs from placement"
                    )
                expected_fragments = self._build_binding_fragments(
                    placement=placement,
                    physical=physical,
                    instance_id=binding.instance_id,
                    worker_id=binding.fragments[0].worker_id,
                    endpoint=binding.fragments[0].endpoint,
                    generation=binding.generation,
                )
                if expected_fragments != binding.fragments:
                    raise WeightManifestError(
                        "target binding differs from current parameter storage"
                    )
                token = self.coordinator.begin_update_from_snapshot(
                    binding.lease_id,
                    binding.generation,
                    full_restore=full_restore,
                )
                self._issued_bindings.pop(binding.lease_id, None)
                return token
        except BaseException:
            if not self.coordinator.has_snapshot(binding.lease_id):
                with self._lock:
                    self._issued_bindings.pop(binding.lease_id, None)
            raise

    def snapshot(
        self,
        *,
        model_id: str,
        revision: str,
        instance_id: str,
        worker_id: str,
        endpoint: str,
        lease_timeout_sec: int | None = None,
        bind_revision_to_generation: bool = False,
    ) -> WeightRuntimeManifest:
        parts = self.snapshot_parts(
            model_id=model_id,
            revision=revision,
            instance_id=instance_id,
            worker_id=worker_id,
            endpoint=endpoint,
            lease_timeout_sec=lease_timeout_sec,
            bind_revision_to_generation=bind_revision_to_generation,
        )
        try:
            return compose_weight_runtime_manifest(parts.placement, parts.binding)
        except BaseException:
            if self.coordinator.has_snapshot(parts.binding.lease_id):
                self.release(parts.binding.lease_id)
            raise

    def snapshot_parts(
        self,
        *,
        model_id: str,
        revision: str,
        instance_id: str,
        worker_id: str,
        endpoint: str,
        lease_timeout_sec: int | None = None,
        bind_revision_to_generation: bool = False,
    ) -> WeightRuntimeManifestParts:
        if not model_id or not revision:
            raise WeightManifestError("placement identifiers must not be empty")
        if not all((instance_id, worker_id, endpoint)):
            raise WeightManifestError("runtime binding identifiers must not be empty")

        lease_id, generation = self.coordinator.acquire_snapshot(
            lease_timeout_sec=lease_timeout_sec,
        )
        snapshot_revision = (
            f"{revision}@generation-{generation}"
            if bind_revision_to_generation
            else revision
        )
        release_on_error = True
        try:
            with self._lock:
                physical = self._collect_physical_parameters()
                self._accept_physical_snapshot(
                    physical=physical,
                    lease_id=lease_id,
                    generation=generation,
                )
                placement = self._placement_from_physical_locked(
                    model_id=model_id,
                    revision=snapshot_revision,
                    physical=physical,
                )
                fragments = self._build_binding_fragments(
                    placement=placement,
                    physical=physical,
                    instance_id=instance_id,
                    worker_id=worker_id,
                    endpoint=endpoint,
                    generation=generation,
                )
                binding = WeightRuntimeBindingManifest(
                    model_id=model_id,
                    revision=snapshot_revision,
                    placement_id=placement.placement_id,
                    instance_id=instance_id,
                    generation=generation,
                    lease_id=lease_id,
                    fragments=fragments,
                )
                self._issued_bindings[lease_id] = binding
            if not self.coordinator.has_snapshot(lease_id):
                raise WeightManifestError("weight snapshot lease expired")
            release_on_error = False
            return WeightRuntimeManifestParts(placement=placement, binding=binding)
        finally:
            if release_on_error:
                with self._lock:
                    self._issued_bindings.pop(lease_id, None)
                if self.coordinator.has_snapshot(lease_id):
                    self.coordinator.release_snapshot(lease_id)

    def placement(
        self,
        *,
        model_id: str,
        revision: str,
    ) -> WeightPlacementManifest:
        if not model_id or not revision:
            raise WeightManifestError("placement identifiers must not be empty")
        key = (model_id, revision)
        with self._lock:
            cached = self._placements.get(key)
        if cached is not None:
            return cached

        lease_id, generation = self.coordinator.acquire_snapshot()
        try:
            with self._lock:
                cached = self._placements.get(key)
                if cached is not None:
                    return cached
                physical = self._collect_physical_parameters()
                self._accept_physical_snapshot(
                    physical=physical,
                    lease_id=lease_id,
                    generation=generation,
                )
                return self._placement_from_physical_locked(
                    model_id=model_id,
                    revision=revision,
                    physical=physical,
                )
        finally:
            if self.coordinator.has_snapshot(lease_id):
                self.coordinator.release_snapshot(lease_id)

    def snapshot_binding(
        self,
        *,
        placement: WeightPlacementManifest,
        instance_id: str,
        worker_id: str,
        endpoint: str,
        lease_timeout_sec: int | None = None,
        full_restore: bool = False,
    ) -> WeightRuntimeBindingManifest:
        if not all((instance_id, worker_id, endpoint)):
            raise WeightManifestError("runtime binding identifiers must not be empty")
        key = (placement.model_id, placement.revision)
        with self._lock:
            expected = self._placements.get(key)
        if expected is None or expected != placement:
            raise WeightManifestError(
                "runtime binding placement was not produced by this manager"
            )

        lease_id, generation = self.coordinator.acquire_target_snapshot(
            full_restore=full_restore,
            lease_timeout_sec=lease_timeout_sec,
        )
        release_on_error = True
        try:
            with self._lock:
                if self._placements.get(key) != placement:
                    raise WeightManifestError("runtime binding placement changed")
                physical = self._collect_physical_parameters()
                if self._placement_layouts.get(key) != _physical_layout_signature(
                    physical
                ):
                    raise WeightManifestError(
                        "runtime binding physical layout differs from placement"
                    )
                self._accept_physical_snapshot(
                    physical=physical,
                    lease_id=lease_id,
                    generation=generation,
                )
                fragments = self._build_binding_fragments(
                    placement=placement,
                    physical=physical,
                    instance_id=instance_id,
                    worker_id=worker_id,
                    endpoint=endpoint,
                    generation=generation,
                )
                binding = WeightRuntimeBindingManifest(
                    model_id=placement.model_id,
                    revision=placement.revision,
                    placement_id=placement.placement_id,
                    instance_id=instance_id,
                    generation=generation,
                    lease_id=lease_id,
                    fragments=fragments,
                )
                self._issued_bindings[lease_id] = binding
            if not self.coordinator.has_snapshot(lease_id):
                raise WeightManifestError("weight snapshot lease expired")
            release_on_error = False
            return binding
        finally:
            if release_on_error:
                with self._lock:
                    self._issued_bindings.pop(lease_id, None)
                if self.coordinator.has_snapshot(lease_id):
                    self.coordinator.release_snapshot(lease_id)

    def _placement_from_physical_locked(
        self,
        *,
        model_id: str,
        revision: str,
        physical: tuple[_PhysicalParameter, ...],
    ) -> WeightPlacementManifest:
        key = (model_id, revision)
        tensors = self._build_placement_tensors(physical=physical)
        placement = WeightPlacementManifest(
            model_id=model_id,
            revision=revision,
            placement_id=_placement_id(tensors=tensors),
            tensors=tensors,
        )
        cached = self._placements.get(key)
        if cached is not None and cached != placement:
            raise WeightManifestError("runtime placement layout changed")
        self._placements[key] = placement
        self._placement_layouts[key] = _physical_layout_signature(physical)
        return placement

    def _collect_physical_parameters(self) -> tuple[_PhysicalParameter, ...]:
        grouped: dict[tuple, tuple[Any, list[str]]] = {}
        for name, parameter in self._model.named_parameters(remove_duplicate=False):
            key = _storage_key(parameter)
            if key not in grouped:
                grouped[key] = (parameter, [])
            grouped[key][1].append(name)
        result = []
        for parameter, names in grouped.values():
            result.append(
                _inspect_parameter(
                    names=tuple(sorted(names)),
                    parameter=parameter,
                    allowed_devices=self._allowed_devices,
                )
            )
        result.sort(key=lambda item: item.names)
        return tuple(result)

    def _accept_physical_snapshot(
        self,
        *,
        physical: tuple[_PhysicalParameter, ...],
        lease_id: str,
        generation: int,
    ) -> None:
        signature = _physical_signature(physical)
        if (
            self._last_signature is not None
            and signature != self._last_signature
            and generation == self._last_generation
        ):
            self.coordinator.poison_uncoordinated_mutation(lease_id)
            raise WeightManifestError(
                "parameter storage changed outside the update coordinator"
            )
        self._last_signature = signature
        self._last_generation = generation

    def _build_placement_tensors(
        self,
        *,
        physical: tuple[_PhysicalParameter, ...],
    ) -> tuple[WeightPlacementTensor, ...]:
        rank = self._topology.rank()
        result = []
        logical_keys = set()
        for item in physical:
            views = self._adapter.describe_parameter(
                names=item.names,
                parameter=item.parameter,
                topology=self._topology,
            )
            if not views:
                raise WeightManifestError(
                    f"adapter returned no views for {item.names[0]}"
                )
            for view in views:
                nbytes = _validate_view(view, item)
                logical_key = (
                    view.tensor_id,
                    view.global_offset,
                    view.local_shape,
                )
                if logical_key in logical_keys:
                    raise WeightManifestError(
                        f"duplicate logical view: {view.tensor_id}"
                    )
                logical_keys.add(logical_key)
                result.append(
                    WeightPlacementTensor(
                        placement_fragment_id=_placement_fragment_id(
                            view=view,
                            names=item.names,
                            dtype=item.dtype,
                            itemsize=item.itemsize,
                            rank=rank,
                        ),
                        tensor_id=view.tensor_id,
                        runtime_name=item.names[0],
                        aliases=item.names,
                        global_shape=view.global_shape,
                        global_offset=view.global_offset,
                        local_shape=view.local_shape,
                        dtype=item.dtype,
                        itemsize=item.itemsize,
                        partition_dim=view.partition_dim,
                        shard_dims=_view_shard_dims(view),
                        layer_id=view.layer_id,
                        expert_id=view.expert_id,
                        layout_fingerprint=view.layout_fingerprint,
                        nbytes=nbytes,
                        byte_offset=view.byte_offset,
                        rank=rank,
                    )
                )
        result.sort(
            key=lambda item: (
                item.tensor_id,
                item.global_offset,
                item.placement_fragment_id,
            )
        )
        return tuple(result)

    def _build_binding_fragments(
        self,
        *,
        placement: WeightPlacementManifest,
        physical: tuple[_PhysicalParameter, ...],
        instance_id: str,
        worker_id: str,
        endpoint: str,
        generation: int,
    ) -> tuple[RuntimeWeightBinding, ...]:
        physical_by_names = {item.names: item for item in physical}
        fragments = []
        for item in placement.tensors:
            physical_item = physical_by_names.get(item.aliases)
            if physical_item is None:
                raise WeightManifestError(
                    f"placement parameter no longer exists: {item.runtime_name}"
                )
            if (
                physical_item.dtype != item.dtype
                or physical_item.itemsize != item.itemsize
                or item.byte_offset < 0
                or item.byte_offset + item.nbytes > physical_item.nbytes
            ):
                raise WeightManifestError(
                    f"placement parameter storage changed: {item.runtime_name}"
                )
            fragments.append(
                RuntimeWeightBinding(
                    placement_fragment_id=item.placement_fragment_id,
                    fragment_id=_fragment_id_from_placement(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        generation=generation,
                        placement=item,
                    ),
                    address=physical_item.address + item.byte_offset,
                    nbytes=item.nbytes,
                    storage_offset=(
                        physical_item.storage_offset
                        + item.byte_offset // physical_item.itemsize
                    ),
                    device=physical_item.device,
                    is_contiguous=True,
                    worker_id=worker_id,
                    endpoint=endpoint,
                )
            )
        return tuple(fragments)


class UnavailableWeightRuntimeManifestManager:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    def invalidate(self) -> None:
        return None

    def snapshot(self, **kwargs) -> WeightRuntimeManifest:
        del kwargs
        raise WeightManifestError(self._reason)

    def placement(self, **kwargs) -> WeightPlacementManifest:
        del kwargs
        raise WeightManifestError(self._reason)

    def snapshot_binding(self, **kwargs) -> WeightRuntimeBindingManifest:
        del kwargs
        raise WeightManifestError(self._reason)

    def snapshot_parts(self, **kwargs) -> WeightRuntimeManifestParts:
        del kwargs
        raise WeightManifestError(self._reason)

    def release(self, lease_id: str) -> None:
        del lease_id
        raise WeightManifestError(self._reason)

    def has_lease(self, lease_id: str) -> bool:
        del lease_id
        return False

    def list_leases(self) -> tuple[WeightSnapshotLeaseStatus, ...]:
        return ()

    def commit_revision(self) -> int:
        raise WeightManifestError(self._reason)


def _topology_from_sglang(
    *, parallel_state: Any, parallel: Any
) -> WeightParallelTopology:
    if parallel_state.dp_rank is not None:
        dp_rank = parallel_state.dp_rank
        dp_size = parallel_state.dp_size
    else:
        dp_rank = parallel_state.moe_dp_rank or 0
        dp_size = parallel_state.moe_dp_size
    return WeightParallelTopology(
        dp_rank=dp_rank,
        dp_size=dp_size,
        tp_rank=parallel_state.tp_rank,
        tp_size=parallel_state.tp_size,
        pp_rank=parallel_state.pp_rank,
        pp_size=parallel_state.pp_size,
        ep_rank=parallel_state.moe_ep_rank,
        ep_size=parallel_state.moe_ep_size,
        moe_tp_rank=parallel.moe_tp_rank,
        moe_tp_size=parallel.moe_tp_size,
        attention_tp_rank=parallel_state.attn_tp_rank,
        attention_tp_size=parallel_state.attn_tp_size,
    )


def create_sglang_weight_runtime_manifest_manager(
    *,
    model: Any,
    config: Any,
    parallel_state: Any,
    parallel: Any,
    allowed_devices: Sequence[str] = ("cuda",),
    quantization: str | None = None,
    lora_enabled: bool = False,
    is_multimodal: bool = False,
    dynamic_expert_placement: bool = False,
    moe_runner_backend: str | None = None,
    fp8_gemm_backend: str | None = None,
    dp_attention_enabled: bool = False,
    coordinator: WeightSnapshotCoordinator | None = None,
):
    return create_weight_runtime_manifest_manager(
        model=model,
        config=config,
        topology=_topology_from_sglang(
            parallel_state=parallel_state,
            parallel=parallel,
        ),
        allowed_devices=allowed_devices,
        quantization=quantization,
        lora_enabled=lora_enabled,
        is_multimodal=is_multimodal,
        dynamic_expert_placement=dynamic_expert_placement,
        moe_runner_backend=moe_runner_backend,
        fp8_gemm_backend=fp8_gemm_backend,
        dp_attention_enabled=dp_attention_enabled,
        coordinator=coordinator,
    )


def create_weight_runtime_manifest_manager(
    *,
    model: Any,
    config: Any,
    topology: WeightParallelTopology,
    allowed_devices: Sequence[str] = ("cuda",),
    quantization: str | None = None,
    lora_enabled: bool = False,
    is_multimodal: bool = False,
    dynamic_expert_placement: bool = False,
    moe_runner_backend: str | None = None,
    fp8_gemm_backend: str | None = None,
    dp_attention_enabled: bool = False,
    coordinator: WeightSnapshotCoordinator | None = None,
):
    if quantization not in (None, "fp8"):
        return UnavailableWeightRuntimeManifestManager(
            f"quantized weight manifests are unsupported: {quantization}"
        )
    if lora_enabled:
        return UnavailableWeightRuntimeManifestManager(
            "LoRA weight manifests are unsupported"
        )
    if dp_attention_enabled:
        return UnavailableWeightRuntimeManifestManager(
            "DP attention weight manifests are unsupported"
        )
    model_type = getattr(config, "model_type", None)
    text_model_types = (
        "qwen3",
        "qwen3_moe",
        "qwen3_5_text",
        "qwen3_5_moe_text",
        "qwen3_next",
    )
    multimodal_model_types = ("qwen3_5", "qwen3_5_moe")
    if is_multimodal and model_type not in multimodal_model_types:
        return UnavailableWeightRuntimeManifestManager(
            f"unsupported multimodal model type for weight manifests: {model_type}"
        )
    if not is_multimodal and model_type not in text_model_types:
        return UnavailableWeightRuntimeManifestManager(
            f"unsupported model type for weight manifests: {model_type}"
        )
    if model_type == "qwen3_next" and moe_runner_backend != "triton":
        return UnavailableWeightRuntimeManifestManager(
            "Qwen3-Next weight manifests require the canonical triton MoE "
            f"runner backend; got {moe_runner_backend!r}"
        )
    if model_type == "qwen3_next" and topology.pp_size != 1:
        return UnavailableWeightRuntimeManifestManager(
            "Qwen3-Next runtime requires PP=1; pipeline parallelism is unsupported"
        )

    from sglang.srt.model_executor.weight_semantics.qwen3_5 import (
        Qwen35MultimodalWeightSemanticsAdapter,
        Qwen35WeightSemanticsAdapter,
    )
    from sglang.srt.model_executor.weight_semantics.qwen3 import (
        Qwen3WeightSemanticsAdapter,
    )
    from sglang.srt.model_executor.weight_semantics.qwen3_next import (
        Qwen3NextWeightSemanticsAdapter,
    )

    up_first_w13_parameters = set()
    num_fused_shared_experts = int(getattr(model, "num_fused_shared_experts", 0))
    modules = getattr(model, "modules", None)
    if modules is not None:
        for module in modules():
            num_fused_shared_experts = max(
                num_fused_shared_experts,
                int(getattr(module, "num_fused_shared_experts", 0)),
            )
            parameter = getattr(module, "w13_weight", None)
            if parameter is None:
                continue
            quant_method = getattr(module, "quant_method", None)
            if bool(getattr(module, "use_flashinfer_trtllm_moe", False)) or bool(
                getattr(quant_method, "load_up_proj_weight_first", False)
            ):
                up_first_w13_parameters.add(id(parameter))
    if (
        model_type
        in (
            "qwen3_5_text",
            "qwen3_5_moe_text",
            "qwen3_5",
            "qwen3_5_moe",
        )
        and num_fused_shared_experts
    ):
        return UnavailableWeightRuntimeManifestManager(
            "Qwen3.5 fused shared-expert runtime allocations are unsupported"
        )

    if is_multimodal:
        text_config = getattr(config, "text_config", None)
        vision_config = getattr(config, "vision_config", None)
        if text_config is None or vision_config is None:
            return UnavailableWeightRuntimeManifestManager(
                "Qwen3.5 multimodal config is missing text_config or vision_config"
            )
        adapter = Qwen35MultimodalWeightSemanticsAdapter(
            text_config=text_config,
            vision_config=vision_config,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameters,
        )
    elif model_type == "qwen3_next":
        adapter = Qwen3NextWeightSemanticsAdapter(
            config=config,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameters,
            num_fused_shared_experts=num_fused_shared_experts,
        )
    elif model_type in ("qwen3", "qwen3_moe"):
        adapter = Qwen3WeightSemanticsAdapter(
            config=config,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameters,
        )
    else:
        adapter = Qwen35WeightSemanticsAdapter(
            config=config,
            dynamic_expert_placement=dynamic_expert_placement,
            up_first_w13_parameter_ids=up_first_w13_parameters,
        )

    if quantization == "fp8":
        from sglang.srt.model_executor.weight_semantics.fp8_block import (
            create_serialized_block_fp8_adapter,
        )

        try:
            adapter = create_serialized_block_fp8_adapter(
                model=model,
                delegate=adapter,
                fp8_gemm_backend=fp8_gemm_backend,
                moe_runner_backend=moe_runner_backend,
            )
        except WeightManifestError as error:
            return UnavailableWeightRuntimeManifestManager(str(error))

    return WeightRuntimeManifestManager(
        model=model,
        adapter=adapter,
        topology=topology,
        allowed_devices=allowed_devices,
        coordinator=coordinator,
    )
