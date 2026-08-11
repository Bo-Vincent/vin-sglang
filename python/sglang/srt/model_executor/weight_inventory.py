from __future__ import annotations

import hashlib
import threading
from math import prod
from typing import Any, Sequence

from sglang.srt.model_executor.weight_inventory_contracts import (
    LogicalTensorView,
    WeightInventoryError,
    WeightParallelTopology,
    WeightPlacementBindingInventories,
    WeightPlacementInventory,
    WeightPlacementInventoryFragment,
    WeightRuntimeBindingInventory,
    WeightRuntimeBindingInventoryFragment,
    WeightSemanticsAdapter,
    _participant_id,
    _placement_fragment_id,
    _placement_id,
    validate_remote_weight_lineage,
    validate_weight_topology_representability,
)
from sglang.srt.model_executor.weight_inventory_physical import (
    PhysicalFragmentLookup as _PhysicalFragmentLookup,
)
from sglang.srt.model_executor.weight_inventory_physical import (
    PhysicalParameter as _PhysicalParameter,
)
from sglang.srt.model_executor.weight_inventory_physical import (
    collect_physical_parameters,
)
from sglang.srt.model_executor.weight_inventory_physical import (
    contiguous_stride as _contiguous_stride,
)
from sglang.srt.model_executor.weight_inventory_physical import (
    physical_layout_signature as _physical_layout_signature,
)
from sglang.srt.model_executor.weight_inventory_physical import (
    physical_signature as _physical_signature,
)
from sglang.srt.model_executor.weight_snapshot import (
    WeightSnapshotCoordinator,
    WeightSnapshotLeaseStatus,
)


def _validate_view(
    view: LogicalTensorView,
    physical: _PhysicalParameter,
) -> int:
    ndim = len(view.global_shape)
    if (
        not view.tensor_id
        or not view.layout_fingerprint
        or len(view.global_offset) != ndim
        or len(view.local_shape) != ndim
    ):
        raise WeightInventoryError(
            f"invalid logical view for {physical.names[0]}: {view.tensor_id}"
        )
    shard_dims = view.shard_dims
    if (
        not isinstance(shard_dims, tuple)
        or any(type(dim) is not int or not 0 <= dim < ndim for dim in shard_dims)
        or tuple(sorted(shard_dims)) != shard_dims
        or len(set(shard_dims)) != len(shard_dims)
    ):
        raise WeightInventoryError(f"invalid shard axes for {view.tensor_id}")
    if not isinstance(view.parallel_axes, tuple):
        raise WeightInventoryError(
            f"invalid logical parallel axes for {view.tensor_id}"
        )
    by_kind = {}
    split_dims = []
    for axis in view.parallel_axes:
        if axis.kind in by_kind:
            raise WeightInventoryError(
                f"duplicate logical parallel axis for {view.tensor_id}: {axis.kind}"
            )
        by_kind[axis.kind] = axis
        if axis.mode == "split":
            if axis.dim is None or axis.dim >= ndim:
                raise WeightInventoryError(
                    f"logical split axis is out of bounds for {view.tensor_id}"
                )
            split_dims.append(axis.dim)
    if tuple(sorted(split_dims)) != shard_dims:
        raise WeightInventoryError(
            f"logical parallel axes conflict with shard axes for {view.tensor_id}"
        )
    if set(by_kind) != {"dp", "tp", "pp", "ep"}:
        raise WeightInventoryError(
            f"logical view must explicitly describe dp/tp/pp/ep: {view.tensor_id}"
        )
    for offset, extent, total in zip(
        view.global_offset, view.local_shape, view.global_shape
    ):
        if offset < 0 or extent <= 0 or offset + extent > total:
            raise WeightInventoryError(f"view is out of bounds: {view.tensor_id}")
    nbytes = prod(view.local_shape) * physical.itemsize
    if (
        view.byte_offset < 0
        or view.byte_offset % physical.itemsize != 0
        or view.byte_offset + nbytes > physical.nbytes
    ):
        raise WeightInventoryError(f"view exceeds parameter storage: {view.tensor_id}")
    return nbytes


def _fragment_id_from_placement(
    *,
    instance_id: str,
    worker_id: str,
    generation: int,
    placement: WeightPlacementInventoryFragment,
) -> str:
    value = (
        f"{instance_id}|{worker_id}|{generation}|{placement.placement_fragment_id}"
    ).encode()
    return hashlib.sha256(value).hexdigest()[:24]


class WeightInventoryManager:
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
        validate_weight_topology_representability(topology)
        self._adapter = adapter
        self._topology = topology
        self._allowed_devices = frozenset(allowed_devices)
        self.coordinator = coordinator or WeightSnapshotCoordinator()
        self._last_signature: tuple | None = None
        self._last_generation: int | None = None
        self._placement_inventories: dict[
            tuple[str, str, int], WeightPlacementInventory
        ] = {}
        self._placement_layouts: dict[tuple[str, str, int], tuple] = {}
        self._placement_lookups: dict[
            tuple[str, str, int], dict[str, _PhysicalFragmentLookup]
        ] = {}
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        self.coordinator.invalidate()

    def release(self, lease_id: str) -> None:
        self.coordinator.release_snapshot(lease_id)

    def renew(self, lease_id: str, *, lease_timeout_sec: int) -> None:
        self.coordinator.renew_snapshot(lease_id, lease_timeout_sec=lease_timeout_sec)

    def has_lease(self, lease_id: str) -> bool:
        return self.coordinator.has_snapshot(lease_id)

    def list_leases(self) -> tuple[WeightSnapshotLeaseStatus, ...]:
        return self.coordinator.list_snapshot_leases()

    def snapshot_inventories(
        self,
        *,
        model_id: str,
        revision: str,
        instance_id: str,
        worker_id: str,
        endpoint: str,
        lease_timeout_sec: int | None = None,
    ) -> WeightPlacementBindingInventories:
        model_id, revision = validate_remote_weight_lineage(
            model_id=model_id,
            revision=revision,
        )
        if not all((instance_id, worker_id, endpoint)):
            raise WeightInventoryError("runtime binding identifiers must not be empty")

        lease_id, generation = self.coordinator.acquire_snapshot(
            lease_timeout_sec=lease_timeout_sec,
        )
        weight_generation = self.coordinator.weight_generation
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
                    revision=revision,
                    weight_generation=weight_generation,
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
                binding = WeightRuntimeBindingInventory(
                    model_id=model_id,
                    revision=revision,
                    placement_inventory_id=placement.inventory_id,
                    instance_id=instance_id,
                    generation=generation,
                    lease_id=lease_id,
                    participant_id=placement.participant_id,
                    fragments=fragments,
                )
            if not self.coordinator.snapshot_is_active(lease_id):
                raise WeightInventoryError(
                    "weight snapshot lease expired during inventory capture"
                )
            release_on_error = False
            return WeightPlacementBindingInventories(
                placement=placement,
                binding=binding,
            )
        finally:
            if release_on_error and self.coordinator.has_snapshot(lease_id):
                self.coordinator.release_snapshot(lease_id)

    def placement_inventory(
        self,
        *,
        model_id: str,
        revision: str,
    ) -> WeightPlacementInventory:
        """Describe only the locally committed logical weight content."""

        return self._placement_inventory_for_generation(
            model_id=model_id,
            revision=revision,
            weight_generation=self.coordinator.weight_generation,
            require_committed=True,
        )

    def target_layout_inventory(
        self,
        *,
        model_id: str,
        revision: str,
        desired_weight_generation: int,
    ) -> WeightPlacementInventory:
        """Describe a target layout before the desired content is activated."""

        return self._placement_inventory_for_generation(
            model_id=model_id,
            revision=revision,
            weight_generation=desired_weight_generation,
            require_committed=False,
        )

    def _placement_inventory_for_generation(
        self,
        *,
        model_id: str,
        revision: str,
        weight_generation: int,
        require_committed: bool,
    ) -> WeightPlacementInventory:
        model_id, revision = validate_remote_weight_lineage(
            model_id=model_id,
            revision=revision,
        )
        if type(weight_generation) is not int or weight_generation <= 0:
            raise WeightInventoryError("weight_generation must be a positive integer")

        lease_id, generation = self.coordinator.acquire_snapshot()
        try:
            if (
                require_committed
                and weight_generation != self.coordinator.weight_generation
            ):
                raise WeightInventoryError(
                    "placement weight_generation is not locally committed"
                )
            key = (model_id, revision, weight_generation)
            with self._lock:
                cached = self._placement_inventories.get(key)
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
                    weight_generation=weight_generation,
                    physical=physical,
                )
        finally:
            if self.coordinator.has_snapshot(lease_id):
                self.coordinator.release_snapshot(lease_id)

    def binding_inventory(
        self,
        *,
        placement: WeightPlacementInventory,
        instance_id: str,
        worker_id: str,
        endpoint: str,
        lease_timeout_sec: int | None = None,
    ) -> WeightRuntimeBindingInventory:
        """Bind only the locally committed source content."""

        return self._binding_inventory(
            placement=placement,
            instance_id=instance_id,
            worker_id=worker_id,
            endpoint=endpoint,
            lease_timeout_sec=lease_timeout_sec,
            require_committed=True,
        )

    def target_binding_inventory(
        self,
        *,
        placement: WeightPlacementInventory,
        instance_id: str,
        worker_id: str,
        endpoint: str,
        lease_timeout_sec: int | None = None,
    ) -> WeightRuntimeBindingInventory:
        """Bind storage for desired target content before activation."""

        return self._binding_inventory(
            placement=placement,
            instance_id=instance_id,
            worker_id=worker_id,
            endpoint=endpoint,
            lease_timeout_sec=lease_timeout_sec,
            require_committed=False,
        )

    def _binding_inventory(
        self,
        *,
        placement: WeightPlacementInventory,
        instance_id: str,
        worker_id: str,
        endpoint: str,
        lease_timeout_sec: int | None,
        require_committed: bool,
    ) -> WeightRuntimeBindingInventory:
        if not all((instance_id, worker_id, endpoint)):
            raise WeightInventoryError("runtime binding identifiers must not be empty")
        key = (
            placement.model_id,
            placement.revision,
            placement.weight_generation,
        )
        with self._lock:
            expected = self._placement_inventories.get(key)
        if expected is None or expected != placement:
            raise WeightInventoryError(
                "runtime binding placement was not produced by this manager"
            )

        lease_id, generation = self.coordinator.acquire_snapshot(
            lease_timeout_sec=lease_timeout_sec
        )
        release_on_error = True
        try:
            if (
                require_committed
                and placement.weight_generation != self.coordinator.weight_generation
            ):
                raise WeightInventoryError(
                    "runtime binding weight_generation is not locally committed"
                )
            with self._lock:
                if self._placement_inventories.get(key) != placement:
                    raise WeightInventoryError("runtime binding placement changed")
                physical = self._collect_physical_parameters()
                if self._placement_layouts.get(key) != _physical_layout_signature(
                    physical
                ):
                    raise WeightInventoryError(
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
                binding = WeightRuntimeBindingInventory(
                    model_id=placement.model_id,
                    revision=placement.revision,
                    placement_inventory_id=placement.inventory_id,
                    instance_id=instance_id,
                    generation=generation,
                    lease_id=lease_id,
                    participant_id=placement.participant_id,
                    fragments=fragments,
                )
            if not self.coordinator.snapshot_is_active(lease_id):
                raise WeightInventoryError(
                    "weight snapshot lease expired during binding capture"
                )
            release_on_error = False
            return binding
        finally:
            if release_on_error and self.coordinator.has_snapshot(lease_id):
                self.coordinator.release_snapshot(lease_id)

    def _placement_from_physical_locked(
        self,
        *,
        model_id: str,
        revision: str,
        weight_generation: int,
        physical: tuple[_PhysicalParameter, ...],
    ) -> WeightPlacementInventory:
        key = (model_id, revision, weight_generation)
        fragments, lookups = self._build_placement_fragments(physical=physical)
        placement = WeightPlacementInventory(
            model_id=model_id,
            revision=revision,
            weight_generation=weight_generation,
            inventory_id=_placement_id(
                model_id=model_id,
                revision=revision,
                weight_generation=weight_generation,
                topology=self._topology,
                fragments=fragments,
            ),
            participant_id=_participant_id(
                model_id=model_id,
                revision=revision,
                topology=self._topology,
            ),
            topology=self._topology,
            fragments=fragments,
        )
        cached = self._placement_inventories.get(key)
        if cached is not None and cached != placement:
            raise WeightInventoryError("runtime placement layout changed")
        self._placement_inventories[key] = placement
        self._placement_layouts[key] = _physical_layout_signature(physical)
        self._placement_lookups[key] = lookups
        return placement

    def _collect_physical_parameters(self) -> tuple[_PhysicalParameter, ...]:
        return collect_physical_parameters(
            model=self._model,
            allowed_devices=self._allowed_devices,
        )

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
            raise WeightInventoryError(
                "parameter storage changed outside the update coordinator"
            )
        self._last_signature = signature
        self._last_generation = generation

    def _build_placement_fragments(
        self,
        *,
        physical: tuple[_PhysicalParameter, ...],
    ) -> tuple[
        tuple[WeightPlacementInventoryFragment, ...],
        dict[str, _PhysicalFragmentLookup],
    ]:
        rank = self._topology.rank()
        result = []
        lookups = {}
        logical_keys = set()
        for item in physical:
            views = self._adapter.describe_parameter(
                names=item.names,
                parameter=item.parameter,
                topology=self._topology,
            )
            if not views:
                raise WeightInventoryError(
                    f"adapter returned no views for {item.names[0]}"
                )
            described = tuple((view, _validate_view(view, item)) for view in views)
            for view, nbytes in described:
                logical_key = (
                    view.tensor_id,
                    view.global_offset,
                    view.local_shape,
                )
                if logical_key in logical_keys:
                    raise WeightInventoryError(
                        f"duplicate logical view: {view.tensor_id}"
                    )
                logical_keys.add(logical_key)
                fragment_id = _placement_fragment_id(
                    tensor_id=view.tensor_id,
                    aliases=view.aliases,
                    global_shape=view.global_shape,
                    global_offset=view.global_offset,
                    local_shape=view.local_shape,
                    dtype=item.dtype,
                    itemsize=item.itemsize,
                    shard_dims=view.shard_dims,
                    parallel_axes=view.parallel_axes,
                    layer_id=view.layer_id,
                    expert_id=view.expert_id,
                    layout_fingerprint=view.layout_fingerprint,
                    nbytes=nbytes,
                    rank=rank,
                )
                if fragment_id in lookups:
                    raise WeightInventoryError(
                        f"duplicate placement fragment identity: {view.tensor_id}"
                    )
                result.append(
                    WeightPlacementInventoryFragment(
                        placement_fragment_id=fragment_id,
                        tensor_id=view.tensor_id,
                        aliases=view.aliases,
                        global_shape=view.global_shape,
                        global_offset=view.global_offset,
                        local_shape=view.local_shape,
                        dtype=item.dtype,
                        itemsize=item.itemsize,
                        shard_dims=view.shard_dims,
                        parallel_axes=view.parallel_axes,
                        layer_id=view.layer_id,
                        expert_id=view.expert_id,
                        layout_fingerprint=view.layout_fingerprint,
                        nbytes=nbytes,
                        rank=rank,
                    )
                )
                lookups[fragment_id] = _PhysicalFragmentLookup(
                    physical_names=item.names,
                    view_byte_offset=view.byte_offset,
                )
        result.sort(
            key=lambda item: (
                item.tensor_id,
                item.global_offset,
                item.placement_fragment_id,
            )
        )
        return tuple(result), lookups

    def _build_binding_fragments(
        self,
        *,
        placement: WeightPlacementInventory,
        physical: tuple[_PhysicalParameter, ...],
        instance_id: str,
        worker_id: str,
        endpoint: str,
        generation: int,
    ) -> tuple[WeightRuntimeBindingInventoryFragment, ...]:
        key = (
            placement.model_id,
            placement.revision,
            placement.weight_generation,
        )
        lookups = self._placement_lookups.get(key)
        if lookups is None:
            raise WeightInventoryError("placement private lookup is unavailable")
        physical_by_names = {item.names: item for item in physical}
        fragments = []
        for item in placement.fragments:
            lookup = lookups.get(item.placement_fragment_id)
            if lookup is None:
                raise WeightInventoryError(
                    f"placement private lookup is missing: {item.tensor_id}"
                )
            physical_item = physical_by_names.get(lookup.physical_names)
            if physical_item is None:
                raise WeightInventoryError(
                    f"placement parameter no longer exists: {lookup.physical_names[0]}"
                )
            if (
                physical_item.dtype != item.dtype
                or physical_item.itemsize != item.itemsize
                or lookup.view_byte_offset < 0
                or lookup.view_byte_offset + item.nbytes > physical_item.nbytes
            ):
                raise WeightInventoryError(
                    f"placement parameter storage changed: {lookup.physical_names[0]}"
                )
            fragments.append(
                WeightRuntimeBindingInventoryFragment(
                    placement_fragment_id=item.placement_fragment_id,
                    fragment_id=_fragment_id_from_placement(
                        instance_id=instance_id,
                        worker_id=worker_id,
                        generation=generation,
                        placement=item,
                    ),
                    address=physical_item.address + lookup.view_byte_offset,
                    nbytes=item.nbytes,
                    storage_offset=(
                        physical_item.storage_offset
                        + lookup.view_byte_offset // physical_item.itemsize
                    ),
                    itemsize=physical_item.itemsize,
                    local_shape=item.local_shape,
                    strides_bytes=tuple(
                        stride * physical_item.itemsize
                        for stride in _contiguous_stride(item.local_shape)
                    ),
                    storage_address=physical_item.storage_address,
                    storage_nbytes=physical_item.storage_nbytes,
                    storage_offset_bytes=(
                        physical_item.storage_offset * physical_item.itemsize
                        + lookup.view_byte_offset
                    ),
                    device=physical_item.device,
                    is_contiguous=True,
                    worker_id=worker_id,
                    endpoint=endpoint,
                )
            )
        return tuple(fragments)


class UnavailableWeightInventoryManager:
    def __init__(self, reason: str) -> None:
        self._reason = reason

    def invalidate(self) -> None:
        return None

    def placement_inventory(self, **kwargs) -> WeightPlacementInventory:
        del kwargs
        raise WeightInventoryError(self._reason)

    def target_layout_inventory(self, **kwargs) -> WeightPlacementInventory:
        del kwargs
        raise WeightInventoryError(self._reason)

    def binding_inventory(self, **kwargs) -> WeightRuntimeBindingInventory:
        del kwargs
        raise WeightInventoryError(self._reason)

    def target_binding_inventory(self, **kwargs) -> WeightRuntimeBindingInventory:
        del kwargs
        raise WeightInventoryError(self._reason)

    def snapshot_inventories(self, **kwargs) -> WeightPlacementBindingInventories:
        del kwargs
        raise WeightInventoryError(self._reason)

    def release(self, lease_id: str) -> None:
        del lease_id
        raise WeightInventoryError(self._reason)

    def has_lease(self, lease_id: str) -> bool:
        del lease_id
        return False

    def list_leases(self) -> tuple[WeightSnapshotLeaseStatus, ...]:
        return ()
