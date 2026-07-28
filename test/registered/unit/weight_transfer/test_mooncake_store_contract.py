from __future__ import annotations

import ctypes
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import product
from math import prod

import pytest

# ruff: noqa: E402

try:
    import mooncake.weight_transfer as mooncake_backend
    from mooncake.weight_transfer import WeightManifest, WeightStore, WeightStoreError
    from mooncake.weight_transfer import planner as mooncake_planner_module
    from mooncake.weight_transfer import store as mooncake_store_module
except ModuleNotFoundError as error:
    if error.name not in {"mooncake", "mooncake.weight_transfer"}:
        raise
    if __name__ == "__main__":
        print("SKIPPED: mooncake.weight_transfer is not installed")
        raise SystemExit(0)
    pytest.skip(
        "mooncake.weight_transfer is not installed",
        allow_module_level=True,
    )

from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    WeightSnapshotCoordinator,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer.api import (
    execute_weight_load,
    load_weight_snapshot,
    materialize_weight_snapshot,
    materialize_weights,
    prepare_weight_load_from_plan,
    prepare_weight_materialization,
)
from sglang.srt.weight_transfer.contracts import RuntimeWeightLocation
from sglang.srt.weight_transfer.mooncake_store import (
    MooncakeWeightStoreProvider,
)
from sglang.srt.weight_transfer.planner import (
    plan_weight_transfer_to_local_target,
)
from sglang.srt.weight_transfer.provider import (
    WeightPayloadIdentity,
    WeightStorageDestination,
    WeightTargetLoadMode,
    WeightTargetLoadSession,
    WeightTargetLoadState,
    WeightTransferCompletionUnknownError,
    WeightTransferError,
)
from sglang.srt.weight_transfer.storage import InMemoryWeightStorageCatalog
from sglang.srt.weight_transfer.storage_file import FileWeightStorageCatalog
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_GLOBAL_SHAPE = (4, 6, 8)
_ITEMSIZE = 2
_MODEL_ID = "contract-model"
_REVISION = "step-1"
_STORAGE_ID = f"weights/default/{_MODEL_ID}/{_REVISION}"


class _AllowAllAttestor:
    def attest(self, _request) -> None:
        pass


_ALLOW_ALL_ATTESTOR = _AllowAllAttestor()


@dataclass
class _MemoryReplicateConfig:
    group_ids: list[str]
    data_type: str
    with_hard_pin: bool


class _InMemoryStore:
    def __init__(
        self,
        *,
        fail_payload_write: int | None = None,
        fail_manifest_put: bool = False,
        fail_decision_put: bool = False,
        fail_range_get: int | None = None,
    ) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_payload_write = fail_payload_write
        self.fail_manifest_put = fail_manifest_put
        self.fail_decision_put = fail_decision_put
        self.fail_range_get = fail_range_get
        self.payload_write_attempts = 0
        self.payload_addresses: list[int] = []
        self.payload_keys: list[str] = []
        self.removed_keys: list[str] = []
        self.get_keys: list[str] = []
        self.range_get_calls = 0
        self.registered: dict[int, int] = {}
        self.register_calls: list[tuple[int, int]] = []
        self.unregister_calls: list[int] = []

    def register_buffer(self, address: int, nbytes: int) -> int:
        assert address not in self.registered
        self.registered[address] = nbytes
        self.register_calls.append((address, nbytes))
        return 0

    def unregister_buffer(self, address: int) -> int:
        self.unregister_calls.append(address)
        del self.registered[address]
        return 0

    def batch_put_from(
        self,
        keys: list[str],
        addresses: list[int],
        sizes: list[int],
        config: _MemoryReplicateConfig,
    ) -> list[int]:
        results = []
        for key, address, nbytes, group_id in zip(
            keys,
            addresses,
            sizes,
            config.group_ids,
            strict=True,
        ):
            del group_id
            self.payload_write_attempts += 1
            self.payload_addresses.append(address)
            self.payload_keys.append(key)
            if self.payload_write_attempts == self.fail_payload_write:
                results.append(-1)
                continue
            assert self.registered[address] >= nbytes
            self.objects[key] = ctypes.string_at(address, nbytes)
            results.append(0)
        return results

    def put(
        self,
        key: str,
        value: bytes,
        config: _MemoryReplicateConfig,
    ) -> int:
        del config
        if self.fail_manifest_put and key.endswith("/manifest"):
            return -1
        if self.fail_decision_put and key.endswith("/decision"):
            return -1
        if key not in self.objects:
            self.objects[key] = bytes(value)
        return 0

    def get(self, key: str) -> bytes:
        self.get_keys.append(key)
        return self.objects[key]

    def is_exist(self, key: str) -> int:
        return int(key in self.objects)

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        return [self.is_exist(key) for key in keys]

    def remove(self, key: str, *, force: bool) -> int:
        assert force
        self.removed_keys.append(key)
        self.objects.pop(key, None)
        return 0

    def get_into_ranges(
        self,
        addresses: list[int],
        all_keys: list[list[str]],
        all_target_offsets: list[list[list[int]]],
        all_source_offsets: list[list[list[int]]],
        all_sizes: list[list[list[int]]],
    ) -> list[list[list[int]]]:
        self.range_get_calls += 1
        if self.range_get_calls == self.fail_range_get:
            raise RuntimeError("range read failed")
        results = []
        for (
            address,
            keys,
            target_groups,
            source_groups,
            size_groups,
        ) in zip(
            addresses,
            all_keys,
            all_target_offsets,
            all_source_offsets,
            all_sizes,
            strict=True,
        ):
            target_nbytes = self.registered[address]
            buffer_results = []
            for key, target_offsets, source_offsets, sizes in zip(
                keys,
                target_groups,
                source_groups,
                size_groups,
                strict=True,
            ):
                payload = self.objects[key]
                range_results = []
                for target_offset, source_offset, nbytes in zip(
                    target_offsets,
                    source_offsets,
                    sizes,
                    strict=True,
                ):
                    assert 0 <= source_offset
                    assert source_offset + nbytes <= len(payload)
                    assert 0 <= target_offset
                    assert target_offset + nbytes <= target_nbytes
                    ctypes.memmove(
                        address + target_offset,
                        payload[source_offset : source_offset + nbytes],
                        nbytes,
                    )
                    range_results.append(nbytes)
                buffer_results.append(range_results)
            results.append(buffer_results)
        return results


def _config_factory(
    group_ids: list[str] | tuple[str, ...],
    record_type: str,
) -> _MemoryReplicateConfig:
    return _MemoryReplicateConfig(
        group_ids=list(group_ids),
        data_type=record_type,
        with_hard_pin=True,
    )


def _box_payload(
    global_offset: tuple[int, ...],
    local_shape: tuple[int, ...],
) -> bytes:
    payload = bytearray()
    for local_coordinate in product(*(range(extent) for extent in local_shape)):
        global_coordinate = tuple(
            begin + local
            for begin, local in zip(
                global_offset,
                local_coordinate,
                strict=True,
            )
        )
        value = (
            global_coordinate[0] * 1_000
            + global_coordinate[1] * 100
            + global_coordinate[2]
        )
        payload.extend(value.to_bytes(_ITEMSIZE, byteorder="little"))
    return bytes(payload)


def _runtime_world(
    side: str,
    *,
    shard_dim: int,
    shard_count: int,
    dp_count: int,
    source_payload: bool,
) -> tuple[
    tuple[WeightPlacementManifest, ...],
    tuple[WeightRuntimeBindingManifest, ...],
    tuple[ctypes.Array, ...],
]:
    placements = []
    bindings = []
    owners = []
    for dp_rank in range(dp_count):
        for shard_rank in range(shard_count):
            local_shape = list(_GLOBAL_SHAPE)
            local_shape[shard_dim] //= shard_count
            global_offset = [0] * len(_GLOBAL_SHAPE)
            global_offset[shard_dim] = shard_rank * local_shape[shard_dim]
            local_shape_tuple = tuple(local_shape)
            global_offset_tuple = tuple(global_offset)
            nbytes = prod(local_shape_tuple) * _ITEMSIZE
            if source_payload:
                payload = _box_payload(
                    global_offset_tuple,
                    local_shape_tuple,
                )
                owner = ctypes.create_string_buffer(payload, nbytes)
            else:
                owner = ctypes.create_string_buffer(nbytes)
                ctypes.memset(ctypes.addressof(owner), 0xA5, nbytes)
            owners.append(owner)

            placement_label = f"{side}:d{dp_rank}:s{shard_rank}"
            placement_fragment_id = f"{placement_label}:placement-fragment"
            runtime_fragment_id = f"{placement_label}:runtime-fragment"
            tensor = WeightPlacementTensor(
                placement_fragment_id=placement_fragment_id,
                tensor_id="layers.0.experts.w1",
                runtime_name="layers.0.experts.w1",
                aliases=("layers.0.experts.w1",),
                global_shape=_GLOBAL_SHAPE,
                global_offset=global_offset_tuple,
                local_shape=local_shape_tuple,
                dtype="uint16",
                itemsize=_ITEMSIZE,
                partition_dim=None,
                shard_dims=(shard_dim,),
                layer_id=0,
                expert_id=None,
                layout_fingerprint="logical-contiguous:uint16:v2",
                nbytes=nbytes,
                byte_offset=0,
                rank=WeightParallelRank(
                    dp=dp_rank,
                    tp=shard_rank,
                ),
            )
            tensors = (tensor,)
            placement = WeightPlacementManifest(
                model_id=_MODEL_ID,
                revision=_REVISION,
                placement_id=compute_weight_placement_id(tuple(tensors)),
                tensors=tuple(tensors),
            )
            placements.append(placement)
            bindings.append(
                WeightRuntimeBindingManifest(
                    model_id=_MODEL_ID,
                    revision=_REVISION,
                    placement_id=placement.placement_id,
                    instance_id=f"{placement_label}:instance",
                    generation=1,
                    lease_id=f"{placement_label}:lease",
                    fragments=(
                        RuntimeWeightBinding(
                            placement_fragment_id=placement_fragment_id,
                            fragment_id=runtime_fragment_id,
                            address=ctypes.addressof(owner),
                            nbytes=nbytes,
                            storage_offset=0,
                            device="cpu",
                            is_contiguous=True,
                            worker_id=placement_label,
                            endpoint=f"{placement_label}:12345",
                        ),
                    ),
                )
            )
    return tuple(placements), tuple(bindings), tuple(owners)


def _restart_runtime_bindings(
    bindings: tuple[WeightRuntimeBindingManifest, ...],
) -> tuple[
    tuple[WeightRuntimeBindingManifest, ...],
    tuple[ctypes.Array, ...],
]:
    restarted = []
    owners = []
    for index, binding in enumerate(bindings):
        fragment = binding.fragments[0]
        payload = ctypes.string_at(
            fragment.address + fragment.storage_offset,
            fragment.nbytes,
        )
        owner = ctypes.create_string_buffer(payload, fragment.nbytes)
        owners.append(owner)
        restarted.append(
            WeightRuntimeBindingManifest(
                model_id=binding.model_id,
                revision=binding.revision,
                placement_id=binding.placement_id,
                instance_id=f"restarted:{index}:instance",
                generation=binding.generation + 1,
                lease_id=f"restarted:{index}:lease",
                fragments=(
                    RuntimeWeightBinding(
                        placement_fragment_id=fragment.placement_fragment_id,
                        fragment_id=f"restarted:{fragment.fragment_id}",
                        address=ctypes.addressof(owner),
                        nbytes=fragment.nbytes,
                        storage_offset=fragment.storage_offset,
                        device=fragment.device,
                        is_contiguous=fragment.is_contiguous,
                        worker_id=f"restarted:{fragment.worker_id}",
                        endpoint=f"restarted:{index}:23456",
                    ),
                ),
            )
        )
    return tuple(restarted), tuple(owners)


def _bind_target_snapshot(
    binding: WeightRuntimeBindingManifest,
    coordinator: WeightSnapshotCoordinator,
) -> WeightRuntimeBindingManifest:
    lease_id, generation = coordinator.acquire_snapshot()
    return WeightRuntimeBindingManifest(
        model_id=binding.model_id,
        revision=binding.revision,
        placement_id=binding.placement_id,
        instance_id=binding.instance_id,
        generation=generation,
        lease_id=lease_id,
        fragments=binding.fragments,
    )


def _weight_store(store: _InMemoryStore) -> WeightStore:
    return WeightStore(
        store,
        config_factory=_config_factory,
        max_ranges_per_request=3,
    )


def _runtime_payload_checksum(location: RuntimeWeightLocation) -> str:
    payload = ctypes.string_at(
        location.address + location.storage_offset,
        location.nbytes,
    )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _payload_identity(
    placements: tuple[WeightPlacementManifest, ...],
    bindings: tuple[WeightRuntimeBindingManifest, ...],
) -> WeightPayloadIdentity:
    binding_by_id = {binding.placement_id: binding for binding in bindings}
    checksums = {}
    for placement in placements:
        binding = binding_by_id[placement.placement_id]
        fragment_by_id = {
            fragment.placement_fragment_id: fragment for fragment in binding.fragments
        }
        for tensor in placement.tensors:
            fragment = fragment_by_id[tensor.placement_fragment_id]
            payload = ctypes.string_at(
                fragment.address + fragment.storage_offset,
                fragment.nbytes,
            )
            checksums[tensor.placement_fragment_id] = (
                f"sha256:{hashlib.sha256(payload).hexdigest()}"
            )
    return WeightPayloadIdentity.create(placements, checksums)


def _upload_plan_signature(plan):
    return (
        plan.session_group_id,
        plan.control_key,
        frozenset(operation.target.object_key for operation in plan.operations),
    )


def _leave_recoverable_payload(
    store: _InMemoryStore,
    placements: tuple[WeightPlacementManifest, ...],
    bindings: tuple[WeightRuntimeBindingManifest, ...],
    *,
    operation_id: str,
):
    request = prepare_weight_materialization(
        source_placements=placements,
        source_bindings=bindings,
        destination=WeightStorageDestination(
            provider="mooncake-store",
            storage_id=_STORAGE_ID,
            object_prefix=_STORAGE_ID,
        ),
        payload_identity=_payload_identity(placements, bindings),
        operation_id=operation_id,
    )
    weight_store = _weight_store(store)
    provider = MooncakeWeightStoreProvider(
        weight_store,
        payload_checksum_verifier=_runtime_payload_checksum,
    )
    prepared = provider.prepare(request)
    ticket = provider.materialization_recovery_ticket(prepared)
    assert ticket is not None
    submission = provider.submit(prepared)
    for _, runtime_manifest in prepared.runtime_manifests:
        submission.receipts.extend(
            weight_store.upload(
                prepared.upload_plan,
                runtime_manifest,
                pre_registered=False,
            )
        )

    plan_signature = _upload_plan_signature(prepared.upload_plan)
    _session_group_id, _control_key, payload_keys = plan_signature
    assert {receipt.object_key for receipt in submission.receipts} == payload_keys
    assert payload_keys <= set(store.objects)
    assert not any(key.endswith("/decision") for key in store.objects)
    assert prepared.upload_plan.manifest.manifest_key not in store.objects
    return request, ticket, plan_signature


def _publish_conflicting_winner(
    store: _InMemoryStore,
    loser_payload_keys: frozenset[str],
) -> tuple[str, bytes, frozenset[str]]:
    sources, bindings, owners = _runtime_world(
        "winner",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    for owner in owners:
        ctypes.memset(ctypes.addressof(owner), 0x5A, ctypes.sizeof(owner))
    receipt = materialize_weights(
        source_placements=sources,
        source_bindings=bindings,
        destination=WeightStorageDestination(
            provider="mooncake-store",
            storage_id=_STORAGE_ID,
            object_prefix=_STORAGE_ID,
        ),
        provider=MooncakeWeightStoreProvider(
            _weight_store(store),
            payload_checksum_verifier=_runtime_payload_checksum,
        ),
        payload_identity=_payload_identity(sources, bindings),
        attestor=_ALLOW_ALL_ATTESTOR,
    )
    winner_payload_keys = frozenset(
        key
        for key in store.objects
        if "/payload/" in key and key not in loser_payload_keys
    )
    assert winner_payload_keys
    return (
        receipt.manifest_key,
        bytes(store.objects[receipt.manifest_key]),
        winner_payload_keys,
    )


def test_mooncake_wrapper_round_trip_preserves_nd_cross_dim_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _InMemoryStore()
    weight_store = _weight_store(store)
    provider = MooncakeWeightStoreProvider(
        weight_store,
        payload_checksum_verifier=_runtime_payload_checksum,
    )
    sources, source_bindings, _source_owners = _runtime_world(
        "source",
        shard_dim=0,
        shard_count=4,
        dp_count=2,
        source_payload=True,
    )

    materialized = materialize_weights(
        source_placements=sources,
        source_bindings=source_bindings,
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id=_STORAGE_ID,
            object_prefix=_STORAGE_ID,
        ),
        provider=provider,
        payload_identity=_payload_identity(sources, source_bindings),
        attestor=_ALLOW_ALL_ATTESTOR,
    )

    manifest_key = f"{_STORAGE_ID}/manifest"
    persisted = WeightManifest.from_json(store.objects[manifest_key])
    selected_source_addresses = {
        binding.fragments[0].address
        for placement, binding in zip(
            sources,
            source_bindings,
            strict=True,
        )
        if placement.tensors[0].rank.dp == 0
    }
    assert len(materialized.stored_placements) == 4
    assert len(materialized.storage_bindings) == 4
    assert {
        placement.tensors[0].rank.dp for placement in materialized.stored_placements
    } == {0}
    assert len(persisted.fragments) == 4
    assert persisted.format_version == 2
    expected_checksums = {
        fragment.placement_fragment_id: fragment.checksum
        for fragment in _payload_identity(sources, source_bindings)
        .select(materialized.stored_placements)
        .fragments
    }
    assert {
        fragment.placement_fragment_id: fragment.checksum
        for binding in materialized.storage_bindings
        for fragment in binding.fragments
    } == expected_checksums
    assert WeightManifest.from_json(store.objects[persisted.manifest_key]) == persisted
    assert store.payload_write_attempts == 4
    assert set(store.payload_addresses) == selected_source_addresses
    assert len(set(store.payload_keys)) == 4
    assert Counter(address for address, _ in store.register_calls) == Counter(
        store.unregister_calls
    )
    assert store.registered == {}

    replan_calls = 0

    def forbid_mooncake_replanning(*args, **kwargs):
        nonlocal replan_calls
        replan_calls += 1
        raise AssertionError("provider must execute the SGLang plan")

    load_weight_store = _weight_store(store)
    load_provider = MooncakeWeightStoreProvider(load_weight_store)
    monkeypatch.setattr(
        load_weight_store,
        "plan_load",
        forbid_mooncake_replanning,
    )
    for module in (
        mooncake_backend,
        mooncake_planner_module,
        mooncake_store_module,
    ):
        monkeypatch.setattr(
            module,
            "plan_stored_transfer",
            forbid_mooncake_replanning,
        )
    for module in (mooncake_backend, mooncake_planner_module):
        monkeypatch.setattr(
            module,
            "plan_stored_transfer_to_target_placements",
            forbid_mooncake_replanning,
        )
    targets, target_bindings, target_owners = _runtime_world(
        "target",
        shard_dim=2,
        shard_count=2,
        dp_count=1,
        source_payload=False,
    )
    manifest_reads_before_load = store.get_keys.count(manifest_key)
    for target, target_binding in zip(
        targets,
        target_bindings,
        strict=True,
    ):
        logical_plan = plan_weight_transfer_to_local_target(
            materialized.stored_placements,
            target,
        )
        request = prepare_weight_load_from_plan(
            logical_plan,
            source_bindings=materialized.storage_bindings,
            target_bindings=(target_binding,),
        )
        receipt = execute_weight_load(
            request,
            provider=load_provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            attestor=_ALLOW_ALL_ATTESTOR,
        )
        assert receipt.plan_digest == request.plan.digest
        assert receipt.total_bytes == target.tensors[0].nbytes

    assert replan_calls == 0
    assert store.range_get_calls > 0
    assert store.get_keys.count(manifest_key) - manifest_reads_before_load == len(
        targets
    )
    assert Counter(address for address, _ in store.register_calls) == Counter(
        store.unregister_calls
    )
    assert store.registered == {}
    for target, owner in zip(targets, target_owners, strict=True):
        tensor = target.tensors[0]
        assert ctypes.string_at(
            ctypes.addressof(owner),
            tensor.nbytes,
        ) == _box_payload(tensor.global_offset, tensor.local_shape)


def test_mooncake_wrapper_rejects_payload_mismatch_before_upload() -> None:
    store = _InMemoryStore()
    sources, source_bindings, source_owners = _runtime_world(
        "source",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    declared_identity = _payload_identity(sources, source_bindings)
    ctypes.memset(
        ctypes.addressof(source_owners[0]),
        0xE7,
        ctypes.sizeof(source_owners[0]),
    )
    provider = MooncakeWeightStoreProvider(
        _weight_store(store),
        payload_checksum_verifier=_runtime_payload_checksum,
    )

    with pytest.raises(WeightTransferError) as error_info:
        materialize_weights(
            source_placements=sources,
            source_bindings=source_bindings,
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id=_STORAGE_ID,
                object_prefix=_STORAGE_ID,
            ),
            provider=provider,
            payload_identity=declared_identity,
            attestor=_ALLOW_ALL_ATTESTOR,
        )

    assert error_info.value.phase == "prepare"
    assert error_info.value.completion_known is True
    assert error_info.value.retryable is False
    assert store.payload_write_attempts == 0
    assert not any("/payload/" in key for key in store.objects)
    assert f"{_STORAGE_ID}/manifest" not in store.objects


def test_mooncake_wrapper_ticket_failure_aborts_prepared_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _InMemoryStore()
    sources, source_bindings, _source_owners = _runtime_world(
        "source",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    provider = MooncakeWeightStoreProvider(
        _weight_store(store),
        payload_checksum_verifier=_runtime_payload_checksum,
    )

    def fail_ticket(_prepared, *, execution_context):
        assert execution_context is None
        raise ValueError("ticket encoding failed")

    monkeypatch.setattr(provider, "_build_recovery_ticket", fail_ticket)

    with pytest.raises(WeightTransferError, match="ticket encoding failed") as raised:
        materialize_weights(
            source_placements=sources,
            source_bindings=source_bindings,
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id=_STORAGE_ID,
                object_prefix=_STORAGE_ID,
            ),
            provider=provider,
            payload_identity=_payload_identity(sources, source_bindings),
            attestor=_ALLOW_ALL_ATTESTOR,
        )

    assert raised.value.completion_known is True
    assert f"{_STORAGE_ID}/manifest" not in store.objects
    assert not any("/payload/" in key for key in store.objects)
    decision_keys = [key for key in store.objects if key.endswith("/decision")]
    assert len(decision_keys) == 1
    assert json.loads(store.objects[decision_keys[0]])["decision"] == "abort"
    assert store.registered == {}


def test_mooncake_wrapper_ticket_cleanup_is_known_without_upload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _InMemoryStore(fail_decision_put=True)
    sources, source_bindings, _source_owners = _runtime_world(
        "source",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    provider = MooncakeWeightStoreProvider(
        _weight_store(store),
        payload_checksum_verifier=_runtime_payload_checksum,
    )

    def fail_ticket(_prepared):
        raise ValueError("ticket encoding failed")

    monkeypatch.setattr(provider, "_build_recovery_ticket", fail_ticket)

    with pytest.raises(WeightTransferError) as raised:
        materialize_weights(
            source_placements=sources,
            source_bindings=source_bindings,
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id=_STORAGE_ID,
                object_prefix=_STORAGE_ID,
            ),
            provider=provider,
            payload_identity=_payload_identity(sources, source_bindings),
            attestor=_ALLOW_ALL_ATTESTOR,
        )

    assert raised.value.completion_known is True
    assert raised.value.code == "PREFLIGHT_CLEANUP_FAILED"
    assert "upload decision is not complete" in str(raised.value)
    assert "upload session has no terminal decision" in str(raised.value)
    assert f"{_STORAGE_ID}/manifest" not in store.objects
    assert not any("/payload/" in key for key in store.objects)
    assert not any(key.endswith("/decision") for key in store.objects)
    assert store.registered == {}


def test_mooncake_wrapper_failed_upload_aborts_fail_closed() -> None:
    store = _InMemoryStore(fail_payload_write=2)
    weight_store = _weight_store(store)
    provider = MooncakeWeightStoreProvider(
        weight_store,
        payload_checksum_verifier=_runtime_payload_checksum,
    )
    sources, source_bindings, _source_owners = _runtime_world(
        "source",
        shard_dim=0,
        shard_count=4,
        dp_count=2,
        source_payload=True,
    )

    with pytest.raises(WeightTransferError) as error_info:
        materialize_weights(
            source_placements=sources,
            source_bindings=source_bindings,
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id=_STORAGE_ID,
                object_prefix=_STORAGE_ID,
            ),
            provider=provider,
            payload_identity=_payload_identity(sources, source_bindings),
            attestor=_ALLOW_ALL_ATTESTOR,
        )

    error = error_info.value
    assert error.code == "BACKEND_FAILURE"
    assert error.completion_known is True
    assert error.cleanup_required is True
    assert isinstance(error.__cause__, WeightStoreError)
    assert "batch_put_from failed" in str(error.__cause__)
    assert store.payload_write_attempts == 2
    assert not any("/payload/" in key for key in store.objects)
    assert f"{_STORAGE_ID}/manifest" not in store.objects

    decision_keys = [key for key in store.objects if key.endswith("/decision")]
    assert len(decision_keys) == 1
    assert json.loads(store.objects[decision_keys[0]])["decision"] == "abort"
    expected_payload_keys = {
        key.replace("/decision", "").split("/sessions/", 1)[0] for key in decision_keys
    }
    assert expected_payload_keys == {_STORAGE_ID}
    removed_payload_keys = {key for key in store.removed_keys if "/payload/" in key}
    assert len(removed_payload_keys) == 4
    assert removed_payload_keys >= set(store.payload_keys)
    assert Counter(address for address, _ in store.register_calls) == Counter(
        store.unregister_calls
    )
    assert store.registered == {}
    with pytest.raises(WeightStoreError, match="manifest get failed"):
        weight_store.load_manifest(f"{_STORAGE_ID}/manifest")


def test_mooncake_wrapper_manifest_publish_failure_is_fail_closed() -> None:
    store = _InMemoryStore(fail_manifest_put=True)
    weight_store = _weight_store(store)
    provider = MooncakeWeightStoreProvider(
        weight_store,
        payload_checksum_verifier=_runtime_payload_checksum,
    )
    sources, source_bindings, _source_owners = _runtime_world(
        "source",
        shard_dim=0,
        shard_count=4,
        dp_count=2,
        source_payload=True,
    )

    with pytest.raises(WeightTransferError) as error_info:
        materialize_weights(
            source_placements=sources,
            source_bindings=source_bindings,
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id=_STORAGE_ID,
                object_prefix=_STORAGE_ID,
            ),
            provider=provider,
            payload_identity=_payload_identity(sources, source_bindings),
            attestor=_ALLOW_ALL_ATTESTOR,
        )

    error = error_info.value
    assert error.code == "COMPLETION_UNKNOWN"
    assert error.completion_known is False
    assert error.cleanup_required is True
    assert error.completion_ticket is not None
    assert isinstance(error.__cause__, WeightTransferCompletionUnknownError)
    assert isinstance(error.__cause__.__cause__, WeightStoreError)
    assert "manifest put failed: -1" in str(error.__cause__.__cause__)
    assert f"{_STORAGE_ID}/manifest" not in store.objects
    assert len([key for key in store.objects if "/payload/" in key]) == 4

    decision_keys = [key for key in store.objects if key.endswith("/decision")]
    assert len(decision_keys) == 1
    assert json.loads(store.objects[decision_keys[0]])["decision"] == "commit"
    assert Counter(address for address, _ in store.register_calls) == Counter(
        store.unregister_calls
    )
    assert store.registered == {}
    with pytest.raises(WeightStoreError, match="manifest get failed"):
        _weight_store(store).load_manifest(f"{_STORAGE_ID}/manifest")


def test_mooncake_wrapper_recovery_aborts_before_upload() -> None:
    store = _InMemoryStore()
    sources, source_bindings, _source_owners = _runtime_world(
        "source",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    request = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=source_bindings,
        destination=WeightStorageDestination(
            provider="mooncake-store",
            storage_id=_STORAGE_ID,
            object_prefix=_STORAGE_ID,
        ),
        payload_identity=_payload_identity(sources, source_bindings),
        operation_id="crash-before-submit",
    )
    first_provider = MooncakeWeightStoreProvider(
        _weight_store(store),
        payload_checksum_verifier=_runtime_payload_checksum,
    )
    prepared = first_provider.prepare(request)
    ticket = first_provider.materialization_recovery_ticket(prepared)
    assert ticket is not None

    with pytest.raises(WeightTransferError) as error_info:
        MooncakeWeightStoreProvider(_weight_store(store)).recover_materialization(
            request,
            completion_ticket=ticket,
        )

    assert error_info.value.code == "RECOVERY_INCOMPLETE_PAYLOAD"
    assert error_info.value.completion_known is True
    assert f"{_STORAGE_ID}/manifest" not in store.objects
    assert not any("/payload/" in key for key in store.objects)
    decision_keys = [key for key in store.objects if key.endswith("/decision")]
    assert len(decision_keys) == 1
    assert json.loads(store.objects[decision_keys[0]])["decision"] == "abort"


def test_mooncake_wrapper_recovers_commit_after_provider_restart(
    tmp_path,
) -> None:
    store = _InMemoryStore(fail_manifest_put=True)
    sources, source_bindings, _source_owners = _runtime_world(
        "source",
        shard_dim=0,
        shard_count=4,
        dp_count=2,
        source_payload=True,
    )
    catalog_path = tmp_path / "weight-catalog.json"
    catalog = FileWeightStorageCatalog(catalog_path)
    payload_identity = _payload_identity(sources, source_bindings)
    destination = WeightStorageDestination(
        provider="mooncake-store",
        storage_id=_STORAGE_ID,
        object_prefix=_STORAGE_ID,
    )

    with pytest.raises(WeightTransferCompletionUnknownError) as error_info:
        materialize_weight_snapshot(
            source_placements=sources,
            source_bindings=source_bindings,
            destination=destination,
            provider=MooncakeWeightStoreProvider(
                _weight_store(store),
                payload_checksum_verifier=_runtime_payload_checksum,
            ),
            catalog=catalog,
            payload_identity=payload_identity,
            publication_id="restart-recovery",
            attestor=_ALLOW_ALL_ATTESTOR,
        )

    ticket = error_info.value.completion_ticket
    assert ticket is not None
    assert ticket.startswith("sglang-mooncake-weight-upload-v1:")
    assert store.payload_write_attempts == 4
    assert f"{_STORAGE_ID}/manifest" not in store.objects

    catalog = FileWeightStorageCatalog(catalog_path)
    attempt = catalog.get_materialization("restart-recovery")
    assert attempt is not None
    assert attempt.completion_ticket == ticket
    store.fail_manifest_put = False
    publication = materialize_weight_snapshot(
        source_placements=sources,
        source_bindings=source_bindings,
        destination=destination,
        provider=MooncakeWeightStoreProvider(_weight_store(store)),
        catalog=catalog,
        payload_identity=payload_identity,
        publication_id="restart-recovery",
        attestor=_ALLOW_ALL_ATTESTOR,
    )

    assert publication.state.value == "published"
    assert store.payload_write_attempts == 4
    assert f"{_STORAGE_ID}/manifest" in store.objects


def test_mooncake_wrapper_rejects_rebound_runtime_without_reupload() -> None:
    store = _InMemoryStore()
    sources, source_bindings, _source_owners = _runtime_world(
        "source",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    request, ticket, plan_signature = _leave_recoverable_payload(
        store,
        sources,
        source_bindings,
        operation_id="rebound-runtime-recovery",
    )
    _session_group_id, _control_key, payload_keys = plan_signature
    assert payload_keys <= set(store.objects)
    payload_writes_before_recovery = store.payload_write_attempts

    restarted_bindings, restarted_owners = _restart_runtime_bindings(source_bindings)
    assert restarted_owners
    restarted_request = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=restarted_bindings,
        destination=request.destination,
        payload_identity=_payload_identity(sources, restarted_bindings),
        operation_id=request.operation_id,
    )
    assert restarted_request.source_placements == request.source_placements
    assert restarted_request.payload_identity == request.payload_identity
    assert restarted_request.source_bindings != request.source_bindings
    assert all(
        restarted.instance_id != original.instance_id
        and restarted.generation != original.generation
        and restarted.lease_id != original.lease_id
        and restarted.fragments[0].endpoint != original.fragments[0].endpoint
        for original, restarted in zip(
            request.source_bindings,
            restarted_request.source_bindings,
            strict=True,
        )
    )

    with pytest.raises(
        WeightTransferError,
        match="ticket differs from the request",
    ) as raised:
        MooncakeWeightStoreProvider(_weight_store(store)).recover_materialization(
            restarted_request,
            completion_ticket=ticket,
        )

    assert raised.value.code == "INVALID_COMPLETION_TICKET"
    assert raised.value.phase == "recover"
    assert raised.value.completion_known is True
    assert raised.value.cleanup_required is True
    assert store.payload_write_attempts == payload_writes_before_recovery
    assert payload_keys <= set(store.objects)
    assert f"{_STORAGE_ID}/manifest" not in store.objects
    assert not any(key.endswith("/decision") for key in store.objects)


def test_mooncake_wrapper_conflict_keeps_first_payload() -> None:
    store = _InMemoryStore()
    first_sources, first_bindings, _first_owners = _runtime_world(
        "first",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    second_sources, second_bindings, second_owners = _runtime_world(
        "second",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    for owner in second_owners:
        ctypes.memset(ctypes.addressof(owner), 0xE7, ctypes.sizeof(owner))
    destination = WeightStorageDestination(
        provider="mooncake-store",
        storage_id=_STORAGE_ID,
        object_prefix=_STORAGE_ID,
    )
    first_provider = MooncakeWeightStoreProvider(
        _weight_store(store),
        payload_checksum_verifier=_runtime_payload_checksum,
    )
    first = materialize_weights(
        source_placements=first_sources,
        source_bindings=first_bindings,
        destination=destination,
        provider=first_provider,
        payload_identity=_payload_identity(first_sources, first_bindings),
        attestor=_ALLOW_ALL_ATTESTOR,
    )
    first_manifest = bytes(store.objects[first.manifest_key])

    with pytest.raises(WeightTransferError) as error_info:
        materialize_weights(
            source_placements=second_sources,
            source_bindings=second_bindings,
            destination=destination,
            provider=MooncakeWeightStoreProvider(
                _weight_store(store),
                payload_checksum_verifier=_runtime_payload_checksum,
            ),
            payload_identity=_payload_identity(second_sources, second_bindings),
            attestor=_ALLOW_ALL_ATTESTOR,
        )

    error = error_info.value
    assert error.code == "STORAGE_CONFLICT"
    assert error.completion_known is True
    assert not isinstance(error, WeightTransferCompletionUnknownError)
    assert store.objects[first.manifest_key] == first_manifest
    assert len([key for key in store.objects if "/payload/" in key]) == 4


def test_mooncake_wrapper_conflict_cleans_exact_loser_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _InMemoryStore()
    loser_sources, loser_bindings, _loser_owners = _runtime_world(
        "loser",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    request, ticket, loser_signature = _leave_recoverable_payload(
        store,
        loser_sources,
        loser_bindings,
        operation_id="loser-conflict-recovery",
    )
    _session_group_id, _control_key, loser_payload_keys = loser_signature
    assert loser_payload_keys <= set(store.objects)
    manifest_key, winner_manifest, winner_payload_keys = _publish_conflicting_winner(
        store,
        loser_payload_keys,
    )

    recovery_store = _weight_store(store)
    abort_upload = recovery_store.abort_upload
    finalize_upload_session = recovery_store.finalize_upload_session
    cleanup_calls = []

    def record_abort(plan, receipts):
        cleanup_calls.append(
            (
                "abort",
                _upload_plan_signature(plan),
                frozenset(receipt.object_key for receipt in receipts),
            )
        )
        return abort_upload(plan, receipts)

    def record_finalize(plan):
        cleanup_calls.append(
            (
                "finalize",
                _upload_plan_signature(plan),
                frozenset(),
            )
        )
        return finalize_upload_session(plan)

    monkeypatch.setattr(recovery_store, "abort_upload", record_abort)
    monkeypatch.setattr(
        recovery_store,
        "finalize_upload_session",
        record_finalize,
    )

    with pytest.raises(WeightTransferError) as error_info:
        MooncakeWeightStoreProvider(recovery_store).recover_materialization(
            request,
            completion_ticket=ticket,
        )

    error = error_info.value
    assert error.code == "STORAGE_CONFLICT"
    assert error.completion_known is True
    assert error.cleanup_required is False
    assert cleanup_calls == [
        ("abort", loser_signature, loser_payload_keys),
        ("finalize", loser_signature, frozenset()),
    ]
    removed_payload_keys = {key for key in store.removed_keys if "/payload/" in key}
    assert loser_payload_keys <= removed_payload_keys
    assert winner_payload_keys.isdisjoint(removed_payload_keys)
    assert loser_payload_keys.isdisjoint(store.objects)
    assert winner_payload_keys <= set(store.objects)
    assert store.objects[manifest_key] == winner_manifest


def test_mooncake_wrapper_conflict_cleanup_failure_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _InMemoryStore()
    loser_sources, loser_bindings, _loser_owners = _runtime_world(
        "loser",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    request, ticket, loser_signature = _leave_recoverable_payload(
        store,
        loser_sources,
        loser_bindings,
        operation_id="loser-conflict-cleanup-unknown",
    )
    _session_group_id, _control_key, loser_payload_keys = loser_signature
    manifest_key, winner_manifest, winner_payload_keys = _publish_conflicting_winner(
        store,
        loser_payload_keys,
    )

    recovery_store = _weight_store(store)
    finalize_upload_session = recovery_store.finalize_upload_session
    cleanup_calls = []

    def fail_abort(plan, receipts):
        cleanup_calls.append(
            (
                "abort",
                _upload_plan_signature(plan),
                frozenset(receipt.object_key for receipt in receipts),
            )
        )
        raise WeightStoreError("abort cleanup unavailable")

    def record_finalize(plan):
        cleanup_calls.append(
            (
                "finalize",
                _upload_plan_signature(plan),
                frozenset(),
            )
        )
        return finalize_upload_session(plan)

    monkeypatch.setattr(recovery_store, "abort_upload", fail_abort)
    monkeypatch.setattr(
        recovery_store,
        "finalize_upload_session",
        record_finalize,
    )

    with pytest.raises(WeightTransferCompletionUnknownError) as error_info:
        MooncakeWeightStoreProvider(recovery_store).recover_materialization(
            request,
            completion_ticket=ticket,
        )

    error = error_info.value
    assert error.completion_known is False
    assert error.completion_ticket == ticket
    assert error.phase == "recover"
    assert cleanup_calls == [
        ("abort", loser_signature, loser_payload_keys),
        ("finalize", loser_signature, frozenset()),
    ]
    assert loser_payload_keys <= set(store.objects)
    assert winner_payload_keys <= set(store.objects)
    assert store.objects[manifest_key] == winner_manifest


def test_mooncake_wrapper_partial_load_target_is_not_ready() -> None:
    store = _InMemoryStore()
    sources, source_bindings, _source_owners = _runtime_world(
        "source",
        shard_dim=0,
        shard_count=4,
        dp_count=1,
        source_payload=True,
    )
    catalog = InMemoryWeightStorageCatalog()
    destination = WeightStorageDestination(
        provider="mooncake-store",
        storage_id=_STORAGE_ID,
        object_prefix=_STORAGE_ID,
    )
    publication = materialize_weight_snapshot(
        source_placements=sources,
        source_bindings=source_bindings,
        destination=destination,
        provider=MooncakeWeightStoreProvider(
            WeightStore(
                store,
                config_factory=_config_factory,
                max_ranges_per_request=1,
            ),
            payload_checksum_verifier=_runtime_payload_checksum,
        ),
        catalog=catalog,
        payload_identity=_payload_identity(sources, source_bindings),
        publication_id="partial-store-load",
        attestor=_ALLOW_ALL_ATTESTOR,
    )
    targets, target_bindings, target_owners = _runtime_world(
        "target",
        shard_dim=1,
        shard_count=1,
        dp_count=1,
        source_payload=False,
    )
    coordinator = WeightSnapshotCoordinator()
    target_binding = _bind_target_snapshot(target_bindings[0], coordinator)
    session = WeightTargetLoadSession(
        target_bindings=(target_binding,),
        owners=target_owners,
        coordinator=coordinator,
    )
    store.fail_range_get = 2

    with pytest.raises(WeightTransferError, match="range read failed") as error_info:
        load_weight_snapshot(
            publication.snapshot.ref,
            catalog=catalog,
            target_placements=targets,
            target_bindings=(target_binding,),
            provider=MooncakeWeightStoreProvider(
                WeightStore(
                    store,
                    config_factory=_config_factory,
                    max_ranges_per_request=1,
                )
            ),
            target_mode=WeightTargetLoadMode.LIVE_UPDATE,
            target_session=session,
            attestor=_ALLOW_ALL_ATTESTOR,
        )

    assert error_info.value.completion_known is True
    assert session.state is WeightTargetLoadState.POISONED
    target_bytes = ctypes.string_at(
        ctypes.addressof(target_owners[0]),
        prod(_GLOBAL_SHAPE) * _ITEMSIZE,
    )
    assert target_bytes != bytes([0xA5]) * len(target_bytes)
    assert target_bytes != _box_payload((0, 0, 0), _GLOBAL_SHAPE)
    with pytest.raises(RuntimeError, match="not ready"):
        session.require_ready()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
