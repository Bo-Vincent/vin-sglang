from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from math import prod
from types import SimpleNamespace

import pytest

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
from sglang.srt.weight_transfer.distributed import WeightStoreUploadOutcome
from sglang.srt.weight_transfer.mooncake_store import (
    MooncakeWeightStoreProvider,
)
from sglang.srt.weight_transfer.planner import (
    plan_weight_transfer_to_local_target,
)
from sglang.test.ci.ci_register import register_cpu_ci

from sglang.srt.weight_transfer.provider import (
    WeightPayloadIdentity,
    WeightStorageDestination,
    WeightTargetLoadSession,
    WeightTargetLoadMode,
    WeightTransferError,
)
from sglang.srt.weight_transfer.storage import (
    InMemoryWeightStorageCatalog,
    WeightSnapshotPublicationState,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@dataclass(frozen=True)
class FakeDescriptor:
    tensor_id: str
    global_shape: tuple[int, ...]
    dtype: str
    itemsize: int
    partition_dim: int | None
    layer_id: int | None
    expert_id: int | None
    layout_fingerprint: str
    shard_dims: tuple[int, ...]


@dataclass(frozen=True)
class FakePlacementFragment:
    placement_fragment_id: str
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    nbytes: int
    rank: WeightParallelRank
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class FakeStoredFragment:
    fragment_id: str
    tensor_id: str
    global_offset: tuple[int, ...]
    local_shape: tuple[int, ...]
    object_key: str
    object_offset: int
    nbytes: int
    checksum: str | None = None


@dataclass(frozen=True)
class FakeWeightManifest:
    model_id: str
    revision: str
    group_id: str
    manifest_key: str
    tensors: tuple[FakeDescriptor, ...]
    fragments: tuple[FakeStoredFragment, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "model_id": self.model_id,
                "revision": self.revision,
                "group_id": self.group_id,
                "manifest_key": self.manifest_key,
                "tensors": [asdict(tensor) for tensor in self.tensors],
                "fragments": [asdict(fragment) for fragment in self.fragments],
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


class FakePlacementManifest:
    @classmethod
    def from_runtime_inventory(cls, inventory):
        descriptors = {}
        fragments = []
        for tensor in inventory.tensors:
            descriptors[tensor.tensor_id] = FakeDescriptor(
                tensor_id=tensor.tensor_id,
                global_shape=tuple(tensor.global_shape),
                dtype=tensor.dtype,
                itemsize=tensor.itemsize,
                partition_dim=tensor.partition_dim,
                layer_id=tensor.layer_id,
                expert_id=tensor.expert_id,
                layout_fingerprint=tensor.layout_fingerprint,
                shard_dims=tuple(tensor.shard_dims),
            )
            fragments.append(
                FakePlacementFragment(
                    placement_fragment_id=tensor.placement_fragment_id,
                    tensor_id=tensor.tensor_id,
                    global_offset=tuple(tensor.global_offset),
                    local_shape=tuple(tensor.local_shape),
                    nbytes=tensor.nbytes,
                    rank=tensor.rank,
                    aliases=tuple(tensor.aliases),
                )
            )
        return SimpleNamespace(
            model_id=inventory.model_id,
            revision=inventory.revision,
            placement_id=inventory.placement_id,
            tensors=tuple(descriptors.values()),
            fragments=tuple(fragments),
        )


class FakeRuntimeBindingManifest:
    @classmethod
    def from_runtime_inventory(cls, inventory):
        return inventory


class FakeLogicalTransferPlan:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeTransferRegion:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakePipelineRouteGroup:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeWeightLoadPlan:
    def __init__(self, *, manifest, transfer):
        self.manifest = manifest
        self.transfer = transfer


class FakeWeightStoreError(RuntimeError):
    pass


class AllowAllAttestor:
    def attest(self, _request) -> None:
        pass


ALLOW_ALL_ATTESTOR = AllowAllAttestor()


class FakeWeightStore:
    def __init__(
        self,
        *,
        fail_upload: bool = False,
        lose_commit_response_once: bool = False,
    ):
        self.fail_upload = fail_upload
        self.lose_commit_response_once = lose_commit_response_once
        self.calls = []
        self.persisted = None

    def prepare_upload(self, source_manifests, *, namespace):
        self.calls.append(("prepare_upload", namespace))
        group_id = f"weights/{namespace}/model/revision"
        operations = []
        stored = []
        for manifest in source_manifests:
            for source in manifest.fragments:
                target = FakeStoredFragment(
                    fragment_id=f"stored:{source.fragment_id}",
                    tensor_id=source.tensor_id,
                    global_offset=source.global_offset,
                    local_shape=source.local_shape,
                    object_key=(f"{group_id}/payload/upload/{source.fragment_id}"),
                    object_offset=0,
                    nbytes=source.nbytes,
                    checksum=f"checksum:{source.fragment_id}",
                )
                stored.append(target)
                operations.append(SimpleNamespace(source=source, target=target))
        weight_manifest = FakeWeightManifest(
            model_id="model",
            revision="revision",
            group_id=group_id,
            manifest_key=f"{group_id}/manifest",
            tensors=tuple(
                sorted(
                    {
                        tensor.tensor_id: tensor
                        for manifest in source_manifests
                        for tensor in manifest.tensors
                    }.values(),
                    key=lambda tensor: tensor.tensor_id,
                )
            ),
            fragments=tuple(stored),
        )
        return SimpleNamespace(
            manifest=weight_manifest,
            operations=tuple(operations),
            session_group_id=f"{group_id}/sessions/upload",
            control_key=f"{group_id}/sessions/upload/decision",
        )

    def upload(self, plan, runtime_manifest, *, pre_registered):
        self.calls.append(("upload", runtime_manifest.instance_id, pre_registered))
        if self.fail_upload:
            raise FakeWeightStoreError("upload failed")
        local_ids = {fragment.fragment_id for fragment in runtime_manifest.fragments}
        return tuple(
            SimpleNamespace(
                fragment_id=operation.target.fragment_id,
                object_key=operation.target.object_key,
                worker_id=operation.source.worker_id,
                checksum=operation.target.checksum,
            )
            for operation in plan.operations
            if operation.source.fragment_id in local_ids
        )

    def commit(self, plan, receipts):
        self.calls.append(("commit", len(receipts)))
        if len(receipts) != len(plan.operations):
            raise FakeWeightStoreError("incomplete receipts")
        self.persisted = plan.manifest
        if self.lose_commit_response_once:
            self.lose_commit_response_once = False
            raise FakeWeightStoreError("commit response lost")
        return self.persisted

    def abort_upload(self, plan, receipts):
        self.calls.append(("abort", len(receipts)))

    def finalize_upload_session(self, plan):
        self.calls.append(("finalize", plan.manifest.group_id))

    def load_manifest(self, manifest_key):
        self.calls.append(("load_manifest", manifest_key))
        if self.persisted is None:
            raise FakeWeightStoreError("manifest missing")
        return self.persisted

    def manifest_exists(self, manifest_key):
        del manifest_key
        return self.persisted is not None

    def load(self, plan, runtime_manifest, *, pre_registered):
        self.calls.append(
            (
                "load",
                len(plan.transfer.operations),
                runtime_manifest.instance_id,
                pre_registered,
            )
        )


class CorruptCommittedManifestStore(FakeWeightStore):
    def __init__(self, field: str) -> None:
        super().__init__()
        self.field = field

    def commit(self, plan, receipts):
        self.calls.append(("commit", len(receipts)))
        if len(receipts) != len(plan.operations):
            raise FakeWeightStoreError("incomplete receipts")
        values = dict(plan.manifest.__dict__)
        if self.field in {
            "model_id",
            "revision",
            "group_id",
            "manifest_key",
        }:
            values[self.field] = f"changed:{values[self.field]}"
        elif self.field == "tensors":
            values["tensors"] = ()
        else:
            fragment = values["fragments"][0]
            updates = {
                "fragment_id": {"fragment_id": f"changed:{fragment.fragment_id}"},
                "tensor_id": {"tensor_id": f"changed:{fragment.tensor_id}"},
                "global_offset": {
                    "global_offset": (fragment.global_offset[0] + 1,)
                    + fragment.global_offset[1:]
                },
                "local_shape": {
                    "local_shape": (fragment.local_shape[0] + 1,)
                    + fragment.local_shape[1:]
                },
                "object_key": {"object_key": f"changed:{fragment.object_key}"},
                "object_offset": {"object_offset": fragment.object_offset + 1},
                "nbytes": {"nbytes": fragment.nbytes + 1},
                "checksum": {"checksum": f"changed:{fragment.checksum}"},
            }
            values["fragments"] = (
                replace(fragment, **updates[self.field]),
                *values["fragments"][1:],
            )
        self.persisted = SimpleNamespace(**values)
        return self.persisted


class FakeDistributedCoordinator:
    def __init__(
        self,
        remote_placement_id: str,
        *,
        omit_remote_ownership: bool = False,
        remote_preflight_error: str | None = None,
        remote_receipt_checksum: str = "valid",
        swap_receipt_ownership: bool = False,
    ) -> None:
        self.rank = 0
        self.world_size = 2
        self.remote_placement_id = remote_placement_id
        self.omit_remote_ownership = omit_remote_ownership
        self.remote_preflight_error = remote_preflight_error
        self.remote_receipt_checksum = remote_receipt_checksum
        self.swap_receipt_ownership = swap_receipt_ownership
        self.upload_plan = None
        self.calls = []

    def prepare_upload(self, factory):
        self.calls.append("prepare")
        self.upload_plan = factory()
        return self.upload_plan

    def exchange_preflight_outcome(self, outcome):
        self.calls.append(("preflight", outcome))
        remote = type(outcome)(
            rank=1,
            error=self.remote_preflight_error,
        )
        return outcome, remote

    def exchange_upload_outcome(self, outcome):
        self.calls.append(("exchange", outcome))
        assert self.upload_plan is not None
        remote_receipts = []
        for operation in self.upload_plan.operations:
            if operation.source.worker_id != self.remote_placement_id:
                continue
            fields = {
                "fragment_id": operation.target.fragment_id,
                "object_key": operation.target.object_key,
                "worker_id": operation.source.worker_id,
            }
            if self.remote_receipt_checksum != "missing":
                fields["checksum"] = (
                    operation.target.checksum
                    if self.remote_receipt_checksum == "valid"
                    else f"sha256:{hashlib.sha256(b'corrupt receipt').hexdigest()}"
                )
            remote_receipts.append(SimpleNamespace(**fields))
        remote = WeightStoreUploadOutcome(
            rank=1,
            placement_ids=(
                () if self.omit_remote_ownership else (self.remote_placement_id,)
            ),
            receipts=(() if self.omit_remote_ownership else tuple(remote_receipts)),
            error=None,
        )
        if self.swap_receipt_ownership:
            local = WeightStoreUploadOutcome(
                rank=outcome.rank,
                placement_ids=outcome.placement_ids,
                receipts=remote.receipts,
                error=outcome.error,
            )
            remote = WeightStoreUploadOutcome(
                rank=remote.rank,
                placement_ids=remote.placement_ids,
                receipts=outcome.receipts,
                error=remote.error,
            )
            return local, remote
        return outcome, remote

    def commit_upload(self, factory):
        self.calls.append("commit")
        return factory()

    def abort_upload(self, factory):
        self.calls.append("abort")
        factory()

    def finalize_upload(self, factory):
        self.calls.append("finalize")
        factory()


def fake_backend(calls):
    def bind_runtime_manifest(placement, binding):
        fragments_by_id = {
            fragment.placement_fragment_id: fragment for fragment in binding.fragments
        }
        runtime_fragments = []
        for fragment in placement.fragments:
            bound = fragments_by_id[fragment.placement_fragment_id]
            runtime_fragments.append(
                SimpleNamespace(
                    fragment_id=bound.fragment_id,
                    placement_fragment_id=fragment.placement_fragment_id,
                    tensor_id=fragment.tensor_id,
                    global_offset=fragment.global_offset,
                    local_shape=fragment.local_shape,
                    address=bound.address,
                    nbytes=bound.nbytes,
                    worker_id=bound.worker_id,
                    endpoint=bound.endpoint,
                    rank=fragment.rank,
                    lease_generation=binding.generation,
                    aliases=fragment.aliases,
                )
            )
        return SimpleNamespace(
            model_id=placement.model_id,
            revision=placement.revision,
            placement_id=placement.placement_id,
            instance_id=binding.instance_id,
            generation=binding.generation,
            lease_id=binding.lease_id,
            tensors=placement.tensors,
            fragments=tuple(runtime_fragments),
        )

    def bind_logical_transfer_plan(logical, targets, *, source_bindings=()):
        assert not source_bindings
        calls["logical"] = logical
        target_manifests = tuple(
            bind_runtime_manifest(placement, binding)
            for placement, binding in zip(
                logical.target_placements,
                targets,
                strict=True,
            )
        )
        target_fragments = {
            fragment.placement_fragment_id: fragment
            for manifest in target_manifests
            for fragment in manifest.fragments
        }
        operations = tuple(
            SimpleNamespace(
                **{
                    **operation.__dict__,
                    "target": target_fragments[operation.target.placement_fragment_id],
                }
            )
            for operation in logical.operations
        )
        return SimpleNamespace(operations=operations)

    return SimpleNamespace(
        LogicalTransferPlan=FakeLogicalTransferPlan,
        PipelineRouteGroup=FakePipelineRouteGroup,
        RuntimeBindingManifest=FakeRuntimeBindingManifest,
        SourcePlacementManifest=FakePlacementManifest,
        StoredFragment=FakeStoredFragment,
        TargetPlacementManifest=FakePlacementManifest,
        TransferRegion=FakeTransferRegion,
        WeightLoadPlan=FakeWeightLoadPlan,
        WeightStoreError=FakeWeightStoreError,
        bind_logical_transfer_plan=bind_logical_transfer_plan,
        bind_runtime_manifest=bind_runtime_manifest,
    )


def placements(
    side: str,
    *,
    shard_dim: int,
    parts: int = 2,
) -> tuple[WeightPlacementManifest, ...]:
    global_shape = (4, 4)
    result = []
    for rank in range(parts):
        local_shape = list(global_shape)
        global_offset = [0, 0]
        local_shape[shard_dim] //= parts
        global_offset[shard_dim] = rank * local_shape[shard_dim]
        worker = f"{side}:t{rank}"
        tensor = WeightPlacementTensor(
            placement_fragment_id=f"{worker}:fragment",
            tensor_id="weight",
            runtime_name="weight",
            aliases=("weight",),
            global_shape=global_shape,
            global_offset=tuple(global_offset),
            local_shape=tuple(local_shape),
            dtype="uint8",
            itemsize=1,
            partition_dim=shard_dim,
            shard_dims=(shard_dim,),
            layer_id=0,
            expert_id=None,
            layout_fingerprint="layout:v1",
            nbytes=prod(local_shape),
            byte_offset=0,
            rank=WeightParallelRank(tp=rank),
        )
        tensors = (tensor,)
        result.append(
            WeightPlacementManifest(
                model_id="model",
                revision="revision",
                placement_id=compute_weight_placement_id(tuple(tensors)),
                tensors=tuple(tensors),
            )
        )
    return tuple(result)


def bindings(
    manifests: tuple[WeightPlacementManifest, ...],
    *,
    address_base: int,
) -> tuple[WeightRuntimeBindingManifest, ...]:
    result = []
    for index, manifest in enumerate(manifests):
        tensor = manifest.tensors[0]
        result.append(
            WeightRuntimeBindingManifest(
                model_id=manifest.model_id,
                revision=manifest.revision,
                placement_id=manifest.placement_id,
                instance_id=f"instance:{manifest.placement_id}",
                generation=1,
                lease_id="lease:1",
                fragments=(
                    RuntimeWeightBinding(
                        placement_fragment_id=tensor.placement_fragment_id,
                        fragment_id=f"runtime:{tensor.placement_fragment_id}",
                        address=address_base + index * 0x1000,
                        nbytes=tensor.nbytes,
                        storage_offset=0,
                        device="cuda:0",
                        is_contiguous=True,
                        worker_id=manifest.placement_id,
                        endpoint=f"{manifest.placement_id}:1",
                    ),
                ),
            )
        )
    return tuple(result)


def payload_identity(
    manifests: tuple[WeightPlacementManifest, ...],
) -> WeightPayloadIdentity:
    return WeightPayloadIdentity.create(manifests, payload_checksums(manifests))


def payload_checksums(
    manifests: tuple[WeightPlacementManifest, ...],
) -> dict[str, str]:
    return {
        tensor.placement_fragment_id: (
            "sha256:"
            + hashlib.sha256(tensor.placement_fragment_id.encode("utf-8")).hexdigest()
        )
        for manifest in manifests
        for tensor in manifest.tensors
    }


def payload_checksum_verifier(manifests):
    checksums = payload_checksums(manifests)

    def verify(location):
        return checksums[location.placement_fragment_id]

    return verify


def route_alias_placements() -> tuple[
    tuple[WeightPlacementManifest, ...],
    WeightPlacementManifest,
]:
    aliases = ("shared.weight", "shared.weight_alias")

    def tensor(
        *,
        fragment_id: str,
        tensor_id: str,
        tensor_aliases: tuple[str, ...],
        pp_rank: int,
    ) -> WeightPlacementTensor:
        return WeightPlacementTensor(
            placement_fragment_id=fragment_id,
            tensor_id=tensor_id,
            runtime_name=tensor_id,
            aliases=tensor_aliases,
            global_shape=(4,),
            global_offset=(0,),
            local_shape=(4,),
            dtype="uint8",
            itemsize=1,
            partition_dim=None,
            shard_dims=(),
            layer_id=0,
            expert_id=None,
            layout_fingerprint="layout:v1",
            nbytes=4,
            byte_offset=0,
            rank=WeightParallelRank(pp=pp_rank),
        )

    source_shared_tensors = (
        tensor(
            fragment_id="source:shared:fragment",
            tensor_id="shared",
            tensor_aliases=aliases,
            pp_rank=1,
        ),
    )
    source_other_tensors = (
        tensor(
            fragment_id="source:other:fragment",
            tensor_id="other",
            tensor_aliases=("other",),
            pp_rank=2,
        ),
    )
    target_tensors = (
        tensor(
            fragment_id="target:a-shared:fragment",
            tensor_id="shared",
            tensor_aliases=aliases,
            pp_rank=7,
        ),
        tensor(
            fragment_id="target:b-shared-alias:fragment",
            tensor_id="shared",
            tensor_aliases=aliases,
            pp_rank=7,
        ),
        tensor(
            fragment_id="target:c-other:fragment",
            tensor_id="other",
            tensor_aliases=("other",),
            pp_rank=7,
        ),
    )
    sources = (
        WeightPlacementManifest(
            model_id="model",
            revision="revision",
            placement_id=compute_weight_placement_id(tuple(source_shared_tensors)),
            tensors=tuple(source_shared_tensors),
        ),
        WeightPlacementManifest(
            model_id="model",
            revision="revision",
            placement_id=compute_weight_placement_id(tuple(source_other_tensors)),
            tensors=tuple(source_other_tensors),
        ),
    )
    target = WeightPlacementManifest(
        model_id="model",
        revision="revision",
        placement_id=compute_weight_placement_id(tuple(target_tensors)),
        tensors=tuple(target_tensors),
    )
    return sources, target


def test_payload_identity_binds_fragment_content_checksums() -> None:
    sources = placements("source", shard_dim=0)
    first = payload_identity(sources)
    checksums = {
        fragment.placement_fragment_id: fragment.checksum
        for fragment in first.fragments
    }
    changed_fragment = first.fragments[0].placement_fragment_id
    checksums[changed_fragment] = (
        f"sha256:{hashlib.sha256(b'changed payload').hexdigest()}"
    )

    second = WeightPayloadIdentity.create(sources, checksums)

    assert second.payload_digest != first.payload_digest
    assert second.fragments != first.fragments


def test_mooncake_store_requires_payload_checksum_verifier_before_upload(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)

    with pytest.raises(
        WeightTransferError,
        match="payload checksum verifier",
    ) as error_info:
        materialize_weights(
            source_placements=sources,
            source_bindings=bindings(sources, address_base=0x10000),
            payload_identity=payload_identity(sources),
            destination=WeightStorageDestination(
                provider="mooncake-store",
                storage_id="weights/default/model/revision",
                object_prefix="weights/default/model/revision",
            ),
            provider=MooncakeWeightStoreProvider(store),
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert error_info.value.phase == "prepare"
    assert error_info.value.completion_known is True
    assert error_info.value.retryable is False
    assert not any(call[0] == "upload" for call in store.calls)


def test_mooncake_store_requires_payload_identity_before_upload(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)

    with pytest.raises(WeightTransferError) as error_info:
        materialize_weights(
            source_placements=sources,
            source_bindings=bindings(sources, address_base=0x10000),
            destination=WeightStorageDestination(
                provider="mooncake-store",
                storage_id="weights/default/model/revision",
                object_prefix="weights/default/model/revision",
            ),
            provider=MooncakeWeightStoreProvider(
                store,
                payload_checksum_verifier=payload_checksum_verifier(sources),
            ),
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert error_info.value.code == "PAYLOAD_IDENTITY_REQUIRED"
    assert store.calls == []


def test_mooncake_store_materializes_and_loads_native_plan(
    monkeypatch,
) -> None:
    calls = {}
    backend = fake_backend(calls)
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)
    source_bindings = bindings(sources, address_base=0x10000)
    provider = MooncakeWeightStoreProvider(
        store,
        namespace="default",
        payload_checksum_verifier=payload_checksum_verifier(sources),
        source_pre_registered=True,
        target_pre_registered=True,
    )

    materialized = materialize_weights(
        source_placements=sources,
        source_bindings=source_bindings,
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider="mooncake-store",
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        provider=provider,
        attestor=ALLOW_ALL_ATTESTOR,
    )

    assert materialized.stored_placements == tuple(
        sorted(sources, key=lambda item: item.placement_id)
    )
    assert materialized.manifest_key == "weights/default/model/revision/manifest"
    assert materialized.total_bytes == 16
    assert materialized.fragment_count == 2
    assert len(materialized.storage_bindings) == 2
    assert [call[0] for call in store.calls] == [
        "prepare_upload",
        "upload",
        "upload",
        "commit",
        "finalize",
    ]

    targets = placements("target", shard_dim=1)
    target_bindings = bindings(targets, address_base=0x20000)
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(sources, targets[0]),
        source_bindings=materialized.storage_bindings,
        target_bindings=(target_bindings[0],),
    )
    receipt = execute_weight_load(
        request,
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
        attestor=ALLOW_ALL_ATTESTOR,
    )

    assert receipt.total_bytes == 8
    assert len(calls["logical"].operations) == len(request.plan.regions)
    assert all(
        isinstance(operation.source, FakeStoredFragment)
        for operation in calls["logical"].operations
    )
    assert store.calls[-2][0] == "load_manifest"
    assert store.calls[-1] == (
        "load",
        len(request.plan.regions),
        target_bindings[0].instance_id,
        True,
    )


def test_mooncake_store_rejects_chunk_expansion_before_io(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    store.max_range_bytes = 1
    sources = placements("source", shard_dim=0)
    source_bindings = bindings(sources, address_base=0x10000)
    provider = MooncakeWeightStoreProvider(
        store,
        max_total_operations=6,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    materialized = materialize_weights(
        source_placements=sources,
        source_bindings=source_bindings,
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        provider=provider,
        attestor=ALLOW_ALL_ATTESTOR,
    )
    target = placements("target", shard_dim=1)[0]
    target_binding = bindings((target,), address_base=0x20000)[0]
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(sources, target),
        source_bindings=materialized.storage_bindings,
        target_bindings=(target_binding,),
    )

    with pytest.raises(
        WeightTransferError,
        match="lowering exceeds the total operation limit",
    ) as raised:
        execute_weight_load(
            request,
            provider=provider,
            target_mode=WeightTargetLoadMode.COLD_START,
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert raised.value.phase == "prepare"
    assert not any(call[0] == "load" for call in store.calls)


def test_mooncake_store_distributed_uploads_only_local_placements(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)
    source_bindings = bindings(sources, address_base=0x10000)
    coordinator = FakeDistributedCoordinator(sources[1].placement_id)
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    receipt = materialize_weights(
        source_placements=sources,
        source_bindings=source_bindings,
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        provider=provider,
        attestor=ALLOW_ALL_ATTESTOR,
    )

    upload_calls = [call for call in store.calls if call[0] == "upload"]
    assert upload_calls == [
        (
            "upload",
            f"instance:{sources[0].placement_id}",
            False,
        )
    ]
    assert len(receipt.storage_bindings) == 2
    assert [call[0] for call in store.calls].count("prepare_upload") == 1
    assert [call[0] for call in store.calls].count("commit") == 1
    assert [call[0] for call in store.calls].count("finalize") == 1
    assert [
        call if isinstance(call, str) else call[0] for call in coordinator.calls
    ] == [
        "prepare",
        "preflight",
        "exchange",
        "commit",
        "finalize",
    ]


def test_mooncake_store_distributed_verifies_only_local_runtime_ranges(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)
    source_bindings = bindings(sources, address_base=0x10000)
    local_placement_id = sources[0].placement_id
    local_checksums = payload_checksums((sources[0],))
    verified = []

    def verify_local(location):
        if location.placement_id != local_placement_id:
            raise AssertionError("attempted to read another rank's runtime address")
        verified.append(location.placement_fragment_id)
        return local_checksums[location.placement_fragment_id]

    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(local_placement_id,),
        coordinator=FakeDistributedCoordinator(sources[1].placement_id),
        payload_checksum_verifier=verify_local,
    )

    materialize_weights(
        source_placements=sources,
        source_bindings=source_bindings,
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        provider=provider,
        attestor=ALLOW_ALL_ATTESTOR,
    )

    assert verified == [tensor.placement_fragment_id for tensor in sources[0].tensors]


def test_mooncake_store_allows_explicit_empty_local_ownership() -> None:
    sources = placements("source", shard_dim=0)
    request = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=bindings(sources, address_base=0x10000),
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider="mooncake-store",
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
    )
    provider = MooncakeWeightStoreProvider(
        FakeWeightStore(),
        local_placement_ids=(),
        coordinator=FakeDistributedCoordinator(sources[0].placement_id),
        payload_checksum_verifier=lambda _location: pytest.fail(
            "a rank with no local ownership must not read runtime addresses"
        ),
    )

    provider._verify_runtime_payload(request)


def test_mooncake_store_distributed_upload_failure_aborts_once(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore(fail_upload=True)
    sources = placements("source", shard_dim=0)
    coordinator = FakeDistributedCoordinator(sources[1].placement_id)
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(WeightTransferError, match="upload failed") as raised:
        materialize_weights(
            source_placements=sources,
            source_bindings=bindings(sources, address_base=0x10000),
            payload_identity=payload_identity(sources),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/default/model/revision",
                object_prefix="weights/default/model/revision",
            ),
            provider=provider,
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert raised.value.completion_known is True
    assert [call[0] for call in store.calls].count("abort") == 1
    assert not any(call[0] == "commit" for call in store.calls)
    assert coordinator.calls.count("abort") == 1


def test_mooncake_store_distributed_ownership_gap_aborts_before_commit(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)
    coordinator = FakeDistributedCoordinator(
        sources[1].placement_id,
        omit_remote_ownership=True,
    )
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(
        WeightTransferError,
        match="ownership is incomplete",
    ):
        materialize_weights(
            source_placements=sources,
            source_bindings=bindings(sources, address_base=0x10000),
            payload_identity=payload_identity(sources),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/default/model/revision",
                object_prefix="weights/default/model/revision",
            ),
            provider=provider,
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert [call[0] for call in store.calls].count("abort") == 1
    assert not any(call[0] == "commit" for call in store.calls)


def test_mooncake_store_remote_preflight_failure_aborts_and_finalizes_before_upload(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)
    coordinator = FakeDistributedCoordinator(
        sources[1].placement_id,
        remote_preflight_error="ValueError: remote upload plan mismatch",
    )
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(
        WeightTransferError,
        match="remote upload plan mismatch",
    ):
        materialize_weights(
            source_placements=sources,
            source_bindings=bindings(sources, address_base=0x10000),
            payload_identity=payload_identity(sources),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/default/model/revision",
                object_prefix="weights/default/model/revision",
            ),
            provider=provider,
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert [call[0] for call in store.calls] == [
        "prepare_upload",
        "abort",
        "finalize",
    ]
    assert [
        call if isinstance(call, str) else call[0] for call in coordinator.calls
    ] == ["prepare", "preflight", "abort", "finalize"]


def test_mooncake_store_rejects_receipts_swapped_between_rank_placements(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)
    coordinator = FakeDistributedCoordinator(
        sources[1].placement_id,
        swap_receipt_ownership=True,
    )
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(
        WeightTransferError,
        match="receipts do not match declared placements",
    ):
        materialize_weights(
            source_placements=sources,
            source_bindings=bindings(sources, address_base=0x10000),
            payload_identity=payload_identity(sources),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/default/model/revision",
                object_prefix="weights/default/model/revision",
            ),
            provider=provider,
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert [call[0] for call in store.calls].count("abort") == 1
    assert not any(call[0] == "commit" for call in store.calls)


@pytest.mark.parametrize(
    "remote_receipt_checksum",
    ["missing", "mismatch"],
)
def test_mooncake_store_rejects_incomplete_or_mismatched_receipt_checksum(
    monkeypatch,
    remote_receipt_checksum: str,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)
    coordinator = FakeDistributedCoordinator(
        sources[1].placement_id,
        remote_receipt_checksum=remote_receipt_checksum,
    )
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(WeightTransferError) as error_info:
        materialize_weights(
            source_placements=sources,
            source_bindings=bindings(sources, address_base=0x10000),
            payload_identity=payload_identity(sources),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/default/model/revision",
                object_prefix="weights/default/model/revision",
            ),
            provider=provider,
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert error_info.value.code == "BACKEND_FAILURE"
    assert error_info.value.completion_known is True
    assert [call[0] for call in store.calls].count("abort") == 1
    assert not any(call[0] == "commit" for call in store.calls)


@pytest.mark.parametrize(
    "field",
    [
        "model_id",
        "revision",
        "group_id",
        "manifest_key",
        "tensors",
        "fragment_id",
        "tensor_id",
        "global_offset",
        "local_shape",
        "object_key",
        "object_offset",
        "nbytes",
        "checksum",
    ],
)
def test_mooncake_store_rejects_committed_manifest_that_differs_from_upload_plan(
    monkeypatch,
    field: str,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = CorruptCommittedManifestStore(field)
    sources = placements("source", shard_dim=0)
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(
        WeightTransferError,
        match="committed manifest differs",
    ):
        materialize_weights(
            source_placements=sources,
            source_bindings=bindings(sources, address_base=0x10000),
            payload_identity=payload_identity(sources),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/default/model/revision",
                object_prefix="weights/default/model/revision",
            ),
            provider=provider,
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert [call[0] for call in store.calls].count("commit") == 1
    assert [call[0] for call in store.calls].count("finalize") == 1


def test_mooncake_store_snapshot_catalog_round_trip(monkeypatch) -> None:
    calls = {}
    backend = fake_backend(calls)
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)
    source_bindings = bindings(sources, address_base=0x10000)
    targets = placements("target", shard_dim=1)
    target_bindings = bindings(targets, address_base=0x20000)
    provider = MooncakeWeightStoreProvider(
        store,
        namespace="default",
        payload_checksum_verifier=payload_checksum_verifier(sources),
        source_pre_registered=True,
        target_pre_registered=True,
    )
    catalog = InMemoryWeightStorageCatalog()

    publication = materialize_weight_snapshot(
        source_placements=sources,
        source_bindings=source_bindings,
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        provider=provider,
        catalog=catalog,
        publication_id="mooncake-round-trip",
        attestor=ALLOW_ALL_ATTESTOR,
    )
    target_coordinator = WeightSnapshotCoordinator()
    lease_id, generation = target_coordinator.acquire_snapshot()
    original_target_binding = target_bindings[0]
    target_binding = WeightRuntimeBindingManifest(
        model_id=original_target_binding.model_id,
        revision=original_target_binding.revision,
        placement_id=original_target_binding.placement_id,
        instance_id=original_target_binding.instance_id,
        generation=generation,
        lease_id=lease_id,
        fragments=original_target_binding.fragments,
    )
    receipt = load_weight_snapshot(
        publication.snapshot.ref,
        catalog=catalog,
        target_placements=(targets[0],),
        target_bindings=(target_binding,),
        provider=provider,
        target_mode=WeightTargetLoadMode.LIVE_UPDATE,
        target_session=WeightTargetLoadSession(
            target_bindings=(target_binding,),
            owners=(store,),
            coordinator=target_coordinator,
        ),
        attestor=ALLOW_ALL_ATTESTOR,
    )

    assert publication.state is WeightSnapshotPublicationState.PUBLISHED
    assert (
        publication.snapshot.ref.manifest_key
        == "weights/default/model/revision/manifest"
    )
    assert receipt.total_bytes == 8
    assert store.calls[-2][0] == "load_manifest"
    assert store.calls[-1][0] == "load"


def test_mooncake_store_recovers_committed_manifest_without_reupload(
    monkeypatch,
) -> None:
    class FailCompleteOnceCatalog(InMemoryWeightStorageCatalog):
        def __init__(self):
            super().__init__()
            self.fail = True

        def complete_materialization(self, materialization_id, snapshot):
            if self.fail:
                self.fail = False
                raise RuntimeError("catalog completion unavailable")
            return super().complete_materialization(materialization_id, snapshot)

    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources = placements("source", shard_dim=0)
    source_bindings = bindings(sources, address_base=0x10000)
    provider = MooncakeWeightStoreProvider(
        store,
        namespace="default",
        payload_checksum_verifier=payload_checksum_verifier(sources),
        source_pre_registered=True,
    )
    catalog = FailCompleteOnceCatalog()
    destination = WeightStorageDestination(
        provider=provider.name,
        storage_id="weights/default/model/revision",
        object_prefix="weights/default/model/revision",
    )

    with pytest.raises(RuntimeError, match="catalog completion unavailable"):
        materialize_weight_snapshot(
            source_placements=sources,
            source_bindings=source_bindings,
            payload_identity=payload_identity(sources),
            destination=destination,
            provider=provider,
            catalog=catalog,
            publication_id="recover-committed",
            attestor=ALLOW_ALL_ATTESTOR,
        )

    publication = materialize_weight_snapshot(
        source_placements=sources,
        source_bindings=source_bindings,
        payload_identity=payload_identity(sources),
        destination=destination,
        provider=provider,
        catalog=catalog,
        publication_id="recover-committed",
        attestor=ALLOW_ALL_ATTESTOR,
    )

    assert publication.state is WeightSnapshotPublicationState.PUBLISHED
    assert [call[0] for call in store.calls].count("prepare_upload") == 1
    assert [call[0] for call in store.calls].count("upload") == len(sources)
    assert [call[0] for call in store.calls].count("commit") == 1
    assert [call[0] for call in store.calls].count("load_manifest") == 1


def test_mooncake_store_commit_response_loss_resolves_persisted_manifest(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore(lose_commit_response_once=True)
    sources = placements("source", shard_dim=0)
    source_bindings = bindings(sources, address_base=0x10000)
    provider = MooncakeWeightStoreProvider(
        store,
        namespace="default",
        payload_checksum_verifier=payload_checksum_verifier(sources),
        source_pre_registered=True,
    )
    catalog = InMemoryWeightStorageCatalog()
    destination = WeightStorageDestination(
        provider=provider.name,
        storage_id="weights/default/model/revision",
        object_prefix="weights/default/model/revision",
    )

    publication = materialize_weight_snapshot(
        source_placements=sources,
        source_bindings=source_bindings,
        payload_identity=payload_identity(sources),
        destination=destination,
        provider=provider,
        catalog=catalog,
        publication_id="commit-response-lost",
        attestor=ALLOW_ALL_ATTESTOR,
    )

    assert publication.state is WeightSnapshotPublicationState.PUBLISHED
    assert [call[0] for call in store.calls].count("prepare_upload") == 1
    assert [call[0] for call in store.calls].count("commit") == 1
    assert [call[0] for call in store.calls].count("load_manifest") == 1


def test_mooncake_store_routes_follow_alias_deduplicated_bound_regions(
    monkeypatch,
) -> None:
    calls = {}
    backend = fake_backend(calls)
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore()
    sources, target = route_alias_placements()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    materialized = materialize_weights(
        source_placements=sources,
        source_bindings=bindings(sources, address_base=0x10000),
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider="mooncake-store",
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        provider=provider,
        attestor=ALLOW_ALL_ATTESTOR,
    )
    target_binding = WeightRuntimeBindingManifest(
        model_id=target.model_id,
        revision=target.revision,
        placement_id=target.placement_id,
        instance_id="instance:target",
        generation=1,
        lease_id="lease:1",
        fragments=tuple(
            RuntimeWeightBinding(
                placement_fragment_id=tensor.placement_fragment_id,
                fragment_id=f"runtime:{tensor.placement_fragment_id}",
                address=(0x20000 if tensor.tensor_id == "shared" else 0x21000),
                nbytes=tensor.nbytes,
                storage_offset=0,
                device="cuda:0",
                is_contiguous=True,
                worker_id="target",
                endpoint="target:1",
            )
            for tensor in target.tensors
        ),
    )
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(sources, target),
        source_bindings=materialized.storage_bindings,
        target_bindings=(target_binding,),
    )

    assert len(request.plan.logical_plan.regions) == 3
    assert len(request.plan.regions) == 2

    execute_weight_load(
        request,
        provider=provider,
        target_mode=WeightTargetLoadMode.COLD_START,
        attestor=ALLOW_ALL_ATTESTOR,
    )

    assert [
        (route.source_pp, route.target_pp, route.operation_indices)
        for route in calls["logical"].pipeline_routes
    ] == [
        (1, 7, (0,)),
        (2, 7, (1,)),
    ]
    assert all(
        index < len(calls["logical"].operations)
        for route in calls["logical"].pipeline_routes
        for index in route.operation_indices
    )


def test_mooncake_store_aborts_failed_materialization(monkeypatch) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    store = FakeWeightStore(fail_upload=True)
    sources = placements("source", shard_dim=0)

    with pytest.raises(Exception, match="upload failed"):
        materialize_weights(
            source_placements=sources,
            source_bindings=bindings(sources, address_base=0x10000),
            payload_identity=payload_identity(sources),
            destination=WeightStorageDestination(
                provider="mooncake-store",
                storage_id="weights/default/model/revision",
                object_prefix="weights/default/model/revision",
            ),
            provider=MooncakeWeightStoreProvider(
                store,
                payload_checksum_verifier=payload_checksum_verifier(sources),
            ),
            attestor=ALLOW_ALL_ATTESTOR,
        )

    assert any(call[0] == "abort" for call in store.calls)
    assert not any(call[0] == "commit" for call in store.calls)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
