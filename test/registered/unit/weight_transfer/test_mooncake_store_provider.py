from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass, replace
from math import prod
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.weight_runtime_manifest import (
    RuntimeWeightBinding,
    WeightParallelRank,
    WeightPlacementManifest,
    WeightPlacementTensor,
    WeightRuntimeBindingManifest,
    WeightRuntimeManifestParts,
    WeightSnapshotCoordinator,
    compute_weight_placement_id,
)
from sglang.srt.weight_transfer import mooncake_store as mooncake_store_api
from sglang.srt.weight_transfer import runtime as runtime_api
from sglang.srt.weight_transfer.api import (
    execute_weight_load,
    execute_weight_materialization,
    load_weight_snapshot,
    materialize_weight_snapshot,
    materialize_weights,
    prepare_weight_load_from_plan,
    prepare_weight_materialization,
)
from sglang.srt.weight_transfer.contracts import RuntimeWeightLocation
from sglang.srt.weight_transfer.distributed import (
    WeightStoreDistributedError,
    WeightStoreUploadOutcome,
)
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
    WeightTransferCompletionUnknownError,
    WeightTransferError,
    WeightTransferExecutionContext,
    WeightTransferReleaseError,
)
from sglang.srt.weight_transfer.runtime import (
    RuntimeWeightPayloadHasher,
    RuntimeWeightSnapshotSource,
)
from sglang.srt.weight_transfer.storage import (
    InMemoryWeightStorageCatalog,
    WeightSnapshotPublicationState,
    weight_source_snapshot_digest,
)
from sglang.test.ci.ci_register import register_cpu_ci

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
        self.recovery_journal = {}
        self.recovery_journal_calls = []

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

    def put_recovery_journal_chunk(self, key, payload):
        self.recovery_journal_calls.append(("put", key, len(payload)))
        self.recovery_journal[key] = bytes(payload)

    def get_recovery_journal_chunk(self, key):
        self.recovery_journal_calls.append(("get", key))
        return self.recovery_journal.get(key)

    def delete_recovery_journal_chunk(self, key):
        self.recovery_journal_calls.append(("delete", key))
        self.recovery_journal.pop(key, None)


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


class OverwideUploadPlanStore(FakeWeightStore):
    def prepare_upload(self, source_manifests, *, namespace):
        plan = super().prepare_upload(source_manifests, namespace=namespace)
        return SimpleNamespace(
            manifest=plan.manifest,
            operations=(*plan.operations, plan.operations[0]),
            session_group_id=plan.session_group_id,
            control_key=plan.control_key,
        )


class FailingPreflightCleanupStore(FakeWeightStore):
    def abort_upload(self, plan, receipts):
        super().abort_upload(plan, receipts)
        raise FakeWeightStoreError("abort cleanup failed")

    def finalize_upload_session(self, plan):
        super().finalize_upload_session(plan)
        raise FakeWeightStoreError("finalize cleanup failed")


class NoFinalizePreflightStore(FakeWeightStore):
    finalize_upload_session = None


class BlockingUploadStore(FakeWeightStore):
    def __init__(self) -> None:
        super().__init__()
        self.upload_started = threading.Event()
        self.release_upload = threading.Event()

    def upload(self, plan, runtime_manifest, *, pre_registered):
        self.upload_started.set()
        assert self.release_upload.wait(timeout=5)
        return super().upload(
            plan,
            runtime_manifest,
            pre_registered=pre_registered,
        )


class BlockingCommitStore(FakeWeightStore):
    def __init__(self) -> None:
        super().__init__()
        self.commit_started = threading.Event()
        self.release_commit = threading.Event()

    def commit(self, plan, receipts):
        self.commit_started.set()
        assert self.release_commit.wait(timeout=5)
        return super().commit(plan, receipts)


class BlockingFinalizeStore(FakeWeightStore):
    def __init__(self) -> None:
        super().__init__()
        self.finalize_started = threading.Event()
        self.release_finalize = threading.Event()

    def finalize_upload_session(self, plan):
        self.finalize_started.set()
        assert self.release_finalize.wait(timeout=5)
        super().finalize_upload_session(plan)


class BlockingLoadManifestStore(FakeWeightStore):
    def __init__(self) -> None:
        super().__init__()
        self.load_started = threading.Event()
        self.release_load = threading.Event()

    def load_manifest(self, manifest_key):
        self.load_started.set()
        assert self.release_load.wait(timeout=5)
        return super().load_manifest(manifest_key)


class BlockingLoadStore(FakeWeightStore):
    def __init__(self) -> None:
        super().__init__()
        self.load_started = threading.Event()
        self.release_load = threading.Event()

    def load(self, plan, runtime_manifest, *, pre_registered):
        self.load_started.set()
        assert self.release_load.wait(timeout=5)
        return super().load(
            plan,
            runtime_manifest,
            pre_registered=pre_registered,
        )


class FailOnceFinalizeStore(FakeWeightStore):
    def __init__(self) -> None:
        super().__init__()
        self.finalize_attempts = 0

    def finalize_upload_session(self, plan):
        self.finalize_attempts += 1
        if self.finalize_attempts == 1:
            raise FakeWeightStoreError("transient finalize failure")
        super().finalize_upload_session(plan)


class PartialRecoveryJournalDeleteStore(FakeWeightStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_delete_once = True
        self.successful_delete_count = 0
        self.not_found_count = 0

    def delete_recovery_journal_chunk(self, key):
        self.recovery_journal_calls.append(("delete", key))
        if self.fail_delete_once and self.successful_delete_count == 1:
            self.fail_delete_once = False
            return -5
        if key not in self.recovery_journal:
            self.not_found_count += 1
            return -704
        self.recovery_journal.pop(key)
        self.successful_delete_count += 1
        return 0


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
        self.root_gathers = []
        self.root_scatters = []
        self.commit_result = None
        self.calls = []
        self.upload_execution_contexts = []

    def prepare_upload(self, factory, *, execution_context=None):
        self.calls.append("prepare")
        self.upload_plan = factory()
        return self.upload_plan

    def gather_object_to_root(self, value, *, phase, execution_context=None):
        del execution_context
        self.calls.append(("gather_root", phase))
        self.root_gathers.append(value)
        if isinstance(value, mooncake_store_api._RankRecoveryProjection):
            return replace(value, rank=0), replace(value, rank=1)
        if self.rank == 0:
            remote_manifests = tuple(
                item
                for item in value.runtime_manifests
                if item[0] == self.remote_placement_id
            )
            remote = replace(
                value,
                rank=1,
                runtime_manifests=remote_manifests,
                local_placement_ids=(self.remote_placement_id,),
            )
            return value, remote
        local_manifests = tuple(
            item
            for item in value.runtime_manifests
            if item[0] in value.local_placement_ids
        )
        root = replace(
            value,
            rank=0,
            local_placement_ids=(self.remote_placement_id,),
        )
        local = replace(
            value,
            rank=1,
            runtime_manifests=local_manifests,
        )
        return root, local

    def scatter_object_from_root(self, values, *, phase, execution_context=None):
        del execution_context
        self.calls.append(("scatter_root", phase))
        assert values is not None
        self.root_scatters.append(values)
        if isinstance(values[0], mooncake_store_api._RankRecoveryResult):
            return values[self.rank]
        self.upload_plan = values[0].upload_plan
        return values[self.rank]

    def exchange_preflight_outcome(self, outcome, *, execution_context=None):
        self.calls.append(("preflight", outcome))
        remote = type(outcome)(
            rank=1,
            error=self.remote_preflight_error,
        )
        return outcome, remote

    def exchange_upload_outcome(self, outcome, *, execution_context=None):
        self.upload_execution_contexts.append(execution_context)
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
                    "sha256:"
                    + hashlib.sha256(
                        operation.source.placement_fragment_id.encode("utf-8")
                    ).hexdigest()
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

    def run_root(
        self,
        phase,
        factory,
        *,
        discard_result=False,
        execution_context=None,
    ):
        del execution_context
        self.calls.append(("run_root", phase))
        try:
            result = factory()
        except BaseException as error:
            raise WeightStoreDistributedError(
                phase,
                str(error),
                completion_unknown=bool(getattr(error, "completion_unknown", False)),
            ) from error
        return None if discard_result else result

    def commit_upload(self, factory, *, execution_context=None):
        self.calls.append("commit")
        self.commit_result = factory()
        return self.commit_result

    def abort_upload(self, factory, *, execution_context=None):
        self.calls.append("abort")
        factory()

    def finalize_upload(self, factory, *, execution_context=None):
        self.calls.append("finalize")
        factory()


class TerminalOutcomeCoordinator(FakeDistributedCoordinator):
    def __init__(
        self,
        remote_placement_id: str,
        business_context: WeightTransferExecutionContext,
        *,
        check_upload: bool,
    ) -> None:
        super().__init__(remote_placement_id)
        self.business_context = business_context
        self.check_upload = check_upload
        self.poisoned = False
        self.terminal_contexts = []

    def _check_terminal_context(self, phase, execution_context) -> None:
        self.terminal_contexts.append((phase, execution_context))
        if (
            execution_context is None
            or execution_context is self.business_context
            or execution_context.expired()
        ):
            self.poisoned = True
            raise WeightStoreDistributedError(
                phase,
                "terminal outcome reused the expired business context",
                completion_unknown=True,
            )

    def exchange_upload_outcome(self, outcome, *, execution_context=None):
        if self.check_upload and outcome.completion_unknown:
            self._check_terminal_context(
                "exchange_upload_outcome",
                execution_context,
            )
        return super().exchange_upload_outcome(
            outcome,
            execution_context=execution_context,
        )

    def commit_upload(self, factory, *, execution_context=None):
        self.calls.append("commit")
        try:
            self.commit_result = factory()
        except BaseException as error:
            self._check_terminal_context("commit_upload", execution_context)
            raise WeightStoreDistributedError(
                "commit_upload",
                str(error),
                completion_unknown=bool(getattr(error, "completion_unknown", False)),
            ) from error
        self._check_terminal_context("commit_upload", execution_context)
        return self.commit_result


class ExpireBeforeCommitCoordinator(TerminalOutcomeCoordinator):
    def run_root(
        self,
        phase,
        factory,
        *,
        discard_result=False,
        execution_context=None,
    ):
        result = super().run_root(
            phase,
            factory,
            discard_result=discard_result,
            execution_context=execution_context,
        )
        if phase == "validate_upload":
            while not self.business_context.expired():
                time.sleep(0.001)
        return result


class RankDivergentCoordinator(FakeDistributedCoordinator):
    def __init__(self, remote_placement_id: str) -> None:
        super().__init__(remote_placement_id)
        self.rank = 1
        self.terminal_manifest = None
        self.store = None

    def exchange_preflight_outcome(self, outcome, *, execution_context=None):
        self.calls.append(("preflight", outcome))
        return type(outcome)(rank=0, error=None), outcome

    def exchange_upload_outcome(self, outcome, *, execution_context=None):
        self.calls.append(("exchange", outcome))
        return None

    def scatter_object_from_root(self, values, *, phase, execution_context=None):
        del execution_context
        self.calls.append(("scatter_root", phase))
        assert values is not None
        self.root_scatters.append(values)
        if isinstance(values[0], mooncake_store_api._RankRecoveryResult):
            return values[self.rank]
        self.upload_plan = values[0].upload_plan
        return values[0]

    def commit_upload(self, factory, *, execution_context=None):
        del factory, execution_context
        self.calls.append("commit")
        assert self.upload_plan is not None
        self.terminal_manifest = self.upload_plan.manifest
        raise WeightStoreDistributedError("commit_upload", "commit response lost")

    def run_root(
        self,
        phase,
        factory,
        *,
        discard_result=False,
        execution_context=None,
    ):
        del execution_context
        self.calls.append(("run_root", phase))
        if phase == "prepare_upload":
            result = factory()
            return None if discard_result else result
        if phase == "validate_upload":
            return None
        if phase == "recover_materialization":
            assert self.store is not None
            self.store.root_active = True
            try:
                result = factory()
            finally:
                self.store.root_active = False
            return None if discard_result else result
        del factory, discard_result
        assert self.terminal_manifest is not None
        return mooncake_store_api._StoreTerminalDecision(
            state=mooncake_store_api._StoreTerminalState.MANIFEST_MATCH,
            manifest=self.terminal_manifest,
        )


class FailingExchangeCoordinator(FakeDistributedCoordinator):
    def __init__(
        self,
        remote_placement_id: str,
        *,
        completion_unknown: bool,
    ) -> None:
        super().__init__(remote_placement_id)
        self.completion_unknown = completion_unknown

    def exchange_upload_outcome(self, outcome, *, execution_context=None):
        del outcome, execution_context
        self.calls.append("exchange_failed")
        raise WeightStoreDistributedError(
            "exchange_upload_outcome",
            "outcome exchange failed",
            completion_unknown=self.completion_unknown,
        )


class DivergentVisibilityStore(FakeWeightStore):
    def __init__(self):
        super().__init__()
        self.root_active = False

    def manifest_exists(self, manifest_key):
        if self.root_active:
            return super().manifest_exists(manifest_key)
        self.calls.append(("non_root_manifest_observation", manifest_key))
        raise AssertionError("non-root rank observed Store visibility")


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


def materialization_request(
    manifests: tuple[WeightPlacementManifest, ...],
    *,
    operation_id: str,
):
    return prepare_weight_materialization(
        source_placements=manifests,
        source_bindings=bindings(manifests, address_base=0x10000),
        payload_identity=payload_identity(manifests),
        destination=WeightStorageDestination(
            provider="mooncake-store",
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        operation_id=operation_id,
    )


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


class _ActiveSnapshotManager:
    def __init__(self, binding: WeightRuntimeBindingManifest) -> None:
        self.binding = binding
        self.attestations = 0
        self.released = False

    def attest_binding(self, binding: WeightRuntimeBindingManifest) -> None:
        assert binding == self.binding
        assert not self.released
        self.attestations += 1

    def has_lease(self, lease_id: str) -> bool:
        return lease_id == self.binding.lease_id and not self.released

    def release(self, lease_id: str) -> None:
        assert self.has_lease(lease_id)
        self.released = True


class _CountingPayloadHasher:
    def __init__(self, checksum: str) -> None:
        self.checksum = checksum
        self.calls = 0

    def __call__(self, _location: RuntimeWeightLocation) -> str:
        self.calls += 1
        return self.checksum


def _captured_runtime_source():
    source_placements = placements("source", shard_dim=0)
    source_bindings = bindings(source_placements, address_base=0x10000)
    placement = source_placements[0]
    binding = source_bindings[0]
    identity = payload_identity((placement,))
    checksum = identity.fragments[0].checksum
    hasher = _CountingPayloadHasher(checksum)
    manager = _ActiveSnapshotManager(binding)
    source = RuntimeWeightSnapshotSource(
        model=object(),
        manager=manager,
        parts=WeightRuntimeManifestParts(
            placement=placement,
            binding=binding,
        ),
        payload_hasher=hasher,
        payload_identity=identity,
    )
    return source, manager, hasher, source_placements, source_bindings


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


def test_captured_runtime_identity_reuses_active_lease_without_rehash(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    source, manager, hasher, source_placements, source_bindings = (
        _captured_runtime_source()
    )
    request = prepare_weight_materialization(
        source_placements=source_placements,
        source_bindings=source_bindings,
        payload_identity=payload_identity(source_placements),
        destination=WeightStorageDestination(
            provider="mooncake-store",
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        operation_id="reuse-captured-payload-identity",
    )
    provider = MooncakeWeightStoreProvider(
        FakeWeightStore(),
        local_placement_ids=(source.placement.placement_id,),
        payload_checksum_verifier=source.payload_checksum,
    )

    provider.prepare(request)

    assert manager.attestations == 1
    assert hasher.calls == 0


def test_captured_runtime_identity_rejects_drift_without_rehash(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    source, manager, hasher, source_placements, source_bindings = (
        _captured_runtime_source()
    )
    checksums = payload_checksums(source_placements)
    changed_fragment_id = source.placement.tensors[0].placement_fragment_id
    checksums[changed_fragment_id] = f"sha256:{hashlib.sha256(b'changed').hexdigest()}"
    changed_identity = WeightPayloadIdentity.create(
        source_placements,
        checksums,
    )
    request = prepare_weight_materialization(
        source_placements=source_placements,
        source_bindings=source_bindings,
        payload_identity=changed_identity,
        destination=WeightStorageDestination(
            provider="mooncake-store",
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        operation_id="reject-captured-payload-drift",
    )
    provider = MooncakeWeightStoreProvider(
        FakeWeightStore(),
        local_placement_ids=(source.placement.placement_id,),
        payload_checksum_verifier=source.payload_checksum,
    )

    with pytest.raises(
        WeightTransferError,
        match="payload identity",
    ):
        provider.prepare(request)

    assert manager.attestations == 1
    assert hasher.calls == 0


def test_runtime_payload_hash_stops_at_chunk_boundary_on_cancel() -> None:
    model = torch.nn.Linear(16, 1, bias=False)
    parameter = model.weight
    nbytes = parameter.numel() * parameter.element_size()
    location = RuntimeWeightLocation(
        placement_id="placement",
        placement_fragment_id="fragment",
        fragment_id="runtime-fragment",
        tensor_id="weight",
        address=parameter.data_ptr(),
        nbytes=nbytes,
        storage_offset=0,
        device="cpu",
        worker_id="worker",
        endpoint="local",
        generation=1,
        lease_id="lease",
        rank=WeightParallelRank(),
        global_offset=(0,),
        local_shape=(parameter.numel(),),
        aliases=("weight",),
    )

    class CancelAfterFirstChunk:
        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return self.checks >= 2

    cancel_signal = CancelAfterFirstChunk()
    execution_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 10,
        cancel_signal=cancel_signal,
    )

    with pytest.raises(TimeoutError, match="cancelled"):
        RuntimeWeightPayloadHasher(model, chunk_bytes=4)(
            location,
            execution_context=execution_context,
        )

    assert cancel_signal.checks == 2


@pytest.mark.parametrize("explicit_context", [False, True])
def test_runtime_capture_bounds_hash_execution(
    monkeypatch,
    explicit_context: bool,
) -> None:
    source, manager, _, _, _ = _captured_runtime_source()
    contexts = []

    class RecordingHasher:
        def __init__(self, _model, *, chunk_bytes):
            assert chunk_bytes == 1024

        def __call__(self, location, *, execution_context=None):
            contexts.append(execution_context)
            return payload_checksums((source.placement,))[
                location.placement_fragment_id
            ]

    manager.snapshot_parts = lambda **_kwargs: source.parts
    monkeypatch.setattr(
        runtime_api,
        "RuntimeWeightPayloadHasher",
        RecordingHasher,
    )
    execution_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 10,
    )
    started = time.time()

    captured = RuntimeWeightSnapshotSource.capture(
        model=object(),
        manager=manager,
        model_id="model",
        revision="revision",
        instance_id="instance",
        worker_id="worker",
        endpoint="local",
        lease_timeout_sec=None if explicit_context else 5,
        checksum_chunk_bytes=1024,
        execution_context=execution_context if explicit_context else None,
    )

    assert captured.payload_identity == source.payload_identity
    assert len(contexts) == 1
    if explicit_context:
        assert contexts == [execution_context]
    else:
        assert contexts[0] is not None
        assert started < contexts[0].deadline_unix_sec <= time.time() + 5


def test_payload_hash_deadline_is_checked_before_store_io(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    verifier_calls = 0

    def verifier(_location):
        nonlocal verifier_calls
        verifier_calls += 1
        return f"sha256:{'a' * 64}"

    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=verifier,
    )
    request = materialization_request(
        sources,
        operation_id="payload-hash-deadline",
    )

    with pytest.raises(WeightTransferError) as error_info:
        provider.prepare(
            request,
            execution_context=WeightTransferExecutionContext(
                deadline_unix_sec=time.time() - 1,
            ),
        )

    assert error_info.value.code == "DEADLINE_EXCEEDED"
    assert error_info.value.completion_known is True
    assert verifier_calls == 0
    assert store.calls == []


def test_payload_hash_cancel_is_checked_between_fragments(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    checksums = payload_checksums(sources)
    cancel_signal = threading.Event()
    verifier_calls = 0
    received_context = None

    def verifier(location, *, execution_context):
        nonlocal verifier_calls, received_context
        verifier_calls += 1
        received_context = execution_context
        cancel_signal.set()
        return checksums[location.placement_fragment_id]

    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=verifier,
    )
    request = materialization_request(
        sources,
        operation_id="payload-hash-cancel",
    )
    execution_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 10,
        cancel_signal=cancel_signal,
    )

    with pytest.raises(WeightTransferError) as error_info:
        provider.prepare(
            request,
            execution_context=execution_context,
        )

    assert error_info.value.code == "CANCELLED"
    assert error_info.value.completion_known is True
    assert verifier_calls == 1
    assert received_context is execution_context
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
        "gather_root",
        "run_root",
        "scatter_root",
        "preflight",
        "exchange",
        "run_root",
        "commit",
        "finalize",
        "run_root",
    ]


def test_mooncake_store_scatters_rank_local_plans_and_compact_commit(
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
    coordinator = FakeDistributedCoordinator(sources[1].placement_id)
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

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

    assert len(coordinator.root_gathers) == 1
    packets = coordinator.root_scatters[0]
    assert len(packets[0].upload_plan.operations) == 2
    assert len(packets[1].upload_plan.operations) == 1
    assert len(packets[1].upload_plan.manifest.fragments) == len(
        packets[0].upload_plan.manifest.fragments
    )
    assert sum(len(packet.upload_plan.operations) for packet in packets) == 3
    assert isinstance(
        coordinator.commit_result,
        mooncake_store_api._StoreCommitDescriptor,
    )
    assert not hasattr(coordinator.commit_result, "fragments")
    assert not hasattr(coordinator.commit_result, "manifest")
    assert coordinator.commit_result.operation_count == 2
    assert coordinator.commit_result.fragment_count == 2


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


def test_known_distributed_exchange_failure_aborts_upload(
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
    coordinator = FailingExchangeCoordinator(
        sources[1].placement_id,
        completion_unknown=False,
    )
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(WeightTransferError) as raised:
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

    assert raised.value.phase == "exchange"
    assert raised.value.completion_known is True
    assert [call[0] for call in store.calls].count("abort") == 1
    assert not provider._pending_materializations


def test_unknown_distributed_exchange_failure_retains_recovery_state(
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
    coordinator = FailingExchangeCoordinator(
        sources[1].placement_id,
        completion_unknown=True,
    )
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
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

    assert raised.value.phase == "exchange"
    assert raised.value.completion_ticket is not None
    assert not any(call[0] == "abort" for call in store.calls)
    assert provider._pending_materializations


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
    ] == [
        "gather_root",
        "run_root",
        "scatter_root",
        "preflight",
        "abort",
        "finalize",
        "run_root",
    ]


def test_distributed_preflight_cleanup_always_enters_finalize_collective(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = NoFinalizePreflightStore()
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

    with pytest.raises(WeightTransferError, match="remote upload plan mismatch"):
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

    assert [
        call if isinstance(call, str) else call[0] for call in coordinator.calls
    ] == [
        "gather_root",
        "run_root",
        "scatter_root",
        "preflight",
        "abort",
        "finalize",
        "run_root",
    ]


def test_mooncake_store_attach_failure_is_gathered_before_cleanup(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    coordinator = FakeDistributedCoordinator(sources[1].placement_id)
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    def fail_attach(_request, _upload_plan):
        raise ValueError("payload identity attach failed")

    monkeypatch.setattr(provider, "_attach_payload_identity", fail_attach)

    with pytest.raises(
        WeightTransferError,
        match="payload identity attach failed",
    ) as raised:
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
    assert [call[0] for call in store.calls] == [
        "prepare_upload",
        "abort",
        "finalize",
    ]
    assert [
        call if isinstance(call, str) else call[0] for call in coordinator.calls
    ] == [
        "gather_root",
        "run_root",
        "scatter_root",
        "preflight",
        "abort",
        "finalize",
    ]


def test_mooncake_store_ticket_failure_is_gathered_before_cleanup(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    coordinator = FakeDistributedCoordinator(sources[1].placement_id)
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    def fail_ticket(_prepared, *, execution_context=None):
        del execution_context
        raise ValueError("ticket encoding failed")

    monkeypatch.setattr(provider, "_build_recovery_ticket", fail_ticket)

    with pytest.raises(WeightTransferError, match="ticket encoding failed") as raised:
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
    assert [call[0] for call in store.calls] == [
        "prepare_upload",
        "abort",
        "finalize",
    ]
    assert [
        call if isinstance(call, str) else call[0] for call in coordinator.calls
    ] == [
        "gather_root",
        "run_root",
        "scatter_root",
        "preflight",
        "abort",
        "finalize",
    ]


def test_mooncake_store_final_validation_failure_is_gathered_before_cleanup(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    coordinator = FakeDistributedCoordinator(sources[1].placement_id)
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    built_tickets = []
    build_ticket = provider._build_recovery_ticket

    def record_ticket(prepared, *, execution_context=None):
        ticket = build_ticket(
            prepared,
            execution_context=execution_context,
        )
        built_tickets.append(ticket)
        return ticket

    def fail_validation(*_args):
        raise ValueError("upload plan validation failed")

    monkeypatch.setattr(provider, "_build_recovery_ticket", record_ticket)
    monkeypatch.setattr(provider, "_validate_upload_plan", fail_validation)
    exposed_tickets = []
    request = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=bindings(sources, address_base=0x10000),
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
    )

    with pytest.raises(
        WeightTransferError,
        match="upload plan validation failed",
    ) as raised:
        execute_weight_materialization(
            request,
            provider=provider,
            attestor=ALLOW_ALL_ATTESTOR,
            completion_ticket_sink=exposed_tickets.append,
        )

    assert built_tickets
    assert exposed_tickets == []
    assert raised.value.completion_known is True
    assert not any(call[0] in {"upload", "commit"} for call in store.calls)
    assert [
        call if isinstance(call, str) else call[0] for call in coordinator.calls
    ] == [
        "gather_root",
        "run_root",
        "scatter_root",
        "preflight",
        "abort",
        "finalize",
        "run_root",
    ]


@pytest.mark.parametrize(
    ("failure_stage", "error_type", "completion_known", "has_ticket"),
    [
        ("ticket", WeightTransferError, True, False),
        ("validation", WeightTransferCompletionUnknownError, False, True),
    ],
)
def test_mooncake_store_preflight_cleanup_preserves_completion_state(
    monkeypatch,
    failure_stage: str,
    error_type: type[WeightTransferError],
    completion_known: bool,
    has_ticket: bool,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FailingPreflightCleanupStore()
    coordinator = FakeDistributedCoordinator(sources[1].placement_id)
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    if failure_stage == "ticket":

        def fail_ticket(_prepared):
            raise ValueError("ticket encoding failed")

        monkeypatch.setattr(provider, "_build_recovery_ticket", fail_ticket)
    else:

        def fail_validation(*_args):
            raise ValueError("upload plan validation failed")

        monkeypatch.setattr(provider, "_validate_upload_plan", fail_validation)

    with pytest.raises(error_type) as raised:
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

    assert raised.value.completion_known is completion_known
    assert (getattr(raised.value, "completion_ticket", None) is not None) is has_ticket
    assert "abort cleanup failed" in str(raised.value)
    assert "finalize cleanup failed" in str(raised.value)
    assert not any(call[0] in {"upload", "commit"} for call in store.calls)
    assert [
        call if isinstance(call, str) else call[0] for call in coordinator.calls
    ] == [
        "gather_root",
        "run_root",
        "scatter_root",
        "preflight",
        "abort",
        "finalize",
    ]


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
    catalog = InMemoryWeightStorageCatalog()

    with pytest.raises(
        WeightTransferCompletionUnknownError,
        match="committed manifest differs",
    ) as raised:
        materialize_weight_snapshot(
            source_placements=sources,
            source_bindings=bindings(sources, address_base=0x10000),
            payload_identity=payload_identity(sources),
            destination=WeightStorageDestination(
                provider=provider.name,
                storage_id="weights/default/model/revision",
                object_prefix="weights/default/model/revision",
            ),
            provider=provider,
            catalog=catalog,
            publication_id=f"committed-manifest-mismatch:{field}",
            attestor=ALLOW_ALL_ATTESTOR,
        )

    attempt = catalog.get_materialization(f"committed-manifest-mismatch:{field}")
    assert attempt is not None
    assert attempt.completion_ticket == raised.value.completion_ticket
    assert attempt.recoverable
    assert catalog.get_publication(f"committed-manifest-mismatch:{field}") is None
    assert [call[0] for call in store.calls].count("commit") == 1
    assert [call[0] for call in store.calls].count("finalize") == 0


def test_mooncake_store_rejects_backend_operation_expansion_before_upload(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = OverwideUploadPlanStore()
    provider = MooncakeWeightStoreProvider(
        store,
        max_total_operations=len(sources),
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(WeightTransferError, match="operation limit") as raised:
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

    assert raised.value.code == "UNSUPPORTED_CAPABILITY"
    assert [call[0] for call in store.calls] == ["prepare_upload", "abort", "finalize"]
    assert not any(call[0] == "upload" for call in store.calls)


@pytest.mark.parametrize("operation_count", [65_520, 65_521, 73_728])
def test_mooncake_recovery_ticket_is_constant_size_for_large_plans(
    monkeypatch,
    operation_count: int,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    prepared = provider.prepare(
        materialization_request(
            sources,
            operation_id=f"constant-ticket:{operation_count}",
        )
    )
    small_ticket = prepared.recovery_ticket
    assert small_ticket is not None
    operation = prepared.upload_plan.operations[0]
    prepared = replace(
        prepared,
        upload_plan=SimpleNamespace(
            **{
                **vars(prepared.upload_plan),
                "operations": (operation,) * operation_count,
            }
        ),
        recovery_ticket=None,
    )

    ticket = provider._build_recovery_ticket(prepared)
    reference = provider._decode_recovery_ticket(ticket)

    assert reference["version"] == 2
    assert set(reference) == {
        "format",
        "version",
        "provider",
        "operation_id",
        "journal_key",
        "manifest_key",
        "generation",
        "request_digest",
        "manifest_digest",
        "journal_digest",
    }
    assert len(ticket) < 2_048
    assert abs(len(ticket) - len(small_ticket)) <= 32


def test_recovery_journal_rollback_is_bounded_by_prepare_deadline(
    monkeypatch,
) -> None:
    rollback_started = threading.Event()
    release_rollback = threading.Event()

    class BlockingRollbackStore(FakeWeightStore):
        def put_recovery_journal_chunk(self, key, payload):
            super().put_recovery_journal_chunk(key, payload)
            if len(self.recovery_journal_calls) == 2:
                raise RuntimeError("journal write failed")

        def delete_recovery_journal_chunk(self, key):
            rollback_started.set()
            release_rollback.wait(timeout=5)
            super().delete_recovery_journal_chunk(key)

    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    provider = MooncakeWeightStoreProvider(
        BlockingRollbackStore(),
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    errors = []

    def prepare():
        try:
            provider.prepare(
                materialization_request(
                    sources,
                    operation_id="journal-bounded-rollback",
                ),
                execution_context=WeightTransferExecutionContext(
                    deadline_unix_sec=time.time() + 0.1,
                ),
            )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=prepare)
    worker.start()
    try:
        assert rollback_started.wait(timeout=1)
        worker.join(timeout=0.5)
        assert not worker.is_alive()
    finally:
        release_rollback.set()
        worker.join(timeout=2)

    assert errors


def test_preflight_cleanup_uses_terminal_context_after_deadline(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    request = materialization_request(
        sources,
        operation_id="expired-preflight-cleanup",
    )
    prepared = provider.prepare(request)
    store.calls.clear()

    provider._cleanup_failed_preflight(
        request,
        prepared.upload_plan,
        ("preflight validation failed",),
        completion_ticket=None,
        execution_context=WeightTransferExecutionContext(
            deadline_unix_sec=time.time() - 1,
        ),
    )

    assert [call[0] for call in store.calls] == ["abort", "finalize"]


def test_recovery_journal_removes_key_when_store_response_is_lost(
    monkeypatch,
) -> None:
    class ResponseLostStore(FakeWeightStore):
        def put_recovery_journal_chunk(self, key, payload):
            super().put_recovery_journal_chunk(key, payload)
            raise RuntimeError("journal response lost after commit")

    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = ResponseLostStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(RuntimeError, match="response lost"):
        provider.prepare(
            materialization_request(
                sources,
                operation_id="journal-response-lost",
            )
        )

    assert store.recovery_journal == {}
    assert any(call[0] == "delete" for call in store.recovery_journal_calls)


@pytest.mark.parametrize(
    "corruption",
    ["missing", "chunk", "generation", "digest", "terminal_state"],
)
def test_mooncake_recovery_journal_corruption_fails_closed(
    monkeypatch,
    corruption: str,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    request = materialization_request(
        sources,
        operation_id=f"journal-corruption:{corruption}",
    )
    prepared = provider.prepare(request)
    ticket = prepared.recovery_ticket
    assert ticket is not None
    reference = provider._decode_recovery_ticket(ticket)
    assert reference["version"] == 2
    index_key = f"{reference['journal_key']}/index"
    if corruption == "missing":
        store.recovery_journal.pop(index_key)
    elif corruption == "chunk":
        chunk_key = next(key for key in store.recovery_journal if "/records/" in key)
        store.recovery_journal[chunk_key] += b"corrupt"
    else:
        index = json.loads(store.recovery_journal[index_key])
        if corruption == "generation":
            index["generation"] += 1
        elif corruption == "digest":
            index["journal_digest"] = f"sha256:{'0' * 64}"
        else:
            index["terminal_state"] = "invalid"
        store.recovery_journal[index_key] = json.dumps(
            index,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    store.calls.clear()

    with pytest.raises(WeightTransferError) as raised:
        provider.recover_materialization(
            request,
            completion_ticket=ticket,
        )

    assert raised.value.code == "INVALID_COMPLETION_TICKET"
    assert store.calls == []


def test_mooncake_store_cleans_recovery_journal_after_success(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

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

    assert any(call[0] == "put" for call in store.recovery_journal_calls)
    assert any(call[0] == "delete" for call in store.recovery_journal_calls)
    assert store.recovery_journal == {}


def test_mooncake_recovery_journal_read_error_preserves_recovery_entry(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    request = materialization_request(
        sources,
        operation_id="journal-read-error",
    )
    prepared = provider.prepare(request)
    ticket = prepared.recovery_ticket
    assert ticket is not None
    journal_before = dict(store.recovery_journal)
    store.recovery_journal_calls.clear()

    def fail_read(key):
        store.recovery_journal_calls.append(("get", key))
        return -5, b""

    monkeypatch.setattr(store, "get_recovery_journal_chunk", fail_read)

    with pytest.raises(
        WeightTransferCompletionUnknownError,
        match="recovery journal read failed.*status -5",
    ) as raised:
        provider.recover_materialization(
            request,
            completion_ticket=ticket,
        )

    assert raised.value.completion_ticket == ticket
    assert store.recovery_journal == journal_before
    assert any(call[0] == "get" for call in store.recovery_journal_calls)
    assert not any(call[0] == "delete" for call in store.recovery_journal_calls)


def test_mooncake_store_retries_partially_deleted_recovery_journal(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = PartialRecoveryJournalDeleteStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    request = materialization_request(
        sources,
        operation_id="journal-partial-delete",
    )
    prepared = provider.prepare(request)
    ticket = prepared.recovery_ticket
    assert ticket is not None

    with pytest.raises(ValueError, match="journal delete failed"):
        provider.discard_materialization_recovery(
            request,
            completion_ticket=ticket,
        )

    assert store.successful_delete_count == 1
    assert store.recovery_journal

    provider.discard_materialization_recovery(
        request,
        completion_ticket=ticket,
    )

    assert store.not_found_count == 1
    assert store.recovery_journal == {}


def test_mooncake_store_post_prepare_validation_is_gathered_before_cleanup(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = OverwideUploadPlanStore()
    coordinator = FakeDistributedCoordinator(sources[1].placement_id)
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        max_total_operations=len(sources),
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    with pytest.raises(WeightTransferError, match="operation limit") as raised:
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
    assert [call[0] for call in store.calls] == [
        "prepare_upload",
        "abort",
        "finalize",
    ]
    assert [
        call if isinstance(call, str) else call[0] for call in coordinator.calls
    ] == [
        "gather_root",
        "run_root",
        "scatter_root",
        "preflight",
        "abort",
        "finalize",
    ]


def test_mooncake_recovery_rejects_operation_count_before_store_io(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    source_bindings = bindings(sources, address_base=0x10000)
    store = FakeWeightStore()
    request = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=source_bindings,
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider="mooncake-store",
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        operation_id="operation-limit-recovery",
    )
    preparing_provider = MooncakeWeightStoreProvider(
        store,
        max_total_operations=len(sources),
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    prepared = preparing_provider.prepare(request)
    ticket = preparing_provider.materialization_recovery_ticket(prepared)
    assert ticket is not None
    store.calls.clear()

    recovering_provider = MooncakeWeightStoreProvider(
        store,
        max_total_operations=len(sources) - 1,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    with pytest.raises(WeightTransferError, match="operation limit") as raised:
        recovering_provider.recover_materialization(
            request,
            completion_ticket=ticket,
        )

    assert raised.value.code == "INVALID_COMPLETION_TICKET"
    assert store.calls == []


def test_mooncake_recovery_ticket_records_source_snapshot_digest(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    source_bindings = bindings(sources, address_base=0x10000)
    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    destination = WeightStorageDestination(
        provider=provider.name,
        storage_id="weights/default/model/revision",
        object_prefix="weights/default/model/revision",
    )
    request = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=source_bindings,
        payload_identity=payload_identity(sources),
        destination=destination,
        operation_id="source-snapshot-recovery",
    )
    prepared = provider.prepare(request)
    ticket = provider.materialization_recovery_ticket(prepared)
    assert ticket is not None
    changed_bindings = tuple(
        WeightRuntimeBindingManifest(
            model_id=item.model_id,
            revision=item.revision,
            placement_id=item.placement_id,
            instance_id=item.instance_id,
            generation=item.generation + 1,
            lease_id="lease:2",
            fragments=item.fragments,
        )
        for item in source_bindings
    )
    reference = provider._decode_recovery_ticket(ticket)
    record = provider._load_recovery_journal(
        request,
        reference,
        execution_context=None,
    )

    assert record["source_snapshot_digest"] == weight_source_snapshot_digest(
        sources,
        source_bindings,
    )
    assert record["source_snapshot_digest"] != weight_source_snapshot_digest(
        sources,
        changed_bindings,
    )


def test_mooncake_store_decodes_legacy_inline_recovery_ticket(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    request = materialization_request(
        sources,
        operation_id="legacy-inline-ticket",
    )
    prepared = provider.prepare(request)
    ticket = prepared.recovery_ticket
    assert ticket is not None
    reference = provider._decode_recovery_ticket(ticket)
    legacy_record = provider._load_recovery_journal(
        request,
        reference,
        execution_context=None,
    )

    legacy_ticket = provider._encode_recovery_ticket(legacy_record)
    decoded = provider._decode_recovery_ticket(legacy_ticket)

    assert decoded["version"] == 1
    assert decoded["operation_id"] == request.operation_id
    assert len(decoded["operations"]) == len(prepared.upload_plan.operations)
    assert len(decoded["receipts"]) == len(prepared.upload_plan.operations)


def test_mooncake_recovery_rejects_changed_source_snapshot_before_store_io(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    source_bindings = bindings(sources, address_base=0x10000)
    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    destination = WeightStorageDestination(
        provider=provider.name,
        storage_id="weights/default/model/revision",
        object_prefix="weights/default/model/revision",
    )
    request = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=source_bindings,
        payload_identity=payload_identity(sources),
        destination=destination,
        operation_id="changed-source-snapshot-recovery",
    )
    prepared = provider.prepare(request)
    ticket = provider.materialization_recovery_ticket(prepared)
    assert ticket is not None
    store.calls.clear()

    def track_manifest_exists(manifest_key):
        store.calls.append(("manifest_exists", manifest_key))
        return False

    store.manifest_exists = track_manifest_exists
    changed_request = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=tuple(
            WeightRuntimeBindingManifest(
                model_id=binding.model_id,
                revision=binding.revision,
                placement_id=binding.placement_id,
                instance_id=binding.instance_id,
                generation=binding.generation + 1,
                lease_id="lease:2",
                fragments=binding.fragments,
            )
            for binding in source_bindings
        ),
        payload_identity=payload_identity(sources),
        destination=destination,
        operation_id=request.operation_id,
    )

    with pytest.raises(WeightTransferError) as raised:
        provider.recover_materialization(
            changed_request,
            completion_ticket=ticket,
        )

    assert raised.value.code == "INVALID_COMPLETION_TICKET"
    assert store.calls == []


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


def test_distributed_commit_recovery_uses_only_root_terminal_observation(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    coordinator = RankDivergentCoordinator(sources[0].placement_id)
    store = DivergentVisibilityStore()
    coordinator.store = store
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[1].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )

    receipt = materialize_weights(
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

    assert receipt.fragment_count == len(sources)
    assert (
        "run_root",
        "materialization.commit.observe_manifest",
    ) in coordinator.calls
    assert not any(call[0] == "non_root_manifest_observation" for call in store.calls)


def test_distributed_ticket_recovery_uses_only_root_terminal_observation(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    coordinator = RankDivergentCoordinator(sources[0].placement_id)
    store = DivergentVisibilityStore()
    coordinator.store = store
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[1].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    request = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=bindings(sources, address_base=0x10000),
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        operation_id="rank-divergent-recovery",
    )
    prepared = provider.prepare(request)
    completion_ticket = provider.materialization_recovery_ticket(prepared)
    assert completion_ticket is not None
    coordinator.terminal_manifest = prepared.upload_plan.manifest
    store.persisted = prepared.upload_plan.manifest

    receipt = provider.recover_materialization(
        request,
        completion_ticket=completion_ticket,
    )

    assert receipt is not None
    assert receipt.fragment_count == len(sources)
    assert (
        "run_root",
        "recover_materialization",
    ) in coordinator.calls
    assert not any(call[0] == "non_root_manifest_observation" for call in store.calls)


def test_recovery_keeps_ticket_when_conflict_cleanup_cannot_be_reconstructed(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    request = prepare_weight_materialization(
        source_placements=sources,
        source_bindings=bindings(sources, address_base=0x10000),
        payload_identity=payload_identity(sources),
        destination=WeightStorageDestination(
            provider=provider.name,
            storage_id="weights/default/model/revision",
            object_prefix="weights/default/model/revision",
        ),
        operation_id="conflict-reconstruction",
    )
    prepared = provider.prepare(request)
    completion_ticket = provider.materialization_recovery_ticket(prepared)
    assert completion_ticket is not None
    store.persisted = replace(
        prepared.upload_plan.manifest,
        revision="conflicting-revision",
    )
    store.calls.clear()

    def fail_reconstruction(*args, **kwargs):
        raise ValueError("reconstruction failed")

    monkeypatch.setattr(
        provider,
        "_reconstruct_recovery_plan",
        fail_reconstruction,
    )

    with pytest.raises(WeightTransferCompletionUnknownError) as raised:
        provider.recover_materialization(
            request,
            completion_ticket=completion_ticket,
        )

    assert raised.value.completion_ticket == completion_ticket
    assert [call[0] for call in store.calls] == ["load_manifest"]


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


def test_mooncake_store_bounded_execution_requires_local_prepare_opt_in(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    request = materialization_request(
        sources,
        operation_id="untrusted-prepare-upload",
    )

    assert provider.probe(request).supports_safe_cancel is False

    with pytest.raises(WeightTransferError) as raised:
        execute_weight_materialization(
            request,
            provider=provider,
            attestor=ALLOW_ALL_ATTESTOR,
            execution_context=WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + 1
            ),
        )

    assert raised.value.code == "UNBOUNDED_PROVIDER"
    assert store.calls == []


@pytest.mark.parametrize(
    ("store_type", "expected_phase"),
    [
        (BlockingLoadManifestStore, "load.prepare_manifest"),
        (BlockingLoadStore, "load"),
    ],
)
def test_mooncake_store_load_obeys_execution_deadline(
    monkeypatch,
    store_type,
    expected_phase: str,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = store_type()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
    )
    materialized = materialize_weights(
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
    target = placements("target", shard_dim=1)[0]
    target_binding = bindings((target,), address_base=0x20000)[0]
    request = prepare_weight_load_from_plan(
        plan_weight_transfer_to_local_target(sources, target),
        source_bindings=materialized.storage_bindings,
        target_bindings=(target_binding,),
    )

    started = time.monotonic()
    try:
        with pytest.raises(WeightTransferCompletionUnknownError):
            execute_weight_load(
                request,
                provider=provider,
                target_mode=WeightTargetLoadMode.COLD_START,
                attestor=ALLOW_ALL_ATTESTOR,
                execution_context=WeightTransferExecutionContext(
                    deadline_unix_sec=time.time() + 0.05,
                ),
            )
        assert time.monotonic() - started < 0.5
        assert store.load_started.is_set()
        pending = provider.pending_native_calls()
        assert len(pending) == 1
        assert pending[0].operation_id == request.operation_id
        assert pending[0].phase == expected_phase
    finally:
        store.release_load.set()

    drained = provider.drain_pending_calls(timeout_ms=1_000)
    assert len(drained) == 1
    assert drained[0].state.value == "succeeded"
    assert provider.pending_native_calls() == ()


def test_mooncake_store_drain_reports_failed_native_call() -> None:
    release_call = threading.Event()
    provider = MooncakeWeightStoreProvider(
        FakeWeightStore(),
        receipt_exchange=lambda _plan, receipts: receipts,
    )

    def fail_after_release():
        assert release_call.wait(timeout=5)
        raise FakeWeightStoreError("native upload failed")

    call = provider._get_or_start_native_call(
        "failed-native-call",
        "upload",
        fail_after_release,
    )
    assert call.owner is provider
    assert provider.drain_pending_calls(timeout_ms=0)[0].state.value == "pending"

    release_call.set()
    drained = provider.drain_pending_calls(timeout_ms=1000)

    assert len(drained) == 1
    assert drained[0].state.value == "failed"
    assert drained[0].error == "FakeWeightStoreError: native upload failed"
    assert call.owner is None
    assert provider.pending_native_calls() == ()


def test_deferred_recovery_cleanup_retires_late_native_call() -> None:
    call_started = threading.Event()
    release_call = threading.Event()
    provider = MooncakeWeightStoreProvider(
        FakeWeightStore(),
        receipt_exchange=lambda _plan, receipts: receipts,
    )

    def block_until_release():
        call_started.set()
        assert release_call.wait(timeout=5)

    call = provider._get_or_start_native_call(
        "deferred-cleanup",
        "upload",
        block_until_release,
    )
    assert call_started.wait(timeout=1)
    provider._defer_recovery_journal_cleanup(
        "deferred-cleanup",
        ("journal-key",),
    )
    release_call.set()

    cleanup_done = provider._deferred_recovery_cleanup_events["deferred-cleanup"]
    assert cleanup_done.wait(timeout=2)
    assert call.owner is None
    assert provider.pending_native_calls() == ()

    replacement = provider._get_or_start_native_call(
        "deferred-cleanup",
        "upload",
        lambda: None,
    )
    assert replacement is not call
    assert replacement.done.wait(timeout=1)


def test_mooncake_store_seal_rejects_new_native_calls_while_pending() -> None:
    call_started = threading.Event()
    release_call = threading.Event()
    provider = MooncakeWeightStoreProvider(
        FakeWeightStore(),
        receipt_exchange=lambda _plan, receipts: receipts,
    )

    def block_until_release():
        call_started.set()
        assert release_call.wait(timeout=5)

    provider._get_or_start_native_call(
        "pending-before-close",
        "upload",
        block_until_release,
    )
    assert call_started.wait(timeout=1)

    try:
        pending = provider.seal_native_calls_for_close()
        assert len(pending) == 1
        assert pending[0].state.value == "pending"
        with pytest.raises(RuntimeError, match="closed to native calls"):
            provider._get_or_start_native_call(
                "rejected-after-close",
                "load",
                lambda: None,
            )
    finally:
        release_call.set()

    drained = provider.drain_pending_calls(timeout_ms=1_000)
    assert len(drained) == 1
    assert drained[0].state.value == "succeeded"
    assert provider.pending_native_calls() == ()


@pytest.mark.parametrize(
    ("store_type", "started_attr", "release_attr", "call_attr"),
    [
        (BlockingUploadStore, "upload_started", "release_upload", "local_upload_call"),
        (BlockingCommitStore, "commit_started", "release_commit", "local_commit_call"),
    ],
)
def test_mooncake_store_data_call_obeys_execution_deadline(
    monkeypatch,
    store_type,
    started_attr: str,
    release_attr: str,
    call_attr: str,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = store_type()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
        prepare_upload_is_local=True,
    )
    request = materialization_request(
        sources,
        operation_id=f"bounded-{call_attr}",
    )

    started = time.monotonic()
    try:
        with pytest.raises(WeightTransferCompletionUnknownError) as raised:
            execute_weight_materialization(
                request,
                provider=provider,
                attestor=ALLOW_ALL_ATTESTOR,
                execution_context=WeightTransferExecutionContext(
                    deadline_unix_sec=time.time() + 0.05
                ),
            )
        assert time.monotonic() - started < 0.5
        assert raised.value.completion_ticket
        assert getattr(store, started_attr).is_set()
        recovery_started = time.monotonic()
        with pytest.raises(WeightTransferCompletionUnknownError):
            provider.recover_materialization(
                request,
                completion_ticket=raised.value.completion_ticket,
                execution_context=WeightTransferExecutionContext(
                    deadline_unix_sec=time.time() + 0.05
                ),
            )
        assert time.monotonic() - recovery_started < 0.5
        pending = provider.pending_native_calls()
        assert len(pending) == 1
        assert pending[0].operation_id == request.operation_id
        assert pending[0].phase == call_attr.removeprefix("local_").removesuffix(
            "_call"
        )
        assert pending[0].state.value == "pending"
    finally:
        getattr(store, release_attr).set()

    drained = provider.drain_pending_calls(timeout_ms=1000)
    assert len(drained) == 1
    assert drained[0].operation_id == request.operation_id
    assert drained[0].state.value == "succeeded"
    assert provider.pending_native_calls() == ()
    submission = provider._pending_materializations[request.operation_id]
    call = getattr(submission, call_attr)
    assert call is not None
    assert call.done.wait(timeout=1)
    recovered = provider.recover_materialization(
        request,
        completion_ticket=raised.value.completion_ticket,
        execution_context=WeightTransferExecutionContext(
            deadline_unix_sec=time.time() + 1
        ),
    )
    assert recovered is not None
    assert recovered.operation_id == request.operation_id
    assert request.operation_id not in provider._pending_materializations


def test_mooncake_store_finalize_timeout_preserves_committed_receipt(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = BlockingFinalizeStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
        prepare_upload_is_local=True,
    )
    request = materialization_request(
        sources,
        operation_id="bounded-finalize",
    )

    started = time.monotonic()
    try:
        with pytest.raises(WeightTransferReleaseError) as raised:
            execute_weight_materialization(
                request,
                provider=provider,
                attestor=ALLOW_ALL_ATTESTOR,
                execution_context=WeightTransferExecutionContext(
                    deadline_unix_sec=time.time() + 0.05
                ),
            )
        assert time.monotonic() - started < 0.5
        assert raised.value.receipt is not None
        assert raised.value.receipt.operation_id == request.operation_id
        assert getattr(
            raised.value.release_error,
            "completion_unknown",
            False,
        )
        assert store.persisted is not None
        assert store.finalize_started.is_set()
        pending = provider.pending_native_calls()
        assert len(pending) == 1
        assert pending[0].operation_id == request.operation_id
        assert pending[0].phase == "finalize"
        assert pending[0].state.value == "pending"
    finally:
        store.release_finalize.set()

    drained = provider.drain_pending_calls(timeout_ms=1000)
    assert len(drained) == 1
    assert drained[0].state.value == "succeeded"
    assert provider.pending_native_calls() == ()
    call = provider._finalize_calls[request.operation_id]
    assert call.done.wait(timeout=1)
    completion_ticket = raised.value.receipt.completion_ticket
    assert completion_ticket is not None
    recovered = provider.recover_materialization(
        request,
        completion_ticket=completion_ticket,
    )
    assert recovered is not None
    assert [call[0] for call in store.calls].count("finalize") == 1
    assert request.operation_id not in provider._finalize_calls


@pytest.mark.parametrize("bounded_recovery", [False, True])
def test_mooncake_store_retries_failed_bounded_finalize_during_recovery(
    monkeypatch,
    bounded_recovery: bool,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FailOnceFinalizeStore()
    provider = MooncakeWeightStoreProvider(
        store,
        payload_checksum_verifier=payload_checksum_verifier(sources),
        prepare_upload_is_local=True,
    )
    request = materialization_request(
        sources,
        operation_id=f"retry-failed-finalize:{bounded_recovery}",
    )

    with pytest.raises(WeightTransferReleaseError) as raised:
        execute_weight_materialization(
            request,
            provider=provider,
            attestor=ALLOW_ALL_ATTESTOR,
            execution_context=WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + 1,
            ),
        )

    receipt = raised.value.receipt
    assert receipt is not None
    assert receipt.completion_ticket is not None
    assert store.finalize_attempts == 1
    failed_call = provider._finalize_calls[request.operation_id]
    assert failed_call.done.wait(timeout=1)
    assert failed_call.error is not None

    recovered = provider.recover_materialization(
        request,
        completion_ticket=receipt.completion_ticket,
        execution_context=(
            WeightTransferExecutionContext(
                deadline_unix_sec=time.time() + 1,
            )
            if bounded_recovery
            else None
        ),
    )

    assert recovered is not None
    assert recovered.operation_id == request.operation_id
    assert store.finalize_attempts == 2
    assert request.operation_id not in provider._finalize_calls


def test_distributed_upload_timeout_converges_to_completion_unknown(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = BlockingUploadStore()
    business_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 0.05
    )
    coordinator = TerminalOutcomeCoordinator(
        sources[1].placement_id,
        business_context,
        check_upload=True,
    )
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
        prepare_upload_is_local=True,
    )
    request = materialization_request(
        sources,
        operation_id="distributed-upload-timeout",
    )

    try:
        with pytest.raises(WeightTransferCompletionUnknownError) as raised:
            execute_weight_materialization(
                request,
                provider=provider,
                attestor=ALLOW_ALL_ATTESTOR,
                execution_context=business_context,
            )
        assert raised.value.completion_ticket
        assert coordinator.poisoned is False
        exchange = next(
            call[1]
            for call in coordinator.calls
            if isinstance(call, tuple) and call[0] == "exchange"
        )
        assert exchange.completion_unknown is True
    finally:
        store.release_upload.set()

    submission = provider._pending_materializations[request.operation_id]
    assert submission.local_upload_call is not None
    assert submission.local_upload_call.done.wait(timeout=1)


def test_distributed_upload_terminal_context_includes_business_window(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    coordinator = FakeDistributedCoordinator(sources[1].placement_id)
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
        prepare_upload_is_local=True,
    )
    business_context = WeightTransferExecutionContext(deadline_unix_sec=time.time() + 2)

    execute_weight_materialization(
        materialization_request(
            sources,
            operation_id="distributed-upload-terminal-window",
        ),
        provider=provider,
        attestor=ALLOW_ALL_ATTESTOR,
        execution_context=business_context,
    )

    assert len(coordinator.upload_execution_contexts) == 1
    terminal_context = coordinator.upload_execution_contexts[0]
    assert terminal_context is not None
    assert terminal_context is not business_context
    assert terminal_context.deadline_unix_sec == pytest.approx(
        business_context.deadline_unix_sec
        + mooncake_store_api._STORE_TERMINAL_CONTROL_TIMEOUT_SEC,
        abs=0.01,
        rel=0,
    )


def test_distributed_commit_not_started_is_known_and_aborted(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = FakeWeightStore()
    business_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 0.05
    )
    coordinator = ExpireBeforeCommitCoordinator(
        sources[1].placement_id,
        business_context,
        check_upload=False,
    )
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
        prepare_upload_is_local=True,
    )
    request = materialization_request(
        sources,
        operation_id="distributed-commit-not-started",
    )

    with pytest.raises(WeightTransferError) as raised:
        execute_weight_materialization(
            request,
            provider=provider,
            attestor=ALLOW_ALL_ATTESTOR,
            execution_context=business_context,
        )

    assert raised.value.code == "DEADLINE_EXCEEDED"
    assert raised.value.completion_known is True
    assert not any(call[0] == "commit" for call in store.calls)
    assert [call[0] for call in store.calls].count("abort") == 1
    assert request.operation_id not in provider._pending_materializations
    assert coordinator.poisoned is False


def test_distributed_commit_timeout_uses_terminal_control_context(
    monkeypatch,
) -> None:
    backend = fake_backend({})
    monkeypatch.setattr(
        MooncakeWeightStoreProvider,
        "_load_backend",
        staticmethod(lambda: backend),
    )
    sources = placements("source", shard_dim=0)
    store = BlockingCommitStore()
    business_context = WeightTransferExecutionContext(
        deadline_unix_sec=time.time() + 0.05
    )
    coordinator = TerminalOutcomeCoordinator(
        sources[1].placement_id,
        business_context,
        check_upload=False,
    )
    provider = MooncakeWeightStoreProvider(
        store,
        local_placement_ids=(sources[0].placement_id,),
        coordinator=coordinator,
        payload_checksum_verifier=payload_checksum_verifier(sources),
        prepare_upload_is_local=True,
    )
    request = materialization_request(
        sources,
        operation_id="distributed-commit-timeout",
    )

    started = time.monotonic()
    try:
        with pytest.raises(WeightTransferCompletionUnknownError) as raised:
            execute_weight_materialization(
                request,
                provider=provider,
                attestor=ALLOW_ALL_ATTESTOR,
                execution_context=business_context,
            )
        assert time.monotonic() - started < 0.5
        assert raised.value.completion_ticket
        assert store.commit_started.is_set()
        assert coordinator.poisoned is False
    finally:
        store.release_commit.set()

    submission = provider._pending_materializations[request.operation_id]
    assert submission.local_commit_call is not None
    assert submission.local_commit_call.done.wait(timeout=1)
    assert coordinator.terminal_contexts
    for _, context in coordinator.terminal_contexts:
        assert context is not business_context
        assert not context.expired()
        assert (
            context.remaining_seconds()
            <= mooncake_store_api._STORE_TERMINAL_CONTROL_TIMEOUT_SEC + 0.5
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
