# ruff: noqa: F401

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.model_executor.weight_runtime_manifest import (
        WeightRuntimeBindingManifest,
    )
    from sglang.srt.weight_transfer.api import (
        WeightTransferPreflight,
        execute_weight_load,
        execute_weight_materialization,
        load_weight_snapshot,
        load_weights,
        load_weights_to_local_target,
        mark_weight_snapshot_serving,
        materialize_weight_snapshot,
        materialize_weight_snapshot_candidate,
        materialize_weights,
        preflight_weight_transfer,
        prepare_weight_load,
        prepare_weight_load_from_plan,
        prepare_weight_load_to_local_target,
        prepare_weight_materialization,
        publish_weight_snapshot,
    )
    from sglang.srt.weight_transfer.binding import (
        bind_weight_source,
        bind_weight_transfer_plan,
        project_source_bindings,
        runtime_manifest_to_parts,
    )
    from sglang.srt.weight_transfer.checkpoint import (
        SemanticCheckpointSource,
        load_checkpoint_weights,
        materialize_checkpoint_weight_snapshot,
        materialize_checkpoint_weights,
    )
    from sglang.srt.weight_transfer.checkpoint_provider import (
        CheckpointLoadStats,
        CheckpointProviderState,
        CheckpointStorageToRuntimeProvider,
        RuntimeRangeWriter,
        StorageRangeReader,
        StorageVersionReader,
    )
    from sglang.srt.weight_transfer.contracts import (
        BoundWeightTransferPlan,
        BoundWeightTransferRegion,
        LogicalPlacementFragment,
        LogicalWeightTransferPlan,
        LogicalWeightTransferRegion,
        PhysicalWeightLocation,
        PipelineRouteGroup,
        PlacementExecutorGroup,
        RuntimeWeightLocation,
        SourceBindingManifest,
        StorageWeightLocation,
        WeightStorageBindingManifest,
        WeightStorageFragmentBinding,
    )
    from sglang.srt.weight_transfer.distributed import (
        LocalWeightStoreDistributedCoordinator,
        RootWeightStorageCatalog,
        TorchDistributedWeightStoreCoordinator,
        WeightStoreDistributedCoordinator,
        WeightStoreDistributedError,
        WeightStoreUploadOutcome,
    )
    from sglang.srt.weight_transfer.lowering import (
        FlatWeightTransferOperation,
        WeightLoweringLimits,
        WeightTransferBatch,
        iter_bounded_transfer_batches,
        lowering_operation_count,
    )
    from sglang.srt.weight_transfer.mooncake import MooncakeWeightTransferProvider
    from sglang.srt.weight_transfer.mooncake_store import MooncakeWeightStoreProvider
    from sglang.srt.weight_transfer.planner import (
        DEFAULT_WEIGHT_PLANNER_LIMITS,
        WeightPlannerLimits,
        plan_weight_transfer,
        plan_weight_transfer_to_local_target,
        project_weight_transfer_plan_to_target,
        project_weight_transfer_plan_to_targets,
        select_weight_storage_placements,
    )
    from sglang.srt.weight_transfer.provider import (
        WeightLoadReceipt,
        WeightLoadRequest,
        WeightMaterializationCompletionTicketProvider,
        WeightMaterializationRecoveryProvider,
        WeightMaterializeReceipt,
        WeightMaterializeRequest,
        WeightPayloadFragmentIdentity,
        WeightPayloadIdentity,
        WeightProviderCapabilities,
        WeightProviderReceipt,
        WeightProviderRequest,
        WeightStorageDestination,
        WeightTargetActivationController,
        WeightTargetLoadMode,
        WeightTargetLoadSession,
        WeightTargetLoadState,
        WeightTransferAttestor,
        WeightTransferCancellationSignal,
        WeightTransferCompletionUnknownError,
        WeightTransferError,
        WeightTransferExecutionContext,
        WeightTransferProvider,
        WeightTransferReleaseError,
        WeightTransferTerminalProof,
        WeightTransferTerminalStatus,
    )
    from sglang.srt.weight_transfer.runtime import (
        RuntimeWeightPayloadHasher,
        RuntimeWeightSnapshotSource,
        materialize_distributed_runtime_weight_snapshot,
        materialize_runtime_weight_snapshot,
        materialize_runtime_weights,
        quarantined_runtime_weight_snapshots,
    )
    from sglang.srt.weight_transfer.storage import (
        InMemoryWeightStorageCatalog,
        StoredWeightSnapshot,
        WeightMaterializationAttempt,
        WeightMaterializationAttemptState,
        WeightMaterializationIntent,
        WeightRevisionHead,
        WeightRevisionState,
        WeightSnapshotPublication,
        WeightSnapshotPublicationState,
        WeightStorageCatalog,
        WeightStorageRef,
        weight_placement_set_digest,
        weight_source_snapshot_digest,
        weight_stored_payload_digest,
    )
    from sglang.srt.weight_transfer.storage_file import FileWeightStorageCatalog
    from sglang.srt.weight_transfer.store_runtime import (
        WeightSnapshotBackend,
        WeightSnapshotBackendFactory,
        WeightSnapshotLoadSpec,
        WeightSnapshotWriteBackendFactory,
        WeightSnapshotWriteSpec,
        open_weight_snapshot_backend,
        open_weight_snapshot_write_backend,
        register_weight_snapshot_write_backend,
    )

_EXPORT_GROUPS = {
    "sglang.srt.model_executor.weight_runtime_manifest": (
        "WeightRuntimeBindingManifest",
    ),
    "sglang.srt.weight_transfer.api": (
        "WeightTransferPreflight",
        "execute_weight_load",
        "execute_weight_materialization",
        "load_weight_snapshot",
        "load_weights",
        "load_weights_to_local_target",
        "mark_weight_snapshot_serving",
        "materialize_weight_snapshot",
        "materialize_weight_snapshot_candidate",
        "materialize_weights",
        "preflight_weight_transfer",
        "prepare_weight_load",
        "prepare_weight_load_from_plan",
        "prepare_weight_load_to_local_target",
        "prepare_weight_materialization",
        "publish_weight_snapshot",
    ),
    "sglang.srt.weight_transfer.binding": (
        "bind_weight_source",
        "bind_weight_transfer_plan",
        "project_source_bindings",
        "runtime_manifest_to_parts",
    ),
    "sglang.srt.weight_transfer.checkpoint": (
        "SemanticCheckpointSource",
        "load_checkpoint_weights",
        "materialize_checkpoint_weight_snapshot",
        "materialize_checkpoint_weights",
    ),
    "sglang.srt.weight_transfer.checkpoint_provider": (
        "CheckpointLoadStats",
        "CheckpointProviderState",
        "CheckpointStorageToRuntimeProvider",
        "RuntimeRangeWriter",
        "StorageRangeReader",
        "StorageVersionReader",
    ),
    "sglang.srt.weight_transfer.contracts": (
        "BoundWeightTransferPlan",
        "BoundWeightTransferRegion",
        "LogicalPlacementFragment",
        "LogicalWeightTransferPlan",
        "LogicalWeightTransferRegion",
        "PhysicalWeightLocation",
        "PipelineRouteGroup",
        "PlacementExecutorGroup",
        "RuntimeWeightLocation",
        "SourceBindingManifest",
        "StorageWeightLocation",
        "WeightStorageBindingManifest",
        "WeightStorageFragmentBinding",
    ),
    "sglang.srt.weight_transfer.distributed": (
        "LocalWeightStoreDistributedCoordinator",
        "RootWeightStorageCatalog",
        "TorchDistributedWeightStoreCoordinator",
        "WeightStoreDistributedCoordinator",
        "WeightStoreDistributedError",
        "WeightStoreUploadOutcome",
    ),
    "sglang.srt.weight_transfer.lowering": (
        "FlatWeightTransferOperation",
        "WeightLoweringLimits",
        "WeightTransferBatch",
        "iter_bounded_transfer_batches",
        "lowering_operation_count",
    ),
    "sglang.srt.weight_transfer.mooncake": ("MooncakeWeightTransferProvider",),
    "sglang.srt.weight_transfer.mooncake_store": ("MooncakeWeightStoreProvider",),
    "sglang.srt.weight_transfer.planner": (
        "DEFAULT_WEIGHT_PLANNER_LIMITS",
        "WeightPlannerLimits",
        "plan_weight_transfer",
        "plan_weight_transfer_to_local_target",
        "project_weight_transfer_plan_to_target",
        "project_weight_transfer_plan_to_targets",
        "select_weight_storage_placements",
    ),
    "sglang.srt.weight_transfer.provider": (
        "WeightLoadReceipt",
        "WeightLoadRequest",
        "WeightMaterializationCompletionTicketProvider",
        "WeightMaterializationRecoveryProvider",
        "WeightMaterializeReceipt",
        "WeightMaterializeRequest",
        "WeightPayloadFragmentIdentity",
        "WeightPayloadIdentity",
        "WeightProviderCapabilities",
        "WeightProviderReceipt",
        "WeightProviderRequest",
        "WeightStorageDestination",
        "WeightTargetActivationController",
        "WeightTargetLoadMode",
        "WeightTargetLoadSession",
        "WeightTargetLoadState",
        "WeightTransferAttestor",
        "WeightTransferCancellationSignal",
        "WeightTransferCompletionUnknownError",
        "WeightTransferError",
        "WeightTransferExecutionContext",
        "WeightTransferProvider",
        "WeightTransferReleaseError",
        "WeightTransferTerminalProof",
        "WeightTransferTerminalStatus",
    ),
    "sglang.srt.weight_transfer.runtime": (
        "RuntimeWeightPayloadHasher",
        "RuntimeWeightSnapshotSource",
        "materialize_distributed_runtime_weight_snapshot",
        "materialize_runtime_weight_snapshot",
        "materialize_runtime_weights",
        "quarantined_runtime_weight_snapshots",
    ),
    "sglang.srt.weight_transfer.storage": (
        "InMemoryWeightStorageCatalog",
        "StoredWeightSnapshot",
        "WeightMaterializationAttempt",
        "WeightMaterializationAttemptState",
        "WeightMaterializationIntent",
        "WeightRevisionHead",
        "WeightRevisionState",
        "WeightSnapshotPublication",
        "WeightSnapshotPublicationState",
        "WeightStorageCatalog",
        "WeightStorageRef",
        "weight_placement_set_digest",
        "weight_source_snapshot_digest",
        "weight_stored_payload_digest",
    ),
    "sglang.srt.weight_transfer.storage_file": ("FileWeightStorageCatalog",),
    "sglang.srt.weight_transfer.store_runtime": (
        "WeightSnapshotBackend",
        "WeightSnapshotBackendFactory",
        "WeightSnapshotLoadSpec",
        "WeightSnapshotWriteBackendFactory",
        "WeightSnapshotWriteSpec",
        "open_weight_snapshot_backend",
        "open_weight_snapshot_write_backend",
        "register_weight_snapshot_write_backend",
    ),
}

_EXPORTS = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
