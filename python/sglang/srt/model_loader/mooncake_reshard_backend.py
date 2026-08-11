from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from mooncake.reshard.weight import (
    MemoryRegistrationLease,
    MooncakeTransferEngineReader,
    TransferCompletionUnknownError,
    TransferEngineError,
    bind_logical_transfer_plan,
    plan_placement_transfer_to_local_target,
)

from sglang.srt.model_executor.mooncake_reshard_adapter import (
    MooncakeCanonicalReshardAdapter,
    PlacementInventoryParticipant,
)
from sglang.srt.model_loader.weight_reshard_backend import (
    WeightReshardBackendError,
    WeightReshardCompletionUnknownError,
    WeightReshardExecutionResult,
)


@dataclass(slots=True)
class _MooncakePreparedWeightReshardTransfer:
    plan: Any
    source_placement: Any
    source_bindings: tuple[Any, ...]
    target_placement: Any
    target_binding: Any
    target_resource: Any
    source_registrations: tuple[Any, ...]
    resources: ExitStack
    reader: Any | None = None
    completion_unknown_retained: bool = False
    closed: bool = False

    @property
    def operation_count(self) -> int:
        return len(self.plan.operations)

    @property
    def nbytes(self) -> int:
        return sum(operation.total_bytes for operation in self.plan.operations)


class MooncakeWeightReshardBackend:
    """Execute copy-only resharding for byte-compatible canonical layouts."""

    def __init__(self) -> None:
        self._adapter = MooncakeCanonicalReshardAdapter()

    @contextmanager
    def prepare(
        self,
        *,
        model: Any,
        source_placement_inventories: tuple[Any, ...],
        source_binding_inventories: tuple[Any, ...],
        target_inventory_builder: Any,
        target_model_id: str,
        target_revision: str,
        target_instance_id: str,
        target_endpoint: str,
        world_group: Any,
    ) -> Iterator[_MooncakePreparedWeightReshardTransfer]:
        resources = ExitStack()
        try:
            source_placement, source_bindings = (
                self._adapter.source_placement_and_bindings(
                    source_placement_inventories,
                    source_binding_inventories,
                )
            )
            self._require_identity(
                source_placement,
                model_id=target_model_id,
                revision=target_revision,
                role="source",
            )
            target_resource = resources.enter_context(
                target_inventory_builder(
                    model=model,
                    model_id=target_model_id,
                    revision=target_revision,
                    weight_generation=source_placement.weight_generation,
                    instance_id=target_instance_id,
                    endpoint=target_endpoint,
                )
            )
            placement_inventory = getattr(target_resource, "placement_inventory", None)
            if placement_inventory is None or not callable(
                getattr(target_resource, "bind", None)
            ):
                raise ValueError(
                    "target inventory session must expose placement_inventory "
                    "and bind()"
                )
            target_participant_id = placement_inventory.participant_id
            target_placement = self._adapter.gather_target_placement(
                PlacementInventoryParticipant(placement_inventory),
                world_group=world_group,
            )
            self._require_identity(
                target_placement,
                model_id=target_model_id,
                revision=target_revision,
                role="target",
            )
            logical_plan = plan_placement_transfer_to_local_target(
                source_placement,
                target_placement,
                target_participant_id,
            )
            target_binding_inventory = resources.enter_context(target_resource.bind())
            target_binding = self._adapter.runtime_binding_manifest(
                target_binding_inventory,
                placement=target_placement,
                placement_inventory=placement_inventory,
            )
            plan = bind_logical_transfer_plan(
                logical_plan,
                (target_binding,),
                source_bindings=source_bindings,
            )
            source_registrations = tuple(
                MemoryRegistrationLease.from_fragment(
                    fragment,
                    runtime_lease_id=binding.lease_id,
                    lease_generation=binding.generation,
                )
                for binding in source_bindings
                for fragment in binding.fragments
            )
            prepared = _MooncakePreparedWeightReshardTransfer(
                plan=plan,
                source_placement=source_placement,
                source_bindings=tuple(source_bindings),
                target_placement=target_placement,
                target_binding=target_binding,
                target_resource=target_resource,
                source_registrations=source_registrations,
                resources=resources,
            )
        except Exception as error:
            resources.close()
            raise WeightReshardBackendError(
                "Mooncake weight reshard preparation failed"
            ) from error

        try:
            yield prepared
        finally:
            if not prepared.completion_unknown_retained:
                self.close_after_terminal(prepared)

    def execute(
        self,
        prepared: _MooncakePreparedWeightReshardTransfer,
        *,
        transfer_engine: Any,
    ) -> WeightReshardExecutionResult:
        reader = MooncakeTransferEngineReader(
            transfer_engine,
            max_batch_operations=8192,
        )
        prepared.reader = reader
        try:
            receipts = reader.execute(
                prepared.plan,
                prepared.source_placement,
                prepared.source_bindings,
                prepared.target_placement,
                prepared.target_binding,
                source_pre_registered=True,
                source_registrations=prepared.source_registrations,
                target_pre_registered=False,
            )
        except TransferCompletionUnknownError as error:
            self.retain_completion_unknown(prepared)
            raise WeightReshardCompletionUnknownError(
                "Mooncake transfer completion is unknown",
                pending_transfer_id=error.pending_transfer_id,
            ) from error
        except TransferEngineError as error:
            raise WeightReshardBackendError(
                "Mooncake transfer reached a terminal failure"
            ) from error
        return WeightReshardExecutionResult(
            nbytes=sum(receipt.nbytes for receipt in receipts),
            segment_count=sum(receipt.operation_count for receipt in receipts),
        )

    def activate(self, prepared: _MooncakePreparedWeightReshardTransfer) -> None:
        activate = getattr(prepared.target_resource, "activate", None)
        if not callable(activate):
            raise WeightReshardBackendError(
                "target inventory session does not support content activation"
            )
        try:
            activate()
        except Exception as error:
            raise WeightReshardBackendError(
                "target logical weight generation activation failed"
            ) from error

    def drain_pending_transfer(
        self,
        prepared: _MooncakePreparedWeightReshardTransfer,
        *,
        pending_transfer_id: str,
        timeout_ms: int,
    ) -> str:
        if prepared.reader is None:
            raise WeightReshardBackendError(
                "Mooncake pending transfer has no execution reader"
            )
        try:
            return prepared.reader.drain_pending_transfer(
                pending_transfer_id,
                timeout_ms=timeout_ms,
            )
        except TransferEngineError as error:
            raise WeightReshardBackendError(
                "Mooncake pending transfer drain failed"
            ) from error

    def retain_completion_unknown(
        self,
        prepared: _MooncakePreparedWeightReshardTransfer,
    ) -> None:
        if prepared.closed:
            raise WeightReshardBackendError(
                "cannot retain resources after terminal close"
            )
        prepared.completion_unknown_retained = True

    def close_after_terminal(
        self,
        prepared: _MooncakePreparedWeightReshardTransfer,
    ) -> None:
        if prepared.closed:
            return
        prepared.resources.close()
        prepared.closed = True
        prepared.completion_unknown_retained = False

    @staticmethod
    def _require_identity(
        placement: Any,
        *,
        model_id: str,
        revision: str,
        role: str,
    ) -> None:
        if (placement.resource_id, placement.revision) != (model_id, revision):
            raise ValueError(
                f"{role} placement identity does not match target model "
                f"{model_id}@{revision}"
            )


__all__ = ["MooncakeWeightReshardBackend"]
