from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol


class WeightReshardBackendError(RuntimeError):
    """SGLang-owned terminal weight reshard backend failure."""


class WeightReshardBackendUnavailableError(WeightReshardBackendError):
    """Configured weight reshard backend cannot be loaded."""


class WeightReshardCompletionUnknownError(WeightReshardBackendError):
    """Execution retained resources because DMA may still access their memory."""

    def __init__(self, message: str, *, pending_transfer_id: str) -> None:
        super().__init__(message)
        self.pending_transfer_id = pending_transfer_id


class PreparedWeightReshardTransfer(Protocol):
    """Opaque backend-owned resources retained through transfer completion."""

    @property
    def operation_count(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    @property
    def completion_unknown_retained(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class WeightReshardExecutionResult:
    nbytes: int
    segment_count: int

    def __post_init__(self) -> None:
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("weight reshard result nbytes must be positive")
        if type(self.segment_count) is not int or self.segment_count <= 0:
            raise ValueError("weight reshard result segment_count must be positive")


def validate_weight_reshard_execution_result(
    prepared: PreparedWeightReshardTransfer,
    result: object,
) -> WeightReshardExecutionResult:
    """Require a complete terminal receipt before post-load or activation."""

    if not isinstance(result, WeightReshardExecutionResult):
        raise WeightReshardBackendError(
            "weight reshard backend returned no valid execution receipt"
        )
    expected_nbytes = prepared.nbytes
    operation_count = prepared.operation_count
    if type(expected_nbytes) is not int or expected_nbytes <= 0:
        raise WeightReshardBackendError("prepared transfer has invalid expected bytes")
    if type(operation_count) is not int or operation_count <= 0:
        raise WeightReshardBackendError("prepared transfer has no operations")
    if result.nbytes != expected_nbytes:
        raise WeightReshardBackendError(
            "weight reshard execution receipt does not cover the prepared bytes"
        )
    if result.segment_count < operation_count:
        raise WeightReshardBackendError(
            "weight reshard execution receipt has fewer segments than operations"
        )
    return result


class WeightReshardBackend(Protocol):
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
    ) -> AbstractContextManager[PreparedWeightReshardTransfer]: ...

    def execute(
        self,
        prepared: PreparedWeightReshardTransfer,
        *,
        transfer_engine: Any,
    ) -> WeightReshardExecutionResult: ...

    def activate(self, prepared: PreparedWeightReshardTransfer) -> None: ...

    def drain_pending_transfer(
        self,
        prepared: PreparedWeightReshardTransfer,
        *,
        pending_transfer_id: str,
        timeout_ms: int,
    ) -> str: ...

    def retain_completion_unknown(
        self,
        prepared: PreparedWeightReshardTransfer,
    ) -> None: ...

    def close_after_terminal(
        self,
        prepared: PreparedWeightReshardTransfer,
    ) -> None: ...


def create_weight_reshard_backend(backend: Any) -> WeightReshardBackend:
    """Load the configured backend and validate its runtime capability."""

    backend_name = getattr(backend, "value", backend)
    if backend_name != "transfer_engine":
        raise WeightReshardBackendUnavailableError(
            f"unsupported weight reshard backend: {backend_name!r}"
        )
    try:
        from sglang.srt.model_loader.mooncake_reshard_backend import (
            MooncakeWeightReshardBackend,
        )

        return MooncakeWeightReshardBackend()
    except Exception as error:
        raise WeightReshardBackendUnavailableError(
            "Mooncake weight reshard backend is unavailable; install a "
            "Mooncake package with the weight reshard Python capability"
        ) from error


__all__ = [
    "PreparedWeightReshardTransfer",
    "WeightReshardBackend",
    "WeightReshardBackendError",
    "WeightReshardBackendUnavailableError",
    "WeightReshardCompletionUnknownError",
    "WeightReshardExecutionResult",
    "create_weight_reshard_backend",
    "validate_weight_reshard_execution_result",
]
