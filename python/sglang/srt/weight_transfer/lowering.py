from __future__ import annotations

from dataclasses import dataclass

from sglang.srt.weight_transfer.contracts import (
    BoundWeightTransferPlan,
    PhysicalWeightLocation,
    RuntimeWeightLocation,
)


@dataclass(frozen=True)
class WeightLoweringLimits:
    max_total_operations: int
    max_batch_operations: int
    max_batch_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_total_operations", "max_batch_operations"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_batch_bytes is not None and (
            type(self.max_batch_bytes) is not int or self.max_batch_bytes <= 0
        ):
            raise ValueError("max_batch_bytes must be a positive integer")


@dataclass(frozen=True)
class FlatWeightTransferOperation:
    region_index: int
    segment_index: int
    chunk_index: int
    source: PhysicalWeightLocation
    target: RuntimeWeightLocation
    source_offset: int
    target_offset: int
    nbytes: int

    def __post_init__(self) -> None:
        for name in (
            "region_index",
            "segment_index",
            "chunk_index",
            "source_offset",
            "target_offset",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.nbytes) is not int or self.nbytes <= 0:
            raise ValueError("nbytes must be a positive integer")


@dataclass(frozen=True)
class WeightTransferBatch:
    operations: tuple[FlatWeightTransferOperation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", tuple(self.operations))
        if not self.operations:
            raise ValueError("transfer batch must not be empty")

    @property
    def total_bytes(self) -> int:
        return sum(operation.nbytes for operation in self.operations)


def lowering_operation_count(
    plan: BoundWeightTransferPlan,
    limits: WeightLoweringLimits,
) -> int:
    max_chunk_bytes = limits.max_batch_bytes
    total = 0
    for region in plan.regions:
        chunks_per_segment = (
            1
            if max_chunk_bytes is None
            else (region.inner_bytes + max_chunk_bytes - 1) // max_chunk_bytes
        )
        total += region.segment_count * chunks_per_segment
    return total


def iter_bounded_transfer_batches(
    plan: BoundWeightTransferPlan,
    limits: WeightLoweringLimits,
):
    """Lazily lower compact N-D regions into bounded flat-copy batches."""

    operation_count = lowering_operation_count(plan, limits)
    if operation_count > limits.max_total_operations:
        raise ValueError(
            "lowering exceeds total operation limit: "
            f"{operation_count} > {limits.max_total_operations}"
        )

    pending: list[FlatWeightTransferOperation] = []
    pending_bytes = 0
    for region_index, region in enumerate(plan.regions):
        for segment_index, (
            source_offset,
            target_offset,
            nbytes,
        ) in enumerate(region.iter_segments()):
            remaining = nbytes
            chunk_index = 0
            while remaining:
                chunk_bytes = (
                    remaining
                    if limits.max_batch_bytes is None
                    else min(remaining, limits.max_batch_bytes)
                )
                if pending and (
                    len(pending) >= limits.max_batch_operations
                    or (
                        limits.max_batch_bytes is not None
                        and pending_bytes + chunk_bytes > limits.max_batch_bytes
                    )
                ):
                    yield WeightTransferBatch(tuple(pending))
                    pending.clear()
                    pending_bytes = 0
                pending.append(
                    FlatWeightTransferOperation(
                        region_index=region_index,
                        segment_index=segment_index,
                        chunk_index=chunk_index,
                        source=region.source,
                        target=region.target,
                        source_offset=source_offset + (nbytes - remaining),
                        target_offset=target_offset + (nbytes - remaining),
                        nbytes=chunk_bytes,
                    )
                )
                pending_bytes += chunk_bytes
                remaining -= chunk_bytes
                chunk_index += 1
                if len(pending) == limits.max_batch_operations or (
                    limits.max_batch_bytes is not None
                    and pending_bytes == limits.max_batch_bytes
                ):
                    yield WeightTransferBatch(tuple(pending))
                    pending.clear()
                    pending_bytes = 0
    if pending:
        yield WeightTransferBatch(tuple(pending))
