"""Pressure-metric forwarding contracts for the load reporter."""

from __future__ import annotations

import pytest

from sglang.srt.load_reporter.config import WorkerMetadata
from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.report_builder import ReportBuilder, SequenceAllocator
from sglang.srt.load_reporter.snapshot_validation import (
    SnapshotValidationError,
    validate_full_snapshot,
)
from sglang.srt.managers.load_snapshot import DisaggregationMetrics, LoadSnapshot
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def snapshot(**overrides) -> LoadSnapshot:
    fields = dict(
        timestamp=1.0,
        dp_rank=0,
        num_running_reqs=2,
        num_waiting_reqs=6,
        num_waiting_uncached_tokens=800,
        num_used_tokens=2_000,
        num_total_tokens=2_500,
        num_active_tokens=1_700,
        max_total_num_tokens=10_000,
        max_running_requests=32,
        total_prefill_uncached_tokens=12_000,
        total_prefill_busy_us=5_500_000,
        decode_moments=[120, 0, 2_200_000, 0, 0, 3_000],
        disaggregation=DisaggregationMetrics(
            mode="decode",
            decode_prealloc_queue_reqs=2,
            decode_transfer_queue_reqs=3,
            decode_retracted_queue_reqs=1,
        ),
    )
    fields.update(overrides)
    return LoadSnapshot(**fields)


def test_pressure_metrics_reach_rank_proto() -> None:
    (rank,) = validate_full_snapshot(
        [snapshot()], expected_dp_ranks={0}, fallback_time_unix_ms=1_000
    )
    report = ReportBuilder("source", 3_000, SequenceAllocator()).build(
        (rank,),
        WorkerMetadata("worker:30000", pb.WORKER_TYPE_DECODE, "model"),
        report_time_unix_ms=1_000,
    )
    wire_rank = report.ranks[0]

    assert wire_rank.num_active_tokens == 1_700
    assert wire_rank.total_prefill_uncached_tokens == 12_000
    assert wire_rank.total_prefill_busy_us == 5_500_000
    assert wire_rank.decode_prealloc_queue_reqs == 2
    assert wire_rank.decode_transfer_queue_reqs == 3
    assert wire_rank.decode_retracted_queue_reqs == 1
    assert wire_rank.total_decode_steps == 120
    assert wire_rank.total_decode_step_us == 2_200_000


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"num_active_tokens": 2_501}, "num_active_tokens must not exceed"),
        ({"decode_moments": [1, 2, 3]}, "decode_moments must contain exactly 6"),
        ({"total_prefill_busy_us": -1}, "total_prefill_busy_us must be in protobuf"),
    ],
)
def test_invalid_pressure_metrics_are_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(SnapshotValidationError, match=message):
        validate_full_snapshot(
            [snapshot(**overrides)], expected_dp_ranks={0}, fallback_time_unix_ms=1_000
        )
