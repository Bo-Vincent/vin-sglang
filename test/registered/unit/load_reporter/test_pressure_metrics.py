"""Load-reporter contracts used by Router pressure policies."""

from __future__ import annotations

import unittest

from sglang.srt.load_reporter.registration import WorkerIdentity
from sglang.srt.load_reporter.report_builder import ReportBuilder, SequenceAllocator
from sglang.srt.load_reporter.store import LatestSnapshotStore, SnapshotValidationError
from sglang.srt.managers.load_snapshot import DisaggregationMetrics, LoadSnapshot


class TestPressureMetricTransport(unittest.TestCase):
    def test_prefill_and_decode_counters_reach_rank_proto(self) -> None:
        store = LatestSnapshotStore()
        store.apply_full_snapshot(
            [
                LoadSnapshot(
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
            ],
            expected_dp_ranks={0},
            collected_at_unix_ms=1_000,
            collected_at_monotonic=1.0,
        )

        report = ReportBuilder("source", 3_000, SequenceAllocator()).build(
            store.view(),
            WorkerIdentity("worker:30000", 3, "model", None),
            report_time_unix_ms=1_000,
        )
        rank = report.ranks[0]

        self.assertEqual(rank.num_active_tokens, 1_700)
        self.assertEqual(rank.total_prefill_uncached_tokens, 12_000)
        self.assertEqual(rank.total_prefill_busy_us, 5_500_000)
        self.assertEqual(rank.decode_prealloc_queue_reqs, 2)
        self.assertEqual(rank.decode_transfer_queue_reqs, 3)
        self.assertEqual(rank.decode_retracted_queue_reqs, 1)
        self.assertEqual(rank.total_decode_steps, 120)
        self.assertEqual(rank.total_decode_step_us, 2_200_000)

    def test_invalid_new_counters_are_rejected_before_publication(self) -> None:
        store = LatestSnapshotStore()
        with self.assertRaisesRegex(
            SnapshotValidationError, "num_active_tokens must not exceed"
        ):
            store.apply_full_snapshot(
                [
                    LoadSnapshot(
                        dp_rank=0,
                        num_total_tokens=10,
                        num_active_tokens=11,
                        max_total_num_tokens=100,
                        max_running_requests=1,
                    )
                ],
                expected_dp_ranks={0},
                collected_at_unix_ms=1_000,
                collected_at_monotonic=1.0,
            )

        with self.assertRaisesRegex(
            SnapshotValidationError, "decode_moments must contain exactly 6 values"
        ):
            store.apply_full_snapshot(
                [
                    LoadSnapshot(
                        dp_rank=0,
                        max_total_num_tokens=100,
                        max_running_requests=1,
                        decode_moments=[1, 2, 3],
                    )
                ],
                expected_dp_ranks={0},
                collected_at_unix_ms=1_000,
                collected_at_monotonic=1.0,
            )

        with self.assertRaisesRegex(
            SnapshotValidationError,
            "total_prefill_busy_us must be in protobuf int64 range",
        ):
            store.apply_full_snapshot(
                [
                    LoadSnapshot(
                        dp_rank=0,
                        max_total_num_tokens=100,
                        max_running_requests=1,
                        total_prefill_busy_us=-1,
                    )
                ],
                expected_dp_ranks={0},
                collected_at_unix_ms=1_000,
                collected_at_monotonic=1.0,
            )


if __name__ == "__main__":
    unittest.main()
