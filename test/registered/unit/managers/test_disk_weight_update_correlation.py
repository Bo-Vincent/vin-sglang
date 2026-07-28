import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
)
from sglang.srt.managers.tokenizer_manager import (
    TokenizerManager,
    _DiskWeightUpdateAttempt,
)
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _response(request_id: str, rank: int) -> UpdateWeightFromDiskReqOutput:
    return UpdateWeightFromDiskReqOutput(
        success=True,
        message=f"rank {rank}",
        request_id=request_id,
        external_dp_rank=rank,
    )


class TestDiskWeightUpdateCorrelation(unittest.TestCase):
    def test_stale_and_duplicate_responses_do_not_complete_attempt(self):
        async def scenario():
            manager = TokenizerManager.__new__(TokenizerManager)
            future = asyncio.get_running_loop().create_future()
            manager.model_update_attempts = {
                "current": _DiskWeightUpdateAttempt(future, expected_workers=2)
            }

            manager._handle_update_weights_from_disk_req_output(_response("stale", 0))
            manager._handle_update_weights_from_disk_req_output(_response("current", 0))
            manager._handle_update_weights_from_disk_req_output(_response("current", 0))
            self.assertFalse(future.done())

            manager._handle_update_weights_from_disk_req_output(_response("current", 1))
            results = await future
            self.assertEqual([result.external_dp_rank for result in results], [0, 1])

        asyncio.run(scenario())

    def test_invalid_rank_is_ignored(self):
        async def scenario():
            manager = TokenizerManager.__new__(TokenizerManager)
            future = asyncio.get_running_loop().create_future()
            manager.model_update_attempts = {
                "current": _DiskWeightUpdateAttempt(future, expected_workers=1)
            }

            manager._handle_update_weights_from_disk_req_output(_response("current", 1))
            self.assertFalse(future.done())

            manager._handle_update_weights_from_disk_req_output(_response("current", 0))
            self.assertEqual((await future)[0].external_dp_rank, 0)

        asyncio.run(scenario())

    def test_missing_responder_releases_waiter_at_deadline(self):
        manager = SimpleNamespace(
            elastic_worker_count=2,
            model_update_attempts={},
            _dispatch_to_scheduler=lambda _request: None,
        )
        request = UpdateWeightFromDiskReqInput(
            model_path="/weights",
            timeout_sec=0.01,
        )

        with self.assertRaisesRegex(RuntimeError, "received 0/2"):
            asyncio.run(
                TokenizerManager._wait_for_model_update_from_disk(manager, request)
            )

        self.assertEqual(manager.model_update_attempts, {})

    def test_public_entry_rejects_invalid_timeout_without_fail_closed(self):
        async def scenario(timeout_sec):
            dispatched = []
            manager = TokenizerManager.__new__(TokenizerManager)
            manager.server_args = SimpleNamespace(
                tokenizer_worker_num=1,
                load_format="auto",
                enable_weight_runtime_manifest=False,
            )
            manager.model_update_lock = RWLock()
            manager.model_update_attempts = {}
            manager.weight_update_fail_closed = False
            manager.auto_create_handle_loop = lambda: None
            manager.abort_request = lambda **_kwargs: None
            manager._dispatch_to_scheduler = dispatched.append

            request = UpdateWeightFromDiskReqInput(
                model_path="/weights",
                timeout_sec=timeout_sec,
            )
            with self.assertRaisesRegex(
                ValueError, "timeout_sec must be a positive finite number"
            ):
                await TokenizerManager.update_weights_from_disk(manager, request)

            self.assertFalse(manager.weight_update_fail_closed)
            self.assertEqual(dispatched, [])

        for timeout_sec in (0, -1, float("inf"), float("nan"), True, "1"):
            with self.subTest(timeout_sec=timeout_sec):
                asyncio.run(scenario(timeout_sec))


if __name__ == "__main__":
    unittest.main()
