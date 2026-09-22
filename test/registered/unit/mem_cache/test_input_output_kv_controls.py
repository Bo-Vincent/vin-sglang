"""Pin lifetime and output exclusion exercise real cache ownership paths."""

import unittest
from array import array

import torch

from sglang.srt import runtime_context
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_radix_cache import ComponentType, UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs


class TestInputOutputKvControls(unittest.TestCase):
    cache_class = RadixCache
    page_size = 1

    def setUp(self):
        runtime_context.publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(runtime_context.reset_context)
        self.kv_pool = MHATokenToKVPool(
            size=64,
            page_size=self.page_size,
            dtype=torch.float32,
            head_num=1,
            head_dim=4,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator_class = (
            TokenToKVPoolAllocator
            if self.page_size == 1
            else PagedTokenToKVPoolAllocator
        )
        kwargs = {} if self.page_size == 1 else {"page_size": self.page_size}
        self.allocator = allocator_class(
            size=64,
            dtype=torch.float32,
            device="cpu",
            kvcache=self.kv_pool,
            need_sort=False,
            **kwargs,
        )
        self.pool = ReqToTokenPool(4, 64, "cpu", False)
        self.cache = self.cache_class(
            CacheInitParams(
                False,
                self.pool,
                self.allocator,
                self.page_size,
                tree_components=(ComponentType.FULL,),
            )
        )

    def req(self, *, pin=False, disable_output=False, input_len=4):
        ids = array("q", [1, 2, 3, 4, 5, 6])
        slots = self.allocator.alloc(
            (6 + self.page_size - 1) // self.page_size * self.page_size
        )
        self.pool.req_to_token[1, :6] = slots[:6].to(torch.int32)
        req = Req(
            "test",
            "",
            ids[:input_len],
            SamplingParams(max_new_tokens=2),
            pin_input_kv=pin,
            disable_output_kv_cache=disable_output,
        )
        req.output_ids = ids[input_len:]
        req.full_untruncated_fill_ids = ids
        req.set_extend_range(0, 6)
        req.kv.req_pool_idx = 1
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 6
        req.last_node = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q")))
        ).last_device_node
        req.lock_receipt = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        return req

    def match_len(self):
        return len(
            self.cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6])))
            ).device_indices
        )

    def test_output_excluded_during_unfinished_recompute(self):
        req = self.req(disable_output=True)
        self.cache.cache_unfinished_req(req)
        self.assertEqual(self.match_len(), 4)
        self.assertEqual(len(req.prefix_indices), 6)

    def test_pin_survives_finish_and_eviction_without_pinning_output(self):
        req = self.req(pin=True)
        self.cache.cache_unfinished_req(req)
        self.cache.cache_finished_req(req, owned_kv_len=6)
        self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.match_len(), 4)
        self.assertEqual(self.allocator.available_size(), 60)

    def test_repeated_pin_is_idempotent_and_split_keeps_pin(self):
        for _ in range(3):
            req = self.req(pin=True, disable_output=True)
            self.cache.cache_finished_req(req, owned_kv_len=6)
        self.cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [1, 2]))))
        self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.match_len(), 4)
        self.assertEqual(self.allocator.available_size(), 60)
        self.assertEqual(self.cache.protected_size(), 4)

    def test_default_retains_output(self):
        req = self.req()
        self.cache.cache_finished_req(req, owned_kv_len=6)
        self.assertEqual(self.match_len(), 6 // self.page_size * self.page_size)

    def test_finished_policy_uses_common_release_path(self):
        req = self.req(disable_output=True)
        release_kv_cache(req, self.cache)
        self.assertEqual(self.match_len(), 4)
        self.assertEqual(self.allocator.available_size(), 60)

    def test_abort_releases_output_and_keeps_pin_until_flush(self):
        """Aborting a decoded request must release its lock, not its input pin."""
        req = self.req(pin=True, disable_output=True, input_len=5)
        self.cache.cache_unfinished_req(req)
        available_before_abort = self.allocator.available_size()

        # Scheduler.abort_request marks running requests for the next result
        # handler, which resolves the finish reason and uses common KV release.
        req.to_finish = FINISH_ABORT("Cancelled by client")
        req.update_finish_state()
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        release_kv_cache(req, self.cache)
        self.assertFalse(req.kv.holds_kv)
        pinned_slots = (5 + self.page_size - 1) // self.page_size * self.page_size
        self.assertEqual(self.allocator.available_size(), 64 - pinned_slots)
        self.assertGreater(self.allocator.available_size(), available_before_abort)

        self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.match_len(), 5 // self.page_size * self.page_size)
        self.assertEqual(self.cache.protected_size(), pinned_slots)
        self.assertEqual(self.allocator.available_size(), 64 - pinned_slots)

        # Match Scheduler.flush_cache's idle reset order: the cache forgets
        # ownership before the request and KV allocators reclaim all slots.
        self.cache.reset()
        self.pool.clear()
        self.allocator.clear()
        self.assertEqual(self.match_len(), 0)
        self.assertEqual(self.cache.protected_size(), 0)
        self.assertEqual(self.allocator.available_size(), 64)

    def test_historical_shared_output_is_not_freed_on_recompute_finish(self):
        first = self.req()
        self.cache.cache_finished_req(first, owned_kv_len=6)
        req = self.req(disable_output=True)
        allocated = self.pool.req_to_token[1, :6].to(torch.int64).clone()
        self.allocator.free(allocated)
        match = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4, 5, 6])))
        )
        n = len(match.device_indices)
        self.pool.req_to_token[1, :n] = match.device_indices.to(torch.int32)
        req.kv.cache_protected_len = n
        req.kv.kv_committed_len = req.kv.kv_allocated_len = n
        req.last_node = match.last_device_node
        req.lock_receipt = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        self.cache.cache_unfinished_req(req)
        self.assertEqual(req.kv.cache_protected_len, n)
        self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.match_len(), n)
        release_kv_cache(req, self.cache)
        self.assertEqual(self.allocator.available_size(), 64 - n)
        self.assertEqual(self.match_len(), n)


class TestUnifiedInputOutputKvControls(TestInputOutputKvControls):
    cache_class = UnifiedRadixCache


class TestPagedInputOutputKvControls(TestInputOutputKvControls):
    page_size = 4

    def test_partial_input_does_not_free_historical_shared_page(self):
        first = self.req()
        first.output_ids = array("q", [5, 6, 7, 8])
        start = self.pool.req_to_token[1, 4].item()
        self.pool.req_to_token[1, 4:8] = torch.arange(start, start + 4)
        first.kv.kv_committed_len = first.kv.kv_allocated_len = 8
        self.cache.cache_finished_req(first, owned_kv_len=8)

        req = self.req(disable_output=True, input_len=5)
        self.allocator.free(self.pool.req_to_token[1, :6].to(torch.int64).clone())
        match = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", range(1, 9))))
        )
        self.pool.req_to_token[1, :8] = match.device_indices.to(torch.int32)
        req.output_ids = array("q", [6, 7, 8, 9])
        private = self.allocator.alloc(4)
        self.pool.req_to_token[1, 8] = private[0].to(torch.int32)
        req.full_untruncated_fill_ids = array("q", range(1, 10))
        req.set_extend_range(0, 9)
        req.kv.cache_protected_len = 8
        req.kv.kv_committed_len = req.kv.kv_allocated_len = 9
        req.last_node = match.last_device_node
        req.lock_receipt = self.cache.inc_lock_ref(req.last_node).to_dec_params()
        self.cache.cache_unfinished_req(req)
        release_kv_cache(req, self.cache)

        free_pages = set(self.allocator.get_all_free_pages().tolist())
        shared_pages = set((match.device_indices // self.page_size).tolist())
        self.assertFalse(free_pages & shared_pages)
        self.assertEqual(self.allocator.available_size(), 56)

    def test_partial_input_pin_has_independent_page(self):
        req = self.req(pin=True, disable_output=True, input_len=5)
        source = self.pool.req_to_token[1, 4].item()
        self.kv_pool.k_buffer[0][source] = 123
        self.kv_pool.v_buffer[0][source] = 456
        self.cache.cache_unfinished_req(req)
        self.cache.cache_finished_req(req, owned_kv_len=6)
        self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.allocator.available_size(), 56)
        self.assertEqual(self.cache.protected_size(), 8)
        self.assertEqual(self.match_len(), 4)
        self.assertIn(source // 4, self.allocator.get_all_free_pages().tolist())
        tail = next(iter(self.cache.pinned_input_tails.values()))
        self.assertTrue(torch.all(self.kv_pool.k_buffer[0][tail[:1]] == 123))
        self.assertTrue(torch.all(self.kv_pool.v_buffer[0][tail[:1]] == 456))

    def test_partial_input_finished_without_unfinished_callback(self):
        req = self.req(pin=True, disable_output=True, input_len=5)
        self.cache.cache_finished_req(req, owned_kv_len=6)
        self.cache.evict(EvictParams(num_tokens=64))
        self.assertEqual(self.match_len(), 4)
        self.assertEqual(self.allocator.available_size(), 56)

    def test_tail_capacity_failure_aborts_request_without_half_pin(self):
        req = self.req(pin=True, input_len=5)
        other = self.allocator.alloc(self.allocator.available_size())
        self.cache.cache_unfinished_req(req)
        self.assertIsNotNone(req.to_finish)
        self.assertFalse(req.pin_input_kv)
        self.assertEqual(len(self.cache.pinned_input_prefixes), 0)
        self.assertEqual(len(self.cache.pinned_input_tails), 0)
        self.cache.cache_finished_req(req, owned_kv_len=6)
        self.cache.evict(EvictParams(num_tokens=64))
        self.allocator.free(other)
        self.assertEqual(self.allocator.available_size(), 64)


class TestPagedUnifiedInputOutputKvControls(TestPagedInputOutputKvControls):
    cache_class = UnifiedRadixCache


if __name__ == "__main__":
    unittest.main()
