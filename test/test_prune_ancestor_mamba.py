"""Tests for _prune_ancestor_mamba: SSM state pruning in multi-turn scenarios."""
import unittest
from array import array

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device


def _setup_cache(max_mamba_ancestors=None):
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    if max_mamba_ancestors is not None:
        server_args.max_mamba_ancestors = max_mamba_ancestors
    set_global_server_args_for_scheduler(server_args)

    size = 128
    dtype = torch.bfloat16
    num_layers = 48
    global_interval = 4
    device = get_device()
    full_attention_layer_ids = [i for i in range(global_interval - 1, num_layers, global_interval)]
    mamba_layers = [i for i in range(num_layers) if i not in full_attention_layer_ids]

    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=4096, n_groups=16,
            num_heads=32, head_dim=128, state_size=128, conv_kernel=4,
        )
        mamba2_cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

    req_to_token_pool = HybridReqToTokenPool(
        size=10, mamba_size=20, mamba_spec_state_size=10, max_context_len=128,
        device=device, enable_memory_saver=False, cache_params=mamba2_cache_params,
        mamba_layer_ids=mamba_layers, enable_mamba_extra_buffer=False,
        speculative_num_draft_tokens=3,
    )
    pool = HybridLinearKVPool(
        size=size, dtype=dtype, page_size=1, head_num=2, head_dim=256,
        full_attention_layer_ids=full_attention_layer_ids,
        enable_kvcache_transpose=False, device=device, enable_memory_saver=False,
        mamba_pool=req_to_token_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=size, dtype=dtype, device=device, kvcache=pool, need_sort=False,
    )
    params = CacheInitParams(
        req_to_token_pool=req_to_token_pool, token_to_kv_pool_allocator=allocator,
        page_size=1, disable=False, max_mamba_ancestors=max_mamba_ancestors,
    )
    tree = MambaRadixCache(params=params)
    return tree, allocator, req_to_token_pool


def _insert_and_prune(tree, alloc, token_ids):
    mamba_slot = tree.req_to_token_pool.mamba_allocator.alloc(1)
    assert mamba_slot is not None
    kv_indices = alloc.alloc(len(token_ids))
    assert kv_indices is not None
    result = tree.insert(InsertParams(
        key=RadixKey(array("q", token_ids), None),
        value=kv_indices, mamba_value=mamba_slot, prev_prefix_len=0,
    ))
    if not result.mamba_exist:
        match = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", token_ids), None)))
        tree._prune_ancestor_mamba(match.last_device_node)
    return result


def _count_mamba_nodes(tree):
    nodes = []
    stack = [tree.root_node]
    while stack:
        n = stack.pop()
        if n.mamba_value is not None:
            nodes.append(n)
        stack.extend(n.children.values())
    return len(nodes)


class TestPruneAncestorMamba(unittest.TestCase):

    # === GREEN TESTS: Expected to pass ===

    def test_no_pruning_when_disabled(self):
        """All 5 Mamba checkpoints kept when pruning is disabled."""
        tree, alloc, _ = _setup_cache(max_mamba_ancestors=None)
        for i in range(1, 6):
            _insert_and_prune(tree, alloc, list(range(1, i * 3 + 1)))
        self.assertEqual(_count_mamba_nodes(tree), 5)

    def test_prune_keeps_last_1(self):
        """With max_mamba_ancestors=1, only the latest Mamba state remains after 5 turns."""
        tree, alloc, pool = _setup_cache(max_mamba_ancestors=1)
        for i in range(5):
            _insert_and_prune(tree, alloc, list(range(1, (i + 1) * 3 + 1)))
        count = _count_mamba_nodes(tree)
        self.assertEqual(count, 1, f"Expected 1 Mamba node, got {count}")

    def test_kv_preserved_after_prune(self):
        """Tombstoned nodes still retain KV cache."""
        tree, alloc, _ = _setup_cache(max_mamba_ancestors=1)
        for i in range(3):
            _insert_and_prune(tree, alloc, list(range(1, (i + 1) * 3 + 1)))
        stack = [tree.root_node]
        while stack:
            n = stack.pop()
            if n != tree.root_node and hasattr(n, 'value') and n.value is not None:
                self.assertTrue(len(n.value) > 0 or len(n.key) == 0,
                    "Non-root node should have KV value after Mamba pruning")
            stack.extend(n.children.values())

    def test_prefix_match_finds_mamba(self):
        """After pruning, prefix match finds the latest Mamba checkpoint."""
        tree, alloc, _ = _setup_cache(max_mamba_ancestors=1)
        turns = []
        for i in range(4):
            t = list(range(1, (i + 1) * 3 + 1))
            turns.append(t)
            _insert_and_prune(tree, alloc, t)

        # Last turn: has Mamba
        m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", turns[-1]), None)))
        self.assertIsNotNone(m.last_device_node.mamba_value, "Last turn should have Mamba")

        # N-1 turn: tombstoned (max_mamba_ancestors=1 keeps only current)
        m2 = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", turns[-2]), None)))
        self.assertIsNone(m2.last_device_node.mamba_value, "N-1 turn should be tombstoned with max=1")

        # Turn 0: also tombstoned
        m0 = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", turns[0]), None)))
        self.assertIsNone(m0.last_device_node.mamba_value, "Turn 0 should be tombstoned")

    def test_locked_nodes_not_pruned(self):
        """Nodes with mamba_lock_ref > 0 are skipped during pruning."""
        tree, alloc, _ = _setup_cache(max_mamba_ancestors=None)  # disable auto-prune first
        nodes = []
        for i in range(3):
            t = list(range(1, (i + 1) * 3 + 1))
            mamba_slot = tree.req_to_token_pool.mamba_allocator.alloc(1)
            kv_indices = alloc.alloc(len(t))
            tree.insert(InsertParams(
                key=RadixKey(array("q", t), None),
                value=kv_indices, mamba_value=mamba_slot, prev_prefix_len=0,
            ))
            m = tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", t), None)))
            nodes.append(m.last_device_node)

        # All 3 should have mamba
        for n in nodes:
            self.assertIsNotNone(n.mamba_value)

        # Lock turn 0, then prune with max=1 from turn 2
        # max=1 means keep 0 ancestors, so node[1] (middle) gets pruned
        # but node[0] is locked so it survives
        nodes[0].mamba_lock_ref = 1
        tree.max_mamba_ancestors = 1
        tree._prune_ancestor_mamba(nodes[2])
        self.assertIsNotNone(nodes[0].mamba_value, "Locked node must not be pruned")
        self.assertIsNone(nodes[1].mamba_value, "Unlocked ancestor pruned with max=1")
        nodes[0].mamba_lock_ref = 0

    def test_max_ancestors_2(self):
        """With max_mamba_ancestors=2, last 2 Mamba states survive (current + 1 ancestor)."""
        tree, alloc, _ = _setup_cache(max_mamba_ancestors=2)
        for i in range(5):
            _insert_and_prune(tree, alloc, list(range(1, (i + 1) * 3 + 1)))
        count = _count_mamba_nodes(tree)
        self.assertLessEqual(count, 2, f"Expected <= 2 Mamba nodes, got {count}")

    def test_independent_sessions(self):
        """Pruning session A doesn't affect session B."""
        tree, alloc, _ = _setup_cache(max_mamba_ancestors=1)
        # Session A
        for i in range(3):
            _insert_and_prune(tree, alloc, list(range(100, 100 + (i + 1) * 3)))
        # Session B
        for i in range(3):
            _insert_and_prune(tree, alloc, list(range(200, 200 + (i + 1) * 3)))
        count = _count_mamba_nodes(tree)
        # max_mamba_ancestors=1: each session keeps only 1 (latest), total = 2
        self.assertEqual(count, 2, f"Two sessions with max=1, expected 2 Mamba nodes, got {count}")

    def test_single_turn_no_prune(self):
        """Single turn has no ancestors to prune."""
        tree, alloc, pool = _setup_cache(max_mamba_ancestors=1)
        _insert_and_prune(tree, alloc, [1, 2, 3])
        self.assertEqual(_count_mamba_nodes(tree), 1)

    def test_mamba_slots_freed(self):
        """Verify freed Mamba slots are returned to allocator."""
        tree, alloc, pool = _setup_cache(max_mamba_ancestors=1)
        avail_before = pool.mamba_allocator.available_size()
        for i in range(5):
            _insert_and_prune(tree, alloc, list(range(1, (i + 1) * 3 + 1)))
        avail_after = pool.mamba_allocator.available_size()
        # 5 slots allocated, 4 freed (only 1 kept), net = 5 - 4 = 1 consumed
        slots_consumed = avail_before - avail_after
        self.assertEqual(slots_consumed, 1,
            f"Expected 1 slot consumed (5 alloc - 4 freed), got {slots_consumed}")


if __name__ == "__main__":
    unittest.main()
