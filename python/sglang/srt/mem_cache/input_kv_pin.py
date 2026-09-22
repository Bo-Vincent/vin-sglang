from __future__ import annotations

from sglang.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams


def _input_key(req):
    from sglang.srt.mem_cache.radix_cache import RadixKey

    return RadixKey(
        req.origin_input_ids[: req.input_kv_len],
        req.extra_key,
        cache_salt=req.cache_salt,
    )


def copy_input_tail(cache, req, available_len: int):
    if not req.pin_input_kv or available_len < req.input_kv_len:
        return
    tail_len = req.input_kv_len % cache.page_size
    if not tail_len:
        return
    key = _input_key(req)
    identity = (tuple(key.token_ids), key.extra_key, key.cache_salt)
    if identity in cache.pinned_input_tails:
        return
    allocator = cache.token_to_kv_pool_allocator
    page = allocator.alloc(cache.page_size)
    if page is None:
        cache.evict(EvictParams(num_tokens=cache.page_size))
        page = allocator.alloc(cache.page_size)
    if page is None:
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        req.to_finish = FINISH_ABORT(
            "Insufficient KV capacity to pin the input tail page",
            status_code=503,
            err_type="ServiceUnavailableError",
        )
        if req.finished():
            req.finished_reason = req.to_finish
        req.pin_input_kv = False
        return
    source = cache.req_to_token_pool.req_to_token[
        req.kv.req_pool_idx, req.input_kv_len - tail_len : req.input_kv_len
    ]
    try:
        # The request keeps its writable boundary page; this copy owns input only.
        allocator.get_kvcache().move_kv_cache(page[:tail_len], source)
    except Exception:
        allocator.free_segment(page, start_pos=0)
        raise
    cache.pinned_input_tails[identity] = page


def pin_input_prefix(cache, req, available_len: int):
    if not req.pin_input_kv or available_len < req.input_kv_len:
        return
    key = _input_key(req).page_aligned(cache.page_size)
    identity = (tuple(key.token_ids), key.extra_key, key.cache_salt)
    if identity not in cache.pinned_input_prefixes:
        match = cache.match_prefix(MatchPrefixParams(key=key))
        assert len(match.device_indices) == len(key)
        # This cache-owned lock is independent of every request's lock receipt.
        cache.inc_lock_ref(match.last_device_node)
        cache.pinned_input_prefixes.add(identity)
