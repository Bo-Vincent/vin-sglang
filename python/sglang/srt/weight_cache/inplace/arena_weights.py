"""Bounded parameter arenas for the stopped-worker experiment only."""

import gc
import time
import weakref

from ..tensor_aliases import capture_tensor_aliases, rebind_tensor_aliases

CAP_BYTES = 1 << 30


def chunks(parameters, names):
    batches, batch, size, dtype = [], [], 0, None
    for name in sorted(names):
        value = parameters[name]
        nbytes = value.numel() * value.element_size()
        if nbytes > CAP_BYTES or not value.is_contiguous():
            raise ValueError(
                "arena requires contiguous parameters no larger than 1 GiB"
            )
        if batch and (size + nbytes > CAP_BYTES or value.dtype != dtype):
            batches.append((batch, size))
            batch, size = [], 0
        batch.append(name)
        size += nbytes
        dtype = value.dtype
    if batch:
        batches.append((batch, size))
    return batches


def active_blocks(snapshot, pointers):
    """Find old tensor addresses still backed by active allocator blocks."""
    alive = []
    for segment in snapshot:
        cursor = segment["address"]
        for block in segment["blocks"]:
            address = block.get("address", cursor)
            if block["state"].startswith("active"):
                alive.extend(
                    p for p in pointers if address <= p < address + block["size"]
                )
            cursor = address + block["size"]
    return sorted(set(alive))


def compact_source(torch, model, parameters, names, profile, owners):
    """Replace each completed chunk before allocating the next one or targets."""
    batches = chunks(parameters, names)
    profile.update(
        source_arena_payload_bytes=sum(size for _, size in batches),
        source_arena_count=len(batches),
        source_arena_max_bytes=max((s for _, s in batches), default=0),
        source_release_chunks=[],
        source_empty_cache_seconds=0.0,
    )
    for names, size in batches:
        aliases = capture_tensor_aliases(model, {n: parameters[n] for n in names})
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        originals = [weakref.ref(parameters[n]) for n in names]
        pointers = [parameters[n].data_ptr() for n in names]
        dtype = parameters[names[0]].dtype
        itemsize = parameters[names[0]].element_size()
        arena = torch.empty(
            size // itemsize, dtype=dtype, device=parameters[names[0]].device
        )
        owners.append(arena)
        offset = 0
        for name in names:
            previous = parameters[name]
            view = arena.narrow(0, offset, previous.numel()).view(previous.shape)
            with torch.no_grad():
                view.copy_(previous)
            replacement = torch.nn.Parameter(view, requires_grad=previous.requires_grad)
            replacement.__dict__.update(previous.__dict__)
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            parent._parameters[leaf] = replacement
            parameters[name] = replacement
            offset += previous.numel()
            del previous, view, replacement, parent
        rebind_tensor_aliases(aliases, parameters)
        del aliases
        torch.cuda.synchronize()
        observed = time.monotonic()
        alive = active_blocks(torch.cuda.memory_snapshot(), pointers)
        snapshot_seconds = time.monotonic() - observed
        fallback_seconds = 0.0
        if alive:
            fallback_started = time.monotonic()
            gc.collect()
            alive = active_blocks(torch.cuda.memory_snapshot(), pointers)
            fallback_seconds = time.monotonic() - fallback_started
        row = dict(
            payload_bytes=size,
            snapshot_seconds=snapshot_seconds,
            gc_fallback_seconds=fallback_seconds,
            allocated_before=before,
            allocated_after=torch.cuda.memory_allocated(),
            reserved_after=torch.cuda.memory_reserved(),
            cumulative_peak_allocated=torch.cuda.max_memory_allocated(),
            old_parameter_objects_alive=sum(ref() is not None for ref in originals),
            old_active_allocation_count=len(alive),
            release_verified=not alive,
            flush_performed=False,
            flush_seconds=0.0,
            reserved_before_flush=None,
            reserved_after_flush=None,
        )
        profile["source_release_chunks"].append(row)
        if alive:
            raise RuntimeError(
                "old changed-source allocation remains active after arena replacement"
            )
        row["reserved_before_flush"] = torch.cuda.memory_reserved()
        flush_started = time.monotonic()
        torch.cuda.empty_cache()
        row["flush_seconds"] = time.monotonic() - flush_started
        row["flush_performed"] = True
        row["reserved_after_flush"] = torch.cuda.memory_reserved()
        profile["source_empty_cache_seconds"] += row["flush_seconds"]
        del arena, originals, pointers
    empty_started = time.monotonic()
    torch.cuda.empty_cache()
    profile["source_final_empty_cache_seconds"] = time.monotonic() - empty_started
    profile["source_empty_cache_seconds"] += profile["source_final_empty_cache_seconds"]
    profile["source_release_verified"] = all(
        r["release_verified"] for r in profile["source_release_chunks"]
    )


def allocate_target(torch, model, sources, aliases, profile):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    result = {n: sources[n].detach() for n in aliases}
    batches = chunks(parameters, set(parameters) - set(aliases))
    profile.update(
        target_arena_count=len(batches),
        target_arena_payload_bytes=sum(s for _, s in batches),
        target_arena_max_bytes=max((s for _, s in batches), default=0),
    )
    for names, size in batches:
        first = parameters[names[0]]
        arena = torch.empty(
            size // first.element_size(),
            dtype=first.dtype,
            device=torch.cuda.current_device(),
        )
        offset = 0
        for name in names:
            parameter = parameters[name]
            result[name] = arena.narrow(0, offset, parameter.numel()).view(
                parameter.shape
            )
            offset += parameter.numel()
        del arena
    return result
