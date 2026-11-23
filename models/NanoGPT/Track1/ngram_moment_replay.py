"""Bring the second moments of the next cycle's local n-gram rows up to date."""
import torch
import triton
import triton.language as tl

ROWS = 32


@triton.jit(do_not_specialize=["count", "offset", "first", "event"])
def _replay_moments(moment, stamps, history, want, count, offset, first, event,
                    BLOCK: tl.constexpr, POSITIVE_HISTORY: tl.constexpr = False):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = slot < count
    row = tl.load(want + offset + slot, live, other=0) - first
    last = tl.load(stamps + row, live, other=event)
    # Use one shared history index per iteration. Each row joins at last + 1.
    due = live & (last < event)
    if POSITIVE_HISTORY:
        value = tl.load(moment + row, due, other=0.0)
        # A positive-zero word never changes under the trainer's beta2 values.
        due = due & (value.to(tl.int32, bitcast=True) != 0)
        replay_from = tl.where(due, last, event)
        start = tl.min(replay_from, axis=0)
    else:
        replay_from = last
        start = tl.min(tl.minimum(last, event), axis=0)
        value = tl.load(moment + row, due, other=0.0)
    for e in range(start + 1, event + 1):
        beta = tl.load(history + e)
        value = tl.where(replay_from < e, value * beta, value)
    tl.store(moment + row, value, due)
    tl.store(stamps + row, event, live & (last != event))


def _hopper(tensor):
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] == 9


def replay_next_moments_(shard, moment, stamps, history, want, lo, hi, first, event, *,
                         positive_history=False):
    if (not _hopper(moment) or shard.device != moment.device or shard.dtype != torch.bfloat16
            or not shard.is_contiguous() or shard.shape[:1] != moment.shape
            or moment.dtype != torch.float32 or stamps.dtype != torch.int32
            or history.dtype != torch.float32 or want.dtype != torch.int32
            or not all(t.device == moment.device and t.ndim == 1 and t.is_contiguous()
                       and t.data_ptr() % 16 == 0 for t in (moment, stamps, history, want))
            or stamps.shape != moment.shape
            or not 0 <= lo <= hi <= want.numel() < 2**31 - ROWS
            or not 0 < event < min(history.numel(), 2**31 - 1)
            or not 0 <= first <= first + moment.numel() < 2**31):
        return False
    count = hi - lo
    if count:
        regions = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                         for t in (moment, stamps, history, want) if t.numel())
        if any(end > first_address for (_, end), (first_address, _) in zip(regions, regions[1:])):
            return False
        _replay_moments[(triton.cdiv(count, ROWS),)](
            moment, stamps, history, want, count, lo, first, event, ROWS,
            POSITIVE_HISTORY=positive_history, num_warps=1,
        )
    return True
