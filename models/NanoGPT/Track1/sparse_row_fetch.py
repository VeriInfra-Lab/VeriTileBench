"""Gather the rows requested by peers and assemble the n-gram cache."""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

WIDTH = 768
ROWS = 4
COLS = 256


@gluon.jit(do_not_specialize=["count", "first", "shard_rows"])
def _serve_rows(shard, asked, served, count, first, shard_rows,
                ROWS: gl.constexpr, COLS: gl.constexpr):
    # Keep the row ID and its eight-word vectors in the same warp.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    row = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    live = row < count
    local = gl.load(asked + row, mask=live, other=0) - first
    valid = (0 <= local) & (local < shard_rows)
    gl.device_assert(valid, "requested shard row out of bounds", mask=live)
    lanes = gl.arange(0, COLS, layout=gl.SliceLayout(0, layout))
    # A rank's n-gram shard can exceed the signed i32 element range.
    source = shard + local[:, None].to(gl.int64) * 768
    destination = served + row[:, None] * 768
    for stripe in gl.static_range(3):
        cols = stripe * COLS + lanes
        bits = gl.load(source + cols[None, :], mask=(live & valid)[:, None], other=0)
        gl.store(destination + cols[None, :], bits, mask=live[:, None])


@gluon.jit(do_not_specialize=["count", "lo", "hi", "first", "shard_rows"])
def _land_ngram(shard, received, want, cache, count, lo, hi, first, shard_rows,
                ROWS: gl.constexpr, COLS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    row = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    live = row < count
    owned = (row >= lo) & (row < hi)
    local = gl.load(want + row, mask=live & owned, other=0) - first
    valid = (0 <= local) & (local < shard_rows)
    gl.device_assert(valid, "requested cache row out of bounds", mask=live & owned)
    wire = row - gl.where(row >= hi, hi - lo, 0)
    source = gl.where(owned, shard, received)
    index = gl.where(owned, local, wire)
    lanes = gl.arange(0, COLS, layout=gl.SliceLayout(0, layout))
    src = source[:, None] + index[:, None].to(gl.int64) * 768
    destination = cache + row[:, None] * 768
    for stripe in gl.static_range(3):
        cols = stripe * COLS + lanes
        bits = gl.load(src + cols[None, :], mask=(live & (~owned | valid))[:, None], other=0)
        gl.store(destination + cols[None, :], bits, mask=live[:, None])


def _bank(t, device):
    return (t.device == device and t.dtype == torch.bfloat16 and t.ndim == 2 and t.shape[1] == WIDTH
            and t.is_contiguous() and t.data_ptr() % 16 == 0)


def _layout(shard, ids, first):
    return (shard.is_cuda and _bank(shard, shard.device) and 0 <= first < 2**31
            and first + shard.shape[0] < 2**31
            and ids.device == shard.device and ids.dtype == torch.int32 and ids.ndim == 1
            and ids.is_contiguous() and ids.data_ptr() % 16 == 0
            and (ids.numel() + ROWS) * WIDTH < 2**31
            and torch.cuda.get_device_capability(shard.device)[0] == 9)


def serve_rows(shard, asked, first):
    if not _layout(shard, asked, first):
        return None
    count = asked.numel()
    served = shard.new_empty((count, WIDTH))
    if count:
        _serve_rows[(triton.cdiv(count, ROWS),)](
            shard, asked, served, count, first, shard.shape[0],
            ROWS, COLS, num_warps=4, debug=True, sanitize_overflow=False,
        )
    return served


def land_ngram_(cache, shard, received, want, lo, hi, first):
    if (not _layout(shard, want, first) or not _bank(cache, shard.device) or not _bank(received, shard.device)
            or not 0 <= lo <= hi <= want.numel() <= cache.shape[0]
            or received.shape[0] != want.numel() - (hi - lo)):
        return False
    count = want.numel()
    if count:
        start = cache.data_ptr()
        end = start + 2 * count * WIDTH
        for tensor in (shard, received, want):
            address = tensor.data_ptr()
            if tensor.numel() and start < address + tensor.numel() * tensor.element_size() and address < end:
                return False
        _land_ngram[(triton.cdiv(count, ROWS),)](
            shard, received, want, cache,
            count, lo, hi, first, shard.shape[0], ROWS, COLS,
            num_warps=4, debug=True, sanitize_overflow=False,
        )
    return True
