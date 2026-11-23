"""Gather valid sampled-softmax IDs and write both FP8 byte layouts."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _candidate_tile(weights, candidates, rows, transposed, m, n, P: gl.constexpr):
    M: gl.constexpr = 128
    # Transfer sixteen adjacent bytes per row and per transposed column.
    IN: gl.constexpr = gl.BlockedLayout([4, 16], [4, 8], [8, 1], [1, 0])
    OUT: gl.constexpr = gl.BlockedLayout([16, 4], [1, 32], [8, 1], [0, 1])
    start_m = m * M
    start_n = n * 128
    rm = gl.arange(0, M, layout=gl.SliceLayout(1, IN))
    rn = gl.arange(0, 128, layout=gl.SliceLayout(0, IN))
    vocab = gl.load(candidates + start_m + rm).to(gl.int32)
    tile = gl.load(weights + (vocab[:, None] * 768 + (start_n + rn[None, :])))
    gl.store(rows + (start_m * 768 + start_n) + (rm[:, None] * 768 + rn[None, :]), tile)
    moved = gl.convert_layout(tile, OUT)
    tm = gl.arange(0, M, layout=gl.SliceLayout(1, OUT))
    tn = gl.arange(0, 128, layout=gl.SliceLayout(0, OUT))
    gl.store(transposed + (start_n * P + start_m) + (tn[None, :] * P + tm[:, None]), moved)


@gluon.jit
def _gather_candidate_rows(weights, candidates, rows, transposed, P: gl.constexpr):
    _candidate_tile(weights, candidates, rows, transposed, gl.program_id(1), gl.program_id(0), P)


@gluon.jit
def _inverse_tile(candidates, inverse, block, P: gl.constexpr):
    start = block * 256
    stop = gl.minimum(start + 256, 50304)
    # Search the last class of each 16-entry block, then count the
    # entries preceding the key in the one block containing its rank.
    SEARCH_BLOCK: gl.constexpr = 16
    lo = 0
    hi = P // SEARCH_BLOCK
    while lo < hi:
        mid = lo + (hi - lo) // 2
        before = gl.load(candidates + (mid + 1) * SEARCH_BLOCK - 1).to(gl.int32) < start
        lo = gl.where(before, mid + 1, lo)
        hi = gl.where(before, hi, mid)

    within = gl.arange(0, 32, layout=gl.BlockedLayout([1], [32], [8], [0]))
    index = lo * SEARCH_BLOCK + within
    value = gl.load(candidates + index, (within < SEARCH_BLOCK) & (index < P), other=50304).to(gl.int32)
    lo = lo * SEARCH_BLOCK + gl.sum((value < start).to(gl.int32), 0)

    lane = gl.arange(0, 256, layout=gl.BlockedLayout([1], [32], [8], [0]))
    cls = start + lane
    slot = lo + lane
    vocab = gl.load(candidates + slot, slot < P, other=50304).to(gl.int32)
    # Each CTA clears and fills only its own interval. The barrier orders
    # those stores even when two different lanes target the same word.
    gl.store(inverse + cls, -1, cls < stop)
    gl.thread_barrier()
    gl.store(inverse + vocab, slot, (vocab >= start) & (vocab < stop))


@gluon.jit
def _gather_candidate_rows_and_inverse(weights, candidates, rows, transposed, inverse, P: gl.constexpr):
    pid = gl.program_id(0)
    if pid < 197:
        _inverse_tile(candidates, inverse, pid, P)
    else:
        tile = pid - 197
        _candidate_tile(weights, candidates, rows, transposed, tile // 6, tile % 6, P)


def gather_sampled_rows(weights, candidates, rows, transposed, *, inverse=None):
    """Fast path for the candidates produced by the pinned CandidateBuilder."""
    p = candidates.numel()
    if (p not in (10240, 14336, 24576)
            or weights.shape != (50304, 768) or candidates.shape != (p,)
            or rows.shape != (p, 768) or transposed.shape != (768, p)
            or weights.dtype != torch.uint8 or rows.dtype != torch.uint8 or transposed.dtype != torch.uint8
            or candidates.dtype != torch.int64 or not weights.is_cuda
            or any(t.device != weights.device or not t.is_contiguous()
                   for t in (weights, candidates, rows, transposed))
            or torch.cuda.get_device_capability(weights.device)[0] != 9):
        return False
    if inverse is None:
        _gather_candidate_rows[(6, p // 128)](weights, candidates, rows, transposed, P=p, num_warps=8, num_stages=2)
    else:
        if (inverse.shape != (50304,) or inverse.dtype != torch.int32 or inverse.device != weights.device
                or not inverse.is_contiguous()):
            return False
        start = inverse.data_ptr()
        stop = start + inverse.numel() * inverse.element_size()
        for tensor in (weights, candidates, rows, transposed):
            address = tensor.data_ptr()
            if start < address + tensor.numel() * tensor.element_size() and address < stop:
                return False
        # CandidateBuilder enumerates the vocabulary bitmap in ascending order.
        _gather_candidate_rows_and_inverse[(197 + 6 * (p // 128),)](
            weights, candidates, rows, transposed, inverse, P=p, num_warps=8, num_stages=2)
    return True
