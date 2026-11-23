"""Refresh the pinned Hopper MLP weight bank and both FP8 cache layouts."""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from track_1_short.perf.kernels.mlp import WEIGHT_SCALE_HEADROOM


@triton.jit
def _mlp_scale_max(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _mlp_delayed_scales(partial0, partial1, scale0, scale1, GAIN: tl.constexpr):
    matrix = tl.program_id(0)
    layer = matrix // 2
    up = matrix % 2 == 0
    partial = tl.where(up, partial0, partial1)
    scale = tl.where(up, scale0, scale1)
    lane = tl.arange(0, 1024)
    values = tl.load(partial + layer * 528 + lane, lane < 528, other=-float("inf"))
    amax = tl.reduce(values, 0, _mlp_scale_max)
    product = amax * GAIN
    clamped = tl.where(product < 1.0e-12, 1.0e-12, product)
    tl.store(scale + layer, clamped)


@gluon.jit
def _mlp_paired_pack(bank, row0, row1, col0, col1, scales0, scales1, partial0, partial1):
    INPUT: gl.constexpr = gl.BlockedLayout([2, 16], [4, 8], [8, 1], [1, 0])
    TRANS: gl.constexpr = gl.BlockedLayout([8, 4], [1, 32], [8, 1], [0, 1])
    matrix = gl.program_id(2)
    tile_h = gl.program_id(1)
    pair_d = gl.program_id(0)
    start_h = tile_h * 64
    start_d = pair_d * 128
    layer = matrix // 2
    up = matrix % 2 == 0
    rr = gl.arange(0, 64, layout=gl.SliceLayout(1, INPUT))
    cc = gl.arange(0, 128, layout=gl.SliceLayout(0, INPUT))
    tile_base = start_h * 768 + start_d
    offset = rr[:, None] * 768 + cc[None, :]
    values = gl.load(bank + (matrix * (2816 * 768) + tile_base) + offset).to(gl.float32)
    scale = gl.load(gl.where(up, scales0, scales1) + layer)
    q = gl.maximum(gl.minimum(values / scale, 448.0), -448.0).to(gl.float8e4nv)
    row = gl.where(up, row0, row1) + layer * (2816 * 768)
    col = gl.where(up, col0, col1) + layer * (2816 * 768)
    gl.store(row + tile_base + offset, q)
    # Reserve the 64-byte reduction buffer before the transpose so its
    # lifetime spans the warp-local exchange and the two buffers cannot alias.
    partials = gl.allocate_shared_memory(gl.float32, (2, 8), gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
    t = gl.convert_layout(q, TRANS)
    th = gl.arange(0, 64, layout=gl.SliceLayout(1, TRANS))
    td = gl.arange(0, 128, layout=gl.SliceLayout(0, TRANS))
    gl.store(col + (start_d * 2816 + start_h) + (td[None, :] * 2816 + th[:, None]), t)
    # Each half owns the same amax slot as its original 64-column tile.
    tile = tile_h * 12 + pair_d * 2
    # Each warp reduces its eight rows for the two 64-column halves.
    warp_halves = gl.permute(gl.reshape(gl.abs(values), (8, 8, 2, 64)), (2, 0, 1, 3))
    row_pairs = gl.reshape(warp_halves, (2, 8, 4, 2, 4, 16))
    # Reduce the 16 columns and both rows held by each lane first.
    # The remaining column and row shuffles must each leave the half bit alone.
    local_amax = gl.max(gl.max(row_pairs, 5), 3)
    warp_amax = gl.max(gl.max(local_amax, 3), 2)
    partials.store(warp_amax)
    grouped = partials.load(gl.BlockedLayout([1, 1], [4, 8], [8, 1], [1, 0]))
    amax = gl.max(grouped, 1)
    halves_id = gl.arange(0, 2, layout=amax.type.layout)
    gl.store(gl.where(up, partial0, partial1) + layer * 528 + tile + halves_id, amax)


def refresh_mlp_weight_bank(bank, up, down, *, delayed):
    """Use the fixed bank strides and cache buffers allocated by GPT.init_mlp_fp8."""
    if (bank.shape != (12, 2, 2816, 768) or bank.dtype != torch.bfloat16
            or not bank.is_contiguous() or not bank.is_cuda
            or torch.cuda.get_device_capability(bank.device)[0] != 9):
        return False
    for scales, partial, row, col in (up, down):
        if (scales.shape != (12,) or partial.shape != (12, 528)
                or row.shape != (12, 2816, 768) or col.shape != (12, 768, 2816)
                or scales.dtype != torch.float32 or partial.dtype != torch.float32
                or row.dtype != torch.float8_e4m3fn or col.dtype != row.dtype
                or any(buf.device != bank.device or not buf.is_contiguous()
                       for buf in (scales, partial, row, col))):
            return False
    scales0, partial0, row0, col0 = up
    scales1, partial1, row1, col1 = down
    if delayed:
        _mlp_delayed_scales[(24,)](partial0, partial1, scales0, scales1,
                                  GAIN=WEIGHT_SCALE_HEADROOM / 448.0, num_warps=4)
    _mlp_paired_pack[(6, 44, 24)](bank, row0, row1, col0, col1, scales0, scales1, partial0, partial1,
                                 num_warps=8, num_stages=2)
    return True
