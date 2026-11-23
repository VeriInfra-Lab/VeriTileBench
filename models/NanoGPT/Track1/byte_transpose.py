"""Warp-local transpose of dense, tile-aligned FP8 storage on Hopper."""

from itertools import combinations

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _transpose_tile(src, dst, M, N, m, n, WIDE: gl.constexpr = False):
    if WIDE:
        INPUT: gl.constexpr = gl.BlockedLayout([2, 16], [8, 4], [8, 1], [1, 0])
    else:
        INPUT: gl.constexpr = gl.BlockedLayout([4, 8], [4, 8], [8, 1], [1, 0])
    OUTPUT: gl.constexpr = gl.BlockedLayout([16, 2], [1, 32], [8, 1], [0, 1])
    rows = m + gl.arange(0, 128, layout=gl.SliceLayout(1, INPUT))
    cols = n + gl.arange(0, 128, layout=gl.SliceLayout(0, INPUT))
    tile = gl.load(src + rows[:, None] * N + cols[None, :])
    moved = gl.convert_layout(tile, OUTPUT)
    dst_rows = n + gl.arange(0, 128, layout=gl.SliceLayout(0, OUTPUT))
    dst_cols = m + gl.arange(0, 128, layout=gl.SliceLayout(1, OUTPUT))
    if WIDE:
        gl.store(dst + (dst_rows[None, :] * M + dst_cols[:, None]), moved)
    else:
        gl.store(dst + dst_rows[None, :] * M + dst_cols[:, None], moved)


@gluon.jit(do_not_specialize=["M_tiles", "N_tiles"])
def _byte_transpose(src, dst, M_tiles, N_tiles):
    _transpose_tile(src, dst, M_tiles * 128, N_tiles * 128, gl.program_id(1) * 128, gl.program_id(0) * 128)


@gluon.jit(do_not_specialize=["M_tiles", "N_tiles"])
def _ce_backward_transposes(x, xt, grad, gt, M_tiles, N_tiles):
    column = gl.program_id(0).to(gl.uint32)
    activation = column < 6
    src = gl.where(activation, x, grad)
    dst = gl.where(activation, xt, gt)
    # The launcher bounds both matrices to fewer than 2**31 bytes.
    # Unsigned pitches let the final byte offsets extend without sign handling.
    width = gl.where(activation, 768, N_tiles.to(gl.uint32) * 128)
    tile = gl.where(activation, column, column - 6)
    _transpose_tile(src, dst, M_tiles.to(gl.uint32) * 128, width, gl.program_id(1) * 128, tile * 128, WIDE=True)


def transpose_hopper_bytes(src, dst, m, n):
    if (src.dtype not in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2)
            or dst.dtype != src.dtype or not src.is_cuda or dst.device != src.device
            or m == 0 or n == 0 or m % 128 or n % 128
            or m // 128 > 65535 or m * n >= 2**31
            or not src.is_contiguous() or not dst.is_contiguous()
            or torch.cuda.get_device_capability(src.device)[0] != 9):
        return False
    src_ptr, dst_ptr = src.data_ptr(), dst.data_ptr()
    if src_ptr % 16 or dst_ptr % 16 or abs(src_ptr - dst_ptr) < m * n:
        return False
    _byte_transpose[(n // 128, m // 128)](
        src.view(torch.uint8), dst.view(torch.uint8), m // 128, n // 128,
        num_warps=8, num_stages=2,
    )
    return True


def transpose_hopper_ce_operands(x, xt, grad, gt):
    """Transpose both FP8 weight-gradient operands over one tiled grid."""
    if x.ndim != 2 or grad.ndim != 2:
        return False
    m, width = x.shape
    g_rows, n = grad.shape
    if (width != 768 or m == 0 or m != g_rows or m % 128 or m // 128 > 65535
            or n < 128 or n % 128 or m * max(n, width) >= 2**31
            or xt.shape != (width, m) or gt.shape != (n, m)
            or x.dtype != torch.float8_e4m3fn or xt.dtype != x.dtype
            or grad.dtype != torch.float8_e5m2 or gt.dtype != grad.dtype
            or not x.is_cuda or any(t.device != x.device or not t.is_contiguous() for t in (x, xt, grad, gt))
            or torch.cuda.get_device_capability(x.device)[0] != 9):
        return False
    regions = [(tensor.data_ptr(), tensor.numel()) for tensor in (x, xt, grad, gt)]
    if any(ptr % 16 for ptr, _ in regions) or any(
            a < b + b_size and b < a + a_size
            for (a, a_size), (b, b_size) in combinations(regions, 2)):
        return False
    _ce_backward_transposes[(6 + n // 128, m // 128)](
        x.view(torch.uint8), xt.view(torch.uint8), grad.view(torch.uint8), gt.view(torch.uint8),
        m // 128, n // 128, num_warps=8, num_stages=2,
    )
    return True
