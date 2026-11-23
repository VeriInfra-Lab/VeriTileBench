"""Warp-local copies and gradient adds for the tied embedding on Hopper."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _warp_bf16_transpose(src, dst, M: gl.constexpr, N: gl.constexpr, ADD: gl.constexpr):
    # Four aligned eight-halfword vectors per lane feed the warp exchange.
    INPUT: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [8, 1], [1, 0])
    OUTPUT: gl.constexpr = gl.BlockedLayout([8, 1], [1, 32], [8, 1], [0, 1])
    m = gl.program_id(1) * 64
    n = gl.program_id(0) * 128
    rows = m + gl.arange(0, 64, layout=gl.SliceLayout(1, INPUT))
    cols = n + gl.arange(0, 128, layout=gl.SliceLayout(0, INPUT))
    tile = gl.load(src + rows[:, None] * N + cols[None, :])
    moved = gl.convert_layout(tile, OUTPUT)
    dst_rows = n + gl.arange(0, 128, layout=gl.SliceLayout(0, OUTPUT))
    dst_cols = m + gl.arange(0, 64, layout=gl.SliceLayout(1, OUTPUT))
    out = dst + (dst_rows[None, :] * M + dst_cols[:, None])
    if ADD:
        old = gl.load(out)
        # Preserve the reference PTX's destination-first operand order.
        moved = gl.inline_asm_elementwise(
            "add.bf16x2 $0, $1, $2;", "=r,r,r", [old, moved],
            dtype=gl.bfloat16, is_pure=True, pack=2,
        )
    gl.store(out, moved)


def transpose_hopper_bf16(src, dst, m, n, *, add):
    shape = (50304, 768) if add else (768, 50304)
    if ((m, n) != shape or src.dtype != torch.bfloat16 or dst.dtype != src.dtype
            or not src.is_cuda or src.device != dst.device
            or not src.is_contiguous() or not dst.is_contiguous()
            or torch.cuda.get_device_capability(src.device)[0] != 9):
        return False
    src_ptr, dst_ptr = src.data_ptr(), dst.data_ptr()
    if src_ptr % 16 or dst_ptr % 16 or abs(src_ptr - dst_ptr) < 2 * m * n:
        return False
    _warp_bf16_transpose[(n // 128, m // 64)](
        src, dst, M=m, N=n, ADD=add, num_warps=8, num_stages=2,
    )
    return True
