"""Refresh the pinned LM-head FP8 caches in both layouts on Hopper."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from track_1_short.perf.kernels.cross_entropy import quantize_lm_head_dual as _reference


@gluon.jit
def _pack_lm_head_tile(W, ROW, COL, inv, PACKED_GAIN: gl.constexpr):
    M: gl.constexpr = 768
    N: gl.constexpr = 50304
    BLOCK_M: gl.constexpr = 64
    BLOCK_N: gl.constexpr = 128
    # Each lane writes two 16-byte rows and four 8-byte columns.
    IN: gl.constexpr = gl.BlockedLayout([2, 16], [4, 8], [8, 1], [1, 0])
    OUT: gl.constexpr = gl.BlockedLayout([8, 4], [1, 32], [8, 1], [0, 1])
    pid_m = gl.program_id(1)
    pid_n = gl.program_id(0)
    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, IN))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, IN))
    base = pid_m * (BLOCK_M * N) + pid_n * BLOCK_N
    col_base = pid_n * (BLOCK_N * M) + pid_m * BLOCK_M
    w = gl.load(W + base + (offs_m[:, None] * N + offs_n[None, :]))
    if PACKED_GAIN:
        bounded = gl.inline_asm_elementwise(
            "{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; "
            "mul.rn.bf16x2 $0, $1, gain; min.bf16x2 $0, $0, upper; }",
            "=r,r", [w], dtype=gl.bfloat16, is_pure=True, pack=2,
        )
    else:
        scaled = w.to(gl.float32) * inv
        # 0x43e0 is BF16 +448. minnum maps NaNs to that bound.
        # The saturating E4M3 conversion also clamps values below -448.
        bounded = gl.inline_asm_elementwise(
            "{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; "
            "cvt.rn.bf16x2.f32 rounded, $2, $1; min.bf16x2 $0, rounded, upper; }",
            "=r,f,f", [scaled], dtype=gl.bfloat16, is_pure=True, pack=2,
        )
    # Two BF16 values fit one packed E4M3 conversion through FP16.
    q = gl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, $1; "
        "cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; "
        "cvt.rn.satfinite.e4m3x2.f16x2 $0, f16s; }",
        "=h,r", [bounded], dtype=gl.float8e4nv, is_pure=True, pack=2,
    )
    gl.store(ROW + base + (offs_m[:, None] * N + offs_n[None, :]), q)
    moved = gl.convert_layout(q, OUT)
    trans_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, OUT))
    trans_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, OUT))
    gl.store(COL + col_base + (trans_n[None, :] * M + trans_m[:, None]), moved)


@gluon.jit
def _quantize_lm_head_gluon(W, ROW, COL, INV):
    inv = gl.load(INV)
    # Exact FP32 reciprocal used by the pinned LM-head weights.
    if inv.to(gl.int32, bitcast=True) == 0x435fffff:
        _pack_lm_head_tile(W, ROW, COL, inv, True)
    else:
        _pack_lm_head_tile(W, ROW, COL, inv, False)


def quantize_lm_head_dual(w, row, col, inv_w_s):
    if (w.shape == row.shape == col.shape == (768, 50304)
            and w.dtype == torch.bfloat16 and row.dtype == col.dtype == torch.float8_e4m3fn
            and w.is_contiguous() and row.is_contiguous() and col.stride() == (1, 768)
            and inv_w_s.shape == (1,) and inv_w_s.dtype == torch.float32
            and w.is_cuda and w.device == row.device == col.device == inv_w_s.device
            and torch.cuda.get_device_capability(w.device)[0] == 9):
        _quantize_lm_head_gluon[(393, 12)](w, row, col, inv_w_s, num_warps=8, num_stages=2)
    else:
        _reference(w, row, col, inv_w_s)
