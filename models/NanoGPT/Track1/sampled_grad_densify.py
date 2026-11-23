"""Densify the sampled LM-head gradient on the pinned Hopper shapes."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _densify_gluon(src, pos, dst, P: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [8, 1], [1, 0])
    v = gl.program_id(0) * 128 + gl.arange(0, 128, layout=gl.SliceLayout(0, L))
    r = gl.program_id(1) * 64 + gl.arange(0, 16, layout=gl.SliceLayout(1, L))
    p = gl.load(pos + v)
    keep = p[None, :] >= 0
    base = src + r[:, None] * P + p[None, :]
    out = dst + (r[:, None] * 50304 + v[None, :])
    # Each lane reuses eight compact addresses through four 16-row phases.
    for phase in gl.static_range(0, 4):
        values = gl.load(base + phase * 16 * P, mask=keep, other=0)
        gl.store(out + phase * 16 * 50304, values)


def densify_sampled_gradient(src, pos, dst):
    if (src.ndim != 2 or src.shape[0] != 768 or src.shape[1] not in (10240, 14336, 24576)
            or dst.shape != (768, 50304) or pos.shape != (50304,)
            or src.dtype != torch.bfloat16 or dst.dtype != src.dtype or pos.dtype != torch.int32
            or not dst.is_cuda or src.device != dst.device or pos.device != dst.device
            or not (src.is_contiguous() and pos.is_contiguous() and dst.is_contiguous())
            or torch.cuda.get_device_capability(dst.device)[0] != 9):
        return False
    src_ptr, pos_ptr, dst_ptr = src.data_ptr(), pos.data_ptr(), dst.data_ptr()
    if (src_ptr % 16 or pos_ptr % 16 or dst_ptr % 16
            or dst_ptr - 2 * src.numel() < src_ptr < dst_ptr + 2 * dst.numel()
            or dst_ptr - 4 * pos.numel() < pos_ptr < dst_ptr + 2 * dst.numel()):
        return False
    _densify_gluon[(393, 12)](src, pos, dst, P=src.shape[1], num_warps=8, num_stages=3)
    return True
