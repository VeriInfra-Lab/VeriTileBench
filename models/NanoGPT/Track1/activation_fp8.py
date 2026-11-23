"""Quantize the training head's BF16 activations in one pass."""

import torch
import triton
import triton.language as tl


ACTIVATION_SCALE = 100 / 448
BLOCK = 512


@triton.jit
def _activation_fp8(X, Y, GAIN: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    x = tl.load(X + index)
    # ATen's scalar division rounds the product through BF16.
    wide = (x.to(tl.float32) * GAIN).to(tl.bfloat16).to(tl.float32)
    finite = wide.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    # At 464 the ties-to-even result is 448. Above it E4M3FN returns a signed NaN.
    overflow = ((wide.to(tl.uint32, bitcast=True) >> 24) | 127).to(tl.uint8)
    bits = tl.where(tl.abs(wide) <= 464.0, finite, overflow)
    tl.store(Y + index, bits)


def quantize_activation(x, scale):
    # Let Inductor fuse the cast with its producer and keep its rounding order.
    if torch.compiler.is_compiling():
        return x.div(scale).to(torch.float8_e4m3fn)
    size = x.numel()
    if (scale != ACTIVATION_SCALE or x.dtype != torch.bfloat16 or not x.is_cuda
            or not x.is_contiguous() or x.is_neg() or not 0 < size < 2**31 or size % BLOCK
            or x.data_ptr() % 16 or (torch.is_grad_enabled() and x.requires_grad)
            or torch.cuda.get_device_capability(x.device)[0] != 9):
        return x.div(scale).to(torch.float8_e4m3fn)
    encoded = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    _activation_fp8[(size // BLOCK,)](x, encoded.view(torch.uint8), 1 / ACTIVATION_SCALE, BLOCK,
                                     num_warps=4, num_stages=1)
    return encoded
