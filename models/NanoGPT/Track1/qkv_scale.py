"""Fuse the QKV partial-amax reduction, floor, and reciprocal multiply."""


@triton.jit
def _qkv_scale_combine(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _qkv_scale_from_partials(partial_amax, scales):
    batch = tl.program_id(0)
    parts = tl.arange(0, 32)
    partials = tl.load(partial_amax + batch * 32 + parts)
    amax = tl.reduce(partials, 0, _qkv_scale_combine)
    # The reference clamp leaves NaN unchanged. The multiply has FP32 operands.
    floored = tl.where(amax < 1.0e-12, 1.0e-12, amax)
    tl.store(scales + batch, floored * (1.0 / 448.0))
