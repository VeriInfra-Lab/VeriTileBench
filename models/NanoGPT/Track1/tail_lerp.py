"""Update two independent FP32 tail averages from one BF16 weight shard."""
from itertools import pairwise

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

BLOCK = 4096
WARPS = 8


@gluon.jit
def _tail_lerp_pair(src, first, second, rate_first, rate_second, BLOCK: gl.constexpr, WARPS: gl.constexpr):
    lane = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=gl.BlockedLayout([4], [32], [WARPS], [0]))
    weight = gl.load(src + lane).to(gl.float32)
    old_first = gl.load(first + lane)
    gl.store(first + lane, gl.fma(rate_first, weight - old_first, old_first))
    old_second = gl.load(second + lane)
    gl.store(second + lane, gl.fma(rate_second, weight - old_second, old_second))


def lerp_shared_source_(first, second, src, rate_first, rate_second) -> bool:
    """Use the full-tile path when all three aligned buffers are disjoint."""
    elements = src.numel()
    if (not src.is_cuda or src.dtype != torch.bfloat16 or elements == 0
            or elements % BLOCK != 0 or elements >= 2**31):
        return False
    if any(dst.numel() != elements or dst.dtype != torch.float32 or dst.device != src.device
           for dst in (first, second)):
        return False
    tensors = (src, first, second)
    if not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors):
        return False
    spans = sorted((t.data_ptr(), t.data_ptr() + elements * t.element_size()) for t in tensors)
    if any(end > start for (_, end), (start, _) in pairwise(spans)):
        return False
    if torch.cuda.get_device_capability(src.device)[0] != 9:
        return False
    _tail_lerp_pair[(elements // BLOCK,)](
        src, first, second, float(rate_first), float(rate_second), BLOCK=BLOCK, WARPS=WARPS,
        num_warps=WARPS, num_stages=2,
    )
    return True
