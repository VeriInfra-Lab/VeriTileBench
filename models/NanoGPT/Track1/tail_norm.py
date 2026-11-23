"""Restore the five Track 1 weight norms on the final validation path."""
import itertools

import torch
import triton
import triton.language as tl


BLOCK = 4096
WIDTH = 768
SIZES = (50304 * WIDTH, 50304 * WIDTH, 24 * 2816 * WIDTH, 48 * 256 * WIDTH, 16 * WIDTH * WIDTH)
ENDS = tuple(itertools.accumulate(size // BLOCK for size in SIZES))
ATEN_REVISION = "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"


@triton.jit
def _restore_tail_norms(embed, head, mlp, qk, vo, ratios, ENDS: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    if block < ENDS[0]:
        bank, local, slot = embed, block, 0
    elif block < ENDS[1]:
        bank, local, slot = head, block - ENDS[0], 1
    elif block < ENDS[2]:
        bank, local, slot = mlp, block - ENDS[1], 2
    elif block < ENDS[3]:
        bank, local, slot = qk, block - ENDS[2], 3
    else:
        bank, local, slot = vo, block - ENDS[3], 4
    lane = local * BLOCK + tl.arange(0, BLOCK)
    ratio = tl.load(ratios + slot)
    value = tl.load(bank + lane).to(tl.float32)
    tl.store(bank + lane, (ratio * value).to(tl.bfloat16))


def can_restore_tail_norms(weights) -> bool:
    """Check the pinned ATen case and full, non-overlapping tiles."""
    if torch.version.git_version != ATEN_REVISION or torch.version.cuda is None:
        return False
    if tuple(t.numel() for t in weights) != SIZES or not weights[0].is_cuda:
        return False
    device = weights[0].device
    if not all(t.dtype == torch.bfloat16 and t.device == device and t.is_contiguous()
               and t.data_ptr() % 16 == 0 for t in weights):
        return False
    spans = sorted((t.data_ptr(), t.data_ptr() + t.numel() * 2) for t in weights)
    if any(left[1] > right[0] for left, right in itertools.pairwise(spans)):
        return False
    return torch.cuda.get_device_capability(device)[0] == 9


def tail_norms(weights, direct: bool):
    if direct:
        # ATen widens each BF16 input inside the same FP32 reduction tree.
        return torch.stack([torch.linalg.vector_norm(w, dtype=torch.float32) for w in weights])
    return torch.stack([w.float().norm() for w in weights])


def restore_tail_norms_(weights, ratios):
    _restore_tail_norms[(ENDS[-1],)](*weights, ratios, ENDS, BLOCK, num_warps=8)


def warm_tail_norms(weights):
    if can_restore_tail_norms(weights):
        ratios = torch.empty(len(weights), dtype=torch.float32, device=weights[0].device)
        _restore_tail_norms.warmup(*weights, ratios, ENDS, BLOCK, num_warps=8, grid=(ENDS[-1],))
