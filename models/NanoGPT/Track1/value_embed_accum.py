"""Accumulate the four value-embedding gradient planes in one Hopper launch."""
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _scatter_four_planes(g0, g1, g2, g3, token_ids, out, V: gl.constexpr):
    pid = gl.program_id(0)
    token = gl.load(token_ids + pid, eviction_policy="evict_last")
    gl.device_assert((0 <= token) & (token < V), "token id out of range")
    # One warp per plane. Adjacent lanes each load eight consecutive words.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    plane = gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    col = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
    g = gl.where(plane < 2, gl.where(plane == 0, g0, g1), gl.where(plane == 2, g2, g3))
    source = g[:, None] + pid * 768 + col[None, :]
    target = out + (plane[:, None] * V + token) * 768 + col[None, :]
    for stripe in gl.static_range(3):
        value = gl.load(source + stripe * 256, eviction_policy="evict_first").to(gl.float32)
        gl.atomic_add(target + stripe * 256, value, sem="relaxed")


def _hopper(tensor):
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] == 9


def accumulate_four_planes_(token_ids, plane_grads, out, vocab, width):
    t = token_ids.numel()
    if (len(plane_grads) != 4 or width != 768 or not 0 < t * width < 2**31
            or not 0 < vocab or out.shape != (4 * vocab, width) or out.numel() >= 2**31
            or not _hopper(out)):
        return False
    if (out.data_ptr() % 16 or token_ids.device != out.device or token_ids.data_ptr() % 16
            or any(g.device != out.device or g.shape != (t, width) or g.dtype != torch.bfloat16
                   for g in plane_grads)):
        return False
    start = out.data_ptr()
    stop = start + out.numel() * out.element_size()
    for g in (token_ids, *plane_grads):
        storage = g.untyped_storage()
        address = storage.data_ptr()
        if start < address + storage.nbytes() and address < stop:
            return False
    grads = [g.contiguous() for g in plane_grads]
    if any(g.data_ptr() % 16 for g in grads):
        return False
    _scatter_four_planes[(t,)](*grads, token_ids, out, V=vocab, num_warps=4, num_stages=1)
    return True
