"""Hopper tiles for the replicated Adam update over the existing flat buffers."""
import torch
import triton
import triton.language as tl

BLOCK = 512


def span_tiles(segments: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Intersect each parameter's flat interval with the aligned tile grid."""
    tiles = []
    begin = 0
    for segment, length in segments:
        end = begin + length
        for block in range(begin // BLOCK, triton.cdiv(end, BLOCK)):
            tile_start = block * BLOCK
            lo, hi = max(begin, tile_start), min(end, tile_start + BLOCK)
            if lo < hi:
                tiles.append((lo, hi, segment))
        begin = end
    return tiles


@triton.jit
def _span_adam(P, G, EA, ES, PLAN, SC, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    lo = tl.load(PLAN + pid * 3)
    hi = tl.load(PLAN + pid * 3 + 1)
    segment = tl.load(PLAN + pid * 3 + 2)
    offs = lo // BLOCK * BLOCK + tl.arange(0, BLOCK)
    base = segment * 8
    b1 = tl.load(SC + base)
    a1 = tl.load(SC + base + 1)
    b2 = tl.load(SC + base + 2)
    a2 = tl.load(SC + base + 3)
    eps = tl.load(SC + base + 4)
    step_size = tl.load(SC + base + 5)
    wd = tl.load(SC + base + 6)
    active = tl.load(SC + base + 7)
    m = (offs >= lo) & (offs < hi) & (active != 0.0)

    g = tl.load(G + offs, mask=m, other=0.0).to(tl.float32)
    ea = tl.load(EA + offs, mask=m, other=0.0)
    es = tl.load(ES + offs, mask=m, other=0.0)
    p = tl.load(P + offs, mask=m, other=0.0).to(tl.float32)
    ea = tl.math.fma(a1, g, ea * b1)
    es = tl.math.fma(a2 * g, g, es * b2)
    u = ea / (tl.sqrt(es) + eps) * step_size
    u = tl.math.fma(wd * p, tl.where(u * p > 0.0, 1.0, 0.0), u)
    tl.store(EA + offs, ea, mask=m)
    tl.store(ES + offs, es, mask=m)
    tl.store(P + offs, (p - u).to(P.dtype.element_ty), mask=m)


def update_spans(param: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor,
                 exp_avg_sq: torch.Tensor, spans: torch.Tensor, scalars: torch.Tensor) -> None:
    """The immutable span descriptors are constructed with the flat buffers."""
    _span_adam[(spans.shape[0],)](param, grad, exp_avg, exp_avg_sq, spans, scalars, BLOCK=BLOCK, num_warps=4)
