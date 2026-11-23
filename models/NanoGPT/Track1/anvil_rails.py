"""Update both ANVIL momentum rails and prepare the BF16 Gram input."""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


BLOCK = 1024


@gluon.jit
def _ordered_lerp(start, end, weight):
    use_end = gl.abs(weight) >= 0.5
    scale = gl.where(use_end, weight - 1.0, weight)
    base = gl.where(use_end, end, start)
    return gl.fma(scale, end - start, base)


@gluon.jit
def _prepare_rails(Grad, Velocity, Momentum, FastBeta, FastWeight, X,
                   CELLS: gl.constexpr, BLOCK: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([8], [32], [4], [0])
    i = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    grad = gl.load(Grad + i).to(gl.float32)
    fast = gl.load(Velocity + i)
    slow = gl.load(Velocity + CELLS + i)
    momentum = gl.load(Momentum)
    beta = gl.load(FastBeta)
    weight = gl.load(FastWeight)
    fast = _ordered_lerp(fast, grad, 1.0 - beta)
    # FP32 word of the original Python expression 1 - RAIL_SLOW_BETA.
    slow = gl.fma(0.019999999552965164, grad - slow, slow)
    blend = gl.fma(weight, fast, (1.0 - weight) * slow)
    lookahead = _ordered_lerp(grad, blend, momentum)
    gl.store(Velocity + i, fast)
    gl.store(Velocity + CELLS + i, slow)
    gl.store(X + i, lookahead.to(gl.bfloat16))


@torch.library.custom_op("veritile_track1::anvil_prepare_rails", mutates_args={"velocity", "x"})
def prepare_rails(grad: torch.Tensor, velocity: torch.Tensor, momentum: torch.Tensor,
                  fast_beta: torch.Tensor, fast_weight: torch.Tensor, x: torch.Tensor) -> None:
    cells = grad.numel()
    _prepare_rails[(cells // BLOCK,)](grad, velocity, momentum, fast_beta, fast_weight, x,
                                    cells, BLOCK, num_warps=4, num_stages=1,
                                    enable_fp_fusion=False)


@prepare_rails.register_fake
def _fake_prepare_rails(grad, velocity, momentum, fast_beta, fast_weight, x):
    return None


@gluon.jit
def _prepare_rails_and_seed(Grad, Velocity, Momentum, FastBeta, FastWeight, X, Counter,
                            CELLS: gl.constexpr, BLOCK: gl.constexpr, MATRICES: gl.constexpr):
    lanes = gl.arange(0, 64, layout=gl.BlockedLayout([1], [32], [4], [0]))
    gl.store(Counter + lanes, 0, (gl.program_id(0) == 0) & (lanes < 6 * MATRICES))
    _prepare_rails(Grad, Velocity, Momentum, FastBeta, FastWeight, X, CELLS, BLOCK)


@torch.library.custom_op("veritile_track1::anvil_prepare_rails_and_seed", mutates_args={"velocity", "x"})
def prepare_rails_and_seed(grad: torch.Tensor, velocity: torch.Tensor, momentum: torch.Tensor,
                           fast_beta: torch.Tensor, fast_weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    cells = grad.numel()
    matrices = grad.shape[0]
    counter = torch.empty((6 * matrices,), dtype=torch.int32, device=x.device)
    _prepare_rails_and_seed[(cells // BLOCK,)](
        grad, velocity, momentum, fast_beta, fast_weight, x, counter,
        cells, BLOCK, matrices, num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return counter


@prepare_rails_and_seed.register_fake
def _fake_prepare_rails_and_seed(grad, velocity, momentum, fast_beta, fast_weight, x):
    return x.new_empty((6 * grad.shape[0],), dtype=torch.int32)
