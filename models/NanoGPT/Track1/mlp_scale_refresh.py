"""FP8 MLP scalar refresh for the pinned Modded-NanoGPT Track 1 trainer."""

import triton
import triton.language as tl


@triton.jit
def _refresh_scale(raw, previous, gain: tl.constexpr, floor: tl.constexpr):
    exponent = (raw.to(tl.uint32, bitcast=True) >> 23) & 255
    finite = tl.where(exponent == 255, 0.0, raw)
    a = tl.minimum(finite, 1.0e12)
    new = tl.maximum(a * gain, floor)
    return tl.where(a > 0.0, new, previous)


@triton.jit
def _refresh_mlp_scalars(
    UP, DOWN, POST_AMAX, POST_SCALE, DPRE_AMAX, DPRE_SCALE, LAMBDAS, FOLD,
    UP_DEQUANT, DOWN_DEQUANT, DOWN_FOLDED, DOWN_DEQUANT_FOLDED,
    SLOTS: tl.constexpr, LAYERS: tl.constexpr, PARALLEL_SLOT: tl.constexpr, PARALLEL_LAYER: tl.constexpr,
    X_SCALE: tl.constexpr, GRAD_SCALE: tl.constexpr, POST_GAIN: tl.constexpr, DPRE_GAIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.arange(0, BLOCK)
    live = slot < SLOTS
    post = tl.load(POST_AMAX + slot, live, other=0.0)
    old_post = tl.load(POST_SCALE + slot, live, other=0.0)
    dpre = tl.load(DPRE_AMAX + slot, live, other=0.0)
    old_dpre = tl.load(DPRE_SCALE + slot, live, other=0.0)
    update_fold = live & ((slot < LAYERS - 1) | (slot == PARALLEL_SLOT))
    layer = tl.where(slot == PARALLEL_SLOT, PARALLEL_LAYER, slot)
    post_lambda = tl.load(LAMBDAS + 2 * layer + 1, update_fold, other=0.0)
    rounded_lambda = post_lambda.to(tl.bfloat16).to(tl.float32)
    old_fold = tl.load(FOLD + slot, live & ~update_fold, other=0.0)
    fold = tl.where(update_fold, rounded_lambda, old_fold)
    up = tl.load(UP + slot, live, other=0.0)
    down = tl.load(DOWN + slot, live, other=0.0)
    up_dequant = up * X_SCALE
    down_dequant = down * GRAD_SCALE

    tl.store(POST_SCALE + slot, _refresh_scale(post, old_post, POST_GAIN, 1.0e-6), live)
    tl.store(DPRE_SCALE + slot, _refresh_scale(dpre, old_dpre, DPRE_GAIN, 1.0e-12), live)
    tl.store(POST_AMAX + slot, 0.0, live)
    tl.store(DPRE_AMAX + slot, 0.0, live)
    tl.store(FOLD + slot, fold, update_fold)
    tl.store(UP_DEQUANT + slot, up_dequant, live)
    tl.store(DOWN_DEQUANT + slot, down_dequant, live)
    tl.store(DOWN_FOLDED + slot, down * fold, live)
    tl.store(DOWN_DEQUANT_FOLDED + slot, down_dequant * fold, live)


def refresh_mlp_scalars(
    up, down, post_amax, post_scale, dpre_amax, dpre_scale, lambdas, fold,
    up_dequant, down_dequant, down_folded, down_dequant_folded,
    *, parallel_slot, parallel_layer, x_scale, grad_scale, post_gain, dpre_gain,
):
    slots = up.numel()
    layers = lambdas.shape[0]
    _refresh_mlp_scalars[(1,)](
        up, down, post_amax, post_scale, dpre_amax, dpre_scale, lambdas, fold,
        up_dequant, down_dequant, down_folded, down_dequant_folded,
        slots, layers, parallel_slot, parallel_layer, x_scale, grad_scale, post_gain, dpre_gain,
        triton.next_power_of_2(slots), num_warps=1, enable_fp_fusion=False,
    )
