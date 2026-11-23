"""Fuse value Adam gradient scaling, the update and workspace clearing."""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

ELEMENTS = 25152 * 768
PROFILE_WARPS = {512: 8, 1024: 4}
BETAS = {period: (0.75 ** (period // 2), 0.95 ** (period // 2)) for period in (2, 4)}


@gluon.jit(do_not_specialize=["step_size", "decay"])
def _update_with_scaled_read(
    first, gradient, second, step_size, weight, decay,
    first_out, second_out, weight_out,
    BLOCK: gl.constexpr, BETA1: gl.constexpr, BETA2: gl.constexpr,
    FIRST_MIX: gl.constexpr, SECOND_MIX: gl.constexpr, CLEAR_ALL: gl.constexpr,
):
    width: gl.constexpr = 8 if BLOCK == 1024 else 2
    warps: gl.constexpr = 4 if BLOCK == 1024 else 8
    layout: gl.constexpr = gl.BlockedLayout([width], [32], [warps], [0])
    lane = gl.arange(0, BLOCK, layout=layout)
    i = gl.program_id(0) * BLOCK + lane

    m = gl.load(first + i)
    half = gl.load(gradient + i)
    v = gl.load(second + i)
    p = gl.load(weight + i).to(gl.float32)

    # A warp with a NaN preserves the scalar conversion and its payload behavior.
    if BLOCK == 1024 and not CLEAR_ALL:
        scaled = gl.inline_asm_elementwise(
            """        {
        .reg .b32 %isnan<4>, %flags, %gain;
        .reg .pred %nanLane, %slow;
        .reg .b16 %a, %b;
        .reg .f32 %lo, %hi;
        mov.b32 %gain, 0x30003000;
        max.NaN.f16x2 %isnan0, $4, $5;
        max.NaN.f16x2 %isnan1, $6, $7;
        max.NaN.f16x2 %flags, %isnan0, %isnan1;
        set.nan.f16x2.f16x2 %flags, %flags, %flags;
        setp.ne.b32 %nanLane, %flags, 0;
        vote.sync.any.pred %slow, %nanLane, 0xffffffff;
        @%slow bra.uni L__half_product_slow;
        mul.rn.f16x2 $0, $4, %gain;
        mul.rn.f16x2 $1, $5, %gain;
        mul.rn.f16x2 $2, $6, %gain;
        mul.rn.f16x2 $3, $7, %gain;
        bra.uni L__half_product_done;
        L__half_product_slow:
        mov.b32 {%a, %b}, $4;
        cvt.f32.f16 %lo, %a;
        cvt.f32.f16 %hi, %b;
        mul.f32 %lo, %lo, 0f3e000000;
        mul.f32 %hi, %hi, 0f3e000000;
        cvt.rn.f16.f32 %a, %lo;
        cvt.rn.f16.f32 %b, %hi;
        mov.b32 $0, {%a, %b};
        mov.b32 {%a, %b}, $5;
        cvt.f32.f16 %lo, %a;
        cvt.f32.f16 %hi, %b;
        mul.f32 %lo, %lo, 0f3e000000;
        mul.f32 %hi, %hi, 0f3e000000;
        cvt.rn.f16.f32 %a, %lo;
        cvt.rn.f16.f32 %b, %hi;
        mov.b32 $1, {%a, %b};
        mov.b32 {%a, %b}, $6;
        cvt.f32.f16 %lo, %a;
        cvt.f32.f16 %hi, %b;
        mul.f32 %lo, %lo, 0f3e000000;
        mul.f32 %hi, %hi, 0f3e000000;
        cvt.rn.f16.f32 %a, %lo;
        cvt.rn.f16.f32 %b, %hi;
        mov.b32 $2, {%a, %b};
        mov.b32 {%a, %b}, $7;
        cvt.f32.f16 %lo, %a;
        cvt.f32.f16 %hi, %b;
        mul.f32 %lo, %lo, 0f3e000000;
        mul.f32 %hi, %hi, 0f3e000000;
        cvt.rn.f16.f32 %a, %lo;
        cvt.rn.f16.f32 %b, %hi;
        mov.b32 $3, {%a, %b};
        L__half_product_done:
        }""",
            constraints="=r,=r,=r,=r,r,r,r,r", args=[half], dtype=gl.float16, is_pure=True, pack=8,
        )
    else:
        raw = half.to(gl.float32)
        scaled = gl.inline_asm_elementwise(
            """{
        .reg .f32 %scaled;
        mul.f32 %scaled, $1, 0f3e000000;
        cvt.rn.f16.f32 $0, %scaled;
    }""",
            constraints="=h,f", args=[raw], dtype=gl.float16, is_pure=True, pack=1,
        )
    g = scaled.to(gl.float32)
    if CLEAR_ALL:
        gl.store(gradient + i, 0.0)
    elif BLOCK == 1024:
        gl.inline_asm_elementwise(
            """{
        .reg .b32 %dirty;
        .reg .pred %write;
        lop3.b32 %dirty, $10, $11, $12, 0xfe;
        or.b32 %dirty, %dirty, $13;
        setp.ne.b32 %write, %dirty, 0;
        mov.b32 $0, 0;
        mov.b32 $1, 0;
        @%write st.global.v4.b32 [$2], {$0, $0, $0, $0};
    }""",
            constraints="=r,=r,l,l,l,l,l,l,l,l,r,r,r,r",
            args=[(gradient + i).to(gl.uint64), half], dtype=gl.uint8, is_pure=False, pack=8,
        )
    else:
        gl.inline_asm_elementwise(
            """{
        .reg .pred %write;
        setp.ne.b32 %write, $3, 0;
        mov.b32 $0, 0;
        @%write st.global.b32 [$1], $0;
    }""",
            constraints="=r,l,l,r", args=[(gradient + i).to(gl.uint64), half],
            dtype=gl.uint16, is_pure=False, pack=2,
        )
    m_new = gl.fma(m, BETA1, g * FIRST_MIX)
    v_new = gl.fma(v, BETA2, (g * g) * SECOND_MIX)

    direction = m_new / (gl.sqrt_rn(v_new) + 1e-10)
    applied_step = step_size * direction
    active = (applied_step * p > 0.0).to(gl.float32)
    masked = p * active
    update = gl.fma(decay, masked, applied_step)
    p_new = p - update

    gl.store(first_out + i, m_new)
    gl.store(second_out + i, v_new)
    gl.store(weight_out + i, p_new)


def launch_scaled_read(weight, raw_gradient, first, second, step_size, decay, period, block, clear_all):
    beta1, beta2 = BETAS[period]
    _update_with_scaled_read[(ELEMENTS // block,)](
        first, raw_gradient, second, step_size.item(), weight, decay.item(), first, second, weight,
        block, beta1, beta2, 1.0 - beta1, 1.0 - beta2, clear_all, num_warps=PROFILE_WARPS[block], num_stages=1,
        enable_fp_fusion=False,
    )


def probe_scaled_read(optimizer, param, raw_gradient, world):
    import torch

    if (world != 8 or raw_gradient.device.type != "cuda"
            or tuple(raw_gradient.shape) != (25152, 768)
            or torch.version.git_version != "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"
            or triton.__version__ != "3.6.0"):
        return None
    cfg = optimizer.param_cfgs[param]
    if cfg.eps != 1e-10 or cfg.comms != "sharded" or cfg.chunk_size != 25152:
        return None
    period = next((tick for tick, betas in BETAS.items() if betas == tuple(cfg.adam_betas)), None)
    if period is None:
        return None
    p = param[optimizer.rank * cfg.chunk_size:(optimizer.rank + 1) * cfg.chunk_size]
    state = optimizer.param_states[param]
    operands = (
        (p, torch.bfloat16, (25152, 768)),
        (raw_gradient, torch.float16, (25152, 768)),
        (state["exp_avg"], torch.float32, (25152, 768)),
        (state["exp_avg_sq"], torch.float32, (25152, 768)),
    )
    if not all(x.device == p.device and x.dtype == dtype and tuple(x.shape) == shape
               and x.is_contiguous() and x.data_ptr() % 16 == 0 for x, dtype, shape in operands):
        return None
    if not all(x.device.type == "cpu" and x.dtype == torch.float32 and x.shape == ()
               for x in (optimizer._step_size_t, optimizer._eff_wd_t)):
        return None
    props = torch.cuda.get_device_properties(p.device)
    if (props.major, props.minor, props.multi_processor_count) != (9, 0, 132):
        return None
    from track_1_short.perf.kernels.read_scale_dispatch import find_installed_launch

    profile = find_installed_launch(period, p.device.index)
    return None if profile is None else (period, profile[0])
