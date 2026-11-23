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

    if BLOCK == 1024:
        direction = gl.inline_asm_elementwise(
            """{
    .reg .b32 %mOr, %denAnd, %denOr, %shifted, %moments, %rootAnd, %rootOr;
    .reg .f32 %inverse, %root, %den<8>, %rsq, %rough, %negativeRough, %halfRsq, %residual;
    .reg .pred %invalid, %slow, %hasMoment, %warpHasMoment, %specialRoot, %warpSpecialRoot;
    and.b32 %rootAnd, $16, $17;
    lop3.b32 %rootAnd, %rootAnd, $18, $19, 0x80;
    lop3.b32 %rootAnd, %rootAnd, $20, $21, 0x80;
    lop3.b32 %rootAnd, %rootAnd, $22, $23, 0x80;
    or.b32 %rootOr, $16, $17;
    lop3.b32 %rootOr, %rootOr, $18, $19, 0xfe;
    lop3.b32 %rootOr, %rootOr, $20, $21, 0xfe;
    lop3.b32 %rootOr, %rootOr, $22, $23, 0xfe;
    and.b32 %rootAnd, %rootAnd, 0x20000000;
    and.b32 %rootOr, %rootOr, 0xc0000000;
    xor.b32 %rootAnd, %rootAnd, 0x20000000;
    or.b32 %shifted, %rootAnd, %rootOr;
    setp.ne.u32 %specialRoot, %shifted, 0;
    or.b32 %mOr, $8, $9;
    lop3.b32 %mOr, %mOr, $10, $11, 0xfe;
    lop3.b32 %mOr, %mOr, $12, $13, 0xfe;
    lop3.b32 %mOr, %mOr, $14, $15, 0xfe;
    and.b32 %mOr, %mOr, 0x7f800000;
    setp.eq.or.u32 %specialRoot, %mOr, 0x7f800000, %specialRoot;
    vote.sync.any.pred %warpSpecialRoot, %specialRoot, 0xffffffff;
    @%warpSpecialRoot bra.uni L__adam_root_original;
    rsqrt.approx.ftz.f32 %rsq, $16;
    mul.rn.ftz.f32 %rough, $16, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $16;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den0, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $17;
    mul.rn.ftz.f32 %rough, $17, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $17;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den1, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $18;
    mul.rn.ftz.f32 %rough, $18, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $18;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den2, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $19;
    mul.rn.ftz.f32 %rough, $19, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $19;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den3, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $20;
    mul.rn.ftz.f32 %rough, $20, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $20;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den4, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $21;
    mul.rn.ftz.f32 %rough, $21, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $21;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den5, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $22;
    mul.rn.ftz.f32 %rough, $22, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $22;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den6, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $23;
    mul.rn.ftz.f32 %rough, $23, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $23;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den7, %root, 0f2edbe6ff;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $8;
    rcp.approx.ftz.f32 %inverse, %den1;
    mul.rn.f32 $1, %inverse, $9;
    rcp.approx.ftz.f32 %inverse, %den2;
    mul.rn.f32 $2, %inverse, $10;
    rcp.approx.ftz.f32 %inverse, %den3;
    mul.rn.f32 $3, %inverse, $11;
    rcp.approx.ftz.f32 %inverse, %den4;
    mul.rn.f32 $4, %inverse, $12;
    rcp.approx.ftz.f32 %inverse, %den5;
    mul.rn.f32 $5, %inverse, $13;
    rcp.approx.ftz.f32 %inverse, %den6;
    mul.rn.f32 $6, %inverse, $14;
    rcp.approx.ftz.f32 %inverse, %den7;
    mul.rn.f32 $7, %inverse, $15;
    bra.uni L__adam_divide_join;
    L__adam_root_original:
    or.b32 %moments, $16, $17;
    lop3.b32 %moments, %moments, $18, $19, 0xfe;
    lop3.b32 %moments, %moments, $20, $21, 0xfe;
    lop3.b32 %moments, %moments, $22, $23, 0xfe;
    and.b32 %moments, %moments, 0x7fffffff;
    setp.ne.u32 %hasMoment, %moments, 0;
    vote.sync.any.pred %warpHasMoment, %hasMoment, 0xffffffff;
    @!%warpHasMoment bra.uni L__adam_zero_direction;
    sqrt.rn.f32 %root, $16;
    add.rn.f32 %den0, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $17;
    add.rn.f32 %den1, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $18;
    add.rn.f32 %den2, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $19;
    add.rn.f32 %den3, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $20;
    add.rn.f32 %den4, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $21;
    add.rn.f32 %den5, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $22;
    add.rn.f32 %den6, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $23;
    add.rn.f32 %den7, %root, 0f2edbe6ff;
    L__adam_roots_done:
    and.b32 %denAnd, %den0, %den1;
    lop3.b32 %denAnd, %denAnd, %den2, %den3, 0x80;
    lop3.b32 %denAnd, %denAnd, %den4, %den5, 0x80;
    lop3.b32 %denAnd, %denAnd, %den6, %den7, 0x80;
    or.b32 %denOr, %den0, %den1;
    lop3.b32 %denOr, %denOr, %den2, %den3, 0xfe;
    lop3.b32 %denOr, %denOr, %den4, %den5, 0xfe;
    lop3.b32 %denOr, %denOr, %den6, %den7, 0xfe;
    and.b32 %denAnd, %denAnd, 0x20000000;
    and.b32 %denOr, %denOr, 0xc0000000;
    xor.b32 %denAnd, %denAnd, 0x20000000;
    or.b32 %shifted, %denAnd, %denOr;
    setp.ne.u32 %invalid, %shifted, 0;
    setp.eq.or.u32 %invalid, %mOr, 0x7f800000, %invalid;
    vote.sync.any.pred %slow, %invalid, 0xffffffff;
    @%slow bra.uni L__adam_divide_fallback;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $8;
    rcp.approx.ftz.f32 %inverse, %den1;
    mul.rn.f32 $1, %inverse, $9;
    rcp.approx.ftz.f32 %inverse, %den2;
    mul.rn.f32 $2, %inverse, $10;
    rcp.approx.ftz.f32 %inverse, %den3;
    mul.rn.f32 $3, %inverse, $11;
    rcp.approx.ftz.f32 %inverse, %den4;
    mul.rn.f32 $4, %inverse, $12;
    rcp.approx.ftz.f32 %inverse, %den5;
    mul.rn.f32 $5, %inverse, $13;
    rcp.approx.ftz.f32 %inverse, %den6;
    mul.rn.f32 $6, %inverse, $14;
    rcp.approx.ftz.f32 %inverse, %den7;
    mul.rn.f32 $7, %inverse, $15;
    bra.uni L__adam_divide_join;
    L__adam_divide_fallback:
    div.full.f32 $0, $8, %den0;
    div.full.f32 $1, $9, %den1;
    div.full.f32 $2, $10, %den2;
    div.full.f32 $3, $11, %den3;
    div.full.f32 $4, $12, %den4;
    div.full.f32 $5, $13, %den5;
    div.full.f32 $6, $14, %den6;
    div.full.f32 $7, $15, %den7;
    bra.uni L__adam_divide_join;
    L__adam_zero_direction:
    add.rn.f32 %den0, $16, 0f2edbe6ff;
    setp.eq.u32 %invalid, %mOr, 0x7f800000;
    vote.sync.any.pred %slow, %invalid, 0xffffffff;
    @%slow bra.uni L__adam_zero_fallback;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $8;
    mul.rn.f32 $1, %inverse, $9;
    mul.rn.f32 $2, %inverse, $10;
    mul.rn.f32 $3, %inverse, $11;
    mul.rn.f32 $4, %inverse, $12;
    mul.rn.f32 $5, %inverse, $13;
    mul.rn.f32 $6, %inverse, $14;
    mul.rn.f32 $7, %inverse, $15;
    bra.uni L__adam_divide_join;
    L__adam_zero_fallback:
    div.full.f32 $0, $8, %den0;
    div.full.f32 $1, $9, %den0;
    div.full.f32 $2, $10, %den0;
    div.full.f32 $3, $11, %den0;
    div.full.f32 $4, $12, %den0;
    div.full.f32 $5, $13, %den0;
    div.full.f32 $6, $14, %den0;
    div.full.f32 $7, $15, %den0;
    L__adam_divide_join:
    }""",
            constraints="=f,=f,=f,=f,=f,=f,=f,=f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f", args=[m_new, v_new],
            dtype=gl.float32, is_pure=True, pack=8,
        )
    elif BLOCK == 512:
        direction = gl.inline_asm_elementwise(
            """{
    .reg .b32 %mOr, %denAnd, %denOr, %shifted, %moments, %rootAnd, %rootOr;
    .reg .f32 %inverse, %root, %den<2>, %rsq, %rough, %negativeRough, %halfRsq, %residual;
    .reg .pred %invalid, %slow, %hasMoment, %warpHasMoment, %specialRoot, %warpSpecialRoot;
    and.b32 %rootAnd, $4, $5;
    or.b32 %rootOr, $4, $5;
    and.b32 %rootAnd, %rootAnd, 0x20000000;
    and.b32 %rootOr, %rootOr, 0xc0000000;
    xor.b32 %rootAnd, %rootAnd, 0x20000000;
    or.b32 %shifted, %rootAnd, %rootOr;
    setp.ne.u32 %specialRoot, %shifted, 0;
    or.b32 %mOr, $2, $3;
    and.b32 %mOr, %mOr, 0x7f800000;
    setp.eq.or.u32 %specialRoot, %mOr, 0x7f800000, %specialRoot;
    vote.sync.any.pred %warpSpecialRoot, %specialRoot, 0xffffffff;
    @%warpSpecialRoot bra.uni L__adam_root_original;
    rsqrt.approx.ftz.f32 %rsq, $4;
    mul.rn.ftz.f32 %rough, $4, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $4;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den0, %root, 0f2edbe6ff;
    rsqrt.approx.ftz.f32 %rsq, $5;
    mul.rn.ftz.f32 %rough, $5, %rsq;
    mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;
    neg.f32 %negativeRough, %rough;
    fma.rn.f32 %residual, %negativeRough, %rough, $5;
    fma.rn.f32 %root, %residual, %halfRsq, %rough;
    add.rn.f32 %den1, %root, 0f2edbe6ff;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $2;
    rcp.approx.ftz.f32 %inverse, %den1;
    mul.rn.f32 $1, %inverse, $3;
    bra.uni L__adam_divide_join;
    L__adam_root_original:
    or.b32 %moments, $4, $5;
    and.b32 %moments, %moments, 0x7fffffff;
    setp.ne.u32 %hasMoment, %moments, 0;
    vote.sync.any.pred %warpHasMoment, %hasMoment, 0xffffffff;
    @!%warpHasMoment bra.uni L__adam_zero_direction;
    sqrt.rn.f32 %root, $4;
    add.rn.f32 %den0, %root, 0f2edbe6ff;
    sqrt.rn.f32 %root, $5;
    add.rn.f32 %den1, %root, 0f2edbe6ff;
    L__adam_roots_done:
    and.b32 %denAnd, %den0, %den1;
    or.b32 %denOr, %den0, %den1;
    and.b32 %denAnd, %denAnd, 0x20000000;
    and.b32 %denOr, %denOr, 0xc0000000;
    xor.b32 %denAnd, %denAnd, 0x20000000;
    or.b32 %shifted, %denAnd, %denOr;
    setp.ne.u32 %invalid, %shifted, 0;
    setp.eq.or.u32 %invalid, %mOr, 0x7f800000, %invalid;
    vote.sync.any.pred %slow, %invalid, 0xffffffff;
    @%slow bra.uni L__adam_divide_fallback;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $2;
    rcp.approx.ftz.f32 %inverse, %den1;
    mul.rn.f32 $1, %inverse, $3;
    bra.uni L__adam_divide_join;
    L__adam_divide_fallback:
    div.full.f32 $0, $2, %den0;
    div.full.f32 $1, $3, %den1;
    bra.uni L__adam_divide_join;
    L__adam_zero_direction:
    add.rn.f32 %den0, $4, 0f2edbe6ff;
    setp.eq.u32 %invalid, %mOr, 0x7f800000;
    vote.sync.any.pred %slow, %invalid, 0xffffffff;
    @%slow bra.uni L__adam_zero_fallback;
    rcp.approx.ftz.f32 %inverse, %den0;
    mul.rn.f32 $0, %inverse, $2;
    mul.rn.f32 $1, %inverse, $3;
    bra.uni L__adam_divide_join;
    L__adam_zero_fallback:
    div.full.f32 $0, $2, %den0;
    div.full.f32 $1, $3, %den0;
    L__adam_divide_join:
    }""",
            constraints="=f,=f,f,f,f,f", args=[m_new, v_new],
            dtype=gl.float32, is_pure=True, pack=2,
        )
    else:
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
