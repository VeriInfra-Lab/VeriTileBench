from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def _mul(x, y):
    return gl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [x, y],
                                     dtype=gl.float32, is_pure=True, pack=1)

@gluon.jit
def mudd_backward(GRAD, MIXED, INVERSE, ENCODED, WIDTH: gl.constexpr, ACTIVATION: gl.constexpr):
    lane: gl.constexpr = gl.BlockedLayout([WIDTH], [32], [1], [0])
    channel = gl.arange(0, 1024, layout=lane)
    row = gl.program_id(0)
    row_offset = row * 768
    live = channel < 768
    g = gl.load(GRAD + row_offset + channel, live, other=0).to(gl.float32)
    x = gl.load(MIXED + row_offset + channel, live, other=0)
    inverse = gl.load(INVERSE + row)
    if WIDTH == 16:
        # The final 512 channels have rounded products before the local
        # reduction. Their masked loads supply positive zeros in padding.
        g_first, g_tail = gl.split(gl.permute(gl.reshape(g, (2, 512)), (1, 0)))
        x_first, x_tail = gl.split(gl.permute(gl.reshape(x, (2, 512)), (1, 0)))
        products = gl.join(g_first * x_first, _mul(g_tail, x_tail))
        products = gl.reshape(gl.permute(products, (1, 0)), (1024,))
        dot = gl.sum(gl.convert_layout(products, lane, assert_trivial=True), 0)
    else:
        dot = gl.sum(gl.where(live, g * x, 0.0), 0)
    inverse2 = _mul(inverse, inverse)
    inverse3 = _mul(inverse, inverse2)
    coefficient = _mul(_mul(_mul(dot, -0.5), inverse3), 0.0013020833333333333)
    correction = _mul(x + x, coefficient)
    dx = gl.fma(inverse, g, correction)
    gl.store(GRAD + row_offset + channel, dx.to(gl.bfloat16), live)
    if ACTIVATION:
        a = _mul(inverse, x)
        a = (a * 4.48).to(gl.float8e4nv)
        gl.store(ENCODED + row_offset + channel, a, live)

def launch_mudd_backward_cached(rows, gradient, residual, inverse, encoded=None):
    width = 16 if encoded is not None else 8
    mudd_backward[(rows,)](gradient, residual, inverse, encoded, width, encoded is not None,
                           num_warps=1, num_stages=1, enable_fp_fusion=True)


def launch_mudd_backward_retained(rows, gradient, residual, inverse):
    mudd_backward[(rows,)](gradient, residual, inverse, None, 16, False,
                           num_warps=1, num_stages=1, enable_fp_fusion=True)
