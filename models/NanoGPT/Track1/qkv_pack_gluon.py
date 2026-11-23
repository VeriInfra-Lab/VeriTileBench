"""QKV quantization once per tile with an explicit FP8 layout conversion."""


@gluon.jit
def _quantize_qkv_tiles_gluon(
    first, second, row, transposed, scale_ptr,
    M: gl.constexpr,
    N: gl.constexpr,
    FIRST_ROWS: gl.constexpr,
    FIRST_STRIDE_B: gl.constexpr,
    SECOND_STRIDE_B: gl.constexpr,
    ROW_STRIDE_B: gl.constexpr,
    TRANSPOSED_STRIDE_B: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    QK_CROP: gl.constexpr = False,
    V_CROP: gl.constexpr = False,
):
    gl.static_assert(BLOCK_M == 64 and BLOCK_N == 128)
    gl.static_assert(FIRST_ROWS % BLOCK_M == 0 and M % BLOCK_M == 0 and N % BLOCK_N == 0)
    INPUT: gl.constexpr = gl.BlockedLayout([2, 8], [4, 8], [8, 1], [1, 0])
    TRANS: gl.constexpr = gl.BlockedLayout([8, 2], [1, 32], [8, 1], [0, 1])
    batch = gl.program_id(2)
    start_m = gl.program_id(0) * BLOCK_M
    start_n = gl.program_id(1) * BLOCK_N
    rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, INPUT))
    cols = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, INPUT))
    qk = start_m < FIRST_ROWS
    bank = gl.where(qk, first, second)
    pitch = gl.where(qk, FIRST_STRIDE_B, SECOND_STRIDE_B)
    bank_row = start_m - gl.where(qk, 0, FIRST_ROWS)
    bank_row = bank_row * gl.where(qk, 2 if QK_CROP else 1, 2 if V_CROP else 1)
    source = bank + batch * pitch + bank_row * N
    values = gl.load(source + rows[:, None] * N + start_n + cols[None, :]).to(gl.float32)
    scaled = values / gl.load(scale_ptr + batch)
    quantized = gl.maximum(gl.minimum(scaled, 448.0), -448.0).to(gl.float8e4nv)
    offs_m = start_m + rows
    offs_n = start_n + cols
    gl.store(row + batch * ROW_STRIDE_B + offs_m[:, None] * N + offs_n[None, :], quantized)
    # Each warp moves its own FP8 bytes and reuses its scratch for both column halves.
    moved = gl.convert_layout(quantized, TRANS)
    trans_m = start_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, TRANS))
    trans_n = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, TRANS))
    gl.store(transposed + batch * TRANSPOSED_STRIDE_B + trans_n[None, :] * M + trans_m[:, None], moved)
