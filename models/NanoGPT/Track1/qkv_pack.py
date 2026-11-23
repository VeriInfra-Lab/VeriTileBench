"""Full-tile FP8 quantization of the two QKV banks."""


@triton.jit
def _quantize_aligned_qkv_tiles(
    first, second, row, transposed, scale_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    FIRST_ROWS: tl.constexpr,
    FIRST_STRIDE_B: tl.constexpr,
    SECOND_STRIDE_B: tl.constexpr,
    ROW_STRIDE_B: tl.constexpr,
    TRANSPOSED_STRIDE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(FIRST_ROWS % BLOCK_M == 0 and M % BLOCK_M == 0 and N % BLOCK_N == 0)
    batch = tl.program_id(2)
    start_m = tl.program_id(0) * BLOCK_M
    start_n = tl.program_id(1) * BLOCK_N
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    # The tile starts and ends inside one bank. Select one base per CTA.
    in_first = start_m < FIRST_ROWS
    bank = tl.where(in_first, first, second)
    pitch = tl.where(in_first, FIRST_STRIDE_B, SECOND_STRIDE_B)
    bank_row = start_m - tl.where(in_first, 0, FIRST_ROWS)
    source = bank + batch * pitch + bank_row * N
    values = tl.load(source + rows[:, None] * N + start_n + cols[None, :]).to(tl.float32)
    scaled = values / tl.load(scale_ptr + batch)
    quantized = tl.maximum(tl.minimum(scaled, 448.0), -448.0).to(tl.float8e4nv)
    offs_m = start_m + rows
    offs_n = start_n + cols
    tl.store(row + batch * ROW_STRIDE_B + offs_m[:, None] * N + offs_n[None, :], quantized)
    tl.store(transposed + batch * TRANSPOSED_STRIDE_B + offs_n[:, None] * M + offs_m[None, :],
             tl.trans(quantized))
