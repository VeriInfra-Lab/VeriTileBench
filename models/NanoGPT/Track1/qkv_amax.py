"""Aligned bank scan inserted before the upstream QKV quantization launcher."""


@triton.jit
def _aligned_packed_amax(
    first, second, partial_amax,
    FIRST_ELEMENTS: tl.constexpr,
    SECOND_ELEMENTS: tl.constexpr,
    FIRST_STRIDE_B: tl.constexpr,
    SECOND_STRIDE_B: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    QK_CROP: tl.constexpr = False,
    V_CROP: tl.constexpr = False,
):
    tl.static_assert(NUM_PARTS == 32)
    tl.static_assert(FIRST_ELEMENTS % BLOCK_SIZE == 0)
    tl.static_assert(SECOND_ELEMENTS % BLOCK_SIZE == 0)
    if QK_CROP or V_CROP:
        tl.static_assert(BLOCK_SIZE == 2048)
    first_tiles: tl.constexpr = FIRST_ELEMENTS // BLOCK_SIZE
    total_tiles: tl.constexpr = (FIRST_ELEMENTS + SECOND_ELEMENTS) // BLOCK_SIZE
    batch = tl.program_id(0)
    part = tl.program_id(1)
    lane = tl.arange(0, BLOCK_SIZE)
    first_count = (first_tiles + NUM_PARTS - 1 - part) // NUM_PARTS
    total_count = (total_tiles + NUM_PARTS - 1 - part) // NUM_PARTS
    local_amax = tl.zeros((BLOCK_SIZE,), tl.float32)
    for i in tl.range(0, first_count):
        tile = part + i * NUM_PARTS
        if QK_CROP:
            tile += (tile // 24) * 24  # A cropped head spans 64 * 768 elements.
        values = tl.load(first + batch * FIRST_STRIDE_B + tile * BLOCK_SIZE + lane).to(tl.float32)
        # The original masked V operand supplies +0. Preserve that FP add.
        local_amax = tl.maximum(local_amax, tl.abs(values + 0.0))
    for i in tl.range(first_count, total_count):
        tile = part + i * NUM_PARTS - first_tiles
        if V_CROP:
            tile += (tile // 24) * 24
        values = tl.load(second + batch * SECOND_STRIDE_B + tile * BLOCK_SIZE + lane).to(tl.float32)
        # Here the original masked QK operand supplies +0 on the left.
        local_amax = tl.maximum(local_amax, tl.abs(0.0 + values))
    tl.store(partial_amax + batch * NUM_PARTS + part, tl.max(local_amax, axis=0))
