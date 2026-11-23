"""Submit the seven pinned attention slots in one grid per refresh stage."""


@triton.jit
def _qkv_scan_bank(bank, lane, part, count, cropped, local_amax, QK: tl.constexpr):
    # Three visits advance by 96 logical tiles and cross four cropped heads.
    stride = tl.where(cropped, 192, 96)
    blocks = count // 3
    for block in tl.range(0, blocks):
        for phase in tl.static_range(3):
            tile = part + 32 * phase
            initial = tile + tl.where(cropped, (tile // 24) * 24, 0)
            offset = block * stride + initial
            values = tl.load(bank + offset * 2048 + lane).to(tl.float32)
            magnitude = tl.abs(values + 0.0) if QK else tl.abs(0.0 + values)
            local_amax = tl.maximum(local_amax, magnitude)
    if not QK:
        for phase in tl.static_range(2):
            if phase < count % 3:
                tile = part + 32 * phase
                initial = tile + tl.where(cropped, (tile // 24) * 24, 0)
                offset = blocks * stride + initial
                values = tl.load(bank + offset * 2048 + lane).to(tl.float32)
                local_amax = tl.maximum(local_amax, tl.abs(0.0 + values))
    return local_amax


@triton.jit
def _grouped_qkv_scan(first, second, partial0, partial1, partial2):
    batch = tl.program_id(0)
    part = tl.program_id(1)
    narrow = batch < 3
    wide = batch >= 5
    local_batch = batch - tl.where(narrow, 0, tl.where(wide, 5, 3))
    partial = tl.where(narrow, partial0, tl.where(wide, partial2, partial1))
    lane = tl.arange(0, 2048)
    first_count = tl.where(wide, 18, 9)
    second_count = tl.where(narrow | wide, 9, 4 + (part < 16).to(tl.int32))
    first += batch * 1179648
    second += batch * 1179648
    local_amax = tl.zeros((2048,), tl.float32)
    local_amax = _qkv_scan_bank(first, lane, part, first_count, ~wide, local_amax, True)
    local_amax = _qkv_scan_bank(second, lane, part, second_count, ~(narrow | wide), local_amax, False)
    tl.store(partial + local_batch * 32 + part, tl.max(local_amax, axis=0))


@triton.jit
def _grouped_qkv_scale(partial0, partial1, partial2, scale0, scale1, scale2):
    batch = tl.program_id(0)
    narrow = batch < 3
    wide = batch >= 5
    local_batch = batch - tl.where(narrow, 0, tl.where(wide, 5, 3))
    partial = tl.where(narrow, partial0, tl.where(wide, partial2, partial1))
    scales = tl.where(narrow, scale0, tl.where(wide, scale2, scale1))
    parts = tl.arange(0, 32)
    partials = tl.load(partial + local_batch * 32 + parts)
    amax = tl.reduce(partials, 0, _qkv_scale_combine)
    floored = tl.where(amax < 1.0e-12, 1.0e-12, amax)
    tl.store(scales + local_batch, floored * (1.0 / 448.0))


@gluon.jit
def _grouped_qkv_pack(first, second, row0, row1, row2, col0, col1, col2, scale0, scale1, scale2):
    INPUT: gl.constexpr = gl.BlockedLayout([2, 16], [4, 8], [8, 1], [1, 0])
    TRANS: gl.constexpr = gl.BlockedLayout([8, 4], [1, 32], [8, 1], [0, 1])
    band = gl.program_id(2)
    start_n = gl.program_id(1) * 128
    narrow = band < 12
    wide = band >= 18
    # Slot beginnings in units of six row tiles: 0, 4, 8, 12, 15, 18, 24.
    # Count preceding beginnings and find the latest one without integer division.
    eligible = ((2 << band) - 1) & 0x1049111
    count, first_band = gl.inline_asm_elementwise(
        "{ popc.b32 $0, $2; bfind.u32 $1, $2; }", constraints="=r,=r,r",
        args=[eligible], dtype=(gl.int32, gl.int32), is_pure=True, pack=1,
    )
    batch = count - 1
    local_batch = batch - gl.where(narrow, 0, gl.where(wide, 5, 3))
    start_m = (band - first_band) * 384 + gl.program_id(0) * 64
    M = gl.where(narrow, 1536, gl.where(wide, 2304, 1152))
    first_rows = gl.where(wide, 1536, 768)
    qk = start_m < first_rows
    bank = gl.where(qk, first, second)
    crop = gl.where(qk, ~wide, ~(narrow | wide))
    bank_row = (start_m - gl.where(qk, 0, first_rows)) * gl.where(crop, 2, 1)
    source = bank + (batch * 1179648 + bank_row * 768 + start_n)
    rows = gl.arange(0, 64, layout=gl.SliceLayout(1, INPUT))
    cols = gl.arange(0, 128, layout=gl.SliceLayout(0, INPUT))
    values = gl.load(source + (rows[:, None] * 768 + cols[None, :])).to(gl.float32)
    scale = gl.where(narrow, scale0, gl.where(wide, scale2, scale1))
    scaled = values / gl.load(scale + local_batch)
    quantized = gl.maximum(gl.minimum(scaled, 448.0), -448.0).to(gl.float8e4nv)
    row = gl.where(narrow, row0, gl.where(wide, row2, row1))
    col = gl.where(narrow, col0, gl.where(wide, col2, col1))
    out = local_batch * M * 768
    gl.store(row + (out + start_m * 768 + start_n) + (rows[:, None] * 768 + cols[None, :]), quantized)
    moved = gl.convert_layout(quantized, TRANS)
    trans_m = gl.arange(0, 64, layout=gl.SliceLayout(1, TRANS))
    trans_n = gl.arange(0, 128, layout=gl.SliceLayout(0, TRANS))
    gl.store(col + (out + start_n * M + start_m) + (trans_n[None, :] * M + trans_m[:, None]), moved)


def refresh_grouped_qkv(caches, first, second):
    if not _native_bank_layout(first, second, 7):
        return False
    buffers = []
    for name, batch, rows in (("narrow", 3, 1536), ("half_v", 2, 1152), ("wide", 2, 2304)):
        if name not in caches:
            return False
        group = _native_cache_buffers(caches[name], first.device, batch, rows)
        if group is None:
            return False
        buffers.append(group)
    partials, scales, row, col = zip(*buffers)
    _grouped_qkv_scan[(7, 32)](first, second, *partials, num_stages=1, num_warps=8)
    _grouped_qkv_scale[(7,)](*partials, *scales, num_warps=1)
    _grouped_qkv_pack[(6, 6, 30)](first, second, *row, *col, *scales, num_stages=2, num_warps=8)
    return True
