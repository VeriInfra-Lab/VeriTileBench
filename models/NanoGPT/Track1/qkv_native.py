"""Refresh the pinned attention caches directly from their head-strided banks."""


def _native_bank_layout(first, second, batch):
    pitch, head_pitch, cols = 12 * 128 * 768, 128 * 768, 768
    strides = (pitch, head_pitch, cols, 1)
    return (first.shape == (batch, 12, 128, cols) and second.shape == (batch, 6, 128, cols)
            and first.stride() == strides and second.stride() == strides
            and first.dtype == second.dtype == torch.bfloat16
            and first.is_cuda and second.device == first.device
            and torch.cuda.get_device_capability(first.device)[0] == 9)


def _native_cache_buffers(cache, device, batch, rows):
    partials, scales, row, col = cache.partial_amax, cache.scales, cache.row, cache.col
    cols = 768
    # The pinned cache creates these four buffers separately in its constructor.
    if (partials.shape != (batch, 32) or scales.shape != (batch,)
            or row.shape != (batch, rows, cols) or col.shape != (batch, cols, rows)
            or partials.dtype != torch.float32 or scales.dtype != torch.float32
            or row.dtype != torch.float8_e4m3fn or col.dtype != row.dtype
            or any(buf.device != device or not buf.is_contiguous()
                   for buf in (partials, scales, row, col))):
        return None
    return partials, scales, row, col


def refresh_native_qkv(cache, first, second, first_width, second_width):
    batch = first.shape[0]
    if ((batch, first_width, second_width) not in ((3, 64, 128), (2, 64, 64), (2, 128, 128))
            or not _native_bank_layout(first, second, batch)):
        return False
    pitch, cols = 12 * 128 * 768, 768
    first_rows, second_rows = 12 * first_width, 6 * second_width
    rows = first_rows + second_rows
    buffers = _native_cache_buffers(cache, first.device, batch, rows)
    if buffers is None:
        return False
    partials, scales, row, col = buffers
    head_crop = dict(QK_CROP=first_width == 64, V_CROP=second_width == 64)
    _aligned_packed_amax[(batch, 32)](
        first, second, partials, FIRST_ELEMENTS=first_rows * cols, SECOND_ELEMENTS=second_rows * cols,
        FIRST_STRIDE_B=pitch, SECOND_STRIDE_B=pitch, NUM_PARTS=32, BLOCK_SIZE=2048,
        num_stages=1, num_warps=8, **head_crop,
    )
    _qkv_scale_from_partials[(batch,)](partials, scales, num_warps=1)
    _quantize_qkv_tiles_gluon[(rows // 64, cols // 128, batch)](
        first, second, row, col, scales, M=rows, N=cols, FIRST_ROWS=first_rows,
        FIRST_STRIDE_B=pitch, SECOND_STRIDE_B=pitch, ROW_STRIDE_B=rows * cols,
        TRANSPOSED_STRIDE_B=rows * cols, BLOCK_M=64, BLOCK_N=128,
        num_stages=2, num_warps=8, **head_crop,
    )
    return True
