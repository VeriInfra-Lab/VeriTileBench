"""Pack the gradient exchanges in want order and clear sent value rows."""
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


WIDTH = 768
ROWS = 4
COLS = 256
ATEN_REVISION = "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"


@gluon.jit(do_not_specialize=["own_rows", "peer_rows", "lo", "source_rows"])
def _value_gradient_packets(src, ids, own, wire, own_rows, peer_rows, lo, source_rows,
                            ROWS: gl.constexpr, COLS: gl.constexpr, PACK_OWN: gl.constexpr = True):
    # One warp owns a row and reuses its routing for all three stripes.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    block = gl.program_id(0)
    lanes = gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    if PACK_OWN:
        own_blocks = ((own_rows.to(gl.uint32) + ROWS - 1) // ROWS).to(gl.int32)
        is_own = block < own_blocks
        dst = gl.where(is_own, own, wire)
        row = (block - gl.where(is_own, 0, own_blocks)) * ROWS + lanes
        live = row < gl.where(is_own, own_rows, peer_rows)
        selected = row + gl.where(is_own, lo, gl.where(row < lo, 0, own_rows))
    else:
        dst = wire
        row = block * ROWS + lanes
        live = row < peer_rows
        selected = row + gl.where(row < lo, 0, own_rows)
    source = gl.load(ids + selected, mask=live, other=0)
    bounded = (source >= 0) & (source < source_rows)
    gl.device_assert(bounded, "value gradient row out of bounds", mask=live)
    columns = gl.arange(0, COLS, layout=gl.SliceLayout(0, layout))
    source_row = src + source[:, None] * 768
    dest_row = dst + row[:, None] * 768
    for stripe in gl.static_range(3):
        cols = stripe * COLS + columns
        bits = gl.load(source_row + cols[None, :], mask=(live & bounded)[:, None], other=0)
        gl.store(dest_row + cols[None, :], bits, mask=live[:, None])


@gluon.jit(do_not_specialize=["peer_rows", "lo", "own_rows"])
def _ngram_gradient_packet(src, wire, peer_rows, lo, own_rows,
                           ROWS: gl.constexpr, COLS: gl.constexpr, CLEAR_SOURCE: gl.constexpr = False):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    lanes = gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    row = gl.program_id(0) * ROWS + lanes
    selected = row + gl.where(row < lo, 0, own_rows)
    columns = gl.arange(0, COLS, layout=gl.SliceLayout(0, layout))
    live = row[:, None] < peer_rows
    source_row = src + selected[:, None] * 768
    dest_row = wire + row[:, None] * 768
    for stripe in gl.static_range(3):
        cols = stripe * COLS + columns
        values = gl.load(source_row + cols[None, :], mask=live, other=0).to(gl.float32)
        # Use the scalar conversion in CUDA's __float2bfloat16, including its NaN handling.
        converted = gl.inline_asm_elementwise(
            "cvt.rn.bf16.f32 $0, $1;", constraints="=h,f", args=[values],
            dtype=gl.bfloat16, is_pure=True, pack=1,
        )
        gl.store(dest_row + cols[None, :], converted, mask=live)
        if CLEAR_SOURCE:
            gl.store(source_row + cols[None, :], 0.0, mask=live)


@triton.jit(do_not_specialize=["count"])
def _clear_value_want(source, ids, count, ROWS: tl.constexpr, COLS: tl.constexpr):
    slot = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    live = slot < count
    row = tl.load(ids + slot, mask=live, other=0)
    cols = tl.arange(0, COLS)
    for stripe in tl.static_range(3):
        offsets = row[:, None] * 768 + cols[None, :] + stripe * COLS
        tl.store(source + offsets, 0.0, mask=live[:, None])


def _supported(source):
    return (source.is_cuda and source.dtype == torch.float16 and source.ndim == 2
            and source.shape[1] == WIDTH and source.numel() + WIDTH * ROWS < 2**31
            and source.is_contiguous() and source.data_ptr() % 16 == 0
            and torch.cuda.get_device_capability(source.device)[0] == 9)


def _value_want_supported(source, ids):
    return (_supported(source) and ids.dtype == torch.int32 and ids.ndim == 1
            and ids.device == source.device and ids.is_contiguous() and ids.data_ptr() % 16 == 0
            and (ids.numel() + ROWS) * WIDTH < 2**31)


def pack_value_gradients(source, ids, lo, hi, *, pack_own=True):
    if not _value_want_supported(source, ids) or not 0 <= lo <= hi <= ids.numel():
        return None
    own_rows = hi - lo
    peer_rows = ids.numel() - own_rows
    own = source.new_empty((own_rows, WIDTH)) if pack_own else None
    wire = source.new_empty((peer_rows, WIDTH))
    blocks = (triton.cdiv(own_rows, ROWS) if pack_own else 0) + triton.cdiv(peer_rows, ROWS)
    if blocks:
        own_output = wire if own is None else own
        _value_gradient_packets[(blocks,)](
            source.view(torch.uint16), ids, own_output.view(torch.uint16), wire.view(torch.uint16),
            own_rows, peer_rows, lo, source.shape[0], ROWS, COLS, PACK_OWN=pack_own,
            num_warps=4, debug=True, sanitize_overflow=False,
        )
    return own, wire


def clear_value_want_(source, ids):
    """Clear this cycle's distinct, in-bounds IDs from value_embed_rows."""
    if not _value_want_supported(source, ids):
        return False
    count = ids.numel()
    if count:
        start = source.data_ptr()
        end = start + source.numel() * source.element_size()
        ids_start = ids.data_ptr()
        ids_end = ids_start + count * ids.element_size()
        if start < ids_end and ids_start < end:
            return False
        _clear_value_want[(triton.cdiv(count, ROWS),)](source, ids, count, ROWS, COLS, num_warps=4)
    return True


def pack_ngram_gradients(source, lo, hi, *, clear_source=False):
    if (torch.version.git_version != ATEN_REVISION or not _supported(source)
            or not 0 <= lo <= hi <= source.shape[0]):
        return None
    own_rows = hi - lo
    peer_rows = source.shape[0] - own_rows
    wire = source.new_empty((peer_rows, WIDTH), dtype=torch.bfloat16)
    if peer_rows:
        _ngram_gradient_packet[(triton.cdiv(peer_rows, ROWS),)](
            source, wire, peer_rows, lo, own_rows, ROWS, COLS, CLEAR_SOURCE=clear_source, num_warps=4,
        )
    return wire
