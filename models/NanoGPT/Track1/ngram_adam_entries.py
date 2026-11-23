"""Build the ordered local and peer row indices for an n-gram Adam event."""
import torch
import triton
import triton.language as tl

BLOCK = 128
ROWS = 4
CHANNELS = 768
TILE = 256


@triton.jit(do_not_specialize=["own", "count", "lo", "first"])
def _pack_entries(want, asked, entries, peer_rows, own, count, lo, first, BLOCK: tl.constexpr):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = slot < count
    local = slot < own
    peer_slot = slot - own
    source = tl.where(local, want, asked)
    index = tl.where(local, lo + slot, peer_slot)
    row = tl.load(source + index, live, other=0) - first
    tl.store(entries + slot, row, live)
    tl.store(peer_rows + peer_slot, row, live & ~local)


@triton.jit(do_not_specialize=["own", "count", "lo", "first"])
def _prepare_entries(want, asked, entries, peer_rows, entry_of_row, grad, own, count, lo, first,
                     ROWS: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr):
    slot = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    live = slot < count
    local = slot < own
    peer_slot = slot - own
    source = tl.where(local, want, asked)
    index = tl.where(local, lo + slot, peer_slot)
    row = tl.load(source + index, live, other=0) - first
    tl.store(entries + slot, row, live)
    tl.store(peer_rows + peer_slot, row, live & ~local)
    # Repeated rows choose one of their entries in the device's atomic
    # order. The scatter launches only after the claim and all zero stores.
    tl.atomic_xchg(entry_of_row + row, slot, sem="relaxed", mask=live)
    column = tl.arange(0, TILE)
    offset = slot[:, None].to(tl.int64) * D + column[None, :]
    for i in tl.static_range(D // TILE):
        tl.store(grad + offset + i * TILE, 0.0, live[:, None])


def _hopper(tensor):
    return tensor.is_cuda and torch.cuda.get_device_capability(tensor.device)[0] == 9


def pack_adam_entries(want, asked, lo, hi, first, *, entry_of_row=None):
    if (not _hopper(want) or want.device != asked.device
            or not all(t.dtype == torch.int32 and t.ndim == 1 and t.is_contiguous()
                       and t.data_ptr() % 16 == 0 for t in (want, asked))
            or not 0 <= lo <= hi <= want.numel() < 2**31
            or not 0 <= first < 2**31):
        return None
    if entry_of_row is not None and (
            entry_of_row.device != want.device or entry_of_row.dtype != torch.int32
            or entry_of_row.ndim != 1 or not entry_of_row.is_contiguous()
            or entry_of_row.data_ptr() % 16):
        return None
    if entry_of_row is not None:
        start = entry_of_row.data_ptr()
        end = start + 4 * entry_of_row.numel()
        if any(t.numel() and start < t.data_ptr() + 4 * t.numel() and t.data_ptr() < end
               for t in (want, asked)):
            return None
    own = hi - lo
    peers = asked.numel()
    count = own + peers
    if count + BLOCK >= 2**31:
        return None
    # Keep the peer index pointer 16-byte aligned for the existing scatter
    # kernel, including when the local cut is not a multiple of four.
    peer_start = (count + 3) // 4 * 4
    storage = want.new_empty(peer_start + peers)
    entries = storage[:count]
    peer_rows = storage[peer_start:]
    if entry_of_row is None:
        if count:
            _pack_entries[(triton.cdiv(count, BLOCK),)](
                want, asked, entries, peer_rows, own, count, lo, first, BLOCK, num_warps=4,
            )
        return entries[:own], peer_rows, entries

    grad = torch.empty((count, CHANNELS), device=want.device, dtype=torch.bfloat16)
    if count:
        _prepare_entries[(triton.cdiv(count, ROWS),)](
            want, asked, entries, peer_rows, entry_of_row, grad, own, count, lo, first,
            ROWS, CHANNELS, TILE, num_warps=4, num_stages=1,
        )
    return entries[:own], peer_rows, entries, grad
