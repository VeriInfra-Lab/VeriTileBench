"""Bucketed lower bounds for the rank-local n-gram wants."""

from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl


VOCAB_SIZE = 84_602_880
BUCKET_WIDTH = 1 << 15
BUCKET_COUNT = triton.cdiv(VOCAB_SIZE, BUCKET_WIDTH)
DIRECTORY_LENGTH = BUCKET_COUNT + 1
EXTRA_WORDS = DIRECTORY_LENGTH + 3
BOUNDARIES = np.arange(DIRECTORY_LENGTH, dtype=np.int32) * BUCKET_WIDTH
QUERY_COUNTS = (32768, 65536, 81920, 98304)
BLOCK = 128


def make_directory(sorted_rows):
    """Build the fixed-width directory on the cycle's preparation thread."""
    if sorted_rows.dtype != np.int32 or sorted_rows.ndim != 1 or not sorted_rows.size:
        return None
    if not 0 <= sorted_rows[0] <= sorted_rows[-1] < VOCAB_SIZE or sorted_rows.size >= 2**31:
        return None
    offsets = np.searchsorted(sorted_rows, BOUNDARIES, side="left")
    if (offsets[1:] - offsets[:-1]).max() >= 256:
        return None
    return offsets.astype(np.int32)


def directory_start(wanted_count):
    return (wanted_count + 3) // 4 * 4


@triton.jit
def bucket_probes(Want, row, lo, hi):
    for _ in tl.static_range(8):
        search = lo < hi
        mid = lo + ((hi - lo).to(tl.uint32) >> 1).to(tl.int32)
        # Once lo == hi, this keeps both endpoints unchanged without an
        # extra selection on the load predicate.
        probes = tl.load(Want + mid, search, other=row)
        higher = probes < row
        lo = tl.where(higher, mid + 1, lo)
        hi = tl.where(higher, hi, mid)
    return lo


@triton.jit
def member_probes(Want, row, lo, end):
    # The last entry is at least the member row, so clamping gives a valid
    # negative comparison whenever the probe would run past the bucket.
    lo = lo.to(tl.uint32)
    last = end.to(tl.uint32) - 1
    for bit in tl.static_range(8):
        distance = 128 >> bit
        index = tl.minimum(lo + (distance - 1), last)
        higher = tl.load(Want + index) < row
        lo = tl.where(higher, lo + distance, lo)
    return lo


@triton.jit
def bucket_lower_bound(Want, Directory, row, live, VOCAB: tl.constexpr, HASHED_ROWS: tl.constexpr = False,
                       MEMBER_ROWS: tl.constexpr = False):
    if HASHED_ROWS:
        bucket = row.to(tl.uint32) >> 15
        lo = tl.load(Directory + bucket)
        hi = tl.load(Directory + bucket + 1)
        if MEMBER_ROWS:
            return member_probes(Want, row, lo, hi)
        else:
            return bucket_probes(Want, row, lo, hi)
    else:
        in_table = (0 <= row) & (row < VOCAB) & live
        bucket = row >> 15
        lo = tl.load(Directory + bucket, in_table, other=0)
        hi = tl.load(Directory + bucket + 1, in_table, other=0)
        slot = bucket_probes(Want, row, lo, hi)
        total = tl.load(Directory + triton.cdiv(VOCAB, 32768))
        return tl.where(row >= VOCAB, total, slot)


@triton.jit
def bucket_slots(Want, Directory, Rows, Slots,
                 COUNT: tl.constexpr, VOCAB: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    live = index < COUNT
    row = tl.load(Rows + index, live, other=0)
    slot = bucket_lower_bound(Want, Directory, row, live, VOCAB)
    tl.store(Slots + index, slot, live)


@dataclass(frozen=True, slots=True)
class SlotQuery:
    want: torch.Tensor
    directory: torch.Tensor
    rows: torch.Tensor
    hashed_rows: bool = False
    member_rows: bool = False

    def materialize(self):
        result = torch.empty_like(self.rows)
        count = self.rows.numel()
        bucket_slots[(triton.cdiv(count, BLOCK),)](
            self.want, self.directory, self.rows, result, count, VOCAB_SIZE, BLOCK, num_warps=4, num_stages=1,
        )
        return result


def prepare_bucket_slots(want, directory, rows, *, hashed_rows=False, member_rows=False):
    if (directory is None or rows.ndim != 1 or rows.numel() not in QUERY_COUNTS
            or not rows.is_cuda or rows.device != want.device or rows.device != directory.device
            or rows.dtype != torch.int32 or want.dtype != torch.int32 or directory.dtype != torch.int32
            or want.ndim != 1 or directory.shape != (DIRECTORY_LENGTH,)
            or not rows.is_contiguous() or not want.is_contiguous() or not directory.is_contiguous()):
        return None
    return SlotQuery(want, directory, rows, hashed_rows, member_rows)


def try_bucket_slots(want, directory, rows):
    query = prepare_bucket_slots(want, directory, rows)
    return None if query is None else query.materialize()


def prime_bucket_slots(device):
    """Compile the four default query sizes before the timed loop."""
    if device.type != "cuda":
        return
    want = torch.zeros(1, dtype=torch.int32, device=device)
    directory = torch.ones(DIRECTORY_LENGTH, dtype=torch.int32, device=device)
    directory[0].zero_()
    rows = torch.zeros(max(QUERY_COUNTS), dtype=torch.int32, device=device)
    for count in QUERY_COUNTS:
        try_bucket_slots(want, directory, rows[:count])
