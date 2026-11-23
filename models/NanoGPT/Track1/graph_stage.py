"""Stage the six inputs read by the captured training forward."""

from itertools import chain

import torch
import triton
import triton.language as tl

from track_1_short.perf.kernels.ngram_slots import BLOCK as SLOT_BLOCK
from track_1_short.perf.kernels.ngram_slots import DIRECTORY_LENGTH, VOCAB_SIZE, SlotQuery, bucket_lower_bound


LAYOUTS = {
    (16384, 64, 3),
    (32768, 96, 2),
    (49152, 128, 1),
    (40960, 128, 1),
    (16384, 64, 1),
}
DTYPES = (torch.int32, torch.int64, torch.int32, torch.int32, torch.float32, torch.float32)
STAGING_DTYPES = DTYPES * 2
TOKEN_BLOCK = 256


@triton.jit
def stage_non_slot_inputs(
    Inputs, Targets, Ends, Mtp, Prefix,
    StaticInputs, StaticTargets, StaticEnds, StaticMtp, StaticPrefix, pid,
    END_ROWS: tl.constexpr, MTP_ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    token = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(StaticInputs + token, tl.load(Inputs + token))
    tl.store(StaticTargets + token, tl.load(Targets + token))

    if pid == 0:
        end = tl.arange(0, 128)
        active = end < END_ROWS
        tl.store(StaticEnds + end, tl.load(Ends + end, mask=active, other=0), mask=active)
        row = tl.arange(0, 4)
        active = row < MTP_ROWS
        tl.store(StaticMtp + row, tl.load(Mtp + row, mask=active, other=0.0), mask=active)
        tl.store(StaticPrefix, tl.load(Prefix))


@triton.jit
def stage_step(
    Inputs, Targets, Ends, Slots, Mtp, Prefix,
    StaticInputs, StaticTargets, StaticEnds, StaticSlots, StaticMtp, StaticPrefix,
    END_ROWS: tl.constexpr, MTP_ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    stage_non_slot_inputs(Inputs, Targets, Ends, Mtp, Prefix,
                          StaticInputs, StaticTargets, StaticEnds, StaticMtp, StaticPrefix, pid,
                          END_ROWS, MTP_ROWS, BLOCK)
    slot = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(StaticSlots + slot, tl.load(Slots + slot))


@triton.jit
def stage_step_with_lookup(
    Inputs, Targets, Ends, RowIDs, Mtp, Prefix, Want, Directory,
    StaticInputs, StaticTargets, StaticEnds, StaticSlots, StaticMtp, StaticPrefix, Slots,
    INPUT_TILES: tl.constexpr, END_ROWS: tl.constexpr, MTP_ROWS: tl.constexpr,
    VOCAB: tl.constexpr, TOKEN_TILE: tl.constexpr, SLOT_TILE: tl.constexpr, HASHED_ROWS: tl.constexpr,
    MEMBER_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < INPUT_TILES:
        stage_non_slot_inputs(Inputs, Targets, Ends, Mtp, Prefix,
                              StaticInputs, StaticTargets, StaticEnds, StaticMtp, StaticPrefix, pid,
                              END_ROWS, MTP_ROWS, TOKEN_TILE)
    else:
        query = (pid - INPUT_TILES) * SLOT_TILE + tl.arange(0, SLOT_TILE)
        row = tl.load(RowIDs + query)
        slot = bucket_lower_bound(Want, Directory, row, True, VOCAB, HASHED_ROWS, MEMBER_ROWS)
        tl.store(StaticSlots + query, slot)
        tl.store(Slots + query, slot)


def stage_layout(sources, destinations, extra=()):
    if len(sources) != 6 or len(destinations) != 6:
        return None
    device = sources[0].device
    if device.type != "cuda" or device.index != torch.cuda.current_device():
        return None
    tokens, ends, mtp = (sources[index].numel() for index in (0, 2, 4))
    if (tokens, ends, mtp) not in LAYOUTS:
        return None
    sizes = (tokens, tokens, ends, 2 * tokens, mtp, 1)
    intervals = []
    buffers = zip(sources + destinations, sizes * 2, STAGING_DTYPES, strict=True)
    for tensor, size, dtype in chain(buffers, extra):
        if (tensor.device != device or tensor.dtype != dtype or tensor.shape != (size,)
                or not tensor.is_contiguous() or tensor.is_neg() or tensor.requires_grad or tensor.is_inference()):
            return None
        address = tensor.data_ptr()
        if address % 16:
            return None
        # The dtype guard fixes the element width for this interval.
        intervals.append((address, address + size * dtype.itemsize))
    intervals.sort()
    if any(left_end > right_start for (_, left_end), (right_start, _) in zip(intervals, intervals[1:])):
        return None
    return tokens, ends, mtp


def try_stage_step_inputs(sources, destinations):
    """Copy disjoint, aligned inputs on the CUDA stream used for graph replay."""
    layout = stage_layout(sources, destinations)
    if layout is None:
        return False

    tokens, ends, mtp = layout
    stage_step[(tokens // TOKEN_BLOCK,)](
        *sources, *destinations, ends, mtp, TOKEN_BLOCK, num_warps=4, num_stages=1,
    )
    torch.autograd.graph.increment_version(destinations)
    return True


def try_stage_step_query(sources, destinations, query):
    """Look up slots once and publish separate buffers for replay and gradient accumulation."""
    if sources[3] is not query.rows:
        return None
    layout = stage_layout(sources, destinations, ((query.want, query.want.numel(), torch.int32),
                                                 (query.directory, DIRECTORY_LENGTH, torch.int32)))
    if layout is None:
        return None

    tokens, ends, mtp = layout
    input_tiles = tokens // TOKEN_BLOCK
    slots = torch.empty_like(query.rows)
    stage_step_with_lookup[(input_tiles + 2 * tokens // SLOT_BLOCK,)](
        *sources, query.want, query.directory, *destinations, slots,
        input_tiles, ends, mtp, VOCAB_SIZE, TOKEN_BLOCK, SLOT_BLOCK, query.hashed_rows, query.member_rows,
        num_warps=4, num_stages=1,
    )
    torch.autograd.graph.increment_version(destinations)
    return slots


def prime_stage_step_query(device):
    """Compile the five fused graph shapes before the timed loop."""
    if device.type != "cuda":
        return
    want = torch.zeros(1, dtype=torch.int32, device=device)
    directory = torch.ones(DIRECTORY_LENGTH, dtype=torch.int32, device=device)
    directory[0].zero_()
    for tokens, ends, mtp in sorted(LAYOUTS):
        sizes = (tokens, tokens, ends, 2 * tokens, mtp, 1)
        sources = tuple(torch.zeros(size, dtype=dtype, device=device) for size, dtype in zip(sizes, DTYPES))
        destinations = tuple(torch.empty_like(tensor) for tensor in sources)
        try_stage_step_query(sources, destinations,
                             SlotQuery(want, directory, sources[3], hashed_rows=True, member_rows=True))
