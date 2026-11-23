"""Install the Track 1 candidate into a dedicated checkout of the pinned baseline."""

import argparse
import ast
from pathlib import Path
import re
import shutil
import subprocess
from textwrap import indent


BASELINE = "4ea6b937337a4889b8cfe3f38a93d120048d8f71"
MODEL = Path("track_1_short/model/gpt.py")
ATTENTION = Path("track_1_short/model/attention.py")
CONFIG = Path("track_1_short/config.py")
SCHEDULE = Path("track_1_short/schedule.py")
STEP_GRAPHS = Path("track_1_short/perf/cuda_graphs/step_graphs.py")
PINNED_BATCHES = Path("track_1_short/perf/pinned_batches.py")
WIDE_YARN = "        self.yarn_wide = Yarn(head_dim, max_seq_len, attn_scale=WIDE_ATTN_SCALE, device=self.device)\n"
STATIONARY_YARN = """        self.yarn_wide = Yarn(
            head_dim, max_seq_len, attn_scale=WIDE_ATTN_SCALE, device=self.device, elide_stationary=True,
        )
"""
NGRAM = Path("track_1_short/perf/kernels/ngram_embed.py")
AMAX = Path("track_1_short/perf/kernels/fp8_attention_quant.py")
CROSS_ENTROPY = Path("track_1_short/perf/kernels/cross_entropy.py")
SAMPLED = Path("track_1_short/sampled_softmax.py")
SAMPLED_CE = Path("track_1_short/perf/kernels/sampled_cross_entropy.py")
ROPE = Path("track_1_short/perf/kernels/qkv_rope.py")
TRANSPOSE = Path("track_1_short/perf/kernels/transpose.py")
REPLICATED_ADAM = Path("track_1_short/perf/replicated_adam.py")
REPLICATED_SPAN_KERNEL = Path("track_1_short/perf/kernels/replicated_adam_spans.py")
ANVIL_EPILOGUE_KERNEL = Path("track_1_short/perf/kernels/anvil_epilogue.py")
ANVIL_TRACE_KERNEL = Path("track_1_short/perf/kernels/anvil_trace.py")
ANVIL_RAILS_KERNEL = Path("track_1_short/perf/kernels/anvil_rails.py")
ANVIL = Path("track_1_short/optim/anvil.py")
POLAR = Path("track_1_short/perf/kernels/polar_express.py")
TAIL = Path("track_1_short/tail_average.py")
LERP = Path("track_1_short/perf/kernels/lerp.py")
VALUE_PULL = Path("track_1_short/perf/value_embed_pull.py")
VALUE_EMBED = Path("track_1_short/perf/kernels/value_embed.py")
NGRAM_TABLE = Path("track_1_short/ngram_table.py")
FINEWEB_DATA = Path("track_1_short/data.py")
NGRAM_ADAM = Path("track_1_short/perf/kernels/ngram_adam.py")
ROW_SCATTER = Path("track_1_short/perf/kernels/row_scatter.py")
ROW_OWNERS = Path("track_1_short/sharded_rows.py")
ROW_PREFETCH = Path("track_1_short/perf/row_prefetch.py")
KERNEL = Path("track_1_short/perf/kernels/mlp_scale_refresh.py")
WEIGHT_KERNEL = Path("track_1_short/perf/kernels/mlp_weights_refresh.py")
HEAD_KERNEL = Path("track_1_short/perf/kernels/lm_head_refresh.py")
ACTIVATION_KERNEL = Path("track_1_short/perf/kernels/activation_fp8.py")
GATHER_KERNEL = Path("track_1_short/perf/kernels/sampled_rows_gather.py")
TRANSPOSE_KERNEL = Path("track_1_short/perf/kernels/byte_transpose.py")
BF16_TRANSPOSE_KERNEL = Path("track_1_short/perf/kernels/bf16_transpose.py")
DENSIFY_KERNEL = Path("track_1_short/perf/kernels/sampled_grad_densify.py")
TAIL_KERNEL = Path("track_1_short/perf/kernels/tail_lerp.py")
TAIL_NORM_KERNEL = Path("track_1_short/perf/kernels/tail_norm.py")
SPARSE_ROW_KERNEL = Path("track_1_short/perf/kernels/sparse_row_pack.py")
ROW_FETCH_KERNEL = Path("track_1_short/perf/kernels/sparse_row_fetch.py")
NGRAM_MOMENT_KERNEL = Path("track_1_short/perf/kernels/ngram_moment_replay.py")
NGRAM_ENTRIES_KERNEL = Path("track_1_short/perf/kernels/ngram_adam_entries.py")
VALUE_ADAM_KERNEL = Path("track_1_short/perf/kernels/value_adam_scatter.py")
VALUE_ACCUM_KERNEL = Path("track_1_short/perf/kernels/value_embed_accum.py")
GRAPH_STAGE_KERNEL = Path("track_1_short/perf/kernels/graph_stage.py")
BATCH_PACKET = Path("track_1_short/perf/batch_packet.py")
NGRAM_SLOTS_KERNEL = Path("track_1_short/perf/kernels/ngram_slots.py")
MUDD_COEFFICIENTS_KERNEL = Path("track_1_short/perf/kernels/mudd_coefficients.py")
MUDD_BACKWARD_KERNEL = Path("track_1_short/perf/kernels/mudd_backward.py")
NATIVE_ADAM = Path("track_1_short/perf/kernels/value_adam_read_scale.py")
NATIVE_ADAM_DISPATCH = Path("track_1_short/perf/kernels/read_scale_dispatch.py")
PACKED_ADAM_REFERENCE = Path(__file__).with_name("value_adam_packed_reference.py")
PRIVATE_KERNELS = (
    NATIVE_ADAM, NATIVE_ADAM_DISPATCH,
    KERNEL, WEIGHT_KERNEL, HEAD_KERNEL, GATHER_KERNEL, TRANSPOSE_KERNEL, BF16_TRANSPOSE_KERNEL, DENSIFY_KERNEL,
    TAIL_KERNEL, TAIL_NORM_KERNEL, SPARSE_ROW_KERNEL, ROW_FETCH_KERNEL, NGRAM_MOMENT_KERNEL, NGRAM_ENTRIES_KERNEL,
    VALUE_ADAM_KERNEL, VALUE_ACCUM_KERNEL, REPLICATED_SPAN_KERNEL, ANVIL_EPILOGUE_KERNEL, ANVIL_TRACE_KERNEL,
    ANVIL_RAILS_KERNEL,
    GRAPH_STAGE_KERNEL, BATCH_PACKET, NGRAM_SLOTS_KERNEL, ACTIVATION_KERNEL, MUDD_COEFFICIENTS_KERNEL,
    MUDD_BACKWARD_KERNEL,
)

DRIVER = Path("train_gpt.py")
DRIVER_SUBSTITUTIONS = (
    ("    model: nn.Module = torch.compile(model, dynamic=False, fullgraph=True)\n",
     "    from track_1_short.perf.kernels.mudd_coefficients import compile_training_model\n"
     "    model: nn.Module = compile_training_model(model, env.device)\n"),
    ("    ngram_slots = row_prefetch.forward_slots(step, batch.ngram_ids)\n",
     "    ngram_slots = row_prefetch.forward_slots(step, batch.ngram_ids, defer_slots=True, hashed_rows=True)\n"),
    ("    step_graphs.forward(step, batch, ngram_slots, training_manager.get_forward_args(sampled_loss))\n",
     "    ngram_slots = step_graphs.forward(step, batch, ngram_slots, training_manager.get_forward_args(sampled_loss))\n"),
    ("""    ngram_table.accumulate_grad(ngram_slots, step_graphs.hold_ngram_grad(
        ngram_grad, pending=len(ngram_table.pending), event_this_step=row_prefetch.is_update(step)))
""",
     """    if row_prefetch.update_positions and row_prefetch.position[step] <= row_prefetch.update_positions[-1]:
        ngram_table.accumulate_grad(ngram_slots, ngram_grad, row_prefetch.is_update(step))
"""),
    ("""    step_graphs = StepGraphs(model, training_manager.optimizer, step_keys, ngram_table, value_embed_pull,
                             max_step_tokens, env.device)
""",
     """    step_graphs = StepGraphs(model, training_manager.optimizer, step_keys, ngram_table, value_embed_pull,
                             env.device)
"""),
)

STEP_GRAPH_SUBSTITUTIONS = (
    ("from track_1_short.ngram_table import MAX_CYCLE_STEPS, NGRAM_DIM, NgramTable\n",
     "from track_1_short.ngram_table import NgramTable\n"),
    ("""(copy on the first step, add on the second). The n-gram sink gradient is held by the table until its
update event, up to MAX_CYCLE_STEPS steps, so each non-event step's copy goes to a private ring slot.
""", """(copy on the first step, add on the second). The n-gram table accumulates non-event sink gradients
on the compute stream before the next replay. Only the event step holds the graph's output buffer.
"""),
    ("training step, warmup and timed: forward(), then backward(), then hold_ngram_grad().\"\"\"\n",
     "training step, warmup and timed: forward() and then backward().\"\"\"\n"),
    ("ngram_table: NgramTable, value_embeds: ValueEmbedPull, max_step_tokens: int,\n",
     "ngram_table: NgramTable, value_embeds: ValueEmbedPull,\n"),
    ("""        self.adam_accumulators: dict[nn.Parameter, Tensor] = {}
        # n-gram sink gradients held until the update event: slot i holds pending entry i of the cycle.
        # Flat, sized for the largest step; each key views its own [2T, NGRAM_DIM] prefix.
        self.ngram_grad_ring = [torch.empty(2 * max_step_tokens * NGRAM_DIM, dtype=torch.bfloat16, device=device)
                                for _ in range(MAX_CYCLE_STEPS)]
""", "        self.adam_accumulators: dict[nn.Parameter, Tensor] = {}\n"),
    ("""    def hold_ngram_grad(self, ngram_grad: Tensor, pending: int, event_this_step: bool) -> Tensor:
        \"\"\"The n-gram gradient the table may keep: the graph's own buffer only if this step's update
        event consumes it before the next replay, else a copy in ring slot `pending`.\"\"\"
        if event_this_step:
            return ngram_grad
        assert pending < len(self.ngram_grad_ring), f"{pending} held n-gram gradients > ring"
        slot = self.ngram_grad_ring[pending][:ngram_grad.numel()].view_as(ngram_grad)
        slot.copy_(ngram_grad)
        return slot

    # ---- capture and checks (warmup only) ----
""", "    # ---- capture and checks (warmup only) ----\n"),
    ("from track_1_short.perf.cuda_graphs.capture_support import WARM_ITERATIONS\n",
     "from track_1_short.perf.cuda_graphs.capture_support import WARM_ITERATIONS\n"
     "from track_1_short.perf.kernels.graph_stage import prime_stage_step_query, try_stage_step_inputs, try_stage_step_query\n"
     "from track_1_short.perf.kernels.ngram_slots import SlotQuery\n"),
    ("        self.current: CapturedStep | None = None\n",
     "        self.current: CapturedStep | None = None\n"
     "        prime_stage_step_query(device)\n"),
    ("    def forward(self, step: int, batch: Batch, ngram_slots: Tensor, cfg: ForwardScheduleConfig):\n",
     "    def forward(self, step: int, batch: Batch, ngram_slots: Tensor | SlotQuery, cfg: ForwardScheduleConfig) -> Tensor:\n"),
    ("            cs = self._capture(key, batch, ngram_slots, cfg)\n",
     "            if isinstance(ngram_slots, SlotQuery):\n"
     "                ngram_slots = ngram_slots.materialize()\n"
     "            cs = self._capture(key, batch, ngram_slots, cfg)\n"),
    ("""        for dst, src in zip(cs.static.buffers(), step_inputs(batch, ngram_slots, cfg)):
            dst.copy_(src)
""",
     """        destinations = cs.static.buffers()
        staged = False
        if isinstance(ngram_slots, SlotQuery):
            query = ngram_slots
            slots = try_stage_step_query(step_inputs(batch, query.rows, cfg), destinations, query)
            staged = slots is not None
            ngram_slots = slots if staged else query.materialize()
        if not staged:
            sources = step_inputs(batch, ngram_slots, cfg)
            if not try_stage_step_inputs(sources, destinations):
                for dst, src in zip(destinations, sources):
                    dst.copy_(src)
"""),
    ("        self.current = cs\n", "        self.current = cs\n        return ngram_slots\n"),
)

PINNED_BATCH_SUBSTITUTIONS = (
    ("from torch import Tensor\n", "from torch import Tensor\n\nfrom track_1_short.perf.batch_packet import BatchPacket\n"),
    ("""    inputs: Tensor        # int32 [max_tokens]
    targets: Tensor       # int64 [max_tokens]
    cum_seqlens: Tensor   # int32 [max_cu_seqlens_rows]
    ngram_ids: Tensor     # int32 [2 * max_tokens]
""",
     "    packet: BatchPacket\n"),
    ("""        pinned = lambda n, dtype: torch.empty(n, dtype=dtype, pin_memory=True)
        self.device = device
        self.slots = [PinnedSlot(pinned(max_tokens, torch.int32), pinned(max_tokens, torch.int64),
                                 pinned(max_cu_seqlens_rows, torch.int32), pinned(2 * max_tokens, torch.int32),
                                 torch.cuda.Event())
""",
     """        self.device = device
        self.slots = [PinnedSlot(BatchPacket(max_tokens, max_cu_seqlens_rows), torch.cuda.Event())
"""),
    ("    def upload(self, inputs: Tensor, targets: Tensor, cum_seqlens: Tensor, ngram_ids: Tensor) -> tuple[Tensor, ...]:\n",
     """    def upload(self, inputs: Tensor, targets: Tensor, cum_seqlens: Tensor, ngram_ids: Tensor,
               *, coalesce: bool = False) -> tuple[Tensor, ...]:
"""),
    ("""        out = []
        for host, buf in ((inputs, slot.inputs), (targets, slot.targets), (cum_seqlens, slot.cum_seqlens),
                          (ngram_ids, slot.ngram_ids)):
            n = host.numel()
            assert n <= buf.numel() and host.dtype == buf.dtype, f"batch tensor {host.dtype}[{n}] overflows its pinned slot"
            buf[:n].copy_(host)
            out.append(buf[:n].to(self.device, non_blocking=True))
        slot.uploaded.record()
        return tuple(out)
""",
     """        out = slot.packet.upload((inputs, targets, cum_seqlens, ngram_ids), self.device, coalesce)
        slot.uploaded.record()
        return out
"""),
)

ANVIL_REFERENCE_NORMALIZATION = """        tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
        d = tr.sqrt() * FROBENIUS_MARGIN + FROBENIUS_EPS
        X = (X.float() / d).bfloat16()
        A = (A.float() / d.square()).bfloat16()
"""

ANVIL_DIVISOR_NORMALIZATION = """        if compact_trace:
            divisors = gram_divisors(A)
            normalize_gram_inputs(X, A, divisors)
        else:
            tr = A.diagonal(dim1=-2, dim2=-1).float().sum(-1)[..., None, None]
            d = tr.sqrt() * FROBENIUS_MARGIN + FROBENIUS_EPS
            d_square = d.square()
            X = (X.float() / d).bfloat16()
            A = (A.float() / d_square).bfloat16()
"""

ANVIL_TRACE_SUBSTITUTIONS = (
    ("from track_1_short.perf.kernels.polar_express import XTX, XXT, ba_plus_cAA\n",
     "from track_1_short.perf.kernels.polar_express import XTX, XXT, ba_plus_cAA\n"
     "from track_1_short.perf.kernels.anvil_trace import can_fuse_gram_trace, gram_divisors, normalize_gram_inputs\n"),
    ("split_baddbmm: bool, fast_beta_t: torch.Tensor, fast_weight_t: torch.Tensor):",
     "split_baddbmm: bool, fast_beta_t: torch.Tensor, fast_weight_t: torch.Tensor,\n"
     "                  compact_trace: bool = False):"),
    *(
        (gram + ANVIL_REFERENCE_NORMALIZATION, gram + ANVIL_DIVISOR_NORMALIZATION)
        for gram in ("        XTX(X, out=A)  # A = X.T @ X\n",
                     "        XXT(X, out=A)  # A = X @ X.mT\n")
    ),
    ("    red_dim: int\n",
     "    red_dim: int\n"
     "    compact_trace: bool = False\n"),
    ("                red_dim=-1 if chunk_shape[-2] >= chunk_shape[-1] else -2,\n",
     "                red_dim=-1 if chunk_shape[-2] >= chunk_shape[-1] else -2,\n"
     "                compact_trace=can_fuse_gram_trace(chunk_shape, param.dtype, device),\n"),
    ("        fast_beta_t=bank.fast_beta, fast_weight_t=bank.fast_weight,\n",
     "        fast_beta_t=bank.fast_beta, fast_weight_t=bank.fast_weight,\n"
     "        compact_trace=bank.compact_trace,\n"),
)

ANVIL_REFERENCE_PREPARATION = """    grad_chunk = grad_chunk.float()
    momentum = momentum_t.to(grad_chunk.dtype)
    velocity[0].lerp_(grad_chunk, 1 - fast_beta_t.to(grad_chunk.dtype))
    velocity[1].lerp_(grad_chunk, 1 - RAIL_SLOW_BETA)
    w = fast_weight_t.to(grad_chunk.dtype)
    blend = w * velocity[0] + (1 - w) * velocity[1]
    g = grad_chunk.lerp_(blend, momentum)

    X = g.bfloat16().contiguous()
"""

ANVIL_RAILS_SUBSTITUTIONS = (
    ("from track_1_short.perf.kernels.anvil_trace import can_fuse_gram_trace, gram_divisors, normalize_gram_inputs\n",
     "from track_1_short.perf.kernels.anvil_trace import can_fuse_gram_trace, gram_divisors, normalize_gram_inputs\n"
     "from track_1_short.perf.kernels.anvil_rails import prepare_rails\n"),
    (ANVIL_REFERENCE_PREPARATION + "    is_tall = g.size(-2) > g.size(-1)\n",
     "    if compact_trace:\n"
     "        X = torch.empty_like(grad_chunk)\n"
     "        prepare_rails(grad_chunk, velocity, momentum_t, fast_beta_t, fast_weight_t, X)\n"
     "    else:\n" + indent(ANVIL_REFERENCE_PREPARATION, "    ") +
     "    is_tall = X.size(-2) > X.size(-1)\n"),
)

ANVIL_EPILOGUE_SUBSTITUTIONS = (
    ("from track_1_short.perf.replicated_adam import ReplicatedAdam\n",
     "from track_1_short.perf.replicated_adam import ReplicatedAdam\n"
     "from track_1_short.perf.kernels.anvil_epilogue import eligible as epilogue_eligible, update_epilogue\n"),
    ("""    # 2. Equalize per-lane update energy
    v_chunk = AnvilAndAdam._rail_equalizer(v_chunk, bank.lane_energy, bank.beta2, bank.red_dim)
    # 3. Update the parameter in place, with weight decay gated on the slow rail's sign. eff_lr is per
    #    matrix: MLP c_proj gets 2x, frozen matrices 0.
    AnvilAndAdam._sign_aligned_decay_update(
        bank.p_slice.view(torch.uint16), bank.mantissa, v_chunk, bank.eff_wd, bank.eff_lr, bank.velocity[1],
    )
""",
     """    p = bank.p_slice.view(torch.uint16)
    gate = bank.velocity[1]
    if epilogue_eligible(v_chunk, bank.lane_energy, p, bank.mantissa, bank.eff_wd, bank.eff_lr,
                         gate, bank.beta2, bank.red_dim):
        update_epilogue(v_chunk, bank.lane_energy, bank.beta2, bank.red_dim, p, bank.mantissa,
                        bank.eff_wd, bank.eff_lr, gate)
    else:
        v_chunk = AnvilAndAdam._rail_equalizer(v_chunk, bank.lane_energy, bank.beta2, bank.red_dim)
        AnvilAndAdam._sign_aligned_decay_update(p, bank.mantissa, v_chunk, bank.eff_wd, bank.eff_lr, gate)
"""),
)

POLAR_COMPACT_GRID = """    if COMPACT_SIDE:
        # Three bits per coordinate, in the original column-major tile order.
        tile_count: tl.constexpr = COMPACT_SIDE * (COMPACT_SIDE + 1) // 2
        row_bits: tl.constexpr = 0x48 if COMPACT_SIDE == 2 else 0x5b2c76356346c688
        col_bits: tl.constexpr = 0x40 if COMPACT_SIDE == 2 else 0x591b692449240000
        shift = (pid % tile_count) * 3
        row = ((tl.full((), row_bits, tl.uint64) >> shift) & 7).to(tl.int32)
        col = ((tl.full((), col_bits, tl.uint64) >> shift) & 7).to(tl.int32)
        return pid // tile_count, row * BLOCK_SIZE_M, col * BLOCK_SIZE_N

"""

POLAR_SUBSTITUTIONS = (
    ("    GROUP_SIZE_M: tl.constexpr,\n):\n    # Split output",
     "    GROUP_SIZE_M: tl.constexpr,\n    COMPACT_SIDE: tl.constexpr = 0,\n):\n"
     + POLAR_COMPACT_GRID + "    # Split output", 1),
    ("    LOWER_UPPER: tl.constexpr,\n", "    LOWER_UPPER: tl.constexpr,\n    COMPACT_SIDE: tl.constexpr = 0,\n", 3),
    ("BLOCK_SIZE_N, GROUP_SIZE_M\n    )", "BLOCK_SIZE_N, GROUP_SIZE_M, COMPACT_SIDE\n    )", 3),
    ("    if skip_block_below_diag or skip_block_above_diag:\n",
     "    if not COMPACT_SIDE and (skip_block_below_diag or skip_block_above_diag):\n", 3),
    ("        LOWER_UPPER=1,\n", "        LOWER_UPPER=1,\n        COMPACT_SIDE=side,\n", 3),
    *tuple((
        f"    grid = (batch_size * triton.cdiv({dim}, BLOCK_SIZE_M) * triton.cdiv({dim}, BLOCK_SIZE_N),)\n",
        f"    side = {dim} // 128 if BLOCK_SIZE_M == BLOCK_SIZE_N == 128 and {dim} in (256, 768) else 0\n"
        f"    tiles = side * (side + 1) // 2 if side else triton.cdiv({dim}, BLOCK_SIZE_M) * triton.cdiv({dim}, BLOCK_SIZE_N)\n"
        "    grid = (batch_size * tiles,)\n",
        count,
    ) for dim, count in (("M", 2), ("K", 1))),
)

ANVIL_FULL_TILE_SUBSTITUTIONS = (
    *(
        (f"\n{indentation}{gram}(X, out=A)  # A = {expression}\n",
         f"\n{indentation}{gram}(X, out=A, full_tiles=compact_trace)  # A = {expression}\n")
        for gram, expression in (("XXT", "X @ X.mT"), ("XTX", "X.T @ X"))
        for indentation in ("        ", "                ")
    ),
    *(
        (f"ba_plus_cAA(A, alpha=c, beta=b, out=B)  # {comment}",
         f"ba_plus_cAA(A, alpha=c, beta=b, out=B, full_tiles=compact_trace)  # {comment}")
        for comment in ("B = b*A + c*(A@A)", "B = b * A + c * A @ A")
    ),
)

ANVIL_COOPERATIVE_SUBSTITUTIONS = (
    ("from track_1_short.perf.kernels.polar_express import XTX, XXT, ba_plus_cAA\n",
     "from track_1_short.perf.kernels.polar_express import XTX, XXT, ba_plus_cAA, "
     "anvil_cooperative_round, anvil_initial_gram\n", 1),
    ("can_fuse_gram_trace, gram_divisors, normalize_gram_inputs\n",
     "can_fuse_gram_trace, gram_divisors, normalize_gram_inputs, supports_cooperative_anvil\n", 1),
    ("import prepare_rails\n",
     "import prepare_rails, prepare_rails_and_seed\n", 1),
    ("                  compact_trace: bool = False):",
     "                  compact_trace: bool = False, cooperative_rounds: bool = False):", 1),
    ("    if compact_trace:\n"
     "        X = torch.empty_like(grad_chunk)\n"
     "        prepare_rails(grad_chunk, velocity, momentum_t, fast_beta_t, fast_weight_t, X)\n",
     "    counter = None\n"
     "    if compact_trace:\n"
     "        X = torch.empty_like(grad_chunk)\n"
     "        if cooperative_rounds:\n"
     "            counter = prepare_rails_and_seed(grad_chunk, velocity, momentum_t, fast_beta_t, fast_weight_t, X)\n"
     "        else:\n"
     "            prepare_rails(grad_chunk, velocity, momentum_t, fast_beta_t, fast_weight_t, X)\n", 1),
    *(
        (f"\n        {gram}(X, out=A, full_tiles=compact_trace)  # A = {expression}\n"
         "        if compact_trace:\n"
         "            divisors = gram_divisors(A)\n",
         "\n        if counter is not None:\n"
         "            divisors = anvil_initial_gram(X, A, counter)\n"
         "        else:\n"
         f"            {gram}(X, out=A, full_tiles=compact_trace)  # A = {expression}\n"
         "            if compact_trace:\n"
         "                divisors = gram_divisors(A)\n"
         "        if compact_trace:\n", 1)
        for gram, expression in (("XTX", "X.T @ X"), ("XXT", "X @ X.mT"))
    ),
    *(
        (f"            if k > 0:\n"
         f"                {gram}(X, out=A, full_tiles=compact_trace)  # A = {expression}\n"
         f"            ba_plus_cAA(A, alpha=c, beta=b, out=B, full_tiles=compact_trace)  # {polynomial}\n",
         f"            if k > 0 and counter is not None:\n"
         f"                anvil_cooperative_round(X, A, B, counter, k, c, b)\n"
         f"            else:\n"
         f"                if k > 0:\n"
         f"                    {gram}(X, out=A, full_tiles=compact_trace)  # A = {expression}\n"
         f"                ba_plus_cAA(A, alpha=c, beta=b, out=B, full_tiles=compact_trace)  # {polynomial}\n", 1)
        for gram, expression, polynomial in (("XTX", "X.T @ X", "B = b*A + c*(A@A)"),
                                              ("XXT", "X @ X.mT", "B = b * A + c * A @ A"))
    ),
    ("    compact_trace: bool = False\n",
     "    compact_trace: bool = False\n"
     "    cooperative_rounds: bool = False\n", 1),
    ("                compact_trace=can_fuse_gram_trace(chunk_shape, param.dtype, device),\n",
     "                compact_trace=can_fuse_gram_trace(chunk_shape, param.dtype, device),\n"
     "                cooperative_rounds=supports_cooperative_anvil(device),\n", 1),
    ("        compact_trace=bank.compact_trace,\n",
     "        compact_trace=bank.compact_trace,\n"
     "        cooperative_rounds=bank.cooperative_rounds,\n", 1),
)

POLAR_GLUON_IMPORTS = """from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared, mbarrier, tma, warpgroup_mma, warpgroup_mma_wait,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonDescriptor
from track_1_short.perf.kernels.anvil_trace import trace_divisors
"""

POLAR_GLUON = """def _dense_bf16_tiles(A, out, shapes, width):
    return (tuple(A.shape) in shapes and tuple(out.shape) == (A.size(0), width, width)
            and A.dtype == out.dtype == torch.bfloat16 and A.device == out.device
            and A.is_contiguous() and out.is_contiguous())


@gluon.jit
def prefetch_left(A, barrier, tile, BarrierBytes: gl.constexpr, batch_row, row, k, pred, Tall: gl.constexpr):
    mbarrier.expect(barrier, BarrierBytes, pred)
    if Tall:
        coords = [batch_row + k, row]
    else:
        coords = [batch_row + row, k]
    tma.async_copy_global_to_shared(A, coords, barrier, tile, pred)


@gluon.jit
def prefetch_right(A, barrier, tile, batch_row, col, k, pred, Tall: gl.constexpr):
    if Tall:
        coords = [batch_row + k, col]
    else:
        coords = [batch_row + col, k]
    tma.async_copy_global_to_shared(A, coords, barrier, tile, pred)


@gluon.jit
def polar_reduction(A, tiles, barriers, acc, batch_row, row, col,
              End: gl.constexpr, Tall: gl.constexpr, Diagonal: gl.constexpr):
    transfer_bytes: gl.constexpr = 32768 if Diagonal else 65536
    prefetch_left(A, barriers.index(0), tiles.index(0), transfer_bytes, batch_row, row, 0, True, Tall)
    if not Diagonal:
        prefetch_right(A, barriers.index(0), tiles.index(2), batch_row, col, 0, True, Tall)
    for chunk in range(End // 128):
        buf = chunk % 2
        phase = (chunk // 2) % 2
        base = tiles.index(buf)
        if Diagonal:
            other = base
        else:
            other = tiles.index(2)
        mbarrier.wait(barriers.index(buf), phase=phase, deps=(base, other))
        if Tall:
            left = base.permute((1, 0))
            right = other
        else:
            left = base
            right = other.permute((1, 0))
        acc = warpgroup_mma(left, right, acc, is_async=True)
        if Diagonal:
            acc = warpgroup_mma_wait(1, deps=(acc,))
        nxt = (chunk + 1) % 2
        more = chunk + 1 < End // 128
        k = (chunk + 1) * 128
        prefetch_left(A, barriers.index(nxt), tiles.index(nxt), transfer_bytes,
                      batch_row, row, k, more, Tall)
        if not Diagonal:
            # The next left copy uses the other box. The fixed right box
            # still belongs to the current group until it retires.
            acc = warpgroup_mma_wait(0, deps=(acc,))
            prefetch_right(A, barriers.index(nxt), tiles.index(2), batch_row, col, k, more, Tall)
    if Diagonal:
        acc = warpgroup_mma_wait(0, deps=(acc,))
    return acc


@gluon.jit
def triangle_coordinates(pid, side: gl.constexpr, tile_count: gl.constexpr):
    row_bits: gl.constexpr = 0x48 if side == 2 else 0x5b2c76356346c688
    col_bits: gl.constexpr = 0x40 if side == 2 else 0x591b692449240000
    shift = (pid % tile_count) * 3
    row = ((gl.full((), row_bits, gl.uint64, gl.BlockedLayout([1], [32], [8], [0]))
            >> shift) & 7).to(gl.int32) * 128
    col = ((gl.full((), col_bits, gl.uint64, gl.BlockedLayout([1], [32], [8], [0]))
            >> shift) & 7).to(gl.int32) * 128
    return row, col


@gluon.jit
def polar(A, C, A_ptr, alpha, beta, K: gl.constexpr, M: gl.constexpr,
          Tall: gl.constexpr, Polynomial: gl.constexpr, L2Bias: gl.constexpr, ReturnOffdiag: gl.constexpr = False):
    end: gl.constexpr = M if Tall else K
    width: gl.constexpr = K if Tall else M
    side: gl.constexpr = width // 128
    tile_count: gl.constexpr = side * (side + 1) // 2
    pid = gl.program_id(0)
    row, col = triangle_coordinates(pid, side, tile_count)
    batch = (pid // tile_count).to(gl.int32)
    offdiag = row != col
    layout: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[8, 1],
                                                    instr_shape=[16, 128, 16])
    acc = gl.full((128, 128), 0, gl.float32, layout)
    tiles = gl.allocate_shared_memory(gl.bfloat16, [3] + A.block_shape, A.layout)
    barriers = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    mbarrier.init(barriers.index(0), count=1)
    mbarrier.init(barriers.index(1), count=1)
    if offdiag:
        acc = polar_reduction(A, tiles, barriers, acc, batch * M, row, col, end, Tall, False)
    else:
        acc = polar_reduction(A, tiles, barriers, acc, batch * M, row, col, end, Tall, True)
    mbarrier.invalidate(barriers.index(0))
    mbarrier.invalidate(barriers.index(1))
    if Polynomial:
        row_major: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [8, 1], [1, 0])
        i = gl.arange(0, 128, layout=gl.SliceLayout(1, row_major))
        j = gl.arange(0, 128, layout=gl.SliceLayout(0, row_major))
        base = batch * width * width
        dst = (row + i[:, None]) * width + (col + j[None, :])
        bias_words = gl.load(A_ptr + base + dst, cache_modifier=\".cg\" if L2Bias else \"\")
        bias = gl.convert_layout(bias_words, layout).to(gl.float32)
        words = (acc * alpha + bias * beta).to(gl.bfloat16)
    else:
        words = acc.to(gl.bfloat16)
    direct = gl.allocate_shared_memory(gl.bfloat16, [128, 128], C.layout)
    if offdiag:
        direct.store(words)
    mirror = gl.allocate_shared_memory(gl.bfloat16, [128, 128], C.layout, words.permute((1, 0)))
    fence_async_shared()
    if offdiag:
        tma.async_copy_shared_to_global(C, [batch * width + row, col], direct)
    tma.async_copy_shared_to_global(C, [batch * width + col, row], mirror)
    # The Gluon lowering commits each TMA store. Wait once for both
    # the shared reads and the completed global writes.
    gl.inline_asm_elementwise(\"cp.async.bulk.wait_group 0; mov.u32 $0, 0;\", \"=r,~{memory}\", [],
                              dtype=gl.int32, is_pure=False, pack=1)
    if ReturnOffdiag:
        return offdiag


@gluon.jit
def incident_tiles(row, SIDE: gl.constexpr):
    if SIDE == 2:
        return 3 << row
    else:
        start = row * (13 - row) // 2
        prefix = gl.inline_asm_elementwise(\"bmsk.clamp.b32 $0, 0, $1;\", \"=r,r\", [start - row],
                                           dtype=gl.uint32, is_pure=True, pack=1)
        tail = gl.inline_asm_elementwise(\"bmsk.clamp.b32 $0, $1, $2;\", \"=r,r,r\", [start, 6 - row],
                                         dtype=gl.uint32, is_pure=True, pack=1)
        return ((prefix & 0xd221) << row) | tail



@gluon.jit
def publish_gram(Counter, bit, needed):
    gl.inline_asm_elementwise(
        \"\"\"{
          .reg .u32 tid, bits, covered;
          .reg .pred observer, ready;
          bar.sync 0;
          mov.u32 tid, %tid.x;
          setp.eq.u32 observer, tid, 0;
          @!observer bra gram_ready;
          fence.proxy.async.global;
          atom.acq_rel.gpu.global.or.b32 bits, [$1], $2;
          or.b32 bits, bits, $2;
          and.b32 covered, bits, $3;
          setp.eq.u32 ready, covered, $3;
          @ready bra gram_ready;
        observe_gram:
          ld.acquire.gpu.global.u32 bits, [$1];
          and.b32 covered, bits, $3;
          setp.eq.u32 ready, covered, $3;
          @!ready bra observe_gram;
        gram_ready:
          bar.sync 0;
          fence.proxy.async.global;
          mov.u32 $0, 0;
        }\"\"\",
        \"=r,l,r,r,~{memory}\", [Counter, bit, needed], dtype=gl.int32, is_pure=False, pack=1,
    )


@gluon.jit(do_not_specialize=[\"round_offset\"])
def cooperative_round(X, Gram, Output, Gram_ptr, Counter, round_offset, alpha, beta,
                      K: gl.constexpr, M: gl.constexpr, Tall: gl.constexpr):
    width: gl.constexpr = K if Tall else M
    polar(X, Gram, None, None, None, K, M, Tall, False, False)
    side: gl.constexpr = width // 128
    tiles: gl.constexpr = side * (side + 1) // 2
    pid = gl.program_id(0)
    batch = pid // tiles
    row, col = triangle_coordinates(pid, side, tiles)
    bit = 1 << (pid % tiles)
    needed = incident_tiles(row // 128, side) | incident_tiles(col // 128, side)
    publish_gram(Counter + round_offset + batch, bit, needed)
    polar(Gram, Output, Gram_ptr, alpha, beta, width, width, False, True, True)


@gluon.jit
def cooperative_initial(X, Gram, Gram_ptr, Divisors, Counter, K: gl.constexpr,
                        M: gl.constexpr, Tall: gl.constexpr):
    width: gl.constexpr = K if Tall else M
    side: gl.constexpr = width // 128
    tiles: gl.constexpr = side * (side + 1) // 2
    offdiag = polar(X, Gram, None, None, None, K, M, Tall, False, False, True)
    if not offdiag:
        batch = gl.program_id(0) // tiles
        gl.inline_asm_elementwise(\"bar.sync 0; fence.proxy.async.global; mov.u32 $0, 0;\",
                                  \"=r,~{memory}\", [], dtype=gl.int32, is_pure=False, pack=1)
        ordinal = gl.atomic_add(Counter + batch, 1, sem=\"acq_rel\", scope=\"gpu\")
        if ordinal == side - 1:
            gl.inline_asm_elementwise(\"fence.proxy.async.global; mov.u32 $0, 0;\", \"=r,~{memory}\", [],
                                      dtype=gl.int32, is_pure=False, pack=1)
            trace_divisors(Gram_ptr, Divisors, batch, width, 2 if width == 256 else 8, 8, True)



def _matrix_counters(counter, x, batches):
    return (counter.device == x.device and counter.dtype == torch.int32
            and tuple(counter.shape) == (6 * batches,) and counter.is_contiguous()
            and counter.data_ptr() % 16 == 0)


@torch.library.custom_op(\"veritile_track1::anvil_initial_gram\", mutates_args={\"a\", \"counter\"})
def anvil_initial_gram(x: torch.Tensor, a: torch.Tensor, counter: torch.Tensor) -> torch.Tensor:
    batches, m, k = x.shape
    tall = m > k
    width = k if tall else m
    shapes = ((6, 256, 768), (2, 768, 768), (3, 2816, 768))
    if not _dense_bf16_tiles(x, a, shapes, width) or not _matrix_counters(counter, x, batches):
        raise ValueError(\"Unsupported ANVIL initial Gram\")
    regions = sorted((tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size())
                     for tensor in (x, a, counter))
    if any(end > start for (_, end), (start, _) in zip(regions, regions[1:])):
        raise ValueError(\"ANVIL initial Gram buffers overlap\")
    divisors = torch.empty((batches, 2, 1), dtype=torch.float32, device=x.device)
    side = width // 128
    tiles = side * (side + 1) // 2
    layout = gl.NVMMASharedLayout(128, 16)
    src = GluonDescriptor.from_tensor(x.flatten(0, 1), [128, 128], layout)
    gram = GluonDescriptor.from_tensor(a.flatten(0, 1), [128, 128], layout)
    cooperative_initial[(batches * tiles,)](src, gram, a, divisors, counter, k, m, tall,
                                            num_warps=8, num_stages=1)
    return divisors


@anvil_initial_gram.register_fake
def _fake_initial_gram(x, a, counter):
    return a.new_empty((a.shape[0], 2, 1), dtype=torch.float32)


@torch.library.custom_op(\"veritile_track1::anvil_cooperative_round\", mutates_args={\"a\", \"b\", \"counter\"})
def anvil_cooperative_round(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                            counter: torch.Tensor, round_number: int,
                            alpha: float, beta: float) -> None:
    batches, m, k = x.shape
    tall = m > k
    width = k if tall else m
    shapes = ((6, 256, 768), (2, 768, 768), (3, 2816, 768))
    gram_shape = (batches, width, width)
    if (not _dense_bf16_tiles(x, a, shapes, width) or tuple(b.shape) != gram_shape
            or b.device != x.device or b.dtype != x.dtype or not b.is_contiguous()
            or not _matrix_counters(counter, x, batches)
            or not 1 <= round_number <= 5):
        raise ValueError(\"Unsupported ANVIL cooperative round\")
    regions = sorted((tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size())
                     for tensor in (x, a, b, counter))
    if any(end > start for (_, end), (start, _) in zip(regions, regions[1:])):
        raise ValueError(\"ANVIL cooperative round buffers overlap\")
    side = width // 128
    tiles = side * (side + 1) // 2
    blocks = batches * tiles
    layout = gl.NVMMASharedLayout(128, 16)
    src = GluonDescriptor.from_tensor(x.flatten(0, 1), [128, 128], layout)
    gram = GluonDescriptor.from_tensor(a.flatten(0, 1), [128, 128], layout)
    dst = GluonDescriptor.from_tensor(b.flatten(0, 1), [128, 128], layout)
    cooperative_round[(blocks,)](src, gram, dst, a, counter, round_number * batches,
                                  alpha, beta, k, m, tall, num_warps=8, num_stages=1,
                                  launch_cooperative_grid=True)


@anvil_cooperative_round.register_fake
def _fake_cooperative_round(x, a, b, counter, round_number, alpha, beta):
    return None


def _native_polar(a, out, alpha, beta, polynomial):
    batches, m, k = a.shape
    tall = m > k
    width = k if tall else m
    side = width // 128
    grid = (batches * side * (side + 1) // 2,)
    layout = gl.NVMMASharedLayout(128, 16)
    src = GluonDescriptor.from_tensor(a.flatten(0, 1), [128, 128], layout)
    dst = GluonDescriptor.from_tensor(out.flatten(0, 1), [128, 128], layout)
    input_ptr = a if polynomial else None
    polar[grid](src, dst, input_ptr, alpha, beta, k, m, tall, polynomial, False,
                num_warps=8, num_stages=1)


@torch.library.custom_op(\"veritile_track1::anvil_tma_gram\", mutates_args={\"out\"})
def _tma_gram(a: torch.Tensor, out: torch.Tensor) -> None:
    _native_polar(a, out, None, None, False)


@torch.library.custom_op(\"veritile_track1::anvil_tma_polynomial\", mutates_args={\"out\"})
def _tma_polynomial(a: torch.Tensor, out: torch.Tensor, alpha: float, beta: float) -> None:
    _native_polar(a, out, alpha, beta, True)


@_tma_polynomial.register_fake
def _fake_tma_polynomial(a, out, alpha, beta):
    return None


@_tma_gram.register_fake
def _fake_tma_gram(a, out):
    return None


"""


def polar_full_wrappers(source: str) -> str:
    for name, shapes, width in (
        ("XXT", "((6, 256, 768), (2, 768, 768))", "M"),
        ("XTX", "((3, 2816, 768),)", "K"),
        ("ba_plus_cAA", "((6, 256, 256), (2, 768, 768), (3, 768, 768))", "M"),
    ):
        pattern = rf"(?ms)^def {name}(\([^\n]+), out: torch.Tensor\):(.*?)(?=^def |^@triton.jit|\Z)"
        match, = re.finditer(pattern, source)
        grid = "    grid = (batch_size * tiles,)\n"
        if match[2].count(grid) != 1:
            raise ValueError(f"Unrecognized Polar Express wrapper for {name}")
        eligible = f"full_tiles and _dense_bf16_tiles(A, out, {shapes}, {width})"
        kernel = "_tma_polynomial(A, out, alpha, beta)" if name == "ba_plus_cAA" else "_tma_gram(A, out)"
        fast = f"    if {eligible}:\n        {kernel}\n        return out\n"
        body = match[2].replace(grid, fast + grid, 1)
        start, stop = match.span()
        source = source[:start] + f"def {name}{match[1]}, out: torch.Tensor, *, full_tiles: bool = False):" + body + source[stop:]
    source = source.replace("import triton.language as tl\n", "import triton.language as tl\n" + POLAR_GLUON_IMPORTS, 1)
    return source.replace("def XXT(A:", POLAR_GLUON + "def XXT(A:", 1)


REPLICATED_ADAM_SUBSTITUTIONS = (
    ("from track_1_short.perf.kernels.replicated_adam import SCALAR_COLUMNS, fused_adam_\n",
     "from track_1_short.perf.kernels.replicated_adam import SCALAR_COLUMNS, fused_adam_\n"
     "from track_1_short.perf.kernels.replicated_adam_spans import span_tiles, update_spans\n"),
    ("    grad_views: dict[nn.Parameter, Tensor]\n",
     "    grad_views: dict[nn.Parameter, Tensor]\n"
     "    span_plan: Tensor | None = None\n"),
    ("        assert offset == n\n"
     "        groups.append(group)\n",
     "        assert offset == n\n"
     "        if (dtype in (torch.bfloat16, torch.float32) and 512 <= n < 2**31 - 512\n"
     "                and device.type == \"cuda\" and torch.cuda.get_device_capability(device)[0] == 9):\n"
     "            tiles = span_tiles([(segment_of[p], p.numel()) for p in members])\n"
     "            group.span_plan = torch.tensor(tiles, dtype=torch.int32, device=device)\n"
     "        groups.append(group)\n"),
    ("            fused_adam_(group.param, group.grad, group.exp_avg, group.exp_avg_sq, group.segment, self.device_table)\n",
     "            if group.span_plan is None:\n"
     "                fused_adam_(group.param, group.grad, group.exp_avg, group.exp_avg_sq, group.segment, self.device_table)\n"
     "            else:\n"
     "                update_spans(group.param, group.grad, group.exp_avg, group.exp_avg_sq, group.span_plan,\n"
     "                             self.device_table)\n"),
)

VALUE_PULL_SUBSTITUTIONS = (
    ("from track_1_short.perf.kernels.value_embed import GRAD_DTYPE\n",
     "from track_1_short.perf.kernels.value_embed import GRAD_DTYPE\n"
     "from track_1_short.perf.kernels.sparse_row_pack import clear_value_want_, pack_value_gradients\n"
     "from track_1_short.perf.kernels.value_adam_scatter import (\n"
     "    merge_sparse_value_gradients_, scatter_value_gradients_, value_gradient_workspace,\n"
     ")\n"),
    ("        self.grad_shard = torch.zeros(self.owners.rows_per_rank, param.shape[1], dtype=GRAD_DTYPE, device=device)\n",
     "        self.grad_shard, self.grad_workspace, self.grad_touched = value_gradient_workspace(\n"
     "            self.owners.rows_per_rank, param.shape[1], GRAD_DTYPE, device, world_size,\n"
     "        )\n"
     "        # Empty tuple denotes a clean workspace. None selects a full clear.\n"
     "        self.grad_previous = ()\n"),
    ("""        compact = self.grad_accum.index_select(0, pull.want)
        wire = torch.cat([compact[:pull.lo], compact[pull.hi:]])
        received, work = self.owners.send_grads(pull, wire)
        self.grads_in_flight = (compact[pull.lo:pull.hi], received, work, wire)
""",
     """        packets = pack_value_gradients(self.grad_accum, pull.want, pull.lo, pull.hi,
                                       pack_own=pull.peer_want is None)
        if packets is None:
            compact = self.grad_accum.index_select(0, pull.want)
            own = compact[pull.lo:pull.hi]
            wire = torch.cat([compact[:pull.lo], compact[pull.hi:]])
        else:
            own, wire = packets
        received, work = self.owners.send_grads(pull, wire)
        self.grads_in_flight = (own, received, work, wire)
"""),
    ("        self.grad_accum.index_fill_(0, self.live.want.long(), 0)\n",
     """        pull = self.live
        # The local scatter will clear the owned rows it reads directly.
        ids = pull.peer_want if self.grads_in_flight[0] is None else pull.want
        if not clear_value_want_(self.grad_accum, ids):
            self.grad_accum.index_fill_(0, ids.long(), 0)
"""),
    ("""        grad = self.grad_shard.zero_()
        scatter_add_rows(own_grad, pull.want[pull.lo:pull.hi] - first, grad)
        scatter_add_rows(received, pull.asked - first, grad)
        grad.mul_(1.0 / self.world_size)  # exact in fp16: world_size is a power of two
""",
     """        grad = self.grad_shard
        direct_own = own_grad is None
        if direct_own:
            own_grad = self.grad_accum[self.own_rows]
        previous = self.grad_previous
        self.grad_previous = None
        if merge_sparse_value_gradients_(grad, self.grad_workspace, self.grad_touched, own_grad, received,
                                        pull.want, pull.asked, pull.lo, pull.hi, first, self.world_size, previous,
                                        direct_own=direct_own):
            if pull.lo != pull.hi or pull.asked.numel():
                self.grad_previous = (pull.want, pull.asked, pull.lo, pull.hi)
            else:
                self.grad_previous = ()
        else:
            grad.zero_()
            if not scatter_value_gradients_(grad, own_grad, received, pull.want, pull.asked, pull.lo, pull.hi, first,
                                            direct_own=direct_own):
                own_rows = pull.want[pull.lo:pull.hi] - first
                if direct_own:
                    full = own_grad
                    own_grad = full.index_select(0, own_rows)
                    full.index_fill_(0, own_rows.long(), 0)
                scatter_add_rows(own_grad, own_rows, grad)
                scatter_add_rows(received, pull.asked - first, grad)
            grad.mul_(1.0 / self.world_size)
"""),
)

VALUE_EMBED_SUBSTITUTIONS = (
    ("import triton.language as tl\n",
     "import triton.language as tl\n"
     "from track_1_short.perf.kernels.value_embed_accum import accumulate_four_planes_\n"),
    ("""    if t == 0:
        return
    for p, grad in enumerate(plane_grads):
""",
     """    if t == 0:
        return
    if accumulate_four_planes_(token_ids, plane_grads, grad_accum, vocab, d):
        return
    for p, grad in enumerate(plane_grads):
"""),
)

ROW_OWNER_SUBSTITUTIONS = (
    ("from torch import Tensor\n", "from torch import Tensor\n"
     "from weakref import ReferenceType\n"
     "from track_1_short.perf.kernels.sparse_row_fetch import serve_rows\n"),
    ("        self.bounds = np.arange(world_size + 1, dtype=np.int64) * self.rows_per_rank\n",
     """        self.bounds = np.arange(world_size + 1, dtype=np.int64) * self.rows_per_rank
        self.bounds_i32 = self.bounds.astype(np.int32) if 0 <= num_rows < 2**31 else None
"""),
    ("        cuts = np.searchsorted(want_host, self.bounds)\n",
     """        # Matching the int32 dtypes avoids widening the entire want list.
        bounds = self.bounds_i32 if self.bounds_i32 is not None and want_host.dtype == np.int32 else self.bounds
        cuts = np.searchsorted(want_host, bounds)
"""),
    ("        send = torch.from_numpy(send_counts.astype(np.int64))\n",
     "        send = torch.from_numpy(send_counts.astype(np.int64, copy=False))\n"),
    ("        served = shard.index_select(0, pull.asked - self.first_row)\n",
     """        served = serve_rows(shard, pull.asked, self.first_row)
        if served is None:
            served = shard.index_select(0, pull.asked - self.first_row)
"""),
    ("    ids_exchange: object | None = None    # in-flight work filling `asked`\n",
     "    ids_exchange: object | None = None    # in-flight work filling `asked`\n"
     "    slot_directory: Tensor | None = None  # optional lookup directory for this immutable want list\n"
     "    slot_members: tuple[ReferenceType[Tensor], ...] = ()\n"),
)

ROW_PREFETCH_SUBSTITUTIONS = (
    ("""  - No fixed-address row-slot ring or pointer fingerprints. The CUDA-graph side of #360's pull (the
    graph sink ring that holds each step's n-gram gradient until the event) lives in
    perf/cuda_graphs/step_graphs.py (StepGraphs.hold_ngram_grad).
""",
     """  - The table accumulates non-event sink gradients before the next CUDA graph replay. The event
    step's sink is consumed by the gradient send before its graph can replay again.
"""),
    ("from track_1_short.perf.value_embed_pull import ValueEmbedPull\n",
     "from track_1_short.perf.value_embed_pull import ValueEmbedPull\n"
     "from track_1_short.perf.kernels.ngram_slots import EXTRA_WORDS, directory_start, make_directory\n"
     "from weakref import ref\n"),
    ("    def upload(self, rows: np.ndarray) -> Tensor:\n",
     "    def upload(self, rows: np.ndarray, directory: np.ndarray | None = None) -> Tensor:\n"),
    ("""        out = torch.empty(n, dtype=torch.int32, device=self.device)
        out.copy_(buf[:n], non_blocking=True)
""",
     """        count = n
        if directory is not None:
            start = directory_start(n)
            count = start + directory.size
            buf[n:start].zero_()
            buf[start:count].copy_(torch.from_numpy(directory))
        out = torch.empty(count, dtype=torch.int32, device=self.device)
        out.copy_(buf[:count], non_blocking=True)
"""),
    ("self.ngram_ring = PinnedUploadRing(max_ngram_rows, device)\n",
     "self.ngram_ring = PinnedUploadRing(max_ngram_rows + EXTRA_WORDS, device)\n"),
    ("""    return (sorted_unique_rows([b.ngram_ids_cpu for b in batches]),
            value_embed_rows([b.inputs_cpu for b in batches]))
""",
     """    ngram_rows = sorted_unique_rows([b.ngram_ids_cpu for b in batches])
    directory = make_directory(ngram_rows)
    members = () if directory is None else tuple(ref(b.ngram_ids) for b in batches)
    return ngram_rows, value_embed_rows([b.inputs_cpu for b in batches]), directory, members
"""),
    ("            ngram_want, value_embed_want = cycle_wants(self._batches(cycle), self.value_embeds.want_rows)\n"
     "            self.ngram.start_cycle(self.ngram.fill(ngram_want))\n",
     "            ngram_want, value_embed_want, _, _ = cycle_wants(self._batches(cycle), self.value_embeds.want_rows)\n"
     "            self.ngram.start_cycle(self.ngram.fill(ngram_want))\n"),
    ("            ngram_want, value_embed_want = future.result()\n"
     "        else:\n"
     "            ngram_want, value_embed_want = cycle_wants(self._batches(cycle), self.value_embeds.want_rows)\n",
     "            ngram_want, value_embed_want, slot_directory, members = future.result()\n"
     "        else:\n"
     "            ngram_want, value_embed_want, slot_directory, members = cycle_wants(self._batches(cycle), self.value_embeds.want_rows)\n"),
    ("        ngram_pull = self.ngram.request(self.staging.ngram_ring.upload(ngram_want), ngram_want)\n",
     """        packet = self.staging.ngram_ring.upload(ngram_want, slot_directory)
        want = packet if slot_directory is None else packet[:ngram_want.size]
        ngram_pull = self.ngram.request(want, ngram_want)
        if slot_directory is not None:
            ngram_pull.slot_directory = packet[directory_start(ngram_want.size):]
            ngram_pull.slot_members = members
"""),
    ("    def forward_slots(self, step: int, row_ids: Tensor) -> Tensor:\n",
     "    def forward_slots(self, step: int, row_ids: Tensor, *, defer_slots: bool = False, hashed_rows: bool = False):\n"),
    ("        return self.ngram.slots(self.ngram.live, row_ids)\n",
     "        return self.ngram.slots(self.ngram.live, row_ids, defer=defer_slots, hashed_rows=hashed_rows,\n"
     "                                member_index=i - self.live_cycle[0])\n"),
)

NGRAM_CPU_HASH = """def _ngram_remainder_(words: np.ndarray, scratch: np.ndarray, modulus: int) -> None:
    # With this modulus the estimated quotient is either exact or one too large.
    np.multiply(words, 51, dtype=np.int64, out=scratch)
    scratch += 1 << 30
    scratch >>= 31
    scratch *= modulus
    np.subtract(words, scratch, out=words, casting="unsafe")
    signs = scratch.view(np.int32)[:words.size]
    np.right_shift(words, 31, out=signs)
    np.bitwise_and(signs, modulus, out=signs)
    words += signs


def _ngram_cpu_rows(x: Tensor, half: int) -> Tensor:
    tokens = x.numpy()
    n = tokens.size
    modulus = half - 1
    output = np.empty(2 * n, dtype=np.int32)
    bigram, trigram = output[:n], output[n:]
    scratch = np.empty(n - 1, dtype=np.int64)
    products = scratch.view(np.int32)[:n - 1]

    bigram[0] = modulus
    np.multiply(tokens[1:], BIGRAM_ROW_MULS[0], out=bigram[1:])
    np.multiply(tokens[:-1], BIGRAM_ROW_MULS[1], out=products)
    # The previous term of row 1 is the reserved ID, with int32 wrapping.
    products[0] = (modulus * BIGRAM_ROW_MULS[1] + (1 << 31)) % (1 << 32) - (1 << 31)
    np.bitwise_xor(bigram[1:], products, out=bigram[1:])
    _ngram_remainder_(bigram[1:], scratch, modulus)

    trigram[:2] = half + modulus
    np.multiply(tokens[2:], TRIGRAM_ROW_MULS[0], out=trigram[2:])
    products = products[:-1]
    np.multiply(tokens[1:-1], TRIGRAM_ROW_MULS[1], out=products)
    np.bitwise_xor(trigram[2:], products, out=trigram[2:])
    np.multiply(tokens[:-2], TRIGRAM_ROW_MULS[2], out=products)
    np.bitwise_xor(trigram[2:], products, out=trigram[2:])
    _ngram_remainder_(trigram[2:], scratch[:-1], modulus)
    trigram[2:] += half
    return torch.from_numpy(output)


"""

NGRAM_TABLE_SUBSTITUTIONS = (
    ("        self.last_event.zero_()\n        self.adam_events = 0\n",
     "        self.last_event.zero_()\n        self.grad_per_slot.zero_()\n        self.adam_events = 0\n"),
    ("""tensor. accumulate_grad keeps (slots, sink grad) per step; at the event the gradient is summed per
cache slot (in fp16) and travels back along the pull's routes in reverse (send_grads, bf16 on the wire),
""",
     """tensor. accumulate_grad adds non-event gradients to their slots in fp16. The event adds its own
gradient and sends the sums back along the pull's routes in reverse (send_grads, bf16 on the wire),
"""),
    ("""        self.pending: list[tuple[Tensor, Tensor]] = []  # (slots, sink grad) per step of the live cycle
""",
     """        self.ngram_event_grad: tuple[Tensor, Tensor] | None = None
        self.ngram_sum_started = False
"""),
    ("        assert self.cached is pull and not self.pending\n",
     """        assert self.cached is pull and self.grads_in_flight is None
        assert self.ngram_event_grad is None and not self.ngram_sum_started
"""),
    ("""    def accumulate_grad(self, slots: Tensor, sink_grad: Tensor):
        \"\"\"Hold one step's (slots, sink gradient) until the cycle's event. `sink_grad` must not be
        rewritten before then (with CUDA graphs: StepGraphs.hold_ngram_grad).\"\"\"
        assert self.cached is self.live, "the forward read a cache that holds no training cycle"
        self.pending.append((slots, sink_grad))
""",
     """    @torch.no_grad()
    def accumulate_grad(self, slots: Tensor, sink_grad: Tensor, event_this_step: bool):
        \"\"\"Accumulate before the next graph replay can overwrite this step's gradient.\"\"\"
        assert self.cached is self.live, "the forward read a cache that holds no training cycle"
        assert self.ngram_event_grad is None
        if event_this_step:
            self.ngram_event_grad = slots, sink_grad
            return
        rows = self.live.num_rows
        assert rows <= self.grad_per_slot.shape[0], f"{rows} cycle rows > gradient buffer"
        per_slot = self.grad_per_slot[:rows]
        scatter_add_rows(sink_grad.contiguous(), slots, per_slot)
        self.ngram_sum_started = True
"""),
    ("""        assert self.pending, "Adam event with no gradient since the last one"
        assert pull.num_rows <= self.grad_per_slot.shape[0], f"{pull.num_rows} cycle rows > gradient buffer"
        per_slot = self.grad_per_slot[:pull.num_rows].zero_()
        for slots, grad in self.pending:
            scatter_add_rows(grad, slots, per_slot)
        self.pending = []
""",
     """        assert self.ngram_event_grad is not None, "Adam event with no gradient since the last one"
        assert pull.num_rows <= self.grad_per_slot.shape[0], f"{pull.num_rows} cycle rows > gradient buffer"
        per_slot = self.grad_per_slot[:pull.num_rows]
        slots, grad = self.ngram_event_grad
        scatter_add_rows(grad, slots, per_slot)
        self.ngram_event_grad = None
        self.ngram_sum_started = False
"""),
    ("""    rows = np.sort(np.concatenate(row_id_arrays))
    if rows.size:
        rows = rows[np.r_[True, rows[1:] != rows[:-1]]]
""",
     """    rows = np.concatenate(row_id_arrays)
    rows.sort()
    if rows.size:
        keep = np.empty(rows.size, dtype=np.bool_)
        keep[0] = True
        np.not_equal(rows[1:], rows[:-1], out=keep[1:])
        rows = rows[keep]
"""),
    ("def ngram_row_ids(x: Tensor) -> Tensor:\n",
     NGRAM_CPU_HASH + "def ngram_row_ids(x: Tensor) -> Tensor:\n"),
    ("    n = x.numel()\n    out = torch.empty(2 * n, dtype=torch.int32)\n",
     "    n = x.numel()\n"
     "    if (type(x) is Tensor and x.device.type == \"cpu\" and x.ndim == 1 and x.is_contiguous()\n"
     "            and n >= 16384 and bigram_mod == trigram_mod == 42301439):\n"
     "        return _ngram_cpu_rows(x, half)\n"
     "    out = torch.empty(2 * n, dtype=torch.int32)\n"),
    ("from track_1_short.perf.kernels.row_scatter import scatter_add_rows\n",
     "from track_1_short.perf.kernels.row_scatter import scatter_add_rows\n"
     "from track_1_short.perf.kernels.sparse_row_pack import pack_ngram_gradients\n"
     "from track_1_short.perf.kernels.sparse_row_fetch import land_ngram_\n"
     "from track_1_short.perf.kernels.ngram_moment_replay import replay_next_moments_\n"
     "from track_1_short.perf.kernels.ngram_adam_entries import pack_adam_entries\n"
     "from track_1_short.perf.kernels.ngram_slots import SlotQuery, prepare_bucket_slots, prime_bucket_slots\n"),
    ("        self.reset()\n", "        self.reset()\n        prime_bucket_slots(device)\n"),
    ("    def slots(self, pull: RowPull, row_ids: Tensor) -> Tensor:\n",
     "    def slots(self, pull: RowPull, row_ids: Tensor, *, defer: bool = False, hashed_rows: bool = False,\n"
     "              member_index: int = -1) -> Tensor | SlotQuery:\n"),
    ("        return torch.searchsorted(pull.want, row_ids, out_int32=True)\n",
     """        members = pull.slot_members
        member = hashed_rows and 0 <= member_index < len(members) and members[member_index]() is row_ids
        query = prepare_bucket_slots(pull.want, pull.slot_directory, row_ids,
                                     hashed_rows=hashed_rows, member_rows=member)
        if query is not None:
            return query if defer else query.materialize()
        return torch.searchsorted(pull.want, row_ids, out_int32=True)
"""),
    ("""        self.cache[:lo].copy_(received[:lo])
        self.cache[hi:n].copy_(received[lo:])
        torch.index_select(self.shard, 0, pull.want[lo:hi] - self.first_row, out=self.cache[lo:hi])
""",
     """        if not land_ngram_(self.cache, self.shard, received, pull.want, lo, hi, self.first_row):
            self.cache[:lo].copy_(received[:lo])
            self.cache[hi:n].copy_(received[lo:])
            torch.index_select(self.shard, 0, pull.want[lo:hi] - self.first_row, out=self.cache[lo:hi])
"""),
    ("""        wire = torch.empty(pull.num_rows - (hi - lo), NGRAM_DIM, dtype=GRAD_WIRE_DTYPE, device=self.device)
        wire[:lo].copy_(per_slot[:lo])
        wire[lo:].copy_(per_slot[hi:])
""",
     """        wire = pack_ngram_gradients(per_slot, lo, hi, clear_source=True)
        reset_whole = wire is None
        if reset_whole:
            wire = torch.empty(pull.num_rows - (hi - lo), NGRAM_DIM, dtype=GRAD_WIRE_DTYPE, device=self.device)
            wire[:lo].copy_(per_slot[:lo])
            wire[lo:].copy_(per_slot[hi:])
"""),
    ("        self.grads_in_flight = (per_slot[lo:hi], received, work, wire)\n",
     "        self.grads_in_flight = (per_slot[lo:hi], received, work, wire, reset_whole)\n"),
    ("        own_grad, received, work, _ = self.grads_in_flight\n",
     "        own_grad, received, work, _, reset_whole = self.grads_in_flight\n"),
    ("        scatter_add_rows(own_grad, own_rows, grad, row_map=self.entry_of_row)\n",
     """        scatter_add_rows(own_grad, own_rows, grad, row_map=self.entry_of_row, clear_source=not reset_whole)
        if reset_whole:
            self.grad_per_slot[:pull.num_rows].zero_()
"""),
    ("""        own_rows = pull.want[pull.lo:pull.hi] - self.first_row
        asked_rows = pull.asked - self.first_row
        entries = torch.cat([own_rows, asked_rows])
""",
     """        packed = pack_adam_entries(pull.want, pull.asked, pull.lo, pull.hi, self.first_row,
                                   entry_of_row=self.entry_of_row)
        if packed is None:
            own_rows = pull.want[pull.lo:pull.hi] - self.first_row
            asked_rows = pull.asked - self.first_row
            entries = torch.cat([own_rows, asked_rows])
        else:
            own_rows, asked_rows, entries, grad = packed
"""),
    ("""        claim_rows(entries, self.entry_of_row)
        grad = torch.zeros(entries.numel(), NGRAM_DIM, dtype=GRAD_MERGE_DTYPE, device=self.device)
""",
     """        if packed is None:
            claim_rows(entries, self.entry_of_row)
            grad = torch.zeros(entries.numel(), NGRAM_DIM, dtype=GRAD_MERGE_DTYPE, device=self.device)
"""),
    ("                   grad_mul=1 / self.world_size, sq_mul=(1 - beta2) / self.world_size ** 2)\n",
     "                   grad_mul=1 / self.world_size, sq_mul=(1 - beta2) / self.world_size ** 2,\n"
     "                   positive_history=True)\n"),
    ("""            bring_rows_current(self.shard, self.exp_avg_sq, self.last_event, self.beta2_history,
                               pull.want[pull.lo:pull.hi] - self.first_row, self.adam_events)
""",
     """            if not replay_next_moments_(self.shard, self.exp_avg_sq, self.last_event, self.beta2_history,
                                        pull.want, pull.lo, pull.hi, self.first_row, self.adam_events,
                                        positive_history=True):
                bring_rows_current(self.shard, self.exp_avg_sq, self.last_event, self.beta2_history,
                                   pull.want[pull.lo:pull.hi] - self.first_row, self.adam_events)
"""),
)

NGRAM_ADAM_SUBSTITUTIONS = (
    ("                      D: tl.constexpr, HAS_GRAD: tl.constexpr, BLOCK: tl.constexpr):\n",
     "                      D: tl.constexpr, HAS_GRAD: tl.constexpr, BLOCK: tl.constexpr,\n"
     "                      POSITIVE_HISTORY: tl.constexpr = False):\n"),
    ("""    last = tl.load(LAST_EVENT + row)
    if HAS_GRAD:
        commit = tl.load(ENTRY_OF_ROW + row) == pid
        # the decays of events last+1 .. event-1; this event's own is part of the update below
        missed = tl.where(commit, event - 1 - last, 0)
    else:
        missed = event - last
    v = tl.load(V + row64)
    for i in range(missed):
        v = v * tl.load(BETA2_HISTORY + (last + 1 + i))
    if HAS_GRAD:
        if commit:
""",
     """    commit = True
    if HAS_GRAD:
        commit = tl.load(ENTRY_OF_ROW + row) == pid
    if commit:
        last = tl.load(LAST_EVENT + row)
        v = tl.load(V + row64)
        # Adam applies the current beta2 with the gradient below.
        end = event - 1 if HAS_GRAD else event
        if POSITIVE_HISTORY:
            last = tl.where(v.to(tl.int32, bitcast=True) == 0, end, last)
        for i in range(end - last):
            v = v * tl.load(BETA2_HISTORY + (last + 1 + i))
        if HAS_GRAD:
"""),
    ("""    else:
        tl.store(V + row64, v)
        tl.store(LAST_EVENT + row, event)
""",
     """        else:
            tl.store(V + row64, v)
            tl.store(LAST_EVENT + row, event)
"""),
    ("               eps: float, step_size: float, decay: float, grad_mul: float, sq_mul: float):\n",
     "               eps: float, step_size: float, decay: float, grad_mul: float, sq_mul: float,\n"
     "               positive_history: bool = False):\n"),
    ("            D=shard.shape[1], HAS_GRAD=True, BLOCK=triton.next_power_of_2(shard.shape[1]), num_warps=4,\n",
     "            D=shard.shape[1], HAS_GRAD=True, BLOCK=triton.next_power_of_2(shard.shape[1]),\n"
     "            POSITIVE_HISTORY=positive_history, num_warps=4,\n"),
)

MAPPED_ROW_SCATTER = """@gluon.jit(do_not_specialize=["count"])
def _scatter_mapped_rows_kernel(source, indices, dest, row_map, count, CLEAR_SOURCE: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    row = 4 * gl.program_id(0) + gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
    columns = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
    live = row < count
    indices = gl.load(indices + row, live, other=0)
    entry = gl.load(row_map + indices, live, other=0)
    src = source + row[:, None].to(gl.int64) * 768 + columns[None, :]
    dst = dest + entry[:, None].to(gl.int64) * 768 + columns[None, :]
    for stripe in gl.static_range(3):
        words = gl.load(src + 256 * stripe, live[:, None], other=0.0).to(gl.float32)
        gl.atomic_add(dst + 256 * stripe, words, sem="relaxed", mask=live[:, None])
        if CLEAR_SOURCE:
            gl.store(src + 256 * stripe, 0.0, live[:, None])


"""

ROW_SCATTER_SUBSTITUTIONS = (
    ("import triton.language as tl\n",
     "import triton.language as tl\nfrom triton.experimental import gluon\n"
     "from triton.experimental.gluon import language as gl\n"),
    ("@triton.jit\ndef _scatter_add_rows_kernel(", MAPPED_ROW_SCATTER + "@triton.jit\ndef _scatter_add_rows_kernel("),
    ("                             HAS_MAP: tl.constexpr):\n",
     "                             HAS_MAP: tl.constexpr, CLEAR_SOURCE: tl.constexpr = False):\n"),
    ("""        val = tl.load(SRC + (sbase + k * BLOCK + o64)).to(tl.float32)
        tl.atomic_add(DST + (dbase + k * BLOCK + o64), val, sem="relaxed")
""",
     """        val = tl.load(SRC + (sbase + k * BLOCK + o64)).to(tl.float32)
        tl.atomic_add(DST + (dbase + k * BLOCK + o64), val, sem="relaxed")
        if CLEAR_SOURCE:
            tl.store(SRC + (sbase + k * BLOCK + o64), 0.0)
"""),
    ("def scatter_add_rows(src: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor, row_map: torch.Tensor | None = None):\n",
     """def scatter_add_rows(src: torch.Tensor, idx: torch.Tensor, dst: torch.Tensor, row_map: torch.Tensor | None = None,
                     *, clear_source: bool = False):
"""),
    ("""    \"\"\"dst[row_map[idx[i]] if row_map is given else idx[i]] += src[i], in dst's dtype; idx may repeat.\"\"\"
""",
     """    \"\"\"Add into dst in storage precision. With clear_source, src is disjoint from the other buffers.\"\"\"
"""),
    ("    if src.shape[0]:\n        _scatter_add_rows_kernel[(src.shape[0],)](\n",
     """    source_type = torch.float16 if clear_source else torch.bfloat16
    if (row_map is not None and src.shape[0] and d == 768
            and src.dtype == source_type and dst.dtype == torch.bfloat16 and idx.dtype == torch.int32
            and src.is_cuda and torch.cuda.get_device_capability(src.device)[0] == 9
            and all(t.device == src.device and t.data_ptr() % 16 == 0 for t in (src, idx, dst, row_map))):
        _scatter_mapped_rows_kernel[(triton.cdiv(src.shape[0], 4),)](
            src, idx, dst, row_map, src.shape[0], clear_source, num_warps=4, num_stages=1,
        )
        return dst
    if src.shape[0]:
        _scatter_add_rows_kernel[(src.shape[0],)](
"""),
    ("            HAS_MAP=row_map is not None, num_warps=max(1, min(8, block // 64)), num_stages=1,\n",
     """            HAS_MAP=row_map is not None, CLEAR_SOURCE=clear_source,
            num_warps=max(1, min(8, block // 64)), num_stages=1,
"""),
)

FINEWEB_PACK = """def _pack_bos_batch(tokens: Tensor, starts: list[int], ends: list[int],
                    count: int, max_seq_len: int, rows: int):
    source = tokens.numpy()
    buf = np.concatenate([source[i:j] for i, j in zip(starts, ends)])
    offsets = np.subtract(ends, starts, dtype=np.int64)
    np.cumsum(offsets, out=offsets)
    # Drop the final input position while retaining it as the last target.
    offsets[-1] -= 1
    if max_seq_len > VIRTUAL_SEQ_CAP:
        offsets = [end for start, stop in pairwise([0, *offsets.tolist()])
                   for end in (*range(start + VIRTUAL_SEQ_CAP, stop, VIRTUAL_SEQ_CAP), stop)]
        assert len(offsets) < rows, f"{len(offsets)} attention segments overflow the {rows}-row cu_seqlens table"

    cumulative = np.full(rows, count, dtype=np.int32)
    cumulative[0] = 0
    cumulative[1:len(offsets) + 1] = offsets
    return (torch.from_numpy(buf[:-1].astype(np.int32)),
            torch.from_numpy(buf[1:].astype(np.int64)), torch.from_numpy(cumulative))


"""

FINEWEB_TORCH_PACK = """            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)
            if max_seq_len > VIRTUAL_SEQ_CAP:
                cum_lengths = split_attention_segments(cum_lengths, VIRTUAL_SEQ_CAP)
                assert len(cum_lengths) < max_num_docs, f"{len(cum_lengths)} attention segments overflow the {max_num_docs}-row cu_seqlens table"
"""

FINEWEB_TORCH_CASTS = """        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
"""

FINEWEB_DATA_SUBSTITUTIONS = (
    ("def distributed_data_generator(filename_pattern:", FINEWEB_PACK + "def distributed_data_generator(filename_pattern:"),
    ("        max_num_docs = cu_seqlens_rows(num_tokens_local)\n",
     "        max_num_docs = cu_seqlens_rows(num_tokens_local)\n        packed = False\n"),
    ("                start_idxs, end_idxs = torch.tensor(seq_starts[rank]), torch.tensor(seq_ends[rank])\n",
     "                starts, ends = seq_starts[rank], seq_ends[rank]\n"),
    (FINEWEB_TORCH_PACK,
     """            packed = (num_tokens_local in TRAIN_MAX_NUM_DOCS and len(starts) < max_num_docs
                      and type(tokens) is Tensor and tokens.device.type == "cpu"
                      and tokens.dtype == torch.uint16 and tokens.ndim == 1 and tokens.is_contiguous())
            if packed:
                _inputs, _targets, _cum_lengths = _pack_bos_batch(
                    tokens, starts, ends, num_tokens_local, max_seq_len, max_num_docs,
                )
            else:
                start_idxs, end_idxs = torch.tensor(starts), torch.tensor(ends)
""" + indent(FINEWEB_TORCH_PACK, "    ")),
    (FINEWEB_TORCH_CASTS, "        if not packed:\n" + indent(FINEWEB_TORCH_CASTS, "    ")),
    ("        inputs, targets, cum_seqlens, ngram_ids = staging.upload(_inputs, _targets, _cum_lengths, _ngram_ids)\n",
     """        inputs, targets, cum_seqlens, ngram_ids = staging.upload(
            _inputs, _targets, _cum_lengths, _ngram_ids, coalesce=packed,
        )
"""),
)

SCHEDULE_PACK_ROWS = """def _upload_weight_rows(rows, device):
    if torch.get_default_dtype() != torch.float32 or not all(0 < len(row) <= 4 for row in rows):
        return [torch.tensor(row, device=device) for row in rows]

    # Each row starts at a 16-byte boundary, including for one-word views.
    table = torch.tensor([row + [0.0] * (4 - len(row)) for row in rows], device=device)
    result = []
    first = 0
    for width, run in groupby(map(len, rows)):
        stop = first + sum(1 for _ in run)
        result.extend(table[first:stop, :width].unbind(0))
        first = stop
    return result


"""

SCHEDULE_SUBSTITUTIONS = (
    ("from itertools import accumulate, pairwise\n", "from itertools import accumulate, groupby, pairwise\n"),
    ("class TrainingSchedule:\n", SCHEDULE_PACK_ROWS + "class TrainingSchedule:\n"),
    ("""        self.mtp_weights = []
        self.prefix_weights = []
        for step in range(self.total_steps + 1):
            stage, t = self.lookup(step)
            w = [a + (b - a) * t for a, b in zip(stage.mtp_weights_start, stage.mtp_weights_end)]
            self.mtp_weights.append(torch.tensor(w, device=device))
            self.prefix_weights.append(torch.tensor([self.prefix_weight(step)], device=device))
""",
     """        mtp_rows = []
        prefix_rows = []
        for step in range(self.total_steps + 1):
            stage, t = self.lookup(step)
            w = [a + (b - a) * t for a, b in zip(stage.mtp_weights_start, stage.mtp_weights_end)]
            mtp_rows.append(w)
            prefix_rows.append([self.prefix_weight(step)])
        self.mtp_weights = _upload_weight_rows(mtp_rows, device)
        self.prefix_weights = _upload_weight_rows(prefix_rows, device)
"""),
)

TAIL_SUBSTITUTIONS = (
    ("from track_1_short.perf.kernels.lerp import lerp_upcast_\n",
     "from track_1_short.perf.kernels.lerp import lerp_upcast_\n"
     "from track_1_short.perf.kernels.tail_lerp import lerp_shared_source_\n"
     "from track_1_short.perf.kernels.tail_norm import (\n"
     "    can_restore_tail_norms, restore_tail_norms_, tail_norms, warm_tail_norms,\n"
     ")\n"),
    ("            lerp_upcast_(scratch, scratch.bfloat16(), 0.5)\n",
     """            source = scratch.bfloat16()
            lerp_upcast_(scratch, source, 0.5)
            if divisible:
                lerp_shared_source_(scratch, torch.zeros_like(scratch), source, 0.5, 0.5)
        warm_tail_norms(tuple(self._param(label).data for label in DECONTRACTION_LABELS))
"""),
    ("""        for acc in self.accumulators:
            if step < self.total_steps - acc.window:
""",
     """        bank_blend = self.accumulators[2]
        bank_rate = (bank_blend.rate(step, self.total_steps)
                     if step >= self.total_steps - bank_blend.window else None)
        fused = set()
        for acc in self.accumulators:
            if step < self.total_steps - acc.window:
"""),
    ("            rate = acc.rate(step, self.total_steps)\n",
     "            rate = bank_rate if acc is bank_blend else acc.rate(step, self.total_steps)\n"),
    ("""                shard = self._own_shard(label)
                if label not in acc.seeded:
""",
     """                if acc is bank_blend and label in fused:
                    continue
                shard = self._own_shard(label)
                if label not in acc.seeded:
"""),
    ("""                    acc.seeded.add(label)
                else:
                    lerp_upcast_(acc.bufs[label], shard, rate)
""",
     """                    acc.seeded.add(label)
                elif acc.tag == "tail-avg" and bank_rate is not None and label in bank_blend.seeded and (
                    lerp_shared_source_(acc.bufs[label], bank_blend.bufs[label], shard, rate, bank_rate)
                ):
                    # Each accumulator has private storage, so moving this bank update is safe.
                    fused.add(label)
                else:
                    lerp_upcast_(acc.bufs[label], shard, rate)
"""),
    ("        norms_before = torch.stack([self._param(l).data.float().norm() for l in DECONTRACTION_LABELS])\n",
     """        weights = tuple(self._param(label).data for label in DECONTRACTION_LABELS)
        direct_norms = can_restore_tail_norms(weights)
        norms_before = tail_norms(weights, direct_norms)
"""),
    ("""        norms_after = torch.stack([self._param(l).data.float().norm() for l in DECONTRACTION_LABELS])
        # Python-float ratios, so the product rounds to bf16 once, on the store.
        ratios = torch.where(norms_after > 0, norms_before / norms_after, torch.ones_like(norms_after)).tolist()
        for label, ratio in zip(DECONTRACTION_LABELS, ratios):
            self._param(label).data.mul_(ratio)
""",
     """        norms_after = tail_norms(weights, direct_norms)
        ratios = torch.where(norms_after > 0, norms_before / norms_after, torch.ones_like(norms_after))
        if direct_norms:
            restore_tail_norms_(weights, ratios)
        else:
            for weight, ratio in zip(weights, ratios.tolist()):
                weight.mul_(ratio)
"""),
)
SAMPLED_IMPORT = "from track_1_short.perf.kernels.transpose import transpose_copy\n"
GATHER_IMPORT = "from track_1_short.perf.kernels.sampled_rows_gather import gather_sampled_rows\n" + SAMPLED_IMPORT
SAMPLED_COPY = """        torch.index_select(lm_head_f8_col.T.view(torch.uint8), 0, cand, out=inputs.rows.view(torch.uint8))
        transpose_copy(inputs.rows.view(torch.uint8), inputs.rows_t.view(torch.uint8))
        inputs.vocab_pos.fill_(-1)
        inputs.vocab_pos.index_copy_(0, cand, self.arange[:P])
"""
GATHER_COPY = """        weights = lm_head_f8_col.T.view(torch.uint8)
        rows = inputs.rows.view(torch.uint8)
        transposed = inputs.rows_t.view(torch.uint8)
        if gather_sampled_rows(weights, cand, rows, transposed, inverse=inputs.vocab_pos):
            return
        torch.index_select(weights, 0, cand, out=rows)
        transpose_copy(rows, transposed)
        inputs.vocab_pos.fill_(-1)
        inputs.vocab_pos.index_copy_(0, cand, self.arange[:P])
"""
SAMPLED_CE_IMPORT = "from track_1_short.sampled_softmax import ALL_CANDIDATE_COUNTS\n"
DENSIFY_IMPORT = (
    "from track_1_short.perf.kernels.sampled_grad_densify import densify_sampled_gradient\n" + SAMPLED_CE_IMPORT
)
DENSIFY_LAUNCH = """    _densify_kernel[(triton.cdiv(dst.shape[1], 256), triton.cdiv(dst.shape[0], 32))](
        src, pos, dst, *dst.shape, src.stride(0), dst.stride(0), BLOCK_R=32, BLOCK_V=256, num_warps=4, num_stages=3)
"""
DENSIFY_DISPATCH = "    if densify_sampled_gradient(src, pos, dst):\n        return\n" + DENSIFY_LAUNCH
ROPE_KEY_GUARD = "    if KEY_OFFSET:\n        # Partial key offset:"
ROPE_PREVIOUS_GUARD = "    if KEY_OFFSET and not SHIFT_STORE:\n        # Partial key offset:"
ROPE_BLOCK_ARGUMENTS = "    BLOCK_M: tl.constexpr,\n    BLOCK_D: tl.constexpr,\n"
ROPE_COMPACT_ARGUMENTS = ("    COMPACT_FACTORS: tl.constexpr = False,\n"
                          "    FORCE_SIGN_BITS: tl.constexpr = False,\n"
                          "    IMPLICIT_STATIONARY: tl.constexpr = False,\n")
ROPE_SHIFT_ARGUMENTS = ROPE_BLOCK_ARGUMENTS + "    SHIFT_STORE: tl.constexpr,\n" + ROPE_COMPACT_ARGUMENTS
ROPE_KERNEL_ANCHOR = "@triton.jit\ndef _qk_norm_rope_forward_kernel("
ROPE_PAIR_HELPER = """@triton.jit
def _rope_pair_factors(factor1, factor2, row_offsets, row_mask, qk_dim: tl.constexpr,
                       BLOCK_D: tl.constexpr, REVERSE: tl.constexpr, FORCE_SIGN_BITS: tl.constexpr,
                       IMPLICIT_STATIONARY: tl.constexpr):
    pair = tl.arange(0, BLOCK_D // 2)
    offsets = row_offsets + pair[None, :]
    mask = row_mask[:, None] & (pair[None, :] < qk_dim // 2)
    if IMPLICIT_STATIONARY:
        # Supply the original BF16 values before the ordered FP32 arithmetic.
        mask = mask & (pair[None, :] < 32)
        cosine = tl.load(factor1 + offsets, mask=mask, other=1.0)
    else:
        cosine = tl.load(factor1 + offsets, mask=mask, other=0.0)
    sine = tl.load(factor2 + offsets, mask=mask, other=0.0)
    # Keep the sign change on the factor before widening and multiplying.
    if FORCE_SIGN_BITS:
        negative_sine = tl.inline_asm_elementwise(
            "xor.b16 $0, $1, 0x8000;", constraints="=h,h",
            args=[sine], dtype=tl.bfloat16, is_pure=True, pack=1,
        )
    else:
        negative_sine = (sine.to(tl.uint16, bitcast=True) ^ 0x8000).to(tl.bfloat16, bitcast=True)
    if REVERSE:
        f2 = tl.interleave(negative_sine, sine)
    else:
        f2 = tl.interleave(sine, negative_sine)
    return tl.interleave(cosine, cosine).to(tl.float32), f2.to(tl.float32)


""" + ROPE_KERNEL_ANCHOR
ROPE_FORWARD_FACTORS = """    factor_ptrs = token[:, None] * factor_stride_t + factor_offset + offs_d[None, :]
    f1 = tl.load(factor1 + factor_ptrs, mask=mask, other=0.0).to(tl.float32)
    f2 = tl.load(factor2 + factor_ptrs, mask=mask, other=0.0).to(tl.float32)
    y = f1 * (x * rstd[:, None]) + f2 * (x_flip * rstd[:, None])
"""
ROPE_COMPACT_FORWARD = """    if COMPACT_FACTORS:
        factor_rows = tl.full((BLOCK_M,), True, tl.int1) if IMPLICIT_STATIONARY and rows % BLOCK_M == 0 else offs_m < rows
        f1, f2 = _rope_pair_factors(factor1, factor2, token[:, None] * factor_stride_t + factor_offset // 2,
                                    factor_rows, qk_dim, BLOCK_D, False, FORCE_SIGN_BITS, IMPLICIT_STATIONARY)
        y = tl.fma(rstd[:, None] * x, f1, (rstd[:, None] * x_flip) * f2)
    else:
""" + indent(ROPE_FORWARD_FACTORS, "    ")
ROPE_FACTOR_SHAPE = "    assert factor1.shape == factor2.shape == (tokens, qk_dim * (2 if paired else 1))\n"
ROPE_COMPACT_SHAPE = """    factor_width = qk_dim * (2 if paired else 1)
    assert factor1.shape == factor2.shape
    if factor1.shape != (tokens, factor_width):
        assert factor1.dtype == factor2.dtype == torch.bfloat16
        assert (qk_dim in (64, 128) and factor1.shape == (tokens, factor_width // 2)) or (
            qk_dim == 128 and not paired and factor1.shape == (tokens, 32)
        )
"""
ROPE_KEY_STORE = "    tl.store(out_k + output_ptrs, y, mask=mask & (input_head[:, None] >= num_heads))\n"
ROPE_SHIFT_STORE = """    if SHIFT_STORE:
        key = input_head[:, None] >= num_heads
        stationary = offs_d[None, :] >= rotary_dim
        tl.store(out_k + output_ptrs, y, mask=mask & key & ((~stationary) | (token[:, None] == 0)))
        tl.store(out_k + output_ptrs + stride_out_t, x * rstd[:, None],
                 mask=mask & key & stationary & (token[:, None] + 1 < rows // heads2))
    else:
        tl.store(out_k + output_ptrs, y, mask=mask & (input_head[:, None] >= num_heads))
"""
ROPE_LAUNCH = """    rows = tokens * heads2
    _qk_norm_rope_forward_kernel[(triton.cdiv(rows, block_m),)](
        qk, factor1, factor2, out_q, out_k,
        rows=rows, num_heads=num_heads, heads2=heads2,
        qk_dim=qk_dim, rotary_dim=rotary_dim,
        stride_qkt=qk.stride(0), stride_qkh=qk.stride(1),
        factor_stride_t=factor1.stride(0), stride_out_t=out_q.stride(0),
        PAIRED=paired, KEY_OFFSET=key_offset,
        BLOCK_M=block_m, BLOCK_D=triton.next_power_of_2(qk_dim), num_warps=num_warps,
    )
    return out_q, out_k
"""
ROPE_SHIFT_LAUNCH = """    _launch_qk_norm_rope(qk, factor1, factor2, out_q, out_k, num_heads,
                         rotary_dim, paired, key_offset, block_m, num_warps)
    return out_q, out_k


@torch.library.custom_op("nanogpt::track1_qk_norm_rope", mutates_args={"out_q", "out_k"})
def _launch_qk_norm_rope(
    qk: torch.Tensor, factor1: torch.Tensor, factor2: torch.Tensor,
    out_q: torch.Tensor, out_k: torch.Tensor, num_heads: int, rotary_dim: int,
    paired: bool, key_offset: bool, block_m: int, num_warps: int,
) -> None:
    tokens, heads2, qk_dim = qk.shape
    # Pointer alignment is checked at launch time, outside Dynamo tracing.
    rows = tokens * heads2
    implicit_stationary = qk_dim == 128 and not paired and factor1.shape[1] == 32
    compact_factors = implicit_stationary or (
        qk_dim in (64, 128) and factor1.shape[1] == (qk_dim * (2 if paired else 1)) // 2
    )
    shift_store = (
        key_offset and not paired and num_warps == 2
        and num_heads == 6 and qk_dim == 128 and rotary_dim == 64
        and tokens in (16384, 32768, 40960, 49152, 262144)
        and qk.stride() == (2304, 128, 1)
        and factor1.stride() == factor2.stride() == (factor1.shape[1], 1)
        and qk.dtype == factor1.dtype == factor2.dtype == torch.bfloat16
        and qk.is_cuda and factor1.device == factor2.device == qk.device
        and torch.cuda.get_device_capability(qk.device)[0] == 9
        and all(t.data_ptr() % 16 == 0 for t in (qk, factor1, factor2, out_q, out_k))
    )
    _qk_norm_rope_forward_kernel[(triton.cdiv(rows, block_m),)](
        qk, factor1, factor2, out_q, out_k,
        rows=rows, num_heads=num_heads, heads2=heads2,
        qk_dim=qk_dim, rotary_dim=rotary_dim,
        stride_qkt=qk.stride(0), stride_qkh=qk.stride(1),
        factor_stride_t=factor1.stride(0), stride_out_t=out_q.stride(0),
        PAIRED=paired, KEY_OFFSET=key_offset, SHIFT_STORE=shift_store,
        COMPACT_FACTORS=compact_factors, FORCE_SIGN_BITS=qk.is_cuda,
        IMPLICIT_STATIONARY=implicit_stationary,
        BLOCK_M=block_m, BLOCK_D=triton.next_power_of_2(qk_dim), num_warps=num_warps,
    )


@_launch_qk_norm_rope.register_fake
def _launch_qk_norm_rope_fake(*args, **kwargs):
    # The caller allocates the outputs. The kernel changes only their contents.
    return None
"""
ROPE_BACKWARD_ARGUMENTS = "    BLOCK_QK: tl.constexpr,\n    BLOCK_V: tl.constexpr,\n"
ROPE_TRIM_ARGUMENTS = ROPE_BACKWARD_ARGUMENTS + "    TRIM_KEY_GRAD: tl.constexpr,\n" + ROPE_COMPACT_ARGUMENTS
ROPE_FIRST_FACTOR_LOAD = """    f1 = tl.load(factor1 + token[:, None] * factor_stride_t + factor_offset + offs_d[None, :],
                 mask=qk_mask, other=0.0).to(tl.float32)
"""
ROPE_FACTOR_LOAD = ROPE_FIRST_FACTOR_LOAD + """    f2_flip = tl.load(factor2 + token[:, None] * factor_stride_t + factor_offset + offs_flip[None, :],
                      mask=qk_mask, other=0.0).to(tl.float32)
"""
ROPE_VECTOR_FACTOR_LOAD = ROPE_FIRST_FACTOR_LOAD + """    if qk_dim == 64 or TRIM_KEY_GRAD:
        # Load contiguous BF16 pairs and reverse each pair in registers.
        f2 = tl.load(factor2 + token[:, None] * factor_stride_t + factor_offset + offs_d[None, :],
                     mask=qk_mask, other=0.0)
        f2_flip = tl.reshape(tl.flip(tl.reshape(f2, (BLOCK_T, BLOCK_QK // 2, 2)), dim=2),
                             (BLOCK_T, BLOCK_QK)).to(tl.float32)
    else:
        f2_flip = tl.load(factor2 + token[:, None] * factor_stride_t + factor_offset + offs_flip[None, :],
                          mask=qk_mask, other=0.0).to(tl.float32)
"""
ROPE_COMPACT_BACKWARD = """    if COMPACT_FACTORS:
        factor_rows = tl.full((BLOCK_T,), True, tl.int1) if IMPLICIT_STATIONARY and tokens % BLOCK_T == 0 else token_mask
        f1, f2_flip = _rope_pair_factors(factor1, factor2, token[:, None] * factor_stride_t + factor_offset // 2,
                                         factor_rows, qk_dim, BLOCK_QK, True, FORCE_SIGN_BITS, IMPLICIT_STATIONARY)
    else:
""" + indent(ROPE_VECTOR_FACTOR_LOAD, "    ")
ROPE_CURRENT_LOAD = "        grad = tl.load(grad_ptr, mask=qk_mask, other=0.0)\n"
ROPE_TRIM_CURRENT_LOAD = """        grad_mask = qk_mask
        if TRIM_KEY_GRAD and qk_kind == 1:
            grad_mask = qk_mask & ((offs_d[None, :] < rotary_dim) | (token[:, None] == 0))
        grad = tl.load(grad_ptr, mask=grad_mask, other=0.0)
"""
ROPE_SUCCESSOR_LOAD = """            next_token = tl.minimum(token + 1, tokens - 1)
            grad_next = tl.load(
                grad_k + next_token[:, None] * stride_grad_k_t + output_head * stride_grad_k_h + offs_d[None, :],
                mask=qk_mask & (token[:, None] < tokens - 1), other=0.0,
"""
ROPE_TRIM_SUCCESSOR_LOAD = """            next_token = tl.minimum(token + 1, tokens - 1)
            next_mask = qk_mask & (token[:, None] < tokens - 1)
            if TRIM_KEY_GRAD:
                next_mask = next_mask & (offs_d[None, :] >= rotary_dim)
            grad_next = tl.load(
                grad_k + next_token[:, None] * stride_grad_k_t + output_head * stride_grad_k_h + offs_d[None, :],
                mask=next_mask, other=0.0,
"""
ROPE_BACKWARD_LAUNCH = """        # Feature rows run BLOCK_T bytes; 32 keeps every transposed store on a full 32 B sector.
        block_t = 32
        _qkv_norm_rope_pack_fp8_backward_kernel[(num_heads, triton.cdiv(tokens, block_t))](
            grad_q, grad_k, grad_v, qk, factor1, factor2, grad_f8, grad_f8_t, grad_scale,
            tokens=tokens, num_heads=num_heads, qk_dim=qk_dim, rotary_dim=ctx.rotary_dim,
            v_dim=v_dim, qkv_dim=qkv_dim,
            stride_qkt=qk.stride(0), stride_qkh=qk.stride(1), factor_stride_t=factor1.stride(0),
            stride_grad_q_t=grad_q.stride(0), stride_grad_q_h=grad_q.stride(1),
            stride_grad_k_t=grad_k.stride(0), stride_grad_k_h=grad_k.stride(1),
            stride_grad_v_t=grad_v.stride(0), stride_grad_v_h=grad_v.stride(1),
            PAIRED=ctx.paired, KEY_OFFSET=ctx.key_offset, BLOCK_T=block_t,
            BLOCK_QK=triton.next_power_of_2(qk_dim), BLOCK_V=triton.next_power_of_2(v_dim),
            num_warps=4 if qk_dim <= 64 else 8,
        )
"""
ROPE_TRIM_LAUNCH = """        _launch_qkv_rope_backward(
            grad_q, grad_k, grad_v, qk, factor1, factor2, grad_f8, grad_f8_t, grad_scale,
            num_heads, ctx.rotary_dim, ctx.paired, ctx.key_offset,
        )
"""
ROPE_BWD_HELPER_ANCHOR = "def _scaled_mm_bf16"
ROPE_BWD_LAUNCHER = """@torch.library.custom_op("nanogpt::track1_qkv_rope_backward", mutates_args={"grad_f8", "grad_f8_t"})
def _launch_qkv_rope_backward(
    grad_q: torch.Tensor, grad_k: torch.Tensor, grad_v: torch.Tensor,
    qk: torch.Tensor, factor1: torch.Tensor, factor2: torch.Tensor,
    grad_f8: torch.Tensor, grad_f8_t: torch.Tensor, grad_scale: torch.Tensor,
    num_heads: int, rotary_dim: int, paired: bool, key_offset: bool,
) -> None:
    tokens, _, qk_dim = qk.shape
    v_dim, qkv_dim = grad_v.shape[2], grad_f8.shape[1]
    # Feature rows run BLOCK_T bytes; 32 keeps every transposed store on a full 32 B sector.
    block_t = 32
    implicit_stationary = qk_dim == 128 and not paired and factor1.shape[1] == 32
    compact_factors = implicit_stationary or (
        qk_dim in (64, 128) and factor1.shape[1] == (qk_dim * (2 if paired else 1)) // 2
    )
    trim_key_grad = (
        key_offset and not paired and rotary_dim == 64
        and num_heads == 6 and qk_dim == v_dim == 128
        and tokens in (16384, 32768, 40960, 49152)
        and qk.stride() == (2304, 128, 1)
        and factor1.stride() == factor2.stride() == (factor1.shape[1], 1)
        and all(t.stride() == (768, 128, 1) for t in (grad_q, grad_k, grad_v))
        and all(t.dtype == torch.bfloat16 for t in (qk, factor1, factor2, grad_q, grad_k, grad_v))
        and qk.is_cuda and torch.cuda.get_device_capability(qk.device)[0] == 9
        and grad_scale.dtype == torch.float32 and grad_scale.shape == (1,)
        and all(t.device == qk.device and t.data_ptr() % 16 == 0
                for t in (qk, factor1, factor2, grad_q, grad_k, grad_v, grad_f8, grad_f8_t, grad_scale))
    )
    _qkv_norm_rope_pack_fp8_backward_kernel[(num_heads, triton.cdiv(tokens, block_t))](
        grad_q, grad_k, grad_v, qk, factor1, factor2, grad_f8, grad_f8_t, grad_scale,
        tokens=tokens, num_heads=num_heads, qk_dim=qk_dim, rotary_dim=rotary_dim,
        v_dim=v_dim, qkv_dim=qkv_dim,
        stride_qkt=qk.stride(0), stride_qkh=qk.stride(1), factor_stride_t=factor1.stride(0),
        stride_grad_q_t=grad_q.stride(0), stride_grad_q_h=grad_q.stride(1),
        stride_grad_k_t=grad_k.stride(0), stride_grad_k_h=grad_k.stride(1),
        stride_grad_v_t=grad_v.stride(0), stride_grad_v_h=grad_v.stride(1),
        PAIRED=paired, KEY_OFFSET=key_offset, BLOCK_T=block_t,
        TRIM_KEY_GRAD=trim_key_grad,
        COMPACT_FACTORS=compact_factors, FORCE_SIGN_BITS=qk.is_cuda,
        IMPLICIT_STATIONARY=implicit_stationary,
        BLOCK_QK=triton.next_power_of_2(qk_dim), BLOCK_V=triton.next_power_of_2(v_dim),
        num_warps=4 if qk_dim <= 64 else 8,
    )


@_launch_qkv_rope_backward.register_fake
def _launch_qkv_rope_backward_fake(*args, **kwargs):
    return None


""" + ROPE_BWD_HELPER_ANCHOR
TRANSPOSE_IMPORT = "import triton.language as tl\n"
CANDIDATE_TRANSPOSE_IMPORT = (
    TRANSPOSE_IMPORT
    + "from track_1_short.perf.kernels.bf16_transpose import transpose_hopper_bf16\n"
    + "from track_1_short.perf.kernels.byte_transpose import transpose_hopper_bytes, transpose_hopper_ce_operands\n"
)
TRANSPOSE_COPY_DEF = "def transpose_copy(src: torch.Tensor, dst: torch.Tensor):"
TRANSPOSE_CUSTOM_COPY = """@torch.library.custom_op("nanogpt::track1_transpose_copy", mutates_args={"dst"})
def transpose_copy(src: torch.Tensor, dst: torch.Tensor) -> None:"""
TRANSPOSE_LAUNCH = "    BLOCK_M, BLOCK_N = 64, 128\n    if src.element_size() == 1 and dst.stride(1) == 1:\n"
CANDIDATE_TRANSPOSE_LAUNCH = (
    "    if transpose_hopper_bytes(src, dst, M, N) or transpose_hopper_bf16(src, dst, M, N, add=False):\n"
    "        return\n\n" + TRANSPOSE_LAUNCH
)
TRANSPOSE_ADD_LAUNCH = "    # 64x128 / 8 warps rather than 32x32 / 4 (record #360):"
BF16_TRANSPOSE_ADD_LAUNCH = ("    if transpose_hopper_bf16(src, dst, M, N, add=True):\n"
                             "        return\n\n" + TRANSPOSE_ADD_LAUNCH)

TRANSPOSE_ADD_SECTION = '# -----------------------------------------------------------------------------\n# Tiled transpose-add kernel:'
TRANSPOSE_COPY_FAKE = """@transpose_copy.register_fake
def _transpose_copy_fake(src, dst):
    return None


@torch.library.custom_op("nanogpt::track1_ce_transpose_operands", mutates_args={"x_t", "grad_t"})
def transpose_ce_operands(x: torch.Tensor, x_t: torch.Tensor, grad: torch.Tensor, grad_t: torch.Tensor) -> None:
    if not transpose_hopper_ce_operands(x, x_t, grad, grad_t):
        transpose_copy(x, x_t)
        transpose_copy(grad, grad_t)


@transpose_ce_operands.register_fake
def _transpose_ce_operands_fake(x, x_t, grad, grad_t):
    return None


""" + TRANSPOSE_ADD_SECTION
TRANSPOSE_SUBSTITUTIONS = (
    (TRANSPOSE_IMPORT, CANDIDATE_TRANSPOSE_IMPORT),
    (TRANSPOSE_COPY_DEF, TRANSPOSE_CUSTOM_COPY),
    (TRANSPOSE_LAUNCH, CANDIDATE_TRANSPOSE_LAUNCH),
    (TRANSPOSE_ADD_SECTION, TRANSPOSE_COPY_FAKE),
    (TRANSPOSE_ADD_LAUNCH, BF16_TRANSPOSE_ADD_LAUNCH),
)
CE_TRANSPOSE_SUBSTITUTIONS = (
    ("from track_1_short.perf.kernels.transpose import transpose_copy\n",
     "from track_1_short.perf.kernels.transpose import transpose_ce_operands\n"),
    ("""    transpose_copy(x_f8, x_f8_T)  # (768, n_rows) row-major

    grad_input_T = torch.empty((n_cols, n_rows), dtype=grad_input.dtype, device=grad_input.device)
    transpose_copy(grad_input, grad_input_T)  # (n_cols, n_rows) row-major
""",
     """    grad_input_T = torch.empty((n_cols, n_rows), dtype=grad_input.dtype, device=grad_input.device)
    transpose_ce_operands(x_f8, x_f8_T, grad_input, grad_input_T)
"""),
)
TRAINING_CE_SUBSTITUTIONS = (
    ("""struct __align__(8) __nv_uchar8 {
    unsigned char data[8];
    __device__ unsigned char& operator[](int i) { return data[i]; }
    __device__ const unsigned char& operator[](int i) const { return data[i]; }
};

__device__ __forceinline__ float ce_e4m3_logit_to_f32(unsigned char b) {
    unsigned int h2;
    unsigned short in = (unsigned short)b;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(in));
    unsigned short lo = (unsigned short)(h2 & 0xFFFFu);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(lo));
    return f;
}

#define CE_IN_T unsigned char
#define CE_IN8_T __nv_uchar8
#define CE_LOGIT_TO_F32(v) ce_e4m3_logit_to_f32(v)

""",
     """struct __align__(8) __nv_fp8_pairs {
    uint16_t data[4];
};

__device__ __forceinline__ void ce_decode_pair(uint16_t bytes, float &lo, float &hi) {
    asm("{ .reg .b32 halves; .reg .b16 low, high; "
        "cvt.rn.f16x2.e4m3x2 halves, %2; "
        "mov.b32 {low, high}, halves; "
        "cvt.f32.f16 %0, low; "
        "cvt.f32.f16 %1, high; }"
        : "=f"(lo), "=f"(hi) : "h"(bytes));
}

#define CE_IN_T unsigned char
#define CE_IN8_T __nv_fp8_pairs

"""),
    ("""      for (int k = 0; k < VEC_WIDTH; k++) {
        float tmp = CE_LOGIT_TO_F32(result[k]);
        tmp = sigmoid(tmp * inv_C + B_div_C);
        result_sigmoid[k] = __float2half(tmp);
      }
""",
     """      for (int k = 0; k < VEC_WIDTH; k += 2) {
        float lo, hi;
        ce_decode_pair(result.data[k / 2], lo, hi);
        __half2 sigma = __floats2half2_rn(sigmoid(lo * inv_C + B_div_C), sigmoid(hi * inv_C + B_div_C));
        result_sigmoid[k] = sigma.x;
        result_sigmoid[k + 1] = sigma.y;
      }
"""),
)
CE_PREDICTION_SUBSTITUTIONS = (
    ("SOFTCAP_A, SOFTCAP_B, SOFTCAP_C = 23.0, 5.0, 7.5\n",
     "SOFTCAP_A, SOFTCAP_B, SOFTCAP_C = 23.0, 5.0, 7.5\nLM_HEAD_GRAD_SCALE = (0.75 / 8) / 448\n"),
    ("    int n_predict,\n    double A_param,\n",
     "    int n_predict_argument,\n    double A_param,\n"),
    ("  constexpr int VEC_WIDTH = 8;\n",
     """#ifdef CE_FIXED_PREDICTIONS
  constexpr int n_predict = CE_FIXED_PREDICTIONS;
#else
  const int n_predict = n_predict_argument;
#endif
  constexpr int VEC_WIDTH = 8;
"""),
    ("  #pragma unroll 25\n", "  #pragma unroll CE_SIGMOID_UNROLL\n"),
    ("  #pragma unroll 4\n  for (int i = 0; i < NUM_LOADS; i++) {",
     "  #pragma unroll CE_GRADIENT_UNROLL\n  for (int i = 0; i < NUM_LOADS; i++) {"),
    ("__launch_bounds__(BLOCK_SIZE, 2)",
     "__launch_bounds__(BLOCK_SIZE, CE_MIN_BLOCKS_PER_SM)"),
    ("def compile_ce_kernel(vocab_size: int):",
     "def compile_ce_kernel(vocab_size: int, predictions: int | None = None):"),
    ('    kernel = torch.cuda._compile_kernel(\n        decls + CE_KERNEL_SOURCE,',
     """    if predictions is not None:
        if predictions not in (1, 2, 3):
            raise ValueError("Expected one to three CE predictions")
        decls += f"#define CE_FIXED_PREDICTIONS {predictions}\\n"
    use_logit_table = predictions in CE_FIXED_COUNTS.get(vocab_size, ())
    sigmoid_unroll, min_blocks = 25, 2
    if use_logit_table:
        sigmoid_unroll = 2
        min_blocks = 6 if vocab_size == CE_KERNEL_VOCAB_SIZE else 8
        decls += "#define CE_USE_LOGIT_TABLE\\n#define CE_DEFAULT_BC\\n"
        if vocab_size != CE_KERNEL_VOCAB_SIZE:
            decls += "#define CE_DEFAULT_A\\n"
        if vocab_size == 10240:
            decls += "#define CE_CACHE_MTP_TARGETS\\n"
    decls += f"#define CE_SIGMOID_UNROLL {sigmoid_unroll}\\n"
    decls += "#define CE_GRADIENT_UNROLL 4\\n"
    decls += f"#define CE_MIN_BLOCKS_PER_SM {min_blocks}\\n"
    kernel = torch.cuda._compile_kernel(
        decls + CE_KERNEL_SOURCE,"""),
    ("    kernel.set_shared_memory_config(vocab_size * 2)\n",
     "    kernel.set_shared_memory_config(vocab_size if use_logit_table else vocab_size * 2)\n"),
    ("ce_fwd_bwd_kernel = compile_ce_kernel(CE_KERNEL_VOCAB_SIZE)\n",
     """CE_FIXED_COUNTS = {10240: (2, 3), 14336: (1,), 24576: (1,), CE_KERNEL_VOCAB_SIZE: (1,)}

ce_fwd_bwd_kernel = compile_ce_kernel(CE_KERNEL_VOCAB_SIZE)
FULL_CE_FIXED_KERNELS = {n: compile_ce_kernel(CE_KERNEL_VOCAB_SIZE, n) for n in CE_FIXED_COUNTS[CE_KERNEL_VOCAB_SIZE]}
"""),
    ("    ce_fwd_bwd_kernel(\n",
     """    use_fixed_kernel = B == SOFTCAP_B and C == SOFTCAP_C and grad_s == LM_HEAD_GRAD_SCALE
    fixed_kernel = FULL_CE_FIXED_KERNELS.get(n_predict) if use_fixed_kernel else None
    kernel = ce_fwd_bwd_kernel if fixed_kernel is None else fixed_kernel
    kernel(
"""),
    ("        shared_mem=CE_KERNEL_VOCAB_SIZE * 2,",
     "        shared_mem=CE_KERNEL_VOCAB_SIZE * (2 if fixed_kernel is None else 1),"),
)
CE_ACTIVE_WEIGHT = """  // Total weight over active predictions at this position (used in the softmax-normalizer
  // gradient term). Include the prefix prediction only when it has a valid target.
  float S_w = 0.0f;

  for (int i = 0; i < n_predict; i++) {
    S_w += mtp_weights[i];
  }
  int64_t ptgt_row = prefix_targets[blockIdx.x];
  if (ptgt_row >= 0) {
    S_w += prefix_weight;
  }
"""
CE_LOGIT_TABLE_SUBSTITUTIONS = (
    ("  float A = (float)A_param;\n  float B = (float)B_param;\n  float C = (float)C_param;",
     """#ifdef CE_DEFAULT_A
  constexpr float A = 23.0f;
#else
  float A = (float)A_param;
#endif
#ifdef CE_DEFAULT_BC
  constexpr float B = 5.0f;
  constexpr float C = 7.5f;
#else
  float B = (float)B_param;
  float C = (float)C_param;
#endif"""),
    ("  float inv_C = 1 / C;",
     """#ifdef CE_DEFAULT_BC
  float inv_C;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv_C) : "f"(C));
#else
  float inv_C = 1 / C;
#endif"""),
    ('extern "C"\n__launch_bounds__(BLOCK_SIZE, CE_MIN_BLOCKS_PER_SM)',
     """#ifdef CE_USE_LOGIT_TABLE
__device__ __forceinline__ unsigned int ce_code_byte(uint2 codes, int i) {
  const unsigned int word = i < 4 ? codes.x : codes.y;
  return __byte_perm(word, 0u, 0x4440u | (i % 4));
}

#ifdef CE_DEFAULT_BC
__device__ __forceinline__ float ce_inverse_lm_head_grad_scale() {
  float result;
  // Keep the launch path's Hopper reciprocal in the lane-zero branch.
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(__int_as_float(0x395b6db7)));
  return result;
}
#endif

__device__ __forceinline__ float ce_apply_gradient_factor(float factor, float dz) {
  float result;
  asm("mul.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(factor), "f"(dz));
  return result;
}
#endif

extern "C"
__launch_bounds__(BLOCK_SIZE, CE_MIN_BLOCKS_PER_SM)"""),
    ("  extern __shared__ __half smem[];",
     """#ifdef CE_USE_LOGIT_TABLE
  struct __align__(8) LogitBytes {
    unsigned char data[8];
  };
  extern __shared__ __align__(8) unsigned char code_smem[];
  __shared__ float2 sigmoid_table[256];
  __shared__ float exp_table[256];
  __shared__ float gradient_factor;
#define CE_LOGIT_AT(i) sigmoid_table[code_smem[i]]
#define CE_Z_AT(i) __fmul_rn(CE_LOGIT_AT(i).x, A)
#else
  extern __shared__ __half smem[];
#define CE_Z_AT(i) (A * __half2float(smem[i]))
#endif"""),
    ("  float thread_sum = 0.0f;\n",
     """#ifdef CE_USE_LOGIT_TABLE
  // Each thread initializes one entry. Keep the same rounded B/C
  // product followed by the same fused multiply-add as the direct scan.
  float decoded, unused;
  ce_decode_pair((unsigned short)threadIdx.x, decoded, unused);
  const __half2 packed = __floats2half2_rn(sigmoid(__fmaf_rn(inv_C, decoded, B_div_C)), 0.0f);
  const __half sigma = packed.x;
  const float sigma_f32 = __half2float(sigma);
  const float scaled = __fmul_rn(sigma_f32, A);
#ifdef CE_DEFAULT_A
  float exp_arg;
  asm("sub.ftz.f32 %0, %1, %2;" : "=f"(exp_arg) : "f"(scaled), "f"(A));
  exp_table[threadIdx.x] = __expf(exp_arg);
#else
  exp_table[threadIdx.x] = __expf(scaled - A);
#endif
#ifdef CE_DEFAULT_BC
#define CE_FACTOR ce_apply_gradient_factor(inv_C * A, ce_inverse_lm_head_grad_scale())
#else
#define CE_FACTOR ((1.0f / C * A) * (1.0f / grad_s))
#endif
#ifdef CE_CACHE_MTP_TARGETS
  __shared__ int cached_targets[n_predict];
  if (threadIdx.x == 0) {
    gradient_factor = CE_FACTOR;
    #pragma unroll
    for (int i = 0; i < n_predict; ++i) {
      const int64_t index = (int64_t)blockIdx.x + i;
      const int64_t target = index < batch_size ? targets[index] : -1;
      cached_targets[i] = (unsigned long long)target < VOCAB_SIZE ? (int)target : -1;
    }
  }
#else
  if (threadIdx.x == 0) gradient_factor = CE_FACTOR;
#endif
#undef CE_FACTOR
  __syncthreads();
#endif
  float thread_sum = 0.0f;
"""),
    ("      CE_IN8_T result = *(CE_IN8_T*)(&block_logit_ptr[idx]);",
     """#ifdef CE_USE_LOGIT_TABLE
      const uint2 codes = *(const uint2*)(&block_logit_ptr[idx]);
      *(uint2*)(&code_smem[idx]) = codes;
      #pragma unroll
      for (int k = 0; k < VEC_WIDTH; k++) {
        thread_sum += exp_table[ce_code_byte(codes, k)];
      }
#else
      CE_IN8_T result = *(CE_IN8_T*)(&block_logit_ptr[idx]);"""),
    ("        thread_sum += __expf(tmp - A);\n      }",
     "        thread_sum += __expf(tmp - A);\n      }\n#endif"),
    ("  __shared__ float block_sums[NUM_WARPS];",
     """#ifdef CE_USE_LOGIT_TABLE
  __shared__ __align__(16) float block_sums[NUM_WARPS];
#else
  __shared__ float block_sums[NUM_WARPS];
#endif"""),
    ("""  float block_sum = 0.0f;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_sum += block_sums[i];
  }""",
     """#ifdef CE_USE_LOGIT_TABLE
  static_assert(NUM_WARPS == 8, "two groups of four warp sums");
  const unsigned long long warp_sums = __cvta_generic_to_shared(block_sums);
  float sums[NUM_WARPS];
  asm volatile ("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                : "=f"(sums[0]), "=f"(sums[1]), "=f"(sums[2]), "=f"(sums[3]) : "l"(warp_sums));
  asm volatile ("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4+16];"
                : "=f"(sums[4]), "=f"(sums[5]), "=f"(sums[6]), "=f"(sums[7]) : "l"(warp_sums));
  // Keep the operand order of the original ordered FP32 sum.
  float block_sum;
  asm ("add.ftz.f32 %0, %1, %2;" : "=f"(block_sum) : "f"(sums[0]), "f"(0.0f));
  #pragma unroll
  for (int i = 1; i < NUM_WARPS; i++) {
    asm ("add.ftz.f32 %0, %1, %2;" : "=f"(block_sum) : "f"(block_sum), "f"(sums[i]));
  }
#else
  float block_sum = 0.0f;
  for (int i = 0; i < NUM_WARPS; i++) {
    block_sum += block_sums[i];
  }
#endif"""),
    (CE_ACTIVE_WEIGHT, "#ifndef CE_USE_LOGIT_TABLE\n" + CE_ACTIVE_WEIGHT + "#endif\n"),
    ("  float lse = A + __logf(block_sum);",
     """  float lse = A + __logf(block_sum);
#ifdef CE_USE_LOGIT_TABLE
""" + CE_ACTIVE_WEIGHT + """
  // Publish the lane's rounded sigmoid with its row probability.
  float probability = __expf(scaled - lse);
  sigmoid_table[threadIdx.x] = make_float2(sigma_f32, probability);
  float normalizer = __fmul_rn(S_w, probability);
  float g = ce_apply_gradient_factor(gradient_factor, normalizer - 0.0f)
      * sigma_f32 * (1.0f - sigma_f32);

  // All lanes finished reading the exponential table at the sum barrier.
  // Reuse its first 256 bytes for the rounded base gradients.
  unsigned char* const base_cache = reinterpret_cast<unsigned char*>(exp_table);
  base_cache[threadIdx.x] = f32_to_fp8_e5m2_fast(g);
  __syncthreads();
#endif"""),
    ("""      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size) {
        float weight = mtp_weights[k];
        int64_t target = targets[target_idx];
        if (target >= 0 && target < VOCAB_SIZE) {
          float z_target = A * __half2float(smem[target]);
          total_loss += weight * (lse - z_target);
        }
      }""",
     """#if defined(CE_USE_LOGIT_TABLE) && defined(CE_CACHE_MTP_TARGETS)
      int target = cached_targets[k];
      if (target >= 0 && target < VOCAB_SIZE) {
        float weight = mtp_weights[k];
        float z_target = CE_Z_AT(target);
        total_loss += weight * (lse - z_target);
      }
#else
      int64_t target_idx = blockIdx.x + k;
      if (target_idx < batch_size) {
        float weight = mtp_weights[k];
        int64_t target = targets[target_idx];
        if (target >= 0 && target < VOCAB_SIZE) {
          float z_target = CE_Z_AT(target);
          total_loss += weight * (lse - z_target);
        }
      }
#endif"""),
    ("        float z_p = A * __half2float(smem[ptgt]);",
     "        float z_p = CE_Z_AT(ptgt);"),
    ("      __half8 sigmoid_us = *(__half8*)(&smem[idx]);",
     """#ifdef CE_USE_LOGIT_TABLE
      const uint2 codes = *(const uint2*)(&code_smem[idx]);
      LogitBytes result_bytes;
      #pragma unroll
      for (int j = 0; j < VEC_WIDTH; j++) {
        result_bytes.data[j] = base_cache[ce_code_byte(codes, j)];
      }
      *(LogitBytes*)(&grad_input[blockIdx.x * VOCAB_SIZE + idx]) = result_bytes;
#else
      __half8 sigmoid_us = *(__half8*)(&smem[idx]);"""),
    ("      *(unsigned long long*)(&grad_input[blockIdx.x * VOCAB_SIZE + idx]) = packed8;",
     "      *(unsigned long long*)(&grad_input[blockIdx.x * VOCAB_SIZE + idx]) = packed8;\n#endif"),
    ("  __syncthreads();\n\n  // Sparse correction for target columns.",
     """#ifdef CE_USE_LOGIT_TABLE
  int64_t corrections[n_predict];
  #pragma unroll
  for (int slot = 0; slot < n_predict; ++slot) {
#ifdef CE_CACHE_MTP_TARGETS
    corrections[slot] = cached_targets[slot];
#else
    const int64_t index = (int64_t)blockIdx.x + slot;
    corrections[slot] = index < batch_size ? targets[index] : -1;
#endif
  }

  // The lane that wrote the eight-byte gradient chunk also patches its
  // targets. Keep the first occurrence of each target as its writer.
  #pragma unroll
  for (int slot = 0; slot < n_predict; ++slot) {
    const int64_t target = corrections[slot];
    if (target < 0 || target >= VOCAB_SIZE ||
        (((unsigned)target / VEC_WIDTH) % BLOCK_SIZE != threadIdx.x)) continue;

    bool first = true;
    #pragma unroll
    for (int prior = 0; prior < slot; ++prior)
      first = first && corrections[prior] != target;
    if (!first) continue;

    const float2 entry = CE_LOGIT_AT(target);
    float sigmoid_u = entry.x;
    float p = entry.y;
    float term1 = S_w * p;
    float term2 = 0.0f;
    #pragma unroll
    for (int j = 0; j < n_predict; ++j)
      if (corrections[j] == target) term2 += mtp_weights[j];
    if (ptgt_row == target) term2 += prefix_weight;
    float grad_z = term1 - term2;
    float grad_x = ce_apply_gradient_factor(gradient_factor, grad_z) * sigmoid_u * (1.0f - sigmoid_u);
    grad_input[blockIdx.x * VOCAB_SIZE + target] = f32_to_fp8_e5m2_fast(grad_x);
  }
  if (prefix_weight > 0.0f && (unsigned long long)ptgt_row < VOCAB_SIZE &&
        (((unsigned)ptgt_row / VEC_WIDTH) % BLOCK_SIZE == threadIdx.x)) {
    bool first = true;
    #pragma unroll
    for (int prior = 0; prior < n_predict; ++prior)
      first = first && corrections[prior] != ptgt_row;
    if (first) {
      const float2 entry = CE_LOGIT_AT(ptgt_row);
      float term2 = prefix_weight + 0.0f;
      float grad_z = S_w * entry.y - term2;
      float grad_x = ce_apply_gradient_factor(gradient_factor, grad_z) * entry.x * (1.0f - entry.x);
      grad_input[blockIdx.x * VOCAB_SIZE + ptgt_row] = f32_to_fp8_e5m2_fast(grad_x);
    }
  }
#else
  __syncthreads();

  // Sparse correction for target columns."""),
    ("    grad_input[blockIdx.x * VOCAB_SIZE + target] = result;\n  }",
     "    grad_input[blockIdx.x * VOCAB_SIZE + target] = result;\n  }\n#endif\n#undef CE_Z_AT\n#undef CE_LOGIT_AT"),
)
SAMPLED_CE_PREDICTION_SUBSTITUTIONS = (
    ("    CE_KERNEL_BLOCK_SIZE,\n    SOFTCAP_A,\n",
     "    CE_FIXED_COUNTS,\n    CE_KERNEL_BLOCK_SIZE,\n    LM_HEAD_GRAD_SCALE,\n    SOFTCAP_A,\n"),
    ("SAMPLED_CE_KERNELS = {p: compile_ce_kernel(p) for p in ALL_CANDIDATE_COUNTS}\n",
     """SAMPLED_CE_KERNELS = {p: compile_ce_kernel(p) for p in ALL_CANDIDATE_COUNTS}
SAMPLED_CE_FIXED_KERNELS = {
    (p, n): compile_ce_kernel(p, n)
    for p in ALL_CANDIDATE_COUNTS
    for n in CE_FIXED_COUNTS.get(p, ())
}
"""),
    ("    SAMPLED_CE_KERNELS[P](\n",
     """    use_fixed_kernel = (A == SOFTCAP_A and B == SOFTCAP_B and C == SOFTCAP_C
                        and grad_s == LM_HEAD_GRAD_SCALE)
    fixed_kernel = SAMPLED_CE_FIXED_KERNELS.get((P, n_predict)) if use_fixed_kernel else None
    kernel = SAMPLED_CE_KERNELS[P] if fixed_kernel is None else fixed_kernel
    kernel(
"""),
    ("        shared_mem=P * 2,",
     "        shared_mem=P * (2 if fixed_kernel is None else 1),"),
)
HEAD_IMPORT = """from track_1_short.perf.kernels.cross_entropy import (
    FusedSoftcappedCrossEntropy,
    lm_head_inverse_scale,
    quantize_lm_head_dual,
)
"""
HEAD_REFRESH_IMPORT = """from track_1_short.perf.kernels.cross_entropy import (
    FusedSoftcappedCrossEntropy,
    lm_head_inverse_scale,
)
from track_1_short.perf.kernels.lm_head_refresh import quantize_lm_head_dual
"""
IMPORT = ("from track_1_short.perf.kernels.mlp_scale_refresh import refresh_mlp_scalars\n"
          "from track_1_short.perf.kernels.mlp_weights_refresh import refresh_mlp_weight_bank\n")
MLP_WEIGHT_CALL = """        quantize_mlp_weights_dual(up_weights, self._mlp_up_scales, self._mlp_up_partial_amax,
                                  row=self._mlp_up_f8, col_t=self._mlp_up_f8_col.transpose(1, 2), update_scales=not exact)
        quantize_mlp_weights_dual(down_weights, self._mlp_down_scales, self._mlp_down_partial_amax,
                                  row=self._mlp_down_f8_row, col_t=self._mlp_down_f8.transpose(1, 2), update_scales=not exact)"""
MLP_WEIGHT_GROUPED = """        if not refresh_mlp_weight_bank(
            self.mlp_bank,
            (self._mlp_up_scales, self._mlp_up_partial_amax, self._mlp_up_f8, self._mlp_up_f8_col.transpose(1, 2)),
            (self._mlp_down_scales, self._mlp_down_partial_amax, self._mlp_down_f8_row, self._mlp_down_f8.transpose(1, 2)),
            delayed=not exact,
        ):
""" + "\n".join("    " + line for line in MLP_WEIGHT_CALL.splitlines())
QKV_IMPORT = "from track_1_short.perf.kernels.fp8_attention_quant import PackedQKVFP8Cache"
NATIVE_IMPORT = QKV_IMPORT + ", refresh_grouped_qkv, refresh_native_qkv"
QKV_LOOP = """            v_heads = self.vo_bank[:2 * num_slots].view(num_slots, 2, H, head_dim, dim)[:, 0]
            start = 0"""
GROUPED_LOOP = """            v_heads = self.vo_bank[:2 * num_slots].view(num_slots, 2, H, head_dim, dim)[:, 0]
            if refresh_grouped_qkv(self.attn_fp8, qk_heads, v_heads):
                return
            start = 0"""
QKV_VIEW = "                qk = qk_heads[slots, :, :qk_dim].reshape(len(layers), 2 * H * qk_dim, dim)"
NATIVE_VIEW = """                if refresh_native_qkv(
                    self.attn_fp8[group], qk_heads[slots], v_heads[slots], qk_dim, v_dim
                ):
                    continue
""" + QKV_VIEW
CALL = """        refresh_mlp_scalars(
            self._mlp_up_scales, self._mlp_down_scales,
            self._mlp_post_amax, self._mlp_post_scales,
            self._mlp_dpre_amax, self._mlp_dpre_scales,
            self.post_lambdas, self._mlp_fold_p,
            self._mlp_up_dequant, self._mlp_dq_bwd,
            self._mlp_down_scales_folded, self._mlp_dq_bwd_folded,
            parallel_slot=PARALLEL_MLP_SLOT, parallel_layer=PARALLEL_MLP_LAYER,
            x_scale=FP8_MLP_X_SCALE, grad_scale=FP8_GRAD_SCALE,
            post_gain=FP8_POST_HEADROOM / e4m3_max, dpre_gain=FP8_DPRE_HEADROOM / e5m2_max,
        )
"""


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(checkout), *args), text=True)


def replacement(source: str) -> str:
    tree = ast.parse(source)
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GPT")
    refresh = next(node for node in model.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_quantize_mlp_weights_and_scales")
    first = next(node for node in refresh.body if isinstance(node, ast.Expr)
                 and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "torch.mul")
    lines = source.splitlines(keepends=True)
    lines[first.lineno - 1:refresh.end_lineno] = [CALL]
    source = "".join(lines)
    anchor = "from track_1_short.perf.kernels.ngram_embed import ngram_embedding\n"
    if source.count(anchor) != 1:
        raise ValueError("Expected exactly one n-gram kernel import in the pinned source")
    source = source.replace(anchor, IMPORT + anchor)
    for before, after in ((QKV_IMPORT, NATIVE_IMPORT), (QKV_LOOP, GROUPED_LOOP), (QKV_VIEW, NATIVE_VIEW),
                          (MLP_WEIGHT_CALL, MLP_WEIGHT_GROUPED), (HEAD_IMPORT, HEAD_REFRESH_IMPORT),
                          (WIDE_YARN, STATIONARY_YARN)):
        if source.count(before) != 1:
            raise ValueError(f"Expected one model substitution site: {before}")
        source = source.replace(before, after)
    ast.parse(source)
    return source


def replace_ngram(source: str) -> str:
    path = Path(__file__).with_name("ngram_tiles.py")
    candidate = path.read_text()
    names = ("_ngram_embed_fwd_kernel", "_ngram_embed_bwd_kernel")
    originals = {node.name: node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names}
    replacements = {node.name: node for node in ast.parse(candidate).body
                    if isinstance(node, ast.FunctionDef) and node.name in names}
    edits = []
    for name in names:
        a, b = originals[name], replacements[name]
        body = candidate.splitlines(keepends=True)[b.decorator_list[0].lineno - 1:b.end_lineno]
        edits.append((a.decorator_list[0].lineno - 1, a.end_lineno, body))
    lines = source.splitlines(keepends=True)
    for start, stop, body in sorted(edits, reverse=True):
        lines[start:stop] = body
    source = "".join(lines)
    changes = {
        "XBLOCK, NUM_WARPS = 1024, 4": "TOKEN_TILE, CHANNEL_TILE, NUM_WARPS = 4, 256, 4",
        "    n = inp.shape[0] * d": "    n = inp.shape[0]",
        "(triton.cdiv(n, XBLOCK),)": "(triton.cdiv(n, TOKEN_TILE), triton.cdiv(d, CHANNEL_TILE))",
        "pool, inp, out, inp.shape[0], n, D=d": "pool, inp, out, n, D=d",
        "XBLOCK=XBLOCK, num_warps": "BT=TOKEN_TILE, BC=CHANNEL_TILE, FULL_ROWS=full_rows, num_warps",
        "_ngram_embed_fwd_kernel[grid](": """full_rows = _embeds_full_rows(d, n, rmask, inp, cache, slots, pool, out)
    _ngram_embed_fwd_kernel[(grid[0],) if full_rows else grid](""",
        "_ngram_embed_bwd_kernel[grid](": """full_rows = _embeds_full_rows(d, n, rmask, inp, g, pool, out)
    _ngram_embed_bwd_kernel[(grid[0],) if full_rows else grid](""",
        "def _launch_shape(pool: torch.Tensor": """def _embeds_full_rows(d, n, rmask, inp, *tensors):
    # The full-width launch has no token or channel tail. T is also aligned
    # so the second slot plane specializes to the same vector loads.
    return (d == 768 and 0 < n <= 262144 and n % 16 == 0 and rmask == 8191 and inp.is_cuda
            and torch.cuda.get_device_capability(inp.device) == (9, 0)
            and inp.data_ptr() % 16 == 0
            and all(t.device == inp.device and t.data_ptr() % 16 == 0 for t in tensors))


def _launch_shape(pool: torch.Tensor""",
    }
    for original, updated in changes.items():
        count = 2 if original.startswith("XBLOCK=XBLOCK") else 1
        if source.count(original) != count:
            raise ValueError(f"Upstream launch differs at {original!r}")
        source = source.replace(original, updated)
    ast.parse(source)
    return source


def replace_qkv_amax(source: str) -> str:
    anchor = "def quantize_dual_layout_packed_batched("
    launch = "    _packed_batched_matrix_amax_partial_kernel[(B, num_parts)]("
    dispatch = """    scan = _packed_batched_matrix_amax_partial_kernel
    if num_parts == 32 and first_rows * N % 2048 == 0 and second_rows * N % 2048 == 0:
        scan = _aligned_packed_amax
    scan[(B, num_parts)]("""
    if source.count(anchor) != 1 or source.count(launch) != 1:
        raise ValueError("The upstream QKV refresh launcher does not match the pinned source")
    code = []
    for filename in ("qkv_amax.py", "qkv_scale.py", "qkv_pack.py", "qkv_pack_gluon.py",
                     "qkv_native.py", "qkv_grouped.py"):
        kernel = Path(__file__).with_name(filename).read_text()
        definition = next(node for node in ast.parse(kernel).body if isinstance(node, ast.FunctionDef))
        start = (definition.decorator_list or [definition])[0].lineno
        code.append("".join(kernel.splitlines(keepends=True)[start - 1:]).rstrip())
    scale = "    torch.mul(partial_amax.amax(dim=1).clamp_(min=1.0e-12), 1.0 / 448.0, out=scales)"
    fused_scale = """    partial_start, scale_start = partial_amax.data_ptr(), scales.data_ptr()
    separate_scale = (partial_start + partial_amax.nbytes <= scale_start
                      or scale_start + scales.nbytes <= partial_start)
    if (num_parts == 32 and partial_amax.dtype == scales.dtype == torch.float32
            and partial_amax.is_contiguous() and scales.is_contiguous()
            and partial_amax.device == scales.device and separate_scale and B > 0):
        _qkv_scale_from_partials[(B,)](partial_amax, scales, num_warps=1)
    else:
    """ + scale
    if source.count(scale) != 1:
        raise ValueError("The upstream QKV scale calculation does not match the pinned source")
    pack_launch = "    _quantize_dual_layout_packed_batched_kernel[grid]("
    pack_dispatch = """    pack = _quantize_dual_layout_packed_batched_kernel
    max_pitch = max(first.stride(0), second.stride(0), total_rows * N)
    if (first_rows % block_m == 0 and total_rows % block_m == 0 and N % block_n == 0
            and B * max_pitch < (1 << 31)):
        if first.is_cuda and torch.cuda.get_device_capability(first.device)[0] == 9:
            pack = _quantize_qkv_tiles_gluon
        else:
            pack = _quantize_aligned_qkv_tiles
    pack[grid]("""
    if source.count(pack_launch) != 1:
        raise ValueError("The upstream QKV pack launcher does not match the pinned source")
    import_anchor = "import triton.language as tl\n"
    if source.count(import_anchor) != 1:
        raise ValueError("The upstream Triton import does not match the pinned source")
    gluon_imports = ("from triton.experimental import gluon\n"
                     "from triton.experimental.gluon import language as gl\n")
    result = (source.replace(import_anchor, import_anchor + gluon_imports)
              .replace(anchor, "\n\n\n".join(code) + "\n\n\n" + anchor)
              .replace(launch, dispatch).replace(scale, fused_scale).replace(pack_launch, pack_dispatch))
    ast.parse(result)
    return result


def replace_sampled(source: str) -> str:
    for original, candidate in ((SAMPLED_IMPORT, GATHER_IMPORT), (SAMPLED_COPY, GATHER_COPY)):
        if source.count(original) != 1:
            raise ValueError(f"Expected one sampled-softmax substitution site: {original}")
        source = source.replace(original, candidate)
    ast.parse(source)
    return source


def replace_transpose(source: str) -> str:
    for original, candidate in TRANSPOSE_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError(f"Expected one transpose substitution site: {original}")
        source = source.replace(original, candidate)
    return source


CE_ACTIVATION_CALL = ("        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)\n",
                      "        x_f8 = quantize_activation(x, x_s)\n")
CE_ACTIVATION_IMPORT = ("import triton.language as tl\n",
                        "import triton.language as tl\n\n"
                        "from track_1_short.perf.kernels.activation_fp8 import quantize_activation\n")
CE_ACTIVATION_SUBSTITUTIONS = (CE_ACTIVATION_IMPORT, CE_ACTIVATION_CALL)


CE_LOGIT_REUSE_SUBSTITUTIONS = (
    ('    const CE_IN_T* __restrict__ logits,',
     """#ifdef CE_USE_LOGIT_TABLE
    const CE_IN_T* logits,
#else
    const CE_IN_T* __restrict__ logits,
#endif"""),
    ("""FULL_CE_FIXED_KERNELS = {n: compile_ce_kernel(CE_KERNEL_VOCAB_SIZE, n) for n in CE_FIXED_COUNTS[CE_KERNEL_VOCAB_SIZE]}""",
     """CE_FIXED_KERNELS = {
    (width, n): compile_ce_kernel(width, n)
    for width, predictions in CE_FIXED_COUNTS.items()
    for n in predictions
}
FULL_CE_FIXED_KERNELS = {n: CE_FIXED_KERNELS[CE_KERNEL_VOCAB_SIZE, n] for n in CE_FIXED_COUNTS[CE_KERNEL_VOCAB_SIZE]}"""),
    ('def _ce_logit_gemm(',
     """@torch.library.custom_op("nanogpt::ce_reuse_logits", mutates_args={"logits", "losses"})
def ce_reuse_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mtp_weights: torch.Tensor,
    prefix_targets: torch.Tensor,
    prefix_weight: torch.Tensor,
    losses: torch.Tensor,
) -> None:
    n_rows, width = logits.shape
    predictions = mtp_weights.shape[0]
    CE_FIXED_KERNELS[width, predictions](
        (n_rows, 1, 1),
        (CE_KERNEL_BLOCK_SIZE, 1, 1),
        (logits, targets, mtp_weights, prefix_targets, prefix_weight, losses, logits,
         n_rows, predictions, SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, LM_HEAD_GRAD_SCALE),
        shared_mem=width,
    )


def _ce_forward_buffers(logits, targets, mtp_weights, prefix_targets, prefix_weight, grad_s, fallback):
    logits = logits.contiguous()
    mtp_weights = mtp_weights.contiguous()
    prefix_weight = prefix_weight.reshape(1).to(torch.float32).contiguous()
    n_rows, width = logits.shape
    predictions = mtp_weights.shape[0]
    losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
    if (grad_s == LM_HEAD_GRAD_SCALE and (width, predictions) in CE_FIXED_KERNELS
            and n_rows * width < 2**31):
        # All tiles copy their own logits into shared memory before the first gradient store.
        ce_reuse_logits(logits, targets, mtp_weights, prefix_targets, prefix_weight, losses)
        grad_input = logits.view(torch.float8_e5m2)
    else:
        grad_input = torch.empty((n_rows, width), dtype=torch.float8_e5m2, device=logits.device)
        fallback(logits, targets, mtp_weights, prefix_targets, prefix_weight, losses, grad_input,
                 n_rows, predictions, SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, grad_s)
    return losses, grad_input


def _ce_logit_gemm("""),
    ("""        n_rows, n_cols = logits.shape
        n_predict = mtp_weights.shape[0]
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        grad_input = torch.empty((n_rows, n_cols), dtype=torch.float8_e5m2, device=logits.device)
        ce_fwd_bwd(logits.contiguous(), targets.contiguous(), mtp_weights.contiguous(),
                   prefix_targets.contiguous(), prefix_weight.reshape(1).to(torch.float32).contiguous(),
                   losses, grad_input, n_rows, n_predict, SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, grad_s)

""",
     """        losses, grad_input = _ce_forward_buffers(
            logits, targets.contiguous(), mtp_weights, prefix_targets.contiguous(), prefix_weight, grad_s, ce_fwd_bwd,
        )

"""),
)

SAMPLED_CE_LOGIT_REUSE_SUBSTITUTIONS = (
    ("""    CE_FIXED_COUNTS,
""",
     """    CE_FIXED_COUNTS,
    CE_FIXED_KERNELS,
"""),
    ("""    _ce_logit_gemm,
""",
     """    _ce_logit_gemm,
    _ce_forward_buffers,
"""),
    ("""SAMPLED_CE_FIXED_KERNELS = {
    (p, n): compile_ce_kernel(p, n)
    for p in ALL_CANDIDATE_COUNTS
    for n in CE_FIXED_COUNTS.get(p, ())
}""",
     """SAMPLED_CE_FIXED_KERNELS = {
    (p, n): CE_FIXED_KERNELS[p, n]
    for p in ALL_CANDIDATE_COUNTS
    for n in CE_FIXED_COUNTS.get(p, ())
}"""),
    ("""        n_rows = x.shape[0]
        P = rows.shape[0]
        x_f8 = quantize_activation(x, x_s)
""",
     "        x_f8 = quantize_activation(x, x_s)\n"),
    ("""        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        grad_input = torch.empty((n_rows, P), dtype=torch.float8_e5m2, device=logits.device)
        sampled_ce_fwd_bwd(logits.contiguous(), target_pos, mtp_weights.contiguous(), prefix_pos,
                           prefix_weight.reshape(1).to(torch.float32).contiguous(), losses, grad_input,
                           n_rows, mtp_weights.shape[0], SOFTCAP_A, SOFTCAP_B, SOFTCAP_C, grad_s)
""",
     """        losses, grad_input = _ce_forward_buffers(
            logits, target_pos, mtp_weights, prefix_pos, prefix_weight, grad_s, sampled_ce_fwd_bwd,
        )
"""),
)


def replace_cross_entropy(source: str) -> str:
    for original, candidate in (
        CE_TRANSPOSE_SUBSTITUTIONS + TRAINING_CE_SUBSTITUTIONS + CE_PREDICTION_SUBSTITUTIONS
        + CE_LOGIT_TABLE_SUBSTITUTIONS + CE_ACTIVATION_SUBSTITUTIONS + CE_LOGIT_REUSE_SUBSTITUTIONS
    ):
        if source.count(original) != 1:
            raise ValueError(f"Expected one cross-entropy substitution site: {original}")
        source = source.replace(original, candidate)
    return source


def replace_densify(source: str) -> str:
    substitutions = ((SAMPLED_CE_IMPORT, DENSIFY_IMPORT), (DENSIFY_LAUNCH, DENSIFY_DISPATCH))
    for original, replacement in (substitutions + SAMPLED_CE_PREDICTION_SUBSTITUTIONS
                                  + CE_ACTIVATION_SUBSTITUTIONS + SAMPLED_CE_LOGIT_REUSE_SUBSTITUTIONS):
        if source.count(original) != 1:
            raise ValueError(f"Expected one sampled gradient substitution site: {original}")
        source = source.replace(original, replacement)
    return source


ROPE_SUBSTITUTIONS = (
    (ROPE_KERNEL_ANCHOR, ROPE_PAIR_HELPER),
    (ROPE_KEY_GUARD, ROPE_PREVIOUS_GUARD),
    (ROPE_BLOCK_ARGUMENTS, ROPE_SHIFT_ARGUMENTS),
    (ROPE_FORWARD_FACTORS, ROPE_COMPACT_FORWARD),
    (ROPE_FACTOR_SHAPE, ROPE_COMPACT_SHAPE),
    (ROPE_KEY_STORE, ROPE_SHIFT_STORE),
    (ROPE_LAUNCH, ROPE_SHIFT_LAUNCH),
    (ROPE_BACKWARD_ARGUMENTS, ROPE_TRIM_ARGUMENTS),
    (ROPE_FACTOR_LOAD, ROPE_COMPACT_BACKWARD),
    (ROPE_CURRENT_LOAD, ROPE_TRIM_CURRENT_LOAD),
    (ROPE_SUCCESSOR_LOAD, ROPE_TRIM_SUCCESSOR_LOAD),
    (ROPE_BACKWARD_LAUNCH, ROPE_TRIM_LAUNCH),
    (ROPE_BWD_HELPER_ANCHOR, ROPE_BWD_LAUNCHER),
)


def replace_rope(source: str) -> str:
    for original, replacement in ROPE_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError(f"Expected one shifted key substitution site: {original}")
        source = source.replace(original, replacement)
    return source


YARN_SUBSTITUTIONS = (
    ("    def __init__(self, head_dim, max_seq_len, paired=False, *, attn_scale: float, device):\n",
     "    def __init__(self, head_dim, max_seq_len, paired=False, *, attn_scale: float, device,\n"
     "                 elide_stationary=False):\n"),
    ("        width = head_dim if not paired else 2 * head_dim\n",
     "        # Each adjacent cosine/sine pair shares one frequency.\n"
     "        self.compact = head_dim in (64, 128)\n"
     "        self.elide_stationary = elide_stationary and not paired and head_dim == 128\n"
     "        width = head_dim if not paired else 2 * head_dim\n"
     "        if self.compact:\n"
     "            width //= 2\n"
     "        if self.elide_stationary:\n"
     "            width = self.ROTARY_DIM // 2\n"),
    ("        angular_freq = angular_freq.repeat_interleave(2)\n"
     "        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)\n"
     "        self.angular_freq = torch.cat([\n"
     "            angular_freq, angular_freq.new_zeros(self.head_dim - self.ROTARY_DIM)\n"
     "        ])\n",
     "        if self.elide_stationary:\n"
     "            self.angular_freq = angular_freq\n"
     "        else:\n"
     "            padding = self.head_dim - self.ROTARY_DIM\n"
     "            if self.compact:\n"
     "                padding //= 2\n"
     "            else:\n"
     "                angular_freq = angular_freq.repeat_interleave(2)\n"
     "            self.angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(padding)])\n"),
    ("        self.factor2[lo:hi, 1::2] *= -1\n        self.valid_rows = hi\n",
     "        # The consumers supply the sine sign when expanding adjacent channels.\n"
     "        if not self.compact:\n"
     "            self.factor2[lo:hi, 1::2] *= -1\n"
     "        self.valid_rows = hi\n"),
)


def replace_yarn(source: str) -> str:
    for original, replacement in YARN_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError(f"Expected one YaRN substitution site: {original}")
        source = source.replace(original, replacement)
    return source


def wide_window_sizes(source: str) -> list[int]:
    assignments = {node.targets[0].id: node.value for node in ast.parse(source).body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    block_size = ast.literal_eval(assignments["BLOCK_SIZE"])
    window_sizes = {
        ast.literal_eval(item.value)[1]
        for stage in assignments["TRAINING_STAGES"].elts
        for item in stage.keywords if item.arg == "window_sizes"
    }
    window_sizes.add(ast.literal_eval(assignments["WS_POST_YARN_EXT"]))
    return [blocks * block_size for blocks in sorted(window_sizes)]


def replace_tail(source: str) -> str:
    for original, replacement in TAIL_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized tail averaging source")
        source = source.replace(original, replacement, 1)
    return source


def replace_value_embed(source: str) -> str:
    for original, replacement in VALUE_EMBED_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized value-embedding backward dispatch")
        source = source.replace(original, replacement, 1)
    return source


def replace_value_pull(source: str) -> str:
    for original, replacement in VALUE_PULL_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized value embedding gradient exchange")
        source = source.replace(original, replacement, 1)
    return source


def replace_ngram_table(source: str) -> str:
    for original, replacement in NGRAM_TABLE_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized n-gram row exchange")
        source = source.replace(original, replacement, 1)
    return source


def replace_fineweb_data(source: str) -> str:
    for original, replacement in FINEWEB_DATA_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized FineWeb batch source")
        source = source.replace(original, replacement, 1)
    return source


def replace_schedule(source: str) -> str:
    for original, replacement in SCHEDULE_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized schedule weight construction")
        source = source.replace(original, replacement, 1)
    return source


def replace_ngram_adam(source: str) -> str:
    for original, replacement in NGRAM_ADAM_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized n-gram Adam update")
        source = source.replace(original, replacement, 1)
    return source


def replace_row_scatter(source: str) -> str:
    for original, replacement in ROW_SCATTER_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized row scatter")
        source = source.replace(original, replacement, 1)
    return source


def replace_row_owners(source: str) -> str:
    for original, replacement in ROW_OWNER_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized peer row request")
        source = source.replace(original, replacement, 1)
    return source


def replace_step_graphs(source: str) -> str:
    for original, replacement in STEP_GRAPH_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized forward graph input staging")
        source = source.replace(original, replacement, 1)
    return source


def replace_driver(source: str) -> str:
    for original, replacement in DRIVER_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized training graph call")
        source = source.replace(original, replacement, 1)
    return source


def replace_row_prefetch(source: str) -> str:
    for original, replacement in ROW_PREFETCH_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized n-gram directory preparation")
        source = source.replace(original, replacement, 1)
    return source


def replace_pinned_batches(source: str) -> str:
    for original, replacement in PINNED_BATCH_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized pinned batch staging")
        source = source.replace(original, replacement, 1)
    return source


def replace_replicated_adam(source: str) -> str:
    for original, replacement in REPLICATED_ADAM_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized replicated Adam update")
        source = source.replace(original, replacement, 1)
    return source


def replace_anvil(source: str) -> str:
    for original, replacement in ANVIL_EPILOGUE_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized ANVIL epilogue")
        source = source.replace(original, replacement, 1)
    for original, replacement in ANVIL_TRACE_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized ANVIL Gram normalization")
        source = source.replace(original, replacement)
    for original, replacement in ANVIL_RAILS_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized ANVIL twin-rail preparation")
        source = source.replace(original, replacement, 1)
    for original, replacement in ANVIL_FULL_TILE_SUBSTITUTIONS:
        if source.count(original) != 1:
            raise ValueError("Unrecognized ANVIL full-tile dispatch")
        source = source.replace(original, replacement, 1)
    for original, replacement, count in ANVIL_COOPERATIVE_SUBSTITUTIONS:
        if source.count(original) != count:
            raise ValueError("Unrecognized ANVIL cooperative dispatch")
        source = source.replace(original, replacement)
    return source


def replace_polar(source: str) -> str:
    for original, replacement, count in POLAR_SUBSTITUTIONS:
        if source.count(original) != count:
            raise ValueError("Unrecognized Polar Express launch")
        source = source.replace(original, replacement)
    return polar_full_wrappers(source)


NATIVE_VALUE_PULL_REPLACEMENTS = (('        )\n        # Empty tuple denotes a clean workspace. None selects a full clear.\n        self.grad_previous = ()\n        self.landing = None  # a started land: (pull, received, work, served), until complete_land\n        self.reset()\n\n', '        )\n        # Empty tuple denotes a clean workspace. None selects a full clear.\n        self.grad_previous = ()\n        self.scale_read_profiles = {}\n        self.landing = None  # a started land: (pull, received, work, served), until complete_land\n        self.reset()\n\n'), ('        if not clear_value_want_(self.grad_accum, ids):\n            self.grad_accum.index_fill_(0, ids.long(), 0)\n\n    @torch.no_grad()\n    def adam_update(self):\n        """Event, part 2 (collective with send_grads): rebuild this rank\'s dense gradient shard and run\n', '        if not clear_value_want_(self.grad_accum, ids):\n            self.grad_accum.index_fill_(0, ids.long(), 0)\n\n    def _current_scale_read(self, grad):\n        if grad.device.type != "cuda":\n            return None, None\n        from torch._inductor.codecache import PyCodeCache\n\n        cfg = self.optimizer.param_cfgs[self.param]\n        signature = (tuple(cfg.adam_betas), len(PyCodeCache.modules))\n        return signature, self.scale_read_profiles.get(signature)\n\n    def _remember_scale_read(self, grad, signature):\n        if signature is None:\n            return\n        from torch._inductor.codecache import PyCodeCache\n        from track_1_short.perf.kernels.value_adam_read_scale import probe_scaled_read\n\n        profile = probe_scaled_read(self.optimizer, self.param, grad, self.world_size)\n        if profile is not None:\n            self.scale_read_profiles[(signature[0], len(PyCodeCache.modules))] = profile\n\n    @torch.no_grad()\n    def adam_update(self):\n        """Event, part 2 (collective with send_grads): rebuild this rank\'s dense gradient shard and run\n'), ('        work.wait()\n        pull, first = self.live, self.owners.first_row\n        grad = self.grad_shard\n        direct_own = own_grad is None\n        if direct_own:\n            own_grad = self.grad_accum[self.own_rows]\n        previous = self.grad_previous\n        self.grad_previous = None\n        if merge_sparse_value_gradients_(grad, self.grad_workspace, self.grad_touched, own_grad, received,\n                                        pull.want, pull.asked, pull.lo, pull.hi, first, self.world_size, previous,\n                                        direct_own=direct_own):\n            if pull.lo != pull.hi or pull.asked.numel():\n                self.grad_previous = (pull.want, pull.asked, pull.lo, pull.hi)\n            else:\n                self.grad_previous = ()\n        else:\n            grad.zero_()\n            if not scatter_value_gradients_(grad, own_grad, received, pull.want, pull.asked, pull.lo, pull.hi, first,\n                                            direct_own=direct_own):\n                own_rows = pull.want[pull.lo:pull.hi] - first\n                if direct_own:\n                    full = own_grad\n', '        work.wait()\n        pull, first = self.live, self.owners.first_row\n        grad = self.grad_shard\n        profile_key, profile = self._current_scale_read(grad)\n        direct_own = own_grad is None\n        if direct_own:\n            own_grad = self.grad_accum[self.own_rows]\n        previous = self.grad_previous\n        self.grad_previous = None\n        next_previous = None\n        sparse = merge_sparse_value_gradients_(grad, self.grad_workspace, self.grad_touched, own_grad, received,\n                                        pull.want, pull.asked, pull.lo, pull.hi, first, self.world_size, previous,\n                                        direct_own=direct_own, scale_read=profile is not None, grouped_local=True)\n        if sparse:\n            if profile is None and (pull.lo != pull.hi or pull.asked.numel()):\n                next_previous = (pull.want, pull.asked, pull.lo, pull.hi)\n            else:\n                next_previous = ()\n        else:\n            if profile is None or previous is None or previous:\n                grad.zero_()\n            else:\n                # The previous consuming launch left both gradient and\n                # marker planes clear. The fallback leaves markers alone.\n                next_previous = ()\n            if not scatter_value_gradients_(grad, own_grad, received, pull.want, pull.asked, pull.lo, pull.hi, first,\n                                            direct_own=direct_own, grouped_local=True):\n                own_rows = pull.want[pull.lo:pull.hi] - first\n                if direct_own:\n                    full = own_grad\n'), ('                    full.index_fill_(0, own_rows.long(), 0)\n                scatter_add_rows(own_grad, own_rows, grad)\n                scatter_add_rows(received, pull.asked - first, grad)\n            grad.mul_(1.0 / self.world_size)\n        self.optimizer.adam_update_shard(self.param, grad)\n', '                    full.index_fill_(0, own_rows.long(), 0)\n                scatter_add_rows(own_grad, own_rows, grad)\n                scatter_add_rows(received, pull.asked - first, grad)\n            if profile is None:\n                grad.mul_(1.0 / self.world_size)\n        if profile is None:\n            self.optimizer.adam_update_shard(self.param, grad)\n            self._remember_scale_read(grad, profile_key)\n        else:\n            self.optimizer.adam_update_shard(self.param, grad, scale_read=(*profile, not sparse))\n        self.grad_previous = next_previous\n'))
NATIVE_VALUE_ANVIL_REPLACEMENTS = (('    # -----------------------------------\n    # Adam update\n\n    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig) -> Tensor:\n        """Apply Adam update to a parameter. Returns the updated p_slice."""\n        beta1, beta2 = p_cfg.adam_betas\n        lr = p_cfg.lr * p_cfg.lr_mul\n', '    # -----------------------------------\n    # Adam update\n\n    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig, *, scale_read=None) -> Tensor:\n        """Apply Adam update to a parameter. Returns the updated p_slice."""\n        beta1, beta2 = p_cfg.adam_betas\n        lr = p_cfg.lr * p_cfg.lr_mul\n'), ('        self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))\n        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)\n\n        AnvilAndAdam._adam_update_step(\n            p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],\n            beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t\n        )\n\n        return p_slice\n\n    @torch.no_grad()\n    def adam_update_shard(self, param: nn.Parameter, grad_chunk: Tensor):\n        """Adam on this rank\'s shard of a sharded parameter whose gradient was reduced outside step()\n        (value_embeds: perf/value_embed_pull.py). Same update, state and config as in step()."""\n        p_cfg = self.param_cfgs[param]\n        assert p_cfg.optim == "adam" and p_cfg.comms == "sharded" and param.grad is None\n        self._adam_update(param, grad_chunk, p_cfg)\n\n    @staticmethod\n    @torch.compile(dynamic=False, fullgraph=True)\n', '        self._step_size_t.fill_(lr * (bias2 ** 0.5 / bias1))\n        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)\n\n        if scale_read is None:\n            AnvilAndAdam._adam_update_step(\n                p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],\n                beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t\n            )\n        else:\n            from track_1_short.perf.kernels.value_adam_read_scale import launch_scaled_read\n\n            launch_scaled_read(p_slice, grad_chunk, p_state["exp_avg"], p_state["exp_avg_sq"],\n                               self._step_size_t, self._eff_wd_t, *scale_read)\n\n        return p_slice\n\n    @torch.no_grad()\n    def adam_update_shard(self, param: nn.Parameter, grad_chunk: Tensor, *, scale_read=None):\n        """Adam on this rank\'s shard of a sharded parameter whose gradient was reduced outside step()\n        (value_embeds: perf/value_embed_pull.py). Same update, state and config as in step()."""\n        p_cfg = self.param_cfgs[param]\n        assert p_cfg.optim == "adam" and p_cfg.comms == "sharded" and param.grad is None\n        self._adam_update(param, grad_chunk, p_cfg, scale_read=scale_read)\n\n    @staticmethod\n    @torch.compile(dynamic=False, fullgraph=True)\n'))

def replace_native_value(source: str, replacements: tuple) -> str:
    for original, replacement in replacements:
        if source.count(original) != 1:
            raise ValueError("Unrecognized value Adam read-scale dispatch")
        source = source.replace(original, replacement)
    return source


TRACKED_PATCHES = {
    DRIVER: replace_driver,
    MODEL: replacement,
    ATTENTION: replace_yarn,
    NGRAM: replace_ngram,
    AMAX: replace_qkv_amax,
    SAMPLED: replace_sampled,
    SAMPLED_CE: replace_densify,
    CROSS_ENTROPY: replace_cross_entropy,
    TRANSPOSE: replace_transpose,
    ROPE: replace_rope,
    TAIL: replace_tail,
    VALUE_PULL: lambda source: replace_native_value(replace_value_pull(source), NATIVE_VALUE_PULL_REPLACEMENTS),
    VALUE_EMBED: replace_value_embed,
    NGRAM_TABLE: replace_ngram_table,
    NGRAM_ADAM: replace_ngram_adam,
    FINEWEB_DATA: replace_fineweb_data,
    SCHEDULE: replace_schedule,
    STEP_GRAPHS: replace_step_graphs,
    PINNED_BATCHES: replace_pinned_batches,
    ROW_OWNERS: replace_row_owners,
    ROW_SCATTER: replace_row_scatter,
    ROW_PREFETCH: replace_row_prefetch,
    REPLICATED_ADAM: replace_replicated_adam,
    ANVIL: lambda source: replace_native_value(replace_anvil(source), NATIVE_VALUE_ANVIL_REPLACEMENTS),
    POLAR: replace_polar,
}


def uses_packed_adam_reference(checkout: Path) -> bool:
    installed = (checkout / NATIVE_ADAM).read_bytes()
    if installed == Path(__file__).with_name(NATIVE_ADAM.name).read_bytes():
        return False
    if installed == PACKED_ADAM_REFERENCE.read_bytes():
        return True
    raise ValueError(f"Candidate kernel differs from the checked-in sources: {NATIVE_ADAM}")


def install(checkout: Path, *, packed_adam_reference: bool = False) -> None:
    if git(checkout, "rev-parse", "HEAD").strip() != BASELINE:
        raise ValueError(f"Expected baseline commit {BASELINE}")
    if git(checkout, "status", "--porcelain", "--", *(str(path) for path in (*TRACKED_PATCHES, *PRIVATE_KERNELS))):
        raise ValueError("The candidate changes tracked files in a dedicated baseline checkout")
    kernels = [checkout / file for file in PRIVATE_KERNELS]
    if (not all((checkout / path).is_file() for path in TRACKED_PATCHES)
            or any(kernel.exists() for kernel in kernels)):
        raise ValueError("Expected the pristine Track 1 source layout")
    before = {path: (checkout / path).read_text() for path in TRACKED_PATCHES}
    updated = {path: patch(before[path]) for path, patch in TRACKED_PATCHES.items()}
    try:
        for path, kernel in zip(PRIVATE_KERNELS, kernels, strict=True):
            source = Path(__file__).with_name(kernel.name)
            if path == NATIVE_ADAM and packed_adam_reference:
                source = PACKED_ADAM_REFERENCE
            shutil.copyfile(source, kernel)
        for path, content in updated.items():
            (checkout / path).write_text(content)
    except OSError:
        for path, content in before.items():
            (checkout / path).write_text(content)
        for kernel in kernels:
            kernel.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--packed-adam-reference", action="store_true",
                        help="install the packed-gradient value Adam reference for comparison")
    args = parser.parse_args()
    install(args.checkout, packed_adam_reference=args.packed_adam_reference)
