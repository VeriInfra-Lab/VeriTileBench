# Modded-NanoGPT Track 1 candidate

Track 1 measures training-section wall time to FineWeb validation loss at
most 3.28 on eight NVIDIA H100s. Record 92 reports 39.914 seconds and mean
validation loss 3.27731 over 18 runs. The rules require a same-hardware
baseline and preserve the token streams. [Leaderboard and rules](https://github.com/KellerJordan/modded-nanogpt#world-record-history) · [Record 92 logs and statistics](https://github.com/KellerJordan/modded-nanogpt/pull/360)

This patch targets the upstream Track 1 trainer at
`4ea6b937337a4889b8cfe3f38a93d120048d8f71`, which packages the record's
ANVIL2 training algorithm. It keeps the original optimizer, sampled-softmax
schedule, training and validation tokens, CUDA graph capture and timed region.
[Baseline source tree](https://github.com/KellerJordan/modded-nanogpt/tree/4ea6b937337a4889b8cfe3f38a93d120048d8f71) · [Upstream MIT license](UPSTREAM_LICENSE)

- `mlp_scale_refresh.py` combines the post-quantization FP8 MLP scalar tail
  into one Triton launch. It retains both amax resets, the FP32 multiplication
  order and the BF16 round trip, and updates buffers in place for graph replay.
- `mlp_weights_refresh.py` refreshes both Hopper MLP projections together.
  One launch publishes the delayed scales from the previous partials. A
  second launch packs adjacent 64×64 tiles using one 64×128 CTA and
  publishes a separate amax for each original tile. This halves the pack
  grid from 12,672 to 6,336 CTAs. The eight warps publish their maxima
  for both halves in a separate 64-byte shared buffer. One lane bit selects
  the column half. Separate row and column shuffles preserve that bit, so
  all writers of each shared word hold the same maximum. One CTA barrier
  publishes the buffer. Each lane writes the row plane with two 128-bit
  stores and the transposed plane with four 64-bit stores. The pack uses
  4,160 shared bytes per CTA and one CTA barrier. The previous paired layout
  emitted 464 non-NOP SM90a instructions, 13 shuffles and used 40 registers.
  The vectorized layout emits 440 instructions, seven shuffles and uses
  31 registers with zero spills. The same packer serves the exact BF16
  bootstrap scales and the delayed schedule.
  [CUDA shared-memory writes](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html#shared-memory-access-patterns)
- `lm_head_refresh.py` specializes the dense 768×50,304 LM-head weight
  cache to full 64×128 tiles on Hopper. The exact bit pattern of the pinned
  inverse scale selects a packed BF16 multiply by 224. A kernel-checked
  finite proof covers all BF16 input bits and establishes the same E4M3
  bytes as the original FP32 multiplication and BF16 rounding. Other gains
  use the FP32 multiply and packed BF16 round-to-nearest step. Both branches
  clamp the upper bound with packed minnum. Two BF16-to-FP16 conversions
  feed each packed saturating E4M3 cast. A separate finite proof checks
  that this intermediate rounding preserves the FP8 byte for every BF16
  encoding. The warp-local exchange supplies both layouts. Each thread
  issues two 128-bit row stores and four 64-bit column stores, replacing
  eight 64-bit stores. The default branch has 182 static SM90a instructions
  along its execution path, down from 202. The generic branch goes from
  282 to 262, including dispatch. Both use 32 registers, zero spills,
  4 KiB shared and three warp barriers. Other shapes use the reference
  launcher.
  [PTX BF16 multiply](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#half-precision-floating-point-instructions-mul)
- `sampled_rows_gather.py` gathers the staged class IDs into both GEMM
  layouts and rebuilds the vocabulary inverse map in a single Hopper
  launch after the candidate upload wait. This replaces the separate
  gather, fill and index-copy launches. Each of the 197 inverse CTAs
  owns one interval. It searches the ends of sixteen-entry blocks, then
  loads the selected block contiguously and uses a warp sum to find the
  first candidate in the interval. The three candidate counts bound
  the binary search at 10, 10 and 11 serial loads, down from 14, 14
  and 15, followed by one coalesced warp load. Each CTA clears its interval
  to -1 and publishes the selected ranks after a CTA barrier.
  Every 128×128 gather uses four warp-local transfers and avoids the
  second read of the row cache, saving 7.5, 10.5 or 18 MiB at the three
  candidate counts. The weight loads and both output stores use 128-bit vectors. The gather arm has 130
  static non-NOP SM90a instructions, down from 147. The fused
  launch uses 32 registers per thread and 4 KiB of shared memory
  per CTA with zero spills.
- `sampled_grad_densify.py` writes the compact BF16 LM-head gradient back
  to all vocabulary columns. Within each 64×128 tile, every lane reuses
  its compact source and destination bases for four 16-row phases.
  Across the three sampled widths, the Hopper build drops from 179 to 146
  static non-NOP SASS instructions and from 48 to 32 registers per thread.
  The launch uses zero shared memory and zero spills. Lean checks the
  inverse lookup, the exact byte pitches of all four phases, each preserved
  halfword, and the positive zero written at each absent class.
- The long-window RoPE epilogue writes a key's stationary channels to the
  next token after normalizing the current row. This eliminates the extra
  QK load and prior-row norm. Across the four training lengths and the
  validation length the Hopper kernel emits 165 static PTX instructions
  versus 214 upstream, with one reciprocal square root per CTA in place
  of two. The first key keeps the ordinary RoPE expression and the final
  key's shifted store is masked. `RopeShift.lean` proves the write routing
  with the original FP32 order and BF16 conversion.
- The long-window backward limits the current K gradient load to rotating
  channels and token zero. It loads the successor gradient only for
  stationary channels. Hopper instruction replay shows 8 KiB of K-gradient
  load requests per interior CTA versus 16 KiB upstream. It checks both
  FP8 layouts against the ordered RMS derivative, including the first
  token addition and the last token's inactive successor load.
  The second rotary factor is loaded contiguously and its adjacent BF16
  lanes are swapped in registers. This eliminates 16 static scalar loads
  and brings the full backward kernel from 1,160 to 1,087 PTX instructions.
  `RopeBackward.lean` proves equality of the two packed gradients and
  unique ownership of both byte layouts.
- The 64-channel backward also uses the vector factor load. This covers
  paired heads with 128-channel V and unpaired heads with 64-channel V.
  Both eliminate 16 static scalar BF16 loads. The paired path removes 69
  PTX instructions and the half-width V path removes 70 across all four
  training lengths. The proof uses the original paired-head gradient and
  rotary-table indices and preserves both FP8 outputs.
- YaRN stores one FP32 frequency and one BF16 cosine/sine per adjacent
  channel pair. The pinned trainer also omits the final 32 pairs from
  wide-head storage. Those pairs have positive-zero frequencies across
  every reachable window. The Triton decoder supplies BF16 +1 and +0,
  restores the original odd-lane sine sign, and uses the same FP32
  contraction order. The three factor tables occupy 128 MiB per rank at
  the default eight-rank validation extent, down from 320 MiB upstream
  and 160 MiB with pair packing alone. On sampled wide-head Hopper CTAs,
  PTX replay checks identical output expressions and factor load requests
  of 512 bytes forward and 4 KiB backward. Pair packing alone requests
  1 KiB and 8 KiB respectively. The decoder adds one static PTX instruction
  to each path and keeps the same register count.
  `YarnTable.lean` and `YarnStationary.lean` prove the storage relation and
  its use by both kernels through reset, window extension, partial rebuild
  and validation completion. The proof checks the signed-zero behavior
  against FP32 coefficient bits derived from the pinned schedule. The
  modeled trigonometric operations share their zero semantics.
  [CUDA sine and cosine semantics](https://docs.nvidia.com/cuda/archive/13.1.2/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html)
- The RoPE forward, RoPE backward and transpose-copy launchers declare their
  destination mutations as PyTorch custom ops. Their pointer checks run at
  launch, keeping the kernels callable under the trainer's full-graph compilation.
- `byte_transpose.py` specializes the MLP and cross-entropy FP8 transposes
  on Hopper. It copies raw bytes with four warp-local exchanges per
  128×128 tile. The compiled dense case uses 4 KiB of shared memory per
  CTA versus 16 KiB upstream and emits 140 PTX instructions versus 351.
  The original launcher handles tails, noncontiguous tensors, BF16
  inputs outside the tied-embedding shapes, and cases outside the alignment
  or index bounds.
- The full and sampled cross-entropy backwards share one launch to
  transpose the E4M3 activations and E5M2 logit gradients before the
  weight-gradient GEMM. The first six column tiles select the activation
  buffers and the remaining tiles select the gradient buffers. Both
  ranges use the same warp-local exchange. SM90a uses 32 registers
  and 8 KiB of shared memory per CTA, with no spills. Guarded fallback
  calls the two existing transposes.
- The shared training cross-entropy packs pairs of FP8 logits into each
  E4M3-to-FP16 decode and rounds each sigmoid pair with one packed FP16
  conversion. With CUDA 12.8 and Hopper, static non-NOP SASS instruction
  counts fall from 1420 to 1319 at 10,240 classes, from 1819 to 1682
  at 14,336, from 2118 to 1897 at 24,576, and from 3597 to 3122 at
  50,304. The 24,576-class kernel uses 127 registers, up from 122.
  The full-width kernel uses 127, down from 128. Shared memory and the
  zero-spill result are unchanged.
- The same training loss compiles five fixed prediction-count variants
  at import for the default schedule. Each fixed kernel stages one logit
  byte per class. Its 256 lanes round the sigmoid to FP16, widen it to
  FP32, and compute the scaled logit and sum contribution for all byte
  encodings. The forward scan loads eight FP8 codes as two 32-bit words
  and extracts each byte with a PTX permutation, as in the base-gradient
  loop. The scan still accumulates the contributions in the original
  order. After the sum barrier, two aligned four-word shared loads
  fetch the eight warp sums for the original ordered FP32 fold. After
  computing the row logsumexp, each lane writes its rounded sigmoid and
  row probability together. Until that publication the lane keeps its
  sigmoid and scaled logit in registers across the scan and reduction.
  This removes one shared load and one shared store per CTA. The full-width
  kernel uses a launch bound of six blocks for the compiler's register
  allocation. Each lane also
  computes one rounded E5M2 base gradient and publishes it in the first
  256 bytes of the exponential table after all lanes finish reading it.
  Both sets of writes share the next CTA barrier. The base loop loads
  each eight-code chunk as two words, selects the codes with PTX byte
  permutations, gathers the cached gradients, and writes all eight bytes
  together. Target corrections use the widened sigmoid and probability.
  The loss recomputes its target logits with the same rounded FP32 multiply.
  Each CTA evaluates 256 gradient exponentials and 256 base gradient
  conversions at every width. At 50,304 classes, the static PTX suffix
  has 4 E5M2 conversion instructions, down from 19 with probability
  caching alone. Hopper disassembly counts 450 static non-NOP SASS
  instructions with register publication. Dynamic shared memory uses
  one byte per class, and total static shared memory stays at 3,120 bytes.
  Lane zero publishes the gradient factor at the initialization barrier.
  The sampled kernels use 32 registers per thread. The 50,304-class
  kernel uses 40, down from 48 with probability caching alone and 127 in
  the direct fixed kernel. All five variants assemble without spills.
  The CUDA occupancy calculator with H100's maximum shared-memory
  carveout gives eight CTAs per SM for
  each sampled width and four for full width. The generic paired kernel
  handles other prediction counts. In the fixed kernels each target's
  original eight-byte store lane also writes its correction. It processes
  each target column once after its base stores. This removes the CTA
  barrier before gradient corrections. The fixed variants handle the
  prefix write after the MTP loop. Each MTP correction still accumulates
  the prefix weight when the raw prefix target matches. The last path
  writes only an eligible prefix target absent from the MTP slots.
  At 10,240 classes lane zero also places the two or three MTP target
  indices in shared memory at the existing initialization barrier.
  It checks the row and vocabulary bounds before narrowing to a signed
  32-bit word, using −1 for an invalid target. Both the loss and sparse
  corrections use that cache. Hopper disassembly counts 498, 550, 501,
  388 and 450 non-NOP SASS instructions for the five fixed configurations.
  [CUDA launch bounds](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
- On the five default training CE configurations, the head reuses its
  E4M3 GEMM output allocation for the E5M2 logit gradients. The CUDA
  launch receives the same byte pointer at both ports. Each CTA snapshots
  its own row before the sum barrier, then overwrites only that row. The
  autograd wrapper saves an E5M2 view of the result for the two backward
  GEMMs. This removes a 2,060,451,840-byte gradient allocation at
  40,960 rows and 50,304 classes per rank. Other configurations use
  separate storage. The Lean execution checks addresses and phase order,
  then proves the same loss and gradient bytes through the reused buffer.
- `bf16_transpose.py` specializes the tied embedding's post-gather weight
  copy and pre-reduce gradient add on Hopper. The kernel issues four 128-bit
  source loads per lane in place of eight 64-bit loads. Transposed matrix
  stores feed four vector reads in the same 4 KiB of shared memory.
  The non-NOP SM90a instruction count falls from 84 to 67 for the copy
  and from 104 to 87 for the add. The copy uses 29 registers instead of
  32, and the add still uses 40. Both assemble without spills. The two
  packed BF16 halves retain the reference destination-first operand
  order. The optimizer's pinned head reduction and gather sites call the
  checked transpose dispatch.
  [PTX matrix stores](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix)
- `ngram_tiles.py` uses one Hopper CTA for four complete 768-channel rows
  of the n-gram lookup or pullback. Each CTA reads the token words and
  computes the sign hashes once, then uses them across the three
  256-channel tiles. Per four tokens, non-NOP SASS counts fall from 540
  to 305 forward and from 297 to 118 backward compared with separate
  channel CTAs. Registers per thread increase from 30 to 37 forward
  and from 28 to 32 backward. Both versions assemble without spills.
  Partial or unaligned layouts retain the masked channel tiles.
- `qkv_amax.py` splits each of the 32 QKV amax parts into a QK loop and a
  V loop. Both visit the original tiles in order. Each iteration loads
  from one bank without a lane predicate. The aligned H100 specialization
  emits a 128-bit load for both banks and removes the FP32 add with zero.
  Both bank lengths must be multiples of the 2,048-element tile. Other
  layouts continue through the original kernel.
- `qkv_scale.py` runs the 32-part amax, FP32 scale floor, and reciprocal
  multiplication in one warp per cache row. It replaces three eager GPU
  operations with one launch for each of the three width groups, saving six
  kernel nodes per attention refresh. The launcher requires separate contiguous
  FP32 scratch and output buffers and retains the original path for other layouts.
- `qkv_pack.py` specializes the final FP8 quantizer to the full 64×128
  tiles of the three width groups. Each CTA selects one bank and pitch
  before constructing lane offsets. Both FP8 layout stores are unmasked.
  The launcher uses the original kernel for partial tiles, bank splits
  inside a tile, and shapes outside the signed i32 offset range.
- `qkv_pack_gluon.py` is the aligned path on Hopper. The kernel quantizes
  each weight once in the row layout, stores that FP8 byte to the row
  cache, and redistributes the completed byte for the transposed cache.
  The compiled packer halves FP32 divisions per thread. Eight warps use
  independent 512-byte regions, for 4 KiB of shared memory per CTA
  compared with 16 KiB upstream. Each warp reuses its region in two phases
  with warp barriers and an ldmatrix load in each phase.
- `qkv_native.py` sends the pinned Hopper trainer's QK and V bank views
  directly to the amax and packing kernels. The head strides preserve V/O
  interleaving and skip the unused half of cropped heads. The three groups
  avoid 6.75 MiB of BF16 reshape output per refresh.
- `qkv_grouped.py` schedules all seven attention layers together on Hopper,
  with one launch per refresh stage. This reduces nine launches to three.
  The amax scan groups three successive tile loads so each cropped-head
  address pattern is computed once. The original ordered per-lane FP32
  folds, CTA reduction and partial stores are preserved. The SM90a scan
  drops from 460 to 411 non-NOP SASS instructions and from 32 to 30
  registers per thread, with 32 bytes of shared memory and no spills.
  `QKVScanSchedule.lean` proves the identical ordered tile sequence for
  all 224 CTAs, including both ways the short V group can finish.
  The pack grid covers 30 bands of six row tiles and all six column tiles.
  It finds the owning layer with a bitmask and two integer instructions.
  The packer computes each tile's three bases once and emits two
  128-bit row stores and four 64-bit transposed stores per lane. On
  the pinned Hopper compiler this cuts static SASS from 406 to 384
  instructions and global stores from eight to six, with 4 KiB of
  shared memory, 31 registers, and no spills.
- `tail_lerp.py` updates the two private FP32 tail averages from one BF16
  load when both averages tick for the same VO or MLP shard. The default
  1,194-step schedule has 63 overlapping ticks and the first seeds the
  tail average. Fusion saves 124 launches and 906.75 MiB of BF16 load
  requests per rank over the remaining ticks. For two updates to a
  4,096-element tile, the compiled pair uses 176 static PTX instructions
  versus 294 across the two individual launches. The Gluon layout keeps
  four adjacent lanes per thread for each vector transfer, so no shared
  memory exchange is needed. The original launcher covers unmatched
  ticks and layouts outside the Hopper path.
- `tail_norm.py` measures the five final weight norms before and after
  averaging using the FP32-output BF16 CUDA specialization in the pinned
  PyTorch 2.10 source. The aligned path removes ten FP32 weight temporaries,
  avoiding about 2.21 GiB per rank in writes to those temporaries and reads
  back into the reductions. A single Triton launch applies the five
  FP32 ratios in place and rounds the outputs to BF16. The ratios stay
  on the GPU from calculation through the last store. Dispatch requires
  the pinned ATen revision, Hopper, and the five disjoint aligned banks.
  [ATen mixed-precision norm](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/ReduceNormKernel.cu)
- `value_embed_accum.py` fuses the four value-embedding backward scatters
  on aligned Hopper buffers. One CTA handles a token, with one warp per
  plane. This replaces four launches with one and four CTAs per token with
  one. Each lane loads eight BF16 words at a time and the compiled kernel
  issues one eight-element FP16 atomic add per stripe. The compiled
  register count is 19 per thread versus 16 for an original CTA, with
  zero shared memory and no spills. Other layouts use the original
  per-plane launcher. The proof retains the ordered scalar updates for
  each address, including repeated token IDs. [PTX vector atomics](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom)
- `sparse_row_pack.py` writes each active value-embedding gradient row
  directly to the local owner's buffer or the all-to-all wire. It avoids
  the intermediate compact copy of each remote row, saving 3,072 bytes
  of read and write requests per remote row at width 768. The installed
  Hopper path packs only the wire when the exchanged peer IDs are ready.
  Value Adam reads the owned rows in place and clears them after their
  last read, saving another 3,072 bytes per owned row relative to packing
  both outputs. The remote clear still runs after both exchanges are issued.
  The owned scatter retains the FP16 atomics and uses 16 registers per
  thread compared with 14 for the compact input, with no spills. The n-gram wire
  packs both remote slices in one launch using the original scalar FP16
  widening and BF16 round-to-nearest operations. The local n-gram slice
  stays FP16. One warp handles the three 256-word stripes of a row,
  using the same row index and bounds for each stripe. The launches
  use one CTA per four rows in place of three. The value wire path
  issues one row-index load instruction and one CTA barrier per group.
  The two gradient exchanges still launch before the value
  buffer is cleared. On Hopper the clear reads the existing int32 want
  IDs and stores positive FP16 zero in one launch, saving the int64
  conversion launch and its temporary of eight bytes per active row.
  The n-gram path checks the pinned ATen revision.
  [ATen CUDA copy and conversion](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/Copy.cu)
  [ATen CUDA index fill](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/IndexKernel.cu#L1433-L1487)
- `sparse_row_fetch.py` subtracts the shard's starting row and gathers
  the requested BF16 words in the same launch, removing the temporary
  int32 index tensor for both tables. The n-gram cache update combines
  the prefix and suffix transfers, local index subtraction, and local
  gather into one launch. It writes the cache in place and leaves its
  unused suffix intact. Each kernel uses 64-bit source addresses for the
  default 10,575,360-row n-gram shard. One CTA handles four full rows with
  three vector transfers per lane and a single row-index load instruction.
  The original calls cover other layouts.
- `ngram_moment_replay.py` brings the next cycle's local FP32 Adam second
  moments up to date. One warp handles up to 32 rows and starts at the
  earliest outstanding event among lanes with nonzero moment words.
  Positive-zero lanes publish only the stamp and already-current lanes
  skip the moment read and both stores. A tile containing only these
  lanes reads no beta2 history. Every other lane keeps the original
  FP32 multiplication order. The SM90a specialization uses 26 registers
  per thread, the same as the full replay, with zero shared memory and
  no spills. The same positive-zero check skips the history loop for
  an Adam row update. Duplicate entries exit before reading the moment
  or stamp unless they own the claim. The compiled update uses 31
  registers per thread compared with 29 upstream, with no spills.
- `ngram_adam_entries.py` combines the two index subtractions and the
  concatenation into one Hopper launch. A single allocation holds the
  packed entries and an aligned mirror for the peer scatter. The own
  scatter reads the packed prefix, and repeated peer IDs retain their
  original positions. The same launch publishes the claims with relaxed
  32-bit exchanges and initializes all 768 BF16 channels of each live
  gradient entry to positive zero. This removes two launches per eligible
  Adam event. The kernel uses 16 registers per thread, zero shared bytes,
  and three 128-bit stores per lane for the gradient.
  The BF16 peer scatter shares the four-row tile with the owned scatter.
  Across four rows its SM90a artifacts use one CTA and 67 static non-NOP
  instruction slots in place of four CTAs and 144 slots. Registers per
  thread increase from 14 to 20, with zero spills. The checked program
  feeds each peer word into the recorded per-address BF16 addition order,
  including duplicate IDs. The training artifact checker also loads the
  Adam entry preparation and both weight-update variants.
- `value_adam_scatter.py` folds both global-to-local index conversions
  into the FP16 scatter kernels. Each Hopper launch reads the original
  pull IDs directly, eliminating two index kernels and their temporary
  buffers. The local scatter precedes the peer scatter, and the merged
  shard is rounded to FP16 after scaling by the inverse world size.
- For the pinned CUDA build at world size eight, value Adam records the
  largest one-based scatter position per row in the existing 32-bit
  marker. The scale launches over the local and received entries when
  their total is at most half the shard's rows. Duplicate positions for
  one row select a single writer. The scale uses the same scalar FP32
  multiply and FP16 write-back as ATen. On a 25,152-row shard the grid
  uses at most 3,144 CTAs in place of 6,288. Static SM90a instruction
  slots per CTA increase from 111 to 131, with 18 registers per thread
  and no spills in either kernel.
  Before a second marked event, the clear kernel visits the previous
  local and received entries and clears each winning row once. It saves
  the two ID tensors and their local cut until the next event. The
  original full-row clear uses 43 static SM90a instruction slots per CTA
  with 14 registers. The indexed clear uses 73 slots with 16 registers,
  zero spills, and a grid of at most 3,144 CTAs on the default shard.
  A clean workspace skips this launch. The next marked event after the
  dense fallback zeros the full allocation before both scatters.
  At the default shard size, the sparse clear removes at least 18.4 MiB of
  FP16 writes per rank and eligible event. This counts the payload bytes only.
  [ATen CUDA multiply](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/BinaryMulKernel.cu)
- `replicated_adam_spans.py` constructs an immutable plan when the flat
  replicated parameter buffers are created. Each Hopper CTA covers the
  intersection of one parameter interval with a 512-element tile and
  reads one row of Adam scalars. The flat gradient all-reduces and
  active-segment logic are preserved. For BF16 the compiled kernel uses
  32 registers per thread and 27 static PTX global load instructions per
  512-element CTA. The original 1,024-element CTA uses 90 registers per
  thread and 104 static global load instructions. Groups below 512
  elements and layouts outside the specialization use the original update.
- `anvil_epilogue.py` fuses the equalized BF16 gradient with the
  two-part ANVIL weight update for the three default Hopper banks.
  TorchInductor emits one fewer Triton launch per bank and removes
  33.75 MiB per rank per step in requested writes and reads of the
  gradient intermediate. The BF16 rounding stays inside the fused
  kernel. Both weight halves and the lane-energy update are preserved.
- `polar_express.py` launches only the lower triangle for the default
  128-by-128 ANVIL Gram and polynomial tiles. Each launch shrinks
  from 24 to 18 CTAs for QK, 72 to 42 for VO, and 108 to 63 for MLP.
  Six Gram and six polynomial calls per bank remove 972 early-exit
  CTAs per rank per optimizer step. The full-tile specialization keeps
  the final transposed store on the diagonal and skips 504 completely
  overwritten tile stores, or 15.75 MiB of requested BF16 stores per rank
  per optimizer step.
  The Hopper full-tile schedule stages 128-by-128 BF16 operands with
  TMA. The left operand alternates between two boxes, so its next
  transfer can start while the current WGMMA group is in flight.
  Off-diagonal tiles refill a fixed right box after retiring that
  group and synchronizing the CTA. Both WGMMA operands view the
  current left box on the diagonal. This reuse saves 4,464 logical
  32 KiB TMA input boxes per rank and ANVIL update across the three
  default banks, or 139.5 MiB of requested payload. The Gram and polynomial
  kernels stage both output orientations in separate shared boxes.
  The output path commits each TMA write, then uses one full wait
  per phase. This waits for shared reads and for global writes to
  complete. The initial Gram and cooperative rounds use the same
  sequence before publishing their output. Each wide
  chunk issues eight WGMMA instructions in order in one commit group.
  The diagonal and off-diagonal CTAs branch once before their
  reduction loops. The diagonal loop expects one transfer and waits
  for at most one outstanding WGMMA group before prefetching. The
  off-diagonal loop expects two transfers and prefetches the next left
  box before its single wait. This brings the emitted non-NOP loop
  instruction slots per 128 reduction words from 97–103 in the common
  loop to 68–74 on the diagonal and 75–82 off the diagonal. The two
  loops increase static code size by 198–205 instructions per variant.
  Gram uses 90 registers and polynomial uses 102. All five variants
  have zero spills. The three boxes use 98,320 dynamic shared bytes
  per CTA, up from 65,552 for the 64-wide input boxes.
- On the pinned H100 profile with cooperative launch support, preparation
  zeros one private counter per matrix during its existing launch. The
  last CTA that publishes a diagonal tile of each first Gram matrix writes
  the original FP32 divisors. The first Gram uses 42 atomic publications
  across all three banks. Off-diagonal CTAs finish after writing their tiles. In each of
  the next five rounds, one Gluon launch computes Gram and the polynomial.
  After completing its TMA stores, each CTA publishes a bit in the private
  mask for its matrix and round with an acquire-release atomic OR. It proceeds
  as soon as the mask covers both rows of the symmetric Gram used by the polynomial and
  the bias tile. The CTA can use the read part of its own atomic when
  that observation covers the inputs. The polynomial otherwise waits with
  acquire loads. The two banks with six tiles per side wait for 6 producers
  on diagonal outputs and 11 on off-diagonal outputs, out of 21 producers
  per matrix. QK waits for 2 and 3 out of 3. The prepare launch zeros the
  initial trace counters and all five masks in the same allocation.
  Compared with separate native Gram, divisor and polynomial launches,
  this saves 18 launches per rank and ANVIL update. The initial Gram uses
  90 registers, and the later rounds use 104. Both use 98,320 dynamic
  shared bytes per CTA and compile without spills. Unsupported
  configurations use the separate launches.
  [CUDA cooperative launches](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-driver-api/group__CUDA__EXEC.html)
  [PTX release and acquire patterns](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-and-acquire-patterns)
- `anvil_trace.py` replaces the two reductions of the first Gram
  diagonal with a single Gluon launch for each of the three default
  banks on the pinned H100 compiler profile. This removes three
  launches per rank per optimizer step and the intermediate FP32
  partial-sum stores and loads. The same reduction publishes the FP32
  divisor and its square for each matrix. A second Gluon launch then
  normalizes the BF16 lookahead and Gram in place, saving three more
  launches per rank per optimizer step. It uses the same full FP32
  divide and BF16 rounding as Inductor, with two 128-bit BF16 loads
  and stores per thread. A 2,048-cell CTA reuses the same reciprocal
  for both vectors and cuts static instructions per cell by about 24%
  from the 1,024-cell version, with 26 registers and no spills.
  The captured wrapper requests 8.625 MiB fewer BF16 allocations
  across the three banks than the previous fused-divisor wrapper.
  The next Gram overwrites the same Gram storage in full.
- `anvil_rails.py` updates both FP32 velocity rails in place and writes
  the BF16 lookahead in one Triton launch. The captured Hopper wrapper
  originally emits a rail update, a copy back to live velocity, and a
  separate lookahead for each bank. The custom operator removes six
  launches per rank per optimizer step. Its mutation schema identifies
  the velocity and the newly allocated lookahead as writable, so
  Inductor can reuse the lookahead storage within the cascade.
  At the three default bank sizes, the eliminated FP32 temporary
  accounts for 202.5 MiB of requested stores and loads per rank per
  step in that wrapper. This measures the named intermediate's
  accesses before cache effects. The preparation CTA now covers 1,024
  cells and uses 128-bit loads for eight BF16 gradients per lane. Both
  FP32 rails and the BF16 lookahead use 128-bit stores. Compared with
  the 512-cell launch, the CTA count is halved and static SM90a
  instructions per cell fall by 18% to 19%, with no spills or shared
  memory. Register counts rise from 28 to 32 for the QK and VO banks
  and from 30 to 40 for the MLP bank.

## Build and verify

Use clean, separate checkouts at the pinned revision. Install the upstream
dependencies and prepare FineWeb according to the upstream instructions.
[Upstream build and data instructions](https://github.com/KellerJordan/modded-nanogpt#running-the-current-record)

```sh
python3 models/NanoGPT/Track1/install_candidate.py "$CANDIDATE"
lake build models.NanoGPT.Track1.CheckSource
python3 models/NanoGPT/Track1/export_source_link.py "$BASELINE" "$CANDIDATE"
lake env lean models/NanoGPT/Track1/Track1Driver.lean
python3 models/NanoGPT/Track1/verify_candidate.py "$BASELINE" "$CANDIDATE" --mode offline --ir-dir "$IR_DIR"
python3 models/NanoGPT/Track1/check_activation_fp8.py "$BASELINE" "$CANDIDATE" "$IR_DIR"
python3 models/NanoGPT/Track1/check_mudd_coefficients.py "$BASELINE" "$CANDIDATE" "$IR_DIR"
python3 models/NanoGPT/Track1/check_mudd_full_graph.py "$BASELINE" "$CANDIDATE" "$IR_DIR/full-mudd"
for phase in gram trace rails epilogue; do
  python3 "models/NanoGPT/Track1/check_anvil_${phase}.py" "$BASELINE" "$CANDIDATE" "$IR_DIR/anvil/$phase"
done
python3 models/NanoGPT/Track1/check_anvil_cooperative.py "$BASELINE" "$CANDIDATE" "$IR_DIR/anvil/cooperative"
python3 models/NanoGPT/Track1/check_training_ce.py "$BASELINE" "$CANDIDATE" "$IR_DIR/training-ce" \
  --nvrtc "$CUDA_HOME/lib64/libnvrtc.so" --include "$CUDA_HOME/include"
lake env lean --run models/NanoGPT/Track1/CheckSource.lean "$BASELINE" "$CANDIDATE" "$IR_DIR"
python3 models/NanoGPT/Track1/verify_candidate.py "$BASELINE" "$CANDIDATE" --mode cpu
```

With the pinned CPU PyTorch wheel, append `--check-inductor` to the
activation check to compare generated CUDA kernel source for the two
head wrappers at all four token counts. This captures RMSNorm through
the FP8 buffer feeding the projection.

The ANVIL checks use the pinned CPU PyTorch wheel to capture Inductor's
Hopper launch plans. The end-to-end artifact checker reads the emitted
TTGIR and PTX of the three rails, the dense, compact and full-tile Gram
and polynomial launches, the trace kernels, and both epilogue schedules.
For the cooperative path it also checks each bank's seeded preparation,
first Gram and divisor launch, and combined Gram and polynomial launch.
The host replay follows the trace counters, per-round masks, divisor and Gram outputs,
and all six projections. Lean proves that a last publisher sees every
tile for its matrix, that a later acquisition covers the polynomial's
actual Gram operands and bias tile, and that the fused phases preserve the original FP32 and BF16 words.
The Lean optimizer call uses the checked variants to compare the same
FP32 rail and divisor words, BF16 lookahead, and upper and lower weight
words through the final validation result.

The MUDD check uses the same pinned CPU wheel and Triton 3.6.0. It
captures the post-layer coefficient projections and residual mix with
the candidate's head wrappers and custom-op schemas. Use
`--baseline-heads` to capture the pinned upstream wrappers. A compile-only
stub stands in for the native CE kernel during capture. The check
compiles and symbolically replays the Hopper PTX for all six default
training configurations and tests the wrapper's allocation, stream,
operand order, and private statistic buffer.

The full-model check lowers all 11 GPT layers at the six default shapes.
It uses shape-only FlashAttention calls during capture and keeps the
candidate's CE boundary and paired transpose. In the first sampled
graph Inductor emits a larger residual kernel with two identical E4M3
stores. The wrapper drops the second store and swaps which of the two
private arenas receives the CE transpose. It keeps both allocations
and removes a 12 MiB write per rank. The check compiles the original and
rewritten Triton kernel under each of Inductor's reduction schedules.
`MuddActivationBinding.lean` checks that the TTIR differs only by the
duplicate store and its destination address calculation.
The offline check also compares TTGIR, including the reduction layouts.
`MuddActivationStorage.lean` proves the row-major and column-major
operands equal and preserves bytes outside the two arenas. This proof
feeds the same end-to-end equivalence through the weight-gradient GEMM.

The same full-model pass keeps the rounded BF16 residual in registers
when one reduction tile covers the 768 channels. Both paths save the
same BF16 buffer for backward. The smaller tiles keep the existing
memory reload. For the two 1024-channel schedules this removes one
masked load, corresponding to 24–72 MiB of requested BF16 reads per
rank and invocation across the default shapes. The offline check
compares TTIR and TTGIR in all 36 shape and schedule combinations.
`MuddResidualStorage.lean` proves equality of the FP8 activation and
the entire BF16 memory image, including addresses outside the saved
residual. Actual device traffic and execution time depend on the
schedule selected on the GPU.

The MUDD backward pass groups nine of the ten BF16 column reductions in
the six default graphs. In the sampled graphs it moves three reductions
before the pointwise packer, which reads all ten input planes and writes
a separate allocation. The wrapper verifies the original kernel body,
allocation origins and stream before combining the launches. It keeps the
two smaller groups if the packing boundary differs.
On the sampled packing fallback the wrapper selects the source
plane's BF16 pointer before loading. It retains the original nine FP32
adds with their zero operands, the multiply by 0.1 and the BF16 cast.
This replaces ten logical loads per output with one while retaining the
row-major store. The artifact checker specializes both pointwise schedules
to each plane and checks the exact addresses and arithmetic in TTIR and
TTGIR.
The new Triton grid selects the input plane on its second axis and
reuses the original kernel's chosen reduction tile and warp count.
The wrapper keeps the original launches until autotuning has selected
a supported configuration on the same stream.
`MuddGradientGrouping.lean` proves that every BF16 output word and the
rest of the destination memory agree. `MuddGradientPacking.lean` proves
that moving the late reductions across packing preserves the entire memory
image, including the packed output and input planes. The artifact checker
checks the wrapper's launch edits and allocation origins, then specializes
TTIR and TTGIR to each plane and checks the original reduction, layouts,
rounding and stores before composing both proofs into the backward replay.

After both reduction autotuners have selected supported configurations,
the four sampled graphs write the packed gradient from their reduction
launches. Matching tile and warp counts let one launch write all ten
columns and their packed planes. Every mixed pair among the one-, eight-,
32-, and 128-row reductions also uses one launch. Each block selects
the ordered reduction by column. Both the ten-column and mixed kernels use up to four shared
column-bit predicates to select the input pointer. The PTX checker
follows these predicates through the pointer selectors to the BF16
loads, and Lean proves the route for every launched column. The mixed
column-zero branch loads its input directly. The paired eight- and
one-row schedules use two warps. Schedules involving 32 rows use four warps, and schedules
involving 128 rows use eight warps. The mixed paths write both outputs
directly, with an FP32 FMA with 0.1 and positive zero for the packed
planes. On the 128-row branch this retains the packed vector store and
removes the shared-memory exchange by using a separate BF16 sum store
for each register group. Matching one-, eight- and 32-row schedules
also use direct Gluon stores. The single-row schedule groups four rows
into each block. For matching 128-row sampled schedules, each warp
owns 16 consecutive rows. Lanes zero and one of each four-lane subgroup
write the two register groups directly. The second lane only exchanges
the operands of the outer FP32 addition, which Lean proves preserves
the result. This uses four butterfly exchanges, one predicated sum
store, and two vectorized packed stores. The checker verifies the local
register selections and the unique writer for each row.
`MuddPackingArithmetic.lean` and `Proofs/MuddPackingFma.lean` share one
Lean kernel check over all 65,536 BF16 encodings. For nonzero finite
values, the check proves that adding positive zero preserves the input,
and the FMA follows from exact rational arithmetic and a single round.
The check evaluates the identities directly for zeros, infinities and NaNs.
`MuddFusedPacking.lean` proves the original eleven-launch sequence
and both fused sequences preserve the same BF16 memory image when
the stream drains the region. Its proof includes mixed dispatch, where
the two batches may choose distinct packing implementations.
`MuddFusedPackingBinding.lean` checks both stores and the ordered
reductions against the compiler artifacts. The
checked path is included in `ArtifactTraining.training_with_compiled_ce`
and its final-validation equality.

The 16,384- and 40,960-row full-vocabulary graphs write the BF16 column
sums, the BF16 inputs to the unchanged projection, and the packed gradient
in one launch for each supported schedule pair. The projection operand widens the rounded BF16 column sum,
multiplies by FP32 0.1, and narrows to BF16. The packed gradient uses an
FP32 FMA with positive zero. Lean checks the FMA rewrite across the
entire BF16 domain.

When an autotuner selects the single-row schedule, both vocabulary paths
assign four rows to a block. Each half-warp preserves the
original four shuffle distances and ordered additions. The artifact
check verifies the row coordinates and butterfly in the emitted PTX,
as well as the three TTGIR stores and the wrapper's projection view.
`MuddFullPacking.lean` models each split launch writing its own columns
to the projection operand. It proves equality of the memory passed to
the rest of the backward graph in both fused schedules. The end-to-end
training proof includes the same full-vocabulary path.

The mixed schedules keep the original per-row reduction trees and
assign further rows to register groups. The Lean checker follows both
PTX branches from their masked loads to the column sums, packed stores
and, in the full-vocabulary case, projection stores. It checks both
dispatch directions for all six pairs and all six default graphs. When columns 1–9 use
the one-row schedule and column 0 uses the 128-row schedule, eight warps
cover 64 rows per block. When columns 1–9 use the 128-row schedule and
column 0 uses the one-row schedule, the grid assigns eight 16-row blocks
to column 0 in each 128-row range. Nine additional blocks cover the
remaining columns. This uses 26 registers per thread in the compiled
Hopper kernel and reduces the 40,960-row launch's total block count from
13,120 to 5,440. The checker verifies the band dispatch, pointer arithmetic,
and unique coverage of every output cell.
For 16,384 rows, the mix of one- and 32-row reductions uses 5,120 blocks.
The preceding two-kernel paths used 37,376 or 8,704 blocks depending on
which bank selected the smaller tile. These are launch grid counts.
Device time is measured separately on the Track 1 setup.

`--mode offline` compiles both n-gram versions, the fused MLP refresh, and
both MLP weight refreshes, both LM-head packers, the sampled gathers with
and without inverse-map fusion at all three counts,
the per-width and grouped QKV paths, the RoPE backward in all three
attention layouts, and the fused graph staging lookups for `sm90`.
With Triton 3.6.0 it emits QKV TTIR, TTGIR and PTX at the training strides
and 16-byte pointer alignment. `CheckSource.lean` checks each width's TTIR
for the original, aligned and native scans, the aligned pack pair, and both
artifacts for the scale update.
On the Gluon path it checks the full TTGIR and executable PTX. It checks
the complete normalized IR and executable PTX for all three grouped launches.
The offline
verifier checks instruction counts, shared memory size, warp-only barriers
and unpredicated global loads and stores. It also replays symbolic BF16
load addresses through the compiled PTX and compares both FP8 writes for
four boundary tiles per width on both Hopper packing paths. For MLP it
follows both FP8 byte planes and the BF16 inputs of the two amax stores,
including the warp shuffles and shared-memory barriers. For the LM head
it follows the guarded BF16 multiply and the generic FP32 multiply, both halves of each packed BF16 round and
minimum, and the saturating cast into both FP8 outputs at the four
corners of the grid. The Lean proof checks the reference and packed casts
on every BF16 bit pattern. For the sampled gather it follows the
selected raw byte after class indirection to both stores at each grid
corner and for each candidate count. The fused replay also checks inverse
stores on empty, partial and full intervals and the CTA barrier between
clearing and publishing. The recognizers
remove debug metadata and resolve TTIR integer constants before matching.
Compiler changes that alter these fragments require updating the corresponding
checked descriptions.

`--mode cpu` replays the Triton functions with PyTorch's BF16 and E4M3
conversions supplying round-to-nearest value semantics. It compares all scale buffers,
both n-gram outputs and every QKV partial amax bitwise, including all three
native-bank stride layouts. It compares all 224 grouped partials and seven
grouped scales bitwise, and checks the delayed MLP scale multiply and floor
with finite values, infinities and NaNs. The cases cover
partially used tiles, short token histories, and zero, non-finite and
near-floor amax inputs. Both FP8 cache layouts of the aligned Triton fallback
are compared byte for byte on CPU. The launcher checks cover overlapping
buffers, strided partials, a non-32-part scale layout, Hopper selection,
and the packer's split-bank and tail fallbacks. The CUDA mode also compares
the outputs of both native QKV paths and of MLP weight refresh in both
bootstrap and delayed mode on Hopper. The LM-head test checks the tail
fallback with the interpreter and compares the actual Hopper packer when
CUDA verification runs on that device.
The sampled-softmax check compares both output byte planes and the
inverse vocabulary map through the actual launcher for each candidate count.
The tail-average check replays the actual host tick logic over four run
lengths, including the overlapping ticks, seeds, and a declined pair
dispatch. Hopper verification also compares the two FP32 output buffers
of the actual fused kernel with the two original GPU launches.
The final ship check runs the host code with synthetic rank shards and
compares the final BF16 weights across dispatch paths. Hopper verification
compares the FP32 norm bits on the five full-size banks, including a
pattern near FP32 underflow, and checks the grouped BF16 scaling stores.
The sparse-packet check covers empty, all-local, all-remote and split cuts
and partial tiles. The n-gram conversion test includes all FP16 encodings.
It compares NaN classification on the CPU and raw BF16 payloads in CUDA
mode. The offline replay checks each destination's bytes, the scalar
conversion path and the value kernel's out-of-range assertion.
The host test runs both implementations' send and clear methods through
the original cycle callback with synthetic transport and compares the
retained local gradients, the wires and the cleared value buffer. The
clear checks every FP16 bit in the buffer, including the untouched rows.
Its Hopper replay follows mask and address arithmetic for empty, full
and partial tiles and checks the positive-zero words in all three stripes.
The row-fetch replay checks the emitted 128-bit loads and stores for each
column stripe, including offsets near the end of the default n-gram shard,
empty local and remote slices, partial tiles and invalid signed IDs. The host
test runs the installed methods to check both waits, the all-to-all split
counts, cache identity, and untouched words beyond the active prefix.
The moment replay compares both installed host methods and the resulting
FP32 bits and event stamps. Coverage includes mixed stamps, a partial warp,
signed zero and subnormals. The offline replay compares the full PTX
multiplication trees and touched addresses through event 382.
The n-gram Adam check runs both installed host paths through the same
recorded wait, claim, local scatter, peer scatter and Adam call. Offline
replay checks both BF16 atomics and the FMA reduction, decay gate, BF16
weights, FP32 moments and event stamps for a row update.
The value Adam check compares both installed scatter paths, the scaled
FP16 shard and the resulting BF16 weights and FP32 moments. It includes a
signed zero and the ties between consecutive FP16 subnormal encodings.
The Hopper replay checks all 768 atomic inputs and addresses for each
tested local or peer routing tile. It also compares the wire-only packet
and the two owned scatter variants, including their last-read source
stores. CPU coverage checks partial cuts, both density branches and the
materializing fallback across successive events.
The value-embedding backward check replays all four planes, repeated IDs
in alternate orders, and both debug token bounds. The Lean driver links
the captured backward's FP16 table to the row exchange and Adam in
`compiled_value_backward_to_adam`.
The marked path is checked across consecutive events with changing IDs,
duplicates, empty inputs, and transitions across the density threshold.
The scalar scale check covers every FP16 encoding, comparing non-NaN bits
and NaN classification in the CPU interpreter. PTX replay checks flag
writes, masked payload accesses, the exact cast–multiply–cast sequence,
and the CTA barrier before flags are cleared.

For the masked 1024-element consumer, three packed half max instructions
propagate NaN classes from the eight words assigned to each lane. A full
warp vote sends every lane to the scalar FP32-conversion sequence if any
word is NaN, preserving the original NaN behavior. Otherwise the kernel
uses four binary16 pair multiplications by 0.125. Lean checks all binary16
codes and the warp guard, then rewrites the accepted PTX fragment to the
same scalar terms consumed by the original Adam result proof. The dense
1024-element consumer and both 512-element entries use scalar conversion.

The default installation uses guarded reciprocal division for both tile
widths and both clear modes. The warp checks the updated second moments
and the first-moment exponent mask. The Lean range proof derives that the
rounded square root plus epsilon falls within the original divider's
unscaled range. The fallback retains the original scaling logic and
reuses the first-moment mask.

| Value Adam tile and clear mode | Packed reference | Guarded reciprocal | Registers per thread |
| --- | ---: | ---: | ---: |
| 512 masked | 106 | 90 | 22 |
| 512 dense | 105 | 89 | 22 |
| 1024 masked | 321 | 244 | 40 |
| 1024 dense | 328 | 251 | 40 |

The counts are executed SM90 instructions along the ordinary path for
both optimizer periods. Native artifact checks also follow the
square-root and division fallbacks, gradient scaling, moment updates,
weight stores, and workspace clears. Each dispatch reads the current
wrapper and PTX when the launch checks reach them. The adapter reuses
inspection results keyed by the complete text, and Lean proves that each
cache produces the same inspection results as fresh evaluation.

`CombinedTraining.lean` binds the checked input-to-validation driver to
the exact installed kernel and its native certificates. Its `≡[]` proof
uses the common runtime and hardware primitives. The reciprocal square
root uses the published PTX instruction contract.
[NVIDIA PTX reciprocal square root](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#floating-point-instructions-rsqrt)

The value Adam Inductor checker takes the pinned CPU build of PyTorch
2.10.0 and Triton 3.6.0. It captures the AOT pointwise graph with the
original CPU FP32 scalar arguments and compiles both optimizer periods
for Hopper. For the guarded install, the checker places the packed
reference artifacts in `IR_DIR` and the installed kernel artifacts in
`IR_DIR/value-adam-guarded`. The full artifact check disassembles the eight
value Adam binaries with `cuobjdump`. Use
`--packed-adam-reference` when installing a separate packed-gradient
candidate for same-node comparison.

```sh
python3 models/NanoGPT/Track1/check_value_adam_inductor.py "$BASELINE" "$IR_DIR" \
  --candidate "$CANDIDATE"
```

The validation artifact check reads the SM90 SASS in `libtorch_cuda.so`
from Torch 2.10.0+cu128 and `libnccl.so.2` from NCCL 2.27.5. The check
binds the FP32 mean instruction stream and follows the one live FP32
word through the LL, LL128 and Simple ring paths. It runs without a GPU.

```sh
python3 models/NanoGPT/Track1/check_validation_cuda.py "$TORCH_CUDA_LIBRARY" "$NCCL_LIBRARY"
```

`ScaleRefresh.lean` proves equality of every modeled refresh buffer,
including masked fold updates and amax resets. `NgramTiles.lean` proves
unique lane ownership and checks the compiler's reduced sign-hash
constants over all int32 token words. `NgramEmbedPTX.lean` proves the
flat division and full-row lane addressing, unmasked read bounds, and
equality of the forward BF16 output and both backward gradient planes.
`NgramEmbedBinding.lean` recognizes the exact normalized flat and full-row
PTX. The offline replay checks the ordered FP32 FMA and per-lane BF16
products, including both predecessor masks and the 64-bit cache
addresses. `QKVAmax.lean` proves that each part processes the same
ordered FP inputs. `QKVLaunch.lean` proves trip counts, address bounds
and disjoint scratch stores. `QKVTTIR.lean` connects the emitted scans to
equal store addresses and FP32 values, including Triton's commutation of
the V loop's add with +0. `QKVScale.lean` proves that the emitted warp
visits each partial exactly once, that the scale reads the scan's scratch
addresses, and that the published scale is unchanged. It treats NaNs
according to VeriTile's FP32 value profile. `QKVPack.lean` proves equal
read addresses and byte stores, signed i32 bounds, and unique coverage in
both FP8 output planes. `QKVGluonLayout.lean` proves unique register ownership
and equal values and addresses in both layout stores. The warp exchange
proves alignment, disjoint scratch across warps, and safe reuse across
the two ordered phases. Lean checks all 8,192 logical cells for agreement
of their store and load phase and byte address. `QKVNative.lean` proves
that the direct amax and pack reads select the same elements as the
reference reshape, including the signed division and offsets in emitted IR.
`QKVGrouped.lean` proves that the slot and band decoders cover the original
launch points exactly once. Its store theorems identify the selected width
buffer and derive each partial, scale and pair of FP8 stores from the
reference operation on the same raw banks. `QKVVectorLayout.lean` proves
the split input and output addresses and both warp-local byte phases.
The grouped packer stores the even and odd rows in separate phases
and the proof checks each emitted byte permutation. `QKVRefreshBinding.lean`
links the three grouped stages and the compiled per-width Hopper
fallback to the driver's FP8 cache refresh. Deferred flush waits for
the QK and VO banks before reading them and publishes the cache
before training resumes. The final flush waits for the banks and drops
the owed cache refresh, as in the pinned source.
`MLPWeightScale.lean` proves the 528 live entries plus masked padding give
the reference maximum and the exact post-multiply clamp. `MLPWeightPack.lean`
proves the paired-grid bijection, both byte planes and each original
partial store. `LMHeadPack.lean` proves that signed-i32 offsets select the
same entries as the i64 reference, and that both cache planes have one
writer for each byte. The common FP32 multiply and BF16 rounding precede
the concrete E4M3 codec and the verified byte exchange. `FP8E4M3.lean`
specifies finite saturation and ties to even, including the subnormal
and exponent carry boundaries. `SampledGather.lean` proves class-address
lowering, full grid coverage, unique store ownership and safe reuse of
the warp exchange across all four phases. `FPEquiv.lean` composes these
paths and the other
substitutions with an empty FP assumption table. [PTX ldmatrix semantics](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix)

## Source-to-validation proof

`export_driver.py` extracts the pinned trainer's module initialization,
`main` and `train_step` with their branches, loops and validation report.
`export_source_link.py` independently checks the driver against the working
trees and emits the precise source fragments around the substitutions.
The Lean linker checks the current file contents, matches the exact
replacement fragments and the added kernel, and compares the remaining
runtime source file by file. The command above checks 52 common files
and rejects changes to control flow, the data module, the optimizer, or
either implementation outside the approved substitutions.

`TrainingEquiv.lean` proves the root `training_end_to_end` specification with
an empty FP assumption table. Its two sides load the checked source images
and execute the driver with the proved local implementations. The execution
keeps startup and warmup calls, graph capture, restoring weights and optimizer,
embedding-state reset and refresh, timed updates, intermediate or final validation,
tail averaging, and the numerical FP32 validation observation after the
validation reduction. `final_fineweb_validation_equal` projects the last
validation step and loss from the two results.

`ArtifactTraining.lean` supplies the same driver with the five checked
training cross-entropy programs, the paired FP8 backward transpose, and
the seven checked validation heads. It also accepts the ordinary copy
and both fused staging variants for each of the five graph shapes,
as well as the three fused sampled weight gathers, both compiled MLP
weight-refresh stages, the Hopper LM-head refresh and the two Hopper
tied-embedding transposes. It checks both RoPE layouts at each forward
length, the full-width and two narrower backward pairs at each training
length, the byte transpose, and the compact and stationary YaRN decoders.
The matched programs supply both the row and transposed stores, as well
as the expanded YaRN factors. The shared Python/CUDA Graph runtime invokes
the certified head cache body at initial and post-warmup refresh, deferred
flush and graph capture sites.
`training_with_compiled_ce` proves the complete run and FP32 validation
trace equal with an empty FP assumption table. For a fixed training launch,
graph replay reads the decoded loss and gradient bytes. The paired backward
transpose uses four 128-bit global loads per lane in place of eight 64-bit
loads and three warp barriers in place of seven. Its two shared-memory
slices use 8 KiB per block, up from 4 KiB. The SM90a assembly has 115 static
non-NOP instructions, up from 113, and both variants use 32 registers with
zero spills. These counts come from compilation for Hopper. Validation
accumulation takes the BF16 head inputs and
replays the captured projection wrapper and loss kernel using the batch's
original token and target IDs.
The dynamic training CE branch uses the paired source evaluator. Common
matrix multiplication, collective transport and the other runtime calls
remain at the numerical driver boundary.

The MUDD coefficient pass matches ten consecutive Inductor launches on
the same stream. It retains the original BF16 allocations and fills
them in one Triton launch. The Hopper grid uses the y-coordinate to
select the output pointer. The coefficient kernel computes the row
with unsigned division by twelve and forms the projection address
from the packed linear index and the 108-word row-stride difference.
`MuddCoefficientsBinding.lean` checks those addresses, the output pointer
selection, and the ordered FP32 multiply and FMA followed by the paired
BF16 conversion.
The graph runtime can read the coefficient projections and resume the
forward with the resulting ten tensors.

On the separated layout, when the ten coefficient kernels feed a
one-warp Hopper RMSNorm and training-head quantization, the wrapper
replaces all eleven launches in one pass. The fused Gluon kernel computes each coefficient once
per group and broadcasts the BF16-rounded value to its consumers.
The two BF16 bias words and two BF16 MU words for each adjacent plane
pair are loaded as one 32-bit word each. Both specializations preserve
the plane order through the original FP32 accumulation.
It writes the original ten coefficient tensors, the FP32 mixed residual,
one FP32 statistic per row, and the E4M3 activations. The 32-lane layout
preserves the original FMA and reduction order. Offline
SM90a assembly uses 72 registers per thread and zero spills, down from
255 registers and 68 bytes each of spill stores and loads in the
reference training RMSNorm launch.

`MuddNorm.lean` proves the shared-memory coefficient routing, store
ownership and the batched `≡` of the two paths with no FP assumptions.
`MuddNormBinding.lean` checks the exact emitted PTX. The wrapper can save
the inverse RMS in the existing statistic allocation when the original
square sum has a single consumer on the same stream and the consumer
matches a supported RMSNorm backward. The backward loads that FP32
word and preserves the original reduction, BF16 residual gradient, and
optional E4M3 rematerialization. Its vector accesses reuse the widened
row base. In the 16-word layout the tail loads provide positive zeros
for padding, and the tail products keep their own FP32 rounding before
the reduction. The wrapper retains the square-sum path
when this producer-consumer match fails.

The offline symbolic replay checks 1,657 forward stores and every
backward store per row with a symbolic row address. `MuddBackward.lean`
proves the store and load address and the unchanged memory outside the
private statistic allocation. `MuddBackwardBinding.lean` checks the
cached forward and all three backward PTX specializations and proves
`cached_equiv` with an empty FP assumption table. With the separated
boundary, the first sampled-head configuration can retain the original
E4M3 activation for the paired CE transpose. It removes a 12 MiB E4M3
write per rank at that shape and extends the lifetime of the forward
allocation. The retained backward uses the same 16-channel-per-lane
reduction layout.
`MuddNormHeadReplay.lean` proves that the retained bytes agree with
rematerialization under IEEE value semantics. The driver passes the
saved inverse to the modeled backward between the gradient GEMMs and
supplies the resulting BF16 gradient to the trunk.

For a captured graph, `GraphStageExecution.lean` replays all five input
copies and both slot writes before invoking the forward. Its member
packet carries the FineWeb hash tensor and directory from the exact
cycle selected by the trainer. The proof recovers the pending slot
array, graph inputs and unchanged bytes for every CTA order.
When the graph already has resolved slot IDs, `GraphPlainPTX.lean`
decodes the ordinary copy, including the four-word slot transfer.
`GraphPlainMemory.lean` proves that the sorted interval guard separates
the source and destination buffers. Every complete CTA schedule copies
all six buffers byte for byte and preserves memory outside the
destination intervals. The driver uses this execution for a physical
copy packet before replaying the graph.

`TrainingCELaunch.lean` proves that every step in the default schedule
selects one of the five fixed launches using the scheduled stage's prediction
count and the source's candidate-count transitions.
The sampled path takes both FP8 layouts and the class-to-position map
from the same checked TTGIR/PTX program. It passes both results to the
training driver before graph replay. The Lean proof uses the uploaded
ascending class order, including the final partial inverse-map tile.

Both head wrappers use `activation_fp8.py` for aligned eager Hopper
inputs with the pinned activation scale. Each lane loads four BF16 words
and stores four E4M3FN bytes, keeping the intervening BF16 rounding in
registers. At 49,152 tokens this removes 144 MiB of eager temporary reads
and writes per invocation before cache effects. Under `torch.compile`,
the wrapper passes through the original division and cast so Inductor
can fuse them with RMSNorm and keep its FP32 intermediate values.

`HeadActivation.lean` checks the conversion for every BF16 magnitude
and preserves the sign separately, including the NaN encoding above the
finite rounding boundary. `HeadActivationBinding.lean` binds the gain,
the ordered comparison, and the byte permutation to the emitted PTX and
proves that each lane's four stores are in range and disjoint.
[ATen scalar division](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
[ATen FP8 copy](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/Copy.cu)

The typed graph hooks in `TrainingHead.lean` express the BF16-buffer
quantization path and send the FP8 activations and weight view to the
forward GEMM. The CE output keeps
the E5M2 derivative for both backward GEMMs, whose scale operands and
accumulation mode are explicit. The driver retains the saved operands
between the graph calls and consumes them during backward. The sampled path uses the saved
class-to-position map and the checked BF16 densify to publish the dense
weight gradient. The shared matrix multiply is a backend operation
on those proved inputs.

```sh
python3 models/NanoGPT/Track1/check_graph_stage.py "$BASELINE" "$CANDIDATE" --ir-dir "$IR_DIR"
python3 models/NanoGPT/Track1/check_validation_ce_inductor.py "$BASELINE" "$OUT/validation-ce"
PATH="$CUDA_HOME/bin:$PATH" lake env lean --run models/NanoGPT/Track1/CheckSource.lean --artifact-execution \
  "$BASELINE" "$CANDIDATE" "$IR_DIR" "$OUT/validation-ce"
```

`CETranspose.lean` proves the two operand ranges cover the combined grid
without duplicate writes. It also proves that the widened loads and the
two shared-memory slices deliver each byte to the corresponding transposed
position, including the release and reuse of each slice.
`compiled_ce_weight_operands` checks every byte
in the two buffers presented to the shared weight-gradient GEMM. The
backward graph replay in the driver receives this transpose operation
through its numerical runtime interface.

`TrainingCE.lean` expresses the packed halfword sigmoid row, each
thread's ordered exponential sum, the five warp shuffles, the eight
warp sums, the MTP and prefix loss, and the base and corrected E5M2
gradient words. `TrainingCEConversions.lean` defines the E4M3 and FP16
widening, nearest-even FP16 and E5M2 narrowing, E5M2 saturation, and the
four binary64 launch argument conversions. It uses the CUDA software
NaN encodings and leaves the four approximate PTX operations in the
native interface. The packed decode and round preserve the two lanes
independently for every operand in this model. `table_lookup_exact` and
`table_scan_exact` establish the table values and sequential sum at all
four widths. `published_table_words` proves little-endian readback for
the packed FP32 pair and exponential contribution.
`staged_gradient_factor` proves that every lane reads the reference
factor from shared memory after raw-logit staging. The probability
publication proofs show that a lane can read its original scaled logit
while other lanes write their probabilities, and that after the barrier
all writes give the same table in any order. They preserve the sigmoid
words, the raw logit codes, and the gradient factor.
`base_gradient_publication` proves readback for all cached E5M2 bytes.
`row_cache_stores_commute` proves that the probability and base-gradient
writes can commit in any order. `packed_shared_code_exact` derives each
index of the base-gradient lookup from the corresponding staged logit
byte. The fixed branch of `training_end_to_end` uses the staged byte model
for both lookups.

`TrainingCEInitBinding.lean` follows the compiled E4M3 unpack, sigmoid,
FP16 rounding and FP32 exponential contribution up to the first barrier.
It checks both shared table addresses and the lane-zero gradient factor.
`compiled_ce_initialized_table` identifies these writes with the values
read from staged memory. For the 10,240-class configurations the
initialization trace also checks lane-zero publication of each MTP
target before the block barrier. `compiled_ce_cached_target` proves
the exact cache word for each slot. Both the loss and sparse gradient
decoders check their cache loads against that publication. The
unsigned vocabulary test and signed extension both agree on the
same valid index or −1 sentinel.

`TrainingCEGatherBinding.lean` also follows the logit scan up to the first
warp shuffle. It checks the incoming eight-byte chunks, all eight byte
selectors, the shared staging addresses and the exact order of FP32
additions. `scan_code_cover` establishes coverage of the input row,
including the full-width tail. `compiled_ce_scanned_leaders` uses the
decoded lane sums as inputs to the compiled warp reduction.

`TrainingCEGatherBinding.lean` follows the compiled byte selectors, shared
cache addresses and four halfword operands of each packed gradient store.
It checks the three barriers, each lane's base-byte publication address,
and that register definitions precede their uses on every control-flow
path. The checker accepts all four unrolled fragments in each of the five
artifacts. Lean executes the compiled loop counters and address arithmetic
for all 256 lanes with a symbolic row. The full-width tail writes only
complete eight-byte chunks inside the row. `compiled_ce_base_grid` proves
that the decoded schedule covers every vocabulary column, and
`compiled_ce_base_chunk` connects each issued chunk to the staged base
gradient model.
`TrainingCEPublishBinding.lean` decodes both cache stores from the PTX
between the sum and gather barriers. It traces the weight and prefix
loads to their CUDA parameters and the sigmoid, scale and warp sums to
their shared addresses. The two vector loads supply warp slots 0–3 and
4–7 to the ordered sum. `compiled_ce_cache_values` proves that the
decoded FP32 probability and rounded E5M2 byte agree with the staged
memory for arbitrary payloads. The approximate PTX operations use the
same native interface as the numerical reference.
`TrainingCEReductionBinding.lean` checks the five shuffle offsets, the
full participation mask and clamp, the FP32 add chain, and the leader
predicate and shared address. `compiled_ce_reduced_cache_values` feeds
the eight decoded leader words into the cache proof. The shuffle
semantics keep the lane's own value when a requested source falls
outside its warp. [NVIDIA PTX shuffle semantics](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync)

`TrainingCELossBinding.lean` checks the compiled loss branch for each live
prediction count, each prefix-target case and each lane-zero outcome.
The targets within the vocabulary and their FP32 values remain symbolic.
The trace checks the CUDA parameters, staged logit bytes, sigmoid
addresses and the row's loss destination. `compiled_ce_row_loss` connects
the resulting ordered FP32 FMA chain to the reference row loss.

`TrainingCESparseBinding.lean` decodes the epilogue reached after the packed
stores. The checker covers every live-prediction count, target equality
pattern, owner-lane assignment and prefix case. It verifies the ordered
FP32 weight sums, the sigmoid and probability reads, the E5M2 conversion,
and the byte-store destination. `compiled_ce_sparse_write` relates those
writes to the staged gradient's column, first eligible slot, lane and
byte. The target class and owner are derived from the actual row.
`compiled_ce_row_gradient` combines the packed stores and sparse epilogue
to recover the reference E5M2 byte at every vocabulary column.
`TrainingCEExecution.lean` composes the decoded shared-memory
initialization, logit copies, reduction and cache stores with both
decoded outputs. The launch predicate discharges the softcap and head
gradient-scale conditions for the linked driver call.
`compiled_ce_backward_operands` passes these decoded row bytes through
the certified transpose to the head-backward GEMM operands.
`compiled_sampled_ce_dense_gradient` follows the GEMM result through the
checked inverse map to the dense BF16 weight-gradient stores.

`first_correction_slot` identifies the first occurrence of every target
that requires a patch. `correction_writer_owns_base` places
the correction on the same lane as the base store. The proof in
`TrainingCEGradientStores.lean` allows arbitrary interleaving of lanes
as long as each lane issues its base stores before its corrections.
`linked_committed_training_ce` combines shared memory staging with output
readback, proving agreement with both graph outputs and preservation
outside the gradient row. `linked_ce_weight_gradient_bytes` follows
the committed E5M2 byte through the fused transpose to the
weight-gradient GEMM operand. `linked_ce_backward_operands` relates
the entire input-gradient and weight-gradient GEMM operand layouts.
`linked_sampled_ce_dense_gradient` composes the common weight-gradient
GEMM with the checked inverse map and dense BF16 stores.

The NVRTC checker uses the installed compilation wrapper and PyTorch
2.10's CUDA flags. For all four vocabulary widths it checks every
thread with symbolic row and FP values. It compares every halfword
write, the ordered sum and all live registers at the first shuffle.
The rest of the PTX matches after register and label renaming, including
the loss FMA, E5M2 writes and target corrections. The checker also
derives the default width/count pairs from the stage configuration,
checks fixed and generic launch selection, and replays each specialized
kernel through the warp reductions and block barriers. A fixed launch
requires the default binary64 B and C arguments and the gradient scale
`(0.75 / 8) / 448` from the model's head initializer. The sampled variants
also require the default A argument. Other scalar values use the
generic launch. Lean proves that the constants give the same FP32
inputs. The scale uses the literal `0x395b6db7` and preserves the lane-zero
`rcp.approx.ftz.f32` operation. The checker rejects changes to the Python
guard, the pinned model initializer, and the compiled literal.
The five SM90a kernels each remove seven or eight non-NOP instructions
compared with the previous fixed launches. Register counts stay at 32
for the sampled variants and 40 for the full vocabulary, with no spills.
For each fixed variant the checker verifies both tables at all 256 indices before the first barrier,
then checks every staged logit byte and the ordered lane sum. After the
sum reduction, it checks the in-place probability computed by each lane
from the row logsumexp and the E5M2 byte computed from that probability
and the row's weight sum before allowing any gradient lookup. The CTA
traces check the loss and all base and corrected gradient bytes with
distinct targets, repeated targets, multiple columns owned by the same
lane, optional prefixes and truncated last-row predictions. They verify
that the base and correction for each byte share one writer, and reject
shifting the correction into another lane's chunk. Lean proves that the
fixed lists visit the same target slots in order and the seed addition
preserves its bit pattern.
Each fixed path selects a two-chunk sum scan and a four-chunk gradient
loop. The checker follows the loop counters and symbolic values for
all 256 lanes. It rejects removing the initialization barrier,
changing the exponential table's multiplier or the probability
exponential's argument scale, swapping the two words in the packed
lookup entry, or publishing the gradient factor or a base byte at the
wrong address.
With `--ptxas` it also
compares register use against the direct fixed-count variants.

```sh
for phase in gram trace rails epilogue; do
  python3 "models/NanoGPT/Track1/check_anvil_${phase}.py" "$BASELINE" "$CANDIDATE" "$IR_DIR/anvil/$phase"
done
python3 models/NanoGPT/Track1/check_training_ce.py "$BASELINE" "$CANDIDATE" "$IR_DIR/training-ce" \
  --nvrtc "$CUDA_HOME/lib64/libnvrtc.so" --include "$CUDA_HOME/include" \
  --ptxas "$CUDA_HOME/bin/ptxas" --nvdisasm "$CUDA_HOME/bin/nvdisasm"
```

`SampledInverse.lean` derives sorted candidate IDs from the host bitmap
and proves that the fused launch writes the same signed position at every
class. The driver invokes the checked gather and inverse map after the
upload wait. `compiled_sampled_forward_backward` links both gathered FP8
byte layouts to the original gather and proves that the saved inverse
produces the reference BF16 gradient at every dense output address.

`TailLerp.lean` computes all four tail-average schedules from the step and
run length and carries both accumulator buffers and their seed flags
through each update. The driver executes that update at the pinned tick
site. The source link checks the complete tail-average module, including
the final blend and gather order. The offline replay checks load and
write addresses and exact FP32 expressions for both kernels at the
edges of the VO and MLP shards, and the Lean recognizer checks their
normalized TTGIR and PTX.

`TailShip.lean` executes the ordered tail blends, rounds each shard write
to BF16, and assembles received shards in rank order at the final driver
site. `TailNorm.lean` models the four per-thread FP32 norm accumulators,
the block reduction, the final reduction of staged block values and the
shared square-root operation. The direct and materialized paths supply
the same FP32 values to that sequence. Lean also checks the FP32 scalar
round trip, the five-bank router and each tile's packed output offsets.
The compiler replay checks complete store expressions at each bank
boundary. The ATen reduction model shares the contraction mode of the
two source instantiations. CUDA conformance of the compiled ATen
instantiations is a separate check.
[ATen CUDA reduction](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/Reduce.cuh)

`FineWebBatches.lean` proves that completing the asynchronous BOS scan
leaves a BOS-aligned batch's input and target tokens identical when the
visible index can finish the batch with a BOS entry left over. It also
proves a counterexample at the boundary of the visible index, where only
the full index has enough entries to finish the batch. The validation
driver reads ordered shard bytes, checks the little-endian header and
payload size, and constructs each rank's unaligned input and target
windows, padded BOS boundaries, and signed int32 n-gram row hashes.
The decoder proves that the four file ranges reconstruct the contiguous
payload. The validation forward call receives the materialized batch, and
the cross-entropy model takes its mask selectors and labels from those arrays.
The source linker checks the reader, boundary sizing, index switch,
and token slices against the pinned loader.

`HostSchedule.lean` computes the pinned Python schedule at the rounding
sites of its binary64 expressions, then narrows the MTP and prefix rows
to FP32. The driver uses its stage geometry for batch selection, its
forward controls for window and loss-weight changes, and its learning
rates, momentum, rail weights, Adam cadence and split step for optimizer
dispatch. Each per-step weight row keeps its original length and words
in a disjoint, 16-byte-aligned slot. The default schedule makes two
tensor uploads during setup. The check calls the installed constructors
and the pinned optimizer dispatch on CPU and compares them to Lean for the
default run plus three overridden base counts.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 models/NanoGPT/Track1/check_host_schedule.py "$BASELINE" "$CANDIDATE" --benchmark
```

`graph_stage.py` batches the six pre-forward staging copies into one Triton
launch for the five default rank-local layouts. It writes the captured buffers
at their existing addresses, bumps their version counters and queues the
original forward replay after the copy. Other layouts use the original six
`copy_` calls. `GraphStage.lean` proves unique ownership and preservation of
every word, including the two FP32 weight fields. The driver invokes that copy
at the checked source call site. `GraphBackward.lean` expresses ANVIL gradient
attachment, the ordered FP32/BF16 Adam first-copy and accumulation paths, and
the direct or ring-buffer n-gram gradient consumed by the optimizer.

The offline checker executes the installed launcher with instrumented
buffers, replays each SM90 PTX store by thread, and exercises the installed
backward wrapper against CPU tensors. Eight-GPU wall time requires the paired
benchmark.

```sh
python3 models/NanoGPT/Track1/check_graph_stage.py "$BASELINE" "$CANDIDATE"
```

`batch_packet.py` transfers the four fields of a packed, BOS-aligned training
batch with one asynchronous H2D copy. Each typed view starts at a 16-byte
boundary, and the same per-slot event gates reuse of the pinned allocation.
Validation and fallback still use separate device copies. The proof in
`PinnedBatch.lean` covers disjoint byte intervals, capacity and the four
recovered arrays. The scheduled stream applies the upload when it
materializes a training batch, before caching it for lookahead.

```sh
OMP_NUM_THREADS=1 python3 models/NanoGPT/Track1/check_batch_packet.py "$BASELINE" "$CANDIDATE"
```

`FineWebTraining.lean` models the BOS-aligned training stream from shard
bytes through rollover, geometry changes, and the ordered batch cache.
The driver opens separate streams for warmup and timed training, then
uses the stream at the `peek` and `take` sites. The CPU packing proof
shows that subtracting one from the final prefix sum is equivalent to
subtracting one from the last document length before the scan. The two
paths concatenate the same source slices, retain the shifted target
window, and place identical virtual attention boundaries. The offline
replay uses the installed Python scheduler and loader for all eight
ranks on synthetic shards.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 models/NanoGPT/Track1/check_fineweb_packing.py "$BASELINE" "$CANDIDATE" --benchmark
```

`NgramRowHash.lean` proves that the CPU loader's NumPy remainder agrees
with signed PyTorch remainder for every int32 input. The proof accounts
for the int64 scratch, arithmetic shift, int32 narrowing, and sign
correction. Both channels retain the original wrapping products and row
offsets. The bigram at position one still hashes the reserved previous ID.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 models/NanoGPT/Track1/check_ngram_row_hash.py "$BASELINE" "$CANDIDATE" --benchmark
```

`ValidationCE.lean` computes the per-token validation loss from the BF16
LM-head projection and the tokenizer's packed canonical mask. A common
Triton launch scans the vocabulary for each of eight 32,768-row slabs
with the emitted FP32 softcap,
masked score, online softmax recurrence, maximum and sum reductions,
libdevice logarithm polynomial, and target gather. Lean proves that
each vocabulary column has one scan slot, each slot has one reduction
owner, and the shuffle offsets stay within their lane group. The
cross-entropy primitive interface supplies the results of the PTX
`ex2.approx` variants and `div.full.f32 1, x`.

`check_validation_ce_inductor.py` extracts the pinned evaluation loss,
including the LM-head matmuls and concatenation, and lowers it offline
with PyTorch 2.10 and Triton 3.6. It checks the source and destination
slices of all eight matmuls and loss outputs. The five default Hopper
configurations and two occupancy variants share the same vocabulary scan
and masked tail state in TTGIR. At the edge CTAs the PTX replay checks
the mask bits, projection addresses, the first and next online updates,
the complete ordered reduction and logarithm, target gathers, and the
output for ignored targets for every slab. The replay covers this loss
subgraph. The full model's launch and layout conformance remains separate.

`ValidationTargetBinding.lean` traces the target class and input token
to their launch buffers and checks the BF16 projection and canonical
mask addresses. Its two mask modes prove the same byte and bit indices.
It then checks the widening, softcap, and masked value used by the loss.
The batch decoder preserves the raw `-100` word as an ignored target.
`ValidationExpBinding.lean` recovers the vocabulary loop counter,
both exponential factors of each online update, and the per-slot
rescale exponentials. It checks the ordered maximum with its left NaN
preference, the negative-infinity guard, and the inputs of the FP32
FMA that updates each partial sum. The loop state starts with zero and
negative infinity and retains both words when the tail predicate is false.
Its two counter forms give the same `ceilDiv` trip count as the row model.
The rescaling check pairs every carried maximum and sum with its
compiled multiply or FMA.
`ValidationScanBinding.lean` follows the packed BF16 projection from
the CUDA parameter through scalar or matrix shared-memory transposes
to the FP32 softcap in each online update. It follows the input token
load, row stride, vocabulary byte counter, packed bit tests and
masked-score select for the same slot. The four-column tile has two
iteration phases because the byte advances every other iteration.
The checker checks each phase's load and update predicates against
the valid columns, then proves the complete loop's maximum and sum.
`ValidationReduceBinding.lean` replays each CTA's butterfly shuffles,
shared stores, barriers, shared loads, and ordered FP32 combines.
The sum and maximum read by each output lane have the reference
expression tree for arbitrary loop-carried FP32 inputs.
`ValidationLogBinding.lean` decodes all eight compiled logarithms in
each configuration. It verifies the normalization, polynomial,
nonfinite branch and zero select. The live-row finish proof then checks
the two subtractions and the final FP32 addition of positive zero
from the reduced sum, maximum and the decoded target logit. The decoder traces
the −100 test to the 64-bit load from the global token index. It checks
that each CTA writes only from the owner lane for each row and that the
payload reaches the correct slab pointer at the slab-local FP32 offset.
`ValidationRowBinding.lean` joins the checked loop states to the
reduction and loss store for each row. It proves the resulting FP32
word agrees with the model for in-vocabulary input tokens and
arbitrary BF16 projection and packed mask words.
`ValidationBatchBinding.lean` uses the driver's uploaded token and
target words, assembles the output in slab order, and connects the
compiled row values to the driver step that adds the FP32 batch mean.
`ValidationProjectionBinding.lean` checks the captured Inductor
wrapper's BF16 GEMM input slices, shared transposed weight, projection
ports and FP32 output aliases. The proof gives each slab GEMM the same
operands as the source evaluation and composes its BF16 outputs with
the checked cross-entropy and local batch mean.

```bash
python3 models/NanoGPT/Track1/check_validation_ce_inductor.py "$BASELINE" "$OUT/validation-ce"
lake env lean --run models/NanoGPT/Track1/CheckSource.lean --validation-ce "$OUT/validation-ce"
```

`ValidationLoss.lean` computes the reported FP32 loss from those
per-token cross-entropies. At the default 262,144 tokens per rank, the
ATen mean uses 32 CTAs of 512 threads, four separate FP32 accumulators
per thread, and descending shuffle offsets. Lean proves each aligned
input word has exactly one owner. The driver adds the five batch means
in Python loop order and multiplies by the FP32-rounded host reciprocal
of five. The NCCL FP32 average scales the first payload by one eighth and
uses an FMA with each subsequent ring payload, including rank zero. The
driver publishes the computed word only on rank zero. The runtime
provides the LM-head projections and the peer contributions in the
observed ring order.
[ATen mean dispatch](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/ReduceMomentKernel.cu)
[ATen mean reduction](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/Reduce.cuh)
[ATen division by a CPU scalar](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
[NCCL FP32 AVG](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/enqueue.cc#L2075-L2115)
[NCCL Reduce ring](https://github.com/NVIDIA/nccl/blob/v2.27.5-1/src/device/reduce.h)

`SparsePackets.lean` proves the prefix and suffix wire routing, disjoint
ownership of local and remote outputs, and the exact selected-row reset.
The full-row launch assigns each live word to one CTA, warp, stripe and
lane word. The checked packet and scratch programs supply the sparse
optimizer calls in the end-to-end execution. Their certificates also
cover the n-gram peer payload before its scratch is cleared.
It proves uniqueness of the generated per-plane want list, absence of
duplicate clear destinations, and preservation of all other gradient bits.
The optimizer site in the driver computes the original warmup and timed
event cadence. There are 382 events in the default timed schedule.
At an event the model sends the value gradients, prepares and sends the
n-gram gradients, and then clears the value rows. Reversed split counts
are bound to the original source. `SparseLanding.lean` proves the
peer-prefix and peer-suffix index mapping, the local lookup, the wide
shard address, and preservation of the untouched cache words. The driver
executes the modeled pulls at the forward and FineWeb validation sites.
`RowOwners.request` uses precomputed int32 owner boundaries for int32
want arrays when the table extent fits. This avoids widening the entire
want list. With 392,319 synthetic unique IDs, the local median for
`request` was 64.50 → 6.37 microseconds with the collective call stubbed.
`RowExchange.lean` proves that narrowing preserves the lower-bound cuts,
that the swapped split counts recover each requester's segment from the
owner's concatenated reply, and that cache rows contain the words from
their owner shards. The eight-rank CPU Gloo check exercised the installed
requests, ID and BF16 exchanges, cache landing, and reverse gradient
routes, including empty peer and local segments.
`RowGradientExchange.lean` reads the packed FP16 gradients in wire
order and applies the BF16 conversion. It proves that the reverse
exchange delivers each payload at the position of its ID in Adam's
`asked` array. Contributions for the same ID from different ranks
remain separate inputs to the Adam scatter.

Non-event graph backpropagations now enqueue their FP16 slot scatter on
the compute stream before the next backward. The update step's scatter
still follows the value-gradient send. The default schedule removes the
four-entry BF16 gradient ring of 576 MiB per rank and its 812 copies.
These copies amount to 180,589,953,024 logical bytes of reads and writes
per rank before accounting for caches. The final two steps still run
backward and their n-gram sinks have no subsequent table update.
`NgramAccumulation.lean` proves equality of the per-step FP16 sums for
every ordered atomic schedule. The compute stream consumes each sink
before the next replay can overwrite it. The source checker matches the
BF16-to-FP16 conversion and all three stripes of the vector atomics.
The driver computes the sum from the typed tape before packing, and
`compiled_ngram_cycle_update` composes it with the reverse row exchange
and the Adam update.

The FP16 accumulation buffer starts at positive zero after reset. The
Hopper packet kernel clears peer rows after loading them for the BF16
wire. It preserves the owned slice until the Adam merge reads it and
clears it in the same launch. The fallback clears the live prefix after
that last read. `compiled_scratch_reuse` proves equal packets and a clean
allocation across arbitrary changing cycle lengths and fused or fallback
events. The source checker matches both GPU consumers and their zero
stores. Each eligible event saves a standalone clear launch.
The owned scatter uses one four-warp CTA per four rows on aligned Hopper
buffers. Each warp reuses the source ID and destination lookup across
all three stripes, with an eight-element BF16 atomic add per stripe.
For four rows the SM90a artifacts change from four CTAs and 192 static
non-NOP instruction slots to one CTA and 101 slots. Registers per thread
increase from 14 to 23, with zero spills. Each BF16 address retains its
ordered scalar adds. The artifact certificate connects the tiled word
selection to the local gradient merge in `NgramAdam.lean`.
For eligible training pulls, the prep thread builds a directory of lower
bounds at 32,768-row boundaries. The directory and sorted want list share
one asynchronous transfer and one slot event. `ngram_slots.py` then searches
at most 255 entries in eight masked probes per query. The four query sizes
are primed before timing. Cold, validation and oversized-bucket pulls use
ATen search. `NgramSlots.lean` proves that the directory brackets the same
left insertion position as global search and that eight probes suffice.
Adjacent slices of the offsets give the same bucket widths used to reject
an oversized bucket. On synthetic sorted want lists of 32,768 to 393,216
rows, the slice subtraction cut median CPU directory construction time by
1.2 to 1.6 microseconds.
The cycle also sorts its newly concatenated row buffer in place and
writes the adjacent-row comparison into one preallocated mask. With
393,216 synthetic input rows split over four batches, one local run
measured median sort and deduplication time at 14.01 → 12.75 ms for
uniform IDs and 12.80 → 11.28 ms for repeated IDs. The check compares the
installed sort with the pinned implementation and verifies that it leaves
the batch arrays intact.
The installed launcher check replays the four SM90 PTX query shapes.

```sh
python3 models/NanoGPT/Track1/check_ngram_slots.py "$BASELINE" "$CANDIDATE"
```

For a reused forward graph with a compatible slot directory, the trainer
hands the read-only slot request to `StepGraphs.forward`. In one launch,
the copy CTAs stage five input arrays and the lookup CTAs publish both
the static slot array and the fresh slot array held for gradient accumulation.
An uncaptured graph resolves the request before cloning its capture inputs.
The driver holds the request between row routing and the first forward
reader, and `GraphStage.lookup_stage_equal` proves equality of both output
views. The fused artifacts are primed before timing.

The training-only variant uses the n-gram row IDs already computed by the
FineWeb batch packer. The proven hash range is `[0, 84602880)` after decoding
the uploaded words as signed i32 values. The launcher uses it to drop the
query range checks and the terminal-directory load. The general lookup
still handles negative and out-of-table queries. The five compiled SM90
shapes each use 16 static global load instructions, down from 17. The
assembled SM90a binaries each have eight fewer non-NOP instructions,
with 18 registers per thread and no spills in either variant.
The masked load also uses the query row as its default. Once the interval
closes, the comparison is false and the midpoint equals both boundaries.
The final kernels remove eight more non-NOP instructions per shape without
changing the register count or the issued lookup addresses.

When the cycle's directory is available, the trainer also records weak
references to the device row tensors from the batches that built its
sorted want list. The fused launch uses inclusive, unmasked search when
the query is the live tensor at the current cycle position. Other queries
keep the masked search. The member search uses binary lifting with
unsigned clamping at the last entry of the bucket. It removes 51 non-NOP
SASS instructions in each of the five SM90a launch shapes, with 18 registers
per thread and no spills. It issues all eight dependent loads for every
query. Comparison with the masked path on H100 remains a timing question.

`GraphStageMemory.lean` interprets the fused launch at byte addresses. The
proof derives disjointness from the interval check, bounds the lookup reads
using the uploaded directory, and proves unique ownership of every copy and
slot byte. A permutation of CTA execution leaves the result unchanged. It
publishes the same two i32 slot arrays and five unchanged input views to the
driver's graph-stage port and preserves memory outside the destinations.
The source certificate also links the FineWeb packer and lookahead cache
to the specialized replay. The proof derives the row words from the
little-endian bytes of either a packed upload or a separate row transfer.
It uses the launcher's checked buffer layout and the prepared directory.
`NgramSlotPTX.lean` proves the bit-width arithmetic, read bounds and
lower-bound result of both searches. `GraphCopyPTX.lean` proves the vector
addresses, scalar read masks, and unique destination bytes for the copy
threads. `GraphLaunchPTX.lean` combines their stores with both slot buffers
and proves the same memory result for each CTA order.
`GraphSlotPTXBinding.lean` checks the full normalized PTX of both branches
in all five shapes. The linked-source theorem `scheduled_compiled_replay`
connects the checked PTX to a scheduled training batch, the
captured input views and pending slots, and the memory outside the writes.
The offline verifier also replays the PTX for all five shapes.
`NgramDirectoryPacket.lean` reconstructs the wanted IDs and directory from
the resident device packet and proves the batch transfer preserves both
views. `GraphStreamBinding.lean` checks the source submission sites.
`scheduled_packet_stream_replay` composes the packet and PTX proofs with
`CudaStreamOrder.lean`'s FIFO semantics. The forward observes the completed
staging result. The pinned-slot proof permits reuse once that slot's current
upload event completes. The proof covers this stage-to-forward boundary
using the runtime's documented stream semantics. The internals of the
captured forward and the other runtime ports retain separate obligations.
[PyTorch graph replay](https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/cuda/CUDAGraph.cpp#L184-L199),
[CUDA stream ordering](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)

`scheduled_member_stream_replay` derives membership from the captured
FineWeb cycle, the tensor identity guard, and the identical row words in
the batch and directory transfers. It replays both uploads in stream
order from the pre-upload memory and proves that the directory packet,
including its padding, preserves all aligned staging input views. It
proves bounds for all unmasked loads, equality of the forward observation
and pending slots, and preservation of bytes outside the staging
destinations. Every staged slot is within the loaded rows of the cycle,
resolves to the query's global row ID, and reads that ID's word from the
landed owner data.
`landed_staged_ngram_replay` uses the cache written by the landing launch
as the embedding's BF16 input, and checks the original and full-row
forward and backward kernels and their read bounds on that input.
For the eight-rank timed run,
`timed_cycle_capacity` derives at most 393,216 rows after sorting and
deduplication, from at most four batches of at most 49,152 tokens each.
The configured validation extent allocates 524,288 cache rows per rank.
`hopper_training_stream_bound` obtains the token limit from the
trainer's stage geometry, and both lookahead and consumption preserve it.
The recognizer checks the full member PTX
for all five launch shapes and rejects changes
to the bucket shift, clamp, step sizes, probe addresses and final stores.

```sh
python3 models/NanoGPT/Track1/check_fused_slot_stage.py "$BASELINE" "$CANDIDATE"
```

The optimizer hooks run the two Adam updates before starting the row
exchanges and land the n-gram cache before marking the next cycle live.
`MomentReplay.lean` proves that each lane's masked warp loop applies
the original chronological suffix of FP32 multiplications. It also
proves disjoint destinations for distinct request IDs and preservation of
rows outside the owned cut. The driver evaluates the replay after both
row sends and before the next cycle's cache land.
The beta2 history starts with an empty live prefix after reset. Lean
derives both FP32 words from the pinned FP64 schedule and proves that
appending each event preserves the prefix invariant, even when the suffix
still contains warmup data. The source and Hopper artifact checks bind
the positive-zero comparison, stamp store, and owner branch. In
`TrainingEquiv.lean`, `compiled_moment_replay_since_reset` and
`compiled_ngram_cycle_since_reset` use the derived invariant to prove
equivalent replay and Adam updates for the installed training paths.
`AdamEntries.lean` proves equality of the three index views and the
alignment and disjointness of the peer mirror. `NgramAdam.lean` executes
the launch-coefficient arithmetic, claim order, two BF16 gradient
scatters, missed moment decays, emitted FMA reduction and cautious
weight update. The execution profile supplies the precise effects of
Hopper's approximate square root and full division instructions.
`prepared_state_equal` allows the two launch sequences to claim different
slots for a repeated row. With the same BF16 scatter order per channel,
each chosen slot receives the same ordered row gradients, so weights,
moments, timestamps and beta history agree.
`ValueAdam.lean` models the second sparse optimizer branch through
local and peer FP16 atomics, binary16 gradient scaling, the two bias
corrections, both FP32 moments, and the BF16 weight update. Its
subnormal counterexample requires preserving the half store before
the optimizer consumes the gradient. `compiled_value_adam_event` derives
the owned interval from the two host lower bounds, proves each direct
read's address and gradient word, and checks that the delayed clears
restore the original allocation after the Adam update. The reference
wrapper passes learning rate and weight decay by value after reading its
CPU FP32 tensors. Both Hopper launch tiles use the same ordered decay FMA.
Lean computes round-to-nearest square root using integer arithmetic and
uses the shared native division effect.

The consuming kernel combines the FP16 scale, Adam update and workspace
clear. It leaves the materialized path available for unmatched launches.
The sparse path clears only vector groups that contain a nonzero bit.
The dense path clears the entire gradient shard. The PTX checker verifies
both scalar parameter declarations, the update expressions, the clear
addresses and the write predicates. The same decoded launch produces the
Adam results and the scratch bytes used by the workspace trace proof.

The local gradient scatter groups four distinct owner rows per CTA.
Each warp handles one row and issues an eight-wide FP16 atomic add per
lane and stripe. The original two-wide kernel uses four warps per row.
The peer launch follows the local launch and retains its original
schedule so repeated destinations accumulate in that order. The direct
arm clears the same FP16 source words. With markers enabled, the first
lane of each warp publishes the original ticket. The PTX checker verifies
all 128 lanes, including the masked tail. The source proof derives the
addresses and unique destination writes from the host owner slice.
The warp checks the tail once before processing its three stripes.
The signed row and column offset is computed once and reused by the
loads, atomics and direct clears. The SM90a build removes all warp
reconvergence instructions in the four local variants. The unmarked
direct path emits 44 static non-NOP SASS instructions, down from 58.
The unmarked staged path emits 36, down from 56. The checker binds
the branch predicate and its forward target before interpreting the
memory operations.
`ValueAdamScale.lean` proves that the maximum position is independent
of the atomic order and that exactly one entry scales each marked row.
The positive positions fit in the signed 32-bit marker. Clearing rows
with nonzero positions restores the positive-zero shard before the next
event. `ValueAdamClaimedClear.lean` proves that traversing the previous
scatter entries clears the same rows and resets every marker for any
entry order. The artifact certificate selects this workspace calculation
in the training driver, preserving the scaled FP16 gradient and optimizer
output.

`ReplicatedAdam.lean` proves that the tile plan gives every active
parameter and moment word a unique store owner with the original scalar
table row. The driver runs the numerical update at the replicated
optimizer call on odd-numbered steps. The artifact check compiles both
BF16 and FP32 versions, then replays every store of aligned and mixed
segment layouts with inactive segments and partial tiles. The matched
expressions share the compiled `sqrt.approx.ftz.f32` and `div.full.f32`
instruction effects.

```sh
python3 models/NanoGPT/Track1/check_replicated_adam.py "$BASELINE" "$CANDIDATE" "$IR_DIR/replicated-adam"
```

`AnvilRails.lean` proves that every rail word has a unique tile owner
and that forwarding the rounded FP32 values produces the same two
live velocity updates and BF16 Gram input as storing and reloading the
temporary. The exact ordered lerp branch and the original blend FMA
are part of the model. The checker follows the real bank dispatch,
replays the two FP32 stores and the BF16 store for the boundary tiles
of every matrix, and checks that the emitted first Gram receives the
freshly prepared output of the rail operator.

```sh
python3 models/NanoGPT/Track1/check_anvil_rails.py "$BASELINE" "$CANDIDATE" "$IR_DIR/anvil-rails"
```

`AnvilGram.lean` proves that the packed coordinate tables enumerate
the same ordered active tiles as the original square grid. On the
default contiguous BF16 shapes, the optimized Hopper kernels compile
the dimensions and strides as constants and omit modulo indexing and
matrix boundary masks. Lean proves that every issued tile lies within
the matrix and every reduction block is full. The original and
specialized paths use the same BF16 operands at each WGMMA step and
issue the same final stores. The two original diagonal writes have
identical address coverage, so the specialized tile issues only the
final transposed store. Every output word is written independently
of prior scratch contents.

`AnvilGramTMA.lean` derives the operand and output addresses from
the flattened tensor descriptors. It proves that the two shared box
indices coincide exactly on the diagonal and that neither index
selects the left box being prefetched. It derives the outstanding
group at each left prefetch from the diagonal path's wait-one and the
off-diagonal path's wait-zero. Each 128-wide box feeds eight
consecutive original WGMMA instructions. The off-diagonal path
retires the current group and synchronizes the CTA before refilling
the fixed right box. The proof separates the direct
and mirrored off-diagonal output addresses. Each output keeps the
original WGMMA accumulator, polynomial FMA, and BF16 rounding.
Both kernels wait for all TMA writes before returning. Lean models a full
wait as completing the pending shared reads and making each destination
word visible. It proves that a separate read wait produces the same final
queue state, and uses the completed words in the kernel's result.
The linked driver runs the first Gram, normalizes it, then runs all six
polynomial rounds with coefficients extracted from the pinned source.
Matrix projections use the shared runtime.
[PTX bulk wait](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)

The offline check compares the TTGIR bodies of the original and compact
masked grids. It replays every active tile of the TMA body through
descriptor addressing, barrier phases, WGMMA issue and retirement,
and the direct and transposed writes. It also checks the explicit
writeback wait in PTX. The trace checker follows the captured Hopper
wrapper through all 36 calls and checks the coefficient values and live
buffer aliases. Each custom operator declares the output as its only
mutation. The checker captures their launch grids and descriptor views
and replays every active CTA of the emitted body, counting the TMA reads
and checking the polynomial's PTX multiply, FMA, and BF16 conversions.

```sh
python3 models/NanoGPT/Track1/check_anvil_gram.py "$BASELINE" "$CANDIDATE" "$IR_DIR/anvil-gram"
```

`AnvilTrace.lean` proves the first Gram's split and fused diagonal
reductions have the same ordered FP32 additions, including the two
zero slots in the 768-wide case. The fused launch applies the pinned
round-to-nearest square root and FMA, then stores the denominator and
its FP32 square. `AnvilNormalization.lean` proves that the compiled
CTA routes read the correct divisor word and that each BF16 address is
in bounds and has exactly one writer. The two outputs match the split
normalizers with the shared Hopper full divide. The verifier replays
the emitted reductions and both normalization stores at every matrix
boundary. It follows all six rounds of the emitted cascade with the
in-place Gram reuse and checks the MLP's rounded projection against
the original pointwise kernel.

```sh
python3 models/NanoGPT/Track1/check_anvil_trace.py "$BASELINE" "$CANDIDATE" "$IR_DIR/anvil-trace"
```

`AnvilEpilogue.lean` models both ordered FMAs of the cautious decay and
the two 16-bit stores. It proves that materializing the normalized BF16
gradient preserves exactly the word used by the fused kernel. The
driver applies this proof at each ANVIL bank after the whitening cascade.
The offline comparison checks the prepared reductions and lane-energy
kernel, the scalar buffer views, and every output expression in boundary
CTAs at both Inductor pointwise tile sizes.

```sh
python3 models/NanoGPT/Track1/check_anvil_epilogue.py "$BASELINE" "$CANDIDATE" "$IR_DIR/anvil-epilogue"
```

The formal execution takes the same file contents, runtime environment,
entropy and observable event order on both sides. Unchanged Python
expressions, PyTorch ops and distributed calls are interpreted by the same
per-operation runtime over the byte-checked common package. The theorem
covers every such runtime, including matching failure, with no premise
equating a pair of training runs or a pair of loss values. Its boundary
is the linked driver's numerical execution model. The Python AST extractor
is checked by regeneration. The artifact check covers the attention scan,
scale and pack stages, the MLP and LM-head refresh, all three sampled gather,
inverse-map and gradient densify shapes, the five long-window RoPE forward shapes,
all four RoPE backward lengths in each attention layout, the compact and
stationary YaRN decoders, the dense FP8 and tied-embedding BF16 transposes,
the tail lerp pair, the grouped norm restoration, the sparse packers, and
the sparse shard gather and cache landing, the scalar and warp n-gram moment replays,
the Adam index packer, both n-gram atomic scatters, the n-gram Adam row
update, the value Adam scatters, scale, and marked clear, the replicated
Adam tile plans, and the ANVIL twin rails, Gram, polynomial, diagonal
reduction, and weight-update kernels.
Full refinement of CPython,
compiled Torch, the GPU instruction models on hardware, and NCCL execution
to this model remains separate. Wall time, source text
in logs and serialization metadata are outside the numerical observation.

## Same-node measurement

On the eight-H100 node, the following commands compare the installed candidate
to the unchanged source on the same data and keep all run outcomes. Use
`DATA_ROOT` as the directory above the upstream `data/fineweb10B` directory.

```sh
python3 models/NanoGPT/Track1/verify_candidate.py "$BASELINE" "$CANDIDATE" --mode cuda
python3 models/NanoGPT/Track1/benchmark_pair.py "$BASELINE" "$CANDIDATE" "$DATA_ROOT" "$RESULTS" --pairs 3
```

The local result is an A/B measurement. The leaderboard result requires an
actual loss and time comparison under its rules.
