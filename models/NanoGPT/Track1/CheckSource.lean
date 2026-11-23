import models.NanoGPT.Track1.ArtifactTraining
import models.NanoGPT.Track1.CombinedTraining
import models.NanoGPT.Track1.MuddFullPackingBinding
import models.NanoGPT.Track1.QKVScaleBinding
import models.NanoGPT.Track1.QKVGluonBinding
import models.NanoGPT.Track1.QKVNativeAmaxBinding
import models.NanoGPT.Track1.QKVGroupedBinding
import models.NanoGPT.Track1.MLPWeightBinding
import models.NanoGPT.Track1.LMHeadBinding
import models.NanoGPT.Track1.SampledGatherBinding
import models.NanoGPT.Track1.SampledDensifyBinding
import models.NanoGPT.Track1.RopeShiftBinding
import models.NanoGPT.Track1.RopeBackwardBinding
import models.NanoGPT.Track1.RopeNarrowBinding
import models.NanoGPT.Track1.YarnBinding
import models.NanoGPT.Track1.YarnStationaryBinding
import models.NanoGPT.Track1.ByteTransposeBinding
import models.NanoGPT.Track1.Bf16TransposeBinding
import models.NanoGPT.Track1.FineWebBatches
import models.NanoGPT.Track1.TailLerpBinding
import models.NanoGPT.Track1.TailNormBinding
import models.NanoGPT.Track1.SparsePacketsBinding
import models.NanoGPT.Track1.SparseLandingBinding
import models.NanoGPT.Track1.MomentReplayBinding
import models.NanoGPT.Track1.NgramAdamBinding
import models.NanoGPT.Track1.NgramAccumulationBinding
import models.NanoGPT.Track1.NgramScratchBinding
import models.NanoGPT.Track1.ValueAdamBinding
import models.NanoGPT.Track1.ValueAdamNativeLoader
import models.NanoGPT.Track1.ValueAdamLocalLoader
import models.NanoGPT.Track1.GraphSlotPTXBinding

/-! Run the proved link check on the checked-out Python files.
The driver and anchors are also regenerated independently by the Python
source check, so this executable need only compare their exact text. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver

def corrupt (image : SourceImage) (path : String) : SourceImage :=
  image.map (fun (name, text) => (name, if name == path then text ++ "\n# Changed\n" else text))

private def checkTrainingCESources (baseline candidate : SourceImage) : IO Unit := do
  for (path, old, changed) in [
      (crossEntropyFile, "SOFTCAP_A, SOFTCAP_B, SOFTCAP_C = 23.0, 5.0, 7.5\n",
        "SOFTCAP_A, SOFTCAP_B, SOFTCAP_C = 23.0, 5.0, 7.0\n"),
      (crossEntropyFile, "LM_HEAD_GRAD_SCALE = (0.75 / 8) / 448",
        "LM_HEAD_GRAD_SCALE = (0.75 / 8) / 449"),
      (crossEntropyFile, "use_fixed_kernel = B == SOFTCAP_B and C == SOFTCAP_C and grad_s == LM_HEAD_GRAD_SCALE",
        "use_fixed_kernel = B == SOFTCAP_B and C == SOFTCAP_C"),
      (sampledCEFile, "and grad_s == LM_HEAD_GRAD_SCALE)", "and grad_s == grad_s)"),
      (crossEntropyFile, "mutates_args={\"logits\", \"losses\"}", "mutates_args={\"losses\"}"),
      (crossEntropyFile, "grad_input = logits.view(torch.float8_e5m2)",
        "grad_input = logits.to(torch.float8_e5m2)"),
      (crossEntropyFile, "and n_rows * width < 2**31", "and n_rows * width < 2**32"),
      (sampledCEFile, "prefix_weight, grad_s, sampled_ce_fwd_bwd,",
        "prefix_weight, grad_s, ce_fwd_bwd,"),
      (crossEntropyFile, "x_f8 = quantize_activation(x, x_s)", "x_f8 = quantize_activation(x, w_s)"),
      (sampledCEFile, "x_f8 = quantize_activation(x, x_s)", "x_f8 = quantize_activation(x, w_s)"),
      (privateActivationKernel, "tl.abs(wide) <= 464.0", "tl.abs(wide) < 464.0"),
      (privateActivationKernel, "if torch.compiler.is_compiling():", "if False:"),
      ("train_gpt.py", "compile_training_model(model, env.device)", "torch.compile(model)"),
      (privateMuddCoefficients, "names[plane] for plane in range(10)", "names[9 - plane] for plane in range(10)"),
      (privateMuddCoefficients, "gl.store(SUM + row, rstd)", "gl.store(SUM + row, total)"),
      (privateMuddCoefficients, "transpose.args[0].id = primary", "transpose.args[0].id = duplicate"),
      (privateMuddCoefficients, "residual_register = tmp129.to(tl.bfloat16)", "residual_register = tmp129"),
      (privateMuddCoefficients, "out_ptr0 = OUT + (FIRST + plane)", "out_ptr0 = OUT + FIRST"),
      (privateMuddCoefficients, "and _independent_column_buffers(body, early + late)", "and True"),
      (privateMuddCoefficients, "in_ptr{9 - column}, {pointer}", "in_ptr{column}, {pointer}"),
      (privateMuddBackward, "inverse = gl.load(INVERSE + row)", "inverse = gl.load(INVERSE)")] do
    let faulty := candidate.map fun (name, text) => (name, if name == path then text.replace old changed else text)
    if faulty == candidate || (link baseline faulty).isSome then
      throw (IO.userError s!"Training CE source link accepted a modification at {old}")
  let changeHeadScale (image : SourceImage) := image.map fun (name, text) =>
    (name, if name == modelFile then text.replace
      "grad_s=(0.75 / 8) / 448" "grad_s=(0.75 / 8) / 449" else text)
  let faultyBaseline := changeHeadScale baseline
  let faultyCandidate := changeHeadScale candidate
  if faultyBaseline == baseline || faultyCandidate == candidate || (link faultyBaseline faultyCandidate).isSome then
    throw (IO.userError "Training CE source link accepted a changed model head gradient scale")

def checkDirectories (baseline candidate : System.FilePath) : IO Unit := do
  let baseline ← readImage baseline .reference
  let actual ← readImage candidate .candidate
  let candidate := (CombinedTraining.packedSource actual).getD actual
  if (CombinedTraining.packedSource (corrupt actual privateNativeAdam)).isSome then
    throw (IO.userError "Source link accepted a modified guarded value Adam kernel")
  let some checked := certify baseline candidate
    | throw (IO.userError "Track 1 source link check failed")
  for (path, old, changed) in [
      (pinnedBatchFile, "        slot.uploaded.record()\n        return out",
        "        return out"),
      (privateBatchPacket, "packet = prefix.to(device, non_blocking=True)",
        "packet = prefix.to(device, non_blocking=False)"),
      (stepGraphsFile, "        cs.forward.replay()",
        "        with torch.cuda.stream(self.stream):\n            cs.forward.replay()"),
      (rowPrefetchFile, "member_index=i - self.live_cycle[0]", "member_index=0"),
      (rowPrefetchFile, "self.ngram.start_cycle(pull.ngram)", "self.ngram.start_cycle(self.ngram.live)"),
      (ngramTableFile, "        assert self.cached is pull and self.grads_in_flight is None\n",
        "        assert self.grads_in_flight is None\n"),
      ("train_gpt.py", "row_prefetch.position[step] <= row_prefetch.update_positions[-1]",
        "row_prefetch.position[step] < row_prefetch.update_positions[-1]"),
      (ngramTableFile, "        scatter_add_rows(sink_grad.contiguous(), slots, per_slot)\n",
        "        scatter_add_rows(sink_grad.contiguous(), slots, per_slot.zero_())\n"),
      (ngramTableFile, "        scatter_add_rows(grad, slots, per_slot)\n        self.ngram_event_grad = None\n",
        "        self.ngram_event_grad = None\n"),
      (ngramTableFile, "pack_ngram_gradients(per_slot, lo, hi, clear_source=True)",
        "pack_ngram_gradients(per_slot, lo, hi, clear_source=False)"),
      (ngramTableFile, "clear_source=not reset_whole", "clear_source=reset_whole"),
      (ngramTableFile, "self.grad_per_slot[:pull.num_rows].zero_()", "self.grad_per_slot[:pull.lo].zero_()"),
      (valuePullFile, "ids = pull.peer_want if self.grads_in_flight[0] is None else pull.want",
        "ids = pull.want"),
      (valuePullFile, "own_grad = self.grad_accum[self.own_rows]", "own_grad = self.grad_accum"),
      (valueEmbedFile, "mutates_args={\"grad_accum\"}", "mutates_args=set()"),
      (valueEmbedFile, "list(plane_grads), grad_accum)", "list(reversed(plane_grads)), grad_accum)"),
      (ngramTableFile, "self.beta2_history[event].fill_(beta2)", "self.beta2_history[event - 1].fill_(beta2)"),
      (ngramTableFile, "                   positive_history=True)\n", "                   positive_history=False)\n"),
      (ngramTableFile, "                                        positive_history=True):\n",
        "                                        positive_history=False):\n"),
      (stepGraphsFile, "        cs.ngram_grad = grads[-1]\n", "        cs.ngram_grad = grads[-2]\n"),
      (ngramTableFile, "        work.wait()\n        lo, hi, n = pull.lo, pull.hi, pull.num_rows\n",
        "        lo, hi, n = pull.lo, pull.hi, pull.num_rows\n"),
      (rowOwnersFile, "received, work = all_to_all_rows(served, pull.recv_counts, pull.send_counts, async_op=True)",
        "received, work = all_to_all_rows(served, pull.send_counts, pull.recv_counts, async_op=True)"),
      (rowPrefetchFile, "self.staging.ngram_ring.upload(ngram_want, slot_directory)",
        "self.staging.ngram_ring.upload(ngram_want)"),
      (ngramTableFile, "members[member_index]() is row_ids", "members[member_index]() is not None"),
      (ngramTableFile, "keep[0] = True", "keep[0] = False"),
      (rowPrefetchFile, "        out.copy_(buf[:count], non_blocking=True)\n        uploaded.record()",
        "        uploaded.record()\n        out.copy_(buf[:count], non_blocking=True)")] do
    let faulty := candidate.map fun (name, text) => (name, if name == path then text.replace old changed else text)
    if faulty == candidate || graphStreamSources faulty then
      throw (IO.userError s!"Graph stream source check accepted a modification at {old}")
  checkTrainingCESources baseline candidate
  for path in ["train_gpt.py", modelFile, ngramFile, amaxFile, crossEntropyFile, sampledFile, sampledCEFile, ropeFile, privateKernel,
                privateWeightKernel, privateHeadKernel, privateGatherKernel, transposeFile,
                privateTransposeKernel, privateBf16TransposeKernel, privateDensifyKernel, dataFile,
                "track_1_short/training.py", anvilFile, tailFile, tailLerpFile, privateTailKernel,
                privateTailNormKernel, valuePullFile, valueEmbedFile, privateValueAccumKernel,
                ngramTableFile, rowOwnersFile, rowPrefetchFile,
                privateSparseRowKernel, privateSparseRowFetchKernel, ngramAdamFile, privateNgramMomentKernel,
                rowScatterFile, privateNgramEntriesKernel, privateValueAdamKernel, privateNativeAdam, replicatedAdamFile,
                replicatedKernelFile, privateReplicatedSpanKernel, privateAnvilEpilogueKernel,
                polarFile, privateAnvilTraceKernel, privateAnvilRailsKernel, stepGraphsFile, privateGraphStageKernel,
                pinnedBatchFile, privateBatchPacket, privateNgramSlots] do
    if (link baseline (corrupt candidate path)).isSome then
      throw (IO.userError s!"Source link accepted a modification outside its rules: {path}")
  let badBank := candidate.map fun (name, text) =>
    (name, if name == modelFile then text.replace "qk_heads[slots], v_heads[slots]"
      "v_heads[slots], qk_heads[slots]" else text)
  if (link baseline badBank).isSome then
    throw (IO.userError "Source link accepted swapped QK and V bank views")
  let badGrouped := candidate.map fun (name, text) =>
    (name, if name == modelFile then text.replace "refresh_grouped_qkv(self.attn_fp8, qk_heads, v_heads)"
      "refresh_grouped_qkv(self.attn_fp8, v_heads, qk_heads)" else text)
  if (link baseline badGrouped).isSome then
    throw (IO.userError "Source link accepted swapped grouped QK and V bank views")
  let badBootstrap := candidate.map fun (name, text) =>
    (name, if name == modelFile then text.replace "delayed=not exact" "delayed=exact" else text)
  if (link baseline badBootstrap).isSome then
    throw (IO.userError "Source link accepted an inverted MLP bootstrap predicate")
  let badTriangle := candidate.map fun (name, text) =>
    (name, if name == polarFile then text.replace "0x5b2c76356346c688"
      "0x5b2c76356346c689" else text)
  if badTriangle == candidate || (link baseline badTriangle).isSome then
    throw (IO.userError "Source link failed to reject a changed ANVIL triangle coordinate")
  for (path, old, changed) in [
      (polarFile, "offdiag = row != col", "offdiag = row == col"),
      (polarFile, "for chunk in range(End // 128):", "for chunk in range(End // 64):"),
      (polarFile, "phase = (chunk // 2) % 2", "phase = chunk % 2"),
      (polarFile, "other = tiles.index(2)", "other = base"),
      (polarFile, "row, col, end, Tall, False)", "row, col, end, Tall, True)"),
      (polarFile, "32768 if Diagonal else 65536", "65536 if Diagonal else 32768"),
      (polarFile, "coords = [batch_row + k, col]", "coords = [batch_row + k, row]"),
      (polarFile, "coords = [batch_row + col, k]", "coords = [batch_row + row, k]"),
      (polarFile, "tma.async_copy_shared_to_global(C, [batch * width + col, row], mirror)",
        "tma.async_copy_shared_to_global(C, [batch * width + row, col], mirror)"),
      (polarFile, "num_warps=8, num_stages=1", "num_warps=4, num_stages=1"),
      (polarFile, "launch_cooperative_grid=True", "launch_cooperative_grid=False"),
      (polarFile, "round_number * batches", "(round_number - 1) * batches"),
      (polarFile, "atom.acq_rel.gpu.global.or.b32", "atom.relaxed.gpu.global.or.b32"),
      (anvilFile, "counter = prepare_rails_and_seed(grad_chunk, velocity, momentum_t, fast_beta_t, fast_weight_t, X)", "counter = None"),
      (anvilFile, "anvil_cooperative_round(X, A, B, counter, k, c, b)",
        "anvil_cooperative_round(X, A, B, counter, k, b, c)"),
      (polarFile, "ordinal == side - 1", "ordinal == side - 2"),
      (polarFile, "if not offdiag:", "if offdiag:"),
      (polarFile, "incident_tiles(row // 128, side) | incident_tiles(col // 128, side)",
        "incident_tiles(row // 128, side)"),
      (polarFile, "publish_gram(Counter + round_offset + batch, bit, needed)",
        "publish_gram(Counter + batch, bit, needed)"),
      (polarFile, "mutates_args={\"out\"}", "mutates_args={\"a\"}"),
      (anvilFile, "full_tiles=compact_trace", "full_tiles=True")] do
    let modified := candidate.map fun (name, text) =>
      (name, if name == path then text.replace old changed else text)
    if modified == candidate || (link baseline modified).isSome then
      throw (IO.userError s!"Source link failed the ANVIL full-tile mutation at {old}")
  let badTrace := candidate.map fun (name, text) =>
    (name, if name == privateAnvilTraceKernel then text.replace "(8, 4, 2, 1, 16)" "(16, 8, 4, 2, 1)" else text)
  if badTrace == candidate || (link baseline badTrace).isSome then
    throw (IO.userError "Source link failed to reject a reassociated ANVIL trace")
  let badDivisors := candidate.map fun (name, text) =>
    (name, if name == privateAnvilTraceKernel then text.replace "(tile >= X_TILES).to(gl.int32)"
      "(tile < X_TILES).to(gl.int32)" else text)
  if badDivisors == candidate || (link baseline badDivisors).isSome then
    throw (IO.userError "Source link failed to reject swapped ANVIL divisors")
  let badRails := candidate.map fun (name, text) =>
    (name, if name == privateAnvilRailsKernel then text.replace "gl.store(Velocity + CELLS + i, slow)"
      "gl.store(Velocity + CELLS + i, fast)" else text)
  if badRails == candidate || (link baseline badRails).isSome then
    throw (IO.userError "Source link failed to reject a swapped ANVIL slow rail")
  let changeHeadRoundTrip (image : SourceImage) := image.map fun (name, text) =>
    (name, if name == crossEntropyFile then text.replace
      "(w.to(tl.float32) * inv).to(tl.bfloat16).to(tl.float32)" "w.to(tl.float32) * inv" else text)
  if (link (changeHeadRoundTrip baseline) (changeHeadRoundTrip candidate)).isSome then
    throw (IO.userError "Source link accepted a different LM-head reference round trip")
  for (path, old, changed) in [
      (modelFile, "logits.masked_fill(dropped, -60.0)", "logits.masked_fill(dropped, -59.0)"),
      (modelLayersFile, "x @ self.weight.type_as(x)", "x @ self.weight.float()"),
      (anvilFile, "lane_energy.clamp_min(1e-10).rsqrt_()", "lane_energy.clamp_min(1e-9).rsqrt_()"),
      ("track_1_short/training.py", "lr=0.008,\n            eps=1e-10", "lr=0.009,\n            eps=1e-10"),
      (anvilFile, "3.923798038567", "3.92"),
      (anvilFile, "FROBENIUS_MARGIN, FROBENIUS_EPS = 1.05, 1e-6",
        "FROBENIUS_MARGIN, FROBENIUS_EPS = 1.05, 1e-4"),
      (anvilFile, "0.85, 0.98, 0.4385, 514", "0.85, 0.97, 0.4385, 514"),
      (replicatedAdamFile, "group.segment[span] = segment_of[p]", "group.segment[span] = 0")] do
    let mutate (image : SourceImage) := image.map fun (name, text) =>
      (name, if name == path then text.replace old changed else text)
    let modified := mutate candidate
    if modified == candidate || (link (mutate baseline) modified).isSome then
      throw (IO.userError s!"Source link failed the numerical contract mutation at {old}")
  for (old, changed) in [("self.staging.wait_uploaded(staged)", "pass"),
      ("candidates = np.flatnonzero(mark[:V])", "candidates = np.flatnonzero(mark)"),
      ("inputs.vocab_pos.fill_(-1)", "inputs.vocab_pos.fill_(0)"),
      ("inputs.vocab_pos.index_copy_(0, cand, self.arange[:P])", "pass"),
      ("self.arange = torch.arange(max_p, dtype=torch.int32", "self.arange = torch.arange(max_p, dtype=torch.int64")] do
    let mutate (image : SourceImage) := image.map fun (name, text) =>
      (name, if name == sampledFile then text.replace old changed else text)
    if (link (mutate baseline) (mutate candidate)).isSome then
      throw (IO.userError s!"Source link accepted a sampled gather contract mutation at {old}")
  let withoutFusedInverse := candidate.map fun (name, text) =>
    (name, if name == sampledFile then text.replace ", inverse=inputs.vocab_pos" "" else text)
  if withoutFusedInverse == candidate || (link baseline withoutFusedInverse).isSome then
    throw (IO.userError "Source link accepted omitting the fused inverse output")
  let skipCandidateWait (image : SourceImage) := image.map fun (name, text) =>
    (name, if name == "track_1_short/perf/sampled_softmax_overlap.py" then
      text.replace "torch.cuda.current_stream().wait_event(staged.slot.uploaded)" "pass" else text)
  if (certify (skipCandidateWait baseline) (skipCandidateWait candidate)).isSome then
    throw (IO.userError "Source link accepted bypassing the candidate upload event")
  let changeReferenceTranspose (image : SourceImage) := image.map fun (name, text) =>
    (name, if name == transposeFile then text.replace "tl.trans(tile)" "tile" else text)
  if (link (changeReferenceTranspose baseline) (changeReferenceTranspose candidate)).isSome then
    throw (IO.userError "Source link accepted a changed reference transpose")
  for (old, changed) in [
      ("lm_head_gather_future.wait()", "pass"),
      ("transpose_add(embed_param.grad, param.grad)", "transpose_add(param.grad, embed_param.grad)")] do
    let mutate (image : SourceImage) := image.map fun (name, text) =>
      (name, if name == anvilFile then text.replace old changed else text)
    if (link (mutate baseline) (mutate candidate)).isSome then
      throw (IO.userError s!"Source link accepted a tied embedding mutation at {old}")
  let changeBackward (image : SourceImage) := image.map fun (name, text) =>
    (name, if name == sampledCEFile then text.replace
      "sampled_densify(grad_w_c.contiguous(), vocab_pos, grad_w)" "pass" else text)
  if (link (changeBackward baseline) (changeBackward candidate)).isSome then
    throw (IO.userError "Source link accepted removing the sampled LM-head gradient write")
  let changeCELayout (image : SourceImage) := image.map fun (name, text) =>
    (name, if name == crossEntropyFile then text.replace "x_f8_T, grad_input_T.T," "x_f8_T, grad_input_T," else text)
  if changeCELayout candidate == candidate || (link (changeCELayout baseline) (changeCELayout candidate)).isSome then
    throw (IO.userError "Source link accepted changing the weight-gradient GEMM layout")
  for (old, changed) in [("float lse = A + __logf(block_sum);", "float lse = A - __logf(block_sum);"),
      ("CE_KERNEL_BLOCK_SIZE = 256", "CE_KERNEL_BLOCK_SIZE = 128")] do
    let mutate (image : SourceImage) := image.map fun (name, text) =>
      (name, if name == crossEntropyFile then text.replace old changed else text)
    if mutate candidate == candidate || (link (mutate baseline) (mutate candidate)).isSome then
      throw (IO.userError s!"Source link accepted a training CE change at {old}")
  let wrongCount := candidate.map fun (name, text) =>
    (name, if name == sampledCEFile then text.replace "SAMPLED_CE_FIXED_KERNELS.get((P, n_predict))"
      "SAMPLED_CE_FIXED_KERNELS.get((P, n_predict + 1))" else text)
  if wrongCount == candidate || (link baseline wrongCount).isSome then
    throw (IO.userError "Source link accepted selecting the wrong sampled CE prediction count")
  let wrongScratch := candidate.map fun (name, text) =>
    (name, if name == sampledCEFile then text.replace "shared_mem=P * (2 if fixed_kernel is None else 1)"
      "shared_mem=P * (1 if fixed_kernel is None else 2)" else text)
  if wrongScratch == candidate || (link baseline wrongScratch).isSome then
    throw (IO.userError "Source link accepted reversing the sampled CE scratch reservations")
  let wrongOwner := candidate.map fun (name, text) =>
    (name, if name == crossEntropyFile then text.replace
      "% BLOCK_SIZE != threadIdx.x" "% BLOCK_SIZE == threadIdx.x" else text)
  if wrongOwner == candidate || (link baseline wrongOwner).isSome then
    throw (IO.userError "Source link accepted changing the CE correction writer")
  let changeRopeFactors (image : SourceImage) := image.map fun (name, text) =>
    (name, if name == ropeFile then text.replace
      "f1 * (x * rstd[:, None]) + f2 * (x_flip * rstd[:, None])"
      "f2 * (x * rstd[:, None]) + f1 * (x_flip * rstd[:, None])" else text)
  if (link (changeRopeFactors baseline) (changeRopeFactors candidate)).isSome then
    throw (IO.userError "Source link accepted swapped RoPE factors")
  for (path, original, changed) in [
      (modelFile, "PAIRED_HEAD_LAYERS = (0, 2, 5)", "PAIRED_HEAD_LAYERS = (0, 1, 5)"),
      (modelFile, "NARROW_HEAD_DIM = 64", "NARROW_HEAD_DIM = 96"),
      (attentionFile, "yarn.factor1[:T], yarn.factor2[:T], H", "yarn.factor1[:T], yarn.factor2[1:T+1], H"),
      ("track_1_short/config.py", "WS_POST_YARN_EXT = 20", "WS_POST_YARN_EXT = 22"),
      ("track_1_short/training.py", "yarn_wide.apply(self.ws_long * BLOCK_SIZE,",
        "yarn_wide.apply(self.ws_long,")] do
    let mutate (image : SourceImage) := image.map fun (name, text) =>
      (name, if name == path then text.replace original changed else text)
    let modified := mutate candidate
    if modified == candidate || (link (mutate baseline) modified).isSome then
      throw (IO.userError s!"Source link failed the attention geometry mutation at {original}")
  let dropFirstKeyGrad (image : SourceImage) := image.map fun (name, text) =>
    (name, if name == ropeFile then text.replace
      "stationary_grad = tl.where(token[:, None] == 0, grad + grad_next, grad_next)"
      "stationary_grad = grad_next" else text)
  if (link (dropFirstKeyGrad baseline) (dropFirstKeyGrad candidate)).isSome then
    throw (IO.userError "Source link accepted dropping the first token's stationary key gradient")
  for (path, original, changed) in [
      (ropeFile, "mutates_args={\"out_q\", \"out_k\"}", "mutates_args={\"out_q\"}"),
      (ropeFile, "@_launch_qkv_rope_backward.register_fake", "# disabled fake registration"),
      (ropeFile, "xor.b16 $0, $1, 0x8000;", "xor.b16 $0, $1, 0x4000;"),
      (ropeFile, "tl.fma(rstd[:, None] * x, f1, (rstd[:, None] * x_flip) * f2)",
        "tl.fma(rstd[:, None] * x_flip, f2, (rstd[:, None] * x) * f1)"),
      (transposeFile, "@transpose_copy.register_fake", "# disabled fake registration")] do
    let modified := candidate.map fun (name, text) =>
      (name, if name == path then text.replace original changed else text)
    if modified == candidate || (link baseline modified).isSome then
      throw (IO.userError s!"Source link failed the compiled launcher mutation at {original}")
  for (old, changed) in [("self._ready.is_set()", "True"), ("tokens[:6_000_000]", "tokens[:5_000_000]"),
      ("HEADER_BYTES = 256 * 4", "HEADER_BYTES = 255 * 4"), ("SHARD_READ_THREADS = 4", "SHARD_READ_THREADS = 3"),
      ("assert next(iter(self.ahead)) == step", "assert step in self.ahead"),
      ("_targets = buf[1:]", "_targets = buf[:-1]")] do
    let mutate (image : SourceImage) := image.map fun (name, text) =>
      (name, if name == dataFile then text.replace old changed else text)
    if (link (mutate baseline) (mutate candidate)).isSome then
      throw (IO.userError s!"Source link accepted a FineWeb batch contract mutation at {old}")
  for (old, changed) in [("offsets[-1] -= 1", "offsets[0] -= 1"),
      ("buf[1:].astype(np.int64)", "buf[:-1].astype(np.int64)")] do
    let modified := candidate.map fun (name, text) =>
      (name, if name == dataFile then text.replace old changed else text)
    if modified == candidate || (link baseline modified).isSome then
      throw (IO.userError s!"Source link failed the FineWeb pack mutation at {old}")
  for (path, old, changed) in [
      (tailFile, "BANK_BLEND_PERIOD = 4", "BANK_BLEND_PERIOD = 2"),
      (tailFile, "rate, bank_rate)", "bank_rate, rate)"),
      (tailFile, "shard.copy_(avg.to(shard.dtype)", "shard.copy_(avg.round().to(shard.dtype)"),
      (tailFile, "by_tag[\"tail-avg\"], by_tag[\"value-embed-avg\"], by_tag[\"bank-blend\"]",
        "by_tag[\"bank-blend\"], by_tag[\"value-embed-avg\"], by_tag[\"tail-avg\"]"),
      (tailFile, "norms_before / norms_after", "norms_after / norms_before"),
      (privateTailNormKernel, "dtype=torch.float32", "dtype=torch.bfloat16"),
      (tailLerpFile, "rate, e - s, s", "rate, s - e, s")] do
    let mutate (image : SourceImage) := image.map fun (name, text) =>
      (name, if name == path then text.replace old changed else text)
    let modified := mutate candidate
    if modified == candidate || (link (mutate baseline) modified).isSome then
      throw (IO.userError s!"Source link failed a tail averaging mutation at {old}")
  for (path, original, replacement) in [
      (valuePullFile, "self.grad_accum.index_fill_(0, ids.long(), 0)",
        "self.grad_accum.index_fill_(0, ids.long(), 1)"),
      (valuePullFile, "clear_value_want_(self.grad_accum, ids)", "clear_value_want_(self.grad_accum, ids[:1])"),
      (valuePullFile, "pack_own=pull.peer_want is None", "pack_own=pull.peer_want is not None"),
      (valuePullFile, "full.index_fill_(0, own_rows.long(), 0)", "full.index_fill_(0, own_rows.long(), 1)"),
      (valuePullFile, "distinct + np.int32(p * vocab_size)", "distinct + np.int32(0)"),
      (privateSparseRowKernel, "tl.store(source + offsets, 0.0", "tl.store(source + offsets, -0.0"),
      (ngramTableFile, "scratch += 1 << 30", "scratch += 1 << 29"),
      ("track_1_short/schedule.py", "table[first:stop, :width].unbind(0)", "table[first:stop, :width - 1].unbind(0)"),
      (ngramTableFile, "np.multiply(tokens[1:-1], TRIGRAM_ROW_MULS[1]", "np.multiply(tokens[2:], TRIGRAM_ROW_MULS[1]"),
      (ngramTableFile, "per_slot[lo:hi], received", "per_slot[hi:], received"),
      (rowOwnersFile, "output_split_sizes=recv_counts", "output_split_sizes=send_counts"),
      (rowOwnersFile, "pull.ids_exchange.wait()", "pass"),
      (rowOwnersFile, "self.bounds.astype(np.int32)", "self.bounds.astype(np.int16)"),
      (rowOwnersFile, "np.searchsorted(want_host, bounds)", "np.searchsorted(want_host, bounds, side=\"right\")"),
      (rowOwnersFile, "return all_to_all_rows(wire, pull.send_counts, pull.recv_counts, async_op=True)",
        "return all_to_all_rows(wire, pull.recv_counts, pull.send_counts, async_op=True)"),
      (ngramTableFile, "work.wait()\n        lo, hi, n", "pass\n        lo, hi, n"),
      (privateSparseRowFetchKernel, "row >= hi, hi - lo", "row > hi, hi - lo"),
      (privateSparseRowFetchKernel, "gl.where(owned, shard, received)", "gl.where(owned, received, shard)"),
      (ngramAdamFile, "v = v * tl.load", "v = v + tl.load"),
      (privateNgramMomentKernel, "replay_from < e, value * beta", "replay_from <= e, value * beta"),
      (privateNgramEntriesKernel, "tl.where(local, want, asked)", "tl.where(local, asked, want)"),
      (rowScatterFile, "tl.atomic_add(DST", "tl.store(DST"),
      (ngramTableFile, "adam_beta2_and_wd_mul(step)", "adam_beta2_and_wd_mul(step + 1)"),
      (valuePullFile, "grad.mul_(1.0 / self.world_size)", "grad.mul_(self.world_size)"),
      (valuePullFile, "scale_read=profile is not None", "scale_read=True"),
      (privateValueAdamKernel, "markers = None if scale_read else touched", "markers = touched"),
      (privateValueAdamKernel, "LOCAL_ROWS = 4", "LOCAL_ROWS = 8"),
      (privateValueAdamKernel, "bra.uni L__local_rows_end", "bra L__local_rows_end"),
      (valuePullFile, "scale_read=(*profile, not sparse)", "scale_read=(*profile, False)"),
      (privateNativeAdam, "setp.ne.b32 %write, $3, 0", "setp.ne.b32 %write, $3, 1"),
      (privateNativeAdam, "step_size.item(), weight, decay.item()",
        "decay.item(), weight, step_size.item()"),
      (valuePullFile, "self._current_scale_read(grad)", "self._current_scale_read(self.grad_accum)"),
      (anvilFile, "self._eff_wd_t, *scale_read)", "self._eff_wd_t, 2, 512)"),
      (privateNativeAdam, "cvt.rn.f16.f32", "cvt.rz.f16.f32"),
      (privateNativeDispatch, "launcher.shared", "0"),
      (privateNativeDispatch, "Path(module.__file__).read_text()", "module.__file__"),
      (privateValueAdamKernel, "ids + offset + slot", "ids + slot"),
      (privateValueAdamKernel, "tl.atomic_max(touched + row, claim_start + slot + 1", "tl.atomic_max(touched + row, 0"),
      (privateValueAdamKernel, "claim == slot + 1", "claim != slot + 1"),
      (privateValueAdamKernel, "triton.cdiv(entries, ROWS)", "triton.cdiv(rows, ROWS)"),
      (privateValueAdamKernel, "storage.zero_()", "grad.zero_()"),
      (privateValueAdamKernel, "tl.debug_barrier()", "pass"),
      (valuePullFile, "self.grad_previous = None", "self.grad_previous = ()"),
      (valuePullFile, "(pull.want, pull.asked, pull.lo, pull.hi)", "(pull.asked, pull.want, pull.lo, pull.hi)"),
      (privateValueAdamKernel, "prior_hi - prior_lo", "prior_hi - prior_lo + 1"),
      (deferredGathersFile, "self._wait(ATTN_BANK_LABELS)", "self._wait(BANK_LABELS)"),
      (fp8RefreshGraphsFile, "(\"attn\", m.quantize_attn_fp8,", "(\"attn\", m.quantize_mlp_fp8,"),
      (anvilFile, "exp_avg_sq.mul_(beta2).addcmul_", "exp_avg_sq.mul_(beta1).addcmul_"),
      ("track_1_short/training.py", "VALUE_EMBED_BETAS[0] ** 2", "VALUE_EMBED_BETAS[0] ** 3"),
      ("track_1_short/training.py", "return step % 2 == 1", "return step % 2 == 0"),
      (rowPrefetchFile, "self.prefetch.ngram.send_grads()\n        self.prefetch.value_embeds.zero_sent_grads()",
        "self.prefetch.value_embeds.zero_sent_grads()\n        self.prefetch.ngram.send_grads()")] do
    let mutate (image : SourceImage) := image.map fun (name, text) =>
      (name, if name == path then text.replace original replacement else text)
    let modified := mutate candidate
    if modified == candidate || (link (mutate baseline) modified).isSome then
      throw (IO.userError s!"Source link failed a sparse exchange mutation at {path}")
  IO.println s!"Linked {checked.sources.common.length} common runtime files and the candidate's private kernels."
  IO.println "Source changes outside the checked substitutions were rejected."

def checkTtirDirectory (ir : System.FilePath) : IO Unit := do
  for (name, width) in [("narrow", QKVLaunch.Width.narrow), ("halfValue", .halfValue), ("wide", .wide)] do
    let oldIR ← IO.FS.readFile (ir / (name ++ "-reference.ttir"))
    let newIR ← IO.FS.readFile (ir / (name ++ "-aligned.ttir"))
    if (QKVTTIR.certifyPair width oldIR newIR).isNone then
      throw (IO.userError s!"Unrecognized QKV TTIR pair for {name}")
    for (original, mutation) in [("arith.addf", "arith.subf"), ("axis = 0", "axis = 1"),
                                  ("tt.addptr %1, %part", "tt.addptr %1, %batch")] do
      if (QKVTTIR.certifyPair width oldIR (newIR.replace original mutation)).isSome then
        throw (IO.userError s!"TTIR link accepted a change to {original} for {name}")
    if (QKVTTIR.certifyPair width (oldIR.replace "arith.maxsi" "arith.minsi") newIR).isSome then
      throw (IO.userError s!"TTIR link accepted a change to the baseline's masked offset for {name}")
    let nativeIR ← IO.FS.readFile (ir / (name ++ "-native.ttir"))
    if (QKVNative.certifyScan width oldIR nativeIR).isNone then
      throw (IO.userError s!"Unrecognized native-bank QKV TTIR for {name}")
    let mutated := nativeIR.replace "1179648 : i32" "1178880 : i32"
    if mutated == nativeIR || (QKVNative.certifyScan width oldIR mutated).isSome then
      throw (IO.userError s!"Native-bank scan pitch mutation failed for {name}")
    if width != .wide then
      let mutated := nativeIR.replace "24 : i32" "25 : i32"
      if mutated == nativeIR || (QKVNative.certifyScan width oldIR mutated).isSome then
        throw (IO.userError s!"Native-bank head crop mutation failed for {name}")
    IO.println s!"Certified {name} QKV TTIR and rejected arithmetic, reduction, and address mutations."

def checkScaleArtifacts (ir : System.FilePath) : IO QKVRefresh.ScaleProgram := do
  let ttir ← IO.FS.readFile (ir / "qkv-scale.ttir")
  let ptx ← IO.FS.readFile (ir / "qkv-scale.ptx")
  let some program := QKVRefresh.ScaleProgram.check ttir ptx
    | throw (IO.userError "Unrecognized QKV scale TTIR or PTX")
  if (QKVScale.certify (ttir.replace "arith.maximumf" "arith.maxnumf") ptx).isSome then
    throw (IO.userError "QKV scale accepted a different NaN policy")
  for (original, mutation) in [("16, 31, -1", "8, 31, -1"), ("0f3B124925", "0f3B124924"),
      ("%p1, %r5, 0;", "%p1, %r5, 1;")] do
    if (QKVScale.certify ttir (ptx.replace original mutation)).isSome then
      throw (IO.userError s!"QKV scale accepted a change to {original}")
  IO.println "Certified the QKV scale TTIR and PTX and rejected reduction, constant, and store mutations."
  pure program

def checkPackArtifacts (ir : System.FilePath) : IO Unit := do
  for (name, width) in [("narrow", QKVLaunch.Width.narrow), ("halfValue", .halfValue), ("wide", .wide)] do
    let oldIR ← IO.FS.readFile (ir / ("pack-" ++ name ++ "-reference.ttir"))
    let newIR ← IO.FS.readFile (ir / ("pack-" ++ name ++ "-aligned.ttir"))
    if (QKVPack.certify width oldIR newIR).isNone then
      throw (IO.userError s!"Unrecognized QKV pack TTIR pair for {name}")
    for (old, changed) in [("rounding = rtne", "rounding = rtz"),
          ("order = array<i32: 1, 0>", "order = array<i32: 0, 1>"), ("arith.divf", "arith.mulf")] do
      if (QKVPack.certify width oldIR (newIR.replace old changed)).isSome then
        throw (IO.userError s!"QKV pack TTIR accepted a change to {old} for {name}")
    let ttgir ← IO.FS.readFile (ir / ("pack-" ++ name ++ "-gluon.ttgir"))
    let ptx ← IO.FS.readFile (ir / ("pack-" ++ name ++ "-gluon.ptx"))
    if (QKVPack.certifyGluon width oldIR ttgir ptx).isNone then
      throw (IO.userError s!"Unrecognized {name} Gluon TTGIR or PTX byte exchange")
    if (QKVPack.certifyGluon width oldIR (ttgir.replace "ttg.convert_layout" "ttg.fake_convert") ptx).isSome then
      throw (IO.userError s!"{name} Gluon accepted a different layout operation")
    for (old, changed) in [("bar.warp.sync", "bar.sync"), ("m8n8.x4", "m8n8.x2"),
        ("4080;", "4064;"), ("0x6420U", "0x6402U"), ("div.full.f32", "mul.f32"),
        ("1179648;", "1178880;")] do
      let mutated := ptx.replace old changed
      if mutated == ptx then
        throw (IO.userError s!"Missing {name} PTX mutation site {old}")
      if (QKVPack.certifyGluon width oldIR ttgir mutated).isSome then
        throw (IO.userError s!"{name} Gluon accepted a change to {old}")
    IO.println s!"Certified {name} QKV packing and the Gluon TTGIR/PTX byte exchange."

def checkGroupedArtifacts (ir : System.FilePath) : IO QKVRefresh.GroupedProgram := do
  let a : Grouped.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / "qkv-grouped-scan.ttir"),
    ← IO.FS.readFile (ir / "qkv-grouped-scan.ptx"),
    ← IO.FS.readFile (ir / "qkv-grouped-scale.ttir"),
    ← IO.FS.readFile (ir / "qkv-grouped-scale.ptx"),
    ← IO.FS.readFile (ir / "qkv-grouped-pack.ttgir"),
    ← IO.FS.readFile (ir / "qkv-grouped-pack.ptx")⟩
  let some program := QKVRefresh.GroupedProgram.check a
    | throw (IO.userError "Unrecognized grouped QKV artifacts")
  let mutations := [
    ("scan pitch", { a with scanIR := a.scanIR.replace "1179648 : i32" "1178880 : i32" }),
    ("scan crop stride", { a with scanPTX := a.scanPTX.replace "786432" "786416" }),
    ("scan tile period", { a with scanIR := a.scanIR.replace "192 : i32" "191 : i32" }),
    ("scan V tail", { a with
        scanIR := a.scanIR.replace "arith.remsi %second_count_7, %c3_i32"
          "arith.remsi %second_count_7, %c2_i32" }),
    ("scale pointer", { a with
        scaleIR := a.scaleIR.replace "arith.select %wide, %scale2, %scale1"
          "arith.select %wide, %scale1, %scale2" }),
    ("scale gain", { a with scalePTX := a.scalePTX.replace "0f3B124925" "0f3B124924" }),
    ("pack bitmask", { a with packIR := a.packIR.replace "17076497 : i32" "17076496 : i32" }),
    ("pack bit scan", { a with packPTX := a.packPTX.replace "bfind.u32" "brev.b32" }),
    ("pack exchange", { a with packPTX := a.packPTX.replace "m8n8.x4" "m8n8.x2" }),
    ("vector store", { a with packPTX := a.packPTX.replace "st.global.v4.b32" "st.global.v2.b32" }),
    ("byte permutation", { a with packPTX := a.packPTX.replace "0x4210U" "0x4201U" }),
    ("tile column pitch", { a with
        packIR := a.packIR.replace "%7 = arith.muli %start_n_2, %M_11"
          "%7 = arith.muli %start_n_2, %c768_i32" })]
  for (label, changed) in mutations do
    if (Grouped.certify changed).isSome then
      throw (IO.userError s!"Grouped QKV failed the mutation check for {label}")
  IO.println "Certified the three grouped QKV GPU stages and rejected width-routing, address and arithmetic mutations."
  pure program

private def checkNativeQKV (ir : System.FilePath) (name : String) (width : QKVLaunch.Width) :
    IO (QKVRefresh.NativeProgram width) := do
  let scans : QKVNative.ScanArtifacts ← pure ⟨
    ← IO.FS.readFile (ir / (name ++ "-reference.ttir")),
    ← IO.FS.readFile (ir / (name ++ "-reference.ptx")),
    ← IO.FS.readFile (ir / (name ++ "-native.ttir")),
    ← IO.FS.readFile (ir / (name ++ "-native.ptx"))⟩
  let packReference ← IO.FS.readFile (ir / ("pack-" ++ name ++ "-reference.ttir"))
  let packTTGIR ← IO.FS.readFile (ir / ("pack-" ++ name ++ "-gluon.ttgir"))
  let packPTX ← IO.FS.readFile (ir / ("pack-" ++ name ++ "-gluon.ptx"))
  let some program := QKVRefresh.NativeProgram.check width scans packReference packTTGIR packPTX
    | throw (IO.userError s!"Unrecognized {name} native QKV artifacts")
  let badRef := scans.referencePTX.replace "shfl.sync.bfly.b32" "shfl.sync.idx.b32"
  let badNative := scans.nativePTX.replace "1179648" "1178880"
  if badRef == scans.referencePTX || (QKVNative.ScanProgram.check width { scans with referencePTX := badRef }).isSome then
    throw (IO.userError s!"Native QKV accepted a reference lane mutation for {name}")
  if badNative == scans.nativePTX || (QKVNative.ScanProgram.check width { scans with nativePTX := badNative }).isSome then
    throw (IO.userError s!"Native QKV accepted a bank pitch mutation for {name}")
  pure program

private def checkQKVRefreshArtifacts (ir : System.FilePath) : IO QKVRefresh.Programs := do
  checkPackArtifacts ir
  let grouped ← checkGroupedArtifacts ir
  let perWidthScale ← checkScaleArtifacts ir
  let narrow ← checkNativeQKV ir "narrow" .narrow
  let halfValue ← checkNativeQKV ir "halfValue" .halfValue
  let wide ← checkNativeQKV ir "wide" .wide
  IO.println "Certified the compiled per-width QKV fallback for all three widths."
  pure { grouped, perWidthScale, perWidth := fun
    | .narrow => narrow
    | .halfValue => halfValue
    | .wide => wide }

def checkMlpWeightArtifacts (ir : System.FilePath) : IO MLPWeights.Program := do
  let a : MLPWeights.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / "mlp-weight-reference.ttir"),
    ← IO.FS.readFile (ir / "mlp-weight-reference.ptx"),
    ← IO.FS.readFile (ir / "mlp-weight-scale.ttir"),
    ← IO.FS.readFile (ir / "mlp-weight-scale.ptx"),
    ← IO.FS.readFile (ir / "mlp-weight-pack.ttgir"),
    ← IO.FS.readFile (ir / "mlp-weight-pack.ptx")⟩
  let some program := MLPWeights.Program.check a
    | throw (IO.userError "Unrecognized MLP weight refresh artifacts")
  let mutations := [
    ("reference reduction", { a with
        referenceIR := a.referenceIR.replace "arith.maxnumf" "arith.maximumf" }),
    ("delayed NaN policy", { a with scaleIR := a.scaleIR.replace "arith.maximumf" "arith.maxnumf" }),
    ("delayed gain", { a with scalePTX := a.scalePTX.replace "0f3B23D70A" "0f3B23D70B" }),
    ("paired reduction axis", { a with packIR := a.packIR.replace "axis = 2" "axis = 1" }),
    ("paired conversion", { a with
        packPTX := a.packPTX.replace "cvt.rn.satfinite.e4m3x2.f32" "cvt.rz.satfinite.e4m3x2.f32" }),
    ("paired reduction barrier", { a with packPTX := a.packPTX.replace "bar.sync" "bar.warp.sync" }),
    ("shared reduction base", { a with packPTX := a.packPTX.replace "+4096]" "+4080]" }),
    ("shared reduction read pitch", { a with packPTX := a.packPTX.replace "%r295+4096]" "%r295+4100]" }),
    ("cross-half shuffle", { a with packPTX := a.packPTX.replace ", 16, 31, -1;" ", 4, 31, -1;" })]
  for (label, changed) in mutations do
    if changed == a || (MLPWeights.certify changed).isSome then
      throw (IO.userError s!"MLP weight checker accepted a change to {label}")
  IO.println "Certified the MLP scale, both FP8 layouts and the two maxima through the separate 64-byte shared buffer."
  return program

def checkLmHeadArtifacts (ir : System.FilePath) : IO LMHead.Program := do
  let a : LMHead.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / "lm-head-reference.ttir"),
    ← IO.FS.readFile (ir / "lm-head-reference.ptx"),
    ← IO.FS.readFile (ir / "lm-head-hopper.ttgir"),
    ← IO.FS.readFile (ir / "lm-head-hopper.ptx")⟩
  let some program := LMHead.Program.check a
    | throw (IO.userError "Unrecognized LM-head pack artifacts")
  let mutations := [
    ("reference bounds", { a with referenceIR := a.referenceIR.replace "arith.cmpi slt" "arith.cmpi sgt" }),
    ("Hopper gain", { a with hopperIR := a.hopperIR.replace "arith.mulf" "arith.divf" }),
    ("BF16 rounding", { a with hopperPTX := a.hopperPTX.replace "cvt.rn.bf16x2.f32" "cvt.rz.bf16x2.f32" }),
    ("fixed gain branch", { a with hopperPTX := a.hopperPTX.replace "1130364927" "1130364928" }),
    ("packed multiplication", { a with hopperPTX := a.hopperPTX.replace "0x43604360" "0x43614360" }),
    ("upper bound", { a with hopperPTX := a.hopperPTX.replace "0x43e043e0" "0x43e043df" }),
    ("FP16 rounding", { a with hopperPTX := a.hopperPTX.replace "cvt.rn.f16.bf16" "cvt.rz.f16.bf16" }),
    ("saturating conversion", { a with hopperPTX := a.hopperPTX.replace ".satfinite.e4m3x2" ".e4m3x2" }),
    ("row store width", { a with hopperPTX := a.hopperPTX.replace "st.global.v4.b32" "st.global.v2.b32" }),
    ("transpose exchange", { a with hopperPTX := a.hopperPTX.replace "m8n8.x4" "m8n8.x2" })]
  for (label, changed) in mutations do
    if changed == a || (LMHead.certify changed).isSome then
      throw (IO.userError s!"LM-head pack checker accepted a change to {label}")
  IO.println "Certified both LM-head packers, including the exact gain guard, packed BF16 multiplication, FP16 rounding and both vectorized FP8 cache layouts."
  return program

def checkSampledGatherArtifacts (ir : System.FilePath) : IO Unit := do
  for c in [SampledRows.Count.first, .second, .third] do
    let stem := s!"sampled-gather-{c.rows}"
    let ttgir ← IO.FS.readFile (ir / (stem ++ ".ttgir"))
    let ptx ← IO.FS.readFile (ir / (stem ++ ".ptx"))
    if (SampledRows.certify c ttgir ptx).isNone then
      throw (IO.userError s!"Unrecognized sampled gather for {c.rows} candidate rows")
    let differentShape := if c == .first then SampledRows.Count.second else .first
    if (SampledRows.certify differentShape ttgir ptx).isSome then
      throw (IO.userError s!"Sampled gather accepted a different transpose stride for {c.rows} rows")
    for (old, changed) in [(", 768,", ", 767,"), ("bar.warp.sync", "bar.sync"),
        ("0x4210U", "0x4211U"), ("st.global.v4.b32", "st.global.v2.b32")] do
      let mutated := ptx.replace old changed
      if mutated == ptx || (SampledRows.certify c ttgir mutated).isSome then
        throw (IO.userError s!"Sampled gather artifact mutation check failed at {old}")
    IO.println s!"Certified sampled gather at {c.rows} rows and rejected class, phase and byte-store mutations."

private def checkSampledInverseArtifact (ir : System.FilePath) (c : SampledRows.Count) :
    IO (SampledInverse.Program c) := do
  let stem := s!"sampled-inverse-{c.rows}"
  let gir ← IO.FS.readFile (ir / (stem ++ ".ttgir"))
  let ptx ← IO.FS.readFile (ir / (stem ++ ".ptx"))
  let some program := SampledInverse.Program.check c gir ptx
    | throw (IO.userError s!"Unrecognized fused sampled gather at {c.rows} rows")
  let different := if c == .first then SampledRows.Count.second else .first
  if (SampledInverse.Program.check different gir ptx).isSome then
    throw (IO.userError "Fused sampled gather accepted compiler artifacts for another count")
  let mutated := gir.replace "gpu.barrier" ""
  if mutated == gir || (SampledInverse.Program.check c mutated ptx).isSome then
    throw (IO.userError "Fused sampled gather accepted TTGIR with its barrier removed")
  let blocks := c.rows / 16
  let limit := s!"%c{blocks}_i32 = arith.constant "
  let mutated := gir.replace (limit ++ toString blocks) (limit ++ toString (blocks - 1))
  if mutated == gir || (SampledInverse.Program.check c mutated ptx).isSome then
    throw (IO.userError "Fused sampled gather accepted a changed TTGIR coarse-search bound")
  let assembly := String.intercalate "\n" (QKVTTIR.normalizePTX ptx)
  for (old, changed) in [("setp.gt.u32 %p1, %r1, 196;", "setp.gt.u32 %p1, %r1, 195;"),
      ("bar.sync 0;", "bar.warp.sync -1;"), ("st.global.b32", "st.global.b16"),
      (", 50048;", ", 50176;"), ("redux.sync.add.s32", "redux.sync.max.s32"),
      ("st.global.v4.b32", "st.global.v2.b32")] do
    let mutated := assembly.replace old changed
    if mutated == assembly || (SampledInverse.Program.check c gir mutated).isSome then
      throw (IO.userError s!"Fused sampled gather artifact mutation check failed at {old}")
  IO.println s!"Certified fused sampled gather at {c.rows} rows and rejected grid, search, warp sum, tail, barrier and vector-store mutations."
  return program

def checkSampledInverseArtifacts (ir : System.FilePath) : IO SampledInverse.Programs := do
  let small ← checkSampledInverseArtifact ir .first
  let medium ← checkSampledInverseArtifact ir .second
  let large ← checkSampledInverseArtifact ir .third
  return fun
    | .first => small
    | .second => medium
    | .third => large

private def checkDensifyArtifact (ir : System.FilePath) (c : SampledRows.Count) (referenceIR referencePTX : String) :
    IO (Densify.Program c) := do
  let stem := s!"densify-hopper-{c.rows}"
  let a : Densify.Artifacts ← pure ⟨referenceIR, referencePTX,
    ← IO.FS.readFile (ir / (stem ++ ".ttgir")),
    ← IO.FS.readFile (ir / (stem ++ ".ptx"))⟩
  let some program := Densify.Program.check c a
    | throw (IO.userError s!"Unrecognized sampled gradient densify for {c.rows} classes")
  let other := if c == .first then SampledRows.Count.second else .first
  if (Densify.certify other a).isSome then
    throw (IO.userError s!"The densify artifact accepted a different compact row stride for {c.rows} classes")
  let compactPitchMutant := a.hopperPTX.replace s!", {32 * c.rows};" s!", {32 * c.rows + 2};"
  let mutations := [
    ("reference mask", { a with referenceIR := a.referenceIR.replace "arith.cmpi sge" "arith.cmpi sgt" }),
    ("missing class predicate", { a with hopperPTX := a.hopperPTX.replace "setp.gt.s32" "setp.ge.s32" }),
    ("positive zero", { a with hopperPTX := a.hopperPTX.replace "%rs2, 0;" "%rs2, 1;" }),
    ("compact phase pitch", { a with hopperPTX := compactPitchMutant }),
    ("dense phase pitch", { a with hopperPTX := a.hopperPTX.replace ", 1609728;" ", 1609730;" }),
    ("store width", { a with hopperPTX := a.hopperPTX.replace "st.global.v4.b32" "st.global.v2.b32" })]
  for (label, changed) in mutations do
    if changed == a || (Densify.certify c changed).isSome then
      throw (IO.userError s!"Sampled gradient densify failed its {label} mutation check")
  IO.println s!"Certified sampled gradient densify at {c.rows} classes, including inverse lookup, four row phases and zero stores."
  return program

def checkDensifyArtifacts (ir : System.FilePath) : IO Densify.Programs := do
  let referenceIR ← IO.FS.readFile (ir / "densify-reference.ttir")
  let referencePTX ← IO.FS.readFile (ir / "densify-reference.ptx")
  let first ← checkDensifyArtifact ir .first referenceIR referencePTX
  let second ← checkDensifyArtifact ir .second referenceIR referencePTX
  let third ← checkDensifyArtifact ir .third referenceIR referencePTX
  return fun
    | .first => first
    | .second => second
    | .third => third

def checkHeadActivationArtifact (ir : System.FilePath) : IO HeadActivationBinding.Program := do
  let ptx ← IO.FS.readFile (ir / "activation-fp8.ptx")
  let some program := HeadActivationBinding.Program.check ptx
    | throw (IO.userError "Unrecognized training-head activation quantizer")
  for (original, changed) in [
      ("0f408F5C29", "0f408F5C28"), ("0f43E80000", "0f43E90000"),
      ("cvt.rn.bf16.f32", "cvt.rz.bf16.f32"), ("setp.le.f32", "setp.lt.f32"),
      ("0x5410U", "0x5401U"), ("[ %rd2 + 0 ]", "[ %rd2 + 4 ]")] do
    let mutant := ptx.replace original changed
    if mutant == ptx || (HeadActivationBinding.Program.check mutant).isSome then
      throw (IO.userError s!"Activation quantizer accepted a change to {original}")
  IO.println "Certified the activation's BF16 rounding, E4M3 overflow and packed FP8 stores."
  return program

def readRopeForward (ir : System.FilePath) (n : RopeShift.Length) : IO (AttentionExecution.Forward n) := do
  let original := s!"rope-reference-{n.tokens}"
  let shifted := s!"rope-shifted-{n.tokens}"
  let a : RopeShift.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / (original ++ ".ttgir")),
    ← IO.FS.readFile (ir / (original ++ ".ptx")),
    ← IO.FS.readFile (ir / (shifted ++ ".ttgir")),
    ← IO.FS.readFile (ir / (shifted ++ ".ptx"))⟩
  let some checked := RopeShift.certify n a
    | throw (IO.userError s!"Unrecognized RoPE epilogue for {n.tokens} tokens")
  let other := if n == .t16k then RopeShift.Length.t32k else .t16k
  if (RopeShift.certify other a).isSome then
    throw (IO.userError s!"RoPE accepted a different token extent for {n.tokens} tokens")
  let last := a.shiftedPTX.replace s!"%r26, {12 * n.tokens - 12};" s!"%r26, {12 * n.tokens};"
  for (label, changed) in [
      ("epsilon", { a with referenceIR := a.referenceIR.replace "1.1920929E-7" "1.0000000E-7" }),
      ("norm FMA", { a with shiftedPTX := a.shiftedPTX.replace "%r37, %r37, %r45;" "%r38, %r38, %r45;" }),
      ("reciprocal square root", { a with shiftedPTX := a.shiftedPTX.replace "rsqrt.approx.ftz.f32" "sqrt.approx.ftz.f32" }),
      ("destination pitch", { a with shiftedPTX := a.shiftedPTX.replace "%rd5, 1536;" "%rd5, 1280;" }),
      ("tail key mask", { a with shiftedPTX := last })] do
    if changed == a || (RopeShift.certify n changed).isSome then
      throw (IO.userError s!"RoPE epilogue for {n.tokens} tokens failed its {label} mutation check")
  IO.println s!"Certified shifted RoPE at {n.tokens} tokens, including the original reduction and all store masks."
  return ⟨a, checked⟩

def readRopeBackward (ir : System.FilePath) (n : RopeShift.Length) : IO (AttentionExecution.Backward n) := do
  let original := s!"rope-backward-reference-{n.tokens}"
  let trimmed := s!"rope-backward-trimmed-{n.tokens}"
  let a : RopeBackward.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / (original ++ ".ttgir")),
    ← IO.FS.readFile (ir / (original ++ ".ptx")),
    ← IO.FS.readFile (ir / (trimmed ++ ".ttgir")),
    ← IO.FS.readFile (ir / (trimmed ++ ".ptx"))⟩
  let some checked := RopeBackward.certify n a
    | throw (IO.userError s!"Unrecognized trimmed RoPE backward for {n.tokens} tokens")
  let other := if n == .t16k then RopeShift.Length.t32k else .t16k
  if (RopeBackward.certify other a).isSome then
    throw (IO.userError s!"RoPE backward accepted a different token extent for {n.tokens} tokens")
  let lastChanged := a.trimmedPTX.replace
    s!"%r194, {n.tokens - 1};" s!"%r194, {n.tokens};"
  let firstChanged := a.trimmedIR.replace "arith.cmpi eq, %f1, %cst_7" "arith.cmpi ne, %f1, %cst_7"
  let factorChanged := a.trimmedIR.replace "arith.xori %y, %y_43" "arith.ori %y, %y_43"
  for (label, changed) in [
      ("key successor channel", { a with trimmedPTX := a.trimmedPTX.replace "%p9, %r196, 3;" "%p9, %r196, 4;" }),
      ("rotary factor permutation", { a with trimmedIR := factorChanged }),
      ("first-token gradient", { a with trimmedIR := firstChanged }),
      ("last-token guard", { a with trimmedPTX := lastChanged }),
      ("RMS correction", { a with referencePTX := a.referencePTX.replace "neg.f32" "abs.f32" }),
      ("FP8 conversion", { a with trimmedPTX := a.trimmedPTX.replace "e4m3x2.f32" "e5m2x2.f32" }),
      ("transpose exchange", { a with trimmedPTX := a.trimmedPTX.replace ".trans.shared" ".shared" })] do
    if changed == a || (RopeBackward.certify n changed).isSome then
      throw (IO.userError s!"RoPE backward for {n.tokens} tokens failed its {label} mutation check")
  IO.println s!"Certified RoPE backward at {n.tokens} tokens, including both FP8 stores and the K load masks."
  return ⟨a, checked⟩

def readRopeNarrow (ir : System.FilePath) (shape : RopeBackward.Narrow.Shape) (n : RopeShift.Length) : IO (AttentionExecution.Narrow shape n) := do
  let shapeName := if shape = .paired then "paired" else "half-v"
  let original := s!"rope-backward-{shapeName}-reference-{n.tokens}"
  let vector := s!"rope-backward-{shapeName}-vector-{n.tokens}"
  let a : RopeBackward.Narrow.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / (original ++ ".ttgir")),
    ← IO.FS.readFile (ir / (original ++ ".ptx")),
    ← IO.FS.readFile (ir / (vector ++ ".ttgir")),
    ← IO.FS.readFile (ir / (vector ++ ".ptx"))⟩
  let some checked := RopeBackward.Narrow.certify shape n a
    | throw (IO.userError s!"Unrecognized {shapeName} RoPE backward for {n.tokens} tokens")
  let otherShape := if shape = .paired then RopeBackward.Narrow.Shape.halfValue else .paired
  let otherLength := if n == .t16k then RopeShift.Length.t32k else .t16k
  if (RopeBackward.Narrow.certify otherShape n a).isSome ||
      (RopeBackward.Narrow.certify shape otherLength a).isSome then
    throw (IO.userError s!"RoPE backward accepted the wrong shape or token extent for {shapeName}")
  let factorChanged := a.vectorIR.replace "arith.xori %y," "arith.ori %y,"
  let routingChanged :=
    if shape = .paired then
      a.vectorIR.replace "arith.divsi %logical_head, %c3_i32" "arith.divsi %logical_head, %c2_i32"
    else
      a.vectorIR.replace "arith.muli %logical_head, %c64_i32" "arith.muli %logical_head, %c32_i32"
  let storeChanged := a.vectorPTX.replace ".trans.shared" ".shared"
  for (label, changed) in [
      ("rotary factor permutation", { a with vectorIR := factorChanged }),
      ("head routing", { a with vectorIR := routingChanged }),
      ("FP8 transpose", { a with vectorPTX := storeChanged })] do
    if changed == a || (RopeBackward.Narrow.certify shape n changed).isSome then
      throw (IO.userError s!"RoPE backward {shapeName} {n.tokens} failed its {label} mutation check")
  IO.println s!"Certified {shapeName} RoPE backward at {n.tokens} tokens, including factor and FP8 store routing."
  return ⟨a, checked⟩

def readByteTranspose (ir : System.FilePath) : IO (AttentionExecution.Bytes) := do
  let a : ByteTranspose.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / "byte-transpose-reference.ttir"),
    ← IO.FS.readFile (ir / "byte-transpose-hopper.ttgir"),
    ← IO.FS.readFile (ir / "byte-transpose-hopper.ptx")⟩
  let some checked := ByteTranspose.certify a
    | throw (IO.userError "Unrecognized byte transpose artifacts")
  for (label, changed) in [
      ("reference mask", { a with referenceIR := a.referenceIR.replace "arith.cmpi slt" "arith.cmpi sle" }),
      ("tile axis", { a with hopperIR := a.hopperIR.replace "tt.get_program_id y" "tt.get_program_id x" }),
      ("dimension pitch", { a with hopperPTX := a.hopperPTX.replace "8064," "8192," }),
      ("warp barrier", { a with hopperPTX := a.hopperPTX.replace "bar.warp.sync" "bar.sync" }),
      ("byte permutation", { a with hopperPTX := a.hopperPTX.replace "0x5410U" "0x5401U" })] do
    if (ByteTranspose.certify changed).isSome then
      throw (IO.userError s!"Byte transpose accepted a change to {label}")
  IO.println "Certified the byte transpose and rejected mask, axis, pitch, barrier and byte permutation mutations."
  return ⟨a, checked⟩

private def trainingLengths {p : RopeShift.Length → Type}
    (read : (n : RopeShift.Length) → IO (p n)) : IO ((n : RopeShift.Length) → Option (p n)) := do
  let p16 ← read .t16k
  let p32 ← read .t32k
  let p40 ← read .t40k
  let p48 ← read .t48k
  return fun
    | .t16k => some p16
    | .t32k => some p32
    | .t40k => some p40
    | .t48k => some p48
    | .validation => none

def checkAttentionExecution (ir : System.FilePath) : IO AttentionExecution.Programs := do
  let f16 ← readRopeForward ir .t16k
  let f32 ← readRopeForward ir .t32k
  let f40 ← readRopeForward ir .t40k
  let f48 ← readRopeForward ir .t48k
  let fv ← readRopeForward ir .validation
  let backward ← trainingLengths (readRopeBackward ir)
  let paired ← trainingLengths (readRopeNarrow ir .paired)
  let half ← trainingLengths (readRopeNarrow ir .halfValue)
  let bytes ← readByteTranspose ir
  return {
    forward := fun
      | .t16k => f16
      | .t32k => f32
      | .t40k => f40
      | .t48k => f48
      | .validation => fv
    backward
    narrow := fun
      | .paired => paired
      | .halfValue => half
    bytes
  }

open RopeShift (Length)
open YarnTable (Layout Phase)

def readCompactDecoder (ir : System.FilePath) (layout : Layout) (phase : Phase)
    (n : Length) : IO (YarnExecution.Compact layout phase) := do
  let name := match layout with
    | .wide => "wide"
    | .paired => "paired"
    | .narrow => "half-v"
  let stage := if phase = .forward then "forward" else "backward"
  let leaf := s!"rope-compact-{stage}-{name}-{n.tokens}"
  let ttgir ← IO.FS.readFile (ir / (leaf ++ ".ttgir"))
  let ptx ← IO.FS.readFile (ir / (leaf ++ ".ptx"))
  let some checked := YarnTable.certify layout phase ttgir ptx
    | throw (IO.userError s!"Unrecognized compact YaRN decoder for {leaf}")
  let signChanged := ptx.replace "0x8000;" "0x4000;"
  let decodeChanged := ttgir.replace "tt.join" "tt.cat"
  if signChanged == ptx || (YarnTable.certify layout phase ttgir signChanged).isSome ||
      decodeChanged == ttgir || (YarnTable.certify layout phase decodeChanged ptx).isSome then
    throw (IO.userError s!"Compact YaRN decoder mutation accepted for {leaf}")
  return ⟨ttgir, ptx, checked⟩

def readStationaryDecoder (ir : System.FilePath) (phase : Phase) (n : Length) :
    IO (YarnExecution.Stationary phase) := do
  let stage := if phase = .forward then "forward" else "backward"
  let leaf := s!"rope-stationary-{stage}-{n.tokens}"
  let ttgir ← IO.FS.readFile (ir / (leaf ++ ".ttgir"))
  let ptx ← IO.FS.readFile (ir / (leaf ++ ".ptx"))
  let some checked := YarnStationary.certify phase ttgir ptx
    | throw (IO.userError s!"Unrecognized stationary YaRN decoder for {leaf}")
  let zeroCosine := ttgir.replace "dense<1.000000e+00>" "dense<0.000000e+00>"
  let wrongPitch := ttgir.replace "dense<32>" "dense<64>"
  let wrongSign := ptx.replace "0x8000;" "0x4000;"
  if zeroCosine == ttgir || (YarnStationary.certify phase zeroCosine ptx).isSome ||
      wrongPitch == ttgir || (YarnStationary.certify phase wrongPitch ptx).isSome ||
      wrongSign == ptx || (YarnStationary.certify phase ttgir wrongSign).isSome then
    throw (IO.userError s!"Stationary YaRN decoder mutation accepted for {leaf}")
  return ⟨ttgir, ptx, checked⟩

private def loadDecoderPhases {p : Phase → Type}
    (read : (phase : Phase) → Length → IO (p phase)) : IO (YarnExecution.Decoders p) := do
  let f16 ← read .forward .t16k
  let f32 ← read .forward .t32k
  let f40 ← read .forward .t40k
  let f48 ← read .forward .t48k
  let fv ← read .forward .validation
  let b16 ← read .backward .t16k
  let b32 ← read .backward .t32k
  let b40 ← read .backward .t40k
  let b48 ← read .backward .t48k
  return {
    forward := fun
      | .t16k => f16
      | .t32k => f32
      | .t40k => f40
      | .t48k => f48
      | .validation => fv
    backward := fun
      | .t16k => some b16
      | .t32k => some b32
      | .t40k => some b40
      | .t48k => some b48
      | .validation => none
  }

def checkYarnExecution (ir : System.FilePath) : IO YarnExecution.Programs := do
  let wide ← loadDecoderPhases (readCompactDecoder ir .wide)
  let paired ← loadDecoderPhases (readCompactDecoder ir .paired)
  let narrow ← loadDecoderPhases (readCompactDecoder ir .narrow)
  let stationary ← loadDecoderPhases (readStationaryDecoder ir)
  IO.println "Certified all 36 compact and stationary YaRN decoder launches."
  return { compact := fun | .wide => wide | .paired => paired | .narrow => narrow, stationary }

def checkCETransposeArtifacts (ir : System.FilePath) : IO CETranspose.Program := do
  let gir ← IO.FS.readFile (ir / "ce-transpose-pair.ttgir")
  let ptx ← IO.FS.readFile (ir / "ce-transpose-pair.ptx")
  let some checked := CETranspose.Program.check gir ptx
    | throw (IO.userError "Unrecognized paired CE transpose")
  let assembly := String.intercalate "\n" (QKVTTIR.normalizePTX ptx)
  for (old, replacement) in [("setp.lt.u32 %p1, %r33, 6;",
        "setp.lt.u32 %p1, %r33, 7;"),
      ("selp.b64 %rd12, %rd9, %rd11, %p1;",
        "selp.b64 %rd12, %rd11, %rd9, %p1;"),
      ("selp.b64 %rd14, %rd10, %rd13, %p1;",
        "selp.b64 %rd14, %rd13, %rd10, %p1;"),
      ("add.s32 %r42, %r41, -768;",
        "add.s32 %r42, %r41, -640;"),
      ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r84, %r85, %r86, %r87}, [%r79+256];",
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r84, %r85, %r86, %r87}, [%r79+272];"),
      ("xor.b32 %r66, %r59, 16;",
        "xor.b32 %r66, %r59, 32;")] do
    let mutated := assembly.replace old replacement
    if mutated == assembly || (CETranspose.certify gir mutated).isSome then
      throw (IO.userError s!"Paired CE transpose accepted PTX mutation {old}")
  let widthMutated := gir.replace "arith.constant 768 : i32" "arith.constant 640 : i32"
  if widthMutated == gir || (CETranspose.certify widthMutated ptx).isSome then
    throw (IO.userError "Paired CE transpose accepted a changed TTGIR activation stride")
  IO.println "Certified both CE transpose ranges, the widened loads and the two shared-memory phases."
  return checked

private def checkBf16TransposeArtifact (ir : System.FilePath) (name : String) (mode : Bf16Transpose.Mode) :
    IO (Bf16Transpose.Program mode) := do
  let stem := "bf16-transpose-" ++ name
  let a : Bf16Transpose.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / (stem ++ "-reference.ttir")),
    ← IO.FS.readFile (ir / (stem ++ "-reference.ptx")),
    ← IO.FS.readFile (ir / (stem ++ "-hopper.ttgir")),
    ← IO.FS.readFile (ir / (stem ++ "-hopper.ptx"))⟩
  let some program := Bf16Transpose.Program.check mode a
    | throw (IO.userError s!"Unrecognized BF16 transpose {name}")
  let other := if mode == .copy then Bf16Transpose.Mode.add else .copy
  if (Bf16Transpose.certify other a).isSome then
    throw (IO.userError s!"BF16 transpose {name} accepted the opposite launch mode")
  let mutations := [
    ("bounds", { a with referenceIR := a.referenceIR.replace "arith.cmpi slt" "arith.cmpi sle" }),
    ("baseline layout", { a with referencePTX := a.referencePTX.replace ".trans.shared" ".shared" }),
    ("load axis", { a with hopperIR := a.hopperIR.replace "tt.get_program_id y" "tt.get_program_id x" }),
    ("matrix store orientation", { a with hopperPTX := a.hopperPTX.replace ".x4.trans.shared" ".x4.shared" }),
    ("shared read width", { a with hopperPTX := a.hopperPTX.replace "ld.shared.v4.b32" "ld.shared.v2.b32" }),
    ("warp barrier", { a with hopperPTX := a.hopperPTX.replace "bar.warp.sync" "bar.sync" })]
  let mutations := if mode == .add then
    let refOrder := a.referencePTX.replace "%r34, %r18, %r134;" "%r34, %r134, %r18;"
    let hopperOrder := a.hopperPTX.replace "%r49, %r17, %r18;" "%r49, %r18, %r17;"
    mutations ++ [("reference add order", { a with referencePTX := refOrder }),
      ("Hopper add order", { a with hopperPTX := hopperOrder })]
    else mutations
  for (label, changed) in mutations do
    if changed == a || (Bf16Transpose.certify mode changed).isSome then
      throw (IO.userError s!"BF16 transpose {name} failed its {label} mutation check")
  IO.println s!"Certified BF16 transpose {name}, including the vector loads and transposed shared stores."
  return program

def checkBf16TransposeArtifacts (ir : System.FilePath) : IO Bf16Transpose.Programs := do
  let copy ← checkBf16TransposeArtifact ir "copy" .copy
  let add ← checkBf16TransposeArtifact ir "add" .add
  return fun | .copy => copy | .add => add

def checkTailLerpArtifacts (ir : System.FilePath) :
    IO ((a : TailLerp.Artifacts) × TailLerp.Certificate a) := do
  let a : TailLerp.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / "tail-lerp-single.ttgir"),
    ← IO.FS.readFile (ir / "tail-lerp-merged.ttgir"),
    ← IO.FS.readFile (ir / "tail-lerp-single.ptx"),
    ← IO.FS.readFile (ir / "tail-lerp-merged.ptx")⟩
  let some cert := TailLerp.certify a
    | throw (IO.userError "Unrecognized tail lerp artifacts")
  for changed in [
      { a with candidateIR := a.candidateIR.replace "tt.splat %rate_second" "tt.splat %rate_first" },
      { a with candidatePTX := a.candidatePTX.replace "fma.rn.f32" "fma.rz.f32" },
      { a with candidatePTX := a.candidatePTX.replace "st.global.v4.b32" "st.global.v2.b32" },
      { a with referencePTX := a.referencePTX.replace "sub.f32" "add.f32" }] do
    if changed == a || (TailLerp.certify changed).isSome then
      throw (IO.userError "Tail lerp accepted a mutated rate, rounding, store, or subtraction")
  IO.println "Certified tail lerp artifacts and rejected rate, rounding, store, and subtraction mutations."
  pure ⟨a, cert⟩

def checkTailNormArtifacts (ir : System.FilePath) :
    IO ((code : String) × (ptx : String) × TailNorm.Certificate code ptx) := do
  let code ← IO.FS.readFile (ir / "tail-norm.ttgir")
  let ptx ← IO.FS.readFile (ir / "tail-norm.ptx")
  let some cert := TailNorm.certify code ptx
    | throw (IO.userError "Unrecognized norm restoration artifacts")
  for (changedIR, changedPTX) in [
      (code.replace "18864" "18863", ptx),
      (code, ptx.replace "3, 4, %p4;" "4, 3, %p4;"),
      (code, ptx.replace "mul.f32" "add.f32"),
      (code, ptx.replace "cvt.rn.bf16x2.f32" "cvt.rz.bf16x2.f32")] do
    if (changedIR == code && changedPTX == ptx) || (TailNorm.certify changedIR changedPTX).isSome then
      throw (IO.userError "Norm restoration accepted a mutated tile edge, ratio, arithmetic, or rounding")
  IO.println "Certified norm restoration and rejected tile, ratio, arithmetic, and rounding mutations."
  pure ⟨code, ptx, cert⟩

def checkSparsePacketArtifacts (ir : System.FilePath) : IO (Σ a, SparsePackets.Certificate a) := do
  let a : SparsePackets.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / "sparse-value.ttgir"),
    ← IO.FS.readFile (ir / "sparse-value.ptx"),
    ← IO.FS.readFile (ir / "sparse-value-wire.ttgir"),
    ← IO.FS.readFile (ir / "sparse-value-wire.ptx"),
    ← IO.FS.readFile (ir / "sparse-ngram.ttgir"),
    ← IO.FS.readFile (ir / "sparse-ngram.ptx"),
    ← IO.FS.readFile (ir / "sparse-value-clear.ttgir"),
    ← IO.FS.readFile (ir / "sparse-value-clear.ptx")⟩
  let some checked := SparsePackets.certify a
    | throw (IO.userError "Unrecognized sparse packet artifacts")
  for changed in [
      { a with valueIR := a.valueIR.replace "tt.assert" "tt.assume" },
      { a with valuePTX := a.valuePTX.replace "%rd2, %rd3, %p3" "%rd3, %rd2, %p3" },
      { a with valueWirePTX := a.valueWirePTX.replace "%ctaid.x" "%ctaid.y" },
      { a with valueWirePTX := a.valueWirePTX.replace "st.global.v4.b32" "st.global.v2.b32" },
      { a with ngramPTX := a.ngramPTX.replace "cvt.rn.bf16.f32" "cvt.rz.bf16.f32" },
      { a with ngramIR := a.ngramIR.replace "arith.cmpi slt" "arith.cmpi sle" },
      { a with clearPTX := a.clearPTX.replace "mov.b32 \t%r2, 0;" "mov.b32 \t%r2, 1;" },
      { a with clearIR := a.clearIR.replace "arith.cmpi slt" "arith.cmpi sle" }] do
    if changed == a || (SparsePackets.certify changed).isSome then
      throw (IO.userError "Sparse packets accepted a mutated assert, destination, rounding, or row boundary")
  IO.println "Certified full-row sparse packets, the value wire-only path, and the sent row clear."
  return ⟨a, checked⟩

def checkNgramAccumulationArtifact (ir : System.FilePath) : IO (Σ body : String, Σ ptx : String, NgramAccumulation.Certificate body ptx) := do
  let body ← IO.FS.readFile (ir / "ngram-accumulate.ttgir")
  let ptx ← IO.FS.readFile (ir / "ngram-accumulate.ptx")
  let some checked := NgramAccumulation.certify body ptx
    | throw (IO.userError "Unrecognized n-gram backward accumulation artifact")
  for (old, new) in [("cvt.f32.bf16", "cvt.f32.f16"), ("cvt.rn.f16.f32", "cvt.rz.f16.f32"),
      ("add.noftz.v2.f16", "add.noftz.v2.bf16"), ("%rd3 + 0 ]", "%rd2 + 0 ]")] do
    let changed := ptx.replace old new
    if changed == ptx || (NgramAccumulation.certify body changed).isSome then
      throw (IO.userError s!"N-gram backward accumulation accepted a mutation at {old}")
  IO.println "Certified the BF16 sink scatter on FP16 slots and checked its conversion, atomic, and destination."
  return ⟨body, ptx, checked⟩

def checkNgramScratchArtifacts (ir : System.FilePath) : IO (Σ a, NgramScratch.Certificate a) := do
  let a : NgramScratch.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / "ngram-scratch-peer-clear.ttgir"),
    ← IO.FS.readFile (ir / "ngram-scratch-peer-clear.ptx"),
    ← IO.FS.readFile (ir / "ngram-scratch-own-clear.ttgir"),
    ← IO.FS.readFile (ir / "ngram-scratch-own-clear.ptx")⟩
  let some checked := NgramScratch.certify a
    | throw (IO.userError "Unrecognized n-gram scratch consumers")
  for (name, changed) in [
      ("peer clear sign", {a with peerPTX := a.peerPTX.replace "%r5, 0;" "%r5, 0x80008000;"}),
      ("peer clear address", {a with peerPTX := a.peerPTX.replace "[ %rd1 + 0 ], { %r5" "[ %rd2 + 0 ], { %r5"}),
      ("owner clear width", {a with ownPTX := a.ownPTX.replace "st.global.v4.b32" "st.global.v2.b32"}),
      ("owner merge type", {a with ownPTX := a.ownPTX.replace "add.noftz.v8.bf16" "add.noftz.v8.f16"}),
      ("owner tail", {a with ownPTX := a.ownPTX.replace "setp.lt.s32" "setp.le.s32"}),
      ("owner clear sign", {a with ownPTX := a.ownPTX.replace "%r7, 0;" "%r7, 0x80008000;"})] do
    if changed == a || (NgramScratch.certify changed).isSome then
      throw (IO.userError s!"N-gram scratch consumers accepted a mutation at {name}")
  IO.println "Certified both n-gram scratch consumers, including the gradient payloads and positive-zero stores."
  return ⟨a, checked⟩

def checkSparseLandingArtifacts (ir : System.FilePath) : IO (Σ a, SparseLanding.Certificate a) := do
  let a : SparseLanding.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / "rows-serve.ttgir"),
    ← IO.FS.readFile (ir / "rows-serve.ptx"),
    ← IO.FS.readFile (ir / "rows-land.ttgir"),
    ← IO.FS.readFile (ir / "rows-land.ptx")⟩
  let some checked := SparseLanding.certify a
    | throw (IO.userError "Unrecognized sparse row fetch artifacts")
  for changed in [
      { a with serveIR := a.serveIR.replace "tt.assert" "tt.assume" },
      { a with servePTX := a.servePTX.replace "mad.wide.s32" "mad.lo.s32" },
      { a with landPTX := a.landPTX.replace "%rd1, %rd2, %p1" "%rd2, %rd1, %p1" },
      { a with landIR := a.landIR.replace "arith.cmpi slt" "arith.cmpi sle" }] do
    if changed == a || (SparseLanding.certify changed).isSome then
      throw (IO.userError "Sparse row fetch accepted a mutated assert, address, bank, or row boundary")
  IO.println "Certified sparse row fetch and rejected assert, address, bank, and boundary mutations."
  return ⟨a, checked⟩

def checkMomentReplayArtifacts (ir : System.FilePath) : IO (Σ a, MomentReplay.Certificate a) := do
  let a : MomentReplay.Artifacts ← pure ⟨
    ← IO.FS.readFile (ir / "ngram-replay-reference.ttgir"),
    ← IO.FS.readFile (ir / "ngram-replay-reference.ptx"),
    ← IO.FS.readFile (ir / "ngram-replay-fallback.ttgir"),
    ← IO.FS.readFile (ir / "ngram-replay-fallback.ptx"),
    ← IO.FS.readFile (ir / "ngram-replay-grouped.ttgir"),
    ← IO.FS.readFile (ir / "ngram-replay-grouped.ptx"),
    ← IO.FS.readFile (ir / "ngram-replay-zero.ttgir"),
    ← IO.FS.readFile (ir / "ngram-replay-zero.ptx")⟩
  let some checked := MomentReplay.certify a
    | throw (IO.userError "Unrecognized n-gram moment replay artifacts")
  for changed in [
      { a with referencePTX := a.referencePTX.replace "mul.f32" "add.f32" },
      { a with fallbackPTX := a.fallbackPTX.replace "mul.f32" "add.f32" },
      { a with groupedPTX := a.groupedPTX.replace "redux.sync.min.s32" "redux.sync.max.s32" },
      { a with groupedPTX := a.groupedPTX.replace "setp.gt.s64" "setp.ge.s64" },
      { a with groupedIR := a.groupedIR.replace "arith.cmpi slt" "arith.cmpi sle" },
      { a with groupedZeroPTX := a.groupedZeroPTX.replace "%p4, %r22, 0;" "%p4, %r22, 0x80000000;" }] do
    if changed == a || (MomentReplay.certify changed).isSome then
      throw (IO.userError "Moment replay accepted a mutated multiply, minimum, event, or row boundary")
  IO.println "Certified n-gram moment replay, including positive-zero detection and the scalar fallback."
  return ⟨a, checked⟩

def checkNgramAdamArtifacts (ir : System.FilePath) : IO (Σ a, NgramAdam.Certificate a) := do
  let load (name : String) : IO NgramAdam.Code := do
    let stem := "ngram-adam-" ++ name
    pure ⟨← IO.FS.readFile (ir / (stem ++ ".ttgir")), ← IO.FS.readFile (ir / (stem ++ ".ptx"))⟩
  let a : NgramAdam.Artifacts ←
    pure ⟨← load "pack", ← load "prepare", ← load "update", ← load "update-zero",
      ← load "scatter-own", ← load "scatter-peer", ← load "scatter-peer-tiled"⟩
  let some checked := NgramAdam.certify a
    | throw (IO.userError "Unrecognized n-gram Adam artifacts")
  let mutate (code : NgramAdam.Code) (old replacement : String) :=
    { code with ptx := code.ptx.replace old replacement }
  for (name, changed) in [
      ("packed index selection", { a with pack := mutate a.pack "%rd5, %rd4, %p3" "%rd4, %rd5, %p3" }),
      ("row claim atomic", { a with prepare := mutate a.prepare "atom.global.gpu.relaxed.exch.b32" "atom.global.gpu.relaxed.add.u32" }),
      ("gradient zero sign", { a with prepare := mutate a.prepare "%r5, 0;" "%r5, 0x80008000;" }),
      ("gradient store width", { a with prepare := mutate a.prepare "st.global.v4.b32" "st.global.v2.b32" }),
      ("Adam FMA rounding", { a with update := mutate a.update "fma.rn.f32" "fma.rz.f32" }),
      ("Adam decay gate", { a with update := mutate a.update "setp.gt.f32" "setp.ge.f32" }),
      ("Adam owner gate", { a with updateZero := mutate a.updateZero "@%p1 bra" "@!%p1 bra" }),
      ("Adam zero gate", { a with updateZero := mutate a.updateZero "%p2, %r152, 0;" "%p2, %r152, 0x80000000;" }),
      ("local scatter conversion", { a with scatterOwn := mutate a.scatterOwn "cvt.rn.bf16.f32" "cvt.rz.bf16.f32" }),
      ("peer scatter atomic", { a with scatterPeer := mutate a.scatterPeer ".add.noftz.v2.bf16" ".max.noftz.v2.bf16" }),
      ("tiled peer atomic", { a with scatterPeerTiled := mutate a.scatterPeerTiled ".add.noftz.v8.bf16" ".add.noftz.v8.f16" }),
      ("tiled peer tail", { a with scatterPeerTiled := mutate a.scatterPeerTiled "setp.lt.s32" "setp.le.s32" }),
      ("tiled peer grid", { a with scatterPeerTiled := mutate a.scatterPeerTiled "%ctaid.x" "%ctaid.y" })] do
    if changed == a then
      throw (IO.userError s!"N-gram Adam mutation target absent for {name}")
    if (NgramAdam.certify changed).isSome then
      throw (IO.userError s!"N-gram Adam accepted a mutation of {name}")
  IO.println "Certified n-gram Adam entry preparation, owner and peer scatters, and the weight, moment, and stamp update."
  return ⟨a, checked⟩

def checkValueAdamArtifacts (ir : System.FilePath) : IO (Σ a, ValueAdam.Certificate a) := do
  let load (name : String) : IO ValueAdam.Code := do
    let stem := "value-adam-" ++ name
    pure ⟨← IO.FS.readFile (ir / (stem ++ ".ttgir")), ← IO.FS.readFile (ir / (stem ++ ".ptx"))⟩
  let a : ValueAdam.Artifacts ←
    pure ⟨← load "reference", ← load "direct", ← load "marked", ← load "own", ← load "own-marked",
      ← load "scale", ← load "clear"⟩
  let some checked := ValueAdam.certify a
    | throw (IO.userError "Unrecognized value Adam scatter, scale, or clear artifacts")
  let changePTX (c : ValueAdam.Code) (old replacement : String) :=
    { c with ptx := c.ptx.replace old replacement }
  for changed in [
      { a with direct := changePTX a.direct "sub.s32" "add.s32" },
      { a with direct := changePTX a.direct ".add.noftz.v2.f16" ".min.noftz.v2.f16" },
      { a with reference := { a.reference with ir := a.reference.ir.replace "fadd, relaxed" "fmax, relaxed" } },
      { a with marked := changePTX a.marked ".relaxed.max.s32" ".relaxed.min.s32" },
      { a with own := changePTX a.own "[ %rd2 + 0 ], { %r3 }" "[ %rd3 + 0 ], { %r3 }" },
      { a with ownMarked := changePTX a.ownMarked "mov.b32 \t%r3, 0;" "mov.b32 \t%r3, 1;" },
      { a with scale := changePTX a.scale "setp.eq.b32" "setp.ne.b32" },
      { a with scale := changePTX a.scale "0f3e000000" "0f3f000000" },
      { a with scale := changePTX a.scale "cvt.rn.f16.f32" "cvt.rz.f16.f32" },
      { a with clear := changePTX a.clear "setp.eq.b32" "setp.ne.b32" },
      { a with clear := changePTX a.clear "bar.sync" "bar.warp.sync" }] do
    if changed == a || (ValueAdam.certify changed).isSome then
      throw (IO.userError "Value Adam accepted a mutated offset, atomic operation, scale, or clear barrier")
  IO.println "Certified value Adam scatters, claim-selected FP16 scaling, and workspace clears."
  return ⟨a, checked⟩

def checkValueEmbeddingArtifacts (ir : System.FilePath) : IO (Σ a, ValueEmbedAccum.Certificate a) := do
  let load (name : String) : IO ValueEmbedAccum.Code := do
    let stem := "value-embed-" ++ name
    pure ⟨← IO.FS.readFile (ir / (stem ++ ".ttgir")), ← IO.FS.readFile (ir / (stem ++ ".ptx"))⟩
  let a : ValueEmbedAccum.Artifacts ←
    pure ⟨← load "reference", ← load "four", ← load "reference-debug", ← load "four-debug"⟩
  let some checked := ValueEmbedAccum.certify a
    | throw (IO.userError "Unrecognized value embedding backward kernels")
  let alter (code : ValueEmbedAccum.Code) (old next : String) :=
    { code with ptx := code.ptx.replace old next }
  for (name, changed) in [
      ("plane separation", { a with four := alter a.four "%r16, 50304," "%r16, 50303," }),
      ("gradient cast", { a with four := alter a.four "cvt.rn.f16.f32" "cvt.rz.f16.f32" }),
      ("vector atomics", { a with four := alter a.four ".v8.f16" ".v4.f16" }),
      ("invalid token", { a with fourDebug := alter a.fourDebug "%p1, %r2, 50304;" "%p1, %r2, 50305;" })] do
    if changed == a || (ValueEmbedAccum.certify changed).isSome then
      throw (IO.userError s!"Value embedding backward accepted a mutation of {name}")
  IO.println "Certified both value embedding backward layouts and the debug token bounds."
  return ⟨a, checked⟩

private def checkGraphSlotArtifact (ir : System.FilePath) (shape : GraphStage.Shape) :
    IO (GraphStage.Program shape) := do
  let name := s!"stage-{shape.tokens}-{shape.ends}-{shape.mtp}.ptx"
  let plainPTX ← IO.FS.readFile (ir / name)
  let some plain := GraphPlainPTX.Program.check shape plainPTX
    | throw (IO.userError s!"Unrecognized graph copy in {name}")
  for (old, changed) in [
      ("%r1, 9;", "%r1, 8;"),
      ("ld.global.v2.b64", "ld.global.v2.b32"),
      ("ld.global.v4.b32", "ld.global.v2.b32"),
      ("st.global.v4.b32 [ %rd28 + 0 ]", "st.global.v4.b32 [ %rd27 + 0 ]"),
      ("@%p4 st.global.b32", "@%p3 st.global.b32")] do
    let faulty := plainPTX.replace old changed
    if faulty == plainPTX || (GraphPlainPTX.Program.check shape faulty).isSome then
      throw (IO.userError s!"Graph copy accepted a mutation at {old} in {name}")
  let name := s!"stage-with-hashed-rows-{shape.tokens}-{shape.ends}-{shape.mtp}.ptx"
  let hashedPTX ← IO.FS.readFile (ir / name)
  let some hashed := GraphSlotPTXBinding.certify shape hashedPTX
    | throw (IO.userError s!"Unrecognized fused graph staging launch in {name}")
  for (old, changed) in [
      ("%r70, %r1, 8;", "%r70, %r1, 7;"),
      ("ld.global.v2.b64 { %rd30, %rd31 }", "ld.global.v2.b32 { %rd30, %rd31 }"),
      ("%r77, %r2, 5, 2;", "%r77, %r2, 4, 2;"),
      ("@%p19 st.global.b32", "@%p20 st.global.b32"),
      ("%rd41 + 0 ], { %r76 }", "%rd39 + 0 ], { %r76 }"),
      ("%r21, %r7, 15;", "%r21, %r7, 14;"),
      ("%rd16, %rd15, 4;", "%rd16, %rd15, 8;"),
      ("@%p2 ld.global.b32", "@%p3 ld.global.b32"),
      ("mov.u32 %r6, %r7;", "mov.u32 %r6, 0x0;"),
      ("%p10, %r6, %r7;", "%p10, %r7, %r6;"),
      ("%r15, %r67, %r62, %p17;", "%r15, %r62, %r67, %p17;"),
      ("st.global.b32 [ %rd26 + 0 ]", "st.global.b32 [ %rd25 + 0 ]")] do
    let faulty := hashedPTX.replace old changed
    if faulty == hashedPTX || (GraphSlotPTXBinding.certify shape faulty).isSome then
      throw (IO.userError s!"Fused graph staging accepted a mutation at {old} in {name}")
  let name := s!"stage-with-member-rows-{shape.tokens}-{shape.ends}-{shape.mtp}.ptx"
  let memberPTX ← IO.FS.readFile (ir / name)
  let some member := GraphSlotPTXBinding.certifyMember shape memberPTX
    | throw (IO.userError s!"Unrecognized fused member lookup in {name}")
  for (old, changed) in [
      ("%r21, %r4, 15;", "%r21, %r4, 14;"),
      ("%r22, %r6, -1;", "%r22, %r6, 0;"),
      ("min.u32", "max.u32"),
      ("%r23, %r5, 127;", "%r23, %r5, 126;"),
      ("ld.global.b32 { %r7 }, [ %rd17 + 0 ];", "ld.global.b32 { %r7 }, [ %rd16 + 0 ];"),
      ("st.global.b32 [ %rd26 + 0 ]", "st.global.b32 [ %rd25 + 0 ]")] do
    let faulty := memberPTX.replace old changed
    if faulty == memberPTX || (GraphSlotPTXBinding.certifyMember shape faulty).isSome then
      throw (IO.userError s!"Member lookup accepted a mutation at {old} in {name}")
  return ⟨plain, hashedPTX, memberPTX, hashed, member⟩

def checkGraphSlotArtifacts (ir : System.FilePath) : IO GraphStage.Programs := do
  let read := checkGraphSlotArtifact ir
  let initial ← read .initial
  let middle ← read .middle
  let bulk ← read .bulk
  let taper ← read .taper
  let extension ← read .extension
  IO.println "Certified the ordinary graph copies and both fused lookup variants in all five SM90a shapes."
  return fun
    | .initial => initial
    | .middle => middle
    | .bulk => bulk
    | .taper => taper
    | .extension => extension

def checkNgramEmbedArtifacts (ir : System.FilePath) : IO (Σ a, NgramEmbedBinding.Certificate a) := do
  let load (kind : String) := IO.FS.readFile (ir / ("ngram-embed-" ++ kind ++ ".ptx"))
  let a : NgramEmbedBinding.Artifacts ←
    pure ⟨← load "flat-fwd", ← load "flat-bwd", ← load "full-fwd", ← load "full-bwd"⟩
  let some checked := NgramEmbedBinding.certify a
    | throw (IO.userError "Unrecognized aligned n-gram embedding PTX")
  for (layout, forward, ptx, old, replacement) in [
      (NgramEmbedBinding.Layout.flat, true, a.flatForward, "715827883", "715827882"),
      (.flat, false, a.flatBackward, "mul.bf16x2", "add.bf16x2"),
      (.fullRows, true, a.fullForward, "fma.rn.f32", "fma.rz.f32"),
      (.fullRows, true, a.fullForward, "5435", "5434"),
      (.fullRows, false, a.fullBackward, "512;", "256;"),
      (.fullRows, false, a.fullBackward, "st.global.v4.b32", "st.global.v2.b32")] do
    let mutant := ptx.replace old replacement
    if mutant == ptx || NgramEmbedBinding.recognizes layout forward mutant then
      throw (IO.userError s!"N-gram embedding accepted changed instruction text at {old}")
  IO.println "Certified the flat and full-row n-gram embeddings and both BF16 gradient planes."
  return ⟨a, checked⟩

private def checkTrainingCEGatherArtifact (ir : System.FilePath) (width : TrainingCE.Width)
    (predictions : Fin 4) : IO (TrainingCEExecution.FixedProgram width predictions) := do
  let name := s!"fixed-{width.columns}-{predictions.val}.ptx"
  let ptx ← IO.FS.readFile (ir / "training-ce" / name)
  let mode := width.defaultSoftcap
  let some checked := TrainingCEExecution.FixedProgram.check width predictions ptx
    | throw (IO.userError s!"Unrecognized fixed training CE execution for {name}")
  let instructions := QKVTTIR.normalizePTX ptx
  let some read := instructions.find? (·.startsWith "ld.global.v2.u32 ")
    | throw (IO.userError s!"Missing coherent logit read in {name}")
  let barriers := instructions.zipIdx.filter fun (line, _) => line == "bar.sync 0;"
  let [(_, first), (_, sumReady), _] := barriers | throw (IO.userError s!"Wrong CE barriers in {name}")
  let late := String.intercalate "\n"
    (instructions.take (sumReady + 1) ++ [read] ++ instructions.drop (sumReady + 1))
  for altered in [late, ptx.replace "ld.global.v2.u32" "ld.global.nc.v2.u32",
      ptx.replace "[ce_fwd_bwd_kernel_param_0]" "[ce_fwd_bwd_kernel_param_6]"] do
    if altered == ptx || TrainingCEGatherBinding.canReuseLogits altered then
      throw (IO.userError s!"Training CE accepted unsafe logit reuse in {name}")
  let some (.registers sigma scaled) := TrainingCEPublishBinding.logitTransport instructions
    | throw (IO.userError s!"Missing live sigmoid and scale registers in {name}")
  for reg in [sigma, scaled] do
    let clobbered := String.intercalate "\n"
      (instructions.take (first + 1) ++ [s!"mov.f32 {reg}, 0f00000000;"] ++ instructions.drop (first + 1))
    if (TrainingCEInitBinding.certify width clobbered mode .lmHead).isSome then
      throw (IO.userError s!"Training CE accepted clobbering live register {reg} in {name}")
  let swapped := instructions.map fun line =>
    match TrainingCEGatherBinding.operands line with
    | ["st.shared.v2.f32", ptr, lo, hi] => "st.shared.v2.f32 " ++ ptr ++ ", {" ++ hi ++ ", " ++ lo ++ "};"
    | _ => line
  if swapped == instructions || (TrainingCEPublishBinding.certify
      (String.intercalate "\n" swapped) predictions mode).isSome then
    throw (IO.userError s!"Training CE accepted exchanging the sigmoid and probability in {name}")
  let initialization := checked.checked.init
  let expectedSlots := if width = .sampled10 then predictions.val else 0
  if initialization.targetSlots != expectedSlots then
    throw (IO.userError s!"Wrong training CE target cache size for {name}")
  if expectedSlots > 0 then
    let symbol := TrainingCEPublishBinding.cachedTargetsSymbol
    let lines := QKVTTIR.normalizePTX ptx
    let changed := lines.map fun line =>
      match TrainingCEGatherBinding.operands line with
      | ["st.shared.u32", pointer, src] =>
        if pointer == s!"[{symbol}]" then s!"st.shared.u32 [{symbol}+4], {src};" else line
      | _ => line
    if changed == lines ||
        (TrainingCEInitBinding.certify width (String.intercalate "\n" changed) mode .lmHead).isSome then
      throw (IO.userError s!"Training CE target cache accepted relocating slot zero in {name}")
  let scanShift := ptx.replace "+2048]" "+2049]"
  if scanShift == ptx || (TrainingCEGatherBinding.certifyScan width scanShift).isSome then
    throw (IO.userError s!"Training CE scan accepted shifting its input or staging address in {name}")
  for (old, changed) in [("mov.b32 {low, high}, halves", "mov.b32 {high, low}, halves"),
      ("rcp.approx.ftz.f32", "rcp.approx.f32"), ("0f40F00000", "0f40E00000"),
      ("0f395B6DB7", "0f395B6DB8")] do
    let mutant := ptx.replace old changed
    if mutant == ptx || (TrainingCEInitBinding.certify width mutant mode .lmHead).isSome then
      throw (IO.userError s!"Training CE table initialization accepted changing {old} in {name}")
  let vectorReads := (QKVTTIR.normalizePTX ptx).filter (·.startsWith "ld.shared.v4.f32 ")
  let vectorMutant := String.intercalate "\n" ((QKVTTIR.normalizePTX ptx).map fun line =>
    if line.startsWith "ld.shared.v4.f32 " then line.replace "+16]" "+12]" else line)
  if vectorReads.length != 2 || !vectorReads.any (fun line => line.contains "+16]") ||
      (TrainingCEPublishBinding.certify vectorMutant predictions mode).isSome then
    throw (IO.userError s!"Training CE warp-sum vector placement check failed for {name}")
  for (old, changed) in [("setp.leu.ftz.f32", "setp.le.ftz.f32"), (", 255;", ", 127;")] do
    let mutant := ptx.replace old changed
    if mutant == ptx || (TrainingCESparseBinding.certify width mutant predictions mode).isSome then
      throw (IO.userError s!"Training CE sparse correction accepted changing {old} in {name}")
  let lines := QKVTTIR.normalizePTX ptx
  let changed := lines.map fun line =>
    match TrainingCEGatherBinding.operands line with
    | ["setp.eq.s64", dst, left, _] => s!"setp.eq.s64 {dst}, {left}, {left};"
    | _ => line
  if changed == lines || (TrainingCESparseBinding.certify width
      (String.intercalate "\n" changed) predictions mode).isSome then
    throw (IO.userError s!"Training CE sparse correction accepted unconditional target equality in {name}")
  for (old, changed) in [(", 16;", ", 15;"), (", 31;", ", 30;")] do
    let mutant := ptx.replace old changed
    if mutant == ptx || (TrainingCEReductionBinding.certify mutant).isSome then
      throw (IO.userError s!"Training CE reduction accepted changing {old} in {name}")
  for (old, changed) in [("setp.ge.s64", "setp.gt.s64"),
      ("[ce_fwd_bwd_kernel_param_5]", "[ce_fwd_bwd_kernel_param_6]")] do
    let mutant := ptx.replace old changed
    if mutant == ptx || (TrainingCELossBinding.certify width mutant predictions mode).isSome then
      throw (IO.userError s!"Training CE loss accepted changing {old} in {name}")
  for (old, changed) in [("0f3FB8AA3B;", "0f3FB8AA3A;"), ("0f3F317218,", "0f3F317219,"),
      ("mul.rn.ftz.f32", "add.ftz.f32")] do
    let mutant := ptx.replace old changed
    if mutant == ptx || (TrainingCEPublishBinding.certify mutant predictions mode).isSome then
      throw (IO.userError s!"Training CE publication accepted changing {old} in {name}")
  for (old, changed) in [("17472;", "17473;"), ("30212;", "30208;"),
      ("st.shared.u8", "st.shared.u16"), ("+6144]", "+6145]")] do
    let mutant := ptx.replace old changed
    if mutant == ptx || (TrainingCEGatherBinding.certify mutant).isSome then
      throw (IO.userError s!"Training CE gather accepted changing {old} in {name}")
  let stride := ptx.replace ", 8192;" ", 8193;"
  if stride == ptx || (TrainingCEGatherBinding.certify stride).isNone ||
      (TrainingCEGatherBinding.certifyGeometry width stride).isSome then
    throw (IO.userError s!"Training CE placement mutation check failed for {name}")
  return checked

def checkTrainingCEGatherArtifacts (ir : System.FilePath) : IO TrainingCEExecution.Programs := do
  let read := checkTrainingCEGatherArtifact ir
  let checked : TrainingCEExecution.Programs :=
    ⟨← read .sampled10 2, ← read .sampled10 3, ← read .sampled14 1, ← read .sampled24 1, ← read .full 1⟩
  IO.println "Certified the default softcap and head gradient scale, table initialization, logit scan, reductions, loss and E5M2 stores in the five training CE PTX variants."
  return checked

private def shapeName : ValidationCE.Tile → String
  | .x1r1024 => "1-1024-8"
  | .x1r512 => "1-512-8"
  | .x16r32 => "16-32-4"
  | .x64r64 => "64-64-16"
  | .x64r4 => "64-4-8"
  | .x8r512 => "8-512-16"
  | .x8r256 => "8-256-16"

private def checkValidationCEArtifact (artifacts : System.FilePath) (source : String)
    (tile : ValidationCE.Tile) : IO (ValidationProjectionBinding.HeadProgram tile) := do
  let shape := shapeName tile
  let ptx ← IO.FS.readFile (artifacts / (shape ++ ".ptx"))
  let some checked := ValidationProjectionBinding.HeadProgram.check tile source ptx
    | throw (IO.userError s!"Unrecognized validation wrapper or loss execution for {shape}")
  if shape == "1-1024-8" then
    for (old, new) in [
        ("(768, 1), 176160768), arg1_1", "(768, 1), 176160769), arg1_1"),
        (".run(arg2_1, arg3_1, buf0, buf4,", ".run(arg2_1, arg3_1, buf4, buf0,"),
        ("buf7 = reinterpret_tensor(buf32, (32768, ), (1, ), 32768)",
          "buf7 = reinterpret_tensor(buf32, (32768, ), (1, ), 0)"),
        ("(768, 1), 0), arg1_1, out=buf0)", "(768, 1), 0), arg2_1, out=buf0)")] do
      let changed := source.replace old new
      if changed == source || (ValidationProjectionBinding.certify changed ptx).isSome then
        throw (IO.userError s!"Validation wrapper accepted changing {old}")
  let mutations := [("0f3E1039F6", "0f3E1039F7"), (", -100;", ", -101;")]
  let geometryMutations := [("_param_10];", "_param_11];"), (", 32768;", ", 32769;"),
    (", 192;", ", 128;"), ("%rd1, 2;", "%rd1, 3;"), ("0fBE088889", "0fBE088888"),
    ("_param_1];", "_param_0];"), ("_param_2];", "_param_3];")]
  for (old, new) in mutations ++ (if shape == "64-4-8" then geometryMutations else []) do
    let changed := ptx.replace old new
    if changed == ptx || (ValidationLogBinding.certify changed).isSome then
      throw (IO.userError s!"Validation checker accepted changing {old} in {shape}")
  let scan := checked.reduction.scan
  if shape == "64-4-8" then
    for (old, new) in [("0f3BBB989D", "0f3BBB989E"), (", 50300;", ", 50296;")] do
      let changed := ptx.replace old new
      if changed == ptx || (ValidationExpBinding.certify changed tile).isSome then
        throw (IO.userError s!"Validation scan accepted changing {old} in {shape}")
    let code := (QKVTTIR.normalizePTX ptx).toArray
    let some update := scan.reduction.updates.head?
      | throw (IO.userError "Validation scan returned no online updates")
    let ["fma.rn.f32", sum, prior, renorm, contribution] :=
      TrainingCEGatherBinding.operands code[update.newSumPC]!
      | throw (IO.userError "Validation scan lost the first online FMA")
    let changed := code.set! update.newSumPC s!"fma.rn.f32 {sum}, {prior}, {contribution}, {renorm};"
    if (ValidationExpBinding.certify (String.intercalate "\n" changed.toList) tile).isSome then
      throw (IO.userError "Validation scan accepted swapping the two online scales")
    let carryMutation := code.map fun line =>
      match TrainingCEGatherBinding.operands line with
      | ["mov.b32", dst, src] =>
        if dst == update.previous.source && src == update.previous.maximum then
          s!"mov.b32 {dst}, {dst};"
        else line
      | _ => line
    if carryMutation == code ||
        (ValidationExpBinding.certify (String.intercalate "\n" carryMutation.toList) tile).isSome then
      throw (IO.userError "Validation scan accepted carrying the old maximum to the next iteration")
    let some slot := scan.reduction.scaled.head?
      | throw (IO.userError "Validation scan returned no rescaled slots")
    let changed := code.set! slot.resultPC s!"mul.f32 {slot.result}, 0f00000000, {slot.scale.output};"
    if (ValidationExpBinding.certify (String.intercalate "\n" changed.toList) tile).isSome then
      throw (IO.userError "Validation scan accepted replacing a carried sum before the rescale")
    let suffix := (code.toList.zipIdx).drop (scan.reduction.latch + 1)
    let some (shuffle, shufflePC) := suffix.find? (fun (line, _) => line.startsWith "shfl.sync.bfly.b32 ")
      | throw (IO.userError "Validation reduction has no butterfly shuffle")
    let ["shfl.sync.bfly.b32", dst, data, "2", "31", "-1"] := TrainingCEGatherBinding.operands shuffle
      | throw (IO.userError "Validation reduction first shuffle is no longer XOR 2")
    let changed := code.set! shufflePC s!"shfl.sync.bfly.b32 {dst}, {data}, 1, 31, -1;"
    if (ValidationReduceBinding.certify (String.intercalate "\n" changed.toList) tile).isSome then
      throw (IO.userError "Validation reduction accepted changing a butterfly offset")
    let some (_, barrier) := (suffix.drop 1).find? (fun (line, pc) => shufflePC < pc &&
      line == "bar.sync 0;") | throw (IO.userError "Validation reduction has no sum publication barrier")
    let changed := code.set! barrier "$no_publication_barrier:"
    if (ValidationReduceBinding.certify (String.intercalate "\n" changed.toList) tile).isSome then
      throw (IO.userError "Validation reduction accepted leaving shared writes unpublished")
    let normalized := String.intercalate "\n" code.toList
    for (old, new) in [
        ("[ %rd83 + 0 ]", "[ %rd83 + 1 ]"),
        ("and.b32 %r156, %r154, 4;", "and.b32 %r156, %r154, 0;"),
        ("selp.f32 %r166, %r165, 0fC2700000, %p27;", "selp.f32 %r166, 0fC2700000, %r165, %p27;"),
        ("@%p26 bra $L__BB0_3;", "@!%p26 bra $L__BB0_3;")] do
      let changed := normalized.replace old new
      if changed == normalized then
        throw (IO.userError s!"Validation scan mutation missing {old}")
      let some core := ValidationExpBinding.certify changed tile
        | throw (IO.userError s!"Mask mutation changed the online update structure at {old}")
      if (ValidationScanBinding.certifyFrom core).isSome then
        throw (IO.userError s!"Validation scan accepted changing {old}")
  if shape == "16-32-4" then
    let code := (QKVTTIR.normalizePTX ptx).toArray
    let some (write, pc) := (code.toList.zipIdx).find?
      (fun (line, _) => line.startsWith "stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 ")
      | throw (IO.userError "Validation scan has no BF16 transpose store")
    let [op, ptr, a, b] := TrainingCEGatherBinding.operands write
      | throw (IO.userError "Validation scan does not expose the two halves of the transpose")
    let changed := code.set! pc (op ++ " " ++ ptr ++ ", {" ++ b ++ ", " ++ a ++ "};")
    if (ValidationScanBinding.certify (String.intercalate "\n" changed.toList) tile).isSome then
      throw (IO.userError "Validation scan accepted exchanging two projection registers at stmatrix")
  return checked

def checkValidationCEArtifacts (artifacts : System.FilePath) : IO ValidationProjectionBinding.Programs := do
  let source ← IO.FS.readFile (artifacts / "validation-ce.py")
  let read := checkValidationCEArtifact artifacts source
  let wideVocabularyTask ← IO.asTask (read .x1r1024)
  let shortVocabularyTask ← IO.asTask (read .x1r512)
  let sixteenTask ← IO.asTask (read .x16r32)
  let sixtyFourTask ← IO.asTask (read .x64r64)
  let sixtyFourShortTask ← IO.asTask (read .x64r4)
  let eightTask ← IO.asTask (read .x8r512)
  let eightShortTask ← IO.asTask (read .x8r256)
  let wideVocabulary ← IO.ofExcept wideVocabularyTask.get
  let shortVocabulary ← IO.ofExcept shortVocabularyTask.get
  let sixteen ← IO.ofExcept sixteenTask.get
  let sixtyFour ← IO.ofExcept sixtyFourTask.get
  let sixtyFourShort ← IO.ofExcept sixtyFourShortTask.get
  let eight ← IO.ofExcept eightTask.get
  let eightShort ← IO.ofExcept eightShortTask.get
  IO.println "Certified the wrapper's BF16 head views and FP32 output aliases, vocabulary projection and mask lookup, tail, online updates, cross-warp reduction, target projection and mask, logarithm and FP32 loss stores for the seven validation PTX variants."
  return fun
    | .x1r1024 => wideVocabulary
    | .x1r512 => shortVocabulary
    | .x16r32 => sixteen
    | .x64r64 => sixtyFour
    | .x64r4 => sixtyFourShort
    | .x8r512 => eight
    | .x8r256 => eightShort

def checkMuddCoefficientsArtifact (ir : System.FilePath) : IO MuddCoefficients.Binding.Program := do
  let ptx ← IO.FS.readFile (ir / "mudd-coefficients.ptx")
  let some program := MuddCoefficients.Binding.Program.check ptx
    | throw (IO.userError "MUDD coefficient PTX did not match")
  for (old, replacement) in [("-1431655765", "-1431655764"), (", 108,", ", 107,"), ("0f3DCCCCCD", "0f3DCCCCCC"),
      ("%rd9, %rd8, %p1", "%rd8, %rd9, %p1"), ("%rs6, %rs7", "%rs7, %rs6")] do
    let mutant := ptx.replace old replacement
    if mutant == ptx || (MuddCoefficients.Binding.Program.check mutant).isSome then
      throw (IO.userError s!"MUDD coefficients accepted a modification at {old}")
  IO.println "Certified the MUDD coefficient arithmetic, packed BF16 stores and ten output pointers."
  pure program

def checkMuddNormArtifact (ir : System.FilePath) : IO MuddNorm.Binding.Program := do
  let ptx ← IO.FS.readFile (ir / "mudd-norm.ptx")
  let some program := MuddNorm.Binding.Program.check ptx
    | throw (IO.userError "MUDD norm PTX did not match")
  for (old, replacement) in [(".reqntid 32", ".reqntid 64"),
      ("%r326, 2, 3;", "%r326, 1, 3;"), ("%rs90, %rs91", "%rs91, %rs90"),
      ("0f3DCCCCCD", "0f3DCCCCCC"),
      ("0f34000000", "0f34000001"), ("rsqrt.approx.ftz.f32", "sqrt.approx.ftz.f32"),
      ("0f408F5C29", "0f408F5C28"), ("mudd_final_norm_param_5];", "mudd_final_norm_param_7];")] do
    let mutant := ptx.replace old replacement
    if mutant == ptx || (MuddNorm.Binding.Program.check mutant).isSome then
      throw (IO.userError s!"MUDD norm accepted a modification at {old}")
  IO.println "Certified MUDD's BF16 coefficient bank, FP32 mixed residual and square sum, and packed E4M3 stores."
  pure program

def checkMuddBackwardArtifact (ir : System.FilePath) : IO MuddBackward.Binding.Pair := do
  let ptx ← IO.FS.readFile (ir / "mudd-norm-cached.ptx")
  let some forward := MuddNorm.Binding.CachedProgram.check ptx
    | throw (IO.userError "MUDD cached forward PTX did not match")
  for (old, replacement) in [("%rd95 + 0 ], { %r221 }", "%rd95 + 0 ], { %r219 }"),
      ("%rs90, %rs91", "%rs91, %rs90"), ("mudd_final_norm_param_4];", "mudd_final_norm_param_3];")] do
    let mutant := ptx.replace old replacement
    if mutant == ptx || (MuddNorm.Binding.CachedProgram.check mutant).isSome then
      throw (IO.userError s!"MUDD cached forward accepted a modification at {old}")
  let check (mode : MuddNorm.BackwardLayout) (file : String) :
      IO (MuddBackward.Binding.Program mode) := do
    let ptx ← IO.FS.readFile (ir / file)
    let some program := MuddBackward.Binding.Program.check mode ptx
      | throw (IO.userError s!"MUDD backward PTX did not match for {file}")
    for (old, replacement) in [(".reqntid 32", ".reqntid 64"),
        ("mudd_backward_param_2];", "mudd_backward_param_1];"),
        ("fma.rn.f32", "mad.f32")] do
      let mutant := ptx.replace old replacement
      if mutant == ptx || (MuddBackward.Binding.Program.check mode mutant).isSome then
        throw (IO.userError s!"MUDD backward {file} accepted a modification at {old}")
    pure program
  let gradient ← check .gradientOnly "mudd-backward-gradient.ptx"
  let encode ← check .withQuantization "mudd-backward-encode.ptx"
  if (MuddBackward.Binding.Program.check .gradientOnly encode.ptx).isSome ||
      (MuddBackward.Binding.Program.check .withQuantization gradient.ptx).isSome then
    throw (IO.userError "MUDD backward accepted the other layout's PTX")
  let ptx ← IO.FS.readFile (ir / "mudd-backward-retained.ptx")
  let some retained := MuddBackward.Binding.RetainedProgram.check ptx
    | throw (IO.userError "MUDD retained activation backward PTX did not match")
  let mutant := ptx.replace "mudd_backward_param_2];" "mudd_backward_param_1];"
  if mutant == ptx || (MuddBackward.Binding.RetainedProgram.check mutant).isSome ||
      (MuddBackward.Binding.RetainedProgram.check encode.ptx).isSome then
    throw (IO.userError "MUDD retained activation backward accepted a different launch")
  IO.println "Certified the cached inverse RMS store and three RMSNorm backward specializations."
  pure ⟨forward, gradient, encode, some retained⟩

def checkMuddActivationArtifacts (ir : System.FilePath) : IO (List MuddActivationStorage.Binding.Program) := do
  let dir := ir / "full-mudd"
  if !(← dir.pathExists) then return []
  let files ← dir.readDir
  let mut programs := []
  for entry in files do
    let filename := entry.fileName
    if filename.startsWith "whole-mudd-16384-10240-3-original-" && filename.endsWith ".ttir" then
      let reference ← IO.FS.readFile entry.path
      let candidate ← IO.FS.readFile (dir / filename.replace "-original-" "-single-")
      let some program := MuddActivationStorage.Binding.Program.check reference candidate
        | throw (IO.userError s!"MUDD whole-graph TTIR check failed for {filename}")
      let mutate := candidate.replace "arith.addf" "arith.subf"
      if mutate == candidate ||
          (MuddActivationStorage.Binding.Program.check reference mutate).isSome ||
          (MuddActivationStorage.Binding.Program.check reference reference).isSome then
        throw (IO.userError s!"MUDD whole-graph TTIR accepted an invalid edit for {filename}")
      programs := program :: programs
  if programs.isEmpty then throw (IO.userError "Missing whole-graph MUDD TTIR captures")
  IO.println s!"Certified the duplicate E4M3 store elimination in {programs.length} reduction schedules."
  pure programs.reverse

private def muddCaptures : List (Nat × Nat × Nat) :=
  [(16384, 10240, 3), (32768, 10240, 2), (49152, 14336, 1), (49152, 24576, 1), (40960, 0, 1), (16384, 0, 1)]

def checkMuddResidualArtifacts (ir : System.FilePath) : IO (List MuddResidualStorage.Binding.Program) := do
  let dir := ir / "full-mudd" / "residual"
  if !(← dir.pathExists) then return []
  let files ← dir.readDir
  let mut programs := []
  for (rows, classes, predictions) in muddCaptures do
    let capture := s!"{rows}-{classes}-{predictions}"
    let mut blocks := []
    for entry in files do
      let filename := entry.fileName
      if filename.startsWith s!"whole-mudd-{capture}-original-" && filename.endsWith ".ttir" then
        let ["whole", "mudd", _, _, _, "original", xText, rText, _] := filename.splitOn "-"
          | throw (IO.userError s!"Malformed MUDD reduction schedule in {filename}")
        let some x := (xText.drop 1).toString.toNat? | throw (IO.userError s!"Malformed MUDD row tile in {filename}")
        let some r := (rText.drop 1).toString.toNat? | throw (IO.userError s!"Malformed MUDD channel tile in {filename}")
        if x == 0 || r == 0 || rows % x != 0 then throw (IO.userError s!"Invalid MUDD schedule in {filename}")
        let reference ← IO.FS.readFile entry.path
        let candidate ← IO.FS.readFile (dir / filename.replace "-original-" "-register-")
        let some program := MuddResidualStorage.Binding.Program.check rows r reference candidate
          | throw (IO.userError s!"MUDD residual TTIR check failed for {filename}")
        let changed := candidate.replace "arith.addf" "arith.subf"
        if changed == candidate || (MuddResidualStorage.Binding.Program.check rows r reference changed).isSome ||
            (768 ≤ r && (MuddResidualStorage.Binding.Program.check rows r reference reference).isSome) then
          throw (IO.userError s!"MUDD residual TTIR accepted an invalid edit for {filename}")
        programs := program :: programs
        blocks := r :: blocks
    if blocks.length != 6 || !(blocks.any (768 ≤ ·)) || !(blocks.any (· < 768)) then
      throw (IO.userError s!"Incomplete MUDD residual schedules for {capture}")
  IO.println s!"Certified BF16 residual register reuse and fallback in {programs.length} reduction schedules."
  pure programs.reverse

def checkMuddGradientArtifacts (ir : System.FilePath) : IO (List MuddGradientGrouping.Binding.Program) := do
  let dir := ir / "full-mudd" / "gradients"
  let mut programs := []
  for (rows, classes, predictions) in muddCaptures do
    let capture := s!"whole-mudd-{rows}-{classes}-{predictions}"
    let sampled := classes != 0
    if sampled then
      let before ← IO.FS.readFile (dir / s!"{capture}-before.py")
      let launches ← IO.FS.readFile (dir / s!"{capture}-launches.py")
      if !MuddGradientPacking.checksWrapper rows before launches then
        throw (IO.userError s!"MUDD packing launch reorder check failed for {capture}")
      let changed := launches.replace "if not _mudd_column_reduction_0: " "if _mudd_column_reduction_0: "
      if changed == launches || MuddGradientPacking.checksWrapper rows before changed then
        throw (IO.userError s!"MUDD packing launch reorder accepted an invalid edit for {capture}")
    let groups := if sampled then [(9, 1), (6, 4), (3, 1)] else [(9, 1)]
    for (planes, first) in groups do
      let hoisted := sampled && planes == 9
      for (tile, warps) in [(1, 2), (8, 2), (32, 4), (128, 8)] do
        let label := s!"{capture}-p{planes}-o{first}"
        let suffix := s!"x{tile}-r16-w{warps}"
        let original ← IO.FS.readFile (dir / s!"{label}-original-{suffix}.ttir")
        let candidate ← IO.FS.readFile (dir / s!"{label}-grouped-{suffix}.ttir")
        let layout ← IO.FS.readFile (dir / s!"{label}-original-{suffix}.ttgir")
        let grouped ← IO.FS.readFile (dir / s!"{label}-grouped-{suffix}.ttgir")
        let some program := MuddGradientGrouping.Binding.Program.check rows planes first tile warps
            original candidate layout grouped hoisted
          | throw (IO.userError s!"MUDD gradient IR check failed for {label}/{suffix}")
        for mutant in [grouped.replace "arith.addf" "arith.subf", grouped.replace "%IN1, %IN0" "%IN0, %IN1"] do
          if mutant == grouped || (MuddGradientGrouping.Binding.Program.check rows planes first tile warps
              original candidate layout mutant hoisted).isSome then
            throw (IO.userError s!"MUDD gradient IR accepted an invalid edit for {label}/{suffix}")
        programs := program :: programs
  IO.println s!"Certified the grouped MUDD backward launches in {programs.length} reduction schedules."
  IO.println s!"Checked the packing reorder in {(programs.filter (·.hoistedPacking)).length} sampled schedules."
  pure programs.reverse

def checkMuddPackingArtifacts (ir : System.FilePath) : IO (List MuddGradientPacking.Binding.Program) := do
  let dir := ir / "full-mudd" / "gradients"
  let mut programs := []
  for (rows, classes, predictions) in muddCaptures do
    if classes == 0 then continue
    let label := s!"whole-mudd-{rows}-{classes}-{predictions}"
    let launches ← IO.FS.readFile (dir / s!"{label}-launches.py")
    let after ← IO.FS.readFile (dir / s!"{label}-after.py")
    unless MuddGradientPacking.Binding.checksLoadEdit launches after do
      throw (IO.userError s!"MUDD packing input rewrite check failed for {label}")
    for (tile, warps) in [(1024, 4), (512, 8)] do
      let suffix := s!"x{tile}-r0-w{warps}"
      let original ← IO.FS.readFile (dir / s!"{label}-packing-original-{suffix}.ttir")
      let candidate ← IO.FS.readFile (dir / s!"{label}-packing-routed-{suffix}.ttir")
      let layout ← IO.FS.readFile (dir / s!"{label}-packing-original-{suffix}.ttgir")
      let routed ← IO.FS.readFile (dir / s!"{label}-packing-routed-{suffix}.ttgir")
      let some program := MuddGradientPacking.Binding.Program.check rows tile warps original candidate layout routed
        | throw (IO.userError s!"MUDD packing input IR check failed for {label}/{suffix}")
      for mutant in [routed.replace "arith.addf" "arith.subf",
          routed.replace "tt.splat %in_ptr9 : " "tt.splat %in_ptr8 : "] do
        if mutant == routed || (MuddGradientPacking.Binding.Program.check rows tile warps
            original candidate layout mutant).isSome then
          throw (IO.userError s!"MUDD packing input IR accepted an invalid edit for {label}/{suffix}")
      programs := program :: programs
  IO.println s!"Certified one selected BF16 load and the ordered packing arithmetic in {programs.length} schedules."
  pure programs.reverse

def checkMuddFusedPackingArtifacts (ir : System.FilePath) : IO (List MuddFusedPacking.Binding.Program) := do
  let dir := ir / "full-mudd" / "gradients"
  let mut programs := []
  for (rows, classes, predictions) in muddCaptures do
    if classes == 0 then continue
    let label := s!"whole-mudd-{rows}-{classes}-{predictions}"
    let before ← IO.FS.readFile (dir / s!"{label}-before.py")
    let grouped ← IO.FS.readFile (dir / s!"{label}-launches.py")
    let launches ← IO.FS.readFile (dir / s!"{label}-fused-launches.py")
    let after ← IO.FS.readFile (dir / s!"{label}-fused-after.py")
    unless MuddGradientPacking.checksFusedWrapper rows before grouped launches &&
        MuddGradientPacking.Binding.checksLoadEdit launches after do
      throw (IO.userError s!"MUDD fusion wrapper check failed for {label}")
    let wrongGuard := launches.replace "if not _mudd_fused_packing_0: " "if _mudd_fused_packing_0: "
    if wrongGuard == launches || MuddGradientPacking.checksFusedWrapper rows before grouped wrongGuard then
      throw (IO.userError s!"MUDD fusion wrapper accepted an inverted guard for {label}")
    for (planes, first) in [(9, 1), (1, 0)] do
      for (tile, warps) in [(1, 2), (8, 2), (32, 4), (128, 8)] do
        let name := s!"{label}-p{planes}-o{first}"
        let suffix := s!"x{tile}-r16-w{warps}"
        let artifact := fun variant ext => IO.FS.readFile (dir / s!"{name}-{variant}-{suffix}.{ext}")
        let reference ← artifact "original" "ttir"
        let candidate ← artifact "fused" "ttir"
        let referenceLayout ← artifact "original" "ttgir"
        let candidateLayout ← artifact "fused" "ttgir"
        let some program := MuddFusedPacking.Binding.Program.check rows planes first tile warps
            reference candidate referenceLayout candidateLayout
          | throw (IO.userError s!"MUDD fusion IR check failed for {name}/{suffix}")
        for mutant in [candidateLayout.replace "1.000000e-01" "2.000000e-01",
            candidateLayout.replace "arith.cmpi slt," "arith.cmpi sle,",
            candidateLayout.replace "%packed = arith.addf" "%packed = arith.subf",
            candidateLayout.replace "arith.addf" "arith.subf"] do
          if mutant == candidateLayout || (MuddFusedPacking.Binding.Program.check rows planes first tile warps
              reference candidate referenceLayout mutant).isSome then
            throw (IO.userError s!"MUDD fusion IR accepted an invalid edit for {name}/{suffix}")
        programs := program :: programs
  IO.println s!"Certified both outputs of MUDD fusion in {programs.length} independently selected schedules."
  pure programs.reverse

def checkMuddSinglePackingArtifacts (ir : System.FilePath) : IO (List MuddFusedPacking.Binding.SingleProgram) := do
  let dir := ir / "full-mudd" / "gradients"
  let mut programs := []
  for (rows, classes, predictions) in muddCaptures do
    if classes == 0 then continue
    let label := s!"whole-mudd-{rows}-{classes}-{predictions}"
    for (tile, warps) in [(1, 2), (8, 2), (32, 4), (128, 8)] do
      let artifact := fun name ext => IO.FS.readFile (dir / s!"{label}-{name}-x{tile}-r16-w{warps}.{ext}")
      let read (ext : String) : IO MuddFusedPacking.Binding.SingleIR := do
        pure ⟨← artifact "p9-o1-original" ext, ← artifact "p1-o0-original" ext, ← artifact "p10-o0-fused" ext⟩
      let ttir ← read "ttir"
      let ttgir ← read "ttgir"
      let some program := MuddFusedPacking.Binding.SingleProgram.check rows tile warps ttir ttgir
        | throw (IO.userError s!"Single-launch MUDD IR check failed for {label}/{tile}/{warps}")
      for altered in [ttgir.fused.replace "tt.get_program_id y : i32" "tt.get_program_id x : i32",
          ttgir.fused.replace "%packed = arith.addf" "%packed = arith.subf",
          ttgir.fused.replace "1.000000e-01" "2.000000e-01"] do
        if altered == ttgir.fused || (MuddFusedPacking.Binding.SingleProgram.check rows tile warps ttir
            {ttgir with fused := altered}).isSome then
          throw (IO.userError s!"Single-launch MUDD IR check accepted an invalid edit for {label}/{tile}/{warps}")
      programs := program :: programs
  IO.println s!"Certified the ten-column MUDD launch in {programs.length} matching schedules."
  pure programs.reverse

private def checkDirectPacking (scaled : Bool) (dir : System.FilePath) (label : String) (rows : Nat) :
    IO (List (MuddFusedPacking.Binding.DirectProgram scaled)) := do
  let read (stem ext : String) := IO.FS.readFile (dir / s!"{stem}.{ext}")
  let variant := if scaled then "scaled" else "direct"
  let mut programs := []
  for (tile, warps) in [(1, 2), (8, 2), (32, 4), (128, 8)] do
    let suffix := s!"x{tile}-r16-w{warps}"
    let groupStem := s!"{label}-p9-o1-original-{suffix}"
    let zeroStem := s!"{label}-p1-o0-original-{suffix}"
    let group ← read groupStem "ttgir"
    let zero ← read zeroStem "ttgir"
    let groupPTX ← read groupStem "ptx"
    let zeroPTX ← read zeroStem "ptx"
    let physical := MuddFusedPacking.Binding.directTile tile
    let columns := if !scaled && tile == 128 then [(10, 0)] else [(1, 0), (9, 1), (10, 0)]
    for (planes, first) in columns do
      let stem := s!"{label}-p{planes}-o{first}-{variant}-x{physical}-r16-w{warps}"
      let fused ← read stem "ttgir"
      let code ← read stem "ptx"
      let ptx : MuddFusedPacking.Binding.DirectPTX := ⟨groupPTX, zeroPTX, code⟩
      let check := MuddFusedPacking.Binding.DirectProgram.check scaled rows planes first tile warps group zero
      let some verified := check fused ptx
        | throw (IO.userError s!"Wrong direct gradient stores for {stem}")
      let changedPort := fused.replace "tt.get_program_id y : i32" "tt.get_program_id x : i32"
      let badArithmetic := if scaled then fused.replace "arith.mulf" "arith.addf"
        else fused.replace "math.fma" "arith.addf"
      let routingMutants := if planes == 10 then [fused.replace "arith.andi " "arith.addi "] else []
      let irMutants := [changedPort, badArithmetic] ++ routingMutants ++
        if !scaled && tile == 128 then [fused.replace "array<i32: 0, 2, 1>" "array<i32: 0, 1, 2>"]
        else []
      for mutant in irMutants do
        if mutant == fused || (check mutant ptx).isSome then
          throw (IO.userError s!"Gradient kernel accepted changed TTGIR for {stem}")
      let routeMasks := if planes == 10 then [code.replace ", 8;" ", 16;"] else []
      let mutants := [code.replace "fma.rn.f32" "fma.rn.ftz.f32"] ++ routeMasks ++
        if tile == 1 then [code.replace ", 8, 31, -1;" ", 16, 31, -1;"]
        else if !scaled && tile == 128 then [code.replace ", 1, 31, -1;" ", 4, 31, -1;",
          code.replace ", 112;" ", 120;", code.replace "setp.lt.u32" "setp.le.u32"] else []
      for mutant in mutants do
        if mutant == code || (check fused {ptx with fused := mutant}).isSome then
          throw (IO.userError s!"Gradient kernel accepted changed PTX for {stem}")
      programs := verified :: programs
  pure programs.reverse

def checkMuddDirectPackingArtifacts (ir : System.FilePath) :
    IO (List (MuddFusedPacking.Binding.DirectProgram false)) := do
  let dir := ir / "full-mudd" / "gradients"
  let captures := muddCaptures.filter (fun (_, classes, _) => classes != 0)
  let programs ← captures.mapM fun (rows, classes, predictions) =>
    checkDirectPacking false dir s!"whole-mudd-{rows}-{classes}-{predictions}" rows
  let programs := programs.flatten
  IO.println s!"Certified direct BF16 writes and the FMA packer in {programs.length} sampled launch variants."
  pure programs

def checkMuddFullPackingArtifacts (ir : System.FilePath) : IO (List (MuddFusedPacking.Binding.DirectProgram true)) := do
  let dir := ir / "full-mudd" / "gradients"
  let mut programs := []
  for rows in [16384, 40960] do
    let label := s!"whole-mudd-{rows}-0-1"
    let before ← IO.FS.readFile (dir / s!"{label}-before.py")
    let after ← IO.FS.readFile (dir / s!"{label}-scaled-after.py")
    unless (MuddFullPacking.verifyFullWrapper rows before after).isSome do
      throw (IO.userError s!"Wrong full-vocabulary projection wrapper for {rows}")
    for mutation in [after.replace "if not _mudd_fused_packing_0: " "if _mudd_fused_packing_0: ",
        after.replace "(10, 64), (64, 1), 896), out=" "(10, 64), (64, 1), 832), out="] do
      if mutation == after || (MuddFullPacking.verifyFullWrapper rows before mutation).isSome then
        throw (IO.userError s!"Full-vocabulary wrapper accepted an invalid edit for {rows}")
    for (tile, warps) in [(512, 8), (1024, 4)] do
      let suffix := s!"x{tile}-r0-w{warps}"
      let scaleSource ← IO.FS.readFile (dir / s!"{label}-projection-scaling-original-{suffix}.ttir")
      let scaleLayout ← IO.FS.readFile (dir / s!"{label}-projection-scaling-original-{suffix}.ttgir")
      unless MuddFullPacking.verifiesProjectionScaling rows tile warps scaleSource scaleLayout do
        throw (IO.userError s!"Wrong reference projection scale for {rows}/{tile}")
      let pack ← IO.FS.readFile (dir / s!"{label}-plane-packing-original-{suffix}.ttir")
      let packLayout ← IO.FS.readFile (dir / s!"{label}-plane-packing-original-{suffix}.ttgir")
      let routed ← IO.FS.readFile (dir / s!"whole-mudd-16384-10240-3-packing-routed-{suffix}.ttir")
      let routedLayout ← IO.FS.readFile (dir / s!"whole-mudd-16384-10240-3-packing-routed-{suffix}.ttgir")
      unless (MuddGradientPacking.Binding.Program.check rows tile warps pack routed packLayout routedLayout).isSome do
        throw (IO.userError s!"Wrong reference packing arithmetic for {rows}/{tile}")
    programs := programs ++ (← checkDirectPacking true dir label rows)
  IO.println s!"Checked the original projection scale and packed reduction and certified all three stores in {programs.length} full-vocabulary launch variants."
  pure programs

def checkMuddMixedPackingArtifacts (scaled : Bool) (ir : System.FilePath) :
    IO (List (MuddFusedPacking.Binding.MixedProgram scaled)) := do
  let dir := ir / "full-mudd" / "gradients"
  let mut programs := []
  for (rows, classes, predictions) in muddCaptures do
    if (classes == 0) != scaled then continue
    let label := s!"whole-mudd-{rows}-{classes}-{predictions}"
    let tiles := [1, 8, 32, 128]
    let schedules := tiles.flatMap fun group => (tiles.filter (· != group)).map (group, ·)
    for (groupTile, zeroTile) in schedules do
      let group ← IO.FS.readFile
        (dir / s!"{label}-p9-o1-original-x{groupTile}-r16-w{MuddFusedPacking.Binding.scheduleWarps groupTile}.ttgir")
      let zero ← IO.FS.readFile
        (dir / s!"{label}-p1-o0-original-x{zeroTile}-r16-w{MuddFusedPacking.Binding.scheduleWarps zeroTile}.ttgir")
      let tile := MuddFusedPacking.Binding.mixedBlock groupTile zeroTile
      let warps := MuddFusedPacking.Binding.mixedWarps groupTile zeroTile
      let variant := s!"{label}-mixed-g{groupTile}-z{zeroTile}-x{tile}-w{warps}"
      let fused ← IO.FS.readFile (dir / s!"{variant}.ttgir")
      let ptx ← IO.FS.readFile (dir / s!"{variant}.ptx")
      let check := MuddFusedPacking.Binding.MixedProgram.check scaled rows groupTile zeroTile group zero
      let some verified := check fused ptx
        | throw (IO.userError s!"Invalid mixed MUDD schedule for {variant}")
      let wrongRoute := fused.replace "arith.andi " "arith.addi "
      if wrongRoute == fused || (check wrongRoute ptx).isSome then
        throw (IO.userError s!"Invalid mixed kernel route check for {variant}")
      let dispatchMutations := if groupTile == 128 && zeroTile == 1 then
          [ptx.replace "setp.gt.u32" "setp.ge.u32", ptx.replace ", -7;" ", -6;"]
        else [ptx.replace "setp.ne.b32" "setp.eq.b32"]
      let mutations := dispatchMutations ++ [ptx.replace ", 8;" ", 16;",
        ptx.replace ", 1, 31, -1;" ", 2, 31, -1;", ptx.replace "0f3DCCCCCD" "0f3DCCCCCF",
        ptx.replace "st.global.b16" "st.global.b32"] ++
        if scaled then [ptx.replace "mul.f32" "add.f32"] else []
      for mutated in mutations do
        if mutated == ptx || (check fused mutated).isSome then
          throw (IO.userError s!"Invalid mixed kernel mutation check for {variant}")
      programs := verified :: programs
  IO.println s!"Certified {programs.length} mixed MUDD schedules using the two original ordered reduction trees."
  pure programs.reverse

private def readAnvilCode (ir : System.FilePath) (stem : String) : IO AnvilGram.Code := do
  pure ⟨← IO.FS.readFile (ir / (stem ++ ".ttgir")),
    ← IO.FS.readFile (ir / (stem ++ ".ptx"))⟩

private def checkAnvilRails (ir : System.FilePath) (bank : AnvilEpilogue.Bank) (stem : String) :
    IO (AnvilRails.Program bank) := do
  let code ← readAnvilCode ir stem
  let some program := AnvilRails.Program.check bank code.ttgir code.ptx
    | throw (IO.userError s!"Unrecognized ANVIL rail artifacts for {stem}")
  for (old, changed) in [("0f3CA3D70A", "0f3CA3D70B"), ("cvt.rn.bf16x2", "cvt.rz.bf16x2"),
      ("st.global.v4.b32", "st.global.v2.b32"), ("1016;", "1008;")] do
    let ptx := code.ptx.replace old changed
    if ptx == code.ptx || (AnvilRails.Program.check bank code.ttgir ptx).isSome then
      throw (IO.userError s!"ANVIL rails accepted changing {old} in {stem}")
  return program

private def checkAnvilTrace (ir : System.FilePath) (bank : AnvilEpilogue.Bank) (stem : String) :
    IO (AnvilTrace.Program bank) := do
  let code ← readAnvilCode ir stem
  let some program := AnvilTrace.Program.check bank code.ttgir code.ptx
    | throw (IO.userError s!"Unrecognized ANVIL trace artifacts for {stem}")
  for (old, changed) in [("0f3F866666", "0f3F866667"), ("sqrt.rn.f32", "sqrt.approx.f32"),
      ("shfl.sync.bfly.b32", "shfl.sync.idx.b32"), ("bar.sync", "bar.warp.sync")] do
    let ptx := code.ptx.replace old changed
    if ptx == code.ptx || (AnvilTrace.Program.check bank code.ttgir ptx).isSome then
      throw (IO.userError s!"ANVIL trace accepted changing {old} in {stem}")
  return program

private def checkAnvilNormalization (ir : System.FilePath) (bank : AnvilEpilogue.Bank)
    (stem : String) : IO (AnvilNormalization.Program bank) := do
  let code ← readAnvilCode ir stem
  let some program := AnvilNormalization.Program.check bank code.ttgir code.ptx
    | throw (IO.userError s!"Unrecognized ANVIL normalization artifacts for {stem}")
  for (old, changed) in [("div.full.f32", "div.approx.f32"), ("cvt.rn.bf16x2", "cvt.rz.bf16x2"),
      ("%ctaid.y", "%ctaid.x"), ("st.global.v4.b32", "st.global.v2.b32")] do
    let ptx := code.ptx.replace old changed
    if ptx == code.ptx || (AnvilNormalization.Program.check bank code.ttgir ptx).isSome then
      throw (IO.userError s!"ANVIL normalization accepted changing {old} in {stem}")
  return program

private def checkAnvilGram (ir : System.FilePath) (shape : AnvilGram.Shape) (stem : String) :
    IO (AnvilGram.Program shape) := do
  let baseline ← readAnvilCode ir (stem ++ "-reference")
  let compact ← readAnvilCode ir (stem ++ "-compact")
  let fullTile ← readAnvilCode ir (stem ++ "-full")
  let artifacts : AnvilGram.Artifacts := ⟨baseline, compact, fullTile⟩
  let some program := AnvilGram.Program.check shape artifacts
    | throw (IO.userError s!"Unrecognized ANVIL tile artifacts for {stem}")
  for (old, changed) in [("wgmma.mma_async", "wgmma.mma_async_changed"),
      ("wgmma.wait_group", "wgmma.commit_group"),
      ("mbarrier.try_wait.parity", "mbarrier.try_wait.changed"),
      ("cp.async.bulk.tensor.2d.shared::cluster.global", "cp.async.bulk.tensor.2d.shared::cta.global"),
      ("fence.proxy.async.shared::cta;", "bar.sync 0;"),
      ("cp.async.bulk.wait_group 0;", "cp.async.bulk.wait_group.read 0;"),
      ("cp.async.bulk.commit_group;", "mov.u32 %r0, 0;")] do
    let ptx := fullTile.ptx.replace old changed
    if ptx == fullTile.ptx ||
        (AnvilGram.Program.check shape { artifacts with fullTile := { fullTile with ptx } }).isSome then
      throw (IO.userError s!"ANVIL tile check accepted changing {old} in {stem}")
  for (old, changed) in [("ttng.wait_barrier", "ttng.inval_barrier"),
      ("%offdiag = arith.cmpi ne,", "%offdiag = arith.cmpi eq,"),
      (", 32768, %", ", 65536, %")] do
    let ttgir := fullTile.ttgir.replace old changed
    if ttgir == fullTile.ttgir ||
        (AnvilGram.Program.check shape { artifacts with fullTile := { fullTile with ttgir } }).isSome then
      throw (IO.userError s!"ANVIL TMA checker accepted changing {old} in {stem}")
  return program

private def checkAnvilEpilogue (ir : System.FilePath) (bank : AnvilEpilogue.Bank)
    (tile : AnvilEpilogue.Tile) (stem : String) :
    IO (AnvilEpilogue.Binding.Program bank tile) := do
  let suffix := match tile with
    | .v1024 => "-1024-4"
    | .v512 => "-512-8"
  let equalize ← readAnvilCode ir (stem ++ "-eq" ++ suffix)
  let decay ← readAnvilCode ir (stem ++ "-decay" ++ suffix)
  let fused ← readAnvilCode ir (stem ++ "-fused" ++ suffix)
  let artifacts : AnvilEpilogue.Binding.Artifacts := ⟨equalize, decay, fused⟩
  let some program := AnvilEpilogue.Binding.Program.check bank tile artifacts
    | throw (IO.userError s!"Unrecognized ANVIL epilogue artifacts for {stem}{suffix}")
  for (old, changed) in [("cvt.rn.bf16", "cvt.rz.bf16"), ("fma.rn.f32", "mad.rn.f32")] do
    let ptx := fused.ptx.replace old changed
    if ptx == fused.ptx ||
        (AnvilEpilogue.Binding.Program.check bank tile { artifacts with fused := { fused with ptx } }).isSome then
      throw (IO.userError s!"ANVIL epilogue accepted changing {old} in {stem}{suffix}")
  return program

private def checkAnvilCooperative (ir : System.FilePath) (bank : AnvilEpilogue.Bank)
    (stem : String) : IO (AnvilGram.Cooperative.Program bank) := do
  let seed ← readAnvilCode ir (stem ++ "-seed")
  let initial ← readAnvilCode ir (stem ++ "-initial")
  let round ← readAnvilCode ir (stem ++ "-round")
  let some program := AnvilGram.Cooperative.Program.check bank seed initial round
    | throw (IO.userError s!"Unrecognized cooperative ANVIL artifacts for {stem}")
  for (old, replacement) in [
      ("atom.acq_rel.gpu.global.or.b32", "atom.relaxed.gpu.global.or.b32"),
      ("ld.acquire.gpu.global.u32", "ld.relaxed.gpu.global.u32"),
      ("fence.proxy.async.global;", "fence.proxy.async.shared::cta;"),
      ("ld.global.cg.", "ld.global.ca."),
      ("and.b32 covered, bits,", "or.b32 covered, bits,"),
      ("cp.async.bulk.wait_group 0;", "cp.async.bulk.wait_group.read 0;"),
      ("cp.async.bulk.commit_group;", "mov.u32 %r0, 0;")] do
    let ptx := round.ptx.replace old replacement
    if ptx == round.ptx || (AnvilGram.Cooperative.Program.check bank seed initial {round with ptx}).isSome then
      throw (IO.userError s!"Cooperative ANVIL accepted changing {old} in {stem}")
  for (old, replacement) in [
      ("atom.global.gpu.acq_rel.add.u32", "atom.global.gpu.relaxed.add.u32"),
      ("setp.ne.b32", "setp.eq.b32"),
      ("fence.proxy.async.global;", "fence.proxy.async.shared::cta;"),
      ("ld.global.cg.", "ld.global.ca."),
      ("cp.async.bulk.wait_group 0;", "cp.async.bulk.wait_group.read 0;"),
      ("cp.async.bulk.commit_group;", "mov.u32 %r0, 0;")] do
    let ptx := initial.ptx.replace old replacement
    if ptx == initial.ptx || (AnvilGram.Cooperative.Program.check bank seed {initial with ptx} round).isSome then
      throw (IO.userError s!"Initial ANVIL accepted changing {old} in {stem}")
  for (old, replacement) in [("tt.store %", "tt.load %"),
      ("arith.constant 0 : i32", "arith.constant 1 : i32")] do
    let ttgir := seed.ttgir.replace old replacement
    if ttgir == seed.ttgir || (AnvilGram.Cooperative.Program.check bank {seed with ttgir} initial round).isSome then
      throw (IO.userError s!"Cooperative ANVIL accepted changing seed {old} in {stem}")
  return program

private def checkAnvilArtifacts (ir : System.FilePath) : IO ArtifactTraining.OptimizerPrograms := do
  let rails := ir / "anvil" / "rails"
  let trace := ir / "anvil" / "trace"
  let gram := ir / "anvil" / "gram"
  let epilogue := ir / "anvil" / "epilogue"
  let cooperative := ir / "anvil" / "cooperative"
  let qkRound ← checkAnvilCooperative cooperative .qk "qk-gram"
  let voRound ← checkAnvilCooperative cooperative .vo "vo-gram"
  let mlpRound ← checkAnvilCooperative cooperative .mlp "mlp-gram"
  let qkRails ← checkAnvilRails rails .qk "qk-gram-rails"
  let voRails ← checkAnvilRails rails .vo "vo-gram-rails"
  let mlpRails ← checkAnvilRails rails .mlp "mlp-gram-rails"
  let qkTrace ← checkAnvilTrace trace .qk "qk-gram-divisors"
  let voTrace ← checkAnvilTrace trace .vo "vo-gram-divisors"
  let mlpTrace ← checkAnvilTrace trace .mlp "mlp-gram-divisors"
  let qkNormalize ← checkAnvilNormalization trace .qk "qk-gram-normalize"
  let voNormalize ← checkAnvilNormalization trace .vo "vo-gram-normalize"
  let mlpNormalize ← checkAnvilNormalization trace .mlp "mlp-gram-normalize"
  let qkGram ← checkAnvilGram gram .qkGram "qk-gram"
  let voGram ← checkAnvilGram gram .voGram "vo-gram"
  let mlpGram ← checkAnvilGram gram .mlpGram "mlp-gram"
  let qkPoly ← checkAnvilGram gram .qkPolynomial "qk-poly"
  let widePoly ← checkAnvilGram gram .widePolynomial "vo-mlp-poly"
  let qkLarge ← checkAnvilEpilogue epilogue .qk .v1024 "anvil-6-256-768"
  let qkSmall ← checkAnvilEpilogue epilogue .qk .v512 "anvil-6-256-768"
  let voLarge ← checkAnvilEpilogue epilogue .vo .v1024 "anvil-2-768-768"
  let voSmall ← checkAnvilEpilogue epilogue .vo .v512 "anvil-2-768-768"
  let mlpLarge ← checkAnvilEpilogue epilogue .mlp .v1024 "anvil-3-2816-768"
  let mlpSmall ← checkAnvilEpilogue epilogue .mlp .v512 "anvil-3-2816-768"
  IO.println "Certified ANVIL rails, 128-wide Gram and polynomial TMA with specialized diagonal and off-diagonal loops and queued stores, divisor trees, in-place normalization, counter seeds and cooperative rounds, and both rounded weight halves."
  return {
    rails := fun
      | .qk => qkRails
      | .vo => voRails
      | .mlp => mlpRails
    trace := fun
      | .qk => qkTrace
      | .vo => voTrace
      | .mlp => mlpTrace
    normalize := fun
      | .qk => qkNormalize
      | .vo => voNormalize
      | .mlp => mlpNormalize
    gram := fun
      | .qkGram => qkGram
      | .voGram => voGram
      | .mlpGram => mlpGram
      | .qkPolynomial => qkPoly
      | .widePolynomial => widePoly
    cooperative := fun
      | .qk => qkRound
      | .vo => voRound
      | .mlp => mlpRound
    epilogue := fun
      | .qk, .v1024 => qkLarge
      | .qk, .v512 => qkSmall
      | .vo, .v1024 => voLarge
      | .vo, .v512 => voSmall
      | .mlp, .v1024 => mlpLarge
      | .mlp, .v512 => mlpSmall
  }

private def checkTailArtifacts (ir : System.FilePath) : IO ArtifactTraining.TailPrograms := do
  let ⟨lerp, lerpCertificate⟩ ← checkTailLerpArtifacts ir
  let ⟨normIR, normPTX, normCertificate⟩ ← checkTailNormArtifacts ir
  pure { lerp, lerpCertificate, normIR, normPTX, normCertificate }

private def disassembleAdam (tool : String) (binary : System.FilePath) : IO String := do
  let result ← IO.Process.output {cmd := tool, args := #["-sass", "-res-usage", binary.toString]}
  unless result.exitCode == 0 do
    throw (IO.userError s!"Could not disassemble {binary}: {result.stderr}")
  return result.stdout

private def loadCombinedConsumer (reference candidate : System.FilePath) (tool : String)
    (period : ValueAdam.Elementwise.Period) (tile : ValueAdam.Tile) (masked : Bool) :
    IO (ValueAdam.Elementwise.CombinedConsumer period tile masked) := do
  let tick := if period == .two then 2 else 4
  let mode := if masked then "masked" else "dense"
  let stem := s!"consume-{mode}-{tick}-{tile.width}"
  let profile : ValueAdam.Elementwise.Profile := ⟨period, tile, true⟩
  let oldPtX ← IO.FS.readFile (reference / (stem ++ ".ptx"))
  let newPtX ← IO.FS.readFile (candidate / (stem ++ ".ptx"))
  let oldMetadata ← IO.FS.readFile (reference / (stem ++ ".json"))
  let newMetadata ← IO.FS.readFile (candidate / (stem ++ ".json"))
  unless profile.metadataMatches oldMetadata && profile.metadataMatches newMetadata do
    throw (IO.userError s!"Wrong native Adam launch metadata for {stem}")
  let oldSASS ← disassembleAdam tool (reference / (stem ++ ".cubin"))
  let newSASS ← disassembleAdam tool (candidate / (stem ++ ".cubin"))
  let some checked := ValueAdam.NativeCombinedTrace.certify oldPtX newPtX oldSASS newSASS .directAfterRoot tick masked tile.laneWidth
    | throw (IO.userError s!"Rejected native Adam divider for {stem}")
  IO.println s!"Certified gradient scaling, workspace clearing, moments, division and weight writeback for {stem}."
  return ⟨oldPtX, newPtX, oldSASS, newSASS, .directAfterRoot, checked⟩

private def loadCombinedAtTile (reference candidate : System.FilePath) (tool : String)
    (tile : ValueAdam.Tile) :
    IO ((period : ValueAdam.Elementwise.Period) → (masked : Bool) →
        ValueAdam.Elementwise.CombinedConsumer period tile masked) := do
  let twoDense ← loadCombinedConsumer reference candidate tool .two tile false
  let twoMasked ← loadCombinedConsumer reference candidate tool .two tile true
  let fourDense ← loadCombinedConsumer reference candidate tool .four tile false
  let fourMasked ← loadCombinedConsumer reference candidate tool .four tile true
  return fun
    | .two, false => twoDense
    | .two, true => twoMasked
    | .four, false => fourDense
    | .four, true => fourMasked

private def loadCombinedPrograms (reference candidate : System.FilePath) (tool : String) :
    IO ValueAdam.Elementwise.CombinedPrograms := do
  let narrow ← loadCombinedAtTile reference candidate tool .x512
  let wide ← loadCombinedAtTile reference candidate tool .x1024
  return ⟨fun period tile masked => match tile with
    | .x512 => narrow period masked
    | .x1024 => wide period masked⟩

def checkCEExecution (baseline candidate ir validation : System.FilePath) (includeArtifacts : Bool := false)
    (guardedArtifacts : Option (System.FilePath × String) := none) : IO Unit := do
  let baseline ← readImage baseline .reference
  let actual ← readImage candidate .candidate
  let packed := CombinedTraining.packedSource actual
  let candidate := packed.getD actual
  let some linked := certify baseline candidate
    | throw (IO.userError "Track 1 source link check failed")
  let combined ← if includeArtifacts && packed.isSome then do
    let some source := CombinedTraining.Sources.check actual linked.sources
      | throw (IO.userError "Native value Adam source link failed")
    let (compiled, tool) := guardedArtifacts.getD (ir / "value-adam-guarded", "cuobjdump")
    let programs ← loadCombinedPrograms ir compiled tool
    pure (some (source, programs))
  else if guardedArtifacts.isSome then
    throw (IO.userError "The guarded value Adam artifacts require the checked guarded kernel source")
  else pure none
  IO.println s!"Linked {linked.sources.common.length} common runtime files and the candidate's private kernels."
  checkTrainingCESources baseline candidate
  let tail ← if includeArtifacts then some <$> checkTailArtifacts ir else pure none
  let qkvRefresh ← if includeArtifacts then some <$> checkQKVRefreshArtifacts ir else pure none
  let attention ← if includeArtifacts then some <$> checkAttentionExecution ir else pure none
  let yarn ← if includeArtifacts then some <$> checkYarnExecution ir else pure none
  let optimizer ← if includeArtifacts then some <$> checkAnvilArtifacts ir else pure none
  let sparse ← if includeArtifacts then do
    let ⟨packetArtifacts, packets⟩ ← checkSparsePacketArtifacts ir
    let ⟨scratchArtifacts, scratch⟩ ← checkNgramScratchArtifacts ir
    let ⟨adamArtifacts, adam⟩ ← checkNgramAdamArtifacts ir
    let ⟨valueArtifacts, value⟩ ← checkValueAdamArtifacts ir
    let valueNative ← ValueAdam.Elementwise.readPrograms ir
    let valueLocal ← ValueAdam.LocalRows.readPrograms ir
    let ⟨fetchArtifacts, fetch⟩ ← checkSparseLandingArtifacts ir
    let ⟨momentArtifacts, moments⟩ ← checkMomentReplayArtifacts ir
    let ⟨ngramAccumIR, ngramAccumPTX, ngramAccum⟩ ← checkNgramAccumulationArtifact ir
    let ⟨valueAccumArtifacts, valueAccum⟩ ← checkValueEmbeddingArtifacts ir
    let ⟨embeddingArtifacts, embedding⟩ ← checkNgramEmbedArtifacts ir
    let programs : ArtifactTraining.SparsePrograms := {
      packetArtifacts := packetArtifacts
      packets := packets
      scratchArtifacts := scratchArtifacts
      scratch := scratch
      adamArtifacts := adamArtifacts
      adam := adam
      valueArtifacts := valueArtifacts
      value := value
      valueNative := valueNative
      valueLocal := valueLocal
      fetchArtifacts := fetchArtifacts
      fetch := fetch
      momentArtifacts := momentArtifacts
      moments := moments
      ngramAccumIR := ngramAccumIR
      ngramAccumPTX := ngramAccumPTX
      ngramAccum := ngramAccum
      valueAccumArtifacts := valueAccumArtifacts
      valueAccum := valueAccum
      embeddingArtifacts := embeddingArtifacts
      embedding := embedding
    }
    pure (some programs)
  else pure none
  let graphStages ← if includeArtifacts then some <$> checkGraphSlotArtifacts ir else pure none
  if includeArtifacts then checkSampledGatherArtifacts ir
  let sampledGather ← if includeArtifacts then some <$> checkSampledInverseArtifacts ir else pure none
  let activation ← if includeArtifacts then some <$> checkHeadActivationArtifact ir else pure none
  let sampledDensify ← if includeArtifacts then some <$> checkDensifyArtifacts ir else pure none
  let tiedTranspose ← if includeArtifacts then some <$> checkBf16TransposeArtifacts ir else pure none
  let headRefresh ← if includeArtifacts then some <$> checkLmHeadArtifacts ir else pure none
  let mlpWeight ← if includeArtifacts then some <$> checkMlpWeightArtifacts ir else pure none
  let mudd ← if includeArtifacts then some <$> checkMuddCoefficientsArtifact ir else pure none
  let norm ← if includeArtifacts then some <$> checkMuddNormArtifact ir else pure none
  let muddPair ← if includeArtifacts then some <$> checkMuddBackwardArtifact ir else pure none
  let muddStorage ← if includeArtifacts then checkMuddActivationArtifacts ir else pure []
  let muddResidual ← if includeArtifacts then checkMuddResidualArtifacts ir else pure []
  let muddGradients ← if includeArtifacts then checkMuddGradientArtifacts ir else pure []
  let muddPackingIR ← if includeArtifacts then checkMuddPackingArtifacts ir else pure []
  let muddFusedPackingIR ← if includeArtifacts then checkMuddFusedPackingArtifacts ir else pure []
  let muddSinglePackingIR ← if includeArtifacts then checkMuddSinglePackingArtifacts ir else pure []
  let directPacking ← if includeArtifacts then checkMuddDirectPackingArtifacts ir else pure []
  let scaledPacking ← if includeArtifacts then checkMuddFullPackingArtifacts ir else pure []
  let mixedPacking ← if includeArtifacts then checkMuddMixedPackingArtifacts false ir else pure []
  let mixedFullPacking ← if includeArtifacts then checkMuddMixedPackingArtifacts true ir else pure []
  let trainingCE ← checkTrainingCEGatherArtifacts ir
  let transpose ← checkCETransposeArtifacts ir
  let validationCE ← checkValidationCEArtifacts validation
  let kernels : ArtifactTraining.Kernels :=
    { trainingCE, ceTranspose := transpose, validationCE, graphStages, sampledGather, optimizer, qkvRefresh, tail, sparse,
      attention, yarn,
      headActivation := activation, sampledDensify, mlpWeight, headRefresh, tiedTranspose, muddCoefficients := mudd, muddNorm := norm, muddPair,
      muddStorage, muddResidual, muddGradients, muddPackingIR, muddFusedPackingIR, muddSinglePackingIR,
      muddDirectPackingIR := directPacking, muddFullPackingIR := scaledPacking,
      muddMixedPackingIR := mixedPacking, muddMixedFullPackingIR := mixedFullPacking }
  match combined with
  | none =>
    if packed.isSome then
      IO.println "Checked the cross-entropy and transpose artifacts for the guarded value Adam candidate."
    else
      let checked := ArtifactTraining.certify linked.sources kernels
      IO.println s!"Linked {checked.kernels.body.length} compiled programs to the full training and validation driver."
  | some (source, valueAdam) =>
    let programs : CombinedTraining.Programs := ⟨kernels, valueAdam⟩
    let _ := CombinedTraining.combined_training_end_to_end source programs
    IO.println s!"Linked {kernels.body.length} compiled programs to the guarded value Adam training and validation driver."

end VeriTile.Bench.Models.NanoGPT.Track1.Driver

def main (args : List String) : IO Unit :=
  match args with
  | ["--combined-artifact-execution", baseline, candidate, ir, validation, compiled, tool] => do
    VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkCEExecution
      ⟨baseline⟩ ⟨candidate⟩ ⟨ir⟩ ⟨validation⟩ true (some (⟨compiled⟩, tool))
  | ["--artifact-execution", baseline, candidate, ir, validation] =>
    VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkCEExecution ⟨baseline⟩ ⟨candidate⟩ ⟨ir⟩ ⟨validation⟩ true
  | ["--ce-execution", baseline, candidate, ir, validation] =>
    VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkCEExecution ⟨baseline⟩ ⟨candidate⟩ ⟨ir⟩ ⟨validation⟩
  | ["--validation-ce", artifacts] => do
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkValidationCEArtifacts ⟨artifacts⟩
  | [baseline, candidate] =>
    VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkDirectories ⟨baseline⟩ ⟨candidate⟩
  | [baseline, candidate, ir] => do
    VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkDirectories ⟨baseline⟩ ⟨candidate⟩
    VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkTtirDirectory ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkScaleArtifacts ⟨ir⟩
    VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkPackArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkGroupedArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkMlpWeightArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkLmHeadArtifacts ⟨ir⟩
    VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkSampledGatherArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkSampledInverseArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkDensifyArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkAttentionExecution ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkYarnExecution ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkCETransposeArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkTrainingCEGatherArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkBf16TransposeArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkTailLerpArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkTailNormArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkSparsePacketArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkNgramAccumulationArtifact ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkNgramScratchArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkSparseLandingArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkMomentReplayArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkNgramAdamArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkValueAdamArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkValueEmbeddingArtifacts ⟨ir⟩
    let _ ← VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkGraphSlotArtifacts ⟨ir⟩
    discard (VeriTile.Bench.Models.NanoGPT.Track1.Driver.checkNgramEmbedArtifacts ⟨ir⟩)
  | _ => throw (IO.userError
      "Expected BASELINE CANDIDATE [IR], --validation-ce DIRECTORY, --[artifact|ce]-execution BASELINE CANDIDATE IR VALIDATION, or --combined-artifact-execution BASELINE CANDIDATE IR VALIDATION VALUE_ADAM CUOBJDUMP")
