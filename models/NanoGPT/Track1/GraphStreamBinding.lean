import models.NanoGPT.Track1.SourceBinding
import models.NanoGPT.Track1.GraphReplayStream
import models.NanoGPT.Track1.NgramScratch

/-! Source sites that issue the selected graph work on the caller's
current CUDA stream. The package linker checks the surrounding Python
and the dispatch guard before this projection applies. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver

private def slotUpload : String := r#"
        slot.uploaded.synchronize()
        out = slot.packet.upload((inputs, targets, cum_seqlens, ngram_ids), self.device, coalesce)
        slot.uploaded.record()
        return out
"#

private def cachedUpload : String := r#"
        self.ahead[step] = self.loader.send(new_params)
"#

private def launchAndReplay : String := r#"
        destinations = cs.static.buffers()
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
        cs.forward.replay()
"#

private def kernelLaunch : String := r#"
    stage_step_with_lookup[(input_tiles + 2 * tokens // SLOT_BLOCK,)](
        *sources, query.want, query.directory, *destinations, slots,
        input_tiles, ends, mtp, VOCAB_SIZE, TOKEN_BLOCK, SLOT_BLOCK, query.hashed_rows, query.member_rows,
        num_warps=4, num_stages=1,
    )
    torch.autograd.graph.increment_version(destinations)
    return slots
"#

private def captureMembers : String := r#"
    ngram_rows = sorted_unique_rows([b.ngram_ids_cpu for b in batches])
    directory = make_directory(ngram_rows)
    members = () if directory is None else tuple(ref(b.ngram_ids) for b in batches)
    return ngram_rows, value_embed_rows([b.inputs_cpu for b in batches]), directory, members
"#

private def publishMembers : String := r#"
        if slot_directory is not None:
            ngram_pull.slot_directory = packet[directory_start(ngram_want.size):]
            ngram_pull.slot_members = members
        self.next_pull = NextPull(event=i, ngram=ngram_pull, value_embeds=value_embed_pull, cycle=cycle)
"#

private def activateCycle : String := r#"
        pull, cycle = self.next_pull, self.next_pull.cycle
        self.next_pull = None
        self.ngram.complete_land()
        self.ngram.start_cycle(pull.ngram)
        self.live_cycle = cycle
"#

private def selectMembers : String := r#"
        members = pull.slot_members
        member = hashed_rows and 0 <= member_index < len(members) and members[member_index]() is row_ids
        query = prepare_bucket_slots(pull.want, pull.slot_directory, row_ids,
                                     hashed_rows=hashed_rows, member_rows=member)
"#

private def uniqueRows : String := r#"
    rows = np.concatenate(row_id_arrays)
    rows.sort()
    if rows.size:
        keep = np.empty(rows.size, dtype=np.bool_)
        keep[0] = True
        np.not_equal(rows[1:], rows[:-1], out=keep[1:])
        rows = rows[keep]
    return rows.astype(np.int32, copy=False)
"#

private def streamSites : List (String × String) :=
  [(pinnedBatchFile, slotUpload),
   (privateBatchPacket, "\n                out.append(buf[:n].to(device, non_blocking=True))\n"),
   (privateBatchPacket, "\n        packet = prefix.to(device, non_blocking=True)\n        return typed_views(packet, ranges)\n"),
   (dataFile, cachedUpload),
   (stepGraphsFile, launchAndReplay),
   (privateGraphStageKernel, kernelLaunch),
   (rowPrefetchFile, captureMembers),
   (rowPrefetchFile, publishMembers),
   (rowPrefetchFile, activateCycle),
   (ngramTableFile, "\n        assert self.cached is pull and self.grads_in_flight is None\n        assert self.ngram_event_grad is None and not self.ngram_sum_started\n        self.live = pull\n"),
   (ngramTableFile, "\n        work.wait()\n        lo, hi, n = pull.lo, pull.hi, pull.num_rows\n"),
   (rowOwnersFile, "\n        pull.recv_counts = recv.tolist()\n        pull.peer_want = pull.peer_rows()\n"),
   (rowPrefetchFile, "\n        value_embed_pull = self.value_embeds.request(self.staging.value_embed_ring.upload(value_embed_want),\n                                                     value_embed_want)\n"),
   (valuePullFile, "\n        first = self.owners.first_row\n        self.own_rows = slice(first, first + self.owners.rows_per_rank)\n"),
   (valuePullFile, "\n        received, work = self.owners.send_grads(pull, wire)\n        self.grads_in_flight = (own, received, work, wire)\n"),
   (valuePullFile, "\n        ids = pull.peer_want if self.grads_in_flight[0] is None else pull.want\n"),
   (valuePullFile, "\n        work.wait()\n        pull, first = self.live, self.owners.first_row\n        grad = self.grad_shard\n        profile_key, profile = self._current_scale_read(grad)\n        direct_own = own_grad is None\n        if direct_own:\n            own_grad = self.grad_accum[self.own_rows]\n"),
   (rowPrefetchFile, "\n        self.prefetch.value_embeds.send_grads()\n        self.prefetch.ngram.send_grads()\n        self.prefetch.value_embeds.zero_sent_grads()\n"),
   (rowPrefetchFile, "\n        self.prefetch.value_embeds.adam_update()\n        self.prefetch.ngram.adam_update(self.step, self.lr, self.weight_decay, self.eps)\n        self.prefetch.serve_next_cycle()\n"),
   (anvilFile, "\n        for label in self.scatter_order:\n            if label == SPARSE_UPDATE_LABEL and sparse_update is not None:\n                sparse_update.launch()\n"),
   (anvilFile, "\n        for label in self.work_order:\n            if label == SPARSE_UPDATE_LABEL and sparse_update is not None:\n                sparse_update.update()\n"),
   (rowOwnersFile, "\n        received, work = all_to_all_rows(served, pull.recv_counts, pull.send_counts, async_op=True)\n"),
   ("train_gpt.py", "\n    ngram_grad = step_graphs.backward()\n    if row_prefetch.update_positions and row_prefetch.position[step] <= row_prefetch.update_positions[-1]:\n        ngram_table.accumulate_grad(ngram_slots, ngram_grad, row_prefetch.is_update(step))\n"),
   (stepGraphsFile, "\n        cs = self.current\n        cs.backward.replay()\n"),
   (crossEntropyFile, "\n    transpose_ce_operands(x_f8, x_f8_T, grad_input, grad_input_T)\n\n    grad_w = torch._scaled_mm("),
   (crossEntropyFile, "\n        grad_x, grad_w = _ce_backward_gemms(grad_input, w_f8_row, x_f8, x_s, w_s, grad_s)\n"),
   (sampledCEFile, "\n        grad_x, grad_w_c = _ce_backward_gemms(grad_input, rows_t, x_f8, *ctx.scales)\n"),
   (stepGraphsFile, "\n                          value_embed_grad=self.value_embeds.grad_accum).sum()\n"),
   (stepGraphsFile, "\n            grads = torch.autograd.grad(cs.loss, leaves, allow_unused=True)\n"),
   (modelFile, "\nVALUE_EMBED_LAYERS = (1, 2, 8, 10)\n"),
   (modelFile, "\n        ve_planes = value_embed_lookup(self.value_embeds, input_seq, len(VALUE_EMBED_LAYERS), value_embed_grad)\n"),
   (valueEmbedFile, "\n@torch.library.custom_op(\"nanogpt::value_embed_grad_accumulate\", mutates_args={\"grad_accum\"})\n"),
   (valueEmbedFile, "\n        torch.ops.nanogpt.value_embed_grad_accumulate(token_ids, list(plane_grads), grad_accum)\n"),
   (valueEmbedFile, "\n    if accumulate_four_planes_(token_ids, plane_grads, grad_accum, vocab, d):\n        return\n"),
   (stepGraphsFile, "\n        leaves = self.params + [cs.ngram_sink]\n"),
   (sampledFile, "\n        self.staging.wait_uploaded(staged)\n        self._gather_rows(lm_head_f8_col, P, inputs)\n"),
   (sampledFile, "\n        if gather_sampled_rows(weights, cand, rows, transposed, inverse=inputs.vocab_pos):\n            return\n"),
   ("track_1_short/perf/sampled_softmax_overlap.py",
    "\n        self.copy_stream.wait_event(self.readers_done)\n        with torch.cuda.stream(self.copy_stream):\n"),
   ("track_1_short/perf/sampled_softmax_overlap.py",
    "\n        slot.uploaded.record(self.copy_stream)\n\n    def wait_uploaded(self, staged: StagedCandidates):\n        torch.cuda.current_stream().wait_event(staged.slot.uploaded)\n"),
   ("track_1_short/perf/sampled_softmax_overlap.py", "\n        self.readers_done.record()\n"),
   (stepGraphsFile, "\n        cs.ngram_grad = grads[-1]\n"),
   (ngramFile, "\n        return None, None, None, None, torch.ops.nanogpt.ngram_embed_bwd(g, pool, inp)\n"),
   (ngramTableFile, "\n        per_slot = self.grad_per_slot[:rows]\n        scatter_add_rows(sink_grad.contiguous(), slots, per_slot)\n        self.ngram_sum_started = True\n"),
   (ngramTableFile, "\n        slots, grad = self.ngram_event_grad\n        scatter_add_rows(grad, slots, per_slot)\n        self.ngram_event_grad = None\n        self.ngram_sum_started = False\n"),
   (ngramTableFile, "\n        self.last_event.zero_()\n        self.grad_per_slot.zero_()\n        self.adam_events = 0\n"),
   (ngramTableFile, "\n        self.beta2_history = torch.zeros(max_events + 1, dtype=torch.float32, device=device)\n"),
   (ngramTableFile, "\n        beta2, wd_mul = adam_beta2_and_wd_mul(step)\n        lr = lr * NGRAM_LR_MUL\n        self.adam_events += 1\n        event = self.adam_events\n"),
   (ngramTableFile, "\n        self.beta2_history[event].fill_(beta2)\n"),
   (ngramTableFile, "\n                   positive_history=True)\n"),
   (ngramTableFile, "\n                                        positive_history=True):\n"),
   (rowPrefetchFile, "\n        self.ngram.start_land(self.next_pull.ngram)\n        # Behind the serves, so it delays neither exchange; it writes no row they read.\n        self.ngram.bring_current(self.next_pull.ngram)\n"),
   (ngramTableFile, "\n        self.grad_per_slot = torch.empty(max_cycle_rows, NGRAM_DIM, dtype=GRAD_ACCUM_DTYPE, device=device)\n"),
   (ngramTableFile, "\n        wire = pack_ngram_gradients(per_slot, lo, hi, clear_source=True)\n        reset_whole = wire is None\n"),
   (ngramTableFile, "\n        scatter_add_rows(own_grad, own_rows, grad, row_map=self.entry_of_row, clear_source=not reset_whole)\n        if reset_whole:\n            self.grad_per_slot[:pull.num_rows].zero_()\n"),
   (rowPrefetchFile, "\n        return [self.batches.peek(self.steps[j]) for j in cycle]\n"),
   (rowPrefetchFile, "\n        k = bisect.bisect_left(self.update_positions, i)\n        last = self.update_positions[k] if k < len(self.update_positions) else len(self.steps) - 1\n        return list(range(i, last + 1))\n"),
   (rowPrefetchFile, "\n        self.update_positions = [i for i, step in enumerate(self.steps) if is_update(step)]\n"),
   ("track_1_short/config.py", "\n    val_batch_size: int = 4 * 64 * 1024 * 8\n"),
   ("train_gpt.py", "\n    val_tokens_per_rank = args.val_batch_size // env.world_size\n"),
   ("train_gpt.py", "\n        max_seq_len=val_tokens_per_rank,\n"),
   (modelFile, "\n        self.register_buffer(\"ngram_cache\", torch.zeros(2 * max_seq_len, ngram_dim, dtype=torch.bfloat16, device=device),\n                             persistent=False)\n"),
   (rowPrefetchFile, "\n        packet = self.staging.ngram_ring.upload(ngram_want, slot_directory)\n"),
   (rowPrefetchFile, "\n        self.prefetch.land_next_cycle()\n"),
   (anvilFile, "\n            sparse_update.finish()\n"),
   (rowPrefetchFile, "\n                                member_index=i - self.live_cycle[0])\n"),
   (ngramTableFile, selectMembers),
   (ngramTableFile, uniqueRows),
   (rowPrefetchFile, "\n        uploaded.synchronize()\n        n = rows.shape[0]\n"),
   (rowPrefetchFile, "\n        out.copy_(buf[:count], non_blocking=True)\n        uploaded.record()\n        return out\n")]

def graphStreamSources (image : SourceImage) : Bool :=
  streamSites.all fun (path, fragment) =>
    match image.lookup path with
    | none => false
    | some text => (replaceOnce text fragment fragment).isSome

def enqueueGraphForward (src : LinkedSources) (other : CudaStreamOrder.Queues GraphReplayStream.Work)
    (compute : Nat) (uploaded : CudaStreamOrder.Event) :
    Option (CudaStreamOrder.Queues GraphReplayStream.Work) :=
  if graphStreamSources src.candidate then some (GraphReplayStream.enqueue other compute uploaded) else none

theorem enqueue_graph_forward_exact (src : LinkedSources) (other : CudaStreamOrder.Queues GraphReplayStream.Work)
    (compute : Nat) (uploaded : CudaStreamOrder.Event) (initial : CudaStreamOrder.Queues GraphReplayStream.Work)
    (enqueued : enqueueGraphForward src other compute uploaded = some initial) :
    initial = GraphReplayStream.enqueue other compute uploaded := by
  unfold enqueueGraphForward at enqueued
  split at enqueued
  · exact (Option.some.inj enqueued).symm
  · contradiction

def enqueueGraphCycleForward (src : LinkedSources) (other : CudaStreamOrder.Queues GraphReplayStream.Work)
    (compute : Nat) (wantUploaded batchUploaded : CudaStreamOrder.Event) :
    Option (CudaStreamOrder.Queues GraphReplayStream.Work) :=
  if graphStreamSources src.candidate then
    some (GraphReplayStream.enqueueCycle other compute wantUploaded batchUploaded)
  else none

theorem enqueue_graph_cycle_forward_exact (src : LinkedSources)
    (other : CudaStreamOrder.Queues GraphReplayStream.Work) (compute : Nat)
    (wantUploaded batchUploaded : CudaStreamOrder.Event) (initial : CudaStreamOrder.Queues GraphReplayStream.Work)
    (enqueued : enqueueGraphCycleForward src other compute wantUploaded batchUploaded = some initial) :
    initial = GraphReplayStream.enqueueCycle other compute wantUploaded batchUploaded := by
  unfold enqueueGraphCycleForward at enqueued
  split at enqueued
  · exact (Option.some.inj enqueued).symm
  · contradiction

#axiomsClean enqueue_graph_forward_exact
#axiomsClean enqueue_graph_cycle_forward_exact

def enqueueNgramBackward (src : LinkedSources) (cycle : NgramAccumulation.Cycle)
    (other : CudaStreamOrder.Queues (NgramAccumulation.Work cycle.cut.rows)) (compute : Nat) :
    Option (CudaStreamOrder.Queues (NgramAccumulation.Work cycle.cut.rows)) :=
  if graphStreamSources src.candidate then
    some {other with pending := fun i => if i = compute then NgramAccumulation.commands cycle else other.pending i}
  else none

/-- The byte-checked training sites issue the backward and its consumer
on the same stream. Completion of that stream gives the reference sum. -/
theorem enqueue_ngram_backward_exact (src : LinkedSources) (cycle : NgramAccumulation.Cycle)
    (compute : Nat) {other initial final : CudaStreamOrder.Queues (NgramAccumulation.Work cycle.cut.rows)}
    (enqueued : enqueueNgramBackward src cycle other compute = some initial)
    (issued : List (Nat × CudaStreamOrder.Command (NgramAccumulation.Work cycle.cut.rows)))
    (run : CudaStreamOrder.Run initial issued final) (finished : final.pending compute = [])
    (capacity : Nat) (fit : cycle.cut.rows ≤ capacity) (ready : NgramScratch.Clean capacity cycle.prior) :
    (CudaStreamOrder.issuedOn compute issued).foldlM (NgramAccumulation.executeFrom cycle.prior) ⟨none, none, none⟩ =
      some ⟨none, none, some (NgramAccumulation.delayed cycle)⟩ := by
  have submitted : initial.pending compute = NgramAccumulation.commands cycle := by
    unfold enqueueNgramBackward at enqueued
    split at enqueued
    · rw [← Option.some.inj enqueued]
      exact if_pos rfl
    · contradiction
  have replay := NgramAccumulation.queued_cycle cycle compute submitted run finished
  unfold NgramAccumulation.execute at replay
  rw [NgramScratch.ready_prefix _ _ _ fit ready] at replay
  exact replay

#axiomsClean enqueue_ngram_backward_exact

end VeriTile.Bench.Models.NanoGPT.Track1.Driver
