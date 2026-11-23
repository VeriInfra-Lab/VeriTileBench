import models.NanoGPT.Track1.SourceBinding
import models.NanoGPT.Track1.GraphStageMemory
import models.NanoGPT.Track1.GraphSlotPTXBinding
import models.NanoGPT.Track1.GraphStreamBinding
import models.NanoGPT.Track1.NgramCycleRows
import models.NanoGPT.Track1.NgramCycleSchedule
import models.NanoGPT.Track1.NgramEmbedBinding
import models.NanoGPT.Track1.RowGradientExchange
import models.NanoGPT.Track1.NgramAccumulationBinding
import models.NanoGPT.Track1.NgramScratchBinding
import models.NanoGPT.Track1.MomentReplayBinding
import models.NanoGPT.Track1.NgramAdamBinding
import models.NanoGPT.Track1.SparsePacketsBinding
import models.NanoGPT.Track1.ValueAdamBinding
import models.NanoGPT.Track1.ValueEmbedAccumBinding
import models.NanoGPT.Track1.SampledInverseBinding
import models.NanoGPT.Track1.SampledDensifyBinding
import models.NanoGPT.Track1.CETransposeBinding
import models.NanoGPT.Track1.TrainingCEGradientStores
import models.NanoGPT.Track1.TrainingCELossBinding
import models.NanoGPT.Track1.TrainingCEInitBinding
import models.NanoGPT.Track1.TrainingCESparseBinding
import models.NanoGPT.Track1.TrainingCEReductionBinding
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

/-! Contextual FP equivalence from the actual Track 1 script entry point
to the final FineWeb validation. The substituted operations have
different numerical implementations. Every other call is resolved in
the byte-checked common source package.

The theorem is in the driver's numerical execution semantics. Python,
PyTorch, Triton, CUDA graph replay and NCCL implement the per-operation
runtime interface. A refinement proof from those systems to this
interface is a separate obligation. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver
open VeriTile Triton.FP
open scoped VeriTile.Spec

/-- The common inputs to a coupled numerical replay. Data files include
the ordered training and validation shard bytes and shared tokenizer
assets. Counters, entropy and the order in which asynchronous events
become observable are runtime inputs even for the pristine trainer. -/
structure TrainingInput where
  files : List (String × ByteArray)
  environment : List (String × String)
  entropy : List UInt64
  eventOrder : List Nat

/-- Both runs resolve the exact same unchanged source and runtime operations.
The boot step supplies common initial inputs; there is no assumed
relation or numeric equality between two runs. -/
structure World (State Value : Type) where
  boot : SourceImage → TrainingInput → Option State
  resolve : SourceImage → Runtime State Value

structure SourceProgram where
  side : TrackProgram
  actual : SourceImage

def runWith {State Value : Type} (p : SourceProgram) (w : World State Value)
    (input : TrainingInput) (calls : PatchedCalls) : Option (Result State) := do
  let common ← normalize p.side p.actual
  let initial ← w.boot common input
  execute (w.resolve common) calls initial

def run {State Value : Type} (p : SourceProgram) (w : World State Value)
    (input : TrainingInput) : Option (Result State) :=
  runWith p w input p.side.calls

def LinkedSources.baseline (src : LinkedSources) : SourceProgram := ⟨.reference, src.reference⟩
def LinkedSources.optimized (src : LinkedSources) : SourceProgram := ⟨.candidate, src.candidate⟩

theorem linked_run_equal (src : LinkedSources) {State Value : Type} (w : World State Value)
    (input : TrainingInput) :
    run src.optimized w input = run src.baseline w input := by
  simp only [run, runWith, LinkedSources.optimized, LinkedSources.baseline, src.candidateOK,
    src.referenceOK, TrackProgram.calls, execution_equal]

inductive TrainingStatement where
  | driver (stmt : Stmt)
  | implementation (call : CallCode)

inductive TrainingIO where
  | fineWebShardBytes | tokenizerAssets | environment | entropy | eventOrder
  | finalRuntimeState | fp32Validations

def sameRun (a b : SourceProgram) : Prop :=
  ∀ (State Value : Type) (w : World State Value) (input : TrainingInput),
    run a w input = run b w input

instance : Spec.ProgramSyntax SourceProgram where
  Statement := TrainingStatement
  Signature := List TrainingIO × List TrainingIO
  signature := fun _ =>
    ([.fineWebShardBytes, .tokenizerAssets, .environment, .entropy, .eventOrder],
     [.finalRuntimeState, .fp32Validations])
  body := fun p =>
    (moduleInitialization ++ main ++ trainingStep).map .driver ++ p.side.body.map .implementation
  structural := some sameRun

/-- Complete driver equivalence for the checked source pair, including the
warmup reset, all training iterations and the final validation readout.
The local kernel proofs discharge the substitutions with an empty rule
table. No equality of optimizer steps, validation, or training is a premise. -/
specification training_end_to_end (src : LinkedSources) :
    src.optimized ≡[([] : Spec.Assumptions TrainingStatement)] src.baseline := by
  refine Spec.FloatingPoint.ofStructural (lhs := src.optimized) (rhs := src.baseline)
    (structural := sameRun) rfl rfl ?_
  exact fun _ _ w input => linked_run_equal src w input

/-- Training graph replay after an access to the scheduled FineWeb stream.
The cache invariant is established on open and preserved by each access. -/
def ScheduledGraphReplay (side : TrackProgram) : Prop :=
  ∀ (stream : FineWeb.ScheduledStream) (consume : Bool) (step : Nat)
    (batch : FineWeb.TokenBatch) (after : FineWeb.ScheduledStream),
    FineWeb.HashedCache stream.ahead →
    stream.access consume step side.calls.packTraining side.calls.ngramRowHash = some (batch, after) →
    ∀ (s : GraphStage.Shape) (n : Nat) (layout : GraphStageMemory.Layout s n)
      (transfer : GraphStageMemory.BatchTransfer) (prior : CudaStores.Memory),
      batch.ngramRows.size = GraphStage.extent s .slots →
      transfer.rowPlacement layout batch →
      let memory := transfer.upload layout batch prior
      ∀ (ready : GraphStageMemory.Ready layout memory),
      ∀ (schedule : List Nat), (GraphStageMemory.grid s).Perm schedule → n < 2147483648 →
      GraphStageMemory.observe layout (GraphStageMemory.launchHashed layout schedule memory) =
        GraphStage.fusedLookup s (GraphStageMemory.readField layout memory .source) ready.want
          (fun i => GraphStageMemory.query layout memory i.val) ∧
      (∀ address, ¬ GraphStageMemory.writes layout address →
        GraphStageMemory.launchHashed layout schedule memory address = memory address) ∧
      FineWeb.HashedCache after.ahead

theorem scheduled_graph_replay (side : TrackProgram) : ScheduledGraphReplay side := by
  intro stream consume step batch after valid access s n layout transfer prior count placement
    memory ready schedule complete fits
  have hashed := FineWeb.access_rows_hashed stream consume step _ _ side.batch_hash valid batch after access
  have upload := GraphStageMemory.transfer_uploads_hashes transfer layout batch prior count placement
  have replay := GraphStageMemory.hashed_replay_refines s n layout memory ready batch hashed.1 upload
    schedule complete fits
  exact ⟨replay.1, replay.2, hashed.2⟩

theorem scheduled_lookup_lane (src : LinkedSources) {s : GraphStage.Shape} {ptx : String}
    (artifact : GraphSlotPTXBinding.Certificate s ptx) (stream : FineWeb.ScheduledStream)
    (consume : Bool) (step : Nat) (batch : FineWeb.TokenBatch) (after : FineWeb.ScheduledStream)
    (cache : FineWeb.HashedCache stream.ahead)
    (access : stream.access consume step src.optimized.side.calls.packTraining
      src.optimized.side.calls.ngramRowHash = some (batch, after))
    {n : Nat} (layout : GraphStageMemory.Layout s n) (transfer : GraphStageMemory.BatchTransfer)
    (prior : CudaStores.Memory) (count : batch.ngramRows.size = GraphStage.extent s .slots)
    (placement : transfer.rowPlacement layout batch)
    (ready : GraphStageMemory.Ready layout (transfer.upload layout batch prior)) (fits : n < 2147483648)
    (word : Fin (GraphStage.extent s .slots)) :
    let memory := transfer.upload layout batch prior
    GraphSlotPTXBinding.interpretLane s ptx layout memory
      (GraphStage.queryCTA s word.val) (GraphStage.queryLane word.val) =
      some (GraphSlotPTXBinding.slotStores layout word.val
        (NgramSlots.native false ready.want (GraphStageMemory.query layout memory word.val))) ∧
    (∀ read ∈ GraphSlotPTXBinding.lookupReads layout memory word.val,
      4 * (read.2 + 1) ≤ read.1.bytes s n) := by
  have hashed := FineWeb.access_rows_hashed stream consume step _ _
    src.optimized.side.batch_hash cache batch after access
  have upload := GraphStageMemory.transfer_uploads_hashes transfer layout batch prior count placement
  exact ⟨artifact.stores n layout _ ready fits batch hashed.1 upload word,
    artifact.reads n layout _ ready fits batch hashed.1 upload word⟩

theorem scheduled_compiled_replay (src : LinkedSources) {s : GraphStage.Shape} {ptx : String}
    (artifact : GraphSlotPTXBinding.Certificate s ptx) (stream : FineWeb.ScheduledStream)
    (consume : Bool) (step : Nat) (batch : FineWeb.TokenBatch) (after : FineWeb.ScheduledStream)
    (cache : FineWeb.HashedCache stream.ahead)
    (access : stream.access consume step src.optimized.side.calls.packTraining
      src.optimized.side.calls.ngramRowHash = some (batch, after))
    {n : Nat} (layout : GraphStageMemory.Layout s n) (transfer : GraphStageMemory.BatchTransfer)
    (prior : CudaStores.Memory) (count : batch.ngramRows.size = GraphStage.extent s .slots)
    (placement : transfer.rowPlacement layout batch)
    (ready : GraphStageMemory.Ready layout (transfer.upload layout batch prior)) (fits : n < 2147483648)
    (schedule : List Nat) (complete : (GraphStageMemory.grid s).Perm schedule) :
    let memory := transfer.upload layout batch prior
    let output := GraphStageMemory.launch layout schedule memory
    GraphSlotPTXBinding.interpretLaunch s ptx layout schedule memory = some output ∧
    GraphStageMemory.observe layout output =
      GraphStage.fusedLookup s (GraphStageMemory.readField layout memory .source) ready.want
        (fun i => GraphStageMemory.query layout memory i.val) ∧
    (∀ address, ¬ GraphStageMemory.writes layout address → output address = memory address) := by
  have hashed := FineWeb.access_rows_hashed stream consume step _ _
    src.optimized.side.batch_hash cache batch after access
  have upload := GraphStageMemory.transfer_uploads_hashes transfer layout batch prior count placement
  exact ⟨artifact.replay n layout _ ready fits batch hashed.1 upload schedule,
    GraphStageMemory.replay_refines s n layout _ ready schedule complete fits⟩

theorem scheduled_stream_replay (src : LinkedSources) {s : GraphStage.Shape} {ptx : String}
    (artifact : GraphSlotPTXBinding.Certificate s ptx) (stream : FineWeb.ScheduledStream)
    (consume : Bool) (step : Nat) (batch : FineWeb.TokenBatch) (after : FineWeb.ScheduledStream)
    (cache : FineWeb.HashedCache stream.ahead)
    (access : stream.access consume step src.optimized.side.calls.packTraining
      src.optimized.side.calls.ngramRowHash = some (batch, after))
    {n : Nat} (layout : GraphStageMemory.Layout s n) (transfer : GraphStageMemory.BatchTransfer)
    (prior : CudaStores.Memory) (count : batch.ngramRows.size = GraphStage.extent s .slots)
    (placement : transfer.rowPlacement layout batch)
    (ready : GraphStageMemory.Ready layout (transfer.upload layout batch prior)) (fits : n < 2147483648)
    (schedule : List Nat) (complete : (GraphStageMemory.grid s).Perm schedule)
    (other initial final : CudaStreamOrder.Queues GraphReplayStream.Work)
    (compute : Nat) (uploaded : CudaStreamOrder.Event)
    (enqueued : enqueueGraphForward src other compute uploaded = some initial)
    (issued : List (Nat × CudaStreamOrder.Command GraphReplayStream.Work))
    (run : CudaStreamOrder.Run initial issued final) (finished : final.pending compute = []) :
    let memory := transfer.upload layout batch prior
    let output := GraphStageMemory.launch layout schedule memory
    GraphReplayStream.execute layout ptx batch transfer schedule (CudaStreamOrder.issuedOn compute issued)
      ⟨prior, []⟩ =
      some ⟨output, [GraphStage.separateLookup s (GraphStageMemory.readField layout memory .source)
        ready.want (fun i => GraphStageMemory.query layout memory i.val)]⟩ ∧
    uploaded ∈ final.completed ∧
    (∀ address, ¬ GraphStageMemory.writes layout address → output address = memory address) := by
  have queue := enqueue_graph_forward_exact src other compute uploaded initial enqueued
  subst initial
  have launch := scheduled_compiled_replay src artifact stream consume step batch after cache access
    layout transfer prior count placement ready fits schedule complete
  have order := GraphReplayStream.replay_ordered layout ptx batch transfer schedule prior
    other final compute uploaded issued run finished
  rw [launch.1, Option.map_some, launch.2.1, GraphStage.lookup_stage_equal] at order
  exact ⟨order, GraphReplayStream.upload_recorded other final compute uploaded issued run finished, launch.2.2⟩

/-- A replay with a prepared directory packet already resident. The
directory and wanted IDs are derived from the packet rather than supplied
as a readiness relation on the staging input. -/
theorem scheduled_packet_stream_replay (src : LinkedSources) {s : GraphStage.Shape} {ptx : String}
    (artifact : GraphSlotPTXBinding.Certificate s ptx) (stream : FineWeb.ScheduledStream)
    (consume : Bool) (step : Nat) (batch : FineWeb.TokenBatch) (after : FineWeb.ScheduledStream)
    (cache : FineWeb.HashedCache stream.ahead)
    (access : stream.access consume step src.optimized.side.calls.packTraining
      src.optimized.side.calls.ngramRowHash = some (batch, after))
    (want : NgramSlots.Want) (directory : NgramDirectoryPacket.Prepared want)
    (layout : GraphStageMemory.Layout s want.size) (transfer : GraphStageMemory.BatchTransfer)
    (prior : CudaStores.Memory) (count : batch.ngramRows.size = GraphStage.extent s .slots)
    (rowPlacement : transfer.rowPlacement layout batch)
    (dirPlacement : layout.base .directory = layout.base .want + 4 * NgramSlots.directoryStart want.size)
    (views : NgramDirectoryPacket.BatchViews layout transfer batch)
    (inputAligned : layout.base (.source .inputs) % 16 = 0)
    (wantAligned : layout.base .want % 16 = 0) (directoryAligned : layout.base .directory % 16 = 0)
    (schedule : List Nat) (complete : (GraphStageMemory.grid s).Perm schedule)
    (other initial final : CudaStreamOrder.Queues GraphReplayStream.Work)
    (compute : Nat) (uploaded : CudaStreamOrder.Event)
    (enqueued : enqueueGraphForward src other compute uploaded = some initial)
    (issued : List (Nat × CudaStreamOrder.Command GraphReplayStream.Work))
    (run : CudaStreamOrder.Run initial issued final) (finished : final.pending compute = []) :
    let resident := NgramDirectoryPacket.upload want (layout.base .want) prior
    let memory := transfer.upload layout batch resident
    let ready := NgramDirectoryPacket.ready_after_batch layout resident
      (NgramDirectoryPacket.upload_ready want directory layout prior dirPlacement) transfer batch
      (NgramDirectoryPacket.batch_disjoint_of_layout layout transfer batch count views
        inputAligned wantAligned directoryAligned)
    let output := GraphStageMemory.launch layout schedule memory
    GraphReplayStream.execute layout ptx batch transfer schedule (CudaStreamOrder.issuedOn compute issued)
      ⟨resident, []⟩ =
      some ⟨output, [GraphStage.separateLookup s (GraphStageMemory.readField layout memory .source)
        ready.want (fun i => GraphStageMemory.query layout memory i.val)]⟩ ∧
    uploaded ∈ final.completed ∧
    (∀ address, ¬ GraphStageMemory.writes layout address → output address = memory address) := by
  exact scheduled_stream_replay src artifact stream consume step batch after cache access layout transfer
    _ count rowPlacement _ directory.fits schedule complete other initial final compute uploaded
    enqueued issued run finished

private theorem uploaded_cycle_members {s : GraphStage.Shape} (cycle : NgramCycleRows.Cycle)
    (live : Nat → Bool) (position object : Nat) (batch : FineWeb.TokenBatch)
    (selected : cycle.select live position object = some batch)
    (layout : GraphStageMemory.Layout s cycle.want.size) (memory : CudaStores.Memory)
    (upload : GraphStageMemory.UploadedHashes layout memory batch)
    (ready : GraphStageMemory.Ready layout memory)
    (contents : GraphStageMemory.readWord32 layout memory .want = (NgramDirectoryPacket.boundedWant cycle.want).read) :
    GraphMemberLaunch.RowsPresent layout memory ready := by
  intro index
  have hashed := NgramCycleRows.selected_hashed cycle live position object batch selected
  rw [GraphStageMemory.uploaded_query_word layout memory batch hashed upload index]
  have within : index.val < batch.ngramRows.size := by rw [upload.count]; exact index.isLt
  obtain ⟨i, inside, wanted⟩ :=
    NgramCycleRows.selected_contains cycle live position object batch selected
      (batch.ngramRows[index.val]'within) (Array.getElem_mem within)
  refine ⟨i, inside, ?_⟩
  change GraphStageMemory.readWord32 layout memory .want i = _
  rw [contents]
  simpa only [NgramDirectoryPacket.boundedWant, if_pos inside] using wanted

/-- Construct the driver's member launch packet from the cycle that
published the directory and the exact FineWeb batch selected for it. -/
def memberCapturePacket {s : GraphStage.Shape} (cycle : NgramCycleRows.Cycle)
    (live : Nat → Bool) (position object : Nat) (batch : FineWeb.TokenBatch)
    (selected : cycle.select live position object = some batch)
    (directory : NgramDirectoryPacket.Prepared cycle.want)
    (layout : GraphStageMemory.Layout s cycle.want.size) (transfer : GraphStageMemory.BatchTransfer)
    (prior : CudaStores.Memory) (count : batch.ngramRows.size = GraphStage.extent s .slots)
    (placement : transfer.rowPlacement layout batch)
    (dirPlacement : layout.base .directory = layout.base .want + 4 * NgramSlots.directoryStart cycle.want.size)
    (rowsAligned : layout.base (.source .slots) % 16 = 0) (wantAligned : layout.base .want % 16 = 0)
    (schedule : List Nat) (complete : (GraphStageMemory.grid s).Perm schedule) : GraphStage.Packet s :=
  let batchMemory := transfer.upload layout batch prior
  let memory := NgramDirectoryPacket.upload cycle.want (layout.base .want) batchMemory
  let ready := NgramDirectoryPacket.upload_ready cycle.want directory layout batchMemory dirPlacement
  have hashed := NgramCycleRows.selected_hashed cycle live position object batch selected
  have staged := GraphStageMemory.transfer_uploads_hashes transfer layout batch prior count placement
  have rows := NgramDirectoryPacket.upload_preserves_aligned_view cycle.want layout batchMemory dirPlacement
    wantAligned (.source .slots) rowsAligned (by decide) (by decide)
  have upload : GraphStageMemory.UploadedHashes layout memory batch :=
    ⟨count, fun i => (congrFun rows i.val).trans (staged.word i)⟩
  ⟨cycle.want.size, layout, memory, ready, directory.fits, batch, hashed, upload, schedule, complete, true,
    fun _ => uploaded_cycle_members cycle live position object batch selected layout memory upload ready
      (NgramDirectoryPacket.uploaded_want cycle.want layout batchMemory)⟩

theorem hopper_training_stream_bound (schedule : HostSchedule.Plan) (plan : TrainingSourcePlan)
    (world : plan.worldSize = 8) (geometry : List (Nat × FineWeb.TrainingGeometry))
    (planned : trainingGeometries schedule plan.steps = some geometry) (stream : FineWeb.ScheduledStream)
    (opened : FineWeb.ScheduledStream.open plan.files plan.readiness plan.worldSize plan.rank geometry =
      some stream) : stream.Bounded 49152 := by
  apply FineWeb.open_bounded _ _ _ _ _ _ _ opened
  intro entry member
  rw [world]
  exact Nat.div_le_div_right (training_geometries_bounded schedule plan.steps geometry planned entry member)

/-- Deduplication and the timed cadence bound the live want list before
either the gather or embedding reads the allocated cache. -/
theorem timed_cycle_capacity (side : TrackProgram) (total start : Nat) (objects : Nat → Nat)
    (stream : FineWeb.ScheduledStream) (hashes : FineWeb.HashedCache stream.ahead)
    (geometry : stream.Bounded 49152) (entries : List NgramCycleRows.Entry) (after : FineWeb.ScheduledStream)
    (captured : NgramCycleRows.capture side.calls.packTraining side.calls.ngramRowHash
      (NgramCycleSchedule.timedRequests total start objects) stream = some (entries, after)) :
    let cycle := NgramCycleRows.ofCapture _ _ side.batch_hash
      (NgramCycleSchedule.timedRequests total start objects) stream hashes entries after captured
    cycle.want.size ≤ 393216 ∧ cycle.want.size ≤ NgramCycleSchedule.defaultCacheRows 8 ∧
      after.Bounded 49152 := by
  intro cycle
  obtain ⟨length, sized, remaining⟩ := NgramCycleRows.capture_sizes _ _ side.batch_size
    (NgramCycleSchedule.timedRequests total start objects) stream 49152 geometry entries after captured
  have count : cycle.entries.length ≤ 4 := by
    change entries.length ≤ 4
    rw [length]
    exact NgramCycleSchedule.timed_requests_le_four total start objects
  have fits : cycle.want.size ≤ 393216 :=
    Nat.le_trans (cycle.want_size 49152 sized) (Nat.mul_le_mul_right (2 * 49152) count)
  exact ⟨fits, Nat.le_trans fits (by decide), remaining⟩

/-- Replay uses the batch selected from the cycle that uploaded the
directory. Each staged slot names the BF16 row landed from its owner. -/
theorem scheduled_member_stream_replay (src : LinkedSources) {s : GraphStage.Shape} {ptx : String}
    (artifact : GraphSlotPTXBinding.MemberCertificate s ptx) (stream : FineWeb.ScheduledStream)
    (requests : List (Nat × Nat)) (entries : List NgramCycleRows.Entry) (after : FineWeb.ScheduledStream)
    (cache : FineWeb.HashedCache stream.ahead)
    (captured : NgramCycleRows.capture src.optimized.side.calls.packTraining
      src.optimized.side.calls.ngramRowHash requests stream = some (entries, after)) :
    let cycle := NgramCycleRows.ofCapture _ _ src.optimized.side.batch_hash requests stream cache entries after captured
    ∀ (live : Nat → Bool) (position object : Nat) (batch : FineWeb.TokenBatch),
    cycle.select live position object = some batch →
    ∀ (directory : NgramDirectoryPacket.Prepared cycle.want)
      (layout : GraphStageMemory.Layout s cycle.want.size) (transfer : GraphStageMemory.BatchTransfer)
      (prior : CudaStores.Memory) (count : batch.ngramRows.size = GraphStage.extent s .slots),
    transfer.rowPlacement layout batch →
    (layout.base .directory = layout.base .want + 4 * NgramSlots.directoryStart cycle.want.size) →
    (∀ f : GraphStage.Field, layout.base (.source f) % 16 = 0) →
    layout.base .want % 16 = 0 →
    ∀ (schedule : List Nat), (GraphStageMemory.grid s).Perm schedule →
    ∀ (other initial final : CudaStreamOrder.Queues GraphReplayStream.Work)
      (compute : Nat) (wantUploaded batchUploaded : CudaStreamOrder.Event),
    enqueueGraphCycleForward src other compute wantUploaded batchUploaded = some initial →
    ∀ (issued : List (Nat × CudaStreamOrder.Command GraphReplayStream.Work)),
    CudaStreamOrder.Run initial issued final → final.pending compute = [] →
    let writeWant := NgramDirectoryPacket.upload cycle.want (layout.base .want)
    let batchMemory := transfer.upload layout batch prior
    let memory := writeWant batchMemory
    let output := GraphStageMemory.launch layout schedule memory
    let stage := GraphSlotPTXBinding.interpretMemberLaunch s ptx layout schedule
    let observed := GraphStage.separateLookup s (GraphStageMemory.readField layout batchMemory .source)
      (NgramDirectoryPacket.boundedWant cycle.want) (fun i => GraphStageMemory.query layout batchMemory i.val)
    GraphReplayStream.executeReplay layout writeWant stage batch transfer (CudaStreamOrder.issuedOn compute issued)
      ⟨prior, []⟩ =
      some ⟨output, [observed]⟩ ∧
    (wantUploaded ∈ final.completed ∧ batchUploaded ∈ final.completed) ∧
    (∀ address, ¬ GraphStageMemory.writes layout address → output address = memory address) ∧
    (∀ (word : Fin (GraphStage.extent s .slots)) read,
      read ∈ GraphSlotPTXBinding.memberReads layout memory word.val →
      4 * (read.2 + 1) ≤ read.1.bytes s cycle.want.size) ∧
    (∀ word : Fin (GraphStage.extent s .slots),
      let slot := (observed.graph .slots word).toNat
      let row := (GraphStageMemory.query layout batchMemory word.val).toNat
      slot < cycle.want.size ∧ observed.pendingSlots word = slot ∧ cycle.want.read slot = row ∧
      ∀ (peers : Nat → NgramSlots.Want) (banks : RowExchange.Banks) (rank : Fin 8)
        (previous : SparseLanding.Rows) (col : SparsePackets.Channel),
        let wanted := Function.update peers rank.val cycle.want
        SparseLanding.directLand (RowExchange.landingInput wanted banks rank previous) slot col =
          some (RowExchange.globalWord banks row col)) := by
  intro cycle live position object batch selected directory layout transfer prior count placement dirPlacement
    sourcesAligned wantAligned schedule complete other initial final compute wantUploaded batchUploaded
    enqueued issued run finished writeWant batchMemory memory output stage observed
  have queue := enqueue_graph_cycle_forward_exact src other compute wantUploaded batchUploaded initial enqueued
  subst initial
  let packet := memberCapturePacket cycle live position object batch selected directory layout transfer prior count
    placement dirPlacement (sourcesAligned .slots) wantAligned schedule complete
  let ready : GraphStageMemory.Ready layout memory := packet.ready
  have written := NgramDirectoryPacket.uploaded_want cycle.want layout batchMemory
  have frames (f : GraphStage.Field) : GraphStageMemory.readWord32 layout memory (.source f) =
      GraphStageMemory.readWord32 layout batchMemory (.source f) :=
    NgramDirectoryPacket.upload_preserves_aligned_view cycle.want layout batchMemory dirPlacement
      wantAligned (.source f) (sourcesAligned f) (by cases f <;> decide) (by cases f <;> decide)
  have upload : GraphStageMemory.UploadedHashes layout memory batch := packet.upload
  have members : GraphMemberLaunch.RowsPresent layout memory ready := packet.rowsPresent rfl
  have hashed : batch.hashedRows := packet.hashed
  have launch : stage memory = some output :=
    artifact.replay _ layout memory ready directory.fits batch hashed upload members schedule
  have replay := GraphStageMemory.replay_refines s _ layout memory ready schedule complete directory.fits
  have contents : ready.want = NgramDirectoryPacket.boundedWant cycle.want := by
    change GraphStageMemory.Ready.want (NgramDirectoryPacket.upload_ready cycle.want directory layout batchMemory
      dirPlacement) = NgramDirectoryPacket.boundedWant cycle.want
    simp only [GraphStageMemory.Ready.want, NgramDirectoryPacket.boundedWant, NgramSlots.Want.mk.injEq, true_and]
    exact written
  have fields : GraphStageMemory.readField layout memory .source =
      GraphStageMemory.readField layout batchMemory .source := by
    funext f i
    cases f <;> simp only [GraphStageMemory.readField, frames]
  have rows (i : Nat) : GraphStageMemory.query layout memory i = GraphStageMemory.query layout batchMemory i := by
    simp only [GraphStageMemory.query, frames]
  have order := GraphReplayStream.replay_cycle_ordered layout writeWant stage batch transfer prior
    other final compute wantUploaded batchUploaded issued run finished
  change _ = (stage memory).map _ at order
  rw [launch, Option.map_some, show GraphStageMemory.observe layout output = _ from replay.1,
    GraphStage.lookup_stage_equal, contents] at order
  rw [fields] at order
  simp only [rows] at order
  refine ⟨order, GraphReplayStream.cycle_uploads_recorded other final compute wantUploaded batchUploaded
    issued run finished, replay.2,
    artifact.reads _ layout memory ready directory.fits batch hashed upload members, ?_⟩
  intro word slot row
  have valid := GraphStageMemory.hashed_query_in_range layout memory batch hashed upload word
  let query := GraphStageMemory.query layout memory word.val
  let index := NgramSlots.nativeNat ready.want query.toNat
  obtain ⟨limit, atIndex⟩ := RowExchange.wanted_member_slot ready.want query.toNat (members word)
  have stored : (observed.graph .slots word).toNat = index := by
    change (UInt64.ofNat (NgramSlots.native false (NgramDirectoryPacket.boundedWant cycle.want)
      (GraphStageMemory.query layout batchMemory word.val))).toNat = index
    rw [← contents, ← rows, NgramSlots.native, if_neg (Int.not_lt.mpr valid.1)]
    exact UInt64.toNat_ofNat_of_lt' (Nat.lt_trans limit (Nat.lt_trans directory.fits (by decide)))
  have readAt : ready.want.read index = cycle.want.read index := by
    rw [congrFun (congrArg NgramSlots.Want.read contents) index]
    exact if_pos limit
  have bounded : slot < cycle.want.size := by
    change (observed.graph .slots word).toNat < cycle.want.size
    rw [stored]
    exact limit
  have route : cycle.want.read slot = row := by
    change cycle.want.read (observed.graph .slots word).toNat =
      (GraphStageMemory.query layout batchMemory word.val).toNat
    rw [stored, ← rows, ← readAt]
    exact atIndex
  have pending : observed.pendingSlots word = slot := by
    change NgramSlots.native false (NgramDirectoryPacket.boundedWant cycle.want)
      (GraphStageMemory.query layout batchMemory word.val) = (observed.graph .slots word).toNat
    rw [stored, ← contents, ← rows, NgramSlots.native, if_neg (Int.not_lt.mpr valid.1)]
  refine ⟨bounded, pending, route, ?_⟩
  intro peers banks rank previous col wanted
  have sameCycle : wanted rank = cycle.want := Function.update_self _ _ _
  have landed := RowExchange.landed_cache_word wanted banks rank previous slot col (sameCycle.symm ▸ bounded)
  rw [sameCycle, route] at landed
  exact landed

structure Certificate where
  sources : LinkedSources
  proof : sources.optimized ≡[([] : Spec.Assumptions TrainingStatement)] sources.baseline
  fusedStage : GraphStageMemory.ReplayRefinement
  hashedFusedStage : ScheduledGraphReplay sources.optimized.side
  streamSources : graphStreamSources sources.candidate = true

/-- Readily executable entry point from two source images to a checked
equivalence certificate. The proof field is filled by the kernel theorem. -/
def certify (baseline candidate : SourceImage) : Option Certificate := do
  let sources ← link baseline candidate
  if stream : graphStreamSources sources.candidate = true then
    some ⟨sources, training_end_to_end sources, GraphStageMemory.replay_refines,
      scheduled_graph_replay sources.optimized.side, stream⟩
  else none

theorem final_fineweb_validation_equal (src : LinkedSources) {State Value : Type}
    (w : World State Value) (input : TrainingInput) :
    (run src.optimized w input).bind Result.finalValidation =
      (run src.baseline w input).bind Result.finalValidation := by
  rw [linked_run_equal]

/-- The sorted enumeration of the uploaded bitmap drives both FP8
layouts. The saved inverse is consumed by the BF16 gradient scatter
with the same signed positions, including -1 for every absent class. -/
theorem compiled_sampled_forward_backward (c : SampledRows.Count)
    {gir ptx : String} (gather : SampledInverse.Certificate c gir ptx)
    {code : Densify.Artifacts} (densify : Densify.Certificate c code)
    (mark : SampledRows.ClassID → Bool) (size : (Densify.flatNonzero mark).length = c.rows)
    (bank : Nat → UInt8) (compactGradient : Nat → UInt16) :
    let uploaded := SampledInverse.Candidates.ofBitmap c mark size
    let inverse := SampledInverse.blockedInverse uploaded
    (∀ p : SampledRows.Point c, SampledRows.Vector.observe bank uploaded.ids p =
      SampledRows.reference c bank uploaded.ids p.row p.column) ∧
    (∀ cls, Densify.encodedPosition (inverse cls) =
      Densify.encodedPosition (Densify.inverseMap uploaded.ids cls)) ∧
    (∀ p, Densify.emitted compactGradient inverse p =
      Densify.reference compactGradient (Densify.inverseMap uploaded.ids) p) := by
  intro uploaded inverse
  have exactMap : inverse = Densify.inverseMap uploaded.ids := gather.inverseMap uploaded
  exact ⟨gather.gatherStores bank uploaded, fun cls => by rw [exactMap],
    fun p => by rw [exactMap, densify.stores]⟩

#axiomsClean compiled_sampled_forward_backward

/-- Every byte presented to the head's weight-gradient GEMM is the byte
at the corresponding token and channel in the original FP8 operand. -/
theorem compiled_ce_weight_operands {gir ptx : String} (artifact : CETranspose.Certificate gir ptx)
    (s : CETranspose.Shape) (source : CETranspose.Operand → Nat → UInt8) (operand : CETranspose.Operand)
    (token : Fin (s.operand operand).rows) (channel : Fin (s.operand operand).cols) :
    CETranspose.paired s source ⟨operand, ByteTranspose.point _ token channel⟩ =
      some (operand, channel.val * (s.operand operand).rows + token.val,
        source operand (token.val * (s.operand operand).cols + channel.val)) := by
  rw [artifact.stores]
  exact CETranspose.staged_operand s source operand token channel

#axiomsClean compiled_ce_weight_operands

/-- The checked training forward retains both the computed loss and the
entire E5M2 row consumed by the two backward GEMMs. -/
theorem linked_training_ce (src : LinkedSources) (w : TrainingCE.Width)
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (row : TrainingCE.Row w) :
    src.optimized.side.calls.trainingCE w op p row =
      src.baseline.side.calls.trainingCE w op p row := by
  change TrainingCE.specialized w op p row = TrainingCE.reference w op p row
  rw [TrainingCE.specialized_equal]

#axiomsClean linked_training_ce

/-- The table and row staging phases may commit their stores in any
order within the respective barrier. The graph observes the same loss
and gradient bytes as the staged execution. -/
theorem linked_staged_training_ce (src : LinkedSources) (w : TrainingCE.Width)
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (row : TrainingCE.Row w)
    (tables : List CudaStores.Store) (initialized : tables.Perm (TrainingCE.tableStores op p))
    (codes : List CudaStores.Store) (staged : codes.Perm (TrainingCE.codeStores row))
    (prior : CudaStores.Memory) :
    TrainingCE.executeStaged op p row tables codes prior =
      src.optimized.side.calls.trainingCE w op p row := by
  rw [linked_training_ce src w op p row]
  exact TrainingCE.staged_loss_gradient_equal op p row tables initialized codes staged prior

#axiomsClean linked_staged_training_ce

/-- The decoded initialization writes the staged FP32 lookup entry and,
on lane zero, the gradient factor. -/
theorem compiled_ce_initialized_table {w : TrainingCE.Width} {ptx : String} {mode : TrainingCE.Softcap}
    {scale : TrainingCE.GradScale} (artifact : TrainingCEInitBinding.Certificate w ptx mode scale)
    (op : TrainingCE.Native)
    (p : TrainingCE.Parameters) (live : Fin 4) (lane : TrainingCE.Lane)
    (tables : List CudaStores.Store) (initialized : tables.Perm (TrainingCE.tableStores op p))
    (prior : CudaStores.Memory) (accepted : mode.accepts p = true) (scaleAccepted : scale.accepts p = true) :
    let code := UInt8.ofNat lane.val
    let memory := CudaStores.commitAll tables prior
    (TrainingCEInitBinding.decode w ptx live (lane.val == 0)).bind
      (TrainingCEInitBinding.Writes.eval op p code) =
      some (TrainingCE.readSharedTable memory code,
        if lane.val == 0 then some (TrainingCE.readGradientFactor memory) else none) := by
  dsimp only
  rw [TrainingCEInitBinding.initialized_table_exact artifact op p live lane accepted scaleAccepted,
    TrainingCE.published_table_words op p tables initialized prior]
  unfold TrainingCE.readGradientFactor
  rw [TrainingCE.published_gradient_factor op p tables initialized prior]

#axiomsClean compiled_ce_initialized_table

/-- For a live prediction the published word encodes its column. A
prediction truncated by the row bound publishes the all-ones sentinel. -/
theorem compiled_ce_cached_target {w : TrainingCE.Width} {ptx : String} {mode : TrainingCE.Softcap}
    {scale : TrainingCE.GradScale} (artifact : TrainingCEInitBinding.Certificate w ptx mode scale)
    (r : TrainingCE.Row w)
    (slot : Nat) (inside : slot < artifact.targetSlots) :
    ((TrainingCEInitBinding.decode w ptx (TrainingCELossBinding.liveSlots r) true).bind
        (fun writes => writes.targets.lookup slot)).map
      (fun source => TrainingCEInitBinding.targetWord (source.bind (TrainingCE.mtpTarget r))) =
      some (TrainingCEInitBinding.targetWord (TrainingCE.mtpTarget r slot)) := by
  rw [TrainingCEInitBinding.decoded_target_slot artifact _ slot inside]
  have bound := TrainingCELossBinding.live_slot r slot (Nat.lt_of_lt_of_le inside artifact.capacity)
  by_cases active : slot < (TrainingCELossBinding.liveSlots r).val
  · simp [active]
  · have truncated : ¬r.row.val + slot < r.batchRows := fun h => active (bound.mpr h)
    simp [active, TrainingCE.mtpTarget, truncated]

#axiomsClean compiled_ce_cached_target

/-- The sum input of each compiled shuffle comes from the ordered scan
of the raw code row through the initialized exponential table. -/
theorem compiled_ce_scanned_leaders {w : TrainingCE.Width} {ptx : String}
    (scan : TrainingCEGatherBinding.ScanCertificate w ptx)
    (reduction : TrainingCEReductionBinding.Certificate ptx) (memory : CudaStores.Memory) :
    reduction.reduction.readLeaders (fun lane =>
      ((TrainingCEGatherBinding.decodeScan w ptx lane).map fun trace => trace.evalSum (fun c =>
        (TrainingCE.readSharedTable memory (memory (TrainingCE.codeAddress + c))).contribution)).getD
        (VeriTile.Triton.FP.Value.ofBits .fp32 0)) = TrainingCE.stagedWarpLeaders w memory := by
  rw [TrainingCEReductionBinding.leaders_exact reduction]
  unfold TrainingCE.stagedWarpLeaders
  congr 1
  funext lane
  rw [TrainingCEGatherBinding.scan_sum_exact scan]
  rfl

/-- Both the packed-store and sparse-store address calculations stay
within the 32-bit row extent on every default training stage. -/
theorem compiled_ce_row_offsets (shape : GraphStage.Shape) (w : TrainingCE.Width)
    (row : Fin shape.tokens) (c : TrainingCE.Column w) :
    (UInt32.ofNat (row.val * w.columns + c.val)).toNat = row.val * w.columns + c.val ∧
    (UInt32.ofNat (row.val * w.columns)).toNat + c.val = row.val * w.columns + c.val ∧
    shape.tokens < 2147483648 := by
  have rows := row.isLt
  have cols := c.isLt
  have product : row.val * w.columns + c.val < 4294967296 := by
    cases shape <;> cases w <;> simp only [GraphStage.Shape.tokens, TrainingCE.Width.columns] at * <;> omega
  simp only [UInt32.toNat_ofNat', Nat.mod_eq_of_lt product, Nat.mod_eq_of_lt
    (show row.val * w.columns < 4294967296 by omega), true_and]
  cases shape <;> decide

#axiomsClean compiled_ce_scanned_leaders
#axiomsClean compiled_ce_row_offsets

set_option maxRecDepth 4096 in
/-- The two compiled cache publications supply the FP32 probability
and the E5M2 byte observed by the later gradient stores. -/
theorem compiled_ce_cache_values {w : TrainingCE.Width} {ptx : String} {mode : TrainingCE.Softcap}
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (row : TrainingCE.Row w)
    (artifact : TrainingCEPublishBinding.Certificate ptx row.predictions mode)
    (tables codes : List CudaStores.Store) (prior : CudaStores.Memory) (code : UInt8)
    (accepted : mode.accepts p = true := by rfl) :
    let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
    let ready := (TrainingCE.stageRow op p row tables codes prior).2
    let input := TrainingCEPublishBinding.Inputs.forPublication p row.weights row.prefixTarget.isSome
      (TrainingCE.stagedWarpLeaders w before) (TrainingCE.readSharedTable before code).logit
      (TrainingCE.readGradientFactor before)
    artifact.writes.eval op input = ((TrainingCE.readSharedProbability ready code).probability,
      ready (TrainingCE.baseGradientAddress + code.toNat)) := by
  let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
  let leaders := TrainingCE.stagedWarpLeaders w before
  let logit := (TrainingCE.readSharedTable before code).logit
  let factor := TrainingCE.readGradientFactor before
  have model := TrainingCE.staged_row_publication op p row tables codes prior code
  have decoded := TrainingCEPublishBinding.publication_exact op p row artifact leaders logit factor accepted
  refine decoded.trans (Prod.ext ?_ ?_)
  · exact congrArg TrainingCE.ProbabilityEntry.probability model.2.1.symm
  · exact model.2.2.symm

#axiomsClean compiled_ce_cache_values

set_option maxRecDepth 4096 in
/-- Feed the leader words written by the decoded shuffle passes into the
publication instructions of the same compiled kernel. -/
theorem compiled_ce_reduced_cache_values {w : TrainingCE.Width} {ptx : String} {mode : TrainingCE.Softcap}
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (row : TrainingCE.Row w)
    (reduction : TrainingCEReductionBinding.Certificate ptx)
    (publication : TrainingCEPublishBinding.Certificate ptx row.predictions mode)
    (tables codes : List CudaStores.Store) (prior : CudaStores.Memory) (code : UInt8)
    (accepted : mode.accepts p = true := by rfl) :
    let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
    let ready := (TrainingCE.stageRow op p row tables codes prior).2
    let leaders := reduction.reduction.readLeaders (TrainingCE.stagedLaneSums w before)
    let input := TrainingCEPublishBinding.Inputs.forPublication p row.weights row.prefixTarget.isSome
      leaders (TrainingCE.readSharedTable before code).logit (TrainingCE.readGradientFactor before)
    publication.writes.eval op input = ((TrainingCE.readSharedProbability ready code).probability,
      ready (TrainingCE.baseGradientAddress + code.toNat)) := by
  simp only [TrainingCEReductionBinding.leaders_exact reduction]
  exact compiled_ce_cache_values op p row publication tables codes prior code accepted

#axiomsClean compiled_ce_reduced_cache_values

set_option maxRecDepth 4096 in
/-- The decoded loss reads the staged sigmoid of each live target,
uses the compiled warp leaders, and writes the reference FP32 row loss. -/
theorem compiled_ce_row_loss {w : TrainingCE.Width} {ptx : String} {mode : TrainingCE.Softcap}
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (row : TrainingCE.Row w)
    (reduction : TrainingCEReductionBinding.Certificate ptx)
    (artifact : TrainingCELossBinding.Certificate w ptx row.predictions mode)
    (tables : List CudaStores.Store) (initialized : tables.Perm (TrainingCE.tableStores op p))
    (codes : List CudaStores.Store) (staged : codes.Perm (TrainingCE.codeStores row))
    (prior : CudaStores.Memory) (accepted : mode.accepts p = true := by rfl) :
    let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
    let ready := (TrainingCE.stageRow op p row tables codes prior).2
    let leaders := reduction.reduction.readLeaders (TrainingCE.stagedLaneSums w before)
    let sigma := fun (c : TrainingCE.Column w) =>
      (TrainingCE.readSharedProbability ready (ready (TrainingCE.codeAddress + c.val))).sigma
    (TrainingCELossBinding.decode w ptx (TrainingCELossBinding.liveSlots row)
        row.prefixTarget.isSome true mode).map
      (Option.map (TrainingCEPublishBinding.Expr.eval op (TrainingCELossBinding.inputs p row leaders sigma))) =
      some (some (TrainingCE.reference w op p row).loss) := by
  let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
  let ready := (TrainingCE.stageRow op p row tables codes prior).2
  let sigma := fun (c : TrainingCE.Column w) =>
    (TrainingCE.readSharedProbability ready (ready (TrainingCE.codeAddress + c.val))).sigma
  simp only [TrainingCEReductionBinding.leaders_exact reduction]
  change (TrainingCELossBinding.decode w ptx (TrainingCELossBinding.liveSlots row)
      row.prefixTarget.isSome true mode).map
    (Option.map (TrainingCEPublishBinding.Expr.eval op
      (TrainingCELossBinding.inputs p row (TrainingCE.stagedWarpLeaders w before) sigma))) = _
  refine (TrainingCELossBinding.loss_exact op p row artifact
    (TrainingCE.stagedWarpLeaders w before) sigma accepted).trans ?_
  have model := congrArg TrainingCE.Output.loss
    (TrainingCE.staged_loss_gradient_equal op p row tables initialized codes staged prior)
  exact congrArg (fun loss => some (some loss)) model

#axiomsClean compiled_ce_row_loss

set_option maxRecDepth 4096 in
/-- Sparse stores read the staged sigmoid and probability for the target
column. Their lane, slot and E5M2 byte agree with the staged gradient. -/
theorem compiled_ce_sparse_write {w : TrainingCE.Width} {ptx : String} {mode : TrainingCE.Softcap}
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (row : TrainingCE.Row w)
    (artifact : TrainingCESparseBinding.Certificate w ptx row.predictions mode)
    (tables codes : List CudaStores.Store) (prior : CudaStores.Memory)
    (lane : TrainingCE.Lane) (c : TrainingCE.Column w) (slot : Nat) (byte : UInt8) :
    let ready := (TrainingCE.stageRow op p row tables codes prior).2
    let entry := fun (column : TrainingCE.Column w) =>
      TrainingCE.readSharedProbability ready (ready (TrainingCE.codeAddress + column.val))
    (TrainingCESparseBinding.decodedWrites ptx op p row lane
      (TrainingCE.readGradientFactor ready) entry mode).any (fun writes => writes.contains (c, slot, byte)) = true ↔
      TrainingCE.columnOwner c = lane ∧
        ((TrainingCE.executeStagedStores op p row tables codes prior).2 c).correction = some (slot, byte) := by
  exact TrainingCESparseBinding.decoded_write_exact op p row artifact lane _ _ c slot byte

#axiomsClean compiled_ce_sparse_write

/-- The packed stores issued by the compiled loop implement the base
stores of the staged gradient model for all input bit patterns. -/
theorem compiled_ce_base_chunk {w : TrainingCE.Width} {ptx : String}
    (artifact : TrainingCEGatherBinding.GeometryCertificate w ptx)
    (lane : TrainingCE.Lane) (trace : List TrainingCEGatherBinding.Placement)
    (ran : TrainingCEGatherBinding.decodedPlacements w ptx lane = some trace)
    (placed : TrainingCEGatherBinding.Placement) (issued : placed ∈ trace)
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (row : TrainingCE.Row w)
    (tables codes : List CudaStores.Store) (priorShared priorOutput : CudaStores.Memory)
    (byte : TrainingCE.Byte) :
    let checked := artifact.checkedPayload lane trace ran placed issued
    let ready := (TrainingCE.stageRow op p row tables codes priorShared).2
    let column : TrainingCE.Column w := ⟨placed.start + byte.val,
      Nat.lt_of_lt_of_le (Nat.add_lt_add_left byte.isLt _) checked.2⟩
    CudaStores.commitAll (placed.fragment.stores ready (TrainingCE.codeAddress + placed.start)
      placed.start TrainingCE.baseGradientAddress) priorOutput column.val =
      ((TrainingCE.executeStagedStores op p row tables codes priorShared).2 column).base := by
  simpa only [TrainingCE.executeStagedStores, TrainingCE.packed_shared_code_exact, Nat.add_assoc] using
    TrainingCEGatherBinding.packed_store_exact artifact.packed placed.fragment
      (artifact.checkedPayload lane trace ran placed issued).1
      (TrainingCE.stageRow op p row tables codes priorShared).2 priorOutput
      (TrainingCE.codeAddress + placed.start) placed.start TrainingCE.baseGradientAddress byte

#axiomsClean compiled_ce_base_chunk

set_option maxRecDepth 4096 in
/-- Compose the numerical publication and the packed readback of the
same compiled kernel at each byte that the loop writes. -/
theorem compiled_ce_published_base_chunk {w : TrainingCE.Width} {ptx : String} {mode : TrainingCE.Softcap}
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (row : TrainingCE.Row w)
    (publication : TrainingCEPublishBinding.Certificate ptx row.predictions mode)
    (artifact : TrainingCEGatherBinding.GeometryCertificate w ptx)
    (lane : TrainingCE.Lane) (trace : List TrainingCEGatherBinding.Placement)
    (ran : TrainingCEGatherBinding.decodedPlacements w ptx lane = some trace)
    (placed : TrainingCEGatherBinding.Placement) (issued : placed ∈ trace)
    (tables codes : List CudaStores.Store) (priorShared priorOutput : CudaStores.Memory)
    (byte : TrainingCE.Byte) (accepted : mode.accepts p = true := by rfl) :
    let before := CudaStores.commitAll codes (CudaStores.commitAll tables priorShared)
    let ready := (TrainingCE.stageRow op p row tables codes priorShared).2
    let code := TrainingCE.readPackedCode ready (TrainingCE.codeAddress + (placed.start + byte.val))
    let input := TrainingCEPublishBinding.Inputs.forPublication p row.weights row.prefixTarget.isSome
      (TrainingCE.stagedWarpLeaders w before) (TrainingCE.readSharedTable before code).logit
      (TrainingCE.readGradientFactor before)
    CudaStores.commitAll (placed.fragment.stores ready (TrainingCE.codeAddress + placed.start)
      placed.start TrainingCE.baseGradientAddress) priorOutput (placed.start + byte.val) =
      (publication.writes.eval op input).2 := by
  let ready := (TrainingCE.stageRow op p row tables codes priorShared).2
  let code := TrainingCE.readPackedCode ready (TrainingCE.codeAddress + (placed.start + byte.val))
  have gathered := compiled_ce_base_chunk artifact lane trace ran placed issued op p row
    tables codes priorShared priorOutput byte
  have written := compiled_ce_cache_values op p row publication tables codes priorShared code accepted
  dsimp only at gathered ⊢
  rw [gathered]
  change ready (TrainingCE.baseGradientAddress + code.toNat) = _
  exact (congrArg Prod.snd written).symm

#axiomsClean compiled_ce_published_base_chunk

/-- Every vocabulary column is covered by the packed store issued by its
owner in an accepted PTX loop. The decoded schedule is checked for all lanes. -/
theorem compiled_ce_base_grid {w : TrainingCE.Width} {ptx : String}
    (artifact : TrainingCEGatherBinding.GeometryCertificate w ptx) (c : TrainingCE.Column w) :
    let lane := TrainingCE.columnOwner c
    let starts := TrainingCEGatherBinding.expectedStarts w lane
    TrainingCEGatherBinding.decodedStarts w ptx lane = some starts ∧
      ∃ (chunk : Fin w.chunks) (byte : TrainingCE.Byte),
        TrainingCE.column chunk lane 0 ∈ starts ∧ TrainingCE.column chunk lane byte = c.val := by
  obtain ⟨chunk, lane, byte, atColumn⟩ := TrainingCE.column_covered w c
  dsimp only
  rw [TrainingCE.column_owner_exact c chunk lane byte atColumn]
  refine ⟨artifact.placements lane, chunk, byte, ?_, atColumn⟩
  apply List.mem_filter.mpr
  refine ⟨List.mem_map.mpr ⟨chunk.val, List.mem_range.mpr chunk.isLt, rfl⟩, ?_⟩
  have start : TrainingCE.column chunk lane 0 ≤ c.val := by
    simp only [TrainingCE.column, Fin.val_zero, Nat.add_zero] at *
    omega
  exact decide_eq_true (Nat.lt_of_le_of_lt start c.isLt)

#axiomsClean compiled_ce_base_grid

set_option maxRecDepth 4096 in
def compiledCEGradientFromCache {w : TrainingCE.Width} (ptx : String) (op : TrainingCE.Native)
    (p : TrainingCE.Parameters) (row : TrainingCE.Row w) (ready : CudaStores.Memory)
    (c : TrainingCE.Column w) (mode : TrainingCE.Softcap) :
    Option UInt8 := do
  let entry := fun (column : TrainingCE.Column w) =>
    TrainingCE.readSharedProbability ready (ready (TrainingCE.codeAddress + column.val))
  let correction ← TrainingCESparseBinding.decodedCorrection ptx op p row
    (TrainingCE.readGradientFactor ready) entry c mode
  let trace ← TrainingCEGatherBinding.decodedPlacements w ptx (TrainingCE.columnOwner c)
  let place ← trace.find? fun place => decide (place.start ≤ c.val ∧ c.val < place.start + 8)
  let base := place.fragment.stores ready (TrainingCE.codeAddress + place.start)
    place.start TrainingCE.baseGradientAddress
  return correction.elim (CudaStores.commitAll base (fun _ => 0) c.val) Prod.snd

def compiledCEGradient {w : TrainingCE.Width} (ptx : String) (op : TrainingCE.Native)
    (p : TrainingCE.Parameters) (row : TrainingCE.Row w) (tables codes : List CudaStores.Store)
    (prior : CudaStores.Memory) (c : TrainingCE.Column w) (mode : TrainingCE.Softcap := .runtime) :
    Option UInt8 :=
  compiledCEGradientFromCache ptx op p row (TrainingCE.stageRow op p row tables codes prior).2 c mode

set_option maxRecDepth 4096 in
/-- Read the compiled E5M2 row at any column. Its owner writes one base
byte and then applies the first eligible correction for that column. -/
theorem compiled_ce_row_gradient {w : TrainingCE.Width} {ptx : String} {mode : TrainingCE.Softcap}
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (row : TrainingCE.Row w)
    (base : TrainingCEGatherBinding.GeometryCertificate w ptx)
    (patch : TrainingCESparseBinding.Certificate w ptx row.predictions mode)
    (tables codes : List CudaStores.Store) (initialized : tables.Perm (TrainingCE.tableStores op p))
    (staged : codes.Perm (TrainingCE.codeStores row)) (prior : CudaStores.Memory) (c : TrainingCE.Column w) :
    compiledCEGradient ptx op p row tables codes prior c mode =
      some ((TrainingCE.reference w op p row).gradient c) := by
  let ready := (TrainingCE.stageRow op p row tables codes prior).2
  let owner := TrainingCE.columnOwner c
  let writes := (TrainingCE.executeStagedStores op p row tables codes prior).2
  have correction := TrainingCESparseBinding.decoded_correction_exact op p row patch
    (TrainingCE.readGradientFactor ready)
    (fun column => TrainingCE.readSharedProbability ready (ready (TrainingCE.codeAddress + column.val))) c
  change _ = some (writes c).correction at correction
  obtain ⟨starts, chunk, byte, covered, address⟩ := compiled_ce_base_grid base c
  change TrainingCEGatherBinding.decodedStarts w ptx owner = _ at starts
  obtain ⟨trace, ran⟩ : ∃ trace, TrainingCEGatherBinding.decodedPlacements w ptx owner = some trace := by
    cases h : TrainingCEGatherBinding.decodedPlacements w ptx owner with
    | none => simp [TrainingCEGatherBinding.decodedStarts, h] at starts
    | some trace => exact ⟨trace, rfl⟩
  simp only [TrainingCEGatherBinding.decodedStarts, ran, Option.map_some, Option.some.injEq] at starts
  have somePlace : (trace.find? fun place => decide (place.start ≤ c.val ∧ c.val < place.start + 8)).isSome := by
    rw [List.find?_isSome]
    rw [← starts] at covered
    obtain ⟨place, issued, same⟩ := List.mem_map.mp covered
    refine ⟨place, issued, decide_eq_true ?_⟩
    have byteBound := byte.isLt
    simp only [TrainingCE.column, Fin.val_zero, Nat.add_zero] at same address
    omega
  obtain ⟨place, found⟩ := Option.isSome_iff_exists.mp somePlace
  have issued := List.mem_of_find?_eq_some found
  have covers := List.find?_some (p := fun (place : TrainingCEGatherBinding.Placement) =>
    decide (place.start ≤ c.val ∧ c.val < place.start + 8)) found
  have within : place.start ≤ c.val ∧ c.val < place.start + 8 := of_decide_eq_true covers
  let offset : TrainingCE.Byte := ⟨c.val - place.start, by omega⟩
  have sameAddress : place.start + offset.val = c.val := by
    change place.start + (c.val - place.start) = c.val
    omega
  have byte_eq := compiled_ce_base_chunk base owner trace ran place issued op p row tables codes prior
    (fun _ => 0) offset
  have sameColumn : (⟨place.start + offset.val,
      Nat.lt_of_lt_of_le (Nat.add_lt_add_left offset.isLt _) (base.checkedPayload owner trace ran place issued).2⟩ :
      TrainingCE.Column w) = c := Fin.ext sameAddress
  change CudaStores.commitAll (place.fragment.stores ready (TrainingCE.codeAddress + place.start)
    place.start TrainingCE.baseGradientAddress) _ (place.start + offset.val) =
    (writes ⟨_, _⟩).base at byte_eq
  rw [sameColumn] at byte_eq
  rw [sameAddress] at byte_eq
  have value := congrArg (fun out : TrainingCE.Output w => out.gradient c)
    (TrainingCE.staged_loss_gradient_equal op p row tables initialized codes staged prior)
  change (writes c).correction.elim (writes c).base Prod.snd = _ at value
  unfold compiledCEGradient compiledCEGradientFromCache
  change (do
    let patch ← TrainingCESparseBinding.decodedCorrection ptx op p row
      (TrainingCE.readGradientFactor ready)
      (fun column => TrainingCE.readSharedProbability ready (ready (TrainingCE.codeAddress + column.val))) c mode
    let trace ← TrainingCEGatherBinding.decodedPlacements w ptx owner
    let place ← trace.find? fun (place : TrainingCEGatherBinding.Placement) =>
      decide (place.start ≤ c.val ∧ c.val < place.start + 8)
    some (patch.elim (CudaStores.commitAll (place.fragment.stores ready
      (TrainingCE.codeAddress + place.start) place.start TrainingCE.baseGradientAddress) (fun _ => 0) c.val) Prod.snd)) = _
  rw [correction, ran]
  simp only [Bind.bind, Option.bind_some]
  rw [found]
  simp only [Option.bind_some, byte_eq, value]

#axiomsClean compiled_ce_row_gradient

/-- Read the graph's loss and gradient after shared input staging and
arbitrarily interleaved output stores that respect each lane's program order. -/
theorem linked_committed_training_ce (src : LinkedSources) (w : TrainingCE.Width)
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (row : TrainingCE.Row w)
    (tables : List CudaStores.Store) (initialized : tables.Perm (TrainingCE.tableStores op p))
    (codes : List CudaStores.Store) (staged : codes.Perm (TrainingCE.codeStores row))
    (priorShared : CudaStores.Memory) (issued : List CudaStores.Store)
    (ordered : TrainingCE.LaneStoreOrder
      (TrainingCE.executeStagedStores op p row tables codes priorShared).2 issued)
    (priorOutput : CudaStores.Memory) :
    let output := src.optimized.side.calls.trainingCE w op p row
    (TrainingCE.executeStagedStores op p row tables codes priorShared).1 = output.loss ∧
      (∀ c : TrainingCE.Column w, CudaStores.commitAll issued priorOutput c.val = output.gradient c) ∧
      (∀ address, w.columns ≤ address → CudaStores.commitAll issued priorOutput address = priorOutput address) := by
  have stagedEqual := linked_staged_training_ce src w op p row tables initialized codes staged priorShared
  refine ⟨congrArg TrainingCE.Output.loss stagedEqual, ?_, ?_⟩
  · intro c
    rw [TrainingCE.interleaved_gradient_writes_exact _ _ ordered, ← stagedEqual]
    rfl
  · exact TrainingCE.gradient_store_frame _ _ ordered priorOutput

#axiomsClean linked_committed_training_ce

structure CEGradientRow (w : TrainingCE.Width) (op : TrainingCE.Native) (p : TrainingCE.Parameters) where
  input : TrainingCE.Row w
  tables : List CudaStores.Store
  initialized : tables.Perm (TrainingCE.tableStores op p)
  codes : List CudaStores.Store
  staged : codes.Perm (TrainingCE.codeStores input)
  priorShared : CudaStores.Memory
  writes : List CudaStores.Store
  ordered : TrainingCE.LaneStoreOrder (TrainingCE.executeStagedStores op p input tables codes priorShared).2 writes
  priorGradient : CudaStores.Memory

def ceGradientBuffer {w : TrainingCE.Width} {tokens : Nat}
    (values : Fin tokens → TrainingCE.Column w → UInt8)
    (address : Nat) : UInt8 :=
  let token := address / w.columns
  if live : token < tokens then values ⟨token, live⟩
    ⟨address % w.columns, Nat.mod_lt _ (by cases w <;> decide)⟩
  else 0

private theorem ce_gradient_buffer_at {w : TrainingCE.Width} {tokens : Nat}
    (values : Fin tokens → TrainingCE.Column w → UInt8)
    (token : Fin tokens) (c : TrainingCE.Column w) :
    ceGradientBuffer values (token.val * w.columns + c.val) = values token c := by
  have positive : 0 < w.columns := by cases w <;> decide
  have row : (token.val * w.columns + c.val) / w.columns = token.val := by
    rw [Nat.mul_comm token.val, Nat.mul_add_div positive, Nat.div_eq_of_lt c.isLt, Nat.add_zero]
  have col : (token.val * w.columns + c.val) % w.columns = c.val := by
    rw [Nat.mul_comm token.val, Nat.mul_add_mod, Nat.mod_eq_of_lt c.isLt]
  simp only [ceGradientBuffer, row, dif_pos token.isLt, col]

def committedCEGradient {w : TrainingCE.Width} {op : TrainingCE.Native}
    {p : TrainingCE.Parameters} {tokens : Nat} (rows : Fin tokens → CEGradientRow w op p) : Nat → UInt8 :=
  ceGradientBuffer (fun token (c : TrainingCE.Column w) =>
    CudaStores.commitAll (rows token).writes (rows token).priorGradient c.val)

def referenceCEGradient (src : LinkedSources) {w : TrainingCE.Width} {op : TrainingCE.Native}
    {p : TrainingCE.Parameters} {tokens : Nat} (rows : Fin tokens → CEGradientRow w op p) : Nat → UInt8 :=
  ceGradientBuffer (fun token => (src.baseline.side.calls.trainingCE w op p (rows token).input).gradient)

theorem linked_ce_gradient_buffer (src : LinkedSources) {w : TrainingCE.Width} {op : TrainingCE.Native}
    {p : TrainingCE.Parameters} {tokens : Nat} (rows : Fin tokens → CEGradientRow w op p) :
    committedCEGradient rows = referenceCEGradient src rows := by
  unfold committedCEGradient referenceCEGradient
  congr 1
  funext token c
  let row := rows token
  have equal := linked_committed_training_ce src w op p row.input row.tables row.initialized
    row.codes row.staged row.priorShared row.writes row.ordered row.priorGradient
  rw [equal.2.1 c, linked_training_ce src w op p row.input]

def ceTransposeInputs (activation gradient : Nat → UInt8) : CETranspose.Operand → Nat → UInt8
  | .activation => activation
  | .gradient => gradient

private def compiledCEGradientBuffer {w : TrainingCE.Width} {tokens : Nat} (ptx : String)
    (op : TrainingCE.Native) (p : TrainingCE.Parameters) (rows : Fin tokens → TrainingCE.Row w)
    (mode : TrainingCE.Softcap) :
    Nat → UInt8 :=
  ceGradientBuffer fun token c =>
    let row := rows token
    (compiledCEGradient ptx op p row (TrainingCE.tableStores op p) (TrainingCE.codeStores row)
      (fun _ => 0) c mode).getD 0

/-- Both head-backward operands follow from the checked CE and transpose
artifacts. The row-major gradient has one E5M2 byte per reference column. -/
theorem compiled_ce_backward_operands {cePTX gir transPTX : String}
    {w : TrainingCE.Width} {mode : TrainingCE.Softcap} (op : TrainingCE.Native) (p : TrainingCE.Parameters)
    (shape : CETranspose.Shape) (rows : Fin (shape.operand .gradient).rows → TrainingCE.Row w)
    (base : TrainingCEGatherBinding.GeometryCertificate w cePTX)
    (patch : ∀ token, TrainingCESparseBinding.Certificate w cePTX (rows token).predictions mode)
    (transpose : CETranspose.Certificate gir transPTX) (activation : Nat → UInt8) :
    let emitted := compiledCEGradientBuffer cePTX op p rows mode
    let spec := ceGradientBuffer (fun token => (TrainingCE.reference w op p (rows token)).gradient)
    emitted = spec ∧ CETranspose.paired shape (ceTransposeInputs activation emitted) =
      CETranspose.staged shape (ceTransposeInputs activation spec) := by
  dsimp only
  have every (token : Fin (shape.operand .gradient).rows) (c : TrainingCE.Column w) :
      (compiledCEGradient cePTX op p (rows token) (TrainingCE.tableStores op p)
        (TrainingCE.codeStores (rows token)) (fun _ => 0) c mode).getD 0 =
        (TrainingCE.reference w op p (rows token)).gradient c := by
    rw [compiled_ce_row_gradient op p (rows token) base (patch token)
      _ _ (.refl _) (.refl _)]
    rfl
  have bytes : compiledCEGradientBuffer cePTX op p rows mode =
      ceGradientBuffer (fun token => (TrainingCE.reference w op p (rows token)).gradient) := by
    unfold compiledCEGradientBuffer
    exact congrArg ceGradientBuffer (funext fun token => funext (every token))
  exact ⟨bytes, by rw [bytes, transpose.stores]⟩

#axiomsClean compiled_ce_backward_operands

/-- The head backward's input-gradient GEMM reads the committed row-major
gradient. Its weight-gradient GEMM reads the two fused transpose outputs. -/
theorem linked_ce_backward_operands (src : LinkedSources) {gir ptx : String}
    (artifact : CETranspose.Certificate gir ptx) (shape : CETranspose.Shape)
    {w : TrainingCE.Width} {op : TrainingCE.Native} {p : TrainingCE.Parameters}
    (rows : Fin (shape.operand .gradient).rows → CEGradientRow w op p)
    (activation : Nat → UInt8) :
    committedCEGradient rows = referenceCEGradient src rows ∧
      CETranspose.paired shape (ceTransposeInputs activation (committedCEGradient rows)) =
        CETranspose.staged shape (ceTransposeInputs activation (referenceCEGradient src rows)) := by
  have gradient := linked_ce_gradient_buffer src rows
  exact ⟨gradient, by rw [gradient, artifact.stores]⟩

#axiomsClean linked_ce_backward_operands

/-- The transposed E5M2 operand presented to the weight-gradient GEMM
reads the bytes committed by the training CE lanes, at the reference
class-major offsets. The activation buffer is the other transpose input. -/
theorem linked_ce_weight_gradient_bytes (src : LinkedSources) {gir ptx : String}
    (artifact : CETranspose.Certificate gir ptx)
    (shape : CETranspose.Shape) (w : TrainingCE.Width)
    (columns : (shape.operand .gradient).cols = w.columns)
    (op : TrainingCE.Native) (p : TrainingCE.Parameters)
    (rows : Fin (shape.operand .gradient).rows → CEGradientRow w op p)
    (activation : Nat → UInt8) (token : Fin (shape.operand .gradient).rows) (c : TrainingCE.Column w) :
    let channel : Fin (shape.operand .gradient).cols := ⟨c.val, columns.symm ▸ c.isLt⟩
    CETranspose.paired shape (ceTransposeInputs activation (committedCEGradient rows))
      ⟨.gradient, ByteTranspose.point _ token channel⟩ =
      some (.gradient, c.val * (shape.operand .gradient).rows + token.val,
        (src.baseline.side.calls.trainingCE w op p (rows token).input).gradient c) := by
  dsimp only
  rw [(linked_ce_backward_operands src artifact shape rows activation).2, CETranspose.staged_operand]
  simp only [ceTransposeInputs, columns, referenceCEGradient, ce_gradient_buffer_at]

#axiomsClean linked_ce_weight_gradient_bytes

def sampledCEWidth : SampledRows.Count → TrainingCE.Width
  | .first => .sampled10
  | .second => .sampled14
  | .third => .sampled24

/-- With the same backend GEMM, the compact weight gradient is fed through
the saved positions to the same dense BF16 stores at every class. The
GEMM is a common operation on the proved operand bytes. -/
theorem linked_sampled_ce_dense_gradient (src : LinkedSources) (c : SampledRows.Count)
    {transposeGir transposePtx : String} (transpose : CETranspose.Certificate transposeGir transposePtx)
    {gatherGir gatherPtx : String} (gather : SampledInverse.Certificate c gatherGir gatherPtx)
    {code : Densify.Artifacts} (densify : Densify.Certificate c code)
    (shape : CETranspose.Shape) (op : TrainingCE.Native) (p : TrainingCE.Parameters)
    (rows : Fin (shape.operand .gradient).rows → CEGradientRow (sampledCEWidth c) op p)
    (activation : Nat → UInt8)
    (weightGradient : (CETranspose.Cell shape → Option (CETranspose.Operand × Nat × UInt8)) → Nat → UInt16)
    (mark : SampledRows.ClassID → Bool) (size : (Densify.flatNonzero mark).length = c.rows) :
    let candidateOperand := CETranspose.paired shape (ceTransposeInputs activation (committedCEGradient rows))
    let referenceOperand := CETranspose.staged shape (ceTransposeInputs activation (referenceCEGradient src rows))
    let uploaded := SampledInverse.Candidates.ofBitmap c mark size
    ∀ point, Densify.emitted (weightGradient candidateOperand) (SampledInverse.blockedInverse uploaded) point =
      Densify.reference (weightGradient referenceOperand) (Densify.inverseMap uploaded.ids) point := by
  dsimp only
  intro point
  rw [(linked_ce_backward_operands src transpose shape rows activation).2,
    gather.inverseMap, densify.stores]

#axiomsClean linked_sampled_ce_dense_gradient

/-- The sampled training path starts at the decoded CE writes and ends
at the dense BF16 weight-gradient stores. The GEMM consumes the same
transposed bytes on both sides. -/
theorem compiled_sampled_ce_dense_gradient (c : SampledRows.Count)
    {cePTX transposeGir transposePTX gatherGir gatherPTX : String}
    {mode : TrainingCE.Softcap}
    (transpose : CETranspose.Certificate transposeGir transposePTX)
    (gather : SampledInverse.Certificate c gatherGir gatherPTX)
    {code : Densify.Artifacts} (densify : Densify.Certificate c code)
    (shape : CETranspose.Shape) (op : TrainingCE.Native) (p : TrainingCE.Parameters)
    (rows : Fin (shape.operand .gradient).rows → TrainingCE.Row (sampledCEWidth c))
    (base : TrainingCEGatherBinding.GeometryCertificate (sampledCEWidth c) cePTX)
    (patch : ∀ token, TrainingCESparseBinding.Certificate (sampledCEWidth c) cePTX (rows token).predictions mode)
    (activation : Nat → UInt8)
    (weightGradient : (CETranspose.Cell shape → Option (CETranspose.Operand × Nat × UInt8)) → Nat → UInt16)
    (mark : SampledRows.ClassID → Bool) (size : (Densify.flatNonzero mark).length = c.rows) :
    let candidate := compiledCEGradientBuffer cePTX op p rows mode
    let spec := ceGradientBuffer (fun token => (TrainingCE.reference (sampledCEWidth c) op p (rows token)).gradient)
    let candidateOperand := CETranspose.paired shape (ceTransposeInputs activation candidate)
    let referenceOperand := CETranspose.staged shape (ceTransposeInputs activation spec)
    let uploaded := SampledInverse.Candidates.ofBitmap c mark size
    ∀ point, Densify.emitted (weightGradient candidateOperand) (SampledInverse.blockedInverse uploaded) point =
      Densify.reference (weightGradient referenceOperand) (Densify.inverseMap uploaded.ids) point := by
  dsimp only
  intro point
  rw [(compiled_ce_backward_operands op p shape rows base patch transpose activation).2,
    gather.inverseMap, densify.stores]

#axiomsClean compiled_sampled_ce_dense_gradient

/-- The checked optimizer sends only the peer packet first. The local
scatter obtains its words from the live owner slice, with the address
computed by the same lower-bound cut as the host. Its last-read clears
restore the reference gradient buffer before the next graph backward. -/
theorem compiled_value_adam_event (checked : Certificate)
    {packCode : SparsePackets.Artifacts} (pack : SparsePackets.Certificate packCode)
    {adamCode : ValueAdam.Artifacts} (adam : ValueAdam.Certificate adamCode)
    (want : NgramSlots.Want) (sourceRows first rows : Nat) (fits : sourceRows < 2 ^ 31)
    (live : ∀ i, i < want.size → want.read i < sourceRows)
    (unique : Function.Injective (fun i : Fin want.size => want.read i.val))
    (gradient : SparsePackets.FP16Rows) (asked : Array Int32)
    (peer : Fin asked.size → SparsePackets.Channel → UInt16)
    (order : (p : ValueAdam.Pull) → ValueAdam.Schedule p) (ar : ValueAdam.Arithmetic)
    (step : Nat) (scalars : ValueAdam.Scalars) (state : ValueAdam.Memory) :
    let s := RowExchange.valueSource want sourceRows first rows fits live unique gradient asked
    let candidate := s.deferredPull peer
    let reference := s.stagedPull peer
    let launch := ValueAdam.configure step scalars
    let c := checked.sources.optimized.side.calls
    let r := checked.sources.baseline.side.calls
    (∀ slot : Fin s.input.cut.own, ∀ col : SparsePackets.Channel,
      s.localRow slot * 768 + col.val < rows * 768 ∧
      first * 768 + (s.localRow slot * 768 + col.val) = s.globalRow slot * 768 + col.val) ∧
    (c.sparsePackets.value s.input).wire = (r.sparsePackets.value s.input).wire ∧
    (SparsePackets.valueOwnedCleared s.input (SparsePackets.valuePeersCleared s.input),
      ValueAdam.event (c.sparseLanding.valueNormalize state.prior) candidate (c.sparseLanding.adamEntries candidate.ids)
        (order candidate) ar launch state) =
    (r.sparsePackets.clear s.input,
      ValueAdam.event (r.sparseLanding.valueNormalize state.prior) reference (r.sparseLanding.adamEntries reference.ids)
        (order reference) ar launch state) := by
  intro s candidate reference launch c r
  exact ⟨s.address_exact, congrArg SparsePackets.Packets.wire (pack.deferredValue s.input),
    adam.sourceUpdate s peer order ar launch state⟩

/-- Several graph backwards accumulate into the same value table. The
following exchange and Adam step consume that accumulated table, and its
owned and remote slots are ready for reuse after the update. -/
theorem compiled_value_backward_to_adam (checked : Certificate)
    {backCode : ValueEmbedAccum.Artifacts} (back : ValueEmbedAccum.Certificate backCode)
    {packCode : SparsePackets.Artifacts} (pack : SparsePackets.Certificate packCode)
    {adamCode : ValueAdam.Artifacts} (adam : ValueAdam.Certificate adamCode)
    (backs : List ValueEmbedAccum.Snapshot) (initial : SparsePackets.FP16Rows)
    (want : NgramSlots.Want) (sourceRows first rows : Nat) (fits : sourceRows < 2 ^ 31)
    (live : ∀ i, i < want.size → want.read i < sourceRows)
    (unique : Function.Injective (fun i : Fin want.size => want.read i.val))
    (asked : Array Int32) (peer : Fin asked.size → SparsePackets.Channel → UInt16)
    (order : (p : ValueAdam.Pull) → ValueAdam.Schedule p) (ar : ValueAdam.Arithmetic)
    (step : Nat) (scalars : ValueAdam.Scalars) (state : ValueAdam.Memory) :
    let source := fun kernel => RowExchange.valueSource want sourceRows first rows fits live unique
      (ValueEmbedAccum.accumulateSteps kernel initial backs) asked
    let fast := source ValueEmbedAccum.fused
    let base := source ValueEmbedAccum.staged
    let candidate := fast.deferredPull peer
    let reference := base.stagedPull peer
    let launch := ValueAdam.configure step scalars
    (SparsePackets.valueDeferred fast.input).wire = (SparsePackets.valueStaged base.input).wire ∧
    (SparsePackets.valueOwnedCleared fast.input (SparsePackets.valuePeersCleared fast.input),
      ValueAdam.event (ValueAdam.clearSelected state.prior) candidate (AdamEntries.fused candidate.ids)
        (order candidate) ar launch state) =
    (SparsePackets.cleared base.input,
      ValueAdam.event ValueAdam.dense reference (AdamEntries.staged reference.ids)
        (order reference) ar launch state) := by
  dsimp only
  rw [back.equivalent]
  exact (compiled_value_adam_event checked pack adam want sourceRows first rows fits live unique
    (ValueEmbedAccum.accumulateSteps ValueEmbedAccum.staged initial backs) asked peer order ar step scalars state).2

/-- The checked trainer publishes one of the two schedule words at each
Adam event. Reset restarts the prefix even with existing backing storage. -/
theorem compiled_moment_replay_since_reset (checked : Certificate) {a : MomentReplay.Artifacts}
    (artifact : MomentReplay.Certificate a) (steps : List Nat) (previous : Nat → NgramAdam.F32)
    (s : MomentReplay.Input) :
    let current := {s with event := steps.length, beta := NgramAdam.publishSteps 0 previous steps}
    MomentReplay.groupedZero current = checked.sources.optimized.side.calls.sparseLanding.moments current ∧
    MomentReplay.groupedZero current = checked.sources.baseline.side.calls.sparseLanding.moments current := by
  intro current
  have history := NgramAdam.published_since_reset steps previous
  have exactWords := artifact.zeroOutputs current history
  constructor
  · change MomentReplay.groupedZero current = MomentReplay.selected current
    rw [MomentReplay.selected_equal]
    exact exactWords
  · exact exactWords

/-- The installed owner-gated kernel runs on the history published since
reset, and commits the same weights, second moments and stamps. -/
theorem compiled_ngram_update_since_reset (checked : Certificate) {a : NgramAdam.Artifacts}
    (artifact : NgramAdam.Certificate a) (steps : List Nat) (step : Nat) (p : NgramAdam.Pull)
    (order : NgramAdam.Schedule p) (native : NgramAdam.Native) (scalars : NgramAdam.Scalars)
    (previous : NgramAdam.Memory) :
    let state := {previous with betaHistory := NgramAdam.publishSteps 0 previous.betaHistory steps}
    let launch := NgramAdam.configure step {scalars with events := steps.length}
    NgramAdam.eventPreparedWith NgramAdam.rowUpdateSkippingZero p
      (checked.sources.optimized.side.calls.sparseLanding.ngramPrepare p order) order native launch state =
    NgramAdam.eventPreparedWith checked.sources.baseline.side.calls.sparseLanding.ngramUpdate p
      (checked.sources.baseline.side.calls.sparseLanding.ngramPrepare p order) order native launch state := by
  intro state launch
  have history : MomentReplay.PositiveHistory state.betaHistory (launch.event - 1) := by
    change MomentReplay.PositiveHistory (NgramAdam.publishSteps 0 previous.betaHistory steps)
      ((steps.length + 1) - 1)
    rw [Nat.add_sub_cancel]
    exact NgramAdam.published_since_reset steps previous.betaHistory
  exact artifact.updateZero p order native launch state history

/-- The compiled full-row kernels return the values substituted at the
driver's n-gram ports, with the pinned SM90a arithmetic and both stores. -/
theorem compiled_ngram_stores (src : LinkedSources) {artifacts : NgramEmbedBinding.Artifacts}
    (checked : NgramEmbedBinding.Certificate artifacts) {tokens : Nat} (s : Ngram.Inputs)
    (shape : NgramEmbedBinding.Supported tokens) (mask : s.signMask = 8191) (t : Fin tokens) (c : Fin Ngram.D) :
    let offset := Ngram.index t.val c.val
    let grad := src.optimized.side.calls.backward true tokens s .packed t c
    NgramEmbedBinding.forward .fullRows artifacts.fullForward s t c =
      some (offset, src.optimized.side.calls.lookup true tokens s .leftFma t c) ∧
    NgramEmbedBinding.backward .fullRows artifacts.fullBackward s t c =
      some ((offset, grad.1), (tokens * Ngram.D + offset, grad.2)) := by
  intro offset grad
  obtain ⟨row, channel⟩ := Ngram.owner_reads t c
  have owner := Ngram.full_lane_owner (Ngram.tileOwner t c)
  have values := NgramEmbedPTX.emitted_forward s (Ngram.FullLane.ofOwner (Ngram.tileOwner t c)) mask
  have grads := NgramEmbedPTX.emitted_backward s (Ngram.FullLane.ofOwner (Ngram.tileOwner t c)) mask
  constructor
  · change _ = some (offset, Ngram.fullForward s _ .leftFma)
    simp only [NgramEmbedBinding.forward, shape, mask, checked.fullForward, true_and, if_true, values,
      Ngram.fullForward, owner, row, channel, offset]
  · change _ = some ((offset, (Ngram.fullBackward s _ .packed).1),
      (tokens * Ngram.D + offset, (Ngram.fullBackward s _ .packed).2))
    simp only [NgramEmbedBinding.backward, shape, mask, checked.fullBackward, true_and, if_true, grads,
      Ngram.fullBackward, owner, row, channel, offset]

def stagedNgramInputs {s : GraphStage.Shape} (observation : GraphStage.Observation s)
    (bank : Ngram.Inputs) : Ngram.Inputs :=
  { bank with
    tokens := fun i => if h : i < GraphStage.extent s .inputs then
      (observation.graph .inputs ⟨i, h⟩).toUInt32 else 0
    slots := fun i => if h : i < GraphStage.extent s .slots then
      (observation.graph .slots ⟨i, h⟩).toNat else 0
    signMask := 8191 }

/-- The member-stage result supplies both int32 input views needed by the
fixed-profile embeddings. The cache allocation covers the staged slots. -/
theorem staged_ngram_replay {s : GraphStage.Shape} {a : NgramEmbedBinding.Artifacts}
    (artifact : NgramEmbedBinding.Certificate a) (observation : GraphStage.Observation s)
    (bank : Ngram.Inputs) (loadedRows cacheRows : Nat) (capacity : loadedRows ≤ cacheRows)
    (fits : loadedRows ≤ 2147483648)
    (loaded : ∀ i : Fin (GraphStage.extent s .slots), (observation.graph .slots i).toNat < loadedRows) :
    let input := stagedNgramInputs observation bank
    (∀ (t : Fin s.tokens) (c : Fin Ngram.D),
      NgramEmbedBinding.forward .fullRows a.fullForward input t c =
        NgramEmbedBinding.forward .flat a.flatForward input t c ∧
      NgramEmbedBinding.backward .fullRows a.fullBackward input t c =
        NgramEmbedBinding.backward .flat a.flatBackward input t c) ∧
    (∀ (forward : Bool) (lane : Ngram.FullLane s.tokens) (buffer : NgramEmbedPTX.InputBuffer) (word : Nat),
      (buffer, word) ∈ NgramEmbedPTX.reads forward input lane → word < buffer.words s.tokens cacheRows) ∧
    (∀ i < 2 * s.tokens, ∀ c : Fin Ngram.D,
      (Int32.ofNat (input.slots i)).toInt * 1536 + (c.val : Int) * 2 =
        (2 * Ngram.index (input.slots i) c.val : Nat)) := by
  intro input
  have supported : NgramEmbedBinding.Supported s.tokens := by cases s <;> decide
  have slots (i : Nat) (inside : i < 2 * s.tokens) : input.slots i < loadedRows := by
    have valid : i < GraphStage.extent s .slots := inside
    simpa only [input, stagedNgramInputs, dif_pos valid] using loaded ⟨i, valid⟩
  exact ⟨NgramEmbedBinding.replay artifact input supported rfl,
    fun _ _ => NgramEmbedPTX.reads_within _ input _ supported.2.2
      (fun i inside => Nat.lt_of_lt_of_le (slots i inside) capacity),
    fun i inside c => NgramEmbedPTX.signed_slot_address _ c.val (Nat.lt_of_lt_of_le (slots i inside) fits)⟩

def landedNgramBank (wanted : Nat → NgramSlots.Want) (banks : RowExchange.Banks) (rank : Fin 8)
    (previous : SparseLanding.Rows) (bank : Ngram.Inputs) : Ngram.Inputs :=
  let land := SparseLanding.directLand (RowExchange.landingInput wanted banks rank previous)
  { bank with cache := fun word =>
      Value.ofBits .bf16 ((land (word / Ngram.D) ⟨word % Ngram.D, Nat.mod_lt _ (by decide)⟩).getD 0).toNat }

/-- Interpret both embedding kernels on the physically landed cache.
Every slot's BF16 operand is the word of its staged global ID. -/
theorem landed_staged_ngram_replay {s : GraphStage.Shape} {a : NgramEmbedBinding.Artifacts}
    (artifact : NgramEmbedBinding.Certificate a) (observation : GraphStage.Observation s)
    (wanted : Nat → NgramSlots.Want) (banks : RowExchange.Banks) (rank : Fin 8)
    (previous : SparseLanding.Rows) (bank : Ngram.Inputs)
    (capacity : (wanted rank).size ≤ NgramCycleSchedule.defaultCacheRows 8)
    (queries : Fin (GraphStage.extent s .slots) → Nat)
    (routed : ∀ i, let slot := (observation.graph .slots i).toNat
      slot < (wanted rank).size ∧ (wanted rank).read slot = queries i) :
    let input := stagedNgramInputs observation (landedNgramBank wanted banks rank previous bank)
    (∀ (i : Fin (GraphStage.extent s .slots)) (c : Fin Ngram.D),
      input.cache (Ngram.index (input.slots i) c) = Value.ofBits .bf16 (RowExchange.globalWord banks (queries i) c).toNat) ∧
    (∀ (t : Fin s.tokens) (c : Fin Ngram.D),
      NgramEmbedBinding.forward .fullRows a.fullForward input t c =
        NgramEmbedBinding.forward .flat a.flatForward input t c ∧
      NgramEmbedBinding.backward .fullRows a.fullBackward input t c =
        NgramEmbedBinding.backward .flat a.flatBackward input t c) ∧
    (∀ (forward : Bool) (lane : Ngram.FullLane s.tokens) (buffer : NgramEmbedPTX.InputBuffer) (word : Nat),
      (buffer, word) ∈ NgramEmbedPTX.reads forward input lane →
        word < buffer.words s.tokens (NgramCycleSchedule.defaultCacheRows 8)) := by
  intro input
  have extent : (wanted rank).size ≤ 2147483648 := Nat.le_trans capacity (by decide)
  have replay := staged_ngram_replay artifact observation (landedNgramBank wanted banks rank previous bank)
    (wanted rank).size (NgramCycleSchedule.defaultCacheRows 8) capacity extent (fun i => (routed i).1)
  refine ⟨?_, replay.1, replay.2.1⟩
  intro i col
  let idx := (observation.graph .slots i).toNat
  have loaded := RowExchange.landed_cache_word wanted banks rank previous idx col (routed i).1
  have row : (wanted rank).read idx = queries i := (routed i).2
  have slot : input.slots i = idx := by simp only [input, stagedNgramInputs, dif_pos i.isLt, idx]
  rw [slot]
  have sameColumn : (⟨Ngram.index idx col % Ngram.D, Nat.mod_lt _ (by decide)⟩ : Fin Ngram.D) = col :=
    Fin.ext (Ngram.index_channel idx col col.isLt)
  change Value.ofBits .bf16
      ((SparseLanding.directLand (RowExchange.landingInput wanted banks rank previous)
        (Ngram.index idx col / Ngram.D) ⟨Ngram.index idx col % Ngram.D, Nat.mod_lt _ (by decide)⟩).getD 0).toNat = _
  rw [Ngram.index_token idx col col.isLt, sameColumn, loaded, row]
  rfl

/-- The gradient tape, per-slot FP16 accumulation and the reverse row
exchange supply the Adam call on each rank. -/
theorem compiled_ngram_cycle_update {ir ptx : String} (artifact : NgramAccumulation.Certificate ir ptx)
    (wanted : Nat → NgramSlots.Want) (previousGradients : Nat → SparsePackets.FP16Rows)
    (earlier : (rank : Nat) → List (NgramAccumulation.Step (wanted rank).size))
    (event : (rank : Nat) → NgramAccumulation.Step (wanted rank).size) (owner : Fin 8)
    (order : (pull : NgramAdam.Pull) → NgramAdam.Schedule pull)
    (native : NgramAdam.Native) (launch : NgramAdam.Launch) (previous : NgramAdam.Memory) :
    let tape := fun (rank : Nat) => (⟨RowExchange.localCut (wanted rank) rank, previousGradients rank,
      earlier rank, event rank⟩ : NgramAccumulation.Cycle)
    let optimized := RowGradientExchange.adamPull wanted (fun rank => NgramAccumulation.online (tape rank)) owner
    let reference := RowGradientExchange.adamPull wanted (fun rank => NgramAccumulation.delayed (tape rank)) owner
    HEq (NgramAdam.eventPrepared optimized (NgramAdam.fusedPrepare optimized (order optimized)) (order optimized)
        native launch previous)
      (NgramAdam.event reference (AdamEntries.staged reference.ids) (order reference) native launch previous) := by
  intro tape optimized reference
  have same : optimized = reference := by simp only [optimized, reference, artifact.cycle]
  rw [same]
  exact heq_of_eq (NgramAdam.prepared_event_equal reference (order reference) native launch previous)

theorem compiled_ngram_cycle_since_reset (checked : Certificate) {a : NgramAdam.Artifacts}
    (adam : NgramAdam.Certificate a) {ir ptx : String} (scatter : NgramAccumulation.Certificate ir ptx)
    (wanted : Nat → NgramSlots.Want) (previousGradients : Nat → SparsePackets.FP16Rows)
    (earlier : (rank : Nat) → List (NgramAccumulation.Step (wanted rank).size))
    (event : (rank : Nat) → NgramAccumulation.Step (wanted rank).size) (owner : Fin 8)
    (order : (pull : NgramAdam.Pull) → NgramAdam.Schedule pull) (native : NgramAdam.Native)
    (steps : List Nat) (step : Nat) (scalars : NgramAdam.Scalars) (previous : NgramAdam.Memory) :
    let tape := fun (rank : Nat) => (⟨RowExchange.localCut (wanted rank) rank, previousGradients rank,
      earlier rank, event rank⟩ : NgramAccumulation.Cycle)
    let optimized := RowGradientExchange.adamPull wanted (fun rank => NgramAccumulation.online (tape rank)) owner
    let reference := RowGradientExchange.adamPull wanted (fun rank => NgramAccumulation.delayed (tape rank)) owner
    let state := {previous with betaHistory := NgramAdam.publishSteps 0 previous.betaHistory steps}
    let launch := NgramAdam.configure step {scalars with events := steps.length}
    HEq (NgramAdam.eventPreparedWith NgramAdam.rowUpdateSkippingZero optimized
          (NgramAdam.fusedPrepare optimized (order optimized)) (order optimized) native launch state)
      (NgramAdam.event reference (AdamEntries.staged reference.ids) (order reference) native launch state) := by
  intro tape optimized reference state launch
  have same : optimized = reference := by simp only [optimized, reference, scatter.cycle]
  rw [same]
  exact heq_of_eq (compiled_ngram_update_since_reset checked adam steps step reference (order reference)
    native scalars previous)

/-- The graph returns the two contiguous sink planes. Scatter consumes
them in the same order as the pending slot tensor. -/
def sinkStep {tokens rows : Nat} (slots : Fin (2 * tokens) → Fin rows)
    (order : Fin rows → SparsePackets.Channel → NgramAdam.Order (2 * tokens))
    (backward : Fin tokens → SparsePackets.Channel → Option ((Nat × Value .bf16) × (Nat × Value .bf16))) :
    NgramAccumulation.Step rows where
  entries := 2 * tokens
  slot := slots
  order := order
  gradient word col :=
    if first : word.val < tokens then
      ((backward ⟨word.val, first⟩ col).map (·.1.2)).getD (Value.zero .bf16)
    else
      let token : Fin tokens := ⟨word.val - tokens, by have := word.isLt; omega⟩
      ((backward token col).map (·.2.2)).getD (Value.zero .bf16)

theorem staged_ngram_sink_step {s : GraphStage.Shape} {a : NgramEmbedBinding.Artifacts}
    (artifact : NgramEmbedBinding.Certificate a) (observation : GraphStage.Observation s)
    (bank : Ngram.Inputs) (rows : Nat)
    (loaded : ∀ i : Fin (2 * s.tokens), observation.pendingSlots i < rows)
    (order : Fin rows → SparsePackets.Channel → NgramAdam.Order (2 * s.tokens)) :
    let slots : Fin (2 * s.tokens) → Fin rows := fun i => ⟨observation.pendingSlots i, loaded i⟩
    let input := stagedNgramInputs observation bank
    sinkStep slots order (NgramEmbedBinding.backward .fullRows a.fullBackward input) =
      sinkStep slots order (NgramEmbedBinding.backward .flat a.flatBackward input) := by
  intro slots input
  have shape : NgramEmbedBinding.Supported s.tokens := by cases s <;> decide
  have emits : (NgramEmbedBinding.backward .fullRows a.fullBackward input :
      Fin s.tokens → SparsePackets.Channel → Option ((Nat × Value .bf16) × (Nat × Value .bf16))) =
      NgramEmbedBinding.backward .flat a.flatBackward input := by
    funext token col
    exact (NgramEmbedBinding.replay artifact input shape rfl token col).2
  rw [emits]

/-- The physical scratch starts with the reset stores. Any sequence of
cycle sizes that fits the allocation emits the reference packets and
restores the whole allocation for the next backward. -/
theorem compiled_scratch_reuse {a : NgramScratch.Artifacts} (artifact : NgramScratch.Certificate a)
    {capacity : Nat} (events : List (NgramScratch.Event capacity)) (before : SparsePackets.FP16Rows) :
    let scratch := NgramAccumulation.zero capacity before
    (NgramScratch.execute events scratch).1 = events.map NgramScratch.reference ∧
      NgramScratch.Clean capacity (NgramScratch.execute events scratch).2 :=
  artifact.cycles events _ (NgramScratch.reset_ready capacity before)

#axiomsClean execution_equal
#axiomsClean linked_run_equal
#axiomsClean training_end_to_end
#axiomsClean scheduled_graph_replay
#axiomsClean scheduled_lookup_lane
#axiomsClean scheduled_compiled_replay
#axiomsClean scheduled_stream_replay
#axiomsClean scheduled_packet_stream_replay
#axiomsClean memberCapturePacket
#axiomsClean scheduled_member_stream_replay
#axiomsClean hopper_training_stream_bound
#axiomsClean timed_cycle_capacity
#axiomsClean final_fineweb_validation_equal
#axiomsClean compiled_value_adam_event
#axiomsClean compiled_value_backward_to_adam
#axiomsClean compiled_moment_replay_since_reset
#axiomsClean compiled_ngram_update_since_reset
#axiomsClean compiled_ngram_stores
#axiomsClean staged_ngram_replay
#axiomsClean landed_staged_ngram_replay
#axiomsClean compiled_ngram_cycle_update
#axiomsClean compiled_ngram_cycle_since_reset
#axiomsClean staged_ngram_sink_step
#axiomsClean compiled_scratch_reuse
#print_fp_assumptions training_end_to_end
end VeriTile.Bench.Models.NanoGPT.Track1.Driver
