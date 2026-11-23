import models.NanoGPT.Track1.ScaleRefresh
import models.NanoGPT.Track1.NgramTiles
import models.NanoGPT.Track1.NgramRowHash
import models.NanoGPT.Track1.FineWebTraining
import models.NanoGPT.Track1.HostSchedule
import models.NanoGPT.Track1.GraphStageExecution
import models.NanoGPT.Track1.PinnedBatch
import models.NanoGPT.Track1.QKVScale
import models.NanoGPT.Track1.QKVScanSchedule
import models.NanoGPT.Track1.MLPWeightPack
import models.NanoGPT.Track1.LMHeadBinding
import models.NanoGPT.Track1.SampledGather
import models.NanoGPT.Track1.SampledInverse
import models.NanoGPT.Track1.RopeShift
import models.NanoGPT.Track1.RopeBackward
import models.NanoGPT.Track1.YarnTable
import models.NanoGPT.Track1.YarnStationary
import models.NanoGPT.Track1.CETranspose
import models.NanoGPT.Track1.TrainingCE
import models.NanoGPT.Track1.ValidationCE
import models.NanoGPT.Track1.Bf16Transpose
import models.NanoGPT.Track1.TailLerp
import models.NanoGPT.Track1.TailNorm
import models.NanoGPT.Track1.SparseLanding
import models.NanoGPT.Track1.NgramAccumulation
import models.NanoGPT.Track1.ValueEmbedAccum
import models.NanoGPT.Track1.ReplicatedAdam
import models.NanoGPT.Track1.AnvilEpilogue
import models.NanoGPT.Track1.AnvilGram
import models.NanoGPT.Track1.MuddGradientGrouping
import models.NanoGPT.Track1.MuddGradientPacking
import models.NanoGPT.Track1.MuddFusedPacking
import models.NanoGPT.Track1.MuddFullPacking
import VeriTile.Spec
import VeriTile.Meta.Specification

/-!
Observable contract of the patched Track 1 calls.

* Scale refresh observes all twelve FP32 buffers, including the two amax
  clears, the live scales, and the retained value of ordinary slot ten.
* The CPU n-gram row hash observes both channels of int32 row IDs.
  Forward observes every BF16 element in [T,768]. Backward observes both
  BF16 [T,768] halves of the upstream sink gradient.
* The QKV scan observes the address and FP32 value of each partial amax
  store for the three compiled attention width groups.
* The fused attention scale update observes the FP32 scale store after
  reducing all 32 partials, clamping to the floor, and multiplying.
* QKV quantization observes both output addresses and the shared FP8 byte.
* MLP weight refresh observes the published scales, every original
  per-tile amax, and both FP8 layouts for both projections.
* LM-head refresh observes both FP8 cache stores after the BF16 round trip.
* Sampled softmax observes both raw-byte layouts indexed by its candidate IDs.
* Sampled gradient densification observes the selected BF16 halfwords and
  positive zero at classes absent from the candidate set.
* YaRN observes every decoded factor and the valid prefix across reset,
  window changes, and completion before validation.
* The long-window RoPE epilogue observes every BF16 query and key, with
  stationary key channels written forward from their producing token.
* The long-window backward observes both FP8 layouts after the selected
  K gradient, RMS correction, and BF16 round trip.
* The 64-channel backward observes both FP8 layouts after permuting
  adjacent rotary factors, with the paired and half-width value layouts.
* Dense FP8 transpose observes the destination offset and exact stored byte.
* Tied-embedding transpose observes every BF16 weight copy and the ordered
  BF16 add into the LM-head gradient.
* Tail averaging observes the seeded flags and both FP32 weights after each tick.
* Tail norm restoration observes the published BF16 weights of all five banks.
* The sparse exchange observes the owner's FP16 gradients and both outgoing
  wire buffers in the row pull's original order.
* These specifications describe the software value semantics and the
  substituted buffer boundary. CUDA compilation, collective scheduling,
  elapsed time and the loss threshold are assessed separately.
-/
namespace VeriTile.Bench.Models.NanoGPT.Track1
open VeriTile Triton.FP
open scoped VeriTile.Spec

private def compactGroupBank (w : QKVLaunch.Width) (qk : Bool) (raw : QKVTTIR.Bank) :
    QKVTTIR.Bank :=
  QKVNative.compactBank w qk (QKVNative.bankView (Grouped.slotBase w) raw)

private def compactGroup (w : QKVLaunch.Width) (raw : QKVPack.Inputs) : QKVPack.Inputs :=
  QKVNative.compactInputs w (QKVNative.inputsView (Grouped.slotBase w) raw)

structure PatchedCalls where
  refresh : Scalars → Buffers → Buffers
  weightRefresh : (hopper : Bool) → QKVPack.Arithmetic → (delayed : Bool) → MLPWeights.Inputs →
    MLPWeights.WeightRefresh
  headRefresh : (hopper : Bool) → QKVTTIR.Bank → Scalar → LMHead.HeadPoint → Option QKVPack.Writes
  sampledGather : SampledInverse.Gather
  sampledDensify : Densify.Dispatch
  yarnTables : YarnTable.Arithmetic → YarnTable.Layout → Nat → YarnTable.State →
    List YarnTable.Step → YarnTable.State
  wideYarnTables : YarnTable.Arithmetic → Nat → YarnStationary.RotatingState →
    List YarnStationary.Step → YarnStationary.Observation
  ropeForward : (fastPath : Bool) → RopeShift.Arithmetic → RopeShift.Inputs →
    (length : RopeShift.Length) → RopeShift.Point length → Option (RopeShift.Plane × Nat × Value .bf16)
  ropeBackward : (fastPath : Bool) → RopeBackward.Target → (length : RopeShift.Length) →
    RopeBackward.Inputs length → RopeBackward.Point length → (Nat × UInt8) × (Nat × UInt8)
  ropeBackwardNarrow : RopeBackward.Target → RopeBackward.Narrow.Inputs →
    (shape : RopeBackward.Narrow.Shape) → (length : RopeShift.Length) →
    RopeBackward.Narrow.Point shape length → (Nat × UInt8) × (Nat × UInt8)
  byteTranspose : (fastPath : Bool) → (shape : ByteTranspose.Shape) → (Nat → UInt8) →
    ByteTranspose.Point shape → Option (Nat × UInt8)
  ceTranspose : CETranspose.Dispatch
  trainingCE : TrainingCE.Run
  validationCE : ValidationCE.BatchRun := ValidationCE.runBatch
  bf16Transpose : Bf16Transpose.Dispatch
  lookup : (fullRows : Bool) → (tokens : Nat) → Ngram.Inputs → Ngram.Contraction →
    Fin tokens → Fin Ngram.D → Value .bf16
  backward : (fullRows : Bool) → (tokens : Nat) → Ngram.Inputs → Ngram.BackwardArithmetic →
    Fin tokens → Fin Ngram.D → Value .bf16 × Value .bf16
  qkvAmax : (grouped : Bool) → QKVTTIR.Bank → QKVTTIR.Bank →
    ((QKVAmax.Lane → Value .fp32) → Value .fp32) →
    (width : QKVLaunch.Width) → Fin (QKVLaunch.Layout.ofWidth width).batches →
    QKVAmax.Part → Nat × Value .fp32
  qkvScale : (grouped : Bool) → (Nat → Value .fp32) → (width : QKVLaunch.Width) →
    Fin (QKVLaunch.Layout.ofWidth width).batches → Nat × Value .fp32
  qkvPack : (grouped : Bool) → QKVPack.Arithmetic → QKVPack.Inputs → (width : QKVLaunch.Width) →
    QKVPack.Point (QKVPack.Geometry.ofWidth width) → Option QKVPack.Writes
  tailUpdate : TailLerp.Update
  tailRestore : TailNorm.Update
  sparsePackets : SparsePackets.Kernels
  valueAccumulation : ValueEmbedAccum.Accumulate
  ngramAccumulation : NgramAccumulation.Accumulator
  sparseLanding : SparseLanding.Kernels
  replicatedAdam : ReplicatedAdam.Step
  anvilRails : AnvilRails.Dispatch
  anvilGram : AnvilGram.Kernel
  anvilInitial : AnvilGram.Initial
  anvilRound : AnvilGram.Round
  anvilNormalize : AnvilTrace.Normalization
  anvilEpilogue : AnvilEpilogue.Epilogue
  ngramRowHash : FineWeb.RowHash
  packTraining : FineWeb.TrainingPack
  scheduleUpload : HostSchedule.Upload
  graphStage : GraphStage.Kernels
  graphLaunch : GraphStage.Launch := GraphStage.referenceLaunch
  graphCopyLaunch : GraphStage.CopyLaunch := GraphStage.referenceCopyLaunch
  headActivation : HeadActivation.Run := HeadActivation.reference
  muddCoefficients : MuddCoefficients.Run := MuddCoefficients.reference
  muddNorm : MuddNorm.TrainingRun := MuddNorm.RowProgram.staged.training .squareSum
  muddBackward : MuddBackward.BatchRun := MuddBackward.execute MuddBackward.backward
  muddActivation : MuddActivationStorage.Read := MuddActivationStorage.Program.duplicate.read
  muddResidual : MuddResidualStorage.Run := MuddResidualStorage.Program.reload.run
  muddGradients : MuddGradientGrouping.Run := MuddGradientGrouping.Program.separate.run
  muddPacking : MuddGradientPacking.Run := MuddGradientPacking.Program.interleaved.run
  muddFusedPacking : MuddFusedPacking.Run := MuddFusedPacking.Program.separate.run
  muddFullPacking : MuddFullPacking.Run := MuddFullPacking.reference

def referenceCalls : PatchedCalls where
  refresh := staged
  weightRefresh := fun _ => MLPWeights.readReference
  headRefresh := fun _ bank gain p => some (LMHead.reference bank gain p)
  sampledGather := SampledInverse.referenceGather
  sampledDensify := fun _ _ => Densify.reference
  yarnTables := YarnTable.reference
  wideYarnTables := YarnStationary.reference
  ropeForward := fun _ op read _ => RopeShift.referenceWrite op read
  ropeBackward := fun _ op _ => RopeBackward.writes false op
  ropeBackwardNarrow := fun op read _ _ => RopeBackward.Narrow.writes false op read
  byteTranspose := fun _ _ => ByteTranspose.reference
  ceTranspose := fun _ => CETranspose.staged
  trainingCE := TrainingCE.reference
  bf16Transpose := fun _ cfg _ => Bf16Transpose.reference cfg
  lookup := fun _ t s contract row col => Ngram.flatForward s t (Ngram.index row.val col.val) contract
  backward := fun _ _ s arithmetic row col => Ngram.flatBackward s (Ngram.index row.val col.val) arithmetic
  qkvAmax := fun _ first second reduce w =>
    QKVTTIR.referenceStore (compactGroupBank w true first) (compactGroupBank w false second) reduce w
  qkvScale := fun _ => QKVScale.referenceStore
  qkvPack := fun _ op raw w point => QKVPack.reference op (compactGroup w raw) point
  tailUpdate := fun _ _ => TailLerp.reference
  tailRestore := TailNorm.reference
  sparsePackets := SparsePackets.staged
  valueAccumulation := ValueEmbedAccum.staged
  ngramAccumulation := NgramAccumulation.delayed
  sparseLanding := SparseLanding.staged
  replicatedAdam := @ReplicatedAdam.reference
  anvilRails := AnvilRails.reference
  anvilGram := AnvilGram.reference
  anvilInitial := AnvilGram.splitInitial AnvilGram.reference
  anvilRound := AnvilGram.splitRound AnvilGram.reference
  anvilNormalize := AnvilTrace.normalize AnvilTrace.original
  anvilEpilogue := AnvilEpilogue.reference
  ngramRowHash := FineWeb.hashRows
  packTraining := FineWeb.packTrainingReference
  scheduleUpload := HostSchedule.uploadReference
  graphStage := GraphStage.staged

def candidateCalls : PatchedCalls where
  headActivation := HeadActivation.fused
  muddCoefficients := MuddCoefficients.fused
  muddNorm := MuddNorm.RowProgram.combined.training .inverse
  muddActivation := MuddActivationStorage.Program.single.read
  muddResidual := MuddResidualStorage.Program.register.run
  muddGradients := MuddGradientGrouping.Program.grouped.run
  muddPacking := MuddGradientPacking.Program.hoisted.run
  muddFusedPacking := MuddFusedPacking.Program.fused.run
  muddFullPacking := MuddFullPacking.candidate
  refresh := fused
  weightRefresh := fun hopper =>
    if hopper then MLPWeights.readWarp else MLPWeights.readReference
  headRefresh := fun hopper bank gain p =>
    if hopper then LMHead.hopperDispatch bank gain p
    else some (LMHead.reference bank gain p)
  sampledGather := SampledInverse.fusedGather
  sampledDensify := fun hopper _ src pos p =>
    if hopper then Densify.emitted src pos p else Densify.reference src pos p
  yarnTables := YarnTable.observed
  wideYarnTables := YarnStationary.candidate
  ropeForward := fun fastPath op read _ p =>
    if fastPath then RopeShift.emittedWrite op read p else RopeShift.referenceWrite op read p
  ropeBackward := fun fastPath op _ => RopeBackward.writes fastPath op
  ropeBackwardNarrow := fun op read _ _ => RopeBackward.Narrow.writes true op read
  byteTranspose := fun fastPath _ bytes p =>
    if fastPath then ByteTranspose.emitted bytes p else ByteTranspose.reference bytes p
  ceTranspose := fun fastPath => if fastPath then CETranspose.unsigned else CETranspose.staged
  trainingCE := TrainingCE.specialized
  bf16Transpose := fun fastPath cfg _ src dst p =>
    if fastPath then Bf16Transpose.emitted cfg src dst p else Bf16Transpose.reference cfg src dst p
  lookup := fun fullRows _ s contract row col =>
    if fullRows then Ngram.fullForward s (Ngram.FullLane.ofOwner (Ngram.tileOwner row col)) contract
    else Ngram.tiledForward s (Ngram.tileOwner row col) contract
  backward := fun fullRows _ s arithmetic row col =>
    if fullRows then Ngram.fullBackward s (Ngram.FullLane.ofOwner (Ngram.tileOwner row col)) arithmetic
    else Ngram.tiledBackward s (Ngram.tileOwner row col) arithmetic
  qkvAmax := fun grouped first second reduce w =>
    if grouped then Grouped.ScanSchedule.perWidthScan first second reduce w
    else QKVTTIR.emittedStore (compactGroupBank w true first) (compactGroupBank w false second) reduce w
  qkvScale := fun grouped => if grouped then Grouped.perWidthScale else QKVScale.fusedStore
  qkvPack := fun grouped op raw w point =>
    if grouped then Grouped.perWidthPack op raw w point
    else some (QKVPack.emitted op (compactGroup w raw) w point)
  tailUpdate := TailLerp.candidate
  tailRestore := TailNorm.grouped
  sparsePackets := SparsePackets.direct
  valueAccumulation := ValueEmbedAccum.fused
  ngramAccumulation := NgramAccumulation.online
  sparseLanding := SparseLanding.direct
  replicatedAdam := @ReplicatedAdam.candidate
  anvilRails := AnvilRails.optimized
  anvilGram := AnvilGram.compact
  anvilInitial := AnvilGram.splitInitial AnvilGram.compact
  anvilRound := AnvilGram.splitRound AnvilGram.compact
  anvilNormalize := AnvilNormalization.packedNormalize AnvilTrace.optimized
  anvilEpilogue := AnvilEpilogue.fused
  ngramRowHash := FineWeb.hashRowsCPU
  packTraining := fun tokens segments count length hash =>
    let batch := FineWeb.packTrainingCPU tokens segments count length hash
    if FineWeb.canPackTraining count then batch.map PinnedBatch.uploadPacked else batch
  scheduleUpload := HostSchedule.uploadPacked
  graphStage := GraphStage.combined

theorem calls_equal : candidateCalls = referenceCalls := by
  have h₁ : candidateCalls.refresh = referenceCalls.refresh := by
    funext c s
    exact buffer_equality c s
  have hw : candidateCalls.weightRefresh = referenceCalls.weightRefresh := by
    funext hopper arithmetic delayed input
    cases hopper
    · rfl
    · exact MLPWeights.warp_refresh_equal arithmetic delayed input
  have hh : candidateCalls.headRefresh = referenceCalls.headRefresh := by
    funext hopper bank gain p
    cases hopper
    · rfl
    · exact LMHead.hopperDispatch_eq bank gain p
  have hd : candidateCalls.sampledDensify = referenceCalls.sampledDensify := by
    funext hopper count src pos p
    cases hopper
    · rfl
    · exact Densify.emitted_equal src pos p
  have hy : candidateCalls.yarnTables = referenceCalls.yarnTables := by
    funext arithmetic layout maxRows state steps
    exact YarnTable.observed_equal arithmetic layout maxRows state steps
  have hwy : candidateCalls.wideYarnTables = referenceCalls.wideYarnTables := by
    funext arithmetic maxRows state steps
    exact YarnStationary.observed_equal arithmetic maxRows state steps
  have hr : candidateCalls.ropeForward = referenceCalls.ropeForward := by
    funext fastPath op read length p
    cases fastPath
    · rfl
    · exact RopeShift.writes_equal op read p
  have hbr : candidateCalls.ropeBackward = referenceCalls.ropeBackward := by
    funext fastPath op length read p
    cases fastPath
    · rfl
    · exact RopeBackward.both_layouts_equal op read p
  have hbn : candidateCalls.ropeBackwardNarrow = referenceCalls.ropeBackwardNarrow := by
    funext op read shape length p
    exact RopeBackward.Narrow.both_layouts_equal op read p
  have ht : candidateCalls.byteTranspose = referenceCalls.byteTranspose := by
    funext fastPath shape bytes p
    cases fastPath
    · rfl
    · exact ByteTranspose.emitted_equal bytes p
  have hce : candidateCalls.ceTranspose = referenceCalls.ceTranspose := by
    funext fastPath
    cases fastPath
    · rfl
    · exact CETranspose.unsigned_stores_equal
  have hb : candidateCalls.bf16Transpose = referenceCalls.bf16Transpose := by
    funext fastPath cfg mode src dst p
    cases fastPath
    · rfl
    · exact Bf16Transpose.emitted_equal cfg src dst p
  have h₂ : candidateCalls.lookup = referenceCalls.lookup := by
    funext fullRows t s contract row col
    cases fullRows
    · exact Ngram.forward_equal s row col contract
    · exact Ngram.full_forward_equal s row col contract
  have h₃ : candidateCalls.backward = referenceCalls.backward := by
    funext fullRows t s arithmetic row col
    cases fullRows
    · exact Ngram.backward_equal s row col arithmetic
    · exact Ngram.full_backward_equal s row col arithmetic
  have h₄ : candidateCalls.qkvAmax = referenceCalls.qkvAmax := by
    funext grouped first second reduce w batch part
    cases grouped
    · exact QKVTTIR.store_lowering _ _ reduce w batch part
    · exact congrArg Prod.snd (Grouped.ScanSchedule.store_equiv first second reduce w batch part)
  have h₅ : candidateCalls.qkvScale = referenceCalls.qkvScale := by
    funext grouped scratch width batch
    cases grouped
    · exact QKVScale.store_equal scratch width batch
    · exact congrArg Prod.snd (Grouped.scale_store_equiv (fun _ => scratch) width batch)
  have h₆ : candidateCalls.qkvPack = referenceCalls.qkvPack := by
    funext grouped op raw w p
    cases grouped
    · exact QKVPack.emitted_equiv op (compactGroup w raw) w p
    · exact congrArg Prod.snd (Grouped.pack_store_equiv op raw.first raw.second (fun _ => raw.scales) w p)
  have htail : candidateCalls.tailUpdate = referenceCalls.tailUpdate := by
    funext eligible elements inputs first second i
    exact TailLerp.candidate_equal eligible elements inputs first second i
  have hrestore : candidateCalls.tailRestore = referenceCalls.tailRestore := by
    funext problem slot i
    exact TailNorm.grouped_equal problem slot i
  have hnorm : candidateCalls.anvilNormalize = referenceCalls.anvilNormalize :=
    (AnvilNormalization.packed_normalize_eq AnvilTrace.optimized).trans
      (congrArg AnvilTrace.normalize AnvilTrace.dispatch_equal)
  have hpack : candidateCalls.packTraining = referenceCalls.packTraining := by
    funext tokens segments count length hash
    simp only [candidateCalls, referenceCalls, FineWeb.pack_training_equal]
    split
    · cases FineWeb.packTrainingReference tokens segments count length hash <;> simp
    · rfl
  simp only [candidateCalls, referenceCalls, PatchedCalls.mk.injEq, true_and]
  exact ⟨h₁, hw, hh, SampledInverse.gather_equal, hd, hy, hwy, hr, hbr, hbn, ht, hce, TrainingCE.specialized_equal,
    hb, h₂, h₃, h₄, h₅, h₆, htail, hrestore,
    SparsePackets.kernels_equal, ValueEmbedAccum.accum_equal, NgramAccumulation.accumulator_equal,
    SparseLanding.kernels_equal, ReplicatedAdam.step_equal,
    AnvilRails.dispatch_equal, AnvilGram.stores_equal,
    congrArg AnvilGram.splitInitial AnvilGram.stores_equal,
    congrArg AnvilGram.splitRound AnvilGram.stores_equal, hnorm, AnvilEpilogue.epilogue_equal,
    FineWeb.hash_rows_cpu_equal, hpack, HostSchedule.upload_equal, GraphStage.kernels_equal,
    HeadActivation.fused_equal, MuddCoefficients.fused_equal, MuddNorm.training_run_eq, MuddActivationStorage.read_eq,
    MuddResidualStorage.run_eq, MuddGradientGrouping.grouped_run_eq, MuddGradientPacking.run_eq,
    MuddFusedPacking.run_eq, MuddFullPacking.run_eq⟩

theorem tail_tick_equal (snapshot : TailLerp.Snapshot) :
    TailLerp.tick candidateCalls.tailUpdate snapshot =
      TailLerp.tick referenceCalls.tailUpdate snapshot := by
  rw [calls_equal]

theorem wide_rope_after_yarn {n : RopeShift.Length} (yarn : YarnTable.Arithmetic) (maxRows : Nat)
    (state : YarnStationary.RotatingState) (steps : List YarnStationary.Step)
    (forward : RopeShift.Arithmetic) (qk : Nat → Value .bf16) (p : RopeShift.Point n)
    (backward : RopeBackward.Target) (inputs : RopeBackward.Inputs n) (grad : RopeBackward.Point n) :
    let run := fun (calls : PatchedCalls) =>
      let table := calls.wideYarnTables yarn maxRows state steps
      (calls.ropeForward true forward (YarnStationary.forwardInputs table qk) n p,
       calls.ropeBackward true backward n (YarnStationary.backwardInputs table inputs) grad)
    run candidateCalls = run referenceCalls := by
  rw [calls_equal]

theorem narrow_forward_after_yarn {n : RopeShift.Length}
    (yarn : YarnTable.Arithmetic) (layout : RopeBackward.Narrow.Shape) (maxRows : Nat)
    (state : YarnTable.State) (steps : List YarnTable.Step) (op : RopeShift.Arithmetic) (qk : Nat → Value .bf16)
    (t : Fin n.tokens) (p : RopeShift.Plane) (h : RopeShift.Head) (d : RopeBackward.Narrow.QkChannel) :
    let s := match layout with | .paired => YarnTable.Layout.paired | .halfValue => .narrow
    YarnTable.narrowForward op qk layout (candidateCalls.yarnTables yarn s maxRows state steps) t p h d =
      YarnTable.narrowForward op qk layout (referenceCalls.yarnTables yarn s maxRows state steps) t p h d := by
  rw [calls_equal]

/-- The partial writes and published scale remain observable alongside
the row-major and transposed FP8 cache bytes. -/
structure QKVRefreshOutput where
  partials : (width : QKVLaunch.Width) → Fin (QKVLaunch.Layout.ofWidth width).batches →
    QKVAmax.Part → Nat × Scalar
  scales : (width : QKVLaunch.Width) → Fin (QKVLaunch.Layout.ofWidth width).batches → Nat × Scalar
  packed : (width : QKVLaunch.Width) → QKVPack.Point (QKVPack.Geometry.ofWidth width) →
    Option QKVPack.Writes

/-- Run the three stages on the raw banks in source order. -/
def refreshCache (calls : PatchedCalls) (grouped : Bool) (op : QKVPack.Arithmetic)
    (first second : QKVTTIR.Bank) (reduce : (QKVAmax.Lane → Scalar) → Scalar) :
    QKVRefreshOutput :=
  let partials w := calls.qkvAmax grouped first second reduce w
  let scales w := calls.qkvScale grouped (QKVScale.partialBuffer w (partials w)) w
  let scaleBank w b :=
    if h : b < (QKVLaunch.Layout.ofWidth w).batches then (scales w ⟨b,h⟩).2 else zero
  ⟨partials, scales, fun w => calls.qkvPack grouped op ⟨first, second, scaleBank w⟩ w⟩

def refreshCacheElement (calls : PatchedCalls) (grouped : Bool) (op : QKVPack.Arithmetic)
    (first second : QKVTTIR.Bank) (reduce : (QKVAmax.Lane → Scalar) → Scalar)
    (w : QKVLaunch.Width) (p : QKVPack.Point (QKVPack.Geometry.ofWidth w)) : Option QKVPack.Writes :=
  (refreshCache calls grouped op first second reduce).packed w p

theorem refreshed_cache_bytes_equal (hopper : Bool) (op : QKVPack.Arithmetic)
    (first second : QKVTTIR.Bank) (reduce : (QKVAmax.Lane → Scalar) → Scalar)
    (w : QKVLaunch.Width) (p : QKVPack.Point (QKVPack.Geometry.ofWidth w)) :
    refreshCacheElement candidateCalls hopper op first second reduce w p =
      refreshCacheElement referenceCalls hopper op first second reduce w p := by
  rw [calls_equal]

inductive CallCode where
  | stagedMlpWeightRefresh | pairedMlpWeightRefresh
  | atenMlpRefresh | tritonMlpRefresh
  | maskedHeadPack | warpHeadPack
  | sampledGatherThenTranspose | rebuildSampledInverse | fusedSampledGatherAndInverse
  | maskedDensify | alignedDensify
  | denseYarn | compactYarn | implicitStationaryYarn
  | previousKeyRope | forwardStoreRope
  | fullKeyGradient | selectedKeyGradient
  | maskedByteTranspose | warpByteTranspose
  | separateCEOperands | pairedCEOperands
  | scalarCESigmoid | pairedCESigmoid | fixedCEPredictions
  | maskedTiedTranspose | warpTiedTranspose
  | flatEmbedding | fullRowEmbedding
  | flatEmbeddingBackward | fullRowEmbeddingBackward
  | maskedPackedAmax | groupedPackedAmax
  | atenQkvScale | groupedQkvScale
  | maskedQkvPack | groupedQkvPack
  | separateTailLerp | sharedTailLerp
  | materializedNorms | groupedNormRestore
  | stagedSparsePackets | directSparsePackets
  | delayedNgramGradientSum | onlineNgramGradientSum
  | separateValueGradientPlanes | fusedValueGradientPlanes
  | stagedSparseLanding | directSparseLanding
  | globalNgramSlots | bucketedNgramSlots
  | perRowMomentReplay | warpMomentReplaySkippingZero
  | ngramAdamRowUpdate | claimedNgramAdamRowUpdate
  | stagedAdamEntries | packedAdamEntries
  | storedValueAdamIndices | directValueAdamIndices
  | denseValueAdamScale | markedValueAdamScale
  | perElementAdamSegments | perTileAdamSegments
  | storedAnvilRails | fusedAnvilRails
  | squareAnvilGrid | triangularAnvilGrid
  | splitGramTrace | fusedGramDivisors
  | storedAnvilGradient | fusedAnvilEpilogue
  | separateGraphCopies | fusedGraphCopy
  | separateLookupStage | fusedLookupStage
  | separateBatchUploads | packedBatchUpload
  | separateMuddCoefficients | fusedMuddCoefficients
  | stagedMuddNorm | fusedMuddNorm

/-- The two executable models use the same inputs and output shape. Torch
operations and the fused scale refresh use FP32 arithmetic, with the explicit
RNE BF16 and widening cast on the post-lambda. Embeddings use BF16 storage
and FP32 arithmetic, including the original optional contraction. -/
inductive TrackProgram where
  | reference | candidate

def TrackProgram.body : TrackProgram → List CallCode
  | .reference => [.stagedMlpWeightRefresh, .atenMlpRefresh, .maskedHeadPack, .sampledGatherThenTranspose,
      .rebuildSampledInverse,
      .maskedDensify, .denseYarn, .previousKeyRope, .fullKeyGradient, .maskedByteTranspose, .separateCEOperands,
      .scalarCESigmoid,
      .maskedTiedTranspose,
      .flatEmbedding, .flatEmbeddingBackward,
      .maskedPackedAmax, .atenQkvScale, .maskedQkvPack, .separateTailLerp, .materializedNorms,
      .stagedSparsePackets, .separateValueGradientPlanes, .delayedNgramGradientSum, .stagedSparseLanding,
      .globalNgramSlots, .perRowMomentReplay,
      .stagedAdamEntries, .ngramAdamRowUpdate,
      .storedValueAdamIndices, .denseValueAdamScale, .perElementAdamSegments,
      .storedAnvilRails, .squareAnvilGrid, .splitGramTrace, .storedAnvilGradient, .separateGraphCopies,
      .separateBatchUploads, .separateLookupStage, .separateMuddCoefficients, .stagedMuddNorm]
  | .candidate => [.pairedMlpWeightRefresh, .tritonMlpRefresh, .warpHeadPack, .fusedSampledGatherAndInverse,
      .alignedDensify, .compactYarn, .implicitStationaryYarn, .forwardStoreRope, .selectedKeyGradient,
      .warpByteTranspose, .pairedCEOperands, .pairedCESigmoid, .fixedCEPredictions, .warpTiedTranspose,
      .fullRowEmbedding, .fullRowEmbeddingBackward,
      .groupedPackedAmax, .groupedQkvScale, .groupedQkvPack, .sharedTailLerp, .groupedNormRestore,
      .directSparsePackets, .fusedValueGradientPlanes, .onlineNgramGradientSum, .directSparseLanding, .bucketedNgramSlots,
      .warpMomentReplaySkippingZero, .packedAdamEntries, .claimedNgramAdamRowUpdate,
      .directValueAdamIndices, .markedValueAdamScale, .perTileAdamSegments,
      .fusedAnvilRails, .triangularAnvilGrid, .fusedGramDivisors, .fusedAnvilEpilogue, .fusedGraphCopy,
      .packedBatchUpload, .fusedLookupStage, .fusedMuddCoefficients, .fusedMuddNorm]

def TrackProgram.calls : TrackProgram → PatchedCalls
  | .reference => referenceCalls
  | .candidate => candidateCalls

theorem TrackProgram.batch_hash (side : TrackProgram) :
    FineWeb.TrainingPack.preservesHash side.calls.packTraining side.calls.ngramRowHash := by
  cases side
  · exact FineWeb.reference_batch_hashed
  · simpa only [TrackProgram.calls, calls_equal] using FineWeb.reference_batch_hashed

theorem TrackProgram.batch_size (side : TrackProgram) :
    FineWeb.TrainingPack.preservesSize side.calls.packTraining := by
  cases side
  · exact FineWeb.reference_batch_size
  · simpa only [TrackProgram.calls, calls_equal] using FineWeb.reference_batch_size

def sameCalls (a b : TrackProgram) : Prop := a.calls = b.calls

instance : Spec.ProgramSyntax TrackProgram where
  Statement := CallCode
  Signature := Nat × Nat × Nat × Format × Format
  signature := fun _ => (12, 11, 768, .fp32, .bf16)
  body := TrackProgram.body
  structural := some sameCalls

specification patched_calls_equiv :
    TrackProgram.candidate ≡[([] : Spec.Assumptions CallCode)] TrackProgram.reference :=
  Spec.FloatingPoint.ofStructural (structural := sameCalls) rfl rfl calls_equal

/-- All callers at the displayed boundary can be composed for arbitrary
steps. Both sides execute the same supplied caller with the proved patch
entry points. -/
theorem modeled_training_equal {State : Type}
    (step : PatchedCalls → State → Nat → State)
    (initial : State) (steps : List Nat) :
    steps.foldl (step candidateCalls) initial = steps.foldl (step referenceCalls) initial := by
  rw [calls_equal]

#axiomsClean calls_equal
#axiomsClean tail_tick_equal
#axiomsClean wide_rope_after_yarn
#axiomsClean narrow_forward_after_yarn
#axiomsClean refreshed_cache_bytes_equal
#axiomsClean patched_calls_equiv
#axiomsClean modeled_training_equal
#print_fp_assumptions patched_calls_equiv
end VeriTile.Bench.Models.NanoGPT.Track1
