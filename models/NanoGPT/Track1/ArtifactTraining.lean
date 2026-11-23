import models.NanoGPT.Track1.TrainingCEExecution
import models.NanoGPT.Track1.NgramScratchBinding
import models.NanoGPT.Track1.NgramAdamBinding
import models.NanoGPT.Track1.NgramEmbeddingExecution
import models.NanoGPT.Track1.ValueAdamWorkspaceTrace
import models.NanoGPT.Track1.ValueAdamLocalLoader
import models.NanoGPT.Track1.AttentionExecution
import models.NanoGPT.Track1.YarnExecution
import models.NanoGPT.Track1.ValueAdamBinding
import models.NanoGPT.Track1.SparseLandingBinding
import models.NanoGPT.Track1.MomentReplayBinding
import models.NanoGPT.Track1.NgramAccumulationBinding
import models.NanoGPT.Track1.ValueEmbedAccumBinding
import models.NanoGPT.Track1.QKVRefreshBinding
import models.NanoGPT.Track1.TailLerpBinding
import models.NanoGPT.Track1.TailNormBinding
import models.NanoGPT.Track1.AnvilRailsBinding
import models.NanoGPT.Track1.AnvilNormalizationBinding
import models.NanoGPT.Track1.AnvilGramBinding
import models.NanoGPT.Track1.AnvilCooperativeBinding
import models.NanoGPT.Track1.AnvilEpilogueBinding
import models.NanoGPT.Track1.ValidationProjectionBinding
import models.NanoGPT.Track1.SampledInverseBinding
import models.NanoGPT.Track1.HeadActivationBinding
import models.NanoGPT.Track1.SampledDensifyBinding
import models.NanoGPT.Track1.MLPWeightBinding
import models.NanoGPT.Track1.LMHeadBinding
import models.NanoGPT.Track1.Bf16TransposeBinding
import models.NanoGPT.Track1.MuddCoefficientsBinding
import models.NanoGPT.Track1.MuddNormBinding
import models.NanoGPT.Track1.MuddNormHeadReplay
import models.NanoGPT.Track1.MuddBackwardBinding
import models.NanoGPT.Track1.MuddActivationBinding
import models.NanoGPT.Track1.MuddGradientPackingBinding
import models.NanoGPT.Track1.MuddFusedPackingBinding

/-! End-to-end driver execution with the graph copies, fused staging and
cross-entropy programs. Unchanged operations use the common runtime.
The dynamic training CE path uses its paired source semantics. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver.ArtifactTraining
open VeriTile.Spec
open scoped VeriTile.Spec

structure OptimizerPrograms where
  rails : AnvilRails.Programs
  trace : AnvilTrace.Programs
  normalize : AnvilNormalization.Programs
  gram : AnvilGram.Programs
  cooperative : AnvilGram.Cooperative.Programs
  epilogue : AnvilEpilogue.Binding.Programs

structure TailPrograms where
  lerp : TailLerp.Artifacts
  lerpCertificate : TailLerp.Certificate lerp
  normIR : String
  normPTX : String
  normCertificate : TailNorm.Certificate normIR normPTX

def TailPrograms.update (p : TailPrograms) : TailLerp.Update :=
  match p.lerpCertificate with
  | ⟨_, _, _, _, _⟩ => TailLerp.candidate

def TailPrograms.restore (p : TailPrograms) : TailNorm.Update :=
  match p.normCertificate with
  | ⟨_, _, _, _⟩ => TailNorm.grouped

theorem TailPrograms.update_eq (p : TailPrograms) : p.update = referenceCalls.tailUpdate := by
  funext eligible elements inputs first second i
  exact p.lerpCertificate.orderedStores eligible elements inputs first second i

theorem TailPrograms.restore_eq (p : TailPrograms) : p.restore = referenceCalls.tailRestore := by
  funext problem slot i
  exact p.normCertificate.values problem slot i

structure SparsePrograms where
  packetArtifacts : SparsePackets.Artifacts
  packets : SparsePackets.Certificate packetArtifacts
  scratchArtifacts : NgramScratch.Artifacts
  scratch : NgramScratch.Certificate scratchArtifacts
  adamArtifacts : NgramAdam.Artifacts
  adam : NgramAdam.Certificate adamArtifacts
  valueArtifacts : ValueAdam.Artifacts
  value : ValueAdam.Certificate valueArtifacts
  valueNative : ValueAdam.Elementwise.NativePrograms
  valueLocal : ValueAdam.LocalRows.Programs
  fetchArtifacts : SparseLanding.Artifacts
  fetch : SparseLanding.Certificate fetchArtifacts
  momentArtifacts : MomentReplay.Artifacts
  moments : MomentReplay.Certificate momentArtifacts
  ngramAccumIR : String
  ngramAccumPTX : String
  ngramAccum : NgramAccumulation.Certificate ngramAccumIR ngramAccumPTX
  valueAccumArtifacts : ValueEmbedAccum.Artifacts
  valueAccum : ValueEmbedAccum.Certificate valueAccumArtifacts
  embeddingArtifacts : NgramEmbedBinding.Artifacts
  embedding : NgramEmbedBinding.Certificate embeddingArtifacts

open ValueAdam (Arithmetic Launch Memory) in
theorem SparsePrograms.grouped_value_event_exact (programs : SparsePrograms)
    (direct marked : Bool) (period : ValueAdam.Elementwise.Period) (s : ValueAdam.SourceSlice)
    (peer : Fin s.asked.size → SparsePackets.Channel → ValueAdam.Half)
    (order : ValueAdam.Schedule (s.deferredPull peer)) (arithmetic : Arithmetic) (launch : Launch)
    (memory : Memory) (rows : memory.rows = 25152)
    (history : ValueAdam.Elementwise.History) (before : ValueAdam.Elementwise.Workspace) (sparse : Bool)
    (valid : history.represents before)
    (matched : ValueAdam.Elementwise.matchesPeriod launch period) :
    let localKernel := programs.valueLocal.get direct marked
    let pull := s.deferredPull peer
    let indices := AdamEntries.fused s.ids
    ValueAdam.Elementwise.consumingEventAfterScatter programs.valueNative period pull indices order
      arithmetic launch memory rows history before sparse
      (ValueAdam.LocalRows.observedMerge localKernel.ptx direct pull indices order) =
        (ValueAdam.event ValueAdam.dense pull indices order arithmetic launch memory,
          before.consumed history sparse) := by
  intro localKernel pull indices
  have exactScatter : ValueAdam.LocalRows.observedMerge localKernel.ptx direct pull indices order =
      ValueAdam.mergeFrom pull indices order := by
    funext workspace
    exact localKernel.checked.merged_exact s peer order workspace
  rw [exactScatter]
  exact ValueAdam.Elementwise.consuming_event_exact programs.valueNative period pull indices order
    arithmetic launch memory rows history before sparse valid matched

def SparsePrograms.sendNgram (p : SparsePrograms) (s : SparsePackets.NgramInput) :
    NgramScratch.Payload s.cut × SparsePackets.FP16Rows :=
  if NgramScratch.recognized p.scratchArtifacts then
    let (packet, scratch) := NgramScratch.send true s
    (packet.tiledWire, scratch)
  else NgramScratch.send false s

theorem SparsePrograms.sendNgram_exact (p : SparsePrograms) (s : SparsePackets.NgramInput) :
    p.sendNgram s = (SparsePackets.ngramDirect s, NgramScratch.clearPeer s.cut s.grad) := by
  simp only [SparsePrograms.sendNgram, p.scratch.matched, if_true, SparsePackets.Packets.tiledWire_eq,
    p.scratch.packet]
  rfl

def SparsePrograms.run (p : SparsePrograms) : SparsePackets.Kernels :=
  if SparsePackets.recognized p.packetArtifacts then
    ⟨fun s => (SparsePackets.valueDeferred s).tiledWire,
      fun s => (p.sendNgram s).1, SparsePackets.clearDirect⟩
  else SparsePackets.staged

theorem SparsePrograms.run_eq (p : SparsePrograms) : p.run = SparsePackets.staged := by
  have value : (fun s => (SparsePackets.valueDeferred s).tiledWire) = SparsePackets.valueStaged := by
    funext s
    rw [SparsePackets.Packets.tiledWire_eq, p.packets.deferredValue]
  have ngram : (fun s => (p.sendNgram s).1) = SparsePackets.ngramStaged := by
    funext s
    rw [p.sendNgram_exact, p.packets.ngram]
  simp only [SparsePrograms.run, p.packets.matched, if_true, SparsePackets.staged, value, ngram,
    funext p.packets.clear]

def SparsePrograms.landing (p : SparsePrograms) : SparseLanding.Kernels :=
  { candidateCalls.sparseLanding with
    serve := if SparseLanding.recognized p.fetchArtifacts then SparseLanding.directServe else SparseLanding.stagedServe
    land := if SparseLanding.recognized p.fetchArtifacts then SparseLanding.directLand else SparseLanding.stagedLand
    moments := if MomentReplay.recognized p.momentArtifacts then MomentReplay.selected else MomentReplay.staged
    ngramPrepare := p.adam.prepareKernel
    ngramUpdate := p.adam.rowKernel
    ngramOwnedWords := p.scratch.ownedWords
    ngramPeerWords := p.adam.peerWords
    valueNormalize := p.value.normalizer
    valueEvent := ValueAdam.Elementwise.checkedEvent p.valueNative }

theorem SparsePrograms.landing_eq (p : SparsePrograms) : p.landing = candidateCalls.sparseLanding := by
  have valueEvent : ValueAdam.Elementwise.checkedEvent p.valueNative = candidateCalls.sparseLanding.valueEvent := by
    funext pull indices order arithmetic launch memory
    rw [ValueAdam.Elementwise.checked_event_eq]
    change _ = ValueAdam.event (ValueAdam.clearSelected memory.prior) pull indices order arithmetic launch memory
    rw [ValueAdam.workspace_normalization_equal] <;> rfl
  simp only [SparsePrograms.landing, valueEvent, p.fetch.matched, p.moments.matched, if_true,
    p.adam.prepareKernel_eq, p.adam.rowKernel_eq, p.scratch.ownedWords_eq, p.adam.peerWords_eq, p.value.normalizer_eq]
  rfl

def SparsePrograms.accumulateNgram (p : SparsePrograms) : NgramAccumulation.Accumulator :=
  if NgramAccumulation.recognized p.ngramAccumIR p.ngramAccumPTX then NgramAccumulation.online
  else NgramAccumulation.delayed

theorem SparsePrograms.accumulateNgram_eq (p : SparsePrograms) :
    p.accumulateNgram = candidateCalls.ngramAccumulation := by
  simp only [SparsePrograms.accumulateNgram, p.ngramAccum.matched, if_true, candidateCalls]

def SparsePrograms.accumulateValue (p : SparsePrograms) : ValueEmbedAccum.Accumulate :=
  if ValueEmbedAccum.recognized p.valueAccumArtifacts then ValueEmbedAccum.fused else ValueEmbedAccum.staged

theorem SparsePrograms.accumulateValue_eq (p : SparsePrograms) :
    p.accumulateValue = candidateCalls.valueAccumulation := by
  simp only [SparsePrograms.accumulateValue, p.valueAccum.matched, if_true, candidateCalls]

structure Kernels where
  sparse : Option SparsePrograms := none
  attention : Option AttentionExecution.Programs := none
  yarn : Option YarnExecution.Programs := none
  optimizer : Option OptimizerPrograms := none
  tail : Option TailPrograms := none
  qkvRefresh : Option QKVRefresh.Programs := none
  trainingCE : TrainingCEExecution.Programs
  ceTranspose : CETranspose.Program
  validationCE : ValidationProjectionBinding.Programs
  graphStages : Option GraphStage.Programs := none
  sampledGather : Option SampledInverse.Programs := none
  headActivation : Option HeadActivationBinding.Program := none
  sampledDensify : Option Densify.Programs := none
  mlpWeight : Option MLPWeights.Program := none
  headRefresh : Option LMHead.Program := none
  tiedTranspose : Option Bf16Transpose.Programs := none
  muddCoefficients : Option MuddCoefficients.Binding.Program := none
  muddNorm : Option MuddNorm.Binding.Program := none
  muddPair : Option MuddBackward.Binding.Pair := none
  muddStorage : List MuddActivationStorage.Binding.Program := []
  muddResidual : List MuddResidualStorage.Binding.Program := []
  muddGradients : List MuddGradientGrouping.Binding.Program := []
  muddPackingIR : List MuddGradientPacking.Binding.Program := []
  muddFusedPackingIR : List MuddFusedPacking.Binding.Program := []
  muddSinglePackingIR : List MuddFusedPacking.Binding.SingleProgram := []
  muddDirectPackingIR : List (MuddFusedPacking.Binding.DirectProgram false) := []
  muddFullPackingIR : List (MuddFusedPacking.Binding.DirectProgram true) := []
  muddMixedPackingIR : List (MuddFusedPacking.Binding.MixedProgram false) := []
  muddMixedFullPackingIR : List (MuddFusedPacking.Binding.MixedProgram true) := []

def Kernels.graphStage (kernels : Kernels) : GraphStage.Kernels :=
  { candidateCalls.graphStage with
    copy := (kernels.graphStages.map GraphStage.Programs.copy).getD candidateCalls.graphStage.copy }

theorem Kernels.graphStage_eq (kernels : Kernels) : kernels.graphStage = candidateCalls.graphStage := by
  unfold Kernels.graphStage
  cases kernels.graphStages with
  | none => rfl
  | some programs =>
    simp only [Option.map_some, Option.getD_some, GraphStage.Programs.copy_eq]
    rw [← GraphStage.copy_equal]
    rfl

def Kernels.norm (kernels : Kernels) : MuddNorm.TrainingRun :=
  match kernels.muddPair, kernels.muddNorm with
  | some pair, _ => pair.forward.training
  | none, some program => program.training
  | none, none => candidateCalls.muddNorm

def Kernels.normBackward (kernels : Kernels) : MuddBackward.BatchRun :=
  (kernels.muddPair.map MuddBackward.Binding.Pair.run).getD candidateCalls.muddBackward

theorem Kernels.norm_eq (kernels : Kernels) : kernels.norm = candidateCalls.muddNorm := by
  unfold Kernels.norm
  split
  · exact ‹MuddBackward.Binding.Pair›.forward.training_eq
  · exact ‹MuddNorm.Binding.Program›.training_eq
  · rfl

theorem Kernels.normBackward_eq (kernels : Kernels) :
    kernels.normBackward = candidateCalls.muddBackward := by
  unfold Kernels.normBackward
  cases kernels.muddPair with
  | none => rfl
  | some pair => exact pair.run_eq

def Kernels.activationStorage (kernels : Kernels) : MuddActivationStorage.Read :=
  (kernels.muddStorage.head?.map MuddActivationStorage.Binding.Program.read).getD candidateCalls.muddActivation

theorem Kernels.activationStorage_eq (kernels : Kernels) :
    kernels.activationStorage = candidateCalls.muddActivation := by
  unfold Kernels.activationStorage
  cases kernels.muddStorage with
  | nil => rfl
  | cons program rest => exact program.read_eq.trans MuddActivationStorage.read_eq.symm

def Kernels.residual (kernels : Kernels) : MuddResidualStorage.Run := fun rows packet =>
  match kernels.muddResidual.find? (fun p => p.rows == rows && p.block == packet.block) with
  | some program => program.run rows packet
  | none => candidateCalls.muddResidual rows packet

theorem Kernels.residual_eq (kernels : Kernels) : kernels.residual = candidateCalls.muddResidual := by
  funext rows packet
  unfold Kernels.residual
  split
  · rename_i program _
    rw [program.run_eq]
    rfl
  · rfl

def Kernels.gradients (kernels : Kernels) : MuddGradientGrouping.Run := fun rows planes packet =>
  match kernels.muddGradients.find? (fun p => p.rows == rows && p.planes == planes && p.first == packet.first &&
      p.tile == packet.tile && p.warps == packet.warps) with
  | some program => program.run rows planes packet
  | none => MuddGradientGrouping.Program.separate.run rows planes packet

theorem Kernels.gradients_eq (kernels : Kernels) : kernels.gradients = candidateCalls.muddGradients := by
  funext rows planes packet
  unfold Kernels.gradients
  split
  · rename_i program _
    rw [program.run_eq]
    rfl
  · rw [← MuddGradientGrouping.grouped_run_eq]
    rfl

def Kernels.packing (kernels : Kernels) : MuddGradientPacking.Run := fun rows packet =>
  if kernels.muddPackingIR.any (·.rows == rows) then
    match kernels.muddGradients.find? (fun p => p.hoistedPacking && p.rows == rows &&
        p.tile == packet.tile && p.warps == packet.warps) with
    | some program => program.runPacking rows packet
    | none => MuddGradientPacking.Program.interleaved.run rows packet
  else MuddGradientPacking.Program.interleaved.run rows packet

theorem Kernels.packing_eq (kernels : Kernels) : kernels.packing = candidateCalls.muddPacking := by
  funext rows packet
  unfold Kernels.packing
  split
  · split
    · rename_i program _
      rw [program.runPacking_eq]
      rfl
    · rw [← MuddGradientPacking.run_eq]
      rfl
  · rw [← MuddGradientPacking.run_eq]
    rfl

private def matchesMixed {scaled : Bool} {rows : Nat} (p : MuddFusedPacking.Inputs rows)
    (ir : MuddFusedPacking.Binding.MixedProgram scaled) : Bool :=
  let warpsFor := MuddFusedPacking.Binding.scheduleWarps
  ir.rows == rows && ir.groupTile == p.group.tile && ir.zeroTile == p.zeroTile &&
    p.group.warps == warpsFor ir.groupTile && p.zeroWarps == warpsFor ir.zeroTile

def Kernels.fusedPacking (kernels : Kernels) : MuddFusedPacking.Run := fun rows p =>
  let direct := fun planes tile warps => kernels.muddDirectPackingIR.any (fun ir =>
    ir.rows == rows && ir.planes == planes && ir.tile == tile && ir.warps == warps)
  let supported := fun planes tile warps => kernels.muddFusedPackingIR.any (fun ir =>
    ir.rows == rows && ir.planes == planes && ir.tile == tile && ir.warps == warps)
  let directSingle := direct 10 p.zeroTile p.zeroWarps
  let directGroup := direct 9 p.group.tile p.group.warps
  let directZero := direct 1 p.zeroTile p.zeroWarps
  let single := kernels.muddSinglePackingIR.any (fun ir =>
    ir.rows == rows && ir.tile == p.zeroTile && ir.warps == p.zeroWarps)
  if kernels.muddMixedPackingIR.any (matchesMixed p) then
    MuddFusedPacking.Program.single.runSelected p true false
  else if p.group.tile == p.zeroTile && p.group.warps == p.zeroWarps &&
      (directSingle || single) then
    MuddFusedPacking.Program.single.runSelected p directSingle false
  else if (directGroup || supported 9 p.group.tile p.group.warps) &&
      (directZero || supported 1 p.zeroTile p.zeroWarps) then
    MuddFusedPacking.Program.fused.runSelected p directGroup directZero
  else
    MuddFusedPacking.Program.separate.run rows p

theorem Kernels.fusedPacking_eq (kernels : Kernels) :
    kernels.fusedPacking = candidateCalls.muddFusedPacking := by
  funext rows p
  unfold Kernels.fusedPacking
  simp only [MuddFusedPacking.Program.runSelected_eq, MuddFusedPacking.single_run_eq, ← MuddFusedPacking.run_eq,
    ite_self, candidateCalls]

def Kernels.fullPacking (kernels : Kernels) : MuddFullPacking.Run := fun rows p project =>
  let supported := fun planes tile warps => kernels.muddFullPackingIR.any (fun ir =>
    ir.rows == rows && ir.planes == planes && ir.tile == tile && ir.warps == warps)
  let ready := if p.group.tile == p.zeroTile && p.group.warps == p.zeroWarps then
      supported 10 p.zeroTile p.zeroWarps
    else supported 9 p.group.tile p.group.warps && supported 1 p.zeroTile p.zeroWarps
  if kernels.muddMixedFullPackingIR.any (matchesMixed p) then
    MuddFullPacking.fullJoined p project
  else MuddFullPacking.fullOptimized p project ready

theorem Kernels.fullPacking_eq (kernels : Kernels) :
    kernels.fullPacking = candidateCalls.muddFullPacking := by
  funext rows p project
  unfold Kernels.fullPacking
  change (if _ then MuddFullPacking.fullJoined p project else MuddFullPacking.fullOptimized p project _) =
    MuddFullPacking.fullOptimized p project true
  simp only [MuddFullPacking.full_joined_eq, MuddFullPacking.full_optimized_eq, ite_self]

def Kernels.calls (kernels : Kernels) : PatchedCalls :=
  {candidateCalls with
    sparsePackets := (kernels.sparse.map SparsePrograms.run).getD candidateCalls.sparsePackets
    sparseLanding := (kernels.sparse.map SparsePrograms.landing).getD candidateCalls.sparseLanding
    lookup := (kernels.sparse.map fun p => NgramEmbedding.lookup p.embeddingArtifacts).getD candidateCalls.lookup
    backward := (kernels.sparse.map fun p => NgramEmbedding.backward p.embeddingArtifacts).getD candidateCalls.backward
    ngramAccumulation := (kernels.sparse.map SparsePrograms.accumulateNgram).getD candidateCalls.ngramAccumulation
    valueAccumulation := (kernels.sparse.map SparsePrograms.accumulateValue).getD candidateCalls.valueAccumulation
    trainingCE := kernels.trainingCE.run
    ceTranspose := kernels.ceTranspose.run
    ropeForward := (kernels.attention.map AttentionExecution.Programs.forwardStore).getD candidateCalls.ropeForward
    ropeBackward := (kernels.attention.map AttentionExecution.Programs.backwardStores).getD candidateCalls.ropeBackward
    ropeBackwardNarrow :=
      (kernels.attention.map AttentionExecution.Programs.narrowStores).getD candidateCalls.ropeBackwardNarrow
    byteTranspose := (kernels.attention.map AttentionExecution.Programs.transposeBytes).getD candidateCalls.byteTranspose
    yarnTables := (kernels.yarn.map YarnExecution.Programs.compactTables).getD candidateCalls.yarnTables
    wideYarnTables := (kernels.yarn.map YarnExecution.Programs.stationaryTables).getD candidateCalls.wideYarnTables
    validationCE := ValidationProjectionBinding.runBatch kernels.validationCE
    headActivation := (kernels.headActivation.map HeadActivationBinding.Program.run).getD HeadActivation.fused
    sampledGather := (kernels.sampledGather.map SampledInverse.Programs.run).getD candidateCalls.sampledGather
    sampledDensify := (kernels.sampledDensify.map Densify.Programs.run).getD candidateCalls.sampledDensify
    bf16Transpose := (kernels.tiedTranspose.map Bf16Transpose.Programs.run).getD candidateCalls.bf16Transpose
    headRefresh := (kernels.headRefresh.map LMHead.Program.run).getD candidateCalls.headRefresh
    weightRefresh := (kernels.mlpWeight.map MLPWeights.Program.run).getD candidateCalls.weightRefresh
    muddCoefficients := (kernels.muddCoefficients.map MuddCoefficients.Binding.Program.run).getD
      candidateCalls.muddCoefficients
    tailUpdate := (kernels.tail.map TailPrograms.update).getD candidateCalls.tailUpdate
    tailRestore := (kernels.tail.map TailPrograms.restore).getD candidateCalls.tailRestore
    qkvAmax := (kernels.qkvRefresh.map QKVRefresh.Programs.scan).getD candidateCalls.qkvAmax
    qkvScale := (kernels.qkvRefresh.map QKVRefresh.Programs.scale).getD candidateCalls.qkvScale
    qkvPack := (kernels.qkvRefresh.map QKVRefresh.Programs.pack).getD candidateCalls.qkvPack
    anvilRails := (kernels.optimizer.map fun p => p.rails.run).getD candidateCalls.anvilRails
    anvilNormalize := (kernels.optimizer.map fun p => p.normalize.run p.trace).getD candidateCalls.anvilNormalize
    anvilGram := (kernels.optimizer.map fun p => p.gram.run).getD candidateCalls.anvilGram
    anvilInitial := (kernels.optimizer.map fun p => p.cooperative.initialize).getD candidateCalls.anvilInitial
    anvilRound := (kernels.optimizer.map fun p => p.cooperative.run).getD candidateCalls.anvilRound
    anvilEpilogue := (kernels.optimizer.map fun p => p.epilogue.run).getD candidateCalls.anvilEpilogue
    muddNorm := kernels.norm
    muddBackward := kernels.normBackward
    muddActivation := kernels.activationStorage
    muddResidual := kernels.residual
    muddGradients := kernels.gradients
    muddPacking := kernels.packing
    muddFusedPacking := kernels.fusedPacking
    muddFullPacking := kernels.fullPacking
    graphStage := kernels.graphStage
    graphLaunch := (kernels.graphStages.map GraphStage.Programs.run).getD GraphStage.referenceLaunch
    graphCopyLaunch := (kernels.graphStages.map GraphStage.Programs.copyLaunch).getD GraphStage.referenceCopyLaunch}

theorem Kernels.calls_eq (kernels : Kernels) : kernels.calls = referenceCalls := by
  have train : kernels.trainingCE.run = candidateCalls.trainingCE :=
    kernels.trainingCE.run_eq.trans TrainingCE.specialized_equal.symm
  have validation : ValidationProjectionBinding.runBatch kernels.validationCE = candidateCalls.validationCE :=
    ValidationProjectionBinding.runBatch_eq kernels.validationCE
  have transpose : kernels.ceTranspose.run = candidateCalls.ceTranspose :=
    kernels.ceTranspose.run_eq.trans (congrArg PatchedCalls.ceTranspose calls_equal).symm
  have graph : (kernels.graphStages.map GraphStage.Programs.run).getD GraphStage.referenceLaunch =
      candidateCalls.graphLaunch := by
    cases kernels.graphStages with
    | none => rfl
    | some programs => exact programs.run_eq
  have copy : (kernels.graphStages.map GraphStage.Programs.copyLaunch).getD GraphStage.referenceCopyLaunch =
      candidateCalls.graphCopyLaunch := by
    cases kernels.graphStages with
    | none => rfl
    | some programs => exact programs.copyLaunch_eq
  have gather : (kernels.sampledGather.map SampledInverse.Programs.run).getD candidateCalls.sampledGather =
      candidateCalls.sampledGather := by
    cases kernels.sampledGather with
    | none => rfl
    | some programs => exact programs.run_eq.trans SampledInverse.gather_equal.symm
  have activation : (kernels.headActivation.map HeadActivationBinding.Program.run).getD HeadActivation.fused =
      candidateCalls.headActivation := by
    cases kernels.headActivation with
    | none => rfl
    | some program => exact program.run_eq.trans HeadActivation.fused_equal.symm
  have dense : (kernels.sampledDensify.map Densify.Programs.run).getD candidateCalls.sampledDensify =
      candidateCalls.sampledDensify := by
    cases kernels.sampledDensify with
    | none => rfl
    | some programs => exact programs.run_eq.trans (congrArg PatchedCalls.sampledDensify calls_equal).symm
  have tied : (kernels.tiedTranspose.map Bf16Transpose.Programs.run).getD candidateCalls.bf16Transpose =
      candidateCalls.bf16Transpose := by
    cases kernels.tiedTranspose with
    | none => rfl
    | some programs => exact programs.run_eq.trans (congrArg PatchedCalls.bf16Transpose calls_equal).symm
  have weight : (kernels.mlpWeight.map MLPWeights.Program.run).getD candidateCalls.weightRefresh =
      candidateCalls.weightRefresh := by
    cases kernels.mlpWeight with
    | none => rfl
    | some program => exact program.run_eq.trans (congrArg PatchedCalls.weightRefresh calls_equal).symm
  have headRefresh : (kernels.headRefresh.map LMHead.Program.run).getD candidateCalls.headRefresh =
      candidateCalls.headRefresh := by
    cases kernels.headRefresh with
    | none => rfl
    | some program => exact program.run_eq.trans (congrArg PatchedCalls.headRefresh calls_equal).symm
  have mudd : (kernels.muddCoefficients.map MuddCoefficients.Binding.Program.run).getD
      candidateCalls.muddCoefficients = candidateCalls.muddCoefficients := by
    cases kernels.muddCoefficients with
    | none => rfl
    | some program => exact program.run_eq.trans MuddCoefficients.fused_equal.symm
  have tailUpdate : (kernels.tail.map TailPrograms.update).getD candidateCalls.tailUpdate =
      candidateCalls.tailUpdate := by
    cases kernels.tail with
    | none => rfl
    | some p => exact p.update_eq.trans (congrArg PatchedCalls.tailUpdate calls_equal).symm
  have tailRestore : (kernels.tail.map TailPrograms.restore).getD candidateCalls.tailRestore =
      candidateCalls.tailRestore := by
    cases kernels.tail with
    | none => rfl
    | some p => exact p.restore_eq.trans (congrArg PatchedCalls.tailRestore calls_equal).symm
  have qkvScan : (kernels.qkvRefresh.map QKVRefresh.Programs.scan).getD candidateCalls.qkvAmax =
      candidateCalls.qkvAmax := by
    cases kernels.qkvRefresh with
    | none => rfl
    | some p => exact p.scan_eq.trans (congrArg PatchedCalls.qkvAmax calls_equal).symm
  have qkvScale : (kernels.qkvRefresh.map QKVRefresh.Programs.scale).getD candidateCalls.qkvScale =
      candidateCalls.qkvScale := by
    cases kernels.qkvRefresh with
    | none => rfl
    | some p => exact p.scale_eq.trans (congrArg PatchedCalls.qkvScale calls_equal).symm
  have qkvPack : (kernels.qkvRefresh.map QKVRefresh.Programs.pack).getD candidateCalls.qkvPack =
      candidateCalls.qkvPack := by
    cases kernels.qkvRefresh with
    | none => rfl
    | some p => exact p.pack_eq.trans (congrArg PatchedCalls.qkvPack calls_equal).symm
  have railsRun : (kernels.optimizer.map fun p => p.rails.run).getD candidateCalls.anvilRails =
      candidateCalls.anvilRails := by
    cases kernels.optimizer with
    | none => rfl
    | some p => exact p.rails.run_eq.trans (congrArg PatchedCalls.anvilRails calls_equal).symm
  have normalizeRun : (kernels.optimizer.map fun p => p.normalize.run p.trace).getD candidateCalls.anvilNormalize =
      candidateCalls.anvilNormalize := by
    cases kernels.optimizer with
    | none => rfl
    | some p => exact (p.normalize.run_eq p.trace).trans (congrArg PatchedCalls.anvilNormalize calls_equal).symm
  have gramRun : (kernels.optimizer.map fun p => p.gram.run).getD candidateCalls.anvilGram =
      candidateCalls.anvilGram := by
    cases kernels.optimizer with
    | none => rfl
    | some p => exact p.gram.run_eq.trans (congrArg PatchedCalls.anvilGram calls_equal).symm
  have initialRun : (kernels.optimizer.map fun p => p.cooperative.initialize).getD candidateCalls.anvilInitial =
      candidateCalls.anvilInitial := by
    cases kernels.optimizer with
    | none => rfl
    | some p =>
      exact p.cooperative.initialize_eq.trans (congrArg PatchedCalls.anvilInitial calls_equal).symm
  have roundRun : (kernels.optimizer.map fun p => p.cooperative.run).getD candidateCalls.anvilRound =
      candidateCalls.anvilRound := by
    cases kernels.optimizer with
    | none => rfl
    | some p =>
      exact p.cooperative.run_eq.trans (congrArg PatchedCalls.anvilRound calls_equal).symm
  have epilogueRun : (kernels.optimizer.map fun p => p.epilogue.run).getD candidateCalls.anvilEpilogue =
      candidateCalls.anvilEpilogue := by
    cases kernels.optimizer with
    | none => rfl
    | some p => exact p.epilogue.run_eq.trans (congrArg PatchedCalls.anvilEpilogue calls_equal).symm
  have sparse : (kernels.sparse.map SparsePrograms.run).getD candidateCalls.sparsePackets =
      candidateCalls.sparsePackets := by
    cases kernels.sparse with
    | none => rfl
    | some p => exact p.run_eq.trans SparsePackets.kernels_equal.symm
  have landing : (kernels.sparse.map SparsePrograms.landing).getD candidateCalls.sparseLanding =
      candidateCalls.sparseLanding := by
    cases kernels.sparse with
    | none => rfl
    | some p => exact p.landing_eq
  have ngramAccum : (kernels.sparse.map SparsePrograms.accumulateNgram).getD candidateCalls.ngramAccumulation =
      candidateCalls.ngramAccumulation := by
    cases kernels.sparse with
    | none => rfl
    | some p => exact p.accumulateNgram_eq
  have valueAccum : (kernels.sparse.map SparsePrograms.accumulateValue).getD candidateCalls.valueAccumulation =
      candidateCalls.valueAccumulation := by
    cases kernels.sparse with
    | none => rfl
    | some p => exact p.accumulateValue_eq
  have lookup : (kernels.sparse.map fun p => NgramEmbedding.lookup p.embeddingArtifacts).getD candidateCalls.lookup =
      candidateCalls.lookup := by
    cases kernels.sparse with
    | none => rfl
    | some p => exact NgramEmbedding.lookup_eq p.embedding
  have embedBack : (kernels.sparse.map fun p => NgramEmbedding.backward p.embeddingArtifacts).getD candidateCalls.backward =
      candidateCalls.backward := by
    cases kernels.sparse with
    | none => rfl
    | some p => exact NgramEmbedding.backward_eq p.embedding
  have rope : (kernels.attention.map AttentionExecution.Programs.forwardStore).getD candidateCalls.ropeForward =
      candidateCalls.ropeForward := by
    cases kernels.attention with
    | none => rfl
    | some p => exact p.forwardStore_eq
  have ropeBack : (kernels.attention.map AttentionExecution.Programs.backwardStores).getD candidateCalls.ropeBackward =
      candidateCalls.ropeBackward := by
    cases kernels.attention with
    | none => rfl
    | some p => exact p.backwardStores_eq
  have ropeNarrow :
      (kernels.attention.map AttentionExecution.Programs.narrowStores).getD candidateCalls.ropeBackwardNarrow =
        candidateCalls.ropeBackwardNarrow := by
    cases kernels.attention with
    | none => rfl
    | some p => exact p.narrowStores_eq
  have bytes : (kernels.attention.map AttentionExecution.Programs.transposeBytes).getD candidateCalls.byteTranspose =
      candidateCalls.byteTranspose := by
    cases kernels.attention with
    | none => rfl
    | some p => exact p.transposeBytes_eq
  have compact : (kernels.yarn.map YarnExecution.Programs.compactTables).getD candidateCalls.yarnTables =
      candidateCalls.yarnTables := by
    cases kernels.yarn with
    | none => rfl
    | some p => exact p.compactTables_eq
  have stationary : (kernels.yarn.map YarnExecution.Programs.stationaryTables).getD candidateCalls.wideYarnTables =
      candidateCalls.wideYarnTables := by
    cases kernels.yarn with
    | none => rfl
    | some p => exact p.stationaryTables_eq
  rw [Kernels.calls, sparse, landing, lookup, embedBack, ngramAccum, valueAccum,
    rope, ropeBack, ropeNarrow, bytes, compact, stationary, train, transpose, validation, kernels.graphStage_eq, graph, copy, gather, activation, dense, tied, weight, headRefresh, mudd,
    tailUpdate, tailRestore, qkvScan, qkvScale, qkvPack, railsRun, normalizeRun, gramRun, initialRun, roundRun, epilogueRun,
    kernels.norm_eq, kernels.normBackward_eq, kernels.activationStorage_eq, kernels.residual_eq, kernels.gradients_eq,
    kernels.packing_eq, kernels.fusedPacking_eq, kernels.fullPacking_eq]
  exact calls_equal

/-- The explicit head returns both saved FP8 operands, the row losses,
and the two BF16 gradients, with dense stores on the sampled path. -/
theorem head_forward_backward_eq (kernels : Kernels) (packet : TrainingHead.ForwardPacket)
    (shape : CETranspose.Shape) (columns : (shape.operand .gradient).cols = packet.width.columns)
    (tokens : (shape.operand .gradient).rows = packet.tokens) (transposeFast densifyFast : Bool) :
    let run := fun (calls : PatchedCalls) =>
      let result := packet.run (TrainingHead.forward calls.headActivation calls.trainingCE calls.muddResidual)
      let saved : TrainingHead.Saved (shape.operand .gradient).rows packet.width :=
        tokens.symm ▸ (packet.tape result.saved).saved
      (result, TrainingHead.backwardWithNorm calls.ceTranspose calls.sampledDensify
        calls.muddBackward calls.muddActivation packet.width packet.gemm
        transposeFast densifyFast shape columns saved)
    run kernels.calls = run referenceCalls := by
  rw [kernels.calls_eq]

/-- The captured backward can rebuild its E4M3 operand from the two
FP32 buffers saved by the checked forward launch. -/
theorem mudd_head_recomputation_eq (kernels : Kernels) (program : MuddNorm.Binding.Program)
    (native : MuddNorm.Native) (packet : MuddNorm.Packet) (width : TrainingCE.Width)
    (shape : CETranspose.Shape) (columns : (shape.operand .gradient).cols = width.columns)
    (rows : (shape.operand .gradient).rows = packet.shape.rows) (gemm : TrainingHead.Backend)
    (saved : TrainingHead.Saved (shape.operand .gradient).rows width) (transposeFast densifyFast : Bool) :
    let output := program.run native packet
    let backward := fun (activation : TrainingHead.View packet.shape.rows 768 UInt8) =>
      TrainingHead.backward kernels.calls.ceTranspose kernels.calls.sampledDensify width gemm transposeFast
        densifyFast shape columns {saved with activation := rows.symm ▸ activation}
    backward (output.reencoded native) = backward output.activations := by
  dsimp only
  rw [program.run_eq, MuddNorm.RowProgram.reencoded_eq]

inductive Program
  | baseline (source : SourceImage)
  | optimized (source : SourceImage) (kernels : Kernels)

def Program.source : Program → SourceProgram
  | .baseline source => ⟨.reference, source⟩
  | .optimized source _ => ⟨.candidate, source⟩

def Program.calls : Program → PatchedCalls
  | .baseline _ => referenceCalls
  | .optimized _ kernels => kernels.calls

def Program.run {State Value : Type} (p : Program) (w : World State Value)
    (input : TrainingInput) : Option (Result State) :=
  runWith p.source w input p.calls

inductive Statement where
  | source (stmt : TrainingStatement)
  | trainingCE (width : TrainingCE.Width) (predictions : Fin 4) (ptx : String)
  | ceTranspose (ttgir ptx : String)
  | validationCE (tile : ValidationCE.Tile) (wrapper ptx : String)
  | sparseValue (wireOnly : Bool) (ttgir ptx : String)
  | sparseNgram (clearPeer : Bool) (ttgir ptx : String)
  | sparseValueClear (ttgir ptx : String)
  | sparseNgramOwnClear (ttgir ptx : String)
  | ngramAdamPack (prepared : Bool) (code : NgramAdam.Code)
  | ngramAdamUpdate (skipZero : Bool) (code : NgramAdam.Code)
  | ngramAdamScatter (peer tiled : Bool) (code : NgramAdam.Code)
  | valueAdamReference (code : ValueAdam.Code)
  | valueAdamNativeUpdate (period : ValueAdam.Elementwise.Period) (tile : ValueAdam.Tile)
      (fused clearAll : Bool) (ptx : String)
  | valueAdamScatter (own marked : Bool) (code : ValueAdam.Code)
  | valueAdamScale (code : ValueAdam.Code)
  | valueAdamLocal (direct marked : Bool) (ptx : String)
  | valueAdamClear (code : ValueAdam.Code)
  | sparseRows (land : Bool) (ttgir ptx : String)
  | ngramMomentReplay (grouped skipZero : Bool) (ttgir ptx : String)
  | ngramGradientAccumulation (ttgir ptx : String)
  | ngramEmbedding (fullRows forward : Bool) (ptx : String)
  | ropeForward (length : RopeShift.Length) (shifted : Bool) (ttgir ptx : String)
  | ropeBackward (length : RopeShift.Length) (trimmed : Bool) (ttgir ptx : String)
  | ropeNarrow (shape : RopeBackward.Narrow.Shape) (length : RopeShift.Length) (vector : Bool) (ttgir ptx : String)
  | byteTransposeReference (ttir : String)
  | byteTransposeHopper (ttgir ptx : String)
  | yarnCompact (layout : YarnTable.Layout) (phase : YarnTable.Phase) (length : RopeShift.Length) (ttgir ptx : String)
  | yarnStationary (phase : YarnTable.Phase) (length : RopeShift.Length) (ttgir ptx : String)
  | valueGradientAccumulation (fused debug : Bool) (code : ValueEmbedAccum.Code)
  | graphCopy (shape : GraphStage.Shape) (ptx : String)
  | graphStage (shape : GraphStage.Shape) (member : Bool) (ptx : String)
  | sampledGather (count : SampledRows.Count) (ttgir ptx : String)
  | headActivation (ptx : String)
  | sampledDensify (count : SampledRows.Count) (ttgir ptx : String)
  | tiedTranspose (mode : Bf16Transpose.Mode) (ttgir ptx : String)
  | mlpWeightScale (ttir ptx : String)
  | mlpWeightPack (ttgir ptx : String)
  | headRefresh (ttgir ptx : String)
  | tailLerp (artifacts : TailLerp.Artifacts)
  | tailNorm (ttgir ptx : String)
  | qkvGroupedScan (ttir ptx : String)
  | qkvGroupedPack (ttgir ptx : String)
  | qkvScale (grouped : Bool) (ttir ptx : String)
  | qkvNativeScan (width : QKVLaunch.Width) (artifacts : QKVNative.ScanArtifacts)
  | qkvNativePack (width : QKVLaunch.Width) (referenceTTIR ttgir ptx : String)
  | anvilRails (bank : AnvilEpilogue.Bank) (ttgir ptx : String)
  | anvilTrace (bank : AnvilEpilogue.Bank) (ttgir ptx : String)
  | anvilNormalize (bank : AnvilEpilogue.Bank) (ttgir ptx : String)
  | anvilGram (shape : AnvilGram.Shape) (artifacts : AnvilGram.Artifacts)
  | anvilCooperativeSeed (bank : AnvilEpilogue.Bank) (ttgir ptx : String)
  | anvilCooperativeInitial (bank : AnvilEpilogue.Bank) (ttgir ptx : String)
  | anvilCooperativeRound (bank : AnvilEpilogue.Bank) (ttgir ptx : String)
  | anvilEpilogue (bank : AnvilEpilogue.Bank) (tile : AnvilEpilogue.Tile)
      (artifacts : AnvilEpilogue.Binding.Artifacts)
  | muddCoefficients (ptx : String)
  | muddNorm (ptx : String)
  | muddCachedForward (ptx : String)
  | muddCachedBackward (reencode : Bool) (ptx : String)
  | muddRetainedBackward (ptx : String)
  | muddActivationStorage (referenceTTIR candidateTTIR : String)
  | muddResidual (rows block : Nat) (referenceTTIR candidateTTIR : String)
  | muddGradients (rows planes first tile warps : Nat) (hoistedPacking : Bool)
      (referenceTTIR candidateTTIR referenceTTGIR candidateTTGIR : String)
  | muddPackedLoad (rows tile warps : Nat) (referenceTTIR candidateTTIR referenceTTGIR candidateTTGIR : String)
  | muddFusedPacking (rows planes first tile warps : Nat)
      (referenceTTIR candidateTTIR referenceTTGIR candidateTTGIR : String)
  | muddSinglePacking (rows tile warps : Nat) (ttir ttgir : MuddFusedPacking.Binding.SingleIR)
  | muddDirectPacking (scaled : Bool) (rows planes first tile warps : Nat) (group zero fused : String)
      (ptx : MuddFusedPacking.Binding.DirectPTX)
  | muddMixedPacking (scaled : Bool) (rows groupTile zeroTile : Nat) (group zero fused ptx : String)

def validationTiles : List ValidationCE.Tile :=
  [.x1r1024, .x1r512, .x16r32, .x64r64, .x64r4, .x8r512, .x8r256]

private def OptimizerPrograms.body (p : OptimizerPrograms) : List Statement :=
  let banks := [AnvilEpilogue.Bank.qk, .vo, .mlp]
  let shapes := [AnvilGram.Shape.qkGram, .voGram, .mlpGram, .qkPolynomial, .widePolynomial]
  (banks.map fun bank =>
    let code := p.rails bank
    .anvilRails bank code.ttgir code.ptx) ++
  (banks.map fun bank =>
    let code := p.trace bank
    .anvilTrace bank code.ttgir code.ptx) ++
  (banks.map fun bank =>
    let code := p.normalize bank
    .anvilNormalize bank code.ttgir code.ptx) ++
  (shapes.map fun shape => .anvilGram shape (p.gram shape).artifacts) ++
  (banks.flatMap fun bank =>
    let code := p.cooperative bank
    [.anvilCooperativeSeed bank code.seed.ttgir code.seed.ptx,
      .anvilCooperativeInitial bank code.initial.ttgir code.initial.ptx,
     .anvilCooperativeRound bank code.round.ttgir code.round.ptx]) ++
  banks.flatMap fun bank =>
    [AnvilEpilogue.Tile.v1024, .v512].map fun tile =>
      .anvilEpilogue bank tile (p.epilogue bank tile).artifacts

private def qkvBody (p : QKVRefresh.Programs) : List Statement :=
  let code := p.grouped.artifacts
  [.qkvGroupedScan code.scanIR code.scanPTX,
   .qkvScale true code.scaleIR code.scalePTX,
   .qkvGroupedPack code.packIR code.packPTX,
   .qkvScale false p.perWidthScale.ttir p.perWidthScale.ptx] ++
  [QKVLaunch.Width.narrow, .halfValue, .wide].flatMap fun width =>
    let native := p.perWidth width
    [.qkvNativeScan width native.scan.artifacts, .qkvNativePack width native.referencePack native.packIR native.packPTX]

private def attentionBody (p : AttentionExecution.Programs) : List Statement :=
  [.byteTransposeReference p.bytes.artifacts.referenceIR,
   .byteTransposeHopper p.bytes.artifacts.hopperIR p.bytes.artifacts.hopperPTX] ++
  YarnExecution.lengths.flatMap fun n =>
    let f := (p.forward n).artifacts
    [Statement.ropeForward n false f.referenceIR f.referencePTX,
     .ropeForward n true f.shiftedIR f.shiftedPTX] ++
    ((p.backward n).map fun program =>
      let b := program.artifacts
      [Statement.ropeBackward n false b.referenceIR b.referencePTX,
       .ropeBackward n true b.trimmedIR b.trimmedPTX]).getD [] ++
    [RopeBackward.Narrow.Shape.paired, .halfValue].flatMap fun shape =>
      ((p.narrow shape n).map fun program =>
        let b := program.artifacts
        [Statement.ropeNarrow shape n false b.referenceIR b.referencePTX,
         .ropeNarrow shape n true b.vectorIR b.vectorPTX]).getD []

private def yarnBody (p : YarnExecution.Programs) : List Statement :=
  YarnExecution.lengths.flatMap fun n =>
    let forward := p.stationary.forward n
    [Statement.yarnStationary .forward n forward.ir forward.ptx] ++
    ((p.stationary.backward n).map fun a => Statement.yarnStationary .backward n a.ir a.ptx).toList ++
    [YarnTable.Layout.wide, .paired, .narrow].flatMap fun layout =>
      let forward := (p.compact layout).forward n
      [Statement.yarnCompact layout .forward n forward.ir forward.ptx] ++
      (((p.compact layout).backward n).map fun a => Statement.yarnCompact layout .backward n a.ir a.ptx).toList

private def nativeValueBody (p : ValueAdam.Elementwise.NativePrograms) : List Statement :=
  [ValueAdam.Elementwise.Period.two, .four].flatMap fun period =>
    [ValueAdam.Tile.x512, .x1024].flatMap fun tile =>
      [.valueAdamNativeUpdate period tile false false (p.reference period tile).1,
       .valueAdamNativeUpdate period tile true false (p.fused period tile false).artifact.1,
       .valueAdamNativeUpdate period tile true true (p.fused period tile true).artifact.1]

private def localValueBody (programs : ValueAdam.LocalRows.Programs) : List Statement :=
  [false, true].flatMap fun direct => [false, true].map fun marked =>
    .valueAdamLocal direct marked (programs.get direct marked).ptx

def Kernels.body (k : Kernels) : List Statement :=
  [.trainingCE .sampled10 2 k.trainingCE.sampled10Two.ptx,
   .trainingCE .sampled10 3 k.trainingCE.sampled10Three.ptx,
   .trainingCE .sampled14 1 k.trainingCE.sampled14.ptx,
   .trainingCE .sampled24 1 k.trainingCE.sampled24.ptx,
   .trainingCE .full 1 k.trainingCE.fullVocab.ptx,
   .ceTranspose k.ceTranspose.ttgir k.ceTranspose.ptx] ++
  (validationTiles.map fun tile =>
    let program := k.validationCE tile
    .validationCE tile program.source program.ptx) ++
  (k.sparse.map fun p =>
    let pack := p.packetArtifacts
    let scratch := p.scratchArtifacts
    let adam := p.adamArtifacts
    let value := p.valueArtifacts
    let fetch := p.fetchArtifacts
    let moments := p.momentArtifacts
    let valueAccum := p.valueAccumArtifacts
    let embedding := p.embeddingArtifacts
    [Statement.sparseValue false pack.valueIR pack.valuePTX,
     .sparseValue true pack.valueWireIR pack.valueWirePTX,
     .sparseNgram false pack.ngramIR pack.ngramPTX,
     .sparseNgram true scratch.peerIR scratch.peerPTX,
     .sparseValueClear pack.clearIR pack.clearPTX,
     .sparseNgramOwnClear scratch.ownIR scratch.ownPTX,
     .ngramAdamPack false adam.pack, .ngramAdamPack true adam.prepare,
     .ngramAdamUpdate false adam.update, .ngramAdamUpdate true adam.updateZero,
     .ngramAdamScatter false false adam.scatterOwn, .ngramAdamScatter true false adam.scatterPeer,
     .ngramAdamScatter true true adam.scatterPeerTiled,
     .valueAdamReference value.reference,
     .valueAdamScatter false false value.direct, .valueAdamScatter false true value.marked,
     .valueAdamScatter true false value.own, .valueAdamScatter true true value.ownMarked,
     .valueAdamScale value.scale, .valueAdamClear value.clear,
     .sparseRows false fetch.serveIR fetch.servePTX, .sparseRows true fetch.landIR fetch.landPTX,
     .ngramMomentReplay false false moments.referenceIR moments.referencePTX,
     .ngramMomentReplay false true moments.fallbackIR moments.fallbackPTX,
     .ngramMomentReplay true false moments.groupedIR moments.groupedPTX,
     .ngramMomentReplay true true moments.groupedZeroIR moments.groupedZeroPTX,
     .ngramGradientAccumulation p.ngramAccumIR p.ngramAccumPTX,
     .ngramEmbedding false true embedding.flatForward,
     .ngramEmbedding false false embedding.flatBackward,
     .ngramEmbedding true true embedding.fullForward,
     .ngramEmbedding true false embedding.fullBackward,
     .valueGradientAccumulation false false valueAccum.reference,
     .valueGradientAccumulation true false valueAccum.four,
     .valueGradientAccumulation false true valueAccum.referenceDebug,
     .valueGradientAccumulation true true valueAccum.fourDebug] ++ nativeValueBody p.valueNative ++ localValueBody p.valueLocal).getD [] ++
  (k.attention.map attentionBody).getD [] ++
  (k.yarn.map yarnBody).getD [] ++
  (k.optimizer.map OptimizerPrograms.body).getD [] ++
  (k.qkvRefresh.map qkvBody).getD [] ++
  (k.tail.map fun p => [Statement.tailLerp p.lerp, .tailNorm p.normIR p.normPTX]).getD [] ++
  (match k.graphStages with
  | none => []
  | some programs => GraphStage.shapes.flatMap fun shape =>
    let program := programs shape
    [.graphCopy shape program.plain.ptx, .graphStage shape false program.hashedPTX,
      .graphStage shape true program.memberPTX]) ++
  (match k.sampledGather with
  | none => []
  | some programs => [SampledRows.Count.first, .second, .third].map fun count =>
    let program := programs count
    .sampledGather count program.ttgir program.ptx) ++
  (k.headActivation.map fun program => Statement.headActivation program.ptx).toList ++
  (match k.sampledDensify with
  | none => []
  | some programs => [SampledRows.Count.first, .second, .third].map fun count =>
    let program := programs count
    .sampledDensify count program.artifacts.hopperIR program.artifacts.hopperPTX) ++
  (match k.tiedTranspose with
  | none => []
  | some programs => [Bf16Transpose.Mode.add, .copy].map fun mode =>
    let artifact := (programs mode).artifacts
    .tiedTranspose mode artifact.hopperIR artifact.hopperPTX) ++
  (match k.mlpWeight with
  | none => []
  | some program => [.mlpWeightScale program.artifacts.scaleIR program.artifacts.scalePTX,
      .mlpWeightPack program.artifacts.packIR program.artifacts.packPTX]) ++
  (k.headRefresh.map fun program => Statement.headRefresh program.artifacts.hopperIR program.artifacts.hopperPTX).toList ++
  (k.muddCoefficients.map fun program => Statement.muddCoefficients program.ptx).toList ++
  (k.muddNorm.map fun program => Statement.muddNorm program.ptx).toList ++
  (match k.muddPair with
  | none => []
  | some pair => [.muddCachedForward pair.forward.ptx,
      .muddCachedBackward false pair.gradient.ptx, .muddCachedBackward true pair.encode.ptx] ++
      (pair.retained.map fun program => Statement.muddRetainedBackward program.ptx).toList) ++
  (k.muddStorage.map fun program => .muddActivationStorage program.referenceTTIR program.candidateTTIR) ++
  (k.muddResidual.map fun program => .muddResidual program.rows program.block program.referenceTTIR program.candidateTTIR) ++
  (k.muddGradients.map fun p => .muddGradients p.rows p.planes p.first p.tile p.warps p.hoistedPacking p.referenceTTIR
    p.candidateTTIR p.referenceTTGIR p.candidateTTGIR) ++
  (k.muddPackingIR.map fun p => .muddPackedLoad p.rows p.tile p.warps p.referenceTTIR p.candidateTTIR
    p.referenceTTGIR p.candidateTTGIR) ++
  (k.muddFusedPackingIR.map fun p => .muddFusedPacking p.rows p.planes p.first p.tile p.warps
    p.referenceTTIR p.candidateTTIR
    p.referenceTTGIR p.candidateTTGIR) ++
  (k.muddSinglePackingIR.map fun p => .muddSinglePacking p.rows p.tile p.warps p.ttir p.ttgir) ++
  (k.muddDirectPackingIR.map fun p =>
    .muddDirectPacking false p.rows p.planes p.first p.tile p.warps p.group p.zero p.fused p.ptx) ++
  (k.muddFullPackingIR.map fun p =>
    .muddDirectPacking true p.rows p.planes p.first p.tile p.warps p.group p.zero p.fused p.ptx) ++
  (k.muddMixedPackingIR.map fun p => .muddMixedPacking false p.rows p.groupTile p.zeroTile p.group p.zero p.fused p.ptx) ++
  (k.muddMixedFullPackingIR.map fun p => .muddMixedPacking true p.rows p.groupTile p.zeroTile p.group p.zero p.fused p.ptx)

def sameRun (a b : Program) : Prop :=
  ∀ (State Value : Type) (w : World State Value) (input : TrainingInput),
    a.run w input = b.run w input

instance : Spec.ProgramSyntax Program where
  Statement := Statement
  Signature := List TrainingIO × List TrainingIO
  signature := fun p => Spec.ProgramSyntax.signature p.source
  body := fun p =>
    (Spec.ProgramSyntax.body p.source).map .source ++
      match p with
      | .baseline _ => []
      | .optimized _ k => k.body
  structural := some sameRun

theorem linked_run_eq (src : LinkedSources) (kernels : Kernels)
    {State Value : Type} (w : World State Value) (input : TrainingInput) :
    (Program.optimized src.candidate kernels).run w input = (Program.baseline src.reference).run w input := by
  simp only [Program.run, Program.source, Program.calls, runWith, src.candidateOK, src.referenceOK,
    kernels.calls_eq]

/-- The driver runs the checked fixed training CE, paired transpose, and
validation heads. The installed MLP artifacts supply the scale and cache
refresh, and the tied embedding uses the checked transpose hooks.
The checked QKV programs refresh all three attention caches.
The checked ANVIL programs run the rails and counter seeds, first Gram
and divisors, cooperative rounds, and epilogue.
The tail programs update the snapshots and restore the final weight norms.
With graph artifacts installed, supported copies
run their checked word transfers and captured launch packets run their
checked staging and lookup. The conclusion includes the FP32 validation trace. -/
specification training_with_compiled_ce (src : LinkedSources) (kernels : Kernels) :
    Program.optimized src.candidate kernels ≡[([] : Spec.Assumptions Statement)] Program.baseline src.reference := by
  refine Spec.FloatingPoint.ofStructural (structural := sameRun) rfl rfl ?_
  exact fun _ _ w input => linked_run_eq src kernels w input

theorem final_validation_eq (src : LinkedSources) (kernels : Kernels)
    {State Value : Type} (w : World State Value) (input : TrainingInput) :
    ((Program.optimized src.candidate kernels).run w input).bind Result.finalValidation =
      ((Program.baseline src.reference).run w input).bind Result.finalValidation := by
  rw [linked_run_eq src kernels]

def Kernels.valueWorkspaceSafe (kernels : Kernels) : Prop :=
  match kernels.sparse with
  | none => True
  | some sparse => ∀ (state : ValueAdam.Elementwise.WorkspaceState)
      (events : List ValueAdam.Elementwise.WorkspaceEvent),
    let next := events.foldl (ValueAdam.Elementwise.WorkspaceEvent.advance sparse.valueNative) state
    next.bytes.beforeScatter next.history = ValueAdam.Elementwise.Workspace.zero ∧
      (next.bytes.beforeDense next.history).gradient = ValueAdam.Elementwise.Workspace.zero.gradient

theorem Kernels.value_workspace_safe (kernels : Kernels) : kernels.valueWorkspaceSafe := by
  unfold Kernels.valueWorkspaceSafe
  split
  · trivial
  · exact ValueAdam.Elementwise.WorkspaceEvent.trace_starts_clean _

structure Certificate where
  sources : LinkedSources
  kernels : Kernels
  equivalent : Program.optimized sources.candidate kernels ≡[([] : Spec.Assumptions Statement)]
    Program.baseline sources.reference
  valueWorkspaceSafe : kernels.valueWorkspaceSafe

def certify (sources : LinkedSources) (kernels : Kernels) : Certificate :=
  ⟨sources, kernels, training_with_compiled_ce sources kernels, kernels.value_workspace_safe⟩

#print_fp_assumptions training_with_compiled_ce
#axiomsClean Kernels.value_workspace_safe
#axiomsClean SparsePrograms.grouped_value_event_exact
#axiomsClean SparsePrograms.sendNgram_exact
#axiomsClean SparsePrograms.run_eq
#axiomsClean SparsePrograms.landing_eq
#axiomsClean SparsePrograms.accumulateNgram_eq
#axiomsClean SparsePrograms.accumulateValue_eq
#axiomsClean Kernels.graphStage_eq
#axiomsClean Kernels.norm_eq
#axiomsClean Kernels.normBackward_eq
#axiomsClean Kernels.activationStorage_eq
#axiomsClean Kernels.residual_eq
#axiomsClean Kernels.gradients_eq
#axiomsClean Kernels.packing_eq
#axiomsClean Kernels.fusedPacking_eq
#axiomsClean Kernels.fullPacking_eq
#axiomsClean Kernels.calls_eq
#axiomsClean head_forward_backward_eq
#axiomsClean mudd_head_recomputation_eq
#axiomsClean linked_run_eq
#axiomsClean TailPrograms.update_eq
#axiomsClean TailPrograms.restore_eq
#axiomsClean training_with_compiled_ce
#axiomsClean final_validation_eq
end VeriTile.Bench.Models.NanoGPT.Track1.Driver.ArtifactTraining
