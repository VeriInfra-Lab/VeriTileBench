import models.NanoGPT.Track1.DriverExecution
import models.NanoGPT.Track1.SourceAnchors
import models.NanoGPT.Track1.ValueAdamNativeSource

/-! Byte-checked linking for the two actual python source trees. The linker
recognizes the exact installed kernels and launcher source. It exposes
the numerical primitives at the boundaries proved in FPEquiv.lean and
returns the common package text for all other calls. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver

abbrev SourceImage := List (String × String)

def modelFile : String := "track_1_short/model/gpt.py"
def modelLayersFile : String := "track_1_short/model/layers.py"
def attentionFile : String := "track_1_short/model/attention.py"
def ngramFile : String := "track_1_short/perf/kernels/ngram_embed.py"
def amaxFile : String := "track_1_short/perf/kernels/fp8_attention_quant.py"
def crossEntropyFile : String := "track_1_short/perf/kernels/cross_entropy.py"
def sampledFile : String := "track_1_short/sampled_softmax.py"
def sampledCEFile : String := "track_1_short/perf/kernels/sampled_cross_entropy.py"
def ropeFile : String := "track_1_short/perf/kernels/qkv_rope.py"
def transposeFile : String := "track_1_short/perf/kernels/transpose.py"
def anvilFile : String := "track_1_short/optim/anvil.py"
def polarFile : String := "track_1_short/perf/kernels/polar_express.py"
def dataFile : String := "track_1_short/data.py"
def tailFile : String := "track_1_short/tail_average.py"
def tailLerpFile : String := "track_1_short/perf/kernels/lerp.py"
def valuePullFile : String := "track_1_short/perf/value_embed_pull.py"
def valueEmbedFile : String := "track_1_short/perf/kernels/value_embed.py"
def ngramTableFile : String := "track_1_short/ngram_table.py"
def ngramAdamFile : String := "track_1_short/perf/kernels/ngram_adam.py"
def rowScatterFile : String := "track_1_short/perf/kernels/row_scatter.py"
def rowOwnersFile : String := "track_1_short/sharded_rows.py"
def rowPrefetchFile : String := "track_1_short/perf/row_prefetch.py"
def replicatedAdamFile : String := "track_1_short/perf/replicated_adam.py"
def replicatedKernelFile : String := "track_1_short/perf/kernels/replicated_adam.py"
def privateKernel : String := "track_1_short/perf/kernels/mlp_scale_refresh.py"
def privateWeightKernel : String := "track_1_short/perf/kernels/mlp_weights_refresh.py"
def privateHeadKernel : String := "track_1_short/perf/kernels/lm_head_refresh.py"
def privateActivationKernel : String := "track_1_short/perf/kernels/activation_fp8.py"
def privateGatherKernel : String := "track_1_short/perf/kernels/sampled_rows_gather.py"
def privateTransposeKernel : String := "track_1_short/perf/kernels/byte_transpose.py"
def privateBf16TransposeKernel : String := "track_1_short/perf/kernels/bf16_transpose.py"
def privateDensifyKernel : String := "track_1_short/perf/kernels/sampled_grad_densify.py"
def privateTailKernel : String := "track_1_short/perf/kernels/tail_lerp.py"
def privateTailNormKernel : String := "track_1_short/perf/kernels/tail_norm.py"
def privateSparseRowKernel : String := "track_1_short/perf/kernels/sparse_row_pack.py"
def privateSparseRowFetchKernel : String := "track_1_short/perf/kernels/sparse_row_fetch.py"
def privateNgramMomentKernel : String := "track_1_short/perf/kernels/ngram_moment_replay.py"
def privateNgramEntriesKernel : String := "track_1_short/perf/kernels/ngram_adam_entries.py"
def privateValueAdamKernel : String := "track_1_short/perf/kernels/value_adam_scatter.py"
def privateNativeAdam : String := "track_1_short/perf/kernels/value_adam_read_scale.py"
def privateNativeDispatch : String := "track_1_short/perf/kernels/read_scale_dispatch.py"
def privateValueAccumKernel : String := "track_1_short/perf/kernels/value_embed_accum.py"
def privateReplicatedSpanKernel : String := "track_1_short/perf/kernels/replicated_adam_spans.py"
def privateAnvilEpilogueKernel : String := "track_1_short/perf/kernels/anvil_epilogue.py"
def privateAnvilTraceKernel : String := "track_1_short/perf/kernels/anvil_trace.py"
def privateAnvilRailsKernel : String := "track_1_short/perf/kernels/anvil_rails.py"
def privateGraphStageKernel : String := "track_1_short/perf/kernels/graph_stage.py"
def fp8RefreshGraphsFile : String := "track_1_short/perf/cuda_graphs/fp8_refresh_graphs.py"
def deferredGathersFile : String := "track_1_short/perf/deferred_gathers.py"
def stepGraphsFile : String := "track_1_short/perf/cuda_graphs/step_graphs.py"
def pinnedBatchFile : String := "track_1_short/perf/pinned_batches.py"
def privateBatchPacket : String := "track_1_short/perf/batch_packet.py"
def privateNgramSlots : String := "track_1_short/perf/kernels/ngram_slots.py"
def privateMuddCoefficients : String := "track_1_short/perf/kernels/mudd_coefficients.py"
def privateMuddBackward : String := "track_1_short/perf/kernels/mudd_backward.py"

def imagePaths : TrackProgram → List String
  | .reference => runtimeFiles
  | .candidate => runtimeFiles ++
      [privateKernel, privateWeightKernel, privateHeadKernel, privateGatherKernel,
       privateTransposeKernel, privateBf16TransposeKernel, privateDensifyKernel, privateTailKernel,
       privateTailNormKernel, privateSparseRowKernel, privateSparseRowFetchKernel, privateNgramMomentKernel,
       privateNgramEntriesKernel, privateValueAdamKernel, privateNativeAdam, privateNativeDispatch, privateValueAccumKernel, privateReplicatedSpanKernel,
       privateAnvilEpilogueKernel, privateAnvilTraceKernel, privateAnvilRailsKernel, privateGraphStageKernel,
       privateBatchPacket, privateNgramSlots, privateActivationKernel, privateMuddCoefficients, privateMuddBackward]

/-- Replacing a section requires an exact, unique occurrence of its source. -/
def replaceOnce (source before after : String) : Option String :=
  match source.splitOn before with
  | [head, tail] => some (head ++ after ++ tail)
  | _ => none

def normalizeText (side : TrackProgram) (path text : String) : Option String :=
  if path == "train_gpt.py" then do
    let text ← match side with
      | .reference => some text
      | .candidate => driverReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedDriver then some text else none
  else if path == modelFile then do
    let text ← replaceOnce text "EVAL_CE_SLAB_ROWS = 32768\n" "EVAL_CE_SLAB_ROWS = 32768\n"
    let text ← replaceOnce text pinnedModelLoss pinnedModelLoss
    let text ← replaceOnce text pinnedCEHeadInit pinnedCEHeadInit
    let text ← replaceOnce text pinnedAttentionWidths pinnedAttentionWidths
    let text ← replaceOnce text pinnedQueryWidth pinnedQueryWidth
    let text ← replaceOnce text pinnedValueWidth pinnedValueWidth
    let text ← replaceOnce text
      (match side with | .reference => referenceWideYarn | .candidate => candidateWideYarn) referenceWideYarn
    let text ← replaceOnce text
      (match side with | .reference => referenceImport | .candidate => candidateImport) referenceImport
    let text ← replaceOnce text
      (match side with | .reference => referenceScaleTail | .candidate => candidateScaleTail) referenceScaleTail
    let text ← replaceOnce text
      (match side with | .reference => referenceMLPWeights | .candidate => candidateMLPWeights) referenceMLPWeights
    let text ← replaceOnce text
      (match side with | .reference => referenceHeadImport | .candidate => candidateHeadImport) referenceHeadImport
    let text ← replaceOnce text
      (match side with | .reference => referenceQKVImport | .candidate => candidateQKVImport) referenceQKVImport
    let text ← replaceOnce text
      (match side with | .reference => referenceQKVLoop | .candidate => candidateQKVLoop) referenceQKVLoop
    replaceOnce text
      (match side with | .reference => referenceQKVView | .candidate => candidateQKVView) referenceQKVView
  else if path == attentionFile then do
    let text ← match side with
      | .reference => pure text
      | .candidate => yarnReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedAttention then some text else none
  else if path == "track_1_short/config.py" then
    if text == pinnedTrainingConfig then some text else none
  else if path == "track_1_short/schedule.py" then do
    let text ← match side with
      | .reference => some text
      | .candidate => scheduleReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedWindowSchedule then some text else none
  else if path == fp8RefreshGraphsFile then
    if text == pinnedFp8RefreshGraphs then some text else none
  else if path == deferredGathersFile then
    if text == pinnedDeferredGathers then some text else none
  else if path == stepGraphsFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => stepGraphReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedStepGraphs then some text else none
  else if path == pinnedBatchFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => pinnedBatchReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedBatchStaging then some text else none
  else if path == "track_1_short/training.py" then do
    let text ← replaceOnce text pinnedAdamDefaults pinnedAdamDefaults
    let text ← replaceOnce text pinnedAdvanceSchedule pinnedAdvanceSchedule
    let text ← replaceOnce text pinnedTrainingReset pinnedTrainingReset
    let text ← replaceOnce text pinnedOptimizerDispatch pinnedOptimizerDispatch
    let text ← replaceOnce text pinnedWarmupAdamCadence pinnedWarmupAdamCadence
    let text ← pinnedValueAdamConfig.foldlM (fun text span => replaceOnce text span span) text
    replaceOnce text pinnedValidationWindow pinnedValidationWindow
  else if path == ngramFile then
    replaceOnce text
      (match side with | .reference => referenceNgram | .candidate => candidateNgram) referenceNgram
  else if path == amaxFile then
    let expected := match side with | .reference => referenceAmax | .candidate => candidateAmax
    if text == expected then some referenceAmax else none
  else if path == crossEntropyFile then do
    let text ← replaceOnce text pinnedHeadPack pinnedHeadPack
    let text ← match side with
    | .reference => some text
    | .candidate =>
      (ceLogitReuseReplacements ++ ceTransposeReplacements ++ trainingCEReplacements ++
        ceActivationReplacements).foldlM
        (fun text (before, after) => replaceOnce text before after) text
    let text ← replaceOnce text pinnedTrainingCEKernel pinnedTrainingCEKernel
    let text ← replaceOnce text pinnedTrainingCEBlock pinnedTrainingCEBlock
    let text ← replaceOnce text pinnedTrainingCEVocabulary pinnedTrainingCEVocabulary
    let text ← replaceOnce text pinnedTrainingCESoftcap pinnedTrainingCESoftcap
    let text ← replaceOnce text pinnedTrainingCECompiler pinnedTrainingCECompiler
    let text ← replaceOnce text pinnedTrainingCELaunch pinnedTrainingCELaunch
    let text ← replaceOnce text pinnedTrainingCEForward pinnedTrainingCEForward
    let text ← replaceOnce text pinnedCELogitGemm pinnedCELogitGemm
    let text ← replaceOnce text pinnedCEHeadBackward pinnedCEHeadBackward
    replaceOnce text pinnedCEBackward pinnedCEBackward
  else if path == sampledFile then do
    let text ← replaceOnce text pinnedCandidateSchedule pinnedCandidateSchedule
    let text ← replaceOnce text pinnedCandidateBuilder pinnedCandidateBuilder
    let text ← replaceOnce text pinnedGatherWait pinnedGatherWait
    let text ← replaceOnce text pinnedSampledArange pinnedSampledArange
    let text ← replaceOnce text pinnedCandidateView pinnedCandidateView
    let text ← replaceOnce text pinnedSampledInverse pinnedSampledInverse
    let text ← replaceOnce text
      (match side with | .reference => referenceGatherImport | .candidate => candidateGatherImport)
      referenceGatherImport
    replaceOnce text (match side with | .reference => referenceGather | .candidate => candidateGather) referenceGather
  else if path == sampledCEFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => do
        let text ← replaceOnce text candidateDensifyImport referenceDensifyImport
        let text ← replaceOnce text candidateDensifyLaunch referenceDensifyLaunch
        (sampledCELogitReuseReplacements ++ sampledCEPredictionReplacements ++ ceActivationReplacements).foldlM
          (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedSampledCE then some text else none
  else if path == ropeFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => ropeReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedRope then some text else none
  else if path == transposeFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => transposeReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedTranspose then some text else none
  else if path == tailFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => tailReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedTailAverages then some text else none
  else if path == tailLerpFile then
    if text == pinnedTailLerp then some text else none
  else if path == valuePullFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => do
        let text ← nativeValuePullReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
        valuePullReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedValuePull then some text else none
  else if path == valueEmbedFile then do
    let text ← match side with
      | .reference => pure text
      | .candidate => valueEmbedReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedValueEmbed then some text else none
  else if path == ngramTableFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => ngramTableReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedNgramTable then some text else none
  else if path == ngramAdamFile then do
    let text ← match side with
      | .reference => pure text
      | .candidate => ngramAdamReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedNgramAdam then some text else none
  else if path == rowScatterFile then do
    let text ← match side with
      | .reference => pure text
      | .candidate => rowScatterReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedRowScatter then some text else none
  else if path == rowOwnersFile then do
    let text ← match side with
      | .reference => pure text
      | .candidate => rowOwnersReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedRowOwners then some text else none
  else if path == rowPrefetchFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => rowPrefetchReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    if text == pinnedRowPrefetch then some text else none
  else if path == replicatedAdamFile then
    let expected := match side with
      | .reference => referenceReplicatedAdam
      | .candidate => candidateReplicatedAdam
    if text == expected then some referenceReplicatedAdam else none
  else if path == replicatedKernelFile then
    if text == pinnedReplicatedAdamKernel then some text else none
  else if path == modelLayersFile then do
    let text ← replaceOnce text pinnedNextMultiple pinnedNextMultiple
    replaceOnce text pinnedEvalProjection pinnedEvalProjection
  else if path == polarFile then
    let expected := match side with | .reference => referencePolar | .candidate => candidatePolar
    if text == expected then some referencePolar else none
  else if path == anvilFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => do
        let text ← nativeValueAnvilReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
        let text ← anvilCooperativeReplacements.foldlM (fun text (before, after, count) =>
          let sections := text.splitOn before
          if sections.length == count + 1 then some (String.intercalate after sections) else none) text
        let substitutions := anvilFullTileReplacements ++ anvilRailsReplacements ++
          anvilTraceReplacements ++ anvilEpilogueReplacements
        substitutions.foldlM (fun text (before, after) => replaceOnce text before after) text
    let text ← replaceOnce text pinnedAnvilStep pinnedAnvilStep
    let text ← replaceOnce text pinnedAnvilCascade pinnedAnvilCascade
    let text ← replaceOnce text pinnedAnvilMaps pinnedAnvilMaps
    let text ← replaceOnce text pinnedAnvilFrobenius pinnedAnvilFrobenius
    let text ← replaceOnce text pinnedAnvilRails pinnedAnvilRails
    let text ← replaceOnce text pinnedAnvilBankUpdate pinnedAnvilBankUpdate
    let text ← replaceOnce text pinnedAnvilEqualizer pinnedAnvilEqualizer
    let text ← replaceOnce text pinnedAnvilDecay pinnedAnvilDecay
    let text ← replaceOnce text pinnedValueAdamDispatch pinnedValueAdamDispatch
    let text ← replaceOnce text pinnedValueAdamPrepare pinnedValueAdamPrepare
    replaceOnce text pinnedValueAdamStep pinnedValueAdamStep
  else if path == dataFile then do
    let text ← match side with
      | .reference => some text
      | .candidate => fineWebDataReplacements.foldlM (fun text (before, after) => replaceOnce text before after) text
    let text ← replaceOnce text pinnedFineWebRead pinnedFineWebRead
    let text ← replaceOnce text pinnedFineWebCumulative pinnedFineWebCumulative
    let text ← replaceOnce text pinnedFineWebBOS pinnedFineWebBOS
    let text ← replaceOnce text pinnedFineWebShard pinnedFineWebShard
    let text ← replaceOnce text pinnedFineWebBatches pinnedFineWebBatches
    let text ← replaceOnce text pinnedFineWebSplit pinnedFineWebSplit
    replaceOnce text pinnedFineWebScheduled pinnedFineWebScheduled
  else some text

def normalize (side : TrackProgram) (image : SourceImage) : Option SourceImage := do
  if image.map Prod.fst != imagePaths side then none else
  match side with
  | .reference => normalizePackage
  | .candidate =>
      if image.lookup privateKernel == some candidateKernel &&
        image.lookup privateWeightKernel == some candidateWeightKernel &&
        image.lookup privateHeadKernel == some candidateHeadKernel &&
        image.lookup privateGatherKernel == some candidateGatherKernel &&
        image.lookup privateTransposeKernel == some candidateTransposeKernel &&
        image.lookup privateBf16TransposeKernel == some candidateBf16TransposeKernel &&
        image.lookup privateDensifyKernel == some candidateDensifyKernel &&
        image.lookup privateTailKernel == some candidateTailKernel &&
        image.lookup privateTailNormKernel == some candidateTailNormKernel &&
        image.lookup privateSparseRowKernel == some candidateSparseRowKernel &&
        image.lookup privateSparseRowFetchKernel == some candidateSparseRowFetchKernel &&
        image.lookup privateNgramMomentKernel == some candidateNgramMomentKernel &&
        image.lookup privateNgramEntriesKernel == some candidateNgramEntriesKernel &&
        image.lookup privateValueAdamKernel == some candidateNativeScatter &&
        image.lookup privateNativeAdam == some candidateNativeAdam &&
        image.lookup privateNativeDispatch == some candidateNativeDispatch &&
        image.lookup privateValueAccumKernel == some candidateValueAccumKernel &&
        image.lookup privateReplicatedSpanKernel == some candidateReplicatedSpanKernel &&
        image.lookup privateAnvilEpilogueKernel == some candidateAnvilEpilogueKernel &&
        image.lookup privateAnvilTraceKernel == some candidateAnvilTraceKernel &&
        image.lookup privateAnvilRailsKernel == some candidateAnvilRailsKernel &&
        image.lookup privateGraphStageKernel == some candidateGraphStageKernel &&
        image.lookup privateBatchPacket == some candidateBatchPacket &&
        image.lookup privateNgramSlots == some candidateNgramSlots &&
        image.lookup privateActivationKernel == some candidateActivationKernel &&
        image.lookup privateMuddCoefficients == some candidateMuddCoefficientsKernel &&
        image.lookup privateMuddBackward == some candidateMuddBackwardKernel then normalizePackage else none
where
  normalizePackage := runtimeFiles.mapM fun path => do
    let text ← image.lookup path
    let text ← normalizeText side path text
    pure (path, text)

/-- This is a source check, discharged by the executable linker. It contains
no equality premise about running either trainer. -/
structure LinkedSources where
  reference : SourceImage
  candidate : SourceImage
  common : SourceImage
  referenceOK : normalize .reference reference = some common
  candidateOK : normalize .candidate candidate = some common

def link (reference candidate : SourceImage) : Option LinkedSources := do
  match hRef : normalize .reference reference with
  | none => none
  | some common =>
    if hCand : normalize .candidate candidate = some common
    then some ⟨reference, candidate, common, hRef, hCand⟩
    else none

def readImage (root : System.FilePath) (side : TrackProgram) : IO SourceImage :=
  (imagePaths side).mapM fun (file : String) => do
    pure (file, ← IO.FS.readFile (root / (file : System.FilePath)))

end VeriTile.Bench.Models.NanoGPT.Track1.Driver
