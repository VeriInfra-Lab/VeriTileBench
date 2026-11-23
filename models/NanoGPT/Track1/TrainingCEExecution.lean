import models.NanoGPT.Track1.TrainingEquiv

/-! Run the checked training cross-entropy stages as one execution.
The initialization and scan populate the shared input, the compiled
publication produces the caches, and the final stages read those caches. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEExecution
open TrainingCE
open VeriTile.Triton FP

structure Certificate (w : Width) (ptx : String) (predictions : Fin 4) (mode : Softcap)
    (scale : GradScale := .runtime) where
  init : TrainingCEInitBinding.Certificate w ptx mode scale
  scan : TrainingCEGatherBinding.ScanCertificate w ptx
  reduction : TrainingCEReductionBinding.Certificate ptx
  publication : TrainingCEPublishBinding.Certificate ptx predictions mode
  loss : TrainingCELossBinding.Certificate w ptx predictions mode
  base : TrainingCEGatherBinding.GeometryCertificate w ptx
  patch : TrainingCESparseBinding.Certificate w ptx predictions mode

def certify (w : Width) (ptx : String) (predictions : Fin 4) (mode : Softcap)
    (scale : GradScale := .runtime) : Option (Certificate w ptx predictions mode scale) := do
  return ⟨← TrainingCEInitBinding.certify w ptx mode scale, ← TrainingCEGatherBinding.certifyScan w ptx,
    ← TrainingCEReductionBinding.certify ptx, ← TrainingCEPublishBinding.certify ptx predictions mode,
    ← TrainingCELossBinding.certify w ptx predictions mode, ← TrainingCEGatherBinding.certifyGeometry w ptx,
    ← TrainingCESparseBinding.certify w ptx predictions mode⟩

private def zero : F32 := Value.ofBits .fp32 0
private def emptyEntry : LogitEntry := ⟨⟨zero, zero⟩, zero⟩

/-- A common layout for the decoded initialization payload. While scanning,
the contribution words are read from shared memory. The sigmoid and
scaled-logit words are delivered to publication by the decoded transport,
so this logical layout also represents the register path. -/
def decodedTables (w : Width) (ptx : String) (op : Native) (p : Parameters) (live : Fin 4) :
    List CudaStores.Store :=
  let laneZero := TrainingCEInitBinding.decode w ptx live true
  let remaining := TrainingCEInitBinding.decode w ptx live false
  let table := Vector.ofFn fun lane : Lane =>
    let trace := if lane.val == 0 then laneZero else remaining
    let writes := trace.bind (TrainingCEInitBinding.Writes.eval op p (UInt8.ofNat lane.val))
    (writes.map Prod.fst).getD emptyEntry
  let factor := (laneZero.bind (TrainingCEInitBinding.Writes.eval op p 0)).bind Prod.snd
  initializationStores table (factor.getD zero)

theorem decoded_tables_exact {w : Width} {ptx : String} {mode : Softcap} {scale : GradScale}
    (init : TrainingCEInitBinding.Certificate w ptx mode scale) (op : Native) (p : Parameters)
    (live : Fin 4) (accepted : mode.accepts p = true) (scaleAccepted : scale.accepts p = true) :
    decodedTables w ptx op p live = tableStores op p := by
  unfold decodedTables tableStores
  refine congrArg₂ initializationStores ?_ ?_
  · apply Vector.ext
    intro i hi
    let lane : Lane := ⟨i, hi⟩
    simp only [Vector.getElem_ofFn]
    have trace :
        (if lane.val == 0 then TrainingCEInitBinding.decode w ptx live true
          else TrainingCEInitBinding.decode w ptx live false) =
        TrainingCEInitBinding.decode w ptx live (lane.val == 0) := by
      cases lane.val == 0 <;> rfl
    change (((if lane.val == 0 then TrainingCEInitBinding.decode w ptx live true
        else TrainingCEInitBinding.decode w ptx live false).bind
      (TrainingCEInitBinding.Writes.eval op p (UInt8.ofNat lane.val))).map Prod.fst).getD _ = _
    rw [trace, TrainingCEInitBinding.initialized_table_exact init op p live lane accepted scaleAccepted]
    simp only [Option.map_some, Option.getD_some, readLogitTable, UInt8.toNat_ofNat',
      lane, Nat.reducePow, Nat.mod_eq_of_lt hi]
  · have first := TrainingCEInitBinding.initialized_table_exact init op p live (0 : Lane) accepted scaleAccepted
    simpa only [Fin.val_zero, beq_self_eq_true, if_true, Option.bind_some, Option.getD_some]
      using congrArg (fun x => (x.bind Prod.snd).getD zero) first

private def readStagedByte {w : Width} (trace : Option TrainingCEGatherBinding.ScanTrace)
    (r : Row w) (column : Column w) : Option UInt8 := do
  let trace ← trace
  let (dst, src) ← trace.chunks.find? fun (dst, _) => decide (dst ≤ column.val ∧ column.val < dst + 8)
  let x := src + (column.val - dst)
  if valid : x < w.columns then some (r.logits ⟨x, valid⟩) else none

private def stagedByte {w : Width} (ptx : String) (r : Row w) (column : Column w) : Option UInt8 :=
  readStagedByte (TrainingCEGatherBinding.decodeScan w ptx (columnOwner column)) r column

theorem staged_byte_exact {w : Width} {ptx : String}
    (cert : TrainingCEGatherBinding.ScanCertificate w ptx) (r : Row w) (column : Column w) :
    stagedByte ptx r column = some (r.logits column) := by
  obtain ⟨trace, start, byte, decoded, copied, offset⟩ := TrainingCEGatherBinding.scan_code_cover cert column
  let covers := fun pair : Nat × Nat => decide (pair.1 ≤ column.val ∧ column.val < pair.1 + 8)
  have found : (trace.chunks.find? covers).isSome := by
    rw [List.find?_isSome]
    have h := byte.isLt
    exact ⟨(start, start), copied, decide_eq_true (by dsimp [covers]; omega)⟩
  obtain ⟨⟨dst, src⟩, index⟩ := Option.isSome_iff_exists.mp found
  have foundCovers : covers (dst, src) = true := List.find?_some (p := covers) index
  have within : dst ≤ column.val ∧ column.val < dst + 8 := of_decide_eq_true foundCovers
  have member := List.mem_of_find?_eq_some index
  have shape := Option.some.inj (decoded.symm.trans (cert.traces (columnOwner column)))
  rw [shape] at member
  simp only [TrainingCEGatherBinding.expectedScan] at member
  obtain ⟨base, _, eq⟩ := List.mem_map.mp member
  have same : src = dst := by simpa using (Prod.ext_iff.mp eq).2.symm.trans (Prod.ext_iff.mp eq).1
  have address : src + (column.val - dst) = column.val := by
    have h : dst ≤ column.val := within.1
    omega
  unfold stagedByte readStagedByte
  rw [decoded]
  simp only [bind, Option.bind_some, index, covers, address, dif_pos column.isLt, Fin.eta]

def decodedCodes {w : Width} (ptx : String) (r : Row w) : List CudaStores.Store :=
  let traces := Vector.ofFn (TrainingCEGatherBinding.decodeScan w ptx)
  codeStores { r with logits := fun c => (readStagedByte traces[(columnOwner c).val] r c).getD 0 }

theorem decoded_codes_exact {w : Width} {ptx : String}
    (cert : TrainingCEGatherBinding.ScanCertificate w ptx) (r : Row w) :
    decodedCodes ptx r = codeStores r := by
  have read (c : Column w) :
      (readStagedByte (TrainingCEGatherBinding.decodeScan w ptx (columnOwner c)) r c).getD 0 = r.logits c := by
    change (stagedByte ptx r c).getD 0 = _
    rw [staged_byte_exact cert r, Option.getD_some]
  unfold decodedCodes codeStores
  simp only [Vector.getElem_ofFn, read]

def decodedLeaders (w : Width) (ptx : String) (reduction : TrainingCEReductionBinding.Reduction)
    (before : CudaStores.Memory) : Fin 8 → F32 :=
  reduction.readLeaders fun lane =>
    ((TrainingCEGatherBinding.decodeScan w ptx lane).map fun trace =>
      trace.evalSum fun c => (readSharedTable before (before (codeAddress + c))).contribution).getD zero

def decodedCache {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (publication : TrainingCEPublishBinding.Publication) (before : CudaStores.Memory)
    (leaders : Fin 8 → F32) : CudaStores.Memory :=
  let payload := fun code => publication.eval op (TrainingCEPublishBinding.Inputs.forPublication
    p r.weights r.prefixTarget.isSome leaders (readSharedTable before code).logit (readGradientFactor before))
  CudaStores.commitAll (compiledCacheStores payload) before

structure Observation (w : Width) where
  loss : F32
  gradient : Column w → Option UInt8

/-- Read the loss word and gradient writes from the actual compilation. -/
def run {w : Width} (ptx : String) (op : Native) (p : Parameters) (r : Row w)
    (mode : Softcap) (prior : CudaStores.Memory) : Option (Observation w) := do
  let tables := decodedTables w ptx op p (TrainingCELossBinding.liveSlots r)
  let codes := decodedCodes ptx r
  let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
  let reduction ← TrainingCEReductionBinding.decode ptx
  let publication ← TrainingCEPublishBinding.decode ptx
  let leaders := decodedLeaders w ptx reduction before
  let ready := decodedCache op p r publication before leaders
  let write ← TrainingCELossBinding.decode w ptx (TrainingCELossBinding.liveSlots r)
    r.prefixTarget.isSome true mode
  let lossExpr ← write
  let sigma := fun (c : Column w) => (readSharedProbability ready (ready (codeAddress + c.val))).sigma
  return ⟨lossExpr.eval op (TrainingCELossBinding.inputs p r leaders sigma),
    fun c => Driver.compiledCEGradientFromCache ptx op p r ready c mode⟩

set_option maxRecDepth 4096 in
private theorem decoded_cache_exact {w : Width} {ptx : String} {mode : Softcap}
    (op : Native) (p : Parameters) (r : Row w)
    (publication : TrainingCEPublishBinding.Certificate ptx r.predictions mode)
    (tables codes : List CudaStores.Store) (prior : CudaStores.Memory) (accepted : mode.accepts p = true) :
    let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
    decodedCache op p r publication.writes before (stagedWarpLeaders w before) =
      (stageRow op p r tables codes prior).2 := by
  let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
  let leaders := stagedWarpLeaders w before
  let lse := logSumExpFromLeaders op p leaders
  let payload := fun code =>
    let entry := (readSharedTable before code).logit.forRow op lse
    (entry.probability, baseGradient p r (fixedWalk r.predictions) (readGradientFactor before) entry)
  have values : (fun code => publication.writes.eval op (TrainingCEPublishBinding.Inputs.forPublication
    p r.weights r.prefixTarget.isSome leaders (readSharedTable before code).logit (readGradientFactor before))) =
      payload := by
    funext code
    exact TrainingCEPublishBinding.publication_exact op p r publication leaders
      (readSharedTable before code).logit (readGradientFactor before) accepted
  change decodedCache op p r publication.writes before leaders = _
  unfold decodedCache
  rw [values]
  dsimp only [payload]
  rw [compiled_cache_stores_exact op p r before lse]
  rfl

set_option maxRecDepth 4096 in
/-- The decoded computations pass their register or shared-memory values to
the next stage. The resulting FP32 loss and each E5M2 byte agree with
the source, for every row, legal launch and prior shared-memory state. -/
theorem run_exact {w : Width} {ptx : String} {mode : Softcap} {scale : GradScale}
    (op : Native) (p : Parameters) (r : Row w) (artifact : Certificate w ptx r.predictions mode scale)
    (prior : CudaStores.Memory) (accepted : mode.accepts p = true) (scaleAccepted : scale.accepts p = true) :
    run ptx op p r mode prior =
      some ⟨(reference w op p r).loss, fun c => some ((reference w op p r).gradient c)⟩ := by
  let tables := tableStores op p
  let codes := codeStores r
  let before := CudaStores.commitAll codes (CudaStores.commitAll tables prior)
  let ready := (stageRow op p r tables codes prior).2
  let leaders := stagedWarpLeaders w before
  let sigma := fun (c : Column w) => (readSharedProbability ready (ready (codeAddress + c.val))).sigma
  have sum : decodedLeaders w ptx artifact.reduction.reduction before = leaders :=
    Driver.compiled_ce_scanned_leaders artifact.scan artifact.reduction before
  have rowLoss := Driver.compiled_ce_row_loss op p r artifact.reduction artifact.loss
    tables (.refl _) codes (.refl _) prior accepted
  simp only [TrainingCEReductionBinding.leaders_exact artifact.reduction] at rowLoss
  change (TrainingCELossBinding.decode w ptx (TrainingCELossBinding.liveSlots r)
    r.prefixTarget.isSome true mode).map
      (Option.map (TrainingCEPublishBinding.Expr.eval op (TrainingCELossBinding.inputs p r leaders sigma))) = _ at rowLoss
  unfold run
  dsimp only
  rw [decoded_tables_exact artifact.init op p _ accepted scaleAccepted, decoded_codes_exact artifact.scan r,
    artifact.reduction.decoded]
  simp only [bind, Option.bind_some]
  rw [artifact.publication.decoded]
  simp only [Option.bind_some]
  change (do
    let loss ← TrainingCELossBinding.decode w ptx (TrainingCELossBinding.liveSlots r)
      r.prefixTarget.isSome true mode
    let expr ← loss
    let decoded := decodedLeaders w ptx artifact.reduction.reduction before
    let cache := decodedCache op p r artifact.publication.writes before decoded
    pure (⟨expr.eval op (TrainingCELossBinding.inputs p r decoded (fun c =>
      (readSharedProbability cache (cache (codeAddress + c.val))).sigma)),
      fun c => Driver.compiledCEGradientFromCache ptx op p r cache c mode⟩ : Observation w)) = _
  dsimp only
  rw [sum, decoded_cache_exact op p r artifact.publication tables codes prior accepted]
  cases trace : TrainingCELossBinding.decode w ptx (TrainingCELossBinding.liveSlots r)
      r.prefixTarget.isSome true mode with
  | none => simp [trace] at rowLoss
  | some write =>
    rw [trace] at rowLoss
    cases write with
    | none => simp at rowLoss
    | some expr =>
      have lossExact : expr.eval op (TrainingCELossBinding.inputs p r leaders sigma) =
          (reference w op p r).loss := by simpa using rowLoss
      change some (⟨expr.eval op (TrainingCELossBinding.inputs p r leaders sigma),
        fun c => Driver.compiledCEGradient ptx op p r tables codes prior c mode⟩ : Observation w) = _
      rw [lossExact]
      apply congrArg some
      apply congrArg (Observation.mk (reference w op p r).loss)
      funext c
      exact Driver.compiled_ce_row_gradient op p r artifact.base artifact.patch tables codes
        (.refl _) (.refl _) prior c

theorem launched_exact {w : Width} {ptx : String} (op : Native) (p : Parameters) (r : Row w)
    (artifact : Certificate w ptx r.predictions (w.launchSoftcap r.predictions p)
      (w.launchGradScale r.predictions p)) (prior : CudaStores.Memory) :
    run ptx op p r (w.launchSoftcap r.predictions p) prior =
      some ⟨(specialized w op p r).loss, fun c => some ((specialized w op p r).gradient c)⟩ := by
  rw [specialized_equal]
  exact run_exact op p r artifact prior (w.launch_softcap_accepted r.predictions p)
    (w.launch_grad_scale_accepted r.predictions p)

/-- The fixed PTX implements the training CE call used by the linked
driver. Both outputs are the ones consumed by the training graph. -/
theorem linked_training_call_exact {w : Width} {ptx : String} (src : Driver.LinkedSources)
    (op : Native) (p : Parameters) (r : Row w)
    (artifact : Certificate w ptx r.predictions (w.launchSoftcap r.predictions p)
      (w.launchGradScale r.predictions p))
    (prior : CudaStores.Memory) :
    let output := src.baseline.side.calls.trainingCE w op p r
    run ptx op p r (w.launchSoftcap r.predictions p) prior =
      some ⟨output.loss, fun c => some (output.gradient c)⟩ := by
  exact run_exact op p r artifact prior (w.launch_softcap_accepted r.predictions p)
    (w.launch_grad_scale_accepted r.predictions p)

private theorem columns_pos (w : Width) : 0 < w.columns := by cases w <;> decide

def logitOffset {w : Width} (row : Nat) (column : Column w) : Nat :=
  row * w.columns + column.val

theorem logit_offset_parts {w : Width} (row : Nat) (column : Column w) :
    logitOffset row column / w.columns = row ∧
      logitOffset row column % w.columns = column.val := by
  have within := column.isLt
  cases w <;> simp only [logitOffset, Width.columns] at * <;> omega

theorem logit_offset_fits {w : Width} {rows : Nat} (row : Fin rows) (column : Column w)
    (launch : rows * w.columns < 2^31) : logitOffset row.val column < 2^31 := by
  calc
    logitOffset row.val column < row.val * w.columns + w.columns :=
      Nat.add_lt_add_left column.isLt _
    _ = (row.val + 1) * w.columns := by rw [Nat.add_mul, Nat.one_mul]
    _ ≤ rows * w.columns := Nat.mul_le_mul_right _ row.isLt
    _ < 2^31 := launch

theorem compiled_logit_offsets {w : Width} {rows : Nat}
    (row : Fin rows) (column : Column w) (launch : rows * w.columns < 2^31) :
    (BitVec.ofNat 32 (row.val * w.columns)).toInt + (column.val : Int) =
      (logitOffset row.val column : Int) ∧
    (BitVec.ofNat 32 (logitOffset row.val column)).toNat = logitOffset row.val column := by
  have cell := logit_offset_fits row column launch
  have base : row.val * w.columns < 2^31 := by
    exact Nat.lt_of_le_of_lt (Nat.le_add_right _ _) cell
  have signed : (BitVec.ofNat 32 (row.val * w.columns)).toInt = (row.val * w.columns : Nat) := by
    rw [BitVec.toInt_ofNat']
    apply Int.bmod_eq_of_le <;> omega
  exact ⟨by simp only [signed, logitOffset, Nat.cast_add],
    by rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (Nat.lt_trans cell (by decide))]⟩

/-- The read phase takes this row from the shared allocation at entry.
It completes before any lane can overwrite one of these addresses. -/
def logitRow {w : Width} (row : Nat) (memory : CudaStores.Memory) (c : Column w) : UInt8 :=
  memory (logitOffset row c)

/-- The verified base writes and sparse corrections provide the final
effective byte of each written column. An unwritten cell retains its word. -/
def overwriteRow {w : Width} (row : Nat) (words : Column w → Option UInt8)
    (memory : CudaStores.Memory) (addr : Nat) : UInt8 :=
  if addr / w.columns = row then
    (words ⟨addr % w.columns, Nat.mod_lt _ (columns_pos w)⟩).getD (memory addr)
  else memory addr

theorem overwrite_row_readback {w : Width} (row : Nat) (words : Column w → Option UInt8)
    (memory : CudaStores.Memory) (c : Column w) :
    logitRow row (overwriteRow row words memory) c =
      (words c).getD (logitRow row memory c) := by
  have ⟨hr, hc⟩ := logit_offset_parts row c
  unfold logitRow overwriteRow
  rw [if_pos hr]
  exact congrArg (fun k => (words k).getD (memory (logitOffset row c))) (Fin.ext hc)

theorem overwrite_keeps_other_row {w : Width} (writer reader : Nat)
    (different : reader ≠ writer) (words : Column w → Option UInt8) (memory : CudaStores.Memory) :
    logitRow (w := w) reader (overwriteRow writer words memory) = logitRow (w := w) reader memory := by
  funext c
  simp only [logitRow, overwriteRow, (logit_offset_parts reader c).1, if_neg different]

theorem overwrite_rows_commute {w : Width} (a b : Nat) (distinct : a ≠ b)
    (left right : Column w → Option UInt8) (memory : CudaStores.Memory) :
    overwriteRow a left (overwriteRow b right memory) =
      overwriteRow b right (overwriteRow a left memory) := by
  funext addr
  by_cases belongs : addr / w.columns = a
  · simp [overwriteRow, belongs, distinct]
  · simp [overwriteRow, belongs]

private def logitSeed {w : Width} (logits : Column w → UInt8) (addr : Nat) : UInt8 :=
  logits ⟨addr % w.columns, Nat.mod_lt _ (columns_pos w)⟩

private theorem read_seed {w : Width} (row : Nat) (logits : Column w → UInt8) :
    logitRow row (logitSeed logits) = logits := by
  funext c
  exact congrArg logits (Fin.ext (logit_offset_parts row c).2)

structure FixedProgram (w : Width) (predictions : Fin 4) where
  ptx : String
  checked : Certificate w ptx predictions w.defaultSoftcap .lmHead
  storage : TrainingCEGatherBinding.StorageCertificate ptx

def FixedProgram.check (w : Width) (predictions : Fin 4) (ptx : String) :
    Option (FixedProgram w predictions) := do
  let storage ← TrainingCEGatherBinding.certifyStorage ptx
  return ⟨ptx, ← certify w ptx predictions w.defaultSoftcap .lmHead, storage⟩

structure Programs where
  sampled10Two : FixedProgram .sampled10 2
  sampled10Three : FixedProgram .sampled10 3
  sampled14 : FixedProgram .sampled14 1
  sampled24 : FixedProgram .sampled24 1
  fullVocab : FixedProgram .full 1

def Programs.select (programs : Programs) (w : Width) (n : Fin 4) : Option (FixedProgram w n) :=
  match w with
  | .sampled10 =>
    if h : 2 = n then some (h ▸ programs.sampled10Two)
    else if h : 3 = n then some (h ▸ programs.sampled10Three)
    else none
  | .sampled14 => if h : 1 = n then some (h ▸ programs.sampled14) else none
  | .sampled24 => if h : 1 = n then some (h ▸ programs.sampled24) else none
  | .full => if h : 1 = n then some (h ▸ programs.fullVocab) else none

theorem Programs.select_isSome (programs : Programs) (w : Width) (n : Fin 4) :
    (programs.select w n).isSome = w.hasFixedPredictions n := by
  have valid := n.isLt
  have choices : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 := by
    simp only [Fin.ext_iff]
    change n.val = 0 ∨ n.val = 1 ∨ n.val = 2 ∨ n.val = 3
    omega
  rcases choices with h | h | h | h <;> subst n <;> cases w <;> rfl

def FixedProgram.runInPlace (w : Width) (op : Native) (p : Parameters) (r : Row w)
    (program : FixedProgram w r.predictions) (memory : CudaStores.Memory) :
    Option (F32 × CudaStores.Memory) := do
  let snapshot := { r with logits := logitRow r.row.val memory }
  let observation ← TrainingCEExecution.run program.ptx op p snapshot w.defaultSoftcap (fun _ => 0)
  return (observation.loss, overwriteRow r.row.val observation.gradient memory)

theorem FixedProgram.runInPlace_exact (w : Width) (op : Native) (p : Parameters) (r : Row w)
    (program : FixedProgram w r.predictions) (memory : CudaStores.Memory)
    (eligible : w.fixedParametersMatch p = true) :
    let expected := reference w op p { r with logits := logitRow r.row.val memory }
    program.runInPlace w op p r memory =
      some (expected.loss, overwriteRow r.row.val (fun c => some (expected.gradient c)) memory) := by
  let snapshot := { r with logits := logitRow r.row.val memory }
  unfold FixedProgram.runInPlace
  dsimp only []
  simp only [Width.fixedParametersMatch, Bool.and_eq_true] at eligible
  rw [run_exact op p snapshot program.checked _ eligible.1 eligible.2]
  rfl

def FixedProgram.run (w : Width) (op : Native) (p : Parameters) (r : Row w)
    (program : FixedProgram w r.predictions) : Output w :=
  if r.batchRows * w.columns < 2^31 then
    let result := program.runInPlace w op p r (logitSeed r.logits)
    (result.map fun (loss, memory) => ⟨loss, logitRow r.row.val memory⟩).getD ⟨zero, fun _ => 0⟩
  else
    let result := TrainingCEExecution.run program.ptx op p r w.defaultSoftcap (fun _ => 0)
    (result.map fun out => ⟨out.loss, fun c => (out.gradient c).getD 0⟩).getD ⟨zero, fun _ => 0⟩

theorem FixedProgram.run_eq (w : Width) (op : Native) (p : Parameters) (r : Row w)
    (program : FixedProgram w r.predictions) (eligible : w.fixedParametersMatch p = true) :
    program.run w op p r = reference w op p r := by
  unfold FixedProgram.run
  split
  · rw [program.runInPlace_exact w op p r _ eligible]
    simp only [read_seed, Option.map_some, Option.getD_some]
    change (⟨(reference w op p r).loss,
      logitRow r.row.val (overwriteRow r.row.val (fun c => some ((reference w op p r).gradient c))
        (logitSeed r.logits))⟩ : Output w) = _
    have gradient : logitRow r.row.val
        (overwriteRow r.row.val (fun c => some ((reference w op p r).gradient c)) (logitSeed r.logits)) =
          (reference w op p r).gradient := by
      funext c
      rw [overwrite_row_readback]
      rfl
    rw [gradient]
  · simp only [Width.fixedParametersMatch, Bool.and_eq_true] at eligible
    rw [run_exact op p r program.checked _ eligible.1 eligible.2]
    rfl

/-- Mirror the launcher guard and dispatch to the checked PTX on the
five fixed configurations. Other arguments use the dynamic paired path. -/
def Programs.run (programs : Programs) : Run := fun w op p r =>
  if w.fixedParametersMatch p then
    match programs.select w r.predictions with
    | some program => program.run w op p r
    | none => paired w op p r
  else paired w op p r

theorem Programs.run_eq (programs : Programs) : programs.run = reference := by
  funext w op p r
  have generic : paired w op p r = reference w op p r :=
    congrFun (congrFun (congrFun (congrFun loss_gradient_equal w) op) p) r
  unfold Programs.run
  split
  · rename_i eligible
    split
    · exact FixedProgram.run_eq _ _ _ _ _ eligible
    · exact generic
  · exact generic

def gradientBuffer {w : Width} {tokens : Nat} (ptx : String) (op : Native) (p : Parameters)
    (rows : Fin tokens → Row w) : Nat → UInt8 :=
  Driver.ceGradientBuffer fun token c =>
    let r := rows token
    ((run ptx op p r (w.launchSoftcap r.predictions p) (fun _ => 0)).bind fun out =>
      out.gradient c).getD 0

theorem gradient_operands_exact {w : Width} {ptx gir transposePTX : String}
    (op : Native) (p : Parameters) (shape : CETranspose.Shape)
    (rows : Fin (shape.operand .gradient).rows → Row w)
    (artifacts : ∀ token, Certificate w ptx (rows token).predictions
      (w.launchSoftcap (rows token).predictions p) (w.launchGradScale (rows token).predictions p))
    (transpose : CETranspose.Certificate gir transposePTX) (activation : Nat → UInt8) :
    let emitted := gradientBuffer ptx op p rows
    let expected := Driver.ceGradientBuffer (fun token => (reference w op p (rows token)).gradient)
    emitted = expected ∧
      CETranspose.unsigned shape (Driver.ceTransposeInputs activation emitted) =
        CETranspose.staged shape (Driver.ceTransposeInputs activation expected) := by
  dsimp only
  have bytes : gradientBuffer ptx op p rows =
      Driver.ceGradientBuffer (fun token => (reference w op p (rows token)).gradient) := by
    unfold gradientBuffer
    apply congrArg Driver.ceGradientBuffer
    funext token c
    dsimp only
    rw [run_exact op p (rows token) (artifacts token) _
      (w.launch_softcap_accepted (rows token).predictions p) (w.launch_grad_scale_accepted (rows token).predictions p)]
    rfl
  exact ⟨bytes, by rw [bytes, transpose.unsignedStores]⟩

/-- Pass the unified CE execution's E5M2 output through the actual
transpose and inverse-map readback to each dense BF16 gradient word. -/
theorem sampled_dense_gradient_exact (c : SampledRows.Count)
    {ptx gir transposePTX gatherGir gatherPTX : String}
    (op : Native) (p : Parameters) (shape : CETranspose.Shape)
    (rows : Fin (shape.operand .gradient).rows → Row (Driver.sampledCEWidth c))
    (artifacts : ∀ token, Certificate (Driver.sampledCEWidth c) ptx (rows token).predictions
      ((Driver.sampledCEWidth c).launchSoftcap (rows token).predictions p)
      ((Driver.sampledCEWidth c).launchGradScale (rows token).predictions p))
    (transpose : CETranspose.Certificate gir transposePTX)
    (gather : SampledInverse.Certificate c gatherGir gatherPTX)
    {code : Densify.Artifacts} (densify : Densify.Certificate c code)
    (activation : Nat → UInt8)
    (weightGradient : (CETranspose.Cell shape → Option (CETranspose.Operand × Nat × UInt8)) → Nat → UInt16)
    (mark : SampledRows.ClassID → Bool) (size : (Densify.flatNonzero mark).length = c.rows) :
    let candidate := gradientBuffer ptx op p rows
    let expected := Driver.ceGradientBuffer
      (fun token => (reference (Driver.sampledCEWidth c) op p (rows token)).gradient)
    let candidateOperand := CETranspose.unsigned shape (Driver.ceTransposeInputs activation candidate)
    let referenceOperand := CETranspose.staged shape (Driver.ceTransposeInputs activation expected)
    let uploaded := SampledInverse.Candidates.ofBitmap c mark size
    ∀ point, Densify.emitted (weightGradient candidateOperand) (SampledInverse.blockedInverse uploaded) point =
      Densify.reference (weightGradient referenceOperand) (Densify.inverseMap uploaded.ids) point := by
  dsimp only
  intro point
  rw [(gradient_operands_exact op p shape rows artifacts transpose activation).2,
    gather.inverseMap, densify.stores]

#axiomsClean decoded_tables_exact
#axiomsClean staged_byte_exact
#axiomsClean decoded_codes_exact
#axiomsClean run_exact
#axiomsClean launched_exact
#axiomsClean linked_training_call_exact
#axiomsClean Programs.select_isSome
#axiomsClean FixedProgram.run_eq
#axiomsClean FixedProgram.runInPlace_exact
#axiomsClean compiled_logit_offsets
#axiomsClean overwrite_keeps_other_row
#axiomsClean overwrite_rows_commute
#axiomsClean Programs.run_eq
#axiomsClean gradient_operands_exact
#axiomsClean sampled_dense_gradient_exact
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEExecution
