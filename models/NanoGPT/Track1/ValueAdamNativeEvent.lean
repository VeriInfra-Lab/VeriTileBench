import models.NanoGPT.Track1.ValueAdamConsumeState
import models.NanoGPT.Track1.ValueAdamCombinedShard

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

open VeriTile.Triton FP
open SparsePackets (Channel)

abbrev Artifact (period : Period) (tile : Tile) (fused : Bool) :=
  (source : String) × Certificate ⟨period, tile, fused⟩ source

structure Consumer (period : Period) (tile : Tile) (clearAll : Bool) where
  artifact : Artifact period tile true
  matched : artifact.2.modeMatches clearAll = true

structure NativePrograms where
  reference : (period : Period) → (tile : Tile) → Artifact period tile false
  fused : (period : Period) → (tile : Tile) → (clearAll : Bool) → Consumer period tile clearAll

private def Launch.staticFields (launch : Launch) : List F32 :=
  [launch.beta1, launch.oneMinusBeta1, launch.beta2, launch.oneMinusBeta2, launch.eps, launch.gradientScale]

def matchesPeriod (launch : Launch) (period : Period) : Prop :=
  Launch.staticFields launch = Launch.staticFields (period.launch launch.event launch.stepSize launch.decay)

instance (launch : Launch) (period : Period) : Decidable (matchesPeriod launch period) :=
  inferInstanceAs (Decidable (_ = _))

theorem matches_period_exact (launch : Launch) (period : Period) (h : matchesPeriod launch period) :
    launch = period.launch launch.event launch.stepSize launch.decay := by
  cases launch with
  | mk event beta1 firstMix beta2 secondMix eps stepSize decay scale =>
    simp only [matchesPeriod, Launch.staticFields, Period.launch, List.cons.injEq, and_true] at h
    rcases h with ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
    rfl

private def channelAt (index : Nat) : Channel :=
  ⟨index % 768, Nat.mod_lt _ (by decide)⟩

private def readFlat (memory : Memory) (index : Nat) : Cell :=
  if live : index / 768 < memory.rows then
    memory.read ⟨index / 768, live⟩ (channelAt index)
  else ⟨Value.zero .bf16, Value.zero .fp32, Value.zero .fp32⟩

private theorem read_flat_exact (memory : Memory) (row : Fin memory.rows) (column : Channel) :
    readFlat memory (row.val * 768 + column.val) = memory.read row column := by
  have columnBounds := column.isLt
  have rowBounds := row.isLt
  have rowIndex : (row.val * 768 + column.val) / 768 = row.val := by omega
  have channelIndex : channelAt (row.val * 768 + column.val) = column := by
    apply Fin.ext
    change (row.val * 768 + column.val) % 768 = column.val
    omega
  simp [readFlat, rowIndex, rowBounds, channelIndex]

private def shardIndex (memory : Memory) (rows : memory.rows = 25152)
    (row : Fin memory.rows) (column : Channel) : Fin (25152 * 768) :=
  ⟨row.val * 768 + column.val, by
    have a := row.isLt
    have b := column.isLt
    omega⟩

private def forAdam (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (arithmetic : Arithmetic) (launch : Launch) (memory : Memory) : Inputs :=
  ⟨readFlat memory, fun i => merge p idx order (i / 768) (channelAt i),
    launch.stepSize, launch.decay, arithmetic.divideFull⟩

private def decodedOutput {period tile source}
    (compiled : Certificate ⟨period, tile, true⟩ source) (inputs : Inputs)
    (memory : Memory) (rows : memory.rows = 25152) : Output memory :=
  let index := shardIndex memory rows
  ⟨fun row col => inputs.gradientWord true (index row col).val,
   fun row col => (compiled.updateShard inputs (index row col)).get (by
     rw [compiled.update_shard_eq]
     rfl)⟩

/-- Use the fused program for the matched shape and selected reference
configuration. Its logical gradient view is the FP16 value seen by Adam. -/
private def observedEvent (programs : NativePrograms) (period : Period) (clearAll : Bool)
    (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (arithmetic : Arithmetic) (launch : Launch) (memory : Memory) (rows : memory.rows = 25152) :
    Output memory :=
  decodedOutput (programs.fused period (arithmetic.tile launch) clearAll).artifact.2
    (forAdam p idx order arithmetic launch memory) memory rows

private theorem shard_coordinates (row : Nat) (col : Channel) :
    (row * 768 + col.val) / 768 = row ∧ channelAt (row * 768 + col.val) = col := by
  have bounded := col.isLt
  constructor
  · omega
  · apply Fin.ext
    change (row * 768 + col.val) % 768 = col.val
    omega

theorem observed_event_eq (programs : NativePrograms) (period : Period) (clearAll : Bool)
    (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (arithmetic : Arithmetic) (launch : Launch) (memory : Memory) (rows : memory.rows = 25152)
    (matched : matchesPeriod launch period) :
    observedEvent programs period clearAll p idx order arithmetic launch memory rows =
      event dense p idx order arithmetic launch memory := by
  have matchLaunch := matches_period_exact launch period matched
  have scale : launch.gradientScale = eighth := congrArg Launch.gradientScale matchLaunch
  have native : period.launch 0 launch.stepSize launch.decay =
      { launch with event := 0 } := by
    rw [matchLaunch]
    rfl
  simp only [observedEvent, decodedOutput, event, Output.mk.injEq]
  constructor
  · funext row col
    obtain ⟨rowIndex, channelIndex⟩ := shard_coordinates row.val col
    simp [shardIndex, forAdam, Inputs.gradientWord, rowIndex, channelIndex, dense,
      scale, scaleRead, halfScale]
  · funext row col
    apply Option.get_of_eq_some
    rw [Certificate.update_shard_eq]
    obtain ⟨rowIndex, channelIndex⟩ := shard_coordinates row.val col
    simp [shardIndex, forAdam, Inputs.gradientWord, read_flat_exact, rowIndex, channelIndex,
      native, ValueAdam.update, dense, scale, scaleRead, halfScale]

def checkedEvent (programs : NativePrograms) (p : Pull) (idx : AdamEntries.Indices p.ids)
    (order : Schedule p) (arithmetic : Arithmetic) (launch : Launch) (memory : Memory) : Output memory :=
  if rows : memory.rows = 25152 then
    let clearAll := decide (memory.rows < 2 * p.ids.count)
    if matchesPeriod launch .two then observedEvent programs .two clearAll p idx order arithmetic launch memory rows
    else if matchesPeriod launch .four then observedEvent programs .four clearAll p idx order arithmetic launch memory rows
    else event dense p idx order arithmetic launch memory
  else event dense p idx order arithmetic launch memory

theorem checked_event_eq (programs : NativePrograms) (p : Pull) (idx : AdamEntries.Indices p.ids)
    (order : Schedule p) (arithmetic : Arithmetic) (launch : Launch) (memory : Memory) :
    checkedEvent programs p idx order arithmetic launch memory =
      event dense p idx order arithmetic launch memory := by
  unfold checkedEvent
  split
  · rename_i rows
    split
    · exact observed_event_eq _ _ _ _ _ _ _ _ _ rows ‹matchesPeriod launch .two›
    · split
      · exact observed_event_eq _ _ _ _ _ _ _ _ _ rows ‹matchesPeriod launch .four›
      · rfl
  · rfl

def Consumer.observedWorkspace (consumer : Consumer period tile clearAll) (raw : Nat → Half)
    (history : History) (before : Workspace) (sparse : Bool) : Workspace :=
  let ready := if sparse then before.beforeScatter history else before.beforeDense history
  ⟨fun row col => if live : row < 25152 then
      consumer.artifact.2.gradientAfterShard raw ⟨row * 768 + col.val, by have := col.isLt; omega⟩
    else 0, ready.markers⟩

theorem Consumer.workspace_exact {period tile clearAll} (consumer : Consumer period tile clearAll)
    (raw : Nat → Half) (history : History) (before : Workspace) (sparse : Bool) :
    consumer.observedWorkspace raw history before sparse = before.consumed history sparse := by
  dsimp only [Consumer.observedWorkspace, Workspace.consumed]
  apply congrArg (fun gradient : HalfShard =>
    Workspace.mk gradient (if sparse then before.beforeScatter history else before.beforeDense history).markers)
  funext row col
  split
  · exact consumer.artifact.2.gradient_after_shard_zero
      (consumer.artifact.2.mode_consumes _ consumer.matched) _ _
  · rfl

theorem Consumer.workspace_represents {period tile clearAll} (consumer : Consumer period tile clearAll)
    (raw : Nat → Half) (history : History) (before : Workspace) (sparse : Bool)
    (valid : history.represents before) :
    (history.afterConsume sparse).represents (consumer.observedWorkspace raw history before sparse) := by
  rw [consumer.workspace_exact]
  exact consume_keeps_history history before sparse valid

private def workspaceInputs (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (arithmetic : Arithmetic) (launch : Launch) (memory : Memory)
    (history : History) (before : Workspace) (sparse : Bool) (scatter : HalfShard → HalfShard) : Inputs :=
  let ready := if sparse then before.beforeScatter history else before.beforeDense history
  { forAdam p idx order arithmetic launch memory with
    gradient := fun i => scatter ready.gradient (i / 768) (channelAt i) }

private theorem workspace_inputs_exact (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (arithmetic : Arithmetic) (launch : Launch) (memory : Memory)
    (history : History) (before : Workspace) (sparse : Bool) (valid : history.represents before) :
    workspaceInputs p idx order arithmetic launch memory history before sparse (mergeFrom p idx order) =
      forAdam p idx order arithmetic launch memory := by
  have clear : (if sparse then before.beforeScatter history else before.beforeDense history).gradient =
      Workspace.zero.gradient := by
    cases sparse
    · exact dense_preparation_zero _ _ valid
    · exact congrArg Workspace.gradient (sparse_preparation_zero _ _ valid)
  simp [workspaceInputs, clear, forAdam, Workspace.zero, merge]

/-- Observe the Adam results and the persistent scratch bytes from the
same decoded launch and the same input after scatter. -/
def consumingEventAfterScatter (programs : NativePrograms) (period : Period)
    (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (arithmetic : Arithmetic) (launch : Launch) (memory : Memory) (rows : memory.rows = 25152)
    (history : History) (before : Workspace) (sparse : Bool)
    (scatter : HalfShard → HalfShard) : Output memory × Workspace :=
  let inputs := workspaceInputs p idx order arithmetic launch memory history before sparse scatter
  let consumer := programs.fused period (arithmetic.tile launch) (!sparse)
  (decodedOutput consumer.artifact.2 inputs memory rows,
   consumer.observedWorkspace inputs.gradient history before sparse)

def consumingEvent (programs : NativePrograms) (period : Period)
    (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (arithmetic : Arithmetic) (launch : Launch) (memory : Memory) (rows : memory.rows = 25152)
    (history : History) (before : Workspace) (sparse : Bool) : Output memory × Workspace :=
  consumingEventAfterScatter programs period p idx order arithmetic launch memory rows history before sparse
    (mergeFrom p idx order)

theorem consuming_event_exact (programs : NativePrograms) (period : Period)
    (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (arithmetic : Arithmetic) (launch : Launch) (memory : Memory) (rows : memory.rows = 25152)
    (history : History) (before : Workspace) (sparse : Bool) (valid : history.represents before)
    (matched : matchesPeriod launch period) :
    consumingEvent programs period p idx order arithmetic launch memory rows history before sparse =
      (event dense p idx order arithmetic launch memory, before.consumed history sparse) := by
  unfold consumingEvent consumingEventAfterScatter
  rw [workspace_inputs_exact p idx order arithmetic launch memory history before sparse valid]
  exact Prod.ext (observed_event_eq programs period (!sparse) p idx order arithmetic launch memory rows matched)
    (Consumer.workspace_exact _ _ _ _ _)

#axiomsClean consuming_event_exact
#axiomsClean Consumer.workspace_exact
#axiomsClean Consumer.workspace_represents
#axiomsClean matches_period_exact
#axiomsClean observed_event_eq
#axiomsClean checked_event_eq


structure CombinedConsumer (period : Period) (tile : Tile) (masked : Bool) where
  oldPTX : String
  ptx : String
  oldSASS : String
  sass : String
  mode : DivisorBand.ZeroMode
  checked : NativeCombinedTrace.Certificate oldPTX ptx oldSASS sass mode
    (if period == .two then 2 else 4) masked tile.laneWidth

structure CombinedPrograms where
  consumer : (period : Period) → (tile : Tile) → (masked : Bool) → CombinedConsumer period tile masked

structure NativeDivider where
  reciprocal : F32 → F32
  nanProduct : Bool → F32 → F32 → F32
  roots : NativeCombinedTrace.RootOps

def NativeDivider.arithmetic (divider : NativeDivider) (base : Arithmetic) : Arithmetic :=
  CombinedDivisor.nativeArithmetic base.tile divider.reciprocal divider.nanProduct

private def CombinedConsumer.observe (consumer : CombinedConsumer period tile masked) (original : Bool)
    (divider : NativeDivider) (input : Inputs) (memory : Memory) (rows : memory.rows = 25152) :
    Output memory :=
  let index := shardIndex memory rows
  ⟨fun row col => input.gradientWord true (index row col).val,
   fun row col => (CombinedDivisor.runShard period tile consumer.checked original input divider.roots
      divider.reciprocal divider.nanProduct (index row col)).get (by
        rw [CombinedDivisor.run_shard_exact]
        rfl)⟩

private theorem CombinedConsumer.observe_eq (consumer : CombinedConsumer period tile masked)
    (divider : NativeDivider) (input : Inputs) (memory : Memory) (rows : memory.rows = 25152) :
    consumer.observe false divider input memory rows = consumer.observe true divider input memory rows := by
  simp only [CombinedConsumer.observe, Output.mk.injEq, true_and]
  funext row col
  have same : CombinedDivisor.runShard period tile consumer.checked false input divider.roots
      divider.reciprocal divider.nanProduct (shardIndex memory rows row col) =
    CombinedDivisor.runShard period tile consumer.checked true input divider.roots
      divider.reciprocal divider.nanProduct (shardIndex memory rows row col) := by
    rw [CombinedDivisor.run_shard_exact, CombinedDivisor.run_shard_exact]
  exact Option.get_congr same

/-- The two executions evaluate their own decoded consumers for the
matched launch profiles. Other profiles use the shared event. -/
def CombinedPrograms.run (programs : CombinedPrograms) (original : Bool) (divider : NativeDivider)
    (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (base : Arithmetic) (launch : Launch) (memory : Memory) : Output memory :=
  let plain := event dense p idx order (divider.arithmetic base) launch memory
  if rows : memory.rows = 25152 then
    let masked := !decide (memory.rows < 2 * p.ids.count)
    let tile := base.tile launch
    let input := forAdam p idx order (divider.arithmetic base) launch memory
    if matchesPeriod launch .two then
      (programs.consumer .two tile masked).observe original divider input memory rows
    else if matchesPeriod launch .four then
      (programs.consumer .four tile masked).observe original divider input memory rows
    else plain
  else plain

theorem CombinedPrograms.run_eq (programs : CombinedPrograms) (divider : NativeDivider)
    (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (base : Arithmetic) (launch : Launch) (memory : Memory) :
    programs.run false divider p idx order base launch memory =
      programs.run true divider p idx order base launch memory := by
  simp only [CombinedPrograms.run]
  by_cases rows : memory.rows = 25152
  · simp only [dif_pos rows]
    by_cases first : matchesPeriod launch .two
    · simp only [if_pos first, CombinedConsumer.observe_eq]
    · simp only [if_neg first]
      by_cases second : matchesPeriod launch .four
      · simp only [if_pos second, CombinedConsumer.observe_eq]
      · simp [second]
  · simp [rows]

#axiomsClean CombinedPrograms.run_eq

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
