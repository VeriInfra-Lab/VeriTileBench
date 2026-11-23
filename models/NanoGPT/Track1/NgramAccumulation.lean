import models.NanoGPT.Track1.ValueAdam
import models.NanoGPT.Track1.CudaStreamOrder

/-! Each captured backward produces BF16 words indexed by this step's
cache slots. The original scatters the saved steps at the event. The
candidate scatters earlier steps immediately and keeps the event's own
scatter after the value-gradient send. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramAccumulation
open VeriTile.Triton FP
open SparsePackets (Channel Cut FP16Rows)

structure Step (rows : Nat) where
  entries : Nat
  slot : Fin entries → Fin rows
  gradient : Fin entries → Channel → Value .bf16
  order : Fin rows → Channel → NgramAdam.Order entries

def addSink (old : UInt16) (next : Value .bf16) : UInt16 :=
  ValueAdam.halfAdd old (ValueAdam.narrow (FP.cast .fp32 Config.ieeeValue next))

/-- The atomic converts BF16 to FP16 before adding. Each channel of a
destination row observes a permutation of that step's contributors. -/
def scatter {rows : Nat} (before : FP16Rows) (step : Step rows) : FP16Rows := fun row col =>
  if inside : row < rows then
    (step.order ⟨row, inside⟩ col).slots.foldl (fun old i =>
      if step.slot i = ⟨row, inside⟩ then addSink old (step.gradient i col) else old) (before row col)
  else before row col

def zero (rows : Nat) (before : FP16Rows) : FP16Rows :=
  fun row col => if row < rows then 0 else before row col

structure Cycle where
  cut : Cut
  prior : FP16Rows
  earlier : List (Step cut.rows)
  event : Step cut.rows

abbrev Accumulator := Cycle → FP16Rows

def delayed : Accumulator := fun cycle =>
  (cycle.earlier ++ [cycle.event]).foldl scatter (zero cycle.cut.rows cycle.prior)

/-- `none` represents scratch still carrying the previous cycle's data.
The first scatter zeroes the active prefix just before reading it. -/
def offer {rows : Nat} (prior : FP16Rows) (ready : Option FP16Rows) (step : Step rows) : Option FP16Rows :=
  some (scatter (ready.getD (zero rows prior)) step)

def online : Accumulator := fun cycle =>
  let ready := cycle.earlier.foldl (offer cycle.prior) none
  scatter (ready.getD (zero cycle.cut.rows cycle.prior)) cycle.event

private theorem offer_prefix {rows : Nat} (prior : FP16Rows) (steps : List (Step rows)) (ready : Option FP16Rows) :
    (steps.foldl (offer prior) ready).getD (zero rows prior) =
      steps.foldl scatter (ready.getD (zero rows prior)) := by
  induction steps generalizing ready with
  | nil => rfl
  | cons step rest ih => simp only [List.foldl_cons, ih, offer, Option.getD_some]

theorem accumulator_equal : online = delayed := by
  funext cycle
  simp only [online, delayed, offer_prefix, Option.getD_none, List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem step_order_changes_half_word :
    addSink (addSink (addSink 0 (Value.ofBits .bf16 0x4500)) (Value.ofBits .bf16 0x3f80))
      (Value.ofBits .bf16 0xc500) = 0 ∧
    addSink (addSink (addSink 0 (Value.ofBits .bf16 0x4500)) (Value.ofBits .bf16 0xc500))
      (Value.ofBits .bf16 0x3f80) = 0x3c00 := by decide +kernel

theorem scatter_outside {rows : Nat} (before : FP16Rows) (step : Step rows)
    (row : Nat) (col : Channel) (outside : rows ≤ row) : scatter before step row col = before row col := by
  simp only [scatter, dif_neg (Nat.not_lt_of_ge outside)]

theorem cycle_tail (cycle : Cycle) (row : Nat) (col : Channel) (outside : cycle.cut.rows ≤ row) :
    online cycle row col = cycle.prior row col := by
  have folded (steps : List (Step cycle.cut.rows)) (initial : FP16Rows) :
      steps.foldl scatter initial row col = initial row col := by
    induction steps generalizing initial with
    | nil => rfl
    | cons step rest ih => rw [List.foldl_cons, ih, scatter_outside _ _ _ _ outside]
  rw [accumulator_equal, delayed, folded]
  simp [zero, Nat.not_lt_of_ge outside]

inductive Work (rows : Nat) where
  | backward (step : Step rows)
  | accumulate
  | event

structure State (rows : Nat) where
  graph : Option (Step rows)
  scratch : Option FP16Rows
  sent : Option FP16Rows

def executeFrom {rows : Nat} (seed : FP16Rows) (state : State rows) :
    CudaStreamOrder.Command (Work rows) → Option (State rows)
  | .work (.backward step) => some {state with graph := some step}
  | .work .accumulate => do
    let step ← state.graph
    pure {state with graph := none, scratch := some (scatter (state.scratch.getD seed) step)}
  | .work .event => do
    let step ← state.graph
    pure ⟨none, none, some (scatter (state.scratch.getD seed) step)⟩
  | .record _ | .wait _ => some state

def execute {rows : Nat} (prior : FP16Rows) (state : State rows) :
    CudaStreamOrder.Command (Work rows) → Option (State rows) :=
  executeFrom (zero rows prior) state

def earlierCommands {rows : Nat} (steps : List (Step rows)) : List (CudaStreamOrder.Command (Work rows)) :=
  steps.flatMap fun step => [.work (.backward step), .work .accumulate]

def commands (cycle : Cycle) : List (CudaStreamOrder.Command (Work cycle.cut.rows)) :=
  earlierCommands cycle.earlier ++ [.work (.backward cycle.event), .work .event]

private theorem execute_earlier {rows : Nat} (prior : FP16Rows) (steps : List (Step rows))
    (scratch sent : Option FP16Rows) :
    (earlierCommands steps).foldlM (execute prior) ⟨none, scratch, sent⟩ =
      some ⟨none, steps.foldl (offer prior) scratch, sent⟩ := by
  induction steps generalizing scratch with
  | nil => rfl
  | cons step rest ih =>
    simp only [earlierCommands, List.flatMap_cons, List.cons_append, List.nil_append, List.foldlM_cons,
      execute, executeFrom, List.foldl_cons, offer]
    exact ih _

/-- The graph's address can be reused on every backward. Its sink is
consumed on the compute stream before the next replay writes it. -/
theorem queued_cycle (cycle : Cycle) {initial final : CudaStreamOrder.Queues (Work cycle.cut.rows)}
    {issued : List (Nat × CudaStreamOrder.Command (Work cycle.cut.rows))} (compute : Nat)
    (submitted : initial.pending compute = commands cycle)
    (run : CudaStreamOrder.Run initial issued final) (finished : final.pending compute = []) :
    (CudaStreamOrder.issuedOn compute issued).foldlM (execute cycle.prior) ⟨none, none, none⟩ =
      some ⟨none, none, some (delayed cycle)⟩ := by
  rw [CudaStreamOrder.fifo_complete run compute finished, submitted, commands, List.foldlM_append,
    execute_earlier, ← accumulator_equal]
  rfl

/-- The host only needs to accumulate steps at or before the final
update position. The remaining captures still run for the dense model. -/
def hasConsumer (updates : List Nat) (position : Nat) : Bool :=
  (updates.getLast?).any fun last => position ≤ last

theorem has_consumer (updates : List Nat) (ordered : updates.Pairwise (· ≤ ·)) (position : Nat) :
    hasConsumer updates position = true ↔ ∃ finish ∈ updates, position ≤ finish := by
  induction updates with
  | nil => simp [hasConsumer]
  | cons first tail ih =>
    cases tail with
    | nil => simp [hasConsumer]
    | cons next rest =>
      have earlier := (List.pairwise_cons.mp ordered).1 next (by simp)
      change hasConsumer (next :: rest) position = true ↔ _
      rw [ih (List.pairwise_cons.mp ordered).2]
      constructor
      · rintro ⟨finish, member, time⟩
        exact ⟨finish, List.mem_cons_of_mem _ member, time⟩
      · rintro ⟨finish, member, time⟩
        rcases List.mem_cons.mp member with same | later
        · subst finish
          exact ⟨next, by simp, Nat.le_trans time earlier⟩
        · exact ⟨finish, later, time⟩

def updatePositions (steps : List Nat) (due : Nat → Bool) : List Nat :=
  (List.range steps.length).filter fun i => (steps[i]?).any due

theorem scheduled_consumer (steps : List Nat) (due : Nat → Bool) (position : Nat) :
    hasConsumer (updatePositions steps due) position = true ↔
      ∃ finish step, position ≤ finish ∧ steps[finish]? = some step ∧ due step = true := by
  have ordered : (updatePositions steps due).Pairwise (· ≤ ·) :=
    (List.pairwise_lt_range.imp Nat.le_of_lt).filter _
  rw [has_consumer _ ordered]
  constructor
  · rintro ⟨finish, member, after⟩
    have fires := (List.mem_filter.mp member).2
    cases entry : steps[finish]? with
    | none => simp [entry] at fires
    | some step => exact ⟨finish, step, after, entry, by simpa [entry] using fires⟩
  · rintro ⟨finish, step, after, entry, updates⟩
    have inside := (List.getElem?_eq_some_iff.mp entry).1
    exact ⟨finish, List.mem_filter.mpr ⟨List.mem_range.mpr inside, by simp [entry, updates]⟩, after⟩

#axiomsClean accumulator_equal
#axiomsClean cycle_tail
#axiomsClean queued_cycle
#axiomsClean step_order_changes_half_word
#axiomsClean has_consumer
#axiomsClean scheduled_consumer
end VeriTile.Bench.Models.NanoGPT.Track1.NgramAccumulation
