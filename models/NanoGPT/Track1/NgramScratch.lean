import models.NanoGPT.Track1.NgramAccumulation

/-! The wire pack reads the peer rows before clearing them. The owned
slice stays live until the Adam scatter reads it. At that point both
routes have consumed their input, so the next cycle can reuse the scratch. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramScratch
open VeriTile.Triton FP
open SparsePackets (Cut Channel FP16Rows Packets NgramInput)

abbrev Payload (cut : Cut) := Packets cut (Value .bf16)

def clearPeer (cut : Cut) (before : FP16Rows) : FP16Rows :=
  fun row col => if row < cut.rows ∧ (row < cut.lo ∨ cut.hi ≤ row) then 0 else before row col

def clearOwn (cut : Cut) (before : FP16Rows) : FP16Rows :=
  fun row col => if cut.lo ≤ row ∧ row < cut.hi then 0 else before row col

def send (fused : Bool) (input : NgramInput) : Payload input.cut × FP16Rows :=
  let wire := (SparsePackets.ngramDirect input).wire
  let live := if fused then clearPeer input.cut input.grad else input.grad
  (⟨fun i col => live (SparsePackets.ownSlot input.cut i).val col, wire⟩, live)

def finish (fused : Bool) (cut : Cut) (before : FP16Rows) : FP16Rows :=
  if fused then clearOwn cut before else NgramAccumulation.zero cut.rows before

theorem owned_slice_live (cut : Cut) (before : FP16Rows) (i : Fin cut.own) (col : Channel) :
    clearPeer cut before (SparsePackets.ownSlot cut i).val col = before (SparsePackets.ownSlot cut i).val col := by
  have ordered := cut.ordered
  have inside := i.isLt
  change i.val < cut.hi - cut.lo at inside
  simp only [clearPeer, SparsePackets.ownSlot]
  rw [if_neg (by omega)]

theorem peer_slot_erased (cut : Cut) (before : FP16Rows) (i : Fin cut.peers) (col : Channel) :
    clearPeer cut before (SparsePackets.peerSlot cut i).val col = 0 := by
  have live := (SparsePackets.peerSlot cut i).isLt
  have ordered := cut.ordered
  have outside : (SparsePackets.peerSlot cut i).val < cut.lo ∨
      cut.hi ≤ (SparsePackets.peerSlot cut i).val := by
    simp only [SparsePackets.peerSlot, Cut.own]
    split <;> omega
  exact if_pos ⟨live, outside⟩

theorem own_slot_erased (cut : Cut) (before : FP16Rows) (i : Fin cut.own) (col : Channel) :
    clearOwn cut before (SparsePackets.ownSlot cut i).val col = 0 := by
  have ordered := cut.ordered
  have inside := i.isLt
  change i.val < cut.hi - cut.lo at inside
  simp only [clearOwn, SparsePackets.ownSlot]
  rw [if_pos (by omega)]

theorem own_source_unique (cut : Cut) : Function.Injective (SparsePackets.ownSlot cut) := by
  intro i j equal
  apply Fin.ext
  have offsets := congrArg Fin.val equal
  change cut.lo + i.val = cut.lo + j.val at offsets
  omega

theorem peer_source_unique (cut : Cut) : Function.Injective (SparsePackets.peerSlot cut) := by
  intro i j equal
  apply Fin.ext
  have offsets := congrArg Fin.val equal
  change i.val + (if i.val < cut.lo then 0 else cut.own) =
    j.val + (if j.val < cut.lo then 0 else cut.own) at offsets
  split at offsets <;> split at offsets <;> omega

theorem every_slot_consumed (cut : Cut) (row : Fin cut.rows) :
    (∃ i, SparsePackets.ownSlot cut i = row) ∨ (∃ i, SparsePackets.peerSlot cut i = row) := by
  have live := row.isLt
  have within := cut.within
  have ordered := cut.ordered
  by_cases front : row.val < cut.lo
  · let slot : Fin cut.peers := ⟨row.val, by unfold Cut.peers Cut.own; omega⟩
    exact Or.inr ⟨slot, Fin.ext (by simp [SparsePackets.peerSlot, slot, front])⟩
  · by_cases back : cut.hi ≤ row.val
    · let slot : Fin cut.peers := ⟨row.val - cut.own, by unfold Cut.peers Cut.own; omega⟩
      have outside : ¬slot.val < cut.lo := by simp only [slot, Cut.own]; omega
      refine Or.inr ⟨slot, Fin.ext ?_⟩
      change slot.val + (if slot.val < cut.lo then 0 else cut.own) = row.val
      rw [if_neg outside]
      change row.val - (cut.hi - cut.lo) + (cut.hi - cut.lo) = row.val
      omega
    · let slot : Fin cut.own := ⟨row.val - cut.lo, by unfold Cut.own; omega⟩
      exact Or.inl ⟨slot, Fin.ext (by simp only [SparsePackets.ownSlot, slot]; omega)⟩

theorem send_payload (fused : Bool) (input : NgramInput) :
    (send fused input).1 = SparsePackets.ngramDirect input := by
  cases fused
  · rfl
  · change Packets.mk _ _ = Packets.mk _ _
    exact Packets.mk.injEq .. |>.mpr ⟨funext fun i => funext (owned_slice_live input.cut input.grad i), rfl⟩

theorem finish_clears_active (fused : Bool) (input : NgramInput) :
    finish fused input.cut (send fused input).2 = NgramAccumulation.zero input.cut.rows input.grad := by
  cases fused
  · rfl
  · funext row col
    have within := input.cut.within
    simp only [finish, send, if_true, clearOwn, clearPeer, NgramAccumulation.zero]
    by_cases own : input.cut.lo ≤ row ∧ row < input.cut.hi
    · have inside : row < input.cut.rows := by omega
      simp [own, inside]
    · simp only [if_neg own]
      have peer : row < input.cut.lo ∨ input.cut.hi ≤ row := by omega
      simp [peer]

def Clean (capacity : Nat) (scratch : FP16Rows) : Prop :=
  ∀ row < capacity, ∀ col, scratch row col = 0

theorem reset_ready (capacity : Nat) (before : FP16Rows) :
    Clean capacity (NgramAccumulation.zero capacity before) := by
  intro row inside col
  exact if_pos inside

def sum (cycle : NgramAccumulation.Cycle) : FP16Rows :=
  NgramAccumulation.scatter (cycle.earlier.foldl NgramAccumulation.scatter cycle.prior) cycle.event

theorem ready_prefix (scratch : FP16Rows) (rows capacity : Nat) (fit : rows ≤ capacity)
    (ready : Clean capacity scratch) : NgramAccumulation.zero rows scratch = scratch := by
  funext row col
  by_cases live : row < rows
  · simp only [NgramAccumulation.zero, if_pos live, ready row (Nat.lt_of_lt_of_le live fit) col]
  · simp only [NgramAccumulation.zero, if_neg live]

theorem sum_from_ready (cycle : NgramAccumulation.Cycle) (capacity : Nat) (fit : cycle.cut.rows ≤ capacity)
    (ready : Clean capacity cycle.prior) : sum cycle = NgramAccumulation.delayed cycle := by
  rw [NgramAccumulation.delayed, ready_prefix _ _ _ fit ready, List.foldl_append, List.foldl_cons, List.foldl_nil]
  rfl

private theorem scatter_reads_row {rows : Nat} (steps : List (NgramAccumulation.Step rows))
    (a b : FP16Rows) (row : Nat) (col : Channel) (same : a row col = b row col) :
    steps.foldl NgramAccumulation.scatter a row col = steps.foldl NgramAccumulation.scatter b row col := by
  induction steps generalizing a b with
  | nil => exact same
  | cons step rest ih =>
    simp only [List.foldl_cons]
    apply ih
    simp only [NgramAccumulation.scatter, same]

private theorem delayed_payload (cycle : NgramAccumulation.Cycle) (different : FP16Rows) :
    SparsePackets.ngramDirect ⟨cycle.cut, NgramAccumulation.delayed {cycle with prior := different}⟩ =
      SparsePackets.ngramDirect ⟨cycle.cut, NgramAccumulation.delayed cycle⟩ := by
  have active (row : Fin cycle.cut.rows) (col : Channel) :
      NgramAccumulation.delayed {cycle with prior := different} row.val col =
        NgramAccumulation.delayed cycle row.val col := by
    unfold NgramAccumulation.delayed
    apply scatter_reads_row
    simp only [NgramAccumulation.zero, if_pos row.isLt]
  change Packets.mk _ _ = Packets.mk _ _
  exact Packets.mk.injEq .. |>.mpr
    ⟨funext fun i => funext (active (SparsePackets.ownSlot cycle.cut i)),
      funext fun i => funext fun col => congrArg SparsePackets.wireCast (active (SparsePackets.peerSlot cycle.cut i) col)⟩

structure Event (capacity : Nat) where
  cycle : NgramAccumulation.Cycle
  fit : cycle.cut.rows ≤ capacity
  fused : Bool

def execute {capacity : Nat} (events : List (Event capacity)) (scratch : FP16Rows) :
    List (Σ cut, Payload cut) × FP16Rows :=
  match events with
  | [] => ([], scratch)
  | event :: rest =>
    let input : NgramInput := ⟨event.cycle.cut, sum {event.cycle with prior := scratch}⟩
    let (packet, live) := send event.fused input
    let (later, last) := execute rest (finish event.fused input.cut live)
    (⟨input.cut, packet⟩ :: later, last)

def reference {capacity : Nat} (event : Event capacity) : Σ cut, Payload cut :=
  ⟨event.cycle.cut, SparsePackets.ngramStaged ⟨event.cycle.cut, NgramAccumulation.delayed event.cycle⟩⟩

/-- Reset initially clears the allocation. Every completed event restores
it, for changing cuts and any mixture of fused and fallback launches. -/
theorem reuse_cycles {capacity : Nat} (events : List (Event capacity)) (scratch : FP16Rows)
    (ready : Clean capacity scratch) :
    (execute events scratch).1 = events.map reference ∧ Clean capacity (execute events scratch).2 := by
  induction events generalizing scratch with
  | nil => exact ⟨rfl, ready⟩
  | cons event rest ih =>
    let cycle := {event.cycle with prior := scratch}
    let input : NgramInput := ⟨event.cycle.cut, sum cycle⟩
    let next := finish event.fused input.cut (send event.fused input).2
    have sums := sum_from_ready cycle capacity event.fit ready
    have nextReady : Clean capacity next := by
      intro row inside col
      change finish event.fused input.cut (send event.fused input).2 row col = 0
      rw [finish_clears_active]
      by_cases live : row < cycle.cut.rows
      · exact if_pos live
      · change NgramAccumulation.zero cycle.cut.rows (sum cycle) row col = _
        rw [NgramAccumulation.zero, if_neg live, sums, ← NgramAccumulation.accumulator_equal,
          NgramAccumulation.cycle_tail cycle row col (Nat.le_of_not_gt live)]
        exact ready row inside col
    obtain ⟨later, clean⟩ := ih next nextReady
    have packet : (send event.fused input).1 =
        SparsePackets.ngramStaged ⟨event.cycle.cut, NgramAccumulation.delayed event.cycle⟩ := by
      rw [send_payload]
      change SparsePackets.ngramDirect ⟨event.cycle.cut, sum cycle⟩ = _
      rw [sums, delayed_payload, SparsePackets.ngram_packets_equal]
    refine ⟨?_, clean⟩
    change Sigma.mk input.cut (send event.fused input).1 :: (execute rest next).1 = (event :: rest).map reference
    rw [packet, later]
    rfl

#axiomsClean send_payload
#axiomsClean finish_clears_active
#axiomsClean every_slot_consumed
#axiomsClean own_source_unique
#axiomsClean peer_source_unique
#axiomsClean ready_prefix
#axiomsClean reuse_cycles
end VeriTile.Bench.Models.NanoGPT.Track1.NgramScratch
