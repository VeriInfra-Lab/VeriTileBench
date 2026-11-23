import models.NanoGPT.Track1.ValueAdamScale

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam

open SparsePackets (Channel)

abbrev Line := (Channel → Half) × Nat

/-- An entry may read the saved position before the winner resets it,
or read zero afterward. In either case it writes only on equality to
its own positive position. -/
def clearEntry {n : Nat} (hit : Fin n → Bool) (line : Line) (i : Fin n) : Line :=
  if hit i && (line.2 == i.val + 1) then (fun _ => 0, 0) else line

private theorem clear_already_zero {n : Nat} (hit : Fin n → Bool) (slots : List (Fin n))
    (words : Channel → Half) : slots.foldl (clearEntry hit) (words, 0) = (words, 0) := by
  induction slots with
  | nil => rfl
  | cons i slots ih => simp [List.foldl_cons, clearEntry, ih]

theorem clear_entry_order {n : Nat} (hit : Fin n → Bool) (slots : List (Fin n)) (line : Line) :
    slots.foldl (clearEntry hit) line =
      if slots.any (fun i => hit i && (line.2 == i.val + 1)) then (fun _ => 0, 0) else line := by
  induction slots with
  | nil => rfl
  | cons i slots ih =>
    cases live : hit i && (line.2 == i.val + 1)
    · simp only [List.foldl_cons, clearEntry, live, Bool.false_eq_true, if_false, List.any_cons,
        Bool.false_or, ih]
    · simp only [List.foldl_cons, clearEntry, live, if_true, List.any_cons, Bool.true_or]
      exact clear_already_zero hit slots _

theorem clear_scattered_line (p : Pull) (idx : AdamEntries.Indices p.ids)
    (own : NgramAdam.Order p.ids.localCount) (peer : NgramAdam.Order p.ids.peerCount)
    (slots : NgramAdam.Order p.ids.count) (before : HalfShard) (row : Nat) :
    slots.slots.foldl (clearEntry fun i => decide ((scaleEntry idx i).toInt = row))
      (before row, scaleClaim p idx own peer row) =
      ((clearMarked (marked p idx) before).1 row, 0) := by
  rw [clear_entry_order]
  have winners : (slots.slots.any (fun i => decide ((scaleEntry idx i).toInt = row) &&
       (scaleClaim p idx own peer row == i.val + 1))) = true ↔ marked p idx row = true := by
    rw [slots.allSlots.any_eq, List.any_eq_true, ← writes_scale_iff p idx own peer row]
    simp only [List.mem_finRange, true_and, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq, writesScale]
  by_cases present : marked p idx row = true
  · simp only [winners.mpr present, if_true, clearMarked, present]
  · have zero : scaleClaim p idx own peer row = 0 := by
      by_contra nonzero
      exact present ((scale_claim_live p idx own peer row).mp nonzero)
    have none : ¬(slots.slots.any (fun i => decide ((scaleEntry idx i).toInt = row) &&
          (scaleClaim p idx own peer row == i.val + 1))) = true :=
      fun existsSlot => present (winners.mp existsSlot)
    rw [if_neg none]
    simp only [clearMarked, present, Bool.false_eq_true, if_false, zero]

def clearIndexedLine (previous : PriorMarked) (row : Nat) : Line :=
  let p := previous.pull
  let idx := previous.indices
  let schedule := previous.order
  let own := schedule.localAtomics 0
  let peer := schedule.peerAtomics 0
  let previousGradient := claimedFrom (fun _ _ => 0) p idx schedule eighth previous.rows
  (List.finRange p.ids.count).foldl (clearEntry fun i => decide ((scaleEntry idx i).toInt = row))
    (previousGradient row, scaleClaim p idx own peer row)

theorem clear_indexed_line_exact (previous : PriorMarked) (row : Nat) :
    clearIndexedLine previous row =
      ((clearMarked (marked previous.pull previous.indices)
        (selected previous.pull previous.indices previous.order eighth previous.rows)).1 row, 0) := by
  let entries : NgramAdam.Order previous.pull.ids.count := ⟨List.finRange _, List.Perm.refl _⟩
  simp only [clearIndexedLine, claimed_from_equal]
  exact clear_scattered_line previous.pull previous.indices
    (previous.order.localAtomics 0) (previous.order.peerAtomics 0) entries
    (selected previous.pull previous.indices previous.order eighth previous.rows) row

def clearIndexedWorkspace : WorkspaceNormalize := fun
  | none => claimedFrom (fun _ _ => 0)
  | some previous => claimedFrom (fun row => (clearIndexedLine previous row).1)

theorem clear_indexed_workspace_equal : clearIndexedWorkspace = clearSelected := by
  funext previous
  cases previous with
  | none => exact claimed_from_equal _
  | some previous =>
    have reset : (fun row => (clearIndexedLine previous row).1) =
        (clearMarked (marked previous.pull previous.indices)
          (selected previous.pull previous.indices previous.order eighth previous.rows)).1 := by
      funext row
      exact congrArg Prod.fst (clear_indexed_line_exact previous row)
    simp only [clearIndexedWorkspace, clearSelected, afterMarked, reset, claimed_from_equal]

#axiomsClean clear_entry_order
#axiomsClean clear_scattered_line
#axiomsClean clear_indexed_line_exact
#axiomsClean clear_indexed_workspace_equal

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam
