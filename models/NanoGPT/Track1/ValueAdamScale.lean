import models.NanoGPT.Track1.ValueAdam

/-! The scatter stores the maximum of the one-based positions of the
entries contributing to a row. The scaling grid visits those entries and
only the one holding the row's saved position writes. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam

open NgramAdam (Order)

def scaleEntry {s : AdamEntries.Input} (idx : AdamEntries.Indices s) (i : Fin s.count) : Int32 :=
  AdamEntries.concatenate s idx.localRows idx.peerRows i

def scaleSlots (s : AdamEntries.Input) (own : Order s.localCount) (peer : Order s.peerCount) :
    List (Fin s.count) :=
  own.slots.map (AdamEntries.localSlot s) ++ peer.slots.map (AdamEntries.peerSlot s)

theorem mem_scaleSlots (s : AdamEntries.Input) (own : Order s.localCount) (peer : Order s.peerCount)
    (i : Fin s.count) : i ∈ scaleSlots s own peer := by
  simp only [scaleSlots, List.mem_append, List.mem_map]
  by_cases isLocal : i.val < s.localCount
  · exact Or.inl ⟨⟨i.val, isLocal⟩, own.allSlots.mem_iff.mpr (List.mem_finRange _), by rfl⟩
  · have inside := i.isLt
    let slot : Fin s.peerCount := ⟨i.val - s.localCount, by unfold AdamEntries.Input.count at inside; omega⟩
    exact Or.inr ⟨slot, peer.allSlots.mem_iff.mpr (List.mem_finRange _), Fin.ext (by
      change s.localCount + (i.val - s.localCount) = i.val
      omega)⟩

theorem scale_entry_fused (s : AdamEntries.Input) (i : Fin s.count) :
    scaleEntry (AdamEntries.fused s) i = AdamEntries.packedRead s i := by
  rw [congrFun AdamEntries.pack_equal s, AdamEntries.packed_exact]
  rfl

private theorem scale_entry_hits (p : Pull) (idx : AdamEntries.Indices p.ids) (row : Nat) :
    (∃ i, (scaleEntry idx i).toInt = row) ↔ marked p idx row = true := by
  rw [marked_iff]
  constructor
  · rintro ⟨i, word⟩
    unfold scaleEntry AdamEntries.concatenate at word
    split at word
    · exact .inl ⟨_, word⟩
    · exact .inr ⟨_, word⟩
  · rintro (⟨i, word⟩ | ⟨i, word⟩)
    · exact ⟨AdamEntries.localSlot p.ids i, by
        simpa [scaleEntry, AdamEntries.concatenate, AdamEntries.localSlot] using word⟩
    · refine ⟨AdamEntries.peerSlot p.ids i, ?_⟩
      have outside : ¬p.ids.localCount + i.val < p.ids.localCount := by omega
      simpa [scaleEntry, AdamEntries.concatenate, AdamEntries.peerSlot, outside] using word

private def updateClaim {n : Nat} (hit : Fin n → Bool) (saved : Nat) (i : Fin n) : Nat :=
  if hit i then max saved (i.val + 1) else saved

private theorem claim_fold {n : Nat} (hit : Fin n → Bool) (slots : List (Fin n)) (saved : Nat) :
    let result := slots.foldl (updateClaim hit) saved
    (result = saved ∨ ∃ i ∈ slots, hit i = true ∧ result = i.val + 1) ∧
      saved ≤ result ∧ ∀ i ∈ slots, hit i = true → i.val + 1 ≤ result := by
  induction slots generalizing saved with
  | nil => simp
  | cons slot rest ih =>
    simp only [List.foldl_cons]
    set next := updateClaim hit saved slot
    obtain ⟨origin, prior, bound⟩ := ih next
    have grow : saved ≤ next := by
      unfold next updateClaim
      split <;> omega
    have chosen : next = saved ∨ hit slot = true ∧ next = slot.val + 1 := by
      unfold next updateClaim
      split
      · rename_i h
        by_cases before : slot.val + 1 ≤ saved
        · exact .inl (Nat.max_eq_left before)
        · exact .inr ⟨h, Nat.max_eq_right (by omega)⟩
      · exact .inl rfl
    refine ⟨?_, grow.trans prior, ?_⟩
    · rcases origin with fromNext | ⟨i, member, hitWord, eq⟩
      · rcases chosen with old | ⟨hitWord, eq⟩
        · exact .inl (fromNext.trans old)
        · exact .inr ⟨slot, List.mem_cons_self, hitWord, fromNext.trans eq⟩
      · exact .inr ⟨i, List.mem_cons_of_mem _ member, hitWord, eq⟩
    · intro i inside hitWord
      rcases List.mem_cons.mp inside with same | later
      · subst i
        have includes : slot.val + 1 ≤ next := by
          simp only [next, updateClaim, hitWord, if_true]
          exact Nat.le_max_right _ _
        exact includes.trans prior
      · exact bound i later hitWord

def scaleClaim (p : Pull) (idx : AdamEntries.Indices p.ids)
    (own : Order p.ids.localCount) (peer : Order p.ids.peerCount) (row : Nat) : Nat :=
  (scaleSlots p.ids own peer).foldl (updateClaim fun i => decide ((scaleEntry idx i).toInt = row)) 0

theorem scale_claim_order_independent (p : Pull) (idx : AdamEntries.Indices p.ids) (row : Nat)
    (own₁ own₂ : Order p.ids.localCount) (peer₁ peer₂ : Order p.ids.peerCount) :
    scaleClaim p idx own₁ peer₁ row = scaleClaim p idx own₂ peer₂ row := by
  have ordered (own otherOwn : Order p.ids.localCount) (peer otherPeer : Order p.ids.peerCount) :
      scaleClaim p idx own peer row ≤ scaleClaim p idx otherOwn otherPeer row := by
    have origin := (claim_fold (fun i => decide ((scaleEntry idx i).toInt = row))
      (scaleSlots p.ids own peer) 0).1
    change scaleClaim p idx own peer row = 0 ∨ _ at origin
    rcases origin with zero | ⟨slot, _, hit, claim⟩
    · simp only [zero, Nat.zero_le]
    · have bound := (claim_fold (fun i => decide ((scaleEntry idx i).toInt = row))
        (scaleSlots p.ids otherOwn otherPeer) 0).2.2
      change scaleClaim p idx own peer row = slot.val + 1 at claim
      rw [claim]
      exact bound slot (mem_scaleSlots p.ids otherOwn otherPeer slot) hit
  exact Nat.le_antisymm (ordered own₁ own₂ peer₁ peer₂) (ordered own₂ own₁ peer₂ peer₁)

def writesScale (p : Pull) (idx : AdamEntries.Indices p.ids)
    (own : Order p.ids.localCount) (peer : Order p.ids.peerCount) (row : Nat)
    (i : Fin p.ids.count) : Prop :=
  (scaleEntry idx i).toInt = row ∧ scaleClaim p idx own peer row = i.val + 1

theorem writes_scale_iff (p : Pull) (idx : AdamEntries.Indices p.ids)
    (own : Order p.ids.localCount) (peer : Order p.ids.peerCount) (row : Nat) :
    (∃ i, writesScale p idx own peer row i) ↔ marked p idx row = true := by
  rw [← scale_entry_hits]
  constructor
  · rintro ⟨i, hit, _⟩
    exact ⟨i, hit⟩
  · rintro ⟨i, hit⟩
    obtain ⟨origin, _, largest⟩ :=
      claim_fold (fun i => decide ((scaleEntry idx i).toInt = row)) (scaleSlots p.ids own peer) 0
    have nonzero := largest i (mem_scaleSlots p.ids own peer i) (by simp [hit])
    rcases origin with zero | ⟨j, _, target, ticket⟩
    · omega
    · exact ⟨j, of_decide_eq_true target, ticket⟩

theorem writes_scale_unique (p : Pull) (idx : AdamEntries.Indices p.ids)
    (own : Order p.ids.localCount) (peer : Order p.ids.peerCount) (row : Nat)
    (i j : Fin p.ids.count) (first : writesScale p idx own peer row i)
    (second : writesScale p idx own peer row j) : i = j := by
  apply Fin.ext
  have same := first.2.symm.trans second.2
  omega

theorem scale_claim_live (p : Pull) (idx : AdamEntries.Indices p.ids)
    (own : Order p.ids.localCount) (peer : Order p.ids.peerCount) (row : Nat) :
    scaleClaim p idx own peer row ≠ 0 ↔ marked p idx row = true := by
  rw [← writes_scale_iff p idx own peer row]
  constructor
  · intro nonzero
    have claim := (claim_fold (fun i => decide ((scaleEntry idx i).toInt = row)) (scaleSlots p.ids own peer) 0).1
    change scaleClaim p idx own peer row = 0 ∨ _ at claim
    rcases claim with zero | ⟨slot, _, hit, ticket⟩
    · contradiction
    · exact ⟨slot, of_decide_eq_true hit, ticket⟩
  · rintro ⟨i, _, eq⟩
    omega

def claimedFrom (before : HalfShard) : Normalize := fun p idx schedule factor rows row col =>
  let value := mergeFrom p idx schedule before row col
  if factor = eighth ∧ 2 * p.ids.count ≤ rows ∧
      scaleClaim p idx (schedule.localAtomics col) (schedule.peerAtomics col) row = 0 then value
  else halfScale factor value

theorem claimed_from_equal (before : HalfShard) : claimedFrom before = selectedFrom before := by
  funext p idx schedule factor rows row col
  have empty : scaleClaim p idx (schedule.localAtomics col) (schedule.peerAtomics col) row = 0 ↔
      Bool.not (marked p idx row) = true := by
    have found := scale_claim_live p idx (schedule.localAtomics col) (schedule.peerAtomics col) row
    constructor
    · intro zero
      cases hit : marked p idx row with
      | false => rfl
      | true => exact False.elim (found.mpr hit zero)
    · intro miss
      by_contra nonzero
      simp only [found.mp nonzero, Bool.not_true, Bool.false_eq_true] at miss
  simp only [claimedFrom, selectedFrom, empty]

def claimedFlags (p : Pull) (idx : AdamEntries.Indices p.ids)
    (own : Order p.ids.localCount) (peer : Order p.ids.peerCount) (row : Nat) : Bool :=
  decide (scaleClaim p idx own peer row ≠ 0)

theorem claimed_flags_equal (p : Pull) (idx : AdamEntries.Indices p.ids)
    (own : Order p.ids.localCount) (peer : Order p.ids.peerCount) :
    claimedFlags p idx own peer = marked p idx := by
  funext row
  unfold claimedFlags
  by_cases nonzero : scaleClaim p idx own peer row ≠ 0
  · rw [decide_eq_true nonzero, (scale_claim_live p idx own peer row).mp nonzero]
  · have original : marked p idx row = false := by
      cases found : marked p idx row
      · rfl
      · exact False.elim (nonzero ((scale_claim_live p idx own peer row).mpr found))
    rw [decide_eq_false nonzero, original]

def clearClaimed : WorkspaceNormalize := fun
  | none => claimedFrom (fun _ _ => 0)
  | some prev =>
    let flags := claimedFlags prev.pull prev.indices (prev.order.localAtomics 0) (prev.order.peerAtomics 0)
    let scaled := claimedFrom (fun _ _ => 0) prev.pull prev.indices prev.order eighth prev.rows
    claimedFrom (clearMarked flags scaled).1

theorem clear_claimed_equal : clearClaimed = clearSelected := by
  funext previous
  cases previous with
  | none => exact claimed_from_equal _
  | some prev =>
    simp only [clearClaimed, clearSelected, afterMarked, claimed_flags_equal, claimed_from_equal, selected]

theorem signed_ticket (rows count : Nat) (slot : Fin count)
    (sparse : 2 * count ≤ rows) (size : (rows + 4) * 768 < 2 ^ 31) :
    (Int32.ofNat (slot.val + 1)).toInt = (slot.val + 1 : Nat) :=
  Int32.toInt_ofNat_of_lt (by have := slot.isLt; omega)

theorem scale_grid_bound (rows count : Nat) (sparse : 2 * count ≤ rows) :
    (count + 3) / 4 ≤ (rows + 7) / 8 := by omega

#axiomsClean scale_entry_fused
#axiomsClean writes_scale_iff
#axiomsClean writes_scale_unique
#axiomsClean scale_claim_order_independent
#axiomsClean scale_claim_live
#axiomsClean clear_claimed_equal
#axiomsClean signed_ticket
#axiomsClean scale_grid_bound

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam
