import models.NanoGPT.Track1.ValueAdam

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam

open SparsePackets (Channel)

private theorem foldl_single_match {α β : Type} (choose : α → Bool)
    (f : β → α → β) (hits : ∀ i j, choose i = true → choose j = true → i = j)
    (slots : List α) (order : slots.Nodup) (start : β) :
    slots.foldl (fun prev slot => if choose slot then f prev slot else prev) start =
      ((slots.find? choose).map (f start)).getD start := by
  induction slots generalizing start with
  | nil => rfl
  | cons i tail ih =>
    obtain ⟨notIn, ordered⟩ := List.nodup_cons.mp order
    rw [List.foldl_cons]
    by_cases take : choose i = true
    · have miss : ∀ j ∈ tail, ¬choose j = true := by
        intro j inTail matched
        have equal := hits i j take matched
        exact notIn (equal ▸ inTail)
      have empty : tail.filter choose = [] := List.filter_eq_nil_iff.mpr miss
      simp only [take, if_true, List.find?_cons_of_pos take, Option.map_some, Option.getD_some]
      rw [← List.foldl_filter, empty]
      rfl
    · simp only [take, List.find?_cons_of_neg take]
      exact ih ordered start

def orderedLocalStore (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (before : HalfShard) : HalfShard :=
  fun row col =>
    ((order.localAtomics col).slots.find? (fun i => decide ((idx.localRows i).toInt = row))).elim
      (before row col) (fun i => halfAdd (before row col) (p.ownGradient i col))

def localStoreThenPeers (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (before : HalfShard) : HalfShard :=
  fun row col =>
    (order.peerAtomics col).slots.foldl (fun acc i =>
      if (idx.peerRows i).toInt = row then halfAdd acc (p.receivedGradient i col) else acc)
      (orderedLocalStore p idx order before row col)

theorem unique_owner_store_exact (p : Pull) (idx : AdamEntries.Indices p.ids)
    (order : Schedule p) (before : HalfShard)
    (distinct : Function.Injective fun i => (idx.localRows i).toInt) :
    localStoreThenPeers p idx order before = mergeFrom p idx order before := by
  funext row col
  unfold localStoreThenPeers mergeFrom orderedLocalStore
  have h := foldl_single_match (α := Fin p.ids.localCount) (β := Half)
    (fun i => decide ((idx.localRows i).toInt = row))
    (fun acc i => halfAdd acc (p.ownGradient i col))
    (fun i j hi hj => distinct ((of_decide_eq_true hi).trans (of_decide_eq_true hj).symm))
    (order.localAtomics col).slots
    ((order.localAtomics col).allSlots.nodup_iff.mpr (List.nodup_finRange _))
    (before row col)
  simp only [decide_eq_true_eq] at h
  rw [h]
  cases (order.localAtomics col).slots.find? (fun i => decide ((idx.localRows i).toInt = row)) <;> rfl

theorem SourceSlice.distinct_local_destinations (s : SourceSlice) :
    Function.Injective fun i => ((AdamEntries.fused s.ids).localRows i).toInt := by
  intro i j same
  change ((AdamEntries.fused s.ids).localRows i).toInt = ((AdamEntries.fused s.ids).localRows j).toInt at same
  rw [AdamEntries.local_exact, AdamEntries.local_exact] at same
  have row := congrArg Int.toNat same
  change s.localRow i = s.localRow j at row
  let col : Channel := ⟨0, by decide⟩
  exact (s.address_unique (x := col) (y := col) (by rw [row])).1

theorem SourceSlice.owner_store_exact (s : SourceSlice) (peer : Fin s.asked.size → Channel → Half)
    (order : Schedule (s.deferredPull peer)) (before : HalfShard) :
    localStoreThenPeers (s.deferredPull peer) (AdamEntries.fused s.ids) order before =
      mergeFrom (s.deferredPull peer) (AdamEntries.fused s.ids) order before :=
  unique_owner_store_exact _ _ _ _ s.distinct_local_destinations

#axiomsClean unique_owner_store_exact
#axiomsClean SourceSlice.distinct_local_destinations
#axiomsClean SourceSlice.owner_store_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam
