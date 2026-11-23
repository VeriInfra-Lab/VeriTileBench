import models.NanoGPT.Track1.RowExchange

/-! Reverse the cached pull's route for the BF16 n-gram gradients.
The owner receives one entry per requested row, in the order of its
existing `asked` array. Duplicate IDs keep each source's contribution. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.RowGradientExchange
open VeriTile.Triton FP
open NgramSlots (Want)
open SparsePackets (Channel FP16Rows Cut)
open RowExchange (localCut packetSlots sendCounts)

def wireSlots (want : Want) (source : Nat) : List Nat :=
  (List.range 8).flatMap (packetSlots want source)

private theorem wire_slot (want : Want) (source : Fin 8) (i : Fin (localCut want source).peers) :
    (wireSlots want source)[i.val]? = some (SparsePackets.peerSlot (localCut want source) i).val := by
  let cut := localCut want source
  have inside := i.isLt
  have ordered := cut.ordered
  have upper := cut.within
  change i.val < cut.rows - (cut.hi - cut.lo) at inside
  rw [wireSlots, RowExchange.peer_order want source]
  by_cases before : i.val < cut.lo
  · rw [List.getElem?_append_left (by simpa only [List.length_range] using before), List.getElem?_range before]
    change some i.val = some (i.val + if i.val < cut.lo then 0 else cut.own)
    simp [before]
  · have lower : cut.lo ≤ i.val := Nat.le_of_not_gt before
    rw [List.getElem?_append_right (by simpa only [List.length_range] using lower), List.length_range]
    change (List.range' cut.hi (cut.rows - cut.hi))[i.val - cut.lo]? = _
    rw [List.getElem?_range' (by omega : i.val - cut.lo < cut.rows - cut.hi), Nat.one_mul]
    change some (cut.hi + (i.val - cut.lo)) = some (i.val + if i.val < cut.lo then 0 else cut.own)
    simp only [if_neg before, Cut.own, Option.some.injEq]
    omega

def wire (want : Want) (gradient : FP16Rows) (source : Fin 8) (col : Channel) : List (Value .bf16) :=
  let input : SparsePackets.NgramInput := ⟨localCut want source, gradient⟩
  (List.finRange input.cut.peers).map fun i => (SparsePackets.ngramDirect input).wire i col

theorem packed_wire_exact (want : Want) (gradient : FP16Rows) (source : Fin 8) (col : Channel) :
    wire want gradient source col = (wireSlots want source).map (fun slot => SparsePackets.wireCast (gradient slot col)) := by
  let cut := localCut want source
  have length : (wireSlots want source).length = cut.peers := by
    rw [wireSlots, RowExchange.peer_order want source, List.length_append, List.length_range, List.length_range']
    have := cut.ordered
    have := cut.within
    change cut.lo + (cut.rows - cut.hi) = cut.peers
    unfold Cut.peers Cut.own
    omega
  apply List.ext_getElem?
  intro i
  simp only [wire, List.getElem?_map]
  by_cases live : i < cut.peers
  · let slot : Fin cut.peers := ⟨i, live⟩
    rw [List.getElem?_eq_getElem (by simpa using live), wire_slot want source slot]
    simp [SparsePackets.ngramDirect, slot]
  · have bound := Nat.le_of_not_gt live
    rw [List.getElem?_eq_none (by simpa using bound), List.getElem?_eq_none (length ▸ bound)]
    rfl

def segment (wanted : Nat → Want) (gradients : Nat → FP16Rows) (source owner : Fin 8)
    (col : Channel) : List (Value .bf16) :=
  let offsets := ((List.range owner.val).map (sendCounts wanted source)).sum
  ((wire (wanted source) (gradients source) source col).drop offsets).take (sendCounts wanted source owner)

theorem segment_exact (wanted : Nat → Want) (gradients : Nat → FP16Rows) (source owner : Fin 8)
    (col : Channel) : segment wanted gradients source owner col =
      (packetSlots (wanted source) source owner).map (fun slot => SparsePackets.wireCast (gradients source slot col)) := by
  let words := fun slot => SparsePackets.wireCast (gradients source slot col)
  let packets := packetSlots (wanted source) source
  let preceding := (List.range owner.val).flatMap packets
  have range : List.range 8 = (List.range owner.val ++ [owner.val]) ++
      (List.range (8 - (owner.val + 1))).map (owner.val + 1 + ·) := by
    have fits := owner.isLt
    have ranks : owner.val + 1 + (8 - (owner.val + 1)) = 8 := by omega
    simpa only [ranks, List.range_succ (n := owner.val)] using
      (List.range_add (n := owner.val + 1) (m := 8 - (owner.val + 1)))
  have length (rank : Nat) : (packets rank).length = sendCounts wanted source rank := by
    simpa only [RowExchange.idPacket, List.length_map] using RowExchange.packet_length wanted source rank
  have offset : ((List.range owner.val).map (sendCounts wanted source)).sum = (preceding.map words).length := by
    simp only [List.length_map, preceding, List.length_flatMap, length]
  unfold segment
  rw [offset, packed_wire_exact, wireSlots, range, List.flatMap_append, List.flatMap_append, List.flatMap_singleton,
    List.map_append, List.map_append, List.append_assoc]
  change ((preceding.map words ++ ((packets owner).map words ++ _)).drop (preceding.map words).length).take
    (sendCounts wanted source owner) = _
  rw [List.drop_append_length, ← length owner, ← List.length_map (f := words) (as := packets owner),
    List.take_append_length]

def received (wanted : Nat → Want) (gradients : Nat → FP16Rows) (owner : Fin 8)
    (col : Channel) : List (Value .bf16) :=
  (List.finRange 8).flatMap fun source => segment wanted gradients source owner col

theorem received_in_asked_order (wanted : Nat → Want) (gradients : Nat → FP16Rows) (owner : Fin 8)
    (col : Channel) :
    let entries := (List.finRange 8).flatMap fun (source : Fin 8) =>
      (packetSlots (wanted source) source owner).map (fun slot =>
        ((wanted source).read slot, SparsePackets.wireCast (gradients source slot col)))
    entries.map Prod.fst = RowExchange.asked wanted owner ∧ entries.map Prod.snd = received wanted gradients owner col := by
  intro entries
  constructor
  · simp only [entries, List.map_flatMap, List.map_map, Function.comp_def, RowExchange.asked, RowExchange.idPacket]
    rw [← List.map_coe_finRange_eq_range, List.flatMap_map]
  · simp only [entries, List.map_flatMap, List.map_map, Function.comp_def, received]
    apply List.flatMap_congr
    intro source _
    exact (segment_exact wanted gradients source owner col).symm

/-- The Adam launch sees the locally owned FP16 slice and the received
BF16 entries indexed by the same IDs as the earlier fetch. -/
def adamPull (wanted : Nat → Want) (gradients : Nat → FP16Rows) (owner : Fin 8) : NgramAdam.Pull :=
  let cut := localCut (wanted owner) owner
  let ids : AdamEntries.Input :=
    ⟨Array.ofFn (fun i : Fin cut.rows => Int32.ofNat ((wanted owner).read i)),
      ((RowExchange.asked wanted owner).map Int32.ofNat).toArray, cut.lo, cut.hi,
      Int32.ofNat (RowExchange.boundary owner), cut.ordered, by simpa only [Array.size_ofFn] using cut.within⟩
  ⟨ids, (SparsePackets.ngramDirect ⟨cut, gradients owner⟩).own,
    fun i col => ((received wanted gradients owner col)[i.val]?).getD (Value.zero .bf16)⟩

theorem adam_peer_entries (wanted : Nat → Want) (gradients : Nat → FP16Rows) (owner : Fin 8) (col : Channel) :
    let pull := adamPull wanted gradients owner
    let entries := (List.finRange 8).flatMap fun (source : Fin 8) =>
      (packetSlots (wanted source) source owner).map (fun slot =>
        ((wanted source).read slot, SparsePackets.wireCast (gradients source slot col)))
    entries.map (Int32.ofNat ∘ Prod.fst) = pull.ids.asked.toList ∧
      ∀ i : Fin pull.ids.peerCount, (entries.map Prod.snd)[i.val]? = some (pull.peerGradient i col) := by
  intro pull entries
  obtain ⟨ids, wire⟩ := received_in_asked_order wanted gradients owner col
  refine ⟨?_, ?_⟩
  · change entries.map (Int32.ofNat ∘ Prod.fst) = ((RowExchange.asked wanted owner).map Int32.ofNat).toArray.toList
    simpa only [List.map_map, List.toList_toArray] using congrArg (List.map Int32.ofNat) ids
  · intro i
    have length : (received wanted gradients owner col).length = (RowExchange.asked wanted owner).length := by
      rw [← wire, ← ids, List.length_map, List.length_map]
    have index : i.val < (received wanted gradients owner col).length := by
      have live := i.isLt
      change i.val < ((RowExchange.asked wanted owner).map Int32.ofNat).toArray.size at live
      simpa only [length, List.size_toArray, List.length_map] using live
    change (entries.map Prod.snd)[i.val]? =
      some (((received wanted gradients owner col)[i.val]?).getD (Value.zero .bf16))
    rw [wire, List.getElem?_eq_getElem index]
    rfl

#axiomsClean packed_wire_exact
#axiomsClean received_in_asked_order
#axiomsClean adam_peer_entries
end VeriTile.Bench.Models.NanoGPT.Track1.RowGradientExchange
