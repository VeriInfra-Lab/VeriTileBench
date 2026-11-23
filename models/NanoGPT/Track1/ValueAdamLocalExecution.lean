import models.NanoGPT.Track1.ValueAdamLocalPTX
import Mathlib.Tactic.Ring

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows
open SparsePackets (Channel)

def targetRow : Address := {first := -1, row := 1}

def slotRow (thread : Nat) : Address := {block := 4, byte := thread / 32}

def wordAddress (row : Address) (thread stripe word : Nat) : Address :=
  let base := Address.literal (thread % 32 * 16 + stripe * 512 : Nat)
  (row.times 1536 |>.plus base).plus (Address.literal (2 * word : Nat))

def atomAt (direct : Bool) (thread stripe word : Nat) : Effect :=
  let row := if direct then targetRow else slotRow thread
  .atom (wordAddress targetRow thread stripe word) (wordAddress row thread stripe word)

theorem expected_has_atom (direct marked : Bool) (thread stripe word : Nat)
    (liveStripe : stripe < 3) (liveWord : word < 8) :
    atomAt direct thread stripe word ∈ expected direct marked thread true := by
  simp only [expected, Bool.not_true, Bool.false_eq_true, if_false]
  apply List.mem_cons_of_mem
  apply List.mem_append_left
  apply List.mem_flatMap.mpr
  refine ⟨stripe, List.mem_range.mpr liveStripe, ?_⟩
  apply List.mem_cons_of_mem
  apply List.mem_append_left
  exact List.mem_map.mpr ⟨word, List.mem_range.mpr liveWord, rfl⟩

def artifactHasAtom (source : String) (direct : Bool) (slot : Fin count) (col : Channel) : Bool :=
  let lane := Lane.ofCell slot col
  let needle := atomAt direct lane.thread.val lane.stripe.val lane.word.val
  (artifactEffects source lane.thread.val true).toOption.any (·.any (fun item => decide (item = needle)))

theorem Certificate.has_atom {direct marked : Bool} {source : String}
    (checked : Certificate direct marked source) (slot : Fin count) (col : Channel) :
    artifactHasAtom source direct slot col = true := by
  let lane := Lane.ofCell slot col
  let needle := atomAt direct lane.thread.val lane.stripe.val lane.word.val
  change (artifactEffects source lane.thread.val true).toOption.any
    (·.any (fun item => decide (item = needle))) = true
  rw [checked.effects_exact lane.thread true]
  change (expected direct marked lane.thread.val true).any (fun item => decide (item = needle)) = true
  exact List.any_eq_true.mpr ⟨needle,
    expected_has_atom direct marked _ _ _ lane.stripe.isLt lane.word.isLt, decide_eq_true rfl⟩

def observedLocal (source : String) (direct : Bool) (p : Pull) (idx : AdamEntries.Indices p.ids)
    (order : Schedule p) (before : HalfShard) : HalfShard :=
  fun row col =>
    ((order.localAtomics col).slots.find? (fun i => decide ((idx.localRows i).toInt = row))).elim
      (before row col) (fun i => if artifactHasAtom source direct i col then
        halfAdd (before row col) (p.ownGradient i col) else before row col)

def observedMerge (source : String) (direct : Bool) (p : Pull) (idx : AdamEntries.Indices p.ids)
    (order : Schedule p) (before : HalfShard) : HalfShard :=
  fun row col =>
    (order.peerAtomics col).slots.foldl (fun acc i =>
      if (idx.peerRows i).toInt = row then halfAdd acc (p.receivedGradient i col) else acc)
      (observedLocal source direct p idx order before row col)

theorem Certificate.merged_exact {direct marked : Bool} {source : String}
    (checked : Certificate direct marked source) (s : SourceSlice)
    (peer : Fin s.asked.size → Channel → Half) (order : Schedule (s.deferredPull peer))
    (before : HalfShard) :
    observedMerge source direct (s.deferredPull peer) (AdamEntries.fused s.ids) order before =
      mergeFrom (s.deferredPull peer) (AdamEntries.fused s.ids) order before := by
  rw [← s.owner_store_exact peer order before]
  funext row col
  simp only [observedMerge, observedLocal, localStoreThenPeers, orderedLocalStore,
    checked.has_atom, if_true]

theorem word_address_exact (row : Address) (input : Inputs) (lane : Lane count) :
    (wordAddress row lane.thread.val lane.stripe.val lane.word.val).eval input =
      row.eval input * 1536 + 2 * (lane.channel.val : Int) := by
  simp [wordAddress, Address.plus, Address.times, Address.literal, Address.eval, Lane.channel]
  ring

#axiomsClean expected_has_atom
#axiomsClean Certificate.has_atom
#axiomsClean Certificate.merged_exact
#axiomsClean word_address_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows
