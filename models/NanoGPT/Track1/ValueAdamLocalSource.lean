import models.NanoGPT.Track1.ValueAdamLocalExecution

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows
open SparsePackets (Channel)

structure LaunchGuard (s : SourceSlice) : Prop where
  rows : s.shardRows = 25152
  sourceEnd : s.first + 25152 ≤ 1048576
  listSize : s.input.cut.rows < 1048576
  ownCount : s.input.cut.own ≤ 25152

def launchInputs (s : SourceSlice) (guard : LaunchGuard s) (lane : Lane s.input.cut.own) : Inputs where
  block := ⟨lane.cta, by
    have block := cta_within_grid lane
    have count := guard.ownCount
    omega⟩
  first := ⟨s.first, by have finish := guard.sourceEnd; omega⟩
  offset := ⟨s.input.cut.lo, by
    have ordered := s.input.cut.ordered
    have within := s.input.cut.within
    have extent := guard.listSize
    omega⟩
  count := ⟨s.input.cut.own, by have count := guard.ownCount; omega⟩
  row := ⟨s.globalRow lane.slot, by
    have selected := (s.selected lane.slot).2
    have shape := guard.rows
    have finish := guard.sourceEnd
    change s.globalRow lane.slot < s.first + s.shardRows at selected
    omega⟩

theorem target_address_exact (s : SourceSlice) (guard : LaunchGuard s) (lane : Lane s.input.cut.own) :
    (wordAddress targetRow lane.thread.val lane.stripe.val lane.word.val).eval (launchInputs s guard lane) =
      (2 * (s.localRow lane.slot * 768 + lane.channel.val) : Nat) := by
  rw [word_address_exact]
  simp [targetRow, launchInputs, Address.eval]
  rw [s.local_row_exact lane.slot]
  have selected := (s.selected lane.slot).1
  change s.first ≤ s.globalRow lane.slot at selected
  omega

theorem staged_address_exact (s : SourceSlice) (guard : LaunchGuard s) (lane : Lane s.input.cut.own) :
    (wordAddress (slotRow lane.thread.val) lane.thread.val lane.stripe.val lane.word.val).eval
      (launchInputs s guard lane) = (2 * (lane.slot.val * 768 + lane.channel.val) : Nat) := by
  rw [word_address_exact]
  simp [slotRow, Lane.slot, launchInputs, Address.eval]
  omega

theorem no_other_lane_writes (s : SourceSlice) (guard : LaunchGuard s)
    (a b : Lane s.input.cut.own)
    (same : (wordAddress targetRow a.thread.val a.stripe.val a.word.val).eval (launchInputs s guard a) =
      (wordAddress targetRow b.thread.val b.stripe.val b.word.val).eval (launchInputs s guard b)) :
    a = b := by
  rw [target_address_exact, target_address_exact] at same
  have indexes : s.localRow a.slot * 768 + a.channel.val =
      s.localRow b.slot * 768 + b.channel.val := by omega
  obtain ⟨sameSlot, sameWord⟩ := s.address_unique indexes
  rw [← lane_exact a, ← lane_exact b, sameSlot, sameWord]

#axiomsClean target_address_exact
#axiomsClean staged_address_exact
#axiomsClean no_other_lane_writes

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows
