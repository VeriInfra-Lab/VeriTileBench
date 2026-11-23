import models.NanoGPT.Track1.ValueAdamUniqueOwner

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows
open SparsePackets (Channel)

structure Lane (count : Nat) where
  cta : Nat
  thread : Fin 128
  stripe : Fin 3
  word : Fin 8
  active : cta * 4 + thread.val / 32 < count

def Lane.slot (lane : Lane count) : Fin count :=
  ⟨lane.cta * 4 + lane.thread.val / 32, lane.active⟩

def Lane.channel (lane : Lane count) : Channel :=
  ⟨lane.stripe.val * 256 + lane.thread.val % 32 * 8 + lane.word.val, by
    have stripe := lane.stripe.isLt
    have word := lane.word.isLt
    omega⟩

def Lane.ofCell (slot : Fin count) (col : Channel) : Lane count where
  cta := slot.val / 4
  thread := ⟨slot.val % 4 * 32 + col.val % 256 / 8, by
    have column := col.isLt
    omega⟩
  stripe := ⟨col.val / 256, by have column := col.isLt; omega⟩
  word := ⟨col.val % 8, Nat.mod_lt _ (by decide)⟩
  active := by
    change slot.val / 4 * 4 + (slot.val % 4 * 32 + col.val % 256 / 8) / 32 < count
    have live := slot.isLt
    omega

theorem slot_exact (slot : Fin count) (col : Channel) : (Lane.ofCell slot col).slot = slot := by
  apply Fin.ext
  change slot.val / 4 * 4 + (slot.val % 4 * 32 + col.val % 256 / 8) / 32 = slot.val
  omega

theorem channel_exact (slot : Fin count) (col : Channel) : (Lane.ofCell slot col).channel = col := by
  apply Fin.ext
  change col.val / 256 * 256 + (slot.val % 4 * 32 + col.val % 256 / 8) % 32 * 8 +
      col.val % 8 = col.val
  omega

theorem lane_exact (lane : Lane count) : Lane.ofCell lane.slot lane.channel = lane := by
  cases lane with
  | mk cta thread stripe word live =>
    have threadBound := thread.isLt
    have stripeBound := stripe.isLt
    have wordBound := word.isLt
    simp only [Lane.ofCell, Lane.slot, Lane.channel, Lane.mk.injEq]
    refine ⟨by omega, Fin.ext ?_, Fin.ext ?_, Fin.ext ?_⟩ <;> dsimp <;> omega

theorem every_cell_has_one_lane (slot : Fin count) (col : Channel) :
    ∃! lane : Lane count, lane.slot = slot ∧ lane.channel = col := by
  refine ⟨Lane.ofCell slot col, ⟨slot_exact slot col, channel_exact slot col⟩, ?_⟩
  intro lane located
  rw [← located.1, ← located.2, lane_exact]

theorem cta_within_grid (lane : Lane count) : lane.cta < (count + 3) / 4 := by
  have live := lane.active
  omega

theorem packed_lane_slot (cta : Nat) (thread : Fin 128) :
    (cta <<< 2) ||| (thread.val >>> 5) = cta * 4 + thread.val / 32 := by
  have threadBound := thread.isLt
  have small : thread.val / 32 < 2 ^ 2 := by omega
  simp only [Nat.shiftRight_eq_div_pow, show (2 : Nat) ^ 5 = 32 by rfl]
  rw [← Nat.shiftLeft_add_eq_or_of_lt small, Nat.shiftLeft_eq]

theorem packed_lane_channel (thread : Fin 128) :
    (thread.val <<< 3) &&& 248 = thread.val % 32 * 8 := by
  rw [show (248 : Nat) = (31 : Nat) <<< 3 by rfl, ← Nat.shiftLeft_and_distrib,
    show (31 : Nat) = 2 ^ 5 - 1 by rfl, Nat.and_two_pow_sub_one_eq_mod, Nat.shiftLeft_eq]

theorem tail_branch_uniform (cta count : Nat) (a b : Fin 128)
    (sameWarp : a.val / 32 = b.val / 32) :
    decide (count ≤ ((cta <<< 2) ||| (a.val >>> 5))) =
      decide (count ≤ ((cta <<< 2) ||| (b.val >>> 5))) := by
  rw [packed_lane_slot, packed_lane_slot, sameWarp]

#axiomsClean tail_branch_uniform
#axiomsClean every_cell_has_one_lane
#axiomsClean cta_within_grid
#axiomsClean packed_lane_slot
#axiomsClean packed_lane_channel

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows
