import models.NanoGPT.Track1.ValueAdamElementwise
import Mathlib.Data.Fintype.Basic

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.ExponentGuard

open VeriTile.Triton FP

private def ternaryBit (table : Nat) (a b c : Bool) : Bool :=
  table.testBit (4 * a.toNat + 2 * b.toNat + c.toNat)

theorem lop3_union (a b c : Bool) :
    ternaryBit 0xfe a b c = (a || b || c) := by
  cases a <;> cases b <;> cases c <;> decide +kernel

theorem lop3_union_mask (a mask c : Bool) :
    ternaryBit 0xc8 a mask c = ((a || c) && mask) := by
  cases a <;> cases mask <;> cases c <;> decide +kernel

def wordBits (word : F32) : BitVec 32 := word.bits

def exponentMask : BitVec 32 := 0x7f800000

def summary (words : List F32) : BitVec 32 :=
  words.foldr (fun word rest => wordBits word ||| rest) 0

def laneMayHaveNaN (words : List F32) : Bool :=
  (summary words &&& exponentMask) == exponentMask

def warpMayHaveNaN (lanes : List (List F32)) : Bool :=
  lanes.any laneMayHaveNaN

def collectiveMayHaveNaN (lanes : List (List F32)) : Bool :=
  laneMayHaveNaN lanes.flatten

private theorem mask_bits (index : Fin 32) :
    exponentMask.getLsbD index.val = (decide (23 ≤ index.val) && decide (index.val < 31)) := by
  revert index
  decide +kernel

private theorem complete_exponent (index : Fin 8) :
    (255 : BitVec 8).getLsbD index.val = true := by
  revert index
  decide +kernel

private theorem nan_exponent (word : F32) (nan : word.isNaN = true) :
    (wordBits word).extractLsb' 23 8 = (255 : BitVec 8) := by
  simp only [Value.isNaN, Bool.and_eq_true] at nan
  have high : word.exponent = 255 := beq_iff_eq.mp nan.1
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.extractLsb'_toNat, Nat.shiftRight_eq_div_pow]
  exact high

private theorem summary_bit_true {words : List F32} {word : F32} (found : word ∈ words)
    {index : Nat} (set : (wordBits word).getLsbD index = true) :
    (summary words).getLsbD index = true := by
  induction words with
  | nil => simp at found
  | cons head tail ih =>
    rcases List.mem_cons.mp found with same | later
    · subst head
      simp [summary, BitVec.getLsbD_or, set]
    · have yes := ih later
      change (wordBits head ||| summary tail).getLsbD index = true
      simp [BitVec.getLsbD_or, yes]

theorem lane_detects_nan (words : List F32) (word : F32) (found : word ∈ words)
    (nan : word.isNaN = true) : laneMayHaveNaN words = true := by
  apply beq_iff_eq.mpr
  apply BitVec.eq_of_getLsbD_eq
  intro index fits
  rw [BitVec.getLsbD_and, mask_bits ⟨index, fits⟩]
  by_cases bit : 23 ≤ index ∧ index < 31
  · have offset : index - 23 < 8 := by omega
    have known := congrArg (fun bits : BitVec 8 => bits.getLsbD (index - 23)) (nan_exponent word nan)
    change ((wordBits word).extractLsb' 23 8).getLsbD (index - 23) =
      (255 : BitVec 8).getLsbD (index - 23) at known
    rw [BitVec.getLsbD_extractLsb', complete_exponent ⟨index - 23, offset⟩] at known
    have plus : 23 + (index - 23) = index := by omega
    simp only [decide_eq_true offset, plus, Bool.true_and] at known
    have laneSet := summary_bit_true found known
    simp [laneSet]
  · by_cases low : 23 ≤ index
    · have high : ¬index < 31 := fun high => bit ⟨low, high⟩
      simp [high]
    · simp [low]

theorem no_nan_on_lane_vote (lanes : List (List F32)) (lane : List F32) (word : F32)
    (inWarp : lane ∈ lanes) (inLane : word ∈ lane)
    (clear : warpMayHaveNaN lanes = false) : word.isNaN = false := by
  cases found : word.isNaN
  · rfl
  · have hit : warpMayHaveNaN lanes = true :=
      List.any_eq_true.mpr ⟨lane, inWarp, lane_detects_nan lane word inLane found⟩
    rw [clear] at hit
    contradiction

theorem no_nan_on_collective_or (lanes : List (List F32)) (lane : List F32) (word : F32)
    (inWarp : lane ∈ lanes) (inLane : word ∈ lane)
    (clear : collectiveMayHaveNaN lanes = false) : word.isNaN = false := by
  cases found : word.isNaN
  · rfl
  · have hit : collectiveMayHaveNaN lanes = true :=
      lane_detects_nan lanes.flatten word (List.mem_flatten.mpr ⟨lane, inWarp, inLane⟩) found
    rw [clear] at hit
    contradiction

#axiomsClean lop3_union
#axiomsClean lop3_union_mask
#axiomsClean lane_detects_nan
#axiomsClean no_nan_on_lane_vote
#axiomsClean no_nan_on_collective_or

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.ExponentGuard
