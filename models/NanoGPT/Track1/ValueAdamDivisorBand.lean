import models.NanoGPT.Track1.ValueAdamExponentGuard
import Mathlib.Data.Rat.Lemmas
import Mathlib.Tactic.NormNum

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.DivisorBand

open VeriTile.Triton FP

abbrev Pair := F32 × F32

private def foldAnd (values : List F32) : BitVec 32 :=
  values.foldr (fun value rest => value.bits &&& rest) (BitVec.allOnes 32)

def bandFailure (denominators : List F32) : BitVec 32 :=
  (((foldAnd denominators &&& 0x20000000) ^^^ 0x20000000) |||
    (ExponentGuard.summary denominators &&& 0xc0000000))

def laneReject (lane : List Pair) : Bool :=
  bandFailure (lane.map Prod.snd) != 0 ||
    ExponentGuard.laneMayHaveNaN (lane.map Prod.fst)

def warpReject (warp : List (List Pair)) : Bool :=
  warp.any laneReject

private theorem fold_and_bit (words : List F32) (pos : Nat) (fits : pos < 32) :
    (foldAnd words).getLsbD pos = words.all (fun word => word.bits.getLsbD pos) := by
  induction words with
  | nil =>
    change (BitVec.allOnes 32).getLsbD pos = true
    rw [BitVec.getLsbD_allOnes, decide_eq_true fits]
  | cons word rest prev =>
    change (ExponentGuard.wordBits word &&& foldAnd rest).getLsbD pos = _
    rw [BitVec.getLsbD_and, prev]
    rfl

private theorem fold_or_bit (words : List F32) (pos : Nat) :
    (ExponentGuard.summary words).getLsbD pos = words.any (fun word => word.bits.getLsbD pos) := by
  induction words with
  | nil => simp [ExponentGuard.summary]
  | cons word rest prev =>
    change (ExponentGuard.wordBits word ||| ExponentGuard.summary rest).getLsbD pos = _
    rw [BitVec.getLsbD_or, prev]
    rfl

private theorem word_bits (words : List F32) (word : F32) (found : word ∈ words)
    (clear : bandFailure words = 0) :
    word.bits.getLsbD 29 = true ∧ word.bits.getLsbD 30 = false ∧ word.bits.getLsbD 31 = false := by
  have read (i : Nat) := congrArg (fun bits : BitVec 32 => bits.getLsbD i) clear
  have set29 : (0x20000000 : BitVec 32).getLsbD 29 = true := by decide +kernel
  have unset30 : (0x20000000 : BitVec 32).getLsbD 30 = false := by decide +kernel
  have unset31 : (0x20000000 : BitVec 32).getLsbD 31 = false := by decide +kernel
  have high29 : (0xc0000000 : BitVec 32).getLsbD 29 = false := by decide +kernel
  have high30 : (0xc0000000 : BitVec 32).getLsbD 30 = true := by decide +kernel
  have high31 : (0xc0000000 : BitVec 32).getLsbD 31 = true := by decide +kernel
  have no29 : (0 : BitVec 32).getLsbD 29 = false := by decide +kernel
  have no30 : (0 : BitVec 32).getLsbD 30 = false := by decide +kernel
  have no31 : (0 : BitVec 32).getLsbD 31 = false := by decide +kernel
  have h29 := read 29
  have h30 := read 30
  have h31 := read 31
  simp only [bandFailure, BitVec.getLsbD_or, BitVec.getLsbD_xor, BitVec.getLsbD_and,
    no29, set29, high29, Bool.and_true, Bool.and_false, Bool.or_false,
    Bool.xor_true, fold_and_bit _ _ (by decide : 29 < 32)] at h29
  simp only [bandFailure, BitVec.getLsbD_or, BitVec.getLsbD_xor, BitVec.getLsbD_and,
    no30, unset30, high30, Bool.and_true, Bool.and_false, Bool.false_or,
    Bool.xor_false, fold_or_bit] at h30
  simp only [bandFailure, BitVec.getLsbD_or, BitVec.getLsbD_xor, BitVec.getLsbD_and,
    no31, unset31, high31, Bool.and_true, Bool.and_false, Bool.false_or,
    Bool.xor_false, fold_or_bit] at h31
  have all29 : words.all (fun v => v.bits.getLsbD 29) = true := by
    cases result : words.all (fun v => v.bits.getLsbD 29)
    · rw [result] at h29
      contradiction
    · rfl
  exact ⟨List.all_eq_true.mp all29 word found,
    Bool.eq_false_iff.mpr (List.any_eq_false.mp h30 word found),
    Bool.eq_false_iff.mpr (List.any_eq_false.mp h31 word found)⟩

private theorem top_three (word : F32)
    (bit29 : word.bits.getLsbD 29 = true) (bit30 : word.bits.getLsbD 30 = false)
    (bit31 : word.bits.getLsbD 31 = false) :
    (word.bits.extractLsb' 29 3) = (1 : BitVec 3) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i fits
  rw [BitVec.getLsbD_extractLsb', decide_eq_true fits, Bool.true_and]
  have cases : i = 0 ∨ i = 1 ∨ i = 2 := by omega
  rcases cases with rfl | rfl | rfl
  · exact bit29
  · exact bit30
  · exact bit31

private theorem word_range (word : F32) (top : word.bits.extractLsb' 29 3 = (1 : BitVec 3)) :
    0x20000000 ≤ word.raw ∧ word.raw < 0x40000000 := by
  have read := congrArg BitVec.toNat top
  rw [BitVec.extractLsb'_toNat, Nat.shiftRight_eq_div_pow] at read
  change (word.raw / 536870912) % 8 = 1 at read
  have fits : word.raw < 4294967296 := word.bits.isLt
  omega

theorem in_band (words : List F32) (word : F32) (found : word ∈ words)
    (clear : bandFailure words = 0) : 0x20000000 ≤ word.raw ∧ word.raw < 0x40000000 := by
  rcases word_bits words word found clear with ⟨a,b,c⟩
  exact word_range word (top_three word a b c)

def rootReject (warp : List (List F32)) : Bool :=
  warp.any fun lane => bandFailure lane != 0

def originalRootHelper (second : F32) : Bool :=
  ((ExponentGuard.wordBits second - 0x0d000000).toNat > 0x727fffff)

theorem root_helper_unneeded (warp : List (List F32)) (lane : List F32) (second : F32)
    (inWarp : lane ∈ warp) (inLane : second ∈ lane) (clear : rootReject warp = false) :
    originalRootHelper second = false := by
  have clearLane := List.any_eq_false.mp clear lane inWarp
  have noField : bandFailure lane = 0 := bne_eq_false_iff_eq.mp (Bool.eq_false_iff.mpr clearLane)
  rcases in_band lane second inLane noField with ⟨lower, upper⟩
  apply decide_eq_false
  change ¬(0x727fffff < (ExponentGuard.wordBits second - 0x0d000000).toNat)
  have noWrap : (0x0d000000 : BitVec 32) ≤ ExponentGuard.wordBits second := by
    change 0x0d000000 ≤ second.raw
    omega
  rw [BitVec.toNat_sub_of_le noWrap]
  change ¬(0x727fffff < second.raw - 0x0d000000)
  omega

#axiomsClean root_helper_unneeded

theorem selected_inputs (warp : List (List Pair)) (lane : List Pair) (pair : Pair)
    (inWarp : lane ∈ warp) (inLane : pair ∈ lane) (clear : warpReject warp = false) :
    pair.1.isNaN = false ∧ 0x20000000 ≤ pair.2.raw ∧ pair.2.raw < 0x40000000 := by
  have localClear : laneReject lane = false := Bool.eq_false_iff.mpr (List.any_eq_false.mp clear lane inWarp)
  have denClear : bandFailure (lane.map Prod.snd) = 0 :=
    bne_eq_false_iff_eq.mp (Bool.or_eq_false_iff.mp localClear).1
  have firstClear := (Bool.or_eq_false_iff.mp localClear).2
  have firstFound : pair.1 ∈ lane.map Prod.fst := List.mem_map.mpr ⟨pair, inLane, rfl⟩
  have secondFound : pair.2 ∈ lane.map Prod.snd := List.mem_map.mpr ⟨pair, inLane, rfl⟩
  have nonNaN := ExponentGuard.no_nan_on_lane_vote [lane.map Prod.fst] (lane.map Prod.fst) pair.1
    (by simp) firstFound (by simpa [ExponentGuard.warpMayHaveNaN] using firstClear)
  rcases word_bits _ _ secondFound denClear with ⟨a,b,c⟩
  exact ⟨nonNaN, word_range _ (top_three _ a b c)⟩


private def quantum (exponent : Nat) : Rat := FP.pow2 ((exponent : Int) - 150)

private theorem quantum_bounds (i : Fin 64) :
    1 / (2 ^ 86 : Rat) ≤ quantum (i.val + 64) ∧ quantum (i.val + 64) ≤ 1 / (2 ^ 23 : Rat) := by
  revert i
  decide +kernel

/-- Positive words in the decoded band stay strictly inside the range that
disables both of the original Hopper divider's scale predicates. -/
theorem denominator_magnitude (word : F32)
    (lower : 0x20000000 ≤ word.raw) (upper : word.raw < 0x40000000) :
    ∃ magnitude : Rat, word.decode = .finite false magnitude ∧
      1 / (2 ^ 126 : Rat) ≤ magnitude ∧ magnitude < 2 ^ 126 := by
  have unsign : word.sign = false := by
    have quotient : word.raw / 2147483648 = 0 := by omega
    simp [Value.sign, Format.fractionBits, quotient]
  have exp : word.exponent = word.raw / 8388608 := by
    have below : word.raw / 8388608 < 256 := by omega
    simpa [Value.exponent, Format.fractionBits] using Nat.mod_eq_of_lt below
  have minExp : 64 ≤ word.exponent := by omega
  have maxExp : word.exponent < 128 := by omega
  have exp0 : word.exponent ≠ 0 := by omega
  have exp255 : word.exponent ≠ 255 := by omega
  have bounded : word.fraction < 8388608 := by
    exact Nat.mod_lt _ (by decide : 0 < 8388608)
  have indexed : word.exponent - 64 + 64 = word.exponent := by omega
  have h := quantum_bounds ⟨word.exponent - 64, by omega⟩
  rw [indexed] at h
  let significand : Rat := (8388608 + word.fraction : Nat)
  have lowerSig : (8388608 : Rat) ≤ significand := by
    change (8388608 : Rat) ≤ ((8388608 + word.fraction : Nat) : Rat)
    exact_mod_cast Nat.le_add_right 8388608 word.fraction
  have upperSig : significand ≤ (16777216 : Rat) := by
    have small : 8388608 + word.fraction ≤ 16777216 := by omega
    change ((8388608 + word.fraction : Nat) : Rat) ≤ (16777216 : Rat)
    exact_mod_cast small
  have powerPos : 0 ≤ quantum word.exponent := le_trans (by norm_num) h.1
  refine ⟨significand * quantum word.exponent, ?_, ?_, ?_⟩
  · simp [Value.decode, unsign, exp0, exp255, quantum, significand, Format.fractionBits]
    exact Or.inl (congrArg FP.pow2 (by omega))
  · calc
      1 / (2 ^ 126 : Rat) ≤ (8388608 : Rat) * (1 / (2 ^ 86 : Rat)) := by norm_num
      _ ≤ significand * quantum word.exponent :=
        mul_le_mul lowerSig h.1 (by norm_num) (le_trans (by norm_num) lowerSig)
  · calc
      significand * quantum word.exponent ≤ (16777216 : Rat) * (1 / (2 ^ 23 : Rat)) :=
        mul_le_mul upperSig h.2 powerPos (by norm_num)
      _ < (2 ^ 126 : Rat) := by norm_num

#axiomsClean selected_inputs
#axiomsClean denominator_magnitude

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.DivisorBand
