import models.NanoGPT.Track1.ValueAdamDivisorBand

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.DivisorBand

open VeriTile.Triton FP

private def quantum (exponent : Nat) : Rat := FP.pow2 ((exponent : Int) - 150)

private theorem quantum_bounds (i : Fin 64) :
    1 / (2 ^ 86 : Rat) ≤ quantum (i.val + 64) ∧ quantum (i.val + 64) ≤ 1 / (2 ^ 23 : Rat) := by
  revert i
  decide +kernel

theorem root_input_range (warp : List (List F32)) (lane : List F32) (word : F32)
    (present : lane ∈ warp) (found : word ∈ lane) (clear : rootReject warp = false) :
    ∃ magnitude : Rat, word.decode = .finite false magnitude ∧
      1 / (2 ^ 63 : Rat) ≤ magnitude ∧ magnitude < 2 := by
  have laneClear := List.any_eq_false.mp clear lane present
  have noFailure : bandFailure lane = 0 := bne_eq_false_iff_eq.mp (Bool.eq_false_iff.mpr laneClear)
  rcases in_band lane word found noFailure with ⟨lower, upper⟩
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
  have upperSig : significand < (16777216 : Rat) := by
    have small : 8388608 + word.fraction < 16777216 := by omega
    change ((8388608 + word.fraction : Nat) : Rat) < (16777216 : Rat)
    exact_mod_cast small
  have powerPos : 0 < quantum word.exponent := lt_of_lt_of_le (by norm_num) h.1
  refine ⟨significand * quantum word.exponent, ?_, ?_, ?_⟩
  · simp [Value.decode, unsign, exp0, exp255, quantum, significand, Format.fractionBits]
    exact Or.inl (congrArg FP.pow2 (by omega))
  · calc
      1 / (2 ^ 63 : Rat) ≤ (8388608 : Rat) * (1 / (2 ^ 86 : Rat)) := by norm_num
      _ ≤ significand * quantum word.exponent :=
        mul_le_mul lowerSig h.1 (by norm_num) (le_trans (by norm_num) lowerSig)
  · calc
      significand * quantum word.exponent < (16777216 : Rat) * quantum word.exponent :=
        mul_lt_mul_of_pos_right upperSig powerPos
      _ ≤ (16777216 : Rat) * (1 / (2 ^ 23 : Rat)) := mul_le_mul_of_nonneg_left h.2 (by norm_num)
      _ = 2 := by norm_num

#axiomsClean root_input_range

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.DivisorBand
