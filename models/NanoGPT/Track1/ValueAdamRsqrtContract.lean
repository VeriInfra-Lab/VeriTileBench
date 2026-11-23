import models.NanoGPT.Track1.ValueAdamFastDenominator
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand

open VeriTile.Triton FP

/-- CUDA 12.8 PTX ISA 9.7.3.17 gives the reciprocal square root a relative
error of at most 2^(-22.9). Convert the instruction's real-number error
bound into the rational inequalities used by the divide certificate. -/
theorem normalized_of_ptx_error (value reciprocal : Rat) (positive : 0 < value)
    (relative : |(reciprocal : ℝ) - (Real.sqrt (value : ℝ))⁻¹| ≤
      (2 : ℝ) ^ (-(229 / 10 : ℝ)) * (Real.sqrt (value : ℝ))⁻¹) :
    0 < reciprocal ∧ 7 / 8 ≤ value * reciprocal ^ 2 ∧ value * reciprocal ^ 2 ≤ 9 / 8 := by
  let root := Real.sqrt (value : ℝ)
  let a : ℝ := reciprocal
  have inputPositive : 0 < (value : ℝ) := by exact_mod_cast positive
  have rootPositive : 0 < root := Real.sqrt_pos.2 inputPositive
  have rootNonzero : root ≠ 0 := ne_of_gt rootPositive
  have weakError : (2 : ℝ) ^ (-(229 / 10 : ℝ)) ≤ 1 / 32 := by
    calc
      _ ≤ (2 : ℝ) ^ (-(5 : ℝ)) :=
        Real.rpow_le_rpow_of_exponent_le (by norm_num) (by norm_num)
      _ = _ := by
        rw [show (-(5 : ℝ)) = ((-5 : Int) : ℝ) by norm_num, Real.rpow_intCast]
        norm_num
  have scaled : |a * root - 1| ≤ 1 / 32 := by
    have scaledWord : (a - root⁻¹) * root = a * root - 1 := by
      rw [sub_mul, inv_mul_cancel₀ rootNonzero]
    rw [← scaledWord, abs_mul, abs_of_pos rootPositive]
    calc
      _ ≤ ((2 : ℝ) ^ (-(229 / 10 : ℝ)) * root⁻¹) * root :=
        mul_le_mul_of_nonneg_right relative rootPositive.le
      _ = (2 : ℝ) ^ (-(229 / 10 : ℝ)) := by rw [mul_assoc, inv_mul_cancel₀ rootNonzero, mul_one]
      _ ≤ _ := weakError
  rcases abs_le.mp scaled with ⟨scaledLow, scaledHigh⟩
  have low : (31 / 32 : ℝ) ≤ a * root := by linarith
  have high : a * root ≤ (33 / 32 : ℝ) := by linarith
  have aPositive : 0 < a := by
    by_contra! nonpositive
    have impossible := mul_nonpos_of_nonpos_of_nonneg nonpositive rootPositive.le
    linarith
  have mulPositive : 0 < a * root := mul_pos aPositive rootPositive
  have squareLow : (31 / 32 : ℝ) ^ 2 ≤ (a * root) ^ 2 := by
    simpa only [pow_two] using mul_le_mul low low (by norm_num) mulPositive.le
  have squareHigh : (a * root) ^ 2 ≤ (33 / 32 : ℝ) ^ 2 := by
    simpa only [pow_two] using mul_le_mul high high mulPositive.le (by norm_num)
  have convert : (a * root) ^ 2 = (value : ℝ) * (reciprocal : ℝ) ^ 2 := by
    have sq : root ^ 2 = (value : ℝ) := Real.sq_sqrt inputPositive.le
    rw [mul_pow, sq, mul_comm]
  rw [convert] at squareLow squareHigh
  have lower : (7 / 8 : ℝ) ≤ (value : ℝ) * (reciprocal : ℝ) ^ 2 := by linarith
  have upper : (value : ℝ) * (reciprocal : ℝ) ^ 2 ≤ (9 / 8 : ℝ) := by linarith
  have aCast : 0 < (reciprocal : ℝ) := aPositive
  have lowCast : ((7 / 8 : Rat) : ℝ) ≤ ((value * reciprocal ^ 2 : Rat) : ℝ) := by
    push_cast
    exact lower
  have highCast : ((value * reciprocal ^ 2 : Rat) : ℝ) ≤ ((9 / 8 : Rat) : ℝ) := by
    push_cast
    exact upper
  exact ⟨Rat.cast_pos.mp aCast, Rat.cast_le.mp lowCast, Rat.cast_le.mp highCast⟩

/-- Build the common primitive from the positive, normal input guarantee
for the emitted PTX instruction. The kernel proof derives every arithmetic
effect after that instruction. -/
def ApproximateRsqrt.fromPTX (run : F32 → F32)
    (error : ∀ (word : F32) (value : Rat), word.decode = .finite false value →
      1 / 2 ^ 63 ≤ value → value < 2 →
      ∃ result : Rat, (run word).decode = .finite false result ∧
        |(result : ℝ) - (Real.sqrt (value : ℝ))⁻¹| ≤
          (2 : ℝ) ^ (-(229 / 10 : ℝ)) * (Real.sqrt (value : ℝ))⁻¹) : ApproximateRsqrt :=
  ⟨run, fun word value decoded min max => by
    obtain ⟨result, output, relative⟩ := error word value decoded min max
    have positive : 0 < value := by linarith
    exact ⟨result, output, normalized_of_ptx_error value result positive relative⟩⟩

#axiomsClean normalized_of_ptx_error
#axiomsClean ApproximateRsqrt.fromPTX

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand
