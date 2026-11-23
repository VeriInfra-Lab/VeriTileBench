import models.NanoGPT.Track1.ValueAdamRootInputRange
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand

/-- The hardware reciprocal-square-root error bound is applied to the
normalized square of its result. This form keeps the remaining operations
on the exact rational values represented by FP32 words. -/
theorem reciprocal_range (x s : Rat) (xLower : 1 / 2 ^ 63 ≤ x) (xUpper : x < 2)
    (sPositive : 0 < s) (errorLow : 7 / 8 ≤ x * s ^ 2) (errorHigh : x * s ^ 2 ≤ 9 / 8) :
    1 / 2 ≤ s ∧ s < 2 ^ 32 := by
  constructor
  · by_contra! small
    have square : s ^ 2 ≤ (1 / 2 : Rat) ^ 2 := by
      simpa only [pow_two] using mul_le_mul small.le small.le sPositive.le (by norm_num)
    have impossible : x * s ^ 2 ≤ 2 * (1 / 2 : Rat) ^ 2 :=
      mul_le_mul xUpper.le square (by positivity) (by norm_num)
    norm_num at impossible
    linarith
  · by_contra! big
    have square : ((2 : Rat) ^ 32) ^ 2 ≤ s ^ 2 := by
      simpa only [pow_two] using mul_le_mul big big (by norm_num) sPositive.le
    have impossible : (1 / 2 ^ 63 : Rat) * ((2 : Rat) ^ 32) ^ 2 ≤ x * s ^ 2 :=
      mul_le_mul xLower square (by norm_num) (by positivity)
    norm_num at impossible
    linarith

theorem exact_residual_range (x s rough : Rat) (xPositive : 0 < x) (sPositive : 0 < s)
    (errorLow : 7 / 8 ≤ x * s ^ 2) (errorHigh : x * s ^ 2 ≤ 9 / 8)
    (roughLower : (31 / 32 : Rat) * (x * s) ≤ rough)
    (roughUpper : rough ≤ (33 / 32 : Rat) * (x * s)) :
    -(x / 4) ≤ x - rough ^ 2 ∧ x - rough ^ 2 ≤ x / 4 := by
  have aPos : 0 < x * s := mul_pos xPositive sPositive
  have roughPos : 0 ≤ rough := by
    exact (mul_nonneg (by norm_num : (0 : Rat) ≤ 31 / 32) aPos.le).trans roughLower
  have lowSquare : ((31 / 32 : Rat) * (x * s)) ^ 2 ≤ rough ^ 2 := by
    simpa only [pow_two] using mul_le_mul roughLower roughLower (by positivity) roughPos
  have highSquare : rough ^ 2 ≤ ((33 / 32 : Rat) * (x * s)) ^ 2 := by
    simpa only [pow_two] using mul_le_mul roughUpper roughUpper roughPos (by positivity)
  have low : (3 / 4 : Rat) * x ≤ rough ^ 2 := by
    calc
      _ ≤ ((31 / 32 : Rat) ^ 2 * (7 / 8)) * x :=
        mul_le_mul_of_nonneg_right (by norm_num) xPositive.le
      _ ≤ ((31 / 32 : Rat) ^ 2 * (x * s ^ 2)) * x :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left errorLow (by positivity)) xPositive.le
      _ = ((31 / 32 : Rat) * (x * s)) ^ 2 := by ring
      _ ≤ _ := lowSquare
  have high : rough ^ 2 ≤ (5 / 4 : Rat) * x := by
    calc
      _ ≤ _ := highSquare
      _ = ((33 / 32 : Rat) ^ 2 * (x * s ^ 2)) * x := by ring
      _ ≤ ((33 / 32 : Rat) ^ 2 * (9 / 8)) * x :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left errorHigh (by positivity)) xPositive.le
      _ ≤ _ := mul_le_mul_of_nonneg_right (by norm_num) xPositive.le
  constructor <;> linarith

/-- The exact input to the final FMA rounding is positive even for the
most negative allowed residual. The interval allows for rounding of the
preceding multiply and residual instructions. -/
theorem exact_correction_range (x s rough half residual : Rat)
    (xPositive : 0 < x) (sPositive : 0 < s)
    (roughLower : (31 / 32 : Rat) * (x * s) ≤ rough)
    (roughUpper : rough ≤ (33 / 32 : Rat) * (x * s))
    (halfNonnegative : 0 ≤ half) (halfUpper : half ≤ (17 / 32 : Rat) * s)
    (residualLower : -(x / 3) ≤ residual) (residualUpper : residual ≤ x / 3) :
    (1 / 2 : Rat) * (x * s) ≤ half * residual + rough ∧
      half * residual + rough ≤ 2 * (x * s) := by
  have absBound : half * (x / 3) ≤ (17 / 96 : Rat) * (x * s) := by
    calc
      _ ≤ ((17 / 32 : Rat) * s) * (x / 3) :=
        mul_le_mul_of_nonneg_right halfUpper (div_nonneg xPositive.le (by norm_num))
      _ = _ := by ring
  have low := mul_le_mul_of_nonneg_left residualLower halfNonnegative
  have high := mul_le_mul_of_nonneg_left residualUpper halfNonnegative
  rw [mul_neg] at low
  have nonnegative : 0 ≤ x * s := (mul_pos xPositive sPositive).le
  constructor <;> linarith

#axiomsClean reciprocal_range
#axiomsClean exact_residual_range
#axiomsClean exact_correction_range

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand
