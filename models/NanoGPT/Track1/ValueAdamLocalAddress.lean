import models.NanoGPT.Track1.ValueAdamLocalLanes
import Mathlib.Tactic.Ring

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows

structure Address where
  block : Int := 0
  first : Int := 0
  offset : Int := 0
  count : Int := 0
  row : Int := 0
  byte : Int := 0
  deriving BEq, DecidableEq

def Address.plus (a b : Address) : Address :=
  ⟨a.block+b.block, a.first+b.first, a.offset+b.offset, a.count+b.count, a.row+b.row, a.byte+b.byte⟩

def Address.times (a : Address) (k : Int) : Address :=
  ⟨a.block*k, a.first*k, a.offset*k, a.count*k, a.row*k, a.byte*k⟩

def Address.literal (x : Int) : Address := {byte := x}

structure Inputs where
  block : Fin 6288
  first : Fin 1048576
  offset : Fin 1048576
  count : Fin 25153
  row : Fin 1048576

def Address.eval (a : Address) (input : Inputs) : Int :=
  a.block * input.block.val + a.first * input.first.val + a.offset * input.offset.val +
    a.count * input.count.val + a.row * input.row.val + a.byte

private def lowerWord (coefficient limit : Int) := min 0 (coefficient * limit)
private def upperWord (coefficient limit : Int) := max 0 (coefficient * limit)

private def Address.lower (a : Address) : Int :=
  lowerWord a.block 6287 + lowerWord a.first 1048575 + lowerWord a.offset 1048575 +
    lowerWord a.count 25152 + lowerWord a.row 1048575 + a.byte

private def Address.upper (a : Address) : Int :=
  upperWord a.block 6287 + upperWord a.first 1048575 + upperWord a.offset 1048575 +
    upperWord a.count 25152 + upperWord a.row 1048575 + a.byte

private theorem word_bounds (coefficient : Int) (value : Nat) (limit : Int) (valid : (value : Int) ≤ limit) :
    lowerWord coefficient limit ≤ coefficient * value ∧
      coefficient * value ≤ upperWord coefficient limit := by
  unfold lowerWord upperWord
  by_cases nonneg : 0 ≤ coefficient
  · exact ⟨le_trans (min_le_left _ _) (Int.mul_nonneg nonneg (by omega)),
      le_trans (Int.mul_le_mul_of_nonneg_left valid nonneg) (le_max_right _ _)⟩
  · have nonpos : coefficient ≤ 0 := by omega
    exact ⟨le_trans (min_le_right _ _) (Int.mul_le_mul_of_nonpos_left nonpos valid),
      le_trans (Int.mul_nonpos_of_nonpos_of_nonneg nonpos (by omega)) (le_max_left _ _)⟩

theorem Address.within_bounds (a : Address) (input : Inputs) :
    a.lower ≤ a.eval input ∧ a.eval input ≤ a.upper := by
  have b := word_bounds a.block input.block.val 6287 (by have bound := input.block.isLt; omega)
  have f := word_bounds a.first input.first.val 1048575 (by have bound := input.first.isLt; omega)
  have o := word_bounds a.offset input.offset.val 1048575 (by have bound := input.offset.isLt; omega)
  have n := word_bounds a.count input.count.val 25152 (by have bound := input.count.isLt; omega)
  have r := word_bounds a.row input.row.val 1048575 (by have bound := input.row.isLt; omega)
  simp only [Address.lower, Address.upper, Address.eval]
  omega

def Address.signed32 (a : Address) : Bool :=
  decide (-2147483648 ≤ a.lower ∧ a.upper < 2147483648)

def Address.unsigned32 (a : Address) : Bool :=
  decide (0 ≤ a.lower ∧ a.upper < 4294967296)

theorem Address.signed32_exact (a : Address) (fits : a.signed32 = true) (input : Inputs) :
    -2147483648 ≤ a.eval input ∧ a.eval input < 2147483648 := by
  have range := a.within_bounds input
  have accepted := of_decide_eq_true fits
  omega

theorem Address.unsigned32_exact (a : Address) (fits : a.unsigned32 = true) (input : Inputs) :
    0 ≤ a.eval input ∧ a.eval input < 4294967296 := by
  have range := a.within_bounds input
  have accepted := of_decide_eq_true fits
  omega

theorem Address.eval_mad (a b : Address) (scale : Int) (input : Inputs) :
    ((a.times scale).plus b).eval input = a.eval input * scale + b.eval input := by
  simp [Address.times, Address.plus, Address.eval]
  ring

/-- The decoded signed multiply-add has its full integer value when the
checked interval fits a 32-bit register. -/
theorem Address.mad_low_exact (a b : Address) (scale : Int) (input : Inputs)
    (fits : ((a.times scale).plus b).signed32 = true) :
    (Int32.ofInt (a.eval input) * Int32.ofInt scale + Int32.ofInt (b.eval input)).toInt =
      ((a.times scale).plus b).eval input := by
  rw [← Int32.ofInt_mul, ← Int32.ofInt_add, ← Address.eval_mad a b scale input]
  have valid := ((a.times scale).plus b).signed32_exact fits input
  exact Int32.toInt_ofInt_of_le valid.1 valid.2

#axiomsClean Address.mad_low_exact
#axiomsClean Address.signed32_exact
#axiomsClean Address.unsigned32_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows
