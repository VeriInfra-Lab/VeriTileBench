import VeriTile.Triton.Float.BitValue
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Fin.Basic

/-! Word semantics for the format changes in the training cross-entropy CTA.
The NaN encodings follow the CUDA header's software implementations. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEConversions
open VeriTile.Triton FP

private def signWord (negative : Bool) (mask : Nat) : Nat :=
  if negative then mask else 0

/-- Round a ratio of unsigned integers to nearest, with ties to even. -/
def roundEven (numerator denominator : Nat) : Nat :=
  let q := numerator / denominator
  let twiceRemainder := 2 * (numerator % denominator)
  if twiceRemainder < denominator then q
  else if denominator < twiceRemainder then q + 1
  else q + q % 2

theorem roundEven_error (numerator denominator : Nat) (positive : 0 < denominator) :
    2 * (roundEven numerator denominator * denominator - numerator) ≤ denominator ∧
    2 * (numerator - roundEven numerator denominator * denominator) ≤ denominator := by
  have qr := Nat.div_add_mod numerator denominator
  have remainder := Nat.mod_lt numerator positive
  simp only [roundEven]
  split
  · rw [Nat.mul_comm (numerator / denominator) denominator]
    omega
  · split
    · rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm (numerator / denominator) denominator]
      omega
    · have parity : numerator / denominator % 2 = 0 ∨ numerator / denominator % 2 = 1 := by
        omega
      rcases parity with even | odd
      · rw [even, Nat.add_zero, Nat.mul_comm (numerator / denominator) denominator]
        omega
      · rw [odd, Nat.add_mul, Nat.one_mul, Nat.mul_comm (numerator / denominator) denominator]
        omega

theorem roundEven_tie (numerator denominator : Nat)
    (tie : 2 * (numerator % denominator) = denominator) :
    roundEven numerator denominator % 2 = 0 := by
  unfold roundEven
  rw [tie]
  simp only [Nat.lt_irrefl, ↓reduceIte]
  omega

/-- Widen a subnormal with exponent step `2^step` into a normal word.
`biasAndStep` is the destination bias plus the source exponent step. -/
private def widenSubnormal (fraction fractionBits biasAndStep : Nat) : Nat :=
  let leading := fraction.log2
  (biasAndStep + leading) * 2 ^ fractionBits +
    (fraction - 2 ^ leading) * 2 ^ (fractionBits - leading)

def e4m3ToHalf (x : UInt8) : UInt16 :=
  let magnitude := x.toNat % 128
  let exponent := magnitude / 8
  let fraction := magnitude % 8
  let sign := signWord (x.toNat ≥ 128) 0x8000
  UInt16.ofNat <|
    if magnitude = 127 then 0x7fff
    else if exponent ≠ 0 then sign + (exponent + 8) * 1024 + fraction * 128
    else if fraction ≠ 0 then sign + widenSubnormal fraction 10 6
    else sign

def halfToFloat (x : UInt16) : Value .fp32 :=
  let magnitude := x.toNat % 32768
  let exponent := magnitude / 1024
  let fraction := magnitude % 1024
  let sign := signWord (x.toNat ≥ 32768) 0x80000000
  Value.ofBits .fp32 <|
    if exponent = 31 ∧ fraction ≠ 0 then 0x7fffffff
    else if exponent = 31 then sign + 0x7f800000
    else if exponent ≠ 0 then sign + (exponent + 112) * 0x800000 + fraction * 8192
    else if fraction ≠ 0 then sign + widenSubnormal fraction 23 103
    else sign

/-- Half and E5M2 have the same exponent bias and minimum normal exponent.
Adding the rounded fraction to the rebased exponent also handles carries
from the fraction into the exponent, including at the subnormal boundary. -/
private def narrowMagnitude (exponent fraction fractionBits limit : Nat) : Nat :=
  if exponent = 0 then 0
  else if exponent > 142 then limit
  else if exponent < 113 then
    roundEven (0x800000 + fraction) (2 ^ (113 - exponent + (23 - fractionBits)))
  else min limit ((exponent - 112) * 2 ^ fractionBits +
    roundEven fraction (2 ^ (23 - fractionBits)))

def floatToHalf (x : Value .fp32) : UInt16 :=
  UInt16.ofNat <| if x.isNaN then 0x7fff else
    signWord x.sign 0x8000 + narrowMagnitude x.exponent x.fraction 10 0x7c00

def floatToE5m2 (x : Value .fp32) : UInt8 :=
  UInt8.ofNat <| if x.isNaN then 0x7f else
    signWord x.sign 0x80 + narrowMagnitude x.exponent x.fraction 2 0x7b

/-- Convert a binary64 launch word using `.rn.ftz`. Only the binade just
below the minimum FP32 normal can round up to that normal before flushing. -/
def launchDoubleToFloat (x : UInt64) : Value .fp32 :=
  let negative := x.toNat ≥ 0x8000000000000000
  let exponent := (x.toNat / 2 ^ 52) % 2048
  let fraction := x.toNat % 2 ^ 52
  Value.ofBits .fp32 <|
    if exponent = 2047 ∧ fraction ≠ 0 then 0x7fffffff
    else signWord negative 0x80000000 +
      if exponent > 1150 then 0x7f800000
      else if exponent > 896 then (exponent - 896) * 0x800000 + roundEven fraction (2 ^ 29)
      else if exponent = 896 ∧ roundEven (2 ^ 52 + fraction) (2 ^ 30) = 0x800000 then 0x800000
      else 0

private def e4m3Rat (x : UInt8) : Option Rat :=
  let e := (x.toNat / 8) % 16
  let f := x.toNat % 8
  if e = 15 ∧ f = 7 then none else
    let m := if e = 0 then Rat.ofInt f * pow2 (-9) else Rat.ofInt (8 + f) * pow2 ((e : Int) - 10)
    some (if x.toNat < 128 then m else -m)

/-- Widening preserves every finite E4M3 value through the halfword, and
the two NaN words remain nonnumeric. -/
theorem e4m3_widen_exact (x : Fin 256) :
    (halfToFloat (e4m3ToHalf (UInt8.ofNat x.val))).toRat? = e4m3Rat (UInt8.ofNat x.val) := by
  decide +kernel +revert

theorem narrow_boundaries :
    floatToHalf (Value.ofBits .fp32 0x33000000) = 0 ∧
    floatToHalf (Value.ofBits .fp32 0xb3000000) = 0x8000 ∧
    floatToHalf (Value.ofBits .fp32 0x33000001) = 1 ∧
    floatToHalf (Value.ofBits .fp32 0x387fe000) = 0x0400 ∧
    floatToHalf (Value.ofBits .fp32 0x477fefff) = 0x7bff ∧
    floatToHalf (Value.ofBits .fp32 0x477ff000) = 0x7c00 ∧
    floatToE5m2 (Value.ofBits .fp32 0x37000000) = 0 ∧
    floatToE5m2 (Value.ofBits .fp32 0x37000001) = 1 ∧
    floatToE5m2 (Value.infinity .fp32 true) = 0xfb ∧
    floatToE5m2 (Value.infinity .fp32) = 0x7b := by
  decide +kernel

theorem launch_boundaries :
    (launchDoubleToFloat 0x4037000000000000).raw = 0x41b80000 ∧
    (launchDoubleToFloat 0x4014000000000000).raw = 0x40a00000 ∧
    (launchDoubleToFloat 0x401e000000000000).raw = 0x40f00000 ∧
    (launchDoubleToFloat 0x3ff0000010000000).raw = 0x3f800000 ∧
    (launchDoubleToFloat 0x3ff0000010000001).raw = 0x3f800001 ∧
    (launchDoubleToFloat 0x380fffffe0000000).raw = 0x00800000 ∧
    (launchDoubleToFloat 0x380fffffdfffffff).raw = 0 ∧
    (launchDoubleToFloat 0xb80fffffdfffffff).raw = 0x80000000 ∧
    (launchDoubleToFloat 0x47efffffefffffff).raw = 0x7f7fffff ∧
    (launchDoubleToFloat 0x47effffff0000000).raw = 0x7f800000 := by
  decide +kernel

#axiomsClean roundEven_error
#axiomsClean roundEven_tie
#axiomsClean e4m3_widen_exact
#axiomsClean narrow_boundaries
#axiomsClean launch_boundaries
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEConversions
