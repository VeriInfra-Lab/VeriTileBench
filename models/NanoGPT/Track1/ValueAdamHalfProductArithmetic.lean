import models.NanoGPT.Track1.ValueAdamReadScale
import Mathlib.Data.Fintype.Basic

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

/-- A binary16 payload has exponent 31 and a nonzero fraction. -/
def halfNaN (word : Half) : Bool :=
  word.toNat / 1024 % 32 == 31 && word.toNat % 1024 != 0

private def roundShift (significand shift : Nat) : Nat :=
  let denominator := 2 ^ shift
  let quotient := significand / denominator
  let remainder := significand % denominator
  quotient + if 2 * remainder > denominator ||
    (2 * remainder == denominator && quotient % 2 == 1) then 1 else 0

/-- Binary16 RNE multiplication by 2^-3. For a non-NaN value this removes
three exponent units or rounds the exact significand in subnormal units.
NaN payloads belong to the original conversion path. -/
def finiteHalfProduct (word : Half) : Half :=
  let sign := word.toNat / 32768 * 32768
  let magnitude := word.toNat % 32768
  let exponent := magnitude / 1024
  let fraction := magnitude % 1024
  let result := if exponent == 31 then magnitude
    else if exponent ≥ 4 then magnitude - 3072
    else if exponent == 0 then roundShift fraction 3
    else roundShift (1024 + fraction) (4 - exponent)
  UInt16.ofNat (sign + result)

def selectHalfProduct (word : Half) : Half :=
  if halfNaN word then scaleRead word else finiteHalfProduct word

def HalfProductSlice (first count : Nat) : Prop :=
  ∀ offset : Fin count,
    let word := UInt16.ofNat (first + offset.val)
    selectHalfProduct word = scaleRead word

theorem HalfProductSlice.append {first left right : Nat}
    (prefixOK : HalfProductSlice first left) (suffixOK : HalfProductSlice (first + left) right) :
    HalfProductSlice first (left + right) := by
  intro offset
  by_cases inPrefix : offset.val < left
  · exact prefixOK ⟨offset.val, inPrefix⟩
  · have fit := offset.isLt
    have suffix := suffixOK ⟨offset.val - left, by omega⟩
    change selectHalfProduct (UInt16.ofNat (first + left + (offset.val - left))) =
      scaleRead (UInt16.ofNat (first + left + (offset.val - left))) at suffix
    have same : first + left + (offset.val - left) = first + offset.val := by omega
    rw [same] at suffix
    exact suffix

/-- Each command checks one small closed interval inside the Lean kernel.
Making the interval proof opaque bounds kernel stack use. -/
syntax "check_binary16_interval " ident num : command

macro_rules
  | `(check_binary16_interval $name:ident $first:num) =>
    `(set_option maxRecDepth 30000 in
      set_option maxHeartbeats 0 in
      private theorem $name : HalfProductSlice $first 1024 := by
        unfold HalfProductSlice
        decide +kernel)

#axiomsClean HalfProductSlice.append

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
