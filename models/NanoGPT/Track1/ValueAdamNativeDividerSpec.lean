import models.NanoGPT.Track1.ValueAdamDivisorBand
import models.NanoGPT.Track1.ValueAdamGuardedReciprocal
import models.NanoGPT.Track1.ValueAdamNativeWeights

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.DivisorBand

open VeriTile.Triton FP

private def config : Config := Config.ieeeValue

private def large (divisor : F32) : Bool :=
  FP.lt config (Value.ofBits .fp32 0x7e800000) divisor.abs

private def normalOrUnordered (divisor : F32) : Bool :=
  !(FP.lt config divisor.abs (Value.ofBits .fp32 0x00800000))

private def quarter (word : F32) : F32 :=
  FP.mul config word (Value.ofBits .fp32 0x3e800000)

private def expand (word : F32) : F32 :=
  FP.mul config word (Value.ofBits .fp32 0x4b800000)

/-- The two scaling decisions in the pinned Hopper dynamic divider inspect
the original denominator. The operations on each original operand retain
their actual multiplication order. -/
def scaledOperand (divisor word : F32) : F32 :=
  let word := if large divisor then quarter word else word
  if normalOrUnordered divisor then word else expand word

/-- Retain a bit-level payload operation for the original multiply on NaN
numerators. On the direct route the checked exponent reduction excludes
every NaN numerator. -/
def product (nanProduct : Bool → F32 → F32 → F32)
    (inverseFirst : Bool) (numerator inverse : F32) : F32 :=
  if numerator.isNaN then nanProduct inverseFirst numerator inverse
  else if inverseFirst then FP.mul config inverse numerator
  else FP.mul config numerator inverse

def referenceDivide (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (numerator divisor : F32) : F32 :=
  product nanProduct true (scaledOperand divisor numerator) (reciprocal (scaledOperand divisor divisor))

def directDivide (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (numerator divisor : F32) : F32 :=
  product nanProduct false numerator (reciprocal divisor)

private theorem positive_abs (divisor : F32) (positive : divisor.sign = false) :
    divisor.abs = divisor := by
  cases divisor with
  | mk bits =>
    change Value.ofBits .fp32 (bits.toNat % 2147483648) = ⟨bits⟩
    unfold Value.ofBits
    congr 1
    apply BitVec.eq_of_toNat_eq
    have fits : bits.toNat < 2147483648 := by
      change (bits.toNat / 2147483648 != 0) = false at positive
      have quotient := bne_eq_false_iff_eq.mp positive
      omega
    change bits.toNat % 2147483648 % 4294967296 = bits.toNat
    rw [Nat.mod_eq_of_lt fits, Nat.mod_eq_of_lt (by omega)]

private theorem lowerWord : (Value.ofBits .fp32 0x00800000).decode =
    .finite false (1 / (2 ^ 126 : Rat)) := by
  decide +kernel

private theorem upperWord : (Value.ofBits .fp32 0x7e800000).decode =
    .finite false (2 ^ 126 : Rat) := by
  decide +kernel

private theorem no_scale (divisor : F32) (value : Rat)
    (decoded : divisor.decode = .finite false value)
    (aboveMinimum : 1 / (2 ^ 126 : Rat) ≤ value) (belowMaximum : value ≤ 2 ^ 126) :
    large divisor = false ∧ normalOrUnordered divisor = true := by
  have sign : divisor.sign = false := by
    dsimp only [Value.decode] at decoded
    split at decoded
    · split at decoded <;> contradiction
    · exact (Decoded.finite.inj decoded).1
  have abs := positive_abs divisor sign
  have gtFalse : ¬(2 ^ 126 : Rat) < value := not_lt.mpr belowMaximum
  have ltFalse : ¬ value < 1 / (2 ^ 126 : Rat) := not_lt.mpr aboveMinimum
  constructor
  · unfold large FP.lt
    simp only [config, prepare, Config.ieeeValue, Bool.false_and, Bool.false_eq_true, if_false,
      upperWord, abs, decoded]
    change decide ((2 ^ 126 : Rat) < value) = false
    exact decide_eq_false gtFalse
  · unfold normalOrUnordered FP.lt
    simp only [config, prepare, Config.ieeeValue, Bool.false_and, Bool.false_eq_true, if_false,
      lowerWord, abs, decoded]
    change (!decide (value < 1 / (2 ^ 126 : Rat))) = true
    rw [decide_eq_false ltFalse]
    rfl

theorem direct_for_finite (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (pair : Pair) (nonNaN : pair.1.isNaN = false) (value : Rat)
    (decoded : pair.2.decode = .finite false value)
    (lower : 1 / (2 ^ 126 : Rat) ≤ value) (upper : value ≤ 2 ^ 126) :
    directDivide reciprocal nanProduct pair.1 pair.2 =
      referenceDivide reciprocal nanProduct pair.1 pair.2 := by
  rcases no_scale pair.2 value decoded lower upper with ⟨notLarge, normal⟩
  have commute := GuardedReciprocal.selected_exact (fun _ => reciprocal pair.2) [] pair.1
  simpa [GuardedReciprocal.selected, GuardedReciprocal.warpHasNaN, GuardedReciprocal.direct,
    GuardedReciprocal.reference, directDivide, referenceDivide, scaledOperand, notLarge,
    normal, product, config, nonNaN] using commute

theorem direct_for_pair (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (pair : Pair) (nonNaN : pair.1.isNaN = false)
    (lower : 0x20000000 ≤ pair.2.raw) (upper : pair.2.raw < 0x40000000) :
    directDivide reciprocal nanProduct pair.1 pair.2 =
      referenceDivide reciprocal nanProduct pair.1 pair.2 := by
  rcases denominator_magnitude pair.2 lower upper with ⟨value, decoded, lower, upper⟩
  exact direct_for_finite reciprocal nanProduct pair nonNaN value decoded lower (le_of_lt upper)

theorem direct_exact (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (warp : List (List Pair)) (lane : List Pair) (pair : Pair)
    (inWarp : lane ∈ warp) (inLane : pair ∈ lane) (clear : warpReject warp = false) :
    directDivide reciprocal nanProduct pair.1 pair.2 =
      referenceDivide reciprocal nanProduct pair.1 pair.2 := by
  rcases selected_inputs warp lane pair inWarp inLane clear with ⟨nonNaN, lower, upper⟩
  exact direct_for_pair reciprocal nanProduct pair nonNaN lower upper

def chosenDivide (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (warp : List (List Pair)) (pair : Pair) : F32 :=
  if warpReject warp then referenceDivide reciprocal nanProduct pair.1 pair.2
  else directDivide reciprocal nanProduct pair.1 pair.2

theorem selected_exact (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (warp : List (List Pair)) (lane : List Pair) (pair : Pair)
    (inWarp : lane ∈ warp) (inLane : pair ∈ lane) :
    chosenDivide reciprocal nanProduct warp pair = referenceDivide reciprocal nanProduct pair.1 pair.2 := by
  cases branch : warpReject warp
  · simpa [chosenDivide, branch] using direct_exact reciprocal nanProduct warp lane pair inWarp inLane branch
  · simp [chosenDivide, branch]

#axiomsClean direct_for_finite
#axiomsClean direct_exact
#axiomsClean selected_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.DivisorBand
