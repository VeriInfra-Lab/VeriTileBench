import models.NanoGPT.Track1.ValueAdamRootBandSpec
import models.NanoGPT.Track1.ValueAdamCombinedDivisorSpec

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand

open VeriTile.Triton FP

private theorem power_bit (position slot : Fin width) :
    (BitVec.ofNat width (2 ^ position.val)).getLsbD slot.val = (position == slot) := by
  rw [BitVec.getLsbD_ofNat, Nat.testBit_two_pow]
  apply Bool.eq_iff_iff.mpr
  simp [slot.isLt, Fin.ext_iff]

private theorem fold_mask_bit (lane : Nat → F32) (slot : Fin width) (positions : List (Fin width))
    (initial : BitVec width) :
    (positions.foldl (fun bits position =>
      if DivisorBand.originalRootHelper (lane position.val)
      then bits ||| BitVec.ofNat width (2 ^ position.val) else bits) initial).getLsbD slot.val =
      (initial.getLsbD slot.val ||
       positions.any (fun position => (position == slot) && DivisorBand.originalRootHelper (lane position.val))) := by
  induction positions generalizing initial with
  | nil => simp
  | cons position rest induction =>
    simp only [List.foldl_cons, List.any_cons]
    cases selected : DivisorBand.originalRootHelper (lane position.val)
    · simp only [Bool.false_eq_true, if_false, Bool.and_false, Bool.false_or]
      exact induction initial
    · simp only [if_true, Bool.and_true]
      rw [induction, BitVec.getLsbD_or, power_bit, Bool.or_assoc]

theorem scalar_helper_mask_bit (lane : Nat → F32) (slot : Fin width) :
    (scalarHelperMask lane width).val.testBit slot.val = DivisorBand.originalRootHelper (lane slot.val) := by
  change ((List.finRange width).foldl (fun (bits : BitVec width) position =>
      if DivisorBand.originalRootHelper (lane position.val)
      then bits ||| BitVec.ofNat width (2 ^ position.val) else bits) 0).getLsbD slot.val = _
  rw [fold_mask_bit]
  have zero : (0 : BitVec width).getLsbD slot.val = false := by
    simp
  rw [zero, Bool.false_or]
  apply Bool.eq_iff_iff.mpr
  constructor
  · intro selected
    obtain ⟨i, _, h⟩ := List.any_eq_true.mp selected
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    simpa [h.1] using h.2
  · intro selected
    exact List.any_eq_true.mpr ⟨slot, List.mem_finRange slot, by simp [selected]⟩

/-- The first mask-and-branch in the inspected slow square-root routine
returns the untouched signed zero on either zero encoding. -/
def helperWithZero (otherwise : F32 → F32) (second : F32) : F32 :=
  if (ZeroSqrt.bits32 second &&& ZeroSqrt.magnitudeMask) == 0 then second else otherwise second

def observedRoot (rsqrt helper : F32 → F32) (second : F32) : F32 :=
  oldRoot rsqrt (helperWithZero helper) second

theorem root_helper_of_zero (second : F32)
    (zero : ZeroSqrt.bits32 second &&& ZeroSqrt.magnitudeMask = 0) :
    DivisorBand.originalRootHelper second = true := by
  cases second with
  | mk raw =>
    change BitVec 32 at raw
    change raw &&& ZeroSqrt.magnitudeMask = 0 at zero
    rcases (ZeroSqrt.zero_magnitude raw).mp zero with same | same
    · subst raw
      decide +kernel
    · subst raw
      decide +kernel

theorem observed_root_signed_zero (rsqrt helper : F32 → F32) (second : F32)
    (zero : ZeroSqrt.bits32 second &&& ZeroSqrt.magnitudeMask = 0) :
    observedRoot rsqrt helper second = second := by
  simp [observedRoot, oldRoot, helperWithZero, root_helper_of_zero second zero, zero]

theorem observed_denominator_of_zero (rsqrt helper : F32 → F32) (second : F32)
    (zero : ZeroSqrt.bits32 second &&& ZeroSqrt.magnitudeMask = 0) :
    FP.add Config.ieeeValue (observedRoot rsqrt helper second) CombinedDivisor.epsilon =
      CombinedDivisor.epsilon := by
  rw [observed_root_signed_zero rsqrt helper second zero]
  exact (CombinedDivisor.zero_denominators second zero).1

#axiomsClean scalar_helper_mask_bit
#axiomsClean observed_denominator_of_zero

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand
