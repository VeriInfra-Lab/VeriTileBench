import models.NanoGPT.Track1.ValueAdam
import Mathlib.Data.Fintype.Basic

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.ZeroSqrt

open VeriTile.Triton FP

def magnitudeMask : BitVec 32 := 0x7fffffff

private theorem magnitude_mask_bits (index : Fin 32) :
    magnitudeMask.getLsbD index.val = decide (index.val < 31) := by
  revert index
  decide +kernel

private theorem negative_zero_bits (index : Fin 32) :
    (0x80000000 : BitVec 32).getLsbD index.val = decide (index.val = 31) := by
  revert index
  decide +kernel

theorem zero_magnitude (bits : BitVec 32) :
    (bits &&& magnitudeMask) = 0 ↔ bits = 0 ∨ bits = 0x80000000 := by
  refine ⟨?_, by rintro (rfl | rfl) <;> decide +kernel⟩
  intro keep
  have low (i : Nat) (lt : i < 31) : bits.getLsbD i = false := by
    have same := congrArg (fun word : BitVec 32 => word.getLsbD i) keep
    simpa [BitVec.getLsbD_and, magnitude_mask_bits ⟨i, by omega⟩, lt] using same
  cases sign : bits.getLsbD 31
  · left
    apply BitVec.eq_of_getLsbD_eq
    intro i fit
    by_cases most : i = 31
    · subst i
      simpa using sign
    · simpa using low i (by omega)
  · right
    apply BitVec.eq_of_getLsbD_eq
    intro i fit
    rw [negative_zero_bits ⟨i, fit⟩]
    by_cases most : i = 31
    · subst i
      simpa using sign
    · simpa [most] using low i (by omega)

private theorem masked_or_zero (a b : BitVec 32) :
    ((a ||| b) &&& magnitudeMask) = 0 ↔
      (a &&& magnitudeMask) = 0 ∧ (b &&& magnitudeMask) = 0 := by
  rw [BitVec.and_or_distrib_right]
  exact BitVec.or_eq_zero_iff

def bits32 (x : F32) : BitVec 32 := x.bits

theorem sqrt_zero_bits (x : F32) (masked : ((bits32 x) &&& magnitudeMask) = 0) :
    sqrtRN x = x := by
  cases x with
  | mk word =>
    rcases (zero_magnitude word).mp masked with zero | negativeZero
    · subst word
      decide +kernel
    · subst word
      decide +kernel

def laneSummary (words : List F32) : BitVec 32 :=
  (words.foldr (fun word bits => (bits32 word) ||| bits) 0) &&& magnitudeMask

theorem lane_summary_zero (words : List F32) :
    laneSummary words = 0 ↔ ∀ x ∈ words, ((bits32 x) &&& magnitudeMask) = 0 := by
  induction words with
  | nil => decide
  | cons head tail ih =>
    simp only [laneSummary, List.foldr_cons, masked_or_zero]
    rw [show ((List.foldr (fun word bits => (bits32 word) ||| bits) 0 tail) &&& magnitudeMask) = 0 ↔
        ∀ x ∈ tail, ((bits32 x) &&& magnitudeMask) = 0 from ih]
    simp

def warpHasMagnitude (warp : List (List F32)) : Bool :=
  warp.any fun lane => laneSummary lane != 0

def read (warp : List (List F32)) (x : F32) : F32 :=
  if warpHasMagnitude warp then sqrtRN x else x

theorem shortcut_exact (warp : List (List F32)) (lane : List F32) (x : F32)
    (lanePresent : lane ∈ warp) (xPresent : x ∈ lane) : read warp x = sqrtRN x := by
  unfold read
  split
  · rfl
  · rename_i clear
    have zero : laneSummary lane = 0 := by
      by_contra notZero
      exact clear (List.any_eq_true.mpr ⟨lane, lanePresent, bne_iff_ne.mpr notZero⟩)
    exact (sqrt_zero_bits x ((lane_summary_zero lane).mp zero x xPresent)).symm

#axiomsClean zero_magnitude
#axiomsClean sqrt_zero_bits
#axiomsClean lane_summary_zero
#axiomsClean shortcut_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.ZeroSqrt
