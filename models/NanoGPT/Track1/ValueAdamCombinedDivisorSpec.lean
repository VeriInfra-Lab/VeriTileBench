import models.NanoGPT.Track1.ValueAdamNativeDividerSpec
import models.NanoGPT.Track1.ValueAdamZeroSqrt

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.CombinedDivisor

open VeriTile.Triton FP

abbrev Moments := F32 × F32

def epsilon : F32 := Value.ofBits .fp32 0x2edbe6ff

def originalDenominator (second : F32) : F32 :=
  FP.add Config.ieeeValue (sqrtRN second) epsilon

def sharedDenominator (second : F32) : F32 :=
  FP.add Config.ieeeValue second epsilon

def toDivision (point : Moments) : DivisorBand.Pair :=
  (point.1, originalDenominator point.2)

def hasMagnitude (warp : List (List Moments)) : Bool :=
  ZeroSqrt.warpHasMagnitude (warp.map (List.map Prod.snd))

def coldReject (warp : List (List Moments)) : Bool :=
  ExponentGuard.warpMayHaveNaN (warp.map (List.map Prod.fst))

theorem selected_signed_zero (warp : List (List Moments)) (lane : List Moments) (point : Moments)
    (inWarp : lane ∈ warp) (inLane : point ∈ lane) (zero : hasMagnitude warp = false) :
    ZeroSqrt.bits32 point.2 &&& ZeroSqrt.magnitudeMask = 0 := by
  let second := lane.map Prod.snd
  have laneThere : second ∈ warp.map (List.map Prod.snd) := List.mem_map.mpr ⟨lane, inWarp, rfl⟩
  have pointThere : point.2 ∈ second := List.mem_map.mpr ⟨point, inLane, rfl⟩
  apply (ZeroSqrt.lane_summary_zero second).mp _ point.2 pointThere
  have isZero := List.any_eq_false.mp zero second laneThere
  exact bne_eq_false_iff_eq.mp (Bool.eq_false_iff.mpr isZero)

theorem zero_denominators (word : F32)
    (zero : ZeroSqrt.bits32 word &&& ZeroSqrt.magnitudeMask = 0) :
    sharedDenominator word = epsilon ∧ originalDenominator word = epsilon := by
  unfold originalDenominator
  rw [ZeroSqrt.sqrt_zero_bits word zero]
  cases word with
  | mk bits =>
    rcases (ZeroSqrt.zero_magnitude bits).mp zero with ordinary | negative
    · subst bits
      decide +kernel
    · subst bits
      decide +kernel

theorem cold_denominators (warp : List (List Moments)) (lane : List Moments) (source point : Moments)
    (inWarp : lane ∈ warp) (sourceInLane : source ∈ lane) (pointInLane : point ∈ lane)
    (zero : hasMagnitude warp = false) :
    sharedDenominator source.2 = epsilon ∧ originalDenominator point.2 = epsilon :=
  ⟨(zero_denominators _ (selected_signed_zero warp lane source inWarp sourceInLane zero)).1,
   (zero_denominators _ (selected_signed_zero warp lane point inWarp pointInLane zero)).2⟩

theorem cold_direct_exact (reciprocal : F32 → F32) (nanProduct : Bool → F32 → F32 → F32)
    (warp : List (List Moments)) (lane : List Moments) (source point : Moments)
    (inWarp : lane ∈ warp) (sourceInLane : source ∈ lane) (pointInLane : point ∈ lane)
    (zero : hasMagnitude warp = false) (clear : coldReject warp = false) :
    DivisorBand.directDivide reciprocal nanProduct point.1 (sharedDenominator source.2) =
      DivisorBand.referenceDivide reciprocal nanProduct point.1 (originalDenominator point.2) := by
  have foundLane : lane.map Prod.fst ∈ warp.map (List.map Prod.fst) :=
    List.mem_map.mpr ⟨lane, inWarp, rfl⟩
  have foundWord : point.1 ∈ lane.map Prod.fst := List.mem_map.mpr ⟨point, pointInLane, rfl⟩
  have nonNaN := ExponentGuard.no_nan_on_lane_vote _ _ _ foundLane foundWord clear
  rcases cold_denominators warp lane source point inWarp sourceInLane pointInLane zero with ⟨d, e⟩
  rw [d, e]
  exact DivisorBand.direct_for_pair reciprocal nanProduct (point.1, epsilon) nonNaN
    (by change 0x20000000 ≤ epsilon.raw; decide +kernel)
    (by change epsilon.raw < 0x40000000; decide +kernel)

#axiomsClean cold_denominators
#axiomsClean cold_direct_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.CombinedDivisor
