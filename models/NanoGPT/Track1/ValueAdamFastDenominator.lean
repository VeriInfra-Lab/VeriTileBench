import models.NanoGPT.Track1.ValueAdamRootBandSpec
import models.NanoGPT.Track1.ValueAdamRootArithmeticBounds
import models.NanoGPT.Track1.ValueAdamRoundingBounds
import models.NanoGPT.Track1.ValueAdamFiniteOps
import models.NanoGPT.Track1.ValueAdamCombinedDivisorSpec

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand

open VeriTile.Triton FP

private def flush : Config := ⟨.nearestEven, true, true⟩
private def halfWord : F32 := Value.ofBits .fp32 0x3f000000
private def eps : Rat := 14411519 / 2 ^ 57

/-- The normalized squared result is a conservative encoding of the
documented relative error of the shared PTX reciprocal square root. -/
structure ApproximateRsqrt where
  run : F32 → F32
  magnitude : ∀ (word : F32) (value : Rat), word.decode = .finite false value →
    1 / 2 ^ 63 ≤ value → value < 2 →
    ∃ result : Rat, (run word).decode = .finite false result ∧ 0 < result ∧
      7 / 8 ≤ value * result ^ 2 ∧ value * result ^ 2 ≤ 9 / 8

private theorem minNormal : pow2 (-126) = (1 / 2 ^ 126 : Rat) := by decide +kernel
private theorem maxSafe : pow2 126 = (2 ^ 126 : Rat) := by decide +kernel

private theorem rounded_residual_bound (value input : Rat) (valueLower : 1 / 2 ^ 63 ≤ value)
    (valueUpper : value < 2) (residualLower : -(value / 4) ≤ input) (residualUpper : input ≤ value / 4) :
    ∃ negative result, (FP.round .fp32 Config.ieeeValue input).decode = .finite negative result ∧
      0 ≤ result ∧ result ≤ value / 3 := by
  let negative : Bool := decide (input < 0)
  let magnitude := if input < 0 then -input else input
  have nonnegative : 0 ≤ magnitude := by
    dsimp only [magnitude]
    split <;> linarith
  have magnitudeUpper : magnitude ≤ value / 4 := by
    dsimp only [magnitude]
    split <;> linarith
  have restore : (if negative then -magnitude else magnitude) = input := by
    dsimp only [negative, magnitude]
    by_cases sign : input < 0 <;> simp [sign]
  by_cases tiny : magnitude < pow2 (-126)
  · obtain ⟨sg, output, decoded, low, high⟩ :=
      round_small_output magnitude negative false false false nonnegative tiny
    rw [restore] at decoded
    rw [minNormal] at high
    exact ⟨sg, output, decoded, low, by linarith⟩
  · have positive : 0 < magnitude := by
      rw [minNormal] at tiny
      linarith
    have upper : magnitude < pow2 126 := by
      rw [maxSafe]
      linarith
    obtain ⟨output, decoded, errorUp, errorDown⟩ :=
      round_normal_error magnitude negative false false (le_of_not_gt tiny) upper
    rw [restore] at decoded
    refine ⟨negative, output, decoded, ?_, ?_⟩ <;> linarith

theorem guarded_denominator_range (rsqrt : ApproximateRsqrt) (warp : List (List F32))
    (lane : List F32) (word : F32) (inWarp : lane ∈ warp) (found : word ∈ lane)
    (clear : DivisorBand.rootReject warp = false) :
    ∃ magnitude : Rat,
      (FP.add Config.ieeeValue (fastRoot rsqrt.run word true) CombinedDivisor.epsilon).decode =
        .finite false magnitude ∧
      (1 / 2 ^ 126 : Rat) ≤ magnitude ∧ magnitude ≤ 2 ^ 126 := by
  obtain ⟨value, valueDecoded, valueLower, valueUpper⟩ :=
    DivisorBand.root_input_range warp lane word inWarp found clear
  obtain ⟨reciprocal, inverseDecoded, inversePositive, errorLow, errorHigh⟩ :=
    rsqrt.magnitude word value valueDecoded valueLower valueUpper
  have valuePositive : 0 < value := by linarith
  obtain ⟨inverseLower, inverseUpper⟩ :=
    reciprocal_range value reciprocal valueLower valueUpper inversePositive errorLow errorHigh
  have valueNormal : pow2 (-126) ≤ value := by rw [minNormal]; linarith
  have inverseNormal : pow2 (-126) ≤ reciprocal := by rw [minNormal]; linarith
  have halfNormal : pow2 (-126) ≤ (1 / 2 : Rat) := by rw [minNormal]; norm_num
  have halfDecode : halfWord.decode = .finite false (1 / 2) := by decide +kernel
  let roughWord := FP.mul flush word (rsqrt.run word)
  let halfOfInverse := FP.mul flush (rsqrt.run word) halfWord
  let residual := FP.fma Config.ieeeValue roughWord.neg roughWord word
  let corrected := FP.fma Config.ieeeValue halfOfInverse residual roughWord
  have productLow : (1 / 2 ^ 64 : Rat) ≤ value * reciprocal := by
    have relation := mul_le_mul valueLower inverseLower (by norm_num : (0 : Rat) ≤ 1 / 2) valuePositive.le
    norm_num at relation ⊢
    exact relation
  have productHigh : value * reciprocal < (2 : Rat) ^ 33 := by
    have relation := mul_lt_mul valueUpper inverseUpper.le inversePositive (by norm_num : (0 : Rat) ≤ 2)
    norm_num at relation ⊢
    exact relation
  have roughRound : roughWord = FP.round .fp32 flush (value * reciprocal) :=
    fp32_mul_normal flush _ _ value reciprocal valueDecoded inverseDecoded valueNormal inverseNormal
  have halfRound : halfOfInverse = FP.round .fp32 flush (reciprocal / 2) := by
    unfold halfOfInverse
    rw [fp32_mul_normal flush _ _ reciprocal (1 / 2) inverseDecoded halfDecode inverseNormal halfNormal]
    have equal : reciprocal * (1 / 2) = reciprocal / 2 := by ring
    rw [equal]
  obtain ⟨rough, roughDecoded, roughErrorUp, roughErrorDown⟩ :=
    round_positive_error (value * reciprocal) true true (by rw [minNormal]; linarith)
      (by rw [maxSafe]; linarith)
  obtain ⟨half, halfDecoded, halfErrorUp, halfErrorDown⟩ :=
    round_positive_error (reciprocal / 2) true true (by rw [minNormal]; linarith)
      (by rw [maxSafe]; linarith)
  change (FP.round .fp32 flush (value * reciprocal)).decode = _ at roughDecoded
  change (FP.round .fp32 flush (reciprocal / 2)).decode = _ at halfDecoded
  rw [← roughRound] at roughDecoded
  rw [← halfRound] at halfDecoded
  have roughLower : (31 / 32 : Rat) * (value * reciprocal) ≤ rough := by linarith
  have roughUpper : rough ≤ (33 / 32 : Rat) * (value * reciprocal) := by linarith
  have halfNonnegative : 0 ≤ half := by linarith
  have halfUpper : half ≤ (17 / 32 : Rat) * reciprocal := by linarith
  obtain ⟨residualLower, residualUpper⟩ :=
    exact_residual_range value reciprocal rough valuePositive inversePositive errorLow errorHigh roughLower roughUpper
  have residualRound : residual = FP.round .fp32 Config.ieeeValue (value - rough ^ 2) :=
    fp32_root_residual roughWord word rough value roughDecoded valueDecoded
  obtain ⟨negative, delta, deltaDecoded, deltaPositive, deltaBound⟩ :=
    rounded_residual_bound value (value - rough ^ 2) valueLower valueUpper residualLower residualUpper
  rw [← residualRound] at deltaDecoded
  let signedDelta := if negative then -delta else delta
  have signedLower : -(value / 3) ≤ signedDelta := by
    cases negative <;> simp [signedDelta] <;> linarith
  have signedUpper : signedDelta ≤ value / 3 := by
    cases negative <;> simp [signedDelta] <;> linarith
  obtain ⟨correctionLower, correctionUpper⟩ :=
    exact_correction_range value reciprocal rough half signedDelta
      valuePositive inversePositive roughLower roughUpper halfNonnegative halfUpper signedLower signedUpper
  have rootRound : corrected = FP.round .fp32 Config.ieeeValue (half * signedDelta + rough) :=
    fp32_root_correction halfOfInverse residual roughWord negative half delta rough halfDecoded deltaDecoded roughDecoded
  obtain ⟨root, rootDecoded, rootErrorUp, rootErrorDown⟩ :=
    round_positive_error (half * signedDelta + rough) false false (by rw [minNormal]; linarith)
      (by rw [maxSafe]; linarith)
  change (FP.round .fp32 Config.ieeeValue (half * signedDelta + rough)).decode = _ at rootDecoded
  rw [← rootRound] at rootDecoded
  have epsDecoded : CombinedDivisor.epsilon.decode = .finite false eps := by decide +kernel
  have epsPositive : 0 < eps := by decide +kernel
  have epsSmall : eps < 1 := by decide +kernel
  have rootLower : (1 / 2 ^ 66 : Rat) ≤ root := by linarith
  have rootUpper : root < (2 : Rat) ^ 35 := by linarith
  have denominatorRound := fp32_add_positive corrected CombinedDivisor.epsilon root eps rootDecoded epsDecoded
  obtain ⟨denominator, denominatorDecoded, denUp, denDown⟩ :=
    round_positive_error (root + eps) false false (by rw [minNormal]; linarith) (by rw [maxSafe]; linarith)
  change (FP.round .fp32 Config.ieeeValue (root + eps)).decode = _ at denominatorDecoded
  rw [← denominatorRound] at denominatorDecoded
  exact ⟨denominator, denominatorDecoded, by linarith, by linarith⟩

#axiomsClean guarded_denominator_range

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.RootBand
