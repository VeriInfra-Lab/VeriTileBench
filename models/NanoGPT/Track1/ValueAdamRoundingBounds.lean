import models.NanoGPT.Track1.ValueAdamBinaryExponent
import Mathlib.Data.Rat.Floor
import Mathlib.Tactic.Linarith

namespace VeriTile.Triton.FP

/-- Integer magnitude used by the nearest-even branch of FP32 rounding. -/
private def nearestInteger (q : Rat) : Nat :=
  let n := q.num.natAbs / q.den
  let rem := q.num.natAbs % q.den
  if 2 * rem > q.den || (2 * rem == q.den && n % 2 == 1) then n + 1 else n

private theorem nearest_integer_bounds (q : Rat) (nonnegative : 0 ≤ q) :
    ⌊q⌋ ≤ (nearestInteger q : Int) ∧ (nearestInteger q : Int) ≤ ⌊q⌋ + 1 := by
  have magnitude := Int.natAbs_of_nonneg (Rat.num_nonneg.mpr nonnegative)
  have division : ⌊q⌋ = ((q.num.natAbs / q.den : Nat) : Int) := by
    rw [Rat.floor_def', Int.natCast_ediv, magnitude]
  rw [division]
  dsimp only [nearestInteger]
  split
  · push_cast
    omega
  · omega

private theorem nearest_integer_error (q : Rat) (nonnegative : 0 ≤ q) :
    (nearestInteger q : Rat) - 1 ≤ q ∧ q < (nearestInteger q : Rat) + 1 := by
  have low := Int.floor_le q
  have high := Int.lt_floor_add_one q
  rcases nearest_integer_bounds q nonnegative with ⟨a, b⟩
  have lo : (⌊q⌋ : Rat) ≤ (nearestInteger q : Rat) := by exact_mod_cast a
  have hi : (nearestInteger q : Rat) ≤ (⌊q⌋ : Rat) + 1 := by exact_mod_cast b
  constructor <;> linarith

private def packNormal (negative : Bool) (exponent significand : Nat) : Value .fp32 :=
  Value.ofBits .fp32 ((if negative then 2147483648 else 0) +
    exponent * 8388608 + significand - 8388608)

private theorem fractionScale : (2 : Nat) ^ 23 = 8388608 := by decide +kernel
private theorem signScale : (2 : Nat) ^ (23 + 8) = 2147483648 := by decide +kernel
private theorem allWords : (2 : Nat) ^ (23 + 9) = 4294967296 := by decide +kernel

private theorem pack_normal_decode (negative : Bool) (exponent significand : Nat)
    (exponentPositive : 0 < exponent) (exponentFinite : exponent < 255)
    (sigLower : 8388608 ≤ significand) (sigUpper : significand < 16777216) :
    (packNormal negative exponent significand).decode =
      .finite negative ((significand : Rat) * pow2 ((exponent : Int) - 150)) := by
  let word := packNormal negative exponent significand
  have raw : word.raw = (if negative then 2147483648 else 0) +
      exponent * 8388608 + significand - 8388608 := by
    simp only [word, packNormal, Value.raw, Value.ofBits, BitVec.toNat_ofNat, Format.width,
      Format.fractionBits, allWords]
    apply Nat.mod_eq_of_lt
    cases negative <;> simp <;> omega
  have sign : word.sign = negative := by
    simp only [Value.sign, Format.fractionBits, signScale, raw]
    cases negative <;> simp <;> omega
  have exp : word.exponent = exponent := by
    simp only [Value.exponent, Format.fractionBits, fractionScale, raw]
    cases negative <;> simp <;> omega
  have frac : word.fraction = significand - 8388608 := by
    simp only [Value.fraction, Format.fractionBits, fractionScale, raw]
    cases negative <;> simp <;> omega
  have exp0 : exponent ≠ 0 := Nat.ne_of_gt exponentPositive
  have exp255 : exponent ≠ 255 := Nat.ne_of_lt exponentFinite
  have significandEq : 8388608 + (significand - 8388608) = significand := by omega
  change word.decode = _
  simp only [Value.decode, exp, frac, sign, Format.fractionBits, beq_iff_eq, exp0, exp255, if_false,
    fractionScale, significandEq, Rat.ofInt_eq_cast, Int.cast_natCast]
  have factor : (exponent : Int) - 127 - ((23 : Nat) : Int) = (exponent : Int) - 150 := by omega
  rw [factor]

#axiomsClean nearest_integer_error
#axiomsClean pack_normal_decode

private theorem pow2_mul (left right : Int) : pow2 left * pow2 right = pow2 (left + right) := by
  simp only [pow2_as_zpow]
  exact (zpow_add₀ (by norm_num : (2 : Rat) ≠ 0) left right).symm

private theorem pow2_mono : Monotone pow2 := by
  intro a b order
  simpa only [pow2_as_zpow] using zpow_le_zpow_right₀ (by norm_num : (1 : Rat) ≤ 2) order

private theorem pow2_pos (e : Int) : 0 < pow2 e := by
  rw [pow2_as_zpow]
  exact zpow_pos (by norm_num) e

set_option maxRecDepth 1600 in
theorem round_normal_error (q : Rat) (negative flushInputs flushResults : Bool)
    (lower : pow2 (-126) ≤ q) (upper : q < pow2 126) :
    ∃ output : Rat,
      (round .fp32 ⟨.nearestEven, flushInputs, flushResults⟩ (if negative then -q else q)).decode = .finite negative output ∧
      output - q ≤ q / 8388608 ∧ q - output ≤ q / 8388608 := by
  let guess : Int := (q.num.natAbs.log2 : Int) - q.den.log2
  let e := if q < pow2 guess then guess - 1 else guess
  let unit := pow2 (e - 23)
  let n := nearestInteger (q / unit)
  have qPos : 0 < q := lt_of_lt_of_le (pow2_pos _) lower
  let input := if negative then -q else q
  have qNotZero : (input == 0) = false := by
    apply beq_eq_false_iff_ne.mpr
    cases negative <;> simp [input, ne_of_gt qPos]
  have absolute : input.abs = q := by
    cases negative <;> simp [input, Rat.abs_neg, Rat.abs_of_nonneg qPos.le]
  have qSign : decide (input < 0) = negative := by
    cases negative
    · simp [input, not_lt_of_ge qPos.le]
    · simp [input, qPos]
  have signedCondition : input < 0 ↔ negative = true := by
    rw [← qSign, decide_eq_true_eq]
  have uPos : 0 < unit := pow2_pos _
  have interval : pow2 e ≤ q ∧ q < pow2 (e + 1) := positive_binade q qPos
  have loExp : (-126 : Int) ≤ e := by
    by_contra! out
    exact (not_le_of_gt interval.2) (lower.trans' (pow2_mono (show e + 1 ≤ -126 by omega)))
  have hiExp : e < 126 := by
    by_contra! out
    exact (not_le_of_gt upper) ((pow2_mono out).trans interval.1)
  have lowUnit : (8388608 : Rat) * unit = pow2 e := by
    change pow2 23 * pow2 (e - 23) = _
    rw [pow2_mul]
    congr 1
    omega
  have highUnit : (16777216 : Rat) * unit = pow2 (e + 1) := by
    change pow2 24 * pow2 (e - 23) = _
    rw [pow2_mul]
    congr 1
    omega
  have gridLow : (8388608 : Rat) ≤ q / unit := (le_div_iff₀ uPos).mpr (lowUnit ▸ interval.1)
  have gridHigh : q / unit < (16777216 : Rat) := (div_lt_iff₀ uPos).mpr (highUnit ▸ interval.2)
  have gridPos : 0 ≤ q / unit := div_nonneg qPos.le uPos.le
  have floorLow : (8388608 : Int) ≤ ⌊q / unit⌋ := Int.le_floor.mpr (by exact_mod_cast gridLow)
  have floorHigh : ⌊q / unit⌋ < (16777216 : Int) := Int.floor_lt.mpr (by exact_mod_cast gridHigh)
  have intBounds := nearest_integer_bounds (q / unit) gridPos
  have nLowInt : (8388608 : Int) ≤ (n : Int) := by
    change ⌊q / unit⌋ ≤ (n : Int) ∧ _ at intBounds
    omega
  have nHighInt : (n : Int) ≤ 16777216 := by
    have cap : (n : Int) ≤ ⌊q / unit⌋ + 1 := intBounds.2
    omega
  have nLow : 8388608 ≤ n := by exact_mod_cast nLowInt
  have nHigh : n ≤ 16777216 := by exact_mod_cast nHighInt
  have gridError := nearest_integer_error (q / unit) gridPos
  have unitRelative : unit ≤ q / 8388608 := by
    apply (le_div_iff₀ (by norm_num : (0 : Rat) < 8388608)).mpr
    rw [mul_comm, lowUnit]
    exact interval.1
  have roundedLow : ((n : Rat) - 1) * unit ≤ q := (le_div_iff₀ uPos).mp gridError.1
  have roundedHigh : q ≤ ((n : Rat) + 1) * unit := le_of_lt ((div_lt_iff₀ uPos).mp gridError.2)
  let carry := n ≥ 16777216
  let exponent := e + (if carry then 1 else 0)
  let significand := if carry then n / 2 else n
  have expFinite : ¬exponent > 127 := by
    dsimp only [exponent]
    split <;> omega
  have sigNormal : ¬significand < 8388608 := by
    dsimp only [significand, carry]
    split <;> omega
  have packed : round .fp32 ⟨.nearestEven, flushInputs, flushResults⟩ (if negative then -q else q) =
      packNormal negative (exponent + 127).toNat significand := by
    have maxExp : max (if q < pow2 guess then guess - 1 else guess) (-126) = e := max_eq_left loExp
    have normalizeExponent : e - 23 + 23 + (if carry then 1 else 0) = exponent := by
      dsimp only [exponent]
      omega
    change round .fp32 ⟨.nearestEven, flushInputs, flushResults⟩ input = _
    unfold round
    dsimp only
    simp only [qNotZero, Bool.false_eq_true, if_false, absolute, Format.fractionBits,
      signedCondition]
    rw [maxExp]
    change (if e - 23 + 23 + (if carry then 1 else 0) > 127 then _ else _) = _
    rw [normalizeExponent, if_neg expFinite]
    change (if significand < 8388608 then _ else _) = _
    rw [if_neg sigNormal]
    simp only [packNormal, fractionScale, signScale]
    have explicitExponent :
        e - ((23 : Nat) : Int) + ((23 : Nat) : Int) + (if carry then 1 else 0) = exponent := by
      dsimp only [exponent]
      omega
    have equalBits :
        (if negative then 2147483648 else 0) +
          (e - ((23 : Nat) : Int) + ((23 : Nat) : Int) + (if carry then 1 else 0) + 127).toNat *
            8388608 + significand - 8388608 =
          (if negative then 2147483648 else 0) +
            (exponent + 127).toNat * 8388608 + significand - 8388608 := by
      rw [explicitExponent]
    cases negative <;> exact congrArg (Value.ofBits .fp32) equalBits
  refine ⟨(n : Rat) * unit, ?_, ?_, ?_⟩
  · by_cases carryCase : n ≥ 16777216
    · have atCarry : n = 16777216 := by omega
      have encodedExp : exponent = e + 1 := by simp [exponent, carry, carryCase]
      have encodedSig : significand = 8388608 := by simp [significand, carry, atCarry]
      rw [packed, encodedExp, encodedSig]
      have expCast : e + 1 + 127 = e + 128 := by omega
      rw [expCast, pack_normal_decode negative (e + 128).toNat 8388608 (by omega)
        (by omega) (by decide) (by decide), atCarry]
      have factor : ((e + 128).toNat : Int) - 150 = e - 22 := by omega
      rw [factor]
      have uDouble : pow2 (e - 22) = 2 * unit := by
        change pow2 (e - 22) = pow2 1 * pow2 (e - 23)
        rw [pow2_mul]
        have same : (1 : Int) + (e - 23) = e - 22 := by omega
        rw [same]
      rw [uDouble]
      have combine : (8388608 : Rat) * (2 * unit) = (16777216 : Rat) * unit := by ring
      simpa only [Nat.cast_ofNat] using congrArg (Decoded.finite negative) combine
    · have belowCarry : n < 16777216 := lt_of_not_ge carryCase
      have encodedExp : exponent = e := by simp [exponent, carry, carryCase]
      have encodedSig : significand = n := by simp [significand, carry, carryCase]
      have rounded : round .fp32 ⟨.nearestEven, flushInputs, flushResults⟩ (if negative then -q else q) =
          packNormal negative (e + 127).toNat n := by rw [packed, encodedExp, encodedSig]
      rw [rounded, pack_normal_decode negative (e + 127).toNat n (by omega) (by omega) nLow belowCarry]
      have factor : ((e + 127).toNat : Int) - 150 = e - 23 := by omega
      rw [factor]
  · linarith [mul_sub ((n : Rat)) 1 unit]
  · linarith [add_mul ((n : Rat)) 1 unit]

theorem round_positive_error (q : Rat) (flushInputs flushResults : Bool)
    (lower : pow2 (-126) ≤ q) (upper : q < pow2 126) :
    ∃ output : Rat,
      (round .fp32 ⟨.nearestEven, flushInputs, flushResults⟩ q).decode = .finite false output ∧
      output - q ≤ q / 8388608 ∧ q - output ≤ q / 8388608 :=
  round_normal_error q false flushInputs flushResults lower upper

#axiomsClean round_normal_error
#axiomsClean round_positive_error

private theorem decode_subnormal_bits (negative : Bool) (significand : Nat)
    (small : significand < 8388608) :
    (Value.ofBits .fp32 ((if negative then 2147483648 else 0) + significand)).decode =
      .finite negative ((significand : Rat) * pow2 (-149)) := by
  let word := Value.ofBits .fp32 ((if negative then 2147483648 else 0) + significand)
  have raw : word.raw = (if negative then 2147483648 else 0) + significand := by
    simp only [word, Value.raw, Value.ofBits, BitVec.toNat_ofNat, Format.width, Format.fractionBits, allWords]
    apply Nat.mod_eq_of_lt
    cases negative <;> simp <;> omega
  have sign : word.sign = negative := by
    simp only [Value.sign, Format.fractionBits, signScale, raw]
    cases negative <;> simp <;> omega
  have exp : word.exponent = 0 := by
    simp only [Value.exponent, Format.fractionBits, fractionScale, raw]
    cases negative <;> simp <;> omega
  have frac : word.fraction = significand := by
    simp only [Value.fraction, Format.fractionBits, fractionScale, raw]
    cases negative <;> simp <;> omega
  change word.decode = _
  simp only [Value.decode, sign, exp, frac, Format.fractionBits, beq_iff_eq, show 0 ≠ 255 by decide,
    if_false, if_true, Rat.ofInt_eq_cast, Int.cast_natCast]
  rfl

private theorem zero_decode (negative : Bool) :
    (Value.zero .fp32 negative).decode = .finite negative 0 := by
  cases negative <;> decide +kernel

set_option maxRecDepth 1600 in
theorem round_small_output (q : Rat) (negative flushInputs flushResults zeroSign : Bool)
    (nonnegative : 0 ≤ q) (small : q < pow2 (-126)) :
    ∃ sign magnitude, (round .fp32 ⟨.nearestEven, flushInputs, flushResults⟩
        (if negative then -q else q) zeroSign).decode = .finite sign magnitude ∧
      0 ≤ magnitude ∧ magnitude ≤ pow2 (-126) := by
  rcases eq_or_lt_of_le nonnegative with qZero | qPos
  · have inputZero : (if negative then -q else q) = 0 := by rw [← qZero]; simp
    refine ⟨zeroSign, 0, ?_, le_rfl, (pow2_pos _).le⟩
    rw [inputZero]
    simpa only [round, beq_self_eq_true, if_true] using zero_decode zeroSign
  let input := if negative then -q else q
  let estimate : Int := (q.num.natAbs.log2 : Int) - q.den.log2
  let e := if q < pow2 estimate then estimate - 1 else estimate
  let unit := pow2 (-149)
  let n := nearestInteger (q / unit)
  have inputNotZero : (input == 0) = false := by
    apply beq_eq_false_iff_ne.mpr
    cases negative <;> simp [input, ne_of_gt qPos]
  have absolute : input.abs = q := by
    cases negative <;> simp [input, Rat.abs_neg, Rat.abs_of_nonneg nonnegative]
  have qSign : decide (input < 0) = negative := by
    cases negative
    · simp [input, not_lt_of_ge nonnegative]
    · simp [input, qPos]
  have signedCondition : input < 0 ↔ negative = true := by rw [← qSign, decide_eq_true_eq]
  have eSmall : e < -126 := by
    have interval : pow2 e ≤ q ∧ q < pow2 (e + 1) := positive_binade q qPos
    by_contra! large
    exact (not_le_of_gt small) ((pow2_mono large).trans interval.1)
  have normalThreshold : (8388608 : Rat) * unit = pow2 (-126) := by decide +kernel
  have gridHigh : q / unit < (8388608 : Rat) :=
    (div_lt_iff₀ (pow2_pos _)).mpr (normalThreshold ▸ small)
  have gridPos : 0 ≤ q / unit := div_nonneg nonnegative (pow2_pos _).le
  have upper : (n : Int) ≤ 8388608 := by
    have floorHigh : ⌊q / unit⌋ < (8388608 : Int) := Int.floor_lt.mpr (by exact_mod_cast gridHigh)
    have cap : (n : Int) ≤ ⌊q / unit⌋ + 1 := (nearest_integer_bounds _ gridPos).2
    omega
  have nHigh : n ≤ 8388608 := by exact_mod_cast upper
  have maximumUnit : max (if q < pow2 estimate then estimate - 1 else estimate) (-126) = -126 :=
    max_eq_right eSmall.le
  have noCarry : ¬n ≥ 16777216 := by omega
  have rounded : round .fp32 ⟨.nearestEven, flushInputs, flushResults⟩ input zeroSign =
      if n < 8388608 then
        if flushResults then Value.zero .fp32 negative
        else Value.ofBits .fp32 ((if negative then 2147483648 else 0) + n)
      else packNormal negative 1 n := by
    unfold round
    dsimp only
    simp only [inputNotZero, Bool.false_eq_true, if_false, absolute, Format.fractionBits,
      signedCondition]
    rw [maximumUnit]
    change (if (-126 : Int) + (if n ≥ 16777216 then 1 else 0) > 127 then _ else _) = _
    simp only [if_neg noCarry]
    change (if (-126 : Int) > 127 then _ else _) = _
    simp only [if_neg (by decide : ¬(-126 : Int) > 127), signScale, fractionScale]
    let significand := if n ≥ 16777216 then n / 2 else n
    let exponent := (-126 : Int) + (if n ≥ 16777216 then 1 else 0)
    have signWord : decide (negative = true) = negative := by cases negative <;> rfl
    have trunc : (2 : Nat) ^ (23 + 1) = 16777216 := by decide +kernel
    simp only [signWord, trunc]
    change (if significand < 8388608 then
        if flushResults then Value.zero .fp32 negative
        else Value.ofBits .fp32 ((if negative then 2147483648 else 0) + significand)
      else packNormal negative (exponent + 127).toNat significand) = _
    have finiteExponent : (exponent + 127).toNat = 1 := by simp [exponent, noCarry]
    rw [finiteExponent]
    simp only [significand, if_neg noCarry]
  change ∃ sign magnitude, (round .fp32 ⟨.nearestEven, flushInputs, flushResults⟩ input zeroSign).decode =
    .finite sign magnitude ∧ 0 ≤ magnitude ∧ magnitude ≤ pow2 (-126)
  rw [rounded]
  by_cases subnormal : n < 8388608
  · rw [if_pos subnormal]
    cases flushResults
    · simp only [Bool.false_eq_true, if_false, decode_subnormal_bits negative n subnormal]
      refine ⟨negative, (n : Rat) * unit, rfl, mul_nonneg (Nat.cast_nonneg _) (pow2_pos _).le, ?_⟩
      rw [← normalThreshold]
      exact mul_le_mul_of_nonneg_right (by exact_mod_cast nHigh) (pow2_pos _).le
    · simp only [if_true, zero_decode]
      exact ⟨negative, 0, rfl, le_rfl, (pow2_pos _).le⟩
  · have atBoundary : n = 8388608 := by omega
    rw [if_neg subnormal, atBoundary, pack_normal_decode negative 1 8388608 (by decide)
      (by decide) (by decide) (by decide)]
    have unitEq : ((1 : Nat) : Int) - 150 = -149 := by decide
    rw [unitEq]
    refine ⟨negative, (8388608 : Rat) * unit, ?_, mul_nonneg (by norm_num) (pow2_pos _).le, by rw [normalThreshold]⟩
    rfl

#axiomsClean round_small_output

end VeriTile.Triton.FP
