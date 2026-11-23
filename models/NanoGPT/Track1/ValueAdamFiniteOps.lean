import models.NanoGPT.Track1.ValueAdamBinaryExponent

namespace VeriTile.Triton.FP

private abbrev Word := Value .fp32

private theorem normalThreshold : (8388608 : Rat) * pow2 (-149) = pow2 (-126) := by decide +kernel
private theorem fractionScale : (2 : Nat) ^ 23 = 8388608 := by decide +kernel
private theorem signScale : (2 : Nat) ^ (23 + 8) = 2147483648 := by decide +kernel
private theorem allWords : (2 : Nat) ^ (23 + 9) = 4294967296 := by decide +kernel

private theorem finiteExponent (word : Word) (sign : Bool) (magnitude : Rat)
    (decoded : word.decode = .finite sign magnitude) : word.exponent ≠ 255 := by
  intro out
  simp only [Value.decode, out, beq_self_eq_true, if_true] at decoded
  split at decoded <;> contradiction

private theorem finite_flags (word : Word) (sign : Bool) (magnitude : Rat)
    (decoded : word.decode = .finite sign magnitude) :
    word.isNaN = false ∧ word.isInf = false := by
  have normal := finiteExponent word sign magnitude decoded
  simp [Value.isNaN, Value.isInf, normal]

theorem prepare_normal_fp32 (word : Word) (sign : Bool) (magnitude : Rat) (cfg : Config)
    (decoded : word.decode = .finite sign magnitude) (lower : pow2 (-126) ≤ magnitude) :
    prepare cfg word = word := by
  have nonzeroExp : word.exponent ≠ 0 := by
    intro expZero
    have decodedSubnormal : word.decode = .finite word.sign ((word.fraction : Rat) * pow2 (-149)) := by
      simp only [Value.decode, expZero, Format.fractionBits, beq_self_eq_true, if_true, if_false,
        show ((0 : Nat) == 255) = false by decide, Bool.false_eq_true, Rat.ofInt_eq_cast, Int.cast_natCast]
      rfl
    rw [decodedSubnormal] at decoded
    have mag := (Decoded.finite.inj decoded).2
    have smaller : word.fraction < 8388608 := by
      unfold Value.fraction
      rw [Format.fractionBits, fractionScale]
      exact Nat.mod_lt _ (by decide)
    have quantum : 0 < pow2 (-149) := by rw [pow2_as_zpow]; exact zpow_pos (by norm_num) _
    have bound := mul_lt_mul_of_pos_right
      (show (word.fraction : Rat) < 8388608 by exact_mod_cast smaller) quantum
    rw [normalThreshold, mag] at bound
    exact not_lt_of_ge lower bound
  simp [prepare, Value.isSubnormal, nonzeroExp]

theorem fp32_neg_decode (word : Word) (sign : Bool) (magnitude : Rat)
    (decoded : word.decode = .finite sign magnitude) :
    word.neg.decode = .finite (!sign) magnitude := by
  have fits : word.raw < 4294967296 := word.bits.isLt
  let payload := word.raw % 2147483648
  have payloadBound : payload < 2147483648 := Nat.mod_lt _ (by decide)
  have negRaw : word.neg.raw = (if word.sign then 0 else 2147483648) + payload := by
    simp only [Value.neg, Value.raw, Value.ofBits, BitVec.toNat_ofNat, Format.width, Format.fractionBits,
      signScale, allWords, payload]
    apply Nat.mod_eq_of_lt
    split <;> omega
  have oldSign : word.sign = sign := by
    have notSpecial := finiteExponent word sign magnitude decoded
    simp only [Value.decode, beq_iff_eq, notSpecial, if_false, Decoded.finite.injEq] at decoded
    exact decoded.1
  have newSign : word.neg.sign = !sign := by
    simp only [Value.sign, Format.fractionBits, signScale, negRaw]
    change ((((if word.sign then 0 else 2147483648) + payload) / 2147483648) != 0) = _
    rw [oldSign]
    cases sign <;> simp <;> omega
  have sameExponent : word.neg.exponent = word.exponent := by
    simp only [Value.exponent, Format.fractionBits, fractionScale, negRaw, payload]
    cases word.sign <;> simp <;> omega
  have sameFraction : word.neg.fraction = word.fraction := by
    simp only [Value.fraction, Format.fractionBits, fractionScale, negRaw, payload]
    cases word.sign <;> simp <;> omega
  have notSpecial := finiteExponent word sign magnitude decoded
  simp only [Value.decode, sameExponent, sameFraction, newSign, beq_iff_eq,
    notSpecial, if_false, Decoded.finite.injEq]
  simp only [Value.decode, beq_iff_eq, notSpecial, if_false, Decoded.finite.injEq] at decoded
  exact ⟨True.intro, decoded.2⟩

theorem fp32_mul_normal (cfg : Config) (x y : Word) (qx qy : Rat)
    (xDecoded : x.decode = .finite false qx) (yDecoded : y.decode = .finite false qy)
    (xNormal : pow2 (-126) ≤ qx) (yNormal : pow2 (-126) ≤ qy) :
    FP.mul cfg x y = FP.round .fp32 cfg (qx * qy) := by
  simp only [FP.mul, prepare_normal_fp32 x false qx cfg xDecoded xNormal,
    prepare_normal_fp32 y false qy cfg yDecoded yNormal, xDecoded, yDecoded]
  rfl

private def signed (sign : Bool) (magnitude : Rat) : Rat :=
  if sign then -magnitude else magnitude

theorem fp32_fma_finite (x y z : Word) (sx sy sz : Bool) (qx qy qz : Rat)
    (xDecoded : x.decode = .finite sx qx) (yDecoded : y.decode = .finite sy qy)
    (zDecoded : z.decode = .finite sz qz) :
    FP.fma Config.ieeeValue x y z = FP.round .fp32 Config.ieeeValue
      (signed (sx != sy) (qx * qy) + signed sz qz)
      (if qx * qy == 0 && qz == 0 && (sx != sy) == sz then sx != sy else false) := by
  rcases finite_flags x sx qx xDecoded with ⟨xNaN, xInf⟩
  rcases finite_flags y sy qy yDecoded with ⟨yNaN, yInf⟩
  rcases finite_flags z sz qz zDecoded with ⟨zNaN, _⟩
  simp only [FP.fma, prepare, Config.ieeeValue, Bool.false_and, Bool.false_eq_true, if_false,
    xNaN, yNaN, zNaN, xInf, yInf, Bool.false_or, Bool.and_false, xDecoded, yDecoded, zDecoded]
  rfl

theorem fp32_add_positive (x y : Word) (qx qy : Rat)
    (xDecoded : x.decode = .finite false qx) (yDecoded : y.decode = .finite false qy) :
    FP.add Config.ieeeValue x y = FP.round .fp32 Config.ieeeValue (qx + qy) := by
  simp only [FP.add, prepare, Config.ieeeValue, Bool.false_and, Bool.false_eq_true, if_false,
    xDecoded, yDecoded]
  change FP.round .fp32 Config.ieeeValue (qx + qy)
    (if qx == 0 && qy == 0 && (false == false) then false else
      Config.ieeeValue.rounding == .towardNegative) = _
  have rounding : (Config.ieeeValue.rounding == .towardNegative) = false := by decide
  simp only [rounding, ite_self]
  rfl

theorem fp32_root_residual (rough x : Word) (r v : Rat)
    (rounded : rough.decode = .finite false r) (original : x.decode = .finite false v) :
    FP.fma Config.ieeeValue rough.neg rough x = FP.round .fp32 Config.ieeeValue (v - r ^ 2) := by
  have negative : rough.neg.decode = .finite true r := fp32_neg_decode rough false r rounded
  rw [fp32_fma_finite rough.neg rough x true false false r r v negative rounded original]
  simp [signed, pow_two, sub_eq_add_neg, add_comm]

theorem fp32_root_correction (half res rough : Word) (negative : Bool) (h d r : Rat)
    (halfDecoded : half.decode = .finite false h) (resDecoded : res.decode = .finite negative d)
    (roughDecoded : rough.decode = .finite false r) :
    FP.fma Config.ieeeValue half res rough = FP.round .fp32 Config.ieeeValue
      (h * (if negative then -d else d) + r) := by
  rw [fp32_fma_finite half res rough false negative false h d r halfDecoded resDecoded roughDecoded]
  cases negative <;> simp [signed, mul_neg]

#axiomsClean fp32_root_residual
#axiomsClean fp32_root_correction

#axiomsClean prepare_normal_fp32
#axiomsClean fp32_neg_decode
#axiomsClean fp32_mul_normal
#axiomsClean fp32_fma_finite
#axiomsClean fp32_add_positive

end VeriTile.Triton.FP
