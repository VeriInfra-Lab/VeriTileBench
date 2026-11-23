import models.NanoGPT.Track1.Proofs.MuddPackingNormalization.Bits0000
import models.NanoGPT.Track1.Proofs.MuddPackingNormalization.Bits2000
import models.NanoGPT.Track1.Proofs.MuddPackingNormalization.Bits4000
import models.NanoGPT.Track1.Proofs.MuddPackingNormalization.Bits6000
import models.NanoGPT.Track1.Proofs.MuddPackingNormalization.Bits8000
import models.NanoGPT.Track1.Proofs.MuddPackingNormalization.BitsA000
import models.NanoGPT.Track1.Proofs.MuddPackingNormalization.BitsC000
import models.NanoGPT.Track1.Proofs.MuddPackingNormalization.BitsE000
import Mathlib.Algebra.Order.Field.Rat
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddPackingArithmetic
open VeriTile.Triton.FP

-- The independent checks cover consecutive blocks of the raw BF16 domain.
private theorem checkedBlocks (hi : Nat) (bound : hi < 64) : blockPasses hi = true := by
  have ordinal : hi / 8 < 8 := by omega
  interval_cases h : hi / 8
  · exact checked0000 hi h
  · exact checked2000 hi h
  · exact checked4000 hi h
  · exact checked6000 hi h
  · exact checked8000 hi h
  · exact checkedA000 hi h
  · exact checkedC000 hi h
  · exact checkedE000 hi h

private theorem checkedWord (bits : Nat) (bound : bits < 65536) : checkedPackingWord bits = true := by
  have hi : bits / 1024 < 64 := by omega
  have lo : bits % 1024 ∈ List.range 1024 := List.mem_range.mpr (Nat.mod_lt _ (by decide))
  have checked := List.all_eq_true.mp (checkedBlocks (bits / 1024) hi) _ lo
  have splitWord : bits / 1024 * 1024 + bits % 1024 = bits := by omega
  simpa only [splitWord] using checked

private theorem value_of_beq {x y : Value .fp32} (same : (x == y) = true) : x = y := by
  cases x with
  | mk xBits =>
    cases y with
    | mk yBits =>
      change (xBits == yBits) = true at same
      exact congrArg Value.mk (eq_of_beq same)

private theorem finite_flags {x : Value .fp32} {sign : Bool} {mag : Rat}
    (h : x.decode = .finite sign mag) : x.isInf = false ∧ x.isNaN = false := by
  by_cases exp : x.exponent = 255
  · have flag : (x.exponent == 255) = true := by simp [exp]
    simp only [Value.decode, flag, ↓reduceIte] at h
    split at h <;> contradiction
  · have flag : (x.exponent == 255) = false := Bool.eq_false_iff.mpr (fun e => exp (eq_of_beq e))
    simp only [Value.isInf, Value.isNaN, flag, Bool.false_and, and_self]

private theorem round_at_nonzero {f : Format} (q : Rat) (left right : Bool) (nonzero : q ≠ 0) :
    round f Config.ieeeValue q left = round f Config.ieeeValue q right := by
  unfold round
  simp [nonzero]

private theorem fma_zero_of_nonzero_finite {x y : Value .fp32} {sign : Bool} {m n : Rat}
    (hx : x.decode = .finite sign m) (hy : y.decode = .finite false n)
    (hm : m ≠ 0) (hn : n ≠ 0) :
    fma Config.ieeeValue x y (Value.zero .fp32) = mul Config.ieeeValue x y := by
  have ⟨xi, xn⟩ := finite_flags hx
  have ⟨yi, yn⟩ := finite_flags hy
  have zn : (Value.zero .fp32).isNaN = false := by decide +kernel
  have hz : (Value.zero .fp32).decode = .finite false 0 := by decide +kernel
  unfold fma mul
  simp only [prepare, Config.ieeeValue, Bool.false_and, Bool.false_eq_true, if_false, hx, hy, hz, xi, xn, yi, yn, zn,
    Bool.and_false, Bool.or_false]
  cases sign
  · change round .fp32 Config.ieeeValue (m * n + 0) _ = round .fp32 Config.ieeeValue (m * n) false
    rw [add_zero]
    exact round_at_nonzero _ _ _ (mul_ne_zero hm hn)
  · change round .fp32 Config.ieeeValue (-(m * n) + 0) _ = round .fp32 Config.ieeeValue (-(m * n)) true
    rw [add_zero]
    exact round_at_nonzero _ _ _ (neg_ne_zero.mpr (mul_ne_zero hm hn))

private theorem special_ok {x : Value .fp32} (h : checkSpecialPacking x = true) :
    addZero (addZero x) = addZero x ∧
    fma Config.ieeeValue x tenth (Value.zero .fp32) = mul Config.ieeeValue (addZero x) tenth := by
  let ⟨stable, fused⟩ := Bool.and_eq_true_iff.mp h
  exact ⟨value_of_beq stable, value_of_beq fused⟩

private theorem packing_checks_sound {x : Value .fp32} (h : checkPackingOps x = true) :
    addZero (addZero x) = addZero x ∧
    fma Config.ieeeValue x tenth (Value.zero .fp32) = mul Config.ieeeValue (addZero x) tenth := by
  cases decode : x.decode with
  | infinity negative => exact special_ok (by simpa [checkPackingOps, decode] using h)
  | nan negative signaling payload => exact special_ok (by simpa [checkPackingOps, decode] using h)
  | finite negative mag =>
      by_cases magZero : mag = 0
      · exact special_ok (by simpa [checkPackingOps, decode, magZero] using h)
      · have addId : (addZero x == x) = true := by
          simpa [checkPackingOps, decode, magZero] using h
        have addEq := value_of_beq addId
        have gain : tenth.decode = .finite false (mkRat 13421773 134217728) := by decide +kernel
        have nonzero : mkRat 13421773 134217728 ≠ 0 := by decide +kernel
        exact ⟨congrArg addZero addEq, by rw [addEq]; exact fma_zero_of_nonzero_finite decode gain magZero nonzero⟩

theorem packing_numeric_eq (value : Value .bf16) :
    let x := cast .fp32 Config.ieeeValue value
    addZero (addZero x) = addZero x ∧
      fma Config.ieeeValue x tenth (Value.zero .fp32) = mul Config.ieeeValue (addZero x) tenth := by
  have checked := checkedWord value.raw value.bits.isLt
  have original : Value.ofBits .bf16 value.raw = value := by
    cases value with
    | mk bits =>
      apply congrArg (@Value.mk .bf16)
      change BitVec.ofNat Format.bf16.width bits.toNat = bits
      simp
  exact packing_checks_sound (by simpa only [checkedPackingWord, original] using checked)

theorem add_zero_twice (value : Value .bf16) :
    addZero (addZero (cast .fp32 Config.ieeeValue value)) = addZero (cast .fp32 Config.ieeeValue value) :=
  (packing_numeric_eq value).1

#axiomsClean packing_numeric_eq
#axiomsClean add_zero_twice
end VeriTile.Bench.Models.NanoGPT.Track1.MuddPackingArithmetic
