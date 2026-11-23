import VeriTile.Triton.Float.ScalarOps
import models.NanoGPT.Track1.FP8E4M3
import models.NanoGPT.Track1.TrainingCEConversions

/-! The training head converts the BF16 result of a scalar division to
E4M3FN. ATen widens before the reciprocal multiply, then writes BF16
before converting to FP8. The fused launch retains that narrowing. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.HeadActivation
open VeriTile.Triton.FP
open TrainingCEConversions (roundEven)

abbrev F32 := Value .fp32
abbrev BF16 := Value .bf16

def gain : F32 := Value.ofBits .fp32 0x408f5c29

def scale (reciprocal : F32) (input : BF16) : BF16 :=
  cast .bf16 Config.ieeeValue (mul Config.ieeeValue (cast .fp32 Config.ieeeValue input) reciprocal)

def magnitudeBits (input : BF16) : Fin 32768 :=
  ⟨input.raw % 32768, Nat.mod_lt _ (by decide)⟩

private def signByte (input : BF16) : Nat := if input.sign then 128 else 0

/-- C10's normal case rebiases the widened word and rounds the mantissa.
    In the subnormal case its FP32 addition to 16384 rounds in units of 2^-9. -/
def atenMagnitude (bits : Fin 32768) : Nat :=
  let exponent := bits.val / 128
  let fraction := bits.val % 128
  if bits.val ≥ 0x43f0 then 127
  else if exponent < 117 then 0
  else if exponent < 121 then roundEven (128 + fraction) (2 ^ (125 - exponent))
  else ((bits.val * 65536 + 0xc407ffff + fraction / 16 % 2) / 1048576) % 256

def magnitudeWide (bits : Fin 32768) : F32 := Value.ofBits .fp32 (bits.val * 65536)

def acceptsMagnitude (bits : Fin 32768) : Bool :=
  match (magnitudeWide bits).toRat? with
  | some magnitude => decide (magnitude ≤ 464)
  | none => false

def nativeMagnitude (bits : Fin 32768) : Nat :=
  if acceptsMagnitude bits then (FP8E4M3.clampConvert (magnitudeWide bits)).toNat else 127

private def segment (high : Fin 128) (low : Fin 256) : Fin 32768 :=
  ⟨high.val * 256 + low.val, by have hi := high.isLt; have lo := low.isLt; omega⟩

set_option maxRecDepth 8192 in
set_option maxHeartbeats 0 in
private theorem checked_magnitudes : ∀ (high : Fin 128) (low : Fin 256),
    atenMagnitude (segment high low) = nativeMagnitude (segment high low) := by
  decide +kernel

theorem magnitude_exact (bits : Fin 32768) : nativeMagnitude bits = atenMagnitude bits := by
  let high : Fin 128 := ⟨bits.val / 256, by have := bits.isLt; omega⟩
  let low : Fin 256 := ⟨bits.val % 256, Nat.mod_lt _ (by decide)⟩
  have same : segment high low = bits := Fin.ext (Nat.div_add_mod' bits.val 256)
  simpa only [same] using (checked_magnitudes high low).symm

def atenCast (input : BF16) : UInt8 :=
  UInt8.ofNat (signByte input + atenMagnitude (magnitudeBits input))

def nativeCast (input : BF16) : UInt8 :=
  UInt8.ofNat (signByte input + nativeMagnitude (magnitudeBits input))

def satfiniteByte (input : BF16) : UInt8 :=
  UInt8.ofNat (signByte input + (FP8E4M3.clampConvert (magnitudeWide (magnitudeBits input))).toNat)

def overflowByte (input : BF16) : UInt8 := UInt8.ofNat (signByte input + 127)

theorem select_native (input : BF16) :
    (if acceptsMagnitude (magnitudeBits input) then satfiniteByte input else overflowByte input) =
      nativeCast input := by
  cases h : acceptsMagnitude (magnitudeBits input) <;>
    simp [nativeCast, nativeMagnitude, satfiniteByte, overflowByte, h]

theorem cast_exact (input : BF16) : nativeCast input = atenCast input := by
  simp only [nativeCast, atenCast, magnitude_exact]

abbrev Run := (size : Nat) → (useKernel : Bool) → (reciprocal : F32) →
  (Fin size → BF16) → Fin size → UInt8

def reference : Run := fun _ _ reciprocal input i => atenCast (scale reciprocal (input i))

def eligible (size : Nat) (enabled : Bool) (reciprocal : F32) : Bool :=
  enabled && decide (reciprocal = gain) && decide (size < 2^31) && size % 512 == 0

def fused : Run := fun size enabled reciprocal input i =>
  if eligible size enabled reciprocal then nativeCast (scale gain (input i))
  else reference size false reciprocal input i

theorem fused_equal : fused = reference := by
  funext size enabled reciprocal input i
  unfold fused
  split
  · rename_i accepted
    simp only [eligible, Bool.and_eq_true, decide_eq_true_eq] at accepted
    simp [accepted.1.1.2, cast_exact, reference]
  · rfl

#axiomsClean magnitude_exact
#axiomsClean cast_exact
#axiomsClean select_native
#axiomsClean fused_equal
end VeriTile.Bench.Models.NanoGPT.Track1.HeadActivation
