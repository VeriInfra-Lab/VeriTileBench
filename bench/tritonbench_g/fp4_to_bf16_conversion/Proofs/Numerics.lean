import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Model
import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_FPReal

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
noncomputable section

namespace RealModel
/-- Mathematical observation of a bitcast. Special encodings have no Real
value; their placeholder is only used in branches excluded by the Real domain. -/
def bits (x : Float32Bits) : ℝ :=
  match (⟨x.bits⟩ : FP.Value .fp32).decode with
  | .finite sign q => if sign then -(q : ℝ) else (q : ℝ)
  | _ => 0

noncomputable instance : Numerics ℝ where
  two := 2
  bits := bits
  narrow := id
  pow2 := fun n => (2 : ℝ)^n
  mul := (· * ·)

/-- The independent E2M1 sign/exponent/mantissa formula agrees with the
complete copied helper, including each integer wrap and its final cast. -/
theorem converted_nibble (q : Bits.Nibble) :
    converted (α := ℝ) q.toNat = RealSpec.e2m1 q := by
  change bits ⟨BitVec.ofInt 32 (FPValue.integerWord q.toNat)⟩ = _
  rw [FPValue.integer_payload]
  unfold bits
  rw [(Bits.all_nibbles q).1]
  exact RealSpec.magnitude_formula q

#axiomsClean converted_nibble
end RealModel

namespace FPModel
abbrev Word := FP.Value .fp32

def embed (x : FP.Value .bf16) : Word := ⟨x.bits.zeroExtend 32⟩
def bf16 (x : Word) : FP.Value .bf16 := ⟨x.bits.truncate 16⟩

/-- Explicit software interpretation. It retains BF16 rounding, NaNs,
subnormals, overflow and signed zeros; no GPU conformance is assumed. -/
noncomputable instance : Numerics Word where
  two := ⟨0x40000000⟩
  bits := fun x => ⟨x.bits⟩
  narrow := fun x => embed (Bits.narrow x)
  pow2 := fun n => FP.round .fp32 .ieeeValue (FP.pow2 n)
  mul := fun x y => embed (FP.mul .ieeeValue (bf16 x) (bf16 y))

#axiomsClean embed
end FPModel

end
end VeriTile.Bench.Repairs.Fp4Scaled
