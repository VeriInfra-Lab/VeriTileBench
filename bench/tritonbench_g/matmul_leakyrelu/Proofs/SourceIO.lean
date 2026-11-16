import bench.tritonbench_g.matmul_leakyrelu.Proofs.Support.PairIO
import VeriTile.Spec
import VeriTile.Triton.Float.DotContraction
import VeriTile.Triton.Float.DotAdmission

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.SourcePair
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu
noncomputable section

/-- Only a local K32 FP32 instruction fragment is registered. No complete
kernel, activation, memory, grid or output relation occurs in the assumption. -/
def fragmentRequest : Whole.Request where
  A := "A"; B := "B"; C := "C"
  rows := 32; inner := 32; innerB := 32; columns := 32
  strideAM := 32; strideAK := 1; strideBK := 32; strideBN := 1
  aContiguous := Bool.true; activation := ""

def separateFragment : List ComputeStmt :=
  FP.DotContraction.separate 32 32 32 "a" "b" "accumulator"
def contractedFragment : List ComputeStmt :=
  FP.DotContraction.seeded 32 32 32 "a" "b" "accumulator"

/-- The shared atoms are exactly the statements of these actual source bodies;
the rule never substitutes a complete kernel or drops a carrier conversion. -/
theorem separateFragment_source : separateFragment =
    ((FPValue.body fragmentRequest.parameters .single).take 3).drop 2 := by rfl
theorem contractedFragment_source : contractedFragment =
    ((FPFused.body fragmentRequest.contractedParameters .single).take 3).drop 2 := by rfl

/-- The shared source-specific experiment is retained with its actual outcome.
This data record supplies no permission when its numerical gates reject. -/
def contractionEntry : Spec.RuleEntry ComputeStmt := FP.DotAdmission.candidate

theorem contractionEntry_source :
    contractionEntry.rule.lhs = separateFragment ∧
    contractionEntry.rule.rhs = contractedFragment := ⟨rfl, rfl⟩

/-- Evidence remains an explicit conditional parameter. The recorded
experiment does not manufacture this proof; any selected evidence must discharge
its own gates and provenance before constructing this value. -/
structure Rules where
  evidence : Spec.Evidence
  accepted : Spec.AcceptedAtom (⟨contractionEntry.rule,evidence⟩ : Spec.RuleEntry ComputeStmt)

def Rules.entry (R : Rules) : Spec.RuleEntry ComputeStmt := ⟨contractionEntry.rule,R.evidence⟩
def Rules.assumptions (R : Rules) : Spec.Assumptions ComputeStmt := [R.entry]
instance : CoeOut Rules (Spec.Assumptions ComputeStmt) := ⟨Rules.assumptions⟩

/-- This actual local atom is interpreted to obtain the dot contraction law
used in the complete source proof. It is not an unused audit witness. -/
theorem admitted (R : Rules) :
    Spec.Derivation R.assumptions separateFragment contractedFragment :=
  Spec.Derivation.atom R.entry (by simp [Rules.assumptions]) R.accepted

#print_fp_assumptions admitted

/-- Interpretation of the registered local typed instruction rewrite. The
model says nothing about successful kernels, safety, frames or final outputs. -/
def Models (R : Spec.Assumptions ComputeStmt) (M : Algebra Pair.Word)
    (D : FPValue.Dot Pair.Word) (F : FPFused.DotAcc Pair.Word) : Prop :=
  Spec.Derivation R separateFragment contractedFragment → Pair.DotAddContract M D F

local instance : Inhabited Pair.Word := ⟨FP.Value.zero .fp32⟩

def IOEquivalent {p : KernelInterface.Parameters} {order : Schedule p.request}
    (R : Spec.Assumptions ComputeStmt) (left right : IOProgram p order) : Prop :=
  ∀ (cfg : FP.Config) (rest : Algebra Pair.Word) (D : FPValue.Dot Pair.Word)
    (F : FPFused.DotAcc Pair.Word) (Q : Whole.Numerics Pair.Word),
    Whole.PrimitiveLaws (Pair.fp32Algebra cfg rest) D Q → F.le = D.le →
    Models R (Pair.fp32Algebra cfg rest) D F →
    (programIO cfg rest D F left).Equivalent (variant cfg rest D F left right)

instance {p : KernelInterface.Parameters} {order : Schedule p.request} :
    Spec.ProgramSyntax (IOProgram p order) where
  Statement := ComputeStmt
  Signature := Unit
  signature := fun _ => ()
  body := fun program => program.kernel.surfaceBody
  structural := some (fun _ _ => False)
  numerical := IOEquivalent
  sameContext := Eq

theorem io_equivalent (R : Rules) (p : KernelInterface.Parameters) (order : Schedule p.request)
    (separateA : p.C ≠ p.A) (separateB : p.C ≠ p.B) :
    IOEquivalent R.assumptions (original p order) (contracted p order) := by
  intro cfg rest D F Q ports comparison model
  have localContract := model (admitted R)
  exact Pair.equivalent (Pair.fp32Algebra cfg rest) D F Q ports localContract comparison
    (Pair.fp32_mul_commutes cfg rest) p order separateA separateB

#axiomsClean io_equivalent
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.SourcePair
