import bench.tritonbench_g.matmul_leakyrelu.Proofs.Unconditional
import VeriTile.Meta.Disprove

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.FPEquiv
open VeriTile.Triton FP.Structural
open SourcePair
open scoped VeriTile.Spec
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 6000000

/-- The universal source-model claim under the current FP32 admission table
(which is empty) has a concrete
countermodel, including legal input bounds, safe complete source schedules and
different final binary16 values. The software interpretation is explicit in
`Unconditional`; this conclusion carries no GPU-conformance assertion. -/
refutation source_pair_disproved :
    ∀ (p : KernelInterface.Parameters) (order : Schedule p.request),
      p.C ≠ p.A → p.C ≠ p.B →
      original p order ≡[FP.DotAdmission.all] contracted p order := by
  intro claim
  exact Unconditional.unconditional_source_pair_disproved
    (by simpa only [FP.DotAdmission.all] using
      claim Unconditional.parameters Unconditional.order (by decide) (by decide))

#axiomsClean source_pair_disproved

/-- Actual repaired separate dot/add source versus the live-accumulator dot
candidate. The local TF32/FP32 contraction rule retains its explicit numerical
premise; strict FP32 slope commutativity reuses the shared software theorem.
The conditional conclusion includes both safe successful complete schedules,
all FP16-tagged output cells and both exact exterior frames. -/
specification source_pair_equiv (R : Rules) (p : KernelInterface.Parameters) (order : Schedule p.request)
    (separateA : p.C ≠ p.A) (separateB : p.C ≠ p.B) :
    original p order ≡[R] contracted p order := by
  apply Spec.FloatingPoint.ofNumerical
  · rfl
  · rfl
  exact io_equivalent R p order separateA separateB

#axiomsClean source_pair_equiv
#print_fp_assumptions source_pair_equiv
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.FPEquiv
