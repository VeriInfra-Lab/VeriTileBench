import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Real

namespace VeriTile.Bench.Repairs.Fp4Scaled.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete repaired scaled conversion in Real semantics, with rounding
erased: high/low E2M1 decoding followed by the corresponding E8M0 power of two.
Scale byte 255 is excluded only here because NaN has no Real value. Safe
execution, every output and the full exterior frame are conclusions. -/
specification fp4_to_bf16_conversion_real (rest : Algebra ℝ) (p : Parameters) (jobs : List Nat)
    (launchable : Launchable p) (fresh : p.input≠p.output ∧ p.scales≠p.output) (order : jobs.Perm (List.range (grid p))) :
    VeriTile.Spec.Real (logicalIO rest p jobs ⊨{RealDomain p} fun xs j=>
      RealSpec.e2m1 (nibble p xs j)*(2 : ℝ)^((scaleByte p xs j : Int)-127)) :=
  RealModel.implements_real rest p jobs fresh order

#specNonCircular RealSpec.e2m1 avoiding [source,Kernels.triton_f4_to_scaled_bf16_kernel,FP.Structural.exec]
#axiomsClean fp4_to_bf16_conversion_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Fp4Scaled.Correct
