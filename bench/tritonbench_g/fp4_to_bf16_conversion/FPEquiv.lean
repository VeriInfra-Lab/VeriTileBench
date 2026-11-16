import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.Fp4Scaled.FPEquiv
open VeriTile Triton FP.Structural ScheduleIO
open scoped VeriTile.Spec
noncomputable section
local instance : Inhabited FPModel.Word := ⟨FP.Value.zero .fp32⟩

/-- Both complete valid CTA orders execute the actual repaired source safely,
produce the same BF16 cells, and preserve the complete exterior frame. The FP
model retains the source casts, NaNs, rounding, underflow and signed zeros;
no floating arithmetic is rearranged and no numerical assumption is admitted. -/
specification fp4_to_bf16_conversion_schedule_equiv (p : Parameters) (a b : List Nat)
    (launchable : Launchable p) (fresh : p.input≠p.output ∧ p.scales≠p.output)
    (left : a.Perm (List.range (grid p))) (right : b.Perm (List.range (grid p))) :
    original p a ≡[assumptions] candidate p b := by
  apply Spec.FloatingPoint.ofStructural (structural := ScheduleIO.Equivalent)
    (lhs := original p a) (rhs := candidate p b) rfl rfl
  intro rest
  apply ProgramIO.EquivalentOn.of_implementsOn
    (fun xs _=>implements rest p a fresh left xs trivial)
  exact fun xs _=>implements rest p b fresh right xs trivial

#print_fp_assumptions fp4_to_bf16_conversion_schedule_equiv
#axiomsClean fp4_to_bf16_conversion_schedule_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.Fp4Scaled.FPEquiv
