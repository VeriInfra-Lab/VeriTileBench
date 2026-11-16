import bench.tritonbench_g.index_select_cat.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.Sources.Mathematical_IndexRewrite

namespace VeriTile.Bench.Repairs.IndexSelectCat.Rewrite
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (p : Layout) : Program where
  kernel := index_select_cat_fwd_kernel_repairedKernel p
  domain := []
def indexCommuted (p : Layout) : Program where
  kernel := index_select_cat_fwd_kernel_index_commutedKernel p
  domain := []

theorem equivalent (p : Layout) : original p ≡[assumptions] indexCommuted p := by
  split_equiv <;> solve_fp

/-- Exact integer-prefix execution equality composes with the unchanged
remainder, so it also transports successful complete execution. -/
theorem execution_eq {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (p : Layout) (s : FP.Structural.State α) :
    FP.Structural.exec M (index_select_cat_fwd_kernel_repairedKernel p) s = FP.Structural.exec M (index_select_cat_fwd_kernel_index_commutedKernel p) s := by
  have firstSteps : FP.Structural.run M ((index_select_cat_fwd_kernel_repairedKernel p).surfaceBody.take 3) s =
      FP.Structural.run M ((index_select_cat_fwd_kernel_index_commutedKernel p).surfaceBody.take 3) s := by
    simp [index_select_cat_fwd_kernel_repairedKernel,index_select_cat_fwd_kernel_index_commutedKernel,ComputeKernel.surfaceBody,FP.Structural.run,
      FP.Structural.step,FP.Structural.evalExpr,FP.Structural.evalOp_unfold,
      FP.Structural.numeric,ComputeDType.eraseDType,Option.bind]
    congr 1
    funext i
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm,Int.mul_comm]
  change FP.Structural.run M (index_select_cat_fwd_kernel_repairedKernel p).surfaceBody s =
    FP.Structural.run M (index_select_cat_fwd_kernel_index_commutedKernel p).surfaceBody s
  rw [←List.take_append_drop 3 (index_select_cat_fwd_kernel_repairedKernel p).surfaceBody,
    ←List.take_append_drop 3 (index_select_cat_fwd_kernel_index_commutedKernel p).surfaceBody]
  rw [FP.Structural.run_append,FP.Structural.run_append,firstSteps]
  rfl

#print_fp_assumptions equivalent
#axiomsClean equivalent
#axiomsClean execution_eq
end VeriTile.Bench.Repairs.IndexSelectCat.Rewrite
