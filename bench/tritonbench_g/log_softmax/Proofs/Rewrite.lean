import bench.tritonbench_g.log_softmax.Proofs.Kernel
import bench.tritonbench_g.log_softmax.Proofs.Mathematical_FP

namespace VeriTile.Bench.Repairs.LogSoftmax.Rewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LogSoftmax
set_option maxRecDepth 20000

theorem forward_profiled [Inhabited A] (algebra : Algebra A) (O X : RegionName)
    (M N K BM C : Nat) (s : State A) :
    FP.Structural.exec (engine algebra) (Source.log_softmax_kernel_repairedKernel O X M N K BM C) s =
      FP.Structural.exec (engine algebra) (Source.log_softmax_kernel_repaired_index_commutedKernel O X M N K BM C) s := by
  have firstSteps : run (engine algebra) ((Source.log_softmax_kernel_repairedKernel O X M N K BM C).surfaceBody.take 3) s =
      run (engine algebra) ((Source.log_softmax_kernel_repaired_index_commutedKernel O X M N K BM C).surfaceBody.take 3) s := by
    simp [Source.log_softmax_kernel_repairedKernel,Source.log_softmax_kernel_repaired_index_commutedKernel,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalOp_unfold,numeric,ComputeDType.eraseDType,Option.bind]
    congr 1
    funext i
    simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm,Int.mul_comm]
  change run (engine algebra) (Source.log_softmax_kernel_repairedKernel O X M N K BM C).surfaceBody s =
    run (engine algebra) (Source.log_softmax_kernel_repaired_index_commutedKernel O X M N K BM C).surfaceBody s
  rw [←List.take_append_drop 3 (Source.log_softmax_kernel_repairedKernel O X M N K BM C).surfaceBody,
    ←List.take_append_drop 3 (Source.log_softmax_kernel_repaired_index_commutedKernel O X M N K BM C).surfaceBody]
  rw [run_append,run_append,firstSteps]
  rfl


theorem backward_profiled [Inhabited A] (algebra : Algebra A) (Y G O : RegionName)
    (M N K BM C : Nat) (s : State A) :
    FP.Structural.exec (engine algebra) (Source.log_softmax_backward_kernel_repairedKernel Y G O M N K BM C) s =
      FP.Structural.exec (engine algebra) (Source.log_softmax_backward_kernel_repaired_index_commutedKernel Y G O M N K BM C) s := by
  have firstSteps : run (engine algebra) ((Source.log_softmax_backward_kernel_repairedKernel Y G O M N K BM C).surfaceBody.take 3) s =
      run (engine algebra) ((Source.log_softmax_backward_kernel_repaired_index_commutedKernel Y G O M N K BM C).surfaceBody.take 3) s := by
    simp [Source.log_softmax_backward_kernel_repairedKernel,Source.log_softmax_backward_kernel_repaired_index_commutedKernel,ComputeKernel.surfaceBody,run,step,
      evalExpr,evalOp_unfold,numeric,ComputeDType.eraseDType,Option.bind]
    congr 1
    funext i
    simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm,Int.mul_comm]
  change run (engine algebra) (Source.log_softmax_backward_kernel_repairedKernel Y G O M N K BM C).surfaceBody s =
    run (engine algebra) (Source.log_softmax_backward_kernel_repaired_index_commutedKernel Y G O M N K BM C).surfaceBody s
  rw [←List.take_append_drop 3 (Source.log_softmax_backward_kernel_repairedKernel Y G O M N K BM C).surfaceBody,
    ←List.take_append_drop 3 (Source.log_softmax_backward_kernel_repaired_index_commutedKernel Y G O M N K BM C).surfaceBody]
  rw [run_append,run_append,firstSteps]
  rfl


theorem engine_default (e : Kernel.Engine A C) : engine e.algebra=e.algebra := rfl

theorem forward_same_run [Inhabited A] (e : Kernel.Engine A C) (O X : RegionName)
    (M N K BM : Nat) (s : State A) :
    FP.Structural.exec e.algebra (Source.log_softmax_kernel_repairedKernel O X M N K BM C) s =
      FP.Structural.exec e.algebra (Source.log_softmax_kernel_repaired_index_commutedKernel O X M N K BM C) s := by
  simpa only [engine_default] using forward_profiled e.algebra O X M N K BM C s

theorem backward_same_run [Inhabited A] (e : Kernel.Engine A C) (Y G O : RegionName)
    (M N K BM : Nat) (s : State A) :
    FP.Structural.exec e.algebra (Source.log_softmax_backward_kernel_repairedKernel Y G O M N K BM C) s =
      FP.Structural.exec e.algebra (Source.log_softmax_backward_kernel_repaired_index_commutedKernel Y G O M N K BM C) s := by
  simpa only [engine_default] using backward_profiled e.algebra Y G O M N K BM C s

#axiomsClean forward_same_run
#axiomsClean backward_same_run
end VeriTile.Bench.Repairs.LogSoftmax.Rewrite
