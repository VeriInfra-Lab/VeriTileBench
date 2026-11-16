import VeriTile.Triton.Float.Erasure
import bench.tritonbench_g.chunk_cumsum_kernel.Proofs.FPIndexBridge

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicReal
open VeriTile.Triton
noncomputable section

theorem projection (input output : RegionName) (T BT : Nat) :
    (chunk_global_cumsum_scalar_kernel_repairedKernel .fp32 .fp32 input output T BT).eraseDType.toAlgorithm? =
      .ok (chunk_global_cumsum_scalar_kernel_repairedKernel .fp32 .fp32 input output T BT).eraseDType.toAlgKernel := by
  simp [chunk_global_cumsum_scalar_kernel_repairedKernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,Except.bind,bind]

end
end VeriTile.Bench.Repairs.ChunkCumsumKernel.PublicReal
