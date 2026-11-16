import bench.tritonbench_g.rmsnorm_fused_llama.Kernels

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
noncomputable section
open VeriTile.Triton
open VeriTile.Bench.Repairs.RmsnormFusedLlama

def typedKernel (v : Bool) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) : ComputeKernel :=
  (if v then _rms_norm_fwd_fused_index_commutedKernel X Y W N B stride else _rms_norm_fwd_fusedKernel X Y W N B stride).withScalarArguments
    [ScalarArgument.fp32 "eps" eps]

end
end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
