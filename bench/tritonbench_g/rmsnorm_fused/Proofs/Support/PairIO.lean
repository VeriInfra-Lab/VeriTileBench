import bench.tritonbench_g.rmsnorm_fused.Kernels
import VeriTile.Triton.Float.ProfiledRewrite

namespace VeriTile.Bench.Repairs.RmsnormFused.SourcePair
open VeriTile Triton FP.ProfiledRewrite
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (X Y W : RegionName) (N B : Nat) : Program := ⟨rms_norm_fwd_fusedKernel X Y W N B, []⟩
def addressCommuted (X Y W : RegionName) (N B : Nat) : Program :=
  ⟨rms_norm_fwd_fused_index_commutedKernel X Y W N B, []⟩

end
end VeriTile.Bench.Repairs.RmsnormFused.SourcePair
