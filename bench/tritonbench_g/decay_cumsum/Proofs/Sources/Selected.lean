import bench.tritonbench_g.decay_cumsum.Kernels

/-! Compatibility entries for the single pasted Python source and its existing
mask-index candidates. Every invocation uses the same complete function body. -/
namespace VeriTile.Bench.Repairs.DecayCumsum.Kernels
open VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
noncomputable section

abbrev quotedForward :=
  fwd_decay_cumsumKernel Storage.fp32

abbrev quotedForwardIndexCommuted :=
  fwd_decay_cumsum_index_commutedKernel Storage.fp32

abbrev quotedPrepare :=
  prepare_qg_kgKernel Storage.fp32 Storage.fp32

abbrev quotedPrepareIndexCommuted :=
  prepare_qg_kg_index_commutedKernel Storage.fp32 Storage.fp32

abbrev quotedBackward :=
  bwd_decay_global_cumsumKernel Storage.fp32 Storage.fp32

abbrev quotedBackwardIndexCommuted :=
  bwd_decay_global_cumsum_index_commutedKernel Storage.fp32 Storage.fp32

end
end VeriTile.Bench.Repairs.DecayCumsum.Kernels
