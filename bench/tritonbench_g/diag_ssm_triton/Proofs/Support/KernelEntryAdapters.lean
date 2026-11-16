import bench.tritonbench_g.diag_ssm_triton.Kernels

/-! Shared entries route to the repaired Python function bodies in
Sources/Body.lean. Every cast, range, branch, mask, tuple update and store
stays in the pasted body. Dtype declarations and runtime argument bindings
remain outside the quotation. These kernels do not depend on proof modules. -/
namespace VeriTile.Bench.Repairs.DiagSsmTriton.Kernels
open VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

def blockSize : Nat := 128

abbrev forwardTyped :=
  diag_ssm_forward_kernel Storage.fp32

abbrev forwardIndexCommuted :=
  diag_ssm_forward_kernel_index_commutedKernel Storage.fp32

abbrev backwardTyped :=
  diag_ssm_backward_kernel Storage.fp32

abbrev backwardIndexCommuted :=
  diag_ssm_backward_kernel_index_commutedKernel Storage.fp32

abbrev complexForwardTyped :=
  diag_ssm_forward_kernel_complexKernel

abbrev complexForwardIndexCommuted :=
  diag_ssm_forward_kernel_complex_index_commutedKernel

abbrev complexBackwardTyped :=
  diag_ssm_backward_kernel_complexKernel

abbrev complexBackwardIndexCommuted :=
  diag_ssm_backward_kernel_complex_index_commutedKernel

end VeriTile.Bench.Repairs.DiagSsmTriton.Kernels
