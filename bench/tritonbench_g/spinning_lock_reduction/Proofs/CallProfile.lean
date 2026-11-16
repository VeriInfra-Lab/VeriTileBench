import bench.tritonbench_g.spinning_lock_reduction.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.SpinningLock.CallProfile
open VeriTile.Triton

theorem original_supported : SourceAdapter.profileSupported spinning_lock_kernelSource=true := rfl
theorem candidate_supported : SourceAdapter.profileSupported spinning_lock_kernel_areaSource=true := rfl

theorem changed_flag_rejected :
    SourceAdapter.decode {spinning_lock_kernelSource with constexprBools := [("P_NEG",true),("C_NEG",false),("LOCK_NEG",false)]} =
      .error "unsupported spinning caller ABI or constexpr binding" := rfl

theorem missing_flag_rejected :
    SourceAdapter.decode {spinning_lock_kernelSource with constexprBools := [("P_NEG",false),("C_NEG",false)]} =
      .error "unsupported spinning caller ABI or constexpr binding" := rfl

theorem changed_pointer_rejected :
    SourceAdapter.decode {spinning_lock_kernelSource with signature := {spinning_lock_kernelSource.signature with
      pointers := [("P",.fp32),("C",.fp32),("locks",.fp32)]}} =
      .error "unsupported spinning caller ABI or constexpr binding" := rfl

#axiomsClean original_supported
#axiomsClean candidate_supported
#axiomsClean changed_flag_rejected
#axiomsClean missing_flag_rejected
#axiomsClean changed_pointer_rejected
end VeriTile.Bench.Repairs.SpinningLock.CallProfile
