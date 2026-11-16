import VeriTile.Triton.Source
import bench.tritonbench_g.context_attn_fwd.Proofs.Sources.Selected

/-! Storage compatibility views instantiate the same copied bodies. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd
open VeriTile.Triton
noncomputable section
def original (storage : Storage) := match storage with
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnFwd._fwd_kernel_int8kvKernel .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnFwd._fwd_kernel_int8kvKernel .bf16
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource.original
def scaleFirst (storage : Storage) := match storage with
  | .fp16 => VeriTile.Bench.Repairs.ContextAttnFwd._fwd_kernel_scale_firstKernel .fp16
  | .bf16 => VeriTile.Bench.Repairs.ContextAttnFwd._fwd_kernel_scale_firstKernel .bf16
  | .fp32 => VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource.scaleFirst
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd
