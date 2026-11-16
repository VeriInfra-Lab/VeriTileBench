import bench.tritonbench_g.lightning_attention.Kernels

namespace VeriTile.Tests.LightningAttentionPastedBody
open VeriTile.Triton
open VeriTile.Bench.Repairs.LightningAttention
open VeriTile.Bench.Repairs.LightningAttention.KernelOnly

/-- Preserve the original ordered ports, including the repeated inter reads.
The source collector's duplicates are meaningful metadata and stay intact. -/
example (p : Parameters) : (_fwd_kernel p).inputs=["query","key","value"] ∧
    (_fwd_kernel p).outputs=["output"] := ⟨rfl,rfl⟩
example (p : Parameters) : (_bwd_intra_kernel p).inputs=["key","value","incoming","query"] ∧
    (_bwd_intra_kernel p).outputs=["queryGradient","keyGradient","valueGradient"] := ⟨rfl,rfl⟩
example (p : Parameters) : (_bwd_inter_kernel p).inputs=["incoming","queryGradient","value","key","key","value",
    "keyGradient","valueGradient","incoming","query"] ∧
    (_bwd_inter_kernel p).outputs=["queryGradient","keyGradient","valueGradient"] := ⟨rfl,rfl⟩

end VeriTile.Tests.LightningAttentionPastedBody
