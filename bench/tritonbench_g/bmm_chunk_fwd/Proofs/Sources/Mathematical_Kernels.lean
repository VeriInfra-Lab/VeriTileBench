import bench.tritonbench_g.bmm_chunk_fwd.Proofs.Sources.Selected
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.TypedSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.BmmChunkFwd
def specialize (a b : Storage) (swapped : Bool) :=
  match a, b, swapped with
  | .fp32, .fp32, Bool.false => VeriTile.Bench.Repairs.BmmChunkFwd.Kernels.mainSource
  | .fp32, .fp32, Bool.true => (fun out_dtype => _bmm_chunk_fwd_kernel_repaired_dot_firstKernel .fp32 .fp32 out_dtype .fp32)
  | .fp32, .fp16, Bool.false => (fun out_dtype => _bmm_chunk_fwd_kernel_repairedKernel .fp32 .fp16 out_dtype .fp16)
  | .fp32, .fp16, Bool.true => (fun out_dtype => _bmm_chunk_fwd_kernel_repaired_dot_firstKernel .fp32 .fp16 out_dtype .fp16)
  | .fp32, .bf16, Bool.false => (fun out_dtype => _bmm_chunk_fwd_kernel_repairedKernel .fp32 .bf16 out_dtype .bf16)
  | .fp32, .bf16, Bool.true => (fun out_dtype => _bmm_chunk_fwd_kernel_repaired_dot_firstKernel .fp32 .bf16 out_dtype .bf16)
  | .fp16, .fp32, Bool.false => (fun out_dtype => _bmm_chunk_fwd_kernel_repairedKernel .fp16 .fp32 out_dtype .fp16)
  | .fp16, .fp32, Bool.true => (fun out_dtype => _bmm_chunk_fwd_kernel_repaired_dot_firstKernel .fp16 .fp32 out_dtype .fp16)
  | .fp16, .fp16, Bool.false => (fun out_dtype => _bmm_chunk_fwd_kernel_repairedKernel .fp16 .fp16 out_dtype .fp16)
  | .fp16, .fp16, Bool.true => (fun out_dtype => _bmm_chunk_fwd_kernel_repaired_dot_firstKernel .fp16 .fp16 out_dtype .fp16)
  | .fp16, .bf16, Bool.false => (fun out_dtype => _bmm_chunk_fwd_kernel_repairedKernel .fp16 .bf16 out_dtype .bf16)
  | .fp16, .bf16, Bool.true => (fun out_dtype => _bmm_chunk_fwd_kernel_repaired_dot_firstKernel .fp16 .bf16 out_dtype .bf16)
  | .bf16, .fp32, Bool.false => (fun out_dtype => _bmm_chunk_fwd_kernel_repairedKernel .bf16 .fp32 out_dtype .bf16)
  | .bf16, .fp32, Bool.true => (fun out_dtype => _bmm_chunk_fwd_kernel_repaired_dot_firstKernel .bf16 .fp32 out_dtype .bf16)
  | .bf16, .fp16, Bool.false => (fun out_dtype => _bmm_chunk_fwd_kernel_repairedKernel .bf16 .fp16 out_dtype .bf16)
  | .bf16, .fp16, Bool.true => (fun out_dtype => _bmm_chunk_fwd_kernel_repaired_dot_firstKernel .bf16 .fp16 out_dtype .bf16)
  | .bf16, .bf16, Bool.false => (fun out_dtype => _bmm_chunk_fwd_kernel_repairedKernel .bf16 .bf16 out_dtype .bf16)
  | .bf16, .bf16, Bool.true => (fun out_dtype => _bmm_chunk_fwd_kernel_repaired_dot_firstKernel .bf16 .bf16 out_dtype .bf16)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.TypedSource
