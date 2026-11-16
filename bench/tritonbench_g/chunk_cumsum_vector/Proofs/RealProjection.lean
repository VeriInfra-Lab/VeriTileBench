import bench.tritonbench_g.chunk_cumsum_vector.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ChunkCumsumVector.PublicRealIO
open VeriTile.Triton

theorem sourceProjection (h : Call) (c : Config) :
    (Kernels.chunk_global_cumsum_vector_kernel h c).eraseDType.toAlgorithm? =
      Except.ok (Kernels.chunk_global_cumsum_vector_kernel h c).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, Kernels.source_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

end VeriTile.Bench.Repairs.ChunkCumsumVector.PublicRealIO
