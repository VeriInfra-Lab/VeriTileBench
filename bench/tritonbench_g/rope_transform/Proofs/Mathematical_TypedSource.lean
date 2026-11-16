import bench.tritonbench_g.rope_transform.Kernels

/-! All storage profiles and both rotation directions use the selected source. -/
namespace VeriTile.Bench.TritonBenchG.RopeTransform.Typed

abbrev source := VeriTile.Bench.Repairs.RopeTransform.Kernels._triton_ropeKernel

theorem source_selected : source=VeriTile.Bench.Repairs.RopeTransform.Kernels._triton_ropeKernel := rfl

end VeriTile.Bench.TritonBenchG.RopeTransform.Typed
