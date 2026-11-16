import bench.tritonbench_g.bgmv_expand_slice.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Pasted
noncomputable section
/-- Complete source equality for both native signed metadata widths, after
its explicit int64 widening and before all execution or numerical premises. -/
theorem original_metadata_widths : _bgmv_expand_slice_kernel .i32 = _bgmv_expand_slice_kernel .i64 := by rfl

theorem candidate_metadata_widths : _bgmv_expand_slice_kernel_candidateKernel .i32 = _bgmv_expand_slice_kernel_candidateKernel .i64 := by rfl

#axiomsClean original_metadata_widths
#axiomsClean candidate_metadata_widths
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Pasted
