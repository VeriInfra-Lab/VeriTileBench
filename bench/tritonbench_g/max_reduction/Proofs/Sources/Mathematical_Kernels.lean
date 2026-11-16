import bench.tritonbench_g.max_reduction.Kernels

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
abbrev ValueSource := RegionName → RegionName → Nat → Nat → ComputeKernel

def stage1 (candidate : Bool) : ValueSource :=
  if candidate then VeriTile.Bench.Repairs.MaxReduction.max_stage1_commutedKernel
  else VeriTile.Bench.Repairs.MaxReduction.max_kernel_1Kernel

def stage2 (candidate : Bool) : ValueSource :=
  if candidate then VeriTile.Bench.Repairs.MaxReduction.max_stage2_complementKernel
  else VeriTile.Bench.Repairs.MaxReduction.max_kernel_2Kernel

def dimensionSource (candidate : Bool) :=
  if candidate then VeriTile.Bench.Repairs.MaxReduction.max_dimension_commutedKernel
  else VeriTile.Bench.Repairs.MaxReduction.max_kernel

/-- Mathematical view derived from the shared repair quotation. -/
noncomputable def max_kernel_1
    (inp mid : RegionName)
    (M BLOCK_SIZE : Nat) :
    ComputeKernel :=
  let k := (stage1 Bool.false inp mid M BLOCK_SIZE).eraseDType.toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

/-- Mathematical view derived from the shared repair quotation. -/
noncomputable def max_kernel_2
    (mid out : RegionName)
    (mid_size BLOCK_MID : Nat) :
    ComputeKernel :=
  let k := (stage2 Bool.false mid out mid_size BLOCK_MID).eraseDType.toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end VeriTile.Bench.TritonBenchG.MaxReduction
