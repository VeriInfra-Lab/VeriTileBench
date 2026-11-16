import bench.tritonbench_g.decay_cumsum.Proofs.Sources.Selected

/-! Typed invocation profiles of the same pasted source, including the
pre-existing fp16/bf16 mathematical APIs. No whole-kernel generator is used. -/
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
open VeriTile.Triton
noncomputable section

def forward (storage : Storage) (swapped : Bool) :=
  if swapped then VeriTile.Bench.Repairs.DecayCumsum.fwd_decay_cumsum_index_commutedKernel storage
  else VeriTile.Bench.Repairs.DecayCumsum.fwd_decay_cumsumKernel storage

def prepare (source gate : Storage) (swapped : Bool) :=
  if swapped then VeriTile.Bench.Repairs.DecayCumsum.prepare_qg_kg_index_commutedKernel source gate
  else VeriTile.Bench.Repairs.DecayCumsum.prepare_qg_kgKernel source gate

def backward (source gate : Storage) (swapped : Bool) :=
  if swapped then VeriTile.Bench.Repairs.DecayCumsum.bwd_decay_global_cumsum_index_commutedKernel source gate
  else VeriTile.Bench.Repairs.DecayCumsum.bwd_decay_global_cumsumKernel source gate

end
end VeriTile.Bench.TritonBenchG.DecayCumsum.TypedSource
