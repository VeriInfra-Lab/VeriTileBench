import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Model

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton

/-- Logical packed input and E8M0 scales. Each scale covers 32 decoded values;
the final partial group still has a scale. -/
def scaleLength (p : Parameters) := (p.count*2+31)/32

def grid (p : Parameters) := (p.count+p.block-1)/p.block

structure Inputs (p : Parameters) where
  packed : Fin p.count → Nat
  scales : Fin (scaleLength p) → Nat

/-- Physical uint8 storage, including E8M0 NaN encoding 255. -/
def ByteDomain (p : Parameters) (xs : Inputs p) : Prop :=
  (∀ i, xs.packed i<256) ∧ (∀ i, xs.scales i<256)

/-- The Real projection has no NaN value. FP equivalence retains byte 255. -/
def RealDomain (p : Parameters) (xs : Inputs p) : Prop :=
  (∀ i, xs.packed i<256) ∧ (∀ i, xs.scales i<255)

theorem block_positive (p : Parameters) : 0<p.block := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> norm_num [Parameters.block,Parameters.groups,Kernels.Tile.block]

theorem block_groups (p : Parameters) : p.block*2=p.groups*32 := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> norm_num [Parameters.block,Parameters.groups,Kernels.Tile.block]

/-- Ordinary signed host allocation size and CUDA one-dimensional grid bounds. -/
def Launchable (p : Parameters) : Prop :=
  p.count*4<2^63 ∧ grid p≤2147483647

/-- On a valid CUDA launch, the copied signed index casts cannot overflow. -/
theorem launch_abi (p : Parameters) (valid : Launchable p) (pid : Nat)
    (live : pid<grid p) (j : Fin (p.block*2)) :
    p.count*2+31<2^63 ∧ pid<2^31 ∧ pid*(p.block*2)+j.val<2^63 := by
  have size:=valid.1
  have pidBound : pid<2147483647 := lt_of_lt_of_le live valid.2
  have laneBound:=j.isLt
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> norm_num [Parameters.block,Parameters.count,Kernels.Tile.block] at size laneBound ⊢ <;> omega

end VeriTile.Bench.Repairs.Fp4Scaled
