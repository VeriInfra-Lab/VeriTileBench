import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Scale
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Layout

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxRecDepth 40000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
attribute [local irreducible] decodeExecution
attribute [local simp] decodeExecution_pid decodeExecution_n_elements_out decodeExecution_n_elements_s
  decodeExecution_mem

private theorem mul_8 (a : Nat) : (a : Int)*8 = ((a*8 : Nat) : Int) := by omega
private theorem mul_16 (a : Nat) : (a : Int)*16 = ((a*16 : Nat) : Int) := by omega
private theorem mul_32 (a : Nat) : (a : Int)*32 = ((a*32 : Nat) : Int) := by omega
private theorem mul_64 (a : Nat) : (a : Int)*64 = ((a*64 : Nat) : Int) := by omega
private theorem mul_128 (a : Nat) : (a : Int)*128 = ((a*128 : Nat) : Int) := by omega

def scaleLoadExecution (rest : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest) ((source p).surfaceBody.drop 31 |>.take 4)
      (decodeExecution rest p s).val = some t} := by
  have houtput := decodeExecution_output rest p s
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp only [Parameters.block,Parameters.groups,Kernels.Tile.block] at houtput
    refine ⟨?_,?_⟩
    swap
    dsimp only [source,Kernels.triton_f4_to_scaled_bf16_kernel,ComputeKernel.surfaceBody,List.drop,List.take]
    simp [Parameters.block,Parameters.groups,Kernels.Tile.block,houtput,
      run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,backend,
      numeric,natLt,numericLt,integralFloorDiv,bop,Broadcast.leftIndex,Broadcast.rightIndex,
      ofFloat,toFloat,Region.cast,Option.bind,ComputeDType.eraseDType,State.setReg]
    try rfl
  )

@[local simp] theorem scaleLoadExecution_s (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleLoadExecution rest p s).val.regs .nat [p.groups] "s" = some (fun j => loadedScale p s j.1.val) := by
  have houtput := decodeExecution_output rest p s
  rcases p with ⟨input,scales,output,count,tile⟩
  cases input
  cases scales
  cases output
  cases tile <;> (
    simp only [Parameters.block,Parameters.groups,Kernels.Tile.block] at houtput
    simp [scaleLoadExecution,Parameters.block,Parameters.groups,Kernels.Tile.block,
      houtput,loadedScale,scaleOffset,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
    all_goals simp only [mul_8,mul_16,mul_32,mul_64,mul_128,← Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
    all_goals (funext j; with_unfolding_all rfl)
  )

@[local simp] theorem scaleLoadExecution_output (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleLoadExecution rest p s).val.regs .bf16 [p.block*2] "output" = some (fun j => converted (laneNibble p s j)) := by
  have houtput := decodeExecution_output rest p s
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp only [Parameters.block,Parameters.groups,Kernels.Tile.block] at houtput
    simp [scaleLoadExecution,Parameters.block,Parameters.groups,Kernels.Tile.block,
      houtput,loadedScale,scaleOffset,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
    all_goals simp only [mul_8,mul_16,mul_32,mul_64,mul_128,← Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
    try rfl
  )

@[local simp] theorem scaleLoadExecution_pid (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleLoadExecution rest p s).val.regs .int [] "pid" = some (fun _ => (s.pids 0 : Int)) := by
  have houtput := decodeExecution_output rest p s
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp only [Parameters.block,Parameters.groups,Kernels.Tile.block] at houtput
    simp [scaleLoadExecution,Parameters.block,Parameters.groups,Kernels.Tile.block,
      houtput,loadedScale,scaleOffset,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
    all_goals simp only [mul_8,mul_16,mul_32,mul_64,mul_128,← Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
    try rfl
  )

@[local simp] theorem scaleLoadExecution_n_elements_out (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleLoadExecution rest p s).val.regs .int [] "n_elements_out" = some (fun _ => ((p.count*2 : Nat) : Int)) := by
  have houtput := decodeExecution_output rest p s
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp only [Parameters.block,Parameters.groups,Kernels.Tile.block] at houtput
    simp [scaleLoadExecution,Parameters.block,Parameters.groups,Kernels.Tile.block,
      houtput,loadedScale,scaleOffset,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
    all_goals simp only [mul_8,mul_16,mul_32,mul_64,mul_128,← Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
    try rfl
  )

@[local simp] theorem scaleLoadExecution_n_elements_s (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleLoadExecution rest p s).val.regs .int [] "n_elements_s" = some (fun _ => (scaleCount p : Int)) := by
  have houtput := decodeExecution_output rest p s
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp only [Parameters.block,Parameters.groups,Kernels.Tile.block] at houtput
    simp [scaleLoadExecution,Parameters.block,Parameters.groups,Kernels.Tile.block,
      houtput,loadedScale,scaleOffset,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
    all_goals simp only [mul_8,mul_16,mul_32,mul_64,mul_128,← Int.natCast_add,Int.toNat_natCast,Int.ofNat_lt]
    try rfl
  )

@[local simp] theorem scaleLoadExecution_mem (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleLoadExecution rest p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest) _ _ _
    (by rcases p with ⟨input,scales,output,count,tile⟩; cases tile <;> rfl)
    (scaleLoadExecution rest p s).property).trans (decodeExecution_mem rest p s)

#axiomsClean scaleLoadExecution
end
end VeriTile.Bench.Repairs.Fp4Scaled
