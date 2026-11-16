import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Decode
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Registers

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxRecDepth 40000
set_option maxHeartbeats 2500000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

def scaleCount (p : Parameters) := (p.count*2+31)/32
def scaleOffset (p : Parameters) (s : State α) (i : Nat) := s.pids 0*p.groups+i

def loadedScale (p : Parameters) (s : State α) (i : Nat) : Nat :=
  if scaleOffset p s i < scaleCount p then (s.mem p.scales (scaleOffset p s i)).read .nat else 0

def scaleExponent (n : Nat) : Int :=
  (BitVec.ofInt 32 ((BitVec.ofNat 16 n).toInt - 127)).toInt

def scaleValue (n : Nat) : α :=
  if n ≠ 255 then Numerics.narrow (Numerics.pow2 (scaleExponent n))
  else Numerics.narrow (Numerics.bits ⟨0x7fc00000⟩)

attribute [local irreducible] signExecution normalExecution expExecution addBiasExecution maskExecution loadExecution

private theorem ceil_int (n : Nat) : ((n : Int)*2+31)/32 = (((n*2+31)/32 : Nat) : Int) := by omega

private def countExecution (rest : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest) ((source p).surfaceBody.take 4) s = some t} := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    refine ⟨?_,?_⟩
    swap
    simp [source,Kernels.triton_f4_to_scaled_bf16_kernel,ComputeKernel.surfaceBody,List.take,run,step,
      evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,numericLt,natLt,integralFloorDiv,
      bop,Broadcast.leftIndex,Broadcast.rightIndex,ComputeDType.eraseDType]
    try rfl
  )

private theorem countExecution_value (rest : Algebra α) (p : Parameters) (s : State α) :
    (countExecution rest p s).val.regs .int [] "n_elements_s" =
      some (fun _ => (scaleCount p : Int)) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> simp [countExecution,scaleCount,
    numeric,numericLt,natLt,integralFloorDiv,bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals (congr 1)

@[local simp] theorem loadExecution_n_elements_s (rest : Algebra α) (p : Parameters) (s : State α) :
    (loadExecution rest p s).val.regs .int [] "n_elements_s" =
      some (fun _ => (scaleCount p : Int)) := by
  have split : ((source p).surfaceBody.drop 0 |>.take 14) =
      ((source p).surfaceBody.take 4) ++ ((source p).surfaceBody.drop 4 |>.take 10) := by
    rcases p with ⟨input,scales,output,count,tile⟩; cases tile <;> rfl
  have ran : run (backend rest) ((source p).surfaceBody.drop 4 |>.take 10)
      (countExecution rest p s).val = some (loadExecution rest p s).val := by
    simpa only [split,run_append,(countExecution rest p s).property,Option.bind_some]
      using (loadExecution rest p s).property
  exact (assignment_run_register (backend rest) "n_elements_s" .int [] _ _ _
    (by rcases p with ⟨input,scales,output,count,tile⟩; cases tile <;> rfl) ran).trans
      (countExecution_value rest p s)

@[local simp] theorem signExecution_n_elements_s (rest : Algebra α) (p : Parameters) (s : State α) :
    (signExecution rest p s).val.regs .int [] "n_elements_s" =
      some (fun _ => (scaleCount p : Int)) := by
  have split : ((source p).surfaceBody.drop 14 |>.take 13) =
      ((source p).surfaceBody.drop 14 |>.take 3) ++
      ((source p).surfaceBody.drop 17 |>.take 1) ++
      ((source p).surfaceBody.drop 18 |>.take 1) ++
      ((source p).surfaceBody.drop 19 |>.take 5) ++
      ((source p).surfaceBody.drop 24 |>.take 3) := by
    rcases p with ⟨input,scales,output,count,tile⟩; cases tile <;> rfl
  have ran : run (backend rest) ((source p).surfaceBody.drop 14 |>.take 13)
      (loadExecution rest p s).val = some (signExecution rest p s).val := by
    rw [split]
    simp only [run_append,(maskExecution rest p s).property,(addBiasExecution rest p s).property,
      (expExecution rest p s).property,(normalExecution rest p s).property,
      (signExecution rest p s).property,Option.bind_some]
  exact (assignment_run_register (backend rest) "n_elements_s" .int [] _ _ _
    (by rcases p with ⟨input,scales,output,count,tile⟩; cases tile <;> rfl) ran).trans
      (loadExecution_n_elements_s rest p s)

attribute [local simp] VeriTile.Bench.Repairs.Fp4Scaled.signExecution_result
  VeriTile.Bench.Repairs.Fp4Scaled.signExecution_pid
  VeriTile.Bench.Repairs.Fp4Scaled.signExecution_n_elements_out
  VeriTile.Bench.Repairs.Fp4Scaled.signExecution_mem

def decodeExecution (rest : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest) ((source p).surfaceBody.drop 27 |>.take 4)
      (signExecution rest p s).val = some t} := by
  have hsign := signExecution_result rest p s
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp only [Parameters.block,Kernels.Tile.block] at hsign
    refine ⟨?_, ?_⟩
    swap
    dsimp only [source,Kernels.triton_f4_to_scaled_bf16_kernel,ComputeKernel.surfaceBody,List.drop,List.take]
    simp [Parameters.block,Kernels.Tile.block,hsign,run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,backend,converted,
      ofFloat,toFloat,ComputeDType.eraseDType,bop,Broadcast.leftIndex,Broadcast.rightIndex,
      Region.cast,Option.bind,State.setReg]
    try rfl
  )

@[local simp] theorem decodeExecution_output (rest : Algebra α) (p : Parameters) (s : State α) :
    (decodeExecution rest p s).val.regs .bf16 [p.block*2] "output" = some (fun j => converted (laneNibble p s j)) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> simp [decodeExecution,Parameters.block,Parameters.groups,Kernels.Tile.block,
    converted,backend,scaleCount,bop,Broadcast.leftIndex,Broadcast.rightIndex]

@[local simp] theorem decodeExecution_pid (rest : Algebra α) (p : Parameters) (s : State α) :
    (decodeExecution rest p s).val.regs .int [] "pid" = some (fun _ => (s.pids 0 : Int)) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> simp [decodeExecution,Parameters.block,Parameters.groups,Kernels.Tile.block,
    converted,backend,scaleCount,bop,Broadcast.leftIndex,Broadcast.rightIndex]

@[local simp] theorem decodeExecution_n_elements_out (rest : Algebra α) (p : Parameters) (s : State α) :
    (decodeExecution rest p s).val.regs .int [] "n_elements_out" = some (fun _ => ((p.count*2 : Nat) : Int)) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> simp [decodeExecution,Parameters.block,Parameters.groups,Kernels.Tile.block,
    converted,backend,scaleCount,bop,Broadcast.leftIndex,Broadcast.rightIndex]

@[local simp] theorem decodeExecution_n_elements_s (rest : Algebra α) (p : Parameters) (s : State α) :
    (decodeExecution rest p s).val.regs .int [] "n_elements_s" = some (fun _ => (scaleCount p : Int)) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> simp [decodeExecution,Parameters.block,Parameters.groups,Kernels.Tile.block,
    converted,backend,scaleCount,bop,Broadcast.leftIndex,Broadcast.rightIndex]

@[local simp] theorem decodeExecution_mem (rest : Algebra α) (p : Parameters) (s : State α) :
    (decodeExecution rest p s).val.mem = s.mem := by
  exact (assignment_run_memory (backend rest) _ _ _
    (by rcases p with ⟨input,scales,output,count,tile⟩; cases tile <;> rfl)
    (decodeExecution rest p s).property).trans (signExecution_mem rest p s)

#axiomsClean decodeExecution
end
end VeriTile.Bench.Repairs.Fp4Scaled
