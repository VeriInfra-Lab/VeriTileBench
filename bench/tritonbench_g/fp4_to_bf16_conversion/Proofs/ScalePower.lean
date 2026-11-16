import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.ScaleValues
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.ScaleOps

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxRecDepth 40000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
attribute [local irreducible] offsetExpr powerExpr signedExponent intConst scaleLoadExecution
attribute [local simp] scaleLoadExecution_s scaleLoadExecution_output scaleLoadExecution_pid
  scaleLoadExecution_n_elements_out scaleLoadExecution_mem

@[local simp] theorem backend_bits (rest : Algebra α) (b : Float32Bits) :
    (backend rest).fp32Bits b=Numerics.bits b := rfl
@[local simp] theorem backend_narrow (rest : Algebra α) (precision : Option ComputeDType) (x : α) :
    (backend rest).cast precision .real .bf16 x=Numerics.narrow x := rfl

private theorem nan_payload : BitVec.ofInt 32 (signed32 2143289344) = (0x7fc00000 : BitVec 32) := by
  decide +kernel

def scaleExecution (rest : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest) ((source p).surfaceBody.drop 35 |>.take 6)
      (scaleLoadExecution rest p s).val = some t} := by
  refine ⟨?_,?_⟩
  swap
  rw [scaleCode_source]
  simp [scaleCode,run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,
    eval_powerExpr,eval_offsetExpr,nan_payload,numeric,natLt,numericLt,numericEq,bop,
    ofFloat,toFloat,Region.cast,Option.bind,Option.map,ComputeDType.eraseDType,
    Broadcast.leftIndex,Broadcast.rightIndex,State.setReg]
  try rfl

@[local simp] theorem scaleExecution_s_fp (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleExecution rest p s).val.regs .bf16 [p.groups] "s_fp" =
      some (fun j=>scaleValue (loadedScale p s j.1.val)) := by
  simp [scaleExecution,scaleValue,scaleExponent,nan_payload,
    bop,Broadcast.leftIndex,Broadcast.rightIndex]
  try rfl

@[local simp] theorem scaleExecution_output (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleExecution rest p s).val.regs .bf16 [p.block*2] "output" =
      some (fun j=>converted (laneNibble p s j)) := by simp [scaleExecution]
@[local simp] theorem scaleExecution_pid (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleExecution rest p s).val.regs .int [] "pid"=some (fun _=>(s.pids 0 : Int)) := by simp [scaleExecution]
@[local simp] theorem scaleExecution_n_elements_out (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleExecution rest p s).val.regs .int [] "n_elements_out"=some (fun _=>((p.count*2 : Nat) : Int)) := by
  simp [scaleExecution]
@[local simp] theorem scaleExecution_mem (rest : Algebra α) (p : Parameters) (s : State α) :
    (scaleExecution rest p s).val.mem=s.mem := by
  exact (assignment_run_memory (backend rest) _ _ _
    (by rw [scaleCode_source]; rfl) (scaleExecution rest p s).property).trans (scaleLoadExecution_mem rest p s)

#axiomsClean scaleExecution
#axiomsClean scaleExecution_s_fp
end
end VeriTile.Bench.Repairs.Fp4Scaled
