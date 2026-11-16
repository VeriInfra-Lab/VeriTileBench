import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.ScalePower
import bench.tritonbench_g.fp4_to_bf16_conversion.Contract

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxRecDepth 40000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
attribute [local irreducible] scaleExecution
attribute [local simp] scaleExecution_output scaleExecution_s_fp scaleExecution_pid
  scaleExecution_n_elements_out scaleExecution_mem

def productCode (p : Parameters) : List ComputeStmt :=
  [.assign .bf16 [p.groups,32] "output" (.alg (.reshape [p.groups,32] (.ref .bf16 [p.block*2] "output"))),
   .assign .bf16 [p.groups,1] "s_fp" (.alg (.reshape [p.groups,1] (.ref .bf16 [p.groups] "s_fp"))),
   .assign .bf16 [p.groups,32] "output" (.alg (.mul .bf16 (.consSame (.consR .nil))
      (.ref .bf16 [p.groups,32] "output") (.ref .bf16 [p.groups,1] "s_fp"))),
   .assign .bf16 [p.block*2] "output" (.alg (.reshape [p.block*2] (.ref .bf16 [p.groups,32] "output")))]

theorem productCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 41 |>.take 4)=productCode p := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> simp only [source,Kernels.triton_f4_to_scaled_bf16_kernel,productCode,Parameters.groups,Parameters.block,
    Kernels.Tile.block,ComputeKernel.surfaceBody,List.drop,List.take] <;> rfl

private theorem reshape_rows (g b : Nat) (eq : g*32=b) (j : TileIndex [b])
    (h : TileShape.product [g,32]=TileShape.product [b]) :
    TileShape.reshapeIndex h j=
      (⟨j.1.val/32,by have:=j.1.isLt;omega⟩,
       ⟨j.1.val%32,Nat.mod_lt _ (by decide)⟩,PUnit.unit) := by
  subst b
  exact reshape_rows32 g j h

theorem reshape_groups (p : Parameters) (j : TileIndex [p.block*2])
    (h : TileShape.product [p.groups,32]=TileShape.product [p.block*2]) :
    TileShape.reshapeIndex h j=
      (⟨j.1.val/32,by have:=j.1.isLt;have:=block_groups p;omega⟩,
       ⟨j.1.val%32,Nat.mod_lt _ (by decide)⟩,PUnit.unit) :=
  reshape_rows p.groups (p.block*2) (block_groups p).symm j h

def productExecution (rest : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest) ((source p).surfaceBody.drop 41 |>.take 4)
      (scaleExecution rest p s).val=some t} := by
  refine ⟨?_,?_⟩
  swap
  rw [productCode_source]
  simp [productCode,run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,
    backend,TileShape.product,block_groups,bop,ofFloat,toFloat,Option.bind,State.setReg]
  try rfl

@[local simp] theorem productExecution_output (rest : Algebra α) (p : Parameters) (s : State α) :
    (productExecution rest p s).val.regs .bf16 [p.block*2] "output" =
      some (fun j=>Numerics.mul (converted (laneNibble p s j))
        (scaleValue (loadedScale p s (j.1.val/32)))) := by
  simp [productExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    reshape_groups,reshape_vector,TileShape.linearIndex,TileShape.product,block_groups,numeric,backend]
  funext j
  congr 1
  congr 1
  congr 1
  apply Prod.ext
  · apply Fin.ext; simp only [Prod.fst]; omega
  · exact Subsingleton.elim _ _

@[local simp] theorem productExecution_pid (rest : Algebra α) (p : Parameters) (s : State α) :
    (productExecution rest p s).val.regs .int [] "pid"=some (fun _=>(s.pids 0 : Int)) := by simp [productExecution]
@[local simp] theorem productExecution_n_elements_out (rest : Algebra α) (p : Parameters) (s : State α) :
    (productExecution rest p s).val.regs .int [] "n_elements_out"=some (fun _=>((p.count*2 : Nat) : Int)) := by
  simp [productExecution]
@[local simp] theorem productExecution_mem (rest : Algebra α) (p : Parameters) (s : State α) :
    (productExecution rest p s).val.mem=s.mem := by
  exact (assignment_run_memory (backend rest) _ _ _
    (by rw [productCode_source]; rfl) (productExecution rest p s).property).trans (scaleExecution_mem rest p s)

#axiomsClean productExecution
#axiomsClean productExecution_output
end
end VeriTile.Bench.Repairs.Fp4Scaled
