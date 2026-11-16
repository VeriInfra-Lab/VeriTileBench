import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Product
import bench.tritonbench_g.fp4_to_bf16.Proofs.SignedIndex

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
open VeriTile.Bench.Repairs.Fp4ToBf16
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxRecDepth 50000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false
attribute [local irreducible] productExecution
attribute [local simp] productExecution_output productExecution_pid productExecution_n_elements_out productExecution_mem

def outputOffset (p : Parameters) (s : State α) (j : TileIndex [p.block*2]) :=
  s.pids 0*(p.block*2)+j.1.val
abbrev active (p : Parameters) (s : State α) (j : TileIndex [p.block*2]) :=
  outputOffset p s j<p.count*2

def laneValue (p : Parameters) (s : State α) (j : TileIndex [p.block*2]) : α :=
  Numerics.mul (converted (laneNibble p s j)) (scaleValue (loadedScale p s (j.1.val/32)))

def written (p : Parameters) (s : State α) : State α :=
  (TileShape.allIndices [p.block*2]).foldl (fun t j=>
    if active p s j then t.write p.output (outputOffset p s j) (.mk .bf16 (laneValue p s j)) else t) s

def storeCode (p : Parameters) : List ComputeStmt :=
  [.assign .int [] "block_start_out" (.alg (.mul .int .nil (.ref .int [] "pid")
      (.mul .int .nil (.constInt p.block) (.constInt 2)))),
   .assign .int [p.block*2] "offsets_out" (.alg (.add .int .scalarL
      (.ref .int [] "block_start_out") (.castNatToInt (.arange (p.block*2))))),
   .assign .bool [p.block*2] "mask_out" (.alg (.lt .int .scalarR
      (.ref .int [p.block*2] "offsets_out") (.ref .int [] "n_elements_out"))),
   .store .bf16 [p.block*2] (.region (Region.cast p.output)
      (.castIntToNat (.ref .int [p.block*2] "offsets_out")))
      (.alg (.ref .bf16 [p.block*2] "output")) (.mask (.ref .bool [p.block*2] "mask_out"))]

theorem storeCode_source (p : Parameters) : (source p).surfaceBody.drop 45=storeCode p := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> simp only [source,Kernels.triton_f4_to_scaled_bf16_kernel,storeCode,Parameters.block,Kernels.Tile.block,
    ComputeKernel.surfaceBody,List.drop,Region.cast] <;> rfl

def storeExecution (rest : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // run (backend rest) ((source p).surfaceBody.drop 45)
      (productExecution rest p s).val=some t} := by
  refine ⟨?_,?_⟩
  swap
  rw [storeCode_source]
  simp [storeCode,run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,store,
    backend,numeric,natLt,numericLt,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    ofFloat,toFloat,Option.bind,State.setReg,ComputeDType.eraseDType,Region.cast]
  try rfl

theorem source_partition (p : Parameters) :
    (source p).surfaceBody=((source p).surfaceBody.take 14)++
      ((source p).surfaceBody.drop 14 |>.take 3)++
      ((source p).surfaceBody.drop 17 |>.take 1)++
      ((source p).surfaceBody.drop 18 |>.take 1)++
      ((source p).surfaceBody.drop 19 |>.take 5)++
      ((source p).surfaceBody.drop 24 |>.take 3)++
      ((source p).surfaceBody.drop 27 |>.take 4)++
      ((source p).surfaceBody.drop 31 |>.take 4)++
      ((source p).surfaceBody.drop 35 |>.take 6)++
      ((source p).surfaceBody.drop 41 |>.take 4)++
      ((source p).surfaceBody.drop 45) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> with_unfolding_all rfl

/-- Complete execution of all 49 source statements, including the returned
helper, scale load, powi, NaN branch, reshapes, multiply and masked store. -/
def execution (rest : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // exec (backend rest) (source p) s=some t} := by
  refine ⟨(storeExecution rest p s).val,?_⟩
  change run _ (source p).surfaceBody s=_
  conv_lhs => rw [source_partition]
  have load : run (backend rest) ((source p).surfaceBody.take 14) s=some (loadExecution rest p s).val := by
    simpa only [List.drop_zero] using (loadExecution rest p s).property
  simp only [run_append,load,(maskExecution rest p s).property,(addBiasExecution rest p s).property,
    (expExecution rest p s).property,(normalExecution rest p s).property,(signExecution rest p s).property,
    (decodeExecution rest p s).property,(scaleLoadExecution rest p s).property,
    (scaleExecution rest p s).property,(productExecution rest p s).property,
    (storeExecution rest p s).property,Option.bind_some]

private theorem masked_memory {ι : Type} (r : RegionName) (off : ι → Nat)
    (value : ι → Cell α) (on : ι → Prop) [DecidablePred on] (l : List ι) (s : State α) :
    (l.foldl (fun t i=>if on i then t.write r (off i) (value i) else t) s).mem=
      l.foldl (fun m i=>if on i then (fun r' o=>if r'=r ∧ o=off i then value i else m r' o) else m) s.mem := by
  induction l generalizing s with
  | nil=>rfl
  | cons i l ih=>
    simp only [List.foldl_cons]
    by_cases hi : on i
    · simp only [hi,if_true];rw [ih];rfl
    · simpa only [hi,if_false] using ih s

theorem tail_memory (rest : Algebra α) (p : Parameters) (start final : State α)
    (pid : Nat) (values : TileIndex [p.block*2] → α)
    (hv : start.regs .bf16 [p.block*2] "output"=some values)
    (hp : start.regs .int [] "pid"=some (fun _=>(pid : Int)))
    (hc : start.regs .int [] "n_elements_out"=some (fun _=>((p.count*2 : Nat) : Int)))
    (ran : run (backend rest) ((source p).surfaceBody.drop 45) start=some final) :
    final.mem=(TileShape.allIndices [p.block*2]).foldl (fun mem j=>
      if pid*(p.block*2)+j.1.val<p.count*2 then
        (fun r o=>if r=p.output ∧ o=pid*(p.block*2)+j.1.val then .mk .bf16 (values j) else mem r o)
      else mem) start.mem := by
  rw [storeCode_source] at ran
  simp [storeCode,run,step,evalExpr,evalComputeOp,FP.Structural.evalOp,store,
    backend,hv,hp,hc,ComputeDType.eraseDType,numeric,numericLt,natLt,
    ofFloat,toFloat,Option.bind,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast] at ran
  cases ran
  simp [masked_memory,bop,Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
  simp only [SignedIndex.twice_nat,SignedIndex.mul_nat,SignedIndex.add_nat,
    SignedIndex.lt_nat,Int.toNat_natCast]
  try with_unfolding_all rfl

theorem execution_memory (rest : Algebra α) (p : Parameters) (s : State α) :
    (execution rest p s).val.mem=(written p s).mem := by
  have tail:=tail_memory rest p (productExecution rest p s).val (storeExecution rest p s).val
    (s.pids 0) (fun j=>laneValue p s j) (productExecution_output rest p s)
    (productExecution_pid rest p s) (productExecution_n_elements_out rest p s)
    (storeExecution rest p s).property
  change (storeExecution rest p s).val.mem=_
  rw [tail,productExecution_mem]
  simp [written,masked_memory,active,outputOffset]

theorem offsets_injective (p : Parameters) (s : State α) : Function.Injective (outputOffset p s) := by
  intro i j h
  have hv : i.1.val=j.1.val := by simpa [outputOffset] using h
  rcases i with ⟨i,⟨⟩⟩;rcases j with ⟨j,⟨⟩⟩
  have : i=j := Fin.ext hv
  cases this;rfl

theorem readback (p : Parameters) (s : State α) (j : TileIndex [p.block*2]) (live : active p s j) :
    (written p s).mem p.output (outputOffset p s j)=.mk .bf16 (laneValue p s j) :=
  State.masked_scatter_readback _ _ _ _ _ _ live (offsets_injective p s)

theorem frame (p : Parameters) (s : State α) (r : RegionName) (o : Nat)
    (miss : r≠p.output ∨ ∀ j : TileIndex [p.block*2],active p s j → o ≠ outputOffset p s j) :
    (written p s).mem r o=s.mem r o :=
  State.masked_scatter_frame _ _ _ _ _ _ _ (miss.imp id (fun h j _=>h j)) _

#axiomsClean execution
#axiomsClean execution_memory
#axiomsClean readback
#axiomsClean frame
end
end VeriTile.Bench.Repairs.Fp4Scaled
