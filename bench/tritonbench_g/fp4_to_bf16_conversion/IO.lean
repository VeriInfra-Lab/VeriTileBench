import bench.tritonbench_g.fp4_to_bf16_conversion.Contract
import VeriTile.Triton.Float.LogicalProgramIO
import VeriTile.Triton.Float.ProgramEquivalence
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Numerics
import VeriTile.Spec

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
noncomputable section

/-- All channels describe the actual repaired source's contiguous allocations. -/
def inputChannels (p : Parameters) : List IOChannel :=
  [⟨.nat,[p.count],p.input⟩,⟨.nat,[scaleLength p],p.scales⟩]
def outputChannels (p : Parameters) : List IOChannel := [⟨.bf16,[p.count*2],p.output⟩]

def memory (p : Parameters) : MemoryIO α where
  inputs := inputChannels p
  outputs := outputChannels p
  buffers := [p.input,p.scales,p.output]
  read := Fin.cases (fun _ j=>j.1.val) (Fin.cases (fun _ j=>j.1.val) (fun i=>Fin.elim0 i))
  inputMask := fun _ _ _=>True
  write := Fin.cases (fun _ j=>j.1.val) (fun i=>Fin.elim0 i)
  outputMask := fun _ _ _=>True
  scratch := []
  scratchWrite := fun i=>Fin.elim0 i
  scratchMask := fun i=>Fin.elim0 i

variable {α : Type} [Inhabited α] [Numerics α]

def invocation (rest : Algebra α) (p : Parameters) (pid : Nat) : Invocation α where
  kernel := source p
  backend := backend rest
  clearRegisters := Bool.true
  pids := fun ax=>if ax=0 then pid else 0
  numPids := some (fun ax=>if ax=0 then grid p else 1)

def programIO (rest : Algebra α) (p : Parameters) (jobs : List Nat) : ProgramIO α α where
  toMemoryIO := memory p
  schedule := fun _=>jobs.map (invocation rest p)
  encode := id
  observe := fun _ _=>id

def inputCoordinates (p : Parameters) : ChannelValues α (inputChannels p) ≃ Inputs p where
  toFun := fun xs=>⟨fun i=>xs ⟨0,by simp [inputChannels]⟩ (i,()),fun i=>xs ⟨1,by simp [inputChannels]⟩ (i,())⟩
  invFun := fun xs=>Fin.cases (fun i=>xs.packed i.1)
    (Fin.cases (fun i=>xs.scales i.1) (fun i=>Fin.elim0 i))
  left_inv := by
    intro xs
    funext i j
    fin_cases i <;> rcases j with ⟨i,u⟩ <;> cases u <;> rfl
  right_inv := by intro xs;cases xs;rfl

def outputCoordinates (p : Parameters) : ChannelValues α (outputChannels p) ≃ (Fin (p.count*2) → α) where
  toFun := fun xs j=>xs ⟨0,by simp [outputChannels]⟩ (j,())
  invFun := fun xs=>Fin.cases (fun j=>xs j.1) (fun i=>Fin.elim0 i)
  left_inv := by intro xs;funext i j;fin_cases i;rcases j with ⟨i,u⟩;cases u;rfl
  right_inv := fun _=>rfl

def logicalIO (rest : Algebra α) (p : Parameters) (jobs : List Nat) :
    LogicalProgramIO α α (Inputs p) (Fin (p.count*2) → α) where
  core := programIO rest p jobs
  inputs := inputCoordinates p
  outputs := outputCoordinates p

/-- High nibble first; the result index also selects its scale group. -/
def nibble (p : Parameters) (xs : Inputs p) (j : Fin (p.count*2)) :
    VeriTile.Bench.TritonBenchG.Fp4ToBf16.Bits.Nibble :=
  let b : BitVec 8 := BitVec.ofNat 8 (xs.packed ⟨j.val/2,by have:=j.isLt;omega⟩)
  if j.val%2=0 then (b >>> 4).truncate 4 else b.truncate 4

def scaleByte (p : Parameters) (xs : Inputs p) (j : Fin (p.count*2)) : Nat :=
  xs.scales ⟨j.val/32,by have:=j.isLt;dsimp [scaleLength];omega⟩

namespace ScheduleIO
local instance : Inhabited FPModel.Word := ⟨FP.Value.zero .fp32⟩

structure Program (p : Parameters) where
  jobs : List Nat

def Program.io {p : Parameters} (prg : Program p) (rest : Algebra FPModel.Word) :
    ProgramIO FPModel.Word FPModel.Word := programIO rest p prg.jobs

def variant {p : Parameters} (rest : Algebra FPModel.Word) (left right : Program p) :
    ProgramIO.Variant (left.io rest) where
  schedule := (right.io rest).schedule
  observe := (left.io rest).observe

def Equivalent {p : Parameters} (left right : Program p) : Prop :=
  ∀ rest : Algebra FPModel.Word,
    (left.io rest).EquivalentOn (variant rest left right)
      (fun xs=>ByteDomain p ((inputCoordinates p) xs))

instance {p : Parameters} : VeriTile.Spec.ProgramSyntax (Program p) where
  Statement := ComputeStmt
  Signature := List IOChannel × List IOChannel
  signature _ := (inputChannels p,outputChannels p)
  body prg := prg.jobs.flatMap (fun _=>(source p).surfaceBody)
  structural := some Equivalent
  sameContext := fun left right=>left=right

def original (p : Parameters) (jobs : List Nat) : Program p := ⟨jobs⟩
def candidate (p : Parameters) (jobs : List Nat) : Program p := ⟨jobs⟩
def assumptions : VeriTile.Spec.Assumptions ComputeStmt := []
end ScheduleIO

end
end VeriTile.Bench.Repairs.Fp4Scaled
