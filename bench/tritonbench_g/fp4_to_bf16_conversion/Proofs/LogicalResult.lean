import bench.tritonbench_g.fp4_to_bf16_conversion.IO
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Execution

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxRecDepth 50000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def Reads (p : Parameters) (xs : Inputs p) (s : State α) : Prop :=
  (∀ i : Fin p.count,(s.mem p.input i.val).read .nat=xs.packed i) ∧
  (∀ i : Fin (scaleLength p),(s.mem p.scales i.val).read .nat=xs.scales i)

def nibbleNat (p : Parameters) (xs : Inputs p) (j : Fin (p.count*2)) : Nat :=
  let b:=xs.packed ⟨j.val/2,by have:=j.isLt;omega⟩
  if j.val%2=0 then b >>> 4 else b &&& 15

/-- Ordered numerical graph: no associativity or distributivity is assumed. -/
def graph (p : Parameters) (xs : Inputs p) (j : Fin (p.count*2)) : α :=
  Numerics.mul (converted (nibbleNat p xs j)) (scaleValue (scaleByte p xs j))

def programState (p : Parameters) (pid : Nat) (s : State α) : State α :=
  {s with
    regs := fun _ _ _ => none
    pids := fun ax => if ax=0 then pid else 0
    numPids := fun ax => if ax=0 then grid p else 1}

theorem active_parent (p : Parameters) (s : State α) (j : TileIndex [p.block*2])
    (live : active p s j) : inputOffset p s (j.1.val/2)<p.count := by
  dsimp [active,outputOffset] at live
  dsimp [inputOffset]
  rw [← Nat.mul_assoc] at live
  omega

theorem address_facts (p : Parameters) (pid k : Nat) :
    pid*p.block+k/2=(pid*(p.block*2)+k)/2 ∧
    pid*p.groups+k/32=(pid*(p.block*2)+k)/32 ∧
    k%2=(pid*(p.block*2)+k)%2 := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> simp only [Parameters.block,Parameters.groups,Kernels.Tile.block] <;> omega

theorem active_scale (p : Parameters) (s : State α) (j : TileIndex [p.block*2])
    (live : active p s j) : scaleOffset p s (j.1.val/32)<scaleCount p := by
  have fact:=(address_facts p (s.pids 0) j.1.val).2.1
  dsimp [active,outputOffset] at live
  dsimp [scaleOffset,scaleCount]
  rw [fact]
  omega

theorem lane_value (p : Parameters) (xs : Inputs p) (s : State α) (reads : Reads p xs s)
    (j : TileIndex [p.block*2]) (live : active p s j) :
    laneValue p s j=graph p xs ⟨outputOffset p s j,live⟩ := by
  have parent:=active_parent p s j live
  have scale:=active_scale p s j live
  have fact:=address_facts p (s.pids 0) j.1.val
  have packedRead:=reads.1 ⟨inputOffset p s (j.1.val/2),parent⟩
  have scaleRead:=reads.2 ⟨scaleOffset p s (j.1.val/32),scale⟩
  simp only [laneValue,laneNibble,loaded,parent,if_true,packedRead,
    loadedScale,scale,scaleRead,graph,nibbleNat,scaleByte,outputOffset]
  simp only [inputOffset,scaleOffset] at packedRead scaleRead ⊢
  simp only [fact.1,fact.2.1,fact.2.2]

theorem program_memory (rest : Algebra α) (p : Parameters) (xs : Inputs p)
    (s : State α) (reads : Reads p xs s) (pid : Nat) (r : RegionName) (offset : Nat) :
    (execution rest p (programState p pid s)).val.mem r offset=
      if live : r=p.output ∧ offset<p.count*2 ∧ offset/(p.block*2)=pid then
        .mk .bf16 (graph p xs ⟨offset,live.2.1⟩) else s.mem r offset := by
  rw [execution_memory]
  have initialReads : Reads p xs (programState p pid s) := reads
  have positive : 0<p.block*2 := by have:=block_positive p;omega
  by_cases here : r=p.output ∧ offset<p.count*2 ∧ offset/(p.block*2)=pid
  · obtain ⟨hr,hn,hp⟩:=here
    subst r
    let j : TileIndex [p.block*2] := (⟨offset%(p.block*2),Nat.mod_lt _ positive⟩,PUnit.unit)
    have off : outputOffset p (programState p pid s) j=offset := by
      dsimp [outputOffset,programState,j]
      have rem:=Nat.mod_add_div offset (p.block*2)
      rw [hp,Nat.mul_comm (p.block*2) pid] at rem
      omega
    have live : active p (programState p pid s) j := by
      change outputOffset p (programState p pid s) j<p.count*2
      rw [off];exact hn
    have value:=readback p (programState p pid s) j live
    rw [lane_value p xs _ initialReads j live] at value
    simp only [off] at value
    rw [value]
    simp only [hn,hp,and_self,dite_true]
  · have miss : r≠p.output ∨ ∀ j : TileIndex [p.block*2],
        active p (programState p pid s) j → offset≠outputOffset p (programState p pid s) j := by
      by_cases hr : r=p.output
      · right
        intro j live eq
        apply here
        refine ⟨hr,?_,?_⟩
        · rw [eq];exact live
        · rw [eq]
          change (pid*(p.block*2)+j.1.val)/(p.block*2)=pid
          rw [Nat.mul_comm pid (p.block*2),Nat.mul_add_div positive,Nat.div_eq_of_lt j.1.isLt,Nat.add_zero]
      · exact Or.inl hr
    rw [frame p _ _ _ miss]
    simp only [here,dite_false,programState]

#axiomsClean lane_value
#axiomsClean program_memory
end
end VeriTile.Bench.Repairs.Fp4Scaled
