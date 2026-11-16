import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.LogicalResult

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxRecDepth 50000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def scheduledMemory (p : Parameters) (xs : Inputs p) (jobs : List Nat) (s : State α) :
    RegionName → Nat → Cell α :=
  fun r o=>if live : r=p.output ∧ o<p.count*2 ∧ o/(p.block*2)∈jobs then
    .mk .bf16 (graph p xs ⟨o,live.2.1⟩) else s.mem r o

def targetMemory (p : Parameters) (xs : Inputs p) (s : State α) : RegionName → Nat → Cell α :=
  fun r o=>if live : r=p.output ∧ o<p.count*2 then .mk .bf16 (graph p xs ⟨o,live.2⟩) else s.mem r o

theorem invocation_run (rest : Algebra α) (p : Parameters) (pid : Nat) (s : State α) :
    (invocation rest p pid).run s=exec (backend rest) (source p) (programState p pid s) := by
  rfl

theorem schedule_result (rest : Algebra α) (p : Parameters) (xs : Inputs p)
    (jobs : List Nat) (s : State α) (fresh : p.input≠p.output ∧ p.scales≠p.output)
    (reads : Reads p xs s) :
    ∃ final,runSchedule (jobs.map (invocation rest p)) s=some final ∧
      final.mem=scheduledMemory p xs jobs s := by
  induction jobs generalizing s with
  | nil=>exact ⟨s,rfl,by funext r o;simp [scheduledMemory]⟩
  | cons pid jobs ih=>
    let next:=(execution rest p (programState p pid s)).val
    have ran : (invocation rest p pid).run s=some next := by
      rw [invocation_run];exact (execution rest p (programState p pid s)).property
    have nextReads : Reads p xs next := by
      constructor
      · intro i
        have mem:=program_memory rest p xs s reads pid p.input i.val
        simp only [fresh.1,false_and,dite_false] at mem
        change (next.mem p.input i.val).read .nat=_
        rw [mem];exact reads.1 i
      · intro i
        have mem:=program_memory rest p xs s reads pid p.scales i.val
        simp only [fresh.2,false_and,dite_false] at mem
        change (next.mem p.scales i.val).read .nat=_
        rw [mem];exact reads.2 i
    obtain ⟨final,ranRest,memory⟩:=ih next nextReads
    refine ⟨final,?_,?_⟩
    · simp only [List.map_cons,runSchedule,ran,Option.bind_some];exact ranRest
    · rw [memory]
      funext r o
      unfold scheduledMemory
      have mem:=program_memory rest p xs s reads pid r o
      change next.mem r o=_ at mem
      rw [mem]
      by_cases hr : r=p.output <;> by_cases hn : o<p.count*2 <;>
        by_cases hp : o/(p.block*2)=pid <;> by_cases hj : o/(p.block*2)∈jobs <;>
        simp [hr,hn,hp,hj]

theorem grid_covers (p : Parameters) (o : Nat) (live : o<p.count*2) : o/(p.block*2)<grid p := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> simp only [grid,Parameters.block,Kernels.Tile.block,Parameters.count] at live ⊢ <;> omega

theorem complete_result (rest : Algebra α) (p : Parameters) (xs : Inputs p)
    (jobs : List Nat) (s : State α) (fresh : p.input≠p.output ∧ p.scales≠p.output)
    (reads : Reads p xs s) (order : jobs.Perm (List.range (grid p))) :
    ∃ final,runSchedule (jobs.map (invocation rest p)) s=some final ∧ final.mem=targetMemory p xs s := by
  obtain ⟨final,ran,memory⟩:=schedule_result rest p xs jobs s fresh reads
  refine ⟨final,ran,?_⟩
  rw [memory]
  funext r o
  by_cases hr : r=p.output <;> by_cases hn : o<p.count*2
  · have member : o/(p.block*2)∈jobs := order.mem_iff.mpr (List.mem_range.mpr (grid_covers p o hn))
    simp [scheduledMemory,targetMemory,hr,hn,member]
  · simp [scheduledMemory,targetMemory,hr,hn]
  · simp [scheduledMemory,targetMemory,hr,hn]
  · simp [scheduledMemory,targetMemory,hr,hn]

#axiomsClean complete_result
end
end VeriTile.Bench.Repairs.Fp4Scaled
