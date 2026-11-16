import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Grid
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Safety

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxRecDepth 50000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def channelTarget (p : Parameters) (xs : ChannelValues α (inputChannels p)) :
    ChannelValues α (outputChannels p) :=
  (outputCoordinates p).symm (fun j=>graph p ((inputCoordinates p) xs) j)

theorem input_reads (rest : Algebra α) (p : Parameters) (jobs : List Nat)
    (xs : ChannelValues α (inputChannels p)) (s : State α)
    (binding : (programIO rest p jobs).InputBinding xs s) :
    Reads p ((inputCoordinates p) xs) s := by
  constructor
  · intro i;exact binding ⟨0,by simp [programIO,memory,inputChannels]⟩ (i,PUnit.unit) trivial
  · intro i;exact binding ⟨1,by simp [programIO,memory,inputChannels]⟩ (i,PUnit.unit) trivial

theorem invocation_safe (rest : Algebra α) (p : Parameters) (jobs : List Nat)
    (xs : ChannelValues α (inputChannels p)) (bounds : RegionName → Nat)
    (backed : (programIO rest p jobs).InBounds bounds xs) (pid : Nat) (s : State α) :
    (invocation rest p pid).Safe bounds s := by
  change TraceSafe (backend rest) bounds (source p) (programState p pid s)
  apply source_safe
  · intro i live
    exact backed.2.1 ⟨0,by simp [programIO,memory,inputChannels]⟩
      (⟨inputOffset p (programState p pid s) i.val,live⟩,PUnit.unit) trivial
  · intro i live
    exact backed.2.1 ⟨1,by simp [programIO,memory,inputChannels]⟩
      (⟨scaleOffset p (programState p pid s) i.val,live⟩,PUnit.unit) trivial
  · intro j live
    exact backed.2.2.1 ⟨0,by simp [programIO,Fp4Scaled.memory,outputChannels]⟩
      (⟨outputOffset p (programState p pid s) j,live⟩,PUnit.unit) trivial

theorem schedule_safe (rest : Algebra α) (p : Parameters) (jobs : List Nat)
    (xs : ChannelValues α (inputChannels p)) (bounds : RegionName → Nat)
    (backed : (programIO rest p jobs).InBounds bounds xs) (s : State α) :
    ScheduleSafe bounds (jobs.map (invocation rest p)) s := by
  induction jobs generalizing s with
  | nil=>trivial
  | cons pid jobs ih=>
    constructor
    · exact invocation_safe rest p (pid::jobs) xs bounds backed pid s
    · intro t _;exact ih backed t

theorem implements (rest : Algebra α) (p : Parameters) (jobs : List Nat)
    (fresh : p.input≠p.output ∧ p.scales≠p.output) (order : jobs.Perm (List.range (grid p))) :
    (programIO rest p jobs).Implements (channelTarget p) := by
  intro xs _ bounds backed s binding
  have reads:=input_reads rest p jobs xs s binding
  obtain ⟨t,ran,memory⟩:=complete_result rest p ((inputCoordinates p) xs) jobs s fresh reads order
  refine ⟨schedule_safe rest p jobs xs bounds backed s,t,ran,?_,?_⟩
  · intro o j _
    fin_cases o
    change mapCell id (t.mem p.output j.1.val)=
      Cell.mk .bf16 (graph p ((inputCoordinates p) xs) j.1)
    rw [memory]
    simp only [targetMemory,j.1.isLt,and_self,dite_true]
    rfl
  · intro r offset outside
    rw [memory]
    have inactive : ¬(r=p.output ∧ offset<p.count*2) := by
      rintro ⟨hr,hn⟩
      have missed:=outside.1 ⟨0,by simp [programIO,Fp4Scaled.memory,outputChannels]⟩ (⟨offset,hn⟩,PUnit.unit) trivial
      exact missed.elim (fun ne=>ne hr) (fun ne=>ne rfl)
    simp only [targetMemory,inactive,dite_false]

/-- Every logical output, safe execution and full exterior frame. -/
theorem logical_implements (rest : Algebra α) (p : Parameters) (jobs : List Nat)
    (fresh : p.input≠p.output ∧ p.scales≠p.output) (order : jobs.Perm (List.range (grid p))) :
    (logicalIO rest p jobs).Implements (fun xs j=>graph p xs j) := by
  exact implements rest p jobs fresh order

#axiomsClean implements
#axiomsClean logical_implements
end
end VeriTile.Bench.Repairs.Fp4Scaled
