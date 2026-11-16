import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.PublicIO
import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Numerics

namespace VeriTile.Bench.Repairs.Fp4Scaled.RealModel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem nibble_value (p : Parameters) (xs : Inputs p) (domain : RealDomain p xs)
    (j : Fin (p.count*2)) : nibbleNat p xs j=(nibble p xs j).toNat := by
  let i : Fin p.count:=⟨j.val/2,by have:=j.isLt;omega⟩
  let b : BitVec 8:=BitVec.ofNat 8 (xs.packed i)
  have byte : b.toNat=xs.packed i := by
    simp only [b,BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt (domain.1 i)
  change (if j.val%2=0 then xs.packed i >>> 4 else xs.packed i &&& 15)=_
  change _=(if j.val%2=0 then (b >>> 4).truncate 4 else b.truncate 4).toNat
  rw [←byte]
  split_ifs
  · exact (FPValue.byte_halves b).1
  · exact (FPValue.byte_halves b).2

/-- Both explicit signed casts preserve every finite physical E8M0 byte. -/
theorem finite_scale_exponent : ∀ b : Fin 255,scaleExponent b.val=(b.val : Int)-127 := by
  decide +kernel

theorem scale_value (b : Nat) (finite : b<255) :
    scaleValue (α := ℝ) b=(2 : ℝ)^((b : Int)-127) := by
  have ne : b≠255 := by omega
  unfold Fp4Scaled.scaleValue
  rw [if_pos ne]
  with_unfolding_all change (2 : ℝ)^(scaleExponent b)=_
  rw [finite_scale_exponent ⟨b,finite⟩]

theorem graph_real (p : Parameters) (xs : Inputs p) (domain : RealDomain p xs)
    (j : Fin (p.count*2)) :
    graph (α := ℝ) p xs j=RealSpec.e2m1 (nibble p xs j)*(2 : ℝ)^((scaleByte p xs j : Int)-127) := by
  unfold graph
  rw [nibble_value p xs domain j,converted_nibble]
  change RealSpec.e2m1 (nibble p xs j)*scaleValue (α := ℝ) (scaleByte p xs j)=_
  have finite : scaleByte p xs j<255 := domain.2 _
  rw [scale_value (scaleByte p xs j) finite]

theorem implements_real (rest : Algebra ℝ) (p : Parameters) (jobs : List Nat)
    (fresh : p.input≠p.output ∧ p.scales≠p.output) (order : jobs.Perm (List.range (grid p))) :
    (logicalIO rest p jobs).ImplementsOn (RealDomain p)
      (fun xs j=>RealSpec.e2m1 (nibble p xs j)*(2 : ℝ)^((scaleByte p xs j : Int)-127)) := by
  intro raw domain
  have eq : (fun j=>graph (α := ℝ) p ((inputCoordinates p) raw) j)=
      (fun j=>RealSpec.e2m1 (nibble p ((inputCoordinates p) raw) j)*
        (2 : ℝ)^((scaleByte p ((inputCoordinates p) raw) j : Int)-127)) :=
    funext (graph_real p _ domain)
  have correct:=implements rest p jobs fresh order raw trivial
  simpa only [channelTarget,eq] using correct

#axiomsClean nibble_value
#axiomsClean finite_scale_exponent
#axiomsClean graph_real
#axiomsClean implements_real
end
end VeriTile.Bench.Repairs.Fp4Scaled.RealModel
