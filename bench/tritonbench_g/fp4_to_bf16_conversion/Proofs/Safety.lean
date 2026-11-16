import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Execution
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Fp4ToBf16
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxRecDepth 50000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

private theorem pure_safe (rest : Algebra α) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (pure : ∀ st∈code,∀ s : State α,Safety.stepSafe (backend rest) bounds st s)
    (s : State α) : Safety.traceSafeList (backend rest) bounds code s := by
  induction code generalizing s with
  | nil=>simp [Safety.traceSafeList_unfold]
  | cons st code ih=>
    apply Safety.traceSafeList_cons_intro
    · exact pure st (by simp) s
    · intro t _;exact ih (fun st member=>pure st (by simp [member])) t

private theorem middle_safe (rest : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (backend rest) bounds ((source p).surfaceBody.drop 8 |>.take 23) s := by
  apply pure_safe
  intro st member t
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp only [source,Kernels.triton_f4_to_scaled_bf16_kernel,ComputeKernel.surfaceBody,List.drop,List.take,
      List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
    all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Op.intToSigned,Op.intToUnsigned,Op.unsignedToInt]
  )

private theorem arithmetic_safe (rest : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (backend rest) bounds ((source p).surfaceBody.drop 35 |>.take 10) s := by
  have code : ((source p).surfaceBody.drop 35 |>.take 10)=scaleCode p++productCode p := by
    rcases p with ⟨input,scales,output,count,tile⟩
    cases tile <;> with_unfolding_all rfl
  rw [code]
  apply pure_safe
  intro st member t
  simp only [scaleCode,productCode,List.mem_append,List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with (rfl|rfl|rfl|rfl|rfl|rfl)|(rfl|rfl|rfl|rfl)
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,offsetExpr,powerExpr,signedExponent,
    VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue.intConst,Op.intToSigned,Op.intToUnsigned,Op.unsignedToInt]

private theorem load_safe (rest : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat)
    (input : ∀ i : Fin p.block,inputOffset p s i.val<p.count → inputOffset p s i.val<bounds p.input) :
    Safety.traceSafeList (backend rest) bounds ((source p).surfaceBody.take 8) s := by
  rcases p with ⟨x,scales,output,count,tile⟩
  cases tile <;> (
    simp only [Parameters.block,Kernels.Tile.block] at input
    simp [source,Kernels.triton_f4_to_scaled_bf16_kernel,ComputeKernel.surfaceBody,List.take,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,backend,numeric,bop,natLt,numericLt,
      integralFloorDiv,Region.cast,Option.bind,ComputeDType.eraseDType]
    simpa only [SignedIndex.mul_nat,SignedIndex.add_nat,SignedIndex.lt_nat,Int.toNat_natCast,
      inputOffset,Parameters.block,Kernels.Tile.block] using input
  )

private theorem scale_access (n count bound pid : Nat)
    (scale : ∀ i : Fin n,pid*n+i.val<count → pid*n+i.val<bound) :
    ∀ i : Fin n,(pid : Int)*(n : Int)+(i.val : Int)<(count : Int) →
      ((pid : Int)*(n : Int)+(i.val : Int)).toNat<bound := by
  intro i live
  have cast : (pid : Int)*(n : Int)+(i.val : Int)=((pid*n+i.val : Nat) : Int) := by
    push_cast
    rfl
  rw [cast,Int.toNat_natCast]
  rw [cast,Int.ofNat_lt] at live
  exact scale i live

private theorem scale_safe (rest : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat)
    (scale : ∀ i : Fin p.groups,scaleOffset p s i.val<scaleCount p → scaleOffset p s i.val<bounds p.scales) :
    Safety.traceSafeList (backend rest) bounds ((source p).surfaceBody.drop 31 |>.take 4)
      (decodeExecution rest p s).val := by
  have hp:=decodeExecution_pid rest p s
  have hc:=decodeExecution_n_elements_s rest p s
  have hm:=decodeExecution_mem rest p s
  rcases p with ⟨x,scales,output,count,tile⟩
  cases tile <;> (
    simp only [Parameters.groups,Parameters.block,Kernels.Tile.block] at hp hc hm scale
    simp [source,Kernels.triton_f4_to_scaled_bf16_kernel,ComputeKernel.surfaceBody,List.drop,List.take,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,backend,numeric,bop,natLt,numericLt,
      integralFloorDiv,Region.cast,Option.bind,ComputeDType.eraseDType,hp,hc,hm]
    simp only [scaleOffset,Parameters.groups,Parameters.block,Kernels.Tile.block] at scale
    with_unfolding_all exact scale_access _ _ _ _ scale
  )

private theorem store_safe (rest : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat)
    (output : ∀ j : TileIndex [p.block*2],active p s j → outputOffset p s j<bounds p.output) :
    Safety.traceSafeList (backend rest) bounds ((source p).surfaceBody.drop 45)
      (productExecution rest p s).val := by
  rw [storeCode_source]
  simp [storeCode,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,backend,numeric,bop,natLt,numericLt,
    store,ofFloat,toFloat,Region.cast,Option.bind,ComputeDType.eraseDType,
    productExecution_pid,productExecution_n_elements_out,productExecution_output]
  simpa only [SignedIndex.twice_nat,SignedIndex.mul_nat,SignedIndex.add_nat,SignedIndex.lt_nat,
    Int.toNat_natCast,outputOffset] using (fun (j : Fin (p.block*2)) live => output (j,PUnit.unit) live)

theorem prefix31_run (rest : Algebra α) (p : Parameters) (s : State α) :
    run (backend rest) ((source p).surfaceBody.take 31) s=some (decodeExecution rest p s).val := by
  have split : (source p).surfaceBody.take 31=((source p).surfaceBody.take 14)++
      ((source p).surfaceBody.drop 14 |>.take 3)++
      ((source p).surfaceBody.drop 17 |>.take 1)++
      ((source p).surfaceBody.drop 18 |>.take 1)++
      ((source p).surfaceBody.drop 19 |>.take 5)++
      ((source p).surfaceBody.drop 24 |>.take 3)++
      ((source p).surfaceBody.drop 27 |>.take 4) := by
    rcases p with ⟨input,scales,output,count,tile⟩;cases tile <;> with_unfolding_all rfl
  conv_lhs=>rw [split]
  have load : run (backend rest) ((source p).surfaceBody.take 14) s=some (loadExecution rest p s).val := by
    simpa only [List.drop_zero] using (loadExecution rest p s).property
  simp only [run_append,load,(maskExecution rest p s).property,(addBiasExecution rest p s).property,
    (expExecution rest p s).property,(normalExecution rest p s).property,(signExecution rest p s).property,
    (decodeExecution rest p s).property,Option.bind_some]

theorem source_safe (rest : Algebra α) (p : Parameters) (s : State α) (bounds : RegionName → Nat)
    (input : ∀ i : Fin p.block,inputOffset p s i.val<p.count → inputOffset p s i.val<bounds p.input)
    (scale : ∀ i : Fin p.groups,scaleOffset p s i.val<scaleCount p → scaleOffset p s i.val<bounds p.scales)
    (output : ∀ j : TileIndex [p.block*2],active p s j → outputOffset p s j<bounds p.output) :
    TraceSafe (backend rest) bounds (source p) s := by
  change Safety.traceSafeList (backend rest) bounds (source p).surfaceBody s
  rw [←List.take_append_drop 31 (source p).surfaceBody,Safety.traceSafeList_append]
  constructor
  · have split : (source p).surfaceBody.take 31=(source p).surfaceBody.take 8++
        ((source p).surfaceBody.drop 8 |>.take 23) := by
      rcases p with ⟨input,scales,output,count,tile⟩;cases tile <;> with_unfolding_all rfl
    rw [split,Safety.traceSafeList_append]
    exact ⟨load_safe rest p s bounds input,fun t _=>middle_safe rest p t bounds⟩
  · intro t ran
    have same:=Option.some.inj ((prefix31_run rest p s).symm.trans ran)
    subst t
    have split : (source p).surfaceBody.drop 31=
        ((source p).surfaceBody.drop 31 |>.take 4)++
        ((source p).surfaceBody.drop 35 |>.take 10)++(source p).surfaceBody.drop 45 := by
      rcases p with ⟨input,scales,output,count,tile⟩;cases tile <;> with_unfolding_all rfl
    rw [split,Safety.traceSafeList_append,Safety.traceSafeList_append]
    refine ⟨⟨scale_safe rest p s bounds scale,fun t _=>arithmetic_safe rest p t bounds⟩,?_⟩
    intro t ran
    have known : run (backend rest)
        (((source p).surfaceBody.drop 31 |>.take 4)++((source p).surfaceBody.drop 35 |>.take 10))
        (decodeExecution rest p s).val=some (productExecution rest p s).val := by
      have split : ((source p).surfaceBody.drop 35 |>.take 10)=
          ((source p).surfaceBody.drop 35 |>.take 6)++((source p).surfaceBody.drop 41 |>.take 4) := by
        rcases p with ⟨input,scales,output,count,tile⟩;cases tile <;> with_unfolding_all rfl
      conv_lhs=>rw [split]
      simp only [run_append,(scaleLoadExecution rest p s).property,(scaleExecution rest p s).property,
        (productExecution rest p s).property,Option.bind_some]
    have same:=Option.some.inj (known.symm.trans ran)
    subst t
    exact store_safe rest p s bounds output

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.Fp4Scaled
