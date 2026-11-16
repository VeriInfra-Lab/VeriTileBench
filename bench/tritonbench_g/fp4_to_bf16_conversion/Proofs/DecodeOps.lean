import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Model

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
theorem exponentCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 18 |>.take 1) = exponentCode ⟨p.input,p.output,p.count,p.block⟩ := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> rfl
theorem normalCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 19 |>.take 5) = normalCode ⟨p.input,p.output,p.count,p.block⟩ := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> rfl
theorem signCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 24 |>.take 3) = signCode ⟨p.input,p.output,p.count,p.block⟩ := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> rfl

theorem maskCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 14 |>.take 3) =
      ((VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue.source
        ⟨p.input,p.output,p.count,p.block⟩).surfaceBody.drop 13 |>.take 3) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> rfl

theorem biasCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 17 |>.take 1) =
      ((VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue.source
        ⟨p.input,p.output,p.count,p.block⟩).surfaceBody.drop 16 |>.take 1) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> rfl

end
end VeriTile.Bench.Repairs.Fp4Scaled
