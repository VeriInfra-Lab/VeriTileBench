import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Model

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
variable {α : Type} [Inhabited α] [Numerics α]
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
private theorem mul_128 (a : Nat) : (a : Int)*128 = ((a*128 : Nat) : Int) := by omega
private theorem mul_256 (a : Nat) : (a : Int)*256 = ((a*256 : Nat) : Int) := by omega
private theorem mul_512 (a : Nat) : (a : Int)*512 = ((a*512 : Nat) : Int) := by omega
private theorem mul_1024 (a : Nat) : (a : Int)*1024 = ((a*1024 : Nat) : Int) := by omega
private theorem mul_2048 (a : Nat) : (a : Int)*2048 = ((a*2048 : Nat) : Int) := by omega

private theorem half_nat (a : Nat) : (a : Int)/2 = ((a/2 : Nat) : Int) :=
  (Int.natCast_ediv a 2).symm

def loadExecution (rest : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // FP.Structural.run (backend rest) ((source p).surfaceBody.drop 0 |>.take 14) s = some t} := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    refine ⟨?_, ?_⟩
    swap
    simp [Parameters.block, Kernels.Tile.block, source, List.drop, List.take, Kernels.triton_f4_to_scaled_bf16_kernel, Kernels.triton_f4_to_scaled_bf16_kernel, FP.Structural.exec, ComputeKernel.surfaceBody,
      FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr, FP.Structural.evalComputeOp,
      FP.Structural.evalOp_unfold, FP.Structural.store, FP.Structural.numeric,
      FP.Structural.bop, FP.Structural.natLt, FP.Structural.numericLt, FP.Structural.numericEq,
      FP.Structural.integralMod, FP.Structural.integralFloorDiv, backend,
      FP.Structural.ofFloat, FP.Structural.toFloat, Region.cast, Option.bind,
      TileShape.product, ComputeDType.eraseDType,
      IntegerExpr.same_left, IntegerExpr.same_right,
      Broadcast.leftIndex, Broadcast.rightIndex,
      reshape_pair, TileShape.splitLastIndex]
    try rfl
  )

#axiomsClean loadExecution

@[local simp] theorem loadExecution_x_pos (rest : Algebra α) (p : Parameters) (s : State α) :
    (loadExecution rest p s).val.regs .nat [p.block*2] "__veritile_statement_jit_10" = some (fun j => positive (laneNibble p s j)) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp [Parameters.block, Kernels.Tile.block, loadExecution, loaded, laneNibble, inputOffset, positive, sign,
      shiftedExponent, addedBias, minusBias, signed32, unsigned32,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
      IntegerExpr.same_left, IntegerExpr.same_right]
    funext j
    dsimp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
    simp only [mul_128, mul_256, mul_512, mul_1024, mul_2048, half_nat, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast,
      Int.ofNat_lt]
    try rfl
  )

@[local simp] theorem loadExecution_sign_f4 (rest : Algebra α) (p : Parameters) (s : State α) :
    (loadExecution rest p s).val.regs .nat [p.block*2] "__veritile_statement_jit_9" = some (fun j => sign (laneNibble p s j)) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp [Parameters.block, Kernels.Tile.block, loadExecution, loaded, laneNibble, inputOffset, positive, sign,
      shiftedExponent, addedBias, minusBias, signed32, unsigned32,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
      IntegerExpr.same_left, IntegerExpr.same_right]
    funext j
    dsimp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
    simp only [mul_128, mul_256, mul_512, mul_1024, mul_2048, half_nat, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast,
      Int.ofNat_lt]
    try rfl
  )

@[local simp] theorem loadExecution_pid (rest : Algebra α) (p : Parameters) (s : State α) :
    (loadExecution rest p s).val.regs .int [] "pid" = some (fun j => (s.pids 0 : Int)) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp [Parameters.block, Kernels.Tile.block, loadExecution, loaded, laneNibble, inputOffset, positive, sign,
      shiftedExponent, addedBias, minusBias, signed32, unsigned32,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
      IntegerExpr.same_left, IntegerExpr.same_right]
    try rfl
  )

@[local simp] theorem loadExecution_n_elements_out (rest : Algebra α) (p : Parameters) (s : State α) :
    (loadExecution rest p s).val.regs .int [] "n_elements_out" = some (fun j => ((p.count*2 : Nat) : Int)) := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp [Parameters.block, Kernels.Tile.block, loadExecution, loaded, laneNibble, inputOffset, positive, sign,
      shiftedExponent, addedBias, minusBias, signed32, unsigned32,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast,
      IntegerExpr.same_left, IntegerExpr.same_right]
    try rfl
  )

@[local simp] theorem loadExecution_mem (rest : Algebra α) (p : Parameters) (s : State α) :
    (loadExecution rest p s).val.mem = s.mem := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;> (
    simp [Parameters.block, Kernels.Tile.block, loadExecution]
  )

attribute [local irreducible] loadExecution


end
end VeriTile.Bench.Repairs.Fp4Scaled
