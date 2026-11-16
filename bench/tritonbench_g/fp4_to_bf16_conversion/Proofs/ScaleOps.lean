import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Scale

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def offsetExpr (n : Nat) : Op .int [n] :=
  .sub .int .scalarR (Op.intToSigned 16 (.castNatToInt (.ref .nat [n] "s"))) (.constInt 127)
def signedExponent (n : Nat) : Op .int [n] := Op.intToSigned 32 (.ref .int [n] "s_offset")
def powerExpr (n : Nat) : Op .bf16 [n] :=
  .castFloat .real .bf16 (.withPrecision .real .fp32
    (.libdevicePowi .scalarL (.const (2.0 : ℝ)) (signedExponent n)))

@[simp] theorem eval_offsetExpr {α : Type} [Inhabited α] (M : Algebra α)
    (precision : Option ComputeDType) (n : Nat) (s : State α) :
    FP.Structural.evalOp M precision (offsetExpr n) s =
      (s.regs .nat [n] "s").map (fun v i=>(BitVec.ofNat 16 (v i)).toInt-127) := by
  rw [offsetExpr,FP.Structural.evalOp,FP.Structural.eval_intToSigned]
  simp only [FP.Structural.evalOp]
  cases he : s.regs .nat [n] "s" <;>
    simp only [bind,pure,Option.bind_none,Option.bind_some,Option.map_none,Option.map_some,
      bop,numeric,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

@[simp] theorem eval_signedExponent {α : Type} [Inhabited α] (M : Algebra α)
    (precision : Option ComputeDType) (n : Nat) (s : State α) :
    FP.Structural.evalOp M precision (signedExponent n) s =
      (s.regs .int [n] "s_offset").map (fun v i=>(BitVec.ofInt 32 (v i)).toInt) := by
  rw [signedExponent,FP.Structural.eval_intToSigned,FP.Structural.evalOp]
  cases s.regs .int [n] "s_offset" <;> rfl

theorem int32_values_valid {shape : TileShape} (v : TileIndex shape → Int) :
    Int32Bits.valuesValid (fun i=>(BitVec.ofInt 32 (v i)).toInt) = Bool.true := by
  simp only [Int32Bits.valuesValid,List.all_eq_true]
  intro i hi
  have hlo := BitVec.le_toInt (BitVec.ofInt 32 (v i))
  have hhi := @BitVec.toInt_lt 32 (BitVec.ofInt 32 (v i))
  simp only [Int32Bits.inRange,decide_eq_true_eq]
  exact ⟨hlo,hhi⟩

theorem int32_bmod_values_valid {shape : TileShape} (v : TileIndex shape → Int) :
    Int32Bits.valuesValid (fun i => (v i).bmod 4294967296) = Bool.true := by
  simpa only [BitVec.toInt_ofInt] using int32_values_valid v

attribute [local irreducible] signedExponent offsetExpr

@[simp] theorem eval_powerExpr {α : Type} [Inhabited α] [Numerics α]
    (rest : Algebra α) (precision : Option ComputeDType) (n : Nat) (s : State α) :
    FP.Structural.evalOp (backend rest) precision (powerExpr n) s =
      (s.regs .int [n] "s_offset").map
        (fun v i=>Numerics.narrow (Numerics.pow2 ((BitVec.ofInt 32 (v i)).toInt))) := by
  have two : (2.0 : ℝ)=2 := by norm_num
  cases h : s.regs .int [n] "s_offset" <;>
    simp [powerExpr,FP.Structural.evalOp,eval_signedExponent,h,backend,two,
      int32_values_valid,int32_bmod_values_valid,bop,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      Option.bind,Option.map]

/-- Proof-only source slices; their equality is checked against all five
actual pasted caller configurations. No slice replaces the source kernel. -/
def scaleCode (p : Parameters) : List ComputeStmt :=
  [.assign .int [p.groups] "s_offset" (.alg (offsetExpr p.groups)),
   .assign .bf16 [p.groups] "s_fp" (.alg (powerExpr p.groups)),
   .assign .int [] "nan_bits" (.alg (intConst 2143289344)),
   .assign .real [] "nan_fp32" (.compute (.bitcast .int32 .fp32 rfl
      (.alg .int32 (.ref .int [] "nan_bits")))),
   .assign .bf16 [] "nan_bf16" (.alg (.castFloat .real .bf16
      (.withPrecision .real .fp32 (.ref .real [] "nan_fp32")))),
   .assign .bf16 [p.groups] "s_fp" (.alg (.where
      (.ne .nat .scalarR (.ref .nat [p.groups] "s") (.constNat 255))
      (.ref .bf16 [p.groups] "s_fp") (.broadcast (.ref .bf16 [] "nan_bf16") [p.groups])))]

theorem scaleCode_source (p : Parameters) :
    ((source p).surfaceBody.drop 35 |>.take 6) = scaleCode p := by
  rcases p with ⟨input,scales,output,count,tile⟩
  cases tile <;>
    simp only [source,Kernels.triton_f4_to_scaled_bf16_kernel,scaleCode,offsetExpr,powerExpr,signedExponent,
      Parameters.groups,Parameters.block,Kernels.Tile.block,ComputeKernel.surfaceBody,List.drop,List.take] <;> rfl

#axiomsClean eval_powerExpr
#axiomsClean scaleCode_source
end
end VeriTile.Bench.Repairs.Fp4Scaled
