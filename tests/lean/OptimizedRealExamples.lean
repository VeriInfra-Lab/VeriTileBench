import bench.examples.TritonBenchVectorAddition.RealCorrect

open VeriTile Triton
open VeriTile.Bench.Examples
open scoped VeriTile.Triton.MaskedKernelIO₂

example (n B : Nat) :
    (TritonBenchVectorAdditionCorrect.optimizedIO n B).kernel =
      TritonBenchVectorAddition.Kernels.optimizedKernel n B := rfl

example (n B : Nat) :
    Spec.Real (TritonBenchVectorAdditionCorrect.optimizedIO n B ⊨ fun xs ys i => xs i + ys i) :=
  TritonBenchVectorAdditionCorrect.vector_addition_optimized_correct n B

open Lean Elab Command in
run_cmd do
  let name := `VeriTile.Bench.Examples.TritonBenchVectorAdditionFPEquiv.vector_addition_equiv
  if (← getEnv).contains name then
    throwError "Real correctness imported an FP equivalence proof: {name}"
