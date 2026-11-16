import bench.tritonbench_g.fp4_to_bf16_conversion.Proofs.Model

namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural

/-- A pure assignment to another register preserves the queried binding,
independently of the arithmetic used to compute the assigned value. -/
def avoidsRegister (name : RegName) : ComputeStmt → Bool
  | .assign _ _ other _ => decide (other ≠ name)
  | _ => Bool.false

theorem assignment_step_register {α : Type} [Inhabited α] (M : Algebra α)
    (name : RegName) (d : TileDType) (shape : TileShape) (st : ComputeStmt) (s t : State α)
    (different : avoidsRegister name st = Bool.true) (ran : step M st s = some t) :
    t.regs d shape name = s.regs d shape name := by
  cases st <;> simp only [avoidsRegister, Bool.false_eq_true] at different
  case assign dtype sh other e =>
    have ne : name ≠ other := Ne.symm (of_decide_eq_true different)
    cases he : evalExpr M e s with
    | none => simp [step,he] at ran
    | some value =>
      simp only [step,he,bind,pure,Option.bind_some,Option.some.injEq] at ran
      cases ran
      exact State.setReg_ne_name _ _ _ _ _ _ _ _ ne

theorem assignment_run_register {α : Type} [Inhabited α] (M : Algebra α)
    (name : RegName) (d : TileDType) (shape : TileShape)
    (code : List ComputeStmt) (s t : State α)
    (different : code.all (avoidsRegister name) = Bool.true) (ran : run M code s = some t) :
    t.regs d shape name = s.regs d shape name := by
  induction code generalizing s t with
  | nil =>
    simp only [run,Option.some.injEq] at ran
    cases ran
    rfl
  | cons st code ih =>
    simp only [List.all_cons,Bool.and_eq_true] at different
    cases hs : step M st s with
    | none => simp [run,hs] at ran
    | some next =>
      have rest : run M code next = some t := by simpa only [run,hs,bind,Option.bind_some] using ran
      exact (ih next t different.2 rest).trans
        (assignment_step_register M name d shape st s next different.1 hs)
end VeriTile.Bench.Repairs.Fp4Scaled
