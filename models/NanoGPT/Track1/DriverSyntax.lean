/-! Control syntax for the pinned Track 1 Python driver. Expressions and
unchanged library calls retain their Python source text at this boundary. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Driver

inductive ValidationCalc where
  | checkBatchCount
  | bindBatchCount
  | loadBatches
  | resetLoss
  | accumulate (forward : String)
  | scale
  | reduce
  deriving Repr

inductive TrainingBatchOp where
  | openSource (warmup : Bool)
  | access (consume : Bool)
  deriving Repr

inductive Stmt where
  | shared (line : Nat) (python : String)
  | diagnostics (line : Nat) (python : String)
  | validationCalc (line : Nat) (op : ValidationCalc)
  | validation (line : Nat) (python : String)
  | trainStep (line : Nat) (invocation : String)
  | trainingBatch (line : Nat) (op : TrainingBatchOp)
  | setupSchedule (line : Nat)
  | advanceSchedule (line : Nat)
  | optimizerStep (line : Nat) (invocation : String)
  | rowForward (line : Nat)
  | sampledGather (line : Nat)
  | graphForward (line : Nat)
  | graphBackward (line : Nat)
  | graphHoldNgram (line : Nat)
  | rowLand (line : Nat)
  | tailTick (line : Nat)
  | tailShip (line : Nat)
  | branch (line : Nat) (test : String) (yes no : List Stmt)
  | loop (line : Nat) (target iterable : String) (body : List Stmt)
  | within (line : Nat) (items : List String) (body : List Stmt)
  | breakLoop (line : Nat)
  deriving Repr

end VeriTile.Bench.Models.NanoGPT.Track1.Driver
