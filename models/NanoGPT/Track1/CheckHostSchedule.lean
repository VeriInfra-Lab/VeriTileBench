import models.NanoGPT.Track1.HostSchedule
import Lean

open VeriTile.Bench.Models.NanoGPT.Track1.HostSchedule
open VeriTile.Bench.Models.NanoGPT.Track1

private def fraction (x : Rat) : Lean.Json := Lean.toJson [x.num, x.den]

private def groupChange (x : Rat × ParameterChange) : Lean.Json :=
  let values := match x.2 with
    | .rateOnly => #[]
    | .momentum m => #[fraction m]
    | .adam a b wd => #[fraction a, fraction b, fraction wd]
  .arr (#[fraction x.1] ++ values)

private def observe (plan : Plan) (step : Nat) : Option Lean.Json := do
  let (stage, progress) ← plan.lookup step
  let f ← plan.forward step
  let o ← plan.optimizer step
  let numeric := o.values step [⟨.anvil, HostFloat64.round ((23 : Rat)/1000)⟩,
    ⟨.valueEmbedding, HostFloat64.round ((56 : Rat)/100)⟩, ⟨.other, HostFloat64.round ((8 : Rat)/1000)⟩]
  pure (.arr #[Lean.toJson stage.batchSize, Lean.toJson stage.maxLength, Lean.toJson stage.shortWindow,
    Lean.toJson stage.longWindow, fraction progress, fraction o.rate, fraction o.momentum,
    Lean.toJson (f.mtp.map (·.raw)), Lean.toJson f.prefixWeight.raw,
    Lean.Json.arr (numeric.groups.toArray.map groupChange), fraction o.fastBeta, fraction o.fastWeight,
    fraction numeric.sparseRate, fraction numeric.sparseDecay, fraction numeric.sparseEpsilon,
    Lean.toJson o.updateAdam, Lean.toJson o.splitEmbed])

private def run (base : Nat) : Option Lean.Json := do
  let schedule ← build base 20
  let rows ← (List.range (schedule.total + 1)).mapM (observe schedule)
  let pairs := schedule.limits.map fun (a, b) => [a, b]
  pure (.arr #[Lean.toJson base, Lean.toJson pairs, Lean.toJson schedule.splitStep, Lean.toJson rows])

def main : IO Unit := do
  let some result := [1122, 600, 1121, 1194].mapM run | throw (IO.userError "invalid host schedule")
  IO.println ((Lean.toJson result).compress)
