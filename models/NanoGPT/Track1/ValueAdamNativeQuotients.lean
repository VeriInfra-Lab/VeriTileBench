import models.NanoGPT.Track1.ValueAdamNativeColdTrace
import models.NanoGPT.Track1.ValueAdamGuardedReciprocal
import models.NanoGPT.Track1.ValueAdamExponentGuard

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.NativeColdTrace

open VeriTile.Triton FP

inductive GuardKind where
  | nan | exponent
  deriving BEq, Repr

def GuardKind.layout : GuardKind → Layout
  | .nan => .nanGuard
  | .exponent => .exponentGuard

def GuardKind.hit (kind : GuardKind) (lanes : List (List F32)) : Bool :=
  match kind with
  | .nan => GuardedReciprocal.warpHasNaN lanes.flatten
  | .exponent => ExponentGuard.warpMayHaveNaN lanes

theorem GuardKind.clear_not_nan (kind : GuardKind) (lanes : List (List F32))
    (lane : List F32) (word : F32) (inWarp : lane ∈ lanes) (inLane : word ∈ lane)
    (clear : kind.hit lanes = false) : word.isNaN = false := by
  cases kind
  · exact GuardedReciprocal.no_nan_on_direct_path _ word
      (List.mem_flatten.mpr ⟨lane, inWarp, inLane⟩) clear
  · exact ExponentGuard.no_nan_on_lane_vote lanes lane word inWarp inLane clear

private def expected (inverseFirst : Bool) (reciprocals : Nat) : Output :=
  ⟨(List.range 8).map fun slot => ⟨slot, inverseFirst⟩, reciprocals⟩

private structure TwoTraces (run : Bool → Option Output) (output : Output) : Type where
  agrees : ∀ gradientNaN : Bool, run gradientNaN = some output

private def checkBoth (run : Bool → Option Output) (output : Output) :
    Option (TwoTraces run output) := do
  if ordinary : run false = some output then
    if special : run true = some output then
      return ⟨by intro gradientNaN; cases gradientNaN <;> assumption⟩
    else none
  else none

structure Certificate (before after : String) (kind : GuardKind) (period : Nat) : Type where
  private mk ::
  private original : ∀ gradientNaN : Bool,
    decode before .reference ⟨gradientNaN, false⟩ period = some (expected true 8)
  private direct : ∀ gradientNaN : Bool,
    decode after kind.layout ⟨gradientNaN, false⟩ period = some (expected false 1)
  private fallback : ∀ gradientNaN : Bool,
    decode after kind.layout ⟨gradientNaN, true⟩ period = some (expected true 8)

def certify (before after : String) (kind : GuardKind) (period : Nat) :
    Option (Certificate before after kind period) := do
  let original ← checkBoth (fun nan => decode before .reference ⟨nan, false⟩ period) (expected true 8)
  let direct ← checkBoth (fun nan => decode after kind.layout ⟨nan, false⟩ period) (expected false 1)
  let fallback ← checkBoth (fun nan => decode after kind.layout ⟨nan, true⟩ period) (expected true 8)
  return ⟨original.agrees, direct.agrees, fallback.agrees⟩

/-- The payload handler can implement the device's choice of NaN bits.
For non-NaN numerators, the model uses the decoded, non-FTZ FP32 multiply
and the same result of the hardware reciprocal instruction. -/
def Quotient.eval (quotient : Quotient) (reciprocal : F32 → F32)
    (payload : Bool → F32 → F32 → F32) (lane : Nat → F32) : F32 :=
  let numerator := lane quotient.slot
  if numerator.isNaN then payload quotient.inverseFirst (reciprocal GuardedReciprocal.epsilon) numerator
  else if quotient.inverseFirst then GuardedReciprocal.reference reciprocal numerator
  else GuardedReciprocal.direct reciprocal numerator

def evaluate (source : String) (layout : Layout) (route : Route) (period : Nat)
    (reciprocal : F32 → F32) (payload : Bool → F32 → F32 → F32)
    (lane : Nat → F32) : Option (List F32) :=
  (decode source layout route period).map fun output =>
    output.terms.map fun term => term.eval reciprocal payload lane

theorem Certificate.effects_eq (checked : Certificate before after kind period)
    (reciprocal : F32 → F32) (payload : Bool → F32 → F32 → F32)
    (gradientNaN : Bool) (lanes : List (List F32)) (lane : Nat → F32)
    (participates : (List.range 8).map lane ∈ lanes) :
    evaluate after kind.layout ⟨gradientNaN, kind.hit lanes⟩ period reciprocal payload lane =
      evaluate before .reference ⟨gradientNaN, false⟩ period reciprocal payload lane := by
  unfold evaluate
  rw [checked.original gradientNaN]
  cases visible : kind.hit lanes
  · rw [checked.direct gradientNaN]
    simp only [expected, Option.map_some, List.map_map, Function.comp_def, Option.some.injEq]
    apply List.map_congr_left
    intro i found
    have localWord : lane i ∈ (List.range 8).map lane := List.mem_map.mpr ⟨i, found, rfl⟩
    have nonNaN := kind.clear_not_nan lanes _ _ participates localWord visible
    have equal : GuardedReciprocal.direct reciprocal (lane i) =
        GuardedReciprocal.reference reciprocal (lane i) :=
      GuardedReciprocal.selected_exact reciprocal [] (lane i)
    simpa [Quotient.eval, nonNaN] using equal
  · rw [checked.fallback gradientNaN]

def evaluateStep (source : String) (layout : Layout) (route : Route) (period : Nat)
    (reciprocal : F32 → F32) (payload : Bool → F32 → F32 → F32)
    (learningRateMul : F32 → F32 → F32) (step : F32) (lane : Nat → F32) :
    Option (List F32) :=
  (evaluate source layout route period reciprocal payload lane).map
    (List.map fun direction => learningRateMul direction step)

theorem Certificate.steps_eq (checked : Certificate before after kind period)
    (reciprocal : F32 → F32) (payload : Bool → F32 → F32 → F32)
    (learningRateMul : F32 → F32 → F32) (step : F32)
    (gradientNaN : Bool) (lanes : List (List F32)) (lane : Nat → F32)
    (participates : (List.range 8).map lane ∈ lanes) :
    evaluateStep after kind.layout ⟨gradientNaN, kind.hit lanes⟩ period
        reciprocal payload learningRateMul step lane =
      evaluateStep before .reference ⟨gradientNaN, false⟩ period
        reciprocal payload learningRateMul step lane := by
  unfold evaluateStep
  rw [checked.effects_eq reciprocal payload gradientNaN lanes lane participates]

#axiomsClean Certificate.effects_eq
#axiomsClean Certificate.steps_eq

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.NativeColdTrace
