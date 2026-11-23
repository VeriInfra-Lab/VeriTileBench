import models.NanoGPT.Track1.ValueAdamNativeQuotients

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.NativeColdTrace

open VeriTile.Triton FP

/-- The ordered native cautious-weight-decay epilogue shared by both kernels. -/
def storedWeightWord (original : BF16) (applied decay : F32) : BF16 :=
  let value := FP.cast .fp32 Config.ieeeValue original
  let product := FP.mul Config.ieeeValue value applied
  let gate := if FP.lt Config.ieeeValue (Value.ofBits .fp32 0) product then
    Value.ofBits .fp32 0x3f800000 else Value.ofBits .fp32 0
  let masked := FP.mul Config.ieeeValue value gate
  let delta := FP.fma Config.ieeeValue masked decay applied
  FP.cast .bf16 Config.ieeeValue (FP.sub Config.ieeeValue value delta)

def evaluateWeights (source : String) (layout : Layout) (route : Route) (period : Nat)
    (reciprocal : F32 → F32) (payload : Bool → F32 → F32 → F32)
    (learningRateMul : F32 → F32 → F32) (step decay : F32) (lane : Nat → F32)
    (original : Nat → BF16) : Option (List BF16) :=
  (evaluateStep source layout route period reciprocal payload learningRateMul step lane).map
    (List.zipWith (fun old applied => storedWeightWord old applied decay) ((List.range 8).map original))

/-- The decoded SASS has the same bit-level epilogue on all eight scaled
directions. Arbitrary matching learning rate, decay and current weights
can therefore consume either checked quotient path. -/
theorem Certificate.weights_eq (checked : Certificate before after kind period)
    (reciprocal : F32 → F32) (payload : Bool → F32 → F32 → F32)
    (learningRateMul : F32 → F32 → F32) (step decay : F32)
    (gradientNaN : Bool) (lanes : List (List F32)) (lane : Nat → F32)
    (original : Nat → BF16) (participates : (List.range 8).map lane ∈ lanes) :
    evaluateWeights after kind.layout ⟨gradientNaN, kind.hit lanes⟩ period
        reciprocal payload learningRateMul step decay lane original =
      evaluateWeights before .reference ⟨gradientNaN, false⟩ period
        reciprocal payload learningRateMul step decay lane original := by
  unfold evaluateWeights
  rw [checked.steps_eq reciprocal payload learningRateMul step gradientNaN lanes lane participates]

#axiomsClean Certificate.weights_eq

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.NativeColdTrace
