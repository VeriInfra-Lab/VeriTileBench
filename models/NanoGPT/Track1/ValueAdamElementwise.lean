import models.NanoGPT.Track1.ValueAdamReadScale

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

open VeriTile.Triton FP
open SparsePackets (Channel)

inductive Port where
  | weight | gradient | first | second | step | decay
  deriving BEq, DecidableEq, Hashable, Repr

inductive Period where
  | two | four
  deriving BEq, DecidableEq, Repr

def Period.beta1 : Period → Nat
  | .two => 0x3f400000
  | .four => 0x3f100000

def Period.beta2 : Period → Nat
  | .two => 0x3f733333
  | .four => 0x3f670a3d

def Period.firstMix : Period → Nat
  | .two => 0x3e800000
  | .four => 0x3ee00000

def Period.secondMix : Period → Nat
  | .two => 0x3d4ccccd
  | .four => 0x3dc7ae14

def Period.launch (period : Period) (event : Nat) (step decay : F32) : Launch :=
  ⟨event, Value.ofBits .fp32 period.beta1, Value.ofBits .fp32 period.firstMix,
    Value.ofBits .fp32 period.beta2, Value.ofBits .fp32 period.secondMix,
    Value.ofBits .fp32 0x2edbe6ff, step, decay, eighth⟩

inductive Term where
  | input (port : Port) (index : Nat)
  | constant (bits : Nat)
  | widenFP16 (value : Term)
  | widenBF16 (value : Term)
  | narrowFP16 (value : Term)
  | narrowBF16 (value : Term)
  | add (left right : Term)
  | sub (left right : Term)
  | mul (left right : Term)
  | fma (left right addend : Term)
  | divideFull (left right : Term)
  | sqrt (value : Term)
  | greater (left right : Term)
  | select (yes no condition : Term)
  deriving BEq, DecidableEq, Inhabited

inductive Data where
  | float (value : F32)
  | half (bits : Half)
  | brain (value : BF16)
  | flag (result : Bool)

structure Inputs where
  cell : Nat → Cell
  gradient : Nat → Half
  step : F32
  decay : F32
  divideFull : F32 → F32 → F32

def Inputs.read (input : Inputs) (index : Nat) : Port → Data
  | .weight => .brain (input.cell index).weight
  | .gradient => .half (input.gradient index)
  | .first => .float (input.cell index).first
  | .second => .float (input.cell index).second
  | .step => .float input.step
  | .decay => .float input.decay

private def floatWord : Data → Option F32
  | .float value => some value
  | _ => none

def Term.eval (input : Inputs) : Term → Option Data
  | .input port index => some (input.read index port)
  | .constant bits => some (.float (Value.ofBits .fp32 bits))
  | .widenFP16 word => do
      let .half value ← word.eval input | none
      return .float (SparsePackets.halfToFP32 value)
  | .widenBF16 word => do
      let .brain value ← word.eval input | none
      return .float (FP.cast .fp32 Config.ieeeValue value)
  | .narrowFP16 word => do
      let value ← (word.eval input).bind floatWord
      return .half (narrow value)
  | .narrowBF16 word => do
      let value ← (word.eval input).bind floatWord
      return .brain (FP.cast .bf16 Config.ieeeValue value)
  | .add a b => do
      let a ← (a.eval input).bind floatWord
      let b ← (b.eval input).bind floatWord
      return .float (FP.add Config.ieeeValue a b)
  | .sub a b => do
      let a ← (a.eval input).bind floatWord
      let b ← (b.eval input).bind floatWord
      return .float (FP.sub Config.ieeeValue a b)
  | .mul a b => do
      let a ← (a.eval input).bind floatWord
      let b ← (b.eval input).bind floatWord
      return .float (FP.mul Config.ieeeValue a b)
  | .fma a b c => do
      let a ← (a.eval input).bind floatWord
      let b ← (b.eval input).bind floatWord
      let c ← (c.eval input).bind floatWord
      return .float (FP.fma Config.ieeeValue a b c)
  | .divideFull a b => do
      let a ← (a.eval input).bind floatWord
      let b ← (b.eval input).bind floatWord
      return .float (input.divideFull a b)
  | .sqrt a => do
      let a ← (a.eval input).bind floatWord
      return .float (sqrtRN a)
  | .greater a b => do
      let a ← (a.eval input).bind floatWord
      let b ← (b.eval input).bind floatWord
      return .flag (FP.lt Config.ieeeValue b a)
  | .select yes no condition => do
      let .flag result ← condition.eval input | none
      let yes ← (yes.eval input).bind floatWord
      let no ← (no.eval input).bind floatWord
      return .float (if result then yes else no)

structure Stores where
  weight : Term
  first : Term
  second : Term
  deriving BEq, DecidableEq

def Stores.eval (input : Inputs) (stores : Stores) : Option Cell := do
  let .brain weight ← stores.weight.eval input | none
  let .float first ← stores.first.eval input | none
  let .float second ← stores.second.eval input | none
  return ⟨weight, first, second⟩

def gradientTerm (fused : Bool) (index : Nat) : Term :=
  let raw := Term.input .gradient index
  if fused then .narrowFP16 (.mul (.widenFP16 raw) (.constant 0x3e000000)) else raw

def expected (period : Period) (_tile : Tile) (fused : Bool) (index : Nat) : Stores :=
  let g := Term.widenFP16 (gradientTerm fused index)
  let p := Term.widenBF16 (.input .weight index)
  let first := Term.fma (.input .first index) (.constant period.beta1)
    (.mul g (.constant period.firstMix))
  let second := Term.fma (.input .second index) (.constant period.beta2)
    (.mul (.mul g g) (.constant period.secondMix))
  let direction := Term.divideFull first (.add (.sqrt second) (.constant 0x2edbe6ff))
  let stepSize := Term.input .step 0
  let decay := Term.input .decay 0
  let step := Term.mul stepSize direction
  let gate := Term.greater (.mul step p) (.constant 0)
  let masked := Term.mul p (.select (.constant 0x3f800000) (.constant 0) gate)
  let delta := Term.fma decay masked step
  ⟨.narrowBF16 (.sub p delta), first, second⟩

def Inputs.gradientWord (input : Inputs) (fused : Bool) (index : Nat) : Half :=
  if fused then scaleRead (input.gradient index) else input.gradient index

theorem gradient_term_exact (input : Inputs) (fused : Bool) (index : Nat) :
    (gradientTerm fused index).eval input = some (.half (input.gradientWord fused index)) := by
  cases fused <;> rfl

theorem expected_exact (input : Inputs) (period : Period) (tile : Tile) (event : Nat)
    (fused : Bool) (index : Nat) :
    (expected period tile fused index).eval input =
      some (update ⟨fun _ => tile, input.divideFull⟩ (period.launch event input.step input.decay)
        index (input.cell index) (input.gradientWord fused index)) := by
  simp (config := {maxSteps := 100000}) [expected, Stores.eval, Term.eval,
    gradient_term_exact, Inputs.read, floatWord, update, Period.launch] <;> rfl

theorem read_scale_expected (raw : Inputs) (period : Period) (tile : Tile) (index : Nat) :
    (expected period tile true index).eval raw =
      (expected period tile false index).eval { raw with gradient := fun i => scaleRead (raw.gradient i) } := by
  rw [expected_exact raw period tile 0, expected_exact _ period tile 0]
  rfl

#axiomsClean expected_exact
#axiomsClean read_scale_expected

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
