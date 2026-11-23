import models.NanoGPT.Track1.ValueAdam
import models.NanoGPT.Track1.TailNorm

/-! Pointwise tail of the ANVIL banks after the ordered reductions.
The normalized gradient is rounded to BF16 before the two decay/update
FMAs. The candidate retains the rounded word in a register. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilEpilogue
open VeriTile.Triton FP
open TailNorm (F32 BF16 zero one widen narrow)

inductive Bank where
  | qk | vo | mlp
  deriving BEq

inductive Tile where
  | v1024 | v512
  deriving DecidableEq

def Bank.dimensions : Bank → Nat × Nat × Nat
  | .qk => (6, 256, 768)
  | .vo => (2, 768, 768)
  | .mlp => (3, 2816, 768)

def Bank.laneLength : Bank → F32
  | .qk => Value.ofBits .fp32 0x43800000
  | .vo | .mlp => Value.ofBits .fp32 0x44400000

def Bank.matrix (bank : Bank) (index : Nat) : Nat :=
  index / (bank.dimensions.2.1 * bank.dimensions.2.2)

def Bank.lane (bank : Bank) (index : Nat) : Nat :=
  match bank with
  | .qk => bank.matrix index * 768 + index % 768
  | .vo | .mlp => index / 768

structure Native where
  reciprocalSqrt : F32 → F32
  divideFull : F32 → F32 → F32

structure Snapshot where
  native : Native
  value : Nat → BF16
  /-- Per-lane sum of squared whitened gradient words. -/
  power : Nat → F32
  laneEnergy : Nat → F32
  /-- Ordered FP32 matrix reductions on the shared prefix. -/
  preSum : Nat → F32
  postSum : Nat → F32
  upper : Nat → UInt16
  mantissa : Nat → UInt16
  slowRail : Nat → F32
  decay : F32
  learningRate : Nat → F32
  tile : Tile := .v1024

structure Outputs where
  upper : Nat → UInt16
  mantissa : Nat → UInt16
  laneEnergy : Nat → F32

private def cfg := Config.ieeeValue

private def clamp (x : F32) : F32 :=
  let floor := Value.ofBits .fp32 0x2edbe6ff
  if x.isNaN || FP.lt cfg floor x then x else floor

def updatedEnergy (bank : Bank) (s : Snapshot) (lane : Nat) : F32 :=
  let mean := s.native.divideFull (s.power lane) bank.laneLength
  let energy := s.laneEnergy lane
  FP.fma cfg (FP.sub cfg mean energy) (Value.ofBits .fp32 0x3dcccccd) energy

def equalize (bank : Bank) (s : Snapshot) (index : Nat) : BF16 :=
  let energy := updatedEnergy bank s (bank.lane index)
  let gain := s.native.reciprocalSqrt (clamp energy)
  let pre := ValueAdam.sqrtRN (FP.mul cfg (s.preSum (bank.matrix index)) bank.laneLength)
  let post := ValueAdam.sqrtRN (s.postSum (bank.matrix index))
  let ratio := s.native.divideFull pre (clamp post)
  narrow (FP.mul cfg (widen (s.value index)) (FP.mul cfg gain ratio))

def update (bank : Bank) (s : Snapshot) (index : Nat) (grad : BF16) : F32 :=
  let raw := (UInt32.ofNat (s.upper index).toNat <<< 16) ||| UInt32.ofNat (s.mantissa index).toNat
  let exact := Value.ofBits .fp32 raw.toNat
  let align := FP.mul cfg (s.slowRail index) exact
  let active := if FP.lt cfg zero align || FP.eq cfg zero align then one else zero
  let lr := s.learningRate (bank.matrix index)
  let decay := FP.mul cfg (FP.mul cfg exact active) s.decay
  let decayed := FP.fma cfg decay.neg lr exact
  FP.fma cfg lr.neg (widen grad) decayed

/-- The observation contains the permanent two-part weight and the
lane-energy update. The CUDA graph's whitened gradient is private. -/
def write (bank : Bank) (s : Snapshot) (grad : Nat → BF16) : Outputs :=
  let word := fun i => update bank s i (grad i)
  ⟨fun i => UInt16.ofNat ((word i).raw / 65536), fun i => UInt16.ofNat (word i).raw, updatedEnergy bank s⟩

abbrev Epilogue := Bank → Snapshot → Outputs

def reference : Epilogue :=
  fun bank s => write bank s (fun i => Value.ofBits .bf16 (equalize bank s i).raw)

def fused : Epilogue := fun bank s => write bank s (equalize bank s)

theorem read_after_store (x : BF16) : Value.ofBits .bf16 x.raw = x := by
  cases x
  simp [Value.ofBits, Value.raw]

theorem epilogue_equal : fused = reference := by
  funext bank s
  simp [fused, reference, read_after_store]

structure Runtime (State : Type) where
  read : Bank → State → Option (Snapshot × State)
  commit : Bank → Outputs → State → Option State

def execute {State : Type} (r : Runtime State) (epilogue : Epilogue)
    (bank : Bank) (state : State) : Option State := do
  let (input, next) ← r.read bank state
  r.commit bank (epilogue bank input) next

#axiomsClean read_after_store
#axiomsClean epilogue_equal

end VeriTile.Bench.Models.NanoGPT.Track1.AnvilEpilogue
