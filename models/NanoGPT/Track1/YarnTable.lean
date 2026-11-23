import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.ScalarOps
import models.NanoGPT.Track1.RopeBackward

/-! The table representation across reset, window changes and validation.
The same scalar operations build a full row and a row of adjacent pairs.
An observation expands the stored BF16 pairs exactly as the QK kernels do. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.YarnTable
open VeriTile.Triton FP

abbrev F32 := Value .fp32
abbrev BF16 := Value .bf16
abbrev Factors := BF16 × BF16

inductive Layout where | narrow | paired | wide
  deriving DecidableEq

def Layout.frequencies : Layout → Nat | .wide => 64 | _ => 32
def Layout.halves : Layout → Nat | .paired => 2 | _ => 1

/-- The paired table stores positions 2t and 2t+1 in its two row halves. -/
def rowPosition (s : Layout) (storageWidth t col : Nat) : Nat :=
  if s = .paired then 2 * t + col / storageWidth else t

def signBit (value : BF16) : BF16 := Value.ofBits .bf16 (value.raw ^^^ 32768)

theorem sign_bit_is_negation (value : BF16) : signBit value = value.neg := by
  have bound : value.raw < 65536 := value.bits.isLt
  have low : (value.raw ^^^ 32768) % 32768 = value.raw % 32768 := by
    simpa using (Nat.xor_mod_two_pow (a := value.raw) (b := 32768) (n := 15))
  have high : (value.raw ^^^ 32768) / 32768 = (value.raw / 32768) ^^^ 1 := by
    simpa using (Nat.xor_div_two_pow (a := value.raw) (b := 32768) (n := 15))
  have hbit : value.raw / 32768 = 0 ∨ value.raw / 32768 = 1 := by omega
  have h : value.raw ^^^ 32768 =
      (if value.raw / 32768 = 0 then 32768 else 0) + value.raw % 32768 := by
    rcases hbit with hbit | hbit <;> simp [hbit] at high ⊢ <;> omega
  by_cases small : value.raw < 32768
  · have notLarge : ¬32768 ≤ value.raw := by omega
    simp [signBit, Value.neg, Value.sign, Format.fractionBits, h, small, notLarge]
  · have large : 32768 ≤ value.raw := by omega
    simp [signBit, Value.neg, Value.sign, Format.fractionBits, h, small, large]

def expandPair (pair : Factors) (odd : Bool) : Factors :=
  (pair.1, if odd then signBit pair.2 else pair.2)

/-- The compile-time backward load sees the swapped component. The pair
index remains the same and the low and high halfwords switch places. -/
theorem backward_factor (pair : Factors) (d : Nat) :
    expandPair pair (decide ((d ^^^ 1) % 2 = 1)) =
      expandPair pair (decide (d % 2 = 0)) := by
  have hp : (d ^^^ 1) % 2 = 1 ↔ d % 2 = 0 := by
    have hs : (d ^^^ 1) % 2 = 1 ↔ d % 2 ≠ 1 := by simp
    omega
  simp [hp]

structure Arithmetic where
  config : Config
  position : Nat → F32
  initial : Nat → F32
  cos : F32 → F32
  sin : F32 → F32

/-- These are the scalar arguments after the common Python calculation
and PyTorch's FP32 casts. The tensor operations keep their source order. -/
structure WindowChange where
  oldWindow : F32
  tau : F32
  alpha : F32
  betaMinusAlpha : F32
  scalingFactor : F32
  oneMinusScale : F32

def rescale (cfg : Config) (w : WindowChange) (f : F32) : F32 :=
  let rotations := FP.div cfg (FP.mul cfg w.oldWindow f) w.tau
  let weight := FP.div cfg (FP.sub cfg rotations w.alpha) w.betaMinusAlpha
  let zero := Value.zero .fp32
  let one := Value.ofBits .fp32 0x3f800000
  let clamped := if FP.lt cfg weight zero then zero else if FP.lt cfg one weight then one else weight
  FP.mul cfg f (FP.add cfg w.scalingFactor (FP.mul cfg clamped w.oneMinusScale))

structure State where
  frequencies : Nat → F32
  factors : Nat → Nat → Factors
  validRows : Nat
  rebuildRows : Nat

def expanded (s : State) : State :=
  { s with
    frequencies := fun d => s.frequencies (d / 2)
    factors := fun t d => expandPair (s.factors t (d / 2)) (decide (d % 2 = 1)) }

def storageWidth (compact : Bool) (s : Layout) : Nat :=
  if compact then s.frequencies else 2 * s.frequencies

def buildCell (compact : Bool) (a : Arithmetic) (s : Layout) (read : Nat → F32) (t d : Nat) : Factors :=
  let width := storageWidth compact s
  let theta := FP.mul a.config (a.position (rowPosition s width t d)) (read (d % width))
  let cosine := FP.cast .bf16 a.config (a.cos theta)
  let sine := FP.cast .bf16 a.config (a.sin theta)
  -- Tensor multiplication by -1 on the finite BF16 sine is exact.
  (cosine, if !compact && d % 2 == 1 then sine.neg else sine)

private theorem pair_indices (s : Layout) (t d : Nat) :
    rowPosition s (storageWidth false s) t d = rowPosition s (storageWidth true s) t (d / 2) ∧
    d % storageWidth false s / 2 = (d / 2) % storageWidth true s := by
  cases s <;> simp [storageWidth, Layout.frequencies, rowPosition] <;> omega

theorem build_cell_equal (a : Arithmetic) (s : Layout) (read : Nat → F32) (t d : Nat) :
    buildCell false a s (fun j => read (j / 2)) t d =
      expandPair (buildCell true a s read t (d / 2)) (decide (d % 2 = 1)) := by
  have ⟨hPos, hFreq⟩ := pair_indices s t d
  simp [buildCell, hPos, hFreq, expandPair, sign_bit_is_negation]

def rebuild (compact : Bool) (a : Arithmetic) (s : Layout) (lo hi : Nat) (table : State) : State :=
  { table with
    factors := fun t d =>
      if lo ≤ t ∧ t < hi then buildCell compact a s table.frequencies t d else table.factors t d
    validRows := hi }

theorem rebuild_equal (a : Arithmetic) (s : Layout) (lo hi : Nat) (table : State) :
    rebuild false a s lo hi (expanded table) = expanded (rebuild true a s lo hi table) := by
  have cells :
      (fun t d => if lo ≤ t ∧ t < hi then
          buildCell false a s (fun j => table.frequencies (j / 2)) t d
        else expandPair (table.factors t (d / 2)) (decide (d % 2 = 1))) =
      (fun t d => expandPair
        (if lo ≤ t ∧ t < hi then buildCell true a s table.frequencies t (d / 2)
         else table.factors t (d / 2)) (decide (d % 2 = 1))) := by
    funext t d
    by_cases inRange : lo ≤ t ∧ t < hi
    · simpa [inRange] using build_cell_equal a s table.frequencies t d
    · simp [inRange]
  simp [rebuild, expanded, cells]

inductive Step where
  | reset
  | limit (rows : Nat)
  | extend (change : WindowChange)
  | complete

def execute (compact : Bool) (a : Arithmetic) (s : Layout) (maxRows : Nat) (t : State) (step : Step) : State :=
  match step with
  | .reset =>
    let initial := fun d =>
      let pair := if compact then d else d / 2
      if pair < 32 then a.initial pair else Value.zero .fp32
    rebuild compact a s 0 maxRows { t with frequencies := initial }
  | .limit rows => { t with rebuildRows := rows }
  | .extend w =>
    rebuild compact a s 0 (min maxRows t.rebuildRows)
      { t with frequencies := fun d => rescale a.config w (t.frequencies d) }
  | .complete =>
    if t.validRows < maxRows then rebuild compact a s t.validRows maxRows t else t

theorem execute_equal (a : Arithmetic) (s : Layout) (maxRows : Nat) (t : State) (step : Step) :
    execute false a s maxRows (expanded t) step = expanded (execute true a s maxRows t step) := by
  cases step with
  | reset =>
    simpa [execute, expanded] using rebuild_equal a s 0 maxRows
      { t with frequencies := fun d => if d < 32 then a.initial d else Value.zero .fp32 }
  | limit rows => rfl
  | extend w =>
    simpa [execute, expanded] using rebuild_equal a s 0 (min maxRows t.rebuildRows)
      { t with frequencies := fun d => rescale a.config w (t.frequencies d) }
  | complete =>
    by_cases incomplete : t.validRows < maxRows
    · simpa [execute, expanded, incomplete] using rebuild_equal a s t.validRows maxRows t
    · simp [execute, expanded, incomplete]

def reference (a : Arithmetic) (s : Layout) (maxRows : Nat) (t : State) (steps : List Step) : State :=
  steps.foldl (execute false a s maxRows) (expanded t)

def observed (a : Arithmetic) (s : Layout) (maxRows : Nat) (t : State) (steps : List Step) : State :=
  expanded (steps.foldl (execute true a s maxRows) t)

theorem observed_equal (a : Arithmetic) (s : Layout) (maxRows : Nat) (t : State) (steps : List Step) :
    observed a s maxRows t steps = reference a s maxRows t steps := by
  induction steps generalizing t with
  | nil => rfl
  | cons step more ih =>
    simp only [observed, reference, List.foldl_cons, execute_equal]
    exact ih (execute true a s maxRows t step)

/-- Feed the table observation through the original eight-values-per-lane
FP32 RMS tree and the ordered FMA for the 64-channel forward kernel. -/
def narrowForward {n : RopeShift.Length} (a : RopeShift.Arithmetic) (qk : Nat → BF16)
    (s : RopeBackward.Narrow.Shape) (factors : State) (t : Fin n.tokens) (p : RopeShift.Plane)
    (h : RopeShift.Head) (d : RopeBackward.Narrow.QkChannel) : RopeShift.Plane × Nat × BF16 :=
  let x := fun j => 2 * RopeBackward.Narrow.qkAddress s t p h j
  let factor := 2 * RopeBackward.Narrow.factorAddress s t h d
  let width := s.factorPitch
  let read : RopeShift.Inputs := fun buf addr =>
    let pair := factors.factors (addr / 2 / width) (addr / 2 % width)
    match buf with
    | .qk => qk addr
    | .f1 => pair.1
    | .f2 => pair.2
  let norm := RopeShift.normWord (k := 3) [4, 2, 1] 0x42800000 x d
  let word := RopeShift.rotateWord norm (x d) (x (RopeBackward.Narrow.scalarPairIndex d)) factor
  (p, 2 * RopeBackward.Narrow.gradAddress s t h d, FP.cast .bf16 a.config (RopeShift.eval a read word))

/-- The persistent byte count, excluding the small angular frequency vector. -/
def factorBytes (compact : Bool) (s : Layout) (rows : Nat) : Nat :=
  2 * 2 * rows * s.halves * storageWidth compact s

theorem half_the_factor_storage (s : Layout) (rows : Nat) :
    factorBytes false s rows = 2 * factorBytes true s rows := by
  cases s <;> simp [factorBytes, storageWidth, Layout.frequencies, Layout.halves] <;> omega

theorem default_rank_storage :
    factorBytes false .narrow 262144 + factorBytes false .paired 262144 + factorBytes false .wide 262144 =
      320 * 2 ^ 20 ∧
    factorBytes true .narrow 262144 + factorBytes true .paired 262144 + factorBytes true .wide 262144 =
      160 * 2 ^ 20 := by decide +kernel

#axiomsClean sign_bit_is_negation
#axiomsClean backward_factor
#axiomsClean build_cell_equal
#axiomsClean rebuild_equal
#axiomsClean execute_equal
#axiomsClean observed_equal
#axiomsClean half_the_factor_storage
#axiomsClean default_rank_storage
end VeriTile.Bench.Models.NanoGPT.Track1.YarnTable
