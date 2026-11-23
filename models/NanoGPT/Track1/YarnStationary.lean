import models.NanoGPT.Track1.SourceAnchors
import models.NanoGPT.Track1.YarnTable

/-! The unpaired wide table has 32 rotating pairs and 32 stationary pairs.
The driver constructs only the rotating pairs. The omitted values are
derived from the pinned window arguments and the exact zero-angle cases. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.YarnStationary
open VeriTile.Triton FP
open YarnTable (F32 BF16 Factors)

abbrev Change := Fin Driver.wideWindowArguments.size

def window (w : Change) : YarnTable.WindowChange :=
  let (old, gain, complement) := Driver.wideWindowArguments[w]
  ⟨Value.ofBits .fp32 old, Value.ofBits .fp32 Driver.windowTauBits, Value.ofBits .fp32 0x3f800000,
    Value.ofBits .fp32 0x41f80000, Value.ofBits .fp32 gain, Value.ofBits .fp32 complement⟩

theorem window_keeps_positive_zero : ∀ w : Change,
    YarnTable.rescale Config.ieeeValue (window w) (Value.zero .fp32) = Value.zero .fp32 := by
  decide +kernel

private def positionRaw (row : Nat) : Nat :=
  if row = 0 then 0 else
    let e := min row.log2 23
    (127 + e) * 2 ^ 23 + (row * 2 ^ (23 - e)) % (2 ^ 23)

/-- Encoding of the nonnegative FP32 arange positions. The driver uses
rows below 2^24, where the integer conversion discards no low bits. -/
def position (row : Nat) : F32 := Value.ofBits .fp32 (positionRaw row)

private theorem position_raw_range (row : Nat) : positionRaw row < 151 * 2 ^ 23 := by
  unfold positionRaw
  split
  · exact Nat.mul_pos (by decide +kernel) (Nat.pow_pos (Nat.succ_pos 1))
  · let e := min row.log2 23
    let m := (row * 2 ^ (23 - e)) % (2 ^ 23)
    change (127 + e) * 2 ^ 23 + m < 151 * 2 ^ 23
    have he : 127 + e + 1 ≤ 151 := by
      have := Nat.min_le_right row.log2 23
      dsimp [e]
      omega
    have hm : m < 2 ^ 23 := Nat.mod_lt _ (Nat.pow_pos (Nat.succ_pos 1))
    refine Nat.lt_of_lt_of_le (Nat.add_lt_add_left hm _) ?_
    rw [← Nat.one_mul (2 ^ 23), ← Nat.add_mul]
    exact Nat.mul_le_mul_right _ he

private theorem position_bits (row : Nat) : (position row).raw < 151 * 2 ^ 23 := by
  unfold position Value.raw Value.ofBits
  rw [BitVec.toNat_ofNat]
  exact Nat.lt_of_le_of_lt (Nat.mod_le _ _) (position_raw_range row)

theorem positive_zero_angle (row : Nat) :
    FP.mul Config.ieeeValue (position row) (Value.zero .fp32) = Value.zero .fp32 := by
  have h := position_bits row
  have sign : (position row).sign = false := by
    have hz : (position row).raw / 2147483648 = 0 := by omega
    simp [Value.sign, Format.fractionBits, hz]
  have nonInf : (position row).exponent ≠ 255 := by
    change (position row).raw / 8388608 % 256 ≠ 255
    omega
  obtain ⟨magnitude, decode⟩ : ∃ q : Rat, (position row).decode = .finite false q := by
    simp [Value.decode, sign, nonInf]
  have zeroDecode : (Value.zero .fp32).decode = .finite false 0 := by decide +kernel
  simp only [FP.mul, FP.prepare, Config.ieeeValue, Bool.false_and, Bool.false_eq_true, ↓reduceIte,
    decode, zeroDecode, Rat.mul_zero]
  rfl

/-- The common trigonometric semantics give cos(0) = 1 and preserve the
zero sign for sin. Nonzero arguments use the same runtime primitives
on both sides. -/
def arithmetic (a : YarnTable.Arithmetic) : YarnTable.Arithmetic where
  config := Config.ieeeValue
  position := position
  initial := a.initial
  cos := fun x => if x.isZero then Value.ofBits .fp32 0x3f800000 else a.cos x
  sin := fun x => if x.isZero then x else a.sin x

def constants : Factors := (Value.ofBits .bf16 0x3f80, Value.zero .bf16)

theorem stationary_factor (a : YarnTable.Arithmetic) (read : Nat → F32)
    (row col : Nat) (zero : read (col % 64) = Value.zero .fp32) :
    YarnTable.buildCell true (arithmetic a) .wide read row col = constants := by
  have cosOne : FP.cast .bf16 Config.ieeeValue (Value.ofBits .fp32 0x3f800000) = constants.1 := by decide +kernel
  have sinZero : FP.cast .bf16 Config.ieeeValue (Value.zero .fp32) = constants.2 := by decide +kernel
  have isZero : (Value.zero .fp32).isZero = true := by decide
  simp [YarnTable.buildCell, YarnTable.storageWidth, YarnTable.Layout.frequencies, YarnTable.rowPosition,
    arithmetic, zero, positive_zero_angle, isZero, cosOne, sinZero, constants]

theorem aligned_tile_live {tiles blockSize : Nat} (block : Fin tiles) (lane : Fin blockSize) :
    block.val * blockSize + lane.val < tiles * blockSize := by
  calc
    _ < block.val * blockSize + blockSize := Nat.add_lt_add_left lane.isLt _
    _ = (block.val + 1) * blockSize := by rw [Nat.add_mul, Nat.one_mul]
    _ ≤ _ := Nat.mul_le_mul_right _ block.isLt

theorem factor_address_in_bounds {tokens : Nat} (row : Fin tokens) (pair : Fin 32) :
    2 * (row.val * 32 + pair.val) + 1 < 2 * tokens * 32 := by
  have := row.isLt
  have := pair.isLt
  omega

/-- Only the rotating pairs have memory. Stored sine entries have their
even-channel sign. The backward decoder exchanges the two signs. -/
structure RotatingState where
  frequencies : Fin 32 → F32
  factors : Nat → Fin 32 → Factors
  validRows : Nat
  rebuildRows : Nat

def pad {α : Type} (suffix : α) (read : Fin 32 → α) (d : Nat) : α :=
  if h : d < 32 then read ⟨d, h⟩ else suffix

def decoded (t : RotatingState) : YarnTable.State where
  frequencies := pad (Value.zero .fp32) t.frequencies
  factors := fun row => pad constants (t.factors row)
  validRows := t.validRows
  rebuildRows := t.rebuildRows

structure Matches (original : YarnTable.State) (s : RotatingState) : Prop where
  frequencies : ∀ d : Fin 64, original.frequencies d.val = (decoded s).frequencies d.val
  factors : ∀ row (d : Fin 64), original.factors row d.val = (decoded s).factors row d.val
  valid : original.validRows = s.validRows
  limit : original.rebuildRows = s.rebuildRows

theorem decoded_matches (s : RotatingState) : Matches (decoded s) s :=
  ⟨fun _ => rfl, fun _ _ => rfl, rfl, rfl⟩

def rebuild (a : YarnTable.Arithmetic) (lo hi : Nat) (s : RotatingState) : RotatingState :=
  { s with
    factors := fun row d => if lo ≤ row ∧ row < hi then
        YarnTable.buildCell true (arithmetic a) .wide (pad (Value.zero .fp32) s.frequencies) row d.val
      else s.factors row d
    validRows := hi }

theorem rebuilt_matches (a : YarnTable.Arithmetic) (lo hi : Nat) {f : YarnTable.State}
    {s : RotatingState} (matched : Matches f s) :
    Matches (YarnTable.rebuild true (arithmetic a) .wide lo hi f) (rebuild a lo hi s) := by
  refine ⟨matched.frequencies, ?_, rfl, matched.limit⟩
  intro row d
  by_cases live : lo ≤ row ∧ row < hi
  · have modD : d.val % 64 = d.val := Nat.mod_eq_of_lt d.isLt
    by_cases rotates : d.val < 32
    · simp [YarnTable.rebuild, rebuild, decoded, live, pad, rotates, YarnTable.buildCell,
        YarnTable.storageWidth, YarnTable.Layout.frequencies, modD, matched.frequencies d]
    · have zero : f.frequencies (d.val % 64) = Value.zero .fp32 := by
        simpa [modD, decoded, pad, rotates] using matched.frequencies d
      simpa [YarnTable.rebuild, rebuild, decoded, live, pad, rotates] using stationary_factor a f.frequencies row d.val zero
  · simpa [YarnTable.rebuild, rebuild, decoded, live] using matched.factors row d

inductive Step where
  | reset
  | limit (rows : Nat)
  | extend (change : Change)
  | complete

def Step.reference : Step → YarnTable.Step
  | .reset => .reset
  | .limit rows => .limit rows
  | .extend w => .extend (window w)
  | .complete => .complete

def execute (a : YarnTable.Arithmetic) (rows : Nat) (s : RotatingState) (step : Step) : RotatingState :=
  match step with
  | .reset => rebuild a 0 rows { s with frequencies := fun d => a.initial d.val }
  | .limit n => { s with rebuildRows := n }
  | .extend w =>
    rebuild a 0 (min rows s.rebuildRows)
      { s with frequencies := fun d => YarnTable.rescale Config.ieeeValue (window w) (s.frequencies d) }
  | .complete =>
    if s.validRows < rows then rebuild a s.validRows rows s else s

theorem execute_matches (a : YarnTable.Arithmetic) (rows : Nat) {f : YarnTable.State} {s : RotatingState}
    (h : Matches f s) (step : Step) :
    Matches (YarnTable.execute true (arithmetic a) .wide rows f step.reference) (execute a rows s step) := by
  cases step with
  | reset =>
    apply rebuilt_matches
    exact ⟨fun d => by simp [decoded, pad, arithmetic], h.factors, h.valid, h.limit⟩
  | limit n => exact ⟨h.frequencies, h.factors, h.valid, rfl⟩
  | extend w =>
    simp only [execute, Step.reference, YarnTable.execute, h.limit]
    apply rebuilt_matches
    refine ⟨?_, h.factors, h.valid, rfl⟩
    intro d
    have original := h.frequencies d
    change YarnTable.rescale Config.ieeeValue (window w) (f.frequencies d.val) = _
    rw [original]
    by_cases rotates : d.val < 32
    · simp [decoded, pad, rotates]
    · simp [decoded, pad, rotates, window_keeps_positive_zero]
  | complete =>
    simp only [Step.reference, execute, YarnTable.execute, h.valid]
    split
    · exact rebuilt_matches a _ _ h
    · exact h

theorem fold_matches (a : YarnTable.Arithmetic) (rows : Nat) (steps : List Step)
    {f : YarnTable.State} {s : RotatingState} (h : Matches f s) :
    Matches ((steps.map Step.reference).foldl (YarnTable.execute true (arithmetic a) .wide rows) f)
      (steps.foldl (execute a rows) s) := by
  induction steps generalizing f s with
  | nil => exact h
  | cons step rest ih =>
    exact ih (execute_matches a rows h step)

structure Observation where
  frequencies : Fin 128 → F32
  factors : Nat → Fin 128 → Factors
  validRows : Nat
  rebuildRows : Nat

def forwardInputs (table : Observation) (qk : Nat → BF16) : RopeShift.Inputs :=
  fun buffer address =>
    let halfword := address / 2
    let entry := table.factors (halfword / 128) ⟨halfword % 128, Nat.mod_lt _ (by decide)⟩
    match buffer with
    | .qk => qk address
    | .f1 => entry.1
    | .f2 => entry.2

def backwardInputs {n : RopeShift.Length} (table : Observation) (inputs : RopeBackward.Inputs n) :
    RopeBackward.Inputs n :=
  { inputs with
    factor1 := fun row channel => (table.factors row.val channel).1
    factor2 := fun row channel => (table.factors row.val channel).2 }

def observe (s : YarnTable.State) : Observation :=
  ⟨fun d => s.frequencies d.val, fun row d => s.factors row d.val, s.validRows, s.rebuildRows⟩

def reference (a : YarnTable.Arithmetic) (rows : Nat) (s : RotatingState) (steps : List Step) : Observation :=
  observe (YarnTable.reference (arithmetic a) .wide rows (decoded s) (steps.map Step.reference))

def candidate (a : YarnTable.Arithmetic) (rows : Nat) (s : RotatingState) (steps : List Step) : Observation :=
  observe (YarnTable.expanded (decoded (steps.foldl (execute a rows) s)))

theorem observed_equal (a : YarnTable.Arithmetic) (rows : Nat) (s : RotatingState) (steps : List Step) :
    candidate a rows s steps = reference a rows s steps := by
  have related := fold_matches a rows steps (decoded_matches s)
  unfold candidate reference
  rw [← YarnTable.observed_equal]
  simp only [YarnTable.observed, observe, YarnTable.expanded, Observation.mk.injEq]
  refine ⟨?_, ?_, related.valid.symm, related.limit.symm⟩
  · funext d
    exact (related.frequencies ⟨d.val / 2, by have := d.isLt; omega⟩).symm
  · funext row d
    exact congrArg (fun pair => YarnTable.expandPair pair (decide (d.val % 2 = 1)))
      (related.factors row ⟨d.val / 2, by have := d.isLt; omega⟩).symm

theorem default_rank_storage :
    YarnTable.factorBytes true .narrow 262144 + YarnTable.factorBytes true .paired 262144
      + 2 * 2 * 262144 * 32 = 128 * 2 ^ 20 := by decide

#axiomsClean window_keeps_positive_zero
#axiomsClean positive_zero_angle
#axiomsClean stationary_factor
#axiomsClean aligned_tile_live
#axiomsClean factor_address_in_bounds
#axiomsClean execute_matches
#axiomsClean observed_equal
#axiomsClean default_rank_storage
end VeriTile.Bench.Models.NanoGPT.Track1.YarnStationary
