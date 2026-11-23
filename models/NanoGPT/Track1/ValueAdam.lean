import models.NanoGPT.Track1.NgramAdam
import Mathlib.Data.Nat.Sqrt

/-! Value-embedding Adam consumes a dense shard assembled by two ordered FP16
scatters. The scaled gradient has an explicit binary16 store and reload before
the FP32 optimizer operations. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam
open VeriTile.Triton FP
open SparsePackets (Channel)
open NgramAdam (Order)

abbrev F32 := Value .fp32
abbrev BF16 := Value .bf16
abbrev Half := UInt16

private def signWord (negative : Bool) : Nat := if negative then 0x8000 else 0

private def nearestEven (magnitude : Rat) : Nat :=
  let q := magnitude.num.natAbs / magnitude.den
  let r := magnitude.num.natAbs % magnitude.den
  if 2 * r > magnitude.den || (2 * r == magnitude.den && q % 2 == 1) then q + 1 else q

/-- The positive quotient in units of the result exponent is rounded once,
including the subnormal region whose unit is 2^-24. -/
def narrow (x : F32) : Half :=
  match x.decode with
  | .nan .. => 0x7e00
  | .infinity negative => UInt16.ofNat (signWord negative + 0x7c00)
  | .finite negative magnitude =>
    let sign := signWord negative
    if magnitude == 0 then UInt16.ofNat sign else
      let estimate : Int := (magnitude.num.natAbs.log2 : Int) - magnitude.den.log2
      let exponent := if magnitude < pow2 estimate then estimate - 1 else estimate
      let unit := max exponent (-14) - 10
      let n := nearestEven (magnitude / pow2 unit)
      let carry := n ≥ 2048
      let exponent := unit + 10 + (if carry then 1 else 0)
      let n := if carry then n / 2 else n
      if exponent > 15 then UInt16.ofNat (sign + 0x7c00)
      else if n < 1024 then UInt16.ofNat (sign + n)
      else UInt16.ofNat (sign + (exponent + 15).toNat * 1024 + n - 1024)

def halfAdd (a b : Half) : Half :=
  narrow (FP.add Config.ieeeValue (SparsePackets.halfToFP32 a) (SparsePackets.halfToFP32 b))

def halfScale (factor : F32) (a : Half) : Half :=
  narrow (FP.mul Config.ieeeValue (SparsePackets.halfToFP32 a) factor)

theorem subnormal_scale_edges :
    let eighth := Value.ofBits .fp32 0x3e000000
    halfScale eighth 1 = 0 ∧ halfScale eighth 4 = 0 ∧
      halfScale eighth 12 = 2 ∧ halfScale eighth 0x8001 = 0x8000 := by
  decide +kernel

theorem scaling_requires_half_store :
    let eighth := Value.ofBits .fp32 0x3e000000
    SparsePackets.halfToFP32 (halfScale eighth 1) ≠
      FP.mul Config.ieeeValue (SparsePackets.halfToFP32 1) eighth := by
  decide +kernel

structure Pull where
  ids : AdamEntries.Input
  ownGradient : Fin ids.localCount → Channel → Half
  receivedGradient : Fin ids.peerCount → Channel → Half

/-- The physical view of `grad_accum[self.own_rows]`. The sorted row pull
selects the owned interval before constructing the device ID array. -/
structure SourceSlice where
  input : SparsePackets.ValueInput
  first : Nat
  shardRows : Nat
  asked : Array Int32
  selected : ∀ i : Fin input.cut.own,
    let row := (input.want (SparsePackets.ownSlot input.cut i)).val
    first ≤ row ∧ row < first + shardRows
  signed : input.sourceRows < 2 ^ 31

def SourceSlice.ids (s : SourceSlice) : AdamEntries.Input where
  want := Array.ofFn fun slot : Fin s.input.cut.rows => Int32.ofNat (s.input.want slot).val
  asked := s.asked
  lo := s.input.cut.lo
  hi := s.input.cut.hi
  first := Int32.ofNat s.first
  ordered := s.input.cut.ordered
  valid := by simpa using s.input.cut.within

def SourceSlice.globalRow (s : SourceSlice) (i : Fin s.input.cut.own) : Nat :=
  (s.input.want (SparsePackets.ownSlot s.input.cut i)).val

def SourceSlice.localRow (s : SourceSlice) (i : Fin s.input.cut.own) : Nat :=
  (AdamEntries.localRead s.ids i).toInt.toNat

theorem SourceSlice.local_row_exact (s : SourceSlice) (i : Fin s.input.cut.own) :
    s.localRow i = s.globalRow i - s.first := by
  have fits : s.globalRow i < 2 ^ 31 :=
    Nat.lt_trans (s.input.want (SparsePackets.ownSlot s.input.cut i)).isLt s.signed
  have first : s.first ≤ s.globalRow i := (s.selected i).1
  simp only [SourceSlice.localRow, AdamEntries.localRead, SourceSlice.ids, Array.getElem_ofFn]
  change (Int32.ofNat (s.globalRow i) - Int32.ofNat s.first).toInt.toNat = _
  rw [SparsePackets.signed_local_offset _ _ fits first, Int.toNat_natCast]

theorem SourceSlice.address_exact (s : SourceSlice) (i : Fin s.input.cut.own) (col : Channel) :
    s.localRow i * 768 + col.val < s.shardRows * 768 ∧
    s.first * 768 + (s.localRow i * 768 + col.val) = s.globalRow i * 768 + col.val := by
  have bounds := s.selected i
  change s.first ≤ s.globalRow i ∧ s.globalRow i < s.first + s.shardRows at bounds
  rw [s.local_row_exact i]
  have := col.isLt
  omega

theorem SourceSlice.address_unique (s : SourceSlice) {a b : Fin s.input.cut.own} {x y : Channel}
    (same : s.localRow a * 768 + x.val = s.localRow b * 768 + y.val) : a = b ∧ x = y := by
  have shifted := congrArg (s.first * 768 + ·) same
  change s.first * 768 + (s.localRow a * 768 + x.val) =
    s.first * 768 + (s.localRow b * 768 + y.val) at shifted
  rw [(s.address_exact a x).2, (s.address_exact b y).2] at shifted
  have both := SparsePackets.clear_destinations_unique s.input
    (a := SparsePackets.ownSlot s.input.cut a) (b := SparsePackets.ownSlot s.input.cut b) shifted
  exact ⟨SparsePackets.own_slot_injective _ both.1, both.2⟩

def SourceSlice.stagedPull (s : SourceSlice) (peer : Fin s.asked.size → Channel → Half) : Pull :=
  ⟨s.ids, (SparsePackets.valueStaged s.input).own, peer⟩

def SourceSlice.deferredPull (s : SourceSlice) (peer : Fin s.asked.size → Channel → Half) : Pull :=
  ⟨s.ids, fun i => SparsePackets.valuePeersCleared s.input (s.first + s.localRow i), peer⟩

theorem SourceSlice.pull_exact (s : SourceSlice) (peer : Fin s.asked.size → Channel → Half) :
    s.deferredPull peer = s.stagedPull peer := by
  suffices own : (fun i : Fin s.input.cut.own =>
      SparsePackets.valuePeersCleared s.input (s.first + s.localRow i)) = (SparsePackets.valueStaged s.input).own by
    exact congrArg (fun values : Fin s.input.cut.own → Channel → Half => Pull.mk s.ids values peer) own
  funext i col
  have first : s.first ≤ s.globalRow i := (s.selected i).1
  rw [s.local_row_exact i, Nat.add_sub_of_le first]
  exact SparsePackets.value_owned_still_live s.input i col

structure Schedule (p : Pull) where
  localAtomics : Channel → Order p.ids.localCount
  peerAtomics : Channel → Order p.ids.peerCount

abbrev HalfShard := Nat → Channel → Half

def mergeFrom (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (before : HalfShard) : HalfShard :=
  fun row col =>
    let localSum := (order.localAtomics col).slots.foldl (fun acc i =>
      if (idx.localRows i).toInt = row then halfAdd acc (p.ownGradient i col) else acc) (before row col)
    (order.peerAtomics col).slots.foldl (fun acc i =>
      if (idx.peerRows i).toInt = row then halfAdd acc (p.receivedGradient i col) else acc) localSum

def merge (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p) : HalfShard :=
  mergeFrom p idx order (fun _ _ => 0)

def marked (p : Pull) (idx : AdamEntries.Indices p.ids) (row : Nat) : Bool :=
  (List.finRange p.ids.localCount).any (fun i => decide ((idx.localRows i).toInt = row)) ||
    (List.finRange p.ids.peerCount).any (fun i => decide ((idx.peerRows i).toInt = row))

private theorem foldFlags {α : Type} (slots : List α) (hit : α → Bool) (flag : Bool) :
    slots.foldl (fun bit i => bit || hit i) flag = (flag || slots.any hit) := by
  induction slots generalizing flag with
  | nil => simp
  | cons i rest ih => simp [List.foldl_cons, ih, Bool.or_assoc]

theorem scatter_flags_exact (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (row : Nat) (ownLane peerLane : Channel) :
    let own := (order.localAtomics ownLane).slots.foldl
      (fun flag i => flag || decide ((idx.localRows i).toInt = row)) false
    (order.peerAtomics peerLane).slots.foldl
      (fun flag i => flag || decide ((idx.peerRows i).toInt = row)) own =
      marked p idx row := by
  dsimp only
  rw [foldFlags, foldFlags, (order.localAtomics ownLane).allSlots.any_eq,
    (order.peerAtomics peerLane).allSlots.any_eq]
  rfl

theorem marked_iff (p : Pull) (idx : AdamEntries.Indices p.ids) (row : Nat) :
    marked p idx row = true ↔
      (∃ i, (idx.localRows i).toInt = row) ∨ (∃ i, (idx.peerRows i).toInt = row) := by
  simp [marked, List.any_eq_true]

private theorem skipFold {n : Nat} (read : Fin n → Int32) (grad : Fin n → Half) (row : Nat)
    (miss : ∀ i, (read i).toInt ≠ row) (slots : List (Fin n)) (start : Half) :
    slots.foldl (fun acc i => if (read i).toInt = row then halfAdd acc (grad i) else acc) start = start := by
  induction slots with
  | nil => rfl
  | cons i tail ih => simp [List.foldl_cons, miss i, ih]

theorem unmarked_zero (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (row : Nat) (col : Channel) (untouched : marked p idx row = false) :
    merge p idx order row col = 0 := by
  have misses : ¬((∃ i, (idx.localRows i).toInt = row) ∨ (∃ i, (idx.peerRows i).toInt = row)) := by
    intro present
    have hit := (marked_iff p idx row).mpr present
    simp [untouched] at hit
  have own : ∀ i, (idx.localRows i).toInt ≠ row := fun i h => misses (.inl ⟨i, h⟩)
  have peer : ∀ i, (idx.peerRows i).toInt ≠ row := fun i h => misses (.inr ⟨i, h⟩)
  unfold merge mergeFrom
  rw [skipFold idx.peerRows (fun i => p.receivedGradient i col) row peer,
      skipFold idx.localRows (fun i => p.ownGradient i col) row own]

abbrev Normalize :=
  (p : Pull) → AdamEntries.Indices p.ids → Schedule p → F32 → Nat → HalfShard

def dense : Normalize := fun p idx order factor _ row col =>
  halfScale factor (merge p idx order row col)

def eighth : F32 := Value.ofBits .fp32 0x3e000000

def selectedFrom (before : HalfShard) : Normalize := fun p idx order factor rows row col =>
  let value := mergeFrom p idx order before row col
  if factor = eighth ∧ 2 * p.ids.count ≤ rows ∧ !(marked p idx row) then value
  else halfScale factor value

def selected : Normalize := selectedFrom (fun _ _ => 0)

theorem eighth_preserves_zero : halfScale eighth 0 = 0 := by decide +kernel

/-- The marker view is aligned after the dense FP16 prefix. Each flag
occupies four bytes inside the single allocation. -/
theorem marker_layout {rows : Nat} (row : Fin rows) :
    (2 * (rows * 768)) % 16 = 0 ∧
      2 * (rows * 768) + 4 * (row.val + 1) ≤ 2 * (rows * 770) := by
  have := row.isLt
  omega

theorem normalization_equal : selected = dense := by
  funext p idx order factor rows row col
  unfold selected selectedFrom dense
  change (if _ then merge p idx order row col else _) = _
  split
  · rename_i skip
    obtain ⟨rfl, _, h⟩ := skip
    have untouched : marked p idx row = false := by simpa using h
    rw [unmarked_zero p idx order row col untouched, eighth_preserves_zero]
  · rfl

/-- Clearing flagged rows writes positive zero and resets their consumed flags.
Already-clear flags and their gradient words are preserved. -/
def clearMarked (flags : Nat → Bool) (gradient : HalfShard) : HalfShard × (Nat → Bool) :=
  (fun row col => if flags row then 0 else gradient row col,
   fun row => if flags row then false else flags row)

/-- Hopper emits four warps per CTA, three stripes per row, and eight
FP16 stores per lane and stripe. -/
def clearCoordinate (cta : Nat) (thread : Fin 128) (stripe : Fin 3) (half : Fin 8) : Nat × Nat :=
  (4 * cta + thread.val / 32, 256 * stripe.val + 8 * (thread.val % 32) + half.val)

theorem clear_coordinate_covers (row : Nat) (col : Channel) :
    ∃ cta thread stripe half, clearCoordinate cta thread stripe half = (row, col.val) := by
  have h := col.isLt
  refine ⟨row / 4, ⟨32 * (row % 4) + col.val % 256 / 8, by omega⟩,
    ⟨col.val / 256, by omega⟩, ⟨col.val % 8, by omega⟩, ?_⟩
  simp only [clearCoordinate, Prod.mk.injEq]
  omega

theorem clear_coordinate_unique {cta₁ cta₂ : Nat} {thread₁ thread₂ : Fin 128}
    {stripe₁ stripe₂ : Fin 3} {half₁ half₂ : Fin 8}
    (h : clearCoordinate cta₁ thread₁ stripe₁ half₁ = clearCoordinate cta₂ thread₂ stripe₂ half₂) :
    cta₁ = cta₂ ∧ thread₁ = thread₂ ∧ stripe₁ = stripe₂ ∧ half₁ = half₂ := by
  simp only [clearCoordinate, Prod.mk.injEq] at h
  simp only [Fin.ext_iff]
  have := thread₁.isLt
  have := thread₂.isLt
  have := stripe₁.isLt
  have := stripe₂.isLt
  have := half₁.isLt
  have := half₂.isLt
  omega

theorem marked_event_clears_to_zero (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (rows : Nat) :
    clearMarked (marked p idx) (selected p idx order eighth rows) = (fun _ _ => 0, fun _ => false) := by
  refine Prod.ext ?_ ?_
  · funext row col
    unfold clearMarked
    dsimp only
    cases h : marked p idx row
    · simp [h, normalization_equal, dense, unmarked_zero p idx order row col h, eighth_preserves_zero]
    · simp [h]
  · funext row
    cases h : marked p idx row <;> simp [clearMarked, h]

/-- The next merge has the same inputs to its first atomic update after a
remembered clear as after an unconditional full clear. -/
theorem next_merge_after_marked (previous p : Pull) (oldIdx : AdamEntries.Indices previous.ids)
    (oldOrder : Schedule previous) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (rows : Nat) :
    mergeFrom p idx order (clearMarked (marked previous oldIdx)
      (selected previous oldIdx oldOrder eighth rows)).1 = merge p idx order := by
  rw [marked_event_clears_to_zero]
  rfl

def afterMarked (previous : Pull) (idx : AdamEntries.Indices previous.ids) (order : Schedule previous)
    (rows : Nat) : Normalize :=
  selectedFrom (clearMarked (marked previous idx) (selected previous idx order eighth rows)).1

theorem after_marked_equal (previous : Pull) (idx : AdamEntries.Indices previous.ids)
    (order : Schedule previous) (rows : Nat) : afterMarked previous idx order rows = selected := by
  unfold afterMarked
  rw [marked_event_clears_to_zero]
  rfl

/-- The prior event and its write schedule determine the saved gradient and
flags. `none` covers the full-clear path and a freshly zeroed workspace. -/
structure PriorMarked where
  pull : Pull
  indices : AdamEntries.Indices pull.ids
  order : Schedule pull
  rows : Nat

abbrev WorkspaceNormalize := Option PriorMarked → Normalize

def clearDense : WorkspaceNormalize := fun _ => dense

def clearSelected : WorkspaceNormalize := fun
  | none => selected
  | some prev => afterMarked prev.pull prev.indices prev.order prev.rows

theorem workspace_normalization_equal : clearSelected = clearDense := by
  funext prior
  cases prior with
  | none => exact normalization_equal
  | some prev => exact (after_marked_equal prev.pull prev.indices prev.order prev.rows).trans normalization_equal

structure Scalars where
  events : Nat
  rate : Float
  weightDecay : Float
  eps : Float
  worldSize : Nat

structure Launch where
  event : Nat
  beta1 : F32
  oneMinusBeta1 : F32
  beta2 : F32
  oneMinusBeta2 : F32
  eps : F32
  stepSize : F32
  decay : F32
  gradientScale : F32

private def f32 (x : Float) : F32 := Value.ofBits .fp32 x.toFloat32.toBits.toNat

def configure (step : Nat) (s : Scalars) : Launch :=
  let t := s.events + 1
  let lr := s.rate * Float.ofBits Driver.valueRateScale
  let period4 := step ≥ Driver.sparsePeriod4Start
  let b1 := Float.ofBits Driver.valueBeta1
  let b2 := Float.ofBits Driver.valueBeta2
  let beta1 := if period4 then b1.pow 2.0 else b1
  let beta2 := if period4 then b2.pow 2.0 else b2
  let wdScale := Float.ofBits (if period4 then Driver.valuePeriod4WeightDecayScale else Driver.valueWeightDecayScale)
  let bias1 := 1.0 - beta1.pow t.toFloat
  let bias2 := 1.0 - beta2.pow t.toFloat
  ⟨t, f32 beta1, f32 (1.0 - beta1), f32 beta2, f32 (1.0 - beta2), f32 s.eps,
    f32 (lr * (bias2.pow 0.5 / bias1)), f32 (lr * lr * s.weightDecay * wdScale),
    f32 (1.0 / s.worldSize.toFloat)⟩

structure Cell where
  weight : BF16
  first : F32
  second : F32

/-- The two default Hopper pointwise configurations for the value shard. -/
inductive Tile where
  | x512
  | x1024

/-- `sqrt.rn.f32` in the software value profile. Scale the radicand so that
the rounded root significand is an integer in units of the result's ulp.
Squaring its exact half-integer midpoint decides rounding without a
host square root or intermediate floating-point rounding. -/
def sqrtRN (x : F32) : F32 :=
  match x.decode with
  | .nan .. | .infinity true => Value.qNaN .fp32
  | .infinity false => x
  | .finite negative magnitude =>
    if magnitude == 0 then x else if negative then Value.qNaN .fp32 else
      let estimate : Int := (magnitude.num.natAbs.log2 : Int) - magnitude.den.log2
      let exponent := if magnitude < pow2 estimate then estimate - 1 else estimate
      let unit := exponent / 2 - 23
      let radicand := magnitude / pow2 (2 * unit)
      let below := Nat.sqrt (radicand.num.natAbs / radicand.den)
      let midpoint := Rat.ofInt below + (1 / 2 : Rat)
      let cutoff := midpoint * midpoint
      let nearest := if radicand > cutoff || (radicand == cutoff && below % 2 == 1)
        then below + 1 else below
      FP.round .fp32 Config.ieeeValue (Rat.ofInt nearest * pow2 unit)

theorem sqrt_rn_edges :
    sqrtRN (Value.ofBits .fp32 1) = Value.ofBits .fp32 0x1a3504f3 ∧
    sqrtRN (Value.ofBits .fp32 0x40000000) = Value.ofBits .fp32 0x3fb504f3 ∧
    sqrtRN (Value.ofBits .fp32 0x7f7fffff) = Value.ofBits .fp32 0x5f7fffff ∧
    sqrtRN (Value.zero .fp32 true) = Value.zero .fp32 true ∧
    sqrtRN (Value.ofBits .fp32 0xbf800000) = Value.qNaN .fp32 := by
  decide +kernel

/-- `div.full.f32` is the remaining native scalar effect. Both source
executions use that operation. The tile selects a launch geometry. -/
structure Arithmetic where
  tile : Launch → Tile
  divideFull : F32 → F32 → F32

def update (ar : Arithmetic) (launch : Launch) (_index : Nat) (old : Cell) (input : Half) : Cell :=
  let g := SparsePackets.halfToFP32 input
  let p := FP.cast .fp32 Config.ieeeValue old.weight
  let c := Config.ieeeValue
  let first := FP.fma c old.first launch.beta1 (FP.mul c g launch.oneMinusBeta1)
  let second := FP.fma c old.second launch.beta2 (FP.mul c (FP.mul c g g) launch.oneMinusBeta2)
  let direction := ar.divideFull first (FP.add c (sqrtRN second) launch.eps)
  let step := FP.mul c launch.stepSize direction
  let gate := FP.lt c (Value.zero .fp32) (FP.mul c step p)
  let mask := if gate then Value.ofBits .fp32 0x3f800000 else Value.zero .fp32
  let masked := FP.mul c p mask
  let delta := FP.fma c launch.decay masked step
  ⟨FP.cast .bf16 c (FP.sub c p delta), first, second⟩

theorem tile_update_equal (left right : Launch → Tile) (divide : F32 → F32 → F32)
    (launch : Launch) (index : Nat) (old : Cell) (input : Half) :
    update ⟨left, divide⟩ launch index old input = update ⟨right, divide⟩ launch index old input :=
  rfl

structure Memory where
  rows : Nat
  read : Fin rows → Channel → Cell
  prior : Option PriorMarked

structure Output (memory : Memory) where
  gradient : Fin memory.rows → Channel → Half
  updated : Fin memory.rows → Channel → Cell

def event (norm : Normalize) (p : Pull) (idx : AdamEntries.Indices p.ids) (order : Schedule p)
    (ar : Arithmetic) (launch : Launch) (memory : Memory) : Output memory :=
  let grad := norm p idx order launch.gradientScale memory.rows
  let scaled := fun (row : Fin memory.rows) col => grad row.val col
  ⟨scaled, fun row col => update ar launch (row.val * 768 + col.val) (memory.read row col) (scaled row col)⟩

abbrev EventKernel :=
  (p : Pull) → AdamEntries.Indices p.ids → Schedule p → Arithmetic → Launch →
    (memory : Memory) → Output memory

theorem event_equal (p : Pull) (order : Schedule p) (ar : Arithmetic) (launch : Launch)
    (memory : Memory) :
    event selected p (AdamEntries.fused p.ids) order ar launch memory =
      event dense p (AdamEntries.staged p.ids) order ar launch memory := by
  rw [AdamEntries.pack_equal, normalization_equal]

theorem remembered_event_equal (previous p : Pull) (idx : AdamEntries.Indices previous.ids)
    (oldOrder : Schedule previous) (order : Schedule p) (ar : Arithmetic) (launch : Launch)
    (memory : Memory) :
    event (afterMarked previous idx oldOrder memory.rows) p (AdamEntries.fused p.ids) order ar launch memory =
      event dense p (AdamEntries.staged p.ids) order ar launch memory := by
  rw [after_marked_equal, event_equal]

theorem source_event_exact (s : SourceSlice) (peer : Fin s.asked.size → Channel → Half)
    (schedule : (p : Pull) → Schedule p) (ar : Arithmetic) (launch : Launch) (memory : Memory) :
    let candidate := s.deferredPull peer
    let reference := s.stagedPull peer
    (SparsePackets.valueOwnedCleared s.input (SparsePackets.valuePeersCleared s.input),
      event (clearSelected memory.prior) candidate (AdamEntries.fused candidate.ids)
        (schedule candidate) ar launch memory) =
    (SparsePackets.cleared s.input,
      event dense reference (AdamEntries.staged reference.ids) (schedule reference) ar launch memory) := by
  intro candidate reference
  have exactPull : candidate = reference := s.pull_exact peer
  rw [exactPull, SparsePackets.value_release_exact, workspace_normalization_equal, clearDense, AdamEntries.pack_equal]

/-- Read the inputs after draining the existing gradient exchange. The
commit applies the scaled gradient and optimizer weights and moments to
the original shard and records the incremented Adam count. -/
structure Runtime (State : Type) where
  wait : State → Option (Pull × State)
  readAdam : State → Option (Scalars × Memory)
  order : (p : Pull) → State → Option (Schedule p)
  arithmetic : Arithmetic
  commit : (memory : Memory) → Launch → Output memory → State → Option State

def execute {State : Type} (r : Runtime State) (entries : AdamEntries.Pack) (runEvent : EventKernel)
    (step : Nat) (state : State) :
    Option State := do
  let (pull, ready) ← r.wait state
  let (scalars, memory) ← r.readAdam ready
  let order ← r.order pull ready
  let launch := configure step scalars
  r.commit memory launch (runEvent pull (entries pull.ids) order r.arithmetic launch memory) ready

#axiomsClean subnormal_scale_edges
#axiomsClean scaling_requires_half_store
#axiomsClean sqrt_rn_edges
#axiomsClean tile_update_equal
#axiomsClean unmarked_zero
#axiomsClean scatter_flags_exact
#axiomsClean marker_layout
#axiomsClean normalization_equal
#axiomsClean clear_coordinate_covers
#axiomsClean clear_coordinate_unique
#axiomsClean marked_event_clears_to_zero
#axiomsClean next_merge_after_marked
#axiomsClean workspace_normalization_equal
#axiomsClean event_equal
#axiomsClean remembered_event_equal
#axiomsClean SourceSlice.address_unique
#axiomsClean source_event_exact
end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam
