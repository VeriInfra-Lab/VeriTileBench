import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit
import models.NanoGPT.Track1.SourceAnchors

/-! One full tile updates the two private tail averages from one weight load.
The observed state includes both FP32 buffers and their seed flags. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TailLerp
open VeriTile.Triton FP

abbrev BF16 := Value .bf16
abbrev F32 := Value .fp32

def tileSize : Nat := 4096

structure Lane (tiles : Nat) where
  block : Fin tiles
  thread : Fin 256
  segment : Fin 4
  vector : Fin 4

def Lane.offset {tiles : Nat} (p : Lane tiles) : Nat :=
  p.block.val * tileSize + p.segment.val * 1024 + p.thread.val * 4 + p.vector.val

theorem lane_live {tiles : Nat} (p : Lane tiles) : p.offset < tiles * tileSize := by
  have hb := p.block.isLt
  have ht := p.thread.isLt
  have hs := p.segment.isLt
  have hv := p.vector.isLt
  unfold Lane.offset tileSize
  omega

theorem index_i32 {tiles : Nat} (p : Lane tiles) (safe : tiles * tileSize < 2 ^ 31) :
    (Int32.ofNat p.offset).toInt = p.offset := by
  apply Int32.toInt_ofNat_of_lt
  have := lane_live p
  omega

def owner {tiles : Nat} (i : Fin (tiles * tileSize)) : Lane tiles where
  block := ⟨i.val / 4096, by have := i.isLt; unfold tileSize at this; omega⟩
  thread := ⟨i.val % 1024 / 4, by omega⟩
  segment := ⟨i.val % 4096 / 1024, by omega⟩
  vector := ⟨i.val % 4, by omega⟩

theorem owner_offset {tiles : Nat} (i : Fin (tiles * tileSize)) : (owner i).offset = i.val := by
  dsimp [owner, Lane.offset, tileSize]
  omega

theorem offset_owner {tiles : Nat} (p : Lane tiles) : owner ⟨p.offset, lane_live p⟩ = p := by
  rcases p with ⟨⟨b, hb⟩, ⟨t, ht⟩, ⟨s, hs⟩, ⟨v, hv⟩⟩
  simp only [owner, Lane.offset, tileSize, Lane.mk.injEq, Fin.mk.injEq]
  omega

theorem stores_disjoint {tiles first second : Nat} (p q : Lane tiles)
    (separate : first + 4 * (tiles * tileSize) ≤ second ∨ second + 4 * (tiles * tileSize) ≤ first) :
    first + 4 * p.offset + 3 < second + 4 * q.offset ∨
      second + 4 * q.offset + 3 < first + 4 * p.offset := by
  have hp := lane_live p
  have hq := lane_live q
  omega

structure Cell where
  value : F32
  seeded : Bool

structure Inputs where
  weight : Nat → BF16
  first : Nat → Cell
  second : Nat → Cell

def blend (rate old weight : F32) : F32 :=
  FP.fma Config.ieeeValue rate (FP.sub Config.ieeeValue weight old) old

def tickCell (weight : BF16) (old : Cell) (rate : Option F32) : Cell :=
  match rate with
  | none => old
  | some rate =>
    let weight := FP.cast .fp32 Config.ieeeValue weight
    ⟨if old.seeded then blend rate old.value weight else weight, true⟩

def reference (inputs : Inputs) (firstRate secondRate : Option F32) (i : Nat) : Cell × Cell :=
  (tickCell (inputs.weight i) (inputs.first i) firstRate,
   tickCell (inputs.weight i) (inputs.second i) secondRate)

def fused {tiles : Nat} (inputs : Inputs) (firstRate secondRate : F32) (p : Lane tiles) :
    Cell × Cell :=
  let i := p.offset
  let weight := FP.cast .fp32 Config.ieeeValue (inputs.weight i)
  (⟨blend firstRate (inputs.first i).value weight, true⟩,
   ⟨blend secondRate (inputs.second i).value weight, true⟩)

def candidate (eligible : Bool) (elements : Nat) (inputs : Inputs)
    (firstRate secondRate : Option F32) (i : Nat) : Cell × Cell :=
  if eligible && (inputs.first i).seeded && (inputs.second i).seeded then
    match firstRate, secondRate with
    | some r, some s =>
      if h : i < elements / tileSize * tileSize then fused inputs r s (owner ⟨i, h⟩)
      else reference inputs firstRate secondRate i
    | _, _ => reference inputs firstRate secondRate i
  else reference inputs firstRate secondRate i

theorem candidate_equal (eligible : Bool) (elements : Nat) (inputs : Inputs)
    (firstRate secondRate : Option F32) (i : Nat) :
    candidate eligible elements inputs firstRate secondRate i =
      reference inputs firstRate secondRate i := by
  unfold candidate
  split
  · rename_i both
    have h := Bool.and_eq_true_iff.mp both
    have first := (Bool.and_eq_true_iff.mp h.1).2
    have second := h.2
    cases firstRate <;> cases secondRate <;> simp only
    split
    · simp [fused, reference, tickCell, owner_offset, first, second]
    · rfl
  · rfl

abbrev Update := Bool → Nat → Inputs → Option F32 → Option F32 → Nat → Cell × Cell

inductive Shard
  | lmHead | embedding | qk | vo | mlp | valueEmbedding
  deriving DecidableEq

structure Snapshot where
  step : Nat
  totalSteps : Nat
  valueEmbeddingUpdates : Bool
  elements : Shard → Nat
  eligible : Shard → Bool
  input : Shard → Inputs

abbrev Output := Shard → Nat → Cell × Cell

open Driver in
def emaRate (step total : Nat) : Option F32 :=
  if step + tailEMAWindow < total then none
  else if step % 2 = 0 then
    if step + tailEMAWindow = total then some (Value.ofBits .fp32 tailEMAOneRateBits) else none
  else some (Value.ofBits .fp32 (if step + 1 = total then tailEMAOneRateBits else tailEMATwoRateBits))

open Driver in
def averageRate (step total : Nat) : Option F32 :=
  if step + tailAverageWindow ≥ total && (step + 1) % tailAveragePeriod == total % tailAveragePeriod then
    some (Value.ofBits .fp32 tailAverageRateBits)
  else none

open Driver in
def bankRate (step total : Nat) : Option F32 :=
  if step + tailBankWindow ≥ total && (step % tailBankPeriod == 1 || step + 2 ≥ total) then
    some (Value.ofBits .fp32 tailBankRateBits)
  else none

open Driver in
def valueRate (step total : Nat) (updated : Bool) : Option F32 :=
  if step + tailValueWindow ≥ total && updated then some (Value.ofBits .fp32 tailValueRateBits)
  else none

/-- The first and second cells are the two private averages of a shard.
A slot belonging to no accumulator has an inactive rate. -/
def tick (update : Update) (s : Snapshot) : Output := fun label =>
  let (first, second) :=
    match label with
    | .lmHead | .embedding => (emaRate s.step s.totalSteps, none)
    | .qk => (none, bankRate s.step s.totalSteps)
    | .vo | .mlp => (averageRate s.step s.totalSteps, bankRate s.step s.totalSteps)
    | .valueEmbedding => (valueRate s.step s.totalSteps s.valueEmbeddingUpdates, none)
  update (s.eligible label) (s.elements label) (s.input label) first second

theorem default_overlap : ((List.range 250).filter (fun i =>
    (averageRate (944 + i) 1194).isSome && (bankRate (944 + i) 1194).isSome)).length = 63 := by
  decide +kernel

theorem default_full_shards :
    (2 * 768 ^ 2) % tileSize = 0 ∧ (3 * 2816 * 768) % tileSize = 0 := by decide

#axiomsClean lane_live
#axiomsClean index_i32
#axiomsClean owner_offset
#axiomsClean offset_owner
#axiomsClean stores_disjoint
#axiomsClean candidate_equal
#axiomsClean default_overlap
end VeriTile.Bench.Models.NanoGPT.Track1.TailLerp
