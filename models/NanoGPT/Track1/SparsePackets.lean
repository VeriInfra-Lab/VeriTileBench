import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit
import models.NanoGPT.Track1.SourceAnchors
import Mathlib.Data.List.FinRange

/-! The local gradient and outgoing wire retain the original want ordering.
Each point copies an FP16 word or performs the original n-gram wire cast. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.SparsePackets
open VeriTile.Triton FP

abbrev Channel := Fin 768
abbrev FP16Rows := Nat → Channel → UInt16
abbrev BF16Rows := Nat → Channel → Value .bf16

structure Cut where
  rows : Nat
  lo : Nat
  hi : Nat
  ordered : lo ≤ hi
  within : hi ≤ rows

def Cut.own (c : Cut) : Nat := c.hi - c.lo
def Cut.peers (c : Cut) : Nat := c.rows - c.own

def ownSlot (c : Cut) (i : Fin c.own) : Fin c.rows :=
  ⟨c.lo + i.val, by have := i.isLt; have := c.ordered; have := c.within; unfold Cut.own at *; omega⟩

def stagedPeerSlot (c : Cut) (i : Fin c.peers) : Fin c.rows :=
  ⟨if i.val < c.lo then i.val else c.hi + (i.val - c.lo), by
    have := i.isLt
    have := c.ordered
    have := c.within
    unfold Cut.peers Cut.own at *
    split <;> omega⟩

def peerSlot (c : Cut) (i : Fin c.peers) : Fin c.rows :=
  ⟨i.val + (if i.val < c.lo then 0 else c.own), by
    have := i.isLt
    have := c.ordered
    have := c.within
    unfold Cut.peers Cut.own at *
    split <;> omega⟩

theorem peer_slot_exact (c : Cut) (i : Fin c.peers) : peerSlot c i = stagedPeerSlot c i := by
  apply Fin.ext
  have := c.ordered
  simp only [peerSlot, stagedPeerSlot, Cut.own]
  split <;> omega

theorem peer_outside_own (c : Cut) (i : Fin c.peers) (j : Fin c.own) :
    peerSlot c i ≠ ownSlot c j := by
  have := j.isLt
  have := c.ordered
  intro equal
  have values := congrArg Fin.val equal
  simp only [peerSlot, ownSlot, Cut.own] at *
  split at values <;> omega

theorem own_slot_injective (c : Cut) : Function.Injective (ownSlot c) := by
  intro i j same
  exact Fin.ext (Nat.add_left_cancel (congrArg Fin.val same : c.lo + i.val = c.lo + j.val))

theorem slots_cover (c : Cut) (row : Fin c.rows) :
    (∃ i, ownSlot c i = row) ∨ (∃ i, peerSlot c i = row) := by
  have live := row.isLt
  have within := c.within
  have ordered := c.ordered
  by_cases front : row.val < c.lo
  · let slot : Fin c.peers := ⟨row.val, by unfold Cut.peers Cut.own; omega⟩
    exact Or.inr ⟨slot, Fin.ext (by simp [peerSlot, slot, front])⟩
  · by_cases back : c.hi ≤ row.val
    · let slot : Fin c.peers := ⟨row.val - c.own, by unfold Cut.peers Cut.own; omega⟩
      have outside : ¬slot.val < c.lo := by simp only [slot, Cut.own]; omega
      refine Or.inr ⟨slot, Fin.ext ?_⟩
      change slot.val + (if slot.val < c.lo then 0 else c.own) = row.val
      rw [if_neg outside]
      change row.val - (c.hi - c.lo) + (c.hi - c.lo) = row.val
      omega
    · let slot : Fin c.own := ⟨row.val - c.lo, by unfold Cut.own; omega⟩
      exact Or.inl ⟨slot, Fin.ext (by simp only [ownSlot, slot]; omega)⟩

theorem tiled_column (stripe : Fin 3) (lane : Fin 256) : stripe.val * 256 + lane.val < 768 := by
  have := stripe.isLt
  have := lane.isLt
  omega

theorem address_range {rows : Nat} (r : Fin rows) (col : Channel) :
    r.val * 768 + col.val < rows * 768 := by
  have := r.isLt
  have := col.isLt
  omega

theorem element_i32 {rows : Nat} (r : Fin rows) (col : Channel) (safe : rows * 768 < 2 ^ 31) :
    (Int32.ofNat (r.val * 768 + col.val)).toInt = r.val * 768 + col.val := by
  exact Int32.toInt_ofNat_of_lt (Nat.lt_trans (address_range r col) safe)

theorem signed_local_offset (row first : Nat) (liveId : row < 2 ^ 31) (range : first ≤ row) :
    (Int32.ofNat row - Int32.ofNat first).toInt = (row - first : Nat) := by
  rw [Int32.toInt_sub, Int32.toInt_ofNat_of_lt liveId,
    Int32.toInt_ofNat_of_lt (Nat.lt_of_le_of_lt range liveId), ← Int.ofNat_sub range]
  exact Int.bmod_eq_of_le (by omega) (by omega)

theorem packed_lane_coverage (r : Nat) (col : Channel) :
    r / 4 * 4 + r % 4 = r ∧
      col.val / 256 * 256 + col.val % 256 = col.val := by
  omega

theorem lowered_thread_coordinates : ∀ t : Fin 128,
    ((t.val >>> 5) &&& 3) = t.val / 32 ∧
      ((t.val <<< 3) &&& 248) = t.val % 32 * 8 := by
  decide +kernel

theorem lowered_thread_owner (row : Fin 4) (col : Fin 256) :
    let t : Fin 128 := ⟨row.val * 32 + col.val / 8, by have := row.isLt; have := col.isLt; omega⟩
    ((t.val >>> 5) &&& 3) = row.val ∧
      ((t.val <<< 3) &&& 248) + col.val % 8 = col.val := by
  dsimp
  have rows := row.isLt
  have columns := col.isLt
  rw [(lowered_thread_coordinates ⟨row.val * 32 + col.val / 8, by omega⟩).1,
      (lowered_thread_coordinates ⟨row.val * 32 + col.val / 8, by omega⟩).2]
  simp only [Fin.val_mk]
  omega

/-- The FP16 sign, five-bit exponent and ten-bit fraction are decoded
before the exact FP32 widening and the BF16 round to nearest even. -/
def halfToFP32 (x : UInt16) : Value .fp32 :=
  let bits := x.toNat
  let negative := bits ≥ 32768
  let exponent := bits / 1024 % 32
  let fraction := bits % 1024
  if exponent = 31 then
    if fraction = 0 then Value.infinity .fp32 negative else Value.qNaN .fp32
  else
    let significand := if exponent = 0 then fraction else 1024 + fraction
    let scale : Int := if exponent = 0 then -24 else Int.ofNat exponent - 25
    let magnitude := Rat.ofInt significand * FP.pow2 scale
    FP.round .fp32 Config.ieeeValue (if negative then -magnitude else magnitude) negative

def wireCast (x : UInt16) : Value .bf16 :=
  FP.cast .bf16 Config.ieeeValue (halfToFP32 x)

theorem wire_cast_edges :
    wireCast 0x8000 = Value.ofBits .bf16 0x8000 ∧
    wireCast 0x0001 = Value.ofBits .bf16 0x3380 ∧
    wireCast 0x7c00 = Value.ofBits .bf16 0x7f80 ∧
    wireCast 0x7e02 = Value.qNaN .bf16 := by decide +kernel

/-- `value_embed_rows` concatenates one copy of the distinct, in-vocabulary
tokens per plane. The plane offsets preserve both bounds and uniqueness. -/
def valueEmbedRow {planes vocab : Nat} (slot : Fin planes × Fin vocab) : Fin (planes * vocab) :=
  ⟨slot.1.val * vocab + slot.2.val, by
    calc
      _ < (slot.1.val + 1) * vocab := by
        rw [Nat.add_mul, Nat.one_mul]
        exact Nat.add_lt_add_left slot.2.isLt _
      _ ≤ _ := Nat.mul_le_mul_right _ slot.1.isLt⟩

theorem value_embed_row_injective {planes vocab : Nat} :
    Function.Injective (@valueEmbedRow planes vocab) := by
  intro a b equal
  have pos : 0 < vocab := Nat.zero_lt_of_lt a.2.isLt
  have words := congrArg Fin.val equal
  change a.1.val * vocab + a.2.val = b.1.val * vocab + b.2.val at words
  rw [Nat.mul_comm a.1.val, Nat.mul_comm b.1.val] at words
  have plane := congrArg (· / vocab) words
  have token := congrArg (· % vocab) words
  simp only [Nat.mul_add_div pos, Nat.div_eq_of_lt a.2.isLt, Nat.div_eq_of_lt b.2.isLt, Nat.add_zero] at plane
  simp only [Nat.mul_add_mod, Nat.mod_eq_of_lt a.2.isLt, Nat.mod_eq_of_lt b.2.isLt] at token
  exact Prod.ext (Fin.ext plane) (Fin.ext token)

/-- The boolean vocabulary mask collapses repeated tokens before the source
concatenates the per-plane arrays. -/
def valueEmbedWant (planes : Nat) {vocab : Nat} (seen : Fin vocab → Bool) :
    List (Fin (planes * vocab)) :=
  ((List.finRange planes).product ((List.finRange vocab).filter seen)).map valueEmbedRow

theorem value_embed_want_distinct (planes : Nat) {vocab : Nat} (seen : Fin vocab → Bool) :
    Function.Injective (valueEmbedWant planes seen).get := by
  apply List.Nodup.injective_get
  exact ((List.nodup_finRange _).product ((List.nodup_finRange _).filter seen)).map
    value_embed_row_injective

structure ValueInput where
  cut : Cut
  sourceRows : Nat
  want : Fin cut.rows → Fin sourceRows
  distinct : Function.Injective want
  value : FP16Rows

structure NgramInput where
  cut : Cut
  grad : FP16Rows

structure Packets (c : Cut) (Wire : Type) where
  own : Fin c.own → Channel → UInt16
  wire : Fin c.peers → Channel → Wire

def valueStaged (s : ValueInput) : Packets s.cut UInt16 :=
  let compact := fun row => s.value (s.want row).val
  { own := fun i => compact (ownSlot s.cut i)
    wire := fun i => compact (stagedPeerSlot s.cut i) }

def valueDirect (s : ValueInput) : Packets s.cut UInt16 where
  own := fun i => s.value (s.want (ownSlot s.cut i)).val
  wire := fun i => s.value (s.want (peerSlot s.cut i)).val

def ngramStaged (s : NgramInput) : Packets s.cut (Value .bf16) where
  own := fun i => s.grad (ownSlot s.cut i).val
  wire := fun i col => wireCast (s.grad (stagedPeerSlot s.cut i).val col)

def ngramDirect (s : NgramInput) : Packets s.cut (Value .bf16) where
  own := fun i => s.grad (ownSlot s.cut i).val
  wire := fun i col => wireCast (s.grad (peerSlot s.cut i).val col)

theorem value_packets_equal (s : ValueInput) : valueDirect s = valueStaged s := by
  simp only [valueDirect, valueStaged, peer_slot_exact]

theorem ngram_packets_equal (s : NgramInput) : ngramDirect s = ngramStaged s := by
  simp only [ngramDirect, ngramStaged, peer_slot_exact]

/-- The effect of `index_fill_` after widening the original int32 IDs. -/
def cleared (s : ValueInput) : FP16Rows := fun i col =>
  if ∃ slot, (s.want slot).val = i then 0 else s.value i col

theorem cleared_at_want (s : ValueInput) (slot : Fin s.cut.rows) (col : Channel) :
    cleared s (s.want slot).val col = 0 := by
  simp [cleared, show ∃ i, (s.want i).val = (s.want slot).val from ⟨slot, rfl⟩]

theorem preserved_outside_want (s : ValueInput) (i : Nat) (col : Channel)
    (outside : ∀ slot, (s.want slot).val ≠ i) : cleared s i col = s.value i col := by
  simp [cleared, not_exists.mpr outside]

/-- Clear the already-sent wire rows. The owned rows stay live in the
original allocation until the local Adam scatter. -/
def valuePeersCleared (s : ValueInput) : FP16Rows := fun row col =>
  if ∃ i, (s.want (peerSlot s.cut i)).val = row then 0 else s.value row col

def valueOwnedCleared (s : ValueInput) (before : FP16Rows) : FP16Rows := fun row col =>
  if ∃ i, (s.want (ownSlot s.cut i)).val = row then 0 else before row col

theorem value_owned_still_live (s : ValueInput) (i : Fin s.cut.own) (col : Channel) :
    valuePeersCleared s (s.want (ownSlot s.cut i)).val col = s.value (s.want (ownSlot s.cut i)).val col := by
  apply if_neg
  rintro ⟨peer, equal⟩
  exact peer_outside_own _ peer i (s.distinct (Fin.ext equal))

def valueDeferred (s : ValueInput) : Packets s.cut UInt16 where
  own := fun i => valuePeersCleared s (s.want (ownSlot s.cut i)).val
  wire := (valueDirect s).wire

theorem value_deferred_exact (s : ValueInput) : valueDeferred s = valueStaged s := by
  rw [← value_packets_equal]
  change Packets.mk _ _ = Packets.mk _ _
  exact Packets.mk.injEq .. |>.mpr ⟨funext fun i => funext (value_owned_still_live s i), rfl⟩

theorem value_release_exact (s : ValueInput) : valueOwnedCleared s (valuePeersCleared s) = cleared s := by
  funext row col
  by_cases present : ∃ slot, (s.want slot).val = row
  · rw [cleared, if_pos present]
    obtain ⟨slot, h⟩ := present
    rcases slots_cover s.cut slot with ⟨own, ownSlot⟩ | ⟨peer, peerSlot⟩
    · exact if_pos ⟨own, by simpa only [ownSlot] using h⟩
    · unfold valueOwnedCleared
      split
      · rfl
      · exact if_pos ⟨peer, by simpa only [peerSlot] using h⟩
  · have noOwn : ¬∃ i, (s.want (ownSlot s.cut i)).val = row := fun ⟨i, h⟩ => present ⟨_, h⟩
    have noPeer : ¬∃ i, (s.want (peerSlot s.cut i)).val = row := fun ⟨i, h⟩ => present ⟨_, h⟩
    simp only [valueOwnedCleared, valuePeersCleared, cleared, if_neg noOwn, if_neg noPeer, if_neg present]

/-- One slot per warp, with three 256-element stripes and eight FP16
stores per lane and stripe. The kernel masks slots past `want`'s end. -/
def clearSlot (cta : Nat) (thread : Fin 128) : Nat := 4 * cta + ((thread.val >>> 5) &&& 3)

def clearColumn (thread : Fin 128) (stripe : Fin 3) (word : Fin 8) : Nat :=
  stripe.val * 256 + ((thread.val <<< 3) &&& 248) + word.val

theorem clear_tile_covers (slot : Nat) (col : Channel) :
    ∃ cta thread stripe word, clearSlot cta thread = slot ∧ clearColumn thread stripe word = col.val := by
  have h := col.isLt
  let t : Fin 128 := ⟨32 * (slot % 4) + col.val % 256 / 8, by omega⟩
  refine ⟨slot / 4, t, ⟨col.val / 256, by omega⟩, ⟨col.val % 8, by omega⟩, ?_, ?_⟩
  · rw [clearSlot, (lowered_thread_coordinates t).1]
    simp only [t]
    omega
  · rw [clearColumn, (lowered_thread_coordinates t).2]
    simp only [t]
    omega

theorem clear_tile_unique {cta₁ cta₂ : Nat} {t₁ t₂ : Fin 128} {s₁ s₂ : Fin 3} {w₁ w₂ : Fin 8}
    (slot : clearSlot cta₁ t₁ = clearSlot cta₂ t₂)
    (column : clearColumn t₁ s₁ w₁ = clearColumn t₂ s₂ w₂) :
    cta₁ = cta₂ ∧ t₁ = t₂ ∧ s₁ = s₂ ∧ w₁ = w₂ := by
  simp only [clearSlot, (lowered_thread_coordinates _).1] at slot
  simp only [clearColumn, (lowered_thread_coordinates _).2] at column
  simp only [Fin.ext_iff]
  have := t₁.isLt
  have := t₂.isLt
  have := s₁.isLt
  have := s₂.isLt
  have := w₁.isLt
  have := w₂.isLt
  omega

/-- Evaluate the x-only launch at a live destination. The quotient and
remainder select the CTA, warp, stripe and word owning that destination. -/
private def packetCoordinates (slot : Nat) (col : Channel) : Nat × Nat :=
  let thread : Fin 128 := ⟨32 * (slot % 4) + col.val % 256 / 8, by omega⟩
  (clearSlot (slot / 4) thread, clearColumn thread ⟨col.val / 256, by have := col.isLt; omega⟩
    ⟨col.val % 8, by omega⟩)

private theorem packet_coordinates_exact (slot : Nat) (col : Channel) :
    packetCoordinates slot col = (slot, col.val) := by
  let thread : Fin 128 := ⟨32 * (slot % 4) + col.val % 256 / 8, by omega⟩
  let stripe : Fin 3 := ⟨col.val / 256, by have := col.isLt; omega⟩
  let word : Fin 8 := ⟨col.val % 8, by omega⟩
  change (clearSlot (slot / 4) thread, clearColumn thread stripe word) = _
  rw [Prod.mk.injEq, clearSlot, clearColumn, (lowered_thread_coordinates thread).1,
    (lowered_thread_coordinates thread).2]
  dsimp only [thread, stripe, word]
  constructor <;> omega

def rowTiled {rows : Nat} {Word : Type} (read : Fin rows → Channel → Word)
    (slot : Fin rows) (col : Channel) : Word :=
  let cell := packetCoordinates slot.val col
  have correct : cell = (slot.val, col.val) := packet_coordinates_exact slot.val col
  read ⟨cell.1, by rw [correct]; exact slot.isLt⟩ ⟨cell.2, by rw [correct]; exact col.isLt⟩

theorem row_tiled_exact {rows : Nat} {Word : Type} (read : Fin rows → Channel → Word)
    (slot : Fin rows) (col : Channel) : rowTiled read slot col = read slot col := by
  dsimp only [rowTiled]
  congr 1 <;> apply Fin.ext <;> simp [packet_coordinates_exact]

def Packets.tiledWire {c : Cut} {Word : Type} (p : Packets c Word) : Packets c Word :=
  { p with wire := rowTiled p.wire }

theorem Packets.tiledWire_eq {c : Cut} {Word : Type} (p : Packets c Word) : p.tiledWire = p := by
  cases p
  simp only [Packets.tiledWire, Packets.mk.injEq, true_and]
  exact funext fun row => funext fun col => row_tiled_exact _ row col

/-- With one CTA per group of four slots, every live slot is launched. -/
theorem row_tile_launched {rows : Nat} (slot : Fin rows) : slot.val / 4 < (rows + 3) / 4 := by
  have := slot.isLt
  omega

/-- The combined value packet lowers the owned/peer CTA selection to an
unsigned minimum with the wrapped subtraction. Legal launch offsets fit
in 32 bits, so underflow selects the unchanged owned offset. -/
theorem lowered_block_route (block own : Nat) (blockFits : block < 2 ^ 32)
    (ownFits : own < 2 ^ 32) :
    min block ((2 ^ 32 + block - own) % 2 ^ 32) =
      if block < own then block else block - own := by
  by_cases inside : block < own
  · rw [if_pos inside, Nat.mod_eq_of_lt (by omega), min_eq_left (by omega)]
  · have offset : 2 ^ 32 + block - own = (block - own) + 2 ^ 32 := by omega
    rw [if_neg inside, offset, Nat.add_mod, Nat.mod_self, Nat.add_zero, Nat.mod_mod,
      Nat.mod_eq_of_lt (by omega), min_eq_right (Nat.sub_le _ _)]

theorem clear_destinations_unique (s : ValueInput) {a b : Fin s.cut.rows} {x y : Channel}
    (same : (s.want a).val * 768 + x.val = (s.want b).val * 768 + y.val) : a = b ∧ x = y := by
  have := x.isLt
  have := y.isLt
  refine ⟨s.distinct (Fin.ext ?_), Fin.ext ?_⟩ <;> omega

theorem clear_address_i32 (s : ValueInput) (slot : Fin s.cut.rows) (col : Channel)
    (safe : s.sourceRows * 768 < 2 ^ 31) :
    (Int32.ofNat ((s.want slot).val * 768 + col.val)).toInt = (s.want slot).val * 768 + col.val := by
  exact element_i32 (s.want slot) col safe

private def clearStore (s : ValueInput) (memory : FP16Rows) (slot : Fin s.cut.rows) : FP16Rows :=
  fun row col => if (s.want slot).val = row then 0 else memory row col

/-- Enumerate the stores of the masked launch. Slot and channel coverage
gives every listed address, and distinct targets make store order immaterial. -/
def clearDirect (s : ValueInput) : FP16Rows := (List.finRange s.cut.rows).foldl (clearStore s) s.value

private theorem clear_fold (s : ValueInput) (slots : List (Fin s.cut.rows)) (memory : FP16Rows)
    (row : Nat) (col : Channel) :
    slots.foldl (clearStore s) memory row col =
      if ∃ slot ∈ slots, (s.want slot).val = row then 0 else memory row col := by
  induction slots generalizing memory with
  | nil => simp
  | cons slot slots ih =>
    simp only [List.foldl_cons, ih, clearStore]
    by_cases h : (s.want slot).val = row
    · simp [h]
    · simp [List.mem_cons, h]

theorem clear_exact (s : ValueInput) : clearDirect s = cleared s := by
  funext row col
  simp [clearDirect, clear_fold, cleared]

structure Kernels where
  value : (s : ValueInput) → Packets s.cut UInt16
  ngram : (s : NgramInput) → Packets s.cut (Value .bf16)
  clear : ValueInput → FP16Rows

def staged : Kernels := ⟨valueStaged, ngramStaged, cleared⟩
def direct : Kernels := ⟨valueDeferred, ngramDirect, clearDirect⟩

theorem kernels_equal : direct = staged := by
  unfold direct staged
  rw [funext value_deferred_exact, funext ngram_packets_equal, funext clear_exact]

theorem saved_peer_bytes (c : Cut) :
    c.peers * 768 * 2 * 2 = c.peers * 3072 := by omega

theorem saved_owned_bytes (c : Cut) : c.own * 768 * 2 * 2 = c.own * 3072 := by omega

def eventDue (warmup : Bool) (step : Nat) : Bool :=
  if warmup || step < Driver.sparsePeriod4Start then step % 2 == 1 else step % 4 == 3

theorem default_event_count : ((List.range 1194).filter (eventDue false)).length = 382 := by
  decide +kernel

/-- Sending either wire records the local FP16 rows for the owner's merge.
The two sends are issued before the value-gradient reset, including the
case of an empty outgoing wire. The transport uses the existing cuts and
split counts supplied by the live row pulls. -/
structure Runtime (State : Type) where
  readValue : State → Option ValueInput
  sendValue : (s : ValueInput) → Packets s.cut UInt16 → State → Option State
  sendNgram : (s : NgramInput) → Packets s.cut (Value .bf16) → State → Option State
  clearValue : FP16Rows → State → Option State

def launch {State : Type} (r : Runtime State) (pack : Kernels)
    (prepareNgram : State → Option (NgramInput × State)) (state : State) : Option State := do
  let ve ← r.readValue state
  let next ← r.sendValue ve (pack.value ve) state
  let (ng, summed) ← prepareNgram next
  let done ← r.sendNgram ng (pack.ngram ng) summed
  r.clearValue (pack.clear ve) done

theorem launch_equal {State : Type} (r : Runtime State)
    (prepareNgram : State → Option (NgramInput × State)) (state : State) :
    launch r direct prepareNgram state = launch r staged prepareNgram state := by
  rw [kernels_equal]

#axiomsClean Packets.tiledWire_eq
#axiomsClean row_tile_launched
#axiomsClean lowered_block_route
#axiomsClean peer_slot_exact
#axiomsClean peer_outside_own
#axiomsClean element_i32
#axiomsClean lowered_thread_owner
#axiomsClean wire_cast_edges
#axiomsClean value_packets_equal
#axiomsClean value_deferred_exact
#axiomsClean value_release_exact
#axiomsClean ngram_packets_equal
#axiomsClean value_embed_row_injective
#axiomsClean value_embed_want_distinct
#axiomsClean clear_tile_covers
#axiomsClean clear_tile_unique
#axiomsClean clear_destinations_unique
#axiomsClean clear_address_i32
#axiomsClean clear_exact
#axiomsClean cleared_at_want
#axiomsClean preserved_outside_want
#axiomsClean launch_equal
#axiomsClean default_event_count
end VeriTile.Bench.Models.NanoGPT.Track1.SparsePackets
