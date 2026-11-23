import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.ScalarOps

/-! The Hopper QK epilogue for the long-window layers. Its 128-channel
RMS expression folds eight squared values per lane and then uses four
butterfly additions. A stationary key channel reads the result from the
previous token in the reference. The candidate writes that result forward
from the producing token. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.RopeShift
open VeriTile.Triton FP

inductive Length where | t16k | t32k | t40k | t48k | validation
  deriving DecidableEq

def Length.tokens : Length → Nat
  | .t16k => 16384 | .t32k => 32768 | .t40k => 40960 | .t48k => 49152 | .validation => 262144

inductive Plane where | query | key
  deriving DecidableEq

def Plane.base : Plane → Nat | .query => 0 | .key => 6

abbrev Head := Fin 6
abbrev Channel := Fin 128

structure Point (n : Length) where
  token : Fin n.tokens
  plane : Plane
  head : Head
  channel : Channel
  deriving DecidableEq

structure Thread (n : Length) where
  block : Fin (n.tokens * 3)
  lane : Fin 64
  reg : Fin 8

def Thread.row {n : Length} (t : Thread n) : Nat := 4 * t.block.val + t.lane.val / 16

def plane (row : Nat) : Plane := if row % 12 < 6 then .query else .key

def cell {n : Length} (t : Thread n) : Point n :=
  ⟨⟨t.row / 12, by
      have := t.block.isLt
      have := t.lane.isLt
      unfold Thread.row
      omega⟩,
    plane t.row, ⟨t.row % 12 % 6, Nat.mod_lt _ (by decide)⟩,
    ⟨8 * (t.lane.val % 16) + t.reg.val, by have := t.reg.isLt; omega⟩⟩

def packedRow {n : Length} (p : Point n) : Nat := 12 * p.token.val + p.plane.base + p.head.val

def owner {n : Length} (p : Point n) : Thread n :=
  ⟨⟨packedRow p / 4, by
      have ht := p.token.isLt
      have hh := p.head.isLt
      cases p with
      | mk tok pk hd d => cases pk <;> simp only [packedRow, Plane.base] at * <;> omega⟩,
    ⟨16 * (packedRow p % 4) + p.channel.val / 8, by have := p.channel.isLt; omega⟩,
    ⟨p.channel.val % 8, Nat.mod_lt _ (by decide)⟩⟩

theorem owner_row {n : Length} (p : Point n) : (owner p).row = packedRow p := by
  have := p.channel.isLt
  unfold owner Thread.row
  dsimp
  omega

theorem owner_covers {n : Length} (p : Point n) : cell (owner p) = p := by
  have hr := owner_row p
  obtain ⟨tok, pk, hd, d⟩ := p
  have hh := hd.isLt
  have hc := d.isLt
  simp only [cell, hr, Point.mk.injEq]
  refine ⟨Fin.ext ?_, ?_, Fin.ext ?_, Fin.ext ?_⟩
  · cases pk <;> simp only [packedRow, Plane.base] <;> omega
  · unfold plane
    cases pk <;> simp only [packedRow, Plane.base] <;> split <;> first | rfl | omega
  · cases pk <;> simp only [packedRow, Plane.base] <;> omega
  · simp only [owner]
    omega

theorem cell_owner_unique {n : Length} (t u : Thread n) (h : cell t = cell u) : t = u := by
  have hτ := congrArg (fun p => p.token.val) h
  have hπ := congrArg Point.plane h
  have hh := congrArg (fun p => p.head.val) h
  have hd := congrArg (fun p => p.channel.val) h
  have hl := t.lane.isLt
  have hl' := u.lane.isLt
  have hr := t.reg.isLt
  have hr' := u.reg.isLt
  simp only [cell, plane] at hτ hπ hh hd
  have rowEq : t.row = u.row := by
    split at hπ <;> split at hπ <;> first | contradiction | omega
  unfold Thread.row at rowEq
  have hb : t.block = u.block := Fin.ext (by omega)
  have hx : t.lane = u.lane := Fin.ext (by omega)
  have hv : t.reg = u.reg := Fin.ext (by omega)
  cases t
  cases u
  congr

/-- The PTX's signed division by 12 is a high multiply and a signed shift.
All row IDs are nonnegative and below the sign bit in the high product. -/
theorem row_division (row : Nat) (h : row < 3145728) :
    (row * 715827883 / 4294967296) / 2 = row / 12 := by omega

theorem logical_head_modulo (h : Fin 12) :
    h.val - ((h.val * 43 / 256 + h.val * 43 / 32768) * 6) = h.val % 6 := by
  decide +kernel +revert

private def compiledRow {n : Length} (t : Thread n) : Nat :=
  ((t.lane.val >>> 4) &&& 3) ||| (t.block.val <<< 2)

private def vectorStart {n : Length} (t : Thread n) : Nat :=
  (t.lane.val <<< 3) &&& 120

private theorem lane_fields (lane : Fin 64) :
    ((lane.val >>> 4) &&& 3) = lane.val / 16 ∧
      ((lane.val <<< 3) &&& 120) = 8 * (lane.val % 16) := by
  decide +kernel +revert

theorem compiled_row_exact {n : Length} (t : Thread n) : compiledRow t = t.row := by
  have hl := t.lane.isLt
  unfold compiledRow Thread.row
  rw [(lane_fields t.lane).1, Nat.or_comm,
    ← Nat.shiftLeft_add_eq_or_of_lt (i := 2) (by omega), Nat.shiftLeft_eq]
  omega

private theorem compiled_row_bound {n : Length} (t : Thread n) : compiledRow t < 12 * n.tokens := by
  have hb := t.block.isLt
  have hl := t.lane.isLt
  rw [compiled_row_exact, Thread.row]
  omega

def compiledCoordinates {n : Length} (t : Thread n) : Nat × Plane × Nat × Nat :=
  let row := compiledRow t
  let token := (row * 715827883 / 4294967296) / 2
  let head := row - 12 * token
  let logical := head - ((head * 43 / 256 + head * 43 / 32768) * 6)
  (token, if head < 6 then .query else .key, logical, vectorStart t + t.reg.val)

theorem compiled_coordinates_exact {n : Length} (t : Thread n) :
    compiledCoordinates t = ((cell t).token.val, (cell t).plane, (cell t).head.val, (cell t).channel.val) := by
  have bound := compiled_row_bound t
  have rlt : compiledRow t < 3145728 := by cases n <;> simp only [Length.tokens] at bound <;> omega
  have delta : t.row - 12 * (t.row / 12) = t.row % 12 := by omega
  dsimp [compiledCoordinates]
  rw [row_division _ rlt, compiled_row_exact, delta]
  simp only [vectorStart, (lane_fields t.lane).2]
  have modulo := logical_head_modulo ⟨t.row % 12, Nat.mod_lt _ (by decide)⟩
  exact congrArg (fun i => (t.row / 12, plane t.row, i, 8 * (t.lane.val % 16) + t.reg.val)) modulo

def qkOffset {n : Length} (p : Point n) (d : Channel) : Nat :=
  2 * (p.token.val * 2304 + (p.plane.base + p.head.val) * 128 + d.val)

def factorOffset {n : Length} (p : Point n) (d : Channel) : Nat :=
  2 * (p.token.val * 128 + d.val)

def outputOffset {n : Length} (p : Point n) : Nat :=
  2 * (p.token.val * 768 + p.head.val * 128 + p.channel.val)

theorem address_bounds {n : Length} (p : Point n) (d : Channel) :
    qkOffset p d + 1 < 2 * n.tokens * 2304 ∧
      factorOffset p d + 1 < 2 * n.tokens * 128 ∧
      outputOffset p + 1 < 2 * n.tokens * 768 ∧ 2 * n.tokens * 2304 < 2^31 := by
  have ht := p.token.isLt
  have hh := p.head.isLt
  have hc := p.channel.isLt
  have hd := d.isLt
  obtain ⟨tok, pk, head, chan⟩ := p
  cases pk <;> cases n <;>
    simp only [qkOffset, factorOffset, outputOffset, Length.tokens, Plane.base] at * <;> omega

theorem output_address_unique {n : Length} (p q : Point n)
    (planeEq : p.plane = q.plane) (addrEq : outputOffset p = outputOffset q) : p = q := by
  have hp := p.head.isLt
  have hq := q.head.isLt
  have hp' := p.channel.isLt
  have hq' := q.channel.isLt
  unfold outputOffset at addrEq
  have ht : p.token = q.token := Fin.ext (by omega)
  have hh : p.head = q.head := Fin.ext (by omega)
  have hc : p.channel = q.channel := Fin.ext (by omega)
  cases p
  cases q
  congr

inductive Buffer where | qk | f1 | f2

/-- Operations are kept in their emitted order through BF16 widening,
FMA, full division, and approximate FTZ reciprocal square root. -/
inductive Word where
  | load (buffer : Buffer) (offset : Nat)
  | literal (bits : Nat)
  | mul (left right : Word)
  | add (left right : Word)
  | fma (a b acc : Word)
  | divFull (left right : Word)
  | rsqrtFtz (x : Word)

def vectorIndex {lanes : Nat} (lane : Fin lanes) (i : Fin 8) : Fin (8 * lanes) :=
  ⟨8 * lane.val + i.val, by have := lane.isLt; have := i.isLt; omega⟩

private def sumSquares {lanes : Nat} (offset : Fin (8 * lanes) → Nat) (lane : Fin lanes) : Word :=
  let x := fun i : Fin 8 => Word.load .qk (offset (vectorIndex lane i))
  let acc := Word.mul (x 1) (x 1)
  ([0, 2, 3, 4, 5, 6, 7] : List (Fin 8)).foldl (fun total i => .fma (x i) (x i) total) acc

private def xorLane {k : Nat} (lane delta : Fin (2 ^ k)) : Fin (2 ^ k) :=
  ⟨lane.val ^^^ delta.val, Nat.xor_lt_two_pow lane.isLt delta.isLt⟩

private def butterfly {k : Nat} :
    List (Fin (2 ^ k)) → (Fin (2 ^ k) → Word) → Fin (2 ^ k) → Word
  | [], input => input
  | delta :: rest, input =>
    butterfly rest (fun lane => .add (input lane) (input (xorLane lane delta)))

private def laneOf {lanes : Nat} (d : Fin (8 * lanes)) : Fin lanes :=
  ⟨d.val / 8, by have := d.isLt; omega⟩

private def partner (d : Channel) : Channel :=
  ⟨d.val ^^^ 1, Nat.xor_lt_two_pow (n := 7) d.isLt (by decide)⟩

/-- Keeps the ordered register fold and each warp shuffle of the head width. -/
def normWord {k : Nat} (steps : List (Fin (2 ^ k))) (width : Nat)
    (offset : Fin (8 * 2 ^ k) → Nat) (channel : Fin (8 * 2 ^ k)) : Word :=
  let total := butterfly steps (sumSquares offset) (laneOf channel)
  .rsqrtFtz (.add (.divFull total (.literal width)) (.literal 0x34000000))

def norm {n : Length} (p : Point n) : Word :=
  normWord (k := 4) [8, 4, 2, 1] 0x43000000 (qkOffset p) p.channel

def unrotated {n : Length} (p : Point n) : Word := .mul (norm p) (.load .qk (qkOffset p p.channel))

def rotateWord (normal : Word) (cur paired factor : Nat) : Word :=
  .fma (.mul normal (.load .qk cur)) (.load .f1 factor)
    (.mul (.mul normal (.load .qk paired)) (.load .f2 factor))

def rotated {n : Length} (p : Point n) : Word :=
  rotateWord (norm p) (qkOffset p p.channel) (qkOffset p (partner p.channel)) (factorOffset p p.channel)

private def previous {n : Length} (p : Point n) : Point n :=
  {p with token := ⟨p.token.val - 1, by have := p.token.isLt; omega⟩}

private def next {n : Length} (p : Point n) (room : p.token.val + 1 < n.tokens) : Point n :=
  {p with token := ⟨p.token.val + 1, room⟩}

private def fromPrevious {n : Length} (p : Point n) : Prop :=
  p.plane = .key ∧ 64 ≤ p.channel.val ∧ 0 < p.token.val

instance {n : Length} (p : Point n) : Decidable (fromPrevious p) := by
  unfold fromPrevious
  infer_instance

def placementMasks {n : Length} (t : Thread n) : Prop :=
    (t.row % 12 ≥ 6 ∧ 63 < vectorStart t ∧ 11 < compiledRow t ↔ fromPrevious (cell t)) ∧
    (t.row % 12 ≥ 6 ∧ (compiledRow t + 11 < 23 ∨ vectorStart t < 64) ↔
      (cell t).plane = .key ∧ ¬fromPrevious (cell t)) ∧
    (t.row % 12 ≥ 6 ∧ 63 < vectorStart t ∧ compiledRow t < 12 * n.tokens - 12 ↔
      (cell t).plane = .key ∧ 64 ≤ (cell t).channel.val ∧ (cell t).token.val + 1 < n.tokens)

theorem compiled_placement_masks {n : Length} (t : Thread n) : placementMasks t := by
  have hreg := t.reg.isLt
  unfold placementMasks
  rw [compiled_row_exact]
  simp only [fromPrevious, cell, plane, vectorStart, (lane_fields t.lane).2]
  split
  · rename_i isQuery
    have isNotKey : ¬6 ≤ t.row % 12 := by omega
    simp [isNotKey]
  · simp only [true_and]
    omega

theorem previous_room {n : Length} (p : Point n) (h : fromPrevious p) :
    (previous p).token.val + 1 < n.tokens := by
  have ht := p.token.isLt
  have hp := h.2.2
  simp only [previous]
  omega

private theorem next_previous {n : Length} (p : Point n) (h : fromPrevious p) :
    next (previous p) (previous_room p h) = p := by
  have hpos := h.2.2
  cases p
  simp only [next, previous, Point.mk.injEq, and_self, and_true]
  apply Fin.ext
  exact Nat.sub_add_cancel hpos

private theorem previous_next {n : Length} (p : Point n) (h : p.token.val + 1 < n.tokens) :
    previous (next p h) = p := by
  cases p
  simp only [next, previous, Point.mk.injEq, and_self, and_true]
  apply Fin.ext
  exact Nat.add_sub_cancel _ _

def reference {n : Length} (p : Point n) : Word :=
  if fromPrevious p then unrotated (previous p) else rotated p

def directStore {n : Length} (src : Point n) : Option (Point n × Word) :=
  if fromPrevious src then none else some (src, rotated src)

def shiftedStore {n : Length} (src : Point n) : Option (Point n × Word) :=
  if room : src.token.val + 1 < n.tokens then
    if src.plane = .key ∧ 64 ≤ src.channel.val then some (next src room, unrotated src) else none
  else none

/-- The reference and candidate select exactly the same expression for
every destination. The shifted store uses the predecessor's own norm. -/
theorem emitted_equal {n : Length} (dst : Point n) :
    (if fromPrevious dst then shiftedStore (previous dst) else directStore dst) =
      some (dst, reference dst) := by
  by_cases h : fromPrevious dst
  · have hp : (previous dst).plane = .key := h.1
    have hc : 64 ≤ (previous dst).channel.val := h.2.1
    simp only [h, if_true, shiftedStore, reference, dif_pos (previous_room dst h),
      hp, hc, and_self, next_previous dst h]
  · simp [h, directStore, reference]

theorem direct_correct {n : Length} (src dst : Point n) (word : Word)
    (h : directStore src = some (dst, word)) : src = dst ∧ ¬fromPrevious dst ∧ word = reference dst := by
  by_cases shifted : fromPrevious src
  · simp [directStore, shifted] at h
  · simp [directStore, shifted] at h
    obtain ⟨rfl, hw⟩ := h
    exact ⟨rfl, shifted, hw.symm.trans (by simp [reference, shifted])⟩

theorem shifted_correct {n : Length} (src dst : Point n) (word : Word)
    (h : shiftedStore src = some (dst, word)) :
    src = previous dst ∧ fromPrevious dst ∧ word = reference dst := by
  unfold shiftedStore at h
  split at h
  · rename_i room
    split at h
    · rename_i isKey
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, hw⟩ := h
      have prior : fromPrevious (next src room) := ⟨isKey.1, isKey.2, Nat.succ_pos _⟩
      exact ⟨(previous_next src room).symm, prior,
        hw.symm.trans (by rw [reference, if_pos prior, previous_next])⟩
    · contradiction
  · contradiction

def store {n : Length} (shift : Bool) (src : Point n) : Option (Point n × Word) :=
  if shift then shiftedStore src else directStore src

theorem store_source_unique {n : Length} (dst lhs rhs : Point n) (a b : Bool) (x y : Word)
    (ha : store a lhs = some (dst, x)) (hb : store b rhs = some (dst, y)) :
    lhs = rhs ∧ a = b := by
  cases a <;> cases b
  · exact ⟨(direct_correct _ _ _ ha).1.trans (direct_correct _ _ _ hb).1.symm, rfl⟩
  · exact ((direct_correct _ _ _ ha).2.1 (shifted_correct _ _ _ hb).2.1).elim
  · exact ((direct_correct _ _ _ hb).2.1 (shifted_correct _ _ _ ha).2.1).elim
  · exact ⟨(shifted_correct _ _ _ ha).1.trans (shifted_correct _ _ _ hb).1.symm, rfl⟩

/-- Supplying the same machine primitives to both expression trees gives
the same BF16 word. The two backend-dependent operations are passed to
the evaluator without assuming rounding identities about them. -/
structure Arithmetic where
  config : Config
  divFull : Value .fp32 → Value .fp32 → Value .fp32
  rsqrtFtz : Value .fp32 → Value .fp32

abbrev Inputs := Buffer → Nat → Value .bf16

def eval (a : Arithmetic) (read : Inputs) : Word → Value .fp32
  | .load b address => FP.cast .fp32 a.config (read b address)
  | .literal bits => Value.ofBits .fp32 bits
  | .mul x y => FP.mul a.config (eval a read x) (eval a read y)
  | .add x y => FP.add a.config (eval a read x) (eval a read y)
  | .fma x y z => FP.fma a.config (eval a read x) (eval a read y) (eval a read z)
  | .divFull x y => a.divFull (eval a read x) (eval a read y)
  | .rsqrtFtz x => a.rsqrtFtz (eval a read x)

def referenceWrite (a : Arithmetic) (read : Inputs) {n : Length} (p : Point n) :
    Option (Plane × Nat × Value .bf16) :=
  some (p.plane, outputOffset p, FP.cast .bf16 a.config (eval a read (reference p)))

def emittedWrite (a : Arithmetic) (read : Inputs) {n : Length} (p : Point n) :
    Option (Plane × Nat × Value .bf16) :=
  (if fromPrevious p then shiftedStore (previous p) else directStore p).map
    fun (dst, word) => (dst.plane, outputOffset dst, FP.cast .bf16 a.config (eval a read word))

theorem writes_equal (a : Arithmetic) (read : Inputs) {n : Length} (p : Point n) :
    emittedWrite a read p = referenceWrite a read p := by
  simp only [emittedWrite, referenceWrite, emitted_equal, Option.map_some]

#axiomsClean owner_covers
#axiomsClean cell_owner_unique
#axiomsClean row_division
#axiomsClean logical_head_modulo
#axiomsClean compiled_coordinates_exact
#axiomsClean compiled_placement_masks
#axiomsClean address_bounds
#axiomsClean output_address_unique
#axiomsClean direct_correct
#axiomsClean shifted_correct
#axiomsClean store_source_unique
#axiomsClean writes_equal
end VeriTile.Bench.Models.NanoGPT.Track1.RopeShift
