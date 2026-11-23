import models.NanoGPT.Track1.SampledGather
import Mathlib.Data.List.FinRange

/-! Sampled cross-entropy's compact BF16 LM-head gradient restores the
full vocabulary layout. The software reference uses a masked gather.
The Hopper path uses signed i32 base offsets and four fixed row phases.
Each selected halfword is preserved, and missing classes receive positive zero. -/

namespace VeriTile.Bench.Models.NanoGPT.Track1.Densify
open SampledRows (Count ClassID Channel)

structure Thread where
  lane : Fin 32
  warp : Fin 8
  reg : Fin 32

abbrev Cell := Fin 64 × Fin 128

def cell (t : Thread) : Cell :=
  (⟨16 * (t.reg.val / 8) + 2 * t.warp.val + t.lane.val / 16, by
      have := t.reg.isLt
      have := t.warp.isLt
      have := t.lane.isLt
      omega⟩,
    ⟨8 * (t.lane.val % 16) + t.reg.val % 8, by omega⟩)

def owner (p : Cell) : Thread :=
  ⟨⟨16 * (p.1.val % 2) + p.2.val / 8, by have := p.2.isLt; omega⟩,
    ⟨p.1.val % 16 / 2, by omega⟩,
    ⟨8 * (p.1.val / 16) + p.2.val % 8, by have := p.1.isLt; omega⟩⟩

theorem owner_covers (p : Cell) : cell (owner p) = p := by
  have := p.1.isLt
  have := p.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [cell, owner] <;> omega

theorem owner_unique (t : Thread) : owner (cell t) = t := by
  cases t with
  | mk lane warp reg =>
    have := lane.isLt
    have := warp.isLt
    have := reg.isLt
    unfold cell owner
    congr <;> dsimp <;> omega

private def tid (warp : Fin 8) (lane : Fin 32) : Nat := 32 * warp.val + lane.val

theorem lane_masks (warp : Fin 8) (lane : Fin 32) :
    ((tid warp lane <<< 3) &&& 120) = 8 * (lane.val % 16) ∧
    ((tid warp lane >>> 4) &&& 15) = 2 * warp.val + lane.val / 16 := by
  decide +kernel +revert

abbrev PositionMap (c : Count) := ClassID → Option (Fin c.rows)

def flatNonzero (mark : ClassID → Bool) : List ClassID := (List.finRange 50304).filter mark

theorem flat_nonzero_nodup (mark : ClassID → Bool) : (flatNonzero mark).Nodup :=
  (List.nodup_finRange 50304).filter mark

theorem flat_nonzero_mem (mark : ClassID → Bool) (id : ClassID) :
    id ∈ flatNonzero mark ↔ mark id = true := by
  simp [flatNonzero, List.mem_finRange]

def stagedClasses (c : Count) (mark : ClassID → Bool) (h : (flatNonzero mark).length = c.rows)
    (i : Fin c.rows) : ClassID :=
  (flatNonzero mark).get (i.cast h.symm)

theorem staged_classes_injective (c : Count) (mark : ClassID → Bool)
    (h : (flatNonzero mark).length = c.rows) : Function.Injective (stagedClasses c mark h) := by
  intro i j hij
  have eq : i.cast h.symm = j.cast h.symm := (flat_nonzero_nodup mark).get_inj_iff.mp hij
  exact @Fin.cast_injective c.rows (flatNonzero mark).length h.symm i j eq

private def writeIndex {c : Count} (ids : Fin c.rows → ClassID) (p : PositionMap c) (i : Fin c.rows) :
    PositionMap c := fun x => if x = ids i then some i else p x

/-- A fill with -1 followed by copying the ordinal of each candidate to its class. -/
def inverseMap {c : Count} (ids : Fin c.rows → ClassID) : PositionMap c :=
  (List.finRange c.rows).foldl (writeIndex ids) (fun _ => none)

private theorem keep_unwritten {c : Count} (ids : Fin c.rows → ClassID) (entries : List (Fin c.rows))
    (p : PositionMap c) (x : ClassID) (missing : ∀ j ∈ entries, x ≠ ids j) :
    entries.foldl (writeIndex ids) p x = p x := by
  induction entries generalizing p with
  | nil => rfl
  | cons j js ih =>
    rw [List.foldl_cons, ih _ (fun i hi => missing i (List.mem_cons_of_mem j hi))]
    simp [writeIndex, missing j (List.mem_cons_self)]

private theorem read_written {c : Count} (ids : Fin c.rows → ClassID) (unique : Function.Injective ids)
    (entries : List (Fin c.rows)) (valid : entries.Nodup) (p : PositionMap c)
    (i : Fin c.rows) (hi : i ∈ entries) :
    entries.foldl (writeIndex ids) p (ids i) = some i := by
  induction entries generalizing p with
  | nil => simp at hi
  | cons j js ih =>
    have ⟨notTail, tailNodup⟩ := List.nodup_cons.mp valid
    rw [List.foldl_cons]
    rcases List.mem_cons.mp hi with hij | tail
    · subst j
      rw [keep_unwritten ids js _ (ids i) (fun k hk he => notTail (unique he ▸ hk))]
      simp [writeIndex]
    · exact ih tailNodup _ tail

theorem inverse_selected {c : Count} (ids : Fin c.rows → ClassID) (unique : Function.Injective ids)
    (i : Fin c.rows) : inverseMap ids (ids i) = some i :=
  read_written ids unique _ (List.nodup_finRange c.rows) _ i (List.mem_finRange i)

theorem inverse_missing {c : Count} (ids : Fin c.rows → ClassID) (x : ClassID)
    (missing : ∀ i, x ≠ ids i) : inverseMap ids x = none :=
  keep_unwritten ids _ _ x (fun i _ => missing i)

theorem inverse_stage_selected (c : Count) (mark : ClassID → Bool) (h : (flatNonzero mark).length = c.rows)
    (i : Fin c.rows) :
    inverseMap (stagedClasses c mark h) (stagedClasses c mark h i) = some i :=
  inverse_selected _ (staged_classes_injective c mark h) i

theorem inverse_stage_missing (c : Count) (mark : ClassID → Bool) (h : (flatNonzero mark).length = c.rows)
    (x : ClassID) (negative : mark x = false) :
    inverseMap (stagedClasses c mark h) x = none := by
  apply inverse_missing
  intro i eq
  have mem : (flatNonzero mark).get (i.cast h.symm) ∈ flatNonzero mark :=
    List.get_mem (flatNonzero mark) (i.cast h.symm)
  change x = (flatNonzero mark).get (i.cast h.symm) at eq
  rw [← eq] at mem
  have marked := (flat_nonzero_mem mark x).mp mem
  rw [negative] at marked
  exact Bool.false_ne_true marked

structure Tile where
  row : Fin 12
  col : Fin 393

structure Point where
  tile : Tile
  localCell : Cell

def Point.row (p : Point) : Channel :=
  ⟨p.tile.row.val * 64 + p.localCell.1.val, by
    have := p.tile.row.isLt
    have := p.localCell.1.isLt
    omega⟩

def Point.col (p : Point) : ClassID :=
  ⟨p.tile.col.val * 128 + p.localCell.2.val, by
    have := p.tile.col.isLt
    have := p.localCell.2.isLt
    omega⟩

def point (r : Channel) (v : ClassID) : Point :=
  ⟨⟨⟨r.val / 64, by have hr : r.val < 768 := r.isLt; omega⟩,
     ⟨v.val / 128, by have hv : v.val < 50304 := v.isLt; omega⟩⟩,
    ⟨r.val % 64, Nat.mod_lt _ (by decide)⟩, ⟨v.val % 128, Nat.mod_lt _ (by decide)⟩⟩

theorem covers (r : Channel) (v : ClassID) : (point r v).row = r ∧ (point r v).col = v := by
  constructor <;> apply Fin.ext <;> simp only [Point.row, Point.col, point] <;> omega

def destOffset (p : Point) : Nat := 2 * (p.row.val * 50304 + p.col.val)

def sourceOffset (c : Count) (r : Channel) (pos : Fin c.rows) : Nat := 2 * (r.val * c.rows + pos.val)

theorem addresses_in_range (c : Count) (p : Point) (i : Fin c.rows) :
    sourceOffset c p.row i + 1 < 2 * 768 * c.rows ∧ destOffset p + 1 < 2 * 768 * 50304 := by
  have hr : p.row.val < 768 := p.row.isLt
  have hc : p.col.val < 50304 := p.col.isLt
  have hi := i.isLt
  cases c <;> simp only [sourceOffset, destOffset, Count.rows] at * <;> omega

theorem dest_address_unique (p q : Point) (h : destOffset p = destOffset q) : p = q := by
  have hcp : p.col.val < 50304 := p.col.isLt
  have hcq : q.col.val < 50304 := q.col.isLt
  unfold destOffset at h
  have hr : p.row.val = q.row.val := by omega
  have hc : p.col.val = q.col.val := by omega
  have hp := p.localCell.1.isLt
  have hq := q.localCell.1.isLt
  have hp' := p.localCell.2.isLt
  have hq' := q.localCell.2.isLt
  simp only [Point.row, Point.col] at hr hc
  have tr : p.tile.row = q.tile.row := Fin.ext (by omega)
  have tc : p.tile.col = q.tile.col := Fin.ext (by omega)
  have cr : p.localCell.1 = q.localCell.1 := Fin.ext (by omega)
  have cc : p.localCell.2 = q.localCell.2 := Fin.ext (by omega)
  cases p with
  | mk pt pr =>
    cases q with
    | mk qt qr =>
      cases pt
      cases qt
      congr
      exact Prod.ext cr cc

private theorem tile_base (shift tile within : Nat) (h : within < 2^shift) :
    within ||| (tile <<< shift) = tile * 2^shift + within := by
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt h, Nat.shiftLeft_eq]

/-- The PTX masks the lane ID to construct the first row and the eight
adjacent position indices. Later row phases reuse these same bases. -/
def compiledBase (tile : Tile) (t : Thread) : Nat × Nat :=
  let thread := tid t.warp t.lane
  let row := ((thread >>> 4) &&& 15) ||| (tile.row.val <<< 6)
  let col := ((thread <<< 3) &&& 120) ||| (tile.col.val <<< 7)
  (row, col + t.reg.val % 8)

def compiledCoords (tile : Tile) (t : Thread) : Nat × Nat :=
  let (row, col) := compiledBase tile t
  (row + 16 * (t.reg.val / 8), col)

theorem compiled_coords_exact (tile : Tile) (t : Thread) :
    compiledCoords tile t =
      ((Point.row ⟨tile, cell t⟩).val, (Point.col ⟨tile, cell t⟩).val) := by
  have hw := t.warp.isLt
  have hl := t.lane.isLt
  obtain ⟨hv, hr⟩ := lane_masks t.warp t.lane
  dsimp [compiledCoords, compiledBase]
  rw [hv, hr, tile_base _ _ _ (by omega), tile_base _ _ _ (by omega)]
  apply Prod.ext <;> simp only [Point.row, Point.col, cell] <;> omega

def encodedPosition {c : Count} (pos : Option (Fin c.rows)) : Int32 :=
  match pos with | none => -1 | some i => Int32.ofNat i.val

private theorem index_i32 {c : Count} (pos : Fin c.rows) :
    (Int32.ofNat pos.val).toInt = pos.val := by
  have := pos.isLt
  apply Int32.toInt_ofNat_of_lt
  cases c <;> simp only [Count.rows] at * <;> omega

private theorem position_none : (encodedPosition (c := .first) none).toInt = -1 := by decide

private theorem add_toNat (a b : Nat) : ((a : Int) + (b : Int)).toNat = a + b := by
  rw [← Int.natCast_add, Int.toNat_natCast]

/-- The compact phase pitch is added to a base formed by two signed
wide address operations. Negative position indices never issue a load. -/
private def emittedHalfword {c : Count} (src : Nat → UInt16) (row phase : Nat)
    (position : Option (Fin c.rows)) : UInt16 :=
  let pos := (encodedPosition position).toInt
  if pos ≥ 0 then
    src (2 * (((Int32.ofNat (row * c.rows)).toInt + pos).toNat) + 32 * phase * c.rows)
  else 0

private theorem compiled_selected {c : Count} (src : Nat → UInt16) (row phase : Nat)
    (h : row + 16 * phase < 768) (i : Fin c.rows) :
    emittedHalfword src row phase (some i) = src (sourceOffset c ⟨row + 16 * phase, h⟩ i) := by
  have hs : row * c.rows < 2^31 := by
    cases c <;> simp only [Count.rows] at * <;> omega
  unfold emittedHalfword sourceOffset encodedPosition
  rw [index_i32 i, Int32.toInt_ofNat_of_lt hs, if_pos (Int.natCast_nonneg i.val), add_toNat]
  congr 1
  cases c <;> simp only [Count.rows] <;> omega

private theorem compiled_missing (c : Count) (src : Nat → UInt16) (row phase : Nat) :
    emittedHalfword (c := c) src row phase none = 0 := by
  have h : (encodedPosition (c := c) none).toInt = -1 := position_none
  simp [emittedHalfword, h]

/-- The store computes its signed i32 multiply-add once, then advances
the byte pointer by a fixed number of full vocabulary rows. -/
def compiledStore (row col phase : Nat) : Nat :=
  2 * (Int32.ofNat (row * 50304 + col)).toInt.toNat + 32 * phase * 50304

theorem compiled_store_exact (row col phase : Nat) (hr : row + 16 * phase < 768) (hc : col < 50304) :
    compiledStore row col phase = 2 * ((row + 16 * phase) * 50304 + col) := by
  have h : row * 50304 + col < 2^31 := by omega
  unfold compiledStore
  rw [Int32.toInt_ofNat_of_lt h, Int.toNat_natCast]
  omega

/-- The old kernel predicates the load and returns a BF16 +0 on an empty slot.
Both its row and column bounds pass at these in-range output coordinates. -/
def reference {c : Count} (src : Nat → UInt16) (pos : PositionMap c) (p : Point) : Nat × UInt16 :=
  (destOffset p, match pos p.col with
    | none => 0
    | some i => src (sourceOffset c p.row i))

/-- The base address of the i32 lane in the position buffer is four times its class. -/
def positionLoad {c : Count} (pos : PositionMap c) (byteOffset : Nat) : Option (Fin c.rows) :=
  if h : byteOffset / 4 < 50304 then pos ⟨byteOffset / 4, h⟩ else none

private theorem position_load {c : Count} (pos : PositionMap c) (col : ClassID) :
    positionLoad pos (4 * col.val) = pos col := by
  have h : 0 < 4 := by decide
  simp only [positionLoad, Nat.mul_div_cancel_left _ h, dif_pos col.isLt, Fin.eta]

def emitted {c : Count} (src : Nat → UInt16) (pos : PositionMap c) (p : Point) : Nat × UInt16 :=
  let t := owner p.localCell
  let (row, col) := compiledBase p.tile t
  let phase := t.reg.val / 8
  (compiledStore row col phase, emittedHalfword src row phase (positionLoad pos (4 * col)))

abbrev Dispatch := Bool → (c : Count) → (Nat → UInt16) → PositionMap c → Point → Nat × UInt16

theorem emitted_equal {c : Count} (src : Nat → UInt16) (pos : PositionMap c) (p : Point) :
    emitted src pos p = reference src pos p := by
  let t := owner p.localCell
  let row := (compiledBase p.tile t).1
  let col := (compiledBase p.tile t).2
  let phase := t.reg.val / 8
  have coords := compiled_coords_exact p.tile t
  rw [show cell t = p.localCell from owner_covers p.localCell] at coords
  change (row + 16 * phase, col) = (p.row.val, p.col.val) at coords
  have hr : row + 16 * phase = p.row.val := congrArg Prod.fst coords
  have hc : col = p.col.val := congrArg Prod.snd coords
  have hrow : row + 16 * phase < 768 := by rw [hr]; exact p.row.isLt
  change (compiledStore row col phase, emittedHalfword src row phase (positionLoad pos (4 * col))) = _
  rw [hc, position_load, compiled_store_exact _ _ _ hrow p.col.isLt]
  unfold reference
  apply Prod.ext
  · exact congrArg (fun r => 2 * (r * 50304 + p.col.val)) hr
  · cases pos p.col
    · exact compiled_missing c src row phase
    · rename_i i
      rw [compiled_selected src row phase hrow i]
      have rowEq : (⟨row + 16 * phase, hrow⟩ : Channel) = p.row := Fin.ext hr
      exact congrArg (fun r => src (sourceOffset c r i)) rowEq

theorem restores_selected (c : Count) (mark : ClassID → Bool) (h : (flatNonzero mark).length = c.rows)
    (src : Nat → UInt16) (row : Channel) (i : Fin c.rows) :
    let p := point row (stagedClasses c mark h i)
    emitted src (inverseMap (stagedClasses c mark h)) p =
      (destOffset p, src (sourceOffset c row i)) := by
  dsimp only
  rw [emitted_equal, reference, (covers _ _).1, (covers _ _).2, inverse_stage_selected]

theorem clears_unselected (c : Count) (mark : ClassID → Bool) (h : (flatNonzero mark).length = c.rows)
    (src : Nat → UInt16) (row : Channel) (col : ClassID) (unselected : mark col = false) :
    let p := point row col
    emitted src (inverseMap (stagedClasses c mark h)) p = (destOffset p, 0) := by
  dsimp only
  rw [emitted_equal, reference, (covers _ _).2, inverse_stage_missing c mark h col unselected]

#axiomsClean inverse_stage_selected
#axiomsClean inverse_stage_missing
#axiomsClean covers
#axiomsClean dest_address_unique
#axiomsClean addresses_in_range
#axiomsClean compiled_coords_exact
#axiomsClean compiled_store_exact
#axiomsClean emitted_equal
#axiomsClean restores_selected
#axiomsClean clears_unselected
#axiomsClean owner_covers
#axiomsClean owner_unique
#axiomsClean lane_masks
end VeriTile.Bench.Models.NanoGPT.Track1.Densify
