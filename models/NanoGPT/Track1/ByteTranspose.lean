import models.NanoGPT.Track1.SampledGather

/-! The dense Hopper path transposes FP8 storage without decoding its bytes.
The launch tiles are exact, both layouts have unique owners, and all
global offsets fit in signed i32. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ByteTranspose
open SampledRows

structure Shape where
  rowTiles : Nat
  colTiles : Nat
  bounded : (rowTiles * 128) * (colTiles * 128) < 2^31

def Shape.rows (s : Shape) : Nat := s.rowTiles * 128
def Shape.cols (s : Shape) : Nat := s.colTiles * 128

structure Point (s : Shape) where
  tileRow : Fin s.rowTiles
  tileCol : Fin s.colTiles
  cell : Cell

private def axis {tiles : Nat} (tile : Fin tiles) (lane : Fin 128) : Fin (tiles * 128) :=
  ⟨tile.val * 128 + lane.val, by have := tile.isLt; have := lane.isLt; omega⟩

private def splitAxis {tiles : Nat} (i : Fin (tiles * 128)) : Fin tiles × Fin 128 :=
  (⟨i.val / 128, by have := i.isLt; omega⟩, ⟨i.val % 128, Nat.mod_lt _ (by decide)⟩)

private theorem axis_split {tiles : Nat} (i : Fin (tiles * 128)) :
    axis (splitAxis i).1 (splitAxis i).2 = i := by
  apply Fin.ext
  simp only [axis, splitAxis]
  omega

private theorem split_axis {tiles : Nat} (tile : Fin tiles) (lane : Fin 128) :
    splitAxis (axis tile lane) = (tile, lane) := by
  have := lane.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [axis, splitAxis] <;> omega

def Point.row {s : Shape} (p : Point s) : Fin s.rows := axis p.tileRow p.cell.1
def Point.col {s : Shape} (p : Point s) : Fin s.cols := axis p.tileCol p.cell.2

def point (s : Shape) (r : Fin s.rows) (c : Fin s.cols) : Point s :=
  ⟨(splitAxis r).1, (splitAxis c).1, (splitAxis r).2, (splitAxis c).2⟩

theorem tiles_cover (s : Shape) (r : Fin s.rows) (c : Fin s.cols) :
    (point s r c).row = r ∧ (point s r c).col = c :=
  ⟨axis_split r, axis_split c⟩

private theorem point_row_col {s : Shape} (p : Point s) : point s p.row p.col = p := by
  cases p
  simp only [point, Point.row, Point.col, split_axis]

private theorem linear_bounded {m n : Nat} (a : Fin m) (b : Fin n) :
    a.val * n + b.val < m * n := by
  have h := Nat.mul_le_mul_right n (Nat.succ_le_of_lt a.isLt)
  rw [Nat.succ_mul] at h
  exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left b.isLt _) h

private theorem linear_injective {m n : Nat} (p q : Fin m × Fin n)
    (h : p.1.val * n + p.2.val = q.1.val * n + q.2.val) : p = q := by
  have pos := Nat.zero_lt_of_lt p.2.isLt
  have major : (p.1.val * n + p.2.val) / n = (q.1.val * n + q.2.val) / n := congrArg (· / n) h
  have minor : (p.1.val * n + p.2.val) % n = (q.1.val * n + q.2.val) % n := congrArg (· % n) h
  rw [Nat.mul_comm p.1.val n, Nat.mul_comm q.1.val n, Nat.mul_add_div pos,
    Nat.mul_add_div pos, Nat.div_eq_of_lt p.2.isLt, Nat.div_eq_of_lt q.2.isLt] at major
  rw [Nat.mul_comm p.1.val n, Nat.mul_comm q.1.val n, Nat.mul_add_mod_self_left,
    Nat.mul_add_mod_self_left, Nat.mod_eq_of_lt p.2.isLt, Nat.mod_eq_of_lt q.2.isLt] at minor
  exact Prod.ext (Fin.ext (by simpa using major)) (Fin.ext minor)

def sourceOffset {s : Shape} (p : Point s) : Nat := p.row.val * s.cols + p.col.val
def destOffset {s : Shape} (p : Point s) : Nat := p.col.val * s.rows + p.row.val

theorem offsets_bounded {s : Shape} (p : Point s) :
    sourceOffset p < s.rows * s.cols ∧ destOffset p < s.rows * s.cols :=
  ⟨linear_bounded p.row p.col, by rw [Nat.mul_comm]; exact linear_bounded p.col p.row⟩

private def i32Offset (major stride minor : Nat) : Nat :=
  ((major * stride) % 2^32 + minor) % 2^32

theorem signed_offsets_exact {s : Shape} (p : Point s) :
    i32Offset p.row.val s.cols p.col.val = sourceOffset p ∧
    i32Offset p.col.val s.rows p.row.val = destOffset p ∧
    sourceOffset p < 2^31 ∧ destOffset p < 2^31 := by
  have hs := Nat.lt_trans (offsets_bounded p).1 s.bounded
  have hd := Nat.lt_trans (offsets_bounded p).2 s.bounded
  unfold sourceOffset at hs
  unfold destOffset at hd
  exact ⟨by unfold i32Offset sourceOffset; rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)],
    by unfold i32Offset destOffset; rw [Nat.mod_eq_of_lt (by omega), Nat.mod_eq_of_lt (by omega)],
    hs, hd⟩

theorem unique_destination {s : Shape} (p q : Point s)
    (h : destOffset p = destOffset q) : p = q := by
  have same : (p.col, p.row) = (q.col, q.row) := linear_injective _ _ h
  have hcol : p.col = q.col := congrArg Prod.fst same
  have hrow : p.row = q.row := congrArg Prod.snd same
  rw [← point_row_col p, ← point_row_col q, hcol, hrow]

theorem row_owner_index (t : WideThread) : wideRowOwner (wideRowIndex t) = t := by
  cases t with
  | mk warp lane reg =>
    have hw := warp.isLt
    have hl := lane.isLt
    have hr := reg.isLt
    unfold wideRowOwner wideRowIndex
    congr <;> dsimp <;> omega

theorem col_owner_index (t : WideThread) : wideColOwner (wideColIndex t) = t := by
  cases t with
  | mk warp lane reg =>
    have hw := warp.isLt
    have hl := lane.isLt
    have hr := reg.isLt
    unfold wideColOwner wideColIndex
    congr <;> dsimp <;> omega

private def tid (warp : Fin 8) (lane : Fin 32) : Nat := 32 * warp.val + lane.val

private theorem lane_masks (warp : Fin 8) (lane : Fin 32) :
    ((tid warp lane >>> 1) &&& 124) = 16 * warp.val + 4 * (lane.val / 8) ∧
    ((tid warp lane <<< 3) &&& 56) = 8 * (lane.val % 8) ∧
    ((tid warp lane <<< 1) &&& 62) = 2 * lane.val ∧
    ((tid warp lane >>> 1) &&& 112) = 16 * warp.val := by
  decide +kernel +revert

private theorem tile_base (tile within : Nat) (h : within < 128) :
    within ||| (tile <<< 7) = tile * 128 + within := by
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt (i := 7) h, Nat.shiftLeft_eq]

/-- The global vector load uses one 8-byte segment for each of four rows
in each 64-column half. The shared-buffer model names each register byte
by its logical row and column. -/
def compiledSource {s : Shape} (p : Point s) (t : WideThread) : Nat :=
  let r0 := ((tid t.warp t.lane >>> 1) &&& 124) ||| (p.tileRow.val <<< 7)
  let c0 := ((tid t.warp t.lane <<< 3) &&& 56) ||| (p.tileCol.val <<< 7)
  ((r0 + t.reg.val / 8 % 4) * s.cols) % 2^32 + c0 + 64 * (t.reg.val / 32) + t.reg.val % 8

/-- Each output lane writes four 16-byte vectors. The higher column half
increases the product by 64 times the row stride; PTX emits 8064 times
the row-tile count after adding one row stride. -/
def compiledDestination {s : Shape} (p : Point s) (t : WideThread) : Nat :=
  let c0 := ((tid t.warp t.lane <<< 1) &&& 62) ||| (p.tileCol.val <<< 7)
  let r0 := ((tid t.warp t.lane >>> 1) &&& 112) ||| (p.tileRow.val <<< 7)
  let pair := t.reg.val / 16
  (c0 * s.rows + (pair % 2 + pair / 2) * s.rows + (pair / 2) * (8064 * s.rowTiles)) % 2^32
    + r0 + t.reg.val % 16

theorem compiled_source_exact {s : Shape} (p : Point s) (t : WideThread) :
    compiledSource p t = sourceOffset {p with cell := wideRowIndex t} := by
  obtain ⟨hr, hc, _, _⟩ := lane_masks t.warp t.lane
  have hw := t.warp.isLt
  have hl := t.lane.isLt
  have ht := t.reg.isLt
  have hbound := (signed_offsets_exact {p with cell := wideRowIndex t}).2.2.1
  unfold compiledSource
  rw [hr, hc, tile_base _ _ (by omega), tile_base _ _ (by omega)]
  unfold sourceOffset Point.row Point.col axis wideRowIndex at hbound ⊢
  dsimp at hbound ⊢
  have row : p.tileRow.val * 128 + (16 * t.warp.val + 4 * (t.lane.val / 8)) + t.reg.val / 8 % 4 =
      p.tileRow.val * 128 + (16 * t.warp.val + 4 * (t.lane.val / 8) + t.reg.val / 8 % 4) := by omega
  rw [row, Nat.mod_eq_of_lt (by omega)]
  omega

theorem compiled_destination_exact {s : Shape} (p : Point s) (t : WideThread) :
    compiledDestination p t = destOffset {p with cell := wideColIndex t} := by
  obtain ⟨_, _, hc, hr⟩ := lane_masks t.warp t.lane
  have hw := t.warp.isLt
  have hl := t.lane.isLt
  have ht := t.reg.isLt
  have hbound := (signed_offsets_exact {p with cell := wideColIndex t}).2.2.2
  unfold compiledDestination
  rw [hc, hr, tile_base _ _ (by omega), tile_base _ _ (by omega)]
  unfold destOffset Point.row Point.col axis wideColIndex at hbound ⊢
  dsimp at hbound ⊢
  let c0 := p.tileCol.val * 128 + 2 * t.lane.val
  have c : (p.tileCol.val * 128 + (64 * (t.reg.val / 32) + 2 * t.lane.val + t.reg.val / 16 % 2)) =
      c0 + (t.reg.val / 16 % 2 + t.reg.val / 16 / 2) + 63 * (t.reg.val / 16 / 2) := by
    dsimp [c0]
    omega
  change (c0 * s.rows + (t.reg.val / 16 % 2 + t.reg.val / 16 / 2) * s.rows +
    t.reg.val / 16 / 2 * (8064 * s.rowTiles)) % 2^32 +
      (p.tileRow.val * 128 + 16 * t.warp.val) + t.reg.val % 16 = _
  rw [c] at hbound ⊢
  have pitch : 8064 * s.rowTiles = 63 * s.rows := by
    unfold Shape.rows
    omega
  have high : t.reg.val / 16 / 2 * (8064 * s.rowTiles) =
      (63 * (t.reg.val / 16 / 2)) * s.rows := by rw [pitch]; ac_rfl
  rw [high, ← Nat.add_mul, ← Nat.add_mul]
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

/-- The original copy's bounds mask at each point of the admitted grid. -/
def reference {s : Shape} (bytes : Nat → UInt8) (p : Point s) : Option (Nat × UInt8) :=
  if p.row.val < s.rows ∧ p.col.val < s.cols then
    some (destOffset p, bytes (sourceOffset p))
  else none

/-- Observe the actual bytes selected by the output lane after shared-memory
exchange, with offsets lowered to the launch's finite-width indexing. -/
def emitted {s : Shape} (bytes : Nat → UInt8) (p : Point s) : Option (Nat × UInt8) :=
  let output := wideColOwner p.cell
  let input := wideRowOwner (decodeLocation (loadLocation output))
  some (compiledDestination p output, bytes (compiledSource p input))

theorem emitted_equal {s : Shape} (bytes : Nat → UInt8) (p : Point s) :
    emitted bytes p = reference bytes p := by
  simp only [emitted, reference, compiled_source_exact, compiled_destination_exact, row_covers,
    load_decodes, col_covers, p.row.isLt, p.col.isLt, and_self, if_true]

#axiomsClean tiles_cover
#axiomsClean signed_offsets_exact
#axiomsClean unique_destination
#axiomsClean row_owner_index
#axiomsClean col_owner_index
#axiomsClean compiled_source_exact
#axiomsClean compiled_destination_exact
#axiomsClean emitted_equal
end VeriTile.Bench.Models.NanoGPT.Track1.ByteTranspose
