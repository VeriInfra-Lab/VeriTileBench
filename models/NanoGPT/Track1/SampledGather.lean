import models.NanoGPT.Track1.QKVGluonLayout

/-! The fused sampled-softmax gather uses the original candidate IDs and
copies their raw bytes to both GEMM layouts. Four warp-local transfers
cover each 128 by 128 tile. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.SampledRows
open QKVPack

inductive Count where | first | second | third
  deriving DecidableEq

def Count.rows : Count → Nat
  | .first => 10240
  | .second => 14336
  | .third => 24576

abbrev ClassID := Fin 50304
abbrev Channel := Fin 768

/-- In the PTX, narrowing the input i64 and multiplying by 768 lower to
a multiplication then an arithmetic shift. Every legal class leaves the
64-bit sign bit clear and the lower 32 bits zero. -/
theorem compiled_class_stride (id : ClassID) :
    (id.val * 3298534883328 % 18446744073709551616) / 4294967296 = id.val * 768 := by
  have hi := id.isLt
  have hs : id.val * 3298534883328 < 9223372036854775808 := by omega
  have hb : id.val * 3298534883328 < 18446744073709551616 := by omega
  rw [Nat.mod_eq_of_lt hb]
  rw [show id.val * 3298534883328 = id.val * 768 * 4294967296 by rw [Nat.mul_assoc]]
  exact Nat.mul_div_cancel _ (by decide)

def expandRow (compressed : Fin 64) (half : Fin 2) : Fin 128 :=
  ⟨compressed.val / 8 * 16 + compressed.val % 8 / 2 * 4 + 2 * half.val + compressed.val % 2,
    by have hc := compressed.isLt; have hh := half.isLt; omega⟩

def compressRow (r : Fin 128) : Fin 64 × Fin 2 :=
  (⟨r.val / 16 * 8 + r.val % 16 / 4 * 2 + r.val % 2, by have := r.isLt; omega⟩,
   ⟨r.val / 2 % 2, Nat.mod_lt _ (by decide)⟩)

theorem expand_compress (r : Fin 128) :
    expandRow (compressRow r).1 (compressRow r).2 = r := by
  decide +kernel +revert

theorem compress_expand (r : Fin 64) (half : Fin 2) :
    compressRow (expandRow r half) = (r, half) := by
  decide +kernel +revert


/-- Each input lane has four rows per eight-column register group.
The output lane has sixteen rows per column. Both layouts keep the same
sixteen-row band in one warp. -/
structure WideThread where
  warp : Fin 8
  lane : Fin 32
  reg : Fin 64

abbrev Cell := Fin 128 × Fin 128

def wideRowIndex (p : WideThread) : Cell :=
  (⟨16 * p.warp.val + 4 * (p.lane.val / 8) + (p.reg.val / 8) % 4, by
      have hw := p.warp.isLt
      have hl := p.lane.isLt
      omega⟩,
    ⟨64 * (p.reg.val / 32) + 8 * (p.lane.val % 8) + p.reg.val % 8,
      by have := p.reg.isLt; omega⟩)

def wideColIndex (p : WideThread) : Cell :=
  (⟨16 * p.warp.val + p.reg.val % 16, by have := p.warp.isLt; omega⟩,
    ⟨64 * (p.reg.val / 32) + 2 * p.lane.val + (p.reg.val / 16) % 2, by
      have hl := p.lane.isLt
      have hr := p.reg.isLt
      omega⟩)

def wideRowOwner (i : Cell) : WideThread where
  warp := ⟨i.1.val / 16, by have := i.1.isLt; omega⟩
  lane := ⟨8 * (i.1.val % 16 / 4) + i.2.val % 64 / 8, by omega⟩
  reg := ⟨32 * (i.2.val / 64) + 8 * (i.1.val % 4) + i.2.val % 8, by
    have := i.2.isLt
    omega⟩

def wideColOwner (i : Cell) : WideThread where
  warp := ⟨i.1.val / 16, by have := i.1.isLt; omega⟩
  lane := ⟨i.2.val % 64 / 2, by omega⟩
  reg := ⟨32 * (i.2.val / 64) + 16 * (i.2.val % 2) + i.1.val % 16, by
    have := i.2.isLt
    omega⟩

theorem row_covers (i : Cell) : wideRowIndex (wideRowOwner i) = i := by
  have hr := i.1.isLt
  have hc := i.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [wideRowIndex, wideRowOwner] <;> omega

theorem col_covers (i : Cell) : wideColIndex (wideColOwner i) = i := by
  have hr := i.1.isLt
  have hc := i.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [wideColIndex, wideColOwner] <;> omega

private def rowProjection (p : WideThread) : ThreadValue × Fin 2 :=
  (⟨p.warp, p.lane, ⟨16 * (p.reg.val / 32) + p.reg.val % 16, by have := p.reg.isLt; omega⟩⟩,
  ⟨p.reg.val / 16 % 2, Nat.mod_lt _ (by decide)⟩)

private def colProjection (p : WideThread) : ThreadValue × Fin 2 :=
  (⟨p.warp, p.lane, ⟨16 * (p.reg.val / 32) + 8 * (p.reg.val / 16 % 2) +
    2 * (p.reg.val % 16 / 4) + p.reg.val % 2, by have := p.reg.isLt; omega⟩⟩,
  ⟨p.reg.val / 2 % 2, Nat.mod_lt _ (by decide)⟩)

structure Location where
  phase : Fin 4
  address : Fin 4096
  deriving DecidableEq

private def liftLocation (l : GluonShared.ByteLocation) (half : Fin 2) : Location :=
  ⟨⟨2 * l.phase.val + half.val, by
    have hc := l.phase.isLt
    have hr := half.isLt
    change l.phase.val < 2 at hc
    omega⟩, l.address⟩

def storeLocation (p : WideThread) : Location :=
  let (owner, half) := rowProjection p
  liftLocation (GluonShared.storeLocation owner) half

def loadLocation (p : WideThread) : Location :=
  let (owner, half) := colProjection p
  liftLocation (GluonShared.loadLocation owner) half

private def baseLocation (l : Location) : GluonShared.ByteLocation :=
  ⟨⟨l.phase.val / 2, by have := l.phase.isLt; omega⟩, l.address⟩

private def rowHalf (l : Location) : Fin 2 :=
  ⟨l.phase.val % 2, Nat.mod_lt _ (by decide)⟩

def decodeLocation (l : Location) : Cell :=
  let (r,c) := GluonShared.decode (baseLocation l)
  (expandRow r (rowHalf l), c)

private theorem decode_lift (l : GluonShared.ByteLocation) (half : Fin 2) :
    decodeLocation (liftLocation l half) = (expandRow (GluonShared.decode l).1 half, (GluonShared.decode l).2) := by
  have hh := half.isLt
  have hp : (⟨(liftLocation l half).phase.val / 2, by have := (liftLocation l half).phase.isLt; omega⟩ : Fin 2) =
      l.phase := Fin.ext (by simp only [liftLocation]; omega)
  have hr : (⟨(liftLocation l half).phase.val % 2, Nat.mod_lt _ (by decide)⟩ : Fin 2) =
      half := Fin.ext (by simp only [liftLocation]; omega)
  have hb : baseLocation (liftLocation l half) = l := by
    unfold baseLocation
    rw [hp]
    rfl
  simp only [decodeLocation, hb, rowHalf, hr]


private theorem expand_row_value (r : Fin 64) (half : Fin 2) (warp within : Nat)
    (hr : r.val = 8 * warp + within) (hi : within < 8) :
    (expandRow r half).val = 16 * warp + 4 * (within / 2) + 2 * half.val + within % 2 := by
  have hdiv : r.val / 8 = warp := by omega
  have hmod : r.val % 8 = within := by omega
  simp only [expandRow, hdiv, hmod]
  omega

theorem store_decodes (p : WideThread) : decodeLocation (storeLocation p) = wideRowIndex p := by
  have hl := p.lane.isLt
  have hr := p.reg.isLt
  let base := (rowProjection p).1
  have hreg : base.reg.val = 16 * (p.reg.val / 32) + p.reg.val % 16 := rfl
  have hdiv : base.reg.val / 16 = p.reg.val / 32 := by omega
  have hcol : base.reg.val % 8 = p.reg.val % 8 := by omega
  have hrow : base.reg.val / 8 % 2 = p.reg.val / 8 % 2 := by omega
  let within := 2 * (p.lane.val / 8) + p.reg.val / 8 % 2
  have hr₁ : (rowIndex base).1.val = 8 * p.warp.val + within := by
    change 8 * p.warp.val + 2 * (p.lane.val / 8) + base.reg.val / 8 % 2 = _
    rw [hrow]
    unfold within
    omega
  have hw : within < 8 := by unfold within; omega
  simp only [storeLocation, decode_lift, shared_store_decodes]
  apply Prod.ext
  · apply Fin.ext
    change (expandRow (rowIndex base).1 (rowProjection p).2).val = _
    rw [expand_row_value _ _ _ _ hr₁ hw]
    change 16 * p.warp.val + 4 * (within / 2) + 2 * (p.reg.val / 16 % 2) + within % 2 =
      16 * p.warp.val + 4 * (p.lane.val / 8) + p.reg.val / 8 % 4
    unfold within
    omega
  · apply Fin.ext
    change 64 * (base.reg.val / 16) + 8 * (p.lane.val % 8) + base.reg.val % 8 =
      64 * (p.reg.val / 32) + 8 * (p.lane.val % 8) + p.reg.val % 8
    rw [hdiv, hcol]

private theorem low_row_bits (r : Nat) :
    let within := 2 * (r % 16 / 4) + r % 2
    4 * (within / 2) + 2 * (r / 2 % 2) + within % 2 = r % 16 := by
  dsimp
  have hdiv : (2 * (r % 16 / 4) + r % 2) / 2 = r % 16 / 4 := by omega
  have hmod : (2 * (r % 16 / 4) + r % 2) % 2 = r % 2 := by omega
  rw [hdiv, hmod]
  have hl : r % 16 % 4 = 2 * (r / 2 % 2) + r % 2 := by omega
  have h := Nat.div_add_mod (r % 16) 4
  omega

theorem load_decodes (p : WideThread) : decodeLocation (loadLocation p) = wideColIndex p := by
  have hr := p.reg.isLt
  let base := (colProjection p).1
  let within := 2 * (p.reg.val % 16 / 4) + p.reg.val % 2
  have hw : within < 8 := by unfold within; omega
  have hreg : base.reg.val = 16 * (p.reg.val / 32) + 8 * (p.reg.val / 16 % 2) + within := by
    change 16 * (p.reg.val / 32) + 8 * (p.reg.val / 16 % 2) +
      2 * (p.reg.val % 16 / 4) + p.reg.val % 2 = _
    simp only [within, Nat.add_assoc]
  have hrow : base.reg.val % 8 = within := by omega
  have hdiv : base.reg.val / 16 = p.reg.val / 32 := by omega
  have hcol : base.reg.val / 8 % 2 = p.reg.val / 16 % 2 := by omega
  have hr₁ : (transposedIndex base).1.val = 8 * p.warp.val + within := by
    change 8 * p.warp.val + base.reg.val % 8 = _
    rw [hrow]
  simp only [loadLocation, decode_lift, transferred_cell]
  apply Prod.ext
  · apply Fin.ext
    change (expandRow (transposedIndex base).1 (colProjection p).2).val = _
    rw [expand_row_value _ _ _ _ hr₁ hw]
    change 16 * p.warp.val + 4 * (within / 2) + 2 * (p.reg.val / 2 % 2) + within % 2 =
      16 * p.warp.val + p.reg.val % 16
    have hbits : 4 * (within / 2) + 2 * (p.reg.val / 2 % 2) + within % 2 = p.reg.val % 16 :=
      low_row_bits p.reg.val
    simpa only [Nat.add_assoc] using congrArg (16 * p.warp.val + ·) hbits
  · apply Fin.ext
    change 64 * (base.reg.val / 16) + 2 * p.lane.val + base.reg.val / 8 % 2 =
      64 * (p.reg.val / 32) + 2 * p.lane.val + p.reg.val / 16 % 2
    rw [hdiv, hcol]

private theorem decode_location_injective {a b : Location} (h : decodeLocation a = decodeLocation b) :
    a = b := by
  have hc := congrArg Prod.snd h
  have hr := congrArg (fun c : Cell => compressRow c.1) h
  simp only [decodeLocation, compress_expand] at hr hc
  have row := congrArg (fun c : Fin 64 × Fin 2 => c.1) hr
  have same : GluonShared.decode (baseLocation a) = GluonShared.decode (baseLocation b) :=
    Prod.ext row hc
  have baseEqual := GluonShared.decode_injective same
  have addr := congrArg GluonShared.ByteLocation.address baseEqual
  have hp := congrArg (fun loc : GluonShared.ByteLocation => loc.phase.val) baseEqual
  have hh := congrArg (fun r : Fin 64 × Fin 2 => r.2.val) hr
  simp only [baseLocation, rowHalf] at addr hp hh
  have phase : a.phase = b.phase := Fin.ext (by omega)
  cases a
  cases b
  congr

theorem shared_copy_matches (r c : Fin 128) :
    storeLocation (wideRowOwner (r,c)) = loadLocation (wideColOwner (r,c)) := by
  apply decode_location_injective
  rw [store_decodes, load_decodes, row_covers, col_covers]

theorem shared_between_warps (p q : WideThread) (h : p.warp ≠ q.warp) :
    (storeLocation p).address ≠ (loadLocation q).address :=
  GluonShared.separate_warps (rowProjection p).1 (colProjection q).1 h

/-- The compiler interleaves pairs of source rows, first the low 64
columns, then the high 64. Every overwrite follows a warp barrier. -/
def warpEvents : List (GluonShared.Event 4) :=
  [.write 0, .barrier, .read 0, .barrier, .write 1, .barrier, .read 1, .barrier,
   .write 2, .barrier, .read 2, .barrier, .write 3, .barrier, .read 3]

theorem phases_safe : GluonShared.checkedPhases warpEvents = some [0, 1, 2, 3] := rfl

def reference (c : Count) (bank : Nat → UInt8) (ids : Fin c.rows → ClassID)
    (row : Fin c.rows) (col : Channel) : (Nat × UInt8) × (Nat × UInt8) :=
  let byte := bank ((ids row).val * 768 + col.val)
  ((row.val * 768 + col.val, byte), (col.val * c.rows + row.val, byte))

def gathered (c : Count) (bank : Nat → UInt8) (ids : Fin c.rows → ClassID)
    (row : Fin c.rows) (col : Channel) : (Nat × UInt8) × (Nat × UInt8) :=
  let stride := ((ids row).val * 3298534883328 % 18446744073709551616) / 4294967296
  let byte := bank (stride + col.val)
  ((row.val * 768 + col.val, byte), (col.val * c.rows + row.val, byte))

theorem stores_equal (c : Count) (bank : Nat → UInt8) (ids : Fin c.rows → ClassID)
    (row : Fin c.rows) (col : Channel) :
    gathered c bank ids row col = reference c bank ids row col := by
  simp only [gathered, reference, compiled_class_stride]


structure Point (count : Count) where
  tileRow : Fin (count.rows / 128)
  tileCol : Fin 6
  rowLane : Fin 128
  colLane : Fin 128

def Point.row {count : Count} (p : Point count) : Fin count.rows :=
  ⟨p.tileRow.val * 128 + p.rowLane.val, by
    have hr := p.tileRow.isLt
    have hl := p.rowLane.isLt
    have hdiv : count.rows / 128 * 128 = count.rows := by cases count <;> decide
    omega⟩

def Point.column {count : Count} (p : Point count) : Channel :=
  ⟨p.tileCol.val * 128 + p.colLane.val, by
    have hc := p.tileCol.isLt
    have hl := p.colLane.isLt
    omega⟩

def point (count : Count) (r : Fin count.rows) (c : Channel) : Point count :=
  ⟨⟨r.val / 128, by
      have hr := r.isLt
      have exactRows : count.rows / 128 * 128 = count.rows := by cases count <;> decide
      omega⟩, ⟨c.val / 128, by have := c.isLt; omega⟩,
    ⟨r.val % 128, Nat.mod_lt _ (by decide)⟩, ⟨c.val % 128, Nat.mod_lt _ (by decide)⟩⟩

theorem point_covers (count : Count) (r : Fin count.rows) (c : Channel) :
    (point count r c).row = r ∧ (point count r c).column = c := by
  constructor <;> apply Fin.ext <;> simp only [point, Point.row, Point.column] <;> omega

theorem point_unique {count : Count} (p q : Point count)
    (hr : p.row = q.row) (hc : p.column = q.column) : p = q := by
  have er := congrArg Fin.val hr
  have ec := congrArg Fin.val hc
  simp only [Point.row, Point.column] at er ec
  have hp := p.rowLane.isLt
  have hq := q.rowLane.isLt
  have hcp := p.colLane.isLt
  have hcq := q.colLane.isLt
  have e₁ : p.tileRow = q.tileRow := Fin.ext (by omega)
  have e₂ : p.tileCol = q.tileCol := Fin.ext (by omega)
  have e₃ : p.rowLane = q.rowLane := Fin.ext (by omega)
  have e₄ : p.colLane = q.colLane := Fin.ext (by omega)
  cases p
  cases q
  congr

theorem stored_addresses_bounded {count : Count} (p : Point count) :
    p.row.val * 768 + p.column.val < 768 * count.rows ∧
      p.column.val * count.rows + p.row.val < 768 * count.rows ∧
      768 * count.rows < 2^31 := by
  have hr := p.row.isLt
  have hc := p.column.isLt
  change p.column.val < 768 at hc
  cases count <;> simp only [Count.rows] at * <;> omega

theorem output_owners_unique {count : Count} (p q : Point count) :
    (p.row.val * 768 + p.column.val = q.row.val * 768 + q.column.val → p = q) ∧
      (p.column.val * count.rows + p.row.val = q.column.val * count.rows + q.row.val → p = q) := by
  have hp := p.row.isLt
  have hq := q.row.isLt
  have hcp : p.column.val < 768 := p.column.isLt
  have hcq : q.column.val < 768 := q.column.isLt
  constructor
  · intro h
    exact point_unique p q (Fin.ext (by omega)) (Fin.ext (by omega))
  · intro h
    cases count <;> simp only [Count.rows] at hp hq h <;>
      exact point_unique p q (Fin.ext (by omega)) (Fin.ext (by omega))

def warpObserved {count : Count} (bank : Nat → UInt8) (ids : Fin count.rows → ClassID)
    (p : Point count) : (Nat × UInt8) × (Nat × UInt8) :=
  let cell := (p.rowLane, p.colLane)
  let (storeRow, storeCol) := wideRowIndex (wideRowOwner cell)
  let (readRow, readCol) := decodeLocation (loadLocation (wideColOwner cell))
  let rowSource := {p with rowLane := storeRow, colLane := storeCol}
  let colSource := {p with rowLane := readRow, colLane := readCol}
  ((gathered count bank ids rowSource.row rowSource.column).1,
    (gathered count bank ids colSource.row colSource.column).2)

theorem warp_observed_equal {count : Count} (bank : Nat → UInt8) (ids : Fin count.rows → ClassID)
    (p : Point count) :
    warpObserved bank ids p = reference count bank ids p.row p.column := by
  simp only [warpObserved, row_covers, load_decodes, col_covers, stores_equal]

#axiomsClean store_decodes
#axiomsClean load_decodes
#axiomsClean shared_copy_matches
#axiomsClean shared_between_warps
#axiomsClean phases_safe
#axiomsClean point_covers
#axiomsClean point_unique
#axiomsClean stored_addresses_bounded
#axiomsClean output_owners_unique
#axiomsClean warp_observed_equal
#axiomsClean compiled_class_stride
#axiomsClean expand_compress
#axiomsClean compress_expand
#axiomsClean stores_equal
end VeriTile.Bench.Models.NanoGPT.Track1.SampledRows
