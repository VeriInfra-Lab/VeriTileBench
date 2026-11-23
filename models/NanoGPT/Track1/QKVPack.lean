import models.NanoGPT.Track1.QKVLaunch
import models.NanoGPT.Track1.ScaleRefresh

/-! One element of the QKV dual-layout quantizer. The full-tile launch
selects a bank before constructing addresses and writes both output planes. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVPack
open VeriTile.Triton.FP QKVLaunch

abbrev RowsPerTile := 64
abbrev ColsPerTile := 128

structure Geometry where
  batches : Nat
  firstBlocks : Nat
  secondBlocks : Nat
  colBlocks : Nat
  firstStride : Nat
  secondStride : Nat

def Geometry.ofWidth (w : Width) : Geometry :=
  let l := Layout.ofWidth w
  ⟨l.batches, l.firstTiles / 24, l.secondTiles / 24, 6, l.firstStride, l.secondStride⟩

def Geometry.firstRows (g : Geometry) : Nat := g.firstBlocks * RowsPerTile
def Geometry.rows (g : Geometry) : Nat := (g.firstBlocks + g.secondBlocks) * RowsPerTile
def Geometry.cols (g : Geometry) : Nat := g.colBlocks * ColsPerTile

structure Point (g : Geometry) where
  batch : Fin g.batches
  rowTile : Fin (g.firstBlocks + g.secondBlocks)
  colTile : Fin g.colBlocks
  rowLane : Fin RowsPerTile
  colLane : Fin ColsPerTile

def Point.startRow {g : Geometry} (p : Point g) : Nat := p.rowTile.val * RowsPerTile
def Point.startCol {g : Geometry} (p : Point g) : Nat := p.colTile.val * ColsPerTile
def Point.row {g : Geometry} (p : Point g) : Nat := p.startRow + p.rowLane.val
def Point.col {g : Geometry} (p : Point g) : Nat := p.startCol + p.colLane.val

theorem full_tile {g : Geometry} (p : Point g) : p.row < g.rows ∧ p.col < g.cols := by
  have hr := p.rowTile.isLt
  have hc := p.colTile.isLt
  have hm := p.rowLane.isLt
  have hn := p.colLane.isLt
  unfold Point.row Point.col Point.startRow Point.startCol Geometry.rows Geometry.cols
  simp only [RowsPerTile, ColsPerTile] at *
  omega

theorem same_bank {g : Geometry} (p : Point g) : p.row < g.firstRows ↔ p.startRow < g.firstRows := by
  have hm := p.rowLane.isLt
  unfold Point.row Point.startRow Geometry.firstRows
  simp only [RowsPerTile] at *
  omega

structure Inputs where
  first : Nat → Value .bf16
  second : Nat → Value .bf16
  scales : Nat → Scalar

/-- The shared scalar instruction chain. Keeping the instruction semantics
parametric covers the same approximate GPU division on both paths as well
as an IEEE software implementation. No algebraic FP rewrite is needed. -/
structure Arithmetic where
  extend : Value .bf16 → Scalar
  divide : Scalar → Scalar → Scalar
  minNumber : Scalar → Scalar → Scalar
  maxNumber : Scalar → Scalar → Scalar
  toE4M3RNE : Scalar → UInt8

def quantize (op : Arithmetic) (x : Value .bf16) (scale : Scalar) : UInt8 :=
  op.toE4M3RNE (op.maxNumber (op.minNumber (op.divide (op.extend x) scale) (literal 448))
    (literal (-448)))

def referenceAddress {g : Geometry} (p : Point g) : Bool × Nat :=
  if p.row < g.firstRows then
    (true, p.batch.val * g.firstStride + (p.row * g.cols + p.col))
  else
    (false, p.batch.val * g.secondStride + ((p.row - g.firstRows) * g.cols + p.col))

def tileAddress {g : Geometry} (p : Point g) : Bool × Nat :=
  let qk := p.startRow < g.firstRows
  let pitch := if qk then g.firstStride else g.secondStride
  let bankRow := p.startRow - (if qk then 0 else g.firstRows)
  (qk, p.batch.val * pitch + bankRow * g.cols + p.rowLane.val * g.cols + p.startCol + p.colLane.val)

theorem selected_address {g : Geometry} (p : Point g) : tileAddress p = referenceAddress p := by
  by_cases h : p.startRow < g.firstRows
  · have hr := (same_bank p).mpr h
    simp only [referenceAddress, tileAddress, if_pos hr, h, decide_true]
    simp [Point.row, Point.col, Nat.add_mul, Nat.add_assoc]
  · have hr : ¬ p.row < g.firstRows := fun x => h ((same_bank p).mp x)
    have hrow : p.row - g.firstRows = p.startRow - g.firstRows + p.rowLane.val := by
      unfold Point.row
      omega
    simp only [referenceAddress, tileAddress, if_neg hr, hrow, h, decide_false]
    simp [Point.col, Nat.add_mul, Nat.add_assoc]

def rowAddress {g : Geometry} (p : Point g) : Nat :=
  p.batch.val * (g.rows * g.cols) + p.row * g.cols + p.col

def transposedAddress {g : Geometry} (p : Point g) : Nat :=
  p.batch.val * (g.rows * g.cols) + p.col * g.rows + p.row

structure Writes where
  rowOffset : Nat
  transposedOffset : Nat
  byte : UInt8
  deriving DecidableEq, Repr

def storePair {g : Geometry} (op : Arithmetic) (inputs : Inputs) (p : Point g)
    (addr : Bool × Nat) : Writes :=
  let x := (if addr.1 then inputs.first else inputs.second) addr.2
  ⟨rowAddress p, transposedAddress p, quantize op x (inputs.scales p.batch.val)⟩

def reference {g : Geometry} (op : Arithmetic) (inputs : Inputs) (p : Point g) : Option Writes :=
  if p.row < g.rows ∧ p.col < g.cols then some (storePair op inputs p (referenceAddress p)) else none

def tiled {g : Geometry} (op : Arithmetic) (inputs : Inputs) (p : Point g) : Writes :=
  storePair op inputs p (tileAddress p)

theorem writes_equal {g : Geometry} (op : Arithmetic) (inputs : Inputs) (p : Point g) :
    some (tiled op inputs p) = reference op inputs p := by
  simp only [reference, tiled, if_pos (full_tile p), selected_address]

/-- All legal rows and columns occur in the launched full tiles. -/
def point (g : Geometry) (batch : Fin g.batches) (r : Fin g.rows) (c : Fin g.cols) : Point g where
  batch := batch
  rowTile := ⟨r.val / RowsPerTile, by
    have h := r.isLt
    simp only [Geometry.rows, RowsPerTile] at h ⊢
    omega⟩
  colTile := ⟨c.val / ColsPerTile, by
    have h := c.isLt
    simp only [Geometry.cols, ColsPerTile] at h ⊢
    omega⟩
  rowLane := ⟨r.val % RowsPerTile, Nat.mod_lt _ (by decide)⟩
  colLane := ⟨c.val % ColsPerTile, Nat.mod_lt _ (by decide)⟩

theorem point_coordinates (g : Geometry) (b : Fin g.batches) (r : Fin g.rows) (c : Fin g.cols) :
    (point g b r c).row = r.val ∧ (point g b r c).col = c.val := by
  simp only [point, Point.row, Point.col, Point.startRow, Point.startCol]
  simpa only [Nat.mul_comm] using And.intro (Nat.div_add_mod r.val RowsPerTile) (Nat.div_add_mod c.val ColsPerTile)

theorem point_injective {g : Geometry} {p q : Point g}
    (hb : p.batch = q.batch) (hr : p.row = q.row) (hc : p.col = q.col) : p = q := by
  have hp := p.rowLane.isLt
  have hq := q.rowLane.isLt
  have hpc := p.colLane.isLt
  have hqc := q.colLane.isLt
  simp only [Point.row, Point.col, Point.startRow, Point.startCol] at hr hc
  simp only [RowsPerTile, ColsPerTile] at *
  have hr₀ : p.rowTile = q.rowTile := Fin.ext (by omega)
  have hr₁ : p.rowLane = q.rowLane := Fin.ext (by omega)
  have hc₀ : p.colTile = q.colTile := Fin.ext (by omega)
  have hc₁ : p.colLane = q.colLane := Fin.ext (by omega)
  cases p
  cases q
  congr

theorem no_collisions (w : Width) (p q : Point (Geometry.ofWidth w)) :
    (rowAddress p = rowAddress q → p = q) ∧
      (transposedAddress p = transposedAddress q → p = q) := by
  obtain ⟨hpr, hpc⟩ := full_tile p
  obtain ⟨hqr, hqc⟩ := full_tile q
  constructor
  all_goals
    intro he
    apply point_injective
    · apply Fin.ext
      cases w <;> simp only [rowAddress, transposedAddress, Geometry.ofWidth, Layout.ofWidth,
        Geometry.rows, Geometry.cols, RowsPerTile, ColsPerTile] at * <;> omega
    · cases w <;> simp only [rowAddress, transposedAddress, Geometry.ofWidth, Layout.ofWidth,
        Geometry.rows, Geometry.cols, RowsPerTile, ColsPerTile] at * <;> omega
    · cases w <;> simp only [rowAddress, transposedAddress, Geometry.ofWidth, Layout.ofWidth,
        Geometry.rows, Geometry.cols, RowsPerTile, ColsPerTile] at * <;> omega

theorem output_bounds (w : Width) (p : Point (Geometry.ofWidth w)) :
    let g := Geometry.ofWidth w
    rowAddress p < g.batches * g.rows * g.cols ∧
      transposedAddress p < g.batches * g.rows * g.cols := by
  have hb := p.batch.isLt
  obtain ⟨hr,hc⟩ := full_tile p
  cases w <;> simp only [rowAddress, transposedAddress, Geometry.ofWidth, Layout.ofWidth,
    Geometry.rows, Geometry.cols, RowsPerTile, ColsPerTile] at * <;> omega

theorem input_bounds (w : Width) (p : Point (Geometry.ofWidth w)) :
    let g := Geometry.ofWidth w
    let (first, offset) := tileAddress p
    offset < (g.batches - 1) * (if first then g.firstStride else g.secondStride) +
      (if first then g.firstRows else g.rows - g.firstRows) * g.cols ∧ offset < 2^31 := by
  have hb := p.batch.isLt
  obtain ⟨hr, hc⟩ := full_tile p
  simp only [selected_address, referenceAddress]
  split <;> cases w <;> simp only [Geometry.ofWidth, Layout.ofWidth, Geometry.firstRows,
    Geometry.rows, Geometry.cols, RowsPerTile, ColsPerTile, if_true, if_false, Bool.false_eq_true] at * <;> omega

theorem guard_bounds (w : Width) :
    let g := Geometry.ofWidth w
    g.batches * max (max g.firstStride g.secondStride) (g.rows * g.cols) < 2^31 := by
  cases w <;> decide

private theorem word_exact {n : Nat} (h : n < 2^31) : (word n).toNat = n := by
  unfold word
  rw [BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (by omega)

/-- Per-lane i32 offsets before the address additions are sign extended. -/
def tiledInputWord {g : Geometry} (p : Point g) : BitVec 32 :=
  let qk := p.startRow < g.firstRows
  let pitch := if qk then g.firstStride else g.secondStride
  let bankRow := word p.startRow - word (if qk then 0 else g.firstRows)
  word p.batch.val * word pitch + bankRow * word g.cols +
    word p.rowLane.val * word g.cols + word p.startCol + word p.colLane.val

theorem input_word (w : Width) (p : Point (Geometry.ofWidth w)) :
    tiledInputWord p = word (tileAddress p).2 := by
  have hmax : (Geometry.ofWidth w).firstRows < 2^32 := by cases w <;> decide
  by_cases h : p.startRow < (Geometry.ofWidth w).firstRows
  · simp only [tiledInputWord, tileAddress, h, if_true, Nat.sub_zero, word,
      BitVec.ofNat_add, BitVec.ofNat_mul, BitVec.sub_zero]
  · have hle := Nat.le_of_not_gt h
    simp only [tiledInputWord, tileAddress, h, if_false, word, BitVec.ofNat_add,
      BitVec.ofNat_mul, BitVec.ofNat_sub_ofNat_of_le _ _ hmax hle]

def rowWord {g : Geometry} (p : Point g) : BitVec 32 :=
  word p.batch.val * word (g.rows * g.cols) + word p.row * word g.cols + word p.col

def transposeWord {g : Geometry} (p : Point g) : BitVec 32 :=
  word p.batch.val * word (g.rows * g.cols) + word p.col * word g.rows + word p.row

private theorem row_word {g : Geometry} (p : Point g) : rowWord p = word (rowAddress p) := by
  simp only [rowWord, rowAddress, word, BitVec.ofNat_add, BitVec.ofNat_mul]

private theorem transpose_word {g : Geometry} (p : Point g) :
    transposeWord p = word (transposedAddress p) := by
  simp only [transposeWord, transposedAddress, word, BitVec.ofNat_add, BitVec.ofNat_mul]

def emitted (op : Arithmetic) (inputs : Inputs) (w : Width) (p : Point (Geometry.ofWidth w)) : Writes :=
  let bank := if p.startRow < (Geometry.ofWidth w).firstRows then inputs.first else inputs.second
  let x := bank (tiledInputWord p).toNat
  ⟨(rowWord p).toNat, (transposeWord p).toNat, quantize op x (inputs.scales p.batch.val)⟩

theorem emitted_equiv (op : Arithmetic) (inputs : Inputs) (w : Width) (p : Point (Geometry.ofWidth w)) :
    some (emitted op inputs w p) = reference op inputs p := by
  have hbytes : (Geometry.ofWidth w).batches * (Geometry.ofWidth w).rows *
      (Geometry.ofWidth w).cols < 2^31 := by cases w <;> decide
  obtain ⟨hr, hc⟩ := output_bounds w p
  have hi := (input_bounds w p).2
  rw [← writes_equal]
  simp only [emitted, input_word w p, word_exact hi, row_word, transpose_word,
    word_exact (Nat.lt_trans hr hbytes), word_exact (Nat.lt_trans hc hbytes), tiled, storePair, tileAddress]
  by_cases h : p.startRow < (Geometry.ofWidth w).firstRows <;> simp [h]

#axiomsClean input_bounds
#axiomsClean guard_bounds
#axiomsClean emitted_equiv
#axiomsClean full_tile
#axiomsClean selected_address
#axiomsClean writes_equal
#axiomsClean point_coordinates
#axiomsClean point_injective
#axiomsClean no_collisions
#axiomsClean output_bounds
end VeriTile.Bench.Models.NanoGPT.Track1.QKVPack
