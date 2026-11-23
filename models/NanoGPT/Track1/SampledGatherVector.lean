import models.NanoGPT.Track1.SampledGather

/-! The 16-byte lanes of the Hopper sampled gather. Four phases move
four input rows per lane through each warp's private exchange slots. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.SampledRows.Vector
open QKVPack
open GluonShared (slot ldmatrixRow)

def rowIndex (p : WideThread) : Cell :=
  (⟨16 * p.warp.val + 4 * (p.lane.val / 8) + p.reg.val / 16, by
      have := p.warp.isLt
      have := p.lane.isLt
      have := p.reg.isLt
      omega⟩,
   ⟨16 * (p.lane.val % 8) + p.reg.val % 16, by omega⟩)

def columnIndex (p : WideThread) : Cell :=
  (⟨16 * p.warp.val + p.reg.val % 16, by have := p.warp.isLt; omega⟩,
   ⟨4 * p.lane.val + p.reg.val / 16, by
      have := p.lane.isLt
      have := p.reg.isLt
      omega⟩)

def rowOwner (i : Cell) : WideThread :=
  ⟨⟨i.1.val / 16, by have := i.1.isLt; omega⟩,
   ⟨8 * (i.1.val % 16 / 4) + i.2.val / 16, by have := i.2.isLt; omega⟩,
   ⟨16 * (i.1.val % 4) + i.2.val % 16, by omega⟩⟩

def columnOwner (i : Cell) : WideThread :=
  ⟨⟨i.1.val / 16, by have := i.1.isLt; omega⟩,
   ⟨i.2.val / 4, by have := i.2.isLt; omega⟩,
   ⟨16 * (i.2.val % 4) + i.1.val % 16, by omega⟩⟩

theorem row_covers (i : Cell) : rowIndex (rowOwner i) = i := by
  have := i.1.isLt
  have := i.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [rowIndex, rowOwner] <;> omega

theorem column_covers (i : Cell) : columnIndex (columnOwner i) = i := by
  have := i.1.isLt
  have := i.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [columnIndex, columnOwner] <;> omega

private theorem thread_ext {p q : WideThread}
    (hw : p.warp = q.warp) (hl : p.lane = q.lane) (hr : p.reg = q.reg) : p = q := by
  cases p
  cases q
  congr

theorem row_owner_unique (p : WideThread) : rowOwner (rowIndex p) = p := by
  have := p.lane.isLt
  have := p.reg.isLt
  apply thread_ext <;> apply Fin.ext <;> simp only [rowIndex, rowOwner] <;> omega

theorem column_owner_unique (p : WideThread) : columnOwner (columnIndex p) = p := by
  have := p.reg.isLt
  apply thread_ext <;> apply Fin.ext <;> simp only [columnIndex, columnOwner] <;> omega

/-- Each phase stores exactly one 16-byte row fragment per lane. -/
def storeLocation (p : WideThread) : Location :=
  ⟨⟨p.reg.val / 16, by have := p.reg.isLt; omega⟩,
   ⟨slot p.warp p.lane + p.reg.val % 16, by
      have := p.warp.isLt
      have := p.lane.isLt
      unfold slot
      omega⟩⟩

/-- LDMATRIX selects one of four matrix rows. PRMT assembles its four
phase bytes at consecutive row offsets in the destination column. -/
def loadLocation (p : WideThread) : Location :=
  let matrix : Fin 4 := ⟨p.reg.val % 16 / 4, by omega⟩
  ⟨⟨p.reg.val % 4, by omega⟩,
   ⟨ldmatrixRow p.warp p.lane matrix + 4 * (p.lane.val % 4) + p.reg.val / 16, by
      have := (GluonShared.ldmatrix_row_aligned p.warp p.lane matrix).2.2
      have := p.warp.isLt
      have := p.reg.isLt
      omega⟩⟩

def decode (loc : Location) : Cell :=
  let a := loc.address.val
  let lane := a / 16 % 32
  (⟨16 * (a / 512) + 4 * (lane / 8) + loc.phase.val, by
      have := loc.address.isLt
      have := loc.phase.isLt
      omega⟩,
   ⟨16 * (lane % 8) + a % 16, by omega⟩)

theorem store_decodes (p : WideThread) : decode (storeLocation p) = rowIndex p := by
  have := p.lane.isLt
  have := p.reg.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [decode, storeLocation, slot, rowIndex] <;> omega

theorem load_decodes (p : WideThread) : decode (loadLocation p) = columnIndex p := by
  have hl := p.lane.isLt
  have hr := p.reg.isLt
  let lane := 8 * (p.reg.val % 16 / 4) + p.lane.val / 4
  let byte := 4 * (p.lane.val % 4) + p.reg.val / 16
  have hlane : lane < 32 := by dsimp [lane]; omega
  have hbyte : byte < 16 := by dsimp [byte]; omega
  let addr := 512 * p.warp.val + 16 * lane + 4 * (p.lane.val % 4) + p.reg.val / 16
  have hn : addr = 512 * p.warp.val + 16 * lane + byte := by
    simp only [addr, byte, Nat.add_assoc]
  have hd : addr / 16 = 32 * p.warp.val + lane := by omega
  have hwarp : addr / 512 = p.warp.val := by omega
  have hslot : addr / 16 % 32 = lane := by rw [hd]; omega
  have hrem : addr % 16 = byte := by omega
  apply Prod.ext <;> apply Fin.ext
  · change 16 * (addr / 512) + 4 * ((addr / 16 % 32) / 8) + p.reg.val % 4 =
      16 * p.warp.val + p.reg.val % 16
    rw [hwarp, hslot]
    dsimp [lane]
    omega
  · change 16 * ((addr / 16 % 32) % 8) + addr % 16 = 4 * p.lane.val + p.reg.val / 16
    rw [hslot, hrem]
    dsimp [lane, byte]
    omega

private theorem decode_injective {a b : Location} (h : decode a = decode b) : a = b := by
  have hr := congrArg (fun i : Cell => i.1.val) h
  have hc := congrArg (fun i : Cell => i.2.val) h
  have := a.phase.isLt
  have := b.phase.isLt
  simp only [decode] at hr hc
  have phase : a.phase = b.phase := Fin.ext (by omega)
  have addr : a.address = b.address := Fin.ext (by omega)
  cases a
  cases b
  congr

theorem shared_copy_matches (i : Cell) :
    storeLocation (rowOwner i) = loadLocation (columnOwner i) :=
  decode_injective (by rw [store_decodes, row_covers, load_decodes, column_covers])

theorem store_unique {p q : WideThread} (h : storeLocation p = storeLocation q) : p = q := by
  have hd := congrArg decode h
  rw [store_decodes, store_decodes] at hd
  simpa only [row_owner_unique] using congrArg rowOwner hd

theorem separate_warps (p q : WideThread) (h : p.warp ≠ q.warp) :
    (storeLocation p).address ≠ (loadLocation q).address := by
  intro same
  have hs := congrArg Fin.val same
  have := p.lane.isLt
  have := q.lane.isLt
  have := q.reg.isLt
  simp only [storeLocation, loadLocation, slot, ldmatrixRow] at hs
  exact h (Fin.ext (by omega))

def rowBase {c : Count} (p : Point c) : Nat :=
  p.tileRow.val * 128 * 768 + p.tileCol.val * 128

def columnBase {c : Count} (p : Point c) : Nat :=
  p.tileCol.val * 128 * c.rows + p.tileRow.val * 128

private def signedWord (n : Nat) : BitVec 64 := (BitVec.ofNat 32 n).signExtend 64

private theorem signed_word (n : Nat) (fits : n < 2^31) : (signedWord n).toNat = n := by
  have word : (BitVec.ofNat 32 n).toNat = n := by
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
  have msb : (BitVec.ofNat 32 n).msb = false :=
    BitVec.msb_eq_false_iff_two_mul_lt.mpr (by rw [word]; omega)
  unfold signedWord
  rw [BitVec.signExtend_eq_setWidth_of_msb_false msb, BitVec.toNat_setWidth_of_le (by decide), word]

private def splitOffset (base offset : Nat) : Nat :=
  (signedWord base + (BitVec.ofNat 32 offset).zeroExtend 64).toNat

private theorem split_offset (base offset : Nat) (fits : base + offset < 2^31) :
    splitOffset base offset = base + offset := by
  have sb := signed_word base (by omega)
  have su : offset < 2^32 := by omega
  have sum : base + offset < 2^64 := by omega
  simp only [splitOffset, BitVec.toNat_add, sb, BitVec.zeroExtend,
    BitVec.toNat_setWidth_of_le (by decide : 32 ≤ 64), BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt su, Nat.mod_eq_of_lt sum]

def rowOffset {c : Count} (p : Point c) : Nat :=
  splitOffset (rowBase p) (p.rowLane.val * 768 + p.colLane.val)

def columnOffset {c : Count} (p : Point c) : Nat :=
  splitOffset (columnBase p) (p.colLane.val * c.rows + p.rowLane.val)

theorem row_offset {c : Count} (p : Point c) :
    rowOffset p = p.row.val * 768 + p.column.val := by
  have shape : rowBase p + (p.rowLane.val * 768 + p.colLane.val) =
      p.row.val * 768 + p.column.val := by
    simp only [rowBase, Point.row, Point.column]
    omega
  have bound := stored_addresses_bounded p
  rw [rowOffset, split_offset _ _ (by rw [shape]; exact Nat.lt_trans bound.1 bound.2.2), shape]

theorem column_offset {c : Count} (p : Point c) :
    columnOffset p = p.column.val * c.rows + p.row.val := by
  have shape : columnBase p + (p.colLane.val * c.rows + p.rowLane.val) =
      p.column.val * c.rows + p.row.val := by
    simp only [columnBase, Point.row, Point.column, Nat.add_mul]
    omega
  have bound := stored_addresses_bounded p
  rw [columnOffset, split_offset _ _ (by rw [shape]; exact Nat.lt_trans bound.2.1 bound.2.2), shape]

theorem split_offsets_fit {c : Count} (p : Point c) :
    rowOffset p < 2^31 ∧ columnOffset p < 2^31 := by
  rw [row_offset, column_offset]
  rcases stored_addresses_bounded p with ⟨r, col, size⟩
  exact ⟨Nat.lt_trans r size, Nat.lt_trans col size⟩

private def copy {c : Count} (bank : Nat → UInt8) (ids : Fin c.rows → ClassID)
    (p : Point c) : (Nat × UInt8) × (Nat × UInt8) :=
  let index := (ids p.row).val * 768 + (p.tileCol.val * 128 + p.colLane.val)
  let byte := bank (signedWord index).toNat
  ((rowOffset p, byte), (columnOffset p, byte))

private theorem input_fits {c : Count} (ids : Fin c.rows → ClassID) (p : Point c) :
    (ids p.row).val * 768 + (p.tileCol.val * 128 + p.colLane.val) < 2^31 := by
  have hclass := (ids p.row).isLt
  have ht := p.tileCol.isLt
  have hl := p.colLane.isLt
  omega

theorem copy_equal {c : Count} (bank : Nat → UInt8) (ids : Fin c.rows → ClassID) (p : Point c) :
    copy bank ids p = reference c bank ids p.row p.column := by
  have fits := input_fits ids p
  simp only [copy, reference, row_offset, column_offset, signed_word _ fits]
  rfl

/-- Observe the row write and the column write after the exchange. -/
def observe {c : Count} (bank : Nat → UInt8) (ids : Fin c.rows → ClassID)
    (p : Point c) : (Nat × UInt8) × (Nat × UInt8) :=
  let tile := (p.rowLane, p.colLane)
  let load := loadLocation (columnOwner tile)
  let producer := rowOwner (decode load)
  let (row, col) := rowIndex (rowOwner tile)
  let (readRow, readCol) := rowIndex producer
  let rowSource := {p with rowLane := row, colLane := col}
  let colSource := {p with rowLane := readRow, colLane := readCol}
  if GluonShared.checkedPhases warpEvents = some [0, 1, 2, 3] ∧ storeLocation producer = load then
    ((copy bank ids rowSource).1, (copy bank ids colSource).2)
  else ((0, 0), (0, 0))

theorem observe_equal {c : Count} (bank : Nat → UInt8) (ids : Fin c.rows → ClassID)
    (p : Point c) : observe bank ids p = reference c bank ids p.row p.column := by
  simp only [observe, row_covers, load_decodes, column_covers, shared_copy_matches,
    phases_safe, eq_self, and_self, if_true, copy_equal]

#axiomsClean row_covers
#axiomsClean column_covers
#axiomsClean column_owner_unique
#axiomsClean shared_copy_matches
#axiomsClean store_unique
#axiomsClean separate_warps
#axiomsClean split_offsets_fit
#axiomsClean copy_equal
#axiomsClean observe_equal
end VeriTile.Bench.Models.NanoGPT.Track1.SampledRows.Vector
