import models.NanoGPT.Track1.QKVNative

/-! The grouped packer emits a 16-byte row store for each of the two
rows owned by a lane. The warp exchanges the even and odd rows in
separate phases and rebuilds each transposed vector with byte selectors. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVPack.Vector
open QKVLaunch
open GluonShared (ByteLocation ldmatrixRow slot)

def rowIndex (p : ThreadValue) : TileIndex :=
  (⟨8 * p.warp.val + 2 * (p.lane.val / 8) + p.reg.val / 16, by
      have := p.warp.isLt
      have := p.lane.isLt
      have := p.reg.isLt
      omega⟩,
   ⟨16 * (p.lane.val % 8) + p.reg.val % 16, by omega⟩)

def transposedIndex (p : ThreadValue) : TileIndex :=
  (⟨8 * p.warp.val + p.reg.val % 8, by have := p.warp.isLt; omega⟩,
   ⟨4 * p.lane.val + p.reg.val / 8, by have := p.lane.isLt; have := p.reg.isLt; omega⟩)

def rowOwner (i : TileIndex) : ThreadValue where
  warp := ⟨i.1.val / 8, by have := i.1.isLt; omega⟩
  lane := ⟨8 * ((i.1.val % 8) / 2) + i.2.val / 16, by have := i.2.isLt; omega⟩
  reg := ⟨16 * (i.1.val % 2) + i.2.val % 16, by omega⟩

def transposedOwner (i : TileIndex) : ThreadValue where
  warp := ⟨i.1.val / 8, by have := i.1.isLt; omega⟩
  lane := ⟨i.2.val / 4, by have := i.2.isLt; omega⟩
  reg := ⟨8 * (i.2.val % 4) + i.1.val % 8, by omega⟩

theorem row_covers (i : TileIndex) : rowIndex (rowOwner i) = i := by
  have := i.1.isLt
  have := i.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [rowIndex, rowOwner] <;> omega

theorem transposed_covers (i : TileIndex) : transposedIndex (transposedOwner i) = i := by
  have := i.1.isLt
  have := i.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [transposedIndex, transposedOwner] <;> omega

private theorem thread_ext {p q : ThreadValue}
    (w : p.warp = q.warp) (l : p.lane = q.lane) (r : p.reg = q.reg) : p = q := by
  cases p
  cases q
  congr

theorem row_owner_unique (p : ThreadValue) : rowOwner (rowIndex p) = p := by
  have := p.warp.isLt
  have := p.lane.isLt
  have := p.reg.isLt
  apply thread_ext <;> apply Fin.ext <;> simp only [rowOwner, rowIndex] <;> omega

theorem transposed_owner_unique (p : ThreadValue) : transposedOwner (transposedIndex p) = p := by
  have := p.warp.isLt
  have := p.lane.isLt
  have := p.reg.isLt
  apply thread_ext <;> apply Fin.ext <;> simp only [transposedOwner, transposedIndex] <;> omega

def storeLocation (p : ThreadValue) : ByteLocation :=
  ⟨GluonShared.phase p, ⟨slot p.warp p.lane + p.reg.val % 16, by
    have := p.warp.isLt
    have := p.lane.isLt
    simp only [slot]
    omega⟩⟩

/-- The three prmt instructions take byte k from each row parity of
matrix j, then take the same two bytes from matrix j+1. -/
private def transposedSelector (reg : Fin 32) : Fin 2 × Fin 4 × Fin 4 :=
  (⟨reg.val % 2, by omega⟩, ⟨(reg.val % 8) / 2, by omega⟩, ⟨reg.val / 8, by have := reg.isLt; omega⟩)

private def permute (a b : Fin 4 → Nat) (selector : Nat) : Fin 4 → Nat := fun i =>
  let source := (selector / 16 ^ i.val) % 8
  if h : source < 4 then a ⟨source, h⟩ else b ⟨source % 4, by omega⟩

private def selectors (column : Fin 4) : Nat × Nat × Nat :=
  match column.val with
  | 0 => (0x40, 0x410, 0x4210)
  | 1 => (0x51, 0x510, 0x5210)
  | 2 => (0x62, 0x610, 0x6210)
  | _ => (0x73, 0x710, 0x7210)

private def prmtRoute (reg : Fin 32) : Nat :=
  let col : Fin 4 := ⟨reg.val / 8, by have := reg.isLt; omega⟩
  let group := (reg.val % 8) / 4
  let pick (phase matrix : Nat) : Fin 4 → Nat := fun k => 16 * phase + 4 * matrix + k.val
  let (a,b,c) := selectors col
  let x := permute (pick 0 (2*group)) (pick 1 (2*group)) a
  let y := permute x (pick 0 (2*group+1)) b
  permute y (pick 1 (2*group+1)) c ⟨reg.val % 4, by omega⟩

private theorem prmt_route (reg : Fin 32) :
    prmtRoute reg =
      16 * (transposedSelector reg).1.val + 4 * (transposedSelector reg).2.1.val +
        (transposedSelector reg).2.2.val := by
  decide +kernel +revert

private def decodePRMT (reg : Fin 32) : Fin 2 × Fin 4 × Fin 4 :=
  let tag := prmtRoute reg
  (⟨tag / 16, by have := prmt_route reg
                 have := (transposedSelector reg).1.isLt
                 have := (transposedSelector reg).2.1.isLt
                 have := (transposedSelector reg).2.2.isLt
                 omega⟩,
   ⟨(tag % 16) / 4, by omega⟩, ⟨tag % 4, by omega⟩)

def loadLocation (p : ThreadValue) : ByteLocation :=
  let (phase, matrix, column) := decodePRMT p.reg
  ⟨phase, ⟨ldmatrixRow p.warp p.lane matrix + 4 * (p.lane.val % 4) + column.val, by
    have hrow := (GluonShared.ldmatrix_row_aligned p.warp p.lane matrix).2.2
    have := p.warp.isLt
    have := column.isLt
    omega⟩⟩

def decode (loc : ByteLocation) : TileIndex :=
  let address := loc.address.val
  let lane := (address / 16) % 32
  (⟨8 * (address / 512) + 2 * (lane / 8) + loc.phase.val, by
    have := loc.address.isLt
    have := loc.phase.isLt
    omega⟩,
   ⟨16 * (lane % 8) + address % 16, by omega⟩)

theorem store_decodes (p : ThreadValue) : decode (storeLocation p) = rowIndex p := by
  have := p.warp.isLt
  have := p.lane.isLt
  have := p.reg.isLt
  apply Prod.ext <;> apply Fin.ext <;>
    simp only [decode, storeLocation, GluonShared.phase, slot, rowIndex] <;> omega

private theorem checked_load :
    ∀ (warp : Fin 8) (lane reg : Fin 32),
      decode (loadLocation ⟨warp,lane,reg⟩) = transposedIndex ⟨warp,lane,reg⟩ := by
  decide +kernel

theorem load_decodes (p : ThreadValue) : decode (loadLocation p) = transposedIndex p :=
  checked_load p.warp p.lane p.reg

private theorem decode_injective {p q : ByteLocation} (h : decode p = decode q) : p = q := by
  have hr := congrArg (fun i : TileIndex => i.1.val) h
  have hc := congrArg (fun i : TileIndex => i.2.val) h
  have := p.address.isLt
  have := q.address.isLt
  have := p.phase.isLt
  have := q.phase.isLt
  simp only [decode] at hr hc
  have hp : p.phase = q.phase := Fin.ext (by omega)
  have ha : p.address = q.address := Fin.ext (by omega)
  cases p
  cases q
  congr

theorem shared_match (i : TileIndex) :
    storeLocation (rowOwner i) = loadLocation (transposedOwner i) := by
  apply decode_injective
  rw [store_decodes, load_decodes, row_covers, transposed_covers]

private def splitOffset (base offset : Nat) : Nat :=
  ((word base).signExtend 64 + (word offset).zeroExtend 64).toNat

private theorem split_offset (base offset : Nat) (h : base + offset < 2^31) :
    splitOffset base offset = base + offset := by
  have hb : base < 2^32 := by omega
  have ho : offset < 2^32 := by omega
  have hw : base + offset < 2^64 := by omega
  have hs : (word base).msb = false := by
    apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
    rw [word, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hb]
    omega
  unfold splitOffset
  rw [BitVec.signExtend_eq_setWidth_of_msb_false hs]
  simp only [BitVec.zeroExtend, BitVec.toNat_add, BitVec.toNat_setWidth_of_le (by decide : 32 ≤ 64),
    word, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hb, Nat.mod_eq_of_lt ho, Nat.mod_eq_of_lt hw]

def inputOffset (origin : Nat) (w : Width) (p : Point (Geometry.ofWidth w)) : Nat :=
  let qk := p.startRow < (Geometry.ofWidth w).firstRows
  let bankRow := (p.startRow - (if qk then 0 else (Geometry.ofWidth w).firstRows)) *
    QKVNative.rowFactor (QKVNative.cropped w qk)
  splitOffset ((origin + p.batch.val) * QKVNative.BankPitch + bankRow * 768 + p.startCol)
    (p.rowLane.val * 768 + p.colLane.val)

def rowOffset (w : Width) (p : Point (Geometry.ofWidth w)) : Nat :=
  let g := Geometry.ofWidth w
  splitOffset ((p.batch.val * g.rows + p.startRow) * 768 + p.startCol)
    (p.rowLane.val * 768 + p.colLane.val)

def columnOffset (w : Width) (p : Point (Geometry.ofWidth w)) : Nat :=
  let g := Geometry.ofWidth w
  let column := p.colLane.val * g.rows
  let warpRow := p.rowLane.val / 8 * 8
  splitOffset (p.batch.val * g.rows * 768 + p.startCol * g.rows + p.startRow)
    (column ||| warpRow) + p.rowLane.val % 8

private theorem or_add (high low : Nat) (hh : high % 64 = 0) (hl : low < 64) :
    high ||| low = high + low := by
  have hd : (high ||| low) / 64 = high / 64 := by
    rw [show 64 = 2^6 by decide, Nat.or_div_two_pow]
    rw [show low / 2^6 = 0 by omega, Nat.or_zero]
  have hm : (high ||| low) % 64 = low := by
    rw [show 64 = 2^6 by decide, Nat.or_mod_two_pow]
    rw [show high % 2^6 = 0 from hh, Nat.zero_or, Nat.mod_eq_of_lt hl]
  omega

theorem row_offset (w : Width) (p : Point (Geometry.ofWidth w)) :
    rowOffset w p = (rowWord p).toNat := by
  have hr := (full_tile p).1
  have hc := (full_tile p).2
  have hb := p.batch.isLt
  have rowWord_eq : rowWord p = word (rowAddress p) := by
    simp only [rowWord, rowAddress, word, BitVec.ofNat_add, BitVec.ofNat_mul]
  rw [rowWord_eq]
  simp only [rowOffset]
  have haddr : rowAddress p < 2^31 := by
    cases w <;> simp only [rowAddress, Geometry.ofWidth, Layout.ofWidth, Geometry.rows, Geometry.cols,
      QKVPack.RowsPerTile, QKVPack.ColsPerTile] at * <;> omega
  change _ = (word (rowAddress p)).toNat
  rw [word, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
  have hoc : (Geometry.ofWidth w).cols = 768 := rfl
  have sum : (p.batch.val * (Geometry.ofWidth w).rows + p.startRow) * 768 + p.startCol +
      (p.rowLane.val * 768 + p.colLane.val) = rowAddress p := by
    simp only [rowAddress, Point.row, Point.col, hoc, Nat.add_mul, Nat.mul_assoc]
    omega
  rw [split_offset _ _ (sum ▸ haddr), sum]

theorem column_offset (w : Width) (p : Point (Geometry.ofWidth w)) :
    columnOffset w p = (transposeWord p).toNat := by
  have hr := (full_tile p).1
  have hc := (full_tile p).2
  have hb := p.batch.isLt
  have hw : p.rowLane.val < 64 := p.rowLane.isLt
  have hh : p.colLane.val * (Geometry.ofWidth w).rows % 64 = 0 := by
    cases w <;> simp only [Geometry.ofWidth, Geometry.rows, Layout.ofWidth, QKVPack.RowsPerTile] <;> omega
  have hl : p.rowLane.val / 8 * 8 < 64 := by omega
  have transposeWord_eq : transposeWord p = word (transposedAddress p) := by
    simp only [transposeWord, transposedAddress, word, BitVec.ofNat_add, BitVec.ofNat_mul]
  rw [transposeWord_eq]
  simp only [columnOffset, or_add _ _ hh hl]
  have haddr : transposedAddress p < 2^31 := by
    cases w <;> simp only [transposedAddress, Geometry.ofWidth, Layout.ofWidth, Geometry.rows, Geometry.cols,
      QKVPack.RowsPerTile, QKVPack.ColsPerTile] at * <;> omega
  have exactAddress : (word (transposedAddress p)).toNat = transposedAddress p := by
    rw [word, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
  rw [exactAddress]
  have hoc : (Geometry.ofWidth w).cols = 768 := rfl
  have sum : p.batch.val * (Geometry.ofWidth w).rows * 768 + p.startCol * (Geometry.ofWidth w).rows +
      p.startRow + (p.colLane.val * (Geometry.ofWidth w).rows + p.rowLane.val / 8 * 8) +
      p.rowLane.val % 8 = transposedAddress p := by
    simp only [transposedAddress, Point.row, Point.col, hoc, Nat.add_mul, Nat.mul_assoc]
    omega
  rw [split_offset _ _ (by omega), sum]

theorem input_offset (origin : Nat) (w : Width) (p : Point (Geometry.ofWidth w))
    (hslot : origin + p.batch.val < 7) :
    inputOffset origin w p = (QKVNative.packWord origin w p).toNat := by
  have hr := (full_tile p).1
  have hc := (full_tile p).2
  have hstart : p.startRow ≤ p.row := Nat.le_add_right _ _
  have hstartCol : p.startCol ≤ p.col := Nat.le_add_right _ _
  have hfirst : (Geometry.ofWidth w).firstRows < 2^32 := by cases w <;> decide
  let first := if p.startRow < (Geometry.ofWidth w).firstRows then 0 else (Geometry.ofWidth w).firstRows
  have hsub : first ≤ p.startRow := by
    dsimp [first]
    split <;> omega
  have hfirst' : first < 2^32 := by
    dsimp [first]
    split <;> omega
  let factor := QKVNative.rowFactor (QKVNative.cropped w (p.startRow < (Geometry.ofWidth w).firstRows))
  have hfactor : factor = 1 ∨ factor = 2 := by unfold factor QKVNative.rowFactor; split <;> simp
  let base := (origin + p.batch.val) * QKVNative.BankPitch + (p.startRow - first) * factor * 768 + p.startCol
  let localOffset := p.rowLane.val * 768 + p.colLane.val
  have hb : base + localOffset < 2^31 := by
    rcases hfactor with h | h <;> dsimp only [base, localOffset] <;> rw [h] <;>
      cases w <;> simp only [QKVNative.BankPitch, Geometry.ofWidth, Layout.ofWidth, Geometry.rows, Geometry.cols,
        QKVPack.RowsPerTile, QKVPack.ColsPerTile, Nat.mul_one] at * <;> omega
  have exactWord : QKVNative.packWord origin w p = word (base + localOffset) := by
    change word (origin + p.batch.val) * word QKVNative.BankPitch +
      (word p.startRow - word first) * word factor * word 768 + word p.rowLane.val * word 768 +
      word p.startCol + word p.colLane.val = _
    rw [show word p.startRow - word first = word (p.startRow - first) from
      BitVec.ofNat_sub_ofNat_of_le _ _ hfirst' hsub]
    simp only [base, localOffset, word, BitVec.ofNat_add, BitVec.ofNat_mul]
    ac_rfl
  change splitOffset base localOffset = _
  rw [split_offset _ _ hb, exactWord, word, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]

def packCell (origin : Nat) (op : Arithmetic) (raw : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) : Writes :=
  let bank := if p.startRow < (Geometry.ofWidth w).firstRows then raw.first else raw.second
  let x := bank (inputOffset origin w p)
  ⟨rowOffset w p, columnOffset w p, quantize op x (raw.scales p.batch.val)⟩

def observe (origin : Nat) (op : Arithmetic) (raw : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) : Option Writes :=
  let i := (p.rowLane, p.colLane)
  let quantized := fun t => packCell origin op raw w (atLane p (rowIndex t))
  let row := quantized (rowOwner i)
  let consumer := transposedOwner i
  let dest := atLane p (transposedIndex consumer)
  let afterMove := quantized (rowOwner (decode (loadLocation consumer)))
  if row.byte == afterMove.byte then some ⟨row.rowOffset, columnOffset w dest, row.byte⟩ else none

theorem observe_eq (origin : Nat) (op : Arithmetic) (raw : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) (hslots : origin + (Layout.ofWidth w).batches ≤ 7) :
    observe origin op raw w p = reference op (QKVNative.compactInputs w (QKVNative.inputsView origin raw)) p := by
  have hslot : origin + p.batch.val < 7 :=
    Nat.lt_of_lt_of_le (Nat.add_lt_add_left p.batch.isLt _) hslots
  have same : packCell origin op raw w p = QKVNative.packFrom origin op raw w p := by
    simp only [packCell, QKVNative.packFrom, input_offset _ _ _ hslot, row_offset, column_offset]
  rw [← emitted_equiv, ← QKVNative.pack_from_bytes _ _ _ _ _ hslot]
  simp only [observe, ← shared_match (p.rowLane, p.colLane), store_decodes, row_covers,
    transposed_covers, atLane]
  change (if (packCell origin op raw w p).byte == (packCell origin op raw w p).byte then
    some (Writes.mk (packCell origin op raw w p).rowOffset (columnOffset w p) (packCell origin op raw w p).byte)
    else none) = _
  rw [same, column_offset]
  simp [QKVNative.packFrom]

#axiomsClean prmt_route
#axiomsClean row_owner_unique
#axiomsClean transposed_owner_unique
#axiomsClean shared_match
#axiomsClean input_offset
#axiomsClean row_offset
#axiomsClean column_offset
#axiomsClean observe_eq
end VeriTile.Bench.Models.NanoGPT.Track1.QKVPack.Vector
