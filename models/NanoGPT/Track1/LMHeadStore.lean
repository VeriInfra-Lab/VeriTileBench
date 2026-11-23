import models.NanoGPT.Track1.LMHeadPack

/-! Physical stores for the Hopper head refresh: sixteen consecutive bytes
per row store and four eight-byte columns after a warp-private exchange. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.LMHead.Store
open QKVPack QKVLaunch
open GluonShared (ByteLocation ldmatrixRow slot)

def rowIndex (p : ThreadValue) : TileIndex :=
  (⟨8 * p.warp.val + 2 * (p.lane.val / 8) + p.reg.val / 16, by
      have := p.warp.isLt
      have := p.lane.isLt
      have := p.reg.isLt
      omega⟩,
   ⟨16 * (p.lane.val % 8) + p.reg.val % 16, by omega⟩)

def rowOwner (i : TileIndex) : ThreadValue :=
  ⟨⟨i.1.val / 8, by have := i.1.isLt; omega⟩,
   ⟨8 * ((i.1.val % 8) / 2) + i.2.val / 16, by have := i.2.isLt; omega⟩,
   ⟨16 * (i.1.val % 2) + i.2.val % 16, by omega⟩⟩

def columnIndex (p : ThreadValue) : TileIndex :=
  (⟨8 * p.warp.val + p.reg.val % 8, by have := p.warp.isLt; omega⟩,
   ⟨4 * p.lane.val + p.reg.val / 8, by
      have := p.lane.isLt
      have := p.reg.isLt
      omega⟩)

def columnOwner (i : TileIndex) : ThreadValue :=
  ⟨⟨i.1.val / 8, by have := i.1.isLt; omega⟩,
   ⟨i.2.val / 4, by have := i.2.isLt; omega⟩,
   ⟨8 * (i.2.val % 4) + i.1.val % 8, by omega⟩⟩

theorem row_covers (i : TileIndex) : rowIndex (rowOwner i) = i := by
  have := i.1.isLt
  have := i.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [rowIndex, rowOwner] <;> omega

theorem column_covers (i : TileIndex) : columnIndex (columnOwner i) = i := by
  have := i.1.isLt
  have := i.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [columnIndex, columnOwner] <;> omega

private theorem thread_ext {p q : ThreadValue}
    (hw : p.warp = q.warp) (hl : p.lane = q.lane) (hr : p.reg = q.reg) : p = q := by
  cases p
  cases q
  congr

theorem row_owner_unique (p : ThreadValue) : rowOwner (rowIndex p) = p := by
  have := p.lane.isLt
  have := p.reg.isLt
  apply thread_ext <;> apply Fin.ext <;> simp only [rowIndex, rowOwner] <;> omega

theorem column_owner_unique (p : ThreadValue) : columnOwner (columnIndex p) = p := by
  have := p.reg.isLt
  apply thread_ext <;> apply Fin.ext <;> simp only [columnIndex, columnOwner] <;> omega

/-- The two vector stores occupy the same address in successive phases. -/
def storeLocation (p : ThreadValue) : ByteLocation :=
  ⟨⟨p.reg.val / 16, by have := p.reg.isLt; omega⟩,
   ⟨slot p.warp p.lane + p.reg.val % 16, by
      have := p.warp.isLt
      have := p.lane.isLt
      unfold slot
      omega⟩⟩

/-- PRMT assembles alternate row bytes from the two LDMATRIX results. -/
def loadLocation (p : ThreadValue) : ByteLocation :=
  let m : Fin 4 := ⟨p.reg.val % 8 / 2, by omega⟩
  ⟨⟨p.reg.val % 2, by omega⟩,
   ⟨ldmatrixRow p.warp p.lane m + 4 * (p.lane.val % 4) + p.reg.val / 8, by
      have := (GluonShared.ldmatrix_row_aligned p.warp p.lane m).2.2
      have := p.warp.isLt
      have := p.reg.isLt
      omega⟩⟩

def decode (loc : ByteLocation) : TileIndex :=
  let a := loc.address.val
  let lane := a / 16 % 32
  (⟨8 * (a / 512) + 2 * (lane / 8) + loc.phase.val, by
      have := loc.address.isLt
      have := loc.phase.isLt
      omega⟩,
   ⟨16 * (lane % 8) + a % 16, by omega⟩)

theorem store_decodes (p : ThreadValue) : decode (storeLocation p) = rowIndex p := by
  have := p.lane.isLt
  have := p.reg.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [decode, storeLocation, slot, rowIndex] <;> omega

theorem load_decodes (p : ThreadValue) : decode (loadLocation p) = columnIndex p := by
  have hl := p.lane.isLt
  have hr := p.reg.isLt
  let slotIndex := 8 * (p.reg.val % 8 / 2) + p.lane.val / 4
  let byteIndex := 4 * (p.lane.val % 4) + p.reg.val / 8
  have hs : slotIndex < 32 := by dsimp [slotIndex]; omega
  have hb : byteIndex < 16 := by dsimp [byteIndex]; omega
  let addr := 512 * p.warp.val + 16 * slotIndex + 4 * (p.lane.val % 4) + p.reg.val / 8
  have hn : addr = 512 * p.warp.val + 16 * slotIndex + byteIndex := by
    simp only [addr, byteIndex, Nat.add_assoc]
  have hd : addr / 16 = 32 * p.warp.val + slotIndex := by omega
  have hwarp : addr / 512 = p.warp.val := by omega
  have hslot : addr / 16 % 32 = slotIndex := by rw [hd]; omega
  have hbyte : addr % 16 = byteIndex := by omega
  apply Prod.ext <;> apply Fin.ext
  · change 8 * (addr / 512) + 2 * ((addr / 16 % 32) / 8) + p.reg.val % 2 =
      8 * p.warp.val + p.reg.val % 8
    rw [hwarp, hslot]
    dsimp [slotIndex]
    omega
  · change 16 * ((addr / 16 % 32) % 8) + addr % 16 = 4 * p.lane.val + p.reg.val / 8
    rw [hslot, hbyte]
    dsimp [slotIndex, byteIndex]
    omega

theorem decode_injective {a b : ByteLocation} (h : decode a = decode b) : a = b := by
  have hr := congrArg (fun i : TileIndex => i.1.val) h
  have hc := congrArg (fun i : TileIndex => i.2.val) h
  have := a.phase.isLt
  have := b.phase.isLt
  simp only [decode] at hr hc
  have phase : a.phase = b.phase := Fin.ext (by omega)
  have addr : a.address = b.address := Fin.ext (by omega)
  cases a
  cases b
  congr

theorem shared_layout_matches (i : TileIndex) :
    storeLocation (rowOwner i) = loadLocation (columnOwner i) :=
  decode_injective (by rw [store_decodes, row_covers, load_decodes, column_covers])

theorem store_unique {p q : ThreadValue} (h : storeLocation p = storeLocation q) : p = q := by
  have hd := congrArg decode h
  rw [store_decodes, store_decodes] at hd
  simpa only [row_owner_unique] using congrArg rowOwner hd

theorem separate_warps (p q : ThreadValue) (h : p.warp ≠ q.warp) :
    (storeLocation p).address ≠ (loadLocation q).address := by
  intro same
  have hs := congrArg Fin.val same
  have := p.lane.isLt
  have := q.lane.isLt
  have := q.reg.isLt
  simp only [storeLocation, loadLocation, slot, ldmatrixRow] at hs
  exact h (Fin.ext (by omega))

theorem pair_layout (warp : Fin 8) (lane : Fin 32) (pair : Fin 16) :
    let lo : ThreadValue := ⟨warp, lane, ⟨pair.val * 2, by omega⟩⟩
    let hi : ThreadValue := ⟨warp, lane, ⟨pair.val * 2 + 1, by omega⟩⟩
    (rowIndex lo).1 = (rowIndex hi).1 ∧
      (rowIndex hi).2.val = (rowIndex lo).2.val + 1 ∧
      (rowIndex lo).2.val % 2 = 0 := by
  dsimp [rowIndex]
  exact ⟨Fin.ext (by dsimp; omega), by omega, by omega⟩

-- Tile bases are signed 32-bit offsets, while the within-tile offsets
-- are zero-extended by PTX before the 64-bit pointer addition.
private def splitOffset (base offset : Nat) : Nat :=
  ((word base).signExtend 64 + (word offset).zeroExtend 64).toNat

private theorem split_offset (base offset : Nat) (h : base + offset < 2^31) :
    splitOffset base offset = base + offset := by
  have hb : base < 2^31 := by omega
  have hu : offset < 2^32 := by omega
  have hb' : base < 2^32 := by omega
  have hw : base + offset < 2^64 := by omega
  have hs : (word base).msb = false := by
    apply BitVec.msb_eq_false_iff_two_mul_lt.mpr
    rw [word, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hb']
    omega
  unfold splitOffset
  rw [BitVec.signExtend_eq_setWidth_of_msb_false hs]
  simp only [BitVec.zeroExtend, BitVec.toNat_add, BitVec.toNat_setWidth_of_le (by decide : 32 ≤ 64),
    word, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hb', Nat.mod_eq_of_lt hu, Nat.mod_eq_of_lt hw]

def rowAt (p : HeadPoint) (col : Nat) : Nat :=
  splitOffset (p.rowTile.val * 3219456 + p.colTile.val * 128) (p.rowLane.val * 50304 + col)

def rowAddr (p : HeadPoint) : Nat := rowAt p p.colLane.val

def columnAddr (p : HeadPoint) : Nat :=
  splitOffset (p.colTile.val * 98304 + p.rowTile.val * 64) (p.colLane.val * 768 + p.rowLane.val)

theorem row_addr (p : HeadPoint) : rowAddr p = rowAddress p := by
  have hb : p.batch.val < 1 := p.batch.isLt
  have hr : p.rowTile.val < 12 := p.rowTile.isLt
  have hc : p.colTile.val < 393 := p.colTile.isLt
  have hm : p.rowLane.val < 64 := p.rowLane.isLt
  have hn : p.colLane.val < 128 := p.colLane.isLt
  unfold rowAddr rowAt
  have limit : p.rowTile.val * 3219456 + p.colTile.val * 128 +
      (p.rowLane.val * 50304 + p.colLane.val) < 2147483648 := by omega
  rw [split_offset _ _ limit]
  change _ = p.batch.val * 38633472 + (p.rowTile.val * 64 + p.rowLane.val) * 50304 +
    (p.colTile.val * 128 + p.colLane.val)
  omega

theorem column_addr (p : HeadPoint) : columnAddr p = transposedAddress p := by
  have hb : p.batch.val < 1 := p.batch.isLt
  have hr : p.rowTile.val < 12 := p.rowTile.isLt
  have hc : p.colTile.val < 393 := p.colTile.isLt
  have hm : p.rowLane.val < 64 := p.rowLane.isLt
  have hn : p.colLane.val < 128 := p.colLane.isLt
  unfold columnAddr
  have limit : p.colTile.val * 98304 + p.rowTile.val * 64 +
      (p.colLane.val * 768 + p.rowLane.val) < 2147483648 := by omega
  rw [split_offset _ _ limit]
  change _ = p.batch.val * 38633472 + (p.colTile.val * 128 + p.colLane.val) * 768 +
    (p.rowTile.val * 64 + p.rowLane.val)
  omega

theorem input_addr (p : HeadPoint) :
    rowAddr p = (BitVec.ofNat 64 p.row * BitVec.ofNat 64 50304 + BitVec.ofNat 64 p.col).toNat := by
  have hb : p.batch.val < 1 := p.batch.isLt
  have hz : p.batch.val = 0 := by omega
  have hword : rowWord p = word p.row * word 50304 + word p.col := by
    have cols : geometry.cols = 50304 := rfl
    rw [rowWord, hz, cols]
    simp [word]
  rw [row_addr, ← (address_words p).1, hword, input_word]

def observe (emit : HeadPoint → Writes) (p : HeadPoint) : Option Writes :=
  let i := (p.rowLane, p.colLane)
  let quantized := fun t => emit (atLane p (rowIndex t))
  let row := quantized (rowOwner i)
  let consumer := columnOwner i
  let dest := columnIndex consumer
  let load := loadLocation consumer
  let producer := rowOwner (decode load)
  let col := quantized producer
  if GluonShared.checkedPhases GluonShared.warpEvents = some [0, 1] ∧
      storeLocation producer = load ∧ row.byte = col.byte then
    some ⟨row.rowOffset, columnAddr (atLane p dest), row.byte⟩
  else none

private theorem at_own_lane (p : HeadPoint) : atLane p (p.rowLane, p.colLane) = p := by
  cases p
  rfl

theorem observe_eq (emit : HeadPoint → Writes) (p : HeadPoint) :
    observe emit p = some ⟨(emit p).rowOffset, columnAddr p, (emit p).byte⟩ := by
  simp only [observe, load_decodes, column_covers, row_covers, at_own_lane,
    GluonShared.phase_order, shared_layout_matches, eq_self, and_self, if_true]

#axiomsClean row_covers
#axiomsClean column_covers
#axiomsClean row_owner_unique
#axiomsClean column_owner_unique
#axiomsClean shared_layout_matches
#axiomsClean store_unique
#axiomsClean separate_warps
#axiomsClean pair_layout
#axiomsClean row_addr
#axiomsClean column_addr
#axiomsClean input_addr
#axiomsClean observe_eq
end VeriTile.Bench.Models.NanoGPT.Track1.LMHead.Store
