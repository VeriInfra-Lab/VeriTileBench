import models.NanoGPT.Track1.QKVPack
import models.NanoGPT.Track1.QKVGluonShared

/-! Register ownership of the two Gluon layouts, and interpretation of
the row store and the store after the FP8-only layout conversion. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVPack
open QKVLaunch

private theorem thread_ext {p q : ThreadValue}
    (hw : p.warp = q.warp) (hl : p.lane = q.lane) (hr : p.reg = q.reg) : p = q := by
  cases p
  cases q
  congr

/-- Blocked layout [2,8] × [4,8] × [8,1], column order first. -/
def rowIndex (p : ThreadValue) : TileIndex :=
  (⟨8 * p.warp.val + 2 * (p.lane.val / 8) + (p.reg.val / 8) % 2, by
      have := p.warp.isLt
      have := p.lane.isLt
      omega⟩,
   ⟨64 * (p.reg.val / 16) + 8 * (p.lane.val % 8) + p.reg.val % 8, by have := p.reg.isLt; omega⟩)

/-- Blocked layout [8,2] × [1,32] × [8,1], row order first. -/
def transposedIndex (p : ThreadValue) : TileIndex :=
  (⟨8 * p.warp.val + p.reg.val % 8, by have := p.warp.isLt; omega⟩,
   ⟨64 * (p.reg.val / 16) + 2 * p.lane.val + (p.reg.val / 8) % 2, by
      have := p.lane.isLt
      have := p.reg.isLt
      omega⟩)

def rowOwner (i : TileIndex) : ThreadValue where
  warp := ⟨i.1.val / 8, by have := i.1.isLt; omega⟩
  lane := ⟨8 * ((i.1.val % 8) / 2) + (i.2.val % 64) / 8, by omega⟩
  reg := ⟨16 * (i.2.val / 64) + 8 * (i.1.val % 2) + i.2.val % 8, by have := i.2.isLt; omega⟩

def transposedOwner (i : TileIndex) : ThreadValue where
  warp := ⟨i.1.val / 8, by have := i.1.isLt; omega⟩
  lane := ⟨(i.2.val % 64) / 2, by omega⟩
  reg := ⟨16 * (i.2.val / 64) + 8 * (i.2.val % 2) + i.1.val % 8, by have := i.2.isLt; omega⟩

theorem row_covers (i : TileIndex) : rowIndex (rowOwner i) = i := by
  have hrow := i.1.isLt
  have hcol := i.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [rowIndex, rowOwner] <;> omega

theorem transposed_covers (i : TileIndex) : transposedIndex (transposedOwner i) = i := by
  have hrow := i.1.isLt
  have hcol := i.2.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [transposedIndex, transposedOwner] <;> omega

theorem row_owner_unique (p : ThreadValue) : rowOwner (rowIndex p) = p := by
  have hw := p.warp.isLt
  have hl := p.lane.isLt
  have hr := p.reg.isLt
  have e₁ : (rowOwner (rowIndex p)).warp = p.warp := Fin.ext (by simp only [rowIndex, rowOwner]; omega)
  have e₂ : (rowOwner (rowIndex p)).lane = p.lane := Fin.ext (by simp only [rowIndex, rowOwner]; omega)
  have e₃ : (rowOwner (rowIndex p)).reg = p.reg := Fin.ext (by simp only [rowIndex, rowOwner]; omega)
  exact thread_ext e₁ e₂ e₃

theorem transposed_owner_unique (p : ThreadValue) : transposedOwner (transposedIndex p) = p := by
  have hw := p.warp.isLt
  have hl := p.lane.isLt
  have hr := p.reg.isLt
  have e₁ : (transposedOwner (transposedIndex p)).warp = p.warp :=
    Fin.ext (by simp only [transposedIndex, transposedOwner]; omega)
  have e₂ : (transposedOwner (transposedIndex p)).lane = p.lane :=
    Fin.ext (by simp only [transposedIndex, transposedOwner]; omega)
  have e₃ : (transposedOwner (transposedIndex p)).reg = p.reg :=
    Fin.ext (by simp only [transposedIndex, transposedOwner]; omega)
  exact thread_ext e₁ e₂ e₃

set_option maxHeartbeats 0 in
set_option maxRecDepth 4096 in
private theorem checked_layout :
    ∀ (r : Fin 64) (c : Fin 128),
      GluonShared.storeLocation (rowOwner (r, c)) =
        GluonShared.loadLocation (transposedOwner (r, c)) := by
  decide +kernel

theorem shared_layout_matches (i : TileIndex) :
    GluonShared.storeLocation (rowOwner i) = GluonShared.loadLocation (transposedOwner i) :=
  checked_layout i.1 i.2

set_option maxHeartbeats 0 in
set_option maxRecDepth 4096 in
private theorem checked_decode :
    ∀ (w : Fin 8) (lane reg : Fin 32),
      GluonShared.decode (GluonShared.storeLocation ⟨w, lane, reg⟩) =
        rowIndex ⟨w, lane, reg⟩ := by
  decide +kernel

theorem shared_store_decodes (p : ThreadValue) :
    GluonShared.decode (GluonShared.storeLocation p) = rowIndex p :=
  checked_decode p.warp p.lane p.reg

theorem shared_store_injective {p q : ThreadValue}
    (h : GluonShared.storeLocation p = GluonShared.storeLocation q) : p = q := by
  have hindex := congrArg GluonShared.decode h
  rw [shared_store_decodes, shared_store_decodes] at hindex
  simpa only [row_owner_unique] using congrArg rowOwner hindex

theorem transferred_cell (p : ThreadValue) :
    GluonShared.decode (GluonShared.loadLocation p) = transposedIndex p := by
  rw [← transposed_owner_unique p, ← shared_layout_matches, shared_store_decodes, row_covers,
    transposed_covers]

theorem redistribution_preserves_byte (bytes : TileIndex → UInt8) (p : ThreadValue) :
    bytes (rowIndex (rowOwner (GluonShared.decode (GluonShared.loadLocation p)))) =
      bytes (transposedIndex p) := by
  rw [row_covers, transferred_cell]

def atLane {g : Geometry} (p : Point g) (i : TileIndex) : Point g :=
  { p with rowLane := i.1, colLane := i.2 }

structure ByteStore where
  offset : Nat
  value : UInt8
  deriving DecidableEq, Repr

/-- The row plane stores the directly quantized register tile. The column
plane first changes which thread holds each byte, then stores at col*M+row. -/
def gluonStores {g : Geometry} (emit : Point g → Writes) (p : Point g) :
    ByteStore × ByteStore :=
  let i := (p.rowLane, p.colLane)
  let quantized := fun t => emit (atLane p (rowIndex t))
  let row := quantized (rowOwner i)
  let consumer := transposedOwner i
  let dest := transposedIndex consumer
  let afterMove := quantized (rowOwner (GluonShared.decode (GluonShared.loadLocation consumer)))
  (⟨row.rowOffset, row.byte⟩, ⟨(transposeWord (atLane p dest)).toNat, afterMove.byte⟩)

theorem gluon_stores_equal (op : Arithmetic) (input : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) :
    gluonStores (emitted op input w) p =
      let e := emitted op input w p
      (⟨e.rowOffset, e.byte⟩, ⟨e.transposedOffset, e.byte⟩) := by
  simp only [gluonStores, row_covers, transposed_covers, transferred_cell]
  rfl

/-- Both observable byte writes are present and their values agree. -/
def observeGluon {g : Geometry} (emit : Point g → Writes) (p : Point g) : Option Writes :=
  let (row, col) := gluonStores emit p
  if row.value == col.value then some ⟨row.offset, col.offset, row.value⟩ else none

def gluonObservation (op : Arithmetic) (input : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) : Option Writes :=
  observeGluon (emitted op input w) p

theorem gluon_equiv (op : Arithmetic) (input : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) : gluonObservation op input w p = reference op input p := by
  simpa [gluonObservation, observeGluon, gluon_stores_equal] using emitted_equiv op input w p

#axiomsClean gluon_stores_equal
#axiomsClean gluon_equiv
#axiomsClean row_covers
#axiomsClean transposed_covers
#axiomsClean row_owner_unique
#axiomsClean transposed_owner_unique
#axiomsClean shared_layout_matches
#axiomsClean shared_store_injective
#axiomsClean transferred_cell
#axiomsClean redistribution_preserves_byte
end VeriTile.Bench.Models.NanoGPT.Track1.QKVPack
