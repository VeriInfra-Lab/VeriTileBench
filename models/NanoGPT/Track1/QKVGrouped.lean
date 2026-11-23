import models.NanoGPT.Track1.QKVVectorLayout

import models.NanoGPT.Track1.QKVScale

/-!
Seven layers share one launch for each of scan, scale and pack. Scan and
scale use a slot ordinal. The pack grid uses 30 bands of six row tiles
each and six column tiles. Each decoded point retains its original cache
buffer and its batch index within that buffer.
-/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Grouped
open QKVLaunch

def gridBand (width : Width) : Nat := match width with
  | .narrow => 0
  | .halfValue => 12
  | .wide => 18

def bandsPerSlot (width : Width) : Nat := match width with
  | .narrow => 4
  | .halfValue => 3
  | .wide => 6

def slotBase (width : Width) : Nat := match width with
  | .narrow => 0
  | .halfValue => 3
  | .wide => 5

def decodeWidth (band : Fin 30) : Width :=
  if band.val < 12 then .narrow else if band.val < 18 then .halfValue else .wide

def selectedBands (band : Fin 30) : BitVec 32 :=
  ((2#32 <<< BitVec.ofNat 32 band.val) - 1#32) &&& BitVec.ofNat 32 0x1049111

def slotDepth (mask : BitVec 32) : Nat :=
  (List.range 32).countP mask.getLsbD

def lastSlotBand (mask : BitVec 32) : Nat :=
  ((List.range 32).reverse.find? mask.getLsbD).getD 0

def fromBand (band : Fin 30) : Width × Nat × Nat :=
  let w := decodeWidth band
  let mask := selectedBands band
  (w, slotDepth mask - 1 - slotBase w, band.val - lastSlotBand mask)

theorem from_band_matches (band : Fin 30) :
    let w := decodeWidth band
    let localBand := band.val - gridBand w
    fromBand band = (w, localBand / bandsPerSlot w, localBand % bandsPerSlot w) := by
  decide +kernel +revert

theorem census (w : Width) :
    bandsPerSlot w * 6 = (QKVPack.Geometry.ofWidth w).firstBlocks +
      (QKVPack.Geometry.ofWidth w).secondBlocks := by
  cases w <;> decide

def toBand (w : Width) (batch within : Nat) (hb : batch < (Layout.ofWidth w).batches)
    (ht : within < bandsPerSlot w) : Fin 30 :=
  ⟨gridBand w + batch * bandsPerSlot w + within, by
    cases w <;> simp [gridBand, bandsPerSlot, Layout.ofWidth] at * <;> omega⟩

theorem from_to (w : Width) (batch : Fin (Layout.ofWidth w).batches) (within : Fin (bandsPerSlot w)) :
    fromBand (toBand w batch.val within.val batch.isLt within.isLt) = (w, batch.val, within.val) := by
  cases w <;> revert batch within <;> decide +kernel

theorem fits (band : Fin 30) :
    let (w, batch, within) := fromBand band
    batch < (Layout.ofWidth w).batches ∧ within < bandsPerSlot w := by
  decide +kernel +revert

theorem ordinal_right_inverse (band : Fin 30) :
    let (w,batch,within) := fromBand band
    gridBand w + batch * bandsPerSlot w + within = band.val := by
  decide +kernel +revert


open VeriTile.Triton.FP QKVAmax QKVNative QKVPack
open QKVTTIR (Bank)

theorem slot_range (w : Width) : slotBase w + (Layout.ofWidth w).batches ≤ 7 := by
  cases w <;> decide

structure Slot where
  width : Width
  batch : Fin (Layout.ofWidth width).batches
  deriving DecidableEq

def encodeSlot (w : Width) (batch : Fin (Layout.ofWidth w).batches) : Fin 7 :=
  ⟨slotBase w + batch.val, Nat.lt_of_lt_of_le (Nat.add_lt_add_left batch.isLt _) (slot_range w)⟩

def decodeSlot (slot : Fin 7) : Slot :=
  if h : slot.val < 3 then ⟨.narrow, ⟨slot.val, h⟩⟩
  else if h' : slot.val < 5 then ⟨.halfValue, ⟨slot.val - 3, by change _ < 2; omega⟩⟩
  else ⟨.wide, ⟨slot.val - 5, by have := slot.isLt; change _ < 2; omega⟩⟩

theorem decode_slot (w : Width) (batch : Fin (Layout.ofWidth w).batches) :
    decodeSlot (encodeSlot w batch) = ⟨w, batch⟩ := by
  cases w <;> revert batch <;> decide +kernel

theorem encode_slot (slot : Fin 7) :
    let selected := decodeSlot slot
    encodeSlot selected.width selected.batch = slot := by
  decide +kernel +revert

/-- These are the scalar counts and branch flags in the shared scan's
two loop headers, prior to the usual signed trip-count calculation. -/
theorem scan_dispatch (slot : Fin 7) :
    let w := (decodeSlot slot).width
    let narrow := slot.val < 3
    let wide := 5 ≤ slot.val
    (if wide then 576 else 288) = (Layout.ofWidth w).firstTiles ∧
    (if narrow ∨ wide then 288 else 144) = (Layout.ofWidth w).secondTiles ∧
    decide (¬wide) = cropped w true ∧
    decide (¬(narrow ∨ wide)) = cropped w false := by
  decide +kernel +revert

theorem pack_dispatch (band : Fin 30) :
    let w := (fromBand band).1
    let narrow := band.val < 12
    let wide := 18 ≤ band.val
    (if narrow then 1536 else if wide then 2304 else 1152) = (Geometry.ofWidth w).rows ∧
    (if wide then 1536 else 768) = (Geometry.ofWidth w).firstRows ∧
    decide (¬wide) = cropped w true ∧
    decide (¬(narrow ∨ wide)) = cropped w false ∧ 0 < slotDepth (selectedBands band) := by
  decide +kernel +revert

theorem band_slot (band : Fin 30) :
    slotDepth (selectedBands band) - 1 = slotBase (fromBand band).1 + (fromBand band).2.1 := by
  decide +kernel +revert

structure PackTile where
  width : Width
  batch : Fin (Layout.ofWidth width).batches
  row : Fin ((Geometry.ofWidth width).firstBlocks + (Geometry.ofWidth width).secondBlocks)
  deriving DecidableEq

def PackTile.point (t : PackTile) (col : Fin 6) (rowLane : Fin 64) (colLane : Fin 128) :
    Point (Geometry.ofWidth t.width) :=
  ⟨t.batch, t.row, col, rowLane, colLane⟩

def decodeTile (band : Fin 30) (laneTile : Fin 6) : PackTile :=
  let selected := fromBand band
  have h := fits band
  ⟨selected.1, ⟨selected.2.1, h.1⟩, ⟨selected.2.2 * 6 + laneTile.val, by
    have hw := census selected.1
    have hi := laneTile.isLt
    have ht : selected.2.2 < bandsPerSlot selected.1 := h.2
    omega⟩⟩

def encodeTile (tile : PackTile) : Fin 30 × Fin 6 :=
  (toBand tile.width tile.batch.val (tile.row.val / 6) tile.batch.isLt (by
    have hrow := tile.row.isLt
    have ht : tile.row.val < bandsPerSlot tile.width * 6 := (census tile.width).symm ▸ hrow
    omega), ⟨tile.row.val % 6, Nat.mod_lt _ (by decide)⟩)

theorem decode_tile (tile : PackTile) :
    decodeTile (encodeTile tile).1 (encodeTile tile).2 = tile := by
  obtain ⟨w, b, r⟩ := tile
  cases w <;> revert b r <;> decide +kernel

theorem encode_tile (band : Fin 30) (laneTile : Fin 6) :
    encodeTile (decodeTile band laneTile) = (band, laneTile) := by
  decide +kernel +revert

/-- The GPU reads the whole raw bank with a global layer ordinal.
The scratch store uses the local layer index and the chosen width buffer. -/
def scanStore (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (slot : Fin 7) (part : Part) : Width × (Nat × Scalar) :=
  let selected := decodeSlot slot
  (selected.width, scanFrom (slot.val - selected.batch.val) first second reduce
    selected.width selected.batch part)

def perWidthScan (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (w : Width) (b : Fin (Layout.ofWidth w).batches) (part : Part) : Nat × Scalar :=
  (scanStore first second reduce (encodeSlot w b) part).2

theorem scan_store_equiv (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (w : Width) (b : Fin (Layout.ofWidth w).batches) (part : Part) :
    scanStore first second reduce (encodeSlot w b) part =
      (w, QKVTTIR.referenceStore (compactBank w true (bankView (slotBase w) first))
        (compactBank w false (bankView (slotBase w) second)) reduce w b part) := by
  have hslot := (encodeSlot w b).isLt
  have hscan := scan_from_equiv (slotBase w) first second reduce w b part hslot
  let selectedStore (s : Slot) := (s.width,
    scanFrom ((encodeSlot w b).val - s.batch.val) first second reduce s.width s.batch part)
  change selectedStore (decodeSlot (encodeSlot w b)) = _
  rw [decode_slot]
  simpa [selectedStore, encodeSlot] using congrArg (fun x => (w, x)) hscan

def scaleStore (scratch : Width → Nat → Scalar) (slot : Fin 7) : Width × (Nat × Scalar) :=
  let selected := decodeSlot slot
  (selected.width, QKVScale.fusedStore (scratch selected.width) selected.width selected.batch)

def perWidthScale (scratch : Nat → Scalar) (w : Width) (b : Fin (Layout.ofWidth w).batches) :
    Nat × Scalar :=
  (scaleStore (fun _ => scratch) (encodeSlot w b)).2

theorem scale_store_equiv (scratch : Width → Nat → Scalar) (w : Width)
    (b : Fin (Layout.ofWidth w).batches) :
    scaleStore scratch (encodeSlot w b) = (w, QKVScale.referenceStore (scratch w) w b) := by
  let selectedStore (s : Slot) := (s.width, QKVScale.fusedStore (scratch s.width) s.width s.batch)
  change selectedStore (decodeSlot (encodeSlot w b)) = _
  rw [decode_slot]
  exact congrArg (fun x => (w, x)) (QKVScale.store_equal (scratch w) w b)

def packStoreWith (store : Nat → Arithmetic → Inputs → (w : Width) →
    Point (Geometry.ofWidth w) → Option Writes) (op : Arithmetic) (first second : Bank) (scales : Width → Nat → Scalar)
    (band : Fin 30) (laneTile col : Fin 6) (rowLane : Fin 64) (colLane : Fin 128) :
    Width × Option Writes :=
  let tile := decodeTile band laneTile
  let origin := slotDepth (selectedBands band) - 1 - tile.batch.val
  let raw : Inputs := ⟨first, second, scales tile.width⟩
  (tile.width, store origin op raw tile.width (tile.point col rowLane colLane))

def packStore := packStoreWith (fun origin op raw w p => observeGluon (packFrom origin op raw w) p)

def vectorPackStore := packStoreWith Vector.observe

def perWidthPack (op : Arithmetic) (raw : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) : Option Writes :=
  let (band, laneTile) := encodeTile ⟨w, p.batch, p.rowTile⟩
  (packStore op raw.first raw.second (fun _ => raw.scales) band laneTile p.colTile p.rowLane p.colLane).2

private theorem pack_store_with_equiv
    (store : Nat → Arithmetic → Inputs → (w : Width) → Point (Geometry.ofWidth w) → Option Writes)
    (correct : ∀ origin op raw w p, origin + (Layout.ofWidth w).batches ≤ 7 →
      store origin op raw w p = reference op (compactInputs w (inputsView origin raw)) p)
    (op : Arithmetic) (first second : Bank) (scales : Width → Nat → Scalar)
    (w : Width) (p : Point (Geometry.ofWidth w)) :
    let (band, laneTile) := encodeTile ⟨w, p.batch, p.rowTile⟩
    packStoreWith store op first second scales band laneTile p.colTile p.rowLane p.colLane =
      (w, reference op (compactInputs w (inputsView (slotBase w) ⟨first, second, scales w⟩)) p) := by
  let t : PackTile := ⟨w, p.batch, p.rowTile⟩
  let band := (encodeTile t).1
  let laneTile := (encodeTile t).2
  change packStoreWith store op first second scales band laneTile p.colTile p.rowLane p.colLane = _
  have htile : decodeTile band laneTile = t := decode_tile t
  have hw : (fromBand band).1 = w := congrArg PackTile.width htile
  have hb : (fromBand band).2.1 = p.batch.val := by
    simpa [decodeTile, t] using congrArg (fun tile : PackTile => tile.batch.val) htile
  have hbase : slotDepth (selectedBands band) - 1 - p.batch.val = slotBase w := by
    rw [band_slot, hw, hb, Nat.add_sub_cancel]
  let selectedStore (tile : PackTile) :=
    let origin := slotDepth (selectedBands band) - 1 - tile.batch.val
    let raw : Inputs := ⟨first, second, scales tile.width⟩
    (tile.width, store origin op raw tile.width (tile.point p.colTile p.rowLane p.colLane))
  change selectedStore (decodeTile band laneTile) = _
  rw [htile]
  change (w, store (slotDepth (selectedBands band) - 1 - p.batch.val) op ⟨first, second, scales w⟩ w p) = _
  rw [hbase, correct _ _ _ _ _ (slot_range w)]

theorem pack_store_equiv (op : Arithmetic) (first second : Bank) (scales : Width → Nat → Scalar)
    (w : Width) (p : Point (Geometry.ofWidth w)) :
    let (band, laneTile) := encodeTile ⟨w, p.batch, p.rowTile⟩
    packStore op first second scales band laneTile p.colTile p.rowLane p.colLane =
      (w, reference op (compactInputs w (inputsView (slotBase w) ⟨first, second, scales w⟩)) p) :=
  pack_store_with_equiv _ pack_from_equiv op first second scales w p

theorem vector_pack_store_equiv (op : Arithmetic) (first second : Bank) (scales : Width → Nat → Scalar)
    (w : Width) (p : Point (Geometry.ofWidth w)) :
    let (band, laneTile) := encodeTile ⟨w, p.batch, p.rowTile⟩
    vectorPackStore op first second scales band laneTile p.colTile p.rowLane p.colLane =
      (w, reference op (compactInputs w (inputsView (slotBase w) ⟨first, second, scales w⟩)) p) :=
  pack_store_with_equiv _ Vector.observe_eq op first second scales w p

#axiomsClean vector_pack_store_equiv

#axiomsClean from_band_matches
#axiomsClean decode_slot
#axiomsClean encode_slot
#axiomsClean scan_dispatch
#axiomsClean pack_dispatch
#axiomsClean decode_tile
#axiomsClean encode_tile
#axiomsClean scan_store_equiv
#axiomsClean scale_store_equiv
#axiomsClean pack_store_equiv

end VeriTile.Bench.Models.NanoGPT.Track1.Grouped
