import models.NanoGPT.Track1.QKVGrouped

/-! The grouped amax visits the same BF16 words in groups of three tiles.
Within a cropped head, advancing by three logical visits adds 96 tiles
and crosses four 24-tile heads. The packed physical address therefore
advances by 192 tiles. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Grouped.ScanSchedule
open QKVLaunch QKVAmax QKVNative
open QKVTTIR (Bank)

def qkCount (slot : Fin 7) : Nat := if 5 ≤ slot.val then 18 else 9

def vCount (slot : Fin 7) (part : Part) : Nat :=
  if slot.val < 3 ∨ 5 ≤ slot.val then 9 else 4 + if part.val < 16 then 1 else 0

def physicalTile (cropped : Bool) (part : Part) (block phase : Nat) : Nat :=
  let logical := word part.val + word 32 * word phase
  let initial := logical + if cropped then logical.sdiv (word 24) * word 24 else 0
  (word block * word (if cropped then 192 else 96) + initial).toNat

def visits (cropped : Bool) (part : Part) (count : Nat) (tail : Bool) : List Nat :=
  let grouped := (List.range (count / 3)).flatMap fun block =>
    (List.range 3).map (physicalTile cropped part block)
  grouped ++ if tail then
    (List.range (count % 3)).map (physicalTile cropped part (count / 3))
  else []

def qkVisits (slot : Fin 7) (part : Part) : List Nat :=
  visits (decide (slot.val < 5)) part (qkCount slot) false

def vVisits (slot : Fin 7) (part : Part) : List Nat :=
  visits (decide (3 ≤ slot.val ∧ slot.val < 5)) part (vCount slot part) true

theorem slot_and_visits (slot : Fin 7) (part : Part) :
    let decoded := decodeSlot slot
    let w := decoded.width
    let layout := Layout.ofWidth w
    let first := visitsI32 layout.firstTiles part
    let total := visitsI32 (layout.firstTiles + layout.secondTiles) part
    slot.val - decoded.batch.val + decoded.batch.val = slot.val ∧
      qkVisits slot part = (List.range first).map
        (fun i => nativeTile (cropped w true) (QKVTTIR.firstTile part i)) ∧
      vVisits slot part = (List.range' first (total - first)).map
        (fun i => nativeTile (cropped w false) (QKVTTIR.secondTile layout part i)) := by
  decide +kernel +revert

def stream (first second : Bank) (slot : Fin 7) (part : Part) (lane : Lane) :
    List QKVTTIR.Scalar :=
  (qkVisits slot part).map (fun tile => QKVTTIR.unmaskedSample first slot.val BankPitch tile lane) ++
  (vVisits slot part).map (fun tile => QKVTTIR.unmaskedSample second slot.val BankPitch tile lane)

theorem stream_eq (first second : Bank) (slot : Fin 7) (part : Part) (lane : Lane) :
    let selected := decodeSlot slot
    let batch := slot.val - selected.batch.val + selected.batch.val
    stream first second slot part lane =
      QKVTTIR.readStream (nativeSample selected.width true first batch)
        (nativeSample selected.width false second batch) selected.width part lane := by
  obtain ⟨batch, qk, v⟩ := slot_and_visits slot part
  simp only [stream, QKVTTIR.readStream, qk, v, batch, List.map_map,
    Function.comp_def, nativeSample]

/-- Reuse the original ordered lane folds and final CTA reduction. Only
the sequence generator and split physical address calculation change. -/
def store (first second : Bank) (reduce : (Lane → QKVTTIR.Scalar) → QKVTTIR.Scalar)
    (slot : Fin 7) (part : Part) : Width × (Nat × QKVTTIR.Scalar) :=
  let selected := decodeSlot slot
  let op := QKVTTIR.ops reduce
  (selected.width, (QKVTTIR.storeWord selected.batch.val part,
    reduce (fun lane => (stream first second slot part lane).foldl op.maximum op.zero)))

theorem store_eq (first second : Bank) (reduce : (Lane → QKVTTIR.Scalar) → QKVTTIR.Scalar)
    (slot : Fin 7) (part : Part) :
    store first second reduce slot part = scanStore first second reduce slot part := by
  simp only [store, scanStore, scanFrom, stream_eq]

def perWidthScan (first second : Bank) (reduce : (Lane → QKVTTIR.Scalar) → QKVTTIR.Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) (part : Part) :
    Nat × QKVTTIR.Scalar :=
  (store first second reduce (encodeSlot w batch) part).2

theorem store_equiv (first second : Bank) (reduce : (Lane → QKVTTIR.Scalar) → QKVTTIR.Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) (part : Part) :
    store first second reduce (encodeSlot w batch) part =
      (w, QKVTTIR.referenceStore (compactBank w true (bankView (slotBase w) first))
        (compactBank w false (bankView (slotBase w) second)) reduce w batch part) :=
  (store_eq first second reduce _ part).trans (scan_store_equiv first second reduce w batch part)

#axiomsClean slot_and_visits
#axiomsClean store_equiv
end VeriTile.Bench.Models.NanoGPT.Track1.Grouped.ScanSchedule
