import models.NanoGPT.Track1.MLPWeightScale
import models.NanoGPT.Track1.QKVVectorLayout

/-! The Hopper grid reserves a 64 × 128 tile for two original 64 × 64 CTAs.
Each original tile keeps its own amax store. The row and transpose byte
planes reuse the verified warp-local layout exchange. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MLPWeights
open VeriTile.Triton.FP QKVPack QKVLaunch
open QKVTTIR (Bank)

set_option maxRecDepth 2048

abbrev H := 2816
abbrev D := 768
abbrev Elements := H * D

def geometry : Geometry := ⟨12, 44, 0, 6, Elements, Elements⟩
abbrev WeightPoint := Point geometry
abbrev RowTile := Fin 44
abbrev ColTile := Fin 12
abbrev ColPair := Fin 6
abbrev Half := Fin 2

def joinTile (pair : ColPair) (half : Half) : ColTile :=
  ⟨pair.val * 2 + half.val, by have hp := pair.isLt; have hh := half.isLt; omega⟩

def splitTile (c : ColTile) : ColPair × Half :=
  (⟨c.val / 2, by have := c.isLt; omega⟩, ⟨c.val % 2, Nat.mod_lt _ (by decide)⟩)

theorem join_split_tile (c : ColTile) : joinTile (splitTile c).1 (splitTile c).2 = c := by
  exact Fin.ext (by have := c.isLt; simp only [joinTile, splitTile]; omega)

theorem split_join_tile (pair : ColPair) (half : Half) :
    splitTile (joinTile pair half) = (pair, half) := by
  have hh := half.isLt
  apply Prod.ext <;> apply Fin.ext <;> simp only [splitTile, joinTile] <;> omega

def joinLane (half : Half) (lane : Fin 64) : Fin 128 :=
  ⟨half.val * 64 + lane.val, by have hh := half.isLt; have hl := lane.isLt; omega⟩

def originalColumn (p : WeightPoint) : Nat :=
  p.colTile.val * 128 + (p.colLane.val / 64) * 64 + p.colLane.val % 64

theorem original_column (p : WeightPoint) : originalColumn p = p.col := by
  simp only [originalColumn, Point.col, Point.startCol, ColsPerTile]
  omega

def weightPoint (m : Matrix) (r : RowTile) (c : ColPair) (h : Fin 64) (d : Fin 128) :
    WeightPoint := ⟨m.layer, r, c, h, d⟩

def inputWord (id : Fin 24) (p : WeightPoint) : BitVec 32 :=
  let tile := word p.startRow * word D + word p.startCol
  let lane := word p.rowLane.val * word D + word p.colLane.val
  word id.val * word Elements + tile + lane

def inputAddress (id : Fin 24) (r c : Nat) : Nat :=
  id.val * Elements + r * D + c

theorem input_word (id : Fin 24) (p : WeightPoint) :
    (inputWord id p).toNat = inputAddress id p.row p.col := by
  obtain ⟨hr, hc⟩ := full_tile p
  have hi := id.isLt
  change p.row < H at hr
  change p.col < D at hc
  have haddr : inputAddress id p.row p.col < 2^32 := by
    change id.val * 2162688 + p.row * 768 + p.col < 4294967296
    change p.row < 2816 at hr
    change p.col < 768 at hc
    omega
  simp only [inputWord, inputAddress, Point.row, Point.col, Nat.add_mul, word,
    ← BitVec.ofNat_mul, ← BitVec.ofNat_add, BitVec.toNat_ofNat]
  have he : id.val * Elements + (p.startRow * D + p.startCol) +
      (p.rowLane.val * D + p.colLane.val) =
      id.val * Elements + (p.startRow * D + p.rowLane.val * D) + (p.startCol + p.colLane.val) := by
    ac_rfl
  rw [he]
  exact Nat.mod_eq_of_lt (by simpa only [inputAddress, Point.row, Point.col, Nat.add_mul] using haddr)

theorem output_words (p : WeightPoint) :
    (rowWord p).toNat = rowAddress p ∧ (transposeWord p).toNat = transposedAddress p := by
  obtain ⟨hr, hc⟩ := full_tile p
  have hb := p.batch.isLt
  change p.row < 2816 at hr
  change p.col < 768 at hc
  change p.batch.val < 12 at hb
  have hrow : rowAddress p < 2^32 := by
    change p.batch.val * 2162688 + p.row * 768 + p.col < 4294967296
    omega
  have hcol : transposedAddress p < 2^32 := by
    change p.batch.val * 2162688 + p.col * 2816 + p.row < 4294967296
    omega
  constructor
  · simpa only [rowWord, word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, BitVec.toNat_ofNat, rowAddress]
      using Nat.mod_eq_of_lt hrow
  · simpa only [transposeWord, word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, BitVec.toNat_ofNat,
      transposedAddress] using Nat.mod_eq_of_lt hcol

def referenceBytes (arith : Arithmetic) (bank : Bank) (scales : WeightScales)
    (down : Bool) (p : WeightPoint) : Writes :=
  let m : Matrix := ⟨p.batch, down⟩
  let x := bank (inputAddress (encodeMatrix m) p.row (originalColumn p))
  ⟨rowAddress p, transposedAddress p, quantize arith x (scales m)⟩

def pairedBytes (arith : Arithmetic) (bank : Bank) (scales : WeightScales)
    (down : Bool) (p : WeightPoint) : Writes :=
  let id := encodeMatrix ⟨p.batch, down⟩
  let x := bank (inputWord id p).toNat
  ⟨(rowWord p).toNat, (transposeWord p).toNat, quantize arith x (scales (decodeMatrix id))⟩

theorem paired_bytes_equal (arith : Arithmetic) (bank : Bank) (scales : WeightScales)
    (down : Bool) (p : WeightPoint) :
    pairedBytes arith bank scales down p = referenceBytes arith bank scales down p := by
  obtain ⟨hr, hc⟩ := output_words p
  simp only [pairedBytes, referenceBytes, original_column, input_word, hr, hc, decode_encode_matrix]

def observeVector (arith : Arithmetic) (bank : Bank) (scales : WeightScales)
    (down : Bool) (p : WeightPoint) : Option Writes :=
  let i := (p.rowLane, p.colLane)
  let quantized := fun t => pairedBytes arith bank scales down (atLane p (Vector.rowIndex t))
  let row := quantized (Vector.rowOwner i)
  let consumer := Vector.transposedOwner i
  let dest := atLane p (Vector.transposedIndex consumer)
  let afterMove := quantized (Vector.rowOwner (Vector.decode (Vector.loadLocation consumer)))
  if row.byte == afterMove.byte then some ⟨row.rowOffset, (transposeWord dest).toNat, row.byte⟩ else none

theorem cache_bytes_equal (arith : Arithmetic) (bank : Bank) (scales : WeightScales)
    (down : Bool) (p : WeightPoint) :
    observeVector arith bank scales down p = some (referenceBytes arith bank scales down p) := by
  simp only [observeVector, ← Vector.shared_match (p.rowLane, p.colLane), Vector.store_decodes,
    Vector.row_covers, Vector.transposed_covers, atLane]
  change (if (pairedBytes arith bank scales down p).byte ==
      (pairedBytes arith bank scales down p).byte then
    some (Writes.mk (pairedBytes arith bank scales down p).rowOffset (transposeWord p).toNat
      (pairedBytes arith bank scales down p).byte) else none) = _
  simp only [beq_self_eq_true, if_true, paired_bytes_equal, (output_words p).2]
  rfl

/-- Nonnegative magnitudes are ordered by their FP32 encodings. A NaN
contributes the least key to maxnumf. If every leaf is a NaN, the key
decodes to a canonical NaN. -/
def magnitudeKey (x : Value .bf16) : Nat :=
  let v := (cast .fp32 Config.ieeeValue x).abs
  if v.isNaN then 0 else v.raw + 1

def magnitudeValue (key : Nat) : Scalar :=
  if key = 0 then Value.qNaN .fp32 else Value.ofBits .fp32 (key - 1)

def maxima (read : Fin 64 → Fin 64 → Value .bf16) : Scalar :=
  magnitudeValue ((List.finRange 64).foldl (fun all h =>
    max all ((List.finRange 64).foldl (fun row d => max row (magnitudeKey (read h d))) 0)) 0)

private def foldMax {α : Type} (items : List α) (key : α → Nat) : Nat :=
  items.foldl (fun value item => max value (key item)) 0

private theorem fold_max_le {α : Type} (items : List α) (key : α → Nat) (bound : Nat)
    (h : ∀ item ∈ items, key item ≤ bound) : foldMax items key ≤ bound := by
  unfold foldMax
  have go : ∀ seed ≤ bound, items.foldl (fun value item => max value (key item)) seed ≤ bound := by
    induction items with
    | nil => exact fun _ hs => hs
    | cons item items ih =>
      intro seed hs
      simp only [List.foldl_cons]
      exact ih (fun other hm => h other (List.mem_cons_of_mem _ hm)) _
        (Nat.max_le.mpr ⟨hs, h item (by simp)⟩)
  exact go 0 (Nat.zero_le _)

private theorem le_fold_max {α : Type} (items : List α) (key : α → Nat) (item : α)
    (member : item ∈ items) : key item ≤ foldMax items key := by
  unfold foldMax
  have initial (xs : List α) (seed : Nat) :
      seed ≤ xs.foldl (fun value x => max value (key x)) seed := by
    induction xs generalizing seed with
    | nil => exact Nat.le_refl _
    | cons x xs ih => exact Nat.le_trans (Nat.le_max_left seed (key x)) (ih _)
  have go : ∀ seed : Nat, key item ≤ items.foldl (fun value x => max value (key x)) seed := by
    induction items with
    | nil => simp at member
    | cons x xs ih =>
      intro seed
      rcases List.mem_cons.mp member with rfl | mem
      · exact Nat.le_trans (Nat.le_max_right seed (key item)) (initial xs _)
      · exact ih mem _
  exact go 0

private def shuffleSources : List Nat → Fin 32 → List (Fin 32)
  | [], lane => [lane]
  | distance :: rest, lane =>
      shuffleSources rest lane ++ shuffleSources rest (Fin.ofNat 32 (lane.val ^^^ distance))

def laneHalf (lane : Fin 32) : Half := Fin.ofNat 2 (lane.val / 4)

private theorem warp_shuffle_covers (reader source : Fin 32) :
    source ∈ shuffleSources [8, 16, 1, 2] reader ↔ laneHalf source = laneHalf reader := by
  decide +kernel +revert

private def sourceRegister (row : Fin 2) (col : Fin 16) : Fin 32 :=
  ⟨16 * row.val + col.val, by have := row.isLt; have := col.isLt; omega⟩

private def warpRow (warp : Fin 8) (lane : Fin 32) (row : Fin 2) : Fin 64 :=
  ⟨8 * warp.val + 2 * (lane.val / 8) + row.val, by
    have := warp.isLt; have := lane.isLt; have := row.isLt; omega⟩

private def warpColumn (lane : Fin 32) (col : Fin 16) : Fin 64 :=
  ⟨16 * (lane.val % 4) + col.val, by omega⟩

theorem local_inputs (warp : Fin 8) (lane : Fin 32) (row : Fin 2) (col : Fin 16) :
    Vector.rowIndex ⟨warp, lane, sourceRegister row col⟩ =
      (warpRow warp lane row, joinLane (laneHalf lane) (warpColumn lane col)) := by
  have := lane.isLt
  have := row.isLt
  have := col.isLt
  apply Prod.ext <;> apply Fin.ext <;>
    simp only [Vector.rowIndex, sourceRegister, warpRow, warpColumn, joinLane, laneHalf, Fin.val_ofNat] <;> omega

private def localMagnitude (read : Fin 64 → Fin 64 → Value .bf16)
    (warp : Fin 8) (lane : Fin 32) (row : Fin 2) (col : Fin 16) : Nat :=
  let cell := Vector.rowIndex ⟨warp, lane, sourceRegister row col⟩
  magnitudeKey (read cell.1 (Fin.ofNat 64 cell.2.val))

private theorem local_magnitude (read : Fin 64 → Fin 64 → Value .bf16)
    (warp : Fin 8) (lane : Fin 32) (row : Fin 2) (col : Fin 16) :
    localMagnitude read warp lane row col =
      magnitudeKey (read (warpRow warp lane row) (warpColumn lane col)) := by
  have hc : Fin.ofNat 64 (joinLane (laneHalf lane) (warpColumn lane col)).val =
      warpColumn lane col := by
    apply Fin.ext
    simp only [Fin.val_ofNat, joinLane]
    have := (warpColumn lane col).isLt
    omega
  simp only [localMagnitude, local_inputs, hc]

/-- Lanes reduce 16 local columns and the two local rows, then exchange
column bits 1 and 0 and row bits 4 and 3. Bit 2 stays fixed. -/
def warpMaximum (read : Fin 64 → Fin 64 → Value .bf16) (warp : Fin 8) (lane : Fin 32) : Nat :=
  foldMax (shuffleSources [8, 16, 1, 2] lane) fun src =>
    foldMax (List.finRange 2) fun row =>
      foldMax (List.finRange 16) (localMagnitude read warp src row)

theorem warp_writers_agree (read : Fin 64 → Fin 64 → Value .bf16) (warp : Fin 8) (a b : Fin 32)
    (half : laneHalf a = laneHalf b) : warpMaximum read warp a = warpMaximum read warp b := by
  let keys := fun src => foldMax (List.finRange 2) fun row =>
    foldMax (List.finRange 16) (localMagnitude read warp src row)
  apply Nat.le_antisymm
  · apply fold_max_le
    intro src member
    exact le_fold_max _ keys src ((warp_shuffle_covers b src).mpr
      (((warp_shuffle_covers a src).mp member).trans half))
  · apply fold_max_le
    intro src member
    exact le_fold_max _ keys src ((warp_shuffle_covers a src).mpr
      (((warp_shuffle_covers b src).mp member).trans half.symm))

def partialStoreAddress (warp : Fin 8) (half : Half) : Nat := 4096 + 8 * warp.val + 4 * half.val

def compiledPartialStore (warp : Fin 8) (lane : Fin 32) : Nat :=
  4096 + (((32 * warp.val + lane.val) >>> 2) &&& 56) + (lane.val &&& 4)

theorem compiled_partial_store (warp : Fin 8) (lane : Fin 32) :
    compiledPartialStore warp lane = partialStoreAddress warp (laneHalf lane) := by
  decide +kernel +revert

private def partialWriter (half : Half) : Fin 32 := Fin.ofNat 32 (4 * half.val)

private theorem writer_half (half : Half) : laneHalf (partialWriter half) = half := by
  decide +kernel +revert

private def loadWarp (lane : Fin 32) : Fin 8 := Fin.ofNat 8 lane.val
private def loadHalf (lane : Fin 32) : Half := Fin.ofNat 2 (lane.val / 8)

def partialLoadAddress (lane : Fin 32) : Nat :=
  4096 + ((lane.val <<< 3) &&& 56) + ((lane.val &&& 8) >>> 1)

theorem partial_addresses (lane : Fin 32) :
    partialLoadAddress lane = partialStoreAddress (loadWarp lane) (loadHalf lane) ∧
    4096 ≤ partialLoadAddress lane ∧ partialLoadAddress lane + 3 < 4160 := by
  decide +kernel +revert

theorem partial_slots_unique (w₁ w₂ : Fin 8) (h₁ h₂ : Half)
    (addr : partialStoreAddress w₁ h₁ = partialStoreAddress w₂ h₂) : w₁ = w₂ ∧ h₁ = h₂ := by
  have := h₁.isLt
  have := h₂.isLt
  unfold partialStoreAddress at addr
  constructor <;> apply Fin.ext <;> omega

def partialPhases : List (QKVPack.GluonShared.Event 1) := [.write 0, .barrier, .read 0]

theorem partials_published : QKVPack.GluonShared.checkedPhases partialPhases = some [0] := rfl

private def outputLane (half : Half) : Fin 32 := Fin.ofNat 32 (8 * half.val)

theorem output_lanes (warp : Fin 8) (lane : Fin 32) :
    ((32 * warp.val + lane.val) &&& 247 = 0) ↔ warp = 0 ∧ lane = outputLane (loadHalf lane) := by
  decide +kernel +revert

private theorem block_shuffle_half (half : Half) (source : Fin 32)
    (h : source ∈ shuffleSources [1, 2, 4] (outputLane half)) : loadHalf source = half := by
  decide +kernel +revert

private theorem block_shuffle_covers (half : Half) (warp : Fin 8) :
    let source : Fin 32 := Fin.ofNat 32 (8 * half.val + warp.val)
    source ∈ shuffleSources [1, 2, 4] (outputLane half) ∧ loadWarp source = warp := by
  decide +kernel +revert

private theorem cell_has_producer (half : Half) (row col : Fin 64) :
    let warp : Fin 8 := Fin.ofNat 8 (row.val / 8)
    let lane : Fin 32 := Fin.ofNat 32 (8 * (row.val % 8 / 2) + 4 * half.val + col.val / 16)
    let rowReg : Fin 2 := Fin.ofNat 2 row.val
    let colReg : Fin 16 := Fin.ofNat 16 col.val
    laneHalf lane = half ∧ warpRow warp lane rowReg = row ∧ warpColumn lane colReg = col := by
  decide +kernel +revert

/-- The last three shuffles combine the eight shared words for the same
column half. Every warp may execute the loads, but only lanes 0 and 8
of warp zero publish the two final values. -/
def blockMaximum (read : Half → Fin 64 → Fin 64 → Value .bf16) (half : Half) : Scalar :=
  magnitudeValue (foldMax (shuffleSources [1, 2, 4] (outputLane half)) fun lane =>
    warpMaximum (read (loadHalf lane)) (loadWarp lane) (partialWriter (loadHalf lane)))

theorem block_maximum_exact (read : Half → Fin 64 → Fin 64 → Value .bf16) (half : Half) :
    blockMaximum read half = maxima (read half) := by
  unfold blockMaximum maxima
  apply congrArg magnitudeValue
  apply Nat.le_antisymm
  · apply fold_max_le
    intro consumer member
    rw [block_shuffle_half half consumer member]
    apply fold_max_le
    intro lane _
    apply fold_max_le
    intro rowReg _
    apply fold_max_le
    intro colReg _
    rw [local_magnitude]
    let row := warpRow (loadWarp consumer) lane rowReg
    exact Nat.le_trans
      (le_fold_max (List.finRange 64) (fun col => magnitudeKey (read half row col)) (warpColumn lane colReg) (by simp))
      (le_fold_max (List.finRange 64) (fun h =>
        foldMax (List.finRange 64) fun d => magnitudeKey (read half h d)) row (by simp))
  · apply fold_max_le
    intro row _
    apply fold_max_le
    intro col _
    let warp : Fin 8 := Fin.ofNat 8 (row.val / 8)
    let lane : Fin 32 := Fin.ofNat 32 (8 * (row.val % 8 / 2) + 4 * half.val + col.val / 16)
    let rowReg : Fin 2 := Fin.ofNat 2 row.val
    let colReg : Fin 16 := Fin.ofNat 16 col.val
    let consumer : Fin 32 := Fin.ofNat 32 (8 * half.val + warp.val)
    obtain ⟨hhalf, hrow, hcol⟩ := cell_has_producer half row col
    change laneHalf lane = half at hhalf
    change warpRow warp lane rowReg = row at hrow
    change warpColumn lane colReg = col at hcol
    obtain ⟨member, hwarp⟩ := block_shuffle_covers half warp
    change consumer ∈ _ at member
    change loadWarp consumer = warp at hwarp
    let colKey := localMagnitude (read half) warp
    let rowKey := fun src r => foldMax (List.finRange 16) (colKey src r)
    let laneKey := fun src => foldMax (List.finRange 2) (rowKey src)
    have word := le_fold_max (List.finRange 16) (colKey lane rowReg) colReg (by simp)
    change localMagnitude (read half) warp lane rowReg colReg ≤ _ at word
    rw [local_magnitude, hrow, hcol] at word
    have laneMember := (warp_shuffle_covers (partialWriter half) lane).mpr (hhalf.trans (writer_half half).symm)
    have warpBound := le_fold_max _ (fun lane =>
      warpMaximum (read (loadHalf lane)) (loadWarp lane) (partialWriter (loadHalf lane))) consumer member
    change warpMaximum (read (loadHalf consumer)) (loadWarp consumer) (partialWriter (loadHalf consumer)) ≤ _ at warpBound
    rw [hwarp, block_shuffle_half half consumer member] at warpBound
    change foldMax (shuffleSources [8, 16, 1, 2] (partialWriter half)) laneKey ≤ _ at warpBound
    exact Nat.le_trans word (Nat.le_trans (le_fold_max _ (rowKey lane) rowReg (by simp))
      (Nat.le_trans (le_fold_max _ laneKey lane laneMember) warpBound))

def originalPartial (bank : Bank) (m : Matrix) (r : RowTile) (c : ColTile) : Bool × (Nat × Scalar) :=
  let offset := m.layer.val * 528 + r.val * 12 + c.val
  let value := maxima fun h d =>
    bank (inputAddress (encodeMatrix m) (r.val * 64 + h.val) (c.val * 64 + d.val))
  (m.down, offset, value)

def pairedPartial (bank : Bank) (id : Fin 24) (r : RowTile) (pair : ColPair) (half : Half) :
    Bool × (Nat × Scalar) :=
  let m := decodeMatrix id
  let addr := word m.layer.val * word 528 + word r.val * word 12 + word pair.val * word 2 + word half.val
  let value := maxima fun h d =>
    bank (inputWord id (weightPoint m r pair h (joinLane half d))).toNat
  (m.down, addr.toNat, value)

theorem partial_store_equal (bank : Bank) (m : Matrix) (r : RowTile) (pair : ColPair) (half : Half) :
    pairedPartial bank (encodeMatrix m) r pair half = originalPartial bank m r (joinTile pair half) := by
  have hb := m.layer.isLt
  have hr := r.isLt
  have hc := pair.isLt
  have hh := half.isLt
  have hlim : m.layer.val * 528 + r.val * 12 + pair.val * 2 + half.val < 2^32 := by omega
  have haddr : (word m.layer.val * word 528 + word r.val * word 12 +
      word pair.val * word 2 + word half.val).toNat =
      m.layer.val * 528 + r.val * 12 + (joinTile pair half).val := by
    simp only [word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt hlim]
    simp only [joinTile, Nat.add_assoc]
  have hread :
      (fun h d => bank (inputWord (encodeMatrix m) (weightPoint m r pair h (joinLane half d))).toNat) =
      (fun h d => bank (inputAddress (encodeMatrix m) (r.val * 64 + h.val)
        ((joinTile pair half).val * 64 + d.val))) := by
    funext h d
    rw [input_word]
    simp only [inputAddress, weightPoint, Point.row, Point.col, Point.startRow, Point.startCol,
      RowsPerTile, ColsPerTile, joinLane, joinTile]
    congr 1
    omega
  simp only [pairedPartial, originalPartial, decode_encode_matrix, haddr, hread]

def warpPartial (bank : Bank) (id : Fin 24) (r : RowTile) (pair : ColPair) (half : Half) :
    Bool × (Nat × Scalar) :=
  let prior := pairedPartial bank id r pair half
  let value := blockMaximum (fun part h d =>
    bank (inputWord id (weightPoint (decodeMatrix id) r pair h (joinLane part d))).toNat) half
  (prior.1, (prior.2.1, value))

theorem warp_partial_eq (bank : Bank) (id : Fin 24) (r : RowTile) (pair : ColPair) (half : Half) :
    warpPartial bank id r pair half = pairedPartial bank id r pair half := by
  unfold warpPartial
  rw [block_maximum_exact]
  rfl

structure Inputs where
  bank : Bank
  scales : WeightScales
  scratch : Scratch

structure WeightRefresh where
  scales : WeightScales
  partials : Matrix → RowTile → ColTile → Bool × (Nat × Scalar)
  bytes : Bool → WeightPoint → Option Writes

def readReference (arith : Arithmetic) (delayed : Bool) (input : Inputs) : WeightRefresh :=
  let scales := referenceScales delayed input.scales input.scratch
  ⟨scales, originalPartial input.bank, fun down p => some (referenceBytes arith input.bank scales down p)⟩

def readPaired (arith : Arithmetic) (delayed : Bool) (input : Inputs) : WeightRefresh :=
  let scales := emittedScales delayed input.scales input.scratch
  ⟨scales, fun m r c => pairedPartial input.bank (encodeMatrix m) r (splitTile c).1 (splitTile c).2,
    fun down p => observeVector arith input.bank scales down p⟩

theorem refresh_equal (arith : Arithmetic) (delayed : Bool) (input : Inputs) :
    readPaired arith delayed input = readReference arith delayed input := by
  simp only [readPaired, readReference, scales_equal, WeightRefresh.mk.injEq, true_and]
  constructor
  · funext m r c
    exact (partial_store_equal input.bank m r _ _).trans
      (congrArg (originalPartial input.bank m r) (join_split_tile c))
  · funext down p
    exact cache_bytes_equal arith input.bank _ down p

/-- Keep the original scale and both byte planes, using the independent
shared reduction for the partial maxima consumed on the next refresh. -/
def readWarp (arith : Arithmetic) (delayed : Bool) (input : Inputs) : WeightRefresh :=
  { readPaired arith delayed input with
    partials := fun m r c => warpPartial input.bank (encodeMatrix m) r (splitTile c).1 (splitTile c).2 }

theorem warp_refresh_equal (arith : Arithmetic) (delayed : Bool) (input : Inputs) :
    readWarp arith delayed input = readReference arith delayed input := by
  have partials : (fun m r c => warpPartial input.bank (encodeMatrix m) r (splitTile c).1 (splitTile c).2) =
      (readPaired arith delayed input).partials := by
    funext m r c
    exact warp_partial_eq _ _ _ _ _
  unfold readWarp
  rw [partials]
  exact refresh_equal arith delayed input

#axiomsClean local_inputs
#axiomsClean block_maximum_exact
#axiomsClean warp_writers_agree
#axiomsClean partial_addresses
#axiomsClean compiled_partial_store
#axiomsClean output_lanes
#axiomsClean partial_slots_unique
#axiomsClean partials_published
#axiomsClean warp_refresh_equal
#axiomsClean join_split_tile
#axiomsClean split_join_tile
#axiomsClean input_word
#axiomsClean output_words
#axiomsClean cache_bytes_equal
#axiomsClean partial_store_equal
#axiomsClean refresh_equal
end VeriTile.Bench.Models.NanoGPT.Track1.MLPWeights
