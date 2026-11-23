import models.NanoGPT.Track1.QKVTTIR
import models.NanoGPT.Track1.QKVGluonLayout

/-! Both refreshes start from the same sliced QK and V banks. The reference
flattens the cropped heads. The Hopper path computes the corresponding
physical addresses directly and preserves the order of amax and FP8 operations. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVNative
open VeriTile.Triton.FP QKVAmax QKVLaunch
open QKVTTIR (Bank)

set_option maxRecDepth 2048

abbrev BankPitch := 1179648

def bankView (origin : Nat) (raw : Bank) : Bank := fun address =>
  raw (origin * BankPitch + address)

def inputsView (origin : Nat) (raw : QKVPack.Inputs) : QKVPack.Inputs :=
  { raw with first := bankView origin raw.first, second := bankView origin raw.second }

@[simp] theorem bank_view_zero (raw : Bank) : bankView 0 raw = raw := by
  funext addr
  simp [bankView]

@[simp] theorem inputs_view_zero (raw : QKVPack.Inputs) : inputsView 0 raw = raw := by
  cases raw
  simp [inputsView]

def cropped (w : Width) (qk : Bool) : Bool :=
  match w with
  | .narrow => qk
  | .halfValue => true
  | .wide => false

def compactPitch (w : Width) (qk : Bool) : Nat :=
  let l := Layout.ofWidth w
  if qk then l.firstStride else l.secondStride

def logicalTiles (w : Width) (qk : Bool) : Nat :=
  let l := Layout.ofWidth w
  if qk then l.firstTiles else l.secondTiles

/-- Flatten head, row, and channel while skipping the omitted rows. -/
def physicalIndex (crop : Bool) (logical : Nat) : Nat :=
  let head := if crop then 49152 else 98304
  logical / head * 98304 + logical % head

theorem physical_in_view (w : Width) (qk : Bool) (logical : Nat)
    (h : logical < logicalTiles w qk * Tile) :
    physicalIndex (cropped w qk) logical < (if qk then BankPitch else 589824) := by
  cases w <;> cases qk <;>
    simp only [physicalIndex, logicalTiles, Layout.ofWidth, cropped, BankPitch, Tile,
      Bool.false_eq_true, if_false, if_true] at * <;> omega

def compactBank (w : Width) (qk : Bool) (raw : Bank) : Bank := fun addr =>
  let pitch := compactPitch w qk
  raw ((addr / pitch) * BankPitch + physicalIndex (cropped w qk) (addr % pitch))

def compactInputs (w : Width) (raw : QKVPack.Inputs) : QKVPack.Inputs :=
  { raw with first := compactBank w true raw.first, second := compactBank w false raw.second }

theorem compact_at (w : Width) (qk : Bool) (raw : Bank) (batch x : Nat)
    (hx : x < compactPitch w qk) :
    compactBank w qk raw (batch * compactPitch w qk + x) =
      raw (batch * BankPitch + physicalIndex (cropped w qk) x) := by
  have hp : 0 < compactPitch w qk := by cases w <;> cases qk <;> decide
  simp only [compactBank]
  rw [Nat.mul_comm batch (compactPitch w qk), Nat.mul_add_div hp, Nat.mul_add_mod, Nat.div_eq_of_lt hx,
    Nat.mod_eq_of_lt hx, Nat.add_zero]

private theorem amax_physical (crop : Bool) (tile : Nat) (lane : Lane) :
    physicalIndex crop (tile * Tile + lane.val) =
      (tile + (if crop then tile / 24 * 24 else 0)) * Tile + lane.val := by
  have hl := lane.isLt
  change lane.val < 2048 at hl
  cases crop <;> simp [physicalIndex, Tile] <;> omega

/-- Signed division in the actual scan TTIR. Its dividend is a valid
nonnegative bank-local tile index, including on the V side. -/
def nativeTile (crop : Bool) (tile : Nat) : Nat :=
  if crop then (word tile + (word tile).sdiv (word 24) * word 24).toNat else tile

private theorem checked_crop :
    ∀ tile : Fin 576, nativeTile true tile.val = tile.val + tile.val / 24 * 24 := by
  decide +kernel

private theorem native_tile (crop : Bool) (tile : Nat) (ht : tile < 576) :
    nativeTile crop tile = tile + (if crop then tile / 24 * 24 else 0) := by
  cases crop
  · simp [nativeTile]
  · exact checked_crop ⟨tile, ht⟩

def nativeSample (w : Width) (qk : Bool) (raw : Bank) (batch tile : Nat) (lane : Lane) :
    Scalar :=
  QKVTTIR.unmaskedSample raw batch BankPitch (nativeTile (cropped w qk) tile) lane

private theorem amax_offset_bound (batch tile skip lane : Nat)
    (hb : batch < 7) (ht : tile < 576) (hs : skip ≤ tile) (hl : lane < 2048) :
    batch * 1179648 + (tile + skip) * 2048 + lane < 2147483648 := by
  omega

private theorem native_sample_eq (w : Width) (qk : Bool) (raw : Bank) (origin batch tile : Nat)
    (lane : Lane) (hb : batch < (Layout.ofWidth w).batches) (hslot : origin + batch < 7)
    (ht : tile < logicalTiles w qk) :
    nativeSample w qk raw (origin + batch) tile lane =
      QKVTTIR.unmaskedSample (compactBank w qk (bankView origin raw)) batch
        (compactPitch w qk) tile lane := by
  have hlim : tile < 576 := Nat.lt_of_lt_of_le ht (by cases w <;> cases qk <;> decide)
  have hl := lane.isLt
  change lane.val < 2048 at hl
  have hlocal : tile * Tile + lane.val < compactPitch w qk := by
    cases w <;> cases qk <;> simp [compactPitch, logicalTiles, Layout.ofWidth, Tile] at * <;> omega
  have hbatch : batch ≤ 2 := by cases w <;> simp [Layout.ofWidth] at hb <;> omega
  have hcompact : elementOffset batch (compactPitch w qk) tile lane < 2^31 := by
    have hstride : compactPitch w qk ≤ 1179648 := by cases w <;> cases qk <;> decide
    have hbatchOffset := Nat.mul_le_mul_left batch hstride
    change batch * compactPitch w qk + tile * 2048 + lane.val < 2147483648
    omega
  have hraw : elementOffset (origin + batch) BankPitch
      (tile + (if cropped w qk then tile / 24 * 24 else 0)) lane < 2^31 := by
    let skip := if cropped w qk then tile / 24 * 24 else 0
    have hskip : skip ≤ tile := by
      unfold skip
      split
      · exact Nat.div_mul_le_self tile 24
      · exact Nat.zero_le tile
    exact amax_offset_bound (origin + batch) tile skip lane.val hslot hlim hskip hl
  simp only [nativeSample, native_tile _ _ hlim, QKVTTIR.unmaskedSample]
  rw [offset_sign_extend _ _ _ _ hraw, offset_sign_extend _ _ _ _ hcompact]
  simp only [Int.toNat_natCast, elementOffset, Nat.add_assoc,
    compact_at w qk (bankView origin raw) batch _ hlocal, amax_physical, bankView, Nat.add_mul]

def scanFrom (origin : Nat) (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) (part : Part) : Nat × Scalar :=
  let op := QKVTTIR.ops reduce
  (QKVTTIR.storeWord batch.val part, reduce (fun lane =>
    (QKVTTIR.readStream (nativeSample w true first (origin + batch.val))
      (nativeSample w false second (origin + batch.val)) w part lane).foldl op.maximum op.zero))

def nativeScan := scanFrom 0

private theorem scan_stream_eq (origin : Nat) (first second : Bank) (w : Width)
    (batch : Fin (Layout.ofWidth w).batches) (part : Part) (lane : Lane)
    (hslot : origin + batch.val < 7) :
    QKVTTIR.readStream (nativeSample w true first (origin + batch.val))
      (nativeSample w false second (origin + batch.val)) w part lane =
      QKVTTIR.emittedStream (compactBank w true (bankView origin first))
        (compactBank w false (bankView origin second))
        w batch.val part lane := by
  obtain ⟨hfirst, htotal⟩ := loop_counts w part
  simp only [QKVTTIR.emittedStream, QKVTTIR.readStream, hfirst, htotal]
  congr 1
  · apply List.map_congr_left
    intro i hi
    have h := List.mem_range.mp hi
    have ht := (visited_iff part (Layout.ofWidth w).firstTiles i).mp h
    rw [QKVTTIR.first_tile w part i h]
    exact native_sample_eq w true first origin batch.val _ lane batch.isLt hslot ht
  · apply List.map_congr_left
    intro i hi
    obtain ⟨j, hj, he⟩ := List.mem_range'.mp hi
    let l := Layout.ofWidth w
    change j < visitCount part (l.firstTiles + l.secondTiles) - visitCount part l.firstTiles at hj
    change i = visitCount part l.firstTiles + 1 * j at he
    have hlo : visitCount part l.firstTiles ≤ i := by omega
    have hhi : i < visitCount part (l.firstTiles + l.secondTiles) := by omega
    have ht : globalTile part i - l.firstTiles < l.secondTiles := by
      have hb : 0 < l.secondTiles := by cases w <;> decide
      have := (visited_iff part (l.firstTiles + l.secondTiles) i).mp hhi
      omega
    have htile : QKVTTIR.secondTile l part i = globalTile part i - l.firstTiles :=
      v_tile_word w part i hlo hhi
    rw [htile]
    exact native_sample_eq w false second origin batch.val _ lane batch.isLt hslot ht

theorem scan_from_equiv (origin : Nat) (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) (part : Part)
    (hslot : origin + batch.val < 7) :
    scanFrom origin first second reduce w batch part =
      QKVTTIR.referenceStore (compactBank w true (bankView origin first))
        (compactBank w false (bankView origin second)) reduce w batch part := by
  rw [← QKVTTIR.store_lowering]
  simp only [scanFrom, scan_stream_eq origin first second w batch part _ hslot, QKVTTIR.emittedStore]

theorem scan_equiv (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) (part : Part) :
    nativeScan first second reduce w batch part =
      QKVTTIR.referenceStore (compactBank w true first) (compactBank w false second) reduce w batch part := by
  have hb : batch.val < 7 := Nat.lt_of_lt_of_le batch.isLt (by cases w <;> decide)
  simpa only [nativeScan, bank_view_zero] using
    scan_from_equiv 0 first second reduce w batch part (by simpa using hb)


/-- The pack reads 64 consecutive rows. Its tile-row origin advances by
128 physical rows only when the head was cropped to 64 logical rows. -/
def rowFactor (crop : Bool) : Nat := if crop then 2 else 1

private theorem pack_physical (crop : Bool) (tile : Nat) (lane : Fin 64) (col : Nat)
    (hc : col < 768) :
    physicalIndex crop ((tile * 64 + lane.val) * 768 + col) =
      (tile * 64 * rowFactor crop + lane.val) * 768 + col := by
  have hl := lane.isLt
  cases crop <;> simp [physicalIndex, rowFactor] <;> omega

private def packOffset (batch bankRow : Nat) (crop : Bool) (lane : Fin 64)
    (startCol colLane : Nat) : BitVec 32 :=
  word batch * word BankPitch + word bankRow * word (rowFactor crop) * word 768 +
    word lane.val * word 768 + word startCol + word colLane

private theorem pack_read (w : Width) (qk : Bool) (raw : Bank) (origin batch tile : Nat)
    (lane : Fin 64) (startCol colLane : Nat) (hb : origin + batch < 7)
    (ht : tile < logicalTiles w qk / 24) (hc : startCol + colLane < 768) :
    raw (packOffset (origin + batch) (tile * 64) (cropped w qk) lane startCol colLane).toNat =
      compactBank w qk (bankView origin raw) (batch * compactPitch w qk + ((tile * 64 + lane.val) * 768 +
        (startCol + colLane))) := by
  have hl := lane.isLt
  have hx : (tile * 64 + lane.val) * 768 + (startCol + colLane) < compactPitch w qk := by
    cases w <;> cases qk <;> simp [compactPitch, logicalTiles, Layout.ofWidth] at * <;> omega
  have haddr : (origin + batch) * BankPitch + (tile * 64 * rowFactor (cropped w qk) + lane.val) * 768 +
      (startCol + colLane) < 2147483648 := by
    change (origin + batch) * 1179648 + (tile * 64 * rowFactor (cropped w qk) + lane.val) * 768 +
      (startCol + colLane) < 2147483648
    cases w <;> cases qk <;> simp [cropped, rowFactor, logicalTiles, Layout.ofWidth] at * <;> omega
  rw [compact_at w qk (bankView origin raw) batch _ hx, pack_physical _ tile lane _ hc]
  simp only [packOffset, word, ← BitVec.ofNat_add, ← BitVec.ofNat_mul, BitVec.toNat_ofNat]
  rw [show (origin + batch) * BankPitch + tile * 64 * rowFactor (cropped w qk) * 768 +
      lane.val * 768 + startCol + colLane =
    (origin + batch) * BankPitch + ((tile * 64 * rowFactor (cropped w qk) + lane.val) * 768 +
      (startCol + colLane)) by simp only [Nat.add_mul]; omega]
  have haddr' : (origin + batch) * BankPitch + ((tile * 64 * rowFactor (cropped w qk) + lane.val) * 768 +
      (startCol + colLane)) < 2^32 := by
    rw [← Nat.add_assoc]
    exact Nat.lt_trans haddr (by decide)
  rw [Nat.mod_eq_of_lt haddr']
  simp only [bankView, Nat.add_mul, Nat.add_assoc]

open QKVPack

def packWord (origin : Nat) (w : Width) (p : Point (Geometry.ofWidth w)) : BitVec 32 :=
  let qk := p.startRow < (Geometry.ofWidth w).firstRows
  let bankRow := word p.startRow - word (if qk then 0 else (Geometry.ofWidth w).firstRows)
  word (origin + p.batch.val) * word BankPitch + bankRow * word (rowFactor (cropped w qk)) * word 768 +
    word p.rowLane.val * word 768 + word p.startCol + word p.colLane.val

def packFrom (origin : Nat) (op : Arithmetic) (raw : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) : Writes :=
  let bank := if p.startRow < (Geometry.ofWidth w).firstRows then raw.first else raw.second
  let x := bank (packWord origin w p).toNat
  ⟨(rowWord p).toNat, (transposeWord p).toNat, quantize op x (raw.scales p.batch.val)⟩

def nativePack := packFrom 0

private theorem pack_input (origin : Nat) (raw : Inputs) (w : Width) (p : Point (Geometry.ofWidth w))
    (hb : origin + p.batch.val < 7) :
    (if p.startRow < (Geometry.ofWidth w).firstRows then raw.first else raw.second) (packWord origin w p).toNat =
    (if p.startRow < (Geometry.ofWidth w).firstRows then (compactInputs w (inputsView origin raw)).first
      else (compactInputs w (inputsView origin raw)).second) (tiledInputWord p).toNat := by
  have hc : p.startCol + p.colLane.val < 768 := (full_tile p).2
  have hi : (tileAddress p).2 < 2^31 := (input_bounds w p).2
  have hword : (tiledInputWord p).toNat = (tileAddress p).2 := by
    rw [input_word, word, BitVec.toNat_ofNat, Nat.mod_eq_of_lt (by omega)]
  rw [hword]
  by_cases h : p.startRow < (Geometry.ofWidth w).firstRows
  · have ht : p.rowTile.val < logicalTiles w true / 24 := by
      change p.rowTile.val < (Layout.ofWidth w).firstTiles / 24
      change p.rowTile.val * 64 < (Layout.ofWidth w).firstTiles / 24 * 64 at h
      omega
    have hstride : (Geometry.ofWidth w).firstStride = compactPitch w true := rfl
    have hcols : (Geometry.ofWidth w).cols = 768 := rfl
    simp only [packWord, tileAddress, h, if_true, decide_true, compactInputs, inputsView, hcols, hstride]
    simpa only [Point.startRow, RowsPerTile, word, Nat.reduceSub, BitVec.sub_zero,
      Nat.sub_zero,
      packOffset, Nat.add_mul, Nat.add_assoc] using
      pack_read w true raw.first origin p.batch.val p.rowTile.val p.rowLane p.startCol p.colLane.val hb ht hc
  · let t := p.rowTile.val - (Geometry.ofWidth w).firstBlocks
    have hle : (Geometry.ofWidth w).firstRows ≤ p.startRow := Nat.le_of_not_gt h
    have hs : p.startRow - (Geometry.ofWidth w).firstRows = t * 64 := by
      simp only [Point.startRow, Geometry.firstRows, t, RowsPerTile, Nat.sub_mul]
    have hmax : (Geometry.ofWidth w).firstRows < 2^32 := by cases w <;> decide
    have hsub : word p.startRow - word (Geometry.ofWidth w).firstRows = word (t * 64) := by
      rw [← hs]
      exact BitVec.ofNat_sub_ofNat_of_le _ _ hmax hle
    have ht : t < logicalTiles w false / 24 := by
      have hp := p.rowTile.isLt
      change (Geometry.ofWidth w).firstBlocks * 64 ≤ p.rowTile.val * 64 at hle
      change p.rowTile.val - (Geometry.ofWidth w).firstBlocks < (Geometry.ofWidth w).secondBlocks
      omega
    have hstride : (Geometry.ofWidth w).secondStride = compactPitch w false := rfl
    have hcols : (Geometry.ofWidth w).cols = 768 := rfl
    simpa only [packWord, tileAddress, h, if_false, decide_false, compactInputs, inputsView, hs, hsub, hstride, hcols,
      packOffset, Nat.add_mul, Nat.add_assoc] using
      pack_read w false raw.second origin p.batch.val t p.rowLane p.startCol p.colLane.val hb ht hc

theorem pack_from_bytes (origin : Nat) (op : Arithmetic) (raw : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) (hslot : origin + p.batch.val < 7) :
    packFrom origin op raw w p = emitted op (compactInputs w (inputsView origin raw)) w p := by
  simp only [packFrom, emitted, pack_input origin raw w p hslot, compactInputs, inputsView]

theorem native_pack_bytes (op : Arithmetic) (raw : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) :
    nativePack op raw w p = emitted op (compactInputs w raw) w p := by
  have hb : p.batch.val < 7 := Nat.lt_of_lt_of_le p.batch.isLt (by cases w <;> decide)
  simpa only [nativePack, inputs_view_zero] using pack_from_bytes 0 op raw w p (by simpa using hb)

theorem pack_from_equiv (origin : Nat) (op : Arithmetic) (raw : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) (hslots : origin + (Layout.ofWidth w).batches ≤ 7) :
    observeGluon (packFrom origin op raw w) p = reference op (compactInputs w (inputsView origin raw)) p := by
  have h : packFrom origin op raw w = emitted op (compactInputs w (inputsView origin raw)) w := by
    funext pt
    exact pack_from_bytes _ _ _ _ _ (Nat.lt_of_lt_of_le (Nat.add_lt_add_left pt.batch.isLt _) hslots)
  rw [h]
  exact gluon_equiv op (compactInputs w (inputsView origin raw)) w p

theorem pack_equiv (op : Arithmetic) (raw : Inputs) (w : Width)
    (p : Point (Geometry.ofWidth w)) :
    observeGluon (nativePack op raw w) p = reference op (compactInputs w raw) p := by
  have h : nativePack op raw w = emitted op (compactInputs w raw) w :=
    funext (native_pack_bytes op raw w)
  rw [h]
  exact gluon_equiv op (compactInputs w raw) w p

#axiomsClean scan_equiv
#axiomsClean physical_in_view
#axiomsClean native_pack_bytes
#axiomsClean pack_equiv

end VeriTile.Bench.Models.NanoGPT.Track1.QKVNative
