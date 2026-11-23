import models.NanoGPT.Track1.QKVLaunch
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.Specification

/-! Value interpretation of the QKV amax TTIR fragment. The emitted code
uses signed i32 loop bounds, BF16 loads, FP32 extension, addition and abs,
and an ordered maxnumf accumulator in each lane. The final tt.reduce uses
the same maxnumf region on both sides. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVTTIR
open VeriTile Triton.FP QKVAmax QKVLaunch
open scoped VeriTile.Spec

abbrev Scalar := Value .fp32
abbrev Bank := Nat → Value .bf16

def ops (reduce : (Lane → Scalar) → Scalar) : Ops Scalar where
  zero := Value.zero .fp32
  add := add Config.ieeeValue
  abs := Value.abs
  maximum := fun x y =>
    if x.isNaN || lt Config.ieeeValue x y then y else x
  reduce := reduce

def banks (first second : Bank) (w : Width) (batch : Nat) : Banks Scalar :=
  let l := Layout.ofWidth w
  ⟨fun i => cast .fp32 Config.ieeeValue (first (batch * l.firstStride + i)),
   fun i => cast .fp32 Config.ieeeValue (second (batch * l.secondStride + i))⟩

private theorem zero_add_commutes (x : Scalar) :
    add Config.ieeeValue (Value.zero .fp32) x = add Config.ieeeValue x (Value.zero .fp32) := by
  unfold add
  have hz : (prepare Config.ieeeValue (Value.zero .fp32)).decode = .finite false 0 := by decide +kernel
  rw [hz]
  cases (prepare Config.ieeeValue x).decode with
  | nan => rfl
  | infinity => rfl
  | finite sign m =>
    cases sign
    · change round .fp32 Config.ieeeValue (0 + m) _ = round .fp32 Config.ieeeValue (m + 0) _
      simp [Rat.zero_add, Rat.add_zero, Bool.and_comm]
    · change round .fp32 Config.ieeeValue (0 + -m) _ = round .fp32 Config.ieeeValue (-m + 0) _
      simp [Rat.zero_add, Rat.add_zero]

def firstTile (part : Part) (i : Nat) : Nat :=
  (word part.val + word i * word 32).toNat

def secondTile (l : Layout) (part : Part) (i : Nat) : Nat :=
  (word part.val + word i * word 32 - word l.firstTiles).toNat

theorem first_tile (w : Width) (part : Part) (i : Nat)
    (h : i < visitCount part (Layout.ofWidth w).firstTiles) :
    firstTile part i = globalTile part i := by
  have ht : (Layout.ofWidth w).firstTiles < 2^32 := by cases w <;> decide
  have hi := Nat.lt_trans ((visited_iff part (Layout.ofWidth w).firstTiles i).mp h) ht
  simpa only [firstTile, globalTile, word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, BitVec.toNat_ofNat]
    using Nat.mod_eq_of_lt hi

def unmaskedSample (bank : Bank) (batch stride tile : Nat) (lane : Lane) : Scalar :=
  let address := (offsetI32 batch stride tile lane).toInt.toNat
  let value := cast .fp32 Config.ieeeValue (bank address)
  (add Config.ieeeValue value (Value.zero .fp32)).abs

def readStream (first second : Nat → Lane → Scalar) (w : Width) (part : Part)
    (lane : Lane) : List Scalar :=
  let l := Layout.ofWidth w
  let split := visitsI32 l.firstTiles part
  let total := visitsI32 (l.firstTiles + l.secondTiles) part
  (List.range split).map (fun i => first (firstTile part i) lane) ++
  (List.range' split (total - split)).map (fun i => second (secondTile l part i) lane)

def emittedStream (first second : Bank) (w : Width) (batch : Nat) (part : Part)
    (lane : Lane) : List Scalar :=
  let l := Layout.ofWidth w
  readStream (unmaskedSample first batch l.firstStride) (unmaskedSample second batch l.secondStride)
    w part lane

theorem stream_lowering (first second : Bank) (reduce) (w : Width) (batch : Fin (Layout.ofWidth w).batches)
    (part : Part) (lane : Lane) :
    emittedStream first second w batch.val part lane =
      alignedStream (ops reduce) (banks first second w batch.val)
        (Layout.ofWidth w).firstTiles (Layout.ofWidth w).secondTiles part lane := by
  let l := Layout.ofWidth w
  obtain ⟨hfirst, htotal⟩ := loop_counts w part
  simp only [emittedStream, readStream, alignedStream]
  rw [hfirst, htotal]
  congr 1
  · apply List.map_congr_left
    intro i hi
    have h := List.mem_range.mp hi
    have ht := (visited_iff part l.firstTiles i).mp h
    have ha := (qk_bounds w batch.val (globalTile part i) lane batch.isLt ht).2
    simp only [first_tile w part i h, unmaskedSample, offset_sign_extend _ _ _ _ ha,
      Int.toNat_natCast, elementOffset, qkSample, ops, banks, Nat.add_assoc]
  · apply List.map_congr_left
    intro i hi
    obtain ⟨j,hj,he⟩ := List.mem_range'.mp hi
    change j < visitCount part (l.firstTiles + l.secondTiles) - visitCount part l.firstTiles at hj
    change i = visitCount part l.firstTiles + 1 * j at he
    have hlo : visitCount part l.firstTiles ≤ i := by omega
    have hhi : i < visitCount part (l.firstTiles + l.secondTiles) := by omega
    have ht : globalTile part i - l.firstTiles < l.secondTiles := by
      have hb : 0 < l.secondTiles := by cases w <;> decide
      have := (visited_iff part (l.firstTiles + l.secondTiles) i).mp hhi
      omega
    have htile : secondTile l part i = globalTile part i - l.firstTiles := v_tile_word w part i hlo hhi
    have ha := (v_bounds w batch.val (globalTile part i - l.firstTiles) lane batch.isLt ht).2
    change elementOffset batch.val l.secondStride (globalTile part i - l.firstTiles) lane < 2^31 at ha
    change unmaskedSample second batch.val l.secondStride (secondTile l part i) lane =
      vSample (ops reduce) (banks first second w batch.val) l.firstTiles part i lane
    simp only [htile, unmaskedSample, offset_sign_extend _ _ _ _ ha,
      Int.toNat_natCast, elementOffset, vSample, ops, banks, Nat.add_assoc, zero_add_commutes, l]

def storeWord (batch : Nat) (part : Part) : Nat :=
  (word batch * word 32 + word part.val).toNat

theorem store_address (w : Width) (batch : Fin (Layout.ofWidth w).batches) (part : Part) :
    storeWord batch.val part = partialOffset batch.val part := by
  have ht : (Layout.ofWidth w).batches * 32 < 2^32 := by cases w <;> decide
  have hi := Nat.lt_trans (partial_store_bounds w batch.val part batch.isLt) ht
  simpa only [storeWord, partialOffset, word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, BitVec.toNat_ofNat]
    using Nat.mod_eq_of_lt hi

/-- The row-major store address and the value passed to the unchanged
lane reduction, using the emitted aligned TTIR instructions. -/
def emittedStore (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) (part : Part) : Nat × Scalar :=
  (storeWord batch.val part,
    reduce (fun lane => (emittedStream first second w batch.val part lane).foldl
      (ops reduce).maximum (ops reduce).zero))

def referenceStore (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) (part : Part) : Nat × Scalar :=
  let l := Layout.ofWidth w
  let op := ops reduce
  let input := banks first second w batch.val
  (storeWord batch.val part, reduce (fun lane =>
    ((List.range (referenceVisitsI32 l part)).map (fun i =>
      originalSample op input l.firstTiles l.secondTiles part i lane)).foldl op.maximum op.zero))

theorem store_lowering (first second : Bank) (reduce : (Lane → Scalar) → Scalar)
    (w : Width) (batch : Fin (Layout.ofWidth w).batches) (part : Part) :
    emittedStore first second reduce w batch part = referenceStore first second reduce w batch part := by
  simp only [emittedStore, referenceStore, reference_loop_count w part,
    stream_lowering first second reduce w batch, ← streams_equal, originalStream]

/-- Both entry points store within the scratch buffer at an address that
uniquely identifies the launching batch and part. -/
theorem emitted_store_address (first second : Bank) (reduce) (w : Width)
    (batch : Fin (Layout.ofWidth w).batches) (part : Part) :
    (emittedStore first second reduce w batch part).1 = partialOffset batch.val part :=
  store_address w batch part

#axiomsClean stream_lowering
#axiomsClean store_lowering
#axiomsClean emitted_store_address
end VeriTile.Bench.Models.NanoGPT.Track1.QKVTTIR
