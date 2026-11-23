import models.NanoGPT.Track1.QKVGluonLayout
import models.NanoGPT.Track1.QKVTTIR
import models.NanoGPT.Track1.FP8E4M3
import Init.Data.Rat.Lemmas

/-! The pinned [768,50304] LM head has full tiles. The Gluon path
retains the FP32 multiplication and BF16 rounding, then moves the FP8
byte to both cache layouts using the warp-local exchange. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.LMHead
open VeriTile.Triton.FP QKVPack QKVLaunch
open QKVTTIR (Bank)

def geometry : Geometry := ⟨1, 12, 0, 393, 38633472, 38633472⟩
abbrev HeadPoint := Point geometry

def byte (x : Value .bf16) (gain : Scalar) : UInt8 :=
  let fp32 := cast .fp32 Config.ieeeValue x
  let quant := cast .bf16 Config.ieeeValue (mul32 fp32 gain)
  let widened := cast .fp32 Config.ieeeValue quant
  FP8E4M3.clampConvert widened

private def choosesUpper : Decoded → Bool
  | .nan .. => true
  | .infinity negative => !negative
  | .finite negative m => !negative && decide (448 < m)

/-- The packed minnum selects a BF16 operand in each half. The comparison
uses the same value order after exact widening to FP32. BF16 +448 is 0x43e0. -/
def upperMin (x : Value .bf16) : Value .bf16 :=
  if choosesUpper (cast .fp32 Config.ieeeValue x).decode then Value.ofBits .bf16 0x43e0 else x

/-- The PTX saturating cast retains NaN as NaN and clamps finite overflow
and infinities to the corresponding signed largest E4M3 value. -/
def satfinite (x : Scalar) : UInt8 :=
  match x.decode with
  | .nan .. => 0x7f
  | .infinity negative => if negative then 0xfe else 0x7e
  | .finite negative m => UInt8.ofNat ((if negative then 128 else 0) + FP8E4M3.magnitude m)

private theorem satfinite_upper :
    satfinite (cast .fp32 Config.ieeeValue (Value.ofBits .bf16 0x43e0)) = 0x7e := by
  decide +kernel

private theorem magnitude_at_upper (m : Rat) (h : 448 < m) :
    FP8E4M3.magnitude m = FP8E4M3.magnitude 448 := by
  have nz : m ≠ 0 := by
    intro e
    subst m
    exact (by decide : ¬ (448 : Rat) < 0) h
  have notLe : ¬ m ≤ (448 : Rat) := Rat.not_le.mpr h
  simp [FP8E4M3.magnitude, BEq.beq, nz, Rat.min_def, notLe]

/-- Agreement after BF16 rounding, including signed zeros, subnormals,
infinities and all NaN payloads. The only removed operation is the
explicit lower clamp already performed by the saturating cast. -/
theorem rounded_byte (x : Value .bf16) :
    satfinite (cast .fp32 Config.ieeeValue (upperMin x)) =
      FP8E4M3.clampConvert (cast .fp32 Config.ieeeValue x) := by
  have limit : UInt8.ofNat (FP8E4M3.magnitude 448) = 0x7e := by decide +kernel
  cases h : (cast .fp32 Config.ieeeValue x).decode with
  | nan =>
    simpa [upperMin, h, choosesUpper, FP8E4M3.clampConvert] using satfinite_upper
  | infinity sign =>
    cases sign
    · simpa [upperMin, h, choosesUpper, FP8E4M3.clampConvert] using satfinite_upper
    · simp [upperMin, h, choosesUpper, satfinite, FP8E4M3.clampConvert]
  | finite sign m =>
    cases sign
    · by_cases hm : 448 < m
      · simpa [upperMin, h, choosesUpper, hm, FP8E4M3.clampConvert,
          magnitude_at_upper m hm, limit] using satfinite_upper
      · simp [upperMin, h, choosesUpper, hm, satfinite, FP8E4M3.clampConvert]
    · simp [upperMin, h, choosesUpper, satfinite, FP8E4M3.clampConvert]

def reference (bank : Bank) (gain : Scalar) (p : HeadPoint) : Writes :=
  let index := BitVec.ofNat 64 p.row * BitVec.ofNat 64 50304 + BitVec.ofNat 64 p.col
  ⟨rowAddress p, transposedAddress p, byte (bank index.toNat) gain⟩

/-- The lanes of a packed instruction are two consecutive registers in
one thread's row tile. -/
theorem pair_layout (warp : Fin 8) (lane : Fin 32) (pair : Fin 16) :
    let lo : ThreadValue := ⟨warp, lane, ⟨pair.val * 2, by omega⟩⟩
    let hi : ThreadValue := ⟨warp, lane, ⟨pair.val * 2 + 1, by omega⟩⟩
    (rowIndex lo).1 = (rowIndex hi).1 ∧
      (rowIndex hi).2.val = (rowIndex lo).2.val + 1 ∧
      (rowIndex lo).2.val % 2 = 0 := by
  dsimp [rowIndex]
  constructor
  · apply Fin.ext
    dsimp
    omega
  · omega

def roundedPair (lo hi : Scalar) : Value .bf16 × Value .bf16 :=
  (upperMin (cast .bf16 Config.ieeeValue lo), upperMin (cast .bf16 Config.ieeeValue hi))

/-- A pair occupies adjacent columns in the row layout. The even column
is in the low half, which the PTX instruction takes as its last input. -/
def packedByte (bank : Bank) (gain : Scalar) (p : HeadPoint) : UInt8 :=
  let first := p.col / 2 * 2
  let scaled := fun col =>
    mul32 (cast .fp32 Config.ieeeValue (bank (word p.row * word 50304 + word col).toNat)) gain
  let bounded := roundedPair (scaled first) (scaled (first + 1))
  let own := if p.col % 2 = 0 then bounded.1 else bounded.2
  satfinite (cast .fp32 Config.ieeeValue own)

theorem packed_byte_eq (bank : Bank) (gain : Scalar) (p : HeadPoint) :
    packedByte bank gain p = byte (bank (word p.row * word 50304 + word p.col).toNat) gain := by
  by_cases h : p.col % 2 = 0
  · have hc : p.col / 2 * 2 = p.col := by omega
    simp only [packedByte, if_pos h, roundedPair, hc, byte]
    exact rounded_byte _
  · have hc : p.col / 2 * 2 + 1 = p.col := by omega
    simp only [packedByte, if_neg h, roundedPair, hc, byte]
    exact rounded_byte _

def emitted (bank : Bank) (gain : Scalar) (p : HeadPoint) : Writes :=
  ⟨(rowWord p).toNat, (transposeWord p).toNat, packedByte bank gain p⟩

theorem cache_bounds (p : HeadPoint) :
    rowAddress p < 38633472 ∧ transposedAddress p < 38633472 := by
  obtain ⟨hr,hc⟩ := full_tile p
  have hb := p.batch.isLt
  change p.row < 768 at hr
  change p.col < 50304 at hc
  change p.batch.val < 1 at hb
  change (p.batch.val * 38633472 + p.row * 50304 + p.col < 38633472) ∧
    (p.batch.val * 38633472 + p.col * 768 + p.row < 38633472)
  omega

theorem address_words (p : HeadPoint) :
    (rowWord p).toNat = rowAddress p ∧ (transposeWord p).toNat = transposedAddress p := by
  have ⟨hr, hc⟩ := cache_bounds p
  have br := Nat.lt_trans hr (show 38633472 < 2^32 by decide)
  have bc := Nat.lt_trans hc (show 38633472 < 2^32 by decide)
  constructor
  · simpa only [rowWord, word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, BitVec.toNat_ofNat, rowAddress]
      using Nat.mod_eq_of_lt br
  · simpa only [transposeWord, word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, BitVec.toNat_ofNat, transposedAddress]
      using Nat.mod_eq_of_lt bc

theorem input_word (p : HeadPoint) :
    (word p.row * word 50304 + word p.col).toNat =
      (BitVec.ofNat 64 p.row * BitVec.ofNat 64 50304 + BitVec.ofNat 64 p.col).toNat := by
  obtain ⟨hr,hc⟩ := full_tile p
  change p.row < 768 at hr
  change p.col < 50304 at hc
  have hl : p.row * 50304 + p.col < 2147483648 := by omega
  have hl32 : p.row * 50304 + p.col < 2^32 := by omega
  have hw : p.row * 50304 + p.col < 2^64 :=
    Nat.lt_trans hl (by decide)
  simp only [word, ← BitVec.ofNat_mul, ← BitVec.ofNat_add, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt hl32, Nat.mod_eq_of_lt hw]

theorem emitted_equal (bank : Bank) (gain : Scalar) (p : HeadPoint) :
    emitted bank gain p = reference bank gain p := by
  simp only [emitted, packed_byte_eq, reference, (address_words p).1, (address_words p).2, input_word]

theorem cache_equiv (bank : Bank) (gain : Scalar) (p : HeadPoint) :
    observeGluon (emitted bank gain) p = some (reference bank gain p) := by
  simp only [observeGluon, gluonStores, row_covers, transposed_covers, transferred_cell]
  change (if (emitted bank gain p).byte == (emitted bank gain p).byte then
    some (Writes.mk (emitted bank gain p).rowOffset (transposeWord p).toNat
      (emitted bank gain p).byte) else none) = _
  simp only [beq_self_eq_true, if_true, emitted_equal, (address_words p).2]
  rfl


theorem unique_cache_writers (p q : HeadPoint) :
    (rowAddress p = rowAddress q → p = q) ∧
    (transposedAddress p = transposedAddress q → p = q) := by
  obtain ⟨hp, hcp⟩ := full_tile p
  obtain ⟨hq, hcq⟩ := full_tile q
  change p.row < 768 at hp
  change q.row < 768 at hq
  change p.col < 50304 at hcp
  change q.col < 50304 at hcq
  have bp : p.batch.val < 1 := p.batch.isLt
  have bq : q.batch.val < 1 := q.batch.isLt
  have hb : p.batch = q.batch := Fin.ext (by omega)
  constructor
  · intro h
    change p.batch.val * 38633472 + p.row * 50304 + p.col =
      q.batch.val * 38633472 + q.row * 50304 + q.col at h
    exact point_injective hb (by omega) (by omega)
  · intro h
    change p.batch.val * 38633472 + p.col * 768 + p.row =
      q.batch.val * 38633472 + q.col * 768 + q.row at h
    exact point_injective hb (by omega) (by omega)

theorem row_cache_coverage (i : Fin 38633472) : ∃ p : HeadPoint, rowAddress p = i.val := by
  have hi := i.isLt
  let r : Fin geometry.rows := ⟨i.val / 50304, by change _ < 768; omega⟩
  let c : Fin geometry.cols := ⟨i.val % 50304, by change _ < 50304; omega⟩
  let b : Fin geometry.batches := ⟨0, by decide⟩
  let p := point geometry b r c
  obtain ⟨hr, hc⟩ := point_coordinates geometry b r c
  refine ⟨p, ?_⟩
  have hb : p.batch.val = 0 := rfl
  rw [rowAddress, hb, Nat.zero_mul, Nat.zero_add, show p.row = r.val from hr, show p.col = c.val from hc]
  change i.val / 50304 * 50304 + i.val % 50304 = i.val
  omega

theorem col_cache_coverage (i : Fin 38633472) : ∃ p : HeadPoint, transposedAddress p = i.val := by
  have hi := i.isLt
  let c : Fin geometry.cols := ⟨i.val / 768, by change _ < 50304; omega⟩
  let r : Fin geometry.rows := ⟨i.val % 768, by change _ < 768; omega⟩
  let b : Fin geometry.batches := ⟨0, by decide⟩
  let p := point geometry b r c
  obtain ⟨hr, hc⟩ := point_coordinates geometry b r c
  refine ⟨p, ?_⟩
  have hb : p.batch.val = 0 := rfl
  rw [transposedAddress, hb, Nat.zero_mul, Nat.zero_add, show p.row = r.val from hr, show p.col = c.val from hc]
  change i.val / 768 * 768 + i.val % 768 = i.val
  omega


#axiomsClean cache_bounds
#axiomsClean input_word
#axiomsClean unique_cache_writers
#axiomsClean row_cache_coverage
#axiomsClean col_cache_coverage
#axiomsClean rounded_byte
#axiomsClean pair_layout
#axiomsClean packed_byte_eq
#axiomsClean cache_equiv
end VeriTile.Bench.Models.NanoGPT.Track1.LMHead
