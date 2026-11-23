import models.NanoGPT.Track1.QKVScale

/-! The delayed-scale launch reads the complete previous amax row for each
of the two projections, then writes the rounded product and scale floor. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MLPWeights
open VeriTile.Triton.FP
open QKVScale (orderKey fromKey floorValue deviceFloor)

set_option maxRecDepth 2048

structure Matrix where
  layer : Fin 12
  down : Bool
  deriving DecidableEq

def decodeMatrix (id : Fin 24) : Matrix :=
  ⟨⟨id.val / 2, by have := id.isLt; omega⟩, id.val % 2 != 0⟩

def encodeMatrix (m : Matrix) : Fin 24 :=
  ⟨m.layer.val * 2 + (if m.down then 1 else 0), by
    have := m.layer.isLt
    cases m.down <;> simp <;> omega⟩

theorem decode_encode_matrix (m : Matrix) : decodeMatrix (encodeMatrix m) = m := by
  obtain ⟨layer, down⟩ := m
  revert layer down
  decide +kernel

theorem encode_decode_matrix (id : Fin 24) : encodeMatrix (decodeMatrix id) = id := by
  decide +kernel +revert

abbrev Scratch := Bool → Nat → Scalar
abbrev WeightScales := Matrix → Scalar

def referenceGain : Scalar := literal (1 / 400)
def deviceGain : Scalar := Value.ofBits .fp32 0x3b23d70a
def padValue : Scalar := Value.ofBits .fp32 0xff800000

theorem encoded_scale_constants :
    deviceGain = referenceGain ∧ deviceFloor = floorValue := by
  decide +kernel

private theorem pad_is_minimum (x : Scalar) : orderKey padValue ≤ orderKey x := by
  have hx : x.raw < 4294967296 := x.bits.isLt
  change 8388607 ≤ orderKey x
  unfold orderKey
  split
  · decide
  · rename_i hn
    split
    · decide
    · split
      · have hb : x.raw ≤ 4286578688 := by
          by_cases hc : x.raw ≤ 4286578688
          · exact hc
          have he : x.raw / 8388608 % 256 = 255 := by omega
          have hf : x.raw % 8388608 ≠ 0 := by omega
          exfalso
          apply hn
          simp [Value.isNaN, Value.exponent, Value.fraction,
            Format.fractionBits, he, hf]
        omega
      · omega

/-- sizePerThread four, 128 threads, and two register groups. -/
private def scaleLane (thread : Fin 128) (reg : Fin 8) : Fin 1024 :=
  ⟨thread.val * 4 + reg.val % 4 + (reg.val / 4) * 512, by
    have ht := thread.isLt
    have hr := reg.isLt
    omega⟩

def scaleLanes : List (Fin 1024) :=
  (List.finRange 128).flatMap fun t => (List.finRange 8).map (scaleLane t)

theorem scale_lane_coverage (i : Fin 1024) : i ∈ scaleLanes := by
  have hi := i.isLt
  let t : Fin 128 := ⟨i.val / 4 % 128, Nat.mod_lt _ (by decide)⟩
  let r : Fin 8 := ⟨i.val % 4 + 4 * (i.val / 512), by omega⟩
  apply List.mem_flatMap.mpr
  refine ⟨t, List.mem_finRange t, List.mem_map.mpr ⟨r, List.mem_finRange r, Fin.ext ?_⟩⟩
  change i.val / 4 % 128 * 4 + (i.val % 4 + 4 * (i.val / 512)) % 4 +
      ((i.val % 4 + 4 * (i.val / 512)) / 4) * 512 = i.val
  omega

def priorValue (scratch : Scratch) (m : Matrix) (i : Fin 528) : Scalar :=
  scratch m.down (m.layer.val * 528 + i.val)

def referenceMax (scratch : Scratch) (m : Matrix) : Nat :=
  (List.finRange 528).foldl (fun a i => max a (orderKey (priorValue scratch m i))) 0

def paddedValue (scratch : Scratch) (m : Matrix) (i : Fin 1024) : Scalar :=
  if h : i.val < 528 then priorValue scratch m ⟨i.val, h⟩ else padValue

/-- The library and device reduction trees compute the same ordered maximum.
The masked lanes contribute negative infinity. -/
theorem padded_max_equal (scratch : Scratch) (m : Matrix) :
    scaleLanes.foldl (fun a i => max a (orderKey (paddedValue scratch m i))) 0 =
      referenceMax scratch m := by
  let input := fun i => orderKey (priorValue scratch m i)
  let padded := fun i => orderKey (paddedValue scratch m i)
  let original := List.finRange 528
  have orig := (QKVScale.foldl_max_le input original 0 _).mp (Nat.le_refl (referenceMax scratch m))
  have device := (QKVScale.foldl_max_le padded scaleLanes 0 _).mp (Nat.le_refl (scaleLanes.foldl
    (fun a i => max a (padded i)) 0))
  apply Nat.le_antisymm
  · apply (QKVScale.foldl_max_le padded scaleLanes 0 _).mpr
    refine ⟨Nat.zero_le _, fun i _ => ?_⟩
    by_cases hi : i.val < 528
    · simpa [padded, paddedValue, hi] using orig.2 ⟨i.val, hi⟩ (List.mem_finRange _)
    · have h0 := orig.2 0 (List.mem_finRange 0)
      simpa [padded, paddedValue, hi] using Nat.le_trans (pad_is_minimum (priorValue scratch m 0)) h0
  · apply (QKVScale.foldl_max_le input original 0 _).mpr
    refine ⟨Nat.zero_le _, fun i _ => ?_⟩
    have h : i.val < 1024 := by have := i.isLt; omega
    simpa [padded, paddedValue, i.isLt] using device.2 ⟨i.val, h⟩ (scale_lane_coverage ⟨i.val, h⟩)

def clampScale (product floor : Scalar) : Scalar :=
  if lt Config.ieeeValue product floor then floor else product

def referenceScale (scratch : Scratch) (m : Matrix) : Scalar :=
  clampScale (mul32 (fromKey (referenceMax scratch m)) referenceGain) floorValue

def emittedScale (scratch : Scratch) (id : Fin 24) : Bool × (Nat × Scalar) :=
  let m := decodeMatrix id
  let k := scaleLanes.foldl (fun a i => max a (orderKey (paddedValue scratch m i))) 0
  (m.down, m.layer.val, clampScale (mul32 (fromKey k) deviceGain) deviceFloor)

theorem delayed_scale_store_equal (scratch : Scratch) (m : Matrix) :
    emittedScale scratch (encodeMatrix m) = (m.down, m.layer.val, referenceScale scratch m) := by
  simp only [emittedScale, decode_encode_matrix, padded_max_equal, encoded_scale_constants.1,
    encoded_scale_constants.2, referenceScale]

def referenceScales (delayed : Bool) (old : WeightScales) (scratch : Scratch) : WeightScales :=
  if delayed then referenceScale scratch else old

def emittedScales (delayed : Bool) (old : WeightScales) (scratch : Scratch) : WeightScales :=
  if delayed then fun m => (emittedScale scratch (encodeMatrix m)).2.2 else old

theorem scales_equal (delayed : Bool) (old : WeightScales) (scratch : Scratch) :
    emittedScales delayed old scratch = referenceScales delayed old scratch := by
  cases delayed
  · simp only [emittedScales, referenceScales, Bool.false_eq_true, if_false]
  · simp only [emittedScales, referenceScales, if_true]
    funext m
    rw [delayed_scale_store_equal]

#axiomsClean decode_encode_matrix
#axiomsClean encode_decode_matrix
#axiomsClean padded_max_equal
#axiomsClean delayed_scale_store_equal
#axiomsClean scales_equal
end VeriTile.Bench.Models.NanoGPT.Track1.MLPWeights
