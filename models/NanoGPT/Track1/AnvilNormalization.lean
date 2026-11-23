import models.NanoGPT.Track1.AnvilTrace

/-! The fused launch assigns a disjoint tile to each CTA. Both matrices
are overwritten at the same addresses from which their BF16 words are loaded. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilNormalization
open AnvilEpilogue (Bank)
open AnvilTrace (Input Output Dispatch Normalization)
open TailNorm (BF16 widen narrow)

inductive Plane where
  | prepared
  | gram
  deriving DecidableEq

def Plane.tiles (bank : Bank) : Plane → Nat
  | .prepared => bank.dimensions.2.1 * bank.dimensions.2.2 / 2048
  | .gram => AnvilTrace.width bank ^ 2 / 2048

def Plane.offset : Plane → Nat
  | .prepared => 0
  | .gram => 1

abbrev Tiles (bank : Bank) := Fin (Plane.tiles bank .prepared + Plane.tiles bank .gram)
abbrev Batch (bank : Bank) := Fin bank.dimensions.1

def route (bank : Bank) (tile : Tiles bank) : Plane × Nat :=
  let split := Plane.tiles bank .prepared
  if tile.val < split then (.prepared, tile.val) else (.gram, tile.val - split)

def laneOffset (lane : Fin 128) (word : Fin 16) : Nat := 8 * lane.val + word.val % 8 + 1024 * (word.val / 8)

theorem lane_offset_exact (lane : Fin 128) :
    ((((UInt32.ofNat lane.val) <<< (3 : UInt32)) &&& 1016).toNat) = 8 * lane.val := by
  decide +kernel +revert

private def tileBase (bank : Bank) (tile : Tiles bank) (batch : Batch bank) : Plane × Nat :=
  let (plane, localTile) := route bank tile
  (plane, (batch.val * plane.tiles bank + localTile) * 2048)

private def completeCoordinates (bank : Bank) (base : Plane × Nat) (batch : Batch bank)
    (offset : Nat) : Plane × Nat × Fin (2 * bank.dimensions.1) :=
  (base.1, base.2 + offset,
    ⟨2 * batch.val + base.1.offset, by
      have hb := batch.isLt
      have hp : base.1.offset < 2 := by cases base.1 <;> decide
      omega⟩)

def coordinates (bank : Bank) (tile : Tiles bank) (batch : Batch bank)
    (lane : Fin 128) (word : Fin 16) : Plane × Nat × Fin (2 * bank.dimensions.1) :=
  completeCoordinates bank (tileBase bank tile batch) batch (laneOffset lane word)

theorem prepared_cells (bank : Bank) :
    2048 * Plane.tiles bank .prepared = bank.dimensions.2.1 * bank.dimensions.2.2 := by
  cases bank <;> decide

theorem gram_cells (bank : Bank) :
    2048 * Plane.tiles bank .gram = AnvilTrace.width bank * AnvilTrace.width bank := by
  cases bank <;> decide

theorem in_bounds (bank : Bank) (cta : Tiles bank) (batch : Batch bank)
    (lane : Fin 128) (word : Fin 16) :
    let (plane, address, _) := coordinates bank cta batch lane word
    address < (bank.dimensions.1 * plane.tiles bank) * 2048 ∧ 2 * address < 2 ^ 31 := by
  have hc := cta.isLt
  have hb := batch.isLt
  have hl := lane.isLt
  have hw := word.isLt
  simp only [coordinates, completeCoordinates, tileBase, route]
  split
  all_goals
    cases bank <;> simp only [Plane.tiles, AnvilTrace.width, Bank.dimensions, laneOffset] at * <;> omega

theorem unique_writer (bank : Bank) (cta cta' : Tiles bank) (batch batch' : Batch bank)
    (lane lane' : Fin 128) (word word' : Fin 16)
    (h : (coordinates bank cta batch lane word).1 = (coordinates bank cta' batch' lane' word').1 ∧
      (coordinates bank cta batch lane word).2.1 = (coordinates bank cta' batch' lane' word').2.1) :
    cta = cta' ∧ batch = batch' ∧ lane = lane' ∧ word = word' := by
  have hc := cta.isLt
  have hc' := cta'.isLt
  have hl := lane.isLt
  have hl' := lane'.isLt
  have hw := word.isLt
  have hw' := word'.isLt
  simp only [coordinates, completeCoordinates, tileBase, route] at h
  split at h <;> split at h
  all_goals
    try simp only [reduceCtorEq] at h
    obtain ⟨_, h⟩ := h
    simp only [Fin.ext_iff]
    cases bank <;> simp only [Plane.tiles, AnvilTrace.width, Bank.dimensions, laneOffset] at * <;> omega

private def preparedTile (bank : Bank)
    (i : Fin (bank.dimensions.2.1 * bank.dimensions.2.2)) : Tiles bank :=
  ⟨i.val / 2048, by
    have hi : i.val < 2048 * Plane.tiles bank .prepared :=
      i.isLt.trans_le (prepared_cells bank).symm.le
    omega⟩

private def gramTile (bank : Bank)
    (i : Fin (AnvilTrace.width bank * AnvilTrace.width bank)) : Tiles bank :=
  ⟨Plane.tiles bank .prepared + i.val / 2048, by
    have hi : i.val < 2048 * Plane.tiles bank .gram :=
      i.isLt.trans_le (gram_cells bank).symm.le
    omega⟩

private def laneOf (index : Nat) : Fin 128 :=
  ⟨index % 1024 / 8, by have := Nat.mod_lt index (by decide : 0 < 1024); omega⟩

private def wordOf (index : Nat) : Fin 16 :=
  ⟨8 * (index % 2048 / 1024) + index % 8, by
    have h₁ := Nat.mod_lt index (by decide : 0 < 2048)
    have h₂ := Nat.mod_lt index (by decide : 0 < 8)
    omega⟩

theorem lane_word_exact (i : Nat) :
    laneOffset (laneOf i) (wordOf i) = i % 2048 := by
  unfold laneOffset laneOf wordOf
  simp only
  have h₁ := Nat.mod_mod_of_dvd i (by decide : 8 ∣ 1024)
  have h₂ := Nat.mod_mod_of_dvd i (by decide : 1024 ∣ 2048)
  omega

theorem prepared_coordinate (bank : Bank) (batch : Batch bank)
    (i : Fin (bank.dimensions.2.1 * bank.dimensions.2.2)) :
    coordinates bank (preparedTile bank i) batch (laneOf i.val) (wordOf i.val) =
      (.prepared, batch.val * (bank.dimensions.2.1 * bank.dimensions.2.2) + i.val,
        ⟨2 * batch.val, by have := batch.isLt; omega⟩) := by
  have hi : i.val < 2048 * Plane.tiles bank .prepared :=
    i.isLt.trans_le (prepared_cells bank).symm.le
  have phase : i.val / 2048 < Plane.tiles bank .prepared := by omega
  simp only [coordinates, completeCoordinates, tileBase, route, preparedTile, phase, if_true, Plane.offset, Nat.add_zero,
    lane_word_exact, Prod.mk.injEq, true_and, Nat.add_mul, Nat.mul_assoc]
  rw [Nat.mul_comm (Plane.tiles bank .prepared), prepared_cells]
  exact ⟨by omega, trivial⟩

theorem gram_coordinate (bank : Bank) (batch : Batch bank)
    (i : Fin (AnvilTrace.width bank * AnvilTrace.width bank)) :
    coordinates bank (gramTile bank i) batch (laneOf i.val) (wordOf i.val) =
      (.gram, batch.val * (AnvilTrace.width bank * AnvilTrace.width bank) + i.val,
        ⟨2 * batch.val + 1, by have := batch.isLt; omega⟩) := by
  have phase : ¬ Plane.tiles bank .prepared + i.val / 2048 < Plane.tiles bank .prepared := by omega
  simp only [coordinates, completeCoordinates, tileBase, route, gramTile, phase, if_false, Plane.offset, Nat.add_sub_cancel_left,
    lane_word_exact, Prod.mk.injEq, true_and, Nat.add_mul, Nat.mul_assoc]
  rw [Nat.mul_comm (Plane.tiles bank .gram), gram_cells]
  exact ⟨by omega, trivial⟩

/-- Scalar prefix of the three Hopper programs. The unselected buffer
address may wrap, and the selected one fits in a signed 32-bit word. -/
private def loweredBase (bank : Bank) (tile : Tiles bank) (batch : Batch bank) : Plane × Nat :=
  let t := UInt32.ofNat tile.val
  let b := UInt32.ofNat batch.val
  let split := UInt32.ofNat (Plane.tiles bank .prepared)
  let gram := decide (t ≥ split)
  let shift : UInt32 := match bank with | .qk => 2097056 | .vo => 2096864 | .mlp => 2096096
  let localTile := t + (if gram then shift else 0)
  let stride : UInt32 :=
    match bank with
    | .qk => if gram then 65536 else 196608
    | .vo => 589824
    | .mlp => if gram then 589824 else 2162688
  let offset := b * stride + (localTile <<< 11)
  (if gram then .gram else .prepared, offset.toNat)

set_option maxRecDepth 10000 in
set_option maxHeartbeats 2000000 in
theorem lowered_base_exact (bank : Bank) (tile : Tiles bank) (batch : Batch bank) :
    loweredBase bank tile batch = tileBase bank tile batch := by
  cases bank <;> decide +kernel +revert

private def loweredCoordinates (bank : Bank) (tile : Tiles bank) (batch : Batch bank)
    (lane : Fin 128) (word : Fin 16) : Plane × Nat × Fin (2 * bank.dimensions.1) :=
  let byteVector := 2 * ((((UInt32.ofNat lane.val) <<< (3 : UInt32)) &&& 1016).toNat)
  completeCoordinates bank (loweredBase bank tile batch) batch (byteVector / 2 + word.val % 8 + 1024 * (word.val / 8))

theorem lowered_coordinates_exact (bank : Bank) (tile : Tiles bank) (batch : Batch bank)
    (lane : Fin 128) (word : Fin 16) :
    loweredCoordinates bank tile batch lane word = coordinates bank tile batch lane word := by
  simp only [loweredCoordinates, lowered_base_exact, lane_offset_exact, coordinates, laneOffset,
    Nat.mul_div_cancel_left _ (by decide : 0 < 2)]

private def readStore (input : Input) {bank : Bank} (divisors : Batch bank → AnvilTrace.Divisors)
    (coord : Plane × Nat × Fin (2 * bank.dimensions.1)) : BF16 :=
  let (plane, address, divisorOffset) := coord
  let value := match plane with | .prepared => input.prepared address | .gram => input.gram address
  narrow (input.divide (widen value) (AnvilTrace.packed divisors divisorOffset))

def packedNormalize (trace : Dispatch) : Normalization := fun bank input =>
  if input.eligible then
    let divisors := trace true bank input.gram
    ⟨divisors,
      fun b i => readStore input divisors (loweredCoordinates bank (preparedTile bank i) b (laneOf i.val) (wordOf i.val)),
      fun b i => readStore input divisors (loweredCoordinates bank (gramTile bank i) b (laneOf i.val) (wordOf i.val))⟩
  else AnvilTrace.normalize trace bank input

theorem packed_normalize_eq (trace : Dispatch) : packedNormalize trace = AnvilTrace.normalize trace := by
  funext bank input
  cases h : input.eligible
  · simp [packedNormalize, h]
  simp only [packedNormalize, AnvilTrace.normalize, h, if_true]
  congr
  · funext b i
    rw [lowered_coordinates_exact, prepared_coordinate]
    change narrow (input.divide (widen _) (AnvilTrace.readDivisors (AnvilTrace.packed _) b).denominator) = _
    rw [AnvilTrace.packed_roundtrip]
  · funext b i
    rw [lowered_coordinates_exact, gram_coordinate]
    change narrow (input.divide (widen _) (AnvilTrace.readDivisors (AnvilTrace.packed _) b).square) = _
    rw [AnvilTrace.packed_roundtrip]

#axiomsClean lowered_base_exact
#axiomsClean lowered_coordinates_exact
#axiomsClean lane_offset_exact
#axiomsClean in_bounds
#axiomsClean unique_writer
#axiomsClean prepared_coordinate
#axiomsClean gram_coordinate
#axiomsClean packed_normalize_eq

end VeriTile.Bench.Models.NanoGPT.Track1.AnvilNormalization
