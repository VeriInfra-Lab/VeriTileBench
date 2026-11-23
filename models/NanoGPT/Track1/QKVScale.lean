import models.NanoGPT.Track1.QKVTTIR
import models.NanoGPT.Track1.ScaleRefresh

/-! FP32 scale update over the 32 scratch values written by each QKV scan.
Scale computation uses the NaN-propagating IEEE maximum, an ordered lower
bound, and a single FP32 multiplication. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVScale
open VeriTile.Triton.FP QKVAmax QKVLaunch

/-- Ordering the sign and magnitude fields gives the IEEE numeric order.
The two zero encodings have one key and all NaNs have the top key. The
positive floor absorbs either zero, and the FP32 multiply canonicalizes
NaN in VeriTile's value profile. -/
def orderKey (x : Scalar) : Nat :=
  if x.isNaN then 0x100000000
  else if x.isZero then 0x7fffffff
  else if x.sign then 0xffffffff - x.raw
  else 0x7fffffff + x.raw

def fromKey (key : Nat) : Scalar :=
  if key == 0x100000000 then Value.qNaN .fp32
  else if key < 0x7fffffff then Value.ofBits .fp32 (0xffffffff - key)
  else Value.ofBits .fp32 (key - 0x7fffffff)

private theorem restore_raw (x : Scalar) : Value.ofBits .fp32 x.raw = x := by
  cases x
  simp [Value.ofBits, Value.raw]

/-- Selection by key recovers the original bits on every ordered nonzero
value, including infinities and subnormals. -/
theorem key_roundtrip (x : Scalar) :
    fromKey (orderKey x) =
      if x.isNaN then Value.qNaN .fp32 else if x.isZero then zero else x := by
  by_cases hn : x.isNaN
  · simp [orderKey, fromKey, hn]
  by_cases hz : x.isZero
  · simp [orderKey, fromKey, zero, hn, hz]
    rfl
  have hraw : x.raw < 4294967296 := x.bits.isLt
  by_cases hs : x.sign
  · have hsign : x.raw / 2147483648 ≠ 0 := by
      simpa [Value.sign, Format.fractionBits] using hs
    have hzero : x.raw ≠ 2147483648 := by
      intro h
      have h' : x.isZero = true := by
        unfold Value.isZero Value.exponent Value.fraction
        rw [h]
        decide +kernel
      exact hz h'
    have hlt : 4294967295 - x.raw < 2147483647 := by omega
    have hnotNaN : 4294967295 - x.raw ≠ 4294967296 := by omega
    have hsub : 4294967295 - (4294967295 - x.raw) = x.raw := by omega
    simp [orderKey, fromKey, hn, hz, hs, hnotNaN, hlt, hsub, restore_raw]
  · have hsign : x.raw / 2147483648 = 0 := by
      simpa [Value.sign, Format.fractionBits] using hs
    have hnotNaN : 2147483647 + x.raw ≠ 4294967296 := by omega
    have hlt : ¬ 2147483647 + x.raw < 2147483647 := by omega
    simp [orderKey, fromKey, hn, hz, hs, hnotNaN, hlt, restore_raw]

def floorValue : Scalar := literal (1 / 1000000000000)
def gain : Scalar := literal (1 / 448)

/-- Constants in the emitted setp/selp and mul instructions. -/
def deviceFloor : Scalar := Value.ofBits .fp32 0x2b8cbccc
def deviceGain : Scalar := Value.ofBits .fp32 0x3b124925

theorem encoded_constants : deviceFloor = floorValue ∧ deviceGain = gain := by decide +kernel

inductive ReduceTree where
  | lane (index : Part)
  | branch (a b : ReduceTree)

def ReduceTree.leaves : ReduceTree → List Part
  | .lane i => [i]
  | .branch a b => a.leaves ++ b.leaves

def ReduceTree.eval (input : Part → Nat) : ReduceTree → Nat
  | .lane i => input i
  | .branch a b => max (a.eval input) (b.eval input)

def xorLane (i j : Part) : Part :=
  ⟨i.val ^^^ j.val, Nat.xor_lt_two_pow (n := 5) i.isLt j.isLt⟩

/-- The five shfl.sync.bfly steps and the final store from lane zero. -/
def butterfly : List Part → (Part → ReduceTree) → Part → ReduceTree
  | [], input => input
  | delta :: rest, input => butterfly rest (fun i => .branch (input i) (input (xorLane i delta)))

def warpTree : ReduceTree :=
  butterfly [16, 8, 4, 2, 1] ReduceTree.lane 0

def allParts : List Part := List.finRange 32

theorem warp_covers : ∀ i : Part, i ∈ warpTree.leaves := by decide +kernel

theorem warp_visits_once : warpTree.leaves.Nodup := by decide +kernel

theorem foldl_max_le {ι : Type} (input : ι → Nat) (lanes : List ι) (seed bound : Nat) :
    lanes.foldl (fun n i => max n (input i)) seed ≤ bound ↔
      seed ≤ bound ∧ ∀ i ∈ lanes, input i ≤ bound := by
  induction lanes generalizing seed with
  | nil => simp
  | cons i rest ih =>
    simp only [List.foldl_cons, ih, Nat.max_le, List.mem_cons]
    constructor
    · rintro ⟨⟨hs, hi⟩, hr⟩
      refine ⟨hs, ?_⟩
      intro j hj
      rcases hj with rfl | hj
      · exact hi
      · exact hr j hj
    · rintro ⟨hs, h⟩
      exact ⟨⟨hs, h i (Or.inl rfl)⟩, fun j hj => h j (Or.inr hj)⟩

private theorem eval_max_le (input : Part → Nat) (tree : ReduceTree) (bound : Nat) :
    tree.eval input ≤ bound ↔ ∀ i ∈ tree.leaves, input i ≤ bound := by
  induction tree with
  | lane i => simp [ReduceTree.eval, ReduceTree.leaves]
  | branch a b ha hb =>
    simp only [ReduceTree.eval, ReduceTree.leaves, Nat.max_le, ha, hb, List.mem_append]
    constructor
    · rintro ⟨hl, hr⟩ i (h | h)
      · exact hl i h
      · exact hr i h
    · intro h
      exact ⟨fun i hi => h i (Or.inl hi), fun i hi => h i (Or.inr hi)⟩

/-- Both reductions compute the least upper bound of the 32 input keys.
This also covers every hardware reduction order with the same lane set. -/
theorem reduce_equal (input : Part → Nat) (tree : ReduceTree) (covers : ∀ i : Part, i ∈ tree.leaves) :
    tree.eval input = allParts.foldl (fun n i => max n (input i)) 0 := by
  apply Nat.le_antisymm
  · apply (eval_max_le _ _ _).mpr
    intro i _
    have hf := (foldl_max_le input allParts 0 _).mp (Nat.le_refl _)
    exact hf.2 i (List.mem_finRange i)
  · apply (foldl_max_le _ _ _ _).mpr
    refine ⟨Nat.zero_le _, fun i _ => ?_⟩
    exact (eval_max_le input tree _).mp (Nat.le_refl _) i (covers i)

theorem scale_store_address (w : Width) (batch : Fin (Layout.ofWidth w).batches) :
    (word batch.val).toNat = batch.val := by
  have hs : (Layout.ofWidth w).batches < 2^32 := by cases w <;> decide
  simpa only [word, BitVec.toNat_ofNat] using Nat.mod_eq_of_lt (Nat.lt_trans batch.isLt hs)

def referenceStore (scratch : Nat → Scalar) (w : Width) (batch : Fin (Layout.ofWidth w).batches) :
    Nat × Scalar :=
  let amax := allParts.foldl (fun n part => max n (orderKey (scratch (partialOffset batch.val part)))) 0
  (batch.val, mul32 (fromKey (max amax (orderKey floorValue))) gain)

def fusedStore (scratch : Nat → Scalar) (w : Width) (batch : Fin (Layout.ofWidth w).batches) :
    Nat × Scalar :=
  let amax := warpTree.eval (fun part => orderKey (scratch (QKVTTIR.storeWord batch.val part)))
  let floored := if amax < orderKey deviceFloor then deviceFloor else fromKey amax
  ((word batch.val).toNat, mul32 floored deviceGain)

theorem store_equal (scratch : Nat → Scalar) (w : Width) (batch : Fin (Layout.ofWidth w).batches) :
    fusedStore scratch w batch = referenceStore scratch w batch := by
  obtain ⟨hf, hg⟩ := encoded_constants
  have hdecode : fromKey (orderKey floorValue) = floorValue := by decide +kernel
  simp only [fusedStore, referenceStore, scale_store_address w batch, QKVTTIR.store_address w batch,
    reduce_equal _ _ warp_covers, hf, hg]
  split
  · rename_i h
    rw [Nat.max_eq_right (Nat.le_of_lt h), hdecode]
  · rename_i h
    rw [Nat.max_eq_left (Nat.le_of_not_gt h)]

/-- Materialize the rows produced by the (batch, part) grid. Every valid
offset has one producer by store_disjoint and partial_store_bounds. -/
def partialBuffer (w : Width) (scan : Fin (Layout.ofWidth w).batches → Part → Nat × Scalar)
    (offset : Nat) : Scalar :=
  let row := offset / 32
  if valid : row < (Layout.ofWidth w).batches then
    (scan ⟨row, valid⟩ ⟨offset % 32, Nat.mod_lt _ (by decide)⟩).2
  else zero

theorem read_scan_store (w : Width) (scan : Fin (Layout.ofWidth w).batches → Part → Nat × Scalar)
    (batch : Fin (Layout.ofWidth w).batches) (part : Part) :
    partialBuffer w scan (partialOffset batch.val part) = (scan batch part).2 := by
  have hpart : part.val < 32 := part.isLt
  have hrow : partialOffset batch.val part / 32 = batch.val := by
    unfold partialOffset
    omega
  have hcol : partialOffset batch.val part % 32 = part.val := by
    unfold partialOffset
    omega
  simp only [partialBuffer, hrow, hcol, dif_pos batch.isLt]

/-- Compose the two QKV kernels at their scratch buffer. All scales consumed
by the common FP8 quantizer agree after every weight refresh. -/
theorem scales_from_weights_equal (first second : QKVTTIR.Bank)
    (reduce : (Lane → Scalar) → Scalar) (w : Width) (batch : Fin (Layout.ofWidth w).batches) :
    fusedStore (partialBuffer w (QKVTTIR.emittedStore first second reduce w)) w batch =
      referenceStore (partialBuffer w (QKVTTIR.referenceStore first second reduce w)) w batch := by
  have hscan : QKVTTIR.emittedStore first second reduce w =
      QKVTTIR.referenceStore first second reduce w := by
    funext row part
    exact QKVTTIR.store_lowering first second reduce w row part
  rw [store_equal, hscan]

#axiomsClean key_roundtrip
#axiomsClean read_scan_store
#axiomsClean scales_from_weights_equal
#axiomsClean encoded_constants
#axiomsClean warp_covers
#axiomsClean warp_visits_once
#axiomsClean reduce_equal
#axiomsClean scale_store_address
#axiomsClean store_equal
end VeriTile.Bench.Models.NanoGPT.Track1.QKVScale
