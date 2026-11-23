import VeriTile.Meta.StatementAudit

/-! The two scans of the packed QKV amax. The proof retains the ordered
per-lane FP additions, abs, maximum and the common final tile reduction.
The candidate changes only how the two read regions are selected. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.QKVAmax

abbrev Tile := 2048
abbrev Parts := 32
abbrev Lane := Fin Tile
abbrev Part := Fin Parts

def visitCount (part : Part) (tiles : Nat) : Nat :=
  (tiles + 31 - part.val) / 32

def globalTile (part : Part) (i : Nat) : Nat := part.val + i * 32

theorem visited_iff (part : Part) (tiles i : Nat) :
    i < visitCount part tiles ↔ globalTile part i < tiles := by
  have hp := part.isLt
  change part.val < 32 at hp
  unfold visitCount globalTile
  omega

theorem count_le (part : Part) (first second : Nat) :
    visitCount part first ≤ visitCount part (first + second) := by
  unfold visitCount
  omega

theorem range_split (n m : Nat) (h : m ≤ n) :
    List.range n = List.range m ++ List.range' m (n-m) := by
  simp only [List.range_eq_range']
  have he := @List.range'_append_1 0 m (n-m)
  rw [Nat.zero_add, Nat.add_sub_of_le h] at he
  exact he.symm

theorem first_index (part : Part) (tiles i : Nat) (lane : Lane)
    (h : i < visitCount part tiles) :
    globalTile part i * Tile + lane.val < tiles * Tile := by
  have hb := (visited_iff part tiles i).mp h
  have hl := lane.isLt
  change lane.val < 2048 at hl
  change globalTile part i * 2048 + lane.val < tiles * 2048
  omega

theorem second_index (part : Part) (first second i : Nat) (lane : Lane)
    (lower : visitCount part first ≤ i)
    (upper : i < visitCount part (first + second)) :
    ¬ globalTile part i * Tile + lane.val < first * Tile ∧
      globalTile part i * Tile + lane.val - first * Tile =
        (globalTile part i - first) * Tile + lane.val ∧
      (globalTile part i - first) * Tile + lane.val < second * Tile := by
  have lo : first ≤ globalTile part i := by
    apply Nat.le_of_not_gt
    intro ht
    exact (Nat.not_lt.mpr lower) ((visited_iff part first i).mpr ht)
  have hi := (visited_iff part (first+second) i).mp upper
  have hl := lane.isLt
  change lane.val < 2048 at hl
  change ¬ globalTile part i * 2048 + lane.val < first * 2048 ∧
    globalTile part i * 2048 + lane.val - first * 2048 =
      (globalTile part i - first) * 2048 + lane.val ∧
    (globalTile part i - first) * 2048 + lane.val < second * 2048
  omega

/-- Launch ownership for every original tile, including nonuniform trip
counts in the last part. There is a unique (part, iteration) producer. -/
theorem visits_once (tiles : Nat) (tile : Fin tiles) :
    ∃ part : Part, ∃ i, i < visitCount part tiles ∧ globalTile part i = tile.val ∧
      ∀ p : Part, ∀ j, globalTile p j = tile.val → p = part ∧ j = i := by
  let p : Part := ⟨tile.val % 32, Nat.mod_lt _ (by decide)⟩
  let i := tile.val / 32
  have h : globalTile p i = tile.val := by
    have ht := Nat.div_add_mod tile.val 32
    simp only [p,i,globalTile]
    omega
  refine ⟨p,i,(visited_iff p tiles i).mpr (by rw [h]; exact tile.isLt),h,?_⟩
  intro q j hj
  have hq := q.isLt
  change q.val < 32 at hq
  change q.val + j * 32 = tile.val at hj
  constructor
  · apply Fin.ext
    change q.val = tile.val % 32
    omega
  · dsimp [i]
    omega

/-- The actual load views include the batch strides and BF16-to-FP32
promotion. Zero is the promoted +0 fill. The backend supplies the original
FP primitives; no algebraic identity about them is needed. -/
structure Ops (Scalar : Type) where
  zero : Scalar
  add : Scalar → Scalar → Scalar
  abs : Scalar → Scalar
  maximum : Scalar → Scalar → Scalar
  reduce : (Lane → Scalar) → Scalar

structure Banks (Scalar : Type) where
  first : Nat → Scalar
  second : Nat → Scalar

def originalSample {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (first second : Nat) (part : Part) (i : Nat) (lane : Lane) : Scalar :=
  let offset := globalTile part i * Tile + lane.val
  let valid := offset < (first + second) * Tile
  let inFirst := offset < first * Tile
  let qk := if valid && inFirst then banks.first offset else ops.zero
  let v := if valid && !inFirst then banks.second (offset - first * Tile) else ops.zero
  ops.abs (ops.add qk v)

def qkSample {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (part : Part) (i : Nat) (lane : Lane) : Scalar :=
  ops.abs (ops.add (banks.first (globalTile part i * Tile + lane.val)) ops.zero)

def vSample {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (first : Nat) (part : Part) (i : Nat) (lane : Lane) : Scalar :=
  ops.abs (ops.add ops.zero (banks.second ((globalTile part i - first) * Tile + lane.val)))

theorem sample_qk {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (first second : Nat) (part : Part) (i : Nat) (lane : Lane)
    (h : i < visitCount part first) :
    originalSample ops banks first second part i lane = qkSample ops banks part i lane := by
  have hq := first_index part first i lane h
  have ht := first_index part (first+second) i lane (Nat.lt_of_lt_of_le h (count_le part first second))
  simp [originalSample, qkSample, hq, ht]

theorem sample_v {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (first second : Nat) (part : Part) (i : Nat) (lane : Lane)
    (lo : visitCount part first ≤ i) (hi : i < visitCount part (first+second)) :
    originalSample ops banks first second part i lane = vSample ops banks first part i lane := by
  obtain ⟨hnot,ho,_⟩ := second_index part first second i lane lo hi
  have ht := first_index part (first+second) i lane hi
  simp [originalSample, vSample, ht, hnot, ho]

def originalStream {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (first second : Nat) (part : Part) (lane : Lane) : List Scalar :=
  (List.range (visitCount part (first+second))).map (fun i =>
    originalSample ops banks first second part i lane)

def alignedStream {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (first second : Nat) (part : Part) (lane : Lane) : List Scalar :=
  let split := visitCount part first
  let total := visitCount part (first+second)
  (List.range split).map (fun i => qkSample ops banks part i lane) ++
    (List.range' split (total-split)).map (fun i => vSample ops banks first part i lane)

theorem streams_equal {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (first second : Nat) (part : Part) (lane : Lane) :
    originalStream ops banks first second part lane = alignedStream ops banks first second part lane := by
  let split := visitCount part first
  let total := visitCount part (first+second)
  have h := count_le part first second
  have hq : (List.range split).map (fun i => originalSample ops banks first second part i lane) =
      (List.range split).map (fun i => qkSample ops banks part i lane) := by
    apply List.map_congr_left
    intro i hi
    exact sample_qk ops banks first second part i lane (List.mem_range.mp hi)
  have hv : (List.range' split (total-split)).map (fun i => originalSample ops banks first second part i lane) =
      (List.range' split (total-split)).map (fun i => vSample ops banks first part i lane) := by
    apply List.map_congr_left
    intro i hi
    obtain ⟨j,hj,he⟩ := List.mem_range'.mp hi
    have hlo : split ≤ i := by omega
    have hhi : i < total := by omega
    exact sample_v ops banks first second part i lane hlo hhi
  change (List.range total).map (fun i => originalSample ops banks first second part i lane) =
    (List.range split).map (fun i => qkSample ops banks part i lane) ++
    (List.range' split (total-split)).map (fun i => vSample ops banks first part i lane)
  rw [range_split total split h, List.map_append, hq, hv]

def originalPartial {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (first second : Nat) (part : Part) : Scalar :=
  ops.reduce fun lane => (originalStream ops banks first second part lane).foldl ops.maximum ops.zero

def alignedPartial {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (first second : Nat) (part : Part) : Scalar :=
  ops.reduce fun lane => (alignedStream ops banks first second part lane).foldl ops.maximum ops.zero

theorem partial_equal {Scalar : Type} (ops : Ops Scalar) (banks : Banks Scalar)
    (first second : Nat) (part : Part) :
    alignedPartial ops banks first second part = originalPartial ops banks first second part := by
  simp only [alignedPartial, originalPartial, ← streams_equal]

#axiomsClean visited_iff
#axiomsClean visits_once
#axiomsClean sample_qk
#axiomsClean sample_v
#axiomsClean streams_equal
#axiomsClean partial_equal
end VeriTile.Bench.Models.NanoGPT.Track1.QKVAmax
