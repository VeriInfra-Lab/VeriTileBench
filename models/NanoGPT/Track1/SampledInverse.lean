import models.NanoGPT.Track1.SampledDensify
import models.NanoGPT.Track1.SampledGatherVector
import models.NanoGPT.Track1.NgramSlots
import Mathlib.Data.List.Sort

/-! A gather CTA owns either one FP8 row tile or one inverse-map interval.
Each inverse CTA clears precisely its interval, then publishes the ranks
of candidates within it. The host's ascending enumeration lets it find
every such candidate by reading at most 256 entries after a lower bound. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.SampledInverse
open SampledRows (Count ClassID)

structure Candidates (count : Count) where
  ids : Fin count.rows → ClassID
  sorted : StrictMono ids

def Candidates.ofBitmap (c : Count) (mark : ClassID → Bool) (size : (Densify.flatNonzero mark).length = c.rows) :
    Candidates c where
  ids := Densify.stagedClasses c mark size
  sorted := by
    intro i j less
    have sorted : (Densify.flatNonzero mark).SortedLT :=
      (List.Pairwise.sublist List.filter_sublist (List.sortedLT_finRange 50304).pairwise).sortedLT
    change (Densify.flatNonzero mark).get (i.cast size.symm) <
      (Densify.flatNonzero mark).get (j.cast size.symm)
    apply sorted.strictMono_get
    exact (Fin.cast_lt_cast _).mpr less

def Candidates.read {c : Count} (s : Candidates c) (i : Nat) : Nat :=
  if live : i < c.rows then (s.ids ⟨i, live⟩).val else 50304

def Candidates.want {c : Count} (s : Candidates c) : NgramSlots.Want where
  size := c.rows
  read := s.read
  ordered i j order live := by
    have low : i < c.rows := Nat.lt_of_le_of_lt order live
    simpa only [Candidates.read, dif_pos low, dif_pos live] using
      (s.sorted.monotone (show (⟨i, low⟩ : Fin c.rows) ≤ ⟨j, live⟩ from order))
  inVocab i live := by
    have bound := (s.ids ⟨i, live⟩).isLt
    simp only [Candidates.read, dif_pos live]
    omega

private theorem spaced {c : Count} (s : Candidates c) (a b : Nat) (order : a ≤ b) (live : b < c.rows) :
    s.read a + (b - a) ≤ s.read b := by
  induction b with
  | zero =>
    have same : a = 0 := by omega
    simp [same]
  | succ b ih =>
    by_cases same : a = b + 1
    · simp [same]
    · have before : b < c.rows := by omega
      have h := ih (by omega) before
      have next : s.read b < s.read (b + 1) := by
        simpa only [Candidates.read, dif_pos before, dif_pos live] using
          (s.sorted (show (⟨b, before⟩ : Fin c.rows) < ⟨b + 1, live⟩ from Nat.lt_succ_self _))
      omega

abbrev Tile := Fin 197
abbrev Lane := Fin 256

def packPID (rowTile colTile : Nat) : Nat := 197 + 6 * rowTile + colTile

theorem gather_pid_exact (rowTile colTile : Nat) (fits : colTile < 6) :
    197 ≤ packPID rowTile colTile ∧
      (packPID rowTile colTile - 197) / 6 = rowTile ∧
      (packPID rowTile colTile - 197) % 6 = colTile := by
  unfold packPID
  omega

def start (t : Tile) : Nat := 256 * t.val
def stop (t : Tile) : Nat := min (start t + 256) 50304
def owner (cls : ClassID) : Tile := ⟨cls.val / 256, by have := cls.isLt; omega⟩

theorem tile_of_class (cls : ClassID) : start (owner cls) ≤ cls.val ∧ cls.val < stop (owner cls) := by
  have := cls.isLt
  simp only [start, stop, owner, Fin.val_mk]
  omega

theorem tiles_disjoint (a b : Tile) (cls : ClassID) (left : start a ≤ cls.val ∧ cls.val < stop a)
    (right : start b ≤ cls.val ∧ cls.val < stop b) : a = b := by
  apply Fin.ext
  simp only [start, stop] at left right
  omega

/-- The PTX clips the start at 50048 before adding the tile width. -/
theorem clipped_stop (t : Tile) : min (start t) 50048 + 256 = stop t := by
  unfold stop
  omega

/-- Exactly one lane clears each valid index and no lane clears an index
outside its CTA's interval. The subsequent publish stores use that interval. -/
theorem clear_owner (t : Tile) (cls : ClassID) :
    (∃ lane : Lane, start t + lane.val = cls.val ∧ cls.val < stop t) ↔ t = owner cls := by
  constructor
  · rintro ⟨lane, address, tail⟩
    exact tiles_disjoint t (owner cls) cls ⟨by omega, tail⟩ (tile_of_class cls)
  · rintro rfl
    rcases tile_of_class cls with ⟨lo, hi⟩
    exact ⟨⟨cls.val - start (owner cls), by unfold stop at hi; omega⟩, Nat.add_sub_of_le lo, hi⟩

theorem clear_lanes_unique (t : Tile) (left right : Lane)
    (same : start t + left.val = start t + right.val) : left = right :=
  Fin.ext (Nat.add_left_cancel same)

def lower {c : Count} (s : Candidates c) (t : Tile) : Nat := NgramSlots.nativeNat s.want (start t)

theorem lower_exact {c : Count} (s : Candidates c) (t : Tile) :
    NgramSlots.IsInsertion s.read (start t) 0 c.rows (lower s t) :=
  NgramSlots.native_insertion s.want (start t)

theorem covers_candidates {c : Count} (s : Candidates c) (t : Tile) (i : Fin c.rows)
    (live : start t ≤ (s.ids i).val ∧ (s.ids i).val < stop t) :
    ∃ lane : Lane, lower s t + lane.val = i.val := by
  let lo := lower s t
  have search := lower_exact s t
  have iRead : s.read i.val = (s.ids i).val := by simp [Candidates.read, i.isLt]
  have upper : lo ≤ i.val := by
    by_contra outside
    have tooSmall := search.2.2.1 i.val (Nat.zero_le _) (Nat.lt_of_not_ge outside)
    rw [iRead] at tooSmall
    exact Nat.not_lt_of_ge live.1 tooSmall
  have seed := search.2.2.2 lo (Nat.le_refl _) (Nat.lt_of_le_of_lt upper i.isLt)
  have diff := spaced s lo i.val upper i.isLt
  have bound : i.val - lo < 256 := by
    rw [iRead] at diff
    simp only [stop] at live
    omega
  exact ⟨⟨i.val - lo, bound⟩, Nat.add_sub_of_le upper⟩

private def laneWriteAt {c : Count} (s : Candidates c) (t : Tile) (bound : Nat)
    (lane : Lane) : Option (Fin c.rows) :=
  let slot := bound + lane.val
  if valid : slot < c.rows then
    let i : Fin c.rows := ⟨slot, valid⟩
    if start t ≤ (s.ids i).val ∧ (s.ids i).val < stop t then some i else none
  else none

def laneWrite {c : Count} (s : Candidates c) (t : Tile) (lane : Lane) : Option (Fin c.rows) :=
  laneWriteAt s t (lower s t) lane

private def inverseWith {c : Count} (s : Candidates c) (rank : Tile → Nat) :
    Densify.PositionMap c := fun cls =>
  let t := owner cls
  ((List.finRange 256).filterMap (laneWriteAt s t (rank t))).find? (fun slot => decide (s.ids slot = cls))

def inverse {c : Count} (s : Candidates c) : Densify.PositionMap c := inverseWith s (lower s)

private theorem inverse_at_selected {c : Count} (s : Candidates c) (i : Fin c.rows) :
    inverse s (s.ids i) = some i := by
  let t := owner (s.ids i)
  have live : start t ≤ (s.ids i).val ∧ (s.ids i).val < stop t := tile_of_class (s.ids i)
  obtain ⟨lane, index⟩ := covers_candidates s t i live
  have laneHit : laneWrite s t lane = some i := by
    dsimp [laneWrite, laneWriteAt]
    rw [index]
    simp only [dif_pos i.isLt, Fin.eta, if_pos live]
  have present : i ∈ (List.finRange 256).filterMap (laneWrite s t) :=
    List.mem_filterMap.mpr ⟨lane, List.mem_finRange lane, laneHit⟩
  change (((List.finRange 256).filterMap (laneWrite s t)).find?
    (fun slot => decide (s.ids slot = s.ids i))) = some i
  generalize split : (((List.finRange 256).filterMap (laneWrite s t)).find?
    (fun slot => decide (s.ids slot = s.ids i))) = result
  cases result with
  | none => exact False.elim (by simpa using (List.find?_eq_none.mp split) i present)
  | some j =>
    have selected := List.find?_some (p := fun slot => decide (s.ids slot = s.ids i)) split
    have same : s.ids j = s.ids i := of_decide_eq_true selected
    rw [s.sorted.injective same]

theorem inverse_exact {c : Count} (s : Candidates c) : inverse s = Densify.inverseMap s.ids := by
  funext cls
  by_cases mark : ∃ i, s.ids i = cls
  · obtain ⟨i, rfl⟩ := mark
    rw [inverse_at_selected, Densify.inverse_selected _ s.sorted.injective]
  · have absent : ∀ i, cls ≠ s.ids i := fun i h => mark ⟨i, h.symm⟩
    rw [Densify.inverse_missing _ _ absent]
    apply List.find?_eq_none.mpr
    intro slot _
    simp [Ne.symm (absent slot)]

/-- The coarse search compares the last candidate of each batch of
sixteen. The same guarded midpoint updates run until the two ends meet. -/
private def afterSearch (read : Nat → Nat) (key : Nat) : Nat → Nat × Nat → Nat × Nat
  | 0, range => range
  | fuel + 1, (lo, hi) => afterSearch read key fuel (NgramSlots.shrink read key lo hi)

private theorem search_finishes (read : Nat → Nat) (key fuel lo hi rank : Nat)
    (short : hi - lo < 2 ^ fuel) (inside : NgramSlots.IsInsertion read key lo hi rank) :
    afterSearch read key fuel (lo, hi) = (rank, rank) := by
  induction fuel generalizing lo hi with
  | zero =>
    have low := inside.1
    have high := inside.2.1
    simp only [Nat.pow_zero] at short
    exact Prod.ext (by dsimp [afterSearch]; omega) (by dsimp [afterSearch]; omega)
  | succ fuel ih =>
    rw [afterSearch]
    let next := NgramSlots.shrink read key lo hi
    change afterSearch read key fuel (next.1, next.2) = (rank, rank)
    apply ih
    · have step : next.2 - next.1 ≤ (hi - lo) / 2 := NgramSlots.shrink_length read key lo hi
      rw [Nat.pow_succ] at short
      omega
    · exact NgramSlots.shrink_preserves read key lo hi rank inside

def blockEnd {c : Count} (s : Candidates c) (block : Nat) : Nat :=
  s.read ((block + 1) * 16 - 1)

private theorem block_count (c : Count) : c.rows / 16 * 16 = c.rows := by
  cases c <;> decide

theorem coarse_bounds {c : Count} (s : Candidates c) (t : Tile) :
    NgramSlots.IsInsertion (blockEnd s) (start t) 0 (c.rows / 16) (lower s t / 16) := by
  have h := lower_exact s t
  have all := block_count c
  refine ⟨Nat.zero_le _, Nat.div_le_div_right h.2.1, ?_, ?_⟩
  · intro i _ before
    exact h.2.2.1 ((i + 1) * 16 - 1) (Nat.zero_le _) (by omega)
  · intro i after live
    exact h.2.2.2 ((i + 1) * 16 - 1) (by omega) (by omega)

def coarseFuel : Count → Nat
  | .first | .second => 10
  | .third => 11

private def coarse {c : Count} (s : Candidates c) (t : Tile) : Nat :=
  (afterSearch (blockEnd s) (start t) (coarseFuel c) (0, c.rows / 16)).1

theorem coarse_finishes {c : Count} (s : Candidates c) (t : Tile) :
    afterSearch (blockEnd s) (start t) (coarseFuel c) (0, c.rows / 16) =
      (lower s t / 16, lower s t / 16) := by
  apply search_finishes _ _ _ _ _ _ ?_ (coarse_bounds s t)
  cases c <;> decide

private def searchLane {c : Count} (s : Candidates c) (block lane : Nat) : Nat :=
  let i := block * 16 + lane
  if lane < 16 ∧ i < c.rows then s.read i else 50304

def blockedLower {c : Count} (s : Candidates c) (t : Tile) : Nat :=
  let block := coarse s t
  let warp := (List.range 32).map fun lane => if searchLane s block lane < start t then 1 else 0
  block * 16 + warp.sum

private theorem lane_is_below {c : Count} (s : Candidates c) (t : Tile) (lane : Nat) :
    searchLane s (lower s t / 16) lane < start t ↔ lane < lower s t % 16 := by
  have h := lower_exact s t
  have ht := t.isLt
  have hs : start t < 50304 := by simp only [start]; omega
  unfold searchLane
  by_cases valid : lane < 16 ∧ lower s t / 16 * 16 + lane < c.rows
  · rw [if_pos valid]
    constructor
    · intro hit
      by_contra tall
      have outside : lower s t ≤ lower s t / 16 * 16 + lane := by omega
      exact Nat.not_lt_of_ge (h.2.2.2 _ outside valid.2) hit
    · intro hit
      exact h.2.2.1 _ (Nat.zero_le _) (by omega)
  · rw [if_neg valid]
    have out : ¬ lane < lower s t % 16 := by
      intro hit
      apply valid
      exact ⟨by omega, by have := h.2.1; omega⟩
    exact ⟨fun bad => False.elim (by omega), fun bad => False.elim (out bad)⟩

private theorem sum_prefix (size rank : Nat) :
    ((List.range size).map fun lane => if lane < rank then 1 else 0).sum = min size rank := by
  induction size with
  | zero => simp
  | succ n ih =>
    simp only [List.range_succ, List.map_append, List.map_cons, List.map_nil, List.sum_append,
      List.sum_cons, List.sum_nil, Nat.add_zero, ih]
    by_cases hit : n < rank
    · simp only [if_pos hit, min_eq_left (Nat.le_of_lt hit), min_eq_left (Nat.succ_le_of_lt hit)]
    · have le : rank ≤ n := by omega
      simp only [if_neg hit, min_eq_right le, min_eq_right (Nat.le_succ_of_le le), Nat.add_zero]

theorem blocked_lower_exact {c : Count} (s : Candidates c) (t : Tile) :
    blockedLower s t = lower s t := by
  unfold blockedLower coarse
  rw [coarse_finishes]
  dsimp only
  simp only [lane_is_below, sum_prefix, min_eq_right (by omega : lower s t % 16 ≤ 32)]
  omega

theorem blocked_rank_bounds {c : Count} (s : Candidates c) (t : Tile) :
    blockedLower s t ≤ c.rows ∧ blockedLower s t + 255 < 2^31 := by
  rw [blocked_lower_exact]
  have max := (lower_exact s t).2.1
  exact ⟨max, by cases c <;> simp only [Count.rows] at max <;> omega⟩

def blockedInverse {c : Count} (s : Candidates c) : Densify.PositionMap c :=
  inverseWith s (blockedLower s)

theorem blocked_inverse_exact {c : Count} (s : Candidates c) :
    blockedInverse s = Densify.inverseMap s.ids := by
  unfold blockedInverse
  have rank : blockedLower s = lower s := funext (blocked_lower_exact s)
  rw [rank]
  exact inverse_exact s

#axiomsClean coarse_finishes
#axiomsClean blocked_lower_exact
#axiomsClean blocked_rank_bounds
#axiomsClean blocked_inverse_exact

abbrev Build := (c : Count) → Candidates c → Densify.PositionMap c
def staged : Build := fun _ s => Densify.inverseMap s.ids
def fused : Build := fun _ s => blockedInverse s

abbrev Prepared (c : Count) :=
  (SampledRows.Point c → (Nat × UInt8) × (Nat × UInt8)) × Densify.PositionMap c

abbrev Gather := Bool → (c : Count) → (Nat → UInt8) → Candidates c → Prepared c

def referenceGather : Gather := fun _ c bank candidates =>
  (fun p => SampledRows.reference c bank candidates.ids p.row p.column, staged c candidates)

def fusedGather : Gather := fun hopper c bank candidates =>
  if hopper then (SampledRows.Vector.observe bank candidates.ids, fused c candidates)
  else referenceGather false c bank candidates

theorem gather_equal : fusedGather = referenceGather := by
  funext hopper c bank candidates
  cases hopper with
  | false => rfl
  | true =>
    refine Prod.ext (funext fun p => SampledRows.Vector.observe_equal bank candidates.ids p) ?_
    exact blocked_inverse_exact candidates

#axiomsClean gather_equal

theorem maps_equal : fused = staged := by
  funext c s
  exact blocked_inverse_exact s

theorem encoded_equal {c : Count} (s : Candidates c) (cls : ClassID) :
    Densify.encodedPosition (inverse s cls) = Densify.encodedPosition (Densify.inverseMap s.ids cls) := by
  rw [inverse_exact]

#axiomsClean Candidates.ofBitmap
#axiomsClean gather_pid_exact
#axiomsClean tiles_disjoint
#axiomsClean clear_owner
#axiomsClean covers_candidates
#axiomsClean maps_equal
#axiomsClean encoded_equal
end VeriTile.Bench.Models.NanoGPT.Track1.SampledInverse
