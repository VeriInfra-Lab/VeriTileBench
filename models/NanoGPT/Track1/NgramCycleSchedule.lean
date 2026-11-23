import models.NanoGPT.Track1.SparsePackets

/-! In the timed run the step IDs are consecutive. The prefetch returns
every position through the first update, or through the last step if
there is no update left. Counting in terms of exclusive endpoints avoids
the negative sentinel used by the Python host array. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramCycleSchedule

def throughUpdate (due : Nat → Bool) : List Nat → List Nat
  | [] => []
  | step :: tail => step :: if due step then [] else throughUpdate due tail

theorem through_update_le (due : Nat → Bool) (remaining : List Nat) :
    (throughUpdate due remaining).length ≤ remaining.length := by
  induction remaining with
  | nil => rfl
  | cons first tail ih =>
    cases stop : due first <;> simp [throughUpdate, stop, ih]

theorem through_update_end (due : Nat → Bool) (before after : List Nat) (last : Nat)
    (noneEarlier : ∀ step ∈ before, due step = false) (endUpdate : due last = true) :
    throughUpdate due (before ++ last :: after) = before ++ [last] := by
  induction before with
  | nil => simp [throughUpdate, endUpdate]
  | cons first rest ih =>
    have skip := noneEarlier first (by simp)
    simp only [List.cons_append, throughUpdate, skip, Bool.false_eq_true, if_false, List.cons.injEq, true_and]
    exact ih (fun step member => noneEarlier step (List.mem_cons_of_mem _ member))

private theorem through_update_le_prefix (due : Nat → Bool) (before after : List Nat)
    (hasUpdate : ∃ step ∈ before, due step = true) :
    (throughUpdate due (before ++ after)).length ≤ before.length := by
  induction before with
  | nil => simp at hasUpdate
  | cons first rest ih =>
    cases stop : due first
    · rcases hasUpdate with ⟨step, member, dueNow⟩
      have later : step ∈ rest := (List.mem_cons.mp member).resolve_left (by
        intro h
        rw [h, stop] at dueNow
        contradiction)
      have within := ih ⟨step, later, dueNow⟩
      simpa [throughUpdate, stop] using within
    · simp [throughUpdate, stop]

theorem through_update_none (due : Nat → Bool) (remaining : List Nat)
    (skips : ∀ step ∈ remaining, due step = false) : throughUpdate due remaining = remaining := by
  induction remaining with
  | nil => rfl
  | cons first tail ih =>
    rw [throughUpdate, skips first (by simp)]
    simp only [Bool.false_eq_true, if_false, List.cons.injEq, true_and]
    exact ih (fun step member => skips step (List.mem_cons_of_mem _ member))

def timedCycle (total start : Nat) : List Nat :=
  throughUpdate (SparsePackets.eventDue false) (List.range' start (total - start))

/-- Lower-bound search in the sorted update positions returns the first
update in the remaining interval. The Python range includes that endpoint. -/
theorem timed_cycle_host_range (total start : Nat) :
    let next := (List.range' start (total - start)).find? (SparsePackets.eventDue false)
    timedCycle total start = List.range' start (((next.map (· + 1)).getD total) - start) := by
  let remaining := List.range' start (total - start)
  change throughUpdate (SparsePackets.eventDue false) remaining =
    List.range' start ((((remaining.find? (SparsePackets.eventDue false)).map (· + 1)).getD total) - start)
  cases found : remaining.find? (SparsePackets.eventDue false) with
  | none =>
    have skips := List.find?_eq_none.mp found
    simp only [Option.map_none, Option.getD_none]
    exact through_update_none _ remaining (fun x hx => Bool.eq_false_iff.mpr (skips x hx))
  | some last =>
    obtain ⟨dueNow, before, after, span, skips⟩ := List.find?_eq_some_iff_append.mp found
    have splitRange : List.range' start (total - start) = before ++ last :: after := span
    obtain ⟨n, within, beforeEq, restEq⟩ := List.range'_eq_append_iff.mp splitRange
    have head := congrArg List.head? restEq
    simp only [List.head?_cons, List.head?_range', Nat.mul_one] at head
    have position : last = start + n := by
      split at head
      · contradiction
      · exact Option.some.inj head
    have len : last + 1 - start = n + 1 := by omega
    simp only [Option.map_some, Option.getD_some]
    rw [span, through_update_end _ _ _ _ (fun x hx => by simpa using skips x hx) dueNow, len]
    symm
    apply List.range'_eq_append_iff.mpr
    refine ⟨n, by omega, beforeEq, ?_⟩
    simp only [position, Nat.mul_one, Nat.add_sub_cancel_left]
    rfl

theorem every_fourth_is_update (step : Nat) (everyFourth : step % 4 = 3) :
    SparsePackets.eventDue false step = true := by
  have odd : step % 2 = 1 := by omega
  simp [SparsePackets.eventDue, odd, everyFourth]

theorem timed_cycle_le_four (total start : Nat) : (timedCycle total start).length ≤ 4 := by
  by_cases enough : 4 ≤ total - start
  · let before := List.range' start 4
    let after := List.range' (start + 4) (total - start - 4)
    have head : List.range' start (total - start) = before ++ after := by
      apply List.range'_eq_append_iff.mpr
      exact ⟨4, enough, rfl, by simp only [after, Nat.mul_one]⟩
    let event := start + (3 - start % 4)
    have within : event ∈ before := by
      rw [List.mem_range'_1]
      dsimp [event]
      omega
    have update : SparsePackets.eventDue false event = true := every_fourth_is_update _ (by
      dsimp [event]
      omega)
    simpa only [timedCycle, head, before, List.length_range'] using
      through_update_le_prefix _ before after ⟨event, within, update⟩
  · have within := through_update_le (SparsePackets.eventDue false) (List.range' start (total - start))
    rw [List.length_range'] at within
    exact Nat.le_trans within (by omega)

def timedRequests (total start : Nat) (objectAt : Nat → Nat) : List (Nat × Nat) :=
  (timedCycle total start).map fun step => (step, objectAt step)

theorem timed_requests_le_four (total start : Nat) (objectAt : Nat → Nat) :
    (timedRequests total start objectAt).length ≤ 4 := by
  simpa only [timedRequests, List.length_map] using timed_cycle_le_four total start

/-- The trainer allocates two cache rows per rank-local validation token. -/
def defaultCacheRows (world : Nat) : Nat := 2 * ((4 * 64 * 1024 * 8) / world)

theorem hopper_cache_size : defaultCacheRows 8 = 524288 := rfl

#axiomsClean through_update_end
#axiomsClean through_update_none
#axiomsClean timed_cycle_host_range
#axiomsClean timed_requests_le_four
end VeriTile.Bench.Models.NanoGPT.Track1.NgramCycleSchedule
