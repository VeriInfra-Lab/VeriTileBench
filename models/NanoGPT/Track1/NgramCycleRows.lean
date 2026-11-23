import models.NanoGPT.Track1.NgramDirectoryPacket
import Init.Data.List.Sort.Lemmas

/-! An ordered cycle contributes the sorted union of the per-batch row
words. The index and live tensor identity are checked together before
the closed-interval search is selected. -/

namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramCycleRows
open FineWeb NgramSlots

def collapse : List Nat → List Nat
  | [] => []
  | item :: tail =>
    match collapse tail with
    | [] => [item]
    | first :: rest => if item = first then first :: rest else item :: first :: rest

theorem mem_collapse (row : Nat) (rows : List Nat) : row ∈ collapse rows ↔ row ∈ rows := by
  induction rows with
  | nil => rfl
  | cons first tail ih =>
    rw [collapse]
    cases h : collapse tail with
    | nil =>
      rw [h] at ih
      simp [← ih]
    | cons next rest =>
      rw [h] at ih
      by_cases same : first = next <;> simp [same, ← ih]

theorem strict_collapse (rows : List Nat) (ordered : rows.Pairwise (· ≤ ·)) :
    (collapse rows).Pairwise (· < ·) := by
  induction rows with
  | nil => exact List.Pairwise.nil
  | cons first tail ih =>
    obtain ⟨order, tailSorted⟩ := List.pairwise_cons.mp ordered
    have strict := ih tailSorted
    rw [collapse]
    cases h : collapse tail with
    | nil => simp
    | cons next remaining =>
      rw [h] at strict
      by_cases same : first = next
      · simp only [if_pos same]
        exact strict
      · simp only [if_neg same]
        refine List.pairwise_cons.mpr ⟨?_, strict⟩
        have head := order next ((mem_collapse next tail).mp (by simp [h]))
        have below : first < next := by omega
        intro row found
        rcases List.mem_cons.mp found with rfl | later
        · exact below
        · exact Nat.lt_trans below ((List.pairwise_cons.mp strict).1 row later)

def collect (batches : List TokenBatch) : List Nat :=
  batches.flatMap fun batch => batch.ngramRows.toList.map UInt32.toNat

def cycleRows (batches : List TokenBatch) : List Nat :=
  collapse ((collect batches).mergeSort (fun a b => decide (a ≤ b)))

theorem collapse_length_le (rows : List Nat) : (collapse rows).length ≤ rows.length := by
  induction rows with
  | nil => simp [collapse]
  | cons first tail ih =>
    cases h : collapse tail with
    | nil => simp [collapse, h]
    | cons next remaining =>
      rw [h] at ih
      by_cases same : first = next <;> simp [collapse, h, same] <;>
        simp only [List.length_cons] at ih <;> omega

private theorem collect_length_le (batches : List TokenBatch) (maxTokens : Nat)
    (hashed : ∀ batch ∈ batches, batch.hashedRows)
    (bounded : ∀ batch ∈ batches, batch.inputs.size ≤ maxTokens) :
    (collect batches).length ≤ batches.length * (2 * maxTokens) := by
  induction batches with
  | nil => simp [collect]
  | cons head tail ih =>
    have headSize := hash_rows_size head.inputs head.ngramRows (hashed _ (by simp))
    have headBound := bounded head (by simp)
    have remaining := ih (fun b h => hashed b (List.mem_cons_of_mem _ h))
      (fun b h => bounded b (List.mem_cons_of_mem _ h))
    simp only [collect, List.flatMap_cons, List.length_append, List.length_map, Array.length_toList,
      List.length_cons, Nat.add_mul, Nat.one_mul] at *
    omega

theorem cycle_rows_length_le (batches : List TokenBatch) (maxTokens : Nat)
    (hashed : ∀ batch ∈ batches, batch.hashedRows)
    (bounded : ∀ batch ∈ batches, batch.inputs.size ≤ maxTokens) :
    (cycleRows batches).length ≤ batches.length * (2 * maxTokens) := by
  unfold cycleRows
  exact Nat.le_trans (collapse_length_le _) (by rw [List.length_mergeSort]; exact collect_length_le _ _ hashed bounded)

theorem mem_cycle_rows (batches : List TokenBatch) (row : Nat) :
    row ∈ cycleRows batches ↔ ∃ batch ∈ batches, ∃ word ∈ batch.ngramRows, word.toNat = row := by
  simp only [cycleRows, mem_collapse, List.mem_mergeSort, collect, List.mem_flatMap,
    List.mem_map, Array.mem_toList_iff]

theorem scheduled_row_is_wanted (batches : List TokenBatch) (batch : TokenBatch)
    (live : batch ∈ batches) (word : UInt32) (row : word ∈ batch.ngramRows) :
    word.toNat ∈ cycleRows batches :=
  (mem_cycle_rows batches _).mpr ⟨batch, live, word, row, rfl⟩

theorem cycle_rows_valid (batches : List TokenBatch)
    (provenance : ∀ batch ∈ batches, batch.hashedRows) (row : Nat) (member : row ∈ cycleRows batches) :
    row < 84602880 := by
  obtain ⟨batch, live, word, present, rfl⟩ := (mem_cycle_rows batches row).mp member
  exact FineWeb.hash_rows_range batch.inputs batch.ngramRows (provenance batch live) word present

theorem cycle_rows_strict (batches : List TokenBatch) : (cycleRows batches).Pairwise (· < ·) := by
  apply strict_collapse
  have sorted := List.pairwise_mergeSort (le := fun a b : Nat => decide (a ≤ b))
    (by intro a b c ab bc; simp only [decide_eq_true_eq] at *; omega)
    (by intro a b; simp only [Bool.or_eq_true, decide_eq_true_eq]; exact Nat.le_total a b)
    (collect batches)
  exact sorted.imp (fun h => of_decide_eq_true h)

def want (batches : List TokenBatch) (provenance : ∀ batch ∈ batches, batch.hashedRows) : Want where
  size := (cycleRows batches).length
  read i := (cycleRows batches)[i]?.getD 0
  inVocab i inside := by
    have readable := List.getElem_mem inside
    simpa only [List.getElem?_eq_getElem inside, Option.getD_some] using
      cycle_rows_valid batches provenance _ readable
  ordered i j order inside := by
    have first := Nat.lt_of_le_of_lt order inside
    by_cases same : i = j
    · simp [same]
    · rw [List.getElem?_eq_getElem first, List.getElem?_eq_getElem inside]
      simp only [Option.getD_some]
      exact Nat.le_of_lt ((List.pairwise_iff_getElem.mp (cycle_rows_strict batches))
        i j first inside (Nat.lt_of_le_of_ne order same))

structure Entry where
  object : Nat
  batch : TokenBatch

structure Cycle where
  entries : List Entry
  hashed : ∀ entry ∈ entries, entry.batch.hashedRows

def capture (pack : TrainingPack) (hash : RowHash) :
    List (Nat × Nat) → ScheduledStream → Option (List Entry × ScheduledStream)
  | [], stream => some ([], stream)
  | (step, object) :: rest, stream => do
    let (batch, stream) ← stream.access false step pack hash
    let (later, after) ← capture pack hash rest stream
    pure (⟨object, batch⟩ :: later, after)

theorem capture_hashed (pack : TrainingPack) (hash : RowHash) (producer : pack.preservesHash hash)
    (requests : List (Nat × Nat)) (stream : ScheduledStream) (prior : HashedCache stream.ahead)
    (entries : List Entry) (after : ScheduledStream) (found : capture pack hash requests stream = some (entries, after)) :
    (∀ entry ∈ entries, entry.batch.hashedRows) ∧ HashedCache after.ahead := by
  induction requests generalizing stream entries after with
  | nil =>
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj found)
    exact ⟨by simp, prior⟩
  | cons request rest ih =>
    obtain ⟨step, object⟩ := request
    simp only [capture, Option.bind_eq_bind, Option.bind_eq_some_iff] at found
    obtain ⟨⟨batch, current⟩, read, ⟨later, final⟩, reads, rfl, rfl⟩ := found
    have head := access_rows_hashed stream false step pack hash producer prior batch current read
    have tail := ih current head.2 later final reads
    refine ⟨?_, tail.2⟩
    intro entry atCycle
    rcases List.mem_cons.mp atCycle with rfl | other
    · exact head.1
    · exact tail.1 entry other

theorem capture_sizes (pack : TrainingPack) (hash : RowHash) (sized : pack.preservesSize)
    (requests : List (Nat × Nat)) (stream : ScheduledStream) (maxTokens : Nat)
    (prior : stream.Bounded maxTokens) (entries : List Entry) (after : ScheduledStream)
    (found : capture pack hash requests stream = some (entries, after)) :
    entries.length = requests.length ∧ (∀ entry ∈ entries, entry.batch.inputs.size ≤ maxTokens) ∧
      after.Bounded maxTokens := by
  induction requests generalizing stream entries after with
  | nil =>
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj found)
    exact ⟨rfl, by simp, prior⟩
  | cons request rest ih =>
    obtain ⟨step, object⟩ := request
    simp only [capture, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.some.injEq] at found
    obtain ⟨⟨batch, current⟩, read, ⟨later, final⟩, reads, rfl, rfl⟩ := found
    have head := access_bounded stream false step maxTokens pack hash sized prior batch current read
    obtain ⟨size, bounded, finalBound⟩ := ih current head.2 later final reads
    refine ⟨by simpa only [List.length_cons] using congrArg (· + 1) size, ?_, finalBound⟩
    intro entry member
    rcases List.mem_cons.mp member with rfl | other
    · exact head.1
    · exact bounded entry other

def Cycle.want (cycle : Cycle) : Want :=
  NgramCycleRows.want (cycle.entries.map Entry.batch) (by
    intro batch live
    obtain ⟨entry, member, rfl⟩ := List.mem_map.mp live
    exact cycle.hashed entry member)

theorem Cycle.want_size (cycle : Cycle) (maxTokens : Nat)
    (bounded : ∀ entry ∈ cycle.entries, entry.batch.inputs.size ≤ maxTokens) :
    cycle.want.size ≤ cycle.entries.length * (2 * maxTokens) := by
  have perBatch (P : TokenBatch → Prop) (tracked : ∀ e ∈ cycle.entries, P e.batch)
      (batch : TokenBatch) (live : batch ∈ cycle.entries.map Entry.batch) : P batch := by
    obtain ⟨entry, member, rfl⟩ := List.mem_map.mp live
    exact tracked entry member
  simpa only [Cycle.want, want, List.length_map] using
    cycle_rows_length_le (cycle.entries.map Entry.batch) maxTokens
      (perBatch _ cycle.hashed) (perBatch _ bounded)

def ofCapture (pack : TrainingPack) (hash : RowHash) (producer : pack.preservesHash hash)
    (requests : List (Nat × Nat)) (stream : ScheduledStream) (prior : HashedCache stream.ahead)
    (entries : List Entry) (after : ScheduledStream)
    (found : capture pack hash requests stream = some (entries, after)) : Cycle :=
  ⟨entries, (capture_hashed pack hash producer requests stream prior entries after found).1⟩

def Cycle.select (cycle : Cycle) (live : Nat → Bool) (position object : Nat) : Option TokenBatch := do
  let entry ← cycle.entries[position]?
  if live entry.object && entry.object == object then some entry.batch else none

theorem selected_contains (cycle : Cycle) (live : Nat → Bool) (position object : Nat) (batch : TokenBatch)
    (selected : cycle.select live position object = some batch) (word : UInt32) (member : word ∈ batch.ngramRows) :
    ContainsRow cycle.want word.toNat := by
  simp only [Cycle.select, Option.bind_eq_bind, Option.bind_eq_some_iff] at selected
  obtain ⟨entry, stored, chosen⟩ := selected
  have same : entry.batch = batch := by
    split at chosen
    · exact Option.some.inj chosen
    · contradiction
  have inCycle : batch ∈ cycle.entries.map Entry.batch :=
    List.mem_map.mpr ⟨entry, List.mem_of_getElem? stored, same⟩
  have wanted := scheduled_row_is_wanted (cycle.entries.map Entry.batch) batch inCycle word member
  obtain ⟨i, within, exactRow⟩ := List.mem_iff_getElem.mp wanted
  refine ⟨i, within, ?_⟩
  simp only [Cycle.want, want, List.getElem?_eq_getElem within, Option.getD_some]
  exact exactRow

theorem selected_hashed (cycle : Cycle) (live : Nat → Bool) (position object : Nat) (batch : TokenBatch)
    (selected : cycle.select live position object = some batch) : batch.hashedRows := by
  simp only [Cycle.select, Option.bind_eq_bind, Option.bind_eq_some_iff] at selected
  obtain ⟨entry, stored, chosen⟩ := selected
  split at chosen
  · exact Option.some.inj chosen ▸ cycle.hashed entry (List.mem_of_getElem? stored)
  · contradiction

#axiomsClean cycle_rows_strict

#axiomsClean mem_cycle_rows
#axiomsClean scheduled_row_is_wanted
#axiomsClean cycle_rows_valid
#axiomsClean strict_collapse
#axiomsClean capture_hashed
#axiomsClean capture_sizes
#axiomsClean Cycle.want_size
#axiomsClean selected_contains
#axiomsClean selected_hashed

end VeriTile.Bench.Models.NanoGPT.Track1.NgramCycleRows
