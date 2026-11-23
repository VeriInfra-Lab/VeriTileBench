import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

/-! The sparse Adam second moments are brought up to date before the next
cycle. A warp can share each beta2 load while observing the original FP32
multiplication order separately for every requested row. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MomentReplay
open VeriTile.Triton FP

abbrev F32 := Value .fp32

structure Row where
  moment : F32
  stamp : Nat
  deriving DecidableEq

abbrev Memory := Nat → Row

structure Input where
  want : Array Int32
  lo : Nat
  hi : Nat
  first : Int32
  shardRows : Nat
  event : Nat
  beta : Nat → F32
  old : Memory

def rowAt (s : Input) (slot : Nat) : Option Nat := do
  if s.lo + slot ≥ s.hi then none else
    let id ← s.want[s.lo + slot]?
    let row := (id - s.first).toInt
    if 0 ≤ row ∧ row < s.shardRows then some row.toNat else none

private theorem rowAt_source (s : Input) (slot row : Nat) (found : rowAt s slot = some row) :
    ∃ id : Int32, s.want[s.lo + slot]? = some id ∧ (id - s.first).toInt = row := by
  unfold rowAt at found
  split at found
  · contradiction
  cases h : s.want[s.lo + slot]? with
  | none => simp [h] at found
  | some id =>
    rw [h] at found
    let offset := (id - s.first).toInt
    change (if 0 ≤ offset ∧ offset < s.shardRows then some offset.toNat else none) = some row at found
    split at found
    · rename_i bound
      have same := Option.some.inj found
      exact ⟨id, rfl, by change offset = (row : Int); omega⟩
    · contradiction

theorem slots_disjoint (s : Input) (a b row : Nat)
    (unique : a ≠ b → s.want[s.lo + a]? ≠ s.want[s.lo + b]?)
    (first : rowAt s a = some row) (second : rowAt s b = some row) : a = b := by
  by_cases different : a = b
  · exact different
  obtain ⟨x, hx, rowX⟩ := rowAt_source s a row first
  obtain ⟨y, hy, rowY⟩ := rowAt_source s b row second
  have localEqual : x - s.first = y - s.first := Int32.toInt.inj (rowX.trans rowY.symm)
  have bits := congrArg Int32.toBitVec localEqual
  simp only [Int32.toBitVec_sub] at bits
  have ids : x = y := Int32.toBitVec_inj.mp (by
    simpa only [BitVec.sub_add_cancel] using congrArg (· + s.first.toBitVec) bits)
  rw [ids] at hx
  exact (unique different (hx.trans hy.symm)).elim

abbrev Lane := Fin 32

def tileStamp (s : Input) (tile : Nat) (lane : Lane) : Nat :=
  match rowAt s (tile * 32 + lane.val) with
  | none => s.event
  | some row => (s.old row).stamp

def firstStamp (s : Input) (tile : Nat) : Nat :=
  (List.finRange 32).foldr (fun lane acc => min (min (tileStamp s tile lane) s.event) acc) s.event

private theorem minFold_le_end {α : Type} (f : α → Nat) (lanes : List α) (event : Nat) :
    lanes.foldr (fun lane acc => min (f lane) acc) event ≤ event := by
  induction lanes with
  | nil => simp
  | cons lane rest ih =>
    exact Nat.le_trans (Nat.min_le_right _ _) ih

private theorem minFold_le_member {α : Type} (f : α → Nat) (lanes : List α) (event : Nat)
    (lane : α) (member : lane ∈ lanes) :
    lanes.foldr (fun lane acc => min (f lane) acc) event ≤ f lane := by
  induction lanes with
  | nil => simp at member
  | cons head rest ih =>
    rcases List.mem_cons.mp member with rfl | h
    · exact Nat.min_le_left _ _
    · exact Nat.le_trans (Nat.min_le_right _ _) (ih h)

theorem firstStamp_le_event (s : Input) (tile : Nat) : firstStamp s tile ≤ s.event :=
  minFold_le_end (fun lane => min (tileStamp s tile lane) s.event) _ _

theorem firstStamp_le_lane (s : Input) (tile : Nat) (lane : Lane) :
    firstStamp s tile ≤ tileStamp s tile lane := by
  exact Nat.le_trans (minFold_le_member (fun i => min (tileStamp s tile i) s.event) _ _ lane (by simp))
    (Nat.min_le_left _ _)

def sequential (beta : Nat → F32) (last : Nat) (value : F32) : Nat → F32
  | 0 => value
  | i + 1 => FP.mul Config.ieeeValue (sequential beta last value i) (beta (last + (i + 1)))

/-- The two FP32 history words of the pinned n-gram schedule. -/
def betaWord (period4 : Bool) : F32 :=
  Value.ofBits .fp32 (if period4 then 0x3f670a3d else 0x3f733333)

def PositiveHistory (beta : Nat → F32) (event : Nat) : Prop :=
  ∀ i, 0 < i → i ≤ event → ∃ period4, beta i = betaWord period4

def holdsHistory (beta : Nat → F32) (event : Nat) : Bool :=
  (List.range event).all fun i => decide (beta (i + 1) = betaWord false ∨ beta (i + 1) = betaWord true)

theorem holds_history_iff (beta : Nat → F32) (event : Nat) :
    holdsHistory beta event = true ↔ PositiveHistory beta event := by
  simp only [holdsHistory, List.all_eq_true, List.mem_range, decide_eq_true_eq]
  constructor
  · intro found i positive bound
    have h := found (i - 1) (by omega)
    rw [show i - 1 + 1 = i by omega] at h
    exact h.elim (fun ok => ⟨false, ok⟩) (fun ok => ⟨true, ok⟩)
  · intro found i bound
    obtain ⟨mode, word⟩ := found (i + 1) (by omega) (by omega)
    cases mode
    · exact Or.inl word
    · exact Or.inr word

private theorem zero_mul_beta (period4 : Bool) :
    FP.mul Config.ieeeValue (Value.zero .fp32) (betaWord period4) = Value.zero .fp32 := by
  cases period4 <;> decide +kernel

theorem sequential_zero (beta : Nat → F32) (last count event : Nat)
    (history : PositiveHistory beta event) (bound : last + count ≤ event) :
    sequential beta last (Value.zero .fp32) count = Value.zero .fp32 := by
  induction count with
  | zero => rfl
  | succ k ih =>
    obtain ⟨period4, word⟩ := history (last + (k + 1)) (by omega) bound
    rw [sequential, ih (by omega), word, zero_mul_beta]

def skipZero (beta : Nat → F32) (last : Nat) (value : F32) (event : Nat) : F32 :=
  if value = Value.zero .fp32 then value else sequential beta last value (event - last)

theorem skip_zero_exact (beta : Nat → F32) (last event : Nat) (value : F32)
    (history : PositiveHistory beta event) :
    skipZero beta last value event = sequential beta last value (event - last) := by
  by_cases zero : value = Value.zero .fp32
  · subst value
    by_cases inside : last ≤ event
    · have bound : last + (event - last) ≤ event := by omega
      simp [skipZero, sequential_zero beta last (event - last) event history bound]
    · simp [skipZero, sequential, show event - last = 0 by omega]
  · simp [skipZero, zero]

def masked (beta : Nat → F32) (last start : Nat) (value : F32) : Nat → F32
  | 0 => value
  | i + 1 =>
    let previous := masked beta last start value i
    let e := start + (i + 1)
    if last < e then FP.mul Config.ieeeValue previous (beta e) else previous

theorem masked_exact (beta : Nat → F32) (last start : Nat) (value : F32)
    (started : start ≤ last) (count : Nat) :
    masked beta last start value count = sequential beta last value (start + count - last) := by
  induction count with
  | zero =>
    simp [masked, Nat.sub_eq_zero_of_le started, sequential]
  | succ i ih =>
    by_cases active : last < start + (i + 1)
    · have next : start + (i + 1) - last = (start + i - last) + 1 := by omega
      have idx : last + ((start + i - last) + 1) = start + (i + 1) := by omega
      simp only [masked, if_pos active, next, sequential, ih, idx]
    · have same : start + (i + 1) - last = start + i - last := by omega
      simp only [masked, if_neg active, same, ih]

abbrev Write := Nat × Row

def stagedWrite (s : Input) (slot : Nat) : Option Write := do
  let row ← rowAt s slot
  let old := s.old row
  some (row, ⟨sequential s.beta old.stamp old.moment (s.event - old.stamp), s.event⟩)

def groupedWrite (s : Input) (slot : Nat) : Option Write := do
  let row ← rowAt s slot
  let old := s.old row
  let start := firstStamp s (slot / 32)
  let moment := if old.stamp < s.event then
    masked s.beta old.stamp start old.moment (s.event - start) else old.moment
  some (row, ⟨moment, if old.stamp = s.event then old.stamp else s.event⟩)

theorem writes_equal (s : Input) (slot : Nat) : groupedWrite s slot = stagedWrite s slot := by
  have coordinates : slot / 32 * 32 + slot % 32 = slot := by omega
  let lane : Lane := ⟨slot % 32, by omega⟩
  have first := firstStamp_le_lane s (slot / 32) lane
  have aligned : slot / 32 * 32 + lane.val = slot := coordinates
  unfold tileStamp at first
  rw [aligned] at first
  unfold groupedWrite stagedWrite
  cases h : rowAt s slot with
  | none => rfl
  | some row =>
    simp only [h] at first
    have enough := firstStamp_le_event s (slot / 32)
    have finish : firstStamp s (slot / 32) + (s.event - firstStamp s (slot / 32)) = s.event :=
      Nat.add_sub_of_le enough
    by_cases due : (s.old row).stamp < s.event
    · have stale : (s.old row).stamp ≠ s.event := by omega
      simp [due, stale, masked_exact _ _ _ _ first, finish]
    · have empty : s.event - (s.old row).stamp = 0 := by omega
      simp [due, empty, sequential]

def applyWrites (emit : Input → Nat → Option Write) (s : Input) : Memory :=
  (List.range (s.hi - s.lo)).foldl (fun table slot =>
    match emit s slot with
    | none => table
    | some (row, value) => fun r => if r = row then value else table r) s.old

abbrev Replay := Input → Memory
def staged : Replay := applyWrites stagedWrite
def grouped : Replay := applyWrites groupedWrite

theorem replay_equal : grouped = staged := by
  unfold grouped staged
  rw [show groupedWrite = stagedWrite by funext s slot; exact writes_equal s slot]

/-- Positive-zero lanes carry `event` into the warp minimum. They retain
their moment word and still publish the new stamp. -/
def ignoreZeroLanes (s : Input) : Input :=
  {s with old := fun row =>
    let before := s.old row
    if before.moment = Value.zero .fp32 then {before with stamp := s.event} else before}

def groupedZeroWrite (s : Input) (slot : Nat) : Option Write :=
  groupedWrite (ignoreZeroLanes s) slot

theorem zero_tile_no_history (s : Input) (tile : Nat)
    (idle : ∀ lane : Lane, ∀ row, rowAt s (tile * 32 + lane.val) = some row →
      (s.old row).moment = Value.zero .fp32 ∨ s.event ≤ (s.old row).stamp) :
    firstStamp (ignoreZeroLanes s) tile = s.event := by
  have same (lane : Lane) : min (tileStamp (ignoreZeroLanes s) tile lane) s.event = s.event := by
    have route : rowAt (ignoreZeroLanes s) (tile * 32 + lane.val) = rowAt s (tile * 32 + lane.val) := rfl
    unfold tileStamp
    rw [route]
    split
    · exact Nat.min_self _
    · rename_i row found
      rcases idle lane row found with zero | current
      · simp [ignoreZeroLanes, zero]
      · by_cases zero : (s.old row).moment = Value.zero .fp32
        · simp [ignoreZeroLanes, zero]
        · simpa only [ignoreZeroLanes, if_neg zero] using Nat.min_eq_right current
  unfold firstStamp
  change (List.finRange 32).foldr
    (fun lane acc => min (min (tileStamp (ignoreZeroLanes s) tile lane) s.event) acc) s.event = s.event
  generalize List.finRange 32 = lanes
  induction lanes with
  | nil => rfl
  | cons lane later ih => rw [List.foldr_cons, same lane, ih, Nat.min_self]

theorem grouped_zero_write_exact (s : Input) (slot : Nat) (history : PositiveHistory s.beta s.event) :
    groupedZeroWrite s slot = stagedWrite s slot := by
  rw [groupedZeroWrite, writes_equal]
  have sameRow : rowAt (ignoreZeroLanes s) slot = rowAt s slot := rfl
  unfold stagedWrite
  rw [sameRow]
  cases found : rowAt s slot with
  | none => rfl
  | some row =>
    by_cases zero : (s.old row).moment = Value.zero .fp32
    · have skipped := skip_zero_exact s.beta (s.old row).stamp s.event (s.old row).moment history
      have unchanged := by simpa only [zero, skipZero, if_pos rfl] using skipped
      simpa [ignoreZeroLanes, zero, sequential] using unchanged
    · simp [ignoreZeroLanes, zero]

def groupedZero : Replay := applyWrites groupedZeroWrite

theorem grouped_zero_exact (s : Input) (history : PositiveHistory s.beta s.event) :
    groupedZero s = staged s := by
  unfold groupedZero staged applyWrites
  simp only [grouped_zero_write_exact s _ history]

/-- Totalize the numerical port for synthetic runtime inputs. In the
installed trainer, the checked history publication protocol supplies only
the two scheduled words, so the kernel can use the specialized branch. -/
def selected : Replay := fun s =>
  if holdsHistory s.beta s.event then groupedZero s else grouped s

theorem selected_equal : selected = staged := by
  funext s
  unfold selected
  split
  · rename_i valid
    exact grouped_zero_exact s ((holds_history_iff ..).mp (by simpa using valid))
  · exact congrFun replay_equal s

theorem unchanged_outside (s : Input) (row : Nat)
    (outside : ∀ slot, slot ∈ List.range (s.hi - s.lo) → rowAt s slot ≠ some row) :
    grouped s row = s.old row := by
  rw [replay_equal]
  have preserve (slots : List Nat) (table : Memory)
      (h : ∀ slot, slot ∈ slots → rowAt s slot ≠ some row) :
      (slots.foldl (fun bank slot =>
        match stagedWrite s slot with
        | none => bank
        | some (r, v) => fun k => if k = r then v else bank k) table) row = table row := by
    induction slots generalizing table with
    | nil => rfl
    | cons slot rest ih =>
      rw [List.foldl_cons, ih _ (fun i member => h i (List.mem_cons_of_mem _ member))]
      unfold stagedWrite
      cases found : rowAt s slot with
      | none => rfl
      | some r =>
        have different : row ≠ r := by
          intro equal
          exact h slot (List.mem_cons_self ..) (equal ▸ found)
        simp [different]
  exact preserve _ _ outside

#axiomsClean firstStamp_le_lane
#axiomsClean slots_disjoint
#axiomsClean masked_exact
#axiomsClean skip_zero_exact
#axiomsClean grouped_zero_exact
#axiomsClean zero_tile_no_history
#axiomsClean writes_equal
#axiomsClean replay_equal
#axiomsClean unchanged_outside
end VeriTile.Bench.Models.NanoGPT.Track1.MomentReplay
