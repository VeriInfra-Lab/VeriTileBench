import VeriTile.Meta.StatementAudit
import Init.Data.UInt.Bitwise

namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramSlots

/-- Characterize the first entry at least as large as the key on [lo, hi). -/
def IsInsertion (read : Nat → Nat) (key lo hi atIndex : Nat) : Prop :=
  lo ≤ atIndex ∧ atIndex ≤ hi ∧
    (∀ i, lo ≤ i → i < atIndex → read i < key) ∧
    (∀ i, atIndex ≤ i → i < hi → key ≤ read i)

theorem insertion_unique (read : Nat → Nat) (key lo hi a b : Nat)
    (ha : IsInsertion read key lo hi a) (hb : IsInsertion read key lo hi b) : a = b := by
  rcases ha with ⟨hla, hah, alt, agt⟩
  rcases hb with ⟨hlb, hbh, blt, bgt⟩
  by_cases hab : a < b
  · have low := blt a hla hab
    have high := agt a (Nat.le_refl _) (by omega)
    omega
  by_cases hba : b < a
  · have low := alt b hlb hba
    have high := bgt b (Nat.le_refl _) (by omega)
    omega
  omega

def lowerKey (row : Nat) := row / 32768 * 32768

theorem row_in_bucket (row : Nat) : lowerKey row ≤ row ∧ row < lowerKey row + 32768 := by
  unfold lowerKey
  omega

theorem bucket_insertion (read : Nat → Nat) (size row a b atIndex : Nat)
    (left : IsInsertion read (lowerKey row) 0 size a)
    (right : IsInsertion read (lowerKey row + 32768) 0 size b)
    (inner : IsInsertion read row a b atIndex) :
    IsInsertion read row 0 size atIndex := by
  have limits := row_in_bucket row
  rcases left with ⟨_, _, leftLow, _⟩
  rcases right with ⟨_, bHi, _, rightHigh⟩
  rcases inner with ⟨lo, hi, before, after⟩
  refine ⟨by omega, by omega, ?_, ?_⟩
  · intro i iz hit
    by_cases beforeBucket : i < a
    · have low := leftLow i iz beforeBucket
      omega
    · exact before i (Nat.le_of_not_gt beforeBucket) hit
  · intro i hit ih
    by_cases afterBucket : b ≤ i
    · have high := rightHigh i afterBucket ih
      omega
    · exact after i hit (Nat.lt_of_not_ge afterBucket)

def shrink (read : Nat → Nat) (key lo hi : Nat) : Nat × Nat :=
  if lo < hi then
    let mid := lo + (hi - lo) / 2
    if read mid < key then (mid + 1, hi) else (lo, mid)
  else (lo, hi)

theorem shrink_preserves (read : Nat → Nat) (key lo hi p : Nat)
    (inside : IsInsertion read key lo hi p) :
    IsInsertion read key (shrink read key lo hi).1 (shrink read key lo hi).2 p := by
  unfold shrink
  split
  · rename_i live
    dsimp only
    let mid := lo + (hi - lo) / 2
    have inRange : lo ≤ mid ∧ mid < hi := by dsimp [mid]; omega
    change IsInsertion read key (if read mid < key then (mid + 1, hi) else (lo, mid)).1
      (if read mid < key then (mid + 1, hi) else (lo, mid)).2 p
    rcases inside with ⟨low, high, before, after⟩
    by_cases s : read mid < key
    · simp only [if_pos s]
      have move : mid < p := by
        by_cases found : mid < p
        · exact found
        · have bad := after mid (Nat.le_of_not_gt found) inRange.2
          omega
      exact ⟨by omega, high, fun i il ih => before i (by omega) ih, after⟩
    · simp only [if_neg s]
      have move : p ≤ mid := by
        by_cases found : p ≤ mid
        · exact found
        · have bad := before mid inRange.1 (Nat.lt_of_not_ge found)
          omega
      exact ⟨low, move, before, fun i il ih => after i il (by omega)⟩
  · exact inside

theorem shrink_length (read : Nat → Nat) (key lo hi : Nat) :
    (shrink read key lo hi).2 - (shrink read key lo hi).1 ≤ (hi - lo) / 2 := by
  unfold shrink
  split
  · dsimp only
    split <;> dsimp <;> omega
  · omega

/-- The emitted midpoint uses one logical shift of an unsigned distance.
The distance fits in i32 and the midpoint is inside the active interval. -/
theorem emitted_midpoint (lo hi : Nat) (valid : lo < hi) (bound : hi - lo < 256) :
    let mid := lo + ((UInt32.ofNat (hi - lo) >>> 1).toNat)
    mid = lo + (hi - lo) / 2 ∧ lo ≤ mid ∧ mid < hi := by
  have fits : hi - lo < UInt32.size := by unfold UInt32.size; omega
  have converted : (UInt32.ofNat (hi - lo)).toNat = hi - lo := by
    change (hi - lo) % 2 ^ 32 = hi - lo
    exact Nat.mod_eq_of_lt fits
  simp only [UInt32.toNat_shiftRight, converted]
  change lo + ((hi - lo) >>> 1) = lo + (hi - lo) / 2 ∧
    lo ≤ lo + ((hi - lo) >>> 1) ∧ lo + ((hi - lo) >>> 1) < hi
  simp only [Nat.shiftRight_eq_div_pow]
  clear converted fits
  exact ⟨True.intro, Nat.le_add_right _ _, by
    change lo + (hi - lo) / 2 < hi
    omega⟩

def locate (read : Nat → Nat) (key : Nat) : Nat → Nat → Nat → Nat
  | 0, lo, _ => lo
  | fuel + 1, lo, hi =>
    let (nextLo, nextHi) := shrink read key lo hi
    locate read key fuel nextLo nextHi

theorem locate_exact (read : Nat → Nat) (key fuel lo hi p : Nat)
    (length : hi - lo < 2 ^ fuel) (inside : IsInsertion read key lo hi p) :
    locate read key fuel lo hi = p := by
  induction fuel generalizing lo hi with
  | zero =>
    change lo = p
    have low := inside.1
    have high := inside.2.1
    simp at length
    omega
  | succ fuel ih =>
    rw [locate]
    let next := shrink read key lo hi
    change locate read key fuel next.1 next.2 = p
    apply ih
    · have shrinkBound : next.2 - next.1 ≤ (hi - lo) / 2 := shrink_length read key lo hi
      rw [Nat.pow_succ] at length
      change next.2 - next.1 < 2 ^ fuel
      omega
    · exact shrink_preserves read key lo hi p inside

/-- The row cache is laid out in the sorted order of the fetched global IDs. -/
structure Want where
  size : Nat
  read : Nat → Nat
  ordered : ∀ i j, i ≤ j → j < size → read i ≤ read j
  inVocab : ∀ i, i < size → read i < 84602880

private theorem insertion_exists (read : Nat → Nat) (key lo length : Nat)
    (ordered : ∀ i j, lo ≤ i → i ≤ j → j < lo + length → read i ≤ read j) :
    ∃ p, IsInsertion read key lo (lo + length) p := by
  induction length generalizing lo with
  | zero =>
    exact ⟨lo, Nat.le_refl _, by omega, fun i hl hr => False.elim (by omega),
      fun i hl hr => False.elim (by omega)⟩
  | succ n ih =>
    by_cases small : read lo < key
    · have ordTail (i j) (first : lo + 1 ≤ i) (order : i ≤ j) (last : j < (lo + 1) + n) :
          read i ≤ read j := ordered i j (by omega) order (by omega)
      obtain ⟨p, fromTail, within, before, after⟩ := ih (lo + 1) ordTail
      refine ⟨p, by omega, by omega, ?_, ?_⟩
      · intro i low high
        by_cases atLo : i = lo
        · simpa [atLo] using small
        · exact before i (by omega) high
      · intro i low high
        exact after i low (by omega)
    · exact ⟨lo, by omega, by omega, fun i a b => False.elim (by omega),
        fun i a b => Nat.le_trans (Nat.le_of_not_gt small) (ordered lo i (Nat.le_refl _) a b)⟩

def nativeNat (want : Want) (row : Nat) : Nat :=
  locate want.read row (want.size.log2 + 1) 0 want.size

theorem native_insertion (want : Want) (row : Nat) :
    IsInsertion want.read row 0 want.size (nativeNat want row) := by
  obtain ⟨p, hp⟩ := insertion_exists want.read row 0 want.size
    (fun i j _ order within => want.ordered i j order (by simpa using within))
  rw [Nat.zero_add] at hp
  have sizeBound : want.size - 0 < 2 ^ (want.size.log2 + 1) := by
    rw [Nat.sub_zero]
    by_cases empty : want.size = 0
    · simp [empty]
    · exact (Nat.log2_lt empty).1 (Nat.lt_succ_self _)
  unfold nativeNat
  rw [locate_exact _ _ _ _ _ p sizeBound hp]
  exact hp

theorem native_nat_within (want : Want) (row : Nat) : nativeNat want row ≤ want.size :=
  (native_insertion want row).2.1

private theorem insertion_mono (read : Nat → Nat) (size left right a b : Nat)
    (keys : left ≤ right) (atLeft : IsInsertion read left 0 size a)
    (atRight : IsInsertion read right 0 size b) : a ≤ b := by
  by_cases ordered : a ≤ b
  · exact ordered
  · have below := atLeft.2.2.1 b (Nat.zero_le _) (by omega)
    have above := atRight.2.2.2 b (Nat.le_refl _) (by have := atLeft.2.1; omega)
    omega

def bucketCount : Nat := 2582

theorem bucket_indices (row : Nat) (valid : row < 84602880) :
    row / 32768 < bucketCount ∧ row / 32768 + 1 ≤ bucketCount := by
  unfold bucketCount
  omega

def directoryStart (rows : Nat) := (rows + 3) / 4 * 4

theorem directory_fits (rows : Nat) :
    rows ≤ directoryStart rows ∧ directoryStart rows ≤ rows + 3 ∧ directoryStart rows % 4 = 0 := by
  unfold directoryStart
  omega

abbrev Lookup := Bool → Want → Int → Nat

structure Request where
  useDirectory : Bool
  want : Want
  rows : Array Int

def Request.resolve (request : Request) (lookup : Lookup) : Array Nat :=
  request.rows.map (lookup request.useDirectory request.want)

def native : Lookup := fun _ want row =>
  if row < 0 then 0 else nativeNat want row.toNat

theorem native_within (want : Want) (row : Int) : native false want row ≤ want.size := by
  unfold native
  split
  · exact Nat.zero_le _
  · exact (native_insertion want row.toNat).2.1

/-- The launch flag records publication of a feasible directory with the same
sorted want list. Prefix boundaries are lower bounds in that list. -/
def bucketed : Lookup := fun useDirectory want row =>
  if !useDirectory then native false want row
  else if row < 0 then 0
  else if row.toNat ≥ 84602880 then want.size
  else
    let lo := nativeNat want (lowerKey row.toNat)
    let hi := nativeNat want (lowerKey row.toNat + 32768)
    if hi - lo < 256 then locate want.read row.toNat 8 lo hi
    else nativeNat want row.toNat

theorem bucketed_exact : bucketed = native := by
  funext flag want query
  cases flag
  · rfl
  unfold bucketed native
  simp only [Bool.not_true, Bool.false_eq_true, if_false]
  by_cases negative : query < 0
  · simp [negative]
  simp only [if_neg negative]
  let row := query.toNat
  let p := nativeNat want row
  let lo := nativeNat want (lowerKey row)
  let hi := nativeNat want (lowerKey row + 32768)
  have hp : IsInsertion want.read row 0 want.size p := native_insertion want row
  split
  · rename_i outside
    have last : IsInsertion want.read row 0 want.size want.size := by
      refine ⟨Nat.zero_le _, Nat.le_refl _, ?_, ?_⟩
      · intro i _ before
        have valid := want.inVocab i before
        omega
      · intro i after before
        omega
    exact insertion_unique _ _ _ _ _ _ last hp
  · change (if hi - lo < 256 then locate want.read row 8 lo hi else p) = p
    split
    · rename_i width
      have hlo := native_insertion want (lowerKey row)
      have hhi := native_insertion want (lowerKey row + 32768)
      have ordered : lo ≤ hi := insertion_mono _ _ _ _ _ _ (Nat.le_add_right _ _) hlo hhi
      have upper : lo + (hi - lo) = hi := by omega
      obtain ⟨localSlot, atLocal⟩ := insertion_exists want.read row lo (hi - lo) (fun i j _ order bound => by
        have below : j < hi := by omega
        exact want.ordered i j order (Nat.lt_of_lt_of_le below hhi.2.1))
      rw [upper] at atLocal
      rw [locate_exact _ _ 8 _ _ localSlot width atLocal]
      exact insertion_unique _ _ _ _ _ _ (bucket_insertion _ _ _ _ _ _ hlo hhi atLocal) hp
    · rfl

/-- The directory in the launch packet contains the left insertion offset
for each fixed bucket boundary. The host declines buckets of width 256 or more. -/
structure DirectoryReady (want : Want) (offset : Nat → Nat) : Prop where
  boundary : ∀ i, i ≤ bucketCount → offset i = nativeNat want (i * 32768)
  boundedBucket : ∀ i, i < bucketCount → offset (i + 1) - offset i < 256

def emittedHashed (want offset : Nat → Nat) (row : Int) : Nat :=
  let i := row.toNat / 32768
  locate want row.toNat 8 (offset i) (offset (i + 1))

/-- The device lookup uses the two uploaded offsets, the eight masked probes,
and the uploaded terminal offset for an ID above the vocabulary. -/
def emitted (want offset : Nat → Nat) (row : Int) : Nat :=
  if row < 0 then 0
  else if row.toNat ≥ 84602880 then offset bucketCount
  else emittedHashed want offset row

theorem emitted_hashed_eq (want offset : Nat → Nat) (row : Int)
    (nonnegative : 0 ≤ row) (within : row.toNat < 84602880) :
    emittedHashed want offset row = emitted want offset row := by
  simp [emitted, Int.not_lt.mpr nonnegative, Nat.not_le.mpr within]

def probes (read : Nat → Nat) (key : Nat) : Nat → Nat → Nat → List Nat
  | 0, _, _ => []
  | fuel + 1, lo, hi =>
    if lo < hi then
      let next := shrink read key lo hi
      (lo + (hi - lo) / 2) :: probes read key fuel next.1 next.2
    else []

theorem shrink_bounds (read : Nat → Nat) (key lo hi : Nat) (ordered : lo ≤ hi) :
    let next := shrink read key lo hi
    lo ≤ next.1 ∧ next.1 ≤ next.2 ∧ next.2 ≤ hi := by
  unfold shrink
  split
  · dsimp only
    split <;> dsimp <;> omega
  · simp [ordered]

theorem probes_in_slice (read : Nat → Nat) (key fuel lo hi i : Nat) (ordered : lo ≤ hi)
    (live : i ∈ probes read key fuel lo hi) : lo ≤ i ∧ i < hi := by
  induction fuel generalizing lo hi with
  | zero => simp [probes] at live
  | succ remaining ih =>
    by_cases active : lo < hi
    · simp only [probes, if_pos active, List.mem_cons] at live
      rcases live with eq | member
      · omega
      · have bounds := shrink_bounds read key lo hi ordered
        have inner := ih _ _ bounds.2.1 member
        exact ⟨Nat.le_trans bounds.1 inner.1, Nat.lt_of_lt_of_le inner.2 bounds.2.2⟩
    · simp [probes, active] at live

theorem directory_slice_bounds (want : Want) (offset : Nat → Nat) (ready : DirectoryReady want offset)
    (row : Nat) (valid : row < 84602880) :
    offset (row / 32768) ≤ offset (row / 32768 + 1) ∧ offset (row / 32768 + 1) ≤ want.size := by
  let bucket := row / 32768
  have bounds := bucket_indices row valid
  have low := ready.boundary bucket (Nat.le_of_lt bounds.1)
  have high := ready.boundary (bucket + 1) bounds.2
  have lowBound := native_insertion want (bucket * 32768)
  have highBound := native_insertion want ((bucket + 1) * 32768)
  have keyOrder : bucket * 32768 ≤ (bucket + 1) * 32768 := Nat.mul_le_mul_right _ (by omega)
  refine ⟨?_, high ▸ highBound.2.1⟩
  · change offset bucket ≤ offset (bucket + 1)
    rw [low, high]
    exact insertion_mono _ _ _ _ _ _ keyOrder lowBound highBound

def ContainsRow (want : Want) (row : Nat) : Prop :=
  ∃ i < want.size, want.read i = row

/-- For a row in the wanted set, both ends of the inclusive interval
address valid entries. The left insertion index is inside that interval. -/
theorem member_slice_bounds (want : Want) (offset : Nat → Nat) (ready : DirectoryReady want offset)
    (row : Nat) (valid : row < 84602880) (member : ContainsRow want row) :
    offset (row / 32768) ≤ nativeNat want row ∧
      nativeNat want row < offset (row / 32768 + 1) ∧
      offset (row / 32768 + 1) ≤ want.size ∧
      offset (row / 32768 + 1) - offset (row / 32768) < 256 := by
  obtain ⟨index, live, value⟩ := member
  have bucket := bucket_indices row valid
  have left := ready.boundary (row / 32768) (Nat.le_of_lt bucket.1)
  have right := ready.boundary (row / 32768 + 1) bucket.2
  have keyBound := row_in_bucket row
  have native := native_insertion want row
  have first : nativeNat want row ≤ index := by
    by_cases earlier : index < nativeNat want row
    · have wrong := native.2.2.1 index (Nat.zero_le _) earlier
      omega
    · omega
  have beforeEnd : index < offset (row / 32768 + 1) := by
    by_cases later : offset (row / 32768 + 1) ≤ index
    · rw [right] at later
      have wrong := (native_insertion want ((row / 32768 + 1) * 32768)).2.2.2 index later live
      unfold lowerKey at keyBound
      omega
    · omega
  have afterStart : offset (row / 32768) ≤ nativeNat want row := by
    by_cases earlier : nativeNat want row < offset (row / 32768)
    · rw [left] at earlier
      have low := (native_insertion want (row / 32768 * 32768)).2.2.1 _ (Nat.zero_le _) earlier
      have high := native.2.2.2 _ (Nat.le_refl _) (show nativeNat want row < want.size by omega)
      unfold lowerKey at keyBound
      omega
    · omega
  exact ⟨afterStart, Nat.lt_of_le_of_lt first beforeEnd,
    (directory_slice_bounds want offset ready row valid).2, ready.boundedBucket _ bucket.1⟩

theorem directory_probes_inside (want : Want) (offset : Nat → Nat) (ready : DirectoryReady want offset)
    (row : Nat) (valid : row < 84602880) (i : Nat)
    (probe : i ∈ probes want.read row 8 (offset (row / 32768)) (offset (row / 32768 + 1))) :
    i < want.size := by
  have bounds := directory_slice_bounds want offset ready row valid
  have member := (probes_in_slice want.read row 8 _ _ i bounds.1 probe).2
  exact Nat.lt_of_lt_of_le member bounds.2

theorem emitted_exact (want : Want) (offset : Nat → Nat) (ready : DirectoryReady want offset)
    (row : Int) : emitted want.read offset row = native false want row := by
  have same := congrFun (congrFun (congrFun bucketed_exact true) want) row
  suffices emitted want.read offset row = bucketed true want row from this.trans same
  unfold emitted bucketed
  simp only [Bool.not_true, Bool.false_eq_true, if_false]
  by_cases negative : row < 0
  · simp [negative]
  simp only [if_neg negative]
  split
  · have vocabulary : 84602880 ≤ bucketCount * 32768 := by decide
    rw [ready.boundary bucketCount (Nat.le_refl _)]
    have terminal : IsInsertion want.read (bucketCount * 32768) 0 want.size want.size := by
      refine ⟨Nat.zero_le _, Nat.le_refl _, ?_, ?_⟩
      · intro i _ inside
        exact Nat.lt_of_lt_of_le (want.inVocab i inside) vocabulary
      · intro i lower upper
        exact False.elim (Nat.not_lt_of_ge lower upper)
    exact insertion_unique _ _ _ _ _ _ (native_insertion want _) terminal
  · rename_i within
    let i := row.toNat / 32768
    have indices := bucket_indices row.toNat (Nat.lt_of_not_ge within)
    have low := ready.boundary i (Nat.le_of_lt indices.1)
    have high := ready.boundary (i + 1) indices.2
    have small := ready.boundedBucket i indices.1
    rw [low, high, Nat.add_mul, Nat.one_mul] at small
    change locate want.read row.toNat 8 (offset i) (offset (i + 1)) =
      (if nativeNat want (lowerKey row.toNat + 32768) - nativeNat want (lowerKey row.toNat) < 256
       then locate want.read row.toNat 8 (nativeNat want (lowerKey row.toNat)) (nativeNat want (lowerKey row.toNat + 32768))
       else nativeNat want row.toNat)
    rw [low, high, Nat.add_mul, Nat.one_mul]
    exact (if_pos small).symm

#axiomsClean emitted_exact
#axiomsClean directory_probes_inside
#axiomsClean member_slice_bounds
#axiomsClean bucketed_exact
#axiomsClean bucket_indices
#axiomsClean directory_fits
#axiomsClean locate_exact
#axiomsClean emitted_midpoint
#axiomsClean insertion_unique
#axiomsClean bucket_insertion

end VeriTile.Bench.Models.NanoGPT.Track1.NgramSlots
