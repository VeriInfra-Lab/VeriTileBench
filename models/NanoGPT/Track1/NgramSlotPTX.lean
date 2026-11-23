import models.NanoGPT.Track1.NgramSlots

/-! The integer registers and predicated global loads of the eight search
rounds in the specialized SM90a lookup block. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramSlotPTX
open NgramSlots

def signed (word : UInt32) : Int := (Int32.ofNat word.toNat).toInt

def queryIndex (copyCTAs pid tid : Nat) : UInt32 :=
  ((UInt32.ofNat pid <<< 7) ||| (UInt32.ofNat tid &&& 127)) - UInt32.ofNat (128 * copyCTAs)

def interval (lo hi : Nat) : UInt32 × UInt32 := (UInt32.ofNat lo, UInt32.ofNat hi)

def midpoint (lo hi : UInt32) : UInt32 := lo + ((hi - lo) >>> 1)

def probe (read : Nat → UInt32) (row : UInt32) (bounds : UInt32 × UInt32) :
    UInt32 × UInt32 :=
  let (lo, hi) := bounds
  let active := signed hi > signed lo
  let mid := midpoint lo hi
  let value := if active then read (signed mid).toNat else row
  let higher := signed value < signed row
  (if higher then mid + 1 else lo, if higher then hi else mid)

def probeIndex (bounds : UInt32 × UInt32) : Option Nat :=
  if signed bounds.1 < signed bounds.2 then some (signed (midpoint bounds.1 bounds.2)).toNat else none

def locateWords (read : Nat → UInt32) (row : UInt32) : Nat → (UInt32 × UInt32) → UInt32
  | 0, bounds => bounds.1
  | fuel + 1, bounds => locateWords read row fuel (probe read row bounds)

def readTrace (read : Nat → UInt32) (row : UInt32) : Nat → (UInt32 × UInt32) → List Nat
  | 0, _ => []
  | fuel + 1, bounds =>
    (probeIndex bounds).toList ++ readTrace read row fuel (probe read row bounds)

def startBounds (offset : Nat → Nat) (row : UInt32) : UInt32 × UInt32 :=
  let bucket := (row >>> 15).toNat
  interval (offset bucket) (offset (bucket + 1))

def lookup (read offset : Nat → Nat) (row : Nat) : UInt32 :=
  let key := UInt32.ofNat row
  locateWords (fun i => UInt32.ofNat (read i)) key 8 (startBounds offset key)

private theorem unsigned_exact (n : Nat) (fits : n < 2147483648) : (UInt32.ofNat n).toNat = n :=
  UInt32.toNat_ofNat_of_lt' (by change n < 4294967296; omega)

private theorem signed_exact (n : Nat) (fits : n < 2147483648) : signed (UInt32.ofNat n) = (n : Int) := by
  rw [signed, unsigned_exact n fits, Int32.toInt_ofNat_of_lt fits]

theorem query_index_exact (copyCTAs pid tid : Nat) (lookupCTA : copyCTAs ≤ pid)
    (inGrid : pid < 1024) (inBlock : tid < 128) :
    signed (queryIndex copyCTAs pid tid) = (128 * (pid - copyCTAs) + tid : Nat) := by
  have mask : (UInt32.ofNat tid &&& 127).toNat = tid := by
    rw [UInt32.toNat_and, unsigned_exact tid (by omega)]
    exact Nat.and_two_pow_sub_one_of_lt_two_pow (n := 7) inBlock
  have before : ((UInt32.ofNat pid <<< 7) ||| (UInt32.ofNat tid &&& 127)).toNat =
      128 * pid + tid := by
    rw [UInt32.toNat_or, UInt32.toNat_shiftLeft, unsigned_exact pid (by omega), mask]
    change ((pid <<< 7) % 4294967296) ||| tid = 128 * pid + tid
    have product : pid <<< 7 = 128 * pid := by simp [Nat.shiftLeft_eq, Nat.mul_comm]
    rw [Nat.mod_eq_of_lt (by rw [product]; omega)]
    rw [← Nat.shiftLeft_add_eq_or_of_lt (i := 7) inBlock, product]
  have start := unsigned_exact (128 * copyCTAs) (by omega)
  have forward : UInt32.ofNat (128 * copyCTAs) ≤
      ((UInt32.ofNat pid <<< 7) ||| (UInt32.ofNat tid &&& 127)) := by
    rw [UInt32.le_iff_toNat_le, start, before]
    omega
  have word : queryIndex copyCTAs pid tid = UInt32.ofNat (128 * (pid - copyCTAs) + tid) := by
    apply UInt32.toNat.inj
    rw [queryIndex, UInt32.toNat_sub_of_le _ _ forward, start, before, unsigned_exact _ (by omega)]
    omega
  rw [word, signed_exact _ (by omega)]

private theorem midpoint_exact (lo hi : Nat) (ordered : lo ≤ hi) (fits : hi < 2147483648) :
    midpoint (UInt32.ofNat lo) (UInt32.ofNat hi) = UInt32.ofNat (lo + (hi - lo) / 2) := by
  have low := unsigned_exact lo (by omega)
  have high := unsigned_exact hi fits
  have subFits : (UInt32.ofNat lo) ≤ (UInt32.ofNat hi) := by
    simpa only [UInt32.le_iff_toNat_le, low, high] using ordered
  apply UInt32.toNat.inj
  rw [midpoint, UInt32.toNat_add, UInt32.toNat_shiftRight, UInt32.toNat_sub_of_le _ _ subFits,
    low, high, Nat.shiftRight_eq_div_pow, unsigned_exact _ (by omega)]
  change (lo + (hi - lo) / 2) % 4294967296 = lo + (hi - lo) / 2
  exact Nat.mod_eq_of_lt (by omega)

private theorem increment_exact (n : Nat) (fits : n + 1 < 2147483648) :
    UInt32.ofNat n + 1 = UInt32.ofNat (n + 1) := by
  apply UInt32.toNat.inj
  rw [UInt32.toNat_add, unsigned_exact n (by omega), unsigned_exact _ fits]
  change (n + 1) % 4294967296 = n + 1
  exact Nat.mod_eq_of_lt (by omega)

theorem probe_index_exact (lo hi : Nat) (ordered : lo ≤ hi) (fits : hi < 2147483648) :
    probeIndex (interval lo hi) = if lo < hi then some (lo + (hi - lo) / 2) else none := by
  have mid := signed_exact (lo + (hi - lo) / 2) (by omega)
  simp only [probeIndex, interval, signed_exact lo (by omega), signed_exact hi fits,
    midpoint_exact lo hi ordered fits, mid, Int.ofNat_lt, Int.toNat_natCast]

theorem probe_exact (want : Want) (key lo hi : Nat)
    (ordered : lo ≤ hi) (upper : hi ≤ want.size) (fits : want.size < 2147483648)
    (keyFits : key < 84602880) :
    probe (fun i => UInt32.ofNat (want.read i)) (UInt32.ofNat key) (interval lo hi) =
      let next := shrink want.read key lo hi
      interval next.1 next.2 := by
  have low := signed_exact lo (by omega)
  have high := signed_exact hi (by omega)
  simp only [probe, interval, low, high, gt_iff_lt, Int.ofNat_lt, shrink]
  by_cases active : lo < hi
  · simp only [if_pos active, midpoint_exact lo hi ordered (by omega),
      signed_exact _ (show lo + (hi - lo) / 2 < 2147483648 by omega), Int.toNat_natCast]
    have readFits := want.inVocab (lo + (hi - lo) / 2) (by omega)
    simp only [signed_exact _ (show want.read (lo + (hi - lo) / 2) < 2147483648 by omega),
      signed_exact key (by omega), Int.ofNat_lt, increment_exact (lo + (hi - lo) / 2) (by omega)]
    by_cases higher : want.read (lo + (hi - lo) / 2) < key <;> simp [higher]
  · have closed : lo = hi := by omega
    subst hi
    simp [midpoint_exact _ _ (Nat.le_refl _) (show lo < 2147483648 by omega)]

theorem locate_words_exact (want : Want) (key fuel lo hi : Nat)
    (ordered : lo ≤ hi) (upper : hi ≤ want.size) (fits : want.size < 2147483648)
    (keyFits : key < 84602880) :
    (locateWords (fun i => UInt32.ofNat (want.read i)) (UInt32.ofNat key) fuel (interval lo hi)).toNat =
      locate want.read key fuel lo hi := by
  induction fuel generalizing lo hi with
  | zero => exact unsigned_exact lo (by omega)
  | succ fuel ih =>
    have bounds := shrink_bounds want.read key lo hi ordered
    rw [locateWords, probe_exact want key lo hi ordered upper fits keyFits, locate]
    exact ih _ _ bounds.2.1 (Nat.le_trans bounds.2.2 upper)

theorem read_trace_exact (want : Want) (key fuel lo hi : Nat)
    (ordered : lo ≤ hi) (upper : hi ≤ want.size) (fits : want.size < 2147483648)
    (keyFits : key < 84602880) :
    readTrace (fun i => UInt32.ofNat (want.read i)) (UInt32.ofNat key) fuel (interval lo hi) =
      probes want.read key fuel lo hi := by
  induction fuel generalizing lo hi with
  | zero => rfl
  | succ fuel ih =>
    have bounds := shrink_bounds want.read key lo hi ordered
    have rest := ih _ _ bounds.2.1 (Nat.le_trans bounds.2.2 upper)
    rw [readTrace, probe_index_exact lo hi ordered (by omega),
      probe_exact want key lo hi ordered upper fits keyFits, rest, probes]
    by_cases active : lo < hi
    · simp [active]
    · cases fuel <;> simp [active, shrink, probes]

theorem bucket_exact (row : Nat) (valid : row < 84602880) :
    (UInt32.ofNat row >>> 15).toNat = row / 32768 := by
  rw [UInt32.toNat_shiftRight, unsigned_exact row (by omega), Nat.shiftRight_eq_div_pow]
  rfl

theorem lookup_reads_fit (want : Want) (offset : Nat → Nat) (directory : DirectoryReady want offset)
    (fits : want.size < 2147483648) (row : Nat) (valid : row < 84602880) (i : Nat)
    (issued : i ∈ readTrace (fun j => UInt32.ofNat (want.read j)) (UInt32.ofNat row) 8
      (startBounds offset (UInt32.ofNat row))) : i < want.size := by
  have bounds := directory_slice_bounds want offset directory row valid
  rw [startBounds, bucket_exact row valid, read_trace_exact want row 8 _ _ bounds.1 bounds.2 fits valid] at issued
  exact directory_probes_inside want offset directory row valid i issued

theorem lookup_exact (want : Want) (offset : Nat → Nat) (directory : DirectoryReady want offset)
    (fits : want.size < 2147483648) (row : Nat) (valid : row < 84602880) :
    (lookup want.read offset row).toNat = native false want row := by
  have bounds := directory_slice_bounds want offset directory row valid
  have search := locate_words_exact want row 8 _ _ bounds.1 bounds.2 fits valid
  rw [lookup, startBounds, bucket_exact row valid, search]
  change emittedHashed want.read offset (row : Int) = _
  rw [emitted_hashed_eq _ _ _ (by omega) (by simpa), emitted_exact want offset directory]

def memberProbe (read : Nat → UInt32) (row : UInt32) (bounds : UInt32 × UInt32) (step : Nat) :
    Nat × (UInt32 × UInt32) :=
  let (lo, hi) := bounds
  let candidate := lo + UInt32.ofNat (step - 1)
  let index := (if candidate ≤ hi then candidate else hi).toNat
  let higher := signed (read index) < signed row
  (index, (if higher then lo + UInt32.ofNat step else lo, hi))

def memberRounds (read : Nat → UInt32) (row : UInt32) : Nat → (UInt32 × UInt32) → UInt32 × List Nat
  | 0, bounds => (bounds.1, [])
  | fuel + 1, bounds =>
    let issued := memberProbe read row bounds (2 ^ fuel)
    let next := memberRounds read row fuel issued.2
    (next.1, issued.1 :: next.2)

def memberLookup (read offset : Nat → Nat) (row : Nat) : UInt32 × List Nat :=
  let key := UInt32.ofNat row
  let bounds := startBounds offset key
  memberRounds (fun i => UInt32.ofNat (read i)) key 8 (bounds.1, bounds.2 - 1)

private theorem member_probe_exact (want : Want) (key lo hi step : Nat)
    (ordered : lo ≤ hi) (upper : hi < want.size) (fits : want.size < 2147483648) (small : step ≤ 128)
    (valid : key < 84602880) :
    memberProbe (fun i => UInt32.ofNat (want.read i)) (UInt32.ofNat key) (interval lo hi) step =
      let mid := min (lo + (step - 1)) hi
      (mid, interval (if want.read mid < key then lo + step else lo) hi) := by
  let candidate := lo + (step - 1)
  let mid := min candidate hi
  have candidateFits : (UInt32.ofNat candidate).toNat = candidate :=
    UInt32.toNat_ofNat_of_lt' (by dsimp [candidate]; change _ < 4294967296; omega)
  have midWord :
      (if UInt32.ofNat candidate ≤ UInt32.ofNat hi then UInt32.ofNat candidate else UInt32.ofNat hi).toNat = mid := by
    have high := unsigned_exact hi (by omega)
    by_cases within : candidate ≤ hi
    · have withinWord : UInt32.ofNat candidate ≤ UInt32.ofNat hi := by
        simpa only [UInt32.le_iff_toNat_le, candidateFits, high] using within
      simp only [if_pos withinWord, candidateFits, mid, Nat.min_eq_left within]
    · have beyondWord : ¬ UInt32.ofNat candidate ≤ UInt32.ofNat hi := by
        simpa only [UInt32.le_iff_toNat_le, candidateFits, high] using within
      simp only [if_neg beyondWord, high, mid, Nat.min_eq_right (by omega : hi ≤ candidate)]
  have readFits := want.inVocab mid (Nat.lt_of_le_of_lt (Nat.min_le_right _ _) upper)
  simp only [memberProbe, interval, ← UInt32.ofNat_add, ← show candidate = lo + (step - 1) from rfl, midWord,
    ← show mid = min candidate hi from rfl, signed_exact (want.read mid) (by omega),
    signed_exact key (by omega), Int.ofNat_lt]
  split <;> rfl

private theorem member_rounds_exact (want : Want) (key fuel lo hi slot : Nat)
    (correct : IsInsertion want.read key 0 want.size slot) (lower : lo ≤ slot) (upper : slot ≤ hi)
    (within : hi < want.size) (fits : want.size < 2147483648) (valid : key < 84602880)
    (length : slot - lo < 2 ^ fuel) (rounds : fuel ≤ 8) :
    let result := memberRounds (fun i => UInt32.ofNat (want.read i)) (UInt32.ofNat key) fuel (interval lo hi)
    result.1.toNat = slot ∧ ∀ i ∈ result.2, i < want.size := by
  induction fuel generalizing lo hi with
  | zero =>
    have exactSlot : lo = slot := by simp only [Nat.pow_zero] at length; omega
    change (UInt32.ofNat lo).toNat = slot ∧ ∀ i ∈ ([] : List Nat), i < want.size
    rw [unsigned_exact lo (by omega)]
    simp [exactSlot]
  | succ fuel ih =>
    let step := 2 ^ fuel
    let mid := min (lo + (step - 1)) hi
    have positive : 0 < step := Nat.two_pow_pos _
    have small : step ≤ 128 := Nat.pow_le_pow_right (by decide) (by omega : fuel ≤ 7)
    have bound : mid ≤ hi := Nat.min_le_right _ _
    have inside : mid < want.size := Nat.lt_of_le_of_lt bound within
    have twice : slot - lo < step * 2 := by simpa only [Nat.pow_succ] using length
    rw [memberRounds, member_probe_exact want key lo hi (2 ^ fuel) (by omega) within fits small valid]
    change _ ∧ ∀ i ∈ _, i < want.size
    have advances : want.read mid < key ↔ lo + step ≤ slot := by
      constructor
      · intro higher
        have earlier : mid < slot := by
          by_cases later : slot ≤ mid
          · exact False.elim ((Nat.not_lt_of_ge (correct.2.2.2 mid later inside)) higher)
          · omega
        by_cases within : lo + (step - 1) ≤ hi
        · simp only [mid, Nat.min_eq_left within] at earlier
          omega
        · simp only [mid, Nat.min_eq_right (Nat.le_of_not_ge within)] at earlier
          omega
      · intro later
        have before : mid ≤ lo + (step - 1) := Nat.min_le_left _ _
        exact correct.2.2.1 mid (Nat.zero_le _) (by omega)
    by_cases higher : want.read mid < key
    · have progress := advances.mp higher
      have done := ih (lo + step) hi progress upper within (by omega) (by omega)
      simp only [show min (lo + (2 ^ fuel - 1)) hi = mid from rfl, if_pos higher]
      exact ⟨done.1, fun i seen => (List.mem_cons.mp seen).elim (fun h => h ▸ inside) (done.2 i)⟩
    · have pending : ¬ lo + step ≤ slot := fun later => higher (advances.mpr later)
      have done := ih lo hi lower upper within (by omega) (by omega)
      simp only [show min (lo + (2 ^ fuel - 1)) hi = mid from rfl, if_neg higher]
      exact ⟨done.1, fun i seen => (List.mem_cons.mp seen).elim (fun h => h ▸ inside) (done.2 i)⟩

set_option maxHeartbeats 1000000 in
theorem member_lookup_exact (want : Want) (offset : Nat → Nat) (directory : DirectoryReady want offset)
    (fits : want.size < 2147483648) (row : Nat) (valid : row < 84602880) (member : ContainsRow want row) :
    (memberLookup want.read offset row).1.toNat = native false want (row : Int) ∧
    (∀ i ∈ (memberLookup want.read offset row).2, i < want.size) := by
  obtain ⟨lo, hi, upper, width⟩ := member_slice_bounds want offset directory row valid member
  let endIndex := offset (row / 32768 + 1)
  have endFits : endIndex < 2147483648 := by dsimp [endIndex]; omega
  have one : (1 : UInt32) ≤ UInt32.ofNat endIndex := by
    rw [UInt32.le_iff_toNat_le, unsigned_exact endIndex endFits]
    change 1 ≤ endIndex
    dsimp [endIndex]
    omega
  have decrement : UInt32.ofNat endIndex - 1 = UInt32.ofNat (endIndex - 1) := by
    apply UInt32.toNat.inj
    rw [UInt32.toNat_sub_of_le _ _ one, unsigned_exact endIndex endFits, unsigned_exact _ (by omega)]
    rfl
  have wired : memberLookup want.read offset row =
      memberRounds (fun i => UInt32.ofNat (want.read i)) (UInt32.ofNat row) 8
        (interval (offset (row / 32768)) (endIndex - 1)) := by
    have addresses : startBounds offset (UInt32.ofNat row) =
        interval (offset (row / 32768)) endIndex := by
      simp only [startBounds, bucket_exact row valid, endIndex]
    simp only [memberLookup, addresses, interval, decrement]
  rw [wired]
  have done := member_rounds_exact want row 8 (offset (row / 32768)) (endIndex - 1) (nativeNat want row)
    (native_insertion want row) lo (by dsimp [endIndex]; omega) (by dsimp [endIndex]; omega) fits valid
    (by change _ < 256; omega) (by decide)
  have nonnegative : ¬ (row : Int) < 0 := by omega
  simpa only [native, if_neg nonnegative, Int.toNat_natCast] using done

#axiomsClean probe_index_exact
#axiomsClean query_index_exact
#axiomsClean probe_exact
#axiomsClean lookup_reads_fit
#axiomsClean lookup_exact
#axiomsClean member_lookup_exact
end VeriTile.Bench.Models.NanoGPT.Track1.NgramSlotPTX
