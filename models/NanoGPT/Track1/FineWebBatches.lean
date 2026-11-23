import Init.Data.Array.Extract
import models.NanoGPT.Track1.NgramRowHash
import VeriTile.Meta.StatementAudit

/-! The BOS-aligned branch of the pinned Shard.next_batch. The list is the
visible BOS index starting at self.i. Returning the unused suffix models
advancing self.i after all ranks have filled their batches. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.FineWeb

def headerBytes : Nat := 256 * 4

/-- The pinned Linux reader uses little-endian int32 header fields and
uint16 token words. Only the first three words of the header carry fields. -/
def littleEndian (bytes : Array UInt8) (offset width : Nat) : Nat :=
  (List.range width).foldl (fun value i =>
    value + (bytes[offset + i]?.getD 0).toNat * 256 ^ i) 0

def shardTokenCount (bytes : ByteArray) : Option Nat := do
  if bytes.size < headerBytes then none else do
    if littleEndian bytes.data 0 4 ≠ 20240520 ∨ littleEndian bytes.data 4 4 ≠ 1 then none else do
      let count := littleEndian bytes.data 8 4
      -- header[2] is signed and becomes the tensor length.
      if count < 2 ^ 31 then some count else none

def readContiguous (data : Array UInt8) (nbytes : Nat) : Array UInt8 :=
  data.extract headerBytes (headerBytes + nbytes)

def readFour (data : Array UInt8) (nbytes : Nat) : Array UInt8 :=
  let e1 := headerBytes + nbytes / 4
  let e2 := headerBytes + nbytes * 2 / 4
  let e3 := headerBytes + nbytes * 3 / 4
  data.extract headerBytes e1 ++ data.extract e1 e2 ++
    data.extract e2 e3 ++ data.extract e3 (headerBytes + nbytes)

theorem four_reads_cover_payload (data : Array UInt8) (nbytes : Nat) :
    readFour data nbytes = readContiguous data nbytes := by
  have h₁ : headerBytes ≤ headerBytes + nbytes / 4 := by omega
  have h₂ : headerBytes + nbytes / 4 ≤ headerBytes + nbytes * 2 / 4 := by omega
  have h₃ : headerBytes + nbytes * 2 / 4 ≤ headerBytes + nbytes * 3 / 4 := by omega
  have h₄ : headerBytes + nbytes * 3 / 4 ≤ headerBytes + nbytes := by omega
  simp [readFour, readContiguous, Array.extract_append_extract, Nat.min_eq_left,
    Nat.max_eq_right, h₁, h₂, h₃, h₄]

def decodeWithReader (read : Array UInt8 → Nat → Array UInt8) (bytes : ByteArray) : Option (Array UInt16) :=
  match shardTokenCount bytes with
  | none => none
  | some count =>
    let payload := read bytes.data (2 * count)
    if payload.size ≠ 2 * count then none else
      some ((Array.range count).map (fun i => UInt16.ofNat (littleEndian payload (2 * i) 2)))

/-- Payload returned by the pinned four-thread file read. Short reads
fail the same byte-count check as `_load_data_shard`. -/
def decodeShard (bytes : ByteArray) : Option (Array UInt16) := decodeWithReader readFour bytes

theorem decode_is_contiguous (bytes : ByteArray) : decodeShard bytes = decodeWithReader readContiguous bytes := by
  simp [decodeShard, decodeWithReader, four_reads_cover_payload]

theorem decode_count (bytes : ByteArray) (tokens : Array UInt16)
    (h : decodeShard bytes = some tokens) : shardTokenCount bytes = some tokens.size := by
  rw [decode_is_contiguous] at h
  cases hCount : shardTokenCount bytes with
  | none => simp [decodeWithReader, hCount] at h
  | some count =>
    simp only [decodeWithReader, hCount] at h
    split at h
    · simp only [reduceCtorEq] at h
    · apply congrArg some
      simpa only [Array.size_map, Array.size_range] using congrArg Array.size (Option.some.inj h)

structure ValidationSource where
  /-- `sorted(glob.glob(filename_pattern))` in Python. -/
  files : List ByteArray
  worldSize : Nat
  rank : Nat

structure TokenBatch where
  inputs : Array UInt32
  targets : Array UInt64
  cumulative : Array UInt32
  ngramRows : Array UInt32
  deriving Inhabited

def bosId : UInt32 := 50256

def cumulativeRows (tokens : Nat) : Nat :=
  if tokens = 16384 then 64 else
  if tokens = 32768 then 96 else
  if tokens = 40960 ∨ tokens = 49152 then 128 else
    128 * ((tokens / 300 + 127) / 128)

def unalignedRankBatch (shard : Array UInt16) (start count : Nat) (hash : RowHash := hashRows) :
    Option TokenBatch :=
  if start + count + 1 > shard.size then none else
    let inputs := (shard.extract start (start + count)).map (UInt32.ofNat ·.toNat)
    let targets := (shard.extract (start + 1) (start + count + 1)).map (UInt64.ofNat ·.toNat)
    let ends := (Array.range count).filter (fun i => inputs[i]?.getD 0 == bosId)
    let rows := cumulativeRows count
    if ends.size + 1 > rows then none else
      let cumulative := #[0] ++ ends.map UInt32.ofNat ++
        Array.replicate (rows - 1 - ends.size) (UInt32.ofNat count)
      (hash inputs).map fun ngramRows => ⟨inputs, targets, cumulative, ngramRows⟩

private def validationBatches (hash : RowHash) (global localCount rank : Nat) :
    Nat → Nat → Array UInt16 → List ByteArray → Option (List TokenBatch)
  | 0, _, _, _ => some []
  | batches + 1, pos, tokens, files => do
    let (pos, tokens, rest) ← if pos + global + 1 ≥ tokens.size then do
      let next :: rest := files | none
      let tokens ← decodeShard next
      some (0, tokens, rest)
    else some (pos, tokens, files)
    let batch ← unalignedRankBatch tokens (pos + rank * localCount) localCount hash
    let rest ← validationBatches hash global localCount rank batches (pos + global) tokens rest
    pure (batch :: rest)

def ValidationSource.load (source : ValidationSource) (globalTokens batches : Nat) (hash : RowHash := hashRows) :
    Option (List TokenBatch) := do
  if batches = 0 then some [] else do
    if source.rank ≥ source.worldSize ∨ globalTokens % source.worldSize ≠ 0 then none else do
      let first :: rest := source.files | none
      let tokens ← decodeShard first
      validationBatches hash globalTokens (globalTokens / source.worldSize) source.rank batches 0 tokens rest

theorem unaligned_batch_shape (shard : Array UInt16) (start count : Nat) (batch : TokenBatch)
    (h : unalignedRankBatch shard start count = some batch) :
    batch.inputs.size = count ∧ batch.targets.size = count ∧
      batch.cumulative.size = cumulativeRows count ∧ batch.ngramRows.size = 2 * count := by
  unfold unalignedRankBatch at h
  split at h
  · contradiction
  · rename_i bounds
    simp only at h
    split at h
    · contradiction
    · rename_i fits
      generalize hInputs : (shard.extract start (start + count)).map (UInt32.ofNat ·.toNat) = inputs at *
      cases hr : hashRows inputs with
      | none => simp only [hr, Option.map_none, reduceCtorEq] at h
      | some rows =>
        have inputSize : inputs.size = count := by
          rw [← hInputs]
          simp only [Array.size_map, Array.size_extract, Nat.min_eq_left (by omega : start + count ≤ shard.size)]
          omega
        have targetSize : ((shard.extract (start + 1) (start + count + 1)).map (UInt64.ofNat ·.toNat)).size = count := by
          simp only [Array.size_map, Array.size_extract, Nat.min_eq_left (by omega : start + count + 1 ≤ shard.size)]
          omega
        simp only [hr, Option.map_some] at h
        rw [← Option.some.inj h]
        have initial : (#[0] : Array UInt32).size = 1 := rfl
        simp only [inputSize, targetSize, hash_rows_size _ _ hr, Array.size_append, Array.size_map, Array.size_replicate,
          initial, true_and, and_true]
        omega

#axiomsClean four_reads_cover_payload
#axiomsClean decode_is_contiguous
#axiomsClean decode_count
#axiomsClean unaligned_batch_shape

abbrev Segment := Nat × Nat

def rankBatch (maxSeqLen shardSize : Nat) : Nat → List Nat → Option (List Segment × List Nat)
  | 0, bos => some ([], bos)
  | _ + 1, [] => none
  | need + 1, start :: bos =>
    let stop := min (bos.headD shardSize) (min (start + maxSeqLen) (start + need + 1))
    (rankBatch maxSeqLen shardSize (need + 1 - (stop - start)) bos).map fun result =>
      ((start, stop) :: result.1, result.2)
termination_by _ bos => bos.length

def allRanks (need maxSeqLen shardSize : Nat) :
    Nat → List Nat → Option (List (List Segment) × List Nat)
  | 0, bos => some ([], bos)
  | ranks + 1, bos => do
    let (segments, remaining) ← rankBatch maxSeqLen shardSize need bos
    let (others, rest) ← allRanks need maxSeqLen shardSize ranks remaining
    pure (segments :: others, rest)

def nextBatch (tokensPerRank maxSeqLen shardSize ranks : Nat) (bos : List Nat) :
    Option (List (List Segment) × List Nat) :=
  allRanks (tokensPerRank + 1) maxSeqLen shardSize ranks bos

private theorem rank_rest_le (maxSeqLen shardSize : Nat) (bos : List Nat) (need : Nat)
    (segments : List Segment) (rest : List Nat)
    (h : rankBatch maxSeqLen shardSize need bos = some (segments, rest)) : rest.length ≤ bos.length := by
  induction bos generalizing need segments rest with
  | nil =>
    cases need <;> simp [rankBatch] at h
    · exact h.2 ▸ Nat.le_refl _
  | cons start tail ih =>
    cases need with
    | zero =>
      simp [rankBatch] at h
      exact h.2 ▸ Nat.le_refl _
    | succ need =>
      rw [rankBatch] at h
      cases he : rankBatch maxSeqLen shardSize
        (need + 1 - (min (tail.headD shardSize) (min (start + maxSeqLen) (start + need + 1)) - start)) tail with
      | none => simp only [he, Option.map_none, reduceCtorEq] at h
      | some payload =>
        rcases payload with ⟨parts, leftover⟩
        simp only [he, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        exact Nat.le_trans (h.2 ▸ ih _ _ _ he) (Nat.le_succ _)

private theorem rank_prefix_extension (maxSeqLen shardSize : Nat) (bos suffix : List Nat) (need : Nat)
    (segments : List Segment) (rest : List Nat) (remains : rest ≠ [])
    (h : rankBatch maxSeqLen shardSize need bos = some (segments, rest)) :
    rankBatch maxSeqLen shardSize need (bos ++ suffix) = some (segments, rest ++ suffix) := by
  induction bos generalizing need segments rest with
  | nil =>
    cases need <;> simp [rankBatch] at h
    exact False.elim (remains h.2)
  | cons start tail ih =>
    cases need with
    | zero =>
      simp only [rankBatch, Option.some.injEq, Prod.mk.injEq] at h ⊢
      exact ⟨h.1, congrArg (· ++ suffix) h.2⟩
    | succ need =>
      let stop := min (tail.headD shardSize) (min (start + maxSeqLen) (start + need + 1))
      let consumed := need + 1 - (stop - start)
      rw [rankBatch] at h
      change (rankBatch maxSeqLen shardSize consumed tail).map
        (fun result => ((start, stop) :: result.1, result.2)) = some (segments, rest) at h
      cases hr : rankBatch maxSeqLen shardSize consumed tail with
      | none => simp only [hr, Option.map_none, reduceCtorEq] at h
      | some payload =>
        rcases payload with ⟨parts, leftover⟩
        simp only [hr, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        have tailNonempty : tail ≠ [] := by
          intro e
          have le := rank_rest_le _ _ tail consumed parts leftover hr
          rw [h.2, e] at le
          exact remains (List.eq_nil_of_length_eq_zero (by simpa using le))
        have head_eq : (tail ++ suffix).headD shardSize = tail.headD shardSize := by
          cases tail with
          | nil => exact (tailNonempty rfl).elim
          | cons _ _ => rfl
        have next := ih consumed parts leftover (h.2 ▸ remains) hr
        simp only [List.cons_append, rankBatch, head_eq]
        change (rankBatch maxSeqLen shardSize consumed (tail ++ suffix)).map _ = _
        simp only [next, Option.map_some, h.2]
        change some ((start, stop) :: parts, rest ++ suffix) = _
        rw [h.1]

private theorem all_rest_le (need maxSeqLen shardSize ranks : Nat) (bos : List Nat)
    (segments : List (List Segment)) (rest : List Nat)
    (h : allRanks need maxSeqLen shardSize ranks bos = some (segments, rest)) :
    rest.length ≤ bos.length := by
  induction ranks generalizing bos segments rest with
  | zero =>
    simp [allRanks] at h
    exact h.2 ▸ Nat.le_refl _
  | succ n ih =>
    rw [allRanks] at h
    cases hr : rankBatch maxSeqLen shardSize need bos with
    | none => simp [hr] at h
    | some payload =>
      rcases payload with ⟨first, tail⟩
      simp only [hr] at h
      cases ht : allRanks need maxSeqLen shardSize n tail with
      | none => simp [ht] at h
      | some result =>
        rcases result with ⟨others, left⟩
        simp [ht] at h
        exact Nat.le_trans (h.2 ▸ ih _ _ _ ht) (rank_rest_le _ _ _ _ _ _ hr)

theorem prefix_extension (need maxSeqLen shardSize ranks : Nat) (bos suffix : List Nat)
    (segments : List (List Segment)) (rest : List Nat) (remains : rest ≠ [])
    (h : allRanks need maxSeqLen shardSize ranks bos = some (segments, rest)) :
    allRanks need maxSeqLen shardSize ranks (bos ++ suffix) = some (segments, rest ++ suffix) := by
  induction ranks generalizing bos segments rest with
  | zero =>
    simp only [allRanks, Option.some.injEq, Prod.mk.injEq] at h ⊢
    exact ⟨h.1, congrArg (· ++ suffix) h.2⟩
  | succ n ih =>
    rw [allRanks] at h
    cases hr : rankBatch maxSeqLen shardSize need bos with
    | none => simp [hr] at h
    | some payload =>
      rcases payload with ⟨first, tail⟩
      simp only [hr] at h
      cases ht : allRanks need maxSeqLen shardSize n tail with
      | none => simp [ht] at h
      | some result =>
        rcases result with ⟨others, left⟩
        simp [ht] at h
        have tailNonempty : tail ≠ [] := by
          intro e
          have le := all_rest_le _ _ _ _ _ _ _ ht
          rw [h.2, e] at le
          exact remains (List.eq_nil_of_length_eq_zero (by simpa using le))
        have next := ih tail others left (h.2 ▸ remains) ht
        rw [allRanks, rank_prefix_extension _ _ _ suffix _ _ _ tailNonempty hr]
        simp [next, h.1, h.2]

/-- A consumed prefix with a remaining BOS gives the same coordinates in
every completion. Thus the readiness event only affects a boundary call
where the current prefix cannot supply every next-BOS lookup. -/
theorem next_batch_prefix_stable (tokens maxSeqLen shardSize ranks : Nat) (visible suffix : List Nat)
    (segments : List (List Segment)) (rest : List Nat) (remains : rest ≠ [])
    (h : nextBatch tokens maxSeqLen shardSize ranks visible = some (segments, rest)) :
    nextBatch tokens maxSeqLen shardSize ranks (visible ++ suffix) = some (segments, rest ++ suffix) :=
  prefix_extension _ _ _ _ _ _ _ _ remains h

/-- The two token arrays in the BOS-aligned Python loader are adjacent
views into the concatenated document slices. -/
def materializeRank (tokens : List UInt16) (segments : List Segment) :
    List UInt16 × List UInt16 :=
  let buf := segments.flatMap fun (start, stop) => (tokens.drop start).take (stop - start)
  (buf.take (buf.length - 1), buf.drop 1)

/-- Observe the input and target token sequences and omit the index cursor.
The event decides whether the full index has replaced the initial prefix. -/
def batchTokens (tokens : List UInt16) (tokensPerRank maxSeqLen ranks : Nat) (visible suffix : List Nat)
    (ready : Bool) : Option (List (List UInt16 × List UInt16)) :=
  (nextBatch tokensPerRank maxSeqLen tokens.length ranks (if ready then visible ++ suffix else visible)).map
    fun result => result.1.map (materializeRank tokens)

theorem tokens_independent_of_readiness (tokens : List UInt16) (tokensPerRank maxSeqLen ranks : Nat)
    (visible suffix : List Nat) (segments : List (List Segment)) (rest : List Nat)
    (remains : rest ≠ [])
    (h : nextBatch tokensPerRank maxSeqLen tokens.length ranks visible = some (segments, rest)) :
    batchTokens tokens tokensPerRank maxSeqLen ranks visible suffix false =
      batchTokens tokens tokensPerRank maxSeqLen ranks visible suffix true := by
  have expanded := next_batch_prefix_stable _ _ _ _ _ suffix _ _ remains h
  simp only [batchTokens, Bool.false_eq_true, ↓reduceIte, h, expanded, Option.map_some]

theorem readiness_counterexample :
    nextBatch 3 2 20 1 [0] = none ∧
      nextBatch 3 2 20 1 [0, 2, 4] = some ([[(0, 2), (2, 4)]], [4]) := by
  constructor <;> decide +kernel

#axiomsClean next_batch_prefix_stable
#axiomsClean tokens_independent_of_readiness
#axiomsClean readiness_counterexample
end VeriTile.Bench.Models.NanoGPT.Track1.FineWeb
