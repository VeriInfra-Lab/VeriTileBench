import models.NanoGPT.Track1.FineWebBatches

/-! The BOS-aligned training stream, including file rollover and the
ordered prefetch cache used by ScheduledBatches. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.FineWeb

def documentLengths (segments : List Segment) : List Int :=
  segments.map fun (first, last) => (last : Int) - first

def subtractLast : List Int → List Int
  | [] => []
  | [x] => [x - 1]
  | x :: y :: ys => x :: subtractLast (y :: ys)

theorem last_shift_commutes_with_scan (lengths : List Int) (acc : Int) :
    (subtractLast lengths).scanl (· + ·) acc =
      if lengths.isEmpty then [acc] else subtractLast (lengths.scanl (· + ·) acc) := by
  induction lengths generalizing acc with
  | nil => rfl
  | cons x rest ih =>
    cases rest with
    | nil => simp [subtractLast, List.scanl, Int.add_sub_assoc]
    | cons y ys =>
      simp only [subtractLast, List.isEmpty_cons, Bool.false_eq_true, if_false]
      rw [List.scanl_cons, List.scanl_cons]
      have unfoldFirst : subtractLast (acc :: List.scanl (· + ·) (acc + x) (y :: ys)) =
          acc :: subtractLast (List.scanl (· + ·) (acc + x) (y :: ys)) := by
        rw [List.scanl_cons]
        rfl
      rw [unfoldFirst]
      exact congrArg (List.cons acc) (by simpa only [List.isEmpty_cons, Bool.false_eq_true, if_false] using ih (acc + x))

def endsReference (segments : List Segment) : List Int :=
  ((subtractLast (documentLengths segments)).scanl (· + ·) 0).drop 1

def endsPacked (segments : List Segment) : List Int :=
  subtractLast (((documentLengths segments).scanl (· + ·) 0).drop 1)

theorem ends_equal (segments : List Segment) : endsPacked segments = endsReference segments := by
  unfold endsPacked endsReference
  rw [last_shift_commutes_with_scan]
  cases h : documentLengths segments with
  | nil => rfl
  | cons x rest =>
    cases rest <;> simp [List.scanl_cons, subtractLast]

theorem rank_lengths (maxSeqLen shardSize : Nat) (bos : List Nat) (need : Nat)
    (segments : List Segment) (rest : List Nat)
    (h : rankBatch maxSeqLen shardSize need bos = some (segments, rest)) :
    (segments.map (fun (first, last) => last - first)).sum = need := by
  induction bos generalizing need segments rest with
  | nil =>
    cases need <;> simp [rankBatch] at h
    simp [h.1]
  | cons start tail ih =>
    cases need with
    | zero => simp [rankBatch] at h; simp [h.1]
    | succ need =>
      rw [rankBatch] at h
      let stop := min (tail.headD shardSize) (min (start + maxSeqLen) (start + need + 1))
      change (rankBatch maxSeqLen shardSize (need + 1 - (stop - start)) tail).map _ = _ at h
      cases e : rankBatch maxSeqLen shardSize (need + 1 - (stop - start)) tail with
      | none => simp only [e, Option.map_none, reduceCtorEq] at h
      | some s =>
        rcases s with ⟨parts, leftover⟩
        have mass := ih _ _ _ e
        have hstop : stop ≤ start + need + 1 := by
          exact Nat.le_trans (Nat.min_le_right ..) (Nat.min_le_right ..)
        simp only [e, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        rw [← h.1, List.map_cons, List.sum_cons, mass]
        change stop - start + (need + 1 - (stop - start)) = need + 1
        omega

#axiomsClean ends_equal
#axiomsClean rank_lengths

theorem rank_stops_in_shard (maxSeqLen shardSize : Nat) (bos : List Nat) (need : Nat)
    (segments : List Segment) (rest : List Nat)
    (within : ∀ i ∈ bos, i ≤ shardSize)
    (h : rankBatch maxSeqLen shardSize need bos = some (segments, rest)) :
    ∀ segment ∈ segments, segment.2 ≤ shardSize := by
  induction bos generalizing need segments rest with
  | nil =>
    cases need <;> simp [rankBatch] at h
    simp [h.1]
  | cons start tail ih =>
    cases need with
    | zero => simp [rankBatch] at h; simp [h.1]
    | succ need =>
      let stop := min (tail.headD shardSize) (min (start + maxSeqLen) (start + need + 1))
      rw [rankBatch] at h
      change (rankBatch maxSeqLen shardSize (need + 1 - (stop - start)) tail).map _ = _ at h
      cases next : rankBatch maxSeqLen shardSize (need + 1 - (stop - start)) tail with
      | none => simp only [next, Option.map_none, reduceCtorEq] at h
      | some payload =>
        rcases payload with ⟨parts, leftover⟩
        have tailBound : ∀ i ∈ tail, i ≤ shardSize := fun i member => within i (by simp [member])
        have nextStop : tail.headD shardSize ≤ shardSize := by
          cases tail with
          | nil => exact Nat.le_refl _
          | cons i more => exact tailBound i (by simp)
        have stopBound : stop ≤ shardSize := Nat.le_trans (Nat.min_le_left ..) nextStop
        simp only [next, Option.map_some, Option.some.injEq, Prod.mk.injEq] at h
        rw [← h.1]
        intro segment member
        rcases List.mem_cons.mp member with rfl | member
        · exact stopBound
        · exact ih _ _ _ tailBound next _ member

def concatenateSegments (tokens : Array UInt16) (segments : List Segment) : Array UInt16 :=
  segments.foldl (fun out (start, stop) => out ++ tokens.extract start stop) #[]

private theorem concat_size (tokens : Array UInt16) (segments : List Segment) (init : Array UInt16)
    (within : ∀ segment ∈ segments, segment.2 ≤ tokens.size) :
    (segments.foldl (fun out (start, stop) => out ++ tokens.extract start stop) init).size =
      init.size + (segments.map (fun (start, stop) => stop - start)).sum := by
  induction segments generalizing init with
  | nil => simp
  | cons s rest ih =>
    rcases s with ⟨start, stop⟩
    have stopBound := within (start, stop) (by simp)
    rw [List.foldl_cons, ih _ (fun s member => within s (by simp [member]))]
    simp only [Array.size_append, Array.size_extract, Nat.min_eq_left stopBound,
      List.map_cons, List.sum_cons, Nat.add_assoc]

theorem rank_materializes_all_words (tokens : Array UInt16) (maxLength need : Nat)
    (bos : List Nat) (segments : List Segment) (rest : List Nat)
    (within : ∀ i ∈ bos, i ≤ tokens.size)
    (h : rankBatch maxLength tokens.size need bos = some (segments, rest)) :
    (concatenateSegments tokens segments).size = need := by
  rw [concatenateSegments, concat_size _ _ _ (rank_stops_in_shard _ _ _ _ _ _ within h),
    rank_lengths _ _ _ _ _ _ h]
  simp [Array.size_empty]

#axiomsClean rank_materializes_all_words


def splitAtCaps (ends : List Nat) (cap : Nat := 2560) : List Nat :=
  ((0 :: ends).zip ends).flatMap fun (start, stop) =>
    ((List.range ((stop - start - 1) / cap)).map fun i => start + (i + 1) * cap) ++ [stop]

abbrev TrainingPack := Array UInt16 → List Segment → Nat → Nat → RowHash → Option TokenBatch

private def packTrainingBatch (boundaries : List Segment → List Int) : TrainingPack :=
  fun tokens segments count maxLength hash => do
    let buf := concatenateSegments tokens segments
    if buf.size ≠ count + 1 then none else do
      let inputs := (buf.extract 0 count).map (UInt32.ofNat ·.toNat)
      let targets := (buf.extract 1 (count + 1)).map (UInt64.ofNat ·.toNat)
      let ends := (boundaries segments).map Int.toNat
      let ends := if maxLength > 2560 then splitAtCaps ends else ends
      let rows := cumulativeRows count
      if ends.length + 1 > rows then none else do
        let packed := #[0] ++ ends.toArray.map UInt32.ofNat ++
          Array.replicate (rows - 1 - ends.length) (UInt32.ofNat count)
        let hashes ← hash inputs
        some ⟨inputs, targets, packed, hashes⟩

def packTrainingReference : TrainingPack := packTrainingBatch endsReference

def canPackTraining (count : Nat) : Prop :=
  count = 16384 ∨ count = 32768 ∨ count = 40960 ∨ count = 49152

instance (count : Nat) : Decidable (canPackTraining count) := by
  unfold canPackTraining
  infer_instance

def packTrainingCPU : TrainingPack := fun tokens segments count length hash =>
  if canPackTraining count then
    packTrainingBatch endsPacked tokens segments count length hash
  else packTrainingReference tokens segments count length hash

theorem pack_training_equal : packTrainingCPU = packTrainingReference := by
  have boundaries : endsPacked = endsReference := funext ends_equal
  funext tokens segments count length hash
  simp [packTrainingCPU, packTrainingReference, boundaries]

def TokenBatch.hashedRows (batch : TokenBatch) : Prop :=
  hashRows batch.inputs = some batch.ngramRows

def TrainingPack.preservesHash (pack : TrainingPack) (hash : RowHash) : Prop :=
  ∀ tokens segments count maxLength batch,
    pack tokens segments count maxLength hash = some batch → batch.hashedRows

private theorem pack_builds_hash (boundaries : List Segment → List Int) :
    TrainingPack.preservesHash (packTrainingBatch boundaries) hashRows := by
  intro tokens segments count maxLength batch packed
  simp only [packTrainingBatch] at packed
  split at packed
  · contradiction
  · split at packed <;> split at packed <;> try contradiction
    all_goals
      simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.some.injEq] at packed
      obtain ⟨rows, hashed, rfl⟩ := packed
      exact hashed

theorem reference_batch_hashed : TrainingPack.preservesHash packTrainingReference hashRows :=
  pack_builds_hash endsReference

theorem candidate_batch_hashed : TrainingPack.preservesHash packTrainingCPU hashRowsCPU := by
  rw [pack_training_equal, hash_rows_cpu_equal]
  exact reference_batch_hashed

def TrainingPack.preservesSize (pack : TrainingPack) : Prop :=
  ∀ tokens segments count maxLength hash batch,
    pack tokens segments count maxLength hash = some batch → batch.inputs.size = count

theorem reference_batch_size : TrainingPack.preservesSize packTrainingReference := by
  intro tokens segments count maxLength hash batch packed
  simp only [packTrainingReference, packTrainingBatch] at packed
  split at packed
  · contradiction
  · rename_i valid
    split at packed <;> split at packed <;> try contradiction
    all_goals
      simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.some.injEq] at packed
      obtain ⟨_, _, rfl⟩ := packed
      have within : count ≤ (concatenateSegments tokens segments).size := by omega
      simp [Array.size_extract, Nat.min_eq_left within]

theorem candidate_batch_size : TrainingPack.preservesSize packTrainingCPU := by
  rw [pack_training_equal]
  exact reference_batch_size

theorem batch_has_in_range_rows (batch : TokenBatch) (hashed : batch.hashedRows)
    (word : UInt32) (inBatch : word ∈ batch.ngramRows) :
    0 ≤ word.toInt32.toInt ∧ word.toInt32.toInt.toNat < 84602880 := by
  have within := hash_rows_range batch.inputs batch.ngramRows hashed word inBatch
  rw [decode_hashed_row word within]
  exact ⟨Int.natCast_nonneg _, within⟩

#axiomsClean pack_training_equal
#axiomsClean reference_batch_hashed
#axiomsClean batch_has_in_range_rows

structure AlignedShard where
  tokens : Array UInt16
  bos : List Nat
  cursor : Nat
  indexed : Bool

def AlignedShard.open (raw : ByteArray) : Option AlignedShard := do
  let tokens ← decodeShard raw
  let bos := (Array.range tokens.size).filter (fun i => tokens[i]?.getD 0 == 50256)
  pure ⟨tokens, bos.toList, 0, false⟩

def AlignedShard.next (s : AlignedShard) (ready : Bool) (count maxLength world : Nat) :
    Option (List (List Segment) × AlignedShard) := do
  let indexed := ready || s.indexed
  let visible := (if indexed then s.bos else s.bos.takeWhile (· < 6000000)).drop s.cursor
  let (segments, rest) ← nextBatch count maxLength s.tokens.size world visible
  pure (segments, {s with cursor := s.cursor + (visible.length - rest.length), indexed})

structure AlignedStream where
  current : AlignedShard
  remaining : List ByteArray
  readiness : List Bool

def AlignedStream.open (files : List ByteArray) (readiness : List Bool) :
    Option AlignedStream := do
  let first :: next :: rest := files | none
  let current ← AlignedShard.open first
  pure ⟨current, next :: rest, readiness⟩

private def alignedNext (world rank count maxLength : Nat) (pack : TrainingPack) (hash : RowHash) :
    AlignedShard → List ByteArray → List Bool → Option (TokenBatch × AlignedStream)
  | shard, raw, signals => do
    let ready :: nextSignals := signals | none
    match shard.next ready count maxLength world with
    | some (segments, current) =>
      let selected ← segments[rank]?
      let batch ← pack shard.tokens selected count maxLength hash
      some (batch, ⟨current, raw, nextSignals⟩)
    | none =>
      let file :: files := raw | none
      let shard ← AlignedShard.open file
      alignedNext world rank count maxLength pack hash shard files nextSignals
termination_by _ raw _ => raw.length

def AlignedStream.next (source : AlignedStream) (world rank global maxLength : Nat)
    (pack : TrainingPack) (hash : RowHash) :
    Option (TokenBatch × AlignedStream) := do
  if rank ≥ world ∨ global % world ≠ 0 then none else
    alignedNext world rank (global / world) maxLength pack hash source.current source.remaining source.readiness

private theorem aligned_next_property {P : TokenBatch → Prop} (world rank count maxLength : Nat) (pack : TrainingPack)
    (hash : RowHash)
    (produces : ∀ tokens segments batch, pack tokens segments count maxLength hash = some batch → P batch)
    (shard : AlignedShard)
    (raw : List ByteArray) (signals : List Bool) (batch : TokenBatch) (next : AlignedStream)
    (step : alignedNext world rank count maxLength pack hash shard raw signals = some (batch, next)) :
    P batch := by
  have packed (source : AlignedShard) {signal signals raw} {segments current}
      (valid : source.next signal count maxLength world = some (segments, current))
      (step : alignedNext world rank count maxLength pack hash source raw (signal :: signals) =
        some (batch, next)) : P batch := by
    rw [alignedNext.eq_def] at step
    simp only [valid, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.some.injEq, Prod.mk.injEq] at step
    obtain ⟨chosen, _, produced, made, rfl, _⟩ := step
    exact produces _ _ produced made
  induction raw generalizing shard signals with
  | nil =>
    cases signals with
    | nil => simp [alignedNext] at step
    | cons signal signals =>
      cases valid : shard.next signal count maxLength world with
      | some available => exact packed shard valid step
      | none => simp [alignedNext, valid] at step
  | cons file rest ih =>
    cases signals with
    | nil => simp [alignedNext] at step
    | cons signal signals =>
      cases valid : shard.next signal count maxLength world with
      | some available => exact packed shard valid step
      | none =>
        rw [alignedNext.eq_def] at step
        simp only [valid, Option.bind_eq_bind, Option.bind_eq_some_iff] at step
        obtain ⟨decoded, _, continues⟩ := step
        exact ih decoded signals continues

theorem next_batch_hashed (source : AlignedStream) (world rank global maxLength : Nat)
    (pack : TrainingPack) (hash : RowHash) (provenance : pack.preservesHash hash)
    (batch : TokenBatch) (next : AlignedStream)
    (step : source.next world rank global maxLength pack hash = some (batch, next)) :
    batch.hashedRows := by
  unfold AlignedStream.next at step
  split at step
  · contradiction
  · exact aligned_next_property _ _ _ _ _ _ (fun _ _ _ made => provenance _ _ _ _ _ made) _ _ _ _ _ step

theorem next_batch_size (source : AlignedStream) (world rank global maxLength : Nat)
    (pack : TrainingPack) (hash : RowHash) (sized : pack.preservesSize)
    (batch : TokenBatch) (next : AlignedStream)
    (step : source.next world rank global maxLength pack hash = some (batch, next)) :
    batch.inputs.size = global / world := by
  unfold AlignedStream.next at step
  split at step
  · contradiction
  · exact aligned_next_property _ _ _ _ _ _ (fun _ _ _ made => sized _ _ _ _ _ _ made) _ _ _ _ _ step

structure TrainingGeometry where
  globalTokens : Nat
  maxLength : Nat

structure ScheduledStream where
  stream : AlignedStream
  worldSize : Nat
  rank : Nat
  pending : List (Nat × TrainingGeometry)
  ahead : List (Nat × TokenBatch)

def ScheduledStream.open (files : List ByteArray) (readiness : List Bool) (worldSize rank : Nat)
    (steps : List (Nat × TrainingGeometry)) : Option ScheduledStream := do
  let stream ← AlignedStream.open files readiness
  some ⟨stream, worldSize, rank, steps, []⟩

private def fetchUntil (step worldSize rank : Nat) (pack : TrainingPack) (hash : RowHash) :
    List (Nat × TrainingGeometry) → AlignedStream → List (Nat × TokenBatch) →
      Option (TokenBatch × List (Nat × TrainingGeometry) × AlignedStream × List (Nat × TokenBatch))
  | [], _, _ => none
  | (next, geometry) :: steps, stream, ahead => do
    if step < next then none else do
      let (batch, stream) ← stream.next worldSize rank geometry.globalTokens geometry.maxLength pack hash
      let ahead := ahead ++ [(next, batch)]
      if next = step then some (batch, steps, stream, ahead)
      else fetchUntil step worldSize rank pack hash steps stream ahead

def ScheduledStream.peek (s : ScheduledStream) (step : Nat) (pack : TrainingPack) (hash : RowHash) :
    Option (TokenBatch × ScheduledStream) :=
  match s.ahead.lookup step with
  | some batch => some (batch, s)
  | none => do
    let (batch, pending, stream, ahead) ← fetchUntil step s.worldSize s.rank pack hash s.pending s.stream s.ahead
    some (batch, {s with pending, stream, ahead})

def ScheduledStream.take (s : ScheduledStream) (step : Nat) (pack : TrainingPack) (hash : RowHash) :
    Option (TokenBatch × ScheduledStream) := do
  let (batch, next) ← s.peek step pack hash
  let (first, _) :: rest := next.ahead | none
  if first = step then some (batch, {next with ahead := rest}) else none

def ScheduledStream.access (s : ScheduledStream) (consume : Bool) (step : Nat)
    (pack : TrainingPack) (hash : RowHash) : Option (TokenBatch × ScheduledStream) :=
  if consume then s.take step pack hash else s.peek step pack hash

def HashedCache (ahead : List (Nat × TokenBatch)) : Prop :=
  ∀ entry ∈ ahead, entry.2.hashedRows

theorem open_cache_hashed (files : List ByteArray) (readiness : List Bool) (worldSize rank : Nat)
    (steps : List (Nat × TrainingGeometry)) (stream : ScheduledStream)
    (opened : ScheduledStream.open files readiness worldSize rank steps = some stream) :
    HashedCache stream.ahead := by
  simp only [ScheduledStream.open, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.some.injEq] at opened
  obtain ⟨_, _, rfl⟩ := opened
  simp [HashedCache]

private theorem fetch_preserves {G : TrainingGeometry → Prop} {B : TokenBatch → Prop}
    (step world rank : Nat) (pack : TrainingPack) (hash : RowHash)
    (produces : ∀ (geometry : TrainingGeometry), G geometry → ∀ source batch next,
      AlignedStream.next source world rank geometry.globalTokens geometry.maxLength pack hash = some (batch, next) →
      B batch)
    (pending : List (Nat × TrainingGeometry)) (stream : AlignedStream) (ahead : List (Nat × TokenBatch))
    (schedule : ∀ entry ∈ pending, G entry.2) (valid : ∀ entry ∈ ahead, B entry.2)
    (batch : TokenBatch) (remaining : List (Nat × TrainingGeometry)) (after : AlignedStream)
    (loaded : List (Nat × TokenBatch))
    (fetch : fetchUntil step world rank pack hash pending stream ahead = some (batch, remaining, after, loaded)) :
    B batch ∧ (∀ entry ∈ remaining, G entry.2) ∧ (∀ entry ∈ loaded, B entry.2) := by
  induction pending generalizing stream ahead with
  | nil => simp [fetchUntil] at fetch
  | cons entry upcoming ih =>
    obtain ⟨key, geometry⟩ := entry
    have head := schedule (key, geometry) (by simp)
    have tail (entry) (member : entry ∈ upcoming) := schedule entry (List.mem_cons_of_mem _ member)
    rw [fetchUntil] at fetch
    split at fetch
    · contradiction
    · simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at fetch
      obtain ⟨⟨generated, forwarded⟩, next, fetch⟩ := fetch
      have sourced := produces geometry head stream generated forwarded next
      have withNew : ∀ entry ∈ ahead ++ [(key, generated)], B entry.2 := by
        intro entry stored
        rcases List.mem_append.mp stored with earlier | single
        · exact valid entry earlier
        · have same := List.mem_singleton.mp single
          simpa only [same] using sourced
      split at fetch
      · simp only [Option.some.injEq, Prod.mk.injEq] at fetch
        obtain ⟨rfl, rfl, _, rfl⟩ := fetch
        exact ⟨sourced, tail, withNew⟩
      · exact ih forwarded _ tail withNew fetch

private theorem peek_preserves {G : TrainingGeometry → Prop} {B : TokenBatch → Prop}
    (stream : ScheduledStream) (step : Nat) (pack : TrainingPack) (hash : RowHash)
    (produces : ∀ (geometry : TrainingGeometry), G geometry → ∀ source batch next,
      AlignedStream.next source stream.worldSize stream.rank geometry.globalTokens geometry.maxLength pack hash =
        some (batch, next) → B batch)
    (schedule : ∀ entry ∈ stream.pending, G entry.2) (valid : ∀ entry ∈ stream.ahead, B entry.2)
    (batch : TokenBatch) (after : ScheduledStream) (peek : stream.peek step pack hash = some (batch, after)) :
    B batch ∧ (∀ entry ∈ after.pending, G entry.2) ∧ (∀ entry ∈ after.ahead, B entry.2) ∧
      after.worldSize = stream.worldSize := by
  unfold ScheduledStream.peek at peek
  cases found : stream.ahead.lookup step with
  | some cached =>
    simp only [found, Option.some.injEq, Prod.mk.injEq] at peek
    obtain ⟨rfl, rfl⟩ := peek
    obtain ⟨front, rest, spread, _⟩ := List.lookup_eq_some_iff.mp found
    have member : (step, cached) ∈ stream.ahead := by rw [spread]; simp
    exact ⟨valid _ member, schedule, valid, rfl⟩
  | none =>
    simp only [found, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.some.injEq, Prod.mk.injEq] at peek
    obtain ⟨⟨loaded, nextSteps, position, cache⟩, fetch, rfl, rfl⟩ := peek
    obtain ⟨loadedOk, pendingOk, aheadOk⟩ := fetch_preserves _ _ _ _ _ produces _ _ _
      schedule valid _ _ _ _ fetch
    exact ⟨loadedOk, pendingOk, aheadOk, rfl⟩

theorem peek_rows_hashed (stream : ScheduledStream) (step : Nat) (pack : TrainingPack) (hash : RowHash)
    (provenance : pack.preservesHash hash) (valid : HashedCache stream.ahead) (batch : TokenBatch)
    (after : ScheduledStream) (peek : stream.peek step pack hash = some (batch, after)) :
    batch.hashedRows ∧ HashedCache after.ahead := by
  have kept := peek_preserves (G := fun _ => True) (B := TokenBatch.hashedRows)
    stream step pack hash (fun g _ src b next h =>
      next_batch_hashed src stream.worldSize stream.rank g.globalTokens g.maxLength pack hash provenance b next h)
    (fun _ _ => trivial) valid batch after peek
  exact ⟨kept.1, kept.2.2.1⟩

def ScheduledStream.Bounded (s : ScheduledStream) (maxTokens : Nat) : Prop :=
  (∀ entry ∈ s.pending, entry.2.globalTokens / s.worldSize ≤ maxTokens) ∧
  (∀ entry ∈ s.ahead, entry.2.inputs.size ≤ maxTokens)

theorem open_bounded (files : List ByteArray) (readiness : List Bool) (world rank maxTokens : Nat)
    (steps : List (Nat × TrainingGeometry)) (stream : ScheduledStream)
    (opened : ScheduledStream.open files readiness world rank steps = some stream)
    (geometry : ∀ entry ∈ steps, entry.2.globalTokens / world ≤ maxTokens) :
    stream.Bounded maxTokens := by
  simp only [ScheduledStream.open, Option.bind_eq_bind, Option.bind_eq_some_iff, Option.some.injEq] at opened
  obtain ⟨_, _, rfl⟩ := opened
  exact ⟨geometry, by simp⟩

theorem peek_bounded (stream : ScheduledStream) (step maxTokens : Nat) (pack : TrainingPack)
    (hash : RowHash) (sized : pack.preservesSize) (valid : stream.Bounded maxTokens)
    (batch : TokenBatch) (after : ScheduledStream) (peek : stream.peek step pack hash = some (batch, after)) :
    batch.inputs.size ≤ maxTokens ∧ after.Bounded maxTokens := by
  have kept := peek_preserves (G := fun g => g.globalTokens / stream.worldSize ≤ maxTokens)
    (B := fun b => b.inputs.size ≤ maxTokens) stream step pack hash
    (fun g bound src b next h => by
      change b.inputs.size ≤ maxTokens
      rw [next_batch_size src stream.worldSize stream.rank g.globalTokens g.maxLength pack hash sized b next h]
      exact bound)
    valid.1 valid.2 batch after peek
  exact ⟨kept.1, fun entry present => by rw [kept.2.2.2]; exact kept.2.1 entry present, kept.2.2.1⟩

theorem take_rows_hashed (stream : ScheduledStream) (step : Nat) (pack : TrainingPack) (hash : RowHash)
    (provenance : pack.preservesHash hash) (valid : HashedCache stream.ahead) (batch : TokenBatch)
    (after : ScheduledStream) (take : stream.take step pack hash = some (batch, after)) :
    batch.hashedRows ∧ HashedCache after.ahead := by
  unfold ScheduledStream.take at take
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at take
  obtain ⟨⟨current, peeked⟩, peek, take⟩ := take
  have preserved := peek_rows_hashed stream step pack hash provenance valid current peeked peek
  cases ex : peeked.ahead with
  | nil => simp [ex] at take
  | cons head rest =>
    obtain ⟨first, token⟩ := head
    simp only [ex] at take
    split at take
    · simp only [Option.some.injEq, Prod.mk.injEq] at take
      obtain ⟨rfl, rfl⟩ := take
      exact ⟨preserved.1, fun entry later => preserved.2 entry (by rw [ex]; exact List.mem_cons_of_mem _ later)⟩
    · contradiction

theorem access_rows_hashed (stream : ScheduledStream) (consume : Bool) (step : Nat)
    (pack : TrainingPack) (hash : RowHash) (provenance : pack.preservesHash hash)
    (valid : HashedCache stream.ahead) (batch : TokenBatch) (after : ScheduledStream)
    (access : stream.access consume step pack hash = some (batch, after)) :
    batch.hashedRows ∧ HashedCache after.ahead := by
  cases consume
  · exact peek_rows_hashed stream step pack hash provenance valid batch after access
  · exact take_rows_hashed stream step pack hash provenance valid batch after access

theorem access_bounded (stream : ScheduledStream) (consume : Bool) (step maxTokens : Nat)
    (pack : TrainingPack) (hash : RowHash) (sized : pack.preservesSize)
    (valid : stream.Bounded maxTokens) (batch : TokenBatch) (after : ScheduledStream)
    (access : stream.access consume step pack hash = some (batch, after)) :
    batch.inputs.size ≤ maxTokens ∧ after.Bounded maxTokens := by
  cases consume
  · exact peek_bounded stream step maxTokens pack hash sized valid batch after access
  · change stream.take step pack hash = some (batch, after) at access
    unfold ScheduledStream.take at access
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at access
    obtain ⟨⟨current, peeked⟩, peek, take⟩ := access
    have kept := peek_bounded stream step maxTokens pack hash sized valid current peeked peek
    cases cache : peeked.ahead with
    | nil => simp [cache] at take
    | cons head tail =>
      obtain ⟨first, token⟩ := head
      simp only [cache] at take
      split at take
      · simp only [Option.some.injEq, Prod.mk.injEq] at take
        obtain ⟨rfl, rfl⟩ := take
        exact ⟨kept.1, kept.2.1, fun entry member =>
          kept.2.2 entry (by rw [cache]; exact List.mem_cons_of_mem _ member)⟩
      · contradiction

#axiomsClean open_cache_hashed
#axiomsClean access_rows_hashed
#axiomsClean next_batch_size
#axiomsClean open_bounded
#axiomsClean access_bounded

theorem peek_cached (s : ScheduledStream) (step : Nat) (pack : TrainingPack) (hash : RowHash)
    (batch : TokenBatch) (present : s.ahead.lookup step = some batch) :
    s.peek step pack hash = some (batch, s) := by
  simp [ScheduledStream.peek, present]

theorem take_cached (s : ScheduledStream) (step : Nat) (pack : TrainingPack) (hash : RowHash)
    (batch : TokenBatch) (rest : List (Nat × TokenBatch)) (first : s.ahead = (step, batch) :: rest) :
    s.take step pack hash = some (batch, {s with ahead := rest}) := by
  have found : s.ahead.lookup step = some batch := by simp [first]
  simp only [ScheduledStream.take, peek_cached _ _ _ _ _ found]
  simp [first]

#axiomsClean peek_cached
#axiomsClean take_cached

end VeriTile.Bench.Models.NanoGPT.Track1.FineWeb
