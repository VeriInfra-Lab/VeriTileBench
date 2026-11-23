import models.NanoGPT.Track1.TailNorm

/-! The replicated Adam tile specialization keeps the original scalar table
and updates the flat buffers in place. Each aligned tile intersects one
parameter interval and receives the segment ID once. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ReplicatedAdam
open VeriTile Triton FP
open TailNorm (F32 zero one)

def blockSize : Nat := 512

structure Segment where
  id : Nat
  width : Nat

structure Span where
  lo : Nat
  hi : Nat
  segment : Nat
  deriving DecidableEq, Repr

def Span.contains (span : Span) (index : Nat) : Bool :=
  decide (span.lo ≤ index ∧ index < span.hi)

def Span.cut (span : Span) (block : Nat) : Span :=
  ⟨max span.lo (block * blockSize), min span.hi ((block + 1) * blockSize), span.segment⟩

private def Span.liveCut (span : Span) (block : Nat) : Option Span :=
  let cut := span.cut block
  if cut.lo < cut.hi then some cut else none

def Span.tiles (span : Span) : List Span :=
  let first := span.lo / blockSize
  let stop := TailNorm.ceilDiv span.hi blockSize
  (List.range' first (stop - first)).filterMap span.liveCut

def referenceSpans (begin : Nat) : List Segment → List Span
  | [] => []
  | segment :: rest =>
    let end_ := begin + segment.width
    ⟨begin, end_, segment.id⟩ :: referenceSpans end_ rest

def candidateSpans (begin : Nat) : List Segment → List Span
  | [] => []
  | segment :: rest =>
    let end_ := begin + segment.width
    (Span.tiles ⟨begin, end_, segment.id⟩) ++ candidateSpans end_ rest

theorem cut_contains (span : Span) (block index : Nat) :
    (span.cut block).contains index ↔
      span.contains index ∧ block * blockSize ≤ index ∧ index < (block + 1) * blockSize := by
  simp only [Span.cut, Span.contains, decide_eq_true_eq]
  omega

theorem live_cut (span : Span) {index : Nat} (live : span.contains index) :
    ∃ cut ∈ span.tiles, cut.contains index := by
  let block := index / blockSize
  have tileLive : (span.cut block).contains index := by
    apply (cut_contains span block index).2
    simp only [block, blockSize]
    exact ⟨live, by omega, by omega⟩
  have inner : (span.cut block).lo < (span.cut block).hi := by
    have := (of_decide_eq_true tileLive)
    omega
  refine ⟨span.cut block, List.mem_filterMap.mpr ⟨block, ?_, ?_⟩, tileLive⟩
  · simp only [List.mem_range'_1, Span.contains, decide_eq_true_eq, TailNorm.ceilDiv, block, blockSize] at *
    omega
  · simp [Span.liveCut, inner]

private theorem liveCut_source (span : Span) (block : Nat) (tile : Span)
    (member : span.liveCut block = some tile) : span.cut block = tile ∧ tile.lo < tile.hi := by
  simp only [Span.liveCut] at member
  split at member
  next live =>
    have same := Option.some.inj member
    exact ⟨same, same ▸ live⟩
  · contradiction

theorem tile_source (span tile : Span) (h : tile ∈ span.tiles) :
    ∃ block, span.cut block = tile := by
  obtain ⟨block, _, member⟩ := List.mem_filterMap.mp h
  exact ⟨block, (liveCut_source span block tile member).1⟩

theorem tiles_nodup (span : Span) : span.tiles.Nodup := by
  unfold Span.tiles
  refine List.Pairwise.filterMap span.liveCut (l := _) ?_ (List.nodup_range' 1)
  intro i j different a ha b hb same
  subst b
  obtain ⟨left, nonempty⟩ := liveCut_source span i a ha
  obtain ⟨right, _⟩ := liveCut_source span j a hb
  have li : (span.cut i).contains a.lo := by simp [left, Span.contains, nonempty]
  have lj : (span.cut j).contains a.lo := by simp [right, Span.contains, nonempty]
  obtain ⟨_, ai, bi⟩ := (cut_contains span i a.lo).1 li
  obtain ⟨_, aj, bj⟩ := (cut_contains span j a.lo).1 lj
  simp only [blockSize] at *
  omega

theorem same_tile_owner (span a b : Span) {index : Nat}
    (ha : a ∈ span.tiles) (hb : b ∈ span.tiles)
    (la : a.contains index) (lb : b.contains index) : a = b := by
  obtain ⟨i, rfl⟩ := tile_source span a ha
  obtain ⟨j, rfl⟩ := tile_source span b hb
  obtain ⟨_, ai, bi⟩ := (cut_contains span i index).1 la
  obtain ⟨_, aj, bj⟩ := (cut_contains span j index).1 lb
  simp only [blockSize] at *
  have : i = j := by omega
  rw [this]

theorem live_lane (span tile : Span) (h : tile ∈ span.tiles) {index : Nat}
    (active : tile.contains index) :
    ∃ lane < blockSize, tile.lo / blockSize * blockSize + lane = index := by
  obtain ⟨block, rfl⟩ := tile_source span tile h
  have ⟨_, lb, ub⟩ := (cut_contains span block index).1 active
  have bounds := of_decide_eq_true active
  simp only [Span.cut, blockSize] at *
  refine ⟨index % 512, by omega, ?_⟩
  omega

theorem tiles_start_at (segments : List Segment) (begin : Nat) {tile : Span}
    (h : tile ∈ candidateSpans begin segments) : begin ≤ tile.lo := by
  induction segments generalizing begin with
  | nil => contradiction
  | cons s rest ih =>
    simp only [candidateSpans, List.mem_append] at h
    rcases h with h | h
    · obtain ⟨block, rfl⟩ := tile_source _ tile h
      exact Nat.le_max_left ..
    · exact Nat.le_trans (Nat.le_add_right ..) (ih (begin + s.width) h)

theorem unique_store_owner (segments : List Segment) (begin index : Nat) {a b : Span}
    (ha : a ∈ candidateSpans begin segments) (hb : b ∈ candidateSpans begin segments)
    (la : a.contains index) (lb : b.contains index) : a = b := by
  induction segments generalizing begin with
  | nil => contradiction
  | cons segment rest ih =>
    let first : Span := ⟨begin, begin + segment.width, segment.id⟩
    change a ∈ first.tiles ++ candidateSpans first.hi rest at ha
    change b ∈ first.tiles ++ candidateSpans first.hi rest at hb
    have below {x : Span} (m : x ∈ first.tiles) (live : x.contains index) : index < first.hi := by
      obtain ⟨block, rfl⟩ := tile_source first x m
      exact (of_decide_eq_true ((cut_contains first block index).1 live).1).2
    have above {x : Span} (m : x ∈ candidateSpans first.hi rest)
        (live : x.contains index) : first.hi ≤ index :=
      Nat.le_trans (tiles_start_at rest first.hi m) (of_decide_eq_true live).1
    rcases List.mem_append.mp ha with ha | ha <;> rcases List.mem_append.mp hb with hb | hb
    · exact same_tile_owner first a b ha hb la lb
    · exact False.elim ((Nat.not_le_of_lt (below ha la)) (above hb lb))
    · exact False.elim ((Nat.not_le_of_lt (below hb lb)) (above ha la))
    · exact ih first.hi ha hb

theorem candidate_spans_nodup (segments : List Segment) (begin : Nat) :
    (candidateSpans begin segments).Nodup := by
  induction segments generalizing begin with
  | nil => exact List.nodup_nil
  | cons segment rest ih =>
    let first : Span := ⟨begin, begin + segment.width, segment.id⟩
    change (first.tiles ++ candidateSpans first.hi rest).Nodup
    refine List.nodup_append.mpr ⟨tiles_nodup first, ih first.hi, ?_⟩
    intro a ha b hb same
    subst b
    obtain ⟨block, _, selected⟩ := List.mem_filterMap.mp ha
    obtain ⟨cut, nonempty⟩ := liveCut_source first block a selected
    have upper : a.hi ≤ first.hi := by
      rw [← cut]
      exact Nat.min_le_left ..
    have lower := tiles_start_at rest first.hi hb
    omega

def owner (spans : List Span) (index : Nat) : Option Nat :=
  (spans.find? (·.contains index)).map (·.segment)

theorem tile_owner (span : Span) (index : Nat) :
    owner span.tiles index = if span.contains index then some span.segment else none := by
  unfold owner
  split
  next live =>
    obtain ⟨cut, mem, contains⟩ := live_cut span live
    cases found : span.tiles.find? (·.contains index) with
    | none => exact False.elim ((List.find?_eq_none.mp found cut mem) contains)
    | some tile =>
      obtain ⟨block, rfl⟩ := tile_source span tile (List.mem_of_find?_eq_some found)
      rfl
  next dead =>
    have noneFound : span.tiles.find? (·.contains index) = none :=
      List.find?_eq_none.mpr fun tile mem live => by
        obtain ⟨block, rfl⟩ := tile_source span tile mem
        exact dead ((cut_contains span block index).1 live).1
    simp [noneFound]

theorem owners_equal (segments : List Segment) (begin index : Nat) :
    owner (candidateSpans begin segments) index = owner (referenceSpans begin segments) index := by
  induction segments generalizing begin with
  | nil => rfl
  | cons segment rest ih =>
    let first : Span := ⟨begin, begin + segment.width, segment.id⟩
    have appended (a b : List Span) :
        owner (a ++ b) index = (owner a index).or (owner b index) := by
      simp only [owner, List.find?_append]
      cases a.find? (·.contains index) <;> rfl
    change owner (first.tiles ++ candidateSpans first.hi rest) index =
      owner ([first] ++ referenceSpans first.hi rest) index
    rw [appended, appended, tile_owner, ih]
    simp [owner, List.find?_cons]
    cases first.contains index <;> rfl

structure Scalars where
  beta1 : F32
  oneMinusBeta1 : F32
  beta2 : F32
  oneMinusBeta2 : F32
  eps : F32
  step : F32
  decay : F32
  active : F32

structure Word (format : Format) where
  parameter : Value format
  first : F32
  second : F32

structure Native where
  sqrtApproxFtz : F32 → F32
  divideFull : F32 → F32 → F32

/-- The arithmetic emitted by both Triton kernels. The second-moment
product rounds before its FMA. The cautious decay has its own ordered
test and FMA before the parameter store. -/
def update {format : Format} (native : Native) (s : Scalars)
    (gradient : Value format) (old : Word format) : Word format :=
  let c := Config.ieeeValue
  if FP.eq c s.active zero then old else
    let g := FP.cast .fp32 c gradient
    let p := FP.cast .fp32 c old.parameter
    let first := FP.fma c s.oneMinusBeta1 g (FP.mul c old.first s.beta1)
    let second := FP.fma c (FP.mul c s.oneMinusBeta2 g) g (FP.mul c old.second s.beta2)
    let direction := native.divideFull first (FP.add c (native.sqrtApproxFtz second) s.eps)
    let delta := FP.mul c direction s.step
    let gate := if FP.lt c zero (FP.mul c delta p) then one else zero
    let delta := FP.fma c (FP.mul c s.decay p) gate delta
    ⟨FP.cast format c (FP.sub c p delta), first, second⟩

abbrev Step := {format : Format} → Native → List Segment → (Nat → Scalars) →
  (Nat → Value format) → (Nat → Word format) → Nat → Word format

def stepWith (layout : Nat → List Segment → List Span) : Step :=
  fun native sizes scalars gradient words index =>
    match owner (layout 0 sizes) index with
    | none => words index
    | some segment => update native (scalars segment) (gradient index) (words index)

def reference : Step := stepWith referenceSpans
def candidate : Step := stepWith candidateSpans

theorem step_equal : @candidate = @reference := by
  funext format native sizes scalars gradient words index
  simp only [candidate, reference, stepWith, owners_equal]

structure Snapshot where
  format : Format
  native : Native
  sizes : List Segment
  scalars : Nat → Scalars
  gradient : Nat → Value format
  words : Nat → Word format
  eligible : Bool

def Snapshot.run (step : Step) (input : Snapshot) : Nat → Word input.format :=
  let run := if input.eligible then @step input.format else @reference input.format
  run input.native input.sizes input.scalars input.gradient input.words

structure Output where
  format : Format
  words : Nat → Word format

structure Runtime (State : Type) where
  read : State → Option (List Snapshot × State)
  write : List Output → State → Option State

def execute {State : Type} (runtime : Runtime State) (step : Step) (state : State) : Option State := do
  let (inputs, next) ← runtime.read state
  runtime.write (inputs.map fun input => ⟨input.format, input.run @step⟩) next

#axiomsClean live_cut
#axiomsClean same_tile_owner
#axiomsClean live_lane
#axiomsClean unique_store_owner
#axiomsClean candidate_spans_nodup
#axiomsClean owners_equal
#axiomsClean step_equal

end VeriTile.Bench.Models.NanoGPT.Track1.ReplicatedAdam
