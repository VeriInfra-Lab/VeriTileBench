import models.NanoGPT.Track1.AdamEntries
import models.NanoGPT.Track1.MomentReplay
import models.NanoGPT.Track1.HostFloat64
import models.NanoGPT.Track1.SparsePackets

/-! Numerical n-gram Adam event. Source and candidate use the same
ordered scatter schedules. The two index packers supply their arrays;
the remainder of the event is computed from the actual gradient payloads.
The persistent optimizer state is independent of the claim order. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.NgramAdam
open VeriTile.Triton FP
open SparsePackets (Channel)

abbrev F32 := Value .fp32
abbrev BF16 := Value .bf16

structure Pull where
  ids : AdamEntries.Input
  ownGradient : Fin ids.localCount → Channel → UInt16
  peerGradient : Fin ids.peerCount → Channel → BF16

/-- A schedule includes every entry. The claim uses a single writer order.
Atomic order is observed per BF16 destination channel. -/
structure Order (n : Nat) where
  slots : List (Fin n)
  allSlots : slots.Perm (List.finRange n)

structure Schedule (p : Pull) where
  claims : Order p.ids.count
  localAtomics : Channel → Order p.ids.localCount
  peerAtomics : Channel → Order p.ids.peerCount

abbrev Claims (n : Nat) := Int32 → Option (Fin n)

def claim {p : Pull} (rows : AdamEntries.Indices p.ids) (s : Schedule p) : Claims p.ids.count :=
  s.claims.slots.foldl (fun bank i row => if rows.entries i = row then some i else bank row)
    (fun _ => none)

private theorem claimFold {p : Pull} (rows : AdamEntries.Indices p.ids) (row : Int32)
    (slots : List (Fin p.ids.count)) (initial : Claims p.ids.count)
    (valid : ∀ i, initial row = some i → rows.entries i = row) :
    let result := (slots.foldl (fun bank i r => if rows.entries i = r then some i else bank r) initial) row
    (∀ i, result = some i → rows.entries i = row) ∧
      (result = none ↔ initial row = none ∧ ∀ i ∈ slots, rows.entries i ≠ row) := by
  induction slots generalizing initial with
  | nil => exact ⟨valid, by simp⟩
  | cons i rest ih =>
    let next : Claims p.ids.count := fun r => if rows.entries i = r then some i else initial r
    have nextValid : ∀ j, next row = some j → rows.entries j = row := by
      intro j chosen
      by_cases hit : rows.entries i = row
      · have same : i = j := by simpa [next, hit] using chosen
        exact same ▸ hit
      · exact valid j (by simpa [next, hit] using chosen)
    obtain ⟨kept, absent⟩ := ih next nextValid
    change (∀ j, _ = some j → rows.entries j = row) ∧
      (_ = none ↔ initial row = none ∧ ∀ j ∈ i :: rest, rows.entries j ≠ row)
    rw [List.foldl_cons, absent]
    refine ⟨kept, ?_⟩
    by_cases hit : rows.entries i = row <;> simp [next, hit]

theorem claim_spec {p : Pull} (rows : AdamEntries.Indices p.ids) (s : Schedule p) (row : Int32) :
    (∀ i, claim rows s row = some i → rows.entries i = row) ∧
      (claim rows s row = none ↔ ∀ i, rows.entries i ≠ row) := by
  obtain ⟨sound, absent⟩ := claimFold rows row s.claims.slots (fun _ => none) (by simp)
  refine ⟨sound, ?_⟩
  show claim rows s row = none ↔ _
  rw [claim, absent]
  simp [s.claims.allSlots.mem_iff]

/-- These are the CTAs that read or write persistent row state. Every
CTA reads its entry and the claim before checking this guard. -/
def issuedRow {p : Pull} (rows : AdamEntries.Indices p.ids) (claims : Claims p.ids.count)
    (slot : Fin p.ids.count) : Option Int32 :=
  let row := rows.entries slot
  if claims row = some slot then some row else none

theorem issued_row_exact {p : Pull} (rows : AdamEntries.Indices p.ids) (s : Schedule p)
    (slot : Fin p.ids.count) (row : Int32) : issuedRow rows (claim rows s) slot = some row ↔
      claim rows s row = some slot := by
  constructor
  · intro access
    change (if claim rows s (rows.entries slot) = some slot then some (rows.entries slot) else none) =
      some row at access
    split at access
    · rename_i chosen
      simpa only [Option.some.inj access] using chosen
    · contradiction
  · intro chosen
    simp [issuedRow, (claim_spec rows s row).1 slot chosen, chosen]

theorem issued_rows_disjoint {p : Pull} (rows : AdamEntries.Indices p.ids) (s : Schedule p)
    (a b : Fin p.ids.count) (row : Int32) (ha : issuedRow rows (claim rows s) a = some row)
    (hb : issuedRow rows (claim rows s) b = some row) : a = b :=
  Option.some.inj (((issued_row_exact rows s a row).mp ha).symm.trans ((issued_row_exact rows s b row).mp hb))

/-- A chosen slot receives exactly the gradients for its row, for any
allowed claim order. -/
private theorem claim_routes {p : Pull} (rows : AdamEntries.Indices p.ids) (s : Schedule p)
    (row query : Int32) (slot : Fin p.ids.count) (chosen : claim rows s row = some slot) :
    claim rows s query = some slot ↔ query = row := by
  constructor
  · intro routed
    exact ((claim_spec rows s query).1 slot routed).symm.trans ((claim_spec rows s row).1 slot chosen)
  · rintro rfl
    exact chosen

abbrev OwnedWords := (p : Pull) → Fin p.ids.localCount → Channel → UInt16
abbrev PeerWords := (p : Pull) → Fin p.ids.peerCount → Channel → BF16

def merge {p : Pull} (rows : AdamEntries.Indices p.ids) (s : Schedule p)
    (owner : Claims p.ids.count) (initial : Fin p.ids.count → Channel → BF16)
    (slot : Fin p.ids.count) (col : Channel) (ownedWords : OwnedWords := Pull.ownGradient)
    (peerWords : PeerWords := Pull.peerGradient) : BF16 :=
  let localSum := (s.localAtomics col).slots.foldl (fun acc i =>
    if owner (rows.localRows i) = some slot then
      FP.add Config.ieeeValue acc (SparsePackets.wireCast (ownedWords p i col))
    else acc) (initial slot col)
  (s.peerAtomics col).slots.foldl (fun acc i =>
    if owner (rows.peerRows i) = some slot then FP.add Config.ieeeValue acc (peerWords p i col)
    else acc) localSum

structure Prepared (p : Pull) where
  indices : AdamEntries.Indices p.ids
  claims : Claims p.ids.count
  initial : Fin p.ids.count → Channel → BF16

abbrev Prepare := (p : Pull) → Schedule p → Prepared p

def prepareStaged (p : Pull) (indices : AdamEntries.Indices p.ids) (schedule : Schedule p) : Prepared p :=
  ⟨indices, claim indices schedule, fun _ _ => Value.zero .bf16⟩

def stagedPrepare : Prepare := fun p schedule => prepareStaged p (AdamEntries.staged p.ids) schedule

/-- Each exchange publishes an entry that the same lane packed. The
allowed last-writer order is the existing claim order. The vector stores
initialize every gradient element before either scatter. -/
def fusedPrepare : Prepare := fun p schedule =>
  ⟨AdamEntries.fused p.ids,
    schedule.claims.slots.foldl (fun bank i r =>
      if AdamEntries.packedRead p.ids i = r then some i else bank r) (fun _ => none),
    fun slot col => Value.ofBits .bf16 (((AdamEntries.gradientWord slot col).map Prod.snd).getD 0).toNat⟩

theorem prepare_equal : fusedPrepare = stagedPrepare := by
  funext p schedule
  have cleared : (fusedPrepare p schedule).initial = (stagedPrepare p schedule).initial := by
    funext slot col
    simp only [fusedPrepare, stagedPrepare, prepareStaged, AdamEntries.gradient_word_exact,
      Option.map_some, Option.getD_some]
    rfl
  have sameClaims : (fusedPrepare p schedule).claims = (stagedPrepare p schedule).claims := by
    change claim (AdamEntries.fused p.ids) schedule = claim (AdamEntries.staged p.ids) schedule
    rw [AdamEntries.pack_equal]
  have indices := congrFun AdamEntries.pack_equal p.ids
  change Prepared.mk _ _ _ = Prepared.mk _ _ _
  exact Prepared.mk.injEq .. |>.mpr ⟨indices, sameClaims, cleared⟩

structure Launch where
  event : Nat
  beta2 : F32
  eps : F32
  stepSize : F32
  decay : F32
  gradMul : F32
  squareMul : F32

structure Scalars where
  events : Nat
  rate : Float
  weightDecay : Float
  eps : Float
  worldSize : Nat

private def launchFloat (x : Float) : F32 := Value.ofBits .fp32 x.toFloat32.toBits.toNat

/-- Decode the pinned positive normal binary64 constant and round the two
beta2 expressions as the host's FP64 calculation and FP32 store do. -/
private def roundedBeta (period4 : Bool) : F32 :=
  let bits := Driver.sparseBeta2.toNat
  let mantissa : Nat := 2 ^ 52 + bits % (2 ^ 52)
  let exponent : Int := (bits / (2 ^ 52) % 2048 : Nat) - 1075
  let b := (mantissa : Rat) * FP.pow2 exponent
  let beta := if period4 then HostFloat64.round (b * b) else b
  FP.round .fp32 Config.ieeeValue beta

theorem rounded_beta_exact (period4 : Bool) : roundedBeta period4 = MomentReplay.betaWord period4 := by
  cases period4 <;> decide +kernel

def betaForStep (step : Nat) : F32 := roundedBeta (decide (step ≥ Driver.sparsePeriod4Start))

/-- CPython computes the coefficients in binary64 before launching the
binary32 kernel. The step uses the post-increment Adam event number. -/
def configure (step : Nat) (s : Scalars) : Launch :=
  let event := s.events + 1
  let b := Float.ofBits Driver.sparseBeta2
  let period4 := step ≥ Driver.sparsePeriod4Start
  let beta := if period4 then b.pow 2.0 else b
  let wdScale := Float.ofBits (if period4 then Driver.sparsePeriod4WeightDecayScale
                               else Driver.sparseWeightDecayScale)
  let lr := s.rate * Float.ofBits Driver.sparseRateScale
  ⟨event, betaForStep step, launchFloat s.eps,
    launchFloat (lr * (1.0 - beta.pow event.toFloat).pow 0.5),
    launchFloat (lr * lr * s.weightDecay * wdScale),
    launchFloat (1.0 / s.worldSize.toFloat),
    launchFloat ((1.0 - beta) / (s.worldSize ^ 2).toFloat)⟩

structure Row where
  weights : Channel → BF16
  moment : F32
  stamp : Nat

structure Memory where
  rows : Int32 → Row
  betaHistory : Nat → F32

def publishBeta (before : Nat → F32) (event : Nat) (beta : F32) (i : Nat) : F32 :=
  if i = event then beta else before i

def publishedHistory (launch : Launch) (previous : Memory) : Nat → F32 :=
  publishBeta previous.betaHistory launch.event launch.beta2

theorem publish_beta_positive (step events : Nat) (before : Nat → F32)
    (earlier : MomentReplay.PositiveHistory before events) :
    MomentReplay.PositiveHistory (publishBeta before (events + 1) (betaForStep step)) (events + 1) := by
  intro idx positive inside
  by_cases latest : idx = events + 1
  · refine ⟨decide (step ≥ Driver.sparsePeriod4Start), ?_⟩
    simpa [publishBeta, betaForStep, latest] using rounded_beta_exact
      (decide (step ≥ Driver.sparsePeriod4Start))
  · simpa only [publishBeta, if_neg latest] using earlier idx positive (by omega)

theorem configured_history (step : Nat) (scalars : Scalars) (previous : Memory)
    (earlier : MomentReplay.PositiveHistory previous.betaHistory scalars.events) :
    MomentReplay.PositiveHistory (publishedHistory (configure step scalars) previous)
      (scalars.events + 1) := publish_beta_positive step scalars.events previous.betaHistory earlier

/-- The event counter restarts at zero at reset. The backing allocation
can hold arbitrary words from warmup above the published prefix. -/
def publishSteps (event : Nat) (history : Nat → F32) : List Nat → (Nat → F32)
  | [] => history
  | step :: later => publishSteps (event + 1) (publishBeta history (event + 1) (betaForStep step)) later

theorem published_steps_positive (steps : List Nat) (event : Nat) (history : Nat → F32)
    (earlier : MomentReplay.PositiveHistory history event) :
    MomentReplay.PositiveHistory (publishSteps event history steps) (event + steps.length) := by
  induction steps generalizing event history with
  | nil => simpa [publishSteps] using earlier
  | cons step later ih =>
    have next := ih (event + 1) _ (publish_beta_positive step event history earlier)
    simpa only [publishSteps, List.length_cons, show event + 1 + later.length = event + (later.length + 1) by omega]
      using next

theorem published_since_reset (steps : List Nat) (memory : Nat → F32) :
    MomentReplay.PositiveHistory (publishSteps 0 memory steps) steps.length := by
  have empty : MomentReplay.PositiveHistory memory 0 := fun _ positive bound => by omega
  simpa only [Nat.zero_add] using published_steps_positive steps 0 memory empty

/-- The unchanged PTX hardware scalar operations. Their exact result
is supplied by the execution profile. All other operations below use
the software floating-point model with the emitted contraction tree. -/
structure Native where
  divFull : F32 → F32 → F32
  sqrtApproxFTZ : F32 → F32

private def widen (x : BF16) : F32 := FP.cast .fp32 Config.ieeeValue x

def butterfly (x : Nat → F32) (mask : Nat) (lane : Nat) : F32 :=
  FP.add Config.ieeeValue (x lane) (x (lane ^^^ mask))

def threadSquares (gradient : Channel → BF16) (thread : Nat) : F32 :=
  let g := fun k : Nat =>
    if inside : thread * 8 + k < 768 then widen (gradient ⟨thread * 8 + k, inside⟩)
    else Value.zero .fp32
  let g₁ := g 1
  [0, 2, 3, 4, 5, 6, 7].foldl
    (fun acc k => FP.fma Config.ieeeValue (g k) (g k) acc) (FP.mul Config.ieeeValue g₁ g₁)

def sumSquares (gradient : Channel → BF16) : F32 :=
  let warps := fun warp : Nat =>
    ([16, 8, 4, 2, 1].foldl butterfly (fun lane => threadSquares gradient (warp * 32 + lane))) 0
  ([2, 1].foldl butterfly warps) 0

private def rowUpdateFrom (native : Native) (launch : Launch) (previous : F32)
    (old : Row) (gradient : Channel → BF16) : Row :=
  let mean := native.divFull (sumSquares gradient) (Value.ofBits .fp32 0x44400000)
  let contribution := FP.mul Config.ieeeValue launch.squareMul mean
  let moment := FP.fma Config.ieeeValue launch.beta2 previous contribution
  let denom := FP.add Config.ieeeValue launch.eps (native.sqrtApproxFTZ moment)
  let weights := fun col =>
    let p := widen (old.weights col)
    let m := FP.mul Config.ieeeValue launch.gradMul (widen (gradient col))
    let direction := native.divFull m denom
    let scaled := FP.mul Config.ieeeValue launch.stepSize direction
    let gate := FP.lt Config.ieeeValue (Value.zero .fp32) (FP.mul Config.ieeeValue scaled p)
    let decay := if gate then FP.mul Config.ieeeValue launch.decay p else Value.zero .fp32
    let update := FP.fma Config.ieeeValue launch.stepSize direction decay
    FP.cast .bf16 Config.ieeeValue (FP.sub Config.ieeeValue p update)
  ⟨weights, moment, launch.event⟩

abbrev RowKernel := Native → Launch → (Nat → F32) → Row → (Channel → BF16) → Row

def rowUpdate (native : Native) (launch : Launch) (beta : Nat → F32)
    (old : Row) (gradient : Channel → BF16) : Row :=
  rowUpdateFrom native launch
    (MomentReplay.sequential beta old.stamp old.moment (launch.event - 1 - old.stamp)) old gradient

def rowUpdateSkippingZero (native : Native) (launch : Launch) (beta : Nat → F32)
    (old : Row) (gradient : Channel → BF16) : Row :=
  rowUpdateFrom native launch (MomentReplay.skipZero beta old.stamp old.moment (launch.event - 1)) old gradient

theorem row_update_skip_exact (native : Native) (launch : Launch) (beta : Nat → F32)
    (old : Row) (gradient : Channel → BF16) (history : MomentReplay.PositiveHistory beta (launch.event - 1)) :
    rowUpdateSkippingZero native launch beta old gradient = rowUpdate native launch beta old gradient := by
  rw [rowUpdateSkippingZero, rowUpdate, MomentReplay.skip_zero_exact _ _ _ _ history]

def selectedRowUpdate : RowKernel := fun native launch beta row gradient =>
  if MomentReplay.holdsHistory beta (launch.event - 1) then
    rowUpdateSkippingZero native launch beta row gradient else rowUpdate native launch beta row gradient

theorem selected_row_update_equal : selectedRowUpdate = rowUpdate := by
  funext native launch beta row gradient
  unfold selectedRowUpdate
  split
  · rename_i valid
    exact row_update_skip_exact _ _ _ _ _ ((MomentReplay.holds_history_iff ..).mp (by simpa using valid))
  · rfl

structure Output (p : Pull) where
  state : Memory
  claims : Claims p.ids.count
  gradient : Fin p.ids.count → Channel → BF16

def eventPreparedWith (update : RowKernel)
    (p : Pull) (prepared : Prepared p) (schedule : Schedule p)
    (native : Native) (launch : Launch) (previous : Memory)
    (ownedWords : OwnedWords := Pull.ownGradient) (peerWords : PeerWords := Pull.peerGradient) : Output p :=
  let beta := publishedHistory launch previous
  let claimed := prepared.claims
  let gradient := fun slot col => merge prepared.indices schedule claimed prepared.initial slot col ownedWords peerWords
  let updated := fun row =>
    match claimed row with
    | none => previous.rows row
    | some slot => update native launch beta (previous.rows row) (gradient slot)
  ⟨⟨updated, beta⟩, claimed, gradient⟩

def eventPrepared := eventPreparedWith rowUpdate

theorem prepared_skip_exact (p : Pull) (prepared : Prepared p) (schedule : Schedule p)
    (native : Native) (launch : Launch) (previous : Memory)
    (history : MomentReplay.PositiveHistory previous.betaHistory (launch.event - 1)) :
    eventPreparedWith rowUpdateSkippingZero p prepared schedule native launch previous =
      eventPrepared p prepared schedule native launch previous := by
  have usable : MomentReplay.PositiveHistory (publishedHistory launch previous) (launch.event - 1) := by
    intro i positive bound
    have past : i ≠ launch.event := by omega
    simpa only [publishedHistory, publishBeta, if_neg past] using history i positive bound
  simp only [eventPrepared, eventPreparedWith]
  congr 2
  funext row
  cases prepared.claims row with
  | none => rfl
  | some slot => exact row_update_skip_exact _ _ _ _ _ usable

def event (p : Pull) (indices : AdamEntries.Indices p.ids) (schedule : Schedule p)
    (native : Native) (launch : Launch) (previous : Memory) : Output p :=
  eventPrepared p (prepareStaged p indices schedule) schedule native launch previous

theorem event_equal (p : Pull) (schedule : Schedule p) (native : Native) (launch : Launch)
    (previous : Memory) :
    event p (AdamEntries.fused p.ids) schedule native launch previous =
      event p (AdamEntries.staged p.ids) schedule native launch previous := by
  rw [AdamEntries.pack_equal]

theorem prepared_event_equal (p : Pull) (schedule : Schedule p) (native : Native) (launch : Launch)
    (previous : Memory) :
    eventPrepared p (fusedPrepare p schedule) schedule native launch previous =
      event p (AdamEntries.staged p.ids) schedule native launch previous := by
  rw [prepare_equal]
  rfl

private theorem event_claim_order (p : Pull) (indices : AdamEntries.Indices p.ids) (a b : Schedule p)
    (localOrder : a.localAtomics = b.localAtomics) (peerOrder : a.peerAtomics = b.peerAtomics)
    (native : Native) (launch : Launch) (previous : Memory) :
    (event p indices a native launch previous).state = (event p indices b native launch previous).state := by
  simp only [event, eventPrepared, eventPreparedWith, prepareStaged, Memory.mk.injEq, and_true]
  funext row
  cases chosenA : claim indices a row with
  | none =>
    have chosenB := (claim_spec indices b row).2.mpr ((claim_spec indices a row).2.mp chosenA)
    rw [chosenB]
  | some slotA =>
    have present : claim indices b row ≠ none := by
      intro missing
      exact (claim_spec indices b row).2.mp missing slotA ((claim_spec indices a row).1 slotA chosenA)
    obtain ⟨slotB, chosenB⟩ := Option.ne_none_iff_exists'.mp present
    rw [chosenB]
    refine congrArg (rowUpdate native launch _ (previous.rows row)) ?_
    funext col
    simp only [merge, claim_routes indices a row _ slotA chosenA,
      claim_routes indices b row _ slotB chosenB, localOrder, peerOrder]

/-- The two kernels can elect different entries for the same row.
At fixed per-channel scatter order the persisted weights, moments,
event stamps and beta history agree, irrespective of those elections. -/
theorem prepared_state_equal (p : Pull) (a b : Schedule p)
    (localOrder : a.localAtomics = b.localAtomics) (peerOrder : a.peerAtomics = b.peerAtomics)
    (native : Native) (launch : Launch) (previous : Memory) :
    (eventPrepared p (fusedPrepare p a) a native launch previous).state =
      (event p (AdamEntries.staged p.ids) b native launch previous).state := by
  rw [prepared_event_equal]
  exact event_claim_order p _ a b localOrder peerOrder native launch previous

theorem memory_outside (p : Pull) (indices : AdamEntries.Indices p.ids) (schedule : Schedule p)
    (native : Native) (launch : Launch) (previous : Memory) (row : Int32)
    (outside : ∀ i, indices.entries i ≠ row) :
    (event p indices schedule native launch previous).state.rows row = previous.rows row := by
  have absent := (claim_spec indices schedule row).2.mpr outside
  simp [event, eventPrepared, eventPreparedWith, prepareStaged, absent]

/-- Wait and clear the in-flight record, evaluate the pinned Python
launch scalars, and commit the modeled scatter and row update. -/
structure Runtime (State : Type) where
  wait : State → Option (Pull × State)
  readScalars : State → Option (Scalars × Memory)
  order : (p : Pull) → State → Option (Schedule p)
  native : Native
  commit : (p : Pull) → Launch → Output p → State → Option State

def execute {State : Type} (r : Runtime State) (prepare : Prepare) (step : Nat) (state : State)
    (update : RowKernel := rowUpdate) (ownedWords : OwnedWords := Pull.ownGradient)
    (peerWords : PeerWords := Pull.peerGradient) :
    Option State := do
  let (pull, ready) ← r.wait state
  let (scalars, previous) ← r.readScalars ready
  let launch := configure step scalars
  let order ← r.order pull ready
  r.commit pull launch (eventPreparedWith update pull (prepare pull order) order r.native launch previous ownedWords peerWords) ready

#axiomsClean event_equal
#axiomsClean prepare_equal
#axiomsClean prepared_event_equal
#axiomsClean prepared_state_equal
#axiomsClean configured_history
#axiomsClean issued_rows_disjoint
#axiomsClean published_since_reset
#axiomsClean prepared_skip_exact
#axiomsClean memory_outside
end VeriTile.Bench.Models.NanoGPT.Track1.NgramAdam
