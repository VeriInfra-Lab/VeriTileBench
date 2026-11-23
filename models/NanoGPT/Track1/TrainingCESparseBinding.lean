import VeriTile.Triton.Float.ScalarOpsLaws
import models.NanoGPT.Track1.TrainingCELossBinding
import Mathlib.Data.Nat.Find

/-! The sparse CE epilogue reads at most three live MTP targets and one
prefix target. The interpreter tracks their equality and lane ownership,
and records the exact ordered expression and address of each FP8 store. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCESparseBinding
open TrainingCE TrainingCEGatherBinding TrainingCEPublishBinding TrainingCEConversions
open VeriTile.Triton FP

structure Quad (α : Type) where
  zero : α
  one : α
  two : α
  three : α

def Quad.at (q : Quad α) (i : Fin 4) : α :=
  match i.val with
  | 0 => q.zero
  | 1 => q.one
  | 2 => q.two
  | _ => q.three

def Quad.ofFn (f : Fin 4 → α) : Quad α := ⟨f 0, f 1, f 2, f 3⟩

theorem Quad.at_ofFn (f : Fin 4 → α) (i : Fin 4) : (Quad.ofFn f).at i = f i := by
  have : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by
    simp only [Fin.ext_iff]
    have := i.isLt
    omega
  rcases this with h | h | h | h <;> subst i <;> rfl

private def quads (xs : List α) : List (Quad α) :=
  xs.flatMap fun a => xs.flatMap fun b => xs.flatMap fun c => xs.map fun d => ⟨a, b, c, d⟩

private theorem mem_quads (xs : List α) (all : ∀ a, a ∈ xs) (q : Quad α) : q ∈ quads xs := by
  cases q
  simp [quads, all]

def slotIndex : TargetSlot → Fin 4
  | .mtp k => if h : k < 3 then ⟨k, by omega⟩ else 3
  | .prefix => 3

structure Path where
  live : Fin 4
  hasPrefix : Bool
  prefixPositive : Bool
  laneLast : Bool
  labels : Quad (Fin 4)
  owners : Quad Bool

def Path.valid (path : Path) : TargetSlot → Bool
  | .mtp k => decide (k < path.live.val)
  | .prefix => path.hasPrefix

def Path.targetClass (path : Path) (slot : TargetSlot) : Option (Fin 4) :=
  if path.valid slot then some (path.labels.at (slotIndex slot)) else none

def Path.owns (path : Path) (slot : TargetSlot) : Bool :=
  path.owners.at (path.labels.at (slotIndex slot))

/-- Equivalent targets use their first index as a representative. -/
def canonical (labels : Quad (Fin 4)) : Bool :=
  (List.finRange 4).all fun i =>
    decide (labels.at i ≤ i) && (labels.at (labels.at i) == labels.at i)

private def target (regs : Registers) (key : String) : Option (Option TargetSlot) :=
  match regs.lookup key with
  | some (.target slot) => some (some slot)
  | some .prefixTarget => some (some .prefix)
  | some (.offset .invalidTarget) => some none
  | _ => none

private def predicate (regs : Registers) (name : String) : Option Bool := do
  let .predicate active ← regs.lookup name | none
  return active

private def addSeed (x y : Expr) : Expr :=
  if x = .literal 0 then .add y x else .add x y

private def integerAdd (regs : Registers) (left right : String) : Option Reg := do
  match getOffset regs left, getOffset regs right with
  | some (.indexed .row 1 i), some (.constant j) =>
    guard (i + j < 3)
    return .offset (.indexed .row 1 (i + j))
  | some (.rowGradient width), _ =>
    return .offset (.gradientTarget width (← target regs right))
  | _, some (.rowGradient width) =>
    if let some (.address .gradient (.constant 0)) := regs.lookup left then
      return .address .gradient (.rowGradient width)
    return .offset (.gradientTarget width (← target regs left))
  | _, _ =>
    match regs.lookup left, target regs right with
    | some (.address .gradient (.rowGradient width)), some slot =>
      return .address .gradient (.gradientTarget width slot)
    | some (.address .codes (.constant 0)), some none =>
      return .address .codes .invalidTarget
    | _, _ => scalar regs ["add.s64", "", left, right]

private def transfer (w : Width) (cache : Nat) (path : Path) (regs : Registers) : List String → Option Reg
  | ["mov.u64", _, "-1"] => some (.offset .invalidTarget)
  | ["mov.u64", dst, src] => (regs.lookup src).orElse fun _ => scalar regs ["mov.u64", dst, src]
  | ["add.s64", _, x, y] => integerAdd regs x y
  | ["setp.ge.s64", _, row, height] => do
    let .indexed .row 1 slot ← getOffset regs row | none
    let .batchRows ← regs.lookup height | none
    guard (slot < 3)
    return .predicate (decide (path.live.val ≤ slot))
  | ["setp.gt.u64", _, source, max] => do
    guard ((← max.toNat?) + 1 == w.columns)
    return .predicate ((← target regs source).all (!path.valid ·))
  | ["setp.gt.u32", _, source, max] => do
    guard ((← max.toNat?) + 1 == w.columns)
    let .targetWord slot ← regs.lookup source | none
    return .predicate slot.isNone
  | ["setp.eq.s64", _, x, y] => do
    let x := (← target regs x).bind path.targetClass
    let y := (← target regs y).bind path.targetClass
    return .predicate (decide (x = y))
  | ["setp.eq.s32", _, x, y] => do
    let .targetWord lhs ← regs.lookup x | none
    let .targetWord rhs ← regs.lookup y | none
    return .predicate (decide (lhs.bind path.targetClass = rhs.bind path.targetClass))
  | ["selp.b64", _, yes, "-1", test] => do
    let .prefixTarget ← regs.lookup yes | none
    return if ← predicate regs test then .prefixTarget else .offset .invalidTarget
  | ["or.pred", _, x, y] => do
    return .predicate ((← predicate regs x) || (← predicate regs y))
  | ["selp.f32", _, yes, no, pred] => do
    let result ← getExpr regs (if ← predicate regs pred then yes else no)
    return .expr result
  | ["ld.global.nc.u64", dst, ptr] => do
    match ← address regs ptr with
    | (.targets, .indexed .row 8 byte) =>
      guard (byte % 8 == 0 && byte / 8 < path.live.val)
      return .target (.mtp (byte / 8))
    | _ => scalar regs ["ld.global.nc.u64", dst, ptr]
  | ["ld.shared.s32", _, ptr] => do
    let (.cachedTargets, .constant byte) ← address regs ptr | none
    guard (byte % 4 == 0 && byte < 4 * cache)
    return .target (.mtp (byte / 4))
  | ["ld.shared.u8", _, ptr] => do
    let (.codes, .target slot) ← address regs ptr | none
    guard (path.valid slot)
    return .logitCode slot
  | ["cvt.u32.u64", dst, source] =>
    match target regs source with
    | some slot => some (.targetWord (slot.filter path.valid))
    | none => scalar regs ["cvt.u32.u64", dst, source]
  | ["shr.u32", _, source, "3"] => do
    let .targetWord slot ← regs.lookup source | none
    return .targetChunk slot
  | ["and.b32", _, source, "255"] => do
    let .targetChunk slot ← regs.lookup source | none
    return .targetOwner slot
  | ["setp.ne.s32", _, owner, lane] => do
    let .targetOwner slot ← regs.lookup owner | none
    let .indexed .lane 1 0 ← getOffset regs lane | none
    -- Masking the low 32 bits of the -1 sentinel produces lane 255.
    -- NVRTC can compute this test before checking vocabulary bounds.
    return .predicate (!(slot.map path.owns).getD path.laneLast)
  | ["add.ftz.f32", _, x, y] => do
    return .expr (addSeed (← getExpr regs x) (← getExpr regs y))
  | [op, dst, weight, zero] =>
    if op == "setp.leu.ftz.f32" || op == "setp.gt.ftz.f32" then do
      guard ((← getExpr regs weight) = .prefixWeight ∧ (← getExpr regs zero) = .literal 0)
      return .predicate (path.prefixPositive == (op == "setp.gt.ftz.f32"))
    else scalar regs [op, dst, weight, zero]
  | insn => scalar regs insn

structure Patch where
  slot : TargetSlot
  value : Expr
  deriving DecidableEq

private structure Program where
  code : Array (List String)
  edges : Array (List Nat)
  start : Nat
  liveIns : Registers
  targetSlots : Nat

private def prepare (w : Width) (ptx : String) (mode : Softcap) : Option Program := do
  let lines := QKVTTIR.normalizePTX ptx
  let edges ← flow lines
  let targetSlots ← TrainingCEInitBinding.checkedTargetSlots w ptx mode
  let start ← correctionEntry ptx
  let (_, stage, barrier) ← decodeStage lines edges
  guard (barrier < start && dominates edges (barrier + 1) start)
  let between := (lines.take start).drop (barrier + 1)
  let saved := stage.filter fun (reg, _) => !(between.any (defines reg))
  let globals := liveIns lines edges start
  let regs := saved.foldl (fun regs (reg, value) => insert regs reg value) globals
  return ⟨lines.toArray.map operands, edges, start, regs, targetSlots⟩

private def execute (program : Program) (w : Width) (path : Path) :
    Nat → Nat → Registers → List Patch → Option (List Patch)
  | 0, _, _, _ => none
  | fuel + 1, pc, regs, observed => do
    guard (program.start ≤ pc)
    match ← program.code[pc]? with
    | ["ret"] => return observed.reverse
    | [test, "bra", _] | [test, "bra.uni", _] =>
      guard (test.startsWith "@")
      let testName := (test.drop 1).toString
      let negated := testName.startsWith "!"
      let reg := if negated then (testName.drop 1).toString else testName
      let goes ← predicate regs reg
      let [taken, fallthrough] ← program.edges[pc]? | none
      execute program w path fuel (if goes != negated then taken else fallthrough) regs observed
    | ["bra", _] | ["bra.uni", _] =>
      let [taken] ← program.edges[pc]? | none
      execute program w path fuel taken regs observed
    | ["ld.shared.v2.f32", sigma, prob, ptr] =>
      let (.sigmoid, .sigmoidOf slot) ← address regs ptr | none
      guard (sigma != prob && path.valid slot)
      let next := insert (insert regs sigma (.expr (.sigmaAt slot))) prob (.expr (.probabilityAt slot))
      execute program w path fuel (pc + 1) next observed
    | ["st.global.u8", pointer, source] =>
      let (.gradient, .gradientTarget width (some slot)) ← address regs pointer | none
      let .e5m2Low value ← regs.lookup source | none
      guard (width == w.columns && path.valid slot)
      execute program w path fuel (pc + 1) regs (⟨slot, value⟩ :: observed)
    | [label] =>
      guard (label.endsWith ":")
      execute program w path fuel (pc + 1) regs observed
    | insn@(_ :: dst :: _) =>
      let value ← transfer w program.targetSlots path regs insn
      execute program w path fuel (pc + 1) (insert regs dst value) observed
    | _ => none

private def Program.run (program : Program) (w : Width) (path : Path) : Option (List Patch) :=
  execute program w path (program.code.size - program.start) program.start program.liveIns []

def decode (w : Width) (ptx : String) (path : Path) (mode : Softcap := .runtime) : Option (List Patch) := do
  let program ← prepare w ptx mode
  program.run w path

def correctionExpr (path : Path) (positions : List Nat) (slot : TargetSlot) : Expr :=
  let representative := path.targetClass slot
  let mtp := positions.foldl (fun acc k =>
    if path.targetClass (.mtp k) = representative then addSeed acc (.weight k) else acc) (.literal 0)
  if path.targetClass .prefix = representative then .add .prefixWeight mtp else mtp

private def gradientExpr (predictions : Fin 4) (path : Path) (slot : TargetSlot) : Expr :=
  let logitSigma := Expr.sigmaAt slot
  let base := Expr.mul (weightExpr predictions) (.probabilityAt slot)
  let correction := correctionExpr path (fixedWalk predictions).forward slot
  .mul (.sub (.literal 0x3f800000) logitSigma) (.mul logitSigma (.mul .factor (.sub base correction)))

private def slotsFor (predictions : Fin 4) : List TargetSlot :=
  (fixedWalk predictions).forward.map TargetSlot.mtp ++ [.prefix]

private def eligibleClass (path : Path) (slot : TargetSlot) : Option (Fin 4) :=
  if slot = .prefix ∧ !path.prefixPositive then none else path.targetClass slot

def expected (predictions : Fin 4) (path : Path) : List Patch :=
  let slots := slotsFor predictions
  slots.zipIdx.filterMap fun (slot, i) => do
    let cls ← eligibleClass path slot
    guard (path.owns slot)
    guard (!(slots.take i).any fun prior => eligibleClass path prior == some cls)
    return ⟨slot, gradientExpr predictions path slot⟩

private def paths : List Path :=
  let labels := (quads (List.finRange 4)).filter canonical
  let flags := [false, true]
  (List.finRange 4).flatMap fun live => flags.flatMap fun hasPrefix =>
    flags.flatMap fun positive => flags.flatMap fun laneLast => labels.flatMap fun label =>
      (quads flags).map fun owners => ⟨live, hasPrefix, positive, laneLast, label, owners⟩

private theorem mem_paths (path : Path) (shape : canonical path.labels = true) : path ∈ paths := by
  rcases path with ⟨live, hasPrefix, positive, laneLast, labels, owners⟩
  refine List.mem_flatMap.mpr ⟨live, List.mem_finRange _, ?_⟩
  refine List.mem_flatMap.mpr ⟨hasPrefix, by cases hasPrefix <;> simp, ?_⟩
  refine List.mem_flatMap.mpr ⟨positive, by cases positive <;> simp, ?_⟩
  refine List.mem_flatMap.mpr ⟨laneLast, by cases laneLast <;> simp, ?_⟩
  refine List.mem_flatMap.mpr ⟨labels, List.mem_filter.mpr ⟨mem_quads _ List.mem_finRange _, shape⟩, ?_⟩
  exact List.mem_map.mpr ⟨owners, mem_quads _ (fun b => by cases b <;> simp) _, rfl⟩

structure Certificate (w : Width) (ptx : String) (predictions : Fin 4) (mode : Softcap := .runtime) : Type where
  traces : ∀ path, canonical path.labels = true →
    decode w ptx path mode = some (expected predictions path)

def certify (w : Width) (ptx : String) (predictions : Fin 4) (mode : Softcap := .runtime) :
    Option (Certificate w ptx predictions mode) :=
  match prepared : prepare w ptx mode with
  | none => none
  | some program =>
    if valid : paths.all (fun path => decide (program.run w path = some (expected predictions path))) then
      some ⟨fun path shape => by
        have check := List.all_eq_true.mp valid _ (mem_paths path shape)
        simpa only [decode, prepared, Option.bind_some] using (of_decide_eq_true check)⟩
    else none

def rowTarget {w : Width} (r : Row w) : TargetSlot → Option (Column w)
  | .mtp k => mtpTarget r k
  | .prefix => r.prefixTarget

private def indexedTarget {w : Width} (r : Row w) (i : Fin 4) : Option (Column w) :=
  if i.val < 3 then mtpTarget r i.val else r.prefixTarget

private def matchesIndex {α : Type} (targets : Fin 4 → α) (value : α) (j : Nat) : Prop :=
  if bound : j < 4 then targets ⟨j, bound⟩ = value else False

private instance {α : Type} [DecidableEq α] (targets : Fin 4 → α) (value : α) (j : Nat) :
    Decidable (matchesIndex targets value j) := by
  unfold matchesIndex
  infer_instance

private theorem matchesIndex_spec {α : Type} (targets : Fin 4 → α) (value : α) (j : Nat) :
    matchesIndex targets value j ↔ ∃ bound : j < 4, targets ⟨j, bound⟩ = value := by
  unfold matchesIndex
  split <;> simp_all

private def firstIndex {α : Type} [DecidableEq α] (targets : Fin 4 → α) (value : α)
    (existsAt : ∃ j, matchesIndex targets value j) : Fin 4 :=
  ⟨Nat.find existsAt, ((matchesIndex_spec ..).mp (Nat.find_spec existsAt)).choose⟩

private theorem firstIndex_value {α : Type} [DecidableEq α] (targets : Fin 4 → α) (value : α)
    (existsAt : ∃ j, matchesIndex targets value j) :
    targets (firstIndex targets value existsAt) = value :=
  ((matchesIndex_spec ..).mp (Nat.find_spec existsAt)).choose_spec

private def labelsFor {α : Type} [DecidableEq α] (targets : Fin 4 → α) (i : Fin 4) : Fin 4 :=
  firstIndex targets (targets i) ⟨i.val, (matchesIndex_spec ..).mpr ⟨i.isLt, rfl⟩⟩

private theorem labelsFor_le {α : Type} [DecidableEq α] (targets : Fin 4 → α) (i : Fin 4) :
    labelsFor targets i ≤ i :=
  Nat.find_min' _ ((matchesIndex_spec ..).mpr ⟨i.isLt, rfl⟩)

private theorem labelsFor_eq {α : Type} [DecidableEq α] (targets : Fin 4 → α) (i j : Fin 4) :
    labelsFor targets i = labelsFor targets j ↔ targets i = targets j := by
  constructor
  · intro same
    exact (firstIndex_value targets _ _).symm.trans
      ((congrArg targets same).trans (firstIndex_value targets _ _))
  · intro same
    unfold labelsFor
    congr 1

def pathFor {w : Width} (p : Parameters) (r : Row w) (lane : Lane) : Path :=
  ⟨TrainingCELossBinding.liveSlots r, r.prefixTarget.isSome, FP.lt ftz (Value.ofBits .fp32 0) p.prefixWeight,
    lane.val == 255, Quad.ofFn (labelsFor (indexedTarget r)), Quad.ofFn fun i =>
      ((indexedTarget r i).map columnOwner) == some lane⟩

theorem canonical_pathFor {w : Width} (p : Parameters) (r : Row w) (lane : Lane) :
    canonical (pathFor p r lane).labels = true := by
  apply List.all_eq_true.mpr
  intro i _
  simp only [pathFor, Quad.at_ofFn, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq]
  exact ⟨labelsFor_le _ _, (labelsFor_eq _ _ _).mpr (firstIndex_value _ _ _)⟩

private theorem indexed_rowTarget {w : Width} (r : Row w) (slot : TargetSlot)
    (bounded : ∀ k, slot = .mtp k → k < 3) :
    indexedTarget r (slotIndex slot) = rowTarget r slot := by
  cases slot with
  | mtp k => simp [indexedTarget, slotIndex, rowTarget, bounded k rfl]
  | «prefix» => rfl

private theorem valid_pathFor {w : Width} (p : Parameters) (r : Row w) (lane : Lane)
    (slot : TargetSlot) (bounded : ∀ k, slot = .mtp k → k < 3) :
    (pathFor p r lane).valid slot = (rowTarget r slot).isSome := by
  cases slot with
  | mtp k =>
    simp only [pathFor, Path.valid, rowTarget, mtpTarget, TrainingCELossBinding.live_slot r k (bounded k rfl)]
    split <;> simp_all
  | «prefix» => rfl

private theorem class_pathFor {w : Width} (p : Parameters) (r : Row w) (lane : Lane)
    (slot : TargetSlot) (bounded : ∀ k, slot = .mtp k → k < 3) :
    (pathFor p r lane).targetClass slot =
      (rowTarget r slot).map (fun _ => labelsFor (indexedTarget r) (slotIndex slot)) := by
  rw [Path.targetClass, valid_pathFor p r lane slot bounded]
  cases rowTarget r slot <;> simp [pathFor, Quad.at_ofFn]

private theorem targets_equal {w : Width} (p : Parameters) (r : Row w) (lane : Lane)
    (s t : TargetSlot) (sb : ∀ k, s = .mtp k → k < 3) (tb : ∀ k, t = .mtp k → k < 3) :
    (pathFor p r lane).targetClass s = (pathFor p r lane).targetClass t ↔ rowTarget r s = rowTarget r t := by
  rw [class_pathFor p r lane s sb, class_pathFor p r lane t tb]
  cases hs : rowTarget r s <;> cases ht : rowTarget r t <;>
    simp [labelsFor_eq, indexed_rowTarget r s sb, indexed_rowTarget r t tb, hs, ht]

private theorem owns_pathFor {w : Width} (p : Parameters) (r : Row w) (lane : Lane)
    (slot : TargetSlot) (bounded : ∀ k, slot = .mtp k → k < 3) :
    (pathFor p r lane).owns slot = (((rowTarget r slot).map columnOwner) == some lane) := by
  simp only [Path.owns, pathFor, Quad.at_ofFn]
  rw [show indexedTarget r (labelsFor (indexedTarget r) (slotIndex slot)) =
    indexedTarget r (slotIndex slot) from firstIndex_value _ _ _, indexed_rowTarget r slot bounded]

def rowInputs {w : Width} (p : Parameters) (r : Row w) (factor : F32)
    (entry : Column w → ProbabilityEntry) : Inputs :=
  let zero := Value.ofBits .fp32 0
  ⟨p, r.weights, r.prefixTarget.isSome, fun _ => zero, ⟨zero, zero⟩, factor,
    fun slot => ((rowTarget r slot).map fun c => (entry c).sigma).getD zero,
    fun slot => ((rowTarget r slot).map fun c => (entry c).probability).getD zero⟩

private theorem addSeed_eval (op : Native) (input : Inputs) (x y : Expr) :
    (addSeed x y).eval op input = FP.add ftz (x.eval op input) (y.eval op input) := by
  unfold addSeed
  split <;> simp only [Expr.eval]
  exact FP.ScalarOpsLaws.add_comm _ _ _

private theorem correction_eval {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (lane : Lane) (factor : F32) (entry : Column w → ProbabilityEntry) (positions : List Nat)
    (positionBound : ∀ k ∈ positions, k < 3) (slot : TargetSlot)
    (bounded : ∀ k, slot = .mtp k → k < 3) (c : Column w) (valid : rowTarget r slot = some c) :
    (correctionExpr (pathFor p r lane) positions slot).eval op (rowInputs p r factor entry) =
      correction p r positions c := by
  have eqMtp (k : Nat) (bound : k < 3) :
      (pathFor p r lane).targetClass (.mtp k) = (pathFor p r lane).targetClass slot ↔
        mtpTarget r k = some c := by
    rw [targets_equal p r lane _ slot (by intro other h; cases h; exact bound) bounded, valid]
    rfl
  have eqPrefix : (pathFor p r lane).targetClass .prefix = (pathFor p r lane).targetClass slot ↔
      r.prefixTarget = some c := by
    rw [targets_equal p r lane _ slot (by simp) bounded, valid]
    rfl
  have fold (items : List Nat) (bounds : ∀ k ∈ items, k < 3) (seed : Expr) :
      (items.foldl (fun acc k => if (pathFor p r lane).targetClass (.mtp k) =
        (pathFor p r lane).targetClass slot then addSeed acc (.weight k) else acc) seed).eval op
          (rowInputs p r factor entry) =
      items.foldl (fun acc k => if mtpTarget r k = some c then FP.add ftz acc (r.weights k) else acc)
        (seed.eval op (rowInputs p r factor entry)) := by
    induction items generalizing seed with
    | nil => rfl
    | cons k rest ih =>
      have bound := bounds k (List.mem_cons_self ..)
      have tail (j : Nat) (inRest : j ∈ rest) : j < 3 := bounds j (List.mem_cons_of_mem k inRest)
      simp only [List.foldl_cons, ih tail, eqMtp k bound]
      split
      · rw [addSeed_eval]
        rfl
      · rfl
  simp only [correctionExpr, correction, eqPrefix, beq_iff_eq]
  split <;> simp only [Expr.eval, fold positions positionBound] <;> rfl

private theorem positionBound (n : Fin 4) (k : Nat) (member : k ∈ (fixedWalk n).forward) : k < 3 := by
  have bound := n.isLt
  rw [fixed_forward, dynamicWalk, List.mem_range] at member
  omega

theorem gradient_eval {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (lane : Lane) (factor : F32) (entry : Column w → ProbabilityEntry) (slot : TargetSlot)
    (bounded : ∀ k, slot = .mtp k → k < 3) (c : Column w) (live : rowTarget r slot = some c) :
    floatToE5m2 ((gradientExpr r.predictions (pathFor p r lane) slot).eval op (rowInputs p r factor entry)) =
      gradientValue p r (fixedWalk r.predictions) factor (entry c) c true := by
  have sameWalk : (fixedWalk r.predictions).correction = (fixedWalk r.predictions).forward := rfl
  have weight := weight_eval op p r (fun _ => Value.ofBits .fp32 0)
    ⟨Value.ofBits .fp32 0, Value.ofBits .fp32 0⟩ factor (rowInputs p r factor entry).targetSigma
    (rowInputs p r factor entry).targetProbability
  change (weightExpr r.predictions).eval op (rowInputs p r factor entry) = _ at weight
  simp only [gradientExpr, Expr.eval, correction_eval op p r lane factor entry _
    (positionBound r.predictions) slot bounded c live, weight]
  simp only [gradientValue, sameWalk]
  simp only [rowInputs, live, Option.map_some, Option.getD_some]
  rfl

private theorem slotBound (n : Fin 4) (slot : TargetSlot) (member : slot ∈ slotsFor n) :
    ∀ k, slot = .mtp k → k < 3 := by
  intro k hk
  subst slot
  apply positionBound n
  simpa [slotsFor] using member

private def rowEligible {w : Width} (p : Parameters) (r : Row w) (slot : TargetSlot) :
    Option (Column w) :=
  if slot = .prefix ∧ !FP.lt ftz (Value.ofBits .fp32 0) p.prefixWeight then none else rowTarget r slot

private theorem eligible_pathFor {w : Width} (p : Parameters) (r : Row w) (lane : Lane)
    (slot : TargetSlot) (bounded : ∀ k, slot = .mtp k → k < 3) :
    eligibleClass (pathFor p r lane) slot =
      (rowEligible p r slot).map (fun _ => labelsFor (indexedTarget r) (slotIndex slot)) := by
  simp only [eligibleClass, rowEligible, show (pathFor p r lane).prefixPositive =
    FP.lt ftz (Value.ofBits .fp32 0) p.prefixWeight from rfl]
  split
  · rfl
  · exact class_pathFor p r lane slot bounded

private theorem prior_matches {w : Width} (p : Parameters) (r : Row w) (lane : Lane)
    (s t : TargetSlot) (sb : ∀ k, s = .mtp k → k < 3) (tb : ∀ k, t = .mtp k → k < 3)
    (c : Column w) (live : rowEligible p r t = some c) :
    eligibleClass (pathFor p r lane) s =
      some (labelsFor (indexedTarget r) (slotIndex t)) ↔ rowEligible p r s = some c := by
  have raw (slot : TargetSlot) (d : Column w) (active : rowEligible p r slot = some d) :
      rowTarget r slot = some d := by
    unfold rowEligible at active
    split at active <;> simp_all
  rw [eligible_pathFor p r lane s sb]
  cases hs : rowEligible p r s with
  | none => simp
  | some d =>
    have hsd := raw s d hs
    have htc := raw t c live
    simp [labelsFor_eq, indexed_rowTarget r s sb, indexed_rowTarget r t tb, hsd, htc]

private theorem eligible_slots {w : Width} (p : Parameters) (r : Row w) :
    (slotsFor r.predictions).map (rowEligible p r) =
      correctionSlots p r (fixedWalk r.predictions).forward := by
  change (slotsFor r.predictions).map (rowEligible p r) =
    (fixedWalk r.predictions).forward.map (mtpTarget r) ++
      [if FP.lt ftz (Value.ofBits .fp32 0) p.prefixWeight then r.prefixTarget else none]
  cases h : FP.lt ftz (Value.ofBits .fp32 0) p.prefixWeight <;>
    simp [slotsFor, rowEligible, rowTarget, h, List.map_map, Function.comp_def]

def patchIndex (predictions : Fin 4) : TargetSlot → Nat
  | .mtp k => k
  | .prefix => predictions.val

private theorem zip_index (n : Fin 4) (slot : TargetSlot) (i : Nat)
    (member : (slot, i) ∈ (slotsFor n).zipIdx) : patchIndex n slot = i := by
  have all : ∀ n : Fin 4, ((slotsFor n).zipIdx.all fun (s, i) => decide (patchIndex n s = i)) = true := by
    decide
  exact of_decide_eq_true (List.all_eq_true.mp (all n) _ member)

private theorem first_slot_iff {w : Width} (p : Parameters) (r : Row w) (i : Nat)
    (slot : TargetSlot) (member : (slot, i) ∈ (slotsFor r.predictions).zipIdx) (c : Column w)
    (eligible : rowEligible p r slot = some c) :
    firstCorrectionSlot p r (fixedWalk r.predictions).forward c = some i ↔
      ∀ prior ∈ (slotsFor r.predictions).take i, rowEligible p r prior ≠ some c := by
  rw [first_correction_slot, ← eligible_slots]
  obtain ⟨bound, atIndex⟩ := List.getElem?_eq_some_iff.mp (List.mk_mem_zipIdx_iff_getElem?.mp member)
  constructor
  · rintro ⟨_, _, before⟩ prior early
    obtain ⟨j, earlier, rfl⟩ := List.mem_take_iff_getElem.mp early
    simpa using before j (by omega)
  · intro before
    refine ⟨by simpa using bound, by simpa [atIndex] using eligible, ?_⟩
    intro j earlier
    simpa using before _ (List.mem_take_iff_getElem.mpr ⟨j, by omega, rfl⟩)

def Patch.observe {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (factor : F32) (entry : Column w → ProbabilityEntry) (patch : Patch) : Option (Column w × Nat × UInt8) :=
  (rowTarget r patch.slot).map fun c =>
    (c, patchIndex r.predictions patch.slot, floatToE5m2 (patch.value.eval op (rowInputs p r factor entry)))

def decodedWrites {w : Width} (ptx : String) (op : Native) (p : Parameters)
    (r : Row w) (lane : Lane) (factor : F32) (entry : Column w → ProbabilityEntry)
    (mode : Softcap := .runtime) :
    Option (List (Column w × Nat × UInt8)) :=
  (decode w ptx (pathFor p r lane) mode).map (List.filterMap (Patch.observe op p r factor entry))

private theorem expected_write {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (lane : Lane) (factor : F32) (entry : Column w → ProbabilityEntry)
    (slot : TargetSlot) (i : Nat) (member : (slot, i) ∈ (slotsFor r.predictions).zipIdx)
    (eligibleCol : Option (Column w)) (eligible : rowEligible p r slot = eligibleCol) :
    ((do
      let cls ← eligibleClass (pathFor p r lane) slot
      guard ((pathFor p r lane).owns slot)
      guard (!((slotsFor r.predictions).take i).any fun prior =>
        eligibleClass (pathFor p r lane) prior == some cls)
      pure ⟨slot, gradientExpr r.predictions (pathFor p r lane) slot⟩ : Option Patch).bind
        (Patch.observe op p r factor entry)) =
    (do
      let c ← eligibleCol
      guard (decide (columnOwner c = lane))
      guard (decide (firstCorrectionSlot p r (fixedWalk r.predictions).forward c = some i))
      some (c, i, gradientValue p r (fixedWalk r.predictions) factor (entry c) c true)) := by
  have bounded := slotBound r.predictions slot (List.fst_mem_of_mem_zipIdx member)
  rw [eligible_pathFor p r lane slot bounded, eligible]
  cases eligibleCol with
  | none => rfl
  | some c =>
    have live : rowTarget r slot = some c := by
      unfold rowEligible at eligible
      split at eligible <;> simp_all
    simp only [Option.map_some, Bind.bind, Option.bind_some]
    have owns : (pathFor p r lane).owns slot = decide (columnOwner c = lane) := by
      rw [owns_pathFor p r lane slot bounded, live]
      cases h : columnOwner c == lane <;> simp_all
    rw [owns]
    have prior : (!((slotsFor r.predictions).take i).any fun s =>
        eligibleClass (pathFor p r lane) s == some (labelsFor (indexedTarget r) (slotIndex slot))) =
        decide (firstCorrectionSlot p r (fixedWalk r.predictions).forward c = some i) := by
      apply Bool.eq_iff_iff.mpr
      simp only [Bool.not_eq_true', List.any_eq_false, beq_iff_eq, decide_eq_true_eq,
        first_slot_iff p r i slot member c eligible]
      apply forall_congr'
      intro s
      apply forall_congr'
      intro early
      exact not_congr (prior_matches p r lane s slot
        (slotBound _ _ (List.mem_of_mem_take early)) bounded c eligible)
    rw [prior]
    by_cases owner : columnOwner c = lane <;>
      by_cases first : firstCorrectionSlot p r (fixedWalk r.predictions).forward c = some i <;>
      simp [guard, owner, first, Patch.observe, live, zip_index _ _ _ member,
        gradient_eval op p r lane factor entry slot bounded c live] <;> rfl

/-- Every sparse write from the compiler has the reference slot, owner
and bit pattern. Conversely a reference correction is present in the
trace of its owner lane. -/
theorem decoded_write_exact {w : Width} {ptx : String} {mode : Softcap}
    (op : Native) (p : Parameters) (r : Row w)
    (artifact : Certificate w ptx r.predictions mode) (lane : Lane) (factor : F32)
    (entry : Column w → ProbabilityEntry) (c : Column w) (i : Nat) (byte : UInt8) :
    (decodedWrites ptx op p r lane factor entry mode).any (fun writes => writes.contains (c, i, byte)) = true ↔
      columnOwner c = lane ∧ (gradientWrites p r (fixedWalk r.predictions) factor entry c).correction =
        some (i, byte) := by
  rw [decodedWrites, artifact.traces _ (canonical_pathFor p r lane)]
  simp only [Option.map_some, Option.any, List.contains_iff_mem, expected, List.filterMap_filterMap,
    List.mem_filterMap]
  constructor
  · rintro ⟨⟨slot, j⟩, member, wrote⟩
    rw [expected_write op p r lane factor entry slot j member _ rfl] at wrote
    cases eligible : rowEligible p r slot with
    | none => simp [eligible] at wrote
    | some col =>
      by_cases owned : columnOwner col = lane <;>
        by_cases first : firstCorrectionSlot p r (fixedWalk r.predictions).forward col = some j <;>
        simp [eligible, Bind.bind, guard, owned, first] at wrote <;>
        try (change none = some _ at wrote; contradiction)
      obtain ⟨rfl, rfl, rfl⟩ := wrote
      exact ⟨owned, by simp [gradientWrites, first]⟩
  · rintro ⟨owned, wrote⟩
    cases first : firstCorrectionSlot p r (fixedWalk r.predictions).forward c with
    | none => simp [gradientWrites, first] at wrote
    | some j =>
      have same : j = i ∧ gradientValue p r (fixedWalk r.predictions) factor (entry c) c true = byte := by
        simpa [gradientWrites, first] using wrote
      obtain ⟨rfl, rfl⟩ := same
      have indexed := (first_correction_slot p r _ c j).mp first
      rw [← eligible_slots] at indexed
      obtain ⟨bound, present, _⟩ := indexed
      have bound' : j < (slotsFor r.predictions).length := by simpa using bound
      let slot := (slotsFor r.predictions)[j]'bound'
      have member : (slot, j) ∈ (slotsFor r.predictions).zipIdx :=
        List.mk_mem_zipIdx_iff_getElem?.mpr (List.getElem?_eq_some_iff.mpr ⟨bound', rfl⟩)
      have eligible : rowEligible p r slot = some c := by simpa [slot] using present
      refine ⟨(slot, j), member, ?_⟩
      rw [expected_write op p r lane factor entry slot j member _ eligible]
      simp [Bind.bind, guard, owned, first]

def decodedCorrection {w : Width} (ptx : String) (op : Native) (p : Parameters) (r : Row w)
    (factor : F32) (entry : Column w → ProbabilityEntry) (c : Column w) (mode : Softcap := .runtime) :
    Option (Option (Nat × UInt8)) :=
  (decodedWrites ptx op p r (columnOwner c) factor entry mode).map fun writes =>
    (writes.find? fun patch => patch.1 == c).map Prod.snd

theorem decoded_correction_exact {w : Width} {ptx : String} {mode : Softcap}
    (op : Native) (p : Parameters) (r : Row w)
    (artifact : Certificate w ptx r.predictions mode) (factor : F32)
    (entry : Column w → ProbabilityEntry) (c : Column w) :
    decodedCorrection ptx op p r factor entry c mode =
      some (gradientWrites p r (fixedWalk r.predictions) factor entry c).correction := by
  let lane := columnOwner c
  let writes := (expected r.predictions (pathFor p r lane)).filterMap (Patch.observe op p r factor entry)
  have ran : decodedWrites ptx op p r lane factor entry mode = some writes := by
    rw [decodedWrites, artifact.traces _ (canonical_pathFor p r lane)]
    rfl
  have exactWrite (i : Nat) (byte : UInt8) :
      (c, i, byte) ∈ writes ↔
        (gradientWrites p r (fixedWalk r.predictions) factor entry c).correction = some (i, byte) := by
    simpa [ran, lane, List.contains_iff_mem] using
      decoded_write_exact op p r artifact lane factor entry c i byte
  unfold decodedCorrection
  rw [ran]
  simp only [Option.map_some, Option.some.injEq]
  cases found : writes.find? (fun patch => patch.1 == c) with
  | none =>
    rw [Option.map_none]
    cases expected : (gradientWrites p r (fixedWalk r.predictions) factor entry c).correction with
    | none => rfl
    | some patch =>
      obtain ⟨i, byte⟩ := patch
      have member := (exactWrite i byte).mpr expected
      exact False.elim (List.find?_eq_none.mp found _ member (by simp))
  | some patch =>
    obtain ⟨col, i, byte⟩ := patch
    have same : col = c := beq_iff_eq.mp
      (List.find?_some (p := fun patch : Column w × Nat × UInt8 => patch.1 == c) found)
    subst col
    exact ((exactWrite i byte).mp (List.mem_of_find?_eq_some found)).symm

#axiomsClean certify
#axiomsClean canonical_pathFor
#axiomsClean gradient_eval
#axiomsClean decoded_write_exact
#axiomsClean decoded_correction_exact
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCESparseBinding
