import models.NanoGPT.Track1.TrainingCEInitBinding

/-! Decode the row-loss store following the cache barrier. Before any
floating-point evaluation, the branch trace checks how many predictions
are live, whether the row has a prefix target and which lane may write.
Targets remain symbolic vocabulary columns. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCELossBinding
open TrainingCE TrainingCEGatherBinding TrainingCEPublishBinding
open VeriTile.Triton FP

private structure Path where
  live : Fin 4
  hasPrefix : Bool
  laneZero : Bool

private def Path.valid (path : Path) : TargetSlot → Bool
  | .mtp k => decide (k < path.live.val)
  | .prefix => path.hasPrefix

private def target (regs : Registers) (key : String) : Option TargetSlot :=
  match regs.lookup key with
  | some (.target slot) => some slot
  | some .prefixTarget => some .prefix
  | _ => none

private def predicate (path : Path) (regs : Registers) (key : String) : Option Bool :=
  match regs.lookup key with
  | some (.predicate test) => some test
  | some .laneNonzero => some (!path.laneZero)
  | _ => none

private def transfer (w : Width) (cache : Nat) (path : Path) (regs : Registers) : List String → Option Reg
  | ["setp.ge.s64", _, row, height] => do
    let .indexed .row 1 slot ← getOffset regs row | none
    let .batchRows ← regs.lookup height | none
    guard (slot < 3)
    return .predicate (decide (path.live.val ≤ slot))
  | ["setp.gt.u64", _, reg, max] => do
    let limit ← max.toNat?
    guard (limit + 1 == w.columns)
    return .predicate (!(path.valid (← target regs reg)))
  | ["ld.shared.u32", _, ptr] => do
    let (.cachedTargets, .constant byte) ← address regs ptr | none
    guard (byte % 4 == 0 && byte < 4 * cache)
    return .cachedTarget32 (byte / 4)
  | ["setp.gt.u32", _, reg, max] => do
    guard ((← max.toNat?) + 1 == w.columns)
    let .cachedTarget32 slot ← regs.lookup reg | none
    return .predicate (!(path.valid (.mtp slot)))
  | ["ld.global.nc.u64", dst, ptr] => do
    match ← address regs ptr with
    | (.targets, .indexed .row 8 byte) =>
      guard (byte % 8 == 0 && byte / 8 < path.live.val)
      return .target (.mtp (byte / 8))
    | _ => scalar regs ["ld.global.nc.u64", dst, ptr]
  | ["ld.shared.u8", _, ptr] => do
    let (.codes, .target slot) ← address regs ptr | none
    guard (path.valid slot)
    return .logitCode slot
  | insn => scalar regs insn

private structure Program where
  code : Array (List String)
  edges : Array (List Nat)
  start : Nat
  stop : Nat
  liveIns : Registers
  targetSlots : Nat

private def prepare (w : Width) (ptx : String) (mode : Softcap) : Option Program := do
  let lines := QKVTTIR.normalizePTX ptx
  guard (lines.contains ".extern .shared .align 16 .b8 code_smem[];" &&
    lines.contains ".maxntid 256, 1, 1")
  let edges ← flow lines
  let targetSlots ← TrainingCEInitBinding.checkedTargetSlots w ptx mode
  let (_, regs, barrier) ← decodeStage lines edges
  let [(_, write)] := lines.zipIdx.filter (fun (line, _) => line.startsWith "st.global.f32 ") | none
  guard (barrier < write && dominates edges (barrier + 1) write)
  let some join := lines[write + 1]? | none
  guard (join.endsWith ":")
  return ⟨lines.toArray.map operands, edges, barrier + 1, write + 1, regs, targetSlots⟩

private def execute (program : Program) (w : Width) (path : Path) :
    Nat → Nat → Registers → Option Expr → Option (Option Expr)
  | 0, _, _, _ => none
  | fuel + 1, pc, regs, observed => do
    if pc == program.stop then return observed
    guard (program.start ≤ pc && pc < program.stop)
    match ← program.code[pc]? with
    | [test, "bra", _] =>
      guard (test.startsWith "@")
      let testName := (test.drop 1).toString
      let negated := testName.startsWith "!"
      let reg := if negated then (testName.drop 1).toString else testName
      let goes ← predicate path regs reg
      let [taken, fallthrough] ← program.edges[pc]? | none
      execute program w path fuel (if goes != negated then taken else fallthrough) regs observed
    | ["st.global.f32", pointer, value] =>
      let (.losses, .indexed .row 4 0) ← address regs pointer | none
      guard observed.isNone
      execute program w path fuel (pc + 1) regs (some (← getExpr regs value))
    | [label] =>
      guard (label.endsWith ":")
      execute program w path fuel (pc + 1) regs observed
    | insn@(_ :: dst :: _) =>
      let value ← transfer w program.targetSlots path regs insn
      execute program w path fuel (pc + 1) (insert regs dst value) observed
    | _ => none

private def Program.run (program : Program) (w : Width) (path : Path) : Option (Option Expr) :=
  execute program w path (program.stop - program.start + 1) program.start program.liveIns none

def decode (w : Width) (ptx : String) (live : Fin 4) (hasPrefix laneZero : Bool)
    (mode : Softcap := .runtime) : Option (Option Expr) := do
  let program ← prepare w ptx mode
  program.run w ⟨live, hasPrefix, laneZero⟩

private def term (loss : Expr) (target : TargetSlot) (weight : Expr) : Expr :=
  .fma weight (.sub lseExpr (.mul (.sigmaAt target) .a)) loss

def expected (predictions live : Fin 4) (hasPrefix : Bool) : Expr :=
  let mtp := ((fixedWalk predictions).forward).foldl (fun acc k =>
    if k < live.val then term acc (.mtp k) (.weight k) else acc) (.literal 0)
  if hasPrefix then term mtp .prefix .prefixWeight else mtp

private def paths : List (Fin 4 × Bool × Bool) :=
  (List.finRange 4).flatMap fun n => [false, true].flatMap fun p =>
    [false, true].map fun tid => (n, p, tid)

structure Certificate (w : Width) (ptx : String) (predictions : Fin 4) (mode : Softcap := .runtime) : Type where
  traces : ∀ live hasPrefix laneZero,
    decode w ptx live hasPrefix laneZero mode =
      some (if laneZero then some ((expected predictions live hasPrefix).specialize mode) else none)

def certify (w : Width) (ptx : String) (predictions : Fin 4) (mode : Softcap := .runtime) :
    Option (Certificate w ptx predictions mode) :=
  match prepared : prepare w ptx mode with
  | none => none
  | some program =>
    if valid : paths.all (fun (live, hasPrefix, laneZero) =>
      decide (program.run w ⟨live, hasPrefix, laneZero⟩ =
        some (if laneZero then some ((expected predictions live hasPrefix).specialize mode) else none))) then
      some ⟨fun live hasPrefix laneZero => by
        have member : (live, hasPrefix, laneZero) ∈ paths := by
          cases hasPrefix <;> cases laneZero <;> simp [paths]
        have check := List.all_eq_true.mp valid _ member
        simpa only [decode, prepared, Option.bind_some] using (of_decide_eq_true check)⟩
    else none

def liveSlots {w : Width} (r : Row w) : Fin 4 :=
  ⟨min 3 (r.batchRows - r.row.val), by omega⟩

def inputs {w : Width} (p : Parameters) (r : Row w) (leaders : Fin 8 → F32)
    (sigma : Column w → F32) : Inputs :=
  let zero := Value.ofBits .fp32 0
  let atTarget := fun slot =>
    match slot with
    | .mtp k =>
      if live : r.row.val + k < r.batchRows then sigma (r.targets ⟨_, live⟩) else zero
    | .prefix => (r.prefixTarget.map sigma).getD zero
  ⟨p, r.weights, r.prefixTarget.isSome, leaders, ⟨zero, zero⟩, zero, atTarget, fun _ => zero⟩

theorem live_slot {w : Width} (r : Row w) (k : Nat) (slot : k < 3) :
    k < (liveSlots r).val ↔ r.row.val + k < r.batchRows := by
  have bound := r.row.isLt
  change k < min 3 (r.batchRows - r.row.val) ↔ _
  omega

private theorem fold_exact {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (leaders : Fin 8 → F32) (sigma : Column w → F32) (slots : List Nat)
    (bounded : ∀ k ∈ slots, k < 3) (seed : Expr) :
    (slots.foldl (fun acc k =>
      if k < (liveSlots r).val then term acc (.mtp k) (.weight k) else acc) seed).eval op
        (inputs p r leaders sigma) =
    slots.foldl (fun acc k =>
      if live : r.row.val + k < r.batchRows then
        FP.fma ftz (r.weights k)
          (FP.sub ftz (logSumExpFromLeaders op p leaders) (FP.mul ftz (sigma (r.targets ⟨_, live⟩)) p.a)) acc
      else acc) (seed.eval op (inputs p r leaders sigma)) := by
  induction slots generalizing seed with
  | nil => rfl
  | cons slot rest ih =>
    have bound := bounded slot (List.mem_cons_self ..)
    have tail : ∀ k ∈ rest, k < 3 := fun k h => bounded k (List.mem_cons_of_mem slot h)
    simp only [List.foldl_cons, ih tail, live_slot r slot bound]
    by_cases active : r.row.val + slot < r.batchRows
    · simp only [if_pos active, dif_pos active, term, Expr.eval, lse_eval, inputs]
    · simp only [if_neg active, dif_neg active]

theorem loss_exact {w : Width} {ptx : String} {mode : Softcap} (op : Native) (p : Parameters) (r : Row w)
    (artifact : Certificate w ptx r.predictions mode) (leaders : Fin 8 → F32) (sigma : Column w → F32)
    (accepted : mode.accepts p = true := by rfl) :
    (decode w ptx (liveSlots r) r.prefixTarget.isSome true mode).map
        (Option.map (Expr.eval op (inputs p r leaders sigma))) =
      some (some (lossFromScaled p r (fixedWalk r.predictions).forward
        (fun c => FP.mul ftz (sigma c) p.a) (logSumExpFromLeaders op p leaders))) := by
  rw [artifact.traces]
  have launch : mode.accepts (inputs p r leaders sigma).parameters = true := accepted
  simp only [if_true, Option.map_some, Expr.eval_for_launch _ _ _ _ launch, expected, lossFromScaled]
  have bounded : ∀ k ∈ (fixedWalk r.predictions).forward, k < 3 := by
    rw [fixed_forward]
    intro k member
    have below := List.mem_range.mp member
    have := r.predictions.isLt
    omega
  have fold := fold_exact op p r leaders sigma (fixedWalk r.predictions).forward bounded (.literal 0)
  cases tag : r.prefixTarget with
  | none => simpa only [tag, Option.isSome_none, if_false] using congrArg (fun x => some (some x)) fold
  | some cls =>
    simp only [Option.isSome_some, if_true, term, Expr.eval, lse_eval]
    simp only [inputs, tag, Option.map_some, Option.getD_some, term] at fold ⊢
    rw [fold]
    rfl

#axiomsClean loss_exact
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCELossBinding
