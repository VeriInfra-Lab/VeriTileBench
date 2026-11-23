import models.NanoGPT.Track1.TrainingCEPublishBinding

/-! Decode the sigmoid register pair or shared stores, and the exponential table writes before the
first block barrier, including the lane-zero gradient scale. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEInitBinding
open TrainingCE TrainingCEConversions TrainingCEGatherBinding
open VeriTile.Triton FP

inductive Parameter where
  | a | b | c | gradScale
  deriving DecidableEq

inductive Expr where
  | literal (bits : Nat)
  | param (field : Parameter)
  | decodedByte (high : Bool)
  | roundedHalf (x : Expr)
  | reciprocal (x : Expr)
  | tanh (x : Expr)
  | exp2 (x : Expr)
  | add (x y : Expr)
  | sub (x y : Expr)
  | mul (x y : Expr)
  | fma (x y z : Expr)
  deriving DecidableEq

def Parameter.value (p : Parameters) : Parameter → F32
  | .a => p.a
  | .b => p.b
  | .c => p.c
  | .gradScale => p.gradScale

def Expr.eval (op : Native) (p : Parameters) (code : UInt8) : Expr → F32
  | .literal bits => Value.ofBits .fp32 bits
  | .param field => field.value p
  | .decodedByte high => TrainingCE.decode (if high then 0 else code)
  | .roundedHalf x => halfToFloat (floatToHalf (x.eval op p code))
  | .reciprocal x => op.reciprocal (x.eval op p code)
  | .tanh x => op.tanh (x.eval op p code)
  | .exp2 x => op.exp2 (x.eval op p code)
  | .add x y => FP.add ftz (x.eval op p code) (y.eval op p code)
  | .sub x y => FP.sub ftz (x.eval op p code) (y.eval op p code)
  | .mul x y => FP.mul ftz (x.eval op p code) (y.eval op p code)
  | .fma x y z => FP.fma ftz (x.eval op p code) (y.eval op p code) (z.eval op p code)

private def Parameter.specialize : Softcap → GradScale → Parameter → Expr
  | .defaultABC, _, .a => .literal 0x41b80000
  | .defaultBC, _, .b | .defaultABC, _, .b => .literal 0x40a00000
  | .defaultBC, _, .c | .defaultABC, _, .c => .literal 0x40f00000
  | _, .lmHead, .gradScale => .literal GradScale.lmHeadWord
  | _, _, field => .param field

private def Expr.specialize (mode : Softcap) (scale : GradScale) : Expr → Expr
  | .param field => field.specialize mode scale
  | .roundedHalf x => .roundedHalf (x.specialize mode scale)
  | .reciprocal x => .reciprocal (x.specialize mode scale)
  | .tanh x => .tanh (x.specialize mode scale)
  | .exp2 x => .exp2 (x.specialize mode scale)
  | .add x y => .add (x.specialize mode scale) (y.specialize mode scale)
  | .sub x y => .sub (x.specialize mode scale) (y.specialize mode scale)
  | .mul x y => .mul (x.specialize mode scale) (y.specialize mode scale)
  | .fma x y z => .fma (x.specialize mode scale) (y.specialize mode scale) (z.specialize mode scale)
  | other => other

private theorem Expr.eval_specialize (op : Native) (p : Parameters) (code : UInt8)
    (mode : Softcap) (scale : GradScale) (expr : Expr) :
    (expr.specialize mode scale).eval op p code = expr.eval op (scale.parameters (mode.parameters p)) code := by
  induction expr with
  | param field =>
    cases field <;> cases mode <;> cases scale <;>
      simp [Expr.specialize, Parameter.specialize, Expr.eval, Parameter.value,
        Softcap.parameters, GradScale.parameters, Parameters.a, Parameters.b, Parameters.c,
        Parameters.gradScale, Softcap.a_word, Softcap.b_word, Softcap.c_word, GradScale.lm_head_word]
  | _ => simp [Expr.specialize, Expr.eval, *]

private inductive Reg where
  | addressReg (value : TrainingCEPublishBinding.Reg)
  | rawParameter (field : Parameter)
  | rawLogitPointer
  | expression (value : Expr)
  | laneHalfword
  | packedRound (low : Expr)
  | rounded (value : Expr)
  | target64 (slot : Option Nat)
  | target32 (slot : Option Nat)

private abbrev Registers := List (String × Reg)

private def save (regs : Registers) (dst : String) (value : Reg) : Registers :=
  (dst, value) :: regs.filter (fun entry => entry.1 != dst)

private def addresses (regs : Registers) : TrainingCEPublishBinding.Registers :=
  regs.filterMap fun (key, value) =>
    match value with
    | .addressReg r => some (key, r)
    | _ => none

private def expression (regs : Registers) (key : String) : Option Expr :=
  match regs.lookup key with
  | some (.expression e) => some e
  | _ => do
    let .literal bits ← TrainingCEPublishBinding.getExpr [] key | none
    return .literal bits

private def inlineAsm (regs : Registers) (line : String) : Option Registers := do
  guard (line.startsWith "{" && line.endsWith "}")
  let code := ((line.drop 1 |>.dropEnd 1).toString.splitOn ";").map operands |>.filter (!·.isEmpty)
  match code with
  | [[".reg", ".b32", pair], [".reg", ".b16", low, high],
      ["cvt.rn.f16x2.e4m3x2", toPair, source], ["mov.b32", toLow, toHigh, fromPair],
      ["cvt.f32.f16", widenLow, fromLow], ["cvt.f32.f16", widenHigh, fromHigh]] =>
    let .laneHalfword ← regs.lookup source | none
    guard (toPair == pair && fromPair == pair && low == toLow && low == fromLow &&
      high == toHigh && high == fromHigh && low != high && widenLow != widenHigh)
    return save (save regs widenLow (.expression (.decodedByte false))) widenHigh (.expression (.decodedByte true))
  | [["cvt.rn.f16x2.f32", output, hi, lo]] =>
    guard ((← expression regs hi) = .literal 0)
    return save regs output (.packedRound (← expression regs lo))
  | [["cvt.f32.f16", output, input]] =>
    let .rounded value ← regs.lookup input | none
    return save regs output (.expression (.roundedHalf value))
  | _ => none

private def transfer (w : Width) (live : Fin 4) (regs : Registers) : List String → Option Reg
  | ["ld.param.f64", _, "[ce_fwd_bwd_kernel_param_9]"] => some (.rawParameter .a)
  | ["ld.param.f64", _, "[ce_fwd_bwd_kernel_param_10]"] => some (.rawParameter .b)
  | ["ld.param.f64", _, "[ce_fwd_bwd_kernel_param_11]"] => some (.rawParameter .c)
  | ["ld.param.f64", _, "[ce_fwd_bwd_kernel_param_12]"] => some (.rawParameter .gradScale)
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_0]"] =>
    -- The logit pointer will be used by the scan after this barrier.
    some .rawLogitPointer
  | ["cvt.rn.ftz.f32.f64", _, source] => do
    let .rawParameter field ← regs.lookup source | none
    return .expression (.param field)
  | ["cvt.u16.u32", _, source] => do
    match regs.lookup source with
    | some (.addressReg (.offset (.indexed .lane 1 0))) => some .laneHalfword
    | some (.packedRound x) => some (.rounded x)
    | _ => none
  | ["cvt.u32.u64", _, source] => do
    match regs.lookup source with
    | some (.addressReg (.offset (.indexed .lane 1 0))) =>
      some (.addressReg (.offset (.indexed .lane 1 0)))
    | some (.target64 slot) => some (.target32 slot)
    | _ => none
  | ["ld.global.nc.u64", dst, ptr] => do
    let args := addresses regs
    match ← TrainingCEPublishBinding.address args ptr with
    | (.targets, .indexed .row 8 byte) =>
      guard (byte % 8 == 0 && byte / 8 < live.val)
      return .target64 (some (byte / 8))
    | _ => (TrainingCEPublishBinding.scalar args ["ld.global.nc.u64", dst, ptr]).map .addressReg
  | ["mov.u64", _, "-1"] => some (.target64 none)
  | ["mov.u64", dst, src] =>
    (regs.lookup src).orElse fun _ =>
      (TrainingCEPublishBinding.scalar (addresses regs) ["mov.u64", dst, src]).map .addressReg
  | ["setp.ge.s64", _, row, height] => do
    let args := addresses regs
    let .indexed .row 1 slot ← TrainingCEPublishBinding.getOffset args row | none
    let .batchRows ← args.lookup height | none
    guard (slot < 3)
    return .addressReg (.predicate (decide (live.val ≤ slot)))
  | ["setp.lt.u64", _, src, limit] => do
    let .target64 slot ← regs.lookup src | none
    guard (limit.toNat? == some w.columns)
    return .addressReg (.predicate slot.isSome)
  | ["selp.b32", _, yes, "-1", test] => do
    let .target32 slot ← regs.lookup yes | none
    let .addressReg (.predicate eligible) ← regs.lookup test | none
    return .target32 (if eligible then slot else none)
  | ["mov.f32", _, src] => (expression regs src).map .expression
  | ["rcp.approx.ftz.f32", _, src] => (expression regs src).map fun x => .expression (.reciprocal x)
  | ["tanh.approx.f32", _, src] => (expression regs src).map fun x => .expression (.tanh x)
  | ["ex2.approx.ftz.f32", _, src] => (expression regs src).map fun x => .expression (.exp2 x)
  | ["fma.rn.ftz.f32", _, x, y, z] => do
    return .expression (.fma (← expression regs x) (← expression regs y) (← expression regs z))
  | [op, dst, x, y] =>
    match op with
    | "add.ftz.f32" => do return .expression (.add (← expression regs x) (← expression regs y))
    | "sub.ftz.f32" => do return .expression (.sub (← expression regs x) (← expression regs y))
    | "mul.ftz.f32" | "mul.rn.ftz.f32" => do
      return .expression (.mul (← expression regs x) (← expression regs y))
    | "add.s64" =>
      let args := addresses regs
      match TrainingCEPublishBinding.getOffset args x, y.toNat? with
      | some (.indexed .row 1 i), some j =>
        if i + j < 3 then some (.addressReg (.offset (.indexed .row 1 (i + j)))) else none
      | _, _ => (TrainingCEPublishBinding.scalar args [op, dst, x, y]).map .addressReg
    | _ => (TrainingCEPublishBinding.scalar (addresses regs) [op, dst, x, y]).map .addressReg
  | insn => (TrainingCEPublishBinding.scalar (addresses regs) insn).map .addressReg

structure Writes where
  sigmoid : Option (Expr × Expr) := none
  contribution : Option Expr := none
  factor : Option Expr := none
  targets : List (Nat × Option Nat) := []
  deriving DecidableEq

private def Writes.specialize (mode : Softcap) (scale : GradScale) (writes : Writes) : Writes :=
  { writes with
    sigmoid := writes.sigmoid.map fun (sigma, scaled) => (sigma.specialize mode scale, scaled.specialize mode scale)
    contribution := writes.contribution.map (Expr.specialize mode scale)
    factor := writes.factor.map (Expr.specialize mode scale) }

private structure Program where
  code : Array String
  edges : Array (List Nat)
  start : Nat
  stop : Nat
  targetSlots : Nat
  transport : TrainingCEPublishBinding.LogitTransport

open TrainingCEPublishBinding (cachedTargetsSymbol)

/-- Require direct access to the cache symbol. The first block barrier
separates its only writes from all reads. Symbol addresses never escape
to registers, so later stores to other shared objects cannot alias it. -/
private def targetLayout (lines : List String) (edges : Array (List Nat)) (barrier : Nat) : Option Nat := do
  let references := lines.zipIdx.filter fun (line, _) => (line.splitOn cachedTargetsSymbol).length > 1
  let declarations := references.filter fun (line, _) => line.startsWith ".shared "
  if references.isEmpty then return 0
  guard (edges.toList.zipIdx.all fun (successors, pc) => pc < barrier || successors.all (barrier < ·))
  let [(layout, _)] := declarations | none
  let count ← ([1, 2, 3] : List Nat).find? fun n =>
    layout == s!".shared .align 4 .b8 {cachedTargetsSymbol}[{4 * n}];" ||
    (n == 1 && layout == s!".shared .align 4 .u32 {cachedTargetsSymbol};")
  for (line, pc) in references do
    if line != layout then
      let ptr ← match operands line with
        | ["st.shared.u32", ptr, _] => if pc < barrier then some ptr else none
        | [op, _, ptr] =>
          if (op == "ld.shared.u32" || op == "ld.shared.s32") && barrier < pc &&
              dominates edges barrier pc then some ptr else none
        | _ => none
      let (.cachedTargets, .constant byte) ← TrainingCEPublishBinding.address [] ptr | none
      guard (byte % 4 == 0 && byte < 4 * count)
  return count

private def prepare (ptx : String) : Option Program := do
  let lines := QKVTTIR.normalizePTX ptx
  guard (lines.contains ".maxntid 256, 1, 1" &&
    lines.contains ".shared .align 8 .b8 _ZZ17ce_fwd_bwd_kernelE13sigmoid_table[2048];" &&
    lines.contains ".shared .align 4 .b8 _ZZ17ce_fwd_bwd_kernelE9exp_table[1024];" &&
    lines.contains ".shared .align 4 .f32 _ZZ17ce_fwd_bwd_kernelE15gradient_factor;")
  let (_, body) ← lines.zipIdx.find? fun (line, _) => line == "{"
  let (_, barrier) ← lines.zipIdx.find? fun (line, _) => line == "bar.sync 0;"
  let edges ← flow lines
  guard (body < barrier && dominates edges body barrier)
  let cache ← targetLayout lines edges barrier
  return ⟨lines.toArray, edges, body + 1, barrier, cache, ← TrainingCEPublishBinding.logitTransport lines⟩

private def execute (program : Program) (w : Width) (live : Fin 4) (laneZero : Bool) :
    Nat → Nat → Registers → Writes → Option Writes
  | 0, _, _, _ => none
  | fuel + 1, pc, regs, writes => do
    if pc == program.stop then
      let sigmoid : Option (Expr × Expr) ← match program.transport with
        | .shared => some writes.sigmoid
        | .registers sigma scaled => do
          guard writes.sigmoid.isNone
          pure (some (← expression regs sigma, ← expression regs scaled))
      return { writes with sigmoid, targets := writes.targets.reverse }
    guard (program.start ≤ pc && pc < program.stop)
    let line ← program.code[pc]?
    if line.startsWith "{" then
      execute program w live laneZero fuel (pc + 1) (← inlineAsm regs line) writes
    else
      match operands line with
      | [test, "bra", _] =>
        guard (test.startsWith "@")
        let testName := (test.drop 1).toString
        let negated := testName.startsWith "!"
        let name := if negated then (testName.drop 1).toString else testName
        let active ← match regs.lookup name with
          | some (.addressReg .laneNonzero) => some (!laneZero)
          | some (.addressReg (.predicate active)) => some active
          | _ => none
        let [taken, fallthrough] ← program.edges[pc]? | none
        execute program w live laneZero fuel (if active != negated then taken else fallthrough) regs writes
      | ["st.shared.v2.f32", ptr, sigma, scaled] =>
        let (.sigmoid, .indexed .lane 8 0) ← TrainingCEPublishBinding.address (addresses regs) ptr | none
        guard writes.sigmoid.isNone
        let pair := (← expression regs sigma, ← expression regs scaled)
        execute program w live laneZero fuel (pc + 1) regs { writes with sigmoid := some pair }
      | ["st.shared.f32", ptr, src] =>
        let whereAt ← TrainingCEPublishBinding.address (addresses regs) ptr
        let value ← expression regs src
        match whereAt with
        | (.exp, .indexed .lane 4 0) =>
          guard writes.contribution.isNone
          execute program w live laneZero fuel (pc + 1) regs { writes with contribution := some value }
        | (.factor, .constant 0) =>
          guard writes.factor.isNone
          execute program w live laneZero fuel (pc + 1) regs { writes with factor := some value }
        | _ => none
      | ["st.shared.u32", ptr, src] =>
        let (.cachedTargets, .constant byte) ← TrainingCEPublishBinding.address (addresses regs) ptr | none
        let .target32 slot ← regs.lookup src | none
        guard (byte % 4 == 0 && byte < 4 * program.targetSlots)
        execute program w live laneZero fuel (pc + 1) regs
          { writes with targets := (byte / 4, slot) :: writes.targets }
      | [label] =>
        guard (label.endsWith ":")
        execute program w live laneZero fuel (pc + 1) regs writes
      | ".reg" :: _ | ".shared" :: _ | ".local" :: _ | ".extern" :: _ =>
        execute program w live laneZero fuel (pc + 1) regs writes
      | insn@(_ :: dst :: _) =>
        execute program w live laneZero fuel (pc + 1) (save regs dst (← transfer w live regs insn)) writes
      | _ => none

private def Program.run (program : Program) (w : Width) (live : Fin 4) (laneZero : Bool) : Option Writes :=
  execute program w live laneZero (program.stop - program.start + 1) program.start [] {}

def decode (w : Width) (ptx : String) (live : Fin 4) (laneZero : Bool) : Option Writes := do
  let program ← prepare ptx
  program.run w live laneZero

private def reciprocalC : Expr := .reciprocal (.param .c)

private def sigmaExpr : Expr :=
  let half := Expr.literal 0x3f000000
  let u := Expr.fma reciprocalC (.decodedByte false) (.mul reciprocalC (.param .b))
  .roundedHalf (.fma (.tanh (.mul u half)) half half)

def expected (slots : Nat) (live : Fin 4) (laneZero : Bool) : Writes :=
  let scaled := Expr.mul sigmaExpr (.param .a)
  let exp := Expr.exp2 (.mul (.sub scaled (.param .a)) (.literal 0x3fb8aa3b))
  let factor := Expr.mul (.mul reciprocalC (.param .a)) (.reciprocal (.param .gradScale))
  let targets := if laneZero then (List.range slots).map fun k =>
    (k, if k < live.val then some k else none) else []
  ⟨some (sigmaExpr, scaled), some exp, if laneZero then some factor else none, targets⟩

structure Certificate (w : Width) (ptx : String) (mode : Softcap := .runtime)
    (scale : GradScale := .runtime) : Type where
  targetSlots : Nat
  traces : ∀ live zero, decode w ptx live zero = some ((expected targetSlots live zero).specialize mode scale)
  capacity : targetSlots ≤ 3

private def Program.matches (program : Program) (w : Width) (mode : Softcap) (scale : GradScale) : Bool :=
  (List.finRange 4).all fun live => [false, true].all fun zero =>
    decide (program.run w live zero = some ((expected program.targetSlots live zero).specialize mode scale))

def certify (w : Width) (ptx : String) (mode : Softcap := .runtime) (scale : GradScale := .runtime) :
    Option (Certificate w ptx mode scale) :=
  match ready : prepare ptx with
  | none => none
  | some program =>
    if capacity : program.targetSlots ≤ 3 then
      if checked : program.matches w mode scale = true then
        some ⟨program.targetSlots, (fun live zero => by
          unfold Program.matches at checked
          have byRow := List.all_eq_true.mp checked live (List.mem_finRange live)
          have exactWrite := List.all_eq_true.mp byRow zero (by cases zero <;> simp)
          simpa only [decode, ready, Option.bind_some] using of_decide_eq_true exactWrite), capacity⟩
      else none
    else none

/-- Loss and correction share the validated target cache layout. The
gradient factor is checked independently against the selected launch. -/
def checkedTargetSlots (w : Width) (ptx : String) (mode : Softcap) : Option Nat := do
  let program ← prepare ptx
  guard (program.targetSlots ≤ 3)
  guard (program.matches w mode .runtime || program.matches w mode .lmHead)
  return program.targetSlots

def Writes.eval (op : Native) (p : Parameters) (code : UInt8) (writes : Writes) :
    Option (LogitEntry × Option F32) := do
  let (si, scaled) ← writes.sigmoid
  let exp ← writes.contribution
  return (⟨⟨si.eval op p code, scaled.eval op p code⟩, exp.eval op p code⟩,
    writes.factor.map (Expr.eval op p code))

private theorem Writes.eval_specialize (op : Native) (p : Parameters) (code : UInt8)
    (mode : Softcap) (scale : GradScale) (writes : Writes) :
    (writes.specialize mode scale).eval op p code = writes.eval op (scale.parameters (mode.parameters p)) code := by
  cases writes with
  | mk sigmoid contribution factor targets =>
    cases sigmoid with
    | none => rfl
    | some pair =>
      obtain ⟨sigma, scaled⟩ := pair
      cases contribution <;> cases factor <;>
        simp [Writes.specialize, Writes.eval, Expr.eval_specialize]

theorem initialized_table_exact {w : Width} {ptx : String} {mode : Softcap} {scale : GradScale}
    (artifact : Certificate w ptx mode scale) (op : Native) (p : Parameters) (live : Fin 4) (lane : Lane)
    (accepted : mode.accepts p = true) (scaleAccepted : scale.accepts p = true) :
    ((decode w ptx live (lane.val == 0)).bind (Writes.eval op p (UInt8.ofNat lane.val))) =
      some (readLogitTable (logitTable op p) (UInt8.ofNat lane.val),
        if lane.val == 0 then some (gradientFactor op p) else none) := by
  rw [artifact.traces, Option.bind_some, Writes.eval_specialize,
    mode.parameters_exact p accepted, scale.parameters_exact p scaleAccepted, table_lookup_exact]
  cases lane.val == 0 <;> simp only [Writes.eval, expected] <;> rfl

/-- The cache slot reads the corresponding row target or the invalid
sentinel. This is the value provenance from the lane-zero store trace. -/
theorem decoded_target_slot {w : Width} {ptx : String} {mode : Softcap} {scale : GradScale}
    (artifact : Certificate w ptx mode scale)
    (live : Fin 4) (slot : Nat) (inside : slot < artifact.targetSlots) :
    (decode w ptx live true).bind (fun writes => writes.targets.lookup slot) =
      some (if slot < live.val then some slot else none) := by
  rw [artifact.traces]
  simp only [Option.bind_some, Writes.specialize, expected, if_true]
  let select := fun k : Nat => if k < live.val then some k else none
  change ((List.range artifact.targetSlots).map (fun k => (k, select k))).lookup slot = some (select slot)
  have lookup (slots : List Nat) (written : slot ∈ slots) :
      (slots.map (fun k => (k, select k))).lookup slot = some (select slot) := by
    induction slots with
    | nil => simp at written
    | cons first rest ih =>
      by_cases same : slot = first
      · subst first
        simp
      · have different : (slot == first) = false := by
          cases h : slot == first <;> simp_all
        simp [List.lookup_cons, different, ih (by simpa [same] using written)]
  exact lookup _ (List.mem_range.mpr inside)

def targetWord {w : Width} (target : Option (Column w)) : UInt32 :=
  (target.map (UInt32.ofNat ∘ Fin.val)).getD 0xffffffff

private def signExtend32 (word : UInt32) : Int :=
  if word.toNat < 2147483648 then word.toNat else (word.toNat : Int) - 4294967296

/-- The loss's unsigned vocabulary test and the gradient's signed load
agree on both live columns and the initialized all-ones sentinel. -/
theorem target_word_exact {w : Width} (target : Option (Column w)) :
    ((targetWord target).toNat < w.columns ↔ target.isSome) ∧
      signExtend32 (targetWord target) = (target.map (Int.ofNat ∘ Fin.val)).getD (-1) := by
  cases target with
  | none => cases w <;> decide
  | some col =>
    have maxWidth : w.columns ≤ 2147483648 := by cases w <;> decide
    have live := col.isLt
    have signedBound : col.val < 2147483648 := Nat.lt_of_lt_of_le live maxWidth
    have raw : (UInt32.ofNat col.val).toNat = col.val := by
      rw [UInt32.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
    simp [targetWord, signExtend32, raw, signedBound, live]

#axiomsClean initialized_table_exact
#axiomsClean decoded_target_slot
#axiomsClean target_word_exact
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEInitBinding
