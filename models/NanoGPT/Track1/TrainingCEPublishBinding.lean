import models.NanoGPT.Track1.TrainingCEGatherBinding

/-! The fixed-count CE instructions between the second and third block
barriers. Symbolic addresses carry the CUDA parameter or shared symbol
and its row or lane offset. Numerical expressions preserve the order
and rounding of every operation in the two cache publications. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEPublishBinding
open TrainingCE TrainingCEGatherBinding TrainingCEConversions
open VeriTile.Triton FP

inductive TargetSlot where
  | mtp (slot : Nat)
  | prefix
  deriving DecidableEq

inductive Expr where
  | literal (bits : Nat)
  | a | prefixWeight | sigma | scaled | factor
  | weight (i : Nat)
  | warp (i : Fin 8)
  | sigmaAt (target : TargetSlot)
  | probabilityAt (target : TargetSlot)
  | add (x y : Expr)
  | sub (x y : Expr)
  | mul (x y : Expr)
  | fma (x y z : Expr)
  | exp2 (x : Expr)
  | log2 (x : Expr)
  | prefix (yes no : Expr)
  deriving DecidableEq

structure Inputs where
  parameters : Parameters
  weights : Nat → F32
  hasPrefix : Bool
  leaders : Fin 8 → F32
  logit : ScaledLogit
  factor : F32
  targetSigma : TargetSlot → F32
  targetProbability : TargetSlot → F32

def Inputs.forPublication (p : Parameters) (weights : Nat → F32) (hasPrefix : Bool)
    (leaders : Fin 8 → F32) (logit : ScaledLogit) (factor : F32) : Inputs :=
  ⟨p, weights, hasPrefix, leaders, logit, factor, fun _ => logit.sigma, fun _ => logit.scaled⟩

def Expr.eval (op : Native) (input : Inputs) : Expr → F32
  | .literal bits => Value.ofBits .fp32 bits
  | .a => input.parameters.a
  | .prefixWeight => input.parameters.prefixWeight
  | .sigma => input.logit.sigma
  | .scaled => input.logit.scaled
  | .factor => input.factor
  | .weight i => input.weights i
  | .warp i => input.leaders i
  | .sigmaAt target => input.targetSigma target
  | .probabilityAt target => input.targetProbability target
  | .add x y => FP.add ftz (x.eval op input) (y.eval op input)
  | .sub x y => FP.sub ftz (x.eval op input) (y.eval op input)
  | .mul x y => FP.mul ftz (x.eval op input) (y.eval op input)
  | .fma x y z => FP.fma ftz (x.eval op input) (y.eval op input) (z.eval op input)
  | .exp2 x => op.exp2 (x.eval op input)
  | .log2 x => op.log2 (x.eval op input)
  | .prefix yes no => if input.hasPrefix then yes.eval op input else no.eval op input

def Expr.specialize (mode : Softcap) : Expr → Expr
  | .a => if mode = .defaultABC then .literal 0x41b80000 else .a
  | .add x y => .add (x.specialize mode) (y.specialize mode)
  | .sub x y => .sub (x.specialize mode) (y.specialize mode)
  | .mul x y => .mul (x.specialize mode) (y.specialize mode)
  | .fma x y z => .fma (x.specialize mode) (y.specialize mode) (z.specialize mode)
  | .exp2 x => .exp2 (x.specialize mode)
  | .log2 x => .log2 (x.specialize mode)
  | .prefix yes no => .prefix (yes.specialize mode) (no.specialize mode)
  | other => other

theorem Expr.eval_specialize (op : Native) (input : Inputs) (mode : Softcap) (expr : Expr) :
    (expr.specialize mode).eval op input =
      expr.eval op {input with parameters := mode.parameters input.parameters} := by
  induction expr with
  | a => cases mode <;> simp [Expr.specialize, Expr.eval, Softcap.parameters, Parameters.a, Softcap.a_word]
  | prefixWeight => cases mode <;> rfl
  | _ => simp [Expr.specialize, Expr.eval, *]

theorem Expr.eval_for_launch (op : Native) (input : Inputs) (mode : Softcap) (expr : Expr)
    (accepted : mode.accepts input.parameters = true) :
    (expr.specialize mode).eval op input = expr.eval op input := by
  rw [Expr.eval_specialize, mode.parameters_exact input.parameters accepted]

structure Publication where
  probability : Expr
  baseGradient : Expr
  deriving DecidableEq

def Publication.eval (op : Native) (input : Inputs) (writes : Publication) : F32 × UInt8 :=
  (writes.probability.eval op input, floatToE5m2 (writes.baseGradient.eval op input))

private def Publication.specialize (mode : Softcap) (writes : Publication) : Publication :=
  ⟨writes.probability.specialize mode, writes.baseGradient.specialize mode⟩

private def sigmoidSymbol := "_ZZ17ce_fwd_bwd_kernelE13sigmoid_table"
private def expSymbol := "_ZZ17ce_fwd_bwd_kernelE9exp_table"
private def sumsSymbol := "_ZZ17ce_fwd_bwd_kernelE10block_sums"
private def factorSymbol := "_ZZ17ce_fwd_bwd_kernelE15gradient_factor"
def cachedTargetsSymbol := "_ZZ17ce_fwd_bwd_kernelE14cached_targets"

inductive Area where
  | weights | prefixes | prefixWeight | sigmoid | exp | warpSums | factor
  | targets | losses | codes | gradient | cachedTargets
  deriving DecidableEq

inductive Index where
  | lane | row
  deriving DecidableEq

inductive Offset where
  | constant (i : Nat)
  | indexed (root : Index) (scale delta : Nat)
  | target (slot : TargetSlot)
  | sigmoidOf (slot : TargetSlot)
  | rowGradient (width : Nat)
  | gradientTarget (width : Nat) (slot : Option TargetSlot)
  | invalidTarget
  deriving DecidableEq

private def Offset.add : Offset → Offset → Option Offset
  | .constant x, .constant y => some (.constant (x + y))
  | .indexed root scale delta, .constant i => some (.indexed root scale (delta + i))
  | _, _ => none

inductive Reg where
  | expr (value : Expr)
  | address (area : Area) (offset : Offset)
  | offset (offset : Offset)
  | rawA
  | prefixTarget
  | prefixTest
  | batchRows
  | laneNonzero
  | rowProduct32 (width : Nat)
  | target (slot : TargetSlot)
  | cachedTarget32 (slot : Nat)
  | targetWord (slot : Option TargetSlot)
  | targetChunk (slot : Option TargetSlot)
  | targetOwner (slot : Option TargetSlot)
  | logitCode (slot : TargetSlot)
  | predicate (value : Bool)
  | e5m2Low (source : Expr)

abbrev Registers := List (String × Reg)

private def hexBits (name : String) : Option Nat := do
  guard (name.startsWith "0f" && name.length == 10)
  ((name.drop 2).toString.toList).foldlM (fun acc c => do
    let d := c.toLower
    let digit := if '0' ≤ d && d ≤ '9' then d.toNat - '0'.toNat
      else if 'a' ≤ d && d ≤ 'f' then 10 + (d.toNat - 'a'.toNat) else 16
    guard (digit < 16)
    return 16 * acc + digit) 0

def getExpr (regs : Registers) (name : String) : Option Expr :=
  if let some (.expr expression) := regs.lookup name then some expression
  else (hexBits name).map .literal

def getOffset (regs : Registers) (name : String) : Option Offset :=
  if let some (.offset index) := regs.lookup name then some index
  else name.toNat?.map .constant

private def symbol (name : String) : Option Area :=
  if name == sigmoidSymbol then some .sigmoid else if name == expSymbol then some .exp
  else if name == sumsSymbol then some .warpSums else if name == factorSymbol then some .factor
  else if name == cachedTargetsSymbol then some .cachedTargets
  else if name == "code_smem" then some .codes
  else none

def address (regs : Registers) (pointer : String) : Option (Area × Offset) := do
  guard (pointer.startsWith "[" && pointer.endsWith "]")
  let text := (pointer.drop 1 |>.dropEnd 1).toString
  let (base, byte) ← match text.splitOn "+" with
    | [name] => some (name, 0)
    | [name, i] => i.toNat?.map fun n => (name, n)
    | _ => none
  let (area, index) ← match regs.lookup base with
    | some (.address area index) => some (area, index)
    | _ => (symbol base).map fun area => (area, Offset.constant 0)
  return (area, ← if byte == 0 then some index else index.add (.constant byte))

def scalar (regs : Registers) : List String → Option Reg
  | ["mov.u32", _, "%tid.x"] => some (.offset (.indexed .lane 1 0))
  | ["mov.u32", _, "%ctaid.x"] => some (.offset (.indexed .row 1 0))
  | ["mov.f32", _, src] => (getExpr regs src).map .expr
  | ["mov.u64", _, sym] => (symbol sym).map fun s => .address s (.constant 0)
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_1]"] => some (.address .targets (.constant 0))
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_2]"] => some (.address .weights (.constant 0))
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_3]"] => some (.address .prefixes (.constant 0))
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_4]"] => some (.address .prefixWeight (.constant 0))
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_5]"] => some (.address .losses (.constant 0))
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_6]"] => some (.address .gradient (.constant 0))
  | ["ld.param.u32", _, "[ce_fwd_bwd_kernel_param_7]"] => some .batchRows
  | ["ld.param.f64", _, "[ce_fwd_bwd_kernel_param_9]"] => some .rawA
  | ["cvt.rn.ftz.f32.f64", _, src] => do
    let .rawA ← regs.lookup src | none
    return .expr .a
  | ["cvta.to.global.u64", _, src] =>
    match regs.lookup src with
    | some (.address .weights offset) => some (.address .weights offset)
    | some (.address .prefixes offset) => some (.address .prefixes offset)
    | some (.address .prefixWeight offset) => some (.address .prefixWeight offset)
    | some (.address .targets offset) => some (.address .targets offset)
    | some (.address .losses offset) => some (.address .losses offset)
    | some (.address .gradient offset) => some (.address .gradient offset)
    | _ => none
  | ["cvt.u64.u32", _, src] => do
    match regs.lookup src with
    | some (.offset (.indexed root 1 delta)) => some (.offset (.indexed root 1 delta))
    | some (.rowProduct32 width) => some (.offset (.rowGradient width))
    | _ => none
  | ["cvt.u32.u64", _, src] => do
    let .indexed .lane 1 0 ← getOffset regs src | none
    return .offset (.indexed .lane 1 0)
  | ["cvt.s64.s32", _, src] => do
    match regs.lookup src with
    | some .batchRows => some .batchRows
    | some (.cachedTarget32 slot) => some (.target (.mtp slot))
    | some (.offset (.indexed root 1 delta)) => some (.offset (.indexed root 1 delta))
    | _ => none
  | ["mul.lo.s32", _, src, imm] => do
    let .indexed .row 1 0 ← getOffset regs src | none
    return .rowProduct32 (← imm.toNat?)
  | ["mul.wide.u32", _, src, imm] => do
    let scale ← imm.toNat?
    guard (scale == 4 || scale == 8)
    match regs.lookup src with
    | some (.offset (.indexed root 1 0)) => some (.offset (.indexed root scale 0))
    | some (.logitCode target) => if scale == 8 then some (.offset (.sigmoidOf target)) else none
    | _ => none
  | ["add.s32", _, src, imm] => do
    let .indexed .row 1 delta ← getOffset regs src | none
    let step ← imm.toNat?
    guard (delta + step ≤ 2)
    return .offset (.indexed .row 1 (delta + step))
  | ["add.s64", _, base, offset] => do
    let .address area i ← regs.lookup base | none
    let delta ← match regs.lookup offset with
      | some (.target target) => some (.target target)
      | some .prefixTarget => some (.target .prefix)
      | _ => getOffset regs offset
    let place ← match i, delta with
      | .constant 0, _ => some delta
      | _, _ => i.add delta
    return .address area place
  | ["ld.global.nc.f32", _, ptr] => do
    match ← address regs ptr with
    | (.prefixWeight, .constant 0) => some (.expr .prefixWeight)
    | (.weights, .constant i) =>
      if i % 4 == 0 && i < 12 then some (.expr (.weight (i / 4))) else none
    | _ => none
  | ["ld.global.nc.u64", _, ptr] => do
    let (.prefixes, .indexed .row 8 0) ← address regs ptr | none
    return .prefixTarget
  | ["setp.gt.s64", _, src, "-1"] => do
    let .prefixTarget ← regs.lookup src | none
    return .prefixTest
  | ["setp.ne.s32", _, src, "0"] => do
    let .indexed .lane 1 0 ← getOffset regs src | none
    return .laneNonzero
  | ["selp.f32", _, yes, no, test] => do
    let .prefixTest ← regs.lookup test | none
    return .expr (Expr.prefix (← getExpr regs yes) (← getExpr regs no))
  | ["ld.shared.f32", _, ptr] => do
    match ← address regs ptr with
    | (.factor, .constant 0) => some (.expr .factor)
    | (.sigmoid, .sigmoidOf target) => some (.expr (.sigmaAt target))
    | (.warpSums, .constant byte) =>
      if inRange : byte / 4 < 8 then
        if byte % 4 == 0 then some (.expr (.warp ⟨byte / 4, inRange⟩)) else none
      else none
    | _ => none
  | ["lg2.approx.ftz.f32", _, arg] => (getExpr regs arg).map fun x => .expr (.log2 x)
  | ["ex2.approx.ftz.f32", _, arg] => (getExpr regs arg).map fun x => .expr (.exp2 x)
  | ["fma.rn.ftz.f32", _, x, y, z] => do
    return .expr (.fma (← getExpr regs x) (← getExpr regs y) (← getExpr regs z))
  | ["cvt.rn.satfinite.e5m2x2.f32", _, hi, lo] => do
    -- The store selects the low byte of the round-and-saturate pair.
    let _ ← getExpr regs hi
    return .e5m2Low (← getExpr regs lo)
  | [op, _, x, y] => do
    let left ← getExpr regs x
    let right ← getExpr regs y
    match op with
    | "add.ftz.f32" => some (.expr (.add left right))
    | "sub.ftz.f32" => some (.expr (.sub left right))
    | "mul.ftz.f32" | "mul.rn.ftz.f32" => some (.expr (.mul left right))
    | _ => none
  | _ => none

def insert (regs : Registers) (dst : String) (value : Reg) : Registers :=
  (dst, value) :: regs.filter (fun entry => entry.1 != dst)

def liveIns (lines : List String) (edges : Array (List Nat)) (barrier : Nat) : Registers := Id.run do
  let mut regs := []
  for (line, pc) in (lines.take barrier).zipIdx do
    if let _ :: dst :: _ := operands line then
      if let some value := scalar regs (operands line) then
        if let some (_, definition) := uniqueDefinition lines edges dst barrier then
          if definition == pc then regs := insert regs dst value
  return regs

/-- The lookup values arrive through the initialized shared table or
through untouched lane registers. The prefix decoder checks the expressions
in the selected registers at the first barrier. -/
inductive LogitTransport where
  | shared
  | registers (sigma scaled : String)
  deriving DecidableEq

def logitTransport (lines : List String) : Option LogitTransport := do
  let [first, second, third] := lines.zipIdx.filterMap (fun (line, pc) =>
    if line == "bar.sync 0;" then some pc else none) | none
  let between := (lines.take second).drop (first + 1)
  let publish := (lines.take third).drop (second + 1)
  match (publish.filter (·.startsWith "st.shared.v2.f32 ")).map operands with
  | [] => some .shared
  | [["st.shared.v2.f32", _, sigma, _]] => do
    let some ["sub.ftz.f32", _, scaled, _] :=
      (publish.find? (·.startsWith "sub.ftz.f32 ")).map operands | none
    guard (sigma.startsWith "%f" && scaled.startsWith "%f" && sigma != scaled)
    guard (between.all fun line =>
      let words := operands line
      !words.contains sigma && !words.contains scaled)
    return .registers sigma scaled
  | _ => none

def decodeStage (lines : List String) (edges : Array (List Nat)) :
    Option (Publication × Registers × Nat) := do
  let packed := lines.contains s!".shared .align 16 .b8 {sumsSymbol}[32];"
  guard (lines.contains s!".shared .align 8 .b8 {sigmoidSymbol}[2048];" &&
    (packed || lines.contains s!".shared .align 4 .b8 {sumsSymbol}[32];") &&
    lines.contains s!".shared .align 4 .f32 {factorSymbol};")
  let table ← tableRegister lines
  let finish ← cacheBarrier lines edges table
  let [_, start] := (lines.take finish).zipIdx.filterMap (fun (line, pc) =>
    if line == "bar.sync 0;" then some pc else none) | none
  let transport ← logitTransport lines
  let mut regs := liveIns lines edges start
  if let .registers sigma scaled := transport then
    regs := insert (insert regs sigma (.expr .sigma)) scaled (.expr .scaled)
  let mut probability : Option Expr := none
  let mut base : Option Expr := none
  let mut vectorLoads := []
  for line in (lines.take finish).drop (start + 1) do
    match operands line with
    | ["ld.shared.v4.f32", s0, s1, s2, s3, ptr] =>
      let (.warpSums, .constant byte) ← address regs ptr | none
      let dests := [s0, s1, s2, s3]
      guard (packed && (byte == 0 || byte == 16) && dests.eraseDups.length == 4)
      for (dst, i) in dests.zipIdx do
        guard (dst.startsWith "%f")
        let slot := byte / 4 + i
        let value ← if bound : slot < 8 then some (Expr.warp ⟨slot, bound⟩) else none
        regs := insert regs dst (.expr value)
      vectorLoads := byte :: vectorLoads
    | ["ld.shared.v2.f32", sigma, scaled, ptr] =>
      guard (transport == .shared)
      let (.sigmoid, .indexed .lane 8 0) ← address regs ptr | none
      guard (sigma != scaled)
      regs := insert (insert regs sigma (.expr .sigma)) scaled (.expr (probability.getD .scaled))
    | ["st.shared.v2.f32", ptr, sigma, prob] =>
      let .registers .. := transport | none
      let (.sigmoid, .indexed .lane 8 0) ← address regs ptr | none
      guard (probability.isNone && getExpr regs sigma == some .sigma)
      probability := some (← getExpr regs prob)
    | ["st.shared.f32", ptr, src] =>
      guard (transport == .shared)
      let (.sigmoid, .indexed .lane 8 4) ← address regs ptr | none
      guard probability.isNone
      probability := some (← getExpr regs src)
    | ["st.shared.u8", ptr, src] =>
      let (.exp, .indexed .lane 1 0) ← address regs ptr | none
      let .e5m2Low value ← regs.lookup src | none
      guard base.isNone
      base := some value
    | insn@(_ :: dst :: _) =>
      regs := insert regs dst (← scalar regs insn)
    | _ => none
  guard (!packed || vectorLoads == [16, 0])
  return (⟨← probability, ← base⟩, regs, finish)

def decode (ptx : String) : Option Publication := do
  let lines := QKVTTIR.normalizePTX ptx
  let stage ← decodeStage lines (← flow lines)
  return stage.1

private def orderedSum {α : Type} (load : α → Expr) : List α → Expr
  | [] => .literal 0
  | i :: rest => rest.foldl (fun acc slot => .add acc (load slot)) (.add (load i) (.literal 0))

def lseExpr : Expr :=
  let sum := orderedSum .warp (List.finRange 8)
  Expr.fma (.log2 sum) (.literal 0x3f317218) .a

def weightExpr (predictions : Fin 4) : Expr :=
  let weights := orderedSum .weight ((fixedWalk predictions).forward)
  .prefix (.add .prefixWeight weights) weights

private def expected (predictions : Fin 4) : Publication :=
  let prob := Expr.exp2 (.mul (.sub .scaled lseExpr) (.literal 0x3fb8aa3b))
  let norm := Expr.mul (weightExpr predictions) prob
  ⟨prob, .mul (.sub (.literal 0x3f800000) .sigma) (.mul .sigma (.mul .factor norm))⟩

structure Certificate (ptx : String) (predictions : Fin 4) (mode : Softcap := .runtime) where
  writes : Publication
  decoded : decode ptx = some writes
  exactExpressions : writes = (expected predictions).specialize mode

def certify (ptx : String) (predictions : Fin 4) (mode : Softcap := .runtime) :
    Option (Certificate ptx predictions mode) :=
  match decoded : decode ptx with
  | none => none
  | some writes =>
    if exact : writes = (expected predictions).specialize mode then some ⟨writes, decoded, exact⟩ else none

private theorem leader_sum (op : Native) (input : Inputs) :
    (orderedSum .warp (List.finRange 8)).eval op input = sumWarpLeaders input.leaders := by
  rfl

theorem lse_eval (op : Native) (input : Inputs) :
    lseExpr.eval op input = logSumExpFromLeaders op input.parameters input.leaders := by
  simp only [lseExpr, Expr.eval, leader_sum]
  rfl

theorem weight_eval {w : Width} (op : Native) (p : Parameters) (r : Row w)
    (leaders : Fin 8 → F32) (logit : ScaledLogit) (factor : F32)
    (sigma probability : TargetSlot → F32) :
    (weightExpr r.predictions).eval op
      ⟨p, r.weights, r.prefixTarget.isSome, leaders, logit, factor, sigma, probability⟩ =
      weightSum p r (fixedWalk r.predictions) := by
  generalize count : r.predictions = n
  have choices : n.val = 0 ∨ n.val = 1 ∨ n.val = 2 ∨ n.val = 3 := by omega
  rcases choices with h | h | h | h
  · rw [show n = 0 from Fin.ext h]
    rfl
  · rw [show n = 1 from Fin.ext h]
    rfl
  · rw [show n = 2 from Fin.ext h]
    rfl
  · rw [show n = 3 from Fin.ext h]
    rfl

/-- Evaluate the decoded FP32 and E5M2 stores with the staged inputs. All
floating-point operands may vary, including nonfinite payloads. The same
native exp2 and log2 operations are used in the publication and the model. -/
theorem publication_exact {ptx : String} {w : Width} {mode : Softcap}
    (op : Native) (p : Parameters) (r : Row w)
    (certificate : Certificate ptx r.predictions mode) (leaders : Fin 8 → F32)
    (logit : ScaledLogit) (scale : F32) (accepted : mode.accepts p = true := by rfl) :
    let input := Inputs.forPublication p r.weights r.prefixTarget.isSome leaders logit scale
    let entry := logit.forRow op (logSumExpFromLeaders op p leaders)
    certificate.writes.eval op input =
      (entry.probability, baseGradient p r (fixedWalk r.predictions) scale entry) := by
  rw [certificate.exactExpressions]
  have launch : mode.accepts
      (Inputs.forPublication p r.weights r.prefixTarget.isSome leaders logit scale).parameters = true :=
    accepted
  simp only [Publication.eval, Publication.specialize, Expr.eval_for_launch _ _ _ _ launch]
  have sum : (weightExpr r.predictions).eval op
      (Inputs.forPublication p r.weights r.prefixTarget.isSome leaders logit scale) =
      weightSum p r (fixedWalk r.predictions) := weight_eval op p r leaders logit scale _ _
  have leadersExact := leader_sum op (Inputs.forPublication p r.weights r.prefixTarget.isSome leaders logit scale)
  simp only [expected, lseExpr, Expr.eval, sum, leadersExact]
  rfl

#axiomsClean publication_exact
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEPublishBinding
