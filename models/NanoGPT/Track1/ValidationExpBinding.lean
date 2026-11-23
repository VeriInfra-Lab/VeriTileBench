import models.NanoGPT.Track1.ValidationLogBinding
import Std.Data.HashMap

/-! Decode the inlined libdevice exponentials in validation's vocabulary
loop and in the final rescale. Each site keeps its actual input and output
registers so the scan and row reduction can consume the same certificate. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValidationExpBinding
open TrainingCEGatherBinding
open VeriTile.Triton FP
open TailNorm (F32 zero one)

private abbrev Instruction := List String × Nat

private structure Code where
  edges : Array (List Nat)
  definitions : Std.HashMap String (List Instruction)
  products : Std.HashMap String (List Instruction)
  selects : Std.HashMap String (List Instruction)

private def buildIndex (lines : List String) (edges : Array (List Nat)) : Code := Id.run do
  let mut definitions := {}
  let mut products := {}
  let mut selects := {}
  for (line, pc) in lines.zipIdx do
    let insn := operands line
    let regs := ((insn.flatMap (·.splitOn "|")).filter (·.startsWith "%")).eraseDups
    for reg in regs do
      if defines reg line then
        definitions := definitions.insert reg ((insn, pc) :: definitions.getD reg [])
    if let ["mul.f32", _, left, _] := insn then
      products := products.insert left ((insn, pc) :: products.getD left [])
    if let ["selp.f32", _, _, no, _] := insn then
      selects := selects.insert no ((insn, pc) :: selects.getD no [])
  return ⟨edges, definitions, products, selects⟩

private def Code.defn (code : Code) (reg : String) (use : Nat) : Option Instruction := do
  let [(insn, pc)] := code.definitions.getD reg [] | none
  guard (pc < use && dominates code.edges pc use)
  return (insn, pc)

private def Code.constant (code : Code) (reg : String) (use : Nat) (literal : String) : Option Unit :=
  if reg == literal then some () else do
    let (["mov.b32", _, word], _) ← code.defn reg use | none
    guard (word == literal)

private def Code.stable (code : Code) (reg : String) (left right : Nat) : Bool :=
  (code.definitions.getD reg []).all fun (_, pc) => pc < min left right || max left right ≤ pc

inductive Phase where
  | online | rescale
  deriving DecidableEq

structure Site where
  phase : Phase
  source : String
  maximum : String
  inputPC : Nat
  output : String
  outputPC : Nat

structure OnlineUpdate where
  previous : Site
  incoming : Site
  oldSum : String
  newSum : String
  newSumPC : Nat
  activeColumn : Option String
  activeColumnPC : Option Nat

structure LoopCounter where
  register : String
  extent : Nat
  limit : Nat
  prebiased : Bool

structure ScaledSlot where
  update : OnlineUpdate
  scale : Site
  accumulator : Option String
  result : String
  resultPC : Nat

structure Decoded where
  header : Nat
  latch : Nat
  counter : LoopCounter
  sites : List Site
  updates : List OnlineUpdate
  scaled : List ScaledSlot

private def inputRegisters (code : Code) (phase : Phase) (input : String) (use : Nat) :
    Option (String × String × Nat) := do
  match phase with
  | .online =>
    let (["sub.f32", _, source, maximum], pc) ← code.defn input use | none
    return (source, maximum, pc)
  | .rescale =>
    let (["selp.f32", _, z, delta, negativeInf], pc) ← code.defn input use | none
    code.constant z pc "0f00000000"
    let (["sub.f32", _, source, maximum], sub) ← code.defn delta pc | none
    let (["setp.eq.f32", _, sameMaximum, bottom], compared) ← code.defn negativeInf pc | none
    code.constant bottom compared "0fFF800000"
    guard (sameMaximum == maximum && code.stable maximum sub compared)
    return (source, maximum, sub)

private def site (code : Code) (loopEnd pc : Nat) (exp changed : String) : Option Site := do
  let [(["mul.f32", output, _, shift], done)] := code.products.getD exp [] | none
  let (["ex2.approx.ftz.f32", _, exponent], expPC) ← code.defn exp done | none
  guard (expPC == pc && exponent == changed)
  let (["shl.b32", _, scale, "23"], scalePC) ← code.defn shift done | none
  let (["fma.rn.ftz.f32", _, input, adjust, residual], fixed) ← code.defn changed pc | none
  code.constant adjust fixed "0f32A57060"
  let (["fma.rn.ftz.f32", _, before, log2e, compensation], residualPC) ← code.defn residual fixed | none
  code.constant log2e residualPC "0f3FB8AA3B"
  let (["neg.f32", _, bias], negated) ← code.defn compensation residualPC | none
  let (["add.f32", _, biasedScale, offset], biasPC) ← code.defn bias negated | none
  code.constant offset biasPC "0fCB40007F"
  guard (scale == biasedScale && code.stable scale scalePC biasPC)
  let (["fma.rm.ftz.f32", _, clamped, width, bias], scalePC) ← code.defn scale scalePC | none
  code.constant width scalePC "0f437C0000"
  code.constant bias scalePC "0f4B400001"
  let (["cvt.ftz.sat.f32.f32", _, rough], clampedPC) ← code.defn clamped scalePC | none
  let (["fma.rn.ftz.f32", _, original, slope, center], roughPC) ← code.defn rough clampedPC | none
  code.constant slope roughPC "0f3BBB989D"
  code.constant center roughPC "0f3F000000"
  guard (input == before && input == original && code.stable input roughPC fixed)
  let phase := if pc < loopEnd then Phase.online else .rescale
  guard ((pc < loopEnd) == (done < loopEnd))
  let (prior, maximum, inputPC) ← inputRegisters code phase input roughPC
  return ⟨phase, prior, maximum, inputPC, output, done⟩

private def onlineScale (code : Code) (site : Site) : Option (String × (Site × String)) := do
  let [(["selp.f32", scaled, one, _, pred], pc)] := code.selects.getD site.output [] | none
  code.constant one pc "0f3F800000"
  let (_, expPC) ← code.defn site.output pc
  let (["setp.eq.f32", _, maximum, bottom], tested) ← code.defn pred pc | none
  code.constant bottom tested "0fFF800000"
  guard (expPC == site.outputPC && maximum == site.maximum)
  return (scaled, (site, pred))

private def checkMaximum (code : Code) (prior incoming : Site) (use : Nat) : Option Unit := do
  guard (prior.maximum == incoming.maximum)
  let (["selp.f32", _, lhs, ordered, unorderedPred], m) ← code.defn prior.maximum use | none
  let (["setp.nan.f32", _, checkedA, checkedB], nanPC) ← code.defn unorderedPred m | none
  let (["selp.f32", _, a, b, greater], selected) ← code.defn ordered m | none
  let (["setp.gt.f32", _, comparedA, comparedB], tested) ← code.defn greater selected | none
  guard (lhs == a && lhs == checkedA && lhs == checkedB && lhs == comparedA &&
    b == comparedB && lhs == prior.source && b == incoming.source)
  guard (code.stable lhs (min nanPC tested) use && code.stable b tested use)

private def carryWrites (code : Code) (old : String) (header latch : Nat) (bottom : String) :
    Option Instruction := do
  let [(write, next), (["mov.b32", _, initial], init)] := code.definitions.getD old [] | none
  guard (init < header && header < next && next < latch && dominates code.edges init header &&
    dominates code.edges next latch)
  if initial != bottom then
    let [(["mov.b32", _, literal], source)] :=
      (code.definitions.getD initial []).filter (fun (_, pc) => pc < init) | none
    guard (literal == bottom && dominates code.edges source init)
  return (write, next)

private def checkCarry (code : Code) (header latch : Nat) (update : OnlineUpdate) :
    Option (Option (String × Nat)) := do
  let oldMax := update.previous.source
  let max := update.previous.maximum
  let (sumWrite, sumPC) ← carryWrites code update.oldSum header latch "0f00000000"
  let (maxWrite, maxPC) ← carryWrites code oldMax header latch "0fFF800000"
  let _ ← code.defn max maxPC
  match sumWrite, maxWrite with
  | "fma.rn.f32" :: _, ["mov.b32", _, source] =>
    guard (sumPC == update.newSumPC && source == max)
    return none
  | ["selp.f32", _, computedSum, previousSum, sumPred], ["selp.f32", _, computedMax, previousMax, maxPred] =>
    let (_, fmaPC) ← code.defn computedSum sumPC
    let ([op, _, column, "50304"], tested) ← code.defn sumPred sumPC | none
    let (_, maxTest) ← code.defn maxPred maxPC
    guard (computedSum == update.newSum && fmaPC == update.newSumPC &&
      previousSum == update.oldSum && computedMax == max && previousMax == oldMax &&
      sumPred == maxPred && tested == maxTest && (op == "setp.lt.u64" || op == "setp.lt.u32"))
    return some (column, tested)
  | _, _ => none

private def decodeUpdates (code : Code) (lines : List String) (loopStart loopEnd : Nat)
    (online : List Site) : Option (List OnlineUpdate) := do
  let scales ← online.mapM (onlineScale code)
  guard ((scales.map Prod.fst).eraseDups.length == scales.length)
  let mut updates := []
  for (line, pc) in (lines.take loopEnd).zipIdx do
    if let ["fma.rn.f32", sum, prior, renorm, contribution] := operands line then
      if let some (incomingSite, incomingGuard) := scales.lookup contribution then
        let (priorSite, priorGuard) ← scales.lookup renorm
        guard (priorGuard == incomingGuard && priorSite.output != incomingSite.output)
        let _ ← code.defn renorm pc
        let _ ← code.defn contribution pc
        checkMaximum code priorSite incomingSite pc
        let update : OnlineUpdate := ⟨priorSite, incomingSite, prior, sum, pc, none, none⟩
        let active ← checkCarry code loopStart loopEnd update
        updates := {update with activeColumn := active.map Prod.fst, activeColumnPC := active.map Prod.snd} :: updates
  let used := updates.flatMap fun update => [update.previous.output, update.incoming.output]
  guard (online.length == used.length && used.eraseDups.length == used.length &&
    online.all (fun site => used.contains site.output))
  return updates.reverse

private def decodeCounter (code : Code) (header latch : Nat) (branch : String) : Option LoopCounter := do
  let pred := (branch.drop 1).toString
  let (["setp.lt.u64", _, counter, endColumn], compared) ← code.defn pred latch | none
  let [(advance, next), (["mov.b64", _, initial], init)] := code.definitions.getD counter [] | none
  guard (init < header && header < next && header < compared && compared < latch && next < latch &&
    dominates code.edges init header && dominates code.edges next latch)
  let (extent, prebiased) ← match advance with
    | ["add.s64", _, previous, step] => do
      let extent ← step.toNat?
      guard (previous == counter && initial.toInt? == some (- (Int.ofNat extent)) &&
        next < compared && dominates code.edges next compared)
      pure (extent, true)
    | ["mov.b64", _, updated] => do
      let (["add.s64", _, previous, step], incremented) ← code.defn updated next | none
      guard (previous == counter && initial == "0" && header < incremented &&
        incremented < compared && compared < next)
      pure (← step.toNat?, false)
    | _ => none
  let limit ← endColumn.toNat?
  return ⟨counter, extent, limit, prebiased⟩

def carriedMaximum (update : OnlineUpdate) : String :=
  if update.activeColumn.isSome then update.previous.source else update.previous.maximum

private def decodeScaled (code : Code) (lines : List String) (latch : Nat)
    (sites : List Site) (updates : List OnlineUpdate) : Option (List ScaledSlot) := do
  let byScale := sites.map (fun site => (site.output, site))
  let mut scaled := []
  for (line, pc) in (lines.drop (latch + 1)).zipIdx (latch + 1) do
    let term := match operands line with
      | ["mul.f32", dst, oldSum, scale] => some (dst, oldSum, scale, none)
      | ["fma.rn.f32", dst, oldSum, scale, addend] => some (dst, oldSum, scale, some addend)
      | _ => none
    if let some (result, oldSum, word, accumulator) := term then
      if let some scale := byScale.lookup word then
        let [update] := updates.filter (fun update => carriedMaximum update == scale.source) | none
        let (_, scalePC) ← code.defn word pc
        guard (scalePC == scale.outputPC && oldSum == update.oldSum &&
          code.stable oldSum latch pc && code.stable scale.source latch scalePC)
        scaled := ⟨update, scale, accumulator, result, pc⟩ :: scaled
  let used := scaled.map (fun slot => slot.scale.output)
  let sourceUpdates := scaled.map (fun slot => slot.update.oldSum)
  guard (used.length == sites.length && used.eraseDups.length == used.length &&
    sourceUpdates.eraseDups.length == updates.length)
  return scaled.reverse

def decode (ptx : String) : Option Decoded := do
  let lines := QKVTTIR.normalizePTX ptx
  let edges ← ValidationLogBinding.validationFlow lines
  let core := buildIndex lines edges
  let [("$L__BB0_1:", loopStart)] := lines.zipIdx.filter (fun (line, _) => line == "$L__BB0_1:") | none
  let [(branch, loopEnd)] := lines.zipIdx.filterMap (fun (line, pc) =>
    match operands line with
    | [pred, "bra", "$L__BB0_1"] => if pred.startsWith "@%" then some (pred, pc) else none
    | _ => none) | none
  let backedges := (edges.toList.zipIdx).flatMap fun (destinations, source) =>
    (destinations.filter (· ≤ source)).map fun destination => (source, destination)
  guard (backedges == [(loopEnd, loopStart)])
  let counter ← decodeCounter core loopStart loopEnd branch
  let exponents := lines.zipIdx.filterMap fun (line, pc) =>
    match operands line with
    | ["ex2.approx.ftz.f32", exp, input] => some (pc, exp, input)
    | _ => none
  guard (loopStart < loopEnd && exponents.all (fun (pc, _, _) => loopStart < pc))
  let sites ← exponents.mapM fun (pc, out, input) => site core loopEnd pc out input
  let updates ← decodeUpdates core lines loopStart loopEnd (sites.filter (·.phase == .online))
  let scaled ← decodeScaled core lines loopEnd (sites.filter (·.phase == .rescale)) updates
  return ⟨loopStart, loopEnd, counter, sites, updates, scaled⟩

structure Certificate (ptx : String) (tile : ValidationCE.Tile) where
  reduction : Decoded
  decoded : decode ptx = some reduction
  extent : reduction.counter.extent = tile.extent
  limit : reduction.counter.limit = 50304 - tile.extent
  updateCount : reduction.updates.length = ValidationCE.slabCount * tile.groups
  scaleCount : (reduction.sites.filter (·.phase == .rescale)).length = ValidationCE.slabCount * tile.groups
  roots : (reduction.scaled.filter (·.accumulator.isNone)).length = ValidationCE.slabCount

def certify (ptx : String) (tile : ValidationCE.Tile) : Option (Certificate ptx tile) :=
  match decoded : decode ptx with
  | none => none
  | some reduction =>
    if extent : reduction.counter.extent = tile.extent then
      if limit : reduction.counter.limit = 50304 - tile.extent then
        if updates : reduction.updates.length = ValidationCE.slabCount * tile.groups then
          if scales : (reduction.sites.filter (·.phase == .rescale)).length = ValidationCE.slabCount * tile.groups then
            if roots : (reduction.scaled.filter (·.accumulator.isNone)).length = ValidationCE.slabCount then
              some ⟨reduction, decoded, extent, limit, updates, scales, roots⟩
            else none
          else none
        else none
      else none
    else none

def LoopCounter.entry (counter : LoopCounter) (iteration : Nat) : Int :=
  Int.ofNat (iteration * counter.extent) - if counter.prebiased then Int.ofNat counter.extent else 0

def LoopCounter.test (counter : LoopCounter) (iteration : Nat) : Int :=
  counter.entry iteration + if counter.prebiased then Int.ofNat counter.extent else 0

theorem loop_entry_next (counter : LoopCounter) (iteration : Nat) :
    counter.entry (iteration + 1) = counter.entry iteration + counter.extent := by
  have castStep :
      Int.ofNat (iteration * counter.extent + counter.extent) =
        Int.ofNat (iteration * counter.extent) + counter.extent := rfl
  simp only [LoopCounter.entry, Nat.add_mul, Nat.one_mul, castStep]
  omega

theorem loop_counter_exact {ptx : String} {tile : ValidationCE.Tile} (artifact : Certificate ptx tile)
    (iteration : Nat) :
    artifact.reduction.counter.test iteration =
      Int.ofNat (ValidationCE.scanColumn tile 0 iteration) := by
  rw [LoopCounter.test, LoopCounter.entry, artifact.extent, ValidationCE.scanColumn, Nat.add_zero, Nat.mul_comm]
  omega

theorem loop_trip_exact {ptx : String} {tile : ValidationCE.Tile} (artifact : Certificate ptx tile)
    (iteration : Nat) :
    artifact.reduction.counter.test iteration < (artifact.reduction.counter.limit : Int) ↔
      iteration + 1 < TailNorm.ceilDiv 50304 tile.extent := by
  rw [loop_counter_exact artifact, artifact.limit]
  cases tile <;> simp [ValidationCE.scanColumn, ValidationCE.Tile.extent, TailNorm.ceilDiv] <;> omega

private def ieee := Config.ieeeValue
private def ftz : Config := ⟨.nearestEven, true, true⟩
private def rmFtz : Config := ⟨.towardNegative, true, true⟩
private def w := Value.ofBits .fp32

private def expPayload (native : ValidationCE.Native) (x : F32) : F32 :=
  let rough := FP.fma ftz x (w 0x3bbb989d) (w 0x3f000000)
  let prepared := FP.prepare ftz rough
  let sat :=
    if prepared.sign || prepared.isNaN || prepared.isZero then zero
    else if FP.lt ieee one prepared then one else prepared
  let scale := FP.fma rmFtz sat (w 0x437c0000) (w 0x4b400001)
  let bias := (FP.add ieee scale (w 0xcb40007f)).neg
  let residual := FP.fma ftz x (w 0x3fb8aa3b) bias
  let correction := FP.fma ftz x (w 0x32a57060) residual
  FP.mul ieee (native.ex2Approx true correction) (w ((UInt32.ofNat scale.raw <<< 23).toNat))

def Site.eval (site : Site) (native : ValidationCE.Native) (source maximum : F32) : F32 :=
  expPayload native (if site.phase == .rescale && maximum == Value.infinity .fp32 true then
    zero else FP.sub ieee source maximum)

theorem eval_online (site : Site) (phase : site.phase = .online) (native : ValidationCE.Native)
    (source maximum : F32) :
    site.eval native source maximum = ValidationCE.expf native (FP.sub ieee source maximum) := by
  simp only [Site.eval, phase]
  rfl

theorem eval_rescale (site : Site) (phase : site.phase = .rescale) (native : ValidationCE.Native)
    (state : ValidationCE.Partial) (maximum : F32) :
    site.eval native state.maximum maximum = ValidationCE.rescale native maximum state := by
  simp only [Site.eval, phase]
  rfl

def onlinePayload (native : ValidationCE.Native) (state : ValidationCE.Partial)
    (incoming : F32) : ValidationCE.Partial :=
  let byOrder := if FP.lt ieee incoming state.maximum then state.maximum else incoming
  let maximum := if state.maximum.isNaN then state.maximum else byOrder
  let scale := fun source =>
    if maximum == Value.infinity .fp32 true then one
    else expPayload native (FP.sub ieee source maximum)
  ⟨maximum, FP.fma ieee state.sum (scale state.maximum) (scale incoming)⟩

theorem online_payload_exact (native : ValidationCE.Native) (state : ValidationCE.Partial)
    (incoming : F32) : onlinePayload native state incoming = ValidationCE.accumulate native state incoming := by
  have sameMaximum :
      (if state.maximum.isNaN then state.maximum
        else if FP.lt ieee incoming state.maximum then state.maximum else incoming) =
        ValidationCE.max state.maximum incoming := by
    change _ = (if FP.lt ieee incoming state.maximum || state.maximum.isNaN then state.maximum else incoming)
    by_cases nan : state.maximum.isNaN <;>
      by_cases gt : FP.lt ieee incoming state.maximum <;> simp_all
  unfold onlinePayload ValidationCE.accumulate
  dsimp only
  rw [sameMaximum]
  rfl

/-- Read the inputs named by the checked FMA and maximum path of a
particular update and evaluate their contribution to its loop state. -/
def OnlineUpdate.eval (update : OnlineUpdate) (native : ValidationCE.Native) (regs : String → F32) :
    ValidationCE.Partial :=
  onlinePayload native ⟨regs update.previous.source, regs update.oldSum⟩ (regs update.incoming.source)

def OnlineUpdate.commit (update : OnlineUpdate) (native : ValidationCE.Native)
    (regs : String → F32) (readColumn : String → Nat) : ValidationCE.Partial :=
  if update.activeColumn.all (fun col => readColumn col < 50304) then update.eval native regs
  else ⟨regs update.previous.source, regs update.oldSum⟩

theorem decoded_online_update_exact {ptx : String} {tile : ValidationCE.Tile}
    (artifact : Certificate ptx tile) (native : ValidationCE.Native) (regs : String → F32)
    (i : Fin artifact.reduction.updates.length) :
    (((decode ptx).bind fun result => result.updates[i.val]?).map fun update => update.eval native regs) =
      some (ValidationCE.accumulate native
        ⟨regs artifact.reduction.updates[i].previous.source, regs artifact.reduction.updates[i].oldSum⟩
        (regs artifact.reduction.updates[i].incoming.source)) := by
  rw [artifact.decoded]
  simp only [Option.bind_some, List.getElem?_eq_getElem i.isLt, Option.map_some,
    OnlineUpdate.eval, online_payload_exact]
  rfl

theorem decoded_loop_commit_exact {ptx : String} {tile : ValidationCE.Tile}
    (artifact : Certificate ptx tile) (native : ValidationCE.Native)
    (regs : String → F32) (readColumn : String → Nat) (i : Fin artifact.reduction.updates.length) :
    let update := artifact.reduction.updates[i]
    let state : ValidationCE.Partial := ⟨regs update.previous.source, regs update.oldSum⟩
    (((decode ptx).bind fun result => result.updates[i.val]?).map
      fun update => update.commit native regs readColumn) =
      some (if update.activeColumn.all (fun col => readColumn col < 50304) then
        ValidationCE.accumulate native state (regs update.incoming.source) else state) := by
  rw [artifact.decoded]
  simp only [Option.bind_some, List.getElem?_eq_getElem i.isLt, Option.map_some,
    OnlineUpdate.commit, OnlineUpdate.eval, online_payload_exact]
  rfl

def ScaledSlot.eval (slot : ScaledSlot) (native : ValidationCE.Native) (regs : String → F32) : F32 :=
  let scale := slot.scale.eval native (regs slot.scale.source) (regs slot.scale.maximum)
  let sum := regs slot.update.oldSum
  match slot.accumulator with
  | none => FP.mul ieee sum scale
  | some acc => FP.fma ieee sum scale (regs acc)

theorem scaled_slot_exact (slot : ScaledSlot) (phase : slot.scale.phase = .rescale)
    (native : ValidationCE.Native) (regs : String → F32) :
    let state : ValidationCE.Partial := ⟨regs slot.scale.source, regs slot.update.oldSum⟩
    let scale := ValidationCE.rescale native (regs slot.scale.maximum) state
    slot.eval native regs = match slot.accumulator with
      | none => FP.mul ieee state.sum scale
      | some acc => FP.fma ieee state.sum scale (regs acc) := by
  let state : ValidationCE.Partial := ⟨regs slot.scale.source, regs slot.update.oldSum⟩
  change slot.eval native regs = _
  unfold ScaledSlot.eval
  rw [eval_rescale slot.scale phase native state]

#axiomsClean eval_online
#axiomsClean eval_rescale
#axiomsClean online_payload_exact
#axiomsClean loop_trip_exact
#axiomsClean decoded_online_update_exact
#axiomsClean decoded_loop_commit_exact
#axiomsClean scaled_slot_exact
end VeriTile.Bench.Models.NanoGPT.Track1.ValidationExpBinding
