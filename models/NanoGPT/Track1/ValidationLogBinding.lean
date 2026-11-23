import models.NanoGPT.Track1.TrainingCEPublishBinding
import models.NanoGPT.Track1.ValidationTargetBinding

/-! Recover the libdevice logarithms in the emitted validation kernel.
Each slab supplies its own reduced FP32 sum. The decoder follows the
range reduction, polynomial, nonfinite branch and final zero select. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValidationLogBinding
open TrainingCEGatherBinding
open VeriTile.Triton FP
open TailNorm (BF16 F32 zero)

inductive Compare where
  | floatLT | wordLT | zero
  deriving DecidableEq

inductive Expr where
  | sum
  | literal (bits : Nat)
  | add (a b : Expr)
  | mul (a b : Expr)
  | fma (a b c : Expr)
  | addBits (a b : Expr)
  | subBits (a b : Expr)
  | andBits (a b : Expr)
  | ofSigned (a : Expr)
  | select (test : Compare) (a b yes no : Expr)
  deriving DecidableEq

private def ieee := Config.ieeeValue
private def ftz : Config := ⟨.nearestEven, true, true⟩
private def f32 := Value.ofBits .fp32
private def bits32 (x : F32) := UInt32.ofNat x.raw

private def Compare.eval (op : Compare) (x y : F32) : Bool :=
  match op with
  | .floatLT => FP.lt ieee x y
  | .wordLT => decide (x.raw < y.raw)
  | .zero => x.isZero

def Expr.eval (sum : F32) : Expr → F32
  | .sum => sum
  | .literal x => f32 x
  | .add a b => FP.add ieee (a.eval sum) (b.eval sum)
  | .mul a b => FP.mul ieee (a.eval sum) (b.eval sum)
  | .fma a b c => FP.fma ftz (a.eval sum) (b.eval sum) (c.eval sum)
  | .addBits a b => f32 (bits32 (a.eval sum) + bits32 (b.eval sum)).toNat
  | .subBits a b => f32 (bits32 (a.eval sum) - bits32 (b.eval sum)).toNat
  | .andBits a b => f32 (bits32 (a.eval sum) &&& bits32 (b.eval sum)).toNat
  | .ofSigned a => FP.round .fp32 ieee (Rat.ofInt (bits32 (a.eval sum)).toInt32.toInt)
  | .select test a b yes no =>
    if test.eval (a.eval sum) (b.eval sum) then yes.eval sum else no.eval sum

private inductive Reg where
  | word (expr : Expr)
  | predicate (test : Compare) (a b : Expr)

private abbrev Registers := List (String × Reg)

private def word (regs : Registers) (name : String) : Option Expr :=
  if let some (.word expr) := regs.lookup name then some expr
  else if let some (.literal bits) := TrainingCEPublishBinding.getExpr [] name then some (.literal bits)
  else name.toInt?.map (fun number => .literal ((number % 4294967296).toNat))

private def transfer (regs : Registers) : List String → Option Reg
  | ["mov.b32", _, src] => (word regs src).map .word
  | ["cvt.rn.f32.s32", _, src] => (word regs src).map fun a => .word (.ofSigned a)
  | ["fma.rn.ftz.f32", _, x, y, z] => do
    return .word (.fma (← word regs x) (← word regs y) (← word regs z))
  | ["selp.f32", _, yes, no, pred] => do
    let .predicate test a b ← regs.lookup pred | none
    return .word (.select test a b (← word regs yes) (← word regs no))
  | [op, _, x, y] => do
    let a ← word regs x
    let b ← word regs y
    match op with
    | "setp.lt.f32" => some (.predicate .floatLT a b)
    | "setp.lt.u32" => some (.predicate .wordLT a b)
    | "add.f32" => some (.word (.add a b))
    | "mul.f32" => some (.word (.mul a b))
    | "add.s32" => some (.word (.addBits a b))
    | "sub.s32" => some (.word (.subBits a b))
    | "and.b32" => some (.word (.andBits a b))
    | _ => none
  | _ => none

private def execute (regs : Registers) (insns : List (List String)) : Option Registers :=
  insns.foldlM (fun regs insn => do
    let _ :: dst :: _ := insn | none
    guard (dst.startsWith "%")
    let value ← transfer regs insn
    return (dst, value) :: regs.filter (·.1 != dst)) regs

private structure Geometry where
  rowShift : Nat
  threads : Nat
  normalReturn : Nat

private def decodeGeometry (lines : List String) : Option Geometry := do
  let [directive] := lines.filter (·.startsWith ".reqntid ") | none
  let [".reqntid", block] := operands directive | none
  let threads ← block.toNat?
  let [row] := lines.filter (defines "%rd1") | none
  let shift ← match operands row with
    | ["cvt.u64.u32", _, _] => some 0
    | ["cvt.s64.s32", _, r] => do
      let [merge] := lines.filter (defines r) | none
      let ["or.b32", _, base, _] := operands merge | none
      let [scaled] := lines.filter (defines base) | none
      let ["shl.b32", _, _, shift] := operands scaled | none
      shift.toNat?
    | _ => none
  guard ([(0, 256), (3, 512), (4, 128), (6, 256), (6, 512)].contains (shift, threads))
  let [("ret;", normalReturn)] := lines.zipIdx.filter
    (fun (line, _) => line == "ret;" || (operands line).contains "exit" ||
      (operands line).contains "ret") | none
  return ⟨shift, threads, normalReturn⟩

private def checkRow (lines : List String) (edges : Array (List Nat))
    (geometry : Geometry) (row : String) (use : Nat) : Option String := do
  let (conversion, pc) ← uniqueDefinition lines edges row use
  if geometry.rowShift == 0 then
    let ["cvt.u64.u32", _, column] := conversion | none
    let (["mov.u32", _, "%ctaid.x"], _) ← uniqueDefinition lines edges column pc | none
    return column
  else
    let ["cvt.s64.s32", _, column] := conversion | none
    let (["or.b32", _, base, laneRow], merged) ← uniqueDefinition lines edges column pc | none
    let (["shl.b32", _, cta, shift], scaled) ← uniqueDefinition lines edges base merged | none
    let (["mov.u32", _, "%ctaid.x"], _) ← uniqueDefinition lines edges cta scaled | none
    let (["and.b32", _, tid, mask], masked) ← uniqueDefinition lines edges laneRow merged | none
    let (["mov.u32", _, "%tid.x"], _) ← uniqueDefinition lines edges tid masked | none
    guard (shift.toNat? == some geometry.rowShift && mask.toNat? == some (2 ^ geometry.rowShift - 1))
    return column

private def laneMask (lines : List String) (edges : Array (List Nat)) (reg : String)
    (use : Nat) : Option Nat := do
  let (instruction, pc) ← uniqueDefinition lines edges reg use
  let (maskOp, maskPC) ← match instruction with
    | ["cvt.u32.u64", _, wide] => do
      let (["cvt.u64.u32", _, narrow], extended) ← uniqueDefinition lines edges wide pc | none
      uniqueDefinition lines edges narrow extended
    | _ => some (instruction, pc)
  let ["and.b32", _, tid, mask] := maskOp | none
  let (["mov.u32", _, "%tid.x"], _) ← uniqueDefinition lines edges tid maskPC | none
  mask.toNat?

private def ownerPredicate (lines : List String) (edges : Array (List Nat))
    (reg : String) (use : Nat) : Nat → Option (Nat → Bool)
  | 0 => none
  | fuel + 1 => do
    let (instruction, pc) ← uniqueDefinition lines edges reg use
    match instruction with
    | ["setp.eq.b32", _, src, "0"] => do
      let mask ← laneMask lines edges src pc
      return fun tid => tid &&& mask == 0
    | ["setp.lt.u32", _, src, limit] => do
      let mask ← laneMask lines edges src pc
      let limit ← limit.toNat?
      return fun tid => tid &&& mask < limit
    | ["and.pred", _, a, b] => do
      let left ← ownerPredicate lines edges a pc fuel
      let right ← ownerPredicate lines edges b pc fuel
      return fun tid => left tid && right tid
    | _ => none

private def checkTarget (lines : List String) (edges : Array (List Nat)) (entry : String)
    (slab : Nat) (row row32 target : String) (use : Nat) : Option String := do
  let (pointer, pc) ← ValidationTargetBinding.Code.load ⟨lines, edges⟩ target use "mov.u64" "b64"
  let (["add.s64", _, base, delta], addr) ← uniqueDefinition lines edges pointer pc | none
  let (["ld.param.b64", _, input], _) ← uniqueDefinition lines edges base addr | none
  guard (input == s!"[{entry}_param_10]")
  let (["shl.b64", _, index, "3"], scaled) ← uniqueDefinition lines edges delta addr | none
  if slab == 0 then
    guard (index == row)
    let _ ← uniqueDefinition lines edges row scaled
  else
    let (["cvt.s64.s32", _, globalRow], extended) ← uniqueDefinition lines edges index scaled | none
    let (["add.s32", _, localRow, offset], added) ← uniqueDefinition lines edges globalRow extended | none
    guard (localRow == row32 && offset.toNat? == some (slab * ValidationCE.slabRows))
    let _ ← uniqueDefinition lines edges row32 added
  return pointer

private def checkFinish (lines : List String) (edges : Array (List Nat)) (entry : String) (geometry : Geometry)
    (slab selected : Nat) (logReg : String) :
    Option (ValidationTargetBinding.MaskMode × String × Nat) := do
  let code := lines.toArray
  let (use, pc) ← lines.zipIdx.find? fun (line, pc) =>
    selected < pc && match operands line with
    | ["sub.f32", _, logInput, _] => logInput == logReg
    | _ => false
  let ["sub.f32", difference, _, delta] := operands use | none
  let (subtraction, deltaPC) ← uniqueDefinition lines edges delta pc
  let ["sub.f32", _, target, maximum] := subtraction | none
  let ["add.f32", result, loss, "0f00000000"] := operands (← code[pc + 1]?) | none
  guard (deltaPC < pc && loss == difference && target.startsWith "%" && maximum.startsWith "%")
  let (_, sourcePC) ← uniqueDefinition lines edges logReg pc
  guard (sourcePC == selected && dominates edges pc (pc + 1))
  let (select, selectPC) ← lines.zipIdx.find? fun (line, index) =>
    pc + 1 < index && match operands line with
    | ["selp.b32", _, "0", live, _] => live == result
    | _ => false
  let ["selp.b32", payload, _, _, ignore] := operands select | none
  let (_, resultPC) ← uniqueDefinition lines edges result selectPC
  guard (resultPC == pc + 1)
  let (test, before) ← uniqueDefinition lines edges ignore selectPC
  let ["setp.eq.b64", _, targetIndex, "-100"] := test | none
  guard (before < selectPC && targetIndex.startsWith "%rd")
  let (store, storePC) ← lines.zipIdx.find? fun (line, index) =>
    selectPC < index && match operands line with
    | [active, "st.global.b32", "[", _, "+", "0", "]", data] =>
      active.startsWith "@%" && data == payload
    | _ => false
  let [active, "st.global.b32", "[", pointer, "+", "0", "]", _] := operands store | none
  guard (storePC < geometry.normalReturn && dominates edges storePC geometry.normalReturn)
  let owner ← ownerPredicate lines edges (active.drop 1).toString storePC 2
  guard ((List.range geometry.threads).all fun tid => owner tid == decide (tid < 2 ^ geometry.rowShift))
  let (_, valuePC) ← uniqueDefinition lines edges payload storePC
  guard (valuePC == selectPC)
  let (address, addressPC) ← uniqueDefinition lines edges pointer storePC
  let ["add.s64", _, base, delta] := address | none
  let (parameter, paramPC) ← uniqueDefinition lines edges base addressPC
  let ["ld.param.b64", _, input] := parameter | none
  let (offset, offsetPC) ← uniqueDefinition lines edges delta addressPC
  let ["shl.b64", _, row, "2"] := offset | none
  guard (paramPC < addressPC && offsetPC < addressPC && addressPC < storePC)
  guard (row == "%rd1" && input == s!"[{entry}_param_{11 + slab}]")
  let row32 ← checkRow lines edges geometry row offsetPC
  let targetPointer ← checkTarget lines edges entry slab row row32 targetIndex before
  let mask ← ValidationTargetBinding.decode ⟨lines, edges⟩ entry slab row row32 targetPointer target deltaPC
  let (_, maximumDef) ← uniqueDefinition lines edges maximum deltaPC
  guard (maximumDef < deltaPC)
  return (mask, maximum, maximumDef)

structure SlabFinish where
  log : Expr
  mask : ValidationTargetBinding.MaskMode
  sum : String
  sumDef : Nat
  maximum : String
  maximumDef : Nat
  owners : Nat

private def fragment (lines : List String) (edges : Array (List Nat)) (entry : String) (geometry : Geometry)
    (start slab : Nat) : Option SlabFinish := do
  let code := lines.toArray
  let ["setp.lt.f32", _, input, "0f00800000"] := operands (← code[start]?) | none
  let body := (lines.drop start).takeWhile (!·.startsWith "@")
  let regs ← execute [(input, .word .sum)] (body.map operands)
  let jump := start + body.length
  let [test, "bra", label] := operands (← code[jump]?) | none
  let .predicate .wordLT normalized top ← regs.lookup (test.drop 1).toString | none
  guard (test.startsWith "@")
  let ["mov.b32", infinity, "0f7F800000"] := operands (← code[jump + 1]?) | none
  let ["fma.rn.ftz.f32", out, norm, product, addend] := operands (← code[jump + 2]?) | none
  guard (infinity == product && infinity == addend && out != infinity && norm != infinity)
  guard ((← word regs norm) = normalized)
  let ordinary ← word regs out
  let exceptional ← execute regs [(operands (← code[jump + 1]?)), operands (← code[jump + 2]?)]
  let joined ← word exceptional out
  let join := jump + 3
  guard (code[join]? == some (label ++ ":") && dominates edges start join)
  let (check, pc) ← lines.zipIdx.find? fun (line, pc) =>
    join < pc && match operands line with
    | ["setp.eq.f32", _, source, "0f00000000"] => source == norm
    | _ => false
  let ["setp.eq.f32", predicate, _, _] := operands check | none
  let ["selp.f32", dst, "0fFF800000", value, condition] := operands (← code[pc + 1]?) | none
  guard (value == out && condition == predicate && dst.startsWith "%")
  guard (dominates edges join pc && dominates edges pc (pc + 1))
  let between := (lines.take pc).drop (join + 1)
  guard (!(between.any fun line => defines norm line || defines out line))
  let (mask, maximum, maximumDef) ← checkFinish lines edges entry geometry slab (pc + 1) dst
  let (_, sumDef) ← uniqueDefinition lines edges input start
  guard (sumDef < start)
  let result := Expr.select .zero normalized (.literal 0) (.literal 0xff800000)
    (.select .wordLT normalized top ordinary joined)
  return ⟨result, mask, input, sumDef, maximum, maximumDef, 2 ^ geometry.rowShift⟩

def validationFlow (lines : List String) : Option (Array (List Nat)) := do
  let text := lines.toArray
  let exits ← lines.zipIdx.mapM fun (line, pc) => do
    if line.startsWith "call.uni __assertfail, " then
      -- Triton ends the exception path with an explicit trap even if
      -- the assertion routine returns.
      guard (text[pc + 1]? == some "}" && text[pc + 2]? == some "trap;")
      return "trap;"
    return line
  flow exits

def decodeSlabs (ptx : String) : Option (List SlabFinish) := do
  let lines := QKVTTIR.normalizePTX ptx
  let [header] := lines.filter (·.startsWith ".visible .entry ") | none
  let [".visible", ".entry", name] := header.splitOn " " | none
  guard (name.endsWith "(")
  let entry := (name.dropEnd 1).toString
  let edges ← validationFlow lines
  let geometry ← decodeGeometry lines
  guard (!dominates edges lines.length geometry.normalReturn)
  guard ((lines.filter fun line => (operands line).any (·.startsWith "st.global.")).length ==
    ValidationCE.slabCount)
  let starts := lines.zipIdx.filterMap fun (line, pc) =>
    match operands line with
    | ["setp.lt.f32", _, _, "0f00800000"] => some pc
    | _ => none
  guard (starts.length == ValidationCE.slabCount)
  starts.zipIdx.mapM fun (start, slab) => fragment lines edges entry geometry start slab

def decode (ptx : String) : Option (List Expr) :=
  (decodeSlabs ptx).map (List.map SlabFinish.log)

private def normal : Expr :=
  .select .floatLT .sum (.literal 0x00800000) (.mul .sum (.literal 0x4b000000)) .sum

private def adjustment : Expr :=
  .select .floatLT .sum (.literal 0x00800000) (.literal 0xc1b80000) (.literal 0)

private def exponent : Expr :=
  .andBits (.addBits normal (.literal 0xc0d55555)) (.literal 0xff800000)

private def power : Expr := .fma (.ofSigned exponent) (.literal 0x34000000) adjustment

private def mantissa : Expr := .add (.subBits normal exponent) (.literal 0xbf800000)

private def coefficients : List Nat :=
  [0x3e1039f6, 0xbdf8cdcc, 0x3e0f2955, 0xbe2ad8b9, 0x3e4ced0b, 0xbe7fff22, 0x3eaaaa78, 0xbf000000]

private def polynomial : Expr :=
  coefficients.foldl (fun acc coeff => Expr.fma acc mantissa (.literal coeff)) (.literal 0xbe055027)

def expected : Expr :=
  let correction := Expr.fma (.mul mantissa polynomial) mantissa mantissa
  let normalLog := Expr.fma power (.literal 0x3f317218) correction
  let specialLog := Expr.fma normal (.literal 0x7f800000) (.literal 0x7f800000)
  .select .zero normal (.literal 0) (.literal 0xff800000)
    (.select .wordLT normal (.literal 0x7f800000) normalLog specialLog)

structure Certificate (ptx : String) : Type where
  parts : List SlabFinish
  decoded : decodeSlabs ptx = some parts
  instructions : decode ptx = some (List.replicate ValidationCE.slabCount expected)

def certify (ptx : String) : Option (Certificate ptx) := do
  match decoded : decodeSlabs ptx with
  | none => none
  | some parts =>
    if matchArtifact : parts.map SlabFinish.log = List.replicate ValidationCE.slabCount expected then
      some ⟨parts, decoded, by simp only [decode, decoded, Option.map_some, matchArtifact]⟩
    else none

private theorem bits_roundtrip (word : UInt32) : bits32 (f32 word.toNat) = word := by
  apply UInt32.toNat_inj.mp
  have bound := word.toNat_lt_size
  change (word.toNat % 4294967296) % 4294967296 = word.toNat
  rw [Nat.mod_eq_of_lt bound, Nat.mod_eq_of_lt bound]

private theorem eval_normal (x : F32) :
    normal.eval x = if FP.lt ieee x (f32 0x00800000) then FP.mul ieee x (f32 0x4b000000) else x := rfl

private theorem eval_adjustment (x : F32) :
    adjustment.eval x = if FP.lt ieee x (f32 0x00800000) then f32 0xc1b80000 else zero := rfl

private theorem eval_exponent (x : F32) :
    bits32 (exponent.eval x) = (bits32 (normal.eval x) + 0xc0d55555) &&& 0xff800000 := by
  simp only [exponent, Expr.eval, bits_roundtrip]
  rfl

private theorem eval_power (x : F32) :
    power.eval x = FP.fma ftz
      (FP.round .fp32 ieee (Rat.ofInt (((bits32 (normal.eval x) + 0xc0d55555) &&& 0xff800000) : UInt32).toInt32.toInt))
      (f32 0x34000000) (adjustment.eval x) := by
  simp only [power, Expr.eval, eval_exponent]

private theorem eval_mantissa (x : F32) :
    mantissa.eval x = FP.add ieee (f32
      (bits32 (normal.eval x) - ((bits32 (normal.eval x) + 0xc0d55555) &&& 0xff800000)).toNat)
      (f32 0xbf800000) := by
  simp only [mantissa, Expr.eval, eval_exponent]

private theorem eval_polynomial (x : F32) :
    polynomial.eval x =
      coefficients.foldl (fun acc coeff => FP.fma ftz acc (mantissa.eval x) (f32 coeff)) (f32 0xbe055027) := by
  have fold (items : List Nat) (seed : Expr) :
      (items.foldl (fun acc coeff => Expr.fma acc mantissa (.literal coeff)) seed).eval x =
        items.foldl (fun acc coeff => FP.fma ftz acc (mantissa.eval x) (f32 coeff)) (seed.eval x) := by
    induction items generalizing seed with
    | nil => rfl
    | cons coeff rest ih => simp only [List.foldl_cons, ih, Expr.eval]
  exact fold _ _

set_option maxRecDepth 4096 in
theorem logf_eval (x : F32) : expected.eval x = ValidationCE.logf x := by
  simp only [expected, Expr.eval, Compare.eval, decide_eq_true_eq, eval_polynomial,
    eval_power, eval_mantissa, eval_normal, eval_adjustment,
    show (f32 0x7f800000).raw = 0x7f800000 from rfl]
  rfl

theorem decoded_logf_exact {ptx : String} (artifact : Certificate ptx) (sum : F32) :
    (decode ptx).map (List.map (Expr.eval sum)) =
      some (List.replicate ValidationCE.slabCount (ValidationCE.logf sum)) := by
  rw [artifact.instructions]
  simp [logf_eval]

/-- The live-row finish uses the decoded logarithm and performs the
target-minus-maximum subtraction before the final subtraction and FP32
addition of +0. The statement applies independently to each slab. -/
theorem decoded_finish_exact {ptx : String} (artifact : Certificate ptx)
    (slab : Fin ValidationCE.slabCount) (target maximum sum : F32) :
    (((decode ptx).bind fun expressions => expressions[slab.val]?).map fun log =>
      FP.add ieee (FP.sub ieee (log.eval sum) (FP.sub ieee target maximum)) zero) =
      some (ValidationCE.finish target maximum sum) := by
  rw [artifact.instructions]
  simp only [Option.bind_some, List.getElem?_replicate, if_pos slab.isLt, Option.map_some, logf_eval]
  rfl

/-- The word forwarded to a slab's FP32 store is zero for the ignored
target word and the exact live-row loss for every other target word. -/
theorem decoded_store_payload_exact {ptx : String} (artifact : Certificate ptx)
    (slab : Fin ValidationCE.slabCount) (targetIndex : UInt64) (target maximum sum : F32) :
    let ignored := targetIndex == 0xffffffffffffff9c
    (((decode ptx).bind fun expressions => expressions[slab.val]?).map fun log =>
      if ignored then zero else FP.add ieee (FP.sub ieee (log.eval sum) (FP.sub ieee target maximum)) zero) =
      some (if ignored then zero else ValidationCE.finish target maximum sum) := by
  rw [artifact.instructions]
  simp only [Option.bind_some, List.getElem?_replicate, if_pos slab.isLt, Option.map_some, logf_eval]
  rfl

/-- The emitted row merge uses the low lane bits. The store predicate
retains exactly one such lane per local row in the decoded CTA. -/
theorem row_word_exact (shift block thread : Nat) (owner : thread < 2 ^ shift) :
    (block <<< shift) ||| (thread &&& (2 ^ shift - 1)) = block * 2 ^ shift + thread := by
  rw [Nat.and_two_pow_sub_one_of_lt_two_pow owner, ← Nat.shiftLeft_add_eq_or_of_lt owner,
    Nat.shiftLeft_eq]

private theorem localRowBound {shift block thread : Nat}
    (validBlock : block < ValidationCE.slabRows / 2 ^ shift) (owner : thread < 2 ^ shift) :
    block * 2 ^ shift + thread < ValidationCE.slabRows := by
  have next := Nat.mul_le_mul_right (2 ^ shift) (Nat.succ_le_of_lt validBlock)
  have tile := Nat.div_mul_le_self ValidationCE.slabRows (2 ^ shift)
  rw [Nat.succ_mul] at next
  omega

/-- The ignored-target select reads the 64-bit word at the slab's
global token index. Its FP32 output uses the slab-local row index. -/
theorem decoded_store_for_row_exact {ptx : String} (artifact : Certificate ptx)
    (slab : Fin ValidationCE.slabCount) (shift block thread : Nat)
    (validBlock : block < ValidationCE.slabRows / 2 ^ shift) (owner : thread < 2 ^ shift)
    (readTarget : Nat → UInt64) (target maximum sum : F32) :
    let row : Fin ValidationCE.slabRows := ⟨block * 2 ^ shift + thread, localRowBound validBlock owner⟩
    let bits := (block <<< shift) ||| (thread &&& (2 ^ shift - 1))
    let targetOffset := (bits + slab.val * ValidationCE.slabRows) <<< 3
    let ignored := readTarget (targetOffset / 8) == 0xffffffffffffff9c
    ((((decode ptx).bind fun expressions => expressions[slab.val]?).map fun log =>
      (bits <<< 2, if ignored then zero else FP.add ieee (FP.sub ieee (log.eval sum)
        (FP.sub ieee target maximum)) zero))) =
      some (row.val * 4, if readTarget (ValidationCE.slabRow slab row) == 0xffffffffffffff9c
        then zero else ValidationCE.finish target maximum sum) := by
  dsimp only
  rw [row_word_exact shift block thread owner, artifact.instructions]
  simp only [Option.bind_some, List.getElem?_replicate, if_pos slab.isLt, Option.map_some, logf_eval,
    Nat.shiftLeft_eq, Nat.reducePow, Nat.mul_div_cancel _ (by decide : 0 < 8), ValidationCE.slabRow,
    Nat.add_comm (block * 2 ^ shift + thread)]
  rfl

private theorem decoded_slab_log {ptx : String} (artifact : Certificate ptx)
    (slab : Fin ValidationCE.slabCount) :
    ∃ finish, ((decodeSlabs ptx).bind fun slabs => slabs[slab.val]?) = some finish ∧ finish.log = expected := by
  have result : ((decode ptx).bind fun logs => logs[slab.val]?) = some expected := by
    rw [artifact.instructions]
    simp only [Option.bind_some, List.getElem?_replicate, if_pos slab.isLt]
  cases parts : decodeSlabs ptx with
  | none => simp [decode, parts] at result
  | some slabs =>
    simp only [decode, parts, Option.map_some, Option.bind_some, List.getElem?_map] at result
    cases atSlab : slabs[slab.val]? with
    | none => simp [atSlab] at result
    | some finish =>
      exact ⟨finish, by simpa using atSlab, by simpa [atSlab] using result⟩

def finishRow (ptx : String) (slab : Fin ValidationCE.slabCount) (native : ValidationCE.Native)
    (target : Option (Fin 50304)) (mask : Nat → UInt8) (projection : Fin 50304 → BF16)
    (maximum sum : F32) : Option F32 :=
  let column := target.getD ⟨0, by decide⟩
  ((decodeSlabs ptx).bind fun parts => parts[slab.val]?).map fun part =>
    let (byte, bit) := part.mask.indices column
    let z := ValidationTargetBinding.payload native
      (ValidationTargetBinding.testMaskByte (mask byte) bit) (projection column)
    if target.isNone then zero
    else FP.add ieee (FP.sub ieee (part.log.eval sum) (FP.sub ieee z maximum)) zero

/-- A certified slab computes the same target logit and final loss as the
row specification once its online reduction has produced the sum and maximum. -/
theorem decoded_store_gather_exact {ptx : String} (artifact : Certificate ptx)
    (slab : Fin ValidationCE.slabCount) (native : ValidationCE.Native) (target : Option (Fin 50304))
    (mask : Nat → UInt8) (projection : Fin 50304 → BF16) (maximum sum : F32) :
    finishRow ptx slab native target mask projection maximum sum =
    some (match target with
      | none => zero
      | some column => ValidationCE.finish
          (ValidationCE.logit native mask column.val (projection column)) maximum sum) := by
  obtain ⟨part, decoded, log⟩ := decoded_slab_log artifact slab
  unfold finishRow
  rw [decoded, Option.map_some, log, logf_eval]
  cases target with
  | none => rfl
  | some column =>
    simp only [Option.getD_some, Option.isNone_some, Bool.false_eq_true, ↓reduceIte]
    rw [ValidationTargetBinding.mask_indices_exact, ValidationTargetBinding.testMaskByte_exact]
    rfl

#axiomsClean decoded_logf_exact
#axiomsClean decoded_finish_exact
#axiomsClean decoded_store_payload_exact
#axiomsClean decoded_store_for_row_exact
#axiomsClean decoded_store_gather_exact
end VeriTile.Bench.Models.NanoGPT.Track1.ValidationLogBinding
