import models.NanoGPT.Track1.ValueAdamReciprocalPTX

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.DivisorBand

private structure Slot where
  output : String
  numerator : String
  denominator : String
  deriving Inhabited

private def opener : String := ".reg .b32 %mOr, %denAnd, %denOr, %shifted;"
private def fallback : String := "L__adam_divide_fallback:"
private def join : String := "L__adam_divide_join:"

private def register (name : String) : Bool :=
  name.startsWith "%r" && ((name.drop 2).toString.toNat?).isSome

private def slot (line : String) : Option Slot := do
  let ["div.full.f32", output, numerator, denominator] := line.splitOn " " | none
  guard (output.endsWith "," && numerator.endsWith "," && denominator.endsWith ";")
  let output := (output.dropEnd 1).toString
  let numerator := (numerator.dropEnd 1).toString
  let denominator := (denominator.dropEnd 1).toString
  guard (register output && register numerator && register denominator)
  return ⟨output, numerator, denominator⟩

private def aggregate (dst operation table : String) (word : Nat → String) (width : Nat) : List String :=
  s!"{operation}.b32 {dst}, {word 0}, {word 1};" ::
    (List.range (width / 2 - 1)).map fun pair =>
      let i := 2 * pair + 2
      s!"lop3.b32 {dst}, {dst}, {word i}, {word (i+1)}, {table};"

private def exponentSummary (slots : Array Slot) : List String :=
  aggregate "%mOr" "or" "0xfe" (fun i => slots[i]!.numerator) slots.size ++
  ["and.b32 %mOr, %mOr, 0x7f800000;"]

private def control (slots : Array Slot) (reuse : Bool := false) : List String :=
  let den := fun i => slots[i]!.denominator
  [opener, ".reg .f32 %inverse;", ".reg .pred %invalid, %slow;"] ++
  aggregate "%denAnd" "and" "0x80" den slots.size ++ aggregate "%denOr" "or" "0xfe" den slots.size ++
  ["and.b32 %denAnd, %denAnd, 0x20000000;",
   "and.b32 %denOr, %denOr, 0xc0000000;",
   "xor.b32 %denAnd, %denAnd, 0x20000000;",
   "or.b32 %shifted, %denAnd, %denOr;",
   "setp.ne.u32 %invalid, %shifted, 0;"] ++
  (if reuse then [] else exponentSummary slots) ++
  ["setp.eq.or.u32 %invalid, %mOr, 0x7f800000, %invalid;",
   "vote.sync.any.pred %slow, %invalid, 0xffffffff;",
   "@%slow bra.uni L__adam_divide_fallback;"]

private def fast (port : Slot) : List String :=
  [s!"rcp.approx.ftz.f32 %inverse, {port.denominator};",
   s!"mul.rn.f32 {port.output}, %inverse, {port.numerator};"]

private def rewrite (lines : List String) : Option (List String) := do
  let before := lines.takeWhile (· != opener)
  let "{" :: reversed := before.reverse | none
  let body := lines.drop before.length
  let leading := body.takeWhile (· != fallback)
  let afterFallback := body.drop (leading.length + 1)
  let original := afterFallback.take 8
  let slots ← original.mapM slot
  guard (slots.length == 8)
  let registers := slots.flatMap fun port => [port.output, port.numerator, port.denominator]
  guard (registers.eraseDups.length == 24)
  let "L__adam_divide_join:" :: "}" :: epilogue := afterFallback.drop 8 | none
  let expected := control slots.toArray ++ slots.flatMap fast ++
    ["bra.uni L__adam_divide_join;", fallback] ++ original ++ [join, "}"]
  guard (body.take expected.length == expected)
  return reversed.reverse ++ original ++ epilogue

/-- Compare the actual whole entry after extracting the exact guard and
retaining the original dynamic divides. The rest of the PTX is preserved
through virtual register renumbering. -/
def comparePTX (baseline candidate : String) : Option (List String) :=
  ReciprocalPTX.compareRewritten baseline candidate rewrite

inductive ZeroMode where
  | dynamic | reciprocal | guardedReciprocal | rootFirst | directAfterRoot
  deriving Repr, BEq

def ZeroMode.rootGuardFirst (mode : ZeroMode) : Bool :=
  mode == .rootFirst || mode == .directAfterRoot

def ZeroMode.guardsRoot (mode : ZeroMode) : Bool :=
  mode == .guardedReciprocal || mode.rootGuardFirst

private def openerWithZero : String := ".reg .b32 %mOr, %denAnd, %denOr, %shifted, %moments;"

private def zeroControl (width : Nat) (second : Nat → String) : List String :=
  [openerWithZero, s!".reg .f32 %inverse, %root, %den<{width}>;",
    ".reg .pred %invalid, %slow, %hasMoment, %warpHasMoment;"] ++
  aggregate "%moments" "or" "0xfe" second width ++
  ["and.b32 %moments, %moments, 0x7fffffff;",
   "setp.ne.u32 %hasMoment, %moments, 0;",
   "vote.sync.any.pred %warpHasMoment, %hasMoment, 0xffffffff;",
   "@!%warpHasMoment bra.uni L__adam_zero_direction;"]

private def originalDivide (port : Slot) (denominator : String) : String :=
  s!"div.full.f32 {port.output}, {port.numerator}, {denominator};"

private def coldBlock (mode : ZeroMode) (slots : Array Slot) (secondZero : String)
    (reuse : Bool) : List String :=
  ["bra.uni L__adam_divide_join;", "L__adam_zero_direction:",
    s!"add.rn.f32 %den0, {secondZero}, 0f2edbe6ff;"] ++
  (if mode == .dynamic then [] else
    (if reuse then [] else exponentSummary slots) ++
    ["setp.eq.u32 %invalid, %mOr, 0x7f800000;",
     "vote.sync.any.pred %slow, %invalid, 0xffffffff;",
     "@%slow bra.uni L__adam_zero_fallback;",
     "rcp.approx.ftz.f32 %inverse, %den0;"] ++
    (slots.toList.map fun port => s!"mul.rn.f32 {port.output}, %inverse, {port.numerator};") ++
    ["bra.uni L__adam_divide_join;", "L__adam_zero_fallback:"]) ++
  slots.toList.map (fun port => originalDivide port "%den0") ++ [join, "}"]

private def readSecond (text : String) : Option String := do
  let ["sqrt.rn.f32", "%root,", word] := text.splitOn " " | none
  guard (word.endsWith ";")
  let word := (word.dropEnd 1).toString
  guard (register word)
  return word

private def readWarmDivide (index : Nat) (text : String) : Option Slot := do
  let ["div.full.f32", out, num, den] := text.splitOn " " | none
  guard (out.endsWith "," && num.endsWith "," && den == s!"%den{index};")
  let output := (out.dropEnd 1).toString
  let numerator := (num.dropEnd 1).toString
  guard (register output && register numerator)
  return ⟨output, numerator, s!"%den{index}"⟩

private def openerWithRootGuard : String :=
  ".reg .b32 %mOr, %denAnd, %denOr, %shifted, %moments, %rootAnd, %rootOr;"

private def sharedControl (mode : ZeroMode) (width : Nat) (second : Nat → String) : List String :=
  if mode.guardsRoot then
    [openerWithRootGuard,
     s!".reg .f32 %inverse, %root, %den<{width}>, %rsq, %rough, %negativeRough, %halfRsq, %residual;",
     ".reg .pred %invalid, %slow, %hasMoment, %warpHasMoment, %specialRoot, %warpSpecialRoot;"] ++
       (if mode.rootGuardFirst then [] else (zeroControl width second).drop 3)
  else zeroControl width second

private def guardedRoots (mode : ZeroMode) (slots : Array Slot) (second : Nat → String) : List String :=
  aggregate "%rootAnd" "and" "0x80" second slots.size ++ aggregate "%rootOr" "or" "0xfe" second slots.size ++
  ["and.b32 %rootAnd, %rootAnd, 0x20000000;",
   "and.b32 %rootOr, %rootOr, 0xc0000000;",
   "xor.b32 %rootAnd, %rootAnd, 0x20000000;",
   "or.b32 %shifted, %rootAnd, %rootOr;",
   "setp.ne.u32 %specialRoot, %shifted, 0;"] ++
  (if mode == .directAfterRoot then
    exponentSummary slots ++
    ["setp.eq.or.u32 %specialRoot, %mOr, 0x7f800000, %specialRoot;"]
  else []) ++
  ["vote.sync.any.pred %warpSpecialRoot, %specialRoot, 0xffffffff;",
   "@%warpSpecialRoot bra.uni L__adam_root_original;"] ++
  (List.range slots.size).flatMap (fun i =>
    [s!"rsqrt.approx.ftz.f32 %rsq, {second i};",
     s!"mul.rn.ftz.f32 %rough, {second i}, %rsq;",
     "mul.rn.ftz.f32 %halfRsq, %rsq, 0f3f000000;",
     "neg.f32 %negativeRough, %rough;",
     s!"fma.rn.f32 %residual, %negativeRough, %rough, {second i};",
     "fma.rn.f32 %root, %residual, %halfRsq, %rough;",
     s!"add.rn.f32 %den{i}, %root, 0f2edbe6ff;"]) ++
  (if mode == .directAfterRoot then slots.toList.flatMap fast ++ ["bra.uni L__adam_divide_join;"]
    else ["bra.uni L__adam_roots_done;"]) ++ ["L__adam_root_original:"]

private def rewriteCombined (mode : ZeroMode) (reuse : Bool) (width : Nat) (lines : List String) (limit : Nat) :
    Option (List String × Nat) := do
  guard (width == 2 || width == 8)
  let opener := if mode.guardsRoot then openerWithRootGuard else openerWithZero
  let before := lines.takeWhile (· != opener)
  let "{" :: reversed := before.reverse | none
  let body := lines.drop before.length
  let pre := body.takeWhile (fun row => !row.startsWith "sqrt.rn.f32 %root, ")
  let roots := (body.drop pre.length).take (2 * width)
  let second ← (List.range width).mapM (fun i => readSecond (roots[2*i]?.getD ""))
  let throughWarm := body.takeWhile (· != fallback)
  let original := (body.drop (throughWarm.length + 1)).take width
  let slots ← (List.range width).mapM (fun i => readWarmDivide i (original[i]?.getD ""))
  let registers := second ++ slots.flatMap fun port => [port.output, port.numerator]
  guard (registers.eraseDups.length == 3 * width)
  let second := second.toArray
  let slots := slots.toArray
  let expectedRoots := (List.range width).flatMap fun i =>
    [s!"sqrt.rn.f32 %root, {second[i]!};", s!"add.rn.f32 %den{i}, %root, 0f2edbe6ff;"]
  let warmed := if mode.guardsRoot then
      guardedRoots mode slots (fun i => second[i]!) ++
        (if mode.rootGuardFirst then (zeroControl width (fun i => second[i]!)).drop 3 else []) ++
        expectedRoots ++ ["L__adam_roots_done:"]
    else expectedRoots
  let expected := sharedControl mode width (fun i => second[i]!) ++ warmed ++ ((control slots reuse).drop 3) ++
    (slots.toList.flatMap fast) ++ ["bra.uni L__adam_divide_join;", fallback] ++ original ++
    coldBlock mode slots second[0]! reuse
  guard (body.take expected.length == expected)
  let canonicalRoots := (List.range width).map fun i =>
    s!"sqrt.rn.f32 %r{limit+i}, {second[i]!};"
  let canonicalDenominators := (List.range width).map fun i =>
    s!"add.rn.f32 %r{limit+width+i}, %r{limit+i}, 0f2EDBE6FF;"
  let canonicalDivides := (List.range width).map fun i =>
    originalDivide slots[i]! s!"%r{limit+width+i}"
  return (reversed.reverse ++ canonicalRoots ++ canonicalDenominators ++ canonicalDivides ++
    body.drop expected.length, limit + 2 * width)

def compareCombinedPTX (baseline candidate : String) (mode : ZeroMode) (width : Nat := 8) : Option (List String) :=
  (ReciprocalPTX.compareRewriteWithRegisters baseline candidate (rewriteCombined mode false width)).orElse fun _ =>
    if mode == .directAfterRoot then
      ReciprocalPTX.compareRewriteWithRegisters baseline candidate (rewriteCombined mode true width)
    else none

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.DivisorBand
