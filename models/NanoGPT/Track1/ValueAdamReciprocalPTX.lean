import models.NanoGPT.Track1.ValueAdamNativeQuotients
import models.NanoGPT.Track1.QKVTTIRBinding

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.ReciprocalPTX

open NativeColdTrace
private structure Slot where
  output : String
  first : String
  second : String
  deriving Inhabited

private def opener : String := ".reg .b32 %stateWords;"
private def regularLabel : String := "L__adam_direction_regular:"
private def doneLabel : String := "L__adam_direction_done:"
private def epsilon : String := "0f2edbe6ff"

private def register (word : String) : Option Nat := do
  guard (word.startsWith "%r")
  (word.drop 2).toString.toNat?

private def readSlot (lines : List String) : Option Slot := do
  let [root, add, divide] := lines | none
  guard (add == s!"add.rn.f32 %denominator, %root, {epsilon};")
  let ["sqrt.rn.f32", "%root,", second] := root.splitOn " " | none
  let ["div.full.f32", output, first, "%denominator;"] := divide.splitOn " " | none
  guard (second.endsWith ";" && output.endsWith "," && first.endsWith ",")
  let (output, first, second) :=
    ((output.dropEnd 1).toString, (first.dropEnd 1).toString, (second.dropEnd 1).toString)
  let _ ← register output
  let _ ← register first
  let _ ← register second
  return ⟨output, first, second⟩

private def bitwiseUnion (word : Nat → String) : List String :=
  [s!"lop3.b32 %stateWords, {word 0}, {word 1}, {word 2}, 0xfe;",
   s!"lop3.b32 %stateWords, %stateWords, {word 3}, {word 4}, 0xfe;",
   s!"lop3.b32 %stateWords, %stateWords, {word 5}, {word 6}, 0xfe;",
   s!"or.b32 %stateWords, %stateWords, {word 7};"]

private def start (second : Nat → String) : List String :=
  [opener,
   ".reg .pred %laneHasMoment, %warpHasMoment, %nanFirst, %warpNanFirst;",
   ".reg .f32 %root, %denominator, %reciprocal;"] ++ bitwiseUnion second ++
  ["and.b32 %stateWords, %stateWords, 0x7fffffff;",
   "setp.ne.b32 %laneHasMoment, %stateWords, 0;",
   "vote.sync.any.pred %warpHasMoment, %laneHasMoment, 0xffffffff;",
   "@!%warpHasMoment bra.uni L__adam_direction_zero;",
   regularLabel]

private def nanTest (kind : GuardKind) (first : Nat → String) : List String :=
  match kind with
  | .nan =>
    s!"setp.nan.f32 %nanFirst, {first 0}, {first 0};" ::
      ((List.range 7).map fun i => s!"setp.nan.or.f32 %nanFirst, {first (i+1)}, {first (i+1)}, %nanFirst;")
  | .exponent => bitwiseUnion first ++
    ["and.b32 %stateWords, %stateWords, 0x7f800000;",
     "setp.eq.b32 %nanFirst, %stateWords, 0x7f800000;"]

private def finish (kind : GuardKind) (slots : Array Slot) : List String :=
  ["bra.uni L__adam_direction_done;", "L__adam_direction_zero:",
    s!"add.rn.f32 %denominator, {slots[0]!.second}, {epsilon};"] ++
  nanTest kind (fun i => slots[i]!.first) ++
  ["vote.sync.any.pred %warpNanFirst, %nanFirst, 0xffffffff;",
   "@%warpNanFirst bra.uni L__adam_direction_regular;",
   "rcp.approx.ftz.f32 %reciprocal, %denominator;"] ++
  (slots.toList.map fun slot => s!"mul.rn.f32 {slot.output}, %reciprocal, {slot.first};") ++
  [doneLabel, "}"]

private def decodeFrame (kind : GuardKind) (lines : List String) :
    Option (List String × List Slot × List String) := do
  let before := lines.takeWhile (· != opener)
  let "{" :: reversed := before.reverse | none
  let rest := lines.drop before.length
  let pre := rest.takeWhile (· != regularLabel)
  let regular := (rest.drop (pre.length + 1)).take 24
  let slots ← (List.range 8).mapM fun i => readSlot ((regular.drop (3*i)).take 3)
  let registers := slots.flatMap fun slot => [slot.output, slot.first, slot.second]
  guard (registers.eraseDups.length == 24)
  let slotArray := slots.toArray
  let header := start (fun i => slotArray[i]!.second)
  let footer := finish kind slotArray
  let expected := header ++ regular ++ footer
  guard (rest.take expected.length == expected)
  return (reversed.reverse, slots, rest.drop expected.length)

private def regDeclaration : String := ".reg .b32 %r<"

private def registerLimit (lines : List String) : Option Nat := do
  let [line] := lines.filter (·.startsWith regDeclaration) | none
  guard (line.endsWith ">;")
  ((line.drop regDeclaration.length).dropEnd 2).toString.toNat?

private def numberedReg (word : String) : Option (Nat × String) := do
  guard (word.startsWith "r")
  let tail := word.drop 1
  let number := (tail.takeWhile Char.isDigit).toString
  return (← number.toNat?, (tail.drop number.length).toString)

private def alphaNormalize (lines : List String) (limit : Nat) : Option (List String) := do
  let mut names : Std.HashMap Nat Nat := {}
  let mut result := []
  for line in lines do
    if line.startsWith regDeclaration then
      result := ".reg .b32 %r<active>;" :: result
    else
      let head :: parts := line.splitOn "%" | none
      let mut normalized := head
      for part in parts do
        match numberedReg part with
        | none => normalized := normalized ++ "%" ++ part
        | some (index, suffix) =>
          guard (0 < index && index < limit)
          let ordinal := (names[index]?).getD (names.size + 1)
          names := names.insert index ordinal
          normalized := normalized ++ s!"%r{ordinal}" ++ suffix
      result := normalized :: result
  return result.reverse

private def replaceFrame (slots : List Slot) (limit : Nat) : List String :=
  let roots := slots.zipIdx.map fun (slot, i) => s!"sqrt.rn.f32 %r{limit+i}, {slot.second};"
  let divisors := (List.range 8).map fun i => s!"add.rn.f32 %r{limit+8+i}, %r{limit+i}, 0f2EDBE6FF;"
  let quotients := slots.zipIdx.map fun (slot, i) =>
    s!"div.full.f32 {slot.output}, {slot.first}, %r{limit+8+i};"
  roots ++ divisors ++ quotients

def compareRewriteWithRegisters (baseline candidate : String)
    (rewrite : List String → Nat → Option (List String × Nat)) : Option (List String) := do
  let before := QKVTTIR.normalizePTX baseline
  let after := QKVTTIR.normalizePTX candidate
  let referenceLimit ← registerLimit before
  let candidateLimit ← registerLimit after
  let _ ← alphaNormalize after candidateLimit
  let normalized ← alphaNormalize before referenceLimit
  let (rewritten, limit) ← rewrite after candidateLimit
  let normalizedCandidate ← alphaNormalize rewritten limit
  guard (normalized == normalizedCandidate)
  return normalized

def compareRewritten (baseline candidate : String)
    (rewrite : List String → Option (List String)) : Option (List String) :=
  compareRewriteWithRegisters baseline candidate (fun lines limit => do
    return (← rewrite lines, limit))

/-- The reciprocal route has an exact checked guard and a separate native
certificate. The original path and every instruction outside the frame
must be the original emitted PTX up to allocation of virtual b32 registers. -/
def compare (baseline candidate : String) (kind : GuardKind) : Option (List String) := do
  let before := QKVTTIR.normalizePTX baseline
  let after := QKVTTIR.normalizePTX candidate
  let referenceLimit ← registerLimit before
  let candidateLimit ← registerLimit after
  let _ ← alphaNormalize after candidateLimit
  let normalized ← alphaNormalize before referenceLimit
  let (leading, slots, trailing) ← decodeFrame kind after
  let replaced ← alphaNormalize (leading ++ replaceFrame slots candidateLimit ++ trailing)
    (candidateLimit + 16)
  guard (normalized == replaced)
  return normalized

structure Certificate (basePTX candidatePTX baseSASS candidateSASS : String)
    (kind : GuardKind) (period : Nat) : Type where
  private mk ::
  private frame : (compare basePTX candidatePTX kind).isSome = true
  native : NativeColdTrace.Certificate baseSASS candidateSASS kind period

def certify (basePTX candidatePTX baseSASS candidateSASS : String)
    (kind : GuardKind) (period : Nat) :
    Option (Certificate basePTX candidatePTX baseSASS candidateSASS kind period) := do
  if frame : (compare basePTX candidatePTX kind).isSome = true then
    let native ← NativeColdTrace.certify baseSASS candidateSASS kind period
    return ⟨frame, native⟩
  else none

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.ReciprocalPTX
