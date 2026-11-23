import models.NanoGPT.Track1.ValueAdamElementwise
import Std.Data.HashMap

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.NativeCombinedTrace

open VeriTile.Triton FP

private structure Instruction where
  address : Nat
  text : String
  deriving Inhabited

private def trim (word : String) : String := word.trimAscii.toString

private def hexDigit (ch : Char) : Option Nat :=
  if '0' ≤ ch && ch ≤ '9' then some (ch.toNat - '0'.toNat)
  else if 'a' ≤ ch && ch ≤ 'f' then some (ch.toNat - 'a'.toNat + 10)
  else none

private def hex (word : String) : Option Nat := do
  guard (!word.isEmpty)
  word.toList.foldlM (fun accumulated ch => do
    return (16 * accumulated) + (← hexDigit ch)) 0

private def offset (word : String) : Option Nat := do
  guard (word.startsWith "0x")
  hex (word.drop 2).toString

private def parseLine (line : String) : Option Instruction := do
  let address :: body :: _ := (trim line).splitOn "*/" | none
  guard (address.startsWith "/*")
  let address ← hex (address.drop 2).toString
  let code :: _ :: _ := body.splitOn ";" | none
  return ⟨address, trim code⟩

private def parse (source : String) : Array Instruction :=
  ((source.splitOn "\n").filterMap parseLine).toArray

private inductive WordPort where
  | first | second | denominator | gradient
  deriving Inhabited, BEq, Repr

private structure Coverage where
  port : WordPort
  slots : Nat
  deriving Inhabited, BEq, Repr

private inductive GuardSource where
  | divideHigh (slot : Nat)
  | divideNormal (slot : Nat)
  | badDivisorBand
  | badRootBand
  | badRootBandOrNaN
  | badBandOrNaN
  | secondNonzero
  | firstExponentClear
  | gradientNonzero
  | gradientNaN
  deriving Inhabited, BEq

private inductive Operand where
  | unknown
  | literal (bits : Nat)
  | threadIndex
  | ctaIndex
  | stridedLane
  | maskedLane
  | tileIndex
  | descriptor
  | momentArg (port : WordPort)
  | momentAddress (port : WordPort)
  | momentResultArg (port : WordPort)
  | momentResultAddress (port : WordPort)
  | oldWeightArg
  | newWeightArg
  | gradientArg
  | gradientAddress
  | gradientNaNCoverage (pairs : Nat)
  | gradientNaNFlags
  | rawGradient (slot : Nat)
  | scaledRawGradient (slot : Nat)
  | scaledGradientPair (pair : Nat)
  | gradient (slot : Nat)
  | squaredGradient (slot : Nat)
  | gradientFirstMix (slot : Nat)
  | gradientSecondMix (slot : Nat)
  | weightInputAddress
  | weightOutputAddress
  | weightPair (low : Nat)
  | highWeightBits (slot : Nat)
  | weight (slot : Nat)
  | decayScalar
  | stepApplied (slot : Nat)
  | cautiousProduct (slot : Nat)
  | cautiousGate (slot : Nat)
  | maskedWeight (slot : Nat)
  | delta (slot : Nat)
  | negativeDelta (slot : Nat)
  | updatedWeight (slot : Nat)
  | updatedPair (low high : Nat)
  | loaded (port : WordPort) (slot : Nat)
  | first (slot : Nat)
  | second (slot : Nat)
  | shiftedRoot (slot : Nat)
  | rootReciprocal (slot : Nat)
  | roughRoot (slot : Nat)
  | negativeRoughRoot (slot : Nat)
  | halfRootReciprocal (slot : Nat)
  | rootResidual (slot : Nat)
  | root (slot : Nat)
  | denominator (slot : Nat)
  | absoluteDenominator (slot : Nat)
  | quarterNumerator (slot : Nat)
  | chosenNumerator (slot divisorSlot : Nat)
  | chosenDenominator (slot : Nat)
  | normalizedNumerator (slot divisorSlot : Nat)
  | normalizedDenominator (slot : Nat)
  | summary (covered : Coverage)
  | allBandBits (covered : Coverage)
  | missingBandBit (port : WordPort)
  | highBandBits (port : WordPort)
  | exponentMask (slots : Nat)
  | inverse (slot : Nat) (scaled : Bool)
  | quotient (slot : Nat)
  | stepScalar
  deriving Inhabited, BEq, Repr

private def eps : Nat := 0x2edbe6ff

private def floatLiteral : String → Option Nat
  | "0.125" => some 0x3e000000
  | "0.4375" => some 0x3ee00000
  | "0.050000000745058059692" => some 0x3d4ccccd
  | "0.097499996423721313477" => some 0x3dc7ae14
  | "0.25" => some 0x3e800000
  | "0.5" => some 0x3f000000
  | "16777216" => some 0x4b800000
  | "0.75" => some 0x3f400000
  | "0.5625" => some 0x3f100000
  | "0.94999998807907104492" => some 0x3f733333
  | "0.90249997377395629883" => some 0x3f670a3d
  | "1.00000001335143196e-10" => some eps
  | "1.175494350822287508e-38" => some 0x00800000
  | "8.50705917302346158658e+37" => some 0x7e800000
  | _ => none

private def register (word : String) : Bool :=
  (word.startsWith "R" && ((word.drop 1).toString.toNat?).isSome) ||
    (word.startsWith "UR" && ((word.drop 2).toString.toNat?).isSome)

structure Quotient where
  slot : Nat
  inverseFirst : Bool
  scaled : Bool
  shared : Bool
  deriving BEq, Repr, Inhabited, DecidableEq

private structure State where
  width : Nat := 8
  registers : Std.HashMap String Operand := {}
  predicates : Std.HashMap String Bool := {}
  guards : Std.HashMap String GuardSource := {}
  divisions : List Quotient := []
  applied : List Nat := []
  storedFirst : List Nat := []
  storedSecond : List Nat := []
  storedWeight : List Nat := []
  gradientLoads : Nat := 0
  gradientClears : Nat := 0
  reciprocals : Nat := 0
  votes : Nat := 0
  gradientNaNFree : Bool := false
  roots : List Nat := []
  rootTests : List Nat := []
  rootHelpers : List Nat := []
  rootSwaps : List Nat := []

private def State.slotMask (state : State) : Nat := 2 ^ state.width - 1

private def read (state : State) (word : String) : Operand :=
  let name := (trim word).replace ".reuse" ""
  if name == "RZ" then .literal 0
  else if name.startsWith "-R" then
    match state.registers[(name.drop 1).toString]? with
    | some (.delta slot) => .negativeDelta slot
    | some (.roughRoot slot) => .negativeRoughRoot slot
    | _ => .unknown
  else if name.startsWith "|" && name.endsWith "|" then
    match state.registers[(name.drop 1 |>.dropEnd 1).toString]? with
    | some (.denominator slot) => .absoluteDenominator slot
    | _ => .unknown
  else
    match state.registers[name]? with
    | some operand => operand
    | none => ((offset name).orElse fun _ => floatLiteral name).map Operand.literal |>.getD .unknown

private def test (state : State) (name : String) : Option Bool :=
  if name == "PT" then some true
  else if name.startsWith "!" then state.predicates[(name.drop 1).toString]?.map not
  else state.predicates[name]?

private def write (state : State) (destination : String) (term : Operand) : State :=
  if register destination then
    { state with registers := state.registers.insert destination term }
  else state

private def writePredicate (state : State) (destination : String) (value : Option Bool) : State :=
  { state with
    predicates := match value with
      | some bit => state.predicates.insert destination bit
      | none => state.predicates.erase destination
    guards := state.guards.erase destination }

private def writeGuard (state : State) (destination : String) (rule : GuardSource) : State :=
  { writePredicate state destination none with guards := state.guards.insert destination rule }

private def coverage : Operand → Option Coverage
  | .loaded .gradient pair => if pair < 4 then some ⟨.gradient, 2 ^ pair⟩ else none
  | .first slot => if slot < 8 then some ⟨.first, 2 ^ slot⟩ else none
  | .second slot => if slot < 8 then some ⟨.second, 2 ^ slot⟩ else none
  | .denominator slot => if slot < 8 then some ⟨.denominator, 2 ^ slot⟩ else none
  | .summary covered => some covered
  | _ => none

private def merge (a b : Operand) : Option Coverage := do
  let a ← coverage a
  let b ← coverage b
  guard (a.port == b.port && (a.slots &&& b.slots) == 0)
  return ⟨a.port, a.slots ||| b.slots⟩

private def reduceThree (a b c : Operand) : Operand :=
  ((merge a b).bind fun ab => merge (.summary ab) c).map Operand.summary |>.getD .unknown

private def bandAnd : Operand → Option Coverage
  | .denominator slot => if slot < 8 then some ⟨.denominator, 2 ^ slot⟩ else none
  | .second slot => if slot < 8 then some ⟨.second, 2 ^ slot⟩ else none
  | .allBandBits covered => some covered
  | _ => none

private def reduceAnd (a b : Operand) : Option Operand := do
  let a ← bandAnd a
  let b ← bandAnd b
  guard (a.port == b.port && (a.slots &&& b.slots) == 0)
  return .allBandBits ⟨a.port, a.slots ||| b.slots⟩

private def addressParts (word : String) : Option (String × String × Nat) := do
  let [descriptor, pointer] := word.splitOn "][" | none
  guard (descriptor.startsWith "desc[")
  let descriptor := (descriptor.drop 5).toString
  let [base, displacement] := pointer.splitOn ".64" | none
  if displacement == "]" then return (descriptor, base, 0)
  guard (displacement.startsWith "+" && displacement.endsWith "]")
  let bytes ← offset (displacement.drop 1 |>.dropEnd 1).toString
  guard (bytes % 4 == 0)
  return (descriptor, base, bytes / 4)

private def loadWide (state : State) (words : Nat) (dst arg : String) : Option State := do
  let start ← (dst.drop 1).toString.toNat?
  let (descriptor, base, first) ← addressParts arg
  guard (read state descriptor == .descriptor)
  let pointer := read state base
  let loaded : Option (State × (Nat → Operand)) := match pointer with
    | .momentAddress port =>
      if words == min 4 state.width && first % words == 0 && first + words ≤ state.width then
        some (state, fun i => Operand.loaded port (first + i))
      else none
    | .weightInputAddress =>
      if first == 0 && 2 * words == state.width then some (state, fun i => Operand.weightPair (2*i))
      else none
    | .gradientAddress =>
      if first == 0 && 2 * words == state.width then
        some ({state with gradientLoads := state.gradientLoads + 1}, Operand.loaded .gradient)
      else none
    | _ => none
  let (state, word) ← loaded
  return (List.range words).foldl (fun state i =>
    write state s!"R{start + i}" (word i)) state

private def storeWide (state : State) (masked : Bool) (words : Nat) (predicate arg src : String) : Option State := do
  let (descriptor, base, first) ← addressParts arg
  guard (read state descriptor == .descriptor)
  let pointer := read state base
  if pointer == .gradientAddress then
    guard (first == 0 && 2 * words == state.width && src == "RZ" && state.gradientLoads == 1)
    guard (if masked then state.guards[predicate]? == some .gradientNonzero else predicate == "PT")
    return {state with gradientClears := state.gradientClears + 1}
  guard (test state predicate == some true)
  if pointer == .weightOutputAddress then
    let start ← (src.drop 1).toString.toNat?
    guard (first == 0 && 2 * words == state.width)
    let terms := (List.range words).map fun i => read state s!"R{start+i}"
    let expected := (List.range words).map fun i => Operand.updatedPair (2*i) (2*i+1)
    guard (terms == expected)
    return {state with storedWeight := List.range state.width ++ state.storedWeight}
  else
    let .momentResultAddress port := pointer | none
    let start ← (src.drop 1).toString.toNat?
    guard (words == min 4 state.width && first % words == 0 && first + words ≤ state.width)
    let slots := (List.range words).map (first + ·)
    let terms := (List.range words).map fun i => read state s!"R{start+i}"
    match port with
    | .first =>
      guard (terms == slots.map Operand.first)
      return {state with storedFirst := slots ++ state.storedFirst}
    | .second =>
      guard (terms == slots.map Operand.second)
      return {state with storedSecond := slots ++ state.storedSecond}
    | .denominator | .gradient => none

private def multiply (state : State) (dst : String) (args : List Operand) : State :=
  match args with
  | [.first i, .literal 0x3e800000] => write state dst (.quarterNumerator i)
  | [.first i, .inverse j false] =>
    if i == j || j == 8 then { write state dst (.quotient i) with divisions := ⟨i, false, false, j == 8⟩ :: state.divisions }
    else write state dst .unknown
  | [.inverse j false, .first i] =>
    if i == j || j == 8 then { write state dst (.quotient i) with divisions := ⟨i, true, false, j == 8⟩ :: state.divisions }
    else write state dst .unknown
  | [.inverse j true, .normalizedNumerator i denominator] =>
    if denominator == j then { write state dst (.quotient i) with divisions := ⟨i, true, true, j == 8⟩ :: state.divisions }
    else write state dst .unknown
  | [.normalizedNumerator i denominator, .inverse j true] =>
    if denominator == j then { write state dst (.quotient i) with divisions := ⟨i, false, true, j == 8⟩ :: state.divisions }
    else write state dst .unknown
  | [.quotient i, .stepScalar] =>
    { write state dst (.stepApplied i) with applied := i :: state.applied }
  | [.weight i, .stepApplied j] =>
    write state dst (if i == j then .cautiousProduct i else .unknown)
  | [.weight i, .cautiousGate j] =>
    write state dst (if i == j then .maskedWeight i else .unknown)
  | _ => write state dst .unknown

inductive Layout where
  | reference
  | dynamicZero
  | sharedReciprocal
  | guardedReciprocal
  | rootFirst
  | directAfterRoot
  deriving BEq, Repr

private def Layout.rootGuardFirst (layout : Layout) : Bool :=
  layout == .rootFirst || layout == .directAfterRoot

private def Layout.fastRootVoteCount : Layout → Nat
  | .rootFirst | .directAfterRoot => 2
  | _ => 3

structure Route where
  gradientNaN : Bool
  bandRejected : Bool
  rootSlowMask : Nat
  secondZero : Bool
  rootRejected : Bool

private def voteValue (layout : Layout) (route : Route) (state : State) (dst src : String) :
    Option Bool := do
  guard (dst == "P0")
  match state.votes with
  | 0 =>
    guard (src == dst && state.guards[dst]? == some .gradientNaN)
    return route.gradientNaN
  | 1 =>
    guard (layout != .reference && src == dst)
    if layout.rootGuardFirst then
      let kind := if layout == .directAfterRoot then GuardSource.badRootBandOrNaN
        else GuardSource.badRootBand
      guard (state.guards[dst]? == some kind)
      return route.rootRejected
    else
      guard (state.guards[dst]? == some .secondNonzero)
      return !route.secondZero
  | 2 =>
    if layout.rootGuardFirst && route.rootRejected then
      guard (src == dst && state.guards[dst]? == some .secondNonzero)
      return !route.secondZero
    else if route.secondZero then
      guard ((layout == .sharedReciprocal || layout == .guardedReciprocal) &&
        src == "!" ++ dst && state.guards[dst]? == some .firstExponentClear)
      return route.bandRejected
    else if layout == .guardedReciprocal then
      guard (src == dst && state.guards[dst]? == some .badRootBand)
      return route.rootRejected
    else
      guard (layout != .reference && src == dst && state.guards[dst]? == some .badBandOrNaN)
      return route.bandRejected
  | 3 =>
    if layout.rootGuardFirst && route.secondZero then
      guard (src == "!" ++ dst && state.guards[dst]? == some .firstExponentClear)
      return route.bandRejected
    else
      guard ((layout == .guardedReciprocal || layout.rootGuardFirst) && !route.secondZero &&
        src == dst && state.guards[dst]? == some .badBandOrNaN)
      return route.bandRejected
  | _ => none

private def applyPredicated (state : State) (predicate op dst : String) (args : List String) :
    Option State := do
  guard (op == "FMUL")
  let [src, _] := args | none
  guard (src == dst)
  match predicate.startsWith "!", state.guards[(if predicate.startsWith "!" then (predicate.drop 1).toString else predicate)]?,
      args.map (read state) with
  | false, some (.divideHigh i), [.denominator j, .literal 0x3e800000] =>
    guard (i == j)
    return write state dst (.chosenDenominator i)
  | true, some (.divideNormal i), [.chosenDenominator j, .literal 0x4b800000] =>
    guard (i == j)
    return write state dst (.normalizedDenominator i)
  | true, some (.divideNormal i), [.chosenNumerator j divisorSlot, .literal 0x4b800000] =>
    guard (i == divisorSlot)
    return write state dst (.normalizedNumerator j i)
  | _, _, _ => none

private def widenedGradient (state : State) (half : String) : Option Operand := do
  let [src, lane] := (half.replace ".reuse" "").splitOn "." | none
  let side ← if lane == "H0_H0" then some 0 else if lane == "H1_H1" then some 1 else none
  match read state src with
  | .loaded .gradient pair => return .rawGradient (2*pair + side)
  | .scaledGradientPair pair => return .gradient (2*pair + side)
  | _ => none

private def gradientNaNMask : Operand → Option Nat
  | .loaded .gradient pair => if pair < 4 then some (2 ^ pair) else none
  | .gradientNaNCoverage pairs => some pairs
  | _ => none

private def mergeGradientNaN (a b : Operand) : Option Operand := do
  let a ← gradientNaNMask a
  let b ← gradientNaNMask b
  guard (a &&& b == 0)
  return .gradientNaNCoverage (a ||| b)

private def gradientOperation (state : State) (period : Nat)
    (op dst : String) (args : List String) : Option State := do
  let first := if period == 2 then 0x3e800000 else 0x3ee00000
  let second := if period == 2 then 0x3d4ccccd else 0x3dc7ae14
  match op, args, args.map (read state) with
  | "HMNMX2.NAN", [_, _, "!PT"], [a, b, _] =>
    return write state dst (← mergeGradientNaN a b)
  | "VHMNMX.NAN", [_, _, _, "!PT"], [a, b, c, _] =>
    return write state dst (← mergeGradientNaN (← mergeGradientNaN a b) c)
  | "HSET2.BF.NAN.AND", [left, right, "PT"], [.gradientNaNCoverage 15, .gradientNaNCoverage 15, _] =>
    guard (left == right)
    return write state dst .gradientNaNFlags
  | "ISETP.NE.U32.AND", ["PT", _, "RZ", "PT"], [_, .gradientNaNFlags, _, _] =>
    return writeGuard state dst .gradientNaN
  | "ISETP.NE.U32.AND", ["PT", _, "RZ", "PT"], [_, .loaded .gradient 0, _, _] =>
    guard (state.width == 2)
    return writeGuard state dst .gradientNonzero
  | "HADD2.F32", ["-RZ", half], _ =>
    return write state dst (← widenedGradient state half)
  | "FMUL", _, [.rawGradient slot, .literal 0x3e000000] =>
    return write state dst (.scaledRawGradient slot)
  | "F2FP.F16.F32.PACK_AB", _, [.scaledRawGradient high, .scaledRawGradient low] =>
    guard (low % 2 == 0 && high == low + 1)
    return write state dst (.scaledGradientPair (low / 2))
  | "HMUL2", [_, "0.125", "0.125"], [.loaded .gradient pair, _, _]
  | "HFMA2.MMA", [_, "0.125", "0.125", "-RZ"], [.loaded .gradient pair, _, _, _] =>
    guard state.gradientNaNFree
    return write state dst (.scaledGradientPair pair)
  | "FMUL", _, [.gradient slot, .literal b] =>
    guard (b == first)
    return write state dst (.gradientFirstMix slot)
  | "FMUL", _, [.gradient slot, .gradient other] =>
    guard (slot == other)
    return write state dst (.squaredGradient slot)
  | "FMUL", _, [.squaredGradient slot, .literal b] =>
    guard (b == second)
    return write state dst (.gradientSecondMix slot)
  | _, _, _ => none

private def applyOperation (layout : Layout) (route : Route) (period : Nat) (state : State)
    (op dst : String) (args : List String) : Option State := do
  let converted := args.map (read state)
  let b1 := if period == 2 then 0x3f400000 else 0x3f100000
  let b2 := if period == 2 then 0x3f733333 else 0x3f670a3d
  match op, dst, args, converted with
  | "VOTE.ANY", _, [input], _ =>
    let value ← voteValue layout route state dst input
    return { writePredicate state dst (some value) with
      votes := state.votes + 1
      gradientNaNFree := state.gradientNaNFree || (state.votes == 0 && !value) }
  | "LDG.E.128", _, [pointer], _ => loadWide state 4 dst pointer
  | "LDG.E.64", _, [pointer], _ => loadWide state 2 dst pointer
  | "LDG.E", _, [pointer], _ => loadWide state 1 dst pointer
  | "ULDC", _, ["c[0x0][0x228]"], _ => return write state dst .stepScalar
  | "ULDC", _, ["c[0x0][0x238]"], _ => return write state dst .decayScalar
  | "LDC.64", _, ["c[0x0][0x230]"], _ => return write state dst .oldWeightArg
  | "LDC.64", _, ["c[0x0][0x250]"], _ => return write state dst .newWeightArg
  | "LDC.64", _, ["c[0x0][0x210]"], _ => return write state dst (.momentArg .first)
  | "LDC.64", _, ["c[0x0][0x218]"], _ => return write state dst .gradientArg
  | "LDC.64", _, ["c[0x0][0x220]"], _ => return write state dst (.momentArg .second)
  | "LDC.64", _, ["c[0x0][0x240]"], _ => return write state dst (.momentResultArg .first)
  | "LDC.64", _, ["c[0x0][0x248]"], _ => return write state dst (.momentResultArg .second)
  | "ULDC.64", _, ["c[0x0][0x208]"], _ => return write state dst .descriptor
  | "S2R", _, ["SR_TID.X"], _ => return write state dst .threadIndex
  | "S2R", _, ["SR_CTAID.X"], _ => return write state dst .ctaIndex
  | "SHF.L.U32", _, [_, shift, "RZ"], [.threadIndex, _, _] =>
    guard (shift == (if state.width == 8 then "0x3" else "0x1"))
    return write state dst .stridedLane
  | "LOP3.LUT", _, [_, mask, "RZ", "0xc0", "!PT"], [.stridedLane, _, _, _, _] =>
    guard (mask == (if state.width == 8 then "0x3f8" else "0x1fe"))
    return write state dst .maskedLane
  | "LEA", _, [_, _, shift], [.ctaIndex, .maskedLane, _] =>
    guard (shift == (if state.width == 8 then "0xa" else "0x9"))
    return write state dst .tileIndex
  | "IMAD.WIDE", _, [_, "0x4", _], [.tileIndex, _, .momentArg port] =>
    return write state dst (.momentAddress port)
  | "IMAD.WIDE", _, [_, "0x4", _], [.tileIndex, _, .momentResultArg port] =>
    return write state dst (.momentResultAddress port)
  | "IMAD.WIDE", _, [_, "0x2", _], [.tileIndex, _, .gradientArg] =>
    return write state dst .gradientAddress
  | "IMAD.WIDE", _, [_, "0x2", _], [.tileIndex, _, .oldWeightArg] =>
    return write state dst .weightInputAddress
  | "IMAD.WIDE", _, [_, "0x2", _], [.tileIndex, _, .newWeightArg] =>
    return write state dst .weightOutputAddress
  | "PRMT", _, [_, "0x7632", _], [.weightPair low, _, _] =>
    return write state dst (.highWeightBits (low+1))
  | "SHF.L.U32", _, [_, "0x10", "RZ"], [.weightPair low, _, _] =>
    return write state dst (.weight low)
  | "SHF.L.U32", _, [_, "0x10", "RZ"], [.highWeightBits slot, _, _] =>
    return write state dst (.weight slot)
  | "FSET.BF.GT.AND", _, [_, "RZ", "PT"], [.cautiousProduct slot, _, _] =>
    return write state dst (.cautiousGate slot)
  | "FFMA", _, _, [.maskedWeight i, .decayScalar, .stepApplied j] =>
    return write state dst (if i == j then .delta i else .unknown)
  | "FADD", _, _, [.weight i, .negativeDelta j] =>
    return write state dst (if i == j then .updatedWeight i else .unknown)
  | "F2FP.BF16.F32.PACK_AB", _, _, [.updatedWeight high, .updatedWeight low] =>
    return write state dst (.updatedPair low high)
  | "LOP3.LUT", _, [_, _, _, "0xfe", "!PT"], [a, b, c, _, _] =>
    return write state dst (reduceThree a b c)
  | "LOP3.LUT", _, [_, _, "RZ", "0xfc", "!PT"], [a, b, _, _, _] =>
    return write state dst ((merge a b).map Operand.summary |>.getD .unknown)
  | "LOP3.LUT", _, [_, _, "RZ", "0xc0", "!PT"], [a, b, _, _, _] =>
    match a, b with
    | .summary ⟨.first, slots⟩, .literal 0x7f800000 =>
      return write state dst (.exponentMask slots)
    | _, _ => return write state dst ((reduceAnd a b).getD .unknown)
  | "LOP3.LUT", _, [_, _, _, "0x80", "!PT"], [a, b, c, _, _] =>
    let terms := (reduceAnd a b).bind (reduceAnd · c)
    return write state dst (terms.getD .unknown)
  | "LOP3.LUT", _, [_, "0x20000000", "RZ", "0xc", "!PT"], [.allBandBits covered, _, _, _, _] =>
    guard (covered.slots == state.slotMask)
    return write state dst (.missingBandBit covered.port)
  | "LOP3.LUT", _, [_, "0x20000000", _, "0x4c", "!PT"], [a, _, b, _, _] =>
    let some (.allBandBits covered) := reduceAnd a b | none
    guard (covered.slots == state.slotMask)
    return write state dst (.missingBandBit covered.port)
  | "LOP3.LUT", _, [_, "0x7f800000", _, "0xc8", "!PT"], [a, _, b, _, _] =>
    let some ⟨.first, slots⟩ := merge a b | none
    guard (slots == state.slotMask)
    return write state dst (.exponentMask slots)
  | "LOP3.LUT", _, [_, "0xc0000000", _, "0xc8", "!PT"], [a, _, b, _, _] =>
    let covered ← merge a b
    guard (covered.slots == state.slotMask)
    return write state dst (.highBandBits covered.port)
  | "LOP3.LUT", _, ["RZ", _, _, "RZ", "0xfc", "!PT"],
      [_, .missingBandBit port, .highBandBits high, _, _, _] =>
    guard (port == high)
    match port with
    | .second => return writeGuard state dst .badRootBand
    | .denominator => return writeGuard state dst .badDivisorBand
    | _ => none
  | "LOP3.LUT", _, ["RZ", _, "0x7fffffff", _, "0xc8", "!PT"], [_, a, _, b, _, _] =>
    let some ⟨.second, slots⟩ := merge a b | none
    guard (slots == state.slotMask)
    return writeGuard state dst .secondNonzero
  | "LOP3.LUT", _, ["RZ", _, "0xc0000000", _, "0xf8", "!PT"],
      [_, .missingBandBit port, _, .summary covered, _, _] =>
    guard (port == covered.port && covered.slots == state.slotMask)
    match port with
    | .denominator => return writeGuard state dst .badDivisorBand
    | .second => return writeGuard state dst .badRootBand
    | .first | .gradient => none
  | "LOP3.LUT", _, ["RZ", _, _, "RZ", "0xfc", "!PT"], [_, a, b, _, _, _] =>
    let some ⟨.gradient, 15⟩ := merge a b | none
    return writeGuard state dst .gradientNonzero
  | "LOP3.LUT", _, ["RZ", _, "0x7fffffff", "RZ", "0xc0", "!PT"],
      [_, .summary ⟨.second, slots⟩, _, _, _, _] =>
    guard (slots == state.slotMask)
    return writeGuard state dst .secondNonzero
  | "ISETP.NE.U32.AND", _, ["PT", _, "0x7f800000", "PT"], [_, .exponentMask slots, _, _] =>
    guard (slots == state.slotMask)
    return writeGuard state dst .firstExponentClear
  | "ISETP.EQ.U32.OR", _, ["PT", _, "0x7f800000", aggregate], [_, .exponentMask slots, _, _] =>
    guard (aggregate == dst && slots == state.slotMask)
    match state.guards[aggregate]? with
    | some .badDivisorBand => return writeGuard state dst .badBandOrNaN
    | some .badRootBand => return writeGuard state dst .badRootBandOrNaN
    | _ => none
  | "ISETP.GT.U32.AND", _, ["PT", _, "0x727fffff", "PT"], [_, .shiftedRoot slot, _, _] =>
    return { writePredicate state dst (some (route.rootSlowMask.testBit slot)) with
      rootTests := slot :: state.rootTests }
  | "FSETP.GT.AND", _, ["PT", _, _, "PT"], [_, .absoluteDenominator slot, .literal 0x7e800000, _] =>
    return writeGuard state dst (.divideHigh slot)
  | "FSETP.GEU.AND", _, ["PT", _, _, "PT"], [_, .absoluteDenominator slot, .literal 0x00800000, _] =>
    return writeGuard state dst (.divideNormal slot)
  | "MOV", _, _, [value] => return write state dst value
  | "FFMA", _, _, [.loaded .first slot, .literal factor, .gradientFirstMix gradient] =>
    return write state dst (if factor == b1 && slot == gradient then .first slot else .unknown)
  | "FFMA", _, _, [.loaded .second slot, .literal factor, .gradientSecondMix gradient] =>
    return write state dst (if factor == b2 && slot == gradient then .second slot else .unknown)
  | "IADD3", _, [_, "-0xd000000", "RZ"], [.second slot, _, _] =>
    return write state dst (.shiftedRoot slot)
  | "MUFU.RSQ", _, _, [.second slot] => return write state dst (.rootReciprocal slot)
  | "FMUL.FTZ", _, _, [.second i, .rootReciprocal j] =>
    return write state dst (if i == j then .roughRoot i else .unknown)
  | "FMUL.FTZ", _, _, [.rootReciprocal i, .literal 0x3f000000] =>
    return write state dst (.halfRootReciprocal i)
  | "FFMA", _, _, [.negativeRoughRoot i, .roughRoot j, .second k] =>
    return write state dst (if i == j && j == k then .rootResidual i else .unknown)
  | "FFMA", _, _, [.rootResidual i, .halfRootReciprocal j, .roughRoot k] =>
    if i == j && j == k then return { write state dst (.root i) with roots := i :: state.roots }
    else return write state dst .unknown
  | "FFMA", _, _, [.halfRootReciprocal i, .rootResidual j, .roughRoot k] =>
    guard (i == j && j == k)
    return { write state dst (.root i) with roots := i :: state.roots, rootSwaps := i :: state.rootSwaps }
  | "FADD", _, _, [.second 0, .literal 0x2edbe6ff] =>
    guard (layout != .reference && route.secondZero &&
      state.votes == (if layout.rootGuardFirst then 3 else 2))
    return write state dst (.denominator 8)
  | "FADD", _, _, [.root slot, .literal b] =>
    return write state dst (if b == eps then .denominator slot else .unknown)
  | "FSEL", _, [_, _, condition], [.quarterNumerator i, .first j, _] =>
    guard (i == j)
    let some (.divideHigh divisorSlot) := state.guards[condition]? | none
    guard (divisorSlot == i || divisorSlot == 8)
    return write state dst (.chosenNumerator i divisorSlot)
  | "MUFU.RCP", _, _, [.denominator slot] =>
    return { write state dst (.inverse slot false) with reciprocals := state.reciprocals + 1 }
  | "MUFU.RCP", _, _, [.normalizedDenominator slot] =>
    return { write state dst (.inverse slot true) with reciprocals := state.reciprocals + 1 }
  | "MUFU.RCP", _, _, _ => none
  | "FMUL", _, _, values =>
    if let some updated := gradientOperation state period op dst args then return updated
    else return multiply state dst values
  | _, _, _, _ =>
    if let some updated := gradientOperation state period op dst args then return updated
    else if dst.startsWith "P" || dst.startsWith "!P" then return writePredicate state dst none
    else return write state dst .unknown

private def observe (op : String) : String × List String :=
  let mnemonic := (op.splitOn " ").headD ""
  let operands := trim (op.drop (mnemonic.length + 1)).toString
  (mnemonic, (operands.splitOn ",").map trim)

private def checkHelperClassification (rows : List String) (address : Nat)
    (input out : String) : Option Unit := do
  let branch := fun i (expectedPredicate : String) => do
    let [p, "BRA", j] := rows[i]!.splitOn " " | none
    guard (p == expectedPredicate && offset j == some (address + 19 * 16))
  branch 2 "@!P0"
  branch 5 "@!P0"
  branch 8 "@P0"
  guard (rows[0]! == s!"LOP3.LUT P0, RZ, {input}, 0x7fffffff, RZ, 0xc0, !PT" &&
    rows[1]! == s!"@!P0 MOV {out}, {input}" &&
    rows[3]! == s!"FSETP.GEU.FTZ.AND P0, PT, {input}, RZ, PT" &&
    rows[4]! == s!"@!P0 MOV {out}, 0x7fffffff" &&
    rows[6]! == s!"FSETP.GTU.FTZ.AND P0, PT, |{input}|, +INF , PT" &&
    rows[7]! == s!"@P0 FADD.FTZ {out}, {input}, 1" &&
    rows[9]! == s!"FSETP.NEU.FTZ.AND P0, PT, |{input}|, +INF , PT")

private def helper (code : Array Instruction) (address : Nat) :
    Option (List String × String × String × String) := do
  let start ← code.findIdx? (·.address == address)
  let rows := ((code.toList.drop start).take 22).map (·.text)
  guard (rows.length ≥ 21)
  let ["@!P0", "MOV", out, input] := rows[1]!.splitOn " " | none
  let out := (out.dropEnd 1).toString
  let ["@P0", "FFMA", scaled, source, "1.84467440737095516160e+19,", "RZ"] :=
    rows[10]!.splitOn " " | none
  let scaled := (scaled.dropEnd 1).toString
  guard (source == input ++ ",")
  let ["@P0", "MUFU.RSQ", inverse, s] := rows[11]!.splitOn " " | none
  let inverse := (inverse.dropEnd 1).toString
  guard (s == scaled)
  let ["@P0", "FMUL.FTZ", approx, scaledWithComma, reciprocal] :=
    rows[12]!.splitOn " " | none
  guard (scaledWithComma == scaled ++ "," && reciprocal == inverse)
  let approx := (approx.dropEnd 1).toString
  let ["@P0", "FADD.FTZ", residual, minusApprox, "-RZ"] := rows[14]!.splitOn " " | none
  guard (minusApprox == "-" ++ approx ++ ",")
  let residual := (residual.dropEnd 1).toString
  let ["@P0", "FFMA", corrected, r, h, squareRoot] := rows[16]!.splitOn " " | none
  let corrected := (corrected.dropEnd 1).toString
  guard (r == residual ++ "," && h == inverse ++ "," && squareRoot == approx)
  let copy := rows[19]! == s!"MOV {input}, {out}"
  let result := if copy then input else out
  let ["HFMA2.MMA", clobbered, "-RZ,", "RZ,", "0,", "0"] :=
    rows[if copy then 20 else 19]!.splitOn " " | none
  let clobbered := (clobbered.dropEnd 1).toString
  let ["RET.REL.NODEC", returns, "0x0"] := rows[if copy then 21 else 20]!.splitOn " " | none
  let scratch := [out, scaled, inverse, approx, residual, corrected, clobbered]
  guard ((input :: returns :: scratch).all register &&
    ([input, scaled, inverse, approx, residual, returns].eraseDups).length == 6 &&
    !scratch.contains returns && !scratch.contains input && clobbered != result)
  checkHelperClassification rows address input out
  guard (rows[10]! == s!"@P0 FFMA {scaled}, {input}, 1.84467440737095516160e+19, RZ" &&
    rows[11]! == s!"@P0 MUFU.RSQ {inverse}, {scaled}" &&
    rows[12]! == s!"@P0 FMUL.FTZ {approx}, {scaled}, {inverse}" &&
    rows[13]! == s!"@P0 FMUL.FTZ {inverse}, {inverse}, 0.5" &&
    rows[14]! == s!"@P0 FADD.FTZ {residual}, -{approx}, -RZ" &&
    rows[15]! == s!"@P0 FFMA {residual}, {approx}, {residual}, {scaled}" &&
    rows[16]! == s!"@P0 FFMA {corrected}, {residual}, {inverse}, {approx}" &&
    rows[17]! == s!"@!P0 MOV {out}, {input}" &&
    rows[18]! == s!"@P0 FMUL.FTZ {out}, {corrected}, 2.3283064365386962891e-10" &&
    rows[if copy then 20 else 19]! == s!"HFMA2.MMA {clobbered}, -RZ, RZ, 0, 0")
  return (scratch.eraseDups, returns, input, result)

private def helperTwo (code : Array Instruction) (address : Nat) :
    Option (List String × String × String × String) := do
  let start ← code.findIdx? (·.address == address)
  let rows := ((code.toList.drop start).take 24).map (·.text)
  let ["@!P0", "MOV", out, input] := rows[1]!.splitOn " " | none
  let out := (out.dropEnd 1).toString
  let ["@P0", "FFMA", scaled, source, "1.84467440737095516160e+19,", "RZ"] :=
    rows[10]!.splitOn " " | none
  let scaled := (scaled.dropEnd 1).toString
  guard (source == input ++ ",")
  let ["@P0", "MUFU.RSQ", inverse, s] := rows[11]!.splitOn " " | none
  let inverse := (inverse.dropEnd 1).toString
  guard (s == scaled)
  let ["@P0", "FMUL.FTZ", approx, scaledWithComma, reciprocal] :=
    rows[12]!.splitOn " " | none
  let approx := (approx.dropEnd 1).toString
  guard (scaledWithComma == scaled ++ "," && reciprocal == inverse)
  let ["@P0", "FMUL.FTZ", half, inv, "0.5"] := rows[13]!.splitOn " " | none
  let half := (half.dropEnd 1).toString
  guard (inv == inverse ++ ",")
  let ["@P0", "FADD.FTZ", residual, minusApprox, "-RZ"] := rows[15]!.splitOn " " | none
  let residual := (residual.dropEnd 1).toString
  guard (minusApprox == "-" ++ approx ++ ",")
  let ["@P0", "FFMA", corrected, r, h, squareRoot] := rows[17]!.splitOn " " | none
  let corrected := (corrected.dropEnd 1).toString
  guard (r == residual ++ "," && h == half ++ "," && squareRoot == approx)
  let ["HFMA2.MMA", clobbered, "-RZ,", "RZ,", "0,", "0"] := rows[19]!.splitOn " " | none
  let clobbered := (clobbered.dropEnd 1).toString
  let copy := rows[20]! == s!"MOV {input}, {out}"
  let result := if copy then input else out
  let move := if copy then 21 else 20
  let ["MOV", returnRegister, link] := rows[move]!.splitOn " " | none
  let returnRegister := (returnRegister.dropEnd 1).toString
  guard (rows[move+1]! == s!"RET.REL.NODEC {returnRegister} 0x0")
  let scratch := [out, scaled, inverse, approx, half, residual, corrected, clobbered]
  guard ((input :: link :: returnRegister :: scratch).all register &&
    ([input, scaled, inverse, approx, residual, link].eraseDups).length == 6 &&
    !scratch.contains link && !scratch.contains input && clobbered != out &&
    ![scaled, approx, residual].contains half && returnRegister != result)
  checkHelperClassification rows address input out
  guard (rows[14]! == s!"@!P0 MOV {out}, {input}" &&
    rows[16]! == s!"@P0 FFMA {residual}, {approx}, {residual}, {scaled}" &&
    rows[18]! == s!"@P0 FMUL.FTZ {out}, {corrected}, 2.3283064365386962891e-10")
  return ((returnRegister :: scratch).eraseDups, link, input, result)

private def callRoot (code : Array Instruction) (state : State) (address next : Nat) : Option State := do
  let (scratch, link, input, result) ← if state.width == 2 then helperTwo code address else helper code address
  let .second slot := read state input | none
  let .literal returnTo := read state link | none
  guard (returnTo == next)
  let state := scratch.foldl (fun st reg => write st reg .unknown) state
  return { writePredicate (write state result (.root slot)) "P0" none with
    roots := slot :: state.roots
    rootHelpers := slot :: state.rootHelpers }

private def require (current : Instruction) (value : Option α) : Except String α :=
  match value with
  | some selected => .ok selected
  | none => .error s!"Unsupported instruction at {current.address}: {current.text}"

private structure DecodedInstruction where
  form : Instruction
  predicate : String
  mnemonic : String
  operands : List String

private def decodeInstruction (form : Instruction) : DecodedInstruction :=
  let (predicate, body) := if form.text.startsWith "@" then
    let tag := (form.text.splitOn " ").headD ""
    ((tag.drop 1).toString, trim (form.text.drop (tag.length + 1)).toString)
  else ("PT", form.text)
  let (mnemonic, operands) := observe body
  ⟨form, predicate, mnemonic, operands⟩

structure Image where
  private mk ::
  private code : Array Instruction
  private table : Std.HashMap Nat DecodedInstruction

def load (source : String) : Image :=
  let code := parse source
  let table := code.foldl (fun table form =>
    table.insert form.address (decodeInstruction form)) ({} : Std.HashMap Nat DecodedInstruction)
  ⟨code, table⟩

private def run (image : Image) (layout : Layout) (route : Route)
    (period : Nat) (masked : Bool) (width : Nat) : Except String State := do
  unless width == 2 || width == 8 do throw "Unsupported lane width"
  let mut pc := 0
  -- Vote zero is reserved for the packed gradient instruction sequence.
  let mut state : State := {width, votes := if masked && width == 8 then 0 else 1}
  for _ in [:image.code.size] do
    let some decoded := image.table[pc]? | throw s!"Missing instruction at {pc}"
    let current := decoded.form
    let predicate := decoded.predicate
    let active := test state predicate
    let op := decoded.mnemonic
    let params := decoded.operands
    if op == "EXIT" then return state
    let next := pc + 16
    pc := next
    if op == "BRA" then
      let condition ← require current active
      let [destination] := params | throw s!"Missing target at {current.address}"
      if condition then pc ← require current (offset destination)
    else if op == "CALL.REL.NOINC" then
      let [target] := params | throw s!"Missing call target at {current.address}"
      if active == some true then state ← require current (callRoot image.code state (← require current (offset target)) next)
      else if active.isNone then throw s!"Unknown call predicate at {current.address}"
    else if (op == "STG.E.128" || op == "STG.E.64" || op == "STG.E") && active != some false then
      let [pointer, src] := params | throw s!"Incorrect global store at {current.address}"
      let words := if op == "STG.E.128" then 4 else if op == "STG.E.64" then 2 else 1
      state ← require current (storeWide state masked words predicate pointer src)
    else if op.startsWith "STG." && active != some false then
      throw s!"Unsupported global store at {current.address}: {current.text}"
    else if active != some false then
      let dst :: src := params | throw s!"Missing operands at {current.address}"
      state ← if active == some true then require current (applyOperation layout route period state op dst src)
        else if state.guards[(if predicate.startsWith "!" then (predicate.drop 1).toString else predicate)]?.isSome then
          require current (applyPredicated state predicate op dst src)
        else pure (write state dst .unknown)
  throw "Did not reach EXIT"

inductive RootPlan where
  | absent
  | classified (helperMask : Nat) (correctionMask : Nat := 0)
  | swappedCorrection
  deriving BEq, Repr, DecidableEq

def RootPlan.normalized : RootPlan → RootPlan
  | .classified mask _ => .classified mask
  | other => other

structure Output where
  terms : List Quotient
  reciprocals : Nat
  root : RootPlan
  deriving BEq, Repr, DecidableEq

def Output.normalized (out : Output) : Output :=
  {out with root := out.root.normalized}

def Image.diagnose (image : Image) (layout : Layout) (route : Route) (period : Nat) (masked : Bool := true) (width : Nat := 8) :
    Except String Output := do
  let result ← run image layout route period masked width
  let outputs := result.divisions.mergeSort (fun a b => a.slot ≤ b.slot)
  unless ((outputs.map (·.slot)) == List.range width) do
    throw s!"Unmatched division slots: {repr outputs}"
  unless result.applied.mergeSort == List.range width do
    throw s!"Unmatched learning-rate multiplications: {repr result.applied}"
  unless result.storedFirst.mergeSort == List.range width && result.storedSecond.mergeSort == List.range width do
    throw s!"Moment stores did not cover both output banks: {repr result.storedFirst}, {repr result.storedSecond}"
  unless result.gradientLoads == 1 && result.gradientClears == 1 do
    throw s!"Expected one checked workspace read and clear, found {result.gradientLoads} and {result.gradientClears}"
  unless result.storedWeight == List.range width do
    throw s!"Weight stores did not cover all updated BF16 weights: {repr result.storedWeight}"
  let usesRoot := layout == .reference || !route.secondZero
  unless result.roots.mergeSort == (if usesRoot then List.range width else []) do
    throw s!"Incorrect square-root coverage: {repr result.roots}"
  let fastRoot := (layout == .guardedReciprocal || layout.rootGuardFirst) &&
    !route.rootRejected && !route.secondZero
  let tests := if usesRoot && !fastRoot then List.range width else []
  unless result.rootTests.mergeSort == tests do
    throw s!"Incorrect square-root classification coverage: {repr result.rootTests}"
  let expectedHelpers := tests.filter route.rootSlowMask.testBit
  unless result.rootHelpers.mergeSort == expectedHelpers do
    throw s!"Incorrect square-root helper trace: {repr result.rootHelpers}"
  let expectedSwaps := if fastRoot then List.range width else []
  unless ((!masked || width == 2) && usesRoot && !fastRoot) || result.rootSwaps.mergeSort == expectedSwaps do
    throw s!"Incorrect correction operand ordering: {repr result.rootSwaps}"
  let expectedVotes := match layout with
    | .reference => 1
    | .dynamicZero => if route.secondZero then 2 else 3
    | .sharedReciprocal => 3
    | .guardedReciprocal => if route.secondZero then 3 else 4
    | .rootFirst => if route.rootRejected then 4 else 3
    | .directAfterRoot => if route.rootRejected then 4 else 2
  unless result.votes == expectedVotes do
    throw s!"Expected {expectedVotes} votes, found {result.votes}"
  let root := if !usesRoot then RootPlan.absent else if fastRoot then .swappedCorrection
    else .classified route.rootSlowMask
    (result.rootSwaps.foldl (fun mask slot => mask ||| (1 <<< slot)) 0)
  return ⟨outputs, result.reciprocals, root⟩

def Image.rawDecode (image : Image) (layout : Layout) (route : Route) (period : Nat) (masked : Bool := true) (width : Nat := 8) : Option Output :=
  (image.diagnose layout route period masked width).toOption

def Image.decode (image : Image) (layout : Layout) (route : Route) (period : Nat) (masked : Bool := true) (width : Nat := 8) : Option Output :=
  (image.rawDecode layout route period masked width).map Output.normalized

def diagnose (source : String) (layout : Layout) (route : Route) (period : Nat) (masked : Bool := true) (width : Nat := 8) :
    Except String Output :=
  (load source).diagnose layout route period masked width

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.NativeCombinedTrace
