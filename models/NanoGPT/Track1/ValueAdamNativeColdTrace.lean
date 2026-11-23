import models.NanoGPT.Track1.ValueAdamElementwise
import Std.Data.HashMap

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.NativeColdTrace

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

private inductive MomentPort where
  | first | second
  deriving Inhabited, BEq, Repr

private structure Coverage where
  port : MomentPort
  slots : Nat
  deriving Inhabited, BEq, Repr

private inductive GuardSource where
  | second (slots : Nat)
  | firstNaN (slots : Nat)
  | firstExponentComplement (slots : Nat)
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
  | momentArg (port : MomentPort)
  | momentAddress (port : MomentPort)
  | momentResultArg (port : MomentPort)
  | momentResultAddress (port : MomentPort)
  | oldWeightArg
  | newWeightArg
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
  | loaded (port : MomentPort) (slot : Nat)
  | first (slot : Nat)
  | second (slot : Nat)
  | shiftedZero
  | summary (covered : Coverage)
  | exponentMask (slots : Nat)
  | inverse
  | quotient (slot : Nat)
  | stepScalar
  deriving Inhabited, BEq, Repr

private def eps : Nat := 0x2edbe6ff

private def floatLiteral : String → Option Nat
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
  deriving BEq, Repr, Inhabited, DecidableEq

private structure State where
  registers : Std.HashMap String Operand := {}
  predicates : Std.HashMap String Bool := {}
  guards : Std.HashMap String GuardSource := {}
  divisions : List Quotient := []
  applied : List Nat := []
  storedFirst : List Nat := []
  storedSecond : List Nat := []
  storedWeight : List Nat := []
  reciprocals : Nat := 0
  votes : Nat := 0

private def read (state : State) (word : String) : Operand :=
  let name := (trim word).replace ".reuse" ""
  if name == "RZ" then .literal 0
  else if name.startsWith "-R" then
    match state.registers[(name.drop 1).toString]? with
    | some (.delta slot) => .negativeDelta slot
    | _ => .unknown
  else if name.startsWith "|" && name.endsWith "|" then
    match state.registers[(name.drop 1 |>.dropEnd 1).toString]? with
    | some (.literal bits) => .literal (bits &&& 0x7fffffff)
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
  | .first slot => if slot < 8 then some ⟨.first, 2 ^ slot⟩ else none
  | .second slot => if slot < 8 then some ⟨.second, 2 ^ slot⟩ else none
  | .summary covered => some covered
  | _ => none

private def merge (a b : Operand) : Option Coverage := do
  let a ← coverage a
  let b ← coverage b
  guard (a.port == b.port && (a.slots &&& b.slots) == 0)
  return ⟨a.port, a.slots ||| b.slots⟩

private def reduceThree (a b c : Operand) : Operand :=
  ((merge a b).bind fun ab => merge (.summary ab) c).map Operand.summary |>.getD .unknown

private def nanPredicate (state : State) (dst operation : String) (operands : List String) : Option State := do
  let ["PT", lhs, rhs, accumulated] := operands | none
  let .first i := read state lhs | none
  guard (lhs == rhs && i < 8)
  let bit := 2 ^ i
  if operation == "FSETP.NAN.AND" then
    guard (accumulated == "PT")
    return writeGuard state dst (.firstNaN bit)
  else
    let .firstNaN covered ← state.guards[accumulated]? | none
    guard (accumulated == dst && (covered &&& bit) == 0)
    return writeGuard state dst (.firstNaN (covered ||| bit))

private def addressParts (word : String) : Option (String × String × Nat) := do
  let [descriptor, pointer] := word.splitOn "][" | none
  guard (descriptor.startsWith "desc[")
  let descriptor := (descriptor.drop 5).toString
  let [base, displacement] := pointer.splitOn ".64" | none
  if displacement == "]" then return (descriptor, base, 0)
  guard (displacement.startsWith "+" && displacement.endsWith "]")
  let bytes ← offset (displacement.drop 1 |>.dropEnd 1).toString
  guard (bytes % 16 == 0)
  return (descriptor, base, bytes / 4)

private def loadWide (state : State) (dst arg : String) : Option State := do
  let start ← (dst.drop 1).toString.toNat?
  let (descriptor, base, first) ← addressParts arg
  guard (read state descriptor == .descriptor)
  let word := match read state base with
    | .momentAddress port => fun i => Operand.loaded port (first + i)
    | .weightInputAddress => fun i => Operand.weightPair (2 * (first + i))
    | _ => fun _ => .unknown
  return (List.range 4).foldl (fun state i =>
    write state s!"R{start + i}" (word i)) state

private def storeMoments (state : State) (arg src : String) : Option State := do
  let (descriptor, base, first) ← addressParts arg
  guard (read state descriptor == .descriptor)
  if read state base == .weightOutputAddress then
    let start ← (src.drop 1).toString.toNat?
    guard (first == 0)
    let terms := (List.range 4).map fun i => read state s!"R{start+i}"
    let expected := (List.range 4).map fun i => Operand.updatedPair (2*i) (2*i+1)
    guard (terms == expected)
    return {state with storedWeight := List.range 8 ++ state.storedWeight}
  else
    let .momentResultAddress port := read state base | return state
    let start ← (src.drop 1).toString.toNat?
    let slots := (List.range 4).map (first + ·)
    let terms := (List.range 4).map fun i => read state s!"R{start+i}"
    match port with
    | .first =>
      guard (terms == slots.map Operand.first)
      return {state with storedFirst := slots ++ state.storedFirst}
    | .second =>
      guard (terms == slots.map Operand.second)
      return {state with storedSecond := slots ++ state.storedSecond}

private def compare (state : State) (operation : String) (operands : List String) : Option Bool := do
  let ["PT", left, right, "PT"] := operands | none
  match operation, read state left, read state right with
  | "FSETP.GT.AND", .literal a, .literal b => some (a > b)
  | "FSETP.GEU.AND", .literal a, .literal b => some (a ≥ b)
  | "ISETP.GT.U32.AND", .shiftedZero, .literal 0x727fffff => some true
  | _, _, _ => none

private def multiply (state : State) (dst : String) (args : List Operand) : State :=
  match args with
  | [.first i, .inverse] =>
    { write state dst (.quotient i) with divisions := ⟨i, false⟩ :: state.divisions }
  | [.inverse, .first i] =>
    { write state dst (.quotient i) with divisions := ⟨i, true⟩ :: state.divisions }
  | [.quotient i, .stepScalar] =>
    { write state dst (.stepApplied i) with applied := i :: state.applied }
  | [.weight i, .stepApplied j] =>
    write state dst (if i == j then .cautiousProduct i else .unknown)
  | [.weight i, .cautiousGate j] =>
    write state dst (if i == j then .maskedWeight i else .unknown)
  | _ => write state dst .unknown

inductive Layout where
  | reference
  | shared
  | nanGuard
  | exponentGuard
  deriving BEq, Repr

structure Route where
  gradientNaN : Bool
  firstGuard : Bool

private def validVote (layout : Layout) (state : State) (dst : String) : Bool :=
  if state.votes == 0 then true
  else
    let expected := if state.votes == 1 then GuardSource.second 255
      else if layout == .exponentGuard then .firstExponentComplement 255 else .firstNaN 255
    state.guards[dst]? == some expected

private def voteValue (layout : Layout) (route : Route) (index : Nat) : Option Bool :=
  match index, layout with
  | 0, _ => some route.gradientNaN
  | 1, .reference => none
  | 1, _ => some false
  | 2, .nanGuard | 2, .exponentGuard => some route.firstGuard
  | _, _ => none

private def applyOperation (layout : Layout) (route : Route) (period : Nat) (state : State)
    (op dst : String) (args : List String) : Option State := do
  let converted := args.map (read state)
  let b1 := if period == 2 then 0x3f400000 else 0x3f100000
  let b2 := if period == 2 then 0x3f733333 else 0x3f670a3d
  match op, dst, args, converted with
  | "VOTE.ANY", _, [input], _ =>
    let expected := if state.votes == 2 && layout == .exponentGuard then "!" ++ dst else dst
    guard (dst == "P0" && input == expected && validVote layout state dst)
    let value ← voteValue layout route state.votes
    return { writePredicate state dst (some value) with votes := state.votes + 1 }
  | "LDG.E.128", _, [pointer], _ => loadWide state dst pointer
  | "STG.E.128", _, [words], _ => storeMoments state dst words
  | "ULDC", _, ["c[0x0][0x228]"], _ => return write state dst .stepScalar
  | "ULDC", _, ["c[0x0][0x238]"], _ => return write state dst .decayScalar
  | "LDC.64", _, ["c[0x0][0x230]"], _ => return write state dst .oldWeightArg
  | "LDC.64", _, ["c[0x0][0x250]"], _ => return write state dst .newWeightArg
  | "LDC.64", _, ["c[0x0][0x210]"], _ => return write state dst (.momentArg .first)
  | "LDC.64", _, ["c[0x0][0x220]"], _ => return write state dst (.momentArg .second)
  | "LDC.64", _, ["c[0x0][0x240]"], _ => return write state dst (.momentResultArg .first)
  | "LDC.64", _, ["c[0x0][0x248]"], _ => return write state dst (.momentResultArg .second)
  | "ULDC.64", _, ["c[0x0][0x208]"], _ => return write state dst .descriptor
  | "S2R", _, ["SR_TID.X"], _ => return write state dst .threadIndex
  | "S2R", _, ["SR_CTAID.X"], _ => return write state dst .ctaIndex
  | "SHF.L.U32", _, [_, "0x3", "RZ"], [.threadIndex, _, _] =>
    return write state dst .stridedLane
  | "LOP3.LUT", _, [_, "0x3f8", "RZ", "0xc0", "!PT"], [.stridedLane, _, _, _, _] =>
    return write state dst .maskedLane
  | "LEA", _, [_, _, "0xa"], [.ctaIndex, .maskedLane, _] =>
    return write state dst .tileIndex
  | "IMAD.WIDE", _, [_, "0x4", _], [.tileIndex, _, .momentArg port] =>
    return write state dst (.momentAddress port)
  | "IMAD.WIDE", _, [_, "0x4", _], [.tileIndex, _, .momentResultArg port] =>
    return write state dst (.momentResultAddress port)
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
  | "LOP3.LUT", _, ["RZ", _, "0x7fffffff", _, "0xc8", "!PT"], [_, a, _, b, _, _] =>
    match merge a b with
    | some ⟨.second, covered⟩ => return writeGuard state dst (.second covered)
    | _ => return writePredicate state dst none
  | "LOP3.LUT", _, [_, "0x7f800000", _, "0xc8", "!PT"], [a, _, b, _, _] =>
    match merge a b with
    | some ⟨.first, covered⟩ => return write state dst (.exponentMask covered)
    | _ => return write state dst .unknown
  | "ISETP.NE.U32.AND", _, ["PT", _, "0x7f800000", "PT"], [_, .exponentMask covered, _, _] =>
    return writeGuard state dst (.firstExponentComplement covered)
  | "FSETP.NAN.AND", _, _, _ | "FSETP.NAN.OR", _, _, _ =>
    nanPredicate state dst op args
  | "MOV", _, _, [value] => return write state dst value
  | "FFMA", _, _, [.loaded .first slot, .literal factor, _] =>
    return write state dst (if factor == b1 then .first slot else .unknown)
  | "FFMA", _, _, [.loaded .second slot, .literal factor, _] =>
    return write state dst (if factor == b2 then .second slot else .unknown)
  | "FADD", _, _, [.second _, .literal b] =>
    return write state dst (if b == eps then .literal eps else .unknown)
  | "IADD3", _, [_, "-0xd000000", "RZ"], [.second _, _, _] =>
    return write state dst .shiftedZero
  | "FSEL", _, [_, _, condition], [a, b, _] =>
    return write state dst (match test state condition with
      | some true => a
      | some false => b
      | none => .unknown)
  | "MUFU.RCP", _, _, [.literal b] =>
    guard (b == eps)
    return { write state dst .inverse with reciprocals := state.reciprocals + 1 }
  | "MUFU.RCP", _, _, _ => none
  | "FMUL", _, _, values => return multiply state dst values
  | _, _, _, _ =>
    if dst.startsWith "P" || dst.startsWith "!P" then
      return writePredicate state dst (compare state op args)
    else return write state dst .unknown

private def helperRet (code : Array Instruction) (address : Nat) : Option (String × String) := do
  let start ← code.findIdx? (·.address == address)
  let rows := (code.toList.drop start).take 22
  let "LOP3.LUT P0, RZ, R0, 0x7fffffff, RZ, 0xc0, !PT" :: move :: jump :: _ := rows.map (·.text) | none
  let ["@!P0", "MOV", temp, "R0"] := move.splitOn " " | none
  guard (temp.endsWith ",")
  let temp := (temp.dropEnd 1).toString
  let ["@!P0", "BRA", joined] := jump.splitOn " " | none
  guard (offset joined == some (address + 19 * 16))
  let [restore, scratch, ret] := (rows.drop 19).map (·.text) | none
  let ["RET.REL.NODEC", returns, "0x0"] := ret.splitOn " " | none
  guard (restore == s!"MOV R0, {temp}" && scratch == s!"HFMA2.MMA {temp}, -RZ, RZ, 0, 0")
  return (temp, returns)

private def callZero (code : Array Instruction) (state : State) (address next : Nat) : Option State := do
  let Operand.second _ := state.registers["R0"]? |>.getD .unknown | none
  let (scratch, link) ← helperRet code address
  let .literal returnTo := state.registers[link]?.getD .unknown | none
  guard (returnTo == next)
  return writePredicate (write state scratch .unknown) "P0" (some false)

private def observe (op : String) : String × List String :=
  let mnemonic := (op.splitOn " ").headD ""
  let operands := trim (op.drop (mnemonic.length + 1)).toString
  (mnemonic, operands.splitOn ", ")

private def require (current : Instruction) (value : Option α) : Except String α :=
  match value with
  | some selected => .ok selected
  | none => .error s!"Unsupported instruction at {current.address}: {current.text}"

private def run (code : Array Instruction) (layout : Layout) (route : Route)
    (period : Nat) : Except String State := do
  let table := code.foldl (fun (map : Std.HashMap Nat String) op => map.insert op.address op.text) {}
  let mut pc := 0
  let mut state : State := {}
  for _ in [:code.size] do
    let some text := table[pc]? | throw s!"Missing instruction at {pc}"
    let current := ⟨pc, text⟩
    let (active, body) := if text.startsWith "@" then
      let predicate := (text.splitOn " ").headD ""
      (test state (predicate.drop 1).toString, trim (text.drop (predicate.length + 1)).toString)
    else (some true, text)
    let (op, params) := observe body
    if op == "EXIT" then return state
    let next := pc + 16
    pc := next
    if op == "BRA" then
      let condition ← require current active
      let [destination] := params | throw s!"Missing target at {current.address}"
      if condition then pc ← require current (offset destination)
    else if op == "CALL.REL.NOINC" then
      let [target] := params | throw s!"Missing call target at {current.address}"
      if active == some true then state ← require current (callZero code state (← require current (offset target)) next)
      else if active.isNone then throw s!"Unknown call predicate at {current.address}"
    else if active != some false then
      let dst :: src := params | throw s!"Missing operands at {current.address}"
      state ← if active == some true then require current (applyOperation layout route period state op dst src)
        else pure (write state dst .unknown)
  throw "Did not reach EXIT"

structure Output where
  terms : List Quotient
  reciprocals : Nat
  deriving BEq, Repr, DecidableEq

def diagnose (source : String) (layout : Layout) (route : Route) (period : Nat) : Except String Output := do
  let result ← run (parse source) layout route period
  let outputs := result.divisions.mergeSort (fun a b => a.slot ≤ b.slot)
  unless ((outputs.map (·.slot)) == List.range 8) do
    throw s!"Unmatched division slots: {repr outputs}"
  unless result.applied.mergeSort == List.range 8 do
    throw s!"Unmatched learning-rate multiplications: {repr result.applied}"
  unless result.storedFirst.mergeSort == List.range 8 && result.storedSecond.mergeSort == List.range 8 do
    throw s!"Moment stores did not cover both output banks: {repr result.storedFirst}, {repr result.storedSecond}"
  unless result.storedWeight == List.range 8 do
    throw s!"Weight stores did not cover all updated BF16 weights: {repr result.storedWeight}"
  let expectedVotes := match layout with
    | .reference => 1
    | .shared => 2
    | _ => 3
  unless result.votes == expectedVotes do
    throw s!"Expected {expectedVotes} votes, found {result.votes}"
  return ⟨outputs, result.reciprocals⟩

def decode (source : String) (layout : Layout) (route : Route) (period : Nat) : Option Output :=
  (diagnose source layout route period).toOption

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.NativeColdTrace
