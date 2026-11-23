import models.NanoGPT.Track1.MuddActivationStorage
import models.NanoGPT.Track1.MuddResidualStorage
import models.NanoGPT.Track1.MuddGradientGrouping
import models.NanoGPT.Track1.MuddGradientPacking
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Check the whole-graph reduction without freezing its arithmetic or
a particular block size. The checked edit erases one duplicate store
and the SSA addresses that reference that store's output pointer. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddActivationStorage.Binding

set_option maxRecDepth 8192

private inductive Token where
  | ssa (name : String)
  | text (char : Char)

private def isIdentifier (c : Char) : Bool :=
  c.isAlphanum || c == '_' || c == '.'

private def tokens : Nat → List Char → List Token
  | 0, _ | _, [] => []
  | fuel + 1, '%' :: rest =>
    let name := rest.takeWhile isIdentifier
    .ssa ("%" ++ String.ofList name) :: tokens fuel (rest.drop name.length)
  | fuel + 1, char :: rest => .text char :: tokens fuel rest

private def tokenize (text : String) : List Token := tokens text.length text.toList

def ssaNames (text : String) : List String :=
  (tokenize text).filterMap fun
    | .ssa name => some name
    | .text _ => none

def rename (text : String) (aliases : Std.HashMap String String) : String :=
  String.join ((tokenize text).map fun
    | .ssa name => (aliases[name]?).getD name
    | .text c => String.singleton c)

private def eraseDuplicate (lines : List String) : Option (List String) := do
  let mut aliases : Std.HashMap String String := ({} : Std.HashMap String String).insert "%out_ptr2" "%out_ptr1"
  let mut definitions : Std.HashMap String String := {}
  let mut retained := []
  let mut dropped := 0
  for line in lines do
    let parts := line.splitOn " = "
    let rhs := if parts.length == 2 then parts[1]! else line
    let refs := ssaNames rhs
    if !line.startsWith "tt.func public @" && refs.any (fun name => aliases.contains name) then
      let replacement := rename rhs aliases
      if rhs.startsWith "tt.store " then
        if retained.head? != some replacement then return ← none
        dropped := dropped + 1
      else if parts.length == 2 && (rhs.startsWith "tt.splat " || rhs.startsWith "tt.addptr ") then
        let previous ← definitions[replacement]?
        aliases := aliases.insert parts[0]! previous
      else
        return ← none
    else
      if parts.length == 2 then
        definitions := definitions.insert rhs parts[0]!
      retained := line :: retained
  if dropped == 1 then some retained.reverse else none

def canonicalNames (lines : List String) : List String := Id.run do
  let mut names : Std.HashMap String String := {}
  let mut result := []
  for line in lines do
    for name in ssaNames line do
      if !names.contains name then
        names := names.insert name s!"%{names.size}"
    result := rename line names :: result
  return result.reverse

def recognizes (reference candidate : String) : Bool :=
  (do
    let original ← QKVTTIR.normalizeTTIR reference
    let rewritten ← QKVTTIR.normalizeTTIR candidate
    let oneStore ← eraseDuplicate original
    pure (canonicalNames oneStore == canonicalNames rewritten)).getD false

structure Program where
  referenceTTIR : String
  candidateTTIR : String
  checked : recognizes referenceTTIR candidateTTIR = true

def Program.check (reference candidate : String) : Option Program :=
  if h : recognizes reference candidate then some ⟨reference, candidate, h⟩ else none

def Program.read (_ : Program) : Read := MuddActivationStorage.Program.single.read

theorem Program.read_eq (program : Program) : program.read = MuddActivationStorage.Program.duplicate.read :=
  MuddActivationStorage.read_eq

#axiomsClean Program.read_eq
end VeriTile.Bench.Models.NanoGPT.Track1.MuddActivationStorage.Binding

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddResidualStorage.Binding
open MuddActivationStorage.Binding (ssaNames rename canonicalNames)

set_option maxRecDepth 8192

private def operands (text : String) : Option (String × String × String) :=
  match text.splitOn ", " with
  | [a, b, c] => some (a, b, c)
  | _ => none

private def canonical (lines : List String) : List String := Id.run do
  let mut constants : Std.HashMap String String := {}
  let mut result := []
  for line in lines do
    match line.splitOn " = " with
    | [name, expr] =>
      if expr.startsWith "arith.constant " then
        constants := constants.insert name ("(" ++ expr ++ ")")
      else
        result := rename line constants :: result
    | _ => result := rename line constants :: result
  return canonicalNames result.reverse

/-- Find the unique BF16 store to the residual port. The later load must
use precisely the same pointer and mask. Live values may reach only
elementwise casts, products, and stores under that same mask. -/
private def replaceReload (lines : List String) : Option (List String) := do
  let mut pointers := ({} : Std.HashSet String).insert "%in_out_ptr0"
  let mut stored : Option (String × String × String × String) := none
  let mut values : Std.HashSet String := {}
  let mut aliases : Std.HashMap String String := {}
  let mut result := []
  let mut published := 0
  for line in lines do
    let parts := line.splitOn " = "
    let isDef := parts.length > 1
    let name := parts.head!
    let expr := if isDef then String.intercalate " = " parts.tail! else line
    let refs := ssaNames expr
    if !line.startsWith "tt.func public @" && refs.any (pointers.contains ·) then
      if isDef && (expr.startsWith "tt.splat " || expr.startsWith "tt.addptr ") then
        pointers := pointers.insert name
      else if line.startsWith "tt.store " then
        let [args, ty] := ((line.drop 9).toString).splitOn " : " | none
        let (ptr, val, mask) ← operands args
        if !pointers.contains ptr || stored.isSome || !ty.endsWith "x!tt.ptr<bf16>>" then return ← none
        stored := some (ptr, val, mask, ty)
      else if isDef && expr.startsWith "tt.load " then
        let (ptr, val, mask, ty) ← stored
        let [args, loadTy] := ((expr.drop 8).toString).splitOn " evictionPolicy = evict_first : " | none
        let (src, pred, _) ← operands args
        if ptr != src || mask != pred || ty != loadTy || !aliases.isEmpty then return ← none
        values := values.insert name
        aliases := aliases.insert name val
        continue
      else
        return ← none
    else if refs.any (values.contains ·) then
      if isDef && (expr.startsWith "arith.extf " || expr.startsWith "arith.mulf " ||
          expr.startsWith "tt.fp_to_fp ") then
        values := values.insert name
      else if line.startsWith "tt.store " then
        let (_, _, mask, _) ← stored
        let [args, _] := ((line.drop 9).toString).splitOn " : " | none
        let (ptr, val, pred) ← operands args
        if values.contains ptr || !values.contains val || pred != mask then return ← none
        published := published + 1
      else
        return ← none
    result := rename line aliases :: result
  if !aliases.isEmpty && (published == 1 || published == 2) then some result.reverse else none

def recognizes (block : Nat) (reference candidate : String) : Bool :=
  (do
    let original ← QKVTTIR.normalizeTTIR reference
    let rewritten ← QKVTTIR.normalizeTTIR candidate
    let expected ← if 768 ≤ block then replaceReload original else some original
    pure (canonical expected == canonical rewritten)).getD false

structure Program where
  rows : Nat
  block : Nat
  referenceTTIR : String
  candidateTTIR : String
  checked : recognizes block referenceTTIR candidateTTIR = true

def Program.check (rows block : Nat) (reference candidate : String) : Option Program :=
  if h : recognizes block reference candidate then some ⟨rows, block, reference, candidate, h⟩ else none

def Program.run (program : Program) : Run := fun rows packet =>
  if program.rows == rows && program.block == packet.block then
    MuddResidualStorage.Program.register.run rows packet
  else MuddResidualStorage.Program.reload.run rows packet

theorem Program.run_eq (program : Program) : program.run = MuddResidualStorage.Program.register.run := by
  unfold Program.run
  rw [← MuddResidualStorage.run_eq]
  simp

#axiomsClean Program.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.MuddResidualStorage.Binding

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddGradientGrouping.Binding
open MuddActivationStorage.Binding (ssaNames rename)
open MuddResidualStorage.Binding (canonical)

set_option maxRecDepth 8192

private def parseOperand (constants : Std.HashMap String Nat) (operand : String) : Option Nat :=
  operand.toNat?.or constants[operand]?

private def scalarOperands (expr : String) (marker : String) (ty : String) : Option (List String) := do
  if !expr.startsWith marker || !expr.endsWith ty then return ← none
  pure (((expr.drop marker.length).dropEnd ty.length).toString.splitOn ", ")

private def inputName (lane : Nat) : String := s!"%IN{lane}"

private def inputHeader (lane : Nat) : String :=
  inputName lane ++ ": !tt.ptr<bf16> {tt.divisibility = 16 : i32}"

private def shapeHeader (line : String) (grouped : Bool) (ports : Nat) : Bool :=
  let suffix := if grouped then String.intercalate ", " ((List.range ports).map inputHeader) ++
      ", %OUT: !tt.ptr<bf16> {tt.divisibility = 16 : i32}" else
    "%in_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %out_ptr0: !tt.ptr<bf16>, " ++
      "%xnumel: i32 {tt.divisibility = 16 : i32}, %r0_numel: i32"
  let ends (sig : String) := line.endsWith ("(" ++ sig ++ ") attributes {noinline = false} {")
  ends suffix || (!grouped && ends (suffix.replace "%out_ptr0: !tt.ptr<bf16>" "%out_ptr0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}"))

private def fixPlane (lines : List String) (plane : Option Nat) (first ports : Nat) : Option (List String) := do
  let mut aliases : Std.HashMap String String := {}
  let mut integers : Std.HashMap String Nat := {}
  let mut gotIn := false
  let mut gotOut := false
  let mut result := []
  for line in lines do
    if line.startsWith "tt.func public @" then
      if !shapeHeader line plane.isSome ports then return ← none
      result := "tt.func public @reduce" :: result
      continue
    let parts := line.splitOn " = "
    let name := parts.head!
    let rhs := String.intercalate " = " parts.tail!
    if let some p := plane then
      if let some [n] := scalarOperands rhs "arith.constant " " : i32" then
        let v ← n.toNat?
        integers := integers.insert name v
      else if rhs == "tt.get_program_id y : i32" then
        integers := integers.insert name p
        continue
      else if let some [a, b] := scalarOperands rhs "arith.cmpi eq, " " : i32" then
        let u ← parseOperand integers a
        let v ← parseOperand integers b
        integers := integers.insert name (if u == v then 1 else 0)
        continue
      else if let some [cond, yes, no] := scalarOperands rhs "arith.select " " : !tt.ptr<bf16>" then
        let pred ← integers[cond]?
        let ptr := if pred == 1 then yes else no
        aliases := aliases.insert name (aliases[ptr]?.getD ptr)
        continue
      else if let some [a, b] := scalarOperands rhs "arith.addi " " : i32" then
        if let some u := parseOperand integers a then
          if let some v := parseOperand integers b then
            integers := integers.insert name (u + v)
            continue
      else if let some [offset] := scalarOperands rhs "tt.addptr %OUT, " " : !tt.ptr<bf16>, i32" then
        let k ← parseOperand integers offset
        if gotOut || k != first + p then return ← none
        aliases := aliases.insert name "%verified_output"
        gotOut := true
        continue
      let expanded := rename line aliases
      let inputs := (ssaNames expanded).filter (·.startsWith "%IN")
      if !inputs.isEmpty then
        if gotIn || inputs != [inputName p] then return ← none
        gotIn := true
      if (ssaNames expanded).contains "%OUT" then return ← none
      result := rename expanded (({} : Std.HashMap String String).insert (inputName p) "%verified_input") :: result
    else
      let renamePorts := (({} : Std.HashMap String String).insert "%in_ptr0" "%verified_input").insert
        "%out_ptr0" "%verified_output"
      result := rename line renamePorts :: result
  if plane.isSome && !(gotIn && gotOut) then return ← none
  pure (canonical result.reverse)

def recognizesReductionPlanes (planes : List Nat) (first ports : Nat) (original grouped : List String) : Bool :=
  (do
    let expected ← fixPlane original none first ports
    pure (planes.all fun plane => fixPlane grouped (some plane) first ports == some expected)).getD false

def recognizesReduction (planes first : Nat) (original grouped : List String) : Bool :=
  recognizesReductionPlanes (List.range planes) first 9 original grouped

def recognizes (planes first : Nat) (reference candidate : String) : Bool :=
  (do
    if (planes, first) ∉ [(9, 1), (6, 4), (3, 1)] then return ← none
    let original ← QKVTTIR.normalizeTTIR reference
    let grouped ← QKVTTIR.normalizeTTIR candidate
    pure (recognizesReduction planes first original grouped)).getD false

private def hasLaunchLayout (tile warps : Nat) (ttgir : String) : Bool :=
  let attrs := s!"\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = {warps} : i32, ttg.target = \"cuda:90\""
  (ttgir.splitOn attrs).length == 2 && (ttgir.splitOn s!"tensor<{tile}x16xf32,").length > 1

structure Program where
  rows : Nat
  planes : Nat
  first : Nat
  tile : Nat
  warps : Nat
  hoistedPacking : Bool
  referenceTTIR : String
  candidateTTIR : String
  referenceTTGIR : String
  candidateTTGIR : String
  supported : (tile, warps) ∈ [(1, 2), (8, 2), (32, 4), (128, 8)] ∧ rows % tile = 0
  sourceChecked : recognizes planes first referenceTTIR candidateTTIR = true
  layoutChecked : recognizes planes first referenceTTGIR candidateTTGIR = true
  scheduleChecked : hasLaunchLayout tile warps candidateTTGIR = true
  packingColumns : hoistedPacking = true → planes = 9 ∧ first = 1

def Program.check (rows planes first tile warps : Nat)
    (referenceTTIR candidateTTIR referenceTTGIR candidateTTGIR : String) (hoistedPacking : Bool := false) :
    Option Program := do
  if plan : (tile, warps) ∈ [(1, 2), (8, 2), (32, 4), (128, 8)] ∧ rows % tile = 0 then
    if source : recognizes planes first referenceTTIR candidateTTIR = true then
      if layout : recognizes planes first referenceTTGIR candidateTTGIR = true then
        if schedule : hasLaunchLayout tile warps candidateTTGIR = true then
          if packing : hoistedPacking = true → planes = 9 ∧ first = 1 then
            return ⟨rows, planes, first, tile, warps, hoistedPacking, referenceTTIR, candidateTTIR,
              referenceTTGIR, candidateTTGIR, plan, source, layout, schedule, packing⟩
  none

def Program.run (program : Program) : Run := fun rows planes packet =>
  if rows == program.rows && planes == program.planes && packet.first == program.first &&
      packet.tile == program.tile && packet.warps == program.warps then
    MuddGradientGrouping.grouped packet
  else MuddGradientGrouping.separate packet

theorem Program.run_eq (program : Program) : program.run = MuddGradientGrouping.Program.grouped.run := by
  funext rows planes packet
  unfold Program.run MuddGradientGrouping.Program.run
  rw [MuddGradientGrouping.run_eq packet]
  simp

def Program.runPacking (program : Program) : MuddGradientPacking.Run := fun rows packet =>
  if program.hoistedPacking && rows == program.rows && packet.tile == program.tile && packet.warps == program.warps then
    MuddGradientPacking.Program.hoisted.run rows packet
  else MuddGradientPacking.Program.interleaved.run rows packet

theorem Program.runPacking_eq (program : Program) : program.runPacking = MuddGradientPacking.Program.hoisted.run := by
  unfold Program.runPacking
  rw [MuddGradientPacking.run_eq]
  simp

#axiomsClean Program.run_eq
#axiomsClean Program.runPacking_eq
end VeriTile.Bench.Models.NanoGPT.Track1.MuddGradientGrouping.Binding
