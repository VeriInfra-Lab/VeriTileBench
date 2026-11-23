import VeriTile.Triton.Float.ScalarOpsLaws
import models.NanoGPT.Track1.MuddFusedPacking
import models.NanoGPT.Track1.MuddActivationBinding
import models.NanoGPT.Track1.Proofs.MuddFourRowReduction

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddFusedPacking.Binding
open MuddActivationStorage.Binding (ssaNames rename canonicalNames)
open MuddGradientGrouping.Binding (recognizesReductionPlanes)

set_option maxRecDepth 8192

private def contains (text part : String) : Bool := (text.splitOn part).length > 1

private def definition (line : String) : String × String :=
  let terms := line.splitOn " = "
  if terms.length == 1 || !(line.startsWith "%" || line.startsWith "#") then ("", line)
  else (terms.head!, String.intercalate " = " terms.tail!)

private def layoutNames (lines : List String) : List (String × String) :=
  ((lines.filter fun line => line.startsWith "#blocked" || line.startsWith "#linear").map definition).reverse

private def inlineLayouts (layouts : List (String × String)) (line : String) : String :=
  layouts.foldl (fun text (name, body) => text.replace name body) line

private def packPort : String := ", %PACK: !tt.ptr<bf16> {tt.divisibility = 16 : i32}"

private def inlineScalarConstants (lines : List String) : List String := Id.run do
  let mut constants : Std.HashMap String String := {}
  let mut result := []
  for line in lines do
    let (name, rhs) := definition line
    if let ["arith.constant", value, ":", "i32"] := rhs.splitOn " " then
      if value.toNat?.isSome then
        constants := constants.insert name value
        continue
    result := rename line constants :: result
  pure result.reverse

/-- Keep a reduction region together so it is compared with the original
region, including its layout and add operand order. -/
private def instructions (lines : List String) : Option (List (List String)) := do
  let mut result := []
  let mut region := []
  for line in lines do
    if region.isEmpty then
      if (definition line).2.startsWith "\"tt.reduce\"(" then region := [line]
      else result := [line] :: result
    else
      region := line :: region
      if line.startsWith "}) :" then
        result := region.reverse :: result
        region := []
  if region.isEmpty then some result.reverse else none

private def reductionProjection (candidate : Bool) (lines : List String) : Option (List String) := do
  let stores := lines.filter (·.startsWith "tt.store ")
  if stores.length != (if candidate then 2 else 1) then return ← none
  let first := stores.head!
  let layouts := layoutNames lines
  let chunks ← instructions (lines.takeWhile (· != first))
  let mut live : Std.HashSet String := (ssaNames first).foldl (fun s x => s.insert x) {}
  let mut selected := [inlineLayouts layouts first]
  for chunk in chunks.reverse do
    let line := chunk.head!
    if line.startsWith "tt.func public @" then
      if candidate && !(contains line (packPort ++ ")")) then return ← none
      selected := line.replace packPort "" :: selected
    else if line.startsWith "module" then
      selected := line :: selected
    else
      let (name, rhs) := definition line
      if live.contains name then
        live := (ssaNames rhs).foldl (fun s x => s.insert x) live
        selected := chunk.map (inlineLayouts layouts) ++ selected
  pure (inlineScalarConstants selected ++ ["tt.return", "}", "}"])

private structure Index where
  cta : Nat := 0
  row : Nat := 0
  col : Nat := 0
  bias : Nat := 0
deriving BEq

private def Index.add (a b : Index) : Index :=
  ⟨a.cta + b.cta, a.row + b.row, a.col + b.col, a.bias + b.bias⟩

private def Index.scale (n : Nat) (a : Index) : Index :=
  ⟨n * a.cta, n * a.row, n * a.col, n * a.bias⟩

private def Index.literal (a : Index) : Option Nat :=
  if a.cta == 0 && a.row == 0 && a.col == 0 then some a.bias else none

private inductive Expr where
  | index (i : Index)
  | predicate (b : Bool)
  | mask (i bound : Index)
  | pointer (port : String) (i : Index)
  | bzero
  | fzero
  | tenth
  | load (p m : Expr)
  | extend (a : Expr)
  | add (a b : Expr)
  | mul (a b : Expr)
  | fma (a b c : Expr)
  | narrow (a : Expr)
  | selected (condition yes no : Expr)
  | reduced
deriving BEq

private def constant (n : Nat) : Expr := .index { bias := n }

private def operand (values : Std.HashMap String Expr) (term : String) : Option Expr :=
  values[term]?.or (term.toNat?.map constant)

private def i32Operands (values : Std.HashMap String Expr) (args : String) : Option (Index × Index) := do
  let [a, b] := args.splitOn ", " | none
  let .index a ← operand values a | none
  let .index b ← operand values b | none
  pure (a, b)

private def liveLane : Expr := .mask { col := 1 } { bias := 12 }

private def inputLoad (tile plane : Nat) : Expr :=
  let lane := if tile == 1 then 0 else 1
  let address := Expr.pointer s!"%IN{plane}" ⟨12 * tile, 12 * lane, 1, 0⟩
  .load address liveLane

private def output (tile first plane : Nat) (direct : Bool) : Expr × Expr × Expr :=
  let column := first + plane
  let lane := if tile == 1 then 0 else 1
  let value := Expr.extend (inputLoad tile plane)
  let total := if direct then Expr.fma value .tenth .fzero else .mul (.add value .fzero) .tenth
  (.pointer "%PACK" ⟨120 * tile, 120 * lane, 1, column * 12⟩, .narrow total, liveLane)

private def checkPack (tile first plane : Nat) (lines : List String) (direct scaled : Bool := false)
    (onlyPacked : Bool := false) : Option Unit := do
  let mut values : Std.HashMap String Expr := {}
  for i in [:10] do
    values := values.insert s!"%IN{i}" (.pointer s!"%IN{i}" {})
  for name in (if scaled then ["%OUT", "%PACK", "%SCALED"] else ["%OUT", "%PACK"]) do
    values := values.insert name (.pointer name {})
  let chunks ← instructions lines
  let mut loads := 0
  let mut stores := 0
  let mut reduced := false
  for chunk in chunks do
    let line := chunk.head!
    if line.startsWith "module" || line.startsWith "#blocked" || line.startsWith "#linear" || line.startsWith "tt.func public @" ||
        line == "tt.return" || line == "}" then continue
    let (target, rhs) := definition line
    if rhs.startsWith "\"tt.reduce\"(" then
      if reduced || values.contains target || chunk.length != 5 then return ← none
      if direct then
        let [input] := ssaNames rhs | return ← none
        if values[input]? != some (.selected liveLane (.extend (inputLoad tile plane)) .fzero) then
          return ← none
      reduced := true
      values := values.insert target .reduced
      continue
    let pieces := rhs.splitOn " : "
    if pieces.length < 2 then return ← none
    let ty := pieces.getLast!
    let code := String.intercalate " : " pieces.dropLast
    let op := (code.splitOn " ").head!
    let args := (code.drop (op.length + 1)).toString
    let refs ← (ssaNames args).mapM (operand values)
    if op == "tt.store" then
      if !target.isEmpty || !(contains ty "!tt.ptr<bf16>") then return ← none
      match stores, refs with
      | 0, [.pointer "%OUT" index, .narrow .reduced] =>
        if onlyPacked || (direct && index != ⟨10 * tile, if tile == 1 then 0 else 10, 0, first + plane⟩) then return ← none
      | 1, [.pointer "%SCALED" index, value] =>
        if !scaled || index != ⟨10 * tile, if tile == 1 then 0 else 10, 0, first + plane⟩ ||
            value != .narrow (.mul (.extend (.narrow .reduced)) .tenth) then return ← none
      | n, [ptr, value, mask] =>
        if n != (if onlyPacked then 0 else if scaled then 2 else 1) || (ptr, value, mask) != output tile first plane direct then
          return ← none
      | _, _ => return ← none
      stores := stores + 1
      continue
    if target.isEmpty || values.contains target then return ← none
    let value ←
      match op, refs with
      | "arith.constant", [] =>
        if args == "dense<0.000000e+00>" && contains ty "xbf16" then some .bzero
        else if args == "dense<0.000000e+00>" && contains ty "xf32" then some .fzero
        else if args == "dense<1.000000e-01>" && contains ty "xf32" then some .tenth
        else if contains ty "i32" then
          let literal := if args.startsWith "dense<" then ((args.drop 6).dropEnd 1).toString else args
          literal.toNat?.map constant
        else none
      | "tt.get_program_id", [] =>
        if args == "y" && ty == "i32" then some (constant plane)
        else if args == "x" && ty == "i32" then some (.index { cta := 1 })
        else none
      | "tt.make_range", [] =>
        let range := fun n : Nat => "{end = " ++ toString n ++ " : i32, start = 0 : i32}"
        if tile == 16 && args == range 16 && ty.startsWith "tensor<16xi32, #ttg.slice<{dim = 1," then
          some (.index { row := 1 })
        else if args == range 16 && ty.startsWith "tensor<16xi32" then some (.index { col := 1 })
        else if args == range tile && ty.startsWith s!"tensor<{tile}xi32" then some (.index { row := 1 })
        else none
      | "tt.expand_dims", [a] =>
        match a with
        | .index i =>
          if ((i.row == 1 || (direct && tile == 1 && i.col == 0)) && args.endsWith "{axis = 1 : i32}") ||
              (i.col == 1 && args.endsWith "{axis = 0 : i32}") then some a else none
        | .reduced => if args.endsWith "{axis = 1 : i32}" then some a else none
        | _ => none
      | "tt.splat", [a] | "tt.broadcast", [a] | "ttg.convert_layout", [a] => some a
      | "arith.addi", _ | "arith.muli", _ => do
        let (a, b) ← i32Operands values args
        if !(contains ty "i32") then none
        else if op == "arith.addi" then some (.index (a.add b))
        else if let some n := a.literal then some (.index (b.scale n))
        else b.literal.map fun n => .index (a.scale n)
      | "arith.andi", _ => do
        if ty != "i32" then none else do
          let (a, b) ← i32Operands values args
          pure (constant ((← a.literal) &&& (← b.literal)))
      | "arith.cmpi", _ => do
        let [cmp, a, b] := args.splitOn ", " | none
        let (a, b) ← i32Operands values (a ++ ", " ++ b)
        if !(contains ty "i32") then none
        else if cmp == "eq" || cmp == "ne" then do
          let x ← a.literal
          let y ← b.literal
          pure (.predicate (if cmp == "eq" then x == y else x != y))
        else if cmp == "slt" then some (.mask a b)
        else none
      | "arith.select", [.predicate p, a, b] => some (if p then a else b)
      | "arith.select", [m@(.mask _ _), a, b] => some (.selected m a b)
      | "tt.addptr", [.pointer port a, .index b] =>
        if contains ty "!tt.ptr<bf16>" then some (.pointer port (a.add b)) else none
      | "tt.addptr", [.pointer port a] => do
        let [_, b] := args.splitOn ", " | none
        let .index b ← operand values b | none
        if contains rhs "!tt.ptr<bf16>" then some (.pointer port (a.add b)) else none
      | "tt.load", [p@(.pointer _ _), m@(.mask _ _), .bzero] =>
        if contains ty "!tt.ptr<bf16>" then some (.load p m) else none
      | "arith.extf", [a] =>
        if contains ty "xbf16" && contains ty "xf32" then some (.extend a) else none
      | "arith.truncf", [a] =>
        if contains ty "xf32" && contains ty "xbf16" then some (.narrow a) else none
      | "arith.addf", [a, b] => if contains ty "xf32" then some (.add a b) else none
      | "arith.mulf", [a, b] => if contains ty "xf32" then some (.mul a b) else none
      | "math.fma", [a, b, c] => if direct && contains ty "xf32" then some (.fma a b c) else none
      | _, _ => none
    if op == "tt.load" then
      if value != inputLoad tile plane then return ← none
      loads := loads + 1
    values := values.insert target value
  -- The single-row TTGIR uses different layouts for the reduction and
  -- packing and duplicates exactly the same masked load.
  if (loads == 1 || (!direct && tile == 1 && loads == 2)) && stores == (if onlyPacked then 1 else if scaled then 3 else 2) then some () else none

def recognizes (tile planes first : Nat) (reference candidate : String) : Bool :=
  (do
    let old ← QKVTTIR.normalizeTTIR reference
    let new ← QKVTTIR.normalizeTTIR candidate
    let reduction ← reductionProjection false old
    let fusedReduction ← reductionProjection true new
    pure (recognizesReductionPlanes (List.range planes) first 10 reduction fusedReduction &&
      (List.range planes).all fun lane => (checkPack tile first lane new).isSome)).getD false

private def launch (tile warps : Nat) (ir : String) : Bool :=
  contains ir s!"\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = {warps} : i32, ttg.target = \"cuda:90\"" &&
    contains ir s!"tensor<{tile}x16xf32,"

structure Program where
  rows : Nat
  planes : Nat
  first : Nat
  tile : Nat
  warps : Nat
  referenceTTIR : String
  candidateTTIR : String
  referenceTTGIR : String
  candidateTTGIR : String
  supported : (planes, first) ∈ [(9, 1), (1, 0)] ∧ (tile, warps) ∈ [(1, 2), (8, 2), (32, 4), (128, 8)] ∧ rows % tile = 0
  sourceChecked : recognizes tile planes first referenceTTIR candidateTTIR = true
  layoutChecked : recognizes tile planes first referenceTTGIR candidateTTGIR = true
  launchChecked : (launch tile warps referenceTTGIR && launch tile warps candidateTTGIR) = true

def Program.check (rows planes first tile warps : Nat)
    (referenceTTIR candidateTTIR referenceTTGIR candidateTTGIR : String) : Option Program := do
  if shapes : (planes, first) ∈ [(9, 1), (1, 0)] ∧
      (tile, warps) ∈ [(1, 2), (8, 2), (32, 4), (128, 8)] ∧ rows % tile = 0 then
    if source : recognizes tile planes first referenceTTIR candidateTTIR then
      if layout : recognizes tile planes first referenceTTGIR candidateTTGIR then
        if schedule : launch tile warps referenceTTGIR && launch tile warps candidateTTGIR then
          return ⟨rows, planes, first, tile, warps, referenceTTIR, candidateTTIR, referenceTTGIR,
            candidateTTGIR, shapes, source, layout, schedule⟩
  none

structure SingleIR where
  group : String
  columnZero : String
  fused : String

def recognizesSingle (tile : Nat) (ir : SingleIR) : Bool :=
  (do
    let originalGroup ← QKVTTIR.normalizeTTIR ir.group
    let originalZero ← QKVTTIR.normalizeTTIR ir.columnZero
    let new ← QKVTTIR.normalizeTTIR ir.fused
    let group ← reductionProjection false originalGroup
    let columnZero ← reductionProjection false originalZero
    let reduction ← reductionProjection true new
    pure (recognizesReductionPlanes ((List.range 9).map (· + 1)) 0 10 group reduction &&
      recognizesReductionPlanes [0] 0 10 columnZero reduction &&
      (List.range 10).all fun plane => (checkPack tile 0 plane new).isSome)).getD false

structure SingleProgram where
  rows : Nat
  tile : Nat
  warps : Nat
  ttir : SingleIR
  ttgir : SingleIR
  supported : (tile, warps) ∈ [(1, 2), (8, 2), (32, 4), (128, 8)] ∧ rows % tile = 0
  sourceChecked : recognizesSingle tile ttir = true
  layoutChecked : recognizesSingle tile ttgir = true
  launchChecked : (launch tile warps ttgir.group && launch tile warps ttgir.columnZero &&
    launch tile warps ttgir.fused) = true

def SingleProgram.check (rows tile warps : Nat) (ttir ttgir : SingleIR) : Option SingleProgram := do
  if supported : (tile, warps) ∈ [(1, 2), (8, 2), (32, 4), (128, 8)] ∧ rows % tile = 0 then
    if source : recognizesSingle tile ttir then
      if layout : recognizesSingle tile ttgir then
        if schedule : launch tile warps ttgir.group && launch tile warps ttgir.columnZero &&
            launch tile warps ttgir.fused then
          return ⟨rows, tile, warps, ttir, ttgir, supported, source, layout, schedule⟩
  none

private def oneReduction (lines : List String) : Option (List String) := do
  let chunks ← instructions lines
  let [reduction] := chunks.filter fun chunk => (definition chunk.head!).2.startsWith "\"tt.reduce\"(" | none
  if reduction.length != 5 || !(contains reduction.head! "<{axis = 1 : i32}>") then none
  else some (canonicalNames (reduction.map (inlineLayouts (layoutNames lines))))

def directTile (tile : Nat) : Nat := if tile == 1 then 4 else tile

private def singleRowLayout : String :=
  "#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [0, 1]}>"

private def fourRowLayout : String :=
  "#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>"

private def matchesScaledReduction (tile : Nat) (old new : List String) (reference candidate : List String) : Bool :=
  if tile == 1 then
    let retiled := reference.map fun line =>
      ((line.replace singleRowLayout fourRowLayout).replace "tensor<1x16xf32" "tensor<4x16xf32").replace
        "tensor<1xf32" "tensor<4xf32"
    old.contains ("#blocked = " ++ singleRowLayout) && new.contains ("#blocked = " ++ fourRowLayout) &&
      retiled == candidate
  else reference == candidate

private def recognizesDirectRowStores (tile : Nat) (planes : List Nat) (first : Nat)
    (reference candidate : String) (scaled : Bool) : Bool :=
  (do
    let old ← QKVTTIR.normalizeTTIR reference
    let new ← QKVTTIR.normalizeTTIR candidate
    let oldTree ← oneReduction old
    let newTree ← oneReduction new
    pure (matchesScaledReduction tile old new oldTree newTree &&
      planes.all fun plane => (checkPack (directTile tile) first plane new true scaled).isSome)).getD false

private def sameReductionAndPorts (planes first tile : Nat) (group zero fused : String) (scaled : Bool) : Bool :=
  if planes == 10 then
    recognizesDirectRowStores tile [0] 0 zero fused scaled &&
      recognizesDirectRowStores tile ((List.range 9).map (· + 1)) 0 group fused scaled
  else if planes == 9 then
    recognizesDirectRowStores tile (List.range 9) first group fused scaled
  else recognizesDirectRowStores tile [0] first zero fused scaled

private def declaredPorts (ir : String) (scaled : Bool) : Bool :=
  (do
    let lines ← QKVTTIR.normalizeTTIR ir
    let [header] := lines.filter (fun (line : String) => line.startsWith "tt.func public @") | none
    let ports := (List.range 10).map (fun i => s!"IN{i}") ++ ["OUT", "PACK"] ++ (if scaled then ["SCALED"] else [])
    let parameters := ports.map (fun p => "%" ++ p ++ ": !tt.ptr<bf16> {tt.divisibility = 16 : i32}")
    pure (header.endsWith ("(" ++ String.intercalate ", " parameters ++ ") attributes {noinline = false} {"))).getD false

private def ptxOperands (opcode line : String) : Option (List String) := do
  let start := opcode ++ " "
  if !line.startsWith start || !line.endsWith ";" then return ← none
  pure (((line.drop start.length).dropEnd 1).toString.splitOn ", ")

private def butterfly (code : List String) (input : Option String := none)
    (distances : List Nat := MuddFourRowReduction.shuffleDistances) : Bool :=
  let checked : Option Bool := do
    let steps := code.filter fun (line : String) => line.startsWith "shfl." || line.startsWith "add.f32 "
    if steps.length != 2 * distances.length then return ← none
    let mut prior := input
    for (distance, index) in distances.zipIdx do
      let i := 2 * index
      let [shuffled, input, shift, "31", "-1"] := (ptxOperands "shfl.sync.bfly.b32" steps[i]!).getD []
        | return ← none
      let [output, lhs, rhs] := (ptxOperands "add.f32" steps[i + 1]!).getD [] | return ← none
      if shift != toString distance || lhs != input || rhs != shuffled ||
          prior.any (· != input) then return ← none
      prior := some output
    some true
  checked.getD false

private def findWithInputs (code : List String) (opcode : String) (inputs : List String) : Option String := do
  let [dest :: _] := (code.filterMap (ptxOperands opcode)).filter (fun args => args.drop 1 == inputs) | none
  pure dest

private def fourRowCoordinates (code : List String) (scaled : Bool) : Bool :=
  (do
    let cta ← findWithInputs code "mov.u32" ["%ctaid.x"]
    let thread ← findWithInputs code "mov.u32" ["%tid.x"]
    let base ← findWithInputs code "shl.b32" [cta, "2"]
    let inner ← findWithInputs code "bfe.u32" [thread, "4", "2"]
    let row ← findWithInputs code "or.b32" [inner, base]
    let col ← findWithInputs code "and.b32" [thread, "15"]
    let _ ← findWithInputs code "mad.lo.s32" [row, "12", col]
    let live ← findWithInputs code "setp.lt.u32" [col, "12"]
    let elected ← findWithInputs code "setp.eq.b32" [col, "0"]
    let stores := fun predicate =>
      (code.filter (fun (line : String) => line.startsWith ("@" ++ predicate ++ " st.global.b16 "))).length
    pure (code.contains ".reqntid 64" && stores live == 1 && stores elected == (if scaled then 2 else 1) &&
      code.all (fun line => !line.startsWith "bar." && !line.startsWith "barrier."))).getD false

private def mixedRegions (ir : List String) :
    Option (List String × String × List String × List String) := do
  let branch := ir.findIdx (fun line => line.startsWith "scf.if ")
  let header := ir.take branch
  let [condition, "{"] := ((ir[branch]!.drop 7).toString.splitOn " ") | none
  let [compare] := header.filter (fun line => (definition line).1 == condition) | none
  let body := ir.drop (branch + 1)
  let fork := body.idxOf "} else {"
  let first := body.take fork
  let rest := body.drop (fork + 1)
  let last := rest.findIdx (· == "}")
  let second := rest.take last
  if first.isEmpty || second.isEmpty || rest.drop last != ["}", "tt.return", "}", "}"] ||
      (first ++ second).any (·.startsWith "scf.") then return ← none
  pure (header.filter (· != compare), (definition compare).2, first, second)

private def mixedRegion (ir : List String) (zero : Bool) : Option (List String) := do
  let (header, compare, first, second) ← mixedRegions ir
  let ["arith.cmpi eq", column, "0 : i32"] := compare.splitOn ", " | none
  if !header.contains s!"{column} = tt.get_program_id y : i32" then return ← none
  pure (header ++ (if zero then first else second) ++ ["tt.return", "}", "}"])

private def irOutput (code : List String) (body : String) : Option String := do
  let [(name, _)] := (code.map definition).filter (fun item => item.2 == body) | none
  if name.isEmpty then none else some name

/-- Verify the physical grid coordinates before checking each branch in its
logical (row block, column) coordinates. -/
private def bandedRegions (ir : List String) : Option (List String × List String) := do
  let (header, compare, first, second) ← mixedRegions ir
  let ["arith.cmpi slt", band, "8 : i32"] := compare.splitOn ", " | none
  if !header.contains s!"{band} = tt.get_program_id y : i32" then return ← none
  let block ← irOutput header "tt.get_program_id x : i32"
  let start ← irOutput header s!"arith.muli {block}, 128 : i32"
  let [bandShift, zeroStart] := first.take 2 | none
  let (shift, shiftBody) := definition bandShift
  let (zeroRow, zeroBody) := definition zeroStart
  let (column, columnBody) := definition second.head!
  if shift.isEmpty || zeroRow.isEmpty || column.isEmpty ||
      shiftBody != s!"arith.muli {band}, 16 : i32" ||
      zeroBody != s!"arith.addi {start}, {shift} : i32" ||
      columnBody != s!"arith.subi {band}, 7 : i32" then return ← none
  let common := header.filter (fun line => (definition line).1 != band)
  let zeroHeader := common.filter (fun line => !([block, start].contains (definition line).1))
  if ir.any (contains · "%banded_block") then return ← none
  let footer := ["tt.return", "}", "}"]
  pure (zeroHeader ++ ["%banded_block = tt.get_program_id x : i32",
    s!"{zeroRow} = arith.muli %banded_block, 16 : i32"] ++ first.drop 2 ++ footer,
    common ++ [s!"{column} = tt.get_program_id y : i32"] ++ second.drop 1 ++ footer)

def mixedBlock (group zeroTile : Nat) : Nat :=
  if group == 1 && zeroTile == 128 then 64
  else if group == 128 || zeroTile == 128 then 128
  else if group == 32 || zeroTile == 32 then 32 else 8

def scheduleWarps (layout : Nat) : Nat :=
  if layout == 128 then 8 else if layout == 32 then 4 else 2

def mixedWarps (group zeroTile : Nat) : Nat := max (scheduleWarps group) (scheduleWarps zeroTile)

private def valuesPerLane (layout : Nat) : Nat := if layout == 1 then 1 else if layout == 8 then 2 else 4

private def mixedLayout (layout warps : Nat) : String :=
  let width := valuesPerLane layout
  "#ttg.blocked<{" ++
    s!"sizePerThread = [1, {width}], threadsPerWarp = [{2 * width}, {16 / width}], warpsPerCTA = [{warps}, 1], order = [1, 0]" ++ "}>"

private def matchesMixedPlane (layout tile warps plane : Nat) (original : String)
    (region : List String) (scaled : Bool) : Bool :=
  (do
    let old ← QKVTTIR.normalizeTTIR original
    let oldTree ← oneReduction old
    let newTree ← oneReduction region
    let oldLayout := if layout == 1 then singleRowLayout else mixedLayout layout (scheduleWarps layout)
    let expected := oldTree.map fun (line : String) =>
      ((line.replace oldLayout (mixedLayout layout warps)).replace
        s!"tensor<{layout}x16xf32" s!"tensor<{tile}x16xf32").replace
        s!"tensor<{layout}xf32" s!"tensor<{tile}xf32"
    pure (old.any (fun (line : String) => line.startsWith "#blocked" && line.endsWith (" = " ++ oldLayout)) &&
      expected == newTree &&
      (checkPack tile 0 plane region true scaled).isSome)).getD false

private def usesBands (group zeroTile : Nat) : Bool := group == 128 && zeroTile == 1

private def recognizedSchedules (group zeroTile : Nat) : Bool :=
  let tiles := [1, 8, 32, 128]
  group != zeroTile && tiles.contains group && tiles.contains zeroTile

def checkMixed (groupTile zeroTile : Nat) (group zero fused : String) (scaled : Bool) : Bool :=
  let tile := mixedBlock groupTile zeroTile
  let warps := mixedWarps groupTile zeroTile
  (do
    if !recognizedSchedules groupTile zeroTile then return false
    let ir ← QKVTTIR.normalizeTTIR fused
    let banded := usesBands groupTile zeroTile
    let (zeroRegion, groupRegion) ← if banded then bandedRegions ir
      else (·, ·) <$> mixedRegion ir true <*> mixedRegion ir false
    pure (declaredPorts fused scaled && launch tile warps fused &&
      launch groupTile (scheduleWarps groupTile) group && launch zeroTile (scheduleWarps zeroTile) zero &&
      matchesMixedPlane zeroTile (if banded then 16 else tile) warps 0 zero zeroRegion scaled &&
      ((List.range 9).map (· + 1)).all (fun plane =>
        matchesMixedPlane groupTile tile warps plane group groupRegion scaled))).getD false

private def compiledBranches (warps : Nat) (ptx : String) : Option (List String × List String × List String) := do
  let code := QKVTTIR.normalizePTX ptx
  let pid ← findWithInputs code "mov.u32" ["%ctaid.y"]
  let guard ← findWithInputs code "setp.ne.b32" [pid, "0"]
  let branches := code.filter (fun line => line.startsWith "bra" || (line.startsWith "@" && contains line " bra"))
  let body := "@" ++ guard ++ " bra "
  let [jumpGroup, jumpZero, exitGroup] := branches | none
  if !jumpGroup.startsWith body || !jumpZero.startsWith "bra.uni " || !exitGroup.startsWith "bra.uni " then
    return ← none
  if !(code.idxOf s!"mov.u32 {pid}, %ctaid.y;" < code.idxOf s!"setp.ne.b32 {guard}, {pid}, 0;" &&
      code.idxOf s!"setp.ne.b32 {guard}, {pid}, 0;" < code.idxOf jumpGroup) then return ← none
  let label := fun (line : String) (start : String) => ((line.drop start.length).dropEnd 1).toString
  let groupTarget := label jumpGroup ("@" ++ guard ++ " bra ")
  let zeroTarget := label jumpZero "bra.uni "
  let exitTarget := label exitGroup "bra.uni "
  let group := code.idxOf (groupTarget ++ ":")
  let zeroLabel := code.idxOf (zeroTarget ++ ":")
  let done := code.idxOf (exitTarget ++ ":")
  if !(code.idxOf jumpGroup < code.idxOf jumpZero && code.idxOf jumpZero < group &&
      group < code.idxOf exitGroup && code.idxOf exitGroup < zeroLabel && zeroLabel < done && done < code.length) ||
      !code.contains s!".reqntid {32 * warps}" then return ← none
  if (code.drop (done + 1)).head? != some "ret;" then return ← none
  pure (code.take (code.idxOf jumpGroup), (code.drop (group + 1)).take (code.idxOf exitGroup - group - 1),
    (code.drop (zeroLabel + 1)).take (done - zeroLabel - 1))

private def assemblyDefinition (line : String) : Option (String × String × List String) := do
  let opcode := (line.splitOn " ").head!
  let destination :: arguments ← ptxOperands opcode line | none
  pure (destination, opcode, arguments)

private def parameterRegister (code : List String) (kernel : String) (index : Nat) : Option String :=
  findWithInputs code "ld.param.b64" [s!"[{kernel}_param_{index}]"]

private def wordAddress (code : List String) (port : String) (locations : List (String × Index))
    (rowGroup tile : Nat) (address : String) (delta : Index) : Bool :=
  let rec evaluate (fuel : Nat) (reg : String) : Option (Index × Nat) := do
    if reg == port then return ({}, 1)
    if let some coordinate := locations.lookup reg then return (coordinate, 0)
    if let some literal := reg.toNat? then return ({bias := literal}, 0)
    let fuel + 1 := fuel | none
    let [(_, op, args)] := (code.filterMap assemblyDefinition).filter (fun d => d.1 == reg) | none
    match op, args with
    | "or.b32", [base, constant] =>
      let offset ← constant.toNat?
      -- Each register group starts after all rows assigned to the preceding groups.
      if offset % rowGroup == 0 && offset < tile &&
          locations.lookup base == some ({row := 1} : Index) then some (⟨0, 1, 0, offset⟩, 0) else none
    | "add.s32", [lhs, rhs] | "add.s64", [lhs, rhs] =>
      let (l, lp) ← evaluate fuel lhs
      let (r, rp) ← evaluate fuel rhs
      some (l.add r, lp + rp)
    | "mul.lo.s32", [source, n] | "mul.wide.s32", [source, n] | "mul.wide.u32", [source, n] =>
      let (word, 0) ← evaluate fuel source | none
      pure (word.scale (← n.toNat?), 0)
    | "mad.lo.s32", [source, n, addend] | "mad.wide.s32", [source, n, addend]
    | "mad.wide.u32", [source, n, addend] =>
      let (word, 0) ← evaluate fuel source | none
      let (offset, ports) ← evaluate fuel addend
      pure ((word.scale (← n.toNat?)).add offset, ports)
    | _, _ => none
  evaluate 8 address == some (delta.scale 2, 1)

private def registerList (arg : String) : List String := arg.splitOn ", "

private def globalLoad (line : String) : Option (String × String × String × List String) := do
  let [instruction, rest] := line.splitOn " { " | none
  let [predicate, opcode] := instruction.splitOn " " | none
  let [values, address] := rest.splitOn " }, [ " | none
  if !predicate.startsWith "@" || !opcode.startsWith "ld.global." || !address.endsWith " + 0 ];" then none
  else some (predicate, opcode, (address.dropEnd 7).toString, registerList values)

private def globalStore (line : String) : Option (String × String × String × List String) := do
  let [instruction, rest] := line.splitOn " [ " | none
  let [predicate, opcode] := instruction.splitOn " " | none
  let [address, values] := rest.splitOn " + 0 ], { " | none
  if !predicate.startsWith "@" || !opcode.startsWith "st.global." || !values.endsWith " };" then none
  else some (predicate, opcode, address, registerList (values.dropEnd 3).toString)

private inductive PlaneRoute where
  | port (index : Nat)
  | choose (mask : Nat) (set clear : PlaneRoute)

private def PlaneRoute.atColumn (column : BitVec 32) : PlaneRoute → Nat
  | .port index => index
  | .choose mask set clear =>
    if column &&& BitVec.ofNat 32 mask != 0 then set.atColumn column else clear.atColumn column

private def gradientPlaneRoute (skipZero parityLast : Bool) : PlaneRoute :=
  let port := PlaneRoute.port
  let choose := PlaneRoute.choose
  if parityLast then
    let byParity := fun offset => choose 8 (port (8 + offset))
      (choose 4 (choose 2 (port (6 + offset)) (port (4 + offset)))
        (choose 2 (port (2 + offset)) (port offset)))
    choose 1 (byParity 1) (byParity 0)
  else
    let pair := fun offset => choose 1 (port (offset + 1)) (port offset)
    let first := if skipZero then port 1 else pair 0
    choose 4 (choose 2 (pair 6) (pair 4)) (choose 8 (pair 8) (choose 2 (pair 2) first))

/-- Either schedule reads the column it was launched for. The mixed
schedule handles column zero on its own launch path. -/
theorem gradient_plane_route (column : Fin 10) (skipZero parityLast : Bool) :
    (gradientPlaneRoute skipZero parityLast).atColumn (BitVec.ofNat 32 column.val) =
      (if skipZero && !parityLast && column.val == 0 then 1 else column.val) := by
  revert column skipZero parityLast
  decide +kernel

#axiomsClean gradient_plane_route

private def PlaneRoute.parameter (code : List String) (kernel plane : String) :
    PlaneRoute → Option String
  | .port index => parameterRegister code kernel index
  | .choose mask set clear => do
    let yes ← set.parameter code kernel plane
    let no ← clear.parameter code kernel plane
    let masked ← findWithInputs code "and.b32" [plane, toString mask]
    (do
      let test ← findWithInputs code "setp.ne.b32" [masked, "0"]
      findWithInputs code "selp.b64" [yes, no, test]) <|> (do
      let test ← findWithInputs code "setp.eq.b32" [masked, "0"]
      findWithInputs code "selp.b64" [no, yes, test])

private def bitSelectedPlane (code : List String) (kernel plane : String)
    (skipZero parityLast : Bool) : Option String :=
  (gradientPlaneRoute skipZero parityLast).parameter code kernel plane

private def splitPackedInput (code : List String) (register : String) : Option (List String) := do
  let suffix := "}, " ++ register ++ ";"
  let [split] := code.filter (fun (line : String) => line.startsWith "mov.b32 {" && line.endsWith suffix) | none
  let [lo, hi] := (((split.drop 9).dropEnd suffix.length).toString).splitOn ", " | none
  some [lo, hi]

private def initialPair (code : List String) (a b : String) : Option String :=
  findWithInputs code "add.f32" [a, b] <|> findWithInputs code "add.f32" [b, a]

open Triton.FP in
theorem four_value_pair_swap (a b c d : Value .fp32) :
    add Config.ieeeValue (add Config.ieeeValue (add Config.ieeeValue a b) c) d =
      add Config.ieeeValue (add Config.ieeeValue (add Config.ieeeValue b a) c) d := by
  rw [ScalarOpsLaws.add_comm Config.ieeeValue a b]

#axiomsClean four_value_pair_swap

private def scalarInputs (code : List String) (width : Nat) (registers : List String) :
    Option (List String × String) := do
  if registers.length != max 1 (width / 2) then return ← none
  let raw ← if width == 1 then some registers else List.flatten <$> registers.mapM (splitPackedInput code)
  let floats ← raw.mapM (fun word => findWithInputs code "cvt.f32.bf16" [word])
  let total ← match floats with
    | [a] => some a
    | [a, b] => initialPair code a b
    | [a, b, c, d] => do
      let first ← initialPair code a b
      let second ← findWithInputs code "add.f32" [first, c]
      findWithInputs code "add.f32" [second, d]
    | _ => none
  pure (floats, total)

private def packedOutputs (code : List String) (floats : List String) : Option (List String) := do
  let products ← floats.mapM (fun value => findWithInputs code "fma.rn.f32" [value, "0f3DCCCCCD", "0f00000000"])
  match products with
  | [a] => List.singleton <$> findWithInputs code "cvt.rn.bf16.f32" [a]
  | [a, b] => List.singleton <$> findWithInputs code "cvt.rn.bf16x2.f32" [b, a]
  | [a, b, c, d] =>
    List.cons <$> findWithInputs code "cvt.rn.bf16x2.f32" [b, a] <*>
      (List.singleton <$> findWithInputs code "cvt.rn.bf16x2.f32" [d, c])
  | _ => none

private def vectorSuffix (width : Nat) : String :=
  if width == 1 then "b16" else if width == 2 then "b32" else "v2.b32"

private def verifiesGradientStores (header code : List String) (scaled : Bool)
    (tile warps layout columnStride : Nat) (kernel plane blockStart input : String)
    (bandStart : Option String := none) : Bool :=
  (do
    let width := valuesPerLane layout
    let groupRows := 2 * width * warps
    let rowGroups := tile / groupRows
    let lanes := 16 / width
    let suffix := vectorSuffix width
    let distance := (MuddFourRowReduction.shuffleDistances.drop width.log2)
    let tid ← findWithInputs code "mov.u32" ["%tid.x"]
    let extracted := findWithInputs code "bfe.u32" [tid, toString lanes.log2, toString groupRows.log2]
    let shifted := if lanes * groupRows == 32 * warps then
        findWithInputs code "shr.u32" [tid, toString lanes.log2]
      else none
    let bases := ([extracted, shifted].filterMap id).filterMap fun low => do
      let localRow ← match bandStart with
        | none => some low
        | some band => findWithInputs code "or.b32" [low, band]
      findWithInputs code "or.b32" [localRow, blockStart]
    if bases.isEmpty then return false
    let scaledLane ← if width == 1 then some tid
      else findWithInputs code "shl.b32" [tid, toString width.log2]
    let lane ← findWithInputs code "and.b32" [scaledLane, toString (16 - width)]
    let elected ← if width == 1 then some lane else findWithInputs code "and.b32" [tid, toString (lanes - 1)]
    let live ← findWithInputs code (if width == 4 then "setp.ne.b32" else "setp.lt.u32") [lane, "12"]
    let leader ← findWithInputs code "setp.eq.b32" [elected, "0"]
    let output ← parameterRegister (header ++ code) kernel 10
    let packed ← parameterRegister (header ++ code) kernel 11
    let loc : List (String × Index) := (bases.map fun row => (row, ({row := 1} : Index))) ++
      [(lane, ({col := 1} : Index)), (plane, ({cta := 1} : Index))]
    let parsed ← (code.filter (contains · "st.global.")).mapM globalStore
    let loads ← (code.filter (contains · "ld.global.")).mapM globalLoad
    if parsed.length != rowGroups * (if scaled then 3 else 2) || loads.length != rowGroups then return false
    let require := fun (predicate opcode port : String) (values : List String) (offset : Index) =>
      (parsed.filter fun (test, op, address, src) =>
        test == ("@" ++ predicate) && op == opcode && src == values &&
          wordAddress (header ++ code) port loc groupRows tile address offset).length == 1
    let steps := code.filter (fun (line : String) => line.startsWith "shfl." || line.startsWith "add.f32 ")
    let shuffleSteps := 2 * distance.length
    let rowSteps := width - 1 + shuffleSteps
    if steps.length != rowGroups * rowSteps then return false
    for i in List.range rowGroups do
      let rowOffset := groupRows * i
      let (predicate, opcode, loadAddr, bf16) := loads[i]!
      if predicate != ("@" ++ live) || opcode != ("ld.global." ++ suffix) ||
          !wordAddress code input loc groupRows tile loadAddr ⟨0, 12, 1, 12 * rowOffset⟩ then return false
      let (floats, sumInput) ← scalarInputs code width bf16
      let masked ← findWithInputs code "selp.f32" [sumInput, "0f00000000", live]
      let reduction := (steps.drop (rowSteps * i + width - 1)).take shuffleSteps
      if !butterfly reduction (some masked) distance then return false
      let reduced := ((ptxOperands "add.f32" reduction.getLast!).getD []).head!
      let rounded ← findWithInputs code "cvt.rn.bf16.f32" [reduced]
      let writeColumn : Index := ⟨columnStride, 10, 0, 10 * rowOffset⟩
      if !require leader "st.global.b16" output [rounded] writeColumn then return false
      if scaled then
        let scalePort ← parameterRegister (header ++ code) kernel 12
        let widened ← findWithInputs code "cvt.f32.bf16" [rounded]
        let multiplied ← findWithInputs code "mul.f32" [widened, "0f3DCCCCCD"]
        let cast ← findWithInputs code "cvt.rn.bf16.f32" [multiplied]
        if !require leader "st.global.b16" scalePort [cast] writeColumn then return false
      let packedResult ← packedOutputs code floats
      if !require live ("st.global." ++ suffix) packed packedResult
          ⟨12 * columnStride, 120, 1, 120 * rowOffset⟩ then return false
    pure true).getD false

private def compiledBandedBranches (ptx : String) : Option (List String × List String × List String) := do
  let code := QKVTTIR.normalizePTX ptx
  let band ← findWithInputs code "mov.u32" ["%ctaid.y"]
  let guard ← findWithInputs code "setp.gt.u32" [band, "7"]
  let branches := code.filter (fun line => line.startsWith "bra" || (line.startsWith "@" && contains line " bra"))
  let [jumpGroup, exitZero] := branches | none
  let conditional := "@" ++ guard ++ " bra "
  if !jumpGroup.startsWith conditional || !exitZero.startsWith "bra.uni " ||
      !code.contains ".reqntid 256" ||
      !(code.idxOf s!"mov.u32 {band}, %ctaid.y;" < code.idxOf s!"setp.gt.u32 {guard}, {band}, 7;" &&
        code.idxOf s!"setp.gt.u32 {guard}, {band}, 7;" < code.idxOf jumpGroup) then return ← none
  let target := fun (line : String) (start : Nat) => ((line.drop start).dropEnd 1).toString ++ ":"
  let split := code.idxOf (target jumpGroup conditional.length)
  let done := code.idxOf (target exitZero "bra.uni ".length)
  let entry := code.idxOf jumpGroup
  let exit := code.idxOf exitZero
  if !(entry < exit && exit < split && split < done && done < code.length) ||
      (code.drop (done + 1)).head? != some "ret;" then return ← none
  pure (code.take entry, (code.drop (split + 1)).take (done - split - 1),
    (code.drop (entry + 1)).take (exit - entry - 1))

private def blockOrigin (code : List String) (tile : Nat) : Option String := do
  let cta ← findWithInputs code "mov.u32" ["%ctaid.x"]
  findWithInputs code "shl.b32" [cta, toString tile.log2]

private def verifiesMixedPTX (groupTile zeroTile : Nat) (scaled : Bool) (ptx : String) : Bool :=
  let tile := mixedBlock groupTile zeroTile
  let warps := mixedWarps groupTile zeroTile
  (do
    let banded := usesBands groupTile zeroTile
    let (header, group, zero) ← if banded then compiledBandedBranches ptx else compiledBranches warps ptx
    let pid ← findWithInputs header "mov.u32" ["%ctaid.y"]
    let kernel := if banded then "mudd_banded_gradients" else "mudd_mixed_gradients"
    let groupColumn ← if banded then findWithInputs group "add.s32" [pid, "-7"] else some pid
    let groupStart ← blockOrigin (if banded then header else group) tile
    let zeroStart ← if banded then some groupStart else blockOrigin zero tile
    let bandStart ← if banded then some <$> findWithInputs zero "shl.b32" [pid, "4"] else some none
    let zeroTileSize := if banded then 16 else tile
    let input ← bitSelectedPlane group kernel groupColumn true false
    let firstInput ← parameterRegister zero kernel 0
    pure (verifiesGradientStores header group scaled tile warps groupTile 1 kernel groupColumn groupStart input &&
      verifiesGradientStores header zero scaled zeroTileSize warps zeroTile 0 kernel pid zeroStart firstInput bandStart &&
      (group ++ zero).all (fun line => !line.startsWith "bar." && !line.startsWith "barrier."))).getD false

private def warpLoadLayout : String :=
  "#ttg.linear<{register = [[0, 1], [0, 2], [8, 0]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], " ++
    "warp = [[16, 0], [32, 0], [64, 0]], block = []}>"

private def warpStoreRow (thread : Fin 256) : Nat :=
  16 * (thread.val / 32) + thread.val / 4 % 8 + 8 * (thread.val % 2)

private def warpWriterId (row : Nat) : Nat :=
  32 * (row / 16) + 4 * (row % 8) + row % 16 / 8

private def warpWriter (row : Fin 128) : Fin 256 :=
  ⟨warpWriterId row.val, by have bound := row.isLt; unfold warpWriterId; omega⟩

private theorem warp_writer_owns (row : Fin 128) :
    (warpWriter row).val % 4 < 2 ∧ warpStoreRow (warpWriter row) = row.val := by
  revert row
  decide +kernel

private theorem warp_writer_inverse (thread : Fin 256) (elected : thread.val % 4 < 2) :
    warpWriterId (warpStoreRow thread) = thread.val := by
  revert thread
  decide +kernel

theorem warp_store_unique (row : Fin 128) :
    ∃! thread : Fin 256, thread.val % 4 < 2 ∧ warpStoreRow thread = row.val := by
  refine ⟨warpWriter row, warp_writer_owns row, ?_⟩
  intro other owns
  apply Fin.ext
  change other.val = warpWriterId row.val
  rw [← owns.2]
  exact (warp_writer_inverse other owns.1).symm

theorem warp_thread_coordinates (tid : Fin 256) :
    let word := BitVec.ofNat 32 tid.val
    let warpRows := (word >>> 1) &&& 112
    let laneRows := (word >>> 2) &&& 7
    let subgroup := word &&& 3
    let row := warpRows ||| laneRows
    (tid.val % 4 < 2 → (if subgroup == 1 then row ||| 8 else row).toNat = warpStoreRow tid) ∧
    (subgroup <<< 2).toNat = 4 * (tid.val % 4) ∧
    (subgroup != 3) = decide (4 * (tid.val % 4) < 12) ∧
    (subgroup.toNat < 2 ↔ tid.val % 4 < 2) ∧
    (subgroup == 1) = decide (tid.val % 4 = 1) := by
  revert tid
  decide +kernel

theorem warp_butterfly_keeps_rows (thread : Fin 256) :
    (thread.val ^^^ 2) / 4 = thread.val / 4 ∧
    (thread.val ^^^ 1) / 4 = thread.val / 4 ∧
    (thread.val ^^^ 2) % 4 = ((thread.val % 4) ^^^ 2) ∧
    (thread.val ^^^ 1) % 4 = ((thread.val % 4) ^^^ 1) := by
  revert thread
  decide +kernel

open Triton.FP in
private def warpButterfly (subtotals : Fin 4 → Value .fp32) (lane : Fin 4) : Value .fp32 :=
  let pair := fun (i : Fin 4) =>
    add Config.ieeeValue (subtotals i) (subtotals (Fin.ofNat 4 (i.val ^^^ 2)))
  add Config.ieeeValue (pair lane) (pair (Fin.ofNat 4 (lane.val ^^^ 1)))

open Triton.FP in
/-- Lane one exchanges the outer add operands without changing the reduction tree. -/
theorem warp_elected_butterfly (subtotals : Fin 4 → Value .fp32) (lane : Fin 4)
    (writer : lane.val < 2) : warpButterfly subtotals lane = warpButterfly subtotals 0 := by
  have elected : lane = 0 ∨ lane = 1 := by
    by_cases zero : lane.val = 0
    · exact Or.inl (Fin.ext zero)
    · exact Or.inr (Fin.ext (by change lane.val = 1; omega))
  rcases elected with rfl | rfl
  · rfl
  · exact ScalarOpsLaws.add_comm Config.ieeeValue _ _

open Triton.FP in
theorem warp_store_reduction_eq (subtotals : Fin 128 → Fin 4 → Value .fp32) (row : Fin 128) :
    ∃! writer : Fin 256, writer.val % 4 < 2 ∧ warpStoreRow writer = row.val ∧
      warpButterfly (subtotals row) (Fin.ofNat 4 writer.val) = warpButterfly (subtotals row) 0 := by
  obtain ⟨writer, owns, unique⟩ := warp_store_unique row
  refine ⟨writer, ⟨owns.1, owns.2, warp_elected_butterfly _ _ owns.1⟩, ?_⟩
  intro other result
  exact unique other ⟨result.1, result.2.1⟩

#axiomsClean warp_store_unique
#axiomsClean warp_thread_coordinates
#axiomsClean warp_butterfly_keeps_rows
#axiomsClean warp_elected_butterfly
#axiomsClean warp_store_reduction_eq

private def warpSelectIR (word src stem : String) : List String :=
  let slice (dim : Nat) (parent : String) := "#ttg.slice<{dim = " ++ toString dim ++ ", parent = " ++ parent ++ "}>"
  let tensor (shape layout : String) := s!"tensor<{shape}x{word}, {layout}>"
  let old := tensor "8x8" (slice 2 "#linear2")
  let lanes := tensor "8x8" (slice 2 "#blocked")
  let single := tensor "8x8x1" "#blocked"
  let all := tensor "8x8x4" "#blocked"
  let pairs := tensor "8x2x8" "#linear1"
  let groups := tensor "8x8x2" "#linear2"
  let p := "%" ++ stem
  [s!"{p}Pair = tt.reshape {src} : {tensor "128" (slice 1 "#linear")} -> {pairs}",
   s!"{p}Groups = tt.trans {p}Pair " ++ "{order = array<i32: 0, 2, 1>}" ++ s!" : {pairs} -> {groups}",
   s!"{p}Left, {p}Right = tt.split {p}Groups : {groups} -> {old}",
   s!"{p}First = ttg.convert_layout {p}Left : {old} -> {lanes}",
   s!"{p}Second = ttg.convert_layout {p}Right : {old} -> {lanes}",
   s!"{p}Hi = tt.expand_dims {p}Second " ++ "{axis = 2 : i32}" ++ s!" : {lanes} -> {single}",
   s!"{p}Lo = tt.expand_dims {p}First " ++ "{axis = 2 : i32}" ++ s!" : {lanes} -> {single}"] ++
   (if word == "i32" then ["%choose = tt.broadcast %group : tensor<1x1x4xi1, #blocked> -> tensor<8x8x4xi1, #blocked>"] else []) ++
   [s!"{p}AllHi = tt.broadcast {p}Hi : {single} -> {all}",
    s!"{p}AllLo = tt.broadcast {p}Lo : {single} -> {all}",
    s!"{p} = arith.select %choose, {p}AllHi, {p}AllLo : tensor<8x8x4xi1, #blocked>, {all}"]

private def warpWritebackIR : List String :=
  let lane := "tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>"
  let axis0 := "tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #blocked}>>"
  ["%sub = tt.make_range {end = 4 : i32, start = 0 : i32} : " ++ lane,
   "%axis0 = tt.expand_dims %sub {axis = 0 : i32} : " ++ lane ++ " -> " ++ axis0,
   "%lane = tt.expand_dims %axis0 {axis = 1 : i32} : " ++ axis0 ++ " -> tensor<1x1x4xi32, #blocked>",
   "%group = arith.cmpi eq, %lane, %one : tensor<1x1x4xi32, #blocked>"] ++
  warpSelectIR "i32" "%rows" "row" ++ warpSelectIR "f32" "%sum" "value" ++
  ["%base = tt.addptr %OUT, %plane : !tt.ptr<bf16>, i32",
   "%offset = arith.muli %row, %ten : tensor<8x8x4xi32, #blocked>",
   "%origin = tt.splat %base : !tt.ptr<bf16> -> tensor<8x8x4x!tt.ptr<bf16>, #blocked>",
   "%ptr = tt.addptr %origin, %offset : tensor<8x8x4x!tt.ptr<bf16>, #blocked>, tensor<8x8x4xi32, #blocked>",
   "%rounded = arith.truncf %value : tensor<8x8x4xf32, #blocked> to tensor<8x8x4xbf16, #blocked>",
   "%writers = arith.cmpi slt, %lane, %two : tensor<1x1x4xi32, #blocked>",
   "%active = tt.broadcast %writers : tensor<1x1x4xi1, #blocked> -> tensor<8x8x4xi1, #blocked>",
   "tt.store %ptr, %rounded, %active : tensor<8x8x4x!tt.ptr<bf16>, #blocked>",
   "tt.return", "}", "}"]

private def recognizesWarpRows (group zero fused : String) : Bool :=
  (do
    let oldGroup ← QKVTTIR.normalizeTTIR group
    let oldZero ← QKVTTIR.normalizeTTIR zero
    let code ← QKVTTIR.normalizeTTIR fused
    let baseTree ← oneReduction oldGroup
    let zeroTree ← oneReduction oldZero
    let warpTree ← oneReduction code
    let endReduce := code.findIdx (fun (line : String) => line.startsWith "}) :")
    let (front, writeback) := code.splitAt (endReduce + 1)
    let origin ← irOutput front "tt.get_program_id x : i32"
    let plane ← irOutput front "tt.get_program_id y : i32"
    let block ← irOutput front s!"arith.muli {origin}, 128 : i32"
    let rowType := "tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>"
    let ramp ← irOutput front ("tt.make_range {end = 128 : i32, start = 0 : i32} : " ++ rowType)
    let base ← irOutput front s!"tt.splat {block} : i32 -> {rowType}"
    let rows ← irOutput front s!"arith.addi {base}, {ramp} : {rowType}"
    let [sum] := (front.map definition).filter (fun (p : String × String) => p.2.startsWith "\"tt.reduce\"(") | none
    let one ← irOutput front "arith.constant dense<1> : tensor<1x1x4xi32, #blocked>"
    let ten ← irOutput front "arith.constant dense<10> : tensor<8x8x4xi32, #blocked>"
    let two ← irOutput front "arith.constant dense<2> : tensor<1x1x4xi32, #blocked>"
    let layoutLines := code.filter (fun (line : String) => line.startsWith "#")
    pure (baseTree == zeroTree &&
      warpTree == baseTree.map (fun (line : String) => line.replace (mixedLayout 128 8) warpLoadLayout) &&
      layoutLines == [
        "#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [8, 1, 1], order = [2, 1, 0]}>",
        "#linear = " ++ warpLoadLayout,
        "#linear1 = #ttg.linear<{register = [[0, 1, 0]], lane = [[0, 0, 0], [0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 4]], warp = [[1, 0, 0], [2, 0, 0], [4, 0, 0]], block = []}>",
        "#linear2 = #ttg.linear<{register = [[0, 0, 1]], lane = [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 4, 0]], warp = [[1, 0, 0], [2, 0, 0], [4, 0, 0]], block = []}>"] &&
      canonicalNames (String.intercalate " " [rows, sum.1, plane, "%OUT", one, ten, two] :: writeback) ==
        canonicalNames ("%rows %sum %plane %OUT %one %ten %two" :: warpWritebackIR) &&
      (List.range 10).all fun column => (checkPack 128 0 column front true false true).isSome)).getD false

private def reduceFourLanes (code : List String) (value live : String) : Option String := do
  let mut sum ← findWithInputs code "selp.f32" [value, "0f00000000", live]
  for distance in ["2", "1"] do
    let moved ← findWithInputs code "shfl.sync.bfly.b32" [sum, distance, "31", "-1"]
    sum ← findWithInputs code "add.f32" [sum, moved]
  pure sum

private def recognizesWarpPTX (ptx : String) : Bool :=
  (do
    let code := QKVTTIR.normalizePTX ptx
    if !code.contains ".reqntid 256" || code.any (fun line =>
        line.startsWith "bar." || line.startsWith "barrier." || contains line ".shared." ||
          line.startsWith "bra" || (line.startsWith "@" && contains line " bra")) then return false
    let kernel := "mudd_gradient_pack_warp"
    let plane ← findWithInputs code "mov.u32" ["%ctaid.y"]
    let start ← blockOrigin code 128
    let tid ← findWithInputs code "mov.u32" ["%tid.x"]
    let warpBits ← findWithInputs code "shr.u32" [tid, "1"]
    let warpRow ← findWithInputs code "and.b32" [warpBits, "112"]
    let localRow ← findWithInputs code "bfe.u32" [tid, "2", "3"]
    let offset ← findWithInputs code "or.b32" [warpRow, localRow]
    let firstRow ← findWithInputs code "or.b32" [offset, start]
    let secondRow ← findWithInputs code "or.b32" [firstRow, "8"]
    let subgroup ← findWithInputs code "and.b32" [tid, "3"]
    let lane ← findWithInputs code "shl.b32" [subgroup, "2"]
    let live ← findWithInputs code "setp.ne.b32" [subgroup, "3"]
    let select ← findWithInputs code "setp.eq.b32" [subgroup, "1"]
    let leader ← findWithInputs code "setp.lt.u32" [subgroup, "2"]
    let mergedRow ← findWithInputs code "selp.b32" [secondRow, firstRow, select]
    let input ← bitSelectedPlane code kernel plane false false
    let output ← parameterRegister code kernel 10
    let packed ← parameterRegister code kernel 11
    let loads ← (code.filter (contains · "ld.global.")).mapM globalLoad
    let stores ← (code.filter (contains · "st.global.")).mapM globalStore
    let [firstLoad, secondLoad] := loads | none
    let [rawStore] := stores.filter (fun (item : String × String × String × List String) => item.2.1 == "st.global.b16") | none
    if stores.length != 3 || (code.filter (fun (line : String) => line.startsWith "shfl.")).length != 4 then return false
    let planeIndex : String × Index := (plane, {cta := 1})
    let locations : List (String × Index) := [(firstRow, {row := 1}), (lane, {col := 1}), planeIndex]
    let readAndPack := fun ((pred, opcode, address, words) : String × String × String × List String)
        (rows : Nat) => do
      if pred != "@" ++ live || opcode != "ld.global.v2.b32" ||
          !wordAddress code input locations 8 16 address ⟨0, 12, 1, 12 * rows⟩ then return ← none
      let (floats, subtotal) ← scalarInputs code 4 words
      let values ← packedOutputs code floats
      let sum ← reduceFourLanes code subtotal live
      let validStore := fun ((test, op, addr, data) : String × String × String × List String) =>
        test == "@" ++ live && op == "st.global.v2.b32" && data == values &&
          wordAddress code packed locations 8 16 addr ⟨12, 120, 1, 120 * rows⟩
      if (stores.filter validStore).length != 1 then return ← none
      pure sum
    let first ← readAndPack firstLoad 0
    let second ← readAndPack secondLoad 8
    let merged ← findWithInputs code "selp.f32" [second, first, select]
    let rounded ← findWithInputs code "cvt.rn.bf16.f32" [merged]
    let (pred, _, address, words) := rawStore
    pure (pred == "@" ++ leader && words == [rounded] &&
      wordAddress code output [(mergedRow, {row := 1}), planeIndex] 8 16 address ⟨1, 10, 0, 0⟩)).getD false

structure DirectPTX where
  group : String
  columnZero : String
  fused : String

private def verifiesDirectBitStores (code : List String) (tile : Nat) (scaled : Bool) : Bool :=
  (do
    let kernel := "mudd_gradient_pack_direct"
    let warps := scheduleWarps tile
    if !code.contains s!".reqntid {32 * warps}" || code.any (fun line =>
        line.startsWith "bar." || line.startsWith "barrier." || contains line ".shared." ||
          line.startsWith "bra" || (line.startsWith "@" && contains line " bra")) then return false
    let plane ← findWithInputs code "mov.u32" ["%ctaid.y"]
    let start ← blockOrigin code (directTile tile)
    let input ← bitSelectedPlane code kernel plane false (tile == 8 || tile == 32)
    pure (verifiesGradientStores [] code scaled (directTile tile) warps tile 1 kernel plane start input)).getD false

private def matchesDirectPTX (tile planes : Nat) (ptx : DirectPTX) (scaled : Bool) : Bool :=
  if tile == 128 && !scaled then recognizesWarpPTX ptx.fused
  else
    let code := QKVTTIR.normalizePTX ptx.fused
    let fmas := code.filter (·.startsWith "fma.")
    !fmas.isEmpty && fmas.all (fun line => (ptxOperands "fma.rn.f32" line).any fun args =>
      args.length == 4 && args.drop 2 == ["0f3DCCCCCD", "0f00000000"]) &&
    (planes != 10 || verifiesDirectBitStores code tile scaled) &&
    (tile != 1 || (fourRowCoordinates code scaled && butterfly code &&
      butterfly (QKVTTIR.normalizePTX ptx.group) && butterfly (QKVTTIR.normalizePTX ptx.columnZero)))

private def acceptsDirect (scaled : Bool) (rows tile planes : Nat) : Prop :=
  if scaled then rows ∈ [16384, 40960]
  else rows ∈ [16384, 32768, 49152] ∧ (tile ∈ [1, 8, 32] ∨ tile = 128 ∧ planes = 10)

instance (scaled : Bool) (rows tile planes : Nat) : Decidable (acceptsDirect scaled rows tile planes) := by
  unfold acceptsDirect
  split <;> infer_instance

structure DirectProgram (scaled : Bool) where
  rows : Nat
  planes : Nat
  first : Nat
  tile : Nat
  warps : Nat
  group : String
  zero : String
  fused : String
  ptx : DirectPTX
  supported : acceptsDirect scaled rows tile planes ∧ (planes, first) ∈ [(1, 0), (9, 1), (10, 0)] ∧
    (tile, warps) ∈ [(1, 2), (8, 2), (32, 4), (128, 8)] ∧ rows % directTile tile = 0
  ports : declaredPorts fused scaled = true
  operations : (if tile == 128 && !scaled then recognizesWarpRows group zero fused
    else sameReductionAndPorts planes first tile group zero fused scaled) = true
  ptxChecked : matchesDirectPTX tile planes ptx scaled = true
  launchChecked : (launch tile warps group && launch tile warps zero && launch (directTile tile) warps fused) = true

def DirectProgram.check (scaled : Bool) (rows planes first tile warps : Nat) (group zero fused : String)
    (ptx : DirectPTX) : Option (DirectProgram scaled) := do
  if supported : acceptsDirect scaled rows tile planes ∧ (planes, first) ∈ [(1, 0), (9, 1), (10, 0)] ∧
      (tile, warps) ∈ [(1, 2), (8, 2), (32, 4), (128, 8)] ∧ rows % directTile tile = 0 then
    if ports : declaredPorts fused scaled then
      if operations : (if tile == 128 && !scaled then recognizesWarpRows group zero fused
    else sameReductionAndPorts planes first tile group zero fused scaled) then
        if ptxChecked : matchesDirectPTX tile planes ptx scaled then
          if launchChecked : launch tile warps group && launch tile warps zero && launch (directTile tile) warps fused then
            return ⟨rows, planes, first, tile, warps, group, zero, fused, ptx, supported, ports, operations,
              ptxChecked, launchChecked⟩
  none

private def supportedRows (scaled : Bool) (rows tile : Nat) : Bool :=
  rows % tile == 0 && (if scaled then [16384, 40960].contains rows else [16384, 32768, 49152].contains rows)

structure MixedProgram (scaled : Bool) where
  rows : Nat
  groupTile : Nat
  zeroTile : Nat
  group : String
  zero : String
  fused : String
  ptx : String
  supported : supportedRows scaled rows (mixedBlock groupTile zeroTile) = true
  irChecked : checkMixed groupTile zeroTile group zero fused scaled = true
  ptxChecked : verifiesMixedPTX groupTile zeroTile scaled ptx = true

theorem MixedProgram.row_coverage {scaled : Bool} (program : MixedProgram scaled) (row : Fin program.rows) :
    let tile := mixedBlock program.groupTile program.zeroTile
    row.val / tile < program.rows / tile := by
  intro tile
  have supported := program.supported
  simp only [supportedRows, Bool.and_eq_true, beq_iff_eq] at supported
  have exactTiles : program.rows % tile = 0 := supported.1
  have completeTiles : tile * (program.rows / tile) = program.rows := by
    simpa [exactTiles] using Nat.div_add_mod program.rows tile
  exact Nat.div_lt_of_lt_mul (by rw [completeTiles]; exact row.isLt)

#axiomsClean MixedProgram.row_coverage

theorem banded_row_offset (block : Fin 384) (band : Fin 8) (lane : Fin 16) :
    (((BitVec.ofNat 32 lane.val ||| (BitVec.ofNat 32 band.val <<< 4)) |||
      (BitVec.ofNat 32 block.val <<< 7))).toNat =
      128 * block.val + 16 * band.val + lane.val := by
  have blockLt := block.isLt
  have bandLt := band.isLt
  have laneLt := lane.isLt
  simp only [BitVec.toNat_or, BitVec.toNat_shiftLeft, BitVec.toNat_ofNat]
  have blockWord : block.val < 2 ^ 32 := by omega
  have bandWord : band.val < 2 ^ 32 := by omega
  have laneWord : lane.val < 2 ^ 32 := by omega
  have blockOffset : block.val <<< 7 < 2 ^ 32 := by simp only [Nat.shiftLeft_eq]; omega
  have bandOffset : band.val <<< 4 < 2 ^ 32 := by simp only [Nat.shiftLeft_eq]; omega
  rw [Nat.mod_eq_of_lt laneWord, Nat.mod_eq_of_lt bandWord, Nat.mod_eq_of_lt blockWord,
    Nat.mod_eq_of_lt bandOffset, Nat.mod_eq_of_lt blockOffset]
  rw [Nat.or_comm lane.val, ← Nat.shiftLeft_add_eq_or_of_lt laneLt]
  have localLt : (band.val <<< 4) + lane.val < 2 ^ 7 := by simp only [Nat.shiftLeft_eq]; omega
  rw [Nat.or_comm ((band.val <<< 4) + lane.val), ← Nat.shiftLeft_add_eq_or_of_lt localLt]
  simp only [Nat.shiftLeft_eq]
  omega

theorem banded_column_offset (band : Fin 17) (group : 8 ≤ band.val) :
    (BitVec.ofNat 32 band.val + BitVec.ofInt 32 (-7)).toNat = band.val - 7 := by
  revert band
  decide +kernel

#axiomsClean banded_row_offset
#axiomsClean banded_column_offset

/-- A band below eight writes column zero in a 16-row slice. Bands 8
through 16 write one column each across the whole 128-row block. -/
def BandedWrite {rows : Nat} (column : Fin 10) (row : Fin rows)
    (writer : Fin (rows / 128) × Fin 17 × Nat) : Prop :=
  let (block, band, lane) := writer
  if band.val < 8 then
    column.val = 0 ∧ lane < 16 ∧ row.val = 128 * block.val + 16 * band.val + lane
  else
    column.val = band.val - 7 ∧ lane < 128 ∧ row.val = 128 * block.val + lane

theorem banded_write_unique {rows : Nat} (aligned : rows % 128 = 0)
    (column : Fin 10) (row : Fin rows) :
    ∃! writer, BandedWrite column row writer := by
  have rowLt := row.isLt
  have columnLt := column.isLt
  let block : Fin (rows / 128) := ⟨row.val / 128, by omega⟩
  by_cases first : column.val = 0
  · let band : Fin 17 := ⟨row.val % 128 / 16, by omega⟩
    let lane := row.val % 16
    have bandLt : band.val < 8 := by dsimp [band]; omega
    refine ⟨(block, band, lane), ?_, ?_⟩
    · simp only [BandedWrite, if_pos bandLt]
      exact ⟨first, Nat.mod_lt _ (by decide), by dsimp [block, band, lane]; omega⟩
    · rintro ⟨otherBlock, otherBand, otherLane⟩ write
      have bandBound := otherBand.isLt
      by_cases ownsZero : otherBand.val < 8
      · simp only [BandedWrite, if_pos ownsZero] at write
        obtain ⟨_, otherLaneLt, rowEq⟩ := write
        refine Prod.ext (Fin.ext ?_) (Prod.ext (Fin.ext ?_) ?_)
        all_goals dsimp [block, band, lane]
        all_goals omega
      · simp only [BandedWrite, if_neg ownsZero] at write
        omega
  · let band : Fin 17 := ⟨column.val + 7, by omega⟩
    let lane := row.val % 128
    have bandGe : ¬band.val < 8 := by dsimp [band]; omega
    refine ⟨(block, band, lane), ?_, ?_⟩
    · simp only [BandedWrite, if_neg bandGe]
      exact ⟨by dsimp [band]; omega, Nat.mod_lt _ (by decide), by dsimp [block, lane]; omega⟩
    · rintro ⟨otherBlock, otherBand, otherLane⟩ write
      by_cases ownsZero : otherBand.val < 8
      · simp only [BandedWrite, if_pos ownsZero] at write
        exact (first write.1).elim
      · simp only [BandedWrite, if_neg ownsZero] at write
        obtain ⟨columnEq, otherLaneLt, rowEq⟩ := write
        refine Prod.ext (Fin.ext ?_) (Prod.ext (Fin.ext ?_) ?_)
        all_goals dsimp [block, band, lane]
        all_goals omega

#axiomsClean banded_write_unique


theorem MixedProgram.banded_writer_unique {scaled : Bool} (program : MixedProgram scaled)
    (group : program.groupTile = 128) (zero : program.zeroTile = 1)
    (column : Fin 10) (row : Fin program.rows) :
    ∃! writer, BandedWrite column row writer := by
  have supported := program.supported
  simp only [supportedRows, Bool.and_eq_true, beq_iff_eq, group, zero] at supported
  apply banded_write_unique _ column row
  simpa [mixedBlock] using supported.1

#axiomsClean MixedProgram.banded_writer_unique

def MixedProgram.check (scaled : Bool) (rows groupTile zeroTile : Nat) (group zero fused ptx : String) :
    Option (MixedProgram scaled) := do
  if supported : supportedRows scaled rows (mixedBlock groupTile zeroTile) then
    if irChecked : checkMixed groupTile zeroTile group zero fused scaled then
      if ptxChecked : verifiesMixedPTX groupTile zeroTile scaled ptx then
        return ⟨rows, groupTile, zeroTile, group, zero, fused, ptx, supported, irChecked, ptxChecked⟩
  none

end VeriTile.Bench.Models.NanoGPT.Track1.MuddFusedPacking.Binding
