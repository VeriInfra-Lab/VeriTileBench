import models.NanoGPT.Track1.MuddGradientPacking
import models.NanoGPT.Track1.QKVTTIRBinding

/-! Specialize the packing kernel to each of the ten planes. The recognized
store includes the exact source and destination offsets, nine ordered FP32
adds of the selected value and zeros, the FP32 scale, and the BF16 cast.
The old and new IR may issue ten and one loads respectively. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddGradientPacking.Binding

private inductive Expr where
  | nat (value : Nat)
  | pred (value : Bool)
  | f32 (value : String)
  | node (op : String) (arguments : List Expr)
deriving BEq, Repr

private def intOp (op : String) (a b : Expr) : Expr :=
  if op == "arith.addi" || op == "arith.muli" then
    if reprStr a ≤ reprStr b then .node op [a, b] else .node op [b, a]
  else .node op [a, b]

private def position (tile : Nat) : Expr :=
  intOp "arith.addi" (intOp "arith.muli" (.node "cta" []) (.nat tile)) (.node "range" [.nat tile])

private def input (column : Nat) : Expr := .node "input" [.nat column]
private def output : Expr := .node "output" []

private def expected (tile plane : Nat) : Expr :=
  let index := position tile
  let offset := intOp "arith.addi" (intOp "arith.remsi" index (.nat 12))
    (intOp "arith.muli" (intOp "arith.divsi" index (.nat 120)) (.nat 12))
  let value := Expr.node "arith.extf" [.node "load-bf16" [.node "address" [input plane, offset]]]
  let selected := fun col => if col == plane then value else .f32 "dense<0.000000e+00>"
  let total := (List.range 9).reverse.foldl (fun sum col => .node "arith.addf" [sum, selected col]) (selected 9)
  .node "store" [.node "address" [output, index],
    .node "arith.truncf" [.node "arith.mulf" [total, .f32 "dense<1.000000e-01>"]]]

private def contains (text part : String) : Bool := (text.splitOn part).length > 1

private def splitType (text : String) : Option (String × String) := do
  let pieces := text.splitOn " : "
  if pieces.length < 2 then return ← none
  pure (String.intercalate " : " pieces.dropLast, pieces.getLast!)

private def canonicalHeader (line : String) : Bool :=
  let arg (port : String) := port ++ ": !tt.ptr<bf16> {tt.divisibility = 16 : i32}"
  let pointers := arg "%in_out_ptr0" :: (List.range 10).map (fun i => arg s!"%in_ptr{i}")
  let signature := String.intercalate ", " (pointers ++ ["%xnumel: i32 {tt.divisibility = 16 : i32}"])
  line.endsWith ("(" ++ signature ++ ") attributes {noinline = false} {")

private def operand (values : Std.HashMap String Expr) (token : String) : Option Expr :=
  values[token]?.or (token.toNat?.map .nat)

private def replay (tile plane : Nat) (lines : List String) : Option (Expr × Nat) := do
  let mut values : Std.HashMap String Expr := ({} : Std.HashMap String Expr).insert "%in_out_ptr0" output
  for i in [:10] do
    values := values.insert s!"%in_ptr{i}" (input (9 - i))
  let mut stores : List Expr := []
  let mut loads := 0
  let mut header := false
  for line in lines do
    if line.startsWith "tt.func public @" then
      if header || !canonicalHeader line then return ← none
      header := true
      continue
    if line.startsWith "module" || line.startsWith "#" || line == "tt.return" || line == "}" then continue
    let parts := line.splitOn " = "
    let target := parts.head!
    let rhs := if parts.length == 1 then line else String.intercalate " = " parts.tail!
    let (code, ty) ← splitType rhs
    let op := (code.splitOn " ").head!
    let args := (code.drop (op.length + 1)).toString
    let inputs := (args.splitOn " ").filterMap fun token =>
      let term := if token.endsWith "," then (token.dropEnd 1).toString else token
      if term.startsWith "%" then some term else none
    let refs ← inputs.mapM (operand values)
    if op == "tt.store" then
      if parts.length != 1 || refs.length != 2 || !contains ty "!tt.ptr<bf16>" then return ← none
      stores := .node "store" refs :: stores
      continue
    if parts.length == 1 || values.contains target then return ← none
    let value ←
      match op, refs with
      | "arith.constant", [] =>
        if ty == "i32" || contains ty "xi32" then
          let text := if args.startsWith "dense<" then ((args.drop 6).dropEnd 1).toString else args
          text.toNat?.map Expr.nat
        else if contains ty "xf32" then some (.f32 args) else none
      | "tt.get_program_id", [] => if args == "x" && ty == "i32" then some (.node "cta" []) else none
      | "tt.make_range", [] =>
        if args == ("{end = " ++ toString tile ++ " : i32, start = 0 : i32}") && contains ty s!"{tile}xi32" then
          some (.node "range" [.nat tile])
        else none
      | "tt.splat", [a] | "ttg.convert_layout", [a] => some a
      | "arith.addi", _ | "arith.muli", _ | "arith.divsi", _ | "arith.remsi", _ => do
        let [left, right] := args.splitOn ", " | none
        let a ← operand values left
        let b ← operand values right
        if !contains ty "i32" then none
        else if op == "arith.remsi" && a == intOp "arith.divsi" (position tile) (.nat 12) && b == .nat 10 then
          some (.nat plane)
        else some (intOp op a b)
      | "arith.cmpi", _ =>
        match args.splitOn ", " with
        | ["eq", a, b] => do
          let .nat x ← operand values a | none
          let .nat y ← operand values b | none
          pure (.pred (x == y))
        | _ => none
      | "arith.select", [.pred cond, a, b] => some (if cond then a else b)
      | "tt.addptr", [a, b] => if contains ty "!tt.ptr<bf16>" then some (.node "address" [a, b]) else none
      | "tt.load", [a] => if args.endsWith "evictionPolicy = evict_last" && contains ty "!tt.ptr<bf16>" then
          some (.node "load-bf16" [a]) else none
      | "arith.extf", [a] => if contains ty "xbf16" && contains ty "xf32" then some (.node op [a]) else none
      | "arith.truncf", [a] => if contains ty "xf32" && contains ty "xbf16" then some (.node op [a]) else none
      | "arith.addf", [a, b] | "arith.mulf", [a, b] =>
        if contains ty "xf32" then some (.node op [a, b]) else none
      | _, _ => none
    if op == "tt.load" then loads := loads + 1
    values := values.insert target value
  let [written] := stores | none
  if header then some (written, loads) else none

def recognizes (tile : Nat) (reference candidate : String) : Bool :=
  (do
    let original ← QKVTTIR.normalizeTTIR reference
    let routed ← QKVTTIR.normalizeTTIR candidate
    pure ((List.range 10).all fun plane => replay tile plane original == some (expected tile plane, 10) &&
      replay tile plane routed == some (expected tile plane, 1))).getD false

private def verifyLoadEdit (before after : String) : Option Unit := do
  let old := before.splitOn "\n"
  let new := after.splitOn "\n"
  if old.length != new.length then return ← none
  let changed := ((old.zip new).zipIdx).filter fun ((a, b), _) => a != b
  if changed.length != 10 then return ← none
  let pointer := (List.range 9).foldl (fun selected i =>
    s!"tl.where(x1 == {i + 1}, in_ptr{8 - i}, {selected})") "in_ptr9"
  let original := fun port =>
    s!"tl.load(in_ptr{port} + (x0 + 12*x2), None, eviction_policy='evict_last').to(tl.float32)"
  let routed := s!"tl.load({pointer} + (x0 + 12*x2), None, eviction_policy='evict_last').to(tl.float32)"
  let first := changed.head!
  let [loadedName, _] := first.1.1.splitOn " = " | none
  let loadedName := loadedName.trimAscii.toString
  for (((old, new), index), port) in changed.zipIdx do
    if index != first.2 + port then return ← none
    let [lhs, rhs] := old.splitOn " = " | none
    let result := if port == 0 then routed else loadedName
    if rhs != original port || new != lhs ++ " = " ++ result then return ← none
  pure ()

def checksLoadEdit (before after : String) : Bool := (verifyLoadEdit before after).isSome

private def launchLayout (tile warps : Nat) (ttgir : String) : Bool :=
  let attrs := s!"\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = {warps} : i32, ttg.target = \"cuda:90\""
  contains ttgir attrs && contains ttgir s!"tensor<{tile}xbf16,"

structure Program where
  rows : Nat
  tile : Nat
  warps : Nat
  referenceTTIR : String
  candidateTTIR : String
  referenceTTGIR : String
  candidateTTGIR : String
  supported : (tile, warps) ∈ [(1024, 4), (512, 8)] ∧ rows * 120 % tile = 0
  sourceChecked : recognizes tile referenceTTIR candidateTTIR = true
  layoutChecked : recognizes tile referenceTTGIR candidateTTGIR = true
  launchChecked : (launchLayout tile warps referenceTTGIR && launchLayout tile warps candidateTTGIR) = true

def Program.check (rows tile warps : Nat) (referenceTTIR candidateTTIR referenceTTGIR candidateTTGIR : String) :
    Option Program := do
  if supported : (tile, warps) ∈ [(1024, 4), (512, 8)] ∧ rows * 120 % tile = 0 then
    if source : recognizes tile referenceTTIR candidateTTIR = true then
      if layout : recognizes tile referenceTTGIR candidateTTGIR = true then
        if launch : (launchLayout tile warps referenceTTGIR && launchLayout tile warps candidateTTGIR) = true then
          return ⟨rows, tile, warps, referenceTTIR, candidateTTIR, referenceTTGIR, candidateTTGIR,
            supported, source, layout, launch⟩
  none

end VeriTile.Bench.Models.NanoGPT.Track1.MuddGradientPacking.Binding
