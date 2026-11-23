import models.NanoGPT.Track1.ValidationExpBinding

/-! Replay the validation reduction from the vocabulary-loop latch to
the FP32 registers consumed by the loss. Each lane starts with a named
partial for its row and reduction slot. The replay checks the actual
shuffle, shared-memory exchange and FP32 operand order for every row. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValidationReduceBinding
open TrainingCEGatherBinding
open VeriTile.Triton FP
open TailNorm (F32 BF16)
open ValidationCE (Tile ReductionOps)

structure Slot where
  slab : Nat
  row : Nat
  column : Nat
  deriving DecidableEq, Repr

inductive Expr where
  | zero
  | priorMax (slot : Slot)
  | priorSum (slot : Slot)
  | maximum (a b : Expr)
  | add (a b : Expr)
  | mul (a b : Expr)
  | fma (a b c : Expr)
  | scale (maximum prior : Expr)
  deriving DecidableEq

abbrev Partials := Slot → ValidationCE.Partial

def Expr.eval (native : ValidationCE.Native) (read : Partials) : Expr → F32
  | .zero => TailNorm.zero
  | .priorMax slot => (read slot).maximum
  | .priorSum slot => (read slot).sum
  | .maximum a b => ValidationCE.max (a.eval native read) (b.eval native read)
  | .add a b => TailNorm.add (a.eval native read) (b.eval native read)
  | .mul a b => FP.mul FP.Config.ieeeValue (a.eval native read) (b.eval native read)
  | .fma a b c => FP.fma FP.Config.ieeeValue (a.eval native read) (b.eval native read) (c.eval native read)
  | .scale m p => ValidationCE.rescale native (m.eval native read) ⟨p.eval native read, TailNorm.zero⟩

private def ops : ReductionOps Expr :=
  ⟨.zero, .maximum, .add, .mul, .fma, .scale⟩

def expected (tile : Tile) (slab row : Nat) : Expr × Expr :=
  ValidationCE.reduceWith ops tile
    (fun column => .priorMax ⟨slab, row, column⟩) (fun column => .priorSum ⟨slab, row, column⟩)

def rowCount : Tile → Nat
  | .x1r1024 | .x1r512 => 1
  | .x16r32 => 16
  | .x64r64 | .x64r4 => 64
  | .x8r512 | .x8r256 => 8

def registerGroups : Tile → List Nat
  | .x1r1024 => [0, 1, 2, 3]
  | .x1r512 => [0, 1]
  | .x16r32 => [1, 0, 3, 2]
  | .x64r4 => [0]
  | .x64r64 => [1, 0, 3, 2, 5, 4, 7, 6]
  | .x8r256 => [1, 0, 2, 3]
  | .x8r512 => [1, 0, 2, 3, 4, 5, 6, 7]

def inputSlot (tile : Tile) (position tid : Nat) : Option Slot := do
  let group ← (registerGroups tile)[position % tile.groups]?
  return ⟨position / tile.groups, tid % rowCount tile, tid / rowCount tile + group * tile.lanes⟩

private inductive Test where
  | flag (value : Bool)
  | greater (a b : Expr)
  | nan (arg : Expr)

private inductive Value where
  | bits (word : UInt32)
  | expr (word : Expr)
  | test (value : Test)
  | select (yes no : Expr) (test : Test)

private abbrev Registers := Std.HashMap String Value

private def read (regs : Registers) (arg : String) : Option Value :=
  if arg == "global_smem" then some (.bits 0)
  else (regs.get? arg).orElse (fun _ => arg.toInt?.map (fun n => .bits (UInt32.ofNat (n % 4294967296).toNat)))

private def getBits (regs : Registers) (arg : String) : Option UInt32 := do
  let .bits word ← read regs arg | none
  return word

private def getExpr (regs : Registers) (arg : String) : Option Expr := do
  let .expr term ← read regs arg | none
  return term

private def getTest (regs : Registers) (arg : String) : Option Test := do
  let .test test ← read regs arg | none
  return test

private def select (regs : Registers) (yes no pred : String) : Option Value := do
  let test ← getTest regs pred
  if let .flag active := test then read regs (if active then yes else no) else
    let a ← getExpr regs yes
    let b ← read regs no
    match test, b with
    | .nan x, .select y z (.greater p q) =>
      guard (a = x ∧ a = y ∧ a = p ∧ z = q)
      return .expr (.maximum a z)
    | .greater x y, .select p q (.nan z) =>
      guard (a = x ∧ a = p ∧ a = z ∧ y = q)
      return .expr (.maximum a q)
    | test, .expr b => return .select a b test
    | _, _ => none

private def integer (regs : Registers) : List String → Option Value
  | ["mov.pred", _, pred] =>
    if pred == "-1" then some (.test (.flag true))
    else if pred == "0" then some (.test (.flag false)) else none
  | [op, _, a] =>
    if ["mov.u32", "mov.b32", "cvt.u32.u64", "cvt.u64.u32"].contains op then
      (getBits regs a).map .bits else none
  | ["bfe.u32", _, a, b, n] => do
    let src ← getBits regs a
    let start ← getBits regs b
    let width ← getBits regs n
    guard (start.toNat + width.toNat ≤ 32)
    return .bits ((src >>> start) &&& UInt32.ofNat (2 ^ width.toNat - 1))
  | ["and.pred", _, a, b] => do
    let .flag left ← getTest regs a | none
    let .flag right ← getTest regs b | none
    return .test (.flag (left && right))
  | [op, _, a, b] => do
    let x ← getBits regs a
    let y ← getBits regs b
    match op with
    | "and.b32" => some (.bits (x &&& y))
    | "or.b32" => some (.bits (x ||| y))
    | "xor.b32" => some (.bits (x ^^^ y))
    | "shl.b32" => some (.bits (if y < 32 then x <<< y else 0))
    | "shr.u32" => some (.bits (if y < 32 then x >>> y else 0))
    | "add.s32" => some (.bits (x + y))
    | "sub.s32" => some (.bits (x - y))
    | "mul.lo.s32" => some (.bits (x * y))
    | "setp.eq.b32" => some (.test (.flag (x == y)))
    | "setp.lt.u32" => some (.test (.flag (x < y)))
    | _ => none
  | _ => none

private def arithmetic (regs : Registers) : List String → Option Value
  | ["selp.b32", _, yes, no, pred] | ["selp.f32", _, yes, no, pred] => select regs yes no pred
  | ["mov.b32", _, src] => read regs src
  | ["setp.gt.f32", _, a, b] => do
    return .test (.greater (← getExpr regs a) (← getExpr regs b))
  | ["setp.nan.f32", _, a, b] => do
    guard (a == b)
    return .test (.nan (← getExpr regs a))
  | ["add.f32", _, a, b] => do
    return .expr (.add (← getExpr regs a) (← getExpr regs b))
  | ["mul.f32", _, a, b] => do
    return .expr (.mul (← getExpr regs a) (← getExpr regs b))
  | ["fma.rn.f32", _, a, b, c] => do
    return .expr (.fma (← getExpr regs a) (← getExpr regs b) (← getExpr regs c))
  | insn => integer regs insn

private def writes (line : String) (insn : List String) : List String :=
  (((insn.flatMap (·.splitOn "|")).filter (·.startsWith "%")).eraseDups).filter
    (fun reg => defines reg line)

structure Instruction where
  pc : Nat
  words : List String
  pred : Option String
  output : List String

def Instruction.ofLine (pc : Nat) (line : String) : Instruction :=
  let tokens := operands line
  let pred := tokens.head?.filter (·.startsWith "@")
  ⟨pc, if pred.isSome then tokens.drop 1 else tokens, pred, writes line tokens⟩

private def integerOp (name : String) : Bool :=
  ["mov.pred", "mov.u32", "mov.b32", "cvt.u32.u64", "cvt.u64.u32", "bfe.u32", "and.pred", "and.b32",
    "or.b32", "xor.b32", "shl.b32", "shr.u32", "add.s32", "sub.s32", "mul.lo.s32", "setp.eq.b32",
    "setp.lt.u32"].contains name

private def liveIns (insns : Array Instruction) (edges : Array (List Nat)) (latch : Nat) :
    Array Instruction := Id.run do
  let mut counts : Std.HashMap String Nat := {}
  for insn in insns do
    for reg in insn.output do
      counts := counts.insert reg (1 + counts.getD reg 0)
  return (insns.extract 0 latch).filter fun i =>
    i.pred.isNone && integerOp (i.words.headD "") &&
      (match i.output with | [reg] => counts.getD reg 0 == 1 | _ => false) &&
      dominates edges i.pc latch

private def initial (data : ValidationExpBinding.Decoded) (tile : Tile)
    (seed : Array Instruction) (tid : Nat) : Option Registers := do
  let mut regs : Registers := ({} : Registers).insert "%tid.x" (.bits (UInt32.ofNat tid))
  for i in seed do
    if let ([dest], some word) := (i.output, integer regs i.words) then
      regs := regs.insert dest word
  for (step, position) in data.updates.zipIdx do
    let slot ← inputSlot tile position tid
    let maximum := ValidationExpBinding.carriedMaximum step
    guard (!(regs.contains maximum) && !(regs.contains step.oldSum) && maximum != step.oldSum)
    regs := (regs.insert maximum (.expr (.priorMax slot))).insert step.oldSum (.expr (.priorSum slot))
  return regs

def straightPath (edges : Array (List Nat)) (first last : Nat) : Option (List Nat) := do
  let mut admissible := Array.replicate edges.size false
  for k in List.range (last - first + 1) do
    let pc := last - k
    let next ← edges[pc]?
    guard (pc == last || next.all (pc < ·))
    admissible := admissible.set! pc (pc == last || next.any (fun n => admissible[n]!))
  let mut pc := first
  let mut visited := []
  for _ in List.range (last - first + 1) do
    visited := pc :: visited
    if pc == last then return visited.reverse
    let next ← edges[pc]?
    let [dest] := next.filter (fun n => admissible[n]!) | none
    pc := dest
  none

private structure State where
  lanes : Array Registers
  shared : Std.HashMap UInt32 Expr := {}
  writes : Std.HashMap UInt32 Expr := {}
  /-- A present key with no owner was read by more than one lane. -/
  reads : Std.HashMap UInt32 (Option Nat) := {}
  scales : Std.HashMap String (Array (Option Expr)) := {}

private def enabled (regs : Registers) (pred : Option String) : Option Bool := do
  match pred with
  | none => return true
  | some raw =>
    let neg := raw.startsWith "@!"
    let .flag p ← getTest regs ((raw.drop (if neg then 2 else 1)).toString) | none
    return p != neg

private def addressWord (regs : Registers) (arg : String) : Option UInt32 := do
  guard (arg.startsWith "[" && arg.endsWith "]")
  getBits regs (arg.drop 1 |>.dropEnd 1).toString

private def store (state : State) (pred : Option String) (ptr data : String) : Option State := do
  let mut staged := state.writes
  for (regs, tid) in state.lanes.toList.zipIdx do
    if ← enabled regs pred then
      let addr ← addressWord regs ptr
      let word ← getExpr regs data
      let readers := state.reads.get? addr
      guard (addr.toNat % 4 == 0 && !staged.contains addr &&
        (readers.isNone || readers == some (some tid)))
      staged := staged.insert addr word
  return { state with writes := staged }

private def load (state : State) (pred : Option String) (dst ptr : String) : Option State := do
  let mut lanes := #[]
  let mut reads := state.reads
  for (regs, tid) in state.lanes.toList.zipIdx do
    if ← enabled regs pred then
      let addr ← addressWord regs ptr
      guard (addr.toNat % 4 == 0 && !state.writes.contains addr)
      let value ← state.shared.get? addr
      let reader := match reads.get? addr with
        | none => some tid
        | some previous => previous.filter (· == tid)
      reads := reads.insert addr reader
      lanes := lanes.push (regs.insert dst (.expr value))
    else lanes := lanes.push regs
  return { state with lanes, reads }

private def bindWord (regs : Registers) (key : String) (value : Option Value) : Registers :=
  if let some word := value then regs.insert key word else regs.erase key

private def shuffle (state : State) (dst src delta : String) : Option State := do
  let offset ← delta.toNat?
  guard (offset < 32)
  let lanes ← (List.range state.lanes.size).toArray.mapM fun lane => do
    let regs ← state.lanes[lane ^^^ offset]?
    return bindWord state.lanes[lane]! dst (getExpr regs src |>.map .expr)
  return { state with lanes }

private def execute (state : State) (i : Instruction) :
    Option State := do
  match i.words with
  | ["bar.sync", "0"] =>
    guard (i.pred.isNone)
    return { state with
      shared := state.writes.fold (fun mem key value => mem.insert key value) state.shared
      reads := {}
      writes := {} }
  | ["st.shared.b32", ptr, data] => store state i.pred ptr data
  | ["st.shared.b32", "[", ptr, "+", "0", "]", data] => store state i.pred ("[" ++ ptr ++ "]") data
  | ["ld.shared.b32", dst, ptr] => load state i.pred dst ptr
  | ["ld.shared.b32", dst, "[", ptr, "+", "0", "]"] => load state i.pred dst ("[" ++ ptr ++ "]")
  | ["shfl.sync.bfly.b32", dst, src, delta, "31", "-1"] =>
    guard (i.pred.isNone)
    shuffle state dst src delta
  | name :: _ =>
    guard (!(name.startsWith "st." || name.startsWith "ld.shared" || name.startsWith "shfl." ||
      name.startsWith "bar." || name.startsWith "atom." || name.startsWith "red." || name.startsWith "stmatrix."))
    let lanes := state.lanes.map fun regs =>
      if (enabled regs i.pred) == some false then regs
      else if let ([dest], some true) := (i.output, enabled regs i.pred) then
        bindWord regs dest (arithmetic regs i.words)
      else i.output.foldl (·.erase ·) regs
    return { state with lanes }
  | [] => return state

abbrev Rows := List (List (Expr × Expr))

private def ownerWords (state : State) (rows : Nat) (reg : String) : Option (Array Expr) :=
  (state.lanes.extract 0 rows).mapM fun lane => getExpr lane reg

def replay (ptx : String) (tile : Tile) (scan : ValidationExpBinding.Certificate ptx tile)
    (finish : ValidationLogBinding.Certificate ptx) : Option Rows := do
  let lines := QKVTTIR.normalizePTX ptx
  let insns := (lines.zipIdx.map fun (line, pc) => Instruction.ofLine pc line).toArray
  let edges ← ValidationLogBinding.validationFlow lines
  let sinks := finish.parts
  let threads := rowCount tile * tile.lanes
  guard (lines.contains s!".reqntid {threads}" && sinks.all (fun sink => sink.owners == rowCount tile))
  let mut last := scan.reduction.latch
  for sink in sinks do
    last := max last (max sink.sumDef sink.maximumDef)
  let path ← straightPath edges (scan.reduction.latch + 1) last
  guard (sinks.all (fun sink => path.contains sink.sumDef && path.contains sink.maximumDef))
  let seed := liveIns insns edges scan.reduction.latch
  let lanes ← (List.range threads).toArray.mapM (initial scan.reduction tile seed)
  let mut state : State := ⟨lanes, {}, {}, {}, {}⟩
  let mut maxima : Std.HashMap Nat (Array Expr) := {}
  let mut sums : Std.HashMap Nat (Array Expr) := {}
  let mut sources : Std.HashMap Nat ValidationExpBinding.Site := {}
  let mut results : Std.HashMap Nat ValidationExpBinding.Site := {}
  for slot in scan.reduction.scaled do
    let site := slot.scale
    guard (path.contains site.inputPC && path.contains site.outputPC && path.contains slot.resultPC)
    guard (!sources.contains site.inputPC && !results.contains site.outputPC)
    sources := sources.insert site.inputPC site
    results := results.insert site.outputPC site
  let maxReads := (sinks.map fun sink => (sink.maximumDef, sink.maximum)).toArray
  let sumReads := (sinks.map fun sink => (sink.sumDef, sink.sum)).toArray
  for pc in path do
    let insn ← insns[pc]?
    if let some site := sources.get? pc then
      guard (insn.pred.isNone)
      let values := state.lanes.map fun regs => do
        return Expr.scale (← getExpr regs site.maximum) (← getExpr regs site.source)
      state := {state with scales := state.scales.insert site.output values}
    if let some site := results.get? pc then
      guard (insn.pred.isNone && insn.output == [site.output])
      let values ← state.scales.get? site.output
      let lanes := (state.lanes.zip values).map fun (regs, word) => bindWord regs site.output (word.map .expr)
      state := {state with lanes, scales := state.scales.erase site.output}
    else state ← execute state insn
    if let some (_, reg) := maxReads.find? (·.1 == pc) then
      maxima := maxima.insert pc (← ownerWords state (rowCount tile) reg)
    if let some (_, reg) := sumReads.find? (·.1 == pc) then
      sums := sums.insert pc (← ownerWords state (rowCount tile) reg)
  sinks.mapM fun sink => do
    let maxima ← maxima.get? sink.maximumDef
    let sums ← sums.get? sink.sumDef
    return (maxima.zip sums).toList

def reference (tile : Tile) : Rows :=
  (List.range ValidationCE.slabCount).map fun slab =>
    (List.range (rowCount tile)).map fun row => expected tile slab row

structure Certificate (ptx : String) (tile : Tile) where
  scan : ValidationExpBinding.Certificate ptx tile
  finish : ValidationLogBinding.Certificate ptx
  exact : replay ptx tile scan finish = some (reference tile)

def certify (ptx : String) (tile : Tile) : Option (Certificate ptx tile) := do
  let scan ← ValidationExpBinding.certify ptx tile
  let finish ← ValidationLogBinding.certify ptx
  if exact : replay ptx tile scan finish = some (reference tile) then some ⟨scan, finish, exact⟩ else none

private theorem foldMapped (f : α → β) (op : α → γ → α) (other : β → γ → β)
    (same : ∀ a x, f (op a x) = other (f a) x) (items : List γ) (init : α) :
    f (items.foldl op init) = items.foldl other (f init) := by
  induction items generalizing init with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, ih, same]

private theorem butterflyMapped (tile : Tile) (f : α → β) (z : α)
    (op : α → α → α) (other : β → β → β)
    (same : ∀ a b, f (op a b) = other (f a) (f b)) (input : Nat → α) :
    f (ValidationCE.butterfly tile z op input) =
      ValidationCE.butterfly tile (f z) other (f ∘ input) := by
  let step (buf : Array α) (offset : Nat) :=
    (List.range tile.lanes).toArray.map fun lane => op (buf[lane]?.getD z) (buf[lane ^^^ offset]?.getD z)
  let stepMapped (buf : Array β) (offset : Nat) :=
    (List.range tile.lanes).toArray.map fun lane =>
      other (buf[lane]?.getD (f z)) (buf[lane ^^^ offset]?.getD (f z))
  have stepExact (buf : Array α) (offset : Nat) :
      (step buf offset).map f = stepMapped (buf.map f) offset := by
    simp [step, stepMapped, Array.map_map, Function.comp_def, same]
  let initial := (List.range tile.lanes).toArray.map input
  change f ((tile.shuffles.foldl step initial)[0]?.getD z) =
    (tile.shuffles.foldl stepMapped ((List.range tile.lanes).toArray.map (f ∘ input)))[0]?.getD (f z)
  have output := foldMapped (fun buf : Array α => buf.map f) step stepMapped stepExact tile.shuffles initial
  have atZero := congrArg (fun buf : Array β => buf[0]?.getD (f z)) output
  simpa only [initial, Array.getElem?_map, Option.getD_map, Array.map_map, List.toArray_range] using atZero

theorem expected_eval (native : ValidationCE.Native) (read : Partials) (tile : Tile) (slab row : Nat) :
    let (maximum, sum) := expected tile slab row
    ⟨maximum.eval native read, sum.eval native read⟩ =
      ValidationCE.reduce native tile (fun column => read ⟨slab, row, column⟩) := by
  let f := Expr.eval native read
  let num := ValidationCE.reductionOps native
  let index := fun lane group => lane + group * tile.lanes
  let mx := fun lane group => Expr.priorMax ⟨slab, row, index lane group⟩
  let su := fun lane group => Expr.priorSum ⟨slab, row, index lane group⟩
  let maxLocal := fun lane =>
    (List.range (tile.groups - 1)).foldl (fun a g => Expr.maximum a (mx lane (g + 1))) (mx lane 0)
  let maxNum := fun lane =>
    (List.range (tile.groups - 1)).foldl (fun a g => num.maximum a (f (mx lane (g + 1)))) (f (mx lane 0))
  let global := ValidationCE.butterfly tile Expr.zero Expr.maximum maxLocal
  let globalNum := ValidationCE.butterfly tile num.default num.maximum maxNum
  have sameGlobal : f global = globalNum := by
    unfold global globalNum
    rw [butterflyMapped tile f .zero .maximum num.maximum (fun _ _ => rfl)]
    congr 1
    funext lane
    exact foldMapped f _ _ (fun _ _ => rfl) _ _
  let product := fun lane group => Expr.mul (su lane group) (.scale global (mx lane group))
  let productNum := fun lane group => num.mul (f (su lane group)) (num.scale globalNum (f (mx lane group)))
  let fuse := fun lane a group => Expr.fma (su lane group) (.scale global (mx lane group)) a
  let fuseNum := fun lane a group => num.fma (f (su lane group)) (num.scale globalNum (f (mx lane group))) a
  have sameProduct (lane group : Nat) : f (product lane group) = productNum lane group := by
    change num.mul (f (su lane group)) (num.scale (f global) (f (mx lane group))) = _
    rw [sameGlobal]
  have sameFuse (lane : Nat) (acc : Expr) (group : Nat) :
      f (fuse lane acc group) = fuseNum lane (f acc) group := by
    change num.fma (f (su lane group)) (num.scale (f global) (f (mx lane group))) (f acc) = _
    rw [sameGlobal]
  let localSum := fun lane => if tile.groups = 1 then product lane 0 else
    (List.range (tile.groups - 2)).foldl (fun a g => fuse lane a (g + 2)) (fuse lane (product lane 1) 0)
  let numSum := fun lane => if tile.groups = 1 then productNum lane 0 else
    (List.range (tile.groups - 2)).foldl (fun a g => fuseNum lane a (g + 2)) (fuseNum lane (productNum lane 1) 0)
  have sameSum : f (ValidationCE.butterfly tile Expr.zero Expr.add localSum) =
      ValidationCE.butterfly tile num.default num.add numSum := by
    rw [butterflyMapped tile f .zero .add num.add (fun _ _ => rfl)]
    congr 1
    funext lane
    change f (localSum lane) = numSum lane
    unfold localSum numSum
    split
    · exact sameProduct lane 0
    · rw [foldMapped f (fun a g => fuse lane a (g + 2)) (fun a g => fuseNum lane a (g + 2))
        (fun a g => sameFuse lane a (g + 2)), sameFuse, sameProduct]
  change (⟨f global, f (ValidationCE.butterfly tile Expr.zero Expr.add localSum)⟩ : ValidationCE.Partial) =
    ⟨globalNum, ValidationCE.butterfly tile num.default num.add numSum⟩
  rw [sameGlobal, sameSum]

def reduced {ptx : String} {tile : Tile} (artifact : Certificate ptx tile)
    (native : ValidationCE.Native) (read : Partials)
    (slab : Fin ValidationCE.slabCount) (row : Fin (rowCount tile)) : Option ValidationCE.Partial :=
    (((replay ptx tile artifact.scan artifact.finish).bind fun slabs =>
      slabs[slab.val]?.bind fun rows => rows[row.val]?).map fun (m, s) =>
        ⟨m.eval native read, s.eval native read⟩)

/-- For any FP32 loop-carried values, the actual two inputs to each
row's loss evaluate to the ordered validation reduction. -/
theorem decoded_reduction_exact {ptx : String} {tile : Tile} (artifact : Certificate ptx tile)
    (native : ValidationCE.Native) (read : Partials)
    (slab : Fin ValidationCE.slabCount) (row : Fin (rowCount tile)) :
    reduced artifact native read slab row =
      some (ValidationCE.reduce native tile (fun column => read ⟨slab.val, row.val, column⟩)) := by
  rw [reduced, artifact.exact]
  simp only [Option.bind_some, reference, List.getElem?_map, List.getElem?_range slab.isLt,
    List.getElem?_range row.isLt, Option.map_some]
  exact congrArg some (expected_eval native read tile slab.val row.val)

/-- The validated reduction and loss instructions agree with the row
model when the input registers contain its vocabulary-loop partials. -/
theorem decoded_row_loss_exact {ptx : String} {tile : Tile} (artifact : Certificate ptx tile)
    (native : ValidationCE.Native) (slab : Fin ValidationCE.slabCount) (row : Fin (rowCount tile))
    (projection : Fin 50304 → BF16) (mask : Nat → UInt8) (target : Option (Fin 50304)) :
    let word := fun column : Fin 50304 => ValidationCE.logit native mask column.val (projection column)
    let partials := (List.range tile.extent).toArray.map (ValidationCE.scan native tile word)
    ((reduced artifact native (fun slot => partials[slot.column]?.getD ValidationCE.empty) slab row).bind
      fun values => ValidationLogBinding.finishRow ptx slab native target mask projection values.maximum values.sum) =
      some (ValidationCE.rowLoss native tile projection mask target) := by
  dsimp only
  rw [decoded_reduction_exact, Option.bind_some, ValidationLogBinding.decoded_store_gather_exact artifact.finish]
  cases target <;> rfl

#axiomsClean decoded_reduction_exact
#axiomsClean decoded_row_loss_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValidationReduceBinding
