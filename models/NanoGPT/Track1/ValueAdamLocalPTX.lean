import models.NanoGPT.Track1.ValueAdamLocalAddress
import models.NanoGPT.Track1.TrainingCEGatherBinding

/-! Decode the real, straight-line SM90a local-scatter instructions.
The CTA index and CPU inputs remain symbolic. Each physical thread is
checked with the row in and out of bounds. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows

private def Address.literal? (a : Address) : Option Nat := do
  guard (a == Address.literal a.byte && a.unsigned32)
  return a.byte.toNat

inductive Port where
  | source | index | target | ticket
  deriving BEq, DecidableEq

private inductive Reg where
  | num (val : Address)
  | ptr (port : Port) (val : Address)
  | pred (val warpUniform : Bool)
  | packed (source : Address)
  | half (source : Address)
  | dead

inductive Effect where
  | lookup (address : Address)
  | read (source : Address)
  | atom (target source : Address)
  | clear (source : Address)
  | mark (address ticket : Address)
  deriving BEq, DecidableEq

private abbrev Regs := Std.HashMap String Reg

private structure State where
  regs : Regs
  effects : List Effect := []

private def number (regs : Regs) (arg : String) : Option Reg := do
  if arg.startsWith "%" then regs.get? arg
  else if arg == "0x0" then some (.num (Address.literal 0))
  else return .num (Address.literal (← arg.toInt?))

private def value (regs : Regs) (arg : String) : Option Address := do
  let .num n ← number regs arg | none
  return n

private def predicate (regs : Regs) (arg : String) : Option Bool := do
  let .pred p _ ← regs.get? arg | none
  return p

private def ptr (regs : Regs) (arg : String) : Option (Port × Address) := do
  guard (arg.startsWith "[" && arg.endsWith "]")
  let name := (arg.drop 1 |>.dropEnd 1).toString
  let (base, off) ← match name.splitOn "+" with
    | [base] => some (base, 0)
    | [base, off] => off.toInt?.map (fun byte => (base, byte))
    | _ => none
  let .ptr port address ← regs.get? base | none
  return (port, address.plus (Address.literal off))

private def parameter (wide : Bool) (arg : String) : Option Reg :=
  match wide, arg with
  | true, "[local_rows_param_0]" => some (.ptr .source {})
  | true, "[local_rows_param_1]" => some (.ptr .index {})
  | true, "[local_rows_param_2]" => some (.ptr .target {})
  | false, "[local_rows_param_3]" => some (.num {offset := 1})
  | false, "[local_rows_param_4]" => some (.num {first := 1})
  | false, "[local_rows_param_5]" => some (.num {count := 1})
  | true, "[local_rows_param_6]" => some (.ptr .ticket {})
  | _, _ => none

private def slot (thread : Nat) : Address := {block := 4, byte := thread / 32}

private def memory (s : State) (op : String) (args : List String) (run : Bool) : Option State := do
  if op == "ld.global.b32" || op == "ld.global.v4.b32" then
    let addr :: names := args.reverse | none
    let names := names.reverse
    let (port, address) ← ptr s.regs addr
    let effect ← match port, names with
      | .index, [_] => some (.lookup address)
      | .source, [_, _, _, _] => some (.read address)
      | _, _ => none
    if !run then
      return {s with regs := names.foldl (fun regs dst => regs.insert dst ((regs.get? dst).getD .dead)) s.regs}
    let regs := match port with
      | .index => s.regs.insert (names.headD "") (.num {row := 1})
      | _ => names.zipIdx.foldl (fun regs (name, i) =>
        regs.insert name (.packed (address.plus (Address.literal (4 * i : Nat))))) s.regs
    return {regs, effects := effect :: s.effects}
  else if op == "st.global.v4.b32" then
    let [addr, a, b, c, d] := args | none
    let (.source, address) ← ptr s.regs addr | none
    if !run then return s
    let payload ← [a, b, c, d].mapM (value s.regs)
    guard (payload == List.replicate 4 (Address.literal 0))
    return {s with effects := .clear address :: s.effects}
  else if op == "atom.global.gpu.relaxed.add.noftz.v8.f16" then
    let discard := args.take 8
    let addr :: words := args.drop 8 | none
    guard (words.length == 8)
    let (.target, address) ← ptr s.regs addr | none
    let regs := discard.foldl (fun regs dst => regs.insert dst .dead) s.regs
    if !run then return {s with regs}
    let mut effects := s.effects
    for (arg, i) in words.zipIdx do
      let .half source ← number s.regs arg | none
      effects := .atom (address.plus (Address.literal (2 * i : Nat))) source :: effects
    return {regs, effects}
  else if op == "atom.global.gpu.relaxed.max.s32" then
    let [discard, addr, ticket] := args | none
    let (.ticket, address) ← ptr s.regs addr | none
    let regs := s.regs.insert discard .dead
    if !run then return {s with regs}
    return {regs, effects := .mark address (← value s.regs ticket) :: s.effects}
  else none

private def integer (address : Address) (isSigned : Bool) : Option Reg := do
  guard (if isSigned then address.signed32 else address.unsigned32)
  return .num address

private def arithmetic (regs : Regs) (thread : Nat) (live : Bool) (op : String)
    (operands : List String) : Option Regs := do
  if op == "mov.b32" || op == "mov.u32" || op == "mov.u16" then
    match operands with
    | [dst, src] => return regs.insert dst (← number regs src)
    | [lo, hi, src] => match ← number regs src with
      | .packed address => return (regs.insert lo (.half address)).insert hi (.half (address.plus (Address.literal 2)))
      | .dead => return (regs.insert lo .dead).insert hi .dead
      | _ => none
    | _ => none
  else
    let dst :: args := operands | none
    let result : Reg ← match op, args with
      | "ld.param.b64", [src] => parameter true src
      | "ld.param.b32", [src] => parameter false src
      | "and.b32", [a, b] => do
        let x ← (← value regs a).literal?
        let y ← (← value regs b).literal?
        pure (.num (.literal (x &&& y)))
      | "bfe.u32", [a, b, c] => do
        let n ← (← value regs a).literal?
        let start ← (← value regs b).literal?
        let width ← (← value regs c).literal?
        guard (start < 32 && start + width ≤ 32)
        pure (.num (.literal ((n >>> start) % 2 ^ width)))
      | "shl.b32", [a, b] => do
        let n ← (← value regs b).literal?
        guard (n < 32)
        integer ((← value regs a).times (2 ^ n)) false
      | "or.b32", [a, b] => do
        let x ← (← value regs a).literal?
        let y ← value regs b
        guard (x < 4 && y == {block := 4})
        integer (y.plus (.literal x)) false
      | "setp.lt.s32", [a, b] | "setp.ge.s32", [a, b] => do
        guard ((← value regs a) == slot thread && (← value regs b) == {count := 1})
        pure (.pred (if op == "setp.lt.s32" then live else !live) true)
      | "setp.eq.b32", [a, b] => do
        pure (.pred ((← (← value regs a).literal?) == (← (← value regs b).literal?)) false)
      | "setp.ne.s32", [a, b] => do
        pure (.pred ((← (← value regs a).literal?) != (← (← value regs b).literal?)) false)
      | "and.pred", [a, b] => do
        pure (.pred ((← predicate regs a) && (← predicate regs b)) false)
      | "selp.b32", [a, b, choose] => do
        number regs (if ← predicate regs choose then a else b)
      | "mad.wide.s32", [a, b, base] => do
        let .ptr port address ← number regs base | none
        let operand ← value regs a
        let n ← (← value regs b).literal?
        guard (operand.signed32 && (Address.literal n).signed32)
        pure (.ptr port (address.plus (operand.times n)))
      | "mad.lo.s32", [a, scale, b] => do
        let x ← value regs a
        let n ← (← value regs scale).literal?
        let y ← value regs b
        guard (x.signed32 && (Address.literal n).signed32 && y.signed32)
        integer ((x.times n).plus y) true
      | "mul.wide.s32", [a, b] | "mul.wide.u32", [a, b] => do
        let operand ← value regs a
        let multiplier ← value regs b
        let n ← multiplier.literal?
        let signed := op == "mul.wide.s32"
        guard (if signed then operand.signed32 && multiplier.signed32
          else operand.unsigned32 && multiplier.unsigned32)
        pure (.num (operand.times n))
      | "sub.s32", [a, b] => do
        integer ((← value regs a).plus ((← value regs b).times (-1))) true
      | "add.s32", [a, b] | "add.s64", [a, b] => do
        let step ← value regs b
        match ← number regs a with
        | .num x => if op == "add.s32" then integer (x.plus step) true else pure (.num (x.plus step))
        | .ptr port x => pure (.ptr port (x.plus step))
        | _ => none
      | _, _ => none
    return regs.insert dst result

private structure Instruction where
  raw : String
  when : Option String
  op : String
  args : List String

private def parseInstruction (raw : String) : Option Instruction := do
  let line := raw.replace "[ " "[" |>.replace " ]" "]" |>.replace " + " "+"
  let words := TrainingCEGatherBinding.operands line
  let first :: rest := words | none
  let (when, tokens) := if first.startsWith "@" then (some (first.drop 1).toString, rest)
    else (none, words)
  let op :: args := tokens | none
  return ⟨raw, when, op, args⟩

private def instruction (s : State) (thread : Nat) (live : Bool) (line : Instruction) : Option State := do
  let enabled ← match line.when with
    | some name => predicate s.regs name
    | none => some true
  if line.op.startsWith "ld.global." || line.op.startsWith "st.global." || line.op.startsWith "atom.global." then
    memory s line.op line.args enabled
  else
    guard line.when.isNone
    return {s with regs := ← arithmetic s.regs thread live line.op line.args}

private def header (lines : List String) : Bool :=
  let pointer := ".u64 .ptr .global .align 1"
  let kinds := [pointer, pointer, pointer, ".u32", ".u32", ".u32", pointer, pointer, pointer]
  let params := kinds.zipIdx.map fun (kind, i) =>
    s!".param {kind} local_rows_param_{i}" ++ (if i < 8 then "," else "")
  lines.filter (·.startsWith ".target ") == [".target sm_90a"] &&
  lines.filter (·.startsWith ".address_size ") == [".address_size 64"] &&
  lines.filter (·.startsWith ".visible .entry ") == [".visible .entry local_rows("] &&
  lines.filter (·.startsWith ".reqntid ") == [".reqntid 128"] &&
  lines.filter (·.startsWith ".param ") == params

private def code (source : String) : Except String (List Instruction) := do
  let lines := QKVTTIR.normalizePTX source
  if !header lines then throw "Unexpected local-row launch signature"
  let ignored (line : String) := line.startsWith ".version " || line.startsWith ".target " ||
    line.startsWith ".address_size " || line.startsWith ".visible .entry " ||
    line.startsWith ".param " || line.startsWith ".reqntid " || line.startsWith ".reg " ||
    (line.startsWith "$L__" && line.endsWith ":") || ["(", ")", "{", "}"].contains line
  let "ret;" :: reversed := (lines.filter (!ignored ·)).reverse | throw "Unexpected local-row return"
  reversed.reverse.mapM fun raw =>
    match parseInstruction raw with
    | some insn => .ok insn
    | none => .error s!"Unexpected instruction: {raw}"

private def execute (body : List Instruction) (thread : Nat) (live : Bool) : Except String (List Effect) := do
  let regs : Regs := ({} : Regs).insert "%tid.x" (.num (.literal thread)) |>.insert "%ctaid.x" (.num {block := 1})
  let mut s : State := {regs}
  let mut jumped := false
  let mut sawBranch := false
  let mut sawDestination := false
  for line in body do
    if line.op == "L__local_rows_end:" then
      if !sawBranch || sawDestination || line.when.isSome || !line.args.isEmpty then
        throw "The local scatter exit label is misplaced"
      jumped := false
      sawDestination := true
    else if line.op == "bra.uni" then
      if sawBranch || line.args != ["L__local_rows_end"] then
        throw "Unexpected local scatter branch destination"
      let some name := line.when | throw "Missing local scatter branch predicate"
      let some (.pred test true) := s.regs.get? name | throw "Branch predicate is not warp-uniform"
      jumped := test
      sawBranch := true
    else if !jumped then
      let some next := instruction s thread live line | throw s!"Unsupported operation for thread {thread}: {line.raw}"
      s := next
  if sawBranch != sawDestination then throw "Missing local scatter exit label"
  return s.effects.reverse

def expected (direct marked : Bool) (thread : Nat) (live : Bool) : List Effect :=
  if !live then [] else
    let row := {first := -1, row := 1 : Address}
    let index := (slot thread).times 4 |>.plus {offset := 4}
    let stripes := (List.range 3).flatMap fun i =>
      let byte := Address.literal (thread % 32 * 16 + i * 512 : Nat)
      let source := (if direct then row else slot thread).times 1536 |>.plus byte
      let target := row.times 1536 |>.plus byte
      let atoms := (List.range 8).map fun j =>
        let delta := Address.literal (2 * j : Nat)
        Effect.atom (target.plus delta) (source.plus delta)
      Effect.read source :: atoms ++ (if direct then [.clear source] else [])
    let marker := if marked && thread % 32 == 0 then [.mark (row.times 4) ((slot thread).plus (.literal 1))] else []
    .lookup index :: stripes ++ marker

def diagnose (direct marked : Bool) (source : String) : Except String Unit := do
  let body ← code source
  for thread in [:128] do
    for live in [false, true] do
      let results ← execute body thread live
      if results != expected direct marked thread live then
        throw s!"Different memory effects for thread {thread}, active={live}"

private def matched (direct marked : Bool) (body : List Instruction) : Bool :=
  (List.range 128).all fun thread => [false, true].all fun live =>
    decide (execute body thread live = .ok (expected direct marked thread live))

structure Certificate (direct marked : Bool) (source : String) where
  private mk ::
  private body : List Instruction
  private sourceCode : code source = .ok body
  private effectsMatch : matched direct marked body = true

def certify (direct marked : Bool) (source : String) : Option (Certificate direct marked source) :=
  match parsed : code source with
  | .error _ => none
  | .ok body => if exact : matched direct marked body = true then some ⟨body, parsed, exact⟩ else none

def artifactEffects (source : String) (thread : Nat) (live : Bool) : Except String (List Effect) := do
  let body ← code source
  execute body thread live

theorem Certificate.effects_exact {direct marked : Bool} {source : String}
    (checked : Certificate direct marked source) (thread : Fin 128) (live : Bool) :
    artifactEffects source thread.val live = .ok (expected direct marked thread.val live) := by
  have every := List.all_eq_true.mp checked.effectsMatch thread.val (List.mem_range.mpr thread.isLt)
  have member : live ∈ [false, true] := by cases live <;> decide
  have exact := List.all_eq_true.mp every live member
  unfold artifactEffects
  rw [checked.sourceCode]
  exact of_decide_eq_true exact

#axiomsClean Certificate.effects_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.LocalRows
