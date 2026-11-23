import models.NanoGPT.Track1.ValueAdamElementwise
import models.NanoGPT.Track1.TrainingCEPublishBinding
import models.NanoGPT.Track1.ValueAdamConsumedWords
import models.NanoGPT.Track1.ValueAdamHalfProductPTX
import models.NanoGPT.Track1.ValueAdamHalfProductExecution

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise

structure Profile where
  period : Period
  tile : Tile
  fused : Bool

def Profile.threads (profile : Profile) : Nat :=
  match profile.tile with
  | .x512 => 256
  | .x1024 => 128

def Profile.sharedBytes (profile : Profile) : Nat :=
  if profile.fused then 0
  else match profile.tile with
  | .x512 => 0
  | .x1024 => 4096

def Profile.metadataMatches (profile : Profile) (raw : String) : Bool :=
  let parsed : Except String Bool := do
    let metadata ← Lean.Json.parse raw
    let target ← metadata.getObjVal? "target"
    return (← target.getObjValAs? String "backend") == "cuda" &&
      (← target.getObjValAs? Nat "arch") == 90 &&
      (← target.getObjValAs? Nat "warp_size") == 32 &&
      (← metadata.getObjValAs? Nat "num_warps") * 32 == profile.threads &&
      (← metadata.getObjValAs? Nat "shared") == profile.sharedBytes &&
      (← metadata.getObjValAs? Nat "num_ctas") == 1 &&
      (← metadata.getObjValAs? Nat "num_stages") == 1 &&
      (← metadata.getObjValAs? Bool "enable_fp_fusion") == !profile.fused
  parsed.toOption.getD false

def Profile.blocks (profile : Profile) : Nat :=
  25152 * 768 / profile.tile.width

private def Profile.entry (profile : Profile) : String :=
  if profile.fused then "_update_with_scaled_read"
  else "triton_poi_fused__to_copy_add_copy__div_gt_mul_sqrt_0"

private def Profile.arguments (profile : Profile) : Array Port :=
  if profile.fused then #[.first, .gradient, .second, .step, .weight, .decay, .first, .second, .weight]
  else #[.weight, .first, .gradient, .second, .step, .decay, .first, .second, .weight]

private def Port.bytes : Port → Nat
  | .gradient | .weight => 2
  | _ => 4

private def Port.scalar : Port → Bool
  | .step | .decay => true
  | _ => false

private def Port.mutable : Port → Bool
  | .weight | .first | .second => true
  | _ => false

/-- Every accepted global index is expressed as a byte offset within a
typed tensor port, with one affine coefficient for the CTA index. -/
private structure Affine where
  slope : Nat
  offset : Nat

private def Affine.plus (a b : Affine) : Affine :=
  ⟨a.slope + b.slope, a.offset + b.offset⟩

private def Affine.times (a : Affine) (k : Nat) : Affine :=
  ⟨a.slope * k, a.offset * k⟩

private def Affine.last (a : Affine) (profile : Profile) : Nat :=
  a.slope * (profile.blocks - 1) + a.offset

private inductive Reg where
  | number (value : Affine)
  | global (port : Port) (offset : Affine)
  | shared (offset : Nat)
  | term (value : Term)
  | pair (low high : Term)
  | bits16 (words : List Nat)
  | bits32 (lo hi : List Nat)
  | dirty (words : List Nat)
  deriving Inhabited

private abbrev Registers := Std.HashMap String Reg
private abbrev StoresByAddress := Std.HashMap (Port × Nat) Term

private abbrev ClearsByIndex := Std.HashMap Nat ClearWrite

private structure Decoded where
  stores : StoresByAddress
  clears : ClearsByIndex

private structure Machine where
  threads : Array Registers
  stores : StoresByAddress := {}
  clears : ClearsByIndex := {}
  shared : Std.HashMap Nat Reg := {}
  pending : Std.HashMap Nat Reg := {}
  sharedRead : Bool := false

private def read (regs : Registers) (name : String) : Option Reg :=
  if name == "global_smem" then some (.shared 0)
  else if name == "0x0" then some (.number ⟨0, 0⟩)
  else (regs.get? name).orElse fun _ =>
    (name.toNat?.map (fun n => .number ⟨0, n⟩)).orElse fun _ => do
      let .literal bits ← TrainingCEPublishBinding.getExpr [] name | none
      return .term (.constant bits)

private def number (regs : Registers) (name : String) : Option Affine := do
  let .number value ← read regs name | none
  return value

private def constant (regs : Registers) (name : String) : Option Nat := do
  let value ← number regs name
  guard (value.slope == 0)
  return value.offset

private def term (regs : Registers) (name : String) : Option Term := do
  let .term value ← read regs name | none
  return value

private def move (regs : Registers) (tokens : List String) (unpack : Bool) : Option Registers := do
  match tokens, unpack with
  | [dst, src], false => return regs.insert dst (← read regs src)
  | [lo, hi, src], true =>
      match ← read regs src with
      | .pair low high => return (regs.insert lo (.term low)).insert hi (.term high)
      | .bits32 low high => return (regs.insert lo (.bits16 low)).insert hi (.bits16 high)
      | _ => none
  | [dst, lo, hi], false => return regs.insert dst (.pair (← term regs lo) (← term regs hi))
  | _, _ => none

private def parseAddress (text : String) : Option (String × Nat) := do
  let condensed := text.replace " " ""
  match condensed.splitOn "+" with
  | [name] => return (name, 0)
  | [name, offset] => return (name, ← offset.toNat?)
  | _ => none

private def portIndex (profile : Profile) (port : Port) (offset : Affine) : Option Nat := do
  guard (!port.scalar && offset.slope == port.bytes * profile.tile.width &&
    offset.offset % port.bytes == 0 && offset.offset < port.bytes * profile.tile.width)
  return offset.offset / port.bytes

private def vectorWidth : String → Option Nat
  | "b32" => some 1
  | "v2.b32" => some 2
  | "v4.b32" => some 4
  | _ => none

private def operandTokens (text : String) : List String :=
  TrainingCEGatherBinding.operands text

private structure MemoryInstruction where
  load : Bool
  space : String
  registers : List (String × Nat)
  pointer : String
  displacement : Nat
  predicate : Option String

private inductive Instruction where
  | memory (operation : MemoryInstruction)
  | scalar (opcode : String) (operands : List String) (unpack : Bool)
  | guardedHalf (pairs : Array GuardedHalf.Registers)
      (exact : ∀ (inputs : Inputs) (slots : Fin 32 → GuardedHalf.Lane Nat) (lane : Fin 32),
        (GuardedHalf.readWarp (fun i => (slots i).map inputs.gradient) lane).map (some ∘ Data.half) =
          ((List.finRange 32).flatMap (GuardedHalf.Lane.words ∘ slots)).map
            (fun index => (GuardedHalf.scalarTerm index).eval inputs))
  | barrier

private def parseInstruction (line : String) : Option Instruction := do
  if line == "bar.sync 0;" then return .barrier
  let first := (line.splitOn " ").headD ""
  let (pred, line) := if first.startsWith "@" then
    (some (first.drop 1).toString, (line.drop (first.length + 1)).toString)
    else (none, line)
  let op := (line.splitOn " ").headD ""
  let text := (line.drop (op.length + 1)).toString
  let parts := op.splitOn "."
  match parts with
  | action :: space :: kind =>
    if (action == "ld" || action == "st") && (space == "global" || space == "shared") then
      let count ← vectorWidth (String.intercalate "." kind)
      let [pre, rest] := text.splitOn "[" | none
      let [ptr, post] := rest.splitOn "]" | none
      let load := action == "ld"
      guard (pred.isNone || (!load && space == "global"))
      guard (if load then (operandTokens post).isEmpty else (operandTokens pre).isEmpty)
      let registers := operandTokens (if load then pre else post)
      guard (registers.length == count)
      let (pointer, displacement) ← parseAddress ptr
      return .memory ⟨load, space, registers.zipIdx, pointer, displacement, pred⟩
    else
      guard pred.isNone
      return .scalar op (operandTokens text) (text.startsWith "{")
  | _ =>
    guard pred.isNone
    return .scalar op (operandTokens text) (text.startsWith "{")

private def memory (profile : Profile) (regs : Registers) (m : Machine)
    (op : MemoryInstruction) : Option (Registers × Machine) := do
  let start ← read regs op.pointer
  let align := 4 * op.registers.length
  let mut regs := regs
  let mut state := m
  match start, op.space with
  | .global port base, "global" =>
    let start := base.plus ⟨0, op.displacement⟩
    guard (start.slope % align == 0 && start.offset % align == 0)
    if op.load then
      guard m.stores.isEmpty
    else
      guard (port.mutable || (profile.fused && port == .gradient))
    let clearing : ClearWrite ← match op.predicate with
      | none => some ClearWrite.always
      | some predicate => do
          guard (port == .gradient)
          let .dirty words ← read regs predicate | none
          pure (.ifDirty words)
    for (name, i) in op.registers do
      let byte := start.plus ⟨0, 4 * i⟩
      let index ← portIndex profile port byte
      if op.load then
        let value ← if port.bytes == 2 then do
          let next ← portIndex profile port (byte.plus ⟨0, 2⟩)
          guard (port != .gradient || (!state.clears.contains index && !state.clears.contains next))
          pure (.pair (.input port index) (.input port next))
        else pure (.term (.input port index))
        regs := regs.insert name value
      else if port == .gradient then
        let next ← portIndex profile port (byte.plus ⟨0, 2⟩)
        let payload ← constant regs name
        guard (payload == 0 && !state.clears.contains index && !state.clears.contains next)
        state := {state with clears := (state.clears.insert index clearing).insert next clearing}
      else if port.bytes == 2 then
        let .pair lo hi ← read regs name | none
        let next ← portIndex profile port (byte.plus ⟨0, 2⟩)
        guard (!state.stores.contains (port, index) && !state.stores.contains (port, next))
        state := {state with stores := (state.stores.insert (port, index) lo).insert (port, next) hi}
      else
        guard (!state.stores.contains (port, index))
        state := {state with stores := state.stores.insert (port, index) (← term regs name)}
  | .shared base, "shared" =>
    let start := base + op.displacement
    guard (start % align == 0 && start + align ≤ profile.sharedBytes)
    for (name, i) in op.registers do
      let byte := start + 4 * i
      if op.load then
        guard (!state.pending.contains byte)
        regs := regs.insert name (← state.shared[byte]?)
        state := {state with sharedRead := true}
      else
        guard (!state.sharedRead && !state.pending.contains byte)
        state := {state with pending := state.pending.insert byte (← read regs name)}
  | _, _ => none
  return (regs, state)

private def constantBinary (opcode : String) (a b : UInt32) : Option UInt32 :=
  match opcode with
  | "add.s32" => some (a + b)
  | "and.b32" => some (a &&& b)
  | "xor.b32" => some (a ^^^ b)
  | "or.b32" => some (a ||| b)
  | _ => none

private def halfSupport : Reg → Option (List Nat)
  | .term (.input .gradient i) => some [i]
  | .bits16 words => some words
  | .number ⟨0, 0⟩ => some []
  | _ => none

private def packedSupport : Reg → Option (List Nat × List Nat)
  | .pair lo hi => do
    return (← halfSupport (.term lo), ← halfSupport (.term hi))
  | .bits32 lo hi => some (lo, hi)
  | .number ⟨0, 0⟩ => some ([], [])
  | _ => none

private def packedBitwiseOr (a b : Reg) : Option Reg := do
  let (loA, hiA) ← packedSupport a
  let (loB, hiB) ← packedSupport b
  return .bits32 (loA ++ loB) (hiA ++ hiB)

private def integerBinary (profile : Profile) (opcode : String) (a b : Reg) : Option Reg := do
  match a, b with
  | .global port offset, .number n | .number n, .global port offset =>
    guard (opcode == "add.s64")
    return .global port (offset.plus n)
  | .shared offset, .number n | .number n, .shared offset =>
    guard (opcode == "add.s32" && n.slope == 0 && offset + n.offset < 2^31)
    return .shared (offset + n.offset)
  | .number a, .number b =>
    if opcode == "add.s64" then return .number (a.plus b)
    if opcode == "mul.wide.s32" then
      guard (a.last profile < 2^31 && b.slope == 0 && b.offset < 2^31)
      return .number (a.times b.offset)
    if opcode == "shl.b32" then
      guard (b.slope == 0 && b.offset < 32)
      let n := a.times (2^b.offset)
      guard (n.last profile < 2^32)
      return .number n
    if a.slope == 0 && b.slope == 0 then
      let value ← constantBinary opcode (UInt32.ofNat a.offset) (UInt32.ofNat b.offset)
      return .number ⟨0, value.toNat⟩
    let (hi, lo) := if a.slope == 0 then (b, a) else (a, b)
    guard (opcode == "or.b32" && hi.slope == profile.tile.width && hi.offset == 0 &&
      lo.slope == 0 && lo.offset < profile.tile.width && (hi.plus lo).last profile < 2^32)
    return .number (hi.plus lo)
  | _, _ =>
    if opcode == "or.b32" then packedBitwiseOr a b
    else if opcode == "or.b16" then do
      return .bits16 ((← halfSupport a) ++ (← halfSupport b))
    else none

private def unaryFloat (opcode : String) (a : Term) : Option Term :=
  match opcode with
  | "cvt.f32.f16" => some (.widenFP16 a)
  | "cvt.f32.bf16" => some (.widenBF16 a)
  | "cvt.rn.f16.f32" => some (.narrowFP16 a)
  | "sqrt.rn.f32" => some (.sqrt a)
  | _ => none

private def binaryFloat (opcode : String) (a b : Term) : Option Reg :=
  match opcode with
  | "add.f32" | "add.rn.f32" => some (.term (.add a b))
  | "sub.f32" | "sub.rn.f32" => some (.term (.sub a b))
  | "mul.f32" | "mul.rn.f32" => some (.term (.mul a b))
  | "div.full.f32" => some (.term (.divideFull a b))
  | "setp.gt.f32" => some (.term (.greater a b))
  | "cvt.rn.bf16x2.f32" => some (.pair (.narrowBF16 b) (.narrowBF16 a))
  | _ => none

private def parameter (profile : Profile) (name : String) (byValue : Bool) : Option Reg := do
  let paramStem := "[" ++ profile.entry ++ "_param_"
  guard (name.startsWith paramStem && name.endsWith "]")
  let index ← ((name.drop paramStem.length).dropEnd 1).toString.toNat?
  let port ← profile.arguments[index]?
  guard (port.scalar == byValue)
  if byValue then return .term (.input port 0)
  else return .global port ⟨0, 0⟩

private def compute (profile : Profile) (regs : Registers) (opcode : String)
    (tokens : List String) (unpack : Bool) : Option Registers := do
  if opcode == "mov.b32" || opcode == "mov.u32" then
    return ← move regs tokens unpack
  let dst :: args := tokens | none
  let result : Reg ← match opcode, args with
    | "ld.param.b64", [name] => parameter profile name false
    | "ld.param.b32", [name] => parameter profile name true
    | "neg.s32", [a] => do
        let n ← constant regs a
        pure (.number ⟨0, (0 - UInt32.ofNat n).toNat⟩)
    | _, [a] => do
        pure (.term (← unaryFloat opcode (← term regs a)))
    | "setp.ne.b16", [a, b] => do
        guard ((← constant regs b) == 0)
        pure (.dirty (← halfSupport (← read regs a)))
    | "setp.ne.b32", [a, b] => do
        guard ((← constant regs b) == 0)
        let (lo, hi) ← packedSupport (← read regs a)
        pure (.dirty (lo ++ hi))
    | "lop3.b32", [a, b, c, "0xfe"] => do
        packedBitwiseOr (← packedBitwiseOr (← read regs a) (← read regs b)) (← read regs c)
    | _, [a, b] =>
        if opcode.endsWith ".f32" then binaryFloat opcode (← term regs a) (← term regs b)
        else integerBinary profile opcode (← read regs a) (← read regs b)
    | "fma.rn.f32", [a, b, c] => do
        pure (.term (.fma (← term regs a) (← term regs b) (← term regs c)))
    | "selp.f32", [a, b, c] => do
        pure (.term (.select (← term regs a) (← term regs b) (← term regs c)))
    | _, _ => none
  return regs.insert dst result

private def executeLine (profile : Profile) (m : Machine) : Instruction → Option Machine
  | .barrier =>
    let published := if m.pending.isEmpty then m.shared else m.pending
    some {m with shared := published, pending := {}, sharedRead := false}
  | .memory op => do
    let mut state := m
    for i in [:m.threads.size] do
      let (regs, updated) ← memory profile state.threads[i]! state op
      state := {updated with threads := updated.threads.set! i regs}
    return state
  | .guardedHalf pairs _ => do
    let .x1024 := profile.tile | none
    guard (profile.fused && m.threads.size % 32 == 0 && pairs.size == 4)
    let mut threads := m.threads
    for i in [:threads.size] do
      let input := m.threads[i]!
      let mut regs := input
      for pair in pairs do
        let .pair (.input .gradient lo) (.input .gradient hi) ← read input pair.input | none
        regs := regs.insert pair.output (.pair (GuardedHalf.scalarTerm lo) (GuardedHalf.scalarTerm hi))
      threads := threads.set! i regs
    return {m with threads}
  | .scalar opcode operands unpack => do
    let mut threads := m.threads
    for i in [:threads.size] do
      threads := threads.set! i (← compute profile threads[i]! opcode operands unpack)
    return {m with threads}

private def metadata (line : String) : Bool :=
  line.startsWith ".version " || line.startsWith ".target " || line.startsWith ".address_size " ||
  line.startsWith ".visible .entry " || line.startsWith ".param " || line.startsWith ".reqntid " ||
  line.startsWith ".reg " || line == ".extern .shared .align 16 .b8 global_smem[];" ||
  ["(", ")", "{", "}"].contains line || (line.startsWith "$" && line.endsWith ":")

private def declarations (profile : Profile) : List String :=
  let pointer := ".u64 .ptr .global .align 1"
  let data := profile.arguments.toList.map fun port => if port.scalar then ".f32" else pointer
  let control := if profile.fused then [pointer, pointer] else [".u32", pointer, pointer]
  let types := data ++ control
  types.zipIdx.map fun (kind, index) =>
    s!".param {kind} {profile.entry}_param_{index}" ++ (if index + 1 < types.length then "," else "")

private abbrev Code := String ⊕ Array GuardedHalf.Registers

private def parsePTX (profile : Profile) (source : String) : Option (List Code) := do
  let lines := QKVTTIR.normalizePTX source
  guard ((lines.filter (·.startsWith ".target ")) == [".target sm_90a"] &&
    (lines.filter (·.startsWith ".address_size ")) == [".address_size 64"] &&
    (lines.filter (·.startsWith ".visible .entry ")) == [".visible .entry " ++ profile.entry ++ "("] &&
    (lines.filter (·.startsWith ".reqntid ")) == [s!".reqntid {profile.threads}"] &&
    (lines.filter (·.startsWith ".param ")) == declarations profile)
  let strip (lines : List String) : List String := lines.filter (!metadata ·)
  if lines.contains GuardedHalf.opener then
    let (pre, pairs, post) ← GuardedHalf.extract lines
    let "ret;" :: revPost := (strip post).reverse | none
    return (strip pre).map Sum.inl ++ [Sum.inr pairs] ++ revPost.reverse.map Sum.inl
  else
    let "ret;" :: revCode := (strip lines).reverse | none
    return revCode.reverse.map Sum.inl

private def decodeResult (profile : Profile) (source : String) : Except String Decoded := do
  let some code := parsePTX profile source | throw "Unexpected entry, target, launch signature or return"
  let threads := (List.range profile.threads).toArray.map fun thread =>
    ({} : Registers).insert "%tid.x" (.number ⟨0, thread⟩) |>.insert "%ctaid.x" (.number ⟨1, 0⟩)
  let mut machine : Machine := {threads}
  for line in code do
    match line with
    | .inl text =>
      let some op := parseInstruction text | throw s!"Unexpected instruction syntax: {text}"
      let some next := executeLine profile machine op | throw s!"Unsupported lane effect: {text}"
      machine := next
    | .inr pairs =>
      let some next := executeLine profile machine (.guardedHalf pairs GuardedHalf.read_warp_terms_exact) |
        throw "Unexpected operands or warp geometry in guarded binary16 scale"
      machine := next
  if !machine.pending.isEmpty then
    throw "Unpublished scratch writes"
  return ⟨machine.stores, machine.clears⟩

private def decode (profile : Profile) (source : String) : Option Decoded :=
  (decodeResult profile source).toOption

private def observe (stores : StoresByAddress) (index : Nat) : Option Stores := do
  return ⟨← stores.get? (Port.weight, index), ← stores.get? (Port.first, index), ← stores.get? (Port.second, index)⟩

private def storesMatch (profile : Profile) (stores : StoresByAddress) : Bool :=
  stores.size == 3 * profile.tile.width &&
    (List.range profile.tile.width).all fun i =>
      decide (observe stores i = some (expected profile.period profile.tile profile.fused i))

private def clearLaneMatches (clears : ClearsByIndex) (index : Nat) : Bool :=
  match clears.get? index with
  | none => false
  | some write => decide (write.covers index)

private def fullClearMatches (profile : Profile) (clears : ClearsByIndex) : Bool :=
  clears.size == profile.tile.width &&
    (List.range profile.tile.width).all (clearLaneMatches clears)

private def clearsMatch (profile : Profile) (clears : ClearsByIndex) : Bool :=
  clears.isEmpty || (profile.fused && fullClearMatches profile clears)

def diagnose (profile : Profile) (source : String) : String :=
  match decodeResult profile source with
  | .error cause => cause
  | .ok output =>
    if !storesMatch profile output.stores then "Decoded output terms differ"
    else if !clearsMatch profile output.clears then "Clearing predicate or index differs"
    else "Accepted"

structure Certificate (profile : Profile) (source : String) where
  private mk ::
  private output : Decoded
  private decoded : decode profile source = some output
  private matched : storesMatch profile output.stores = true
  private clearMatched : clearsMatch profile output.clears = true

def certify (profile : Profile) (source : String) : Option (Certificate profile source) :=
  match h₁ : decode profile source with
  | none => none
  | some output =>
    if h₂ : storesMatch profile output.stores = true then
      if h₃ : clearsMatch profile output.clears = true then some ⟨output, h₁, h₂, h₃⟩ else none
    else none

def Certificate.consumes (checked : Certificate profile source) : Bool :=
  !checked.output.clears.isEmpty

def Certificate.modeMatches (checked : Certificate profile source) (clearAll : Bool) : Bool :=
  checked.consumes &&
    (List.range profile.tile.width).all fun index =>
      match checked.output.clears.get? index with
      | some .always => clearAll
      | some (.ifDirty _) => !clearAll
      | none => false

theorem Certificate.mode_consumes {profile source} (checked : Certificate profile source)
    (clearAll : Bool) (matched : checked.modeMatches clearAll = true) : checked.consumes = true := by
  simp only [Certificate.modeMatches, Bool.and_eq_true] at matched
  exact matched.1

def Certificate.gradientAfter (checked : Certificate profile source) (input : Nat → Half)
    (index : Nat) : Half :=
  match checked.output.clears.get? index with
  | none => input index
  | some write => write.apply input index

theorem Certificate.gradient_after_zero {profile source} (checked : Certificate profile source)
    (consumes : checked.consumes = true) (input : Nat → Half) (index : Fin profile.tile.width) :
    checked.gradientAfter input index.val = 0 := by
  have nonempty : checked.output.clears.isEmpty = false := by
    cases h : checked.output.clears.isEmpty
    · rfl
    · simp [Certificate.consumes, h] at consumes
  have matched := checked.clearMatched
  simp only [clearsMatch, nonempty, Bool.false_or, Bool.and_eq_true] at matched
  have every := matched.2
  simp only [fullClearMatches, Bool.and_eq_true] at every
  have lane := List.all_eq_true.mp every.2 index.val (List.mem_range.mpr index.isLt)
  cases found : checked.output.clears.get? index.val with
  | none =>
    rw [clearLaneMatches, found] at lane
    contradiction
  | some write =>
    rw [clearLaneMatches, found] at lane
    have covered : write.covers index.val := of_decide_eq_true lane
    rw [Certificate.gradientAfter, found]
    exact ClearWrite.zeros write input index.val covered

def Certificate.update (checked : Certificate profile source) (input : Inputs) (index : Nat) :
    Option Cell :=
  (observe checked.output.stores index).bind (Stores.eval input)

theorem Certificate.update_eq {profile source} (checked : Certificate profile source)
    (input : Inputs) (index : Fin profile.tile.width) :
    checked.update input index = (expected profile.period profile.tile profile.fused index).eval input := by
  have h := checked.matched
  simp only [storesMatch, Bool.and_eq_true] at h
  have same := List.all_eq_true.mp h.2 index.val (List.mem_range.mpr index.isLt)
  have eq := of_decide_eq_true same
  simp [Certificate.update, eq]

theorem native_read_scale_eq (period : Period) (tile : Tile) (reference fused : String)
    (r : Certificate ⟨period, tile, false⟩ reference) (q : Certificate ⟨period, tile, true⟩ fused)
    (input : Inputs) (i : Fin tile.width) :
    q.update input i = r.update {input with gradient := fun j => scaleRead (input.gradient j)} i := by
  rw [q.update_eq input i, r.update_eq _ i]
  exact read_scale_expected input period tile i

#axiomsClean Certificate.gradient_after_zero
#axiomsClean Certificate.update_eq
#axiomsClean native_read_scale_eq

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise
