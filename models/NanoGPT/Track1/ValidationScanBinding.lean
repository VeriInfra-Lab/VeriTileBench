import models.NanoGPT.Track1.ValidationReduceBinding

/-! Bind the BF16 projection and canonical mask used by the online
validation update. Replay packed loads and shared transposes over every
lane. The four-column tile uses two iteration phases, so the mask byte
address and bit selector are affine or constant on each phase. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValidationScanBinding
open TrainingCEGatherBinding
open ValidationCE (Tile)
open ValidationReduceBinding (Instruction Slot inputSlot rowCount)

structure Affine where
  fixed : Int
  block : Int := 0
  columnBase : Int := 0
  deriving DecidableEq, Repr

def Affine.eval (a : Affine) (block columnBase : Nat) : Int :=
  a.fixed + a.block * block + a.columnBase * columnBase

private def Affine.add (a b : Affine) : Affine :=
  ⟨a.fixed + b.fixed, a.block + b.block, a.columnBase + b.columnBase⟩

private def Affine.scale (a : Affine) (n : Int) : Affine :=
  ⟨a.fixed * n, a.block * n, a.columnBase * n⟩

private def Affine.constant (a : Affine) : Option Int :=
  if a.block == 0 && a.columnBase == 0 then some a.fixed else none

structure Address where
  param : Nat
  byte : Affine
  deriving DecidableEq, Repr

def projectionAddress (tile : Tile) (slot : Slot) : Address :=
  ⟨2 + slot.slab, ⟨2 * (slot.row * 50304 + slot.column), 2 * (rowCount tile * 50304), 2⟩⟩

theorem projection_byte_exact (tile : Tile) (slot : Slot) (block iteration : Nat) :
    (projectionAddress tile slot).byte.eval block (tile.extent * iteration) =
      (2 * ((block * rowCount tile + slot.row) * 50304 +
        ValidationCE.scanColumn tile slot.column iteration) : Int) := by
  cases tile <;> simp [projectionAddress, Affine.eval, ValidationCE.scanColumn, rowCount, Tile.extent] <;> omega

inductive Bound where
  | always
  | below (index : Affine) (limit : Nat)
  deriving DecidableEq

private structure Domain where
  tile : Tile
  phase : Nat

private def phaseCount : Tile → Nat
  | .x64r4 => 2
  | _ => 1

private def Domain.first (d : Domain) : Nat := d.tile.extent * d.phase

private def Domain.step (d : Domain) : Nat := d.tile.extent * phaseCount d.tile

private def Domain.steps (d : Domain) : Nat :=
  TailNorm.ceilDiv (TailNorm.ceilDiv 50304 d.tile.extent - d.phase) (phaseCount d.tile)

private def Domain.specialize (d : Domain) (a : Affine) : Affine :=
  ⟨a.fixed + a.columnBase * d.first, a.block, a.columnBase * d.step⟩

private def Domain.interval (d : Domain) (a : Affine) : Int × Int :=
  let blockMax : Int := (ValidationCE.slabRows / rowCount d.tile - 1 : Nat)
  let stepMax : Int := (d.steps - 1 : Nat)
  (a.fixed + min 0 (a.block * blockMax) + min 0 (a.columnBase * stepMax),
   a.fixed + max 0 (a.block * blockMax) + max 0 (a.columnBase * stepMax))

private def signed32 (d : Domain) (a : Affine) : Bool :=
  let (lo, hi) := d.interval a
  decide (-2147483648 ≤ lo ∧ hi < 2147483648)

private def unsigned32 (d : Domain) (a : Affine) : Bool :=
  let (lo, hi) := d.interval a
  decide (0 ≤ lo ∧ hi < 4294967296)

private def liveIterations (steps step col limit : Nat) : Nat :=
  if col ≥ limit then 0 else min steps (1 + (limit - 1 - col) / step)

private theorem liveIterations_exact {steps step : Nat} (col limit i : Nat)
    (iteration : i < steps) (positive : 0 < step) :
    i < liveIterations steps step col limit ↔ col + i * step < limit := by
  unfold liveIterations
  split
  · omega
  · rw [Nat.lt_min]
    have quotient : i ≤ (limit - 1 - col) / step ↔ i * step ≤ limit - 1 - col :=
      Nat.le_div_iff_mul_le positive
    omega

private def verifyBound (d : Domain) (firstColumn : Nat) : Bound → Bool
  | .always => liveIterations d.steps d.step firstColumn 50304 == d.steps
  | .below a limit =>
    a.block == 0 && a.columnBase == d.step && 0 ≤ a.fixed &&
      liveIterations d.steps d.step a.fixed.toNat limit == liveIterations d.steps d.step firstColumn 50304

def Bound.active (bound : Bound) (block base : Nat) : Prop :=
  match bound with
  | .always => True
  | .below index limit => index.eval block base < (limit : Int)

private theorem load_guard_exact (d : Domain) (col block i : Nat) (bound : Bound)
    (valid : i < d.steps) (checked : verifyBound d col bound = true) :
    bound.active block i ↔ col + i * d.step < 50304 := by
  have step : 0 < d.step := by
    obtain ⟨tile, phase⟩ := d
    cases tile <;> simp [Domain.step, Tile.extent, phaseCount]
  rw [← liveIterations_exact col 50304 i valid step]
  cases bound with
  | always =>
    simp [verifyBound] at checked
    simp [Bound.active, checked, valid]
  | below index limit =>
    simp only [verifyBound, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at checked
    obtain ⟨⟨⟨blockTerm, baseTerm⟩, fixed⟩, steps⟩ := checked
    rw [← steps, liveIterations_exact index.fixed.toNat limit i valid step]
    have asNat : (index.fixed.toNat : Int) = index.fixed := Int.toNat_of_nonneg fixed
    simp only [Bound.active, Affine.eval, blockTerm, baseTerm, Int.zero_mul, Int.add_zero]
    rw [Int.mul_comm (d.step : Int) (i : Int)]
    omega

private inductive Stage where
  | widened | shifted | affine | log2 | exp | denominator | inverse | scaled
  deriving DecidableEq

structure MaskAddress where
  tokenByte : Affine
  columnByte : Affine
  deriving DecidableEq, Repr

structure MaskBit where
  address : MaskAddress
  index : Nat
  deriving DecidableEq, Repr

structure Score where
  projection : Address
  mask : MaskBit
  deriving DecidableEq, Repr

structure Update where
  score : Score
  liveSteps : Nat
  deriving DecidableEq, Repr

private structure TokenOffset where
  inputByte : Affine
  stride : Int
  offset : Affine

private def TokenOffset.interval (d : Domain) (token : TokenOffset) : Int × Int :=
  let (lo, hi) := d.interval token.offset
  (lo + min 0 (50303 * token.stride), hi + max 0 (50303 * token.stride))

private def TokenOffset.signed32 (d : Domain) (token : TokenOffset) : Bool :=
  let (lo, hi) := token.interval d
  decide (-2147483648 ≤ lo ∧ hi < 2147483648)

private inductive MaskWord where
  | literal (word : Nat)
  | byte (value : MaskAddress)
  | bit (value : MaskBit)

private inductive Reg where
  | number (word : Affine)
  | address (value : Address)
  | pred (test : Bound)
  | literal (bits : Nat)
  | half (origin : Address)
  | pair (lo hi : Option Address)
  | value (kind : Stage) (origin : Address)
  | token (offset : TokenOffset)
  | maskPointer (offset : TokenOffset)
  | maskWord (word : MaskWord)
  | maskPair (lo hi : MaskWord)
  | maskPredicate (bit : MaskBit) (dropWhenTrue : Bool)
  | truth (value : Bool)
  | score (value : Score)

private abbrev Regs := Std.HashMap String Reg

private def lit (name : String) : Option Nat := do
  let .literal bits ← TrainingCEPublishBinding.getExpr [] name | none
  return bits

private def get (regs : Regs) (arg : String) : Option Reg :=
  if arg == "global_smem" || arg == "0x0" then some (.number ⟨0, 0, 0⟩)
  else (regs.get? arg).orElse (fun _ =>
    (arg.toInt?.map (fun i => .number ⟨i, 0, 0⟩)).orElse (fun _ => (lit arg).map .literal))

private def number (regs : Regs) (arg : String) : Option Affine := do
  let .number value ← get regs arg | none
  return value

private def constant (regs : Regs) (arg : String) : Option Int :=
  (number regs arg).bind Affine.constant

private def bytes (regs : Regs) (ptr : String) : Option (Reg × Nat) := do
  guard (ptr.startsWith "[" && ptr.endsWith "]")
  let inner := (ptr.drop 1 |>.dropEnd 1).toString
  match inner.splitOn "+" with
  | [base] => return (← get regs base, 0)
  | [base, offset] => return (← get regs base, ← offset.toNat?)
  | _ => none

private def addReg : Reg → Reg → Option Reg
  | .number a, .number b => some (.number (a.add b))
  | .address a, .number b | .number b, .address a => some (.address ⟨a.param, a.byte.add b⟩)
  | .token tok, .number b | .number b, .token tok =>
    some (.token {tok with offset := tok.offset.add b})
  | .address a, .token tok | .token tok, .address a =>
    if a.param == 1 then some (.maskPointer {tok with offset := tok.offset.add a.byte}) else none
  | _, _ => none

private def bit32 (a : Int) : UInt32 :=
  UInt32.ofNat (a % 4294967296).toNat

private def bitwise (d : Domain) (regs : Regs) (op a b : String) : Option Reg := do
  let b ← constant regs b
  let mask := bit32 b
  let low : Int := 2 ^ (Nat.log2 mask.toNat + 1)
  if let some (.token token) := get regs a then
    let (lo, hi) := token.interval d
    let off := token.offset
    guard (op == "or.b32" && 0 ≤ lo && hi < 2147483648 && 0 ≤ b && b < 2147483648 &&
      token.stride % low == 0 && off.block % low == 0 && off.columnBase % low == 0)
    return .token {token with offset := {off with fixed := (bit32 off.fixed ||| mask).toNat}}
  let a ← number regs a
  if op == "or.b64" then
    guard (unsigned32 d a && 0 ≤ b && b < 4294967296)
  if op == "shr.u16" || op == "shr.u32" || op == "shr.u64" then
    guard (unsigned32 d a && 0 ≤ b && b < 32)
    if op == "shr.u16" then
      guard ((d.interval a).2 < 65536)
    let scale : Int := 2 ^ b.toNat
    guard (a.block % scale == 0 && a.columnBase % scale == 0)
    return .number ⟨a.fixed / scale, a.block / scale, a.columnBase / scale⟩
  else if let some x := a.constant then
    let word := bit32 x
    let result ← match op with
      | "and.b32" => some (word &&& mask)
      | "and.b16" => some ((word &&& mask) &&& 0xffff)
      | "or.b32" | "or.b64" => some (word ||| mask)
      | "xor.b32" => some (word ^^^ mask)
      | _ => none
    return .number ⟨result.toNat, 0, 0⟩
  else
    guard (unsigned32 d a)
    guard (a.block % low == 0 && a.columnBase % low == 0)
    let fixed := bit32 a.fixed
    match op with
    | "and.b32" => return .number ⟨(fixed &&& mask).toNat, 0, 0⟩
    | "or.b32" | "or.b64" => return .number {a with fixed := (fixed ||| mask).toNat}
    | _ => none

private def addressOp (d : Domain) (regs : Regs) : List String → Option Reg
  | ["ld.param.b64", _, param] => do
    let [_, idx] := param.splitOn "_param_" | none
    guard (idx.endsWith "]")
    return .address ⟨← (idx.dropEnd 1).toString.toNat?, ⟨0, 0, 0⟩⟩
  | ["mov.pred", _, "-1"] => some (.pred .always)
  | ["not.pred", _, src] => do
    let .truth pred ← get regs src | none
    return .truth (!pred)
  | ["selp.b32", _, yes, no, cond] => do
    let .truth pred ← get regs cond | none
    get regs (if pred then yes else no)
  | ["neg.s32", _, src] => do
    return .number ((← number regs src).scale (-1))
  | [op, _, src] => do
    if ["mov.b32", "mov.u32", "mov.b64", "mov.u64", "mov.u16", "mov.b16"].contains op then
      get regs src
    else
      match ← get regs src with
      | word@(.token token) =>
        guard (op == "cvt.s64.s32" && token.signed32 d)
        return word
      | word@(.number a) =>
        let valid : Bool := match op with
          | "cvt.s64.s32" => signed32 d a
          | "cvt.u64.u32" => unsigned32 d a
          | "cvt.u32.u64" => signed32 d a || unsigned32 d a
          | "cvt.u16.u32" | "cvt.u32.u16" =>
            let (lo, hi) := d.interval a
            decide (0 ≤ lo ∧ hi < 65536)
          | _ => false
        guard valid
        return word
      | _ => none
  | ["bfe.u32", _, source, start, width] => do
    let word := bit32 (← constant regs source)
    let start ← start.toNat?
    let width ← width.toNat?
    guard (start + width ≤ 32)
    let part := (word >>> UInt32.ofNat start) &&& UInt32.ofNat (2 ^ width - 1)
    return .number ⟨part.toNat, 0, 0⟩
  | ["mad.wide.s32", _, x, size, y] | ["mad.wide.u32", _, x, size, y] => do
    let a ← number regs x
    guard (signed32 d a && unsigned32 d a)
    addReg (.number (a.scale (← constant regs size))) (← get regs y)
  | ["mad.lo.s32", _, x, size, y] => do
    addReg (.number ((← number regs x).scale (← constant regs size))) (← get regs y)
  | [op, _, x, y] => do
    match op with
    | "add.s32" | "add.s64" =>
      let word ← addReg (← get regs x) (← get regs y)
      if let .token tok := word then
        guard (op == "add.s32" && tok.signed32 d)
      if let .maskPointer _ := word then
        guard (op == "add.s64")
      return word
    | "mul.lo.s32" =>
      let factor ← constant regs y
      match ← get regs x with
      | .number a => return .number (a.scale factor)
      | .token tok =>
        let result := {tok with stride := tok.stride * factor, offset := tok.offset.scale factor}
        guard (result.signed32 d)
        return .token result
      | _ => none
    | "mul.wide.s32" =>
      let a ← number regs x
      guard (signed32 d a)
      return .number (a.scale (← constant regs y))
    | "shl.b32" | "shl.b64" =>
      let exp ← y.toNat?
      guard (exp < if op == "shl.b32" then 32 else 64)
      return .number ((← number regs x).scale (2 ^ exp))
    | "setp.lt.u32" | "setp.lt.u64" =>
      let index ← number regs x
      guard (unsigned32 d index)
      return .pred (.below index (← y.toNat?))
    | "setp.lt.s32" | "setp.gt.u32" =>
      let .token token ← get regs x | none
      let limit ← constant regs y
      let (lo, hi) := token.interval d
      guard (0 ≤ lo && hi < 2147483648)
      if op == "setp.lt.s32" && limit ≤ lo then some (.truth false)
      else if op == "setp.gt.u32" && hi ≤ limit then some (.truth false)
      else none
    | _ => bitwise d regs op x y
  | _ => none

private def half : Option Reg → Option Address
  | some (.half a) | some (.pair (some a) _) => some a
  | _ => none

private def highHalf : Option Reg → Option Address
  | some (.pair _ high) => high
  | _ => none

private def floatOperand (regs : Regs) (name : String) (bits : Nat) : Bool :=
  match get regs name with
  | some (.literal word) => word == bits
  | _ => false

private def changeStage (regs : Regs) (name : String) (before after : Stage) : Option Reg := do
  let .value stage addr ← get regs name | none
  guard (stage = before)
  return .value after addr

private def maskWord (regs : Regs) (arg : String) : Option MaskWord := do
  match ← get regs arg with
  | .maskWord word => some word
  | .number a =>
    let word ← a.constant
    guard (0 ≤ word && word < 65536)
    return .literal word.toNat
  | _ => none

private def MaskWord.and : MaskWord → MaskWord → Option MaskWord
  | .byte address, .literal word => do
    let index := Nat.log2 word
    guard (index < 8 && word == 2 ^ index)
    return .bit ⟨address, index⟩
  | _, _ => none

private def maskOp (regs : Regs) : List String → Option Reg
  | ["shl.b16", _, one, count] => do
    guard ((← constant regs one) == 1)
    let shift ← constant regs count
    guard (0 ≤ shift && shift < 8)
    return .number ⟨2 ^ shift.toNat, 0, 0⟩
  | ["and.b16", _, src, bit] => do
    return .maskWord (← (← maskWord regs src).and (← maskWord regs bit))
  | ["and.b32", _, src, bit] => do
    let .maskPair a b ← get regs src | none
    let .maskPair x y ← get regs bit | none
    return .maskPair (← a.and x) (← b.and y)
  | [op, _, src, "0"] => do
    guard (op == "setp.eq.b16" || op == "setp.ne.b16")
    let .bit bit ← maskWord regs src | none
    return .maskPredicate bit (op == "setp.ne.b16")
  | _ => none

private def projectionOp (d : Domain) (regs : Regs) : List String → Option Reg
  | ["cvt.f32.bf16", _, src] => return .value .widened (← half (get regs src))
  | ["add.f32", _, src, value] =>
    if floatOperand regs value 0x40a00000 then changeStage regs src .widened .shifted
    else if floatOperand regs value 0x3f800000 then changeStage regs src .exp .denominator
    else none
  | ["fma.rn.f32", _, src, scale, zero] => do
    guard (floatOperand regs scale 0xbe088889 && floatOperand regs zero 0)
    changeStage regs src .shifted .affine
  | ["mul.f32", _, src, value] =>
    if floatOperand regs value 0x3fb8aa3b then changeStage regs src .affine .log2
    else if floatOperand regs value 0x41b80000 then changeStage regs src .inverse .scaled
    else none
  | ["ex2.approx.f32", _, src] => changeStage regs src .log2 .exp
  | ["div.full.f32", _, one, src] => do
    guard (floatOperand regs one 0x3f800000)
    changeStage regs src .denominator .inverse
  | ["selp.f32", _, yes, no, tested] => do
    let .maskPredicate bit dropWhenTrue ← get regs tested | none
    let (drop, keep) := if dropWhenTrue then (yes, no) else (no, yes)
    guard (floatOperand regs drop 0xc2700000)
    let .value .scaled proj ← get regs keep | none
    return .score ⟨proj, bit⟩
  | ["shr.u32", _, src, "16"] =>
    ((highHalf (get regs src)).map .half).orElse (fun _ => addressOp d regs ["shr.u32", "", src, "16"])
  | ["prmt.b32", _, a, b, mode] => do
    let choose ← if mode == "0x5410U" then some half else
      if mode == "0x7632U" then some highHalf else none
    return .pair (choose (get regs a)) (choose (get regs b))
  | insn => (maskOp regs insn).orElse fun _ => addressOp d regs insn

private def assign (regs : Regs) (dst : String) (word : Option Reg) : Regs :=
  if let some word := word then regs.insert dst word else regs.erase dst

private def memoryWord (regs : Regs) (ptr : String) : Option Nat := do
  let (.number offset, byte) ← bytes regs ptr | none
  let fixed ← offset.constant
  guard (0 ≤ fixed)
  return fixed.toNat + byte

private def normalizeBracket : List String → List String
  | "[" :: reg :: "+" :: off :: "]" :: rest => ("[" ++ reg ++ "+" ++ off ++ "]") :: rest
  | a :: rest => a :: normalizeBracket rest
  | [] => []

private def loadBound (regs : Regs) : Option String → Option Bound
  | none => some .always
  | some name => do
    guard (name.startsWith "@%")
    let .pred test ← get regs (name.drop 1).toString | none
    return test

private def globalLoad (d : Domain) (regs : Regs) (pred : Option String)
    (dst : List String) (ptr : String) (packed : Bool) : Option Regs := do
  let (.address address, byte) ← bytes regs ptr | none
  let offset := address.byte.add ⟨byte, 0, 0⟩
  guard (2 ≤ address.param && address.param < 10 && offset.fixed ≥ 0 &&
    offset.fixed % 2 == 0 && offset.block == 2 * (rowCount d.tile * 50304) && offset.columnBase == 2 * d.step)
  let start := offset.fixed.toNat / 2
  let col := start % 50304
  let count := if packed then 2 * dst.length else 1
  guard (start / 50304 < rowCount d.tile && d.first ≤ col && col + count ≤ d.first + d.tile.extent)
  let bound ← loadBound regs pred
  guard ((List.range count).all fun i => verifyBound d (col + i) bound)
  let origin := fun displacement => { address with byte := offset.add ⟨displacement, 0, 0⟩ }
  return (dst.zipIdx).foldl (fun regs (dest, i) => regs.insert dest
    (if packed then .pair (some (origin (4 * i))) (some (origin (4 * i + 2)))
     else .half (origin 0))) regs

private def inputLoad (d : Domain) (regs : Regs) (ptr : String) : Option Reg := do
  let (.address address, delta) ← bytes regs ptr | none
  let a := address.byte.add ⟨delta, 0, 0⟩
  guard (address.param == 0 && a.fixed ≥ 0 && a.fixed % 4 == 0 &&
    a.fixed.toNat / 4 < ValidationCE.slabCount * ValidationCE.slabRows &&
    a.block == 4 * rowCount d.tile && a.columnBase == 0)
  return .token ⟨a, 1, ⟨0, 0, 0⟩⟩

private def maskLoad (d : Domain) (regs : Regs) (pred : Option String) (ptr : String) : Option Reg := do
  let (.maskPointer token, delta) ← bytes regs ptr | none
  let byte := token.offset.add ⟨delta, 0, 0⟩
  guard (token.stride == 6288 && byte.fixed ≥ 0 && byte.block == 0 && byte.columnBase * 8 == d.step)
  let bound ← loadBound regs pred
  guard (verifyBound d (byte.fixed.toNat * 8) bound)
  return .maskWord (.byte ⟨token.inputByte, byte⟩)

private structure State where
  lanes : Array Regs
  published : Std.HashMap Nat Address := {}
  pending : Std.HashMap Nat Address := {}
  readers : Std.HashMap Nat (Option Nat) := {}

private def writeHalf (state : State) (tid addr : Nat) (origin : Address) : Option State := do
  let prior := state.readers.get? addr
  guard (addr % 2 == 0 && !state.pending.contains addr && (prior.isNone || prior == some (some tid)))
  return {state with pending := state.pending.insert addr origin}

private def readHalf (state : State) (tid addr : Nat) : Option (State × Address) := do
  guard (addr % 2 == 0 && !state.pending.contains addr)
  let atom ← state.published.get? addr
  let reader := match state.readers.get? addr with
    | none => some tid
    | some old => old.filter (· == tid)
  return ({state with readers := state.readers.insert addr reader}, atom)

private def ordinaryStore (state : State) (ptr arg : String) (packed : Bool) : Option State := do
  let mut result := state
  for (regs, tid) in state.lanes.toList.zipIdx do
    let addr ← memoryWord regs ptr
    result ← writeHalf result tid addr (← half (get regs arg))
    if packed then
      result ← writeHalf result tid (addr + 2) (← highHalf (get regs arg))
  return result

private def ordinaryLoad (state : State) (ptr : String) (destinations : List String) (packed : Bool) :
    Option State := do
  let mut result := state
  let mut lanes := #[]
  for (regs, tid) in state.lanes.toList.zipIdx do
    let start ← memoryWord regs ptr
    let mut regs := regs
    for (dst, i) in destinations.zipIdx do
      let addr := start + 2 * i
      let (afterLo, lo) ← readHalf result tid addr
      result := afterLo
      if packed then
        let (afterHi, hi) ← readHalf result tid (addr + 2)
        result := afterHi
        regs := regs.insert dst (.pair (some lo) (some hi))
      else regs := regs.insert dst (.half lo)
    lanes := lanes.push regs
  return {result with lanes}

private def matrixPair (lanes : Array Regs) (tid matrix : Nat) (ptr : String) (transposed : Bool) :
    Option (Nat × Nat) := do
  let lane := tid % 32
  let first := tid - lane + 8 * matrix
  if transposed then
    let x ← memoryWord (← lanes[first + 2 * (lane % 4)]?) ptr
    let y ← memoryWord (← lanes[first + 2 * (lane % 4) + 1]?) ptr
    return (x + 2 * (lane / 4), y + 2 * (lane / 4))
  else
    let addr ← memoryWord (← lanes[first + lane / 4]?) ptr
    return (addr + 4 * (lane % 4), addr + 4 * (lane % 4) + 2)

private def matrixStore (state : State) (ptr : String) (input : List String) : Option State := do
  let mut result := state
  for (regs, tid) in state.lanes.toList.zipIdx do
    for (name, matrix) in input.zipIdx do
      let (a, b) ← matrixPair state.lanes tid matrix ptr true
      result ← writeHalf result tid a (← half (get regs name))
      result ← writeHalf result tid b (← highHalf (get regs name))
  return result

private def matrixLoad (state : State) (ptr : String) (output : List String) : Option State := do
  let mut result := state
  let mut lanes := #[]
  for (regs, tid) in state.lanes.toList.zipIdx do
    let mut regs := regs
    for (dst, matrix) in output.zipIdx do
      let (a, b) ← matrixPair state.lanes tid matrix ptr false
      let (afterA, x) ← readHalf result tid a
      let (afterB, y) ← readHalf afterA tid b
      result := afterB
      regs := regs.insert dst (.pair (some x) (some y))
    lanes := lanes.push regs
  return {result with lanes}

private def splitPair (regs : Regs) (low high word : String) : Regs :=
  match get regs word with
  | some (.maskPair lo hi) => (regs.insert low (.maskWord lo)).insert high (.maskWord hi)
  | raw => assign (assign regs low ((half raw).map .half)) high ((highHalf raw).map .half)

private def packPair (regs : Regs) (dst lo hi : String) : Regs :=
  match maskWord regs lo, maskWord regs hi with
  | some l, some h => regs.insert dst (.maskPair l h)
  | _, _ => regs.insert dst (.pair (half (get regs lo)) (half (get regs hi)))

private def step (d : Domain) (state : State) (instruction : Instruction) : Option State := do
  let insn := normalizeBracket instruction.words
  match insn with
  | ["bar.sync", "0"] =>
    guard (instruction.pred.isNone)
    return {state with
      published := state.pending.fold (fun mem key value => mem.insert key value) state.published
      pending := {}
      readers := {}}
  | ["st.shared.b16", ptr, src] | ["st.shared.b32", ptr, src] =>
    guard (instruction.pred.isNone)
    ordinaryStore state ptr src (insn.headD "" == "st.shared.b32")
  | ["ld.shared.b16", dest, ptr] | ["ld.shared.b32", dest, ptr] =>
    guard (instruction.pred.isNone)
    ordinaryLoad state ptr [dest] (insn.headD "" == "ld.shared.b32")
  | ["ld.shared.v2.b16", lo, hi, ptr] =>
    guard (instruction.pred.isNone)
    ordinaryLoad state ptr [lo, hi] false
  | ["stmatrix.sync.aligned.m8n8.x2.trans.shared.b16", ptr, _, _]
  | ["stmatrix.sync.aligned.m8n8.x4.trans.shared.b16", ptr, _, _, _, _] =>
    guard (instruction.pred.isNone)
    matrixStore state ptr (insn.drop 2)
  | ["ldmatrix.sync.aligned.m8n8.x2.shared.b16", _, _, ptr]
  | ["ldmatrix.sync.aligned.m8n8.x4.shared.b16", _, _, _, _, ptr] =>
    guard (instruction.pred.isNone)
    matrixLoad state ptr ((insn.drop 1).take (insn.length - 2))
  | ["ld.global.L1::evict_last.L2::cache_hint.b8", dst, ptr, _] =>
    let lanes ← state.lanes.mapM fun regs => do
      return regs.insert dst (← maskLoad d regs instruction.pred ptr)
    return {state with lanes}
  | "ld.global.L1::evict_last.L2::cache_hint.b16" :: rest
  | "ld.global.L1::evict_last.L2::cache_hint.b32" :: rest
  | "ld.global.L1::evict_last.L2::cache_hint.v2.b32" :: rest
  | "ld.global.L1::evict_last.L2::cache_hint.v4.b32" :: rest =>
    let packed := !(insn.headD "").endsWith ".b16"
    let destinations := rest.take (rest.length - 2)
    let ptr ← rest[rest.length - 2]?
    let lanes ← state.lanes.mapM fun regs => globalLoad d regs instruction.pred destinations ptr packed
    return {state with lanes}
  | ["mov.b32", first, second, third] =>
    guard (instruction.pred.isNone)
    match instruction.output with
    | [_, _] => return {state with lanes := state.lanes.map fun regs => splitPair regs first second third}
    | [dst] => return {state with lanes := state.lanes.map fun regs => packPair regs dst second third}
    | _ => none
  | name :: _ =>
    guard (!(name.startsWith "st." || name.startsWith "ld.global" || name.startsWith "ld.shared" ||
      name.startsWith "stmatrix." || name.startsWith "ldmatrix." || name.startsWith "bar." ||
      name.startsWith "atom." || name.startsWith "red."))
    let lanes := state.lanes.map fun regs =>
      match instruction.output with
      | [dst] => assign regs dst
          (if instruction.pred.isNone then projectionOp d regs insn else none)
      | _ => instruction.output.foldl (·.erase ·) regs
    return {state with lanes}
  | [] => return state

private abbrev Definitions := Std.HashMap String (List Instruction)

private def definitions (insns : Array Instruction) : Definitions := Id.run do
  let mut result : Definitions := {}
  for insn in insns do
    for dst in insn.output do
      result := result.insert dst (insn :: result.getD dst [])
  return result

private def prologue (insns : Array Instruction) (defs : Definitions)
    (edges : Array (List Nat)) (header : Nat) : Array Instruction :=
  let unique := fun (i : Instruction) => i.output.any fun dst =>
    match defs.getD dst [] with
    | [_] => true
    | [load, init] =>
      i.pc == load.pc && load.words.head? == some "ld.global.L1::evict_last.L2::cache_hint.b32" &&
        init.words == ["mov.u32", dst, "0x0"]
    | _ => false
  (insns.extract 0 header).filter fun i =>
    i.pred.isNone && unique i && dominates edges i.pc header

private def byteCounters (tile : Tile) (edges : Array (List Nat)) (header latch : Nat)
    (defs : Definitions) : List (String × String × Nat) :=
  defs.toList.filterMap fun (reg, writes) => do
    let [advance, init] := writes | none
    let ["add.s32", _, previous, amount] := advance.words | none
    let ["mov.b32", _, initial] := init.words | none
    let amount ← amount.toNat?
    guard (previous == reg && amount > 0 && amount * 8 == tile.extent &&
      init.pred.isNone && advance.pred.isNone && init.pc < header && header < advance.pc &&
      advance.pc < latch && dominates edges init.pc header && dominates edges advance.pc latch)
    return (reg, initial, amount)

private def seed (d : Domain) (insns : Array Instruction) (counter : ValidationExpBinding.LoopCounter)
    (byteCounters : List (String × String × Nat)) (tid : Nat) : Regs := Id.run do
  let mut regs : Regs := {}
  regs := (regs.insert "%tid.x" (.number ⟨tid, 0, 0⟩)).insert "%ctaid.x" (.number ⟨0, 1, 0⟩)
  for i in insns do
    if let [dst] := i.output then
      regs := match normalizeBracket i.words with
        | ["ld.global.L1::evict_last.L2::cache_hint.b32", _, ptr, _] =>
          assign regs dst (inputLoad d regs ptr)
        | ["mov.b32", _, lo, hi] => packPair regs dst lo hi
        | operands => assign regs dst (projectionOp d regs operands)
  for (reg, initializer, amount) in byteCounters do
    if let some init := constant regs initializer then
      regs := regs.insert reg (.number ⟨init + amount * d.phase, 0, amount * phaseCount d.tile⟩)
  let first : Int := if counter.prebiased then (d.first : Int) - counter.extent else d.first
  return regs.insert counter.register (.number ⟨first, 0, d.step⟩)

private def branchTaken (regs : Regs) (raw : String) : Option Bool := do
  let (negative, reg) := if raw.startsWith "@!" then (true, raw.drop 2) else (false, raw.drop 1)
  let .truth value ← get regs reg.toString | none
  return value != negative

abbrev Layout := List (List (List Update))

private def replayPhase (d : Domain) (insns init : Array Instruction) (edges : Array (List Nat))
    (path : List Nat) (data : ValidationExpBinding.Decoded)
    (byteCounters : List (String × String × Nat)) : Option (List (List Update)) := do
  let threads := rowCount d.tile * d.tile.lanes
  let mut state : State := ⟨(List.range threads).toArray.map (seed d init data.counter byteCounters), {}, {}, {}⟩
  let mut incoming : Std.HashMap Nat String := {}
  let mut tailInputs : Std.HashMap Nat String := {}
  for update in data.updates do
    guard (path.contains update.incoming.inputPC && !(incoming.contains update.incoming.inputPC))
    incoming := incoming.insert update.incoming.inputPC update.incoming.source
    if let (some pc, some col) := (update.activeColumnPC, update.activeColumn) then
      guard (path.contains pc && (tailInputs.get? pc).all (· == col))
      tailInputs := tailInputs.insert pc col
  let mut marks : Std.HashMap Nat (List Score) := {}
  let mut bounds : Std.HashMap Nat (List Affine) := {}
  for (pc, next) in path.zip path.tail do
    if let some name := tailInputs.get? pc then
      let indices ← state.lanes.toList.mapM fun regs => number regs name
      guard (indices.all (unsigned32 d))
      bounds := bounds.insert pc indices
    if let some src := incoming.get? pc then
      let addresses ← state.lanes.toList.mapM fun regs => do
        let .score origin ← get regs src | none
        return origin
      marks := marks.insert pc addresses
    let instruction ← insns[pc]?
    if let [target, fallthrough] := edges[pc]! then
      let some pred := instruction.pred | none
      guard (state.lanes.all fun regs =>
        match branchTaken regs pred with
        | some taken => (if taken then target else fallthrough) == next
        | none => false)
    state ← step d state instruction
  data.updates.mapM fun update => do
    let addresses ← marks.get? update.incoming.inputPC
    let cols := addresses.map fun origin => origin.projection.byte.fixed.toNat / 2 % 50304
    match update.activeColumnPC with
    | none =>
      guard (cols.all fun col => verifyBound d col .always)
      return addresses.map fun score => ⟨score, d.steps⟩
    | some pc =>
      let indices ← bounds.get? pc
      guard (indices.length == cols.length &&
        (indices.zip cols).all (fun (index, col) => verifyBound d col (.below index 50304)))
      return (addresses.zip indices).map fun (score, index) =>
        ⟨score, liveIterations d.steps d.step index.fixed.toNat 50304⟩

def replay (ptx : String) (tile : Tile) (scan : ValidationExpBinding.Certificate ptx tile) : Option Layout := do
  let lines := QKVTTIR.normalizePTX ptx
  let data := scan.reduction
  let edges ← ValidationLogBinding.validationFlow lines
  guard ((List.range data.header).all fun pc => edges[pc]? == some [pc + 1])
  let path ← ValidationReduceBinding.straightPath edges (data.header + 1) data.latch
  let insns := (lines.zipIdx.map fun (line, pc) => Instruction.ofLine pc line).toArray
  let threads := rowCount tile * tile.lanes
  guard (lines.contains s!".reqntid {threads}")
  let defs := definitions insns
  let init := prologue insns defs edges data.header
  let counters := byteCounters tile edges data.header data.latch defs
  (List.range (phaseCount tile)).mapM fun phase =>
    replayPhase ⟨tile, phase⟩ insns init edges path data counters

private def score (d : Domain) (slot : Slot) : Score :=
  let projection := projectionAddress d.tile slot
  let col := slot.column + d.first
  ⟨{projection with byte := d.specialize projection.byte},
   ⟨⟨⟨4 * (slot.slab * ValidationCE.slabRows + slot.row), 4 * rowCount d.tile, 0⟩,
       ⟨(col / 8 : Nat), 0, (d.step / 8 : Nat)⟩⟩, col % 8⟩⟩

private def expectedUpdate (d : Domain) (slot : Slot) : Update :=
  ⟨score d slot, liveIterations d.steps d.step (slot.column + d.first) 50304⟩

open VeriTile.Triton FP

/-- Read the score through the launch's projection, input token and mask
buffers. The input token is shared with the target gather's mask lookup. -/
def Score.eval (s : Score) (native : ValidationCE.Native) (block step : Nat)
    (tokens : Nat → Fin 50304) (mask : Nat → UInt8)
    (bf16 : Nat → Int → TailNorm.BF16) : TailNorm.F32 :=
  let row := (s.mask.address.tokenByte.eval block step).toNat / 4
  let word := mask ((tokens row).val * 6288 + (s.mask.address.columnByte.eval block step).toNat)
  ValidationTargetBinding.payload native (word.toNat.testBit s.mask.index)
    (bf16 s.projection.param (s.projection.byte.eval block step))

private theorem phased_column (tile : Tile) (col iteration : Nat) :
    let d : Domain := ⟨tile, iteration % phaseCount tile⟩
    ValidationCE.scanColumn tile col iteration = col + d.first + iteration / phaseCount tile * d.step := by
  cases tile <;> simp [ValidationCE.scanColumn, Domain.first, Domain.step, Tile.extent, phaseCount] <;> omega

private theorem mask_indices (tile : Tile) (col iteration : Nat) :
    let d : Domain := ⟨tile, iteration % phaseCount tile⟩
    let column := ValidationCE.scanColumn tile col iteration
    (col + d.first) / 8 + iteration / phaseCount tile * (d.step / 8) = column / 8 ∧
    (col + d.first) % 8 = column % 8 := by
  have column := phased_column tile col iteration
  cases tile <;> simp [Domain.first, Domain.step, Tile.extent, phaseCount,
    ValidationCE.scanColumn] at * <;> omega

private theorem phase_steps (tile : Tile) (iteration : Nat)
    (valid : iteration < TailNorm.ceilDiv 50304 tile.extent) :
    iteration / phaseCount tile < (Domain.mk tile (iteration % phaseCount tile)).steps := by
  cases tile <;> simp [Domain.steps, Tile.extent, phaseCount, TailNorm.ceilDiv] at * <;> omega

private theorem update_guard_exact (tile : Tile) (slot : Slot) (iteration : Nat)
    (valid : iteration < TailNorm.ceilDiv 50304 tile.extent) :
    let d : Domain := ⟨tile, iteration % phaseCount tile⟩
    iteration / phaseCount tile < (expectedUpdate d slot).liveSteps ↔
      ValidationCE.scanColumn tile slot.column iteration < 50304 := by
  let d : Domain := ⟨tile, iteration % phaseCount tile⟩
  change iteration / phaseCount tile < liveIterations d.steps d.step (slot.column + d.first) 50304 ↔ _
  have step : 0 < d.step := by cases tile <;> simp [d, Domain.step, Tile.extent, phaseCount]
  rw [liveIterations_exact _ _ _ (phase_steps tile iteration valid) step, phased_column tile slot.column]


private theorem reference_score_exact (tile : Tile) (slot : Slot) (native : ValidationCE.Native)
    (block iteration : Nat) (tokens : Nat → Fin 50304) (mask : Nat → UInt8)
    (bf16 : Nat → Int → TailNorm.BF16) :
    let d : Domain := ⟨tile, iteration % phaseCount tile⟩
    (score d slot).eval native block (iteration / phaseCount tile) tokens mask bf16 =
      ValidationCE.logit native
        (fun byte => mask ((tokens (slot.slab * ValidationCE.slabRows + block * rowCount tile + slot.row)).val *
          6288 + byte))
        (ValidationCE.scanColumn tile slot.column iteration)
        (bf16 (2 + slot.slab) ((projectionAddress tile slot).byte.eval block (tile.extent * iteration))) := by
  obtain ⟨maskByte, maskBit⟩ := mask_indices tile slot.column iteration
  let d : Domain := ⟨tile, iteration % phaseCount tile⟩
  let step := iteration / phaseCount tile
  have base : d.first + step * d.step = tile.extent * iteration := by
    simpa [d, step, ValidationCE.scanColumn, Nat.mul_comm] using (phased_column tile 0 iteration).symm
  have address : (d.specialize (projectionAddress tile slot).byte).eval block step =
      (projectionAddress tile slot).byte.eval block (tile.extent * iteration) := by
    rw [← base]
    simp only [Domain.specialize, Affine.eval, Int.natCast_add, Int.natCast_mul, Int.mul_add]
    ac_rfl
  have row : ((score d slot).mask.address.tokenByte.eval block step).toNat / 4 =
      slot.slab * ValidationCE.slabRows + block * rowCount tile + slot.row := by
    have raw : (score d slot).mask.address.tokenByte.eval block step =
        (4 * (slot.slab * ValidationCE.slabRows + block * rowCount tile + slot.row) : Nat) := by
      simp only [score, Affine.eval, Int.natCast_mul, Int.natCast_add, Int.zero_mul, Int.add_zero, Int.mul_add]
      ac_rfl
    rw [raw, Int.toNat_natCast, Nat.mul_div_cancel_left _ (by decide)]
  have byte : ((score d slot).mask.address.columnByte.eval block step).toNat =
      (slot.column + d.first) / 8 + step * (d.step / 8) := by
    simp only [score, Affine.eval, Int.zero_mul, Int.add_zero]
    rw [Int.mul_comm, ← Int.natCast_mul, ← Int.natCast_add, Int.toNat_natCast]
  have bit : (score d slot).mask.index = (slot.column + d.first) % 8 := rfl
  change (score d slot).eval native block step tokens mask bf16 = _
  unfold Score.eval
  rw [row, byte, bit, maskByte, maskBit]
  change ValidationTargetBinding.payload native _ (bf16 (2 + slot.slab) ((d.specialize
    (projectionAddress tile slot).byte).eval block step)) = _
  rw [address]
  rfl

def reference (tile : Tile) : Option Layout := do
  (List.range (phaseCount tile)).mapM fun phase =>
    (List.range (ValidationCE.slabCount * tile.groups)).mapM fun update =>
      (List.range (rowCount tile * tile.lanes)).mapM fun tid =>
        (inputSlot tile update tid).map (expectedUpdate ⟨tile, phase⟩)

structure Certificate (ptx : String) (tile : Tile) where
  scan : ValidationExpBinding.Certificate ptx tile
  words : Layout
  decoded : replay ptx tile scan = some words
  exact : reference tile = some words

def certifyFrom {ptx : String} {tile : Tile} (scan : ValidationExpBinding.Certificate ptx tile) :
    Option (Certificate ptx tile) :=
  match expected : reference tile with
  | none => none
  | some words =>
    if exact : replay ptx tile scan = some words then
      some ⟨scan, words, exact, expected⟩
    else none

def certify (ptx : String) (tile : Tile) : Option (Certificate ptx tile) := do
  let scan ← ValidationExpBinding.certify ptx tile
  certifyFrom scan

/-- Each compiled softmax input reads the expected projection word and
the canonical mask byte and bit for the same vocabulary slot. -/
theorem decoded_score_layout_exact {ptx : String} {tile : Tile} (proof : Certificate ptx tile) :
    replay ptx tile proof.scan = reference tile :=
  proof.decoded.trans proof.exact.symm

private theorem getElem_mapM_some {f : α → Option β} {xs : List α} {ys : List β}
    (run : xs.mapM f = some ys) {i : Nat} (valid : i < xs.length) :
    ∃ y, ys[i]? = some y ∧ f xs[i] = some y := by
  induction xs generalizing ys i with
  | nil => simp at valid
  | cons x xs ih =>
    cases head : f x with
    | none => simp [List.mapM_cons, head] at run
    | some y =>
      cases tail : xs.mapM f with
      | none => simp [List.mapM_cons, head, tail] at run
      | some rst =>
        have eq : y :: rst = ys := by simpa [List.mapM_cons, head, tail] using run
        subst ys
        cases i with
        | zero => exact ⟨y, rfl, head⟩
        | succ i =>
          have rest : i < xs.length := by simpa using valid
          obtain ⟨elem, found, evaluated⟩ := ih tail rest
          exact ⟨elem, found, evaluated⟩

def Layout.at (words : Layout) (tile : Tile) (iteration position tid : Nat) : Option Update := do
  let updates ← words[iteration % phaseCount tile]?
  let lanes ← updates[position]?
  lanes[tid]?

private theorem decoded_word_exact {ptx : String} {tile : Tile} (proof : Certificate ptx tile)
    (iteration position tid : Nat)
    (activeUpdate : position < ValidationCE.slabCount * tile.groups)
    (activeLane : tid < rowCount tile * tile.lanes)
    {slot : Slot} (route : inputSlot tile position tid = some slot) :
    ((replay ptx tile proof.scan).bind fun words => words.at tile iteration position tid) =
      some (expectedUpdate ⟨tile, iteration % phaseCount tile⟩ slot) := by
  have phase : iteration % phaseCount tile < (List.range (phaseCount tile)).length := by
    rw [List.length_range]
    exact Nat.mod_lt _ (by cases tile <;> decide)
  have expected := proof.exact
  unfold reference at expected
  obtain ⟨updates, found, updateRows⟩ := getElem_mapM_some expected phase
  simp only [List.getElem_range] at updateRows
  obtain ⟨lanes, rowAt, laneWords⟩ := getElem_mapM_some updateRows (by simpa using activeUpdate)
  simp only [List.getElem_range] at laneWords
  obtain ⟨word, laneAt, result⟩ := getElem_mapM_some laneWords (by simpa using activeLane)
  simp only [List.getElem_range, route, Option.map_some, Option.some.injEq] at result
  subst word
  rw [proof.decoded]
  change proof.words.at tile iteration position tid = _
  unfold Layout.at
  rw [found]
  change (updates[position]?.bind fun lanes => lanes[tid]?) = _
  rw [rowAt]
  exact laneAt

def rowMask (tile : Tile) (slot : Slot) (block : Nat)
    (tokens : Nat → Fin 50304) (mask : Nat → UInt8) (byte : Nat) : UInt8 :=
  let row := slot.slab * ValidationCE.slabRows + block * rowCount tile + slot.row
  mask ((tokens row).val * 6288 + byte)

def rowProjection (tile : Tile) (slot : Slot) (block : Nat)
    (bf16 : Nat → Int → TailNorm.BF16) (column : Nat) : TailNorm.BF16 :=
  bf16 (2 + slot.slab) (2 * ((block * rowCount tile + slot.row) * 50304 + column))

def Update.commit (update : Update) (native : ValidationCE.Native) (block step : Nat)
    (tokens : Nat → Fin 50304) (mask : Nat → UInt8) (bf16 : Nat → Int → TailNorm.BF16)
    (state : ValidationCE.Partial) : ValidationCE.Partial :=
  if step < update.liveSteps then
    ValidationExpBinding.onlinePayload native state (update.score.eval native block step tokens mask bf16)
  else state

/-- The decoded score expression consumed by the online exponential
uses the same FP32 payload for any legal token, mask, and BF16 buffers. -/
theorem decoded_score_exact {ptx : String} {tile : Tile} (proof : Certificate ptx tile)
    (native : ValidationCE.Native) (block iteration position tid : Nat)
    (tokens : Nat → Fin 50304) (mask : Nat → UInt8) (bf16 : Nat → Int → TailNorm.BF16)
    (activeUpdate : position < ValidationCE.slabCount * tile.groups)
    (activeLane : tid < rowCount tile * tile.lanes)
    {slot : Slot} (route : inputSlot tile position tid = some slot) :
    (((replay ptx tile proof.scan).bind fun words => words.at tile iteration position tid).map
      fun word => word.score.eval native block (iteration / phaseCount tile) tokens mask bf16) =
      some (ValidationCE.logit native (rowMask tile slot block tokens mask)
        (ValidationCE.scanColumn tile slot.column iteration)
        (rowProjection tile slot block bf16 (ValidationCE.scanColumn tile slot.column iteration))) := by
  rw [decoded_word_exact proof iteration position tid activeUpdate activeLane route, Option.map_some]
  change some ((score _ slot).eval _ _ _ _ _ _) = _
  rw [reference_score_exact, projection_byte_exact]
  rfl

private theorem expected_commit_exact (tile : Tile) (slot : Slot) (native : ValidationCE.Native)
    (block iteration : Nat) (tokens : Nat → Fin 50304) (mask : Nat → UInt8)
    (bf16 : Nat → Int → TailNorm.BF16) (state : ValidationCE.Partial)
    (valid : iteration < TailNorm.ceilDiv 50304 tile.extent) :
    let column := ValidationCE.scanColumn tile slot.column iteration
    let word := ValidationCE.logit native (rowMask tile slot block tokens mask) column
      (rowProjection tile slot block bf16 column)
    (expectedUpdate ⟨tile, iteration % phaseCount tile⟩ slot).commit native block
      (iteration / phaseCount tile) tokens mask bf16 state =
      if column < 50304 then ValidationCE.accumulate native state word else state := by
  dsimp only [Update.commit]
  rw [if_congr (update_guard_exact tile slot iteration valid) rfl rfl]
  have value := reference_score_exact tile slot native block iteration tokens mask bf16
  dsimp only at value
  rw [projection_byte_exact] at value
  split
  · change ValidationExpBinding.onlinePayload native state ((score _ slot).eval _ _ _ _ _ _) = _
    rw [value, ValidationExpBinding.online_payload_exact]
    rfl
  · rfl

/-- Replay one lane's checked vocabulary loop. The launch counter
determines the trip count and the decoded selects determine each update. -/
def runSlot {ptx : String} {tile : Tile} (proof : Certificate ptx tile)
    (native : ValidationCE.Native) (block position tid : Nat)
    (tokens : Nat → Fin 50304) (mask : Nat → UInt8) (bf16 : Nat → Int → TailNorm.BF16) :
    Option ValidationCE.Partial :=
  let counter := proof.scan.reduction.counter
  let n := TailNorm.ceilDiv (counter.limit + counter.extent) counter.extent
  (List.range n).foldlM (fun state iteration => do
    let word ← proof.words.at tile iteration position tid
    return word.commit native block (iteration / phaseCount tile) tokens mask bf16 state)
    ValidationCE.empty

private theorem foldlM_some {f : α → β → Option α} {g : α → β → α}
    {xs : List β} (step : ∀ state x, x ∈ xs → f state x = some (g state x)) (init : α) :
    xs.foldlM f init = some (xs.foldl g init) := by
  induction xs generalizing init with
  | nil => rfl
  | cons head tail ih =>
    simp only [List.foldlM_cons, List.foldl_cons, step init head (List.mem_cons_self)]
    exact ih (fun state x member => step state x (List.mem_cons_of_mem _ member)) _

/-- The sum and maximum carried out of the compiled vocabulary loop
match the reference scan for every slot, including a partial last tile. -/
theorem decoded_scan_exact {ptx : String} {tile : Tile} (proof : Certificate ptx tile)
    (native : ValidationCE.Native) (block position tid : Nat)
    (tokens : Nat → Fin 50304) (mask : Nat → UInt8) (bf16 : Nat → Int → TailNorm.BF16)
    (activeUpdate : position < ValidationCE.slabCount * tile.groups)
    (activeLane : tid < rowCount tile * tile.lanes)
    {slot : Slot} (route : inputSlot tile position tid = some slot) :
    runSlot proof native block position tid tokens mask bf16 =
      some (ValidationCE.scan (vocab := 50304) native tile
        (fun column => ValidationCE.logit native (rowMask tile slot block tokens mask) column.val
          (rowProjection tile slot block bf16 column.val)) slot.column) := by
  have size : proof.scan.reduction.counter.limit + proof.scan.reduction.counter.extent = 50304 := by
    rw [proof.scan.limit, proof.scan.extent]
    cases tile <;> decide
  have count :
      TailNorm.ceilDiv (proof.scan.reduction.counter.limit + proof.scan.reduction.counter.extent)
        proof.scan.reduction.counter.extent =
      TailNorm.ceilDiv 50304 tile.extent := by
    rw [size]
    exact congrArg (TailNorm.ceilDiv 50304) proof.scan.extent
  unfold runSlot ValidationCE.scan
  simp only [bind, count]
  apply foldlM_some
  intro state iteration member
  have valid : iteration < TailNorm.ceilDiv 50304 tile.extent := List.mem_range.mp member
  have word := decoded_word_exact proof iteration position tid activeUpdate activeLane route
  rw [proof.decoded, Option.bind_some] at word
  rw [word]
  have commit := expected_commit_exact tile slot native block iteration tokens mask bf16 state valid
  by_cases live : ValidationCE.scanColumn tile slot.column iteration < 50304
  · simpa only [dif_pos live, if_pos live] using congrArg some commit
  · simpa only [dif_neg live, if_neg live] using congrArg some commit

#axiomsClean projection_byte_exact
#axiomsClean load_guard_exact
#axiomsClean decoded_score_exact
#axiomsClean decoded_scan_exact

end VeriTile.Bench.Models.NanoGPT.Track1.ValidationScanBinding
