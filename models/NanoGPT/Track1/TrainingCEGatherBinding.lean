import models.NanoGPT.Track1.QKVTTIRBinding
import models.NanoGPT.Track1.TrainingCE

/-! Decode the packed base-gradient payloads in the fixed-count CE PTX.
Each straight-line fragment starts with a two-word code load and ends with
a four-halfword gradient store. Control flow supplies the two addresses.
The decoder follows the byte selectors, cache addresses and packed store
operands. It gives no meaning to a branch or a sparse correction store. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEGatherBinding
open TrainingCE CudaStores

def operands (line : String) : List String :=
  let text := line.replace "," " " |>.replace "{" " " |>.replace "}" " "
    |>.replace ";" "" |>.replace "\r" ""
  (text.splitOn " ").filter (!·.isEmpty)

private def indirect (s : String) : Option String := do
  guard (s.startsWith "[" && s.endsWith "]")
  let a := (s.drop 1 |>.dropEnd 1).toString
  guard (a.startsWith "%rd")
  return a

private def displacement (input : String) : Option (String × Nat) :=
  match input.splitOn "+" with
  | [reg] => some (reg, 0)
  | [reg, offset] => (offset.toNat?).map fun n => (reg, n)
  | _ => none

private def tableSymbol := "_ZZ17ce_fwd_bwd_kernelE9exp_table"

/-- Register destinations in the dialect emitted by the fixed CE compiler.
Vector destinations start in a brace group. Stores, labels and directives
introduce no register definitions. -/
private def destinations (line : String) : List String :=
  let text := if line.startsWith "@" then
    String.intercalate " " ((line.splitOn " ").drop 1)
    else line
  match text.splitOn " " with
  | op :: _ =>
    if op.startsWith "st." || op.startsWith "." || op.startsWith "{" then []
    else
      let tail := (text.drop (op.length + 1)).toString
      let fields := if tail.startsWith "{" then operands ((tail.splitOn "}").headD "")
        else (((tail.splitOn ",").headD "").splitOn "|").map (·.trimAscii.toString)
      fields.filter (·.startsWith "%")
  | [] => []

def defines (reg : String) (line : String) : Bool :=
  if line.startsWith "{" then
    ((line.drop 1).toString.splitOn ";").any fun insn =>
      (destinations insn.trimAscii.toString).contains reg
  else (destinations line).contains reg

def tableRegister (lines : List String) : Option (String × Nat) := do
  guard (lines.contains s!".shared .align 4 .b8 {tableSymbol}[1024];")
  let [(reg, pc)] := lines.zipIdx.filterMap (fun (line, pc) =>
    match operands line with
    | ["mov.u64", reg, symbol] => if symbol == tableSymbol then some (reg, pc) else none
    | _ => none)
    | none
  guard (reg.startsWith "%rd")
  guard ((lines.filter (defines reg)).length == 1)
  return (reg, pc)

def flow (lines : List String) : Option (Array (List Nat)) := do
  let labels := lines.zipIdx.filterMap fun (line, pc) =>
    if line.endsWith ":" then some ((line.dropEnd 1).toString, pc) else none
  guard ((labels.map Prod.fst).eraseDups.length == labels.length)
  let edges ← lines.zipIdx.mapM fun (line, pc) => do
    let tokens := operands line
    let conditional := line.startsWith "@"
    let actual := if conditional then tokens.drop 1 else tokens
    let fallthrough := if pc + 1 < lines.length then [pc + 1] else []
    match actual with
    | ["ret"] | ["exit"] | ["trap"] => return if conditional then fallthrough else []
    | [op, name] =>
      if op == "bra" || op == "bra.uni" then
        return (← labels.lookup name) :: (if conditional then fallthrough else [])
      else
        guard (!(op.startsWith "brx" || op.startsWith "call"))
        return fallthrough
    | op :: _ =>
      guard (!(op.startsWith "bra" || op.startsWith "brx" || op.startsWith "call"))
      return fallthrough
    | [] => return fallthrough
  return edges.toArray

/-- Find whether the use is reachable while skipping the instruction
that defines its input. Reject graph traversals that exhaust their fuel. -/
private def reachableWithout (edges : Array (List Nat)) (skip use : Nat) :
    Nat → List Nat → Array Bool → Option Bool
  | _, [], _ => some false
  | 0, _ :: _, _ => none
  | fuel + 1, pc :: todo, seen =>
    if pc == skip || seen[pc]! then reachableWithout edges skip use fuel todo seen
    else if pc == use then some true
    else reachableWithout edges skip use fuel (edges[pc]! ++ todo) (seen.set! pc true)

def dominates (edges : Array (List Nat)) (definition use : Nat) : Bool :=
  reachableWithout edges definition use (2 * edges.size + 1) [0] (Array.replicate edges.size false)
    == some false

def uniqueDefinition (lines : List String) (edges : Array (List Nat))
    (reg : String) (use : Nat) : Option (List String × Nat) := do
  let [(line, pc)] := lines.zipIdx.filter (fun (line, _) => defines reg line) | none
  guard (dominates edges pc use && pc != use)
  return (operands line, pc)

/-- Check that each lane writes the byte at exp_table + threadIdx.x
between the sum barrier and the barrier that releases the gather. -/
def cacheBarrier (lines : List String) (edges : Array (List Nat))
    (table : String × Nat) : Option Nat := do
  let [_, sumReady, cacheReady] := lines.zipIdx.filterMap (fun (line, pc) =>
    if line == "bar.sync 0;" then some pc else none) | none
  let [([_, address, _], publish)] := lines.zipIdx.filterMap (fun (line, pc) =>
    let insn := operands line
    if insn.head? == some "st.shared.u8" then some (insn, pc) else none) | none
  guard (sumReady < publish && publish < cacheReady && dominates edges sumReady publish &&
    dominates edges publish cacheReady)
  let dest ← indirect address
  let (["add.s64", out, base, lane64], definition) ← uniqueDefinition lines edges dest publish | none
  guard (out == dest && base == table.1 && dominates edges table.2 definition)
  let (["cvt.u64.u32", _, lane], definition) ← uniqueDefinition lines edges lane64 definition | none
  let (["mov.u32", _, "%tid.x"], _) ← uniqueDefinition lines edges lane definition | none
  return cacheReady

private inductive AddressOperand where
  | offset
  | port (n : Nat)
  deriving DecidableEq

/-- Resolve the live global address bases through the emitted parameter
loads and pointer additions. The scan's self-increment retains its
seed's base. The scan and store decoders separately check its offsets. -/
private def addressOperand (lines : List String) : Nat → String → Option AddressOperand
  | 0, _ => none
  | fuel + 1, reg => do
    if reg.toInt?.isSome then return .offset
    let definitions := (lines.filter (defines reg)).map operands
    let initial := definitions.filter fun insn =>
      match insn with
      | ["add.s64", dst, src, delta] => !(dst == reg && src == reg && delta.toInt?.isSome)
      | _ => true
    let origins ← initial.mapM fun insn =>
      match insn with
      | ["ld.param.u64", _, parameter] => do
        let key := (parameter.drop 1 |>.dropEnd 1).toString
        let stem := "ce_fwd_bwd_kernel_param_"
        guard (parameter.startsWith "[" && parameter.endsWith "]" && key.startsWith stem)
        let port ← (key.drop stem.length).toString.toNat?
        guard (port < 7)
        return .port port
      | ["cvta.to.global.u64", _, src] => addressOperand lines fuel src
      | ["add.s64", _, a, b] => do
        let x ← addressOperand lines fuel a
        let y ← addressOperand lines fuel b
        match x, y with
        | .offset, p | p, .offset => some p
        | _, _ => none
      | ["ld.global.nc.u64", _, _] | ["ld.shared.s32", _, _] => some .offset
      | ["mov.u64", _, value] =>
        if value.toInt?.isSome then some .offset else addressOperand lines fuel value
      | [op, _, src] =>
        if (op == "cvt.s64.s32" || op == "cvt.u64.u32") &&
            (src.drop 2).toString.toNat?.isSome && src.startsWith "%r" then some .offset else none
      | [op, _, src, pitch] =>
        if (op == "mul.wide.u32" || op == "mul.wide.s32") &&
            (src.drop 2).toString.toNat?.isSome && src.startsWith "%r" && pitch.toNat?.isSome
        then some .offset else none
      | _ => none
    let start ← origins.head?
    guard (origins.all (decide <| · = start))
    return start

private def storageAccess (lines : List String) (sumReady : Nat)
    (pc : Nat) (line : String) : Option Bool := do
  let tokens := operands line
  let tokens := if line.startsWith "@" then tokens.drop 1 else tokens
  let op := tokens.headD ""
  if op.startsWith "ld.global." || op.startsWith "st.global." then
    let [operand] := tokens.filter (·.startsWith "[") | none
    let (reg, _) ← displacement (← indirect operand)
    let .port port ← addressOperand lines lines.length reg | none
    if op.startsWith "ld.global." then
      if port == 0 then
        guard (op == "ld.global.v2.u32" && pc < sumReady)
        return true
      guard (port == 1 || port == 2 || port == 3 || port == 4)
      guard (op == "ld.global.nc.u64" || op == "ld.global.nc.f32")
    else
      guard (sumReady < pc)
      guard ((port == 5 && op == "st.global.f32") ||
        (port == 6 && (op == "st.global.v4.u16" || op == "st.global.u8")))
  else
    if op.startsWith "ld." || op.startsWith "st." then
      guard (op.startsWith "ld.shared." || op.startsWith "st.shared." || op.startsWith "ld.param.")
    guard (!(op.startsWith "atom" || op.startsWith "red." || op.startsWith "cp.async"))
    guard (!(line.startsWith "{" && (line.contains "ld." || line.contains "st.")))
    guard (!(pc < sumReady && (op == "ret" || op == "exit" || op == "trap")))
  return false

/-- The second CTA barrier separates reads from writes through the same
global logit allocation. No control edge may skip it or return to the
read phase. The other four input ports and the loss port stay separate. -/
def canReuseLogits (ptx : String) : Bool :=
  let checked : Option Unit := do
    let lines := QKVTTIR.normalizePTX ptx
    let barriers := lines.zipIdx.filterMap fun (line, pc) =>
      if line == "bar.sync 0;" then some pc else none
    let [_, sumReady, _] := barriers | none
    let edges ← flow lines
    guard (edges.toList.zipIdx.all fun (next, pc) =>
      next.all fun target => if pc < sumReady then target ≤ sumReady else sumReady ≤ target)
    let reads ← lines.zipIdx.mapM fun (line, pc) => storageAccess lines sumReady pc line
    guard (reads.contains true)
    return ()
  checked.isSome

structure StorageCertificate (ptx : String) : Type where
  packedReadsBeforeOverwrite : canReuseLogits ptx = true

def certifyStorage (ptx : String) : Option (StorageCertificate ptx) :=
  if h : canReuseLogits ptx = true then some ⟨h⟩ else none

#axiomsClean certifyStorage

private inductive Payload where
  | constant (n : Nat)
  | word (offset : Nat)
  | code (offset : Nat)
  | wideCode (offset : Nat)
  | lookupAddress (offset : Nat)
  | gradientByte (offset : Nat)
  | pairWord (lo hi : Nat)
  | halfWord (lo hi : Nat)
  deriving Inhabited

private abbrev Registers := List (String × Payload)

private def Registers.read (regs : Registers) (key : String) : Option Payload :=
  match regs.lookup key with
  | some v => some v
  | none => key.toNat?.map .constant

private def Registers.set (regs : Registers) (reg : String) (v : Payload) : Registers :=
  (reg, v) :: regs.filter (fun p => p.1 != reg)

private def Registers.forget (regs : Registers) (reg : String) : Registers :=
  regs.filter (fun p => p.1 != reg)

private def transfer (cache : String) (regs : Registers) : List String → Option Registers
  | ["mov.u32", dst, imm] => do
      return regs.set dst (.constant (← imm.toNat?))
  | ["prmt.b32", dst, first, second, selector] => do
      let .constant select ← regs.read selector | none
      match ← regs.read first, ← regs.read second with
      | .word offset, .constant 0 =>
        guard (17472 ≤ select && select ≤ 17475)
        return regs.set dst (.code (offset + select - 17472))
      | .gradientByte hi, .gradientByte lo =>
        guard (select == 30212)
        return regs.set dst (.pairWord lo hi)
      | _, _ => none
  | ["cvt.u64.u32", dst, src] =>
      match regs.read src with
      | some (.code i) => some (regs.set dst (.wideCode i))
      | _ => some (regs.forget dst)
  | ["add.s64", dst, base, index] => do
      if base == cache then
        let .wideCode i ← regs.read index | none
        return regs.set dst (.lookupAddress i)
      else
        return regs.forget dst
  | ["add.s32", dst, _, _] => some (regs.forget dst)
  | ["ld.shared.u8", dst, ptr] => do
      let .lookupAddress i ← regs.read (← indirect ptr) | none
      return regs.set dst (.gradientByte i)
  | ["cvt.u16.u32", dst, src] => do
      let .pairWord lo hi ← regs.read src | none
      return regs.set dst (.halfWord lo hi)
  | _ => none

structure Chunk where
  loadPC : Nat
  storePC : Nat
  inputRegister : String
  displacement : Nat
  outputRegister : String
  inputBytes : List Nat
  deriving Inhabited, DecidableEq

private def pack (regs : Registers) : List String → Option (List Nat)
  | [] => some []
  | reg :: later => do
    let .halfWord lo hi ← regs.read reg | none
    return lo :: hi :: (← pack regs later)

/-- Repeated chunks may use selector registers defined in a block
printed later in the PTX. Seed a fragment only with constant definitions
that dominate its input load and have no other assignment. -/
private def gatherChunks (table : String × Nat) (lines : List String) : Option (List Chunk) := do
  let edges ← flow lines
  let published ← cacheBarrier lines edges table
  let constants := lines.zipIdx.filterMap fun (line, pc) => do
    let ["mov.u32", reg, value] := operands line | none
    let n ← value.toNat?
    guard (n == 0 || (17472 ≤ n && n ≤ 17475))
    guard ((lines.filter (defines reg)).length == 1)
    return (reg, n, pc)
  let mut chunks := []
  let mut registers : Registers := []
  let mut pending : Option (Nat × String × Nat) := none
  for (line, pc) in lines.zipIdx do
    match operands line with
    | ["ld.shared.v2.u32", lo, hi, addr] =>
      guard pending.isNone
      guard (lo != hi)
      guard (published < pc && dominates edges published pc && dominates edges table.2 pc)
      let (base, offset) ← displacement (← indirect addr)
      pending := some (pc, base, offset)
      registers := constants.filterMap fun (reg, n, definition) =>
        if dominates edges definition pc then some (reg, .constant n) else none
      registers := Registers.set (Registers.set registers lo (.word 0)) hi (.word 4)
    | ["st.global.v4.u16", addr, x0, x1, x2, x3] =>
      let some (load, base, offset) := pending | none
      let bytes ← pack registers [x0, x1, x2, x3]
      chunks := ⟨load, pc, base, offset, ← indirect addr, bytes⟩ :: chunks
      pending := none
    | other =>
      if pending.isSome then
        registers ← transfer table.1 registers other
  guard pending.isNone
  return chunks.reverse

def decode (ptx : String) : Option (List Chunk) := do
  let lines := QKVTTIR.normalizePTX ptx
  let table ← tableRegister lines
  let chunks ← gatherChunks table lines
  let first :: _ := chunks | none
  guard (chunks.all fun c => c.inputRegister == first.inputRegister)
  guard ((chunks.map Chunk.displacement).mergeSort == [0, 2048, 4096, 6144])
  return chunks

def Chunk.stores (chunk : Chunk) (shared : Memory) (input output table : Nat) : List Store :=
  chunk.inputBytes.zipIdx |>.map fun (source, dest) =>
    ⟨output + dest, shared (table + (readPackedCode shared (input + source)).toNat)⟩

structure Certificate (ptx : String) where
  chunks : List Chunk
  decoded : decode ptx = some chunks
  routing : ∀ chunk ∈ chunks, chunk.inputBytes = List.range 8

def certify (ptx : String) : Option (Certificate ptx) := do
  if h : (decode ptx).isSome then
    let chunks := (decode ptx).get h
    if route : chunks.all (fun c => c.inputBytes == List.range 8) then
      let decoded : decode ptx = some chunks := Option.some_get h |>.symm
      return ⟨chunks, decoded, fun c hmem => by
        have test := (List.all_eq_true.mp route) c hmem
        exact beq_iff_eq.mp test⟩
    else none
  else none

/-- For any packed input and cache, the compiled store writes the cached
gradient of byte j to position j. The formal execution uses the decoded
payload, including the halfword order in the vector store. -/
theorem packed_store_exact {ptx : String} (cert : Certificate ptx)
    (chunk : Chunk) (inCode : chunk ∈ cert.chunks)
    (shared prior : Memory) (input output table : Nat) (byte : Byte) :
    commitAll (chunk.stores shared input output table) prior (output + byte.val) =
      shared (table + (shared (input + byte.val)).toNat) := by
  have route := cert.routing chunk inCode
  unfold Chunk.stores
  rw [route]
  have rows : (List.range 8).zipIdx = (List.range 8).map (fun i => (i, i)) := by decide
  rw [rows, List.map_map]
  simp only [Function.comp_def, packed_shared_code_exact]
  let stores := (List.range 8).map fun i =>
    (⟨output + i, shared (table + (shared (input + i)).toNat)⟩ : Store)
  have unique : (stores.map Store.address).Nodup := by
    simp only [stores, List.map_map, Function.comp_def]
    apply List.pairwise_map.mpr
    exact List.nodup_range.imp fun neq eq => neq (Nat.add_left_cancel eq)
  have member : (⟨output + byte.val,
      shared (table + (shared (input + byte.val)).toNat)⟩ : Store) ∈ stores :=
    List.mem_map.mpr ⟨byte.val, List.mem_range.mpr byte.isLt, rfl⟩
  exact unique_store_result _ _ _ unique member

/-! Address execution for the integer loop following the loss store.
The row is symbolic. A row-column offset denotes the lower 32 bits of
row * vocabulary + column, widened by the PTX before the global store.
The predicates use only lane-relative columns and loop counters. -/
private inductive Index where
  | integer (i : Int)
  | row
  | codes (offset : Nat)
  | prefixParameter
  | prefixPointer
  | targetParameter
  | targetPointer
  | inputParameter
  | inputBase
  | logits (offset : Nat)
  | outputParameter
  | outputBase
  | rowColumn (offset : Nat)
  | widenedColumn (offset : Nat)
  | output (offset : Nat)
  deriving Inhabited

private abbrev Indices := List (String × Index)

private def intOf (state : Indices) (reg : String) : Option Int :=
  match state.lookup reg with
  | some (.integer i) => some i
  | _ => reg.toInt?

private def natOf (state : Indices) (reg : String) : Option Nat := do
  let i ← intOf state reg
  guard (0 ≤ i)
  return i.toNat

private def signed32 (n : Int) : Bool :=
  -2147483648 ≤ n && n < 2147483648

private def calculate (w : Width) (lane : Lane) (state : Indices) : List String → Option Index
  | ["mov.u32", _, "%tid.x"] => some (.integer lane.val)
  | ["mov.u32", _, "%ctaid.x"] => some .row
  | ["mov.u32", _, n] => do
    let i ← intOf state n
    guard (signed32 i)
    return .integer i
  | ["mov.u64", _, "code_smem"] => some (.codes 0)
  | ["mov.u64", _, src] => state.lookup src
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_0]"] => some .inputParameter
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_1]"] => some .targetParameter
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_6]"] => some .outputParameter
  | ["ld.param.u64", _, "[ce_fwd_bwd_kernel_param_3]"] => some .prefixParameter
  | ["cvta.to.global.u64", _, src] => do
    match state.lookup src with
    | some .inputParameter => some .inputBase
    | some .prefixParameter => some .prefixPointer
    | some .targetParameter => some .targetPointer
    | some .outputParameter => some .outputBase
    | _ => none
  | ["shl.b32", _, src, bits] => do
    let n ← natOf state src
    let k ← bits.toNat?
    guard (k < 32)
    let result := Int.ofNat ((n * 2^k) % 4294967296)
    guard (signed32 result)
    return .integer result
  | ["cvt.s64.s32", _, src] => do
    let i ← intOf state src
    guard (signed32 i)
    return .integer i
  | ["cvt.u64.u32", _, src] =>
    match state.lookup src with
    | some (.rowColumn col) => some (.widenedColumn col)
    | _ => (natOf state src).map fun n => .integer n
  | ["cvt.u32.u64", _, src] => do
    let n ← natOf state src
    return .integer (n % 4294967296)
  | ["mul.lo.s32", _, row, width] => do
    let .row ← state.lookup row | none
    guard (width.toNat? == some w.columns)
    return .rowColumn 0
  | ["mad.lo.s32", _, row, width, offset] => do
    let .row ← state.lookup row | none
    guard (width.toNat? == some w.columns)
    return .rowColumn (← natOf state offset)
  | ["add.s32", _, left, right] => do
    match state.lookup left with
    | some (.rowColumn col) =>
      let out := (Int.ofNat col) + (← intOf state right)
      guard (0 ≤ out)
      return .rowColumn out.toNat
    | _ =>
      let a ← intOf state left
      let b ← intOf state right
      let out := a + b
      guard (signed32 a && signed32 b && signed32 out)
      return .integer out
  | ["add.s64", _, left, right] => do
    match state.lookup left, state.lookup right with
    | some (.codes base), _ => return .codes (base + (← natOf state right))
    | some .outputBase, some (.widenedColumn col) => return .output col
    | some .inputBase, some (.widenedColumn col) => return .logits col
    | some (.widenedColumn col), _ => return .widenedColumn (col + (← natOf state right))
    | _, some (.widenedColumn col) => return .widenedColumn ((← natOf state left) + col)
    | some (.logits col), _ => return .logits (col + (← natOf state right))
    | _, _ => return .integer ((← intOf state left) + (← intOf state right))
  | [op, _, left, right] => do
    let a ← intOf state left
    let b ← intOf state right
    guard (signed32 a && signed32 b)
    let test ← match op with
      | "setp.eq.s32" => some (a == b)
      | "setp.ne.s32" => some (a != b)
      | "setp.gt.s32" => some (decide (a > b))
      | "and.pred" => some (a == 1 && b == 1)
      | _ => none
    return .integer (if test then 1 else 0)
  | _ => none

private def updateIndex (w : Width) (lane : Lane) (state : Indices)
    (insn destinations : List String) : Indices :=
  let rest := state.filter fun p => !destinations.contains p.1
  match destinations, calculate w lane state insn with
  | [dst], some index => (dst, index) :: rest
  | _, _ => rest

private def predecessors (edges : Array (List Nat)) : Array (List Nat) :=
  edges.toList.zipIdx.foldl (fun incoming (successors, src) =>
    successors.foldl (fun a dst => a.modify dst (src :: ·)) incoming)
    (Array.replicate edges.size [])

private def markLive (incoming : Array (List Nat)) :
    Nat → List Nat → Array Bool → Option (Array Bool)
  | _, [], seen => some seen
  | 0, _ :: _, _ => none
  | fuel + 1, pc :: todo, seen =>
    if seen[pc]! then markLive incoming fuel todo seen
    else markLive incoming fuel (incoming[pc]! ++ todo) (seen.set! pc true)

private structure Loop where
  entry : Nat
  instructions : Array (List String × List String)
  edges : Array (List Nat)
  live : Array Bool
  initial : List (List String × List String)
  chunks : List Chunk

/-- Only assignments that uniquely define a live register before the
one-way entry may initialize the loop. Later instructions update it
during execution. -/
private def entryDefinitions (lines : List String) (edges : Array (List Nat)) (entry : Nat) :
    List (List String × List String) :=
  let prelude := lines.take entry
  prelude.zipIdx.filterMap fun (line, pc) => do
    let [reg] := destinations line | none
    guard (!line.startsWith "@" && (prelude.filter (defines reg)).length == 1)
    guard (dominates edges pc entry)
    return (operands line, [reg])

private def compileLoop (ptx : String) (chunks : List Chunk) : Option Loop := do
  let lines := QKVTTIR.normalizePTX ptx
  let edges ← flow lines
  let [lossStore] := lines.zipIdx.filterMap (fun (line, pc) =>
    if line.startsWith "st.global.f32 " then some pc else none) | none
  let entry := lossStore + 1
  guard (reachableWithout edges edges.size entry (2 * edges.size + 1) [0]
    (Array.replicate edges.size false) == some true)
  guard (edges.toList.zipIdx.all fun (successors, pc) =>
    pc < entry || successors.all (entry ≤ ·))
  let returns := lines.zipIdx.filterMap fun (line, pc) =>
    if line == "ret;" then some pc else none
  guard (!returns.isEmpty && returns.all (dominates edges entry))
  guard (chunks.all fun c => dominates edges entry c.loadPC && dominates edges c.loadPC c.storePC)
  let live ← markLive (predecessors edges) (2 * edges.size + chunks.length)
    (chunks.map Chunk.storePC) (Array.replicate edges.size false)
  let instructions := lines.toArray.map fun line => (operands line, destinations line)
  let initial := entryDefinitions lines edges entry
  return ⟨entry, instructions, edges, live, initial, chunks⟩

/-- The first instruction after the packed base-store loop. Every edge
leaving the region that can still issue a packed store must reach this
same instruction. -/
def correctionEntry (ptx : String) : Option Nat := do
  let chunks ← decode ptx
  let loop ← compileLoop ptx chunks
  let exits := loop.edges.toList.zipIdx.flatMap fun (successors, pc) =>
    if loop.entry ≤ pc && loop.live[pc]! then successors.filter (!loop.live[·]!)
    else []
  let [entry] := exits.eraseDups | none
  return entry

private def nextPC (edges : Array (List Nat)) (state : Indices) (pc : Nat) (insn : List String) : Option Nat := do
  match insn with
  | pred :: "bra" :: _ | pred :: "bra.uni" :: _ =>
    guard (pred.startsWith "@")
    let pred := (pred.drop 1).toString
    let inverted := pred.startsWith "!"
    let reg := if inverted then (pred.drop 1).toString else pred
    let value ← intOf state reg
    guard (value == 0 || value == 1)
    if (value == 1) != inverted then (edges[pc]?).bind List.head? else some (pc + 1)
  | op :: _ =>
    if op == "bra" || op == "bra.uni" then (edges[pc]?).bind List.head?
    else do
      guard (!op.startsWith "@")
      return pc + 1
  | [] => some (pc + 1)

structure Placement where
  start : Nat
  fragment : Chunk

private def loopTrace (loop : Loop) (w : Width) (lane : Lane) :
    Nat → Nat → Indices → Option Nat → List Placement → Option (List Placement)
  | 0, _, _, _, _ => none
  | fuel + 1, pc, state, pending, stores => do
    if !(loop.live[pc]?.getD false) then
      guard pending.isNone
      return stores.reverse
    let (insn, defs) ← loop.instructions[pc]?
    let next ← nextPC loop.edges state pc insn
    if let some chunk := loop.chunks.find? (fun c => c.loadPC == pc) then
      guard pending.isNone
      let .codes base ← state.lookup chunk.inputRegister | none
      loopTrace loop w lane fuel next (updateIndex w lane state insn defs)
        (some (base + chunk.displacement)) stores
    else if let some chunk := loop.chunks.find? (fun c => c.storePC == pc) then
      let source ← pending
      let .output column ← state.lookup chunk.outputRegister | none
      guard (source == column && column + 8 ≤ w.columns)
      loopTrace loop w lane fuel next state none (⟨column, chunk⟩ :: stores)
    else
      if pending.isNone then
        let harmless : Bool := match insn with
          | [label] => label.endsWith ":"
          | [".pragma", "\"nounroll\""] => true
          | ["bra", _] | ["bra.uni", _] => true
          | [test, "bra", _] | [test, "bra.uni", _] => test.startsWith "@"
          | _ => (calculate w lane state insn).isSome
        guard harmless
      loopTrace loop w lane fuel next (updateIndex w lane state insn defs) pending stores

private def Loop.execute (loop : Loop) (w : Width) (lane : Lane) : Option (List Placement) :=
  let init := loop.initial.foldl (fun state (i, defs) => updateIndex w lane state i defs) []
  loopTrace loop w lane (loop.instructions.size * (w.chunks + 1)) loop.entry init none []

/-- Ordered starts of all live eight-column chunks owned by this lane. -/
def expectedStarts (w : Width) (lane : Lane) : List Nat :=
  (List.range w.chunks).map (fun chunk => column chunk lane 0) |>.filter (· < w.columns)

def decodedPlacements (w : Width) (ptx : String) (lane : Lane) : Option (List Placement) := do
  let packed ← certify ptx
  let loop ← compileLoop ptx packed.chunks
  loop.execute w lane

def decodedStarts (w : Width) (ptx : String) (lane : Lane) : Option (List Nat) :=
  (decodedPlacements w ptx lane).map (List.map Placement.start)

structure GeometryCertificate (w : Width) (ptx : String) where
  packed : Certificate ptx
  placements : ∀ lane, decodedStarts w ptx lane = some (expectedStarts w lane)
  checkedPayload : ∀ lane (trace : List Placement), decodedPlacements w ptx lane = some trace →
    ∀ issued ∈ trace, issued.fragment ∈ packed.chunks ∧ issued.start + 8 ≤ w.columns

private def Loop.laneValid (loop : Loop) (fragments : List Chunk) (w : Width) (lane : Lane) : Bool :=
  match loop.execute w lane with
  | none => false
  | some trace => (trace.map Placement.start == expectedStarts w lane) &&
      trace.all (fun issued => decide (issued.fragment ∈ fragments ∧ issued.start + 8 ≤ w.columns))

def certifyGeometry (w : Width) (ptx : String) : Option (GeometryCertificate w ptx) :=
  match recognized : certify ptx with
  | none => none
  | some packed =>
    match decoded : compileLoop ptx packed.chunks with
    | none => none
    | some loop =>
      if valid : (List.finRange 256).all (loop.laneValid packed.chunks w) then
        have result (lane : Lane) : decodedPlacements w ptx lane = loop.execute w lane := by
          simp [decodedPlacements, recognized, decoded]
        some ⟨packed, (fun lane => by
          have checked := (List.all_eq_true.mp valid) lane (List.mem_finRange lane)
          rw [decodedStarts, result]
          cases decodedLane : loop.execute w lane with
          | none => simp [Loop.laneValid, decodedLane] at checked
          | some trace =>
            simp only [Loop.laneValid, decodedLane, Bool.and_eq_true] at checked
            exact congrArg some (beq_iff_eq.mp checked.1)), fun lane trace ran issued member => by
          have checked := (List.all_eq_true.mp valid) lane (List.mem_finRange lane)
          rw [result] at ran
          simp only [Loop.laneValid, ran, Bool.and_eq_true] at checked
          exact of_decide_eq_true ((List.all_eq_true.mp checked.2) issued member)⟩
      else none

#axiomsClean packed_store_exact
#axiomsClean certifyGeometry

private inductive ScanValue where
  | word (firstColumn : Nat)
  | code (column : Nat)
  | tableDisplacement (column : Nat)
  | expAddress (column : Nat)
  | contribution (column : Nat)
  | sum (reversed : List Nat)

private abbrev Values := List (String × ScanValue)

private def setScanValue (regs : Values) (reg : String) (value : ScanValue) : Values :=
  (reg, value) :: regs.filter (fun r => r.1 != reg)

structure ScanTrace where
  chunks : List (Nat × Nat)
  sums : List Nat
  deriving DecidableEq

private structure ScanProgram where
  instructions : Array (List String × List String)
  edges : Array (List Nat)
  start : Nat
  stop : Nat
  sumReg : String
  expTable : String
  initial : List (List String × List String)

private def prepareScan (ptx : String) : Option ScanProgram := do
  let lines := QKVTTIR.normalizePTX ptx
  let edges ← flow lines
  let (table, definition) ← tableRegister lines
  let (_, barrier) ← lines.zipIdx.find? (fun (line, _) => line == "bar.sync 0;")
  let (_, firstShuffle) ← lines.zipIdx.find? (fun (line, _) => line.startsWith "shfl.sync.down.b32 ")
  let stop := firstShuffle - 5
  let entry := barrier + 1
  guard (entry < stop && dominates edges definition entry && dominates edges entry stop)
  guard (edges.toList.zipIdx.all fun (successors, pc) => pc < entry || successors.all (entry ≤ ·))
  let ["mov.b32", _, sumReg] := operands (← lines[stop]?) | none
  let code := lines.toArray.map fun line => (operands line, destinations line)
  return ⟨code, edges, entry, stop, sumReg, table, entryDefinitions lines edges entry⟩

private def scanValue (p : ScanProgram) (idx : Indices) (values : Values)
    (insn : List String) : Option ScanValue :=
  match insn with
  | ["mov.f32", _, "0f00000000"] => some (.sum [])
  | ["prmt.b32", _, input, zero, selector] => do
    let .word col ← values.lookup input | none
    guard (natOf idx zero == some 0)
    let s ← natOf idx selector
    guard (17472 ≤ s && s ≤ 17475)
    return .code (col + s - 17472)
  | ["mul.wide.u32", _, input, "4"] => do
    let .code col ← values.lookup input | none
    return .tableDisplacement col
  | ["add.s64", _, table, src] => do
    guard (table == p.expTable)
    let .tableDisplacement col ← values.lookup src | none
    return .expAddress col
  | ["ld.shared.f32", _, addr] => do
    let reg ← indirect addr
    let .expAddress col ← values.lookup reg | none
    return .contribution col
  | ["add.ftz.f32", _, lhs, rhs] =>
    match values.lookup lhs, values.lookup rhs with
    | some (.sum acc), some (.contribution col) => some (.sum (col :: acc))
    -- The zero seed can appear on either side of the first FP32 add.
    -- FP.ScalarOpsLaws.add_comm justifies that normalization.
    | some (.contribution col), some (.sum []) => some (.sum [col])
    | some (.contribution col), _ =>
      if rhs == "0f00000000" then some (.sum [col]) else none
    | _, some (.contribution col) =>
      if lhs == "0f00000000" then some (.sum [col]) else none
    | _, _ => none
  | _ => none

private def runScan (p : ScanProgram) (w : Width) (lane : Lane) :
    Nat → Nat → Indices → Values → List (Nat × Nat) → Option ScanTrace
  | 0, _, _, _, _ => none
  | fuel + 1, pc, idx, values, chunks => do
    if pc == p.stop then
      let .sum cols ← values.lookup p.sumReg | none
      return ⟨chunks.reverse, cols.reverse⟩
    guard (p.start ≤ pc && pc < p.stop)
    let (insn, defs) ← p.instructions[pc]?
    let advance := runScan p w lane fuel (pc + 1)
    match insn with
    | ["ld.global.nc.v2.u32", lo, hi, addr]
    | ["ld.global.v2.u32", lo, hi, addr] =>
      let (base, offset) ← displacement (← indirect addr)
      let .logits start ← idx.lookup base | none
      let col := start + offset
      guard (lo != hi && col + 8 ≤ w.columns)
      let v := setScanValue (setScanValue values lo (.word col)) hi (.word (col + 4))
      advance (idx.filter fun e => e.1 != lo && e.1 != hi) v chunks
    | ["st.shared.v2.u32", addr, lo, hi] =>
      let (base, offset) ← displacement (← indirect addr)
      let .codes start ← idx.lookup base | none
      let .word low ← values.lookup lo | none
      let .word high ← values.lookup hi | none
      let col := start + offset
      guard (high == low + 4 && col + 8 ≤ w.columns)
      advance idx values ((col, low) :: chunks)
    | [".pragma", "\"nounroll\""] => advance idx values chunks
    | [label] =>
      guard (label.endsWith ":")
      advance idx values chunks
    | _ =>
      let next ← nextPC p.edges idx pc insn
      let rest := values.filter fun entry => !(defs.contains entry.1)
      match defs, scanValue p idx values insn with
      | [dst], some value =>
        runScan p w lane fuel next (idx.filter (·.1 != dst)) (setScanValue rest dst value) chunks
      | [dst], none =>
        let integer ← calculate w lane idx insn
        runScan p w lane fuel next ((dst, integer) :: idx.filter (·.1 != dst)) rest chunks
      | [], _ =>
        guard (insn.headD "" == "bra" || insn.headD "" == "bra.uni" || (insn.headD "").startsWith "@")
        runScan p w lane fuel next idx values chunks
      | _, _ => none

private def ScanProgram.execute (p : ScanProgram) (w : Width) (lane : Lane) : Option ScanTrace :=
  let init := p.initial.foldl (fun state (insn, defs) => updateIndex w lane state insn defs) []
  let values : Values := p.initial.filterMap fun (insn, _) =>
    match insn with
    | ["mov.f32", dst, "0f00000000"] => some (dst, .sum [])
    | _ => none
  runScan p w lane (p.instructions.size * (w.chunks + 1)) p.start init values []

def decodeScan (w : Width) (ptx : String) (lane : Lane) : Option ScanTrace := do
  let program ← prepareScan ptx
  program.execute w lane

def expectedScan (w : Width) (lane : Lane) : ScanTrace :=
  { chunks := (expectedStarts w lane).map fun col => (col, col)
    sums := (tableScan w).chunks.flatMap fun chunk =>
      (List.finRange 8).filterMap fun byte =>
        let c := column chunk lane byte
        if c < w.columns then some c else none }

structure ScanCertificate (w : Width) (ptx : String) : Type where
  traces : ∀ lane, decodeScan w ptx lane = some (expectedScan w lane)

def certifyScan (w : Width) (ptx : String) : Option (ScanCertificate w ptx) := do
  match parsed : prepareScan ptx with
  | none => none
  | some program =>
    if checked : (List.finRange 256).all (fun lane =>
        decide (program.execute w lane = some (expectedScan w lane))) then
      some ⟨fun lane => by
        have trace := of_decide_eq_true (List.all_eq_true.mp checked lane (List.mem_finRange lane))
        simpa only [decodeScan, parsed, Option.bind_some] using trace⟩
    else none

def ScanTrace.evalSum (trace : ScanTrace) (read : Nat → F32) :
    F32 :=
  trace.sums.foldl (fun acc col => VeriTile.Triton.FP.add ftz acc (read col))
    (VeriTile.Triton.FP.Value.ofBits .fp32 0)

theorem scan_sum_exact {w : Width} {ptx : String} (cert : ScanCertificate w ptx)
    (lane : Lane) (memory : Memory) :
    (decodeScan w ptx lane).map (fun trace => trace.evalSum fun c =>
      (readSharedTable memory (memory (codeAddress + c))).contribution) =
      some (stagedLaneSums w memory lane) := by
  rw [cert.traces]
  simp only [Option.map_some, Option.some.injEq, ScanTrace.evalSum, expectedScan, List.foldl_flatMap,
    List.foldl_filterMap, stagedLaneSums, laneSumBy]
  congr 1
  funext acc chunk
  congr 1
  funext sum byte
  dsimp only
  by_cases live : column chunk lane byte < w.columns
  · simp only [if_pos live, dif_pos live]
    rfl
  · simp only [if_neg live, dif_neg live]

/-- Each staged byte comes from the same column of the input row. The
decoded chunk trace covers every column in its original owner lane. -/
theorem scan_code_cover {w : Width} {ptx : String} (cert : ScanCertificate w ptx)
    (c : Column w) :
    ∃ (trace : ScanTrace) (start : Nat) (byte : Byte),
      decodeScan w ptx (columnOwner c) = some trace ∧ (start, start) ∈ trace.chunks ∧
        start + byte.val = c.val := by
  obtain ⟨chunk, lane, byte, address⟩ := column_covered w c
  have owner := column_owner_exact c chunk lane byte address
  let start := column chunk lane 0
  have reached : start + byte.val = c.val := by simpa [start, column] using address
  refine ⟨expectedScan w (columnOwner c), start, byte, cert.traces _, ?_, reached⟩
  rw [owner]
  apply List.mem_map.mpr
  refine ⟨start, List.mem_filter.mpr ⟨?_, decide_eq_true ?_⟩, rfl⟩
  · exact List.mem_map.mpr ⟨chunk.val, List.mem_range.mpr chunk.isLt, rfl⟩
  · have bound := c.isLt
    omega

#axiomsClean certifyScan
#axiomsClean scan_sum_exact
#axiomsClean scan_code_cover

end VeriTile.Bench.Models.NanoGPT.Track1.TrainingCEGatherBinding
