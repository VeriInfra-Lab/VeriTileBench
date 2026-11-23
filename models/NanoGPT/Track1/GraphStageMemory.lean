import models.NanoGPT.Track1.CudaStores
import models.NanoGPT.Track1.GraphStage
import models.NanoGPT.Track1.FineWebTraining
import models.NanoGPT.Track1.PinnedBatch
import models.NanoGPT.Track1.NgramSlotPTX
import Init.Data.List.FinRange

/-! The fused launch on physical byte addresses. The host guard separates
input and capture intervals. Allocation contributes a fresh interval for the
pending slot result. Copy CTAs and lookup CTAs share the launch but write each
destination byte once. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphStageMemory
open GraphStage CudaStores

inductive Output where
  | graph (field : Field)
  | pending
  deriving DecidableEq

inductive Buffer where
  | source (field : Field)
  | want
  | directory
  | destination (out : Output)
  deriving DecidableEq

def width : Field → Nat
  | .targets => 8
  | _ => 4

def Output.bytes (s : Shape) : Output → Nat
  | .graph f => extent s f * width f
  | .pending => extent s .slots * 4

def Buffer.bytes (s : Shape) (n : Nat) : Buffer → Nat
  | .source f => extent s f * width f
  | .want => n * 4
  | .directory => (NgramSlots.bucketCount + 1) * 4
  | .destination out => out.bytes s

inductive DType where
  | int32 | int64 | float32

def DType.itemsize : DType → Nat
  | .int64 => 8
  | .int32 | .float32 => 4

def fieldDType : Field → DType
  | .targets => .int64
  | .mtp | .prefix => .float32
  | .inputs | .ends | .slots => .int32

def Buffer.dtype : Buffer → DType
  | .source f | .destination (.graph f) => fieldDType f
  | .want | .directory | .destination .pending => .int32

def Buffer.elements (s : Shape) (n : Nat) : Buffer → Nat
  | .source f | .destination (.graph f) => extent s f
  | .want => n
  | .directory => NgramSlots.bucketCount + 1
  | .destination .pending => extent s .slots

theorem Buffer.dtype_bytes (s : Shape) (n : Nat) (b : Buffer) :
    b.elements s n * b.dtype.itemsize = b.bytes s n := by
  cases b with
  | source f => cases f <;> rfl
  | want | directory => rfl
  | destination o =>
    cases o with
    | graph f => cases f <;> rfl
    | pending => rfl

def Separated (s : Shape) (n : Nat) (base : Buffer → Nat) (a b : Buffer) : Prop :=
  base a + a.bytes s n ≤ base b ∨ base b + b.bytes s n ≤ base a

/-- This is the byte interval condition tested by the launcher, including the
fresh allocation of the pending result. It makes no assertion about values. -/
structure Layout (s : Shape) (n : Nat) where
  base : Buffer → Nat
  separate : ∀ a b, a ≠ b → Separated s n base a b

def Adjacent (s : Shape) (n : Nat) (base : Buffer → Nat) : List Buffer → Prop
  | a :: b :: rest => base a + a.bytes s n ≤ base b ∧ Adjacent s n base (b :: rest)
  | _ => True

private theorem head_before_rest (s : Shape) (n : Nat) (base : Buffer → Nat)
    (a : Buffer) (rest : List Buffer) (guard : Adjacent s n base (a :: rest))
    (b : Buffer) (present : b ∈ rest) : base a + a.bytes s n ≤ base b := by
  induction rest generalizing a with
  | nil => simp at present
  | cons next others ih =>
    have h := guard.1
    rcases List.mem_cons.mp present with rfl | inside
    · exact h
    · have later := ih next guard.2 inside
      omega

/-- The sorted interval check suffices for all pairs, even non-neighbors. -/
theorem adjacent_separates (s : Shape) (n : Nat) (base : Buffer → Nat) (order : List Buffer)
    (guard : Adjacent s n base order) (a b : Buffer) (ha : a ∈ order) (hb : b ∈ order) (hne : a ≠ b) :
    Separated s n base a b := by
  induction order with
  | nil => simp at ha
  | cons head rest ih =>
    rcases List.mem_cons.mp ha with rfl | amongRest
    · have other : b ∈ rest := (List.mem_cons.mp hb).resolve_left (Ne.symm hne)
      exact Or.inl (head_before_rest s n base _ rest guard b other)
    · rcases List.mem_cons.mp hb with rfl | other
      · exact Or.inr (head_before_rest s n base _ rest guard a amongRest)
      · have remains : Adjacent s n base rest := by
          cases rest with
          | nil => exact trivial
          | cons x xs => exact guard.2
        exact ih remains amongRest other

def checkedLayout (s : Shape) (n : Nat) (base : Buffer → Nat) (order : List Buffer)
    (complete : ∀ b, b ∈ order) (guard : Adjacent s n base order) : Layout s n where
  base := base
  separate a b different := adjacent_separates s n base order guard a b (complete a) (complete b) different

/-- After checking shapes and dtypes, the launcher can compute all byte
intervals from the declared format. Sorted adjacent checks still imply
pairwise separation across captured ports. -/
def checkedTensorLayout (s : Shape) (n : Nat) (base : Buffer → Nat) (order : List Buffer)
    (complete : ∀ b, b ∈ order)
    (guard : order.IsChain fun a b => base a + a.elements s n * a.dtype.itemsize ≤ base b) :
    Layout s n := by
  apply checkedLayout s n base order complete
  simp only [Buffer.dtype_bytes] at guard
  clear complete
  induction order with
  | nil => trivial
  | cons a rest ih =>
    cases rest with
    | nil => trivial
    | cons b rest =>
      obtain ⟨head, tail⟩ := List.isChain_cons_cons.mp guard
      exact ⟨head, ih tail⟩

def Contains {s n} (layout : Layout s n) (buffer : Buffer) (address : Nat) : Prop :=
  layout.base buffer ≤ address ∧ address < layout.base buffer + buffer.bytes s n

theorem separated_addresses {s n} (layout : Layout s n) {a b : Buffer} (distinct : a ≠ b)
    {address : Nat} (ha : Contains layout a address) (hb : Contains layout b address) : False := by
  have apart := layout.separate a b distinct
  unfold Separated at apart
  unfold Contains at ha hb
  omega

def readWord32 {s n} (layout : Layout s n) (memory : Memory) (buffer : Buffer) (i : Nat) : Nat :=
  if 4 * (i + 1) ≤ buffer.bytes s n then
    let base := layout.base buffer + 4 * i
    let byte := fun j => (memory (base + j)).toNat
    byte 0 + 256 * byte 1 + 65536 * byte 2 + 16777216 * byte 3
  else 0

def query {s n} (layout : Layout s n) (memory : Memory) (i : Nat) : Int :=
  (Int32.ofNat (readWord32 layout memory (.source .slots) i)).toInt

def lookup {s n} (layout : Layout s n) (memory : Memory) (i : Nat) : Nat :=
  NgramSlots.emitted (readWord32 layout memory .want)
    (readWord32 layout memory .directory) (query layout memory i)

structure Ready {s n} (layout : Layout s n) (memory : Memory) : Prop where
  ordered : ∀ i j, i ≤ j → j < n →
    readWord32 layout memory .want i ≤ readWord32 layout memory .want j
  inVocab : ∀ i, i < n → readWord32 layout memory .want i < 84602880
  directory : NgramSlots.DirectoryReady ⟨n, readWord32 layout memory .want, ordered, inVocab⟩
    (readWord32 layout memory .directory)

def Ready.want {s n} {layout : Layout s n} {memory : Memory} (ready : Ready layout memory) :
    NgramSlots.Want :=
  ⟨n, readWord32 layout memory .want, ready.ordered, ready.inVocab⟩

theorem lookup_exact {s n} {layout : Layout s n} {memory : Memory} (ready : Ready layout memory) (i : Nat) :
    lookup layout memory i = NgramSlots.native false ready.want (query layout memory i) :=
  NgramSlots.emitted_exact ready.want _ ready.directory _

structure Cell (s : Shape) where
  out : Output
  offset : Fin (out.bytes s)

def outputPorts : List Output :=
  [.graph .inputs, .graph .targets, .graph .ends, .graph .slots, .graph .mtp, .graph .prefix, .pending]

def cells (s : Shape) : List (Cell s) :=
  outputPorts.flatMap fun out => (List.finRange (out.bytes s)).map (Cell.mk out)

theorem mem_cells {s : Shape} (cell : Cell s) : cell ∈ cells s := by
  rcases cell with ⟨out, index⟩
  apply List.mem_flatMap.mpr
  refine ⟨out, ?_, List.mem_map.mpr ⟨index, List.mem_finRange _, rfl⟩⟩
  cases out with
  | graph field => cases field <;> decide
  | pending => decide

def owner {s : Shape} (cell : Cell s) : Nat :=
  match cell.out with
  | .graph .slots | .pending => queryCTA s (cell.offset.val / 4)
  | .graph f => cell.offset.val / width f / f.tile

theorem owner_in_grid {s : Shape} (cell : Cell s) : owner cell < s.blocks + queryBlocks s := by
  rcases cell with ⟨out, index⟩
  have bound := index.isLt
  cases out with
  | graph f =>
    cases f <;> cases s <;>
      simp [owner, queryCTA, queryBlocks, extent, Shape.tokens, Shape.ends, Shape.mtp, Shape.blocks,
        Output.bytes, width, Field.tile] at bound ⊢ <;> omega
  | pending =>
    cases s <;> simp [owner, queryCTA, queryBlocks, extent, Shape.tokens, Shape.blocks, Output.bytes]
      at bound ⊢ <;> omega

def Cell.address {s n} (layout : Layout s n) (cell : Cell s) : Nat :=
  layout.base (.destination cell.out) + cell.offset.val

theorem cell_inside {s n} (layout : Layout s n) (cell : Cell s) :
    Contains layout (.destination cell.out) (cell.address layout) := by
  have bound := cell.offset.isLt
  unfold Contains Cell.address
  simp only [Buffer.bytes]
  omega

private theorem different_ports {s n} (layout : Layout s n) (a b : Cell s) (neq : a.out ≠ b.out) :
    a.address layout ≠ b.address layout := by
  intro equal
  have distinct : Buffer.destination a.out ≠ Buffer.destination b.out :=
    fun h => neq (Buffer.destination.inj h)
  exact separated_addresses layout distinct (cell_inside layout a) (equal ▸ cell_inside layout b)

def slotByte (slot offset : Nat) : UInt8 := UInt8.ofNat (slot / 256 ^ (offset % 4))

def emittedByte {s n} (layout : Layout s n) (memory : Memory) (cell : Cell s) : UInt8 :=
  let offset := cell.offset.val
  match cell.out with
  | .graph .slots | .pending => slotByte (lookup layout memory (offset / 4)) offset
  | .graph f => memory (layout.base (.source f) + offset)

def referenceByte {s n} {layout : Layout s n} {memory : Memory} (ready : Ready layout memory)
    (cell : Cell s) : UInt8 :=
  let offset := cell.offset.val
  match cell.out with
  | .graph .slots | .pending =>
      slotByte (NgramSlots.native false ready.want (query layout memory (offset / 4))) offset
  | .graph f => memory (layout.base (.source f) + offset)

theorem emitted_byte_exact {s n} {layout : Layout s n} {memory : Memory} (ready : Ready layout memory)
    (cell : Cell s) : emittedByte layout memory cell = referenceByte ready cell := by
  rcases cell with ⟨out, offset⟩
  cases out with
  | graph f =>
    cases f <;> simp [emittedByte, referenceByte, lookup_exact ready]
  | pending => simp [emittedByte, referenceByte, lookup_exact ready]

def Buffer.readOnly : Buffer → Prop
  | .destination _ => False
  | _ => True

def Reads {s n} (layout : Layout s n) (address : Nat) : Prop :=
  ∃ buffer, buffer.readOnly ∧ Contains layout buffer address

theorem word_read_agrees {s n} (layout : Layout s n) (buffer : Buffer)
    (i : Nat) (mem next : Memory) (agree : ∀ a, Contains layout buffer a → mem a = next a) :
    readWord32 layout mem buffer i = readWord32 layout next buffer i := by
  unfold readWord32
  split
  · rename_i bound
    have same (byte : Nat) (inside : byte < 4) :
        mem (layout.base buffer + 4 * i + byte) = next (layout.base buffer + 4 * i + byte) := by
      apply agree
      unfold Contains
      omega
    simp only [same 0 (by decide), same 1 (by decide), same 2 (by decide), same 3 (by decide)]
  · rfl

theorem word_read_supported {s n} (layout : Layout s n) (buffer : Buffer)
    (readable : buffer.readOnly) (i : Nat) (mem next : Memory)
    (agree : ∀ a, Reads layout a → mem a = next a) :
    readWord32 layout mem buffer i = readWord32 layout next buffer i :=
  word_read_agrees layout buffer i mem next (fun a here => agree a ⟨buffer, readable, here⟩)

private theorem lookup_reads {s n} (layout : Layout s n) (i : Nat) (mem next : Memory)
    (agree : ∀ a, Reads layout a → mem a = next a) :
    lookup layout mem i = lookup layout next i := by
  have want := funext (fun i => word_read_supported layout .want trivial i mem next agree)
  have directory := funext (fun i => word_read_supported layout .directory trivial i mem next agree)
  unfold lookup query
  rw [want, directory, word_read_supported layout (.source .slots) trivial i mem next agree]

private theorem copy_byte_reads {s n} (layout : Layout s n) (f : Field)
    (offset : Fin ((Output.graph f).bytes s)) (mem next : Memory)
    (agree : ∀ a, Reads layout a → mem a = next a) :
    mem (layout.base (.source f) + offset.val) = next (layout.base (.source f) + offset.val) := by
  apply agree
  refine ⟨.source f, trivial, ?_⟩
  have within := offset.isLt
  change offset.val < extent s f * width f at within
  simp only [Contains, Buffer.bytes]
  omega

private theorem emitted_byte_reads {s n} (layout : Layout s n) (cell : Cell s) (mem next : Memory)
    (agree : ∀ a, Reads layout a → mem a = next a) :
    emittedByte layout mem cell = emittedByte layout next cell := by
  rcases cell with ⟨out, offset⟩
  cases out with
  | graph f =>
    cases f <;> simp only [emittedByte]
    all_goals first | exact copy_byte_reads layout _ offset mem next agree
                    | rw [lookup_reads layout _ mem next agree]
  | pending => simp only [emittedByte, lookup_reads layout _ mem next agree]

private theorem address_injective {s n} (layout : Layout s n) (a b : Cell s)
    (same : a.address layout = b.address layout) : a = b := by
  by_cases samePort : a.out = b.out
  · rcases a with ⟨a, i⟩
    rcases b with ⟨b, j⟩
    change a = b at samePort
    subst b
    have sameIndex : i = j := Fin.ext (Nat.add_left_cancel same)
    simp [sameIndex]
  · exact False.elim (different_ports layout a b samePort same)

private theorem fin_range_unique (n : Nat) : (List.finRange n).Nodup := by
  rw [List.Nodup, List.pairwise_iff_getElem]
  intro i j hi hj order same
  have indexes := congrArg Fin.val same
  simp only [List.getElem_finRange, Fin.val_cast] at indexes
  omega

private theorem cell_address_pairs {s n} (layout : Layout s n) :
    (cells s).Pairwise (fun a b => a.address layout ≠ b.address layout) := by
  unfold cells
  apply List.pairwise_flatMap.mpr
  refine ⟨?_, ?_⟩
  · intro out _
    rw [List.pairwise_map]
    apply (fin_range_unique (out.bytes s)).imp
    intro a b unequal same
    exact unequal (Fin.ext (Nat.add_left_cancel same))
  · have baseDistinct : outputPorts.Nodup := by decide
    apply baseDistinct.imp
    intro a b unequal x hx y hy
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp hx
    obtain ⟨j, _, rfl⟩ := List.mem_map.mp hy
    exact different_ports layout _ _ unequal

def assigned (s : Shape) (pid : Nat) : List (Cell s) :=
  (cells s).filter (fun cell => decide (owner cell = pid))

theorem assigned_nodup {s n} (layout : Layout s n) (pid : Nat) : (assigned s pid).Nodup := by
  unfold assigned
  apply List.Pairwise.filter
  apply (cell_address_pairs layout).imp
  intro a b different same
  exact different (congrArg (Cell.address layout) same)

theorem assigned_addresses_nodup {s n} (layout : Layout s n) (pid : Nat) :
    ((assigned s pid).map (Cell.address layout)).Nodup := by
  rw [List.Nodup, List.pairwise_map]
  exact (cell_address_pairs layout).filter _

/-- Each captured-copy byte is emitted by the CTA and Triton lane whose
word address and mask include it. Slot writes use the lookup CTAs. -/
theorem assigned_copy_iff {s : Shape} (f : Field) (copy : f ≠ .slots)
    (offset : Fin ((Output.graph f).bytes s)) (pid : Nat) :
    (Cell.mk (.graph f) offset) ∈ assigned s pid ↔
    ∃ lane, valid s f pid lane ∧ pid * f.tile + lane = offset.val / width f := by
  have nonzero : 0 < f.tile := by cases f <;> decide
  have ownerEquation : owner (Cell.mk (.graph f) offset) = (offset.val / width f) / f.tile := by
    cases f <;> simp [owner] at copy ⊢
  have validOwner : valid s f (offset.val / width f / f.tile) (offset.val / width f % f.tile) := by
    have within := offset.isLt
    cases f <;> cases s <;> simp [valid, extent, Shape.tokens, Shape.ends, Shape.mtp, Shape.blocks,
      Output.bytes, width, Field.tile, Field.firstOnly] at within ⊢ <;> omega
  rw [assigned, List.mem_filter]
  simp only [mem_cells, true_and, decide_eq_true_eq, ownerEquation]
  constructor
  · intro eq
    refine ⟨offset.val / width f % f.tile, eq ▸ validOwner, ?_⟩
    rw [← eq]
    exact Nat.div_add_mod' _ _
  · rintro ⟨lane, live, address⟩
    rw [← address, Nat.mul_comm pid f.tile, Nat.mul_add_div nonzero, Nat.div_eq_of_lt live.2.1, Nat.add_zero]

def slotOutput : Bool → Output
  | false => .graph .slots
  | true => .pending

def slotCell (s : Shape) (pending : Bool) (word : Fin (extent s .slots)) (byte : Fin 4) : Cell s :=
  ⟨slotOutput pending, ⟨4 * word.val + byte.val, by
    have n := word.isLt
    have b := byte.isLt
    cases pending <;> simp only [slotOutput, Output.bytes, width] <;> omega⟩⟩

theorem assigned_slot_iff {s : Shape} (pending : Bool) (word : Fin (extent s .slots))
    (byte : Fin 4) (pid : Nat) :
    slotCell s pending word byte ∈ assigned s pid ↔ pid = queryCTA s word.val := by
  have b := byte.isLt
  have cellIndex : (4 * word.val + byte.val) / 4 = word.val := by omega
  cases pending <;>
    simp only [assigned, List.mem_filter, mem_cells, true_and, decide_eq_true_eq, slotCell,
      slotOutput, owner, cellIndex] <;> exact eq_comm

def makeStore {s n} (layout : Layout s n) (memory : Memory) (cell : Cell s) : Store :=
  ⟨cell.address layout, emittedByte layout memory cell⟩

def block {s n} (layout : Layout s n) (pid : Nat) : Block where
  read := Reads layout
  stores memory := (assigned s pid).map (makeStore layout memory)
  readsOnly mem next agree := List.map_congr_left fun cell _ =>
    congrArg (Store.mk (cell.address layout)) (emitted_byte_reads layout cell mem next agree)

def grid (s : Shape) : List Nat := List.range (s.blocks + queryBlocks s)

def launch {s n} (layout : Layout s n) (schedule : List Nat) (memory : Memory) : Memory :=
  execute (schedule.map (block layout)) memory

private theorem mem_assigned {s pid} {cell : Cell s} (member : cell ∈ assigned s pid) :
    owner cell = pid := by simpa only [assigned, List.mem_filter, decide_eq_true_eq] using
      (List.mem_filter.mp member).2

private theorem block_writes_destination {s n} (layout : Layout s n) (pid : Nat) (memory : Memory)
    (store : Store) (member : store ∈ (block layout pid).stores memory) :
    ∃ cell : Cell s, owner cell = pid ∧ makeStore layout memory cell = store := by
  obtain ⟨cell, here, rfl⟩ := List.mem_map.mp member
  exact ⟨cell, mem_assigned here, rfl⟩

private theorem writes_skip_reads {s n} (layout : Layout s n) (memory : Memory) (pid : Nat)
    (store : Store) (member : store ∈ (block layout pid).stores memory) :
    ¬ Reads layout store.address := by
  obtain ⟨cell, _, rfl⟩ := block_writes_destination layout pid memory store member
  intro reads
  obtain ⟨buffer, ro, atBuffer⟩ := reads
  have different : Buffer.destination cell.out ≠ buffer := by
    intro same
    subst buffer
    exact ro
  exact separated_addresses layout different (cell_inside layout cell) atBuffer

private theorem flat_store_addresses {s n} (layout : Layout s n) (memory : Memory) :
    (((grid s).flatMap fun pid => (block layout pid).stores memory).map Store.address).Nodup := by
  rw [List.Nodup, List.pairwise_map, List.pairwise_flatMap]
  refine ⟨?_, ?_⟩
  · intro pid _
    change ((assigned s pid).map (makeStore layout memory)).Pairwise _
    rw [List.pairwise_map]
    exact (cell_address_pairs layout).filter _
  · have distinct : (grid s).Nodup := List.nodup_range
    apply distinct.imp
    intro a b unequal x hx y hy
    obtain ⟨first, atFirst, rfl⟩ := block_writes_destination layout a memory x hx
    obtain ⟨second, atSecond, rfl⟩ := block_writes_destination layout b memory y hy
    intro same
    have sameCell := address_injective layout first second same
    exact unequal (atFirst.symm.trans (sameCell ▸ atSecond))

private theorem block_reads_independent {s n} (layout : Layout s n) (memory : Memory)
    (writer : Block) (inGrid : writer ∈ (grid s).map (block layout))
    (store : Store) (writes : store ∈ writer.stores memory)
    (reader : Block) (readerInGrid : reader ∈ (grid s).map (block layout)) :
    ¬ reader.read store.address := by
  obtain ⟨writePid, _, rfl⟩ := List.mem_map.mp inGrid
  obtain ⟨readPid, _, rfl⟩ := List.mem_map.mp readerInGrid
  exact writes_skip_reads layout memory writePid store writes

def writes {s n} (layout : Layout s n) (address : Nat) : Prop :=
  ∃ cell : Cell s, cell.address layout = address

theorem schedules_equal {s n} (layout : Layout s n) (memory : Memory)
    (schedule : List Nat) (complete : (grid s).Perm schedule) :
    launch layout (grid s) memory = launch layout schedule memory := by
  unfold launch
  apply launch_order_independent _ _ _ (complete.map (block layout)) (block_reads_independent layout memory)
  rw [List.flatMap_map]
  exact flat_store_addresses layout memory

/-- All CTA schedules write the reference result in both slot buffers and the
other captured inputs. Every byte outside those seven buffers is preserved. -/
theorem launch_exact {s n} (layout : Layout s n) (memory : Memory) (ready : Ready layout memory)
    (schedule : List Nat) (complete : (grid s).Perm schedule) :
    (∀ cell : Cell s, launch layout schedule memory (cell.address layout) = referenceByte ready cell) ∧
    (∀ address, ¬ writes layout address → launch layout schedule memory address = memory address) := by
  rw [← schedules_equal layout memory schedule complete]
  unfold launch
  rw [independent_block_reads _ memory memory (fun _ _ _ _ => rfl) (block_reads_independent layout memory),
    List.flatMap_map]
  constructor
  · intro cell
    have here : owner cell ∈ grid s := List.mem_range.mpr (owner_in_grid cell)
    have scheduled : cell ∈ assigned s (owner cell) := List.mem_filter.mpr ⟨mem_cells cell, by simp⟩
    have listed : makeStore layout memory cell ∈
        (grid s).flatMap (fun pid => (block layout pid).stores memory) := List.mem_flatMap.mpr
      ⟨owner cell, here, List.mem_map.mpr ⟨cell, scheduled, rfl⟩⟩
    rw [← emitted_byte_exact ready cell]
    exact unique_store_result _ _ _ (flat_store_addresses layout memory) listed
  · intro addr outside
    apply preserve_unwritten
    intro item present equal
    obtain ⟨pid, _, here⟩ := List.mem_flatMap.mp present
    obtain ⟨cell, _, rfl⟩ := block_writes_destination layout pid memory item here
    exact outside ⟨cell, equal⟩

theorem stage_reads_preserved {s n} (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (schedule : List Nat) (complete : (grid s).Perm schedule)
    (address : Nat) (input : Reads layout address) : launch layout schedule memory address = memory address := by
  apply (launch_exact layout memory ready schedule complete).2
  rintro ⟨cell, equal⟩
  obtain ⟨buffer, readOnly, here⟩ := input
  have apart : Buffer.destination cell.out ≠ buffer := by
    intro same
    subst buffer
    exact readOnly
  exact separated_addresses layout apart (equal ▸ cell_inside layout cell) here

private theorem slot_byte_roundtrip (slot : Nat) (fits : slot < 4294967296) :
    (slotByte slot 0).toNat + 256 * (slotByte slot 1).toNat +
      65536 * (slotByte slot 2).toNat + 16777216 * (slotByte slot 3).toNat = slot := by
  have decoded (n : Nat) : (UInt8.ofNat n).toNat = n % 256 := by rfl
  simp only [slotByte, decoded]
  change (slot / 1) % 256 + 256 * ((slot / 256) % 256) + 65536 * ((slot / 65536) % 256) +
    16777216 * ((slot / 16777216) % 256) = slot
  rw [Nat.div_one]
  omega

theorem read_word32_of_bytes {s n} (layout : Layout s n) (memory : Memory)
    (buffer : Buffer) (word value : Nat) (inside : 4 * (word + 1) ≤ buffer.bytes s n)
    (fits : value < 4294967296)
    (bytes : ∀ j < 4, memory (layout.base buffer + 4 * word + j) = slotByte value j) :
    readWord32 layout memory buffer word = value := by
  unfold readWord32
  rw [if_pos inside]
  simp only [bytes 0 (by decide), bytes 1 (by decide), bytes 2 (by decide), bytes 3 (by decide)]
  exact slot_byte_roundtrip value fits

/-- Read either published slot tensor using its actual little-endian i32
word width. This identifies the value consumed by replay and the value kept
for gradient accumulation with the original full-table lower bound. -/
theorem published_slot_word {s n} (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (schedule : List Nat) (complete : (grid s).Perm schedule)
    (pending : Bool) (word : Fin (extent s .slots)) (fits : n < 2147483648) :
    readWord32 layout (launch layout schedule memory) (.destination (slotOutput pending)) word.val =
      NgramSlots.native false ready.want (query layout memory word.val) := by
  let result := NgramSlots.native false ready.want (query layout memory word.val)
  have resultBound : result ≤ n := NgramSlots.native_within ready.want _
  have readFits : 4 * (word.val + 1) ≤ (Buffer.destination (slotOutput pending)).bytes s n := by
    have index := word.isLt
    cases pending <;> simp only [slotOutput, Buffer.bytes, Output.bytes, width] <;> omega
  have byte (i : Nat) (inside : i < 4) :
      launch layout schedule memory (layout.base (.destination (slotOutput pending)) + 4 * word.val + i) =
        slotByte result i := by
    have exactByte := (launch_exact layout memory ready schedule complete).1 (slotCell s pending word ⟨i, inside⟩)
    have wordIndex : (4 * word.val + i) / 4 = word.val := by omega
    have byteIndex : (4 * word.val + i) % 4 = i % 4 := by omega
    cases pending <;> simpa only [slotCell, slotOutput, Cell.address, Nat.add_assoc, referenceByte,
      slotByte, wordIndex, byteIndex, result] using exactByte
  unfold readWord32
  rw [if_pos readFits]
  simp only [byte 0 (by decide), byte 1 (by decide), byte 2 (by decide), byte 3 (by decide)]
  exact slot_byte_roundtrip result (by omega)

private theorem published_copy_piece {s n} (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (schedule : List Nat) (complete : (grid s).Perm schedule)
    (field : Field) (copied : field ≠ .slots) (word : Nat)
    (within : 4 * (word + 1) ≤ extent s field * width field) :
    readWord32 layout (launch layout schedule memory) (.destination (.graph field)) word =
      readWord32 layout memory (.source field) word := by
  let target := layout.base (.destination (.graph field))
  let source := layout.base (.source field)
  have getByte (i : Nat) (small : i < 4) :
      launch layout schedule memory (target + 4 * word + i) = memory (source + 4 * word + i) := by
    have inside : 4 * word + i < (Output.graph field).bytes s := by
      simp only [Output.bytes]
      omega
    have exactByte := (launch_exact layout memory ready schedule complete).1 ⟨.graph field, ⟨4 * word + i, inside⟩⟩
    cases field <;>
      simp only [Cell.address, referenceByte, target, source, Nat.add_assoc] at exactByte ⊢
    all_goals first | exact exactByte | exact False.elim (copied rfl)
  unfold readWord32
  simp only [Buffer.bytes, Output.bytes, within, if_true]
  change (launch layout schedule memory (target + 4 * word + 0)).toNat +
      256 * (launch layout schedule memory (target + 4 * word + 1)).toNat +
      65536 * (launch layout schedule memory (target + 4 * word + 2)).toNat +
      16777216 * (launch layout schedule memory (target + 4 * word + 3)).toNat = _
  rw [getByte 0 (by decide), getByte 1 (by decide), getByte 2 (by decide), getByte 3 (by decide)]

def readField {s n} (layout : Layout s n) (memory : Memory) (buffer : Field → Buffer) :
    StaticInputs s :=
  fun field word =>
    let piece := readWord32 layout memory (buffer field)
    match field with
    | .targets => UInt64.ofNat (piece (2 * word.val) + 4294967296 * piece (2 * word.val + 1))
    | _ => UInt64.ofNat (piece word.val)

def observe {s n} (layout : Layout s n) (memory : Memory) : Observation s :=
  ⟨readField layout memory (fun f => .destination (.graph f)),
    fun word => readWord32 layout memory (.destination .pending) word.val⟩

/-- The byte-addressed execution has the same observation as the original
lookup followed by the graph copies, for every complete CTA schedule. -/
theorem launch_matches_observation {s n} (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (schedule : List Nat) (complete : (grid s).Perm schedule)
    (fits : n < 2147483648) :
    observe layout (launch layout schedule memory) =
      separateLookup s (readField layout memory .source) ready.want (fun i => query layout memory i.val) := by
  simp only [observe, separateLookup, eager, Observation.mk.injEq]
  constructor
  · funext field word
    have within := word.isLt
    cases field <;> simp only [readField, replaceSlots]
    · rw [published_copy_piece layout memory ready schedule complete .inputs (by decide) word.val
        (by simp only [width]; omega)]
    · rw [published_copy_piece layout memory ready schedule complete .targets (by decide) (2 * word.val)
        (by simp only [width]; omega),
        published_copy_piece layout memory ready schedule complete .targets (by decide) (2 * word.val + 1)
        (by simp only [width]; omega)]
    · rw [published_copy_piece layout memory ready schedule complete .ends (by decide) word.val
        (by simp only [width]; omega)]
    · exact congrArg UInt64.ofNat (published_slot_word layout memory ready schedule complete false word fits)
    · rw [published_copy_piece layout memory ready schedule complete .mtp (by decide) word.val
        (by simp only [width]; omega)]
    · rw [published_copy_piece layout memory ready schedule complete .prefix (by decide) word.val
        (by simp only [width]; omega)]
  · funext word
    exact published_slot_word layout memory ready schedule complete true word fits

/-- The launch refines the graph-stage port used by the training driver
and has the same frame for every ordering of the launched CTAs. -/
def ReplayRefinement : Prop :=
  ∀ (s : Shape) (n : Nat) (layout : Layout s n) (memory : Memory) (ready : Ready layout memory)
    (schedule : List Nat), (grid s).Perm schedule → n < 2147483648 →
    observe layout (launch layout schedule memory) =
      fusedLookup s (readField layout memory .source) ready.want (fun i => query layout memory i.val) ∧
    (∀ address, ¬ writes layout address → launch layout schedule memory address = memory address)

theorem replay_refines : ReplayRefinement := by
  intro s n layout memory ready schedule complete fits
  rw [lookup_stage_equal]
  exact ⟨launch_matches_observation layout memory ready schedule complete fits,
    (launch_exact layout memory ready schedule complete).2⟩

/-- The slot-query tensor was uploaded from this FineWeb batch. The
hash identity is proved at packing and maintained by the lookahead cache. -/
structure UploadedHashes {s n} (layout : Layout s n) (memory : Memory)
    (batch : FineWeb.TokenBatch) : Prop where
  count : batch.ngramRows.size = extent s .slots
  word : ∀ i : Fin (extent s .slots),
    readWord32 layout memory (.source .slots) i.val =
      (batch.ngramRows[i.val]'(by rw [count]; exact i.isLt)).toNat

/-- The packed loader uploads four adjacent fields. The fallback uploads
the n-gram tensor separately to its own allocation. -/
inductive BatchTransfer where
  | packed (base : Nat) (padding : Memory)
  | separate

def BatchTransfer.rowPlacement {s n} (transfer : BatchTransfer) (layout : Layout s n)
    (batch : FineWeb.TokenBatch) : Prop :=
  match transfer with
  | .packed base _ =>
    layout.base (.source .slots) = base + PinnedBatch.start (PinnedBatch.counts batch) .ngram
  | .separate => True

def BatchTransfer.upload {s n} (transfer : BatchTransfer) (layout : Layout s n)
    (batch : FineWeb.TokenBatch) (prior : Memory) : Memory :=
  match transfer with
  | .packed base padding => PinnedBatch.uploadBytes batch base padding prior
  | .separate => PinnedBatch.uploadFieldBytes batch .ngram (layout.base (.source .slots)) prior

def BatchTransfer.touches {s n} (transfer : BatchTransfer) (layout : Layout s n)
    (batch : FineWeb.TokenBatch) (address : Nat) : Prop :=
  match transfer with
  | .packed base _ => base ≤ address ∧ address < base + PinnedBatch.stop (PinnedBatch.counts batch) .ngram
  | .separate =>
    let base := layout.base (.source .slots)
    base ≤ address ∧ address < base + PinnedBatch.size (PinnedBatch.counts batch) .ngram

theorem BatchTransfer.frame {s n} (transfer : BatchTransfer) (layout : Layout s n)
    (batch : FineWeb.TokenBatch) (prior : Memory) (address : Nat)
    (outside : ¬ transfer.touches layout batch address) :
    transfer.upload layout batch prior address = prior address := by
  cases transfer with
  | separate => exact PinnedBatch.uploadFieldBytes_frame batch .ngram _ prior address outside
  | packed base padding => exact PinnedBatch.uploadBytes_frame batch base padding prior address outside

theorem transfer_uploads_hashes {s n} (transfer : BatchTransfer) (layout : Layout s n)
    (batch : FineWeb.TokenBatch) (prior : Memory) (count : batch.ngramRows.size = extent s .slots)
    (placement : transfer.rowPlacement layout batch) :
    UploadedHashes layout (transfer.upload layout batch prior) batch := by
  refine ⟨count, ?_⟩
  intro i
  have index : i.val < batch.ngramRows.size := by rw [count]; exact i.isLt
  have fits : 4 * (i.val + 1) ≤ (Buffer.source .slots).bytes s n := by
    have inside := i.isLt
    simp only [Buffer.bytes, width]
    omega
  let row := (batch.ngramRows[i.val]'index).toNat
  have byte (j : Fin 4) :
      transfer.upload layout batch prior (layout.base (.source .slots) + 4 * i.val + j.val) =
        slotByte row j.val := by
    have within := Nat.mod_eq_of_lt j.isLt
    cases transfer with
    | packed base padding =>
      change layout.base (.source .slots) =
        base + PinnedBatch.start (PinnedBatch.counts batch) .ngram at placement
      simp only [BatchTransfer.upload, placement]
      have copied := PinnedBatch.upload_byte_exact batch base padding prior .ngram ⟨i.val, index⟩ j
      simpa only [slotByte, row, within, PinnedBatch.Field.width, PinnedBatch.words,
        Nat.mul_comm i.val 4, UInt32.toNat_toUInt64] using copied
    | separate =>
      have copied := PinnedBatch.upload_field_byte_exact batch .ngram (layout.base (.source .slots))
        prior ⟨i.val, index⟩ j
      simpa only [BatchTransfer.upload, slotByte, row, within, PinnedBatch.Field.width, PinnedBatch.words,
        Nat.mul_comm i.val 4, UInt32.toNat_toUInt64] using copied
  unfold readWord32
  rw [if_pos fits]
  simp only [byte ⟨0, by decide⟩, byte ⟨1, by decide⟩, byte ⟨2, by decide⟩, byte ⟨3, by decide⟩]
  exact slot_byte_roundtrip row (UInt32.toNat_lt (batch.ngramRows[i.val]'index))

theorem hashed_query_in_range {s n} (layout : Layout s n) (memory : Memory)
    (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows)
    (upload : UploadedHashes layout memory batch) (i : Fin (extent s .slots)) :
    0 ≤ query layout memory i.val ∧ (query layout memory i.val).toNat < 84602880 := by
  have index : i.val < batch.ngramRows.size := by rw [upload.count]; exact i.isLt
  let word := batch.ngramRows[i.val]'index
  have member : word ∈ batch.ngramRows := Array.getElem_mem index
  have bound := FineWeb.hash_rows_range batch.inputs batch.ngramRows hashed word member
  have cast : (Int32.ofNat word.toNat).toInt = word.toInt32.toInt := by
    rw [Int32.toInt_ofNat_of_lt (Nat.lt_trans bound (by decide)), FineWeb.decode_hashed_row word bound]
    rfl
  have uploaded : query layout memory i.val = word.toInt32.toInt := by
    unfold query
    rw [upload.word i]
    exact cast
  rw [uploaded]
  exact FineWeb.batch_has_in_range_rows batch hashed word member

theorem uploaded_query_word {s n} (layout : Layout s n) (memory : Memory)
    (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows)
    (upload : UploadedHashes layout memory batch) (i : Fin (extent s .slots)) :
    (query layout memory i.val).toNat =
      (batch.ngramRows[i.val]'(by rw [upload.count]; exact i.isLt)).toNat := by
  have inside : i.val < batch.ngramRows.size := by rw [upload.count]; exact i.isLt
  have bound := FineWeb.hash_rows_range batch.inputs batch.ngramRows hashed _ (Array.getElem_mem inside)
  unfold query
  rw [upload.word i, Int32.toInt_ofNat_of_lt (Nat.lt_trans bound (by decide)), Int.toNat_natCast]

def lookupHashed {s n} (layout : Layout s n) (memory : Memory) (i : Nat) : Nat :=
  (NgramSlotPTX.lookup (readWord32 layout memory .want)
    (readWord32 layout memory .directory) (query layout memory i).toNat).toNat

def hashedByte {s n} (layout : Layout s n) (memory : Memory) (cell : Cell s) : UInt8 :=
  let offset := cell.offset.val
  match cell.out with
  | .graph .slots | .pending => slotByte (lookupHashed layout memory (offset / 4)) offset
  | .graph f => memory (layout.base (.source f) + offset)

theorem hashed_byte_reads {s n} (layout : Layout s n) (cell : Cell s) (mem next : Memory)
    (agree : ∀ address, Reads layout address → mem address = next address) :
    hashedByte layout mem cell = hashedByte layout next cell := by
  have sizes (i : Nat) : lookupHashed layout mem i = lookupHashed layout next i := by
    have want := funext (fun i => word_read_supported layout .want trivial i mem next agree)
    have directory := funext (fun i => word_read_supported layout .directory trivial i mem next agree)
    unfold lookupHashed query
    rw [want, directory, word_read_supported layout (.source .slots) trivial i mem next agree]
  rcases cell with ⟨out, offset⟩
  cases out with
  | graph f =>
    cases f <;> simp only [hashedByte]
    all_goals first | exact copy_byte_reads layout _ offset mem next agree | rw [sizes]
  | pending => simp only [hashedByte, sizes]

private theorem hashed_byte_eq {s n} (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (fits : n < 2147483648) (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows)
    (upload : UploadedHashes layout memory batch) (cell : Cell s) :
    hashedByte layout memory cell = emittedByte layout memory cell := by
  have same (offset : Nat) (inside : offset < extent s .slots * 4) :
      lookupHashed layout memory (offset / 4) = lookup layout memory (offset / 4) := by
    let i : Fin (extent s .slots) := ⟨offset / 4, by omega⟩
    have range := hashed_query_in_range layout memory batch hashed upload i
    have machine := NgramSlotPTX.lookup_exact ready.want _ ready.directory fits _ range.2
    have signed := Int.toNat_of_nonneg range.1
    exact (by simpa only [Ready.want, signed] using machine :
      lookupHashed layout memory (offset / 4) = NgramSlots.native false ready.want (query layout memory i.val))
      |>.trans (lookup_exact ready i.val).symm
  rcases cell with ⟨out, offset⟩
  cases out with
  | graph f =>
    cases f <;> simp only [hashedByte, emittedByte]
    rw [same offset.val offset.isLt]
  | pending => simp only [hashedByte, emittedByte, same offset.val offset.isLt]

def hashedStore {s n} (layout : Layout s n) (memory : Memory) (cell : Cell s) : Store :=
  ⟨cell.address layout, hashedByte layout memory cell⟩

def hashedBlock {s n} (layout : Layout s n) (pid : Nat) : Block where
  read := Reads layout
  stores memory := (assigned s pid).map (hashedStore layout memory)
  readsOnly mem next agree := List.map_congr_left fun cell _ =>
    congrArg (Store.mk (cell.address layout)) (hashed_byte_reads layout cell mem next agree)

def launchHashed {s n} (layout : Layout s n) (schedule : List Nat) (memory : Memory) : Memory :=
  execute (schedule.map (hashedBlock layout)) memory

private theorem hashed_blocks_preserve_reads {s n} (layout : Layout s n) (memory : Memory)
    (schedule : List Nat) (writer : Block) (inSchedule : writer ∈ schedule.map (hashedBlock layout))
    (store : Store) (writes : store ∈ writer.stores memory)
    (reader : Block) (readsSchedule : reader ∈ schedule.map (hashedBlock layout)) :
    ¬ reader.read store.address := by
  obtain ⟨wpid, _, rfl⟩ := List.mem_map.mp inSchedule
  obtain ⟨rpid, _, rfl⟩ := List.mem_map.mp readsSchedule
  obtain ⟨cell, owned, rfl⟩ := List.mem_map.mp writes
  exact writes_skip_reads layout memory wpid (makeStore layout memory cell)
    (List.mem_map.mpr ⟨cell, owned, rfl⟩)

theorem hashed_launch_equal {s n} (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (fits : n < 2147483648)
    (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows) (upload : UploadedHashes layout memory batch)
    (schedule : List Nat) : launchHashed layout schedule memory = launch layout schedule memory := by
  have genericRead (writer : Block) (inSchedule : writer ∈ schedule.map (block layout))
      (store : Store) (writes : store ∈ writer.stores memory)
      (reader : Block) (readsSchedule : reader ∈ schedule.map (block layout)) :
      ¬ reader.read store.address := by
    obtain ⟨wpid, _, rfl⟩ := List.mem_map.mp inSchedule
    obtain ⟨rpid, _, rfl⟩ := List.mem_map.mp readsSchedule
    exact writes_skip_reads layout memory wpid store writes
  unfold launchHashed launch
  rw [independent_block_reads _ memory memory (fun _ _ _ _ => rfl)
      (hashed_blocks_preserve_reads layout memory schedule),
    independent_block_reads _ memory memory (fun _ _ _ _ => rfl) genericRead,
    List.flatMap_map, List.flatMap_map]
  have perCTA (pid : Nat) : (hashedBlock layout pid).stores memory = (block layout pid).stores memory :=
    List.map_congr_left fun cell _ =>
      congrArg (Store.mk (cell.address layout)) (hashed_byte_eq layout memory ready fits batch hashed upload cell)
  simp only [perCTA]

def HashedReplayRefinement : Prop :=
  ∀ (s : Shape) (n : Nat) (layout : Layout s n) (memory : Memory) (ready : Ready layout memory)
    (batch : FineWeb.TokenBatch), batch.hashedRows → UploadedHashes layout memory batch →
    ∀ (schedule : List Nat), (grid s).Perm schedule → n < 2147483648 →
    observe layout (launchHashed layout schedule memory) =
      fusedLookup s (readField layout memory .source) ready.want (fun i => query layout memory i.val) ∧
    (∀ address, ¬ writes layout address → launchHashed layout schedule memory address = memory address)

theorem hashed_replay_refines : HashedReplayRefinement := by
  intro s n layout memory ready batch hashed upload schedule complete fits
  rw [hashed_launch_equal layout memory ready fits batch hashed upload schedule]
  exact replay_refines s n layout memory ready schedule complete fits

#axiomsClean transfer_uploads_hashes
#axiomsClean hashed_query_in_range
#axiomsClean hashed_replay_refines

#axiomsClean adjacent_separates
#axiomsClean Buffer.dtype_bytes
#axiomsClean checkedTensorLayout
#axiomsClean lookup_exact
#axiomsClean schedules_equal
#axiomsClean launch_exact
#axiomsClean assigned_copy_iff
#axiomsClean assigned_slot_iff
#axiomsClean published_slot_word
#axiomsClean launch_matches_observation
#axiomsClean replay_refines

end VeriTile.Bench.Models.NanoGPT.Track1.GraphStageMemory
