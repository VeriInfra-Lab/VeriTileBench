import models.NanoGPT.Track1.GraphPlainPTX

/-! The ordinary stage's physical byte stores and read footprint. The host
guard covers exactly the six source and six destination intervals. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphPlainMemory
open GraphStage CudaStores
open GraphStageMemory (width)

abbrev Port := Bool × Field

def bytes (s : Shape) (f : Field) : Nat := extent s f * width f

structure Layout (s : Shape) where
  base : Port → Nat
  aligned : ∀ port, base port % 16 = 0
  separate : ∀ a b, a ≠ b →
    base a + bytes s a.2 ≤ base b ∨ base b + bytes s b.2 ≤ base a

def Layout.contains {s : Shape} (layout : Layout s) (port : Port) (address : Nat) : Prop :=
  layout.base port ≤ address ∧ address < layout.base port + bytes s port.2

def checkedLayout (s : Shape) (base : Port → Nat) (ports : List Port)
    (complete : ∀ port, port ∈ ports)
    (aligned : ∀ port ∈ ports, base port % 16 = 0)
    (guard : ports.IsChain fun a b => base a + bytes s a.2 ≤ base b) : Layout s := by
  refine ⟨base, fun port => aligned port (complete port), ?_⟩
  let before : Port → Port → Prop := fun a b => base a + bytes s a.2 ≤ base b
  letI : Trans before before before := ⟨fun {a b c} ab bc => by
    dsimp [before] at ab bc ⊢
    omega⟩
  have pairs : ports.Pairwise before := (show ports.IsChain before from guard).pairwise
  have order : ∀ (order : List Port), order.Pairwise before → ∀ a b, a ≠ b →
      a ∈ order → b ∈ order → before a b ∨ before b a := by
    intro order pairs
    induction order with
    | nil => simp
    | cons first rest ih =>
      intro a b different ha hb
      obtain ⟨head, tail⟩ := List.pairwise_cons.mp pairs
      rcases List.mem_cons.mp ha with rfl | present
      · exact Or.inl (head _ ((List.mem_cons.mp hb).resolve_left (Ne.symm different)))
      rcases List.mem_cons.mp hb with rfl | other
      · exact Or.inr (head _ present)
      · exact ih tail a b different present other
  exact fun a b different => order ports pairs a b different (complete a) (complete b)

private theorem different_ports {s : Shape} (layout : Layout s) (a b : Port) (unequal : a ≠ b)
    (address : Nat) (first : layout.contains a address) (second : layout.contains b address) : False := by
  have apart := layout.separate a b unequal
  unfold Layout.contains at first second
  omega

def blockBytes (s : Shape) (pid : Nat) : List (Field × Nat) :=
  (List.finRange 128).flatMap fun tid => (GraphPlainPTX.words s pid tid true).flatMap GraphCopyPTX.wordBytes

theorem vector_alignment {s : Shape} (layout : Layout s) (dest : Bool)
    (pid : Nat) (fits : pid < 192) (tid : Fin 128) :
    let first := (GraphCopyPTX.index (UInt32.ofNat pid) (UInt32.ofNat tid.val)).toNat
    let slot := (GraphPlainPTX.slotIndex (UInt32.ofNat pid) (UInt32.ofNat tid.val)).toNat
    (layout.base (dest, .inputs) + first * 4) % 8 = 0 ∧
      (layout.base (dest, .targets) + first * 8) % 16 = 0 ∧
      (layout.base (dest, .slots) + slot * 4) % 16 = 0 := by
  dsimp only
  rw [GraphCopyPTX.index_exact pid tid.val fits tid.isLt,
    GraphPlainPTX.slot_index_exact pid tid.val fits tid.isLt]
  have inputs := layout.aligned (dest, .inputs)
  have targets := layout.aligned (dest, .targets)
  have slots := layout.aligned (dest, .slots)
  omega

private theorem width_pos (f : Field) : 0 < width f := by cases f <;> decide

theorem block_byte_iff (s : Shape) (pid : Nat) (live : pid < s.blocks) (f : Field) (i : Nat) :
    (f, i) ∈ blockBytes s pid ↔ i < bytes s f ∧ i / width f / f.tile = pid := by
  simp only [blockBytes, List.mem_flatMap, List.mem_finRange, true_and, GraphCopyPTX.byte_of_word]
  rw [bytes, ← Nat.div_lt_iff_lt_mul (width_pos f)]
  constructor
  · rintro ⟨tid, _, written, rfl⟩
    exact ((GraphPlainPTX.stores_word_iff s pid live tid f _).mp written).imp_right And.left
  · rintro ⟨bound, owner⟩
    let tid : Fin 128 := ⟨_, GraphPlainPTX.word_lane_bound s f ⟨i / width f, bound⟩⟩
    exact ⟨tid, _, (GraphPlainPTX.stores_word_iff s pid live tid f _).mpr ⟨bound, owner, rfl⟩, rfl⟩

def read {s : Shape} (layout : Layout s) (addr : Nat) : Prop :=
  ∃ f : Field, layout.contains (false, f) addr

def makeStore {s : Shape} (layout : Layout s) (mem : Memory) (byte : Field × Nat) : Store :=
  ⟨layout.base (true, byte.1) + byte.2, mem (layout.base (false, byte.1) + byte.2)⟩

def block {s : Shape} (layout : Layout s) (pid : Nat) : Block where
  read := read layout
  stores mem := if pid < s.blocks then (blockBytes s pid).map (makeStore layout mem) else []
  readsOnly mem next same := by
    split
    · rename_i live
      apply List.map_congr_left
      rintro ⟨field, i⟩ member
      have inside := ((block_byte_iff s pid live field i).mp member).1
      have atSource : read layout (layout.base (false, field) + i) := by
        refine ⟨field, ?_⟩
        unfold Layout.contains
        dsimp only
        omega
      simp only [makeStore, same _ atSource]
    · rfl

def launch {s : Shape} (layout : Layout s) (order : List Nat) (mem : Memory) : Memory :=
  execute (order.map (block layout)) mem

private theorem store_produced {s : Shape} (layout : Layout s) (mem : Memory)
    (pid : Nat) (w : Store) :
    w ∈ (block layout pid).stores mem ↔
      ∃ (f : Field) (i : Nat), pid < s.blocks ∧ i < bytes s f ∧ i / width f / f.tile = pid ∧
        makeStore layout mem (f, i) = w := by
  by_cases live : pid < s.blocks
  · simp only [block, List.mem_map, Prod.exists, block_byte_iff s pid live, live, if_true, true_and, and_assoc]
  · simp [block, live]

private theorem store_inside {s : Shape} (layout : Layout s) (mem : Memory)
    (pid : Nat) (w : Store) (produced : w ∈ (block layout pid).stores mem) :
    ∃ f : Field, layout.contains (true, f) w.address := by
  obtain ⟨field, i, _, inside, _, rfl⟩ := (store_produced layout mem pid w).mp produced
  refine ⟨field, ?_⟩
  unfold Layout.contains makeStore
  dsimp only
  omega

private theorem launch_stores {s : Shape} (layout : Layout s) (order : List Nat) (mem : Memory) :
    launch layout order mem = commitAll (order.flatMap fun pid => (block layout pid).stores mem) mem := by
  unfold launch
  rw [independent_block_reads _ mem mem (fun _ _ _ _ => rfl)]
  · simp only [List.flatMap_map]
  · intro writer hasWriter store produced reader hasReader
    obtain ⟨pid, _, rfl⟩ := List.mem_map.mp hasWriter
    obtain ⟨_, _, rfl⟩ := List.mem_map.mp hasReader
    obtain ⟨b, dest⟩ := store_inside layout mem pid store produced
    rintro ⟨a, src⟩
    exact different_ports layout (true, b) (false, a) (by simp) store.address dest src

private theorem owner_in_grid (s : Shape) (f : Field) (i : Nat) (inside : i < bytes s f) :
    i / width f / f.tile < s.blocks := by
  cases f <;> cases s <;>
    simp [bytes, width, Field.tile, Shape.blocks, Shape.tokens, Shape.ends, Shape.mtp, extent] at inside ⊢ <;> omega

private theorem commit_byte (stores : List Store) (mem : Memory) (address : Nat) (value : UInt8)
    (same : ∀ w ∈ stores, w.address = address → w.word = value)
    (existsWrite : ∃ w ∈ stores, w.address = address) :
    commitAll stores mem address = value := by
  induction stores generalizing mem with
  | nil => simp at existsWrite
  | cons item rest ih =>
    have tail : ∀ w ∈ rest, w.address = address → w.word = value :=
      fun w h => same w (List.mem_cons_of_mem item h)
    by_cases later : ∃ w ∈ rest, w.address = address
    · exact ih (commit item mem) tail later
    · obtain ⟨w, member, atAddr⟩ := existsWrite
      have first : w = item := (List.mem_cons.mp member).resolve_right (fun h => later ⟨w, h, atAddr⟩)
      subst w
      rw [commitAll, preserve_unwritten _ _ _ (fun w live atAddr => later ⟨w, live, atAddr⟩),
        commit, if_pos atAddr.symm]
      exact same item (List.mem_cons_self ..) atAddr

theorem launch_byte {s : Shape} (layout : Layout s) (order : List Nat)
    (complete : (List.range s.blocks).Perm order) (mem : Memory)
    (field : Field) (i : Nat) (inside : i < bytes s field) :
    launch layout order mem (layout.base (true, field) + i) = mem (layout.base (false, field) + i) := by
  rw [launch_stores]
  apply commit_byte
  · intro store present address
    obtain ⟨pid, _, made⟩ := List.mem_flatMap.mp present
    obtain ⟨port, j, _, fits, _, rfl⟩ := (store_produced layout mem pid store).mp made
    dsimp only [makeStore] at address ⊢
    have exactPort : port = field := by
      by_contra different
      apply different_ports layout (true, port) (true, field) (by simpa using different)
        (layout.base (true, field) + i)
      · unfold Layout.contains
        rw [← address]
        dsimp only
        omega
      · unfold Layout.contains
        dsimp only
        omega
    subst port
    exact congrArg (fun offset : Nat => mem (layout.base (false, field) + offset)) (Nat.add_left_cancel address)
  · let pid := i / width field / field.tile
    have live : pid < s.blocks := owner_in_grid s field i inside
    refine ⟨makeStore layout mem (field, i), List.mem_flatMap.mpr ⟨pid,
      complete.mem_iff.mp (List.mem_range.mpr live), ?_⟩, rfl⟩
    exact (store_produced layout mem pid _).mpr ⟨field, i, live, inside, rfl, rfl⟩

def writes {s : Shape} (layout : Layout s) (address : Nat) : Prop :=
  ∃ field : Field, layout.contains (true, field) address

theorem launch_frame {s : Shape} (layout : Layout s) (order : List Nat) (mem : Memory)
    (address : Nat) (outside : ¬writes layout address) :
    launch layout order mem address = mem address := by
  rw [launch_stores]
  apply preserve_unwritten
  intro w present eq
  obtain ⟨pid, _, made⟩ := List.mem_flatMap.mp present
  exact outside (eq ▸ store_inside layout mem pid w made)

def observe {s : Shape} (layout : Layout s) (dest : Bool) (mem : Memory) : StaticInputs s :=
  fun f word =>
    UInt64.ofNat (((List.range (width f)).map fun byte =>
      (mem (layout.base (dest, f) + (word.val * width f + byte))).toNat * 256 ^ byte).sum)

theorem launch_observes_copy {s : Shape} (layout : Layout s) (order : List Nat)
    (complete : (List.range s.blocks).Perm order) (mem : Memory) :
    observe layout true (launch layout order mem) = observe layout false mem := by
  funext f word
  unfold observe
  apply congrArg (UInt64.ofNat ∘ List.sum)
  apply List.map_congr_left
  intro byte live
  have byteBound := List.mem_range.mp live
  have wordBound := word.isLt
  have fits : word.val * width f + byte < bytes s f := by
    cases f <;> simp only [bytes, width] at byteBound ⊢ <;> omega
  rw [launch_byte layout order complete mem f _ fits]

def interpret {s : Shape} (program : GraphPlainPTX.Program s) (layout : Layout s)
    (order : List Nat) (mem : Memory) : Option Memory :=
  if GraphPlainPTX.recognizes s program.ptx then some (launch layout order mem) else none

theorem interpret_byte {s : Shape} (program : GraphPlainPTX.Program s) (layout : Layout s) (order : List Nat)
    (complete : (List.range s.blocks).Perm order) (mem : Memory) (f : Field)
    (i : Nat) (inside : i < bytes s f) :
    (interpret program layout order mem).map (· (layout.base (true, f) + i)) =
      some (mem (layout.base (false, f) + i)) := by
  simp only [interpret, program.recognized, if_true, Option.map_some, launch_byte layout order complete mem f i inside]

theorem interpret_frame {s : Shape} (program : GraphPlainPTX.Program s) (layout : Layout s) (order : List Nat)
    (mem : Memory) (address : Nat) (outside : ¬writes layout address) :
    (interpret program layout order mem).map (· address) = some (mem address) := by
  simp only [interpret, program.recognized, if_true, Option.map_some, launch_frame layout order mem address outside]

#axiomsClean checkedLayout
#axiomsClean vector_alignment
#axiomsClean block_byte_iff
#axiomsClean launch_byte
#axiomsClean launch_frame
#axiomsClean launch_observes_copy
#axiomsClean interpret_byte
#axiomsClean interpret_frame
end VeriTile.Bench.Models.NanoGPT.Track1.GraphPlainMemory
