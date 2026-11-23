import models.NanoGPT.Track1.GraphLaunchPTX

/-! Both graph launch variants use the same CTA and byte ownership.
In the member variant, the role of the row buffer is unchanged and the
slot value is computed by the inclusive search. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphMemberLaunch
open GraphStage GraphStageMemory CudaStores

def lookup {s n} (layout : Layout s n) (memory : Memory) (index : Nat) : UInt32 × List Nat :=
  NgramSlotPTX.memberLookup (readWord32 layout memory .want) (readWord32 layout memory .directory)
    (query layout memory index).toNat

def RowsPresent {s n} (layout : Layout s n) (memory : Memory) (ready : Ready layout memory) : Prop :=
  ∀ i : Fin (extent s .slots), NgramSlots.ContainsRow ready.want (query layout memory i.val).toNat

def emittedByte {s n} (member : Bool) (layout : Layout s n) (memory : Memory) (cell : Cell s) : UInt8 :=
  if member then
    let offset := cell.offset.val
    match cell.out with
    | .graph .slots | .pending => slotByte (lookup layout memory (offset / 4)).1.toNat offset
    | .graph _ => hashedByte layout memory cell
  else hashedByte layout memory cell

private theorem bytes_read {s n} (member : Bool) (layout : Layout s n) (cell : Cell s)
    (a b : Memory) (same : ∀ address, Reads layout address → a address = b address) :
    emittedByte member layout a cell = emittedByte member layout b cell := by
  cases member
  · exact hashed_byte_reads layout cell a b same
  · have value (i : Nat) : lookup layout a i = lookup layout b i := by
      have want := funext (fun j => word_read_supported layout .want trivial j a b same)
      have directory := funext (fun j => word_read_supported layout .directory trivial j a b same)
      unfold lookup query
      rw [want, directory, word_read_supported layout (.source .slots) trivial i a b same]
    rcases cell with ⟨out, offset⟩
    cases out with
    | graph f =>
      cases f <;> simp only [emittedByte, if_true, value] <;>
        exact hashed_byte_reads layout _ a b same
    | pending => simp only [emittedByte, if_true, value]

private theorem bytes_equal {s n} (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (fits : n < 2147483648)
    (batch : FineWeb.TokenBatch) (hashed : batch.hashedRows)
    (upload : UploadedHashes layout memory batch) (members : RowsPresent layout memory ready) (cell : Cell s) :
    emittedByte true layout memory cell = emittedByte false layout memory cell := by
  have equal (offset : Nat) (within : offset < extent s .slots * 4) :
      (lookup layout memory (offset / 4)).1.toNat = lookupHashed layout memory (offset / 4) := by
    let i : Fin (extent s .slots) := ⟨offset / 4, by omega⟩
    have valid := (hashed_query_in_range layout memory batch hashed upload i).2
    have present := NgramSlotPTX.member_lookup_exact ready.want _ ready.directory fits _ valid (members i)
    exact present.1.trans (NgramSlotPTX.lookup_exact ready.want _ ready.directory fits _ valid).symm
  rcases cell with ⟨out, offset⟩
  cases out with
  | graph f =>
    cases f
    case slots =>
      simp only [emittedByte, if_true, Bool.false_eq_true, if_false, hashedByte, equal _ offset.isLt]
    all_goals rfl
  | pending => simp only [emittedByte, if_true, Bool.false_eq_true, if_false, hashedByte, equal _ offset.isLt]

def cta {s n} (member : Bool) (layout : Layout s n) (pid : Nat) : Block where
  read := Reads layout
  stores memory := (GraphLaunchPTX.ctaCells s pid).map fun cell =>
    ⟨cell.address layout, emittedByte member layout memory cell⟩
  readsOnly a b same := List.map_congr_left fun cell _ =>
    congrArg (Store.mk (cell.address layout)) (bytes_read member layout cell a b same)

def launch {s n} (layout : Layout s n) (schedule : List Nat) (memory : Memory) : Memory :=
  execute (schedule.map (cta true layout)) memory

private theorem cta_write_frame {s n} (member : Bool) (layout : Layout s n) (schedule : List Nat)
    (memory : Memory) (writer : Block) (written : writer ∈ schedule.map (cta member layout))
    (store : Store) (atWriter : store ∈ writer.stores memory)
    (reader : Block) (reading : reader ∈ schedule.map (cta member layout)) : ¬ reader.read store.address := by
  obtain ⟨pid, _, rfl⟩ := List.mem_map.mp written
  obtain ⟨_, _, rfl⟩ := List.mem_map.mp reading
  obtain ⟨cell, _, rfl⟩ := List.mem_map.mp atWriter
  rintro ⟨buffer, readOnly, here⟩
  have different : Buffer.destination cell.out ≠ buffer := by
    intro same
    subst buffer
    exact readOnly
  exact separated_addresses layout different (cell_inside layout cell) here

theorem launch_equal {s n} (layout : Layout s n) (memory : Memory)
    (ready : Ready layout memory) (fits : n < 2147483648) (batch : FineWeb.TokenBatch)
    (hashed : batch.hashedRows) (upload : UploadedHashes layout memory batch)
    (members : RowsPresent layout memory ready) (schedule : List Nat) :
    launch layout schedule memory = GraphLaunchPTX.launch layout schedule memory := by
  change execute (schedule.map (cta true layout)) memory = execute (schedule.map (cta false layout)) memory
  rw [independent_block_reads _ memory memory (fun _ _ _ _ => rfl)
      (cta_write_frame true layout schedule memory),
    independent_block_reads _ memory memory (fun _ _ _ _ => rfl)
      (cta_write_frame false layout schedule memory), List.flatMap_map, List.flatMap_map]
  have ctas (pid : Nat) : (cta true layout pid).stores memory = (cta false layout pid).stores memory :=
    List.map_congr_left fun cell _ =>
      congrArg (Store.mk (cell.address layout)) (bytes_equal layout memory ready fits batch hashed upload members cell)
  simp only [ctas]

#axiomsClean launch_equal

end VeriTile.Bench.Models.NanoGPT.Track1.GraphMemberLaunch
