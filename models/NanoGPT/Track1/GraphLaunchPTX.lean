import models.NanoGPT.Track1.GraphCopyPTX

/-! Compose the copy threads and the lookup threads into CTA byte effects.
The two slot buffers are enumerated by buffer and byte, which groups the
same scalar writes made by the 128 lookup lanes. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.GraphLaunchPTX
open GraphStage GraphStageMemory CudaStores

private theorem slot_tile_count (s : Shape) (pending : Bool) :
    (slotOutput pending).bytes s = 512 * queryBlocks s := by cases s <;> cases pending <;> decide

def slotAt (s : Shape) (pid : Nat) (live : pid < s.blocks + queryBlocks s)
    (pending : Bool) (byte : Fin 512) : Cell s :=
  ⟨slotOutput pending, ⟨(pid - s.blocks) * 512 + byte.val, by
    have h := byte.isLt
    have count : 0 < queryBlocks s := by cases s <;> decide
    rw [slot_tile_count]
    omega⟩⟩

def slotCells (s : Shape) (pid : Nat) (live : pid < s.blocks + queryBlocks s) : List (Cell s) :=
  [false, true].flatMap fun pending => (List.finRange 512).map (slotAt s pid live pending)

/-- Each byte in the enumerated tile uses the lookup thread selected by
the compiled address arithmetic. -/
theorem slot_byte_address (s : Shape) (pid : Nat) (after : s.blocks ≤ pid)
    (live : pid < s.blocks + queryBlocks s) (pending : Bool) (byte : Fin 512) :
    (slotAt s pid live pending byte).offset.val =
      4 * (NgramSlotPTX.signed (NgramSlotPTX.queryIndex s.blocks pid (byte.val / 4))).toNat + byte.val % 4 := by
  have limit : s.blocks + queryBlocks s ≤ 1024 := by cases s <;> decide
  have bound := byte.isLt
  rw [NgramSlotPTX.query_index_exact s.blocks pid _ after (Nat.lt_of_lt_of_le live limit) (by omega),
    Int.toNat_natCast]
  simp only [slotAt]
  omega

private theorem slot_owner {s : Shape} (pid : Nat) (after : s.blocks ≤ pid)
    (live : pid < s.blocks + queryBlocks s) (pending : Bool) (byte : Fin 512) :
    owner (slotAt s pid live pending byte) = pid := by
  have h := byte.isLt
  cases pending <;> simp only [slotAt, slotOutput, owner, queryCTA] <;> omega

private theorem slot_cells_nodup (s : Shape) (pid : Nat) (live : pid < s.blocks + queryBlocks s) :
    (slotCells s pid live).Nodup := by
  unfold slotCells
  refine List.pairwise_flatMap.mpr ⟨?_, ?_⟩
  · intro pending _
    rw [List.pairwise_map]
    apply (List.nodup_finRange 512).imp
    intro a b different same
    have offsets := congrArg (fun c : Cell s => c.offset.val) same
    exact different (Fin.ext (Nat.add_left_cancel offsets))
  · apply (show [false, true].Nodup by decide).imp
    intro a b different x hx y hy same
    obtain ⟨i, _, rfl⟩ := List.mem_map.mp hx
    obtain ⟨j, _, rfl⟩ := List.mem_map.mp hy
    have ports := congrArg Cell.out same
    cases a <;> cases b <;> simp [slotAt, slotOutput] at ports different

private theorem slot_offset_member {s : Shape} (pid : Nat) (after : s.blocks ≤ pid)
    (live : pid < s.blocks + queryBlocks s) (pending : Bool) (offset : Fin ((slotOutput pending).bytes s))
    (owned : owner (Cell.mk (slotOutput pending) offset) = pid) :
    Cell.mk (slotOutput pending) offset ∈ slotCells s pid live := by
  have tile : s.blocks + offset.val / 4 / 128 = pid := by cases pending <;> exact owned
  let byte : Fin 512 := ⟨offset.val % 512, Nat.mod_lt _ (by decide)⟩
  have equal : slotAt s pid live pending byte = Cell.mk (slotOutput pending) offset := by
    have location : (pid - s.blocks) * 512 + byte.val = offset.val := by
      dsimp only [byte]
      omega
    exact congrArg (Cell.mk (slotOutput pending)) (Fin.ext location)
  refine List.mem_flatMap.mpr ⟨pending, ?_, List.mem_map.mpr ⟨byte, List.mem_finRange _, equal⟩⟩
  cases pending <;> decide

theorem slot_cells_permute {s n} (layout : Layout s n) (pid : Nat) (after : s.blocks ≤ pid)
    (live : pid < s.blocks + queryBlocks s) :
    (slotCells s pid live).Perm (assigned s pid) := by
  apply (List.perm_ext_iff_of_nodup (slot_cells_nodup s pid live) (assigned_nodup layout pid)).2
  intro cell
  constructor
  · intro found
    obtain ⟨pending, _, member⟩ := List.mem_flatMap.mp found
    obtain ⟨byte, _, rfl⟩ := List.mem_map.mp member
    exact List.mem_filter.mpr ⟨mem_cells _, by simp [slot_owner pid after live pending byte]⟩
  · intro found
    have ownerEq : owner cell = pid := of_decide_eq_true (List.mem_filter.mp found).2
    rcases cell with ⟨out, offset⟩
    cases out with
    | pending => exact slot_offset_member pid after live true offset ownerEq
    | graph f =>
      by_cases slots : f = .slots
      · subst f
        exact slot_offset_member pid after live false offset ownerEq
      · obtain ⟨_, valid, _⟩ := (assigned_copy_iff f slots offset pid).mp found
        exact False.elim (Nat.not_lt_of_ge after valid.1)

def ctaCells (s : Shape) (pid : Nat) : List (Cell s) :=
  if copy : pid < s.blocks then GraphCopyPTX.copyCells s pid copy
  else if live : pid < s.blocks + queryBlocks s then slotCells s pid live
  else []

theorem cta_cells_permute {s n} (layout : Layout s n) (pid : Nat) :
    (ctaCells s pid).Perm (assigned s pid) := by
  unfold ctaCells
  split
  · exact GraphCopyPTX.copy_cells_permute layout pid ‹pid < s.blocks›
  · split
    · exact slot_cells_permute layout pid (by omega) ‹pid < s.blocks + queryBlocks s›
    · have empty : assigned s pid = [] := by
        apply List.eq_nil_iff_forall_not_mem.mpr
        intro cell present
        have owns : owner cell = pid := of_decide_eq_true (List.mem_filter.mp present).2
        have live := owner_in_grid cell
        omega
      simp [empty]

def cta {s n} (layout : Layout s n) (pid : Nat) : Block where
  read := Reads layout
  stores memory := (ctaCells s pid).map (hashedStore layout memory)
  readsOnly mem next agree := List.map_congr_left fun cell _ =>
    congrArg (Store.mk (cell.address layout)) (hashed_byte_reads layout cell mem next agree)

def launch {s n} (layout : Layout s n) (schedule : List Nat) (memory : Memory) : Memory :=
  execute (schedule.map (cta layout)) memory

theorem launch_equal {s n} (layout : Layout s n) (schedule : List Nat) (memory : Memory) :
    launch layout schedule memory = launchHashed layout schedule memory := by
  induction schedule generalizing memory with
  | nil => rfl
  | cons pid rest ih =>
    have same := (cta_cells_permute layout pid).map (hashedStore layout memory)
    have unique : (((hashedBlock layout pid).stores memory).map Store.address).Nodup := by
      simpa only [hashedBlock, List.map_map, Function.comp_def, hashedStore] using assigned_addresses_nodup layout pid
    have commit := unique_stores_commute _ _ memory unique same.symm
    change launch layout rest (commitAll ((cta layout pid).stores memory) memory) =
      launchHashed layout rest (commitAll ((hashedBlock layout pid).stores memory) memory)
    rw [ih, commit]
    rfl

#axiomsClean slot_cells_permute
#axiomsClean slot_byte_address
#axiomsClean launch_equal
end VeriTile.Bench.Models.NanoGPT.Track1.GraphLaunchPTX
