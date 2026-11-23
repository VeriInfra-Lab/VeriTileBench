import models.NanoGPT.Track1.AnvilGram

namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilGram.TMA
open VeriTile.Triton FP
open AnvilEpilogue (Bank)
open TailNorm (F32 BF16 zero widen narrow)

def transposed : Bank → Operation → Bool
  | .mlp, .gram => true
  | _, _ => false

/-- Shape of the contiguous 2D tensor supplied by the host after
flattening the batch and row dimensions. -/
def shape (bank : Bank) (op : Operation) : Nat × Nat :=
  (batches bank * inputHeight bank op, inputWidth bank op)

/-- Coordinate of a 128 by 128 operand block, or its physical
128 by 128 transpose in the tall Gram. -/
def origin (bank : Bank) (op : Operation) (batch block chunk : Nat) : Nat × Nat :=
  let base := batch * inputHeight bank op
  if transposed bank op then (base + 128 * chunk, 128 * block)
  else (base + 128 * block, 128 * chunk)

def coordinates (bank : Bank) (op : Operation) (batch block chunk : Nat)
    (row : Fin 128) (k : Fin 128) : Nat × Nat :=
  let (y, x) := origin bank op batch block chunk
  if transposed bank op then (y + k.val, x + row.val)
  else (y + row.val, x + k.val)

def address (bank : Bank) (op : Operation) (batch block chunk : Nat)
    (row : Fin 128) (k : Fin 128) : Nat :=
  let (y, x) := coordinates bank op batch block chunk row k
  y * (shape bank op).2 + x

theorem address_exact (bank : Bank) (op : Operation) (t : Tile) (block chunk : Nat)
    (row : Fin 128) (k : Fin 128) :
    address bank op t.batch block chunk row k =
      operandOffset bank op t (128 * block + row.val) (128 * chunk + k.val) := by
  cases bank <;> cases op <;>
    simp [address, coordinates, origin, shape, operandOffset, transposed, Nat.add_mul, Nat.add_assoc]

theorem all_reads_in_bounds (bank : Bank) (op : Operation) (batch : Fin (batches bank))
    (block : Fin (Bank.side bank)) (chunk : Fin (reductionWidth bank op / 128))
    (row : Fin 128) (k : Fin 128) :
    let (y, x) := coordinates bank op batch.val block.val chunk.val row k
    y < (shape bank op).1 ∧ x < (shape bank op).2 := by
  have hb := batch.isLt
  have hblock := block.isLt
  have hc := chunk.isLt
  have hr := row.isLt
  have hk := k.isLt
  cases bank <;> cases op <;>
    simp only [coordinates, origin, shape, transposed, Bool.false_eq_true, if_false, if_true,
      batches, inputHeight, inputWidth, outputWidth, reductionWidth, Bank.side, Bank.dimensions,
      show (Bank.qk == Bank.mlp) = false from rfl, show (Bank.vo == Bank.mlp) = false from rfl,
      show (Bank.mlp == Bank.mlp) = true from rfl] at * <;> omega

theorem ordered_dot_step (chunk : Nat) (group : Fin 8) (k : Fin 16) :
    128 * chunk + (16 * group.val + k.val) = 16 * (8 * chunk + group.val) + k.val := by
  omega

/-- On the diagonal both descriptor views read the same staged BF16 box. -/
private def boxBytes : Nat := 128 * 128 * 2

def transferBytes (tile : Tile) : Nat :=
  if tile.row = tile.col then boxBytes else 2 * boxBytes

private def bytesAvoided (bank : Bank) (op : Operation) : Nat :=
  let chunks := reductionWidth bank op / 128
  (compactGrid bank).foldl (fun sum tile => sum + chunks * (2 * boxBytes - transferBytes tile)) 0

/-- Requested global-to-shared payload avoided in one default ANVIL update. -/
def updateBytesAvoided : Nat :=
  let banks := [Bank.qk, .vo, .mlp]
  let oneRound := banks.foldl (fun sum bank =>
    sum + bytesAvoided bank .gram + bytesAvoided bank (.polynomial zero zero)) 0
  AnvilGram.coefficients.length * oneRound

set_option maxRecDepth 5000 in
theorem updateBytesAvoided_exact : updateBytesAvoided = 146276352 := by decide

/-- The two mbarriers track alternate stages. A parity bit distinguishes
consecutive uses. -/

def slot (chunk : Nat) : Nat := chunk % 2

def parity (chunk : Nat) : Nat := chunk / 2 % 2

def advanceSlot (prev : Nat) : Nat :=
  if prev + 1 ≥ 2 then 0 else prev + 1

def advanceParity (prev phase : Nat) : Nat :=
  if prev + 1 ≥ 2 then phase ^^^ 1 else phase

theorem next_slot (chunk : Nat) : advanceSlot (slot chunk) = slot (chunk + 1) := by
  unfold advanceSlot slot
  split <;> omega

theorem next_parity (chunk : Nat) :
    advanceParity (slot chunk) (parity chunk) = parity (chunk + 1) := by
  have xorBit (p : Fin 2) : p.val ^^^ 1 = 1 - p.val := by decide +revert
  have hp : parity chunk < 2 := Nat.mod_lt _ (by decide)
  have hx : parity chunk ^^^ 1 = 1 - parity chunk := xorBit ⟨parity chunk, hp⟩
  unfold advanceParity
  rw [hx]
  unfold slot parity
  split <;> omega

/-- After issuing chunk `step`, waiting for at most one outstanding
WGMMA group retires every earlier chunk. The next TMA issue reuses
the slot occupied by `step - 1` once the ring is warm. -/
theorem reused_slot_retired (step prior : Nat)
    (older : prior ≤ step) (same : slot prior = slot (step + 1)) :
    prior < step := by
  unfold slot at same
  omega

/-- The prologue fills slot zero. Every iteration consumes the committed
slot and fills the other one if another chunk remains. -/
private def buffersBefore (chunks : Nat) : Nat → Nat → Option Nat
  | 0 => fun i => if i = 0 ∧ 0 < chunks then some 0 else none
  | step + 1 => fun i =>
      if i = slot (step + 1) ∧ step + 1 < chunks then some (step + 1)
      else buffersBefore chunks step i

private def readyChunk (chunks step : Nat) : Nat :=
  (buffersBefore chunks step (slot step)).getD 0

theorem ready_chunk_exact (chunks step : Nat) (active : step < chunks) :
    readyChunk chunks step = step := by
  cases step
  · simp [readyChunk, buffersBefore, slot, active]
  · simp [readyChunk, buffersBefore, active]

private theorem reduction_is_wide (bank : Bank) (op : Operation) : reductionWidth bank op % 128 = 0 := by
  cases op with
  | gram => cases bank <;> decide
  | polynomial => simp [reductionWidth, outputWidth, Nat.mul_mod]

private theorem chunk_for_group (bank : Bank) (op : Operation) (step : Nat)
    (active : step < reductionWidth bank op / 16) :
    step / 8 < reductionWidth bank op / 128 := by
  have hf := reduction_is_wide bank op
  omega

private def kWord (step : Nat) (k : Fin 16) : Fin 128 :=
  ⟨16 * (step % 8) + k.val, by have := k.isLt; omega⟩

theorem descriptor_dot_address (bank : Bank) (op : Operation) (tile : Tile)
    (block step : Nat) (row : Fin 128) (k : Fin 16) :
    address bank op tile.batch block (step / 8) row (kWord step k) =
      operandOffset bank op tile (block * 128 + row.val) (16 * step + k.val) := by
  rw [address_exact]
  have hrow : 128 * block + row.val = block * 128 + row.val := by omega
  have hk : 128 * (step / 8) + (kWord step k).val = 16 * step + k.val := by
    simp only [kWord]
    omega
  rw [hrow, hk]

private abbrev OperandBox := Fin 128 → Fin 128 → BF16

private def leftBox (chunk : Nat) : Nat := slot chunk

private def rightBox (tile : Tile) (chunk : Nat) : Nat :=
  if tile.row = tile.col then leftBox chunk else 2

/-- The next left load can start while the current WGMMA still reads either operand. -/
theorem next_left_prefetch_disjoint (tile : Tile) (chunk : Nat) :
    leftBox (chunk + 1) ≠ leftBox chunk ∧ leftBox (chunk + 1) ≠ rightBox tile chunk := by
  unfold rightBox leftBox slot
  split <;> omega

private def waitGroups (keep : Nat) (issued : List Nat) : List Nat :=
  issued.drop (issued.length - keep)

private def waitLimit (tile : Tile) : Nat :=
  if tile.row = tile.col then 1 else 0

private def inFlight (tile : Tile) : Nat → List Nat
  | 0 => []
  | chunk + 1 => waitGroups (waitLimit tile) (inFlight tile chunk ++ [chunk])

theorem groups_after_chunk_wait (tile : Tile) (chunk : Nat) :
    inFlight tile (chunk + 1) = if tile.row = tile.col then [chunk] else [] := by
  by_cases diagonal : tile.row = tile.col
  · induction chunk with
    | zero => simp [inFlight, waitGroups, waitLimit, diagonal]
    | succ chunk ih =>
      rw [inFlight, ih]
      simp [waitGroups, waitLimit, diagonal]
  · simp [inFlight, waitGroups, waitLimit, diagonal]

/-- The diagonal path waits for at most one group before prefetching.
The off-diagonal path has already drained the preceding group. -/
private def pendingAtLeftRefill (tile : Tile) (chunk : Nat) : List Nat :=
  if tile.row = tile.col then inFlight tile (chunk + 1) else inFlight tile chunk ++ [chunk]

theorem left_refill_preserves_pending_operands (tile : Tile) (chunk group : Nat)
    (live : group ∈ pendingAtLeftRefill tile chunk) :
    leftBox (chunk + 1) ≠ leftBox group ∧ leftBox (chunk + 1) ≠ rightBox tile group := by
  have same : pendingAtLeftRefill tile chunk = [chunk] := by
    by_cases diagonal : tile.row = tile.col
    · simp [pendingAtLeftRefill, diagonal, groups_after_chunk_wait]
    · cases chunk with
      | zero => simp [pendingAtLeftRefill, diagonal, inFlight]
      | succ chunk => simp [pendingAtLeftRefill, diagonal, groups_after_chunk_wait]
  rw [same, List.mem_singleton] at live
  subst group
  exact next_left_prefetch_disjoint tile chunk

/-- The off-diagonal transfer into the fixed right box starts after wait zero. -/
theorem right_refill_has_no_pending_operand (tile : Tile) (chunk : Nat)
    (offDiagonal : tile.row ≠ tile.col) : inFlight tile (chunk + 1) = [] := by
  rw [groups_after_chunk_wait, if_neg offDiagonal]

theorem rightBox_alias (tile : Tile) (chunk : Nat) :
    rightBox tile chunk = leftBox chunk ↔ tile.row = tile.col := by
  unfold rightBox leftBox slot
  split <;> omega

theorem operand_boxes_in_bounds (tile : Tile) (chunk : Nat) :
    leftBox chunk < 3 ∧ rightBox tile chunk < 3 := by
  unfold rightBox leftBox slot
  split <;> omega

private def loadBox (bank : Bank) (op : Operation) (input : Input) (tile : Tile)
    (block chunk : Nat) : OperandBox :=
  fun i k => input.values (address bank op tile.batch block chunk i k)

private def operandsAt (bank : Bank) (op : Operation) (input : Input) (tile : Tile)
    (chunk : Nat) : OperandBox × OperandBox :=
  let left := loadBox bank op input tile tile.row chunk
  (left, if rightBox tile chunk = leftBox chunk then left
    else loadBox bank op input tile tile.col chunk)

theorem operands_exact (bank : Bank) (op : Operation) (input : Input) (tile : Tile)
    (chunk : Nat) :
    operandsAt bank op input tile chunk =
      (loadBox bank op input tile tile.row chunk, loadBox bank op input tile tile.col chunk) := by
  by_cases diagonal : tile.row = tile.col <;> simp [operandsAt, rightBox_alias, diagonal]

private def groupsAtChunks (chunks : List Nat) : List Nat :=
  chunks.flatMap fun chunk => List.range' (8 * chunk) 8

private theorem groupsAtChunks_eq_range (chunks : Nat) :
    groupsAtChunks (List.range chunks) = List.range (8 * chunks) := by
  induction chunks with
  | zero => rfl
  | succ chunks ih =>
      rw [groupsAtChunks, List.range_succ, List.flatMap_append]
      change groupsAtChunks (List.range chunks) ++ List.range' (8 * chunks) 8 = _
      rw [ih, show 8 * (chunks + 1) = 8 * chunks + 8 by omega]
      simp only [List.range_eq_range']
      simpa only [Nat.zero_add] using (List.range'_append_1 (s := 0) (m := 8 * chunks) (n := 8))

/-- One loaded box feeds eight consecutive native WGMMA groups. -/
private def wideGroups (bank : Bank) (op : Operation) : List Nat :=
  groupsAtChunks (List.range (reductionWidth bank op / 128))

theorem wide_groups_exact (bank : Bank) (op : Operation) :
    wideGroups bank op = List.range (reductionWidth bank op / 16) := by
  have aligned := reduction_is_wide bank op
  rw [wideGroups, groupsAtChunks_eq_range]
  congr 1
  omega

def element (bank : Bank) (op : Operation) (input : Input) (tile : Tile)
    (i j : Fin 128) : BF16 :=
  let operands := (wideGroups bank op).map fun step (k : Fin 16) =>
    let chunk := readyChunk (reductionWidth bank op / 128) (step / 8)
    let (left, right) := operandsAt bank op input tile chunk
    (left i (kWord step k), right j (kWord step k))
  let accumulated := operands.foldl input.wgmma16 zero
  match op with
  | .gram => narrow accumulated
  | .polynomial alpha beta =>
      let row := tile.row * 128 + i.val
      let col := tile.col * 128 + j.val
      let bias := mul Config.ieeeValue beta (widen (loadBias false bank input tile row col))
      narrow (fma Config.ieeeValue alpha accumulated bias)

theorem element_exact (bank : Bank) (op : Operation) (input : Input)
    (tile : Tile) (i j : Fin 128) :
    element bank op input tile i j = AnvilGram.element false bank op input tile i j := by
  have same : (List.range (reductionWidth bank op / 16)).map
      (fun step (k : Fin 16) =>
        let chunk := readyChunk (reductionWidth bank op / 128) (step / 8)
        let (left, right) := operandsAt bank op input tile chunk
        (left i (kWord step k), right j (kWord step k))) =
    (List.range (reductionWidth bank op / 16)).map
      (fun step (k : Fin 16) =>
        (loadOperand false bank op input tile (tile.row * 128 + i.val) (16 * step + k.val),
         loadOperand false bank op input tile (tile.col * 128 + j.val) (16 * step + k.val))) := by
    apply List.map_congr_left
    intro step hs
    funext k
    have active := chunk_for_group bank op step (List.mem_range.mp hs)
    simp only [ready_chunk_exact _ _ active, operands_exact, loadBox, descriptor_dot_address, loadOperand,
      Bool.false_eq_true, if_false]
  simp only [element, wide_groups_exact, AnvilGram.element, same]
  cases op <;> rfl

def outputAddress (bank : Bank) (tile : Tile) (row col : Nat) : Nat :=
  (tile.batch * outputWidth bank + row) * outputWidth bank + col

theorem output_address_exact (bank : Bank) (tile : Tile) (row col : Nat) :
    outputAddress bank tile row col = outputOffset bank tile row col := by
  simp [outputAddress, outputOffset, Nat.add_mul, Nat.add_assoc]

theorem output_box_in_bounds (bank : Bank) (tile : Tile) (scheduled : tile ∈ compactGrid bank)
    (i j : Fin 128) :
    let width := outputWidth bank
    let row := tile.row * 128 + i.val
    let col := tile.col * 128 + j.val
    (tile.batch * width + row < batches bank * width ∧ col < width) ∧
      (tile.batch * width + col < batches bank * width ∧ row < width) := by
  obtain ⟨hb, hr, hc⟩ := scheduled_tile_bounds bank tile scheduled
  have hrow := lane_in_bounds bank hr i
  have hcol := lane_in_bounds bank hc j
  have batch_row_bound (n : Nat) (within : n < outputWidth bank) :
      tile.batch * outputWidth bank + n < batches bank * outputWidth bank := by
    calc
      _ < tile.batch * outputWidth bank + outputWidth bank := Nat.add_lt_add_left within _
      _ = (tile.batch + 1) * outputWidth bank := by simp [Nat.add_mul]
      _ ≤ _ := Nat.mul_le_mul_right _ hb
  exact ⟨⟨batch_row_bound _ hrow, hcol⟩, ⟨batch_row_bound _ hcol, hrow⟩⟩

/-- Different output rows stay disjoint when the two TMA stores run together. -/
theorem offdiag_outputs_disjoint (bank : Bank) (tile : Tile)
    (scheduled : tile ∈ compactGrid bank) (offdiag : tile.row ≠ tile.col)
    (i j u v : Fin 128) :
    outputAddress bank tile (128 * tile.row + i.val) (128 * tile.col + j.val) ≠
      outputAddress bank tile (128 * tile.col + u.val) (128 * tile.row + v.val) := by
  have jBound := (output_box_in_bounds bank tile scheduled i j).1.2
  have vBound := (output_box_in_bounds bank tile scheduled v u).2.2
  have iLane := i.isLt
  have uLane := u.isLt
  cases bank <;>
    simp only [outputAddress, outputWidth, Bank.side] at jBound vBound ⊢ <;> omega

/-- TMA stores can still own shared-memory reads or only global writes.
The intermediate read wait makes the source reusable. A full wait drains
committed stores and publishes their destination words. -/
structure OutputStoreQueue (Word : Type) where
  reading : List Word
  writing : List Word
  visible : List Word

def OutputStoreQueue.waitReads {Word : Type} (queue : OutputStoreQueue Word) :
    OutputStoreQueue Word :=
  ⟨[], queue.writing ++ queue.reading, queue.visible⟩

def OutputStoreQueue.waitFull {Word : Type} (queue : OutputStoreQueue Word) :
    OutputStoreQueue Word :=
  ⟨[], [], queue.visible ++ (queue.writing ++ queue.reading)⟩

theorem one_full_wait_eq_two_waits {Word : Type} (queue : OutputStoreQueue Word) :
    queue.waitReads.waitFull = queue.waitFull := by
  simp [OutputStoreQueue.waitReads, OutputStoreQueue.waitFull]

theorem full_wait_leaves_no_pending_stores {Word : Type} (queue : OutputStoreQueue Word) :
    queue.waitFull.reading = [] ∧ queue.waitFull.writing = [] :=
  ⟨rfl, rfl⟩

private def tileStores (bank : Bank) (op : Operation) (input : Input) (tile : Tile) : TileStores :=
  let row := fun i : Fin 128 => tile.row * 128 + i.val
  let col := fun j : Fin 128 => tile.col * 128 + j.val
  let value := element bank op input tile
  let direct := fun i j => some (outputAddress bank tile (row i) (col j), value i j)
  let mirror := fun i j => some (outputAddress bank tile (col j) (row i), value i j)
  ⟨tile, (if tile.row = tile.col then fun _ _ => none else direct), mirror⟩

theorem tile_stores_exact (bank : Bank) (op : Operation) (input : Input) (tile : Tile) :
    tileStores bank op input tile = issueStores true false bank op input tile := by
  by_cases diagonal : tile.row = tile.col <;>
    simp [tileStores, issueStores, TileStores.finalWrites, stores, diagonal, output_address_exact, element_exact]

/-- The kernel issues exactly the direct and mirrored words established
by `tile_stores_exact`. Both committed writes are visible at completion. -/
theorem all_output_words_visible (bank : Bank) (op : Operation) (input : Input) (tile : Tile)
    (i j : Fin 128) :
    let writes := tileStores bank op input tile
    let committed := [writes.direct i j, writes.transpose i j].filterMap id
    let queue : OutputStoreQueue (Nat × BF16) := ⟨committed, [], []⟩
    queue.waitFull.visible =
      [(issueStores true false bank op input tile).direct i j,
        (issueStores true false bank op input tile).transpose i j].filterMap id := by
  simp [OutputStoreQueue.waitFull, tile_stores_exact]

/-- Project the store queue onto one destination. The two off-diagonal
orientations have disjoint addresses, and the diagonal only issues its
mirrored orientation. Each destination therefore has at most one word. -/
private def completedWord (word : Option (Nat × BF16)) : Option (Nat × BF16) :=
  let queue : OutputStoreQueue (Nat × BF16) := ⟨word.toList, [], []⟩
  queue.waitFull.visible.head?

private theorem completedWord_eq (word : Option (Nat × BF16)) : completedWord word = word := by
  cases word <;> rfl

private def completedTileStores (bank : Bank) (op : Operation) (input : Input)
    (tile : Tile) : TileStores :=
  let stores := tileStores bank op input tile
  { stores with direct := fun i j => completedWord (stores.direct i j),
                transpose := fun i j => completedWord (stores.transpose i j) }

private theorem completedTileStores_eq (bank : Bank) (op : Operation) (input : Input)
    (tile : Tile) : completedTileStores bank op input tile = tileStores bank op input tile := by
  unfold completedTileStores
  simp only [completedWord_eq]

private def storeAt (bank : Bank) (op : Operation) (input : Input) (p : Cell bank) :
    Option (Nat × BF16) :=
  let i : Fin 128 := ⟨p.row.val % 128, by omega⟩
  let j : Fin 128 := ⟨p.col.val % 128, by omega⟩
  let writes := completedTileStores bank op input p.tile
  if p.row.val / 128 ≤ p.col.val / 128 then writes.transpose j i else writes.direct i j

def run : Kernel := fun bank op input =>
  let grid := compactGrid bank
  let words := fun p : Cell bank =>
    let previous := input.freshOutput (outputAddress bank p.tile p.row.val p.col.val)
    if p.tile ∈ grid then ((storeAt bank op input p).map Prod.snd).getD previous else previous
  ⟨grid.map (completedTileStores bank op input), words⟩

theorem run_eq : run = compact := by
  funext bank op input
  simp only [run, compact, Output.mk.injEq]
  constructor
  · apply List.map_congr_left
    intro tile _
    rw [completedTileStores_eq, tile_stores_exact]
  · funext p
    simp only [AnvilGram.overwrite, storeAt, AnvilGram.storeAt, completedTileStores_eq,
      tile_stores_exact, output_address_exact]

#axiomsClean one_full_wait_eq_two_waits
#axiomsClean full_wait_leaves_no_pending_stores
#axiomsClean all_output_words_visible
#axiomsClean next_slot
#axiomsClean updateBytesAvoided_exact
#axiomsClean next_parity
#axiomsClean reused_slot_retired
#axiomsClean ready_chunk_exact
#axiomsClean descriptor_dot_address
#axiomsClean next_left_prefetch_disjoint
#axiomsClean groups_after_chunk_wait
#axiomsClean left_refill_preserves_pending_operands
#axiomsClean right_refill_has_no_pending_operand
#axiomsClean rightBox_alias
#axiomsClean operand_boxes_in_bounds
#axiomsClean operands_exact
#axiomsClean wide_groups_exact
#axiomsClean element_exact
#axiomsClean output_box_in_bounds
#axiomsClean offdiag_outputs_disjoint
#axiomsClean tile_stores_exact
#axiomsClean run_eq

#axiomsClean address_exact
#axiomsClean all_reads_in_bounds
#axiomsClean ordered_dot_step

end VeriTile.Bench.Models.NanoGPT.Track1.AnvilGram.TMA
