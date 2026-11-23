import models.NanoGPT.Track1.ValidationScanBinding

/-! Connect the vocabulary-loop states to the compiled reduction and
loss store for each slab and row of the validation launch. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValidationRowBinding
open ValidationCE (Tile)
open ValidationReduceBinding (Slot inputSlot registerGroups rowCount)
open ValidationScanBinding (rowMask rowProjection)

private def position (tile : Tile) (slab column : Nat) : Nat :=
  tile.groups * slab + (registerGroups tile).idxOf (column / tile.lanes)

private def lane (tile : Tile) (row column : Nat) : Nat :=
  rowCount tile * (column % tile.lanes) + row

private theorem group_index (tile : Tile) : ∀ g : Fin tile.groups,
    (registerGroups tile).idxOf g.val < tile.groups ∧
      (registerGroups tile)[(registerGroups tile).idxOf g.val]? = some g.val := by
  cases tile <;> decide

private theorem packed_bound {stride outer inner limit : Nat}
    (i : inner < stride) (o : outer < limit) : stride * outer + inner < stride * limit := by
  calc
    stride * outer + inner < stride * outer + stride := Nat.add_lt_add_left i _
    _ = stride * (outer + 1) := by rw [Nat.mul_succ]
    _ ≤ stride * limit := Nat.mul_le_mul_left _ (Nat.succ_le_of_lt o)

private theorem routed_slot (tile : Tile) (slab : Fin ValidationCE.slabCount)
    (row : Fin (rowCount tile)) (column : Fin tile.extent) :
    position tile slab.val column.val < ValidationCE.slabCount * tile.groups ∧
      lane tile row.val column.val < rowCount tile * tile.lanes ∧
      inputSlot tile (position tile slab.val column.val) (lane tile row.val column.val) =
        some ⟨slab.val, row.val, column.val⟩ := by
  have groupBound : column.val / tile.lanes < tile.groups := by
    have h := column.isLt
    cases tile <;> simp [Tile.extent, Tile.groups, Tile.lanes] at * <;> omega
  obtain ⟨indexBound, group⟩ := group_index tile ⟨column.val / tile.lanes, groupBound⟩
  have localBound : column.val % tile.lanes < tile.lanes :=
    Nat.mod_lt _ (by cases tile <;> decide)
  refine ⟨?_, packed_bound row.isLt localBound, ?_⟩
  · rw [Nat.mul_comm]
    exact packed_bound indexBound slab.isLt
  unfold inputSlot position lane
  rw [Nat.mul_add_mod, Nat.mod_eq_of_lt indexBound, group]
  change some (⟨_, _, _⟩ : Slot) = _
  rw [Nat.mul_add_div (by cases tile <;> decide : 0 < tile.groups),
    Nat.div_eq_of_lt indexBound, Nat.add_zero,
    Nat.mul_add_mod, Nat.mod_eq_of_lt row.isLt,
    Nat.mul_add_div (by cases tile <;> decide : 0 < rowCount tile), Nat.div_eq_of_lt row.isLt,
    Nat.add_zero, Nat.mul_comm (column.val / tile.lanes) tile.lanes, Nat.mod_add_div]

private theorem mapM_some {f : α → Option β} {g : α → β} {xs : List α}
    (each : ∀ x ∈ xs, f x = some (g x)) : xs.mapM f = some (xs.map g) := by
  induction xs with
  | nil => rfl
  | cons head tail ih =>
    simp [List.mapM_cons, each head (List.mem_cons_self),
      ih (fun x member => each x (List.mem_cons_of_mem _ member))]

def runRow {ptx : String} {tile : Tile} (compiled : ValidationReduceBinding.Certificate ptx tile)
    (scan : ValidationScanBinding.Certificate ptx tile) (native : ValidationCE.Native)
    (block : Nat) (slab : Fin ValidationCE.slabCount) (row : Fin (rowCount tile))
    (tokens : Nat → Fin 50304) (mask : Nat → UInt8) (bf16 : Nat → Int → TailNorm.BF16)
    (target : Option (Fin 50304)) : Option TailNorm.F32 := do
  let partials ← (List.range tile.extent).mapM fun col =>
    ValidationScanBinding.runSlot scan native block (position tile slab.val col)
      (lane tile row.val col) tokens mask bf16
  let p := partials.toArray
  let reduced ← ValidationReduceBinding.reduced compiled native
    (fun slot => p[slot.column]?.getD ValidationCE.empty) slab row
  let slot : Slot := ⟨slab.val, row.val, 0⟩
  ValidationLogBinding.finishRow ptx slab native target (rowMask tile slot block tokens mask)
    (fun column => rowProjection tile slot block bf16 column.val) reduced.maximum reduced.sum

/-- The checked initial state, online loop, cross-warp reduction and
loss writeback together compute the validation row for arbitrary BF16
projections, packed mask words, and in-vocabulary input tokens. -/
theorem decoded_row_exact {ptx : String} {tile : Tile}
    (compiled : ValidationReduceBinding.Certificate ptx tile)
    (scan : ValidationScanBinding.Certificate ptx tile) (native : ValidationCE.Native)
    (block : Nat) (slab : Fin ValidationCE.slabCount) (row : Fin (rowCount tile))
    (tokens : Nat → Fin 50304) (mask : Nat → UInt8) (bf16 : Nat → Int → TailNorm.BF16)
    (target : Option (Fin 50304)) :
    let slot : Slot := ⟨slab.val, row.val, 0⟩
    runRow compiled scan native block slab row tokens mask bf16 target =
      some (ValidationCE.rowLoss native tile (fun column => rowProjection tile slot block bf16 column.val)
        (rowMask tile slot block tokens mask) target) := by
  let slot : Slot := ⟨slab.val, row.val, 0⟩
  let read : Fin 50304 → TailNorm.F32 := fun col =>
    ValidationCE.logit native (rowMask tile slot block tokens mask) col.val
      (rowProjection tile slot block bf16 col.val)
  have partials :
      (List.range tile.extent).mapM (fun col => ValidationScanBinding.runSlot scan native block
        (position tile slab.val col) (lane tile row.val col) tokens mask bf16) =
      some ((List.range tile.extent).map (ValidationCE.scan native tile read)) := by
    apply mapM_some
    intro col member
    let index : Fin tile.extent := ⟨col, List.mem_range.mp member⟩
    obtain ⟨pos, tid, route⟩ := routed_slot tile slab row index
    exact ValidationScanBinding.decoded_scan_exact scan native block _ _ tokens mask bf16 pos tid route
  unfold runRow
  rw [partials]
  change ((ValidationReduceBinding.reduced compiled native
    (fun slot => ((List.range tile.extent).map (ValidationCE.scan native tile read)).toArray[slot.column]?.getD
      ValidationCE.empty) slab row).bind fun values =>
    ValidationLogBinding.finishRow ptx slab native target (rowMask tile slot block tokens mask)
      (fun col => rowProjection tile slot block bf16 col.val) values.maximum values.sum) = _
  rw [← List.map_toArray]
  exact ValidationReduceBinding.decoded_row_loss_exact compiled native slab row
    (fun col => rowProjection tile slot block bf16 col.val) (rowMask tile slot block tokens mask) target

#axiomsClean decoded_row_exact
end VeriTile.Bench.Models.NanoGPT.Track1.ValidationRowBinding
