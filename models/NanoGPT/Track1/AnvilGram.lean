import models.NanoGPT.Track1.AnvilEpilogue
import models.NanoGPT.Track1.AnvilRails
import models.NanoGPT.Track1.AnvilNormalization
import models.NanoGPT.Track1.SourceAnchors

/-! Compact scheduling for the ordered BF16 ANVIL Gram and polynomial tiles.
Unchanged WGMMA instructions consume the same sixteen inputs and accumulator
at each substep. Each diagonal tile only issues its final transposed store. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.AnvilGram
open VeriTile.Triton FP
open AnvilEpilogue (Bank)
open TailNorm (F32 BF16 zero widen narrow)

structure Tile where
  batch : Nat
  row : Nat
  col : Nat
  deriving DecidableEq, Repr

def Bank.side : Bank → Nat
  | .qk => 2
  | .vo | .mlp => 6

def batches (bank : Bank) : Nat := bank.dimensions.1

def denseTile (side pid : Nat) : Option Tile :=
  let localId := pid % (side * side)
  let groupWidth := 8 * side
  let firstRow := localId / groupWidth * 8
  let groupHeight := min (side - firstRow) 8
  let inner := localId % groupWidth
  let m := firstRow + inner % groupHeight
  let n := inner / groupHeight
  if m < n then none else some ⟨pid / (side * side), m, n⟩

def compactTile (side pid : Nat) : Tile :=
  let count := side * (side + 1) / 2
  let rows : UInt64 := if side == 2 then 0x48 else 0x5b2c76356346c688
  let cols : UInt64 := if side == 2 then 0x40 else 0x591b692449240000
  let shift := UInt64.ofNat (3 * (pid % count))
  ⟨pid / count, ((rows >>> shift) &&& 7).toNat, ((cols >>> shift) &&& 7).toNat⟩

def denseCount (bank : Bank) : Nat := batches bank * Bank.side bank * Bank.side bank
def compactCount (bank : Bank) : Nat := batches bank * (Bank.side bank * (Bank.side bank + 1) / 2)

def squareGrid (bank : Bank) : List Tile :=
  (List.range (denseCount bank)).filterMap (denseTile (Bank.side bank))

def compactGrid (bank : Bank) : List Tile :=
  (List.range (compactCount bank)).map (compactTile (Bank.side bank))

set_option maxRecDepth 4096 in
theorem ordered_grid_equal (bank : Bank) : compactGrid bank = squareGrid bank := by
  cases bank <;> decide

set_option maxRecDepth 4096 in
theorem compact_has_unique_tiles (bank : Bank) : (compactGrid bank).Nodup := by
  cases bank <;> decide

theorem removed_blocks_per_step :
    12 * ([Bank.qk, .vo, .mlp].map fun b => denseCount b - compactCount b).sum = 972 := by
  decide

inductive Operation where
  | gram
  | polynomial (alpha beta : F32)

def outputWidth (bank : Bank) : Nat := 128 * Bank.side bank

def inputWidth (bank : Bank) : Operation → Nat
  | .gram => bank.dimensions.2.2
  | .polynomial .. => outputWidth bank

def inputHeight (bank : Bank) : Operation → Nat
  | .gram => bank.dimensions.2.1
  | .polynomial .. => outputWidth bank

def reductionWidth (bank : Bank) : Operation → Nat
  | .gram => if bank == .mlp then bank.dimensions.2.1 else bank.dimensions.2.2
  | .polynomial .. => outputWidth bank

set_option maxRecDepth 4096 in
theorem scheduled_tile_bounds (bank : Bank) (t : Tile) (h : t ∈ compactGrid bank) :
    t.batch < batches bank ∧ t.row < Bank.side bank ∧ t.col < Bank.side bank := by
  have hAll : (compactGrid bank).all (fun t =>
      decide (t.batch < batches bank ∧ t.row < Bank.side bank ∧ t.col < Bank.side bank)) = true := by
    cases bank <;> decide
  exact of_decide_eq_true (List.all_eq_true.mp hAll t h)

theorem reduction_is_full (bank : Bank) (op : Operation) : reductionWidth bank op % 64 = 0 := by
  cases op with
  | gram => cases bank <;> decide
  | polynomial => simp [reductionWidth, outputWidth, Nat.mul_mod]

theorem lane_in_bounds (bank : Bank) {tile : Nat} (h : tile < Bank.side bank) (lane : Fin 128) :
    tile * 128 + lane.val < outputWidth bank := by
  have hl := lane.isLt
  simp only [outputWidth]
  omega

def operandOffset (bank : Bank) (op : Operation) (tile : Tile) (row k : Nat) : Nat :=
  let pitch := inputWidth bank op
  let batch := tile.batch * inputHeight bank op * pitch
  match bank, op with
  | .mlp, .gram => batch + k * pitch + row
  | _, _ => batch + row * pitch + k

def outputOffset (bank : Bank) (tile : Tile) (row col : Nat) : Nat :=
  let width := outputWidth bank
  tile.batch * width * width + row * width + col

inductive Launch where
  | dense | triangular | fullTiles

structure Input where
  values : Nat → BF16
  /-- Same native value operation on each pair of sixteen BF16 terms,
  using the matching WGMMA accumulator and unchanged native rounding. -/
  wgmma16 : F32 → (Fin 16 → BF16 × BF16) → F32
  /-- Initial contents of the reused output, and of a fresh allocation.
  The two memories can have completely different words. -/
  retainedOutput : Nat → BF16
  freshOutput : Nat → BF16
  launchMode : Launch := .fullTiles

def loadOperand (masked : Bool) (bank : Bank) (op : Operation) (input : Input)
    (tile : Tile) (row k : Nat) : BF16 :=
  if masked then
    if k < reductionWidth bank op then
      input.values (operandOffset bank op tile (row % outputWidth bank) k)
    else narrow zero
  else input.values (operandOffset bank op tile row k)

def loadBias (masked : Bool) (bank : Bank) (input : Input) (tile : Tile) (row col : Nat) : BF16 :=
  if masked && decide (¬ (row < outputWidth bank ∧ col < outputWidth bank)) then narrow zero
  else input.values (outputOffset bank tile row col)

def element (masked : Bool) (bank : Bank) (op : Operation) (input : Input) (tile : Tile)
    (laneRow laneCol : Fin 128) : BF16 :=
  let row := tile.row * 128 + laneRow.val
  let col := tile.col * 128 + laneCol.val
  let operands := (List.range (reductionWidth bank op / 16)).map fun step (k : Fin 16) =>
      let kk := 16 * step + k.val
      (loadOperand masked bank op input tile row kk, loadOperand masked bank op input tile col kk)
  let accumulated := operands.foldl input.wgmma16 zero
  match op with
  | .gram => narrow accumulated
  | .polynomial alpha beta =>
      let bias := FP.mul Config.ieeeValue beta (widen (loadBias masked bank input tile row col))
      narrow (FP.fma Config.ieeeValue alpha accumulated bias)

theorem full_tile_element (bank : Bank) (op : Operation) (input : Input) (tile : Tile)
    (h : tile ∈ compactGrid bank) (i j : Fin 128) :
    element false bank op input tile i j = element true bank op input tile i j := by
  obtain ⟨_, hr, hc⟩ := scheduled_tile_bounds bank tile h
  have hi := lane_in_bounds bank hr i
  have hj := lane_in_bounds bank hc j
  have loads : (List.range (reductionWidth bank op / 16)).map
      (fun step (k : Fin 16) =>
        (loadOperand false bank op input tile (tile.row * 128 + i.val) (16 * step + k.val),
         loadOperand false bank op input tile (tile.col * 128 + j.val) (16 * step + k.val))) =
    (List.range (reductionWidth bank op / 16)).map
      (fun step (k : Fin 16) =>
        (loadOperand true bank op input tile (tile.row * 128 + i.val) (16 * step + k.val),
         loadOperand true bank op input tile (tile.col * 128 + j.val) (16 * step + k.val))) := by
    apply List.map_congr_left
    intro step hstep
    funext k
    have hs := List.mem_range.mp hstep
    have hk := k.isLt
    have hrc := reduction_is_full bank op
    have bound : 16 * step + k.val < reductionWidth bank op := by omega
    simp [loadOperand, bound, Nat.mod_eq_of_lt hi, Nat.mod_eq_of_lt hj]
  simp only [element]
  rw [loads]
  cases op <;> simp [loadBias, hi, hj]

structure TileStores where
  target : Tile
  direct : Fin 128 → Fin 128 → Option (Nat × BF16)
  transpose : Fin 128 → Fin 128 → Option (Nat × BF16)

def stores (masked : Bool) (bank : Bank) (op : Operation) (input : Input) (tile : Tile) : TileStores :=
  let write := fun row col i j =>
    if masked && decide (¬ (row < outputWidth bank ∧ col < outputWidth bank)) then none
    else some (outputOffset bank tile row col, element masked bank op input tile i j)
  let direct := fun i j => write (tile.row * 128 + i.val) (tile.col * 128 + j.val) i j
  let transpose := fun i j => write (tile.col * 128 + j.val) (tile.row * 128 + i.val) i j
  ⟨tile, direct, transpose⟩

theorem full_tile_stores (bank : Bank) (op : Operation) (input : Input) (tile : Tile)
    (h : tile ∈ compactGrid bank) : stores false bank op input tile = stores true bank op input tile := by
  obtain ⟨_, hr, hc⟩ := scheduled_tile_bounds bank tile h
  simp only [stores, TileStores.mk.injEq, true_and]
  constructor <;> funext i j <;>
    simp [lane_in_bounds bank hr i, lane_in_bounds bank hc j, full_tile_element bank op input tile h i j]

/-- The two source stores overlap on the diagonal. The mirrored store has
the same destination for lane `(i,j)` as the direct store at that lane. -/
theorem diagonal_store_addresses (masked : Bool) (bank : Bank) (op : Operation) (input : Input)
    (tile : Tile) (h : tile.row = tile.col) (i j : Fin 128) :
    ((stores masked bank op input tile).direct i j).map Prod.fst =
      ((stores masked bank op input tile).transpose j i).map Prod.fst := by
  simp [stores, h]
  split <;> rfl

/-- Final write image of the tile. A direct diagonal store is shadowed by
the later store with identical address coverage. -/
def TileStores.finalWrites (issued : TileStores) : TileStores :=
  if issued.target.row = issued.target.col then { issued with direct := fun _ _ => none }
  else issued

def issueStores (skipDiagonal masked : Bool) (bank : Bank) (op : Operation) (input : Input)
    (tile : Tile) : TileStores :=
  let issued := stores masked bank op input tile
  if skipDiagonal then issued.finalWrites else issued

def owningTile (batch row col : Nat) : Tile := ⟨batch, max row col, min row col⟩

set_option maxRecDepth 4096 in
set_option maxHeartbeats 2000000 in
theorem owning_tile_scheduled (bank : Bank) :
    ∀ (b : Fin (batches bank)) (r c : Fin (Bank.side bank)),
      owningTile b.val r.val c.val ∈ compactGrid bank := by
  cases bank <;> decide

structure Cell (bank : Bank) where
  batch : Fin (batches bank)
  row : Fin (outputWidth bank)
  col : Fin (outputWidth bank)

def Cell.tile {bank : Bank} (p : Cell bank) : Tile :=
  owningTile p.batch.val (p.row.val / 128) (p.col.val / 128)

theorem cell_tile_scheduled {bank : Bank} (p : Cell bank) : p.tile ∈ compactGrid bank := by
  have hr := p.row.isLt
  have hc := p.col.isLt
  simp only [outputWidth] at hr hc
  exact owning_tile_scheduled bank p.batch
    ⟨p.row.val / 128, by omega⟩ ⟨p.col.val / 128, by omega⟩

def storeAt (skipDiagonal masked : Bool) (bank : Bank) (op : Operation) (input : Input) (p : Cell bank) :
    Option (Nat × BF16) :=
  let i : Fin 128 := ⟨p.row.val % 128, by omega⟩
  let j : Fin 128 := ⟨p.col.val % 128, by omega⟩
  let issued := issueStores skipDiagonal masked bank op input p.tile
  -- The transposed write is last on the diagonal tile as well.
  if p.row.val / 128 ≤ p.col.val / 128 then issued.transpose j i else issued.direct i j

theorem diagonal_store_elision (masked : Bool) (bank : Bank) (op : Operation) (input : Input)
    (p : Cell bank) : storeAt true masked bank op input p = storeAt false masked bank op input p := by
  simp only [storeAt, issueStores, Bool.false_eq_true, if_true, if_false]
  split
  · unfold TileStores.finalWrites
    split <;> rfl
  · rename_i above
    have le : p.col.val / 128 ≤ p.row.val / 128 := by omega
    have different : p.row.val / 128 ≠ p.col.val / 128 := by omega
    simp [TileStores.finalWrites, stores, Cell.tile, owningTile, max_eq_left le, min_eq_right le, different]

theorem store_at_address (bank : Bank) (op : Operation) (input : Input) (p : Cell bank) :
    (storeAt false false bank op input p).map Prod.fst = some (outputOffset bank p.tile p.row.val p.col.val) := by
  have hr : p.row.val / 128 * 128 + p.row.val % 128 = p.row.val := by omega
  have hc : p.col.val / 128 * 128 + p.col.val % 128 = p.col.val := by omega
  simp only [storeAt]
  split
  · rename_i le
    simp [issueStores, stores, outputOffset, Cell.tile, owningTile, max_eq_right le, min_eq_left le, hr, hc]
  · rename_i notLe
    have le : p.col.val / 128 ≤ p.row.val / 128 := by omega
    simp [issueStores, stores, outputOffset, Cell.tile, owningTile, max_eq_left le, min_eq_right le, hr, hc]

def overwrite (skipDiagonal masked : Bool) (bank : Bank) (grid : List Tile) (op : Operation) (input : Input)
    (prior : Nat → BF16) (p : Cell bank) : BF16 :=
  let original := prior (outputOffset bank p.tile p.row.val p.col.val)
  if p.tile ∈ grid then ((storeAt skipDiagonal masked bank op input p).map Prod.snd).getD original
  else original

theorem full_overwrite (bank : Bank) (op : Operation) (input : Input) (prior other : Nat → BF16)
    (p : Cell bank) : overwrite true false bank (compactGrid bank) op input prior p =
      overwrite false true bank (compactGrid bank) op input other p := by
  simp [overwrite, cell_tile_scheduled p, diagonal_store_elision]
  have same : storeAt false false bank op input p = storeAt false true bank op input p := by
    simp [storeAt, issueStores, full_tile_stores bank op input p.tile (cell_tile_scheduled p)]
  rw [← same]
  have address := store_at_address bank op input p
  cases h : storeAt false false bank op input p with
  | none => simp [h] at address
  | some word => rfl

structure Output (bank : Bank) where
  /-- Per-tile final write image after the launch. -/
  tiles : List TileStores
  words : Cell bank → BF16

abbrev Kernel := (bank : Bank) → Operation → Input → Output bank

def reference : Kernel := fun b op input =>
  ⟨(squareGrid b).map (fun t => (issueStores false true b op input t).finalWrites),
    overwrite false true b (squareGrid b) op input input.retainedOutput⟩

def compact : Kernel := fun b op input =>
  ⟨(compactGrid b).map (issueStores true false b op input),
    overwrite true false b (compactGrid b) op input input.freshOutput⟩

theorem stores_equal : compact = reference := by
  funext b op input
  simp only [compact, reference, ← ordered_grid_equal b, Output.mk.injEq]
  refine ⟨List.map_congr_left ?_, funext (full_overwrite b op input _ _)⟩
  intro t member
  simp [issueStores, full_tile_stores b op input t member]

/-- The compact launch with its original per-cell masks and stores. -/
def triangular : Kernel := fun b op input =>
  ⟨(compactGrid b).map (fun t => (issueStores false true b op input t).finalWrites),
    overwrite false true b (compactGrid b) op input input.retainedOutput⟩

theorem triangular_equal : triangular = reference := by
  funext b op input
  simp only [triangular, reference, ordered_grid_equal]

/-- The projection coefficient stays a Python FP64 word. The polynomial
scalars are the FP32 words passed to Triton. -/
structure Coefficients where
  projection : UInt64
  beta : F32
  alpha : F32

private def w := Value.ofBits .fp32

def coefficients : List Coefficients :=
  Driver.anvilCoefficientWords.map fun (a, b, c) => ⟨a, w b, w c⟩

set_option maxRecDepth 4096 in
theorem removed_diagonal_words :
    2 * coefficients.length * ([Bank.qk, .vo, .mlp].map (fun bank =>
      ((compactGrid bank).filter fun tile => tile.row == tile.col).length)).sum * 128 * 128 = 8257536 := by
  decide

structure Runtime (State : Type) where
  readPreparation : (bank : Bank) → State → Option (AnvilRails.Input bank × State)
  commitPreparation : (bank : Bank) → AnvilRails.Output bank → State → Option State
  read : Bank → Operation → State → Option (Input × State)
  commit : (bank : Bank) → Operation → Output bank → State → Option State
  readNormalization : Bank → State → Option (AnvilTrace.Input × State)
  commitNormalization : (bank : Bank) → AnvilTrace.Output bank → State → Option State
  project : Bank → UInt64 → State → Option State

def launch {State : Type} (rt : Runtime State) (kernel : Kernel) (bank : Bank)
    (op : Operation) (state : State) : Option State := do
  let (input, state) ← rt.read bank op state
  rt.commit bank op (kernel bank op input) state

abbrev Initial := (State : Type) → Runtime State → Bank → AnvilTrace.Normalization → State → Option State

def splitInitial (gram : Kernel) : Initial := fun _ rt bank normalize state => do
  let state ← launch rt gram bank .gram state
  let (input, state) ← rt.readNormalization bank state
  rt.commitNormalization bank (normalize bank input) state

abbrev Round := (State : Type) → Runtime State → Bank → Coefficients → State → Option State

def splitRound (kernel : Kernel) : Round := fun _ rt bank c state =>
  (launch rt kernel bank .gram state).bind (launch rt kernel bank (.polynomial c.alpha c.beta))

/-- The first Gram precedes normalization. Rounds two through six
recompute it before applying the polynomial and the matrix update. -/
def cascade {State : Type} (rt : Runtime State) (prepare : AnvilRails.Dispatch)
    (kernel : Kernel) (normalize : AnvilTrace.Normalization)
    (bank : Bank) (state : State) (runRound : Round := splitRound kernel)
    (runInitial : Initial := splitInitial kernel) : Option State := do
  let (input, state) ← rt.readPreparation bank state
  let state ← rt.commitPreparation bank (prepare bank input) state
  let state ← runInitial State rt bank normalize state
  coefficients.zipIdx.foldlM (fun state (c, round) => do
    let state ← if round == 0 then launch rt kernel bank (.polynomial c.alpha c.beta) state
      else runRound State rt bank c state
    rt.project bank c.projection state) state

theorem cascade_equal {State : Type} (rt : Runtime State) (bank : Bank) (s : State) :
    cascade rt AnvilRails.optimized compact (AnvilNormalization.packedNormalize AnvilTrace.optimized) bank s =
      cascade rt AnvilRails.reference reference (AnvilTrace.normalize AnvilTrace.original) bank s := by
  rw [AnvilRails.dispatch_equal, stores_equal, AnvilNormalization.packed_normalize_eq, AnvilTrace.dispatch_equal]

#axiomsClean ordered_grid_equal
#axiomsClean compact_has_unique_tiles
#axiomsClean scheduled_tile_bounds
#axiomsClean full_tile_element
#axiomsClean full_tile_stores
#axiomsClean diagonal_store_addresses
#axiomsClean diagonal_store_elision
#axiomsClean removed_diagonal_words
#axiomsClean cell_tile_scheduled
#axiomsClean store_at_address
#axiomsClean full_overwrite
#axiomsClean stores_equal
#axiomsClean cascade_equal

end VeriTile.Bench.Models.NanoGPT.Track1.AnvilGram
