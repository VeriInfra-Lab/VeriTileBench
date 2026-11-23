import models.NanoGPT.Track1.QKVGluonShared
import VeriTile.Triton.Float.ScalarOps

/-! The tied embedding reads a transposed BF16 gradient before reduction
and a transposed BF16 weight after the LM-head gather. The optimized
kernel exchanges four 128-bit vectors per lane using transposed matrix stores. The packed
addition keeps the reference destination and source operand order. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.Bf16Transpose
open VeriTile.Triton FP QKVPack

inductive Mode where | copy | add
  deriving DecidableEq

def Mode.rowTiles : Mode → Nat | .copy => 12 | .add => 786
def Mode.colTiles : Mode → Nat | .copy => 393 | .add => 6
def Mode.rows (mode : Mode) : Nat := mode.rowTiles * 64
def Mode.cols (mode : Mode) : Nat := mode.colTiles * 128

structure Thread where
  warp : Fin 8
  lane : Fin 32
  reg : Fin 32

private def threadID (t : Thread) : Nat := 32 * t.warp.val + t.lane.val

/-- The four matrix registers contain the eight consecutive halfwords
loaded for this row. The column of each fragment selects the provider
of its 16-byte shared row address. -/
private def matrixProvider (t : Thread) : Nat :=
  32 * t.warp.val + 8 * (t.reg.val % 8 / 2) + 2 * (t.lane.val % 4) + t.reg.val % 2

private def sharedSwizzle (tid : Nat) : Nat :=
  ((tid <<< 4) &&& 112) ^^^ ((tid <<< 2) &&& 992)

def matrixStoreAddress (t : Thread) : Nat :=
  let provider := matrixProvider t
  ((provider <<< 7) &&& 3072) + sharedSwizzle provider + 2 * (t.lane.val / 4)

def vectorLoadAddress (t : Thread) : Nat :=
  let tid := threadID t
  ((tid <<< 9) &&& 3072) + sharedSwizzle tid + 2 * (t.reg.val % 8)

private def decodeSharedAddress (address : Nat) : Nat × Nat × Nat :=
  ((address >>> 7) &&& 7,
   ((address >>> 4) &&& 1) ||| ((address >>> 9) &&& 6) |||
     (((address >>> 2) &&& 24) ^^^ ((address >>> 7) &&& 24)),
   (address >>> 1) &&& 7)

theorem matrix_store_location (t : Thread) :
    matrixStoreAddress t + 1 < 4096 ∧
      decodeSharedAddress (matrixStoreAddress t) =
        (t.warp.val, 8 * (t.lane.val % 4) + t.reg.val % 8, t.lane.val / 4) := by
  rcases t with ⟨warp, lane, reg⟩
  decide +kernel +revert

theorem vector_load_location (t : Thread) :
    vectorLoadAddress t + 1 < 4096 ∧
      decodeSharedAddress (vectorLoadAddress t) = (t.warp.val, t.lane.val, t.reg.val % 8) := by
  rcases t with ⟨warp, lane, reg⟩
  decide +kernel +revert

theorem shared_writer_unique (a b : Thread) (samePhase : a.reg.val / 8 = b.reg.val / 8)
    (sameAddress : matrixStoreAddress a = matrixStoreAddress b) : a = b := by
  have h := congrArg decodeSharedAddress sameAddress
  rw [(matrix_store_location a).2, (matrix_store_location b).2] at h
  have warpEq : a.warp.val = b.warp.val := congrArg Prod.fst h
  have c : 8 * (a.lane.val % 4) + a.reg.val % 8 = 8 * (b.lane.val % 4) + b.reg.val % 8 :=
    congrArg (fun pair => pair.2.1) h
  have r : a.lane.val / 4 = b.lane.val / 4 := congrArg (fun pair => pair.2.2) h
  have w : a.warp = b.warp := Fin.ext warpEq
  have l : a.lane = b.lane := Fin.ext (by omega)
  have reg : a.reg = b.reg := Fin.ext (by omega)
  cases a
  cases b
  cases w
  cases l
  cases reg
  rfl

structure Tile (mode : Mode) where
  row : Fin mode.rowTiles
  col : Fin mode.colTiles

structure Point (mode : Mode) where
  tile : Tile mode
  rowLane : Fin 64
  colLane : Fin 128

def Point.row {mode : Mode} (p : Point mode) : Fin mode.rows :=
  ⟨64 * p.tile.row.val + p.rowLane.val, by
    have := p.tile.row.isLt
    have := p.rowLane.isLt
    unfold Mode.rows
    omega⟩

def Point.col {mode : Mode} (p : Point mode) : Fin mode.cols :=
  ⟨128 * p.tile.col.val + p.colLane.val, by
    have := p.tile.col.isLt
    have := p.colLane.isLt
    unfold Mode.cols
    omega⟩

def point (mode : Mode) (r : Fin mode.rows) (c : Fin mode.cols) : Point mode :=
  ⟨⟨⟨r.val / 64, by have h := r.isLt; unfold Mode.rows at h; omega⟩,
     ⟨c.val / 128, by have h := c.isLt; unfold Mode.cols at h; omega⟩⟩,
    ⟨r.val % 64, Nat.mod_lt _ (by decide)⟩,
    ⟨c.val % 128, Nat.mod_lt _ (by decide)⟩⟩

theorem tiles_cover (mode : Mode) (r : Fin mode.rows) (c : Fin mode.cols) :
    (point mode r c).row = r ∧ (point mode r c).col = c := by
  constructor <;> apply Fin.ext <;> simp only [point, Point.row, Point.col] <;> omega

def outputOwner {mode : Mode} (p : Point mode) : Thread :=
  ⟨⟨p.rowLane.val / 8, by have := p.rowLane.isLt; omega⟩,
    ⟨p.colLane.val % 32, Nat.mod_lt _ (by decide)⟩,
    ⟨8 * (p.colLane.val / 32) + p.rowLane.val % 8, by have := p.colLane.isLt; omega⟩⟩

def inputPoint {mode : Mode} (tile : Tile mode) (t : Thread) : Point mode :=
  ⟨tile, ⟨8 * t.warp.val + t.lane.val / 4, by
    have := t.warp.isLt
    have := t.lane.isLt
    omega⟩,
    ⟨32 * (t.reg.val / 8) + 8 * (t.lane.val % 4) + t.reg.val % 8, by
      have := t.reg.isLt
      omega⟩⟩

def outputPoint {mode : Mode} (tile : Tile mode) (t : Thread) : Point mode :=
  ⟨tile, ⟨8 * t.warp.val + t.reg.val % 8, by have := t.warp.isLt; omega⟩,
    ⟨32 * (t.reg.val / 8) + t.lane.val, by have := t.reg.isLt; have := t.lane.isLt; omega⟩⟩

def inputThread (t : Thread) : Thread :=
  ⟨t.warp, ⟨4 * (t.reg.val % 8) + t.lane.val / 8, by have := t.lane.isLt; omega⟩,
    ⟨8 * (t.reg.val / 8) + t.lane.val % 8, by have := t.reg.isLt; omega⟩⟩

theorem shared_load_matches (t : Thread) :
    matrixStoreAddress (inputThread t) = vectorLoadAddress t ∧
      (inputThread t).warp = t.warp ∧ (inputThread t).reg.val / 8 = t.reg.val / 8 := by
  rcases t with ⟨warp, lane, reg⟩
  decide +kernel +revert

theorem producer_cell {mode : Mode} (tile : Tile mode) (t : Thread) :
    inputPoint tile (inputThread t) = outputPoint tile t := by
  have hl := t.lane.isLt
  simp only [inputPoint, inputThread, outputPoint, Point.mk.injEq, true_and]
  constructor
  · apply Fin.ext
    change 8 * t.warp.val + (4 * (t.reg.val % 8) + t.lane.val / 8) / 4 =
      8 * t.warp.val + t.reg.val % 8
    omega
  · apply Fin.ext
    change 32 * ((8 * (t.reg.val / 8) + t.lane.val % 8) / 8) +
      8 * ((4 * (t.reg.val % 8) + t.lane.val / 8) % 4) +
      (8 * (t.reg.val / 8) + t.lane.val % 8) % 8 = 32 * (t.reg.val / 8) + t.lane.val
    omega

theorem output_covered {mode : Mode} (p : Point mode) : outputPoint p.tile (outputOwner p) = p := by
  cases p
  unfold outputPoint outputOwner
  congr <;> dsimp <;> omega

theorem output_owner_unique {mode : Mode} (tile : Tile mode) (p q : Thread)
    (h : outputPoint tile p = outputPoint tile q) : p = q := by
  have hr := congrArg (fun i => i.rowLane.val) h
  have hc := congrArg (fun i => i.colLane.val) h
  have hp := p.lane.isLt
  have hq := q.lane.isLt
  simp only [outputPoint] at hr hc
  have hw : p.warp = q.warp := Fin.ext (by omega)
  have hl : p.lane = q.lane := Fin.ext (by omega)
  have hv : p.reg = q.reg := Fin.ext (by omega)
  cases p
  cases q
  congr

def sourceOffset {mode : Mode} (p : Point mode) : Nat :=
  2 * (p.row.val * mode.cols + p.col.val)

def destOffset {mode : Mode} (p : Point mode) : Nat :=
  2 * (p.col.val * mode.rows + p.row.val)

theorem global_offsets_bounded {mode : Mode} (p : Point mode) :
    sourceOffset p + 1 < 2 * mode.rows * mode.cols ∧
      destOffset p + 1 < 2 * mode.rows * mode.cols ∧ 2 * mode.rows * mode.cols < 2^31 := by
  have hr := p.row.isLt
  have hc := p.col.isLt
  cases mode <;>
    simp only [sourceOffset, destOffset, Mode.rows, Mode.cols, Mode.rowTiles, Mode.colTiles] at * <;> omega

theorem unique_destination {mode : Mode} (p q : Point mode)
    (h : destOffset p = destOffset q) : p = q := by
  have hp := p.rowLane.isLt
  have hq := q.rowLane.isLt
  have hcp := p.colLane.isLt
  have hcq := q.colLane.isLt
  have hpr := p.row.isLt
  have hqr := q.row.isLt
  unfold destOffset at h
  have hr : p.row.val = q.row.val := by
    cases mode <;> simp only [Mode.rows, Mode.rowTiles] at h hpr hqr <;> omega
  have hc : p.col.val = q.col.val := by
    rw [hr] at h
    cases mode <;> simp only [Mode.rows, Mode.rowTiles] at h <;> omega
  simp only [Point.row, Point.col] at hr hc
  have htr : p.tile.row = q.tile.row := Fin.ext (by omega)
  have htc : p.tile.col = q.tile.col := Fin.ext (by omega)
  have hlr : p.rowLane = q.rowLane := Fin.ext (by omega)
  have hlc : p.colLane = q.colLane := Fin.ext (by omega)
  cases p with
  | mk pt pr pc =>
    cases q with
    | mk qt qr qc =>
      cases pt
      cases qt
      congr

private theorem lane_masks (warp : Fin 8) (lane : Fin 32) :
    let tid := 32 * warp.val + lane.val
    ((tid >>> 2) &&& 63) = 8 * warp.val + lane.val / 4 ∧
    ((tid <<< 3) &&& 24) = 8 * (lane.val % 4) ∧
    (tid &&& 31) = lane.val ∧ ((tid >>> 2) &&& 56) = 8 * warp.val := by
  decide +kernel +revert

private theorem tile_base (shift tile within : Nat) (h : within < 2^shift) :
    within ||| (tile <<< shift) = tile * 2^shift + within := by
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt h, Nat.shiftLeft_eq]

/-- Each global source lane loads one row with four fixed column
offsets. The row pitch is formed in signed i32 before wide scaling. -/
def compiledSource {mode : Mode} (tile : Tile mode) (t : Thread) : Nat :=
  let r := ((threadID t >>> 2) &&& 63) ||| (tile.row.val <<< 6)
  let c := ((threadID t <<< 3) &&& 24) ||| (tile.col.val <<< 7)
  2 * ((r * mode.cols) % 2^32 + c + 32 * (t.reg.val / 8) + t.reg.val % 8)

/-- The destination halfword offset forms a signed i32 multiply-add
before scaling. The vector element contributes its final two bytes. -/
def compiledDestination {mode : Mode} (tile : Tile mode) (t : Thread) : Nat :=
  let r := ((threadID t >>> 2) &&& 56) ||| (tile.row.val <<< 6)
  let c := (threadID t &&& 31) ||| (tile.col.val <<< 7)
  2 * ((c * mode.rows + r + (t.reg.val / 8) * (32 * mode.rows)) % 2^32 + t.reg.val % 8)

theorem compiled_source_exact {mode : Mode} (tile : Tile mode) (t : Thread) :
    compiledSource tile t = sourceOffset (inputPoint tile t) := by
  obtain ⟨hr, hc, _, _⟩ := lane_masks t.warp t.lane
  have hw := t.warp.isLt
  have hl := t.lane.isLt
  have ht := t.reg.isLt
  have h := (global_offsets_bounded (inputPoint tile t)).1
  have hsize := (global_offsets_bounded (inputPoint tile t)).2.2
  unfold compiledSource threadID
  rw [hr, hc, tile_base _ _ _ (by omega), tile_base _ _ _ (by omega)]
  unfold sourceOffset Point.row Point.col inputPoint at h ⊢
  dsimp at h ⊢
  rw [Nat.mul_comm tile.row.val 64]
  rw [Nat.mod_eq_of_lt (by omega)]
  omega

theorem compiled_destination_exact {mode : Mode} (tile : Tile mode) (t : Thread) :
    compiledDestination tile t = destOffset (outputPoint tile t) := by
  obtain ⟨_, _, hc, hr⟩ := lane_masks t.warp t.lane
  have hw := t.warp.isLt
  have hl := t.lane.isLt
  have ht := t.reg.isLt
  have h := (global_offsets_bounded (outputPoint tile t)).2.1
  have hsize := (global_offsets_bounded (outputPoint tile t)).2.2
  unfold compiledDestination threadID
  rw [hr, hc, tile_base _ _ _ (by omega), tile_base _ _ _ (by omega)]
  unfold destOffset Point.row Point.col outputPoint at h ⊢
  dsimp at h ⊢
  have pitch : (t.reg.val / 8) * (32 * mode.rows) = (32 * (t.reg.val / 8)) * mode.rows := by ac_rfl
  rw [pitch, Nat.add_right_comm, ← Nat.add_mul]
  have col : tile.col.val * 128 + t.lane.val + 32 * (t.reg.val / 8) =
      128 * tile.col.val + (32 * (t.reg.val / 8) + t.lane.val) := by omega
  rw [col, Nat.mod_eq_of_lt (by omega)]
  omega

abbrev BF := Value .bf16
abbrev Bank := Nat → BF

def reference (cfg : Config) {mode : Mode} (src dst : Bank) (p : Point mode) : Nat × BF :=
  let atDst := destOffset p
  let atSrc := src (sourceOffset p)
  (atDst, match mode with
    | .copy => atSrc
    | .add => FP.add cfg (dst atDst) atSrc)

private def pairMember (reg : Fin 32) (half : Fin 2) : Fin 32 :=
  ⟨2 * (reg.val / 2) + half.val, by have := reg.isLt; have := half.isLt; omega⟩

private def member (reg : Fin 32) : Fin 2 := ⟨reg.val % 2, Nat.mod_lt _ (by decide)⟩

private theorem pair_member (reg : Fin 32) : pairMember reg (member reg) = reg := by
  apply Fin.ext
  simp only [pairMember, member]
  omega

private theorem thread_pair_member (t : Thread) :
    { t with reg := pairMember t.reg (member t.reg) } = t := by
  cases t
  simp only [pair_member]

/-- Each lane of the PTX instruction rounds the sum of its own two
operands to BF16. Exchanging halfwords leaves the packed operands intact. -/
def addPacked (cfg : Config) (left right : Fin 2 → BF) (half : Fin 2) : BF :=
  FP.add cfg (left half) (right half)

def emitted (cfg : Config) {mode : Mode} (src dst : Bank) (p : Point mode) : Nat × BF :=
  let t := outputOwner p
  let incoming := fun (t : Thread) => src (compiledSource p.tile (inputThread t))
  let target := compiledDestination p.tile t
  (target, match mode with
    | .copy => incoming t
    | .add =>
      let halves := fun i => { t with reg := pairMember t.reg i }
      addPacked cfg (fun i => dst (compiledDestination p.tile (halves i)))
        (fun i => incoming (halves i)) (member t.reg))

theorem emitted_equal (cfg : Config) {mode : Mode} (src dst : Bank) (p : Point mode) :
    emitted cfg src dst p = reference cfg src dst p := by
  cases mode <;> simp only [emitted, reference, addPacked, thread_pair_member, compiled_source_exact,
    compiled_destination_exact, producer_cell, output_covered]

abbrev Dispatch := (fastPath : Bool) → Config → (mode : Mode) → Bank → Bank → Point mode → Nat × BF

structure Operands where
  hopper : Bool
  config : Config
  source : Bank
  destination : Bank

structure Runtime (State : Type) where
  read : Mode → State → Option (Operands × State)
  write : (mode : Mode) → (Point mode → Nat × BF) → State → Option State

def execute {State : Type} (runtime : Runtime State) (dispatch : Dispatch) (mode : Mode)
    (state : State) : Option State := do
  let (bank, next) ← runtime.read mode state
  runtime.write mode (dispatch bank.hopper bank.config mode bank.source bank.destination) next

/-- The first continuation runs before the tied lm_head reduction.
The second runs after the lm_head gather wait. -/
structure OptimizerHooks (State : Type) where
  beforeHeadReduce : State → Option State
  afterHeadGather : State → Option State

def warpEvents : List (GluonShared.Event 4) :=
  [.write 0, .barrier, .read 0, .barrier, .write 1, .barrier, .read 1, .barrier,
   .write 2, .barrier, .read 2, .barrier, .write 3, .barrier, .read 3]

theorem phases_safe : GluonShared.checkedPhases warpEvents = some [0, 1, 2, 3] := rfl

#axiomsClean tiles_cover
#axiomsClean output_covered
#axiomsClean output_owner_unique
#axiomsClean global_offsets_bounded
#axiomsClean unique_destination
#axiomsClean compiled_source_exact
#axiomsClean compiled_destination_exact
#axiomsClean emitted_equal
#axiomsClean phases_safe
#axiomsClean matrix_store_location
#axiomsClean vector_load_location
#axiomsClean shared_load_matches
#axiomsClean producer_cell
#axiomsClean shared_writer_unique

end VeriTile.Bench.Models.NanoGPT.Track1.Bf16Transpose
