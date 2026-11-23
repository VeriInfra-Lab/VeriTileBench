import models.NanoGPT.Track1.ByteTranspose

/-! The two FP8 inputs of the cross-entropy weight-gradient GEMM have the
same token axis. Concatenate their column-tile grids and keep their
address spaces and transposed layouts separate. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.CETranspose

inductive Operand where
  | activation | gradient
  deriving DecidableEq

structure Shape where
  tokens : Nat
  classes : Nat
  activationFits : (tokens * 128) * (6 * 128) < 2^31
  gradientFits : (tokens * 128) * (classes * 128) < 2^31

def Shape.operand (s : Shape) : Operand → ByteTranspose.Shape
  | .activation => ⟨s.tokens, 6, s.activationFits⟩
  | .gradient => ⟨s.tokens, s.classes, s.gradientFits⟩

abbrev Cell (s : Shape) := (o : Operand) × ByteTranspose.Point (s.operand o)
abbrev X (s : Shape) := Fin (6 + s.classes)

def launchX {s : Shape} : Cell s → X s
  | ⟨.activation, p⟩ => ⟨p.tileCol.val, by have := p.tileCol.isLt; change _ < 6 at this; omega⟩
  | ⟨.gradient, p⟩ => ⟨6 + p.tileCol.val, Nat.add_lt_add_left p.tileCol.isLt 6⟩

def route (s : Shape) (x : X s) (y : Fin s.tokens) (cell : SampledRows.Cell) : Cell s :=
  if activation : x.val < 6 then ⟨.activation, ⟨y, ⟨x.val, activation⟩, cell⟩⟩
  else ⟨.gradient, ⟨y, ⟨x.val - 6, by
    change x.val - 6 < s.classes
    have := x.isLt
    omega⟩, cell⟩⟩

def launchY {s : Shape} : Cell s → Fin s.tokens
  | ⟨.activation, p⟩ => p.tileRow
  | ⟨.gradient, p⟩ => p.tileRow

theorem route_exact {s : Shape} (p : Cell s) :
    route s (launchX p) (launchY p) p.2.cell = p := by
  rcases p with ⟨operand, ⟨y, x, cell⟩⟩
  cases operand with
  | activation =>
    have bound : x.val < 6 := x.isLt
    simp [route, launchX, launchY, bound]
  | gradient =>
    simp [route, launchX, launchY]

theorem launch_covers (s : Shape) (x : X s) (y : Fin s.tokens) (cell : SampledRows.Cell) :
    launchX (route s x y cell) = x ∧ launchY (route s x y cell) = y := by
  unfold route
  split
  · simp [launchX, launchY]
  · rename_i outside
    have same : 6 + (x.val - 6) = x.val := by omega
    exact ⟨Fin.ext same, rfl⟩

abbrev Stores := (s : Shape) → (Operand → Nat → UInt8) → Cell s → Option (Operand × Nat × UInt8)
abbrev Dispatch := Bool → Stores

def staged : Stores := fun _ source p =>
  (ByteTranspose.reference (source p.1) p.2).map (p.1, ·)

def paired : Stores := fun s source p =>
  let tile := route s (launchX p) (launchY p) p.2.cell
  (ByteTranspose.emitted (source tile.1) tile.2).map (tile.1, ·)

theorem stores_equal : paired = staged := by
  funext s source p
  unfold paired staged
  let observe := fun tile : Cell s => (ByteTranspose.emitted (source tile.1) tile.2).map (tile.1, ·)
  exact (congrArg observe (route_exact p)).trans (by simp only [observe, ByteTranspose.emitted_equal])

def byteOffset (offset : Nat) : Nat := (UInt32.ofNat offset).toUInt64.toNat

private theorem byteOffset_exact (offset : Nat) (bounded : offset < 2^31) :
    byteOffset offset = offset := by
  unfold byteOffset
  rw [UInt32.toNat_toUInt64, UInt32.toNat_ofNat', Nat.mod_eq_of_lt]
  exact Nat.lt_trans bounded (by decide)

theorem extended_pitches_exact {s : ByteTranspose.Shape} (p : ByteTranspose.Point s) :
    byteOffset (p.row.val * s.cols) + p.col.val = ByteTranspose.sourceOffset p ∧
    byteOffset (p.col.val * s.rows) + p.row.val = ByteTranspose.destOffset p := by
  have bounds := ByteTranspose.signed_offsets_exact p
  have input := bounds.2.2.1
  have output := bounds.2.2.2
  unfold ByteTranspose.sourceOffset at input
  unfold ByteTranspose.destOffset at output
  unfold ByteTranspose.sourceOffset ByteTranspose.destOffset
  rw [byteOffset_exact _ (by omega), byteOffset_exact _ (by omega)]
  exact ⟨rfl, rfl⟩

namespace Wide
open SampledRows (WideThread wideColIndex wideColOwner col_covers)

structure InputByte where
  thread : Fin 256
  pair : Fin 2
  byte : Fin 16
  half : Fin 2

def loadedCell (input : InputByte) : SampledRows.Cell :=
  (⟨2 * (input.thread.val / 4) + input.byte.val / 2 % 2, by
      have := input.thread.isLt
      omega⟩,
    ⟨64 * input.half.val + 16 * (input.thread.val % 4) + 8 * input.pair.val +
      2 * (input.byte.val / 4) + input.byte.val % 2, by
      have := input.half.isLt
      have := input.pair.isLt
      have := input.byte.isLt
      omega⟩)

/-- The two vector stores interleave pairs of bytes from adjacent rows. -/
def sharedStoreAddress (input : InputByte) : Nat :=
  let tid := input.thread.val
  let packed := (((tid &&& 3) <<< 5) ||| ((tid <<< 4) &&& 3968)) |||
    (if tid.testBit 2 then 4112 else 0)
  (packed ^^^ (input.pair.val <<< 4)) + input.byte.val

/-- Lane selection in the two ldmatrix reads, followed by the PRMT that
selects the even or odd byte from each halfword. -/
def sharedLoadAddress (output : WideThread) : Nat :=
  let word := output.reg.val % 16 / 4
  let byte := output.reg.val % 4
  let matrix := 2 * (word % 2) + byte / 2
  let provider := 32 * output.warp.val + 8 * matrix + output.lane.val / 4
  let base := ((provider <<< 3) &&& 128) +
    (((provider <<< 4) &&& 3696) ^^^ (if provider.testBit 3 then 4112 else 0))
  base + 256 * (word / 2) + 4 * (output.lane.val % 4) +
    2 * (byte % 2) + output.reg.val / 16 % 2

private def decodeAddress (address : Nat) : Nat × Nat × Nat :=
  ((((address >>> 4) &&& 248) ||| ((address >>> 5) &&& 3)) ||| ((address >>> 10) &&& 4),
    ((address >>> 4) &&& 1) ^^^ ((address >>> 12) &&& 1), address &&& 15)

def readInput (output : WideThread) : InputByte :=
  let (thread, pair, byte) := decodeAddress (sharedLoadAddress output)
  ⟨Fin.ofNat 256 thread, Fin.ofNat 2 pair, Fin.ofNat 16 byte,
    ⟨output.reg.val / 32, by have := output.reg.isLt; omega⟩⟩

theorem store_location (input : InputByte) :
    let address := sharedStoreAddress input
    address < 8192 ∧ address / 512 % 8 = input.thread.val / 32 ∧
      decodeAddress address = (input.thread.val, input.pair.val, input.byte.val) := by
  rcases input with ⟨thread, pair, byte, half⟩
  revert thread pair byte half
  decide +kernel

theorem unique_shared_store (a b : InputByte) (sameHalf : a.half = b.half)
    (sameAddress : sharedStoreAddress a = sharedStoreAddress b) : a = b := by
  have decoded := congrArg decodeAddress sameAddress
  rw [(store_location a).2.2, (store_location b).2.2] at decoded
  have ht := congrArg Prod.fst decoded
  have hp := congrArg (fun p : Nat × Nat × Nat => p.2.1) decoded
  have hb := congrArg (fun p : Nat × Nat × Nat => p.2.2) decoded
  have thread : a.thread = b.thread := Fin.ext ht
  have pair : a.pair = b.pair := Fin.ext hp
  have byte : a.byte = b.byte := Fin.ext hb
  cases a
  cases b
  cases thread
  cases pair
  cases byte
  cases sameHalf
  rfl

theorem shared_reads (output : WideThread) :
    sharedStoreAddress (readInput output) = sharedLoadAddress output ∧
      (readInput output).thread.val / 32 = output.warp.val ∧
      loadedCell (readInput output) = wideColIndex output := by
  rcases output with ⟨warp, lane, reg⟩
  revert warp lane reg
  decide +kernel

/-- Each write and read event covers both disjoint 16-byte vectors. -/
def phases : List (QKVPack.GluonShared.Event 2) :=
  [.write 0, .barrier, .read 0, .barrier, .write 1, .barrier, .read 1]

theorem phases_ready : QKVPack.GluonShared.checkedPhases phases = some [0, 1] := rfl

private theorem source_lane_bits (tid : Fin 256) :
    (tid.val >>> 1) &&& 126 = 2 * (tid.val / 4) ∧
      (tid.val &&& 3) <<< 4 = 16 * (tid.val % 4) := by
  revert tid
  decide +kernel

private theorem dest_lane_bits (warp : Fin 8) (lane : Fin 32) :
    ((32 * warp.val + lane.val) >>> 1) &&& 112 = 16 * warp.val ∧
      ((32 * warp.val + lane.val) <<< 1) &&& 62 = 2 * lane.val := by
  revert warp lane
  decide +kernel

private theorem tile_origin (tile within : Nat) (bound : within < 128) :
    within ||| (tile <<< 7) = 128 * tile + within := by
  rw [Nat.or_comm, ← Nat.shiftLeft_add_eq_or_of_lt (i := 7) bound, Nat.shiftLeft_eq]
  omega

def compiledSource {s : ByteTranspose.Shape} (p : ByteTranspose.Point s) (input : InputByte) : Nat :=
  let tid := input.thread.val
  let row := (((tid >>> 1) &&& 126) ||| (p.tileRow.val <<< 7)) + input.byte.val / 2 % 2
  let col := (((tid &&& 3) <<< 4) ||| (p.tileCol.val <<< 7)) +
    64 * input.half.val + 8 * input.pair.val + 2 * (input.byte.val / 4) + input.byte.val % 2
  byteOffset (row * s.cols) + col

def compiledDestination {s : ByteTranspose.Shape} (p : ByteTranspose.Point s) (output : WideThread) : Nat :=
  let tid := 32 * output.warp.val + output.lane.val
  let row := ((tid >>> 1) &&& 112) ||| (p.tileRow.val <<< 7)
  let col := (((tid <<< 1) &&& 62) ||| (p.tileCol.val <<< 7)) +
    64 * (output.reg.val / 32) + output.reg.val / 16 % 2
  byteOffset (col * s.rows + row) + output.reg.val % 16

theorem compiled_source_exact {s : ByteTranspose.Shape} (p : ByteTranspose.Point s) (input : InputByte) :
    compiledSource p input = ByteTranspose.sourceOffset {p with cell := loadedCell input} := by
  let cell := { p with cell := loadedCell input }
  have rowLt := input.thread.isLt
  obtain ⟨row, col⟩ := source_lane_bits input.thread
  simp only [compiledSource, row, col]
  rw [tile_origin p.tileRow.val (2 * (input.thread.val / 4)) (by omega),
    tile_origin p.tileCol.val (16 * (input.thread.val % 4)) (by omega)]
  have r : 128 * p.tileRow.val + 2 * (input.thread.val / 4) + input.byte.val / 2 % 2 =
      cell.row.val := by
    change _ = p.tileRow.val * 128 + (2 * (input.thread.val / 4) + input.byte.val / 2 % 2)
    omega
  have c : 128 * p.tileCol.val + 16 * (input.thread.val % 4) + 64 * input.half.val +
      8 * input.pair.val + 2 * (input.byte.val / 4) + input.byte.val % 2 = cell.col.val := by
    change _ = p.tileCol.val * 128 +
      (64 * input.half.val + 16 * (input.thread.val % 4) + 8 * input.pair.val +
        2 * (input.byte.val / 4) + input.byte.val % 2)
    omega
  rw [r, c]
  exact (extended_pitches_exact cell).1

theorem compiled_destination_exact {s : ByteTranspose.Shape} (p : ByteTranspose.Point s) (output : WideThread) :
    compiledDestination p output = ByteTranspose.destOffset {p with cell := wideColIndex output} := by
  let cell := { p with cell := wideColIndex output }
  have warpLt := output.warp.isLt
  have laneLt := output.lane.isLt
  obtain ⟨row, col⟩ := dest_lane_bits output.warp output.lane
  simp only [compiledDestination, row, col]
  rw [tile_origin p.tileRow.val (16 * output.warp.val) (by omega),
    tile_origin p.tileCol.val (2 * output.lane.val) (by omega)]
  have r : 128 * p.tileRow.val + 16 * output.warp.val + output.reg.val % 16 = cell.row.val := by
    change _ = p.tileRow.val * 128 + (16 * output.warp.val + output.reg.val % 16)
    omega
  have c : 128 * p.tileCol.val + 2 * output.lane.val + 64 * (output.reg.val / 32) +
      output.reg.val / 16 % 2 = cell.col.val := by
    change _ = p.tileCol.val * 128 +
      (64 * (output.reg.val / 32) + 2 * output.lane.val + output.reg.val / 16 % 2)
    omega
  rw [c]
  have bound := (ByteTranspose.signed_offsets_exact cell).2.2.2
  change cell.col.val * s.rows + cell.row.val < 2 ^ 31 at bound
  rw [← r] at bound
  rw [byteOffset_exact _ (by omega)]
  change _ = cell.col.val * s.rows + cell.row.val
  rw [← r]
  omega

#axiomsClean compiled_source_exact
#axiomsClean compiled_destination_exact

def emitted {s : ByteTranspose.Shape} (bytes : Nat → UInt8) (p : ByteTranspose.Point s) :
    Option (Nat × UInt8) :=
  let output := wideColOwner p.cell
  some (compiledDestination p output, bytes (compiledSource p (readInput output)))

theorem emitted_equal {s : ByteTranspose.Shape} (bytes : Nat → UInt8) (p : ByteTranspose.Point s) :
    emitted bytes p = ByteTranspose.reference bytes p := by
  have read := (shared_reads (wideColOwner p.cell)).2.2
  have output := col_covers p.cell
  simp only [emitted, compiled_source_exact, compiled_destination_exact, read, output,
    ByteTranspose.reference, p.row.isLt, p.col.isLt, and_self, if_true]

#axiomsClean store_location
#axiomsClean unique_shared_store
#axiomsClean shared_reads
#axiomsClean phases_ready
#axiomsClean emitted_equal
end Wide

/-- The widened CE transpose uses two shared slices per warp. -/
def unsigned : Stores := fun s source p =>
  let tile := route s (launchX p) (launchY p) p.2.cell
  (Wide.emitted (source tile.1) tile.2).map (tile.1, ·)

theorem unsigned_stores_equal : unsigned = staged := by
  funext s source p
  unfold unsigned staged
  let observe := fun tile : Cell s => (Wide.emitted (source tile.1) tile.2).map (tile.1, ·)
  exact (congrArg observe (route_exact p)).trans (by simp only [observe, Wide.emitted_equal])

theorem staged_operand (s : Shape) (source : Operand → Nat → UInt8) (operand : Operand)
    (token : Fin (s.operand operand).rows) (channel : Fin (s.operand operand).cols) :
    staged s source ⟨operand, ByteTranspose.point _ token channel⟩ =
      some (operand, channel.val * (s.operand operand).rows + token.val,
        source operand (token.val * (s.operand operand).cols + channel.val)) := by
  have ⟨row, col⟩ := ByteTranspose.tiles_cover (s.operand operand) token channel
  simp [staged, ByteTranspose.reference, ByteTranspose.destOffset, ByteTranspose.sourceOffset,
    row, col, token.isLt, channel.isLt]

theorem unique_store {s : Shape} (p q : Cell s)
    (same : (p.1, ByteTranspose.destOffset p.2) = (q.1, ByteTranspose.destOffset q.2)) : p = q := by
  rcases p with ⟨p, left⟩
  rcases q with ⟨q, right⟩
  change (p, ByteTranspose.destOffset left) = (q, ByteTranspose.destOffset right) at same
  obtain ⟨rfl, addr⟩ := Prod.mk.injEq _ _ _ _ |>.mp same
  rw [ByteTranspose.unique_destination _ _ addr]

#axiomsClean route_exact
#axiomsClean launch_covers
#axiomsClean stores_equal
#axiomsClean extended_pitches_exact
#axiomsClean unsigned_stores_equal
#axiomsClean staged_operand
#axiomsClean unique_store
end VeriTile.Bench.Models.NanoGPT.Track1.CETranspose
