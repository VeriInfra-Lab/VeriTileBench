import models.NanoGPT.Track1.ValidationRowBinding
import models.NanoGPT.Track1.DriverExecution

/-! Assemble validation's CTA outputs into the buffer read by the
driver's FP32 batch mean. Each slab supplies a contiguous range of rows. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValidationBatchBinding
open ValidationCE (Tile slabCount slabRows)
open ValidationReduceBinding (rowCount)
open ValidationScanBinding (rowProjection)

abbrev Rows := slabCount * slabRows

structure Location (tile : Tile) where
  slab : Fin slabCount
  block : Nat
  row : Fin (rowCount tile)

def locate (tile : Tile) (i : Fin Rows) : Location tile :=
  let localRow := i.val % slabRows
  ⟨⟨i.val / slabRows, (Nat.div_lt_iff_lt_mul (by decide : 0 < slabRows)).mpr i.isLt⟩,
    localRow / rowCount tile,
    ⟨localRow % rowCount tile, Nat.mod_lt _ (by cases tile <;> decide)⟩⟩

theorem locate_slab_row (tile : Tile) (i : Fin Rows) :
    let loc := locate tile i
    loc.block * rowCount tile + loc.row.val = i.val % slabRows := by
  change i.val % slabRows / rowCount tile * rowCount tile + i.val % slabRows % rowCount tile = _
  rw [Nat.mul_comm _ (rowCount tile), Nat.add_comm, Nat.mod_add_div]

private theorem locate_row (tile : Tile) (i : Fin Rows) :
    let loc := locate tile i
    loc.slab.val * slabRows + loc.block * rowCount tile + loc.row.val = i.val := by
  change i.val / slabRows * slabRows + i.val % slabRows / rowCount tile * rowCount tile +
    i.val % slabRows % rowCount tile = i.val
  have localEq := Nat.mod_add_div (i.val % slabRows) (rowCount tile)
  have slabEq := Nat.mod_add_div i.val slabRows
  rw [Nat.mul_comm slabRows] at slabEq
  rw [Nat.mul_comm (rowCount tile)] at localEq
  omega

structure Ports where
  bf16 : Nat → Int → TailNorm.BF16
  mask : Nat → UInt8
  device : TailNorm.Device

/-- The head's eight projection views and the tokenizer's mask have the
same row and byte coordinates as the kernel launch. -/
def Ports.projection (ports : Ports) (tile : Tile) : ValidationCE.Projection where
  rows := Rows
  vocab := 50304
  projection := fun i col =>
    let loc := locate tile i
    rowProjection tile ⟨loc.slab.val, loc.row.val, 0⟩ loc.block ports.bf16 col.val
  mask := fun token byte => ports.mask (token.val * 6288 + byte)
  tile := tile
  device := ports.device

private def tokenAt (batch : ValidationCE.ProjectedBatch) (width : batch.vocab = 50304)
    (row : Nat) : Fin 50304 :=
  if valid : row < batch.rows then Fin.cast width (batch.input ⟨row, valid⟩) else 0

def runBatchRow {ptx : String} {tile : Tile}
    (compiled : ValidationReduceBinding.Certificate ptx tile)
    (scan : ValidationScanBinding.Certificate ptx tile) (native : ValidationCE.Native)
    (ports : Ports) (batch : ValidationCE.ProjectedBatch) (layout : batch.toProjection = ports.projection tile)
    (i : Fin batch.rows) : Option TailNorm.F32 :=
  let rows := congrArg ValidationCE.Projection.rows layout
  let width := congrArg ValidationCE.Projection.vocab layout
  let loc := locate tile (Fin.cast rows i)
  ValidationRowBinding.runRow compiled scan native loc.block loc.slab loc.row
    (tokenAt batch width) ports.mask ports.bf16 ((batch.target i).map (Fin.cast width))

theorem decoded_batch_row_exact {ptx : String} {tile : Tile}
    (compiled : ValidationReduceBinding.Certificate ptx tile)
    (scan : ValidationScanBinding.Certificate ptx tile) (native : ValidationCE.Native)
    (ports : Ports) (batch : ValidationCE.ProjectedBatch) (layout : batch.toProjection = ports.projection tile)
    (i : Fin batch.rows) :
    runBatchRow compiled scan native ports batch layout i =
      some (ValidationCE.rowLoss native batch.tile (batch.projection i)
        (batch.mask (batch.input i)) (batch.target i)) := by
  cases batch with
  | mk geometry input target =>
    change geometry = ports.projection tile at layout
    subst geometry
    let loc := locate tile i
    let slot : ValidationReduceBinding.Slot := ⟨loc.slab.val, loc.row.val, 0⟩
    let batch : ValidationCE.ProjectedBatch := ⟨ports.projection tile, input, target⟩
    change ValidationRowBinding.runRow compiled scan native loc.block loc.slab loc.row
      (tokenAt batch rfl) ports.mask ports.bf16 ((target i).map (Fin.cast rfl)) =
      some (ValidationCE.rowLoss native tile (fun c => rowProjection tile slot loc.block ports.bf16 c.val)
        (fun byte => ports.mask ((input i).val * 6288 + byte)) (target i))
    rw [ValidationRowBinding.decoded_row_exact]
    have maskEq : ValidationScanBinding.rowMask tile slot loc.block (tokenAt batch rfl) ports.mask =
        fun byte => ports.mask ((input i).val * 6288 + byte) := by
      funext byte
      unfold ValidationScanBinding.rowMask
      change ports.mask ((tokenAt batch rfl
        (loc.slab.val * slabRows + loc.block * rowCount tile + loc.row.val)).val * 6288 + byte) = _
      rw [locate_row tile i]
      unfold tokenAt
      rw [dif_pos i.isLt]
      rfl
    have targetEq : (target i).map (Fin.cast rfl) = target i := by cases target i <;> rfl
    rw [maskEq]
    exact congrArg (fun target => some
      (ValidationCE.rowLoss native tile (fun c => rowProjection tile slot loc.block ports.bf16 c.val)
        (fun byte => ports.mask ((input i).val * 6288 + byte)) target)) targetEq

private def replayArray (f : Fin n → Option α) : Option (Array α) :=
  (((List.finRange n).mapM f).map List.toArray)

private theorem replayArray_eq {f : Fin n → Option α} {g : Fin n → α}
    (same : ∀ i, f i = some (g i)) : replayArray f = some (Array.ofFn g) := by
  have forward : f = fun i => pure (g i) := funext same
  unfold replayArray
  rw [forward, List.mapM_pure]
  change some ((List.finRange n).map g).toArray = _
  rw [← List.toArray_ofFn, List.finRange, List.map_ofFn]
  rfl

def runMean {ptx : String} {tile : Tile}
    (compiled : ValidationReduceBinding.Certificate ptx tile)
    (scan : ValidationScanBinding.Certificate ptx tile) (native : ValidationCE.Native)
    (ports : Ports) (batch : ValidationCE.ProjectedBatch) (layout : batch.toProjection = ports.projection tile) :
    Option TailNorm.F32 :=
  (replayArray (runBatchRow compiled scan native ports batch layout)).map
    (fun words => ValidationLoss.batchMean ⟨words, 0, batch.device⟩)

/-- The reported local validation batch mean consumes the certified
row outputs in slab order. The input and target IDs are the ones
decoded by the same FineWeb binding used by the driver. -/
theorem decoded_batch_mean_exact {ptx : String} {tile : Tile}
    (compiled : ValidationReduceBinding.Certificate ptx tile)
    (scan : ValidationScanBinding.Certificate ptx tile) (native : ValidationCE.Native)
    (ports : Ports) (inputs : Array UInt32) (targets : Array UInt64)
    (batch : ValidationCE.ProjectedBatch) (bound : (ports.projection tile).bindTokens inputs targets = some batch) :
    let layout := (ValidationCE.bound_tokens_are_source _ _ _ _ bound).1
    runMean compiled scan native ports batch layout =
      some (ValidationLoss.batchMean (ValidationCE.losses native batch)) := by
  dsimp only
  unfold runMean
  rw [replayArray_eq (decoded_batch_row_exact compiled scan native ports batch _)]
  rfl

#axiomsClean decoded_batch_row_exact
#axiomsClean decoded_batch_mean_exact
end VeriTile.Bench.Models.NanoGPT.Track1.ValidationBatchBinding
