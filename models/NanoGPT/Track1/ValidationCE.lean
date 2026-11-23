import Init.Data.Array.FinRange
import models.NanoGPT.Track1.ValidationLoss

/-! Numerical validation cross-entropy after the BF16 LM-head projection.
The native interface is at the remaining approximate PTX instructions.
Mask routing, libdevice range reduction and polynomial arithmetic, online
softmax state, and the ordered accumulation use the concrete FP32 model. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValidationCE
open VeriTile.Triton FP
open TailNorm (F32 BF16 add zero one ceilDiv)

structure Native where
  ex2Approx : Bool → F32 → F32
  reciprocalFull : F32 → F32

private def w : Nat → F32 := Value.ofBits .fp32
private def negInf : F32 := Value.infinity .fp32 true
private def cfg := Config.ieeeValue
private def ftz : Config := ⟨.nearestEven, true, true⟩
private def rmFtz : Config := ⟨.towardNegative, true, true⟩

private def sat (word : F32) : F32 :=
  let x := prepare ftz word
  if x.sign || x.isNaN || x.isZero then zero else if FP.lt cfg one x then one else x

/-- The `__nv_expf` sequence inlined into the SM90 kernels.
The exponent scale is formed by shifting the rounded offset's raw bits. -/
def expf (native : Native) (x : F32) : F32 :=
  let rough := FP.fma ftz x (w 0x3bbb989d) (w 0x3f000000)
  let exponent := FP.fma rmFtz (sat rough) (w 0x437c0000) (w 0x4b400001)
  let residual := FP.fma ftz x (w 0x3fb8aa3b) (add exponent (w 0xcb40007f)).neg
  let corrected := FP.fma ftz x (w 0x32a57060) residual
  let scale := w ((UInt32.ofNat exponent.raw <<< 23).toNat)
  FP.mul cfg (native.ex2Approx true corrected) scale

/-- The `__nv_logf` instruction sequence preserves the order of the
subnormal normalization, mantissa polynomial, and final exponent FMA. -/
def logf (x : F32) : F32 :=
  let tiny := FP.lt cfg x (w 0x00800000)
  let normalized := if tiny then FP.mul cfg x (w 0x4b000000) else x
  let adjustment := if tiny then w 0xc1b80000 else zero
  let bits := UInt32.ofNat normalized.raw
  let exponent := (bits + 0xc0d55555) &&& 0xff800000
  let fraction := w (bits - exponent).toNat
  let expFloat := FP.round .fp32 cfg (Rat.ofInt exponent.toInt32.toInt)
  let power := FP.fma ftz expFloat (w 0x34000000) adjustment
  let mantissa := add fraction (w 0xbf800000)
  let polynomial := ([0x3e1039f6, 0xbdf8cdcc, 0x3e0f2955, 0xbe2ad8b9,
      0x3e4ced0b, 0xbe7fff22, 0x3eaaaa78, 0xbf000000] : List Nat).foldl
    (fun acc coefficient => FP.fma ftz acc mantissa (w coefficient)) (w 0xbe055027)
  let correction := FP.fma ftz (FP.mul cfg mantissa polynomial) mantissa mantissa
  let log := if normalized.raw < 0x7f800000 then FP.fma ftz power (w 0x3f317218) correction
    else FP.fma ftz normalized (Value.infinity .fp32) (Value.infinity .fp32)
  if normalized.isZero then negInf else log

/-- Dynamo's captured default lowering keeps the softcap in FP32 after
loading the BF16 projection. The multiply by negative 1/7.5 is an FMA
with a positive-zero addend, then the exponential uses the fast EX2. -/
def softcap (native : Native) (projection : BF16) : F32 :=
  let affine := FP.fma cfg (add (TailNorm.widen projection) (w 0x40a00000)) (w 0xbe088889) zero
  let power := native.ex2Approx false (FP.mul cfg affine (w 0x3fb8aa3b))
  FP.mul cfg (native.reciprocalFull (add power one)) (w 0x41b80000)

def canonicalBit (read : Nat → UInt8) (column : Nat) : Bool :=
  (read (column / 8)).toNat.testBit (column % 8)

def logit (native : Native) (read : Nat → UInt8) (column : Nat) (projection : BF16) : F32 :=
  if canonicalBit read column then w 0xc2700000 else softcap native projection

/-- `triton_helpers.maximum`, including its preference for the left NaN
and the right operand on finite ties. -/
def max (a b : F32) : F32 := if FP.lt cfg b a || a.isNaN then a else b

structure Partial where
  maximum : F32
  sum : F32

def empty : Partial := ⟨negInf, zero⟩

def accumulate (native : Native) (state : Partial) (z : F32) : Partial :=
  let m := max state.maximum z
  let scale := fun old => if m == negInf then one else expf native (FP.sub cfg old m)
  ⟨m, FP.fma cfg state.sum (scale state.maximum) (scale z)⟩

def rescale (native : Native) (m : F32) (state : Partial) : F32 :=
  expf native (if m == negInf then zero else FP.sub cfg state.maximum m)

def finish (target maximum sum : F32) : F32 :=
  add (FP.sub cfg (logf sum) (FP.sub cfg target maximum)) zero

/-- Default Inductor reduction configurations and the two occupancy-driven
variants. The constructor records the row tile and reduction extent. -/
inductive Tile where
  | x1r1024 | x1r512 | x16r32 | x64r64 | x64r4 | x8r512 | x8r256
  deriving BEq, Repr

def Tile.extent : Tile → Nat
  | .x1r1024 => 1024
  | .x1r512 => 512
  | .x16r32 => 32
  | .x64r64 => 64
  | .x64r4 => 4
  | .x8r512 => 512
  | .x8r256 => 256

def Tile.lanes : Tile → Nat
  | .x1r1024 | .x1r512 => 256
  | .x16r32 | .x64r64 => 8
  | .x64r4 => 4
  | .x8r512 | .x8r256 => 64

def Tile.shuffles : Tile → List Nat
  | .x1r1024 | .x1r512 => [16, 8, 4, 2, 1, 128, 64, 32]
  | .x16r32 => [1, 4, 2]
  | .x64r64 => [4, 2, 1]
  | .x64r4 => [2, 1]
  | .x8r512 | .x8r256 => [2, 1, 32, 16, 8, 4]

def Tile.groups (tile : Tile) : Nat := tile.extent / tile.lanes

def scanColumn (tile : Tile) (offset iteration : Nat) : Nat :=
  tile.extent * iteration + offset

/-- Each reduction slot advances by the compiled tile extent. When the
last tile is partial the emitted selects keep the previous state. -/
def scan (native : Native) (tile : Tile) {vocab : Nat}
    (read : Fin vocab → F32) (offset : Nat) : Partial :=
  (List.range (ceilDiv vocab tile.extent)).foldl (fun state iteration =>
    let column := scanColumn tile offset iteration
    if h : column < vocab then accumulate native state (read ⟨column, h⟩)
    else state) empty

def butterfly (tile : Tile) (default : α) (combine : α → α → α)
    (read : Nat → α) : α :=
  let initial := (List.range tile.lanes).toArray.map read
  let reduced := tile.shuffles.foldl (fun words offset =>
    (List.range tile.lanes).toArray.map fun lane =>
      combine (words[lane]?.getD default) (words[lane ^^^ offset]?.getD default)) initial
  reduced[0]?.getD default

structure ReductionOps (α : Type) where
  default : α
  maximum : α → α → α
  add : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  scale : α → α → α

/-- The ordered reduction works on numeric words and on the expression
trees decoded from the hardware. `scale` takes the global and slot maxima. -/
def reduceWith (ops : ReductionOps α) (tile : Tile)
    (readMax readSum : Nat → α) : α × α :=
  let slot := fun lane group => lane + group * tile.lanes
  let m := butterfly tile ops.default ops.maximum fun lane =>
    (List.range (tile.groups - 1)).foldl
      (fun acc group => ops.maximum acc (readMax (slot lane (group + 1)))) (readMax (slot lane 0))
  let total := butterfly tile ops.default ops.add fun lane =>
    let term := fun group => readSum (slot lane group)
    let scale := fun group => ops.scale m (readMax (slot lane group))
    let product := fun group => ops.mul (term group) (scale group)
    if tile.groups = 1 then product 0 else
      let fuse := fun acc group => ops.fma (term group) (scale group) acc
      (List.range (tile.groups - 2)).foldl (fun acc group => fuse acc (group + 2)) (fuse (product 1) 0)
  (m, total)

def reductionOps (native : Native) : ReductionOps F32 :=
  ⟨zero, max, add, FP.mul cfg, FP.fma cfg, fun m a => rescale native m ⟨a, zero⟩⟩

def reduce (native : Native) (tile : Tile) (read : Nat → Partial) : Partial :=
  let (m, s) := reduceWith (reductionOps native) tile (fun i => (read i).maximum) (fun i => (read i).sum)
  ⟨m, s⟩

def rowLoss (native : Native) (tile : Tile) {vocab : Nat}
    (projection : Fin vocab → BF16) (mask : Nat → UInt8) (target : Option (Fin vocab)) : F32 :=
  match target with
  | none => zero
  | some target =>
    let read := fun column : Fin vocab => logit native mask column.val (projection column)
    let states := (List.range tile.extent).toArray.map (scan native tile read)
    let result := reduce native tile (fun offset => states[offset]?.getD empty)
    finish (read target) result.maximum result.sum

def slabRows : Nat := 32768
def slabCount : Nat := 8

def slabRow (slab : Fin slabCount) (row : Fin slabRows) : Nat :=
  slab.val * slabRows + row.val

theorem slab_row_covers (index : Fin (slabRows * slabCount)) :
    ∃ (slab : Fin slabCount) (row : Fin slabRows), slabRow slab row = index.val := by
  have live := index.isLt
  simp only [slabRows, slabCount] at live
  refine ⟨⟨index.val / 32768, by unfold slabCount; omega⟩,
    ⟨index.val % 32768, by unfold slabRows; omega⟩, ?_⟩
  dsimp [slabRow, slabRows]
  omega

theorem slab_row_unique {s t : Fin slabCount} {i j : Fin slabRows}
    (same : slabRow s i = slabRow t j) : s = t ∧ i = j := by
  have il := i.isLt
  have jl := j.isLt
  simp only [Fin.ext_iff, slabRow, slabRows] at *
  omega

structure Projection where
  rows : Nat
  vocab : Nat
  projection : Fin rows → Fin vocab → BF16
  mask : Fin vocab → Nat → UInt8
  /-- All eight slabs share the configuration of the fused reduction launch. -/
  tile : Tile
  device : TailNorm.Device

abbrev channels : Nat := 768
abbrev vocabulary : Nat := 50304
abbrev BF16Matrix (rows cols : Nat) := Fin rows → Fin cols → BF16
abbrev Matmul :=
  BF16Matrix slabRows channels → BF16Matrix channels vocabulary → BF16Matrix slabRows vocabulary

/-- The validation head consumes the last block's activations, the tied
embedding, and the tokenizer's packed vocabulary mask. -/
structure HeadInput where
  multiply : Matmul
  activation : Nat → BF16
  weight : BF16Matrix channels vocabulary
  mask : Nat → UInt8
  device : TailNorm.Device

def HeadInput.projection (head : HeadInput) (tile : Tile) : Projection where
  rows := slabCount * slabRows
  vocab := vocabulary
  projection := fun i column =>
    let slab := i.val / slabRows
    let view := fun (row : Fin slabRows) (channel : Fin channels) =>
      head.activation ((slab * slabRows + row.val) * channels + channel.val)
    head.multiply view head.weight ⟨i.val % slabRows, Nat.mod_lt _ (by decide)⟩ column
  mask := fun token byte => head.mask (token.val * 6288 + byte)
  tile := tile
  device := head.device

inductive BatchInput where
  | head (tile : Tile) (input : HeadInput)
  | projected (projection : Projection)

def BatchInput.projection : BatchInput → Projection
  | .head tile input => input.projection tile
  | .projected p => p

structure ProjectedBatch extends Projection where
  input : Fin rows → Fin vocab
  target : Fin rows → Option (Fin vocab)

/-- The raw int64 value used for PyTorch's ignored cross-entropy target. -/
def ignoredTarget : UInt64 := 0xffffffffffffff9c

/-- Decode the target word from the validation batch. -/
def targetColumn (vocab : Nat) (word : UInt64) : Option (Option (Fin vocab)) :=
  if word == ignoredTarget then some none
  else if bound : word.toNat < vocab then some (some ⟨word.toNat, bound⟩)
  else none

/-- Fill CE's canonical-mask selectors and targets from the uploaded
FineWeb arrays. An out-of-range input or target fails the batch. -/
def Projection.bindTokens (p : Projection) (inputs : Array UInt32) (targets : Array UInt64) :
    Option ProjectedBatch :=
  if shape : p.rows = inputs.size ∧ p.rows = targets.size then
    let input := fun i : Fin p.rows => inputs[i.val]'(Nat.lt_of_lt_of_eq i.isLt shape.1)
    let target := fun i : Fin p.rows => targets[i.val]'(Nat.lt_of_lt_of_eq i.isLt shape.2)
    let decoded := fun i => targetColumn p.vocab (target i)
    let valid := fun i => (input i).toNat < p.vocab ∧ (decoded i).isSome
    if bounds : (Array.finRange p.rows).all (fun i => decide (valid i)) = true then
      have each (i : Fin p.rows) : valid i := by
        have h := Array.all_eq_true.mp bounds i.val (by simp)
        exact of_decide_eq_true (by simpa only [Array.getElem_finRange, Fin.cast_mk] using h)
      some { toProjection := p
             input := fun i => ⟨(input i).toNat, (each i).1⟩
             target := fun i => (decoded i).get (each i).2 }
    else none
  else none

theorem bound_tokens_are_source (p : Projection) (inputs : Array UInt32) (targets : Array UInt64)
    (batch : ProjectedBatch) (loaded : p.bindTokens inputs targets = some batch) :
    batch.toProjection = p ∧ ∀ i : Fin batch.rows,
      some (batch.input i).val = inputs[i.val]?.map UInt32.toNat ∧
      some (batch.target i) = targets[i.val]?.bind (targetColumn batch.vocab) := by
  unfold Projection.bindTokens at loaded
  split at loaded
  · rename_i shape
    simp only at loaded
    split at loaded
    · cases Option.some.inj loaded
      refine ⟨rfl, fun i => ?_⟩
      rw [Array.getElem?_eq_getElem (Nat.lt_of_lt_of_eq i.isLt shape.1),
          Array.getElem?_eq_getElem (Nat.lt_of_lt_of_eq i.isLt shape.2)]
      refine ⟨rfl, ?_⟩
      exact Option.some_get _
    · contradiction
  · contradiction

def losses (native : Native) (batch : ProjectedBatch) : ValidationLoss.Batch :=
  ⟨Array.ofFn (fun row : Fin batch.rows =>
    rowLoss native batch.tile (batch.projection row)
      (batch.mask (batch.input row)) (batch.target row)), 0, batch.device⟩

abbrev BatchRun := Native → BatchInput → Array UInt32 → Array UInt64 → Option F32

def runBatch : BatchRun := fun native input tokens targets => do
  let batch ← input.projection.bindTokens tokens targets
  return ValidationLoss.batchMean (losses native batch)

#axiomsClean bound_tokens_are_source

theorem scan_column_covers (tile : Tile) {vocab column : Nat} (h : column < vocab) :
    ∃ iteration < ceilDiv vocab tile.extent, ∃ offset < tile.extent,
      scanColumn tile offset iteration = column := by
  refine ⟨column / tile.extent, ?_, column % tile.extent, ?_, ?_⟩ <;>
    cases tile <;> simp [Tile.extent, ceilDiv, scanColumn] <;> omega

theorem scan_column_unique (tile : Tile) {a b i j : Nat}
    (ha : a < tile.extent) (hb : b < tile.extent)
    (same : scanColumn tile a i = scanColumn tile b j) : a = b ∧ i = j := by
  cases tile <;> simp [Tile.extent, scanColumn] at * <;> omega

theorem reduction_slot_covers (tile : Tile) {offset : Nat} (h : offset < tile.extent) :
    ∃ lane < tile.lanes, ∃ group < tile.groups, lane + group * tile.lanes = offset := by
  refine ⟨offset % tile.lanes, ?_, offset / tile.lanes, ?_, ?_⟩ <;>
    cases tile <;> simp [Tile.extent, Tile.lanes, Tile.groups] at * <;> omega

theorem reduction_slot_unique (tile : Tile) {a b i j : Nat}
    (ha : a < tile.lanes) (hb : b < tile.lanes)
    (same : a + i * tile.lanes = b + j * tile.lanes) : a = b ∧ i = j := by
  cases tile <;> simp [Tile.lanes] at * <;> omega

theorem shuffle_lane_bound (tile : Tile) {lane offset : Nat}
    (active : lane < tile.lanes) (shuffle : offset ∈ tile.shuffles) :
    lane ^^^ offset < tile.lanes := by
  have bound : offset < tile.lanes := by
    cases tile <;> simp [Tile.lanes, Tile.shuffles] at * <;> omega
  cases tile
  · exact Nat.xor_lt_two_pow (n := 8) active bound
  · exact Nat.xor_lt_two_pow (n := 8) active bound
  · exact Nat.xor_lt_two_pow (n := 3) active bound
  · exact Nat.xor_lt_two_pow (n := 3) active bound
  · exact Nat.xor_lt_two_pow (n := 2) active bound
  · exact Nat.xor_lt_two_pow (n := 6) active bound
  · exact Nat.xor_lt_two_pow (n := 6) active bound

theorem ignored_target (native : Native) (tile : Tile) {vocab : Nat}
    (projection : Fin vocab → BF16) (mask : Nat → UInt8) :
    rowLoss native tile projection mask none = zero := rfl

theorem log_unit : logf one = zero := by decide +kernel

theorem log_zero : logf zero = negInf := by decide +kernel

theorem mask_targets_one_byte (bits : UInt8) (column : Nat) :
    canonicalBit (fun i => if i = column / 8 then bits else 0) column = bits.toNat.testBit (column % 8) := by
  simp [canonicalBit]

theorem dropped_logit (native : Native) (read : Nat → UInt8) (column : Nat) (projection : BF16)
    (drop : canonicalBit read column = true) : logit native read column projection = w 0xc2700000 := by
  simp [logit, drop]

#axiomsClean log_unit
#axiomsClean log_zero
#axiomsClean dropped_logit
#axiomsClean scan_column_covers
#axiomsClean scan_column_unique
#axiomsClean reduction_slot_covers
#axiomsClean reduction_slot_unique
#axiomsClean shuffle_lane_bound
#axiomsClean slab_row_covers
#axiomsClean slab_row_unique
end VeriTile.Bench.Models.NanoGPT.Track1.ValidationCE
