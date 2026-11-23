import models.NanoGPT.Track1.HeadActivation
import models.NanoGPT.Track1.SourceAnchors
import models.NanoGPT.Track1.TrainingCE
import models.NanoGPT.Track1.CETranspose
import models.NanoGPT.Track1.SampledDensify
import models.NanoGPT.Track1.MuddBackward
import models.NanoGPT.Track1.MuddActivationStorage
import models.NanoGPT.Track1.MuddResidualStorage

/-! The two autograd head wrappers. Both use the same typed external
matrix multiplication, whose operands, FP8 formats, scales and
accumulation mode are explicit. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.TrainingHead
open VeriTile.Triton.FP
open TrainingCE (F32 Width)
open TrainingCEConversions (launchDoubleToFloat)

abbrev View (m n : Nat) (α : Type) := Fin m → Fin n → α

inductive InputType where
  | e4m3fn
  | e5m2

structure Product (m k n : Nat) where
  /-- The left operand has row-major storage and the right has column-major storage. -/
  left : View m k UInt8
  right : View k n UInt8
  leftType : InputType
  rightType : InputType
  scaleA : F32
  scaleB : F32
  fastAccumulation : Bool

structure Backend where
  logit : {m k n : Nat} → Product m k n → View m n UInt8
  gradient : {m k n : Nat} → Product m k n → View m n (Value .bf16)

def activationScale : F32 := launchDoubleToFloat Driver.pinnedCEHeadActivationBits
def activationReciprocal : F32 := launchDoubleToFloat Driver.pinnedCEHeadActivationReciprocalBits
def weightScale : F32 := launchDoubleToFloat Driver.pinnedCEHeadWeightBits

theorem activation_reciprocal_eq : activationReciprocal = HeadActivation.gain := by decide +kernel

def columnMajor {m n : Nat} (buffer : Nat → UInt8) : View m n UInt8 :=
  fun i j => buffer (j.val * m + i.val)

inductive WeightTarget (w : Width) where
  | fullVocab (same : w = .full)
  | sampled (c : SampledRows.Count) (columns : w.columns = c.rows) (positions : Densify.PositionMap c)

set_option maxRecDepth 4096 in
structure Input (n : Nat) (w : Width) where
  hidden : Fin (n * 768) → Value .bf16
  activationFast : Bool
  weightColumns : Nat → UInt8
  weightRows : Nat → UInt8
  weightTarget : WeightTarget w
  parameters : TrainingCE.Parameters
  predictions : Fin 4
  weights : Nat → F32
  targets : Fin n → Fin w.columns
  prefixTargets : Fin n → Option (Fin w.columns)
  encoded : Option (View n 768 UInt8) := none
  residual : Option (MuddResidualStorage.Packet n) := none

structure Saved (n : Nat) (w : Width) where
  activation : View n 768 UInt8
  gradient : View n w.columns UInt8
  weightRows : Nat → UInt8
  weightTarget : WeightTarget w
  gradientScale : F32
  norm : Option (MuddNorm.Saved n) := none
  residualMemory : Option MuddResidualStorage.Memory := none

structure ForwardOutput (n : Nat) (w : Width) where
  loss : Fin n → F32
  saved : Saved n w

abbrev Forward := (tokens : Nat) → (width : Width) → Backend → TrainingCE.Native →
  Input tokens width → ForwardOutput tokens width

def forward (quantize : HeadActivation.Run) (runCE : TrainingCE.Run)
    (runResidual : MuddResidualStorage.Run := MuddResidualStorage.Program.reload.run) : Forward :=
    fun n w gemm native input =>
  let residual := if input.encoded.isSome then none else input.residual.map (runResidual n)
  let activation := match input.encoded, residual with
    | some activation, _ => activation
    | none, some output => output.activation
    | none, none =>
      let quantized := quantize (n * 768) input.activationFast activationReciprocal input.hidden
      fun (row : Fin n) (col : Fin 768) =>
        quantized ⟨row.val * 768 + col.val, by have a := row.isLt; have b := col.isLt; omega⟩
  let logits := gemm.logit ⟨activation, columnMajor input.weightColumns, .e4m3fn, .e4m3fn,
    activationScale, weightScale, true⟩
  let parameters := { input.parameters with gradScaleParam := Driver.pinnedCEHeadGradBits }
  let rows := fun i => runCE w native parameters
    ⟨n, i, input.predictions, input.targets, input.prefixTargets i, input.weights, logits i⟩
  ⟨fun i => (rows i).loss, ⟨activation, fun i => (rows i).gradient,
    input.weightRows, input.weightTarget, parameters.gradScale, none,
    residual.map MuddResidualStorage.Output.stored⟩⟩

private def rowPacked {m n : Nat} (matrix : View m n UInt8) (offset : Nat) : UInt8 :=
  if inRow : offset / n < m then
    if inCol : offset % n < n then matrix ⟨_, inRow⟩ ⟨_, inCol⟩ else 0
  else 0

theorem row_packed_exact {m n : Nat} (matrix : View m n UInt8) (row : Fin m) (col : Fin n) :
    rowPacked matrix (row.val * n + col.val) = matrix row col := by
  have nz : 0 < n := Nat.zero_lt_of_lt col.isLt
  unfold rowPacked
  rw [Nat.mul_comm row.val n, Nat.mul_add_div nz, Nat.div_eq_of_lt col.isLt, Nat.add_zero,
    Nat.mul_add_mod, Nat.mod_eq_of_lt col.isLt]
  simp [row.isLt, col.isLt]

def transposedByte (transpose : CETranspose.Stores) (shape : CETranspose.Shape)
    (src : CETranspose.Operand → Nat → UInt8) (operand : CETranspose.Operand)
    (r : Fin (shape.operand operand).rows) (c : Fin (shape.operand operand).cols) : UInt8 :=
  let address := c.val * (shape.operand operand).rows + r.val
  ((transpose shape src ⟨operand, ByteTranspose.point _ r c⟩).bind fun (port, dst, data) =>
    if port = operand ∧ dst = address then some data else none).getD 0

theorem transposed_byte_exact (transpose : CETranspose.Stores) (stores : transpose = CETranspose.staged)
    (shape : CETranspose.Shape) (src : CETranspose.Operand → Nat → UInt8) (operand : CETranspose.Operand)
    (r : Fin (shape.operand operand).rows) (c : Fin (shape.operand operand).cols) :
    transposedByte transpose shape src operand r c = src operand (r.val * (shape.operand operand).cols + c.val) := by
  rw [transposedByte, stores, CETranspose.staged_operand]
  simp

private def denseWeights {w : Width} (dispatch : Densify.Dispatch) (fast : Bool)
    (target : WeightTarget w) (compact : View 768 w.columns (Value .bf16)) :
    View 768 50304 (Value .bf16) :=
  match target with
  | .fullVocab same => fun r c => compact r (c.cast (by rw [same]; rfl))
  | .sampled count cols positions => fun r c =>
    let read := fun (byte : Nat) =>
      let row := byte / 2 / count.rows
      let slot := byte / 2 % count.rows
      if hr : row < 768 then
        if hs : slot < count.rows then
          UInt16.ofNat (compact ⟨row, hr⟩ ⟨slot, cols.symm ▸ hs⟩).raw
        else 0
      else 0
    Value.ofBits .bf16 (dispatch fast count read positions (Densify.point r c)).2.toNat

structure Gradients (n : Nat) where
  activation : View n 768 (Value .bf16)
  weight : View 768 50304 (Value .bf16)

abbrev Backward := (w : Width) → Backend → Bool → Bool → (shape : CETranspose.Shape) →
  (shape.operand .gradient).cols = w.columns → Saved (shape.operand .gradient).rows w →
  Gradients (shape.operand .gradient).rows

def backwardWithNorm (transpose : CETranspose.Dispatch) (densify : Densify.Dispatch)
    (norm : MuddBackward.BatchRun) (readActivation : MuddActivationStorage.Read) : Backward :=
    fun w gemm transposeFast densifyFast shape columns saved =>
  let dx := gemm.gradient ⟨saved.gradient, columnMajor saved.weightRows, .e5m2, .e4m3fn,
    saved.gradientScale, weightScale, false⟩
  let (residualGrad, activation) := match saved.norm with
    | some packet => norm (shape.operand .gradient).rows packet dx saved.activation
    | none => (dx, saved.activation)
  let src := fun operand =>
    match operand with
    | .activation => rowPacked (readActivation (shape.operand .gradient).rows activation)
    | .gradient => rowPacked saved.gradient
  let activationT := fun (c : Fin 768) (r : Fin (shape.operand .gradient).rows) =>
    transposedByte (transpose transposeFast) shape src .activation r c
  let gradientTView := fun r (c : Fin w.columns) =>
    transposedByte (transpose transposeFast) shape src .gradient r (c.cast columns.symm)
  let dw := gemm.gradient ⟨activationT, gradientTView, .e4m3fn, .e5m2,
    activationScale, saved.gradientScale, false⟩
  ⟨residualGrad, denseWeights densify densifyFast saved.weightTarget dw⟩

def backward (transpose : CETranspose.Dispatch) (densify : Densify.Dispatch) : Backward :=
  backwardWithNorm transpose densify (MuddBackward.execute MuddBackward.backward)
    MuddActivationStorage.Program.duplicate.read

structure ForwardPacket where
  tokens : Nat
  width : Width
  gemm : Backend
  native : TrainingCE.Native
  input : Input tokens width
  norm : Option (MuddNorm.Saved tokens) := none

def ForwardPacket.run (packet : ForwardPacket) (f : Forward) : ForwardOutput packet.tokens packet.width :=
  f packet.tokens packet.width packet.gemm packet.native packet.input

structure Tape where
  tokens : Nat
  width : Width
  gemm : Backend
  saved : Saved tokens width

def ForwardPacket.tape (packet : ForwardPacket) (saved : Saved packet.tokens packet.width) : Tape :=
  ⟨packet.tokens, packet.width, packet.gemm, {saved with norm := packet.norm}⟩

structure BackwardLayout (tape : Tape) where
  transposeFast : Bool
  densifyFast : Bool
  shape : CETranspose.Shape
  rows : (shape.operand .gradient).rows = tape.tokens
  columns : (shape.operand .gradient).cols = tape.width.columns

def BackwardLayout.run {tape : Tape} (layout : BackwardLayout tape) (f : Backward) : Gradients tape.tokens :=
  let saved : Saved (layout.shape.operand .gradient).rows tape.width := layout.rows.symm ▸ tape.saved
  layout.rows ▸ f tape.width tape.gemm layout.transposeFast layout.densifyFast layout.shape layout.columns saved

#axiomsClean row_packed_exact
#axiomsClean activation_reciprocal_eq
#axiomsClean transposed_byte_exact
end VeriTile.Bench.Models.NanoGPT.Track1.TrainingHead
