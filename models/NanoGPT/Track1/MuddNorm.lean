import models.NanoGPT.Track1.MuddCoefficients
import Mathlib.Data.Fin.Basic
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

/-! The post-MUDD residual mix and its compiled RMS reduction.
The default CUDA launch stages each coefficient plane through shared
memory using ldmatrix. The fused layout computes it at the consuming lane. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddNorm
open VeriTile.Triton.FP
open VeriTile.Spec
open scoped VeriTile.Spec

abbrev F32 := Value .fp32
abbrev BF16 := Value .bf16

def channel (lane word : Nat) : Nat :=
  16 * lane + word % 16 + 512 * (word / 16)

def writeOffset (lane word : Nat) : Nat :=
  let base := (((lane <<< 4) &&& 240) ^^^ ((lane <<< 1) &&& 32)) ||| ((lane <<< 7) &&& 768)
  (base ^^^ (64 * (word % 2))) + 2 * ((word % 16) / 2)

def readOffset (lane word : Nat) : Nat :=
  let addressLane := 8 * ((word % 8) / 2) + 2 * (lane % 4) + word % 2
  let bank := addressLane &&& 24
  let offset := (((bank <<< 5) ||| ((addressLane <<< 4) &&& 112)) ^^^ (bank <<< 2))
  offset + 128 * ((word % 16) / 8) + 2 * (lane / 4)

def storeLane (address : Nat) : Nat :=
  let lo := address / 16
  let hi := address / 256
  (lo &&& 9) ||| ((hi &&& 3) <<< 1) ||| (((lo >>> 1 ^^^ hi) &&& 1) <<< 4)

def storeWord (address : Nat) : Nat :=
  2 * (address / 2 % 8) + ((address >>> 6 ^^^ address >>> 9) &&& 1)

def stagedGroupIndex (lane word : Nat) : Nat :=
  let address := readOffset lane word
  (storeLane address + 32 * (storeWord address + 16 * (word / 16))) / 64

theorem writer_address_roundtrip (lane : Fin 32) (word : Fin 16) :
    let address := writeOffset lane.val word.val
    storeLane address = lane.val ∧ storeWord address = word.val := by
  have checked : ∀ (lane : Fin 32) (word : Fin 16),
      let address := writeOffset lane.val word.val
      storeLane address = lane.val ∧ storeWord address = word.val := by decide +kernel
  exact checked lane word

theorem reader_address_roundtrip (lane : Fin 32) (word : Fin 32) :
    let address := readOffset lane.val word.val
    writeOffset (storeLane address) (storeWord address) = address := by
  have checked : ∀ (lane : Fin 32) (word : Fin 32),
      let address := readOffset lane.val word.val
      writeOffset (storeLane address) (storeWord address) = address := by decide +kernel
  exact checked lane word

theorem staged_group_exact (lane word : Fin 32) :
    stagedGroupIndex lane.val word.val = channel lane.val word.val / 64 := by
  have checked : ∀ lane word : Fin 32,
      stagedGroupIndex lane.val word.val = channel lane.val word.val / 64 := by decide +kernel
  exact checked lane word

structure Input where
  residual : Fin 768 → BF16
  sources : Fin 10 → Fin 768 → BF16
  coefficients : Fin 10 → Fin 12 → BF16

structure Native where
  divFull : F32 → F32 → F32
  rsqrtFtz : F32 → F32
  e4m3 : F32 → UInt8

private def zero32 : F32 := Value.zero .fp32
private def extend (x : BF16) : F32 := cast .fp32 Config.ieeeValue x
private def add32 := @add .fp32 Config.ieeeValue
private def mul32 := @mul .fp32 Config.ieeeValue
private def fma32 := @fma .fp32 Config.ieeeValue

def groupWeight (input : Input) (plane : Fin 10) (group : Nat) : BF16 :=
  if h : group < 12 then input.coefficients plane ⟨group, h⟩ else Value.zero .bf16

abbrev Layout := Fin 32 → Fin 32 → Nat

def directGroup : Layout := fun lane word => channel lane.val word.val / 64
def stagedGroup : Layout := fun tid word => stagedGroupIndex tid.val word.val

theorem layout_eq : stagedGroup = directGroup := by
  funext tid word
  exact staged_group_exact tid word

def mix (layout : Layout) (input : Input) (tid word : Fin 32) : F32 :=
  let col := channel tid.val word.val
  if h : col < 768 then
    (List.finRange 10).foldl (fun acc plane =>
      let coefficient := extend (groupWeight input plane (layout tid word))
      fma32 coefficient (extend (input.sources plane ⟨col, h⟩)) acc)
      (extend (input.residual ⟨col, h⟩))
  else zero32

private def mappedWord (word : Fin n) (base : Nat) (h : n + base ≤ 32) : Fin 32 :=
  ⟨word.val + base, by have := word.isLt; omega⟩

/-- The first 16 words use the multiply/FMA sequence of the CUDA launch.
The second 16 square separately because of the channel mask. -/
def squaredPartial (lane : Fin 32) (x : Fin 32 → F32) : F32 :=
  let init := mul32 (x 1) (x 1)
  let first := ((List.finRange 16).erase 1).foldl (fun acc word =>
    let value := x (mappedWord word 0 (by decide))
    fma32 value value acc) init
  (List.finRange 16).foldl (fun acc word =>
    let value := x (mappedWord word 16 (by decide))
    let square := if lane.val < 16 then mul32 value value else zero32
    add32 square acc) first

def xorReduce (strides : List Nat) (leaf : Nat → F32) : Nat → F32 :=
  strides.foldl (fun accum stride lane => add32 (accum lane) (accum (lane ^^^ stride))) leaf

def reduction (mixed : Fin 32 → Fin 32 → F32) (lane : Fin 32) : F32 :=
  xorReduce [16, 8, 4, 2, 1] (fun i =>
    if h : i < 32 then squaredPartial ⟨i, h⟩ (mixed ⟨i, h⟩) else zero32) lane.val

def logicalLane (col : Fin 768) : Fin 32 :=
  ⟨col.val % 512 / 16, by have h := Nat.mod_lt col.val (by decide : 0 < 512); omega⟩

def logicalWord (col : Fin 768) : Fin 32 :=
  ⟨col.val % 16 + 16 * (col.val / 512), by
    have h₁ := col.isLt
    have h₂ := Nat.mod_lt col.val (by decide : 0 < 16)
    omega⟩

theorem channel_exact (col : Fin 768) :
    channel (logicalLane col).val (logicalWord col).val = col.val := by
  have checked : ∀ col : Fin 768, channel (logicalLane col).val (logicalWord col).val = col.val := by
    decide +kernel
  exact checked col

theorem channel_inverse (lane word : Fin 32) (live : channel lane.val word.val < 768) :
    logicalLane ⟨channel lane.val word.val, live⟩ = lane ∧
    logicalWord ⟨channel lane.val word.val, live⟩ = word := by
  have checked : ∀ (lane word : Fin 32) (live : channel lane.val word.val < 768),
      logicalLane ⟨channel lane.val word.val, live⟩ = lane ∧
      logicalWord ⟨channel lane.val word.val, live⟩ = word := by decide +kernel
  exact checked lane word live

def coefficientColumn (group : Fin 12) : Fin 768 :=
  ⟨64 * group.val, by have := group.isLt; omega⟩

/-- Masking the coefficient store to channel multiples of 64 gives
one owner in each group, even when several lanes consume the value. -/
theorem coefficient_owner (group : Fin 12) :
    ∃! pair : Fin 32 × Fin 32, channel pair.1.val pair.2.val = 64 * group.val := by
  let col := coefficientColumn group
  refine ⟨(logicalLane col, logicalWord col), channel_exact col, ?_⟩
  rintro ⟨lane, word⟩ same
  have live : channel lane.val word.val < 768 := by
    rw [same]
    exact col.isLt
  have own := channel_inverse lane word live
  simp only [same] at own
  exact Prod.ext own.1.symm own.2.symm

theorem launch_indices_fit (shape : MuddCoefficients.Shape) (row : Fin shape.rows) :
    row.val * 768 + 767 < 2 ^ 31 ∧
    row.val * 120 + 119 < 2 ^ 31 ∧
    row.val * 12 + 11 < 2 ^ 31 := by
  have bound : shape.rows ≤ 49152 := by cases shape <;> decide
  have index := row.isLt
  omega

structure Output where
  mixedResidual : Fin 768 → F32
  sumSquares : F32
  encoded : Fin 768 → UInt8

def inverseRMS (native : Native) (squares : F32) : F32 :=
  native.rsqrtFtz (add32 (native.divFull squares (Value.ofBits .fp32 0x44400000))
    (Value.ofBits .fp32 0x34000000))

def encodeWithInverse (native : Native) (inverse mixed : F32) : UInt8 :=
  native.e4m3 (mul32 (mul32 inverse mixed) (Value.ofBits .fp32 0x408f5c29))

def run (layout : Layout) (native : Native) (input : Input) : Output :=
  let mixed := mix layout input
  let sumSquares := reduction mixed
  let rstd := fun lane => inverseRMS native (sumSquares lane)
  { mixedResidual := fun col => mixed (logicalLane col) (logicalWord col)
    sumSquares := sumSquares 0
    encoded := fun col =>
      let lane := logicalLane col
      let word := logicalWord col
      native.e4m3 (mul32 (mul32 (mixed lane word) (rstd lane)) (Value.ofBits .fp32 0x408f5c29)) }

inductive Program where
  | shared | direct

def Program.layout : Program → Layout
  | .shared => stagedGroup
  | .direct => directGroup

def sameOutput (left right : Program) : Prop :=
  ∀ native input, run left.layout native input = run right.layout native input

instance : Spec.ProgramSyntax Program where
  Statement := Unit
  Signature := Unit
  signature := fun _ => ()
  body := fun _ => []
  structural := some sameOutput

specification direct_equal : Program.direct ≡[([] : Spec.Assumptions Unit)] Program.shared := by
  refine Spec.FloatingPoint.ofStructural (structural := sameOutput) rfl rfl ?_
  intro native input
  change run directGroup native input = run stagedGroup native input
  rw [layout_eq]

structure RowPacket where
  shape : MuddCoefficients.Shape
  row : Fin shape.rows
  projection : MuddCoefficients.Inputs
  residual : Fin 768 → BF16
  sources : Fin 10 → Fin 768 → BF16

abbrev RowCoefficients := Fin 10 → Fin 12 → BF16

def RowPacket.bankIndex (p : RowPacket) (group : Fin 12) : Fin p.shape.plane :=
  ⟨p.row.val * 12 + group.val, by
    have row := p.row.isLt
    have col := group.isLt
    unfold MuddCoefficients.Shape.plane
    omega⟩

def RowPacket.referenceCoefficients (p : RowPacket) : RowCoefficients :=
  fun plane group => MuddCoefficients.reference p.shape p.projection plane (p.bankIndex group)

def RowPacket.combinedCoefficients (p : RowPacket) : RowCoefficients :=
  fun plane group =>
    let cf := p.projection
    MuddCoefficients.applyCoefficient (cf.bias (14 + plane.val)) (cf.mu (p.row.val * 10 + plane.val))
      (cf.delta (p.row.val * 120 + plane.val * 12 + group.val))

theorem RowPacket.coefficients_eq (p : RowPacket) : p.combinedCoefficients = p.referenceCoefficients := by
  funext plane group
  have h := group.isLt
  unfold RowPacket.combinedCoefficients RowPacket.referenceCoefficients
    MuddCoefficients.reference MuddCoefficients.referencePlane MuddCoefficients.coefficientAt
  simp [RowPacket.bankIndex, Nat.mul_comm p.row.val 12, Nat.mul_add_div,
    Nat.div_eq_of_lt h, Nat.mod_eq_of_lt h]

structure RowOutput where
  coefficients : RowCoefficients
  head : Output

inductive RowProgram where
  | staged | combined

def RowProgram.run (program : RowProgram) (native : Native) (p : RowPacket) : RowOutput :=
  let coef := match program with
    | .staged => p.referenceCoefficients
    | .combined => p.combinedCoefficients
  let layout := match program with | .staged => stagedGroup | .combined => directGroup
  ⟨coef, MuddNorm.run layout native ⟨p.residual, p.sources, coef⟩⟩

def sameRow (left right : RowProgram) : Prop :=
  ∀ native input, left.run native input = right.run native input

theorem combined_run_eq : sameRow .combined .staged := by
  intro native input
  simp only [RowProgram.run, input.coefficients_eq, layout_eq]

structure Packet where
  shape : MuddCoefficients.Shape
  projection : MuddCoefficients.Inputs
  residual : Fin shape.rows → Fin 768 → BF16
  sources : Fin 10 → Fin shape.rows → Fin 768 → BF16

def Packet.row (packet : Packet) (r : Fin packet.shape.rows) : RowPacket :=
  ⟨packet.shape, r, packet.projection, packet.residual r, fun plane => packet.sources plane r⟩

structure BatchOutput (shape : MuddCoefficients.Shape) where
  coefficients : MuddCoefficients.Bank shape
  mixedResidual : Fin shape.rows → Fin 768 → F32
  sumSquares : Fin shape.rows → F32
  activations : Fin shape.rows → Fin 768 → UInt8

abbrev Run := Native → (p : Packet) → BatchOutput p.shape

def batchOf (runRow : Native → RowPacket → RowOutput) : Run := fun native packet =>
  let result := fun row => runRow native (packet.row row)
  let coefficients := fun plane (index : Fin packet.shape.plane) =>
    let row : Fin packet.shape.rows := ⟨index.val / 12, by
      have h := index.isLt
      unfold MuddCoefficients.Shape.plane at h
      omega⟩
    let group : Fin 12 := ⟨index.val % 12, Nat.mod_lt _ (by decide)⟩
    (result row).coefficients plane group
  ⟨coefficients, fun row => (result row).head.mixedResidual,
    fun row => (result row).head.sumSquares, fun row => (result row).head.encoded⟩

def RowProgram.batch (program : RowProgram) : Run := batchOf program.run

theorem RowProgram.batch_eq : RowProgram.combined.batch = RowProgram.staged.batch := by
  funext native packet
  have row : (fun r => RowProgram.combined.run native (packet.row r)) =
      fun r => RowProgram.staged.run native (packet.row r) := by
    funext r
    exact combined_run_eq native (packet.row r)
  unfold RowProgram.batch batchOf
  rw [row]

inductive BackwardLayout where
  | withQuantization
  | gradientOnly

structure Saved (rows : Nat) where
  layout : BackwardLayout
  native : Native
  mixedResidual : Fin rows → Fin 768 → F32
  inverse : Fin rows → F32
  activation : Fin rows → Fin 768 → UInt8
  activation_eq : activation = fun row col => encodeWithInverse native (inverse row) (mixedResidual row col)

inductive Statistic where
  | squareSum
  | inverse

def Statistic.store (statistic : Statistic) (native : Native) (squares : F32) : F32 :=
  match statistic with
  | .squareSum => squares
  | .inverse => inverseRMS native squares

def Statistic.load (statistic : Statistic) (native : Native) (word : F32) : F32 :=
  match statistic with
  | .squareSum => inverseRMS native word
  | .inverse => word

structure TrainingOutput (shape : MuddCoefficients.Shape) where
  coefficients : MuddCoefficients.Bank shape
  saved : BackwardLayout → Saved shape.rows
  activations : Fin shape.rows → Fin 768 → UInt8

abbrev TrainingRun := Native → (p : Packet) → TrainingOutput p.shape

/-- The native forward encodes the same FP32 values available to the head
backward in its saved operands. -/
def Run.Consistent (run : Run) : Prop :=
  ∀ native packet, let output := run native packet
    output.activations = fun row col =>
      encodeWithInverse native (inverseRMS native (output.sumSquares row)) (output.mixedResidual row col)

/-- The scratch allocation has a single consumer. The logical output carries
the FP32 value this consumer uses after decoding the scratch word. -/
def Run.training (run : Run) (statistic : Statistic) (consistent : run.Consistent) : TrainingRun := fun native packet =>
  let output := run native packet
  let stored := fun row => statistic.store native (output.sumSquares row)
  ⟨output.coefficients, fun layout =>
    ⟨layout, native, output.mixedResidual, fun row => statistic.load native (stored row), output.activations,
      by cases statistic <;> exact consistent native packet⟩,
    output.activations⟩

theorem cached_store_load (run : Run) (consistent : run.Consistent) :
    run.training .inverse consistent = run.training .squareSum consistent := rfl

theorem Run.training_congr {left right : Run} (h : left = right) (statistic : Statistic)
    (hl : left.Consistent) (hr : right.Consistent) :
    left.training statistic hl = right.training statistic hr := by
  subst right
  rfl

def sameBatch (left right : RowProgram) : Prop :=
  ∀ native packet, left.batch native packet = right.batch native packet

instance : Spec.ProgramSyntax RowProgram where
  Statement := Unit
  Signature := Unit
  signature := fun _ => ()
  body := fun _ => []
  structural := some sameBatch

specification combined_equal : RowProgram.combined ≡[([] : Spec.Assumptions Unit)] RowProgram.staged := by
  refine Spec.FloatingPoint.ofStructural (structural := sameBatch) rfl rfl ?_
  intro native packet
  rw [RowProgram.batch_eq]

#axiomsClean RowProgram.batch_eq
#axiomsClean cached_store_load
#axiomsClean Run.training_congr
#axiomsClean RowPacket.coefficients_eq
#axiomsClean combined_equal
#print_fp_assumptions combined_equal

#axiomsClean channel_exact
#axiomsClean coefficient_owner
#axiomsClean launch_indices_fit
#axiomsClean writer_address_roundtrip
#axiomsClean reader_address_roundtrip
#axiomsClean staged_group_exact
#axiomsClean direct_equal
#print_fp_assumptions direct_equal

end VeriTile.Bench.Models.NanoGPT.Track1.MuddNorm
