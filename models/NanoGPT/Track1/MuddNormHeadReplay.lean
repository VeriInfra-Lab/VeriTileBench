import VeriTile.Triton.Float.ScalarOpsLaws
import models.NanoGPT.Track1.MuddNorm

/-! The head backward can reconstruct the FP8 activations from the
saved FP32 mixed residual and square sum. The XOR reduction returns the
same rounded sum in every lane under the IEEE value semantics. -/

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddNorm
open VeriTile.Triton.FP

private def add32 := @add .fp32 Config.ieeeValue
private def mul32 := @mul .fp32 Config.ieeeValue

private def step (values : Nat → F32) (stride lane : Nat) : F32 :=
  add32 (values lane) (values (lane ^^^ stride))

private def Invariant (values : Nat → F32) (offset : Nat) : Prop :=
  ∀ lane, values (lane ^^^ offset) = values lane

private theorem step_new (values : Nat → F32) (stride : Nat) :
    Invariant (step values stride) stride := by
  intro lane
  simp only [step, Nat.xor_assoc, Nat.xor_self, Nat.xor_zero]
  exact ScalarOpsLaws.add_comm _ _ _

private theorem step_keeps {values : Nat → F32} {offset : Nat}
    (h : Invariant values offset) (stride : Nat) :
    Invariant (step values stride) offset := by
  intro lane
  unfold step
  have exchange : (lane ^^^ offset) ^^^ stride = (lane ^^^ stride) ^^^ offset := by
    calc
      (lane ^^^ offset) ^^^ stride = lane ^^^ (offset ^^^ stride) := Nat.xor_assoc ..
      _ = lane ^^^ (stride ^^^ offset) := by rw [Nat.xor_comm offset stride]
      _ = (lane ^^^ stride) ^^^ offset := (Nat.xor_assoc ..).symm
  rw [exchange, h lane, h (lane ^^^ stride)]

private theorem keeps_all (strides : List Nat) {values : Nat → F32} {offset : Nat}
    (h : Invariant values offset) : Invariant (strides.foldl step values) offset := by
  induction strides generalizing values with
  | nil => exact h
  | cons s tail ih => exact ih (step_keeps h s)

private theorem xor_reduce_invariant (strides : List Nat) (values : Nat → F32) (offset : Nat)
    (member : offset ∈ strides) : Invariant (xorReduce strides values) offset := by
  change Invariant (strides.foldl step values) offset
  induction strides generalizing values with
  | nil => contradiction
  | cons stride more ih =>
    rcases List.mem_cons.1 member with same | tail
    · subst offset
      exact keeps_all more (step_new values stride)
    · exact ih (step values stride) tail

private def masks : List Nat := [16, 8, 4, 2, 1]

private theorem clears_lane (lane : Fin 32) :
    masks.foldl (fun l mask => l ^^^ (l &&& mask)) lane.val = 0 := by
  have checked : ∀ l : Fin 32, masks.foldl (fun l mask => l ^^^ (l &&& mask)) l.val = 0 := by
    decide +kernel
  exact checked lane

private theorem clear_preserves (f : Nat → F32) (strides : List Nat)
    (hn : ∀ mask ∈ strides, ∀ l, l &&& mask = 0 ∨ l &&& mask = mask)
    (hi : ∀ mask ∈ strides, Invariant f mask) (lane : Nat) :
    f (strides.foldl (fun l mask => l ^^^ (l &&& mask)) lane) = f lane := by
  induction strides generalizing lane with
  | nil => rfl
  | cons mask rest ih =>
    simp only [List.foldl_cons]
    rw [ih (fun m hm => hn m (List.mem_cons_of_mem _ hm))
      (fun m hm => hi m (List.mem_cons_of_mem _ hm))]
    rcases hn mask (by simp) lane with zero | same
    · rw [zero, Nat.xor_zero]
    · rw [same, hi mask (by simp) lane]

private theorem stride_bit (mask : Nat) (belongs : mask ∈ masks) (lane : Nat) :
    lane &&& mask = 0 ∨ lane &&& mask = mask := by
  have size : mask < 2 ^ 5 := by
    simp [masks] at belongs
    omega
  have bound := Nat.and_lt_two_pow lane size
  have checked : ∀ (word mask : Fin 32), mask.val ∈ masks →
      word.val &&& mask.val = 0 ∨ word.val &&& mask.val = mask.val := by decide +kernel
  rw [← Nat.mod_eq_of_lt bound, Nat.and_mod_two_pow, Nat.mod_eq_of_lt size]
  exact checked ⟨lane % 32, Nat.mod_lt _ (by decide)⟩ ⟨mask, size⟩ belongs

theorem warp_sum_uniform (seed : Nat → F32) (lane : Fin 32) :
    xorReduce masks seed lane.val = xorReduce masks seed 0 := by
  let f := xorReduce masks seed
  have h := clear_preserves f masks stride_bit (fun mask h => xor_reduce_invariant masks seed mask h) lane.val
  rw [clears_lane lane] at h
  exact h.symm

theorem reduction_uniform (mixed : Fin 32 → Fin 32 → F32) (lane : Fin 32) :
    reduction mixed lane = reduction mixed 0 := by
  unfold reduction
  exact warp_sum_uniform _ lane

theorem mul_operands_commute32 (x y : F32) : mul32 x y = mul32 y x := by
  have signs : ∀ a b : Bool, (a != b) = (b != a) := by decide +kernel
  unfold mul32 mul
  cases (prepare Config.ieeeValue x).decode <;> cases (prepare Config.ieeeValue y).decode <;>
    simp only [signs, Rat.mul_comm]

def encodedFromSaved (native : Native) (output : Output) (column : Fin 768) : UInt8 :=
  let mean := native.divFull output.sumSquares (Value.ofBits .fp32 0x44400000)
  let inverse := native.rsqrtFtz (add32 mean (Value.ofBits .fp32 0x34000000))
  native.e4m3 (mul32 (mul32 inverse (output.mixedResidual column)) (Value.ofBits .fp32 0x408f5c29))

theorem head_recomputation_eq (layout : Layout) (native : Native) (input : Input) :
    encodedFromSaved native (run layout native input) = (run layout native input).encoded := by
  funext column
  unfold encodedFromSaved run
  dsimp only
  rw [reduction_uniform (mix layout input) (logicalLane column)]
  exact congrArg (fun x => native.e4m3 (mul32 x (Value.ofBits .fp32 0x408f5c29)))
    (mul_operands_commute32 _ _)

def BatchOutput.reencoded {shape : MuddCoefficients.Shape} (output : BatchOutput shape)
    (native : Native) (row : Fin shape.rows) : Fin 768 → UInt8 :=
  encodedFromSaved native ⟨output.mixedResidual row, output.sumSquares row, output.activations row⟩

theorem RowProgram.reencoded_eq (program : RowProgram) (native : Native) (packet : Packet) :
    (program.batch native packet).reencoded native = (program.batch native packet).activations := by
  funext row column
  cases program with
  | staged =>
    exact congrFun (head_recomputation_eq stagedGroup native
      ⟨(packet.row row).residual, (packet.row row).sources, (packet.row row).referenceCoefficients⟩) column
  | combined =>
    exact congrFun (head_recomputation_eq directGroup native
      ⟨(packet.row row).residual, (packet.row row).sources, (packet.row row).combinedCoefficients⟩) column

theorem RowProgram.consistent (program : RowProgram) : program.batch.Consistent :=
  fun native packet => (program.reencoded_eq native packet).symm

def RowProgram.training (program : RowProgram) (statistic : Statistic) : TrainingRun :=
  program.batch.training statistic program.consistent

theorem training_run_eq :
    RowProgram.combined.training .inverse = RowProgram.staged.training .squareSum := by
  exact (cached_store_load _ _).trans (Run.training_congr RowProgram.batch_eq .squareSum _ _)

theorem RowProgram.training_activation (program : RowProgram) (statistic : Statistic)
    (native : Native) (packet : Packet) (layout : BackwardLayout) :
    ((program.training statistic native packet).saved layout).activation =
      (program.training statistic native packet).activations := rfl

#axiomsClean RowProgram.reencoded_eq
#axiomsClean mul_operands_commute32
#axiomsClean warp_sum_uniform
#axiomsClean reduction_uniform
#axiomsClean head_recomputation_eq
#axiomsClean RowProgram.consistent
#axiomsClean training_run_eq
#axiomsClean RowProgram.training_activation

end VeriTile.Bench.Models.NanoGPT.Track1.MuddNorm
