import models.NanoGPT.Track1.MuddNorm
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

/-! The full-graph epilogue saves the mixed residual as BF16 for backward.
When one reduction tile covers the row, the forward can quantize that
same rounded word directly. Only live columns feed the masked FP8 stores. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddResidualStorage
open VeriTile.Spec
open scoped VeriTile.Spec
open VeriTile.Triton.FP

abbrev BF16 := Value .bf16
abbrev F32 := Value .fp32
abbrev Memory := Nat → BF16

structure Packet (rows : Nat) where
  native : MuddNorm.Native
  block : Nat
  mixed : Fin rows → Fin 768 → F32
  sumSquares : Fin rows → F32
  base : Nat
  memory : Memory

structure Output (rows : Nat) where
  activation : Fin rows → Fin 768 → UInt8
  stored : Memory

def byteAddress {rows : Nat} (base : Nat) (row : Fin rows) (col : Fin 768) : Nat :=
  base + 2 * (row.val * 768 + col.val)

def writes {rows : Nat} (packet : Packet rows) (addr : Nat) : Prop :=
  packet.base ≤ addr ∧ (addr - packet.base) % 2 = 0 ∧ (addr - packet.base) / 2 / 768 < rows

instance {rows : Nat} (packet : Packet rows) (addr : Nat) : Decidable (writes packet addr) := by
  unfold writes
  infer_instance

def store {rows : Nat} (packet : Packet rows) : Memory := fun addr =>
  if h : writes packet addr then
    let index := (addr - packet.base) / 2
    cast .bf16 Config.ieeeValue (packet.mixed ⟨index / 768, h.2.2⟩
      ⟨index % 768, Nat.mod_lt _ (by decide)⟩)
  else packet.memory addr

theorem read_written {rows : Nat} (packet : Packet rows) (row : Fin rows) (col : Fin 768) :
    store packet (byteAddress packet.base row col) = cast .bf16 Config.ieeeValue (packet.mixed row col) := by
  have d : (byteAddress packet.base row col - packet.base) / 2 = row.val * 768 + col.val := by
    unfold byteAddress
    omega
  have even : (byteAddress packet.base row col - packet.base) % 2 = 0 := by
    unfold byteAddress
    omega
  have base : packet.base ≤ byteAddress packet.base row col := by
    unfold byteAddress
    omega
  have rowIndex : (row.val * 768 + col.val) / 768 = row.val := by omega
  have colIndex : (row.val * 768 + col.val) % 768 = col.val := by omega
  have live : writes packet (byteAddress packet.base row col) := by
    unfold writes
    rw [d, rowIndex]
    exact ⟨base, even, row.isLt⟩
  unfold store
  rw [dif_pos live]
  have r : (⟨(byteAddress packet.base row col - packet.base) / 2 / 768, live.2.2⟩ : Fin rows) = row := by
    apply Fin.ext
    change (byteAddress packet.base row col - packet.base) / 2 / 768 = row.val
    rw [d, rowIndex]
  have c : (⟨(byteAddress packet.base row col - packet.base) / 2 % 768, Nat.mod_lt _ (by decide)⟩ :
      Fin 768) = col := by
    apply Fin.ext
    change (byteAddress packet.base row col - packet.base) / 2 % 768 = col.val
    rw [d, colIndex]
  dsimp only
  rw [r, c]

theorem unchanged {rows : Nat} (packet : Packet rows) (addr : Nat) (dead : ¬ writes packet addr) :
    store packet addr = packet.memory addr := by simp [store, dead]

inductive Program where
  | reload
  | register

abbrev Run := (rows : Nat) → Packet rows → Output rows

def Program.run (program : Program) : Run := fun _ packet =>
  let stored := store packet
  let read := fun row col =>
    match program with
    | .reload => stored (byteAddress packet.base row col)
    | .register =>
      if 768 ≤ packet.block then cast .bf16 Config.ieeeValue (packet.mixed row col)
      else stored (byteAddress packet.base row col)
  ⟨fun r c =>
    let wide := cast .fp32 Config.ieeeValue (read r c)
    let normalized := mul Config.ieeeValue wide (MuddNorm.inverseRMS packet.native (packet.sumSquares r))
    packet.native.e4m3 (mul Config.ieeeValue normalized (Value.ofBits .fp32 0x408f5c29)),
    stored⟩

theorem run_eq : Program.register.run = Program.reload.run := by
  funext rows packet
  simp only [Program.run, read_written, ite_self]

def same (p q : Program) : Prop := p.run = q.run

instance : Spec.ProgramSyntax Program where
  Statement := Unit
  Signature := Unit
  signature := fun _ => ()
  body := fun _ => []
  structural := some same

specification retained_equiv :
    Program.register ≡[([] : Spec.Assumptions Unit)] Program.reload := by
  exact Spec.FloatingPoint.ofStructural (structural := same) rfl rfl run_eq

#print_fp_assumptions retained_equiv
#axiomsClean read_written
#axiomsClean unchanged
#axiomsClean retained_equiv
#axiomsClean run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.MuddResidualStorage
