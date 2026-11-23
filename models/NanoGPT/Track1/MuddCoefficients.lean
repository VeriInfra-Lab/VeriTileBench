import Mathlib.Data.Fin.Basic
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

/-! The ten coefficient planes consumed by the post-layer MUDD mix.
Each output is BF16. The reference fuses the scaled bias addition and
the separately rounded FP32 product from the group projection. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddCoefficients
open VeriTile.Triton.FP
open VeriTile.Spec
open scoped VeriTile.Spec

inductive Shape where
  | initial | middle | taper | bulk

def Shape.rows : Shape → Nat
  | .initial => 16384
  | .middle => 32768
  | .taper => 40960
  | .bulk => 49152

def Shape.plane (s : Shape) : Nat := s.rows * 12

def Shape.blocks (s : Shape) : Nat := s.plane / 512

theorem plane_multiple (s : Shape) : s.blocks * 512 = s.plane := by
  cases s <;> decide

abbrev BF16 := Value .bf16

structure Inputs where
  mu : Nat → BF16
  bias : Nat → BF16
  delta : Nat → BF16

def applyCoefficient (bias mu delta : BF16) : BF16 :=
  let b := cast .fp32 Config.ieeeValue bias
  let m := cast .fp32 Config.ieeeValue mu
  let d := cast .fp32 Config.ieeeValue delta
  let scale := Value.ofBits .fp32 0x3dcccccd
  let result := fma Config.ieeeValue (add Config.ieeeValue b m) scale (mul Config.ieeeValue d scale)
  cast .bf16 Config.ieeeValue result

def coefficientAt (input : Inputs) (plane offset : Nat) : BF16 :=
  let row := offset / 12
  let group := offset % 12
  applyCoefficient (input.bias (14 + plane)) (input.mu (row * 10 + plane))
    (input.delta (row * 120 + plane * 12 + group))

structure Store where
  port : Fin 10
  offset : Nat
  value : BF16

def referencePlane (shape : Shape) (input : Inputs) (plane : Fin 10) (index : Fin shape.plane) :
    Store :=
  ⟨plane, index.val, coefficientAt input plane.val index.val⟩

def launchedCoefficient (shape : Shape) (input : Inputs) (plane : Fin 10)
    (cta : Fin shape.blocks) (withinTile : Fin 512) : Store :=
  let index := 512 * cta.val + withinTile.val
  ⟨plane, index, coefficientAt input plane.val index⟩

def packedPlane (shape : Shape) (input : Inputs) (plane : Fin 10) (index : Fin shape.plane) :
    Store :=
  let cta : Fin shape.blocks := ⟨index.val / 512, by
    have fit : index.val < shape.blocks * 512 := by simpa only [plane_multiple] using index.isLt
    omega⟩
  let withinTile : Fin 512 := ⟨index.val % 512, Nat.mod_lt _ (by decide)⟩
  launchedCoefficient shape input plane cta withinTile

theorem launched_covers (shape : Shape) (input : Inputs) (plane : Fin 10) (index : Fin shape.plane) :
    packedPlane shape input plane index = referencePlane shape input plane index := by
  have offset := Nat.div_add_mod index.val 512
  simp only [packedPlane, launchedCoefficient, referencePlane, offset, Shape.plane, Nat.mul_assoc]

theorem launch_fits (shape : Shape) (cta : Fin shape.blocks) (i : Fin 512) :
    512 * cta.val + i.val < shape.plane := by
  have extent := plane_multiple shape
  have tile := cta.isLt
  have index := i.isLt
  omega

theorem unique_store (shape : Shape)
    (plane other : Fin 10) (cta next : Fin shape.blocks) (i j : Fin 512)
    (port : plane = other) (address : 512 * cta.val + i.val = 512 * next.val + j.val) :
    plane = other ∧ cta = next ∧ i = j := by
  have left := i.isLt
  have right := j.isLt
  exact ⟨port, Fin.ext (by omega), Fin.ext (by omega)⟩

/-- The unsigned multiply-high and shift emitted for division by twelve.
The PTX literal is interpreted as a 32-bit word by mul.hi.u32. -/
def ptxDivide (index : Nat) : Nat :=
  let reciprocal := (BitVec.ofInt 32 (-1431655765)).toNat
  let high := index * reciprocal / 2^32
  high / 8

theorem ptx_division_exact (index : Nat) (bound : index < 589824) :
    ptxDivide index = index / 12 := by
  simp only [ptxDivide, Nat.div_div_eq_div_mul]
  change index * 2863311531 / 34359738368 = index / 12
  have value : index * 2863311531 = 34359738368 * (index / 12) +
      (4 * (index / 12) + 2863311531 * (index % 12)) := by omega
  have rem : 4 * (index / 12) + 2863311531 * (index % 12) < 34359738368 := by omega
  rw [value, Nat.mul_add_div (by decide : 0 < 34359738368), Nat.div_eq_of_lt rem, Nat.add_zero]

theorem packed_lanes_share_row (tile : Nat) (lane : Fin 128) (word : Fin 4) :
    let first := 512 * tile + 4 * lane.val
    (first + word.val) / 12 = first / 12 ∧ (first + word.val) % 12 = first % 12 + word.val := by
  have w := word.isLt
  dsimp
  omega

#axiomsClean ptx_division_exact
#axiomsClean packed_lanes_share_row

abbrev Bank (shape : Shape) := Fin 10 → Fin shape.plane → BF16
abbrev Run := (shape : Shape) → Inputs → Bank shape

structure Packet where
  shape : Shape
  input : Inputs

def reference : Run := fun shape input plane index => (referencePlane shape input plane index).value

def fused : Run := fun shape input plane index => (packedPlane shape input plane index).value

theorem fused_equal : fused = reference := by
  funext shape input plane index
  exact congrArg Store.value (launched_covers shape input plane index)

inductive Program where
  | separate | packed

def Program.run (p : Program) (s : Shape) (input : Inputs) (plane : Fin 10) (index : Fin s.plane) :
    Store :=
  match p with
  | .separate => referencePlane s input plane index
  | .packed => packedPlane s input plane index

def sameRun (a b : Program) : Prop := ∀ (s : Shape) (input : Inputs) (plane : Fin 10) (index : Fin s.plane),
  a.run s input plane index = b.run s input plane index

instance : Spec.ProgramSyntax Program where
  Statement := Unit
  Signature := Unit
  signature := fun _ => ()
  body := fun _ => []
  structural := some sameRun

specification packed_equal : Program.packed ≡[([] : Spec.Assumptions Unit)] Program.separate := by
  refine Spec.FloatingPoint.ofStructural (structural := sameRun) rfl rfl ?_
  exact launched_covers

#axiomsClean launched_covers
#axiomsClean launch_fits
#axiomsClean unique_store
#axiomsClean fused_equal
#axiomsClean packed_equal
#print_fp_assumptions packed_equal

end VeriTile.Bench.Models.NanoGPT.Track1.MuddCoefficients
