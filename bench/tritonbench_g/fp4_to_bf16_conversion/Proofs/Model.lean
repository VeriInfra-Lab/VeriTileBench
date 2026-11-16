import bench.tritonbench_g.fp4_to_bf16_conversion.Kernels
import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_FPOperations
import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_FPLayout

/-! The source arithmetic is evaluated once with an explicit primitive model.
The Real instance erases rounding; FP instances retain it. Neither instance
assumes an equality between complete programs. -/
namespace VeriTile.Bench.Repairs.Fp4Scaled
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16.FPValue
noncomputable section

namespace Kernels
inductive Tile where
  | b128 | b256 | b512 | b1024 | b2048
  deriving DecidableEq

def Tile.block : Tile → Nat
  | .b128 => 128 | .b256 => 256 | .b512 => 512 | .b1024 => 1024 | .b2048 => 2048
end Kernels

structure Parameters where
  input : RegionName
  scales : RegionName
  output : RegionName
  count : Nat
  tile : Kernels.Tile

def Parameters.block (p : Parameters) := p.tile.block
def Parameters.groups (p : Parameters) := p.block/16

def source (p : Parameters) : ComputeKernel :=
  Kernels.triton_f4_to_scaled_bf16_kernel p.input p.scales p.output p.count p.block

/-- Only the numerical primitives actually used by the source. -/
class Numerics (α : Type) where
  two : α
  bits : Float32Bits → α
  narrow : α → α
  pow2 : Int → α
  mul : α → α → α

variable {α : Type} [Inhabited α] [Numerics α]

def backend (rest : Algebra α) : Algebra α :=
  { rest with
    fp32Bits := Numerics.bits
    literal := fun p d q => if q=2 then Numerics.two else rest.literal p d q
    libdevicePowi := fun _ => some (fun _ n => Numerics.pow2 n)
    binary := fun p d op x y => match d,op with
      | .bf16,.mul => Numerics.mul x y
      | _,_ => rest.binary p d op x y
    cast := fun p src dst x => match src,dst with
      | .real,.bf16 | .fp32,.bf16 => Numerics.narrow x
      | _,_ => if src=dst then x else rest.cast p src dst x }

def converted (n : Nat) : α :=
  Numerics.narrow (Numerics.bits ⟨BitVec.ofInt 32 (integerWord n)⟩)

def inputOffset (p : Parameters) (s : State α) (i : Nat) := s.pids 0*p.block+i

def loaded (p : Parameters) (s : State α) (i : Nat) : Nat :=
  if inputOffset p s i < p.count then (s.mem p.input (inputOffset p s i)).read .nat else 0

def laneNibble (p : Parameters) (s : State α) (j : TileIndex [p.block*2]) : Nat :=
  let b := loaded p s (j.1.val/2)
  if j.1.val%2=0 then b >>> 4 else b &&& 15

end
end VeriTile.Bench.Repairs.Fp4Scaled
