import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddPackingArithmetic
open VeriTile.Triton.FP

def addZero (x : Value .fp32) : Value .fp32 :=
  add Config.ieeeValue x (Value.zero .fp32)

def tenth : Value .fp32 := Value.ofBits .fp32 0x3dcccccd

def checkSpecialPacking (x : Value .fp32) : Bool :=
  (addZero (addZero x) == addZero x) &&
    (fma Config.ieeeValue x tenth (Value.zero .fp32) == mul Config.ieeeValue (addZero x) tenth)

def checkPackingOps (x : Value .fp32) : Bool :=
  match x.decode with
  | .finite _ mag => if mag == 0 then checkSpecialPacking x else addZero x == x
  | .infinity _ | .nan .. => checkSpecialPacking x

/-- Check both packing identities together for each BF16 word. A nonzero
finite value only needs the exact identity for adding positive zero. -/
def checkedPackingWord (bits : Nat) : Bool :=
  checkPackingOps (cast .fp32 Config.ieeeValue (Value.ofBits .bf16 bits))

def blockPasses (hi : Nat) : Bool :=
  (List.range 1024).all fun lo => checkedPackingWord (hi * 1024 + lo)

end VeriTile.Bench.Models.NanoGPT.Track1.MuddPackingArithmetic
