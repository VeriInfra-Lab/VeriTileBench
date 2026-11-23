import models.NanoGPT.Track1.Proofs.MuddPackingNormalization

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddPackingFma
open VeriTile.Triton.FP
open MuddPackingArithmetic (addZero)

abbrev tenth : Value .fp32 := MuddPackingArithmetic.tenth

theorem packing_fma_eq (value : Value .bf16) :
    let x := cast .fp32 Config.ieeeValue value
    fma Config.ieeeValue x tenth (Value.zero .fp32) = mul Config.ieeeValue (addZero x) tenth :=
  (MuddPackingArithmetic.packing_numeric_eq value).2

#axiomsClean packing_fma_eq
end VeriTile.Bench.Models.NanoGPT.Track1.MuddPackingFma
