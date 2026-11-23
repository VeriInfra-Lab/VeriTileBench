import VeriTile.Triton.Float.BitValue

/-! The finite binary64 arithmetic used by the pinned Python host
calculations. Values are represented by their exact rational magnitudes. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.HostFloat64
open VeriTile.Triton.FP (pow2)

/-- Round a nonnegative rational to the nearest natural, with even ties. -/
def nearestNat (x : Rat) : Nat :=
  let q := x.num.natAbs / x.den
  let r := x.num.natAbs % x.den
  if 2 * r > x.den || (2 * r == x.den && q % 2 == 1) then q + 1 else q

def roundPositive (x : Rat) : Rat :=
  if x == 0 then 0 else
    let estimate : Int := (x.num.natAbs.log2 : Int) - x.den.log2
    let exponent := if x < pow2 estimate then estimate - 1 else estimate
    let unit := pow2 (max exponent (-1022) - 52)
    (nearestNat (x / unit) : Rat) * unit

def round (x : Rat) : Rat := if x < 0 then -roundPositive (-x) else roundPositive x

end VeriTile.Bench.Models.NanoGPT.Track1.HostFloat64
