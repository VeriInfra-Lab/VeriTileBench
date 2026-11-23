import VeriTile.Triton.Float.BitValue

/-! E4M3 finite encoding for the CUDA FP8 cache model. The magnitude is
clamped at 448. The least subnormal is 2^-9, the exponent bias is seven,
and the three-bit fraction rounds to nearest even. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.FP8E4M3
open VeriTile.Triton.FP

private def roundEven (x : Rat) : Nat :=
  let q := x.num.natAbs / x.den
  let remainder := x.num.natAbs % x.den
  if 2 * remainder > x.den || (2 * remainder == x.den && q % 2 == 1) then q + 1 else q

def magnitude (x : Rat) : Nat :=
  if x == 0 then 0
  else
    let x := min x 448
    let estimate : Int := (x.num.natAbs.log2 : Int) - x.den.log2
    let exponent := if x < pow2 estimate then estimate - 1 else estimate
    let binade := max exponent (-6)
    let significand := roundEven (x / pow2 (binade - 3))
    if significand < 8 then significand
    else (binade + 7).toNat * 8 + significand - 8

/-- The input chain is minnum(x,448), maxnum(_,−448), then the
round-to-nearest-even saturating conversion. NaN picks the upper bound. -/
def clampConvert (x : Value .fp32) : UInt8 :=
  match x.decode with
  | .nan .. => 0x7e
  | .infinity negative => if negative then 0xfe else 0x7e
  | .finite negative m => UInt8.ofNat ((if negative then 128 else 0) + magnitude m)

#guard clampConvert (Value.zero .fp32 true) == 0x80
#guard clampConvert (Value.qNaN .fp32) == 0x7e
#guard magnitude 25 == 0x5c
#guard magnitude 124 == 0x70
#guard magnitude 448 == 0x7e
#guard magnitude (1 / 1024) == 0x00
#guard magnitude (3 / 1024) == 0x02

end VeriTile.Bench.Models.NanoGPT.Track1.FP8E4M3
