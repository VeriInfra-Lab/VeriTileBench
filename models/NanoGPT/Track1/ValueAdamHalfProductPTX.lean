import models.NanoGPT.Track1.QKVTTIRBinding

namespace VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise.GuardedHalf

structure Registers where
  output : String
  input : String
  deriving Inhabited

def opener : String := ".reg .b32 %isnan<4>, %flags, %gain;"

private def doneLabel : String := "L__half_product_done:"

private def packedRegister (name : String) : Bool :=
  name.startsWith "%r" && ((name.drop 2).toString.toNat?).isSome

private def pairFromMultiply (line : String) : Option Registers := do
  let [op, dst, src, gain] := line.splitOn " " | none
  guard (op == "mul.rn.f16x2" && gain == "%gain;" &&
    dst.endsWith "," && src.endsWith ",")
  let dst := (dst.dropEnd 1).toString
  let src := (src.dropEnd 1).toString
  guard (packedRegister dst && packedRegister src)
  return ⟨dst, src⟩

private def scalarPath (pair : Registers) : List String :=
  [ "mov.b32 {%a, %b}, " ++ pair.input ++ ";"
  , "cvt.f32.f16 %lo, %a;"
  , "cvt.f32.f16 %hi, %b;"
  , "mul.f32 %lo, %lo, 0f3e000000;"
  , "mul.f32 %hi, %hi, 0f3e000000;"
  , "cvt.rn.f16.f32 %a, %lo;"
  , "cvt.rn.f16.f32 %b, %hi;"
  , "mov.b32 " ++ pair.output ++ ", {%a, %b};"
  ]

private def packedPath (pair : Registers) : String :=
  s!"mul.rn.f16x2 {pair.output}, {pair.input}, %gain;"

private def sequence (pairs : Array Registers) : List String :=
  [ opener
  , ".reg .pred %nanLane, %slow;"
  , ".reg .b16 %a, %b;"
  , ".reg .f32 %lo, %hi;"
  , "mov.b32 %gain, 0x30003000;"
  , s!"max.NaN.f16x2 %isnan0, {pairs[0]!.input}, {pairs[1]!.input};"
  , s!"max.NaN.f16x2 %isnan1, {pairs[2]!.input}, {pairs[3]!.input};"
  , "max.NaN.f16x2 %flags, %isnan0, %isnan1;"
  , "set.nan.f16x2.f16x2 %flags, %flags, %flags;"
  , "setp.ne.b32 %nanLane, %flags, 0;"
  , "vote.sync.any.pred %slow, %nanLane, 0xffffffff;"
  , "@%slow bra.uni L__half_product_slow;"
  ] ++ pairs.toList.map packedPath ++
  ["bra.uni L__half_product_done;", "L__half_product_slow:"] ++
  pairs.toList.flatMap scalarPath ++ [doneLabel]

private def decode (body : List String) : Option (Array Registers) := do
  let candidates := (body.drop 12).take 4
  guard (candidates.length == 4)
  let pairs ← candidates.mapM pairFromMultiply
  let regs := pairs.flatMap fun pair => [pair.output, pair.input]
  guard (regs.eraseDups.length == regs.length)
  let pairs := pairs.toArray
  guard (body == sequence pairs)
  return pairs

/-- The scoped PTX fragment has four disjoint packed inputs and outputs.
The returned before and suffix preserve their original instruction order. -/
def extract (lines : List String) :
    Option (List String × Array Registers × List String) := do
  let before := lines.takeWhile (· != opener)
  let "{" :: revPrefix := before.reverse | none
  let remainder := lines.drop before.length
  let body := remainder.takeWhile (· != doneLabel)
  let last :: "}" :: suffix := remainder.drop body.length | none
  guard (last == doneLabel)
  let pairs ← decode (body ++ [last])
  return (revPrefix.reverse, pairs, suffix)

/-- A diagnostic for the normalized compiler output. -/
def matchesFragment (source : String) : Bool :=
  (extract (QKVTTIR.normalizePTX source)).isSome

end VeriTile.Bench.Models.NanoGPT.Track1.ValueAdam.Elementwise.GuardedHalf
