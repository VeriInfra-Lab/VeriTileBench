import models.NanoGPT.Track1.TrainingCEGatherBinding
import models.NanoGPT.Track1.ValidationCE

/-! Recover the target gather used by the compiled validation loss.
The address checks trace both loads to the launch arguments. The scalar
checks preserve the BF16 conversion, sigmoid and packed mask test. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.ValidationTargetBinding
open TrainingCEGatherBinding
open VeriTile.Triton FP
open TailNorm (BF16 F32 zero)

structure Code where
  lines : List String
  edges : Array (List Nat)

private def Code.defn (code : Code) (reg : String) (use : Nat) :=
  uniqueDefinition code.lines code.edges reg use

def Code.load (code : Code) (reg : String) (use : Nat)
    (init op : String) : Option (String × Nat) := do
  let [(initial, initPC), (load, pc)] :=
    code.lines.zipIdx.filter (fun (line, _) => defines reg line) | none
  guard (operands initial == [init, reg, "0x0"] && pc == initPC + 1)
  let [loadOp, dst, "[", ptr, "+", "0", "]", _] := operands load | none
  guard (loadOp == "ld.global.L1::evict_last.L2::cache_hint." ++ op && dst == reg &&
    pc < use && dominates code.edges pc use)
  return (ptr, pc)

private def checkParameter (code : Code) (entry : String) (arg : Nat)
    (reg : String) (use : Nat) : Option Unit := do
  let (["ld.param.b64", _, input], _) ← code.defn reg use | none
  guard (input == s!"[{entry}_param_{arg}]")

private def checkRowWord (code : Code) (row row32 arg : String) (use : Nat) : Option Unit := do
  if arg == row32 then
    let _ ← code.defn row32 use
    return ()
  else
    let (["cvt.u32.u64", _, source], pc) ← code.defn arg use | none
    guard (source == row)
    let _ ← code.defn row pc
    return ()

private def projectionColumn (code : Code) (entry : String) (slab : Nat)
    (row row32 ptr : String) (use : Nat) : Option (String × Nat) := do
  let (addr, addrPC) ← code.defn ptr use
  let positions : String × Nat × String × Nat ←
    if slab == 0 then do
      let ["mad.wide.s32", _, scaled, "2", baseOffset] := addr | none
      pure (scaled, addrPC, baseOffset, addrPC)
    else do
      let ["add.s64", _, baseOffset, offset] := addr | none
      let (["shl.b64", _, wide, "1"], scaled) ← code.defn offset addrPC | none
      let (["cvt.s64.s32", _, elements], extended) ← code.defn wide scaled | none
      pure (elements, extended, baseOffset, addrPC)
  let (rowElements, rowPC, baseOffset, basePC) := positions
  let (["mul.lo.s32", _, localRow, "50304"], scaled) ← code.defn rowElements rowPC | none
  checkRowWord code row row32 localRow scaled
  let (["add.s64", _, base, byteOffset], pc) ← code.defn baseOffset basePC | none
  checkParameter code entry (2 + slab) base pc
  let (["shl.b64", _, column, "1"], converted) ← code.defn byteOffset pc | none
  return (column, converted)

private def checkColumn (code : Code) (column targetPointer : String) (use : Nat) : Option Unit := do
  let (["selp.b64", _, wrapped, current, negative], pc) ← code.defn column use | none
  let (["add.s64", _, a, "50304"], _) ← code.defn wrapped pc | none
  let (["setp.lt.s64", _, b, "0"], _) ← code.defn negative pc | none
  guard (a == current && b == current)
  let (["selp.b64", _, "0", target, ignored], sanitized) ← code.defn current pc | none
  let (["setp.eq.b64", _, source, "-100"], tested) ← code.defn ignored sanitized | none
  guard (source == target)
  let (ptr, _) ← code.load target tested "mov.u64" "b64"
  guard (ptr == targetPointer)

private def checkInput (code : Code) (entry : String) (slab : Nat)
    (row32 id : String) (use : Nat) : Option Unit := do
  let (["selp.b32", _, wrapped, input, negative], pc) ← code.defn id use | none
  let (["add.s32", _, a, "50304"], _) ← code.defn wrapped pc | none
  let (["setp.lt.s32", _, b, "0"], tested) ← code.defn negative pc | none
  guard (a == input && b == input)
  let (ptr, loadPC) ← code.load input tested "mov.u32" "b32"
  let ([op, _, index, "4", base], address) ← code.defn ptr loadPC | none
  guard (op == "mad.wide.s32" || (slab == 0 && op == "mad.wide.u32"))
  checkParameter code entry 0 base address
  if slab == 0 then
    guard (index == row32)
  else
    let (["add.s32", _, row, offset], _) ← code.defn index address | none
    guard (row == row32 && offset.toNat? == some (slab * ValidationCE.slabRows))

private def checkSignedQuotient (code : Code) (column quotient : String) (use : Nat) : Option Unit := do
  let (["shr.s64", _, biased, "3"], pc) ← code.defn quotient use | none
  let (["add.s64", _, source, bias], adjusted) ← code.defn biased pc | none
  let (["shr.u64", _, sign, "61"], logical) ← code.defn bias adjusted | none
  let (["shr.s64", _, original, "63"], _) ← code.defn sign logical | none
  guard (source == column && original == column)

private def checkRemainder (code : Code) (reg : String) (use : Nat) : Option String := do
  let (["sub.s64", _, quotient, product], pc) ← code.defn reg use | none
  let (["mul.lo.s64", _, estimate, "6288"], multiplied) ← code.defn product pc | none
  let (["add.s64", _, upper, sign], estimated) ← code.defn estimate multiplied | none
  let (["shr.s64", _, high, "5"], shifted) ← code.defn upper estimated | none
  let (["shr.u64", _, source, "63"], _) ← code.defn sign estimated | none
  let (["mul.hi.s64", _, numerator, "93876560171549881"], _) ← code.defn high shifted | none
  guard (high == source && numerator == quotient)
  return quotient

inductive MaskMode where
  | logical | signedRemainder
  deriving DecidableEq

private def checkMask (code : Code) (entry : String) (slab : Nat)
    (row32 column ptr shift : String) (load test : Nat) : Option MaskMode := do
  let (addr, addrPC) ← code.defn ptr load
  let (["and.b32", _, narrow, mask], masked) ← code.defn shift test | none
  let (["cvt.u32.u64", _, bitOffset], converted) ← code.defn narrow masked | none
  match addr with
  | ["mad.wide.u32", _, id, "6288", firstByte] =>
    let (["add.s64", _, base, byte], pc) ← code.defn firstByte addrPC | none
    checkParameter code entry 1 base pc
    let (["shr.u64", _, source, "3"], _) ← code.defn byte pc | none
    checkInput code entry slab row32 id addrPC
    guard (source == column && bitOffset == column && mask == "7")
    return .logical
  | ["add.s64", _, firstByte, rowOffset] =>
    let (["add.s64", _, base, byte], pc) ← code.defn firstByte addrPC | none
    checkParameter code entry 1 base pc
    let (["cvt.s64.s32", _, scaled], ext) ← code.defn rowOffset addrPC | none
    let (["mul.lo.s32", _, id, "6288"], mul) ← code.defn scaled ext | none
    checkInput code entry slab row32 id mul
    let quotient ← checkRemainder code byte pc
    checkSignedQuotient code column quotient pc
    let (["sub.s64", _, unaligned, aligned], sub) ← code.defn bitOffset converted | none
    let (["shl.b64", _, parts, "3"], _) ← code.defn aligned sub | none
    guard (unaligned == column && parts == quotient && mask == "255")
    return .signedRemainder
  | _ => none

def decode (code : Code) (entry : String) (slab : Nat)
    (row row32 targetPointer output : String) (use : Nat) : Option MaskMode := do
  let (["selp.f32", _, "0fC2700000", live, masked], selected) ← code.defn output use | none
  let (["mul.f32", _, reciprocal, "0f41B80000"], scaled) ← code.defn live selected | none
  let (["div.full.f32", _, one, denominator], divided) ← code.defn reciprocal scaled | none
  let (["mov.b32", _, "0f3F800000"], _) ← code.defn one divided | none
  let (["add.f32", _, power, "0f3F800000"], added) ← code.defn denominator divided | none
  let (["ex2.approx.f32", _, exponent], raised) ← code.defn power added | none
  let (["mul.f32", _, affine, "0f3FB8AA3B"], log2) ← code.defn exponent raised | none
  let (["fma.rn.f32", _, biased, "0fBE088889", "0f00000000"], fused) ← code.defn affine log2 | none
  let (["add.f32", _, widened, "0f40A00000"], shifted) ← code.defn biased fused | none
  let (["cvt.f32.bf16", _, fromMem], castPC) ← code.defn widened shifted | none
  let (projection, projectionPC) ← code.load fromMem castPC "mov.u16" "b16"
  let (column, checkPC) ← projectionColumn code entry slab row row32 projection projectionPC
  checkColumn code column targetPointer checkPC
  let (["setp.ne.b16", _, bit, "0"], tested) ← code.defn masked selected | none
  let (["and.b16", _, moved, "1"], isolated) ← code.defn bit tested | none
  let (["shr.u16", _, byte, shift], movedPC) ← code.defn moved isolated | none
  let (["and.b16", _, maskByte, "255"], narrowed) ← code.defn byte movedPC | none
  let (mask, maskPC) ← code.load maskByte narrowed "mov.u16" "b8"
  checkMask code entry slab row32 column mask shift maskPC movedPC

private def f32 := Value.ofBits .fp32
private def ieee := Config.ieeeValue

/-- The selects before the gather map the ignored int64 word to zero
and apply the signed wrap used by Inductor's indexing. -/
def gatherColumn (word : UInt64) : UInt64 :=
  let sanitized := if word == ValidationCE.ignoredTarget then 0 else word
  if sanitized.toNat ≥ 0x8000000000000000 then sanitized + 50304 else sanitized

theorem gatherColumn_live (column : Fin 50304) :
    gatherColumn (UInt64.ofNat column.val) = UInt64.ofNat column.val := by
  have bounded := column.isLt
  have raw : (UInt64.ofNat column.val).toNat = column.val := by
    rw [UInt64.toNat_ofNat', Nat.mod_eq_of_lt (by omega)]
  have live : UInt64.ofNat column.val != ValidationCE.ignoredTarget := by
    apply bne_iff_ne.mpr
    intro equal
    have : column.val = 0xffffffffffffff9c := by
      rw [← raw, equal]
      rfl
    omega
  simp [gatherColumn, bne_iff_ne.mp live, raw, show ¬ 0x8000000000000000 ≤ column.val by omega]

theorem gatherColumn_ignored : gatherColumn ValidationCE.ignoredTarget = 0 := by decide

/-- Within the asserted column range the signed shift's bias is zero.
The signed high multiply estimates the quotient for division by 6288. -/
def MaskMode.indices (mode : MaskMode) (column : Fin 50304) : Nat × Nat :=
  let k := column.val
  match mode with
  | .logical => (k >>> 3, k &&& 7)
  | .signedRemainder =>
    let quotient := (k + ((k / 2 ^ 63) >>> 61)) >>> 3
    let high := quotient * 93876560171549881 / 2 ^ 64
    let div6288 := (high >>> 5) + (high >>> 63)
    (quotient - div6288 * 6288, (k - (quotient <<< 3)) &&& 255)

theorem mask_indices_exact (mode : MaskMode) (column : Fin 50304) :
    mode.indices column = (column.val / 8, column.val % 8) := by
  have b := column.isLt
  have mask7 (n : Nat) : n &&& 7 = n % 8 := Nat.and_two_pow_sub_one_eq_mod n 3
  have mask255 (n : Nat) : n &&& 255 = n % 256 := Nat.and_two_pow_sub_one_eq_mod n 8
  cases mode with
  | logical => simp [MaskMode.indices, mask7, Nat.shiftRight_eq_div_pow]
  | signedRemainder =>
    simp only [MaskMode.indices, Nat.shiftRight_eq_div_pow, Nat.shiftLeft_eq, Nat.reducePow]
    have sign : column.val / 9223372036854775808 = 0 := by omega
    have estimate : column.val / 8 * 93876560171549881 / 18446744073709551616 < 32 := by
      omega
    simp only [sign, Nat.zero_div, Nat.add_zero]
    rw [Nat.div_eq_of_lt estimate,
      Nat.div_eq_of_lt (Nat.lt_trans estimate (by decide : 32 < 9223372036854775808))]
    simp only [Nat.zero_add, Nat.zero_mul, Nat.sub_zero, mask255]
    congr 1
    omega

def testMaskByte (byte : UInt8) (shift : Nat) : Bool :=
  ((byte.toNat &&& 255) >>> shift) &&& 1 != 0

theorem testMaskByte_exact (byte : UInt8) (shift : Nat) :
    testMaskByte byte shift = byte.toNat.testBit shift := by
  have byteMask : byte.toNat &&& 255 = byte.toNat :=
    Nat.and_two_pow_sub_one_of_lt_two_pow (n := 8) byte.toNat_lt
  unfold testMaskByte
  rw [byteMask]
  conv => rhs; rw [← Nat.add_zero shift, ← Nat.testBit_shiftRight, Nat.testBit_zero]
  have odd (n : Nat) : n &&& 1 = n % 2 := Nat.and_two_pow_sub_one_eq_mod n 1
  rw [odd]
  apply Bool.eq_iff_iff.mpr
  simp only [bne_iff_ne, decide_eq_true_eq]
  omega

/-- The selected target logit from the checked BF16 load and mask drop bit. -/
def payload (native : ValidationCE.Native) (dropped : Bool) (projection : BF16) : F32 :=
  let shifted := FP.add ieee (TailNorm.widen projection) (f32 0x40a00000)
  let affine := FP.fma ieee shifted (f32 0xbe088889) zero
  let exp := native.ex2Approx false (FP.mul ieee affine (f32 0x3fb8aa3b))
  let reciprocal := native.reciprocalFull (FP.add ieee exp (f32 0x3f800000))
  if dropped then f32 0xc2700000 else FP.mul ieee reciprocal (f32 0x41b80000)

theorem payload_exact (native : ValidationCE.Native) (read : Nat → UInt8)
    (column : Nat) (projection : BF16) :
    payload native ((read (column / 8)).toNat.testBit (column % 8)) projection =
      ValidationCE.logit native read column projection := rfl

theorem gathered_payload_exact (mode : MaskMode) (native : ValidationCE.Native)
    (read : Nat → UInt8) (projection : Fin 50304 → BF16) (column : Fin 50304) :
    let addressed := mode.indices column
    payload native (testMaskByte (read addressed.1) addressed.2) (projection column) =
      ValidationCE.logit native read column.val (projection column) := by
  dsimp only
  rw [mask_indices_exact, testMaskByte_exact]
  rfl

#axiomsClean gatherColumn_live
#axiomsClean mask_indices_exact
#axiomsClean testMaskByte_exact
#axiomsClean payload_exact
#axiomsClean gathered_payload_exact
end VeriTile.Bench.Models.NanoGPT.Track1.ValidationTargetBinding
