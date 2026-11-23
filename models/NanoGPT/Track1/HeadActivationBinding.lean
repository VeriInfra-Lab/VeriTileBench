import models.NanoGPT.Track1.HeadActivation
import models.NanoGPT.Track1.QKVTTIRBinding
import Std.Tactic.BVDecide

/-! The SM90a activation quantizer loads four contiguous BF16 words and
stores four FP8 bytes per lane. The checked instruction stream includes
the scalar rounding, overflow selection, and packed byte routing. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.HeadActivationBinding
open HeadActivation

private def assembly : List String := QKVTTIR.normalizePTX r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry _activation_fp8(
.param .u64 .ptr .global .align 1 _activation_fp8_param_0,
.param .u64 .ptr .global .align 1 _activation_fp8_param_1,
.param .u64 .ptr .global .align 1 _activation_fp8_param_2,
.param .u64 .ptr .global .align 1 _activation_fp8_param_3
)
.reqntid 128
{
.reg .pred %p<5>;
.reg .b16 %rs<13>;
.reg .b32 %r<48>;
.reg .b64 %rd<6>;
$L__func_begin0:
ld.param.b64 %rd3, [_activation_fp8_param_0];
ld.param.b64 %rd4, [_activation_fp8_param_1];
$L__tmp0:
mov.u32 %r8, %ctaid.x;
shl.b32 %r9, %r8, 9;
mov.u32 %r10, %tid.x;
shl.b32 %r11, %r10, 2;
and.b32 %r12, %r11, 508;
or.b32 %r13, %r12, %r9;
cvt.s64.s32 %rd5, %r13;
mad.wide.s32 %rd1, %r13, 2, %rd3;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
ld.global.v2.b32 { %r1, %r2 }, [ %rd1 + 0 ];
mov.b32 {%rs3, %rs4}, %r1;
cvt.f32.bf16 %r14, %rs4;
cvt.f32.bf16 %r15, %rs3;
mov.b32 {%rs5, %rs6}, %r2;
cvt.f32.bf16 %r16, %rs6;
cvt.f32.bf16 %r17, %rs5;
mul.f32 %r18, %r17, 0f408F5C29;
mul.f32 %r19, %r16, 0f408F5C29;
mul.f32 %r20, %r15, 0f408F5C29;
mul.f32 %r21, %r14, 0f408F5C29;
cvt.rn.bf16.f32 %rs7, %r21;
cvt.rn.bf16.f32 %rs8, %r20;
cvt.rn.bf16.f32 %rs9, %r19;
cvt.rn.bf16.f32 %rs10, %r18;
cvt.f32.bf16 %r5, %rs10;
cvt.f32.bf16 %r6, %rs9;
cvt.f32.bf16 %r3, %rs8;
cvt.f32.bf16 %r4, %rs7;
cvt.rn.satfinite.e4m3x2.f32 %rs1, %r4, %r3;
cvt.rn.satfinite.e4m3x2.f32 %rs2, %r6, %r5;
shr.u32 %r22, %r4, 24;
shr.u32 %r23, %r3, 24;
shr.u32 %r24, %r6, 24;
shr.u32 %r25, %r5, 24;
prmt.b32 %r26, %r25, %r24, 0x3340U;
prmt.b32 %r27, %r23, %r22, 0x3340U;
prmt.b32 %r28, %r27, %r26, 0x5410U;
or.b32 %r29, %r28, 2139062143;
abs.f32 %r30, %r6;
abs.f32 %r31, %r5;
abs.f32 %r32, %r4;
abs.f32 %r33, %r3;
setp.le.f32 %p1, %r33, 0f43E80000;
setp.le.f32 %p2, %r32, 0f43E80000;
setp.le.f32 %p3, %r31, 0f43E80000;
setp.le.f32 %p4, %r30, 0f43E80000;
prmt.b32 %r34, %r29, 0, 0x7773U;
shr.s16 %rs11, %rs2, 8;
cvt.u32.u16 %r35, %rs11;
selp.b32 %r36, %r35, %r34, %p4;
prmt.b32 %r37, %r29, 0, 0x7772U;
cvt.u32.u16 %r38, %rs2;
selp.b32 %r39, %r38, %r37, %p3;
prmt.b32 %r40, %r39, %r36, 0x3340U;
prmt.b32 %r41, %r29, 0, 0x7771U;
shr.s16 %rs12, %rs1, 8;
cvt.u32.u16 %r42, %rs12;
selp.b32 %r43, %r42, %r41, %p2;
prmt.b32 %r44, %r29, 0, 0x7770U;
cvt.u32.u16 %r45, %rs1;
selp.b32 %r46, %r45, %r44, %p1;
prmt.b32 %r47, %r46, %r43, 0x3340U;
prmt.b32 %r7, %r47, %r40, 0x5410U;
add.s64 %rd2, %rd4, %rd5;
st.global.b32 [ %rd2 + 0 ], { %r7 };
ret;
$L__tmp1:
$L__func_end0:
}
"#

def recognizes (ptx : String) : Bool := QKVTTIR.normalizePTX ptx == assembly

abbrev Byte := BitVec 8
abbrev Word := BitVec 32

private def byte (word : Word) (i : Nat) : Byte := word.extractLsb' (8 * i) 8

private def permute (a b : Word) (selector : Nat) : Word :=
  let pick (i : Nat) :=
    let j := selector / 16 ^ i % 8
    if j < 4 then byte a j else byte b (j - 4)
  (pick 3).append ((pick 2).append ((pick 1).append (pick 0)))

private def packedSelect (keep : Fin 4 → Bool) (finite overflow : Fin 4 → Byte) : Word :=
  let first := ((finite 1).append (finite 0)).setWidth 32
  let second := ((finite 3).append (finite 2)).setWidth 32
  let exceptions := (overflow 3).append ((overflow 2).append ((overflow 1).append (overflow 0)))
  let a := if keep 0 then first else permute exceptions 0 0x7770
  let b := if keep 1 then (((first.extractLsb' 0 16).sshiftRight 8).setWidth 32)
    else permute exceptions 0 0x7771
  let c := if keep 2 then second else permute exceptions 0 0x7772
  let d := if keep 3 then (((second.extractLsb' 0 16).sshiftRight 8).setWidth 32)
    else permute exceptions 0 0x7773
  permute (permute a b 0x3340) (permute c d 0x3340) 0x5410

private theorem shifted_high (high low : Byte) :
    ((high.append low).sshiftRight 8).extractLsb' 0 8 = high := by
  apply BitVec.eq_of_getLsbD_eq_iff.mpr
  intro i inside
  have within : 8 + i < 16 := by omega
  have afterLow : ¬ 8 + i < 8 := by omega
  simp only [BitVec.getLsbD_extractLsb', BitVec.getLsbD_sshiftRight, Nat.zero_add, inside, within,
    show ¬16 ≤ i by omega, decide_true, decide_false, Bool.not_false, Bool.true_and, if_true]
  change (high ++ low).getLsbD (8 + i) = high.getLsbD i
  rw [BitVec.getLsbD_append, if_neg afterLow, Nat.add_sub_cancel_left]

theorem packed_select_exact (keep : Fin 4 → Bool) (finite overflow : Fin 4 → Byte) (i : Fin 4) :
    byte (packedSelect keep finite overflow) i.val =
      if keep i then finite i else overflow i := by
  have parts : i.val = 0 ∨ i.val = 1 ∨ i.val = 2 ∨ i.val = 3 := by
    have bound := i.isLt
    omega
  have parts : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by
    simpa only [Fin.ext_iff] using parts
  rcases parts with rfl | rfl | rfl | rfl
  · change byte (packedSelect keep finite overflow) 0 = _
    simp only [packedSelect, permute, byte]
    bv_normalize
    simp_all
  · change byte (packedSelect keep finite overflow) 1 = _
    simp only [packedSelect, permute, byte]
    bv_normalize
    simp_all
    rename_i impossible
    exact impossible (shifted_high (finite 1) (finite 0))
  · change byte (packedSelect keep finite overflow) 2 = _
    simp only [packedSelect, permute, byte]
    bv_normalize
    simp_all
  · change byte (packedSelect keep finite overflow) 3 = _
    simp only [packedSelect, permute, byte]
    bv_normalize
    simp_all
    rename_i impossible
    exact impossible (shifted_high (finite 3) (finite 2))

def laneWord (data : Fin 4 → BF16) : Word :=
  let scaled := fun i => scale gain (data i)
  packedSelect (fun i => acceptsMagnitude (magnitudeBits (scaled i)))
    (fun i => (satfiniteByte (scaled i)).toBitVec) (fun i => (overflowByte (scaled i)).toBitVec)

theorem lane_word_exact (data : Fin 4 → BF16) (i : Fin 4) :
    UInt8.ofBitVec (byte (laneWord data) i.val) = atenCast (scale gain (data i)) := by
  rw [laneWord, packed_select_exact]
  cases h : acceptsMagnitude (magnitudeBits (scale gain (data i))) <;>
    simpa only [h, Bool.false_eq_true, if_false, if_true] using
      (select_native (scale gain (data i))).trans (cast_exact _)

def index (pid tid : Word) : Word := (pid <<< 9) ||| ((tid <<< 2) &&& 508)

theorem index_exact (pid tid : Nat) (block : pid < 4194304) (lane : tid < 128) :
    (index (BitVec.ofNat 32 pid) (BitVec.ofNat 32 tid)).toNat = 512 * pid + 4 * tid := by
  have hp : pid < 2^32 := by omega
  have ht : tid < 2^32 := by omega
  simp only [index, BitVec.toNat_or, BitVec.toNat_and, BitVec.toNat_shiftLeft, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt hp, Nat.mod_eq_of_lt ht]
  change (pid <<< 9) % 4294967296 ||| ((tid <<< 2) % 4294967296 &&& 508) = _
  have pshift : pid <<< 9 = 512 * pid := by simp [Nat.shiftLeft_eq, Nat.mul_comm]
  have tshift : tid <<< 2 = 4 * tid := by simp [Nat.shiftLeft_eq, Nat.mul_comm]
  rw [Nat.mod_eq_of_lt (by rw [pshift]; omega), Nat.mod_eq_of_lt (by rw [tshift]; omega)]
  have mask : 508 = 127 <<< 2 := rfl
  rw [mask, ← Nat.shiftLeft_and_distrib, Nat.and_two_pow_sub_one_of_lt_two_pow (n := 7) lane,
    ← Nat.shiftLeft_add_eq_or_of_lt (i := 9) (by rw [tshift]; omega), pshift, tshift]

theorem lane_span (size pid : Nat) (bound : size < 2^31) (whole : size % 512 = 0)
    (block : pid < size / 512) (tid : Fin 128) :
    let first := (index (BitVec.ofNat 32 pid) (BitVec.ofNat 32 tid.val)).toNat
    first + 4 ≤ size ∧ first % 4 = 0 := by
  rw [index_exact pid tid.val (by omega) tid.isLt]
  have within := tid.isLt
  omega

theorem unique_store (first second : Nat) (l r : Fin 128) (x y : Fin 4)
    (same : 512 * first + 4 * l.val + x.val = 512 * second + 4 * r.val + y.val) :
    first = second ∧ l = r ∧ x = y := by
  have left := l.isLt
  have right := r.isLt
  have hx := x.isLt
  have hy := y.isLt
  exact ⟨by omega, Fin.ext (by omega), Fin.ext (by omega)⟩

def readLane (size : Nat) (input : Fin size → BF16) (first : Nat) : Fin 4 → BF16 := fun i =>
  if fits : first + i.val < size then input ⟨_, fits⟩ else VeriTile.Triton.FP.Value.zero .bf16

def interpretedByte (size : Nat) (input : Fin size → BF16) (idx : Fin size) : UInt8 :=
  let pid := idx.val / 512
  let tid := idx.val % 512 / 4
  let part := idx.val % 512 % 4
  let first := (index (BitVec.ofNat 32 pid) (BitVec.ofNat 32 tid)).toNat
  UInt8.ofBitVec (byte (laneWord (readLane size input first)) part)

theorem interpreted_exact (size : Nat) (fits : size < 2^31) (input : Fin size → BF16) (idx : Fin size) :
    interpretedByte size input idx = atenCast (scale gain (input idx)) := by
  have atIndex := idx.isLt
  let part : Fin 4 := ⟨idx.val % 512 % 4, Nat.mod_lt _ (by decide)⟩
  have start := index_exact (idx.val / 512) (idx.val % 512 / 4) (by omega) (by omega)
  have position : 512 * (idx.val / 512) + 4 * (idx.val % 512 / 4) + part.val = idx.val := by
    dsimp only [part]
    omega
  dsimp only [interpretedByte]
  rw [start, lane_word_exact (i := part)]
  simp only [readLane, position, dif_pos atIndex]

structure Program where
  ptx : String
  checked : recognizes ptx = true

def Program.check (ptx : String) : Option Program :=
  if h : recognizes ptx = true then some ⟨ptx, h⟩ else none

def Program.run (program : Program) : Run := fun size enabled reciprocal input idx =>
  if eligible size enabled reciprocal && recognizes program.ptx then interpretedByte size input idx
  else reference size false reciprocal input idx

theorem Program.run_eq (program : Program) : program.run = reference := by
  funext size enabled reciprocal input idx
  simp only [Program.run, program.checked, Bool.and_true]
  split
  · rename_i accepted
    simp only [eligible, Bool.and_eq_true, decide_eq_true_eq] at accepted
    rcases accepted with ⟨⟨⟨_, equal⟩, fits⟩, _⟩
    rw [equal, interpreted_exact size fits]
    rfl
  · rfl

#axiomsClean packed_select_exact
#axiomsClean lane_word_exact
#axiomsClean index_exact
#axiomsClean lane_span
#axiomsClean unique_store
#axiomsClean interpreted_exact
#axiomsClean Program.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.HeadActivationBinding
