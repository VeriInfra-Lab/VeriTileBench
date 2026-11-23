import models.NanoGPT.Track1.MuddCoefficients
import models.NanoGPT.Track1.HeadActivationBinding

namespace VeriTile.Bench.Models.NanoGPT.Track1.MuddCoefficients.Binding
open VeriTile.Spec
open scoped VeriTile.Spec

/-- The checked kernel uses one output pointer per plane. The launcher
chooses the row extent through the x-grid and uses exactly ten y-blocks. -/
private def assembly : List String := QKVTTIR.normalizePTX r#"
.version 8.7
.target sm_90a
.address_size 64
.visible .entry mudd_coefficients(
.param .u64 .ptr .global .align 1 mudd_coefficients_param_0,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_1,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_2,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_3,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_4,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_5,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_6,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_7,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_8,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_9,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_10,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_11,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_12,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_13,
.param .u64 .ptr .global .align 1 mudd_coefficients_param_14
)
.reqntid 128
{
.reg .pred %p<10>;
.reg .b16 %rs<14>;
.reg .b32 %r<40>;
.reg .b64 %rd<28>;
$L__func_begin0:
ld.param.b64 %rd5, [mudd_coefficients_param_0];
ld.param.b64 %rd6, [mudd_coefficients_param_1];
$L__tmp0:
mov.u32 %r5, %ctaid.x;
shl.b32 %r6, %r5, 9;
ld.param.b64 %rd7, [mudd_coefficients_param_2];
ld.param.b64 %rd8, [mudd_coefficients_param_3];
mov.u32 %r7, %tid.x;
shl.b32 %r8, %r7, 2;
ld.param.b64 %rd9, [mudd_coefficients_param_4];
and.b32 %r9, %r8, 508;
ld.param.b64 %rd10, [mudd_coefficients_param_5];
or.b32 %r10, %r9, %r6;
ld.param.b64 %rd11, [mudd_coefficients_param_6];
ld.param.b64 %rd12, [mudd_coefficients_param_7];
mov.u32 %r11, %ctaid.y;
mul.hi.u32 %r12, %r10, -1431655765;
shr.u32 %r13, %r12, 3;
ld.param.b64 %rd13, [mudd_coefficients_param_8];
ld.param.b64 %rd14, [mudd_coefficients_param_9];
mad.wide.u32 %rd15, %r11, 2, %rd6;
ld.param.b64 %rd16, [mudd_coefficients_param_10];
add.s64 %rd1, %rd15, 28;
ld.param.b64 %rd17, [mudd_coefficients_param_11];
mov.u16 %rs1, 0x0;
ld.global.b16 { %rs1 }, [ %rd1 + 0 ];
ld.param.b64 %rd18, [mudd_coefficients_param_12];
cvt.f32.bf16 %r14, %rs1;
mad.lo.s32 %r15, %r13, 10, %r11;
mad.wide.u32 %rd2, %r15, 2, %rd5;
mov.u16 %rs2, 0x0;
ld.global.b16 { %rs2 }, [ %rd2 + 0 ];
mov.u16 %rs3, 0x0;
ld.global.b16 { %rs3 }, [ %rd2 + 0 ];
mov.u16 %rs4, 0x0;
ld.global.b16 { %rs4 }, [ %rd2 + 0 ];
mov.u16 %rs5, 0x0;
ld.global.b16 { %rs5 }, [ %rd2 + 0 ];
mad.lo.s32 %r16, %r11, 12, %r10;
mad.lo.s32 %r17, %r13, 108, %r16;
mad.wide.u32 %rd3, %r17, 2, %rd7;
mov.u32 %r1, 0x0;
mov.u32 %r2, 0x0;
ld.global.v2.b32 { %r1, %r2 }, [ %rd3 + 0 ];
setp.eq.b32 %p1, %r11, 1;
selp.b64 %rd19, %rd9, %rd8, %p1;
setp.eq.b32 %p2, %r11, 2;
selp.b64 %rd20, %rd10, %rd19, %p2;
setp.eq.b32 %p3, %r11, 3;
selp.b64 %rd21, %rd11, %rd20, %p3;
setp.eq.b32 %p4, %r11, 4;
selp.b64 %rd22, %rd12, %rd21, %p4;
setp.eq.b32 %p5, %r11, 5;
selp.b64 %rd23, %rd13, %rd22, %p5;
setp.eq.b32 %p6, %r11, 6;
selp.b64 %rd24, %rd14, %rd23, %p6;
setp.eq.b32 %p7, %r11, 7;
selp.b64 %rd25, %rd16, %rd24, %p7;
setp.eq.b32 %p8, %r11, 8;
selp.b64 %rd26, %rd17, %rd25, %p8;
setp.eq.b32 %p9, %r11, 9;
selp.b64 %rd27, %rd18, %rd26, %p9;
mad.wide.s32 %rd4, %r10, 2, %rd27;
mov.b32 %r18, {%rs2, %rs3};
mov.b32 {%rs6, %rs7}, %r18;
cvt.f32.bf16 %r19, %rs6;
cvt.f32.bf16 %r20, %rs7;
mov.b32 {%rs8, %rs9}, %r1;
cvt.f32.bf16 %r21, %rs8;
cvt.f32.bf16 %r22, %rs9;
add.rn.f32 %r23, %r14, %r20;
add.rn.f32 %r24, %r14, %r19;
mul.rn.f32 %r25, %r22, 0f3DCCCCCD;
mul.rn.f32 %r26, %r21, 0f3DCCCCCD;
fma.rn.f32 %r27, %r24, 0f3DCCCCCD, %r26;
fma.rn.f32 %r28, %r23, 0f3DCCCCCD, %r25;
cvt.rn.bf16x2.f32 %r3, %r28, %r27;
mov.b32 %r29, {%rs4, %rs5};
mov.b32 {%rs10, %rs11}, %r29;
cvt.f32.bf16 %r30, %rs10;
cvt.f32.bf16 %r31, %rs11;
mov.b32 {%rs12, %rs13}, %r2;
cvt.f32.bf16 %r32, %rs12;
cvt.f32.bf16 %r33, %rs13;
add.rn.f32 %r34, %r14, %r31;
add.rn.f32 %r35, %r14, %r30;
mul.rn.f32 %r36, %r33, 0f3DCCCCCD;
mul.rn.f32 %r37, %r32, 0f3DCCCCCD;
fma.rn.f32 %r38, %r35, 0f3DCCCCCD, %r37;
fma.rn.f32 %r39, %r34, 0f3DCCCCCD, %r36;
cvt.rn.bf16x2.f32 %r4, %r39, %r38;
st.global.v2.b32 [ %rd4 + 0 ], { %r3, %r4 };
ret;
$L__tmp1:
$L__func_end0:
}
"#

def recognizes (ptx : String) : Bool := QKVTTIR.normalizePTX ptx == assembly

structure Program where
  ptx : String
  checked : recognizes ptx = true

def Program.check (ptx : String) : Option Program :=
  if h : recognizes ptx then some ⟨ptx, h⟩ else none

/-- The nine uniform selects start at output zero. -/
private def selectedPort (plane : Fin 10) : Fin 10 :=
  (List.finRange 10).drop 1 |>.foldl (fun selected port =>
    if plane = port then port else selected) (0 : Fin 10)

theorem selected_port_exact (plane : Fin 10) : selectedPort plane = plane := by
  have checked : ∀ plane : Fin 10, selectedPort plane = plane := by decide +kernel
  exact checked plane

private def muOffset (row plane : Nat) : Nat :=
  ((BitVec.ofNat 32 row * 10) + BitVec.ofNat 32 plane).toNat

private def deltaOffset (row plane index : Nat) : Nat :=
  ((BitVec.ofNat 32 row * 108) + (BitVec.ofNat 32 plane * 12 + BitVec.ofNat 32 index)).toNat

/-- The low-word multiply-adds used for the two load offsets are exact over
every supported training shape. -/
theorem packed_offsets_exact (row plane index : Nat)
    (rowFits : row < 49152) (planeFits : plane < 10) (indexFits : index < 589824) :
    muOffset row plane = row * 10 + plane ∧
      deltaOffset row plane index = row * 108 + (plane * 12 + index) := by
  simp only [muOffset, deltaOffset, BitVec.toNat_add, BitVec.toNat_mul, BitVec.toNat_ofNat]
  change ((row % 4294967296 * 10) % 4294967296 + plane % 4294967296) % 4294967296 = _ ∧
    ((row % 4294967296 * 108) % 4294967296 +
      ((plane % 4294967296 * 12) % 4294967296 + index % 4294967296) % 4294967296) % 4294967296 = _
  omega

def interpreted (shape : Shape) (input : Inputs) (plane : Fin 10) (index : Fin shape.plane) : Store :=
  let tile := index.val / 512
  let tid := index.val % 512 / 4
  let part := index.val % 512 % 4
  let start := (HeadActivationBinding.index (BitVec.ofNat 32 tile) (BitVec.ofNat 32 tid)).toNat
  let row := ptxDivide start
  ⟨selectedPort plane, start + part,
    applyCoefficient (input.bias (14 + plane.val)) (input.mu (muOffset row plane.val))
      (input.delta (deltaOffset row plane.val start + part))⟩

theorem interpreted_exact (shape : Shape) (input : Inputs) (plane : Fin 10) (index : Fin shape.plane) :
    interpreted shape input plane index = referencePlane shape input plane index := by
  have indexBound := index.isLt
  have fits : shape.plane ≤ 589824 := by cases shape <;> decide
  have addressEq := HeadActivationBinding.index_exact (index.val / 512) (index.val % 512 / 4) (by omega) (by omega)
  let start := 512 * (index.val / 512) + 4 * (index.val % 512 / 4)
  let offset := index.val % 512 % 4
  have exactIndex : start + offset = index.val := by omega
  have size : start < 589824 := by omega
  have row : start / 12 = index.val / 12 := by
    have aligned : start % 4 = 0 := by omega
    omega
  obtain ⟨mu, deltaOffsetEq⟩ := packed_offsets_exact (index.val / 12) plane.val start
    (by omega) plane.isLt size
  have delta : index.val / 12 * 108 + (plane.val * 12 + start) + offset =
      (index.val / 12 * 120 + plane.val * 12) + index.val % 12 := by omega
  simp only [interpreted, addressEq]
  change (⟨selectedPort plane, start + offset,
    applyCoefficient (input.bias (14 + plane.val)) (input.mu (muOffset (ptxDivide start) plane.val))
      (input.delta (deltaOffset (ptxDivide start) plane.val start + offset))⟩ : Store) =
    referencePlane shape input plane index
  rw [ptx_division_exact start size, row, selected_port_exact, exactIndex, mu, deltaOffsetEq, delta]
  rfl

def Program.run (_ : Program) : Run :=
  fun shape input plane index => (interpreted shape input plane index).value

theorem Program.run_eq (program : Program) : program.run = reference := by
  funext shape input plane index
  exact congrArg Store.value (interpreted_exact shape input plane index)

#axiomsClean selected_port_exact
#axiomsClean packed_offsets_exact
#axiomsClean interpreted_exact
#axiomsClean Program.run_eq

end VeriTile.Bench.Models.NanoGPT.Track1.MuddCoefficients.Binding
