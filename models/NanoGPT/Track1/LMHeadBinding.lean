import models.NanoGPT.Track1.LMHeadGain
import models.NanoGPT.Track1.LMHeadOutput
import models.NanoGPT.Track1.LMHeadStore
import models.NanoGPT.Track1.QKVTTIRBinding

/-! The artifact recognizer includes the FP32 multiply, BF16 round trip,
packed FP16 to FP8 conversion and both Hopper cache stores. -/
namespace VeriTile.Bench.Models.NanoGPT.Track1.LMHead
open VeriTile.Triton.FP QKVPack QKVLaunch
open QKVTTIR (Bank)

private def bounded (gain : Scalar) (x : Value .bf16) : Value .bf16 :=
  if gain = Gain.pinned then upperMin (Gain.scale x)
  else upperMin (cast .bf16 Config.ieeeValue (mul32 (cast .fp32 Config.ieeeValue x) gain))

private theorem half_bounded_eq (gain : Scalar) (x : Value .bf16) :
    halfByte (bounded gain x) = byte x gain := by
  by_cases same : gain = Gain.pinned
  · rw [same, bounded, if_pos rfl, half_byte_eq]
    exact Gain.byte_eq x
  · rw [bounded, if_neg same, byte]
    exact bounded_half_byte _

private def packedHalfByte (bank : Bank) (gain : Scalar) (p : HeadPoint) : UInt8 :=
  let first := p.colLane.val / 2 * 2
  let halfInput := fun col => bounded gain (bank (Store.rowAt p col))
  let halves := (halfInput first, halfInput (first + 1))
  halfByte (if p.colLane.val % 2 = 0 then halves.1 else halves.2)

private theorem packed_half_byte_eq (bank : Bank) (gain : Scalar) (p : HeadPoint) :
    packedHalfByte bank gain p =
      byte (bank (BitVec.ofNat 64 p.row * BitVec.ofNat 64 50304 + BitVec.ofNat 64 p.col).toNat) gain := by
  have haddress := Store.input_addr p
  unfold Store.rowAddr at haddress
  by_cases parity : p.colLane.val % 2 = 0
  · have own : p.colLane.val / 2 * 2 = p.colLane.val := by omega
    simp only [packedHalfByte, if_pos parity, own, haddress, half_bounded_eq]
  · have own : p.colLane.val / 2 * 2 + 1 = p.colLane.val := by omega
    simp only [packedHalfByte, if_neg parity, own, haddress, half_bounded_eq]

private def emittedHalf (bank : Bank) (gain : Scalar) (p : HeadPoint) : Writes :=
  ⟨Store.rowAddr p, Store.columnAddr p, packedHalfByte bank gain p⟩

/-- The scalar branch uses the exact gain bits. Both tile paths share
the packed conversion and the same row and column store layouts. -/
def hopperDispatch (bank : Bank) (gain : Scalar) (p : HeadPoint) : Option Writes :=
  Store.observe (emittedHalf bank gain) p

theorem hopperDispatch_eq (bank : Bank) (gain : Scalar) (p : HeadPoint) :
    hopperDispatch bank gain p = some (reference bank gain p) := by
  simp only [hopperDispatch, Store.observe_eq, emittedHalf, Store.row_addr, Store.column_addr,
    packed_half_byte_eq, reference]

private def referenceGraph : String := r#"
module {
tt.func public @_quantize_lm_head_dual_kernel(%W: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ROW: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %COL: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %INV: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%w = arith.constant dense<0.000000e+00> : tensor<64x128xbf16>
%q = arith.constant dense<-4.480000e+02> : tensor<64x128xf32>
%q_0 = arith.constant dense<4.480000e+02> : tensor<64x128xf32>
%pid_m = tt.get_program_id x : i32
%pid_n = tt.get_program_id y : i32
%offs_m = arith.muli %pid_m, 64 : i32
%offs_m_1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
%offs_m_2 = tt.splat %offs_m : i32 -> tensor<64xi32>
%offs_m_3 = arith.addi %offs_m_2, %offs_m_1 : tensor<64xi32>
%offs_m_4 = arith.extsi %offs_m_3 : tensor<64xi32> to tensor<64xi64>
%offs_n = arith.muli %pid_n, 128 : i32
%offs_n_5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
%offs_n_6 = tt.splat %offs_n : i32 -> tensor<128xi32>
%offs_n_7 = arith.addi %offs_n_6, %offs_n_5 : tensor<128xi32>
%offs_n_8 = arith.extsi %offs_n_7 : tensor<128xi32> to tensor<128xi64>
%mask = tt.expand_dims %offs_m_4 {axis = 1 : i32} : tensor<64xi64> -> tensor<64x1xi64>
%mask_9 = arith.extsi %M : i32 to i64
%mask_10 = tt.splat %mask_9 : i64 -> tensor<64x1xi64>
%mask_11 = arith.cmpi slt, %mask, %mask_10 : tensor<64x1xi64>
%mask_12 = tt.expand_dims %offs_n_8 {axis = 0 : i32} : tensor<128xi64> -> tensor<1x128xi64>
%mask_13 = arith.extsi %N : i32 to i64
%mask_14 = tt.splat %mask_13 : i64 -> tensor<1x128xi64>
%mask_15 = arith.cmpi slt, %mask_12, %mask_14 : tensor<1x128xi64>
%mask_16 = tt.broadcast %mask_11 : tensor<64x1xi1> -> tensor<64x128xi1>
%mask_17 = tt.broadcast %mask_15 : tensor<1x128xi1> -> tensor<64x128xi1>
%mask_18 = arith.andi %mask_16, %mask_17 : tensor<64x128xi1>
%inv = tt.load %INV : !tt.ptr<f32>
%w_19 = tt.splat %mask_13 : i64 -> tensor<64x1xi64>
%w_20 = arith.muli %mask, %w_19 : tensor<64x1xi64>
%w_21 = tt.splat %W : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>>
%w_22 = tt.addptr %w_21, %w_20 : tensor<64x1x!tt.ptr<bf16>>, tensor<64x1xi64>
%w_23 = tt.broadcast %w_22 : tensor<64x1x!tt.ptr<bf16>> -> tensor<64x128x!tt.ptr<bf16>>
%w_24 = tt.broadcast %mask_12 : tensor<1x128xi64> -> tensor<64x128xi64>
%w_25 = tt.addptr %w_23, %w_24 : tensor<64x128x!tt.ptr<bf16>>, tensor<64x128xi64>
%w_26 = tt.load %w_25, %mask_18, %w : tensor<64x128x!tt.ptr<bf16>>
%q_27 = arith.extf %w_26 : tensor<64x128xbf16> to tensor<64x128xf32>
%q_28 = tt.splat %inv : f32 -> tensor<64x128xf32>
%q_29 = arith.mulf %q_27, %q_28 : tensor<64x128xf32>
%q_30 = arith.truncf %q_29 : tensor<64x128xf32> to tensor<64x128xbf16>
%q_31 = arith.extf %q_30 : tensor<64x128xbf16> to tensor<64x128xf32>
%q_32 = arith.minnumf %q_31, %q_0 : tensor<64x128xf32>
%q_33 = arith.maxnumf %q_32, %q : tensor<64x128xf32>
%0 = tt.splat %ROW : !tt.ptr<f8E4M3FN> -> tensor<64x1x!tt.ptr<f8E4M3FN>>
%1 = tt.addptr %0, %w_20 : tensor<64x1x!tt.ptr<f8E4M3FN>>, tensor<64x1xi64>
%2 = tt.broadcast %1 : tensor<64x1x!tt.ptr<f8E4M3FN>> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
%3 = tt.addptr %2, %w_24 : tensor<64x128x!tt.ptr<f8E4M3FN>>, tensor<64x128xi64>
%4 = tt.fp_to_fp %q_33, rounding = rtne : tensor<64x128xf32> -> tensor<64x128xf8E4M3FN>
tt.store %3, %4, %mask_18 : tensor<64x128x!tt.ptr<f8E4M3FN>>
%mask_t = tt.expand_dims %offs_n_8 {axis = 1 : i32} : tensor<128xi64> -> tensor<128x1xi64>
%mask_t_34 = tt.splat %mask_13 : i64 -> tensor<128x1xi64>
%mask_t_35 = arith.cmpi slt, %mask_t, %mask_t_34 : tensor<128x1xi64>
%mask_t_36 = tt.expand_dims %offs_m_4 {axis = 0 : i32} : tensor<64xi64> -> tensor<1x64xi64>
%mask_t_37 = tt.splat %mask_9 : i64 -> tensor<1x64xi64>
%mask_t_38 = arith.cmpi slt, %mask_t_36, %mask_t_37 : tensor<1x64xi64>
%mask_t_39 = tt.broadcast %mask_t_35 : tensor<128x1xi1> -> tensor<128x64xi1>
%mask_t_40 = tt.broadcast %mask_t_38 : tensor<1x64xi1> -> tensor<128x64xi1>
%mask_t_41 = arith.andi %mask_t_39, %mask_t_40 : tensor<128x64xi1>
%5 = tt.splat %mask_9 : i64 -> tensor<128x1xi64>
%6 = arith.muli %mask_t, %5 : tensor<128x1xi64>
%7 = tt.splat %COL : !tt.ptr<f8E4M3FN> -> tensor<128x1x!tt.ptr<f8E4M3FN>>
%8 = tt.addptr %7, %6 : tensor<128x1x!tt.ptr<f8E4M3FN>>, tensor<128x1xi64>
%9 = tt.broadcast %8 : tensor<128x1x!tt.ptr<f8E4M3FN>> -> tensor<128x64x!tt.ptr<f8E4M3FN>>
%10 = tt.broadcast %mask_t_36 : tensor<1x64xi64> -> tensor<128x64xi64>
%11 = tt.addptr %9, %10 : tensor<128x64x!tt.ptr<f8E4M3FN>>, tensor<128x64xi64>
%12 = tt.trans %q_33 {order = array<i32: 1, 0>} : tensor<64x128xf32> -> tensor<128x64xf32>
%13 = tt.fp_to_fp %12, rounding = rtne : tensor<128x64xf32> -> tensor<128x64xf8E4M3FN>
tt.store %11, %13, %mask_t_41 : tensor<128x64x!tt.ptr<f8E4M3FN>>
tt.return
}
}
"#

private def referencePtx : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _quantize_lm_head_dual_kernel(
.param .u64 .ptr .global .align 1 _quantize_lm_head_dual_kernel_param_0,
.param .u64 .ptr .global .align 1 _quantize_lm_head_dual_kernel_param_1,
.param .u64 .ptr .global .align 1 _quantize_lm_head_dual_kernel_param_2,
.param .u64 .ptr .global .align 1 _quantize_lm_head_dual_kernel_param_3,
.param .u32 _quantize_lm_head_dual_kernel_param_4,
.param .u32 _quantize_lm_head_dual_kernel_param_5,
.param .u64 .ptr .global .align 1 _quantize_lm_head_dual_kernel_param_6,
.param .u64 .ptr .global .align 1 _quantize_lm_head_dual_kernel_param_7
)
.reqntid 256
{
.reg .pred %p<11>;
.reg .b16 %rs<129>;
.reg .b32 %r<367>;
.reg .b64 %rd<29>;
$L__func_begin0:
ld.param.b64 %rd10, [_quantize_lm_head_dual_kernel_param_0];
ld.param.b64 %rd11, [_quantize_lm_head_dual_kernel_param_1];
$L__tmp0:
mov.u32 %r99, %ctaid.x;
ld.param.b64 %rd12, [_quantize_lm_head_dual_kernel_param_2];
mov.u32 %r100, %ctaid.y;
shl.b32 %r101, %r99, 6;
ld.param.b64 %rd1, [_quantize_lm_head_dual_kernel_param_3];
ld.param.b32 %r102, [_quantize_lm_head_dual_kernel_param_4];
mov.u32 %r103, %tid.x;
ld.param.b32 %r104, [_quantize_lm_head_dual_kernel_param_5];
bfe.u32 %r105, %r103, 3, 5;
shl.b32 %r106, %r103, 4;
and.b32 %r107, %r106, 48;
or.b32 %r108, %r105, %r101;
or.b32 %r109, %r108, 32;
or.b32 %r110, %r107, %r101;
cvt.s64.s32 %rd13, %r110;
shl.b32 %r111, %r100, 7;
and.b32 %r112, %r103, 7;
shl.b32 %r113, %r112, 4;
shr.u32 %r114, %r103, 2;
bfe.u32 %r115, %r103, 2, 6;
or.b32 %r116, %r113, %r111;
or.b32 %r117, %r115, %r111;
or.b32 %r118, %r117, 64;
cvt.u64.u32 %rd14, %r116;
cvt.u64.u32 %rd15, %r117;
cvt.u64.u32 %rd16, %r118;
cvt.s64.s32 %rd17, %r102;
setp.lt.s32 %p5, %r108, %r102;
setp.lt.s32 %p6, %r109, %r102;
setp.lt.s32 %p7, %r116, %r104;
and.pred %p1, %p5, %p7;
and.pred %p2, %p6, %p7;
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd1 + 0 ];
mul.wide.s32 %rd18, %r104, %r108;
mul.wide.s32 %rd19, %r104, %r109;
shl.b64 %rd20, %rd18, 1;
add.s64 %rd21, %rd10, %rd20;
shl.b64 %rd22, %rd19, 1;
add.s64 %rd23, %rd10, %rd22;
mul.wide.u32 %rd24, %r116, 2;
add.s64 %rd2, %rd21, %rd24;
add.s64 %rd3, %rd2, 16;
add.s64 %rd4, %rd23, %rd24;
add.s64 %rd5, %rd4, 16;
mov.b32 %r6, 0;
mov.u32 %r2, %r6;
mov.u32 %r3, %r6;
mov.u32 %r4, %r6;
mov.u32 %r5, %r6;
@%p1 ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd2 + 0 ];
mov.b32 {%rs33, %rs34}, %r2;
mov.b32 {%rs35, %rs36}, %r3;
mov.b32 {%rs37, %rs38}, %r4;
mov.b32 {%rs39, %rs40}, %r5;
mov.u32 %r7, %r6;
mov.u32 %r8, %r6;
mov.u32 %r9, %r6;
mov.u32 %r10, %r6;
@%p1 ld.global.v4.b32 { %r7, %r8, %r9, %r10 }, [ %rd3 + 0 ];
mov.b32 {%rs41, %rs42}, %r7;
mov.b32 {%rs43, %rs44}, %r8;
mov.b32 {%rs45, %rs46}, %r9;
mov.b32 {%rs47, %rs48}, %r10;
mov.u32 %r11, %r6;
mov.u32 %r12, %r6;
mov.u32 %r13, %r6;
mov.u32 %r14, %r6;
@%p2 ld.global.v4.b32 { %r11, %r12, %r13, %r14 }, [ %rd4 + 0 ];
mov.b32 {%rs49, %rs50}, %r11;
mov.b32 {%rs51, %rs52}, %r12;
mov.b32 {%rs53, %rs54}, %r13;
mov.b32 {%rs55, %rs56}, %r14;
mov.u32 %r15, %r6;
mov.u32 %r16, %r6;
mov.u32 %r17, %r6;
mov.u32 %r18, %r6;
@%p2 ld.global.v4.b32 { %r15, %r16, %r17, %r18 }, [ %rd5 + 0 ];
mov.b32 {%rs57, %rs58}, %r15;
mov.b32 {%rs59, %rs60}, %r16;
mov.b32 {%rs61, %rs62}, %r17;
mov.b32 {%rs63, %rs64}, %r18;
cvt.f32.bf16 %r119, %rs33;
cvt.f32.bf16 %r120, %rs34;
cvt.f32.bf16 %r121, %rs35;
cvt.f32.bf16 %r122, %rs36;
cvt.f32.bf16 %r123, %rs37;
cvt.f32.bf16 %r124, %rs38;
cvt.f32.bf16 %r125, %rs39;
cvt.f32.bf16 %r126, %rs40;
cvt.f32.bf16 %r127, %rs41;
cvt.f32.bf16 %r128, %rs42;
cvt.f32.bf16 %r129, %rs43;
cvt.f32.bf16 %r130, %rs44;
cvt.f32.bf16 %r131, %rs45;
cvt.f32.bf16 %r132, %rs46;
cvt.f32.bf16 %r133, %rs47;
cvt.f32.bf16 %r134, %rs48;
cvt.f32.bf16 %r135, %rs49;
cvt.f32.bf16 %r136, %rs50;
cvt.f32.bf16 %r137, %rs51;
cvt.f32.bf16 %r138, %rs52;
cvt.f32.bf16 %r139, %rs53;
cvt.f32.bf16 %r140, %rs54;
cvt.f32.bf16 %r141, %rs55;
cvt.f32.bf16 %r142, %rs56;
cvt.f32.bf16 %r143, %rs57;
cvt.f32.bf16 %r144, %rs58;
cvt.f32.bf16 %r145, %rs59;
cvt.f32.bf16 %r146, %rs60;
cvt.f32.bf16 %r147, %rs61;
cvt.f32.bf16 %r148, %rs62;
cvt.f32.bf16 %r149, %rs63;
cvt.f32.bf16 %r150, %rs64;
mul.f32 %r151, %r1, %r119;
mul.f32 %r152, %r1, %r120;
mul.f32 %r153, %r1, %r121;
mul.f32 %r154, %r1, %r122;
mul.f32 %r155, %r1, %r123;
mul.f32 %r156, %r1, %r124;
mul.f32 %r157, %r1, %r125;
mul.f32 %r158, %r1, %r126;
mul.f32 %r159, %r1, %r127;
mul.f32 %r160, %r1, %r128;
mul.f32 %r161, %r1, %r129;
mul.f32 %r162, %r1, %r130;
mul.f32 %r163, %r1, %r131;
mul.f32 %r164, %r1, %r132;
mul.f32 %r165, %r1, %r133;
mul.f32 %r166, %r1, %r134;
mul.f32 %r167, %r1, %r135;
mul.f32 %r168, %r1, %r136;
mul.f32 %r169, %r1, %r137;
mul.f32 %r170, %r1, %r138;
mul.f32 %r171, %r1, %r139;
mul.f32 %r172, %r1, %r140;
mul.f32 %r173, %r1, %r141;
mul.f32 %r174, %r1, %r142;
mul.f32 %r175, %r1, %r143;
mul.f32 %r176, %r1, %r144;
mul.f32 %r177, %r1, %r145;
mul.f32 %r178, %r1, %r146;
mul.f32 %r179, %r1, %r147;
mul.f32 %r180, %r1, %r148;
mul.f32 %r181, %r1, %r149;
mul.f32 %r182, %r1, %r150;
cvt.rn.bf16.f32 %rs65, %r151;
cvt.rn.bf16.f32 %rs66, %r152;
cvt.rn.bf16.f32 %rs67, %r153;
cvt.rn.bf16.f32 %rs68, %r154;
cvt.rn.bf16.f32 %rs69, %r155;
cvt.rn.bf16.f32 %rs70, %r156;
cvt.rn.bf16.f32 %rs71, %r157;
cvt.rn.bf16.f32 %rs72, %r158;
cvt.rn.bf16.f32 %rs73, %r159;
cvt.rn.bf16.f32 %rs74, %r160;
cvt.rn.bf16.f32 %rs75, %r161;
cvt.rn.bf16.f32 %rs76, %r162;
cvt.rn.bf16.f32 %rs77, %r163;
cvt.rn.bf16.f32 %rs78, %r164;
cvt.rn.bf16.f32 %rs79, %r165;
cvt.rn.bf16.f32 %rs80, %r166;
cvt.rn.bf16.f32 %rs81, %r167;
cvt.rn.bf16.f32 %rs82, %r168;
cvt.rn.bf16.f32 %rs83, %r169;
cvt.rn.bf16.f32 %rs84, %r170;
cvt.rn.bf16.f32 %rs85, %r171;
cvt.rn.bf16.f32 %rs86, %r172;
cvt.rn.bf16.f32 %rs87, %r173;
cvt.rn.bf16.f32 %rs88, %r174;
cvt.rn.bf16.f32 %rs89, %r175;
cvt.rn.bf16.f32 %rs90, %r176;
cvt.rn.bf16.f32 %rs91, %r177;
cvt.rn.bf16.f32 %rs92, %r178;
cvt.rn.bf16.f32 %rs93, %r179;
cvt.rn.bf16.f32 %rs94, %r180;
cvt.rn.bf16.f32 %rs95, %r181;
cvt.rn.bf16.f32 %rs96, %r182;
shl.b32 %r183, %r103, 3;
and.b32 %r184, %r183, 896;
bfe.s32 %r185, %r103, 3, 1;
and.b32 %r186, %r185, 2064;
bfe.s32 %r187, %r103, 7, 1;
and.b32 %r188, %r187, 4128;
or.b32 %r189, %r113, %r184;
or.b32 %r190, %r186, %r188;
xor.b32 %r191, %r190, %r189;
mov.b32 %r192, global_smem;
add.s32 %r193, %r192, %r191;
mov.b32 %r194, {%rs71, %rs72};
mov.b32 %r195, {%rs69, %rs70};
mov.b32 %r196, {%rs67, %rs68};
mov.b32 %r197, {%rs65, %rs66};
st.shared.v4.b32 [%r193], {%r197, %r196, %r195, %r194};
mov.b32 %r198, {%rs79, %rs80};
mov.b32 %r199, {%rs77, %rs78};
mov.b32 %r200, {%rs75, %rs76};
mov.b32 %r201, {%rs73, %rs74};
st.shared.v4.b32 [%r193+1024], {%r201, %r200, %r199, %r198};
xor.b32 %r202, %r191, 64;
add.s32 %r203, %r192, %r202;
mov.b32 %r204, {%rs87, %rs88};
mov.b32 %r205, {%rs85, %rs86};
mov.b32 %r206, {%rs83, %rs84};
mov.b32 %r207, {%rs81, %rs82};
st.shared.v4.b32 [%r203+8192], {%r207, %r206, %r205, %r204};
mov.b32 %r208, {%rs95, %rs96};
mov.b32 %r209, {%rs93, %rs94};
mov.b32 %r210, {%rs91, %rs92};
mov.b32 %r211, {%rs89, %rs90};
st.shared.v4.b32 [%r203+9216], {%r211, %r210, %r209, %r208};
bar.sync 0;
shl.b32 %r212, %r112, 11;
and.b32 %r213, %r106, 496;
and.b32 %r214, %r114, 48;
shl.b32 %r215, %r103, 5;
and.b32 %r216, %r215, 1024;
or.b32 %r217, %r212, %r216;
xor.b32 %r218, %r213, %r214;
or.b32 %r219, %r217, %r218;
add.s32 %r220, %r192, %r219;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r221, %r222, %r223, %r224}, [%r220];
mov.b32 {%rs97, %rs98}, %r221;
mov.b32 {%rs99, %rs100}, %r222;
mov.b32 {%rs101, %rs102}, %r223;
mov.b32 {%rs103, %rs104}, %r224;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r225, %r226, %r227, %r228}, [%r220+512];
mov.b32 {%rs105, %rs106}, %r225;
mov.b32 {%rs107, %rs108}, %r226;
mov.b32 {%rs109, %rs110}, %r227;
mov.b32 {%rs111, %rs112}, %r228;
xor.b32 %r229, %r219, 64;
add.s32 %r230, %r192, %r229;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r231, %r232, %r233, %r234}, [%r230];
mov.b32 {%rs113, %rs114}, %r231;
mov.b32 {%rs115, %rs116}, %r232;
mov.b32 {%rs117, %rs118}, %r233;
mov.b32 {%rs119, %rs120}, %r234;
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%r235, %r236, %r237, %r238}, [%r230+512];
mov.b32 {%rs121, %rs122}, %r235;
mov.b32 {%rs123, %rs124}, %r236;
mov.b32 {%rs125, %rs126}, %r237;
mov.b32 {%rs127, %rs128}, %r238;
cvt.f32.bf16 %r239, %rs97;
cvt.f32.bf16 %r240, %rs98;
cvt.f32.bf16 %r241, %rs99;
cvt.f32.bf16 %r242, %rs100;
cvt.f32.bf16 %r243, %rs101;
cvt.f32.bf16 %r244, %rs102;
cvt.f32.bf16 %r245, %rs103;
cvt.f32.bf16 %r246, %rs104;
cvt.f32.bf16 %r247, %rs105;
cvt.f32.bf16 %r248, %rs106;
cvt.f32.bf16 %r249, %rs107;
cvt.f32.bf16 %r250, %rs108;
cvt.f32.bf16 %r251, %rs109;
cvt.f32.bf16 %r252, %rs110;
cvt.f32.bf16 %r253, %rs111;
cvt.f32.bf16 %r254, %rs112;
cvt.f32.bf16 %r255, %rs113;
cvt.f32.bf16 %r256, %rs114;
cvt.f32.bf16 %r257, %rs115;
cvt.f32.bf16 %r258, %rs116;
cvt.f32.bf16 %r259, %rs117;
cvt.f32.bf16 %r260, %rs118;
cvt.f32.bf16 %r261, %rs119;
cvt.f32.bf16 %r262, %rs120;
cvt.f32.bf16 %r263, %rs121;
cvt.f32.bf16 %r264, %rs122;
cvt.f32.bf16 %r265, %rs123;
cvt.f32.bf16 %r266, %rs124;
cvt.f32.bf16 %r267, %rs125;
cvt.f32.bf16 %r268, %rs126;
cvt.f32.bf16 %r269, %rs127;
cvt.f32.bf16 %r270, %rs128;
cvt.f32.bf16 %r271, %rs65;
cvt.f32.bf16 %r272, %rs66;
cvt.f32.bf16 %r273, %rs67;
cvt.f32.bf16 %r274, %rs68;
cvt.f32.bf16 %r275, %rs69;
cvt.f32.bf16 %r276, %rs70;
cvt.f32.bf16 %r277, %rs71;
cvt.f32.bf16 %r278, %rs72;
cvt.f32.bf16 %r279, %rs73;
cvt.f32.bf16 %r280, %rs74;
cvt.f32.bf16 %r281, %rs75;
cvt.f32.bf16 %r282, %rs76;
cvt.f32.bf16 %r283, %rs77;
cvt.f32.bf16 %r284, %rs78;
cvt.f32.bf16 %r285, %rs79;
cvt.f32.bf16 %r286, %rs80;
cvt.f32.bf16 %r287, %rs81;
cvt.f32.bf16 %r288, %rs82;
cvt.f32.bf16 %r289, %rs83;
cvt.f32.bf16 %r290, %rs84;
cvt.f32.bf16 %r291, %rs85;
cvt.f32.bf16 %r292, %rs86;
cvt.f32.bf16 %r293, %rs87;
cvt.f32.bf16 %r294, %rs88;
cvt.f32.bf16 %r295, %rs89;
cvt.f32.bf16 %r296, %rs90;
cvt.f32.bf16 %r297, %rs91;
cvt.f32.bf16 %r298, %rs92;
cvt.f32.bf16 %r299, %rs93;
cvt.f32.bf16 %r300, %rs94;
cvt.f32.bf16 %r301, %rs95;
cvt.f32.bf16 %r302, %rs96;
min.f32 %r303, %r239, 0f43E00000;
min.f32 %r304, %r240, 0f43E00000;
min.f32 %r305, %r241, 0f43E00000;
min.f32 %r306, %r242, 0f43E00000;
min.f32 %r307, %r243, 0f43E00000;
min.f32 %r308, %r244, 0f43E00000;
min.f32 %r309, %r245, 0f43E00000;
min.f32 %r310, %r246, 0f43E00000;
min.f32 %r311, %r247, 0f43E00000;
min.f32 %r312, %r248, 0f43E00000;
min.f32 %r313, %r249, 0f43E00000;
min.f32 %r314, %r250, 0f43E00000;
min.f32 %r315, %r251, 0f43E00000;
min.f32 %r316, %r252, 0f43E00000;
min.f32 %r317, %r253, 0f43E00000;
min.f32 %r318, %r254, 0f43E00000;
min.f32 %r319, %r255, 0f43E00000;
min.f32 %r320, %r256, 0f43E00000;
min.f32 %r321, %r257, 0f43E00000;
min.f32 %r322, %r258, 0f43E00000;
min.f32 %r323, %r259, 0f43E00000;
min.f32 %r324, %r260, 0f43E00000;
min.f32 %r325, %r261, 0f43E00000;
min.f32 %r326, %r262, 0f43E00000;
min.f32 %r327, %r263, 0f43E00000;
min.f32 %r328, %r264, 0f43E00000;
min.f32 %r329, %r265, 0f43E00000;
min.f32 %r330, %r266, 0f43E00000;
min.f32 %r331, %r267, 0f43E00000;
min.f32 %r332, %r268, 0f43E00000;
min.f32 %r333, %r269, 0f43E00000;
min.f32 %r334, %r270, 0f43E00000;
min.f32 %r335, %r271, 0f43E00000;
min.f32 %r336, %r272, 0f43E00000;
min.f32 %r337, %r273, 0f43E00000;
min.f32 %r338, %r274, 0f43E00000;
min.f32 %r339, %r275, 0f43E00000;
min.f32 %r340, %r276, 0f43E00000;
min.f32 %r341, %r277, 0f43E00000;
min.f32 %r342, %r278, 0f43E00000;
min.f32 %r343, %r279, 0f43E00000;
min.f32 %r344, %r280, 0f43E00000;
min.f32 %r345, %r281, 0f43E00000;
min.f32 %r346, %r282, 0f43E00000;
min.f32 %r347, %r283, 0f43E00000;
min.f32 %r348, %r284, 0f43E00000;
min.f32 %r349, %r285, 0f43E00000;
min.f32 %r350, %r286, 0f43E00000;
min.f32 %r351, %r287, 0f43E00000;
min.f32 %r352, %r288, 0f43E00000;
min.f32 %r353, %r289, 0f43E00000;
min.f32 %r354, %r290, 0f43E00000;
min.f32 %r355, %r291, 0f43E00000;
min.f32 %r356, %r292, 0f43E00000;
min.f32 %r357, %r293, 0f43E00000;
min.f32 %r358, %r294, 0f43E00000;
min.f32 %r359, %r295, 0f43E00000;
min.f32 %r360, %r296, 0f43E00000;
min.f32 %r361, %r297, 0f43E00000;
min.f32 %r362, %r298, 0f43E00000;
min.f32 %r363, %r299, 0f43E00000;
min.f32 %r364, %r300, 0f43E00000;
min.f32 %r365, %r301, 0f43E00000;
min.f32 %r366, %r302, 0f43E00000;
max.f32 %r59, %r303, 0fC3E00000;
max.f32 %r60, %r304, 0fC3E00000;
max.f32 %r61, %r305, 0fC3E00000;
max.f32 %r62, %r306, 0fC3E00000;
max.f32 %r63, %r307, 0fC3E00000;
max.f32 %r64, %r308, 0fC3E00000;
max.f32 %r65, %r309, 0fC3E00000;
max.f32 %r66, %r310, 0fC3E00000;
max.f32 %r67, %r311, 0fC3E00000;
max.f32 %r68, %r312, 0fC3E00000;
max.f32 %r69, %r313, 0fC3E00000;
max.f32 %r70, %r314, 0fC3E00000;
max.f32 %r71, %r315, 0fC3E00000;
max.f32 %r72, %r316, 0fC3E00000;
max.f32 %r73, %r317, 0fC3E00000;
max.f32 %r74, %r318, 0fC3E00000;
max.f32 %r75, %r319, 0fC3E00000;
max.f32 %r76, %r320, 0fC3E00000;
max.f32 %r77, %r321, 0fC3E00000;
max.f32 %r78, %r322, 0fC3E00000;
max.f32 %r79, %r323, 0fC3E00000;
max.f32 %r80, %r324, 0fC3E00000;
max.f32 %r81, %r325, 0fC3E00000;
max.f32 %r82, %r326, 0fC3E00000;
max.f32 %r83, %r327, 0fC3E00000;
max.f32 %r84, %r328, 0fC3E00000;
max.f32 %r85, %r329, 0fC3E00000;
max.f32 %r86, %r330, 0fC3E00000;
max.f32 %r87, %r331, 0fC3E00000;
max.f32 %r88, %r332, 0fC3E00000;
max.f32 %r89, %r333, 0fC3E00000;
max.f32 %r90, %r334, 0fC3E00000;
max.f32 %r19, %r335, 0fC3E00000;
max.f32 %r20, %r336, 0fC3E00000;
max.f32 %r21, %r337, 0fC3E00000;
max.f32 %r22, %r338, 0fC3E00000;
max.f32 %r23, %r339, 0fC3E00000;
max.f32 %r24, %r340, 0fC3E00000;
max.f32 %r25, %r341, 0fC3E00000;
max.f32 %r26, %r342, 0fC3E00000;
max.f32 %r27, %r343, 0fC3E00000;
max.f32 %r28, %r344, 0fC3E00000;
max.f32 %r29, %r345, 0fC3E00000;
max.f32 %r30, %r346, 0fC3E00000;
max.f32 %r31, %r347, 0fC3E00000;
max.f32 %r32, %r348, 0fC3E00000;
max.f32 %r33, %r349, 0fC3E00000;
max.f32 %r34, %r350, 0fC3E00000;
max.f32 %r35, %r351, 0fC3E00000;
max.f32 %r36, %r352, 0fC3E00000;
max.f32 %r37, %r353, 0fC3E00000;
max.f32 %r38, %r354, 0fC3E00000;
max.f32 %r39, %r355, 0fC3E00000;
max.f32 %r40, %r356, 0fC3E00000;
max.f32 %r41, %r357, 0fC3E00000;
max.f32 %r42, %r358, 0fC3E00000;
max.f32 %r43, %r359, 0fC3E00000;
max.f32 %r44, %r360, 0fC3E00000;
max.f32 %r45, %r361, 0fC3E00000;
max.f32 %r46, %r362, 0fC3E00000;
max.f32 %r47, %r363, 0fC3E00000;
max.f32 %r48, %r364, 0fC3E00000;
max.f32 %r49, %r365, 0fC3E00000;
max.f32 %r50, %r366, 0fC3E00000;
add.s64 %rd25, %rd11, %rd18;
add.s64 %rd26, %rd11, %rd19;
add.s64 %rd6, %rd25, %rd14;
add.s64 %rd7, %rd26, %rd14;
cvt.rn.satfinite.e4m3x2.f32 %rs1, %r20, %r19;
cvt.rn.satfinite.e4m3x2.f32 %rs2, %r22, %r21;
cvt.rn.satfinite.e4m3x2.f32 %rs3, %r24, %r23;
cvt.rn.satfinite.e4m3x2.f32 %rs4, %r26, %r25;
cvt.rn.satfinite.e4m3x2.f32 %rs5, %r28, %r27;
cvt.rn.satfinite.e4m3x2.f32 %rs6, %r30, %r29;
cvt.rn.satfinite.e4m3x2.f32 %rs7, %r32, %r31;
cvt.rn.satfinite.e4m3x2.f32 %rs8, %r34, %r33;
cvt.rn.satfinite.e4m3x2.f32 %rs9, %r36, %r35;
cvt.rn.satfinite.e4m3x2.f32 %rs10, %r38, %r37;
cvt.rn.satfinite.e4m3x2.f32 %rs11, %r40, %r39;
cvt.rn.satfinite.e4m3x2.f32 %rs12, %r42, %r41;
cvt.rn.satfinite.e4m3x2.f32 %rs13, %r44, %r43;
cvt.rn.satfinite.e4m3x2.f32 %rs14, %r46, %r45;
cvt.rn.satfinite.e4m3x2.f32 %rs15, %r48, %r47;
cvt.rn.satfinite.e4m3x2.f32 %rs16, %r50, %r49;
mov.b32 %r51, {%rs1, %rs2};
mov.b32 %r52, {%rs3, %rs4};
mov.b32 %r53, {%rs5, %rs6};
mov.b32 %r54, {%rs7, %rs8};
@%p1 st.global.v4.b32 [ %rd6 + 0 ], { %r51, %r52, %r53, %r54 };
mov.b32 %r55, {%rs9, %rs10};
mov.b32 %r56, {%rs11, %rs12};
mov.b32 %r57, {%rs13, %rs14};
mov.b32 %r58, {%rs15, %rs16};
@%p2 st.global.v4.b32 [ %rd7 + 0 ], { %r55, %r56, %r57, %r58 };
setp.lt.s32 %p8, %r117, %r104;
setp.lt.s32 %p9, %r118, %r104;
setp.lt.s32 %p10, %r110, %r102;
and.pred %p3, %p10, %p8;
and.pred %p4, %p10, %p9;
mad.lo.s64 %rd27, %rd17, %rd15, %rd12;
mad.lo.s64 %rd28, %rd17, %rd16, %rd12;
add.s64 %rd8, %rd27, %rd13;
add.s64 %rd9, %rd28, %rd13;
cvt.rn.satfinite.e4m3x2.f32 %rs17, %r60, %r59;
cvt.rn.satfinite.e4m3x2.f32 %rs18, %r62, %r61;
cvt.rn.satfinite.e4m3x2.f32 %rs19, %r64, %r63;
cvt.rn.satfinite.e4m3x2.f32 %rs20, %r66, %r65;
cvt.rn.satfinite.e4m3x2.f32 %rs21, %r68, %r67;
cvt.rn.satfinite.e4m3x2.f32 %rs22, %r70, %r69;
cvt.rn.satfinite.e4m3x2.f32 %rs23, %r72, %r71;
cvt.rn.satfinite.e4m3x2.f32 %rs24, %r74, %r73;
cvt.rn.satfinite.e4m3x2.f32 %rs25, %r76, %r75;
cvt.rn.satfinite.e4m3x2.f32 %rs26, %r78, %r77;
cvt.rn.satfinite.e4m3x2.f32 %rs27, %r80, %r79;
cvt.rn.satfinite.e4m3x2.f32 %rs28, %r82, %r81;
cvt.rn.satfinite.e4m3x2.f32 %rs29, %r84, %r83;
cvt.rn.satfinite.e4m3x2.f32 %rs30, %r86, %r85;
cvt.rn.satfinite.e4m3x2.f32 %rs31, %r88, %r87;
cvt.rn.satfinite.e4m3x2.f32 %rs32, %r90, %r89;
mov.b32 %r91, {%rs17, %rs18};
mov.b32 %r92, {%rs19, %rs20};
mov.b32 %r93, {%rs21, %rs22};
mov.b32 %r94, {%rs23, %rs24};
@%p3 st.global.v4.b32 [ %rd8 + 0 ], { %r91, %r92, %r93, %r94 };
mov.b32 %r95, {%rs25, %rs26};
mov.b32 %r96, {%rs27, %rs28};
mov.b32 %r97, {%rs29, %rs30};
mov.b32 %r98, {%rs31, %rs32};
@%p4 st.global.v4.b32 [ %rd9 + 0 ], { %r95, %r96, %r97, %r98 };
ret;
$L__tmp1:
$L__func_end0:
}
"#

private def hopperGraph : String := r#"
#blocked = #ttg.blocked<{sizePerThread = [2, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 4], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
tt.func public @_quantize_lm_head_gluon(%W: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %ROW: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %COL: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %INV: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
%cst = arith.constant dense<50304> : tensor<64x1xi32, #blocked>
%cst_0 = arith.constant dense<768> : tensor<1x128xi32, #blocked1>
%inv = tt.load %INV : !tt.ptr<f32>
%0 = tt.bitcast %inv : f32 -> i32
%1 = arith.cmpi eq, %0, 1130364927 : i32
scf.if %1 {
%pid_m = tt.get_program_id y : i32
%pid_n = tt.get_program_id x : i32
%offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%base = arith.muli %pid_m, 3219456 : i32
%base_1 = arith.muli %pid_n, 128 : i32
%base_2 = arith.addi %base, %base_1 : i32
%col_base = arith.muli %pid_n, 98304 : i32
%col_base_3 = arith.muli %pid_m, 64 : i32
%col_base_4 = arith.addi %col_base, %col_base_3 : i32
%w = tt.addptr %W, %base_2 : !tt.ptr<bf16>, i32
%w_5 = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%w_6 = arith.muli %w_5, %cst : tensor<64x1xi32, #blocked>
%w_7 = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%w_8 = tt.broadcast %w_6 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
%w_9 = tt.broadcast %w_7 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%w_10 = arith.addi %w_8, %w_9 : tensor<64x128xi32, #blocked>
%w_11 = tt.splat %w : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%w_12 = tt.addptr %w_11, %w_10 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%w_13 = tt.load %w_12 : tensor<64x128x!tt.ptr<bf16>, #blocked>
%bounded = tt.elementwise_inline_asm "{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 $0, $1, gain; min.bf16x2 $0, $0, upper; }" {constraints = "=r,r", packed_element = 2 : i32, pure = true} %w_13 : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #blocked>
%q = tt.elementwise_inline_asm "{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, $1; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 $0, f16s; }" {constraints = "=h,r", packed_element = 2 : i32, pure = true} %bounded : tensor<64x128xbf16, #blocked> -> tensor<64x128xf8E4M3FN, #blocked>
%2 = tt.addptr %ROW, %base_2 : !tt.ptr<f8E4M3FN>, i32
%3 = tt.splat %2 : !tt.ptr<f8E4M3FN> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%4 = tt.addptr %3, %w_10 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x128xi32, #blocked>
tt.store %4, %q : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%moved = ttg.convert_layout %q : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #blocked1>
%trans_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%trans_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%5 = tt.addptr %COL, %col_base_4 : !tt.ptr<f8E4M3FN>, i32
%6 = tt.expand_dims %trans_n {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%7 = arith.muli %6, %cst_0 : tensor<1x128xi32, #blocked1>
%8 = tt.expand_dims %trans_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
%9 = tt.broadcast %7 : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%10 = tt.broadcast %8 : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%11 = arith.addi %9, %10 : tensor<64x128xi32, #blocked1>
%12 = tt.splat %5 : !tt.ptr<f8E4M3FN> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
%13 = tt.addptr %12, %11 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<64x128xi32, #blocked1>
tt.store %13, %moved : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
} else {
%pid_m = tt.get_program_id y : i32
%pid_n = tt.get_program_id x : i32
%offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
%offs_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
%base = arith.muli %pid_m, 3219456 : i32
%base_1 = arith.muli %pid_n, 128 : i32
%base_2 = arith.addi %base, %base_1 : i32
%col_base = arith.muli %pid_n, 98304 : i32
%col_base_3 = arith.muli %pid_m, 64 : i32
%col_base_4 = arith.addi %col_base, %col_base_3 : i32
%w = tt.addptr %W, %base_2 : !tt.ptr<bf16>, i32
%w_5 = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
%w_6 = arith.muli %w_5, %cst : tensor<64x1xi32, #blocked>
%w_7 = tt.expand_dims %offs_n {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
%w_8 = tt.broadcast %w_6 : tensor<64x1xi32, #blocked> -> tensor<64x128xi32, #blocked>
%w_9 = tt.broadcast %w_7 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
%w_10 = arith.addi %w_8, %w_9 : tensor<64x128xi32, #blocked>
%w_11 = tt.splat %w : !tt.ptr<bf16> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
%w_12 = tt.addptr %w_11, %w_10 : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
%w_13 = tt.load %w_12 : tensor<64x128x!tt.ptr<bf16>, #blocked>
%scaled = arith.extf %w_13 : tensor<64x128xbf16, #blocked> to tensor<64x128xf32, #blocked>
%scaled_14 = tt.splat %inv : f32 -> tensor<64x128xf32, #blocked>
%scaled_15 = arith.mulf %scaled, %scaled_14 : tensor<64x128xf32, #blocked>
%bounded = tt.elementwise_inline_asm "{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, $2, $1; min.bf16x2 $0, rounded, upper; }" {constraints = "=r,f,f", packed_element = 2 : i32, pure = true} %scaled_15 : tensor<64x128xf32, #blocked> -> tensor<64x128xbf16, #blocked>
%q = tt.elementwise_inline_asm "{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, $1; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 $0, f16s; }" {constraints = "=h,r", packed_element = 2 : i32, pure = true} %bounded : tensor<64x128xbf16, #blocked> -> tensor<64x128xf8E4M3FN, #blocked>
%2 = tt.addptr %ROW, %base_2 : !tt.ptr<f8E4M3FN>, i32
%3 = tt.splat %2 : !tt.ptr<f8E4M3FN> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%4 = tt.addptr %3, %w_10 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x128xi32, #blocked>
tt.store %4, %q : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked>
%moved = ttg.convert_layout %q : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #blocked1>
%trans_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
%trans_n = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
%5 = tt.addptr %COL, %col_base_4 : !tt.ptr<f8E4M3FN>, i32
%6 = tt.expand_dims %trans_n {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
%7 = arith.muli %6, %cst_0 : tensor<1x128xi32, #blocked1>
%8 = tt.expand_dims %trans_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
%9 = tt.broadcast %7 : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%10 = tt.broadcast %8 : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
%11 = arith.addi %9, %10 : tensor<64x128xi32, #blocked1>
%12 = tt.splat %5 : !tt.ptr<f8E4M3FN> -> tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
%13 = tt.addptr %12, %11 : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>, tensor<64x128xi32, #blocked1>
tt.store %13, %moved : tensor<64x128x!tt.ptr<f8E4M3FN>, #blocked1>
}
tt.return
}
}
"#

private def hopperPtx : String := r#"
.version 8.7
.target sm_90a
.address_size 64
.extern .shared .align 16 .b8 global_smem[];
.visible .entry _quantize_lm_head_gluon(
.param .u64 .ptr .global .align 1 _quantize_lm_head_gluon_param_0,
.param .u64 .ptr .global .align 1 _quantize_lm_head_gluon_param_1,
.param .u64 .ptr .global .align 1 _quantize_lm_head_gluon_param_2,
.param .u64 .ptr .global .align 1 _quantize_lm_head_gluon_param_3,
.param .u64 .ptr .global .align 1 _quantize_lm_head_gluon_param_4,
.param .u64 .ptr .global .align 1 _quantize_lm_head_gluon_param_5
)
.reqntid 256
{
.reg .pred %p<2>;
.reg .b16 %rs<65>;
.reg .b32 %r<254>;
.reg .b64 %rd<41>;
$L__func_begin0:
ld.param.b64 %rd3, [_quantize_lm_head_gluon_param_2];
ld.param.b64 %rd2, [_quantize_lm_head_gluon_param_1];
ld.param.b64 %rd1, [_quantize_lm_head_gluon_param_0];
ld.param.b64 %rd4, [_quantize_lm_head_gluon_param_3];
$L__tmp0:
mov.u32 %r1, 0x0;
ld.global.b32 { %r1 }, [ %rd4 + 0 ];
setp.ne.b32 %p1, %r1, 1130364927;
@%p1 bra $L__BB0_2;
$L__tmp1:
mov.u32 %r208, %ctaid.y;
mov.u32 %r209, %ctaid.x;
mov.u32 %r210, %tid.x;
shr.u32 %r211, %r210, 2;
and.b32 %r212, %r211, 62;
shl.b32 %r213, %r210, 4;
and.b32 %r214, %r213, 112;
shl.b32 %r215, %r209, 7;
mad.lo.s32 %r216, %r208, 3219456, %r215;
shl.b32 %r217, %r208, 6;
mad.lo.s32 %r218, %r209, 98304, %r217;
cvt.s64.s32 %rd33, %r216;
mad.wide.s32 %rd34, %r216, 2, %rd1;
mul.lo.s32 %r219, %r212, 50304;
or.b32 %r220, %r219, %r214;
cvt.u64.u32 %rd35, %r220;
mad.wide.u32 %rd23, %r220, 2, %rd34;
add.s64 %rd24, %rd23, 16;
add.s64 %rd25, %rd23, 100608;
or.b32 %r221, %r220, 8;
mad.wide.u32 %rd36, %r221, 2, %rd34;
add.s64 %rd26, %rd36, 100608;
mov.u32 %r160, 0x0;
mov.u32 %r161, 0x0;
mov.u32 %r162, 0x0;
mov.u32 %r163, 0x0;
ld.global.v4.b32 { %r160, %r161, %r162, %r163 }, [ %rd23 + 0 ];
mov.u32 %r164, 0x0;
mov.u32 %r165, 0x0;
mov.u32 %r166, 0x0;
mov.u32 %r167, 0x0;
ld.global.v4.b32 { %r164, %r165, %r166, %r167 }, [ %rd24 + 0 ];
mov.u32 %r168, 0x0;
mov.u32 %r169, 0x0;
mov.u32 %r170, 0x0;
mov.u32 %r171, 0x0;
ld.global.v4.b32 { %r168, %r169, %r170, %r171 }, [ %rd25 + 0 ];
mov.u32 %r172, 0x0;
mov.u32 %r173, 0x0;
mov.u32 %r174, 0x0;
mov.u32 %r175, 0x0;
ld.global.v4.b32 { %r172, %r173, %r174, %r175 }, [ %rd26 + 0 ];
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r176, %r160, gain; min.bf16x2 %r176, %r176, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r177, %r161, gain; min.bf16x2 %r177, %r177, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r178, %r162, gain; min.bf16x2 %r178, %r178, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r179, %r163, gain; min.bf16x2 %r179, %r179, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r180, %r164, gain; min.bf16x2 %r180, %r180, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r181, %r165, gain; min.bf16x2 %r181, %r181, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r182, %r166, gain; min.bf16x2 %r182, %r182, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r183, %r167, gain; min.bf16x2 %r183, %r183, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r184, %r168, gain; min.bf16x2 %r184, %r184, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r185, %r169, gain; min.bf16x2 %r185, %r185, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r186, %r170, gain; min.bf16x2 %r186, %r186, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r187, %r171, gain; min.bf16x2 %r187, %r187, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r188, %r172, gain; min.bf16x2 %r188, %r188, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r189, %r173, gain; min.bf16x2 %r189, %r189, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r190, %r174, gain; min.bf16x2 %r190, %r190, upper; }
{ .reg .b32 gain, upper; mov.b32 gain, 0x43604360; mov.b32 upper, 0x43e043e0; mul.rn.bf16x2 %r191, %r175, gain; min.bf16x2 %r191, %r191, upper; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r176; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs49, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r177; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs50, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r178; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs51, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r179; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs52, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r180; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs53, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r181; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs54, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r182; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs55, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r183; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs56, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r184; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs57, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r185; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs58, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r186; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs59, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r187; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs60, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r188; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs61, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r189; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs62, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r190; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs63, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r191; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs64, f16s; }
add.s64 %rd37, %rd2, %rd33;
add.s64 %rd27, %rd37, %rd35;
add.s64 %rd28, %rd27, 50304;
mov.b32 %r192, {%rs49, %rs50};
mov.b32 %r193, {%rs51, %rs52};
mov.b32 %r194, {%rs53, %rs54};
mov.b32 %r195, {%rs55, %rs56};
st.global.v4.b32 [ %rd27 + 0 ], { %r192, %r193, %r194, %r195 };
mov.b32 %r196, {%rs57, %rs58};
mov.b32 %r197, {%rs59, %rs60};
mov.b32 %r198, {%rs61, %rs62};
mov.b32 %r199, {%rs63, %rs64};
st.global.v4.b32 [ %rd28 + 0 ], { %r196, %r197, %r198, %r199 };
and.b32 %r222, %r213, 4080;
mov.b32 %r223, global_smem;
add.s32 %r224, %r223, %r222;
st.shared.v4.b32 [%r224], {%r192, %r193, %r194, %r195};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r225, %r226, %r227, %r228}, [%r224];
bar.warp.sync -1;
st.shared.v4.b32 [%r224], {%r196, %r197, %r198, %r199};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r229, %r230, %r231, %r232}, [%r224];
and.b32 %r233, %r211, 56;
shl.b32 %r234, %r210, 2;
and.b32 %r235, %r234, 124;
cvt.s64.s32 %rd38, %r218;
add.s64 %rd39, %rd3, %rd38;
mul.lo.s32 %r236, %r235, 768;
or.b32 %r237, %r236, %r233;
cvt.u64.u32 %rd40, %r237;
add.s64 %rd29, %rd39, %rd40;
add.s64 %rd30, %rd29, 768;
add.s64 %rd31, %rd29, 1536;
add.s64 %rd32, %rd29, 2304;
prmt.b32 %r238, %r225, %r229, 0x40U;
prmt.b32 %r239, %r238, %r226, 0x410U;
prmt.b32 %r200, %r239, %r230, 0x4210U;
prmt.b32 %r240, %r227, %r231, 0x40U;
prmt.b32 %r241, %r240, %r228, 0x410U;
prmt.b32 %r201, %r241, %r232, 0x4210U;
st.global.v2.b32 [ %rd29 + 0 ], { %r200, %r201 };
prmt.b32 %r242, %r225, %r229, 0x51U;
prmt.b32 %r243, %r242, %r226, 0x510U;
prmt.b32 %r202, %r243, %r230, 0x5210U;
prmt.b32 %r244, %r227, %r231, 0x51U;
prmt.b32 %r245, %r244, %r228, 0x510U;
prmt.b32 %r203, %r245, %r232, 0x5210U;
st.global.v2.b32 [ %rd30 + 0 ], { %r202, %r203 };
prmt.b32 %r246, %r225, %r229, 0x62U;
prmt.b32 %r247, %r246, %r226, 0x610U;
prmt.b32 %r204, %r247, %r230, 0x6210U;
prmt.b32 %r248, %r227, %r231, 0x62U;
prmt.b32 %r249, %r248, %r228, 0x610U;
prmt.b32 %r205, %r249, %r232, 0x6210U;
st.global.v2.b32 [ %rd31 + 0 ], { %r204, %r205 };
prmt.b32 %r250, %r225, %r229, 0x73U;
prmt.b32 %r251, %r250, %r226, 0x710U;
prmt.b32 %r206, %r251, %r230, 0x7210U;
prmt.b32 %r252, %r227, %r231, 0x73U;
prmt.b32 %r253, %r252, %r228, 0x710U;
prmt.b32 %r207, %r253, %r232, 0x7210U;
st.global.v2.b32 [ %rd32 + 0 ], { %r206, %r207 };
$L__tmp2:
bra.uni $L__BB0_3;
$L__BB0_2:
$L__tmp3:
mov.u32 %r82, %ctaid.y;
mov.u32 %r83, %ctaid.x;
mov.u32 %r84, %tid.x;
shr.u32 %r85, %r84, 2;
and.b32 %r86, %r85, 62;
shl.b32 %r87, %r84, 4;
and.b32 %r88, %r87, 112;
shl.b32 %r89, %r83, 7;
mad.lo.s32 %r90, %r82, 3219456, %r89;
shl.b32 %r91, %r82, 6;
mad.lo.s32 %r92, %r83, 98304, %r91;
cvt.s64.s32 %rd15, %r90;
mad.wide.s32 %rd16, %r90, 2, %rd1;
mul.lo.s32 %r93, %r86, 50304;
or.b32 %r94, %r93, %r88;
cvt.u64.u32 %rd17, %r94;
mad.wide.u32 %rd5, %r94, 2, %rd16;
add.s64 %rd6, %rd5, 16;
add.s64 %rd7, %rd5, 100608;
or.b32 %r95, %r94, 8;
mad.wide.u32 %rd18, %r95, 2, %rd16;
add.s64 %rd8, %rd18, 100608;
mov.u32 %r2, 0x0;
mov.u32 %r3, 0x0;
mov.u32 %r4, 0x0;
mov.u32 %r5, 0x0;
ld.global.v4.b32 { %r2, %r3, %r4, %r5 }, [ %rd5 + 0 ];
mov.b32 {%rs17, %rs18}, %r2;
mov.b32 {%rs19, %rs20}, %r3;
mov.b32 {%rs21, %rs22}, %r4;
mov.b32 {%rs23, %rs24}, %r5;
mov.u32 %r6, 0x0;
mov.u32 %r7, 0x0;
mov.u32 %r8, 0x0;
mov.u32 %r9, 0x0;
ld.global.v4.b32 { %r6, %r7, %r8, %r9 }, [ %rd6 + 0 ];
mov.b32 {%rs25, %rs26}, %r6;
mov.b32 {%rs27, %rs28}, %r7;
mov.b32 {%rs29, %rs30}, %r8;
mov.b32 {%rs31, %rs32}, %r9;
mov.u32 %r10, 0x0;
mov.u32 %r11, 0x0;
mov.u32 %r12, 0x0;
mov.u32 %r13, 0x0;
ld.global.v4.b32 { %r10, %r11, %r12, %r13 }, [ %rd7 + 0 ];
mov.b32 {%rs33, %rs34}, %r10;
mov.b32 {%rs35, %rs36}, %r11;
mov.b32 {%rs37, %rs38}, %r12;
mov.b32 {%rs39, %rs40}, %r13;
mov.u32 %r14, 0x0;
mov.u32 %r15, 0x0;
mov.u32 %r16, 0x0;
mov.u32 %r17, 0x0;
ld.global.v4.b32 { %r14, %r15, %r16, %r17 }, [ %rd8 + 0 ];
mov.b32 {%rs41, %rs42}, %r14;
mov.b32 {%rs43, %rs44}, %r15;
mov.b32 {%rs45, %rs46}, %r16;
mov.b32 {%rs47, %rs48}, %r17;
cvt.f32.bf16 %r96, %rs17;
cvt.f32.bf16 %r97, %rs18;
cvt.f32.bf16 %r98, %rs19;
cvt.f32.bf16 %r99, %rs20;
cvt.f32.bf16 %r100, %rs21;
cvt.f32.bf16 %r101, %rs22;
cvt.f32.bf16 %r102, %rs23;
cvt.f32.bf16 %r103, %rs24;
cvt.f32.bf16 %r104, %rs25;
cvt.f32.bf16 %r105, %rs26;
cvt.f32.bf16 %r106, %rs27;
cvt.f32.bf16 %r107, %rs28;
cvt.f32.bf16 %r108, %rs29;
cvt.f32.bf16 %r109, %rs30;
cvt.f32.bf16 %r110, %rs31;
cvt.f32.bf16 %r111, %rs32;
cvt.f32.bf16 %r112, %rs33;
cvt.f32.bf16 %r113, %rs34;
cvt.f32.bf16 %r114, %rs35;
cvt.f32.bf16 %r115, %rs36;
cvt.f32.bf16 %r116, %rs37;
cvt.f32.bf16 %r117, %rs38;
cvt.f32.bf16 %r118, %rs39;
cvt.f32.bf16 %r119, %rs40;
cvt.f32.bf16 %r120, %rs41;
cvt.f32.bf16 %r121, %rs42;
cvt.f32.bf16 %r122, %rs43;
cvt.f32.bf16 %r123, %rs44;
cvt.f32.bf16 %r124, %rs45;
cvt.f32.bf16 %r125, %rs46;
cvt.f32.bf16 %r126, %rs47;
cvt.f32.bf16 %r127, %rs48;
mul.f32 %r18, %r1, %r96;
mul.f32 %r19, %r1, %r97;
mul.f32 %r20, %r1, %r98;
mul.f32 %r21, %r1, %r99;
mul.f32 %r22, %r1, %r100;
mul.f32 %r23, %r1, %r101;
mul.f32 %r24, %r1, %r102;
mul.f32 %r25, %r1, %r103;
mul.f32 %r26, %r1, %r104;
mul.f32 %r27, %r1, %r105;
mul.f32 %r28, %r1, %r106;
mul.f32 %r29, %r1, %r107;
mul.f32 %r30, %r1, %r108;
mul.f32 %r31, %r1, %r109;
mul.f32 %r32, %r1, %r110;
mul.f32 %r33, %r1, %r111;
mul.f32 %r34, %r1, %r112;
mul.f32 %r35, %r1, %r113;
mul.f32 %r36, %r1, %r114;
mul.f32 %r37, %r1, %r115;
mul.f32 %r38, %r1, %r116;
mul.f32 %r39, %r1, %r117;
mul.f32 %r40, %r1, %r118;
mul.f32 %r41, %r1, %r119;
mul.f32 %r42, %r1, %r120;
mul.f32 %r43, %r1, %r121;
mul.f32 %r44, %r1, %r122;
mul.f32 %r45, %r1, %r123;
mul.f32 %r46, %r1, %r124;
mul.f32 %r47, %r1, %r125;
mul.f32 %r48, %r1, %r126;
mul.f32 %r49, %r1, %r127;
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r19, %r18; min.bf16x2 %r50, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r21, %r20; min.bf16x2 %r51, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r23, %r22; min.bf16x2 %r52, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r25, %r24; min.bf16x2 %r53, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r27, %r26; min.bf16x2 %r54, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r29, %r28; min.bf16x2 %r55, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r31, %r30; min.bf16x2 %r56, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r33, %r32; min.bf16x2 %r57, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r35, %r34; min.bf16x2 %r58, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r37, %r36; min.bf16x2 %r59, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r39, %r38; min.bf16x2 %r60, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r41, %r40; min.bf16x2 %r61, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r43, %r42; min.bf16x2 %r62, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r45, %r44; min.bf16x2 %r63, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r47, %r46; min.bf16x2 %r64, rounded, upper; }
{ .reg .b32 rounded, upper; mov.b32 upper, 0x43e043e0; cvt.rn.bf16x2.f32 rounded, %r49, %r48; min.bf16x2 %r65, rounded, upper; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r50; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs1, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r51; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs2, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r52; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs3, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r53; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs4, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r54; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs5, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r55; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs6, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r56; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs7, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r57; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs8, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r58; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs9, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r59; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs10, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r60; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs11, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r61; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs12, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r62; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs13, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r63; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs14, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r64; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs15, f16s; }
{ .reg .b16 lo, hi; .reg .b32 f16s; mov.b32 {lo, hi}, %r65; cvt.rn.f16.bf16 lo, lo; cvt.rn.f16.bf16 hi, hi; mov.b32 f16s, {lo, hi}; cvt.rn.satfinite.e4m3x2.f16x2 %rs16, f16s; }
add.s64 %rd19, %rd2, %rd15;
add.s64 %rd9, %rd19, %rd17;
add.s64 %rd10, %rd9, 50304;
mov.b32 %r66, {%rs1, %rs2};
mov.b32 %r67, {%rs3, %rs4};
mov.b32 %r68, {%rs5, %rs6};
mov.b32 %r69, {%rs7, %rs8};
st.global.v4.b32 [ %rd9 + 0 ], { %r66, %r67, %r68, %r69 };
mov.b32 %r70, {%rs9, %rs10};
mov.b32 %r71, {%rs11, %rs12};
mov.b32 %r72, {%rs13, %rs14};
mov.b32 %r73, {%rs15, %rs16};
st.global.v4.b32 [ %rd10 + 0 ], { %r70, %r71, %r72, %r73 };
and.b32 %r128, %r87, 4080;
mov.b32 %r129, global_smem;
add.s32 %r130, %r129, %r128;
st.shared.v4.b32 [%r130], {%r66, %r67, %r68, %r69};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r131, %r132, %r133, %r134}, [%r130];
bar.warp.sync -1;
st.shared.v4.b32 [%r130], {%r70, %r71, %r72, %r73};
bar.warp.sync -1;
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r135, %r136, %r137, %r138}, [%r130];
and.b32 %r139, %r85, 56;
shl.b32 %r140, %r84, 2;
and.b32 %r141, %r140, 124;
cvt.s64.s32 %rd20, %r92;
add.s64 %rd21, %rd3, %rd20;
mul.lo.s32 %r142, %r141, 768;
or.b32 %r143, %r142, %r139;
cvt.u64.u32 %rd22, %r143;
add.s64 %rd11, %rd21, %rd22;
add.s64 %rd12, %rd11, 768;
add.s64 %rd13, %rd11, 1536;
add.s64 %rd14, %rd11, 2304;
prmt.b32 %r144, %r131, %r135, 0x40U;
prmt.b32 %r145, %r144, %r132, 0x410U;
prmt.b32 %r74, %r145, %r136, 0x4210U;
prmt.b32 %r146, %r133, %r137, 0x40U;
prmt.b32 %r147, %r146, %r134, 0x410U;
prmt.b32 %r75, %r147, %r138, 0x4210U;
st.global.v2.b32 [ %rd11 + 0 ], { %r74, %r75 };
prmt.b32 %r148, %r131, %r135, 0x51U;
prmt.b32 %r149, %r148, %r132, 0x510U;
prmt.b32 %r76, %r149, %r136, 0x5210U;
prmt.b32 %r150, %r133, %r137, 0x51U;
prmt.b32 %r151, %r150, %r134, 0x510U;
prmt.b32 %r77, %r151, %r138, 0x5210U;
st.global.v2.b32 [ %rd12 + 0 ], { %r76, %r77 };
prmt.b32 %r152, %r131, %r135, 0x62U;
prmt.b32 %r153, %r152, %r132, 0x610U;
prmt.b32 %r78, %r153, %r136, 0x6210U;
prmt.b32 %r154, %r133, %r137, 0x62U;
prmt.b32 %r155, %r154, %r134, 0x610U;
prmt.b32 %r79, %r155, %r138, 0x6210U;
st.global.v2.b32 [ %rd13 + 0 ], { %r78, %r79 };
prmt.b32 %r156, %r131, %r135, 0x73U;
prmt.b32 %r157, %r156, %r132, 0x710U;
prmt.b32 %r80, %r157, %r136, 0x7210U;
prmt.b32 %r158, %r133, %r137, 0x73U;
prmt.b32 %r159, %r158, %r134, 0x710U;
prmt.b32 %r81, %r159, %r138, 0x7210U;
st.global.v2.b32 [ %rd14 + 0 ], { %r80, %r81 };
$L__tmp4:
$L__BB0_3:
ret;
$L__tmp5:
$L__func_end0:
}
"#


structure Artifacts where
  referenceIR : String
  referencePTX : String
  hopperIR : String
  hopperPTX : String
  deriving BEq

private def lines (s : String) : List String := (s.splitOn "\n").filter (!·.isEmpty)

def matchesArtifacts (a : Artifacts) : Bool :=
  QKVTTIR.normalizeTTIR a.referenceIR == some (lines referenceGraph) &&
  QKVTTIR.normalizePTX a.referencePTX == lines referencePtx &&
  QKVTTIR.normalizeTTIR a.hopperIR == some (lines hopperGraph) &&
  QKVTTIR.normalizePTX a.hopperPTX == lines hopperPtx

structure Certificate (a : Artifacts) : Type where
  matched : matchesArtifacts a = true
  stores : ∀ (bank : Bank) (gain : Scalar) (p : HeadPoint),
    hopperDispatch bank gain p = some (reference bank gain p)
  rowCoverage : ∀ i : Fin 38633472, ∃ p : HeadPoint, rowAddress p = i.val
  colCoverage : ∀ i : Fin 38633472, ∃ p : HeadPoint, transposedAddress p = i.val
  uniqueWriters : ∀ p q : HeadPoint,
    (rowAddress p = rowAddress q → p = q) ∧
    (transposedAddress p = transposedAddress q → p = q)

def certify (a : Artifacts) : Option (Certificate a) :=
  if h : matchesArtifacts a = true then
    some ⟨h, hopperDispatch_eq, row_cache_coverage, col_cache_coverage, unique_cache_writers⟩
  else none

structure Program where
  artifacts : Artifacts
  checked : Certificate artifacts

def Program.check (a : Artifacts) : Option Program := do
  return ⟨a, ← certify a⟩

def Program.run (program : Program) (hopper : Bool) (bank : Bank) (gain : Scalar)
    (p : HeadPoint) : Option Writes :=
  if hopper && matchesArtifacts program.artifacts then hopperDispatch bank gain p
  else some (reference bank gain p)

theorem Program.run_eq (program : Program) :
    program.run = fun _ bank gain p => some (reference bank gain p) := by
  funext hopper bank gain p
  simp only [Program.run, program.checked.matched, Bool.and_true]
  cases hopper
  · rfl
  · exact program.checked.stores bank gain p

#axiomsClean hopperDispatch_eq
#axiomsClean certify
#axiomsClean Program.run_eq
end VeriTile.Bench.Models.NanoGPT.Track1.LMHead
